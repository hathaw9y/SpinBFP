from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@dataclass(frozen=True)
class TorchrunContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def cleanup(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def torchrun_context() -> TorchrunContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return TorchrunContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        distributed=distributed,
    )


def resolve_dtype(dtype: str, config) -> Optional[torch.dtype]:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype != "auto":
        raise ValueError(f"Unsupported dtype: {dtype}")

    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = config_dtype.replace("torch.", "")
        if config_dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if config_dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if config_dtype in {"float32", "fp32", "float"}:
            return torch.float32
    if config_dtype in {torch.float16, torch.bfloat16, torch.float32}:
        return config_dtype
    return None


def load_llama_for_torchrun(
    model_name: str,
    *,
    dtype: str = "auto",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    clone_tied_lm_head: bool = True,
) -> tuple[torch.nn.Module, AutoTokenizer, TorchrunContext]:
    ctx = torchrun_context()
    config = AutoConfig.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if config.model_type != "llama":
        raise ValueError(f"refactor/model_load.py supports only LLaMA models, got {config.model_type}.")
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation

    should_clone_lm_head = bool(
        clone_tied_lm_head and getattr(config, "tie_word_embeddings", False)
    )
    if should_clone_lm_head:
        config.tie_word_embeddings = False

    torch_dtype = resolve_dtype(dtype, config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        token=token,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if should_clone_lm_head:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model.to(ctx.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, ctx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a LLaMA causal LM under torchrun.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-tied-lm-head", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, ctx = load_llama_for_torchrun(
        args.model,
        dtype=args.dtype,
        token=args.access_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        clone_tied_lm_head=not args.keep_tied_lm_head,
    )

    if ctx.is_main_process:
        print(
            "Loaded LLaMA model "
            f"model={args.model} "
            f"device={ctx.device} "
            f"world_size={ctx.world_size} "
            f"vocab_size={len(tokenizer)}"
        )
    ctx.cleanup()


if __name__ == "__main__":
    main()
