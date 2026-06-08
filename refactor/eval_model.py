from __future__ import annotations

import argparse
import sys
from pathlib import Path

import datasets
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refactor.attention_bfp import AttentionMatmulBFPConfig, add_attention_matmul_bfp_to_llama
from refactor.fuse_rmsnorm import FuseRMSNormStats, fuse_llama_rmsnorm
from refactor.learned_rotation import load_learned_rotation_state
from refactor.model_load import TorchrunContext, load_llama_for_torchrun
from refactor.rotation import RotationStats, apply_llama_random_hadamard_rotations
from refactor.trainable_rotation import (
    TrainableRotationConfig,
    TrainableRotationStats,
    rotation_filename,
    setup_trainable_llama_rotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fused LLaMA model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--rotation-path", default=None)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-center-embeddings", action="store_true")
    parser.add_argument("--keep-rmsnorm-modules", action="store_true")
    parser.add_argument("--keep-tied-lm-head", action="store_true")
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--random-rotation", action="store_true")
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--rotation-init", choices=["random_hadamard", "hadamard"], default="random_hadamard")
    parser.add_argument("--rotation-compute-dtype", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument("--online-had-group-size", type=int, default=32)
    parser.add_argument("--w-down-had-group-size", type=int, default=32)
    parser.add_argument("--qk-had-group-size", type=int, default=32)
    parser.add_argument("--no-online-down-proj-had", action="store_true")
    parser.add_argument("--no-online-o-proj-had", action="store_true")
    parser.add_argument("--no-qk-online-had", action="store_true")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--bfp-group-size", type=int, default=32)
    parser.add_argument("--qk-matmul-bits", type=int, default=None)
    parser.add_argument("--av-matmul-bits", type=int, default=None)
    parser.add_argument("--qk-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--av-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def qk_matmul_bits(args: argparse.Namespace) -> int:
    return args.qk_matmul_bits if args.qk_matmul_bits is not None else args.kv_bits


def av_matmul_bits(args: argparse.Namespace) -> int:
    return args.av_matmul_bits if args.av_matmul_bits is not None else args.kv_bits


def rotation_compute_dtype(args: argparse.Namespace) -> torch.dtype:
    return torch.float64 if args.rotation_compute_dtype == "fp64" else torch.float32


def trainable_rotation_config(args: argparse.Namespace) -> TrainableRotationConfig:
    return TrainableRotationConfig(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        kv_bits=args.kv_bits,
        bfp_group_size=args.bfp_group_size,
        rotation_block_size=args.rotation_block_size,
        rotation_init=args.rotation_init,
        rotation_seed=args.rotation_seed,
        online_down_proj=not args.no_online_down_proj_had,
        online_o_proj=not args.no_online_o_proj_had,
        online_qk=not args.no_qk_online_had,
        online_had_group_size=args.online_had_group_size,
        w_down_had_group_size=args.w_down_had_group_size,
        qk_had_group_size=args.qk_had_group_size,
        qk_matmul_bits=qk_matmul_bits(args),
        av_matmul_bits=av_matmul_bits(args),
        qk_matmul_bfp_group_size=args.qk_matmul_bfp_group_size,
        av_matmul_bfp_group_size=args.av_matmul_bfp_group_size,
        compute_dtype=rotation_compute_dtype(args),
    )


def attention_implementation(args: argparse.Namespace) -> str | None:
    if args.attn_implementation is not None:
        return args.attn_implementation
    if qk_matmul_bits(args) < 16 or av_matmul_bits(args) < 16:
        return "eager"
    return None


def find_rotation_path(args: argparse.Namespace, cfg: TrainableRotationConfig) -> Path | None:
    if args.no_rotate:
        if args.rotation_path is not None or args.experiment_dir is not None:
            print("--no-rotate is set; ignoring rotation checkpoint arguments.")
        return None
    if args.random_rotation:
        if args.rotation_path is not None or args.experiment_dir is not None:
            print("--random-rotation is set; ignoring rotation checkpoint arguments.")
        return None
    if args.rotation_path is not None:
        return Path(args.rotation_path)
    if args.experiment_dir is None:
        return None

    experiment_dir = Path(args.experiment_dir)
    expected = experiment_dir / rotation_filename(cfg)
    if expected.exists():
        return expected
    legacy = experiment_dir / "rotation.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"rotation checkpoint not found: {expected}")


def load_fused_llama_for_eval(
    args: argparse.Namespace,
) -> tuple[
    torch.nn.Module,
    object,
    TorchrunContext,
    FuseRMSNormStats,
    RotationStats | TrainableRotationStats | None,
    Path | None,
    int,
]:
    model, tokenizer, ctx = load_llama_for_torchrun(
        args.model,
        dtype=args.dtype,
        token=args.access_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=attention_implementation(args),
        clone_tied_lm_head=not args.keep_tied_lm_head,
    )
    stats = fuse_llama_rmsnorm(
        model,
        center_embeddings=not args.no_center_embeddings,
        replace_norms=not args.keep_rmsnorm_modules,
    )
    rotation_stats = None
    train_cfg = trainable_rotation_config(args)
    rotation_path = find_rotation_path(args, train_cfg)
    loaded_rotation_tensors = 0
    if not args.no_rotate:
        if rotation_path is None:
            rotation_stats = apply_llama_random_hadamard_rotations(
                model,
                rotation_block_size=args.rotation_block_size,
                seed=args.rotation_seed,
                online_down_proj=not args.no_online_down_proj_had,
                online_o_proj=not args.no_online_o_proj_had,
                online_qk=not args.no_qk_online_had,
            )
        else:
            rotation_stats = setup_trainable_llama_rotations(
                model,
                train_cfg,
            )
            loaded_rotation_tensors = load_learned_rotation_state(model, rotation_path)
            for param in model.parameters():
                param.requires_grad = False
    existing_attention_matmul_wrappers = (
        rotation_stats.attention_matmul_wrappers
        if isinstance(rotation_stats, TrainableRotationStats)
        else 0
    )
    attention_matmul_wrappers = add_attention_matmul_bfp_to_llama(
        model,
        AttentionMatmulBFPConfig(
            qk_bits=qk_matmul_bits(args),
            av_bits=av_matmul_bits(args),
            qk_group_size=args.qk_matmul_bfp_group_size,
            av_group_size=args.av_matmul_bfp_group_size,
        ),
    )
    model.spinbfp_attention_matmul_wrappers = (
        existing_attention_matmul_wrappers + attention_matmul_wrappers
    )
    return model, tokenizer, ctx, stats, rotation_stats, rotation_path, loaded_rotation_tensors


def eval_tokens(
    tokenizer,
) -> torch.Tensor:
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
    text = "\n\n".join(dataset["text"])
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    seqlen: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    previous_use_cache = model.config.use_cache
    model.config.use_cache = False

    nsamples = input_ids.numel() // seqlen
    if nsamples == 0:
        raise ValueError(
            f"Not enough eval tokens ({input_ids.numel()}) for max_length={seqlen}."
        )
    input_ids = input_ids[:, : nsamples * seqlen].reshape(nsamples, seqlen).to(device)

    total_loss = 0.0
    total_tokens = 0
    for start in tqdm(range(0, nsamples, batch_size), desc="Eval batches"):
        batch = input_ids[start : start + batch_size]
        logits = model(input_ids=batch).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.detach().cpu())
        total_tokens += shift_labels.numel()

    model.config.use_cache = previous_use_cache
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main() -> None:
    args = parse_args()
    (
        model,
        tokenizer,
        ctx,
        fuse_stats,
        rotation_stats,
        rotation_path,
        loaded_rotation_tensors,
    ) = load_fused_llama_for_eval(args)

    if ctx.is_main_process:
        print(
            "Prepared fused LLaMA for evaluation "
            f"model={args.model} "
            f"device={ctx.device} "
            f"world_size={ctx.world_size} "
            f"vocab_size={len(tokenizer)} "
            f"fused_linears={fuse_stats.fused_linears} "
            f"replaced_norms={fuse_stats.replaced_norms}"
        )
        if isinstance(rotation_stats, RotationStats):
            print(
                "Applied random Hadamard rotations "
                f"block_size={rotation_stats.block_size or 'full'} "
                f"seed={rotation_stats.seed} "
                f"offline_input={rotation_stats.offline_input_linears} "
                f"offline_output={rotation_stats.offline_output_linears} "
                f"online_linears={rotation_stats.online_linears} "
                f"qk_wrappers={rotation_stats.qk_wrappers}"
            )
        elif isinstance(rotation_stats, TrainableRotationStats):
            print(
                "Loaded learned rotations "
                f"path={rotation_path} "
                f"loaded_tensors={loaded_rotation_tensors} "
                f"block_size={rotation_stats.block_size or 'full'} "
                f"wrapped_linears={rotation_stats.wrapped_linears} "
                f"rotation_tensors={rotation_stats.rotation_tensors} "
                f"qk_wrappers={rotation_stats.qk_wrappers} "
                f"attention_matmul_wrappers={rotation_stats.attention_matmul_wrappers}"
            )
        else:
            print("Evaluating without rotations")
        print(
            "Applied attention matmul BFP "
            f"wrapped_layers={getattr(model, 'spinbfp_attention_matmul_wrappers', 0)}"
        )
        tokens = eval_tokens(tokenizer)
        ppl = evaluate_ppl(
            model,
            tokens,
            seqlen=args.max_length,
            batch_size=args.batch_size,
            device=ctx.device,
        )
        print(f"wikitext2 ppl: {ppl:.4f}")

    ctx.cleanup()


if __name__ == "__main__":
    main()
