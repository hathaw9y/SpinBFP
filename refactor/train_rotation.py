from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import datasets
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refactor.fuse_rmsnorm import fuse_llama_rmsnorm
from refactor.model_load import TorchrunContext, load_llama_for_torchrun
from refactor.trainable_rotation import (
    TrainableRotationConfig,
    rotation_filename,
    rotation_parameters,
    save_rotation_state,
    setup_trainable_llama_rotations,
)
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLaMA rotation matrices.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-tied-lm-head", action="store_true")
    parser.add_argument("--no-center-embeddings", action="store_true")
    parser.add_argument("--keep-rmsnorm-modules", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--bfp-group-size", type=int, default=32)
    parser.add_argument("--qk-matmul-bits", type=int, default=None)
    parser.add_argument("--av-matmul-bits", type=int, default=None)
    parser.add_argument("--qk-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--av-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--rotation-compute-dtype", choices=["fp32", "fp64"], default="fp32")
    parser.add_argument(
        "--rotation-init",
        choices=["random_hadamard", "hadamard"],
        default="random_hadamard",
    )
    parser.add_argument("--rotation-seed", type=int, default=0)
    parser.add_argument("--online-had-group-size", type=int, default=32)
    parser.add_argument("--w-down-had-group-size", type=int, default=32)
    parser.add_argument("--qk-had-group-size", type=int, default=32)
    parser.add_argument("--no-online-down-proj-had", action="store_true")
    parser.add_argument("--no-online-o-proj-had", action="store_true")
    parser.add_argument("--no-qk-online-had", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(args: argparse.Namespace, tokenizer, ctx: TorchrunContext) -> DataLoader:
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
    )
    dataset = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(args.max_length, 2048),
    )
    return DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
        collate_fn=default_data_collator,
    )


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
        qk_matmul_bits=args.qk_matmul_bits if args.qk_matmul_bits is not None else args.kv_bits,
        av_matmul_bits=args.av_matmul_bits if args.av_matmul_bits is not None else args.kv_bits,
        qk_matmul_bfp_group_size=args.qk_matmul_bfp_group_size,
        av_matmul_bfp_group_size=args.av_matmul_bfp_group_size,
        compute_dtype=rotation_compute_dtype(args),
    )


def prepare_model(args: argparse.Namespace):
    model, tokenizer, ctx = load_llama_for_torchrun(
        args.model,
        dtype=args.dtype,
        token=args.access_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        clone_tied_lm_head=not args.keep_tied_lm_head,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    tokenizer.model_max_length = args.max_length
    tokenizer.padding_side = "right"
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    fuse_stats = fuse_llama_rmsnorm(
        model,
        center_embeddings=not args.no_center_embeddings,
        replace_norms=not args.keep_rmsnorm_modules,
    )
    train_cfg = trainable_rotation_config(args)
    rotation_stats = setup_trainable_llama_rotations(model, train_cfg)
    model.train()
    return model, tokenizer, ctx, fuse_stats, rotation_stats, train_cfg


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model, tokenizer, ctx, fuse_stats, rotation_stats, train_cfg = prepare_model(args)
    base_model = model
    params = rotation_parameters(base_model)
    optimizer = SGDG(params, lr=args.learning_rate, stiefel=True)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_steps,
    )

    if ctx.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    dataloader = make_dataloader(args, tokenizer, ctx)
    step = 0
    running_loss = 0.0

    if ctx.is_main_process:
        print(
            "Starting rotation training "
            f"model={args.model} "
            f"world_size={ctx.world_size} "
            f"fused_linears={fuse_stats.fused_linears} "
            f"wrapped_linears={rotation_stats.wrapped_linears} "
            f"rotation_tensors={rotation_stats.rotation_tensors} "
            f"qk_wrappers={rotation_stats.qk_wrappers} "
            f"attention_matmul_wrappers={rotation_stats.attention_matmul_wrappers} "
            f"block_size={rotation_stats.block_size or 'full'}"
        )

    while step < args.max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(ctx.device, non_blocking=True)
            labels = batch["labels"].to(ctx.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += float(loss.detach().cpu())

            if ctx.is_main_process and step % args.logging_steps == 0:
                avg_loss = running_loss / args.logging_steps
                running_loss = 0.0
                print(f"step={step} loss={avg_loss:.6f}")

            if step >= args.max_steps:
                break

    if ctx.distributed:
        dist.barrier()

    if ctx.is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = save_rotation_state(unwrap_model(model), output_dir / rotation_filename(train_cfg))
        print(f"Saved rotations to {path}")

    ctx.cleanup()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
