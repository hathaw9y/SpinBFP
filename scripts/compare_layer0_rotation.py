#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.rotation_utils import apply_rotation_left, apply_rotation_right, rotation_total_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Compare layer-0 rotated LLaMA weights.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--rotation-path", default=None)
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--rotation-init", choices=["random_hadamard", "hadamard"], default="random_hadamard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def resolve_dtype(dtype):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def hadamard_matrix(size):
    if size & (size - 1) != 0:
        raise ValueError(f"Hadamard size must be power of two, got {size}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < size:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / (size**0.5)


def make_rotation(size, block_size, init):
    if block_size > 0:
        if size % block_size != 0:
            raise ValueError(f"rotation size {size} must be divisible by block size {block_size}")
        return torch.stack(
            [make_rotation(block_size, 0, init) for _ in range(size // block_size)],
            dim=0,
        )

    h = hadamard_matrix(size)
    if init == "hadamard":
        return h
    signs = torch.randint(0, 2, (size,), dtype=torch.float64) * 2 - 1
    return signs[:, None] * h


def load_rotations(args, hidden_size, head_dim):
    if args.rotation_path is None:
        torch.manual_seed(args.seed)
        return (
            make_rotation(hidden_size, args.rotation_block_size, args.rotation_init),
            make_rotation(head_dim, args.rotation_block_size, args.rotation_init),
        )

    state = torch.load(args.rotation_path, map_location="cpu")
    return state["R1"].double(), state["model.layers.0.self_attn.R2"].double()


def dense_rotation(rotation):
    if rotation.dim() == 2:
        return rotation
    return torch.block_diag(*list(rotation))


def apply_r2_helper(weight, r2, transpose=False):
    had_dim = rotation_total_dim(r2)
    if transpose:
        shape = weight.shape
        temp = weight.reshape(-1, shape[-1] // had_dim, had_dim)
        return apply_rotation_right(temp, r2, torch.float64).reshape(shape)

    wt = weight.t()
    shape = wt.shape
    temp = wt.reshape(-1, shape[-1] // had_dim, had_dim)
    return apply_rotation_right(temp, r2, torch.float64).reshape(shape).t()


def apply_r2_dense(weight, r2, transpose=False):
    r2_dense = dense_rotation(r2)
    had_dim = r2_dense.shape[0]
    if transpose:
        shape = weight.shape
        temp = weight.reshape(-1, shape[-1] // had_dim, had_dim)
        return (temp @ r2_dense).reshape(shape)

    wt = weight.t()
    shape = wt.shape
    temp = wt.reshape(-1, shape[-1] // had_dim, had_dim)
    return (temp @ r2_dense).reshape(shape).t()


def compare(name, helper, dense, save_dir=None):
    diff = (helper - dense).abs()
    print(
        f"{name:10s} shape={tuple(helper.shape)} "
        f"max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e} "
        f"rot_norm={helper.float().norm().item():.6e}"
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(helper.cpu(), save_dir / f"layer0_{name}.pt")


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        token=args.access_token,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if args.fuse:
        from utils.fuse_norm_utils import fuse_layer_norms

        fuse_layer_norms(model)

    layer = model.model.layers[0]
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // model.config.num_attention_heads
    r1, r2 = load_rotations(args, hidden_size, head_dim)
    r1_dense = dense_rotation(r1)
    save_dir = Path(args.save_dir) if args.save_dir else None

    print(f"R1 shape={tuple(r1.shape)} dense={tuple(r1_dense.shape)}")
    print(f"R2 shape={tuple(r2.shape)} dense={tuple(dense_rotation(r2).shape)}")
    print(f"fused={args.fuse} init={args.rotation_init} block={args.rotation_block_size}")

    modules = {
        "q_proj": layer.self_attn.q_proj.weight.detach().double(),
        "k_proj": layer.self_attn.k_proj.weight.detach().double(),
        "v_proj": layer.self_attn.v_proj.weight.detach().double(),
        "o_proj": layer.self_attn.o_proj.weight.detach().double(),
        "gate_proj": layer.mlp.gate_proj.weight.detach().double(),
        "up_proj": layer.mlp.up_proj.weight.detach().double(),
        "down_proj": layer.mlp.down_proj.weight.detach().double(),
    }

    for name in ["q_proj", "k_proj", "gate_proj", "up_proj"]:
        weight = modules[name]
        compare(
            name,
            apply_rotation_right(weight, r1, torch.float64),
            weight @ r1_dense,
            save_dir,
        )

    v_base_helper = apply_rotation_right(modules["v_proj"], r1, torch.float64)
    v_base_dense = modules["v_proj"] @ r1_dense
    compare(
        "v_proj",
        apply_r2_helper(v_base_helper, r2, transpose=False),
        apply_r2_dense(v_base_dense, r2, transpose=False),
        save_dir,
    )

    o_base_helper = apply_rotation_left(modules["o_proj"], r1, torch.float64, transpose=True)
    o_base_dense = r1_dense.t() @ modules["o_proj"]
    compare(
        "o_proj",
        apply_r2_helper(o_base_helper, r2, transpose=True),
        apply_r2_dense(o_base_dense, r2, transpose=True),
        save_dir,
    )

    compare(
        "down_proj",
        apply_rotation_left(modules["down_proj"], r1, torch.float64, transpose=True),
        r1_dense.t() @ modules["down_proj"],
        save_dir,
    )


if __name__ == "__main__":
    main()
