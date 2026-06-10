#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from utils import hadamard_utils as repo_hadamard_utils  # noqa: E402
except ImportError:
    repo_hadamard_utils = None


BFP_DEFAULT_BLOCK_SIZE = 32
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
LAYER_MODULE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
WEIGHT_COLUMN_ORDER = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
LLAMA_R1_RIGHT_ROLES = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "lm_head"}
LLAMA_R1_LEFT_ROLES = {"o_proj", "down_proj"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare LLaMA weight BFP quantization error before and after "
            "random Hadamard rotation."
        )
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "none", "cpu"],
        help="Use 'auto' for HF device_map='auto', 'none' for --device, or 'cpu'.",
    )
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--w-groupsize", type=int, default=32)
    parser.add_argument("--w-clip-ratio", type=float, default=1.0)
    parser.add_argument("--w-scale-method", choices=["absmax", "topk"], default="absmax")
    parser.add_argument("--w-topk", type=int, default=1)
    parser.add_argument(
        "--rotation-mode",
        choices=["llama-r1", "last-dim", "first-dim"],
        default="llama-r1",
        help=(
            "llama-r1: q/k/v/gate/up rotate input axis, o/down rotate output axis. "
            "last-dim: rotate every weight's input axis. first-dim: rotate every output axis."
        ),
    )
    parser.add_argument(
        "--hadamard-block-size",
        type=int,
        default=0,
        help=(
            "0: full random Hadamard over the selected dim, -1: use BFP block size, "
            "N: block-diagonal random Hadamard with block size N."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-by", choices=["layer", "matrix", "module"], default="matrix")
    parser.add_argument(
        "--module-suffixes",
        nargs="*",
        default=None,
        help="Optional suffix filter, e.g. q_proj k_proj v_proj o_proj gate_proj up_proj down_proj.",
    )
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--total-only", action="store_true")
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def torch_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    return torch.float16


def _is_pow2(value):
    return value > 0 and value & (value - 1) == 0


def _compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def resolve_bfp_block_size(groupsize):
    return BFP_DEFAULT_BLOCK_SIZE if groupsize == -1 else groupsize


def _bfp_topk_shared_scale(x, *, maxq, finfo, clip_ratio=1.0, topk=1):
    if topk <= 0:
        raise ValueError(f"BFP topk must be positive, got {topk}.")

    abs_x = torch.abs(x)
    clipped_abs = abs_x * clip_ratio
    nonzero = clipped_abs > 0
    nonzero_count = nonzero.sum(dim=-1, keepdim=True)
    safe_abs = torch.where(nonzero, clipped_abs, torch.ones_like(clipped_abs))
    scale_exp = torch.ceil(torch.log2(safe_abs / maxq))
    neg_inf = torch.full_like(scale_exp, -float("inf"))
    scale_exp = torch.where(nonzero, scale_exp, neg_inf)
    sorted_exp = torch.sort(scale_exp, dim=-1, descending=True).values
    effective_k = torch.minimum(
        nonzero_count,
        torch.full_like(nonzero_count, min(topk, x.shape[-1])),
    ).clamp(min=1)
    shared_exp = sorted_exp.gather(dim=-1, index=effective_k.long() - 1)
    shared_exp = torch.where(nonzero_count == 0, torch.zeros_like(shared_exp), shared_exp)
    scale = 2 ** shared_exp
    return torch.clamp(scale, min=finfo.tiny, max=finfo.max)


def _bfp_scale_and_shift(x, *, maxq, finfo, clip_ratio, scale_method, topk):
    if scale_method == "absmax":
        xmax = torch.amax(torch.abs(x), dim=-1, keepdim=True) * clip_ratio
        safe_xmax = torch.where(xmax == 0, torch.ones_like(xmax), xmax)
        shared_shift = torch.ceil(torch.log2(safe_xmax / maxq))
        scale = 2 ** shared_shift
        scale = torch.clamp(scale, min=finfo.tiny, max=finfo.max)
        scale = torch.where(xmax == 0, torch.ones_like(scale), scale)
        shared_shift = torch.where(xmax == 0, torch.zeros_like(shared_shift), shared_shift)
        return scale, shared_shift
    if scale_method == "topk":
        scale = _bfp_topk_shared_scale(
            x,
            maxq=maxq,
            finfo=finfo,
            clip_ratio=clip_ratio,
            topk=topk,
        )
        return scale, torch.log2(scale)
    raise ValueError(f"Unknown BFP scale method: {scale_method}.")


def bfp_fake_quant(
    x,
    bits=4,
    block_size=BFP_DEFAULT_BLOCK_SIZE,
    clip_ratio=1.0,
    scale_method="absmax",
    topk=1,
):
    if bits < 2:
        raise ValueError(f"BFP bits must be at least 2, got {bits}.")
    if block_size <= 0:
        raise ValueError(f"BFP block size must be positive, got {block_size}.")

    minq = -(2 ** (bits - 1))
    maxq = 2 ** (bits - 1) - 1
    orig_shape = x.shape
    orig_dtype = x.dtype
    orig_finfo = torch.finfo(orig_dtype)

    compute_dtype = _compute_dtype(x.dtype)
    x = x.to(dtype=compute_dtype)
    finfo = torch.finfo(compute_dtype)
    x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=-finfo.max)

    pad = (block_size - x.shape[-1] % block_size) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    x = x.reshape(-1, x.shape[-1] // block_size, block_size)
    scale, _ = _bfp_scale_and_shift(
        x,
        maxq=maxq,
        finfo=finfo,
        clip_ratio=clip_ratio,
        scale_method=scale_method,
        topk=topk,
    )

    q = torch.clamp(torch.round(x / scale), minq, maxq)
    q = torch.nan_to_num(q, nan=0.0, posinf=maxq, neginf=minq)
    xhat = (q * scale).reshape(*orig_shape[:-1], -1)
    xhat = torch.nan_to_num(xhat, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    xhat = torch.clamp(xhat, min=orig_finfo.min, max=orig_finfo.max)

    if pad:
        xhat = xhat[..., : orig_shape[-1]]
    return xhat.reshape(orig_shape).to(dtype=orig_dtype)


def bfp_shift_stats(
    x,
    bits=4,
    block_size=BFP_DEFAULT_BLOCK_SIZE,
    clip_ratio=1.0,
    scale_method="absmax",
    topk=1,
):
    if bits < 2:
        raise ValueError(f"BFP bits must be at least 2, got {bits}.")
    if block_size <= 0:
        raise ValueError(f"BFP block size must be positive, got {block_size}.")

    maxq = 2 ** (bits - 1) - 1
    compute_dtype = _compute_dtype(x.dtype)
    x = x.to(dtype=compute_dtype)
    finfo = torch.finfo(compute_dtype)
    x = torch.nan_to_num(x, nan=0.0, posinf=finfo.max, neginf=-finfo.max)

    pad = (block_size - x.shape[-1] % block_size) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    x = x.reshape(-1, x.shape[-1] // block_size, block_size)
    _, shared_shift = _bfp_scale_and_shift(
        x,
        maxq=maxq,
        finfo=finfo,
        clip_ratio=clip_ratio,
        scale_method=scale_method,
        topk=topk,
    )
    shared_shift = shared_shift.float()
    return {
        "shift_blocks": shared_shift.numel(),
        "shift_sum": shared_shift.sum().item(),
        "shift_abs_sum": shared_shift.abs().sum().item(),
        "shift_min": shared_shift.min().item(),
        "shift_max": shared_shift.max().item(),
    }


def _power2_hadamard_transform(x):
    n = x.shape[-1]
    if not _is_pow2(n):
        raise ValueError(f"Power-of-two Hadamard requires pow2 dim, got {n}.")

    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        h *= 2
    return y.reshape_as(x) / math.sqrt(n)


def _hadamard_transform(x):
    if repo_hadamard_utils is not None:
        return repo_hadamard_utils.matmul_hadU(x)
    return _power2_hadamard_transform(x)


def resolve_rotation_block_size(rotation_block_size, dim, bfp_block_size=32):
    if rotation_block_size == -1:
        block_size = bfp_block_size
    elif rotation_block_size == 0:
        block_size = None
    else:
        block_size = rotation_block_size

    if block_size is None:
        if repo_hadamard_utils is None and not _is_pow2(dim):
            raise ValueError(
                f"Full Hadamard for dimension {dim} requires repo utils/hadamard_utils.py. "
                "Use --hadamard-block-size 32 on a minimal checkout."
            )
        if repo_hadamard_utils is not None:
            try:
                repo_hadamard_utils.get_hadK(dim)
            except AssertionError as exc:
                raise ValueError(f"Full Hadamard is not supported for dimension {dim}.") from exc
        return None

    if not _is_pow2(block_size):
        raise ValueError(f"Hadamard block size must be a power of 2, got {block_size}.")
    if dim % block_size != 0:
        raise ValueError(f"Dimension {dim} must be divisible by block size {block_size}.")
    return block_size


def apply_random_hadamard_to_last_dim(x, block_size=None, seed=0):
    dim = x.shape[-1]
    compute_dtype = _compute_dtype(x.dtype)
    x_dtype = x.dtype
    x = x.to(dtype=compute_dtype)
    sign_block_size = dim if block_size is None else block_size

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), dtype=torch.int64)
    signs = signs.to(device=x.device, dtype=compute_dtype).mul_(2).sub_(1)
    signs = signs.reshape(dim // sign_block_size, sign_block_size)

    x_blocks = x.reshape(*x.shape[:-1], dim // sign_block_size, sign_block_size)
    output = _hadamard_transform(x_blocks * signs)
    return output.reshape_as(x).to(dtype=x_dtype)


def module_matches_suffix(name, suffixes):
    if not suffixes:
        return True
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes)


def layer_key(name):
    match = LAYER_RE.search(name)
    if match is None:
        return name
    return f"layer_{int(match.group(1)):02d}"


def matrix_key(name):
    match = LAYER_MODULE_RE.search(name)
    if match is None:
        return (layer_key(name), name.split(".")[-1])
    return (f"layer_{int(match.group(1)):02d}", match.group(2).split(".")[-1])


def stat_key(name, group_by):
    if group_by == "module":
        return name
    if group_by == "matrix":
        return matrix_key(name)
    return layer_key(name)


def role_from_name(name):
    return name.split(".")[-1]


def rotation_axis_for_module(name, rotation_mode):
    if rotation_mode == "last-dim":
        return "last"
    if rotation_mode == "first-dim":
        return "first"

    role = role_from_name(name)
    if role in LLAMA_R1_RIGHT_ROLES:
        return "last"
    if role in LLAMA_R1_LEFT_ROLES:
        return "first"
    return "last"


def rotate_weight(weight, *, axis, rotation_block_size_arg, bfp_block_size, seed):
    if axis == "last":
        dim = weight.shape[-1]
        block_size = resolve_rotation_block_size(
            rotation_block_size_arg,
            dim,
            bfp_block_size=bfp_block_size,
        )
        return apply_random_hadamard_to_last_dim(weight, block_size=block_size, seed=seed)
    if axis == "first":
        dim = weight.shape[0]
        block_size = resolve_rotation_block_size(
            rotation_block_size_arg,
            dim,
            bfp_block_size=bfp_block_size,
        )
        return apply_random_hadamard_to_last_dim(weight.t(), block_size=block_size, seed=seed).t()
    raise ValueError(f"Unsupported rotation axis: {axis}")


def snr_db(signal_sum_sq, sse):
    if sse == 0.0:
        return float("inf")
    if signal_sum_sq == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_sum_sq / sse)


def row_for_weight(name, module, args):
    weight = module.weight.detach()
    bfp_block_size = weight.shape[-1] if args.w_groupsize == 0 else resolve_bfp_block_size(args.w_groupsize)
    axis = rotation_axis_for_module(name, args.rotation_mode)

    weight_float = weight.float()
    signal_sum_sq = weight_float.pow(2).sum().item()
    q = bfp_fake_quant(
        weight,
        bits=args.w_bits,
        block_size=bfp_block_size,
        clip_ratio=args.w_clip_ratio,
        scale_method=args.w_scale_method,
        topk=args.w_topk,
    )
    no_rot_sse = (weight_float - q.float()).pow(2).sum().item()
    no_rot_shift = bfp_shift_stats(
        weight,
        bits=args.w_bits,
        block_size=bfp_block_size,
        clip_ratio=args.w_clip_ratio,
        scale_method=args.w_scale_method,
        topk=args.w_topk,
    )
    del q

    rotated = rotate_weight(
        weight,
        axis=axis,
        rotation_block_size_arg=args.hadamard_block_size,
        bfp_block_size=bfp_block_size,
        seed=args.seed,
    )
    rotated_float = rotated.float()
    q_rot = bfp_fake_quant(
        rotated,
        bits=args.w_bits,
        block_size=bfp_block_size,
        clip_ratio=args.w_clip_ratio,
        scale_method=args.w_scale_method,
        topk=args.w_topk,
    )
    rot_sse = (rotated_float - q_rot.float()).pow(2).sum().item()
    rot_shift = bfp_shift_stats(
        rotated,
        bits=args.w_bits,
        block_size=bfp_block_size,
        clip_ratio=args.w_clip_ratio,
        scale_method=args.w_scale_method,
        topk=args.w_topk,
    )
    del rotated, q_rot

    key = stat_key(name, args.group_by)
    if isinstance(key, tuple):
        layer, weight_name = key
        row_name = ".".join(key)
    else:
        layer, weight_name = key, None
        row_name = key
    numel = module.weight.numel()
    no_rot_mse = no_rot_sse / max(numel, 1)
    rot_mse = rot_sse / max(numel, 1)
    ratio = float("inf") if no_rot_mse == 0.0 else rot_mse / no_rot_mse
    return {
        "name": row_name,
        "layer": layer,
        "weight": weight_name,
        "module_name": name,
        "rotation_axis": axis,
        "modules": 1,
        "numel": numel,
        "signal_sum_sq": signal_sum_sq,
        "no_rot_sse": no_rot_sse,
        "random_rot_sse": rot_sse,
        "no_rot_mse": no_rot_mse,
        "random_rot_mse": rot_mse,
        "ratio_random_over_no_rot": ratio,
        "reduction_pct": (1.0 - ratio) * 100.0 if math.isfinite(ratio) else float("-inf"),
        "no_rot_snr_db": snr_db(signal_sum_sq, no_rot_sse),
        "random_rot_snr_db": snr_db(signal_sum_sq, rot_sse),
        "no_rot_absmax": weight_float.abs().max().item(),
        "random_rot_absmax": rotated_float.abs().max().item(),
        "shift_blocks": no_rot_shift["shift_blocks"],
        "no_rot_shift_sum": no_rot_shift["shift_sum"],
        "random_rot_shift_sum": rot_shift["shift_sum"],
        "no_rot_abs_shift_sum": no_rot_shift["shift_abs_sum"],
        "random_rot_abs_shift_sum": rot_shift["shift_abs_sum"],
        "no_rot_shift_mean": no_rot_shift["shift_sum"] / max(no_rot_shift["shift_blocks"], 1),
        "random_rot_shift_mean": rot_shift["shift_sum"] / max(rot_shift["shift_blocks"], 1),
        "shift_mean_delta": (
            rot_shift["shift_sum"] / max(rot_shift["shift_blocks"], 1)
            - no_rot_shift["shift_sum"] / max(no_rot_shift["shift_blocks"], 1)
        ),
        "no_rot_abs_shift_mean": no_rot_shift["shift_abs_sum"] / max(no_rot_shift["shift_blocks"], 1),
        "random_rot_abs_shift_mean": rot_shift["shift_abs_sum"] / max(rot_shift["shift_blocks"], 1),
        "abs_shift_mean_delta": (
            rot_shift["shift_abs_sum"] / max(rot_shift["shift_blocks"], 1)
            - no_rot_shift["shift_abs_sum"] / max(no_rot_shift["shift_blocks"], 1)
        ),
        "no_rot_shift_min": no_rot_shift["shift_min"],
        "no_rot_shift_max": no_rot_shift["shift_max"],
        "random_rot_shift_min": rot_shift["shift_min"],
        "random_rot_shift_max": rot_shift["shift_max"],
    }


def target_linears(model, *, include_lm_head=False, module_suffixes=None):
    lm_head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        if module is lm_head and not include_lm_head:
            continue
        if isinstance(module, torch.nn.Linear) and module_matches_suffix(name, module_suffixes):
            yield name, module


def aggregate_rows(rows, group_by):
    if group_by == "module":
        return rows

    grouped = {}
    for row in rows:
        key = row["name"]
        if key not in grouped:
            grouped[key] = {
                "name": row["name"],
                "layer": row["layer"],
                "weight": row["weight"],
                "module_name": "",
                "rotation_axis": "mixed",
                "modules": 0,
                "numel": 0,
                "signal_sum_sq": 0.0,
                "no_rot_sse": 0.0,
                "random_rot_sse": 0.0,
                "no_rot_absmax": 0.0,
                "random_rot_absmax": 0.0,
                "shift_blocks": 0,
                "no_rot_shift_sum": 0.0,
                "random_rot_shift_sum": 0.0,
                "no_rot_abs_shift_sum": 0.0,
                "random_rot_abs_shift_sum": 0.0,
                "no_rot_shift_min": float("inf"),
                "no_rot_shift_max": -float("inf"),
                "random_rot_shift_min": float("inf"),
                "random_rot_shift_max": -float("inf"),
            }
        acc = grouped[key]
        acc["modules"] += row["modules"]
        acc["numel"] += row["numel"]
        acc["signal_sum_sq"] += row["signal_sum_sq"]
        acc["no_rot_sse"] += row["no_rot_sse"]
        acc["random_rot_sse"] += row["random_rot_sse"]
        acc["no_rot_absmax"] = max(acc["no_rot_absmax"], row["no_rot_absmax"])
        acc["random_rot_absmax"] = max(acc["random_rot_absmax"], row["random_rot_absmax"])
        acc["shift_blocks"] += row["shift_blocks"]
        acc["no_rot_shift_sum"] += row["no_rot_shift_sum"]
        acc["random_rot_shift_sum"] += row["random_rot_shift_sum"]
        acc["no_rot_abs_shift_sum"] += row["no_rot_abs_shift_sum"]
        acc["random_rot_abs_shift_sum"] += row["random_rot_abs_shift_sum"]
        acc["no_rot_shift_min"] = min(acc["no_rot_shift_min"], row["no_rot_shift_min"])
        acc["no_rot_shift_max"] = max(acc["no_rot_shift_max"], row["no_rot_shift_max"])
        acc["random_rot_shift_min"] = min(acc["random_rot_shift_min"], row["random_rot_shift_min"])
        acc["random_rot_shift_max"] = max(acc["random_rot_shift_max"], row["random_rot_shift_max"])
    return [finalize_acc(acc) for acc in grouped.values()]


def finalize_acc(acc):
    numel = max(acc["numel"], 1)
    no_rot_mse = acc["no_rot_sse"] / numel
    rot_mse = acc["random_rot_sse"] / numel
    ratio = float("inf") if no_rot_mse == 0.0 else rot_mse / no_rot_mse
    acc = dict(acc)
    acc["no_rot_mse"] = no_rot_mse
    acc["random_rot_mse"] = rot_mse
    acc["ratio_random_over_no_rot"] = ratio
    acc["reduction_pct"] = (1.0 - ratio) * 100.0 if math.isfinite(ratio) else float("-inf")
    acc["no_rot_snr_db"] = snr_db(acc["signal_sum_sq"], acc["no_rot_sse"])
    acc["random_rot_snr_db"] = snr_db(acc["signal_sum_sq"], acc["random_rot_sse"])
    shift_blocks = max(acc.get("shift_blocks", 0), 1)
    acc["no_rot_shift_mean"] = acc["no_rot_shift_sum"] / shift_blocks
    acc["random_rot_shift_mean"] = acc["random_rot_shift_sum"] / shift_blocks
    acc["shift_mean_delta"] = acc["random_rot_shift_mean"] - acc["no_rot_shift_mean"]
    acc["no_rot_abs_shift_mean"] = acc["no_rot_abs_shift_sum"] / shift_blocks
    acc["random_rot_abs_shift_mean"] = acc["random_rot_abs_shift_sum"] / shift_blocks
    acc["abs_shift_mean_delta"] = acc["random_rot_abs_shift_mean"] - acc["no_rot_abs_shift_mean"]
    return acc


def total_row(rows):
    total = {
        "name": "total",
        "layer": "total",
        "weight": None,
        "module_name": "",
        "rotation_axis": "mixed",
        "modules": sum(row["modules"] for row in rows),
        "numel": sum(row["numel"] for row in rows),
        "signal_sum_sq": sum(row["signal_sum_sq"] for row in rows),
        "no_rot_sse": sum(row["no_rot_sse"] for row in rows),
        "random_rot_sse": sum(row["random_rot_sse"] for row in rows),
        "no_rot_absmax": max((row["no_rot_absmax"] for row in rows), default=0.0),
        "random_rot_absmax": max((row["random_rot_absmax"] for row in rows), default=0.0),
        "shift_blocks": sum(row["shift_blocks"] for row in rows),
        "no_rot_shift_sum": sum(row["no_rot_shift_sum"] for row in rows),
        "random_rot_shift_sum": sum(row["random_rot_shift_sum"] for row in rows),
        "no_rot_abs_shift_sum": sum(row["no_rot_abs_shift_sum"] for row in rows),
        "random_rot_abs_shift_sum": sum(row["random_rot_abs_shift_sum"] for row in rows),
        "no_rot_shift_min": min((row["no_rot_shift_min"] for row in rows), default=0.0),
        "no_rot_shift_max": max((row["no_rot_shift_max"] for row in rows), default=0.0),
        "random_rot_shift_min": min((row["random_rot_shift_min"] for row in rows), default=0.0),
        "random_rot_shift_max": max((row["random_rot_shift_max"] for row in rows), default=0.0),
    }
    return finalize_acc(total)


def sorted_rows(rows, group_by):
    if group_by == "matrix":
        weight_order = {name: idx for idx, name in enumerate(WEIGHT_COLUMN_ORDER)}

        def key(row):
            layer_num = int(row["layer"].split("_")[-1]) if row["layer"].startswith("layer_") else 10**9
            weight_num = weight_order.get(row["weight"], len(weight_order))
            return (layer_num, weight_num, row["weight"] or "")

        return sorted(rows, key=key)
    if group_by == "layer":
        return sorted(rows, key=lambda row: row["layer"])
    return sorted(rows, key=lambda row: row["module_name"])


def format_table(rows, total, group_by):
    name_label = "matrix" if group_by == "matrix" else group_by
    lines = [
        (
            f"{name_label:<36} {'mods':>5} {'numel':>14} "
            f"{'no_rot_mse':>14} {'rand_rot_mse':>14} {'ratio':>10} "
            f"{'reduction':>10} {'no_shift':>9} {'rot_shift':>9} {'d_shift':>9} "
            f"{'no_rot_snr':>11} {'rand_snr':>11}"
        )
    ]
    for row in rows + [total]:
        label = row["name"] if group_by != "module" else row["module_name"]
        lines.append(
            f"{label:<36} "
            f"{row['modules']:5d} "
            f"{row['numel']:14d} "
            f"{row['no_rot_mse']:14.6e} "
            f"{row['random_rot_mse']:14.6e} "
            f"{row['ratio_random_over_no_rot']:10.4f} "
            f"{row['reduction_pct']:9.2f}% "
            f"{row['no_rot_shift_mean']:9.3f} "
            f"{row['random_rot_shift_mean']:9.3f} "
            f"{row['shift_mean_delta']:9.3f} "
            f"{row['no_rot_snr_db']:10.2f}dB "
            f"{row['random_rot_snr_db']:10.2f}dB"
        )
    return "\n".join(lines)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    if args.device == "cuda" and args.device_map != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    torch.manual_seed(args.seed)
    dtype = torch_dtype(args.dtype)
    model_kwargs = {
        "token": args.access_token,
        "local_files_only": args.local_files_only,
        "torch_dtype": dtype,
    }
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    elif args.device_map == "cpu":
        model_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.device_map == "none":
        model.to(torch.device(args.device))
    model.eval()

    module_rows = []
    with torch.no_grad():
        for name, module in target_linears(
            model,
            include_lm_head=args.include_lm_head,
            module_suffixes=args.module_suffixes,
        ):
            module_rows.append(row_for_weight(name, module, args))

    if not module_rows:
        raise ValueError("No Linear modules matched the requested filters.")

    rows = sorted_rows(aggregate_rows(module_rows, args.group_by), args.group_by)
    total = total_row(module_rows)
    display_rows = [] if args.total_only else rows
    all_rows = [total] if args.total_only else rows + [total]

    print(
        "LLaMA weight BFP quantization error comparison\n"
        f"model={args.model_name}\n"
        f"w_bits={args.w_bits}, w_groupsize={args.w_groupsize}, "
        f"clip_ratio={args.w_clip_ratio}, scale_method={args.w_scale_method}, topk={args.w_topk}\n"
        f"rotation_mode={args.rotation_mode}, hadamard_block_size={args.hadamard_block_size}, "
        f"seed={args.seed}\n"
        "ratio = random_rot_mse / no_rot_mse, reduction = 1 - ratio\n"
        "shift = BFP block shared exponent in scale = 2^shift\n"
    )
    print(format_table(display_rows, total, args.group_by))

    if args.csv_output is not None:
        write_csv(args.csv_output, all_rows)
        print(f"\nWrote CSV: {args.csv_output}")
    if args.json_output is not None:
        write_json(args.json_output, all_rows)
        print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
