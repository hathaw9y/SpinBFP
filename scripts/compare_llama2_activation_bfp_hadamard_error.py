#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def _is_pow2(value):
    return value > 0 and value & (value - 1) == 0


def _compute_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def resolve_bfp_block_size(groupsize):
    return BFP_DEFAULT_BLOCK_SIZE if groupsize == -1 else groupsize


def resolve_bfp_block_size_for_tensor(groupsize, x):
    return x.shape[-1] if groupsize == 0 else resolve_bfp_block_size(groupsize)


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
    if scale_method == "absmax":
        xmax = torch.amax(torch.abs(x), dim=-1, keepdim=True) * clip_ratio
        safe_xmax = torch.where(xmax == 0, torch.ones_like(xmax), xmax)
        scale = 2 ** torch.ceil(torch.log2(safe_xmax / maxq))
        scale = torch.clamp(scale, min=finfo.tiny, max=finfo.max)
        scale = torch.where(xmax == 0, torch.ones_like(scale), scale)
    elif scale_method == "topk":
        scale = _bfp_topk_shared_scale(
            x,
            maxq=maxq,
            finfo=finfo,
            clip_ratio=clip_ratio,
            topk=topk,
        )
    else:
        raise ValueError(f"Unknown BFP scale method: {scale_method}.")

    q = torch.clamp(torch.round(x / scale), minq, maxq)
    q = torch.nan_to_num(q, nan=0.0, posinf=maxq, neginf=minq)
    xhat = (q * scale).reshape(*orig_shape[:-1], -1)
    xhat = torch.nan_to_num(xhat, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
    xhat = torch.clamp(xhat, min=orig_finfo.min, max=orig_finfo.max)

    if pad:
        xhat = xhat[..., : orig_shape[-1]]
    return xhat.reshape(orig_shape).to(dtype=orig_dtype)


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
    x = x.contiguous()
    if repo_hadamard_utils is not None:
        return repo_hadamard_utils.matmul_hadU(x)
    return _power2_hadamard_transform(x)


def resolve_rotation_block_size(rotation_block_size, hidden_size, bfp_block_size=32):
    if rotation_block_size == -1:
        block_size = bfp_block_size
    elif rotation_block_size == 0:
        block_size = None
    else:
        block_size = rotation_block_size

    if block_size is None:
        if repo_hadamard_utils is None and not _is_pow2(hidden_size):
            raise ValueError(
                f"Full Hadamard for dimension {hidden_size} requires repo utils/hadamard_utils.py. "
                "Use --hadamard-block-size 32 on a minimal checkout."
            )
        if repo_hadamard_utils is not None:
            try:
                repo_hadamard_utils.get_hadK(hidden_size)
            except AssertionError as exc:
                raise ValueError(f"Full Hadamard is not supported for dimension {hidden_size}.") from exc
        return None

    if not _is_pow2(block_size):
        raise ValueError(f"Hadamard block size must be a power of 2, got {block_size}.")
    if hidden_size % block_size != 0:
        raise ValueError(f"Dimension {hidden_size} must be divisible by block size {block_size}.")
    return block_size


def apply_rotation_to_last_dim(x, block_size=None, seed=0):
    dim = x.shape[-1]
    if block_size is not None:
        if not _is_pow2(block_size):
            raise ValueError(f"Hadamard block size must be a power of 2, got {block_size}.")
        if dim % block_size != 0:
            raise ValueError(f"Dimension {dim} must be divisible by block size {block_size}.")

    compute_dtype = _compute_dtype(x.dtype)
    x_dtype = x.dtype
    x = x.to(dtype=compute_dtype).contiguous()
    sign_block_size = dim if block_size is None else block_size

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), dtype=torch.int64)
    signs = signs.to(device=x.device, dtype=compute_dtype).mul_(2).sub_(1)
    signs = signs.reshape(dim // sign_block_size, sign_block_size)

    x_blocks = x.reshape(*x.shape[:-1], dim // sign_block_size, sign_block_size)
    output = _hadamard_transform(x_blocks * signs)
    return output.reshape_as(x).to(dtype=x_dtype)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare LLaMA activation BFP quantization error with and without "
            "a random Hadamard transform on WikiText-2 samples."
        )
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "none"],
        help="Use 'auto' for HF device_map='auto', or 'none' to move the model to --device.",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="First non-overlapping WikiText-2 chunk index. sample-index=0 starts at the first token.",
    )
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--a-groupsize", type=int, default=-1)
    parser.add_argument("--a-clip-ratio", type=float, default=1.0)
    parser.add_argument("--a-scale-method", choices=["absmax", "topk"], default="absmax")
    parser.add_argument("--a-topk", type=int, default=1)
    parser.add_argument(
        "--hadamard-block-size",
        type=int,
        default=0,
        help=(
            "0: full random Hadamard over the last dim, -1: use BFP block size, "
            "N: block-diagonal random Hadamard with block size N."
        ),
    )
    parser.add_argument(
        "--group-by",
        choices=["layer", "matrix", "module"],
        default="matrix",
    )
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


def new_entry(key):
    if isinstance(key, tuple):
        layer, weight = key
        name = ".".join(key)
    else:
        layer, weight = key, None
        name = key
    return {
        "name": name,
        "layer": layer,
        "weight": weight,
        "modules": set(),
        "calls": 0,
        "numel": 0,
        "signal_sum_sq": 0.0,
        "no_had_sse": 0.0,
        "random_had_sse": 0.0,
        "no_had_absmax": 0.0,
        "random_had_absmax": 0.0,
    }


def snr_db(signal_sum_sq, sse):
    if sse == 0.0:
        return float("inf")
    if signal_sum_sq == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_sum_sq / sse)


def finalize_entry(entry):
    numel = max(entry["numel"], 1)
    no_had_mse = entry["no_had_sse"] / numel
    random_had_mse = entry["random_had_sse"] / numel
    ratio = float("inf") if no_had_mse == 0.0 else random_had_mse / no_had_mse
    return {
        "name": entry["name"],
        "layer": entry["layer"],
        "weight": entry["weight"],
        "modules": len(entry["modules"]) if isinstance(entry["modules"], set) else entry["modules"],
        "calls": entry["calls"],
        "numel": entry["numel"],
        "no_had_mse": no_had_mse,
        "random_had_mse": random_had_mse,
        "ratio_random_over_no_had": ratio,
        "reduction_pct": (1.0 - ratio) * 100.0 if math.isfinite(ratio) else float("-inf"),
        "no_had_snr_db": snr_db(entry["signal_sum_sq"], entry["no_had_sse"]),
        "random_had_snr_db": snr_db(entry["signal_sum_sq"], entry["random_had_sse"]),
        "no_had_absmax": entry["no_had_absmax"],
        "random_had_absmax": entry["random_had_absmax"],
        "signal_sum_sq": entry["signal_sum_sq"],
        "no_had_sse": entry["no_had_sse"],
        "random_had_sse": entry["random_had_sse"],
    }


class HadamardBFPActivationErrorCollector:
    def __init__(
        self,
        *,
        bits,
        groupsize,
        clip_ratio,
        scale_method,
        topk,
        hadamard_block_size_arg,
        seed,
        group_by,
    ):
        self.bits = bits
        self.groupsize = groupsize
        self.clip_ratio = clip_ratio
        self.scale_method = scale_method
        self.topk = topk
        self.hadamard_block_size_arg = hadamard_block_size_arg
        self.seed = seed
        self.group_by = group_by
        self.stats = OrderedDict()

    def _entry(self, name):
        key = stat_key(name, self.group_by)
        if key not in self.stats:
            self.stats[key] = new_entry(key)
        return self.stats[key]

    def hook(self, name):
        def pre_hook(module, inputs):
            if len(inputs) == 0 or not torch.is_floating_point(inputs[0]):
                return

            x = inputs[0].detach()
            block_size = resolve_bfp_block_size_for_tensor(self.groupsize, x)
            hadamard_block_size = resolve_rotation_block_size(
                self.hadamard_block_size_arg,
                x.shape[-1],
                bfp_block_size=block_size,
            )

            entry = self._entry(name)
            entry["modules"].add(name)
            entry["calls"] += 1
            entry["numel"] += x.numel()

            x_float = x.float()
            entry["signal_sum_sq"] += x_float.pow(2).sum().item()
            entry["no_had_absmax"] = max(entry["no_had_absmax"], x_float.abs().max().item())

            q = bfp_fake_quant(
                x,
                bits=self.bits,
                block_size=block_size,
                clip_ratio=self.clip_ratio,
                scale_method=self.scale_method,
                topk=self.topk,
            )
            entry["no_had_sse"] += (x_float - q.float()).pow(2).sum().item()
            del q

            x_had = apply_rotation_to_last_dim(
                x,
                block_size=hadamard_block_size,
                seed=self.seed,
            )
            x_had_float = x_had.float()
            entry["random_had_absmax"] = max(
                entry["random_had_absmax"],
                x_had_float.abs().max().item(),
            )
            q_had = bfp_fake_quant(
                x_had,
                bits=self.bits,
                block_size=block_size,
                clip_ratio=self.clip_ratio,
                scale_method=self.scale_method,
                topk=self.topk,
            )
            entry["random_had_sse"] += (x_had_float - q_had.float()).pow(2).sum().item()
            del x_had, q_had

        return pre_hook

    def rows(self):
        return [finalize_entry(entry) for entry in self.stats.values()]


def target_linears(model, *, include_lm_head=False, module_suffixes=None):
    lm_head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        if module is lm_head and not include_lm_head:
            continue
        if isinstance(module, torch.nn.Linear) and module_matches_suffix(name, module_suffixes):
            yield name, module


def add_hooks(model, collector, *, include_lm_head=False, module_suffixes=None):
    handles = []
    for name, module in target_linears(
        model,
        include_lm_head=include_lm_head,
        module_suffixes=module_suffixes,
    ):
        try:
            handle = module.register_forward_pre_hook(collector.hook(name), prepend=True)
        except TypeError:
            handle = module.register_forward_pre_hook(collector.hook(name))
        handles.append(handle)
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def load_wikitext2_samples(tokenizer, *, split, seqlen, sample_index, nsamples, device):
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, got {nsamples}.")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    start = sample_index * seqlen
    end = start + nsamples * seqlen
    if input_ids.shape[1] < end:
        raise ValueError(
            f"WikiText-2 {split} has only {input_ids.shape[1]} tokens, "
            f"cannot read sample_index={sample_index}, nsamples={nsamples}, seqlen={seqlen}."
        )
    return input_ids[:, start:end].reshape(nsamples, seqlen).to(device)


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
    return sorted(rows, key=lambda row: row["name"])


def total_row(rows):
    total = {
        "name": "total",
        "layer": "total",
        "weight": None,
        "modules": sum(row["modules"] for row in rows),
        "calls": sum(row["calls"] for row in rows),
        "numel": sum(row["numel"] for row in rows),
        "signal_sum_sq": sum(row["signal_sum_sq"] for row in rows),
        "no_had_sse": sum(row["no_had_sse"] for row in rows),
        "random_had_sse": sum(row["random_had_sse"] for row in rows),
        "no_had_absmax": max((row["no_had_absmax"] for row in rows), default=0.0),
        "random_had_absmax": max((row["random_had_absmax"] for row in rows), default=0.0),
    }
    return finalize_entry(total)


def format_table(rows, total, group_by):
    name_label = "matrix" if group_by == "matrix" else group_by
    lines = [
        (
            f"{name_label:<36} {'calls':>7} {'numel':>14} "
            f"{'no_had_mse':>14} {'rand_had_mse':>14} {'ratio':>10} "
            f"{'reduction':>10} {'no_had_snr':>11} {'rand_snr':>11}"
        )
    ]
    for row in rows + [total]:
        label = row["name"]
        lines.append(
            f"{label:<36} "
            f"{row['calls']:7d} "
            f"{row['numel']:14d} "
            f"{row['no_had_mse']:14.6e} "
            f"{row['random_had_mse']:14.6e} "
            f"{row['ratio_random_over_no_had']:10.4f} "
            f"{row['reduction_pct']:9.2f}% "
            f"{row['no_had_snr_db']:10.2f}dB "
            f"{row['random_had_snr_db']:10.2f}dB"
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")

    torch.manual_seed(args.seed)
    dtype = torch_dtype(args.dtype)
    token_kwargs = {
        "token": args.access_token,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **token_kwargs)

    model_kwargs = {
        "token": args.access_token,
        "local_files_only": args.local_files_only,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }
    device = torch.device(args.device)
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.device_map == "none":
        model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    input_device = next(model.parameters()).device
    input_ids = load_wikitext2_samples(
        tokenizer,
        split=args.split,
        seqlen=args.seqlen,
        sample_index=args.sample_index,
        nsamples=args.nsamples,
        device=input_device,
    )

    collector = HadamardBFPActivationErrorCollector(
        bits=args.a_bits,
        groupsize=args.a_groupsize,
        clip_ratio=args.a_clip_ratio,
        scale_method=args.a_scale_method,
        topk=args.a_topk,
        hadamard_block_size_arg=args.hadamard_block_size,
        seed=args.seed,
        group_by=args.group_by,
    )
    handles = add_hooks(
        model,
        collector,
        include_lm_head=args.include_lm_head,
        module_suffixes=args.module_suffixes,
    )
    if not handles:
        raise ValueError("No Linear modules matched the requested filters.")

    try:
        with torch.no_grad():
            for start in range(0, input_ids.shape[0], args.batch_size):
                model(input_ids=input_ids[start : start + args.batch_size])
    finally:
        remove_hooks(handles)

    rows = sorted_rows(collector.rows(), args.group_by)
    total = total_row(rows)
    display_rows = [] if args.total_only else rows
    all_rows = [total] if args.total_only else rows + [total]
    print(
        "LLaMA activation BFP quantization error comparison\n"
        f"model={args.model_name}, dataset=wikitext2/{args.split}, "
        f"sample_index={args.sample_index}, nsamples={args.nsamples}, "
        f"seqlen={args.seqlen}, batch_size={args.batch_size}\n"
        f"a_bits={args.a_bits}, a_groupsize={args.a_groupsize}, "
        f"clip_ratio={args.a_clip_ratio}, scale_method={args.a_scale_method}, topk={args.a_topk}, "
        f"hadamard_block_size={args.hadamard_block_size}, seed={args.seed}\n"
        "ratio = random_had_mse / no_had_mse, reduction = 1 - ratio\n"
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
