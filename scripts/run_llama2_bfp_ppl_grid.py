#!/usr/bin/env python
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PPL_RE = re.compile(r"(?P<dataset>\w+)\s+ppl:\s+(?P<ppl>[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


CASES = (
    {
        "case": "activation_only",
        "w_bits": 16,
        "a_bits": 4,
        "kv_bits": 16,
    },
    {
        "case": "weight_only",
        "w_bits": 4,
        "a_bits": 16,
        "kv_bits": 16,
    },
    {
        "case": "weight_activation",
        "w_bits": 4,
        "a_bits": 4,
        "kv_bits": 16,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run WikiText-2 PPL grid for activation-only, weight-only, and W+A BFP "
            "with and without rotation."
        )
    )
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--eval-nsamples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bfp-group-size", type=int, default=32)
    parser.add_argument("--bfp-exponent-rounding", choices=["floor", "ceil"], default="floor")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--rotation-compute-dtype", choices=["fp64", "fp32"], default="fp64")
    parser.add_argument("--rotation-block-size", type=int, default=0)
    parser.add_argument(
        "--rotation-block-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Rotation block sizes to evaluate for rotated runs. "
            "0 means full Hadamard; 32 means block-wise 32. "
            "Defaults to --rotation-block-size for backward compatibility."
        ),
    )
    parser.add_argument("--rotation-init", choices=["random_hadamard", "hadamard"], default="random_hadamard")
    parser.add_argument("--w-down-had-group-size", type=int, default=32)
    parser.add_argument("--qk-had-group-size", type=int, default=32)
    parser.add_argument(
        "--rotation-mode",
        choices=["random", "checkpoint"],
        default="random",
        help="random uses --random-rotation; checkpoint passes --rotation-path/--experiment-dir to eval_ppl.py.",
    )
    parser.add_argument("--rotation-path", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--output-csv", default="llama2_bfp_ppl_grid_wikitext2.csv")
    parser.add_argument("--log-dir", default="ppl_grid_logs")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def rotation_specs(args):
    specs = [{"rotate": False, "label": "no_rotate", "block_size": None}]
    block_sizes = args.rotation_block_sizes
    if block_sizes is None:
        block_sizes = [args.rotation_block_size]
    for block_size in block_sizes:
        if block_size == 0:
            label = f"{args.rotation_mode}_full"
        else:
            label = f"{args.rotation_mode}_block{block_size}"
        specs.append({"rotate": True, "label": label, "block_size": block_size})
    return specs


def build_eval_command(args, case, spec):
    cmd = [
        args.python,
        str(REPO_ROOT / "eval_ppl.py"),
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--eval-nsamples",
        str(args.eval_nsamples),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--seed",
        str(args.seed),
        "--w-bits",
        str(case["w_bits"]),
        "--a-bits",
        str(case["a_bits"]),
        "--kv-bits",
        str(case["kv_bits"]),
        "--bfp-group-size",
        str(args.bfp_group_size),
        "--bfp-exponent-rounding",
        args.bfp_exponent_rounding,
        "--dtype",
        args.dtype,
        "--rotation-compute-dtype",
        args.rotation_compute_dtype,
        "--rotation-block-size",
        str(spec["block_size"] if spec["block_size"] is not None else args.rotation_block_size),
        "--rotation-init",
        args.rotation_init,
        "--w-down-had-group-size",
        str(args.w_down_had_group_size),
        "--qk-had-group-size",
        str(args.qk_had_group_size),
        "--qk-matmul-bits",
        "16",
        "--av-matmul-bits",
        "16",
    ]
    if args.access_token is not None:
        cmd.extend(["--access-token", args.access_token])

    if not spec["rotate"]:
        cmd.append("--no-rotate")
        return cmd

    if args.rotation_mode == "random":
        cmd.append("--random-rotation")
        return cmd

    if args.rotation_path is not None:
        cmd.extend(["--rotation-path", args.rotation_path])
    if args.experiment_dir is not None:
        cmd.extend(["--experiment-dir", args.experiment_dir])
    if args.rotation_path is None and args.experiment_dir is None:
        raise ValueError("--rotation-mode checkpoint requires --rotation-path or --experiment-dir.")
    return cmd


def run_command(cmd, log_path):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines = []
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)
        return_code = process.wait()

    output = "".join(output_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=output)

    matches = list(PPL_RE.finditer(output))
    if not matches:
        raise RuntimeError(f"Could not parse PPL from output. See log: {log_path}")
    return float(matches[-1].group("ppl"))


def main():
    args = parse_args()
    rows = []
    for case in CASES:
        for spec in rotation_specs(args):
            label = f"{case['case']}_{spec['label']}"
            log_path = Path(args.log_dir) / f"{label}.log"
            cmd = build_eval_command(args, case, spec)
            ppl = run_command(cmd, log_path)
            row = {
                "case": case["case"],
                "rotate": spec["rotate"],
                "rotation_mode": "none" if not spec["rotate"] else args.rotation_mode,
                "rotation_label": spec["label"],
                "rotation_block_size": spec["block_size"],
                "w_bits": case["w_bits"],
                "a_bits": case["a_bits"],
                "kv_bits": case["kv_bits"],
                "dataset": args.dataset,
                "eval_nsamples": args.eval_nsamples,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "bfp_group_size": args.bfp_group_size,
                "ppl": ppl,
                "log_path": str(log_path),
            }
            rows.append(row)
            print(f"RESULT {label}: ppl={ppl:.4f}", flush=True)

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary")
    for row in rows:
        rotate_label = row["rotation_label"]
        print(
            f"{row['case']:<18} {rotate_label:<10} "
            f"W{row['w_bits']} A{row['a_bits']} KV{row['kv_bits']} "
            f"PPL={row['ppl']:.4f}"
        )
    print(f"\nWrote CSV: {output_csv}")


if __name__ == "__main__":
    main()
