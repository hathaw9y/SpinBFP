import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rotquant import Hook, apply_rotate, prepare_model_for_rotate
from rotquant.reconstruction import (
    RECONSTRUCTION_ORDER,
    reconstruct_weight_groups,
    save_reconstructed_weight_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct Linear weights for BFP activations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model id (for example: meta-llama/Llama-2-7b-hf, facebook/opt-1.3b)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Reconstruct weights after applying Hadamard rotation without BFP.",
    )
    parser.add_argument(
        "--online_rotate",
        action="store_true",
        help="Use online intermediate/QK rotation when --rotate is enabled.",
    )
    parser.add_argument("--bfp_bits", type=int, default=8)
    parser.add_argument("--bfp_block_size", type=int, default=128)
    parser.add_argument("--bfp_qkv_bits", type=int, default=None)
    parser.add_argument("--bfp_o_bits", type=int, default=None)
    parser.add_argument("--bfp_up_gate_bits", type=int, default=None)
    parser.add_argument("--bfp_down_bits", type=int, default=None)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(RECONSTRUCTION_ORDER),
        choices=list(RECONSTRUCTION_ORDER),
        help="Module groups to reconstruct, in the order provided.",
    )
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_seq_len", type=int, default=2048)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--row_chunk", type=int, default=2048)
    return parser.parse_args()


def _disable_init():
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def _load_model(model_id: str, device: str):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if hasattr(model.config, "do_layer_norm_before") and not model.config.do_layer_norm_before:
        raise ValueError("post-LN OPT (for example opt-350m) is not supported.")
    return model, tokenizer


def _build_hook(args) -> Hook:
    hook = Hook()
    hook.bfp = True
    hook.bfp_bits = args.bfp_bits
    hook.bfp_block_size = args.bfp_block_size
    hook.bfp_qkv_bits = args.bfp_qkv_bits
    hook.bfp_o_bits = args.bfp_o_bits
    hook.bfp_up_gate_bits = args.bfp_up_gate_bits
    hook.bfp_down_bits = args.bfp_down_bits
    hook.online_rotate = args.online_rotate
    hook.orth_group_size = args.bfp_block_size
    return hook


def _build_rotate_only_hook(args) -> Hook:
    hook = Hook()
    hook.bfp = False
    hook.weight_bfp = False
    hook.online_rotate = args.online_rotate
    hook.orth_group_size = args.bfp_block_size
    return hook


def _model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _output_dir(args) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    stage = "rotate" if args.rotate else "raw"
    return Path("recon") / _model_dir_name(args.model) / stage


def _bits_label(args) -> str:
    return f"bfp{args.bfp_bits}"


def main():
    args = parse_args()
    set_seed(args.seed)
    _disable_init()

    model, tokenizer = _load_model(args.model, args.device)
    hook = _build_hook(args)

    if args.rotate:
        print("Preparing fused rotated model without BFP ...")
        rotate_hook = _build_rotate_only_hook(args)
        apply_rotate(
            model,
            args.device,
            rotate_hook,
            rotate="hadamard",
        )
    else:
        print("Preparing fused pre-rotation model ...")
        prepare_model_for_rotate(model)

    reconstructed = reconstruct_weight_groups(
        model,
        tokenizer,
        hook,
        args.device,
        groups=args.groups,
        nsamples=args.calib_samples,
        seq_len=args.calib_seq_len,
        ridge=args.ridge,
        blend=args.blend,
        row_chunk=args.row_chunk,
    )

    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, weights in reconstructed.items():
        path = output_dir / f"recon_{group}_{_bits_label(args)}.pt"
        save_reconstructed_weight_state(
            str(path),
            weights,
            metadata={
                "group": group,
                "groups": args.groups,
                "stage": "rotate" if args.rotate else "raw",
                "model": args.model,
                "bfp_bits": args.bfp_bits,
                "bfp_block_size": args.bfp_block_size,
                "rotate": args.rotate,
                "online_rotate": args.online_rotate,
                "bfp_qkv_bits": args.bfp_qkv_bits,
                "bfp_o_bits": args.bfp_o_bits,
                "bfp_up_gate_bits": args.bfp_up_gate_bits,
                "bfp_down_bits": args.bfp_down_bits,
                "calib_samples": args.calib_samples,
                "calib_seq_len": args.calib_seq_len,
                "ridge": args.ridge,
                "blend": args.blend,
            },
        )
        print(f"Saved {group} reconstructed weights to {path}")

    if not reconstructed:
        print("No reconstructed weights were saved.")


if __name__ == "__main__":
    main()
