import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rotquant import Hook, prepare_model_for_rotate
from rotquant.reconstruction import (
    reconstruct_down_o_weights,
    save_reconstructed_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct down/o projection weights for BFP activations."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model id (for example: meta-llama/Llama-2-7b-hf, facebook/opt-1.3b)",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bfp_bits", type=int, default=8)
    parser.add_argument("--bfp_block_size", type=int, default=128)
    parser.add_argument("--bfp_o_bits", type=int, default=None)
    parser.add_argument("--bfp_down_bits", type=int, default=None)
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
    hook.bfp_o_bits = args.bfp_o_bits
    hook.bfp_down_bits = args.bfp_down_bits
    return hook


def main():
    args = parse_args()
    set_seed(args.seed)
    _disable_init()

    model, tokenizer = _load_model(args.model, args.device)
    hook = _build_hook(args)

    print("Preparing fused pre-rotation model ...")
    prepare_model_for_rotate(model)

    reconstruct_down_o_weights(
        model,
        tokenizer,
        hook,
        args.device,
        nsamples=args.calib_samples,
        seq_len=args.calib_seq_len,
        ridge=args.ridge,
        blend=args.blend,
        row_chunk=args.row_chunk,
    )
    save_reconstructed_weights(
        args.output,
        model,
        hook,
        metadata={
            "model": args.model,
            "bfp_bits": args.bfp_bits,
            "bfp_block_size": args.bfp_block_size,
            "bfp_o_bits": args.bfp_o_bits,
            "bfp_down_bits": args.bfp_down_bits,
            "calib_samples": args.calib_samples,
            "calib_seq_len": args.calib_seq_len,
            "ridge": args.ridge,
            "blend": args.blend,
        },
    )
    print(f"Saved reconstructed weights to {args.output}")


if __name__ == "__main__":
    main()
