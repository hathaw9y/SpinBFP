import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rotquant import Hook, apply_rotate
from utils import eval_ppl_wikitext


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a HF causal LM with optional rotation and BFP."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model id (for example: meta-llama/Llama-2-7b-hf, facebook/opt-1.3b)",
    )
    parser.add_argument("--ppl_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Apply Hadamard weight/activation rotation before evaluation.",
    )
    parser.add_argument(
        "--online_rotate",
        action="store_true",
        help="Enable online MLP/FFN intermediate rotation and post-RoPE Q/K Hadamard rotation.",
    )
    parser.add_argument(
        "--bfp",
        action="store_true",
        help="Apply BFP to linear inputs and attention QK matmul inputs.",
    )
    parser.add_argument("--bfp_bits", type=int, default=8)
    parser.add_argument("--bfp_block_size", type=int, default=128)
    parser.add_argument("--bfp_qkv_bits", type=int, default=None)
    parser.add_argument("--bfp_o_bits", type=int, default=None)
    parser.add_argument("--bfp_up_gate_bits", type=int, default=None)
    parser.add_argument("--bfp_down_bits", type=int, default=None)
    parser.add_argument("--bfp_qk_bits", type=int, default=None)
    parser.add_argument(
        "--weight_bfp",
        action="store_true",
        help="Apply BFP to linear weights in the W.T layout used by linear inputs.",
    )
    parser.add_argument("--weight_bfp_bits", type=int, default=8)
    parser.add_argument("--weight_bfp_block_size", type=int, default=128)
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


def _model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _build_hook(args, model_dir: str) -> Hook:
    hook = Hook()
    hook.bfp = args.bfp
    hook.bfp_bits = args.bfp_bits
    hook.bfp_block_size = args.bfp_block_size
    hook.bfp_qkv_bits = args.bfp_qkv_bits
    hook.bfp_o_bits = args.bfp_o_bits
    hook.bfp_up_gate_bits = args.bfp_up_gate_bits
    hook.bfp_down_bits = args.bfp_down_bits
    hook.bfp_qk_bits = args.bfp_qk_bits
    hook.weight_bfp = args.weight_bfp
    hook.weight_bfp_bits = args.weight_bfp_bits
    hook.weight_bfp_block_size = args.weight_bfp_block_size
    hook.online_rotate = args.online_rotate
    hook.orth_group_size = args.bfp_block_size
    hook.model_dir = model_dir
    return hook


def main():
    args = parse_args()
    set_seed(args.seed)
    _disable_init()

    model, tokenizer = _load_model(args.model, args.device)
    model_dir = _model_dir_name(args.model)
    hook = _build_hook(args, model_dir)

    rotate = "hadamard" if args.rotate else None
    print(f"Apply rotate={rotate or 'none'}")
    apply_rotate(model, args.device, hook, rotate=rotate)

    print("\n--- PPL evaluation on WikiText-2 ---")
    ppl = eval_ppl_wikitext(model, tokenizer, seq_len=args.ppl_seq_len, device=args.device)
    print(f"\n[{args.model}] PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()
