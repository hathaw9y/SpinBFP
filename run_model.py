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
    parser.add_argument(
        "--bfp_exponent_stats",
        action="store_true",
        help="Print average shared exponent for each BFP application location.",
    )
    parser.add_argument("--disable_bfp_lm_head", action="store_true")
    parser.add_argument("--disable_bfp_down_proj", action="store_true")
    parser.add_argument("--disable_bfp_gate_proj", action="store_true")
    parser.add_argument("--disable_bfp_up_proj", action="store_true")
    parser.add_argument("--disable_bfp_k_proj", action="store_true")
    parser.add_argument("--disable_bfp_o_proj", action="store_true")
    parser.add_argument("--disable_bfp_q_proj", action="store_true")
    parser.add_argument("--disable_bfp_qk_matmul_key", action="store_true")
    parser.add_argument("--disable_bfp_qk_matmul_query", action="store_true")
    parser.add_argument("--disable_bfp_v_proj", action="store_true")
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


def _disabled_bfp_positions(args) -> set[str]:
    disabled = set()
    flag_to_position = {
        "disable_bfp_lm_head": ("lm_head.input",),
        "disable_bfp_down_proj": ("mlp.down_proj.input", "fc2.input"),
        "disable_bfp_gate_proj": ("mlp.gate_proj.input",),
        "disable_bfp_up_proj": ("mlp.up_proj.input", "fc1.input"),
        "disable_bfp_k_proj": ("self_attn.k_proj.input",),
        "disable_bfp_o_proj": ("self_attn.o_proj.input", "self_attn.out_proj.input"),
        "disable_bfp_q_proj": ("self_attn.q_proj.input",),
        "disable_bfp_qk_matmul_key": ("self_attn.qk_matmul.key",),
        "disable_bfp_qk_matmul_query": ("self_attn.qk_matmul.query",),
        "disable_bfp_v_proj": ("self_attn.v_proj.input",),
    }
    for flag, positions in flag_to_position.items():
        if getattr(args, flag):
            disabled.update(positions)
    return disabled


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
    hook.bfp_shared_exponent_stats = args.bfp_exponent_stats
    hook.disabled_bfp_positions = _disabled_bfp_positions(args)
    hook.online_rotate = args.online_rotate
    hook.orth_group_size = args.bfp_block_size
    hook.model_dir = model_dir
    return hook


def _print_bfp_stat_table(title, name_label, rows, rate_rows, name_width) -> None:
    if not rows:
        return

    rate_by_name = {item["name"]: item for item in rate_rows}
    print(f"\n--- {title} ---")
    print(
        f"{name_label:{name_width}s} {'mean':>10s} {'variance':>10s} "
        f"{'min':>6s} {'max':>6s} {'zero_m':>10s} "
        f"{'sh>=m-1':>10s} {'sh>=m':>10s}"
    )
    for item in rows:
        rate = rate_by_name.get(item["name"], {})
        print(
            f"{item['name']:{name_width}s} "
            f"{item['mean']:10.4f} "
            f"{item['variance']:10.4f} "
            f"{item['min']:6.0f} "
            f"{item['max']:6.0f} "
            f"{rate.get('zero_mantissa_rate', float('nan')):10.4f} "
            f"{rate.get('shift_ge_mbits_minus1_rate', float('nan')):10.4f} "
            f"{rate.get('shift_ge_mbits_rate', float('nan')):10.4f}"
        )


def _print_bfp_shared_exponent_stats(hook) -> None:
    averages = hook.bfp_shared_exponent_averages()
    if not averages:
        print("\n--- BFP shared exponent stats ---")
        print("No BFP shared exponent stats were collected.")
        return

    _print_bfp_stat_table(
        "BFP shared exponent stats",
        "location",
        averages,
        hook.bfp_rate_averages(),
        72,
    )
    _print_bfp_stat_table(
        "BFP shared exponent stats by position",
        "position",
        hook.bfp_shared_exponent_position_averages(),
        hook.bfp_rate_position_averages(),
        32,
    )
    _print_bfp_stat_table(
        "BFP shared exponent stats by layer",
        "layer",
        hook.bfp_shared_exponent_layer_averages(),
        hook.bfp_rate_layer_averages(),
        12,
    )
    _print_bfp_shift_stats(hook)


def _print_bfp_shift_stats(hook) -> None:
    _print_bfp_stat_table(
        "BFP bit shift stats",
        "location",
        hook.bfp_shift_averages(),
        hook.bfp_rate_averages(),
        72,
    )
    _print_bfp_stat_table(
        "BFP bit shift stats by position",
        "position",
        hook.bfp_shift_position_averages(),
        hook.bfp_rate_position_averages(),
        32,
    )
    _print_bfp_stat_table(
        "BFP bit shift stats by layer",
        "layer",
        hook.bfp_shift_layer_averages(),
        hook.bfp_rate_layer_averages(),
        12,
    )


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
    if args.bfp_exponent_stats:
        _print_bfp_shared_exponent_stats(hook)


if __name__ == "__main__":
    main()
