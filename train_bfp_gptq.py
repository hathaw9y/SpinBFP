import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate BFP-GPTQ weights under an existing BFP/rotation runtime."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--rotation-path", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-dataset", choices=["wikitext2"], default="wikitext2")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--w-bits", type=int, default=16)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--bfp-group-size", type=int, default=32)
    parser.add_argument("--qk-matmul-bits", type=int, default=None)
    parser.add_argument("--av-matmul-bits", type=int, default=None)
    parser.add_argument("--qk-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--av-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--online-had-group-size", type=int, default=32)
    parser.add_argument("--w-down-had-group-size", type=int, default=32)
    parser.add_argument("--qk-had-group-size", type=int, default=32)
    parser.add_argument("--rotation-block-size", type=int, default=0)
    parser.add_argument(
        "--rotation-init",
        choices=["random_hadamard", "hadamard"],
        default="random_hadamard",
    )
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--w-gptq-bits", type=int, default=4)
    parser.add_argument("--w-gptq-group-size", type=int, default=32)
    parser.add_argument("--w-gptq-damp-pct", type=float, default=0.01)
    parser.add_argument("--w-gptq-clip-ratio", type=float, default=1.0)
    parser.add_argument("--no-reorder", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_arg, model_name, token=None):
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    config = AutoConfig.from_pretrained(model_name, token=token)
    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = config_dtype.replace("torch.", "")
        if config_dtype in ["bfloat16", "bf16"]:
            return torch.bfloat16
        if config_dtype in ["float16", "fp16", "half"]:
            return torch.float16
    if config_dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cfg(args):
    from bfp_llama.config import ExperimentConfig

    return ExperimentConfig(
        model=args.model,
        max_length=args.seqlen,
        seed=args.seed,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        kv_bits=args.kv_bits,
        bfp_group_size=args.bfp_group_size,
        w_bfp_group_size=args.bfp_group_size,
        a_bfp_group_size=args.bfp_group_size,
        kv_bfp_group_size=args.bfp_group_size,
        online_had_group_size=args.online_had_group_size,
        w_down_had_group_size=args.w_down_had_group_size,
        qk_had_group_size=args.qk_had_group_size,
        qk_matmul_bits=args.qk_matmul_bits if args.qk_matmul_bits is not None else args.kv_bits,
        av_matmul_bits=args.av_matmul_bits if args.av_matmul_bits is not None else args.kv_bits,
        qk_matmul_bfp_group_size=args.qk_matmul_bfp_group_size,
        av_matmul_bfp_group_size=args.av_matmul_bfp_group_size,
        rotation_block_size=args.rotation_block_size,
        rotation_init=args.rotation_init,
        rotate=not args.no_rotate,
    )


def find_rotation_path(args, cfg):
    if not cfg.rotate:
        return None
    if args.rotation_path is not None:
        return args.rotation_path
    if args.experiment_dir is None:
        raise FileNotFoundError(
            "Pass --rotation-path or --experiment-dir for BFP-GPTQ with rotation."
        )

    from bfp_llama.modeling import rotation_filename

    experiment_dir = Path(args.experiment_dir)
    expected = experiment_dir / rotation_filename(cfg)
    if expected.exists():
        return str(expected)

    matches = sorted(experiment_dir.glob("R_*_*_*_*.bin"))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise FileNotFoundError(
            f"{expected.name} not found and multiple rotation files exist: {names}. "
            "Pass --rotation-path explicitly."
        )
    raise FileNotFoundError(f"rotation file not found: {expected}")


def _tensor_to(x, device):
    return x.to(device) if torch.is_tensor(x) else x


@torch.no_grad()
def capture_first_layer_inputs(model, batches, device):
    from utils.rotation_utils import apply_rotation_right

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if hasattr(model, "_bfp_input_rotation_hook"):
        model._bfp_input_rotation_hook.remove()
        delattr(model, "_bfp_input_rotation_hook")

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    layers[0] = layers[0].to(device)

    inps = [None] * len(batches)
    attention_masks = [None] * len(batches)
    position_ids = [None] * len(batches)
    cache = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            if hasattr(model, "bfp_R1"):
                hidden_states = apply_rotation_right(
                    hidden_states,
                    model.bfp_R1.weight,
                    torch.float64,
                )
            inps[cache["i"]] = hidden_states.detach().cpu()
            attention_masks[cache["i"]] = kwargs.get("attention_mask")
            position_ids[cache["i"]] = kwargs.get("position_ids")
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in tqdm(batches, desc="Collect layer 0 inputs"):
        try:
            model(input_ids=batch.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return inps, attention_masks, position_ids


def _layer_targets(layer, layer_idx):
    targets = []
    for name, module in layer.named_modules():
        if hasattr(module, "_effective_weight") and hasattr(module, "_quant_input"):
            if getattr(module, "role", None) == "lm_head":
                continue
            full_name = f"model.layers.{layer_idx}.{name}" if name else f"model.layers.{layer_idx}"
            targets.append((full_name, module))
    return targets


def _register_hessian_hooks(targets, group_size, device):
    stats = {}
    handles = []
    for full_name, module in targets:
        in_features = module.weight.shape[1]
        if in_features % group_size != 0:
            raise ValueError(
                f"{full_name} in_features={in_features} is not divisible by GPTQ group size {group_size}."
            )
        n_groups = in_features // group_size
        stats[full_name] = torch.zeros(
            n_groups,
            group_size,
            group_size,
            device=device,
            dtype=torch.float32,
        )

        def make_hook(name, mod, n_groups):
            def hook(_module, inputs):
                x = mod._quant_input(inputs[0]).detach().float()
                x = x.reshape(-1, mod.weight.shape[1])
                x = x.reshape(-1, n_groups, group_size)
                stats[name] += 2.0 * torch.einsum("nkg,nkh->kgh", x, x)

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(full_name, module, n_groups)))
    return stats, handles


@torch.no_grad()
def run_layer(layer, inps, attention_masks, position_ids, device):
    outs = []
    for idx, hidden_states in enumerate(inps):
        out = layer(
            hidden_states.to(device),
            attention_mask=_tensor_to(attention_masks[idx], device),
            position_ids=_tensor_to(position_ids[idx], device),
        )[0]
        outs.append(out.detach().cpu())
    return outs


def apply_layer_bfp_gptq(layer, layer_idx, stats, args):
    from utils.bfp_gptq import bfp_gptq_from_block_hessians

    saved = {}
    for full_name, module in tqdm(_layer_targets(layer, layer_idx), desc=f"GPTQ layer {layer_idx}", leave=False):
        W_eff = module._effective_weight().detach().float()
        result = bfp_gptq_from_block_hessians(
            W_eff,
            stats[full_name],
            bits=args.w_gptq_bits,
            group_size=args.w_gptq_group_size,
            damp_pct=args.w_gptq_damp_pct,
            clip_ratio=args.w_gptq_clip_ratio,
            reorder=not args.no_reorder,
        )
        W_q = result["W_quant"].to(device=module.linear.weight.device, dtype=torch.float32)
        if hasattr(module, "bfp_gptq_weight"):
            module.bfp_gptq_weight.data.copy_(W_q)
        else:
            module.register_buffer("bfp_gptq_weight", W_q)
        saved[full_name] = W_q.detach().cpu()
    return saved


@torch.no_grad()
def calibrate_bfp_gptq(model, batches, args, device):
    inps, attention_masks, position_ids = capture_first_layer_inputs(model, batches, device)
    all_weights = {}
    layers = model.model.layers

    for layer_idx in tqdm(range(len(layers)), desc="BFP-GPTQ layers"):
        layer = layers[layer_idx].to(device)
        targets = _layer_targets(layer, layer_idx)
        stats, handles = _register_hessian_hooks(targets, args.w_gptq_group_size, device)
        _ = run_layer(layer, inps, attention_masks, position_ids, device)
        for handle in handles:
            handle.remove()

        all_weights.update(apply_layer_bfp_gptq(layer, layer_idx, stats, args))
        inps = run_layer(layer, inps, attention_masks, position_ids, device)

        layers[layer_idx] = layer.cpu()
        del layer, stats
        torch.cuda.empty_cache()
    return all_weights


def main():
    args = parse_args()
    set_seed(args.seed)

    from bfp_llama.data import random_calibration_loader
    from bfp_llama.modeling import load_llama_causal_lm, load_llama_tokenizer, setup_bfp_llama

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, args.model, token=args.access_token)
    cfg = make_cfg(args)
    rotation_path = find_rotation_path(args, cfg)

    print(f"Loading model: {args.model}")
    print(f"Runtime quantization: {args.w_bits}_{args.a_bits}_{args.kv_bits}")
    print(f"BFP-GPTQ weight quantization: {args.w_gptq_bits}-bit, group={args.w_gptq_group_size}")
    print(f"QK/AV matmul BFP bits: {cfg.qk_matmul_bits}/{cfg.av_matmul_bits}")
    if cfg.rotate:
        print(f"Using rotation path: {rotation_path}")

    tokenizer = load_llama_tokenizer(args.model, args.seqlen, token=args.access_token)
    model = load_llama_causal_lm(args.model, dtype, token=args.access_token, cfg=cfg)
    model = setup_bfp_llama(
        model,
        cfg,
        trainable_rotations=False,
        rotation_path=rotation_path,
    )
    model.eval()

    calib = random_calibration_loader(
        args.calib_dataset,
        tokenizer,
        nsamples=args.calib_samples,
        seqlen=args.seqlen,
        seed=args.seed,
    )
    batches = [
        torch.cat(calib[i : i + args.batch_size], dim=0)
        for i in range(0, len(calib), args.batch_size)
    ]

    weights = calibrate_bfp_gptq(model, batches, args, device)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "model": args.model,
                "rotation_path": rotation_path,
                "runtime_bits": (args.w_bits, args.a_bits, args.kv_bits),
                "w_gptq_bits": args.w_gptq_bits,
                "w_gptq_group_size": args.w_gptq_group_size,
                "calib_dataset": args.calib_dataset,
                "calib_samples": args.calib_samples,
                "seqlen": args.seqlen,
                "seed": args.seed,
            },
            "weights": weights,
        },
        output_path,
    )
    print(f"Saved BFP-GPTQ weights: {output_path}")


if __name__ == "__main__":
    main()
