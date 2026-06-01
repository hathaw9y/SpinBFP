import argparse
import sys
import weakref
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REFACTOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = REFACTOR_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from bfp_llama import (
    load_llama_causal_lm,
    load_llama_tokenizer,
    load_opt_causal_lm,
    load_opt_tokenizer,
    rotation_filename,
    setup_bfp_llama,
    setup_bfp_opt as setup_bfp_opt_common,
)
from bfp_llama.bfp import bfp_quant_dequant, set_bfp_exponent_rounding
from utils.quant_utils import (
    set_bfp_exponent_rounding as set_quant_utils_bfp_exponent_rounding,
)
from bfp_llama.config import ExperimentConfig
from bfp_llama.data import eval_tokens
from utils.rotation_utils import apply_rotation_left, apply_rotation_right, rotation_total_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA BFP perplexity.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--rotation-path", default=None)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default=None)
    parser.add_argument("--eval-nsamples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--bfp-group-size", type=int, default=32)
    parser.add_argument("--bfp-exponent-rounding", choices=["floor", "ceil"], default="floor")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--rotation-compute-dtype", choices=["fp64", "fp32"], default="fp64")
    parser.add_argument("--online-had-group-size", type=int, default=32)
    parser.add_argument("--w-down-had-group-size", type=int, default=32)
    parser.add_argument("--qk-had-group-size", type=int, default=32)
    parser.add_argument("--qk-matmul-bits", type=int, default=None)
    parser.add_argument("--av-matmul-bits", type=int, default=None)
    parser.add_argument("--qk-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--av-matmul-bfp-group-size", type=int, default=32)
    parser.add_argument("--rotation-block-size", type=int, default=32)
    parser.add_argument("--no-rotate", action="store_true")
    return parser.parse_args()


def load_config(args):
    return ExperimentConfig(
        model=args.model,
        max_length=args.max_length,
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
        qk_matmul_bits=args.qk_matmul_bits or args.kv_bits,
        av_matmul_bits=args.av_matmul_bits or args.kv_bits,
        qk_matmul_bfp_group_size=args.qk_matmul_bfp_group_size,
        av_matmul_bfp_group_size=args.av_matmul_bfp_group_size,
        rotation_block_size=args.rotation_block_size,
        rotate=not args.no_rotate,
    )


def resolve_dtype(dtype_arg, model_name, token=None):
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16

    hf_config = AutoConfig.from_pretrained(model_name, token=token)
    config_dtype = getattr(hf_config, "torch_dtype", None)
    if isinstance(config_dtype, str):
        config_dtype = config_dtype.replace("torch.", "")
        if config_dtype in ["bfloat16", "bf16"]:
            return torch.bfloat16
        if config_dtype in ["float16", "fp16", "half"]:
            return torch.float16
    if config_dtype == torch.bfloat16:
        return torch.bfloat16
    if config_dtype == torch.float16:
        return torch.float16
    return torch.float16


def resolve_rotation_compute_dtype(dtype_arg="fp64"):
    return torch.float32 if dtype_arg == "fp32" else torch.float64


class OptEvalRotationLinear(nn.Module):
    def __init__(self, linear, role, cfg, model, compute_dtype, r2_module=None):
        super().__init__()
        self.linear = linear
        self.role = role
        self.cfg = cfg
        object.__setattr__(self, "_model_ref", weakref.ref(model))
        self.compute_dtype = compute_dtype
        object.__setattr__(self, "_r2_ref", weakref.ref(r2_module) if r2_module is not None else None)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _r1(self):
        model = self._model_ref()
        return None if model is None else getattr(model, "R1", None)

    def _r2(self):
        r2_ref = getattr(self, "_r2_ref", None)
        r2_module = r2_ref() if r2_ref is not None else None
        return None if r2_module is None else r2_module.weight

    def _apply_r2_to_weight(self, weight, transpose=False):
        r2 = self._r2()
        if r2 is None:
            return weight
        had_dim = rotation_total_dim(r2)
        dtype = weight.dtype
        if transpose:
            shape = weight.shape
            temp = weight.reshape(-1, shape[-1] // had_dim, had_dim)
            temp = apply_rotation_right(temp, r2, self.compute_dtype)
            return temp.reshape(shape).to(dtype)
        wt = weight.t()
        shape = wt.shape
        temp = wt.reshape(-1, shape[-1] // had_dim, had_dim)
        temp = apply_rotation_right(temp, r2, self.compute_dtype)
        return temp.reshape(shape).t().to(dtype)

    def _effective_weight(self):
        weight = self.linear.weight
        r1_module = self._r1()
        if not self.cfg.rotate or r1_module is None:
            return weight
        r1 = r1_module.weight
        dtype = weight.dtype
        if self.role in ["q_proj", "k_proj", "v_proj", "fc1"]:
            weight = apply_rotation_right(weight, r1, self.compute_dtype)
        elif self.role in ["out_proj", "fc2"]:
            weight = apply_rotation_left(weight, r1, self.compute_dtype, transpose=True)
        if self.role == "v_proj":
            weight = self._apply_r2_to_weight(weight, transpose=False)
        elif self.role == "out_proj":
            weight = self._apply_r2_to_weight(weight, transpose=True)
        return weight

    def forward(self, x):
        x_dtype = x.dtype
        if self.role == "lm_head":
            r1_module = self._r1()
            if self.cfg.rotate and r1_module is not None:
                r1 = r1_module.weight
                x = apply_rotation_right(x, r1, self.compute_dtype, transpose=True)
            return self.linear(x).to(x_dtype)

        x = bfp_quant_dequant(x, self.cfg.a_bits, self.cfg.a_bfp_group_size)
        weight = self._effective_weight()
        weight = bfp_quant_dequant(weight, self.cfg.w_bits, self.cfg.w_bfp_group_size)
        out = nn.functional.linear(x, weight, self.linear.bias).to(x_dtype)
        if self.role == "v_proj":
            out = bfp_quant_dequant(out, self.cfg.kv_bits, self.cfg.kv_bfp_group_size)
        return out


def setup_bfp_opt(model, cfg, rotation_path=None, compute_dtype=torch.float64):
    for param in model.parameters():
        param.requires_grad = False

    rotations = None
    if cfg.rotate:
        if rotation_path is None:
            raise FileNotFoundError("OPT rotation eval requires a rotation file unless --no-rotate is set.")
        rotations = torch.load(rotation_path, map_location="cpu")
        r1 = rotations["R1"].cuda()
        model.R1 = nn.Module()
        model.R1.register_buffer("weight", r1.to(torch.float32))
        for idx, layer in enumerate(model.model.decoder.layers):
            key = f"model.decoder.layers.{idx}.self_attn.R2"
            if key not in rotations:
                raise KeyError(f"OPT rotation file is missing {key}; retrain OPT rotations with R2 support.")
            layer.self_attn.R2 = nn.Module()
            layer.self_attn.R2.register_buffer("weight", rotations[key].cuda().to(torch.float32))

    for layer in model.model.decoder.layers:
        attn = layer.self_attn
        r2 = getattr(attn, "R2", None)
        attn.q_proj = OptEvalRotationLinear(attn.q_proj, "q_proj", cfg, model, compute_dtype)
        attn.k_proj = OptEvalRotationLinear(attn.k_proj, "k_proj", cfg, model, compute_dtype)
        attn.v_proj = OptEvalRotationLinear(attn.v_proj, "v_proj", cfg, model, compute_dtype, r2_module=r2)
        attn.out_proj = OptEvalRotationLinear(attn.out_proj, "out_proj", cfg, model, compute_dtype, r2_module=r2)
        layer.fc1 = OptEvalRotationLinear(layer.fc1, "fc1", cfg, model, compute_dtype)
        layer.fc2 = OptEvalRotationLinear(layer.fc2, "fc2", cfg, model, compute_dtype)
    model.lm_head = OptEvalRotationLinear(model.lm_head, "lm_head", cfg, model, compute_dtype)
    if cfg.rotate and not hasattr(model, "_bfp_opt_input_rotation_hook"):
        def rotate_input(module, args, kwargs):
            if not hasattr(model, "R1"):
                return args, kwargs

            def rotate(x):
                return apply_rotation_right(x, model.R1.weight, compute_dtype)

            if len(args) > 0:
                return (rotate(args[0]),) + args[1:], kwargs
            kwargs["hidden_states"] = rotate(kwargs["hidden_states"])
            return args, kwargs

        model._bfp_opt_input_rotation_hook = model.model.decoder.layers[0].register_forward_pre_hook(
            rotate_input, with_kwargs=True
        )
    return model


@torch.no_grad()
def evaluate_forward(model, input_ids, seqlen, batch_size, device):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    nsamples = input_ids.numel() // seqlen
    input_ids = input_ids[:, : nsamples * seqlen].reshape(nsamples, seqlen).to(device)

    total_loss = 0.0
    total_tokens = 0
    for start in tqdm(range(0, nsamples, batch_size), desc="Eval batches"):
        batch = input_ids[start : start + batch_size]
        logits = model(input_ids=batch).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.float().item()
        total_tokens += shift_labels.numel()

    model.config.use_cache = use_cache
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def find_rotation_path(args, cfg):
    if not cfg.rotate:
        if args.rotation_path is not None:
            print("--no-rotate is set; ignoring --rotation-path.")
        return None
    if args.rotation_path is not None:
        return args.rotation_path
    if args.experiment_dir is None:
        return None

    experiment_dir = Path(args.experiment_dir)
    expected = experiment_dir / rotation_filename(cfg)
    if expected.exists():
        return str(expected)

    matches = sorted(
        path
        for path in experiment_dir.glob("R_*_*_*_*.bin")
        if path.name.count("_") == 4
    )
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        available = ", ".join(path.name for path in matches)
        raise FileNotFoundError(
            f"{expected.name} not found and multiple rotation files exist: {available}. "
            "Pass --w-bits/--a-bits/--kv-bits or --rotation-path explicitly."
        )
    legacy = experiment_dir / f"R_{cfg.w_bits}_{cfg.a_bits}_{cfg.kv_bits}.bin"
    if legacy.exists():
        return str(legacy)

    matches = sorted(experiment_dir.glob("R_*_*_*.bin"))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        available = ", ".join(path.name for path in matches)
        raise FileNotFoundError(
            f"{expected.name} not found and multiple legacy rotation files exist: {available}. "
            "Pass --rotation-path explicitly."
        )
    raise FileNotFoundError(f"rotation file not found: {expected}")


@torch.no_grad()
def evaluate_layerwise(model, input_ids, seqlen, batch_size, device):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if hasattr(model, "_bfp_input_rotation_hook"):
        model._bfp_input_rotation_hook.remove()
        delattr(model, "_bfp_input_rotation_hook")

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    layers[0] = layers[0].to(device)

    nsamples = input_ids.numel() // seqlen
    input_ids = input_ids[:, : nsamples * seqlen].reshape(nsamples, seqlen).to(device)
    batches = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]

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
            inps[cache["i"]] = hidden_states
            attention_masks[cache["i"]] = kwargs["attention_mask"]
            position_ids[cache["i"]] = kwargs["position_ids"]
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in batches:
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = [None] * len(batches)
    for layer_idx in tqdm(range(len(layers)), desc="Eval layers"):
        layer = layers[layer_idx].to(device)
        for batch_idx in range(len(batches)):
            outs[batch_idx] = layer(
                inps[batch_idx],
                attention_mask=attention_masks[batch_idx],
                position_ids=position_ids[batch_idx],
            )[0]
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    total_loss = 0.0
    total_tokens = 0
    for hidden_states, batch in zip(inps, batches):
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.float().item()
        total_tokens += shift_labels.numel()

    model.config.use_cache = use_cache
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main():
    args = parse_args()
    set_bfp_exponent_rounding(args.bfp_exponent_rounding)
    set_quant_utils_bfp_exponent_rounding(args.bfp_exponent_rounding)
    cfg = load_config(args)
    dataset_name = args.dataset or "wikitext2"
    dtype = resolve_dtype(args.dtype, args.model, token=args.access_token)
    rotation_compute_dtype = resolve_rotation_compute_dtype(args.rotation_compute_dtype)
    print(f"Using compute dtype: {dtype}")
    print(f"Using rotation compute dtype: {rotation_compute_dtype}")
    print(f"Using BFP exponent rounding: {args.bfp_exponent_rounding}")

    hf_config = AutoConfig.from_pretrained(args.model, token=args.access_token)
    rotation_path = find_rotation_path(args, cfg)
    if cfg.rotate:
        print(f"Using rotation path: {rotation_path}")
    else:
        print("Evaluating without rotation.")

    if hf_config.model_type == "llama":
        model = load_llama_causal_lm(args.model, dtype=dtype, token=args.access_token, cfg=cfg)
        model = setup_bfp_llama(
            model,
            cfg,
            trainable_rotations=False,
            rotation_path=rotation_path if cfg.rotate else None,
        )
        model.cuda()
        tokenizer = load_llama_tokenizer(args.model, cfg.max_length, token=args.access_token)
        eval_fn = evaluate_layerwise
    elif hf_config.model_type == "opt":
        model = load_opt_causal_lm(args.model, dtype=dtype, token=args.access_token).cuda()
        model = setup_bfp_opt_common(
            model,
            cfg,
            trainable_rotations=False,
            rotation_path=rotation_path if cfg.rotate else None,
            compute_dtype=rotation_compute_dtype,
        )
        tokenizer = load_opt_tokenizer(args.model, cfg.max_length, token=args.access_token)
        eval_fn = evaluate_forward
    else:
        raise ValueError(f"bfp_refactor eval supports LLaMA and OPT configs, got {hf_config.model_type}")

    model.config.use_cache = False
    tokens = eval_tokens(
        dataset_name,
        tokenizer,
        seqlen=cfg.max_length,
        nsamples=args.eval_nsamples,
        seed=cfg.seed,
    )
    ppl = eval_fn(model, tokens, cfg.max_length, args.batch_size, "cuda")
    print(f"{dataset_name} ppl: {ppl:.4f}")


if __name__ == "__main__":
    main()
