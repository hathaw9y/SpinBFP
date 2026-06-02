import copy
import functools
import types
import weakref
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .bfp import bfp_quant_dequant
from .hadamard import (
    apply_block_had,
    apply_exact_had_to_linear,
    apply_full_had,
    apply_head_had,
    apply_r2_to_weight,
    hadamard,
    random_hadamard,
)
from utils.rotation_utils import apply_rotation_left, apply_rotation_right


class RotationModule(nn.Module):
    def __init__(self, matrix, trainable):
        super().__init__()
        if trainable:
            self.weight = nn.Parameter(matrix.to(torch.float32))
        else:
            self.register_buffer("weight", matrix.to(torch.float32))


def _random_rotation(size, cfg, device):
    rotation_init = getattr(cfg, "rotation_init", "random_hadamard")
    if rotation_init == "hadamard":
        init_fn = hadamard
    elif rotation_init == "random_hadamard":
        init_fn = random_hadamard
    else:
        raise ValueError(f"Unsupported rotation init: {rotation_init}")

    block_size = getattr(cfg, "rotation_block_size", 0)
    if block_size and block_size > 0:
        if size % block_size != 0:
            raise ValueError(f"rotation size {size} must be divisible by block size {block_size}")
        return torch.stack(
            [init_fn(block_size, device) for _ in range(size // block_size)],
            dim=0,
        )
    return init_fn(size, device)


def _copy_func_with_new_globals(func, globals_dict):
    new_func = types.FunctionType(
        func.__code__,
        globals_dict,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
    new_func = functools.update_wrapper(new_func, func)
    new_func.__kwdefaults__ = copy.copy(func.__kwdefaults__)
    return new_func


def _add_wrapper_after_function_call(module, method_name, function_name, wrapper_factory):
    original = getattr(module, method_name).__func__
    method_globals = dict(original.__globals__)
    wrapper = wrapper_factory(method_globals[function_name])
    method_globals[function_name] = wrapper
    patched = _copy_func_with_new_globals(original, method_globals)
    setattr(module, method_name, patched.__get__(module))
    return wrapper


class BfpRotationLinear(nn.Module):
    def __init__(self, linear, role, cfg, model=None, attn=None):
        super().__init__()
        self.linear = linear
        self.role = role
        self.cfg = cfg
        object.__setattr__(self, "_model_ref", weakref.ref(model) if model is not None else None)
        object.__setattr__(self, "_attn_ref", weakref.ref(attn) if attn is not None else None)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _r1(self):
        model_ref = getattr(self, "_model_ref", None)
        model = model_ref() if model_ref is not None else None
        if model is None or not hasattr(model, "bfp_R1"):
            return None
        return model.bfp_R1.weight

    def _r2(self):
        attn_ref = getattr(self, "_attn_ref", None)
        attn = attn_ref() if attn_ref is not None else None
        if attn is None or not hasattr(attn, "bfp_R2"):
            return None
        return attn.bfp_R2.weight

    def _effective_weight(self):
        weight = self.linear.weight
        if not self.cfg.rotate:
            return weight

        r1 = self._r1()
        if r1 is None:
            return weight

        dtype = weight.dtype
        if self.role in ["q_proj", "k_proj", "gate_proj", "up_proj", "lm_head"]:
            weight = apply_rotation_right(weight, r1, torch.float64)
        elif self.role in ["o_proj", "down_proj"]:
            weight = apply_rotation_left(weight, r1, torch.float64, transpose=True)
        elif self.role == "v_proj":
            weight = apply_rotation_right(weight, r1, torch.float64)

        if self.role == "v_proj":
            weight = apply_r2_to_weight(weight, self._r2(), transpose=False)
        elif self.role == "o_proj":
            weight = apply_r2_to_weight(weight, self._r2(), transpose=True)

        return weight

    def _quant_input(self, x):
        if self.role == "lm_head":
            return x
        if self.cfg.rotate and self.role == "down_proj":
            if self.cfg.w_down_had_group_size > 0:
                x = apply_block_had(x, self.cfg.w_down_had_group_size)
            else:
                x = apply_full_had(x)
        return bfp_quant_dequant(x, self.cfg.a_bits, self.cfg.a_bfp_group_size)

    def forward(self, x):
        x_dtype = x.dtype
        x = self._quant_input(x)
        use_gptq_weight = hasattr(self, "bfp_gptq_weight")
        if use_gptq_weight:
            weight = self.bfp_gptq_weight.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        else:
            weight = self._effective_weight()
        if self.role != "lm_head" and not use_gptq_weight:
            weight = bfp_quant_dequant(weight, self.cfg.w_bits, self.cfg.w_bfp_group_size)
        out = nn.functional.linear(x, weight, self.linear.bias).to(x_dtype)
        if self.role == "v_proj":
            out = bfp_quant_dequant(out, self.cfg.kv_bits, self.cfg.kv_bfp_group_size)
        return out



class OptBfpRotationLinear(nn.Module):
    def __init__(self, linear, role, cfg, model=None, attn=None, compute_dtype=torch.float64):
        super().__init__()
        self.linear = linear
        self.role = role
        self.cfg = cfg
        self.compute_dtype = compute_dtype
        object.__setattr__(self, "_model_ref", weakref.ref(model) if model is not None else None)
        object.__setattr__(self, "_attn_ref", weakref.ref(attn) if attn is not None else None)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _r1(self):
        model_ref = getattr(self, "_model_ref", None)
        model = model_ref() if model_ref is not None else None
        if model is None or not hasattr(model, "bfp_R1"):
            return None
        return model.bfp_R1.weight

    def _r2(self):
        attn_ref = getattr(self, "_attn_ref", None)
        attn = attn_ref() if attn_ref is not None else None
        if attn is None or not hasattr(attn, "bfp_R2"):
            return None
        return attn.bfp_R2.weight

    def _apply_r2_to_weight(self, weight, transpose=False):
        return apply_r2_to_weight(weight, self._r2(), transpose=transpose)

    def _effective_weight(self):
        weight = self.linear.weight
        if not self.cfg.rotate:
            return weight

        r1 = self._r1()
        if r1 is None:
            return weight

        dtype = weight.dtype
        compute_dtype = self.compute_dtype
        if self.role in ["q_proj", "k_proj", "v_proj", "fc1", "lm_head"]:
            weight = apply_rotation_right(weight, r1, compute_dtype)
        elif self.role in ["out_proj", "fc2"]:
            weight = apply_rotation_left(weight, r1, compute_dtype, transpose=True)

        if self.role == "v_proj":
            weight = self._apply_r2_to_weight(weight, transpose=False)
        elif self.role == "out_proj":
            weight = self._apply_r2_to_weight(weight, transpose=True)
        return weight

    def forward(self, x):
        x_dtype = x.dtype
        if self.role == "lm_head":
            weight = self._effective_weight()
            return nn.functional.linear(x, weight, self.linear.bias).to(x_dtype)

        x = bfp_quant_dequant(x, self.cfg.a_bits, self.cfg.a_bfp_group_size)
        use_gptq_weight = hasattr(self, "bfp_gptq_weight")
        if use_gptq_weight:
            weight = self.bfp_gptq_weight.to(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        else:
            weight = self._effective_weight()
        if not use_gptq_weight:
            weight = bfp_quant_dequant(weight, self.cfg.w_bits, self.cfg.w_bfp_group_size)
        out = nn.functional.linear(x, weight, self.linear.bias).to(x_dtype)
        if self.role == "v_proj":
            out = bfp_quant_dequant(out, self.cfg.kv_bits, self.cfg.kv_bfp_group_size)
        return out


class QKRotationBfpWrapper(nn.Module):
    def __init__(self, func, cfg):
        super().__init__()
        self.func = func
        self.cfg = cfg

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        if self.cfg.rotate:
            q = apply_head_had(q.float(), self.cfg.qk_had_group_size).to(q.dtype)
            k = apply_head_had(k.float(), self.cfg.qk_had_group_size).to(k.dtype)
        k = bfp_quant_dequant(k, self.cfg.kv_bits, self.cfg.kv_bfp_group_size)
        return q, k


class _AttentionMatmulBfpTorchProxy:
    def __init__(self, torch_module, cfg):
        self._torch = torch_module
        self.cfg = cfg
        self.matmul_count = 0

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def matmul(self, left, right):
        matmul_idx = self.matmul_count % 2
        self.matmul_count += 1
        if matmul_idx == 0 and self.cfg.qk_matmul_bits < 16:
            left = bfp_quant_dequant(
                left,
                self.cfg.qk_matmul_bits,
                self.cfg.qk_matmul_bfp_group_size,
            )
            right = bfp_quant_dequant(
                right.transpose(-2, -1),
                self.cfg.qk_matmul_bits,
                self.cfg.qk_matmul_bfp_group_size,
            ).transpose(-2, -1)
        elif matmul_idx == 1 and self.cfg.av_matmul_bits < 16:
            left = bfp_quant_dequant(
                left,
                self.cfg.av_matmul_bits,
                self.cfg.av_matmul_bfp_group_size,
            )
            right = bfp_quant_dequant(
                right,
                self.cfg.av_matmul_bits,
                self.cfg.av_matmul_bfp_group_size,
            )
        return self._torch.matmul(left, right)


def _fuse_ln_linear(layernorm, linear_layers):
    for linear in linear_layers:
        target = linear.linear if isinstance(linear, BfpRotationLinear) else linear
        dtype = target.weight.dtype
        weight = target.weight.data.double()
        target.weight.data = (weight * layernorm.weight.double()).to(dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if target.bias is None:
                target.bias = nn.Parameter(
                    torch.zeros(target.out_features, dtype=torch.float64, device=target.weight.device)
                )
            target.bias.data = target.bias.data.double() + torch.matmul(
                weight, layernorm.bias.double()
            )
            target.bias.data = target.bias.data.to(dtype)


def _fuse_llama_norms(model):
    embed = model.model.embed_tokens
    embed.weight.data = (
        embed.weight.data.double() - embed.weight.data.double().mean(dim=-1, keepdim=True)
    ).to(embed.weight.dtype)

    for layer in model.model.layers:
        _fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        _fuse_ln_linear(
            layer.post_attention_layernorm,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )
        layer.input_layernorm.weight.data = torch.ones_like(layer.input_layernorm.weight.data)
        layer.post_attention_layernorm.weight.data = torch.ones_like(
            layer.post_attention_layernorm.weight.data
        )

    _fuse_ln_linear(model.model.norm, [model.lm_head])
    model.model.norm.weight.data = torch.ones_like(model.model.norm.weight.data)


def _replace_linears(model, cfg):
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj = BfpRotationLinear(attn.q_proj, "q_proj", cfg, model=model, attn=attn)
        attn.k_proj = BfpRotationLinear(attn.k_proj, "k_proj", cfg, model=model, attn=attn)
        attn.v_proj = BfpRotationLinear(attn.v_proj, "v_proj", cfg, model=model, attn=attn)
        attn.o_proj = BfpRotationLinear(attn.o_proj, "o_proj", cfg, model=model, attn=attn)
        layer.mlp.gate_proj = BfpRotationLinear(layer.mlp.gate_proj, "gate_proj", cfg, model=model)
        layer.mlp.up_proj = BfpRotationLinear(layer.mlp.up_proj, "up_proj", cfg, model=model)
        layer.mlp.down_proj = BfpRotationLinear(layer.mlp.down_proj, "down_proj", cfg, model=model)

    model.lm_head = BfpRotationLinear(model.lm_head, "lm_head", cfg, model=model)


def _load_rotation_state(rotation_path):
    if rotation_path is None:
        return None
    return torch.load(rotation_path, map_location="cpu")


def _set_rotations(model, cfg, trainable, rotation_path=None, rotations=None):
    if rotations is None:
        rotations = _load_rotation_state(rotation_path)
    if rotations is not None:
        r1 = rotations["R1"].cuda()
    else:
        r1 = _random_rotation(model.config.hidden_size, cfg, "cuda")

    model.bfp_R1 = RotationModule(r1, trainable=trainable)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for idx, layer in enumerate(model.model.layers):
        if rotations is None:
            r2 = _random_rotation(head_dim, cfg, "cuda")
        else:
            r2 = rotations[f"model.layers.{idx}.self_attn.R2"].cuda()
        layer.self_attn.bfp_R2 = RotationModule(r2, trainable=trainable)


def _add_input_rotation_hook(model):
    if hasattr(model, "_bfp_input_rotation_hook"):
        return

    def hook(module, args, kwargs):
        if not hasattr(model, "bfp_R1"):
            return args, kwargs

        def rotate(x):
            return apply_rotation_right(x, model.bfp_R1.weight, torch.float64)

        if len(args) > 0:
            return (rotate(args[0]),) + args[1:], kwargs
        kwargs["hidden_states"] = rotate(kwargs["hidden_states"])
        return args, kwargs

    model._bfp_input_rotation_hook = model.model.layers[0].register_forward_pre_hook(
        hook, with_kwargs=True
    )


def _add_qk_wrappers(model, cfg):
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "apply_rotary_pos_emb_bfp_wrapper"):
            continue
        wrapper = _add_wrapper_after_function_call(
            attn,
            "forward",
            "apply_rotary_pos_emb",
            functools.partial(QKRotationBfpWrapper, cfg=cfg),
        )
        attn.apply_rotary_pos_emb_bfp_wrapper = wrapper


def add_attention_matmul_bfp(model, cfg):
    if cfg.qk_matmul_bits >= 16 and cfg.av_matmul_bits >= 16:
        return
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "bfp_attention_matmul_proxy"):
            continue
        original = attn.forward.__func__
        method_globals = dict(original.__globals__)
        proxy = _AttentionMatmulBfpTorchProxy(method_globals["torch"], cfg)
        method_globals["torch"] = proxy
        patched = _copy_func_with_new_globals(original, method_globals)
        setattr(attn, "forward", patched.__get__(attn))
        attn.bfp_attention_matmul_proxy = proxy


def _rotate_down_weights(model, cfg):
    for layer in model.model.layers:
        base = layer.mlp.down_proj.linear
        apply_exact_had_to_linear(base, cfg.w_down_had_group_size)


def setup_bfp_llama(model, cfg, trainable_rotations, rotation_path=None):
    assert getattr(model.config, "model_type", None) == "llama", (
        "bfp_refactor intentionally supports only LLaMA-family models"
    )

    for param in model.parameters():
        param.requires_grad = False

    rotations = _load_rotation_state(rotation_path) if cfg.rotate and rotation_path is not None else None
    if cfg.rotate:
        _set_rotations(model, cfg, trainable=trainable_rotations, rotation_path=rotation_path, rotations=rotations)

    _replace_linears(model, cfg)
    if cfg.rotate:
        _add_input_rotation_hook(model)
        _fuse_llama_norms(model)
        _rotate_down_weights(model, cfg)

    if cfg.kv_bits < 16:
        _add_qk_wrappers(model, cfg)
    add_attention_matmul_bfp(model, cfg)
    return model



def _opt_layers(model):
    return model.model.decoder.layers


def _replace_opt_linears(model, cfg, compute_dtype):
    for layer in _opt_layers(model):
        attn = layer.self_attn
        attn.q_proj = OptBfpRotationLinear(attn.q_proj, "q_proj", cfg, model=model, attn=attn, compute_dtype=compute_dtype)
        attn.k_proj = OptBfpRotationLinear(attn.k_proj, "k_proj", cfg, model=model, attn=attn, compute_dtype=compute_dtype)
        attn.v_proj = OptBfpRotationLinear(attn.v_proj, "v_proj", cfg, model=model, attn=attn, compute_dtype=compute_dtype)
        attn.out_proj = OptBfpRotationLinear(attn.out_proj, "out_proj", cfg, model=model, attn=attn, compute_dtype=compute_dtype)
        layer.fc1 = OptBfpRotationLinear(layer.fc1, "fc1", cfg, model=model, compute_dtype=compute_dtype)
        layer.fc2 = OptBfpRotationLinear(layer.fc2, "fc2", cfg, model=model, compute_dtype=compute_dtype)
    model.lm_head = OptBfpRotationLinear(model.lm_head, "lm_head", cfg, model=model, compute_dtype=compute_dtype)


def _set_opt_rotations(model, cfg, trainable, rotation_path=None, rotations=None):
    if rotations is None:
        rotations = _load_rotation_state(rotation_path)
    if rotations is not None:
        r1 = rotations["R1"].cuda()
    else:
        r1 = _random_rotation(model.config.hidden_size, cfg, "cuda")

    model.bfp_R1 = RotationModule(r1, trainable=trainable)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for idx, layer in enumerate(_opt_layers(model)):
        if rotations is None:
            r2 = _random_rotation(head_dim, cfg, "cuda")
        else:
            key = f"model.decoder.layers.{idx}.self_attn.R2"
            if key not in rotations:
                raise KeyError(f"OPT rotation file is missing {key}; retrain OPT rotations with R2 support.")
            r2 = rotations[key].cuda()
        layer.self_attn.bfp_R2 = RotationModule(r2, trainable=trainable)


def _add_opt_input_rotation_hook(model, compute_dtype):
    if hasattr(model, "_bfp_opt_input_rotation_hook"):
        return

    def hook(module, args, kwargs):
        if not hasattr(model, "bfp_R1"):
            return args, kwargs

        def rotate(x):
            return apply_rotation_right(x, model.bfp_R1.weight, compute_dtype)

        if len(args) > 0:
            return (rotate(args[0]),) + args[1:], kwargs
        kwargs["hidden_states"] = rotate(kwargs["hidden_states"])
        return args, kwargs

    model._bfp_opt_input_rotation_hook = model.model.decoder.layers[0].register_forward_pre_hook(
        hook, with_kwargs=True
    )


def setup_bfp_opt(model, cfg, trainable_rotations=False, rotation_path=None, compute_dtype=torch.float64):
    assert getattr(model.config, "model_type", None) == "opt", "setup_bfp_opt expects an OPT model"

    for param in model.parameters():
        param.requires_grad = False

    rotations = _load_rotation_state(rotation_path) if cfg.rotate and rotation_path is not None else None
    if cfg.rotate:
        _set_opt_rotations(model, cfg, trainable=trainable_rotations, rotation_path=rotation_path, rotations=rotations)

    _replace_opt_linears(model, cfg, compute_dtype=compute_dtype)
    if cfg.rotate:
        _add_opt_input_rotation_hook(model, compute_dtype=compute_dtype)
    return model


def save_opt_rotations(model, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {"R1": model.bfp_R1.weight.detach().cpu()}
    for idx, layer in enumerate(_opt_layers(model)):
        state[f"model.decoder.layers.{idx}.self_attn.R2"] = (
            layer.self_attn.bfp_R2.weight.detach().cpu()
        )
    path = output_dir / rotation_filename(cfg)
    torch.save(state, path)
    return path


def rotation_parameters(model):
    params = []
    if hasattr(model, "bfp_R1"):
        params.append(model.bfp_R1.weight)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers
    else:
        layers = []
    for layer in layers:
        if hasattr(layer.self_attn, "bfp_R2"):
            params.append(layer.self_attn.bfp_R2.weight)
    return params


def _hadamard_type(group_size):
    return "B" if group_size is not None and group_size > 0 else "F"


def rotation_suffix(cfg):
    suffix = (
        _hadamard_type(cfg.w_down_had_group_size)
        + _hadamard_type(cfg.qk_had_group_size)
    )
    block_size = getattr(cfg, "rotation_block_size", 0)
    if block_size and block_size > 0:
        suffix += f"_B{block_size}"
    return suffix


def rotation_filename(cfg):
    return f"R_{cfg.w_bits}_{cfg.a_bits}_{cfg.kv_bits}_{rotation_suffix(cfg)}.bin"


def save_rotations(model, output_dir, cfg):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {"R1": model.bfp_R1.weight.detach().cpu()}
    for idx, layer in enumerate(model.model.layers):
        state[f"model.layers.{idx}.self_attn.R2"] = (
            layer.self_attn.bfp_R2.weight.detach().cpu()
        )
    path = output_dir / rotation_filename(cfg)
    torch.save(state, path)
    return path


def load_llama_causal_lm(model_name, dtype, token=None, cfg=None):
    config = AutoConfig.from_pretrained(model_name, token=token)
    assert config.model_type == "llama", "bfp_refactor supports only LLaMA configs"
    if cfg is not None and (cfg.qk_matmul_bits < 16 or cfg.av_matmul_bits < 16):
        config._attn_implementation = "eager"
    clone_lm_head = False
    if getattr(config, "tie_word_embeddings", False):
        config.tie_word_embeddings = False
        clone_lm_head = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        token=token,
    )
    if clone_lm_head:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    return model



def load_opt_causal_lm(model_name, dtype, token=None):
    config = AutoConfig.from_pretrained(model_name, token=token)
    assert config.model_type == "opt", "load_opt_causal_lm expects an OPT config"
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        token=token,
    )


def load_opt_tokenizer(model_name, max_length, token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_llama_tokenizer(model_name, max_length, token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
