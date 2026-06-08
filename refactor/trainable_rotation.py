from __future__ import annotations

import math
import copy
import functools
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn

from refactor.attention_bfp import AttentionMatmulBFPConfig, add_attention_matmul_bfp_to_llama
from refactor.bfp import bfp_quant_dequant
from refactor.rotation import (
    apply_hadamard_to_last_dim,
    apply_headwise_hadamard_to_last_dim,
    resolve_rotation_block_size,
)


@dataclass(frozen=True)
class TrainableRotationConfig:
    w_bits: int = 4
    a_bits: int = 4
    kv_bits: int = 4
    bfp_group_size: int = 32
    rotation_block_size: int = 32
    rotation_init: str = "random_hadamard"
    rotation_seed: int = 0
    online_down_proj: bool = True
    online_o_proj: bool = True
    online_qk: bool = True
    online_had_group_size: int = 32
    w_down_had_group_size: int = 32
    qk_had_group_size: int = 32
    qk_matmul_bits: int = 4
    av_matmul_bits: int = 4
    qk_matmul_bfp_group_size: int = 32
    av_matmul_bfp_group_size: int = 32
    compute_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class TrainableRotationStats:
    wrapped_linears: int
    rotation_tensors: int
    qk_wrappers: int
    attention_matmul_wrappers: int
    block_size: Optional[int]


def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def resolve_hadamard_group_size(group_size: int, dim: int) -> Optional[int]:
    if group_size <= 0:
        return None
    return resolve_rotation_block_size(group_size, dim)


def _hadamard_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not _is_pow2(size):
        raise ValueError(f"Hadamard size must be a power of two, got {size}.")
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < size:
        h = torch.cat(
            (
                torch.cat((h, h), dim=1),
                torch.cat((h, -h), dim=1),
            ),
            dim=0,
        )
    return h / math.sqrt(size)


def _initial_rotation(
    size: int,
    *,
    init: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rotation = _hadamard_matrix(size, device=device, dtype=torch.float32)
    if init == "hadamard":
        return rotation
    if init != "random_hadamard":
        raise ValueError(f"Unsupported rotation_init: {init}.")
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (size,), device=device, dtype=torch.float32)
    signs = signs.mul_(2).sub_(1)
    return signs[:, None] * rotation


class RotationModule(nn.Module):
    def __init__(self, size: int, *, init: str, seed: int, device: torch.device):
        super().__init__()
        self.weight = nn.Parameter(_initial_rotation(size, init=init, seed=seed, device=device))


class BlockDiagRotationModule(nn.Module):
    def __init__(
        self,
        size: int,
        block_size: int,
        *,
        init: str,
        seed: int,
        device: torch.device,
    ):
        super().__init__()
        if size % block_size != 0:
            raise ValueError(f"size={size} must be divisible by block_size={block_size}.")
        self.blocks = nn.ParameterList(
            [
                nn.Parameter(
                    _initial_rotation(
                        block_size,
                        init=init,
                        seed=seed + idx,
                        device=device,
                    )
                )
                for idx in range(size // block_size)
            ]
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.stack(tuple(self.blocks), dim=0)


def make_rotation_module(
    size: int,
    block_size: Optional[int],
    *,
    init: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    if block_size is None:
        return RotationModule(size, init=init, seed=seed, device=device)
    return BlockDiagRotationModule(size, block_size, init=init, seed=seed, device=device)


def is_block_rotation(rotation: torch.Tensor) -> bool:
    return rotation.dim() == 3


def apply_rotation_right(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    transpose: bool = False,
) -> torch.Tensor:
    dtype = x.dtype
    if not is_block_rotation(rotation):
        matrix = rotation.transpose(-1, -2) if transpose else rotation
        return (x.to(compute_dtype) @ matrix.to(compute_dtype)).to(dtype)

    num_blocks, block_size, _ = rotation.shape
    if x.shape[-1] != num_blocks * block_size:
        raise ValueError(
            f"last dim {x.shape[-1]} does not match block rotation dim {num_blocks * block_size}."
        )
    matrix = rotation.transpose(-1, -2) if transpose else rotation
    blocks = x.reshape(*x.shape[:-1], num_blocks, block_size).to(compute_dtype)
    out = torch.einsum("...bi,bij->...bj", blocks, matrix.to(compute_dtype))
    return out.reshape_as(x).to(dtype)


def apply_rotation_left(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    transpose: bool = False,
) -> torch.Tensor:
    dtype = x.dtype
    if not is_block_rotation(rotation):
        matrix = rotation.transpose(-1, -2) if transpose else rotation
        return (matrix.to(compute_dtype) @ x.to(compute_dtype)).to(dtype)

    num_blocks, block_size, _ = rotation.shape
    if x.shape[0] != num_blocks * block_size:
        raise ValueError(
            f"first dim {x.shape[0]} does not match block rotation dim {num_blocks * block_size}."
        )
    matrix = rotation.transpose(-1, -2) if transpose else rotation
    blocks = x.reshape(num_blocks, block_size, *x.shape[1:]).to(compute_dtype)
    out = torch.einsum("bij,bj...->bi...", matrix.to(compute_dtype), blocks)
    return out.reshape_as(x).to(dtype)


def apply_head_rotation_to_weight(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    transpose: bool,
) -> torch.Tensor:
    if transpose:
        shape = weight.shape
        dim = rotation.shape[0] * rotation.shape[-1] if is_block_rotation(rotation) else rotation.shape[-1]
        temp = weight.reshape(-1, shape[-1] // dim, dim)
        return apply_rotation_right(
            temp,
            rotation,
            compute_dtype=compute_dtype,
        ).reshape(shape).to(weight.dtype)

    wt = weight.t()
    shape = wt.shape
    dim = rotation.shape[0] * rotation.shape[-1] if is_block_rotation(rotation) else rotation.shape[-1]
    temp = wt.reshape(-1, shape[-1] // dim, dim)
    return apply_rotation_right(temp, rotation, compute_dtype=compute_dtype).reshape(shape).t().to(weight.dtype)


class TrainableRotationLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        role: str,
        cfg: TrainableRotationConfig,
        model: nn.Module,
        r2_module: Optional[nn.Module] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        self.linear = linear
        self.role = role
        self.cfg = cfg
        object.__setattr__(self, "_model_ref", weakref.ref(model))
        object.__setattr__(self, "_r2_ref", weakref.ref(r2_module) if r2_module is not None else None)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _r1(self) -> torch.Tensor:
        model = self._model_ref()
        if model is None:
            raise RuntimeError("parent model reference is gone")
        return model.spinbfp_R1.weight

    def _r2(self) -> Optional[torch.Tensor]:
        r2_ref = getattr(self, "_r2_ref", None)
        r2_module = None if r2_ref is None else r2_ref()
        return None if r2_module is None else r2_module.weight

    def _effective_weight(self) -> torch.Tensor:
        weight = self.linear.weight
        r1 = self._r1()
        compute_dtype = self.cfg.compute_dtype

        if self.role in {"q_proj", "k_proj", "gate_proj", "up_proj", "lm_head"}:
            weight = apply_rotation_right(weight, r1, compute_dtype=compute_dtype)
        elif self.role == "v_proj":
            weight = apply_rotation_right(weight, r1, compute_dtype=compute_dtype)
            r2 = self._r2()
            if r2 is not None:
                weight = apply_head_rotation_to_weight(
                    weight,
                    r2,
                    compute_dtype=compute_dtype,
                    transpose=False,
                )
        elif self.role == "o_proj":
            weight = apply_rotation_left(weight, r1, compute_dtype=compute_dtype, transpose=True)
            r2 = self._r2()
            if r2 is not None:
                weight = apply_head_rotation_to_weight(
                    weight,
                    r2,
                    compute_dtype=compute_dtype,
                    transpose=True,
                )
            if self.cfg.online_o_proj:
                weight = self._apply_online_weight_hadamard(weight)
        elif self.role == "down_proj":
            weight = apply_rotation_left(weight, r1, compute_dtype=compute_dtype, transpose=True)
            if self.cfg.online_down_proj:
                weight = apply_hadamard_to_last_dim(weight, self.block_size)
        return weight

    def _apply_online_weight_hadamard(self, weight: torch.Tensor) -> torch.Tensor:
        if self.num_heads is None or self.head_dim is None:
            return apply_hadamard_to_last_dim(weight, self.block_size)
        return apply_headwise_hadamard_to_last_dim(
            weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=self.block_size,
        )

    def _quant_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.role == "lm_head":
            return x
        if self.role == "o_proj" and self.cfg.online_o_proj:
            if self.num_heads is None or self.head_dim is None:
                x = apply_hadamard_to_last_dim(x, self.block_size)
            else:
                x = apply_headwise_hadamard_to_last_dim(
                    x,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    block_size=self.block_size,
                )
        elif self.role == "down_proj" and self.cfg.online_down_proj:
            x = apply_hadamard_to_last_dim(x, self.block_size)
        return bfp_quant_dequant(x, self.cfg.a_bits, self.cfg.bfp_group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = self._quant_input(x)
        weight = self._effective_weight()
        if self.role != "lm_head":
            weight = bfp_quant_dequant(weight, self.cfg.w_bits, self.cfg.bfp_group_size)
        out = nn.functional.linear(x, weight, self.linear.bias).to(dtype)
        if self.role == "v_proj":
            out = bfp_quant_dequant(out, self.cfg.kv_bits, self.cfg.bfp_group_size)
        return out


def _copy_func_with_new_globals(func: Callable, globals_dict: dict) -> Callable:
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


class QKOnlineHadamardBFPWrapper(nn.Module):
    def __init__(
        self,
        func: Callable,
        cfg: TrainableRotationConfig,
        block_size: Optional[int],
    ):
        super().__init__()
        self.func = func
        self.cfg = cfg
        self.block_size = block_size

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        q = apply_hadamard_to_last_dim(q, self.block_size)
        k = apply_hadamard_to_last_dim(k, self.block_size)
        k = bfp_quant_dequant(k, self.cfg.kv_bits, self.cfg.bfp_group_size)
        return q, k


def add_qk_online_hadamard_bfp(
    attn: nn.Module,
    *,
    cfg: TrainableRotationConfig,
    block_size: Optional[int],
) -> bool:
    if hasattr(attn, "spinbfp_qk_online_hadamard_bfp_wrapper"):
        return False
    original = attn.forward.__func__
    method_globals = dict(original.__globals__)
    if "apply_rotary_pos_emb" not in method_globals:
        raise KeyError("attention forward globals do not contain apply_rotary_pos_emb.")
    wrapper = QKOnlineHadamardBFPWrapper(
        method_globals["apply_rotary_pos_emb"],
        cfg,
        block_size,
    )
    method_globals["apply_rotary_pos_emb"] = wrapper
    patched = _copy_func_with_new_globals(original, method_globals)
    setattr(attn, "forward", patched.__get__(attn))
    attn.spinbfp_qk_online_hadamard_bfp_wrapper = wrapper
    return True


def _add_input_rotation_hook(model: nn.Module, cfg: TrainableRotationConfig) -> None:
    if hasattr(model, "_spinbfp_input_rotation_hook"):
        return

    def hook(_module, args, kwargs):
        def rotate(x: torch.Tensor) -> torch.Tensor:
            return apply_rotation_right(
                x,
                model.spinbfp_R1.weight,
                compute_dtype=cfg.compute_dtype,
            )

        if args:
            return (rotate(args[0]), *args[1:]), kwargs
        kwargs["hidden_states"] = rotate(kwargs["hidden_states"])
        return args, kwargs

    model._spinbfp_input_rotation_hook = model.model.layers[0].register_forward_pre_hook(
        hook,
        with_kwargs=True,
    )


def setup_trainable_llama_rotations(
    model: nn.Module,
    cfg: TrainableRotationConfig,
) -> TrainableRotationStats:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a LLaMA causal LM with model.model.layers.")

    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    block_size = resolve_rotation_block_size(cfg.rotation_block_size, hidden_size)
    r2_block_size = resolve_rotation_block_size(cfg.rotation_block_size, head_dim)
    o_had_block_size = resolve_hadamard_group_size(cfg.online_had_group_size, head_dim)
    qk_had_block_size = resolve_hadamard_group_size(cfg.qk_had_group_size, head_dim)
    intermediate_size = getattr(model.config, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = model.model.layers[0].mlp.down_proj.in_features
    down_had_block_size = resolve_hadamard_group_size(
        cfg.w_down_had_group_size,
        intermediate_size,
    )

    model.spinbfp_R1 = make_rotation_module(
        hidden_size,
        block_size,
        init=cfg.rotation_init,
        seed=cfg.rotation_seed,
        device=device,
    )

    wrapped = 0
    qk_wrappers = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        attn.spinbfp_R2 = make_rotation_module(
            head_dim,
            r2_block_size,
            init=cfg.rotation_init,
            seed=cfg.rotation_seed + 10_000 + layer_idx,
            device=device,
        )
        r2 = attn.spinbfp_R2

        attn.q_proj = TrainableRotationLinear(attn.q_proj, "q_proj", cfg, model, r2)
        attn.k_proj = TrainableRotationLinear(attn.k_proj, "k_proj", cfg, model, r2)
        attn.v_proj = TrainableRotationLinear(attn.v_proj, "v_proj", cfg, model, r2)
        attn.o_proj = TrainableRotationLinear(
            attn.o_proj,
            "o_proj",
            cfg,
            model,
            r2,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=o_had_block_size,
        )
        mlp.up_proj = TrainableRotationLinear(mlp.up_proj, "up_proj", cfg, model)
        mlp.gate_proj = TrainableRotationLinear(mlp.gate_proj, "gate_proj", cfg, model)
        mlp.down_proj = TrainableRotationLinear(
            mlp.down_proj,
            "down_proj",
            cfg,
            model,
            block_size=down_had_block_size,
        )
        wrapped += 7

        if cfg.online_qk and add_qk_online_hadamard_bfp(
            attn,
            cfg=cfg,
            block_size=qk_had_block_size,
        ):
            qk_wrappers += 1

    model.lm_head = TrainableRotationLinear(model.lm_head, "lm_head", cfg, model)
    wrapped += 1
    _add_input_rotation_hook(model, cfg)
    attention_matmul_wrappers = add_attention_matmul_bfp_to_llama(
        model,
        AttentionMatmulBFPConfig(
            qk_bits=cfg.qk_matmul_bits,
            av_bits=cfg.av_matmul_bits,
            qk_group_size=cfg.qk_matmul_bfp_group_size,
            av_group_size=cfg.av_matmul_bfp_group_size,
        ),
    )
    return TrainableRotationStats(
        wrapped_linears=wrapped,
        rotation_tensors=len(rotation_parameters(model)),
        qk_wrappers=qk_wrappers,
        attention_matmul_wrappers=attention_matmul_wrappers,
        block_size=block_size,
    )


def rotation_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = list(model.spinbfp_R1.parameters()) if hasattr(model, "spinbfp_R1") else []
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "spinbfp_R2"):
            params.extend(layer.self_attn.spinbfp_R2.parameters())
    return params


def _hadamard_type(group_size: int | None) -> str:
    return "B" if group_size is not None and group_size > 0 else "F"


def rotation_suffix(cfg: TrainableRotationConfig) -> str:
    suffix = (
        _hadamard_type(cfg.w_down_had_group_size)
        + _hadamard_type(cfg.qk_had_group_size)
    )
    if cfg.rotation_block_size > 0:
        suffix += f"_B{cfg.rotation_block_size}"
    return suffix


def rotation_filename(cfg: TrainableRotationConfig) -> str:
    return f"R_{cfg.w_bits}_{cfg.a_bits}_{cfg.kv_bits}_{rotation_suffix(cfg)}.bin"


def rotation_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {"R1": model.spinbfp_R1.weight.detach().cpu()}
    for idx, layer in enumerate(model.model.layers):
        state[f"model.layers.{idx}.self_attn.R2"] = layer.self_attn.spinbfp_R2.weight.detach().cpu()
    return state


def save_rotation_state(model: nn.Module, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rotation_state_dict(model), path)
    return path
