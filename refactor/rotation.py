from __future__ import annotations

import copy
import functools
import math
import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from fast_hadamard_transform import hadamard_transform
from utils import hadamard_utils


@dataclass(frozen=True)
class RotationStats:
    offline_input_linears: int
    offline_output_linears: int
    online_linears: int
    qk_wrappers: int
    block_size: Optional[int]
    seed: int


def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def resolve_rotation_block_size(rotation_block_size: int, hidden_size: int) -> Optional[int]:
    if rotation_block_size == 0:
        return None
    if rotation_block_size < 0:
        raise ValueError("rotation_block_size must be 0 for full rotation or a positive power-of-two block size.")
    if not _is_pow2(rotation_block_size):
        raise ValueError(f"rotation_block_size must be a power of two, got {rotation_block_size}.")
    if hidden_size % rotation_block_size != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by rotation_block_size={rotation_block_size}."
        )
    return rotation_block_size


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _fork_rng_for_device(device: torch.device):
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    return torch.random.fork_rng(devices=devices)


def _random_signs(dim: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with _fork_rng_for_device(device):
        torch.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), device=device, dtype=torch.int8)
    return signs.to(dtype=dtype).mul_(2).sub_(1)


def _hadamard_last_dim(x: torch.Tensor, block_size: Optional[int]) -> torch.Tensor:
    dim = x.shape[-1]
    if block_size is None:
        if _is_pow2(dim):
            return hadamard_transform(x.contiguous()) / math.sqrt(dim)
        try:
            had_k, k = hadamard_utils.get_hadK(dim)
        except AssertionError as exc:
            raise ValueError(
                f"Full Hadamard last dim {dim} is not supported by utils.hadamard_utils.get_hadK."
            ) from exc
        return hadamard_utils.matmul_hadU_cuda(x.contiguous(), had_k, k)
    if not _is_pow2(block_size):
        raise ValueError(f"Hadamard block size must be a power of two, got {block_size}.")
    if dim % block_size != 0:
        raise ValueError(f"last dim {dim} must be divisible by block size {block_size}.")
    shape = x.shape
    x = x.reshape(-1, block_size)
    x = hadamard_transform(x.contiguous()) / math.sqrt(block_size)
    return x.reshape(shape)


def _random_hadamard_last_dim(
    x: torch.Tensor,
    *,
    block_size: Optional[int],
    seed: int,
) -> torch.Tensor:
    dim = x.shape[-1]
    dtype = x.dtype
    compute_dtype = _compute_dtype(dtype)
    x = x.to(dtype=compute_dtype)

    signs = _random_signs(dim, seed=seed, device=x.device, dtype=compute_dtype)
    if block_size is not None:
        signs = signs.reshape(dim // block_size, block_size)
        x = x.reshape(*x.shape[:-1], dim // block_size, block_size)
        return _hadamard_last_dim(x * signs, block_size).reshape(*x.shape[:-2], dim).to(dtype)
    return _hadamard_last_dim(x * signs, None).to(dtype)


@torch.no_grad()
def absorb_random_hadamard_on_input(
    module: torch.nn.Module,
    *,
    block_size: Optional[int],
    seed: int,
) -> None:
    if not hasattr(module, "weight"):
        raise TypeError(f"Expected a module with weight, got {type(module).__name__}.")
    module.weight.data = _random_hadamard_last_dim(
        module.weight.data,
        block_size=block_size,
        seed=seed,
    )


@torch.no_grad()
def absorb_random_hadamard_on_output(
    linear: torch.nn.Module,
    *,
    block_size: Optional[int],
    seed: int,
) -> None:
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError(f"Expected torch.nn.Linear, got {type(linear).__name__}.")
    linear.weight.data = _random_hadamard_last_dim(
        linear.weight.data.t(),
        block_size=block_size,
        seed=seed,
    ).t()
    if linear.bias is not None:
        linear.bias.data = _random_hadamard_last_dim(
            linear.bias.data,
            block_size=block_size,
            seed=seed,
        )


def apply_hadamard_to_last_dim(x: torch.Tensor, block_size: Optional[int]) -> torch.Tensor:
    dtype = x.dtype
    return _hadamard_last_dim(x.to(dtype=_compute_dtype(dtype)), block_size).to(dtype)


def apply_headwise_hadamard_to_last_dim(
    x: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    block_size: Optional[int],
) -> torch.Tensor:
    if x.shape[-1] != num_heads * head_dim:
        raise ValueError(
            "last dim must equal num_heads * head_dim: "
            f"{x.shape[-1]} != {num_heads} * {head_dim}."
        )
    dtype = x.dtype
    compute_dtype = _compute_dtype(dtype)
    x = x.to(dtype=compute_dtype)
    if block_size is None:
        x = x.reshape(*x.shape[:-1], num_heads, head_dim)
        return _hadamard_last_dim(x, None).reshape(*x.shape[:-2], num_heads * head_dim).to(dtype)
    if head_dim % block_size != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by block_size={block_size}.")
    x = x.reshape(*x.shape[:-1], num_heads, head_dim // block_size, block_size)
    return _hadamard_last_dim(x, block_size).reshape(*x.shape[:-3], num_heads * head_dim).to(dtype)


def _online_hadamard_pre_hook(block_size: Optional[int]):
    def hook(_module, inputs):
        if not inputs:
            return inputs
        return (apply_hadamard_to_last_dim(inputs[0], block_size), *inputs[1:])

    return hook


def _headwise_online_hadamard_pre_hook(num_heads: int, head_dim: int, block_size: Optional[int]):
    def hook(_module, inputs):
        if not inputs:
            return inputs
        return (
            apply_headwise_hadamard_to_last_dim(
                inputs[0],
                num_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
            ),
            *inputs[1:],
        )

    return hook


@torch.no_grad()
def add_online_hadamard_to_linear(linear: torch.nn.Linear, *, block_size: Optional[int]) -> None:
    absorb_hadamard_on_input(linear, block_size=block_size)
    if hasattr(linear, "_spinbfp_online_hadamard_handle"):
        linear._spinbfp_online_hadamard_handle.remove()
    linear._spinbfp_online_hadamard_handle = linear.register_forward_pre_hook(
        _online_hadamard_pre_hook(block_size)
    )
    linear.spinbfp_online_hadamard_block_size = block_size


@torch.no_grad()
def add_headwise_online_hadamard_to_linear(
    linear: torch.nn.Linear,
    *,
    num_heads: int,
    head_dim: int,
    block_size: Optional[int],
) -> None:
    linear.weight.data = apply_headwise_hadamard_to_last_dim(
        linear.weight.data,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
    )
    if hasattr(linear, "_spinbfp_online_hadamard_handle"):
        linear._spinbfp_online_hadamard_handle.remove()
    linear._spinbfp_online_hadamard_handle = linear.register_forward_pre_hook(
        _headwise_online_hadamard_pre_hook(num_heads, head_dim, block_size)
    )
    linear.spinbfp_online_hadamard_block_size = block_size
    linear.spinbfp_online_hadamard_num_heads = num_heads
    linear.spinbfp_online_hadamard_head_dim = head_dim


@torch.no_grad()
def absorb_hadamard_on_input(linear: torch.nn.Linear, *, block_size: Optional[int]) -> None:
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError(f"Expected torch.nn.Linear, got {type(linear).__name__}.")
    dtype = linear.weight.dtype
    linear.weight.data = apply_hadamard_to_last_dim(linear.weight.data, block_size).to(dtype)


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


class QKOnlineHadamardWrapper(torch.nn.Module):
    def __init__(self, func: Callable, block_size: Optional[int]):
        super().__init__()
        self.func = func
        self.block_size = block_size

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        q = apply_hadamard_to_last_dim(q, self.block_size)
        k = apply_hadamard_to_last_dim(k, self.block_size)
        return q, k


def add_qk_online_hadamard(attn: torch.nn.Module, *, block_size: Optional[int]) -> bool:
    if hasattr(attn, "spinbfp_qk_online_hadamard_wrapper"):
        return False
    original = attn.forward.__func__
    method_globals = dict(original.__globals__)
    if "apply_rotary_pos_emb" not in method_globals:
        raise KeyError("attention forward globals do not contain apply_rotary_pos_emb.")
    wrapper = QKOnlineHadamardWrapper(method_globals["apply_rotary_pos_emb"], block_size)
    method_globals["apply_rotary_pos_emb"] = wrapper
    patched = _copy_func_with_new_globals(original, method_globals)
    setattr(attn, "forward", patched.__get__(attn))
    attn.spinbfp_qk_online_hadamard_wrapper = wrapper
    return True


@torch.no_grad()
def apply_llama_random_hadamard_rotations(
    model: torch.nn.Module,
    *,
    rotation_block_size: int = 32,
    seed: int = 0,
    online_down_proj: bool = True,
    online_o_proj: bool = True,
    online_qk: bool = True,
) -> RotationStats:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a LLaMA causal LM with model.model.layers.")

    hidden_size = model.config.hidden_size
    block_size = resolve_rotation_block_size(rotation_block_size, hidden_size)
    num_heads = model.config.num_attention_heads
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    head_dim = hidden_size // num_heads

    input_count = 0
    output_count = 0
    online_count = 0
    qk_count = 0

    absorb_random_hadamard_on_input(model.model.embed_tokens, block_size=block_size, seed=seed)
    input_count += 1

    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp

        for linear in (attn.q_proj, attn.k_proj, attn.v_proj, mlp.up_proj, mlp.gate_proj):
            absorb_random_hadamard_on_input(linear, block_size=block_size, seed=seed)
            input_count += 1

        for linear in (attn.o_proj, mlp.down_proj):
            absorb_random_hadamard_on_output(linear, block_size=block_size, seed=seed)
            output_count += 1

        if online_o_proj:
            if block_size is None:
                add_online_hadamard_to_linear(attn.o_proj, block_size=None)
            else:
                add_headwise_online_hadamard_to_linear(
                    attn.o_proj,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                )
            online_count += 1

        if online_down_proj:
            add_online_hadamard_to_linear(mlp.down_proj, block_size=block_size)
            online_count += 1

        if online_qk and add_qk_online_hadamard(attn, block_size=block_size):
            qk_count += 1

    absorb_random_hadamard_on_input(model.lm_head, block_size=block_size, seed=seed)
    input_count += 1

    return RotationStats(
        offline_input_linears=input_count,
        offline_output_linears=output_count,
        online_linears=online_count,
        qk_wrappers=qk_count,
        block_size=block_size,
        seed=seed,
    )
