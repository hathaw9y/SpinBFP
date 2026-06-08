from __future__ import annotations

import copy
import functools
import types
from dataclasses import dataclass
from typing import Callable

import torch

from refactor.bfp import bfp_quant_dequant


@dataclass(frozen=True)
class AttentionMatmulBFPConfig:
    qk_bits: int
    av_bits: int
    qk_group_size: int = 32
    av_group_size: int = 32


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


class AttentionMatmulBFPTorchProxy:
    """Proxy torch.matmul inside LLaMA attention forward for QK/AV BFP operands."""

    def __init__(self, torch_module, cfg: AttentionMatmulBFPConfig):
        self._torch = torch_module
        self.cfg = cfg
        self.matmul_count = 0

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def matmul(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        matmul_idx = self.matmul_count % 2
        self.matmul_count += 1

        if matmul_idx == 0 and self.cfg.qk_bits < 16:
            left = bfp_quant_dequant(left, self.cfg.qk_bits, self.cfg.qk_group_size)
            right = bfp_quant_dequant(
                right.transpose(-2, -1),
                self.cfg.qk_bits,
                self.cfg.qk_group_size,
            ).transpose(-2, -1)
        elif matmul_idx == 1 and self.cfg.av_bits < 16:
            left = bfp_quant_dequant(left, self.cfg.av_bits, self.cfg.av_group_size)
            right = bfp_quant_dequant(right, self.cfg.av_bits, self.cfg.av_group_size)

        return self._torch.matmul(left, right)


def add_attention_matmul_bfp_to_llama(
    model: torch.nn.Module,
    cfg: AttentionMatmulBFPConfig,
) -> int:
    if cfg.qk_bits >= 16 and cfg.av_bits >= 16:
        return 0

    wrapped = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "spinbfp_attention_matmul_bfp_proxy"):
            continue

        original = attn.forward.__func__
        method_globals = dict(original.__globals__)
        if "torch" not in method_globals:
            raise KeyError("attention forward globals do not contain torch.")

        proxy = AttentionMatmulBFPTorchProxy(method_globals["torch"], cfg)
        method_globals["torch"] = proxy
        patched = _copy_func_with_new_globals(original, method_globals)
        setattr(attn, "forward", patched.__get__(attn))
        attn.spinbfp_attention_matmul_bfp_proxy = proxy
        wrapped += 1

    return wrapped

