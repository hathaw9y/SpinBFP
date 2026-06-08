from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except ImportError:  # pragma: no cover - depends on transformers version.
    LlamaRMSNorm = None


class RMSN(torch.nn.Module):
    """RMS normalization without a learned scale."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(input_dtype)


@dataclass(frozen=True)
class FuseRMSNormStats:
    centered_embeddings: int
    fused_linears: int
    replaced_norms: int


def _is_rmsnorm(module: torch.nn.Module) -> bool:
    if LlamaRMSNorm is not None and isinstance(module, LlamaRMSNorm):
        return True
    return module.__class__.__name__ == "LlamaRMSNorm"


def _norm_eps(module: torch.nn.Module) -> float:
    return float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-5)))


def _require_linear(module: torch.nn.Module) -> torch.nn.Linear:
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"Expected torch.nn.Linear, got {type(module).__name__}.")
    return module


@torch.no_grad()
def fuse_rmsnorm_into_linears(
    rmsnorm: torch.nn.Module,
    linears: Iterable[torch.nn.Linear],
) -> int:
    """Fold an RMSNorm scale into adjacent Linear input dimensions in-place."""
    if not hasattr(rmsnorm, "weight"):
        raise TypeError(f"{type(rmsnorm).__name__} has no learned RMSNorm weight.")

    gamma = rmsnorm.weight.detach().double()
    fused = 0
    for linear in linears:
        linear = _require_linear(linear)
        if linear.weight.shape[1] != gamma.numel():
            raise ValueError(
                "RMSNorm weight size must match Linear input features: "
                f"{gamma.numel()} != {linear.weight.shape[1]}."
            )

        dtype = linear.weight.dtype
        device = linear.weight.device
        weight = linear.weight.data.double()
        linear.weight.data = (weight * gamma.to(device=device)).to(dtype)
        fused += 1

        bias = getattr(rmsnorm, "bias", None)
        if bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64, device=device)
                )
            linear.bias.data = (
                linear.bias.data.double()
                + torch.matmul(weight, bias.detach().double().to(device=device))
            ).to(dtype)

    return fused


@torch.no_grad()
def center_token_embeddings(model: torch.nn.Module) -> int:
    """Center token embedding rows as used by QuaRot before rotation."""
    embed_tokens = model.model.embed_tokens
    dtype = embed_tokens.weight.dtype
    weight = embed_tokens.weight.data.double()
    embed_tokens.weight.data = (weight - weight.mean(dim=-1, keepdim=True)).to(dtype)
    return 1


def replace_rmsnorm_with_rmsn(module: torch.nn.Module, hidden_size: int) -> int:
    replaced = 0
    for name, child in module.named_children():
        if _is_rmsnorm(child):
            setattr(module, name, RMSN(hidden_size, eps=_norm_eps(child)))
            replaced += 1
        else:
            replaced += replace_rmsnorm_with_rmsn(child, hidden_size)
    return replaced


@torch.no_grad()
def fuse_llama_rmsnorm(
    model: torch.nn.Module,
    *,
    center_embeddings: bool = True,
    replace_norms: bool = True,
) -> FuseRMSNormStats:
    """Fuse LLaMA RMSNorm weights into neighboring Linear modules."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a LLaMA causal LM with model.model.layers.")
    if not hasattr(model.model, "norm") or not hasattr(model, "lm_head"):
        raise TypeError("Expected final model.model.norm and model.lm_head modules.")

    centered = center_token_embeddings(model) if center_embeddings else 0
    fused = 0

    for layer in model.model.layers:
        fused += fuse_rmsnorm_into_linears(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fused += fuse_rmsnorm_into_linears(
            layer.post_attention_layernorm,
            [layer.mlp.up_proj, layer.mlp.gate_proj],
        )

    fused += fuse_rmsnorm_into_linears(model.model.norm, [model.lm_head])

    if replace_norms:
        replaced = replace_rmsnorm_with_rmsn(model, model.config.hidden_size)
    else:
        for layer in model.model.layers:
            layer.input_layernorm.weight.data = torch.ones_like(layer.input_layernorm.weight)
            layer.post_attention_layernorm.weight.data = torch.ones_like(
                layer.post_attention_layernorm.weight
            )
        model.model.norm.weight.data = torch.ones_like(model.model.norm.weight)
        replaced = 0

    return FuseRMSNormStats(
        centered_embeddings=centered,
        fused_linears=fused,
        replaced_norms=replaced,
    )
