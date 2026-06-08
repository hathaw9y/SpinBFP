from __future__ import annotations

import torch
import torch.nn.functional as F


class BFPQuantDequant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int, group_size: int):
        if bits >= 16:
            return x
        if bits < 2:
            raise ValueError(f"BFP bits must be >= 2, got {bits}.")
        if group_size <= 0:
            raise ValueError(f"BFP group_size must be positive, got {group_size}.")
        dtype = x.dtype
        shape = x.shape
        pad = (group_size - shape[-1] % group_size) % group_size
        if pad:
            x = F.pad(x, (0, pad))
        padded_shape = x.shape
        grouped = x.reshape(*padded_shape[:-1], padded_shape[-1] // group_size, group_size)
        absmax = grouped.abs().amax(dim=-1, keepdim=True)
        nonzero = absmax > 0
        safe_absmax = torch.where(nonzero, absmax.float(), torch.ones_like(absmax.float()))
        scale = torch.pow(2.0, torch.floor(torch.log2(safe_absmax)))
        scale = torch.where(nonzero, scale, torch.ones_like(scale)).to(grouped.dtype)

        denom = 2 ** (bits - 1)
        minq = -denom
        maxq = denom - 1
        q = torch.clamp(torch.round(grouped / scale * denom), minq, maxq)
        out = (scale * (q / denom)).reshape(padded_shape)
        if pad:
            out = out[..., : shape[-1]]
        return out.reshape(shape).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def bfp_quant_dequant(x: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    return BFPQuantDequant.apply(x, bits, group_size)
