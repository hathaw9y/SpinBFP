from __future__ import annotations

from typing import Callable

import torch


def quantize_bfp(
    x: torch.Tensor,
    bits: int,
    group_size: int = 32,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Fake-quantize with the repo's existing BFP semantics."""
    if bits >= 16:
        return x.float()
    if bits < 2:
        raise ValueError(f"BFP quantization needs at least 2 bits, got {bits}.")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} must be divisible by group_size {group_size}."
        )

    shape = x.shape
    grouped = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True) * clip_ratio
    nonzero = absmax > 0
    safe = torch.where(nonzero, absmax, torch.ones_like(absmax))
    scale = torch.pow(2.0, torch.floor(torch.log2(safe)))
    scale = torch.where(nonzero, scale, torch.ones_like(scale))

    maxq = (2**bits) - 1
    denom = (maxq + 1) / 2
    q = torch.clamp(torch.round(grouped.abs() / scale * denom), 0, maxq)
    q = q * torch.sign(grouped)
    return (scale * (q / denom)).reshape(shape).to(torch.float32)


def _bfp_group_scales(
    W: torch.Tensor,
    bits: int,
    group_size: int,
    clip_ratio: float,
) -> torch.Tensor:
    if bits >= 16:
        return torch.ones(
            W.shape[0], W.shape[1] // group_size, 1, device=W.device, dtype=torch.float32
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if W.shape[-1] % group_size != 0:
        raise ValueError(
            f"W columns {W.shape[-1]} must be divisible by group_size {group_size}."
        )

    grouped = W.float().reshape(W.shape[0], W.shape[1] // group_size, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True) * clip_ratio
    nonzero = absmax > 0
    safe = torch.where(nonzero, absmax, torch.ones_like(absmax))
    scale = torch.pow(2.0, torch.floor(torch.log2(safe)))
    return torch.where(nonzero, scale, torch.ones_like(scale))


def _quantize_bfp_column_fixed_scale(
    col: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    if bits >= 16:
        return col.float()
    maxq = (2**bits) - 1
    denom = (maxq + 1) / 2
    q = torch.clamp(torch.round(col.abs().float() / scale * denom), 0, maxq)
    q = q * torch.sign(col.float())
    return scale * (q / denom)


def cholesky_inverse_stable(H: torch.Tensor, damp_pct: float = 0.01) -> torch.Tensor:
    H = 0.5 * (H.float() + H.float().t())
    diag = torch.diag(H)
    damp = damp_pct * torch.mean(diag).clamp(min=torch.finfo(torch.float32).tiny)
    eye = torch.eye(H.shape[0], device=H.device, dtype=torch.float32)
    for _ in range(8):
        try:
            L = torch.linalg.cholesky(H + damp * eye)
            return torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            damp = damp * 10
    return torch.linalg.pinv(H + damp * eye)


def gptq(
    W: torch.Tensor,
    H: torch.Tensor,
    quantize_fn: Callable[[torch.Tensor, int], torch.Tensor],
    damp_pct: float = 0.01,
) -> torch.Tensor:
    """Column-wise GPTQ with inverse-Hessian error propagation."""
    if W.dim() != 2:
        raise ValueError(f"W must be 2D, got shape {tuple(W.shape)}.")
    if H.shape != (W.shape[1], W.shape[1]):
        raise ValueError(f"H must have shape {(W.shape[1], W.shape[1])}, got {tuple(H.shape)}.")

    W_work = W.float().clone()
    W_quant = torch.empty_like(W_work)
    H_inv = cholesky_inverse_stable(H, damp_pct)
    eps = torch.finfo(torch.float32).tiny

    for col in range(W_work.shape[1]):
        q_col = quantize_fn(W_work[:, col : col + 1], col).reshape(-1)
        W_quant[:, col] = q_col
        err = (W_work[:, col] - q_col) / H_inv[col, col].clamp(min=eps)
        if col + 1 < W_work.shape[1]:
            W_work[:, col + 1 :] -= err.unsqueeze(1) @ H_inv[col, col + 1 :].unsqueeze(0)
    return W_quant


def static_act_reorder(
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort columns by descending Hessian diagonal inside each BFP group."""
    if W.shape[1] % group_size != 0:
        raise ValueError(
            f"W columns {W.shape[1]} must be divisible by group_size {group_size}."
        )
    hdiag = H.diag() if H.dim() == 2 else H
    if hdiag.numel() != W.shape[1]:
        raise ValueError(f"H diagonal length {hdiag.numel()} must equal W columns {W.shape[1]}.")

    perm = []
    for start in range(0, W.shape[1], group_size):
        local = torch.argsort(hdiag[start : start + group_size], descending=True)
        perm.append(local + start)
    perm = torch.cat(perm).to(W.device)
    return W[:, perm], perm


def bfp_gptq(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    group_size: int = 32,
    damp_pct: float = 0.01,
    clip_ratio: float = 1.0,
    reorder: bool = True,
) -> dict:
    """
    GPTQ PTQ using this repo's existing BFP quantizer.

    W is shaped (out_features, in_features). X is calibration input shaped
    (n_samples, in_features). No rotation is applied.
    """
    if W.dim() != 2 or X.dim() != 2:
        raise ValueError("W and X must be 2D tensors.")
    if W.shape[1] != X.shape[1]:
        raise ValueError(f"W columns {W.shape[1]} must equal X dim {X.shape[1]}.")

    W_work = W.float()
    X_work = X.float()
    H = 2.0 * (X_work.t() @ X_work)

    if reorder:
        W_gptq, perm = static_act_reorder(W_work, H, group_size)
        H_gptq = H[perm][:, perm]
        inv_perm = torch.argsort(perm)
    else:
        W_gptq = W_work
        H_gptq = H
        perm = torch.arange(W.shape[1], device=W.device)
        inv_perm = perm

    scales = _bfp_group_scales(W_gptq, bits, group_size, clip_ratio)

    def quantize_col(col: torch.Tensor, col_idx: int) -> torch.Tensor:
        group_idx = col_idx // group_size
        return _quantize_bfp_column_fixed_scale(
            col,
            scales[:, group_idx, :],
            bits,
        )

    Wq_gptq = gptq(W_gptq, H_gptq, quantize_col, damp_pct=damp_pct)
    Wq = Wq_gptq[:, inv_perm]
    return {
        "W_quant": Wq.to(torch.float32),
        "H": H.to(torch.float32),
        "scales": scales.to(torch.float32),
        "permutation": perm,
        "bits": bits,
        "group_size": group_size,
        "clip_ratio": clip_ratio,
        "reorder": reorder,
    }
