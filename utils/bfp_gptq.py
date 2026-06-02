from __future__ import annotations

import os
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
    if X_work.shape[1] % group_size != 0:
        raise ValueError(
            f"X columns {X_work.shape[1]} must be divisible by group_size {group_size}."
        )
    x_groups = X_work.reshape(-1, X_work.shape[1] // group_size, group_size)
    H_blocks = 2.0 * torch.einsum("nkg,nkh->kgh", x_groups, x_groups)
    return bfp_gptq_from_block_hessians(
        W_work,
        H_blocks,
        bits=bits,
        group_size=group_size,
        damp_pct=damp_pct,
        clip_ratio=clip_ratio,
        reorder=reorder,
    )


def bfp_gptq_from_block_hessians(
    W: torch.Tensor,
    H_blocks: torch.Tensor,
    bits: int = 4,
    group_size: int = 32,
    damp_pct: float = 0.01,
    clip_ratio: float = 1.0,
    reorder: bool = True,
) -> dict:
    """
    BFP-GPTQ using one Hessian block per BFP group.

    H_blocks has shape (in_features / group_size, group_size, group_size). This
    keeps the correction local to the same columns that share a BFP scale.
    """
    if W.dim() != 2:
        raise ValueError(f"W must be 2D, got shape {tuple(W.shape)}.")
    if W.shape[1] % group_size != 0:
        raise ValueError(
            f"W columns {W.shape[1]} must be divisible by group_size {group_size}."
        )
    expected = (W.shape[1] // group_size, group_size, group_size)
    if tuple(H_blocks.shape) != expected:
        raise ValueError(f"H_blocks must have shape {expected}, got {tuple(H_blocks.shape)}.")

    W_work = W.float()
    W_quant = torch.empty_like(W_work)
    perms = []
    scales = []

    for group_idx, start in enumerate(range(0, W.shape[1], group_size)):
        end = start + group_size
        W_block = W_work[:, start:end]
        H_block = H_blocks[group_idx].to(device=W.device, dtype=torch.float32)

        if reorder:
            local_order = torch.argsort(H_block.diag(), descending=True)
            inv_order = torch.argsort(local_order)
            W_gptq = W_block[:, local_order]
            H_gptq = H_block[local_order][:, local_order]
        else:
            local_order = torch.arange(group_size, device=W.device)
            inv_order = local_order
            W_gptq = W_block
            H_gptq = H_block

        group_scale = _bfp_group_scales(W_gptq, bits, group_size, clip_ratio)[:, 0, :]

        def quantize_col(col: torch.Tensor, _col_idx: int) -> torch.Tensor:
            return _quantize_bfp_column_fixed_scale(col, group_scale, bits)

        Wq_gptq = gptq(W_gptq, H_gptq, quantize_col, damp_pct=damp_pct)
        W_quant[:, start:end] = Wq_gptq[:, inv_order]
        perms.append(local_order + start)
        scales.append(group_scale.unsqueeze(1))

    perm = torch.cat(perms)
    return {
        "W_quant": W_quant.to(torch.float32),
        "H_blocks": H_blocks.to(torch.float32),
        "scales": torch.cat(scales, dim=1).to(torch.float32),
        "permutation": perm,
        "bits": bits,
        "group_size": group_size,
        "clip_ratio": clip_ratio,
        "reorder": reorder,
    }


def block_hessian_weighted_error(
    W: torch.Tensor,
    W_quant: torch.Tensor,
    H_blocks: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if W.shape != W_quant.shape:
        raise ValueError(f"W and W_quant shapes must match, got {tuple(W.shape)} and {tuple(W_quant.shape)}.")
    if W.dim() != 2:
        raise ValueError(f"W must be 2D, got shape {tuple(W.shape)}.")
    if W.shape[1] % group_size != 0:
        raise ValueError(
            f"W columns {W.shape[1]} must be divisible by group_size {group_size}."
        )
    expected = (W.shape[1] // group_size, group_size, group_size)
    if tuple(H_blocks.shape) != expected:
        raise ValueError(f"H_blocks must have shape {expected}, got {tuple(H_blocks.shape)}.")

    total = torch.zeros((), device=W.device, dtype=torch.float32)
    diff = W_quant.float() - W.float()
    for group_idx, start in enumerate(range(0, W.shape[1], group_size)):
        end = start + group_size
        group_diff = diff[:, start:end]
        H = H_blocks[group_idx].to(device=W.device, dtype=torch.float32)
        total = total + torch.einsum("og,gh,oh->", group_diff, H, group_diff)
    return total


def apply_bfp_gptq_weights(model, state_or_path, strict: bool = True):
    if isinstance(state_or_path, (str, bytes, os.PathLike)):
        state = torch.load(state_or_path, map_location="cpu")
    else:
        state = state_or_path
    weights = state.get("weights", state)
    modules = dict(model.named_modules())
    missing = []
    loaded = 0
    for name, weight in weights.items():
        module = modules.get(name)
        if module is None:
            missing.append(name)
            continue
        target_shape = getattr(module, "weight", None)
        if target_shape is not None and tuple(module.weight.shape) != tuple(weight.shape):
            raise ValueError(
                f"BFP-GPTQ weight shape mismatch for {name}: "
                f"{tuple(weight.shape)} != {tuple(module.weight.shape)}"
            )
        tensor = weight.detach().to(dtype=torch.float32)
        if hasattr(module, "bfp_gptq_weight"):
            module.bfp_gptq_weight.data.copy_(tensor.to(module.bfp_gptq_weight.device))
        else:
            module.register_buffer("bfp_gptq_weight", tensor)
        loaded += 1
    if strict and missing:
        raise KeyError(f"Missing modules for BFP-GPTQ weights: {missing[:5]}")
    return loaded


def bfp_gptq_weight_state(model) -> dict:
    weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "bfp_gptq_weight"):
            weights[name] = module.bfp_gptq_weight.detach().cpu()
    return weights
