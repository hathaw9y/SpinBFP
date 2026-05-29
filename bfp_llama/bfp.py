import torch


BFP_EXPONENT_ROUNDING = "floor"


def set_bfp_exponent_rounding(mode):
    if mode not in {"floor", "ceil"}:
        raise ValueError(f"Unsupported BFP exponent rounding mode: {mode}")
    global BFP_EXPONENT_ROUNDING
    BFP_EXPONENT_ROUNDING = mode


def _round_log2(x):
    log2_x = torch.log2(x)
    if BFP_EXPONENT_ROUNDING == "ceil":
        return torch.ceil(log2_x)
    return torch.floor(log2_x)


def _mantissa_denominator(maxq):
    if BFP_EXPONENT_ROUNDING == "ceil":
        return maxq + 1
    return (maxq + 1) / 2


def _scale_from_absmax(xmax):
    xmax_f = xmax.float()
    nonzero = xmax_f > 0
    safe = torch.where(nonzero, xmax_f, torch.ones_like(xmax_f))
    scale = torch.pow(2.0, _round_log2(safe))
    scale = torch.where(nonzero, scale, torch.ones_like(scale))
    return scale.to(dtype=xmax.dtype)


def _bfp_quant_dequant_impl(x, bits, group_size, clip_ratio):
    if bits >= 16:
        return x
    assert bits >= 2, "BFP needs at least 2 mantissa bits"

    x_dtype = x.dtype
    maxq = (2**bits) - 1
    denom = _mantissa_denominator(maxq)

    if group_size is not None and group_size > 0:
        assert (
            x.shape[-1] % group_size == 0
        ), f"last dim {x.shape[-1]} must be divisible by BFP group size {group_size}"
        shape = x.shape
        grouped = x.reshape(*shape[:-1], shape[-1] // group_size, group_size)
        absmax = torch.amax(torch.abs(grouped), dim=-1, keepdim=True) * clip_ratio
        scale = _scale_from_absmax(absmax)
        q = torch.clamp(torch.round(torch.abs(grouped) / scale * denom), 0, maxq)
        q = q * torch.sign(grouped)
        return (scale * (q / denom)).reshape(shape).to(x_dtype)

    flat = x.reshape(-1, x.shape[-1])
    absmax = torch.amax(torch.abs(flat), dim=-1, keepdim=True) * clip_ratio
    scale = _scale_from_absmax(absmax)
    q = torch.clamp(torch.round(torch.abs(flat) / scale * denom), 0, maxq)
    q = q * torch.sign(flat)
    return (scale * (q / denom)).reshape_as(x).to(x_dtype)


class BfpQuantDequant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, group_size, clip_ratio):
        return _bfp_quant_dequant_impl(x, bits, group_size, clip_ratio)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


def bfp_quant_dequant(x, bits, group_size=32, clip_ratio=1.0):
    return BfpQuantDequant.apply(x, bits, group_size, clip_ratio)
