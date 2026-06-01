# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import transformers

from train_utils.quant_linear import QuantizeLinear
from utils import hadamard_utils
from utils.utils import HadamardTransform


BFP_EXPONENT_ROUNDING = "floor"


def set_bfp_exponent_rounding(mode):
    if mode not in {"floor", "ceil"}:
        raise ValueError(f"Unsupported BFP exponent rounding mode: {mode}")
    global BFP_EXPONENT_ROUNDING
    BFP_EXPONENT_ROUNDING = mode


def _round_bfp_log2(x):
    log2_x = torch.log2(x)
    if BFP_EXPONENT_ROUNDING == "ceil":
        return torch.ceil(log2_x)
    return torch.floor(log2_x)


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def bfp_mantissa_denominator(maxq):
    if BFP_EXPONENT_ROUNDING == "ceil":
        return maxq + 1
    return (maxq + 1) / 2


def bfp_scale_from_absmax(xmax):
    xmax_float = xmax.float()
    nonzero = xmax_float > 0
    safe_xmax = torch.where(nonzero, xmax_float, torch.ones_like(xmax_float))
    scale = torch.pow(2.0, _round_bfp_log2(safe_xmax))
    scale = torch.where(nonzero, scale, torch.ones_like(scale))
    return scale.to(dtype=xmax.dtype)


def bfp_quant(x, scale, maxq):
    scale = scale.to(x.device)
    maxq = maxq.to(x.device)
    mantissa_denominator = bfp_mantissa_denominator(maxq)
    q = torch.clamp(
        torch.round(torch.abs(x) / scale * mantissa_denominator), 0, maxq
    )
    q = q * torch.sign(x)
    return q, scale


def bfp_dequant(q, scale, maxq):
    maxq = maxq.to(q.device)
    mantissa_denominator = bfp_mantissa_denominator(maxq)
    return scale * (q / mantissa_denominator)


def bfp_quant_dequant(x, scale, maxq):
    q, scale = bfp_quant(x, scale, maxq)
    return bfp_dequant(q, scale, maxq)


def bfp_quant_dequant_weight_groupwise(x, scale, maxq, groupsize):
    init_shape = x.shape
    grouped = x.reshape(x.shape[-2], x.shape[-1] // groupsize, groupsize)
    return bfp_quant_dequant(grouped, scale, maxq).reshape(init_shape)


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        maxq = maxq.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        maxq = maxq.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class BFPSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        return bfp_quant_dequant(x, scale, maxq)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class BFPGroupwiseWeightSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq, groupsize):
        return bfp_quant_dequant_weight_groupwise(x, scale, maxq, groupsize)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16
        self.quant_method = "int"

    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.quant_method == "bfp":
            return BFPSTEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.quant_method == "bfp":
            return bfp_quant(x, self.scale, self.maxq)
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self,
        bits: int,
        groupsize: int = -1,
        sym: bool = False,
        clip_ratio: float = 1.0,
        quant_method: str = "int",
        bfp_groupsize: int = -1,
    ) -> None:
        assert quant_method in ["int", "bfp"], f"Unsupported quant_method: {quant_method}"
        if quant_method == "bfp":
            assert bits >= 2, "BFP quantization needs at least 2 bits"
            self.maxq = torch.tensor(2**bits - 1)
            sym = True
        else:
            _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.quant_method = quant_method
        self.bfp_groupsize = bfp_groupsize
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def _effective_bfp_groupsize(self):
        if self.bfp_groupsize > 0:
            return self.bfp_groupsize
        return self.groupsize

    def find_bfp_params_per_token_groupwise(self, x, groupsize) -> None:
        assert (
            x.shape[-1] % groupsize == 0
        ), f"Last dimension {x.shape[-1]} must be divisible by BFP groupsize {groupsize}"
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // groupsize, groupsize
        )

        xmax = torch.amax(torch.abs(reshaped_x), dim=3, keepdim=True) * self.clip_ratio
        self.scale = bfp_scale_from_absmax(xmax)
        self.zero = torch.zeros_like(self.scale)
        self.scale = self.scale.repeat(1, 1, 1, groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, groupsize).reshape(init_shape)

    def find_params_per_token_groupwise(self, x) -> None:
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.quant_method == "bfp":
            groupsize = self._effective_bfp_groupsize()
            if groupsize > 0:
                self.find_bfp_params_per_token_groupwise(x, groupsize)
                return

            reshaped_x = x.reshape((-1, x.shape[-1]))
            xmax = (
                torch.amax(torch.abs(reshaped_x), dim=1, keepdim=True)
                * self.clip_ratio
            )
            self.scale = bfp_scale_from_absmax(xmax).repeat(1, reshaped_x.shape[-1])
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
            return

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.online_block_had = False
        self.online_had_groupsize = -1
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            if self.quantizer.quant_method == "bfp":
                str_ += f" (BFP)"
            else:
                str_ += (
                    f" (Asymmetric Per-Token)"
                    if not self.quantizer.sym
                    else f" (Symmetric Per-Token)"
                )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            if self.out_quantizer.quant_method == "bfp":
                str_ += f" (BFP)"
            else:
                str_ += (
                    f" (Asymmetric Per-Token)"
                    if not self.out_quantizer.sym
                    else f" (Symmetric Per-Token)"
                )

        return str_

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_block_had:
            assert (
                self.online_had_groupsize > 0
            ), "online_had_groupsize should be positive for block Hadamard"
            if self.fp32_had:
                x = hadamard_utils.matmul_block_hadU_cuda(
                    x.float(), self.online_had_groupsize
                ).to(x_dtype)
            else:
                x = hadamard_utils.matmul_block_hadU_cuda(
                    x, self.online_had_groupsize
                )

        elif self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.quant_method = "int"

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
        quant_method: str = "int",
        bfp_groupsize: int = -1,
    ) -> None:
        assert quant_method in ["int", "bfp"], f"Unsupported quant_method: {quant_method}"
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        self.quant_method = quant_method
        self.bfp_groupsize = bfp_groupsize
        if quant_method == "bfp":
            assert bits >= 2, "BFP quantization needs at least 2 bits"
            self.sym = True
            self.mse = False
            self.maxq = torch.tensor(2**bits - 1)
        elif sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def _effective_bfp_groupsize(self):
        if self.bfp_groupsize > 0:
            return self.bfp_groupsize
        return self.weight_groupsize

    def find_params_weight_bfp_groupwise(self, x, groupsize) -> None:
        assert (
            x.shape[-1] % groupsize == 0
        ), f"Last dimension {x.shape[-1]} must be divisible by BFP groupsize {groupsize}"
        x = x.reshape(x.shape[-2], x.shape[-1] // groupsize, groupsize)

        xmax = torch.amax(torch.abs(x), dim=-1, keepdim=True)
        self.scale = bfp_scale_from_absmax(xmax)
        self.zero = torch.zeros(1, dtype=self.scale.dtype, device=self.scale.device)

    def find_params_weight_groupwise(self, x) -> None:
        init_shape = x.shape
        x = x.reshape(
            x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize
        )

        xmax = torch.amax(x, dim=-1, keepdim=True)
        xmin = torch.amin(x, dim=-1, keepdim=True)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, self.weight_groupsize)
        self.zero = self.zero.repeat(1, 1, self.weight_groupsize)

        if self.mse:
            best = torch.full(
                [x.shape[0], x.shape[1]], float("inf"), device=x.device
            ).type_as(x)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = sym_quant_dequant(x, scale1, self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = asym_quant_dequant(x, scale1, zero1, self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, -1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(init_shape)
        self.zero = self.zero.reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        if self.quant_method == "bfp":
            groupsize = self._effective_bfp_groupsize()
            if groupsize > 0:
                self.find_params_weight_bfp_groupwise(x, groupsize)
                return
            elif self.perchannel:
                x = x.flatten(1)
            else:
                x = x.flatten().unsqueeze(0)

            xmax = torch.amax(torch.abs(x), dim=1)
            self.scale = bfp_scale_from_absmax(xmax)
            self.zero = torch.zeros_like(self.scale)
            if not self.perchannel:
                tmp = shape[0]
                self.scale = self.scale.repeat(tmp)
                self.zero = self.zero.repeat(tmp)

            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return

        if self.weight_groupsize > 0:
            # group-wise per-token quantization
            self.find_params_weight_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.quant_method == "bfp":
                groupsize = self._effective_bfp_groupsize()
                if groupsize > 0 and self.scale.dim() == 3:
                    return BFPGroupwiseWeightSTEQuantize.apply(
                        x, self.scale, self.maxq, groupsize
                    ).to(x_dtype)
                return BFPSTEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            scale = self.scale.to(x.device)
            maxq = self.maxq.to(x.device)
            if self.quant_method == "bfp":
                groupsize = self._effective_bfp_groupsize()
                if groupsize > 0 and scale.dim() == 3:
                    init_shape = x.shape
                    grouped = x.reshape(
                        x.shape[-2], x.shape[-1] // groupsize, groupsize
                    )
                    q, scale = bfp_quant(grouped, scale, maxq)
                    dequant = bfp_dequant(q, scale, maxq).reshape(init_shape)
                    return dequant.to(x_dtype), q.reshape(init_shape), scale
                q, scale = bfp_quant(x, scale, maxq)
                return bfp_dequant(q, scale, maxq).to(x_dtype), q, scale
            q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
            return (scale * q).to(x_dtype), q, scale
        else:
            return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
) -> None:
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) is torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) is torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
