# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor

from utils.rotation_utils import apply_rotation_left, apply_rotation_right


def _rotation_compute_dtype(module):
    return getattr(module, "rotation_compute_dtype", torch.float64)


class QuantizeLinear(nn.Linear):
    def forward(
        self,
        input: Tensor,
        R1=None,
        R2=None,
        transpose=False,
    ) -> Tensor:
        # quantize weight
        if R1 is not None:
            dtype = self.weight.dtype
            compute_dtype = _rotation_compute_dtype(self)
            if not transpose:
                weight = apply_rotation_right(self.weight, R1, compute_dtype)
            else:
                weight = apply_rotation_left(self.weight, R1, compute_dtype, transpose=True)
            if R2 is not None:
                # Each head dim = 128 for Llama model
                had_dim = R2.shape[0] * R2.shape[-1] if R2.dim() == 3 else R2.shape[0]
                dtype = weight.dtype
                if transpose:
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = apply_rotation_right(temp, R2, compute_dtype)
                    weight = temp.reshape(init_shape)
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = apply_rotation_right(temp, R2, compute_dtype)
                    weight = temp.reshape(transposed_shape).t()
            weight = weight.to(dtype)
        else:
            weight = self.weight
        if hasattr(self, "quantizer"):
            dtype = weight.dtype
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(dtype)

        return nn.functional.linear(input, weight, self.bias)
