import math

import torch

from bfp_refactor.utils import hadamard_utils
from bfp_refactor.utils.utils import HadamardTransform


def random_hadamard(size, device):
    return hadamard_utils.random_hadamard_matrix(size, device).to(torch.float32)


def apply_full_had(x):
    had_k, k = hadamard_utils.get_hadK(x.shape[-1])
    return hadamard_utils.matmul_hadU_cuda(x, had_k, k)


def apply_block_had(x, group_size):
    return hadamard_utils.matmul_block_hadU_cuda(x, group_size)


def apply_head_had(x, group_size):
    if group_size is not None and group_size > 0:
        return apply_block_had(x, group_size)
    return HadamardTransform.apply(x.contiguous()) / math.sqrt(x.shape[-1])


def apply_exact_had_to_linear(linear, group_size):
    hadamard_utils.apply_exact_had_to_linear(linear, had_dim=group_size, output=False)


def apply_r2_to_weight(weight, r2, transpose):
    if r2 is None:
        return weight
    had_dim = r2.shape[0]
    if transpose:
        shape = weight.shape
        temp = weight.reshape(-1, shape[-1] // had_dim, had_dim)
        return (temp.to(torch.float64) @ r2.to(torch.float64)).reshape(shape).to(
            weight.dtype
        )

    wt = weight.t()
    shape = wt.shape
    temp = wt.reshape(-1, shape[-1] // had_dim, had_dim)
    return (temp.to(torch.float64) @ r2.to(torch.float64)).reshape(shape).t().to(
        weight.dtype
    )
