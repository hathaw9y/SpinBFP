import torch


def is_block_diag_rotation(rotation):
    return rotation is not None and rotation.dim() == 3


def rotation_total_dim(rotation):
    if is_block_diag_rotation(rotation):
        return rotation.shape[0] * rotation.shape[-1]
    return rotation.shape[-1]


def apply_rotation_right(x, rotation, compute_dtype=torch.float64, transpose=False):
    if rotation is None:
        return x

    x_dtype = x.dtype
    if not is_block_diag_rotation(rotation):
        matrix = rotation.t() if transpose else rotation
        return (x.to(compute_dtype) @ matrix.to(compute_dtype)).to(x_dtype)

    num_blocks, block_size, _ = rotation.shape
    if x.shape[-1] != num_blocks * block_size:
        raise ValueError(
            f"last dim {x.shape[-1]} does not match block-diag rotation dim "
            f"{num_blocks * block_size}"
        )

    matrix = rotation.transpose(-1, -2) if transpose else rotation
    x_blocks = x.reshape(*x.shape[:-1], num_blocks, block_size).to(compute_dtype)
    out = torch.einsum("...bi,bij->...bj", x_blocks, matrix.to(compute_dtype))
    return out.reshape_as(x).to(x_dtype)


def apply_rotation_left(x, rotation, compute_dtype=torch.float64, transpose=False):
    if rotation is None:
        return x

    x_dtype = x.dtype
    if not is_block_diag_rotation(rotation):
        matrix = rotation.t() if transpose else rotation
        return (matrix.to(compute_dtype) @ x.to(compute_dtype)).to(x_dtype)

    num_blocks, block_size, _ = rotation.shape
    if x.shape[0] != num_blocks * block_size:
        raise ValueError(
            f"first dim {x.shape[0]} does not match block-diag rotation dim "
            f"{num_blocks * block_size}"
        )

    matrix = rotation.transpose(-1, -2) if transpose else rotation
    x_blocks = x.reshape(num_blocks, block_size, *x.shape[1:]).to(compute_dtype)
    out = torch.einsum("bij,bj...->bi...", matrix.to(compute_dtype), x_blocks)
    return out.reshape_as(x).to(x_dtype)
