from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _torch_load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected rotation checkpoint to be a dict, got {type(state).__name__}.")
    return state


@torch.no_grad()
def _copy_rotation_weight(module: nn.Module, tensor: torch.Tensor, *, name: str) -> None:
    if hasattr(module, "blocks"):
        blocks = module.blocks
        if tensor.dim() != 3:
            raise ValueError(f"{name} expected a block rotation tensor, got shape {tuple(tensor.shape)}.")
        if len(blocks) != tensor.shape[0]:
            raise ValueError(
                f"{name} block count mismatch: module has {len(blocks)}, checkpoint has {tensor.shape[0]}."
            )
        for block_idx, (param, value) in enumerate(zip(blocks, tensor)):
            if param.shape != value.shape:
                raise ValueError(
                    f"{name}[{block_idx}] shape mismatch: module has {tuple(param.shape)}, "
                    f"checkpoint has {tuple(value.shape)}."
                )
            param.copy_(value.to(device=param.device, dtype=param.dtype))
        return

    if not hasattr(module, "weight"):
        raise TypeError(f"{name} target module has no weight.")
    weight = module.weight
    if weight.shape != tensor.shape:
        raise ValueError(
            f"{name} shape mismatch: module has {tuple(weight.shape)}, checkpoint has {tuple(tensor.shape)}."
        )
    weight.copy_(tensor.to(device=weight.device, dtype=weight.dtype))


@torch.no_grad()
def load_learned_rotation_state(model: nn.Module, path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"rotation checkpoint not found: {path}")

    state = _torch_load_weights(path)
    if "R1" not in state:
        raise KeyError(f"rotation checkpoint is missing R1: {path}")
    _copy_rotation_weight(model.spinbfp_R1, state["R1"], name="R1")

    loaded = 1
    for idx, layer in enumerate(model.model.layers):
        key = f"model.layers.{idx}.self_attn.R2"
        if key not in state:
            raise KeyError(f"rotation checkpoint is missing {key}: {path}")
        _copy_rotation_weight(layer.self_attn.spinbfp_R2, state[key], name=key)
        loaded += 1

    return loaded
