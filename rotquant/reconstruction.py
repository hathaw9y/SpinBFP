from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import bfp_quantize_activation


def _linear_bfp_bits(linear: nn.Linear, hook) -> int:
    category = getattr(linear, "_spinkv_bfp_category", None)
    override = {
        "o": getattr(hook, "bfp_o_bits", None),
        "down": getattr(hook, "bfp_down_bits", None),
    }.get(category, None)
    return getattr(hook, "bfp_bits", 8) if override is None else override


def _calibration_input_ids(tokenizer, nsamples: int, seq_len: int, device: str):
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    input_ids = enc.input_ids[:, : nsamples * seq_len]
    if input_ids.shape[1] < nsamples * seq_len:
        raise ValueError(
            f"Not enough calibration tokens: need {nsamples * seq_len}, "
            f"got {input_ids.shape[1]}"
        )
    return input_ids.reshape(nsamples, seq_len).to(device)


def _target_linears(model, hook):
    if model.model_type == "llama2":
        for layer_idx, layer in enumerate(model.model.layers):
            targets = (
                (f"model.layers.{layer_idx}.self_attn.o_proj", layer.self_attn.o_proj),
                (f"model.layers.{layer_idx}.mlp.down_proj", layer.mlp.down_proj),
            )
            for name, linear in targets:
                if hook.is_bfp_enabled_for_position(f"{name}.input"):
                    yield name, linear
    elif model.model_type == "opt":
        for layer_idx, layer in enumerate(model.model.decoder.layers):
            targets = (
                (f"model.decoder.layers.{layer_idx}.self_attn.out_proj", layer.self_attn.out_proj),
                (f"model.decoder.layers.{layer_idx}.fc2", layer.fc2),
            )
            for name, linear in targets:
                if hook.is_bfp_enabled_for_position(f"{name}.input"):
                    yield name, linear
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")


def reconstructed_weight_state(model, hook):
    return {
        name: linear.weight.detach().cpu()
        for name, linear in _target_linears(model, hook)
    }


def save_reconstructed_weights(path: str, model, hook, metadata=None) -> None:
    state = {
        "weights": reconstructed_weight_state(model, hook),
        "metadata": metadata or {},
    }
    torch.save(state, path)


def load_reconstructed_weights(model, path: str, strict: bool = True) -> None:
    state = torch.load(path, map_location="cpu")
    weights = state.get("weights", state)
    modules = dict(model.named_modules())
    loaded = 0

    for name, weight in weights.items():
        module = modules.get(name)
        if module is None:
            if strict:
                raise KeyError(f"Reconstructed weight target not found: {name}")
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Reconstructed weight target is not nn.Linear: {name}")
        if tuple(module.weight.shape) != tuple(weight.shape):
            raise ValueError(
                f"Shape mismatch for {name}: model has {tuple(module.weight.shape)}, "
                f"checkpoint has {tuple(weight.shape)}"
            )
        module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
        loaded += 1

    if strict and loaded != len(weights):
        raise RuntimeError(f"Loaded {loaded} of {len(weights)} reconstructed weights.")
    print(f"Loaded {loaded} reconstructed weights from {path}")


@torch.no_grad()
def _accumulate_linear_reconstruction(
    model,
    input_ids: torch.Tensor,
    linear: nn.Linear,
    hook,
    row_chunk: int,
):
    device = linear.weight.device
    in_features = linear.in_features
    out_features = linear.out_features
    bits = _linear_bfp_bits(linear, hook)
    block_size = getattr(hook, "bfp_block_size", 128)

    gram = torch.zeros((in_features, in_features), device=device, dtype=torch.float32)
    rhs = torch.zeros((in_features, out_features), device=device, dtype=torch.float32)

    def collect(_, inputs):
        x = inputs[0].detach()
        xq = bfp_quantize_activation(x, block_size=block_size, mbits=bits)
        x = x.reshape(-1, in_features)
        xq = xq.reshape(-1, in_features)
        weight = linear.weight.detach().to(device=device, dtype=torch.float32)

        for start in range(0, x.shape[0], row_chunk):
            end = min(start + row_chunk, x.shape[0])
            xb = x[start:end].to(device=device, dtype=torch.float32)
            xqb = xq[start:end].to(device=device, dtype=torch.float32)
            target = xb @ weight.T
            gram.add_(xqb.T @ xqb)
            rhs.add_(xqb.T @ target)

    handle = linear.register_forward_pre_hook(collect)
    try:
        for batch in input_ids:
            model(batch.unsqueeze(0), use_cache=False)
    finally:
        handle.remove()

    return gram, rhs


@torch.no_grad()
def reconstruct_down_o_weights(
    model,
    tokenizer,
    hook,
    device: str,
    nsamples: int = 128,
    seq_len: int = 2048,
    ridge: float = 1e-4,
    blend: float = 1.0,
    row_chunk: int = 2048,
) -> None:
    input_ids = _calibration_input_ids(tokenizer, nsamples, seq_len, device)
    targets = list(_target_linears(model, hook))
    if not targets:
        print("No down/o projection BFP reconstruction targets found.")
        return

    print(
        f"Reconstructing {len(targets)} down/o projection weights "
        f"with {nsamples} WikiText-2 calibration samples..."
    )
    for name, linear in tqdm(targets):
        gram, rhs = _accumulate_linear_reconstruction(
            model, input_ids, linear, hook, row_chunk
        )
        damp = ridge * gram.diagonal().mean().clamp(min=1e-8)
        gram.diagonal().add_(damp)

        try:
            chol = torch.linalg.cholesky(gram)
            solution = torch.cholesky_solve(rhs, chol)
        except torch.linalg.LinAlgError:
            solution = torch.linalg.solve(gram, rhs)

        new_weight = solution.T.to(dtype=linear.weight.dtype)
        if blend < 1.0:
            new_weight = torch.lerp(linear.weight.data, new_weight, blend)
        linear.weight.data.copy_(new_weight)

        del gram, rhs, solution, new_weight
        if linear.weight.is_cuda:
            torch.cuda.empty_cache()
