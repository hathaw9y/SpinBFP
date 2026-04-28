from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import bfp_quantize_activation


RECONSTRUCTION_ORDER = (
    "down_proj",
    "o_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "up_proj",
    "gate_proj",
    "lm_head",
)


def _linear_bfp_bits(linear: nn.Linear, hook) -> int:
    category = getattr(linear, "_spinkv_bfp_category", None)
    override = {
        "qkv": getattr(hook, "bfp_qkv_bits", None),
        "o": getattr(hook, "bfp_o_bits", None),
        "up_gate": getattr(hook, "bfp_up_gate_bits", None),
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
    for _, targets in reconstruction_targets_by_group(model, hook, groups=("down_proj", "o_proj")):
        yield from targets


def reconstruction_targets_by_group(model, hook, groups=None):
    requested = tuple(groups or RECONSTRUCTION_ORDER)
    targets_by_group = {group: [] for group in requested}

    def add(group, name, linear):
        if group not in targets_by_group:
            return
        if hook.is_bfp_enabled_for_position(f"{name}.input"):
            targets_by_group[group].append((name, linear))

    if model.model_type == "llama2":
        for layer_idx, layer in enumerate(model.model.layers):
            prefix = f"model.layers.{layer_idx}"
            attn, mlp = layer.self_attn, layer.mlp
            add("q_proj", f"{prefix}.self_attn.q_proj", attn.q_proj)
            add("k_proj", f"{prefix}.self_attn.k_proj", attn.k_proj)
            add("v_proj", f"{prefix}.self_attn.v_proj", attn.v_proj)
            add("o_proj", f"{prefix}.self_attn.o_proj", attn.o_proj)
            add("gate_proj", f"{prefix}.mlp.gate_proj", mlp.gate_proj)
            add("up_proj", f"{prefix}.mlp.up_proj", mlp.up_proj)
            add("down_proj", f"{prefix}.mlp.down_proj", mlp.down_proj)
        add("lm_head", "lm_head", model.lm_head)
    elif model.model_type == "opt":
        for layer_idx, layer in enumerate(model.model.decoder.layers):
            prefix = f"model.decoder.layers.{layer_idx}"
            attn = layer.self_attn
            add("q_proj", f"{prefix}.self_attn.q_proj", attn.q_proj)
            add("k_proj", f"{prefix}.self_attn.k_proj", attn.k_proj)
            add("v_proj", f"{prefix}.self_attn.v_proj", attn.v_proj)
            add("o_proj", f"{prefix}.self_attn.out_proj", attn.out_proj)
            add("up_proj", f"{prefix}.fc1", layer.fc1)
            add("down_proj", f"{prefix}.fc2", layer.fc2)
        add("lm_head", "lm_head", model.lm_head)
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")

    for group in requested:
        targets = targets_by_group.get(group, [])
        if targets:
            yield group, targets


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


def save_reconstructed_weight_state(path: str, weights, metadata=None) -> None:
    torch.save({"weights": weights, "metadata": metadata or {}}, path)


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
        if name == "lm_head":
            module.weight = nn.Parameter(module.weight.data.clone())
        module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
        loaded += 1

    if strict and loaded != len(weights):
        raise RuntimeError(f"Loaded {loaded} of {len(weights)} reconstructed weights.")
    print(f"Loaded {loaded} reconstructed weights from {path}")


def load_reconstructed_weight_path(model, path: str, strict: bool = True) -> None:
    recon_path = Path(path)
    if recon_path.is_dir():
        files = sorted(recon_path.glob("recon_*.pt"))
        if strict and not files:
            raise FileNotFoundError(f"No recon_*.pt files found in {path}")
        for file in files:
            load_reconstructed_weights(model, str(file), strict=strict)
        return

    load_reconstructed_weights(model, path, strict=strict)


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
def reconstruct_linear_weight(
    model,
    input_ids: torch.Tensor,
    name: str,
    linear: nn.Linear,
    hook,
    ridge: float,
    blend: float,
    row_chunk: int,
):
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

    result = new_weight.detach().cpu()
    del gram, rhs, solution, new_weight
    if linear.weight.is_cuda:
        torch.cuda.empty_cache()
    return name, result


@torch.no_grad()
def reconstruct_weight_groups(
    model,
    tokenizer,
    hook,
    device: str,
    groups=None,
    nsamples: int = 128,
    seq_len: int = 2048,
    ridge: float = 1e-4,
    blend: float = 1.0,
    row_chunk: int = 2048,
):
    input_ids = _calibration_input_ids(tokenizer, nsamples, seq_len, device)
    grouped_targets = list(reconstruction_targets_by_group(model, hook, groups=groups))
    if not grouped_targets:
        print("No BFP reconstruction targets found.")
        return {}

    reconstructed = {}
    for group, targets in grouped_targets:
        print(f"Reconstructing {group}: {len(targets)} modules")
        group_weights = {}
        for name, linear in tqdm(targets):
            target_name, weight = reconstruct_linear_weight(
                model,
                input_ids,
                name,
                linear,
                hook,
                ridge=ridge,
                blend=blend,
                row_chunk=row_chunk,
            )
            group_weights[target_name] = weight
        reconstructed[group] = group_weights
    return reconstructed


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
