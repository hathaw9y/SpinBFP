import torch
import torch.nn as nn

from .fusion import fuse_norms
from .rotation import (
    absorb_R_input, absorb_R_output, absorb_R_into_embedding,
    patch_online_rotate, patch_linear_bfp, apply_linear_weight_bfp,
)
from .attention.llama import patch_llama_attention
from .attention.opt import patch_opt_attention


def _random_hadamard_matrix(size: int, device):
    from hadamard_utils import random_hadamard_matrix
    return random_hadamard_matrix(size, device=device)


def add_model_type(model) -> None:
    name = model.config._name_or_path.lower()
    if 'llama-2' in name:
        model.model_type = 'llama2'
    elif 'opt' in name:
        model.model_type = 'opt'
    else:
        raise ValueError(f"Unsupported Model: {model.config._name_or_path}")


def _patch_online_input_rotate(linear: nn.Linear, R: torch.Tensor, hook) -> None:
    """Hadamard intermediate path처럼 weight 흡수와 runtime input rotate를 함께 적용."""
    absorb_R_input(linear, R)
    patch_online_rotate(linear, R, hook)


def _apply_llama_hadamard_rotate(model, device, hook) -> None:
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    head_dim = hidden // model.config.num_attention_heads

    R_res = _random_hadamard_matrix(hidden, device=device)
    R_mlp = _random_hadamard_matrix(intermediate, device=device)
    R_head = _qk_rotation(model, device, hook)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj,
                     mlp.gate_proj, mlp.up_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.o_proj, R_res)
        absorb_R_output(mlp.down_proj, R_res)

        if hook.online_rotate:
            _patch_online_input_rotate(mlp.down_proj, R_mlp, hook)

        patch_llama_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_mlp, R_head
    torch.cuda.empty_cache()


def _apply_opt_hadamard_rotate(model, device, hook) -> None:
    hidden = model.config.hidden_size
    ffn_dim = model.config.ffn_dim
    head_dim = hidden // model.config.num_attention_heads

    if model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    R_res = _random_hadamard_matrix(hidden, device=device)
    R_ffn = _random_hadamard_matrix(ffn_dim, device=device)
    R_head = _qk_rotation(model, device, hook)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.decoder.layers):
        attn = layer.self_attn
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.out_proj, R_res)
        absorb_R_input(layer.fc1, R_res)
        absorb_R_output(layer.fc2, R_res)

        if hook.online_rotate:
            _patch_online_input_rotate(layer.fc2, R_ffn, hook)

        patch_opt_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_ffn, R_head
    torch.cuda.empty_cache()


def _qk_rotation(model, device, hook):
    if not hook.online_rotate:
        return None
    if model.model_type == 'llama2':
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    elif model.model_type == 'opt':
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")
    return _random_hadamard_matrix(head_dim, device=device)


def _patch_attention_only(model, device, hook) -> None:
    """rotation 없이 attention patch만 적용 (R_head=None)."""
    R_head = _qk_rotation(model, device, hook)
    if model.model_type == 'llama2':
        for layer_idx, layer in enumerate(model.model.layers):
            patch_llama_attention(layer.self_attn, R_head, layer_idx, hook)
    elif model.model_type == 'opt':
        for layer_idx, layer in enumerate(model.model.decoder.layers):
            patch_opt_attention(layer.self_attn, R_head, layer_idx, hook)
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")


def _tag_linear_bfp_categories(model) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._spinkv_bfp_name = name

    if model.model_type == 'llama2':
        for layer in model.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                proj._spinkv_bfp_category = 'qkv'
            attn.o_proj._spinkv_bfp_category = 'o'
            for proj in (mlp.gate_proj, mlp.up_proj):
                proj._spinkv_bfp_category = 'up_gate'
            mlp.down_proj._spinkv_bfp_category = 'down'
    elif model.model_type == 'opt':
        for layer in model.model.decoder.layers:
            attn = layer.self_attn
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                proj._spinkv_bfp_category = 'qkv'
            attn.out_proj._spinkv_bfp_category = 'o'
            layer.fc1._spinkv_bfp_category = 'up_gate'
            layer.fc2._spinkv_bfp_category = 'down'
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")


def prepare_model_for_rotate(model) -> None:
    add_model_type(model)
    fuse_norms(model)
    _tag_linear_bfp_categories(model)


def apply_rotate(
    model,
    device,
    hook,
    rotate: str | None = "hadamard",
    pre_rotate_callback=None,
) -> None:
    """
    rotate='hadamard'   : fuse_norms + Hadamard 회전 + attention patch
    rotate=None         : fuse_norms + attention patch without weight-space rotation
    """
    prepare_model_for_rotate(model)
    if pre_rotate_callback is not None:
        pre_rotate_callback()

    if rotate is None:
        _patch_attention_only(model, device, hook)
        if getattr(hook, 'weight_bfp', False):
            _apply_linear_weight_bfp(model, hook)
        if getattr(hook, 'bfp', False):
            _patch_linear_bfp(model, hook)
        return

    if rotate == 'hadamard':
        if model.model_type == 'llama2':
            _apply_llama_hadamard_rotate(model, device, hook)
        elif model.model_type == 'opt':
            _apply_opt_hadamard_rotate(model, device, hook)
        else:
            raise ValueError(f"Unsupported model_type: {model.model_type}")
    else:
        raise ValueError(f"Unsupported rotate: {rotate}")

    if getattr(hook, 'weight_bfp', False):
        _apply_linear_weight_bfp(model, hook)

    if getattr(hook, 'bfp', False):
        _patch_linear_bfp(model, hook)


def _patch_linear_bfp(model, hook) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            patch_linear_bfp(module, hook)


def _apply_linear_weight_bfp(model, hook) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            apply_linear_weight_bfp(module, hook)
