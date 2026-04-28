import types

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from utils import bfp_quantize_activation


def _qk_bfp_bits(hook) -> int:
    bits = getattr(hook, "bfp_qk_bits", None)
    return getattr(hook, "bfp_bits", 8) if bits is None else bits


def patch_llama_attention(attn_module, R_head, layer_idx: int, hook) -> None:
    """Patch LLaMA attention for optional post-RoPE Q/K rotation and BFP."""
    rotate = R_head is not None

    def rotate_head(query_states, key_states):
        if not rotate:
            return query_states, key_states
        R_h = R_head.to(query_states.dtype)
        return query_states @ R_h, key_states @ R_h

    def patched_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        query_states, key_states = rotate_head(query_states, key_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if hook.bfp:
            qk_bits = _qk_bfp_bits(hook)
            query_stat_name = f"model.layers.{layer_idx}.self_attn.qk_matmul.query"
            key_stat_name = f"model.layers.{layer_idx}.self_attn.qk_matmul.key"
            if hook.is_bfp_enabled_for_position(query_stat_name):
                query_states = bfp_quantize_activation(
                    query_states, hook.bfp_block_size, qk_bits,
                    stat_hook=hook,
                    stat_name=query_stat_name,
                )
            if hook.is_bfp_enabled_for_position(key_stat_name):
                key_states = bfp_quantize_activation(
                    key_states, hook.bfp_block_size, qk_bits,
                    stat_hook=hook,
                    stat_name=key_stat_name,
                )

        if attention_mask is not None and attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            bsz, q_len, self.hidden_size
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    attn_module.forward = types.MethodType(patched_forward, attn_module)
