import types

import torch
import torch.nn as nn

from utils import bfp_quantize_activation


def _qk_bfp_bits(hook) -> int:
    bits = getattr(hook, "bfp_qk_bits", None)
    return getattr(hook, "bfp_bits", 8) if bits is None else bits


def patch_opt_attention(attn_module, R_head, layer_idx: int, hook) -> None:
    """Patch OPT attention for optional Q/K rotation and BFP."""
    rotate = R_head is not None

    def rotate_head(query_states, key_states):
        if not rotate:
            return query_states, key_states
        R_h = R_head.to(query_states.dtype)
        return query_states @ R_h, key_states @ R_h

    def patched_forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz)

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            query_states, key_states = rotate_head(query_states, key_states)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            query_states, key_states = rotate_head(query_states, key_states)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        if hook.bfp:
            qk_bits = _qk_bfp_bits(hook)
            query_states = bfp_quantize_activation(
                query_states, hook.bfp_block_size, qk_bits,
                stat_hook=hook,
                stat_name=f"model.decoder.layers.{layer_idx}.self_attn.query_states",
            )
            key_states = bfp_quantize_activation(
                key_states, hook.bfp_block_size, qk_bits,
                stat_hook=hook,
                stat_name=f"model.decoder.layers.{layer_idx}.self_attn.key_states",
            )

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}"
                )
            attn_weights = (
                layer_head_mask.view(1, -1, 1, 1)
                * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    attn_module.forward = types.MethodType(patched_forward, attn_module)
