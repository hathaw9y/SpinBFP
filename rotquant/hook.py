class Hook:
    """Shared options used by rotation and BFP monkey patches."""

    bfp = False
    bfp_bits = 8
    bfp_block_size = 128
    bfp_qkv_bits = None
    bfp_o_bits = None
    bfp_up_gate_bits = None
    bfp_down_bits = None
    bfp_qk_bits = None
    weight_bfp = False
    weight_bfp_bits = 8
    weight_bfp_block_size = 128
    online_rotate = False
    orth_group_size = 128
    model_dir = None
