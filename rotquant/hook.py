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

    def __init__(self):
        self.bfp_shared_exponent_stats = False
        self._bfp_shared_exponent_stats = {}

    def record_bfp_shared_exponent(self, name, shared_exp):
        if not self.bfp_shared_exponent_stats:
            return

        count = shared_exp.numel()
        if count == 0:
            return

        stat = self._bfp_shared_exponent_stats.setdefault(
            name,
            {"sum": 0.0, "count": 0, "calls": 0},
        )
        stat["sum"] += shared_exp.detach().float().sum().item()
        stat["count"] += count
        stat["calls"] += 1

    def bfp_shared_exponent_averages(self):
        averages = []
        for name, stat in sorted(self._bfp_shared_exponent_stats.items()):
            count = stat["count"]
            mean = stat["sum"] / count if count else float("nan")
            averages.append(
                {
                    "name": name,
                    "mean": mean,
                    "count": count,
                    "calls": stat["calls"],
                }
            )
        return averages
