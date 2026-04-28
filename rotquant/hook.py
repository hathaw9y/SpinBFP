import re


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
        self.bfp_round_only_exp_le16 = False
        self.bfp_strict_round_exp_ge17 = False
        self.bfp_shared_exponent_stats = False
        self._bfp_shared_exponent_stats = {}
        self.disabled_bfp_positions = set()

    def is_bfp_enabled_for_position(self, name):
        return self._bfp_position_name(name) not in self.disabled_bfp_positions

    def record_bfp_shared_exponent(self, name, shared_exp):
        if not self.bfp_shared_exponent_stats:
            return

        count = shared_exp.numel()
        if count == 0:
            return

        shared_exp_float = shared_exp.detach().float()
        stat = self._bfp_shared_exponent_stats.setdefault(
            name,
            {
                "sum": 0.0,
                "sum_sq": 0.0,
                "count": 0,
                "calls": 0,
                "min": float("inf"),
                "max": float("-inf"),
            },
        )
        stat["sum"] += shared_exp_float.sum().item()
        stat["sum_sq"] += shared_exp_float.square().sum().item()
        stat["count"] += count
        stat["calls"] += 1
        stat["min"] = min(stat["min"], shared_exp_float.min().item())
        stat["max"] = max(stat["max"], shared_exp_float.max().item())

    def bfp_shared_exponent_averages(self):
        averages = []
        for name, stat in sorted(
            self._bfp_shared_exponent_stats.items(),
            key=lambda item: self._bfp_location_sort_key(item[0]),
        ):
            averages.append(self._bfp_stat_row(name, stat))
        return averages

    def bfp_shared_exponent_layer_averages(self):
        layer_stats = {}
        for name, stat in self._bfp_shared_exponent_stats.items():
            layer_idx = self._bfp_layer_idx(name)
            if layer_idx is None:
                continue

            layer_name = f"layer.{layer_idx}"
            layer_stat = layer_stats.setdefault(
                layer_name,
                {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "calls": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
            )
            layer_stat["sum"] += stat["sum"]
            layer_stat["sum_sq"] += stat["sum_sq"]
            layer_stat["count"] += stat["count"]
            layer_stat["calls"] += stat["calls"]
            layer_stat["min"] = min(layer_stat["min"], stat["min"])
            layer_stat["max"] = max(layer_stat["max"], stat["max"])

        return [
            self._bfp_stat_row(name, stat)
            for name, stat in sorted(
                layer_stats.items(),
                key=lambda item: self._bfp_layer_sort_key(item[0]),
            )
        ]

    def bfp_shared_exponent_position_averages(self):
        position_stats = {}
        for name, stat in self._bfp_shared_exponent_stats.items():
            position_name = self._bfp_position_name(name)
            position_stat = position_stats.setdefault(
                position_name,
                {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "count": 0,
                    "calls": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                },
            )
            position_stat["sum"] += stat["sum"]
            position_stat["sum_sq"] += stat["sum_sq"]
            position_stat["count"] += stat["count"]
            position_stat["calls"] += stat["calls"]
            position_stat["min"] = min(position_stat["min"], stat["min"])
            position_stat["max"] = max(position_stat["max"], stat["max"])

        return [
            self._bfp_stat_row(name, stat)
            for name, stat in sorted(
                position_stats.items(),
                key=lambda item: self._bfp_location_sort_key(item[0]),
            )
        ]

    @staticmethod
    def _bfp_stat_row(name, stat):
        count = stat["count"]
        mean = stat["sum"] / count if count else float("nan")
        variance = stat["sum_sq"] / count - mean * mean if count else float("nan")
        variance = max(variance, 0.0)
        return {
            "name": name,
            "mean": mean,
            "variance": variance,
            "min": stat["min"],
            "max": stat["max"],
            "count": count,
            "calls": stat["calls"],
        }

    @staticmethod
    def _bfp_layer_idx(name):
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _bfp_position_name(name):
        return re.sub(r"^.*\.layers\.\d+\.", "", name, count=1)

    @classmethod
    def _bfp_location_sort_key(cls, name):
        parts = re.split(r"(\d+)", name)
        return tuple(int(part) if part.isdigit() else part for part in parts)

    @staticmethod
    def _bfp_layer_sort_key(name):
        match = re.search(r"layer\.(\d+)$", name)
        return int(match.group(1)) if match is not None else name
