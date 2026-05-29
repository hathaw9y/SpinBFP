import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    model: str
    max_length: int = 2048
    seed: int = 0
    w_bits: int = 4
    a_bits: int = 4
    kv_bits: int = 4
    bfp_group_size: int = 32
    w_bfp_group_size: int = 32
    a_bfp_group_size: int = 32
    kv_bfp_group_size: int = 32
    online_had_group_size: int = -1
    w_down_had_group_size: int = -1
    qk_had_group_size: int = -1
    qk_matmul_bits: int = 4
    av_matmul_bits: int = 4
    qk_matmul_bfp_group_size: int = 32
    av_matmul_bfp_group_size: int = 32
    rotate: bool = True
    fp32_had: bool = False

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
