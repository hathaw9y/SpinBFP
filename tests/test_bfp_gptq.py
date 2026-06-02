import unittest
import sys
import types
from types import SimpleNamespace

import torch

if "fast_hadamard_transform" not in sys.modules:
    fast_hadamard_transform = types.ModuleType("fast_hadamard_transform")
    fast_hadamard_transform.hadamard_transform = lambda x, scale=1.0: x * scale
    sys.modules["fast_hadamard_transform"] = fast_hadamard_transform

from bfp_llama.modeling import BfpRotationLinear
from utils.bfp_gptq import _bfp_group_scales, bfp_gptq, bfp_gptq_from_block_hessians, gptq, quantize_bfp


def reference_bfp(x, bits=4, group_size=32, clip_ratio=1.0):
    shape = x.shape
    grouped = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True) * clip_ratio
    nonzero = absmax > 0
    safe = torch.where(nonzero, absmax, torch.ones_like(absmax))
    scale = torch.pow(2.0, torch.floor(torch.log2(safe)))
    scale = torch.where(nonzero, scale, torch.ones_like(scale))
    maxq = (2**bits) - 1
    denom = (maxq + 1) / 2
    q = torch.clamp(torch.round(grouped.abs() / scale * denom), 0, maxq)
    q = q * torch.sign(grouped)
    return (scale * (q / denom)).reshape(shape)


class BFPGPTQTest(unittest.TestCase):
    def test_quantize_bfp_matches_existing_bfp_quantizer(self):
        torch.manual_seed(0)
        x = torch.randn(8, 32)
        expected = reference_bfp(x, bits=4, group_size=32, clip_ratio=1.0)
        actual = quantize_bfp(x, bits=4, group_size=32, clip_ratio=1.0)
        self.assertTrue(torch.equal(actual, expected.float()))

    def test_gptq_accepts_column_quantizer(self):
        torch.manual_seed(1)
        W = torch.randn(16, 16)
        X = torch.randn(32, 16)
        H = 2.0 * X.t() @ X

        def quantize_col(col, _col_idx):
            return torch.round(col * 4) / 4

        q = gptq(W, H, quantize_col)
        self.assertEqual(q.shape, W.shape)
        self.assertTrue(torch.isfinite(q).all())

    def test_bfp_gptq_shape_dtype_and_metadata(self):
        torch.manual_seed(2)
        W = torch.randn(64, 64)
        X = torch.randn(32, 64)
        out = bfp_gptq(W, X, bits=4, group_size=32)
        self.assertEqual(out["W_quant"].shape, W.shape)
        self.assertEqual(out["W_quant"].dtype, torch.float32)
        self.assertEqual(out["bits"], 4)
        self.assertEqual(out["group_size"], 32)

    def test_bfp_gptq_proxy_improves_over_plain_bfp_rtn_for_seeded_case(self):
        torch.manual_seed(3)
        W = torch.randn(128, 128)
        X = torch.randn(64, 128)
        y = X @ W.t()
        rtn = quantize_bfp(W, bits=4, group_size=32)
        out = bfp_gptq(W, X, bits=4, group_size=32)

        rtn_rel = torch.mean((X @ rtn.t() - y) ** 2) / torch.mean(y**2)
        gptq_rel = torch.mean((X @ out["W_quant"].t() - y) ** 2) / torch.mean(y**2)
        self.assertLess(float(gptq_rel), float(rtn_rel))

    def test_bfp_gptq_records_fixed_scales_from_pre_gptq_weight(self):
        torch.manual_seed(4)
        W = torch.randn(3, 4)
        H_blocks = torch.eye(2).repeat(2, 1, 1)

        out = bfp_gptq_from_block_hessians(
            W,
            H_blocks,
            bits=4,
            group_size=2,
            reorder=False,
        )

        expected_scales = _bfp_group_scales(W, bits=4, group_size=2, clip_ratio=1.0)
        self.assertTrue(torch.equal(out["scales"], expected_scales))

    def test_loaded_bfp_gptq_weight_skips_runtime_weight_bfp(self):
        linear = torch.nn.Linear(4, 2, bias=False)
        cfg = SimpleNamespace(
            rotate=False,
            a_bits=16,
            a_bfp_group_size=2,
            w_bits=2,
            w_bfp_group_size=2,
            kv_bits=16,
            kv_bfp_group_size=2,
        )
        module = BfpRotationLinear(linear, "q_proj", cfg)
        gptq_weight = torch.tensor(
            [[0.30, 0.60, 1.70, -2.20], [0.11, -0.43, 0.91, 1.33]],
            dtype=torch.float32,
        )
        module.register_buffer("bfp_gptq_weight", gptq_weight)
        x = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float32)

        actual = module(x)
        expected = torch.nn.functional.linear(x, gptq_weight)

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
