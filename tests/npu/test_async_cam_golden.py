# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Optional CPU PyTorch regression tests for the FP16/BF16 golden contract."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from types import ModuleType
from typing import ClassVar

from tests.npu.async_cam_common import HarnessConfig, NativeHarness, quantize_row


class AsyncCamGoldenTests(unittest.TestCase):
    torch: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("CPU PyTorch is not installed")
        cls.torch = importlib.import_module("torch")

    def harness(
        self, dtype_name: str, *, top_k: int = 2, dynamic_quant: int = 0
    ) -> NativeHarness:
        harness = NativeHarness(
            HarnessConfig(dtype=dtype_name, top_k=top_k, dynamic_quant=dynamic_quant)
        )
        harness.torch = self.torch
        harness.dtype = (
            self.torch.float16 if dtype_name == "fp16" else self.torch.bfloat16
        )
        harness.rank = 0
        return harness

    def test_dispatch_to_combine_golden_accepts_both_dtypes_and_quant_modes(self):
        for dtype_name in ("fp16", "bf16"):
            for dynamic_quant in (0, 1):
                for rank in (0, 1):
                    with self.subTest(dtype=dtype_name, quant=dynamic_quant, rank=rank):
                        harness = self.harness(dtype_name, dynamic_quant=dynamic_quant)
                        harness.rank = rank
                        x, _, weights, _, _, expert_y, raw_y = harness._golden_dispatch(
                            0
                        )
                        golden = harness._combine_golden(expert_y, weights)
                        unquantized = harness._combine_golden(raw_y, weights)
                        self.assertEqual(golden.dtype, harness.dtype)
                        self.assertEqual(unquantized.dtype, harness.dtype)
                        self.assertEqual(golden.device.type, "cpu")
                        self.assertEqual(
                            golden.shape,
                            (harness.config.lengths[rank], harness.config.hidden_size),
                        )
                        budget = (
                            harness._quantization_budget(
                                x.float().abs().amax(dim=1), weights
                            )
                            if dynamic_quant
                            else 0
                        )
                        failures: list[str] = []
                        harness._check(
                            golden, unquantized, budget, 0, "CPU golden", failures
                        )
                        self.assertEqual(failures, [])

    def test_combine_golden_uses_selected_dtype_and_final_rounding(self):
        torch = self.torch
        for dtype_name, large, rounded in (
            ("fp16", 2048, 1024),
            ("bf16", 256, 128),
        ):
            with self.subTest(dtype=dtype_name):
                harness = self.harness(dtype_name)
                expert_y = torch.tensor(
                    [[[large, 2, -4], [1, 6, 2]]], dtype=harness.dtype
                )
                weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
                golden = harness._combine_golden(expert_y, weights)
                self.assertEqual(golden.dtype, harness.dtype)
                self.assertEqual(golden.device.type, "cpu")
                # The first lane is exactly halfway between representable values;
                # the final FP16/BF16 cast rounds it to the even lower value.
                self.assertEqual(golden.tolist(), [[rounded, 4, -1]])

    def test_combine_golden_accumulates_in_fp32_slot_order(self):
        torch = self.torch
        for dtype_name in ("fp16", "bf16"):
            with self.subTest(dtype=dtype_name):
                harness = self.harness(dtype_name, top_k=3)
                expert_y = torch.tensor([[[4096], [1], [-4096]]], dtype=harness.dtype)
                weights = torch.tensor([[4096, 1, 4096]], dtype=torch.float32)
                # FP32 computes (2**24 + 1) - 2**24 as zero. An FP64 sum or a
                # reordered cancellation would produce one before the final cast.
                golden = harness._combine_golden(expert_y, weights)
                self.assertEqual(golden.dtype, harness.dtype)
                self.assertEqual(golden.tolist(), [[0]])

    def test_quantization_budget_broadcasts_per_token_for_default_lengths(self):
        torch = self.torch
        harness = self.harness("fp16")
        for rows in (7, 5):
            with self.subTest(rows=rows):
                amax = torch.arange(1, rows + 1, dtype=torch.float32)
                weights = torch.tensor([[-0.25, 0.75]] * rows)
                budget = harness._quantization_budget(amax, weights)
                # Eight experts give a maximum transform of 39/32. The worst
                # row has amax=rows and sum(abs(weights))=1.
                self.assertAlmostEqual(budget, rows * 39 / (127 * 32), places=6)

    def test_quantization_budget_does_not_silently_broadcast_over_slots(self):
        torch = self.torch
        harness = self.harness("fp16")
        amax = torch.tensor([1, 10], dtype=torch.float32)
        weights = torch.tensor([[10, 1], [0.1, -0.2]], dtype=torch.float32)
        # With N=K, broadcasting amax over slots would silently give 20 instead
        # of the correct maximum weighted row magnitude, max(11, 3)=11.
        budget = harness._quantization_budget(amax, weights)
        self.assertAlmostEqual(budget, 11 * 39 / (127 * 32), places=6)

    def test_exact_comparison_rejects_signed_zero_but_tolerance_accepts_it(self):
        torch = self.torch
        for dtype_name in ("fp16", "bf16"):
            with self.subTest(dtype=dtype_name):
                harness = self.harness(dtype_name)
                actual = torch.tensor([-0.0], dtype=harness.dtype)
                expected = torch.tensor([0.0], dtype=harness.dtype)
                failures: list[str] = []
                harness._check(actual, expected, 0, 0, "exact", failures)
                self.assertEqual(len(failures), 1)
                self.assertIn("bitwise_mismatches", failures[0])
                failures = []
                harness._check(actual, expected, 0.001, 0, "tolerant", failures)
                self.assertEqual(failures, [])

    def test_comparison_rejects_wrong_golden_dtype(self):
        torch = self.torch
        for dtype_name in ("fp16", "bf16"):
            with self.subTest(dtype=dtype_name):
                harness = self.harness(dtype_name)
                actual = torch.tensor([1.0], dtype=harness.dtype)
                expected = torch.tensor([1.0], dtype=torch.float64)
                failures: list[str] = []
                harness._check(actual, expected, 1, 1, "dtype", failures)
                self.assertEqual(len(failures), 1)
                self.assertIn("shape/dtype mismatch", failures[0])

    def test_comparison_accepts_near_values_and_rejects_excess_error(self):
        torch = self.torch
        for dtype_name in ("fp16", "bf16"):
            with self.subTest(dtype=dtype_name):
                harness = self.harness(dtype_name)
                expected = torch.tensor([1, -2], dtype=harness.dtype)
                actual = torch.tensor([1.125, -2.125], dtype=harness.dtype)
                failures: list[str] = []
                harness._check(actual, expected, 0.125, 0, "within", failures)
                self.assertEqual(failures, [])
                harness._check(actual, expected, 0.0625, 0, "outside", failures)
                self.assertEqual(len(failures), 1)

    def test_quantization_scalar_and_tensor_round_fp32_product_before_rint(self):
        torch = self.torch
        for dtype_name in ("fp16", "bf16"):
            with self.subTest(dtype=dtype_name):
                harness = self.harness(dtype_name)
                values = [1.5, 0.75]
                quantized, _, _ = harness._quantize_contract(
                    torch.tensor([values], dtype=harness.dtype)
                )
                # The FP32 multiplication rounds 0.75 * (127 / 1.5) to 63.5;
                # round-to-nearest-even therefore emits 64, rather than 63.
                self.assertEqual(quantized.tolist(), [[127, 64]])
                self.assertEqual(quantize_row(values), [127, 64])


if __name__ == "__main__":
    unittest.main()
