# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU checks for benchmark metric definitions and readiness sequencing."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.npu.async_cam_performance import FocusSchedule, selected_ops, summarize

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class BenchmarkContractTests(unittest.TestCase):
    def test_percentiles_do_not_pool_or_interpolate_samples(self):
        values = list(range(1, 21))
        report = summarize(values)
        self.assertEqual(report["count"], 20)
        self.assertEqual(report["median_ms"], 10.5)
        self.assertEqual(report["p95_ms"], 19)
        self.assertEqual(summarize([]), {"count": 0})
        self.assertEqual(summarize([2.0])["p95_ms"], 2.0)

    def test_focus_excludes_peer_wait_samples(self):
        self.assertEqual(selected_ops("send"), ("dispatch_send", "combine_send"))
        self.assertEqual(selected_ops("recv"), ("dispatch_recv", "combine_recv"))
        self.assertEqual(len(selected_ops("cycle")), 4)

    @patch("tests.npu.async_cam_performance.time.sleep")
    def test_sender_publishes_readiness_without_waiting_for_receive(self, sleep):
        store = Mock()
        schedule = FocusSchedule(store, "recv", 1, 2, 2, 50, 200, 20)
        schedule("dispatch_send")
        store.set.assert_called_once_with("dispatch/send/1", "ready")
        store.wait.assert_not_called()
        sleep.assert_called_once_with(0.05)

    @patch("tests.npu.async_cam_performance.time.sleep")
    def test_receiver_waits_for_all_senders_only_before_first_chunk(self, sleep):
        store = Mock()
        schedule = FocusSchedule(store, "recv", 3, 2, 2, 50, 200, 20)
        schedule("dispatch_recv")
        schedule("dispatch_recv")
        store.wait.assert_called_once_with(["dispatch/send/0", "dispatch/send/1"])
        sleep.assert_called_once_with(0.2)

    @patch("tests.npu.async_cam_performance.time.sleep")
    def test_combine_sender_uses_attention_receivers_and_moe_local_index(self, sleep):
        store = Mock()
        schedule = FocusSchedule(store, "send", 3, 2, 2, 50, 200, 20)
        schedule("combine_send")
        store.wait.assert_called_once_with(["combine/recv/0", "combine/recv/1"])
        sleep.assert_called_once_with(0.07)

    @patch("tests.npu.async_cam_performance.time.sleep")
    def test_cycle_adds_no_artificial_delay_or_store_traffic(self, sleep):
        store = Mock()
        schedule = FocusSchedule(store, "cycle", 0, 2, 2, 50, 200, 20)
        for operation in selected_ops("cycle"):
            schedule(operation)
        self.assertEqual(store.mock_calls, [])
        sleep.assert_not_called()

    def test_cli_dry_run_needs_no_installed_device_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "tests/npu/async_cam_performance.py",
                "--dry-run",
                "--measure-mode",
                "both",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["modes"], ["send", "recv"])
        self.assertIn("no device execution", report["status"])
        self.assertNotIn("ranks", report)

    def test_cli_rejects_empty_samples_and_nonfinite_delays(self):
        for arguments in (
            ("--iterations", "0"),
            ("--warmup", "-1"),
            ("--receiver-delay-ms", "nan"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-S",
                        "tests/npu/async_cam_performance.py",
                        "--dry-run",
                        *arguments,
                    ],
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
