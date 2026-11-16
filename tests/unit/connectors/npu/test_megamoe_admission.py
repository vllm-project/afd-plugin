# SPDX-License-Identifier: Apache-2.0
"""FFN admission regressions, runnable without vLLM or an NPU."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

path = Path(__file__).resolve().parents[4] / "afd_plugin/connectors/npu/megamoe_work.py"
spec = importlib.util.spec_from_file_location("megamoe_work", path)
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)


def queue(published=None):
    keys = {}
    store = Mock()
    store.set = keys.__setitem__
    store.get = keys.__getitem__
    store.check = lambda names: all(key in keys for key in names)
    store.delete_key = keys.__delitem__
    return work.ExpertWorkQueue(
        store, len(published) if published is not None else 1, 1,
        lambda n: (n - 1) % 32 + 1,
        read_published=Mock(return_value=published) if published is not None else None,
    )


class TestMegaMoEAdmission(unittest.TestCase):
    def test_single_window_claims_without_device_snapshot(self):
        reader = queue()
        reader.store.set("work/0/0", "42 1")
        self.assertEqual(reader.poll(0), (0, 42, 1))
        self.assertIsNone(reader.poll(0))

    def test_two_lanes_at_batch_boundary(self):
        lane0 = queue([[0], [0], [26]])
        lane1 = queue([[0], [0], [4]])
        lane0.last[2], lane1.last[2] = 26, 3
        lane0.store.set("work/2/0", "0 27")
        lane1.store.set("work/2/0", "42 4")
        self.assertIsNone(lane0.poll(0))
        self.assertEqual(lane0.last[2], 26)
        self.assertEqual(lane1.poll(0), (2, 42, 4))
        lane0.read_published.return_value[2][0] = 27
        self.assertEqual(lane0.poll(0), (2, 0, 27))

    def test_scan_past_unready_tp_group_and_wrap(self):
        reader = queue([[1, 32], [1, 1]])
        reader.last = [32, 0]
        reader.store.set("work/0/0", "0 1")
        reader.store.set("work/1/0", "7 1")
        self.assertEqual(reader.poll(0), (1, 7, 1))
        reader.read_published.assert_called_once_with()
        reader.read_published.return_value[0][1] = 1
        self.assertEqual(reader.poll(0), (0, 0, 1))
        reader.read_published.reset_mock()
        self.assertIsNone(reader.poll(0))
        reader.read_published.assert_not_called()


if __name__ == "__main__":
    unittest.main()
