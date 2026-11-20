# SPDX-License-Identifier: Apache-2.0
"""FFN admission regressions, runnable without vLLM or an NPU."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

path = Path(__file__).resolve().parents[4] / "afd_plugin/connectors/npu/mega_work.py"
spec = importlib.util.spec_from_file_location("mega_work", path)
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
        store,
        len(published) if published is not None else 1,
        1,
        lambda iteration: (iteration - 1) % 32 + 1,
        read_published=(lambda: published) if published is not None else None,
    )


def advance(reader, source, count):
    for iteration in range(1, count + 1):
        sequence = (iteration - 1) % 32 + 1
        reader.store.set(f"work/{source}/0", f"42 {sequence}")
        assert reader.poll(0) == (source, 42, sequence)


class TestMegaAdmission(unittest.TestCase):
    def test_publication_wraps_and_rejects_skipped_generation(self):
        writer = queue()
        for iteration in range(1, 98):
            sequence = (iteration - 1) % 32 + 1
            writer.publish(0, 42, sequence)
            self.assertEqual(writer.store.get("work/0/0"), f"42 {sequence}")
            writer.store.delete_key("work/0/0")
        with self.assertRaisesRegex(RuntimeError, "Out-of-order"):
            writer.publish(0, 42, 3)
        self.assertFalse(writer.store.check(["work/0/0"]))

    def test_single_window_claims_without_device_snapshot(self):
        reader = queue()
        reader.store.set("work/0/0", "42 1")
        self.assertEqual(reader.poll(0), (0, 42, 1))
        self.assertIsNone(reader.poll(0))

    def test_scan_past_unready_tp_group_at_wrap(self):
        published = [[0, 0], [1, 1]]
        reader = queue(published)
        for sequence in range(1, 33):
            published[0][:] = [sequence, sequence]
            reader.store.set("work/0/0", f"42 {sequence}")
            self.assertEqual(reader.poll(0), (0, 42, sequence))
        published[0][:] = [1, 32]
        reader.store.set("work/0/0", "0 1")
        reader.store.set("work/1/0", "7 1")
        self.assertEqual(reader.poll(0), (1, 7, 1))
        self.assertIsNone(reader.poll(0))
        self.assertTrue(reader.store.check(["work/0/0"]))
        published[0][1] = 1
        self.assertEqual(reader.poll(0), (0, 0, 1))
        self.assertIsNone(reader.poll(0))

    def test_independent_lane_generations(self):
        lane0, lane1 = queue(), queue()
        advance(lane0, 0, 32)
        advance(lane1, 0, 3)
        lane0.store.set("work/0/0", "0 1")
        lane1.store.set("work/0/0", "42 4")
        self.assertEqual(lane1.poll(0), (0, 42, 4))
        self.assertEqual(lane0.poll(0), (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
