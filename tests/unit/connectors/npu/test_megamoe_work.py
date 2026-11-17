# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded work descriptors and ring-sequence ownership."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_independent_dp_and_slow_readers():
    # Load only the protocol module. The connector package imports vLLM.
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/megamoe_work.py"
    )
    spec = importlib.util.spec_from_file_location("work", path)
    work = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(work)
    store = torch.distributed.HashStore()

    def queue():
        return work.ExpertWorkQueue(store, 2, 2, lambda n: (n - 1) % 32 + 1)

    writers = [queue(), queue()]
    readers = [queue(), queue()]
    for iteration in range(1, 70):
        active = [0, 1] if iteration % 3 == 0 else [0]
        expected = []
        for dp in active:
            sequence = writers[dp].next_sequence(dp)
            layer = iteration % 43
            writers[dp].publish(dp, layer, sequence)
            expected.append((dp, layer, sequence))
        for ffn in (1, 0):
            items = [readers[ffn].poll(ffn) for _ in active]
            assert sorted(items) == expected
            assert readers[ffn].poll(ffn) is None
        assert store.num_keys() == 0

    store.set("work/0/0", "0 1")
    with pytest.raises(RuntimeError, match="Out-of-order"):
        readers[0].poll(0)
