# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Byte reordering, independent of device format casting."""

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.connectors.npu.async_mega.mega_weights import _nz_bytes  # noqa: E402


def test_nz_packing_preserves_checkpoint_bytes():
    data = torch.randint(0, 256, (2, 32, 48), dtype=torch.uint8)
    packed = _nz_bytes(data)
    assert packed.is_contiguous()
    for e in range(2):
        for k in range(3):
            for n in range(2):
                tile = data[e, n * 16 : (n + 1) * 16, k * 16 : (k + 1) * 16]
                assert torch.equal(packed[e, k, n], tile)
