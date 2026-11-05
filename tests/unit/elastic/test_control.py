# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Exercise the real wire schema without requiring a device PyTorch build."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def metadata(monkeypatch):
    torch = types.ModuleType("torch")
    # metadata.py reads these module-level attributes at import time.
    torch.float8_e4m3fn = "float8_e4m3fn"  # type: ignore[attr-defined]
    torch.int32 = "int32"  # type: ignore[attr-defined]
    torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    torch.dtype = type  # type: ignore[attr-defined]
    context = types.ModuleType("vllm.forward_context")
    context.DPMetadata = type("DPMetadata", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    path = Path(__file__).parents[3] / "afd_plugin/connectors/metadata.py"
    spec = importlib.util.spec_from_file_location("_afd_metadata_cpu_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_stop_round_trip(metadata):
    payload = metadata.AFDControlPayload({}, False, False, stop=True)
    encoded = metadata.encode_control_payload(payload)
    decoded = metadata.decode_control_payload(encoded)
    assert decoded.stop
    assert decoded.dp_metadata_list == {}
    assert not decoded.is_graph_capturing


def test_old_wire_payload_is_still_a_batch(metadata):
    payload = metadata.decode_control_payload(b'{"dp_metadata_list":{}}')
    assert not payload.stop


def test_stop_cannot_contain_batches(metadata):
    with pytest.raises(ValueError, match="STOP"):
        metadata.AFDControlPayload({0: None}, False, False, stop=True)
