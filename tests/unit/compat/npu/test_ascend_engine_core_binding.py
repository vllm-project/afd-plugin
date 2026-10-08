# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check AFD's early Ascend config binding for DP EngineCore children."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

from vllm.v1.engine.core import EngineCoreProc  # noqa: E402
from vllm_ascend.patch.platform import patch_engine_core  # noqa: E402

from afd_plugin.compat.npu import runtime  # noqa: E402
from afd_plugin.compat.patches.npu import (
    ascend_config as afd_ascend_config,  # noqa: E402
)
from afd_plugin.compat.patches.npu.ascend_config import (  # noqa: E402
    run_afd_ascend_engine_core,
)


@pytest.mark.parametrize(
    ("role", "is_async", "expects_binding"),
    [
        ("attention", False, True),
        ("attention", True, False),
        ("ffn", False, True),
        ("ffn", True, True),
    ],
)
def test_early_ascend_config_binding_preserves_async_attention(
    monkeypatch, role, is_async, expects_binding
):
    config = SimpleNamespace()
    original = staticmethod(lambda *_args, **_kwargs: None)
    monkeypatch.setattr(EngineCoreProc, "run_engine_core", original)
    monkeypatch.setattr(
        runtime,
        "parse_optional_afd_config",
        lambda _config, validate: SimpleNamespace(role=role),
    )
    monkeypatch.setattr(runtime, "is_afd_async_dp", lambda _config: is_async)

    applied = runtime.apply_afd_ascend_engine_core_config_patch_if_needed(config)

    assert applied is expects_binding
    assert EngineCoreProc.run_engine_core is (
        run_afd_ascend_engine_core if expects_binding else original.__func__
    )


def test_early_ascend_config_binding_skips_non_afd(monkeypatch):
    original = staticmethod(lambda *_args, **_kwargs: None)
    monkeypatch.setattr(EngineCoreProc, "run_engine_core", original)
    monkeypatch.setattr(
        runtime, "parse_optional_afd_config", lambda *_args, **_kwargs: None
    )

    assert not runtime.apply_afd_ascend_engine_core_config_patch_if_needed(
        SimpleNamespace()
    )
    assert EngineCoreProc.run_engine_core is original.__func__


def test_child_installs_afd_factory_before_ascend_config_init(monkeypatch):
    calls = []
    monkeypatch.setattr(
        afd_ascend_config,
        "apply_afd_ascend_config_patch",
        lambda: calls.append("afd_factory"),
    )

    def native_entry(*args, dp_rank, local_dp_rank, **kwargs):
        calls.append("ascend_entry")
        assert calls == ["afd_factory", "ascend_entry"]
        return (args, dp_rank, local_dp_rank, kwargs)

    monkeypatch.setattr(patch_engine_core, "_run_engine_core_patch_func", native_entry)

    assert run_afd_ascend_engine_core(
        "worker", dp_rank=1, local_dp_rank=0, vllm_config="config"
    ) == (("worker",), 1, 0, {"vllm_config": "config"})
