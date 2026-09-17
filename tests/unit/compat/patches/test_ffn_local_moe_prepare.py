# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role gating for the FFN-local MoE prepare/finalize selector.

AFD pre-routes rows to the FFN rank that owns them, so the FFN role must not
run another all-to-all. Every other process -- the Attention role, and plain
vLLM -- has to reach upstream's selector untouched: this patch is installed
process-wide, so getting the gate wrong changes runs that have nothing to do
with AFD.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.compat.patches import ffn_local_moe_prepare as patch

ALL_KERNEL_FLAGS = (
    "use_deepep_ht_kernels",
    "use_deepep_ll_kernels",
    "use_deepep_v2_kernels",
    "use_fi_nvl_two_sided_kernels",
    "use_fi_nvl_one_sided_kernels",
    "use_nixl_ep_kernels",
    "use_mori_kernels",
)


def _moe(**overrides):
    flags = dict.fromkeys(ALL_KERNEL_FLAGS, False)
    flags.update(overrides)
    return SimpleNamespace(moe_parallel_config=SimpleNamespace(**flags))


@pytest.fixture
def upstream(monkeypatch):
    calls: list[tuple] = []

    def selector(*args, **kwargs):
        calls.append((args, kwargs))
        return "upstream"

    monkeypatch.setattr(patch, "_UPSTREAM_SELECTOR", selector)
    return calls


@pytest.fixture
def local(monkeypatch):
    calls: list[dict] = []

    def local_prepare(**kwargs):
        calls.append(kwargs)
        return "local"

    monkeypatch.setattr(patch, "make_moe_prepare_and_finalize_no_dp_ep", local_prepare)
    return calls


def test_a_non_ffn_process_reaches_upstream_unchanged(monkeypatch, upstream, local):
    monkeypatch.setattr(patch, "_is_afd_ffn_role", lambda: False)

    result = patch.maybe_make_prepare_finalize(_moe(), use_monolithic=True)

    assert result == "upstream"
    assert local == []
    # The arguments must arrive exactly as given; this is a pass-through.
    assert upstream[0][1] == {"use_monolithic": True}


def test_the_ffn_role_prepares_locally(monkeypatch, upstream, local):
    monkeypatch.setattr(patch, "_is_afd_ffn_role", lambda: True)

    result = patch.maybe_make_prepare_finalize(_moe(), use_monolithic=True)

    assert result == "local"
    assert upstream == []
    assert local == [{"use_monolithic": True}]


@pytest.mark.parametrize("flag", ALL_KERNEL_FLAGS)
def test_kernels_that_own_their_dispatch_are_left_alone(
    monkeypatch,
    upstream,
    local,
    flag,
):
    monkeypatch.setattr(patch, "_is_afd_ffn_role", lambda: True)

    result = patch.maybe_make_prepare_finalize(_moe(**{flag: True}))

    assert result == "upstream", f"{flag} must keep its own collective"
    assert local == []


def test_the_moe_may_arrive_as_a_keyword(monkeypatch, upstream, local):
    monkeypatch.setattr(patch, "_is_afd_ffn_role", lambda: True)

    assert patch.maybe_make_prepare_finalize(moe=_moe()) == "local"


def test_installing_twice_rebinds_once():
    # Already installed at import; a second call must be a no-op rather than
    # wrapping the wrapper.
    installed = patch.all2all_utils_module.maybe_make_prepare_finalize

    patch.apply_local_moe_prepare()

    assert patch.all2all_utils_module.maybe_make_prepare_finalize is installed
    assert installed is patch.maybe_make_prepare_finalize


def test_an_unresolvable_config_is_not_the_ffn_role(monkeypatch):
    def boom():
        raise RuntimeError("no vllm config in this process")

    monkeypatch.setattr(patch, "get_current_vllm_config", boom)

    assert patch._is_afd_ffn_role() is False
