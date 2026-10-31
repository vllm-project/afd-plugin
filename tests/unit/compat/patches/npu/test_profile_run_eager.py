# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""The profile pass must not compile, and must leave real routing behind.

The patch under test wraps the Attention model's profile pass in
``disable_compilation``. Two things then have to hold at the same time:

* during the profile pass, the balanced expert ids are still substituted, so the
  memory peak is measured against a spread-out MoE load;
* the first compile happens on the formal warmup, with the profile flag already
  off, so the ids the graph carries are live ones.

A test that only checked the second would pass if the patch disabled the
balancing instead of the compilation, which is the failure mode being avoided.

The module resolves vLLM-Ascend on first use, so these run on a CPU-only host
against stand-ins. They therefore show where the guard sits, not what the NPU
profile pass measures.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from afd_plugin.compat.patches.npu import profile_run_eager

_FAKE_MC2_TOKEN_CAPACITY = 128
_FAKE_MAX_NUM_TOKENS = 4096


class _FakeMoECommType:
    MC2 = "MC2"
    FUSED_MC2 = "FUSED_MC2"


class _FakeModel:
    """Carries the flag ``disable_compilation`` toggles on the real model."""

    def __init__(self) -> None:
        self.do_not_compile = False


class _ParentRunner:
    """Stands in for the runner upstream inherits from."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.flag_seen_by_parent: list[bool] = []

    def get_model(self) -> _FakeModel:
        return self.model

    def profile_run(self) -> None:
        self.flag_seen_by_parent.append(self.model.do_not_compile)


def _install_stand_ins(monkeypatch, *, mc2_capacity=_FAKE_MC2_TOKEN_CAPACITY):
    """Point the patch module at stand-ins, which must happen before first use."""

    profile_run_eager._for_testing_reset()

    @contextmanager
    def fake_disable_compilation(model):
        previous = model.do_not_compile
        model.do_not_compile = True
        try:
            yield
        finally:
            model.do_not_compile = previous

    @contextmanager
    def fake_override(enabled):
        del enabled
        yield

    class FakeNPUModelRunner(_ParentRunner):
        """The class whose ``profile_run`` the patch replaces."""

        def profile_run(self) -> None:  # pragma: no cover - replaced by the patch
            raise AssertionError("the patched method must be used")

    # Present the pinned upstream identity: the installer's ownership check reads
    # these two attributes, and a stand-in should look like the method it replaces.
    FakeNPUModelRunner.profile_run.__module__ = profile_run_eager._UPSTREAM_MODULE
    FakeNPUModelRunner.profile_run.__qualname__ = (
        profile_run_eager._UPSTREAM_QUALNAME
    )

    monkeypatch.setattr(profile_run_eager, "NPUModelRunner", FakeNPUModelRunner)
    monkeypatch.setattr(
        profile_run_eager,
        "disable_compilation",
        fake_disable_compilation,
    )
    monkeypatch.setattr(
        profile_run_eager,
        "get_mc2_tokens_capacity",
        lambda: mc2_capacity,
    )
    monkeypatch.setattr(
        profile_run_eager,
        "select_moe_comm_method",
        lambda *args, **kwargs: _FakeMoECommType.MC2,
    )
    monkeypatch.setattr(profile_run_eager, "MoECommType", _FakeMoECommType)
    monkeypatch.setattr(
        profile_run_eager,
        "override_mrv2_in_profile_run",
        fake_override,
    )
    return FakeNPUModelRunner


def _make_runner(monkeypatch, **kwargs):
    fake_runner_type = _install_stand_ins(monkeypatch, **kwargs)

    runner = fake_runner_type()
    runner.max_num_tokens = _FAKE_MAX_NUM_TOKENS
    runner.vllm_config = object()
    runner.dummy_runs: list[int] = []

    def dummy_run(num_tokens, **call_kwargs):
        del call_kwargs
        runner.dummy_runs.append(num_tokens)

    runner._dummy_run = dummy_run
    return runner


def test_the_whole_profile_pass_runs_with_compilation_off(monkeypatch):
    """The parent profile call, the part upstream leaves unprotected, is guarded."""

    runner = _make_runner(monkeypatch)

    profile_run_eager.profile_run(runner)

    assert runner.flag_seen_by_parent == [True], (
        "the parent profile pass ran outside the guard: "
        f"{runner.flag_seen_by_parent}"
    )
    assert runner.dummy_runs == [_FAKE_MC2_TOKEN_CAPACITY]
    # Released again, so the formal warmup can compile.
    assert runner.model.do_not_compile is False


def test_the_guard_is_released_when_the_profile_raises(monkeypatch):
    runner = _make_runner(monkeypatch)

    def explode(_self):
        raise RuntimeError("profile exploded")

    # ``super(NPUModelRunner, self)`` starts after the patched class, so the parent
    # is what has to fail for the guard's cleanup path to be exercised.
    monkeypatch.setattr(_ParentRunner, "profile_run", explode)

    with pytest.raises(RuntimeError, match="profile exploded"):
        profile_run_eager.profile_run(runner)

    assert runner.model.do_not_compile is False, (
        "a failed profile left compilation disabled for the process"
    )


def test_no_mc2_dummy_run_when_the_capacity_does_not_apply(monkeypatch):
    """The guard is not conditional on the MC2 branch being taken."""

    runner = _make_runner(monkeypatch, mc2_capacity=None)

    profile_run_eager.profile_run(runner)

    assert runner.dummy_runs == []
    assert runner.flag_seen_by_parent == [True]


def test_the_patch_is_idempotent(monkeypatch):
    fake_runner_type = _install_stand_ins(monkeypatch)

    profile_run_eager.apply_profile_run_eager_patch()
    installed_once = fake_runner_type.profile_run
    assert installed_once is profile_run_eager.profile_run

    profile_run_eager.apply_profile_run_eager_patch()
    assert fake_runner_type.profile_run is installed_once


def test_the_install_refuses_a_foreign_implementation(monkeypatch):
    fake_runner_type = _install_stand_ins(monkeypatch)

    def foreign_profile_run(self) -> None:  # pragma: no cover - never called
        raise AssertionError("the foreign method must not be replaced")

    monkeypatch.setattr(fake_runner_type, "profile_run", foreign_profile_run)

    with pytest.raises(
        profile_run_eager.ProfileRunPatchConflict,
        match="is owned by",
    ):
        profile_run_eager.apply_profile_run_eager_patch()

    assert fake_runner_type.profile_run is foreign_profile_run


def test_the_install_refuses_to_overwrite_after_being_replaced(monkeypatch):
    fake_runner_type = _install_stand_ins(monkeypatch)
    profile_run_eager.apply_profile_run_eager_patch()

    def foreign_profile_run(self) -> None:  # pragma: no cover - never called
        raise AssertionError("the foreign method must not be replaced")

    monkeypatch.setattr(fake_runner_type, "profile_run", foreign_profile_run)

    with pytest.raises(
        profile_run_eager.ProfileRunPatchConflict,
        match="replaced after this patch was installed",
    ):
        profile_run_eager.apply_profile_run_eager_patch()

    assert fake_runner_type.profile_run is foreign_profile_run
