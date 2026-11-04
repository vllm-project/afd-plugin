# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Run the Attention model's profile pass without compilation.

Upstream source: vllm_ascend/worker/v2/model_runner.py (8d4409d),
``NPUModelRunner.profile_run``.

Patch reason: the profile pass runs with ``in_profile_run=True`` so that MoE load
is spread over every expert instead of whatever the startup inputs happen to hit.
Only the extra MC2 dummy run is wrapped in ``disable_compilation``; the standard
``super().profile_run()`` that follows is not, so it can be the first call to
compile the Attention model. Compilation traces ``in_profile_run`` as a constant,
and the compiled wrapper carries no guard and does not re-trace when the flag goes
back to False, so the balanced expert ids stay substituted for real requests and
override the live routing.

Patch functionality: hold compilation off for the whole profile pass. The
balanced routing then stays an eager-only behaviour, and the first compile
happens on the formal warmup, where the flag is already False.

Signature: matches upstream; no added parameters.

The vLLM-Ascend names are resolved on first use rather than at import time. The
patch is installed inside a worker process and needs those modules, but keeping
the import out of module scope lets a CPU-only host import this module and run the
guard's behaviour against stand-ins.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

# Filled by ``_ensure_upstream_imports`` on first use, or by a test that supplies
# its own stand-ins. Never reset once set.
NPUModelRunner: Any = None
disable_compilation: Any = None
get_mc2_tokens_capacity: Any = None
override_mrv2_in_profile_run: Any = None
select_moe_comm_method: Any = None
MoECommType: Any = None


def _ensure_upstream_imports() -> None:
    """Resolve the vLLM-Ascend names once, leaving supplied stand-ins alone."""

    global NPUModelRunner, disable_compilation, get_mc2_tokens_capacity
    global override_mrv2_in_profile_run, select_moe_comm_method, MoECommType

    if NPUModelRunner is not None:
        return

    from vllm_ascend.ascend_forward_context import (
        MoECommType as upstream_moe_comm_type,
    )
    from vllm_ascend.ascend_forward_context import (
        get_mc2_tokens_capacity as upstream_get_mc2_tokens_capacity,
    )
    from vllm_ascend.ascend_forward_context import (
        override_mrv2_in_profile_run as upstream_override_mrv2_in_profile_run,
    )
    from vllm_ascend.ascend_forward_context import (
        select_moe_comm_method as upstream_select_moe_comm_method,
    )
    from vllm_ascend.worker.utils import (
        disable_compilation as upstream_disable_compilation,
    )
    from vllm_ascend.worker.v2.model_runner import (
        NPUModelRunner as upstream_model_runner,
    )

    MoECommType = upstream_moe_comm_type
    get_mc2_tokens_capacity = upstream_get_mc2_tokens_capacity
    override_mrv2_in_profile_run = upstream_override_mrv2_in_profile_run
    select_moe_comm_method = upstream_select_moe_comm_method
    disable_compilation = upstream_disable_compilation
    NPUModelRunner = upstream_model_runner


# Patch reason: see the module docstring. The AFD difference is the guard around
# the whole body; everything inside is the upstream body unchanged.
# Patch functionality: forward with compilation disabled for the whole pass.
# Signature: matches upstream; no added parameters.
def profile_run(self) -> None:
    _ensure_upstream_imports()

    # ### PATCH START: keep the profile pass out of compilation
    with disable_compilation(self.get_model()):
        # ### PATCH END: keep the profile pass out of compilation
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and self.max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                # Use a call-scoped bypass because skip_compiled would require runner-specific ForwardContext plumbing.
                with disable_compilation(self.get_model()):
                    self._dummy_run(
                        mc2_tokens_capacity,
                        skip_attn=True,
                        skip_eplb=True,
                        is_profile=True,
                    )
            super(NPUModelRunner, self).profile_run()


_UPSTREAM_MODULE = "vllm_ascend.worker.v2.model_runner"
_UPSTREAM_QUALNAME = "NPUModelRunner.profile_run"
_ORIGINAL_ATTR = "_afd_plugin_profile_run_eager_original"


class ProfileRunPatchConflict(RuntimeError):
    """Raised when ``NPUModelRunner.profile_run`` is not this patch's to replace."""


def apply_profile_run_eager_patch() -> None:
    """Install the patch, raising if another implementation owns the target.

    The install must not degrade quietly. Whether the Attention model compiles
    while ``in_profile_run`` is true decides whether the balanced ids get traced
    into a guard-free graph, so a refused install that the caller ignores would
    leave exactly the silent failure this patch exists to remove. The caller lets
    the exception abort initialization instead.
    """

    _ensure_upstream_imports()

    if NPUModelRunner.profile_run is profile_run:
        return

    if getattr(NPUModelRunner, _ORIGINAL_ATTR, None) is not None:
        raise ProfileRunPatchConflict(
            "NPUModelRunner.profile_run was replaced after this patch was installed",
        )

    current = NPUModelRunner.profile_run

    # Partial detection only. A patch that used functools.wraps keeps these two
    # attributes pointing at the upstream names, so a pass here does not prove the
    # body is still the pinned one; it only rules out a plainly foreign install.
    if (
        current.__module__ != _UPSTREAM_MODULE
        or current.__qualname__ != _UPSTREAM_QUALNAME
    ):
        raise ProfileRunPatchConflict(
            "NPUModelRunner.profile_run is owned by "
            f"{current.__module__}.{current.__qualname__}, not by the pinned "
            "vLLM-Ascend implementation",
        )

    setattr(NPUModelRunner, _ORIGINAL_ATTR, current)
    NPUModelRunner.profile_run = profile_run


def _for_testing_reset() -> None:
    """Drop resolved upstream objects. Only tests use this."""

    global NPUModelRunner, disable_compilation, get_mc2_tokens_capacity
    global override_mrv2_in_profile_run, select_moe_comm_method, MoECommType

    if NPUModelRunner is not None and hasattr(NPUModelRunner, _ORIGINAL_ATTR):
        delattr(NPUModelRunner, _ORIGINAL_ATTR)

    NPUModelRunner = None
    disable_compilation = None
    get_mc2_tokens_capacity = None
    override_mrv2_in_profile_run = None
    select_moe_comm_method = None
    MoECommType = None


__all__ = [
    "MoECommType",
    "ProfileRunPatchConflict",
    "NPUModelRunner",
    "apply_profile_run_eager_patch",
    "disable_compilation",
    "get_mc2_tokens_capacity",
    "override_mrv2_in_profile_run",
    "profile_run",
    "select_moe_comm_method",
]

_ = ModuleType  # keep the import meaningful for type-only readers
