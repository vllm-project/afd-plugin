# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.forward_context import (  # noqa: E402
    get_forward_context,
    override_forward_context,
)
from vllm.v1.worker.workspace import use_workspace_lane  # noqa: E402

from afd_plugin.compat.patches import gpu_deepseek_v4_dbo as patches  # noqa: E402
from tests.unit.fakes.deepseek_v4 import WorkspaceCase  # noqa: E402


@pytest.mark.parametrize("prior_override", [False, True])
def test_v4_workspace_reservation_error_restores_owner_and_forward_context(
    monkeypatch, prior_override
):
    error = ValueError("native reservation failed")
    case = WorkspaceCase(monkeypatch, num_lanes=2, error=error)
    lane = 0 if prior_override else 1
    case.ubatch = 0 if prior_override else 1
    if prior_override:
        case.install_prior_slot_override()
    # Public context/returned-storage observations establish restoration; no
    # private selector/array identity assertions are used.
    with use_workspace_lane(lane), override_forward_context(case.outer_context):
        outside = get_forward_context()
        before = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        before.fill_(71)
        with pytest.raises(ValueError, match="native reservation failed") as caught:
            patches._reserve_flashmla_ubatch_workspace(case.model, case.config)
        after = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        assert caught.value is error
        assert get_forward_context() is outside
        assert after.data_ptr() == before.data_ptr()
        assert after.item() == 71


def test_v4_workspace_reservation_success_restores_instance_owner_and_context(
    monkeypatch,
):
    case = WorkspaceCase(monkeypatch, num_lanes=2)
    case.install_prior_slot_override()
    # Fully reserve the original selected owner first so the helper must keep
    # this returned storage while routing the remaining independent slots.
    before = case.request(0, 0)
    before.fill_(37)
    with override_forward_context(case.outer_context):
        outside = get_forward_context()
        patches._reserve_flashmla_ubatch_workspace(case.model, case.config)
        # Existing override selects its own owner, not the helper's last slot.
        case.ubatch = 0
        after = case.manager.get_simultaneous(((1,), torch.uint8))[0]
        assert get_forward_context() is outside
        assert after.data_ptr() == before.data_ptr()
        assert after.item() == 37
