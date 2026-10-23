# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import pytest

from afd_plugin.ffn_faas.control import SessionControl, SessionPeerFailureError
from tests.unit.fakes.distributed_store import DisconnectedStore


@pytest.mark.parametrize("operation", ["read", "write", "check"])
def test_store_disconnect_retains_the_operation_and_original_cause(
    operation: str,
) -> None:
    error = RuntimeError("connection closed by owner")
    control = SessionControl(DisconnectedStore(error))
    with pytest.raises(
        SessionPeerFailureError, match=f"Store {operation} failed"
    ) as result:
        if operation == "read":
            control.read("chunk-shape/0")
        elif operation == "write":
            control.write("chunk-shape/0", "payload")
        else:
            control.ready(["chunk-shape/0"])
    assert result.value.__cause__ is error
