# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host FFN lifecycle defaults shared by connector implementations."""

import pytest

from afd_plugin.connectors.base import FFNExecutionSession


@pytest.mark.parametrize("shutdown_requested", [False, True])
def test_default_session_stops_locally_without_drain(shutdown_requested):
    session = FFNExecutionSession()
    assert session.keep_serving(shutdown_requested) is not shutdown_requested
    assert not session.requires_drain
    session.before_first_step()
    session.after_step()
    session.drain(lambda: pytest.fail("Default session does not drain remote work"))
    session.finish_receiver()
    session.validate_shutdown()
