# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone FFN service entry with the MRV2 FFN execution probe.

The service builds no vLLM Worker, so it cannot load ``--worker-extension-cls``
and has no health RPC to report through. This entry imports the same probe
(which patches the shared GPU FFN runner), polls the completed-window marker on
a daemon thread in place of the health RPC, then runs the unchanged
``afd_plugin.ffn_faas`` CLI. The E2E harness launches it under ``torchrun``
only for cases with graph/DBO evidence checks.
"""

from __future__ import annotations

import threading
import time

from afd_plugin.ffn_faas.__main__ import main
from tests.e2e.mrv2_evidence import flush_ffn_service_evidence

POLL_INTERVAL_S = 0.2


def _poll_completed_window() -> None:
    while True:
        time.sleep(POLL_INTERVAL_S)
        flush_ffn_service_evidence()


if __name__ == "__main__":
    threading.Thread(
        target=_poll_completed_window, name="afd-e2e-ffn-evidence", daemon=True
    ).start()
    raise SystemExit(main())
