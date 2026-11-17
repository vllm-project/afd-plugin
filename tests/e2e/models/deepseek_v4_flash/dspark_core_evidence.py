# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Request-window DSpark observations. Never use these probes for timing runs."""

import json
import os
from functools import wraps
from pathlib import Path
from typing import Any

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager

from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from tests.e2e.models.deepseek_v4_flash.dspark_evidence import (
    DSparkEagerEvidence,
    snapshot,
)

# Only the separate instrumented deployment changes FFN graph submission grouping.
if os.environ.get("AFD_DSPARK_GRAPH_NUMERICS") == "1":
    import tests.e2e.models.deepseek_v4_flash.dspark_ffn_graph_evidence  # noqa: F401


class DSparkCoreEvidence(DSparkEagerEvidence):
    pass


def publish(directory: Path, rank: int, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{rank}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)


def install_attention_observers() -> None:
    """The HTTP collective RPC reaches one DP, so observe each worker locally."""
    directory = Path(os.environ["AFD_DSPARK_EVIDENCE_DIR"])
    marker = directory / "requests-active"
    execute = AFDNPUAttentionModelRunnerV2.execute_model
    target = ModelAclGraphManager.run_fullgraph
    draft = DFlashAclGraphManager.run_fullgraph
    events: dict[str, list[dict[str, Any]]] = {"target": [], "draft": []}

    @wraps(execute)
    def executed(runner, scheduler_output, *args, **kwargs):
        result = execute(runner, scheduler_output, *args, **kwargs)
        if (
            marker.exists()
            and scheduler_output.total_num_scheduled_tokens > 0
            and not kwargs.get("dummy_run", False)
            and not kwargs.get("is_profile", False)
            and not getattr(runner, "_dspark_attention_recorded", False)
        ):
            record = snapshot(runner)
            publish(directory / "attention-ranks", record["rank"], record)
            runner._dspark_attention_recorded = True
        return result

    def observe(kind, runner, desc):
        if not marker.exists() or events[kind]:
            return
        metadata = runner._afd_pending_metadata
        if metadata is None:
            return
        real = list(metadata.tokens_unpadded_lens)
        if desc.num_tokens <= 0 or sum(real) <= 0:
            return
        events[kind].append({"tokens": desc.num_tokens, "real": real})
        rank = runner.connector.world_rank
        publish(directory / "attention-graph", rank, {"rank": rank, "events": events})

    @wraps(target)
    def target_replay(manager, desc):
        result = target(manager, desc)
        observe("target", manager.model_runner, desc)
        return result

    @wraps(draft)
    def draft_replay(manager, desc):
        result = draft(manager, desc)
        observe("draft", manager.speculator.afd_runner, desc)
        return result

    AFDNPUAttentionModelRunnerV2.execute_model = executed
    ModelAclGraphManager.run_fullgraph = target_replay
    DFlashAclGraphManager.run_fullgraph = draft_replay


install_attention_observers()
