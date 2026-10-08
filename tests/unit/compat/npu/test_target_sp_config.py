# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check the target Ascend config switch before AFD's explicit-worker fix."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import cast

import pytest

from afd_plugin.compat.npu.runtime_config import fix_all2all_backend_for_afd


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "worker_cls",
    ["auto", "afd_plugin.v1.worker.npu.attention_worker.AFDNPUAttentionWorker"],
)
def test_target_config_switch_survives_afd_worker_fix(monkeypatch, enabled, worker_cls):
    ascend = pytest.importorskip("vllm_ascend.ascend_config")
    from vllm.config import ParallelConfig

    parallel = ParallelConfig(
        enable_expert_parallel=True,
        tensor_parallel_size=4,
        data_parallel_size=2,
        all2all_backend="allgather_reducescatter",
        worker_cls=worker_cls,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        additional_config={},
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
    )
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1" if enabled else "0")
    # Execute the actual pinned upstream switch without initializing hardware,
    # scheduler or quantization settings in the rest of derive_and_validate.
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(ascend.AscendConfig.derive_and_validate))
    )
    statements = cast(ast.FunctionDef, tree.body[0]).body
    start = next(
        i
        for i, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "flashcomm_explicitly_enabled"
            for target in node.targets
        )
    )
    end = next(
        i
        for i in range(start, len(statements))
        if isinstance(statements[i], ast.If)
        and ast.unparse(cast(ast.If, statements[i]).test) == "not effective_flashcomm"
    )
    namespace = dict(
        vars(ascend),
        vc=config,
        vllm_config=config,
        self=SimpleNamespace(enable_dsa_cp=False),
    )
    exec(
        compile(
            ast.Module(body=statements[start : end + 1], type_ignores=[]),
            "<target-ascend-SP-config>",
            "exec",
        ),
        namespace,
    )
    assert parallel.use_sequence_parallel_moe is enabled
    fix_all2all_backend_for_afd(config)
    assert parallel.use_sequence_parallel_moe is enabled
    assert parallel.all2all_backend == (
        "allgather_reducescatter" if enabled else "flashinfer_all2allv"
    )
