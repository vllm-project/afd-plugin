# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contracts for native dispatch and execute boundaries, without an NPU runtime."""

from __future__ import annotations

import ast
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER_PATH = REPO_ROOT / "afd_plugin/v1/worker/npu/attention_model_runner_v2.py"
PATCH_PATH = REPO_ROOT / "afd_plugin/compat/patches/npu/mrv2_ubatching.py"


class _GraphMode(Enum):
    NONE = 0
    FULL = 1
    FULL_DECODE_ONLY = 2


@dataclass(frozen=True)
class _DispatchResult:
    cg_mode: _GraphMode
    num_tokens: int
    num_reqs: int = 1
    num_ubatches: int = 1


@dataclass(frozen=True)
class _Agreement:
    num_tokens_across_dp: Tensor
    epoch: str = "existing agreement"


def _load_functions(path, names, namespace):
    functions: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(
            ast.Module(body=future + functions, type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )


@pytest.fixture
def dispatch_contract():
    # Native supplies the selected descriptor and agreement. The strict keyword
    # boundary rejects the retired dp_sync name instead of silently swallowing it.
    def dispatch(
        cudagraph_manager,
        num_reqs,
        num_tokens,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=None,
        need_eager=False,
        num_active_loras=0,
        parallel_config=None,
        allow_ubatching=False,
        uniform_decode=False,
        dp_sync_state=None,
    ):
        return cudagraph_manager, dp_sync_state

    native = SimpleNamespace(dispatch_cg_and_sync_dp=dispatch)
    namespace = {
        "contextmanager": contextmanager,
        "replace": replace,
        "CUDAGraphMode": _GraphMode,
        "v2_model_runner": native,
        "model_runner": native,
        "check_ubatch_thresholds": lambda config, *args: config.eligible,
        "get_num_ubatches": lambda config: config.num_ubatches,
    }
    _load_functions(RUNNER_PATH, {"_use_camp2p_dp_padding"}, namespace)
    _load_functions(PATCH_PATH, {"use_async_cam_dbo_dispatch"}, namespace)
    return SimpleNamespace(
        native=native,
        camp2p=namespace["_use_camp2p_dp_padding"],
        async_dbo=namespace["use_async_cam_dbo_dispatch"],
    )


@pytest.mark.parametrize("mode", [_GraphMode.NONE, _GraphMode.FULL])
@pytest.mark.parametrize("fail", [False, True])
def test_camp2p_reuses_native_agreement_and_restores_dispatch(
    dispatch_contract, mode, fail
):
    native = dispatch_contract.native
    original = native.dispatch_cg_and_sync_dp
    counts = [3, 7] if mode == _GraphMode.NONE else [8, 8]
    descriptor = _DispatchResult(mode, counts[0])
    agreement = _Agreement(torch.tensor(counts))
    expected_failure = (
        pytest.raises(RuntimeError, match="request failed") if fail else nullcontext()
    )
    with expected_failure, dispatch_contract.camp2p(2, 1):
        result, reused = native.dispatch_cg_and_sync_dp(
            descriptor, 1, counts[0], 1, 2, 0, dp_sync_state=agreement
        )
        assert result.num_tokens == (7 if mode == _GraphMode.NONE else 8)
        assert result.num_reqs == descriptor.num_reqs
        assert reused.epoch == agreement.epoch
        assert reused.num_tokens_across_dp.tolist() == (
            [7, 7] if mode == _GraphMode.NONE else counts
        )
        assert agreement.num_tokens_across_dp.tolist() == counts
        if mode == _GraphMode.FULL:
            assert result is descriptor
            assert reused is agreement
        if fail:
            raise RuntimeError("request failed")
    assert native.dispatch_cg_and_sync_dp is original


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize(
    "tokens, eligible, allow_ubatching, local_ubatches",
    [(1, True, True, 1), (7, True, True, 2), (7, False, True, 1), (7, True, False, 1)],
)
def test_async_dbo_preserves_native_agreement_and_dp1_eligibility(
    dispatch_contract, dp_size, tokens, eligible, allow_ubatching, local_ubatches
):
    native = dispatch_contract.native
    original = native.dispatch_cg_and_sync_dp
    descriptor = _DispatchResult(_GraphMode.NONE, tokens)
    agreement = _Agreement(torch.full((dp_size,), tokens)) if dp_size > 1 else None
    parallel = SimpleNamespace(eligible=eligible, num_ubatches=2)
    with dispatch_contract.async_dbo():
        result, reused = native.dispatch_cg_and_sync_dp(
            descriptor,
            1,
            tokens,
            1,
            dp_size,
            0,
            parallel_config=parallel,
            allow_ubatching=allow_ubatching,
            uniform_decode=True,
            dp_sync_state=agreement,
        )
        assert result.num_ubatches == (local_ubatches if dp_size == 1 else 1)
        assert result.num_tokens == tokens
        assert reused is agreement
    assert native.dispatch_cg_and_sync_dp is original


@pytest.mark.parametrize("scope_name", ["camp2p", "async_dbo"])
def test_native_dispatch_failure_restores_binding(dispatch_contract, scope_name):
    native = dispatch_contract.native

    def failed_dispatch(*args, **kwargs):
        raise RuntimeError("native dispatch failed")

    native.dispatch_cg_and_sync_dp = failed_dispatch
    scope = (
        dispatch_contract.camp2p(2, 1)
        if scope_name == "camp2p"
        else dispatch_contract.async_dbo()
    )
    with pytest.raises(RuntimeError, match="native dispatch failed"), scope:
        native.dispatch_cg_and_sync_dp(None, 1, 3, 1, 2, 0, dp_sync_state=None)
    assert native.dispatch_cg_and_sync_dp is failed_dispatch


@pytest.fixture
def execute_contract():
    provider: ContextVar[Callable[[SimpleNamespace], None] | None] = ContextVar(
        "test_afd_provider", default=None
    )
    skip_dp = ContextVar("test_skip_dp", default=False)

    @contextmanager
    def metadata_scope(installer):
        token = provider.set(installer)
        try:
            yield
        finally:
            provider.reset(token)

    @contextmanager
    def dp_scope():
        token = skip_dp.set(True)
        try:
            yield
        finally:
            skip_dp.reset(token)

    class NativeRunner:
        vllm_config: SimpleNamespace
        afd_config: SimpleNamespace
        prof: None
        cudagraph_manager: None
        ubatch_runner: None
        _afd_pending_metadata: None
        _afd_suppress_metadata_send: bool
        _is_warmup: bool
        _afd_is_graph_capturing: bool
        _afd_is_graph_replaying: bool
        _afd_is_profile: bool
        install_afd_metadata_on_forward_context: Callable[[SimpleNamespace], None]

        def execute_model(
            self,
            scheduler_output,
            intermediate_tensors=None,
            dummy_run=False,
            skip_attn_for_dummy_run=False,
            is_profile=False,
            context_len=0,
            valid_dummy_state_slots=False,
            randomize_inputs=False,
        ):
            context = SimpleNamespace(additional_kwargs={})
            cast(Callable[[SimpleNamespace], None], provider.get())(context)
            if scheduler_output.fail:
                raise RuntimeError("native execution failed")
            return {
                "inputs": "randomized" if randomize_inputs else "unchanged",
                "metadata": context.additional_kwargs,
                "skip_dp": skip_dp.get(),
            }

    runner_class = next(
        node
        for node in ast.parse(RUNNER_PATH.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "AFDNPUAttentionModelRunnerV2"
    )
    runner_class.bases = [ast.Name(id="NativeRunner", ctx=ast.Load())]
    execute_method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    execute_method.decorator_list = []
    runner_class.body = [execute_method]
    future = ast.parse("from __future__ import annotations").body
    namespace = {
        "NativeRunner": NativeRunner,
        "CUDAGraphMode": _GraphMode,
        "nullcontext": nullcontext,
        "step_afd_npu_profiler": lambda profiler: None,
        "skip_dp_coordination": dp_scope,
        "_use_afd_ubatch_preparation": lambda runner: nullcontext(),
        "use_afd_metadata_provider": metadata_scope,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=future + [runner_class], type_ignores=[])
            ),
            str(RUNNER_PATH),
            "exec",
        ),
        namespace,
    )

    def make_runner(async_dp):
        runner_factory = cast(
            Callable[[], NativeRunner], namespace["AFDNPUAttentionModelRunnerV2"]
        )
        runner = runner_factory()
        runner.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=_GraphMode.NONE)
        )
        runner.afd_config = SimpleNamespace(
            async_dp=async_dp, num_attention_ranks=2, num_ffn_ranks=2
        )
        runner.prof, runner.cudagraph_manager, runner.ubatch_runner = None, None, None
        runner._afd_pending_metadata = None
        runner._afd_suppress_metadata_send = False
        runner._is_warmup = False
        runner._afd_is_graph_capturing = False
        runner._afd_is_graph_replaying = False
        runner._afd_is_profile = False
        runner.install_afd_metadata_on_forward_context = lambda context: (
            context.additional_kwargs.update(afd_metadata="installed")
        )
        return runner

    return SimpleNamespace(make_runner=make_runner, provider=provider, skip_dp=skip_dp)


@pytest.mark.parametrize("async_dp", [False, True])
@pytest.mark.parametrize("randomize_inputs", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_execute_preserves_native_input_policy_and_restores_scopes(
    execute_contract, async_dp, randomize_inputs, fail
):
    runner = execute_contract.make_runner(async_dp)
    scheduler = SimpleNamespace(total_num_scheduled_tokens=7, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match="native execution failed"):
            runner.execute_model(
                scheduler, is_profile=True, randomize_inputs=randomize_inputs
            )
    else:
        result = runner.execute_model(
            scheduler, is_profile=True, randomize_inputs=randomize_inputs
        )
        assert result == {
            "inputs": "randomized" if randomize_inputs else "unchanged",
            "metadata": {"afd_metadata": "installed"},
            "skip_dp": async_dp,
        }
    assert execute_contract.provider.get() is None
    assert execute_contract.skip_dp.get() is False


def test_execute_keeps_native_default_input_policy(execute_contract):
    runner = execute_contract.make_runner(False)
    result = runner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=7, fail=False)
    )
    assert result["inputs"] == "unchanged"
