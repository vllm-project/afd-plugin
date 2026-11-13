# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV4 single-deployment declarations, separate from dedicated/multi-pod runs.

Importing IDs is safe. Checkpoint quantization is resolved only by build_case
in the runner child. Existing launch-profile helpers remain model-owned and
are also used by the unmigrated dedicated and multi-pod executors.
"""

from __future__ import annotations

import argparse
import json

from tests.e2e.case import (
    CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
    CLEANUP_PROFILE_STRICT,
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    NPU_ASYNC_TERMINATION_TIMEOUT_S,
    NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S,
    NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S,
    Case,
    CaseEnvironment,
    DboConfig,
    Deployment,
    Gsm8kSettings,
    Gsm8kTask,
    HashInputCheck,
    Mrv2Check,
    Phase,
    Role,
)
from tests.e2e.models.deepseek_v4_flash import config, gpu_config
from tests.e2e.models.deepseek_v4_flash.completions import CompletionTask

GPU_BASIC_IDS = gpu_config.DSV4_GPU_BASIC_SCENARIOS
GPU_DBO_IDS = gpu_config.DSV4_GPU_DBO_SCENARIOS
NPU_ASYNC_IDS = (config.DSV4_ASYNC_CAM_SCENARIO,)
NPU_SYNC_IDS = config.DSV4_SYNC_CAMP2P_SCENARIOS
GSM8K_SAMPLES = 7
GSM8K_THRESHOLD = 0.27
DBO_CONCURRENCY = 12
DBO_MINIMUM_SAMPLES = 2 * DBO_CONCURRENCY
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 8


def _budget(arguments: list[str], name: str) -> int | None:
    """Transfer a model profile's budget to its deployment declaration."""
    if name not in arguments:
        return None
    index = arguments.index(name)
    value = int(arguments[index + 1])
    del arguments[index : index + 2]
    return value


def _profile_arguments(
    case_id: str, environment: CaseEnvironment
) -> argparse.Namespace:
    args = argparse.Namespace(
        scenario=case_id,
        model=environment.model,
        device_backend=environment.device_backend,
        afd_host=environment.afd_host,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        completion_output_path=environment.output_path / "completions.json",
        gsm8k_output_path=environment.output_path / "gsm8k",
        use_decode_bench_connector=False,
    )
    if environment.device_backend == "gpu":
        gpu_config.configure_scenario(args)
    elif case_id in NPU_ASYNC_IDS:
        config.configure_scenario(args)
    else:
        config.configure_sync_camp2p_scenario(args)
    return args


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend == "gpu":
        return _gpu_case(case_id, environment)
    if environment.device_backend == "npu" and case_id in (
        *NPU_ASYNC_IDS,
        *NPU_SYNC_IDS,
    ):
        return _npu_case(case_id, environment)
    raise ValueError(f"unsupported DSV4 {environment.device_backend} case {case_id!r}")


def _gpu_case(case_id: str, environment: CaseEnvironment) -> Case:
    baseline, graph, dbo, attention_ranks, ffn_ranks = (
        gpu_config.DSV4_GPU_SCENARIO_SETTINGS[case_id]
    )
    args = _profile_arguments(case_id, environment)
    max_num_seqs = _budget(args.common_vllm_arg, "--max-num-seqs")
    max_num_batched_tokens = _budget(args.common_vllm_arg, "--max-num-batched-tokens")
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    roles = ("baseline",) if baseline else ("attention", "ffn")
    checks = (
        (Mrv2Check(roles, graph=graph, dbo=dbo),)
        if not baseline or graph or dbo
        else ()
    )
    phases = []
    for index, name in enumerate(gpu_config.DSV4_GPU_HASH_WINDOWS):
        hash_checks = (
            ()
            if baseline
            else (
                HashInputCheck(
                    graph=graph,
                    dbo=dbo,
                    compare_with=gpu_config.DSV4_GPU_HASH_WINDOWS[index - 1]
                    if index
                    else None,
                ),
            )
        )
        phases.append(
            Phase(
                name,
                CompletionTask(
                    first_request_index=index * config.DSV4_CONCURRENT_REQUESTS,
                    answer_repetitions=gpu_config.DSV4_GPU_ANSWER_REPETITIONS,
                ),
                (*checks, *hash_checks),
            )
        )
    phases.append(
        Phase(
            "gsm8k",
            Gsm8kTask(
                Gsm8kSettings(
                    sample_limit=DBO_MINIMUM_SAMPLES if dbo else GSM8K_SAMPLES,
                    threshold=GSM8K_THRESHOLD,
                    num_concurrent=DBO_CONCURRENCY if dbo else None,
                    minimum_samples=DBO_MINIMUM_SAMPLES if dbo else 1,
                )
            ),
            checks,
        )
    )
    return Case(
        id=case_id,
        model="DeepSeek-V4-Flash",
        served_model_name_prefix="dsv4-flash-gpu",
        deployment=Deployment(
            backend="gpu",
            attention=Role(attention_ranks, execution=execution),
            ffn=None if baseline else Role(ffn_ranks, execution=execution),
            connector=None if baseline else "P2pNcclAFDConnector",
            dbo=DboConfig(DBO_DECODE_THRESHOLD, DBO_PREFILL_THRESHOLD) if dbo else None,
            cudagraph_capture_size=gpu_config.DSV4_GPU_CAPTURE_SIZE if graph else None,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        common_vllm_args=tuple(args.common_vllm_arg),
        phases=tuple(phases),
    )


def _npu_case(case_id: str, environment: CaseEnvironment) -> Case:
    asynchronous = case_id in NPU_ASYNC_IDS
    shape = None if asynchronous else config.DSV4_SYNC_SHAPES[case_id]
    args = _profile_arguments(case_id, environment)
    max_num_seqs = _budget(args.common_vllm_arg, "--max-num-seqs")
    max_num_batched_tokens = _budget(args.common_vllm_arg, "--max-num-batched-tokens")
    graph = shape is not None and config.sync_use_graph(shape)
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    attention_ranks = (
        config.DSV4_ATTENTION_RANKS if shape is None else shape.attention_ranks
    )
    ffn_ranks = config.DSV4_FFN_RANKS if shape is None else shape.ffn_ranks
    attention_tp = (
        config.DSV4_ATTENTION_TP_SIZE if shape is None else shape.attention_tp_size
    )
    ffn_tp = 1 if shape is None else shape.ffn_tp_size
    if asynchronous:
        # The existing async path enables native DBO only on Attention, using
        # upstream defaults; it does not claim a DBO execution acceptance check.
        args.attention_vllm_arg.remove("--enable-dbo")
    compilation = None if shape is None else config.sync_compilation_config(shape)
    return Case(
        id=case_id,
        model="DeepSeek-V4-Flash",
        served_model_name_prefix="dsv4-flash" if asynchronous else "dsv4-flash-sync",
        deployment=Deployment(
            backend="npu",
            attention=Role(
                attention_ranks // attention_tp,
                attention_tp,
                execution,
                dbo=DboConfig() if asynchronous else None,
                vllm_args=tuple(args.attention_vllm_arg),
                additional_config=tuple(
                    config.additional_config(
                        "attention", async_cam=asynchronous
                    ).items()
                ),
            ),
            ffn=Role(
                ffn_ranks // ffn_tp,
                ffn_tp,
                execution,
                vllm_args=tuple(args.ffn_vllm_arg),
                additional_config=tuple(
                    config.additional_config("ffn", async_cam=asynchronous).items()
                ),
            ),
            connector=args.afd_connector,
            compute_gate_on_attention=args.compute_gate_on_attention,
            connector_extra_config=tuple(
                (
                    json.loads(args.afd_connector_extra_config[0])
                    if args.afd_connector_extra_config
                    else {}
                ).items()
            ),
            pin_attention_address=args.attention_data_parallel_address is not None,
            cleanup_profile=CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY
            if asynchronous
            else CLEANUP_PROFILE_STRICT,
            shutdown_timeout_s=NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S
            if asynchronous
            else DEFAULT_SHUTDOWN_TIMEOUT_S,
            worker_shutdown_timeout_s=NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S
            if asynchronous
            else None,
            termination_timeout_s=NPU_ASYNC_TERMINATION_TIMEOUT_S
            if asynchronous
            else config.DSV4_PROCESS_TERMINATION_TIMEOUT_S,
            cudagraph_capture_size=(
                shape.cudagraph_capture_size if shape is not None and graph else None
            ),
            compilation_config=tuple(
                (key, tuple(value) if isinstance(value, list) else value)
                for key, value in compilation.items()
            )
            if compilation is not None
            else (),
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        common_vllm_args=tuple(args.common_vllm_arg),
        phases=(
            Phase(
                "evaluation",
                CompletionTask(
                    check_answer=True if shape is None else shape.check_answer
                ),
                (Mrv2Check(("attention", "ffn"), graph=True),) if graph else (),
            ),
        ),
    )


if __name__ == "__main__":
    from tests.e2e.runner import case_main

    raise SystemExit(
        case_main(
            build_case, (*GPU_BASIC_IDS, *GPU_DBO_IDS, *NPU_ASYNC_IDS, *NPU_SYNC_IDS)
        )
    )
