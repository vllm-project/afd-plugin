#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSpark eager on vLLM 0.32: optional target-only P, A TP1 and four F shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import afd_plugin
from afd_plugin.validation import CAM_MAX_RECEIVE_TOKENS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prefill", "attention", "ffn"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--afd-host", required=True)
    parser.add_argument("--afd-port", type=int, default=12499)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--devices", required=True)
    parser.add_argument("--attention-dp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prefill-tp", type=int, default=4)
    parser.add_argument("--kv-transfer-config", type=Path)
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--model-name", default="dsv4")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--disable-draft", action="store_true", help="Run the native PD/AFD baseline"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    is_ffn = args.role == "ffn"
    is_prefill = args.role == "prefill"
    if args.disable_draft and args.evidence_dir is not None:
        parser.error("DSpark ownership evidence requires a loaded draft model")
    if is_prefill and args.kv_transfer_config is None:
        parser.error("prefill requires a DSpark PD --kv-transfer-config file")
    if is_ffn and args.kv_transfer_config is not None:
        parser.error("FFN does not participate in P-to-D KV transfer")
    api_port = args.api_port or (7102 if is_prefill else 7101 if is_ffn else 7100)
    env = os.environ.copy()
    env.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_PLUGINS=(
            "ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
            if is_prefill
            else "ascend,afd"
        ),
        ASCEND_RT_VISIBLE_DEVICES=args.devices,
        VLLM_ASCEND_ENABLE_FLASHCOMM1="0",
        AFD_ASYNC_CAM_LAYERED_GMM="0",
        HCCL_SOCKET_IFNAME=args.interface,
        GLOO_SOCKET_IFNAME=args.interface,
        TP_SOCKET_IFNAME=args.interface,
        HCCL_IF_IP=args.local_ip,
        HCCL_BUFFSIZE="256",
        LCCL_BUFFER_SIZE="408",
        BATCH_SIZE_FACTOR=str(1024 / CAM_MAX_RECEIVE_TOKENS),
        OMP_NUM_THREADS="4",
    )
    vendor = (
        Path(afd_plugin.__file__).resolve().parent
        / "_cann_ops_custom/vendors/afd-plugin"
    )
    env["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        item for item in (str(vendor), env.get("ASCEND_CUSTOM_OPP_PATH")) if item
    )
    env["AFD_CUST_OPAPI_LIB_PATH"] = str(vendor / "op_api/lib/libcust_opapi.so")
    env["LD_LIBRARY_PATH"] = ":".join(
        item
        for item in (str(vendor / "op_api/lib"), env.get("LD_LIBRARY_PATH"))
        if item
    )
    additional = {
        "enable_cpu_binding": False,
        "enable_force_eplb": False,
        "enable_dsa_cp": False,
        "multistream_dsv4_dsa_overlap": False,
        "afd": {
            "role": args.role,
            "connector": "CAMAsyncAFDConnector",
            "async_dp": True,
            "host": args.afd_host,
            "port": args.afd_port,
            "num_attention_ranks": args.attention_dp,
            "num_ffn_ranks": 4,
            "compute_gate_on_attention": True,
            "connector_extra_config": {"dynamicQuant": 1, "attn_ranks_per_dp": 1},
        },
    }
    if is_prefill:
        del additional["afd"]
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(api_port),
        "--served-model-name",
        args.model_name,
        "--api-server-count",
        "1",
        "--shutdown-timeout",
        "60",
        "--tensor-parallel-size",
        str(args.prefill_tp) if is_prefill else "1",
        "--data-parallel-size",
        "1" if is_prefill else "4" if is_ffn else str(args.attention_dp),
        "--data-parallel-rpc-port",
        str(api_port + 10000),
        "--enable-expert-parallel",
        "--enforce-eager",
        "--seed",
        "1024",
        "--all2all-backend",
        "flashinfer_all2allv" if is_ffn else "allgather_reducescatter",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "4",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.85",
        "--quantization",
        "ascend",
        "--attention_config.indexer_kv_dtype",
        "int8",
        "--tokenizer-mode",
        "deepseek_v4",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--no-disable-hybrid-kv-cache-manager",
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": 4}),
        "--additional-config",
        json.dumps(additional),
    ]
    if not is_prefill and not args.disable_draft:
        command += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "model": args.model,
                    "num_speculative_tokens": 5,
                    "enforce_eager": True,
                }
            ),
        ]
    elif is_prefill and not args.disable_draft:
        command += [
            "--worker-cls",
            "afd_plugin.v1.worker.npu.dspark_prefill.AFDDSparkPrefillWorker",
        ]
    if args.kv_transfer_config is not None:
        transfer_config = json.loads(args.kv_transfer_config.read_text())
        extra = transfer_config["kv_connector_extra_config"]
        extra["decode"]["dp_size"] = args.attention_dp
        extra["prefill"]["tp_size"] = args.prefill_tp
        if args.disable_draft:
            transfer_config["kv_connector"] = "MooncakeConnector"
            transfer_config["kv_connector_module_path"] = (
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"
            )
            for name in tuple(extra):
                if name.startswith("dspark_"):
                    del extra[name]
        command += [
            "--kv-transfer-config",
            json.dumps(transfer_config),
        ]
    if args.evidence_dir is not None:
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["AFD_DSPARK_EVIDENCE_DIR"] = str(args.evidence_dir.resolve())
        repo_root = Path(__file__).resolve().parents[4]
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        evidence = (
            "dspark_prefill_evidence.DSparkPrefillEvidence"
            if is_prefill
            else "dspark_evidence.DSparkEagerEvidence"
        )
        command += [
            "--worker-extension-cls",
            f"tests.e2e.models.deepseek_v4_flash.{evidence}",
        ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        key: env[key]
                        for key in (
                            "ASCEND_RT_VISIBLE_DEVICES",
                            "VLLM_PLUGINS",
                            "AFD_ASYNC_CAM_LAYERED_GMM",
                            "BATCH_SIZE_FACTOR",
                        )
                    },
                },
                indent=2,
            )
        )
    else:
        os.execvpe(sys.executable, command, env)


if __name__ == "__main__":
    main()
