#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# AF-disaggregated (2A2F) decode-only benchmark.
#
# This is the DECODE side of prefill-decode (PD) disaggregation with the prefill
# instance faked out. Compare recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/
# prefill_decode_disaggregation/2p1a1f_graph_dbo.sh, where the decode side
# (attention + ffn) carries an LMCacheConnectorV1 kv_consumer and pulls real KV
# from a separate 1P prefill instance via a proxy.
#
# Here the AFDDecodeBenchConnector REPLACES that consumer connector on the
# ATTENTION instance and fabricates the KV: every prompt token except the last
# is reported as externally computed, and the KV cache is filled with dummy
# values. So requests skip prefill and go straight into decode -- letting you
# stress the decode path with arbitrary ISL, with NO prefill instance, no
# LMCache producer, and no proxy. Throughput/latency are meaningful; generated
# text is garbage. The connector is attached to attention only (the ffn side
# needs no kv-transfer-config), matching tests/e2e/runner.py.
#
# The connector lives in tools/benchmarks/ and is NOT shipped in the wheel, so
# it is loaded purely via kv_connector_module_path and needs the repo root on
# PYTHONPATH (inherited by the vLLM scheduler/worker subprocesses).
#
# Usage:
#   MODEL_PATH=/path/to/DeepSeek-V2-Lite tools/benchmarks/decode_bench_server.sh
# then, once both instances are ready, drive load against the attention port:
#   MODEL_PATH=/path/to/DeepSeek-V2-Lite tools/benchmarks/request_generator.sh
set -euo pipefail

# --- make tools.benchmarks.decode_bench importable in vLLM subprocesses -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}

# dummy-KV fill params for the decode-bench connector
FILL_MEAN=${FILL_MEAN:-0.015}
FILL_STD=${FILL_STD:-0.0}
DECODE_BENCH_KV_CONFIG=$(cat <<JSON
{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"fill_mean":${FILL_MEAN},"fill_std":${FILL_STD}}}
JSON
)

# --- attention instance (decode-bench connector attached here) ----------------
CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 2,
            "num_ffn_ranks": 2
        }
    }' \
    --kv-transfer-config "$DECODE_BENCH_KV_CONFIG" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > attn.log 2>&1 &

# --- ffn instance (no decode-bench connector) ---------------------------------
CUDA_VISIBLE_DEVICES=2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 2,
            "num_ffn_ranks": 2
        }
    }' \
    --max-num-seqs 64 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 12 \
    --max-num-batched-tokens 64 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --host 127.0.0.1 \
    --port 18306 \
    --trust-remote-code > ffn.log 2>&1 &

wait
