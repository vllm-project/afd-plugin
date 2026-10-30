# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# 4A2F graph + DBO, FP8 payload: Attention DP4 x TP1 on GPUs 0-3; FFN DP2 on
# GPUs 4,5. DBO must be enabled on both roles.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-235B-A22B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=1

CUDA_VISIBLE_DEVICES=4,5 vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "DeepEPM2NAFDConnector",
            "host": "127.0.0.1",
            "port": 6239,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 2,
            "compute_gate_on_attention": true,
            "afd_process_group_timeout_s": 3600,
            "connector_extra_config": {"payload_format": "fp8_block128_fp32"}
        }
    }' \
    --max-num-seqs 512 \
    --max-num-batched-tokens 3072 \
    --max-model-len 4096 \
    --max-cudagraph-capture-size 512 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]}' \
    --cpu-distributed-timeout-seconds 3600 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 8 \
    --host 127.0.0.1 \
    --port 18001 > ffn.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "DeepEPM2NAFDConnector",
            "host": "127.0.0.1",
            "port": 6239,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 2,
            "compute_gate_on_attention": true,
            "afd_process_group_timeout_s": 3600,
            "connector_extra_config": {"payload_format": "fp8_block128_fp32"}
        }
    }' \
    --max-num-seqs 512 \
    --max-num-batched-tokens 3072 \
    --max-model-len 4096 \
    --max-cudagraph-capture-size 512 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]}' \
    --cpu-distributed-timeout-seconds 3600 \
    --enable-dbo \
    --dbo-decode-token-threshold 2 \
    --dbo-prefill-token-threshold 8 \
    --host 127.0.0.1 \
    --port 18000 > attn.log 2>&1 &

wait
