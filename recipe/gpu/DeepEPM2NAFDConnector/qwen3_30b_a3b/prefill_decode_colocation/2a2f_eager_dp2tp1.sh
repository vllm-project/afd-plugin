# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# 2A2F eager: Attention DP2 x TP1 on GPUs 0,1; FFN DP2 on GPUs 2,3.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
export VLLM_USE_V2_MODEL_RUNNER=1

CUDA_VISIBLE_DEVICES=2,3 vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "DeepEPM2NAFDConnector",
            "host": "127.0.0.1",
            "port": 6239,
            "num_attention_ranks": 2,
            "num_ffn_ranks": 2,
            "compute_gate_on_attention": true
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --max-model-len 4096 \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 18001 > ffn.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "DeepEPM2NAFDConnector",
            "host": "127.0.0.1",
            "port": 6239,
            "num_attention_ranks": 2,
            "num_ffn_ranks": 2,
            "compute_gate_on_attention": true
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --max-model-len 4096 \
    --enforce-eager \
    --host 127.0.0.1 \
    --port 18000 > attn.log 2>&1 &

wait
