# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3.5-122B-A10B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=1

# EPLB is enabled on the FFN role only: the Attention role ships hidden
# states without computing MoE routing (compute_gate_on_attention=false)
# and its remote-expert shells reject enable_eplb. Qwen3.5-122B-A10B has
# 256 routed experts, so 32 redundant experts yield 288 physical experts
# (144 per FFN rank at EP size 2).

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 114688 \
    --mamba-cache-mode align \
    --all2all-backend allgather_reducescatter \
    --seed 0 \
    --additional-config '{
        "afd": {
            "role": "attention",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 2
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > attn.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 uv run vllm serve "$MODEL_PATH" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-config '{"num_redundant_experts": 32}' \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 114688 \
    --mamba-cache-mode align \
    --all2all-backend allgather_reducescatter \
    --seed 0 \
    --additional-config '{
        "afd": {
            "role": "ffn",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": 6269,
            "num_attention_ranks": 4,
            "num_ffn_ranks": 2
        }
    }' \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{
        "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[64]
    }' \
    --host 127.0.0.1 \
    --port 18305 \
    --trust-remote-code > ffn.log 2>&1 &

wait
