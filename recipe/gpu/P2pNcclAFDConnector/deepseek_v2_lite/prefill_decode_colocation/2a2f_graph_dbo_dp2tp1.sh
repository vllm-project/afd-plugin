# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
export VLLM_USE_V2_MODEL_RUNNER=1

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
