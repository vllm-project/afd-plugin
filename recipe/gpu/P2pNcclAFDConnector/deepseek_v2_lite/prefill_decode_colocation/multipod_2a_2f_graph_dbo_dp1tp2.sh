# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Use .agents/skills/deploy-afd-k8s/SKILL.md to deploy this recipe
# on 2 pods: one attention pod and one FFN pod. The deployment layer
# places each role in its own StatefulSet; AFD_ROLE selects the role and
# POD is the pod's zero-based ordinal within it (0 for both pods here).

MODEL_PATH=${MODEL_PATH:-/path/model_weights/DeepSeek-V2-Lite}
export VLLM_USE_V2_MODEL_RUNNER=1

: "${AFD_CONNECTOR_HOST:?not set -- deploy via the deploy-afd-k8s skill, which provides this pod env}"
AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-6269}

if [ "$AFD_ROLE" == "attention" ]; then
  CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size 1 \
      --tensor-parallel-size 2 \
      --enable-expert-parallel \
      --additional-config '{
          "afd": {
              "role": "attention",
              "connector": "P2pNcclAFDConnector",
              "host": "'"${AFD_CONNECTOR_HOST}"'",
              "port": '"${AFD_CONNECTOR_PORT}"',
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
fi

if [ "$AFD_ROLE" == "ffn" ]; then
  CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size 1 \
      --tensor-parallel-size 2 \
      --enable-expert-parallel \
      --additional-config '{
          "afd": {
              "role": "ffn",
              "connector": "P2pNcclAFDConnector",
              "host": "'"${AFD_CONNECTOR_HOST}"'",
              "port": '"${AFD_CONNECTOR_PORT}"',
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
      --port 18305 \
      --trust-remote-code > ffn.log 2>&1 &
fi

wait
