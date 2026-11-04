# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# Use .agents/skills/deploy-afd-k8s/SKILL.md to deploy this recipe
# on 3 pods: two attention pods and one FFN pod. The deployment layer
# places each role in its own StatefulSet; AFD_ROLE selects the role and
# POD is the pod's zero-based ordinal within it. Attention shards with
# POD>0 run headless with a data-parallel start rank of POD*2.

MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3.5-122B-A10B-FP8}
export VLLM_USE_V2_MODEL_RUNNER=1

: "${AFD_CONNECTOR_HOST:?not set -- deploy via the deploy-afd-k8s skill, which provides this pod env}"
AFD_CONNECTOR_PORT=${AFD_CONNECTOR_PORT:-1239}

# Per-ordinal conventions, derived from the pod's StatefulSet ordinal:
# POD>0 shards run headless and start at data-parallel rank
# POD*ATTENTION_DP_RANKS_PER_POD.
ATTENTION_DP_RANKS_PER_POD=2
ATTENTION_DP_START_RANK=${ATTENTION_DP_START_RANK:-$(( ${POD:-0} * ATTENTION_DP_RANKS_PER_POD ))}
ATTENTION_HEADLESS=${ATTENTION_HEADLESS:-$([ "${POD:-0}" -gt 0 ] && echo 1 || echo 0)}
: "${ATTENTION_DP_ADDRESS:?not set -- deploy via the deploy-afd-k8s skill, which provides this pod env}"
ATTENTION_DP_RPC_PORT=${ATTENTION_DP_RPC_PORT:-13345}

HEADLESS_ARGS=()
if [[ "$ATTENTION_HEADLESS" == "1" ]]; then
  HEADLESS_ARGS=(--headless)
fi

DP_START_RANK_ARGS=()
if [[ "$ATTENTION_DP_START_RANK" != "0" ]]; then
  DP_START_RANK_ARGS=(--data-parallel-start-rank "$ATTENTION_DP_START_RANK")
fi

if [ "$AFD_ROLE" == "attention" ]; then
  CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size 4 \
      --data-parallel-size-local "$ATTENTION_DP_RANKS_PER_POD" \
      "${DP_START_RANK_ARGS[@]}" \
      --data-parallel-address "$ATTENTION_DP_ADDRESS" \
      --data-parallel-rpc-port "$ATTENTION_DP_RPC_PORT" \
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
              "afd_process_group_timeout_s": 120,
              "host": "'"${AFD_CONNECTOR_HOST}"'",
              "port": '"${AFD_CONNECTOR_PORT}"',
              "num_attention_ranks": 4,
              "num_ffn_ranks": 2
          }
      }' \
      --max-num-seqs 32 \
      --max-num-batched-tokens 8192 \
      --max-cudagraph-capture-size 32 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[32]
      }' \
      "${HEADLESS_ARGS[@]}" \
      --host 127.0.0.1 \
      --port 18305 > attn.log 2>&1 &
fi

if [ "$AFD_ROLE" == "ffn" ]; then
  CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve "$MODEL_PATH" \
      --data-parallel-size 2 \
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
              "role": "ffn",
              "connector": "P2pNcclAFDConnector",
              "afd_process_group_timeout_s": 120,
              "host": "'"${AFD_CONNECTOR_HOST}"'",
              "port": '"${AFD_CONNECTOR_PORT}"',
              "num_attention_ranks": 4,
              "num_ffn_ranks": 2
          }
      }' \
      --max-num-seqs 32 \
      --max-num-batched-tokens 8192 \
      --max-cudagraph-capture-size 32 \
      --compilation-config '{
          "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[32]
      }' \
      --host 127.0.0.1 \
      --port 18305 > ffn.log 2>&1 &
fi

wait
