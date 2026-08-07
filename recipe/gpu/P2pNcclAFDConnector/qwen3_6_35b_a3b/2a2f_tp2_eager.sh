#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to Qwen3.6-35B-A3B}"

vllm_bin="${VLLM_BIN:-vllm}"
host="${HOST:-127.0.0.1}"
attention_port="${ATTENTION_PORT:-18081}"
ffn_port="${FFN_PORT:-18082}"
afd_port="${AFD_PORT:-6280}"
log_dir="${AFD_QWEN36_LOG_DIR:-${TMPDIR:-/tmp}/qwen36-afd-logs}"
mkdir -p "$log_dir"

common_args=(
  --tensor-parallel-size 2
  --enforce-eager
  --dtype bfloat16
  --max-model-len 4096
  --language-model-only
  --generation-config vllm
  --seed 0
  --disable-cascade-attn
  --moe-backend triton
  --no-enable-expert-parallel
)

ffn_config=$(printf '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"%s","port":%s,"num_attention_ranks":2,"num_ffn_ranks":2,"compute_gate_on_attention":true}}' "$host" "$afd_port")
attention_config=$(printf '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"%s","port":%s,"num_attention_ranks":2,"num_ffn_ranks":2,"compute_gate_on_attention":true}}' "$host" "$afd_port")

ffn_pid=""
attention_pid=""
cleanup() {
  for pid in "$attention_pid" "$ffn_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  wait "$attention_pid" 2>/dev/null || true
  wait "$ffn_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="${FFN_GPUS:-2,3}" \
  VLLM_PLUGINS=afd VLLM_USE_V2_MODEL_RUNNER=0 \
  VLLM_USE_FLASHINFER_SAMPLER=0 "$vllm_bin" serve "$MODEL_PATH" \
  --served-model-name qwen36-afd --host "$host" --port "$ffn_port" \
  --additional-config "$ffn_config" "${common_args[@]}" \
  >"$log_dir/ffn.log" 2>&1 &
ffn_pid=$!

setsid env CUDA_VISIBLE_DEVICES="${ATTENTION_GPUS:-0,1}" \
  VLLM_PLUGINS=afd VLLM_USE_V2_MODEL_RUNNER=0 \
  VLLM_USE_FLASHINFER_SAMPLER=0 "$vllm_bin" serve "$MODEL_PATH" \
  --served-model-name qwen36-afd --host "$host" --port "$attention_port" \
  --additional-config "$attention_config" "${common_args[@]}" \
  >"$log_dir/attention.log" 2>&1 &
attention_pid=$!

echo "Attention API: http://${host}:${attention_port}/v1"
echo "Logs: $log_dir"
wait "$attention_pid"
