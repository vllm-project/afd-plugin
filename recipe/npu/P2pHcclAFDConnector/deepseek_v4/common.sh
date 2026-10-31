#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

afd_die() {
  printf '[dsv4-afd] ERROR: %s\n' "$*" >&2
  exit 2
}

require_uint() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || afd_die "$name must be an unsigned integer, got '$value'"
}

configure_parallelism() {
  require_uint ATTENTION_RANKS "$ATTENTION_RANKS"
  require_uint FFN_RANKS "$FFN_RANKS"
  require_uint TENSOR_PARALLEL_SIZE "$TENSOR_PARALLEL_SIZE"
  ((ATTENTION_RANKS > 0 && ATTENTION_RANKS <= 8 && FFN_RANKS > 0 && FFN_RANKS <= 8)) \
    || afd_die "each role must use 1..8 local devices; use separate hosts for overlapping device IDs"
  [[ "$TENSOR_PARALLEL_SIZE" == 1 || "$TENSOR_PARALLEL_SIZE" == 2 ]] \
    || afd_die "DeepSeek-V4 AFD supports TENSOR_PARALLEL_SIZE=1 or 2"
  ((ATTENTION_RANKS % TENSOR_PARALLEL_SIZE == 0 && FFN_RANKS % TENSOR_PARALLEL_SIZE == 0)) \
    || afd_die "Attention and FFN rank counts must be divisible by TENSOR_PARALLEL_SIZE"
  ((ATTENTION_RANKS % FFN_RANKS == 0 || FFN_RANKS % ATTENTION_RANKS == 0)) \
    || afd_die "P2pHcclAFDConnector requires one role rank count to be an integer multiple of the other"
  if [[ "$TENSOR_PARALLEL_SIZE" == 2 && "$ATTENTION_RANKS" != "$FFN_RANKS" ]]; then
    afd_die "DeepSeek-V4 AFD TP2 requires equal Attention and FFN rank counts"
  fi
  ATTENTION_DP_SIZE=$((ATTENTION_RANKS / TENSOR_PARALLEL_SIZE))
  FFN_DP_SIZE=$((FFN_RANKS / TENSOR_PARALLEL_SIZE))
  case "$ROLE" in
    attention) ROLE_DP_SIZE="$ATTENTION_DP_SIZE" ;;
    ffn) ROLE_DP_SIZE="$FFN_DP_SIZE" ;;
    *) afd_die "invalid AFD role: $ROLE" ;;
  esac
}

validate_devices() {
  local attention_count=0 ffn_count=0 device
  declare -A attention_seen=() ffn_seen=()

  IFS=',' read -r -a attention_devices_array <<<"$ATTENTION_DEVICES"
  IFS=',' read -r -a ffn_devices_array <<<"$FFN_DEVICES"
  attention_count="${#attention_devices_array[@]}"
  ffn_count="${#ffn_devices_array[@]}"

  ((attention_count == ATTENTION_RANKS)) \
    || afd_die "ATTENTION_DEVICES must contain $ATTENTION_RANKS devices"
  ((ffn_count == FFN_RANKS)) \
    || afd_die "FFN_DEVICES must contain $FFN_RANKS devices"

  for device in "${attention_devices_array[@]}"; do
    [[ "$device" =~ ^[0-7]$ ]] \
      || afd_die "A5 recipe device IDs must be integers in [0, 7], got '$device'"
    [[ -z "${attention_seen[$device]:-}" ]] \
      || afd_die "Attention device list repeats device $device"
    attention_seen[$device]=1
  done
  for device in "${ffn_devices_array[@]}"; do
    [[ "$device" =~ ^[0-7]$ ]] \
      || afd_die "A5 recipe device IDs must be integers in [0, 7], got '$device'"
    [[ -z "${ffn_seen[$device]:-}" ]] \
      || afd_die "FFN device list repeats device $device"
    ffn_seen[$device]=1
    if [[ "${ATTENTION_HOST_IP:-$HCCL_IF_IP}" == "${FFN_HOST_IP:-$HCCL_IF_IP}" ]]; then
      [[ -z "${attention_seen[$device]:-}" ]] \
        || afd_die "Attention and FFN device lists overlap at device $device"
    fi
  done
}

validate_model_config() {
  "$PYTHON_BIN" - "$MODEL_PATH/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    config = json.loads(path.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(f"[dsv4-afd] ERROR: model config does not exist: {path}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"[dsv4-afd] ERROR: invalid model config JSON: {path}: {exc}")

quant = config.get("quantization_config")
expected = {
    "model_type": (config.get("model_type"), "deepseek_v4"),
    "architectures": (config.get("architectures"), ["DeepseekV4ForCausalLM"]),
    "num_nextn_predict_layers": (config.get("num_nextn_predict_layers"), 1),
    "quant_method": (
        quant.get("quant_method") if isinstance(quant, dict) else None,
        "fp8",
    ),
    "activation_scheme": (
        quant.get("activation_scheme") if isinstance(quant, dict) else None,
        "dynamic",
    ),
    "fmt": (quant.get("fmt") if isinstance(quant, dict) else None, "e4m3"),
    "scale_fmt": (
        quant.get("scale_fmt") if isinstance(quant, dict) else None,
        "ue8m0",
    ),
    "weight_block_size": (
        quant.get("weight_block_size") if isinstance(quant, dict) else None,
        [128, 128],
    ),
}
mismatches = [
    f"{name}={actual!r} (expected {wanted!r})"
    for name, (actual, wanted) in expected.items()
    if actual != wanted
]
if config.get("expert_dtype") not in (None, "fp4"):
    mismatches.append(
        f"expert_dtype={config.get('expert_dtype')!r} (expected missing or 'fp4')"
    )
if mismatches:
    raise SystemExit(
        "[dsv4-afd] ERROR: unsupported A5 DeepSeek-V4 checkpoint: "
        + ", ".join(mismatches)
    )
PY
}

build_a5_pd_kv_config() {
  # This A5-only recipe maps local logical ranks to physical NPU endpoint files.
  # Do not call this helper from A3 recipes, FFN, or standalone Decode-AF.
  local kv_role="$1" engine_id="$2" kv_port="$3" devices="$4"
  "$PYTHON_BIN" - "$kv_role" "$engine_id" "$kv_port" \
    "$PREFILL_DP_SIZE" "$PREFILL_TP_SIZE" "$ATTENTION_RANKS" "${TENSOR_PARALLEL_SIZE:-1}" \
    "${ASCEND_LOCAL_COMM_RES_PATH:-}" "$devices" <<'PY'
import json
import sys
from pathlib import Path

role, engine_id, port, prefill_dp, prefill_tp, attention_ranks, attention_tp, resource_dir, devices = sys.argv[1:]
if role not in ("kv_producer", "kv_consumer"):
    raise SystemExit("[dsv4-afd] ERROR: A5 PD requires a KV producer or consumer")
if not engine_id:
    raise SystemExit("[dsv4-afd] ERROR: Mooncake engine ID cannot be empty")
resource_root = Path(resource_dir)
if not resource_root.is_absolute() or not resource_root.is_dir():
    raise SystemExit(
        "[dsv4-afd] ERROR: A5 PD ASCEND_LOCAL_COMM_RES_PATH must be an "
        f"existing local absolute directory, got {resource_dir!r}"
    )
try:
    port_number = int(port)
    prefill_dp_size, prefill_tp_size, attention_rank_count, attention_tp_size = map(
        int, (prefill_dp, prefill_tp, attention_ranks, attention_tp)
    )
    device_ids = [int(device) for device in devices.split(",")]
except ValueError as exc:
    raise SystemExit(f"[dsv4-afd] ERROR: invalid A5 PD numeric config: {exc}") from exc
if not 0 < port_number < 65536:
    raise SystemExit("[dsv4-afd] ERROR: Mooncake KV port is outside 1..65535")
if (prefill_dp_size < 1 or prefill_tp_size < 1
        or prefill_dp_size * prefill_tp_size > 8
        or not 1 <= attention_rank_count <= 8
        or attention_tp_size not in (1, 2)
        or attention_rank_count % attention_tp_size):
    raise SystemExit("[dsv4-afd] ERROR: invalid A5 PD DP/TP or local rank count")
if prefill_tp_size < attention_tp_size:
    raise SystemExit("[dsv4-afd] ERROR: Mooncake PD requires Prefill TP >= Attention TP")
attention_dp_size = attention_rank_count // attention_tp_size
expected_devices = (
    prefill_dp_size * prefill_tp_size if role == "kv_producer" else attention_rank_count
)
if (len(device_ids) != expected_devices or len(set(device_ids)) != len(device_ids)
        or any(device < 0 or device > 7 for device in device_ids)):
    raise SystemExit("[dsv4-afd] ERROR: A5 PD device list does not match local role ranks")
for device in device_ids:
    endpoint = resource_root / f"ub_endpoint_npu_{device}.json"
    try:
        with endpoint.open(encoding="utf-8") as file:
            json.load(file)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"[dsv4-afd] ERROR: unreadable or invalid A5 HIXL endpoint {endpoint}: {exc}"
        ) from exc
print(json.dumps({
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": role,
    "engine_id": engine_id,
    "kv_port": port_number,
    "kv_parallel_size": 1,
    "kv_connector_extra_config": {
        "prefill": {"dp_size": prefill_dp_size, "tp_size": prefill_tp_size},
        "decode": {"dp_size": attention_dp_size, "tp_size": attention_tp_size},
        "ascend_local_comm_res_path": str(resource_root),
    },
}, separators=(",", ":")))
PY
}

check_port_free() {
  local host="$1" port="$2" purpose="$3"
  "$PYTHON_BIN" - "$host" "$port" "$purpose" <<'PY'
import socket
import sys

host, raw_port, purpose = sys.argv[1:]
port = int(raw_port)
addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
last_error = None
for family, socktype, proto, _, address in addresses:
    sock = socket.socket(family, socktype, proto)
    try:
        sock.bind(address)
    except OSError as exc:
        last_error = exc
    else:
        sock.close()
        break
    finally:
        try:
            sock.close()
        except OSError:
            pass
else:
    raise SystemExit(
        f"[dsv4-afd] ERROR: {purpose} port {host}:{port} is unavailable: {last_error}"
    )
PY
}

configure_dspark() {
  DSPARK_ARGS=()
  DSPARK_BLOCK_SIZE=0
  if [[ "${ENABLE_DSPARK:-0}" == 1 ]]; then
    DSPARK_BLOCK_SIZE="$("$PYTHON_BIN" - "$MODEL_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    size = json.load(file).get("dspark_block_size")
if type(size) is not int or size < 1:
    raise SystemExit("[dsv4-afd] ERROR: DSpark requires a positive dspark_block_size")
print(size)
PY
)"
    # vLLM 0.23 uses the MTP schema; Ascend selects DSpark from the checkpoint.
    DSPARK_CONFIG="$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"greedy","enforce_eager":true}' "$DSPARK_BLOCK_SIZE")"
    DSPARK_ARGS=(--speculative-config "$DSPARK_CONFIG")
  fi
}

configure_graph_capture() {
  # Attention dispatches the whole per-DP-rank token batch before U2 splitting.
  # FFN has no speculative config: its actual graphs are captured from A metadata.
  local graph_config
  graph_config="$("$PYTHON_BIN" - "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" \
    "${DSPARK_BLOCK_SIZE:-0}" "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" \
    "${CUDAGRAPH_CAPTURE_SIZES:-}" <<'PY'
import sys

raw_seqs, raw_budget, raw_spec, raw_max, raw_sizes = sys.argv[1:]
try:
    seqs, budget, spec = map(int, (raw_seqs, raw_budget, raw_spec))
    sizes = [int(size) for size in raw_sizes.split()]
    query_len = spec + 1
    required = seqs * query_len
    maximum = int(raw_max) if raw_max else max(sizes, default=required)
except ValueError as exc:
    raise SystemExit(f"[dsv4-afd] ERROR: Graph settings must be integers: {exc}") from exc
if seqs < 1 or budget < 1 or spec < 0 or maximum < 1 or any(size < 1 for size in sizes):
    raise SystemExit("[dsv4-afd] ERROR: Graph settings must be positive (spec tokens may be zero)")
if maximum < required:
    raise SystemExit(
        f"[dsv4-afd] ERROR: Graph max {maximum} does not cover {seqs} sequences x "
        f"{query_len} tokens = {required} before U2; unset capture overrides to derive defaults"
    )
if maximum > budget:
    raise SystemExit("[dsv4-afd] ERROR: Graph max exceeds MAX_NUM_BATCHED_TOKENS")
if maximum % query_len:
    raise SystemExit(
        f"[dsv4-afd] ERROR: Graph max must be a multiple of decode query length {query_len}"
    )
if sizes:
    if max(sizes) != maximum:
        raise SystemExit("[dsv4-afd] ERROR: largest capture size must equal Graph max")
    # Mirror vLLM's speculative decode alignment without losing the largest bucket.
    sizes = sorted({((size + query_len - 1) // query_len) * query_len for size in sizes})
    if required not in sizes:
        raise SystemExit(
            f"[dsv4-afd] ERROR: capture sizes must include full Decode shape {required}; "
            "FULL_DECODE_ONLY excludes buckets above the sequence limit"
        )
else:
    sizes = []
    size = query_len
    while size < maximum:
        sizes.append(size)
        size *= 2
    sizes.append(maximum)
    sizes = sorted(set(sizes + [required]))
print(maximum, *sizes)
PY
)" || afd_die "invalid Graph capture configuration"
  read -r MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES <<<"$graph_config"
}

configure_execution() {
  EXECUTION_ARGS=()
  UBATCH_ARGS=()
  SCHEDULING_ARGS=(--no-async-scheduling)

  case "${EXECUTION_MODE}:${U_BATCHES}" in
    eager:1)
      EXECUTION_ARGS=(--enforce-eager)
      ;;
    eager:2)
      EXECUTION_ARGS=(--enforce-eager)
      UBATCH_ARGS=(
        --enable-dbo
        --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
        --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
      )
      # Keep the eager U2 default while allowing serial/overlap comparisons.
      export AFD_HCCL_EAGER_U2_STREAM_OVERLAP="${AFD_HCCL_EAGER_U2_STREAM_OVERLAP:-1}"
      case "$AFD_HCCL_EAGER_U2_STREAM_OVERLAP" in
        0|1) ;;
        *) afd_die "AFD_HCCL_EAGER_U2_STREAM_OVERLAP must be 0 or 1" ;;
      esac
      ;;
    full-decode-only:2)
      configure_graph_capture
      read -r -a capture_sizes_array <<<"$CUDAGRAPH_CAPTURE_SIZES"
      EXECUTION_ARGS=(
        --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
        --cudagraph-capture-sizes "${capture_sizes_array[@]}"
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
      )
      UBATCH_ARGS=(
        --enable-dbo
        --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
        --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
      )
      export AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0
      # Default to all-on while preserving explicit 0/1 overrides.
      export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP="${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP:-1}"
      export AFD_HCCL_GRAPH_U2_HYBRID_DAG="${AFD_HCCL_GRAPH_U2_HYBRID_DAG:-1}"
      export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM="${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM:-1}"
      export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM="${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM:-1}"
      export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER="${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER:-1}"
      ;;
    eager:*)
      afd_die "eager recipe requires U_BATCHES=1 or 2"
      ;;
    full-decode-only:*)
      afd_die "the public FULL_DECODE_ONLY recipe requires U_BATCHES=2"
      ;;
    *)
      afd_die "EXECUTION_MODE must be eager or full-decode-only"
      ;;
  esac
}

preflight_role() {
  : "${MODEL_PATH:?Set MODEL_PATH to the A5 DeepSeek-V4 model directory}"
  : "${NIC_NAME:?Set NIC_NAME to the container network interface}"
  : "${HCCL_IF_IP:?Set HCCL_IF_IP to the IPv4 address on NIC_NAME}"

  configure_parallelism
  [[ "${ENABLE_MTP:-0}" == 0 ]] || afd_die "this recipe requires ENABLE_MTP=0"
  [[ "${ENABLE_DSPARK:-0}" == 0 || "${ENABLE_DSPARK:-0}" == 1 ]] \
    || afd_die "ENABLE_DSPARK must be 0 or 1"
  [[ "${ENABLE_PD:-0}" == 0 || "${ENABLE_PD:-0}" == 1 ]] \
    || afd_die "ENABLE_PD must be 0 or 1"
  [[ "$ROLE" == attention || "${ENABLE_DSPARK:-0}" == 0 ]] \
    || afd_die "DSpark config belongs only to Attention"
  [[ "$ROLE" == attention || "${ENABLE_PD:-0}" == 0 ]] \
    || afd_die "PD KV consumer belongs only to Attention"
  if [[ "$ROLE" == ffn && "${FFN_HOST_IP:-$HCCL_IF_IP}" != "$HCCL_IF_IP" ]]; then
    afd_die "FFN_HOST_IP must equal the local HCCL_IF_IP on FFN"
  fi
  if [[ "$ROLE" == attention && "${ATTENTION_HOST_IP:-$HCCL_IF_IP}" != "$HCCL_IF_IP" ]]; then
    afd_die "ATTENTION_HOST_IP must equal the local HCCL_IF_IP on Attention"
  fi
  if [[ "${ATTENTION_HOST_IP:-$HCCL_IF_IP}" != "${FFN_HOST_IP:-$HCCL_IF_IP}" ]]; then
    [[ "$AFD_HOST" == "${FFN_HOST_IP:-}" ]] \
      || afd_die "cross-host AFD_HOST must equal FFN_HOST_IP"
  fi

  require_uint API_PORT "$API_PORT"
  require_uint AFD_PORT "$AFD_PORT"
  require_uint HCCL_IF_BASE_PORT "$HCCL_IF_BASE_PORT"
  require_uint MAX_MODEL_LEN "$MAX_MODEL_LEN"
  require_uint MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS"
  require_uint MAX_NUM_SEQS "$MAX_NUM_SEQS"
  require_uint DBO_DECODE_TOKEN_THRESHOLD "$DBO_DECODE_TOKEN_THRESHOLD"
  require_uint DBO_PREFILL_TOKEN_THRESHOLD "$DBO_PREFILL_TOKEN_THRESHOLD"
  ((API_PORT > 0 && API_PORT < 65536)) || afd_die "API_PORT is outside 1..65535"
  ((AFD_PORT > 0 && AFD_PORT < 65536)) || afd_die "AFD_PORT is outside 1..65535"
  ((HCCL_IF_BASE_PORT > 0 && HCCL_IF_BASE_PORT < 65536)) \
    || afd_die "HCCL_IF_BASE_PORT is outside 1..65535"
  ((MAX_MODEL_LEN > 0)) || afd_die "MAX_MODEL_LEN must be positive"
  ((MAX_NUM_BATCHED_TOKENS > 0)) \
    || afd_die "MAX_NUM_BATCHED_TOKENS must be positive"
  ((MAX_NUM_SEQS > 0)) || afd_die "MAX_NUM_SEQS must be positive"

  command -v "$PYTHON_BIN" >/dev/null || afd_die "Python command not found: $PYTHON_BIN"
  command -v "$VLLM_BIN" >/dev/null || afd_die "vLLM command not found: $VLLM_BIN"
  command -v ip >/dev/null || afd_die "the image must provide the ip command"
  command -v setsid >/dev/null || afd_die "the image must provide setsid"
  [[ -d "/sys/class/net/$NIC_NAME" ]] || afd_die "network interface does not exist: $NIC_NAME"
  ip -o -4 addr show dev "$NIC_NAME" scope global \
    | awk '{split($4, address, "/"); print address[1]}' \
    | awk -v expected="$HCCL_IF_IP" '$0 == expected {found=1} END {exit !found}' \
    || afd_die "HCCL_IF_IP=$HCCL_IF_IP is not assigned to NIC_NAME=$NIC_NAME"

  validate_devices
  validate_model_config

  local vllm_version
  vllm_version="$("$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

import afd_plugin  # noqa: F401
import vllm_ascend  # noqa: F401

print(version("vllm"))
PY
)"
  [[ "$vllm_version" == 0.23.0* ]] \
    || afd_die "this recipe requires vLLM 0.23.0, found $vllm_version"

  check_port_free "$API_HOST" "$API_PORT" "$ROLE API"
  check_port_free "$HCCL_IF_IP" "$HCCL_IF_BASE_PORT" "$ROLE HCCL base"
  if [[ "$ROLE" == ffn ]]; then
    check_port_free "$AFD_HOST" "$AFD_PORT" "AFD rendezvous"
  fi

  configure_execution

  export ASCEND_RT_VISIBLE_DEVICES="$ROLE_DEVICES"
  export HCCL_IF_IP
  export HCCL_IF_BASE_PORT
  export VLLM_HOST_IP="$HCCL_IF_IP"
  export GLOO_SOCKET_IFNAME="$NIC_NAME"
  export TP_SOCKET_IFNAME="$NIC_NAME"
  export HCCL_SOCKET_IFNAME="$NIC_NAME"
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"
  export HCCL_BUFFSIZE
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
  export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd}"
  export VLLM_USE_V1=1

  printf '[dsv4-afd] role=%s devices=%s ranks=%s dp=%s tp=%s mode=%s u_batches=%s model=%s\n' \
    "$ROLE" "$ROLE_DEVICES" "$ROLE_RANKS" "$ROLE_DP_SIZE" "$TENSOR_PARALLEL_SIZE" \
    "$EXECUTION_MODE" "$U_BATCHES" "$MODEL_PATH"
  if [[ "$EXECUTION_MODE" == full-decode-only ]]; then
    printf '[dsv4-afd] graph_max=%s capture_sizes=[%s] decode_query_len=%s\n' \
      "$MAX_CUDAGRAPH_CAPTURE_SIZE" "$CUDAGRAPH_CAPTURE_SIZES" "$((${DSPARK_BLOCK_SIZE:-0} + 1))"
  fi
}

run_role_service() {
  local service_pid='' service_status=0

  stop_service_group() {
    [[ -n "$service_pid" ]] || return 0
    kill -TERM -- "-$service_pid" 2>/dev/null || true
    local attempt
    for attempt in {1..30}; do
      kill -0 -- "-$service_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  }

  trap 'stop_service_group; exit 130' INT
  trap 'stop_service_group; exit 143' TERM
  trap 'stop_service_group' EXIT

  setsid "$@" &
  service_pid=$!
  set +e
  wait "$service_pid"
  service_status=$?
  set -e
  stop_service_group
  trap - EXIT INT TERM
  return "$service_status"
}
