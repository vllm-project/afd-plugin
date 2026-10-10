# DeepSeek-V4 A5 A4F2 HCCL P2P Recipe

This recipe launches a standalone DeepSeek-V4 Attention/FFN disaggregated
service on one eight-device Atlas A5 host. It uses
`P2pHcclAFDConnector`, vLLM 0.23, TP1, and an A4F2 topology:

```text
client -> Attention API (DP4, NPU 0-3) -> FFN workers (DP2, NPU 4-5)
                    A0,A1 <-----------> F0
                    A2,A3 <-----------> F1
```

NPU 6 and 7 remain unused. This is a functional recipe. It does not establish
accuracy or performance results.

## Runtime baseline

Use an image or environment containing this exact software family:

| Component | Baseline |
| --- | --- |
| Hardware | One eight-device Atlas A5 host |
| vLLM | `releases/v0.23.0`, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`, commit `11ee45653b199a097805b87011824a81ffa51b95` |
| afd-plugin | `feat/dsv4-afd-phase1-delivery`; `02d0779` is the regression starting point. Use the frozen source commit in the delivery manifest for the updated recipe |
| CANN/HCCL | The exact versions recorded in the validated image manifest |

Do not source another CANN installation after entering the image. Record the
image digest and the exact CANN, HCCL, torch, torch-npu, vLLM, vLLM-Ascend, and
afd-plugin versions before treating a run as evidence.

The scripts require the native A5 DeepSeek-V4 checkpoint contract:

- `model_type=deepseek_v4` and architecture `DeepseekV4ForCausalLM`;
- `num_nextn_predict_layers=1`;
- dynamic FP8 `e4m3` weights with `ue8m0` scales and block size `[128, 128]`;
- optional `expert_dtype=fp4`.

The scripts use `--block-size 32`, safetensors prefetch, and automatic KV cache
dtype. They do not pass `--quantization ascend` for this native checkpoint.

## Configuration

Set these variables in the shell that launches both roles:

```bash
export MODEL_PATH=/models/DeepSeek-V4-Flash
export NIC_NAME=eth0
export HCCL_IF_IP=192.0.2.10
export AFD_HOST=127.0.0.1
export AFD_PORT=29761

export ATTENTION_DEVICES=0,1,2,3
export FFN_DEVICES=4,5
export ATTENTION_RANKS=4
export FFN_RANKS=2
export TENSOR_PARALLEL_SIZE=1
```

Replace the example NIC and IP with values visible inside the runtime
namespace. The preflight rejects an IP that is not assigned to `NIC_NAME`.

Role defaults:

| Setting | Attention | FFN |
| --- | ---: | ---: |
| API/internal port | 8910 | 8911 |
| HCCL base port | 51000 | 52000 |
| `max-num-batched-tokens` | 4096 | 8192 |
| `max-num-seqs` | 8 | 8 |
| Visible devices | 0,1,2,3 | 4,5 |

The FFN port is an internal vLLM launcher setting, not an HTTP health endpoint.
Only the Attention port serves client requests.

## Eager U1

Eager U1 is the default:

```bash
export EXECUTION_MODE=eager
export U_BATCHES=1
```

Start FFN first and Attention immediately afterward. Do not wait for FFN to
finish initialization because the two roles rendezvous with each other:

```bash
mkdir -p /logs/dsv4-afd

nohup bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh \
  >/logs/dsv4-afd/ffn.log 2>&1 &
echo $! >/logs/dsv4-afd/ffn.pid

nohup bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh \
  >/logs/dsv4-afd/attention.log 2>&1 &
echo $! >/logs/dsv4-afd/attention.pid
```

## FULL_DECODE_ONLY Graph U2

The validated optional graph mode fixes U2 to two stages and disables async
scheduling:

```bash
export EXECUTION_MODE=full-decode-only
export U_BATCHES=2
```

The scripts then enable DBO, set decode/prefill thresholds to 2/12, select
`FULL_DECODE_ONLY`, and enable the validated Graph/U2 compute, hybrid DAG,
Attention three-stream, FFN receive-stream, and FFN cross-layer paths. Override
capture sizes only when the replacement list has been validated with the same
image and workload:

```bash
export MAX_CUDAGRAPH_CAPTURE_SIZE=8
export CUDAGRAPH_CAPTURE_SIZES='1 2 4 8'
```

Start the roles with the same back-to-back commands used for eager mode.

The five Graph/U2 switches default to `1`. Explicit `0` overrides are preserved
by `common.sh`; set them in the shells launching Attention and FFN before start.
For an optional all-off comparison:

```bash
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
```

Only `0` and `1` are supported. If disabling FFN receive-stream separately,
also disable FFN cross-layer: cross-layer `1` requires receive-stream `1`.
Record effective values on all ranks and use a fresh run ID for the comparison.
Default all-on and explicit all-off results must each identify their configuration.

## Readiness and request

Wait until the Attention API responds:

```bash
until curl -fsS --max-time 10 http://127.0.0.1:8910/v1/models \
  >/tmp/dsv4-afd-models.json; do
  sleep 5
done
```

FFN has no HTTP readiness endpoint. Require both FFN ranks to enter the
connector loop:

```bash
test "$(grep -c 'AFD FFN EngineCore started; workers run connector loop.' \
  /logs/dsv4-afd/ffn.log)" -ge 2
```

Then send requests only to Attention:

```bash
curl -fsS http://127.0.0.1:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-afd",
    "messages": [{"role": "user", "content": "Write a short greeting."}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

For functional acceptance, run batch sizes 1, 8, and 32, cancel an in-flight
request, verify a subsequent request succeeds, and confirm a Graph/U2 run
records real stage 0 and stage 1 execution. These checks do not replace a
separate golden-output or performance evaluation.

## Shutdown

Stop Attention first so it can send the FFN shutdown payload. Stop FFN only
after all FFN ranks have logged that payload or after the documented timeout:

```bash
kill -TERM "$(cat /logs/dsv4-afd/attention.pid)"
wait "$(cat /logs/dsv4-afd/attention.pid)" 2>/dev/null || true

grep 'AFD NPU FFN received Attention shutdown payload' /logs/dsv4-afd/ffn.log
kill -TERM "$(cat /logs/dsv4-afd/ffn.pid)" 2>/dev/null || true
wait "$(cat /logs/dsv4-afd/ffn.pid)" 2>/dev/null || true
npu-smi info
```

Both launchers place vLLM in its own process group. A signal or launcher failure
terminates remaining processes in that role before returning a nonzero status.

## Troubleshooting

- **Preflight rejects the model:** use the native A5 DeepSeek-V4 checkpoint;
  do not rewrite an unrelated checkpoint's `config.json` to bypass the check.
- **Import or operator failure before loading:** verify the image manifest,
  driver compatibility, and that only the image's CANN environment is active.
- **AFD rendezvous timeout:** start FFN and Attention back-to-back, verify
  `AFD_HOST`/`AFD_PORT` are identical, and ensure no stale deployment owns the
  port.
- **HCCL connection failure:** verify device lists, `NIC_NAME`, `HCCL_IF_IP`,
  HCCL base ports, and the exact HCCL build recorded by the image.
- **Attention ready but requests hang:** require both FFN loop markers and
  inspect both logs for rank mapping, Graph key, or shutdown messages.
- **Graph/U2 shape falls back:** the implementation intentionally runs an
  unseen live shape through eager U2; add capture sizes only after a complete
  cold-start functional validation.

For connector internals and the complete capability boundary, see the
[HCCL P2P connector guide](../../../../docs/npu/HCCL_P2P_CONNECTOR_USER_GUIDE.md).

## 1030 PD + AFD + DSpark delivery recipe

The standalone quick start above remains the A4F2 baseline. The scripts in this
directory also support the five PD combinations below. They are manual role
launchers; run them from the same checkout and pinned runtime on each host.
These combinations require fresh hardware and accuracy evidence before they can
be called validated.

| Case | `EXECUTION_MODE` | `U_BATCHES` | Attention `ENABLE_DSPARK` |
| --- | --- | ---: | ---: |
| `pd-afd` | `eager` | 1 | 0 |
| `pd-afd-dspark` | `eager` | 1 | 1 |
| `pd-afd-graph-u2` | `full-decode-only` | 2 | 0 |
| `pd-afd-dspark-u2` | `eager` | 2 | 1 |
| `pd-afd-dspark-graph-u2` | `full-decode-only` | 2 | 1 |

Use a native A5 DeepSeek-V4 checkpoint satisfying the preflight contract above.
For DSpark cases, `MODEL_PATH` must be the DSpark checkpoint whose
`config.json` declares `dspark_block_size`. Use the same checkpoint and runtime
on all three roles. Attention derives `num_speculative_tokens` from that value
and runs DSpark eager even when target Decode uses Graph. In the pinned vLLM
0.23 stack, the launcher passes the compatibility method `mtp`; vLLM-Ascend
selects `AscendDSparkProposer` from the checkpoint's `dspark_block_size` marker.

### A5-only Mooncake HIXL configuration

For **A5 PD only**, Prefill producer and Decode Attention consumer must each
write their own local absolute resource path into
`kv_connector_extra_config.ascend_local_comm_res_path`, for example:

```json
{
  "kv_connector_extra_config": {
    "ascend_local_comm_res_path": "/etc/hixlep"
  }
}
```

Merge this field with the existing `prefill` and `decode` entries in the KV
configuration. The directory and resource files must exist on the corresponding
host; `/etc/hixlep` is an example. Setting `ASCEND_LOCAL_COMM_RES_PATH` alone
does not configure the connector. **A3 must not receive this field or require
this variable/directory. FFN does not create a Mooncake KV connector, and
standalone Decode-AF does not use this PD setting.**

The A5 Prefill/Attention launchers now write this field into the actual KV
configuration. Preflight requires an existing absolute directory and readable,
valid JSON files named `ub_endpoint_npu_<physical-device-ID>.json` for every
local device used by that role. The contents still need hardware validation.
Set `ASCEND_LOCAL_COMM_RES_PATH` explicitly before starting A5 PD. See the
[merged A5 validation guide](../../../../docs/npu/DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md)
for the M1 gate, default-enabled Graph switches, role setup and evidence checks.

### Common setup

Run these exports on **each** host before starting its roles, replacing the
example addresses and paths. `HCCL_IF_IP` must be assigned to `NIC_NAME` on the
current host. `PREFILL_HOST_IP`, `ATTENTION_HOST_IP`, and `FFN_HOST_IP` must be
identical across the role shells and reachable from each other.

```bash
export MODEL_PATH=/models/DeepSeek-V4-Flash
export NIC_NAME=eth0
export HCCL_IF_IP=192.0.2.10
export PREFILL_HOST_IP=192.0.2.10
export ATTENTION_HOST_IP=192.0.2.11
export FFN_HOST_IP=192.0.2.12
export AFD_HOST="$FFN_HOST_IP"
export AFD_PORT=29761
export PREFILL_ENGINE_ID=dsv4-prefill
export DECODE_ENGINE_ID=dsv4-decode
export PREFILL_KV_PORT=30000
export DECODE_KV_PORT=31000
export ATTENTION_RANKS=4
export FFN_RANKS=2
export TENSOR_PARALLEL_SIZE=1
export PREFILL_TP_SIZE=1
export PYTHON_BIN=/path/to/matched-venv/bin/python
export VLLM_BIN=/path/to/matched-venv/bin/vllm
```

The image must provide the matching CANN/HCCL, torch-npu, vLLM 0.23,
vLLM-Ascend and Mooncake transfer engine. Do not mix host CANN libraries with
the selected image.

For **one eight-device A5 host**, also set:

```bash
export PREFILL_DP_SIZE=2 PREFILL_DEVICES=0,1
export ATTENTION_DEVICES=2,3,4,5
export FFN_DEVICES=6,7
export PREFILL_HOST_IP="$HCCL_IF_IP"
export ATTENTION_HOST_IP="$HCCL_IF_IP"
export FFN_HOST_IP="$HCCL_IF_IP"
export AFD_HOST="$FFN_HOST_IP"
```

For **three A5 hosts**, set `PREFILL_DP_SIZE=8` on all role shells and
`PREFILL_DEVICES=0,1,2,3,4,5,6,7` on Prefill, `ATTENTION_DEVICES=0,1,2,3`
on Attention, and `FFN_DEVICES=0,1` on FFN. Device IDs can repeat on different
hosts. The A/F ratio remains A4F2; Prefill has a different card count. The
cross-host AFD rendezvous address is the FFN host IP, not `127.0.0.1`. Keep
the Mooncake KV port ranges, HCCL base ports (`50000/51000/52000` by default),
and AFD port reachable on the required interfaces. FFN's `8911` port is not a
health endpoint.

### Manual start and stop

Set the case's execution variables and start each role from a fresh shell.
Use a fresh log per cold start and preserve it for evidence collection. Start
Prefill first, then FFN and Attention back-to-back; FFN can wait for Attention
at AFD rendezvous. Start the proxy after both HTTP backends respond.

```bash
export EXECUTION_MODE=full-decode-only U_BATCHES=2
export ENABLE_PD=1 ENABLE_DSPARK=1

nohup bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_prefill.sh >prefill.log 2>&1 &
ENABLE_PD=0 ENABLE_DSPARK=0 nohup bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh >ffn.log 2>&1 &
nohup bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh >attention.log 2>&1 &
nohup bash recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_proxy.sh >proxy.log 2>&1 &
```

On three hosts, run only the role assigned to each host; start the proxy on a
host that can reach both APIs. The FFN role always runs with `ENABLE_PD=0` and
`ENABLE_DSPARK=0`. Attention is the Mooncake KV consumer and the sole DSpark
drafter. Check Prefill and Attention `/health`, then require two FFN
connector-loop log markers. Send validation traffic through proxy port 9000.
To stop, terminate Attention first and wait for FFN's shutdown marker, then
stop FFN, proxy and Prefill. Preserve the four logs for the functional,
accuracy and performance reports. These new combinations still require A5
hardware validation before they can be called supported.
