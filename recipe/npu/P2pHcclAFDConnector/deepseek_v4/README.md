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
| afd-plugin | `feat/dsv4-afd-phase1-delivery`, commit `57bcde14cc0da248bfcb8641476b8426811a64c5` |
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
