# DeepSeek-V4 A5 HCCL P2P Recipe

This recipe launches a standalone DeepSeek-V4 Attention/FFN disaggregated
service with `P2pHcclAFDConnector` and vLLM 0.23. The default is TP1 and
A4F2 on one eight-device Atlas A5 host:

```text
client -> Attention API (DP4, NPU 0-3) -> FFN workers (DP2, NPU 4-5)
                    A0,A1 <-----------> F0
                    A2,A3 <-----------> F1
```

NPU 6 and 7 remain unused in this standalone topology.

## Runtime baseline

Use an image or environment containing this exact software family:

| Component | Baseline |
| --- | --- |
| Hardware | One eight-device Atlas A5 host |
| vLLM | `releases/v0.23.0`, commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`, commit `11ee45653b199a097805b87011824a81ffa51b95` |
| afd-plugin | `feat/dsv4-afd`; record the exact commit with `git rev-parse HEAD` |
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

`ATTENTION_RANKS` and `FFN_RANKS` count physical devices, including TP workers.
Each role's DP size is its rank count divided by `TENSOR_PARALLEL_SIZE`; use
the same TP size on both roles. The launchers support TP1 and TP2, with equal
A/F rank counts for TP2 and an integer rank-count ratio for TP1. Each role
uses 1–8 local devices. Match device lists to those counts and use disjoint
lists when A/F share a host.

For A4F4 on one host, change both launch shells to:

```bash
export ATTENTION_RANKS=4 FFN_RANKS=4
export ATTENTION_DEVICES=0,1,2,3 FFN_DEVICES=4,5,6,7
export TENSOR_PARALLEL_SIZE=1   # A DP4, F DP4
# Or use TENSOR_PARALLEL_SIZE=2 for A DP2/TP2, F DP2/TP2.
```

For A4F8/TP1, use separate A/F hosts with eight devices listed on F. Set
`ATTENTION_HOST_IP` and `FFN_HOST_IP` consistently in both shells and set
`AFD_HOST="$FFN_HOST_IP"`.

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

The optional graph mode fixes U2 to two stages and disables async
scheduling:

```bash
export EXECUTION_MODE=full-decode-only
export U_BATCHES=2
```

The scripts then enable DBO, set decode/prefill thresholds to 2/12, select
`FULL_DECODE_ONLY`, and enable the Graph/U2 compute, hybrid DAG,
Attention three-stream, FFN receive-stream, and FFN cross-layer paths. Override
capture sizes only when the replacement list has been validated with the same
image and workload. By default, the scripts derive sizes from `MAX_NUM_SEQS`
and the active speculative configuration:

```bash
unset MAX_CUDAGRAPH_CAPTURE_SIZE CUDAGRAPH_CAPTURE_SIZES
export MAX_NUM_SEQS=24
```

Attention defaults for this batch limit:

| Decode configuration | Graph max (tokens per DP rank) | Capture sizes |
| --- | ---: | --- |
| DSpark off | 24 | `1 2 4 8 16 24` |
| DSpark on, checkpoint block size 5 | 144 | `6 12 24 48 96 144` |

Dispatch compares the token count before U2 splitting. DSpark covers up to
`MAX_NUM_SEQS × (dspark_block_size + 1)` tokens. Explicit capture overrides
must cover this bound, include the exact full Decode shape, end at the declared
maximum, fit the role's batched-token budget, and use a maximum divisible by
the decode query length. A list without an explicit maximum supplies its own
maximum. Default buckets grow by powers of two and include the exact bound.

FFN runs without a speculative configuration. Its default CLI capture sizes
follow its own `MAX_NUM_SEQS`; remote FFN graphs follow Attention's capture
metadata. Size FFN's token buffers for the A/F aggregation and speculative
shapes. For A4F2/TP1 with A BS24, start F with `MAX_NUM_SEQS=48` and its
8192-token budget. Leave capture overrides unset in each role shell for
automatic defaults.

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

For an eager U2 comparison, `AFD_HCCL_EAGER_U2_STREAM_OVERLAP` defaults to `1`
and preserves an explicit `0` or `1`. Set it in **both** the Attention and FFN
launch shells after loading the site configuration:

```bash
export EXECUTION_MODE=eager U_BATCHES=2
export AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0
```

This keeps DBO/U2 enabled and disables eager stream overlap. The five Graph
switches are separate; setting them to `0` does not disable eager overlap or
Graph mode. `full-decode-only:2` always sets eager overlap to `0`. Restart the
roles into a fresh run directory before testing, and check effective values on
all six ranks. Use the request checks below for each configuration.

## Readiness and request

Wait until the Attention API responds:

```bash
until curl -fsS --max-time 10 http://127.0.0.1:8910/v1/models \
  >/tmp/dsv4-afd-models.json; do
  sleep 5
done
```

FFN has no HTTP readiness endpoint. Require all FFN DP engines to enter the
connector loop:

```bash
test "$(grep -c 'AFD FFN EngineCore started; workers run connector loop.' \
  /logs/dsv4-afd/ffn.log)" -ge "$((FFN_RANKS / TENSOR_PARALLEL_SIZE))"
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

For functional acceptance, configure `MAX_NUM_SEQS=32`, run batch sizes 1, 8,
and 32, cancel an in-flight request, verify a subsequent request succeeds,
and confirm a Graph/U2 run
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

## PD + AFD + DSpark

The standalone quick start above remains the A4F2 baseline. The scripts in this
directory also support the five PD combinations below. They are manual role
launchers; run them from the same checkout and pinned runtime on each host.

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
local device used by that role. Set `ASCEND_LOCAL_COMM_RES_PATH` explicitly
before starting A5 PD.

### Common setup

Run these exports from the repository root on **each** host before starting
its roles, replacing the example addresses and paths. `HCCL_IF_IP` must be
assigned to `NIC_NAME` on the current host. `PREFILL_HOST_IP`,
`ATTENTION_HOST_IP`, and `FFN_HOST_IP` must be
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
export GPU_MEMORY_UTILIZATION=0.91
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep
export PYTHON_BIN=/path/to/matched-venv/bin/python
export VLLM_BIN=/path/to/matched-venv/bin/vllm
export RECIPE="$PWD/recipe/npu/P2pHcclAFDConnector/deepseek_v4"
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT="/logs/dsv4-afd/$RUN_ID"
mkdir -p "$RUN_ROOT"
```

Generate `RUN_ID` once and reuse it in all role and client shells for the same
run. Each host keeps its own local logs and PID files under `RUN_ROOT`.

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

These are example topologies. Prefill uses `PREFILL_DP_SIZE × PREFILL_TP_SIZE`
physical devices. The generated Mooncake configuration records the actual
P DP/TP and A DP/TP, with Prefill TP at least Attention TP. For example,
A4F4/TP2 uses A DP2/TP2 and F DP2/TP2; a separate four-device Prefill host
can use `PREFILL_DP_SIZE=2 PREFILL_TP_SIZE=2` and `PREFILL_DEVICES=0,1,2,3`.
Set topology variables consistently on all role shells.

### Manual start and stop

Set the case's execution variables and start each role from a fresh shell.
Use a fresh log per cold start and preserve it for evidence collection. Start
Prefill first, then FFN and Attention back-to-back; FFN can wait for Attention
at AFD rendezvous. Start the proxy after both HTTP backends respond.

```bash
export MODEL_PATH=/models/DeepSeek-V4-Flash-DSpark
export EXECUTION_MODE=full-decode-only U_BATCHES=2
export ENABLE_PD=1 ENABLE_DSPARK=1

nohup bash "$RECIPE/afd_prefill.sh" >"$RUN_ROOT/prefill.log" 2>&1 &
echo $! >"$RUN_ROOT/prefill.pid"
until curl -fsS --max-time 10 "http://$PREFILL_HOST_IP:8100/health" >/dev/null; do
  sleep 5
done

ENABLE_PD=0 ENABLE_DSPARK=0 nohup bash "$RECIPE/afd_ffn.sh" >"$RUN_ROOT/ffn.log" 2>&1 &
echo $! >"$RUN_ROOT/ffn.pid"
nohup bash "$RECIPE/afd_attention.sh" >"$RUN_ROOT/attention.log" 2>&1 &
echo $! >"$RUN_ROOT/attention.pid"
until curl -fsS --max-time 10 "http://$ATTENTION_HOST_IP:8910/health" >/dev/null; do
  sleep 5
done

nohup bash "$RECIPE/afd_proxy.sh" >"$RUN_ROOT/proxy.log" 2>&1 &
echo $! >"$RUN_ROOT/proxy.pid"
```

On three hosts, run only the role assigned to each host; start the proxy on a
host that can reach both APIs. The FFN role always runs with `ENABLE_PD=0` and
`ENABLE_DSPARK=0`. Attention is the Mooncake KV consumer and the sole DSpark
drafter. Check Prefill and Attention `/health`, then require
`FFN_RANKS / TENSOR_PARALLEL_SIZE` FFN
connector-loop log markers. Send validation traffic through proxy port 9000.
To stop, stop sending new requests, let active requests drain, and stop Proxy.
Then terminate Attention and wait for FFN's shutdown marker before stopping
FFN and Prefill. On each host, use the PID files for its local roles:

```bash
kill -TERM "$(cat "$RUN_ROOT/proxy.pid")"
kill -TERM "$(cat "$RUN_ROOT/attention.pid")"
# Wait for all FFN ranks to log the shutdown payload before stopping FFN.
grep 'AFD NPU FFN received Attention shutdown payload' "$RUN_ROOT/ffn.log"
kill -TERM "$(cat "$RUN_ROOT/ffn.pid")"
kill -TERM "$(cat "$RUN_ROOT/prefill.pid")"
npu-smi info
```

### Request checks

Send requests to the Proxy after Prefill and Attention are ready:

```bash
export BASE_URL=http://127.0.0.1:9000
curl -fsS --max-time 300 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-afd",
    "messages": [{"role": "user", "content": "Write a short greeting."}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

Check the response text, then repeat with streaming and concurrent requests.
Cancel an in-flight request and confirm a subsequent request succeeds. For
Graph/U2, check online Graph replay and both U2 stages; for DSpark, collect
drafted/accepted counts before and after requests.

### Functional regression tests

Run these CPU unit tests from the repository root in a Python environment with
`pytest`, CPU PyTorch, `msgspec`, `httpx` and `fastapi`:

```bash
"$PYTHON_BIN" -m pytest \
  tests/unit/compat/patches/test_dspark_pd.py \
  tests/unit/compat/patches/test_dspark_pd_layout.py \
  tests/unit/compat/patches/test_mxfp_alltoallv.py \
  tests/unit/v1/worker/test_npu_mla_graph.py \
  tests/unit/tools/test_a5_pd_recipe.py \
  tests/unit/tools/test_proxy_response_errors.py
```

These tests cover DSpark scheduling, KV layouts and request tracking, MXFP
routing, Graph/U2 and RoPE state, role configuration, and Proxy error handling.

With the pinned vLLM/vLLM-Ascend runtime installed, also run the MTP runner
and model construction regressions:

```bash
"$PYTHON_BIN" -m pytest \
  tests/unit/v1/worker/test_npu_runtime.py \
  tests/unit/model_executor/models/test_deepseek_v4_construction.py
```

### Accuracy evaluation

Run GSM8K through the same Proxy endpoint with the offline dataset and an
environment containing `lm_eval`. Record the dataset hash, task configuration,
model/runtime versions, sample count, few-shot count and decoding settings:

```bash
export TASK_DIR=/data/eval/offline-gsm8k-task
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PYTHON_BIN" -m lm_eval \
  --model local-completions \
  --model_args "model=dsv4-afd,base_url=$BASE_URL/v1/completions,max_tokens=1024,tokenized_requests=False,tokenizer=$MODEL_PATH" \
  --include_path "$TASK_DIR" --tasks gsm8k \
  --num_fewshot 5 --batch_size 1 --limit 1319 \
  --gen_kwargs 'temperature=0,top_p=1' --seed 1024 \
  --output_path "$RUN_ROOT/gsm8k" --log_samples
```

Keep the protocol identical across compared configurations. Save every sample
and report the completed sample count and `exact_match,strict-match` score.

### Performance measurement

For performance runs, set the following in the role launch shells and restart
the roles into a fresh run directory:

```bash
export AFD_HCCL_STAGE_DIAGNOSTICS=0
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export VLLM_LOGGING_LEVEL=INFO
```

Use the matching vLLM benchmark CLI; this example measures random input load:

```bash
"$VLLM_BIN" bench serve \
  --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
  --model dsv4-afd --tokenizer "$MODEL_PATH" \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 128 --max-concurrency 32 --request-rate inf --ignore-eos \
  --save-result --result-dir "$RUN_ROOT/performance" \
  --result-filename benchmark.json
```

Warm up before measurement and run each configuration three times with the
same inputs and generation settings. For DSpark comparisons, use the same real
text/code dataset through Prefill and Proxy. Report all runs and their median,
request success rate, output tokens/s, TTFT/TPOT percentiles, peak HBM and DSpark
acceptance rate. Report both total and per-device throughput, counting every
active role, including Prefill; record device count and topology for each group.
