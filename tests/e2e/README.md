# End-to-End Tests

These tests validate DeepSeek-V2-Lite on real GPU or Ascend NPU hardware,
Qwen3 MoE on real GPU hardware, and Qwen3.6 MoE through the Qwen3.5/3.6
adapter family on real CUDA hardware.
The DeepSeek-V2-Lite GPU gate runs seven scenarios: `baseline-graph`,
`afd-v2-eager-dp2`, `afd-v2-graph-dp2`, `afd-v2-eager-tp2`,
`afd-v2-graph-tp2`, `afd-v2-eager-dbo-dp2`, and `afd-v2-graph-dbo-dp2`.
The V2 DP2 cases replace the duplicate GPU `afd-eager-2a2f`,
`afd-graph-2a2f`, and `afd-graph-dbo-2a2f` cases. NPU retains those three
2A2F cases plus `baseline-graph`. Qwen3 MoE retains four GPU gate cases:
baseline and AFD eager, graph, and graph+DBO on 2A1F. Qwen3.6 MoE retains
three: baseline and AFD eager and graph on 2A1F. GDN DBO is unsupported.

The full GPU model suite collects 19 cases, down from 23 before GPU V1
retirement: three duplicate DeepSeek cases are replaced, and Qwen3.6 GDN
graph+DBO support is explicitly withdrawn. The three duplicate cases have
retained counterparts with the same 2A2F DP2/TP1 topology and execution mode:

| Removed GPU case | Retained GPU case |
| --- | --- |
| `afd-eager-2a2f` | `afd-v2-eager-dp2` |
| `afd-graph-2a2f` | `afd-v2-graph-dp2` |
| `afd-graph-dbo-2a2f` | `afd-v2-graph-dbo-dp2` |

DeepSeek retains 12 cases, Qwen3 MoE four, and Qwen3.6 MoE three.
DeepSeek and Qwen3 MoE scenario coverage remains; Qwen3.6 retains ordinary
eager/graph coverage but loses its GDN DBO case. The current suite therefore
does not preserve all historical functional coverage. Workloads are not
identical: the retained DP2 eager/DBO comparison already uses 128 samples
instead of the removed eager case's
seven or graph+DBO case's 24-sample minimum. Ready/merge CI keeps the baseline
and six V2 cases; weekly CI includes Qwen3 MoE graph+DBO, but no Qwen3.6 DBO case.

Ordinary cases evaluate the first 7 GSM8K samples; DBO cases have a
24-sample minimum and 12 concurrent requests. The GPU DP2 eager/eager-DBO/
graph-DBO comparison retains 128 samples, concurrency 12 and the 0.27
accuracy threshold. GPU cases check accuracy, service liveness and cleanup.
AFD GPU and NPU graph/DBO cases load a test-only worker extension: after readiness
the harness resets Attention counters through native collective RPC, then creates
a shared marker to arm FFN at its next call entry. Attention reports through RPC
after evaluation; a completion marker requests FFN summaries through its existing
periodic worker health RPC. The harness waits for every rank while checking
service liveness, with a 30-second limit, before teardown.
The marker is removed after process cleanup. FFN does not service HTTP utilities.
Every participating rank must report the requested replay or two-stage
execution. Synchronous DBO requires matching A/F layouts; Async CAM verifies
Attention replicas independently because their DP layouts may differ.
graph+DBO requires consecutive completed two-stage replays of the same layout.
Dummy/profile/capture work and empty second microbatches do not qualify.
There is no per-step logging or runner-class
identity check. Pure eager and native baseline cases do not load this extension;
performance comparisons must omit it. NPU synchronous graph+DBO requires
completed joint two-stage replays, rather than separate graph/split counters.
Async DBO requires two nonempty Attention stages during evaluation. If `AFD_E2E_DEVICES` is set, that value
is used as-is; otherwise the defaults are:

- `0,1,2,3` for the gate scenarios. DeepSeek DP2 cases use the first two for
  Attention DP2/TP1 and the last two for FFN DP2/TP1/EP2. GPU TP2 cases use
  DP1/TP2 on both roles. `baseline-graph` uses all four for DP4/TP1/EP4.
- The 2A1F cases use the first two for Attention and the third for FFN,
  leaving the fourth device idle. DeepSeek-V2-Lite eager/graph are local-only;
  its graph+DBO case is selected by weekly CI. Qwen3 MoE selects all three AFD
  modes; Qwen3.6 MoE selects only AFD eager and graph.

Tests run sequentially and must not skip. Every GSM8K evaluation uses 8
few-shot examples and a 4096-token maximum model length.

See the [E2E testing design](../../docs/design/module/e2e_testing.md) before
adding a model or case.

## Run

Run from the repository root. The environment needs `vllm`, `pytest`,
`afd_plugin`, `lm_eval`, `datasets`, and `huggingface_hub` (install
by running `uv sync --group dev --group e2e-tests`). NPU also needs
`torch_npu`.

The selected test downloads/caches `openai/gsm8k` and its Hugging Face model
when the backend model env var is unset. Point `HF_HOME` at a persistent cache
if you want to reuse downloads across runs.

GPU:

```bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=gpu
# Optional: export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_GPU_E2E_MODEL=/path/to/model
```

NPU:

```bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=npu
# Optional: export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
```

Then run the selected model suite:

```bash
# DeepSeek-V2-Lite GPU gate (AFD_E2E_BACKEND=gpu)
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-dp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-tp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-tp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dbo-dp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-dbo-dp2]"

# DeepSeek-V2-Lite NPU gate (AFD_E2E_BACKEND=npu)
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a2f]"

# Qwen3 MoE
python -m pytest -q -s \
  tests/e2e/models/qwen3_moe/test_qwen3_moe.py

# Qwen3.6 MoE (text-only CUDA lane)
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_GPU_E2E_MODEL=/path/to/Qwen3.6-35B-A3B
python -m pytest -q -s \
  tests/e2e/models/qwen3_6/test_qwen3_6.py
```

Success means seven GPU DeepSeek gate cases, four NPU DeepSeek gate cases,
four GPU Qwen3 MoE cases, or three GPU Qwen3.6 MoE cases, passed with zero skips.
Running the entire DeepSeek module also selects
local cases: 12 on GPU, seven on NPU.

The Qwen3.5/3.6 adapter-family suite uses `Qwen/Qwen3.6-35B-A3B` and
`P2pNcclAFDConnector` on CUDA. The runner now selects native ModelRunnerV2
for every GPU scenario, including the baseline; FFN remains connector-driven.
Earlier V1 hardware records retain their original scope. Qwen3.5/3.6 GDN
DBO is unsupported under native hybrid/multimodal validation. Earlier H20
Graph + DBO and six-cell recurrent-state/CUDA Graph parity results describe
the removed adapter and are historical evidence only.
Every scenario passes `--language-model-only`;
multimodal execution is not covered. `baseline-graph` uses native DP4/TP1/EP4
on four devices. The two AFD scenarios use synchronous 2A1F: Attention on
the first two devices and FFN on the third. The fourth device remains unused
by AFD scenarios. The suite uses eight-shot GSM8K with a 4096-token model
length and a 0.27 minimum exact-match gate: seven samples for baseline,
eager, and graph. NPU, multimodal,
`compute_gate_on_attention=true`, pipeline-parallel, asynchronous, and
multi-node execution are not covered; quantization is unverified.

### DeepSeek-V2-Lite 2A1F cases

The DeepSeek-V2-Lite ready/merge gate selects 2A2F; weekly CI also runs
2A1F Graph + DBO. The 2A1F eager/graph cases remain local-only. They use the
first two devices for Attention DP2/TP1 and the third for FFN DP1/TP1/EP1,
and evaluate seven GSM8K samples, with the existing 24-sample minimum for DBO.

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a1f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a1f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a1f]"
```

### GPU ModelRunnerV2 scenarios

The GPU-only ModelRunnerV2 regression matrix contains eight representative
scenarios:

- `afd-v2-eager-1a1f` and `afd-v2-graph-1a1f`
- `afd-v2-eager-dp2` and `afd-v2-graph-dp2`
- `afd-v2-eager-tp2` and `afd-v2-graph-tp2`
- `afd-v2-eager-dbo-dp2` and `afd-v2-graph-dbo-dp2`

The 1A1F scenarios use two devices and are local-only. DP2 and TP2 use four
devices, split evenly between Attention and FFN. CI selects those four cases
and the two DP2 DBO cases, plus the native baseline, on `l4_4`. These rows are
regression scenarios, not a production topology allowlist. Other valid DP/TP topologies use the same AFD and native
vLLM topology contracts.

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py" \
  -k 'afd-v2'
```

### GPU MRV2 DBO development comparison

The `afd-v2-eager-dbo-dp2` and `afd-v2-graph-dbo-dp2` scenarios cover
the eager and graph DBO paths. Together with `afd-v2-eager-dp2`, these use the same
2A2F DP2/TP1 topology, 128 GSM8K samples, 12 concurrent requests, eight-shot
prompts, and `max_num_seqs=8`. The native token budget, prefix caching,
chunked-prefill and async-scheduling defaults and caller choices are preserved. Only DBO
and graph mode change; the graph row uses `FULL_DECODE_ONLY` with capture size 8.
`AFD_GSM8K_LIMIT` overrides the sample count consistently across the three rows
(with a shared minimum of 24 for diagnostics).

The GPU validator accepts exactly two microbatches at Attention DP1 or
DP > 1; FFN remains connector-driven. The DP1 path has H20 DeepSeek-V2-Lite
DP1/TP2 Graph + DBO validation with 128 GSM8K samples, recorded in
[PR #454](https://github.com/vllm-project/afd-plugin/pull/454).
NPU retains the separate eager DBO and graph
constraints documented in the [execution-platform matrix](../../docs/design/module/execution_platforms.md#tested-runtime-matrix).
The GPU harness collects bounded per-rank summaries of actual two-stage and
FULL-replay execution during evaluation. Historical hardware records retain
the execution assertions of the test revision that produced them. The restored
summary-based probe was revalidated on four H20s on 2026-10-14 (SLURM 5299):
eager+DBO passed at 40/128 (0.3125) and graph+DBO at 42/128 (0.328125).
Every Attention/FFN DP2/TP1 rank reported completed two-stage execution; the
graph case also confirmed matching A/F layouts and consecutive same-layout
replays. This targeted validation is not a fresh run of the current 19 GPU scenarios.
The `afd-graph-dbo-2a1f` case also selects
GPU V2 with a different topology and workload.
These generic MRV2 scenarios remain CUDA-only. NPU native DBO needs separate
model qualification; Async CAM enables it only on Attention.
Representative 300-question results and their limits are recorded in
[PR #425](https://github.com/vllm-project/afd-plugin/pull/425); these comparisons
do not establish numerical equivalence or full-dataset qualification.

### Weekly GSM8K

The weekly pipeline runs eight scenarios: all four Qwen3 MoE cases
(baseline, eager, graph, and graph+DBO), the three Qwen3.6 MoE cases
(baseline, eager, and graph), and DeepSeek-V2-Lite `afd-graph-dbo-2a1f`,
all with the default GSM8K sample limit. To reproduce locally:

```bash
# Qwen3 MoE (all four scenarios)
python -m pytest -q -s tests/e2e/models/qwen3_moe/test_qwen3_moe.py

# Qwen3.6 MoE (all three scenarios; GDN DBO unsupported)
python -m pytest -q -s tests/e2e/models/qwen3_6/test_qwen3_6.py
```

For a full 1319-sample run, export `AFD_GSM8K_LIMIT=all` before invoking
pytest. Without `AFD_GSM8K_LIMIT`, these scenarios evaluate the first 7
samples, with the existing 24-sample floor for DBO.

## DSV4 Flash async CAM concurrent requests (local, 16 NPUs)

`afd-dsv4-flash-async-cam-dp2tp4-ep8` runs Attention DP2/TP4 on the first
eight devices and FFN DP8/TP1/EP8 on the last eight. This is a standalone
Ascend 910C case, outside the four-device PR gate. Use DSV4 Flash W4A8
weights and a DSV4-capable vLLM/vLLM-Ascend runtime with CAM operators.

The fixed deployment uses eager execution, MBT=8192, max-model-len=1048576,
max-num-seqs=16, block-size=128, memory utilization=0.7, and seed=1024.
CAM uses `dynamicQuant=1`, Attention-side gating, and the legacy
`async_moe_ubatching` flag, which enables two native DBO children on
Attention. The retained `async_moe_split="token"` field is ignored by the
current runtime. Native DBO requires target-device validation.
The ten short prompts below do not guarantee the native DBO thresholds are
reached. This case retains its original concurrent-answer and cleanup oracle;
DSV4 DSA DBO requires separate live above-threshold qualification.
FlashComm1 is enabled only on Attention. CPU binding and
128-thread weight loading follow the reference prefill scripts. Prefix
caching and KV transfer are not enabled; FFN does not enable native DBO.

After startup, an async HTTP client schedules ten independent chat requests together.
They ask for `12 + 7` through `21 + 7`, with temperature=0, thinking=false,
and max_tokens=256. Every response must contain one nonempty answer and
finish with `stop`; HTTP errors, truncated answers, missing responses, or
non-overlapping request timings fail the case. The test prints all outputs
and saves complete responses and monotonic timestamps in the pytest temporary
directory, including per-request errors and partial results on cancellation.
SIGTERM/SIGINT cancels pending HTTP operations before service cleanup, without
waiting for the request timeout. It does **not** run GSM8K or establish general
model accuracy.
Service liveness and owned-process cleanup are also required to pass.
This 16-NPU case allows 60 seconds for service shutdown before escalation;
Attention SIGKILL escalation remains a failure. FFN uses the existing scoped
async CAM cleanup exception.

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V4-Flash-W4A8
export HCCL_IF_IP='<local-communication-ip>'
export HCCL_SOCKET_IFNAME=eth0
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py::test_deepseek_v4_flash_async_cam[afd-dsv4-flash-async-cam-dp2tp4-ep8]'
```

Defaults: API ports 19280/19281, AFD rendezvous port 6455, startup timeout
1800 seconds. Override these using `AFD_NPU_DSV4_E2E_API_PORT`,
`AFD_NPU_DSV4_E2E_AFD_PORT`, and `AFD_NPU_E2E_STARTUP_TIMEOUT`.
`AFD_NPU_E2E_VLLM_BIN` selects the executable. Build the plugin-owned 910C
operators before running. Model and all sixteen device IDs must be supplied explicitly; missing setup fails rather than skips.

## DSV4 Flash sync CAMP2P concurrent requests (local, 4 or 16 NPUs)

Two local-only scenarios run DeepSeek V4 Flash over the synchronous
`CAMP2pAFDConnector` — no CAM vendor package, since the plugin's own a2e/e2a
operators carry the activations and the Hash-layer token ids — each at its host's
recorded launch shape:

| Scenario | Host | Deployment | Devices |
| --- | --- | --- | --- |
| `afd-dsv4-flash-sync-camp2p-2a2f` | A5 (Ascend 950) | Attention DP2/TP1 + FFN DP2/TP1, expert parallel, ACL graph (`FULL_DECODE_ONLY`, capture 16), 4096 context, native DBO off, 128-token block, prefix caching off | 4 |
| `afd-dsv4-flash-sync-camp2p-8a8f` | A3 (Ascend 910C) | Attention DP2/TP4 + FFN DP8/TP1, expert parallel, eager, 8192 context | 16 |

Build the operators for the target SOC first (`SOC_VERSION=ascend950` on A5,
`910c` on A3). The device list is role-defining — the first `attention_ranks`
entries go to Attention and the rest to FFN, so A5 passes `2,3,0,1` for its
recorded mapping and A3 passes `0-7` for Attention with `8-15` for FFN — and a
list sized for the other host fails rather than skips.

Both profiles are **smoke cases**: the async case's ten concurrent chat requests
(`12 + 7` … `21 + 7`, temperature=0, thinking=false, max_tokens=256) must be served
together, each returning a nonempty answer that finished. Neither compares the
answer with the expected sum — A5 corrupts part of a concurrent batch (below) and
A3 has not been validated against the oracle — and `check_answer` on a profile
turns the exact check back on once its host is validated.

Deployment differences worth knowing:

- **A5** runs with DBO off, `--block-size 128`, and
  `--no-enable-prefix-caching`; it uses `HCCL_BUFFSIZE=2048`, the plain
  allocator, and no required NIC variable.
- **A3** drops an inherited `HCCL_BUFFSIZE`, sizes its own CAMP2P domains through
  `connector_extra_config`, and requires `HCCL_IF_IP` and `HCCL_SOCKET_IFNAME`.
- `--quantization` is resolved from the checkpoint: A5's FP8/W4A8 checkpoint
  decides, A3's int8 W8A8 loads through `ascend`.
- Both disable `multistream_dsv4_dsa_overlap` and `enable_dsa_cp` and leave
  the gate on FFN; KV transfer is not enabled. Shutdown allows 60 seconds, and
  the async FFN cleanup exception does not apply because no CAM receive is
  pending.

**Known limitation: A5 produces incorrect answers under concurrent load even
with DBO off.** A3 answer correctness remains unverified. Passing these smoke
cases does not establish synchronous DSV4 correctness. Their profiles and
limitations originate in [PR #359](https://github.com/vllm-project/afd-plugin/pull/359);
they are separate from the V2-Lite and asynchronous DSV4 accuracy evidence
recorded in [PR #425](https://github.com/vllm-project/afd-plugin/pull/425#issuecomment-6063910923).

```bash
export AFD_E2E_BACKEND=npu
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V4-Flash
# A5: four dies, Attention on 2,3 and FFN on 0,1
export AFD_E2E_DEVICES=2,3,0,1
export HCCL_IF_IP='<local-communication-ip>'   # optional on A5
export HCCL_SOCKET_IFNAME=eth0               # optional on A5
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py::test_deepseek_v4_flash_sync_camp2p[afd-dsv4-flash-sync-camp2p-2a2f]'
# A3: sixteen dies, Attention on 0-7 and FFN on 8-15
export AFD_E2E_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_IF_IP='<local-communication-ip>'   # required on A3
export HCCL_SOCKET_IFNAME=eth0              # required on A3
python -m pytest -q -s \
  'tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py::test_deepseek_v4_flash_sync_camp2p[afd-dsv4-flash-sync-camp2p-8a8f]'
```

Defaults: API ports 19380/19381, AFD rendezvous port 6456, startup timeout
1800 seconds. Override these using `AFD_NPU_DSV4_SYNC_E2E_API_PORT`,
`AFD_NPU_DSV4_SYNC_E2E_AFD_PORT`, and `AFD_NPU_E2E_STARTUP_TIMEOUT`.
`AFD_NPU_E2E_VLLM_BIN` selects the executable. The model and exactly the
scenario's device count must be supplied; a list sized for the other shape
fails rather than skips.

## Run with the Codex skill

The repository includes the [`run-e2e`](../../.agents/skills/run-e2e/SKILL.md)
skill. Open the repository in Codex and ask, for example:

```text
Use run-e2e to run the Qwen3 MoE GPU E2E tests with HF_HOME
/data/huggingface.
```

Provide `HF_HOME` and `AFD_E2E_BACKEND`. `AFD_E2E_DEVICES` is optional; when
unset, the test module picks the defaults above. The model path is optional
when Hugging Face download is available. The skill checks prerequisites, runs
the same four tests, and reports failures and process cleanup.

## NPU async CAM smoke test

This separate test still reads `AFD_E2E_DEVICES`. It uses four NPUs: the first
two for Attention TP=2, and the last two for FFN DP=2/TP=1. It sends one
prompt and requests 32 tokens. It does not run GSM8K.

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_cam
```

The CANN runtime and source-built AFD custom operators must be installed. Missing
model configuration or a device list other than four unique IDs fails the
test.

## NPU async CAM ubatching test

This case runs the `afd-async-ubatch` scenario (`CAMAsyncAFDConnector` with
native MRV2 two-stage DBO on Attention). It uses three NPUs: the first
two for Attention DP=1/TP=2, and the last one for FFN DP=1/TP=1. Unlike the
smoke test above, it reuses the shared runner's GSM8K path with batch size 2
(first 7 samples, 8-shot, sample-count and accuracy gates).

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_ubatch
```

The device count is derived from the scenario's Attention/FFN rank constants,
not hard-coded; `AFD_E2E_DEVICES` may supply any three unique NPU IDs. The
topology is fixed at 2A1F, with TP=2 on Attention as required by token split.

Prerequisites match the async CAM smoke test (CAM/CANN runtime and custom
operators), plus a reachable GSM8K dataset source — offline pods need a local
HF mirror (`HF_ENDPOINT`). Both async CAM tests configure a 4096 MB buffer only
for connector-owned HCCL groups and remove `HCCL_BUFFSIZE` from child process
environments so unrelated groups retain their normal defaults. See
[`docs/npu/TROUBLESHOOTING.md`](../../docs/npu/TROUBLESHOOTING.md).
