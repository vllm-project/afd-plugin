---
title: E2E testing
kind: module
status: normative
owners:
  - "@yujuancao07"
primary_code_paths:
  - "tests/e2e/runner.py"
  - "tests/e2e/process_utils.py"
  - "tests/e2e/environment.py"
  - "tests/e2e/accuracy/**"
  - "tests/e2e/models/**"
related_code_paths:
  - "tests/e2e/README.md"
  - ".agents/skills/run-e2e/SKILL.md"
  - ".agents/skills/deploy-dsv4-flash-async/**"
depends_on:
  - "plugin_boundary.md"
  - "attention_runtime.md"
  - "ffn_runtime.md"
  - "connector_contracts.md"
  - "model_integration.md"
  - "execution_platforms.md"
  - "compatibility_and_patches.md"
validation_paths:
  - "tests/e2e/test_runner.py"
  - "tests/e2e/test_deploy_dsv4_flash_async_skill.py"
  - "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py"
  - "tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py"
  - "tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py"
  - "tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py"
  - "tests/e2e/environment.py"
  - "tests/e2e/models/deepseek_v4_flash/config.py"
  - "tests/e2e/models/deepseek_v4_flash/completions.py"
  - "tests/e2e/models/qwen3_moe/test_qwen3_moe.py"
  - "tests/e2e/models/qwen3_6/test_qwen3_6.py"
upstream_refs:
  - "vLLM 0.30.0 serving and shutdown interfaces"
  - "lm-evaluation-harness GSM8K task and local-completions API"
  - "pytest parameterized test IDs"
verified_platform_refs:
  - "CUDA DeepSeek-V2-Lite"
  - "Ascend NPU DeepSeek-V2-Lite"
  - "CUDA Qwen3 MoE"
  - "CUDA Qwen3.6 MoE"
related_issues: []
last_reviewed: 2026-10-13
---

# E2E testing

## Scope

An E2E case starts real vLLM services, runs a real workload, validates the
result, and cleans up every process it started.

This contract is independent of GitHub Actions and Buildkite. CI selects pytest
node IDs and supplies hardware paths; it does not redefine a case. Unit tests,
operator tests, benchmarks, and performance tests are outside this scope.

## Structure

```text
tests/e2e/
├── models/<model>/test_*.py  # Test entries and case lists
├── accuracy/<task>.py       # Accuracy tool and result parsing
├── environment.py           # Backend and role launch environment
├── runner.py                # Service startup, evaluation, and cleanup
└── process_utils.py         # Process-group termination and reaping
```

New model files do not copy service startup, lm-eval, signal handling, or
cleanup. Production code does not depend on the E2E harness.

## Normative invariants

- `E2E-INV-001` — A case **MUST** have a stable lower-kebab-case ID and cover
  behavior not already covered by an existing case.
- `E2E-INV-002` — A PR case **MUST NOT** use more than four unique devices.
  AFD gate topology **MUST** match the selected suite: DeepSeek-V2-Lite
  defaults use 2A2F; Qwen3 MoE and Qwen3.6 MoE use 2A1F. Weekly CI also
  selects the existing DeepSeek-V2-Lite 2A1F Graph + DBO case.
- `E2E-INV-003` — Cases sharing devices or ports **MUST** run sequentially,
  remain order-independent, and release owned process groups before the next
  case.
- `E2E-INV-004` — Gate cases **MUST** fail on missing setup, service failure,
  incomplete results, failed validation, or failed cleanup. They **MUST NOT**
  use `skip`, `xfail`, or success-on-empty behavior.
- `E2E-INV-005` — The harness **MUST** check service liveness before and after
  evaluation. Accuracy cases **MUST** also check evaluator exit status, sample
  count, `NaN`, and accuracy.
- `E2E-INV-006` — Child processes **MUST** use owned process groups.
  Cancellation **MUST** send `SIGTERM`, use a bounded grace period, then reap
  every leader. A harness `SIGKILL` escalation **MUST** fail the case.
- `E2E-INV-007` — CI **MUST NOT** lower the sample count, accuracy threshold,
  or required case set. It **MUST** select cases by pytest node ID.
- `E2E-INV-008` — A shared case **MUST** pass without skips on every platform
  where CI selects it. New behavior **MUST** have focused unit tests and real
  hardware evidence.

## Required coverage

| Case | Runtime | Devices | Purpose |
| --- | --- | ---: | --- |
| `afd-eager-2a2f` | Ascend NPU AFD eager, 2A2F | 4 | NPU lifecycle and eager smoke test. |
| `afd-graph-2a2f` | Ascend NPU AFD graph, 2A2F | 4 | NPU graph path. |
| `afd-graph-dbo-2a2f` | Ascend NPU AFD graph + DBO, 2A2F | 4 | NPU graph path with DBO. |
| `baseline-graph` | Native vLLM graph, DP4/TP1/EP4 | 4 | Non-AFD control. |
| `afd-v2-eager-dp2` | CUDA ModelRunnerV2 eager, Attention DP2 + FFN DP2 | 4 | V2 data-parallel lifecycle and accuracy. |
| `afd-v2-graph-dp2` | CUDA ModelRunnerV2 graph, Attention DP2 + FFN DP2 | 4 | V2 data-parallel full-decode graph path. |
| `afd-v2-eager-tp2` | CUDA ModelRunnerV2 eager, Attention TP2 + FFN TP2 | 4 | V2 tensor-parallel lifecycle and accuracy. |
| `afd-v2-graph-tp2` | CUDA ModelRunnerV2 graph, Attention TP2 + FFN TP2 | 4 | V2 tensor-parallel full-decode graph path. |
| `afd-v2-eager-dbo-dp2` | CUDA ModelRunnerV2 eager DBO, Attention DP2 + FFN DP2 | 4 | Accuracy and lifecycle with eager DBO enabled. |
| `afd-v2-graph-dbo-dp2` | CUDA ModelRunnerV2 graph DBO, Attention DP2 + FFN DP2 | 4 | Accuracy and lifecycle with graph DBO enabled. |

CUDA ready/merge CI selects seven DeepSeek-V2-Lite cases: the native baseline
and the six DP2/TP2/DBO V2 cases. The V2 DP2 cases replace the three duplicate
GPU 2A2F cases; the eager/DBO comparison retains 128 samples, concurrency 12
and the 0.27 accuracy threshold. The backend-neutral 2A2F IDs remain available
for NPU. The GPU module collects 12 cases, including local 1A1F and 2A1F
coverage; the NPU module collects its seven existing cases. Across the three
GPU model suites, 19 cases remain: DeepSeek 12, Qwen3 MoE four, and Qwen3.6
MoE three. Compared with the historical 23, three duplicate DeepSeek cases
are replaced and Qwen3.6 GDN DBO coverage is explicitly withdrawn.

GPU cases gate accuracy, service liveness and cleanup. AFD GPU graph/DBO
cases additionally use a test-only worker extension to collect bounded per-rank
execution summaries after evaluation. The summaries must prove actual replay
or two-stage execution; dummy/profile/capture work does not qualify. Pure
eager and native baseline cases, and performance comparisons, omit the probe.
NPU DBO retains its live split check from existing runtime logs.
CUDA ready/merge CI runs in one 80-minute job. Put slower coverage in a
scheduled job.

Prefer graph coverage. Keep one eager smoke test unless a feature cannot run in
graph mode.

`afd-eager-async-cam` is a separate NPU-only smoke test. It uses four devices
for Attention DP1/TP2 and FFN DP2/TP1/EP2. It is not part of the PR gate above.
The NPU async CAM cases may require SIGKILL escalation because a pending CAM
receive cannot be interrupted cleanly. For these scenarios, the runner
adds a unique run id and role marker to each launched process environment.
After normal process-group cleanup, it finds every FFN process carrying those
markers through `/proc/*/environ` and sends SIGKILL to each matching PID. This
also finds workers that changed process groups or were reparented after their
launcher exited, without selecting another concurrent E2E run. NPU workers can
stay blocked in uninterruptible driver teardown for tens of seconds after
SIGKILL (measured up to ~45s on A3), so this path waits up to 120s before
reporting survivors. Marker-scan signal-delivery and survivor failures, plus
normal process-reaping failures, remain fatal. This test-scoped exception
should be removed when the runtime supports graceful cancellation.

`afd-dsv4-flash-async-cam-dp2tp4-ep8` is a separate local-only DSV4 Flash
W4A8 case using 16 Ascend NPUs: Attention DP2/TP4 and FFN DP8/TP1/EP8.
It uses eager async CAM and MBT=8192. Its checked-in legacy ubatching flag
enables two native DBO children on Attention; `async_moe_split="token"` is
accepted but ignored. Native DBO requires target-device validation. It validates
ten simultaneous chat requests, records their outputs and overlapping
request intervals, and checks service liveness and cleanup. Its cancellable
async HTTP client saves per-request responses/errors even on failure or
interruption, before the runner tears down services. Model-specific fixed
settings live alongside the model entrypoint. It does not run
GSM8K or claim general accuracy coverage. It uses the same scoped async NPU
FFN cleanup exception above and is not selected by the four-device PR gate.

`afd-dsv4-flash-sync-camp2p-2a2f` (A5) and `afd-dsv4-flash-sync-camp2p-8a8f`
(A3) are local synchronous DSV4 Flash profiles using `CAMP2pAFDConnector` and
FFN-side gating. A5 uses four devices, Attention/FFN DP2/TP1, ACL graph, and
4096 context; A3 uses sixteen devices, Attention DP2/TP4, FFN DP8/TP1, eager
execution, and the 8192/1024 budget. Plugin a2e/e2a operators transfer
activations and Hash-layer token IDs without a CAM vendor package. These
profiles check ten concurrent, nonempty, finished responses and service
liveness; they do not check answer correctness. A5 keeps DBO off and still
produces incorrect concurrent answers. A3 answer correctness remains
unverified. These smoke cases do not qualify synchronous DSV4 accuracy; see
the [profile limits](../../../tests/e2e/README.md#dsv4-flash-sync-camp2p-concurrent-requests-local-4-or-16-npus).
The async case retains its exact-answer check.

The 2A1F cases use three of the four devices (two Attention ranks and one
FFN rank). DeepSeek-V2-Lite eager/graph remain local-only; its graph+DBO
case and all Qwen3/Qwen3.6 suite scenarios run in weekly CI.

The ModelRunnerV2 matrix is CUDA-only in the current E2E harness. Its local
1A1F cases are `afd-v2-eager-1a1f` and `afd-v2-graph-1a1f`; CI selects the four
DP2/TP2 and two DBO cases by exact node ID on `l4_4`. Ascend ModelRunnerV2
has separate V2-Lite hardware evidence recorded in [PR #425](https://github.com/vllm-project/afd-plugin/pull/425#issuecomment-6063910923);
the generic V2 scenario entry points remain CUDA-only.

The Qwen3.5/3.6 adapter-family suite uses `Qwen/Qwen3.6-35B-A3B`, the native
Qwen3.5/3.6 model boundary, and `--language-model-only`. All current GPU
scenarios select native ModelRunnerV2; earlier V1 hardware records retain
their original scope. Its default suite uses native DP4/TP1/EP4 for
`baseline-graph`, and synchronous AFD 2A1F for `afd-eager` and `afd-graph`.
Qwen3.5/3.6 GDN DBO is temporarily withdrawn. Native hybrid/multimodal
validation applies; its graph+DBO node
is removed from the suite and weekly CI. Earlier H20
Graph + DBO and six-cell recurrent-state/CUDA Graph parity results describe
the removed adapter and are historical evidence only. Earlier evidence also
recorded an FFN CUDA illegal memory access during a live two-ubatch decode
split before v0.30. Neither historical record qualifies current GDN DBO support.
Qwen3 MoE retains its graph+DBO case in
[weekly E2E](../../../tests/e2e/README.md#weekly-gsm8k).
Multimodal, NPU, `compute_gate_on_attention=true`, pipeline-parallel,
asynchronous, and multi-node execution are outside this case; quantization is
unverified.

## Accuracy gate

| Setting | PR | Weekly |
| --- | ---: | ---: |
| Task | GSM8K | GSM8K |
| Few-shot examples | 8 | 8 |
| Generated-token limit | 512 | 512 |
| Samples | first 7 for ordinary cases; default DBO floors at 24; V2 DP2 eager/DBO comparison defaults to 128 | first 7; all DBO cases floor at 24 |
| Metric | GSM8K exact match | GSM8K exact match |
| Minimum accuracy | DeepSeek-V2-Lite smoke (up to 24 samples): 0.25; otherwise 0.27 | DeepSeek-V2-Lite smoke (up to 24 samples): 0.25; otherwise 0.27 |
| Cases | seven CUDA DeepSeek-V2-Lite cases; four existing NPU DeepSeek-V2-Lite cases | seven Qwen3 MoE / Qwen3.6 MoE cases plus DeepSeek-V2-Lite `afd-graph-dbo-2a1f` |

An accuracy of `0.27` requires at least 2 correct answers out of 7, 7 out of
24, or 35 out of 128. The DeepSeek smoke threshold of `0.25` requires 6 out
of 24; the lower threshold applies only at sample counts up to 24.

`afd-v2-eager-dp2`, `afd-v2-eager-dbo-dp2`, and
`afd-v2-graph-dbo-dp2` form the baseline/eager-DBO/graph-DBO comparison.
`V2_DBO_SAMPLE_LIMIT=128` and 12 concurrent requests are selected by
`runner.py`; the explicit local sample override still floors at 24.

- PR and weekly CI leave `AFD_GSM8K_LIMIT` unset.
- Set `AFD_GSM8K_LIMIT=all` locally for a full 1319-sample run.
- Other limits are for local debugging, not CI gates.
- DBO scenarios (`--enable-dbo`) run GSM8K with 12 concurrent requests and
  floor the sample count at 24: a sequential client never satisfies the
  DP-wide split agreement, so live requests would never run as two ubatches
  and only warmup/capture would exercise the split path. The NPU gate asserts
  that at least one live two-ubatch step was recorded. GPU AFD graph/DBO cases
  use a test-only worker extension. After readiness, native collective RPC resets
  Attention counters and a shared marker arms FFN at its next call entry.
  Attention emits its summary through RPC after evaluation; a completion marker
  requests FFN summaries through its existing periodic worker health RPC. The
  harness waits up to 30 seconds for every rank while checking service liveness
  before teardown. It removes the markers after process cleanup; FFN does not
  service HTTP utilities. Every
  Attention/FFN rank must execute the requested feature. DBO requires matching
  two-stage A/F layouts; graph+DBO requires consecutive completed replays of
  the same layout.
  Dummy/profile/capture work and empty second microbatches do not qualify.
  Pure eager and native baseline cases do not load the extension;
  performance comparisons omit it. There is no per-step execution logging.
  `AFD_GSM8K_LIMIT=all` still bypasses the floor. On vLLM 0.30 the runner sets the DBO decode token
  threshold to 2 because upstream validates both DBO thresholds against the
  two-microbatch count; the prefill threshold stays at 8. `AFD_E2E_API_PORT_BASE`
  (default `8000`) offsets the API server ports on shared machines, and
  Default and dedicated GPU V2 scenarios preserve the upstream
  chunked-prefill default and explicit caller choices. The MRV2 B/E/G
  comparison retains the native token budget.
- CI leaves `AFD_GSM8K_THRESHOLD` unset or raises it.
- Use the official GSM8K task, `HF_HOME`, and `results_*.json`. Do not commit a
  seven-row dataset or custom task YAML.

## Adding a case

1. State the coverage gap and why an existing case cannot catch it.
2. Choose one stable case ID and one fixed configuration.
3. Confirm platforms, devices, ports, CI tier, and runtime budget.
4. Reuse `runner.py`; add a task adapter only for a new evaluation task.
5. Do not add unit tests for harness code under `tests/e2e`; running the case
   validates commands, topology, flags, and result checks.
6. Run the case on every selected platform. Record versions, duration, result.
7. Update `tests/e2e/README.md`. Update the `run-e2e` skill only when commands,
   variables, prerequisites, or the default suite change.

Model entrypoints reuse the subprocess and download helpers in
`tests/conftest.py`.

## Host integration checks

Run host-only process/signal integration coverage explicitly with
`pytest tests/e2e/test_runner.py`; it launches local processes and sockets
without model weights or accelerators.

The DSV4 deployment skill also has host-only shell integration coverage. Run
`python -m unittest discover -s tests/e2e -p 'test_deploy_dsv4_flash_async_skill.py'`
to check its CLI/environment contract, benchmark result handling, and frozen
workload with local subprocesses. It uses a CLI double and synthetic benchmark
results; it provides no device, model accuracy, or performance evidence.
