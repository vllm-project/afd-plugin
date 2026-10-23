---
name: run-e2e
description: Use when the user asks to run, validate, or diagnose the AFD plugin's DeepSeek-V2-Lite GPU/NPU, Qwen3 MoE GPU, or Qwen3.6 MoE CUDA end-to-end tests through the Qwen3.5/3.6 adapter family, including PR-gate E2E, GSM8K-7 accuracy, graph, eager, DBO, or 2A2F scenarios.
---

# Run AFD E2E Tests

## Scope

Run one of the model suites:

- `tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py` on GPU or NPU
- `tests/e2e/models/qwen3_moe/test_qwen3_moe.py` on GPU
- `tests/e2e/models/qwen3_6/test_qwen3_6.py` on CUDA (text-only Qwen3.6
  evidence for the Qwen3.5/3.6 adapter family)

To split a scenario's ranks across multiple pods/nodes on a Kubernetes
cluster instead of running single-process, see
`resources/k8-multi-pod.md`.

Each suite contains four gate scenarios:
DeepSeek-V2-Lite on GPU has seven gate scenarios:

- `baseline-graph`
- `afd-v2-eager-dp2` and `afd-v2-graph-dp2`
- `afd-v2-eager-tp2` and `afd-v2-graph-tp2`
- `afd-v2-eager-dbo-dp2` and `afd-v2-graph-dbo-dp2`

The V2 DP2 cases replace the duplicate GPU 2A2F cases. NPU retains its four
DeepSeek gate cases: `baseline-graph`, `afd-eager-2a2f`, `afd-graph-2a2f`,
and `afd-graph-dbo-2a2f`. Qwen3 MoE retains four GPU cases: baseline plus
AFD eager, graph, and graph+DBO on 2A1F. Qwen3.6 MoE retains three: baseline
plus AFD eager and graph on 2A1F. Qwen3.5/3.6 GDN DBO is unsupported.

Ordinary cases evaluate the first 7 GSM8K samples, with the existing
24-sample minimum for DBO. GPU `afd-v2-eager-dp2`, `afd-v2-eager-dbo-dp2`
and `afd-v2-graph-dbo-dp2` retain 128 samples, concurrency 12,
`max_num_seqs=8` and the 0.27 accuracy threshold. GPU cases check accuracy,
service liveness and cleanup. AFD GPU graph/DBO cases also use a test-only
worker extension: after readiness, native collective RPC resets Attention
counters and a shared marker arms FFN at its next call entry. Attention reports
through RPC after evaluation; a completion marker requests FFN summaries through
its existing periodic worker health RPC. The harness waits up to 30 seconds for
every rank while checking service liveness before teardown. The markers are
removed after process cleanup; the FFN daemon does not service HTTP utilities. All Attention/FFN ranks must
execute the requested feature. DBO requires matching two-stage A/F layouts;
graph+DBO requires consecutive completed replays of the same layout. Dummy/profile/
capture work and empty second microbatches do not qualify.
Pure eager and native baseline cases do not load the probe,
and performance comparisons omit it. No per-step logging or runner identity
checks are used. NPU DBO retains its existing live split check.

DeepSeek 2A2F cases use 2 Attention ranks and 2 FFN ranks.
`baseline-graph` uses native DP4/TP1/EP4. The 2A1F cases (`afd-eager-2a1f`,
`afd-graph-2a1f`, `afd-graph-dbo-2a1f`) use 2 Attention ranks and 1 FFN rank.
DeepSeek-V2-Lite 2A1F eager/graph are local-only; its Graph + DBO case
also runs in weekly CI. All Qwen3 MoE and Qwen3.6 MoE 2A1F scenarios are
suite gate scenarios.

## Workflow

### 1. Select the backend

Honor an explicit backend. Otherwise inspect nvidia-smi -L and npu-smi info.
If both are available, ask which to use. If neither is available, stop.

### 2. Validate prerequisites

Before starting pytest, confirm:

- AFD_E2E_DEVICES contains the device IDs required by test cases.
- The backend model variable is set to a local path, or the environment can
  download the selected suite's checkpoint via huggingface_hub.
- The selected vllm command runs.
- pytest, afd_plugin, lm_eval, datasets, and huggingface_hub are importable
  (`uv sync --group dev --group e2e-tests` installs all of these.
- HF_HOME points to the Hugging Face cache used for GSM8K and model weights.
- HF_ENDPOINT is reachable: `gsm8k.py` defaults the lm-eval child to
  `https://hf-mirror.com`, so an unreachable mirror stalls or fails at GSM8K
  dataset resolution only after the servers are already up and devices
  reserved. Confirm the default mirror is reachable or override it.
- GPU: the selected devices are visible to CUDA.
- NPU: torch_npu and the Ascend runtime work.

`lm_eval` is declared in the `e2e-tests` uv dependency-group in
`pyproject.toml` and pinned in `uv.lock` (and baked into the CI image), so
the runner never needs an ad-hoc `pip install` for it. This keeps E2E runs
reproducible and removes a manual setup step every runner environment used
to require by hand.

Fail before pytest when a prerequisite is missing; never turn it into a skip.

Set HF_HOME before every run. The pytest entrypoint downloads/caches GSM8K and
the model when the backend model env var is unset.

### 3. Configure the run

For GPU:

~~~bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_GPU_E2E_MODEL=/path/to/model
# Optional: export AFD_GPU_E2E_VLLM_BIN=/path/to/vllm
~~~

For NPU:

~~~bash
export HF_HOME=/path/to/huggingface
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3
# Optional if the model is already local:
# export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
# Optional: export AFD_NPU_E2E_VLLM_BIN=/path/to/vllm
~~~

Device order defines roles: DeepSeek DP2 cases use the first two devices
for Attention DP2/TP1 and the last two for FFN DP2/TP1/EP2. GPU TP2 cases use
DP1/TP2 on both roles. `baseline-graph`
uses the first four for native DP4/TP1/EP4.
The 2A1F scenarios use the first
two for Attention and the third for FFN (the gate topology for Qwen3 MoE and Qwen3.6 MoE).

### 4. Run

From the repository root, stream output in the foreground. For the GPU
DeepSeek-V2-Lite gate (`AFD_E2E_BACKEND=gpu`):

~~~bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-dp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-tp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-tp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-eager-dbo-dp2]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-v2-graph-dbo-dp2]"
~~~

For the existing NPU DeepSeek gate (`AFD_E2E_BACKEND=npu`):

~~~bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a2f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a2f]"
~~~

For Qwen3 MoE on GPU, run:

~~~bash
python -m pytest -q -s \
  tests/e2e/models/qwen3_moe/test_qwen3_moe.py
~~~

For Qwen3.6 MoE on CUDA, run:

~~~bash
export AFD_E2E_BACKEND=gpu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_GPU_E2E_MODEL=/path/to/Qwen3.6-35B-A3B
python -m pytest -q -s \
  tests/e2e/models/qwen3_6/test_qwen3_6.py
~~~

The Qwen3.6 suite uses `Qwen/Qwen3.6-35B-A3B` and selects native vLLM V2
for every GPU runner invocation, including the baseline. FFN remains
connector-driven; there is no GPU V1 fallback. Qwen3.5/3.6 GDN DBO is
unsupported under native hybrid/multimodal validation. Earlier H20 Graph +
DBO and recurrent-state/CUDA Graph parity results describe the removed
adapter and are historical evidence only. The suite passes
`--language-model-only` to every runner invocation. It is CUDA-only and
text-only: `baseline-graph` is native DP4/TP1/EP4, while the AFD scenarios are
synchronous 2A1F with Attention on the first two devices and FFN on the third.
Multimodal, NPU, `compute_gate_on_attention=true`, asynchronous,
pipeline-parallel, and multi-node execution are not covered; quantization is
unverified.

Do not add backend markers or run scenarios in parallel; they share devices.

For the local DeepSeek-V2-Lite 2A1F cases, run the same pytest entrypoint with
`[afd-eager-2a1f]`, `[afd-graph-2a1f]`, or `[afd-graph-dbo-2a1f]`.

On cancellation, forward SIGTERM and allow over 90 seconds for cleanup.

### 5. Report

Success means all selected cases pass with zero skips: seven for the GPU
DeepSeek gate; four for the NPU DeepSeek gate or GPU Qwen3 MoE suite;
three for the GPU Qwen3.6 MoE suite.
The entire DeepSeek module includes local cases and collects 12 on GPU or
seven on NPU. Report the actual selected count, failed scenario, first
actionable error, and cleanup status. Any skip is a gate failure.

## Environment reference

| Variable | Backend | Required |
|---|---|---|
| AFD_E2E_BACKEND | both | yes: gpu or npu |
| AFD_E2E_DEVICES | both | yes: four unique IDs for the default suite |
| AFD_GPU_E2E_MODEL | GPU | no; downloads the selected suite's model when unset |
| AFD_GPU_E2E_VLLM_BIN | GPU | no; defaults to vllm |
| AFD_NPU_E2E_MODEL | NPU | no; downloads the selected suite's model when unset |
| AFD_NPU_E2E_VLLM_BIN | NPU | no; defaults to vllm |
| HF_HOME | both | recommended; HF dataset/model cache |
| HF_ENDPOINT | both | recommended; `gsm8k.py` defaults the lm-eval child to https://hf-mirror.com, confirm reachable or override |
