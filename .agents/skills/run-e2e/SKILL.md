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

- baseline-graph
- afd-eager-2a2f (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use afd-eager-2a1f)
- afd-graph-2a2f (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use afd-graph-2a1f)
- afd-graph-dbo-2a2f (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use afd-graph-dbo-2a1f)

Each gate scenario evaluates the first 7 GSM8K samples. The 2A2F AFD gate
scenarios (DeepSeek-V2-Lite only) use 2 Attention ranks and 2 FFN ranks.
`baseline-graph` uses native DP4/TP1/EP4. The 2A1F cases (`afd-eager-2a1f`,
`afd-graph-2a1f`, `afd-graph-dbo-2a1f`) use 2 Attention ranks and 1 FFN rank.
Note that for DeepSeek-V2-Lite 2A1F are local-only scenarios, while for Qwen3 MoE and
Qwen3.6 MoE they are the suite's gate scenarios.

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

Device order defines roles: the 2A2F AFD scenarios use the first two devices
for Attention DP2/TP1 and the last two for FFN DP2/TP1/EP2. `baseline-graph`
uses the first four for native DP4/TP1/EP4.
The 2A1F scenarios use the first
two for Attention and the third for FFN (used local-only for DeepSeek-V2-Lite; and
gate topology for Qwen3 MoE and Qwen3.6 MoE).

### 4. Run

From the repository root, stream output in the foreground:

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

The Qwen3.6 suite uses `Qwen/Qwen3.6-35B-A3B`, vLLM V1, and passes
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

Success means the selected suite reports 4 passed and 0 skipped. Report the
failed scenario, first actionable error, and cleanup status. Any skip is a
gate failure.

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
