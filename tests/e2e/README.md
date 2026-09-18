# End-to-End Tests

These tests validate DeepSeek-V2-Lite on real GPU or Ascend NPU hardware,
Qwen3 MoE on real GPU hardware, and Qwen3.6 MoE through the Qwen3.5/3.6
adapter family on real CUDA hardware.
Each default gate runs four scenarios:

- `baseline-graph`
- `afd-eager-2a2f` (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use `afd-eager-2a1f`)
- `afd-graph-2a2f` (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use `afd-graph-2a1f`)
- `afd-graph-dbo-2a2f` (DeepSeek-V2-Lite gate only; Qwen3 MoE/Qwen3.6 use `afd-graph-dbo-2a1f`)

Each scenario evaluates the first 7 GSM8K samples; the DBO scenarios run 24
samples with 12 concurrent requests so that live requests actually execute
as two ubatches (see the accuracy gate in
`docs/design/module/e2e_testing.md`). If `AFD_E2E_DEVICES` is set,
that value is used as-is; otherwise the defaults are:

- `0,1,2,3` for the gate scenarios. The 2A2F AFD cases (DeepSeek-V2-Lite gate
  only) use the first two for Attention DP2/TP1 and the last two for FFN
  DP2/TP1/EP2; `baseline-graph` uses all four for DP4/TP1/EP4.
- The 2A1F cases use the first two for Attention and the third for FFN,
  leaving the fourth device idle. For DeepSeek-V2-Lite these are local-only
  cases; for Qwen3 MoE and Qwen3.6 MoE they are the suite's gate scenarios.

Tests run sequentially and must not skip. Every GSM8K evaluation uses 8
few-shot examples and a 4096-token maximum model length.

See the [E2E testing design](../../docs/design/module/e2e_testing.md) before
adding a model or case.

## Run

Run from the repository root. The environment needs `vllm`, `pytest`,
`afd_plugin`, `lm_eval`, `datasets`, and `huggingface_hub`. NPU also needs
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
# DeepSeek-V2-Lite gate scenarios
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

Success means 4 passed and 0 skipped.

The repository CUDA E2E evidence for the Qwen3.5/3.6 adapter family uses
`Qwen/Qwen3.6-35B-A3B`, vLLM V1, and the
`P2pNcclAFDConnector` on CUDA. Every scenario passes `--language-model-only`;
multimodal execution is not covered. `baseline-graph` uses native DP4/TP1/EP4
on four devices. The three AFD scenarios use synchronous 2A1F: Attention on
the first two devices and FFN on the third. The fourth device remains unused
by AFD scenarios. The suite uses the same GSM8K-7, eight-shot, 4096-token,
0.27 minimum exact-match gate as the other default suites. NPU, multimodal,
`compute_gate_on_attention=true`, pipeline-parallel, asynchronous, and
multi-node execution are not covered; quantization is unverified.

### DeepSeek-V2-Lite local 2A1F cases

The DeepSeek-V2-Lite 2A1F scenarios are local-only; its CI gate selects the
2A2F scenarios. They use the first two devices for Attention DP2/TP1 and the
third for FFN DP1/TP1/EP1, and run GSM8K-7.

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a1f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a1f]" \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a1f]"
```

### GPU ModelRunnerV2 evidence matrix

The GPU-only ModelRunnerV2 regression matrix contains six representative
scenarios:

- `afd-v2-eager-1a1f` and `afd-v2-graph-1a1f`
- `afd-v2-eager-dp2` and `afd-v2-graph-dp2`
- `afd-v2-eager-tp2` and `afd-v2-graph-tp2`

The 1A1F scenarios use two devices and are local-only. DP2 and TP2 use four
devices, split evenly between Attention and FFN, and run in the CI gate on
`l4_4`. These rows record hardware-tested coverage; they are not a production
topology allowlist. Other valid DP/TP topologies use the same AFD and native
vLLM topology contracts.

```bash
python -m pytest -q -s \
  "tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py" \
  -k 'afd-v2'
```

### Weekly GSM8K

The weekly pipeline runs the Qwen3 MoE and Qwen3.6 MoE suites (baseline,
eager, and graph scenarios; the DBO scenario is excluded pending the FFN
CUDA fault investigation from build 68) plus the DeepSeek-V2-Lite
`afd-graph-dbo-2a1f` scenario, all with the default GSM8K sample limit. To
reproduce locally:

```bash
# Qwen3 MoE (all four scenarios)
python -m pytest -q -s tests/e2e/models/qwen3_moe/test_qwen3_moe.py

# Qwen3.6 MoE (all four scenarios)
python -m pytest -q -s tests/e2e/models/qwen3_6/test_qwen3_6.py
```

For a full 1319-sample run, export `AFD_GSM8K_LIMIT=all` before invoking
pytest. Without `AFD_GSM8K_LIMIT`, each scenario evaluates the first 7
samples.

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
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py
```

The CAM/CANN runtime and custom operators must already be installed. Missing
model configuration or a device list other than four unique IDs fails the
test.

## NPU async CAM ubatching test

This case runs the `afd-async-ubatch` scenario (`CAMAsyncAFDConnector` with
AFD-managed two-stage MoE token-split ubatching). It uses three NPUs: the first
two for Attention DP=1/TP=2, and the last one for FFN DP=1/TP=1. Unlike the
smoke test above, it reuses the shared runner's GSM8K path with batch size 2
(first 7 samples, 8-shot, sample-count and accuracy gates).

```bash
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
python -m pytest -q -s \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py -k async_ubatch
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
