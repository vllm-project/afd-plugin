# DeepSeek-V2-Lite synchronous NPU: v0.28.0

## Runtime and evidence

Use vLLM `v0.28.0` at `2cf0a6915ce544dc493a0990f2ea38d81601128a`
and vLLM-Ascend `bd69bad88fc19e1aeeea585416d408df8bda8fef`.
Source identity takes precedence over an image tag. This experiment used an
existing image with these source checkouts, not a newly qualified release image.
See the [installation stack](../../../../README.md#ascend-npu-installation).

Hardware: Ascend 910C/A3, four visible dies; openEuler 24.03 LTS-SP4,
Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0,
Triton 3.5.0 / Triton Ascend 3.2.2, memfabric-hybrid / memcache-hybrid 1.2.0.
AFD CANN ops were built with `AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=910c`.
The environment's FastAPI conflict was explicitly accepted; inherited optional
profiling/dependency conflicts remain. `pip check` is not claimed clean.

Evidence at AFD `1b4ca7b` (2026-09-16):

| Cell | Layout | Result |
| --- | --- | --- |
| Native control | TP1/DP1 eager, no AFD | Real request returned `Paris`, clean exit |
| baseline-graph | DP4/TP1/EP4 | GSM8K-7: 2/7 strict and flexible |
| afd-eager-2a2f / afd-graph-2a2f / afd-graph-dbo-2a2f | Attention DP2/TP1, FFN DP2/TP1/EP2 | Each 2/7; all pass |
| afd-eager-2a1f / afd-graph-2a1f / afd-graph-dbo-2a1f | Attention DP2/TP1, FFN DP1/TP1 | Each 2/7; all pass |

The seven NPU cells passed without skips. A subsequent GPU-only V2 selection
was rejected before startup; those six cases are outside this NPU suite.
Scoped unit suite: 970 passed, 3 skipped, 1 deselected. Skips are the opt-in
2-rank operator test and two missing-extension tests (extension installed).
Three DSV4 async test files and one async-CAM profile case were excluded.
Separate real 3-rank CAMP2P tests validated equal token blocks; unequal blocks
reproduced the 2A1F defect fixed by `1b4ca7b`.

**Full GSM8K was stopped and deferred by the requester. No full-accuracy or
performance claim is made.** BF16, V1, synchronous CAMP2P and FFN-side gate
are the hardware-tested scope. Async CAM/DSV4 are excluded; NPU V2, W8A8,
PCP, Attention-side gate, speculative decoding and larger/multi-node topologies
are not qualified by these results. Historical V3.2 recipes retain their pins.

## Prepare

From the AFD repository root, in the matched Ascend environment:

```bash
AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=910c \
  python -m pip install --no-deps --no-build-isolation -e .
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model,ascend_model_loader,afd
export AFD_E2E_BACKEND=npu
export AFD_E2E_DEVICES=0,1,2,3
export AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite
export AFD_NPU_E2E_VLLM_BIN=/path/to/environment/bin/vllm
export HF_HOME=/path/to/huggingface
export HF_ENDPOINT=https://huggingface.co
python -c 'from datasets import load_dataset; load_dataset("openai/gsm8k", "main")'
```

Use the checkpoint selected for your deployment; the validation checkpoint was
BF16 DeepSeek-V2-Lite. Install `lm_eval==0.4.13`, `datasets==5.0.1` and
`tenacity==9.1.4` in the evaluation environment. When using a system-site-packages
venv, verify that its `vllm` executable exists: inheriting a distribution does
not create its console script in the venv.

## Run the seven tested cells

This invokes the same paired runner used for the recorded results. It starts
both roles, checks readiness, evaluates requests and cleans up owned processes.
Ports 8000, 8001 and 1239 must be free. Run serially on the selected devices.

```bash
export AFD_GSM8K_LIMIT=7
python -m pytest -x -s -o addopts='' -ra \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[baseline-graph]' \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a2f]' \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a2f]' \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a2f]' \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-eager-2a1f]' \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-2a1f]' \
  'tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py::test_deepseek_v2_lite[afd-graph-dbo-2a1f]'
```

For the deferred full evaluation, set `AFD_GSM8K_LIMIT=all` and rerun the same
command. The runner expects 1319 samples per cell and retains the 0.27
threshold. Unsetting the variable selects **7**, not the full dataset.
The full command was started but not completed in this experiment.
On cancellation, send SIGTERM to pytest and allow its runner to clean up.
Verify no owned serving processes, listeners or NPU allocations remain.

CAMP2P fan-in requires equal per-Attention chunks. The V1 runner requests DP
padding even for eager execution when Attention ranks outnumber FFN ranks.
Do not remove that padding based on native MoE's variable-size capability.
