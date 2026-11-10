# AFD Recipes

This directory contains deployment and benchmark recipes for AFD connectors.
Read each recipe's support status before running it: some historical experiment
records are retained for provenance but are not supported by the current
runtime. Both GPU and Ascend target vLLM 0.32. The
[Ascend validation record](https://github.com/vllm-project/afd-plugin/pull/507) links the tested
scope; the V3.2 recipes below retain their historical versions.

## Directory layout

Recipes are organized by hardware backend, connector, and model:

- Hardware backend: `gpu` or `npu`.
- Connector: the exact connector class name, such as `P2pNcclAFDConnector`,
  `DeepEPM2NAFDConnector`, `CAMP2pAFDConnector`, or `CAMAsyncAFDConnector`.
- Model: the model family or variant in lowercase snake case, such as
  `deepseek_v2_lite` or `deepseek_v3_2`.

## Available recipes

| Hardware | Connector | Model | Recommended stage | Status | Recipe |
| --- | --- | --- | --- | --- | --- |
| GPU | `P2pNcclAFDConnector` | DeepSeek-V2-Lite | Decode | Deployment examples; see the linked README for validation scope | [Launch examples](gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md) |
| GPU (SM90+) | `DeepEPM2NAFDConnector` | DeepSeek-V2-Lite | Decode | Deployment examples matching the checked-in E2E layouts; hardware qualification is separate | [Launch examples](gpu/DeepEPM2NAFDConnector/deepseek_v2_lite/README.md) |
| GPU (SM90+ for `DeepEPM2NAFDConnector`) | `DeepEPM2NAFDConnector` or `P2pNcclAFDConnector` | Qwen3-30B-A3B | Prefill / decode; decode-only benchmark | Graph, eager and DBO scripts for any `LAYOUT` (default 2A2F; multi-pod such as `2A0F,0A2F` or `1A1F,1A1F`), the layouts of the Qwen3 MoE DeepEP M2N E2E cases | [Graph, eager, DBO](gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/README.md) |
| GPU (SM90+ for `DeepEPM2NAFDConnector`) | `DeepEPM2NAFDConnector` or `P2pNcclAFDConnector` | Qwen3-235B-A22B-FP8 | Prefill / decode; decode-only benchmark | FP8-payload graph, eager and DBO scripts for any `LAYOUT` (2A2F, 4A2F, 2 x 4A1F, ...); H200-class FFN GPUs | [Graph, eager, DBO](gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md) |
| GPU | `P2pNcclAFDConnector` | Qwen3.5-122B-A10B-FP8 | Decode | Graph script for any `LAYOUT` (default 3 pods, `2A0F,2A0F,0A2F`) | [Graph](gpu/P2pNcclAFDConnector/qwen3_5_122b_a10b_fp8/prefill_decode_colocation/graph.sh) |
| Ascend NPU | `CAMP2pAFDConnector` | DeepSeek-V3.2 | Decode | Historical v0.19.1rc1 recipe; not qualified for the 0.32 target | [Synchronous decode](npu/CAMP2pAFDConnector/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V3.2 | Prefill / decode | Historical v0.19.1rc1/v0.26 recipes; standard V3.2 is unsupported by the current Async CAM path | [Async CAM](npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V4-Flash W4A8/W8A8 DSpark | Separate P / decode, or colocated | v0.32; eager A TP1, F layered eager/FULL. W4A8 PD: native full-draft P TP4, A DP2, F EP4. W8A8 A/F: A DP2, F EP8; PD not qualified | [DSpark A/F](npu/CAMAsyncAFDConnector/deepseek_v4_flash/README.md) |

Open the model-level README before running a recipe. It documents the required
hardware and runtime baseline, topology, environment variables, launch order,
support status, and known limitations. Unless a recipe says otherwise, run its
commands from the repository root.

## Adding a recipe

Add new content under `recipe/<hardware>/<connector>/<model>/`. Each model
directory should contain a `README.md` describing prerequisites, topology,
launch commands, validation steps, and limitations. Keep model-specific launch
scripts, configuration files, and result images in the same directory so the
recipe can be moved or linked as one unit.

GPU `prefill_decode_colocation` recipes are short scripts on the shared
[colocation launcher](gpu/common/README.md): a recipe sets its model,
connector, default `LAYOUT` and `vllm serve` flags, and the launcher handles
placement across GPUs and pods. Start a new one from an existing recipe.
