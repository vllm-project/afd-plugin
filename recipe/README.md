# AFD Recipes

This directory contains deployment and benchmark recipes for AFD connectors.
Both GPU and Ascend target vLLM 0.34.0. See the
[installation guide](../docs/installation.md) for the runtime pins and the
[Ascend upgrade record](https://github.com/vllm-project/afd-plugin/pull/651)
for A3 validation. The V3.2 recipes below retain their original runtime pins.

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
| GPU (SM90+ for `DeepEPM2NAFDConnector`) | `DeepEPM2NAFDConnector` | Qwen3-30B-A3B | Prefill / decode; decode-only benchmark | 2A2F graph, eager, graph+DBO and eager Attention-TP2 recipes; multi-pod layouts remain configurable through `LAYOUT` | [Graph, eager, DBO](gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/README.md) |
| GPU (SM90+ for `DeepEPM2NAFDConnector`) | `DeepEPM2NAFDConnector` | Qwen3-235B-A22B-FP8 | Prefill / decode; decode-only benchmark | FP8-payload 2A2F eager, graph and DBO recipes, plus two-pod 4A1F DBO; H200-class FFN GPUs | [Graph, eager, DBO](gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/README.md) |
| GPU | `P2pNcclAFDConnector` | Qwen3-30B-A3B | Prefill / decode; decode-only benchmark | 2A2F graph recipe with the DeepEP M2N recipe's settings, for connector comparison | [Graph](gpu/P2pNcclAFDConnector/qwen3_30b_a3b/README.md) |
| GPU | `P2pNcclAFDConnector` | Qwen3-235B-A22B-FP8 | Prefill / decode; decode-only benchmark | 2A2F graph recipe with the DeepEP M2N recipe's settings, for connector comparison | [Graph](gpu/P2pNcclAFDConnector/qwen3_235b_a22b_fp8/README.md) |
| GPU | `P2pNcclAFDConnector` | Qwen3.5-122B-A10B-FP8 | Decode | Three-pod graph recipe (`2A0F,2A0F,0A2F`); other layouts remain configurable through `LAYOUT` | [Graph](gpu/P2pNcclAFDConnector/qwen3_5_122b_a10b_fp8/prefill_decode_colocation/multipod_2a_2a_2f_graph.sh) |
| Ascend NPU | `CAMP2pAFDConnector` | DeepSeek-V3.2 | Decode | Historical v0.19.1rc1 recipe; current-runtime qualification is pending | [Synchronous decode](npu/CAMP2pAFDConnector/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V3.2 | Prefill / decode | Historical v0.19.1rc1/v0.26 recipes; standard V3.2 is unsupported by the current Async CAM path | [Async CAM](npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V4-Flash W4A8/W8A8 DSpark | Separate P / decode, or colocated | vLLM 0.34.0 recipe with layered FFN eager/FULL. A3 upgrade evidence covers W4A8 PD with target and draft decode graphs. See the recipe for earlier W4A8/W8A8 evidence and its limits. | [DSpark A/F](npu/CAMAsyncAFDConnector/deepseek_v4_flash/README.md) |

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
