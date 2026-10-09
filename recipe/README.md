# AFD Recipes

This directory contains deployment and benchmark recipes for AFD connectors.
Read each recipe's support status before running it: some historical experiment
records are retained for provenance but are not supported by the current
vLLM 0.26 runtime.

## Directory layout

Recipes are organized by hardware backend, connector, and model:

```text
recipe/
├── gpu/
│   └── P2pNcclAFDConnector/
│       └── deepseek_v2_lite/
└── npu/
    ├── CAMAsyncAFDConnector/
    │   ├── deepseek_v3_2/
    │   └── deepseek_v4_flash/
    └── CAMP2pAFDConnector/
        └── deepseek_v3_2/
```

Directory names follow these conventions:

- Hardware backend: `gpu` or `npu`.
- Connector: the exact connector class name, such as `P2pNcclAFDConnector`,
  `CAMP2pAFDConnector`, or `CAMAsyncAFDConnector`.
- Model: the model family or variant in lowercase snake case, such as
  `deepseek_v2_lite` or `deepseek_v3_2`.

## Available recipes

| Hardware | Connector | Model | Recommended stage | v0.26 status | Recipe |
| --- | --- | --- | --- | --- | --- |
| GPU | `P2pNcclAFDConnector` | DeepSeek-V2-Lite | Decode | Validated | [Launch examples](gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md) |
| Ascend NPU | `CAMP2pAFDConnector` | DeepSeek-V3.2 | Decode | Validated | [Synchronous decode](npu/CAMP2pAFDConnector/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V3.2 | Prefill / decode | Experimental v0.26 DP+TP/SP path; post-fix DP2TP8+EP16 token split reached `0.9522` strict match on the complete GSM8K evaluation; legacy PCP8 results are v0.19-only | [Async CAM](npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md) |
| Ascend NPU | `CAMAsyncAFDConnector` | DeepSeek-V4-Flash W8A8 | Prefill / decode | Experimental v0.26 single-node DP4TP2 Attention + EP8 FFN tool stack for issue #227 performance validation | [Async CAM](npu/CAMAsyncAFDConnector/deepseek_v4_flash/README.md) |

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
