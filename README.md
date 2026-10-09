<p align="center">
  <img src="docs/assets/afd-plugin-logo.png" alt="AFD Plugin" width="500">
</p>

<p align="center">
| <a href="docs/design/module/index.md"><b>Documentation</b></a> | <a href="recipe/README.md"><b>Recipe</b></a> | <a href="https://docs.google.com/presentation/d/16DMlDLmwtgYCktXiayoM3W36YsoPURq15g50OJ5Iv-8/edit?usp=sharing"><b>Slides</b></a> | <a href="https://deepwiki.com/vllm-project/afd-plugin"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://vllm-dev.slack.com/archives/C0B4C1D84GG"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.png"><b>WeChat</b></a> |
</p>

## Overview

**afd-plugin** is a [vLLM](https://github.com/vllm-project/vllm)
external plugin for **Attention-FFN Disaggregation (AFD)**. It provides
plugin-owned worker classes, model runners, model wrappers, connectors,
configuration validation, compatibility shims, and hardware-gated integration
tests for GPU and Ascend NPU deployments.

> [!NOTE]
> This project is still experimental and needs more large-scale testing across
> different hardware backends.

The target runtime is **vLLM `v0.30.0`** at commit `ced6857a`; the Ascend
integration targets vLLM-Ascend `8d4409d6`. The plugin does not modify the vLLM
source tree. AFD behavior is installed through the `vllm.general_plugins` entry
point, `--additional-config`, automatically selected role workers, plugin-owned
model wrappers, and narrow version-scoped compatibility shims.

## Architecture

![afd-plugin architecture](docs/assets/vllm-afd-plugin-architecture.svg)

## Current Status

Core runtime support:

- vLLM plugin registration, AFD configuration, and runtime validation.
- Attention/FFN workers, model runners, model wrappers, and connector-driven
  execution for CUDA and Ascend NPU.
- Eager and `FULL_DECODE_ONLY` graph execution, plus backend-specific profiling
  support.
- Native DBO with exactly two ubatches on CUDA V1/V2 and the synchronous
  Ascend V1 path; Ascend V2 DBO remains unsupported.
- DeepSeek MoE handoff at the remote-experts boundary on CUDA, with the gate
  placed on either Attention or FFN.

Model support:

| Model family | Registered architectures | Plugin model wrappers | Notes |
| --- | --- | --- | --- |
| DeepSeekV2 / DeepSeekV3 / DeepSeekV3.2 | `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM` | `AFDDeepseekForCausalLM`, `AFDDeepseekV2ForCausalLM`, `AFDDeepseekV3ForCausalLM` | DeepSeekV3.2 uses `AFDDeepseekV3ForCausalLM`. Each AFD role constructs and loads only its role-required model components, while shared embedding, normalization, and output components remain available where required by the model lifecycle. |
| Qwen3 MoE | `Qwen3MoeForCausalLM` | `AFDQwen3MoeForCausalLM` | CUDA with `compute_gate_on_attention=false`. |
| Qwen3.5 / Qwen3.6 MoE | `Qwen3_5MoeForConditionalGeneration` | `AFDQwen3_5MoeForConditionalGeneration` | CUDA text-only Qwen3.6-35B-A3B with `--language-model-only`, native DP4/TP1/EP4, and synchronous AFD 2A1F eager/graph. DBO remains excluded by a known FFN CUDA fault; see the [E2E limits](docs/design/module/e2e_testing.md). |

Connector support:

See the [recipe index](recipe/README.md) for deployment and benchmark examples.

| Connector | Platform | Recommend Stage | Sync or Async | Graph Support | Notes |
| --- | --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | CUDA | Decode | Sync | `FULL_DECODE_ONLY` CUDA graph | FFN ranks are ordered before Attention ranks. `num_attention_ranks` must be greater than or equal to `num_ffn_ranks`; the Attention ranks are spread over the FFN ranks in blocks that differ in size by at most one. See the [DeepSeek V2 Lite recipe](recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md). |
| `CAMP2pAFDConnector` | Ascend NPU | Decode | Sync | V1 `FULL_DECODE_ONLY`; V2 also `FULL` | Uses HCCL/CAMP2P custom ops. Ascend ops build by default on NPU platforms. See the [synchronous DeepSeek V3.2 recipe](recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md). |
| `CAMAsyncAFDConnector` | Ascend NPU | Prefill / decode | Async | Not supported | Experimental DP+TP/SP path with AFD-managed two-stage MoE ubatching; native DBO and PCP are unsupported. V2-Lite and DSV4 Flash W4A8 have [representative v0.30 validation](https://github.com/vllm-project/afd-plugin/pull/425#issuecomment-6063910923). The [legacy PCP8 recipe](recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md) requires `release/v0.19.1rc1`. |

Connector implementations are grouped by backend package:
`afd_plugin.connectors.gpu` for GPU-only connectors,
`afd_plugin.connectors.npu` for NPU-only connectors.

Known gaps:

- The GPU target is exactly vLLM `0.30.0`; other vLLM versions are not
  claimed as supported on GPU.
- NPU hardware evidence covers representative DeepSeek-V2-Lite and DSV4
  Flash W4A8 deployments; see the [runtime matrix](docs/design/module/execution_platforms.md#tested-runtime-matrix).
  Other models and topologies remain unverified. Historical v0.26 results retain
  their recorded vLLM-Ascend `80d8c194f` environment.
- NPU MRV2 many-to-one padding and DSV2 Async SP with TP-sharded shared
  experts have known correctness defects deferred to separate fixes; see the
  [runtime matrix](docs/design/module/execution_platforms.md#tested-runtime-matrix).
- Synchronous DSV4 E2E profiles are smoke cases; A5 concurrent answers remain
  incorrect. See the [profile limits](tests/e2e/README.md#dsv4-flash-sync-camp2p-concurrent-requests-local-4-or-16-npus).
- GPU ModelRunnerV2 supports synchronous `P2pNcclAFDConnector` with
  `compute_gate_on_attention=false`, including two-microbatch DBO and
  `FULL_DECODE_ONLY`. H20 validation covers DeepSeek-V2-Lite, 2A2F DP2/TP1;
  see the [MRV2 DBO scenarios](tests/e2e/README.md#gpu-mrv2-dbo-development-comparison).
  Representative 300-question comparisons do not establish numerical
  equivalence; see the [runtime matrix](docs/design/module/execution_platforms.md#tested-runtime-matrix). NPU MRV2 DBO is unsupported.
- GPU and NPU E2E tests are opt-in and require real hardware plus model weights.
- GPU CUDA graph support is limited to `FULL_DECODE_ONLY`.
- Native DBO is limited to exactly two ubatches and is not supported by
  `CAMAsyncAFDConnector`.
- Qwen3 MoE currently rejects Attention-side gate placement, sequence-parallel
  MoE, EPLB, pipeline parallelism, speculative decoding, LoRA, and NPU.
- Qwen3.5/Qwen3.6 MoE adapter-family support is limited to the CUDA text-only
  lane; repository hardware E2E evidence currently covers Qwen3.6-35B-A3B.
  NPU and multimodal execution are unsupported/unverified;
  `compute_gate_on_attention=true`, pipeline parallelism, asynchronous and
  multi-node execution are unsupported; quantization is unverified; no
  performance claim is made.
- PCP-based NPU model-runner-v1 deployments from v0.19.1rc1 are not supported
  on v0.30.

## Install

Requires Python **3.10-3.13** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vllm-project/afd-plugin.git
cd afd-plugin
uv sync --group dev
```

`vllm` is an optional runtime extra so CPU-only or macOS development
environments can still run import/config tests without a CUDA wheel.

### GPU installation

For Linux / CUDA-capable environments only; Ascend users should skip this
command:

```bash
uv sync --group dev --extra vllm
```

The optional extra pins `vllm==0.30.0`.

### Ascend NPU installation

The Ascend target uses the source pair below. Representative v0.30 validation
uses Ascend 910C / Atlas A3; see the [recorded environment and results](https://github.com/vllm-project/afd-plugin/pull/425#issuecomment-6063910923).
Install a compatible driver and firmware, and confirm the devices with
`npu-smi info`.

| Component | Version |
| --- | --- |
| Python | `3.10` or `3.11` |
| vLLM | `0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | commit [`8d4409d6`](https://github.com/vllm-project/vllm-ascend/commit/8d4409d6256d8a6729140ddcc0d1889e3f96cdd6) |
| CANN / torch / torch-npu | Use the mutually compatible versions required by that vLLM-Ascend source snapshot. |

> Hardware evidence is scoped to the recorded models, configurations, and
> sample counts; it does not cover every NPU deployment or guarantee clean
> device shutdown.

#### Environment

The Ascend environment targets vLLM-Ascend commit `8d4409d6` and vLLM
`0.30.0`. The recorded A3 environment uses CANN 9.1.0, torch 2.10.0+cpu,
and torch-npu 2.10.0.post4. The repository does not claim a released container
tag for either target. Use the
[installation guide at that source snapshot](https://github.com/vllm-project/vllm-ascend/blob/8d4409d6256d8a6729140ddcc0d1889e3f96cdd6/docs/source/getting_started/installation.md)
to prepare a matching A3/openEuler environment, then install AFD from the
repository root. Do not reuse the former v0.19.1rc1 image as a v0.30 runtime.

#### Install AFD

From the AFD repository root:

```bash
AFD_BUILD_ASCEND_OPS=1 \
SOC_VERSION=ascend910_9391 \
python -m pip install -v --no-build-isolation --no-deps -e .
```

`--no-deps` preserves the matched NPU runtime, and `--no-build-isolation` uses
its CANN/torch-npu toolchain. The command above forces the Ascend op build for
reproducibility. Otherwise, leave `AFD_BUILD_ASCEND_OPS` unset to use the
auto-detection described in [Development](#development), set it to `1` if
detection misses an Ascend environment, or set it to `0` to skip the ops.

#### Verify

```bash
python - <<'PY'
import torch
import torch_npu
import vllm_ascend

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

assert torch.npu.is_available(), "torch-npu cannot see an Ascend device"
ensure_afd_ascend_ops_loaded()
print("AFD_OPS_OK")
PY
```

After `AFD_OPS_OK`, the environment is ready to run the NPU examples and E2E
tests. See the
[synchronous NPU recipe](recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md).
For implementation details, see the
[Attention runtime design](docs/design/module/attention_runtime.md) and
[FFN runtime design](docs/design/module/ffn_runtime.md).

## Using the Plugin

Install or sync the distribution as `vllm-afd-plugin`. Python imports use the
`afd_plugin` package name.

AFD is configured through vLLM `--additional-config`. There is no separate
`--afd-config` flag. When AFD is configured and `--worker-cls` is omitted, the
plugin automatically selects the Attention or FFN worker for the active CUDA
or standard Ascend NPU platform. Explicit AFD worker paths remain accepted for
compatibility with existing commands, but are not required or stable launch
interfaces.

The examples below use model runner v1. For GPU MRV2, set
`VLLM_USE_V2_MODEL_RUNNER=1` on both roles and use the
[dedicated E2E scenarios](tests/e2e/README.md#gpu-mrv2-dbo-development-comparison).
MRV2 DBO requires two microbatches and Attention DP > 1.
For the MRV1 examples below, select:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

GPU Attention-side shape:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

GPU FFN-side shape:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --served-model-name deepseek-v2-lite-afd-ffn \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18001 \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

NPU uses the same config channel with `CAMP2pAFDConnector`; the plugin selects
the NPU worker automatically:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"CAMP2pAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

Attention and FFN may be started in either order. Send requests only to the
Attention API server. FFN workers are connector-driven; scheduler-driven FFN
`execute_model()` calls fail fast.

For repeatable local smoke testing, prefer the bundled runner:

```bash
export HF_HOME=/path/to/huggingface
python -c 'from datasets import load_dataset; load_dataset("openai/gsm8k", "main")'

uv run python tests/e2e/runner.py \
  --model /path/to/DeepSeek-V2-Lite \
  --device-backend gpu \
  --scenario afd-eager-2a1f \
  --attention-devices 0,1 \
  --ffn-devices 2 \
  --gsm8k-output-path /tmp/afd-e2e-gsm8k \
  --api-port-base 18000 \
  --afd-port 6239 \
  --common-vllm-arg=--trust-remote-code
```

For NPU, use `--device-backend npu`; the runner maps the same device arguments
to `ASCEND_RT_VISIBLE_DEVICES` and selects `CAMP2pAFDConnector`.

## AFD Config

The canonical config shape is:

```json
{
  "afd": {
    "role": "attention",
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 1239,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 1,
    "compute_gate_on_attention": false,
    "connector_extra_config": {}
  }
}
```

`role` must be `attention` or `ffn`. `connector` must be `P2pNcclAFDConnector`,
`CAMP2pAFDConnector`, or `CAMAsyncAFDConnector`. AFD is active when
`additional_config["afd"]` is present and passes common AFD config validation;
omit `additional_config["afd"]` to disable AFD. Connector-owned
`connector_extra_config` is strictly validated by the selected connector parser
when the connector/runtime is constructed. The plugin also accepts selected
compatibility aliases such as `afd_role`, `afd_connector`, `afd_host`, and
`afd_port`.

## Development

Run the default CPU-safe checks:

```bash
uv run pytest
uv run ruff check .
```

The NPU-dependent unit suite must run in an Ascend development environment;
see [Running NPU unit tests](docs/npu/TESTING.md).

Native C/C++ sources are grouped by backend under `csrc/`: Ascend/CANN sources
live in `csrc/npu`, including the `a2e` and `e2a` ACLNN operators, and
`csrc/gpu` is reserved for GPU native sources.

Ascend custom ops are built automatically only when the build environment looks
like an Ascend NPU platform, for example when `torch_npu`, CANN environment
variables, or the default Ascend toolkit path are present. GPU builds skip
Ascend ops by default. Set `AFD_BUILD_ASCEND_OPS=1` or
`AFD_BUILD_ASCEND_OPS=0` to override the auto-detection.

## E2E Test

To run E2E tests, use the [`run-e2e` skill](.agents/skills/run-e2e/SKILL.md).

## License

afd-plugin is licensed under the [Apache License 2.0](LICENSE).

## Cite

If you find afd-plugin helpful in your research or projects, please
consider citing it:

```bibtex
@misc{afdplugin2026,
  title={afd-plugin: Attention-FFN Disaggregation for vLLM},
  author={AFD Plugin Contributors},
  year={2026},
  howpublished={\url{https://github.com/vllm-project/afd-plugin}},
}
```
