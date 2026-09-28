# afd-plugin

<p align="center">
| <a href="docs/design/module/index.md"><b>Documentation</b></a> | <a href="recipe/README.md"><b>Recipe</b></a> | <a href="https://deepwiki.com/vllm-project/afd-plugin"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://vllm-dev.slack.com/archives/C0B4C1D84GG"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.jpg"><b>WeChat</b></a> |
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

The supported runtime baselines are **vLLM `v0.23.0` and `v0.26.0`**. Features
remain version scoped: the DeepSeek-V4 Atlas A5 HCCL P2P recipe uses v0.23,
while the DeepSeek-V3.2 Atlas A3 CAM recipes use v0.26. The plugin does not
modify the vLLM source tree. AFD behavior is installed through the
`vllm.general_plugins` entry point, `--additional-config`, automatically
selected role workers, plugin-owned model wrappers, and narrow version-scoped
compatibility shims.

## Architecture

![afd-plugin architecture](docs/assets/vllm-afd-plugin-architecture.svg)

## Current Status

Core runtime support:

- vLLM plugin registration, AFD configuration, and runtime validation.
- Attention/FFN workers, model runners, model wrappers, and connector-driven
  execution for CUDA and Ascend NPU.
- Eager and `FULL_DECODE_ONLY` graph execution, plus backend-specific profiling
  support.
- Native DBO with exactly two ubatches on CUDA and the supported Ascend paths.
- DeepSeek MoE handoff at the remote-experts boundary on CUDA and DeepSeek-V4
  Ascend NPU, with backend-specific gate placement.

Model support:

| Model family | Registered architectures | Plugin model wrappers | Notes |
| --- | --- | --- | --- |
| DeepSeekV2 / DeepSeekV3 / DeepSeekV3.2 | `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM` | `AFDDeepseekForCausalLM`, `AFDDeepseekV2ForCausalLM`, `AFDDeepseekV3ForCausalLM` | DeepSeekV3.2 uses `AFDDeepseekV3ForCausalLM`. Each AFD role constructs and loads only its role-required model components, while shared embedding, normalization, and output components remain available where required by the model lifecycle. |
| DeepSeekV4 | `DeepseekV4ForCausalLM` | `AFDDeepseekV4ForCausalLM` | The pinned v0.23 Ascend path supports role-selective model loading, synchronous HCCL P2P and Window communication, eager and `FULL_DECODE_ONLY` Graph execution, U2, and MTP/DSpark integration within the validator's feature boundaries. The public A5 A4F2 recipe uses HCCL P2P with MTP and DSpark disabled; Window AFD remains a separately documented code path. |

Connector support:

See the [recipe index](recipe/README.md) for deployment and benchmark examples.

| Connector | Platform | Recommend Stage | Sync or Async | Graph Support | Notes |
| --- | --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | CUDA | Decode | Sync | `FULL_DECODE_ONLY` CUDA graph | FFN ranks are ordered before Attention ranks. `num_attention_ranks` must be greater than or equal to `num_ffn_ranks` and divisible by it. See the [DeepSeek V2 Lite recipe](recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md). |
| `P2pHcclAFDConnector` | Ascend NPU | Decode | Sync | `FULL_DECODE_ONLY` ACL graph | Uses standard PyTorch distributed HCCL send/receive and supports integer-multiple A/F ratios in either direction. The [A5 DeepSeek-V4 A4F2 recipe](recipe/npu/P2pHcclAFDConnector/deepseek_v4/README.md) is pinned to v0.23. See the [connector guide](docs/npu/HCCL_P2P_CONNECTOR_USER_GUIDE.md). |
| `WindowAFDConnector` | Ascend NPU | Decode | Lock-step or async DP | `FULL_DECODE_ONLY` ACL graph | Uses HCCL CommContext Window operators, Attention-side routing, and a dedicated shared-expert FFN rank. The implementation supports one or two Window slots; async DP allows Attention sessions to advance independently. See the [code-level connector guide](docs/npu/WINDOW_AFD_CONNECTOR_USER_GUIDE.md). |
| `CAMP2pAFDConnector` | Ascend NPU | Decode | Sync | `FULL_DECODE_ONLY` ACL graph | Uses HCCL/CAMP2P custom ops. Ascend ops build by default on NPU platforms. See the [synchronous DeepSeek V3.2 recipe](recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md). |
| `CAMAsyncAFDConnector` | Ascend NPU | Prefill | Async | Not supported | Validated on v0.26 without PCP or Dual Batch. The checked-in [PCP8 recipe](recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md) records the earlier v0.19.1rc1 experiment and must be used with the `release/v0.19.1rc1` branch. |

Connector implementations are grouped by backend package:
`afd_plugin.connectors.gpu` for GPU-only connectors,
`afd_plugin.connectors.npu` for NPU-only connectors.

Known gaps:

- vLLM versions other than the exact `0.23.0` and `0.26.0` baselines are not
  claimed as supported; individual connectors and models remain tied to their
  documented baseline.
- vLLM/vLLM-Ascend model runner v2 is not supported.
- GPU and NPU E2E tests are opt-in and require real hardware plus model weights.
- GPU CUDA graph support is limited to `FULL_DECODE_ONLY`.
- Native DBO is limited to exactly two ubatches and is not supported by
  `CAMAsyncAFDConnector`.
- `WindowAFDConnector` is code-supported on the pinned v0.23 DeepSeek-V4 path,
  but this repository does not yet publish a hardware-qualified launch recipe
  for it.
- PCP-based NPU model-runner-v1 deployments from v0.19.1rc1 are not supported
  on v0.26.

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

The optional extra pins `vllm==0.26.0`.

### Ascend NPU installation

AFD's Ascend paths have separate version-scoped baselines. Install a compatible
driver and firmware, confirm the devices with `npu-smi info`, and select the
profile required by the recipe:

| Profile | Platform | vLLM | vLLM-Ascend | Connector/model |
| --- | --- | --- | --- | --- |
| A3 CAM | openEuler 22.03 aarch64, Ascend 910C / Atlas A3 | `0.26.0` | [`80d8c194f`](https://github.com/vllm-project/vllm-ascend/commit/80d8c194f7584b17fe08065ea99a130916f6b0e7) | CAM, DeepSeek-V3.2 |
| A5 DeepSeek-V4 | Single-host eight-device Atlas A5 | `0.23.0` (`0fc695fc`) | `rfc/vllm_cann` (`3da28f941`) | HCCL P2P A4F2 recipe; Window AFD code path |

Use Python 3.10 or 3.11 and the mutually compatible CANN, HCCL, torch, and
torch-npu versions recorded by the selected runtime or image manifest.

#### Environment

The v0.26 integration was refreshed against vLLM-Ascend commit `80d8c194f`;
the repository does not currently claim a released v0.26 container tag. Use the
[installation guide at that source snapshot](https://github.com/vllm-project/vllm-ascend/blob/80d8c194f7584b17fe08065ea99a130916f6b0e7/docs/source/installation.md)
to prepare a matching A3/openEuler environment, then install AFD from the
repository root. Do not reuse the former v0.19.1rc1 image as a v0.26 runtime.

The v0.23 A5 path is distributed as a pinned image because its CANN and HCCL
packages must remain an auditable unit. Follow the
[DeepSeek-V4 HCCL P2P recipe](recipe/npu/P2pHcclAFDConnector/deepseek_v4/README.md),
record the final image digest and package versions, and do not mix a host CANN
toolkit into the container. The HCCL P2P connector uses standard distributed
send/receive and does not require the CAMP2P custom operators.

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

After `AFD_OPS_OK`, the environment is ready to run the CAM NPU examples and
E2E tests. See the
[synchronous CAM recipe](recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md).
For the v0.23 A5 path, use the separately pinned
[HCCL P2P recipe](recipe/npu/P2pHcclAFDConnector/deepseek_v4/README.md).
The experimental Window path has separate runtime and checkpoint constraints;
read the [Window connector guide](docs/npu/WINDOW_AFD_CONNECTOR_USER_GUIDE.md)
before constructing a launch configuration.
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

GPU model runner v2 is not supported. Select model runner v1 before starting
either GPU role:

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

Launch Attention and FFN back-to-back because either side can wait during AFD
rendezvous. The A5 recipe starts FFN first and Attention immediately afterward.
Send requests only to the Attention API server. FFN workers are
connector-driven; scheduler-driven FFN `execute_model()` calls fail fast.

For repeatable local smoke testing, prefer the bundled runner:

```bash
uv run python tests/e2e/runner.py \
  --model /path/to/DeepSeek-V2-Lite \
  --device-backend gpu \
  --num-attention-ranks 1 \
  --num-ffn-ranks 1 \
  --attention-gpus 0 \
  --ffn-gpus 1 \
  --api-port-base 18000 \
  --afd-port 6239 \
  --common-vllm-arg=--trust-remote-code
```

For NPU, use `--device-backend npu`; the generic runner maps the same device
arguments to `ASCEND_RT_VISIBLE_DEVICES` and selects `CAMP2pAFDConnector`. Use
the dedicated DeepSeek-V4 recipe for `P2pHcclAFDConnector` and A5 A4F2.

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
`P2pHcclAFDConnector`, `WindowAFDConnector`, `CAMP2pAFDConnector`, or
`CAMAsyncAFDConnector`. AFD is active when
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
