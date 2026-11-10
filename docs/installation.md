# Installation

Requires Python **3.11-3.13** and [uv](https://docs.astral.sh/uv/).

## Source setup

```bash
git clone https://github.com/vllm-project/afd-plugin.git
cd afd-plugin
uv sync --group dev
```

`vllm` is an optional runtime extra so CPU-only or macOS development
environments can still run import/config tests without a CUDA wheel.

## GPU installation

For Linux / CUDA-capable environments only; Ascend users should skip this
command:

```bash
uv sync --group dev --extra vllm
```

The optional GPU extra pins `vllm==0.32.0`.

`DeepEPM2NAFDConnector` also needs the `afdep` extra (afdEP-comm) and
DeepGEMM, which is built from source; see
[Installation](gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#installation) in
the connector guide.

```bash
uv sync --group dev --extra vllm --extra afdep
```

## Ascend NPU installation

Use the pinned source pair and tested stack below. Install a compatible driver
and firmware, then confirm device visibility with `npu-smi info`.

| Component | Version |
| --- | --- |
| Python | Package range `3.11–3.13`; tested `3.12.13` |
| vLLM | `0.32.0`, commit `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM-Ascend | commit [`bd42eea0`](https://github.com/vllm-project/vllm-ascend/commit/bd42eea01d8312c4df469fabaed45f77a12df4d3) |
| Tested CANN / torch / torch-npu | `9.2` / `2.10.0+cpu` / `2.10.0.post7` |
| Tested Triton Ascend / CAM | `3.6` / `1.3` |

See the [support reference](support.md#validation-limits) for hardware coverage
and known shutdown limitations.

### Environment

The recorded A3 environment uses the pinned pair and stack above. The
repository does not claim a released container tag for the Ascend target. Use the
[installation guide at that source snapshot](https://github.com/vllm-project/vllm-ascend/blob/bd42eea01d8312c4df469fabaed45f77a12df4d3/docs/source/getting_started/installation.md)
to prepare a matching A3/openEuler environment, then install AFD from the
repository root.

### Install AFD

From the AFD repository root:

```bash
AFD_BUILD_ASCEND_OPS=1 \
SOC_VERSION=ascend910_9391 \
python -m pip install -v --no-build-isolation --no-deps -e .
```

`--no-deps` preserves the matched runtime; `--no-build-isolation` uses its
CANN/torch-npu toolchain. `AFD_BUILD_ASCEND_OPS=1` forces the native build;
`0` skips it. When unset, setup detects the Ascend environment. See the
[native build guide](../csrc/npu/README.md) for build controls and artifacts.

### Verify

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

After `AFD_OPS_OK`, choose a [deployment recipe](../recipe/README.md).
Use the [NPU test guide](../tests/npu/README.md) to validate the runtime.
