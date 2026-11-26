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

The optional GPU extra pins `vllm==0.34.0`.

`DeepEPM2NAFDConnector` also needs the `afdep` extra (afdEP-comm). Its
DeepGEMM experts use the DeepGEMM bundled in vLLM CUDA builds; see
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
| vLLM | `0.34.0`, commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| vLLM-Ascend | commit [`7dcdb690`](https://github.com/vllm-project/vllm-ascend/commit/7dcdb690caf879a17ab07382880d5773f5202e20) |
| Tested CANN / torch / torch-npu | `9.3.0` / `2.10.0+cpu` / `2.10.0.post7` |
| Tested Triton Ascend / CAM | `3.6.0.post2` / `1.3.0` |

See the [support reference](support.md#validation-limits) for hardware coverage
and known shutdown limitations.

### Environment

The recorded A3 environment uses the official Ubuntu 22.04 arm64 nightly
from [vLLM-Ascend build 45558867839](https://github.com/vllm-project/vllm-ascend/actions/runs/45558867839).
Pull the pinned manifest before installing AFD from the repository root.

```text
quay.io/ascend/vllm-ascend@sha256:daa1bc6a1d780b49678ba49fa7f296cdad3d8a16a1994784c02c4754a2d4d562
```

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

On the Ubuntu A3 image, expose the installed AFD vendor before launching a
CAMP2P service so CANN can register the A2E/E2A tiling functions on startup.
From the repository root used for the editable install, run the following
with the CANN environment already activated.

```bash
afd_vendor="$PWD/afd_plugin/_cann_ops_custom/vendors/afd-plugin"
export ASCEND_CUSTOM_OPP_PATH="$afd_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$afd_vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AFD_CUST_OPAPI_LIB_PATH="$afd_vendor/op_api/lib/libcust_opapi.so"
```

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
