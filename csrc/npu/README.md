# Ascend NPU Custom Ops Build

AFD CAMP2P depends on plugin-owned Ascend custom operators for
Attention-to-Expert and Expert-to-Attention transfers:

- `torch.ops.afd_ascend.a2e`
- `torch.ops.afd_ascend.e2a`

The 910C build also contains four experimental routed-only CAM operators:
`afd_async_dispatch_send`, `afd_async_dispatch_recv`, `afd_async_combine_send`,
and `afd_async_combine_recv`, under the same `torch.ops.afd_ascend` namespace.
Their compact protocol and future connector migration are documented in
[the routed-only operator guide](../../docs/npu/CAM_ASYNC_ROUTED_OPS.md).

The NPU native sources live in this directory:

```text
csrc/npu/ascend_kernels   # ACLNN operator run package (npu_op_* build system)
csrc/npu/pybind           # PyTorch extension (_C_ascend)
csrc/npu/scripts          # build drivers
```

`ascend_kernels/` is organised by operator, with shared headers deduplicated
into `utils/`:

```text
ascend_kernels/
  CMakeLists.txt           # npu_op_package(... TYPE RUN) driver
  AddCustom.json           # msopgen input template
  operator_registry.json   # SOC generation -> operators, plus per-op metadata
  cmake_files/             # cmake/, op_host/, op_kernel/ build fragments
  a2e/{op_api,op_host,op_kernel}
  e2a/{op_api,op_host,op_kernel}
  afd_async_*/{op_api,op_host,op_kernel}  # four routed-only ops
  grouped_matmul_swiglu_quant_v2_layered/{op_api,op_host,op_kernel}
  grouped_matmul_layered/{op_api,op_host,op_kernel}
  utils/op_host/           # CAM host logging/check/tiling-base headers
  utils/op_kernel/         # comm_args.h, data_copy.h, moe_distribute_base.h, layered/
```

The supported SOC generations and the operators compiled for each are declared
in `ascend_kernels/operator_registry.json`; `scripts/select_ops.py` resolves the
operator list from that registry for a given SOC.

## Default Behavior

Ascend custom ops build by default. Set `AFD_BUILD_ASCEND_OPS=0` only when
intentionally skipping the NPU extension, for example on local CPU or macOS
development machines. The Python package remains import-safe when the extension
is not built.

## Build In An Ascend Environment

Install the package with Ascend custom ops enabled. Use
`--no-build-isolation` so pip builds against the current CANN/torch-npu
environment; pip's isolated build environment may not include CANN-provided
Python modules or packages such as `numpy`, `cmake`, and `tbe`.

```bash
SOC_VERSION=910c \
pip install -e . -v --no-build-isolation
```

Common environment variables:

- `ASCEND_HOME_PATH`: CANN toolkit path. Defaults to
  `/usr/local/Ascend/ascend-toolkit/latest`.
- `TORCH_NPU_PATH`: optional path to the `torch_npu` package.
- `SOC_VERSION`: `910c`, `ascend910_93*`, and `ascend910_9392` build
  A2E/E2A plus the four routed-only CAM operators for Ascend 910C.
  `950`, `ascend950*`, and `Ascend950*` build only A2E/E2A for Atlas A5
  (`ascend950`); the new CAM PyTorch registrations are also excluded.
- `MAX_JOBS`: number of parallel CMake build jobs for the PyTorch extension.
- `AFD_BUILD_JOBS`: number of parallel CMake build jobs for the ACLNN operator
  project. Defaults to `8`.
- `CMAKE_BUILD_TYPE`: ACLNN operator build type, `Release` (default) or `Debug`.
- `AFD_SKIP_ACLNN_BUILD=1`: skip rebuilding the ACLNN operator package and
  build the PyTorch extension against an existing custom-op installation.

The setup flow calls:

```text
csrc/npu/build_aclnn.sh                 # entry point, selects the SOC argument
csrc/npu/scripts/compile_ascend_proj.sh # msopgen + cmake for one SOC generation
```

`compile_ascend_proj.sh` resolves the operator list through
`scripts/select_ops.py`, generates the project with `msopgen`, patches the
generated `CMakePresets.json` through `scripts/set_conf.py`, then emits a
deterministically named run package at `csrc/npu/output/AFD_<soc>.run`.

Staging copies each selected operator's `op_host/`, `op_kernel/`, and `op_api/`
trees recursively, so operators that keep sources in nested directories (for
example `grouped_matmul_swiglu_quant_v2_layered`, whose tiling implementation
lives in `op_host/op_tiling/`) keep that layout in the generated project.
`cmake_files/op_host/CMakeLists.txt` therefore collects host sources recursively
and hands them to both `npu_op_code_gen` and `cust_optiling`; a top-level-only
collection would drop a nested operator's tiling registration from the run
package without failing the registry or Meta tests. After an Ascend build, that
registration can be re-checked in the packaged library, for example with
`strings .../vendors/afd-plugin/.../libcust_optiling.so | grep -i <OpName>`.

The generated artifacts are installed into the Python package:

```text
afd_plugin/_C_ascend*.so
afd_plugin/_cann_ops_custom/
```

`csrc/npu/output/` and `csrc/npu/build/` are build outputs and are not tracked.

## vLLM-Ascend Coexistence

AFD custom ops must coexist with vLLM-Ascend in the same process:

- The Python extension is owned by this plugin package, for example
  `afd_plugin._C_ascend`.
- A2E/E2A are registered under `torch.ops.afd_ascend`, not under
  vLLM-Ascend's `torch.ops._C_ascend` namespace.
- The CANN custom-op package is installed under the AFD vendor path,
  `afd_plugin/_cann_ops_custom/vendors/afd-plugin/...`.
- The loader uses the package-local `libcust_opapi.so` path through
  `AFD_CUST_OPAPI_LIB_PATH`; it must not rely on a bare
  `dlopen("libcust_opapi.so")`.

Load the extension at runtime with:

```python
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
```

## Current Scope

A2E/E2A continue to serve the existing CAMP2P connector. The four 910C
routed-only CAM operators are an experimental native integration and are not
yet selected by `CAMAsyncAFDConnector`, which retains its external legacy CAM
interface.

`grouped_matmul_swiglu_quant_v2_layered` is a fused
grouped-matmul + SiLU + dynamic-quantization CAM operator (A4W4 / A8W4-MSD
variants) built for `ascend910_93`. It is registered as
`torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered` (inference + Meta; no
autograd grad) and is excluded from `ascend950` builds. Nothing selects it at
runtime yet; switching a connector/model path to it is a separate follow-up.

For CANN 9.0.1, the layered ACLNN entry explicitly registers
`grouped_matmul_swiglu_quant_v2_layered.json` as a kernel configuration
candidate. The default runtime name conversion of `V2Layered` does not match
opbuild's filename (issue #372). No extra JSON aliases are installed.

No new NPU correctness, performance, or ACL graph support is claimed without
device validation. The new operators are inference-only and do not transfer
shared-expert payloads.

## Layered GMM device group list

`torch.ops.afd_ascend.grouped_matmul_layered` requires a device `group_list`
tensor. It must be a nonempty 1D `int64` tensor on the same
NPU as `x[0]`, with one entry per expert in each layer's weight tensor.
`group_list_type=1` means counts and passes the tensor directly to ACLNN.
`group_list_type=0` means cumulative offsets, which are converted to counts on
the NPU. It accepts one merged 2D activation in `x`, `split_item=3`, and a
one-element device `int64` `layer_index`. Output rows come from `x[0].shape[0]`;
the caller can supply a capacity-sized activation without a group-count read.
`per_token_scale` is optional at the binding level; A8W4 MSD requires a
capacity-sized device `float32` tensor. An absent `group_list` and other split
modes are not supported by this binding.
