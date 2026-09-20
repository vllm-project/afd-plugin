# Ascend NPU Custom Ops Build

AFD CAMP2P depends on plugin-owned Ascend custom operators for
Attention-to-Expert and Expert-to-Attention transfers:

- `torch.ops.afd_ascend.a2e`
- `torch.ops.afd_ascend.e2a`

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
  utils/op_kernel/         # comm_args.h, data_copy.h, moe_distribute_base.h
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
  `a2e;e2a` for Ascend 910C. `950`, `ascend950*`, and `Ascend950*` build
  the same operators for Atlas A5 (`ascend950`).
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

This build path covers the AFD A2E/E2A operators required by the first CAMP2P
connector path. Quantized, ACL graph, and gate-on-attention paths are handled
separately.
