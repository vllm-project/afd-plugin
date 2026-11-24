# Ascend NPU Custom Ops Build

AFD CAMP2P depends on plugin-owned Ascend custom operators for
Attention-to-Expert and Expert-to-Attention transfers:

- `torch.ops.afd_ascend.a2e`
- `torch.ops.afd_ascend.e2a`

The 910C and 950 builds also contain four experimental routed-only CAM operators:
`afd_async_dispatch_send`, `afd_async_dispatch_recv`, `afd_async_combine_send`,
and `afd_async_combine_recv`, under the same `torch.ops.afd_ascend` namespace.
Their compact protocol is documented in
[the routed-only operator guide](../../docs/design/npu/cam_async_routed_ops.md).

The 950 implementations use the HCCL MC2 input windows and MTE transfers. They
share the compact `comm_args` / `group_name` schema, int32 expert IDs and four
dispatch-receive outputs with the 910C implementation. Callers select the chip
at build time and use the same unsuffixed torch, ACLNN and kernel names.

The 950 sources come from the default MTE line in
[GuangyuZhu04/afd-plugin at 4d134dc](https://github.com/GuangyuZhu04/afd-plugin/tree/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels),
an ancestor of the [0.32 donor](https://github.com/GuangyuZhu04/afd-plugin/commit/d43535c92fdfe26334c0abb54bfa6d33e9616e34).
Each 950 source directory registers its own SoC and reuses the common
compact ACLNN interface. The 2026-11-23 creation and history entries in the
new native files record their addition to this Ascend 950 port. The packaged
LICENSE includes the 2025 Huawei notice for the original operator sources.
The four vector-only kernel entries carry the donor's
[AIV-only task declaration](https://github.com/GuangyuZhu04/afd-plugin/commit/cb303bbbd18b055919f86b194f2b754fa4e794e5).
The plugin builds the compact transport without a CCU runtime or custom
session class. The compact kernels and binding retain the Huawei MIT notice
from the original import. The source distribution, wheel and native vendor
install carry the repository license and third-party notice.
[Original import](https://github.com/vllm-project/afd-plugin/pull/361)
· [LICENSE](../../LICENSE)

The native files added for the A5 port use the same MIT, Huawei 2026 and
AFD plugin contributor notices. This includes the six helpers below.
Their filenames and include paths use the `afd_async_` prefix in this
repository. The table links each original file at the commit that added it.

| Header under `ascend_kernels/utils/op_kernel/` | Original file and introducing commit |
| --- | --- |
| `afd_async_combine_tile_plan.h` | [`cam_async_combine_tile_plan.h` at `4d134dc9`](https://github.com/GuangyuZhu04/afd-plugin/blob/4d134dc955c58727c08511c6043ee01279eb6279/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_combine_tile_plan.h) |
| `afd_async_dispatch_pipeline.h` | [`cam_async_dispatch_pipeline.h` at `5574c9e0`](https://github.com/GuangyuZhu04/afd-plugin/blob/5574c9e0fd76a1e952e44359740b2dd5ce9982c8/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_dispatch_pipeline.h) |
| `afd_async_receive_cache.h` | [`cam_async_receive_cache.h` at `52504ac3`](https://github.com/GuangyuZhu04/afd-plugin/blob/52504ac3ce1eace526bd1f5148f5045fe7efaa25/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_cache.h) |
| `afd_async_receive_runs.h` | [`cam_async_receive_runs.h` at `fbb268be`](https://github.com/GuangyuZhu04/afd-plugin/blob/fbb268be5362827bc69f984685f595eae3f5b53e/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_receive_runs.h) |
| `afd_async_sparse_ranks.h` | [`cam_async_sparse_ranks.h` at `3a265b45`](https://github.com/GuangyuZhu04/afd-plugin/blob/3a265b45aa0b21b9d33038ec30d5a220668456cb/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_sparse_ranks.h) |
| `afd_async_window.h` | [`cam_async_window.h` at `7d773dcb`](https://github.com/GuangyuZhu04/afd-plugin/blob/7d773dcb5f1e73987d80b61f1191da2069057a34/csrc/npu/ascend_kernels/utils/op_kernel/cam_async_window.h) |

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
  afd_async_*/{op_api,op_host,op_kernel}  # common ACLNN interfaces and 910C implementations
  afd_async_*_arch35/{op_host,op_kernel}  # 950 compact MTE implementations
  grouped_matmul_swiglu_quant_v2_layered/{op_api,op_host,op_kernel}
  grouped_matmul_layered/{op_api,op_host,op_kernel}
  utils/op_host/           # CAM host logging/check/tiling-base headers
  utils/op_kernel/         # comm_args.h, data_copy.h, moe_distribute_base.h, layered/
```

The supported SOC generations and the operators compiled for each are declared
in `ascend_kernels/operator_registry.json`; `scripts/select_ops.py` resolves the
operator list from that registry for a given SOC.

## Pull Mega MC2 dependency

Set `connector_extra_config.ffn_mega_moe=true` on every Attention and FFN
rank to enable MegaMoE and its dedicated pull protocol. With the switch off,
CAM retains its original push Dispatch/Combine and FFN execution path.
Pull is not a standalone transport option: there is no `dispatch_mode`
setting or separate pull DispatchRecv/CombineSend operator. MegaMoE requires
`dynamicQuant=1` and per-channel W4A8 or W8A8 weights.
Per-group weights are not supported by MegaMoE.

The A3 `AfdMegaMoePullNative` kernel directly includes the installed CANN MC2
W4A8 `BlockMmad`. The build supports the CANN 9.1 MegaMoE header layout and
the CANN 9.2 layout with Catlass under `ascendc/common/catlass/include`. Set
`ASCEND_CANN_PACKAGE_PATH` or `ASCEND_HOME_PATH` to the CANN installation.
When this operator is selected, configuration fails if its MC2 headers are
missing; builds that exclude it do not need this dependency. The dependency
tree is not copied into this repository.

One native launch selects a layer and drains one source transaction. Receive
capacity follows ordinary CAM: `int(262144 * BATCH_SIZE_FACTOR)`, default 262144
rows. Only transactions exceeding this capacity use multiple device rounds;
capacity is independent of the MC2 matrix tile size. Callers must not launch
once per receive round. The workspace allocation ABI and Attention Combine
interface are unchanged. Both quantization plans use SwiGLU. W4A8 applies the
model `swiglu_limit` in both ACT paths as
`SiLU(min(gate, L)) * clamp(up, -L, L)`; zero, the default, disables clipping.
W8A8 does not apply `swiglu_limit`, matching the Layered W8A8 path.

For W4A8 intermediate widths up to 3072, ACT directly uses the installed MC2
W4A8 two-stage UB epilogue and expert-group synchronization. Wider supported
W4A8 rows and W8A8 use bounded-column activation. Return-address caching is
bounded independently of receive capacity. Large capacity reserves HBM scratch
without computing padded rows; for H4096/I2048 the default capacity needs about
12.5 GiB per FFN executor, excluding weights and communication windows. This
includes a 2 GiB buffer for BF16 output staging.

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
  `950`, `ascend950*`, and `Ascend950*` build A2E/E2A and the four compact
  MTE operators for Atlas A5.
- `MAX_JOBS`: number of parallel CMake build jobs for the PyTorch extension.
- `AFD_BUILD_JOBS`: number of parallel CMake build jobs for the ACLNN operator
  project. Defaults to `8`.
- `CMAKE_BUILD_TYPE`: ACLNN operator build type, `Release` (default) or `Debug`.
- `AFD_SKIP_ACLNN_BUILD=1`: skip rebuilding the ACLNN operator package and
  build the PyTorch extension against an existing custom-op installation.
  At async CAM startup, each worker checks the four compact operator schemas
  and resolves their eight ACLNN execution and workspace-query symbols before
  joining the CAM communication group. If the installed library is stale,
  the error identifies its path and missing symbols.

The setup flow calls:

```text
csrc/npu/build_aclnn.sh                 # entry point, selects the SOC argument
csrc/npu/scripts/compile_ascend_proj.sh # msopgen + cmake for one SOC generation
```

`compile_ascend_proj.sh` resolves the operator list through
`scripts/select_ops.py`, generates the project with `msopgen`, patches the
generated `CMakePresets.json` through `scripts/set_conf.py`, then emits a
deterministically named run package at `csrc/npu/output/AFD_<soc>.run`.

For 950, `select_ops.py` maps each public `afd_async_*` selection to its
`afd_async_*_arch35` source directory. The staging step uses the common
`afd_async_*/op_api/` directory when the variant shares its ACLNN interface.
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
- The loader opens the package-local API through `AFD_CUST_OPAPI_LIB_PATH`.
  The 950 wheel installs it as `libafd_cust_opapi.so` and appends the AFD vendor
  after existing vendors in `ASCEND_CUSTOM_OPP_PATH` during plugin registration.
  Native Ascend operators retain their own API lookup, including the
  DeepSeek-V4 sparse-attention metadata implementation. The 910C package
  and older installations use `libcust_opapi.so`.

Load the extension at runtime with:

```python
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
```

Prepare native and AFD `ASCEND_CUSTOM_OPP_PATH` vendor entries before launching
Python when using the layered operators. GE can initialize infer-shape
registrations while importing native modules; adding the AFD path afterward
cannot repair that process's missing registration. Preserve the CANN Python
and library paths from the activated environment. The
[standalone guide](../../docs/npu/STANDALONE_FFN_SERVICE.md#launch) shows the
paired vendor setup.

## Current Scope

A2E/E2A serve `CAMP2pAFDConnector`. `CAMAsyncAFDConnector` uses the plugin's
compact routed-only CAM operators; see the [operator contract](../../docs/design/npu/cam_async_routed_ops.md).
The standalone FFN entry point reuses this protocol and connector; see
[the standalone service guide](../../docs/npu/STANDALONE_FFN_SERVICE.md).

`grouped_matmul_swiglu_quant_v2_layered` is a fused
grouped-matmul + SiLU + dynamic-quantization CAM operator (A8W8 / A4W4 /
A8W4-MSD variants) built for `ascend910_93`. It is registered as
`torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered` (inference + Meta; no
autograd grad) and is excluded from `ascend950` builds. The default Async CAM
layered executor uses it for W4A8 and W8A8.

The trailing `swiglu_limit` argument defaults to `0.0` (no clipping). A finite
positive `L` computes `SiLU(min(gate, L)) * clamp(up, -L, L)` before output
quantization on A8W4 and A4W4. The gate has no lower clamp. Positive limits on
A8W8 are rejected. The layered W4A8 executor forwards the model's limit, including
DeepSeek-V4's `10.0`, in eager and graph execution. Rebuild both the native
extension and CANN package together when upgrading this interface.

A8W8 accepts an NZ weight as either an explicit five-dimensional view or the
three-dimensional logical view `npu_format_cast` produces over NZ storage; the
dimension and shape checks accept both, as the A8W4/A4W4 path already did.
`layer_index` is validated by element count at the ACLNN entry
(`GetViewShape().GetShapeSize() == 1`), not by the storage shape the tiling
context exposes - a rank-1 one-element `int64` tensor can arrive there as rank 2
`[1, 1]`.

Note both attributes reach the tiling only as data, while the tiling cache is
keyed on shape and tiling key: calls that share a shape share the first call's
tiling and its `group_list_type` / `dequant_mode` values. Vary the shape when a
process needs both values of either attribute.

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
