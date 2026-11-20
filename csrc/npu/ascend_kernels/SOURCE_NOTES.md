# Mega source attribution

The `afd_async_atten_send`, `afd_async_atten_recv`, and
`afd_async_mega_ffn` operators adapt the Mega operators in CANN
`ops-transformer/mc2`. Their original CANN copyright and license notices
remain in place. The appended blocks use the UMDK-style Huawei copyright
notice and record the split operator adaptations for AFD integration.

Header descriptions and history fields follow the compact format used by
UMDK `src/cam/comm_operator/ascend_kernels/fused_deep_moe/op_kernel/fused_deep_moe.cpp`.
The UMDK MIT notice is not applied to CANN source. The 2026-11-17 history
entry records the adaptation history, not original creation.
Existing original creation dates are preserved.

`mega_utils` contains only the shared layer-selection, sequence-limit,
sequence-ring and host-compatibility headers. Their existing copyright and
license notices remain unchanged. MC2 common, OpBase, Blaze and Tensor API
headers are consumed from the installed CANN SDK rather than vendored here.
The local reference checkouts used for this review are `ops-transformer`
and `umdk` alongside this repository.

## Reference revisions

- CANN operator reference: [ops-transformer local upstream](https://gitcode.com/lqy16/ops-transformer),
  revision `b7dd23b2999cc5ea052ab92a033ba640395b6ee4`, paths
  `mc2/atten_send`, `mc2/atten_recv`, and related Mega sources.
- Header format reference: [UMDK fused_deep_moe.cpp](https://gitcode.com/openeuler/umdk/blob/master/src/cam/comm_operator/ascend_kernels/fused_deep_moe/op_kernel/fused_deep_moe.cpp).
  The local template read was
  `src/cam/comm_operator/ascend_kernels/fused_deep_moe_w4a8/op_kernel/fused_deep_moe.cpp`
  (the local checkout uses this renamed directory). Local UMDK checkout revision: `ab039f0a384854a4d5f504b9c3425bc232186735`.

## License location

The CANN OSL v2 text is retained in [`LICENSE-CANN`](LICENSE-CANN).
The repository root license is Apache; therefore
the original source headers' reference to a root `LICENSE` does not resolve
to the CANN license in this repository. The appended header identifies CANN
OSL v2; the link above locates its retained text. No license text has been rewritten.

## Python interface and connector protocol

The split operator Python code came from `lqy16/umdk` at
`ab039f0a384854a4d5f504b9c3425bc232186735`, under
`src/cam/comm_operator/pybind/umdk_afd_async/`, with corresponding development
sources under `mc2/common/torch_extension/` in ops-transformer.
These are split operator development sources, not an official CANN Python API.
Existing copyright and license notices are preserved.

`afd_plugin/compat/npu/ops.py` loads `afd_plugin._C_afd_async_mega` and registers the
three `afd_ascend` operator schemas, Meta implementations and native bindings.
It does not manage windows, sequences, tickets or publication polling.

`afd_plugin/connectors/npu/mega_runtime.py` owns window lifecycle,
configuration construction, sequence validation, dispatch tickets and PUBLISHED
reads. `mega_layout.py` owns the Python window geometry. The connector uses
these modules directly. Window access uses the existing native
`CommContextManager.get_local_buffer_tensor()` binding; the native extension
does not currently expose kernel layout calculation. The reader preserves its
dedicated DMA copy stream and synchronizes only that stream.

AFD adaptations include private OPP loading, the `afd_ascend` namespace,
argument validation and draining HCCL warmup before channel creation.
The redundant `common/comm_context.py` re-export and `ops.py` aliases are removed.

The Python source license is retained at `afd_plugin/connectors/npu/LICENSE-CANN`;
it also covers the migrated schema/Meta adapter block in `compat/npu/ops.py`.
