# Mega source attribution

The `afd_async_atten_send`, `afd_async_atten_recv`, and
`afd_async_mega_ffn` operators adapt the MegaMoE operators in CANN
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
  `mc2/megamoe_dispatch_send`, `mc2/megamoe_combine_recv`, and related MegaMoE sources.
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
`src/cam/comm_operator/pybind/umdk_cam_megamoe/`, with corresponding development
sources under `mc2/common/torch_extension/` in ops-transformer.
The layout and runtime wrappers were added during split operator development
in ops-transformer commit `97c9a376d43c12fc41d461296b0c53c0fd70ce69` as
`megamoe_split_layout.py` and `megamoe_split.py`. They were subsequently migrated
through UMDK into AFD as `mega_layout.py` and `mega_runtime.py`. These newly
written Python wrappers use the AFD project's Apache-2.0 headers; their original
development location does not make them official CANN Python API sources.

`tests/unit/connectors/npu/test_mega_sequence_ring.py` is a development test
migrated from `umdk/test/cam/megamoe/test_sequence_ring.py`. Its Python test
driver also uses Apache-2.0. The included native `mega_sequence_ring.h` retains
its own existing license notice.

`afd_plugin/compat/npu/ops.py` loads `afd_plugin._C_afd_async_mega` and registers the
three `afd_ascend` operator schemas, Meta implementations and native bindings.
It does not manage windows, sequences, tickets or publication polling.

`afd_plugin/connectors/npu/async_mega/mega_runtime.py` owns window lifecycle,
configuration construction, sequence validation, dispatch tickets and PUBLISHED
reads. `mega_layout.py` owns the Python window geometry. The connector uses
these modules directly. Window access uses the existing native
`CommContextManager.get_local_buffer_tensor()` binding; the native extension
does not currently expose kernel layout calculation. The reader preserves its
dedicated DMA copy stream and synchronizes only that stream.

AFD adaptations include private OPP loading, the `afd_ascend` namespace,
argument validation and draining HCCL warmup before channel creation.
The redundant `common/comm_context.py` re-export and `ops.py` aliases are removed.

The migrated schema/Meta adapter block in `compat/npu/ops.py` retains its
existing CANN notice and refers to [`LICENSE-CANN`](LICENSE-CANN).
No separate connector-local license copy is kept. CANN-derived native operator
sources retain their original notices and remain covered by that license.
