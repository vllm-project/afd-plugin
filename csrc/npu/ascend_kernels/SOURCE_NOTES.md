# MegaMoE source attribution

The `cam_megamoe_dispatch_send`, `cam_megamoe_combine_recv`, and
`cam_megamoe_fused_moe` operators adapt the MegaMoE operators in CANN
`ops-transformer/mc2`. Their original CANN copyright and license notices
remain in place. The appended blocks use the UMDK-style Huawei copyright
notice and record the split operator adaptations for AFD integration.

Header descriptions and history fields follow the compact format used by
UMDK `src/cam/comm_operator/ascend_kernels/fused_deep_moe/op_kernel/fused_deep_moe.cpp`.
The UMDK MIT notice is not applied to CANN source. The 2026-11-17 history
entry records the adaptation history, not original creation.
Existing original creation dates are preserved.

Vendored dependencies under `megamoe_utils` retain their original license
and copyright notices; no AFD ownership is asserted over unchanged copies.
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

The CANN OSL v2 text is retained in [`megamoe_utils/LICENSE`](megamoe_utils/LICENSE),
with additional dependency license files under `megamoe_utils/opbase` and
`megamoe_utils/ops-tensor`. The repository root license is Apache; therefore
the original source headers' reference to a root `LICENSE` does not resolve
to the CANN license in this repository. The appended header identifies CANN
OSL v2; the link above locates its retained text. No license text has been rewritten.
