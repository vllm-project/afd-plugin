# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

The vLLM 0.30 integration has GPU hardware evidence for
`P2pNcclAFDConnector`. On vLLM-Ascend `8d4409d6`, NPU
`CAMP2pAFDConnector` and `CAMAsyncAFDConnector` have representative V2-Lite
and DSV4 Flash W4A8 functional and 300-question accuracy evidence; see the
[runtime matrix](../../docs/design/module/execution_platforms.md#tested-runtime-matrix).
Historical DeepSeek-V3.2 CAM async evidence uses v0.26 without PCP. Its PCP8
recipe is retained for v0.19.1rc1 and must be used with the
`release/v0.19.1rc1` branch.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
