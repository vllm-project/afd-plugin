# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `P2pHcclAFDConnector` is
  implemented by `afd_plugin.connectors.npu.p2p_hccl`,
  `WindowAFDConnector` by `afd_plugin.connectors.npu.window`,
  `CAMP2pAFDConnector` by `afd_plugin.connectors.npu.camp2p`, and
  `CAMAsyncAFDConnector` by `afd_plugin.connectors.npu.async_cam`.

The vLLM 0.26 support matrix validates GPU `P2pNcclAFDConnector` and NPU CAM
connectors. CAM async is validated without PCP on v0.26; its PCP8 recipe is
retained for v0.19.1rc1 and must be used with the `release/v0.19.1rc1` branch.
The pinned vLLM 0.23 DeepSeek-V4 path uses `P2pHcclAFDConnector`; see the
[HCCL P2P connector guide](../../docs/npu/HCCL_P2P_CONNECTOR_USER_GUIDE.md) and
[Atlas A5 A4F2 recipe](../../recipe/npu/P2pHcclAFDConnector/deepseek_v4/README.md).
The same v0.23 DeepSeek-V4 source line also contains an experimental
`WindowAFDConnector`; its M2N topology, Attention-side routing, Window slots,
and implementation limits are described in the
[Window connector guide](../../docs/npu/WINDOW_AFD_CONNECTOR_USER_GUIDE.md).
No hardware-qualified Window launch recipe is published here yet.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
