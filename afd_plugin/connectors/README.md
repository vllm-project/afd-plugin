# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

Both backends target vLLM `0.32.0`; Ascend additionally targets
vLLM-Ascend `bd42eea0`. See the
[runtime matrix](../../docs/design/module/execution_platforms.md#tested-runtime-matrix)
for tested configurations and backend-specific validation limits.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
