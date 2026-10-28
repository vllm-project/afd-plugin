# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

H20 GPU `P2pNcclAFDConnector` results for selected vLLM 0.32
configurations used other revisions containing additional fixes; they do not
qualify this upgrade-only baseline. See the
[runtime matrix](../../docs/design/module/execution_platforms.md#tested-runtime-matrix)
for their scope. Earlier NPU baseline runs on vLLM-Ascend `8d4409d6`
have representative V2-Lite and DSV4 Flash W4A8 functional and
300-question accuracy evidence for `CAMP2pAFDConnector` and
`CAMAsyncAFDConnector`. Shared changes in this GPU upgrade were CPU-tested;
this branch was not retested on NPU hardware.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
