# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

The vLLM 0.28.0 support matrix hardware-validates GPU `P2pNcclAFDConnector`
(DeepSeek-V2-Lite 2A2F: eager, FULL_DECODE_ONLY CUDA Graph, and DBO on NVIDIA
L20X). The NPU connectors `CAMP2pAFDConnector` and `CAMAsyncAFDConnector` keep
their v0.26 baseline evidence, but the current release gates the plugin on
vLLM 0.28.0, which the v0.26 NPU runtime does not satisfy; NPU execution is
unsupported until the NPU upgrade lands. CAM async's PCP8 recipe is retained
for v0.19.1rc1 and must be used with the `release/v0.19.1rc1` branch.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
