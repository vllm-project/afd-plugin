# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

The vLLM 0.28.0 support matrix hardware-validates GPU `P2pNcclAFDConnector`
(DeepSeek-V2-Lite 2A2F: eager, FULL_DECODE_ONLY CUDA Graph, and DBO on NVIDIA
L20X). Ascend `CAMP2pAFDConnector` is hardware-tested on vLLM 0.28.0
with vLLM-Ascend `bd69bad88fc19e1aeeea585416d408df8bda8fef`:
DeepSeek-V2-Lite BF16, V1, TP1, 2A2F/2A1F eager, FULL_DECODE_ONLY
and DBO, GSM8K-7 only. Full accuracy is deferred by the requester.
See the [current recipe](../../recipe/npu/CAMP2pAFDConnector/deepseek_v2_lite/README.md).
Async CAM was excluded from this upgrade experiment; its v0.26 evidence is
historical. Its PCP8 recipe requires `release/v0.19.1rc1`.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
