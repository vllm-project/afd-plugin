# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations.
    - `P2pNcclAFDConnector` is implemented by `afd_plugin.connectors.gpu.p2p`.
    - `DeepEPM2NAFDConnector` is implemented by the
      `afd_plugin.connectors.gpu.deepep_m2n` package:
        - `connector.py`: the connector, its `connector_extra_config` schema,
          the startup handshake, the per-stage plan, and its control plane.
        - `layout.py`: rank ordering and expert ownership over the AFD world.
        - `network.py`: NVLink-domain discovery and the GIN/hybrid network
          plan.
        - `experts.py`: FFN expert execution with the `deepgemm` or
          `fused_experts` backend.
        - `deepgemm.py`: the DeepGEMM grouped GEMMs, imported from the
          installed `deep_gemm` package.
        - `fp8_quant.py`: the fused SiLU, top-k weight and FP8 block-128
          requantization step between the two FP8 expert GEMMs.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

Both backends target vLLM `0.32.0`; Ascend additionally targets
vLLM-Ascend `bd42eea0`. See the
[runtime matrix](../../docs/design/module/execution_platforms.md#tested-runtime-matrix)
for tested configurations and backend-specific validation limits.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`:

- `base.py`: `AFDConnectorBase`, `AFDControlPlane`, and
  `AFDRoutedExpertsConnector`, the abstract subclass for connectors whose FFN
  receives routed (token, local expert) rows and runs only its local experts.
- `metadata.py`: payloads, transfer metadata and context, the control-plane
  codec, and the routed-experts weight types `LocalExpertWeights` and
  `FFNLocalExperts`.
- `factory.py`: connector registration and construction.

User guides:
[NCCL P2P](../../docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md),
[DeepEP M2N](../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md),
[CAM P2P](../../docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md), and
[CAM async](../../docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md). Contracts are
described in
[connector contracts](../../docs/design/module/connector_contracts.md).
