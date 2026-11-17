# AFD documentation

## Deploy and use

1. [Install](installation.md) the pinned CUDA or Ascend environment.
2. Check [supported configurations and validation limits](support.md).
3. Choose a [deployment recipe](../recipe/README.md) and configure the
   [Attention/FFN roles](configuration.md).

| Backend | Connector guides |
| --- | --- |
| CUDA | [NCCL P2P](gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md), [DeepEP M2N](gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md) |
| Ascend NPU | [CAMP2P](npu/CAM_P2P_CONNECTOR_USER_GUIDE.md), [CAM async](npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md) |

## Develop

- [Module design and code ownership](design/module/index.md)
- [Contribution rules](../AGENTS.md) and [test maintenance](UT_MAINTENANCE.md)
- [Native operator builds](../csrc/README.md)
- NPU design: [Async FFN runtime](design/npu/async_ffn_runtime.md) and
  [routed CAM operators](design/npu/cam_async_routed_ops.md)

## Validate and troubleshoot

- [Model serving and accuracy E2E](../tests/e2e/README.md), including
  [multi-pod deployments](../tests/e2e/multi_pod/README.md)
- [GPU connector tests](../tests/gpu/README.md)
- [NPU runtime and operator tests](../tests/npu/README.md)
- [NPU troubleshooting](npu/TROUBLESHOOTING.md)
- Historical measurement: [Layered W13/W2 graph validation](reports/layered_w2_mdl_validation.md)
  (environment, source revisions, and results retained with the report)
