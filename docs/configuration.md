# AFD configuration

Pass settings through vLLM `--additional-config` under the `afd` key. Omit
`afd` to disable the plugin. When `--worker-cls` is omitted, AFD selects the
Attention or FFN worker for CUDA or standard Ascend NPU. Explicit worker paths
are accepted for compatibility, but are not stable launch interfaces.

Both roles require `VLLM_USE_V2_MODEL_RUNNER=1`. Attention uses native
ModelRunnerV2; FFN is connector-driven. Send requests only to the Attention API.

## Common fields

```json
{
  "afd": {
    "role": "attention",
    "connector": "P2pNcclAFDConnector",
    "async_dp": false,
    "host": "127.0.0.1",
    "port": 1239,
    "afd_process_group_timeout_s": 120,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 1,
    "compute_gate_on_attention": false,
    "connector_extra_config": {}
  }
}
```

| Field | Default | Meaning and constraint |
| --- | --- | --- |
| `role` | `attention` | `attention` or `ffn`. |
| `connector` | `P2pNcclAFDConnector` | One of the connectors linked below. |
| `async_dp` | `false` | Must be `true` for CAM async or GPU async, and `false` for synchronous connectors. |
| `host` | `127.0.0.1` | Shared, reachable rendezvous host; placement depends on the connector. |
| `port` | `1239` | Rendezvous port in `1..65535`, separate from the HTTP service ports. |
| `afd_process_group_timeout_s` | `120` | Positive rendezvous timeout in seconds; also bounds the NPU native DBO startup barrier, not device-kernel execution. |
| `num_attention_ranks` | `1` | Positive total Attention worker count, including DP/TP ranks. |
| `num_ffn_ranks` | `1` | Positive total FFN worker count, including DP/TP ranks. |
| `compute_gate_on_attention` | `false` | Gate placement; allowed values depend on the connector and model. |
| `connector_extra_config` | `{}` | Connector-specific mapping; unknown keys fail in the selected connector parser. |

The example describes Attention in a 2A1F deployment. Set `role` to `ffn` on
FFN; all ranks must agree on rendezvous and topology. Role rank is derived at
runtime and is not a configuration field. Increase the rendezvous timeout when
model loading or rank placement delays group initialization.

## Compatibility aliases

`afd_connector`, `afd_role`, `afd_host`, `afd_port`, and `async` map to
`connector`, `role`, `host`, `port`, and `async_dp`. Supplying an alias and its
canonical key together is rejected. Unknown top-level AFD keys are rejected;
put connector-specific settings under `connector_extra_config`.

## Connector-specific settings

Each guide owns its extra fields, topology rules and launch examples:

- [NCCL P2P](gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md#configuration): synchronous CUDA;
  extra config must be empty.
- [DeepEP M2N](gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#configuration): synchronous CUDA;
  Attention-side routing and FFN TP1.
- [GPU async](design/async_gpu_cross_node_research.md): experimental CUDA,
  direct peer writes locally and NVSHMEM RMA remotely. Cross-node use requires
  the [optional GPU extension](../csrc/gpu/README.md). Requires `async_dp=true`,
  `compute_gate_on_attention=true`, static EP and FFN TP1.
  Hardware evidence is limited to H20/DeepSeek-V2-Lite-Chat.
- [CAMP2P](npu/CAM_P2P_CONNECTOR_USER_GUIDE.md#afd-configuration): synchronous NPU;
  both common and connector-local gate settings must be `false`.
- [CAM async](npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md#afd-configuration): NPU;
  requires `async_dp=true`, `compute_gate_on_attention=true`, and `dynamicQuant=1`.

See [supported configurations](support.md) before combining graph, DBO,
parallelism, or quantization settings.
