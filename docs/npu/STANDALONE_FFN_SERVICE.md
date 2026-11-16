# Standalone CAM FFN service

Run DeepSeek-V4 Flash routed experts in independent F-only processes with
`python -m afd_plugin.ffn_faas serve`. The service reuses native parameter
loading and the existing CAM execution path without constructing a complete
CausalLM, vLLM Worker, upstream ModelRunner, EngineCore, scheduler or KV cache.

## Supported configuration

- DeepSeek-V4 Flash with homogeneous per-channel ModelSlim W4A8 on A3/910C.
- Static expert parallelism with F TP=1. EPLB, elastic EP and redundant experts
  are unsupported.
- Main-model routed MoE layers only. MTP layers, gates, shared experts and
  Attention weights are not loaded by F.
- The existing Attention model, router, shared experts, padding, DBO hooks and
  `CAMAsyncAFDConnector` are reused. F supports layered eager or FULL Graph
  execution.

Configuration, expert mapping, parameter loading and quantization depend on
vLLM/vLLM-Ascend. Other models, quantization schemes and device backends are
not supported by this entry point.

## Architecture

```text
CPU Service / host runner -> FFNBackend
  Ascend CAM backend
    AscendFFNEnvironment: device, F-only compute groups, CPU affinity
    CAMRoutedModel: parameter loading, native owner and cached views
      V4ModelAdapter / V4RoutedWeightOwner: V4 metadata and native loading
    Async CAM runtime: A/F group, DS/DR/CS/CR, Graph and streams
```

The backend adapts the complete CAM runtime lifecycle. Communication, compute,
Graph execution remains in that runtime. See the
[FFN runtime design](../design/module/ffn_runtime.md#runtime-selection) for
module responsibilities and ownership.

## Launch

Keep the existing Attention launch and its `CAMAsyncAFDConnector` configuration.
See the [connector guide](CAM_ASYNC_CONNECTOR_USER_GUIDE.md) and
[eager/FULL runtime guide](../design/npu/async_ffn_runtime.md) for its existing contract.
It must use eager MRV2, Attention-side gating and dynamic quantization. Native
Attention DBO uses the existing integration contract; do not enable DBO on F.

For a single-node 12A/4F deployment, with A using logical devices 0–11 and a
matching dispatch envelope of 2048 tokens:

```bash
# Activate the existing CANN environment and keep its Python/library paths.
# GE must see both native vendor paths before the first torch-npu import.
afd_cam_vendor=/path/to/afd_plugin/_cann_ops_custom/vendors/afd-plugin
ascend_native_vendor=/path/to/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$ascend_native_vendor:$afd_cam_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export SOC_VERSION=910c
export VLLM_PLUGINS=ascend,afd,ascend_model,ascend_model_loader,ascend_kv_connector
# Keep BATCH_SIZE_FACTOR, HCCL/network settings and receive capacity consistent
# with the existing A launch and CAM runtime guide.
export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15
torchrun --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 \
  -m afd_plugin.ffn_faas serve /path/to/model --backend ascend-cam \
  --trust-remote-code --dtype bfloat16 --quantization ascend \
  --enable-expert-parallel --data-parallel-size 4 \
  --max-num-batched-tokens 2048 --max-num-seqs 256 \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --additional-config '{
    "afd": {
      "connector":"CAMAsyncAFDConnector","role":"ffn","async_dp":true,
      "host":"127.0.0.1","port":29600,
      "num_attention_ranks":12,"num_ffn_ranks":4,
      "compute_gate_on_attention":true,
      "connector_extra_config":{
        "dynamicQuant":1,"attn_ranks_per_dp":4
      }
    },
    "enable_prefill_mc2":false,"enable_force_eplb":false,"enable_cpu_binding":true,
    "ascend_fusion_config":{"fusion_ops_gmmswigluquant":true}
  }' --status-port 29700
```

Engine arguments use vLLM's `FlexibleArgumentParser`, `EngineArgs.add_cli_args`
and `EngineArgs.from_cli_args`; definitions, JSON/dotted options, defaults and
configuration validation come from the pinned upstream stack. The selected
Ascend CLI imports that layer, while importing the Service, host runner or
platform adapter module remains CPU-only. No serving engine is constructed.

Use `--additional-config` for the existing AFD topology, connector and HCCL
options. Engine capacities and compilation are passed through
without rebuilding a second configuration. Only backend selection, management
endpoints, status files and shutdown control add standalone options. The old
`--full-graph`, `--attention-ranks`, `--attention-tp`, `--ffn-ranks`, `--devices`
and standalone stream/budget flags have been removed; update launch scripts to
the standard options above. Device selection uses the native `--device-ids`
mapping or `ASCEND_RT_VISIBLE_DEVICES` plus the launcher's `LOCAL_RANK`.

The external launcher's `WORLD_SIZE`, `--data-parallel-size` and
`additional_config.afd.num_ffn_ranks` must agree. An omitted
`--data-parallel-rank` is taken from `RANK`; an explicit value must match it.
The executor defaults to `external_launcher` because torchrun already owns
these processes. Set `VLLM_USE_V2_MODEL_RUNNER=1` as on A. All other engine
options retain upstream defaults; the example explicitly selects the qualified
W4A8 profile. Reusing the parser does not qualify additional models, speculative
loading or serving/KV features for this service.

The host receive-loop join uses `--receiver-shutdown-timeout` (default: five
seconds). It is separate from vLLM's `--shutdown-timeout`, whose EngineCore
shutdown semantics do not govern this standalone receiver.

CAM creates the separate `[A0 ... A11, F0 ... F3]` world using the same host,
port, group name, topology, quantization, layer table and startup signatures as
traditional F. Use a different rendezvous port for the F-only compute world
and CAM.

`--max-num-batched-tokens` must match A's dispatch envelope; it is not the
maximum request context length. F allocates no request KV cache. To use eager,
replace `--compilation-config` with `--enforce-eager`. FULL preserves paired
warmup, private graph pools and device layer selection. FFN two-stream
execution and its core-budget options were removed in #550; older launch
commands must omit those options.

## Management and shutdown

Each local F rank exposes a management endpoint on `--status-port + LOCAL_RANK`.
`--status-port 0` chooses an ephemeral port.

- `/status` reports loading, CAM pairing, phase, completed submission batches
  and resource-release state. Batches are not requests or full model forwards.
- `/health` reports host liveness. `/ready` becomes available after native
  weights and paired CAM startup are prepared. Neither proves progress of an
  outstanding device receive.
- `POST /stop`, SIGINT and SIGTERM request a stop between completed batches.
  The host joins its receive loop for `--receiver-shutdown-timeout` seconds.

The shutdown timeout does not cancel a CAM operator or prove device quiescence.
It applies after startup; native loading and rendezvous retain their existing
timeouts. CAM pairing follows native weight loading. An unresponsive startup
requires whole-deployment isolation.

After receiver exit and device completion, the service releases CAM/Graph
resources, then native compute groups, and finally the routed parameter owner.
If native work fails or a receive remains pending, it reports failure with
`resources_released=false` and retains those owners until an immediate nonzero
process exit, bypassing Python destructors. Diagnostic and HTTP-cleanup errors
cannot bypass this exit.

The operator must tear down the entire paired A/F deployment after such a
failure. A completed client request does not establish that the next
pre-submitted receive has finished. Graceful bounded drain, independent hot
restart and device cancellation are unsupported.
