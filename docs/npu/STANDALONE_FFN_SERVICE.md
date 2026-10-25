# Standalone CAM FFN service

The standalone entry point loads DeepSeek-V4 Flash per-channel ModelSlim W4A8
routed experts in an external F-only process group. After native loading,
sharding and packing complete, it retires the loading groups and lends the
original parameter tensors to the existing Async CAM runtime. It does not
construct a complete CausalLM, vLLM Worker, upstream ModelRunner, EngineCore,
request scheduler or KV cache.

This is a staged independent-service delivery under
[RFC #391](https://github.com/vllm-project/afd-plugin/issues/391). The megakernel
attachment remains an architectural reference for parameter ownership and
lifecycle. This revision does not implement a fused communication/compute
megakernel or qualify the attachment's broader session and bounded-drain
semantics. The earlier transport work remains available separately.

## Scope and dependencies

This entry point reuses `CAMAsyncAFDConnector`, `AsyncCAMW4A8Executor`,
`AsyncNPUFFNModelRunner`, `AsyncFFNTwoStream` and `AsyncCAMGraphStartup`. The
Attention model, router, shared experts, padding, DBO hooks and connector are
the existing CAM path. There is no FaaS-specific connector, wire ABI or
PUBLISH/PREPARE/FINISH control operator in this implementation.

The supported model is DeepSeek-V4 Flash with homogeneous per-channel W4A8
routed parameters on A3/910C, static expert parallelism and F TP=1. All main
model MoE layers are loaded; MTP layers, gates, shared experts and Attention
weights are not instantiated. The native ModelSlim factory, checkpoint loader
and quantizer own expert mapping, NZ packing, encoded scales and compensation.

Layered V4 execution requires the separate common SwiGLU clamp correctness
fix: gate is capped above, up is clamped symmetrically, and both clamps precede
SwiGLU and activation quantization. A zero limit preserves unlimited activation.
Rebuild the plugin extension and CANN package together after this API change;
an older binary cannot implement the new `swiglu_limit` argument. This
dependency applies to traditional CAM and standalone F equally. It does not
include adaptive tiling, W2 `.out` or a new transport. The original ACLNN entry
point keeps its unlimited-activation ABI; the new binding resolves a distinct
clamp-aware symbol rather than passing new arguments to an older library.

## Launch

Keep the existing Attention launch and its `CAMAsyncAFDConnector` configuration.
See the [connector guide](CAM_ASYNC_CONNECTOR_USER_GUIDE.md) and
[eager/FULL runtime guide](ASYNC_FFN_RUNTIME.md) for its existing contract.
It must use eager MRV2, Attention-side gating and dynamic quantization. The
documented native Attention DBO combination remains supported; do not enable
DBO on F.

For a single-node 12A/4F deployment, with A using logical devices 0–11 and a
matching dispatch envelope of 2048 tokens:

```bash
# Activate the existing CANN environment and keep its Python/library paths.
# GE must see both native vendor paths before the first torch-npu import.
afd_cam_vendor=/path/to/afd_plugin/_cann_ops_custom/vendors/afd-plugin
ascend_native_vendor=/path/to/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$ascend_native_vendor:$afd_cam_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export AFD_ASYNC_CAM_LAYERED_GMM=1
export SOC_VERSION=910c
export VLLM_PLUGINS=ascend,afd,ascend_model,ascend_model_loader,ascend_kv_connector
# Keep BATCH_SIZE_FACTOR, HCCL/network settings and receive capacity consistent
# with the existing A launch and CAM runtime guide.
torchrun --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 \
  -m afd_plugin.ffn_faas serve /path/to/model \
  --host 127.0.0.1 --port 29600 \
  --attention-ranks 12 --attention-tp 4 --ffn-ranks 4 \
  --devices 12,13,14,15 --max-num-batched-tokens 2048 \
  --full-graph --ffn-two-stream \
  --ffn-compute-aic-num 20 --ffn-compute-aiv-num 40 \
  --ffn-combine-aiv-num 8 --status-port 29700
```

`--devices` indexes the process's visible logical devices, not physical IDs.
When F has a restricted visibility list, use `--devices 0,1,2,3` instead. The
external launcher's `WORLD_SIZE` is F-only. CAM creates the separate
`[A0 ... A11, F0 ... F3]` world using the same host, port, group name, topology,
quantization, layer table and startup signatures as traditional F. Use a
different rendezvous port for the F-only loading group and CAM.

`--max-num-batched-tokens` must match A's dispatch envelope; it is not the
maximum request context length. F creates configuration objects for native
loading, but allocates no request KV cache. Omit `--full-graph` to use the
mature layered eager path. Two-stream execution supports both eager and FULL;
omitted core budgets use the existing CAM defaults. FULL preserves paired
warmup, private graph pools, device layer selection and FIFO slot ownership.

## Management and shutdown

Each local F rank exposes a management endpoint on `--status-port + LOCAL_RANK`.
`--status-port 0` chooses an ephemeral port. `/status` reports model loading,
CAM pairing, phase, completed submission batches and resource-release state.
These batches are not requests or complete model forwards. `/health` is host
liveness; `/ready` is available only after weights and paired CAM startup are
prepared. Neither endpoint proves progress of an outstanding device receive.

`POST /stop`, SIGINT and SIGTERM request a stop between completed batches.
The host joins its receive loop for `--shutdown-timeout` seconds. This timeout
does **not** cancel a CAM operator or prove device quiescence.
It applies after startup; native loading and rendezvous keep their existing
timeouts. An unresponsive startup also requires whole-deployment isolation.

- After a successful receiver exit and completion fence, release graphs,
  connector resources and finally the routed weight owner.
- If startup or execution fails, or a receive is still pending, report failure
  with `resources_released=false`. Do not destroy its communicator, drop
  captured pools or release weights still referenced by native work.
- The CLI retains owners until an immediate, nonzero process exit, bypassing
  Python destructors for that unsafe session. The operator must tear down the whole
  paired A/F deployment; independent hot restart or lossless bounded drain is
  not supported by the mature CAM contract.

Stop or restart policies must account for CAM's potentially unbounded DR. A
host signal, an HTTP response or a completed client request does not establish
that the next pre-submitted receive has finished.

## Validation and performance reporting

CPU tests cover metadata/configuration and host ownership decisions; they do
not qualify native packing, HCCL, Graph capture or pending device work. Native
qualification for this revision covered 24 clamp/unlimited and empty/tail
numerical cases, finite synthetic eager/FULL CAM execution, and the original
43-layer checkpoint with one A peer and four independent F processes. The
FULL/two-stream outputs matched eager exactly. A final extra transaction proved
the next F batch was pending before stop; all F processes retained their owners
and exited through paired-process isolation.

The [native A fixture](../../tests/npu/standalone_cam_attention.py) drives these
independent layer inputs and can hold its communication owner for stop checks.
This qualifies FFN/CAM behavior, not Attention, KV, HTTP inference, whole-model
accuracy, production-topology throughput or latency. Reuse applicable existing
qualification when the execution path is unchanged.

Performance comparisons keep model/W4A8, topology, input, concurrency, output
length, explicit prefix-cache disablement and runtime warmup comparable.
Report batch elapsed time and mean/tail TTFT separately. Historical results
for the separate FaaS transport are not measurements of this backend. This
architecture change does not promise a speedup over the mature CAM backend.
