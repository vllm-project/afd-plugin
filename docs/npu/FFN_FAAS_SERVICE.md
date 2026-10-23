# Independent FFN service (DeepSeek-V2 BF16 / DeepSeek-V4 W4A8)

> Draft: full-model native numerical equivalence and normal drain are not yet
> established for every topology. The shutdown settings below still require a
> successful device rerun. This document describes the current implementation
> and launch contract.

The standalone service serves two checkpoint contracts over the same session:
unquantized BF16 DeepSeek-V2 (fused device megakernel) and DeepSeek-V4 Flash
ModelSlim per-channel W4A8 (host-controlled native GMMs). Both require Ascend
A3, eager Attention, contiguous F EP and F TP1. Attention keeps routing, dense
layers and shared experts; F loads only its routed expert shard. DBO, sequence
parallelism, PP, CP, quantization mismatch and speculative decoding are
rejected. The service makes no fusion or speedup claim for V4; V2 fused
behavior and its historical performance experiments do not establish V4
acceptance.

The fused BF16 `afd_ffn_faas` `EXECUTE` entry polls HCCL windows, selects the
source and physical layer, computes SwiGLU experts with Cube projections, AIV
FP32 accumulation and BF16 boundaries, writes remote results and publishes
completion. After warmup the service uses `SERVE` to retire up to eight such
transactions per launch and reports one aggregate receipt. There is no separate
recv/GMM/send fallback. The V4 backend instead reads a bounded host receipt
between `PREPARE` and `FINISH` and launches native quantized GMMs. See
[the ABI](FFN_FAAS_ABI.md) for both device protocols.

V2 routing stays in FP32. Before dispatch, the model connector rounds selected
probabilities to BF16 and converts them back to the protocol's FP32 storage.
This matches the native BF16 token-unpermute contract; it is distinct from
changing route selection or expert arithmetic. V4 keeps its already scaled
FP32 routing probabilities.

## Build

Use matching pinned vLLM/vLLM-Ascend dependencies and rebuild AFD's Ascend
extension and `afd_ffn_faas` operator from this source. The standard build
cleans its private native build/output/vendor directories. For development,
selecting only the FFN FaaS operator avoids rebuilding unrelated operators:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:$PYTHONPATH"
AFD_BUILD_OPS=afd_ffn_faas AFD_BUILD_JOBS=4 MAX_JOBS=4 \
  AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=910c python setup.py build_ext --inplace
```

Omit `AFD_BUILD_OPS` to build all default operators. Verify the loaded extension
and installed vendor are from this checkout. The repository kernel, staged
kernel and installed dynamic kernel source must have identical SHA256 values;
stale incremental packaging is not reproducible build evidence. The recorded
environment uses torch 2.10.0, torch-npu 2.10.0.post4 and CANN 9.1.0; other
CANN releases have not been validated for this operator.

Native custom-op bootstrap must precede the first device initialization;
internal NPU format must be enabled for native W4A8 NZ tensors. The service
preserves native vendor priority and appends AFD to `ASCEND_CUSTOM_OPP_PATH`.
AFD-specific APIs use `AFD_CUST_OPAPI_LIB_PATH`. Putting AFD first can resolve a
native GMM API from its CANN dependency with an incompatible signature. Loading
creates temporary F-local native groups for expert sharding/quantization and
tears them down before the A/F transport session. F starts no complete vLLM
worker/model runner/EngineCore.

Integration uses vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and
vLLM-Ascend `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`. The launch plan enables
the functional Ascend plugins `ascend,ascend_model,ascend_model_loader,ascend_kv_connector`;
A additionally enables `afd`. The platform plugin alone does not register the
native V4 model. A and the native reference use BF16 main KV cache and INT8
indexer KV cache: `--kv-cache-dtype bfloat16 --attention-config
'{"indexer_kv_dtype":"int8"}'`.

## Launch

Each launcher starts one process per rank. Both roles must use the same local
checkpoint, topology, capacity, slots and window size. The F launcher WORLD_SIZE
contains only F ranks. Session ranks are independently ordered `[A..., F...]`
and the checkpoint architecture selects the executor. For a DeepSeek-V4 Flash
deployment (12 A ranks DP3/TP4, four F ranks EP4/TP1), after setting the
allocated four devices visible:

```bash
torchrun --nproc-per-node=4 --master-port=29501 \
  -m afd_plugin.ffn_faas serve /path/to/DeepSeek-V4-Flash \
  --attention-ranks 12 --attention-tp 4 --ffn-ranks 4 \
  --host 127.0.0.1 --port 29600 --capacity 8 --slots 2 \
  --status-port 29700 --startup-timeout 900
```

For BF16 DeepSeek-V2-Lite (A DP2/TP2 and F EP2/TP1 within one allocated
six-device host):

```bash
torchrun --standalone --nproc-per-node=2 -m afd_plugin.ffn_faas \
  serve /path/to/DeepSeek-V2-Lite --devices 4,5 \
  --attention-ranks 4 --attention-tp 2 --ffn-ranks 2 \
  --host 127.0.0.1 --port 29600 --capacity 128 --slots 2 \
  --window-mb 64 --status-port 29700 --status-file output/ffn-status.json
```

A uses `FaaSAsyncAFDConnector` with matching model, ranks, capacity, slots and
HCCL window. For V2 use `--enforce-eager --dtype bfloat16
--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel` and:

```json
{"afd":{"connector":"FaaSAsyncAFDConnector","role":"attention","async":true,
"compute_gate_on_attention":true,"num_attention_ranks":4,"num_ffn_ranks":2,
"host":"127.0.0.1","port":29600,"connector_extra_config":{
"attn_ranks_per_dp":2,"capacity":128,"slots":2,"hccl_buffer_size":64,
"timeout_s":30,"startup_timeout_s":300}}}
```

Device selection must follow the allocation's logical-to-physical mapping. Do
not replace an existing allocation visibility mask with guessed physical IDs.
The F `--devices` option binds logical IDs inside that allocation. Readiness is
paired: all windows initialize, F proves an idle native launch, A sends zero-input
expert work on two routed layers, then both sides acknowledge completion.

The production connector uses `prepare_cam_dispatch_payload` with SP disabled:
all TP members have equal padded row counts and execute the same ordered model
calls. It derives identical chunk counts locally, with no per-layer Store
exchange or polling. The native protocol verifies TP sequence/layer identities.
All-zero calls still use one transport generation. FaaS marks its TP padding
routes with invalid expert IDs; CAM keeps its existing zero-filled IDs.

The standalone client also accepts explicitly unaligned TP rows. Only this
contract uses one overwritable shape/identity Store cell per A rank and agrees
the maximum chunk count within that source. Exhausted members publish zero work
for every remaining chunk; other DP sources and F ranks do not participate.
Finishing a call's first native chunk requires every TP to have read its
descriptor and published that chunk, so a fast TP cannot overwrite a descriptor
its peer still needs. This bounded layout agreement is not an F schedule.

The small-workload E2E tools print launch plans; inspect paths and resource
allocation before executing their commands:

```bash
python -m tests.e2e.models.deepseek_v4_flash.ffn_faas_inference --help
python -m tests.e2e.models.deepseek_v2_lite.ffn_faas_inference --help
python -m afd_plugin.ffn_faas serve --help
```

Save warmup and measured outputs; compare token IDs, finish reasons and finite
log probabilities with explicit, independently selected tolerances.

## Management and shutdown

Each F process serves CPU-only `/health`, `/ready`, `/status` and `/stop`
management endpoints at status-port + local rank. Readiness follows complete
loading, session agreement, window initialization and paired zero-input warmup.
Status separates loading, connecting, warmup, ready, draining, stopped and
failed. SIGINT/SIGTERM request stop. Stop during active work aborts; normal
completion requires A to finish accepted requests and close the session.

The main worker thread binds the device, creates FFNModelRunner and owns device
execution. Model owns weights; `NPUExecutor` borrows them and owns the private
process group, windows, stream, events and state. `NativeW4A8Executor` extends
that executor for V4. Management threads only read locked CPU state. `/health`,
`/ready`, `/status` are GET endpoints; `POST /stop` and SIGTERM request a bounded
session abort. Bind management HTTP to a trusted interface; the default is
loopback, without authentication. SIGTERM/SIGINT handlers only assign a
lock-free pending flag. Ordinary execution boundaries observe it, revoke
readiness and perform cleanup; neither service state locks nor Event condition
locks are acquired inside the signal handler.

Normal drain starts after Attention has finished every accepted request's model
layers. A publishes its last consumed generation; F exits after all sources end.
The FaaS Attention model runner closes its connector before upstream shutdown
synchronizes or frees model resources. The tested launcher sets
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=90` and gives engine shutdown 120 seconds;
the upstream five-second default killed workers before this drain completed.
The V4 plan sets `--shutdown-timeout 120` and
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=120`; vLLM's default abort shutdown can
kill the Store-owning A worker while F still uses the rendezvous Store. Request
A's normal shutdown before stopping F. All ranks acknowledge their final native
event before the group is destroyed and borrowed weights are dropped. An F stop
while Attention is live is a failure, never a successful drain. Store heartbeats
and bounded native polls detect failed peers. Device-event waits and quiescence
have a configured timeout; startup rendezvous has its separate startup timeout.

If a peer dies or an event cannot complete, quiescence cannot be proven. The
standalone service records `resources_released=false` and exits its process
without invoking unsafe HCCL/Python destructors. The launcher and device runtime
reclaim that process's allocation. Such an exit is not reported as normal
cleanup. The dedicated FaaS Attention connector applies the same process
boundary when its worker cannot prove quiescence; it records the failure and
exits while still holding its executor. Other connectors keep their existing
shutdown behavior. Do not restart another session on the same allocation until
the previous process/device access has ended.

## Device probes

`tests/npu/ffn_faas_native.py` exercises raw protocol admission, independent F
ordering, backpressure, SERVE receipts and per-expert plus combine comparisons
before credit return. `tests/npu/ffn_faas_service.py` is an Attention-only
client of the separate service, using all checkpoint layers, repeated slots and
an independent CPU reference. `tests/npu/ffn_faas_service_driver.py` launches
both roles for normal and idle/in-flight/killed-peer/lost-store/signal fault
cases. Diagnostic probes are excluded from production timing claims. The
full-model driver in `tests/e2e/models/deepseek_v2_lite/ffn_faas_driver.py`
starts the actual vLLM Attention workers and independent F processes. Its
`passed` field validates process completion and lifecycle only; numerical
acceptance requires the separate saved-output comparison described in the
validation report.

See [the control ABI](FFN_FAAS_ABI.md) for generation, credit and completion
ownership. CPU checks and successful loading alone are not full-model or native
numerical acceptance.
