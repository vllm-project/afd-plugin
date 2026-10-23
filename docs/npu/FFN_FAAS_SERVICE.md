# Independent FFN service for DeepSeek-V4

> Draft: full-model native numerical equivalence remains unverified; further
> precision investigation is deferred. Normal drain with the shutdown settings
> below has been verified on 12A4F. This document describes the current
> implementation and launch contract.

The initial integration targets DeepSeek-V4 Flash ModelSlim per-channel W4A8
on Ascend A3, BF16 activations, eager Attention, contiguous F EP and F TP1.
Attention retains its native Hash/sqrtsoftplus router, shared experts, HC and
TP/SP behavior. F loads only routed parameters and runs native quantization,
GMM1/SwiGLU/clamp and GMM2. The backend performs a bounded host metadata read
between transport preparation and computation. It makes no fusion or speedup
claim. V2 fusion is a separate follow-up.

Use matching pinned vLLM/vLLM-Ascend dependencies and rebuild AFD's Ascend
extension and `afd_ffn_faas` operator from this source. Native custom-op
bootstrap must precede the first device initialization; internal NPU format
must be enabled for native W4A8 NZ tensors. The service preserves native vendor
priority and appends AFD to `ASCEND_CUSTOM_OPP_PATH`. AFD-specific APIs use
`AFD_CUST_OPAPI_LIB_PATH`. Putting AFD first can resolve a native GMM API from
its CANN dependency with an incompatible signature. Loading creates temporary
F-local native groups for expert sharding/quantization and tears them down before the A/F
transport session. F starts no complete vLLM worker/model runner/EngineCore.

Integration uses vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and
vLLM-Ascend `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`. The launch plan enables
the functional Ascend plugins `ascend,ascend_model,ascend_model_loader,ascend_kv_connector`;
A additionally enables `afd`. The platform plugin alone does not register the
native V4 model. A and the native reference use BF16 main KV cache and INT8
indexer KV cache: `--kv-cache-dtype bfloat16 --attention-config
'{"indexer_kv_dtype":"int8"}'`.

Set `HCCL_BUFFSIZE` before Python/device initialization to the same MiB value as
the connector's `hccl_buffer_size` and F's `--window-mb`; the small launch plan
sets all three to 64. This also keeps legacy Async CAM comparisons valid:
its host tiling reads the environment independently of the per-domain HCCL
allocation. An unset variable uses the host helper's 408 MiB fallback even
when the domain allocates 64 MiB, placing the window-tail state out of bounds.
See [the window contract](CAM_ASYNC_ROUTED_OPS.md#communication-window-configuration).

The small-workload E2E tool prints a launch plan; inspect paths and resource
allocation before executing its commands:

```bash
python -m tests.e2e.models.deepseek_v4_flash.ffn_faas_inference --help
python -m afd_plugin.ffn_faas serve --help
```

The deployment topology used for integration is 12 A ranks (DP3/TP4) and four
F ranks (EP4/TP1). Launch F with an F-only torchrun world, explicit visible
devices and a separate A/F rendezvous port. For example, after setting the
allocated four devices visible:

```bash
torchrun --nproc-per-node=4 --master-port=29501 \
  -m afd_plugin.ffn_faas serve /path/to/DeepSeek-V4-Flash \
  --attention-ranks 12 --attention-tp 4 --ffn-ranks 4 \
  --host 127.0.0.1 --port 29600 --capacity 8 --slots 2 \
  --status-port 29700 --startup-timeout 900
```

A uses `FaaSAsyncAFDConnector` with matching model, ranks, capacity, slots and
HCCL window. The plan tool emits the complete A command and connector settings.
Use its `freeze` command with the actual V4 tokenizer once, then send that same
frozen input to a trusted native reference and the FaaS service. The native
reference topology and the FaaS topology have different device budgets, so
these small runs establish behavior/numerics and descriptive timing only.
Save warmup and measured outputs; compare token IDs, finish reasons and finite
log probabilities with explicit, independently selected tolerances.

Each F process serves CPU-only `/health`, `/ready`, `/status` and `/stop`
management endpoints at status-port + local rank. Readiness follows complete
loading, session agreement, window initialization and paired zero-input warmup.
Status separates loading, connecting, warmup, ready, draining, stopped and
failed. SIGINT/SIGTERM request stop. Stop during active work aborts; normal
completion requires A to finish accepted requests and close the session.
The A launch must allow this peer cleanup to finish. The plan sets
`--shutdown-timeout 120` and `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=120`;
vLLM's default abort shutdown can kill the Store-owning A worker while F still
uses the rendezvous Store. Request A's normal shutdown before stopping F.
The 2026-10-23 12A4F run completed all 16 small HTTP requests, returned zero
from both launchers, and left every F rank stopped with resources released.
All 16 DIE were free of processes afterward; no forced cleanup was needed.

## Small same-resource comparison

On 2026-10-23, both services ran sequentially on the same 16 DIE with
12A4F (A DP3/TP4, F EP4/TP1), the pinned dependencies above, BF16 activations,
W4A8 weights, eager execution and 64 MiB communication windows. The legacy
AFD source was `25027746ee8ff4b3776fc836825139e3db0a7856`; the FaaS runtime
was `2a210bce13c17e4ed24a012d0a74ed7b5f241789`. Both used the same frozen four
short prompts, concurrency 2, a 16-token output cap, one warmup and three
measured rounds. Every round produced 64 output tokens. Timing covers the
first HTTP send through the final response, including client scheduling;
startup and shutdown are excluded. No profiler or tensor observer ran during
measurement.

| Service | Measured rounds (s) | Mean (s) | Output tokens/s |
| --- | --- | --- | --- |
| Legacy AFD | 5.022, 5.507, 4.773 | 5.101 | 12.547 |
| Native-operator FaaS | 5.243, 5.306, 5.358 | 5.303 | 12.070 |

The FaaS mean was 3.96% higher; these three-round descriptive results show no
speedup on this small workload and establish no broader performance claim.
Both arms completed all 16 HTTP requests. Full-model numerical acceptance
was deferred, so successful requests are not evidence of numerical equivalence.
This compares the service implementations, whose transport and input
quantization differ; it does not isolate a compute-fusion effect.

The FaaS repeat again exited with both launchers returning zero and all four
F ranks stopped with resources released. Its A API log printed upstream
`EngineDeadError` after intentional MPClient shutdown completed; the prior
F-side Store cleanup failure did not recur. Legacy A returned zero, while
its F API parent returned one when SIGTERM interrupted pending API
initialization. Neither run required forced cleanup, and all DIE were idle
before the next service started. Legacy readiness used the A HTTP endpoint
and all four F daemon loops; its unused F HTTP endpoint did not become ready.

An unknown device failure or unsafe peer cleanup terminates the isolated worker
without releasing potentially accessed storage. Do not restart another session
on the same allocation until the previous process/device access has ended.
See [the control ABI](FFN_FAAS_ABI.md) for generation, credit and completion
ownership. CPU checks and successful loading alone are not full-model or
native numerical acceptance.
