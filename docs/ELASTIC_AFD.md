# Elastic AFD development build

This implements the GPU TP=1 path of the EEP-reuse design against vLLM
**0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0** and afd-plugin
**8bb14be66d9f940b1b132b141d986214c0210353**. It has CPU unit coverage and
source review; it has **not yet been validated on GPUs**. Do not treat the
example below as a qualified deployment recipe.

## Implemented flow

- The existing `/scale_elastic_ep` route defaults to Attention. `role=ffn`
  changes only the FFN pool. The integer is the selected role's **DP size**.
- A uses the native EEP client, `CoreEngineActorManager`,
  `reinitialize_distributed`, `ElasticEPScalingState`, notifications, KV budget
  synchronization, and routing updates. No EEP state machine is copied.
- The worker executor replaces expert mapping/weight migration/EPLB operations
  with paired no-ops and actual checkpoint reloads. Retained A rebuild their
  runner and workspace, then initialize KV through the native allocator path.
- F uses separate Ray actors with a UniProcExecutor and the existing background
  FFN loop. F-only scaling releases every F role, resizes the actor pool, and
  reconstructs each model against the new F DP/EP group. It does not reload
  only the weights of an old model object.
- Every change drains requests, reaches native DP pause consensus, sends STOP
  over the existing metadata channel, joins F, and releases graphs before
  communication groups. Both roles connect concurrently; F starts receiving
  before A profile/warmup/capture. Service resumes only after this completes.
- A scaling waits for **every target EngineCore** to report
  `eep_scaling_state is None`. Native `RECONFIGURE_FINISHED` alone is too early.
- The official endpoint plugin's `init_state` runs the same cross-role finish
  before HTTP starts. Initial/new A defer connector/profile/warmup and stay
  paused after native READY.

The old M1 implementation informed F ownership, role-config isolation, and
repeatable stop/restart. The implementation does not carry over the global
ActorManager hierarchy, FFN EngineCore/ZMQ identities, exception-driven stop,
in-place expert reload, or hard-coded graph pools.

## Current boundaries

GPU only, `P2pNcclAFDConnector`, MRV1, A DP >= 2, F DP >= 1, A ranks >= F
ranks, TP=PP=PCP=DCP=1. Existing GPU non-divisible mappings are retained.
One API process and internal load balancing are required. Specify Ray DP and
the `uni` role executor separately. DBO, speculative decoding, LoRA, sleep,
KV transfer and KV retention are outside this build. KV memory is fixed
explicitly. Model/precision qualification starts with DeepSeek-V2-Lite.

Graph-release/recapture code is included, but neither eager nor CUDA graph
execution is hardware-qualified yet. Start validation in eager mode.

**Ascend is not implemented as an elastic backend in this build.** Static NPU
AFD remains available. Elastic configuration fails early with the stateless
HCCL qualification requirement. The pinned Ascend worker/platform lacks the
EEP device-group support needed for the shared A state machine; this needs
the N0 group-creation/collective/switch/destruction test before an NPU worker
implementation can be accepted. Receiving STOP is supported by the existing
NPU metadata loop, but that alone does not provide NPU elasticity.

Concurrent resize is rejected using the existing scaling gate. There is no
rollback/retry/recovery system. A failure after communication teardown needs
a service restart; this build makes no failure-availability guarantee.

## Candidate launch on an isolated Ray GPU cluster

Install the pinned vLLM runtime and this plugin checkout on every Ray node.
The complete candidate sequence peaks at six available GPUs (4 A + 2 F).
The initial topology needs three. Use the same model checkpoint on each node.
The code uses the actual GPU assigned to each Ray placement, including A
placements when F is colocated. It does not reserve whole nodes for a role.

```bash
export VLLM_PLUGINS=afd,afd_elastic
export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve "$MODEL" \
  --served-model-name elastic-afd \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --enable-expert-parallel --enable-elastic-ep \
  --data-parallel-backend ray --distributed-executor-backend uni \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --tensor-parallel-size 1 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 1073741824 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","num_attention_ranks":2,"num_ffn_ranks":1},"afd_elastic":{"ffn_tensor_parallel_size":1}}'
```

The 1 GiB KV budget above is an example, not a measured memory recommendation.
Allow room for the largest tested topology, communication buffers and captures.
Do not set `--enable-eplb`. The plugin narrowly exempts its explicit attention
worker from EPLB and async-EPLB/NIXL prerequisites; native EEP keeps its checks.
The AFD rendezvous host and ports are distributed from F0's actual node.

Candidate scale chain, issuing the next call only after the preceding succeeds:

```bash
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":4,"drain_timeout":300}'
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":2,"role":"ffn","drain_timeout":300}'
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":2,"drain_timeout":300}'
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":1,"role":"ffn","drain_timeout":300}'
```

`POST /is_scaling_elastic_ep` remains available while an AFD resize is active.
Other requests receive the native scaling middleware's 503 response.

## Validation and remaining work

CPU tests are in `tests/unit/elastic`. They cover real orchestration methods
with device/actor doubles, API routing, the gate and drain boundary, STOP wire
compatibility, workspace/KV ordering, and the zero-new-remote actor regression.
These tests cannot establish NCCL progress, model accuracy or memory safety.

Hardware acceptance still requires:

1. Static AFD and native EEP baselines on the chosen model/runtime.
2. Initial 2A1F construction without premature forward; same-topology STOP and
   restart; then the full `2A1F → 4A1F → 4A2F → 2A2F → 2A1F` chain.
3. Per-engine counters proving new A receive requests, and F expert/weight
   coverage checks (actor counts alone are insufficient).
4. Fixed request/accuracy comparison against each topology's static baseline,
   plus memory/placement/process-group cleanup measurements.
5. CUDA graph recapture after eager passes. NPU N0 and implementation are
   separate remaining work, not covered by GPU results.

The compatibility patches record their pinned source and AFD differences.
Source review closed two concrete bugs: workspace reset on A runner reload,
and upstream `[-0:]` accidentally waiting on old remote actors. Upgrade work
must recopy upstream seams and reapply marked changes; this build does not
implement newer `prepare_elastic_ep`/`commit_elastic_ep` APIs.
