# Elastic AFD on Ascend NPU — recipe and validation record

Status: the resize ladder below was validated on a single-machine 16x
Ascend 910 cluster with DeepSeek-V2-Lite (W8A8) on the vLLM v0.26 /
MRV1 stack (40 soak cycles, 160 resizes, no leaks; four rungs under
8-way concurrent load with drain/503 semantics). The elastic code was
since ported to this repository's current target (vLLM v0.30.0,
ModelRunnerV2, building on the elastic-EP platform groundwork
submitted to vllm-ascend): CPU unit suites pass, while a hardware
revalidation on the 0.30 stack is pending. Evidence below predates the port unless
marked otherwise.

This document covers the NPU-specific behavior of the AFD elastic resize
(`/scale_elastic_ep`). The orchestration flow, API semantics, and drain
boundary are shared with the GPU build and are documented in
[ELASTIC_AFD.md](../ELASTIC_AFD.md).

## Launch

```bash
# 1. Ray head with NPU resource detection
ray start --head ...  # ASCEND_RT_VISIBLE_DEVICES pins the visible chips

# 2. Attention-role API server (example: 2A2F eager)
vllm serve /path/to/DeepSeek-V2-Lite-W8A8 \
  --worker-cls afd_plugin.v1.worker.npu.AFDNPUAttentionWorker \
  --enable-expert-parallel --enable-elastic-ep \
  --data-parallel-backend ray --distributed-executor-backend uni \
  --data-parallel-size 2 --tensor-parallel-size 1 \
  --enforce-eager --max-model-len 4096 --max-num-seqs 16 \
  --max-num-batched-tokens 16 --gpu-memory-utilization 0.2 \
  --additional-config '{"afd": {"role": "attention",
      "connector": "CAMP2pAFDConnector", "host": "127.0.0.1",
      "port": 6241, "num_attention_ranks": 2, "num_ffn_ranks": 2,
      "compute_gate_on_attention": false},
    "afd_elastic": {"ffn_tensor_parallel_size": 1}}'
```

## Resize

```bash
curl -X POST http://127.0.0.1:8000/scale_elastic_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size": 4, "role": "attention", "drain_timeout": 120}'
curl -X POST http://127.0.0.1:8000/scale_elastic_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size": 1, "role": "ffn", "drain_timeout": 120}'
```

## NPU-specific behavior and fixes

### CAMP2p uniform row-count protocol

The a2e/e2a kernels move `batch_size / (attn_size / ffn_size)` rows to or
from **every** Attention rank's shared-memory window; they cannot carry
unequal per-rank counts. Two rules follow, and both are enforced in Python:

1. `send_attn_output` pads every Attention rank up to the largest token
   count across all Attention ranks in the step (extra rows are dropped on
   the return path). Decode steps are usually already uniform, so padding
   cost lands on unbalanced prefill batches only.
2. The FFN side sizes everything to the same published count: the a2e/e2a
   `batch_size`, the MC2 `xActiveMask` (its dim0 must equal `x` dim0, or
   `npu_moe_distribute_dispatch_v2` fails with error 561002), and the ACL
   graph keys (`make_ffn_graph_key(..., uniform_padding=True)`), so a
   replayed FULL_DECODE_ONLY graph was captured for the batch shape the
   current step produces. Under-estimating here also makes the kernel write
   past the end of `expand_x`.

The GPU connector keeps its contiguous-group mapping; the shared graph-key
helper defaults to it and the NPU runner opts into the uniform semantics.

### Fresh FFN actors on every NPU resize

A kept FFN actor process cannot re-establish its HCCL communicators across
a world-size change: the reused rank's `hcclCommInitRootInfoConfig`
rootInfo rendezvous never meets the fresh rank's and stalls to its 180 s
timeout. `FFNActorManager.resize_actors` therefore replaces the whole FFN
actor set with fresh processes on any NPU size change (weight reload
≈15 s per rank). Attention engines are unaffected: fresh connector
generations get fresh HCCL IPC windows, and the per-launch magic counters
reset with them.

### HCCL group names are unique per connector generation

Every rebuild appends the connector port to the HCCL group names, so a
retired generation's groups never collide with the new one's. Combined
with `release_roles` before removal, this prevents HBM leaks across
repeated resizes.

## Validation record (2026-10-22, eager, DSV2Lite 2A2F)

Baseline completions at `temperature=0`, then resize, then the same
probes. Post-resize outputs were **byte-identical to baseline** at every
rung:

| Rung | Result |
| --- | --- |
| A 2→4 | HTTP 200, outputs identical, coherent |
| A 4→2 | HTTP 200, outputs identical, coherent |
| F 2→1 | HTTP 200, outputs identical, coherent |
| F 1→2 | HTTP 200, outputs identical, coherent |

Root causes found and fixed on the way (all three reproduce without the
fixes and disappear with them):

1. **Row misplacement after resize** — kernels even-split per source; the
   Python side passed raw sums. Symptom: confidently-wrong completions
   that differed across identical requests. Fixed by the uniform
   publication protocol above.
2. **F 1→2 hang** — reused FFN actor's HCCL rendezvous never completed
   ("No rank in the communicator can connect to the root node"). Fixed by
   the fresh-actors policy.
3. **A 2→4 MoE dispatch failure** — `xActiveMask dim0 != x dim0`
   (`aclnnMoeDistributeDispatchV4`, error 561002): the mask was sized to
   real tokens while `x` carried uniform-padded rows; sibling ranks also
   under-allocated `expand_x` and hung in vector-core spin-wait. Fixed by
   sizing the FFN runner's token counts (mask + graph keys) to the
   published uniform rows.

## Remaining work

- FULL_DECODE_ONLY graph-mode restart + resize cycle on NPU (the
  uniform-padding graph keys are in place and unit-tested, but the
  hardware cycle is not yet run).
- Eager DBO / DBO+graph rungs.
- Concurrent traffic during resize, repeated resize soak, and
  memory/communicator leak checks across many generations.
- Mixed-role chain `2A1F → 4A1F → 4A2F → 2A2F → 2A1F`.
