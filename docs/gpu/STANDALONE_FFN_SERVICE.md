# Standalone GPU FFN service

Run a registered CUDA routed-experts model in independent F-only processes
with `torchrun -m afd_plugin.ffn_faas serve --backend gpu-deepep-m2n`. The
service loads only the routed experts, reuses vLLM's native loader and the
existing DeepEP M2N FFN runtime, and constructs no vLLM Worker, upstream
ModelRunner, EngineCore, scheduler, KV cache or complete CausalLM. Attention
keeps its existing `vllm serve` launch, model, connector and DBO path.

## Supported configuration

| Supported | Rejected at startup |
| --- | --- |
| A registered CUDA routed-experts model; currently `Qwen3MoeForCausalLM`, BF16, every layer MoE, no shared experts | Unregistered models, FP8 or any `quantization_config`, dense or partial-MoE Qwen3, shared experts |
| `DeepEPM2NAFDConnector` with `payload_format: bf16` | P2P, afdEP-direct, flexible placement, FP8 payload |
| F TP=1, EP over all F ranks, linear expert placement | EPLB, elastic EP, redundant experts, PP/CP, LoRA, speculative decoding |
| Eager, FULL_DECODE_ONLY CUDA graphs, DBO with two lanes (eager and graph) | Other graph modes, more than two micro-batches |
| SM90 or newer (H100, H200, GB200) | Pre-Hopper GPUs |

The routed experts must divide evenly across the F ranks.

## Architecture

```text
CPU service / host runner -> FFNBackend
  DeepEPM2NFFNBackend (ffn_faas/platforms/gpu/backend.py)
    GPUFFNEnvironment: GPU, F-only NCCL world and vLLM EP group
    AFD routed-experts model selected by the model registry
    GPUFFNRoutedExpertsModelRunner: the Worker path's FFN runner
      DeepEPM2NAFDConnector: handshake, control plane, lanes, experts
```

Two worlds per deployment: the `torchrun` F-only world (vLLM's world, DP and
EP groups, used only to build FusedMoE and place experts) and the AFD M2N
world `[F0 .. Fn-1, A0 .. Am-1]` the connector creates at `--host:--port`.
Each F step runs `GPUFFNModelRunner.execute_connector_driven_step()`, the same
method the Worker-path `AFDFFNWorker` loop calls, so the data path, buckets,
lanes, graphs and DBO order are identical to a `vllm serve` FFN. See the
[FFN runtime design](../design/module/ffn_runtime.md#runtime-selection).

Compared with a `vllm serve` FFN, F does not allocate the embedding, final
norm and LM head (about 1.16 GiB per rank for Qwen3-30B-A3B).

## Launch

Keep the Attention half of an existing recipe under
`recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/` unchanged and replace only
the FFN half. Example: 2A2F, graph and DBO on one node, Attention on GPUs 0
and 1, F on GPUs 2 and 3:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
  --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel \
  --additional-config '{"afd":{"role":"attention","connector":"DeepEPM2NAFDConnector",
      "host":"127.0.0.1","port":6239,"num_attention_ranks":2,"num_ffn_ranks":2,
      "compute_gate_on_attention":true,"afd_process_group_timeout_s":600}}' \
  --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096 \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,32,64]}' \
  --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8 \
  --host 127.0.0.1 --port 18000 &

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29500 \
  -m afd_plugin.ffn_faas serve "$MODEL_PATH" --backend gpu-deepep-m2n \
  --host 127.0.0.1 --port 6239 \
  --attention-ranks 2 --ffn-ranks 2 \
  --graph --dbo --afd-timeout 600 \
  --connector-extra-config '{}' \
  --status-port 29700
```

Three ports: the `torchrun` master port (F-only world), the AFD rendezvous
`--port` (equal to Attention's `afd.port`) and the management port base.

| Option | Meaning |
| --- | --- |
| `--host`, `--port` | AFD rendezvous; equal to Attention's `afd.host`/`afd.port` |
| `--attention-ranks`, `--ffn-ranks` | equal to Attention's AFD config; `WORLD_SIZE` must equal `--ffn-ranks` |
| `--graph` | FULL_DECODE_ONLY CUDA graphs on F (default eager) |
| `--dbo` | two DeepEP lanes; required exactly when Attention runs `--enable-dbo` |
| `--connector-extra-config` | `DeepEPM2NAFDConnector` options (JSON object) |
| `--afd-timeout` | `afd_process_group_timeout_s` for the M2N rendezvous (default 120) |

F takes no Attention TP, batch size, model length or capture sizes: it adopts
Attention's TP size, dispatch capacity and bucket ladder in the M2N handshake,
and graph keys come from Attention's control payloads.

### Configuration parity

The handshake fails on both roles when they disagree:

| Must agree | Attention | F |
| --- | --- | --- |
| Rank counts, AFD host/port | `afd` config | `--attention-ranks`, `--ffn-ranks`, `--host`, `--port` |
| `payload_format`, `network`, `bucket_granularity`, `cpu_sync_min_tokens` | `connector_extra_config` | `--connector-extra-config` |
| `num_lanes` | `--enable-dbo` | `--dbo` (an explicit `num_lanes` must agree) |
| `capacity_tokens_per_rank` | derived or explicit | adopted, or explicit and equal |
| Hidden size, experts, top-k, BF16 | checkpoint | the same checkpoint files |

`expert_backend`, `expert_num_sms`, `num_sms`, `num_qps` and the DeepEP
timeouts are F-local tuning knobs.

### Startup ordering

Attention joins the M2N rendezvous at the end of its model load; F joins after
its expert load. Independently started services must pair within
`afd_process_group_timeout_s` on both roles, or both fail. Raise it on both
sides to cover the expected start skew (600 s above).

## Management and readiness

Each local F rank serves `--status-port + LOCAL_RANK` (`0` chooses a free
port; each rank prints its address at start).

- `/status`: phase, `loaded_experts`/`total_experts` (this rank's experts over
  all layers; vLLM's loader reports no per-expert progress, so both values are
  reported together after loading), `parameter_bytes`, `local_expert_first`/
  `local_expert_last`, `m2n_paired`, `attn_tp_size`,
  `capacity_tokens_per_rank`, `num_buckets`, `network_hybrid`,
  `compute_ep_world_size`, `num_lanes`, `graph_mode`, `dbo` and
  `completed_batches` (one per control payload, including warmup, capture
  and replay steps).
- `/ready`: paired with Attention and receiving. CUDA graphs are captured
  later, driven by Attention's own capture; gate client traffic on
  Attention's health endpoint, which comes up only after that.
- `/health`: host liveness.

## Shutdown

`POST /stop`, SIGINT and SIGTERM set a stop flag. An idle F cancels its
blocked control receive (the control plane aborts its own NCCL group), its
receive loop exits, and it releases CUDA graphs, DeepEP lanes, the F-only
groups and the experts, in that order.

Releasing a DeepEP lane is a barrier over every M2N rank, so F finishes its
release only when Attention also shuts down (within the DeepEP GPU timeout,
120 s by default). Recommended order:

1. drain client traffic;
2. `POST /stop` to every F rank;
3. stop Attention right away;
4. F exits 0 with `resources_released=true` in its status.

A step that is in flight when the stop arrives completes first; it needs
Attention to finish the same step. If Attention dies or stops without F,
F's pending control receive cannot complete: once
`afd_process_group_timeout_s` passes, PyTorch's NCCL watchdog aborts the F
ranks (SIGABRT, so no status file is written) and `torchrun` exits non-zero.
A failure F detects itself (for example a DeepEP timeout in a step) reports
`failed` with `resources_released=false` and exits non-zero. A restarted Attention needs a
new M2N world, so restart F with it. Hot restart, reconnect and graceful
in-flight drain are not supported.
