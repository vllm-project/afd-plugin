# Qwen3-30B-A3B AFD Recipe

One launch script for Qwen3-30B-A3B in a 2A2F prefill/decode colocation on a
single 4-GPU node, with a selectable AFD connector and an optional KV
connector. With the decode-bench KV connector it doubles as a decode-only
benchmark over real KV (see [Decode-only benchmark](#decode-only-benchmark)).

## Prerequisites

- 4 GPUs on one node. `DeepEPM2NAFDConnector` needs SM90 or newer (H100, H200,
  GB200) and DeepEP/DeepGEMM installed; `P2pNcclAFDConnector` runs on any
  A/H-class GPU.
- vLLM `v0.30.0` and the `afd-plugin` package in the same environment.
- Qwen3-30B-A3B weights. The script defaults to
  `/path/model_weights/Qwen3-30B-A3B`; override with `MODEL_PATH=...`.
- Free ports: `6269` for the AFD connector, `18305`/`18306` for the Attention
  and FFN HTTP servers.

## Topology

| GPUs | Role      | Parallelism          | Port  |
|------|-----------|----------------------|-------|
| 0, 1 | Attention | DP2 / TP1            | 18305 |
| 2, 3 | FFN       | DP2 / TP1, EP2       | 18306 |

Both roles run `FULL_DECODE_ONLY` CUDA graphs; prefill runs eager in chunks of
`MAX_NUM_BATCHED_TOKENS`. Send traffic to the Attention port.

## Running

Run from the repository root:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
# DeepEP M2N (default)
bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
# NCCL P2P: same script with AFD_CONNECTOR=P2pNcclAFDConnector, or its entry point
bash recipe/gpu/P2pNcclAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```

The script backgrounds both roles, logs to `attn.log` and `ffn.log` in the
current directory, and waits. Wait for `Application startup complete` in
`attn.log` before sending traffic.

Every knob is an environment variable documented at the top of the script:
connector (`AFD_CONNECTOR`), KV connector (`KV_CONNECTOR`, or a raw
`KV_TRANSFER_CONFIG` for any other connector), prefix caching, DBO,
`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, graph capture sizes, devices, ports,
and `EXTRA_ARGS` for anything else.

## Decode-only benchmark

`KV_CONNECTOR=decode_bench` attaches `AFDDecodeBenchConnector`
(`tools/benchmarks/decode_bench.py`) to Attention in its prompt-pool mode:

1. **Warmup.** Each Attention DP rank turns its first `DECODE_BENCH_NUM_PROMPTS`
   requests into a pool of seeded random prompts of `DECODE_BENCH_PROMPT_LEN`
   tokens (the same pool on every rank, whatever the requests carried) and
   prefills them for real, one output token each.
2. **Pinning.** When a warmup request finishes, the connector keeps its KV
   blocks allocated for the server's lifetime (it reports the blocks as still
   being sent and never completes the send). Pinned blocks cannot be evicted
   and stay in the prefix cache.
3. **Measured requests.** Every later request is remapped round-robin onto a
   pool prompt, whatever it asked for. The prefix cache covers every prompt
   token but the last (the prompt length is rounded to one token past a block
   boundary), so the request prefills one token and decodes over real KV, on
   whichever rank serves it.

Start the server, then drive it with the two-phase client:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
KV_CONNECTOR=decode_bench \
DECODE_BENCH_NUM_PROMPTS=1000 \
DECODE_BENCH_PROMPT_LEN=1025 \
DECODE_BENCH_MAX_OUTPUT_LEN=256 \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh

python tools/benchmarks/decode_bench_client.py \
    --port 18305 --dp-size 2 --num-prompts 1000 \
    --num-requests 4000 --output-len 256 --max-concurrency 64 \
    --result-file /tmp/results/decode_bench.json
```

The client warms every rank with `--num-prompts` requests pinned by the
`X-data-parallel-rank` header, then sends the measured requests through vLLM's
load balancer and reports TTFT, TPOT, ITL, and throughput. It also reports
`prefilled tokens/request` from the server's prefix-cache counters: `1.00`
means every measured request was decode-only. `--num-prompts` must match
`DECODE_BENCH_NUM_PROMPTS`, and `--output-len` must not exceed
`DECODE_BENCH_MAX_OUTPUT_LEN`.

### Sizing the pool

The pool must fit each Attention rank's KV cache next to the running requests.
At startup the connector checks, per rank,

```text
num_prompts * ceil((prompt_len + 1) / block_size)
  + max_num_seqs * ceil((max_output_len + 1) / block_size)  <=  free KV blocks
```

and fails with the largest `num_prompts` that fits. Qwen3-30B-A3B holds about
96 KiB of BF16 KV per token, so 1000 prompts of 1025 tokens need about 94 GiB
per rank: an H200 fits it, an H100 needs fewer or shorter prompts. Keep
`num_prompts` at or above `MAX_NUM_SEQS` (the connector warns otherwise), or
concurrent requests share physical KV blocks and attention reads look cheaper
than they are.

### Limitations

- Generated text is meaningless: prompts are random tokens, and the client's
  prompt is discarded.
- Measured requests decode at a context of `prompt_len` regardless of the
  client's prompt length.
- Pinned warmup requests keep each Attention engine loop stepping while idle
  (one busy CPU core per DP rank, no GPU work). Restart the server to free the
  pool.
- `DECODE_BENCH_NUM_PROMPTS=0` keeps the older dummy-KV mode, which writes no
  KV at all.
