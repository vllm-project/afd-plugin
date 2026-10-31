# Qwen3-30B-A3B AFD Recipe

Launch scripts for Qwen3-30B-A3B (BF16) in a 2A2F prefill/decode colocation on
a single 4-GPU node with vLLM `v0.32.0`. They use the same layouts as the Qwen3
MoE DeepEP M2N E2E cases (`afd-deepep-m2n-*-2a2f` in
[`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios)).
`2a2f_graph_dp2tp1.sh` also takes a selectable AFD connector and an optional KV
connector; with the decode-bench KV connector it doubles as a decode-only
benchmark over real KV (see [Decode-only benchmark](#decode-only-benchmark)).

## Prerequisites

- 4 GPUs on one node. `DeepEPM2NAFDConnector` needs SM90 or newer (H100, H200,
  GB200) and the environment described in the
  [connector recipes README](../README.md#prerequisites);
  `P2pNcclAFDConnector` runs on any A/H-class GPU.
- Qwen3-30B-A3B weights on disk. The scripts default to
  `/path/model_weights/Qwen3-30B-A3B`; override with `MODEL_PATH=...`.
- Free ports `6239` (AFD rendezvous), `18000` (Attention API) and `18001`
  (FFN).

## Layout

`2A2F`: Attention on GPUs 0,1 and FFN DP2 on GPUs 2,3. FFN rank 0 owns experts
0-63 and FFN rank 1 owns experts 64-127.

| GPUs | Role      | Parallelism          | Port  |
|------|-----------|----------------------|-------|
| 0, 1 | Attention | DP2 / TP1            | 18000 |
| 2, 3 | FFN       | DP2 / TP1, EP2       | 18001 |

```text
prefill_decode_colocation/
├── 2a2f_eager_dp2tp1.sh       # Attention DP2 x TP1, eager
├── 2a2f_graph_dp2tp1.sh       # Attention DP2 x TP1, FULL_DECODE_ONLY graph; every knob an env var
├── 2a2f_graph_dbo_dp2tp1.sh   # Attention DP2 x TP1, graph + DBO
└── 2a2f_graph_dp1tp2.sh       # Attention DP1 x TP2, FULL_DECODE_ONLY graph
```

The graph scripts run `FULL_DECODE_ONLY` CUDA graphs; prefill runs eager in
chunks of `--max-num-batched-tokens`. Send traffic to the Attention port.

## Configuration notes

- Both roles set `compute_gate_on_attention: true` for `DeepEPM2NAFDConnector`
  (`false` for `P2pNcclAFDConnector`), use the same rank counts, and run with
  `--enable-expert-parallel`. The FFN runs TP 1.
- `connector_extra_config` is left empty, so the payload is BF16 and the FFN
  runs DeepGEMM experts.
- The graph scripts capture batch sizes 8, 16, 32 and 64. The connector
  derives its dispatch buckets from the capture sizes and compiles one
  transport kernel set per bucket at startup.
- In the DBO script, both roles pass `--enable-dbo` with the same thresholds;
  the connector then uses two lanes. `2a2f_graph_dp2tp1.sh` does the same with
  `ENABLE_DBO=1`.
- The dispatch capacity is derived from `--max-num-batched-tokens` and the
  largest capture size, so lowering `--max-num-batched-tokens` shrinks the
  transport buffers.

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

Every knob of `2a2f_graph_dp2tp1.sh` is an environment variable documented at
the top of the script: connector (`AFD_CONNECTOR`), KV connector
(`KV_CONNECTOR`, or a raw `KV_TRANSFER_CONFIG` for any other connector), prefix
caching, DBO, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, graph capture sizes,
devices, ports, and `EXTRA_ARGS` for anything else.

## Checking a deployment

```bash
curl -s http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

For an accuracy check, run the matching E2E case, for example
`afd-deepep-m2n-graph-2a2f`; see [`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios).

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

Start the server, then drive it with `tools/benchmarks/decode_bench_client.py`,
which runs every phase with `vllm bench serve`:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
KV_CONNECTOR=decode_bench \
DECODE_BENCH_NUM_PROMPTS=1000 \
DECODE_BENCH_PROMPT_LEN=1025 \
DECODE_BENCH_MAX_OUTPUT_LEN=256 \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh

python tools/benchmarks/decode_bench_client.py \
    --port 18000 --dp-size 2 --pool-prompts 1000 --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench \
    -- --num-prompts 4000 --random-output-len 256 \
       --max-concurrency 64 --request-rate 20
```

1. **Warmup.** One `vllm bench serve` run per Attention rank, pinned with the
   `X-data-parallel-rank` header, sends `--pool-prompts` requests to that rank.
   `--warmup-concurrency` sets their concurrency; `--warmup-bench-args` (last
   before `--`) passes anything else to these runs.
2. **Pre-check.** From the server's prefix-cache counters, every warmup
   request must have looked up `--prompt-len` prompt tokens, so the pool
   prompts are as long as the measured prompts.
3. **Measured run.** `vllm bench serve --dataset-name random --random-input-len
   <prompt-len>` plus every argument after `--`: request rate, concurrency,
   number of requests, output length, burstiness, percentiles, and so on. The
   client sets the dataset, input length, endpoint, and result file itself and
   rejects them after `--`.
4. **Post-check.** Over the measured run the server must have prefilled one
   token per finished request (decode only), with no failed requests and no
   preemptions. Otherwise the client exits non-zero.

`--result-dir` receives vllm bench's own result JSON for each run
(`warmup-rank<N>.json`, `measured.json`) plus `decode_bench_report.json` with
the checks. `--pool-prompts` must match `DECODE_BENCH_NUM_PROMPTS`,
`--prompt-len` must match `DECODE_BENCH_PROMPT_LEN`, and `--random-output-len`
must not exceed `DECODE_BENCH_MAX_OUTPUT_LEN`. The client needs vLLM and the
model's tokenizer installed, as vllm bench does.

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

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
- `FULL_DECODE_ONLY` is the only CUDA graph mode.
