# Decode-only benchmark

This directory contains the decode-only benchmark used to measure decode
throughput and latency over long, real KV contexts while avoiding prefill
work during the measured phase.

- `decode_bench.py` implements the server-side KV connector,
  `AFDDecodeBenchConnector`.
- `decode_bench_client.py` wraps `vllm bench serve`, performs warmup, and
  checks that the measured run was decode-only.

## How it works

The connector runs in its prompt-pool mode when `kv_connector_extra_config`
sets `num_prompts > 0`:

1. **Warmup.** Each Attention DP rank turns its first `num_prompts` requests
   into a pool of seeded random prompts of `prompt_len` tokens (the same pool
   on every rank, whatever the requests carried) and prefills them for real,
   one output token each.
2. **Pinning.** When a warmup request finishes, the connector takes a
   reference of its own on the request's full prompt blocks and holds it for
   the server's lifetime, while the request itself is freed as usual. Pinned
   blocks cannot be evicted and stay in the prefix cache.
3. **Measured requests.** Every later request is remapped round-robin onto a
   pinned pool prompt, whatever it asked for. The prefix cache covers every
   prompt token but the last (the prompt length is rounded to one token past a
   block boundary), so the request prefills one token and decodes over real
   KV, on whichever rank serves it. A request that arrives before any prompt
   is pinned may miss the cache; the connector logs it and the client's checks
   fail the run.

`kv_connector_extra_config` keys: `num_prompts` (pool size per Attention DP
rank, `0` keeps the dummy-KV mode below), `prompt_len` (default 1025), `seed`
(default 0) and `max_output_len` (the longest measured output; it sizes the
KV check and defaults to `max_model_len - prompt_len`).

## Running with an AFD recipe

The `DeepEPM2NAFDConnector` recipes attach the connector to Attention when
`DECODE_BENCH` holds its `kv_connector_extra_config`; see the
[recipes README](../../recipe/gpu/DeepEPM2NAFDConnector/README.md#decode-only-benchmark).
Every layout works, including multi-pod ones. From the repository root:

```bash
export MODEL_PATH=/path/to/Qwen3-30B-A3B
DECODE_BENCH='{"num_prompts": 500, "prompt_len": 1025, "max_output_len": 256}' \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh

python tools/benchmarks/decode_bench_client.py \
    --port 18000 --dp-size 2 --pool-prompts 500 --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench \
    -- --num-prompts 2000 --random-output-len 256 \
       --max-concurrency 64 --request-rate inf
```

Use the Attention port, not the FFN port. `--dp-size` is the Attention DP
size over all pods.

## Running with vanilla vLLM

The client is not AFD-specific. It works with any vLLM server that has the
connector attached. For a single vanilla vLLM process on four GPUs, start the
server with tensor parallelism 4 and the connector's `--kv-transfer-config`:

```bash
export PYTHONPATH=$PWD
export MODEL_PATH=/path/to/model

vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"num_prompts":500,"prompt_len":1025,"seed":0,"max_output_len":256}}'
```

Then run the client with `--dp-size 1`, because a vanilla TP4 server has one
logical engine:

```bash
python tools/benchmarks/decode_bench_client.py \
    --port 8000 --dp-size 1 --pool-prompts 500 --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench_vanilla \
    -- --num-prompts 3000 --random-output-len 256 \
       --max-concurrency 256 --request-rate inf
```

## Client phases

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

## Client options

- `--dp-size` is the number of Attention data-parallel ranks.
- `--pool-prompts` must match the server's `num_prompts`, and `--prompt-len`
  its `prompt_len`; `--random-output-len` must not exceed `max_output_len`.
- `--result-dir` receives the warmup and measured benchmark JSON.
- `--skip-warmup` measures only, against an already warmed pool.
- Every argument after `--` is passed unchanged to `vllm bench serve`, for
  example `--num-prompts`, `--random-output-len`, `--max-concurrency`,
  `--request-rate`, `--percentile-metrics`, `--metric-percentiles` and
  `--save-detailed`.

The client needs vLLM and the model's tokenizer installed, as vllm bench does.

## Output files

Each run writes `warmup-rank<N>.json` and `warmup-rank<N>.log`,
`measured.json`, and `decode_bench_report.json`, which holds the client's
checks, including whether every measured request prefilled exactly one token
and whether any requests were preempted.

## Pool sizing

The pool must fit each Attention rank's KV cache next to the running
requests. At startup the connector checks, per rank,

```text
num_prompts * ceil((prompt_len + 1) / block_size)
  + max_num_seqs * ceil((max_output_len + 1) / block_size)  <=  free KV blocks
```

and fails with the largest `num_prompts` that fits. Keep `num_prompts` at or
above `--max-num-seqs` (the connector warns otherwise), or concurrent requests
share physical KV blocks and attention reads look cheaper than they are. The
recipe READMEs give each model's KV size per token.

## Limitations

- Generated text is meaningless: prompts are random tokens, and the client's
  prompt is discarded.
- Measured requests decode at a context of `prompt_len` regardless of the
  client's prompt length.
- The pool stays pinned for the server's lifetime; restart the server to free
  it.
- `num_prompts: 0` keeps the older dummy-KV mode, which writes no KV at all:
  every prompt token but the last is reported as externally computed.
- The connector's name includes `AF`, but it does not require the AFD plugin;
  it uses vLLM's standard KV-transfer interface.
