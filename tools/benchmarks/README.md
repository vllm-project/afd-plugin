# Decode-only benchmark

This directory contains the decode-only benchmark used to measure decode
throughput and latency over long, real KV contexts while avoiding large
prefill work during the measured phase.

- `decode_bench.py` implements the server-side KV connector.
- `decode_bench_client.py` wraps `vllm bench serve`, performs warmup, and writes
  the benchmark checks.

## How it works

The benchmark starts by bringing up a vLLM serving instance using one of the
recipes under `recipe/gpu/`. The recipe attaches the decode-bench KV connector
to the Attention instance and starts the server.

After the server is ready, the benchmark has two phases:

1. **Warmup.** The server-side connector creates a pool of seeded prompts of
   `--prompt-len` tokens and prefills them once. Their KV blocks are pinned in
   the prefix cache.
2. **Measured.** The client sends requests that the connector remaps onto the
   pool prompts. Each measured request prefills only its last prompt token and
   then decodes for the requested number of output tokens.

This makes the benchmark decode-only while still exercising real KV-cache
attention over the full prompt length.

## Running with an AFD recipe

The AFD recipes already support the connector through environment variables.
For example, to run the Qwen3-235B-A22B-FP8 4A2F recipe:

```bash
export MODEL_PATH=/path/to/Qwen3-235B-A22B-FP8

KV_CONNECTOR=decode_bench \
DECODE_BENCH_NUM_PROMPTS=500 \
DECODE_BENCH_PROMPT_LEN=1025 \
DECODE_BENCH_MAX_OUTPUT_LEN=256 \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh
```

Then run the client from the repository root:

```bash
python tools/benchmarks/decode_bench_client.py \
    --port 18000 \
    --dp-size 4 \
    --pool-prompts 500 \
    --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench \
    -- --num-prompts 3000 \
       --random-output-len 256 \
       --max-concurrency 128 \
       --request-rate inf
```

Use the Attention port, not the FFN port.

## Running with vanilla vLLM

The client is not AFD-specific. It works with any vLLM server that has the
decode-bench KV connector attached.

For a single vanilla vLLM process on six GPUs, start the server with tensor
parallelism 6 and the connector’s `--kv-transfer-config`:

```bash
export PYTHONPATH=$PWD
export MODEL_PATH=/path/to/model

vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 6 \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{"num_prompts":500,"prompt_len":1025,"seed":0,"max_output_len":256}}'
```

Then run the benchmark client with `--dp-size 1`, because a vanilla TP6 server
has one logical engine:

```bash
python tools/benchmarks/decode_bench_client.py \
    --port 8000 \
    --dp-size 1 \
    --pool-prompts 500 \
    --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench_vanilla \
    -- --num-prompts 3000 \
       --random-output-len 256 \
       --max-concurrency 256 \
       --request-rate inf
```

To repeat the run at concurrency 512, rerun the same command with
`--max-concurrency 512` and a different `--result-dir`.

## Pool sizing

The connector checks that the prompt pool and running requests fit in the KV
cache:

```text
num_prompts * ceil((prompt_len + 1) / block_size)
  + max_num_seqs * ceil((max_output_len + 1) / block_size)
  <= free KV blocks
```

Increase `num_prompts` only if the server still has enough free KV blocks after
reserving space for the configured `max_num_seqs` and `max_output_len`.

## Client options

- `--dp-size` is the number of Attention data-parallel ranks. Use `4` for the
  AFD 4A2F recipe and `1` for a single vanilla vLLM engine.
- `--pool-prompts` must match the server’s `num_prompts`.
- `--prompt-len` must match the server’s `prompt_len`.
- `--result-dir` receives the warmup and measured benchmark JSON.
- `--skip-warmup` measures only, against an already warmed pool.
- Every argument after `--` is passed unchanged to `vllm bench serve`.

Useful `vllm bench serve` arguments include:

- `--num-prompts`
- `--random-output-len`
- `--max-concurrency`
- `--request-rate`
- `--percentile-metrics`
- `--metric-percentiles`
- `--save-detailed`

## Output files

Each run writes:

- `warmup-rank<N>.json` and `warmup-rank<N>.log`
- `measured.json`
- `decode_bench_report.json`

`decode_bench_report.json` contains the connector’s checks, including whether
every measured request prefilled exactly one token and whether any requests
were preempted.

## Notes

- The generated text is meaningless because the prompts are random.
- Restart the server to release the pinned pool.
- The connector’s name includes `AF`, but it does not require the AFD plugin;
  it uses vLLM’s standard KV-transfer interface.
