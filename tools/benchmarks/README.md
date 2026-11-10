# Decode-only benchmark

This directory contains the decode-only benchmark, which measures decode
throughput and latency at a chosen input length without prefill work.

- `decode_bench.py` implements the server-side KV connector,
  `AFDDecodeBenchConnector`.
- `request_generator.sh` wraps `vllm bench serve` to drive load against the
  server.

## How it works

The connector emulates the decode side of prefill-decode disaggregation with
the prefill instance faked out. For every new request, the scheduler side
reports every prompt token except the last as externally computed, so the
request prefills one token and goes straight to decode at a context of the
full prompt length. No prefill instance, KV producer or proxy is needed.

The connector writes no KV values: the blocks allocated for the "external"
tokens keep whatever the KV cache held before. Throughput and latency are
meaningful; generated text is not. A preempted request that restarts from
scratch is reported as external again.

Attach the connector to Attention only; the FFN needs no
`--kv-transfer-config`. The connector lives in `tools/benchmarks`, which is
not in the wheel, so it is loaded through `kv_connector_module_path` and needs
the repository root on `PYTHONPATH`.

## Running with an AFD recipe

The `DeepEPM2NAFDConnector` recipes, and the `P2pNcclAFDConnector` entry
points that reuse them, attach the connector to Attention when `DECODE_BENCH`
holds its `kv_connector_extra_config` as JSON (`{}` is enough) and add the
repository root (`AFD_REPO`, default the current directory) to `PYTHONPATH`;
see the
[recipes README](../../recipe/gpu/DeepEPM2NAFDConnector/README.md#decode-only-benchmark).
Every layout works, including multi-pod ones. From the repository root:

```bash
export MODEL_PATH=/path/to/Qwen3-30B-A3B
DECODE_BENCH='{}' \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh

# once Attention answers on port 18000:
INPUT_LEN=1024 OUTPUT_LEN=256 NUM_PROMPTS=2000 MAX_CONCURRENCY=64 \
    REQUEST_RATE=inf RESULT_DIR=/tmp/results/decode_bench \
    tools/benchmarks/request_generator.sh
```

Use the Attention port, not the FFN port. `INPUT_LEN` plus `OUTPUT_LEN` must
fit the recipe's `--max-model-len`.

## Running with vanilla vLLM

The connector is not AFD-specific. For a single vanilla vLLM process, pass
the same `--kv-transfer-config`:

```bash
export PYTHONPATH=$PWD
vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"AFDDecodeBenchConnector","kv_connector_module_path":"tools.benchmarks.decode_bench","kv_role":"kv_both","kv_connector_extra_config":{}}'
```

and run `request_generator.sh` with `PORT=8000`.

## Checking that a run was decode-only

The server's `/metrics` split prompt tokens by source. Over a decode-only
run, `vllm:prompt_tokens_by_source{source="local_compute"}` grows by exactly
one per finished request (`vllm:request_success`). The rest of each prompt is
counted under `source="external_kv_transfer"`, or `source="local_cache_hit"`
for the few prompt blocks the prefix cache already holds.

## Limitations

- KV contents are uninitialized, so attention reads arbitrary values and the
  generated text is meaningless.
- `kv_connector_extra_config` accepts `fill_mean` and `fill_std`, but the
  worker never writes them into the KV cache (`start_load_kv` is a no-op).
- The connector's name includes `AFD`, but it does not require the AFD plugin;
  it uses vLLM's standard KV-transfer interface.
