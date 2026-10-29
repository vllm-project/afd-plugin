# Qwen3-235B-A22B-FP8 AFD Recipe

One launch script for Qwen3-235B-A22B-FP8 in a 4A2F prefill/decode colocation
on a single node with 6 GPUs, with a selectable AFD connector and an optional
KV connector. With the decode-bench KV connector it doubles as a decode-only
benchmark over real KV, exactly like the
[Qwen3-30B-A3B recipe](../qwen3_30b_a3b/README.md#decode-only-benchmark).

## Prerequisites

- 6 GPUs with about 140 GB each on one node (H200). `DeepEPM2NAFDConnector`
  needs SM90 or newer and DeepEP/DeepGEMM installed; `P2pNcclAFDConnector`
  runs on any A/H-class GPU with enough memory.
- vLLM `v0.30.0` and the `afd-plugin` package in the same environment.
- Qwen3-235B-A22B-FP8 weights (block-128 FP8). The script defaults to
  `/path/model_weights/Qwen3-235B-A22B-FP8`; override with `MODEL_PATH=...`.
- Free ports: `6269` for the AFD connector, `18305`/`18306` for the Attention
  and FFN HTTP servers.

## Topology

| GPUs       | Role      | Parallelism          | Port  |
|------------|-----------|----------------------|-------|
| 0, 1, 2, 3 | Attention | DP4 / TP1            | 18305 |
| 4, 5       | FFN       | DP2 / TP1, EP2       | 18306 |

`ATTN_TP=4` runs Attention as one DP1/TP4 engine on the same four GPUs instead
(`num_attention_ranks` stays 4); raise `MAX_NUM_SEQS` and the capture sizes
with it, since one engine then holds every running request.

Both roles run `FULL_DECODE_ONLY` CUDA graphs (sizes 8 to 128 by default);
prefill runs eager in chunks of `MAX_NUM_BATCHED_TOKENS`. Send traffic to the
Attention port. `DeepEPM2NAFDConnector` routes on Attention and dispatches FP8
tokens (`M2N_PAYLOAD_FORMAT=fp8_block128_fp32`); `P2pNcclAFDConnector` sends
BF16 hidden states and routes on the FFN.

## Running

Run from the repository root:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-235B-A22B-FP8
# DeepEP M2N (default)
bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh
# NCCL P2P: same script with AFD_CONNECTOR=P2pNcclAFDConnector, or its entry point
bash recipe/gpu/P2pNcclAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh
```

The script backgrounds both roles, logs to `attn.log` and `ffn.log` in the
current directory, and waits. Wait for `Application startup complete` in
`attn.log` before sending traffic. A cold weight load can take tens of minutes
and the two roles may finish minutes apart, so the recipe raises the AFD join
timeout (`AFD_PROCESS_GROUP_TIMEOUT_S`) and vLLM's CPU-group timeout to 2 h.

Every knob is an environment variable documented at the top of the script,
including DBO (`ENABLE_DBO=1`), `MAX_NUM_SEQS` per Attention rank, graph
capture sizes, devices, ports, `EXTRA_ARGS` for both roles, and
`ATTN_EXTRA_ARGS` / `FFN_EXTRA_ARGS` for one role. For example,
`ATTN_EXTRA_ARGS="--api-server-count 1 --cudagraph-metrics"` logs how many
Attention steps ran in each CUDA graph and how many ran eager (vLLM disables
the stats logger with more than one API server, and DP4 starts four).

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the
`P2pNcclAFDConnector` FFN: it hangs on its first step.

## Decode-only benchmark

`KV_CONNECTOR=decode_bench` attaches `AFDDecodeBenchConnector` to Attention;
the pool, pinning, and remapping work as described for
[Qwen3-30B-A3B](../qwen3_30b_a3b/README.md#decode-only-benchmark). Drive it
with `--dp-size 4`:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-235B-A22B-FP8
KV_CONNECTOR=decode_bench \
DECODE_BENCH_NUM_PROMPTS=500 \
DECODE_BENCH_PROMPT_LEN=1025 \
DECODE_BENCH_MAX_OUTPUT_LEN=256 \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/4a2f_graph_dp4tp1.sh

python tools/benchmarks/decode_bench_client.py \
    --port 18305 --dp-size 4 --pool-prompts 500 --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench \
    -- --num-prompts 3000 --random-output-len 256 \
       --max-concurrency 128 --request-rate 10
```

### Sizing the pool

Qwen3-235B holds 188 KiB of BF16 KV per token (94 layers, 4 KV heads of 128).
With the defaults, each Attention rank needs
`500 * ceil(1026 / 16) + 128 * ceil(257 / 16)` blocks, about 555K tokens or
97 GiB; an H200 Attention rank has room for about 640K tokens. Keep
`DECODE_BENCH_NUM_PROMPTS` at or above `MAX_NUM_SEQS`.
