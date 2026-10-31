# Qwen3-235B-A22B-FP8 AFD Recipe

Launch scripts for Qwen3-235B-A22B-FP8 in a 4A2F prefill/decode colocation on
a single node with 6 GPUs, with vLLM `v0.32.0`. `4a2f_graph_dp4tp1.sh` also
takes a selectable AFD connector and an optional KV connector; with the
decode-bench KV connector it doubles as a decode-only benchmark over real KV,
exactly like the
[Qwen3-30B-A3B recipe](../qwen3_30b_a3b/README.md#decode-only-benchmark).

## Prerequisites

- 6 GPUs with about 140 GB each on one node (H200). `DeepEPM2NAFDConnector`
  needs SM90 or newer and the environment described in the
  [connector recipes README](../README.md#prerequisites);
  `P2pNcclAFDConnector` runs on any A/H-class GPU with enough memory.
- Qwen3-235B-A22B-FP8 weights on disk (block-128 FP8). The scripts default to
  `/path/model_weights/Qwen3-235B-A22B-FP8`; override with `MODEL_PATH=...`.
- Free ports `6239` (AFD rendezvous), `18000` (Attention API) and `18001`
  (FFN).

## Layout

`4A2F`: Attention DP4 x TP1 on GPUs 0-3 and FFN DP2 on GPUs 4,5. Each FFN rank
owns half of the 128 experts.

| GPUs       | Role      | Parallelism          | Port  |
|------------|-----------|----------------------|-------|
| 0, 1, 2, 3 | Attention | DP4 / TP1            | 18000 |
| 4, 5       | FFN       | DP2 / TP1, EP2       | 18001 |

```text
prefill_decode_colocation/
├── 4a2f_graph_dp4tp1.sh       # 256 sequences per Attention rank, FULL_DECODE_ONLY graph; every knob an env var
└── 4a2f_graph_dbo_dp4tp1.sh   # 512 sequences per Attention rank, graph + DBO
```

`ATTN_TP=4` runs `4a2f_graph_dp4tp1.sh`'s Attention as one DP1/TP4 engine on
the same four GPUs instead (`num_attention_ranks` stays 4); raise
`MAX_NUM_SEQS` and the capture sizes with it, since one engine then holds every
running request.

Both roles run `FULL_DECODE_ONLY` CUDA graphs (sizes 8 to 256 by default,
including 48, 112 and 192); prefill runs eager in chunks of
`MAX_NUM_BATCHED_TOKENS`. Send traffic to the Attention port.
`DeepEPM2NAFDConnector` routes on Attention and dispatches FP8 tokens
(`M2N_PAYLOAD_FORMAT=fp8_block128_fp32`); `P2pNcclAFDConnector` sends BF16
hidden states and routes on the FFN.

## Configuration notes

- `connector_extra_config` sets `payload_format: "fp8_block128_fp32"`. The
  FFN must load the block-128 FP8 checkpoint; the startup handshake rejects a
  mismatch between the payload format and the loaded expert weights.
- The FFN keeps vLLM's default `--moe-backend auto`, which the plugin pins to
  `triton` so the expert weights stay in the layout the connector reads.
- `afd_process_group_timeout_s` and `--cpu-distributed-timeout-seconds` are
  raised to 3600 s because loading the weights can take longer than the
  default 120 s join timeout, especially from cold storage, and the two roles
  may finish minutes apart.
- The capture sizes cover `--max-num-seqs`. The connector derives its dispatch
  buckets from them and compiles one transport kernel set per bucket at
  startup.
- `expert_num_sms` (default `0`, all SMs) and `num_sms` tune how SMs are split
  between DeepGEMM experts and the transport kernels; see the connector guide.
  Pass them through `M2N_EXTRA_CONFIG`, for example `M2N_EXTRA_CONFIG='"expert_num_sms": 130'`.

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
`attn.log` before sending traffic.

Every knob of `4a2f_graph_dp4tp1.sh` is an environment variable documented at
the top of the script, including DBO (`ENABLE_DBO=1`), `MAX_NUM_SEQS` per
Attention rank, graph capture sizes, devices, ports, `EXTRA_ARGS` for both
roles, and `ATTN_EXTRA_ARGS` / `FFN_EXTRA_ARGS` for one role. For example,
`ATTN_EXTRA_ARGS="--api-server-count 1 --cudagraph-metrics"` logs how many
Attention steps ran in each CUDA graph and how many ran eager (vLLM disables
the stats logger with more than one API server, and DP4 starts four).

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the
`P2pNcclAFDConnector` FFN: it hangs on its first step.

## Checking a deployment

```bash
curl -s http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

The answer should name Paris. An FFN log line `Using TRITON Fp8 MoE backend`
confirms the expected expert-weight layout.

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
    --port 18000 --dp-size 4 --pool-prompts 500 --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench \
    -- --num-prompts 3000 --random-output-len 256 \
       --max-concurrency 128 --request-rate 10
```

### Sizing the pool

Qwen3-235B holds 188 KiB of BF16 KV per token (94 layers, 4 KV heads of 128).
With the defaults, each Attention rank needs
`500 * ceil(1026 / 16) + 256 * ceil(257 / 16)` blocks, about 590K tokens or
106 GiB; an H200 Attention rank has room for about 640K tokens. Keep
`DECODE_BENCH_NUM_PROMPTS` at or above `MAX_NUM_SEQS`.

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
- `FULL_DECODE_ONLY` is the only CUDA graph mode.
