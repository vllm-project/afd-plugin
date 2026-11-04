# DeepEPM2NAFDConnector Recipes

Launch scripts for running Attention-FFN Disaggregation (AFD) with
`DeepEPM2NAFDConnector`, the expert-routed M2N connector for Hopper (SM90) and
newer GPUs, with vLLM `v0.32.0`. See
[`docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md`](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md)
for installation, the configuration contract, the required vLLM settings, and
the placements the connector accepts.

This README covers what every recipe here shares; each model's `README.md`
lists only its defaults and model-specific notes.

## Recipes

Every model directory holds the same three scripts in
`prefill_decode_colocation/`:

| Script | Mode |
| --- | --- |
| `graph.sh` | `FULL_DECODE_ONLY` CUDA graphs; prefill runs eager |
| `eager.sh` | Eager (`--enforce-eager`) |
| `dbo.sh` | `FULL_DECODE_ONLY` CUDA graphs and dual-batch overlap on both roles (`--enable-dbo`, decode/prefill token thresholds 2 and 8) |

```text
.
├── deepseek_v2_lite/       # BF16 payload
├── qwen3_30b_a3b/          # BF16 payload
└── qwen3_235b_a22b_fp8/    # FP8 payload (fp8_block128_fp32)
```

Each script is self-contained, so it can be copied or mounted on its own (the
`deploy-afd-k8s` skill mounts the single file). Above the line
`The rest is the same in every DeepEPM2NAFDConnector recipe` a script sets its
model path, layout default, tuning flags (`TUNING`), connector extras and
mode flags (`MODE`); everything below that line is identical in all of them.

## Prerequisites

- SM90 or newer GPUs. The default `2A2F` layout uses 4 GPUs on one host; see
  the model README for memory needs.
- vLLM `v0.32.0`, the plugin, afdEP-comm (the `afdep` extra) and DeepGEMM
  installed in one environment, with the elfutils headers available to
  DeepGEMM's JIT. See
  [Installation](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#installation).
- When afdEP-comm was built against the `nvidia-nccl` wheel, export its
  location before running a script:

  ```bash
  NCCL=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
  export AFDEP_NCCL_ROOT=$NCCL LD_LIBRARY_PATH=$NCCL/lib:${LD_LIBRARY_PATH:-}
  ```

- Free ports `6239` (AFD rendezvous), `18000` (Attention API) and `18001`
  (FFN, serves no requests); across pods also `13345` (Attention DP RPC) and
  `13355` (FFN DP RPC).

The scripts call `vllm serve` directly rather than through `uv run`, so they
use the environment that holds afdEP-comm and DeepGEMM.

## Layout

`LAYOUT` lists the pods in order, one `<a>A<f>F` entry each: `a` Attention
ranks and `f` FFN ranks. Inside a pod, the FFN ranks take the first `f` GPUs
and the Attention ranks the next `a` GPUs; every pod runs the same script with
the same `LAYOUT` and its own `POD=AFD_<k>` (`k` counts from 0; one pod needs
no `POD`).

| `LAYOUT` | Pods | Per pod |
| --- | --- | --- |
| `2A2F` (default) | 1 | FFN DP2 on GPUs 0,1, Attention DP2 on GPUs 2,3 |
| `4A2F` | 1 | FFN on GPUs 0,1, Attention DP4 on GPUs 2-5 |
| `2A0F,0A2F` | 2 | pod 0: Attention DP2; pod 1: FFN DP2 |
| `1A1F,1A1F` | 2 | FFN on GPU 0, Attention on GPU 1 |
| `4A1F,4A1F` | 2 | FFN on GPU 0, Attention on GPUs 1-4 |

`ATTN_TP` splits each pod's Attention GPUs into TP groups (it must divide each
pod's `a`), so `ATTN_TP=2` runs `2A2F`'s Attention as one DP1 x TP2 engine;
`num_attention_ranks` still counts GPUs. The FFN always runs TP 1, and its DP
rank (its EP rank) decides which experts it loads.

The script derives every DP flag from `LAYOUT` and `POD`. A role held by one
pod gets only `--data-parallel-size`; a role spread over pods is led by its
first pod, and the others start headless from their start rank. The
connector checks the placement at startup and names the pods' shapes when it
rejects one; the
[supported layouts](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#supported-layouts)
table lists which placements it accepts (for example `4A0F,0A2F` is not, its
pods differ in size).

## Recipe settings

Every setting is an environment variable; the defaults reproduce the
published recipe.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `/path/model_weights/<model>` | Weights directory or HF id. |
| `LAYOUT` | `2A2F` | Pods and their ranks; see [Layout](#layout). |
| `POD` | `AFD_0` | This pod's entry in `LAYOUT`, `AFD_<k>`. |
| `ATTN_TP` | `1` | Attention TP size. |
| `CUDAGRAPH_CAPTURE_SIZES` | per model | Ascending decode graph sizes (`graph.sh`, `dbo.sh`); the largest is the maximum capture size. |
| `VLLM_ARGS` | empty | Extra `vllm serve` flags for both roles. |
| `ATTN_ARGS` / `FFN_ARGS` | empty | Extra `vllm serve` flags for one role. |
| `DECODE_BENCH` | empty | Turns on the decode-only benchmark; see [Decode-only benchmark](#decode-only-benchmark). |
| `AFD_REPO` | current directory | Repository root that holds `tools/benchmarks` (with `DECODE_BENCH`). |
| `M2N_EXTRA_CONFIG` | empty | Extra `connector_extra_config` members, for example `'"expert_num_sms": 130'`. |
| `HEAD_HOST` | `127.0.0.1` on one pod | Address of pod 0 when it leads both roles (multi-pod, no Kubernetes). |
| `ATTN_HEAD_HOST` / `FFN_HEAD_HOST` | `HEAD_HOST` | Address of the first pod with Attention / FFN ranks, when they differ. The FFN head also hosts the AFD rendezvous. |
| `AFD_CONNECTOR` | `DeepEPM2NAFDConnector` | `P2pNcclAFDConnector` runs the same layout on NCCL P2P (used by the P2P recipe entry points). |

`VLLM_ARGS`, `ATTN_ARGS`, `FFN_ARGS` and any arguments after the script name
come after the recipe's own flags, so a repeated flag overrides its default
(argparse keeps the last value). They are split on whitespace without shell
quoting, so pass JSON values without spaces. For example:

```bash
# 512 sequences per Attention rank, no prefix caching on Attention
VLLM_ARGS="--max-num-seqs 512" ATTN_ARGS="--no-enable-prefix-caching" \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh
# Count graph and eager Attention steps (vLLM disables the stats logger with
# more than one API server)
ATTN_ARGS="--api-server-count 1 --cudagraph-metrics" bash .../graph.sh
```

Raise `CUDAGRAPH_CAPTURE_SIZES` together with `--max-num-seqs`, so decode
batches stay in a captured size.

## Running on one host

Run a script from the repository root:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh
```

The script starts the FFN server and then the Attention server in the
background, writes `ffn.log` and `attn.log` into the current directory, and
waits. Wait for `Application startup complete` in `attn.log`, then send
traffic to the Attention server on port `18000`. The first start on a host
compiles transport and DeepGEMM kernels, so it is slower than later starts.

## Running on several pods

Start the same script on every pod with the same settings and the pod's own
`POD`. Without Kubernetes, point the pods at the leaders' addresses:

```bash
# 2 x 1A1F: pod 0 leads both roles
LAYOUT=1A1F,1A1F POD=AFD_0 HEAD_HOST=<pod 0 address> bash .../graph.sh   # pod 0
LAYOUT=1A1F,1A1F POD=AFD_1 HEAD_HOST=<pod 0 address> bash .../graph.sh   # pod 1

# 2A0F,0A2F: Attention on pod 0, FFN on pod 1
LAYOUT=2A0F,0A2F POD=AFD_0 ATTN_HEAD_HOST=<pod 0> FFN_HEAD_HOST=<pod 1> bash .../graph.sh
LAYOUT=2A0F,0A2F POD=AFD_1 ATTN_HEAD_HOST=<pod 0> FFN_HEAD_HOST=<pod 1> bash .../graph.sh
```

Send traffic to port `18000` of the first pod with Attention ranks; the other
pods serve no requests. Pass the fabric's NCCL settings (for example
`NCCL_IB_HCA`) through the environment. The roles can finish loading minutes
apart, so the recipes raise `afd_process_group_timeout_s` and
`--cpu-distributed-timeout-seconds` to 3600 s.

## Deploying on Kubernetes

The **`deploy-afd-k8s`** skill (`.agents/skills/deploy-afd-k8s/SKILL.md`)
deploys a recipe as one pod per `LAYOUT` entry, sizes each pod's GPUs from its
entry, and creates the Services the recipe's default host names point at
(`vllm-ffn-p2p-service`, `vllm-attn-dp-service`, `vllm-ffn-dp-service`). The
image must contain afdEP-comm and DeepGEMM.

## Checking a deployment

```bash
curl -s http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

The answer should name Paris. For an accuracy check, run the matching E2E case
(`afd-deepep-m2n-*`); see
[`tests/e2e`](../../../tests/e2e/README.md#deepep-m2n-scenarios).

## Decode-only benchmark

`DECODE_BENCH` attaches the decode-only benchmark's KV connector
(`AFDDecodeBenchConnector`) to Attention and takes its
`kv_connector_extra_config` as JSON: every Attention DP rank prefills a pool
of `num_prompts` random prompts of `prompt_len` tokens, pins their KV, and
remaps every later request onto the pool, so measured requests decode over
real KV without prefill. Set `max_output_len` to the longest output you
measure; it sizes the startup KV check. The connector lives in
`tools/benchmarks`, which is not in the wheel: run from the repository root or
set `AFD_REPO`.

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
DECODE_BENCH='{"num_prompts": 500, "prompt_len": 1025, "max_output_len": 256}' \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh

python tools/benchmarks/decode_bench_client.py \
    --port 18000 --dp-size 2 --pool-prompts 500 --prompt-len 1025 \
    --result-dir /tmp/results/decode_bench \
    -- --num-prompts 2000 --random-output-len 256 \
       --max-concurrency 64 --request-rate inf
```

`--dp-size` is the Attention DP size over all pods (total Attention ranks
divided by `ATTN_TP`; 2 for `2A2F`, `2A0F,0A2F` and `1A1F,1A1F`), and
`--pool-prompts` and `--prompt-len` must match `DECODE_BENCH`. The benchmark
works in every layout and mode. How the pool, the client and its checks work,
and how to size the pool, are described in
[`tools/benchmarks/README.md`](../../../tools/benchmarks/README.md); the model
READMEs give each model's KV size per token.

## Configuration notes

- Both roles set `compute_gate_on_attention: true` and run with
  `--enable-expert-parallel` and `--enable-ep-weight-filter`, so each FFN rank
  reads only its own experts.
- The FFN keeps vLLM's default `--moe-backend auto`, which the plugin pins to
  `triton` so the expert weights stay in the layout the connector reads.
- The connector derives its dispatch buckets from the CUDA graph capture
  sizes and compiles one transport kernel set per bucket at startup. The
  dispatch capacity is the larger of `--max-num-batched-tokens` and the
  largest capture size, so lowering `--max-num-batched-tokens` shrinks the
  transport buffers.
- With DBO the connector uses two lanes; DBO must be on for both roles.
- `expert_num_sms` and `num_sms` split SMs between the DeepGEMM experts and
  the transport kernels; pass them with `M2N_EXTRA_CONFIG`. See
  [`connector_extra_config`](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#connector_extra_config).
- `FULL_DECODE_ONLY` is the only CUDA graph mode.
