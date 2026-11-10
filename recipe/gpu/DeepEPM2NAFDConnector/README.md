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

Each script sets the model, its tuning and mode flags, and runs the shared
[colocation launcher](../common/README.md), which documents the layout,
every setting, and how to run a recipe directly or in a pod.

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

`LAYOUT` (default `2A2F`, one pod of 4 GPUs) lists the pods, one `<a>A<f>F`
entry each; see [Layout](../common/README.md#layout) and
[Settings](../common/README.md#settings). Examples for this connector:
`4A2F` (one 6-GPU node), `2A0F,0A2F` and `1A1F,1A1F` (two pods), `4A1F,4A1F`
(two 5-GPU pods). `ATTN_TP=2` runs `2A2F`'s Attention as one DP1 x TP2 engine;
the FFN runs TP 1, and its DP rank (its EP rank) decides which experts it
loads. The connector checks the placement at startup and names the pods'
shapes when it rejects one; the
[supported layouts](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#supported-layouts)
table lists which placements it accepts (for example `4A0F,0A2F` is not, its
pods differ in size).

## Running

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh
```

Send traffic to the Attention server on port `18000` once `attn.log` prints
`Application startup complete`. The first start on a host compiles transport
and DeepGEMM kernels, so it is slower than later starts. Raise
`CUDAGRAPH_CAPTURE_SIZES` together with `--max-num-seqs`, so decode batches
stay in a captured size. Across pods, pass the fabric's NCCL settings (for
example `NCCL_IB_HCA`) through the environment; see
[Running directly](../common/README.md#running-directly) for hosts without
Kubernetes.

## Deploying on Kubernetes

The **`deploy-afd-k8s`** skill (`.agents/skills/deploy-afd-k8s/SKILL.md`)
deploys a recipe as one pod per `LAYOUT` entry; see
[Running in a pod](../common/README.md#running-in-a-pod). The image must
contain afdEP-comm and DeepGEMM.

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
`kv_connector_extra_config` as JSON (`{}` is enough). The connector reports
every prompt token but the last as externally computed, so requests skip
prefill and decode at a context of the full prompt length over unwritten KV.
The connector lives in `tools/benchmarks`, which is not in the wheel: run from
the repository root or set `AFD_REPO`.

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
DECODE_BENCH='{}' \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/graph.sh

INPUT_LEN=1024 OUTPUT_LEN=256 NUM_PROMPTS=2000 \
    MAX_CONCURRENCY=64 REQUEST_RATE=inf RESULT_DIR=/tmp/results/decode_bench \
    tools/benchmarks/request_generator.sh
```

Drive the Attention port (18000). The benchmark works in every layout and
mode; the connector is
[`tools/benchmarks/decode_bench.py`](../../../tools/benchmarks/decode_bench.py).

## Configuration notes

- Both roles set `compute_gate_on_attention: true` and run with
  `--enable-expert-parallel` and `--enable-ep-weight-filter` (in each
  recipe's `SERVE_ARGS`), so each FFN rank reads only its own experts.
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
