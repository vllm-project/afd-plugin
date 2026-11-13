# DeepSeek-V2-Lite AFD Examples

End-to-end launch scripts for running DeepSeek-V2-Lite with the AFD
(Attention-FFN Disaggregation) plugin targeting vLLM `v0.32.0`. These are
deployment examples, not v0.32 recipe validation. See the
[runtime matrix](../../../../docs/support.md#tested-runtime-matrix)
for tested E2E configurations.

> [!NOTE]
> `P2pNcclAFDConnector` is an example connector implementation. Contributions
> of high-performance communication connectors and new approaches to AFD are
> welcome.

## Prerequisites

- Install [NIXL](https://github.com/ai-dynamo/nixl).
- Four GPUs for the default `2A2F` colocation layout and the `2p1a1f_*`
  examples; `LAYOUT=4A4F` needs eight GPUs, and `LAYOUT=2A0F,0A2F` two GPUs
  on each of two pods.
- vLLM `v0.32.0` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- DeepSeek-V2-Lite weights on disk. Colocation scripts default to
  `/path/model_weights/DeepSeek-V2-Lite`; disaggregation scripts default to
  `/home/models/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0`.
  Set `MODEL_PATH=...` to the local weights when launching.
- Free ports `6239` (AFD rendezvous), `18000` and `18001` (Attention and
  FFN HTTP) for colocation, or `18301`–`18305` and `6269` for
  prefill/decode disaggregation.

## Directory layout

```text
.
├── prefill_decode_disaggregation/        # prefill_decode_disaggregation, 2P1A1F topology
│   ├── 2p1a1f_eager_dbo.sh
│   └── 2p1a1f_graph_dbo.sh
└── prefill_decode_colocation/             # any LAYOUT, ATTN_TP, FFN_TP
    ├── 2a2f_eager_dbo_dp2tp1.sh           # 2A2F, eager + DBO, DP2/TP1
    ├── 2a2f_graph_dbo_dp2tp1.sh           # 2A2F, graph + DBO, DP2/TP1
    ├── 2a2f_eager_dbo_dp1tp2.sh           # 2A2F, eager + DBO, DP1/TP2
    ├── 2a2f_graph_dbo_dp1tp2.sh           # 2A2F, graph + DBO, DP1/TP2
    ├── 4a4f_eager_dbo_dp2tp2.sh           # 4A4F, eager + DBO, DP2/TP2
    ├── 4a4f_graph_dbo_dp2tp2.sh           # 4A4F, graph + DBO, DP2/TP2
    └── multipod_2a_2f_graph_dbo_dp1tp2.sh # 2A0F,0A2F, graph + DBO, DP1/TP2
```

### 1. Prefill/Decode Disaggregation — `1a1f`

5 processes, 4 GPU workers + 1 proxy server:

| GPUs | Role                | Port  |
|------|---------------------|-------|
| 0    | Prefill (Colocated) | 18301 |
| 1    | Prefill (Colocated) | 18302 |
| 2    | Decode (Attention)  | 18303 |
| 3    | Decode (FFN)        | 18304 |
| /    | Proxy Server        | 18305 |

### 2. Prefill/Decode Colocation — `2a2f`

The colocation recipes run on the shared
[colocation launcher](../../common/README.md): `LAYOUT` places the ranks and
each script names its default DP/TP shape (FFN first, Attention second, and
Attention HTTP port 18000):

| Script suffix | Attention / FFN | GPUs |
| --- | --- | --- |
| `2a2f_*_dp2tp1` | DP2 x TP1 each | 4 |
| `2a2f_*_dp1tp2` | DP1 x TP2 each | 4 |
| `4a4f_*_dp2tp2` | DP2 x TP2 each | 8 |
| `multipod_2a_2f_*_dp1tp2` | DP1 x TP2 each, one pod per role | 2 + 2 |

## Running

Pick a script and execute it from the repository root. Each script
backgrounds its workers and writes per-worker logs (`afd_prefill0.log`, `afd_prefill1.log`, `attn.log`, `ffn.log`) in the current directory.

Wait for `attn.log` (and `afd_prefill0.log`, `afd_prefill1.log` in disaggregation) to print the `Application startup complete` line
before sending traffic.

### prefill_decode_colocation

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh
```

### prefill_decode_disaggregation

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V2_MODEL_RUNNER=1
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh
```

### Running the benchmark

Confirm the serving stack is ready before running:

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export RESULT_DIR=/tmp/results
export RESULT_FILENAME=2a2f_graph_dbo_dp1tp2.json
bash tools/benchmarks/request_generator.sh
```

The script runs `vllm bench serve`; it does not wait for readiness or send a
separate smoke request. By default it sends 1024 random requests (1024 input
tokens / 128 output tokens) at request rate 5 with `--max-concurrency 32`
against `127.0.0.1:18000` (the colocation Attention port; set `PORT=18305`
for the disaggregation proxy), and saves the JSON result to
`$RESULT_DIR/$RESULT_FILENAME`. Override `HOST`, `PORT`, `MODEL_PATH`,
`NUM_PROMPTS`, `REQUEST_RATE`, `MAX_CONCURRENCY`, `INPUT_LEN`, and `OUTPUT_LEN`
for smaller smoke runs or larger throughput sweeps.

## Common AFD configuration

Every AFD worker is wired through `--additional-config` with the same
shape; `role` differs between attention and FFN. The colocation launcher
builds it from `LAYOUT` (the disaggregation scripts write it inline, on port
6269); print it with `AFD_DRY_RUN=1`:

```jsonc
{
  "afd": {
    "role": "attention",            // or "ffn"
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 6239,
    "num_attention_ranks": 2,       // Attention GPUs in LAYOUT
    "num_ffn_ranks": 2,             // FFN GPUs in LAYOUT
    "compute_gate_on_attention": false,
    "afd_process_group_timeout_s": 3600
  }
}
```

DBO (Dual Batch Overlap) is turned on for all examples with
`--dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12`.
These threshold values are example defaults; tune them for the actual workload
and deployment configuration.

The `graph_dbo` recipes capture `FULL_DECODE_ONLY` CUDA graphs of size `64`
(`CUDAGRAPH_CAPTURE_SIZES=64`); the `eager_dbo` recipes run
`--enforce-eager`. Adjust
`CUDAGRAPH_CAPTURE_SIZES` to the actual batch sizes and available GPU memory.
