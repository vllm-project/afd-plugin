# DeepSeek-V2-Lite AFD Examples

End-to-end launch scripts for running DeepSeek-V2-Lite with the AFD
(Attention-FFN Disaggregation) plugin on vLLM `v0.28.0`. The 2A2F
eager/graph/DBO family was hardware-validated on NVIDIA L20X through
`tests/e2e/models/deepseek_v2_lite/` on vLLM `v0.28.0`; the individual
launch scripts below were inspected against that runtime and not
re-run one by one.

> [!NOTE]
> `P2pNcclAFDConnector` is an example connector implementation. Contributions
> of high-performance communication connectors and new approaches to AFD are
> welcome.

## Prerequisites

- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 4 GPUs(A/H-class, tested against L20X).
- vLLM `v0.28.0` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- DeepSeek-V2-Lite weights on disk. All scripts default to
  `/path/model_weights/DeepSeek-V2-Lite`; override with
  `MODEL_PATH=...` when launching.
- A free TCP port `6269` on `127.0.0.1` for the AFD p2p connector, and
  ports `18301`/`18302`/`18303`/`18305` for the vLLM HTTP servers.

## Directory layout

```
.
├── prefill_decode_disaggregation/        # prefill_decode_disaggregation, 2P1A1F topology
│   ├── 2p1a1f_eager_dbo.sh
│   └── 2p1a1f_graph_dbo.sh
└── prefill_decode_colocation/             # prefill_decode_colocation, 2A2F topology
    ├── 2a2f_eager_dbo_dp1tp2.sh
    ├── 2a2f_eager_dbo_dp2tp1.sh
    ├── 2a2f_graph_dbo_dp1tp2.sh
    └── 2a2f_graph_dbo_dp2tp1.sh
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

2 processes, two GPUs each:

| GPUs | Role      | Port  |
|------|-----------|-------|
| 0, 1 | Attention | 18305 |
| 2, 3 | FFN       | 18305 |

The four variants cover the TP/DP cross product:

| File                            | DP | TP |
|---------------------------------|----|----|
| `2a2f_*_dp1tp2.sh`              | 1  | 2  |
| `2a2f_*_dp2tp1.sh`              | 2  | 1  |

## Running

Pick a script and execute it from the repository root. Each script
backgrounds its workers and writes per-worker logs (`afd_prefill0.log`, `afd_prefill1.log`, `attn.log`, `ffn.log`) in the current directory.

Wait for `attn.log` (and `afd_prefill0.log`, `afd_prefill1.log` in disaggregation) to print the `Application startup complete` line
before sending traffic.

### prefill_decode_colocation
```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V2_MODEL_RUNNER=0
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh
```

### prefill_decode_disaggregation

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V2_MODEL_RUNNER=0
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/2p1a1f_graph_dbo.sh
```

### Running the benchmark

Once the serving stack is up, run:

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export MODEL_NAME=$MODEL_PATH
export RESULT_DIR=/tmp/results
export RESULT_FILENAME=2a2f_graph_dbo_dp1tp2.json
bash tools/benchmarks/benchmark.sh
```

The script waits for `http://$HOST:$PORT/v1/models`, sends one completion
smoke request, then runs `vllm bench serve`. By default it fires 1024 random
requests (1024 input tokens / 128 output tokens) at request rate 5 with
`--max-concurrency 32` against `127.0.0.1:18305`, and dumps the JSON result to
`$RESULT_DIR/$RESULT_FILENAME`. Override `HOST`, `PORT`, `MODEL_NAME`,
`NUM_PROMPTS`, `REQUEST_RATE`, `MAX_CONCURRENCY`, `INPUT_LEN`, and `OUTPUT_LEN`
for smaller smoke runs or larger throughput sweeps.

## Common AFD configuration

Every AFD worker is wired through `--additional-config` with the same
shape; `role` differs between attention and FFN:

```jsonc
{
  "afd": {
    "role": "attention",            // or "ffn"
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 6269,
    "num_attention_ranks": 1,      // 2 in 2A2F
    "num_ffn_ranks": 1             // 2 in 2A2F
  }
}
```

DBO (Dual Batch Overlap) is turned on for all examples with
`--dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12`.
These threshold values are example defaults; tune them for the actual workload
and deployment configuration.

### Switching eager → graph

Graph mode replaces `--enforce-eager` with:

```
--max-cudagraph-capture-size 64
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
                       "cudagraph_capture_sizes":[64]}'
```

The capture size of `64` is an example value; adjust
`--max-cudagraph-capture-size` and `cudagraph_capture_sizes` together based on
the actual batch sizes and available GPU memory.
