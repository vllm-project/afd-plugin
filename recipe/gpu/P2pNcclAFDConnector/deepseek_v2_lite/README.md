# DeepSeek-V2-Lite AFD Examples

End-to-end launch scripts for running DeepSeek-V2-Lite with the AFD
(Attention-FFN Disaggregation) plugin targeting vLLM `v0.32.0`. These are
deployment examples, not v0.32 recipe validation. See the
[runtime matrix](../../../../docs/design/module/execution_platforms.md#tested-runtime-matrix)
for tested E2E configurations.

> [!NOTE]
> `P2pNcclAFDConnector` is an example connector implementation. Contributions
> of high-performance communication connectors and new approaches to AFD are
> welcome.

## Prerequisites

- Install [NIXL](https://github.com/ai-dynamo/nixl).
- Four GPUs for the `2a2f_*` and `2p1a1f_*` examples; the `4a4f_*`
  examples require eight GPUs. The multipod example uses two GPUs per pod
  across two pods.
- vLLM `v0.32.0` and the `afd-plugin` package installed in the same
  environment (see repository root `AGENTS.md`).
- DeepSeek-V2-Lite weights on disk. Colocation scripts default to
  `/path/model_weights/DeepSeek-V2-Lite`; disaggregation scripts default to
  `/home/models/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0`.
  Set `MODEL_PATH=...` to the local weights when launching.
- A free TCP port `6269` on `127.0.0.1` for the AFD p2p connector, and
  HTTP ports `18305` and `18306` for local colocation, or
  `18301`–`18305` for prefill/decode disaggregation.

## Directory layout

```text
.
├── prefill_decode_disaggregation/        # prefill_decode_disaggregation, 2P1A1F topology
│   ├── 2p1a1f_eager_dbo.sh
│   └── 2p1a1f_graph_dbo.sh
└── prefill_decode_colocation/             # prefill_decode_colocation, 2A2F/4A4F topology
    ├── 2a2f_eager_dbo_dp1tp2.sh
    ├── 2a2f_eager_dbo_dp2tp1.sh
    ├── 2a2f_graph_dbo_dp1tp2.sh
    ├── 2a2f_graph_dbo_dp2tp1.sh
    ├── 4a4f_eager_dbo_dp2tp2.sh
    ├── 4a4f_graph_dbo_dp2tp2.sh
    └── multipod_2a_2f_graph_dbo_dp1tp2.sh  # 2a2f_graph_dbo_dp1tp2, split across 2 k8s pods
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
| 2, 3 | FFN       | 18306 |

The four `2a2f_*` variants cover the TP/DP cross product; `4a4f_*` doubles
both attention and FFN ranks (DP=2 on each side); `multipod_2a_2f_graph_dbo_dp1tp2.sh`
is the same `2a2f_graph_dbo_dp1tp2` config, but gated on a `POD` env var
(`ATTENTION_0` or `FFN_0`) for the `deploy-afd-k8s` skill to deploy it
across 2 pods (see repo-root `recipe/gpu/P2pNcclAFDConnector/README.md`);
its Attention and FFN HTTP ports can both be 18305 because they are in
separate pods --
unlike the scripts above, it is **not** meant to be run directly: with
`POD` unset, neither `vllm serve` block executes, and its
`AFD_CONNECTOR_HOST` default (`vllm-ffn-p2p-service`) only resolves inside
the cluster the skill deploys it to:

| File                                  | DP | TP |
|---------------------------------------|----|----|
| `2a2f_*_dp1tp2.sh`                    | 1  | 2  |
| `2a2f_*_dp2tp1.sh`                    | 2  | 1  |
| `4a4f_*_dp2tp2.sh`                    | 2  | 2  |
| `multipod_2a_2f_graph_dbo_dp1tp2.sh`  | 1  | 2  |

## Running

Pick a script and execute it from the repository root. Each script
backgrounds its workers and writes per-worker logs (`afd_prefill0.log`, `afd_prefill1.log`, `attn.log`, `ffn.log`) in the current directory.

Wait for `attn.log` (and `afd_prefill0.log`, `afd_prefill1.log` in disaggregation) to print the `Application startup complete` line
before sending traffic.

### prefill_decode_colocation

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V2_MODEL_RUNNER=1
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
against `127.0.0.1:18305`, and saves the JSON result to
`$RESULT_DIR/$RESULT_FILENAME`. Override `HOST`, `PORT`, `MODEL_PATH`,
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

```bash
--max-cudagraph-capture-size 64
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
                       "cudagraph_capture_sizes":[64]}'
```

The capture size of `64` is an example value; adjust
`--max-cudagraph-capture-size` and `cudagraph_capture_sizes` together based on
the actual batch sizes and available GPU memory.
