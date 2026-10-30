# Qwen3-30B-A3B with DeepEPM2NAFDConnector

Launch scripts for Qwen3-30B-A3B (BF16) on `DeepEPM2NAFDConnector` with vLLM
`v0.30.0`. They use the same layouts as the Qwen3 MoE DeepEP M2N E2E cases
(`afd-deepep-m2n-*-2a2f` in [`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios)).

## Prerequisites

- 4 SM90 or newer GPUs on one node.
- The environment described in the
  [connector recipes README](../README.md#prerequisites).
- Qwen3-30B-A3B weights on disk. The scripts default to
  `/path/model_weights/Qwen3-30B-A3B`; override with `MODEL_PATH=...`.
- Free ports `6239` (AFD rendezvous), `18000` (Attention API) and `18001`
  (FFN).

## Layout

`2A2F`: Attention on GPUs 0,1 and FFN DP2 on GPUs 2,3. FFN rank 0 owns experts
0-63 and FFN rank 1 owns experts 64-127.

```text
prefill_decode_colocation/
├── 2a2f_eager_dp2tp1.sh       # Attention DP2 x TP1, eager
├── 2a2f_graph_dp2tp1.sh       # Attention DP2 x TP1, FULL_DECODE_ONLY graph
├── 2a2f_graph_dbo_dp2tp1.sh   # Attention DP2 x TP1, graph + DBO
└── 2a2f_graph_dp1tp2.sh       # Attention DP1 x TP2, FULL_DECODE_ONLY graph
```

## Configuration notes

- Both roles set `compute_gate_on_attention: true`, use the same rank counts,
  and run with `--enable-expert-parallel`. The FFN runs TP 1.
- `connector_extra_config` is left empty, so the payload is BF16 and the FFN
  runs DeepGEMM experts.
- The graph scripts capture batch sizes 8, 16, 32 and 64. The connector
  derives its dispatch buckets from the capture sizes and compiles one
  transport kernel set per bucket at startup.
- In the DBO script, both roles pass `--enable-dbo` with the same thresholds;
  the connector then uses two lanes.
- The dispatch capacity is derived from `--max-num-batched-tokens` and the
  largest capture size, so lowering `--max-num-batched-tokens` shrinks the
  transport buffers.

## Checking a deployment

```bash
curl -s http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

For an accuracy check, run the matching E2E case, for example
`afd-deepep-m2n-graph-2a2f`; see [`tests/e2e`](../../../../tests/e2e/README.md#deepep-m2n-scenarios).

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
- `FULL_DECODE_ONLY` is the only CUDA graph mode.
