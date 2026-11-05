# Qwen3-235B-A22B-FP8 with DeepEPM2NAFDConnector

Launch scripts for Qwen3-235B-A22B-FP8 on `DeepEPM2NAFDConnector` with vLLM
`v0.30.0`, using the FP8 dispatch payload.

## Prerequisites

- 6 SM90 or newer GPUs on one node with enough memory for the model. This
  configuration was run on H200.
- The environment described in the
  [connector recipes README](../README.md#prerequisites).
- Qwen3-235B-A22B-FP8 weights on disk (block-128 FP8). The scripts default to
  `/path/model_weights/Qwen3-235B-A22B-FP8`; override with `MODEL_PATH=...`.
- Free ports `6239` (AFD rendezvous), `18000` (Attention API) and `18001`
  (FFN).

## Layout

`4A2F`: Attention DP4 x TP1 on GPUs 0-3 and FFN DP2 on GPUs 4,5. Each FFN rank
owns half of the 128 experts.

```text
prefill_decode_colocation/
├── 4a2f_graph_dp4tp1.sh       # 256 sequences per Attention rank, FULL_DECODE_ONLY graph
└── 4a2f_graph_dbo_dp4tp1.sh   # 512 sequences per Attention rank, graph + DBO
```

## Configuration notes

- `connector_extra_config` sets `payload_format: "fp8_block128_fp32"`. The
  FFN must load the block-128 FP8 checkpoint; the startup handshake rejects a
  mismatch between the payload format and the loaded expert weights.
- The FFN keeps vLLM's default `--moe-backend auto`, which the plugin pins to
  `triton` so the expert weights stay in the layout the connector reads.
- `afd_process_group_timeout_s` and `--cpu-distributed-timeout-seconds` are
  raised to 3600 s because loading the weights can take longer than the
  default 120 s join timeout, especially from cold storage.
- The capture sizes cover `--max-num-seqs`. The connector derives its dispatch
  buckets from them and compiles one transport kernel set per bucket at
  startup.
- `expert_num_sms` (default `0`, all SMs) and `num_sms` tune how SMs are split
  between DeepGEMM experts and the transport kernels; see the connector guide.

## Checking a deployment

```bash
curl -s http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL_PATH"'","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

The answer should name Paris. An FFN log line `Using TRITON Fp8 MoE backend`
confirms the expected expert-weight layout.

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
- `FULL_DECODE_ONLY` is the only CUDA graph mode.
