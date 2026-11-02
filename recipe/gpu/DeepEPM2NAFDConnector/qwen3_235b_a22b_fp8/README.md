# Qwen3-235B-A22B-FP8 with DeepEPM2NAFDConnector

Launch scripts for Qwen3-235B-A22B-FP8 on `DeepEPM2NAFDConnector` with vLLM
`v0.30.0`, using the FP8 dispatch payload.

## Prerequisites

- 6 SM90 or newer GPUs on one node with enough memory for the model, or two
  pods of 5 such GPUs on an RDMA fabric for the multi-pod recipe. These
  configurations were run on H200.
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
├── 4a2f_graph_dp4tp1.sh          # 256 sequences per Attention rank, FULL_DECODE_ONLY graph
├── 4a2f_graph_dbo_dp4tp1.sh      # 512 sequences per Attention rank, graph + DBO
└── multipod_4a1f_4a1f_graph.sh   # 2 pods x 4A1F, FULL_DECODE_ONLY graph (DBO optional)
```

### Multi-pod: 2 x 4A1F

`multipod_4a1f_4a1f_graph.sh` runs on two pods of 5 GPUs. Each pod runs one
FFN rank on GPU 0 and four Attention ranks on GPUs 1-4 (Attention DP8 x TP1,
FFN DP2/EP2 in total), so every FFN GPU holds half of the experts, as in 4A2F.
Pod 0 leads both DP groups and hosts the AFD rendezvous; pod 1 starts both
servers headless and serves no requests. Send traffic to pod 0's port `18000`.

Only `POD` differs between pods:

```bash
# Pod 0
POD=AFD_0 bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/multipod_4a1f_4a1f_graph.sh
# Pod 1
POD=AFD_1 bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/multipod_4a1f_4a1f_graph.sh
```

On Kubernetes the script reaches pod 0 through the Services that the
`deploy-afd-k8s` skill creates. Without Kubernetes, set `HEAD_HOST` to pod 0's
address on both pods; pod 0 must accept ports `6239`, `13345` (Attention DP)
and `13355` (FFN DP). Pass the fabric's NCCL settings (for example
`NCCL_IB_HCA`) through the environment.

`NUM_PODS`, `ATTN_PER_POD` and `FFN_PER_POD` select another symmetric layout
(`2 x 6A2F`, ...); the tuning values (`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`,
`MAX_MODEL_LEN`, `CAPTURE_SIZES`, `PAYLOAD_FORMAT`) and DBO (`ENABLE_DBO=1`,
`DBO_DECODE_TOKEN_THRESHOLD`, `DBO_PREFILL_TOKEN_THRESHOLD`) are environment
variables with the 4A2F values as defaults. The connector logs the placement
it found and the afdEP-comm mode it picked, for example
`DeepEP M2N transport: per_domain f=1 domains=2x5 placement=4A1F,4A1F`.

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
