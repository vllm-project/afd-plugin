# Qwen3-235B-A22B-FP8 with DeepEPM2NAFDConnector

Qwen3-235B-A22B-FP8 recipes with the FP8 dispatch payload. The scripts,
layouts, settings, multi-pod runs and the decode-only benchmark are described
once in the [connector recipes README](../README.md); this page lists what is
specific to the model.

```text
prefill_decode_colocation/
├── 4a2f_eager_dp4tp1.sh       # eager, FFN DP2/TP1, Attention DP4/TP1
├── 4a2f_graph_dp4tp1.sh       # graph, FFN DP2/TP1, Attention DP4/TP1
├── 4a2f_dbo_dp4tp1.sh         # graph + DBO, FFN DP2/TP1, Attention DP4/TP1
└── multipod_4a1f_4a1f_dbo.sh  # 2 pods, graph + DBO, Attention DP8/TP1
```

## Defaults

| Setting | `4a2f_graph_dp4tp1.sh`, `4a2f_eager_dp4tp1.sh` | `4a2f_dbo_dp4tp1.sh`, `multipod_4a1f_4a1f_dbo.sh` |
| --- | --- | --- |
| `LAYOUT` | `4A2F` | `4A2F` / `4A1F,4A1F` |
| `--max-num-seqs` | 256 per Attention DP rank | 512 |
| `--max-num-batched-tokens` | 3072 | 3072 |
| `--max-model-len` | 4096 | 4096 |
| `CUDAGRAPH_CAPTURE_SIZES` | 8,12,16,24,32,48,64,96,128,192,256 | 8,16,32,64,96,128,192,256,384,512 |
| `connector_extra_config` | `"payload_format": "fp8_block128_fp32"` | same |

Common layouts:

| `LAYOUT` | GPUs | Notes |
| --- | --- | --- |
| `2A2F` | 4 on one host | each FFN GPU holds half of the 128 experts |
| `4A2F` | 6 on one host | Attention DP4 on GPUs 2-5 |
| `4A1F,4A1F` | 2 pods of 5 | Attention DP8, FFN DP2/EP2 |

## Notes

- Every layout with two FFN ranks puts about 110 GB of FP8 experts on each
  FFN GPU, so it needs H200-class GPUs (about 140 GB); four or more FFN ranks
  fit smaller GPUs. Attention ranks hold no experts
  (`--enable-ep-weight-filter`).
- The FFN must load the block-128 FP8 checkpoint; the startup handshake
  rejects a mismatch between the payload format and the loaded expert weights.
- `ATTN_TP=4` with `LAYOUT=4A2F` runs Attention as one DP1/TP4 engine; raise
  `--max-num-seqs` and the capture sizes with it, since one engine then holds
  every running request.
- Qwen3-235B holds 188 KiB of BF16 KV per token (94 layers, 4 KV heads of
  128); an H200 Attention rank has room for about 640K tokens.

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
