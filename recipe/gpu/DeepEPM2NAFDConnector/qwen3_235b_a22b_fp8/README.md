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
├── 4a2f_graph_dp4tp1_eplb.sh  # graph + EPLB, FFN DP2/EP2, Attention DP4/TP1
├── multipod_4a1f_4a1f_dbo.sh  # 2 pods, graph + DBO, Attention DP8/TP1
├── multipod_2a2f_graph_eplb.sh  # 2 pods of 2A2F, graph + EPLB, 4A4F total
└── multipod_8a8f_graph_eplb.sh  # 2 pods, graph + EPLB, 8A on pod 0, 8F on pod 1
```

## Defaults

| Setting | `4a2f_graph_dp4tp1.sh`, `4a2f_eager_dp4tp1.sh` | `4a2f_dbo_dp4tp1.sh`, `multipod_4a1f_4a1f_dbo.sh` |
| --- | --- | --- |
| `LAYOUT` | `4A2F` | `4A2F` / `4A1F,4A1F` / `8A0F,0A8F` |
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
| `2A2F,2A2F` | 2 pods of 4 | 4A4F: Attention DP4, FFN DP4/EP4 |

## Notes

- Every layout with two FFN ranks puts about 110 GB of FP8 experts on each
  FFN GPU, so it needs H200-class GPUs (about 140 GB); four or more FFN ranks
  fit smaller GPUs. Attention ranks hold no experts
  (`--enable-ep-weight-filter`).
- `multipod_2a2f_graph_eplb.sh` spreads 4A4F over two 4-GPU pods: each pod
  runs FFN DP2 on GPUs 0-1 and Attention DP2 on GPUs 2-3, so each FFN rank
  holds 36 of the 144 physical experts (128 routed + 16 redundant).
- The FFN must load the block-128 FP8 checkpoint; the startup handshake
  rejects a mismatch between the payload format and the loaded expert weights.
- `ATTN_TP=4` with `LAYOUT=4A2F` runs Attention as one DP1/TP4 engine; raise
  `--max-num-seqs` and the capture sizes with it, since one engine then holds
  every running request.
- Qwen3-235B holds 188 KiB of BF16 KV per token (94 layers, 4 KV heads of
  128); an H200 Attention rank has room for about 640K tokens.

## Limitations

- Qwen3 MoE only; see the connector guide for the full list of requirements.
