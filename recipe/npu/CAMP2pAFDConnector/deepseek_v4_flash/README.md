# DeepSeek-V4-Flash on A5 with A2E MXFP8 transport quantization

Single host, 8× Ascend 950 (`Ascend950DT_9574`), 4A4F: attention DP4/TP1 on
devices 0-3, FFN DP4/TP1 on devices 4-7, connector `CAMP2pAFDConnector`.

The attention side quantizes the FFN activation to MXFP8 (E4M3 + E8M0 block
scale) and transports FP8 instead of BF16, which halves the A2E payload.

## Configurations

| Piece | Value |
| --- | --- |
| `connector_extra_config.quant_mode` | `4` (MXFP8) on **both** roles; the only switch |
| `--max-num-batched-tokens` | `3072` — a measurement setting, not a requirement |
| `--max-num-seqs` | `128` |
| `--compilation-config` | `cudagraph_mode=FULL_DECODE_ONLY`, `cudagraph_capture_sizes=[8,16,32,64,128]` |
| `--max-model-len` | `9216` |
| `kv-cache-dtype` | `fp8`, `attention-config={"indexer_kv_dtype":"fp8"}` |
| `HCCL_BUFFSIZE` | `2048` |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:False` |

## Measured (128 concurrent requests, 4×8192 in / 32 out, two runs each)

| Run | `quant_mode=0` (BF16 transport) | `quant_mode=4` (MXFP8) | Δ |
| --- | --- | --- | --- |
| chunk 3072 | 236,378 token/s | **318,519 token/s** | **+34.7%** |
| chunk 4096 | 218,646 token/s | 297,245 token/s | +35.9% |

TTFT median 3748 ms → 2905 ms (−22.5%). All runs: `Failed Requests = 0`.

GSM8K: 93.86% (`quant_mode=0`) vs **93.71%** (`quant_mode=4`), i.e. about −0.15 points, with
roughly ±0.5 points of measurement spread on this benchmark (the same path
measured 92.87% once in an earlier configuration — do not claim lossless).

## What the measurement depends on

**1. Load has to be high enough for the transport to matter.**
The numbers above were taken at **128 concurrent requests**. The same configuration
measured **−0.3%** instead of +34.7% at 8 concurrent: at that load each layer's
round-trip wait is dominated by its fixed latency rather than by the activation
payload, so the machine is idle enough that the transport hides behind compute.
Treat 128 as the load this was validated at, not as a threshold derived from the
configuration — `--max-num-seqs` is a capacity limit (128 per rank here, so up to
512 in flight across DP4) and says nothing about when the box saturates. Every
performance number taken before this was understood, including the −12.7% and −5.7%
conclusions, came from a partially idle box.

**2. `--max-num-batched-tokens 3072`, not 4096.**
This is purely a measurement setting: the default 2048 is equally correct, and
nothing about the transport needs a particular chunk. 3072 was used because it is
the largest value at which all four ranks fill to the cap. At 4096 only the first
step reaches 4096 and later steps fall to ~2500, which costs throughput for *both*
the baseline and the quantized run; the ratio stays ~35% but the absolute numbers
drop.

## Why the FFN restores BF16

The FFN side dequantizes the transported pair back to the model dtype before its
MoE runs. That is not a fallback: two FFN consumers need a model-dtype activation
regardless — the router reads the activation itself, and the shared experts
quantize their input unconditionally — so a dequantization happens either way. Only
the transport is quantized; every stage after the receive keeps the layout and
dtype it has in BF16 mode.

**A variant that was evaluated and has since been removed.** Letting the routed
experts take the transported FP8 directly — skipping the MoE's internal
quantization — measured **296,934 token/s against 318,519, i.e. 6.8% slower**: it
saves one fused quantization while adding a second live copy of the activation plus
one extra DP `all_gather` per layer for the scale. It also hit an operator limit
this path never triggers (`MoeGatingTopKHash` faults with `aivec error 263` once the
gathered row count reaches ~14,288, chunk ≈ 3,570). It is not in this tree; the
measurement is kept so the trade-off stays on record.

## Running it

Start the FFN role first, then the attention role, and confirm the transport mode
before trusting a benchmark number: the two configurations differ only by
`connector_extra_config.quant_mode`, so a run at `0` looks identical to one at `4`
in every other respect. Tear down with `pkill -TERM -f 'vllm serve'` and give the
NPUs a moment before starting a new run, otherwise the next process sees the
devices as busy.

No launch script is carried in the repository: the pieces that matter are the
configuration table above and what the measurement depends on.

## A non-AFD baseline

The comparison this recipe does **not** settle is "AFD with a quantized transport
versus serving the same model on the same eight devices without AFD". For that,
run one process over all eight devices with `--data-parallel-size 8`, omit
`--additional-config` (that omission is what turns AFD off), and keep everything
else (chunk, `--max-num-seqs`, capture sizes, fp8 KV, model length) identical.

**Two things to compare carefully.**

`DP8 × 3072` gives **24,576** tokens per step, twice the AFD side's
`DP4 × 3072 = 12,288`. Halve the chunk on the baseline for a step-size-matched
run (`8 × 1536 = 12,288`).

`DP` also feeds `world_size_across_dp`, which selects the MoE communication
method: `MC2` while `num_tokens <= mc2_tokens_capacity`, then `ALLGATHER` while
`world_size <= num_experts_per_tok`, otherwise `ALLTOALL`. AFD runs at
`world_size = 4`; the baseline at `8`. Confirm from the logs which path each run
actually took before comparing throughput — a different method is a confounder,
not a result.

This baseline has not been measured, so this recipe makes no claim about it. It
does not change the within-AFD result above either: the +34.7% is the repair of an
AFD-specific cost (the per-layer A2E round trip, measured at 49.5% of
attention-side device time).


