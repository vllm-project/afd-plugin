# GPU vLLM 0.34 upgrade validation record

Status: **implemented, partial GPU validation**. The AFD eager path is validated
on the target GPU; the AFD CUDA Graph path is blocked by an upstream vLLM 0.34
regression that reproduces without AFD. This record separates what was exercised
on hardware from what was not.

## Frozen identity

| Item | Value |
| --- | --- |
| AFD branch | `upgrade/gpu-vllm-034` |
| AFD base | `40b4a49` |
| Current vLLM | `0.32.0` = `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Target vLLM | `0.34.0` = `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` |
| Python | 3.12.13 |
| PyTorch | 2.14.0+cu130 |
| CUDA runtime / toolkit | 13.0 |
| Driver | 570.133.20 |
| GPU | 8 x NVIDIA L20X, 143771 MiB each, compute capability 8.9 (SM89) |

## Hardware evidence

The qualification model is DeepSeek-V2-Lite at
`/data/models/hub/models--deepseek-ai--Deepseek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0`
(GSM8K, 8-shot, `max-model-len=4096`). The runs below used a vLLM 0.34 venv with
the AFD checkout installed editable.

| Cell | Execution | GPUs | Result | Evidence |
| --- | --- | --- | --- | --- |
| AFD, DeepSeek-V2-Lite, `afd-v2-eager-1a1f` smoke | eager, `P2pNcclAFDConnector` | 2 | pass; 7 GSM8K samples, accuracy 0.285714 (smoke threshold 0.25) | `artifacts/v2-afd-eager-1a1f-g/run.json` (`phases[0].result`) |
| AFD, DeepSeek-V2-Lite, `afd-v2-eager-1a1f` full GSM8K | eager, `P2pNcclAFDConnector` | 2 | pending at the time of this record | `artifacts/v2-afd-eager-1a1f-fullacc/` |
| Native baseline, `baseline-graph` smoke | CUDA Graph, no AFD | 4 (Attention DP4) | pass; 7 GSM8K samples, accuracy 0.285714; cleanup passed, shutdown graceful | `artifacts/v2-baseline-graph4/pytest/test_deepseek_v2_lite_baseline0/baseline-graph/run.json` |
| AFD, DeepSeek-V2-Lite, `afd-v2-graph-1a1f` | CUDA Graph, `P2pNcclAFDConnector` | 2 | blocked; see below | `artifacts/v2-afd-graph-1a1f/attention.log` |
| Native no-AFD compiled-prefill probe | compile, no AFD | 1 | blocked; same failure | `artifacts/native-graph-prefill-probe/serve.log` |

Static checks: the local `pytest tests/unit` run on vLLM 0.34 leaves only the six
environment errors in `tests/unit/ffn_faas/test_main.py` caused by missing GPU
device inference; no code failures.

## CUDA Graph blocker (upstream)

The AFD graph case fails on its first completion request. The attention
EngineCore reports

```text
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx(...)
```

inside the compiled MLA `kv_b_proj`, then the process exits and the API server
stops listening on `127.0.0.1:19840`.

The same failure reproduces **without AFD**: a plain `vllm serve` of the same
model on one L20 with the default `max_num_batched_tokens` and a roughly
1.8k-token prompt fails in compiled prefill with the same
`CUBLAS_STATUS_EXECUTION_FAILED`, followed by
`cudaErrorIllegalAddress` / sticky error. The eager path handles the same prompt
correctly. The native `baseline-graph` case passes only because it sets
`max_num_batched_tokens=8`, so its prefill is chunked into eight-token pieces
and never enters the failing large-prefill compiled shape.

Conclusion: this is a vLLM 0.34.0 compiled-prefill issue on L20 (SM89), not an
AFD change. It is not worked around by changing AFD production code; a chunked
prefill workaround would no longer exercise the intended graph path and is not
counted as validation.

## Coverage gaps

- AFD CUDA Graph on vLLM 0.34 is **not validated** on L20 because of the
  upstream compiled-prefill failure above.
- DeepSeek-V4-Flash is **not validated** on L20: vLLM 0.34's DeepGEMM path
  targets SM90/SM100/SM120 while L20 is SM89, so the V4 FP8 MLA path is outside
  the supported hardware set. This is a hardware/toolchain limit, not an AFD
  regression.
- Full GSM8K accuracy on the AFD eager path is recorded separately when the run
  completes; the 7-sample smoke result above is not a full accuracy claim.
- DBO, TP2/DP2 AFD graph variants, and multi-pod cases were not run; they remain
  outside this record.
