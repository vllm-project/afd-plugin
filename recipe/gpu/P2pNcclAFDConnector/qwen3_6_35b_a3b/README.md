# Qwen3.6-35B-A3B CUDA AFD correctness recipe

This recipe validates the Qwen3.6 MoE experts boundary on CUDA. It is not a
performance recipe.

## Scope

- vLLM `0.26.0`, ModelRunner V1, original BF16 checkpoint.
- One host with four CUDA GPUs: Attention `0,1` / TP2 and FFN `2,3` / TP2.
- `P2pNcclAFDConnector`, synchronous NCCL P2P, eager execution and
  `FULL_DECODE_ONLY` CUDA Graph with `compilation_config.mode=0`.
- Text-only execution through `--language-model-only`.

The checked gates cover GPU initialization rejection for Attention-side gate,
1A1F eager with the FFN-local router, 2A2F TP2 eager state isolation, and TP2
`FULL_DECODE_ONLY` Graph batch 1. Async communication, DBO, SP/PP,
multi-node, quantization, and performance measurements are outside this recipe.

## Start the AFD stack

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
bash recipe/gpu/P2pNcclAFDConnector/qwen3_6_35b_a3b/2a2f_tp2_eager.sh
```

The script starts the FFN role first, then Attention. It writes logs to
`${TMPDIR:-/tmp}/qwen36-afd-logs` by default; set `AFD_QWEN36_LOG_DIR` to
override that location. It terminates both process groups on exit.

## Correctness gate

Run the opt-in test with the checkpoint path and four GPU IDs:

```bash
export AFD_QWEN3_6_E2E_MODEL=/path/to/Qwen3.6-35B-A3B
export AFD_QWEN3_6_E2E_GPUS=0,1,2,3
pytest tests/e2e/models/qwen3_6/test_e2e_gpu.py -q
```

The test starts a same-topology native TP2 oracle and then AFD 2A2F TP2. It
requires exact generated token IDs and top-5 token sets; corresponding
logprobs must match a complete native cold-start observation within `1e-2`.

## Validated evidence

The CUDA validation used Qwen3.6-35B-A3B original BF16 weights on four RTX PRO
6000 Blackwell GPUs. Native TP2 and AFD 2A2F TP2 matched across batch 1, batch
2, repeated requests, and A/B/A interleaving: token IDs and top-5 sets were
exact, and the maximum logprob absolute error was `0.0`.

Attention owns attention/KV and hybrid state; it has no gate, routed-expert,
or shared-expert checkpoint weights. FFN owns and executes the native router,
routed experts, shared expert, and shared-expert gate, and has an empty
KV-cache spec. `compute_gate_on_attention=true` is rejected during GPU
connector initialization. Raw responses and logs are intentionally not tracked.

## Capability runner

Use the narrow runner to execute one gate and write its JUnit XML and JSON
summary outside the worktree:

```bash
python scripts/qwen36_v026/run_capability_matrix.py \
  --topology 2a2f --gate-side ffn --mode graph --batch-size 2 \
  --compare-native --cleanup
```

The runner deliberately rejects unsupported topologies, gate placements, and
modes. Native vLLM 0.26 DBO exact-oracle coverage remains a separate Draft
limitation and is not exercised by this recipe.

CUDA Graph correctness uses `FULL_DECODE_ONLY` with
`{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}`. This intentionally
keeps Inductor compilation out of the correctness baseline; do not treat it as
a performance configuration.
