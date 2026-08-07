# Qwen3.6-35B-A3B CUDA AFD correctness recipe

This recipe validates the Qwen3.6 MoE experts boundary on CUDA. It is not a
performance recipe.

## Scope

- vLLM `0.26.0`, ModelRunner V1, original BF16 checkpoint.
- One host with four CUDA GPUs: Attention `0,1` / TP2 and FFN `2,3` / TP2.
- `P2pNcclAFDConnector`, synchronous NCCL P2P, eager execution and
  `FULL_DECODE_ONLY` CUDA Graph with `compilation_config.mode=0`.
- Text-only execution through `--language-model-only`.

The checked gates cover 1A1F eager for both router placements, 2A2F TP2 eager
state isolation, TP2 Graph batch 1/2, TP2/EP2 Graph batch 1, DP2/EP2 eager,
and AFD-owned two-ubatch DBO + Graph runtime execution. Async communication,
SP/PP, multi-node, quantization, and performance measurements are outside this
recipe.

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

The test starts a native TP2 oracle and then AFD 2A2F TP2. It requires exact
generated token IDs, exact per-step top-5 token sets, and exact logprobs.

## Validated evidence

The CUDA validation used Qwen3.6-35B-A3B original BF16 weights on four RTX PRO
6000 Blackwell GPUs. Native TP2 and AFD 2A2F TP2 matched across batch 1, batch
2, repeated requests, and A/B/A interleaving: token IDs and top-5 sets were
exact, and the maximum logprob absolute error was `0.0`.

Attention owns the router checkpoint weights and router computation, along
with attention/KV and hybrid state; it has no routed or shared expert weights.
FFN owns native routed/shared experts and has an empty KV-cache spec. To
preserve the native model tree and loader contract inherited from PR #176,
FFN retains a dormant native router module, but it loads no router checkpoint
weights and never executes router computation. Raw responses and logs are
intentionally not tracked.

## Capability runner

Use the narrow runner to execute one gate and write its JUnit XML and JSON
summary outside the worktree:

```bash
python scripts/qwen36_v026/run_capability_matrix.py \
  --topology 2a2f --gate-side attention --mode graph --batch-size 2 \
  --compare-native --cleanup
```

The runner deliberately rejects combinations that do not have an exact gate.
For example, native vLLM 0.26 DBO requires an installed DeepEP or NIXL all2all
kernel. On hosts without either kernel, the `2a2f/dbo-graph` gate proves two
real AFD ubatches, FFN Graph capture, and replay, but it does not claim an
exact native DBO comparison.

CUDA Graph correctness uses `FULL_DECODE_ONLY` with
`{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}`. This intentionally
keeps Inductor compilation out of the correctness baseline; do not treat it as
a performance configuration.
