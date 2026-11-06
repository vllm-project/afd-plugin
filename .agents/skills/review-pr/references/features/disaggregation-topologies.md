# Disaggregation Topologies

Load with the primary module contract whenever a review touches role
placement, rank counts, or launch scripts: attention/FFN colocation vs
prefill-decode disaggregation, the A/F rank ratios, and the recipe scripts that
encode them.

Designs: `docs/design/module/attention_runtime.md`,
`docs/design/module/ffn_runtime.md`, `docs/design/module/e2e_testing.md`;
recipes under `recipe/gpu/` and `recipe/npu/`.

## Topology map

| Topology | Where it lives |
| --- | --- |
| Baseline (native DP/TP, no AFD) | `baseline-graph` E2E scenarios; native DP4/TP1/EP4 |
| CUDA NCCL P2P, colocation | `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/` (2A2F and 4A4F, eager/graph, DBO variants, plus `multipod_2a_2f_graph_dbo_dp1tp2.sh`) |
| CUDA NCCL P2P, disaggregation | `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_disaggregation/` (2P1A1F eager/graph DBO) |
| CUDA NCCL P2P, Qwen3.5 multi-pod | `recipe/gpu/P2pNcclAFDConnector/qwen3_5_122b_a10b_fp8/prefill_decode_colocation/multipod_2a_2a_2f_graph.sh` |
| CUDA DeepEP M2N (routed experts) | `recipe/gpu/DeepEPM2NAFDConnector/{deepseek_v2_lite,qwen3_30b_a3b,qwen3_235b_a22b_fp8}/` (2A2F, and 4A2F for the 235B FP8 case) |
| NPU CAM P2P disaggregation | `recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/` (attention/FFN/baseline scripts) |
| NPU async CAM | `recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy/` and `deepseek_v4_flash/` (DSpark A/F) |
| Gate matrices | `tests/e2e/models/*/` scenario lists; `docs/design/module/e2e_testing.md` |

## Feature checks

- Validate `num_attention_ranks` / `num_ffn_ranks` against the actual world
  size on every rank before collectives; a misconfiguration must fail at
  startup with a named mismatch, not deadlock mid-run.
- Keep the per-suite topology contract: DeepSeek-V2-Lite gate cases default to
  the V2 DP2/TP2 families on GPU and 2A2F on NPU; Qwen3 MoE and Qwen3.6 MoE use
  2A1F; DeepEP M2N requires at least two FFN ranks for EP, so its DeepSeek and
  Qwen3 cases run 2A2F (4A2F for 235B FP8). A new case that breaks the suite's
  default needs an explicit reason in the design page.
- Keep the four-device bound (`E2E-INV-002`): a PR case must not exceed four
  unique devices without changing that normative invariant.
- Treat a launch-script change in `recipe/` as a user contract: env vars,
  device counts, and model paths must stay consistent with the E2E scenarios
  and user guides that document them. `VLLM_USE_V2_MODEL_RUNNER=1` is required
  on both roles in every current recipe — a recipe that drops it will fail at
  startup validation.
- Keep baseline comparability: a topology or launch change that affects E2E
  gates must state how `baseline-graph` and AFD scenarios remain comparable.
- Cover the matrix edge the change implies (graph/eager, DBO on/off, DBO +
  2A1F, TP variants) in at least one named scenario or an explicit follow-up.
- Cross-node setups additionally name host/port and rank assumptions; multi-pod
  runners live in `tests/e2e/multi_pod/` and
  `.agents/skills/run-e2e/resources/k8-multi-pod.md`. Flag silent assumptions
  about network interfaces, NVLink domains, or device ordering.

Name the scenario or recipe that exercises the changed topology; for a new
topology, require the E2E case and design-page update together.
