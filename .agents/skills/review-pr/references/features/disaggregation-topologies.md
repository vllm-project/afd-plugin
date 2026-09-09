# Disaggregation Topologies

Load with the primary module contract whenever a review touches role
placement, rank counts, or launch scripts: attention/FFN colocation vs
prefill-decode disaggregation, 2A2F vs 2A1F rank counts, and the recipe
scripts that encode them.

Designs: `docs/design/module/attention_runtime.md`,
`docs/design/module/ffn_runtime.md`; recipes under `recipe/gpu/` and
`recipe/npu/`.

## Topology map

| Topology | Where it lives |
| --- | --- |
| Baseline (native DP/TP, no AFD) | `baseline-graph` E2E scenarios; native DP4/TP1/EP4 |
| GPU NCCL P2P, colocation + disaggregation | `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/` |
| NPU CAM P2P disaggregation | `recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/` |
| NPU async CAM | `recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/` |
| 2A2F / 2A1F gates | `tests/e2e/models/*/` scenario matrix |

## Feature checks

- Validate `num_attention_ranks` / `num_ffn_ranks` against the actual world
  size on every rank before collectives; a 2A1F misconfiguration must fail at
  startup with a named mismatch, not deadlock mid-run.
- Treat a launch-script change in `recipe/` as a user contract: env vars,
  device counts, and model paths in recipes must stay consistent with the E2E
  scenarios and user guides that document them.
- Keep baseline comparability: a topology or launch change that affects E2E
  gates must state how `baseline-graph` and AFD scenarios remain comparable.
- Cover the matrix edge the change implies (graph/eager, DBO on/off, DBO +
  2A1F) in at least one named scenario or an explicit follow-up.
- Cross-node setups (`tools/itask/launch_dsv4_afd_cross_node.sh`) additionally
  name host/port and rank assumptions; flag silent assumptions about network
  interfaces or device ordering.

Name the scenario or recipe that exercises the changed topology; for a new
topology, require the E2E case and design-page update together.
