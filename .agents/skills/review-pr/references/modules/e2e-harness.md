# E2E Harness and Gate Selection

Primary design: `docs/design/module/e2e_testing.md` in the reviewed head. That
page is `normative`; its `E2E-INV-001` … `E2E-INV-008` invariants are
blockers when broken. Author-side runner: the `run-e2e` skill.

Use for `tests/e2e/**`, the Buildkite case selection in
`.buildkite/cuda/*.yml`, `.buildkite/common/scripts/skip_ci.py` as it affects
what runs, and the `run-e2e` / `deploy-dsv4-flash-async` skill text that
describes cases.

## Contract checks

- Case IDs are stable lower-kebab-case and unique (`E2E-INV-001`). Renaming a
  case silently breaks every CI node ID and every recipe/README that names it;
  require the rename to land with the CI, docs, and skill updates.
- Gate cases never skip (`E2E-INV-004`): no `skip`, `xfail`, or
  success-on-empty. A new `pytest.importorskip` or a conditional `return` in a
  gate case is a P1. Missing setup must fail.
- CI selects cases by pytest node ID and must not lower sample counts,
  thresholds, or the required set (`E2E-INV-007`). Compare
  `.buildkite/cuda/test-ready.yml` / `test-merge.yml` / `test-weekly.yml`
  against the module's `SCENARIOS` tuple: a case added to the module but not
  selected (or selected but renamed) is drift worth reporting.
- The current gate set, which a change must keep coherent:
  - DeepSeek-V2-Lite GPU: seven cases — `baseline-graph`, `afd-v2-eager-dp2`,
    `afd-v2-graph-dp2`, `afd-v2-eager-tp2`, `afd-v2-graph-tp2`,
    `afd-v2-eager-dbo-dp2`, `afd-v2-graph-dbo-dp2`; the module also holds
    local-only `afd-v2-{eager,graph}-1a1f`, `afd-v2-graph-dbo-tp2`, and the
    `afd-{eager,graph,graph-dbo}-2a1f` cases.
  - DeepSeek-V2-Lite NPU: `baseline-graph` plus `afd-eager-2a2f`,
    `afd-graph-2a2f`, `afd-graph-dbo-2a2f` (selected by
    `AFD_E2E_BACKEND=npu`).
  - Qwen3 MoE GPU: `baseline-graph`, `afd-eager-2a1f`, `afd-graph-2a1f`,
    `afd-graph-dbo-2a1f`. The module also collects the DeepEP M2N 2A2F
    scenarios (`afd-deepep-m2n-{eager,graph,graph-dbo}-2a2f` and the two
    Attention-TP variants) through `test_qwen3_moe_deepep_m2n`; CI selects node
    IDs explicitly and does not run them.
  - Qwen3.6 MoE GPU: `baseline-graph`, `afd-eager-2a1f`, `afd-graph-2a1f`
    (GDN DBO unsupported).
  - DeepSeek-V4 Flash: separate `test_gpu.py`, `test_async_cam_npu.py`, and
    `test_sync_camp2p_npu.py` suites; DSV4 Flash W4A8 async CAM is a local-only
    16-NPU case.
- Accuracy plumbing: ordinary cases evaluate the first 7 GSM8K samples, DBO
  cases require a 24-sample minimum, and the GPU DP2 eager/eager-DBO/graph-DBO
  comparison keeps 128 samples, concurrency 12, `max_num_seqs=8`, and the 0.27
  accuracy threshold. A change here is a gate change.
- Execution evidence: GPU and NPU graph/DBO cases load a test-only worker
  extension and must report completed replays or two-stage execution on every
  participating rank; dummy, profile, capture, and empty-second-microbatch work
  does not qualify. Pure eager and baseline cases must not load the extension,
  and performance comparisons must omit it.
- Process hygiene: child processes use owned process groups; cancellation is
  SIGTERM, bounded grace, then reaping. The NPU Async CAM marker-based SIGKILL
  sweep and its 120 s survivor window are a documented test-scoped exception,
  not a general strategy.
- Harness code is not unit-tested (`tests/unit` covers `afd_plugin` only);
  validate harness changes by running the affected E2E cases or
  `tests/e2e/test_runner.py` / `test_deploy_dsv4_flash_async_skill.py`.
  Production code must never import the harness.

A PR that changes a case, sample count, device layout, or gate selection
without updating `docs/design/module/e2e_testing.md` and the `run-e2e` skill
narrative is a P2 (P1 when the change weakens a blocking gate).
