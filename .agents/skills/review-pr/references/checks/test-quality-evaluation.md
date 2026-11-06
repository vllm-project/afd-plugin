# Test Quality Evaluation

Use when tests change in the diff, are absent for risky changed code, or may
pass without exercising production behavior. Standards live in AGENTS.md
("Test Conventions") and `docs/UT_MAINTENANCE.md`; this file is the reviewer's
application of them.

## CPU-safe imports first

The unit suite must run on a CPU-only host without torch, torch_npu, or vLLM.
Enforce:

- Runtime dependencies (`vllm`, `vllm_ascend`, `torch_npu`, `torch`) are
  gated with `pytest.importorskip` (or the `_require_npu_runtime()` helper in
  `tests/unit/v1/worker/test_npu_runtime.py`) **before** the guarded module is
  imported — inside each test or a module-level guard that runs before the
  import, never after.
- Module-level imports in test files stay CPU-safe; a top-level
  `from afd_plugin.v1.worker... import ...` in a test file breaks collection
  on CPU hosts even if every test body is guarded. This is the recurring
  regression class; check it before anything else.
- New tests pass in the CI selection: `pytest tests/unit -m "not gpu and not
  vllm_runtime"`, plus the four CPU-only `tests/npu` modules CI names
  (`test_async_cam_harness.py`, `test_async_cam_performance.py`,
  `test_async_cam_golden.py`, `test_gmm_layered_performance.py`). Run (or
  reason through) the selection for the changed directory.

## Markers and selection

| Marker | Meaning | Misuse to flag |
| --- | --- | --- |
| `vllm_runtime` | Needs importable vLLM runtime | Used to hide a test that should be CPU-safe |
| `gpu` / `npu` | Hardware-gated integration | Applied to tests that never touch the device (or vice versa) |
| `e2e` | Full stack, weights + hardware | Used for mocked tests |
| `eval` | Datasets/accuracy | Missing dataset guard |
| `slow` | > 120 s | Unmarked long tests that inflate CI time |

Note that `tests/npu` is not uniformly device-gated: several modules there are
CPU host-harness checks that CI runs directly. Do not demand an `npu` marker on
those, and do not assume a `tests/npu` path implies hardware.

## Quality bar

- Assert the seam, not the plumbing: return values, raised errors and their
  messages, protocol messages sent to a peer, final file or environment state,
  config-validation outcomes. Private attributes, internal-collaborator
  arguments, and internal helper call counts are forbidden. Call order is a
  valid assertion only when the order is itself the contract (connector
  handshakes, service announce, DBO stage order).
- The refactor test: if the implementation is restructured without changing a
  protocol, the test must stay green. A test that cannot fail when behavior
  regresses is not coverage — flag it and ask for deletion rather than
  retention.
- Tests exercise production behavior, not a mock echo: patched functions get
  AFD-on and AFD-off assertions; state machines get abort/timeout paths;
  fail-fast contracts assert the error message, not just `pytest.raises`.
- Shared fakes live in `tests/unit/fakes/`, one module per upstream concept; a
  builder defined in two test modules is a layout bug. A test patching more
  than ~4 attributes of the module under test is testing its own mocks.
- Setup over ~15 lines belongs in a fixture or factory; a test body over ~50
  lines needs `# long-test: <reason>`; the third test with the same prefix and
  setup becomes a `parametrize` table.
- Deterministic: no sleeps, no wall-clock assertions, no reliance on set
  iteration order. Concurrency is coordinated with events or bounded
  conditions. Numerical tolerances are justified.
- Text artifacts (build manifests, patch markers, docs links) may be asserted
  as text. Production Python modules may not: no test reads
  `afd_plugin/**/*.py` and asserts on a substring.
- File placement mirrors the source tree under `tests/unit/<mirrored path>/`,
  and a new test file must fall under an existing `validation_paths` entry in
  `docs/design/module/`. A test file owned by no module is a finding — or, if
  it is a full-stack check, it belongs in `tests/e2e`.
- Upstream signature alignment stays in the single table in
  `tests/unit/compat/patches/test_signatures.py`; inheritance alignment is
  mypy's job, not pytest's.
- Device-independent logic (planners, rank mapping, config validation, layered
  layout math) is tested CPU-side with `parametrize`; device-only behavior is
  honestly delegated to E2E or `tests/gpu`/`tests/npu` rather than
  fake-simulated in unit tests.

## Lifecycle

- A PR that removes behavior deletes its tests and now-unused helpers in the
  same PR. Skipping is not retirement; a skip needs an issue link.
- Merging duplicate tests is acceptable only when the retained test still
  detects the relevant failures; similar bodies or equal expected values are
  not sufficient justification.
- Never make CI green with retries, weaker checks, relaxed tolerances, or
  deleted necessary coverage. A gate case turned into a skip is a P1.

## Finding severity

A risky change without a test that would protect it is a P2 finding (P1 if the
untested path already regressed before); name the specific behavior to cover
and the cheapest harness for it. Do not demand coverage the repo's structure
already provides elsewhere (E2E gates, package tests, upstream CI).
