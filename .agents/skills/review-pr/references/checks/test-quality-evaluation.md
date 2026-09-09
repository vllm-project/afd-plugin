# Test Quality Evaluation

Use when tests change in the diff, are absent for risky changed code, or may
pass without exercising production behavior.

## CPU-safe imports first

The unit suite must run on a CPU-only host without torch. The recurring
regression class (fixed in `f5fa9fc`) is a test that imports a worker or
connector module before guarding its runtime dependencies. Enforce:

- Runtime dependencies (`vllm`, `vllm_ascend`, `torch_npu`, `torch`) are
  gated with `pytest.importorskip` (or the `_require_npu_runtime()` helper in
  `tests/unit/v1/worker/test_npu_runtime.py`) **before** the guarded module is
  imported — inside each test or a module-level guard that runs before the
  import, never after.
- Module-level imports in test files stay CPU-safe; a top-level
  `from afd_plugin.v1.worker... import ...` in a test file breaks collection
  on CPU hosts even if every test body is guarded.
- New tests pass in the CI selection: `pytest tests/unit -m "not gpu and not
  vllm_runtime"`. Run (or reason through) the selection for the changed
  directory.

## Markers and selection

| Marker | Meaning | Misuse to flag |
| --- | --- | --- |
| `vllm_runtime` | Needs importable vLLM runtime | Used to hide a test that should be CPU-safe |
| `gpu` / `npu` | Hardware-gated integration | Applied to tests that never touch the device (or vice versa) |
| `e2e` | Full stack, weights + hardware | Used for mocked tests |
| `eval` | Datasets/accuracy | Missing dataset guard |
| `slow` | > 120 s | Unmarked long tests that inflate CI time |

## Quality bar

- Tests exercise production behavior, not a mock echo: patched functions get
  AFD-on and AFD-off assertions; state machines get abort/timeout paths;
  fail-fast contracts assert the error message, not just `pytest.raises`.
- Device-independent logic (planners, rank mapping, config validation) is
  tested CPU-side with `parametrize`; device-only behavior is honestly
  delegated to E2E scenarios rather than fake-simulated in unit tests.
- A risky change without a test that would protect it is a P2 finding (P1 if
  the untested path already regressed before); name the specific behavior to
  cover and the cheapest harness for it.
- Do not demand coverage the repo's structure already provides elsewhere
  (E2E gates, package tests, upstream CI).
