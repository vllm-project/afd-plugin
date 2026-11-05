# AFD test maintenance

Unit tests protect supported AFD behavior at module seams. Test count and coverage
percentage are indicators, not growth or deletion targets. An absence of failures
or an author's familiarity does not establish that a test is unnecessary:
reading a test may itself prevent mistakes.

Follow the existing [Test Conventions](../AGENTS.md#test-conventions) for file
ownership and layout, shared fakes, setup and size limits, and upstream signature
alignment.

- **Test real behavior through the lightest valid path.** Exercise real AFD
  logic; fakes may isolate upstream, transport, or device dependencies, but must
  not replace the behavior being checked. Assert observable results, errors,
  protocol messages, or final state against the intended contract or an
  independent reference. Do not assert private attributes or internal helper
  calls. Assert call order only when the order itself is the contract. Text
  assertions may cover artifacts, not production Python source.

- **Reuse before adding.** Review affected coverage and extend nearby tests,
  fixtures, or parameter tables before creating new tests. Add only meaningful
  missing behavior or boundaries. Existing tests may be sufficient; explain
  their coverage rather than adding a test for every helper, branch, or PR.
  Keep one behavior per test and combine repeated setup through shared helpers.

- **Maintain coverage with the behavior.** Preserve effective coverage through
  refactoring; replace tests coupled to obsolete internals. When behavior is
  removed, delete its dedicated tests and unused helpers in the same PR.
  Keep rejection or compatibility checks only where the current contract
  requires them. Merge duplicates only when the retained tests still detect the
  relevant failures; similar bodies or equal expected values are insufficient.
  Historical regressions and upstream fakes follow the same lifecycle.
  Preserve unique, necessary design rationale in the owning documentation or
  code comments before deleting its only explanation; documentation does not
  replace needed execution evidence.

- **Diagnose failures and demonstrate fixes.** Distinguish a product regression,
  an obsolete expectation, and an environment failure before changing tests.
  For a reproducible bug, show failure on the base commit for the reported
  reason and success with the fix; rebuild affected native code when needed.
  Do not make CI green with retries, weaker checks, relaxed tolerances, or
  deleted necessary coverage. Skipping is not retirement; a temporary skip for
  an unresolved bug needs an issue link.

- **Keep tests deterministic.** Do not use sleeps, wall-clock assertions, or set
  iteration order. Coordinate concurrency with events or bounded conditions.
  Compare exact contracts exactly and justify numerical tolerances.

- **Respect dependency and device boundaries.** Unit tests cover `afd_plugin`;
  validate E2E harness changes through E2E cases. Unit tests must collect without
  torch or vLLM: use `pytest.importorskip` before guarded imports. Device
  correctness belongs in `tests/gpu`, `tests/npu`, or `tests/e2e`, using real
  hardware. CPU fakes and dependency skips do not establish runtime correctness.

- **Make coverage decisions and execution visible.** Use the existing PR Test
  Plan / Test Result sections to explain affected coverage and why it was reused,
  added, replaced, merged, or removed. Record commands, runtime versions, results,
  skips, and pending validation. Ensure CI commands, markers, paths, and
  dependencies select and execute the intended tests. A passing remaining suite
  alone does not justify deletion.
