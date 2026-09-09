# Active Verification

Use when hardware, a running server, or a runnable affected path is available
and the review needs evidence beyond static reading and CI.

## Choose the narrowest level

| Available environment | Verification |
| --- | --- |
| GPU host with devices free | Narrowest changed unit tests, then the single most relevant E2E scenario via `run-e2e` |
| NPU host | Manual per `docs/npu/TESTING.md`; only what the user explicitly asked to run |
| CPU-only trusted checkout | Import preflight + `pytest tests/unit/<area> -m "not gpu and not vllm_runtime"` |
| Static only (untrusted head, no sandbox) | SHA-addressed reads + existing CI evidence; report execution as a gap |

## Execute safely

1. Reassert the frozen head SHA and snapshot fingerprint immediately before
   running anything.
2. Start from the narrowest command that can change a conclusion; a full E2E
   suite is never the first run.
3. E2E runs go through the `run-e2e` skill with its prerequisite checks
   (devices, model path, `HF_HOME`, mirror reachability); a missing
   prerequisite fails the plan — it never becomes a silent skip.
4. Record every run: head SHA, snapshot fingerprint, command, result, Python
   and platform, dependency fingerprint.
5. Classify failures before drawing conclusions: environment/setup failures
   are not product evidence; a product failure on the changed path is a
   finding; an unrelated pre-existing failure is context, cited not fixed.

Never simulate device evidence: no fabricated NPU numbers, no "should pass"
substitutions for hardware gates. If the gap cannot be closed here, the
finding states exactly which run on which hardware is owed.
