# Active Verification

Use when hardware, a running server, or a runnable affected path is available
and the review needs evidence beyond static reading and CI.

## Choose the narrowest level

| Available environment | Verification |
| --- | --- |
| GPU host with devices free | Narrowest changed unit tests, then the single most relevant E2E scenario via `run-e2e` |
| GPU host for connector device tests | `tests/gpu/test_deepep_m2n_gpu.py` on SM90+ with afdEP-comm and DeepGEMM; most cases need 4 GPUs, the Attention-TP cases need 6 |
| NPU host | Manual per `tests/npu/README.md`; only what the user explicitly asked to run |
| CPU-only trusted checkout | Import preflight + `pytest tests/unit/<area> -m "not gpu and not vllm_runtime"` |
| Static only (untrusted head, no sandbox) | SHA-addressed reads + existing CI evidence; report execution as a gap |

## Execute safely

1. Reassert the frozen head SHA and snapshot fingerprint immediately before
   running anything.
2. Start from the narrowest command that can change a conclusion; a full E2E
   suite is never the first run.
3. E2E runs go through the `run-e2e` skill with its prerequisite checks
   (devices, model path, `HF_HOME`, mirror reachability); a missing prerequisite
   fails the plan — it never becomes a silent skip.
4. Record every run: head SHA, snapshot fingerprint, command, result, Python
   and platform, dependency fingerprint, and the vLLM/vLLM-Ascend revisions.
5. Classify failures before drawing conclusions: environment/setup failures
   are not product evidence; a product failure on the changed path is a
   finding; an unrelated pre-existing failure is context, cited not fixed.
   Known open failures in the current tree include DeepSeek-V2-Lite DBO
   shutdown and NPU graceful exit — do not present them as this PR's regression
   without a base-commit comparison.

Never simulate device evidence: no fabricated NPU numbers, no "should pass"
substitutions for hardware gates. If the gap cannot be closed here, the
finding states exactly which run on which hardware is owed.
