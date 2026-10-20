# Rolling vLLM upgrade status

## Current checkpoint

- Updated: 2026-10-20.
- Status: shared branch initialized; no rolling vLLM advancement completed.
- Canonical repository and result remote:
  [vllm-project/afd-plugin](https://github.com/vllm-project/afd-plugin).
- Upgrade branch: `codex/vllm-main-sync` in the canonical repository.
- AFD main incorporated:
  [`d5b9567bbb2cae55d0a80e2541c1c7162ca1ecf9`](https://github.com/vllm-project/afd-plugin/commit/d5b9567bbb2cae55d0a80e2541c1c7162ca1ecf9).
- Inherited vLLM baseline: official `v0.30.0`,
  [`ced6857afa0ea7b2e3f0846a62e1394e90f15607`](https://github.com/vllm-project/vllm/commit/ced6857afa0ea7b2e3f0846a62e1394e90f15607).
- Ascend target remains
  [`8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`](https://github.com/vllm-project/vllm-ascend/commit/8d4409d6256d8a6729140ddcc0d1889e3f96cdd6).
- Skill: [PR #476](https://github.com/vllm-project/afd-plugin/pull/476),
  incorporated through commit `d6fb93627ee3aaec311a3948562fbc4fb6612064`.
  The PR remains subject to its separate review and merge process.
- Production code and dependency pins match the incorporated AFD main.
  Initialization adds only the skill and this record.

## Approved initialization scope and cost

The user requested a long-lived upgrade branch inside the canonical AFD
repository for shared maintenance, carrying the English skill and a public
status record. This approval covers branch initialization and publication only.

- Required checks: skill packaging, metadata, relative references, repository
  pre-commit checks, and the final diff against the incorporated main.
- Initialization result: `quick_validate.py`, metadata/reference checks, and
  repository `pre-commit` passed. The final change contains only the three skill
  files and this status record.
- New runtime preparation or builds: none.
- Accelerator budget and actual use: **0 GPU-minutes; 0 NPU-minutes**.
- New vLLM target and observed main: not selected or sampled during initialization.
  Retain the inherited `v0.30.0` starting point until a maintenance round is
  proposed, approved, implemented, and validated.

## Existing evidence and its limits

[PR #476's test results](https://github.com/vllm-project/afd-plugin/pull/476)
record an independent, bounded GPU trial. Its AFD production source was
`7ace946bb7d9b2fd89b8c57e4df9df4f0a82a45b`. Its experimental vLLM target was
`0.30.0` plus three real upstream cherry-picks, identified locally as
`4a08eaa42cbf1e945cfe3d5c7b716c503b8a9d0e`:

- [`f730a93d`](https://github.com/vllm-project/vllm/commit/f730a93d2bf2cad3002578238d5a1035ee1f4177):
  active-sequence admission cap.
- [`2307a1d1`](https://github.com/vllm-project/vllm/commit/2307a1d16cdd0077cd882bb9fc41b453f6f1999e):
  conditional ready-handshake RPC.
- [`861a299a`](https://github.com/vllm-project/vllm/commit/861a299adc72bbbb1b60853b6173075465a6b425):
  MoE dummy-token routing during profiling.

The trial passed 8 target-contract checks and 217 affected runtime unit cases
(3 NPU-only skips). DeepSeek-V2-Lite `baseline-graph`, `afd-eager-2a1f`, and
`afd-graph-2a1f` each passed the existing 7-sample GSM8K smoke gate; live graph
replay was observed on every participating rank. The runtime used H20-3e GPUs,
Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, and unchanged baseline native
artifacts with verified target Python sources. Full details are in the PR.

Historical allocation cost was 4 GPUs for a cumulative 21 minutes 29 seconds,
including failed attempts, equivalent to about 85.93 GPU-minutes. Resources were
released. This is evidence for that trial, not a default budget for later rounds.

The experimental target is not an official release or a completed rolling
checkpoint. Its code, overlay, and private execution artifacts are not included
in this branch. No new model execution was performed for initialization.
NPU, DBO, other models and topologies, full accuracy, performance, and this
branch's runtime remain unvalidated by initialization. The later target-selection,
approval, and cost-selection instructions have not had an independent forward
trial. Preserve these limits when reusing the historical evidence.

## Next action

On the next manual invocation, inspect both upstream repositories and this record.
Recommend a fixed vLLM target, the AFD main interval to merge, adaptation scope,
available runtime, required affected checks, and an accelerator budget for human
approval. Account for the entire interval since the inherited `v0.30.0` baseline.
Do not adopt the experimental target as the starting point or assume a fixed
release sequence. After approval, maintain this same branch and update the
completed checkpoint, actual cost, evidence, and deferred validation here.
