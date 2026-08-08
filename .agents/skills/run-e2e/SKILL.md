---
name: run-e2e
description: Run the AFD plugin's e2e test suite — the tests marked @pytest.mark.gpu or @pytest.mark.npu (51 tests: 30 GPU + 21 NPU) under tests/e2e in the afd-plugin repo. Invoke when the user wants to run these e2e/GPU/NPU tests and says things like "test all", "test accuracy limit 5", "test features", "test models", "跑 e2e 测试", "跑 gpu/npu 测试". Auto-detects the hardware backend (GPU vs NPU), validates the toolchain, provisions model/device paths, and runs pytest by marker. Do NOT invoke for unit tests, for test_runner.py, or for "test this function" style requests.
---

# Run AFD E2E Tests

Runs the AFD plugin e2e suite **by marker** (`-m "gpu or npu"`). The suite has
30 GPU and 21 NPU tests. The GPU set contains 20 shared/DeepSeek tests using
`AFD_GPU_E2E_MODEL` and 10 Qwen3 MoE tests using
`AFD_QWEN3_MOE_E2E_MODEL`. `tests/e2e/features/test_ops_npu.py` is
intentionally out of scope.

## Hard constraints

- **Tests-only.** Never edit anything under `afd_plugin/` (source). You may only
  read tests and run commands. If a test is broken in a way that needs a source
  fix, stop and tell the user — do not patch source.
- **`lm_eval` must never be added to `pyproject.toml` / `uv.lock`.** It is
  deliberately absent (re-adding re-breaks `uv sync --locked`). If accuracy needs
  it and it's missing, tell the user to `pip install lm-eval` standalone in their
  env. See "Pre-flight validation" below.

## Trigger & args grammar

No slash command — triggered by natural language. Parse the user's message:

| Message | category | gsm8k limit |
|---|---|---|
| `test` | (ask) | (ask if accuracy) |
| `test all` | all (includes accuracy) | ask full/custom |
| `test accuracy` | accuracy | ask full/custom |
| `test accuracy limit 5` | accuracy | 5 |
| `test all limit 100` | all | 100 |
| `test features` / `test models` | that category | n/a (no gsm8k) |

Backend is **never** an argument — always auto-detected from hardware.
`limit N` only matters when accuracy is in scope; if given for features/models,
note "limit ignored — that category has no gsm8k".

## Workflow

### 1. Parse the request
Extract `category` ∈ {all, accuracy, features, models} and optional `limit N`
(1–1319). If category is absent, ask with AskUserQuestion.

### 2. Detect backend (gpu vs npu)
Probe the machine. Run these (don't fail hard; gather what's there):
- GPU: `command -v nvidia-smi && nvidia-smi -L` → GPU count & names. Fallback:
  `python -c "import torch; print(torch.cuda.device_count())"`.
- NPU: `command -v npu-smi && npu-smi info`. Fallback: `python -c "import torch_npu; print('ok')"`
  and `echo $ASCEND_RT_VISIBLE_DEVICES`.

Decide:
- Only GPU → backend = gpu.
- Only NPU → backend = npu.
- Both → ask the user which to run.
- Neither → STOP. This box can't run e2e. Tell the user clearly (e.g. "no CUDA
  and no Ascend runtime detected — run this on an L20/910C box").

### 3. Pre-flight validation
Run the checklist below for the detected backend. The point is to turn a silent
"20 skipped" into an actionable "you're missing X". Gather status, then report a
short pre-flight summary before running.

**Always (both backends):**
- `vllm` binary runs: `vllm --version` (or `$AFD_GPU_E2E_VLLM_BIN` / `$AFD_NPU_E2E_VLLM_BIN`, default `vllm`).
- AFD plugin loadable by vllm: `python -c "import afd_plugin"` works (PYTHONPATH=repo root, or installed). `build_env` sets `VLLM_PLUGINS=afd` (gpu) / `ascend,afd` (npu).
- Every configured model path (`AFD_*_E2E_MODEL`) exists on disk.
- pytest importable.
- Hardware count meets tier (see step 6).

**GPU-only:**
- `uv` present + project synced: `uv --version` and `.venv/` exists (GPU path runs `uv run pytest`).
- CUDA visible + GPU count (nvidia-smi).
- ≥2 GPUs for 1A1F; 4 for TP/2A2F.

**NPU-only:**
- `torch_npu` importable (the `npu_available` fixture skips otherwise).
- Ascend runtime: `npu-smi` works, `ASCEND_RT_VISIBLE_DEVICES` set, CANN present.
- attn + ffn device IDs (`AFD_NPU_ATTN_DEVICES` default 0, `AFD_NPU_FFN_DEVICES` default 1).
- ≥2 NPUs for 1A1F; 4 for TP/2A2F.

**accuracy / gsm8k only (category includes accuracy):**
- `lm_eval` installed: `python -c "import lm_eval"`. If missing → tell user to
  `pip install lm-eval` **standalone** (never add to pyproject/uv.lock).
- (NPU) offline task dir `AFD_NPU_GSM8K_TASK_DIR` exists — if unset, try the
  known default `/root/.cache/gsm8k`; ask the user if neither is present.
- `AFD_GSM8K_THRESHOLD` (0.20) / `AFD_GSM8K_TOLERANCE` (0.05) have defaults — optional.

If a **required** prereq is missing and the user can't trivially fix it, stop and
report rather than running a guaranteed-all-skip batch.

### 4. Provision (read env first, ask only what's missing)
Check these env vars; if a required one is unset, ask the user to input it:
- GPU shared/DeepSeek tests: `AFD_GPU_E2E_MODEL`.
- GPU Qwen3 MoE tests: `AFD_QWEN3_MOE_E2E_MODEL`.
- GPU devices: `AFD_GPU_E2E_GPUS` (default `0,1,2,3`).
- NPU: `AFD_NPU_E2E_MODEL` (required), `AFD_NPU_ATTN_DEVICES` (default 0),
  `AFD_NPU_FFN_DEVICES` (default 1).
- accuracy NPU: `AFD_NPU_GSM8K_TASK_DIR`.

For `all` or `models`, run every configured model family and report the other
family as skipped when its model variable is absent. For `accuracy` or
`features`, `AFD_GPU_E2E_MODEL` remains required.

Use AskUserQuestion for discrete choices (category, full-vs-custom gsm8k). For
free-form paths/device-lists, ask in prose so the user can paste a path.

### 5. gsm8k count (only if accuracy is in scope: category = accuracy or all)
If the message already gave `limit N`, use it. Otherwise ask:
- **Full** → leave `AFD_GSM8K_LIMIT` unset (runs all 1319).
- **Custom** → user enters N in 1–1319 → set `AFD_GSM8K_LIMIT=N`.

### 6. Hardware tier prediction
Tell the user how many collected tests will actually run vs skip. Account for
missing model-family variables as well as device count:
- **2 devices** (e.g. L20 dual-card): 1A1F tests run (graph/profiler/serving +
  accuracy + 1A1F model tests); **TP and 2A2F tests skip**.
- **4 devices**: all run.
State this explicitly so "X skipped" later is not a surprise.

### 7. Run (marker path)
Set env from steps 4–5, then:

**GPU:**
```bash
cd /path/to/afd-plugin
[AFD_GPU_E2E_MODEL=<shared-or-deepseek-model>] \
  [AFD_QWEN3_MOE_E2E_MODEL=<qwen-model>] \
  AFD_GPU_E2E_GPUS=<gpus> \
  [AFD_GSM8K_LIMIT=<N>] \
  uv run pytest -m gpu tests/e2e/<category>
```

**NPU:**
```bash
cd /path/to/afd-plugin
AFD_NPU_E2E_MODEL=<model> AFD_NPU_ATTN_DEVICES=<d0> AFD_NPU_FFN_DEVICES=<d1> \
  [AFD_GSM8K_LIMIT=<N>] [AFD_NPU_GSM8K_TASK_DIR=<dir>] \
  python -m pytest -m npu tests/e2e/<category>
```

`<category>` path mapping: `all` → `tests/e2e` (the `-m` filter picks the right
tests; `test_runner.py` in the root is unmarked and excluded automatically),
`accuracy` → `tests/e2e/accuracy`, `features` → `tests/e2e/features`,
`models` → `tests/e2e/models`.

Stream the run live (don't swallow output). e2e servers take minutes to start;
keep `startup_timeout` in mind (fixtures allow ~900s).

### 8. Report
Parse the pytest summary. Report:
- counts: passed / failed / errors / skipped.
- **why each skipped** — map back to the prereq it was missing (model unset,
  <2 GPUs, lm_eval missing, DBO-on-NPU, etc.). This is the main value-add over
  a bare "20 skipped".
- any failures: surface the failing test id + the relevant log tail.
- note whether servers were cleaned up (the fixtures tear down on exit; if a run
  was interrupted, mention checking for stray `vllm serve` processes).

## Env var reference (what the tests actually read)

| Var | Backend | Default | Required? |
|---|---|---|---|
| `AFD_GPU_E2E_MODEL` | gpu | — | required for shared/DeepSeek tests |
| `AFD_QWEN3_MOE_E2E_MODEL` | gpu | — | required for the 10 Qwen3 MoE tests |
| `AFD_GPU_E2E_GPUS` | gpu | `0,1,2,3` | no |
| `AFD_GPU_E2E_VLLM_BIN` | gpu | `vllm` | no |
| `AFD_NPU_E2E_MODEL` | npu | — | yes (else all npu skip) |
| `AFD_NPU_ATTN_DEVICES` | npu | `0` | no |
| `AFD_NPU_FFN_DEVICES` | npu | `1` | no |
| `AFD_NPU_VLLM_BIN` | npu | `vllm` | no |
| `AFD_GSM8K_LIMIT` | accuracy | unset=full(1319) | no |
| `AFD_GSM8K_THRESHOLD` | accuracy | `0.20` | no |
| `AFD_GSM8K_TOLERANCE` | accuracy | `0.05` | no |
| `AFD_NPU_GSM8K_TASK_DIR` | npu accuracy | — | yes for npu gsm8k |

## Quick reference: what runs where

- **GPU box (L20 etc.)**: `-m gpu` → up to 30 tests across configured models.
- **NPU box (910C etc.)**: `-m npu` → up to 21 tests (DBO variants self-skip; DBO
  not supported on NPU yet).
- **Dev box, no hw**: stop at detection — can't run e2e.
