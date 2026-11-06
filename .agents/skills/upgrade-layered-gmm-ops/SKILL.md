---
name: upgrade-layered-gmm-ops
description: Replay the A3 layered selection changes onto a new ops-transformer revision for grouped_matmul_layered and grouped_matmul_swiglu_quant_v2_layered, including the ACLNN/pybind adapters and tests. Use when the upstream GMM implementation changes and the AFD NPU operators must be refreshed. Do not use for Ascend 950/A5 support, unrelated GMM operators, or ordinary NPU bug fixes.
---

# Upgrade A3 Layered GMM Operators

Use this skill when a new `ops-transformer` revision changes either source
operator:

- `gmm/grouped_matmul`
- `gmm/grouped_matmul_swiglu_quant_v2`

The target is the AFD plugin's `ascend910_93` / 910C implementation:

- `grouped_matmul_layered` (W2)
- `grouped_matmul_swiglu_quant_v2_layered` (W13)

The migration is a three-way replay. Treat the current AFD layered tree as the
product contract, the old upstream tree as the previous base, and the new
upstream tree as the update. Do not regenerate from the new upstream tree by
blind copying: AFD deliberately drops A5, arch35, unsupported ACLNN variants,
and unsupported quantization columns.

## Required Inputs

Resolve these before editing:

```text
old_upstream: exact previous ops-transformer source tree
new_upstream: exact new ops-transformer source tree
afd_plugin: repository root
upstream_identity: tag, commit, package version, or a recorded snapshot identity
soc: ascend910_93 (910C/A3)
```

The local `ops-transformer-v9.0.1` snapshot is not a git repository. If an
upstream tree has no commit identity, record its version, source archive hash,
or a reproducible directory hash in the migration report. Never describe a
directory-only comparison as a commit-to-commit migration.

Do not include `ascend950`, `arch35`, `grouped_matmul_apt.cpp`, or the A5
`c8afe37` work in this skill. A5 is a separate migration path.

Read the repository `AGENTS.md`, then read:

- `references/layered-invariants.md`
- `references/rename-and-trim-map.md`
- `references/traps.md`
- `references/validation.md`

## Workflow

### 1. Freeze the identities and worktree

Record the AFD branch, `HEAD`, dirty files, old upstream identity, new upstream
identity, and the two source paths. Preserve unrelated worktree changes. Stop
if a planned operator file already contains an overlapping user change.

### 2. Classify before editing

Run the read-only classifier for both operators:

```bash
python .agents/skills/upgrade-layered-gmm-ops/scripts/compare_layered_ops.py \
  classify --kind gmm \
  --upstream "$NEW_UPSTREAM/gmm/grouped_matmul" \
  --afd csrc/npu/ascend_kernels/grouped_matmul_layered

python .agents/skills/upgrade-layered-gmm-ops/scripts/compare_layered_ops.py \
  classify --kind swiglu \
  --upstream "$NEW_UPSTREAM/gmm/grouped_matmul_swiglu_quant_v2" \
  --afd csrc/npu/ascend_kernels/grouped_matmul_swiglu_quant_v2_layered
```

Then run the three-way report against the old source:

```bash
python .agents/skills/upgrade-layered-gmm-ops/scripts/compare_layered_ops.py \
  threeway --kind gmm \
  --old-upstream "$OLD_UPSTREAM/gmm/grouped_matmul" \
  --upstream "$NEW_UPSTREAM/gmm/grouped_matmul" \
  --afd csrc/npu/ascend_kernels/grouped_matmul_layered
```

Repeat with `--kind swiglu`. Every `HAND_MERGE` row requires a semantic
review against the invariant table. `REPLAY_AND_REVIEW` still requires checking
the rename and trim map; the classifier is triage, not authorization to copy.

### 3. Rebase the source tree, then replay the AFD delta

For each upstream file that remains in A3 scope:

1. Start from the new upstream file.
2. Apply the mechanical rename map.
3. Reapply only the layered changes from the invariant table.
4. Reapply the A3 trim decisions.
5. Compare the result with the current AFD file and inspect the complete diff.

Keep the upstream implementation intact around each patch. Mark every new or
changed AFD-specific block with a short marker:

```cpp
// ### PATCH START: select layer from device index
...
// ### PATCH END: select layer from device index
```

The existing `// layered:` comments explain the current port, but future
migrations must use the required `PATCH START/END` form for newly touched
blocks. Do not hide a large upstream rewrite behind delegation or reflection.

### 4. Update the AFD integration surface

Update only when the upstream schema or generated signatures require it:

- `csrc/npu/ascend_kernels/operator_registry.json`
- `csrc/npu/ascend_kernels/CMakeLists.txt` and nested host-source collection
- `csrc/npu/pybind/CMakeLists.txt`
- `csrc/npu/pybind/torch_binding_grouped_matmul.cpp`
- `csrc/npu/pybind/torch_binding_gmm_swiglu_quant.cpp`
- matching ACLNN headers and L0/L2 wrappers under each `op_api/`
- the focused compatibility and NPU test files

The pybind layer is a stable AFD adapter. Refresh its call only if the ACLNN
signature, output contract, schema, or upstream behavior changed. It must keep
Meta, PrivateUse1, and AutogradPrivateUse1 registrations aligned.

### 5. Review the tests as contracts

Update the CPU-collectable Meta/dispatcher tests for schema, list lengths,
layer_index shape/dtype, output shape/dtype, and rejected inputs. Update NPU
precision tests when the supported dtype or quantization contract changes.
Every runtime test must distinguish layer 0 from a nonzero layer with different
weights; using identical layers does not prove selection.

### 6. Run required validation

The required local/remote gate is:

```bash
SOC_VERSION=910c pytest -q tests/unit/package/ tests/npu/test_gmm_layered.py \
  tests/npu/test_gmm_swiglu_quant_v2_layered.py
```

On the Ascend build host, sync the repository without `.git`, then run:

```bash
SOC_VERSION=910c pip install -e . -v --no-build-isolation
```

The host described by the repository notes has no NPU cards. Compilation,
package checks, Meta tests, `ruff`, and shell checks are evidence. They do not
prove numerical device correctness. Record the NPU precision command as
pending unless a real 910C device runs it:

```bash
SOC_VERSION=910c pytest -q \
  tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py \
  tests/npu/test_gmm_layered_device_group_list.py
```

### 7. Report and stop at the correct gate

The migration report must include source identities, the classifier output,
files deliberately excluded, each manually resolved conflict, tests run,
build artifacts, and the exact device tests that remain pending. Do not claim
precision or performance validation from a compile-only host.

## Stop Conditions

Stop before production edits and report the blocker when:

- the new upstream changes the operator input order or meaning outside the
  documented layer contract;
- a new dtype/configuration cannot be classified as supported A3 scope or an
  explicit trim;
- a tiling data layout changes without a matching kernel and generated-schema
  review;
- an upstream change touches an AFD layered anchor and cannot be replayed while
  preserving the invariant table;
- the ACLNN signature no longer matches the pybind schema;
- the required CPU/package gate fails for an unexplained reason.

Do not add compatibility branches for unknown upstream shapes. Resolve the
new upstream contract first or stop.
