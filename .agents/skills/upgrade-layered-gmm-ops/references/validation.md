# Validation

## Required CPU and package checks

Run from the repository root. The unit tests must collect without torch, vLLM,
or an Ascend device unless the test explicitly guards those imports.

```bash
python .agents/skills/upgrade-layered-gmm-ops/scripts/compare_layered_ops.py \
  classify --kind gmm --upstream "$NEW_UPSTREAM/gmm/grouped_matmul" \
  --afd csrc/npu/ascend_kernels/grouped_matmul_layered
python .agents/skills/upgrade-layered-gmm-ops/scripts/compare_layered_ops.py \
  classify --kind swiglu --upstream "$NEW_UPSTREAM/gmm/grouped_matmul_swiglu_quant_v2" \
  --afd csrc/npu/ascend_kernels/grouped_matmul_swiglu_quant_v2_layered
pytest -q tests/unit/package/ \
  tests/npu/test_gmm_layered.py \
  tests/npu/test_gmm_swiglu_quant_v2_layered.py
ruff check .agents/skills/upgrade-layered-gmm-ops
```

Run the repository's normal package checks after source changes:

```bash
pytest -q tests/unit/package/
```

## Ascend compilation

Sync the working tree to the documented remote host with `rsync`; do not use
git to move the source. Build only A3:

```bash
SOC_VERSION=910c pip install -e . -v --no-build-isolation
```

Check that both operator names occur in the selected build and that the swiglu
tiling sources under `op_host/op_tiling/` were collected recursively. The
packaged `libcust_optiling.so` should contain the layered registration names.

## Device evidence

When a real 910C device is available, run the focused tests:

```bash
SOC_VERSION=910c pytest -q \
  tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py \
  tests/npu/test_gmm_layered_device_group_list.py
```

The precision cases must include multiple layers with distinct weights and a
nonzero layer index. Performance output is informative and cannot replace
precision evidence. A remote compilation container without NPU cards can
provide build and static evidence only.

## Required report fields

```text
old_upstream_identity:
new_upstream_identity:
afd_head:
gmm_three_way_summary:
swiglu_three_way_summary:
excluded_upstream_files:
manual_conflicts:
cpu_tests:
ascend_build:
device_precision: PASS | PENDING | FAIL
known_risks:
```
