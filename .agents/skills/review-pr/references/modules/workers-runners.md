# Workers and Model Runners

Primary design: `docs/design/module/attention_runtime.md` and
`docs/design/module/ffn_runtime.md` in the reviewed head.

Use for `afd_plugin/v1/worker/` on both platforms: `AFDAttentionWorker` /
`AFDFFNWorker` and `AFDNPU*Worker`, the model runners (V1 and V2 variants),
attention/FFN metadata builders, `cuda_graph.py`, `dbo.py`, and the ubatch
wrappers. Does not own platform selection
([execution-platforms.md](execution-platforms.md)), connector transport, or
model wrappers ([model-integration.md](model-integration.md)).

## Contract checks

- Preserve the role split: FFN-side `execute_model()` is connector-driven and
  fails fast on scheduler-driven calls; attention-side runners never construct
  FFN components and vice versa.
- Keep runner V1/V2 in pairs: a change to shared runner behavior lands in both
  variants or states the exclusion; GPU keeps V2 gated off
  (`VLLM_USE_V2_MODEL_RUNNER=0` required) while NPU V2 exists — flag any change
  that silently moves that gate.
- Enforce the patch contract on every patched runner method (the NPU
  attention model runner is the heaviest patch site): marked AFD differences,
  upstream-copy fidelity against the pinned ref, identical signatures.
- Keep graph capture sound: metadata must be prepared identically for capture
  and replay; capture-only state must not leak into eager paths; DBO yield and
  ubatch wrappers behave the same under graph and eager.
- Keep the lazy export map in `v1/worker/__init__.py` CPU-safe: dotted-path
  resolution must not import runtime modules eagerly.
- Update the GPU and NPU twins together or prove platform neutrality; the NPU
  runners (largest patch surface) and GPU runners drift independently.

Test role contracts, metadata building, and gating in
`tests/unit/v1/worker/` using the CPU-safe runtime-guard convention (see
[test-quality-evaluation.md](../checks/test-quality-evaluation.md)); name the
GPU/NPU gap for anything not run.
