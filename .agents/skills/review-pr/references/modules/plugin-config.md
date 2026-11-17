# Plugin Entry and Config

Primary design: `docs/design/module/plugin_boundary.md` in the reviewed head.
This page is `normative`; its `ENTRY-INV-*` and `CFG-INV-*` invariants are
blockers when broken.

Use for plugin registration (`register_afd()`), the `AFDConfig` dataclass and
schema, `parse_afd_config` / `parse_optional_afd_config` /
`validate_afd_config`, extra-config coercion, `afd_plugin/envs.py` flags, and
`afd_plugin/async_cam_startup.py` (the host-only Async CAM graph startup
protocol). Plugin config does not own platform/worker selection, patch
mechanics, or worker construction.

## Contract checks

- Keep `register_afd()` import-time work minimal, idempotent, and
  version-guarded; importing `afd_plugin` must stay CPU-safe (no torch,
  torch_npu, or vLLM runtime imports at module level). The package uses
  module-level `__getattr__` for the runtime classes and
  `_force_spawn_multiprocessing_if_requested()` at import time — that call is
  opt-in behind `AFD_FORCE_SPAWN_MULTIPROCESSING=1` and must stay that way.
- Parse AFD settings only from the `afd` key of `--additional-config`; route
  every derived consumption through validated `AFDConfig` fields, not raw
  dicts, and fail fast on unsupported combinations with an actionable message.
  Short keys are canonical (`connector`, `role`, `port`, `host`,
  `num_attention_ranks`, `num_ffn_ranks`, `compute_gate_on_attention`,
  `async`→`async_dp`, `afd_process_group_timeout_s`,
  `connector_extra_config`); the `afd_*` names survive only as read-only
  compatibility aliases, and unknown keys belong under
  `connector_extra_config`, not at the top level.
- Give every new `AFDConfig` field a default, validation, a stated role and
  platform implication (attention vs FFN, GPU vs NPU), and a docs update;
  changing a default or meaning of an existing field is a user-contract change.
  `compute_hash()` covers only graph-affecting fields — a new field that
  affects graph capture must join it or capture will be reused wrongly.
- Read `envs.py` flags at use time, keep them opt-in and documented, and never
  let an env flag silently override validated config in production paths.
  Today `envs.py` holds a single diagnostic flag
  (`AFD_FORCE_BALANCED_TOPK_IDS`); a new runtime-behavior flag needs a design
  and guide entry, not just a constant.
- Keep the plugin entry point (`vllm.general_plugins` → `afd_plugin:register_afd`)
  the only installation path; no side effects from importing submodules.
- `async_cam_startup.py` is host-only and must not import device runtimes: it
  is imported on both roles before and after HCCL setup.

Test registration idempotence, config parse/validate rejections, defaults, and
CPU-safe import in `tests/unit/config/` (`test_config.py`,
`test_validation.py`), `tests/unit/test_envs.py`,
`tests/unit/test_async_cam_startup_wiring.py`, and
`tests/unit/package/test_package.py`.
