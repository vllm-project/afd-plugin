# Plugin Entry and Config

Primary design: `docs/design/module/plugin_boundary.md` in the reviewed head.

Use for plugin registration (`register_afd()`), the `AFDConfig` dataclass and
schema, `parse_afd_config`/`validate_afd_config`, extra-config coercion, and
`afd_plugin/envs.py` flags. Plugin config does not own platform/worker
selection, patch mechanics, or worker construction.

## Contract checks

- Keep `register_afd()` import-time work minimal, idempotent, and
  version-guarded; importing `afd_plugin` must stay CPU-safe (no torch,
  torch_npu, or vLLM runtime imports at module level).
- Parse AFD settings only from the `afd` key of `--additional-config`; route
  every derived consumption through validated `AFDConfig` fields, not raw
  dicts, and fail fast on unsupported combinations with an actionable message.
- Give every new `AFDConfig` field a default, validation, a stated role and
  platform implication (attention vs FFN, GPU vs NPU), and a docs update;
  changing a default or meaning of an existing field is a user-contract change.
- Read `envs.py` flags at use time, keep them opt-in and documented, and never
  let an env flag silently override validated config in production paths.
- Keep the plugin entry point (`vllm.general_plugins`) the only installation
  path; no side effects from importing submodules.

Test registration idempotence, config parse/validate rejections, defaults, and
CPU-safe import in `tests/unit/config/` and `tests/unit/package/`.
