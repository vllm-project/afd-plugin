# Distributed Topology

Primary design: role lifecycle pages `docs/design/module/attention_runtime.md`
and `docs/design/module/ffn_runtime.md`; connector handoff in
`docs/design/module/connector_contracts.md`.

Use for `afd_plugin/distributed/`: `init_afd_process_group` and the
`DefaultProcessGroupSwitcher` (`afd_process_group.py`), `AFDRankMapping`,
`validate_p2p_topology`, `resolve_role_rank`, and `build_rank_mapping`
(`topology.py`). Does not own connector transfer semantics or worker startup
ordering.

## Contract checks

- Keep one owner for AFD process-group creation and switching; workers,
  connectors, and models must not create overlapping groups or bypass the
  switcher.
- Keep rank mappings total and bijective across the attention and FFN worlds;
  `validate_p2p_topology` must reject mismatched world sizes (2A2F/2A1F)
  before any collective runs.
- Derive backend choice (HCCL vs NCCL) from the detected platform; no silent
  fallback to a different backend.
- Resolve roles and ranks deterministically from the same inputs on every
  rank; any divergence is a startup hang or collective mismatch.
- Release AFD groups and restored defaults on shutdown and engine restart; no
  leaked process groups or stale switcher state.

Test mapping construction, validation rejections, and teardown in
`tests/unit/distributed/`; name the E2E topology (2A2F or 2A1F) that covers
real collectives.
