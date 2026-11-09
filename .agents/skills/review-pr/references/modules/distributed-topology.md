# Distributed Topology

Primary design: role lifecycle pages `docs/design/module/attention_runtime.md`
and `docs/design/module/ffn_runtime.md`; connector handoff in
`docs/design/module/connector_contracts.md`.

Use for `afd_plugin/distributed/`: `init_afd_process_group`,
`create_hccl_process_group_options`, and `DefaultProcessGroupSwitcher`
(`afd_process_group.py`), plus `AFDRankMapping`, `topology_from_config`,
`validate_p2p_topology`, `resolve_role_rank`, `subgroup_attention_block`, and
`build_rank_mapping` (`topology.py`). Does not own connector transfer
semantics or worker startup ordering.

## Contract checks

- Keep one owner for AFD process-group creation and switching; workers,
  connectors, and models must not create overlapping groups or bypass the
  switcher.
- Keep rank mappings total and bijective across the attention and FFN worlds;
  `validate_p2p_topology` must reject mismatched world sizes before any
  collective runs.
- Derive backend choice (HCCL vs NCCL) from the detected platform; no silent
  fallback to a different backend.
- Resolve roles and ranks deterministically from the same inputs on every
  rank; any divergence is a startup hang or collective mismatch. Per-process
  role ranks are runtime state resolved from vLLM's parallel placement and are
  deliberately absent from `AFDConfig` — a PR that stores one there is a
  contract change.
- Topology assumptions differ per connector and must stay coherent with them:
  P2P and CAM connectors use FFN-then-Attention world ordering with role ranks
  equal to DP x TP; DeepEP M2N treats the AFD world as one symmetric EP group
  with attention ranks holding dummy expert slots, requires `num_experts` to
  be divisible by the FFN rank count, and lets attention TP ranks act as
  separate union ranks. `afd_process_group_timeout_s` (default 120 s) is the
  rendezvous knob; `hccl_buffer_size` sizes connector-owned HCCL domains only.
- Release AFD groups and restored defaults on shutdown and engine restart; no
  leaked process groups or stale switcher state.

Test mapping construction, validation rejections, and teardown in
`tests/unit/distributed/` (`test_topology.py`, `test_afd_process_group.py`);
name the E2E topology (1A1F, 2A1F, 2A2F, 4A2F, 4A4F, 2P1A1F, or multi-pod)
that covers real collectives.
