# Connectors

Primary design: `docs/design/module/connector_contracts.md` in the reviewed
head; backend guides in `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`,
`docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md`,
`docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md`, and
`docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`.

Use for `afd_plugin/connectors/`: `AFDConnectorBase`,
`AFDRoutedExpertsConnector`, `AFDControlPlane`, `ConnectorExtraInfo`,
`AFDConnectorFactory`, and the concrete backends — `gpu/p2p.py` (NCCL P2P),
`gpu/deepep_m2n/` (DeepEP routed-experts M2N), `npu/camp2p.py`
(HCCL/CAM P2P), and `npu/async_cam.py` (async CAM). Connectors do not own
process-group creation or rank mapping ([distributed-topology.md](distributed-topology.md)),
model-side proxies, or role workers.

## Contract checks

- Register backends through `AFDConnectorFactory`; role workers, runners, and
  models must stay connector-agnostic and never branch on a concrete
  connector type. The factory stores lazy loaders, so importing it must not
  import CUDA or Ascend implementations. Registration alone does not admit an
  external connector: `AFDConfig` validates `connector` against its own
  allow-list in `afd_plugin/config.py`, and the design page records that gap as
  draft. A PR claiming "extensible connectors" must close both.
- Change both sides of the transfer boundary together: `ConnectorExtraInfo`
  handshake, tensor shapes/dtypes/layout, and completion signaling must agree
  between attention and FFN implementations, including metadata versioning
  across mixed-version ranks.
- Preserve failure semantics on the transfer path: timeout, retry,
  cancellation, and close must surface as loud errors, not a silent hang of
  both roles; every state machine handles abort before completion.
- Declare graph-capture compatibility per connector (for example
  `FULL_DECODE_ONLY` CUDA graph constraints for the CUDA connectors, ACL graph
  constraints for CAM) and encode unsupported combinations in
  `feature_validation`, not runtime surprises.
- Keep `connector_extra_config` connector-owned: unknown keys must fail in the
  selected connector's parser, not be silently ignored. Adding a key to one
  backend is a user-contract change that needs the guide and design-page table
  updated together.

## Routed-experts transports

`DeepEPM2NAFDConnector` extends the contract through
`AFDRoutedExpertsConnector`: routing runs on Attention, FFN ranks receive one
row per (token, local expert) pair, run only local experts via `run_experts()`,
and send weighted rows back. `synchronize_lanes()` joins connector-forked
streams at the end of each step. When reviewing changes here:

- The FFN worker selects `GPUFFNRoutedExpertsModelRunner` from the connector
  class, not from the connector name; keep that selection type-based.
- The runner pins vLLM's MoE backend to `triton` because other backends
  reorder expert weights in place without changing dtype or shape, producing
  wrong outputs with no error. A new backend needs a layout proof, not an
  allow-list entry.
- Dispatch/combine are collectives over all ranks, including ranks with no
  tokens for a layer; a zero-length split is valid, an empty list is not.
- Weight loading must report what each rank actually loaded through
  `init_afd_connector(moe_layer_indices=, local_experts=)`; skipping the
  post-load check silently admits mismatched expert placement.

## Vendored Ascend artifacts

Additions under `connectors/npu/bin/` (`CAM_ascend*_openEuler_*.run`,
`umdk_cam_op_lib-*.whl`) are design-level changes: require source, version, and
target CANN/Ascend runtime justification. They are pre-commit-excluded by
design, which is not a license to skip review.

Test each backend in `tests/unit/connectors/` (state machines, handshake,
failure paths, CPU-safe), name the device suite (`tests/gpu/` for M2N,
`tests/npu/` for CAM) and the E2E gate scenario that exercises the real
transport.
