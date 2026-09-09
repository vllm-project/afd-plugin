# Connectors

Primary design: `docs/design/module/connector_contracts.md` in the reviewed
head; backend guides in `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md` and
`docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md` / `docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md`.

Use for `afd_plugin/connectors/`: `AFDConnectorBase`, `AFDControlPlane`,
`ConnectorExtraInfo`, `AFDConnectorFactory`, and the concrete backends —
`gpu/p2p.py` (NCCL P2P), `npu/camp2p.py` (HCCL/CAM P2P), `npu/async_cam.py`
(experimental async CAM). Connectors do not own process-group creation or rank
mapping ([distributed-topology.md](distributed-topology.md)), model-side
proxies, or role workers.

## Contract checks

- Register backends through `AFDConnectorFactory`; role workers, runners, and
  models must stay connector-agnostic and never branch on a concrete
  connector type.
- Change both sides of the transfer boundary together: `ConnectorExtraInfo`
  handshake, tensor shapes/dtypes/layout, and completion signaling must agree
  between attention and FFN implementations, including metadata versioning
  across mixed-version ranks.
- Preserve failure semantics on the transfer path: timeout, retry,
  cancellation, and close must surface as loud errors, not a silent hang of
  both roles; every state machine handles abort before completion.
- Declare graph-capture compatibility per connector (for example
  FULL_DECODE_ONLY CUDA graph for P2pNccl, ACL graph constraints for CAM) and
  encode unsupported combinations in `feature_validation`, not runtime
  surprises.
- Treat additions under `connectors/npu/bin/` (vendored `.run`/`.whl`
  artifacts) as design-level changes: require source, version, and target
  CANN/Ascend runtime justification; they are pre-commit-excluded by design,
  which is not a license to skip review.

Test each backend in `tests/unit/connectors/` (state machines, handshake,
failure paths, CPU-safe) and name the E2E gate scenario that exercises the
real transport.
