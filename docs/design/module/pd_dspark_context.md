---
title: DSpark PD auxiliary context
kind: module
status: draft
owners:
  - "@hsliuustc0106"
  - "@jiangkuaixue123"
primary_code_paths:
  - "afd_plugin/distributed/kv_transfer/dspark_context.py"
  - "afd_plugin/distributed/kv_transfer/dspark_mooncake.py"
related_code_paths:
  - "afd_plugin/v1/worker/npu/**"
depends_on:
  - "connector_contracts.md"
  - "model_integration.md"
validation_paths:
  - "tests/unit/distributed/kv_transfer/**"
upstream_refs:
  - "vLLM 0.32.0 5df63a7203cdb9328d824a27b682e77c7b9928be"
  - "vLLM-Ascend bd42eea01d8312c4df469fabaed45f77a12df4d3"
verified_platform_refs: []
related_issues:
  - "#499"
last_reviewed: 2026-10-30
---

# DSpark PD auxiliary context

The CPU-safe helper defines the descriptor for one allocation's auxiliary
prompt tail and maps absolute positions to the actual allocated draft cache
groups. It imports neither PyTorch nor vLLM. The parent connector contract
already owns `tests/unit/distributed/**`; this document records the focused
helper contract and its mirrored tests.

The descriptor carries request ID, allocation generation, actual P-computed
prompt length, a nonempty half-open window, ordered capture boundaries and
hidden size. Its portable encoding preserves those fields without converting
numbers or silently accepting missing identity information. A generation
identifies an allocation, not merely a reused external request ID.

For the native DSV4 Mooncake prompt cut, a prompt of N tokens is prefetched
through N-1 tokens when N is greater than one. The descriptor uses that actual
P length; D recomputes the final prompt token before proposing. Window
positions remain absolute when earlier rows are omitted.

Slot mapping consumes the complete per-group allocation block tables,
including leading null blocks. For each position it selects the block at
`position // block_size`; block zero produces `-1`, matching native DSpark's
padding rule. It never writes physical block zero for an evicted SWA row.
Each output list corresponds to a loaded draft attention layer in the supplied
layer-to-group order. CAM execution layer IDs do not participate in this map.

Transfer completion, cancellation, staging-buffer ownership and device KV
writes belong to the PD adapter and model worker. Pure helper tests establish
descriptor and slot contracts only; real PD and SWA correctness require NPU
end-to-end evidence.

## Native Mooncake carrier extension

`AFDDSparkMooncakeConnector` retains native scheduler admission, cache
registration, transfer engine, peer selection and target block lifetime. P
loads only the target model. The runner configures its native capture layers
after loading, then calls `capture_prefill()` for each prefill chunk. P's
selected TP source accumulates the bounded tail in registered, owned storage.
Unselected TP ranks do not retain redundant tails.

The scheduler extends native handoff parameters with a source allocation
generation, actual P-computed prompt length and complete source block tables.
D records its own allocation generation and complete destination tables.
The native sender socket accepts additional state/acknowledgement messages;
an existing socket and transfer-engine read carry target pages and the tail.
Source leases identify both P and D allocation generations. Retries are
idempotent for that lease; another D allocation cannot release or reuse it.

Target payloads are matched by their actual KV layer names. Read planning uses
the separate P/D base pointers and physical block strides and each layer's
payload size. It excludes draft bytes even if target and draft pages occupy
one packed allocation. Startup rejects noncompact layouts and unsupported
placement rather than inferring that P and D share physical packing.

Before native cache initialization on D, the runner calls
`bind_draft_model()`. The connector waits until native `set_attn()` has bound
the model's real KV groups. Its `get_finished()` advances auxiliary projection
and native DSV4 SWA writes on the model-worker thread, including
`ActiveKVConnector.no_forward()` rounds. Receive completion is delayed until
the device event proves writes finished. Failures are reported through native
failed-receive and invalid-block outputs. Cancellation retains allocated blocks
until the already-submitted reads and writes complete.

The auxiliary pool is independently registered with the existing transfer
engine because its global registration helper accepts only its first call.
`shutdown()` stops new leases, joins the owned socket and receive work, waits
for local device access, unregisters the native owner's target regions, then
unregisters the auxiliary pool. When the engine exposes peer disconnection,
the native helper performs it first; Mooncake 0.3.13 uses its supported
`unregister_memory` API after the same transfer drain. This precedes
native cache tensor release. A live source
lease or receiver-exit failure retains its resources and reports failure.

Initial support is BF16 target activations with the existing W4A8 checkpoint,
eager execution, PP/CP1, Decode TP1 and prefix caching disabled. Prefix reuse,
circular target state layouts and FULL graphs require separate qualification.
The native thread-constructor bindings are replaced only during serialized
startup and restored immediately. This seam can be removed when upstream
provides thread factories and an auxiliary-state message hook.
