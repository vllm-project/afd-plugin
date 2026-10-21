# CAM Runtime Profiling: Design and Usage

Runtime profiling captures a shared window across `CAMAsyncAFDConnector`
Attention and FFN workers. Use the [connector guide](CAM_ASYNC_CONNECTOR_USER_GUIDE.md)
and [pinned installation stack](../../README.md#ascend-npu-installation) to
launch inference before following the capture example below.

Profiling `mode="FULL"` selects detailed collection; FFN
`cudagraph_mode="FULL"` selects [graph execution](ASYNC_FFN_RUNTIME.md).
Either profiling mode can capture eager FFN or supported MRV2 FFN FULL replay.
Attention remains eager.

## Module relationships

Modules live in [`afd_plugin/profiling`](../../afd_plugin/profiling).

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Device-independent enablement, duration limits, metadata size and host clock identity. |
| `control.py` | HTTP coordinator: discovery, clock calibration, prepare/commit, polling, cleanup and manifests. |
| `endpoint.py` | Authenticated vLLM routes forwarding commands through collective worker RPC. |
| `worker.py` | Attention/FFN session wiring, RPC handling and profiler exclusivity. |
| `session.py` | Capture generations, command ordering, leases, admission and deadlines. |
| `native.py` | Model-thread native lifecycle and background archive scheduling. |
| `model.py` | Runner/DBO hooks and CPU batch counts for Attention dispatch annotations. |
| `tracing.py` | Per-stream phase marks, pending-work accounting and END markers. |
| `npu.py` | FFN pinned control copy and safe lifecycle completion fences. |
| `storage.py` | Bounded JSONL writer and durable native archives. |
| `native_trace.py` | Offline parsing, safe archive extraction and native timestamp import. |
| `pipeline.py` | Clock alignment, phase reconstruction, coverage checks and Chrome export. |
| `analyze.py` | Offline CLI combining manifests, JSONL and native output. |
| `fragments.py` | CPU snapshot/slicing helpers for older request-identity captures. |

Control flows through `control -> endpoint -> worker -> session -> native`.
The endpoint's `compat/patches/profile_rpc.py` adapter preserves every internal
DP engine's response; `native` constructs backends through `compat/npu/profiler.py`.
Runner/model/connector hooks feed `model -> tracing -> native MSTX marks`, while
FFN safe points use `npu -> session -> native`. Offline export follows
`analyze -> native_trace -> pipeline`; workers never parse live traces.

## Session lifecycle and ownership

The coordinator returns HTTP 202 with a session UUID. It moves through
`STARTING -> RUNNING -> STOPPING -> COMPLETED`, or `FAILED` on errors.
Workers use the following states:

| Transition | Action |
| --- | --- |
| `IDLE/COMPLETED/FAILED -> PREPARED` | Validate command and lease, reject busy/unfinished resources, create writer and pending backend; lazily create the LIGHTWEIGHT tracer. |
| `PREPARED -> STARTING` | Commit a future shared start with clock offsets, converted to local monotonic deadlines. |
| `STARTING -> RUNNING` | Control thread requests startup; model thread starts native collection and opens LIGHTWEIGHT admission. |
| `RUNNING -> STOPPING` | Explicit stop, duration/lease expiry or error closes admission and requests safe native stop. |
| `STOPPING -> COMPLETED/FAILED` | Drain admitted work, stop native collection, write SUMMARY and close JSONL. |

Uncommitted PREPARED leases also expire through stop. Identical retries are
idempotent; stale/conflicting commands are rejected. Failed prepare, calibration
or commit triggers coordinator cleanup. Atomic `manifest.json` persistence
supports recovery against administrator-configured targets; worker leases
request stop even if the coordinator disappears.

Drain and writer flush each have a two-second deadline; native operations have
a ten-second deadline. Native errors or missed native deadlines poison backend
reuse and require a worker restart. `details.started_at_ns` reports actual
startup, which can shorten coverage within the shared window. A window ending
before all backends activate fails explicitly.

| Owner | Work |
| --- | --- |
| Coordinator event loop | HTTP control, worker discovery, clock probes, polling and manifest persistence. |
| vLLM API / worker RPC | Authenticate/validate commands and update session state. Attention status RPCs also advance native lifecycle on its model thread. |
| Worker `afd-cam-control` thread | Enforce deadlines, request lifecycle changes and finalize summaries; never call native start/stop. |
| Attention model / DBO threads | Run inference hooks and stream marks; the model thread owns native start/step/stop. |
| FFN model thread | Own native lifecycle and query receive/control fences independently of worker RPCs. |
| Writer / archive threads | Write bounded JSONL batches or compress finalized output, without native lifecycle or live parsing. |

Runtime collection is independent of the runner's static `prof`. Prepare rejects
a configured static AFD profiler or active upstream profiler. The pinned
[`WorkerProfiler._active`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/profiler/wrapper.py)
reserves the backend during delayed starts, when `is_running` is false, and
clears on automatic `max_iterations` stop. CAM can then prepare without an
extra stop RPC. An active runtime session or poisoned backend rejects upstream
starts; revalidate this private upstream state when upgrading vLLM.

## FULL and LIGHTWEIGHT

Both modes use TorchNPU continuously between start and stop, without the static
`WAIT/WARMUP/ACTIVE/REPEAT/SKIP_FIRST` schedule or environment mutation.

| Capture | Backend and output | Maximum duration |
| --- | --- | --- |
| Eager LIGHTWEIGHT | NPU-only `Level_none`, `afd.cam` MSTX phase transitions. | Seven days |
| FFN graph LIGHTWEIGHT | NPU-only `Level0` DR/CS kernel boundaries; graph replay does not rerun Python/MSTX hooks. | Seven days |
| FULL | CPU/NPU `Level2`, shapes and configured stack/modules; native detailed traces. | One hour |

`AFD_NPU_ATTENTION_PROFILER_WITH_STACK` and `AFD_NPU_FFN_PROFILER_WITH_STACK`
apply to runtime FULL. LIGHTWEIGHT disables CPU callbacks, shapes, stacks,
memory, FLOPs and counters; eager capture also disables kernel/task profiling.
It reuses stream tracks and phase messages without timing Event pools,
collector scans, request snapshots or per-layer JSONL. Python Event recording
wrappers are disabled while completion fences remain. Control/writer threads
sleep until notified or a deadline expires. Storage grows with transitions;
long capture duration does not imply a fixed total storage bound.

LIGHTWEIGHT exports three phases per stream:

- **COMPUTE**: contiguous compute regions, including across layer boundaries.
- **COMMUNICATION**: CAM sends, or graph combine-send kernel intervals.
- **WAIT**: receives and DBO yields; graph dispatch-receive kernel intervals.

These include submission gaps and receive transfer/polling, so they are stream
states rather than pure kernel busy time. Separate streams preserve DBO overlap.
Eager intervals start at the first observed mark; a host END mark clips the
capture and closes indefinitely waiting streams. Graph capture produces more
raw data than eager mark-only capture. Its reconstruction covers observed
kernel boundaries; missing, overlapping or unpaired DR/CS makes export `PARTIAL`.

Attention send annotations carry CPU-derived `microbatch_id`, `layer_idx`,
`num_requests` and `num_tokens`, excluding padding and describing the batch
before TP sharding. The microbatch ID is a step-local slot; a request crossing
slices counts in both. Merged COMPUTE intervals are not individual layer times.
Unknown batches remain unannotated; FFN adds no identity metadata. There are no
extra marks, Events, D2H copies or barriers for these annotations. Counts do not
establish independent request totals or request allocation by DP.

### Clock alignment and safe FFN stop

Workers are grouped by Linux boot ID, with zero offset on the coordinator's
host. Three post-prepare probes intersect offset bounds
`[worker_time - response_received, worker_time - request_sent]` for each remote
host. Contradictions fail before commit; workers without boot IDs keep separate
estimates. Offset/uncertainty assumes stable clocks during capture and excludes
device-to-host mapping error and long-term drift.

FFN uses two untimed fences: one before receive, fencing preceding compute and
combine-send, and one for control-copy/submitted-work completion. The model
thread advances native lifecycle while querying completion; the pre-receive
fence is queried only for pending stop. Runtime FFN uses the pinned TorchNPU
`stop_trace` without full-device synchronization, retaining GC/Event cleanup,
finalization and trace handling. Attention/static synchronization is unchanged;
no receive cancellation or extra CAM control messages are added.

FFN graph stop waits for the replay batch to complete. Continue workload through
stop: a blocked DR cannot be cancelled, and an incomplete batch can hit the
native deadline. Profiling does not change graph resource ownership or shutdown.
Layered FFN keeps layer selection on device and skips unused control copies in
LIGHTWEIGHT/idle; legacy FFN and FULL retain three control values. Profiling
does not change CAM operators or the wire protocol. Stream/MSTX bindings target
TorchNPU 2.10 and require revalidation on upgrade. Disabled profiling allocates
no profiling resources.

## End-to-end capture

### Configure servers and coordinator

Install the endpoint entry point on every server. Before launching Attention
and FFN, set the same nonempty secret and a writable directory for the worker
user; retain any additional deployment plugins:

```bash
export VLLM_PLUGINS=ascend,afd,afd_profile
export AFD_CAM_PROFILE_ENABLE=1
export AFD_CAM_PROFILE_TOKEN='<shared-secret>'
export AFD_CAM_PROFILE_DIR=/srv/cam-profiles
```

Keep static AFD/upstream profiling stopped. In a separate shell, start one
coordinator and list each server API URL once, including FFN:

```bash
export AFD_CAM_PROFILE_TARGETS='["http://attention:8000", "http://ffn:8001"]'
export AFD_CAM_PROFILE_TOKEN='<shared-secret>'
export AFD_CAM_PROFILE_DIR='./cam-profiles'
python -m afd_plugin.profiling.control --host 127.0.0.1 --port 8765
```

### Start, status and stop

In another shell with the same token, start a sixty-second session:

```bash
session_id=$(curl --fail-with-body -sS -X POST \
  http://127.0.0.1:8765/cam/profile/start \
  -H "Authorization: Bearer ${AFD_CAM_PROFILE_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"mode":"LIGHTWEIGHT","duration":60.0}' | \
  python3 -c 'import json, sys; print(json.load(sys.stdin)["session_id"])')

curl --fail-with-body -sS \
  "http://127.0.0.1:8765/cam/profile/${session_id}" \
  -H "Authorization: Bearer ${AFD_CAM_PROFILE_TOKEN}"
```

Repeat GET until `RUNNING`, then send normal inference traffic through Attention.
Let the duration expire, or request early stop:

```bash
curl --fail-with-body -sS -X POST \
  "http://127.0.0.1:8765/cam/profile/${session_id}/stop" \
  -H "Authorization: Bearer ${AFD_CAM_PROFILE_TOKEN}"
```

Stop returns HTTP 202 before cleanup completes. Repeat the status GET until
`COMPLETED` or `FAILED`, continuing workload through graph stop. Inspect `errors`,
worker `error`, `details.profiler_poisoned` and `data_quality`. For detailed
traces, start a separate `{"mode":"FULL","duration":60.0}` session after
completion; `{"mode":"LIGHTWEIGHT","duration":86400.0}` collects one day.

### Collect artifacts and export

The coordinator writes `<session_id>/manifest.json`; each worker writes
`<session_id>/<worker_id>.jsonl` containing SESSION/SUMMARY records. FULL keeps
native output in `<session_id>/<worker_id>/full/`.

After LIGHTWEIGHT stop, an archive thread compresses each worker's native
output to `<session_id>/<worker_id>/lightweight.tar.gz`, verifies it and makes it
durable before deleting raw files. Raw and temporary archive data coexist
during compression; failure retains raw data or a completed archive for recovery.
Session completion can precede archiving. Check live worker
`details.native_profiler.archive_state`, `archive_error` and `trace_dir` on
each server; the coordinator's terminal snapshot is not refreshed:

```bash
curl --fail-with-body -sS -X POST \
  http://ffn:8001/cam/profile/internal/status \
  -H "Authorization: Bearer ${AFD_CAM_PROFILE_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${session_id}\"}"
```

Once archives finish, merge every host's session directory with the manifest;
the coordinator does not download artifacts. For this two-host example:

```bash
rsync -a "attention:/srv/cam-profiles/${session_id}/" \
  "./cam-profiles/${session_id}/"
rsync -a "ffn:/srv/cam-profiles/${session_id}/" \
  "./cam-profiles/${session_id}/"

python -m afd_plugin.profiling.analyze "./cam-profiles/${session_id}" \
  --native --output ./pipeline.json.gz
```

Use the matching TorchNPU/CANN environment. `--native` safely extracts archives
to temporary directories and runs native parsing offline when needed. Open
`pipeline.json.gz` in a Chrome trace viewer such as Perfetto; omit `.gz` for
compact JSON. Existing native exports and older Event/fragment captures remain
readable. For FULL, inspect `full/` with native TorchNPU trace tools.

Check the export's `data_quality` as well as the manifest: offline validation
can find missing markers/kernel pairs after a successful stop. Dropped records,
unresolved work, missing clocks and calibration uncertainty limit conclusions.
