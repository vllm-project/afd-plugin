# Async FFN FaaS control ABI v1 (910C/A3)

The control operator transports BF16 activation rows and int32 expert IDs.
It performs source/layer selection, grouping, packing, result scatter,
publication and consumption. DeepSeek-V4 computation uses native W4A8
operators between `PREPARE` and `FINISH`. It is a host-controlled backend.
The control operator has no weight inputs or expert projection implementation.
Python and native bindings must be built from matching sources.

## Session and work identity

Session ranks are `[A0, ..., F0, ...]`; each A DP source has `attention_tp`
consecutive A members. All ranks compare the model identity, ABI, precision,
quantization, clamp, dimensions, layer map, topology, slots and capacity before
creating the communication group. Native window initialization must finish on
all ranks before paired warmup and readiness.

Work identity is `(source, sequence, model_layer)`, with a contribution from
each A TP member, including zero-token contributions. Sequence starts at one,
increases across layers and requests, and selects slot `(sequence - 1) % slots`.
It cannot wrap; `2**31 - 2` is the last admitted work generation. Signals are
aligned 32-bit generations with one writer per ready/credit field, each in a
separate 64-byte line. Metadata uses int64 fields. Missing layers are invalid.

F's window contains input slots per A rank; A's contains result slots per F
rank. Input metadata remains stable until credit returns. Payload contains
BF16 rows and int32 routes; -1 marks padding. F writes only its owned routes,
and A reads each valid route only from its owner. A retains already scaled
FP32 probabilities locally and applies them once during combine.

## Native control operations

| Mode | Value | Role and result |
| --- | --- | --- |
| INITIALIZE | 0 | Clear local windows/state. |
| PUBLISH | 1 | A publishes payload/metadata/ready or returns backpressure. |
| COMBINE | 3 | A validates all F results, combines, then returns credit. |
| END_SOURCE | 4 | A publishes its final consumed sequence boundary. |
| ABORT_SESSION | 5 | Publish terminal peer failure. |
| PEEK_RESULTS | 6 | Diagnostic route outputs; no credit or state advance. |
| EXCHANGE | 8 | A publishes, then combines only on publication success. |
| SCOPED_EXCHANGE | 9 | Bounded A model-scope exchange. |
| PREPARE | 10 | F selects one ready source and packs its local expert rows. |
| FINISH | 11 | F scatters computed rows and publishes their ready signal. |

Native modes 2 and 7 are rejected. Python `EXECUTE` is an executor-level
transaction; the native backend implements it with PREPARE, computation and
FINISH. The operator takes x, IDs, probabilities, layer map, request, mutable
state and private byte scratch, plus mode/config/group attributes; returns y
and eight int64 status fields. The native config has 13 fields.

## Ownership and ordering

State progresses through free → input published → preparing/computing → result
published → A consumption completed → credit returned. Publication completes
payload MTE3, joins local writers, completes metadata MTE3, then writes ready.
Consumers poll signals through MTE2 DMA and complete that transfer before
reading data. Local execution completion, remote ready and returned credit
have different meanings.

Each F independently chooses ready sources round-robin after all source TP
members agree. There is no F-F collective or global per-layer step schedule.
MC2 establishes direct-window resources using an AlltoAll descriptor; protocol
progress uses point-to-point window accesses. Readiness scans are finite and
may return IDLE. A validates exact sequence/layer/count from every F.

PREPARE preserves source, sequence and layer in executor scratch; its completed
64-byte receipt exposes only code/layer/packed-row count to the host. Each local
expert group is padded to 16 rows. The route map retains TP member/capacity
positions and marks padding. Native quantization and GMMs use the packed rows;
only valid routes are scattered. Zero local work still requires FINISH and a
result publication. PREPARE and host metadata reads return no input credit.
FINISH runs after computation/output copy on the same stream. A combines in
FP32, casts once to BF16, then returns all F credits.

One F transaction owns scratch and borrowed weights until retirement. Ordinary
A calls also have one active operation. EXCHANGE backpressure means nothing was
published; retry preserves identity. A combine timeout means publication already
occurred; retry only COMBINE. Duplicate/skipped publication or consumption is
invalid. Successful EOS is terminal, including EOS(0), and requires both A
counters to equal the boundary.

A model scope queues at most four serial exchanges on one communication stream.
The C++ submitter retains inputs/outputs before enqueue and holds pending work
through partial failure. Producer events order publication; done events and
recordStream order consumer storage use. All statuses are validated before pool
reuse and model return. Any scope failure aborts the session and discards the
worker's speculative Attention/KV state. Failed output clearing does not make a
failed operation successful and does not return credit.

Completion flushes torch-npu dispatch, waits on the session stream with a finite
ACL timeout, then copies the status. The ACL timeout does not separately bound
the preceding host dispatch flush. Enqueue/wait failure retains all owners.
Management endpoints do not synchronize devices. Host heartbeat checks detect
peer errors between bounded operations.

Normal shutdown drains accepted requests to every source's EOS boundary. Abort
is a distinct terminal failure. After the last local operation, peers acknowledge
quiescence before the transport group or model weights are released. Unknown
peer safety requires process exit while retaining resources; Store failure or
an abort notification alone does not establish device quiescence.
