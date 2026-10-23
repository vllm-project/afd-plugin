# Async FFN FaaS native ABI v1 (910C/A3)

The control operator transports BF16 activation rows and int32 expert IDs.
It performs source/layer selection, grouping, packing, result scatter,
publication and consumption. One operator serves two compute contracts:

- DeepSeek-V4 Flash uses native W4A8 operators between host-controlled
  `PREPARE` and `FINISH`. The control operator only provisions packed rows and
  receipts; the host launches the quantized GMMs.
- DeepSeek-V2 BF16 uses the mixed-core `EXECUTE` entry, which runs both
  projections and SwiGLU on device between the same control phases. `SERVE`
  repeats that bounded transaction for the standalone service.

Both contracts pass the executor-borrowed BF16 weight view (`w13`, `w2`) and
the int32 model-layer map. Control modes never project with them; the fused
compute modes validate their layout against the negotiated session. Python and
native bindings must be built from matching sources.

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

## Fused BF16 weights and execution

FFN weights are `w13[layer, local_expert, 2I, H]` (gate then up) and
`w2[layer, local_expert, H, I]`. `layer_map[model_layer]` selects the compact
weight layer on device; missing layers are rejected. The session fixes expert
placement, so every F rank owns `num_experts / ffn_ranks` local experts for each
selected model layer. Model code validates the full index, dtype and shape
before allocation and keeps ownership of the device tensors; the executor only
borrows them until peer quiescence.

The mixed 1 AIC / 2 AIV entry groups routes by local expert, pads each group to
16 rows, packs BF16 activations in executor-owned scratch, runs both
projections on Cube with FP32 accumulation and BF16 outputs, evaluates SwiGLU
on AIV, then scatters valid routes and publishes completion. Projection tiles
are 128 columns; the route map retains TP member/capacity positions. The task
requests CANN's Matmul system workspace because the generated AIC wrapper clears
it before entering `EXECUTE`, including idle work; that is distinct from the
executor's persistent scratch.

`EXECUTE` returns after one source work item or a finite poll budget. Idle is
normal and releases the paired Cube half. `SERVE` runs at most
`MAX_WORK_PER_SERVE` (eight) fused iterations and reports one aggregate receipt
in status word 5; each work still publishes its own ready signal and advances
its source cursor immediately, so no A waits for the rest of the batch. A
finite readiness scan may end early; idle, drain and peer failure end both AIV
and AIC loops together. A partial batch ending in `DRAINED` still reports its
completed count. `EXECUTE` remains the single-work primitive used by protocol
diagnostics and V4 host transactions.

## Native control operations

| Mode | Value | Role and result |
| --- | --- | --- |
| INITIALIZE | 0 | Clear local windows/state. |
| PUBLISH | 1 | A publishes payload/metadata/ready or returns backpressure. |
| EXECUTE | 2 | V2 fused receive/compute/publish; one transaction, or idle. |
| COMBINE | 3 | A validates all F results, combines, then returns credit. |
| END_SOURCE | 4 | A publishes its final consumed sequence boundary. |
| ABORT_SESSION | 5 | Publish terminal peer failure. |
| PEEK_RESULTS | 6 | Diagnostic route outputs; no credit or state advance. |
| SERVE | 7 | Bounded multi-work fused V2 batch with an aggregate receipt. |
| EXCHANGE | 8 | A publishes, then combines only on publication success. |
| SCOPED_EXCHANGE | 9 | Bounded A model-scope exchange. |
| PREPARE | 10 | F selects one ready source and packs its local expert rows. |
| FINISH | 11 | F scatters computed rows and publishes their ready signal. |

The operator takes x, IDs, probabilities, w13, w2, layer map, request, mutable
state and private byte scratch, plus mode/config/group attributes; returns y
and eight int64 status fields. The native config has 13 fields. Compute modes
validate weight dtype and exact layer/expert geometry before enqueue.

## Host V4 transaction

PREPARE preserves source, sequence and layer in executor scratch; its completed
64-byte receipt exposes only code/layer/packed-row count to the host. Each local
expert group is padded to 16 rows. The route map retains TP member/capacity
positions and marks padding. Native quantization and GMMs use the packed rows;
only valid routes are scattered. Zero local work still requires FINISH and a
result publication. PREPARE and host metadata reads return no input credit.
FINISH runs after computation/output copy on the same stream. A combines in
FP32, casts once to BF16, then returns all F credits.

## Ownership and ordering

State progresses through free → input published → preparing/computing → result
published → A consumption completed → credit returned. Publication completes
payload MTE3, joins local writers, completes metadata MTE3, then writes ready.
Consumers poll signals through MTE2 DMA and complete that transfer before
reading data. Local execution completion, remote ready and returned credit
have different meanings.

Each F independently chooses ready sources round-robin after all source TP
members agree. There is no F-F collective or global per-layer step schedule.
MC2 establishes direct-window resources using an AllToAll descriptor; protocol
progress uses point-to-point window accesses. Readiness scans are finite and
may return IDLE. A validates exact sequence/layer/count from every F.

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
ACL timeout, then copies the status. `SERVE` copies the terminal code and
aggregate count in one header read; only a completed, valid receipt establishes
its count: OK requires positive work, IDLE requires zero, and drain/failure may
retain completed work. The ACL timeout does not separately bound the preceding
host dispatch flush. Enqueue/wait failure retains all owners. Management
endpoints do not synchronize devices. Host heartbeat checks detect peer errors
between bounded operations.

Normal shutdown drains accepted requests to every source's EOS boundary. Abort
is a distinct terminal failure. After the last local operation, peers acknowledge
quiescence before the transport group or model weights are released. Unknown
peer safety requires process exit while retaining resources; Store failure or
an abort notification alone does not establish device quiescence.

## Validation semantics

For native probe validation, CPU projections use FP64 to avoid FP32 reference
error crossing a BF16 midpoint. The reference rounds gate/up projection outputs
to BF16, evaluates SwiGLU in FP32 and rounds its output to BF16, then rounds the
down projection to BF16. A sums BF16 expert results times FP32 top-k weights in
FP32 and casts once. Tolerances are fixed by the hardware test before run;
full-model comparison must additionally validate these model semantics.

The BF16 DeepSeek Attention connector preserves the pinned native
token-unpermute probability boundary: it selects routes in FP32, rounds selected
probabilities to the activation dtype, then encodes those rounded values as
FP32 transport/combine inputs. The generic Executor and standalone client use
the explicit probabilities their caller supplies. Expert computation, slot
identity and FP32 accumulation do not change at this model boundary.
