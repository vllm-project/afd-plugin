# Ascend 910C operator tests

Standalone harnesses and opt-in device tests for the Ascend operators this
plugin owns. They run on 910C outside CI, and each one states whether it needs a
device, a multi-rank launch, or nothing but a standard Python interpreter.

| File | Needs | Purpose |
| --- | --- | --- |
| `async_cam_precision.py` | torchrun, 4 ranks | Independent dispatch and weighted-combine reference |
| `async_cam_performance.py` | torchrun, 4 ranks | Per-operator and complete-round measurements |
| `async_cam_common.py` | - | Protocol planning, HCCL setup, and four-stage execution |
| `gmm_layered_performance.py` | 1 process, 1 device | Layered W13/W2 latency under eager and ACL graph |
| `test_gmm_swiglu_quant_v2_layered_precision.py` | pytest, 1 device | Layered W13 numerics, both quantization scenarios |
| `test_gmm_layered_device_group_list.py` | pytest, 1 device | Layered W2 with a device-produced group list |
| `test_async_cam_layered_w4a8.py` | pytest, 1 device | The `AsyncCAMW4A8Executor` chain end to end |
| `test_attention_gate_numerics.py` | pytest, 1 device | Gate, router, and activation components |
| `__init__.py` | - | Package marker |

The first group exercises the four routed async CAM operators added by
[PR #361](https://github.com/vllm-project/afd-plugin/pull/361):
`torch.ops.afd_ascend.afd_async_{dispatch,combine}_{send,recv}`. Harness and
hardware follow-up:
[issue #369](https://github.com/vllm-project/afd-plugin/issues/369).
They use the compact routed-only protocol and do not use the external CAM
connector, model weights, or shared-expert tensors.

The second group covers the layered routed-expert operators,
`gmm_swiglu_quant_v2_layered` (W13) and `grouped_matmul_layered` (W2), which
select their layer on device from a `layer_index` input. Those are the operators
`AsyncCAMW4A8Executor` drives, and their `all_*` lists hold one element per layer
rather than one per expert. Their validation entry points are in
[docs/npu/TESTING.md](../../docs/npu/TESTING.md), and the two precision suites
are described under [Layered operator precision](#layered-operator-precision)
below.

The async CAM harness implements the AFD schemas documented in
[CAM_ASYNC_ROUTED_OPS.md](../../docs/npu/CAM_ASYNC_ROUTED_OPS.md). The input
values, route table, quantization result, and weighted sum are all produced
inside this repository from the raw inputs; the harness depends on no external
reference directory at runtime.

## Python-only validation

From the repository root, no torch or Ascend installation is needed for:

```bash
python tests/npu/async_cam_precision.py --help
python tests/npu/async_cam_precision.py --dry-run
python tests/npu/async_cam_performance.py --dry-run
python tests/npu/gmm_layered_performance.py --dry-run
python -m unittest tests.unit.compat.npu.test_async_cam_harness \
  tests.unit.compat.npu.test_async_cam_performance \
  tests.unit.compat.npu.test_async_cam_golden \
  tests.unit.compat.npu.test_gmm_layered_performance
```

Dry runs validate topology, positive per-rank lengths, receive capacity,
integer-address limits, buffer requirements, and planned expert chunks.
They execute no native operators. Normal execution requires torchrun and fails
if the runtime, device, or one of the four registered operators is missing.
`gmm_layered_performance.py --dry-run` is a single-process plan instead: it
prints the derived `M` and `C`, every case on both launch paths, and any
modelling warning, and it runs under `python3 -S` so a device import fails
loudly rather than silently.

`test_async_cam_golden` runs tensor regression tests when CPU PyTorch is
installed and otherwise reports skips. It needs no torch-npu or NPU. It checks
FP16/BF16 output rounding, FP32 accumulation order, quantization budgets and
strict bit comparisons; the other three test modules need only standard Python.
`test_gmm_layered_performance` guards what the layered benchmark decides to
measure - the `M` and `C` derivations, the apportionment, the percentiles, the
case matrix, and the report rendering - because an error there produces
plausible numbers for the wrong problem, which is harder to notice than a
crash.

## Native setup

Use Ascend 910C with the matching PyTorch, torch-npu, CANN/HCCL stack and an
AFD extension built with all four operators. Follow the native build guide:

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 \
  pip install -e . -v --no-build-isolation
```

Launch one process per NPU, with Attention ranks first and MoE ranks second.
All ranks must receive identical script arguments. The default topology is
2 Attention + 2 MoE, Attention TP=2, four experts per MoE, top-k=2, and unequal
Attention lengths `7,5`. `--lengths 32` broadcasts one length to every Attention
rank. Multi-node launches use the usual torchrun rendezvous arguments;
`LOCAL_RANK` selects the local device and `RANK` is the global HCCL rank.

The harness creates a Gloo world for control traffic and a separate full-rank
HCCL group for payloads. It passes the actual HCCL communicator name to every
operator. `comm_id=0` and FP16 `comm_args` are reserved ABI placeholders.
No ordinary HCCL collective is issued on the payload group's windows.

`--window-mb` sets `HCCL_BUFFSIZE` and the HCCL group's `hccl_buffer_size`
to the same value before native initialization. AFD host tiling reads
`HCCL_BUFFSIZE` directly, keeping its window size aligned with the
harness's HCCL allocation. `--capacity`
sets `BATCH_SIZE_FACTOR=capacity/262144`; it must be a power of two and fit
every TP-aggregated single expert. Multiple experts may require multiple
receive chunks. The preflight rejects configurations outside the harness's
conservative bounds for the current kernels; these bounds are not a claim
that all accepted configurations have been qualified on hardware.

## Precision

```bash
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_precision.py --dtype fp16 --dynamic-quant 0

# Multiple chunks, quantized BF16, and repeated use of the same HCCL windows.
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_precision.py --dtype bf16 --dynamic-quant 1 --capacity 4

# Empty experts and an entirely empty MoE participant still send completion.
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_precision.py --routing sparse --capacity 16

# Two independent Attention groups: receive order may vary by readiness.
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=6 \
  tests/npu/async_cam_precision.py --attn-ranks 4 --moe-ranks 2 --tp-size 2 \
  --lengths 7,5,3,1 --capacity 4
```

Run both dtypes with quantization 0 and 1, and both balanced/sparse routing.
The default input has distinct token/rank/layer values, distinct top-k IDs,
nonuniform FP32 weights, and a deterministic expert-dependent affine transform.

### CPU golden

The comparison baseline is a CPU golden built from the raw inputs, not from
anything the device returns. Every participating rank's `x`, `expert_ids` and
`expert_weights` are generated once on the CPU, and the golden is derived from
that same tensor that is handed to the operator, so no unrounded FP32 input can
enter the reference. Floating-point payloads, synthetic expert outputs, and the
final combine golden use the selected FP16/BF16 dtype. Device metadata is a
*compared object* and never an input to the reference: routing is rebuilt
independently from the raw `expert_ids`,
otherwise a wrong metadata field would be folded silently into the answer.

The checks answer different questions and are reported under separate labels:

| Layer | What the golden provides | Contract |
| --- | --- | --- |
| Routing and movement | Destination rank, expert and token for every logical route | Exact |
| Quantized dispatch | INT8 values and FP32 scales | Bit-identical under the versioned quantization contract |
| Weighted combine | FP32 multiply/accumulate in top-k slot order, then FP16/BF16 rounding | Compare against a golden with the same dtype as the device output |
| Quantization error | Quantized and unquantized CPU goldens in the selected FP16/BF16 dtype | Separate run-derived absolute budget |

Dispatch is validated per receive chunk: unquantized payloads, quantized INT8
values, and the FP32 dequantization scales must all be bit-identical to the
golden. Quantization is lossy; a payload or a scale that the source only copies
is not, so no tolerance is granted there. The quantization error itself is
reported separately, under its own budget — see the combine note below.

### Quantization contract

The dynamic-quantization contract is versioned in
`async_cam_common._quantize_contract()`, which is the single place allowed to
encode it. It mirrors `QuantProcess()` in
`csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel`: per-row FP32
`amax`, multiplier `127/amax`, FP32 product rounding followed by
round-half-to-even, and the reciprocal
`1.0f/(127.0f/amax)` stored as the dequantization scale. The stored value is the
reciprocal of an already-rounded quotient, so it is *not* bit-identical to
`amax/127` in FP32 for most rows; the golden reproduces the kernel's form because
the scales are compared against the device. If the kernel's rounding, range, or
reciprocal convention changes, that function has to change with it and the
contract version must be bumped. A tolerance must never be widened to absorb a
contract change. The same scalar arithmetic is restated in plain Python (`f32`,
`quant_dynamic_scale`, `quant_stored_scale`, `quantize_row`) so CPU tests can pin
it without a tensor library.

Combine uses one CPU golden in the selected FP16/BF16 dtype. The CPU follows the
native arithmetic: multiply and accumulate in FP32 in top-k slot order, then
round once to the output dtype. FP32 is an intermediate arithmetic dtype; the
acceptance baseline is the rounded FP16/BF16 result.

With quantization enabled, a separate CPU golden applies the same expert
transform and weighted combination to the unquantized inputs. Both goldens have
the selected FP16/BF16 dtype. Their difference is checked against an explicit,
run-derived absolute error budget with `rtol=0`, independently of the device
comparison. Quantization error is reported separately from the combine tolerance.

Final combine uses `atol=rtol=0.001` for FP16 and `atol=rtol=0.008` for BF16,
with nonfinite mismatches rejected. Mismatch reports include the number of
bitwise-differing elements, `max_abs_error`, `relative_l2`, the count outside
tolerance, and the first differing index. FP64 is used only to calculate error
diagnostics, never as the golden acceptance dtype. Zero-tolerance checks compare
bit patterns, including the sign of zero.

Only valid payload/scale rows are inspected; unused capacity is uninitialized.
The receive anchor is FP16/BF16 and is separate from INT8 send placeholders.
Inputs deliberately have nonzero row maxima for dynamic quantization; zero-row
quantization, duplicate expert IDs, invalid device metadata injection, graph
execution, and concurrent layers require separate hardware qualification.

### Layered operator precision

Two suites cover the layered routed-expert operators. Both are pytest modules
gated on `AFD_RUN_ASCEND_OP_RUNTIME=1`, both need one device, and neither
performs any collective.

```bash
SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 LOCAL_RANK=0 \
  pytest -o addopts= -v -ra -s \
  tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py \
  tests/npu/test_gmm_layered_device_group_list.py
```

`test_gmm_swiglu_quant_v2_layered_precision.py` collects 11 cases against W13.
It covers A8W8 and A8W4, and it needs two reference strategies because only one
scenario has a usable built-in counterpart:

| Scenario | Reference | Why |
| --- | --- | --- |
| `a8w8` | built-in `npu_grouped_matmul_swiglu_quant_v2` | Same operator family on the selected layer, so a differential comparison holds |
| `a8w4` | CPU golden in the module | The built-in derives its output width from the weight's own last dimension, which for an `[E, K, N // 8]` int32 weight is `N // 8` while the kernel unpacks int4 and expects `N`, so it allocates an output eight times too narrow |

It also checks that the two layers produce different output, so a wrong
`layer_index` cannot pass unnoticed, and that both `group_list_type` values are
correct in either call order.

**Each attribute combination gets its own activation row count, deliberately.**
The tiling is cached per shape and tiling key, and neither `group_list_type` nor
`dequant_mode` is part of that key - both reach the tiling only as data, and the
A8W8 fusion path pins a constant key. Two combinations sharing a shape in one
process therefore reuse whichever tiling ran first, *along with its attribute
values*. That is not hypothetical: it made this suite report a failure while the
operator was correct. `test_variant_shapes_are_distinct` asserts the invariant
in pure Python, so it also runs in the unopted-in collection.

`test_gmm_layered_device_group_list.py` collects 4 cases against W2, over both
scenarios and both `group_list_type` values, and is the regression for a
device-produced group list (the interface merged in PR #384). Set
`AFD_GMM_PROFILE_TRACE=/path/to/trace` to export one Chrome trace per group-list
representation; only the layered invocation is profiled.

Both suites are A8W4 MSD on one side or the other, and that path may rewrite the
activation buffer it was handed. Every invocation therefore gets its own clone;
without it, whichever op runs first changes what the second sees.

## Performance

```bash
timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_performance.py --measure-mode both \
  --warmup 2 --iterations 10 --output .cache/async-cam-performance.json

timeout --kill-after=30s 600s torchrun --standalone --nproc_per_node=4 \
  tests/npu/async_cam_performance.py --measure-mode cycle \
  --warmup 5 --iterations 50 --dtype bf16 --dynamic-quant 1 \
  --output .cache/async-cam-cycle.json
```

An untimed precision round must pass before warmup. Warmup samples are excluded
from measured summaries. `both` runs separate `send` and `recv` schedules:

- `send`: receivers announce readiness before senders launch. Sender staggering
  and a receive guard are configured by `--sender-stagger-ms` and
  `--receiver-guard-ms`. Summaries include dispatch-send and combine-send.
- `recv`: senders announce readiness before native launch, then receivers wait
  `--receiver-delay-ms`. Only receive operators are synchronously timed;
  unmeasured sends are not synchronized before receivers can post.
- `cycle`: all four operators are measured without artificial delays. This is
  a serialized harness round with a synthetic expert transform.

Readiness describes a host about to launch, not a confirmed running kernel.
Delays apply only to the first call of each phase so later chunks drain freely.
Use a separate device-profiler run to establish whether peer waits or sender
overlap remain; this harness does not claim pure transfer or kernel latency.

| Metric | Boundary |
| --- | --- |
| `host_ms` | Native call through NPU synchronization, including launch and waits |
| `event_ms` | NPU stream events around the native call, including device waits |
| `lifecycle_ms` | Whole round including CPU metadata, transforms, control and delays |

CPU oracle work is excluded from measured rounds. Per-op intervals exclude the
expert transform and deliberate scheduling delay. Each dispatch-receive and
combine-send chunk is one sample; unequal chunk counts must not be treated as
equal request counts. JSON includes raw samples, per-rank mean/median/p95/min/max,
and the per-iteration maximum across ranks for complete-round latency. The p95
uses the nearest-rank definition. There is no inferred bandwidth or model
throughput claim.

Successful JSON reports are written after normal process-group teardown.

## Layered computation operator benchmark

`gmm_layered_performance.py` measures the two layered routed-expert operators on
the DeepSeek V4 Flash W4A8 / EP4 FFN path:

| Operator | Role |
| --- | --- |
| `gmm_swiglu_quant_v2_layered` | W13 grouped GEMM, SwiGLU, per-token quant |
| `grouped_matmul_layered` | W2 grouped GEMM, bfloat16 output |

Geometry is derived from the model rather than configured as an arbitrary grid:
`M = top-k * T / EP` from the per-rank source batch, and
`C = 262144 * BATCH_SIZE_FACTOR` from the host tiling. `C` appears in every
tensor shape and so participates in the tiling cache key; `M` reaches the kernel
only through `group_list` and does not. They are swept separately and reported
together with their ratio. `M/C == 1` exactly is unreachable, because `M` carries
a factor of three while `C` must be a power of two - but that does not cap
utilisation near the default sweep. Only `C` is constrained to a power of two, so
a source batch that is not one can land just under capacity: `--source-tokens
43690 --capacity 65536` gives `M = 65535` and `M/C = 0.99998`, still under the
uint16 bound. Each reported row states the utilisation it ran at.

```bash
# Plan only; no torch, no torch_npu, no device.
python3 tests/npu/gmm_layered_performance.py --dry-run

# Check the concatenated baseline, then measure and archive the report.
LOCAL_RANK=0 python3 tests/npu/gmm_layered_performance.py \
  --validate-baseline --report
```

It is a script, not a pytest module, so `AFD_RUN_ASCEND_OP_RUNTIME` does not
gate it and `SOC_VERSION=910c` belongs to the build rather than the run. The
only variable it reads is `LOCAL_RANK`, which selects the device and defaults
to 0. `--source-tokens` and `--capacity` are the run-time geometry inputs; no
environment variable feeds them.

`--only w13,w2,chain` selects the operators, `--distribution` chooses the routing
shape (`balanced` for steady-state load, `skewed` to introduce idle experts and
group-size spread), and `--layer-mode` chooses between repeating one layer and
rotating the whole list. Rotation is what a model forward does; its total is
reported divided by the layer count, so both modes are per-layer.

`--execution-mode eager,aclgraph` selects the launch paths and defaults to both,
because they answer different questions and only the pair separates them:

| Mode | What it measures |
| --- | --- |
| `eager` | Each operator call goes to the device as it is issued |
| `aclgraph` | One measured unit is captured into a reusable `NPUGraph`, then each sample replays it |

The report's execution-mode table puts the two side by side. Its speedup is the
graph's own contribution, distinct from the fusion contribution in the results
table, since both columns recompute the same weights on the same input with the
same layer selection. Both aggregation tables carry a Geometry column: a run may
sweep several `--source-tokens` or `--capacity` values that share every other
attribute, and keying without the geometry would let the last workload overwrite
the earlier ones. `aclgraph` warms up outside the capture and restores the
input before each start event, so neither one-time tiling nor the copy lands
inside a measured interval. Capturability is not assumed: the capture succeeding
on 910C is the evidence that neither binding nor either op_api entry reads a
tensor back to the host. What this harness does not do is compare the two paths
numerically - it times them - so graph-versus-eager agreement remains a precision
question.

Both paths reach the deployment as a walk over the layer list, so `rotate` is the
row that describes a real step. Eager issues one call per layer back to back, and
the deployment's captured graph is replayed once per layer, also back to back;
either way a step pays one timed unit's fixed cost per `L` layers, so a layer
costs `c + f / L`. The `single` pair instead charges a full `f` to every layer,
which is the cost of an isolated call rather than of a step, so it overstates the
gain. On 910C for `w13` the `rotate` pair reports about 1.03x.

How the graph is captured does not change that: one graph holding the whole list
replayed once and a one-layer graph replayed `L` times both cost `L * c + f`,
because only the first replay pays the pipeline fill and the rest hide behind it.
This harness captures the first form, the deployment uses the second.

What the graph removes is precisely the per-unit fixed cost: across the two
launch paths the decomposition table's `f` collapses while its per-call `c`
barely moves. Treat the graphed `f` as having fallen to the measurement floor
rather than as a precise ratio, since it is a difference of two nearly equal
numbers.

A results ratio below 1 does not mean the kernel is slower, and W2 is the case
that shows why. Under eager its steady-state figure is slightly under 1.00, but
the decomposition puts the two sides' per-call `c` within a fraction of a percent
of each other while the layered side carries the larger fixed term; nearly all of
the difference is that term over the layer count. Under a graph, W2 returns to
parity with its `c` still matching. Read a ratio as "is this configuration
faster", not "is this kernel faster".

W2 is compared against the fused built-in `npu_grouped_matmul`. W13 has no usable
fused counterpart, because that operator derives its output width from the
weight's own last dimension rather than unpacking packed int4, so its baseline is
the concatenation `npu_grouped_matmul` -> `npu_swiglu` -> `npu_dynamic_quant`.
`--validate-baseline` drives both that concatenation and the fused operator with
byte-identical tensors and checks that they agree before any timing run.

Both operators use the A8W4 MSD path, which may rewrite the activation buffer it
was handed. Each timed unit therefore gets a restored input, materialized and
synchronized before the start event so the copy never enters a measured interval.

Results print to the screen. `--report` additionally writes
`layered_computation_operator_benchmark.md`, which covers both operators in one
file.

It is a single-process script, not a `torchrun` one: it issues no collective, so
there is no rank layout to agree on. It loads the extension through the same
weaker check the other layered tests use, which requires the two compute
operators but not the four `afd_async_*` communication operators. `LOCAL_RANK`
selects the device and defaults to 0. It reads no `BATCH_SIZE_FACTOR` from the
environment - `--capacity` is the input and the factor is derived from it - so a
value exported for a CAM run does not perturb it.

## Hardware evidence still required

The local Python checks do not establish Ascend correctness, CANN compilation,
HCCL connectivity, or performance. Record the exact source revision, command,
device placement, CANN/HCCL/driver versions, PyTorch/torch-npu versions, window
sizes, and JSON/log artifacts when running on hardware. Compare performance
only with identical workload, topology, warmup, iterations, and measurement mode.

`--timeout-seconds` bounds control operations and arms a per-round watchdog.
Some native kernels poll device flags without a process-group timeout; a stuck
round terminates its worker rather than trying to synchronize it during cleanup.
The external Linux `timeout` in the examples also bounds startup and teardown.
Device cleanup after failure still requires validation on the target stack.
