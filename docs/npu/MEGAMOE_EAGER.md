# DeepSeek-V4-Flash on A5 with asynchronous MegaMoE

The connector is now named `CAMAsyncMegamoeAFDConnector` and requires
`afd.async=true`. The PR branch is based on upstream `main` at `0d6ec9e`.

See [A5 development notes](A5_DSV4_ASYNC.md) for the pinned #537 environment,
operator dependencies, precision/weight handling and configuration contract.
The old 0.30 launch instructions do not apply to this branch.

The retained `afd-dsv4-flash-megamoe-eager-2a2f` E2E scenario generates a 2A2F eager
configuration for later device verification; it has not been run on A5.

## Bundled operator protocol

The bundled Python API exports `split_sequence` and `SPLIT_SEQUENCE_MAX` from
`afd_plugin.compat.npu.megamoe`. The maximum comes from the native constant
(`constexpr int64_t`, currently 32), shared by host tiling and device comparison.
Each lane/DP advances 1 through 32 and wraps to 1. The per-buffer launch tag also
wraps; each fused launch clears its local AIV done tags before the launch barrier.
CPU mailbox generations and TP PUBLISHED snapshots require exact equality.
Device waits use half-ring ordering and distinguish waiting for this generation
from waiting for its predecessor.

After the fused operator's final AIV barrier, the leader reaffirms TAKEN before
publishing READY. This covers a missed nonblocking early fan-in. After READY is
published to all sources, the leader waits for the previous output generation to
be consumed, including sources with empty routes. These waits bound the sequence
skew required by half-ring comparison.

## Device layer selection

Like the A3 async CAM layered executor, enable the feature on **every FFN
worker** with:

```bash
export AFD_ASYNC_CAM_LAYERED_GMM=1
```

Use `CAMAsyncMegamoeAFDConnector` and keep `--enforce-eager` on both roles.
Do not enable the switch on Attention. FFN publishes the mode and sorted model
layer IDs through the startup Store; all FFN ranks must agree. Attention adopts
that mapping before creating the communication windows. The default (unset or
0) retains host-selected, single-layer calls.

In layered mode, FFN builds four static TensorLists (W1, W2, scale1, scale2),
ordered by model layer ID, with one stacked expert tensor per layer. These lists
reference the already prepared weights; no second copy is allocated. Attention
maps `context.metadata.layer_idx` to the corresponding bank slot and supplies
it as `megamoe_dispatch_send(..., layer_index=slot)`. For example, model IDs
`[2, 7]` occupy slots `[0, 1]`. In legacy mode every slot is 0 because FFN passes
only the requested layer's weights.

The dispatch kernel publishes the slot with the generation. FFN submits the
same complete weight bank on every call; fused_moe reads the device task record
and selects W1/W2/scales internally. combine_recv continues to use its dispatch
ticket and does not need a layer argument. All TP members must publish the same
slot for a task, as checked by the kernel.

The CPU mailbox still schedules DP, generation and lane, and retains the model
layer ID for legacy calls. This change does not remove PUBLISHED admission,
change ubatch scheduling, or add graph execution support. All A/F processes must
run the updated connector because initialization now exchanges layer metadata.

CPU checks:

```bash
python -m pytest tests/unit/connectors/npu/test_megamoe_work.py tests/unit/connectors/npu/test_megamoe_admission.py tests/unit/connectors/npu/test_megamoe_sequence_ring.py tests/unit/connectors/npu/test_megamoe_layers.py
```

These check host protocol messages and the actual C++ comparison arithmetic;
they do not validate NPU store visibility. On A5, rebuild both bundled artifacts,
run more than 96 dispatch/combine calls per active lane/DP (three ring laps),
and include empty routes and uneven TP publication. Verify accuracy against the
reference as well as continued progress with DBO enabled and disabled.

For A5 verification, compare the same requests with the FFN switch set to 0 and
1, including more than one model layer and at least three sequence laps per
active lane/DP. Check numerical outputs as well as forward progress. The host
unit tests validate mapping and rank agreement, not device numerical accuracy.
