# DeepSeek-V4-Flash on A5 with asynchronous Mega

The connector is now named `CAMAsyncMegaAFDConnector` and requires
`afd.async=true`. The PR branch is based on upstream `main` at `0d6ec9e`.

See [A5 development notes](A5_DSV4_ASYNC.md) for the pinned #537 environment,
operator dependencies, precision/weight handling and configuration contract.
The old 0.30 launch instructions do not apply to this branch.

The retained `afd-dsv4-flash-mega-eager-2a2f` E2E scenario generates a 2A2F eager
configuration for later device verification; it has not been run on A5.

## Bundled operator protocol

The connector protocol uses `split_sequence` and `SPLIT_SEQUENCE_MAX` from
`afd_plugin.connectors.npu.mega_runtime`. The maximum comes from the native constant
(`constexpr int64_t`, currently 32), shared by host tiling and device comparison.
Each lane/DP advances 1 through 32 and wraps to 1. The per-buffer launch tag also
wraps; each fused launch clears its local AIV done tags before the launch barrier.
Within each stage, the fused kernel scans DPs round-robin. It claims a DP once
all its Attention ranks have published the next generation for that DP.
Device waits use half-ring ordering and distinguish waiting for this generation
from waiting for its predecessor.

After the fused operator's final AIV barrier, the leader reaffirms TAKEN before
publishing READY. This covers a missed nonblocking early fan-in. After READY is
published to all sources, the leader waits for the previous output generation to
be consumed, including sources with empty routes. These waits bound the sequence
skew required by half-ring comparison.

## Device layer selection

Use `CAMAsyncMegaAFDConnector` and keep `--enforce-eager` on both roles.
FFN ranks publish their sorted model layer IDs through the startup Store.
Attention adopts that mapping before creating the communication windows.

FFN builds four static TensorLists (W1, W2, scale1, scale2),
ordered by model layer ID, with one stacked expert tensor per layer. These lists
reference the already prepared weights; no second copy is allocated. Attention
maps `context.metadata.layer_idx` to the corresponding bank slot and supplies
it as `atten_send(..., layer_index=slot)`. For example, model IDs
`[2, 7]` occupy slots `[0, 1]`.

The dispatch kernel publishes the slot with the generation. FFN submits the
same complete weight bank on every call; mega_ffn reads the device task record
and selects W1/W2/scales internally. atten_recv continues to use its dispatch
ticket and does not need a layer argument. All TP members must publish the same
slot for a task, as checked by the kernel.

With DBO, each FFN stream submits one fused call at a time on its own window.
An idle kernel waits for work on that stage; stopping it can require terminating the workers.

CPU checks:

```bash
python -m pytest tests/unit/connectors/npu/test_mega_topology.py tests/unit/connectors/npu/test_mega_sequence_ring.py tests/unit/connectors/npu/test_mega_layers.py
```

These check host protocol messages and the actual C++ comparison arithmetic;
they do not validate NPU store visibility. On A5, rebuild both bundled artifacts,
run more than 96 dispatch/combine calls per active lane/DP (three ring laps),
and include empty routes and uneven TP publication. Verify accuracy against the
reference as well as continued progress with DBO enabled and disabled.

With DBO enabled, also send work only to stage 0, then only to stage 1.
Verify that the active stage returns results while the other waits.
