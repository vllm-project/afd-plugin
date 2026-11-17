# DeepSeek-V4-Flash on A5 with asynchronous MegaMoE

The connector is now named `CAMAsyncMegamoeAFDConnector` and requires
`afd.async=true`. Its AFD base is `update/v0.32.0` at `2917805`.

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

Layer selection remains host-side: FFN submits one layer's weights and dispatch
uses the default bank index 0. Do not pass a model layer index without also
switching FFN to a matching full weight bank.

CPU checks:

```bash
python -m pytest tests/unit/connectors/npu/test_megamoe_work.py tests/unit/connectors/npu/test_megamoe_admission.py tests/unit/connectors/npu/test_megamoe_sequence_ring.py
```

These check host protocol messages and the actual C++ comparison arithmetic;
they do not validate NPU store visibility. On A5, rebuild both bundled artifacts,
run more than 96 dispatch/combine calls per active lane/DP (three ring laps),
and include empty routes and uneven TP publication. Verify accuracy against the
reference as well as continued progress with DBO enabled and disabled.
