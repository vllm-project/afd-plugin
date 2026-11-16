# DeepSeek-V4-Flash on A5 with asynchronous MegaMoE

The connector is now named `CAMAsyncMegamoeAFDConnector` and requires
`afd.async=true`. Its AFD base is `update/v0.32.0` at `2917805`.

See [A5 development notes](A5_DSV4_ASYNC.md) for the pinned #537 environment,
operator dependencies, precision/weight handling and configuration contract.
The old 0.30 launch instructions do not apply to this branch.

The retained `afd-dsv4-flash-megamoe-eager-2a2f` E2E scenario generates a 2A2F eager
configuration for later device verification; it has not been run on A5.
