# NPU troubleshooting

This guide covers common startup and runtime failures for the Ascend CAM
connectors. Use it together with the connector-specific setup guide:

- [CAM async connector](CAM_ASYNC_CONNECTOR_USER_GUIDE.md)
- [CAM P2P connector](CAM_P2P_CONNECTOR_USER_GUIDE.md)

Before troubleshooting, confirm that every Attention and FFN process uses the
same model, AFD topology, rendezvous address, and connector settings. Also use
the vLLM, vLLM-Ascend, CANN, and CAM versions documented by the selected
connector guide.

## CAM operators cannot be loaded

Typical errors include:

```text
aclnnCamMoeDistributeDispatchRecv not found
PTA call acl api failed
missing async CAM operator
```

Build the plugin-owned operators on every node and restart AFD:

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 pip install -e . -v --no-build-isolation
```

The loader logs the `afd-plugin` vendor library path and requires all four
`afd_ascend.afd_async_*` registrations. It does not fall back to external CAM.

## Runtime libraries cannot be loaded

If `libhccl.so` cannot be found, make sure the Ascend toolkit environment is
loaded before starting AFD. The plugin loader prepends its vendor paths and
preserves the existing CANN library paths.

The toolkit environment does not include the NNAL ATB library. If startup
reports that `libatb.so` cannot be found, load the ATB environment after the
toolkit environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3 -c 'import ctypes; ctypes.CDLL("libatb.so")'
```

Prefer the checked-in launch scripts so every role receives the same library
paths.

## HCCL fails to allocate memory

An `EL0004` error during startup usually means that the vLLM memory pool left
too little device memory for HCCL. The CAM async validation recipe reserves
more memory for communication with:

```bash
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
```

and starts vLLM with:

```text
--gpu-memory-utilization 0.8
```

Start with the values in the recipe for your connector. If allocation still
fails, stop stale worker processes and reduce `--gpu-memory-utilization` before
reducing `HCCL_BUFFSIZE`.

Use `npu-smi info` to check for workers left by an earlier deployment. Only
terminate processes that belong to that deployment.

## HCCL connection times out

For `EI0006`, socket timeout, or rendezvous timeout errors, verify all of the
following:

- `host` is reachable from every participating node and `port` is free;
- `num_attention_ranks` and `num_ffn_ranks` match the processes that were
  started;
- all ranks use the same connector and HCCL settings;
- multi-node ranks have working HCCL connectivity and are placed on a
  supported network topology.

Large deployments may also need a longer connection timeout:

```bash
export HCCL_CONNECT_TIMEOUT=3600
```

Apply the same timeout to every process.

AFD group rendezvous also has its own `afd_process_group_timeout_s` setting
(default 120 seconds) inside `additional_config["afd"]`. It does not inherit
`HCCL_CONNECT_TIMEOUT`; neither setting is a guaranteed deadline for a device
kernel polling CAM completion flags.

## The first CAM async request hangs

`CAMAsyncAFDConnector` requires AFD async-DP on every role:

```json
{
  "afd": {
    "connector": "CAMAsyncAFDConnector",
    "async": true,
    "compute_gate_on_attention": true
  }
}
```

For Attention data parallelism greater than one, all Attention engines must use
the AFD async-DP scheduling path. Confirm that the plugin, vLLM, and
vLLM-Ascend versions match the connector guide and are the same on every rank.
Also check that all configured Attention and FFN ranks reached connector
initialization; a missing rank prevents the CAM collective from completing.

## CAM async fails with `507015` or an AI Core timeout

Do not reduce `HCCL_BUFFSIZE` below the value used by the validated recipe. CAM
dispatch uses capacity-sized buffers, so a small request can still require the
full communication buffer.

If `connector_extra_config.hccl_buffer_size` is set, align `HCCL_BUFFSIZE`
with it on every rank. The former sizes the HCCL domain; the latter is read by
AFD host tiling for capacity and window-tail addresses. A mismatch can affect
correctness even when HCCL allocation succeeds. After changing native
operators, rebuild both the extension and vendor package on every node.

Sequence parallel deployments can also fail intermittently when asynchronous
kernel launch is enabled. Use:

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

If the failure remains, disable sequence parallelism and verify the plain TP
topology first.

## FFN workers do not exit

An idle async FFN worker can remain blocked in `async_dispatch_recv` during
shutdown. Allow the normal service shutdown to finish before forcing process
termination. Before restarting the deployment, use `npu-smi info` to confirm
that the old workers no longer hold device memory. A process left in an
uninterruptible device wait may require the platform's NPU runtime recovery
procedure.

## Inspect async MoE split shapes

For native DBO ubatching, enable the shape-only diagnostic on the
Attention process:

```bash
export AFD_ASYNC_MOE_LAYOUT_LOG=1
```

The log reports token extents, padding, CAM-local slices, and FFN result
gathering. Disable it after diagnosis to keep normal service logs concise.

Native DBO owns request/token split boundaries. Enable DBO only on
Attention, with exactly two ubatches and both token thresholds at least two.
NPU MRV2 DBO requires eager execution; synchronous CAMP2P
additionally requires Attention DP > 1, while Async CAM permits DP1.

An `NPU DBO peer failed` or `Async CAM receive failed` error means the
host-side handoff was aborted. Restart the A/F worker group; waking the peer
does not recover a partially used device window. Use the
[executor tests](TESTING.md#native-dbo-contracts-and-device-execution) to
separate thread/stream failures from communication and model correctness.
