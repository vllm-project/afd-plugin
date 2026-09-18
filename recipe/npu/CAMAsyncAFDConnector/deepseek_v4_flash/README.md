# DeepSeek-V4-Flash Async CAM Recipe

This recipe launches DeepSeek-V4-Flash W8A8 on one 16-NPU Ascend 910C node:

- NPUs 0-7: Attention `DP4 x TP2`, FlashComm1 sequence parallelism enabled;
- NPUs 8-15: FFN `DP8 x TP1 / EP8`, FlashComm1 disabled;
- one 16-rank CAM world containing 8 Attention and 8 FFN ranks;
- AFD-managed two-stage token split, dynamic CAM quantization, eager execution,
  chunked prefill, and the vLLM async scheduler;
- `max_num_batched_tokens=65536`, shared compressor workspace, and a 4096 MB
  CAM HCCL buffer.

The two roles share one node but run as separate vLLM processes. Only the
Attention process owns the public inference endpoint.

> [!WARNING]
> DeepSeek-V4 support over `CAMAsyncAFDConnector` remains experimental. This is
> the user-facing launch tool stack for the performance work tracked by
> [AFD issue #227](https://github.com/vllm-project/afd-plugin/issues/227),
> rewritten against the configuration contract on
> `vllm-project/afd-plugin:main`. The performance numbers below are historical
> evidence for the recorded validation cell; this recipe itself has not been
> requalified on NPU hardware.

For the connector contract and package installation instructions, read the
[CAM Async Connector User Guide](../../../../docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md).

## Runtime and model

The issue #227 performance cell used:

- Ascend 910C, Python 3.12, and CANN 9.0.1;
- vLLM commit `568afb3a1`;
- vLLM-Ascend commit `80d8c194f`, the `2e01d4c5` head commit from the
  still-unmerged
  [vLLM-Ascend PR #15452](https://github.com/vllm-project/vllm-ascend/pull/15452),
  and the follow-up slot-mapping fix `e19e14da`;
- CAM 209.x operator packages documented in the connector guide;
- the complete `DeepSeek-V4-Flash-w8a8-mtp` checkpoint.

PR #15452 commit `2e01d4c5` provides the DeepSeek-V4 compressor-tail workspace
reuse; `e19e14da` applies its slot-mapping correction on top. Check out or
apply those exact commits in order. PR #15452 is still open, so do not assume a
released vLLM-Ascend checkout already contains CWS support.

The checkpoint's `config.json` must select `model_type=deepseek_v4` or the
`DeepseekV4ForCausalLM` architecture. The checkpoint may contain MTP weights,
but this recipe does not enable speculative decoding.

Install the AFD plugin from the target `main` checkout and install the CAM
operators before launching. Source the matching CANN environment in each
terminal. Verify that the four CAM async operators are registered:

```bash
python3 - <<'PY'
import torch
import umdk_cam_op_lib  # noqa: F401

for name in (
    "async_dispatch_send",
    "async_dispatch_recv",
    "async_combine_send",
    "async_combine_recv",
):
    assert hasattr(torch.ops.umdk_cam_op_lib, name), name
print("CAM async operators are available")
PY
```

## Why these settings

The issue #227 performance matrix established the operational contract used
here:

- FlashComm1 must be enabled only for Attention; FFN TP1 rejects it.
- Attention `DP4TP2` plus FFN `EP8` was the best measured single-node layout.
- Token split outperformed request split under sustained load in that layout.
- `max_num_batched_tokens=65536` requires the DeepSeek-V4 shared compressor
  workspace from vLLM-Ascend `e19e14da7`.
- The `DP4TP2 / EP8`, MBT 65536 run failed with a smaller HCCL buffer and was
  stable with 4096 MB.
- Both roles must use identical CAM rank counts, split settings, rendezvous
  address, and `max_num_batched_tokens`.

Do not add vLLM native DBO flags. `async_moe_ubatching` is the separate,
AFD-managed MoE pipeline. The recipe also deliberately leaves the experimental
`prefill_token_sum` Attention DPLB policy unset; the issue #227 measurements
found no throughput benefit over the request-count policy for the measured
workload, and that policy is not part of the target `main` configuration
contract.

## Launch

Run both commands on the same 16-NPU node. `LOCAL_IP` must be the address on
`NIC_NAME`, and the same reachable address is used as the CAM rendezvous host.
The launcher defaults to Attention devices 0-7 and FFN devices 8-15.

Start FFN in terminal 1:

```bash
source /usr/local/Ascend/cann-9.0.1/set_env.sh

MODEL_PATH=/path/to/DeepSeek-V4-Flash-w8a8-mtp \
LOCAL_IP=<node-ip> \
NIC_NAME=<npu-nic> \
bash recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash/launch.sh ffn
```

Start Attention in terminal 2:

```bash
source /usr/local/Ascend/cann-9.0.1/set_env.sh

MODEL_PATH=/path/to/DeepSeek-V4-Flash-w8a8-mtp \
LOCAL_IP=<node-ip> \
NIC_NAME=<npu-nic> \
bash recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash/launch.sh attention
```

Wait for the Attention endpoint:

```bash
until curl --fail --silent http://127.0.0.1:8000/v1/models >/dev/null; do
  sleep 5
done
```

Then send a smoke request:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek_v4_flash",
    "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
    "max_tokens": 1
  }'
```

## Overrides

The launcher accepts these optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AFD_HOST` | `LOCAL_IP` | CAM rendezvous host; must identify Attention rank 0. |
| `AFD_PORT` | `1239` | CAM rendezvous port. |
| `ATTENTION_PORT` / `FFN_PORT` | `8000` / `8001` | Role-local API ports. |
| `ATTENTION_DEVICES` / `FFN_DEVICES` | `0-7` / `8-15` | Comma-separated device lists. |
| `MAX_MODEL_LEN` | `70000` | Maximum sequence length. |
| `MAX_NUM_BATCHED_TOKENS` | `65536` | Scheduler MBT; it must match on both roles. |
| `MAX_NUM_SEQS` | `128` | Maximum scheduled sequences. |
| `GPU_MEMORY_UTILIZATION` | `0.8` | Per-role device memory utilization. |
| `HCCL_BUFFER_SIZE_MB` | `4096` | Connector-scoped CAM HCCL buffer size. |
| `CAM_VENDOR_PATH` | CANN 9.0.1 CAM vendor path | Installed CAM operator root. |

Keep the topology and model-semantic settings unchanged unless the new
combination is validated independently.

## Issue #227 performance summary

The matching single-node `DP4TP2 / EP8`, token-split, MBT 65536 matrix used a
512-request long-prefill workload. Every rate completed 512/512 requests with
no failures:

| Offered rate | Effective input tokens/s | TTFT p99 | Peak 15s service rate |
| --- | ---: | ---: | ---: |
| 0.5x | 17,075 | 9.0s | 30K tokens/s |
| 0.75x | 25,230 | 10.7s | 41K tokens/s |
| 1.0x | 32,830 | 15.6s | 45K tokens/s |
| 1.25x | 37,321 | 25.6s | 47K tokens/s |
| 1.5x | 40,096 | 34.3s | 47K tokens/s |

Against the best baseline point at each offered rate, the recorded AFD cell
improved effective throughput by 4.8% at 1.0x, 6.6% at 1.25x, and 8.6% at
1.5x. At 1.0x, TTFT p50 improved by 37.4%, TTFT p99 by 22.1%, and the fraction
of requests meeting a 10-second TTFT SLO increased from 65.2% to 85.2%.

These values describe the exact issue #227 software, checkpoint, and workload
cell. They are not a portable performance guarantee or a validation result for
a different checkout.

## Limitations

Current constraints include:

- eager execution only; ACL graph mode is unsupported;
- FlashComm1 only on Attention and only with TP greater than one;
- no context parallelism, prefix caching, speculative decoding, KV transfer,
  or native DBO;
- exactly two AFD MoE stages;
- W8A8 CAM dynamic quantization is required for DeepSeek-V4-Flash;
- all ranks need the same CAM/CANN packages, connector settings, and model;
- teardown must remove both vLLM process trees before relaunching, or stale
  workers can retain NPU memory and CAM port 1239.
