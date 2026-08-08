# CAMAsyncAFDConnector For DeepSeek-V3.2 Recipe

> [!NOTE]
> The commands and measurements in this document target
> vLLM/vLLM-Ascend `v0.19.1rc1`. Use the AFD Plugin branch
> `release/v0.19.1rc1` when running them. For the current v0.26 full-model
> DP2TP8 Attention + EP16 accuracy test, use the
> [v0.26 accuracy recipe](v0_26_accuracy/README.md).

This recipe describes how to run DeepSeek-V3.2 with the AFD CAM async
connector on Ascend NPU.

For the connector's complete configuration contract, rank derivation, data
flow, native DBO distinction, and limitations, see the
[CAM Async Connector User Guide](../../../../docs/npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md).

## Introduction

`CAMAsyncAFDConnector` provides an Ascend CAM-backed asynchronous AFD connector
for the Attention-FFN disaggregation path. It uses CAM async dispatch/combine
operators to move MoE activations between attention-side and FFN-side ranks,
allowing the prefill computation to be split across independent attention and
expert workers.

Current limitation: CAM async currently supports only the prefill stage in a
prefill/decode disaggregated deployment.

AFD provides the following backend-specific connectors:

- GPU: `P2pNcclAFDConnector`.
- NPU: `CAMP2pAFDConnector` and `CAMAsyncAFDConnector`.

## Image and Hardware Requirements

- Hardware: Ascend 910C only.
- Image: `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler`.

```bash
docker pull quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler
```

## Installing Operator Packages

Run the following commands from the repository root inside the container:

```bash
bash afd_plugin/connectors/npu/bin/CAM_ascend910_93_openEuler_aarch64.run
pip install afd_plugin/connectors/npu/bin/umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl
```

## AFD Config Explanation

The AFD runtime is enabled through the `afd` object passed to
`--additional-config`. The same topology-level values must be used by the
attention and FFN commands so all workers join the same CAM async group.

In an async deployment, `--max-num-batched-tokens` must also be set to the
same value on the Attention (A) and FFN (F) sides. This recipe uses `140000`
for both sides.

| Field | Meaning |
|-------|---------|
| `connector` | Selects the AFD connector implementation. Use `CAMAsyncAFDConnector` for CAM async. |
| `async` | Enables async-DP execution, which is required by `CAMAsyncAFDConnector`. |
| `role` | Worker role in the AFD split. Use `attention` for prefill attention workers and `ffn` for expert workers. |
| `host` / `port` | Rendezvous address for the async CAM HCCL process group. Set `host` to the IP address of the node that owns attention rank 0; all attention and FFN workers must use the same `host` and `port`. |
| `num_attention_ranks` | Total attention-side ranks in the AFD topology. In this recipe, `DP3PCP8` gives `3 * 8 = 24`. |
| `num_ffn_ranks` | Total FFN-side ranks in the AFD topology. In this recipe, `EP8` gives `8`. |
| `compute_gate_on_attention` | Runs MoE routing/gating on the attention side before dispatching activations to FFN ranks. |

`connector_extra_config` carries CAM async-specific knobs:

| Field | Meaning |
|-------|---------|
| `dynamicQuant` | Enables dynamic quantization metadata for CAM dispatch/combine. |
| `async_moe_ubatching` | Enables AFD-managed MoE ubatching instead of vLLM native DBO. |
| `async_moe_num_ubatches` | Number of async MoE stages. The current CAM async setup uses `2`. |
| `async_moe_split` | Split policy for async MoE ubatches. This recipe uses request-level splitting. |
| `attn_ranks_per_dp` | Number of attention ranks per DP replica. With `PCP8`, this value is `8`. |

Do not add `--enable-dbo`, `--dbo-decode-token-threshold`, or
`--dbo-prefill-token-threshold` to these commands. Those flags enable vLLM
native DBO, which CAM async rejects. `async_moe_ubatching` is AFD-managed,
MoE-only request-boundary staging and is not vLLM native DBO.

## Experiment Configuration

### Model

- Model: DeepSeekV3.2 W8A8.
- Hardware: 2 Node Ascend 910C.
- Because of device-count limits, the experiment uses a reduced model with the
  first 10 layers only: 3 dense layers and 7 MoE layers.

### Benchmark

**NOTE:** CAM async supports only the prefill stage. The benchmark deployment
below does not use prefill/decode disaggregation because the dataset uses an
output length of `1` to simulate a prefill workload.

Dataset:

- File:
  [`tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl`](../../../../tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl)
- The prompt-length distribution is derived from a real-world workload.
- Prompt text is randomized and intentionally has no semantic meaning.
- Every request uses an output length of `1`.

The dataset is stored with Git LFS and excluded from the default LFS fetch, so
a normal clone does not download its contents. Before running this benchmark,
fetch the dataset from the repository root:

```bash
git lfs pull \
  --include="tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl" \
  --exclude=""
```

Run the benchmark command from the repository root:

```bash
vllm bench serve \
  --backend vllm \
  --model deepseek_v3_2 \
  --tokenizer /path/to/DeepSeek-V3.2 \
  --tokenizer-mode auto \
  --trust-remote-code \
  --endpoint /v1/completions \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name custom \
  --dataset-path tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl \
  --skip-chat-template \
  --custom-output-len 1 \
  --request-rate 10 \
  --no-oversample \
  --disable-shuffle \
  --num-warmups 0 \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 25,50,90,95,99 \
  --save-result \
  --save-detailed \
  --result-dir ./bench_results \
  --result-filename ttft_by_prompt_len.json
```

### Baseline

- Topology: `DP4PCP8`.
- Forced expert balancing is enabled through `additional_config`. The baseline
  leaves `force_load_balance_topn_per_rank` unset so all routed experts
  participate.

<details>
<summary>Node0 Deployment Command (DP4PCP8)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --max-num-seqs 8 \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --no-enable-chunked-prefill \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --additional-config '{
    "enable_force_load_balance": true
  }'
```

</details>

<details>
<summary>Node1 Deployment Command (DP4PCP8)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1

vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --headless \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --max-num-seqs 8 \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --no-enable-chunked-prefill \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --additional-config '{
    "enable_force_load_balance": true
  }'
```
</details>

### AFD CAM async

- Attention side: `DP3PCP8`.
- FFN side: `EP8`.
- Connector: `CAMAsyncAFDConnector`.
- Current scope: PD-disaggregated prefill stage only.

<details>
<summary>Node0 Attention Deployment Command (DP0-DP1)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export AFD_FORCE_BALANCED_TOPK_IDS=1

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 32 \
  --enforce-eager \
  --served-model-name deepseek_v3_2 \
  --quantization ascend \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --prefill-context-parallel-size 8 \
  --no-enable-chunked-prefill \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --additional-config '{
    "afd": {
      "connector": "CAMAsyncAFDConnector",
      "async": true,
      "role": "attention",
      "host": "33.215.117.43",
      "port": 1239,
      "num_attention_ranks": 24,
      "num_ffn_ranks": 8,
      "compute_gate_on_attention": true,
      "connector_extra_config": {
        "dynamicQuant": 1,
        "async_moe_ubatching": true,
        "async_moe_num_ubatches": 2,
        "async_moe_split": "request",
        "attn_ranks_per_dp": 8
      }
    }
  }'
```

</details>

<details>
<summary>Node1 Attention Deployment Command (DP2)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export AFD_FORCE_BALANCED_TOPK_IDS=1

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /path/to/DeepSeek-V3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 32 \
  --served-model-name deepseek_v3_2 \
  --enforce-eager \
  --quantization ascend \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 33.215.117.43 \
  --data-parallel-rpc-port 29550 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --headless \
  --prefill-context-parallel-size 8 \
  --no-enable-chunked-prefill \
  --enable-expert-parallel \
  --additional-config '{
    "afd": {
      "connector": "CAMAsyncAFDConnector",
      "async": true,
      "role": "attention",
      "host": "33.215.117.43",
      "port": 1239,
      "num_attention_ranks": 24,
      "num_ffn_ranks": 8,
      "compute_gate_on_attention": true,
      "connector_extra_config": {
        "dynamicQuant": 1,
        "async_moe_ubatching": true,
        "async_moe_num_ubatches": 2,
        "async_moe_split": "request",
        "attn_ranks_per_dp": 8
      }
    }
  }'
```

</details>

<details>
<summary>Node1 FFN Deployment Command (EP8)</summary>

```bash
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV

export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export AFD_FORCE_BALANCED_TOPK_IDS=1

ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
vllm serve /path/to/DeepSeek-V3.2 \
  --port 8001 \
  --max-num-seqs 2 \
  --enforce-eager \
  --quantization ascend \
  --max-num-batched-tokens 140000 \
  --served-model-name deepseek_v3_2 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --additional-config '{
    "afd": {
      "connector": "CAMAsyncAFDConnector",
      "async": true,
      "role": "ffn",
      "host": "33.215.117.43",
      "port": 1239,
      "num_attention_ranks": 24,
      "num_ffn_ranks": 8,
      "compute_gate_on_attention": true,
      "connector_extra_config": {
        "dynamicQuant": 1,
        "async_moe_ubatching": true,
        "async_moe_num_ubatches": 2,
        "async_moe_split": "request",
        "attn_ranks_per_dp": 8
      }
    }
  }'
```

</details>

## Experiment Results

**Note: The results below were measured with forced expert balancing enabled
and with the reduced 10-layer model described above.**

![Text-matched dataset median TTFT comparison](text_matched_dp_afd_median_ttft.png)

On the dataset mentioned above, AFD CAM async consistently reduces Median/P50
TTFT compared with the `DP4PCP8 TP1` baseline across the measured request
rates. The gap becomes more visible at higher load: at 10 RPS and 12 RPS, AFD
is about 7.2s faster than the baseline, with 12 RPS improving from 15.1s to
8.0s.
