# DeepSeek-V4-Flash DSpark A/F execution

This recipe configures target and draft A/F execution using the existing
Async CAM transport. Attention retains attention, routing, shared experts,
request KV and native speculative decoding. The FFN group loads one read-only
draft expert set, sharded by EP, and serves target and draft from the same loop.

Use a DSV4-Flash W4A8 or W8A8 checkpoint with three draft layers and a five-token
draft block. The runtime baseline is vLLM
`5df63a7203cdb9328d824a27b682e77c7b9928be` and vLLM-Ascend
`bd42eea01d8312c4df469fabaed45f77a12df4d3` on Ascend 910C/A3.
Install AFD using the repository NPU installation guide and initialize CANN.

```text
Attention Runner                       Existing Async FFN Runner
  target attention/router/shared        target routed weights
       layers 0..42 ------------------> compute and return
  native verification and sampling
  draft attention/router/shared         draft routed weights
       layers 43..45 -----------------> compute and return
  native proposal for the next iteration
```

Native checkpoint/KV names remain `mtp.0..2`; CAM layer IDs are a separate
namespace. Draft routing is non-Hash. AFD wraps native proposal execution with
the draft's AFD metadata; sampling, verification and KV updates remain native.

## A/F configuration

Run from the repository root in Bash. Replace the model path, communication IP
and network interface. A TP1/DP1 and F EP4 use five distinct DIEs.

```bash
MODEL=/absolute/path/to/dsv4-flash-dspark-w4a8
LOCAL_IP=REPLACE_WITH_COMMUNICATION_IP
INTERFACE=eth0
NUM_A=1
NUM_F=4
A_DEVICES=0
F_DEVICES=4,5,6,7
F_LAYERED_GMM=0
FFN_EXEC_ARGS=(--enforce-eager)

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export AFD_ASYNC_CAM_LAYERED_GMM=0 BATCH_SIZE_FACTOR=0.00390625
export HCCL_SOCKET_IFNAME="$INTERFACE" GLOO_SOCKET_IFNAME="$INTERFACE"
export TP_SOCKET_IFNAME="$INTERFACE" HCCL_IF_IP="$LOCAL_IP"
export HCCL_BUFFSIZE=256 LCCL_BUFFER_SIZE=408 OMP_NUM_THREADS=4

OPP=$(python3 -c 'from pathlib import Path; import afd_plugin; print(Path(afd_plugin.__file__).resolve().parent / "_cann_ops_custom/vendors/afd-plugin")')
export ASCEND_CUSTOM_OPP_PATH="$OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export AFD_CUST_OPAPI_LIB_PATH="$OPP/op_api/lib/libcust_opapi.so"
export LD_LIBRARY_PATH="$OPP/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

COMMON_ARGS=(
  --host 0.0.0.0 --served-model-name dsv4 --api-server-count 1
  --enable-expert-parallel --seed 1024
  --max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 4
  --block-size 128 --gpu-memory-utilization 0.85 --quantization ascend
  --attention_config.indexer_kv_dtype int8 --tokenizer-mode deepseek_v4
  --trust-remote-code --no-enable-prefix-caching --enable-chunked-prefill
  --no-disable-hybrid-kv-cache-manager
)
DSPARK_CONFIG="{\"method\":\"dspark\",\"model\":\"$MODEL\",\"num_speculative_tokens\":5,\"enforce_eager\":true}"
```

For DP2 set `NUM_A=2` and `A_DEVICES=0,1`. The W4A8 example uses F EP4.
For a W8A8 checkpoint, size F for its larger weights; the validated W8A8
configuration uses `NUM_F=8` and `F_DEVICES=8,9,10,11,12,13,14,15`.
Target and draft Attention use eager execution. F supports legacy eager,
layered eager and layered FULL, with one combined target/draft weight table.
The example selects legacy F. Set `F_LAYERED_GMM=1` for layered eager; keep
`AFD_ASYNC_CAM_LAYERED_GMM=0` for A and P.
The existing layered eager path can be slower for small decode batches; use
ordinary GMM eager when choosing this deployment for eager throughput.

For F FULL, replace the F settings above before launching:

```bash
F_LAYERED_GMM=1
FFN_EXEC_ARGS=(--compilation-config '{"cudagraph_mode":"FULL"}')
```

F captures one DR → layered FFN → CS transaction and repeatedly replays it for
both target and draft layers. The bounded replay group is independent of request
and target/draft phase boundaries. A and the native draft configuration retain
eager execution. FULL requires matching W4A8 or W8A8 layouts on all layers; unsupported
graph modes fail startup. DSpark does not support DBO or FFN two-stream.

Layered W4A8 forwards the model's `swiglu_limit` for both target and draft
experts in eager and FULL. A positive limit clamps gate above and up on both
sides before SwiGLU; zero disables clipping. Rebuild the AFD native extension
and CANN package together when upgrading this interface. Layered W8A8 retains
its unclipped computation; use legacy F when W8A8 checkpoint clamp behavior
is required.

Launch F and A with their usual Async CAM arguments, adding the same
`--speculative-config "$DSPARK_CONFIG"` to both:

```bash
ASCEND_RT_VISIBLE_DEVICES="$F_DEVICES" VLLM_PLUGINS=ascend,afd \
AFD_ASYNC_CAM_LAYERED_GMM="$F_LAYERED_GMM" \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" "${FFN_EXEC_ARGS[@]}" \
  --port 7101 --data-parallel-rpc-port 17101 \
  --tensor-parallel-size 1 --data-parallel-size "$NUM_F" \
  --all2all-backend flashinfer_all2allv --speculative-config "$DSPARK_CONFIG" \
  --additional-config "{
    \"enable_cpu_binding\":false,\"enable_force_eplb\":false,
    \"enable_dsa_cp\":false,\"multistream_dsv4_dsa_overlap\":false,
    \"afd\":{
      \"role\":\"ffn\",\"connector\":\"CAMAsyncAFDConnector\",\"async_dp\":true,
      \"host\":\"$LOCAL_IP\",\"port\":12499,
      \"num_attention_ranks\":$NUM_A,\"num_ffn_ranks\":$NUM_F,
      \"compute_gate_on_attention\":true,
      \"connector_extra_config\":{\"dynamicQuant\":1,\"attn_ranks_per_dp\":1}
    }
  }" >ffn.log 2>&1 &
```

```bash
ASCEND_RT_VISIBLE_DEVICES="$A_DEVICES" VLLM_PLUGINS=ascend,afd \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" --enforce-eager \
  --port 7100 --data-parallel-rpc-port 17100 \
  --tensor-parallel-size 1 --data-parallel-size "$NUM_A" \
  --all2all-backend allgather_reducescatter --speculative-config "$DSPARK_CONFIG" \
  --additional-config "{
    \"enable_cpu_binding\":false,\"enable_force_eplb\":false,
    \"enable_dsa_cp\":false,\"multistream_dsv4_dsa_overlap\":false,
    \"afd\":{
      \"role\":\"attention\",\"connector\":\"CAMAsyncAFDConnector\",\"async_dp\":true,
      \"host\":\"$LOCAL_IP\",\"port\":12499,
      \"num_attention_ranks\":$NUM_A,\"num_ffn_ranks\":$NUM_F,
      \"compute_gate_on_attention\":true,
      \"connector_extra_config\":{\"dynamicQuant\":1,\"attn_ranks_per_dp\":1}
    }
  }" >attention.log 2>&1 &
```

## Prefill/Decode deployment

P uses the upstream worker, complete DSpark model and upstream KV transfer
configuration. Apply the upstream speculative configuration to P as required
by that deployment. Configure A's native KV consumer through
`--kv-transfer-config`; F has no request KV and is not a PD participant.
The pinned Ascend DSV4-Flash A3 recipe uses `MooncakeHybridConnector`.
If `VLLM_PLUGINS` is restricted, include `ascend_kv_connector` on both P and A
to register the upstream connector.

## Verification

```bash
python3 tests/e2e/models/deepseek_v4_flash/dspark_eager.py \
  --url http://127.0.0.1:7100 \
  --output-dir /absolute/path/to/dspark-results \
  --attention-dp "$NUM_A" --ffn-ranks "$NUM_F"
```

For a native PD deployment, point `--url` at its proxy and add
`--control-url http://127.0.0.1:7100 --pd`. The checker saves responses for
native DSpark comparison, exercises SWA boundaries and chunked prompts, and
checks proposal/acceptance counters. Optional ownership evidence uses
`tests.e2e.models.deepseek_v4_flash.dspark_evidence.DSparkEagerEvidence`
through the native worker-extension hook; disable observers when benchmarking.
For this check, enable `VLLM_SERVER_DEV_MODE=1` on A/F, add the extension with
`--worker-extension-cls`, and set `AFD_DSPARK_EVIDENCE_DIR` to F's evidence
directory. Pass that directory to the checker with `--evidence-dir`. In layered
mode it requires all 46 target/draft layers to return work on each F rank.
Layered execution observations read device headers and add synchronization;
their timings are not performance measurements.

For graph numerical checks, use
`tests.e2e.models.deepseek_v4_flash.dspark_ffn_graph_evidence.DSparkFFNGraphEvidence`
on F and the ordinary evidence extension on A. After startup, create
`$AFD_DSPARK_EVIDENCE_DIR/graph-proof` and its empty `enable` marker. This functional-only
extension submits one replay per completion fence and compares each completed
layer with the original layered eager tables, saving inputs and outputs. It
also collects capture/replay and dispatch evidence. Add `--evidence-dir` and
`--verify-graph-numerics` to the checker command. The checker requires complete
target/draft layer coverage and successful nonempty bitwise comparisons on every
declared F rank before reporting success. Missing or incomplete evidence fails
after the bounded evidence wait; a numerical mismatch fails immediately.
The `graph-proof/enable` marker also enables this validation automatically.
Use a fresh evidence directory for each deployment. The checker copies the
validated records to `graph-proof.json` in its output directory and records
`graph_numerics_verified` in `result.json`. Run ordinary bounded-group
liveness checks and every performance comparison without either extension.

For a layered/reference FFN performance comparison, keep DSpark enabled on P,
A and F in both runs, toggle only F's `AFD_ASYNC_CAM_LAYERED_GMM`, and disable
the evidence extension. For a FULL/layered-eager comparison, keep layered GMM
enabled in both runs and change only F's execution arguments.

Remove `--speculative-config` on A/F for a speculation-disabled control, and
use the corresponding upstream P configuration when testing PD. Record the
actual checkpoint, topology, input, sampling, cache and runtime settings.
