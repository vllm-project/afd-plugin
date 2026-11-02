# DeepSeek-V4-Flash DSpark eager (vLLM 0.32)

This recipe runs target-only Prefill and A/F-separated target and draft models
on Decode. It uses native `vllm serve` commands. FULL capture/replay is a
separate phase of #499.

Use the DSV4-Flash W4A8 checkpoint with three draft layers and a five-token
draft block. The qualified runtime is Ascend 910C/A3 with vLLM
`5df63a7203cdb9328d824a27b682e77c7b9928be` and vLLM-Ascend
`bd42eea01d8312c4df469fabaed45f77a12df4d3`. Install AFD and build its packaged
NPU operators using the repository installation guide, then initialize CANN.

```text
P TP4: target only
  target KV + auxiliary tail -> Decode initializes allocated draft SWA slots

Decode-A TP1 per DP                     Shared Decode-F EP4
  target attention/router/shared         target routed weights
       layer 0..42  -------------------> compute and return
  native verification and sampling
  draft attention/router/shared          draft routed weights
       layer 43..45 -------------------> compute and return
  native proposal for the next iteration
```

Each A DP owns its draft parameters, request state and draft KV. The F group
loads one read-only logical draft expert set, partitioned across four ranks.
Checkpoint/KV names remain `mtp.0..2`; CAM layer IDs are a separate namespace.
Draft routing is non-Hash. Proposal, verification, sampling and KV updates
remain native. See the [PD state contract](../../../../docs/design/module/pd_dspark_context.md).

## Environment

Run the following blocks from the repository root in the same Bash shell.
Replace the model path, communication IP and interface. The default placement
uses nine distinct DIEs: P 0–3, A 4, F 5–8.

```bash
MODEL=/absolute/path/to/dsv4-flash-dspark-w4a8
LOCAL_IP=<local-communication-ip>
INTERFACE=eth0
CONFIG_DIR=recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash
NUM_A=1
A_DEVICES=4
F_DEVICES=5,6,7,8

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
  --shutdown-timeout 60 --enable-expert-parallel --enforce-eager --seed 1024
  --max-model-len 4096 --max-num-batched-tokens 512 --max-num-seqs 4
  --block-size 128 --gpu-memory-utilization 0.85 --quantization ascend
  --attention_config.indexer_kv_dtype int8 --tokenizer-mode deepseek_v4
  --trust-remote-code --no-enable-prefix-caching --enable-chunked-prefill
  --no-disable-hybrid-kv-cache-manager
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}'
)
DSPARK_CONFIG="{\"method\":\"dspark\",\"model\":\"$MODEL\",\"num_speculative_tokens\":5,\"enforce_eager\":true}"
```

For DP2, set `NUM_A=2`, `A_DEVICES=4,5`, and `F_DEVICES=6,7,8,9`. In **both**
PD JSON files, change `kv_connector_extra_config.decode.dp_size` to `2`.
The P TP size remains four. The context window and capture boundaries in the
JSON files must match the checkpoint; startup rejects a tail shorter than the
loaded draft SWA window. Receive capacity is 1024 rows. The legacy FFN path
preserves the native SwiGLU clamp; layered W4A8 is rejected in this phase.

## Separate Prefill and Decode

P loads no draft. F does not participate in P-to-D KV transfer. Start P, F and
A in this order; paired A/F communication warmup completes before serving.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" \
  --port 7102 --data-parallel-rpc-port 17102 \
  --tensor-parallel-size 4 --data-parallel-size 1 \
  --all2all-backend allgather_reducescatter \
  --worker-cls afd_plugin.v1.worker.npu.dspark_prefill.AFDDSparkPrefillWorker \
  --kv-transfer-config "$(cat "$CONFIG_DIR/pd_prefill.json")" \
  --additional-config '{
    "enable_cpu_binding":false,"enable_force_eplb":false,
    "enable_dsa_cp":false,"multistream_dsv4_dsa_overlap":false
  }' >prefill.log 2>&1 &
```

```bash
ASCEND_RT_VISIBLE_DEVICES="$F_DEVICES" VLLM_PLUGINS=ascend,afd \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" \
  --port 7101 --data-parallel-rpc-port 17101 \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --all2all-backend flashinfer_all2allv --speculative-config "$DSPARK_CONFIG" \
  --additional-config "{
    \"enable_cpu_binding\":false,\"enable_force_eplb\":false,
    \"enable_dsa_cp\":false,\"multistream_dsv4_dsa_overlap\":false,
    \"afd\":{
      \"role\":\"ffn\",\"connector\":\"CAMAsyncAFDConnector\",\"async_dp\":true,
      \"host\":\"$LOCAL_IP\",\"port\":12499,
      \"num_attention_ranks\":$NUM_A,\"num_ffn_ranks\":4,
      \"compute_gate_on_attention\":true,
      \"connector_extra_config\":{\"dynamicQuant\":1,\"attn_ranks_per_dp\":1}
    }
  }" >ffn.log 2>&1 &
```

```bash
ASCEND_RT_VISIBLE_DEVICES="$A_DEVICES" VLLM_PLUGINS=ascend,afd \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" \
  --port 7100 --data-parallel-rpc-port 17100 \
  --tensor-parallel-size 1 --data-parallel-size "$NUM_A" \
  --all2all-backend allgather_reducescatter --speculative-config "$DSPARK_CONFIG" \
  --kv-transfer-config "$(cat "$CONFIG_DIR/pd_decode.json")" \
  --additional-config "{
    \"enable_cpu_binding\":false,\"enable_force_eplb\":false,
    \"enable_dsa_cp\":false,\"multistream_dsv4_dsa_overlap\":false,
    \"afd\":{
      \"role\":\"attention\",\"connector\":\"CAMAsyncAFDConnector\",\"async_dp\":true,
      \"host\":\"$LOCAL_IP\",\"port\":12499,
      \"num_attention_ranks\":$NUM_A,\"num_ffn_ranks\":4,
      \"compute_gate_on_attention\":true,
      \"connector_extra_config\":{\"dynamicQuant\":1,\"attn_ranks_per_dp\":1}
    }
  }" >attention.log 2>&1 &
```

After P port 7102 and A port 7100 are healthy, start the unchanged upstream PD
proxy. F has no user-facing request API.

```bash
python3 /path/to/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 7200 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 7102 \
  --decoder-hosts 127.0.0.1 --decoder-ports 7100
```

For **colocated prefill/decode**, omit P and the proxy, omit A's
`--kv-transfer-config`, and use separate A/F devices, for example A 0 and
F 4–7. Send requests directly to A port 7100.

For the **speculation-disabled baseline**, omit `--speculative-config` on A/F
and omit the custom `--worker-cls` on P. In both PD JSON files, use
`"kv_connector":"MooncakeConnector"` and
`"kv_connector_module_path":"vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"`,
and remove the `dspark_*` keys. Keep the same checkpoint, placement and budgets.

## Verification

```bash
curl -fsS http://127.0.0.1:7100/health
python3 tests/e2e/models/deepseek_v4_flash/dspark_eager.py \
  --url http://127.0.0.1:7200 --control-url http://127.0.0.1:7100 --pd \
  --output-dir /absolute/path/to/pd-results --attention-dp "$NUM_A"
```

The checker covers one-token prompts, SWA 127/128/129, a 513-token chunked
prompt, concurrent requests, early stop, disconnect/reuse, and nonzero draft
and accepted-token metrics on each A DP. It saves raw responses for comparison
with native DSpark. For colocated testing, use A's URL and omit `--pd` and
`--control-url`; this also tests batched prompts.

Optional ownership evidence uses vLLM's existing worker extension hook. Before
starting A/F, set `PYTHONPATH` to the repository root,
`VLLM_SERVER_DEV_MODE=1` and `AFD_DSPARK_EVIDENCE_DIR` to a fresh absolute
directory, and add
`--worker-extension-cls tests.e2e.models.deepseek_v4_flash.dspark_evidence.DSparkEagerEvidence`
to both commands. Pass that directory as `--evidence-dir` to the checker.
For P, use a separate directory and
`--worker-extension-cls tests.e2e.models.deepseek_v4_flash.dspark_prefill_evidence.DSparkPrefillEvidence`;
its `afd_dspark_prefill_snapshot` RPC checks target-only construction.
Disable these observers for performance measurements.

## Shutdown and limits

Stop ingress, then Decode-A, Decode-F and P, allowing the configured 60-second
grace period. A seals/drains submissions; F exits its bounded receive and
acknowledges device completion. PD transfers and KV writes finish before their
memory is unregistered. Incomplete work retains resources and reports failure.

The pinned A3 runtime has been exercised with P TP4, A DP1/DP2 and F EP4 on
one host, including in-flight Decode/F shutdown. Four fixed native DSpark
controls matched all 128 generated token IDs in each PD topology. This is not
general model-accuracy evidence: colocated A TP1 can diverge from native TP4 at
a quantized target-logit tie.

Native Prefill finalization can still report heap corruption after region
unregistration; it was also reproduced with native Mooncake and speculation
disabled. Verified Decode/F drain does not establish clean Prefill process
exit. Multi-host placement, prefix reuse, general preemption stress, peer
failure recovery and FULL graphs require separate qualification.
