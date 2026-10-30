# DeepSeek-V4-Flash DSpark eager (vLLM 0.32)

This recipe provides the eager path for #499: target-only Prefill, native
P-to-D target KV transfer with auxiliary-state export, and Decode with both
target and draft split across A/F. It also supports colocated prefill/decode
for isolating the A/F integration. FULL capture/replay remains a separate phase.

Use the DSV4-Flash W4A8 DSpark checkpoint with three draft layers and a block
size of five. Develop against main after #511, with pinned vLLM
`5df63a7203cdb9328d824a27b682e77c7b9928be` and vLLM-Ascend
`bd42eea01d8312c4df469fabaed45f77a12df4d3`, on Ascend 910C/A3.
Build AFD's packaged NPU operators as
described in the repository installation guide.

```text
Attention (TP1 per DP)                   FFN (four expert shards)
  target: attention, router, shared        target routed weights
       send layer 0..42  ----------------> compute
       merge            <---------------- return
  native verification and sampling
  draft: attention, router, shared         draft routed weights
       send layer 43..45 ----------------> compute
       merge            <---------------- return
  native DSpark proposal for next round
```

Checkpoint names remain `mtp.0..2`; the wire offset prevents target/draft
collisions. Draft routing is non-Hash. F loads its routed shards once and
serves every A DP from the same receive loop. The native draft algorithm,
context-KV updates, Markov head, confidence head, and rejection sampling stay
in vLLM/Ascend.

Each DP owns its A-side draft parameters, request state and draft KV. The F
group shares one read-only logical set of draft experts, partitioned across
its four ranks. Increasing A DP does not duplicate F expert weights.

## Colocated prefill/decode

Run from the repository root. Initialize CANN normally, then set the actual
model, local communication IP, and interface. Reserve distinct physical DIEs
for the roles. The example uses A on DIE 0 and F on DIEs 4–7.

```bash
MODEL=/absolute/path/to/dsv4-flash-dspark-w4a8
LOCAL_IP=<local-communication-ip>
SCRIPT=recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash/dspark_eager.py

python3 "$SCRIPT" ffn --model "$MODEL" --local-ip "$LOCAL_IP" \
  --afd-host "$LOCAL_IP" --interface eth0 --devices 4,5,6,7 >ffn.log 2>&1 &
python3 "$SCRIPT" attention --model "$MODEL" --local-ip "$LOCAL_IP" \
  --afd-host "$LOCAL_IP" --interface eth0 --devices 0 >attention.log 2>&1 &
```

Both roles must use the same checkpoint, A DP size, and CAM address/port.
For A DP2, pass `--attention-dp 2` to both commands and use `--devices 0,1`
on A. Use `--dry-run` to print the resolved command. F is connector-driven;
the user API is A's port 7100.

Fixed initial settings: max context 4096, batch-token budget 512, max requests
4, CAM receive capacity 1024, prefix caching disabled, TP1/PP1/CP1, no DBO or
EPLB, and eager target and draft. The recipe uses the legacy FFN path so the
existing SwiGLU clamp is retained. These settings are functional checks, not
throughput recommendations.

The DSpark validator rejects layered W4A8 execution until its nonzero
SwiGLU-clamp behavior is qualified.

## Separate Prefill and Decode

Use distinct devices for P, Decode-A and Decode-F. This DP2 example uses ten
DIEs on one host; tested multi-host placement is outside this recipe's scope.
Keep `MODEL`, `LOCAL_IP` and `SCRIPT` from the preceding example.

```bash
CONFIG_DIR=recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash

python3 "$SCRIPT" prefill --model "$MODEL" --local-ip "$LOCAL_IP" \
  --afd-host "$LOCAL_IP" --devices 0,1,2,3 --attention-dp 2 \
  --kv-transfer-config "$CONFIG_DIR/pd_prefill.json" >prefill.log 2>&1 &
python3 "$SCRIPT" ffn --model "$MODEL" --local-ip "$LOCAL_IP" \
  --afd-host "$LOCAL_IP" --devices 6,7,8,9 --attention-dp 2 >ffn.log 2>&1 &
python3 "$SCRIPT" attention --model "$MODEL" --local-ip "$LOCAL_IP" \
  --afd-host "$LOCAL_IP" --devices 4,5 --attention-dp 2 \
  --kv-transfer-config "$CONFIG_DIR/pd_decode.json" >attention.log 2>&1 &
```

For DP1 use `--attention-dp 1` on all roles and one A device. The launcher
keeps P/D TP and DP metadata consistent. P runs native TP4 without a drafter;
D A ranks use TP1. The JSON templates select the AFD extension of native
Mooncake, three capture boundaries and a 128-row tail. These boundaries and
the window must match the loaded checkpoint. Startup rejects a tail shorter
than the actual draft SWA requirement.

After P port 7102 and A port 7100 are healthy, run the unchanged upstream
proxy with their addresses:

```bash
python3 /path/to/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 7200 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 7102 \
  --decoder-hosts 127.0.0.1 --decoder-ports 7100
```

The native proxy forwards the additional handoff fields. P accumulates the
required tail across prefill chunks. D receives actual target pages and that
tail, uses its allocated draft slots for native projection/KV writes, and
reports transfer completion only after those writes finish. Absolute positions
and P/D allocation generations are preserved. See the
[PD contract](../../../../docs/design/module/pd_dspark_context.md).

For a performance baseline, pass `--disable-draft` to all three role launchers.
It omits speculative configuration and uses native `MooncakeConnector` without
auxiliary capture. Keep the same checkpoint, placement, batch limits and proxy.
Omit `--evidence-dir` from both measured deployments.

## Verify

After A becomes healthy, request several decode rounds:

```bash
curl -fsS http://127.0.0.1:7100/health
curl -fsS http://127.0.0.1:7100/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Check that both roles log `AFD DSpark` with wire IDs `(43, 44, 45)` and that
A's `/metrics` reports nonzero draft and accepted-token counts. Test prompt
lengths on both sides of the 128-token sliding window, a prompt longer than
512 tokens, mixed concurrent requests, and request reuse after EOS.

For test-only ownership and F dispatch evidence, pass the same absolute
`--evidence-dir` to both launchers. Use a fresh directory for each deployment.
The worker extension records host work-item
IDs without adding tensor reads or device fences. F health checks write one
JSON per CAM rank. After completed requests, each F record must show routed
experts, no attention/shared experts, and returns for layers 43, 44, and 45.
A's `afd_dspark_snapshot` collective RPC must show the opposite ownership.
Disable the fixture for ordinary serving and performance work.

The functional checker saves responses, checks early stop/cancellation and
request reuse, and verifies that every F shard returns all three draft layers:

```bash
python3 tests/e2e/models/deepseek_v4_flash/dspark_eager.py \
  --url http://127.0.0.1:7100 --output-dir /absolute/path/to/results \
  --evidence-dir /absolute/path/to/worker-evidence --attention-dp 1
```

For DP2, use `--attention-dp 2`; both A engines must report actual draft and
accepted-token counts. Omitting `--evidence-dir` permits the same request
checks against a native DSpark service.

For separate PD, give P its own evidence directory as well. Its
`afd_dspark_prefill_snapshot` RPC verifies target-only construction. Then run:

```bash
python3 tests/e2e/models/deepseek_v4_flash/dspark_eager.py \
  --url http://127.0.0.1:7200 --control-url http://127.0.0.1:7100 --pd \
  --output-dir /absolute/path/to/pd-results \
  --evidence-dir /absolute/path/to/decode-worker-evidence --attention-dp 2
```

PD cases include one-token prompts, SWA boundaries 127/128/129, a 513-token
chunked prompt, concurrent independent requests, early stop and disconnect/reuse.
The proxy handles single-prompt handoffs; colocated tests retain batched-prompt
coverage. Compare saved token IDs with a native DSpark run on the pinned
runtime, and retain any topology-dependent target-logit differences.

## Shutdown

Stop ingress first, then Decode-A, Decode-F and P, allowing the configured
60-second shutdown grace. Startup completes paired MC2 warmup before serving,
so waiting for the first request does not leave F in cold connection setup.
A seals submissions, drains queued exchanges and
publishes completion. F observes all A acknowledgements, exits its device
receivers, synchronizes and acknowledges exit before releasing resources.
An idle eager receive periodically returns without running experts, preventing
a permanently resident kernel from hitting the device watchdog.

The PD carrier joins transfer work and unregisters target and auxiliary regions
before native cache storage is freed. It uses native peer disconnection when
the engine provides that interface. An incomplete read,
write or receive prevents normal resource release and reports failure. Process
termination after such a failure is cleanup, not an orderly-shutdown pass.
