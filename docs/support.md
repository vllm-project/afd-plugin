# Supported configurations

AFD is experimental. Registration and accepted configuration do not establish
hardware qualification; evidence below is scoped to its model, topology and
source revision. See [installation](installation.md) for the pinned runtime stack.

## Models

| Model family | Execution scope |
| --- | --- |
| DeepSeek V2 / V3 / V3.2 | CUDA P2P and synchronous Ascend paths; DeepEP M2N admits V2 only. Each role loads its required components. |
| DeepSeek V4 Flash | CUDA P2P or Ascend CAMP2P / Async CAM; backend-specific limits are in the matrix below. |
| Qwen3 MoE | CUDA; P2P gates on FFN, DeepEP M2N gates on Attention. Rejects sequence-parallel MoE, EPLB, PP, speculative decoding, LoRA and NPU. |
| Qwen3.5 / Qwen3.6 MoE | CUDA text-only with `--language-model-only`; evidence covers Qwen3.6-35B-A3B native DP4/TP1/EP4 and synchronous AFD 2A1F eager/graph. GDN DBO, Attention-side routing, PP, async and multi-node execution are unsupported; NPU/multimodal are unsupported or unverified, quantization is unverified, and no performance claim is made. |

Architecture aliases and wrapper ownership are recorded in
[model integration](design/module/model_integration.md#model-registration).
For deployment commands, use the [recipe index](../recipe/README.md).

## Tested runtime matrix

This table separates accepted runtime configurations from validation evidence.
Both backends require ModelRunnerV2. A test case or a historical result does not
qualify every model/topology combination on the current target.

| Platform/path | Execution | Ubatching | Routing/quantization limits | Evidence |
| --- | --- | --- | --- | --- |
| CUDA V2 + `P2pNcclAFDConnector` | Eager or `FULL_DECODE_ONLY` native V2 CUDA Graph | Exactly two native microbatches; Attention DP1 or DP > 1 | Model-specific gate placement; PP/CP, elastic EP, EPLB, SP MoE, and compile SP rejected; role ranks equal DP x TP | Source-specific GPU validation is linked below; coverage is limited to the recorded model/topology combinations. DeepSeek-V2-Lite has known DBO shutdown failures. Qwen3.5/3.6 GDN DBO is unsupported. |
| CUDA V2 + `GpuAsyncAFDConnector` (experimental) | Eager or `FULL_DECODE_ONLY`; independent local FFN graphs | Two native Attention microbatches; FFN handles independent work items | Attention-side routing, static EP, FFN TP1, linear expert placement; mapped NVLink or NVSHMEM IBRC RMA with optional GPU extension | Targeted vLLM 0.32.0 H20/DeepSeek-V2-Lite-Chat BF16 2A2F: single-node validation plus two-node transport, eager/graph+DBO model parity and 128-sample GSM8K. Other models, quantization, Attention TP2 and IBGDA are unqualified. [Exact evidence and limits](design/async_gpu_cross_node_research.md) |
| CUDA V2 + `DeepEPM2NAFDConnector` | Eager or `FULL_DECODE_ONLY` native V2 CUDA Graph; SM90 or newer | Exactly two native microbatches, enabled on both roles; Attention DP1 or DP > 1, TP1 or TP2 | Qwen3 MoE (all layers MoE, no shared expert) or DeepSeek-V2; `compute_gate_on_attention=true`; FFN TP1 with EP over all FFN ranks, linear placement, `allgather_reducescatter` all2all and the Triton MoE weight layout; BF16 or block-128 FP8 payload; EPLB, PP/CP and SP MoE rejected | Historical vLLM 0.30 Qwen3-30B-A3B 2A2F GSM8K E2E for eager, graph, graph + DBO, and Attention TP2 eager/graph on H100 and H200, and for eager, graph and graph + DBO on GB200; two-node 2A2F eager, graph and graph + DBO with Attention and FFN on separate nodes. DeepSeek-V2-Lite: [E2E definitions](../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py). [GPU connector device tests](../tests/gpu/README.md) |
| Ascend V2 + `CAMP2pAFDConnector` | Eager, `FULL`, or `FULL_DECODE_ONLY` native V2 ACL Graph | Eager or graph native DBO with two ubatches and Attention DP > 1; plain MLA graph DBO requires `FULL_DECODE_ONLY` | `compute_gate_on_attention=false`; PP/CP, elastic EP and EPLB rejected; model-specific SP policy; role ranks equal DP x TP | Historical 0.30 V2-Lite 2A2F FULL 300-question comparison and 2A1F functional checks; current 0.32 Lite evidence is in the [validation record](https://github.com/vllm-project/afd-plugin/pull/507) |
| Ascend V2 + `CAMAsyncAFDConnector` | Eager Attention; layered FFN eager or FULL | Two native DBO microbatches on Attention, including DP1 and with FFN FULL; FFN stays outside the native DBO schedule | Experimental DeepSeek-V4 W4A8/W8A8 on A3/910C; `async_dp=true`, `compute_gate_on_attention=true`, and `dynamicQuant=1`; PP/CP unsupported; plugin-owned async CAM ops required | Historical DSV4 layered comparisons and limitations are in the [validation record](https://github.com/vllm-project/afd-plugin/pull/507); they predate removal of the legacy path |

The 300-question Ascend evidence in this table is historical. The 0.32
[validation record](https://github.com/vllm-project/afd-plugin/pull/507) separately lists the
eight Lite 100-question cells, W4A8 fixture, and observed shutdown failures.

The Async CAM row supports DeepSeek-V4 with static EP, PP=PCP=DCP=1,
and Attention-local SP. DeepSeek-V2/V2-Lite are no longer supported by
Async CAM; standard DeepSeek-V3.2 was already rejected by model admission.
Finegrained TP, explicit shared-expert DP configuration, and native
MTP and draft-model paths other than DeepSeek-V4 DSpark are excluded. DSpark
keeps Attention/native proposal eager and permits independent layered FFN FULL;
it does not support DBO.
FFN remains connector-driven. The row's upgrade evidence retains its original
scope; MRV2 deployment settings are documented in the
[CAM async guide](npu/CAM_ASYNC_CONNECTOR_USER_GUIDE.md#modelrunnerv2-deployment).

Both backends target vLLM `0.32.0` (`5df63a72`); Ascend additionally targets
vLLM-Ascend `bd42eea0`. Source-specific
H20 model, topology, execution-mode and accuracy results are recorded in
the [GPU upgrade validation matrix](https://github.com/vllm-project/afd-plugin/pull/501)
and [runtime-fix validation matrix](https://github.com/vllm-project/afd-plugin/pull/502).
Those records identify the exact tested revisions. Earlier GPU results on
vLLM 0.30 do not establish vLLM 0.32 hardware qualification.

## Validation limits

- GPU accuracy evidence covers recorded first-300 comparisons and the GPU
  async first-128 GSM8K comparison, not
  full-dataset accuracy, numerical equivalence, or performance improvement.
- DeepSeek-V2-Lite TP2 Graph + DBO and DP2 eager DBO can require force
  killing during shutdown. Inference success does not imply graceful exit.
- Native V4 DBO requires a compatible upstream communication backend. The
  recorded native V4 accuracy reference uses Graph without DBO.
- NPU 0.32 evidence is limited to the configurations and checks in the
  [validation record](https://github.com/vllm-project/afd-plugin/pull/507); the seven AFD Lite
  cases failed graceful exit. Historical NPU results do not qualify 0.32.

Historical [NPU validation](https://github.com/vllm-project/afd-plugin/pull/425#issuecomment-6063910923)
and [DBO repair](https://github.com/vllm-project/afd-plugin/pull/430#issuecomment-6063840311)
records retain their original versions and sample counts. Synchronous DSV4
profiles remain smoke-only, with known incorrect A5 concurrent answers; see
their [E2E limits](../tests/e2e/README.md#dsv4-flash-sync-camp2p-concurrent-requests-local-4-or-16-npus).
GPU/NPU rank topology and connector resource rules remain owned by
[connector contracts](design/module/connector_contracts.md).

The repository does not record a canonical CUDA container or assert an
official tag for the Ascend source pin `bd42eea0`; environment evidence is
not an authoritative package tag.
