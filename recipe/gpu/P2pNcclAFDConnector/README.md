# P2pNcclAFDConnector Recipes

Launch scripts for running Attention-FFN Disaggregation (AFD) with
`P2pNcclAFDConnector`, a GPU point-to-point connector built on vLLM's
`PyNcclCommunicator`. See
[`docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`](../../../docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md)
for the connector's rank mapping and configuration contract.

## Directory layout

```text
.
├── deepseek_v2_lite/
│   ├── README.md
│   ├── prefill_decode_colocation/       # 2A2F, DP/TP variants, eager/graph
│   └── prefill_decode_disaggregation/   # 2P1A1F
└── qwen3_5_122b_a10b_fp8/
    └── prefill_decode_colocation/       # multipod_2a_2a_2f_graph.sh: 4A2F split across 2 attention pods, graph
```

Each model directory holds its own recipe scripts; open its `README.md`
(where present) before running anything in it -- it documents prerequisites,
topology, ports, and known limitations for that model.

Recipes under `prefill_decode_colocation/` background an attention worker and
an FFN worker (no prefill split). Recipes under `prefill_decode_disaggregation/`
add a separate prefill stage and a proxy in front of it -- see the caveat in
"Deploying on Kubernetes" below.

## Running on a local host

Run any script directly from the repository root, e.g.:

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh
```

Each script backgrounds its workers and writes per-worker logs (`attn.log`,
`ffn.log`, and for disaggregation `afd_prefill*.log`) into the current
directory. Wait for `Application startup complete` in each log before
sending traffic. See the model README for exact prerequisites (GPU count,
weights, ports) and the benchmark command for that model.

## Deploying on Kubernetes

Use the **`deploy-afd-k8s`** skill (`.agents/skills/deploy-afd-k8s/SKILL.md`)
to stand up a recipe from this folder on a Kubernetes/OpenShift cluster --
see that skill for the full contract (PVC/ConfigMap/pod/Service lifecycle,
scope, and requirements).

The skill requires the recipe path (e.g.
`recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp1tp2.sh`)
plus the image, model id, and PVC name it needs. Placement is never asked
for: a recipe without a top-of-file placement comment deploys as a single
pod; a recipe whose header describes a per-pod placement (e.g.
`qwen3_5_122b_a10b_fp8/prefill_decode_colocation/multipod_2a_2a_2f_graph.sh`,
`deepseek_v2_lite/prefill_decode_colocation/multipod_2a_2f_graph_dbo_dp1tp2.sh`)
deploys exactly that many pods with exactly that per-pod setup -- see the
skill's `resolve-recipe.md` for the header convention.
`prefill_decode_disaggregation` recipes are out of scope for the skill; run
those locally instead.

Pods and Services are left running after deployment so weights stay warm for
follow-up runs; see the skill's teardown step to tear them down explicitly.
