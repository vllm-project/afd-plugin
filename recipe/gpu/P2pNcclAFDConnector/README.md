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
│   ├── prefill_decode_colocation/       # dbo.sh, eager_dbo.sh (any LAYOUT, ATTN_TP, FFN_TP)
│   └── prefill_decode_disaggregation/   # 2P1A1F
├── qwen3_30b_a3b/prefill_decode_colocation/          # graph.sh, default 2A2F
├── qwen3_235b_a22b_fp8/prefill_decode_colocation/    # graph.sh, default 4A2F
└── qwen3_5_122b_a10b_fp8/prefill_decode_colocation/  # graph.sh, default 2A0F,2A0F,0A2F (3 pods)
```

Each model directory holds its own recipe scripts; open its `README.md`
(where present) before running anything in it.

The `prefill_decode_colocation/` recipes run an Attention server and an FFN
server (no prefill split) through the shared
[colocation launcher](../common/README.md), which documents the layout,
settings, ports, and how to run a recipe directly or in a pod. Recipes under
`prefill_decode_disaggregation/` add a separate prefill stage and a proxy in
front of it -- see the caveat in "Deploying on Kubernetes" below.

## Running on a local host

```bash
export MODEL_PATH=/path/model_weights/DeepSeek-V2-Lite
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/dbo.sh
```

A colocation recipe writes `attn.log` and `ffn.log` into the current
directory; send traffic to port `18000` once `attn.log` prints
`Application startup complete`.

## Deploying on Kubernetes

Use the **`deploy-afd-k8s`** skill (`.agents/skills/deploy-afd-k8s/SKILL.md`)
to stand up a recipe from this folder on a Kubernetes/OpenShift cluster --
see that skill for the full contract (PVC/ConfigMap/pod/Service lifecycle,
scope, and requirements).

The skill takes the recipe path (e.g.
`recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/dbo.sh`)
plus the image, model id, and PVC name, and deploys one pod per `LAYOUT`
entry (for example `LAYOUT=2A0F,0A2F ATTN_TP=2 FFN_TP=2` on the DeepSeek
recipe, or the Qwen3.5 recipe's default three pods).
`prefill_decode_disaggregation` recipes are out of scope for the skill; run
those locally instead.

Pods and Services are left running after deployment so weights stay warm for
follow-up runs; see the skill's teardown step to tear them down explicitly.
