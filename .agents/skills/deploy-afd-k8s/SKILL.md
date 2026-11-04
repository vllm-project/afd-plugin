---
name: deploy-afd-k8s
description: Use when the user asks to deploy, serve, or stand up an AFD GPU recipe on Kubernetes/OpenShift from a recipe .sh script - creating the model PVC, recipe ConfigMap, serving StatefulSet(s), and Service(s), and waiting until the endpoint answers. Placement across pods is never asked for or assumed -- it is read directly off the recipe's own top-of-file comment block (however that block describes it, strict or prose): a recipe whose comment describes a per-pod placement deploys exactly that many pods with exactly that per-pod setup; a recipe without one deploys as a single pod. Do not use for local (non-k8s) serving, NPU recipes, prefill-decode disaggregation recipes, or E2E correctness testing (see run-e2e).
---

# Deploy an AFD GPU recipe on Kubernetes

Stand up serving StatefulSet(s) -- one per role, whose pods run a local AFD
recipe script -- on a
Kubernetes/OpenShift cluster and leave a reachable OpenAI-compatible
endpoint behind.

## Contract

On success this skill guarantees, in the target namespace:

| | |
|---|---|
| Endpoint | `http://${DEPLOY_NAME}-service:${CLIENT_PORT}` (OpenAI-compatible, in-namespace; `DEPLOY_NAME` defaults to `vllm`, `CLIENT_PORT` to `18305`) |
| Serving | the model named by `MODEL_ID` |
| StatefulSet(s) / Service | one StatefulSet per role (`${DEPLOY_NAME}-pod-<role>`, `replicas` = that role's pod count, pods `<name>-<ordinal>`); plain recipes use one both-roles StatefulSet `${DEPLOY_NAME}-pod`; client Service `${DEPLOY_NAME}-service` (an ExternalName at the attention head pod). Every object the deployment creates (StatefulSets, pods, Services, ConfigMap) is labeled `afd-deployment=<DEPLOY_NAME>` so the whole deployment selects/reports/tears down with `-l afd-deployment=<DEPLOY_NAME>`; pods and StatefulSets also carry `app=afd-recipe,role=serve`, other objects `app=afd-recipe`. The model PVC is deliberately not labeled (shared, user-named state). |
| Model cache | `PVC_NAME`, mounted at `/models`, `HF_HOME=/models/.hf_home` |
| Left running | yes, deliberately -- so weights stay warm for follow-up runs |

Recovery scope: a crashed pod's container restarts and a deleted pod is
recreated automatically, and a role-per-pod deployment (each role on its
own pod) recovers from a peer-pod crash with no operator action. A
**split-role** deployment (one role across multiple pods, e.g. a headless
attention shard) does **not** recover automatically from the loss of its
head pod: an idle surviving shard never detects the peer loss and never
rejoins the restarted peer's rendezvous. Remediation is restarting the
deployment's pods together (`kubectl delete pod -l afd-deployment=<name>`;
the StatefulSets recreate them) -- never one role or one shard on its own,
for the same collective-startup reason that the StatefulSets pin
`updateStrategy: OnDelete`. Root cause is connector-level peer-loss
detection, not this skill.

Whenever a recipe's header spans more than one pod, up to two more
internal-only Services may also exist:
`${DEPLOY_NAME}-ffn-p2p-service` (rendezvous -- always, when the plan has
more than one pod) and `${DEPLOY_NAME}-attn-dp-service` (only when the
attention role is itself split across more than one pod; there is
deliberately no FFN counterpart -- see [services.md](services.md)). All but the
governing Service are ExternalName pointers at head pods -- callers never talk to these directly. See
[services.md](references/services.md).

## Scope

- Recipes under `recipe/gpu/P2pNcclAFDConnector/**` (GPU only, not NPU),
  **`prefill_decode_colocation` only** (attention + FFN workers, no prefill
  split).
  `prefill_decode_disaggregation` recipes are **out of scope**: they need a
  NIXL-enabled image plus a proxy, and resolve a `SCRIPT_DIR`-relative path
  that this skill's flat script mount (step 4b) breaks. Ask the user for a
  different workflow for disaggregation.
- Recipes are hand-written shell scripts, not JSON configs -- there is no
  generator step and no manifest to read.
- A recipe is **placement-driven** (more than one pod) exactly when its
  top-of-file comment block describes a per-pod placement -- read that
  block, don't infer shape from flags. Today's recipes spell it as one
  `# For pod N: POD=...` line per pod, but the comment isn't guaranteed to
  use that exact grammar going forward. See
  [resolve-recipe.md](references/resolve-recipe.md).
- Not for `tools/benchmarks/decode_bench_server.sh` (local, non-k8s).

## Requirements

- A live, authenticated `kubectl`/`oc` session able to create/delete
  StatefulSets, Pods, Services, ConfigMaps, and PVCs in the target namespace.
- `envsubst` (from `gettext`) locally.
- An image, pushed where the cluster can pull it. Ask the user whether to
  use an image they already have, or to build one from `docker/Dockerfile.ci`
  and push it to a registry they provide:

  ```bash
  IMAGE=<registry>/<repo>:<tag>
  docker build -t "$IMAGE" -f docker/Dockerfile.ci .
  docker push "$IMAGE"
  ```

  The image supplies only the `afd-plugin` install. The recipe script is
  mounted fresh from local disk on every run (step 4b), so editing or adding
  a recipe never requires a rebuild.
- An `hf-token-secret` Secret with a `token` key, needed when `MODEL_ID` is
  a HF Hub repo id (not needed when weights are already staged on the PVC):
  ```bash
  kubectl create secret generic hf-token-secret --from-literal=token=<hf_token>
  ```
- Nodes with enough free GPUs for `GPU_COUNT` (per pod).

## Workflow

Load these in order; each covers one phase and nothing else. Services must
exist **before** any Pod is deployed -- see
[services.md](references/services.md) for why:

| Reference | Covers |
|---|---|
| [resolve-recipe.md](references/resolve-recipe.md) | Step 1: read the recipe's serve-block fields and model id; read its placement (or lack of one) from its header comment -- never ask. |
| [deploy.md](references/deploy.md) | Steps 2-4f: GPU_COUNT, confirm, PVC, ConfigMap, fsGroup, Services (step 4d -- which of the four, see [services.md](references/services.md)), StatefulSet(s) via `templates/statefulset.yaml`, wait with `kubectl rollout status`. |
| [report-and-teardown.md](references/report-and-teardown.md) | Step 5: report the endpoint/model/nodes. Step 6: teardown (pods/Services only -- never the PVC unless asked). |

All k8s manifests live in `templates/*.yaml`, applied via `envsubst | kubectl
apply -f -`; no manifest is inlined in these docs.
