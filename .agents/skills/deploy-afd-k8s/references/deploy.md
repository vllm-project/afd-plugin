# 2-4. Compute GPU_COUNT, confirm, and deploy

## 2. GPU_COUNT

Plain single-pod recipe (`PLAN_IS_PLAIN=1` from
[resolve-recipe.md](resolve-recipe.md)): union of every literal
`CUDA_VISIBLE_DEVICES=` in the script -- one pod runs every role block.

```bash
GPU_COUNT="$(grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' "$RECIPE_SCRIPT_PATH" \
  | cut -d= -f2 | tr ',' '\n' | sort -u | wc -l)"
```

Placement-driven recipe: all pods of a role share one StatefulSet
template, so GPU count is per role -- count the devices in the literal
`CUDA_VISIBLE_DEVICES=` inside the recipe's role block (attention or
FFN). Every pod sharing a role shares that literal, so this is a plain
text extraction:

```bash
role_block() {  # $1=script $2=ATTENTION|FFN
  awk -v role="$2" '
    /AFD_ROLE/ && $0 ~ tolower(role) { inblock=1; next }
    inblock && /^fi/ { inblock=0 }
    inblock { print }
  ' "$1"
}
devices="$(role_block "$RECIPE_SCRIPT_PATH" "$ROLE" \
  | grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' | head -1 | cut -d= -f2)"
GPU_COUNT=$(( $(grep -o ',' <<<"$devices" | wc -l) + 1 ))
```

## 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `CLIENT_PORT`, `RECIPE_SCRIPT_PATH`, `DEPLOY_NAME` (default
`vllm`; change it to run a second deployment alongside an existing one --
it prefixes every resource name and labels every object it creates
`afd-deployment: <DEPLOY_NAME>`, so report/teardown/cleanup can select the
whole deployment with `-l afd-deployment=<DEPLOY_NAME>`), and the plan as a
table of `role | replicas | GPU_COUNT`
(read straight from the recipe's header -- no shape to negotiate, just
restate what was parsed). Flag that step 4 **deletes every existing
StatefulSet** (and its pods) carrying this deployment's label -- including
ones from a previous plan with different names (e.g. a leftover
single-pod `${DEPLOY_NAME}-pod` after switching to a split-role plan).

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
CLIENT_PORT=<n>   # from the script's --port; default 18305
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
DEPLOY_NAME=${DEPLOY_NAME:-vllm}
# An unset context namespace resolves to empty here, which would silently
# corrupt head DNS (<pod>.<svc>..svc.cluster.local) and the fsGroup lookup
# below -- default it, and pass -n "$NAMESPACE" on every kubectl call in
# this skill rather than relying on the context.
NAMESPACE="$(kubectl config view --minify -o jsonpath='{..namespace}')"
NAMESPACE="${NAMESPACE:-default}"
# The cluster DNS domain is not always cluster.local -- read it off the
# CoreDNS config, falling back to the default. Override manually
# (CLUSTER_DOMAIN=...) if the cluster runs a different DNS stack.
CLUSTER_DOMAIN="$(kubectl -n kube-system get configmap coredns \
  -o jsonpath='{.data.Corefile}' 2>/dev/null \
  | awk '/^\s*kubernetes\s/ {print $2; exit}')"
CLUSTER_DOMAIN="${CLUSTER_DOMAIN:-cluster.local}"
```

**Node placement is left to the scheduler by default.** Only add
`podAntiAffinity` forcing Pods onto separate nodes if the caller explicitly
asks to exercise cross-node placement -- add this to every StatefulSet's
pod-template `spec` in `templates/statefulset.yaml` before applying. No
extra label is needed: pods already carry `afd-deployment:
${TEMPLATE_DEPLOY_NAME}` (envsubst fills it in), so keying the rule on it
repels only this deployment's pods -- a second deployment in the namespace
stays free to share the node:

```yaml
  spec:
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - {key: afd-deployment, operator: In, values: ["${TEMPLATE_DEPLOY_NAME}"]}
            topologyKey: kubernetes.io/hostname
```

## 4a. Model PVC

Size for the model being served. An existing undersized PVC is reused
silently and fails mid-download, so set `MODEL_PVC_SIZE` deliberately.
`PVC_ACCESS_MODE` is derived, not asked: `ReadWriteMany` whenever the plan
has more than one pod -- an RWO volume can only attach from one node, and
the second Pod deadlocks `Pending` with `Multi-Attach error` otherwise.

```bash
MODEL_PVC_SIZE=${MODEL_PVC_SIZE:-100Gi}
STORAGE_CLASS=${STORAGE_CLASS:-}
PVC_ACCESS_MODE=ReadWriteOnce   # ReadWriteMany if the plan has >1 pod

if kubectl -n "$NAMESPACE" get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  echo "PVC ${PVC_NAME} exists; reusing its warm HF_HOME cache for ${MODEL_ID}"
  echo "NOTE: verify its size fits ${MODEL_ID}, and its accessModes include"
  echo "      ReadWriteMany if this plan has >1 pod -- an RWO reuse deadlocks the 2nd pod"
else
  envsubst '${PVC_NAME} ${MODEL_PVC_SIZE} ${STORAGE_CLASS} ${PVC_ACCESS_MODE}' \
    < templates/pvc.yaml | grep -v 'storageClassName: *$' \
    | kubectl apply -n "$NAMESPACE" -f -
fi
```

If an existing single-pod RWO PVC already holds downloaded weights you want
to reuse for a multi-pod plan, provision a new RWX PVC and `cp -a` the data
across (via a short-lived Pod pinned to the node that holds the RWO mount)
rather than re-downloading.

## 4b. Recipe ConfigMap

Lets the pod run any local recipe -- edited, uncommitted, or new -- without
rebuilding the image. Run from wherever `RECIPE_SCRIPT_PATH` resolves:

```bash
kubectl -n "$NAMESPACE" create configmap "${DEPLOY_NAME}-recipe-script" \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -n "$NAMESPACE" -f -
# Label it like every other object of the deployment (create configmap has
# no --labels flag), so teardown/cleanup can select it by label.
kubectl -n "$NAMESPACE" label configmap "${DEPLOY_NAME}-recipe-script" \
  "afd-deployment=$DEPLOY_NAME" --overwrite
```

## 4c. fsGroup

Must be inside the namespace's allowed range or the pod is rejected at
admission. Look it up once, reuse for every pod:

```bash
# Note the jsonpath escaping: dots in the annotation key are escaped with a
# backslash, but the slashes are NOT -- escaping them too silently yields an
# empty string and falls back to 1000, which an OpenShift SCC then rejects
# at admission (`1000 is not an allowed group`).
FS_GROUP="$(kubectl -n "$NAMESPACE" get ns "$NAMESPACE" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
FS_GROUP="${FS_GROUP:-1000}"
```

## 4d. Services

Create every Service that applies -- see [services.md](services.md) for
which, and why this must happen **before** deploying any StatefulSet in 4e
below. The governing Service is always created; the rest are ExternalName
pointers built from one shared template. Gate each optional endpoint in
code on the plan (never create them unconditionally -- a single-pod deploy
would get ExternalName Services pointing at pods that do not exist):

```bash
governing_svc="${DEPLOY_NAME}-recipe-sts"
head_dns() { echo "$1.${governing_svc}.${NAMESPACE}.svc.${CLUSTER_DOMAIN}"; }
endpoint() { # $1=service name, $2=head pod name
  TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" TEMPLATE_SVC_NAME="$1" \
  TEMPLATE_ENDPOINT="$(head_dns "$2")" \
  envsubst '${TEMPLATE_DEPLOY_NAME} ${TEMPLATE_SVC_NAME} ${TEMPLATE_ENDPOINT}' \
    < templates/service-endpoint.yaml | kubectl apply -n "$NAMESPACE" -f -
}

# Governing (identity) Service -- always.
TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" envsubst '${TEMPLATE_DEPLOY_NAME}' \
  < templates/service-governing.yaml | kubectl apply -n "$NAMESPACE" -f -

# Client-facing endpoint -- always; targets the attention head pod
# (or the sole pod of a plain recipe).
if [ "$PLAN_IS_PLAIN" = "1" ]; then
  endpoint "${DEPLOY_NAME}-service" "${DEPLOY_NAME}-pod-0"
else
  endpoint "${DEPLOY_NAME}-service" "${DEPLOY_NAME}-pod-attention-0"
fi

# Rendezvous endpoint -- only when the plan spans more than one pod;
# targets the FFN head pod, which owns the TCPStore.
[ "${TOTAL_PODS:-1}" -gt 1 ] \
  && endpoint "${DEPLOY_NAME}-ffn-p2p-service" "${DEPLOY_NAME}-pod-ffn-0"

# DP-coordination endpoint -- only when the attention role itself is
# split across >1 pod; targets the attention head pod. (No FFN counterpart:
# nothing consumes an FFN_DP_ADDRESS today -- no env, no recipe. Add one
# back, gated the same way, if a recipe ever needs it.)
[ "${ATTN_SPLIT:-0}" = "1" ] \
  && endpoint "${DEPLOY_NAME}-attn-dp-service" "${DEPLOY_NAME}-pod-attention-0"
```

`TOTAL_PODS` (>1 when the recipe spans pods) and `ATTN_SPLIT` (1 when the
attention role spans >1 pod) come from the plan parsed in
[resolve-recipe.md](resolve-recipe.md), never asked.

## 4e. Deploy StatefulSet(s)

One StatefulSet per role the plan uses (a plain recipe deploys a single
both-roles StatefulSet), with `replicas` = that role's pod count. Pod
identity is self-derived: each pod reads its role from `AFD_ROLE` and its
zero-based index from its hostname (`<statefulset>-<ordinal>`), so no
per-pod env is injected at all:

```bash
deploy_role_sts() {
  local role="$1" replicas="$2" gpu="$3"
  local sts_name="${DEPLOY_NAME}-pod"
  [ "$role" != "both" ] && sts_name="${DEPLOY_NAME}-pod-${role}"

  TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE" TEMPLATE_MODEL="$MODEL_ID" TEMPLATE_POD="$sts_name" \
  TEMPLATE_GPU="$gpu" TEMPLATE_PVC="$PVC_NAME" TEMPLATE_FSGROUP="$FS_GROUP" \
  TEMPLATE_CLIENT_PORT="$CLIENT_PORT" TEMPLATE_REPLICAS="$replicas" TEMPLATE_ROLE="$role" \
  TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" \
  envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_POD} ${TEMPLATE_GPU} ${TEMPLATE_PVC} ${TEMPLATE_FSGROUP} ${TEMPLATE_CLIENT_PORT} ${TEMPLATE_REPLICAS} ${TEMPLATE_ROLE} ${TEMPLATE_DEPLOY_NAME}' \
    < templates/statefulset.yaml | kubectl apply -n "$NAMESPACE" -f -
}

# Pre-deploy cleanup, by label and once for the whole deployment -- not per
# StatefulSet name inside deploy_role_sts: a previous deploy of this
# DEPLOY_NAME with a different plan leaves differently-named StatefulSets
# behind (e.g. single-pod `${DEPLOY_NAME}-pod` after switching to a
# split-role plan), still holding GPUs. The label selects exactly this
# deployment's objects, never a second deployment's.
kubectl -n "$NAMESPACE" delete sts -l "afd-deployment=$DEPLOY_NAME" \
  --ignore-not-found
```

Plain recipe, one both-roles StatefulSet:

```bash
deploy_role_sts both 1 "$GPU_COUNT"
```

Placement-driven recipe, e.g. the qwen3.5 3-pod plan (2 attention pods,
1 FFN pod; deploy FFN first so it owns the rendezvous):

```bash
deploy_role_sts ffn 1 2
deploy_role_sts attention 2 2
```

A re-applied ConfigMap (step 4b) or StatefulSet template does **not**
restart running pods (`updateStrategy: OnDelete` in the template). AFD
startup is a collective across every pod of the deployment, so pods must
restart together -- after changing the recipe or the template, either
re-run the `kubectl delete sts` + apply above, or restart every pod at
once:

```bash
kubectl delete pod -l afd-deployment="$DEPLOY_NAME"
```

Never restart one role or one shard on its own: a restarted shard cannot
rejoin the surviving pods' DP world and the deployment deadlocks (the
split-role gap documented in SKILL.md), and a restarted role alone leaves
the other role running the old recipe/config. A StatefulSet's default
`RollingUpdate` has exactly this problem -- one pod at a time, highest
ordinal first, waiting for Ready -- which is why the template pins
`OnDelete`.

## 4f. Wait for readiness

One wait per pod (not `kubectl rollout status` -- that only works with
`RollingUpdate`, and the template pins `updateStrategy: OnDelete`); the
readinessProbe only passes once the pod's own engine is up, so this replaces
log-marker polling (budget 60 minutes -- model download and graph capture
dominate):

```bash
for pod in <every pod of the plan, e.g. vllm-pod-ffn-0 vllm-pod-attention-0 vllm-pod-attention-1>; do
  kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$pod" --timeout=60m \
    || { kubectl -n "$NAMESPACE" describe "pod/$pod" | tail -40; \
         kubectl -n "$NAMESPACE" logs "pod/$pod" --tail=100; exit 1; }
done
```

Container restarts during multi-pod startup are **expected, not failures**:
a pod whose peer's rendezvous is not up yet fails fast and crash-loops
until the peer (deployed first) is ready, then recovers on a later retry --
do not abort the wait on restart counts or `ERROR:` lines alone. Only
report the endpoint ready once every StatefulSet has rolled out; diagnose
individual pods with `kubectl logs pod/<sts_name>-<ordinal>` (add
`--previous` for the incarnation before a restart).

Startup markers are keyed on the fixed log names every colocation recipe
uses today -- see `templates/statefulset.yaml`'s `ready_marker()` for the
exact lines. A recipe using different log names needs that template
updated.
