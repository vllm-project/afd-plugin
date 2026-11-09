# 2-4. Confirm and deploy

## 2. GPU_COUNT

Use the `GPU_COUNT` recorded by [resolve-recipe.md](resolve-recipe.md) --
one value for a plain or both-roles (`POD=AFD_<index>`) recipe
(`GPU_COUNT`), one per role for a split-role plan (`ATTN_GPU_COUNT`,
`FFN_GPU_COUNT`; every pod of a role shares its StatefulSet template, so it
also shares the GPU count). Don't re-read the script here; all recipe
parsing lives in step 1.

## 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `CLIENT_PORT`, `RECIPE_SCRIPT_PATH`, `DEPLOY_NAME` (default
`vllm`; change it to run a second deployment alongside an existing one --
it prefixes every resource name and labels every object it creates
`afd-deployment: <DEPLOY_NAME>`, so report/teardown/cleanup can select the
whole deployment with `-l afd-deployment=<DEPLOY_NAME>`), and the plan as a
table of `role | replicas | GPU_COUNT`
(replicas straight from the recipe's header comment -- one `both` row for a
both-roles plan, one row per role for a split-role plan; GPU counts from
resolve-recipe.md -- no shape to negotiate, just restate what was parsed).
Flag that step 4 **deletes every existing
StatefulSet** (and its pods) carrying this deployment's label -- including
ones from a previous plan with different names (e.g. a leftover
single-pod `${DEPLOY_NAME}-pod` after switching to a split-role plan) --
and that any deployment left behind by the earlier Pod-based version of
this skill is deleted too, under a separate confirmation (steps 3b-3c).

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
CLIENT_PORT=<n>   # from the script's --port; default 18305
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
DEPLOY_NAME=${DEPLOY_NAME:-vllm}
# GPU counts recorded in step 1: GPU_COUNT for a plain or both-roles
# (POD=AFD_<index>) recipe, or one per role for a split-role plan. Plan
# shape, also from step 1: BOTH_ROLES, TOTAL_PODS, ATTN_SPLIT, FFN_SPLIT.
GPU_COUNT=<n>          # plain or both-roles recipe only
ATTN_GPU_COUNT=<n>     # split-role recipe only
FFN_GPU_COUNT=<n>      # split-role recipe only
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

## 3b. Detect a legacy Pod-based deployment

The earlier version of this skill deployed **bare Pods** and
**selector-based Services** with fixed `vllm-*` names and no `afd-deployment`
label (whatever `DEPLOY_NAME` you pick now, that is where they are). Nothing
in steps 4d-4e removes them, and they conflict with the new deployment on
every axis:

- **Same pod names for a split role.** A legacy pod is
  `vllm-pod-attention-0`; that is also the exact name the new
  `vllm-pod-attention` StatefulSet creates for ordinal 0. The StatefulSet
  cannot create a pod over the existing one and stalls at that ordinal.
- **Same GPUs and, for a plain recipe, the same `ReadWriteOnce` model
  volume.** A still-running legacy pod keeps them, so the new pod stays
  `Pending` (insufficient GPUs, or `Multi-Attach error` on the volume).
- **Same Service names, different `spec.type`.** `vllm-service` and
  `vllm-ffn-p2p-service` / `vllm-attn-dp-service` used to be
  selector-based ClusterIP (headless for the rendezvous/DP ones); the new
  ones are ExternalName. Kubernetes refuses a service type change to or
  from ExternalName, so step 4d cannot even apply -- and a surviving
  selector-based `vllm-service` keeps routing the endpoint at the legacy
  head pod instead of the new one.

Detect them by the one label they do **not** carry (read-only, safe to run
any time):

```bash
# Legacy bare Pods: app + role labels, but no afd-deployment label.
kubectl -n "$NAMESPACE" get pods -l 'app=afd-recipe,role=serve,!afd-deployment' -o wide
# Legacy selector-based Services: labeled app=afd-recipe, unlabeled
# afd-deployment (the new governing/ExternalName Services carry it).
kubectl -n "$NAMESPACE" get svc -l 'app=afd-recipe,!afd-deployment' -o wide
# Legacy fixed-name recipe ConfigMap (the new one is
# ${DEPLOY_NAME}-recipe-script, which is not affected).
kubectl -n "$NAMESPACE" get configmap afd-recipe-script
```

`-o wide` shows each legacy pod's node and phase: the phase tells you
whether it still holds GPUs, the node tells you where an `RWO` volume is
currently attached. These objects are also exactly the ones the step 6
teardown misses, because it selects on `afd-deployment`.

## 3c. Remove the legacy deployment -- only with consent

A legacy deployment may be serving right now: its pods run with
`restartPolicy: Never` and no controller, so a deleted pod is gone for good
and nothing recreates it. If 3b found anything, **stop and ask the caller
before deleting**, stating the objects and that the endpoint stays down from
the first delete until the new pods report `Ready` in step 4f. The two
deployments cannot overlap -- same GPUs, same RWO volume for a plain recipe,
same pod and Service names -- so there is no "keep both running" option.

Only after explicit confirmation, delete exactly what 3b selected, by the
same label selectors (never `app=afd-recipe` alone, never a list of names
that could drift), and **before step 4d** so the endpoint is switched away
from the legacy head pod only once that pod is gone:

```bash
kubectl -n "$NAMESPACE" delete pod -l 'app=afd-recipe,role=serve,!afd-deployment'
kubectl -n "$NAMESPACE" delete svc -l 'app=afd-recipe,!afd-deployment'
kubectl -n "$NAMESPACE" delete configmap afd-recipe-script --ignore-not-found
# The new pods cannot start while the legacy ones hold the GPUs and the
# model volume -- wait until the pods are really gone (graceful termination
# of a serving pod takes a while), not just until the delete returns.
# Polled rather than `kubectl wait --for=delete`, which fails when the
# selector matches nothing -- and "already gone" is a success here.
legacy_pods="$(kubectl -n "$NAMESPACE" get pod \
  -l 'app=afd-recipe,role=serve,!afd-deployment' -o name)"
for _ in $(seq 1 120); do
  [ -z "$legacy_pods" ] && break
  sleep 5
  legacy_pods="$(kubectl -n "$NAMESPACE" get pod \
    -l 'app=afd-recipe,role=serve,!afd-deployment' -o name)"
done
[ -z "$legacy_pods" ] || echo "still terminating after 10m: $legacy_pods"
```

If the caller declines, stop: report the blocking objects from 3b and do not
deploy on top of them -- the new StatefulSets would stall exactly as
described there. Never delete the model PVC in this step: after the legacy
pods are gone a new pod landing on a different node can still show a
transient `Multi-Attach error` until the old node's kubelet detaches the
`RWO` volume, and that clears on its own.

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

# Head pod of each role. A both-roles plan (a plain recipe or a
# `POD=AFD_<index>` one) has a single ${DEPLOY_NAME}-pod StatefulSet, so
# every role's head is its ordinal 0 -- the same pod serves the API server,
# owns the rendezvous TCPStore, and leads both DP worlds. A split-role plan
# has one StatefulSet per role, so each head is that StatefulSet's ordinal 0.
if [ "${BOTH_ROLES:-1}" = "1" ]; then
  attn_head="${DEPLOY_NAME}-pod-0"
  ffn_head="${DEPLOY_NAME}-pod-0"
else
  attn_head="${DEPLOY_NAME}-pod-attention-0"
  ffn_head="${DEPLOY_NAME}-pod-ffn-0"
fi

# Governing (identity) Service -- always.
TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" envsubst '${TEMPLATE_DEPLOY_NAME}' \
  < templates/service-governing.yaml | kubectl apply -n "$NAMESPACE" -f -

# Client-facing endpoint -- always; targets the attention head pod (the sole
# pod of a plain recipe, the API-server pod of every other plan).
endpoint "${DEPLOY_NAME}-service" "$attn_head"

# Rendezvous endpoint -- only when the plan spans more than one pod;
# targets the FFN head pod, which owns the TCPStore.
[ "${TOTAL_PODS:-1}" -gt 1 ] \
  && endpoint "${DEPLOY_NAME}-ffn-p2p-service" "$ffn_head"

# DP-coordination endpoints -- only for a role that is itself split across
# >1 pod. The FFN one exists for the `POD=AFD_<index>` recipes, which point
# their FFN DP-RPC peers at ${DEPLOY_NAME}-ffn-dp-service; no split-role
# recipe reads an FFN DP address today, and the template injects no
# FFN_DP_ADDRESS env for it.
[ "${ATTN_SPLIT:-0}" = "1" ] \
  && endpoint "${DEPLOY_NAME}-attn-dp-service" "$attn_head"
[ "${FFN_SPLIT:-0}" = "1" ] \
  && endpoint "${DEPLOY_NAME}-ffn-dp-service" "$ffn_head"
```

`BOTH_ROLES` (1 when every pod runs both roles), `TOTAL_PODS` (>1 when the
recipe spans pods), and `ATTN_SPLIT` / `FFN_SPLIT` (1 when that role spans
>1 pod) come from the plan parsed in
[resolve-recipe.md](resolve-recipe.md), never asked.

## 4e. Deploy StatefulSet(s)

One StatefulSet per role the plan uses -- a plain recipe and a
`POD=AFD_<index>` plan both deploy the single both-roles StatefulSet, a
split-role plan one StatefulSet per role -- with `replicas` = that role's
pod count. Pod identity is self-derived: each pod reads its role from
`AFD_ROLE` and its zero-based index from its hostname
(`<statefulset>-<ordinal>`), so no per-pod env is injected at all:

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
# deployment's objects, never a second deployment's. Legacy bare Pods have
# no afd-deployment label and are not selected here -- step 3c removed them
# under its own confirmation.
kubectl -n "$NAMESPACE" delete sts -l "afd-deployment=$DEPLOY_NAME" \
  --ignore-not-found
```

Plain recipe, one both-roles StatefulSet:

```bash
deploy_role_sts both 1 "$GPU_COUNT"
```

Split-role recipe, e.g. the qwen3.5 3-pod plan (2 attention pods, 1 FFN
pod; deploy FFN first so it owns the rendezvous):

```bash
deploy_role_sts ffn 1 "$FFN_GPU_COUNT"
deploy_role_sts attention 2 "$ATTN_GPU_COUNT"
```

Both-roles multi-pod plan, e.g. the DeepEP M2N `POD=AFD_<index>` 2 x 4A1F
plan: still one StatefulSet, scaled with `replicas`. Each pod derives its
own role and ordinal, and those recipes read the ordinal as `${POD#AFD_}`
-- so no per-pod env is injected here either:

```bash
deploy_role_sts both 2 "$GPU_COUNT"
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

## 4f. Wait for the pods to exist, then for Running and Ready

One wait per pod, not `kubectl rollout status` -- that only works with
`RollingUpdate`, and the template pins `updateStrategy: OnDelete`.

Bounded 15-minute wait per pod for `phase=Running` (unschedulable pods never
reach `Failed`, so an unbounded wait would spin forever), then a 60-minute
wait per pod for its `Ready` condition (model download and graph capture
dominate).

Both waits poll the API with `-o jsonpath` instead of using `kubectl wait`:
`--for=jsonpath=...` needs kubectl >= 1.31 and this skill also runs under
much older clients (v1.18 in the current CI image, which reports
`error: unrecognized condition`), and the `Ready` loop has to distinguish
`phase=Failed` and dump logs anyway. `--for=condition=Ready` is portable
but cannot tell a dead pod from a cold one.

`Ready` is the single readiness signal, and the template's `readinessProbe`
drives it: the pod turns Ready only once its own engine is up (`/work/ready`
written after every log marker and, on the attention head, a `/health`
answer), and the same condition is what puts the attention head behind
`${DEPLOY_NAME}-service`. When the in-pod wrapper gives up it prints
`ERROR:` with log tails and exits PID 1, so the container restarts and the
pod never reports `Ready` by accident.

Timeouts are nested on purpose: the in-pod startup deadline
(`STARTUP_TIMEOUT_S` in `templates/statefulset.yaml`, 55 minutes, one budget
shared by every log marker and `/health`) is shorter than the 60-minute wait
here, so a slow pod always fails with its own `ERROR:` reason instead of an
outer timeout. Change the two together.

Applying a StatefulSet only creates the StatefulSet: its controller creates
the pods asynchronously, so between step 4e and the pods existing there is a
window -- seconds normally, and the plain single-pod recipe has nothing
else to wait on in between. `kubectl wait` does **not** wait for creation:
on a pod that does not exist yet it fails with `NotFound` immediately, and
`--timeout` does not apply. So confirm existence first, bounded at 5
minutes, and treat "never created" as a StatefulSet problem (a stalled
ordinal, an unschedulable template, a rejected update) rather than a slow
pod -- the block prints the StatefulSet and its events. `--for=create` would
be the one-liner, but it is not available in every kubectl this skill runs
under.

```bash
pods="<every pod of the plan, e.g. vllm-pod-ffn-0 vllm-pod-attention-0 vllm-pod-attention-1>"

# The StatefulSet controller creates the pods after the apply returns, so
# wait for each pod to exist before asking `kubectl wait` about it.
wait_for_pod() {
  local pod=$1
  local deadline=$(( $(date +%s) + 300 ))
  until kubectl -n "$NAMESPACE" get pod "$pod" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "pod $pod was never created after 5m"
      kubectl -n "$NAMESPACE" get sts -l "afd-deployment=$DEPLOY_NAME" -o wide
      kubectl -n "$NAMESPACE" describe sts "${pod%-*}" | tail -40
      exit 1
    fi
    sleep 5
  done
}

for pod in $pods; do
  wait_for_pod "$pod"
  deadline=$(( $(date +%s) + 900 ))
  while true; do
    phase="$(kubectl -n "$NAMESPACE" get pod "$pod" \
      -o jsonpath='{.status.phase}' 2>/dev/null || true)"
    [ "$phase" = "Running" ] && break
    if [ "$phase" = "Failed" ] || [ "$(date +%s)" -ge "$deadline" ]; then
      echo "pod $pod did not reach Running (phase=${phase:-unknown})"
      kubectl -n "$NAMESPACE" logs "pod/$pod" --tail=200
      kubectl -n "$NAMESPACE" describe "pod/$pod" | tail -40
      exit 1
    fi
    sleep 5
  done
done

for pod in $pods; do
  deadline=$(( $(date +%s) + 3600 ))
  while true; do
    read -r phase ready < <(kubectl -n "$NAMESPACE" get pod "$pod" \
      -o jsonpath='{.status.phase} {.status.conditions[?(@.type=="Ready")].status}' \
      2>/dev/null || true)
    [ "$ready" = "True" ] && break
    if [ "$phase" = "Failed" ] || [ "$(date +%s)" -ge "$deadline" ]; then
      echo "pod $pod not Ready (phase=${phase:-unknown})"
      kubectl -n "$NAMESPACE" logs "$pod" --tail=200
      exit 1
    fi
    sleep 10
  done
done
```

Container restarts during multi-pod startup are **expected, not failures**:
a pod whose peer's rendezvous is not up yet fails fast and crash-loops
until the peer (deployed first) is ready, then recovers on a later retry --
do not abort the wait on restart counts or `ERROR:` lines alone. Only
report the endpoint ready once every pod in the plan is `Ready` -- one pod
becoming ready doesn't imply the others did. Diagnose individual pods with
`kubectl logs pod/<sts_name>-<ordinal>` (add `--previous` for the
incarnation before a restart).

Startup markers are keyed on the fixed log names every colocation recipe
uses today -- see `templates/statefulset.yaml`'s `ready_marker()` for the
exact lines. A recipe using different log names needs that template
updated.
