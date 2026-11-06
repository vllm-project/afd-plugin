# 2-4. Confirm and deploy

## 2. GPU_COUNT

Use the `GPU_COUNT` recorded by [resolve-recipe.md](resolve-recipe.md) --
one value for a plain recipe, one per plan entry. Don't re-read the script
here; all recipe parsing lives in step 1.

## 3. Confirm before touching the cluster

State and confirm: target cluster/namespace, `AFD_PLUGIN_IMAGE`, `MODEL_ID`,
`PVC_NAME`, `CLIENT_PORT`, `RECIPE_SCRIPT_PATH`, and either `GPU_COUNT`
(plain recipe) or the plan as a table of `pod_name | role | GPU_COUNT`
(placement-driven recipe, straight from resolve-recipe.md -- no shape to
negotiate, just restate what was parsed). Flag that step 4 **deletes any
existing Pod(s)** with the same name(s) about to be deployed.

```bash
AFD_PLUGIN_IMAGE=<image>
MODEL_ID=<model-id>
PVC_NAME=<pvc-name>
CLIENT_PORT=<n>   # from the script's --port; default 18305
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
```

**Node placement is left to the scheduler by default.** Only add
`podAntiAffinity` forcing Pods onto separate nodes if the caller explicitly
asks to exercise cross-node placement -- add this to every Pod's
`metadata.labels` and `spec` in `templates/pod.yaml` before applying:

```yaml
  labels:
    afd-multinode-group: afd-multinode
  spec:
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - {key: afd-multinode-group, operator: In, values: ["afd-multinode"]}
            topologyKey: kubernetes.io/hostname
```

## 4a. Model PVC

Size for the model being served. An existing undersized PVC is reused
silently and fails mid-download, so set `MODEL_PVC_SIZE` deliberately. `PVC_ACCESS_MODE` is derived, not asked:
`ReadWriteMany` whenever the plan (if any) has more than one pod -- an RWO
volume can only attach from one node, and the second Pod deadlocks
`Pending` with `Multi-Attach error` otherwise.

```bash
MODEL_PVC_SIZE=${MODEL_PVC_SIZE:-100Gi}
STORAGE_CLASS=${STORAGE_CLASS:-}
PVC_ACCESS_MODE=ReadWriteOnce   # ReadWriteMany if the plan has >1 pod

if kubectl get pvc "${PVC_NAME}" >/dev/null 2>&1; then
  echo "PVC ${PVC_NAME} exists; reusing its warm HF_HOME cache for ${MODEL_ID}"
  echo "NOTE: verify its size fits ${MODEL_ID}, and its accessModes include"
  echo "      ReadWriteMany if this plan has >1 pod -- an RWO reuse deadlocks the 2nd pod"
else
  envsubst '${PVC_NAME} ${MODEL_PVC_SIZE} ${STORAGE_CLASS} ${PVC_ACCESS_MODE}' \
    < templates/pvc.yaml | grep -v 'storageClassName: *$' | kubectl apply -f -
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
kubectl create configmap afd-recipe-script \
  --from-file=recipe.sh="${RECIPE_SCRIPT_PATH}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 4c. fsGroup

Must be inside the namespace's allowed range or the pod is rejected at
admission. Look it up once, reuse for every pod:

```bash
FS_GROUP="$(kubectl get ns "$(kubectl config view --minify -o jsonpath='{..namespace}')" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
FS_GROUP="${FS_GROUP:-1000}"
```

## 4d. Services

Create every Service that applies -- see [services.md](services.md) for
which, and why this must happen **before** deploying any Pod in 4e below.
Everything a Service needs (ports, selectors, which of the four apply)
comes from the plan parsed in resolve-recipe.md, so none of this waits on a
Pod existing.

## 4e. Deploy pod(s)

One call per plan entry (or a single call with no overrides for a plain
recipe) against the one shared `templates/pod.yaml`:

```bash
deploy_pod() {
  local pod_name="$1" gpu="$2" attn_role="$3" ffn_role="$4" extra_env_pairs="$5"
  # extra_env_pairs: space-separated VAR=value tokens straight off this
  # plan entry's header line (including POD=... itself), empty for a plain
  # recipe. Build the spliced env-entries block for templates/pod.yaml.
  # Leading (not trailing) newlines keep the `# ${TEMPLATE_EXTRA_ENV}`
  # placeholder line a harmless comment when this is empty, and valid
  # sibling list items when it isn't.
  local extra_env=""
  for pair in $extra_env_pairs; do
    extra_env="$extra_env
        - {name: ${pair%%=*}, value: \"${pair#*=}\"}"
  done

  kubectl delete pod "$pod_name" --ignore-not-found

  TEMPLATE_IMAGE="$AFD_PLUGIN_IMAGE" TEMPLATE_MODEL="$MODEL_ID" TEMPLATE_POD="$pod_name" \
  TEMPLATE_GPU="$gpu" TEMPLATE_PVC="$PVC_NAME" TEMPLATE_FSGROUP="$FS_GROUP" \
  TEMPLATE_CLIENT_PORT="$CLIENT_PORT" TEMPLATE_ATTN_ROLE="$attn_role" TEMPLATE_FFN_ROLE="$ffn_role" \
  TEMPLATE_EXTRA_ENV="$extra_env" \
  envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_POD} ${TEMPLATE_GPU} ${TEMPLATE_PVC} ${TEMPLATE_FSGROUP} ${TEMPLATE_CLIENT_PORT} ${TEMPLATE_ATTN_ROLE} ${TEMPLATE_FFN_ROLE} ${TEMPLATE_EXTRA_ENV}' \
    < templates/pod.yaml | kubectl apply -f -
}
```

Plain recipe, one pod, both roles head, no overrides:

```bash
deploy_pod vllm-pod "$GPU_COUNT" head head ""
```

Placement-driven recipe, one call per entry parsed in
[resolve-recipe.md](resolve-recipe.md), e.g. the qwen3.5 3-pod plan:

```bash
deploy_pod vllm-pod-attention-0 2 head none "POD=ATTENTION_0"
deploy_pod vllm-pod-attention-1 2 worker none "POD=ATTENTION_1 ATTENTION_HEADLESS=1 ATTENTION_DP_START_RANK=2"
deploy_pod vllm-pod-ffn-0 2 none head "POD=FFN_0"
```

## 4f. Wait for Running, then for each pod's Ready condition

Bounded 15-minute wait per pod for `phase=Running` (unschedulable pods never
reach `Failed`, so an unbounded wait would spin forever), then a 60-minute
wait per pod for its `Ready` condition (model download and graph capture
dominate).

`Ready` is the single readiness signal. The pod's `readinessProbe` only
turns it on after the in-pod startup script has seen every log marker, got
`/health` from the attention head, and printed `=== pod <name> READY ===`;
the same condition is what puts the attention head behind `vllm-service`.
If the in-pod script gives up, it prints `ERROR:` with log tails and exits,
the pod goes `Failed`, and the loop below stops at once.

Timeouts are nested on purpose: the in-pod startup deadline
(`STARTUP_TIMEOUT_S` in `templates/pod.yaml`, 55 minutes, one budget shared
by every log and `/health`) is shorter than the 60-minute wait here, so a
slow pod always fails with its own `ERROR:` reason instead of an outer
timeout. Change the two together.

```bash
for pod_name in <every deployed pod_name>; do
  kubectl wait pod "$pod_name" --for=jsonpath='{.status.phase}'=Running --timeout=15m \
    || { kubectl describe pod "$pod_name" | tail -40; exit 1; }
done

for pod_name in <every deployed pod_name>; do
  deadline=$(( $(date +%s) + 3600 ))
  while true; do
    read -r phase ready < <(kubectl get pod "$pod_name" \
      -o jsonpath='{.status.phase} {.status.conditions[?(@.type=="Ready")].status}' || true)
    [ "$ready" = "True" ] && break
    if [ "$phase" = "Failed" ] || [ "$(date +%s)" -ge "$deadline" ]; then
      echo "pod $pod_name not Ready (phase=${phase:-unknown})"
      kubectl logs "$pod_name" --tail=200
      exit 1
    fi
    sleep 10
  done
done
```

Only report the endpoint ready once every pod in the plan is `Ready` -- one
pod becoming ready doesn't imply the others did.

Startup markers are keyed on the fixed log names every colocation recipe
uses today -- see `templates/pod.yaml`'s `ready_marker()` for the exact
lines. A recipe using different log names needs that template updated.
