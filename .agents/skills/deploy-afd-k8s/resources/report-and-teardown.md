# 5. Report back

Tell the caller: the endpoint `http://${DEPLOY_NAME}-service:${CLIENT_PORT}`
(in namespace `${NAMESPACE}`, resolved in deploy.md step 3),
`MODEL_ID` (and whether it's a HF repo id or an in-container path),
`PVC_NAME`, `GPU_COUNT` (per role, if the plan spans pods), the recipe deployed,
its `--max-num-batched-tokens`, and the node(s) it landed on. Select by the
`afd-deployment` label, not `app=afd-recipe`, so a second deployment in the
namespace is not reported as part of this one:

```bash
kubectl -n "$NAMESPACE" get pods -l "afd-deployment=$DEPLOY_NAME" -o wide
kubectl -n "$NAMESPACE" get sts -l "afd-deployment=$DEPLOY_NAME"
```

Every node matters if a follow-up pod needs to attach the same
`ReadWriteOnce` PVC -- it can only attach from that node.

**Ask when to delete the running pods.** End the report by asking the
caller -- the user, or the task that invoked this skill (e.g.
`run-vllm-bench-k8s`) -- when the pods should be deleted: right after its
follow-up work, at a later point it names, or never (keep them warm). The
caller owns that decision and triggers step 6 when the time comes; this
skill never tears down on its own.

# 6. Teardown

Left running by design, so weights stay warm -- report what's up and how to
remove it; do not delete unless asked. Everything this deployment created
is labeled `afd-deployment=<DEPLOY_NAME>`, so delete by label -- no
hand-maintained name lists, and a second deployment in the namespace is
untouched:

```bash
kubectl -n "$NAMESPACE" delete sts -l "afd-deployment=$DEPLOY_NAME"
kubectl -n "$NAMESPACE" delete svc,configmap -l "afd-deployment=$DEPLOY_NAME"
```

The model PVC is intentionally not deleted (it is also not labeled -- it is
shared, user-named state, not part of the deployment's label set) -- deleting
it discards the warm cache and forces a full re-download.
