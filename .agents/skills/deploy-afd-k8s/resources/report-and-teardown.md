# 5. Report back

Tell the caller: the endpoint `http://vllm-service:${CLIENT_PORT}`,
`MODEL_ID` (and whether it's a HF repo id or an in-container path),
`PVC_NAME`, `GPU_COUNT` (per pod, if more than one), the recipe deployed,
its `--max-num-batched-tokens`, and the node(s) it landed on:

```bash
kubectl get pods -l app=afd-recipe -o wide
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
remove it; do not delete unless asked:

```bash
kubectl delete pod <pod_name...>
kubectl delete svc vllm-service vllm-ffn-p2p-service vllm-attn-dp-service vllm-ffn-dp-service --ignore-not-found
```

The model PVC is intentionally not listed -- deleting it discards the warm
cache and forces a full re-download.
