# Fetching reports whose pod is already gone

The main skill's step 5 copies reports straight out of the `vllm-bench`
pod, which holds open until the copy is done. Use this instead when that pod
no longer exists:

- the run was interrupted, or the pod was deleted, evicted, or hit
  `activeDeadlineSeconds`
- you want an **earlier** `RUN_ID` from a previous run, e.g. to complete a
  comparison (see [comparing-runs.md](comparing-runs.md))
- you want to see which `RUN_ID`s are on the PVC at all

Reports live on a reports PVC, independently of any pod, so every past run
is still retrievable as long as that PVC exists. Mount it from a short-lived
busybox helper, pinned to the node where that PVC is attached -- it is
`ReadWriteOnce`, so a second pod can only attach it from the node that
already holds it, otherwise `Multi-Attach error for volume ...` leaves the
helper `Pending` indefinitely. Pick the node from whatever already has it,
never from a serving pod's name (`vllm-pod` does not exist any more; a
multi-pod plan's pods are `${DEPLOY_NAME}-pod-<role>-<ordinal>`):

```bash
kubectl get pods -A -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}' \
  | grep -i bench || true   # last bench pod, if it still exists
```

Set `RUN_ID` to the run you want and `LOCAL_DIR` to where it should land:

```bash
RUN_ID=<run-id>
LOCAL_DIR="./reports/${RUN_ID}"
```

```bash
export HELPER=vllm-bench-reports-copy
# Pin only while another pod still holds vllm-bench-reports -- normally the
# last vllm-bench pod, or an earlier copy helper. An empty value means
# nothing holds it, so any node can attach it: drop the `nodeName:` line
# below in that case rather than emitting it empty.
export VLLM_NODE="<node from the lookup above>"
```

Both are exported because `envsubst` only reads **exported** variables: an
unexported name substitutes as empty with no error, which would render an
empty `metadata.name` and make `kubectl apply` reject the helper pod.

```bash
kubectl delete pod "${HELPER}" --ignore-not-found
envsubst '${HELPER} ${VLLM_NODE}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${HELPER}
  labels:
    app: vllm-bench
    role: reports-copy
spec:
  restartPolicy: Never
  nodeName: ${VLLM_NODE}
  containers:
    - name: copy
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: reports
          mountPath: /reports
          readOnly: true
  volumes:
    - name: reports
      persistentVolumeClaim:
        claimName: vllm-bench-reports
        readOnly: true
EOF

deadline=$(( $(date +%s) + 300 ))
until [ "$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null)" = "Running" ]; do
  phase="$(kubectl get pod "${HELPER}" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [ "$phase" = "Failed" ] || [ "$(date +%s)" -ge "$deadline" ]; then
    echo "helper pod not Running (phase=${phase:-<none>})"
    kubectl describe pod "${HELPER}" | tail -30
    kubectl delete pod "${HELPER}" --ignore-not-found
    exit 1
  fi
  sleep 5
done

kubectl exec "${HELPER}" -- ls /reports          # which RUN_IDs are on the PVC
mkdir -p "${LOCAL_DIR}"
kubectl cp "${HELPER}:/reports/${RUN_ID}/." "${LOCAL_DIR}"
kubectl delete pod "${HELPER}" --ignore-not-found
```

`kubectl exec ... -- ls /reports` is the cheap way to see what is on the PVC
before choosing a `RUN_ID`.

The helper mounts the PVC `readOnly`, so it cannot disturb a run in
progress. Pinning is harmful only when the node is wrong, so leave
`nodeName:` out entirely when nothing holds the claim (see the note above
the manifest).

If the PVC itself is gone, the reports are gone: nothing here recovers them.
Copy reports out promptly rather than relying on the PVC as archival storage.
