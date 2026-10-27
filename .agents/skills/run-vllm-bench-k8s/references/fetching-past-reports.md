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
busybox helper, pinned to the serving node -- the PVC is `ReadWriteOnce`, so
a second pod can only attach it from the node where it is already mounted,
otherwise `Multi-Attach error for volume ...` leaves the helper `Pending`
indefinitely.

Set `RUN_ID` to the run you want and `LOCAL_DIR` to where it should land:

```bash
RUN_ID=<run-id>
LOCAL_DIR="./reports/${RUN_ID}"
```

```bash
HELPER=vllm-bench-reports-copy
VLLM_NODE="$(kubectl get pod vllm-pod -o jsonpath='{.spec.nodeName}')"

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
progress. It is still safe to skip the node pin only if no other pod has the
PVC attached -- pinning is harmless either way, so keep it.

If the PVC itself is gone, the reports are gone: nothing here recovers them.
Copy reports out promptly rather than relying on the PVC as archival storage.
