# Multi-pod AFD E2E on Kubernetes

Use this instead of the single-process pytest workflow when a scenario's pod
layout splits Attention and FFN ranks across more than one pod — optionally
spread across nodes, for a genuine cross-node fabric test — rather than
running everything in one process on one machine.

This walkthrough assumes the model under test is
`deepseek-ai/DeepSeek-V2-Lite`; substitute `MODEL`, `MODEL_PVC`, and the
scenario's topology if you're running a different model.

## How it works

The e2e test on Kubernetes runs as a Kubernetes Indexed Job that creates one
pod per pod-layout entry, behind a headless Service that gives each pod a
stable DNS name (`<job-name>-<index>.<job-name>`). Every pod runs the
identical in-pod runner command and derives its own role (which
Attention/FFN ranks to launch) from its Kubernetes-assigned completion
index, then rendezvous with its peers over a shared store before serving
and evaluating GSM8K. No process outside the pods holds test state or
drives the run: you apply the Service and Job once, and every pod decides
its own role, coordinates with its peers, and reports its own pass/fail
through its container exit code.

## Prerequisites

- An image with the AFD plugin, the full repo (including `tests/`), and the
  E2E test dependencies installed. `docker/Dockerfile.ci` builds this: its
  deps stage runs `uv export --group dev --group e2e-tests`, so `pytest`,
  `datasets`, and `huggingface_hub` (from `dev`) and `lm_eval[api]` (from
  `e2e-tests`, which pulls in `scipy` transitively) are already installed —
  there's no separate `lm_eval`/`scipy` install step to add.

  ```bash
  docker build -f docker/Dockerfile.ci -t <registry>/afd-plugin-e2e:<tag> .
  docker push <registry>/afd-plugin-e2e:<tag>
  ```

  Use an already-built image instead if one meeting that contract exists.

  **Building for OpenShift:** `Dockerfile.ci`'s final stage isn't usable
  as-is on OpenShift — build a variant with these changes:

  - Replace `COPY --link . .` with a plain `COPY . .`; `buildah` (the
    builder behind OpenShift's `BuildConfig`s) rejects `--link`.
  - OpenShift's default restricted Security Context Constraint (SCC) runs
    the container as an arbitrary, unpredictable UID that only belongs to
    group `0`, against an otherwise read-only image filesystem. Give that
    UID somewhere to write by setting a writable `HOME` and making the app
    directory group-writable:

    ```dockerfile
    ENV HOME=/work/home
    RUN mkdir -p /work/home \
        && chgrp -R 0 ${APP_DIR} /work \
        && chmod -R g=u ${APP_DIR} /work
    ```

  The Job template below relies on both of these — it also sets `fsGroup`
  under `securityContext` for the same SCC restriction (see **Optional
  additions**).
- A pre-existing, pre-warmed PersistentVolumeClaim holding the model
  weights. Nothing here creates or populates it — mount it into every pod
  yourself, as the Job template below does.
- `kubectl` configured against the target namespace/context, with rights to
  create/delete Services and Jobs.

## Choose a pod layout

A pod layout is a comma-separated list of `<int>A<int>F` entries, one per
pod: the number of Attention ranks and FFN ranks that pod should launch.
The number of entries fixes the pod count. For example, `2A0F,0A2F` is a
2-pod layout where pod 0 carries both Attention ranks and pod 1 carries
both FFN ranks; that string becomes the runner's `--pod-layout` argument
below, and its entry count becomes `NUM_PODS`, computed automatically from
`POD_LAYOUT` in the Deploy script — you don't set it yourself.

The ranks across all entries must sum to the scenario's topology (a
`2a2f` scenario needs 2 Attention and 2 FFN ranks in total), and no single
pod's rank count for a role may exceed that role's TP size — a TP group
cannot span pods.

This assumes every pod requests the same number of GPUs (`GPUS_PER_POD`
below): the layout can vary how many Attention/FFN ranks each pod carries,
but the Job template applies one shared `resources` block to every pod, so
per-pod GPU counts aren't supported as written.

## Deploy

Set the run's parameters, then apply a headless Service and an Indexed Job
built from them:

```bash
set -a   # envsubst reads the environment, so these must be exported
JOB_NAME=afd-e2e-run
NAMESPACE=afd-e2e
IMAGE=<registry>/afd-plugin-e2e:<tag>
SCENARIO=afd-graph-2a2f
POD_LAYOUT=2A0F,0A2F
NUM_PODS=$(($(tr -cd ',' <<<"$POD_LAYOUT" | wc -c) + 1))   # entry count of POD_LAYOUT
RUN_ID=$(date +%s)
MODEL=deepseek-ai/DeepSeek-V2-Lite
GSM8K_OUTPUT_PATH=/work/gsm8k-results
MODEL_PVC=deepseek-v2-lite-weights
GPUS_PER_POD=2
CPU_REQUEST=${CPU_REQUEST:-16}
CPU_LIMIT=${CPU_LIMIT:-32}
MEMORY_REQUEST=${MEMORY_REQUEST:-128Gi}
MEMORY_LIMIT=${MEMORY_LIMIT:-200Gi}
SHM_SIZE=${SHM_SIZE:-16Gi}
set +a

envsubst '${JOB_NAME} ${NAMESPACE} ${IMAGE} ${SCENARIO} ${POD_LAYOUT} ${NUM_PODS} ${RUN_ID} ${MODEL} ${GSM8K_OUTPUT_PATH} ${MODEL_PVC} ${GPUS_PER_POD} ${CPU_REQUEST} ${CPU_LIMIT} ${MEMORY_REQUEST} ${MEMORY_LIMIT} ${SHM_SIZE}' <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels: {app: afd-e2e, run: ${JOB_NAME}}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector: {app: afd-e2e, run: ${JOB_NAME}}
  ports:
    - {name: rendezvous, port: 29500}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  labels: {app: afd-e2e, run: ${JOB_NAME}}
spec:
  completionMode: Indexed
  completions: ${NUM_PODS}
  parallelism: ${NUM_PODS}      # must equal completions, or a not-yet-started
                                 # pod can never reach a peer's rendezvous barrier
  backoffLimit: 0                # a retried pod would rejoin barriers that have
                                  # already moved past it; a failed pod fails the run
  template:
    metadata:
      labels: {app: afd-e2e, run: ${JOB_NAME}}
    spec:
      restartPolicy: Never
      subdomain: ${JOB_NAME}      # ties each pod's DNS name to the Service above
      volumes:
        - name: model-storage
          persistentVolumeClaim: {claimName: ${MODEL_PVC}}
        - name: dshm               # a private /dev/shm per pod, not shared scratch
          emptyDir: {medium: Memory, sizeLimit: ${SHM_SIZE}}
        - name: work
          emptyDir: {}
      containers:
        - name: e2e
          image: ${IMAGE}
          imagePullPolicy: Always
          workingDir: /opt/afd-plugin
          # The work emptyDir shadows whatever the image created there, so its
          # subdirectories must be recreated on every start.
          command: ["/bin/bash", "-c", "set -euo pipefail\nmkdir -p /work/home /work/tmp /work/hf_modules\nexec \"$@\"", "afd-e2e"]
          args:
            - python
            - -m
            - tests.e2e.multi_pod.runner
            - --scenario
            - ${SCENARIO}
            - --pod-layout
            - ${POD_LAYOUT}
            - --run-id
            - ${RUN_ID}
            - --model
            - ${MODEL}
            - --gsm8k-output-path
            - ${GSM8K_OUTPUT_PATH}
            - --store-host
            - ${JOB_NAME}-0.${JOB_NAME}
            - --pod-address-template
            - ${JOB_NAME}-{index}.${JOB_NAME}
          env:
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - {name: HOME, value: /work/home}
            - {name: TMPDIR, value: /work/tmp}
            - {name: USER, value: afd}
            - {name: LOGNAME, value: afd}
            - {name: XDG_CACHE_HOME, value: /work/xdg}
            - {name: TORCHINDUCTOR_CACHE_DIR, value: /work/inductor}
            - {name: TRITON_CACHE_DIR, value: /work/triton}
            - {name: VLLM_CACHE_ROOT, value: /work/vllm}
            - {name: UV_CACHE_DIR, value: /work/uv}
            - {name: HF_MODULES_CACHE, value: /work/hf_modules}
            - {name: PYTHONDONTWRITEBYTECODE, value: "1"}
            - {name: PYTHONUNBUFFERED, value: "1"}
          resources:
            requests: {nvidia.com/gpu: "${GPUS_PER_POD}", cpu: ${CPU_REQUEST}, memory: ${MEMORY_REQUEST}}
            limits: {nvidia.com/gpu: "${GPUS_PER_POD}", cpu: ${CPU_LIMIT}, memory: ${MEMORY_LIMIT}}
          volumeMounts:
            - {name: model-storage, mountPath: /models}
            - {name: dshm, mountPath: /dev/shm}
            - {name: work, mountPath: /work}
EOF
```

`completionMode: Indexed` is what makes Kubernetes inject a
`JOB_COMPLETION_INDEX` env var into each pod — that's how a pod learns which
entry of `POD_LAYOUT` is its own; nothing in the pod spec sets it directly.

## Optional additions

- **Set an env var for the launched vLLM processes** (not the container
  itself): append to `args`, once per variable —
  `- --pod-env`, `- KEY=VALUE`. The in-pod runner merges these into every
  vLLM process it launches.
- **Override or add a container-level env var:** add or replace an entry
  under the container's `env:` list.
- **A hard ceiling on the Job's own runtime**, independent of how long you
  wait on it below: add `activeDeadlineSeconds: <seconds>` under `spec:` on
  the Job.
- **Set `fsGroup` to satisfy the namespace's Security Context Constraint
  (SCC) range (OpenShift):** add `securityContext: {fsGroup: <n>}` under
  the pod template's `spec:`, using a group id the namespace's SCC actually
  allows — check with `oc get scc restricted -o yaml` (or whichever SCC the
  namespace binds) or ask a cluster admin.

## Run and observe

Watch pods leave `Pending`:

```bash
kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME} -w
```

Block until the Job completes or times out:

```bash
kubectl wait job/${JOB_NAME} -n ${NAMESPACE} --for=condition=complete --timeout=5400s
```

A non-zero result here means either the Job failed or the wait itself timed
out — it does not distinguish the two. Read each pod's own terminal state to
get the real result:

```bash
kubectl get pods -n ${NAMESPACE} -l job-name=${JOB_NAME} \
  -o custom-columns='POD:.metadata.name,INDEX:.metadata.labels.batch\.kubernetes\.io/job-completion-index,EXIT:.status.containerStatuses[0].state.terminated.exitCode'

kubectl logs -n ${NAMESPACE} <pod-name>
```

The run passed only if every pod's exit code is `0`; a pod with no terminal
state yet (still running when the wait timed out) is not a pass.

## Clean up

```bash
kubectl delete job/${JOB_NAME} service/${JOB_NAME} -n ${NAMESPACE} --ignore-not-found
```

A completed Job's pod template is immutable, so re-applying under the same
`JOB_NAME` fails until the old Job is deleted — delete before re-running,
not after. Skip deletion (leave the Job, Service, and pods running) when you
want to `kubectl exec` into a pod afterward for interactive follow-up.
