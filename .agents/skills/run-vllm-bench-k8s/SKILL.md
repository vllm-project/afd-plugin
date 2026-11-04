---
name: run-vllm-bench-k8s
description: Use when the user asks to benchmark an AFD GPU recipe on Kubernetes/OpenShift with vllm bench serve (or the repo's tools/benchmarks/request_generator.sh) - measuring throughput, TTFT, TPOT, or ITL for a deployed recipe and producing a report for that run. Deploys via deploy-afd-k8s. Do not use for local (non-k8s) benchmarking, NPU recipes, or E2E correctness testing (see run-e2e).
---

# Benchmark an AFD recipe with `vllm bench serve`

Deploy a recipe, drive load against it with the repo's
[tools/benchmarks/request_generator.sh](../../../tools/benchmarks/request_generator.sh)
(a thin wrapper around `vllm bench serve`), and produce a self-contained
report for **that one run**.

One invocation = one run = one `RUN_ID` = one report.

## Scope

- GPU colocation recipes only -- whatever `deploy-afd-k8s` supports,
  including multi-pod plans: one pod per role, or one role split across
  pods. On a split-role plan the attention head pod is the only replica
  serving the client, and its loss does not self-heal (see the recovery
  scope in `deploy-afd-k8s`); mid-sweep that looks like `completed <
  num_prompts`, not saturation. Such a run is also not comparable to the
  single-pod recipe of the same model -- say so in the report.
- Not for local benchmarking (run `request_generator.sh` directly against
  `decode_bench_server.sh` for that) or correctness E2E (`run-e2e`).
- Requires everything `deploy-afd-k8s` requires. No extra image: the bench
  pod runs `AFD_PLUGIN_IMAGE`, which already carries the repo at
  `/opt/afd-plugin` and vLLM's `bench` CLI.

## Workflow

### 1. Establish the run identity

```bash
RECIPE_SCRIPT_PATH=<local-recipe-script-path>
RUN_ID="$(basename "${RECIPE_SCRIPT_PATH}" .sh)-$(date +%Y%m%d-%H%M%S)"
LOCAL_DIR="./reports/${RUN_ID}"
echo "RUN_ID=${RUN_ID}"
```

Every artifact is keyed on `RUN_ID`, which is what keeps one run's numbers
from being confused with another's.

If the recipe is not provided, ask which recipe to use. Use the `./recipe`
folder as the list of possible recipes.

### 2. Deploy the recipe

Follow **`deploy-afd-k8s`** with `RECIPE_SCRIPT_PATH`. It returns
`DEPLOY_NAME` (default `vllm`), `NAMESPACE`, `MODEL_ID`, `PVC_NAME`,
`GPU_COUNT`, the node(s) the pods landed on, the recipe's
`--max-num-batched-tokens`, and an endpoint at
`http://${DEPLOY_NAME}-service:${CLIENT_PORT}`, where `CLIENT_PORT` is the
recipe's attention `--port` (default 18305). `GPU_COUNT` is one number for a
plain recipe and a per-role value when the plan spans pods.

Carry `DEPLOY_NAME`, `NAMESPACE`, and `CLIENT_PORT` into every later step:
the endpoint name and port belong to the deployment, so a hard-coded
`vllm-service:18305` only works while both defaults happen to match.

The bench pod runs the same image the deployment does -- `AFD_PLUGIN_IMAGE`,
the value you passed to `deploy-afd-k8s`. That skill only uses it as a prefix
assignment on its own `envsubst`, so it sets nothing you can read back later;
the manifest's `TEMPLATE_IMAGE` is initialised from it in step 4.

```bash
export DEPLOY_NAME=${DEPLOY_NAME:-vllm}
export NAMESPACE="<namespace reported by deploy-afd-k8s>"
export CLIENT_PORT="<the recipe's attention --port>"      # default 18305
export GPU_COUNT="<n>"                                     # or per role, see above
export DEPLOY_NODES="$(kubectl -n "$NAMESPACE" get pods \
  -l "afd-deployment=$DEPLOY_NAME" \
  -o jsonpath='{range .items[*]}{.spec.nodeName}{"\n"}{end}' \
  | sort -u | paste -sd,)"
```

`vllm bench serve` is pointed at that endpoint with
`HOST=${DEPLOY_NAME}-service` `PORT=${CLIENT_PORT}`, and
`MODEL_PATH=${MODEL_ID}` -- it must name exactly what the pod serves, since
the model id is sent in each request and used to load the tokenizer.

On a multi-pod plan the client reaches the **attention head pod**, so the
attention role's `--max-num-batched-tokens` is the budget that governs
backpressure in step 3. The roles can differ (attention `64`, FFN unset, in
the `2a2f`/`2a2a2f` plans) -- use the attention role's, not an average and
not the FFN role's.

### 3. Choose the workload

**If the user did not say what to run, ask before deploying anything.** Put
the question as a choice between the named workloads below, and say which
one you will use if they have no preference. Do not silently pick one -- the
workload *is* the result: a throughput number without its prompt shape and
offered load is not quotable.

Ask in this form:

> Which workload should I benchmark `<recipe>` with?
>   1. **balanced** (default) -- random 1024-token prompts, 256-token
>      outputs, Poisson arrivals swept 2 -> 16 req/s. General serving
>      behaviour and where latency starts to climb.
>   2. **max-throughput** -- same prompts, all requests issued at once,
>      concurrency swept 16 -> 128. The peak tokens/s this recipe can reach.
>   3. **prefix-heavy** -- 10 shared 256-token prefixes reused across
>      requests. Prefix-cache behaviour.
>   4. **long-context** -- 8192-token prompts, 256-token outputs. Prefill
>      cost on long inputs.
>   5. **smoke** -- 64 short requests, ~1 minute. Checks the wiring only;
>      never a published number.
>   Or give me your own ISL/OSL, rate and concurrency.

#### Built-in defaults

Each workload is a complete set of values -- adopt them as-is unless the
user overrides a specific knob. `SWEEP` is a list of `rate:concurrency`
points; `vllm bench serve` is **single-shot**, so each point is one separate
invocation of the wrapper (see step 4).

| Workload | `DATASET_NAME` | `INPUT_LEN` | `OUTPUT_LEN` | `NUM_PROMPTS` | `SWEEP` (`rate:conc`) | Extra flags |
|---|---|---|---|---|---|---|
| **balanced** (default) | `random` | 1024 | 256 | 512 | `2:64 4:64 8:64 16:64` | -- |
| **max-throughput** | `random` | 1024 | 256 | 512 | `inf:16 inf:32 inf:64 inf:128` | -- |
| **prefix-heavy** | `prefix_repetition` | (unused) | (unused) | 512 | `2:64 4:64 8:64` | `--prefix-repetition-prefix-len 256 --prefix-repetition-suffix-len 256 --prefix-repetition-num-prefixes 10 --prefix-repetition-output-len 256` |
| **long-context** | `random` | 8192 | 256 | 256 | `1:32 2:32 4:32` | -- |
| **smoke** | `random` | 256 | 128 | 64 | `inf:8` | -- |

`REQUEST_RATE=inf` means "issue every request at time 0", so with
`MAX_CONCURRENCY` it is a closed-loop test measuring peak throughput. A
finite rate uses Poisson arrivals and measures behaviour under an offered
load, which is what latency SLOs care about. The two answer different
questions -- do not mix them in a single sweep.

`prefix_repetition` ignores `INPUT_LEN`/`OUTPUT_LEN` and takes its shape
from its own flags, which go in `EXTRA_ARGS`. It must **not** be given
`--dataset-path` (nor may `random`); vLLM rejects that combination.

#### Other datasets

`--dataset-name` also accepts `sharegpt`, `burstgpt`, `sonnet`, `hf`,
`custom`, and several multimodal variants. These need a real corpus, so pass
`--dataset-path <file-or-hf-id>` through `EXTRA_ARGS` and mount it into the
bench pod. Use one when the user asks for realistic traffic rather than
synthetic prompts; otherwise prefer `random`, whose ISL/OSL are exactly
controllable and therefore reproducible.

#### Sizing the sweep against the recipe

Match offered load to the recipe's `--max-num-batched-tokens` from step 2 --
on a multi-pod plan, the attention role's, since that is the pod the client
talks to. A recipe pinned at 64 absorbs a 1024-token prompt in ~16
chunked-prefill steps, so a high rate puts it in permanent backpressure and
the report measures queueing rather than the recipe. Long-context makes this
sharper: an 8192-token prompt at that budget needs ~128 steps.

Sweep until at least one point saturates -- TTFT climbing, or `completed`
below `NUM_PROMPTS`. A sweep where nothing saturates measured headroom, not
capacity, and the report has to say so.

### 4. Run the sweep

The pod runs the AFD image, so the wrapper is already on disk at
`/opt/afd-plugin/tools/benchmarks/request_generator.sh`. Two details:

- The wrapper calls `uv run vllm bench serve`. The image's base Python
  environment is already correct, so a `uv` shim on `PATH` strips `uv run` --
  resolved before the shim is installed, since afterwards `command -v uv`
  would find the shim itself and recurse.
- `EXTRA_ARGS` adds `e2el` and p90. `vllm bench serve` defaults
  `--percentile-metrics` to `ttft,tpot,itl` (no `e2el`) and
  `--metric-percentiles` to `99` only, so without this the report cannot
  show end-to-end latency or p90.

Export the workload chosen in step 3 -- these are the `balanced` defaults.
`envsubst` only sees **exported** variables -- an unexported name
substitutes as empty with no error -- so the block below re-exports every
name it interpolates, including step 1's `RUN_ID` and the values carried
over from step 2:

```bash
# deploy-afd-k8s passes its image as a prefix assignment to its own envsubst,
# so TEMPLATE_IMAGE is never left set in this shell -- initialise it from the
# deployment's image. envsubst skips unexported names silently, and an empty
# RUN_ID, MODEL_ID or PVC_NAME gives a pod that honours none of the run.
export TEMPLATE_IMAGE="${TEMPLATE_IMAGE:-$AFD_PLUGIN_IMAGE}"
[ -n "$TEMPLATE_IMAGE" ] || { echo "set AFD_PLUGIN_IMAGE to the image deploy-afd-k8s used"; exit 1; }
export MODEL_ID PVC_NAME RUN_ID DEPLOY_NAME CLIENT_PORT
export VLLM_NODE          # empty is fine when you are not pinning

export DATASET_NAME=random
export INPUT_LEN=1024
export OUTPUT_LEN=256
export NUM_PROMPTS=512
export SWEEP="2:64 4:64 8:64 16:64"      # rate:concurrency
export WORKLOAD_ARGS=""                  # e.g. the --prefix-repetition-* flags

# Every bench-pod command below is namespaced with -n "$NAMESPACE" (exported
# in step 2): the deployment's Services only exist there, so an un-namespaced
# create would leave the pod in the kubeconfig context's namespace, where
# ${DEPLOY_NAME}-service does not resolve.
kubectl -n "$NAMESPACE" delete pod vllm-bench --ignore-not-found

envsubst '${TEMPLATE_IMAGE} ${MODEL_ID} ${RUN_ID} ${SWEEP} ${DATASET_NAME} ${INPUT_LEN} ${OUTPUT_LEN} ${NUM_PROMPTS} ${WORKLOAD_ARGS} ${PVC_NAME} ${VLLM_NODE} ${DEPLOY_NAME} ${CLIENT_PORT}' <<'EOF' | kubectl -n "$NAMESPACE" apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-bench-reports
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: vllm-bench
  labels:
    app: vllm-bench
spec:
  restartPolicy: Never
  activeDeadlineSeconds: 14400
  containers:
    - name: bench
      image: ${TEMPLATE_IMAGE}
      imagePullPolicy: Always
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -uo pipefail
          mkdir -p /work/bin
          REAL_UV="$(command -v uv || true)"
          printf '#!/bin/bash\nif [ "$1" = "run" ]; then\n  shift\n  exec "$@"\nfi\nREAL_UV="%s"\n[ -n "$REAL_UV" ] || { echo "uv not found in image" >&2; exit 127; }\nexec "$REAL_UV" "$@"\n' "$REAL_UV" > /work/bin/uv
          chmod +x /work/bin/uv
          export PATH="/work/bin:$PATH"

          GEN=/opt/afd-plugin/tools/benchmarks/request_generator.sh
          [ -x "$GEN" ] || { echo "ERROR: $GEN missing from image"; sleep infinity; }

          OUT="/reports/${RUN_ID}"
          mkdir -p "$OUT"
          rc=0
          for point in ${SWEEP}; do
            rate="${point%%:*}"
            conc="${point##*:}"
            echo "=== sweep point rate=${rate} concurrency=${conc} ==="
            MODEL_PATH="${MODEL_ID}" \
            HOST="${DEPLOY_NAME}-service" PORT="${CLIENT_PORT}" \
            DATASET_NAME="${DATASET_NAME}" \
            INPUT_LEN="${INPUT_LEN}" OUTPUT_LEN="${OUTPUT_LEN}" \
            NUM_PROMPTS="${NUM_PROMPTS}" \
            REQUEST_RATE="${rate}" MAX_CONCURRENCY="${conc}" \
            RESULT_DIR="$OUT" \
            RESULT_FILENAME="rate_${rate}_conc_${conc}.json" \
            EXTRA_ARGS="--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,99 --label rate_${rate}_conc_${conc} ${WORKLOAD_ARGS}" \
            bash "$GEN" || { echo "sweep point rate=${rate} conc=${conc} FAILED"; rc=1; }
          done

          echo "$rc" > "$OUT/.exit_code"
          echo "=== sweep finished, exit=$rc; holding pod so reports can be copied ==="
          sleep infinity
      env:
        - name: USER
          value: "vllm"
        - name: HOME
          value: "/work/home"
        - name: HF_HOME
          value: "/models/.hf_home"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
              optional: true
      resources:
        requests:
          cpu: "4"
          memory: 16Gi
        limits:
          cpu: "8"
          memory: 32Gi
      volumeMounts:
        - name: work
          mountPath: /work
        - name: reports
          mountPath: /reports
        - name: model-storage
          mountPath: /models
          readOnly: true
  volumes:
    - name: work
      emptyDir: {}
    - name: reports
      persistentVolumeClaim:
        claimName: vllm-bench-reports
    - name: model-storage
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
        readOnly: true
EOF
```

The loop does not `set -e`: one failing sweep point records `rc=1` but the
remaining points still run, so a single bad rate does not discard the whole
sweep. The sentinel is written either way.

**The model-storage mount above is required whenever `MODEL_ID` is an
in-container path** rather than a HF repo id (step 2) -- which is the common
case, since every recipe in this repo defaults `MODEL_PATH` to one. Without
it, `vllm bench serve` resolves `--model` as a HF repo id and fails with
`HFValidationError: Repo id must be in the form ...`, because it cannot load
the tokenizer from a path that doesn't exist in the bench pod. If `MODEL_ID`
is a genuine HF repo id instead, the mount is harmless to leave in place.

**Pin the pod to a node holding the model PVC, only if that PVC is
`ReadWriteOnce`**:

```bash
kubectl -n "$NAMESPACE" get pvc "${PVC_NAME}" -o jsonpath='{.spec.accessModes[0]}'
```

If that prints `ReadWriteOnce`, a second pod can only attach the PVC from
the node that already holds it. `export VLLM_NODE=<one of DEPLOY_NODES from
step 2>` -- a plain recipe has exactly one, so there is nothing to choose --
add `${VLLM_NODE}` to the `envsubst` variable list above, and add
`nodeName: ${VLLM_NODE}` at the pod spec level (sibling of `restartPolicy`)
-- otherwise the pod leaves `Pending` with `Multi-Attach error for volume
...`. If it prints `ReadWriteMany` (which `deploy-afd-k8s` derives for every
plan spanning more than one pod), skip `nodeName` entirely -- no pinning
needed, any node can attach it.

### 5. Wait for completion, then collect

The pod holds after the sweep, so completion is the sentinel file, not a
`Succeeded` phase -- which also preserves the exit code:

```bash
# Never foreground-follow the log: the container holds with `sleep infinity`
# until activeDeadlineSeconds (4h), so `kubectl logs -f` never returns and
# the sentinel polling below would never run. Read progress without -f, or
# follow it in the background.
kubectl -n "$NAMESPACE" logs pod/vllm-bench --tail=200 2>/dev/null || true
# kubectl -n "$NAMESPACE" logs -f pod/vllm-bench > "/tmp/${RUN_ID}.log" 2>&1 &

deadline=$(( $(date +%s) + 7200 ))
RC=""
while true; do
  phase="$(kubectl -n "$NAMESPACE" get pod vllm-bench -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  case "$phase" in
    "")      echo "vllm-bench pod is gone"; exit 1 ;;
    Failed)  echo "vllm-bench Failed (OOM, or activeDeadlineSeconds hit)"
             kubectl -n "$NAMESPACE" describe pod vllm-bench | tail -20
             kubectl -n "$NAMESPACE" logs vllm-bench --tail=100
             echo "finished points are still on vllm-bench-reports -- see references/fetching-past-reports.md"
             exit 1 ;;
  esac
  RC="$(kubectl -n "$NAMESPACE" exec vllm-bench -- cat "/reports/${RUN_ID}/.exit_code" 2>/dev/null || true)"
  [ -n "$RC" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out after 2h waiting for the sweep (phase=${phase})"
    kubectl -n "$NAMESPACE" logs vllm-bench --tail=50; exit 1
  fi
  sleep 10
done

[ "$RC" = "0" ] || echo "at least one sweep point failed; the report must say which"
echo "=== sweep finished, exit=${RC} ==="

mkdir -p "${LOCAL_DIR}"
kubectl -n "$NAMESPACE" cp "vllm-bench:/reports/${RUN_ID}/." "${LOCAL_DIR}"
ls "${LOCAL_DIR}"

kubectl -n "$NAMESPACE" delete pod vllm-bench
```

Delete only **after** confirming the files are local -- that is the whole
reason the pod holds.

**If that pod is already gone** -- an interrupted run, or an earlier
`RUN_ID` -- reports remain on the `vllm-bench-reports` PVC. See
[references/fetching-past-reports.md](references/fetching-past-reports.md).

### 6. Record the run manifest

```bash
cat > "${LOCAL_DIR}/run.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "harness": "vllm-bench-serve",
  "recipe": "${RECIPE_SCRIPT_PATH}",
  "model_id": "${MODEL_ID}",
  "image": "${TEMPLATE_IMAGE}",
  "gpu_count": "${GPU_COUNT}",
  "max_num_batched_tokens": <the attention role's, from deploy step 2>,
  "workload": "<balanced|max-throughput|prefix-heavy|long-context|smoke|custom>",
  "dataset": "${DATASET_NAME}",
  "workload_args": "${WORKLOAD_ARGS}",
  "input_len": ${INPUT_LEN},
  "output_len": ${OUTPUT_LEN},
  "num_prompts": ${NUM_PROMPTS},
  "sweep": "${SWEEP}",
  "nodes": "${DEPLOY_NODES}",
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
```

Recording the workload and sweep next to the numbers is what keeps the
report interpretable after the cluster is torn down, and lets
`comparing-runs.md` check that two runs are actually comparable.

`gpu_count` stays quoted so both shapes are valid JSON: a single number for
a plain recipe, `attention=<n>,ffn=<n>` when the plan spans pods. `nodes`
lists every node of the deployment, not only the one the bench pod used --
that is what a later run compares to tell whether it reused the same
hardware.

### 7. Report this run

`${LOCAL_DIR}` holds one JSON per sweep point
(`rate_<rate>_conc_<conc>.json`). Each is a flat object -- no nesting -- with
the run envelope (`date`, `model_id`, `num_prompts`, `request_rate`,
`max_concurrency`, `label`) merged with the metrics.

**TTFT and TPOT are required in every report**: they separate prefill cost
from decode cost, and an AFD topology can move one without moving the other,
so end-to-end numbers alone hide the effect.

| Report as | Key in the result JSON | Means |
|---|---|---|
| **TTFT** | `mean_ttft_ms`, `median_ttft_ms`, `p90_ttft_ms`, `p99_ttft_ms` | Prefill + queue wait. Rises first under backpressure -- the earliest saturation signal. |
| **TPOT** | `mean_tpot_ms`, `p90_/p99_tpot_ms` | Decode cost per output token, excluding the first. |
| ITL | `mean_itl_ms`, `p90_/p99_itl_ms` | Measured gap between consecutive streamed tokens. Related to TPOT, not identical -- do not substitute one for the other. |
| E2EL | `mean_e2el_ms`, `p90_/p99_e2el_ms` | End-to-end per request. Present only because step 4 passes `e2el` in `--percentile-metrics`. |
| Throughput | `output_throughput`, `total_token_throughput`, `request_throughput` | tokens/s and req/s. |
| Completion | `completed` vs `num_prompts`, plus `failed` | Error/drop rate. |

All latency keys are **milliseconds** (the `_ms` suffix is literal); vLLM
reports throughput in tokens/s. State units once and stay consistent.

```
Run:      <RUN_ID>
Harness:  vllm bench serve (random ISL <n> / OSL <n>, <num_prompts> prompts/point)
Recipe:   <path>   (GPUs: <n or per role>, max-num-batched-tokens: <attention role's>)
Model:    <MODEL_ID>   (nodes: <from run.json>)
Profile:  <name> -- sweep <rate:conc ...>

| Rate | Conc | Output tok/s | TTFT mean | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | Completed |
|------|------|--------------|-----------|----------|-----------|----------|-----------|-----------|
```

Close with the saturation point -- the first sweep point where TTFT departs
from flat (it moves before end-to-end latency) or `completed` drops below
`num_prompts` -- and name any point that failed outright. If nothing
saturated, say so: the sweep measured headroom and the ceiling is above what
was offered.

Before calling a low `completed` a capacity ceiling, check it is not the
deployment falling over: on a split-role plan the attention head pod is a
single replica whose loss does not self-heal (see Scope), and it surfaces as
dropped requests rather than as a crash of the bench pod. This prints
`name<TAB>node<TAB>restarts` per pod:

```bash
kubectl -n "$NAMESPACE" get pods -l "afd-deployment=$DEPLOY_NAME" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\t"}{.status.containerStatuses[0].restartCount}{"\n"}{end}'
```

A restart count that moved during the sweep makes that point an outage, not
a measurement -- rerun it and say so.

### 8. Comparing several runs

See [references/comparing-runs.md](references/comparing-runs.md) --
confirming two runs are comparable and reporting per-point deltas. Both
recipes being compared (e.g. an AFD recipe and its non-AFD baseline) must be
given explicitly by the user; never derive one from the other.

### 9. Tear down the deployment

`deploy-afd-k8s` leaves the serving StatefulSet(s) and Service(s) running by
default so a *follow-up* run can skip the model download and graph capture,
but this skill's own run is done once the report is produced -- tear the
whole deployment down at the end, every time, without waiting to be asked.
Delete by label: a multi-pod plan has no `vllm-pod` to name, and the
single-pod case is a StatefulSet `${DEPLOY_NAME}-pod` whose pod the
StatefulSet would recreate anyway:

```bash
kubectl -n "$NAMESPACE" delete pod vllm-bench --ignore-not-found   # already deleted in step 5 on the happy path
kubectl -n "$NAMESPACE" delete sts -l "afd-deployment=$DEPLOY_NAME"
kubectl -n "$NAMESPACE" delete svc,configmap -l "afd-deployment=$DEPLOY_NAME"
kubectl -n "$NAMESPACE" get pods -l "afd-deployment=$DEPLOY_NAME"   # must print nothing
```

That removes every role's StatefulSet and pods together -- deleting one role
on its own strands the peers' collectives. Keep `vllm-bench-reports` and the
model PVC (`PVC_NAME`, deliberately unlabeled) until every run of interest
has been copied out.
