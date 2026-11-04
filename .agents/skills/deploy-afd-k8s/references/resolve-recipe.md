# 1. Resolve the recipe, its model, and its placement

Take the recipe path from the user, or ask. It must be a colocation recipe
under `recipe/gpu/P2pNcclAFDConnector/**/prefill_decode_colocation/` (GPU
only; prefill/decode disaggregation recipes are out of scope -- see
`SKILL.md` Scope).

Read the script and note, per `vllm serve` block (each backgrounded process
ending `> <name>.log 2>&1 &`):

- `CUDA_VISIBLE_DEVICES`, `--data-parallel-size`, `--tensor-parallel-size`
- the `"afd": {"role": ...}` block in `--additional-config`
- eager vs graph, and every other flag (`--enable-expert-parallel`,
  `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`,
  `--trust-remote-code`, host/port)

Record `--max-num-batched-tokens` and report it to the caller later -- it
caps how much prefill work a single step can absorb.

Resolve `MODEL_ID`: read the script's `MODEL_PATH` default. It is either a
HF Hub id (map the recipe's model directory name to one, e.g.
`deepseek_v2_lite` -> `deepseek-ai/DeepSeek-V2-Lite`; ask if the mapping
isn't obvious) or a **placeholder** path of the form
`/path/model_weights/<name>` -- every colocation recipe uses exactly this
literal `/path/` stand-in today, regardless of what's actually staged on any
given cluster's PVC. Never pass this through verbatim: it names only the
model directory (`<name>`), not a real in-container path. Derive the real
`MODEL_ID` as `<MOUNT_PATH>/<name>`, where `<MOUNT_PATH>` is the PVC mount
point fixed by `templates/statefulset.yaml` (`/models`) -- e.g.
`/path/model_weights/Qwen3.5-122B-A10B-FP8` resolves to
`/models/Qwen3.5-122B-A10B-FP8`. Before deploying, confirm that directory
actually exists on the target `PVC_NAME` (if it already exists, `kubectl
exec` into any pod already mounting it, e.g. `ls /models`, or spin up a
short-lived busybox pod mounting the PVC) -- don't rely on the recipe's
claim; a mismatch here fails fast inside the pod with a confusing
HF-repo-id validation error instead of a clear message.

Tell the caller which form it is, and the resolved value if derived.

Set `CLIENT_PORT` to the `--port` value on the attention (or sole, for a
single-pod recipe) `vllm serve` block -- `templates/statefulset.yaml` rebinds this
port off loopback and `${DEPLOY_NAME}-service` (step 4d, see
[services.md](services.md)) exposes it, whatever value is actually there.

## Read the placement plan -- never ask

Look at the top-of-file comment block (above the first `vllm serve`/env-var
line). If it says nothing about per-pod placement, this is a plain
single-pod recipe: one pod, both roles run unconditionally (`AFD_ROLE`
is `both`, `POD` is `0`) -- continue to [deploy.md](deploy.md) with no
plan (`PLAN_IS_PLAIN=1`).

If it does describe a placement across multiple pods, read and interpret
that description directly. Today's recipes describe per-role pod counts:

```
# on 3 pods: two attention pods and one FFN pod. [...] Attention shards
# with POD>0 run headless with a data-parallel start rank of POD*2.
```

Whatever the phrasing, extract the only two facts the deployment needs:

- **per-role pod counts** (how many attention pods, how many FFN pods)
- whether either role is split across more than one pod (that alone
  decides the optional DP-coordination Services)

and expose the plan to later steps as shell variables (deploy.md step 4d
gates the optional Services on them): `PLAN_IS_PLAIN=1` for a single-pod
recipe; otherwise `TOTAL_PODS` (sum of the per-role pod counts) and
`ATTN_SPLIT=1` iff the attention role itself spans more than one pod.

For each entry, derive (used by [deploy.md](deploy.md) and
[services.md](services.md)):

- **k8s StatefulSet name**: `${DEPLOY_NAME}-pod-<role>` (e.g. with the
default `DEPLOY_NAME=vllm`, the attention StatefulSet is
`vllm-pod-attention`), with `replicas` = that role's pod count.
`DEPLOY_NAME` (default `vllm`) prefixes every resource the deployment
creates -- StatefulSets, Services, ConfigMap -- and labels every object
it creates with `afd-deployment: <DEPLOY_NAME>`, so two deployments with
different names coexist in one namespace without colliding (and the whole
deployment can be selected, reported, and torn down as a unit with `-l
afd-deployment=<DEPLOY_NAME>`). Kubernetes appends the pod
ordinal itself, so pods are `<statefulset-name>-<ordinal>` (e.g.
`vllm-pod-attention-0`, `vllm-pod-attention-1`) -- the ordinal suffix is
a StatefulSet invariant and cannot be removed.
- **POD**: the pod's zero-based index within its role's StatefulSet,
self-derived by the wrapper from the pod's hostname -- never injected as
k8s env. Recipes gate their blocks on `AFD_ROLE` (attention/ffn) and
derive per-index conventions themselves (e.g. headless attention shards
at `POD>0`, DP start ranks from `POD`), so there are no per-pod env
overrides to carry from the plan.
