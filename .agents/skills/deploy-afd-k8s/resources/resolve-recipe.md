# 1. Resolve the recipe, its model, and its placement

Take the recipe path from the user, or ask. It must be a colocation recipe
under `recipe/gpu/P2pNcclAFDConnector/**/prefill_decode_colocation/` or
`recipe/gpu/DeepEPM2NAFDConnector/**/prefill_decode_colocation/` (GPU
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

## GPU_COUNT -- record it here

Record `GPU_COUNT` now; [deploy.md](deploy.md) step 2 uses it as-is. This is
the only step that parses the recipe for GPU counts -- don't re-derive them
later, and never derive them from ranks or DP sizes.

Placement-driven recipe: record one value **per role** (`ATTN_GPU_COUNT`,
`FFN_GPU_COUNT`), not one per pod -- every pod of a role runs the same
StatefulSet template, so the GPU request is per role. It is the number of
devices in the literal `CUDA_VISIBLE_DEVICES=` of the `vllm serve` block
that role runs, i.e. the one inside the recipe's branch that tests
`$AFD_ROLE` for that role (`attention` or `ffn`). Every pod sharing a role
shares that literal, so read it off the script.

Plain single-pod recipe (no placement in the header -- see the next
section): one pod runs every role block, so `GPU_COUNT` is the union of
every literal `CUDA_VISIBLE_DEVICES=` in the script:

```bash
GPU_COUNT="$(grep -oE 'CUDA_VISIBLE_DEVICES=[0-9,]+' "$RECIPE_SCRIPT_PATH" \
  | cut -d= -f2 | tr ',' '\n' | sort -u | wc -l)"
```

### Pods that run both roles: `POD=AFD_<index>`

DeepEP M2N multi-pod recipes (e.g.
`DeepEPM2NAFDConnector/qwen3_235b_a22b_fp8/prefill_decode_colocation/multipod_4a1f_4a1f_graph.sh`)
describe one line per pod -- `POD=AFD_0`, `POD=AFD_1`, ... -- and every pod
runs one FFN and one Attention server, deriving its own DP flags (headless,
start rank) from its index. This is not a split-role plan: every pod holds
both roles, so it deploys as the single both-roles `${DEPLOY_NAME}-pod`
StatefulSet with `replicas` = the number of entries (see the next section).
The scripts read their index as `${POD#AFD_}`, so the numeric ordinal this
skill's pods self-derive is accepted unchanged -- never inject `POD` as env.

- **`GPU_COUNT`**: `ATTN_PER_POD + FFN_PER_POD`, the same for every pod.
  These recipes build `CUDA_VISIBLE_DEVICES` with `$(seq ...)`, so the
  literal-device rules above and below don't apply:

  ```bash
  default() { grep -oE "$1:-[0-9]+" "$RECIPE_SCRIPT_PATH" | grep -oE '[0-9]+$'; }
  GPU_COUNT=$(( ${ATTN_PER_POD:-$(default ATTN_PER_POD)} + ${FFN_PER_POD:-$(default FFN_PER_POD)} ))
  ```
- **Pod count**: the number of `POD=AFD_<index>` lines, which must equal the
  script's own `NUM_PODS` default. Overriding `NUM_PODS`, `ATTN_PER_POD` or
  `FFN_PER_POD` needs the same values on every pod; the StatefulSet
  template injects no recipe env, so add them to its `env` list before
  applying.
- **Ports and Service names**: ports are literals in these scripts (the
  `"port": N` of the `afd` JSON, and the last argument of `role_dp "$F"
  ...` / `role_dp "$A" ...`), so nothing downstream needs them -- the
  Services are ExternalName pointers, not port mappings. The names come
  from the scripts' fixed `HEAD_HOST` defaults, not from the template's
  `AFD_CONNECTOR_HOST`/`ATTENTION_DP_ADDRESS`, and `HEAD_HOST` is unset on
  k8s -- so these recipes expect `DEPLOY_NAME=vllm`; see
  [services.md](services.md) for the names and the override.

## Read the placement plan -- never ask

Look at the top-of-file comment block (above the first `vllm serve`/env-var
line). If it says nothing about per-pod placement, this is a plain
single-pod recipe: one pod, both roles run unconditionally (`AFD_ROLE`
is `both`, `POD` is `0`) -- continue to [deploy.md](deploy.md) with the
single-pod plan (`BOTH_ROLES=1`, `TOTAL_PODS=1`).

If it does describe a placement across multiple pods, read and interpret
that description directly. Today's recipes describe either per-role pod
counts:

```
# on 3 pods: two attention pods and one FFN pod. [...] Attention shards
# with POD>0 run headless with a data-parallel start rank of POD*2.
```

or one identical both-roles pod per line (`POD=AFD_0`, `POD=AFD_1`, ...;
see the section above).

Whatever the phrasing, extract the only three facts the placement comment
carries:

- **pod count** (one pod per entry)
- **per-role pod counts** (how many attention pods, how many FFN pods)
- whether every pod runs **both** roles (the `POD=AFD_<index>` shape) or one

and expose the plan to later steps as shell variables (deploy.md step 4d
gates the optional Services on them):

- a plain single-pod recipe is `BOTH_ROLES=1`, `TOTAL_PODS=1`
- otherwise `BOTH_ROLES=1` iff every pod runs both roles, `TOTAL_PODS`
  (the pod count), and `ATTN_SPLIT=1` / `FFN_SPLIT=1` iff that role itself
  spans more than one pod -- for a both-roles plan with more than one pod,
  that is both of them.

For each entry, derive (used by [deploy.md](deploy.md) and
[services.md](services.md)):

- **k8s StatefulSet name**: `${DEPLOY_NAME}-pod-<role>` (e.g. with the
default `DEPLOY_NAME=vllm`, the attention StatefulSet is
`vllm-pod-attention`), with `replicas` = that role's pod count. A
both-roles plan has no per-role split to name: it is the single
`${DEPLOY_NAME}-pod` StatefulSet with `replicas` = `TOTAL_PODS`.
`DEPLOY_NAME` (default `vllm`) prefixes every resource and labels every
object `afd-deployment: <DEPLOY_NAME>` (SKILL.md's contract). Kubernetes
appends the pod ordinal itself, so pods are `<statefulset>-<ordinal>`
(e.g. `vllm-pod-attention-0`) -- the suffix cannot be removed.
- **POD**: the pod's zero-based index within its role's StatefulSet,
self-derived by the wrapper from the pod's hostname -- never injected as
k8s env. An `AFD_<index>` plan's pods are `${DEPLOY_NAME}-pod-<ordinal>`,
and those recipes read that ordinal as-is (`K=${POD#AFD_}`). Recipes gate
their blocks on `AFD_ROLE` (attention/ffn) and
derive per-index conventions themselves (e.g. headless attention shards
at `POD>0`, DP start ranks from `POD`), so there are no per-pod env
overrides to carry from the plan.
