# 1. Resolve the recipe, model, and pod resources

Take the recipe path from the user, or ask. It must be a colocation recipe
under `recipe/gpu/P2pNcclAFDConnector/**/prefill_decode_colocation/` or
`recipe/gpu/DeepEPM2NAFDConnector/**/prefill_decode_colocation/` (GPU
only; prefill/decode disaggregation recipes are out of scope -- see
`SKILL.md` Scope).

Every such recipe states its `MODEL_PATH`, default `LAYOUT`, mode, and serve
flags, then sources the shared launcher,
`recipe/gpu/common/afd_colocation.sh`. Treat the launcher as the execution
contract: do not reconstruct or parse its `vllm serve` commands to determine
Kubernetes resources.

## Model

Resolve `MODEL_ID` from the caller's `MODEL_PATH` if given, else the recipe's
default. It is either a HF Hub id or the placeholder
`/path/model_weights/<name>` used by every colocation recipe.
Never pass this through verbatim: derive the real `MODEL_ID` as
`/models/<name>` (`/models` is the PVC mount point fixed by
`templates/pod.yaml`), e.g. `/path/model_weights/Qwen3.5-122B-A10B-FP8`
resolves to `/models/Qwen3.5-122B-A10B-FP8`. Before deploying, confirm that
directory exists on the target `PVC_NAME` (`kubectl exec` into any pod
already mounting it, e.g. `ls /models`, or a short-lived busybox pod
mounting the PVC); a mismatch fails inside the pod with a confusing
HF-repo-id validation error. Tell the caller which form it is, and the
resolved value if derived.

## Pod resource plan -- never ask

Take `LAYOUT` from the caller's settings, else the recipe's default; never
ask. The selected `LAYOUT`, not a recipe filename or descriptive comment,
controls the plan. Split it by comma into one `<a>A<f>F` entry per pod. For
entry `k`, `a` is that pod's Attention GPU count and `f` its FFN GPU count:

- **k8s Pod name**: `vllm-pod-afd-<k>`. A one-entry `LAYOUT` (e.g. `2A2F`)
  is a plain single pod: `vllm-pod`, no `POD` needed.
- **`GPU_COUNT`**: `a + f`, passed to `templates/pod.yaml` as `TEMPLATE_GPU`
  so the container requests that many `nvidia.com/gpu` devices.
- **`afd-attn-node-role`**: `none` when `a` is 0, `head` on the first entry
  with `a > 0`, `worker` on later ones. **`afd-ffn-node-role`**: the same
  with `f`. For example, `2A0F,0A2F` gives `head none` then `none head`.
- **Container env**: every pod gets the selected `LAYOUT`, its own
  `POD=AFD_<k>`, and the caller's other recipe overrides unchanged. Leave
  `HEAD_HOST`, `ATTN_HEAD_HOST`, and `FFN_HEAD_HOST` unset; their defaults
  name the Services in [services.md](services.md). With `DECODE_BENCH`, also
  set `AFD_REPO=/opt/afd-plugin`.

The launcher owns role-to-device ordering, DP/TP command construction, and
startup validation of the placement. Record the pod/resource table for
[deploy.md](deploy.md), which uses it as-is.

## Launcher ports

Read the client and internal Service ports from the launcher's fixed assignments:

```bash
RECIPE_LIB_PATH=recipe/gpu/common/afd_colocation.sh
lib_port() {
    grep -oE "\b$1=[0-9]+" "$RECIPE_LIB_PATH" | head -1 | cut -d= -f2
}

CLIENT_PORT="$(lib_port ATTN_PORT)"
AFD_CONNECTOR_PORT="$(lib_port AFD_PORT)"
ATTN_DP_RPC_PORT="$(lib_port ATTN_DP_RPC_PORT)"
FFN_DP_RPC_PORT="$(lib_port FFN_DP_RPC_PORT)"
```

## Other deployment values

Record the effective `--max-num-batched-tokens` -- the caller's `VLLM_ARGS`
value if overridden, else the recipe's `SERVE_ARGS` value -- for the final
report.
