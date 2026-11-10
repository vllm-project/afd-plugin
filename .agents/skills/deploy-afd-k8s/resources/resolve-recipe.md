# 1. Resolve the recipe, its model, and its placement

Take the recipe path from the user, or ask. It must be a colocation recipe
under `recipe/gpu/P2pNcclAFDConnector/**/prefill_decode_colocation/` or
`recipe/gpu/DeepEPM2NAFDConnector/**/prefill_decode_colocation/` (GPU
only; prefill/decode disaggregation recipes are out of scope -- see
`SKILL.md` Scope).

Every such recipe is a short script on the shared launcher
`recipe/gpu/common/afd_colocation.sh` (see `recipe/gpu/common/README.md`):
the recipe sets `MODEL_PATH`, `AFD_CONNECTOR`, `LAYOUT`, `MODE` and its
`SERVE_ARGS`; the launcher derives devices, DP flags and ports. Never parse
the `vllm serve` command out of the script -- print it:

```bash
RECIPE_LIB_PATH=recipe/gpu/common/afd_colocation.sh
AFD_DRY_RUN=1 LAYOUT=<layout> POD=AFD_<k> <caller settings> bash "$RECIPE_SCRIPT_PATH"
```

prints, per role this pod runs, the `CUDA_VISIBLE_DEVICES=... vllm serve ...`
line. Record `--max-num-batched-tokens` and report it to the caller later --
it caps how much prefill work a single step can absorb.

## Model

Resolve `MODEL_ID` from the script's `MODEL_PATH` default. It is either a
HF Hub id or a **placeholder** path of the form `/path/model_weights/<name>`
-- every colocation recipe uses exactly this literal `/path/` stand-in.
Never pass this through verbatim: derive the real `MODEL_ID` as
`/models/<name>` (`/models` is the PVC mount point fixed by
`templates/pod.yaml`), e.g. `/path/model_weights/Qwen3.5-122B-A10B-FP8`
resolves to `/models/Qwen3.5-122B-A10B-FP8`. Before deploying, confirm that
directory exists on the target `PVC_NAME` (`kubectl exec` into any pod
already mounting it, e.g. `ls /models`, or a short-lived busybox pod
mounting the PVC); a mismatch fails inside the pod with a confusing
HF-repo-id validation error. Tell the caller which form it is, and the
resolved value if derived.

## Placement plan -- never ask

The placement is `LAYOUT`: one pod per comma-separated `<a>A<f>F` entry, in
order, pod `k` running `POD=AFD_<k>`, every pod with the same `LAYOUT`. Take
`LAYOUT` from the caller's settings, else from the script's
`LAYOUT=${LAYOUT:-...}` default; never ask. For entry `k` with `a`
Attention and `f` FFN GPUs:

- **k8s Pod name**: `vllm-pod-afd-<k>`. A one-entry `LAYOUT` (e.g. `2A2F`)
  is a plain single pod: `vllm-pod`, no `POD` needed.
- **`GPU_COUNT`**: `a + f` (FFN on the pod's first `f` GPUs).
- **`afd-attn-node-role`**: `none` when `a` is 0, `head` on the first entry
  with `a > 0`, `worker` on later ones. **`afd-ffn-node-role`**: the same
  with `f`. For example `2A0F,0A2F` is `head none` and `none head`,
  `1A1F,1A1F` is `head head` and `worker worker`, and `2A0F,2A0F,0A2F` is
  `head none`, `worker none` and `none head`.
- **Container env**: `POD=AFD_<k>` and `LAYOUT=<layout>`, plus every setting
  the caller gives (`MODEL_PATH`, `ATTN_TP`, `FFN_TP`, `VLLM_ARGS`,
  `ATTN_ARGS`, `FFN_ARGS`, `DECODE_BENCH`, `M2N_EXTRA_CONFIG`, NCCL
  variables), the same on every pod. Leave `HEAD_HOST`, `ATTN_HEAD_HOST` and
  `FFN_HEAD_HOST` unset: the defaults name the Services from
  [services.md](services.md). With `DECODE_BENCH`, also set
  `AFD_REPO=/opt/afd-plugin` (the repository in `docker/Dockerfile.ci`
  images), since the connector lives in `tools/benchmarks` and the recipe
  runs from `/work`.
- Whether the connector accepts the placement is checked at startup, not
  here; see the connector guide's supported layouts.

## Ports

The launcher fixes them on one line (`ATTN_PORT=18000 FFN_PORT=18001
AFD_PORT=6239 ATTN_DP_RPC_PORT=13345 FFN_DP_RPC_PORT=13355`); read them from
it rather than assuming:

```bash
lib_port() { grep -oE "\b$1=[0-9]+" "$RECIPE_LIB_PATH" | head -1 | cut -d= -f2; }
CLIENT_PORT=$(lib_port ATTN_PORT)   # the Attention API, exposed by vllm-service
```

This step is the only place that parses the recipe:
[deploy.md](deploy.md) uses the `GPU_COUNT` and plan recorded here as-is.
