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
point fixed by `templates/pod.yaml` (`/models`) -- e.g.
`/path/model_weights/Qwen3.5-122B-A10B-FP8` resolves to
`/models/Qwen3.5-122B-A10B-FP8`. Before deploying, confirm that directory
actually exists on the target `PVC_NAME` (if it already exists, `kubectl
exec` into any pod already mounting it, e.g. `ls /models`, or spin up a
short-lived busybox pod mounting the PVC) -- don't rely on the recipe's
claim; a mismatch here fails fast inside the pod with a confusing
HF-repo-id validation error instead of a clear message.

Tell the caller which form it is, and the resolved value if derived.

Set `CLIENT_PORT` to the `--port` value on the attention (or sole, for a
single-pod recipe) `vllm serve` block -- `templates/pod.yaml` rebinds this
port off loopback and `vllm-service` (step 4d, see
[services.md](services.md)) exposes it, whatever value is actually there.

## Read the placement plan -- never ask

Look at the top-of-file comment block (above the first `vllm serve`/env-var
line). If it says nothing about per-pod placement, this is a plain
single-pod recipe: one pod, `POD` unset, both roles run unconditionally --
continue to [deploy.md](deploy.md) with no plan.

If it does describe a placement across multiple pods, read and interpret
that description directly.

Example of a placement description:
```
# For pod 1: POD=ATTENTION_0
# For pod 2: POD=ATTENTION_1, ATTENTION_HEADLESS=1, ATTENTION_DP_START_RANK=2
# For pod 3: POD=FFN_0
```

Or
```
"runs
on 3 pods: the first two run ATTENTION_0 and ATTENTION_1 -- the second one
headless, starting at data-parallel rank 2 -- and the third runs FFN_0"
```

Whatever the phrasing, read it and extract, per pod, in the order the
comment presents them:

- that pod's `POD=<value>` identity
- any further container-env overrides for that pod only (e.g.
  `ATTENTION_HEADLESS=1`, `ATTENTION_DP_START_RANK=2`)

The recipe already states ranks/ordering per pod -- there is nothing to
derive or ask about, regardless of how it's phrased. (A quick
`grep -E '^# For pod [0-9]+: '` is a fine way to spot today's convention at
a glance, but its absence doesn't mean "no plan" -- check the comment block
itself before concluding this is a plain single-pod recipe.)

For each entry, derive (used by [deploy.md](deploy.md) and
[services.md](services.md)):

- **k8s Pod name**: `vllm-pod-$(echo "$POD" | tr 'A-Z_' 'a-z-')`, e.g.
  `vllm-pod-attention-0`.
- **`afd-attn-node-role`**: `none` unless `POD` starts with `ATTENTION_`, in
  which case `worker` if that line's `ATTENTION_HEADLESS` override is `1`,
  else `head`.
- **`afd-ffn-node-role`**: symmetric, `none` unless `POD` starts with `FFN_`.
