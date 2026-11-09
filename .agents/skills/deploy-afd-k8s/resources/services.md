# Services

Create as step 4d of [deploy.md](deploy.md) -- **before** deploying any Pod
(step 4e), not after (idempotent -- safe to re-apply either way). The
headless Services below are what a role's head Pod *binds* its own
rendezvous/DP-RPC server to during its own startup, within seconds of the
container starting; if that Service doesn't exist yet, the bind/resolve
fails fast and, since every pod has `restartPolicy: Never`, it never gets a
second chance. Which of the four to create is derived from the plan parsed
in [resolve-recipe.md](resolve-recipe.md), never asked.

## `vllm-service` -- always

Client-facing. Selects whichever pod holds `afd-attn-node-role: head` --
true for the sole pod of a plain recipe, and for whichever pod is the
attention head of a placement plan.

```bash
CLIENT_PORT="$CLIENT_PORT" envsubst '${CLIENT_PORT}' \
  < templates/service-client.yaml | kubectl apply -f -
```

## `vllm-ffn-p2p-service` -- iff the plan has more than one pod

Internal only, carries AFD rendezvous traffic to whichever pod holds
`afd-ffn-node-role: head`. Must exist before any multi-pod deploy, since the
recipe's own `AFD_CONNECTOR_HOST` default now points at this name directly.
Traffic to a headless Service goes straight to the pod IP, and there's one
rendezvous port, so a single static port entry is enough -- read
`AFD_CONNECTOR_PORT` off the recipe's own shell default:

```bash
AFD_CONNECTOR_PORT="$(grep -oE 'AFD_CONNECTOR_PORT:-[0-9]+' "$RECIPE_SCRIPT_PATH" | grep -oE '[0-9]+')"
AFD_CONNECTOR_PORT="$AFD_CONNECTOR_PORT" envsubst '${AFD_CONNECTOR_PORT}' \
  < templates/service-ffn-p2p.yaml | kubectl apply -f -
```

## `vllm-attn-dp-service` / `vllm-ffn-dp-service` -- iff that role is split

Only when **more than one** plan entry's `POD` carries the same role (i.e.
that role is split across pods, not just placed in its own single pod).
Carries that role's DP-RPC coordination traffic to its head pod. One per
split role, generic over role via `templates/service-dp-coord.yaml`:

```bash
TEMPLATE_ROLE=attn
TEMPLATE_PORT="$(grep -oE 'ATTENTION_DP_RPC_PORT:-[0-9]+' "$RECIPE_SCRIPT_PATH" | grep -oE '[0-9]+')"
TEMPLATE_ROLE="$TEMPLATE_ROLE" TEMPLATE_PORT="$TEMPLATE_PORT" \
  envsubst '${TEMPLATE_ROLE} ${TEMPLATE_PORT}' < templates/service-dp-coord.yaml | kubectl apply -f -
```

All three internal Services (`vllm-ffn-p2p-service`, `vllm-attn-dp-service`,
`vllm-ffn-dp-service`) are headless with `publishNotReadyAddresses: true` --
see the comments in their own templates for why.
