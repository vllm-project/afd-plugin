# Services

Create as step 4d of [deploy.md](deploy.md) -- **before** deploying any
StatefulSet (step 4e), not after (idempotent -- safe to re-apply either
way). Recipes resolve these names within seconds of the container starting
(the FFN head binds its rendezvous server to the ffn-p2p name); if a name
does not resolve at that moment, startup fails fast and burns a restart.
Pods do restart under the template's `restartPolicy: Always`, but a
restart costs minutes of model load and graph capture, so creating the
Services first is the cheap and correct order.

There are two kinds:

## `${DEPLOY_NAME}-recipe-sts` -- governing, always

Headless identity Service every StatefulSet's `serviceName` points at.
`publishNotReadyAddresses: true`, selector `app=afd-recipe,
afd-deployment=<DEPLOY_NAME>`, no ports. It exists for two reasons:
StatefulSets require a `serviceName`, and it gives every pod a stable DNS
name (`<pod>.<governing>.<namespace>.svc.<cluster-domain>`) that resolves
even before the pod is ready -- which the ExternalName endpoints below
point at.

```bash
TEMPLATE_DEPLOY_NAME="$DEPLOY_NAME" envsubst '${TEMPLATE_DEPLOY_NAME}' \
  < templates/service-governing.yaml | kubectl apply -n "$NAMESPACE" -f -
```

## ExternalName endpoints -- one per entry point

Built from the single shared `templates/service-endpoint.yaml`; each is a
named CNAME at one specific head pod's DNS name (the role's StatefulSet
ordinal-0 pod). Selector-based Services cannot express "the head pod":
every pod of a role shares one StatefulSet template, so head-ness is not a
label -- it is the ordinal. The head pod's DNS is stable across pod
recreation and tracks its current IP automatically.

| Service | When | Points at |
|---|---|---|
| `${DEPLOY_NAME}-service` | always | attention head (or the sole pod of a plain recipe) |
| `${DEPLOY_NAME}-ffn-p2p-service` | plan has >1 pod | FFN head (owns the rendezvous TCPStore) |
| `${DEPLOY_NAME}-attn-dp-service` | attention split across >1 pod | attention head |
| `${DEPLOY_NAME}-ffn-dp-service` | FFN split across >1 pod | FFN head |

Which of the optional ones applies comes from the plan parsed in
[resolve-recipe.md](resolve-recipe.md) (`BOTH_ROLES`, `TOTAL_PODS`,
`ATTN_SPLIT`, `FFN_SPLIT`), never asked -- and deploy.md gates each
`endpoint` call in code accordingly. A both-roles plan (a plain recipe or a
`POD=AFD_<index>` one) has a single `${DEPLOY_NAME}-pod` StatefulSet, so
every head above is ordinal 0 of that one StatefulSet.

The StatefulSet template provides the deployment's service names as fixed
pod env (`AFD_CONNECTOR_HOST`, `ATTENTION_DP_ADDRESS`); recipes that need
them require them and fail fast when missing, and carry no baked-in
defaults that could only ever be right for one `DEPLOY_NAME`. The DeepEP
M2N `POD=AFD_<index>` recipes are the exception: they default their
`HEAD_HOST` to the fixed `vllm-ffn-p2p-service`, `vllm-attn-dp-service`
and `vllm-ffn-dp-service` names and leave it unset on k8s, so they expect
`DEPLOY_NAME=vllm`. Under any other `DEPLOY_NAME`, inject `HEAD_HOST`
pointing at this deployment's head pod (it serves the rendezvous and both
DP-RPC ports) -- see [resolve-recipe.md](resolve-recipe.md).

`${DEPLOY_NAME}-ffn-dp-service` carries no pod env of its own: it exists
because the `POD=AFD_<index>` recipes point their FFN DP-RPC peers at that
name (the template injects no `FFN_DP_ADDRESS` and no split-role recipe
reads an FFN DP address today).
