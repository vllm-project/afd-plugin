# Shared launcher for the GPU colocation recipes

Every GPU `prefill_decode_colocation` recipe (`DeepEPM2NAFDConnector` and
`P2pNcclAFDConnector`) is a short script that states a model, a connector, a
default layout and its `vllm serve` flags, and then hands over to
[`afd_colocation.sh`](afd_colocation.sh). The launcher owns everything the
recipes share: the placement (`LAYOUT` and `POD` to GPUs and data-parallel
flags), the `afd` additional config, the decode-only benchmark hook, the
override variables, and the two `vllm serve` processes.

## Writing a recipe

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Qwen3-30B-A3B (BF16) on DeepEPM2NAFDConnector, 2A2F graph + DBO.
# Placement: one pod per comma-separated LAYOUT entry (default 2A2F), pod k
# running POD=AFD_<k> with the same LAYOUT; FFN on each pod's first GPUs.
# Settings and running in a pod: recipe/gpu/common/README.md
# shellcheck disable=SC1090,SC2034  # the sourced launcher reads these variables

AFD_CONNECTOR=${AFD_CONNECTOR:-DeepEPM2NAFDConnector}
LAYOUT=${LAYOUT:-2A2F}
MODEL_PATH=${MODEL_PATH:-/path/model_weights/Qwen3-30B-A3B}
ATTN_TP=${ATTN_TP:-1} FFN_TP=${FFN_TP:-1}
MODE=graph CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-8,16,32,64}   # ascending
SERVE_ARGS=(--enable-ep-weight-filter --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096
            --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 8)

source "${AFD_RECIPE_LIB:-$(dirname "${BASH_SOURCE[0]}")/../../../common/afd_colocation.sh}"
afd_colocation_serve "$@"
```

| Recipe variable | Meaning |
| --- | --- |
| `AFD_CONNECTOR` | Connector class. `P2pNcclAFDConnector` routes on the FFN and drops `CONNECTOR_CONFIG`; any other connector computes the gate on Attention. |
| `LAYOUT` | Default placement; see [Layout](#layout). |
| `MODEL_PATH` | Default weights directory or HF id. |
| `MODE` | `graph` (`FULL_DECODE_ONLY` CUDA graphs of `CUDAGRAPH_CAPTURE_SIZES`, the largest being the maximum capture size) or `eager` (`--enforce-eager`). |
| `SERVE_ARGS` | Array of model and tuning flags passed to both roles (DBO, `--enable-ep-weight-filter`, batch limits, dtype, ...). |
| `CONNECTOR_CONFIG` | Optional `connector_extra_config` members, for example `'"payload_format": "fp8_block128_fp32"'`. |

The launcher always adds `--enable-expert-parallel`, `--trust-remote-code`,
`--cpu-distributed-timeout-seconds 3600` and `afd_process_group_timeout_s:
3600` (the roles can finish loading minutes apart). Its ports are fixed:
`18000` Attention API, `18001` FFN (serves no requests), `6239` AFD
rendezvous, `13345` and `13355` the Attention and FFN DP RPC ports across
pods. The default launcher path assumes the recipe sits at
`recipe/gpu/<connector>/<model>/<stage>/`.

## Layout

`LAYOUT` lists the pods in order, one `<a>A<f>F` entry each: `a` Attention
GPUs and `f` FFN GPUs. Inside a pod, the FFN takes the first `f` GPUs and the
Attention the next `a`; every pod runs the same script with the same
`LAYOUT` and its own `POD=AFD_<k>` (`k` counts from 0; one pod needs no
`POD`).

| `LAYOUT` | Pods | Per pod |
| --- | --- | --- |
| `2A2F` | 1 | FFN DP2 on GPUs 0,1, Attention DP2 on GPUs 2,3 |
| `4A2F` | 1 | FFN on GPUs 0,1, Attention DP4 on GPUs 2-5 |
| `2A0F,0A2F` | 2 | pod 0: Attention DP2; pod 1: FFN DP2 |
| `1A1F,1A1F` | 2 | FFN on GPU 0, Attention on GPU 1 |
| `2A0F,2A0F,0A2F` | 3 | pods 0-1: Attention DP4 (pod 1 headless); pod 2: FFN DP2 |

`ATTN_TP` and `FFN_TP` split each pod's GPUs of that role into TP groups (they
must divide each pod's `a` or `f`); `num_attention_ranks` and `num_ffn_ranks`
still count GPUs. A role held by one pod gets only `--data-parallel-size`; a
role spread over pods is led by its first pod, and the others start headless
from their start rank. Each connector checks the placement at startup; see
its user guide for the layouts it accepts.

## Settings

Every setting is an environment variable; the defaults reproduce the recipe.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | recipe | Weights directory or HF id. |
| `LAYOUT` | recipe | Pods and their GPUs; see [Layout](#layout). |
| `POD` | `AFD_0` | This pod's entry in `LAYOUT`, `AFD_<k>`. |
| `ATTN_TP` / `FFN_TP` | `1` | TP size of the Attention / FFN role. |
| `CUDAGRAPH_CAPTURE_SIZES` | recipe | Ascending decode graph sizes (graph recipes). |
| `AFD_CONNECTOR` | recipe | Connector class, for example `P2pNcclAFDConnector` to run an M2N recipe on NCCL P2P. |
| `VLLM_ARGS` | empty | Extra `vllm serve` flags for both roles. |
| `ATTN_ARGS` / `FFN_ARGS` | empty | Extra `vllm serve` flags for one role. |
| `M2N_EXTRA_CONFIG` | empty | Extra `connector_extra_config` members, for example `'"expert_num_sms": 130'`. |
| `DECODE_BENCH` | empty | `kv_connector_extra_config` JSON of the decode-only benchmark connector on Attention (`{}` is enough); see [`tools/benchmarks`](../../../tools/benchmarks/). |
| `AFD_REPO` | current directory | Repository root that holds `tools/benchmarks` (with `DECODE_BENCH`). |
| `HEAD_HOST` | `127.0.0.1` on one pod | Address of pod 0 when it leads both roles (several pods, no Kubernetes). |
| `ATTN_HEAD_HOST` / `FFN_HEAD_HOST` | `HEAD_HOST`, else the Kubernetes Service names | Address of the first pod with Attention / FFN GPUs. The FFN head also hosts the AFD rendezvous. |
| `SERVE_HOST` | `127.0.0.1` | API server bind address (`0.0.0.0` to serve outside the pod). |
| `AFD_RECIPE_LIB` | next to the recipe in the repository | Path of `afd_colocation.sh`, for a recipe copied out of the repository. |
| `AFD_DRY_RUN` | empty | Print this pod's `vllm serve` commands instead of running them. |

`VLLM_ARGS`, `ATTN_ARGS`, `FFN_ARGS` and any arguments after the script name
come after the recipe's flags, so a repeated flag overrides its default. They
are split on whitespace without shell quoting, so pass JSON without spaces:

```bash
VLLM_ARGS="--max-num-seqs 512" ATTN_ARGS="--no-enable-prefix-caching" \
    bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```

## Running directly

From any directory (logs go to the current directory):

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```

The script starts the FFN and Attention servers in the background, writes
`ffn.log` and `attn.log`, and waits. Send traffic to port `18000` once
`attn.log` prints `Application startup complete`. To see what a pod would
run without starting anything:

```bash
AFD_DRY_RUN=1 LAYOUT=2A0F,0A2F POD=AFD_1 bash .../2a2f_graph_dp2tp1.sh
```

Several hosts without Kubernetes run the same script with their own `POD`
and the leaders' addresses:

```bash
LAYOUT=2A0F,0A2F POD=AFD_0 ATTN_HEAD_HOST=<pod 0> FFN_HEAD_HOST=<pod 1> bash .../2a2f_graph_dp2tp1.sh
LAYOUT=2A0F,0A2F POD=AFD_1 ATTN_HEAD_HOST=<pod 0> FFN_HEAD_HOST=<pod 1> bash .../2a2f_graph_dp2tp1.sh
```

## Running in a pod

A recipe needs `afd_colocation.sh` next to it. Ship both files and point
`AFD_RECIPE_LIB` at the launcher:

- **`deploy-afd-k8s` skill** (`.agents/skills/deploy-afd-k8s/`): puts both
  files into the `afd-recipe-script` ConfigMap (`recipe.sh` and
  `afd_colocation.sh`), mounts it at `/recipe`, and sets
  `AFD_RECIPE_LIB=/recipe/afd_colocation.sh` and `SERVE_HOST=0.0.0.0`. It
  creates one pod per `LAYOUT` entry and the Services that the default head
  host names resolve to. Edited recipes need no image rebuild.
- **A pod you manage**: copy both files and run the recipe with the launcher
  path:

  ```bash
  kubectl cp recipe/gpu/common/afd_colocation.sh <pod>:/work/afd_colocation.sh
  kubectl cp <recipe>.sh <pod>:/work/recipe.sh
  kubectl exec <pod> -- bash -c 'cd /work && SERVE_HOST=0.0.0.0 \
      AFD_RECIPE_LIB=/work/afd_colocation.sh MODEL_PATH=/models/<model> bash recipe.sh'
  ```

- **An image with this repository** (`docker/Dockerfile.ci` puts it in
  `/opt/afd-plugin`): run the recipe in place, as the image's copy finds its
  launcher: `bash /opt/afd-plugin/recipe/gpu/<connector>/<model>/prefill_decode_colocation/<recipe>.sh`.
