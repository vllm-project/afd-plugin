# DeepEPM2NAFDConnector Recipes

Launch scripts for running Attention-FFN Disaggregation (AFD) with
`DeepEPM2NAFDConnector`, the expert-routed M2N connector for Hopper (SM90) and
newer GPUs. See
[`docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md`](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md)
for installation, the configuration contract, the required vLLM settings, and
supported layouts.

## Directory layout

```text
.
├── deepseek_v2_lite/
│   ├── README.md
│   └── prefill_decode_colocation/   # 2A2F: eager, graph, graph + DBO
├── qwen3_30b_a3b/
│   ├── README.md
│   └── prefill_decode_colocation/   # 2A2F: eager, graph, graph + DBO, Attention TP2
└── qwen3_235b_a22b_fp8/
    ├── README.md
    └── prefill_decode_colocation/   # 4A2F FP8 payload: graph, graph + DBO
```

Each model directory holds its own scripts; open its `README.md` before
running anything in it.

## Prerequisites

- SM90 or newer GPUs on one node: 4 for the DeepSeek-V2-Lite and
  Qwen3-30B-A3B recipes, 6 for the Qwen3-235B-A22B-FP8 recipes.
- vLLM `v0.32.0`, the plugin, afdEP-comm (the `afdep` extra) and DeepGEMM
  installed in one environment, with the elfutils headers available to
  DeepGEMM's JIT. See
  [Installation](../../../docs/gpu/DEEPEP_M2N_CONNECTOR_USER_GUIDE.md#installation).
- When afdEP-comm was built against the `nvidia-nccl` wheel, export its
  location before running a script:

  ```bash
  NCCL=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
  export AFDEP_NCCL_ROOT=$NCCL LD_LIBRARY_PATH=$NCCL/lib:${LD_LIBRARY_PATH:-}
  ```

The scripts call `vllm serve` directly rather than through `uv run`, so they
use the environment that holds afdEP-comm and DeepGEMM.

## Running on a local host

Run any script from the repository root, for example:

```bash
export MODEL_PATH=/path/model_weights/Qwen3-30B-A3B
bash recipe/gpu/DeepEPM2NAFDConnector/qwen3_30b_a3b/prefill_decode_colocation/2a2f_graph_dp2tp1.sh
```

Each script starts the FFN server and then the Attention server in the
background and writes `ffn.log` and `attn.log` into the current directory.
Wait for `Application startup complete` in both logs, then send traffic to the
Attention server on port `18000`. The AFD rendezvous uses port `6239`; the FFN
server listens on `18001` but serves no requests.

The first start on a node compiles transport and DeepGEMM kernels, so it is
slower than later starts.

## Deploying on Kubernetes

The recipes have no placement header, so the
**`deploy-afd-k8s`** skill (`.agents/skills/deploy-afd-k8s/SKILL.md`) deploys
each one as a single pod. The image must contain afdEP-comm and DeepGEMM.
