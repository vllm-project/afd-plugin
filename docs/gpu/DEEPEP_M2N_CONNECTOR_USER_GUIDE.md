## Background

`DeepEPM2NAFDConnector` is a CUDA connector that routes tokens on the Attention
side and sends each token only to the FFN ranks that own its selected experts
(an M:N, expert-routed exchange). Dispatch and combine run on
[afdEP-comm](https://github.com/afdEP/afdEP-comm)'s DeepEP-derived
`ElasticBuffer`; the FFN runs its local experts with
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) grouped GEMMs by default.
This guide documents installation, the configuration contract, the required
vLLM settings, supported layouts, and operational limits.

## When to use `DeepEPM2NAFDConnector`

Use this connector for CUDA decode deployments of a supported MoE model on
Hopper (SM90) or newer GPUs, when Attention and FFN run as separate vLLM
processes. Unlike `P2pNcclAFDConnector`, which moves every token's hidden
state to one FFN rank, this connector computes the gate and top-k on Attention
and dispatches only the routed copies of each token to the FFN ranks that hold
those experts. FFN ranks run only routed experts and send the weighted rows
back; Attention sums each token's top-k contributions in combine.

Supported models:

- Qwen3 MoE (`Qwen3MoeForCausalLM`) whose layers are all MoE and which has no
  shared expert.
- DeepSeek-V2 (`DeepseekV2ForCausalLM`, validated with DeepSeek-V2-Lite).
  Attention runs the dense layers and the shared experts; the shared experts
  overlap the dispatch and the FFN. With Attention TP > 1 each TP rank runs
  them, with replicated weights, on the token slice it dispatches.

DeepSeek-V3, GLM-DSA and Qwen3.5/Qwen3.6 MoE are rejected with this connector.

## How it works

Let `A = num_attention_ranks` and `F = num_ffn_ranks`.

### AFD world

The connector joins one AFD world of size `F + A` ordered FFN-first, as the
P2P connector does:

```text
world rank:   0    1   ...  F-1   F    F+1  ...  F+A-1
member:       F0   F1  ...  F_    A0   A1   ...  A_
```

DeepEP sees this world as one symmetric expert-parallel group. FFN rank `g`
owns real experts `[g * E/F, (g + 1) * E/F)`, so `num_experts` must be
divisible by `F`; Attention ranks own dummy expert slots that routing never
selects. Any Attention:FFN ratio works.

### Data path

For every MoE layer and stage, in the same order on every rank:

- Attention computes the gate and top-k, then `send_attn_output()` dispatches
  the routed rows; `recv_ffn_output()` combines the returned rows.
- FFN `recv_attn_output()` receives the rows for its local experts,
  `run_experts()` runs them, and `send_ffn_output()` sends the weighted rows
  back.

Dispatch and combine are collectives over the whole AFD world: every rank
takes part in every MoE layer and stage, including ranks with no tokens. An
FFN rank can therefore receive zero rows for a layer.

### Control plane

Attention rank 0 sends each step's `AFDControlPayload` to every FFN rank.
Every rank derives the same per-stage plan from it (dispatch bucket, host
sync, graph mode), so there is no per-layer control traffic.

### Startup handshake

Before allocating transport buffers, all ranks exchange a handshake and fail
together if they disagree. The handshake checks the payload format, model
shapes, lane count (DBO), `network`, bucket settings, GPU
architecture (SM90 or newer), and the FFN deployment (expert parallel, EP
size, FFN TP=1, linear expert placement, no EPLB, all2all backend and expert
weight format). The FFN adopts the Attention TP size, dispatch capacity and
bucket ladder reported by the Attention ranks.

## Installation

The connector needs three components in addition to the plugin and vLLM:

1. **afdEP-comm**, installed by the plugin's `afdep` extra (the `afdep-comm`
   package, pinned in `pyproject.toml`). Uninstall an older `afdep` package
   first; both provide the `afdep` module.

   ```bash
   python -m pip uninstall -y afdep
   python -m pip install --no-deps \
     "afdep-comm @ git+https://github.com/afdEP/afdEP-comm.git@<commit pinned in pyproject.toml>"
   python -m afdep doctor --build
   ```

   afdEP-comm builds against NCCL with device headers. When NCCL comes from the
   `nvidia-nccl` wheel, point `AFDEP_NCCL_ROOT` at it and add its `lib`
   directory to `LD_LIBRARY_PATH` on both roles:

   ```bash
   NCCL=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
   export AFDEP_NCCL_ROOT=$NCCL LD_LIBRARY_PATH=$NCCL/lib:${LD_LIBRARY_PATH:-}
   ```

2. **DeepGEMM**, required by the default `deepgemm` expert backend. It is not a
   plugin extra (its build imports `torch` and needs git submodules), so build
   it from source:

   ```bash
   # DeepGEMM's JIT includes elfutils/libdwfl.h.
   apt-get install -y libdw-dev
   git clone https://github.com/deepseek-ai/DeepGEMM.git
   cd DeepGEMM
   git submodule update --init --recursive
   python setup.py bdist_wheel
   # --no-deps: DeepGEMM's own install path reinstalls torch.
   python -m pip install --no-deps dist/*.whl
   ```

   - The elfutils headers (`libdw-dev`, and `libelf-dev` it depends on) are
     needed when building DeepGEMM and at runtime, because DeepGEMM compiles
     its kernels just in time. Without root, extract the `libdw-dev` and
     `libelf-dev` packages that match the installed elfutils runtime with
     `dpkg -x`, and point `CPLUS_INCLUDE_PATH`, `C_INCLUDE_PATH` and
     `LIBRARY_PATH` at them for both the build and the served processes.
   - When torch comes from the pip CUDA wheels, its headers include
     `cusparse.h` and other CUDA headers shipped by the `nvidia-*` wheels.
     Add that wheel `include` directory to `CPLUS_INCLUDE_PATH` for the build.
   - The `fused_experts` expert backend (vLLM's Triton kernels) runs without
     DeepGEMM; it is a reference backend.

3. A GPU with compute capability 9.0 or newer on every rank.

## Configuration

AFD configuration is supplied through vLLM's `--additional-config` under the
`afd` key. Attention and FFN processes use the same values except `role`.

```jsonc
{
  "afd": {
    "role": "attention",
    "connector": "DeepEPM2NAFDConnector",
    "host": "127.0.0.1",
    "port": 6239,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 2,
    "compute_gate_on_attention": true,
    "connector_extra_config": {
      "payload_format": "bf16"
    }
  }
}
```

### Common fields

| Field | Required value or meaning |
| --- | --- |
| `role` | `"attention"` or `"ffn"`. |
| `connector` | `"DeepEPM2NAFDConnector"`. |
| `host`, `port` | AFD rendezvous address. All ranks use the same values; `host` must be reachable from every rank and host the first FFN rank. |
| `num_attention_ranks` | Attention DP x TP. |
| `num_ffn_ranks` | FFN DP (FFN TP is 1). |
| `compute_gate_on_attention` | Must be `true` on both roles. |
| `afd_process_group_timeout_s` | AFD world join timeout, default 120 s. Raise it when one role loads weights much more slowly than the other. |
| `async` / `async_dp` | Must remain `false`. |

### `connector_extra_config`

Unknown keys are rejected. All keys are optional.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `payload_format` | `str` | `"bf16"` | Dispatch wire format: `"bf16"` or `"fp8_block128_fp32"` (FP8 e4m3 rows with one fp32 scale per 128 channels). The FFN must load matching expert weights: BF16 weights for `bf16`, block-128 FP8 weights (for example an FP8 checkpoint with `weight_block_size=[128, 128]`) for `fp8_block128_fp32`. |
| `expert_backend` | `str` | `"deepgemm"` | `"deepgemm"` (DeepGEMM grouped GEMMs) or `"fused_experts"` (vLLM `fused_experts`, a reference path). |
| `capacity_tokens_per_rank` | `int \| null` | `null` | Largest dispatch one Attention rank issues per call (with Attention TP, its slice); sizes the transport buffers. `null` uses the larger of `max_num_batched_tokens` and the largest CUDA-graph capture size, divided by the Attention TP size. |
| `num_lanes` | `int \| null` | `null` | In-flight micro-batches, one transport buffer each, `1` or `2`. `null` uses vLLM's micro-batch count: 2 with DBO, otherwise 1. |
| `bucket_granularity` | `int` | `1` | Dispatch buckets are rounded up to this multiple, then to the next rung of the bucket ladder derived from the CUDA-graph capture sizes. Every step uses one of a few kernel sets compiled at startup. |
| `cpu_sync_min_tokens` | `int` | `0` | Eager steps whose bucket is below this value skip the host sync that reads receive counts; FFN ranks then allocate the worst-case receive size. `0` always syncs in eager mode. Graph steps never sync. |
| `num_sms` | `int` | `0` | SMs used by the dispatch/combine kernels. `0` uses a fixed fraction of the device. |
| `num_qps` | `int` | `0` | RDMA queue pairs per transport buffer. `0` uses the transport's default. |
| `expert_num_sms` | `int` | `0` | SMs used by the FFN's DeepGEMM expert kernels; must be even. `0` uses all SMs. With DBO, a value that leaves SMs for the other lane's transport kernels avoids delaying overlapping GEMMs. |
| `cpu_timeout_s` | `int` | `120` | Transport host-wait timeout. |
| `gpu_timeout_s` | `int` | `120` | Transport kernel-spin timeout. |
| `network` | `object` | see below | Fabric settings. |

`network` keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `gin` | `"auto"` | NCCL GIN (RDMA between NVLink domains). `auto` enables it only when ranks span several NVLink domains; `on` and `off` force it. `off` is rejected when ranks span several domains. |
| `hybrid` | `"auto"` | DeepEP hybrid mode (rail-aligned RDMA plus NVLink forwarding). `auto` means on. |
| `rdma_traffic_class` | `3` | GIN service level / traffic class. |

The connector sets DeepEP's GIN switch (`EP_DISABLE_GIN`) itself from the
network plan; do not set it in the launch environment. NICs are NCCL's own
per-GPU choice. On RoCE whose rails are separate, unrouted subnets, pin every
rank to NICs on one shared rail with `NCCL_IB_HCA`.

`num_lanes` and the DBO setting must match on both roles; the Attention-side
capacity and bucket ladder are adopted by the FFN in the handshake. An FFN rank
that sets `capacity_tokens_per_rank` explicitly must agree with Attention.

## Required vLLM settings

| Setting | Role | Requirement |
| --- | --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER=1` | Both | Native ModelRunnerV2 is required. |
| `--enable-expert-parallel` | Both | Required. FFN EP size equals `num_ffn_ranks`. |
| `--tensor-parallel-size` | FFN | Must be `1`. Attention may use TP > 1. |
| Expert placement | FFN | `linear` (vLLM's default `--expert-placement-strategy`). |
| `--all2all-backend` | FFN | `allgather_reducescatter` (vLLM's default). The connector moves the rows itself, so vLLM's MoE all-to-all never runs; this backend allocates no buffers and keeps vLLM's own DeepEP out of the process. |
| `--moe-backend` | FFN | `triton` or `auto`. `auto` is pinned to `triton`; any other backend is rejected because it reorders the expert weights that the connector reads. |
| `--enable-eplb` | Both | Not supported. |
| `--enable-dbo` | Both | Optional, but must match on both roles. DBO uses exactly two micro-batches. |
| CUDA graphs | Both | Eager or `FULL_DECODE_ONLY`. |
| Pipeline / context parallelism, sequence-parallel MoE | Both | Not supported. |

## Supported layouts

| Layout | Status |
| --- | --- |
| `F >= 1`, any `A`, `num_experts % F == 0` | Supported. The Qwen3 E2E cases use `2A2F`, which is the smallest layout that runs expert parallelism over several FFN ranks. |
| Attention DP x TP | Supported, including TP > 1: each TP rank dispatches its slice of the DP rank's tokens and the TP group all-gathers the combined rows. |
| FFN DP x TP1 | Required. FFN experts are not TP-sharded. |
| One node | Supported; ranks reach each other over NVLink. |
| Several nodes | Supported when every node (NVLink domain) holds an equal, contiguous block of world ranks, with FFN ranks first. Writing each node's ranks as `<A>A<F>F`, the layouts `2A0F,0A2F` (Attention on one node, FFN on another) and `4A0F,0A4F` are valid; `4A1F,4A1F` (FFN ranks not contiguous) and `4A0F,0A2F` (unequal blocks) are not. The fabric must route between the NICs NCCL picks. |

## Minimal launch shape

`2A2F` on one node with Qwen3-30B-A3B (BF16), Attention DP2 on GPUs 0-1 and
FFN DP2 on GPUs 2-3:

Attention side:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/Qwen3-30B-A3B \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"DeepEPM2NAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":2,"num_ffn_ranks":2,"compute_gate_on_attention":true}}'
```

FFN side:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
CUDA_VISIBLE_DEVICES=2,3 vllm serve /path/to/Qwen3-30B-A3B \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --host 127.0.0.1 \
  --port 18001 \
  --additional-config '{"afd":{"role":"ffn","connector":"DeepEPM2NAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":2,"num_ffn_ranks":2,"compute_gate_on_attention":true}}'
```

Send API traffic only to the Attention server. For eager, DBO, Attention TP
and FP8 examples, see the
[DeepEP M2N recipes](../../recipe/gpu/DeepEPM2NAFDConnector/README.md).

## Validation

- CPU unit tests:
  [`tests/unit/connectors/test_deepep_m2n_connector.py`](../../tests/unit/connectors/test_deepep_m2n_connector.py)
  runs against a fake afdEP module when afdEP-comm is not installed.
- GPU connector tests: [`tests/gpu`](../../tests/gpu/README.md) checks
  dispatch, DeepGEMM experts and combine against references on one node.
- Model E2E: the Qwen3 MoE cases `afd-deepep-m2n-*-2a2f` in
  [`tests/e2e`](../../tests/e2e/README.md#deepep-m2n-scenarios) (eager, graph,
  graph + DBO, and Attention TP2), the same five DeepSeek-V2-Lite cases and
  the two-pod cases in
  `tests/e2e/models/qwen3_moe/test_qwen3_moe_multi_pod.py`.

## Requirements and limitations

- SM90 or newer on every rank; the handshake rejects older GPUs.
- Qwen3 MoE with every layer MoE and no shared expert, or DeepSeek-V2.
- `compute_gate_on_attention=true` on both roles; EPLB is not supported.
- FFN TP must be 1; FFN EP size equals `num_ffn_ranks`.
- `num_experts` must be divisible by `num_ffn_ranks`.
- CUDA graphs are limited to `FULL_DECODE_ONLY`; DBO uses exactly two
  micro-batches and must be enabled on both roles or neither.
- Every rank must take part in every MoE layer and stage. Missing ranks,
  mismatched settings, or a crashed peer fail at the handshake or after the
  transport timeouts (`cpu_timeout_s`, `gpu_timeout_s`).
- There is no fallback to another transport.
