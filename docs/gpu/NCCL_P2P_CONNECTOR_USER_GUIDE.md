## Background

P2pNcclAFDConnector (implemented with vLLM's PyNcclCommunicator) is a GPU-backed point-to-point connector for CUDA deployments. This guide documents its usage, configuration contract, rank mapping, and operational limits to help users and maintainers configure it correctly.

## When to use `P2pNcclAFDConnector`

Use this connector for CUDA deployments that disaggregate Attention and FFN
workers and exchange hidden states synchronously through NCCL point-to-point
communication. On vLLM 0.30, DeepSeek MoE layers keep the native MoE forward
contract and replace only the local experts with an AFD remote-experts proxy.
The MoE gate may run on Attention or FFN.

It supports both prefill and decode which all support eager mode. CUDA graph support is currently limited to `FULL_DECODE_ONLY`, which is mainly used in decode instance. The checked-in DeepSeek V2 Lite recipes cover colocated and prefill/decode-disaggregated deployments.

## How it works

Throughout this section, let `A = num_attention_ranks` and `F = num_ffn_ranks`. The topology rule (`A >= F`) makes `min_size = min(A, F) = F`. Each subgroup's `ratio` is the number of Attention peers it holds; when `F` divides `A` every subgroup holds `A / F` of them, and otherwise the counts differ by one. One physical process sits at up to three different rank numbers — an AFD world rank, a subgroup rank, and a control-plane (`p2p`) rank — all derived deterministically from the role, role rank, and topology counts.

### AFD world (shared rendezvous)

The connector creates one AFD NCCL world of size `F + A`, rendezvoused at `tcp://host:port`, ordered FFN-first:

```text
world rank:   0    1   ...  F-1   F    F+1  ...  F+A-1
member:       F0   F1  ...  F_    A0   A1   ...  A_
```

FFN role rank `i` gets world rank `i`; Attention role rank `j` gets world rank `F + j`. This world fixes the global ordering; the actual traffic flows through the two derived structures below.

### Data plane: one subgroup per FFN rank

Each FFN rank `k` owns subgroup `k`, containing itself plus the consecutive Attention peers assigned to it: Attention rank `a` joins subgroup `a * F // A`, which is `a // ratio` whenever `F` divides `A`. Inside a subgroup the FFN rank is always subgroup rank `0` and the Attention peers occupy subgroup ranks `1..ratio`.

Each subgroup is its own process group, rendezvoused on the AFD world's store under a `PrefixStore` of its own so no extra port is needed, carrying two NCCL communicators: Attention-to-FFN for hidden states and FFN-to-Attention for FFN outputs. Per layer/stage, each Attention peer sends its hidden states to subgroup rank `0`; the FFN rank receives from ranks `1..ratio` in one NCCL group, concatenates along the token dimension, runs FFN work, splits the output by the recorded sequence lengths, and sends the slices back to the originating Attention ranks in one NCCL group. The data path uses vLLM `PyNcclCommunicator.send()` / `recv()` on the current CUDA stream; the FFN rank wraps its receives and its sends in `group_start()` / `group_end()`.

### Control plane: the DP metadata group

Tensor shapes vary per step (token counts per DP rank), so before data transfers can be posted, the FFN side needs the per-stage token counts. A separate NCCL process group named `p2p`, of size `F + min_size = 2F`, carries this DP metadata. It rendezvouses at the same `tcp://host:port` as the AFD world but as a distinct group:

- All `F` FFN ranks join, keeping their role rank as their `p2p` rank (`0..F-1`).
- Only the first `min_size` Attention ranks (role ranks `0..F-1`) join, at `p2p` ranks `F..2F-1`. Attention ranks with role rank `>= F` do not participate; their metadata send is a no-op.

The pairing is 1:1 by index: Attention role rank `r` sends the control payload to FFN rank `r`. The payload carries the whole per-rank token-count vector plus graph-capture/warmup flags, so one sender per FFN rank is enough — each FFN rank derives every Attention peer's token count from that vector, including peers that are not in the control-plane group.

In code, the control plane is a pluggable module rather than part of the connector interface. `P2pNcclAFDConnector` creates a `P2pNcclAFDControlPlane` instance — an implementation of the `AFDControlPlane` contract from `afd_plugin/connectors/base.py` — and exposes it as `connector.control_plane`. The Attention-side runner calls `control_plane.update_state_from_dp_metadata(...)` and `control_plane.send_dp_metadata_list(...)` before each step; the FFN worker loop blocks on `control_plane.recv_dp_metadata_list()`. Because the connector exposes a `control_plane`, each FFN step is triggered by the arrival of a DP metadata payload (connectors without a DP metadata control plane leave `control_plane` as `None` and drive FFN steps from the connector itself).

### Worked example: `4A2F` (`ratio = 2`, `min_size = 2`)

```text
AFD world (port):        F0=0  F1=1  A0=2  A1=3  A2=4  A3=5

Data plane:
  subgroup 0 (prefix afd_subgroup_0):   F0(rank 0) <-> A0(rank 1), A1(rank 2)
  subgroup 1 (prefix afd_subgroup_1):   F1(rank 0) <-> A2(rank 1), A3(rank 2)

Control plane "p2p" group (port, size 4):
  p2p ranks:             F0=0  F1=1  A0=2  A1=3      (A2, A3 excluded)
  metadata flow:         A0 -> F0,   A1 -> F1
```

All three groups are derived from just `host`, `port`, the role, and the rank counts; any mismatch across processes makes the collective rendezvous hang or fail. Note that `num_attention_ranks` and `num_ffn_ranks` are worker totals, not engine counts: attention DP=2 x TP=2 means `num_attention_ranks=4`, and the connector expands the per-DP token vector across TP peers when the vector is shorter than the attention rank count.

## Configuration

AFD configuration is supplied through vLLM's `--additional-config` under the `afd` key. The presence of the `afd` object enables AFD; omit it to disable AFD. Attention and FFN processes must use the same rendezvous address and topology counts; only `role` and the device assignment differ. The plugin selects the role-specific worker automatically.

```jsonc
{
  "afd": {
    "role": "attention",
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 6239,
    "num_attention_ranks": 1,
    "num_ffn_ranks": 1,
    "compute_gate_on_attention": false,
    "afd_process_group_timeout_s": 120,
    "connector_extra_config": {}
  }
}
```

### Fields

| Field | Type | Default | Required / meaning |
| --- | --- | --- | --- |
| `role` | `"attention" \| "ffn"` | `"attention"` | Role owned by this process. Attention sends hidden states; FFN receives and returns FFN outputs. |
| `connector` | `str` | `"P2pNcclAFDConnector"` | Must be `P2pNcclAFDConnector` for this GPU path. |
| `host` | `str` | `"127.0.0.1"` | Non-empty rendezvous/control-plane host. All participating ranks must use a reachable, identical value. Host must be the first rank of FFN. |
| `port` | `int` | `1239` | Rendezvous port on `host`, valid range `1..65535`. It must be free and reachable. Subgroups share this rendezvous and need no ports of their own. |
| `num_attention_ranks` | `int` | `1` | Total number of AFD Attention ranks, including DP/TP-derived worker ranks. Must be positive. |
| `num_ffn_ranks` | `int` | `1` | Total number of AFD FFN ranks, including DP/TP-derived worker ranks. Must be positive. |
| `compute_gate_on_attention` | `bool` | `false` | When `false`, FFN owns the native gate and experts. When `true`, Attention owns the native gate and transfers router logits to the FFN external-router expert path. |
| `afd_process_group_timeout_s` | `int` | `120` | Rendezvous timeout, in seconds, for the AFD world process group join. Must be positive. The right value depends on rank count, placement, and model size; deployment tooling should compute it and pass it in. |
| `connector_extra_config` | `dict` | `{}` | Must remain empty; `P2pNcclAFDConnector` does not currently support connector-specific options. |
| `async` / `async_dp` | `bool` | `false` | Must remain `false` for `P2pNcclAFDConnector`; AFD async mode requires `CAMAsyncAFDConnector`. |

Compatibility aliases currently accepted are `afd_role`, `afd_connector`, `afd_host`, and `afd_port`. Canonical field names should be used in new examples.

The connector factory derives each process's role rank from its global DP rank
and local PCP/TP coordinates before connector construction. Do not configure a
role rank.

## Topology rules

`P2pNcclAFDConnector` currently requires:

```text
num_attention_ranks >= num_ffn_ranks
```

The Attention ranks are then spread over the FFN ranks in consecutive blocks that differ in size by at most one: Attention rank `a` joins the subgroup of FFN rank `a * F // A`. When `F` divides `A` every FFN rank maps to the same number of Attention ranks, as before.

Examples:

| Topology | Valid? | Mapping |
| --- | --- | --- |
| `1A1F` | Yes | `F0 <-> A0` |
| `2A2F` | Yes | `F0 <-> A0`, `F1 <-> A1` |
| `4A2F` | Yes | `F0 <-> A0,A1`, `F1 <-> A2,A3` |
| `1A2F` | No | Attention rank count is smaller than FFN rank count. |
| `3A2F` | Yes | `F0 <-> A0,A1`, `F1 <-> A2` |

## Minimal launch shape

Select native ModelRunnerV2 before starting either GPU role. Attention uses
the native V2 runner; FFN remains connector-driven. GPU V1 is unavailable.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

The minimal DP1/TP1 commands below illustrate the launch shape. H20
hardware validation covers DeepSeek-V2-Lite DP1/TP2 Graph + DBO with 128
GSM8K samples and live two-stage/full-replay assertions; see the
[PR #454](https://github.com/vllm-project/afd-plugin/pull/454).

Attention side:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

FFN side:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /path/to/model \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --enable-dbo \
  --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 \
  --host 127.0.0.1 \
  --port 18001 \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

The HTTP ports (`18000` / `18001` above) are vLLM service ports and are separate from the AFD NCCL rendezvous base port (`6239`). Use distinct CUDA devices for the two roles.

For complete `1A1F`, `2A2F`, `4A4F`, eager, DBO, and CUDA graph examples, see `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite`.

## ModelRunnerV2

Set `VLLM_USE_V2_MODEL_RUNNER=1` on both roles to select the V2 pairing.
Only Attention uses the native V2 runner; FFN remains connector-driven.
Current validation requires model-specific gate placement, static EP, PP=PCP=DCP=1,
role-rank totals equal to DP x TP, and a registered AFD architecture. Elastic
EP, EPLB, sequence-parallel MoE and compile SP are rejected.

Eager and `FULL_DECODE_ONLY` graph execution are accepted. V2 DBO requires
exactly two microbatches and accepts Attention DP1 or DP > 1. The generic
[GPU E2E runner](../../tests/e2e/runner.py) covers eager/graph DP2 and TP2,
including `afd-v2-eager-dbo-dp2` and `afd-v2-graph-dbo-dp2`. H20 validation
also covers the existing DeepSeek-V2-Lite DP1/TP2 Graph + DBO recipe with
128 GSM8K samples and live two-stage/full-replay assertions. Model-specific
exclusions still apply; a registered model does not qualify every topology.

## Receive buffers and memory

FFN retains hidden-state receive buffers only for graph-capture shapes.
Eager prefill/mixed steps allocate these buffers per receive, preventing a
persistent hidden-state allocation for every observed token count. Captured
buffers keep stable addresses for replay and are released on connector close.
This is not a general cache-size limit: the smaller input-ID receive cache
still retains per-shape entries when graphs are enabled, and graph storage
is owned separately by the FFN runner.

FFN replays a cached graph only when Attention's control payload explicitly
sets `is_graph_replaying` and the graph key matches. A matching token shape
alone does not authorize replay. These behaviors are covered by the
[connector tests](../../tests/unit/connectors/test_p2p_connector.py) and
[FFN graph-policy tests](../../tests/unit/v1/worker/test_cuda_graph.py).

## Requirements and limitations to document in code

- CUDA-capable PyTorch/vLLM environment with NCCL and vLLM's `PyNcclCommunicator` available.
- Hidden-state tensors must reside on CUDA devices; CPU tensors are rejected.
- All ranks must agree on `host`, `port`, rank counts, model hidden size/dtype, and role-rank assignment.
- The rendezvous port on `host` must be free and reachable.
- Initialization is collective: missing ranks, mismatched counts, or duplicate role ranks can cause initialization failure or timeout.
- Current GPU CUDA graph support is `FULL_DECODE_ONLY`; GPU DBO plus CUDA graph is limited to exactly two ubatches.
- CUDA remote experts do not currently support EPLB on the Attention role.
- To enable DBO, set `--enable-dbo`, and configure the threshold with `--dbo-decode-token-threshold` and `--dbo-prefill-token-threshold`. See `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite` for examples.
- The repository recipes currently validate specific GPU layouts (including DeepSeek V2 Lite and tested A/H-class hardware). Cross-node placement is verified for DeepSeek-V2-Lite `2A2F` with the FFN ranks on separate hosts, over TCP (ENA) and over EFA; other topologies and hardware should be treated as unverified.
- There is no automatic fallback from this connector to another transport. Select an NPU connector explicitly on Ascend.
