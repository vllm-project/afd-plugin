## Background

`FastAFDAFDConnector` runs the AFD expert exchange with the
[FastAFD](https://github.com/hao-ai-lab/FastAFD) M2N MegaMoE kernels on
Blackwell (SM100). Attention routes every token; per MoE layer one kernel sends
each token to the FFN ranks that own its experts and a second one collects and
reduces the results, with the shared expert computed in between. FFN runs one persistent expert kernel per step
for all MoE layers. Ranks synchronize through NVLink symmetric memory instead
of per-layer NCCL send/receive pairs.

The kernels are vendored from FastAFD
`3c7161949310b6d59d6b4cf9bf997a4935c8113b` under `csrc/gpu/fastafd`
(MIT, with the DeepGEMM MIT notice kept).

## When to use `FastAFDAFDConnector`

- Blackwell GPUs (SM100) in one NVLink domain, one GPU per Attention or FFN
  rank, `num_attention_ranks + num_ffn_ranks <= 72`. Validated so far on one
  Attention and one FFN rank (2xB200).
- A supported model with FP8 checkpoint weights. Currently Qwen3.5/3.6 MoE
  (`qwen3_5_moe_text`) FP8 checkpoints with 128x128 block scales, such as
  `Qwen/Qwen3.6-35B-A3B-FP8`.
- Prefill, mixed and decode steps all use the same kernels. Attention supports
  eager and `FULL_DECODE_ONLY` CUDA graphs.

Use `P2pNcclAFDConnector` for other GPUs, other models or BF16 checkpoints.

## Build

The kernels are an opt-in extension. Install vLLM 0.30.0 first, then build
against its PyTorch:

```bash
AFD_BUILD_FASTAFD_OPS=1 python -m pip install -v --no-build-isolation --no-deps -e .
```

The build fetches CUTLASS v4.2.1 and fmt 11.2.0 headers and checks their
SHA-256. The FastAFD, DeepGEMM, CUTLASS and fmt license notices are installed
under `afd_plugin/_fastafd/licenses/`. Without network access, point `AFD_FASTAFD_CUTLASS_DIR` and
`AFD_FASTAFD_FMT_DIR` at local copies of those releases. The kernels compile
with nvcc the first time each shape runs (`CUDA_HOME` must point at a CUDA 13
toolkit) and cache under `DG_JIT_CACHE_DIR`, `~/.cache/afd_plugin/fastafd_jit`
by default.

## How it works

- **Union group.** Attention ranks take union ranks `0..A-1` and FFN ranks
  `A..A+F-1`. The connector creates this NCCL group at `tcp://host:port`,
  checks that all ranks agree on lanes and buffer shape and use distinct GPUs,
  and allocates one symmetric-memory buffer per microbatch lane.
- **Attention.** The Qwen MoE block computes the native router (softmax top-k,
  renormalized when `norm_topk_prob` is set). The connector's `dispatch`
  kernel quantizes the hidden states to per-32 UE8M0 FP8 and sends each token
  to its experts' FFN ranks; the gated shared expert runs locally meanwhile,
  and `combine` waits for the routed results and reduces the top-k.
- **FFN.** `FastAFDExpertExecutor` loads only its share of routed experts
  (`num_experts / F` per rank) from the checkpoint, requantizes the block
  scales to per-32 UE8M0 (one lossy e4m3 round trip), and launches one
  persistent kernel per step plan. It does not build the vLLM model or keep
  CUDA graphs; `FastAFDFFNModelRunner` only forwards step plans to it.
- **Control plane.** Attention union rank 0 sends each step plan
  (`AFDControlPayload`) to every FFN rank. Token data never travels on it.

Every Attention rank must run every layer and lane of a step in the same
order. The kernels synchronize on the GPU, so a rank that dies mid-step leaves
its peers spinning; only the kernels' NVLink barriers give up, after 300 s.

## Configuration

```json
{
  "afd": {
    "role": "attention",
    "connector": "FastAFDAFDConnector",
    "host": "127.0.0.1",
    "port": 6239,
    "num_attention_ranks": 1,
    "num_ffn_ranks": 1,
    "compute_gate_on_attention": true
  }
}
```

- `compute_gate_on_attention` must be `true`.
- `connector_extra_config` must be empty.
- Both roles need the same `--max-num-batched-tokens`, DBO setting and model.
  The symmetric buffers hold `max_num_batched_tokens` tokens per Attention
  rank.
- Tensor parallelism must be 1 on both roles.

Each role checks at startup, before any buffer is allocated: SM100 GPU,
bfloat16 activations, an FP8 checkpoint with 128x128 block scales, a MoE block
in every layer, a distinct GPU per rank, and the same lanes, layer count and
buffer shape on every rank. A mismatch fails with an error that names it.

## Minimal launch shape

Start both roles with `VLLM_USE_V2_MODEL_RUNNER=1`. FFN:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --language-model-only --max-model-len 4096 \
  --data-parallel-size 1 --tensor-parallel-size 1 --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 18001 \
  --additional-config '{"afd":{"role":"ffn","connector":"FastAFDAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1,"compute_gate_on_attention":true}}'
```

Attention:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --language-model-only --max-model-len 4096 \
  --data-parallel-size 1 --tensor-parallel-size 1 --enable-expert-parallel \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"FastAFDAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1,"compute_gate_on_attention":true}}'
```

Send requests to Attention only.

## Validation

- Kernel parity: `torchrun --nproc-per-node 2 -m
  tests.e2e.operators.fastafd_m2n_reference --model <FP8 checkpoint>` checks
  the fused and the split dispatch/combine kernels, with checkpoint and skewed
  routing, against a reference that applies the same per-32 quantization.
- End to end: `tests/e2e/runner.py` with `--afd-connector FastAFDAFDConnector
  --compute-gate-on-attention` and an FP8 checkpoint.

## Limitations

- SM100 only; one NVLink domain; no multi-node yet.
- FP8 block-scaled checkpoints only. The routed experts run in per-32 UE8M0
  FP8 (activations and the SwiGLU intermediate are quantized too), so outputs
  differ from the native block-FP8 path at the level of that format.
- Qwen3.5/3.6 MoE only. Qwen3.5/3.6 GDN does not support DBO, so these models
  use one lane.
- TP 1 on both roles; no EPLB, PP or speculative decoding.
