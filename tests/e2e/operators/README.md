# Async CAM source migration numerical checks

Build the 910C AFD package first. Run these checks serially on otherwise idle
NPUs; retain stdout, the software/checkpoint manifest and generated artifacts.
No operator source modification is required.

For each `TP` in `1 2 4`, dtype in `float16 bfloat16`, and quantization in `0 1`:

```bash
torchrun --standalone --nproc-per-node=$((TP + 2)) \
  -m tests.e2e.operators.async_cam_roundtrip \
  --tp "$TP" --dtype "$DTYPE" --dynamic-quant "$QUANT"
```

This uses two FFN ranks and forces multi-chunk reception by reducing receive
capacity. An independent CPU oracle checks routed weighted sums; each run
includes empty ranks, sparse routes, first/last experts, a single-token batch,
and repeated window use. Tolerances are fixed in the script before execution.

After the communication check, use one idle NPU for the checkpoint probe:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m tests.e2e.operators.moe_checkpoint_reference \
  --family dsv2 --model /path/to/DeepSeek-V2-Lite --dtype bfloat16 \
  --output /path/to/evidence/dsv2-moe-bf16
```

Repeat DSV2 with `--dtype float16`. After DSV2 acceptance, run with
`--family dsv4 --dtype bfloat16` and the DSV4 checkpoint. DSV4 defaults to
layer 0 (Hash) and `num_hash_layers` (ordinary router); DSV2 defaults to its
first MoE layer. `--layers` can select other real checkpoint layers.

The probe loads only the selected MoE layer, using the pinned native loader
and native post-load quantization processing. The Attention instance applies
the actual AFD role filter and owns a separately loaded shared MLP. Native
full MoE output supplies the reference for routed and final output. An
additional CPU oracle computes shared output directly from raw checkpoint
weights, including dynamic W8A8 quantization and DSV4 SwiGLU clamping. W8A8
requires the actual checkpoint's `weight_scale` and symmetric `weight_offset`;
FP8 inverse scales are not interchangeable with this format.

The probe saves fixed inputs/token IDs, routing, all reference/AFD outputs and
max-absolute/relative-L2 errors. Fixed tolerances apply to each component, so
small outputs cannot pass merely because of the absolute tolerance. It checks
MoE semantics independently of the transport test, but does not replace
DP2/TP4/EP8 E2E or two-stage token-layout tests. A failed or unexecuted probe
must be reported as such rather than treated as evidence of accuracy.

## CUDA MoE reference probe

Use one otherwise idle CUDA GPU with the pinned vLLM runtime:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tests.e2e.operators.moe_reference_cuda \
  --model /path/to/DeepSeek-V2-Lite --output /path/to/evidence/dsv2-moe-cuda
```

The probe truncates the checkpoint after each selected layer (the first MoE
layer by default; `--layers` selects others) and runs fixed bfloat16 inputs of
1, 7, 64 and 512 tokens through three paths: the native vLLM DeepSeek MoE
layer, the FFN role's `compute_ffn_output`, and the Attention role's gate
followed by the FFN role's `compute_experts_output`. Both AFD outputs must equal
native bit for bit.

An fp32 reference computed directly from the checkpoint tensors checks every
path. The relative L2 error must stay within 0.02 on tokens whose top-k choice
is not a near tie, meaning the k-th and (k+1)-th routing probabilities differ by
at least 2%. Near-tie tokens are counted in the report and excluded from that
bound, because bfloat16 logits may flip them. The probe saves inputs, logits
and outputs for every case plus `report.json`.

It checks one rank's FFN MoE semantics. It does not replace connector transport
or topology E2E. A new FFN backend can be checked as another path against the
same references.
