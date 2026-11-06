# DSpark Draft-Model Speculation

Load with the primary module contract when a change touches DSpark: the
draft-model registration (`DSparkDraftModel` →
`AFDDSparkDeepseekV4ForCausalLM`), the speculator seam
(`afd_plugin/v1/worker/npu/dspark_speculator.py`), the target/draft layered FFN
wiring (`models/npu/deepseek_v4_dspark.py`), or the DSpark E2E evidence
harnesses under `tests/e2e/models/deepseek_v4_flash/`.

Designs: `docs/design/module/model_integration.md`,
`docs/design/module/execution_platforms.md`,
`docs/design/module/attention_runtime.md`;
guide: `recipe/npu/CAMAsyncAFDConnector/deepseek_v4_flash/README.md`.

## Scope

DSpark is an Ascend-only draft model paired with a DeepSeek-V4 target under
`CAMAsyncAFDConnector`. The attention role keeps native eager proposal; the FFN
role may independently use layered eager or `FULL` graph execution. Supported
combinations are enumerated in `execution_platforms.md`; everything else is
excluded rather than best-effort.

## Feature checks

- Respect the excluded set: DSpark does not support DBO or FFN two-stream, and
  it does not extend to model families outside DeepSeek-V4. A PR that enables
  either — even behind a flag — needs a design-page change and device evidence
  first.
- The speculator patch (`AFDDSparkSpeculator.set_attn`, `.propose`) must follow
  the patch contract against the pinned vLLM-Ascend ref, including signature
  parity and `Patch reason:`/`Patch functionality:` comments.
- Target and draft must agree on layered FFN weight extraction
  (`_extract_async_cam_layered_weights`) and on the layer set each side runs;
  a mismatch fails at startup or, worse, silently runs a partial draft.
- The prefill/decode split used by the DSpark recipes is a launch contract:
  `MODEL`, `DSPARK_CONFIG`, TP/DP/EP counts, and the upstream worker choice on
  P must stay consistent between the recipe, the README, and the E2E harness.
- Evidence for a behavioral claim comes from the DSpark evidence harnesses
  (`dspark_eager.py`, `dspark_evidence.DSparkEagerEvidence`,
  `dspark_ffn_graph_evidence.DSparkFFNGraphEvidence`), which exercise SWA
  boundaries and chunked prompts. A unit test alone cannot qualify draft
  execution.

Test the draft/target wiring CPU-side in
`tests/unit/model_executor/models/npu/test_deepseek_v4_dspark.py` and name the
NPU run for anything device-dependent.
