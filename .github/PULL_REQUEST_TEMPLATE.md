<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION AND MAKE SURE THE CHECKLIST ITEMS HAVE BEEN CONSIDERED.

## Purpose

<!-- What changed and why? Link the issue/RFC/design notes when available. -->

## Issue

- Related issue(s): #
- Closing keyword, only if fully resolved: Closes #

## Scope

- In scope:
- Out of scope:

## Implementation Notes

<!--
Call out vLLM extension points, plugin-owned classes, compatibility shims,
or any behavior that intentionally differs from the original AFD commit.
-->

## Test Plan

<!-- Commands or manual checks planned. Include CPU-only and GPU-gated coverage separately when relevant. -->

## Test Result

<!-- Paste command results, skip reasons, links to GPU validation, or a short explanation if not run. -->

## Docs Impact

- Files updated:
- If none, reason:

---

<details>
<summary>Essential PR Checklist</summary>

- [ ] Purpose is clear and linked to public context when possible.
- [ ] Scope is bounded.
- [ ] Compatibility with vLLM v0.28.0 is considered.
- [ ] No changes are made to the vLLM source checkout.
- [ ] Plugin-owned classes or explicit dotted class paths are preferred over monkey patches.
- [ ] Any compat shim or monkey patch is isolated, idempotent, version-guarded, documented, and tested.
- [ ] Imports remain CPU-safe; CUDA-heavy work is delayed or GPU-gated.
- [ ] Validation evidence is included, including skipped GPU tests when applicable.
- [ ] Documentation impact is stated.

</details>
