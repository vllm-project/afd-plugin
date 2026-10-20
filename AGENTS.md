# AFD Plugin Development Guidelines

This document provides instructions for contributors to the AFD Plugin project.
Follow these guidelines to keep code changes maintainable, focused, and
consistent.

## Code Style

### Python Conventions

- **Imports**: All imports at the top of the file. Valid exceptions for
  inline/deferred imports:
    - Circular imports (use inline imports)
    - Lazy loading for worker/isolation processes
    - Note: Type-checking imports wrapped in `if TYPE_CHECKING:` should still
      be placed at the top of the file.

- **Global Variables**: Avoid new global variables. Pass dependencies explicitly
  through function parameters.

    **Allowed:**
    - Constants named `ALL_UPPER_CASE`
    - Immutable configuration objects

    **Requires Approval:**
    - Any new mutable global state

- **No Magic Numbers**: Use named constants with descriptive names:

    ```python
    # Bad
    if seq_len > 2048: ...

    # Good
    MAX_CONTEXT_LENGTH = 2048
    if seq_len > MAX_CONTEXT_LENGTH: ...
    ```

- **Descriptive Naming**: Use names that describe functionality, not
  implementation details.

    ```python
    # Bad
    is_deepseek_v3_r1
    flag1
    tmp_var

    # Good
    supports_dynamic_temperature
    uses_speculative_decoding
    ```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `AFDAttentionModelRunner`,
  `AFDNPUAttentionModelRunner`, `GPUFFNModelRunner`)
- **Functions/Methods**: `snake_case` (e.g., `forward_pass`,
  `compute_attention`)
- **Constants**: `ALL_UPPER_CASE` (e.g., `MAX_BATCH_SIZE`)
- **Variables**: `snake_case` (e.g., `token_ids`, `sequence_lengths`)

## Development Notes

- When using data structures from vLLM or vLLM Ascend, access their functions
  and member variables directly. Avoid `getattr`, `hasattr`, or proactively
  raising custom exceptions unless necessary. This keeps upstream compatibility
  issues visible through the original error when upgrading, and allows static
  type checkers such as mypy or pyright to detect missing or renamed attributes
  early.

- Avoid `Any` and `object` in parameter types unless necessary. Prefer concrete
  types that describe the expected contract.

- Do not split simple functions into excessive helper functions. Extract helpers
  only when they reduce meaningful complexity, avoid real duplication, or match
  an established local pattern.

## Test Conventions

Unit tests protect behavior at module seams. A test that cannot fail when the
behavior regresses is not coverage; delete it.

### What a test asserts

Assert the seam, not the plumbing.

- **Allowed**: return values, raised errors and their messages, protocol
  messages sent to a peer (connector payloads, metadata structures), final
  file or environment state, config-validation outcomes.
- **Forbidden**: private attributes of the object under test, arguments passed
  to an internal collaborator, call counts of an internal helper. Call order is
  a valid assertion only when the order is itself the contract (connector
  handshakes, service announce, DBO stage order).

The refactor test: if the implementation is restructured without changing a
protocol, the test MUST stay green.

### Setup and fakes

- Shared fakes live in `tests/unit/fakes/`, one module per upstream concept
  (`vllm_config`, `afd_config`, `connector`, `dp_metadata`). A builder defined
  in two test modules is a layout bug.
- Do not rebuild an object graph inside a test. Setup over ~15 lines belongs in
  a fixture or a factory.
- A test that patches more than ~4 attributes of the module under test is
  testing its own mocks. Raise the patch target or split the test.

### Size and shape

- One behavior per test. The name states the observable behavior
  (`test_ffn_worker_stops_loop_before_parent_shutdown`), not the internal call
  chain (`test_ffn_worker_calls_x_then_y`).
- A test body over ~50 lines needs `# long-test: <reason>` on the `def` line.
- The third test with the same prefix and setup becomes a `parametrize` table.
  Do not copy-paste a body a third time.
- Tests are deterministic: no sleeps, no wall-clock assertions, no reliance on
  set iteration order.

### Lifecycle

- A PR that removes behavior deletes the tests in the same PR. Skipping is not
  a retirement mechanism; a skip needs an issue link.
- A new test file MUST fall under an existing `validation_paths` entry in
  `docs/design/module/`. If no module owns it, it is an E2E test.
- Test files mirror the source tree
  (`tests/unit/<mirrored path>/test_<module>.py`). Do not create a file per
  feature or per acceptance bullet.
- `tests/unit` covers `afd_plugin` only. Harness code under `tests/e2e` is
  validated by running the E2E cases, not by unit tests.

### Boundaries

- `tests/unit` MUST collect on a CPU-only host without `torch` or vLLM.
  Runtime imports are gated with `pytest.importorskip` before the guarded
  module is imported.
- Device behavior is verified in `tests/npu` or `tests/e2e`. Never simulate
  device behavior with fakes and then treat it as evidence.
- Text artifacts (build manifests, patch markers, docs links) may be asserted
  as text. Production Python modules may not: no test reads
  `afd_plugin/**/*.py` and asserts on a substring.

### Upstream contract checks

Exactly one module owns upstream signature alignment, as a single parametrized
table covering patch functions. Inheritance alignment is mypy's job, not
pytest's.

## Patching Requirement

**Strict Review Required**: All new patches must undergo thorough architectural
review.

Reviewers must verify:

- The patch targets the correct upstream component
- The patch is minimal and focused
- Performance implications are understood
- A long-term plan exists for upstream contribution or removal

When copying or wrapping upstream code, mark every AFD-specific difference with
`# ### PATCH START: ...` and `# ### PATCH END: ...` comments. Keep the marker
text short and specific so reviewers can compare the patch against upstream
quickly.

Every patch function must have comments immediately above the function that
explain why that upstream function is patched and what behavior the patch adds
or changes. The patched function signature, including return type, must match
the upstream function exactly. If a patch must add parameters, document those
new parameters in the comments immediately above the patch function.

AFD patches are developed against pinned vLLM and vLLM-Ascend tags. By default,
patch functions should copy the corresponding upstream function from that tag
and mark only AFD-specific differences with `# ### PATCH START: ...` and
`# ### PATCH END: ...`. When upgrading to a new upstream tag, copy the new
upstream function again and re-apply the marked AFD differences. Avoid using
`_original_*` delegation as the main non-AFD path unless the upstream function
is too large or unsuitable for local expansion; such exceptions must be called
out in the patch function comments.

**Required Pattern**: AFD-specific functionality should be implemented via:

1. **Patching**:
    - `afd_plugin/compat/patches/` - vLLM compatibility patches
    - `afd_plugin/compat/patches/npu/` - NPU-specific compatibility patches
    - `afd_plugin/compat/npu/` - Ascend NPU runtime compatibility patches
    - Patch is not the best solution for all cases. Use it when necessary.

2. **Inheritance**:
    - Extend vLLM model runners or workers for AFD-specific behavior
    - Add AFD-specific components via composition
    - Keep backend-specific operators in clearly scoped backend modules

3. **External upstream contributions** where appropriate

**Example Patch Pattern:**

```python
# Upstream source: vllm/some_upstream_module.py
class UpstreamClass:
    def route_request(self, request: Request, priority: int = 0) -> RouteResult:
        route = self.router.pick(request, priority)
        return self.scheduler.enqueue(request, route)
```

```python
# AFD patch: afd_plugin/compat/patches/example_patch.py
from vllm.some_upstream_module import UpstreamClass

# Patch reason: upstream route_request does not know about AFD's connector
# routing policy.
# Patch functionality: use AFD routing for AFD requests while delegating
# non-AFD requests through the copied upstream logic unchanged.
# Signature: matches upstream; no added parameters.
def route_request(self, request: Request, priority: int = 0) -> RouteResult:
    # ### PATCH START: AFD custom routing
    if request.is_afd:
        route = self.afd_router.pick(request, priority)
    else:
        route = self.router.pick(request, priority)
    # ### PATCH END: AFD custom routing
    return self.scheduler.enqueue(request, route)

UpstreamClass.route_request = route_request
```
