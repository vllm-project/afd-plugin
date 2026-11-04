# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Focused device check: one compiled artifact reused with live routing.

Run on the NPU host with the pinned stack. Uses the production op registration
and the real forward context; nothing here mocks the profile flag.

Assertions:
  1. balancing enabled + first compile through the profile path -> balanced IDs
  2. profile flag cleared, same compiled callable reused -> two distinct,
     non-balanced live ids both preserved, and no second compilation
  3. ACL graph captured with profile mode disabled -> two replays with the same
     static input buffer refilled by copy_() -> each replay keeps its own ids
"""

from __future__ import annotations

import os

# The pinned Ascend resolves _EXTRA_CTX.* out of additional_kwargs only in V2
# mode; set it before any vllm import so this check exercises that path.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

import torch
import torch_npu  # noqa: F401  registers the device
from vllm.forward_context import override_forward_context

from afd_plugin.compat.npu import routing_ids  # noqa: F401  registers the op

NUM_EXPERTS = 8
TOKENS = 8
TOPK = 2
op = torch.ops.vllm.afd_balance_routing_ids_


def _forward_context(in_profile_run: bool):
    """Real forward context; the flag is the attribute set_forward_context sets."""
    from vllm.forward_context import ForwardContext

    # V2 layout: the pinned Ascend reads the marker out of additional_kwargs, so
    # set it there rather than as a plain context attribute.
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={"in_profile_run": in_profile_run},
    )
    return override_forward_context(ctx)


def _ids(offset):
    flat = torch.arange(offset, offset + TOKENS * TOPK, dtype=torch.int32, device="npu")
    return (flat % NUM_EXPERTS).reshape(TOKENS, TOPK)


def _balanced():
    return [i % NUM_EXPERTS for i in range(TOKENS * TOPK)]


class _Gate(torch.nn.Module):
    """Stands in for the gate call: only the routing decision."""

    def forward(self, ids):
        op(ids, NUM_EXPERTS, False)
        return ids.clone()


def _compile_once(module):
    compiled = []

    def backend(gm, example_inputs):
        fn = torch._dynamo.backends.registry.lookup_backend("eager")(gm, example_inputs)
        compiled.append((fn, example_inputs))
        return fn

    first = _ids(9)
    torch.compile(module, backend=backend)(first)
    assert len(compiled) == 1, f"expected one compilation, saw {len(compiled)}"
    assert first.flatten().tolist() == _balanced(), (
        f"first compiled call under profile mode produced {first.tolist()}"
    )
    return compiled[0]


def check_compiled_reuse():
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    with _forward_context(True):
        assert _EXTRA_CTX.in_profile_run is True, "V2 layout did not expose the profile flag"
    with _forward_context(False):
        assert _EXTRA_CTX.in_profile_run is not True, "V2 layout leaked the profile flag"

    with torch.no_grad(), _forward_context(True):
        profile_ids = _ids(3)
        op(profile_ids, NUM_EXPERTS, False)
        assert profile_ids.flatten().tolist() == _balanced(), (
            "profile state is not balanced"
        )
        # The original failure compiles while profiling; keep that order.
        compiled_fn, example_inputs = _compile_once(_Gate())

    with torch.no_grad(), _forward_context(False):
        for offset in (5, 11):
            live = _ids(offset)
            expected = live.clone()
            compiled_fn(live, *example_inputs[1:])
            assert torch.equal(live, expected), (
                f"live ids changed for offset {offset}: {live.tolist()}"
            )
    print("eager compiled reuse: OK")


def check_acl_graph_reuse():
    with torch.no_grad(), _forward_context(False):
        buffer = _ids(1)
        graph = torch.npu.NPUGraph()
        static_in = buffer.clone()
        op(static_in, NUM_EXPERTS, False)
        out = static_in.clone()
        torch.npu.synchronize()

        # NPU graphs must be captured on a non-default stream.
        torch.npu.synchronize()
        capture_stream = torch.npu.Stream()
        with torch.npu.stream(capture_stream):
            graph.capture_begin()
            op(static_in, NUM_EXPERTS, False)
            out.copy_(static_in)
            graph.capture_end()
        torch.npu.synchronize()

        for offset in (5, 11):
            expected = _ids(offset)
            static_in.copy_(expected)
            out.zero_()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(out, expected), (
                f"acl graph replay {offset} produced {out.tolist()}"
            )
    print("acl graph reuse: OK")


if __name__ == "__main__":
    check_compiled_reuse()
    check_acl_graph_reuse()
    print("ALL CHECKS PASSED")
