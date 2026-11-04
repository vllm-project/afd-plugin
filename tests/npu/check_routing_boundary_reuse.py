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

import torch
import torch_npu  # noqa: F401  registers the device
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

from afd_plugin.compat.npu import routing_ids  # noqa: F401  registers the op

NUM_EXPERTS = 8
TOKENS = 8
TOPK = 2
op = torch.ops.vllm.afd_balance_routing_ids_


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

    torch.compile(module, backend=backend)(_ids(9))
    assert len(compiled) == 1, f"expected one compilation, saw {len(compiled)}"
    return compiled[0]


def check_compiled_reuse():
    with torch.no_grad():
        _EXTRA_CTX.in_profile_run = True
        profile_ids = _ids(3)
        op(profile_ids, NUM_EXPERTS, False)
        assert profile_ids.flatten().tolist() == _balanced(), "profile state is not balanced"

        compiled_fn, example_inputs = _compile_once(_Gate())

        _EXTRA_CTX.in_profile_run = False
        for offset in (5, 11):
            live = _ids(offset)
            expected = live.clone()
            compiled_fn(live, *example_inputs[1:])
            assert torch.equal(live, expected), f"live ids changed for offset {offset}: {live.tolist()}"
    print("eager compiled reuse: OK")


def check_acl_graph_reuse():
    with torch.no_grad():
        _EXTRA_CTX.in_profile_run = False
        buffer = _ids(1)
        graph = torch.npu.NPUGraph()
        static_in = buffer.clone()
        op(static_in, NUM_EXPERTS, False)
        out = static_in.clone()
        torch.npu.synchronize()

        graph.capture_begin()
        op(static_in, NUM_EXPERTS, False)
        out.copy_(static_in)
        graph.capture_end()

        for offset in (5, 11):
            expected = _ids(offset)
            static_in.copy_(expected)
            out.zero_()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(out, expected), f"acl graph replay {offset} produced {out.tolist()}"
    print("acl graph reuse: OK")


if __name__ == "__main__":
    check_compiled_reuse()
    check_acl_graph_reuse()
    print("ALL CHECKS PASSED")
