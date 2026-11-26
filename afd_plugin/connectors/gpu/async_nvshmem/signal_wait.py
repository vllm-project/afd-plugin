# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded mapped-peer signal waits, including CUDA Graph replays."""

import torch
import triton
import triton.language as tl


@triton.jit
def _timer():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
    )


@triton.jit(do_not_specialize=["index", "value"])
def _wait_signal(flags, index, value, timeout_ns):
    start = _timer()
    observed = tl.inline_asm_elementwise(
        "ld.acquire.sys.global.u64 $0, [$1];",
        "=l,l",
        [flags + index],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    while observed != value:
        if _timer() - start >= timeout_ns:
            tl.device_print("AFD async GPU reply wait timed out; signal index", index)
            tl.inline_asm_elementwise(
                "trap; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
            )
        tl.inline_asm_elementwise(
            "nanosleep.u32 1000; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        observed = tl.inline_asm_elementwise(
            "ld.acquire.sys.global.u64 $0, [$1];",
            "=l,l",
            [flags + index],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )


def wait_signal(flags: torch.Tensor, index: int, value: int, timeout_ns: int) -> None:
    _wait_signal[(1,)](flags, index, value, timeout_ns, num_warps=1)
