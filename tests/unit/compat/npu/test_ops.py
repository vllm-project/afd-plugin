# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Smoke tests for the A2E/E2A Ascend custom operators.

Default pytest/script execution validates that the extension registers the ops
and that their Meta kernels preserve the expected shape/dtype contracts.

The real NPU kernel path needs a two-rank HCCL group.  Run it explicitly with:

    AFD_RUN_ASCEND_OP_RUNTIME=1 torchrun --standalone --nproc_per_node=2 \
        tests/unit/compat/npu/test_ops.py

The A5 1A1F mp.spawn hardware sample is:

    python tests/unit/compat/npu/a2e_e2a_a5.py
"""

from __future__ import annotations

import os
import sys

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

try:
    import torch
except ModuleNotFoundError:
    torch = None


_SCRIPT_MODE = __name__ == "__main__"


class _SkipInScriptError(RuntimeError):
    pass


def _skip(reason: str) -> None:
    if _SCRIPT_MODE:
        raise _SkipInScriptError(reason)

    import pytest

    pytest.skip(reason)


def _load_ops_or_skip() -> None:
    if torch is None:
        _skip("PyTorch is required for A2E/E2A custom op smoke tests")
    try:
        ensure_afd_ascend_ops_loaded()
    except RuntimeError as exc:
        _skip(str(exc))


def _assert_tensor_spec(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device_type: str,
) -> None:
    assert tuple(tensor.shape) == shape
    assert tensor.dtype is dtype
    assert tensor.device.type == device_type


def test_a2e_e2a_meta_contracts() -> None:
    _load_ops_or_skip()

    batch_size = 4
    hidden_size = 16
    topk = 1
    expert_rank_size = 1
    attention_rank_size = 1

    expert_x = torch.empty(
        (batch_size, hidden_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    expert_a2e = torch.ops.afd_ascend.a2e(
        expert_x,
        None,
        None,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        0,
        "meta_group",
        1,
        0,
    )
    assert len(expert_a2e) == 5
    _assert_tensor_spec(
        expert_a2e[0],
        (batch_size, hidden_size),
        torch.bfloat16,
        "meta",
    )
    _assert_tensor_spec(expert_a2e[1], (batch_size, topk), torch.int32, "meta")
    _assert_tensor_spec(expert_a2e[2], (batch_size, topk), torch.float32, "meta")
    _assert_tensor_spec(expert_a2e[3], (1,), torch.int32, "meta")
    _assert_tensor_spec(expert_a2e[4], (batch_size,), torch.bool, "meta")

    attention_x = torch.empty(
        (batch_size, hidden_size),
        device="meta",
        dtype=torch.bfloat16,
    )
    attention_a2e = torch.ops.afd_ascend.a2e(
        attention_x,
        None,
        None,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        1,
        "meta_group",
        1,
        0,
    )
    _assert_tensor_spec(attention_a2e[0], (1, 1), torch.bfloat16, "meta")
    _assert_tensor_spec(attention_a2e[1], (1, 1), torch.int32, "meta")
    _assert_tensor_spec(attention_a2e[2], (1, 1), torch.float32, "meta")
    _assert_tensor_spec(attention_a2e[3], (1,), torch.int32, "meta")
    _assert_tensor_spec(attention_a2e[4], (1,), torch.bool, "meta")

    expert_e2a = torch.ops.afd_ascend.e2a(
        expert_a2e[0],
        expert_a2e[3],
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        0,
        "meta_group",
        1,
    )
    _assert_tensor_spec(expert_e2a, (1, 1), torch.bfloat16, "meta")

    attention_e2a = torch.ops.afd_ascend.e2a(
        attention_x,
        attention_a2e[3],
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        1,
        "meta_group",
        1,
    )
    _assert_tensor_spec(
        attention_e2a,
        (batch_size, hidden_size),
        torch.bfloat16,
        "meta",
    )


def _get_default_hccl_group_name(rank: int) -> str:
    import torch.distributed as dist

    group = dist.distributed_c10d._get_default_group()
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def test_a2e_e2a_runtime_roundtrip() -> None:
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        _skip("set AFD_RUN_ASCEND_OP_RUNTIME=1 to run the two-rank NPU kernel test")

    _load_ops_or_skip()

    try:
        import torch.distributed as dist
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise AssertionError("torch_npu and torch.distributed are required") from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise AssertionError(
            "runtime A2E/E2A smoke requires torchrun with exactly 2 ranks",
        )

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    rank = int(os.environ["RANK"])
    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    assert dist.get_world_size() == 2
    group_ep = _get_default_hccl_group_name(rank)

    batch_size = 4
    hidden_size = 16
    topk = 1
    expert_rank_size = 1
    attention_rank_size = 1
    aiv_num = int(os.environ.get("AFD_A2E_E2A_AIV_NUM", "1"))

    expected = (
        torch.arange(batch_size * hidden_size, dtype=torch.float32, device="npu")
        .reshape(batch_size, hidden_size)
        .to(torch.bfloat16)
    )
    x = expected if rank == 1 else torch.zeros_like(expected)

    dist.barrier()
    a2e_out = torch.ops.afd_ascend.a2e(
        x,
        None,
        None,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        rank,
        group_ep,
        aiv_num,
        0,
    )
    torch.npu.synchronize()
    dist.barrier()

    expand_x, _, _, atten_batch_size, _ = a2e_out
    if rank == 0:
        torch.testing.assert_close(
            expand_x.cpu().float(),
            expected.cpu().float(),
            rtol=0,
            atol=0,
        )

    e2a_input = expand_x if rank == 0 else torch.empty_like(expected)
    dist.barrier()
    x_roundtrip = torch.ops.afd_ascend.e2a(
        e2a_input,
        atten_batch_size,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        rank,
        group_ep,
        aiv_num,
    )
    torch.npu.synchronize()
    dist.barrier()

    if rank == 1:
        torch.testing.assert_close(
            x_roundtrip.cpu().float(),
            expected.cpu().float(),
            rtol=0,
            atol=0,
        )

    dist.barrier()


def _run_as_script() -> int:
    failures: list[tuple[str, BaseException]] = []
    skipped: list[tuple[str, str]] = []

    for test_func in (test_a2e_e2a_meta_contracts, test_a2e_e2a_runtime_roundtrip):
        try:
            test_func()
        except _SkipInScriptError as exc:
            skipped.append((test_func.__name__, str(exc)))
        except BaseException as exc:  # noqa: BLE001
            failures.append((test_func.__name__, exc))
        else:
            print(f"PASS {test_func.__name__}", flush=True)

    for name, reason in skipped:
        print(f"SKIP {name}: {reason}", flush=True)

    for name, exc in failures:
        print(f"FAIL {name}: {exc!r}", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_as_script())
