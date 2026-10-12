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


def test_a2e_mxfp8_meta_transport_contracts() -> None:
    _load_ops_or_skip()
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e4m3 is None or e8m0 is None:
        _skip("PyTorch FP8 E4M3/E8M0 dtypes are required for MXFP8 contract tests")

    for fp8_dtype in (e4m3, torch.float8_e5m2):
        batch_size = 4
        hidden_size = 65
        topk = 1
        x = torch.empty((batch_size, hidden_size), device="meta", dtype=fp8_dtype)
        x_scale = torch.empty(
            (batch_size, (hidden_size + 31) // 32), device="meta", dtype=e8m0
        )
        output = torch.ops.afd_ascend.a2e(
            x,
            None,
            None,
            batch_size,
            hidden_size,
            topk,
            1,
            1,
            0,
            "meta_group",
            4,
            0,
            x_scale,
            4,
        )
        assert len(output) == 6
        _assert_tensor_spec(output[0], (batch_size, hidden_size), fp8_dtype, "meta")
        _assert_tensor_spec(output[1], (batch_size, topk), torch.int32, "meta")
        _assert_tensor_spec(output[2], (batch_size, topk), torch.float32, "meta")
        _assert_tensor_spec(output[3], (1,), torch.int32, "meta")
        _assert_tensor_spec(output[4], (batch_size,), torch.bool, "meta")
        _assert_tensor_spec(
            output[5], (batch_size, (hidden_size + 31) // 32), e8m0, "meta"
        )
        receiver_output = torch.ops.afd_ascend.a2e(
            torch.empty((0, hidden_size), device="meta", dtype=fp8_dtype),
            None,
            None,
            batch_size,
            hidden_size,
            topk,
            1,
            1,
            0,
            "meta_group",
            4,
            0,
            quant_mode=4,
        )
        assert len(receiver_output) == 6
        _assert_tensor_spec(
            receiver_output[0], (batch_size, hidden_size), fp8_dtype, "meta"
        )
        _assert_tensor_spec(
            receiver_output[5], (batch_size, (hidden_size + 31) // 32), e8m0, "meta"
        )
        attention_output = torch.ops.afd_ascend.a2e(
            x,
            None,
            None,
            batch_size,
            hidden_size,
            topk,
            1,
            1,
            1,
            "meta_group",
            4,
            0,
            x_scale=x_scale,
            quant_mode=4,
        )
        assert len(attention_output) == 6
        _assert_tensor_spec(attention_output[0], (1, 1), fp8_dtype, "meta")
        _assert_tensor_spec(attention_output[5], (1, 1), e8m0, "meta")
        routed_output = torch.ops.afd_ascend.a2e(
            x,
            torch.empty((batch_size, topk), device="meta", dtype=torch.int32),
            torch.empty((batch_size, topk), device="meta", dtype=torch.float32),
            batch_size,
            hidden_size,
            topk,
            1,
            1,
            1,
            "meta_group",
            4,
            1,
            x_scale=x_scale,
            quant_mode=4,
        )
        assert len(routed_output) == 6
        _assert_tensor_spec(routed_output[5], (1, 1), e8m0, "meta")


def test_a2e_mxfp8_meta_rejects_invalid_inputs() -> None:
    _load_ops_or_skip()
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e4m3 is None or e8m0 is None:
        _skip("PyTorch FP8 E4M3/E8M0 dtypes are required for MXFP8 contract tests")
    x = torch.empty((4, 65), device="meta", dtype=e4m3)
    scale = torch.empty((4, 3), device="meta", dtype=e8m0)
    invalid_inputs = (
        (x.to(torch.bfloat16), scale, 4),
        (x.to(torch.uint8), scale, 4),
        (x, None, 0),
        (x.to(torch.float8_e5m2), None, 0),
        (x, scale.to(torch.float32), 4),
        (x, scale[:, :2], 4),
        (x, None, 4),
        (torch.empty((65, 4), device="meta", dtype=e4m3).t(), scale, 4),
        (x, torch.empty((3, 4), device="meta", dtype=e8m0).t(), 4),
        (x, scale, 3),
    )
    for invalid_x, invalid_scale, mode in invalid_inputs:
        try:
            torch.ops.afd_ascend.a2e(
                invalid_x,
                None,
                None,
                4,
                65,
                1,
                1,
                1,
                1,
                "meta_group",
                4,
                0,
                x_scale=invalid_scale,
                quant_mode=mode,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"A2E accepted invalid MXFP8 inputs for mode {mode}")

    empty_x = torch.empty((0, 65), device="meta", dtype=e4m3)
    empty_scale = torch.empty((0, 3), device="meta", dtype=e8m0)
    receiver_mismatches = ((empty_x, scale), (x, empty_scale))
    for invalid_x, invalid_scale in receiver_mismatches:
        try:
            torch.ops.afd_ascend.a2e(
                invalid_x,
                None,
                None,
                4,
                65,
                1,
                1,
                1,
                0,
                "meta_group",
                4,
                0,
                x_scale=invalid_scale,
                quant_mode=4,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "A2E accepted mismatched receiver activation and scale rows"
            )

    invalid_topologies = ((1, 2, 1, 1), (33, 33, 0, 3))
    for expert_ranks, attention_ranks, rank, aiv_num in invalid_topologies:
        try:
            torch.ops.afd_ascend.a2e(
                x,
                None,
                None,
                4,
                65,
                1,
                expert_ranks,
                attention_ranks,
                rank,
                "meta_group",
                aiv_num,
                0,
                x_scale=scale,
                quant_mode=4,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "A2E accepted an unsupported MXFP8 topology or core count"
            )

    try:
        torch.ops.afd_ascend.a2e(
            x,
            None,
            None,
            4,
            65,
            1,
            1,
            1,
            1,
            "meta_group",
            2,
            1,
            x_scale=scale,
            quant_mode=4,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("A2E accepted gate=1 with fewer than 3 AIV cores")


def test_a2e_mxfp8_meta_packet_boundaries() -> None:
    _load_ops_or_skip()
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e4m3 is None or e8m0 is None:
        _skip("PyTorch FP8 E4M3/E8M0 dtypes are required for MXFP8 contract tests")

    # Packet alignment puts these adjacent hidden sizes on either side of
    # int32 addressing. Empty receiver inputs avoid allocating payload storage.
    valid_hidden_size = 2082408192
    invalid_hidden_size = 2082408352
    output = torch.ops.afd_ascend.a2e(
        torch.empty((0, valid_hidden_size), device="meta", dtype=e4m3),
        None,
        None,
        1,
        valid_hidden_size,
        1,
        1,
        1,
        0,
        "meta_group",
        4,
        0,
        quant_mode=4,
    )
    assert len(output) == 6
    _assert_tensor_spec(output[0], (1, valid_hidden_size), e4m3, "meta")
    _assert_tensor_spec(output[5], (1, (valid_hidden_size + 31) // 32), e8m0, "meta")
    try:
        torch.ops.afd_ascend.a2e(
            torch.empty((0, invalid_hidden_size), device="meta", dtype=e4m3),
            None,
            None,
            1,
            invalid_hidden_size,
            1,
            1,
            1,
            0,
            "meta_group",
            4,
            0,
            quant_mode=4,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("A2E accepted a packet exceeding int32 addressing")

    # Gate-1 routing IDs must fit their per-core UB budget.
    topk_within_ub = 48896
    topk_exceeding_ub = topk_within_ub + 1
    receiver_x = torch.empty((0, 65), device="meta", dtype=e4m3)
    output = torch.ops.afd_ascend.a2e(
        receiver_x,
        None,
        None,
        1,
        65,
        topk_within_ub,
        1,
        1,
        0,
        "meta_group",
        4,
        1,
        quant_mode=4,
    )
    assert len(output) == 6
    try:
        torch.ops.afd_ascend.a2e(
            receiver_x,
            None,
            None,
            1,
            65,
            topk_exceeding_ub,
            1,
            1,
            0,
            "meta_group",
            4,
            1,
            quant_mode=4,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("A2E accepted gate=1 routing IDs beyond UB capacity")


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


def test_a2e_mxfp8_runtime_transport() -> None:
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        _skip("set AFD_RUN_ASCEND_OP_RUNTIME=1 to run the two-rank NPU kernel test")
    _load_ops_or_skip()
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise AssertionError("MXFP8 transport requires exactly two torchrun ranks")
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", str(rank))))
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    group_ep = _get_default_hccl_group_name(rank)
    batch_size, hidden_size, topk = 4, 65, 2
    scale_width = (hidden_size + 31) // 32
    # Construct storage bytes directly: this test performs no quantization.
    expected_x = (torch.arange(batch_size * hidden_size) % 127).to(torch.uint8)
    expected_x = expected_x.reshape(batch_size, hidden_size)
    expected_scale = (torch.arange(batch_size * scale_width) % 32 + 112).to(torch.uint8)
    expected_scale = expected_scale.reshape(batch_size, scale_width)
    for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if rank == 1:
            x = expected_x.to("npu").view(fp8_dtype)
            x_scale = expected_scale.to("npu").view(torch.float8_e8m0fnu)
        else:
            x = torch.empty((0, hidden_size), dtype=fp8_dtype, device="npu")
            x_scale = torch.empty(
                (0, scale_width), dtype=torch.float8_e8m0fnu, device="npu"
            )
        expert_ids = torch.zeros((batch_size, topk), dtype=torch.int32, device="npu")
        scales = torch.full((batch_size, topk), 1.0 / topk, device="npu")
        for compute_gate in (0, 1):
            dist.barrier()
            output = torch.ops.afd_ascend.a2e(
                x,
                expert_ids,
                scales,
                batch_size,
                hidden_size,
                topk,
                1,
                1,
                rank,
                group_ep,
                4,
                compute_gate,
                x_scale=x_scale,
                quant_mode=4,
            )
            torch.npu.synchronize()
            assert len(output) == 6
            assert output[0].dtype == fp8_dtype
            if rank == 0:
                assert torch.equal(output[0].view(torch.uint8).cpu(), expected_x)
                assert torch.equal(output[5].view(torch.uint8).cpu(), expected_scale)
            else:
                _assert_tensor_spec(output[0], (1, 1), fp8_dtype, "npu")
                _assert_tensor_spec(output[5], (1, 1), torch.float8_e8m0fnu, "npu")
            dist.barrier()


def _run_as_script() -> int:
    failures: list[tuple[str, BaseException]] = []
    skipped: list[tuple[str, str]] = []

    for test_func in (
        test_a2e_e2a_meta_contracts,
        test_a2e_mxfp8_meta_transport_contracts,
        test_a2e_mxfp8_meta_rejects_invalid_inputs,
        test_a2e_mxfp8_meta_packet_boundaries,
        test_a2e_e2a_runtime_roundtrip,
        test_a2e_mxfp8_runtime_transport,
    ):
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

    for name, error in failures:
        print(f"FAIL {name}: {error!r}", flush=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_as_script())
