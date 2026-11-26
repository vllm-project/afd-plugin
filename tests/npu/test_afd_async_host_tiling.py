# SPDX-License-Identifier: MIT
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Description: Compile and execute compact-v2 host budgets without CANN or an NPU.
# Create: 2026-11-23
# Note: A host C++ compiler runs the same tiling budget calculations used in the
#       build.
# History: 2026-11-23 add to the Ascend 950 port
"""Compile and execute compact-v2 host budgets without CANN or an NPU.

The dispatch-receive boundary below is derived from its actual kernel buffer
layout. Reducing active AIVs must account for the larger per-core address slice;
otherwise a shape accepted for 48 cores can overflow the same UB with one core.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_UB_BUDGET_BYTES = 192 * 1024
_PROBE_SOURCE = r"""
#include <iostream>
#include "afd_async_host_tiling.h"

int main()
{
    using namespace afd_async::host;
    std::cout << "active_one " << GetActiveAivCount(1) << '\n';
    std::cout << "active_full " << GetActiveAivCount(48) << '\n';
    std::cout << "active_larger_platform " << GetActiveAivCount(96) << '\n';
    std::cout << "source_one " << GetSourceActiveAivCount(96, 1) << '\n';
    std::cout << "source_five " << GetSourceActiveAivCount(48, 5) << '\n';
    std::cout << "source_seven " << GetSourceActiveAivCount(48, 7) << '\n';
    std::cout << "source_at_cap " << GetSourceActiveAivCount(96, 48) << '\n';
    std::cout << "source_over_cap " << GetSourceActiveAivCount(96, 49) << '\n';
    std::cout << "source_no_platform " << GetSourceActiveAivCount(0, 7) << '\n';
    std::cout << "source_bad_batch " << GetSourceActiveAivCount(48, 0) << '\n';
    std::cout << "send_budget_baseline "
        << DispatchSendUbBytes(2, 512, 32, 48, false) << '\n';
    std::cout << "send_budget_shrunk "
        << DispatchSendUbBytes(2, 512, 32, 1, false) << '\n';
    std::cout << "cr_budget_baseline " << CombineRecvUbBytes(2, 512, 32, 48) << '\n';
    std::cout << "cr_budget_shrunk " << CombineRecvUbBytes(2, 512, 32, 1) << '\n';
    std::cout << "recv_one_core "
        << DispatchRecvUbBytes(4, 2, 64, 1, 65535, 2, 262144, false) << '\n';
    std::cout << "recv_full_cores "
        << DispatchRecvUbBytes(4, 2, 64, 48, 65535, 2, 262144, false) << '\n';
    std::cout << "combine_flag_scratch " << CombineSendUbBytes(4, 2, 48, 32) << '\n';
}
"""


@pytest.fixture(scope="module")
def host_budget_probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, int]:
    compiler = next(
        (path for name in ("g++", "clang++", "c++") if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        return pytest.skip(
            "A GNU-compatible C++17 compiler is required for the host budget probe"
        )
    root = Path(__file__).resolve().parents[2]
    project = tmp_path_factory.mktemp("afd_async_host_tiling")
    source = project / "probe.cpp"
    executable = project / ("probe.exe" if os.name == "nt" else "probe")
    source.write_text(_PROBE_SOURCE, encoding="utf-8")
    compilation = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "csrc/npu/ascend_kernels/utils/op_host"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
    execution = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=30
    )
    assert execution.returncode == 0, execution.stdout + execution.stderr
    return {
        key: int(value) for key, value in map(str.split, execution.stdout.splitlines())
    }


def test_compact_statistics_limit_active_cores(host_budget_probe: dict[str, int]):
    assert host_budget_probe["active_one"] == 1
    assert host_budget_probe["active_full"] == 48
    assert host_budget_probe["active_larger_platform"] == 48


def test_receive_slice_budget_rejects_single_core_ub_overflow(
    host_budget_probe: dict[str, int],
):
    # DR Init reserves 64 + 128 + 32 + 64 + 128 + 32 = 448 bytes:
    # header, compact metadata, output counts, counts/flags, token, TP counts.
    # One source owns 131070 address entries. Its slice is 262144 bytes on one
    # AIV, or 5472 aligned bytes per AIV on 48 cores. No CANN data is required.
    assert host_budget_probe["recv_one_core"] == 262592
    assert host_budget_probe["recv_full_cores"] == 5920
    assert host_budget_probe["recv_one_core"] > _UB_BUDGET_BYTES
    assert host_budget_probe["recv_full_cores"] <= _UB_BUDGET_BYTES


def test_combine_shared_scratch_fits_completion_flags(
    host_budget_probe: dict[str, int],
):
    # CS has 192 metadata bytes. For H=32 a token uses 64 bytes, but 48 flags
    # use 192; both reuse one region after payload completion. A token-only
    # bound of 256 would accept UB that cannot publish the completion prefix.
    assert host_budget_probe["combine_flag_scratch"] == 384


def test_paired_source_core_count_is_bounded_by_tokens(
    host_budget_probe: dict[str, int],
):
    assert host_budget_probe["source_one"] == 1
    assert host_budget_probe["source_five"] == 5
    assert host_budget_probe["source_seven"] == 7
    assert host_budget_probe["source_at_cap"] == 48
    assert host_budget_probe["source_over_cap"] == 48
    assert host_budget_probe["source_no_platform"] == 0
    assert host_budget_probe["source_bad_batch"] == 0


def test_source_selection_does_not_redefine_baseline_budget(
    host_budget_probe: dict[str, int],
):
    assert (
        host_budget_probe["send_budget_baseline"]
        > host_budget_probe["send_budget_shrunk"]
    )
    assert (
        host_budget_probe["cr_budget_baseline"] > host_budget_probe["cr_budget_shrunk"]
    )
