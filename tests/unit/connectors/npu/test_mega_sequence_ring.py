# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
"""Exercise the device's actual arithmetic on the host, independently of CANN."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SUPPORT = Path(__file__).resolve().parents[4] / "csrc/npu/ascend_kernels/mega_utils"


class SequenceRingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "requires a host C++ compiler")
    def test_device_predicate_over_four_laps(self):
        source = r"""
#define __aicore__
#include "mega_sequence_ring.h"
#include <cassert>
using namespace AfdAsync;
int main()
{
    const int limit = SPLIT_SEQUENCE_MAX;
    auto wire = [limit](int n) { return (n - 1) % limit + 1; };
    for (int n = 1; n <= 4 * limit; ++n) {
        for (bool previous : {false, true}) {
            const int target = n - int(previous);
            assert(SplitSequenceReached(0, wire(n), previous) ==
                   (previous && wire(n) == 1));
            for (int observed = 1; observed <= 5 * limit; ++observed) {
                const int skew = observed - target;
                if (skew < -limit / 2 || skew > limit / 2) { continue; }
                assert(SplitSequenceReached(wire(observed), wire(n), previous) ==
                       (skew >= 0 && skew < limit / 2));
            }
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "sequence_ring"
            subprocess.run(
                [
                    "c++",
                    "-std=c++11",
                    "-include",
                    "initializer_list",
                    "-I",
                    str(SUPPORT),
                    "-x",
                    "c++",
                    "-",
                    "-o",
                    str(exe),
                ],
                input=source,
                text=True,
                check=True,
            )
            subprocess.run([str(exe)], check=True)
