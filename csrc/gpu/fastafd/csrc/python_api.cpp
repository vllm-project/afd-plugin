// SPDX-License-Identifier: MIT
// Adapted from FastAFD's DeepGEMM snapshot (csrc/python_api.cpp at FastAFD
// 3c7161949310b6d59d6b4cf9bf997a4935c8113b). Registers only the APIs the M2N
// MegaMoE path needs; get_token_alignment_for_mega_moe mirrors the
// csrc/apis/mega.hpp binding of that snapshot.
#include <pybind11/pybind11.h>
#include <torch/python.h>

// The API headers rely on utilities that the dropped APIs used to include
// first; the JIT compiler header brings them in the original order.
#include "jit/compiler.hpp"

#include "apis/layout.hpp"
#include "apis/mega_m2n.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C_fastafd
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FastAFD M2N MegaMoE kernels for the AFD plugin";

    deep_gemm::layout::register_apis(m);
    deep_gemm::mega_m2n::register_apis(m);
    deep_gemm::runtime::register_apis(m);
    m.def("get_token_alignment_for_mega_moe", [] {
        return deep_gemm::layout::kLCMCandidateBlockM;
    });
}
