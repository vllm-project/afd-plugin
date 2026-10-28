// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <cmath>
#include <torch/extension.h>
#include <torch/library.h>
#include "pytorch_extension/op_api_common.h"
#include "pytorch_extension/NPUBridge.h"
#include "utils/op_kernel/afd_native_mega_workspace.h"
#include "utils/op_kernel/afd_pull_layout.h"
#include "afd_mega_moe_pull_native/op_api/aclnn_afd_mega_moe_pull_native.h"

namespace {
int64_t workspace_size(int64_t rows, int64_t h, int64_t i, int64_t experts, int64_t tp) {
    TORCH_CHECK(rows > 0 && rows <= Cam::NATIVE_MAX_ROWS && h > 0 && h <= 8192 &&
        i > 0 && i <= 8192 && h % 256 == 0 && i % 256 == 0 &&
        experts > 0 && experts <= 128 && tp > 0 && tp <= 384,
        "native Mega workspace dimensions are unsupported");
    return Cam::NativeMegaWorkspace(rows, h, i, experts, tp).bytes;
}

template <bool EXECUTE>
at::Tensor native_mega(const at::Tensor &anchor, const at::Tensor &state,
    const at::Tensor &layer_map, const at::Tensor &scratch,
    at::TensorList w13, at::TensorList w2, at::TensorList s13, at::TensorList s2,
    at::TensorList b13, at::TensorList b2, at::IntArrayRef c,
    c10::string_view group_name, double routed_scale) {
    TORCH_CHECK(c.size() == 14, "native Mega config requires 14 fields");
    for (auto n : c) TORCH_CHECK(n > 0 && n <= 262144, "native Mega config out of range");
    TORCH_CHECK(c[8] == c[4] + c[5] && c[7] >= c[5] && c[7] < c[8] &&
        c[0] % c[9] == 0 && c[5] % c[9] == 0 && c[11] == 1 && c[12] == 1 &&
        std::isfinite(routed_scale), "native Mega requires a valid F rank and per-channel weights");
    Cam::Pull::Shape shape{uint64_t(c[0] / c[9]), uint64_t(c[1]), uint64_t(c[3]),
        uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[9])};
    Cam::Pull::Layout layout{};
    TORCH_CHECK(Cam::Pull::MakeLayout(shape, layout), "native Mega pull shape unsupported");
    auto check = [&](const at::Tensor &x, at::ScalarType dtype) {
        TORCH_CHECK(x.device() == anchor.device() && x.scalar_type() == dtype &&
            x.is_contiguous() && !x.requires_grad(), "native Mega tensor has wrong device, dtype or layout");
    };
    check(anchor, at::kBFloat16);
    TORCH_CHECK(anchor.numel() == 1, "native Mega requires a one-element anchor");
    check(state, at::kHalf);
    check(layer_map, at::kLong);
    TORCH_CHECK(layer_map.dim() == 1 && layer_map.numel() > 0 && layer_map.numel() <= 65536,
        "native Mega layer map must cover model layer IDs");
    TORCH_CHECK(state.dim() == 1 && state.numel() >= int64_t(4 *
        Cam::Pull::StateWords(c[5] / c[9], layer_map.numel())), "native Mega pull state is missing");
    check(scratch, at::kByte);
    TORCH_CHECK(scratch.dim() == 1 && scratch.numel() == workspace_size(c[10], c[1], c[2], c[6], c[9]),
        "native Mega scratch has the wrong size");
    const std::vector<std::pair<at::TensorList, at::ScalarType>> lists{
        {w13, at::kInt}, {w2, at::kInt}, {s13, at::kLong}, {s2, at::kLong},
        {b13, at::kFloat}, {b2, at::kFloat}};
    for (const auto &list : lists) {
        TORCH_CHECK(int64_t(list.first.size()) == c[13], "native Mega layer list length mismatch");
        for (const auto &tensor : list.first) check(tensor, list.second);
    }
    for (int64_t l = 0; l < c[13]; ++l) {
        if constexpr (EXECUTE) {
            TORCH_CHECK(vllm_ascend::NPUBridge::GetNpuStorageImplDesc(w13[l]).npu_format_ == ACL_FORMAT_FRACTAL_NZ &&
                vllm_ascend::NPUBridge::GetNpuStorageImplDesc(w2[l]).npu_format_ == ACL_FORMAT_FRACTAL_NZ,
                "native Mega requires physically NZ packed weights");
        }
        TORCH_CHECK(w13[l].sizes() == at::IntArrayRef({c[6], c[1], c[2] / 4}) &&
            w2[l].sizes() == at::IntArrayRef({c[6], c[2], c[1] / 8}), "native Mega packed weights mismatch");
        TORCH_CHECK(s13[l].numel() == c[6] * 2 * c[2] && s2[l].numel() == c[6] * c[1] &&
            b13[l].numel() == c[6] * 2 * c[2] && b2[l].numel() == c[6] * c[1],
            "native Mega scale/compensation dimensions mismatch");
    }
    auto completion = at::empty({1}, anchor.options().dtype(at::kLong));
    if constexpr (EXECUTE) {
        std::string group(group_name.data(), group_name.size());
        auto group_ptr = group.data();
        EXEC_NPU_CMD(aclnnAfdMegaMoePullNative, anchor, state, layer_map, scratch,
            w13, w2, s13, s2, b13, b2, c, group_ptr, routed_scale, completion);
    }
    return completion;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(afd_ascend, m) {
    m.def("afd_mega_moe_pull_workspace_size(int rows, int hidden, int intermediate, int experts, int tp) -> int",
          &workspace_size);
    m.def("afd_mega_moe_pull_native(Tensor anchor, Tensor(a!) state, Tensor layer_map, Tensor(b!) scratch, "
          "Tensor[] w13, Tensor[] w2, Tensor[] s13, Tensor[] s2, Tensor[] b13, Tensor[] b2, "
          "int[] config, str group_name, float routed_scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, m) {
    m.impl("afd_mega_moe_pull_native", &native_mega<true>);
}
TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, m) {
    m.impl("afd_mega_moe_pull_native", &native_mega<true>);
}
TORCH_LIBRARY_IMPL(afd_ascend, Meta, m) {
    m.impl("afd_mega_moe_pull_native", &native_mega<false>);
}
