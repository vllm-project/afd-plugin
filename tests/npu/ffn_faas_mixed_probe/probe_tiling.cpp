// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// CPU-only platform/tiling query. No ACL runtime initialization or device call.
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "tiling/tiling_api.h"
#include "probe_layout.h"

using namespace FaaSProbe;

namespace {
void Require(bool value, const char *reason)
{
    if (!value) {
        throw std::runtime_error(reason);
    }
}
}  // namespace

int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "usage: probe_tiling OUTPUT_BINARY");
        auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance(PROBE_SOC_VERSION);
        Require(platform != nullptr, "CPU platform descriptor unavailable");
        uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        constexpr uint64_t DOUBLE_L0A = TILE_M * 128ULL * 2 * 2;
        constexpr uint64_t DOUBLE_L0B = TILE_N * 128ULL * 2 * 2;
        constexpr uint64_t VECTOR_UB = 2ULL * I * 2 + 3ULL * I * sizeof(float);
        Require(DOUBLE_L0A <= l0a && DOUBLE_L0B <= l0b, "L0A/B candidate exceeds platform");
        Require(VECTOR_UB <= ub, "AIV UB exceeds platform");
        AscendC::tiling::TCubeTiling tilings[2]{};
        const uint32_t widths[] = {2 * I, H};
        const uint32_t reductions[] = {H, I};
        std::cout << "{\"device_execution\":false,\"target\":\"" << PROBE_SOC_VERSION << "\",";
        std::cout << "\"l1_bytes\":" << l1 << ",\"l0a_bytes\":" << l0a
                  << ",\"l0b_bytes\":" << l0b << ",\"l0c_bytes\":" << l0c
                  << ",\"ub_bytes\":" << ub << ",\"vector_ub_bytes\":" << VECTOR_UB
                  << ",\"scratch_bytes\":" << SCRATCH_BYTES
                  << ",\"platform_lib_workspace_bytes\":" << platform->GetLibApiWorkSpaceSize()
                  << ",\"projections\":[";
        for (uint32_t projection = 0; projection < 2; ++projection) {
            matmul_tiling::MatmulApiTiling tile(*platform);
            using matmul_tiling::TPosition;
            using matmul_tiling::CubeFormat;
            using matmul_tiling::DataType;
            tile.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
            tile.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, true);
            tile.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16);
            tile.SetBias(false);
            tile.SetOrgShape(TILE_M, widths[projection], reductions[projection]);
            tile.SetShape(TILE_M, TILE_N, reductions[projection]);
            tile.SetFixSplit(TILE_M, TILE_N, 128);
            tile.SetBufferSpace(l1, l0c, ub);
            Require(tile.GetTiling(tilings[projection]) == 0, "Matmul API tiling rejected shape");
            matmul_tiling::SysTilingTempBufSize used;
            Require(MatmulGetTmpBufSizeV2(tilings[projection], used) == 0, "workspace query failed");
            Require(used.l1Size <= l1 && used.l0cSize <= l0c && used.ubSize <= ub,
                    "Matmul API resource demand exceeds platform");
            if (projection != 0) {
                std::cout << ",";
            }
            std::cout << "{\"full_n\":" << widths[projection]
                      << ",\"k\":" << reductions[projection]
                      << ",\"api_l1_bytes\":" << used.l1Size
                      << ",\"api_l0c_bytes\":" << used.l0cSize
                      << ",\"api_ub_bytes\":" << used.ubSize << "}";
        }
        std::ofstream output(argv[1], std::ios::binary);
        output.write(reinterpret_cast<const char *>(tilings), sizeof(tilings));
        Require(output.good(), "tiling output write failed");
        std::cout << "],\"tiling_bytes\":" << sizeof(tilings) << "}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
