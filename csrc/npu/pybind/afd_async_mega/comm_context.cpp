/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_context.cpp
 * \brief comm_context implementation supporting both KFC and HCCL Channel modes.
 *
 */

#include <torch/extension.h>
#include <acl/acl_rt.h>
#include <limits>
#include <utility>
#include "hccl_common.h"

namespace op_api {

// ======================== Common constants ========================

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint32_t HCCL_CONTEXT_TAG_MAX_LEN = 255;

constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr int64_t DEFAULT_RANK_NUM_PER_SERVER = 2;

enum class TopoType : uint32_t {
    INTRA_SUPER_NODE = 0, // 超节点内通信（默认）
    CROSS_SUPER_NODE = 1, // 跨超节点通信
};

struct RankLinkInfo {
    CommProtocol protocol;
    uint32_t layer;
};

struct LayerRanks {
    uint32_t layer;
    std::vector<uint32_t> ranks;
};

// ======================== 通信上下文 ==========================
struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0; // 通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE] = {};
    ChannelHandle hcommHandle_[HCCL_MAX_RANK_SIZE] = {}; // ROCE或者URMA通信所需句柄
};

// ======================== Common Types and Utilities ========================

enum class BackendMode : uint8_t {
    UNINITIALIZED,
    KFC,
    CHANNEL
};

static const char *GetSocName()
{
    static const char *socName = aclrtGetSocName();
    return socName;
}

BackendMode ResolveBackend(const py::object &backend)
{
    if (py::isinstance<py::str>(backend)) {
        auto mode = backend.cast<std::string>();
        if (mode == "channel")
            return BackendMode::CHANNEL;
        if (mode == "kfc")
            return BackendMode::KFC;
        TORCH_CHECK(false, "backend string must be 'kfc' or 'channel', got '", mode, "'");
    }

    if (py::isinstance<py::dict>(backend)) {
        auto dict = backend.cast<py::dict>();
        TORCH_CHECK(dict.size() > 0, "backend dict must not be empty");

        const char *socName = GetSocName();
        TORCH_CHECK(socName != nullptr, "aclrtGetSocName returned nullptr");

        for (auto item : dict) {
            std::string key = py::cast<std::string>(item.first);
            std::string value = py::cast<std::string>(item.second);
            if (value == "channel" || value == "kfc") {
                if (std::strstr(socName, key.c_str()) != nullptr) {
                    ASCEND_LOGI("Matched SoC name '%s' with key '%s', using backend '%s'", socName, key.c_str(),
                                value.c_str());
                    return value == "channel" ? BackendMode::CHANNEL : BackendMode::KFC;
                }
            } else {
                TORCH_CHECK(false, "backend dict value must be 'kfc' or 'channel', got '", value, "'");
            }
        }

        TORCH_CHECK(false, "No matching SoC name found for '", socName, "' in backend dict");
    }

    TORCH_CHECK(false, "backend must be a string ('kfc' or 'channel') or a dict");
}

static void CopyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    at::Tensor hostContext =
        at::from_blob(const_cast<CommContext *>(&context), {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostContext);
}

// ======================== KFC Mode ========================
class KfcContextBuilder {
public:
    void Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor,
               void **localDeviceBuffer)
    {
        CommContext mc2ContextHost;
        GetMc2Context(mc2ContextHost, worldSize, cclBufferSize, group.c_str());
        TORCH_CHECK(mc2ContextHost.epRankId < HCCL_MAX_RANK_SIZE, "Invalid local rank id: ", mc2ContextHost.epRankId);
        *localDeviceBuffer = reinterpret_cast<void *>(mc2ContextHost.epHcclBuffer_[mc2ContextHost.epRankId]);

        CopyContextToTensor(mc2ContextHost, contextTensor);
    }

private:
    void CollectRankBuffers(HcclComm &comm, int64_t worldSize, int64_t &cclBufferSize, CommContext &mc2Context)
    {
        uint32_t ctxIndex = 0;
        uint32_t rankId = 0;
        auto hcclRet = HcclGetRankIdFunc(comm, &rankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclGetRankIdFunc failed, ret: ", hcclRet);
        mc2Context.epRankId = rankId;
        const char *socName = GetSocName();
        void *remoteAddr = nullptr;
        uint64_t commSize = 0;
        if (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr && worldSize > 8) {
            HcclResult ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
            TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
            cclBufferSize = static_cast<int64_t>(commSize);
            mc2Context.epHcclBuffer_[rankId] = (uint64_t)remoteAddr;
            return;
        }
        for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
            commSize = 0;
            remoteAddr = nullptr;
            HcclResult ret;
            if (rankId == remoteRankId) {
                ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
                cclBufferSize = commSize;
            } else {
                ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBufFunc(comm, remoteRankId, &remoteAddr, &commSize));
            }
            TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
            mc2Context.epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
        }
    }

    void CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize, const char *groupName,
                           std::string algConfig, uint32_t opType, CommContext &mc2Context)
    {
        HcclResult ret =
            static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfigFunc(opArgs, const_cast<char *>(algConfig.c_str())));
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclCommGetHandleWithNameFunc(groupName, &commHandle));
        TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);
        void *opsResCtx;
        ret = static_cast<HcclResult>(HcclCreateOpResCtxFunc(commHandle, opType, opArgs, &opsResCtx));
        TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);
        mc2Context.kfcContextAddr = (uint64_t)opsResCtx;

        uint32_t rankId = 0;
        uint32_t worldSizeHccl = 0;
        ret = static_cast<HcclResult>(HcclGetRankSizeFunc(commHandle, &worldSizeHccl));
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclGetRankIdFunc(commHandle, &rankId));
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);
        TORCH_CHECK(rankId < worldSizeHccl, "rankId:", rankId, " worldSizeHccl:", worldSizeHccl,
                    " worldSize:", worldSize);
        TORCH_CHECK(worldSize == worldSizeHccl, "worldSize:", worldSize, " != worldSizeHccl:", worldSizeHccl);
    }

    void GetMc2Context(CommContext &mc2ContextHost, int64_t worldSize, int64_t &cclBufferSize, const char *groupStr)
    {
        InitHcclFunctions();
        void *opArgs = nullptr;
        HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgsFunc(&opArgs));
        TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);
        uint8_t commEngine = COMM_ENGINE_AIV;
        ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngineFunc(opArgs, (uint8_t)commEngine));
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);
        HcclComm commHandle;
        const char *socName = GetSocName();
        const bool is910B = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
        const bool isMultiServer = worldSize > 8;
        const std::string algConfig =
            is910B && isMultiServer ? "BatchWrite=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
        const uint32_t opType = is910B && isMultiServer ? 18 : 8; // 18: BatchWrite, 8: AllToAll
        CreateHcclContext(commHandle, opArgs, worldSize, groupStr, algConfig, opType, mc2ContextHost);
        ret = static_cast<HcclResult>(HcclKfcFreeOpArgsFunc(opArgs));
        TORCH_CHECK(ret == 0, "getHcclKfcFreeOpArgs failed, ret:", ret);
        CollectRankBuffers(commHandle, worldSize, cclBufferSize, mc2ContextHost);
    }
};

// ======================== HCCL Channel Mode ========================

class HcclChannelContextBuilder {
public:
    void Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor,
               const std::string &commAlg, const std::string &opName, int64_t customCclBufferSize = 0,
               const py::object &customCclBufferSizeResolver = py::none(), void **customDeviceBuffer = nullptr,
               HcclMemHandle *customMemHandle = nullptr)
    {
        ASCEND_LOGI("Start to get CommContext Tensor, group: %s", group.c_str());
        InitHcclEngineCtxFunctions();
        customCclBufferSize_ = customCclBufferSize;
        customDeviceBuffer_ = customDeviceBuffer;
        customMemHandle_ = customMemHandle;

        HcclComm hcclHandle;
        AcquireHcclHandle(group, hcclHandle);

        CommProtocol protocol;
        if (commAlg == "urma") {
            protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        } else {
            protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        }
        GetCommProtocol(hcclHandle, protocol);
        rankNumPerServer_ = rankNumPerUbDomain_;
        ResolveCustomCclBufferSize(worldSize, customCclBufferSizeResolver);

        CommContext commContextStruct;
        BuildContext(hcclHandle, group, opName, protocol, commContextStruct, cclBufferSize);
        commContextStruct.rankSizePerServer = rankNumPerServer_;

        CopyContextToTensor(commContextStruct, contextTensor);
        ASCEND_LOGI("Get CommContext Tensor Success, group: %s, ccl_buffer_size: %ld", group.c_str(), cclBufferSize);
    }

    void AcquireHcclHandle(const std::string &group, HcclComm &hcclHandle)
    {
        auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &hcclHandle);
        TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group.c_str(), ", ret: ", aclnnRet);
        ASCEND_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
    }

    void ResolveCustomCclBufferSize(int64_t worldSize, const py::object &customCclBufferSizeResolver)
    {
        if (topoType_ != TopoType::CROSS_SUPER_NODE || customCclBufferSizeResolver.is_none()) {
            return;
        }

        int64_t serverNum = worldSize / rankNumPerServer_;
        customCclBufferSize_ = customCclBufferSizeResolver(serverNum).cast<int64_t>();
        ASCEND_LOGI("Recalculated cross-server CCL buffer size: %ld bytes, serverNum: %ld", customCclBufferSize_,
                    serverNum);
    }

    void BuildContext(const HcclComm &hcclHandle, const std::string &group, const std::string &opName,
                      const CommProtocol &protocol, CommContext &commContextStruct, int64_t &cclBufferSize)
    {
        std::string mc2ContextTag = std::string(group) + opName;
        TORCH_CHECK(mc2ContextTag.size() <= HCCL_CONTEXT_TAG_MAX_LEN, "Mc2ContextTag is too long, max size is ",
                    HCCL_CONTEXT_TAG_MAX_LEN, ", got ", mc2ContextTag.size());

        CommEngine engine = CommEngine::COMM_ENGINE_AIV;
        void *ctx = nullptr;
        uint64_t hcclBuffSize = 0;

        GetOrCreateContext(hcclHandle, mc2ContextTag, engine, protocol, ctx, hcclBuffSize, commContextStruct);

        cclBufferSize = hcclBuffSize;
    }

    void GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol)
    {
        ASCEND_LOGI("Start to get HCCL communication protocol");
        uint32_t layerNum = 0;
        uint32_t *layerList = nullptr;
        auto ret = HcclRankGraphGetLayersFunc(commHandle, &layerList, &layerNum);
        TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", ret);
        TORCH_CHECK(layerList != nullptr && layerNum > 0, "Get HCCL layers returned empty layer list");

        if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
            GetRankSizePerServer(commHandle, rankNumPerUbDomain_);
            return;
        }

        CheckProtocolSupport(commHandle, layerList, layerNum, protocol);
    }

    // Protocol discovery may probe layers that do not connect the rank pair; treat those probes as unsupported.
    bool SupportsProtocol(const HcclComm &commHandle, uint32_t layerId, uint32_t srcRankId, uint32_t dstRankId,
                          const CommProtocol &protocol) const
    {
        CommLink *linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        if (hcclRet != HCCL_SUCCESS || linksList == nullptr || netLinkNum == 0) {
            ASCEND_LOGW("Get HCCL links failed when checking protocol support, srcRankId: %u, dstRankId: %u, "
                        "layerId: %u, ret: %d, netLinkNum: %u",
                        srcRankId, dstRankId, layerId, hcclRet, netLinkNum);
            return false;
        }
        return HasLinkWithProtocol(linksList, netLinkNum, protocol);
    }

    void CheckProtocolSupport(const HcclComm &commHandle, const uint32_t *layerList, uint32_t layerNum,
                              const CommProtocol &protocol)
    {
        uint32_t srcRankId = GetRankId(commHandle);
        uint32_t rankSize = GetRankSize(commHandle);
        rankLinkMap_.clear();
        rankNumPerUbDomain_ = 0;

        LayerRanks ubDomain;
        if (!FindUbDomain(commHandle, layerList, layerNum, protocol, srcRankId, ubDomain)) {
            TORCH_CHECK(false, "Failed to determine UB domain for rank ", srcRankId,
                        ", rank has no peer supporting protocol ", static_cast<int>(protocol));
        }
        ValidateRankLinkMap(rankSize, srcRankId);
        rankNumPerUbDomain_ = static_cast<uint32_t>(ubDomain.ranks.size());
        TORCH_CHECK(rankNumPerUbDomain_ <= rankSize, "UB domain rank count ", rankNumPerUbDomain_,
                    " exceeds rank size ", rankSize);
        ASCEND_LOGI("Layer %u is current rank's UB domain, rankNumPerUbDomain_: %u", ubDomain.layer,
                    rankNumPerUbDomain_);

        if (rankNumPerUbDomain_ == rankSize) {
            topoType_ = TopoType::INTRA_SUPER_NODE;
            return;
        }

        TORCH_CHECK(protocol == CommProtocol::COMM_PROTOCOL_UB_MEM, "UBC_CTP must be supported by all ranks, got ",
                    rankNumPerUbDomain_, " of ", rankSize, " ranks in the resolved domain");

        TORCH_CHECK(rankSize >= rankNumPerUbDomain_ && rankSize % rankNumPerUbDomain_ == 0,
                    "rankNumPerUbDomain_ must be less than rankSize and divisible, rankNumPerUbDomain_: ",
                    rankNumPerUbDomain_, ", rankSize: ", rankSize);

        TORCH_CHECK(CheckIntraUbDomainProtocol(commHandle, layerList, layerNum, srcRankId), "Rank ", srcRankId,
                    " does not support UBC_CTP with peers inside its UB domain");
        TORCH_CHECK(CheckCrossUbDomainProtocols(commHandle, layerList, layerNum, srcRankId), "Rank ", srcRankId,
                    " has no UBG link to a peer outside its UB domain");
        ValidateRankLinkMap(rankSize, srcRankId);
        TORCH_CHECK(rankLinkMap_.size() == rankSize - 1, "Incomplete topology info for rank ", srcRankId, ", recorded ",
                    rankLinkMap_.size(), " of ", rankSize - 1);

        topoType_ = TopoType::CROSS_SUPER_NODE;
        ASCEND_LOGI("Cross-server confirmed, use UBC_CTP inside UB domain and UBG across UB domains");
    }

    // Extend the UB domain through consecutive inner-to-outer layers; record each peer's innermost usable layer.
    bool FindUbDomain(const HcclComm &commHandle, const uint32_t *layerList, uint32_t layerNum,
                      const CommProtocol &domainProtocol, uint32_t srcRankId, LayerRanks &ubDomain)
    {
        bool hasDomainLayer = false;
        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            LayerRanks layer = GetLayerRanks(commHandle, layerList[layerIndex]);
            if (!SupportsDomainProtocolWithAllRanks(commHandle, layer, domainProtocol, srcRankId)) {
                break;
            }
            RecordDomainLayerRanks(domainProtocol, layer, srcRankId);
            ubDomain = std::move(layer);
            hasDomainLayer = true;
        }
        return hasDomainLayer;
    }

    // Resolve each recorded UB-domain peer to a layer supporting UBC_CTP and update rankLinkMap_.
    bool CheckIntraUbDomainProtocol(const HcclComm &commHandle, const uint32_t *layerList, uint32_t layerNum,
                                    uint32_t srcRankId)
    {
        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            LayerRanks layer = GetLayerRanks(commHandle, layerList[layerIndex]);
            for (uint32_t dstRank : layer.ranks) {
                auto linkIter = rankLinkMap_.find(dstRank);
                if (dstRank == srcRankId || linkIter == rankLinkMap_.end() ||
                    linkIter->second.protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                if (SupportsProtocol(commHandle, layer.layer, srcRankId, dstRank,
                                     CommProtocol::COMM_PROTOCOL_UBC_CTP)) {
                    linkIter->second = {CommProtocol::COMM_PROTOCOL_UBC_CTP, layer.layer};
                }
            }
        }
        for (const auto &linkEntry : rankLinkMap_) {
            if (linkEntry.second.protocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                ASCEND_LOGW("Rank %u does not support UBC_CTP with rank %u inside its UB domain", srcRankId,
                            linkEntry.first);
                return false;
            }
        }
        return true;
    }

    // Search all layers because the cross-domain protocol may differ from the protocol used to identify the UB domain.
    bool CheckCrossUbDomainProtocols(const HcclComm &commHandle, const uint32_t *layerList, uint32_t layerNum,
                                     uint32_t srcRankId)
    {
        bool hasCrossDomainUbgLink = false;
        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            LayerRanks layer = GetLayerRanks(commHandle, layerList[layerIndex]);
            for (uint32_t dstRank : layer.ranks) {
                if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
                    continue;
                }
                if (SupportsProtocol(commHandle, layer.layer, srcRankId, dstRank, CommProtocol::COMM_PROTOCOL_UB_RTP)) {
                    ASCEND_LOGI("Rank %u does support UBG with cross-domain rank %u in layer %u", srcRankId, dstRank,
                                layer.layer);
                    hasCrossDomainUbgLink = true;
                    rankLinkMap_[dstRank] = {CommProtocol::COMM_PROTOCOL_UB_RTP, layer.layer};
                }
            }
        }
        // The caller separately verifies that every remote rank has a resolved topology entry.
        return hasCrossDomainUbgLink;
    }

    LayerRanks GetLayerRanks(const HcclComm &commHandle, uint32_t layerId) const
    {
        uint32_t rankNum = 0;
        uint32_t *rankList = nullptr;
        auto hcclRet = HcclRankGraphGetRanksByLayerFunc(commHandle, layerId, &rankList, &rankNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);
        TORCH_CHECK(rankList != nullptr && rankNum > 0, "Layer ", layerId, " returned an empty rank list");
        return {layerId, std::vector<uint32_t>(rankList, rankList + rankNum)};
    }

    bool SupportsDomainProtocolWithAllRanks(const HcclComm &commHandle, const LayerRanks &layer,
                                            const CommProtocol &domainProtocol, uint32_t srcRankId) const
    {
        for (uint32_t dstRank : layer.ranks) {
            if (dstRank == srcRankId || rankLinkMap_.count(dstRank) > 0) {
                continue;
            }
            if (!SupportsProtocol(commHandle, layer.layer, srcRankId, dstRank, domainProtocol)) {
                return false;
            }
        }
        return true;
    }

    // Record a peer once so its innermost usable layer is retained.
    void RecordDomainLayerRanks(const CommProtocol &protocol, const LayerRanks &layer, uint32_t srcRankId)
    {
        for (uint32_t dstRank : layer.ranks) {
            if (dstRank != srcRankId && rankLinkMap_.count(dstRank) == 0) {
                rankLinkMap_[dstRank] = {protocol, layer.layer};
            }
        }
    }

    static bool HasLinkWithProtocol(const CommLink *linksList, uint32_t netLinkNum, const CommProtocol &protocol)
    {
        for (uint32_t linkIdx = 0; linkIdx < netLinkNum; ++linkIdx) {
            if (linksList[linkIdx].linkAttr.linkProtocol == protocol) {
                return true;
            }
        }
        return false;
    }

    void ValidateRankLinkMap(uint32_t rankSize, uint32_t srcRankId) const
    {
        for (const auto &linkEntry : rankLinkMap_) {
            TORCH_CHECK(linkEntry.first < rankSize && linkEntry.first != srcRankId, "Invalid topology rank ",
                        linkEntry.first, " for local rank ", srcRankId, " and rank size ", rankSize);
        }
    }

    uint32_t GetRankId(const HcclComm &commHandle) const
    {
        uint32_t rankId = 0;
        auto hcclRet = HcclGetRankIdFunc(commHandle, &rankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);
        return rankId;
    }

    uint32_t GetRankSize(const HcclComm &commHandle) const
    {
        uint32_t rankSize = 0;
        auto hcclRet = HcclGetRankSizeFunc(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
        TORCH_CHECK(rankSize > 0 && rankSize <= HCCL_MAX_RANK_SIZE, "Invalid HCCL rank size: ", rankSize);
        return rankSize;
    }

    // ---- Channel management helpers ----

    void InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId, const CommProtocol &protocol,
                         std::vector<HcclChannelDesc> &channelDesc)
    {
        uint32_t channelNum = channelDesc.size();
        auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);
        ASCEND_LOGI("HCCL channel init success");

        uint32_t netLayerNum = 0;
        uint32_t *netLayerList = nullptr;
        GetNetLayers(commHandle, netLayerList, netLayerNum);
        TORCH_CHECK(netLayerNum > 0, "Get HCCL net layers failed, netLayerNum is ", netLayerNum);

        for (uint32_t dstRank = 0; dstRank < rankDim; ++dstRank) {
            if (dstRank == srcRankId) {
                continue;
            }
            uint32_t channelId = (dstRank > srcRankId) ? (dstRank - 1) : dstRank;
            RankLinkInfo linkInfo = ResolveLinkInfo(dstRank, protocol, netLayerList, netLayerNum);
            CommLink *links = nullptr;
            GetHcclCommLink(commHandle, linkInfo.layer, srcRankId, dstRank, linkInfo.protocol, links);
            channelDesc[channelId].channelProtocol = linkInfo.protocol;
            channelDesc[channelId].remoteRank = dstRank;
            channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
            channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
            channelDesc[channelId].memHandles = customMemHandle_;
            channelDesc[channelId].memHandleNum = 1;
        }
    }

    RankLinkInfo ResolveLinkInfo(uint32_t dstRank, const CommProtocol &protocol, const uint32_t *netLayerList,
                                 uint32_t netLayerNum) const
    {
        auto linkIter = rankLinkMap_.find(dstRank);
        if (linkIter != rankLinkMap_.end()) {
            return linkIter->second;
        }
        TORCH_CHECK(netLayerNum == HCCL_COMM_LAYERS_MTE_CCU, "Topology info not found for dstRank: ", dstRank,
                    " in a multi-layer topology");
        return {protocol, netLayerList[HCCL_COMM_LAYERS_UB_MEM]};
    }

    void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                            const CommProtocol &protocol, const CommEngine &engine, CommContext *commContextStruct)
    {
        ASCEND_LOGI("Start to get HCCL communication channel");
        TORCH_CHECK(rankDim > 0 && rankDim <= HCCL_MAX_RANK_SIZE, "Invalid HCCL rank size: ", rankDim);
        TORCH_CHECK(srcRankId < rankDim, "Invalid local rank ", srcRankId, " for rank size ", rankDim);
        uint32_t channelNum = rankDim - 1;
        std::vector<HcclChannelDesc> channelDesc(channelNum);

        InitHcclChannel(commHandle, rankDim, srcRankId, protocol, channelDesc);

        auto hcclRet =
            HcclChannelAcquireFunc(commHandle, engine, channelDesc.data(), channelNum, commContextStruct->hcommHandle_);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", hcclRet);
    }

    // ---- Resource management helpers ----

    void GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                             CommContext *commContextStruct, uint32_t rankSize, uint64_t &hcclBuffSize,
                             const std::string &targetTag)
    {
        ASCEND_LOGI("Start to get HCCL communication resource");
        uint32_t rankId = commContextStruct->epRankId;

        GetHcclCommChannel(commHandle, rankSize, rankId, protocol, engine, commContextStruct);
        ASCEND_LOGI("Get HCCL communication channel success, channel num is: %u", rankSize - 1);

        GetRegisteredCommResource(commHandle, commContextStruct, rankSize, targetTag);
        hcclBuffSize = static_cast<uint64_t>(customCclBufferSize_);

        ASCEND_LOGI("Get HCCL CommResource success");
    }

    void AllocateAndRegisterDeviceBuffer(const HcclComm &commHandle, const std::string &memBufferTag)
    {
        TORCH_CHECK(customDeviceBuffer_ != nullptr && customMemHandle_ != nullptr,
                    "custom buffer owner pointers must not be null");
        TORCH_CHECK(customCclBufferSize_ > 0, "customCclBufferSize must be greater than 0");
        TORCH_CHECK(customCclBufferSize_ <= std::numeric_limits<int64_t>::max(),
                    "customCclBufferSize is too large: ", customCclBufferSize_);

        uint64_t bufferSizeBytes = static_cast<uint64_t>(customCclBufferSize_);
        if (*customDeviceBuffer_ == nullptr) {
            aclError ar = aclrtMalloc(customDeviceBuffer_, bufferSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMalloc(", bufferSizeBytes, " ) failed, ret=", ar);
            ar = aclrtMemset(*customDeviceBuffer_, bufferSizeBytes, 0, bufferSizeBytes);
            TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMemset(customDeviceBuffer_) failed, ret=", ar);
        }
        if (*customMemHandle_ == nullptr) {
            CommMem mem;
            mem.type = COMM_MEM_TYPE_DEVICE;
            mem.addr = *customDeviceBuffer_;
            mem.size = bufferSizeBytes;
            auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, customMemHandle_);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", bufferSizeBytes,
                        ") failed, ret=", hcclRet);
        }
    }

    void GetRegisteredCommResource(const HcclComm &commHandle, CommContext *commContextStruct, uint32_t rankSize,
                                   const std::string &targetTag)
    {
        uint32_t rankId = commContextStruct->epRankId;
        commContextStruct->epHcclBuffer_[rankId] = reinterpret_cast<uint64_t>(*customDeviceBuffer_);
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            if (peer == rankId) {
                continue;
            }
            uint32_t idx = (peer < rankId) ? peer : (peer - 1);
            uint32_t memNum = 0;
            CommMem *remoteMems = nullptr;
            char **memTags = nullptr;
            auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, commContextStruct->hcommHandle_[idx], &memNum,
                                                        &remoteMems, &memTags);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", peer, ") failed, ret=", hcclRet);
            bool hasTargetMem = false;
            for (uint32_t j = 0; j < memNum; ++j) {
                if (memTags == nullptr || remoteMems == nullptr) {
                    break;
                }
                if (memTags[j] != nullptr && targetTag == memTags[j]) {
                    commContextStruct->epHcclBuffer_[peer] = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                    hasTargetMem = true;
                    break;
                }
            }
            TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
        }
    }

    // ---- Context lifecycle helpers ----

    void CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag, const CommEngine &engine,
                       const CommProtocol &protocol, void *&ctx, CommContext *commContextStruct, uint64_t &hcclBuffSize)
    {
        ASCEND_LOGI("Start to create HCCL context");
        uint64_t commContextSize = sizeof(CommContext);
        auto hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), engine, commContextSize, &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);
        ASCEND_LOGI("Create HCCL context success, ctx: %p", ctx);

        hcclRet = HcclGetRankIdFunc(commHandle, &commContextStruct->epRankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);
        ASCEND_LOGI("Get rank ID success, rankId is: %u", commContextStruct->epRankId);

        uint32_t rankSize = 0;
        hcclRet = HcclGetRankSizeFunc(commHandle, &rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
        ASCEND_LOGI("Get rank size success, rankSize is: %u", rankSize);

        std::string memBufferTag = mc2ContextTag + "_buffer";
        TORCH_CHECK(memBufferTag.size() <= HCCL_CONTEXT_TAG_MAX_LEN, "MemBufferTag is too long, max size is ",
                    HCCL_CONTEXT_TAG_MAX_LEN, ", got ", memBufferTag.size());
        AllocateAndRegisterDeviceBuffer(commHandle, memBufferTag);

        GetHcclCommResource(commHandle, engine, protocol, commContextStruct, rankSize, hcclBuffSize, memBufferTag);

        hcclRet =
            HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(), commContextStruct, commContextSize, 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
        ASCEND_LOGI("Copy context from host to device success");
    }

    void GetOrCreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag, const CommEngine &engine,
                            const CommProtocol &protocol, void *&ctx, uint64_t &hcclBuffSize,
                            CommContext &commContextStruct)
    {
        uint64_t ctxSize = 0;
        auto hcclRet = HcclEngineCtxGetFunc(commHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize);
        if (hcclRet != HCCL_SUCCESS) {
            CreateContext(commHandle, mc2ContextTag, engine, protocol, ctx, &commContextStruct, hcclBuffSize);
        } else {
            GetHcclBufferSize(commHandle, hcclBuffSize);
        }
    }

    // ---- Static HCCL query helpers ----

    static void GetHcclBufferSize(const HcclComm &commHandle, uint64_t &hcclBuffSize)
    {
        void *tempBuffer = nullptr;
        auto hcclRet = HcclGetHcclBufferFunc(commHandle, &tempBuffer, &hcclBuffSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Buffer Size failed, ret: ", hcclRet);
    }

    static void GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
    {
        auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", hcclRet);
        TORCH_CHECK(netLayerList != nullptr && netLayerNum > 0, "Get HCCL layers returned empty layer list");
        ASCEND_LOGI("Get HCCL layers success, netLayerNum is: %u", netLayerNum);
    }

    static void GetRankSizePerServer(const HcclComm &commHandle, uint32_t &rankSizePerServer)
    {
        uint32_t *netLayerList = nullptr;
        uint32_t netLayerNum = 0;
        GetNetLayers(commHandle, netLayerList, netLayerNum);

        uint32_t netLayers = netLayerList[GET_LOCAL_SERVER_RANK_SIZE_LAYER];
        auto hcclRet = HcclRankGraphGetRankSizeByLayerFunc(commHandle, netLayers, &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL rank size per server failed, ret: ", hcclRet);
        ASCEND_LOGI("Get HCCL rank size per server success, rankSizePerServer is: %u", rankSizePerServer);
    }

    static void GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId, uint32_t dstRankId,
                                const CommProtocol &protocol, CommLink *&links)
    {
        CommLink *linksList = nullptr;
        uint32_t netLinkNum = 0;
        auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Communication link failed, ret: ", hcclRet);
        TORCH_CHECK(linksList != nullptr && netLinkNum > 0, "The Net Link Is nullptr. srcRankId is ", srcRankId,
                    ", dstRankId is ", dstRankId, ", layerId is ", netLayerId);
        ASCEND_LOGI("Get HCCL Rank Links Success Links Num is: %u", netLinkNum);
        uint32_t index = 0;
        for (; index < netLinkNum; ++index) {
            if (linksList[index].linkAttr.linkProtocol == protocol) {
                links = &linksList[index];
                break;
            }
        }
        TORCH_CHECK(index < netLinkNum, "No matching communication protocol found in HCCL links, protocol is ",
                    static_cast<int>(protocol));
    }

    TopoType GetTopoType() const
    {
        return topoType_;
    }
    int64_t GetRankNumPerServer() const
    {
        return rankNumPerServer_;
    }

    // Per-peer protocol and layer selected during topology discovery and consumed by channel creation.
    std::unordered_map<uint32_t, RankLinkInfo> rankLinkMap_;
    uint32_t rankNumPerUbDomain_ = 0;
    TopoType topoType_ = TopoType::INTRA_SUPER_NODE;
    int64_t rankNumPerServer_ = DEFAULT_RANK_NUM_PER_SERVER;
    int64_t customCclBufferSize_ = 0;
    void **customDeviceBuffer_ = nullptr;
    HcclMemHandle *customMemHandle_ = nullptr;
};

// ======================== CommContextManager ========================

class CommContextManager {
public:
    CommContextManager(const std::string &group, int64_t worldSize, const py::object &backend = py::str("kfc"),
                       const std::string &commAlg = "ub-mem", const std::string &opName = "moe_dispatch_ffn_combine",
                       int64_t customCclBufferSize = 0, const py::object &customCclBufferSizeResolver = py::none())
        : group_(group),
          commAlg_(commAlg),
          opName_(opName),
          worldSize_(worldSize),
          backend_(backend),
          mode_(BackendMode::UNINITIALIZED),
          cclBufferSize_(0),
          topoType_(TopoType::INTRA_SUPER_NODE),
          rankNumPerServer_(DEFAULT_RANK_NUM_PER_SERVER),
          customCclBufferSize_(customCclBufferSize),
          customCclBufferSizeResolver_(customCclBufferSizeResolver)
    {}

    ~CommContextManager()
    {
        try {
            Destroy();
        } catch (const std::exception &e) {
            ASCEND_LOGE("CommContextManager destroy failed: %s", e.what());
        }
    }

    at::Tensor CreateContext()
    {
        EnsureResolved();
        at::Tensor context = at::empty({ContextTensorSize()}, at::TensorOptions()
                                                                  .dtype(at::kInt)
                                                                  .device(c10::DeviceType::PrivateUse1)
                                                                  .memory_format(c10::MemoryFormat::Contiguous));
        DispatchBuild(context);
        return context;
    }

    void UpdateGroup(const std::string &group, at::Tensor &contextTensor)
    {
        TORCH_CHECK(
            customCclBufferSizeResolver_.is_none() || (customMemHandle_ == nullptr && customDeviceBuffer_ == nullptr),
            "update_group cannot reuse an allocated channel buffer with customCclBufferSizeResolver; "
            "create a replacement CommContextManager instead");
        group_ = group;
        cclBufferSize_ = 0;
        EnsureResolved();
        DispatchBuild(contextTensor);
    }

    void Destroy()
    {
        if (customMemHandle_ != nullptr || customDeviceBuffer_ != nullptr) {
            aclError aclRet = aclrtSynchronizeDevice();
            TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret=", aclRet);
        }
        if (customMemHandle_ != nullptr) {
            customMemHandle_ = nullptr;
        }
        if (customDeviceBuffer_ != nullptr) {
            aclError aclRet = aclrtFree(customDeviceBuffer_);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtFree(customDeviceBuffer_) failed, ret=", aclRet);
            customDeviceBuffer_ = nullptr;
        }
        localDeviceBuffer_ = nullptr;
    }

    int64_t CclBufferSize() const
    {
        return cclBufferSize_;
    }
    int64_t GetTopoType() const
    {
        return static_cast<int64_t>(topoType_);
    }
    int64_t GetRankNumPerServer() const
    {
        return static_cast<int64_t>(rankNumPerServer_);
    }

    at::Tensor GetLocalBufferTensor(const py::object &dtype, int64_t offset) const
    {
        TORCH_CHECK(localDeviceBuffer_ != nullptr, "Local CCL buffer is not initialized.");
        TORCH_CHECK(offset >= 0, "offset must be non-negative, got ", offset, ".");
        torch::ScalarType scalarType = torch::python::detail::py_object_to_dtype(dtype);
        int64_t elementBytes = static_cast<int64_t>(c10::elementSize(scalarType));
        TORCH_CHECK(elementBytes > 0, "Invalid element size for requested dtype.");
        int64_t totalElements = cclBufferSize_ / elementBytes;
        TORCH_CHECK(offset <= totalElements, "offset must be in [0, ", totalElements, "], got ", offset, ".");

        void *data = static_cast<void *>(static_cast<uint8_t *>(localDeviceBuffer_) + offset * elementBytes);
        auto options = at::TensorOptions()
                           .dtype(scalarType)
                           .device(c10::DeviceType::PrivateUse1)
                           .memory_format(c10::MemoryFormat::Contiguous);
        return at::from_blob(data, {totalElements - offset}, options);
    }

private:
    static int64_t ContextTensorSize()
    {
        return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    }

    void EnsureResolved()
    {
        if (mode_ == BackendMode::UNINITIALIZED) {
            mode_ = ResolveBackend(backend_);
        }
    }

    void DispatchBuild(at::Tensor &tensor)
    {
        switch (mode_) {
            case BackendMode::KFC: {
                KfcContextBuilder builder;
                builder.Build(group_, worldSize_, cclBufferSize_, tensor, &localDeviceBuffer_);
                return;
            }
            case BackendMode::CHANNEL: {
                HcclChannelContextBuilder builder;
                builder.Build(group_, worldSize_, cclBufferSize_, tensor, commAlg_, opName_, customCclBufferSize_,
                              customCclBufferSizeResolver_, &customDeviceBuffer_, &customMemHandle_);
                localDeviceBuffer_ = customDeviceBuffer_;
                topoType_ = builder.GetTopoType();
                rankNumPerServer_ = builder.GetRankNumPerServer();
                return;
            }
            default:
                TORCH_CHECK(false, "Unknown backend mode: ", static_cast<int>(mode_));
        }
    }

    std::string group_;
    std::string commAlg_;
    std::string opName_;
    int64_t worldSize_;
    py::object backend_;
    BackendMode mode_;
    int64_t cclBufferSize_ = 0;
    TopoType topoType_ = TopoType::INTRA_SUPER_NODE;
    int64_t rankNumPerServer_ = DEFAULT_RANK_NUM_PER_SERVER;
    int64_t customCclBufferSize_ = 0;
    py::object customCclBufferSizeResolver_;
    void *customDeviceBuffer_ = nullptr;
    void *localDeviceBuffer_ = nullptr;
    HcclMemHandle customMemHandle_ = nullptr;
};

// Bind the CommContextManager class to Python module
void RegisterCommContext(pybind11::module_ &m)
{
    py::class_<CommContextManager>(m, "CommContextManager")
        .def(py::init<const std::string &, int64_t, const py::object &, const std::string &, const std::string &,
                      int64_t, const py::object &>(),
             py::arg("group"), py::arg("worldSize"), py::arg("backend") = std::string("kfc"),
             py::arg("commAlg") = std::string("ub-mem"), py::arg("opName") = std::string("moe_dispatch_ffn_combine"),
             py::arg("customCclBufferSize") = 0, py::arg("customCclBufferSizeResolver") = py::none())
        .def("create_context", &CommContextManager::CreateContext)
        .def("update_group", &CommContextManager::UpdateGroup, py::arg("group"), py::arg("contextTensor").noconvert())
        .def("destroy", &CommContextManager::Destroy)
        .def("get_local_buffer_tensor", &CommContextManager::GetLocalBufferTensor, py::arg("dtype"),
             py::arg("offset") = 0)
        .def_property_readonly("ccl_buffer_size", &CommContextManager::CclBufferSize)
        .def_property_readonly("topo_type", &CommContextManager::GetTopoType)
        .def_property_readonly("rank_num_per_server", &CommContextManager::GetRankNumPerServer);
}
} // namespace op_api
