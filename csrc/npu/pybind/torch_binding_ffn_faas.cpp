// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <torch/extension.h>
#include <torch/library.h>
#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "pytorch_extension/op_api_common.h"
#include "afd_ffn_faas/op_api/aclnn_afd_ffn_faas.h"

namespace afd_plugin::faas {
constexpr int64_t CONFIG_FIELDS = 13;
constexpr int64_t STATUS_FIELDS = 8;
constexpr int64_t PUBLISH_MODE = 1;
constexpr int64_t EXCHANGE_MODE = 8;
constexpr int64_t SCOPED_EXCHANGE_MODE = 9;
constexpr int64_t FINISH_MODE = 11;
constexpr size_t MAX_SCOPED_EXCHANGES = 4;
enum Field : int64_t {
    HIDDEN, INTERMEDIATE, EXPERTS, TOPK, ATTENTION_RANKS, FFN_RANKS,
    ATTENTION_TP, SLOTS, CAPACITY, WORLD_RANK, MODEL_LAYERS, LOADED_LAYERS,
    POLL_BUDGET
};

int64_t completion_code(const at::Tensor &status)
{
    int64_t code = 0;
    const auto error = aclrtMemcpy(&code, sizeof(code), status.data_ptr<int64_t>(),
                                   sizeof(code), ACL_MEMCPY_DEVICE_TO_HOST);
    TORCH_CHECK(error == ACL_SUCCESS, "FFN FaaS completion copy failed: ", error);
    return code;
}

void synchronize_completion(const at::Tensor &status, int64_t timeout_ms)
{
    TORCH_CHECK(status.scalar_type() == at::kLong && status.numel() == STATUS_FIELDS && status.is_contiguous(),
                "FFN FaaS completion requires its int64 status tensor");
    TORCH_CHECK(timeout_ms > 0 && timeout_ms <= std::numeric_limits<int32_t>::max(),
                "FFN FaaS completion timeout must fit positive milliseconds");
    // stream() flushes torch-npu's host task queue before exposing the ACL
    // stream. One active transaction owns this stream without a host
    // polling/sleep cadence.
    const auto stream = c10_npu::getCurrentNPUStream(status.device().index()).stream();
    TORCH_CHECK(stream != nullptr, "FFN FaaS host dispatch queue failed");
    const auto error = aclrtSynchronizeStreamWithTimeout(stream, static_cast<int32_t>(timeout_ms));
    TORCH_CHECK(error == ACL_SUCCESS, "FFN FaaS completion synchronize failed: ", error);
}

int64_t wait_completion(const at::Tensor &status, int64_t timeout_ms)
{
    synchronize_completion(status, timeout_ms);
    return completion_code(status);
}

std::vector<int64_t> wait_many(const c10::List<at::Tensor> &statuses, int64_t timeout_ms)
{
    TORCH_CHECK(!statuses.empty() && statuses.size() <= MAX_SCOPED_EXCHANGES,
                "FFN FaaS model scope requires a bounded nonempty status list");
    const at::Tensor last = statuses.get(statuses.size() - 1);
    for (const auto &value : statuses) {
        const at::Tensor status = value;
        TORCH_CHECK(status.device() == last.device() && status.scalar_type() == at::kLong &&
                        status.numel() == STATUS_FIELDS && status.is_contiguous(),
                    "invalid FFN FaaS model scope status");
    }
    synchronize_completion(last, timeout_ms);
    std::vector<int64_t> codes;
    codes.reserve(statuses.size());
    for (const auto &value : statuses) {
        codes.push_back(completion_code(value));
    }
    return codes;
}

std::tuple<int64_t, int64_t, int64_t> wait_selected_work(const at::Tensor &status, int64_t timeout_ms)
{
    synchronize_completion(status, timeout_ms);
    std::array<int64_t, STATUS_FIELDS> header{};
    const auto error = aclrtMemcpy(header.data(), sizeof(header), status.data_ptr<int64_t>(),
                                   sizeof(header), ACL_MEMCPY_DEVICE_TO_HOST);
    TORCH_CHECK(error == ACL_SUCCESS, "FFN FaaS selected-work copy failed: ", error);
    // PREPARE has retired. The executor may select borrowed native layer
    // parameters, but scratch/routes and the source credit stay owned until
    // FINISH publishes the result. This backend explicitly uses the host.
    return {header[0], header[3], header[4]};
}

template <bool EXECUTE_NPU>
std::tuple<at::Tensor, at::Tensor> run(
    const at::Tensor &x, const at::Tensor &expert_ids,
    const at::Tensor &probabilities,
    const at::Tensor &layers, const at::Tensor &request, const at::Tensor &state,
    const at::Tensor &scratch,
    int64_t mode, at::IntArrayRef config, c10::string_view group_name)
{
    TORCH_CHECK(config.size() == CONFIG_FIELDS, "FFN FaaS requires 13 configuration fields");
    TORCH_CHECK(mode >= 0 && mode <= FINISH_MODE && mode != 2 && mode != 7,
                "unsupported FFN FaaS control mode");
    for (const auto &tensor : {x, expert_ids, probabilities, layers, request, state, scratch}) {
        TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(),
                    "FFN FaaS inputs must be contiguous on one device");
        TORCH_CHECK(!tensor.requires_grad(), "FFN FaaS supports inference only");
    }
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                    x.size(1) == config[HIDDEN] && x.size(0) <= config[CAPACITY],
                "x must be BF16[tokens, hidden] within capacity");
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2 &&
                    expert_ids.size(0) == x.size(0) && expert_ids.size(1) == config[TOPK],
                "expert_ids must be int32[tokens, topk]");
    TORCH_CHECK((probabilities.scalar_type() == at::kFloat ||
                     probabilities.scalar_type() == at::kBFloat16) &&
                    probabilities.sizes() == expert_ids.sizes(),
                "probabilities must be float32 or BF16[tokens, topk]");
    TORCH_CHECK(layers.scalar_type() == at::kInt && layers.numel() == config[MODEL_LAYERS],
                "layer map must contain one int32 per model layer");
    TORCH_CHECK(request.scalar_type() == at::kLong && request.numel() >= 3,
                "request must contain int64 sequence, layer and token count");
    TORCH_CHECK(state.scalar_type() == at::kLong && config[ATTENTION_TP] > 0 &&
                    state.numel() >= 1 + config[ATTENTION_RANKS] / config[ATTENTION_TP],
                "executor state is too small");
    TORCH_CHECK(scratch.scalar_type() == at::kByte && scratch.dim() == 1,
                "executor scratch must be a contiguous byte buffer");
    auto y = mode == 6 ? at::empty({x.size(0), config[TOPK], config[HIDDEN]}, x.options()) : at::empty_like(x);
    auto status = at::empty({STATUS_FIELDS}, x.options().dtype(at::kLong));
    if constexpr (EXECUTE_NPU) {
        std::string group(group_name.data(), group_name.size());
        char *group_ptr = group.data();
        EXEC_NPU_CMD(aclnnAfdFfnFaas, x, expert_ids, probabilities,
                     layers, request, state, scratch, mode, config, group_ptr, y, status);
        if (mode == PUBLISH_MODE || mode == EXCHANGE_MODE) {
            // Hand publication to ACL before the model enqueues shared-expert
            // work. This flushes host dispatch, without waiting for device
            // completion; retirement still owns validation and buffer reuse.
            const auto stream = c10_npu::getCurrentNPUStream(x.device().index()).stream();
            TORCH_CHECK(stream != nullptr, "FFN FaaS publication dispatch failed");
        }
    }
    return {y, status};
}

struct ScopedContext {
    at::Tensor layers, state, scratch;
    std::vector<int64_t> config;
    std::string group;
};

class ScopedWork {
public:
    ScopedWork(c10_npu::NPUStream stream, std::shared_ptr<c10_npu::NPUEvent> producer,
               std::shared_ptr<ScopedContext> context)
        : stream_(stream), producer_(std::move(producer)), context_(std::move(context)) {}

    std::tuple<at::Tensor, at::Tensor> submit(
        const at::Tensor &x, const at::Tensor &ids, const at::Tensor &probabilities,
        const at::Tensor &request_host, const at::Tensor &request,
        int64_t sequence, int64_t layer, int64_t tokens)
    {
        TORCH_CHECK(!attempted_, "FFN FaaS scoped work can only be submitted once");
        attempted_ = true;
        // The Python-owned submitter already retains this work. Keep every
        // input/output even if a subsequent enqueue/event fails.
        retained_ = {x, ids, probabilities, request_host, request};
        const auto &config = context_->config;
        TORCH_CHECK(x.device() == stream_.device(), "FFN FaaS scoped stream device mismatch");
        TORCH_CHECK(request_host.is_cpu() && request_host.scalar_type() == at::kLong &&
                        request_host.is_contiguous() && request_host.numel() >= 3,
                    "FFN FaaS scoped request requires contiguous CPU int64 storage");
        const auto producer_stream = c10_npu::getCurrentNPUStream(x.device().index());
        c10_npu::NPUStreamGuard guard(stream_);
        // Share ordinary native shape/dtype validation and output allocation.
        // This specialization does not enqueue the fused device operation.
        std::tie(output_, status_) = run<false>(
            x, ids, probabilities, context_->layers,
            request, context_->state, context_->scratch,
            SCOPED_EXCHANGE_MODE, config, context_->group);
        TORCH_CHECK(config[WORLD_RANK] < config[ATTENTION_RANKS] && sequence > 0 &&
                        layer >= 0 && layer < config[MODEL_LAYERS] &&
                        tokens >= 0 && tokens <= x.size(0),
                    "invalid FFN FaaS scoped request identity");
        auto *identity = request_host.data_ptr<int64_t>();
        identity[0] = sequence;
        identity[1] = layer;
        identity[2] = tokens;
        request.copy_(request_host, true);
        producer_->record(producer_stream);
        producer_->block(stream_);
        const at::IntArrayRef native_config(config);
        char *group_ptr = context_->group.data();
        EXEC_NPU_CMD(aclnnAfdFfnFaas, x, ids, probabilities,
                     context_->layers, request,
                     context_->state, context_->scratch, SCOPED_EXCHANGE_MODE,
                     native_config, group_ptr, output_, status_);
        done_.record(stream_);
        recorded_ = true;
        return {output_, status_};
    }

    at::Tensor consume()
    {
        TORCH_CHECK(recorded_, "FFN FaaS scoped work has no submitted completion event");
        const auto consumer = c10_npu::getCurrentNPUStream(output_.device().index());
        done_.block(consumer);
        c10_npu::NPUCachingAllocator::recordStream(output_.storage().data_ptr(), consumer);
        return output_;
    }

private:
    c10_npu::NPUStream stream_;
    std::shared_ptr<c10_npu::NPUEvent> producer_;
    c10_npu::NPUEvent done_;
    std::shared_ptr<ScopedContext> context_;
    std::vector<at::Tensor> retained_;
    at::Tensor output_, status_;
    bool attempted_ = false;
    bool recorded_ = false;
};

class ScopedSubmitter {
public:
    ScopedSubmitter(const at::Tensor &layers,
                    const at::Tensor &state, const at::Tensor &scratch,
                    std::vector<int64_t> config, std::string group,
                    int64_t stream_id, int64_t device_index, int64_t device_type)
        : stream_(c10_npu::NPUStream::unpack3(
              stream_id, static_cast<c10::DeviceIndex>(device_index),
              static_cast<c10::DeviceType>(device_type))),
          context_(std::make_shared<ScopedContext>(ScopedContext{
              layers, state, scratch, std::move(config), std::move(group)}))
    {
        TORCH_CHECK(stream_.device().type() == c10::DeviceType::PrivateUse1,
                    "FFN FaaS scoped submission requires an NPU stream");
        for (auto &producer : producers_) {
            producer = std::make_shared<c10_npu::NPUEvent>();
        }
    }

    std::tuple<at::Tensor, at::Tensor, std::shared_ptr<ScopedWork>> submit(
        const at::Tensor &x, const at::Tensor &ids, const at::Tensor &probabilities,
        const at::Tensor &request_host, const at::Tensor &request,
        int64_t sequence, int64_t layer, int64_t tokens, int64_t slot)
    {
        TORCH_CHECK(!failed_, "FFN FaaS scoped submitter has an untracked failure");
        TORCH_CHECK(slot >= 0 && static_cast<size_t>(slot) < producers_.size(),
                    "FFN FaaS scoped request slot is outside its bounded pool");
        // The Python executor reuses a slot only after wait_many completes the
        // comm stream. Its previous producer wait has therefore also completed.
        try {
            // Python owns this submitter throughout submission/unsafe exit.
            // Retain the new work here before its first possible enqueue,
            // including failure before a Python return value can be created.
            pending_ = std::make_shared<ScopedWork>(stream_, producers_[slot], context_);
            auto result = pending_->submit(x, ids, probabilities, request_host, request,
                                            sequence, layer, tokens);
            return {std::get<0>(result), std::get<1>(result), pending_};
        } catch (...) {
            failed_ = true;
            throw;
        }
    }

    at::Tensor consume(const std::shared_ptr<ScopedWork> &work)
    {
        auto result = work->consume();
        // A Python NativeCompletion already owns the successful work. Failed
        // consume keeps pending_ rather than dropping an uncertain dependency.
        if (pending_ == work) {
            pending_.reset();
        }
        return result;
    }

private:
    c10_npu::NPUStream stream_;
    std::shared_ptr<ScopedContext> context_;
    std::shared_ptr<ScopedWork> pending_;
    std::array<std::shared_ptr<c10_npu::NPUEvent>, MAX_SCOPED_EXCHANGES> producers_;
    bool failed_ = false;
};

void register_scoped_submitter(pybind11::module_ &module)
{
    pybind11::class_<ScopedWork, std::shared_ptr<ScopedWork>>(module, "FFNFaasScopedWork")
        .def("consume", &ScopedWork::consume, pybind11::call_guard<pybind11::gil_scoped_release>());
    pybind11::class_<ScopedSubmitter>(module, "FFNFaasScopedSubmitter")
        .def(pybind11::init<const at::Tensor &, const at::Tensor &, const at::Tensor &,
                           std::vector<int64_t>, std::string, int64_t, int64_t, int64_t>())
        .def("submit", &ScopedSubmitter::submit, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("consume", &ScopedSubmitter::consume, pybind11::call_guard<pybind11::gil_scoped_release>());
}
}  // namespace afd_plugin::faas

TORCH_LIBRARY_FRAGMENT(afd_ascend, m)
{
    m.def("ffn_faas_wait(Tensor status, int timeout_ms) -> int");
    m.def("ffn_faas_wait_selected(Tensor status, int timeout_ms) -> (int, int, int)");
    m.def("ffn_faas_wait_many(Tensor[] statuses, int timeout_ms) -> int[]");
    m.def("ffn_faas(Tensor x, Tensor expert_ids, Tensor probabilities, "
          "Tensor layers, Tensor request, Tensor(a!) state, Tensor(b!) scratch, "
          "int mode, int[] config, str group_name) -> (Tensor, Tensor)");

}
TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, m)
{
    m.impl("ffn_faas_wait", &afd_plugin::faas::wait_completion);
    m.impl("ffn_faas_wait_selected", &afd_plugin::faas::wait_selected_work);
    m.impl("ffn_faas_wait_many", &afd_plugin::faas::wait_many);
    m.impl("ffn_faas", &afd_plugin::faas::run<true>);
}
TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, m)
{
    m.impl("ffn_faas_wait", &afd_plugin::faas::wait_completion);
    m.impl("ffn_faas_wait_selected", &afd_plugin::faas::wait_selected_work);
    m.impl("ffn_faas_wait_many", &afd_plugin::faas::wait_many);
    m.impl("ffn_faas", &afd_plugin::faas::run<true>);
}
TORCH_LIBRARY_IMPL(afd_ascend, Meta, m)
{
    m.impl("ffn_faas", &afd_plugin::faas::run<false>);
}
