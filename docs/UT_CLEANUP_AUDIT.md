# UT 全量清理记录

审查基线是本地 `codex/ut-maintenance` 的上一轮工作树：HEAD 为 `652e249872bc`，已经包含前三轮整理。本轮对该工作树剩余的 **83 个 UT 文件、734 个测试函数定义**逐个检查被测代码、断言观察点、输入是否属于支持范围，以及跨文件重复；不是从中抽样。生产代码未修改。

## 整个 PR 的数量变化

相对提交基线 `652e249872bc968fc645c68ba6683182e0f69368`，整个 PR（包含此前各轮整理和本次审计修正）的最终统计如下。生产代码未修改。

| 统计口径 | PR 基线 | 最终工作树 | 净变化 |
| --- | ---: | ---: | ---: |
| UT Python 文件（含辅助代码） | 126 | 107 | -19 |
| UT 测试文件 | 96 | 78 | -18 |
| UT 测试函数定义 | 946 | 617 | -329 |
| UT Python 行数 | 35,177 | 26,634 | -8,543 |

这些变化包含迁移到设备测试目录、合并和参数化，不能理解为删除了 329 项独立行为覆盖。

## 最后一轮 83 个文件审查的数量变化

| 统计口径 | 清理前 | 清理后 | 净变化 |
| --- | ---: | ---: | ---: |
| UT Python 文件（含辅助代码） | 112 | 107 | -5 |
| UT 测试文件 | 83 | 78 | -5 |
| UT 测试函数定义 | 734 | 617 | -117 |
| UT Python 行数 | 30,142 | 26,634 | -3,508 |

`测试文件`按 `test_*.py` / `*_test.py` 统计；`函数定义`不展开 parametrize，也不把 pytest subtests 当独立函数。数量变化包含删除、合并、改名和迁移，不能据此推断删掉了多少独立行为。

五个原 UT 文件退出目录：一个仅测试 upstream 的文件淘汰，四个文件合入对应模块。另有一个真实 CUDA 路由测试迁至 [tests/gpu/test_attention_routing.py](../tests/gpu/test_attention_routing.py)，保留 native fused router 和 softmax/top-k 两个对照，并接入 ready/merge 的原 L4 Simple Test 环节。

| 原文件 | 处理及去向 |
| --- | --- |
| `compat/patches/test_async_dp_dplb_runtime.py` | 删除。三个测试直接执行 upstream DPLBAsyncMPClient，没有 AFD 被测代码。 |
| `config/test_async_cam_dbo.py` | 合并至 [compat/npu/test_runtime_config.py](../tests/unit/compat/npu/test_runtime_config.py)。 |
| `compat/npu/test_dsv4_async_validation.py` | 合并至 [compat/npu/test_feature_validation.py](../tests/unit/compat/npu/test_feature_validation.py)。 |
| `model_executor/models/test_attention_gate_activation.py` | 合并至 [model_executor/models/npu/test_deepseek_v2_attention_gate.py](../tests/unit/model_executor/models/npu/test_deepseek_v2_attention_gate.py)。 |
| `model_executor/models/test_w4a8_attention_gate.py` | 合并至 [model_executor/models/npu/test_deepseek_v2_attention_gate.py](../tests/unit/model_executor/models/npu/test_deepseek_v2_attention_gate.py)。 |

## 保留和删除的判断

- 保留能指出独立 AFD 约束的结果检查，例如 checkpoint 所有权、peer 载荷、native 外部 ABI、事务/stream 协议、资源释放、配置拒绝，以及实际 CPU 数值。
- 删除只执行 upstream 的测试、绕过公开配置入口制造的无效场景、继承/类名/私有缓存布局检查、内部 helper 的参数或次数快照，以及其他 owner 已验证的同一规则。
- 合并重复测试体和不影响被测分支的全交叉参数；不同有效边界仍保留字面预期。GPU/NPU fakes 用于主机编排或 ABI，不承担设备正确性证明。

具体收敛包括：配置门禁从 runtime 文件移到 CPU owner；token 聚合和 graph key 归各自 utility；私有 DSV4 validator 通过公开入口测试；P2P 发送/接收观察实际 custom-op 载荷和返回 metadata；关闭检查以 weakref 观察 payload 释放；MTP 检查使用公开 getter；native model hook 检查构造出的参数所有权。

## 独立评审后的三项修正（2026-11-05）

这三项修正只修改了三个测试文件和本报告，未修改生产代码。本文统计和逐项结论还包含下节的补充核查结果。

1. **纠正 eager 输出期望。** `test_async_ffn_runtime.py` 原先把 layered/non-layered 两条路径都与完整 dispatch buffer 比较，会拒绝正确的 non-layered 输出。现在分别检查有效前三行、零 token 的 `(1, 32)` BF16 全零占位，以及 layered 的两个 `(8, 32)` 容量输出，仍检查原始返回 header。输入改用逐行不同的数值，使切片内容也可区分。
2. **补回 eager hidden buffer 生命周期。** 先前将 `test_eager_state_does_not_cache_hidden_buffers` 整体判为私有布局检查，是错误的保留价值判断。新用例 `test_eager_steps_release_hidden_buffers_with_graphs_enabled` 经真实 control update 和 receive 路径，在 graphs 开启时连续处理 3/7/5 token 的 eager step，以分配对象的弱引用证明 payload 消费后不再保留 hidden buffer。现有 graph receive 地址复用检查保持不变，承担独立职责，不能替代 eager 分配路径。
3. **贯通 receiver 正常退出与真实关闭。** 原测试只检查公共错误入口，不能发现正常退出交接丢失。将这组成功/失败用例移入 CPU 可运行的 `test_async_cam_startup_wiring.py`，经过真实 runner 构造、主机 graph capture、worker 退出、`stop_ffn_server_loop()`、runner `shutdown()` 和 graph close；成功时 graph 弱引用失效，失败时 graph 仍存活且公共入口继续报告错误。未替换 runner shutdown，也未设置或断言 receiver-exit 私有标志。

这些修正保留行为约束，不恢复缓存 key/字典布局、私有状态或内部调用参数/次数断言。旧断言使用私有字段，并不意味着其保护的资源生命周期可以一并删除；替代测试必须实际经过能产生同一回归的路径。

此前单列的 layered 元数据消费、async-DP output-task 启动两个疑点已完成补充核查，结论和处理如下。

## 两项疑点的补充核查（2026-11-05）

结论是：layered 的 runner 接缝存在行为级 UT 覆盖缺口；async-DP 被删除的 output-task 调用断言没有独立保留价值。本次只强化已有的一个参数化用例并更新本报告，不增加测试文件或函数，不修改生产代码。

### Layered 元数据：补强现有输出检查

真实 eager 路径为 `execute_connector_driven_step()` → `_ffn_forward_connector_driven()` → connector 接收 payload → `AsyncCAMLayeredExecutor` → `send_ffn_output()`。FULL 则在 `_capture_graph()` 中建立同一收包、计算、发包事务。层号来自 CAM `header[2]`，专家 token 数来自 `group_list`，每行量化 scale 来自 `dynamic_scales`；host 的静态 `layer_ids` 循环不决定当前收到的工作属于哪层。

替代测试核对结果：

- [executor UT](../tests/unit/model_executor/test_async_cam_layered.py) 和 [W4A8](../tests/npu/test_async_cam_layered_w4a8.py) / [W8A8](../tests/npu/test_async_cam_layered_w8a8.py) 设备测试直接调用 executor，保护其算子 ABI、选层和数值，不经过 runner 的这两个参数传递位置。
- [graph UT](../tests/unit/v1/worker/test_async_ffn_graph.py) 主要验证资源生命周期；它检查 header 身份，但 fake 的输出不依赖分组和 scale。将 eager/capture 两处传入的分组都反转，修正前 runtime/graph 的 **9 项仍全部通过**。
- [有限 native 比较](../tests/npu/async_cam_runtime.py) 和 [模型 E2E](../tests/e2e/models/deepseek_v4_flash/ffn_graph.py) 以保存的 layered eager 输出对照 FULL。[E2E observer](../tests/e2e/models/deepseek_v4_flash/ffn_graph_evidence.py) 从原 context 记录 header 和 token 总数；这不能证明计算消费了相同的专家分组。依据其比较口径，两种模式若同时传错同一分组，仍可能互相一致。这是代码分析，未在设备上运行该注入，也未将“E2E 未运行”本身作为缺口证据。

因此，先前仅以“peer 返回原 header”说明替代覆盖充分，是不完整的。已强化 [现有 runtime 用例](../tests/unit/v1/worker/test_async_ffn_runtime.py)：toy FFN 按收到的层号、两个专家的分组和逐行 scale 产生不同结果；字面预期的前三行增益分别为 5/20/30，toy 的空 payload 返回完整容量的零输出。只断言 peer 输出的值、shape、dtype 和原始 header，未恢复内部 collaborator 的实参/次数断言。这验证主机元数据接线，不证明 CANN 运算正确性。

### Async-DP output-task：不恢复调用断言

核对共同 pin **vLLM 0.32.0 / `5df63a7203cdb9328d824a27b682e77c7b9928be`** 的 [core client](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core_client.py) 和 [AsyncLLM](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/async_llm.py)：

- 在事件循环内构造 client 时，`AsyncMPClient.__init__()` 已启动 output queue task；循环外构造时延迟启动。
- `AsyncMPClient.get_output_async()` 在取结果前启动该 task；`_call_utility_async()` 等待 utility 结果前也有独立启动入口。
- AsyncLLM 在构造或首次 `add_request()` 时启动 output handler，handler 通过 `get_output_async()` 消费结果。因此正常请求交付不只依赖 AFD `add_request_async()` 尾部的那一次调用。

内部负载均衡使用的 `DPLBAsyncMPClient` 也继承上述 getter。另核对固定 vLLM-Ascend `bd42eea01d8312c4df469fabaed45f77a12df4d3` 的平台注册、[engine 入口补丁](https://github.com/vllm-project/vllm-ascend/blob/bd42eea01d8312c4df469fabaed45f77a12df4d3/vllm_ascend/patch/platform/patch_engine_core.py)、balance/dyntra DP engine 补丁：这些相关改动未替换上述 client 构造器、getter 或 output-task 启动方法。

使用该固定源码的真实 Python 构造器、socket 消费协程、getter，以及当前 AFD 的 `add_request_async()`，仅替代外部传输端点；对照保留/移除 add-request 启动调用 × 循环内/外构造 client，**四种组合都能经公开 getter 取得预期请求结果**。这确认旧探针中的“调用被移除”不足以推出正常请求挂起。现有 [AFD client 用例](../tests/unit/compat/patches/test_async_dp_engine.py) 继续保护 AFD 拥有的 ADD 路由、wave/client metadata 和 FIRST_REQ 策略，不补回 output-ready 私有标记或 helper 调用断言。

### 本次补充核查的验证

| 检查 | 结果 | 证据范围 |
| --- | --- | --- |
| 当前 runtime/graph 主机适配运行 | 9 passed | CPU Torch 2.15.0；外部设备/runtime 导入使用替身，执行真实 runner Python 方法和当前测试体 |
| executor UT 与 async-DP patch UT | 75 passed | CPU Torch 环境，`test_async_cam_layered.py` 和 `test_async_dp_engine.py` |
| 无 torch 的相关 UT | 19 passed / 1 skipped | async-DP patch 用例实际运行；runtime 模块按缺少 vLLM 的门禁跳过 |
| Layered 分组错误注入 | 修正前 9 passed；修正后 1 failed / 8 passed | 在内存中反转 eager/capture 传入 executor 的分组；新增的输出差异检查失败 |
| Layered 层号错误注入 | 修正前 2 passed；修正后 1 failed / 1 passed | 改传合法但错误的层号 2，返回 context 不变；layered 行因输出数值不符而失败 |
| Layered scale 错误注入 | 修正后 1 failed / 1 passed | 将传入 scale 替换为全 1；layered 行因输出数值不符而失败 |
| 固定上游输出交付探针 | 4 组均交付成功 | 真实固定源码的方法；假传输端点；不构成完整 vLLM 服务或设备验证 |
| Ruff / format / mypy 3.11 / Markdownlint / git diff --check | 通过 | 本次修改的测试文件、审计文档及工作树空白检查 |

本地证据保存在 `/tmp/afd-ut-layered-*.xml`、`/tmp/afd-ut-layered-direct-and-asyncdp.xml` 和 `/tmp/afd-ut-two-suspicions-cpu.xml`。一次性主机适配/错误注入与 output-task 探针未加入仓库；生产源码未改写。没有新测例定义，当前仍是 **78 个 UT 测试文件、617 个测试函数定义**。本次未执行真实 GPU/NPU、通信或模型 E2E。

## 83 个原文件逐项结论

以下路径均相对于 `tests/unit/`。定义数是原文件到当前同路径的变化，迁移目标会单独增加；每个保留文件都给出独立价值。

| 文件 | 定义数 | 处理 | 独立价值或替代入口 |
| --- | ---: | --- | --- |
| [compat/npu/test_ascend_config_lifecycle.py](../tests/unit/compat/npu/test_ascend_config_lifecycle.py) | 11 → 6 | 保留并删并/改写 | AFD namespace 隔离/配置对象身份、native singleton 发布与刷新、验证失败/partial 时保留旧缓存、Async CAM SP 的 shared-expert DP 决策、已加载 native caller 实际使用新工厂。 |
| [compat/npu/test_ascend_engine_core_binding.py](../tests/unit/compat/npu/test_ascend_engine_core_binding.py) | 3 → 3 | 保留 | 早于 child AscendConfig 初始化的入口绑定、sync/async Attention/FFN 分支及非 AFD 不绑定；生命周期工厂测试没有执行 EngineCoreProc 入口。 |
| `compat/npu/test_dsv4_async_validation.py` | 5 → 0 | 合并 | 原五项 connector/gate/quantization 行为改由公开 feature validator 的五行表验证，删除私有 helper 测试文件。 |
| [compat/npu/test_feature_validation.py](../tests/unit/compat/npu/test_feature_validation.py) | 9 → 10 | 保留并删并/改写 | 公开 NPU feature validator 对同步/async/DBO/DSpark 的支持边界、transport capacity、MRV2 layout；DSV4 token-id connector 与量化门禁由公共入口验证。 |
| [compat/npu/test_profiler.py](../tests/unit/compat/npu/test_profiler.py) | 6 → 6 | 保留 | Ascend profiler 的独立启用变量、角色默认调度/输出目录、torch_npu API 配置和生命周期；CUDA 配置不同，不能用 CUDA 表替代。 |
| [compat/npu/test_routing_ids.py](../tests/unit/compat/npu/test_routing_ids.py) | 5 → 5 | 保留并删并/改写 | AFD 自定义 op 的 profile/live/force 输出、mutation schema/fake tensor、绕过 guards 的编译结果重用与 Inductor 路径。 |
| [compat/npu/test_runtime_config.py](../tests/unit/compat/npu/test_runtime_config.py) | 1 → 4 | 保留并删并/改写 | 非 SP backend 修正、保留 native SP/backend、Attention async ubatching 归一化与幂等、其他角色/connector 不归一化、两项 native 阈值拒绝。 |
| [compat/patches/npu/test_mrv2_ubatching.py](../tests/unit/compat/patches/npu/test_mrv2_ubatching.py) | 5 → 5 | 保留并删并/改写 | CAMP2P 的 eager padding 与 native graph agreement、async DP1 eligibility、native dispatch 失败恢复、execute 的 profile/randomize 输入策略和 ContextVar 退出恢复；不模拟设备数值。 |
| `compat/patches/test_async_dp_dplb_runtime.py` | 3 → 0 | 淘汰 | 无；全部测试直接调用上游 DPLBAsyncMPClient，没有 AFD 被测代码。 |
| [compat/patches/test_async_dp_engine.py](../tests/unit/compat/patches/test_async_dp_engine.py) | 14 → 12 | 保留并删并/改写 | AFD async 选择普通 EngineCore、非 async/native 保留 DP、构造中信号导致真实 shutdown、后覆盖重新安装实际生效、wave 协调/地址/store 生命周期、请求 ADD/FIRST_REQ 消息和版本门禁。 |
| [compat/patches/test_async_dp_forward_context.py](../tests/unit/compat/patches/test_async_dp_forward_context.py) | 2 → 2 | 保留 | async ForwardContext 不带跨 DP metadata；同步路径保留 coordinator 返回的 token counts。两者均观察最终上下文，不断言原函数/内部 helper 调用标记。 |
| [compat/patches/test_config_validation.py](../tests/unit/compat/patches/test_config_validation.py) | 5 → 3 | 保留并删并/改写 | 真实 AFD EngineArgs/VllmConfig wrapper 的最终 worker、后端/DBO 保留、非 AFD native 拒绝、平台更新失败恢复；不在测试中重写 wrapper 决策。 |
| [compat/patches/test_engine_core.py](../tests/unit/compat/patches/test_engine_core.py) | 12 → 11 | 保留并删并/改写 | FFN 不创建 KV/scheduler 请求状态、late-loaded KV 空响应/ready 协议、CUDA/NPU Attention/native 初始化和 KV 对齐、async request-count 消息、FFN start/check/stop RPC 与 readiness/failure。 |
| [compat/patches/test_gpu_deepseek_v4_dbo.py](../tests/unit/compat/patches/test_gpu_deepseek_v4_dbo.py) | 2 → 2 | 保留 | V4 workspace 失败仍保留原 lane/ForwardContext/存储内容；仅对空 DBO Attention 跳过 padding 数值计算。 |
| [compat/patches/test_gpu_mrv2_ubatching.py](../tests/unit/compat/patches/test_gpu_mrv2_ubatching.py) | 11 → 8 | 保留并删并/改写 | DP1 child metadata、decode/prefill 阈值、graph bucket 与 eager fallback、DP2 agreement 重用/不同 batch 拒绝，以及实际 KV 初始化期间 DP1 工厂生效并在成功/异常后释放。 |
| [compat/patches/test_signatures.py](../tests/unit/compat/patches/test_signatures.py) | 1 → 1 | 保留 | 全仓唯一的 upstream signature 参数表：与安装的原始定义比较参数和返回标注，避免已安装 patch 自比；仓库明确要求此项，行为测试不替代函数签名契约。 |
| [compat/test_ascend_ops.py](../tests/unit/compat/test_ascend_ops.py) | 4 → 4 | 保留并删并/改写 | 缺扩展的错误、插件 vendor 路径与最终 loader 环境、四个 async op 缺失的拒绝；不将符号存在当设备算子执行证明。 |
| [compat/test_profiler.py](../tests/unit/compat/test_profiler.py) | 5 → 5 | 保留 | CUDA profiler 的角色默认配置、目录优先级、向 torch profiler 提交的真实外部 API 设置，以及 optional profiler 的 step/stop 最终状态。 |
| [compat/test_runtime.py](../tests/unit/compat/test_runtime.py) | 4 → 4 | 保留 | 安装失败后重试真正生效、缺 MLA resolver 的错误、从当前 ForwardContext 取 stage registry 与无 AFD 时 native fallback，以及 MRV2 context 的公有 metadata/input_ids。 |
| [compat/test_vllm.py](../tests/unit/compat/test_vllm.py) | 5 → 4 | 保留并删并/改写 | vLLM 模块版本门禁、缺少运行时的 CPU 导入、严格报错与非严格 warning 是不同的对外结果。 |
| `config/test_async_cam_dbo.py` | 3 → 0 | 合并 | 三项行为并入实际源模块 runtime_config 的测试文件；取消不被读取的 DP 参数及重复非法值。 |
| [config/test_config.py](../tests/unit/config/test_config.py) | 19 → 19 | 保留 | 公共配置解析、别名冲突以外的角色/连接器拒绝、可选命名空间、布尔/整数转换、图哈希随有效配置变化；这些入口不依赖设备。 |
| [config/test_validation.py](../tests/unit/config/test_validation.py) | 7 → 9 | 保留并删并/改写 | CPU 部署策略唯一 owner：stack/worker/role、GPU/NPU runtime 支持边界、routed gate/模型/TP、graph/DBO/并行限制与拒绝时不导入 model runtime。 |
| [connectors/test_async_cam_connector.py](../tests/unit/connectors/test_async_cam_connector.py) | 25 → 19 | 保留并删并/改写 | async CAM 的 factory/schema、全局 A-first topology、CANN dispatch/combine ABI、匹配 pending 事务的 FIFO/错误拒绝、work-item token/layer/expert-count metadata、空 rank 回传、路由委托/拒绝与关闭释放 payload。 |
| [connectors/test_async_cam_startup.py](../tests/unit/connectors/test_async_cam_startup.py) | 16 → 13 | 保留并删并/改写 | 全 world nonce/plan 协商、模式/签名/层表不一致拒绝、two-stream Attention capability barrier、全部 phase/service readiness、peer failure 优先和 late join 传播、容量 float32/uint16 边界。 |
| [connectors/test_camp2p_connector.py](../tests/unit/connectors/test_camp2p_connector.py) | 13 → 11 | 保留并删并/改写 | 同步 CAMP2P 的 stage padding/opaque op payload、strided A/F 分组、role core budget、HCCL group 独立 options、各 ubatch 通信名和 lazy missing-op 失败。 |
| [connectors/test_camp2p_token_ids.py](../tests/unit/connectors/test_camp2p_token_ids.py) | 7 → 7 | 保留并删并/改写 | Hash/token-ID wire 编码的 int64→int32/列复制与 zero scales、接收裁剪和不足拒绝，以及 send/recv 两端 mode 一致；依赖 torch 的 CPU helper 执行不需要 CANN。 |
| [connectors/test_deepep_m2n_connector.py](../tests/unit/connectors/test_deepep_m2n_connector.py) | 60 → 54 | 保留并删并/改写 | 独立的 DeepEP 配置、model/transport/network handshake、A/F expert placement、TP token slices/capacity/lane plan、graph/eager bucket 一致性、FP8 格式/行分区、资源取消、外部 afdEP buffer 契约和 lane stream/event/lifetime 协议。 |
| [connectors/test_deepep_m2n_network.py](../tests/unit/connectors/test_deepep_m2n_network.py) | 8 → 7 | 保留并删并/改写 | 主机/NVLink fabric domain 区分、跨 domain 必须 GIN、显式 gin on/off 和最终 EP_DISABLE_GIN 环境；独立于 connector 的 model/transport handshake。 |
| [connectors/test_metadata.py](../tests/unit/connectors/test_metadata.py) | 8 → 9 | 保留 | DP/SP token layout 的上下文恢复、JSON 控制载荷 round trip（含 stage/graph/profile flags）、接收的两处关闭时机、只翻译本地取消错误、空/非法 transfer metadata。 |
| [connectors/test_p2p_connector.py](../tests/unit/connectors/test_p2p_connector.py) | 24 → 18 | 保留并删并/改写 | connector 构造/工厂 rank/多模态 text config、TP peer/control 路由、各 Attention peer 的 token 布局、空 rank padding、custom-op 注册与 grouped 传输、shutdown/cancel、graphs 开启时 eager hidden buffer 的释放和跨节点复用 store。 |
| [connectors/test_p2p_experts_contract.py](../tests/unit/connectors/test_p2p_experts_contract.py) | 8 → 6 | 保留并删并/改写 | 公开 A/F payload 的 hidden/router logits/token IDs 的 wire 字段顺序、拒绝不对齐输入、FFN peer fan-in 顺序和值、IDs dtype 与 graph 缓冲区实际重用；与一般 connector topology/control 测试不同。 |
| [distributed/test_afd_process_group.py](../tests/unit/distributed/test_afd_process_group.py) | 1 → 1 | 保留 | HCCL options 中配置显式 buffer 大小且每组独立对象，避免不同 process group 共用可变 options；不执行设备通信。 |
| [distributed/test_topology.py](../tests/unit/distributed/test_topology.py) | 7 → 7 | 保留 | 全局 DP/PCP/TP 线性 rank、越界拒绝、A/F 均匀分区/双方一致性/控制消息覆盖，以及独立字面拓扑 oracle；这些控制协议规则不依赖某个 connector 实现。 |
| [model_executor/models/npu/test_deepseek_v2_attention_gate.py](../tests/unit/model_executor/models/npu/test_deepseek_v2_attention_gate.py) | 8 → 6 | 保留并删并/改写 | V2/V4 post-quant scale 被实际 CPU FFN 消费的数值、已有 scale 与 refinalize 后新 scale；合并 native MLP activation/W4A8 ABI 两文件，保留可解释的外部 Ascend 输入契约。 |
| [model_executor/models/npu/test_deepseek_v4.py](../tests/unit/model_executor/models/npu/test_deepseek_v4.py) | 7 → 7 | 保留并删并/改写 | Ascend V4 role 权重、模型间/层间 DSA buffer 隔离、native rope 范围、公共 MTP getter 的分配条件及 checkpoint/runtime stage 别名。 |
| [model_executor/models/npu/test_deepseek_v4_dspark.py](../tests/unit/model_executor/models/npu/test_deepseek_v4_dspark.py) | 2 → 2 | 保留 | DSpark 提供的 spec-layer ID 构造仍走 AFD wrapper，MTP runtime/checkpoint stage 别名权重归属和 one-shot native loader；区别于 target 模型普通层。 |
| `model_executor/models/test_attention_gate_activation.py` | 1 → 0 | 合并 | 五个独立 activation/scaling 分支合并到源模块对应的 npu/test_deepseek_v2_attention_gate.py；不再单建 feature 文件。 |
| [model_executor/models/test_deepseek_v2_construction.py](../tests/unit/model_executor/models/test_deepseek_v2_construction.py) | 23 → 22 | 保留并删并/改写 | DeepSeek-V2/V3 A/F dense/MoE/Attention weight allocation、gate dtype/checkpoint加载、CUDA 内外 router、M2N shared 所有权、native factory 刷新、FP16 routed-scale 交付与模型层的 SP/激活拒绝。 |
| [model_executor/models/test_deepseek_v2_proxy.py](../tests/unit/model_executor/models/test_deepseek_v2_proxy.py) | 19 → 18 | 保留并删并/改写 | DeepSeek proxy 的 send/yield/recv 协议和有效路由载荷、FP16/scaling/shared 输出、TP 分片/补零/gather、native/async forward 分流与缺 metadata 拒绝。 |
| [model_executor/models/test_deepseek_v2_sequence_parallel.py](../tests/unit/model_executor/models/test_deepseek_v2_sequence_parallel.py) | 2 → 2 | 保留 | NPU async Attention SP 的 gather/本地 dense/重新 scatter 与跨层 token 顺序和数值；把 shared output 放回正确 token 布局，不能由 non-SP proxy 用例替代。 |
| [model_executor/models/test_deepseek_v2_weight_policy.py](../tests/unit/model_executor/models/test_deepseek_v2_weight_policy.py) | 5 → 4 | 保留并删并/改写 | 按 layer/stage/dense-vs-MoE/gate/shared ownership 分类、一次消费 checkpoint iterator、向 native loader 交付正确权重，以及 M2N/async shared 权重放在 Attention。 |
| [model_executor/models/test_deepseek_v4_cam_layout.py](../tests/unit/model_executor/models/test_deepseek_v4_cam_layout.py) | 2 → 2 | 保留 | V4 Hash router 的 token IDs/FP32 logits 在 profile 与真实 SP shard 上保持相同边界，并在 compute 后还原 padded parent tokens。 |
| [model_executor/models/test_deepseek_v4_construction.py](../tests/unit/model_executor/models/test_deepseek_v4_construction.py) | 8 → 6 | 保留并删并/改写 | CUDA V4 A/F stage/head 权重所有权、MegaMoE 后端拒绝、spec decode 公共 MTP buffer 以及 native MoE hash-layer ABI。 |
| [model_executor/models/test_deepseek_v4_proxy.py](../tests/unit/model_executor/models/test_deepseek_v4_proxy.py) | 6 → 6 | 保留并删并/改写 | V4 独有的 token IDs wire/padding、无 IDs/错误维度在发送前拒绝、双角色非法计算入口错误；观察 peer 而非 metadata lookup 内部次序。 |
| [model_executor/models/test_deepseek_v4_shared_experts.py](../tests/unit/model_executor/models/test_deepseek_v4_shared_experts.py) | 4 → 4 | 保留 | V4 shared experts 的四种 replicated/sharded×TP/SP 所有权，all-gather/reduce-scatter/all-reduce 的输出与 stage 依赖；保留有效 MRV2 context，去掉已禁止的 V1 轴。 |
| [model_executor/models/test_deepseek_v4_weight_policy.py](../tests/unit/model_executor/models/test_deepseek_v4_weight_policy.py) | 4 → 4 | 保留 | CUDA V4 decoder 内 attn/ffn/mHC/head 的明确 checkpoint 规则、全局 fallback、one-shot loader 不漏不重消费；CUDA classifier 与 Ascend 独立。 |
| [model_executor/models/test_forward_context.py](../tests/unit/model_executor/models/test_forward_context.py) | 5 → 3 | 保留并删并/改写 | 从公开 additional_kwargs 读取 metadata、不读历史属性；profiling 跨两层仍执行匹配的 connector send/recv 和最终输出。 |
| [model_executor/models/test_model_utils.py](../tests/unit/model_executor/models/test_model_utils.py) | 7 → 7 | 保留 | 公共 model_config 拷贝/非 AFD identity、模型别名转换/unknown 拒绝、NPU 导入边界和支持 transport 的正确别名；不维护 isinstance/MRO 接线。 |
| [model_executor/models/test_qwen3_5_contract.py](../tests/unit/model_executor/models/test_qwen3_5_contract.py) | 6 → 6 | 保留 | Qwen3.5/3.6 hybrid 特有的 TP 完成输出、native internal router 归属、cache 模式早拒绝、text-only vision stage 及 AFD metadata/capacity；非 Qwen3 可替代。 |
| [model_executor/models/test_qwen3_5_weight_policy.py](../tests/unit/model_executor/models/test_qwen3_5_weight_policy.py) | 2 → 2 | 保留 | hybrid linear_attention/self_attention 与 decoder norm/MLP 的 checkpoint 归属、native loader 一次消费；没有复制整套 native attention 数值测试。 |
| [model_executor/models/test_qwen3_moe_construction.py](../tests/unit/model_executor/models/test_qwen3_moe_construction.py) | 16 → 12 | 保留并删并/改写 | Qwen3 A/F 真实 named_parameters 所有权、Attention gate 的原生 checkpoint 路径、native model 构造 hook、无 local experts 的 LM metadata、模型层真实不支持项、FFN 输出与错误角色拒绝。 |
| [model_executor/models/test_qwen3_moe_weight_policy.py](../tests/unit/model_executor/models/test_qwen3_moe_weight_policy.py) | 4 → 4 | 保留并删并/改写 | Qwen3 checkpoint 的全局/Attention/norm/MLP 所有权、Attention routing 时 gate 两边都加载、one-shot 原对象交给 native loader、config 驱动的 MoE layer inventory。 |
| `model_executor/models/test_w4a8_attention_gate.py` | 1 → 0 | 合并 | W4A8 stacked/list 权重、int32 view、scale/bias 与 per-channel 传给真实外部 Ascend MLP API 的 ABI 检查，合并到实际源模块测试。 |
| [model_executor/test_async_cam_dbo.py](../tests/unit/model_executor/test_async_cam_dbo.py) | 3 → 3 | 保留 | 同一物理 CAM window 的跨 ubatch recv-before-next-send 协议，owner 消费唯一结果、无 DBO 时直接 recv 和失败事务不重新发送；是可观察 wire 次序。 |
| [model_executor/test_async_cam_layered.py](../tests/unit/model_executor/test_async_cam_layered.py) | 14 → 11 | 保留并删并/改写 | layered 执行器提交给外部 CANN custom ops 的量化 ABI、layer selector、capacity/scale 输出，公开权重提取/不兼容量化拒绝与完整 target/draft coverage 门禁。 |
| [model_executor/test_async_cam_layout.py](../tests/unit/model_executor/test_async_cam_layout.py) | 1 → 1 | 保留 | 公开 layout helper 的 uneven TP 分片、padding 后 routes/IDs/logits 同步裁剪、gather 恢复 parent tokens；检查 tensor 值与有效 layout 输出。 |
| [model_executor/test_deepseek_attention_metadata.py](../tests/unit/model_executor/test_deepseek_attention_metadata.py) | 5 → 5 | 保留并删并/改写 | SFA/MLA/DSA rope 和 builder 输入不互相污染，public RopeDataProxy lookup/slice/pad 数值、stage 独立且 replay 地址稳定更新；只检查实际 native consumer 能读到的输出。 |
| [model_executor/test_deepseek_v4_hash_ids.py](../tests/unit/model_executor/test_deepseek_v4_hash_ids.py) | 10 → 8 | 保留并删并/改写 | Hash IDs 的 int64/flatten/计数边界、remote peer 载荷与缺 IDs 时不发送，以及模型 FP32 gate logits/profile routing 输出。 |
| [model_executor/test_routing_simulator.py](../tests/unit/model_executor/test_routing_simulator.py) | 4 → 4 | 保留并删并/改写 | 公开 benchmark routing strategy 的确定性/跨 EP rank 平衡、用户 topn/不整除专家池拒绝、归一化权重/地址复用及真实 vLLM registry/capture 结果。 |
| [package/test_ascend_build_files.py](../tests/unit/package/test_ascend_build_files.py) | 10 → 9 | 保留并删并/改写 | setup.py 的平台默认/显式覆盖、实际 select_ops CLI 的 SOC 清单、子集、错误和 SHMEM 筛选；打包行为与运行时不同。 |
| [package/test_ascend_op_host_build.py](../tests/unit/package/test_ascend_op_host_build.py) | 2 → 2 | 保留 | 真实 CMake probe 验证嵌套 tiling 源文件进入 codegen/tiling 库，以及 opapi_math 链接而非聚合 opapi；文件存在和 registry 测试无法发现这些遗漏。 |
| [package/test_cam_async_build.py](../tests/unit/package/test_cam_async_build.py) | 5 → 5 | 保留 | 950 拒绝 async 算子、混合算子选择顺序、sdist 保留构建输入/许可证、staging 冲突、跳过 ACLNN 时仍传递绑定 SOC；均是可观测构建边界。 |
| [package/test_package.py](../tests/unit/package/test_package.py) | 2 → 2 | 保留 | 入口元数据和真实 register_afd 的外部注册结果、原生模型保留、CUDA/NPU 分支与幂等；不检查私有注册表。 |
| [test_async_cam_startup_wiring.py](../tests/unit/test_async_cam_startup_wiring.py) | 13 → 14 | 保留并删并/改写 | CPU 启动接线的 two-stream core/Ascend env 门禁、HCCL 前拒绝、native warmup/capture 与 service announce 的就绪/失败次序、worker stop/join/release、正常 receiver 退出后真实 graph close 释放资源及失败保留、factory 选择与 one-shot model/draft loading。 |
| [test_envs.py](../tests/unit/test_envs.py) | 3 → 1 | 保留并删并/改写 | 两个受支持诊断环境变量的公开字面名称、缺省/真假输入；单一参数表，不复用被测 ENV 名称当 oracle。 |
| [v1/worker/npu/test_attention_model_runner_v2.py](../tests/unit/v1/worker/npu/test_attention_model_runner_v2.py) | 1 → 1 | 保留 | draft context 使用自己的 batch 长度和 transaction ID，正常/异常退出后 following target 继续原事务且 native DP opt-out 恢复；非普通 stage metadata。 |
| [v1/worker/npu/test_ubatch_runner.py](../tests/unit/v1/worker/npu/test_ubatch_runner.py) | 4 → 4 | 保留 | Ascend DSA compressor 的 actual/padded rows、stage 持久 RoPE/block table 地址及数据隔离、prepare 失败恢复 native factory；与仅 metadata materialize 和 CUDA 对齐逻辑不同。 |
| [v1/worker/test_async_ffn_graph.py](../tests/unit/v1/worker/test_async_ffn_graph.py) | 5 → 5 | 保留并删并/改写 | 单 stream graph 事务的外部 graph replay/reset/fence 次序、peer recv/send 仅在 capture 的契约、错误后禁止重试/释放，以及 weakref 证明临时资源真实保留/释放。 |
| [v1/worker/test_async_ffn_runtime.py](../tests/unit/v1/worker/test_async_ffn_runtime.py) | 2 → 1 | 保留并删并/改写 | async eager 的实际返回 tensor、dtype 和原始 header：non-layered 有效行/零 token 浮点占位与 layered 容量输出；layered 输出实际依赖收到的层号、专家分组及逐行 scale。receiver 成功/失败生命周期用例移至 CPU startup owner，并继续走真实关闭。 |
| [v1/worker/test_async_ffn_two_stream.py](../tests/unit/v1/worker/test_async_ffn_two_stream.py) | 6 → 6 | 保留并删并/改写 | 双 stream 的 receive/compute/send event 依赖、第三项前退休、full replay 的 slot lifetime、warmup peer failure/timeout、不重复提交和失败后不释放/不重试；weakref/外部 stream/graph API 证明主机编排契约。 |
| [v1/worker/test_attention_model_runner.py](../tests/unit/v1/worker/test_attention_model_runner.py) | 16 → 13 | 保留并删并/改写 | 公共 metadata installer 的单/双 stage 控制载荷、padding/真实 token 长度、native kwargs 保留、child 不重发、provider/no-control-plane 与 capture/profile flags。 |
| [v1/worker/test_cuda_graph.py](../tests/unit/v1/worker/test_cuda_graph.py) | 14 → 11 | 保留并删并/改写 | CPU CUDA policy 的 eager/graph/ubatching 支持、按 stage/DP/TP 值构造确定 graph key 与 A/F 聚合；graph key 规则只有此模块 owner。 |
| [v1/worker/test_dbo.py](../tests/unit/v1/worker/test_dbo.py) | 5 → 3 | 保留并删并/改写 | public yield 的外部 torch custom-op 交付、mutation/fake schema，缺 optional runtime 时输入不变；不检查内部 register/helper 调用数。 |
| [v1/worker/test_ffn_metadata.py](../tests/unit/v1/worker/test_ffn_metadata.py) | 2 → 2 | 保留 | 唯一 CPU token 计数聚合/TP→DP 投影 oracle，含 idle、fallback、不能整除的 subgroup；其他 runner 文件不再重测薄包装。 |
| [v1/worker/test_ffn_model_runner.py](../tests/unit/v1/worker/test_ffn_model_runner.py) | 32 → 25 | 保留并删并/改写 | GPU FFN 控制面驱动的多 layer/stage 结果、隐藏状态/IDs/router payload、native update 接口、graph eager/replay 分流、scheduler 拒绝，以及 receiver 取消/超时/错误和 shutdown 资源释放。 |
| [v1/worker/test_ffn_routed_experts_model_runner.py](../tests/unit/v1/worker/test_ffn_routed_experts_model_runner.py) | 12 → 9 | 保留并删并/改写 | routed FFN 的 DBO recv/send/stream join 协议、单 stage 顺序、graph miss 控制 flag、拒绝非 routed connector、vLLM 权重视图/FP8 scale 与不兼容 backend 拒绝。 |
| [v1/worker/test_model_runner_v2.py](../tests/unit/v1/worker/test_model_runner_v2.py) | 45 → 32 | 保留并删并/改写 | GPU MRV2 capture/replay 的控制消息与 stage 载荷、错误/源契约变化的拒绝、native KV-only context 不发送、重复事务、worker factory 隔离及 V4 对齐/SM/workspace 输入。 |
| [v1/worker/test_npu_mla_graph.py](../tests/unit/v1/worker/test_npu_mla_graph.py) | 7 → 6 | 保留并删并/改写 | MLA graph metadata 按 layer-major/stage-minor 合并、workspace 共享与错误拒绝，以及异常后的原 native ForwardContext 恢复。 |
| [v1/worker/test_npu_model_runner_v2.py](../tests/unit/v1/worker/test_npu_model_runner_v2.py) | 20 → 16 | 保留并删并/改写 | NPU execute/capture 的 native provider/DP scope、CAMP2P many-to-one padding、ACL descriptor 保留、MLA replay 的跨 stage 记录、Ascend input slicing/上下文隔离和 async local DBO。；异常后恢复被临时替换的 native 公共 binding，保护后续非 AFD consumer。 |
| [v1/worker/test_npu_runtime.py](../tests/unit/v1/worker/test_npu_runtime.py) | 35 → 21 | 保留并删并/改写 | NPU FFN 独立的 native Ascend ForwardContext、structured shared 输出、真实 loop/chunk 边界、ACL graph 调度、CPU affinity、错误传播、stop/join/release 和 timeout 重试语义；不复制共享 metadata mixin。 |
| [v1/worker/test_npu_ubatch_outputs.py](../tests/unit/v1/worker/test_npu_ubatch_outputs.py) | 2 → 2 | 保留 | Ascend native V2 (hidden, list[aux]) 输出 cat 的实际值，以及 scoped native merge binding 的恢复；普通 torch.cat/切片不能处理这个输出形状。 |
| [v1/worker/test_runtime_classpaths.py](../tests/unit/v1/worker/test_runtime_classpaths.py) | 3 → 3 | 保留 | 配置发布的 FQCN 可解析、GPU worker 导入不要求可选 connector backend、NPU runtime 在隔离子进程可解析；不把成功导入当设备执行。 |

## 消失或改名的定义及依据

这里逐项记录原文件中不再存在的测试定义。标注“合并/迁移/改名”的项目仍有对应覆盖。

### `compat/npu/test_ascend_config_lifecycle.py`

删除两个模拟生命周期 probe 及其子进程/hash/手写 native backend/graph-size 框架；保留真实 AFD factory AST 执行。

- `test_parent_and_fresh_child_hashes_and_cached_graph_sizes`：子进程仅重跑手写 upstream 模拟器；hash 与人为图尺寸筛选不提供真实 Ascend 父子进程证据，AFD wrapper 的序列化输入/重入由配置测试负责。
- `test_platform_failure_restores_backend_and_scoped_state`：config_validation 的真实 AFD wrapper 成功/异常矩阵已覆盖后端/DBO 恢复；此处重建 native platform 及私有临时状态。
- `test_runtime_bootstrap_failure_keeps_published_config`：与 native validation 失败后仍保留已发布配置/缓存的用例重复，不额外维护同一恢复路径的早期失败位置。
- `test_runtime_bootstrap_reaches_config_without_mutating_input`：逐项检查复制的 upstream runtime bootstrap 字段转发；AFD namespace/输入不变由同文件主用例覆盖。
- `test_target_derived_fields_are_ignored_as_constructor_inputs`：检查复制的上游纯派生字段过滤列表，不对应 AFD 差异；不为 upstream 私有派生字段逐个维护测试。

### `compat/npu/test_dsv4_async_validation.py`

合并到 test_feature_validation.py，不再单独构造私有 helper 的额外参数类型。

- `test_dsv4_accepts_the_camp2p_connector`：合并至 tests/unit/compat/npu/test_feature_validation.py；合并到 test_feature_validation.py，不再单独构造私有 helper 的额外参数类型。
- `test_dsv4_async_accepts_target_baseline`：合并至 tests/unit/compat/npu/test_feature_validation.py；合并到 test_feature_validation.py，不再单独构造私有 helper 的额外参数类型。
- `test_dsv4_async_rejects_deferred_features`：合并至 tests/unit/compat/npu/test_feature_validation.py；合并到 test_feature_validation.py，不再单独构造私有 helper 的额外参数类型。
- `test_dsv4_async_requires_attention_side_gate`：合并至 tests/unit/compat/npu/test_feature_validation.py；合并到 test_feature_validation.py，不再单独构造私有 helper 的额外参数类型。
- `test_dsv4_rejects_connectors_without_a_token_id_channel`：合并至 tests/unit/compat/npu/test_feature_validation.py；合并到 test_feature_validation.py，不再单独构造私有 helper 的额外参数类型。

### `compat/npu/test_feature_validation.py`

DSpark 去掉三个不改变被测规则的 role/多错误组合；模型/TP/SP/DBO 的 18 项全交叉压成五个独立分支；合并 DSV4 私有 helper 文件的五种实际配置。

### `compat/npu/test_routing_ids.py`

删除仅演示普通 Python 被 trace 固化的反例、其 class 和可变全局 flag；它不执行 AFD op。

### `compat/npu/test_runtime_config.py`

接收 config/test_async_cam_dbo.py 的三项行为；移除不影响逻辑的 DP 交叉与阈值 0/1 重复。

### `compat/patches/npu/test_mrv2_ubatching.py`

交叉核对完成：provider/DP scope 与默认/randomize 输入策略收敛到此 CPU owner；NPU runner 文件仅保留 native 协商/输入/MLA/stage 的独立行为。

### `compat/patches/test_async_dp_dplb_runtime.py`

- `test_native_dplb_distributes_tied_burst_with_optimistic_counts`：仅验证 vLLM 自身的负载均衡/完成请求行为，AFD UT 不拥有。
- `test_native_dplb_releases_completed_request`：仅验证 vLLM 自身的负载均衡/完成请求行为，AFD UT 不拥有。
- `test_native_dplb_uses_reported_request_counts`：仅验证 vLLM 自身的负载均衡/完成请求行为，AFD UT 不拥有。

### `compat/patches/test_async_dp_engine.py`

重绑定和 client alias 改查入口执行结果；去掉只反映内部 task helper 调用的 flags 断言，补查 ADD 消息。

补充核查已用固定上游源码确认 constructor/getter/utility 的独立 output-task 启动路径；移除 add-request 尾部启动调用仍可交付结果。维持原删减，不恢复 output-ready 标记断言，详见两项疑点的核查。

- `test_async_dp_engine_patch_reload_is_idempotent`：每个用例通过 loader 重载模块，且非 async/native 分支已有相同输出检查；独立用例重复检查 dp.kind。
- `test_engine_starts_without_optional_logging_config`：只在 fake 上手工删除非 AFD 配置属性；未建立该形状在当前已固定运行时中的独立 AFD 输入契约，正常启动/关闭已有覆盖。

### `compat/patches/test_config_validation.py`

- `test_config_validation_finalizes_async_attention_patch_after_ascend`：仅断言三个内部 installer 的具体调用链，没有验证 async 安装结果；该行为由 async engine 与 early binding 测试负责。
- `test_patch_preserves_native_dbo_model_restrictions`：只执行 fake native 自己的限制方法，并检查方法身份；没有经过 AFD 配置处理。真实 native 拒绝与 AFD 后端放宽由结果矩阵覆盖。

### `compat/patches/test_engine_core.py`

保留 readiness log 时序：deploy-afd-k8s/templates/pod.yaml 实际消费该日志作为 FFN ready 条件；不是普通日志格式测试。移除历史 helper 不存在及 fake original_init 标记检查。

- `test_ffn_early_init_satisfies_native_ready_handshake`：改名为 test_ffn_noop_scheduler_exposes_native_ready_protocol；原测试未执行真实 early-init，不再由名称宣称该证据。
- `test_ffn_noop_scheduler_tolerates_late_patch_load`：与 early_init 的 direct noop protocol 检查完全重叠，后者还覆盖 ec_connector 和 unfinished_requests；两者都没有模拟各自声称的加载时序。

### `compat/patches/test_gpu_mrv2_ubatching.py`

删除 GraphManager 内部 dispatch 搜索次数/顺序断言；KV 初始化由符号/参数检查改为实际 child metadata 和 native factory 返回结果。

- `test_attention_kv_initialization_scopes_factory_and_preserves_native_arguments`：改为 test_kv_initialization_scopes_dp1_metadata：检验实际 child metadata 和退出后的 native factory 结果，移除参数转发快照。
- `test_dp2_keeps_native_coordination_and_all_dispatch_arguments`：只对 native collaborator 的转发参数与调用次数做镜像断言；DP2 的 sync/graph 最终 descriptor 与 agreement 测试保留。
- `test_factory_keeps_native_runner_and_changes_only_dp1_instance`：构造器参数/类型和实例私有方法覆盖检查；DP1 的实际子上下文输出与 KV 初始化工厂用例提供行为覆盖。
- `test_scopes_restore_native_alias_after_exception_and_nested_call`：符号身份和人为嵌套本身没有独立运行结果；KV 初始化的成功/异常真实工厂结果检查覆盖恢复。

### `compat/test_ascend_ops.py`

移除内部 P2P loader 调用次数断言。

### `compat/test_vllm.py`

- `test_version_support_matches_target_release`：同一版本谓词由门禁表调用，移除直接重复测试。

### `config/test_async_cam_dbo.py`

合并到 tests/unit/compat/npu/test_runtime_config.py；有效/无效阈值边界及 role/connector 限制保留。

- `test_async_normalization_does_not_enable_ffn_or_synchronous_dbo`：合并至 tests/unit/compat/npu/test_runtime_config.py；合并到 tests/unit/compat/npu/test_runtime_config.py；有效/无效阈值边界及 role/connector 限制保留。
- `test_native_ubatching_revalidates_thresholds`：合并至 tests/unit/compat/npu/test_runtime_config.py；合并到 tests/unit/compat/npu/test_runtime_config.py；有效/无效阈值边界及 role/connector 限制保留。
- `test_native_ubatching_uses_dbo_at_every_dp_size`：合并至 tests/unit/compat/npu/test_runtime_config.py；合并到 tests/unit/compat/npu/test_runtime_config.py；有效/无效阈值边界及 role/connector 限制保留。

### `config/test_validation.py`

接收 runtime 文件中的非重复 validator 条目；CPU-safe shared builder 由 fakes/vllm_config.py 提供。

### `connectors/test_async_cam_connector.py`

移除日志 helper 内容/顺序和私有 pending queue 布局；关闭改为 weakref 证明各 stage payload 释放，receive anchor dtype 改查实际送入 CANN op 的 tensor。

- `test_async_connector_disables_dp_metadata_control_plane`：factory construction 已明确断言 control_plane is None。
- `test_async_connector_init_creates_attention_first_hccl_group`：主要复刻 init_afd_process_group 的参数/回调身份和私有 placeholder；rank layout、factory 及实际 CAM ABI 由各自测试负责。
- `test_async_connector_init_uses_one_hccl_group_for_all_dp`：重复 topology 已说明的 world/rank，并只检查 group builder 调用参数。
- `test_async_rejects_fused_shared_expert_ids`：直接调用 feature_validation 私有 helper；移入该模块公开 validator 的实际配置表。
- `test_async_select_experts_native_fallback`：数学行为完全由 native Ascend router 决定；AFD 委托/返回与拒绝由本文件保护，真实 fused 数值由 tests/npu 保护。
- `test_dispatch_failure_does_not_leave_pending_routing`：只断言私有 pending queue；没有独立输出，且失败后的 native transport 重试不属于受支持事务语义。

### `connectors/test_async_cam_startup.py`

删除 private store key/helper 断言与 callback 计数；通过 peer 观察到的错误/就绪结果判断。

- `test_empty_graph_signature_cannot_advertise_support`：合并无 loaded executor 的错误表。
- `test_full_rejects_incompatible_layout`：合并 signature/layer 不一致的字面错误表。
- `test_mixed_requested_modes_are_not_silently_downgraded`：合并到执行计划拒绝表。
- `test_unsupported_full_rejects_without_automatic_fallback`：合并到执行计划拒绝表。

### `connectors/test_camp2p_connector.py`

算子参数是 CANN ABI/peer wire contract，保留；移除 options helper 调用次数。

- `test_camp2p_extra_info_coerces_integer_bool_values`：检查配置值 1 转成 gate=true，但该 gate 在当前同步 CAMP2P 中被 feature validator 明确拒绝；通用布尔转换已有 common config 覆盖。
- `test_camp2p_extra_info_rejects_unknown_mix_placement`：改用一般 unknown key 的公开 schema 拒绝，不为已经淘汰的字段名单逐个保留专用用例。
- `test_camp2p_recv_attn_output_drives_the_operator_ids_mode`：test_camp2p_token_ids 已用相同入口验证开/关 IDs mode、ids dtype 和顺序，并可在有 torch 而无 Ascend 的 host 执行。

### `connectors/test_camp2p_token_ids.py`

保留 host stub，因为合并进 test_camp2p_connector 会引入 torch_npu gate、丢掉 CPU-only-with-torch 可执行性；删除的是 NPU-gated 文件中的重复 recv mode 测试。

### `connectors/test_deepep_m2n_connector.py`

合并四种重复 plan_stage 设置为字面输入/输出表；host fake stream 只验证依赖顺序，不是设备并行或内存安全的验证。

- `test_connector_is_registered`：其余 factory construction 已通过名称实例化真实 connector；类名/module 字符串不增加行为覆盖。
- `test_handshake_accepts_consistent_ranks`：并入同一 handshake 成功表的 TP1/TP2 设置，保留 alignment/network/capacity 的独立预期。
- `test_handshake_gives_ffn_the_attention_tp_size_capacity_and_ladder`：合并进 test_handshake_resolves_attention_transport_settings，TP1/TP2 共用成功协议表。
- `test_no_sync_bucket_limit`：内部 helper 的计算已被公开 bucket_ladder 和 plan_stage 输出表覆盖；不独立维护一份内部算式表。
- `test_plan_stage_gives_buckets_above_the_no_sync_limit_the_capacity`：合并 plan_stage eager capacity fallback 输出表。
- `test_plan_stage_keeps_a_ladder_bucket_for_eager_steps_without_host_sync`：合并 plan_stage no-sync eager 与 threshold 输出表。
- `test_plan_stage_rounds_graph_buckets_up_the_ladder_without_host_sync`：合并 plan_stage graph/idle/capture bucket 输出表。
- `test_plan_stage_sizes_the_bucket_for_the_tp_slice`：与其他相同入口/设置的 plan_stage 输出用例合并为一张字面预期表。

### `connectors/test_deepep_m2n_network.py`

unknown network key 保留一个通用反例，不为已退役键逐个保留拒绝用例。

- `test_network_config_survives_a_round_trip`：同样的三字段值和 nested round trip 已由 test_deepep_m2n_connector 的 test_config_parses_network_section_and_tunables 经真实 NetworkConfig 验证。

### `connectors/test_p2p_connector.py`

metadata 序列化测试移入 test_metadata；wire send 测试移到专家 payload 的公开 send_attn_output 路径。

- `test_eager_state_does_not_cache_hidden_buffers`：原删除依据不成立；它保护 graphs 服务中 eager hidden buffer 不随长度累积的独立资源约束。改由 `test_eager_steps_release_hidden_buffers_with_graphs_enabled` 的分配弱引用检查保留，不恢复私有 cache 布局断言；graph receive 地址复用不是该分配路径的替代覆盖。
- `test_graph_state_adds_input_ids_buffer_when_hidden_buffer_exists`：只检查内部 cache key/对象；实际 graph receive 的缓冲区重用由专家 payload 协议测试验证。
- `test_p2p_connector_preserves_deepseek_attention_side_gate`：仅检查仍未初始化，与 gate 是否被保留无关；不能检测名称声称的回归。
- `test_p2p_dp_metadata_serialization_uses_json_payload`：移至 connectors/test_metadata.py，同一实际 JSON encode/decode 行为保留。
- `test_p2p_hidden_state_send_uses_registered_custom_op`：只验证内部 _send_hidden_states wrapper 的参数转发；send_attn_output 的真实 wire custom-op 消息测试替代。
- `test_p2p_module_exports_connector_class`：只检查类的 `__module__` 字符串；实际 factory construction 已验证可加载。
- `test_p2p_topology_supports_equal_and_integer_multiple_attention_counts`：直接调用已在 distributed/test_topology 完整验证的 build_rank_mapping，没有执行 connector。

### `connectors/test_p2p_experts_contract.py`

send 观察点下移至 torch.ops.vllm.afd_p2p_send；meta tensors 仅用于 wire shape/dtype 检查，不作为设备执行证据。

- `test_attention_gate_reuses_hidden_state_send_for_router_logits`：原用例仅观察内部 _send_hidden_states 调用，合入公开 send_attn_output 的外部 custom-op wire 表。
- `test_attention_send_orders_token_ids_after_hidden_states`：相同 send 设置合入 wire 字段表，验证 IDs 顺序及 dtype。
- `test_ffn_gate_sends_only_hidden_states`：相同 send 设置合入 wire 字段表，验证无额外消息。
- `test_router_shape_is_validated_before_any_send`：原内部 helper spy 改成实际 send custom-op wire 观察；非法 payload 仍必须在任何消息发送前拒绝。

### `model_executor/models/npu/test_deepseek_v2_attention_gate.py`

合并 activation/W4A8 与空 rank 用例；32/16/8 组合分别压为 5/4/4 行；公共数值/外部 ABI 保留，删除继承 alias 与无结果的 cache 属性框架。

- `test_disabled_role_or_backend_does_not_import_npu_helper`：仅约束 lazy import/缓存字段缺失；未检验模型输出，角色构造和量化行为已有明确结果覆盖。
- `test_missing_canonical_scale_keeps_the_native_attribute_error`：构造缺少原生 W8A8 必备属性的 owner，只检查 Python AttributeError。
- `test_only_stacked_w8a8_owners_receive_a_scale`：仅检查实现缓存是否存在，没有走相应量化 FFN；W4A8/NONE 的真实外部 ABI 由合并后的用例覆盖。
- `test_present_invalid_cache_is_not_silently_repaired`：人为把 native 派生 cache 设为 None 后断言 None，没有已支持的用户输入路径。
- `test_v2_family_models_inherit_the_post_quant_lifetime`：未覆写的继承方法逐 alias 重跑；不为继承关系复制行为测试。

### `model_executor/models/npu/test_deepseek_v4.py`

去掉不可部署 PP 非末 stage 参数及 shared 权重同规则 suffix 全交叉。

### `model_executor/models/test_attention_gate_activation.py`

32 组全交叉压成默认/自定义、BF16/FP16、topk 已缩放/未缩放的五行表；外部 native ABI 参数保留。

- `test_attention_gate_passes_activation_and_scaling_contract`：合并至 tests/unit/model_executor/models/npu/test_deepseek_v2_attention_gate.py；32 组全交叉压成默认/自定义、BF16/FP16、topk 已缩放/未缩放的五行表；外部 native ABI 参数保留。

### `model_executor/models/test_deepseek_v2_construction.py`

删除 fake construction call recorder 断言、继承/model_cls 接线；去掉不可部署的 PP2/CUDA-Async 和 ROCm AITER 参数。保留公共参数/加载结果与 native API 边界。

- `test_v32_indexer_buffer_is_allocated_only_on_attention`：只计数固定 torch.empty 尺寸，不能证明 indexer 使用了该 buffer；属于分配实现约束，不保留伪运行时证据。

### `model_executor/models/test_deepseek_v2_proxy.py`

删除 internal gate/topk/schedule 实参快照，connector/gate fake 与 V4/GPU 共享，真实 CUDA 用例移出 UT。

- `test_routing_proxy_matches_native_router`：迁往 tests/gpu，保留真实 CUDA 对照；不算逻辑删除。

### `model_executor/models/test_deepseek_v2_weight_policy.py`

ownership 表按七个实际分支保留代表，删除同一 stage 下 q/k/v、投影、scale/bias 后缀交叉；不重复测试 native packing。

- `test_native_model_loader_packs_mha_qkv`：AFDDeepseekV2Model 不覆写 load_weights；该用例测试的是继承的 vLLM QKV packing，且重建四项 native helper。AFD 自己的 filter/迭代/委托结果保留。

### `model_executor/models/test_deepseek_v4_construction.py`

- `test_v4_decoder_constructor_rejects_unknown_role`：绕过 parse_afd_config，构造当前公开配置入口不会接受的 role。
- `test_v4_decoder_layer_pins_sequence_parallel_off`：只断言内部 flag；GPU V4 的 SP 不支持由部署 validator 负责，native MoE ABI 也检查 SP=False。
- `test_v4_ffn_model_has_no_head_mhc_parameters_or_mtp_buffer`：改名聚焦 FFN head 权重；MTP 由公开 getter 验证，移除 private buffer 检查。

### `model_executor/models/test_deepseek_v4_proxy.py`

- `test_remote_v4_ffn_validates_token_ids_before_metadata_lookup`：改名为 test_remote_v4_ffn_rejects_bad_ids_before_sending；检查错误前没有 peer 消息。

### `model_executor/models/test_forward_context.py`

空 rank routed/shared 工作移入 npu/test_deepseek_v2_attention_gate.py 并复用 fakes/ascend_moe.py，删除在测试内重建五个 native 模块的辅助代码；layout 改用真实 AFD helper。

- `test_async_model_forward_preserves_pp_boundaries`：AFD 当前 validator 明确拒绝 PP；两项 fake 非首/非末 PP 路径无有效部署入口，且主要 spy 内部 schedule。
- `test_deepseek_afd_ffn_skips_empty_rank_local_moe_work`：移至 npu/test_deepseek_v2_attention_gate.py 并复用 native ABI fake；空 routed/shared 四种组合保留，重复 scaling 交叉由 activation 表负责。

### `model_executor/models/test_qwen3_moe_construction.py`

去掉 fake builder 清单、proxy 类型和内部 flags；native model hook 已改为检验构造出的 decoder named_parameters。

- `test_attention_role_gives_the_proxy_the_attention_tp_size`：只检查内部 proxy.tp_size 参数接线；共享 RemoteFFNProxy 的 TP dispatch/gather 输出与真实 TP2 用例覆盖行为。
- `test_attention_role_selects_routing_proxy_by_connector_only`：人为绕过 model validator，把已明确拒绝的 P2P + Attention gate 配置直接送入 layer；不属于当前有效构造路径。
- `test_ffn_role_preserves_native_dense_layer_schedule`：只检查 is_moe_layer 与 fake 构造调用；相同 _is_moe_layer 规则由 get_experts_layer_indices 的实际模型清单验证。
- `test_gate_on_attention_is_accepted_with_deepep_m2n`：私有 validator 的无异常接线与真实带 gate 角色构造配置重复；拒绝条件矩阵保留。

### `model_executor/models/test_qwen3_moe_weight_policy.py`

标准 ownership 表 13→6，按 stage/全局/gate 的规则分区；Attention gate 表保留 gate/shared_gate/gate_proj 名称区别，防止把不同实际路径误分给 gate。

### `model_executor/models/test_w4a8_attention_gate.py`

16 组笛卡尔积压为三种非空 ABI 和一种空输入；不宣称 fake 返回值证明 W4A8 核数值。

- `test_w4a8_cam_mlp_contract`：合并至 tests/unit/model_executor/models/npu/test_deepseek_v2_attention_gate.py；16 组笛卡尔积压为三种非空 ABI 和一种空输入；不宣称 fake 返回值证明 W4A8 核数值。

### `model_executor/test_async_cam_layered.py`

- `test_layered_debug_logging_does_not_read_metadata`：私有日志 helper 的格式/关键字测试，不是 graph 执行或设备同步验证。
- `test_runner_preserves_raw_context_and_chunk_capacity`：原测试只 spy 内部 executor 实参；仅以返回原 header 说明替代充分的依据不足。现已由 runtime 用例的层号/专家分组/scale → 实际发送数值，以及容量/原 header 检查保留其行为价值，不恢复实参快照。
- `test_runner_switch_off_does_not_touch_model_or_ops`：无断言，只在手写 fake 上运行关闭开关的私有函数。

### `model_executor/test_deepseek_attention_metadata.py`

删除 _data/idx、私有 buffer 字典布局和 native getter 次数断言，保留值、源不变和公共 tensor storage。

### `model_executor/test_deepseek_v4_hash_ids.py`

- `test_model_sharded_ids_are_not_split_again`：与 aligned IDs 输出用例经过同一分支，只换字面值；真实 explicit model-sharded IDs 的 remote wire 测试保留。
- `test_rejects_global_ids_for_local_router`：与 unalignable count 拒绝同一规则；remote IDs 无消息/错误仍保留。

### `model_executor/test_routing_simulator.py`

private cycle/buffer 测试改走 route_tokens；删除不可能由正常 native process group 产生的 ep_size=0、越界 rank 等手工对象参数。

### `package/test_ascend_build_files.py`

- `test_ascend_operator_registry_covers_both_soc_generations`：重复清单/源码布局并强加可选 metadata 必填；由 CLI 输出与构建消费结果负责。

### `test_async_cam_startup_wiring.py`

删除不改变逻辑的 DBO 轴、重复布尔拼写和私有 thread 状态；保留 operator peer/readiness 的对外时序。

从 runtime owner 接收 receiver 成功/失败用例；使用真实 worker 停止和 runner graph close，以 graph 弱引用观察成功释放、失败保留。主机 graph/transport 双替身不证明 NPU 设备完成。

### `test_envs.py`

三个重复 loop 测试归并为一个表；不同 reader 都必须读正确 env。

- `test_async_cam_layered_gmm_defaults_and_boolean_values`：合并到 test_diagnostic_environment_flags，以公开字面 env 名称为输入。
- `test_force_balanced_topk_ids_env_accepts_true_values`：合并到 test_diagnostic_environment_flags，同一布尔输入规则只保留一个测试体。
- `test_force_balanced_topk_ids_env_defaults_to_disabled`：合并到 test_diagnostic_environment_flags，缺省值仍有独立参数行。

### `v1/worker/test_async_ffn_graph.py`

观察 compute 产生的资源和 native graph API，去掉 private payload/output/replay_count 断言；不把 fake graph 当 NPU capture 验证。

### `v1/worker/test_async_ffn_runtime.py`

删除内部 compute 参数/call list 与私有 graph/loop status 断言，按 wire 输出判断。

评审后修正错误的统一容量期望：分别检查 non-layered 有效行与零 token 占位、layered 容量输出。原 receiver 用例只观察公共错误入口的替代说明不足，已移至 CPU startup owner 补上真实关闭，详见上述三项修正。

补充核查进一步让 layered fake 消费真实传入的层号、专家分组和逐行 scale，以字面预期的发送数值判断；删除无人读取的 `observed` 记录，保留既有参数化测试数量。

### `v1/worker/test_async_ffn_two_stream.py`

去掉私有计数器和 slot payload/output 布局断言，不以 fake streams 宣称 NPU 并发已验证。

### `v1/worker/test_attention_model_runner.py`

- `test_attention_runner_does_not_skip_single_stage_context`：install_afd_metadata 主用例已经检查同样的单 stage 控制载荷和 tokens。
- `test_attention_runtime_allows_full_decode_only_cuda_graph`：同上；保留 cuda_graph 模块唯一 policy 表。
- `test_attention_runtime_rejects_unsupported_cuda_graph_modes`：直接调用 cuda_graph 模块的同一 policy，既未执行 runner，也无独立覆盖。

### `v1/worker/test_cuda_graph.py`

- `test_make_ffn_graph_key_different_tokens_for_dp1_tp2`：同一输入展开/投影函数、同样 setup，合并为公共 key 字面输出表；不重复写四份测试体。
- `test_make_ffn_graph_key_dp1_tp1_unchanged`：同一输入展开/投影函数、同样 setup，合并为公共 key 字面输出表；不重复写四份测试体。
- `test_make_ffn_graph_key_expands_dp1_tp2`：同一输入展开/投影函数、同样 setup，合并为公共 key 字面输出表；不重复写四份测试体。
- `test_make_ffn_graph_key_expands_dp2_tp2`：同一输入展开/投影函数、同样 setup，合并为公共 key 字面输出表；不重复写四份测试体。

### `v1/worker/test_dbo.py`

- `test_dbo_yield_prefers_native_context_without_importing_legacy`：仅 native helper 调用和历史 import 禁止；迁移后的 CPU/classpath 测试已经保护 optional NPU 隔离。
- `test_dbo_yield_without_native_context_does_not_import_npu_runtime`：只调用 private helper 并 spy native dbo_yield；共享 public custom-op 路径保护 DBO 契约，无独立 AFD 结果。

### `v1/worker/test_ffn_model_runner.py`

- `test_attention_side_gate_processes_only_experts_layers`：与两 stage 的 static routing spec / wire 输出用例同一执行分支；后者还按 hidden/layer/router 值验证最终响应，不保留单 stage 的内部 model.calls 检查。
- `test_ffn_constructor_defers_connector_init_and_owns_model_and_graphs`：六处 patch 后只核对构造调用顺序、属性及私有 graph dict；不检验任何执行结果。
- `test_ffn_runner_graph_key_preserves_attention_peer_shapes`：重复 cuda_graph graph-key 字面输出表。
- `test_ffn_runner_makes_original_style_graph_key`：直接调用公共 utility，重复 cuda_graph 模块的输入排序表。
- `test_ffn_runner_matches_p2p_token_count_aggregation`：直接测 _make_ffn_dp_metadata；聚合规则由 ffn_metadata 唯一表、execute_model 两个 stage 的真实 ForwardContext 消费覆盖。
- `test_ffn_runner_projects_tensor_parallel_counts_to_dp_metadata`：同一投影规则由 ffn_metadata 表和执行上下文负责，不维护薄包装测试。
- `test_ffn_runner_steps_gpu_profiler`：仅一个 StepProfiler spy 的 step 次数；profiler 模块生命周期负责，FFN 主执行行为不需要单独造一份。

### `v1/worker/test_ffn_routed_experts_model_runner.py`

- `test_initialize_passes_moe_layers_and_experts_to_the_connector`：只断言内部 init_afd_connector kwargs；实际握手 placement/层一致性由 connector 协议测试覆盖。
- `test_load_model_takes_each_moe_layers_local_experts`：内部 converter 被 lambda 替换后检查 local_experts 字典接线，不证明加载出的权重或专家执行。
- `test_routed_runner_reuses_the_connector_driven_ffn_runner`：issubclass 继承结构断言，根规范明确交给 mypy。

### `v1/worker/test_model_runner_v2.py`

两张独立 validator 表转到 config/test_validation.py（CPU 可执行），不把转移算成行为删除；减少 helper/flag/private sidecar 断言。

- `test_dp1_native_token_seams_preserve_real_token_count`：仅执行 upstream dp_utils，没有 AFD 被测代码。
- `test_ffn_worker_selects_routed_runner_for_routed_connector`：六处 patch 后只断言 runner class 与重复的 triton 配置，未使用专家输出；公开 backend policy 和 routed 协议测试保留。
- `test_gpu_v2_validator_outcome`：移至 config/test_validation.py，CPU 无 vLLM 环境也可运行，非重复规则保留。
- `test_non_afd_model_config_and_native_runner_identity_are_unchanged`：get_afd_model_config 的非 AFD 返回 identity 已由 model_utils 测试；其余仅构造 untouched native sentinel。
- `test_npu_async_v2_constructor_validates_before_native_and_connector_init`：仅 native/factory 调用链和返回对象身份，无模型/peer 行为；配置门禁已转至 CPU owner。
- `test_npu_v2_validator_outcome`：移至 config/test_validation.py，CPU 无 vLLM 环境也可运行，非重复规则保留。
- `test_v2_capture_profile_only_rejects_partial_event_stream`：profile_only 在未设置 descriptor 截断时不改变此分支；重复 source_drift 表的 missing 事件拒绝。
- `test_v2_constructor_defers_connector_initialization`：仅 fake native init 事件与 connector identity；加载后的实际 announce 顺序与失败 cleanup 保留。
- `test_v2_load_model_reports_moe_layers_to_routed_connector`：只 spy 内部 connector init kwargs；真实 model inventory 与跨 rank handshake 一致性在各自 seam 检验。
- `test_v2_metadata_provider_has_complete_transaction_and_dp_dependencies`：metadata 长度/事务递增和 capture DP counts 已由 owner 的 installer 与 repeated execute 验证。
- `test_v2_validator_accepts_deepep_m2n_attention_tp`：与 config owner 的 routed transport policy 合并；在那里补 TP2 的独立边界。
- `test_v2_validator_rejects_deepep_m2n_ffn_tp`：与 config owner routed policy 合并；FFN TP2 拒绝边界转移。
- `test_v2_validator_requires_gate_on_attention_for_deepep_m2n`：config owner 已按 A/F 两个 role 覆盖 gate 必选策略。

### `v1/worker/test_npu_mla_graph.py`

- `test_preimport_real_module_surfaces_installed_runtime_failures`：测试本文件的 import helper 自身，没有执行 AFD。

### `v1/worker/test_npu_model_runner_v2.py`

padding 全拓扑×全 rank×三模式交叉压成八个有独立目的的输入/输出；删除私有 sidecar 快照，保留外部 native ABI 和控制结果。

- `test_camp2p_padding_preserves_graph_descriptor_and_restores_dispatch`：mode FULL/NONE、成功/失败已在 CPU dispatch 协议表保护；额外 native dataclass 字段由 replace 原样携带，并非新 AFD 规则。公开 execute 的 many-to-one 原生协商/graph fallback 八行表保留。
- `test_execute_forwards_dummy_profile_and_context_then_restores`：与 CPU compat/patches/npu/test_mrv2_ubatching.py 执行同一真实 execute_model 函数体，重复 provider/async DP scope；余下内容是 native kwargs 转发快照。保留 CPU 默认/randomize、成功/异常的输出表与本文件实际 native DP dispatch 场景。
- `test_mla_capture_registry_uses_exact_native_descriptor_and_skips_warmup`：只检查 private capture dict；合入 run_fullgraph 的实际记录消费，不维护缓存布局。
- `test_native_dummy_run_reaches_afd_execute_with_context_state`：调用未覆写的 native _dummy_run，主要检查参数转发；AFD execute 的 DP/provider scope 已直接覆盖。

### `v1/worker/test_npu_runtime.py`

- `test_async_cam_dbo_configuration_preserves_thresholds`：同一 runtime_config 函数已在 CPU owner 覆盖，DP 交叉未被函数读取。
- `test_npu_async_feature_validation_rejects_native_draft_models`：对相同 validator 逐个重测 upstream SpeculativeConfig 的 method→uses_draft_model 分类；CPU feature owner 已保护 draft 拒绝。
- `test_npu_attention_async_connector_skips_dp_metadata_control_plane`：同一 mixin 的 no-control-plane 分支已有公共 provider 表。
- `test_npu_attention_runner_builds_and_sets_metadata`：未覆写的 AFDMetadataProviderMixin，同一 public installer 由 Attention owner 文件覆盖。
- `test_npu_attention_runner_builds_dp_fallback`：直接 private fallback 包装；公共 send metadata 表观察实际发出的 fallback tokens。
- `test_npu_attention_runner_constructor_does_not_initialize_connector`：大量 patch 后只检查内部 factory 参数和初始化 flag；实际模型 load 后 connector announce 顺序保留。
- `test_npu_attention_runner_sends_graph_flags`：相同 mixin 和相同 native DPMetadata，无 NPU 差异；共享 owner 的 wire flag 表负责。
- `test_npu_attention_runner_sends_per_ubatch_dp_metadata`：相同 mixin，移交共享 owner 的不等长 stage 载荷检查。
- `test_npu_ffn_connector_uses_model_device_index_for_dp_worker`：六处 patch 后只记录内部 factory rank/local_rank 参数，不是设备绑定或数据传输验证。
- `test_npu_ffn_runner_dp_path_invokes_model_with_hidden_states_and_layer`：仅检查内部 model.calls；有效同步执行的 wire 输出已覆盖，混入 async state 不增加支持场景。
- `test_npu_ffn_runner_filters_dense_layers_when_gate_runs_on_attention`：绕过 runner factory 对 legacy sync FFN 手动设置 Attention gate；当前该配置由 AsyncNPUFFNModelRunner 接管，模型 MoE inventory 另有覆盖。
- `test_npu_ffn_runner_graph_key_uses_ffn_aggregated_token_counts`：私有 wrapper 重复公共 cuda_graph/ffn_metadata 的字面表。
- `test_npu_ffn_runner_requires_compute_hook`：人为使用缺少必备模型接口的 SimpleNamespace，只得到 Python AttributeError，没有 AFD 拒绝契约。
- `test_worker_registers_fla_before_model`：九处 patch 后断言完整内部调用链，没有 native FLA 结果；不把此模拟当 NPU 算子可用性证明。

## 前三项修正的验证与证据范围

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 原无 torch/vLLM 环境的 CI Simple Test 命令 | 678 passed / 100 skipped / 8 deselected，另 27 subtests passed | `tests/unit` 加 CI 中四个 NPU host harness/reference 文件，`not gpu and not vllm_runtime`；receiver 的成功/失败两行实际运行 |
| 独立 CPU Torch 环境的 UT | 754 passed / 70 skipped / 8 deselected，另 10 subtests passed | `tests/unit`，同一 marker 选择；含可在 CPU 运行的量化 payload/数值、路由 IDs、layout/host 编排检查 |
| 定向主机运行时适配 | 4 passed | CPU Torch 2.15.0，临时替代外部运行时导入/设备与 transport；执行当前两行 eager 测试、真实 P2P 分配/接收路径的弱引用测试、现有 capture 地址复用测试；不等同于原 Ascend/CUDA 环境执行 |
| Ruff / format / mypy 3.11 | 通过 | 本次修改的三个 Python 文件；上一轮全量静态检查已通过 |
| Markdownlint / git diff --check | 通过 | 审计报告格式 / 工作树空白检查 |
| Buildkite schema | 上一轮通过 | ready / merge 渲染后的 pipeline，保留迁移测试的执行入口；本次未改 CI |

2026-11-05 为上述修正增加三次仅在进程内的定向错误注入，均修改真实生产 Python 方法，未改写生产文件：

- 移除 P2P capture 条件：新 eager buffer 用例因分配弱引用仍存活而失败。
- 移除 worker 正常 receiver-exit 状态交接：成功退出用例在真实 graph close 收到 `Async graph resources require receiver exit and device completion` 而失败；失败保留分支仍通过。
- 取消 non-layered 有效 token 行裁剪：对应 eager 用例因 `(8, 32)` 与 `(3, 32)` 输出不符而失败；layered 行仍通过。

上一轮另有两次仅在测试进程内的错误注入，未改写生产文件：

- 将 DP1/TP2 graph key 改回错误的 fallback 值：对应表中 2 行失败，另外 2 行通过。说明统一后的 owner 会发现被移除薄包装测试原先保护的 TP 展开问题。
- 将 W8A8 post-quant FP32 scale 改为全零：V2/V4 × 三种 dtype 的 6 行均因输出不正确而失败。说明保留测试检验了 scale 的实际消费，不只检查字段存在或 helper 被调用。

本地缺少 vLLM、vLLM-Ascend、torch_npu 和对应设备；需要这些依赖的用例未执行。CPU fakes/AST 运行和静态检查不等同于真实 CUDA/NPU、分布式通信或 E2E 通过。迁移的 CUDA 路由用例仍需由 L4 CI 执行，本轮没有运行 E2E。
