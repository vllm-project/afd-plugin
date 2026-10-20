# 验证与覆盖

此 skill 的命令和配置已迁移到固定 0.30 源码契约，硬件资格按具体运行记录认定。
AFD 原始模型、DBO 和 FFN FULL 的已有证据入口为
[PR #462](https://github.com/vllm-project/afd-plugin/pull/462)、
[原生 DBO](../../../../docs/npu/TESTING.md#native-dbo-contracts-and-device-execution) 和
[FFN FULL](../../../../docs/npu/ASYNC_FFN_RUNTIME.md#validation)。
PR 中复用的旧运行、维护者确认和新运行各有范围，不能改写为本次 skill 的新硬件结果。

## 按任务选择覆盖

| 范围 | 要回答的问题与证据 |
| --- | --- |
| 启动和普通请求 | 实际加载目标 checkpoint 与 MRV2；A API、F receive loop 就绪；串行及并发生成能完成并按预定 oracle 检查答案、stop、长度 |
| 原生 DBO | 保留短请求；另用原生阈值以上的 DSV4 DSA 工作，包含不齐的 prefill 长度和跨 child 边界请求；每个 A rank 完成两个非空阶段 |
| layered 开关 | 同一 W4A8 checkpoint 下普通 eager 与 layered eager 成对比较，检查实际路径、输出精度、显存和完成行为 |
| FFN FULL | 与对应 layered eager 比较生成 token/输出、捕获与请求期间 replay；A 可分别关闭/开启 DBO，不把 DBO-off fixture 外推到 DBO-on |
| 拓扑和长 prefill | 先选实际使用的单机/跨机拓扑，覆盖 MBT 分块和最长请求；按需要扩展，不机械执行所有组合 |
| prefill 性能 | 使用冻结 workload、实际 MBT 和完整错误结果；隔离功能观察器、IO 日志等额外同步开销 |
| P/D | 经 proxy 完成多 token 生成，并确认真实 P/D 请求与 KV 传输；健康接口不足以证明链路 |
| 停止 | 核对本次 worker、端口和设备释放，分别记录正常退出、强制终止和残留 |

原始 DSV4 十个短请求可能不触发 DBO；保留它们的断言，不靠降低阈值把它们改成另一个用例。
开关 on/off、纯 CPU contract 或 synthetic operator 测试都不替代 checkpoint 的实际生成。
精度范围和容差采用项目标准或用户明确接受的标准，不能为通过而放宽。

当前 FULL 的公开真实模型 fixture 为 per-channel W4A8、4A/4F、A TP4/F TP1、
receive capacity 128、DBO off；它不是内置 EP8、大 MBT 或跨机部署的资格证明。
其观察器为功能证据增加同步，不用于吞吐测量。调用已有 fixture 前先读其参数与资源布局：

```bash
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph --help
```

## 对照实验的含义

AFD 与原生 EP 可以在同资源预算、checkpoint、负载和服务要求下使用各自合理配置，
测量该部署组合的收益。并行拓扑不必强行相同；记录活跃与预留设备、并行度、
MBT、并发、缓存、预热及显存配置，限制结论的适用范围。

隔离单项机制时保持其余条件可比：DBO off/on、普通 eager/layered eager、
layered eager/FULL 分别成对。当前 layered W13 未应用 `swiglu_limit`，普通 eager 可能应用，
因此普通 eager/FULL 的差异不能全部归因于 graph。
单 token prefill benchmark 可以衡量该 workload 的 prefill 收益，不能独立证明模型精度、
PD 或完整业务收益。原始失败和负结果同样保留。

MRV1 与旧 request/token splitter 已退役；Attention graph、PP/PCP/DCP、MTP、
W8A8 FFN FULL、真实 per-group W4A8 checkpoint、A5 Async CAM 不作为本 skill 的已支持组合。
原生 D 的 MTP 和 graph 是另一条路径，须单独核查，不能沿用 AFD Attention 的限制判断。
同步 CAMP2P 保留在仓库原有 E2E；部分 DSV4 同步案例只检查 liveness，不是精度资格。
运行时 profiling 和独立算子性能工具只在目标版本实际具备时使用，先检查相关 PR 是否已合入。
