# A5 多层 MegaFFN 常驻专家池（实验）

Dhq_AFD 分支：多层 routed experts 共用一个 persistent runner。
原有 normal 模式保留每层独立 runner。本文补充早期单层同步实验文档。

```text
vLLM 加载各层 BF16 w13/w2
           │ post-load hook
           ▼
各层 convert_routed_expert_weights
           │ 按专家维合并四个 weight/scale tensor
           ▼
一个 pooled MegaFFNBackend / 一个 MegaFfnRunner
           │ build_graph(max_tokens) → start_persistent() 一次
           ▼
层 0 本地 IDs ── +0 ────┐
层 1 本地 IDs ── +E0 ───┤ submit_ffn → 对应专家计算 → clone 输出
                        └─ 生命周期结束 stop_persistent() 一次
```

## 初始化与调用

- 模型 post-load hook 在最终权重布局确定后、首次构图前调用 prepare_persistent_expert_pool。
- offset 按实际 routed 层的专家数累加，跳过 dense 层仍正确。
- 本地 ID 先验证在 [0,E_layer) 内，再加 offset，避免错误 ID 选到另一层专家。
- 各层 H、I、top-k、max_tokens 必须一致；weight/scale 的 dtype、device、除专家维以外的布局也必须一致。
- 固定输入容量，短请求补零；输出独立存储，不会被下一请求覆盖。
- F 同步处理请求；未提供并发提交、热重载或多个 daemon。
- 原模型仍持有源 BF16 权重，另常驻转换后的专家池；未实现释放源权重的优化。

## MegaKernel 原生改动

examples/operators 是实际参与运行时构建的代码。
register 编入 libmk_operators.a，再链接 ascendmk_rt；Device kernel 也要重新编译。

- Python capture 传 num_experts；sort schema 的输出和 workspace 按真实 E 定形。
- sort/init-routing host tiler 从张量形状取得 E，传入完整 [0,E) 范围。
- count Device kernel 原 256/257 元素 UB 队列改按 E/E+1 分配。
- 每次 replay 清零专家计数，避免跨层、跨请求残留。
- host 检查排序与计数共用 TPipe 的 UB 预算，过大 M/K/E 在构图时报错。
- 原 C++ 示例同步传 num_experts。未改内部任务调度或设备持有权机制。

## 验证与边界

独立实卡脚本：megakernel/tests/integration/test_mega_ffn_pooled_reference.py。
生成并重载两层 BF16 FFN 测试权重，每层 E=256、H=I=256、K=2。
按层 0→1→0→1、M=128/8/1/64 提交，第二层 ID 达到 511。
对照独立 BF16 矩阵计算，容差 rtol=0.12、atol=0.002，包含内部 FP8 量化误差。
已验证旧输出保留、一次 pk_main、四次 submit、一次正常 stop。

服务验证目录：/mnt/share/d00893600/实验记录/afd-megaffn-pool-20261119/service。
两层 DeepSeek V4 结构、每层 E=8、dummy BF16 权重，A=1、F=2。
连续两轮独立启动均成功：各两次 HTTP 请求、每次生成 4 token；
每轮 F 共一次 pk_main、18 次 submit（含预热）。第二轮已移除临时诊断。
首轮曾在首个真实请求超时，重启后成功，尚未确认该次超时根因。

这与独立 512 专家测试是两项验证；不代表真实 DeepSeek-V4-Flash-0731
量化权重、模型精度或性能验证。上述两层实验当时未覆盖完整 PD、多 F、异步通信算子。

## 三层 PD 回归（2026-11-23）

三层 DeepSeek V4 结构、每层 E=8、dummy BF16 权重；专家池共 24 个专家，
层偏移为 0/8/16。P、D 使用独立 Attention 通道连接同一个 F，A/F 通信仍为
A2E/E2A，客户端只请求 PD Proxy，由 Proxy 完成 P→D 的 KV 参数传递。

- 两个非流式请求、一个流式请求均返回 HTTP 200，每次生成 8 token。
- D 外部 KV 查询增量 208、命中增量 205；每次请求在 F 上先执行 P，随后执行 D。
- 常驻 kernel 启动一次、提交 87 次、停止一次；三层各调用 29 次（含启动 profile）。
- P/D 退出码均为 0，未出现堆损坏日志；F daemon 正常停止。
  测试清理时 F 外层 API 因 KeyboardInterrupt 返回 1，Proxy 因 SIGTERM 返回 -15。
- 本次环境使用独立 Mooncake 符号隔离副本，规避与 SDK 的 C++ ABI 符号冲突。
  该环境补丁不属于本仓库代码，也不是上游 Mooncake 修复。

验证的是小规模 dummy 模型的通信与生命周期，不代表真实权重精度、性能、
多 F 或异步 dispatch/combine 已通过。后续提交前检查仅重跑 CPU 单测和 Host tiling 测试。
