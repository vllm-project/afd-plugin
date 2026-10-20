# 验证与覆盖

此 skill 的命令和配置已迁移到固定 0.30 源码契约，硬件资格按具体运行记录认定。
AFD 原始模型、DBO 和 FFN FULL 的已有证据入口为
[PR #462](https://github.com/vllm-project/afd-plugin/pull/462)、
[原生 DBO](../../../../docs/npu/TESTING.md#native-dbo-contracts-and-device-execution) 和
[FFN FULL](../../../../docs/npu/ASYNC_FFN_RUNTIME.md#validation)。
PR 中复用的旧运行、维护者确认和新运行各有范围，不能改写为本次 skill 的新硬件结果。

## 按任务选择覆盖

以下是按任务选取的验证层次，不是每次都要执行的运行时矩阵。

| 范围 | 要回答的问题与证据 |
| --- | --- |
| skill、脚本和测试准确性 | 主机 shell 集成测试；固定版本真实 CLI、插件注册依赖与 benchmark JSON 契约。替身通过不能代替真实接口或硬件证据 |
| 最小真机 smoke | 实际加载目标 checkpoint 与 AFD MRV2；A API、F receive loop 就绪；短生成检查响应结构、非空输出、finish reason 和 token 上限，并确认请求后服务仍存活。P/D 另须确认真实 KV 传输 |
| 生成功能与正确性 | 按目标工作负载覆盖串行、并发生成，使用预定 oracle 检查答案、stop、长度；短 smoke 不构成精度验收 |
| 原生 DBO | 保留短请求；另用原生阈值以上的 DSV4 DSA 工作，包含不齐的 prefill 长度和跨 child 边界请求；每个 A rank 完成两个非空阶段 |
| layered 开关 | 同一 W4A8 checkpoint 下普通 eager 与 layered eager 成对比较，检查实际路径、输出精度、显存和完成行为 |
| FFN FULL | 与对应 layered eager 比较生成 token/输出、捕获与请求期间 replay；A 可分别关闭/开启 DBO，不把 DBO-off fixture 外推到 DBO-on |
| 拓扑和长 prefill | 先选实际使用的单机/跨机拓扑，覆盖 MBT 分块和最长请求；按需要扩展，不机械执行所有组合 |
| prefill 性能 | 使用冻结 workload、实际 MBT 和完整错误结果；隔离功能观察器、IO 日志等额外同步开销 |
| P/D | 经 proxy 完成多 token 生成，并确认真实 P/D 请求与 KV 传输；健康接口不足以证明链路 |
| Decode draft 选择 | 按 checkpoint 区分普通 MTP 与 DSpark；核对实际加载类、speculative 配置和生成，不能把另一种 draft 的结果沿用为通过 |
| 停止 | 核对本次 worker、端口和设备释放，分别记录正常退出、强制终止和残留 |

原始 DSV4 十个短请求可能不触发 DBO；保留它们的断言，不靠降低阈值把它们改成另一个用例。
开关 on/off、纯 CPU contract 或 synthetic operator 测试都不替代 checkpoint 的实际生成。
精度范围和容差采用项目标准或用户明确接受的标准，不能为通过而放宽。

## 主机检查与真机复验

从仓库根目录运行所附 shell 集成测试，无需 torch、vLLM、模型或 NPU：

```bash
python3 -m unittest discover -s tests/e2e -p test_deploy_dsv4_flash_async_skill.py -v
```

这些测试运行真实 shell 入口，但 CLI 与 benchmark 结果为替身。核对固定版本真实接口时，
将入口生成的完整参数交给该版本 parser/validator，并核对最终子进程的插件白名单；
只运行 `--help` 不证明完整配置可被接受。benchmark 替身须遵循真实 client 的结果写入规则，
不限并发的 `max_concurrency` 是 JSON `null`，显式上限是整数，详见 [压测契约](benchmark.md)。

主机检查可发现入口问题，应优先于分配设备或加载模型。需要真机 smoke 时，复用已确认匹配的
原生算子、checkpoint 和环境，使用独立日志/PID 目录及明确启动/请求时限。失败时先定位首个
有效错误，修复后只复验受影响链路；不以主机测试通过代替用户要求的成功短请求，
也不为一个入口问题展开环境重建或运行时能力矩阵。

请求返回后再次检查 health、worker 和首个错误。native MRV2 可能在生成结束后的 KV-only
维护步骤创建不含 `BatchDescriptor` 的上下文；该步骤不执行模型，也不应发布 A/F 事务。
曾观察到先返回一个 token、随后因误注入 AFD 元数据而退出，单次 HTTP 200 不能掩盖此失败。

## 2026-10-20 最小 smoke

本轮验证 skill 和测试的准确性，并复验 AFD 短请求及跨机 P/D。固定源码为
vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、
vLLM-Ascend `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`；
CANN 9.1.0、torch 2.10.0+cpu、torch-npu 2.10.0.post4、Python 3.12.13。
两台 910C 节点各 16 DIE，加载相同 per-channel W4A8 DSpark checkpoint。

- P 为 legacy A DP2TP4 / F EP8，A/F eager、layered 关闭；D 为原生 DP16TP1 / EP16，
  `FULL_DECODE_ONLY`、async scheduling、DSpark 5-token draft。
- 容量限制为 max model len 65536、P MBT 8192、A/F/D max seqs 2、
  `BATCH_SIZE_FACTOR=0.0625`、HCCL buffer 256 MB；不是默认容量组合的资格证明。
- AFD 直接请求生成一个 token `World`，请求后 P/D health 均为 200。
- 实际 `curl_test.sh` 经 proxy 返回 32 个 reasoning token、finish reason 为 `length`；
  consumer 日志中 `num_external_tokens=86`，同一请求有 KV transfer start/finish，
  rejection sampler 有 `drafted=5` 记录；请求后 P/D health 均为 200。
- 主机 shell 集成测试 16 项通过；NPU V2 runner 的 CPU 单元模块 149 项通过。
  新增 KV-only 回归在修复前复现真实 descriptor 错误，修复后通过。

修正了 benchmark null 契约、P connector 插件白名单、Decode draft 选择和 NPU V2 的
KV-only 元数据安装。原始失败记录仍保留；此短 smoke 不评价答案精度、吞吐、原生 DBO、
FFN FULL、其他拓扑或长请求。

## 扩展运行时验收

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
原生 D 的 MTP、DSpark 和 graph 是另一条路径，须单独核查，不能沿用 AFD Attention 的限制判断。
同步 CAMP2P 保留在仓库原有 E2E；部分 DSV4 同步案例只检查 liveness，不是精度资格。
运行时 profiling 和独立算子性能工具只在目标版本实际具备时使用，先检查相关 PR 是否已合入。
