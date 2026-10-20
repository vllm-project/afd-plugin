---
name: deploy-dsv4-flash-async
description: 在固定 vLLM 0.30 / Ascend 环境中部署 DeepSeek-V4-Flash Async CAM AFD 或原生 EP 对照，验证服务，并按需运行固定负载 RPS/MBT 压测；也提供需独立验收的 P/D 组合入口。
---

# DSV4-Flash Async 部署与压测

根据用户需求准备配置、执行部署或测量性能。以用户要求的交付状态为终点；
配置准备、本地命令验证、硬件请求成功、精度和性能验收分别报告。

只验证本 skill、脚本或所附测试的准确性时，先读 [验证与覆盖](references/validation.md)，
从主机契约检查及固定版本真实接口检查开始；目标平台为 NPU 不表示需要全量运行时验收。
真机 smoke 按用户选定的范围执行，复用匹配环境和产物；不自动扩展到 DBO、FULL、
精度、性能或全部拓扑。发现入口缺陷后修复并复验受影响路径，直到约定的交付状态。

## 选择部署与执行模式

Prefill 实现和 P/D 分离是两个独立维度：

| Prefill 实现 | prefill_only（KV=0） | P/D 分离（KV=1） | 参考 |
| --- | --- | --- | --- |
| AFD / Async CAM | `run_prefill.sh` | D + `run_prefill.sh` + proxy | [AFD](references/afd-prefill.md) |
| 原生 full-model EP | `run_prefill_full.sh` | D + `run_prefill_full.sh` + proxy | [原生 EP](references/baseline.md) |

PD 组合另读 [PD](references/pd.md)，独立验证 proxy 请求与 KV 传输。
`prefill_only` 表示不启用 KV connector、直接向 P 请求一个输出 token 的测量约定，
不是专门的服务器开关。Async CAM、Attention 原生 DBO 和 Decode 的异步调度是不同机制。

AFD 两个角色必须选择 MRV2；Attention 保持 eager，DBO 仅在 Attention 上开启，
使用原生阈值和两 microbatch。旧 `async_moe_split` 不再控制切分。
FFN 可选普通 eager、W4A8 layered eager 或窄范围的 layered FULL；
具体配置、量化和容量边界见 AFD 参考。不要把算子覆盖或同步 CAMP2P 的能力
直接外推到本 skill 的 Async CAM 路径。

## 准备环境与脚本

使用本 skill 的 `scripts/`，将绝对路径设为 `DSV4_SCRIPT_DIR`。
用户指定其他脚本目录时先核对差异。入口清单见 [脚本](references/scripts.md)。
先读 [环境与固定版本](references/environment.md) 和所选模式参考，再检查入口脚本。
脚本选择现有环境，不负责自动升级依赖。

实际部署时，从现有环境查明目标节点/容器、已分配设备、checkpoint 及其量化、运行时和源码路径、
网络接口、拓扑、容量参数、端口，以及本次独立日志/PID 目录。
缺少实际部署所必需的信息时询问；只准备方案时明确待填项。
源码路径默认以 skill 所在 AFD checkout 及其同级 vLLM/Ascend 目录为起点，
按真实安装覆盖。`source common_env.sh` 会修改环境并创建日志目录。

执行远端启动、构建或计算前遵守适用的服务器健康、设备分配和清理规则。
核对 A/F 设备不重叠、DP/TP/rank 数及跨节点分段一致；
确认所用 checkpoint、CLI 和匹配的插件原生算子可用。
W4A8 的实际加载路径、容量和启动经验见 [W4A8](references/w4a8-startup.md)。

## 执行、验证与交付

启动脚本采用 nohup；返回 PID 只说明已提交。跨节点命令均提交后再等待全局 ready，
按本次启动时限观察首个有效错误，不做无信息价值的反复重启。
Attention API 可用以及 FFN capture/receive loop 就绪共同构成 AFD readiness，
不要求 FFN HTTP API 就绪。
短请求结束后再次检查服务存活及错误日志，不能把返回过一个 token 当作完整 smoke 通过。

按 [验证与覆盖](references/validation.md) 选择与请求及变更相称的验证。
单 token smoke、生成正确性、实际 DBO 执行、graph replay 和性能有不同证据，
开关设置或 `/health` 成功不能替代实际路径。沿用历史证据时说明适用版本和范围。

用户要求性能测试时读 [固定负载压测](references/benchmark.md)。使用冻结数据和
独立 `DSV4_BENCH_RUN_ID`，保持部署与测量配置一致；MBT 标签不会修改服务器。
对比 AFD 和原生 EP 时对齐资源预算、checkpoint、负载和服务要求，记录各自合理配置；
隔离 DBO、layered 或 graph 的影响时使用对应的成对实验。

报告实际配置、版本与构建产物、节点/设备、运行路径、命令、日志、结果和证据限制。
推理成功与清理结果分开记录。`stop_local.sh` 只向 PID_DIR 内记录的 PID 发信号，
使用前核对归属，之后检查本次 worker、端口和设备是否释放；不按用户名清理。
FULL 的阻塞 receive 可能需要进程级清理，不能把强制结束写成正常退出。
