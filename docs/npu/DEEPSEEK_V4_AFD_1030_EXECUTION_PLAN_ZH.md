# DeepSeek-V4 AFD 1030 后续执行计划

## 1. 目标

在 `2026-10-30` 前，以 `afd-plugin-phase1-delivery` 为唯一交付分支，完成 A5 上DeepSeek-V4 的以下组合验证：

```text
Prefill --Mooncake KV--> Decode Attention --P2pHccl AFD--> Decode FFN
                                      |
                                      +-- DSpark draft
                                      +-- target FULL_DECODE_ONLY Graph
                                      +-- U2 microbatch
```

执行顺序固定为先同机、再跨机。功能和最终精度是交付门禁；性能只形成测量报告，不设置收益比例门槛。

本文只描述从当前状态到 1030 交付的具体工作，不重复历史演进、旧性能实验和已关闭问题。

现场命令和证据归档统一使用[A5 合并验证指导书](DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md)。
本文规定范围、顺序和完成定义，指导书规定操作步骤；两者使用同一矩阵和门禁。
`【本地已完成】` 标记本地代码/离线验证，`【A5 手工执行】` 标记需现场完成的操作。
具体分工、当前执行入口及回传物见[手工交接清单](DEEPSEEK_V4_AFD_A5_MANUAL_HANDOFF_ZH.md)。
2026-09-16 的旧 A4F4/all-on 指导仅保留历史参考，其分支、上游提交、双机拓扑和 draft Graph 不再用于本期正式验收；本期五个 Graph/U2 多流开关默认开启，支持显式关闭。

## 2. 当前起点

截至 `2026-10-08`，已经确认的基线如下：

| 项目 | 当前结论 |
|---|---|
| afd-plugin | `feat/dsv4-afd-phase1-delivery`，提交 `02d0779a2b11d29e3635f3dc9ec3ce5a606bf583` |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`（0.23 系列） |
| vLLM-Ascend | `11ee45653b199a097805b87011824a81ffa51b95` |
| 已验证拓扑 | 单机 A5、A4F2、TP1、MTP off、standalone Decode-AF |
| 已验证模式 | eager/U1；`FULL_DECODE_ONLY` Graph/U2 |
| Graph/U2 证据 | Graph capture 完成；live U2 两个 stage 实际执行；请求返回 HTTP 200；无 HCCL/NPU fatal |
| 已关闭问题 | MXFP AllToAllV 兼容；eager U2 overlap 开关；mixed Graph DP padding |
| 当前执行策略（2026-10-09 更新） | 五个 Graph/U2 多流开关默认开启（1），支持显式关闭（0）；实际组合仍按功能矩阵验收。功能诊断可开启，性能测试必须关闭诊断同步 |

这组结果只证明 standalone Decode-AF 可用。以下内容尚未在本交付分支上完成 A5 验收：

- Mooncake PD 和 Proxy 请求路由；
- DSpark 与 Decode Attention 的组合；
- PD + AFD + DSpark + Graph/U2 完整组合；
- A/F 跨机通信和三角色异构部署；
- 最终 GSM8K 精度与性能记录。

当前 worktree 中的 Prefill、Proxy、PD 和 DSpark recipe 改动属于在研代码，不计入上述已验证基线，必须通过本计划的阶段门禁后才能更新状态。

2026-10-09 更新：`common.sh` 的五个 Graph/U2 开关默认开启，并保留显式 0/1 覆盖。`【本地已完成】` A5 Prefill/Attention 的 Mooncake 配置已接入本机绝对路径 `ascend_local_comm_res_path`，并检查角色设备对应的 endpoint JSON；此字段仅用于 A5，A3 不注入；旧 `tools/dsv4/run_phase1_a5_*` 和 `mooncake_pd_manual` 入口不在交付分支中。`【A5 手工执行】` 仍需环境审计、HIXL 资源内容核对、安装路径确认和 standalone 回归。本文的后续步骤是待验证计划，不是通过报告。

2026-10-10 现场进展：已回传 standalone EOS 与 C1 执行成功日志；C2 四个 drafter 已加载，直连 Attention 已有 proposal/acceptance，但 Proxy 请求在 Mooncake KV 接收时报 block 长度不匹配，随后 hybrid cache 失败处理触发单组解包错误。`【本地已完成】` afd-plugin scheduler 兼容补丁在异步 PD load 时延后 DSpark lookahead，保留正常 decode 的预测分配，并修正 `failure_policy=fail` 的多组请求失败处理。冻结源码 CPU 分配重放和回归通过，尚未证明 A5 C2 F0。`【A5 手工执行】` 下一步按统一指导书第 3.4 节更新/冻结插件提交、冷启动 C2、确认四个 EngineCore 补丁标记，再经 Proxy 执行 1 请求 EOS→5 请求 smoke→92 请求 full；C2 F0 未通过不进入 C3。各阶段完整门禁仍按本计划核对，三机、精度和性能继续待执行。

## 3. 固定范围

### 3.1 验收拓扑

| 部署 | 拓扑 | 用途 |
|---|---|---|
| 同机 | `P2 + A4 + F2`，共 8 张 A5，TP1 | 先关闭配置、资源竞争和组合功能问题 |
| 跨机 | Prefill、Decode Attention、Decode FFN 分别部署；主验收为 `P8 + A4 + F2`，TP1 | 验证 PD 和 A/F 同时跨机，以及不同角色卡数的异构部署 |

跨机环境不足时，不得用同机结果替代跨机验收，也不得将 A3 结果写成 A5 结论。

### 3.2 功能矩阵

同机和跨机均按以下顺序执行。上一项未通过，不进入下一项。

| ID | 组合 | Target 模式 | U | DSpark |
|---|---|---|---:|---:|
| C1 | PD + AFD | eager | 1 | off |
| C2 | PD + AFD + DSpark | eager | 1 | on |
| C3 | PD + AFD + Graph | `FULL_DECODE_ONLY` | 2 | off |
| C4 | PD + AFD + DSpark + U2 | eager | 2 | on |
| C5 | PD + AFD + DSpark + Graph + U2 | `FULL_DECODE_ONLY` | 2 | on |

C5 是最终交付组合。Graph 只要求 target Decode 路径；DSpark draft 允许保持 eager。

### 3.3 本期不做

- 不修改 vLLM-Ascend；兼容逻辑、配置和 recipe 只在 afd-plugin 内实现。
- 不扩展 U3、TP2、PP/SP/CP/DCP、Window connector 或异步 HCCL。
- 不引入额外 MTP 组合；DSpark 使用其模型配置确定的 speculative token 数。
- 五个 Graph/U2 多流开关保持默认开启，支持显式关闭用于定位或回退；每轮冻结完整开关配置，默认与关闭配置的结果分别记录，不互相替代。
- 不做全部矩阵的逐 token exact；最终精度仅对 C5 使用冻结的 GSM8K 口径。

## 4. 执行阶段

### M1：冻结环境和交付脚本（10 月 8 日至 10 月 10 日）

本地负责配置写入、静态/单元检查及交付分支 Git 提交；离线包作为备用；以下环境审计、资源确认、GSM8K 口径冻结和实机回归标记为 `【A5 手工执行】`。

工作项：

1. 记录镜像 digest、CANN/HCCL、torch、torch-npu、vLLM、vLLM-Ascend、afd-plugin 精确版本及 Python 安装路径。
2. 将当前在研 recipe 拆成可审查提交：配置生成、Prefill、Attention、FFN、Proxy、验证器各自职责明确，保留 standalone 启动兼容性。
3. 固定同机和跨机的设备、IP、网卡、端口及启动顺序；启动前检查端口和本机设备列表。
4. **仅对 A5 PD 配置**增加 HIXL 本地资源检查：
   - `ASCEND_LOCAL_COMM_RES_PATH` 必须指向各自本机实际绝对路径，示例 `/etc/hixlep`；
   - Prefill producer 和 Decode Attention consumer 必须把该值写入
     `kv_connector_extra_config.ascend_local_comm_res_path`；
   - 只设置环境变量而未写入 `kv_connector_extra_config` 视为未配置；
   - A3 不注入 `ascend_local_comm_res_path`，也不要求上述环境变量或目录；
   - FFN 不创建 Mooncake KV connector。
5. 在 10 月 10 日前确认一台同机验证 A5、三台跨机验证 A5、target/draft 权重、各节点 HIXL 资源文件和离线 GSM8K 数据均已到位，并冻结 GSM8K 题目数、prompt、解码参数和最低通过分数。
6. 在新 recipe 上复跑一次 standalone A4F2 Graph/U2，确保没有破坏 `02d0779` 基线。
7. Graph/U2 五个多流开关默认开启（1），recipe 支持显式关闭（0）；逐 rank 核对最终生效值与本轮配置一致。
   C4 的 eager/U2 专用开关 `AFD_HCCL_EAGER_U2_STREAM_OVERLAP=1`
   与 Graph/U2 五个开关分开记录，不将二者混为同一策略。
8. 冻结新 afd-plugin 提交、角色配置和验证入口。`02d0779` 是回归起点，
   后续正式结果记录实际新提交；上游两个提交保持不变。正式运行使用干净源码树。

完成标志：配置单测和 shell 静态检查通过；standalone 冷启动、batch 1/8/32、真实 U2、正常停机和 NPU 清理通过；输出一份固定环境清单。

### M2：【A5 手工执行】同机 PD + AFD 最小闭环（10 月 11 日至 10 月 14 日）

先执行 C1，不启用 DSpark、Graph 或 U2。只解决以下链路：

```text
Client -> Proxy -> Prefill -> Mooncake KV -> Decode Attention -> AFD FFN
```

必须证明：

- Prefill 和 Decode Attention 建立真实 Mooncake session，KV 传输完成；
- 首 token 能从 Prefill 正确衔接到 Decode；
- KV connector 只挂在 Prefill/Decode Attention，FFN 只运行 AFD connector loop；
- batch 1/8/32、流式输出、取消后恢复和二次冷启动通过；
- Attention 先退出并向 FFN 交接 shutdown，所有角色无残留 NPU 进程。

完成标志：C1 同机 F0 通过并归档完整启动参数、角色日志、请求结果和 cleanup 结果。

### M3：【A5 手工执行】同机完整组合（10 月 15 日至 10 月 19 日）

依次执行 C2、C3、C4、C5，按以下对照关系隔离变量；不把 C2→C3
当作单变量增量：

1. C2 对照 C1，增加 DSpark；证明 proposal/verify 和 PD + AFD eager/U1 共存，draft 只在 Attention 侧。
2. C3 对照 C1，切换到默认五开关全开的 Graph/U2 路径；证明 PD KV 完成后的在线 replay，并验证未捕获 shape 回退 eager。
3. C4 对照 C2，增加真实 U2；确认两个 stage 都执行 draft/target 所需协议。
4. C5 对照 C4，切换 target 为 Graph，五个 Graph 多流开关默认开启；draft 保持 eager，完成同机最终组合。

每项都必须先通过单请求和 batch 1，再扩到 batch 8/32。发生 hang 时记录最后一个 Attention/FFN/PD 边界，需要关闭开关定位时另建 run ID，保存配置和结果，不同时改多个独立变量或用回退结果覆盖原失败。

完成标志：同机 C1-C5 全部 F0 通过；C5 完成两轮独立冷启动后的完整功能回归和正常停机。

### M4：【A5 手工执行】跨机异构部署（10 月 20 日至 10 月 25 日）

先固定三台 A5 的角色、设备和地址，再复跑 C1-C5。跨机前置检查包括：

- 三台机器使用相同代码提交、镜像、模型和解码参数；
- `PREFILL_HOST_IP`、`ATTENTION_HOST_IP`、`FFN_HOST_IP`、网卡和路由可达；
- Mooncake KV 端口、HCCL base port、AFD rendezvous port 和 API port 无冲突；
- AFD rendezvous 使用 FFN 主机地址，不使用 `127.0.0.1`；
- 每台机器分别校验本地 HIXL 资源文件和可见设备。

C1 用于关闭跨机网络、Mooncake 和 AFD 基础问题；C2-C5 只在 C1 完整通过后执行。
所有角色日志必须带统一 run ID 和机器/角色标识，便于按同一请求重建时间线。

完成标志：跨机 C1-C5 全部 F0 通过；C5 完成两轮独立冷启动，日志能同时证明远程 KV、远程 A/F、DSpark、Graph replay 和 U2 两阶段实际生效。

### M5：最终精度、性能记录和交付（10 月 26 日至 10 月 30 日）

GSM8K 和性能采集为 `【A5 手工执行】`；报告检查、源码整理和最终打包在收到实机证据后继续完成。

1. 在 M1 冻结 GSM8K 数据集版本、题目数、prompt 模板、解码参数、target/draft 权重和最低通过分数。
2. 只对同机 C5 和跨机 C5 执行最终 GSM8K；按固定分数判定，不做逐 token 对比。
3. 对两个 C5 记录吞吐、TTFT、TPOT、峰值 HBM 和 DSpark acceptance rate；关闭
   `AFD_HCCL_STAGE_DIAGNOSTICS`、FFN compute-sync diagnostics 和 DEBUG 日志后测量。
4. 整理启动 recipe、环境模板、验证入口、证据索引、支持边界、已知问题和回退步骤。
5. 在干净分支上重跑静态检查、单元测试和关键 smoke，创建最终提交和交付 tag。

完成标志：同机和跨机 C5 的功能与 GSM8K 均通过；性能数据可复现且不作为阻断；交付物可以从空闲环境按文档完成一次冷启动、验证和正常停止。

## 5. 统一验收门禁

### 5.1 功能请求

每个 C1-C5 至少覆盖：

- 并发 1、8、32；
- 短输入、长输入和 KV block 边界输入；
- streaming、EOS、`max_tokens` 截断；
- 请求取消以及取消后的恢复请求；
- C2/C3/C4 至少一轮全新冷启动和正常停止；同机与跨机 C1、C5
  均至少两轮独立冷启动，使用新 run ID、新目录和完整请求集，轮间清空角色进程及端口。

F0 定义：上述请求覆盖、适用的真实路径证据和运行质量门禁全部通过。
请求脚本返回 0 只关闭请求子项；人工证据表中有 `pending`、`blocked`
或缺少日志定位时，不得将该点写成 F0 通过。

### 5.2 真实路径证据

| 特性 | 必须出现的证据 |
|---|---|
| PD | Mooncake producer/consumer session、KV transfer 完成、首 token 衔接成功 |
| AFD | Attention/FFN rank mapping 正确，A2F/F2A 均有在线传输记录 |
| DSpark | draft proposal 和 target verify 实际执行，记录 proposed/accepted 数量 |
| Graph | capture 完成、live replay 命中，并有一个未捕获 shape 的 eager fallback |
| U2 | 所有 Attention rank 的在线请求都观测到 stage 0 和 stage 1 |

仅 HTTP 200、仅进程存活或仅完成 Graph capture 都不能单独判定通过。

### 5.3 运行质量

- 无 Python traceback、EngineCore fatal、OOM、请求 timeout、HCCL watchdog 或 NPU `507xxx` 首错；停机级联错误也必须在报告中标明先后关系。
- Attention 停止后所有 FFN rank 收到 shutdown，进程退出，业务端口释放，NPU 无残留。
- 每次结果记录三个源码提交、dirty state、启动环境和请求时间范围。
- 功能诊断运行与性能运行分开；诊断同步数据不得用于性能结论。

## 6. 问题处理顺序

1. C1 失败时只定位环境、HIXL、Mooncake、Proxy 和基础 AFD，不进入 DSpark/Graph。
2. C2 失败时先在 eager/U1 隔离 DSpark，不使用 U2 或 Graph 掩盖问题。
3. C3/C5 hang 时按 PD、Attention、FFN 最后进度定位；可显式关闭五个 Graph/U2 开关建立回退对照，并记录所有实际值及依赖关系。回退结果不能自动替代默认全开组合验收。
4. 跨机失败先用相同提交回归同机 C1/C5，再判断是代码回归还是网络/资源差异。
5. 发现平台算子或 HCCL 问题时保存最小复现和精确软件版本；本期不直接修改 vLLM-Ascend 绕过。
6. 性能回退不阻塞 1030 功能交付，但 OOM、timeout、错误输出或无法正常清理必须阻塞。

## 7. 交付物

- afd-plugin 固定提交和 tag；
- 同机与跨机环境模板、角色启动/停止 recipe；
- C1-C5 功能验证入口和 C5 GSM8K 入口；
- 每次运行的 `runtime.json`、请求结果、角色日志、NPU/端口清理记录和 SHA256；
- 一份最终验收表：同机/跨机各 5 个功能点，加 2 个 C5 精度点；
- 性能记录：吞吐、TTFT、TPOT、峰值 HBM、DSpark acceptance rate；
- 已知限制和回退说明。

## 8. 1030 完成定义

只有同时满足以下条件，才可宣布本期完成：

1. 同机 C1-C5 全部通过；
2. 跨机 C1-C5 全部通过；
3. 同机和跨机 C5 均通过冻结的 GSM8K 门禁；
4. C5 有真实 PD、AFD、DSpark、Graph replay 和 U2 两阶段证据；
5. 冷启动、请求、取消恢复、正常停机和资源清理闭环；
6. 固定版本、脚本、报告和证据可由另一环境按文档复现。

若跨机资源、target/draft 权重或冻结数据集未按期就绪，应明确标记为外部阻塞，不得用同机结果、A3 历史证据或仅 HTTP 成功降级替代。
