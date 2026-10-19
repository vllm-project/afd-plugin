# DeepSeek-V4 AFD A5 1030 合并验证指导书

更新日期：2026-10-16（性能主流程收敛为单台 8 卡 A5，保留两机跨机补充验证；不更新已有实机验证结论）。本文合并旧 A5 dSpark/PD 指导与[1030 执行计划](DEEPSEEK_V4_AFD_1030_EXECUTION_PLAN_ZH.md)，是本期现场操作入口。
执行计划规定交付范围和日期，本文规定现场命令、验证顺序、证据与失败处理。
本文是待执行指导，不代表组合已经通过 A5 验收。

**执行标记**：`【本地已完成】` 为代码/配置的离线检查；`【A5 手工执行】` 为需要你在 A5 实际环境完成的操作。首次执行从[手工交接清单](DEEPSEEK_V4_AFD_A5_MANUAL_HANDOFF_ZH.md)的 A5-01 开始。**2026-10-19 最新跨机核对**：10-16 日 cross-C5 full 92 项通过、90 个非流式响应全部非空且无输出 token ID `0`、四 DP 与 Proxy 请求归零。实际为两机 P2+A4+F2，P/F 在 `.18`、A 在 `.14`；缺 Attention 侧日志/metrics，site.env 与实际地址不符，P/F 末尾有 Killed。当前按[cross-C5 本轮结果第 5 节](DEEPSEEK_V4_AFD_CROSS_C5_20261016_RESULT_ZH.md)补材料，完整 F0、第二轮和三机 P8+A4+F2 尚不能据此关闭。此前各日期的单机状态和对照步骤保留作历史，不要求为确认本轮请求结果立即重跑 full。

## 0. AFD 性能验证：先读本节，再按步骤操作

**目标：在 Graph、Graph+DSpark 两类常用配置下，比较原生 Decode 与 AFD P2P 的输出吞吐、每卡效率及延迟。** 后文保留完整命令和故障记录；首次阅读只需按本节顺序跳转。

### 0.1 主要只对比两组

| 场景 | 原生基线 | AFD 方案 | 正式配对 |
|---|---|---|---|
| Graph，不开 DSpark | 完整 Decode + Graph | A4F2 + Graph/U2 | N-G 对 B2（C3） |
| Graph，开启 DSpark | 完整 Decode + Graph + DSpark | A4F2 + Graph/U2 + DSpark | N-GS 对 B3（C5） |

每对使用同一 checkpoint、输入文件、输出预算和解码参数；DSpark 的 draft 都使用 eager。原生微批能力需实测，若原生与 AFD 的 U 配置不同，结果说明中保留差异。eager 只用来排查和辅助解释，不展开整套性能矩阵。

**主要性能对比先在同一台 8 卡 A5 上依次执行，四套配置不同时运行。** 使用下面的固定卡位，完整单机配置见[第 4.1 节](#41-同机设备配置)。

| 方案 | Prefill | Decode / Attention | 独立 FFN | 实际使用 NPU |
|---|---|---|---|---:|
| 原生 Graph，DSpark 关/开 | P2：0、1 | D4：2–5 | 无，6、7 空闲 | 6 |
| AFD Graph/U2，DSpark 关/开 | P2：0、1 | A4：2–5 | F2：6、7 | 8 |

Graph 和 DSpark 在各角色分配的卡上执行，不额外增加角色卡数；模型、KV、Graph 和 draft 是否能装入显存仍须逐点实测。这是 6 卡与 8 卡两种部署方案的对比，需同时报总吞吐和每卡吞吐。单机可验证同机 HCCL P2P 的收益；跨机网络开销和扩展性另按[第 4.2.1 节](#421-两机补充拓扑同时覆盖远程-kv-和远程-af)补测，正式三机验收保持原要求。

### 0.2 操作顺序

1. **先关闭已有功能问题。** 沿用现有 4K 配置和 C1→C5 顺序，核实输出、Graph 在线执行、DSpark 指标和正常停服。文档目前记录的 C3 输出/默认全开、C4 指标/停服及 C5 待验证项仍未关闭；详见[第 6.3 节](#63-保存-metrics-before生成精确-token-id-输入)。这些问题未关闭时，不开始正式收益统计。
2. **冻结单机环境与输入。** 按[第 3 节](#3-a5-手工执行第-1-步准备现场配置和审计环境)填写真实 IP、网卡、模型路径和版本，按第 4.1 节放置角色，P/A/F/Proxy 的角色 IP 均设为本机可达 IP。准备同一份真实文本/代码输入及请求预算，保存文件 hash。DSpark 主测使用真实 Prefill/KV。
3. **先准备四份启动配置。** 原生 Graph、AFD Graph、原生 Graph+DSpark、AFD Graph+DSpark，各自使用新 RUN_ID。AFD 复用 C3/C5；原生使用[第 8.3.2 节](#832-准备非-afd-原生-decode-基线-b0n-gn-gs)的候选命令，不传 AFD role，也不启动独立 FFN。原生 Graph/DSpark 候选参数仍须现场验收。
4. **四组先各过一次 16K、BS8 的预检。** 先设置下面的长度/容量，再按[第 6.1–6.2 节](#61-启动-prefill然后立即相邻启动-ffn-与-attention)启动和检查 ready：AFD 为 P ready→F→紧接着 A→Proxy；原生为 P ready→D ready→Proxy。先发 smoke，再发少量真实 16K 输入；检查实际长度、输出、所有 rank、在线 replay/U2、DSpark 增量、HBM 和正常停服。smoke 通过不等于长输入或性能通过。
5. **升到 BS24，运行第一轮正式比较。** 重新冷启动，分别设置原生 D/A 为 24、F 序列容量起点为 48；核验新的 Graph shape、DSpark token budget 和 HBM。单独 warmup 96 请求，等待归零；正式每轮 576 请求，重复 3 轮，客户端在途容量至少覆盖 96，并核对实际活跃数。发压/采集步骤见[第 8.4.3 节](#843-aisbench-负载与执行步骤)。
6. **最后扩展到 32K、64K。** 依次先 BS8/16 再尝试 24；四组每个合法点仍测 3 轮。OOM 或只能排队时记录实际可支持的 BS，不写成达到了 BS24。按需扫描 BS8/16/24/32，比较各自最佳合法吞吐；8K 和输出 512–1536 的分布放在主流程之后补充。
7. **产出两张收益表。** Graph、Graph+DSpark 各一张，每行记录输入长度、D/A 的 BS、F 容量、完成/失败数、输出 tokens/s、每卡 tokens/s、TPOT P95、TTFT P95、HBM；DSpark 表再加接受率。保留 3 轮原始值并报中位数，不只报最快一轮。

第 4 步在所有角色终端加载现场配置后覆盖以下参数；示例只针对**首个 16K、BS8 预检点**，不覆盖整个长度矩阵：

```bash
export ISL=16384 OSL=1024
export MAX_MODEL_LEN=$((ISL + OSL))  # 17408；按实际模板 token 数复核
export DECODE_BS=8
export FFN_SEQ_CAPACITY=16          # A4/F2 的序列容量起点
export MAX_NUM_SEQS="$DECODE_BS"     # Prefill、原生 D、Attention 的初始设置
export EXECUTION_MODE=full-decode-only U_BATCHES=2
export ENABLE_PD=1 ENABLE_MTP=0
# 不开 DSpark 的配对：所有角色使用同一 Flash checkpoint。
export MODEL_PATH="$FLASH_MODEL_PATH" ENABLE_DSPARK=0
# 开 DSpark 的配对：改为以下两项；Prefill/FFN 仍使用该 DSpark checkpoint。
# export MODEL_PATH="$DSPARK_MODEL_PATH" ENABLE_DSPARK=1
```

AFD 的 FFN 按第 6.1 节原启动命令保存日志/PID，但在 `nohup` 前添加角色专用环境：

```bash
MAX_NUM_SEQS="$FFN_SEQ_CAPACITY" ENABLE_PD=0 ENABLE_DSPARK=0 \
  nohup bash "$RECIPE/afd_ffn.sh" > "$RUN_ROOT/ffn.log" 2>&1 &
echo $! > "$RUN_ROOT/ffn.pid"
```

这条命令替代该轮原 FFN 启动命令，不重复启动。Attention 继续使用 `MAX_NUM_SEQS=8`。DSpark 开启后，16/48 仅是 FFN 的序列容量起点，verify/draft 的 token、buffer 和 capture 预算另行核验；不能只改 BS 就开始计时。现有默认 capture sizes `1 2 4 8` 也不证明 BS24 已覆盖。

正式计时前，按[第 8.2 节](#82-关闭诊断后重新冷启动性能-c5)关闭诊断并新冷启动；通过 Proxy 发同一份真实输入。第 8.3 节 random CLI 和第 8.4 节 synthetic 配置只作辅助压力测试，不能替代 DSpark 主测数据入口。AISBench 的真实数据配置、输出预算传递及版本必须先核实，未就绪时该项保持 pending。

**读结果只记住两个数：总输出 tokens/s，以及总输出 tokens/s÷实际用卡数。** 当前真实 PD 布局，原生除以 6、AFD 除以 8，均包含 P2；两组均预留同一台 8 卡整机，另报按整机 8 卡计算的效率，不混淆实际用卡与预留成本。只有经验证的纯 Decode 测量才分别除以 D4 的 4 卡和 A4F2 的 6 卡。64K 输入加 1024 输出至少需 `MAX_MODEL_LEN=66560`，还需实测 KV、HBM 和 Graph，不保证 BS24 能容纳。

**下一步的第一项操作：先补齐现有功能问题的证据和修复复测；关闭后，交付四份 16K/BS8 的实际启动参数、请求结果及正常停服记录，再进入 BS24 正式比较。**

## 1. 合并后的范围与门禁

| 项目 | 本期统一口径 |
|---|---|
| afd-plugin | 唯一交付分支 `feat/dsv4-afd-phase1-delivery`；`02d0779a2b11d29e3635f3dc9ec3ce5a606bf583` 是回归起点，正式运行冻结修正后的新提交 |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，0.23 系列 |
| vLLM-Ascend | `11ee45653b199a097805b87011824a81ffa51b95`；不修改上游源码 |
| 环境 | A5 现场匹配的镜像、CANN/HCCL、torch/torch-npu；冻结精确版本、build 和镜像 digest |
| 同机 | 一台 8 卡 A5：Prefill DP2/NPU 0-1、Attention DP4/NPU 2-5、FFN DP2/NPU 6-7；TP1 |
| 跨机 | 三台 A5：Prefill DP8/NPU 0-7、Attention DP4/NPU 0-3、FFN DP2/NPU 0-1；TP1 |
| 两机补充验证 | 两台 8 卡 A5 可按第 4.2.1 节部署 P2F2+A4，同时覆盖远程 KV 和远程 A/F；基于现有脚本约束设计，尚未实机验证，不替代三机验收 |
| Graph | target Decode `FULL_DECODE_ONLY/U2`，五个 Graph 多流开关默认开启（1），支持显式关闭（0）；async scheduling off |
| DSpark | 只在 Attention 上运行 draft；draft eager，speculative token 数来自配套模型配置 |
| 最终精度 | 同机和跨机 C5 各做一次冻结口径的 GSM8K；不逐 token exact |
| 性能 | 两个 C5 记录吞吐、TTFT、TPOT、峰值 HBM、acceptance rate；不设置收益比例门槛 |

第 8.2 节用于记录 C5 性能。第 8.4 节规定主要性能矩阵：**AFD P2P+Graph/U2、AFD P2P+Graph/U2+DSpark**，分别与功能匹配的原生 Decode 对照，重点验证 BS24 和 8K/16K/32K/64K 输入。eager/U1 仅作辅助归因。第 8.3 节提供真实 PD 的执行入口，承接这两个主要组合的端到端吞吐及延迟验证；纯 Decode 容量实验单独记录入口、KV 来源及资源分母。C1–C5 全部启用 AFD，彼此比较不能给出 AFD 开关本身的收益。

旧指导的权重可信性、精确版本审计、并发 1/8/32、取消恢复、资源清理和失败采集继续保留。
旧 `3da28f9`、A4F4/all-on 和两机 P8/A4F4 证据作为历史参考，不能代替本期验收。
旧 no-AFD dSpark 验证可用于故障隔离，不再强制复跑旧 S1–S3 整套矩阵。
旧指导“单机不能完整验证 PD”针对旧拓扑；P2+A4+F2 是本期待验证的同机拓扑。

同机、跨机分别按以下顺序执行；前一个点的 F0 未通过，不进入下一个点。

| ID | 组合 | `EXECUTION_MODE` | `U_BATCHES` | Attention `ENABLE_DSPARK` | 独立冷启动轮数 |
|---|---|---|---:|---:|---:|
| C1 | PD + AFD | `eager` | 1 | 0 | 2 |
| C2 | PD + AFD + DSpark | `eager` | 1 | 1 | 至少 1 |
| C3 | PD + AFD + Graph/U2 | `full-decode-only` | 2 | 0 | 至少 1 |
| C4 | PD + AFD + DSpark + eager/U2 | `eager` | 2 | 1 | 至少 1 |
| C5 | PD + AFD + DSpark + Graph/U2 | `full-decode-only` | 2 | 1 | 2 |

C2 对照 C1，C3 对照 C1，C4 对照 C2，C5 对照 C4。C2→C3 不是单变量切换。
每轮使用新的 run ID、进程、日志目录；轮间完成停服、端口释放和 NPU 清理。
F0 = 请求覆盖 + 适用的真实路径证据 + 运行质量 + 停服清理全部通过。

## 2. 第 0 步：关闭 M1 前置问题，冻结可执行版本

2026-10-09 执行更新：`【本地已完成】` 五个 Graph/U2 开关默认开启并支持显式关闭；A5 PD HIXL 的 JSON 写入及本机 endpoint 文件检查已实现并经过离线测试。`【A5 手工执行】` 环境审计、资源内容有效性与实机 smoke 仍须完成。

| 项目与执行标记 | 完成条件 |
|---|---|
| 【本地已完成；A5 手工复核】Graph/U2 五个多流开关 | 默认值为 1，支持通过环境变量显式设为 0；启动后逐 rank 核对实际值与本轮冻结配置一致。`common.sh` 已保留显式覆盖值 |
| 【本地已完成；A5 手工复核】HIXL JSON 写入与文件检查 | `build_a5_pd_kv_config` 已将本机绝对路径写入 Prefill/Attention 配置，检查对应物理 NPU 的 `ub_endpoint_npu_<ID>.json` 可读且为有效 JSON；现场还须确认资源内容与设备匹配。A3 不套用，FFN 不加载 KV connector |
| 【本地准备；A5 手工验收】recipe 交付 | 本地修改提交至 `feat/dsv4-afd-phase1-delivery`，现场按交接清单优先从 Git 更新；离线包作为备用。安装路径、冷启动与完整组合仍须验证，正式运行冻结发布的完整提交 |
| 【本地已完成】验证入口 | 使用本分支 `recipe/npu/P2pHcclAFDConnector/deepseek_v4/` 和本文请求脚本，不调用旧 `tools/dsv4/` 命令 |
| 【A5 手工执行】冻结 GSM8K 现场口径 | 固定离线数据/task 配置、题目数、fewshot/prompt、解码参数、评分键和最低分数；现场 lm-eval CLI 与版本一并记录 |

C4 的 eager/U2 专用开关 `AFD_HCCL_EAGER_U2_STREAM_OVERLAP=1` 是现有 recipe 的独立策略。
Graph/U2 时该 eager 专用开关为 0，另外五个 Graph 多流开关默认均为 1，可显式设为 0。不得将 eager/U2 开关与 Graph 多流开关混为一个配置；C4 仍须在本期实际验收。

### 2.1 五个 Graph/U2 开关：默认开启，可显式关闭

| 环境变量 | 默认值 | 关闭值 |
|---|---:|---:|
| `AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP` | 1 | 0 |
| `AFD_HCCL_GRAPH_U2_HYBRID_DAG` | 1 | 0 |
| `AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM` | 1 | 0 |
| `AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM` | 1 | 0 |
| `AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER` | 1 | 0 |

默认验证 C3/C5 使用五项全开。关闭对照或回退时，在启动 Attention/FFN 的终端中先显式设置为 0，recipe 会保留这些设置：

```bash
# 可选关闭配置；默认验证不执行此段。
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
```

只接受 0/1。若单独关闭 `FFN_RECV_STREAM`，也必须关闭 `FFN_CROSS_LAYER`；后者为 1 时要求前者为 1。Attention/FFN 按同一轮配置启动，保存实际开关值。
关闭配置使用新 run ID，完成同样的功能和资源清理门禁；不能复用默认配置的日志。

### 2.2 `ascend_local_comm_res_path`：仅 A5 PD 使用

A5 的 Prefill producer 和 Decode Attention consumer 分别在 Mooncake 的`kv_connector_extra_config` 中增加以下字段，值来自**各自本机实际绝对路径**。
`/etc/hixlep` 是示例，资源文件必须适配本机；相对路径或另一节点的目录不合格。

```json
{
  "kv_connector_extra_config": {
    "ascend_local_comm_res_path": "/etc/hixlep"
  }
}
```

这是字段位置示例，须与现有 `prefill`、`decode` 配置合并，不能替换完整 `--kv-transfer-config`。仅设置 `ASCEND_LOCAL_COMM_RES_PATH` 环境变量不算写入。
**A3 不注入此字段，也不要求该变量或 `/etc/hixlep` 目录；FFN 不创建 Mooncake KV connector；standalone Decode-AF 不使用此 PD 配置。** 不修改跨平台共享模板为一律注入该字段。`【本地已完成】` A5 recipe 使用专用配置生成函数，字段已经接入实际 `--kv-transfer-config`；缺失本机 endpoint 文件时启动前报错。`【A5 手工执行】` 检查资源内容与设备对应关系，并保留脚本打印的实际 KV JSON。

### 2.3 静态检查与版本冻结

交付前执行以下静态检查；这只证明语法和入口存在，不能替代配置单测或 A5 smoke：

```bash
cd "$AFD_PLUGIN_ROOT"
for script in common.sh afd_prefill.sh afd_attention.sh afd_ffn.sh afd_proxy.sh; do
  bash -n "recipe/npu/P2pHcclAFDConnector/deepseek_v4/$script"
done
"$PYTHON_BIN" tools/validation/a5_1030_requests.py --help
git status --short
```

`【本地已完成】` 五开关默认/覆盖检查、HIXL JSON 写入、endpoint 文件预检及 mock launcher 参数测试。未在本地部署或重启模型服务；`【A5 手工执行】` 按交接清单完成实机门禁。
上述代码项关闭后，填入真实 `AFD_EXPECTED_COMMIT`，再执行下一节。

## 3. 【A5 手工执行】第 1 步：准备现场配置和审计环境

### 3.1 每个节点准备两个配置文件

所有节点共用同一份 `site.env`，示例地址和模型路径必须替换。不要直接运行示例 IP。
本文现场日志检索统一使用 `grep`，无需在容器安装 ripgrep。
三个角色在 DSpark 点使用相同配套 checkpoint，只有 Attention 开启 speculative config。固定 vLLM 0.23 栈通过兼容入口 `method=mtp` 加载 DSpark；不能因日志显示 `mtp` 就判成普通 MTP，须结合 checkpoint 的 `dspark_block_size`、四个 `AscendDSparkProposer`/drafter 和本轮 proposed/accepted 增量确认。

```bash
# site.env：所有节点一致，填写后冻结 SHA256。
export PREFILL_HOST_IP=192.0.2.10
export ATTENTION_HOST_IP=192.0.2.11
export FFN_HOST_IP=192.0.2.12
export PROXY_HOST_IP=192.0.2.10
export AFD_HOST="$FFN_HOST_IP"
export AFD_PORT=29761
export PREFILL_API_PORT=8100
export ATTENTION_API_PORT=8910
export PROXY_PORT=9000
export PREFILL_KV_PORT=30000
export DECODE_KV_PORT=31000
export PREFILL_ENGINE_ID=dsv4-prefill
export DECODE_ENGINE_ID=dsv4-decode
export PREFILL_HCCL_IF_BASE_PORT=50000
export ATTENTION_HCCL_IF_BASE_PORT=51000
export FFN_HCCL_IF_BASE_PORT=52000
export ATTENTION_RANKS=4
export FFN_RANKS=2
export TENSOR_PARALLEL_SIZE=1
export PREFILL_TP_SIZE=1
export FLASH_MODEL_PATH=/home/models/DeepSeek-V4-Flash
export DSPARK_MODEL_PATH=/home/models/DeepSeek-V4-Flash-DSpark
export AFD_EXPECTED_COMMIT='填写M1冻结后的完整提交'
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=32
export GPU_MEMORY_UTILIZATION=0.91
export ATTENTION_MAX_NUM_BATCHED_TOKENS=4096
export FFN_MAX_NUM_BATCHED_TOKENS=8192
export PREFILL_MAX_NUM_BATCHED_TOKENS=4096
export MAX_CUDAGRAPH_CAPTURE_SIZE=8
export CUDAGRAPH_CAPTURE_SIZES='1 2 4 8'
export DBO_DECODE_TOKEN_THRESHOLD=2
export DBO_PREFILL_TOKEN_THRESHOLD=12
export ENABLE_MTP=0
```

`MAX_NUM_SEQS=32` 是本次请求覆盖配置，必须在 M1 冻结并实测 HBM；现有 recipe 默认 8。若现场只能使用 8，须明确区分“32 个客户端并发”和“32 个活跃序列”，并记录队列行为；不能用排队成功代替未捕获 shape 的 Graph fallback 证据。A5 DSpark checkpoint 在 `GPU_MEMORY_UTILIZATION=0.90` 时实测 Prefill 仅余 1.63 GiB KV cache，低于 4096 所需的 1.70 GiB，因此本矩阵冻结为 0.91；C1–C5 使用同一值，不以降低 `MAX_MODEL_LEN` 绕过。**此 export 必须写入本轮实际 `SITE_FILE` 并在每个启动终端加载，不能只修改指导书或请求终端；未设置时 recipe 仍默认 0.90。** 2026-10-13 C4 前次附件在 0.90 下失败，最新 0.91 重试成功、两个 Worker 均有 2.54 GiB KV；修正位置与证据见第 6.1.1 节。

`node.env` 是本机配置，各节点可以不同，分别保存 SHA256；不再要求本机路径/IP 不同的所有 env 文件字节完全相同。

```bash
# node.env：每个节点填本机真实路径、网卡及 IP。
export AFD_PLUGIN_ROOT=/root/dsv4-afd-hccl/src/afd-plugin-phase1-delivery
export VLLM_ROOT=/root/dsv4-afd-hccl/src/vllm-release-v0.23.0
export VLLM_ASCEND_ROOT=/root/dsv4-afd-hccl/src/vllm-ascend-rfc-vllm-cann
export VENV_ROOT=/root/dsv4-afd-hccl/venv
export CANN_ROOT=/usr/local/Ascend/cann-9.2.0
export NIC_NAME=eth0
export HCCL_IF_IP=192.0.2.10
# 仅 A5 Prefill/Attention 使用；填写本机实际绝对路径，A3 不设置。
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep
export RUN_BASE=/data/validation/dsv4-afd-1030
export IMAGE_DIGEST='填写实际镜像digest'
export CANN_BUILD='填写冻结的精确版本和build'
export HCCL_PACKAGE='填写精确包名版本build及来源'
```

`RUN_BASE` 写在 `node.env`，表示本机保存验证记录的基础目录。`RUN_ID`、
`RUN_ROOT` 在每轮运行的终端中设置：PD 预检按交接清单 A5-02，standalone
按本指导书第 5 节，C1–C5 按第 4.3 节。下文已将目录初始化放在首次写文件之前。

CANN 路径以冻结的 A5 镜像为准；上面的 9.2.0 仅是现场路径示例，不宣称新组合已在该版本验证。不要回退到开发机 `/mnt/workspace/code/.ascend/cann-9.0.0`，不要混用多个 CANN 安装。**HIXL 配置仅用于 A5**：Prefill producer 和 Decode Attention consumer 分别将本机绝对路径写入 `kv_connector_extra_config.ascend_local_comm_res_path`，路径必须存在且资源文件适配本节点。仅 export 环境变量不足；P/A 路径可不同，不能填另一节点的目录。A3 不注入此字段，也不要求 `/etc/hixlep` 或 `ASCEND_LOCAL_COMM_RES_PATH`；FFN 不创建 Mooncake connector。

每个角色终端加载环境，`SITE_FILE`、`NODE_FILE` 替换为本机保存的文件：

```bash
set -euo pipefail
export SITE_FILE=/data/validation/config/site.env
export NODE_FILE=/data/validation/config/node.env
source "$SITE_FILE"
source "$NODE_FILE"
test -f "$CANN_ROOT/set_env.sh"
source "$CANN_ROOT/set_env.sh"
source "$VENV_ROOT/bin/activate"
export PYTHON_BIN="$VENV_ROOT/bin/python"
export VLLM_BIN="$VENV_ROOT/bin/vllm"
export RECIPE="$AFD_PLUGIN_ROOT/recipe/npu/P2pHcclAFDConnector/deepseek_v4"
export API_PORT="$ATTENTION_API_PORT"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
# 内部 HTTP 地址直连；这是 HTTP 环境代理的豁免，不会关闭 PD Proxy 服务。
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,$HCCL_IF_IP,$PREFILL_HOST_IP,$ATTENTION_HOST_IP,$FFN_HOST_IP,$PROXY_HOST_IP"
export no_proxy="$NO_PROXY"
test -d "/sys/class/net/$NIC_NAME"
ip -o -4 addr show dev "$NIC_NAME" scope global
```

如果本机 IP 的 `/metrics` 返回 504，但 `127.0.0.1` 成功，先在服务本机执行
`curl --noproxy '*' -v --max-time 10 "http://$HCCL_IF_IP:$ATTENTION_API_PORT/metrics"`。
绕过代理后成功才可确认 HTTP 环境代理是原因；仍失败则检查 `ss -ltnp`、容器网络、路由与防火墙。
`a5_1030_requests.py` 使用 `trust_env=False` 直连所填 URL，避免意外走 HTTP 环境代理。
更新后的 `tools/proxy_server.py` 对 Prefill/Decode 转发和动态节点探测也使用 `trust_env=False`。旧版 Proxy 仍继承 HTTP 环境代理，必须在启动它的终端设置 `NO_PROXY/no_proxy`；只在请求客户端执行 export 或 `curl --noproxy '*'` 不会改变已运行 Proxy 的环境。
本机 standalone 客户端可使用 loopback；跨机 HTTP 必须使用对端可达地址。
不要因此把 HCCL/AFD 的 `HCCL_IF_IP` 或跨机 rendezvous 地址改成 loopback。

### 3.2 冻结版本、权重和安装位置

每次运行前执行，任意 mismatch 或 dirty 停止。正式源码冻结后不要 reset、屏蔽检查或临时修改 vLLM/vLLM-Ascend。环境安装检查指向实际源码，不能仅凭 `git HEAD` 判定Python 正在使用哪个 checkout。

```bash
test "$(git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD)" = "$AFD_EXPECTED_COMMIT"
test "$(git -C "$VLLM_ROOT" rev-parse HEAD)" = 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
test "$(git -C "$VLLM_ASCEND_ROOT" rev-parse HEAD)" = 11ee45653b199a097805b87011824a81ffa51b95
for repo in "$AFD_PLUGIN_ROOT" "$VLLM_ROOT" "$VLLM_ASCEND_ROOT"; do
  test -z "$(git -C "$repo" status --porcelain)"
done
"$PYTHON_BIN" - <<'PY'
import importlib
from importlib.metadata import version
for module, package in [('torch', 'torch'), ('torch_npu', 'torch-npu'),
                        ('vllm', 'vllm'), ('vllm_ascend', 'vllm-ascend'),
                        ('afd_plugin', 'vllm-afd-plugin')]:
    item = importlib.import_module(module)
    print(package, version(package), item.__file__)
PY
"$VLLM_BIN" serve --help=all > /tmp/dsv4-1030-serve-help.txt
grep -nE 'kv-transfer|speculative|enable-dbo|no-async|cudagraph' /tmp/dsv4-1030-serve-help.txt
npu-smi info
```

进入新 run 前，目标 NPU 不得有遗留业务进程，空闲 HBM 应回到各卡一致的现场基线；任一卡明显偏高时先清理并重新采集 `npu-before.txt`。例如其他卡约 4.9 GiB 而单卡仍为 56 GiB，不能作为独立冷启动继续执行。

记录镜像 digest、驱动、CANN 和独立 HCCL 包名/版本/build/来源；保存镜像或安装包提供的版本清单及查询原始输出。只写“最新 HCCL”不通过。所有节点须使用同一冻结软件组合及模型 manifest；CANN 不混装，后续性能 trace 的解析器版本与采集版本匹配。

DSpark 不由普通 Flash 权重修改 config 产生。配套权重的 config 若已被现场修改，创建独立 view，配置来自可信原始交付，其余文件软链接；源目录与 view 必须不同：

```bash
export WEIGHT_ROOT=/models/DeepSeek-V4-Flash-DSpark
export CONFIG_JSON=/models/original-config/DSpark.config.json
test -f "$CONFIG_JSON"
test "$WEIGHT_ROOT" != "$DSPARK_MODEL_PATH"
test ! -e "$DSPARK_MODEL_PATH"
mkdir -p "$DSPARK_MODEL_PATH"
find "$WEIGHT_ROOT" -mindepth 1 -maxdepth 1 ! -name config.json \
  -exec ln -s '{}' "$DSPARK_MODEL_PATH/" \;
cp -a "$CONFIG_JSON" "$DSPARK_MODEL_PATH/config.json"
sha256sum "$DSPARK_MODEL_PATH/config.json" "$DSPARK_MODEL_PATH/model.safetensors.index.json"
"$PYTHON_BIN" - "$DSPARK_MODEL_PATH/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
n = c['dspark_block_size']
assert type(n) is int and n > 0
assert c['dspark_target_layer_ids']
print({'dspark_block_size': n, 'dspark_target_layer_ids': c['dspark_target_layer_ids']})
PY
```

已有可信、未改动的 DSpark 目录时，直接指定该目录并记录 hash，跳过 view 创建。
recipe 还须通过 native A5 FP8 checkpoint 检查。冻结 target/draft 权重来源、配置、index 和交付权重 manifest 的 SHA256；不能用配置 hash 代替权重一致性证据。

### 3.3 【A5 手工执行】C2/C4/C5 的 DSpark 算子依赖检查

在 Attention 节点使用启动服务的同一个 `PYTHON_BIN`，加载同一 CANN/venv 后、启动任何本轮服务前执行。固定 vLLM-Ascend `11ee45653` 的 DSpark 模型构造函数会导入外部 `ascend_ops`，并从 `_ascend_dsv4`、`_ascend_v4` 或 `custom` 命名空间查找两个 fused attention 算子。仅能导入 vLLM-Ascend 或通过 C1 不足以证明这项依赖就绪。

```bash
"$PYTHON_BIN" - <<'PY'
import importlib
import sys
import torch
import torch_npu
from importlib.metadata import packages_distributions, version

print('python:', sys.executable, flush=True)
module = importlib.import_module('ascend_ops')
print('ascend_ops:', module.__file__)
for package in packages_distributions().get('ascend_ops', []):
    print('distribution:', package, version(package))
for name in ('npu_sparse_attn_sharedkv', 'npu_sparse_attn_sharedkv_metadata'):
    for namespace in ('_ascend_dsv4', '_ascend_v4', 'custom'):
        qualified = f'{namespace}::{name}'
        if torch._C._dispatch_has_kernel(qualified):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, 'PrivateUse1'):
                raise RuntimeError(f'DSpark selected operator lacks an NPU kernel: {qualified}')
            print('NPU kernel:', qualified)
            break
    else:
        raise RuntimeError(f'DSpark NPU kernel is missing: {name}')
print('DSpark operator import/registration check passed')
PY
```

若报 `No module named 'ascend_ops'`，先检查模块是否只装在另一 Python 环境，或现场是否已有配套离线 wheel。`AFD_BUILD_ASCEND_OPS=0` 只关闭 afd-plugin 的 CAMP2P A2E/E2A 扩展构建，与 DSpark 的 `ascend_ops` 是不同依赖；改成 1 不会补齐本错误。固定版本在 A5 上禁用通用自定义算子路径，不能仅以 `_C_ascend` 中存在同名绑定就替换当前 DSpark 依赖。

```bash
# 查找现场已有的 Python 包、扩展或离线 wheel；不要求安装 grep 之外的检索工具。
find /home /opt /data /usr/local -type f \
  \( -path '*/ascend_ops/__init__.py' -o -name 'ascend_ops*.so' \
     -o -name 'ascend_ops*.whl' -o -name 'ascend-ops*.whl' \) \
  -print 2>/dev/null || true
```

**仅在配套包来源和兼容性已确认后安装**：wheel 必须匹配现场 CPU 架构、Python 3.12、torch/torch_npu 2.10、冻结 CANN build 和 Ascend950，且包含以上两个 NPU kernel；不从名称相近的公开包猜测安装。若 wheel 已在容器，执行：

```bash
export DSPARK_OPS_WHEEL=/data/packages/填写已确认的配套wheel文件名.whl
test -f "$DSPARK_OPS_WHEEL"
sha256sum "$DSPARK_OPS_WHEEL"
"$PYTHON_BIN" -m pip install --no-index --no-deps "$DSPARK_OPS_WHEEL"
# 随后重新执行本节的 Python import/registration 检查。
```

缺少已确认的包时先回传查找输出，C2/C4/C5 保持 blocked。导入和注册通过后记录包来源/版本/hash，使用新的 run ID 冷启动 C2；这项预检仍需由实际 draft 加载及在线 proposed/accepted 增量完成实机验证。

### 3.4 【A5 手工执行】C2 的 PD KV 布局修复及复测

**2026-10-13 最新状态**：附件 `c34df56188ed459cb5bb5467ac8f3aa9.gz` 已确认 EOS、smoke 和 full 92 项全部通过，并发 32 短/长各 32 项均返回 64 tokens/`length`；取消恢复后四 DP running/waiting 与 Proxy 请求数归零。Prefill/Attention 的 2/4 个 tracking 标记和 6→5 组真实传输存在，未再出现 `finish req not in reqs to process`。下面 10-12 日的失败描述是修复历史，当前无需继续按“32 并发超时”排障。

**剩余门禁是停服**：09:22:22 UTC 停 Attention 后出现四次 Segfault；FFN shutdown payload 标记为 0，随后 FFN 停止报错，09:26:00 UTC 强杀两个 EngineCore。最终设备/服务端口空闲，但不能认定正常停服通过。当前 launcher 的 TERM trap 会向整个服务进程组发送 TERM，须核对这一行为与 Attention 先通知 FFN、再释放 KV/控制面的退出顺序；Segfault 的底层原因尚未由 Python 日志确定。先完成停服修复和 A5 复测，不进入 C3，也不要求为确认本次请求结果立即重发 92 项。证据、手工步骤和 C3 配置见[本轮核对结果](DEEPSEEK_V4_AFD_C2_20261013_RESULT_ZH.md)。

**适用错误**：经 Proxy 的 EOS 报 `JSONDecodeError: Expecting value`，Attention 的 Mooncake 接收仍报 `operands could not be broadcast together with shapes (0,) (3,)`。

2026-10-12 附件 `cb15c79fe8614ade8b3fc654a8ec62d7.gz` 中，四个 EngineCore 已启用 10-10 scheduler 补丁；请求失败后仍继续运行，证明失败隔离已生效。KV 传输仍失败：同一请求 Prefill 发出 **6 组** block ID，Attention 接收端只有 **5 组**。DSpark 增加缓存层后，上游会重新分组和排列物理张量；组号与注册地址数组下标不能跨两端直接配对。上次只修复异步 lookahead 和 hybrid fail，未解决此布局问题。Proxy 又把后端 500 吞成空 HTTP 200，才导致请求脚本解析空 JSON；更改 EOS prompt 或只修解析异常不能修复 KV 传输。

`【本地已完成】` 新修复 `afd_plugin/compat/patches/npu/dspark_pd_layout.py` 在 Prefill/Attention Worker 导入完成后自动注册。Prefill 握手发送每个 target 缓存层的层名、组号、实际地址、块步长与组件布局；Attention 按层名映射两端组号，分别使用两端步长，传输前检查布局和块数量。DSpark 独有缓存仅保留在 Attention 本地，不从 Prefill 拉取。两端可以继续分别为 6/5 组；**不要求组数相等，也不按较短列表强行截断错配组**。原 `dspark_pd.py` 的异步分配及 hybrid fail 修复保留。非 DSpark 接收端保持原传输路径。

Proxy 非流式后端失败现在返回带错误内容的 HTTP 502；流式失败发送 SSE error。请求脚本在解析前保存 `<请求名>.response.http.json` 和 `<请求名>.response.body`，包括 HTTP 状态、头和原始响应体。失败仍判为不通过。

本地真实 CPU 张量的 6→5 组数据拷贝、不同物理排列/步长、A5 indexer 重叠视图、非法布局拒绝、msgpack/pickle 握手及错误响应回归已通过；**NPU 实际传输、模型输出和 C2 F0 仍待 A5 复测**。保持 vLLM/vLLM-Ascend 提交、CANN、DSpark、五个默认开关及 A5 本机 HIXL 绝对路径。

**2026-10-12 full 并发复测更新**：附件 `78b041cf6b3b4f5cb30c3dddef3819de.gz` 确认 EOS 自然结束、真实 6→5 组 KV 拷贝成功，EOS 后四 DP 请求归零；full 的 1/8 并发短/长输入均返回成功。在 32 并发短输入阶段，25 个请求返回 64 tokens/`finish_reason=length`，7 个请求等候 300 秒后 `ReadTimeout`。这 7 个的 KV 拷贝已成功，随后报 `finish req not in reqs to process`，一直处于 `WAITING_FOR_REMOTE_KVS`：固定 Mooncake `start_load_kv` 先将全批请求入队，最后才登记 tracker；快速传输完成在登记前到达，通知被丢弃。延长超时或降低并发不能修复此竞态。

`【本地已完成】` 同一 `dspark_pd_layout.py` 现在先登记全批请求，再发布任何 KV 接收任务；已完成请求不再在函数末尾重复登记。范围仍限于启用布局协议的 DSpark PD Worker，普通 Worker 保持固定上游路径。1/8/32 请求的确定性“完成早于入队返回”测试通过。请求脚本发生传输/读取超时时保存 `<请求名>.transport.error.json`（请求名、异常类型、请求 ID、时间及耗时），summary 不再只有空 error。

**10-12 日旧附件仅确认 EOS 和部分负载通过，当时 C2 F0/full 未通过**。当时 32 并发长输入、边界、SSE、取消恢复尚未执行；停服阶段另有 Attention `Segfault`/强杀和 FFN 清理报错，`shutdown-failed.txt` 已记录，不能把设备最终空闲等同于正常清理通过。下列命令保留为该修复的操作入口；10-13 日最新请求结果已通过，当前按本节顶部处理剩余停服问题。

**1. 【A5 手工执行】保留当前失败日志，按第 7 节正常停止 Prefill、Attention、FFN、Proxy 并确认设备/端口释放。** 在已经按第 3.1 节加载现场环境的终端更新代码；下面是本轮实际源码路径：

```bash
export AFD_PLUGIN_ROOT=/home/z00569729/dsv4-afd-a128f32-r3/src/afd-plugin-phase1-delivery
test -z "$(git -C "$AFD_PLUGIN_ROOT" status --porcelain)"
git -C "$AFD_PLUGIN_ROOT" remote get-url origin
git -C "$AFD_PLUGIN_ROOT" fetch origin feat/dsv4-afd-phase1-delivery
git -C "$AFD_PLUGIN_ROOT" switch feat/dsv4-afd-phase1-delivery
git -C "$AFD_PLUGIN_ROOT" merge --ff-only origin/feat/dsv4-afd-phase1-delivery
export AFD_EXPECTED_COMMIT="$(git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD)"
printf 'AFD_EXPECTED_COMMIT=%s\n' "$AFD_EXPECTED_COMMIT"
```

核对提交与本次发布回执一致。`SITE_FILE` 是第 3.1 节实际 `source` 的配置文件，不是旧 `RUN_ROOT` 中的归档副本；**在这个文件中替换 `export AFD_EXPECTED_COMMIT=...`**，避免下次加载环境又恢复旧值：

```bash
: "${SITE_FILE:?请先设置为第 3.1 节实际 source 的 site.env 绝对路径}"
test -f "$SITE_FILE"
grep -n '^export AFD_EXPECTED_COMMIT=' "$SITE_FILE"
sed -i "s/^export AFD_EXPECTED_COMMIT=.*/export AFD_EXPECTED_COMMIT='$AFD_EXPECTED_COMMIT'/" "$SITE_FILE"
grep -n '^export AFD_EXPECTED_COMMIT=' "$SITE_FILE"
sha256sum "$SITE_FILE"
```

**2. 【A5 手工执行】确认同一个服务 Python 加载新源码及匹配的 scheduler/Mooncake API。** 已有 editable 安装指向上述目录时，Git 更新后重启即可；导入路径不同则先按交接清单第 2.2 节安装本次插件。

```bash
"$PYTHON_BIN" - "$AFD_PLUGIN_ROOT" <<'PY'
import sys
from pathlib import Path
import afd_plugin
from afd_plugin.compat.patches.npu.dspark_pd import apply_afd_dspark_pd_scheduler_patch
from afd_plugin.compat.patches.npu.dspark_pd_layout import apply_afd_dspark_pd_layout_patch
source = Path(afd_plugin.__file__).resolve()
assert source.is_relative_to(Path(sys.argv[1]).resolve()), source
assert apply_afd_dspark_pd_scheduler_patch(), 'vLLM version/scheduler API mismatch'
assert apply_afd_dspark_pd_layout_patch(), 'vLLM version/Mooncake API mismatch'
print('DSpark PD scheduler + layer layout patches registered:', source)
PY
```

**3. 【A5 手工执行】四个角色冷启动新的同机 C2。** **Prefill 也必须加载新插件并重新启动**，否则会报 `Prefill lacks layer layout metadata`；只重启 Attention 或只重跑请求脚本不生效。 按第 3.1/3.2、4.1/4.3/4.4、6.1/6.2 节依次执行。第 4.3 节的两个目录配置示例都改为 `CASE=C2`，使用新的实际 `CYCLE/RUN_ID`；本机只创建目录一次，其他角色使用相同值。设备 P0–1/A2–5/F6–7，Attention 为 `eager/U1/DSpark on`。在各角色开关配置前设置：

```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

第 4.3 节现在保留这个值，不会再次覆盖为 INFO。Attention 启动后检查四个不同 EngineCore DP rank 均打印补丁启用标记，并确认四个 drafter 加载、FFN 两个 rank 进入 loop：

```bash
grep -nF 'AFD DSpark PD scheduler compatibility active:' "$RUN_ROOT/attention.log"
test "$(grep -cF 'AFD DSpark PD scheduler compatibility active:' "$RUN_ROOT/attention.log" || true)" -ge 4
grep -nE 'AscendDSparkProposer|drafter|AFD FFN EngineCore started' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
```

还须确认 Prefill 的两个 Worker 和 Attention 的四个 Worker 打印布局握手标记：

```bash
grep -nF 'AFD DSpark PD layer layout active:' "$RUN_ROOT/prefill.log" "$RUN_ROOT/attention.log"
grep -nF 'AFD DSpark PD request tracking active: register before enqueue' \
  "$RUN_ROOT/prefill.log" "$RUN_ROOT/attention.log"
test "$(grep -cF 'AFD DSpark PD layer layout active:' "$RUN_ROOT/prefill.log" || true)" -ge 2
test "$(grep -cF 'AFD DSpark PD layer layout active:' "$RUN_ROOT/attention.log" || true)" -ge 4
```

新修复版还须在 P 两个、A 四个 Worker 各看到 `AFD DSpark PD request tracking active: register before enqueue`，否则尚未启用并发登记修复。布局标记含 `role`、`groups`、`target_layers` 和逐层组号；两端 target 层名必须对应，组号可以不同。仅看到 API health 200 或直连 Attention 成功不足以证明 C2；缺少补丁标记时先核对源码路径、版本及插件注册日志。

**4. 【A5 手工执行】先经 Proxy 发送 1 个 EOS 请求。** 在同机服务终端执行，不加 `--standalone`；`BASE_URL` 的 9000 是 Proxy，`ATTENTION_URL` 的 8910 仅用于管理接口和指标：

```bash
export BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-before.txt"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite eos \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --output "$RUN_ROOT/requests-eos-$(date +%Y%m%dT%H%M%S)" \
  --run-id "$RUN_ID-eos"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-after-eos.txt"
```

请求后检查本轮传输完成证据：

```bash
grep -nF 'AFD DSpark PD layer KV transfer:' "$RUN_ROOT/attention.log"
grep -nE 'broadcast together|hybrid KV load failed|Prefill lacks layer layout|insufficient remote blocks|incompatible cache layout|finish req not in reqs to process|EngineDeadError' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/proxy.log" || true
```

成功日志包含请求 ID、`local_groups`/`remote_groups`、`regions`、`bytes` 和耗时。本轮 EOS 应有 `regions>0`、`bytes>0`，允许出现 `local_groups=5 remote_groups=6`。失败时必须保留对应 EOS 输出目录的 `eos.response.http.json`、`eos.response.body`、`request_summary.json` 及四角色日志；HTTP 502 或布局检查报错仍是不通过。

成功要求：脚本退出 0、响应自然 EOS（`finish_reason=stop`）、Proxy 请求归零；同一请求有真实 Prefill→Attention KV 接收证据，Attention 无原 broadcast/解包异常、EngineCore 保持运行。检查本轮 proposed/accepted 增量；一次短请求不能代替完整 DSpark 门禁。失败时保留三角色和 Proxy 日志、请求 JSON/summary、metrics，停止扩大流量。

**5. 【A5 手工执行】EOS 通过后，再执行 5 请求 smoke；成功后才运行完整请求集。**

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite smoke \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --output "$RUN_ROOT/requests-smoke" --run-id "$RUN_ID-smoke"
```

**6. 【A5 手工执行】smoke 通过后，完整重跑 full。** 第 6.3 节重新保存 full 的 metrics-before/生成 token pool。每次使用独立输出目录，不能复用已存在的 `$RUN_ROOT/requests`；脚本会拒绝覆盖：

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite full \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --token-pool "$RUN_ROOT/token-pool.json" \
  --output "$RUN_ROOT/requests-full-$(date +%Y%m%dT%H%M%S)" \
  --run-id "$RUN_ID-full"
```

本次保持原 1/8/32 并发和 300 秒请求超时。成功须包含 32 短/长、全部边界和 SSE/取消恢复，不以原先 44 个成功响应代替本次完整结果。再次失败时，保留 `*.transport.error.json`、`*.response.http.json`、原始响应及四角色日志/metrics；定位首次异常后停止扩大流量。按第 6.5/7 节核对逐组 KV、DSpark 和正常清理证据。只在 C2 F0 完整通过后进入 C3。本次更新后的其他矩阵点也冻结同一修复提交。

## 4. 【A5 手工执行】第 2 步：选择拓扑、验证点和新运行目录

### 4.1 同机设备配置

在单机所有角色终端执行：

```bash
export TOPOLOGY=single
export PREFILL_DP_SIZE=2
export PREFILL_DEVICES=0,1
export ATTENTION_DEVICES=2,3,4,5
export FFN_DEVICES=6,7
export PREFILL_HOST_IP="$HCCL_IF_IP"
export ATTENTION_HOST_IP="$HCCL_IF_IP"
export FFN_HOST_IP="$HCCL_IF_IP"
export PROXY_HOST_IP="$HCCL_IF_IP"
export AFD_HOST="$FFN_HOST_IP"
```

### 4.2 三机设备配置

三机所有角色终端使用一致的拓扑设置及 `site.env` 角色地址；各自 `HCCL_IF_IP` 必须等于本节点角色 IP。Proxy 可部署在 Prefill 主机。

```bash
export TOPOLOGY=cross
export PREFILL_DP_SIZE=8
export PREFILL_DEVICES=0,1,2,3,4,5,6,7
export ATTENTION_DEVICES=0,1,2,3
export FFN_DEVICES=0,1
export AFD_HOST="$FFN_HOST_IP"
```

不同机器的设备 ID 可重复；同机 P/A/F 列表必须互斥。跨机 AFD rendezvous 使用 FFN IP，不使用 localhost。逐项检查路由、防火墙及实际监听地址；API、Mooncake 端口范围、HCCL base 及其派生端口、AFD rendezvous 均须可达且无冲突。仅检查 base port 不充分。

### 4.2.1 两机补充拓扑：同时覆盖远程 KV 和远程 A/F

**两机可以安排三个角色，但下列方案仅依据代码约束设计，尚未在 A5 实机验证。** 本期正式跨机验收仍为第 4.2 节三机拓扑；两机结果独立归档，不直接替代三机功能、精度及性能记录。

| 方案 | 机器 1 | 机器 2 | 能覆盖的跨机链路 |
|---|---|---|---|
| P8+A4F2 | Prefill：0–7；Proxy | Attention：0–3；FFN：4–5 | Prefill→Attention KV；A/F 在同机，不能证明跨机 AFD |
| P2F2+A4（跨机 AFD 验证建议） | Prefill：0–1；FFN：6–7；Proxy | Attention：0–3 | Prefill→Attention KV，以及 Attention↔FFN |

P2F2+A4 共使用 8 张 NPU，两机其余卡空闲。Prefill 与 FFN 共机可能竞争主机内存、CPU 和网络，性能结果必须保留这个部署条件；不能直接与 P8 的三机结果比较并归因为机器数或 AFD 收益。

将下面配置写入两节点共用的实际 `SITE_FILE`，替换示例 IP 后冻结 SHA256；每次加载第 3.1 节环境后均应生效。**选择本节后，不再执行第 4.1/4.2 节覆盖这些地址和设备。**

```bash
# 两台主机实际可达的地址；不要直接运行示例 IP。
export NODE1_IP=192.0.2.10
export NODE2_IP=192.0.2.11
export TOPOLOGY=cross
export PREFILL_HOST_IP="$NODE1_IP"
export FFN_HOST_IP="$NODE1_IP"
export PROXY_HOST_IP="$NODE1_IP"
export ATTENTION_HOST_IP="$NODE2_IP"
export AFD_HOST="$FFN_HOST_IP"
export PREFILL_DP_SIZE=2
export PREFILL_TP_SIZE=1
export PREFILL_DEVICES=0,1
export ATTENTION_RANKS=4
export ATTENTION_DEVICES=0,1,2,3
export FFN_RANKS=2
export FFN_DEVICES=6,7
export TENSOR_PARALLEL_SIZE=1
```

本地 `node.env` 的 `HCCL_IF_IP` 在机器 1 为 `NODE1_IP`、机器 2 为 `NODE2_IP`；网卡、源码/模型路径和 `ASCEND_LOCAL_COMM_RES_PATH` 均填写本机值。机器 1 的 Prefill 检查 NPU 0/1 的 HIXL endpoint，机器 2 的 Attention 检查 NPU 0–3；FFN 不创建 KV connector。共机 Prefill/FFN 使用第 3.1 节不同的 HCCL base port，并复核派生端口无冲突。

按第 4.3/4.4 节生成统一 RUN_ID 和各节点运行目录，RUN_ID 中注明 `two-host-p2f2-a4`；`runtime.json` 中保留完整角色 IP/设备，并另存 `topology.txt` 记录两机布局，不能只写 `TOPOLOGY=cross`。同一主机只创建目录一次，不同角色日志和 PID 文件各自保存。

启动和验证沿用第 6/7 节：机器 1 先启动 Prefill 并确认 ready，然后在机器 1 启动 FFN、紧接着在机器 2 启动 Attention，不等待 FFN 单独 ready；两侧 ready 后机器 1 启动 Proxy。先 EOS、smoke，再 full；两机 C1–C5 仍逐点通过 F0 后推进。采集同一在线请求的远程 KV 接收和 A2F/F2A 完成证据，核对实际远端 IP。正常停服仍为停止新请求、请求归零、停 Proxy、停 Attention、等待 FFN shutdown、停 FFN、最后停 Prefill。

### 4.3 C1–C5：设置 CASE、轮次和运行目录

本节用于 PD 的 C1–C5；standalone 先执行第 5 节的初始化和配置。
每个角色终端使用同一 CASE/CYCLE/RUN_ID。RUN_ID 在协调终端确定一次，再复制到各节点，不能各节点独立生成时间戳。
加载第 3.1 节环境并选择第 4.1、4.2 或 4.2.1 节拓扑后，**每台主机只在第一个角色终端创建目录一次**：

```bash
export CASE=C1
export CYCLE=1
# 每轮替换为新的实际值；本轮所有角色使用相同值。
export RUN_ID=20261008T180000-single-C1-r1
: "${RUN_BASE:?先按第 3.1 节加载 node.env 中的 RUN_BASE}"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
```

**本机其他角色终端**加载第 3.1 节环境及相同拓扑后，只执行以下赋值，不重复创建目录：

```bash
export CASE=C1
export CYCLE=1
export RUN_ID=20261008T180000-single-C1-r1
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
test -d "$RUN_ROOT"
```

随后**每个角色终端**执行本轮开关配置：

```bash
case "$CASE" in
  C1) export EXECUTION_MODE=eager U_BATCHES=1 ENABLE_DSPARK=0 ;;
  C2) export EXECUTION_MODE=eager U_BATCHES=1 ENABLE_DSPARK=1 ;;
  C3) export EXECUTION_MODE=full-decode-only U_BATCHES=2 ENABLE_DSPARK=0 ;;
  C4) export EXECUTION_MODE=eager U_BATCHES=2 ENABLE_DSPARK=1 ;;
  C5) export EXECUTION_MODE=full-decode-only U_BATCHES=2 ENABLE_DSPARK=1 ;;
  *) exit 2 ;;
esac
export MODEL_PATH="$FLASH_MODEL_PATH"
if [[ "$ENABLE_DSPARK" == 1 ]]; then
  export MODEL_PATH="$DSPARK_MODEL_PATH"
fi
export ENABLE_PD=1
export AFD_HCCL_STAGE_DIAGNOSTICS=1
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP="${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP:-1}"
export AFD_HCCL_GRAPH_U2_HYBRID_DAG="${AFD_HCCL_GRAPH_U2_HYBRID_DAG:-1}"
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM="${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM:-1}"
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM="${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM:-1}"
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER="${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER:-1}"
```

RUN_ID 示例每轮必须改成新的实际值，且与 CASE/TOPOLOGY/CYCLE 一致。

上述 CASE 中的 `ENABLE_DSPARK` 表示 **Attention 的 draft 开关**，同时用于先为各角色选择同一配套 `MODEL_PATH`。C2/C4/C5 选择 DSpark checkpoint 后，第 6.1 节仅在 FFN 启动命令前临时传入 `ENABLE_DSPARK=0`；这不会重新选择 Flash checkpoint，也不会关闭 Attention 的 DSpark。Prefill 不添加 speculative config。不要在共用终端中用全局 `export ENABLE_DSPARK=0` 代替 FFN 的命令前缀，否则随后启动 Attention 会继承错误值。

### 4.4 保存启动前环境和运行清单

C1–C5 完成第 4.3 节配置后执行；standalone 完成第 5 节配置后执行。
在每台主机的一个已配置终端采集一次，然后再运行对应的 `nohup` 启动命令：

```bash
test -d "$RUN_ROOT"
sha256sum "$SITE_FILE" "$NODE_FILE" > "$RUN_ROOT/config.sha256"
cp "$SITE_FILE" "$RUN_ROOT/site.env"
cp "$NODE_FILE" "$RUN_ROOT/node.env"
npu-smi info > "$RUN_ROOT/npu-before.txt"
date -Iseconds > "$RUN_ROOT/start-time.txt"
"$PYTHON_BIN" - "$RUN_ROOT/runtime.json" <<'PY'
import importlib, json, os, subprocess, sys
from importlib.metadata import version
names = '''RUN_ID TOPOLOGY CASE CYCLE IMAGE_DIGEST CANN_ROOT CANN_BUILD HCCL_PACKAGE
ASCEND_LOCAL_COMM_RES_PATH MODEL_PATH PREFILL_HOST_IP ATTENTION_HOST_IP FFN_HOST_IP
PROXY_HOST_IP HCCL_IF_IP NIC_NAME PREFILL_DEVICES ATTENTION_DEVICES FFN_DEVICES
PREFILL_DP_SIZE ATTENTION_RANKS FFN_RANKS PREFILL_API_PORT ATTENTION_API_PORT PROXY_PORT
PREFILL_KV_PORT DECODE_KV_PORT AFD_HOST AFD_PORT MAX_NUM_SEQS MAX_MODEL_LEN GPU_MEMORY_UTILIZATION
EXECUTION_MODE U_BATCHES ENABLE_PD ENABLE_DSPARK AFD_HCCL_STAGE_DIAGNOSTICS
AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS AFD_HCCL_EAGER_U2_STREAM_OVERLAP
AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP
AFD_HCCL_GRAPH_U2_HYBRID_DAG AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM
AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER'''.split()
runtime = {'environment': {k: os.environ.get(k) for k in names},
           'python': sys.executable, 'sources': {}, 'packages': {}}
for key in ('AFD_PLUGIN_ROOT', 'VLLM_ROOT', 'VLLM_ASCEND_ROOT'):
    root = os.environ[key]
    runtime['sources'][key] = {
        'path': root,
        'commit': subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': subprocess.check_output(['git', '-C', root, 'status', '--porcelain'], text=True)}
for module, package in [('torch', 'torch'), ('torch_npu', 'torch-npu'),
                        ('vllm', 'vllm'), ('vllm_ascend', 'vllm-ascend'),
                        ('afd_plugin', 'vllm-afd-plugin')]:
    runtime['packages'][package] = {'version': version(package),
                                   'source': importlib.import_module(module).__file__}
with open(sys.argv[1], 'w') as f:
    json.dump(runtime, f, ensure_ascii=False, indent=2)
    f.write('\n')
PY
```

保存有效环境和运行清单，至少包括三个源码提交/dirty state、Python package/version/source、镜像、CANN/HCCL、角色地址/设备/端口、CASE/CYCLE、模型 hashes、最终开关。
只采集本次相关环境变量，避免把无关认证信息写入证据。上述 runtime 保存启动前意图值；角色脚本可能覆盖设置，启动后还须归档实际 argv 和逐 rank 的诊断配置日志，将二者与 runtime 对照，不能仅凭此 JSON 判定最终开关已生效。

## 5. 【A5 手工执行】第 3 步：先回归 standalone A4F2 Graph/U2

M1 修正后的新 recipe 必须先通过一次 standalone 回归，然后才能进入同机 C1。
在单台空闲 A5，**先在同一个启动终端执行第 3 节环境加载与版本检查，再执行下面整段**。
`RUN_ROOT` 的初始化就放在这里：在设备配置和两条 `nohup` 命令之前。
本节在一个终端启动 FFN 和 Attention，无需重复创建目录；预检和 C1 的记录各用新的 RUN_ID。

```bash
# 1. 本轮日志目录：RUN_BASE 已在 node.env 中定义并加载。
: "${RUN_BASE:?先按第 3.1 节加载 node.env 中的 RUN_BASE}"
export TOPOLOGY=single CASE=standalone CYCLE=1
export RUN_ID="$(date +%Y%m%dT%H%M%S)-single-standalone-r${CYCLE}"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"

# 2. standalone 的设备、模型和开关。
export ATTENTION_DEVICES=0,1,2,3
export FFN_DEVICES=4,5
export ATTENTION_HOST_IP="$HCCL_IF_IP"
export FFN_HOST_IP="$HCCL_IF_IP"
export AFD_HOST="$FFN_HOST_IP"
export MODEL_PATH="$FLASH_MODEL_PATH"
export ENABLE_PD=0 ENABLE_DSPARK=0
export EXECUTION_MODE=full-decode-only U_BATCHES=2
export AFD_HCCL_STAGE_DIAGNOSTICS=1
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=1
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=1
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
# 本节客户端与 Attention 在同一主机，HTTP 使用 loopback。
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
export BASE_URL="$ATTENTION_URL"
```

在上述同一终端执行**第 4.4 节采集命令**，保存本轮生效配置，然后紧接着启动：

```bash
# 3. 启动 FFN 与 Attention；此时 RUN_ROOT 已存在。
test -d "$RUN_ROOT"
nohup bash "$RECIPE/afd_ffn.sh" > "$RUN_ROOT/ffn.log" 2>&1 &
echo $! > "$RUN_ROOT/ffn.pid"
nohup bash "$RECIPE/afd_attention.sh" > "$RUN_ROOT/attention.log" 2>&1 &
echo $! > "$RUN_ROOT/attention.pid"
```

Attention ready 且两个 FFN rank 进入 loop 后，使用第 6.2 节的 Attention readiness/FFN loop 检查（跳过 Prefill 和 Proxy）。
先执行下面的 5 请求 `smoke`，通过后再按第 6.3 节生成 token pool，执行第 6.4 节的完整 92 请求验证。

这里的 **`--standalone` 加在 Python 请求验证脚本 `tools/validation/a5_1030_requests.py` 的命令上**，不加到 `afd_ffn.sh`、`afd_attention.sh` 或 `vllm serve` 的启动命令。
它是无须填写值的标志，告诉验证脚本本轮没有 Proxy，跳过 Proxy 的 `/healthcheck` 和 `request_num` 检查；其余请求检查及 Attention 请求归零检查保留。
业务请求的目标由 `--base-url` 决定，`--standalone` 不自动改写 URL，所以两个 URL 都明确填写 Attention 地址。

在当前 standalone 终端执行以下**轻量请求检查命令**，不需要 token pool：

```bash
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --standalone --suite smoke \
  --base-url "$ATTENTION_URL" \
  --attention-url "$ATTENTION_URL" \
  --output "$RUN_ROOT/requests-smoke" \
  --run-id "$RUN_ID"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-after.txt"
```

验证 capture、在线 replay、全部 Attention rank 的 stage 0/1、未捕获 shape fallback、正常停机及 NPU 清理。
没有 Prefill/Proxy，也不要求 PD/DSpark 证据。完成后全部停服，再恢复同机 P2+A4+F2 设备配置；不得将 standalone 的进程或日志复用到 C1。

## 6. 【A5 手工执行】第 4 步：每个 C1–C5 的启动、ready 和请求步骤

同机角色在本机执行；两机/三机只在分配的主机执行其角色。每轮角色使用同一 RUN_ID。

### 6.1 启动 Prefill，然后立即相邻启动 FFN 与 Attention

Prefill 节点：

```bash
# 已在 site/node 和 CASE 配置之后、runtime 采集之前设置；启动时再核对。
test "${GPU_MEMORY_UTILIZATION:-}" = 0.91
test "$MAX_MODEL_LEN" = 4096
nohup bash "$RECIPE/afd_prefill.sh" > "$RUN_ROOT/prefill.log" 2>&1 &
echo $! > "$RUN_ROOT/prefill.pid"
```

FFN 节点，然后 Attention 节点；不要等待 FFN ready 后才启动 Attention，它们需要 rendezvous。**C4 的 FFN 设 `ENABLE_DSPARK=0` 是正确的**：完整 DSpark draft 在 Attention Worker 上运行；远端 FFN 执行 target 模型的 FFN 计算，不另起 drafter。FFN 也不直接收取 Prefill 的 PD KV，因此 FFN 进程始终使用 `ENABLE_PD=0 ENABLE_DSPARK=0`；系统的 PD/DSpark 由 Attention 的配置开启。

同机 C4 的 A/F 参数应为：

| 角色 | `ENABLE_PD` | `ENABLE_DSPARK` | 执行模式 / U | `MODEL_PATH` |
|---|---:|---:|---|---|
| Attention | 1 | 1 | eager / 2 | `$DSPARK_MODEL_PATH` |
| FFN | 0 | 0 | eager / 2 | `$DSPARK_MODEL_PATH` |

下面 `变量=值 nohup ...` 只覆盖此次 FFN 启动及其子进程环境，**不会把父终端原有的 `ENABLE_DSPARK=1` 改为 0**，随后 Attention 仍继承第 4.3 节的 C4 配置。FFN 启动脚本的 preflight 明确拒绝 `ENABLE_DSPARK=1`，会报告 `DSpark config belongs only to Attention`。

```bash
# FFN 节点：只给本次 FFN 进程传入 0；MODEL_PATH 仍为本轮配套 checkpoint。
ENABLE_PD=0 ENABLE_DSPARK=0 nohup bash "$RECIPE/afd_ffn.sh" \
  > "$RUN_ROOT/ffn.log" 2>&1 &
echo $! > "$RUN_ROOT/ffn.pid"
```

```bash
# Attention 节点：C4 应保留 eager/U2、ENABLE_PD=1、ENABLE_DSPARK=1。
nohup bash "$RECIPE/afd_attention.sh" > "$RUN_ROOT/attention.log" 2>&1 &
echo $! > "$RUN_ROOT/attention.pid"
```

### 6.1.1 【A5 手工执行】C4 Prefill 的 KV 预算不足

附件 `cf4f76620aea40d8aa3c8c77d0c252b2.gz` 的实际 Prefill CLI 为 `gpu_memory_utilization=0.90`、4096、P2/TP1、DSpark checkpoint；两个 Worker 仅有 1.63 GiB KV，低于单条 4096 请求需要的 1.70 GiB，首次在 KV 容量检查失败。最新附件 `6eb6a279f37c4040843516b1966b7354.gz` 的 `prefill-retry-20261013T072805.log` 已确认 0.91、两个 Worker 均有 2.54 GiB KV、ready 成功，随后 C4 full 92 项请求检查通过；不用再重复此次重试。剩余项见[C4 本轮核对结果](DEEPSEEK_V4_AFD_C4_20261013_RESULT_ZH.md)。

设置必须在 **Prefill 启动终端、加载配置之后、nohup 之前** 生效。修改实际 `SITE_FILE` 的 `export GPU_MEMORY_UTILIZATION=...` 为 0.91（缺行则加在 `export MAX_NUM_SEQS=32` 后），正式新轮在三个角色终端加载并先采集 runtime/config。以下是失败时的历史操作入口：A/F 已启动时，可仅诊断重试 Prefill；先确认旧 P 进程和端口已释放，日志另存 `prefill-retry-时间戳.log`、PID 另存 `prefill-retry.pid`，停止时使用新 PID，不覆盖旧错误材料。具体命令见[C3/C4 核对第 4 节](DEEPSEEK_V4_AFD_C3_C4_20261013_REVIEW_ZH.md)。不降低 4096 或修改模型 config 来绕过；重试 ready 不等于 C4 F0。

### 6.2 判断 ready 后再启动 Proxy

协调终端检查 Prefill/Attention，最多等待 20 分钟，失败保存日志并停止推进：

```bash
wait_http() {
  local url="$1" name="$2" deadline=$((SECONDS + 1200))
  until curl --noproxy '*' -fsS --max-time 5 "$url" > "$RUN_ROOT/$name-ready.txt"; do
    if ((SECONDS >= deadline)); then return 1; fi
    sleep 5
  done
}
wait_http "http://$PREFILL_HOST_IP:$PREFILL_API_PORT/health" prefill
wait_http "http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT/health" attention
curl --noproxy '*' -fsS "http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT/v1/models" \
  > "$RUN_ROOT/models.json"
```

FFN 节点要求两个 rank 均进入 connector loop。计数只作初筛，最终核对 rank 身份，重复日志不能充当另一个 rank：

```bash
grep -nF 'AFD FFN EngineCore started; workers run connector loop.' "$RUN_ROOT/ffn.log"
test "$(grep -cF 'AFD FFN EngineCore started; workers run connector loop.' "$RUN_ROOT/ffn.log" || true)" -ge 2
grep -nF 'AFD HCCL diagnostics configuration:' "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
```

上一个 `grep` 在两机/三机分别对本机日志执行；日志不在本机时只检查本机角色日志。Graph 点逐 rank 核对最终五个 Graph 开关与本轮配置一致：默认均为 1，显式关闭项应为 0；配置与生效值不一致时返回 M1，不开始流量。`8911` 是 FFN 内部设置，不是 HTTP健康接口。
单个 Attention HTTP API 的 health 只作入口检查，另需核对四个 Attention rank ready、四个 drafter（DSpark 点）及两个 FFN rank ready。
PD 的 C2/C4/C5 还须按第 3.4 节确认四个 EngineCore DP rank 的 `AFD DSpark PD scheduler compatibility active:` 标记。

Prefill/Attention ready 且 FFN loop 完整后，在 Proxy 主机执行：

```bash
nohup bash "$RECIPE/afd_proxy.sh" > "$RUN_ROOT/proxy.log" 2>&1 &
echo $! > "$RUN_ROOT/proxy.pid"
export BASE_URL="http://$PROXY_HOST_IP:$PROXY_PORT"
export ATTENTION_URL="http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT"
wait_http "$BASE_URL/healthcheck" proxy
"$PYTHON_BIN" - "$RUN_ROOT/proxy-ready.txt" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
assert c['status'] == 'ok'
assert c['prefill_instances'] == 1 and c['decode_instances'] == 1
assert c['request_num'] == 0
PY
```

### 6.2.1 【A5 手工执行】Proxy 转发 Prefill 返回 504

2026-10-13 C3 附件 `e5c46c41caf044ef8dbf0fb6c688296d.gz` 中，Proxy 请求 `http://7.246.125.18:8100/v1/chat/completions` 连续三次返回 504，随后客户端收到 Proxy 的 500。Prefill 的 `/health` 直连成功，但没有对应 chat POST 日志。Attention 四 rank Graph capture 已完成，`EXECUTION_MODE=full-decode-only`、`U_BATCHES=2`、`ENABLE_DSPARK=0` 与 C3 一致；该 504 首先发生在 Prefill HTTP 转发，尚未到 Decode Graph。

首要排查项是 **Proxy 进程继承的 HTTP 环境代理，以及该进程的 `NO_PROXY/no_proxy` 是否覆盖后端 IP**。附件不包含这些进程环境变量，故尚不能仅凭 504 确认网关来源。旧 `tools/proxy_server.py` 的两处 HTTPX 客户端默认继承环境代理；`curl --noproxy '*'` 的成功检查和 Proxy `/healthcheck` 的实例计数都不能证明实际转发走直连。

**现场立即修正**：在 `node.env` 末尾加入下面两行，并在**启动 Proxy 的终端**重新加载 `site.env`、`node.env`（二者的实际文件路径沿用第 3.1 节）。不要将 HCCL/AFD 地址改成 loopback。

```bash
# 加入 node.env 末尾；site.env 必须先加载。
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,$HCCL_IF_IP,$PREFILL_HOST_IP,$ATTENTION_HOST_IP,$FFN_HOST_IP,$PROXY_HOST_IP"
export no_proxy="$NO_PROXY"
```

在 Proxy 终端执行，先保存已启动进程的豁免项，再加载修正。不要输出可能包含凭据的代理 URL：

```bash
tr '\0' '\n' < "/proc/$(cat "$RUN_ROOT/proxy.pid")/environ" \
  | grep -E '^(NO_PROXY|no_proxy)=' || true
source "$SITE_FILE"
source "$NODE_FILE"
curl --noproxy '*' -fsS --max-time 10 \
  "http://$PREFILL_HOST_IP:$PREFILL_API_PORT/health"
curl --noproxy '*' -fsS --max-time 10 \
  "http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT/health"
```

两个 health 都成功时，停止旧 Proxy 并等待 9000 端口释放，再按第 6.2 节启动 Proxy；**运行后的 export 不会改变旧进程，必须重启 Proxy**。保留旧 `proxy.log`，新日志使用独立文件。然后先经 Proxy 发送单个 EOS，请求使用 `--suite eos`、新的 `--output`，不加 `--standalone`；Prefill/Attention 两边都应出现该业务请求。EOS 成功后再 smoke/full。health 失败时先恢复角色服务，不只重启 Proxy；附件末尾 01:25:56 UTC 已有三个角色的 EngineDeadError，不能假设它们仍在运行。

**代码修正**：本分支为 Proxy 的 Prefill/Decode 客户端和动态节点 `/models` 探测显式设置 `trust_env=False`，内部 HTTP 始终按配置地址直连。CPU 回归使用真实本机后端和返回 504 的模拟环境代理，验证默认 HTTPX 会被代理拦截，而修正后的 Prefill、Decode 转发及探测均直达后端；不依赖临时关闭 Graph/U2。A5 实机转发仍需现场复测。

设计复核：修改属于自有 Proxy 的内部 HTTP 通信，不改 vLLM/Ascend、AFD/KV 协议或模型执行；沿用客户端复用，无新增全局状态或热路径网络调用。环境代理豁免作为内部服务直连策略保留，未来更新上游 Proxy 示例时继续核对两处客户端设置。

另一个归档问题是附件 `site.env` 的 `AFD_EXPECTED_COMMIT=567ff2d…`，而 `runtime.json` 的实际插件提交为 `6c4dc19…`。这是旧冻结配置不一致，不是已确定的 504 根因。更新交付代码后，将所选完整提交写回实际 `SITE_FILE`，重新采集 runtime 和配置 hash；不要只修改临时 shell 变量。

### 6.3 保存 metrics-before，生成精确 token-ID 输入

**2026-10-19 cross-C5 更新**：最新两机 full 请求结果已通过；本轮补采及后续冷启动安排按[cross-C5 结果第 5 节](DEEPSEEK_V4_AFD_CROSS_C5_20261016_RESULT_ZH.md)。以下 10-13 日 C3/C4 段落为当时记录，不代表当前跨机验证仍停留在 C4。

**2026-10-13 C4 最新更新**：附件 `6eb6a279f37c4040843516b1966b7354.gz` 在 0.91 下 Prefill 重试成功，P/A/F 使用 DSpark checkpoint；full 92 项请求检查通过、90 个非流式响应全部非空且无 token ID `0`、A/Proxy 请求归零，四 Attention rank 的 drafted/accepted 和在线 U2 已确认。但 Attention 停服四次 Segfault、FFN 同步报 507899，DSpark metrics before/after 仍为 0，缺本轮 runtime/config 和最终清理快照，完整 F0 保持 pending。按[C4 本轮证据与补采步骤](DEEPSEEK_V4_AFD_C4_20261013_RESULT_ZH.md)继续，不立即重发已通过的 full；C5 按现场要求暂缓分析。

**C3 剩余项不变**：附件 `48e1a5df943b46b2bc4bc77d7437d0b0.gz` 在实际插件 `f3c8399…` 上完成定向 33 项和 full 92 项，请求检查通过、真实 Graph/U2、A/Proxy 请求归零；但六 A/F rank 的五开关实际仍全 `0`，定向 1 条/full 8 条有尾部 token `0`，其中最长连续 49 个，并非原输出退化已关闭。停服文件记录失败，Attention 四次 Segfault、FFN 正常 shutdown 标记为 0；最终清理不能代替正常停服。旧 INFO RoPE 标记可能被插件 root 日志级别过滤，本次仅改为 WARNING 便于核验，不认定缺标记就是旧源码。详见[C3/C4 前次证据与操作步骤](DEEPSEEK_V4_AFD_C3_C4_20261013_REVIEW_ZH.md)。C4 eager/DSpark 请求成功不能替代正式默认全开 C3 和输出正确性验证，三机/最终精度验收仍待前置门禁关闭。

业务请求经 Proxy；`/tokenize` 与 `/metrics` 是 Attention 管理调用，不计作 PD 请求。
使用 token-ID prompt，避免字符长度被误写为 KV block 边界长度。

```bash
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-before.txt"
"$PYTHON_BIN" - "$RUN_ROOT/tokenize-request.json" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump({'model': 'dsv4-afd', 'prompt': 'The sky is blue. ' * 2048,
               'add_special_tokens': False}, f)
PY
curl --noproxy '*' -fsS --max-time 60 "$ATTENTION_URL/tokenize" \
  -H 'Content-Type: application/json' \
  --data-binary "@$RUN_ROOT/tokenize-request.json" > "$RUN_ROOT/token-pool.json"
```

**2026-10-14 多样长输入对照**：上述旧输入重复同一句话，作为已有故障复现的
基线保留。需要构造多样输入时，用新增工具保存独立 token pool：

```bash
DIVERSE_POOL_ROOT="$RUN_ROOT/token-input-diverse-$(date +%Y%m%dT%H%M%S)"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_token_pool.py" \
  --attention-url "$ATTENTION_URL" \
  --output "$DIVERSE_POOL_ROOT"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite full-to-c8-short \
  --base-url "$BASE_URL" \
  --attention-url "$ATTENTION_URL" \
  --token-pool "$DIVERSE_POOL_ROOT/token-pool.json" \
  --output "$RUN_ROOT/requests-prefix-diverse-$(date +%Y%m%dT%H%M%S)" \
  --run-id "$RUN_ID-prefix-diverse"
```

工具使用固定的12段英文自然文本，涵盖海洋、历史、软件、交通、烘焙等不同主题，
不通过随机整数冒充 token ID。由当前 Attention `/tokenize` 返回真实 IDs，要求
至少1024个非负整数 token；长输入仍取前1024个，边界输入仍按原长度切片，短输入
不变。调用只分词，不发送推理请求。输出目录必须不存在，原重复输入不会被覆盖；
保存原始文本、分词响应与 prompt/pool SHA256，便于冻结和重跑。新 pool 也可供
`--suite full` 使用。standalone 的请求验证需添加 `--standalone`。

保持模型、并发、输出预算、服务配置和启动前的业务历史一致，再比较重复输入与
多样输入。内容改变可能影响 MoE 路由、DSpark 接受率和各DP完成时机；多样输入
通过不能当作原故障已修复，重复输入也不是天然非法。

### 6.4 执行完整请求集

Proxy 主机或可达的验证客户端执行；token-pool 文件从 Attention 拷贝到该节点。
`full` 覆盖完整验收请求集，不是启动探活或性能 benchmark；先通过 `smoke` 再运行：

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite full \
  --base-url "$BASE_URL" \
  --attention-url "$ATTENTION_URL" \
  --token-pool "$RUN_ROOT/token-pool.json" \
  --output "$RUN_ROOT/requests" \
  --run-id "$RUN_ID"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-after.txt"
```

上面的命令用于 C1–C5，`BASE_URL` 指向 Proxy，不添加 `--standalone`。
standalone 的完整验证在上面的 Python 命令添加 `--standalone`，并使用第 5 节已定义的 loopback `ATTENTION_URL`/`BASE_URL`。
保留 `--suite full` 和 `--token-pool`，输出到新目录 `requests/`；第 5 节的 `requests-smoke/` 只保存轻量检查。

| `--suite` | 推理请求数 | 用途 |
|---|---:|---|
| `smoke` | 5 | EOS、1 个短输入、正常 SSE、取消 SSE、恢复请求；启动后先执行 |
| `eos` | 1 | 单独定位自然 EOS，不重复并发或边界输入 |
| `tail` | 4 | EOS、正常 SSE、取消 SSE、恢复请求；补测旧脚本在 EOS 处停止后尚未执行的项目 |
| `full`（默认） | 92 | 完整功能请求覆盖；正式请求门禁使用此结果 |
| `c8-short` | 9 | EOS + 8 并发短输入，定向诊断，不需要 token pool |
| `c32-short` | 33 | EOS + 32 并发短输入，定向诊断，不需要 token pool |
| `full-to-c8-short` | 11 | EOS → c1-short → c1-long → c8-short，保留 full 前缀；需要 token pool |

HTTP `/metrics`/`healthcheck` 查询不计入上表的推理请求数。`full` 的 92 个请求由
`2 × (1 + 8 + 32) = 82` 个长/短输入并发请求、6 个边界请求及4个EOS/SSE/取消/恢复请求构成。
并发用于覆盖 DP 调度和 U2 执行，边界输入用于检查 KV block 长度；不是每轮都做 GSM8K。
脚本现在先执行 EOS，失败后立即停止，不再先跑 88 个负载/边界请求。
子套件的成功不等同于完整门禁：查看 summary 的 `suite` 与
`full_request_suite_passed`；只有 `full` 成功会将后者写为 `true`。

**EOS 报错后的单独复测（standalone 在服务本机执行）**：

```bash
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --standalone --suite eos \
  --base-url "$ATTENTION_URL" --attention-url "$ATTENTION_URL" \
  --output "$RUN_ROOT/requests-eos-$(date +%Y%m%dT%H%M%S)" \
  --run-id "$RUN_ID-eos"
```

EOS 通过后，可将该命令的 `--suite eos` 改为 `--suite tail`，并将输出目录及 run ID 的
`eos` 后缀改为 `tail`，补测 4 个剩余项目。保留原来的 88 个响应及本次补测时间窗，
用于定位和审阅；补测结果不会自动把原失败 summary 或完整 F0 改为通过。

EOS 使用 `/v1/chat/completions`、user message 和服务的 DeepSeek-V4 聊天模板，
显式关闭 thinking；不设置人为 `stop` 字符串、不强制忽略 EOS。
旧脚本给 `/v1/completions` 直接传一句裸文本，可能出现续写而非简短回答，达到 1024 token
并返回 `finish_reason=length`；这是 EOS 用例未满足，不能单独据此判定 AFD 数据面错误。
仍不自然结束时，用 `--suite eos --eos-prompt '现场冻结的用户输入' --eos-max-tokens 1024`
单独定位，保存 `eos.request.json` 与 `eos.json`；不能通过取消 EOS 门禁伪造通过。

脚本依赖环境已有的 `httpx`；输出目录必须不存在，避免覆盖旧证据。脚本依次检查：

1. 独立 chat EOS 请求，默认最多 1024 token，必须自然 `finish_reason=stop`；
2. 并发 1、8、32 的短输入和 1024 token 长输入，每条强制输出 64 token 并检查截断；
3. 31/32/33/63/64/65 token 输入，检查 PD 后 `usage.prompt_tokens` 没有改变；
4. 正常 SSE，必须有文本、结束原因和 `[DONE]`；
5. SSE 收到实际文本后关闭客户端连接，随后恢复请求成功；
6. 60 秒内 Proxy `request_num=0`，Attention 所有已暴露的 running/waiting gauge 为 0；standalone 只跳过 Proxy 检查，仍要求 Attention 请求归零。

每个请求原始响应和 SSE 文件写入 `requests/`，结果为 `request_summary.json`。
非流式请求体另存为 `<name>.request.json`，便于核对模型名、输入和 EOS 参数。
本 recipe 的 served model name 是 `dsv4-afd`；填写 `dsv4` 会返回 model-not-found 404。
EOS 用例若未自然结束，先冻结合适的 EOS prompt/预算再重跑，不可直接取消 EOS 门禁。
脚本返回 0 只表示请求子项通过，不自动判断首 token 内容、远程链路、Graph 或完整 F0。
取消后还要核对本次请求的上游 abort/cancel 处理及 Prefill KV/session 资源释放，不能仅用 Proxy 计数归零代替。当前 Proxy 自行生成上游 request ID，客户端发送的`X-Request-Id` 不保证原样转发；使用响应 ID、请求时间窗及 Proxy 的 ID 映射定位，缺少映射时补采诊断证据，不把客户端标签当作已经贯通的服务端 ID。

### 6.4.1 C5 HCCL/AICPU 故障的一次采集

2026-10-13 后续 full 在 `c8-short` 阶段发生 FFN launcher 参数异常。可独立
执行该批，套件先检查自然 EOS，再发送 8 条短输入；不执行 full 的前序长输入：

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite c8-short \
  --base-url "$BASE_URL" \
  --attention-url "$ATTENTION_URL" \
  --output "$RUN_ROOT/requests-c8-short-$(date +%Y%m%dT%H%M%S)" \
  --run-id "$RUN_ID-c8-short"
```

不需要 `--token-pool`；standalone 时添加 `--standalone`，两个 URL 指向 Attention。
核对 `batch-c8-short.summary.json` 中 `passed_count=8`、`failed_count=0`，以及
`request_summary.json` 中 `request_checks_passed=true`。独立执行通过仍不能排除
full 前序请求/shape 切换触发的故障；保留服务是否刚启动和此前请求序列。
已经发生 FFN worker failure 的服务需按停服/清理/启动流程恢复后再测。

若独立 `c8-short` 通过而 full 在该批失败，使用以下入口保留前序请求历史：

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite full-to-c8-short \
  --base-url "$BASE_URL" \
  --attention-url "$ATTENTION_URL" \
  --token-pool "$RUN_ROOT/token-pool.json" \
  --output "$RUN_ROOT/requests-full-to-c8-short-$(date +%Y%m%dT%H%M%S)" \
  --run-id "$RUN_ID-full-to-c8-short"
```

严格按 EOS → c1-short → c1-long → c8-short 执行，共 11 个推理请求。
使用原 full 的 token pool（前 1024 个 token），每批完成后立即执行下一批，
不在中间重启服务或额外等待请求归零；仅在整个序列结束后检查请求归零。
该入口不执行 c8-long、c32、边界或 SSE 请求，结果的
`full_request_suite_passed` 保持 `false`。保持原 C5 服务配置；优先在重新启动并
完成 ready 的服务上执行。若沿用已做过其他请求的服务，记录此前请求序列。
保存三个批次 summary、逐条请求/响应、四角色日志和 metrics；失败时仍等待本批
所有客户端结果落盘。CLI 的 `--token-pool` 缺失会在发请求前报错。

针对 2026-10-13 C5 并发 32 超时、FFN PID 83688/83687 的故障，现场 FFN
节点可执行交付分支中的采集脚本。`RUN_ROOT` 必须指向失败轮的日志目录；
`C5_TASK_DIR` 指向原服务工作目录或安装根，可补充异常 dump 搜索位置。

```bash
RUN_ROOT="${RUN_ROOT:?请先设置为失败 C5 的日志目录}" \
C5_TASK_DIR=/home/z00569729/dsv4-afd-a128f32-r3 \
bash "$AFD_PLUGIN_ROOT/tools/validation/collect_c5_evidence.sh"
```

脚本保存现存服务材料、CANN run/debug 轮转日志、extra-info/atrace、版本信息，
并使用 driver 的 msnpureport 导出设备系统日志；相关进程仍存在时保存其工作目录、
选定环境和库加载路径。日志中的其他 Worker PID 会自动加入采集。
若 msnpureport 提示容器内需要 `--docker`，脚本自动带该参数重试一次，保留两次
输出和退出码；`DEVICE_EXPORT_FINAL_EXIT=0` 及导出文件清单用于核对采集是否成功。
空 extra-info 目录会明确标记为没有文件，不能当作已取得异常 dump。
本次固定故障窗口为 UTC 08:27:47–08:27:51，默认搜索还包括 `/root`、
`/home/z00569729` 和 `/tmp`。用于其他故障时须调整脚本中的固定 PID、时间窗口及搜索根。

采集不重启服务、不发送信号、不切换 CANN，也不导入 NPU runtime。
结束后回传打印的 `OUTPUT=…tar.gz`；缺失项、权限错误和命令退出码保存在包内
`meta/collection-status.txt`，tar 错误另存于打印的 `TAR_STATUS` 路径。
采集时状态不能替代故障时状态；现存材料无法恢复未落盘或已老化的异常 dump。

### 6.4.2 eager U2 overlap 与 Graph 的对照验证

2026-10-14 的附件已确认：五个 Graph 开关均为 0 时，真正 eager/U2 的 full
92 项及同进程连续两次 `full-to-c8-short` 均通过，但实际
`eager_u2_stream_overlap=1`；FULL_DECODE_ONLY/U2、五开关均为 0、eager overlap
为 0 的 full 在 c8-long 失败。因此需要补测 eager overlap=0，控制这一差异。

`common.sh` 现已保留 eager/U2 的 `AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0/1` 显式
设置，未设置或空值仍默认 1；非法值在启动前报错。FULL_DECODE_ONLY/U2 仍将其
固定为 0。这个变量与五个 Graph 开关独立，设为 0 仍保留 DBO 和 U2。

| 对照组 | CASE / target 模式 | eager overlap | 五个 Graph 开关 | 用途 |
|---|---|---:|---|---|
| E1 | C4 / eager，GraphMode.NONE | 1 | 全 0 | 已有 full 及两次连续前缀通过的基线 |
| E0 | C4 / eager，GraphMode.NONE | 0 | 全 0 | 本次先补测，排查 eager overlap 的影响 |
| G0 | C5 / FULL_DECODE_ONLY | 0 | 全 0 | 已有 full 失败；E0 通过后可在相同冻结版本复核 |

对照期间固定源码、CANN/torch_npu、PD/DSpark、P2+A4+F2、U2、DBO 阈值、模型、
内存配置、诊断设置和请求输入。只在诊断命名中使用 E0/E1/G0；CASE 仍按表填写。
拉取交付分支后更新本轮 `AFD_EXPECTED_COMMIT`，按第 4.4 节保存新版本及配置。
不要将本地其他 CANN 环境替换到现场 A5 配置。

**先执行 E0**：在 Attention、FFN 两个启动终端加载 site/node 配置及第 4.3 节
C4 配置后，启动前都执行以下设置；只在客户端 export 不会改变已有服务：

```bash
export EXECUTION_MODE=eager U_BATCHES=2
export AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
```

按第 7 节停服，分别使用新的 RUN_ID/RUN_ROOT，按第 4.4、6.1、6.2 节采集、
启动并完成 ready。Attention 保持 `ENABLE_PD=1 ENABLE_DSPARK=1`；FFN 使用
第 6.1 节命令前缀 `ENABLE_PD=0 ENABLE_DSPARK=0`，两侧仍使用 DSpark checkpoint。
实际日志须确认 target `enforce_eager=True` / `GraphMode.NONE`，并逐 rank 核对
四个 Attention、两个 FFN 的 `eager_u2_stream_overlap=0` 和五个 Graph 开关为 0：

```bash
grep -nF 'AFD HCCL diagnostics configuration:' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
```

使用已通过的 diverse token pool 的**原文件绝对路径**，不能重新生成输入。
在客户端设置一次，下列两次冷启动均沿用它；文件应有 1549 个 token，前 1024
个 token 有 571 个不同 ID，现场既有文件 SHA256 为
`b4040dba955525d816eabdf8bd89188f499980496b0629b20b3b943e765f5e79`。

```bash
export TOKEN_POOL_FILE=/填写原diverse输入目录的绝对路径/token-pool.json
test -f "$TOKEN_POOL_FILE"
sha256sum "$TOKEN_POOL_FILE" > "$RUN_ROOT/comparison-token-pool.sha256"
```

第一次冷启动完成 ready 后，直接执行 full，不先运行 smoke 或前缀：

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite full \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --token-pool "$TOKEN_POOL_FILE" \
  --output "$RUN_ROOT/requests-e0-full" \
  --run-id "$RUN_ID-e0-full"
```

保存结果和日志后，再使用新 RUN_ID/RUN_ROOT **重新冷启动 E0 服务**，不要沿用
full 执行后的服务。在同一批服务进程上连续执行两次前缀，两次之间不重启、
不发其他推理请求；第一遍失败即停止第二遍，保留故障现场：

```bash
(
  set -e
  test -f "$TOKEN_POOL_FILE"
  sha256sum "$TOKEN_POOL_FILE" > "$RUN_ROOT/comparison-token-pool.sha256"
  for pass in 1 2; do
    "$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
      --suite full-to-c8-short \
      --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
      --token-pool "$TOKEN_POOL_FILE" \
      --output "$RUN_ROOT/requests-e0-prefix-$pass" \
      --run-id "$RUN_ID-e0-prefix-$pass"
  done
)
```

full 应有 92 项通过、`full_request_suite_passed=true`；每次前缀应有 11 项通过，
`request_checks_passed=true`，其中 c8-short 的 `passed_count=8`。每次结束应确认
Attention/Proxy 请求归零；保存两次前缀的相同服务 PID、runtime、实际配置、
四角色日志和逐条请求/响应，不能仅凭 shell 没有报错判断结果。

若 E0 的 full 和两次前缀均通过，说明成功不依赖 eager overlap=1，可进一步用
相同冻结版本复核 G0：按 C5 配置启动，五个 Graph 开关仍为 0，eager overlap
实际为 0，重新冷启动后直接执行 full。若 E0 失败，保留首次设备异常及失败阶段，
再用同一冻结版本将 E0 启动配置的 eager overlap 改为 1，冷启动复测相同序列，
排查 eager 串行执行路径。已有 E1 成功无需无条件重复。
以上结果用于缩小执行路径范围，尚不能确定具体错误的 Graph 参数、buffer 或 event。

### 6.5 逐项填写真实路径证据

证据必须来自在线请求时间窗，区分 capture/warmup 和业务执行。每条证据记录机器、角色/rank、日志文件、行号或 trace 时间戳、请求 ID。全局关键词出现一次不够。

| 项目 | 适用点 | 通过条件 |
|---|---|---|
| PD | C1–C5 | producer/consumer 建立 Mooncake session，KV transfer 完成，首 token 衔接正确；取消后 KV 资源释放 |
| AFD | C1–C5 | A0/A1→F0、A2/A3→F1 mapping 正确，在线 A2F/F2A 都有传输完成证据；跨机使用实际远程 IP |
| DSpark | C2/C4/C5 | 四个 Attention rank 加载完整 drafter，block size 与模型一致；proposal/target verify 生效；本轮 proposed/accepted 增量均大于 0 |
| Graph | C3/C5 | capture 完成、在线 target replay 命中；至少一个未捕获的真实 shape 回退 eager，结果正确且服务可恢复 |
| U2 | C3/C4/C5 | 四个 Attention rank 均有在线 stage 0 与 stage 1；FFN 对应协议完成；不能只保存配置 `U_BATCHES=2` |
| 停服 | 全部 | 两个 FFN rank 均收到 Attention shutdown，正常退出，业务端口释放，NPU 无残留 |

辅助检索命令（现场可能有不同日志措辞，不能把关键词命中直接写成通过）：

```bash
grep -nEi 'mooncake|session|transfer|first.token|kv' "$RUN_ROOT"/*.log \
  > "$RUN_ROOT/pd-evidence-candidates.txt" || true
grep -nEi 'stage=0|stage=1|stage 0|stage 1|replay|fallback|capture|draft|specul|verify' \
  "$RUN_ROOT"/*.log > "$RUN_ROOT/feature-evidence-candidates.txt" || true
grep -nEi 'Traceback|EngineCore.*fatal|out of memory|OOM|timeout|watchdog|507[0-9]{3}|ERR99999' \
  "$RUN_ROOT"/*.log > "$RUN_ROOT/error-candidates.txt" || true
```

DSpark metrics 使用 before/after counter 增量，逐 rank 核对，不使用进程累计值代替本轮业务生效。采集实际指标名、标签和 proposed/accepted 定义；acceptance rate 明确分子分母。
最新 C4 四 engine 的日志已有正的 drafted/accepted，但导出的两个 token counter 在 before/after 均为 0；记录为业务生效有证据、metrics 证据待核实，不能把日志区间计数直接填作 Prometheus counter 增量。详见[C4 结果第 2 节](DEEPSEEK_V4_AFD_C4_20261013_RESULT_ZH.md)。
若默认日志没有 PD/首 token/Graph/rank 证据，补充诊断采集或 trace 后再判定，保持 pending。
只保存 HTTP 200、capture 完成或开关值不能关闭功能门禁。

Graph fallback 不能预设“并发 32 必然触发”。先确认本轮实际 capture key，再选未捕获shape 发请求，保存 fallback 日志与正确结果；必要时单独运行诊断用 capture sizes 配置并明确标记。不能把诊断配置的结果替代正式 C5 配置或性能记录。

## 7. 【A5 手工执行】第 5 步：正常停服、采集和点位判定

先停止注入新请求，等待请求归零，再停 Proxy；standalone 跳过 Proxy。在各角色节点执行：

```bash
# Proxy 节点
kill -TERM "$(cat "$RUN_ROOT/proxy.pid")"
```

```bash
# Attention 节点：先停 Attention，向 FFN 发送 shutdown。
date -Iseconds > "$RUN_ROOT/stop-time.txt"
kill -TERM "$(cat "$RUN_ROOT/attention.pid")"
```

FFN 节点等待最多 60 秒，逐 rank 核对 shutdown，然后停止仍存活的 FFN launcher：

```bash
deadline=$((SECONDS + 60))
until [[ "$(grep -cF 'AFD NPU FFN received Attention shutdown payload' "$RUN_ROOT/ffn.log" || true)" -ge 2 ]]; do
  if ((SECONDS >= deadline)); then
    printf 'shutdown gate failed\n' > "$RUN_ROOT/shutdown-failed.txt"
    break
  fi
  sleep 1
done
kill -TERM "$(cat "$RUN_ROOT/ffn.pid")" 2>/dev/null || true
```

Prefill 节点最后停 Prefill。每个节点保存退出结果、进程表、端口及 NPU 状态，等待清理完成后才进入下一轮。`wait` 仅能在启动该子进程的原终端使用；换终端不把 wait 的 “不是子进程”错误当服务退出码。

```bash
# Prefill 节点
kill -TERM "$(cat "$RUN_ROOT/prefill.pid")"
# 每个节点：待本机所有本轮角色退出后采集。
npu-smi info > "$RUN_ROOT/npu-after.txt"
ps -eo pid,ppid,lstart,args > "$RUN_ROOT/processes-after.txt"
ss -ltnp > "$RUN_ROOT/ports-after.txt"
date -Iseconds > "$RUN_ROOT/cleanup-time.txt"
```

现场无 `ss` 时，用 Python socket bind 验证本轮本机 API/内部端口、KV 端口及派生范围、HCCL/AFD 端口均释放，并保存输出；不能仅因工具不存在跳过门禁。
timeout、OOM、业务期 HCCL/NPU 首错、错误输出或无法清理均阻断。
停服期 traceback/级联错误保留时间线及首错，不沿用旧环境“停服噪声”结论自动豁免。
若发生强制清理，本轮不能写成正常停服通过。

每轮写 `validation_summary.json`，建议字段如下。初始状态为 pending，完成证据核对后填入 true/false，并在 `evidence` 中给出定位；不得只把所有字段手工改成 true：

```json
{
  "run_id": "实际run ID",
  "topology": "single或cross",
  "case": "C1到C5",
  "cycle": 1,
  "status": "pending",
  "request_gate": null,
  "pd_gate": null,
  "afd_gate": null,
  "dspark_gate": "not_applicable或true/false",
  "graph_gate": "not_applicable或true/false",
  "u2_gate": "not_applicable或true/false",
  "shutdown_gate": null,
  "npu_cleanup_gate": null,
  "port_cleanup_gate": null,
  "golden_checked": false,
  "gsm8k_checked": false,
  "evidence": []
}
```

所有适用 gate 为 true 才能 `status=passed`。C1/C5 的两轮都 passed 后才关闭点位。
同机 C1–C5 全部关闭后，三机按同样流程从 C1 重新执行；不能跳到三机 C5。

## 8. 【A5 手工执行】第 6 步：GSM8K、端到端性能与 Decode 容量对照

### 8.1 M1 冻结精度协议，M5 执行

保存 `gsm8k_protocol.json`：数据集/task 配置及 hash、实际题目数/题目顺序、prompt/fewshot、seed、temperature/top_p/max_tokens/stop、模型 hashes、lm-eval 版本/提交、评分键及最低分。
同机与跨机使用同一协议；最低分不能测试后修改，未冻结分数不能宣布精度通过。
现有 NPU GSM8K pytest 会启动 1A1F standalone，不适合作为本期 Proxy+C5 精度入口。

C5 功能关闭后，以新目录重新启动 C5，使用已有 Proxy，在离线数据已到位的客户端执行。
下面变量必须来自冻结协议，`TASK_DIR` 内的 gsm8k task 指向本机离线数据，不访问在线数据集。
先用本机 `python -m lm_eval --help` 核对 CLI；不临时升级 lm-eval。

```bash
export TASK_DIR=/data/eval/offline-gsm8k-task
export GSM8K_LIMIT=1319
export GSM8K_FEWSHOT=5
export GSM8K_MAX_TOKENS=1024
export GSM8K_MIN_SCORE='填写M1冻结的最低分'
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"$PYTHON_BIN" -m lm_eval \
  --model local-completions \
  --model_args "model=dsv4-afd,base_url=$BASE_URL/v1/completions,max_tokens=$GSM8K_MAX_TOKENS,tokenized_requests=False,tokenizer=$MODEL_PATH" \
  --include_path "$TASK_DIR" --tasks gsm8k \
  --num_fewshot "$GSM8K_FEWSHOT" --batch_size 1 --limit "$GSM8K_LIMIT" \
  --gen_kwargs 'temperature=0,top_p=1' --seed 1024 \
  --output_path "$RUN_ROOT/gsm8k" --log_samples \
  2>&1 | tee "$RUN_ROOT/gsm8k.console.log"
```

1319/5/1024/seed 1024 是建议配置示例，正式值以 M1 协议为准。所有 defaults/stop/template 在 task/eval 配置中冻结；CLI 参数与配置不一致时先停止修正。
保留全部样本输出及分数；选择本次实际 results 文件，明确固定评分键`results.gsm8k.exact_match,strict-match`，与 `GSM8K_MIN_SCORE` 比较，保存比较结果。
实际完成样本数必须等于冻结题目数；数据缺失、样本错误、skip 或解析失败不能记为通过。
自然语言最终答案评分不等于模型输出的逐 token golden。

### 8.2 关闭诊断后重新冷启动性能 C5

只修改并记录诊断设置，保持同一已验收 C5 的模型、软件及拓扑：

```bash
export AFD_HCCL_STAGE_DIAGNOSTICS=0
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export VLLM_LOGGING_LEVEL=INFO
# 正常停服后使用新 run ID，重新执行 C5 启动与 ready；确认最终生效值。
```

客户端使用本机 vLLM 0.23 的 benchmark CLI，示例负载在 M1 冻结后使用：

```bash
"$VLLM_BIN" bench serve \
  --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
  --model dsv4-afd --tokenizer "$MODEL_PATH" \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 128 --max-concurrency 32 --request-rate inf --ignore-eos \
  --save-result --result-dir "$RUN_ROOT/performance" --result-filename benchmark.json \
  2>&1 | tee "$RUN_ROOT/performance.console.log"
```

固定 warmup、测量次数和数据长度；记录全部请求成功率、吞吐、TTFT/TPOT 的均值及分位数，不能只保留最快一轮。另采集所有角色/rank 的 NPU HBM 峰值及 DSpark metrics 增量；`npu-smi` 采样注明间隔和可能遗漏瞬时峰值，不能把当前 HBM 当精确峰值。
性能期间保留模型/开关生效证据，但不启用同步诊断或 DEBUG 热路径日志。
收益不足不阻断；请求错误、OOM、timeout 或无法清理仍阻断。

### 8.3 【A5 手工执行】真实 PD 端到端的非 AFD/AFD 对照

本节通过 Proxy 和真实 Prefill/KV 链路检查第 8.4 节主要组合的端到端收益及延迟，尤其用于 DSpark 的真实语义输入/KV、接受率和实际输出吞吐验证。纯 Decode 容量结果与本节分开报告。本节不改变 C1–C5 的功能验收顺序，也不表示原生 Decode 或两机组合已通过实机验证。先完成所测配置的功能、输出正确性、精度和正常停服检查，再用独立性能 RUN_ID 冷启动。主要场景的优先级不代表前置问题已关闭；当前文档保留的 C3/C5 问题未关闭前，性能状态仍为 pending。

#### 8.3.1 冻结对照组和资源口径

| 组名 | Decode 配置 | 要回答的问题 |
|---|---|---|
| B0 | 原生完整 Decode，PD on、eager、无 speculative config、无 DBO | 辅助基线，不作为唯一主基线 |
| B1 | C1：PD+AFD、eager/U1、DSpark off | 辅助归因，与 B0 配对 |
| N-G | 原生完整 Decode，PD on、target Decode Graph、DSpark off | Graph 主场景的原生基线 |
| B2 | C3：PD+AFD+Graph/U2、DSpark off | 主要场景一，与 N-G 配对 |
| N-GS | 原生完整 Decode，PD on、target Decode Graph+DSpark，draft eager | DSpark 主场景的原生基线 |
| B3 | C5：PD+AFD+DSpark+Graph/U2，draft eager | 主要场景二，与 N-GS 配对 |

功能门禁通过后，主要性能测量优先完成 N-G/B2、N-GS/B3；B0/B1 只选代表点作辅助对照，不要求先跑完整 eager 容量矩阵。B2 对 B1、B3 对 B2 用于解释优化增益，不能替代原生基线。B3 若只对比 B0，结果应标为“AFD+Graph/U2+DSpark 相对 eager 原生 Decode 的组合收益”。原生 Graph/DSpark 或微批能力不支持时，保留缺项和配置差异，不把 eager 结果冒充功能匹配的主基线，也不将仅有 AFD 组的结果写成相对原生收益。

在运行前保存 `performance_protocol.md`，两组共用并冻结以下内容：

- 软件提交、镜像/CANN/HCCL、模型/量化/tokenizer hashes、seed、解码参数、prefix cache 策略。B0/B1、N-G/B2 各配对使用同一 Flash checkpoint；N-GS/B3 使用同一 DSpark checkpoint 和 draft 配置。另比较 DSpark 开关的增益时，核对 target 权重 hashes 是否一致，不将不同权重的效果单独归因于 DSpark。
- Prefill DP/TP/卡数、KV connector、Proxy、输入集及顺序、输入/输出长度、并发、请求速率、warmup、测量请求数和重复次数。
- 每个角色的主机/物理卡、原生 Decode 的 DP/TP/EP 与最终通信后端、最大活跃序列数、batch token 上限、HBM/KV 容量及 Graph capture sizes。原生/AFD 对配置的含义可能不同，保留实际值及差异。
- 预先约定的 TTFT/TPOT 延迟目标、成功率要求、统计方式，以及“占用卡数”还是“预留整机卡数”的成本口径。

**A4F2 的 Decode 使用 6 卡，不能与原生 D4 的 4 卡吞吐直接相除后称为等资源 AFD 收益。** 同资源对照应先验证原生 DP/TP/EP 的合法组合、权重/专家分片和显存，不能为凑 6 卡直接假定 DP6/EP6 可用。若只有 D4 能运行，则两机 P2+D4 使用 6 卡、P2+A4F2 使用 8 卡；分别报告总吞吐、实际卡数、每卡吞吐和预留整机成本，并将结论限定为这两种部署方案。每卡归一化也不等同于严格的等资源因果对照。

#### 8.3.2 准备非 AFD 原生 Decode 基线 B0/N-G/N-GS

本目录没有独立的 DeepSeek-V4 非 AFD launcher。`afd_attention.sh`/`afd_ffn.sh` 固定传入 `additional_config["afd"]`，不能通过设置 `ENABLE_AFD=0` 或仅关闭 DSpark 得到 B0。B0 不启动独立 FFN，原生 Decode 必须同时执行完整 Attention 和 FFN。

下面给出 **P2+D4 候选基线**，不是等卡数基线。主测先按第 4.1 节在单机运行 P2/Proxy 与原生 D4：P 使用 NPU 0、1，D 使用 2–5，6、7 空闲；各角色 IP 均为本机可达 IP。跨机补充时可按第 4.2.1 节在机器 1 运行 P2/Proxy、机器 2 运行 D4（NPU 0–3）。两种拓扑的 `ATTENTION_HOST_IP/ATTENTION_API_PORT` 均指向原生 Decode。停止上一组全部服务并确认清理后，使用包含拓扑及实际组名（B0、N-G 或 N-GS）的新 RUN_ID；加载第 3.1 节环境并初始化本机 RUN_ROOT。

在各角色启动终端先选择同一个 `PERF_GROUP`，设置以下配置，再按第 4.4 节保存运行清单（这三个原生 CASE 为本节扩展标签，不执行第 4.3 节的 AFD CASE 分支）：

```bash
export PERF_GROUP=N-G       # 首测 N-G；另一主基线用 N-GS；B0 仅为 eager 辅助
export CASE="$PERF_GROUP"
export MODEL_PATH="$FLASH_MODEL_PATH" EXECUTION_MODE=full-decode-only
export U_BATCHES=1          # 原生首次候选采用 U1；不调用 AFD configure_execution
export ENABLE_PD=1 ENABLE_DSPARK=0 ENABLE_MTP=0
case "$PERF_GROUP" in
  B0) export EXECUTION_MODE=eager ;;
  N-G) ;;
  N-GS) export MODEL_PATH="$DSPARK_MODEL_PATH" ENABLE_DSPARK=1 ;;
  *) printf 'Unknown PERF_GROUP: %s\n' "$PERF_GROUP" >&2; exit 2 ;;
esac
export AFD_HCCL_STAGE_DIAGNOSTICS=0
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export VLLM_LOGGING_LEVEL=INFO
```

先按第 6.1 节启动 Prefill，等待 ready。原生 Decode 的启动终端在已加载相同组配置且 RUN_ROOT 已存在后执行以下候选命令；单机时两个角色在同一主机，跨机时在各自节点执行。不传 AFD additional config 或 DBO 参数，也不套用 AFD 的临时 all2all backend；N-GS 单独添加 DSpark speculative config。实际通信后端由固定的上游平台选择，须记录启动后的最终值。

```bash
set -euo pipefail
test -d "$RUN_ROOT"
test "$PREFILL_DP_SIZE" = 2
test "$ATTENTION_RANKS" = 4
test "$HCCL_IF_IP" = "$ATTENTION_HOST_IP"
export BASELINE_DEVICES="$ATTENTION_DEVICES"  # 单机 2,3,4,5；两机 0,1,2,3
export ASCEND_RT_VISIBLE_DEVICES="$BASELINE_DEVICES"
export VLLM_HOST_IP="$HCCL_IF_IP"
export HCCL_IF_BASE_PORT="$ATTENTION_HCCL_IF_BASE_PORT"
export TP_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_V1=1
# 保留与 Prefill 相同的插件集合；AFD 按 additional_config 是否含 afd 激活。
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
source "$RECIPE/common.sh"
NATIVE_EXECUTION_ARGS=(--enforce-eager)
NATIVE_DSPARK_ARGS=()
if [[ "$PERF_GROUP" == N-G || "$PERF_GROUP" == N-GS ]]; then
  read -r -a native_capture_sizes <<< "$CUDAGRAPH_CAPTURE_SIZES"
  NATIVE_EXECUTION_ARGS=(
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --cudagraph-capture-sizes "${native_capture_sizes[@]}"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  )
fi
if [[ "$PERF_GROUP" == N-GS ]]; then
  DSPARK_BLOCK_SIZE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dspark_block_size"])' "$MODEL_PATH/config.json")"
  [[ "$DSPARK_BLOCK_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || afd_die "N-GS requires a DSpark checkpoint with dspark_block_size"
  DSPARK_CONFIG="$(printf '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"greedy","enforce_eager":true}' "$DSPARK_BLOCK_SIZE")"
  NATIVE_DSPARK_ARGS=(--speculative-config "$DSPARK_CONFIG")
fi
# 此 helper 的 decode.dp_size=4 对应本例原生 D4，不表示启用 AFD。
KV_CONFIG="$(build_a5_pd_kv_config kv_consumer \
  "$DECODE_ENGINE_ID" "$DECODE_KV_PORT" "$BASELINE_DEVICES")"
printf '%s\n' "$KV_CONFIG" > "$RUN_ROOT/baseline-kv-config.json"
nohup "$VLLM_BIN" serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$ATTENTION_API_PORT" \
  --api-server-count 1 --served-model-name dsv4-afd \
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$ATTENTION_MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --seed 1024 \
  --tokenizer-mode deepseek_v4 --no-enable-prefix-caching \
  --safetensors-load-strategy prefetch --block-size 32 --kv-cache-dtype auto \
  --no-async-scheduling --kv-transfer-config "$KV_CONFIG" \
  "${NATIVE_EXECUTION_ARGS[@]}" "${NATIVE_DSPARK_ARGS[@]}" \
  > "$RUN_ROOT/decode.log" 2>&1 &
echo $! > "$RUN_ROOT/decode.pid"
```

另存 `topology.txt`，明确主机数、P2/Proxy 和 D4 的实际卡位、FFN 卡数为 0；共用环境里遗留的 `FFN_DEVICES` 不表示原生组占用了这些卡。单机 P/D 卡位必须互斥；不要沿用两机 D4 的 0–3 与本机 P2 的 0、1 重叠。若候选 D4 启动、KV 接收或正确性失败，保存原始错误并停止性能比较，不能将启动成功写成可用基线。另选合法的原生并行配置时，同时调整两端 KV JSON 的 decode DP/TP、卡表及运行清单；本例 helper 固定 D4/TP1，其他布局需单独生成匹配配置，不能只改 CLI。

按第 6.2 节检查 Prefill 和原生 Decode health、四个 Decode rank ready，然后启动 Proxy；原生组跳过 FFN loop/AFD mapping 检查。经 Proxy 发送 EOS、smoke、full 并检查真实 KV 接收、输出正确、请求归零；使用冻结精度协议验证质量。日志/实际 argv 应显示原生模型和 worker、无 AFD role；B0 为 eager/无 speculative config，N-G 为 Graph/无 speculative config，N-GS 为 target Graph/DSpark draft eager。保留 `dsv4-afd` 服务名仅为两组客户端一致，不代表启用 AFD。

原生组正常停服顺序为停止新请求、请求归零、停 Proxy、`kill -TERM "$(cat "$RUN_ROOT/decode.pid")"`、最后停 Prefill，检查所有原生子进程、NPU 和端口释放。没有 FFN shutdown 门禁。B1 则按 C1 的启动/停服流程，使用独立 RUN_ID，仍保留 P2 和相同模型、请求配置。每组按下一节关闭诊断并重新冷启动性能服务，不复用功能测试后的进程。

上面按 `PERF_GROUP` 生成三套候选参数，尚未经过 A5 实机验收。N-G/N-GS 的 capture sizes 必须覆盖实际负载；分别验收原生 Graph replay、DSpark KV/target verify 和输出质量，AFD 专用 DSpark 补丁存在不代表原生 N-GS 已兼容。原生首次候选不启用 DBO/U2，而 AFD 使用 U2；该差异保留时，只能声明两种部署方案的组合收益。若要进一步隔离 AFD 本身的收益，先确认原生 DBO/U2 支持情况，再建立功能匹配并通过验收的新配置；不在计时过程中临时加参数。原生主场景尚未验收时标 pending，不使用 B0 冒充。

#### 8.3.3 关闭诊断并运行配对压测

所有服务启动终端设置第 8.2 节三个诊断变量，B1 在第 4.3 节 CASE 配置**之后**覆盖 `AFD_HCCL_STAGE_DIAGNOSTICS=0`，避免被功能配置重置。正常停服后以新 RUN_ID 重启、确认 ready，并归档实际 argv、有效开关和拓扑。先验证路径证据，再关闭诊断计时，不在计时窗口开启同步诊断、profiler 或 DEBUG 热路径日志。

压测客户端可固定在 Proxy 主机；两组使用同一位置并确认客户端 CPU/网络未饱和。`BASE_URL` 始终指向 Proxy，不将 B0 的端到端请求与 B1 直连 Attention 的请求混比。直连 Attention 不等于使用相同外部 KV 的纯 Decode 测试；dummy-KV 压测也不替代本节真实 PD 的结果。

下面的短输入矩阵和 random CLI 是入口预检、固定负载及延迟辅助模板，不是第 8.4 节长上下文主要矩阵。初跑可从 128/128、并发 8 做小规模预检；主矩阵改用第 8.4 节的输入长度、BS/客户端容量和代表业务的真实请求集，尤其不使用随机文本的接受率代表 DSpark 业务收益。

| 输入 token / 输出 token | 观察重点 | 最大并发 |
|---|---|---|
| 128 / 128 | 与第 8.2 节负载衔接 | 1、8、32 |
| 128 / 1024 | Decode 吞吐、TPOT | 1、8、32 |
| 2048 / 128 | Prefill、KV 传输、TTFT | 1、8、32 |

每组每个负载/并发点先 warmup 32 请求，再测 3 轮、每轮 1024 请求；这些是起始协议示例，正式数量在运行前冻结，并确保测量足够长以覆盖稳定服务。两组完全相同，不取最快一轮。`MAX_MODEL_LEN=4096` 能覆盖表中长度，但仍需核验实际输入/输出 token 数，不能把请求的长度参数当成实际完成长度。

客户端先核对固定 vLLM 0.23 的参数，保存帮助；不升级 CLI：

```bash
"$VLLM_BIN" bench serve --help > "$RUN_ROOT/bench-serve-help.txt"
```

确认支持下面的 seed 和分位数参数后，逐负载/并发点执行以下整段。替换 `PERF_GROUP`、`ISL/OSL/CONCURRENCY`；`RUN_ROOT` 使用本组独立性能目录。random 输入用于可复现压力测试，不能替代质量评估；固定 CLI/tokenizer/seed，正式业务负载另冻结同一份输入文件及 hash。

```bash
set -euo pipefail
export PERF_GROUP=N-G                # N-G/B2、N-GS/B3 配对；本段 random 负载仅作辅助
export ISL=128 OSL=1024 CONCURRENCY=32
export BASE_URL="http://$PROXY_HOST_IP:$PROXY_PORT"
export ATTENTION_URL="http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT"
export PERF_DIR="$RUN_ROOT/performance/$PERF_GROUP-i$ISL-o$OSL-c$CONCURRENCY"
mkdir -p "$RUN_ROOT/performance"
mkdir "$PERF_DIR"                    # 拒绝覆盖已有点位

run_perf_batch() {
  local phase="$1" count="$2" seed="$3"
  curl --noproxy '*' -fsS --max-time 10 "$ATTENTION_URL/metrics" \
    > "$PERF_DIR/$phase.metrics-before.txt"
  "$VLLM_BIN" bench serve \
    --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
    --model dsv4-afd --tokenizer "$MODEL_PATH" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --seed "$seed" --num-prompts "$count" --max-concurrency "$CONCURRENCY" \
    --request-rate inf --ignore-eos \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
    --save-result --result-dir "$PERF_DIR" --result-filename "$phase.json" \
    2>&1 | tee "$PERF_DIR/$phase.console.log"
  curl --noproxy '*' -fsS --max-time 10 "$ATTENTION_URL/metrics" \
    > "$PERF_DIR/$phase.metrics-after.txt"
  # 固定版本结果应含 completed；字段不同先核对本机格式，不跳过成功数检查。
  "$PYTHON_BIN" - "$PERF_DIR/$phase.json" "$count" \
    "$BASE_URL" "$ATTENTION_URL" "$PERF_DIR/$phase.drain.json" <<'PY'
import json
import re
import sys
import time
from pathlib import Path

import httpx

with open(sys.argv[1], encoding="utf-8") as file:
    result = json.load(file)
assert result["completed"] == int(sys.argv[2]), result
deadline = time.monotonic() + 60
with httpx.Client(trust_env=False, timeout=10) as client:
    while True:
        response = client.get(f"{sys.argv[3]}/healthcheck")
        response.raise_for_status()
        proxy = response.json()
        response = client.get(f"{sys.argv[4]}/metrics")
        response.raise_for_status()
        gauges = {}
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            values = re.findall(
                rf"^{re.escape(name)}(?:\{{[^\n]*\}})? ([^\s]+)",
                response.text,
                re.MULTILINE,
            )
            assert values, f"missing metric: {name}"
            gauges[name] = [float(value) for value in values]
        snapshot = {"proxy": proxy, "decode_gauges": gauges}
        Path(sys.argv[5]).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        if proxy["status"] == "ok" and proxy["request_num"] == 0 and all(
            value == 0 for values in gauges.values() for value in values
        ):
            break
        assert time.monotonic() < deadline, snapshot
        time.sleep(1)
PY
}

run_perf_batch warmup 32 1023
for repeat in 1 2 3; do
  run_perf_batch "measure-$repeat" 1024 1024
done
```

本段在每次 batch 结束后最多等待 60 秒，检查 Proxy request_num 和 Decode 的 running/waiting 全部归零，保存 `*.drain.json`；缺指标、请求未归零或成功数不足会停止后续 batch。若出现超时/OOM/错误输出，按第 9 节保留现场。CLI 退出 0 不代表所有请求成功，另核对每轮结果的失败数、实际输入/输出总 token。脚本中的 completed 检查只做初筛，不代替这些检查。

上述 `request-rate inf` 测饱和吞吐。完成后另建带实际 RPS 的结果目录，将函数内 `--request-rate inf` 改成冻结的数值，对两组施加相同到达速率，逐级提高 RPS；若最大并发上限先限制发压，应明确记录。按预先约定的 TTFT/TPOT 分位数和成功率目标，比较两组最大可承载流量。饱和吞吐与相同延迟目标下的服务容量分别报告。

HBM 在每个服务节点单独采样并保存时间戳；在 warmup 前启动，最后一轮结束后停止。下面示例每 5 秒采样，采样间隔也纳入协议：

```bash
(
  while :; do
    date -Iseconds
    npu-smi info
    sleep 5
  done
) > "$RUN_ROOT/npu-performance-samples.txt" 2>&1 &
echo $! > "$RUN_ROOT/npu-sampler.pid"
# 全部测量结束后，在本机执行：
# kill -TERM "$(cat "$RUN_ROOT/npu-sampler.pid")"
```

这是设备 HBM 的采样最大值，可能遗漏瞬时峰值；同机其他进程也会影响数值。以日志/进程表确认卡的归属，分别报告 Prefill、Attention/原生 Decode、FFN，勿把当前 HBM 或各卡不同时间的峰值之和称为精确同时峰值。DSpark 组按每轮 metrics-before/after 计算 proposed/accepted 增量，并写明 acceptance rate 分子分母；counter 不可用时标为缺失，不填写 0 或用进程累计值冒充本轮值。

#### 8.3.4 计算收益并归档结果

每个相同输入/输出长度、并发和请求速率单独比较，保留 3 轮原始值，报告中位数及最小/最大值；不混合不同负载，也不将 warmup 纳入统计。

| 指标 | 计算或报告方式 |
|---|---|
| 输出吞吐收益 | `(AFD 输出 tokens/s ÷ 基线输出 tokens/s - 1) × 100%`；不混用输入+输出总吞吐 |
| TPOT 降低比例 | `(1 - AFD TPOT ÷ 基线 TPOT) × 100%`；分别比较均值、P95、P99，负值表示变慢 |
| TTFT | 分别报告均值/P95/P99，判断 Decode 收益是否伴随首 token 延迟恶化 |
| 单卡输出吞吐 | 输出 tokens/s ÷ 卡数；端到端分母包含 Prefill 与所有 Decode/A/F 卡，另列预留整机卡数 |
| 服务容量 | 相同 TTFT/TPOT 延迟目标和成功率下的最大可承载 RPS；记录实际完成 RPS |
| 运行质量与资源 | 成功/失败数、错误类型、实际 token 数、各角色 HBM 采样最大值、DSpark acceptance rate（适用时） |

由于本节请求经过 Proxy，不将端到端 tokens/s 除以 Decode 卡数后标为“纯 Decode 每卡吞吐”。若另报按 Decode 卡数归一化的值，必须同时给出 Prefill 配置及其是否构成瓶颈。

汇总表至少包含：`group/topology/ISL/OSL/concurrency/request_rate/repeat/P卡数/D或A+F卡数/预留卡数/成功数/失败数/实际输入token/实际输出token/输出tokens每秒/TTFT均值及P95/P99/TPOT均值及P95/P99/HBM/acceptance rate/是否满足延迟目标`。连同协议、模型/环境 hash、实际启动参数、两节点或三节点日志、metrics、所有 JSON/console 和正常清理证据归档。

只有两组质量合格、请求完整成功、资源及功能差异已说明，才计算并声明对应范围的收益。收益为负也如实报告；Prefill 饱和、客户端发压不足、网络竞争或卡数不同，应列为解释条件。第 8.2 节“不设置收益比例门槛”保持不变，本节增加收益测量方法，不新增交付收益阈值。

### 8.4 参考 V3.2 recipe：Graph/U2、DSpark 的主要吞吐验证设计

参考来源：[用户指定的 V3.2 CAMP2p recipe](https://github.com/wenhow/afd-plugin/tree/feat/dsv4-afd-phase1-delivery/recipe/npu/CAMP2pAFDConnector/deepseek_v3_2)，本仓库对应 [README](../../recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md) 及同目录三个 launcher。本节根据本地源码核对方法，不将 V3.2 的结果或兼容性当作 V4 已验证结果。

参考实验的核心是：原生 EP 与 A/F 分离各自选择可运行的 per-rank batch size，在相同负载分布下测 Decode 饱和输出吞吐，除以全部部署的物理 die 数。它不是固定客户端并发 32 的在线延迟测试。V4 沿用这个容量比较思路，但继续使用本期 A5、冻结的 vLLM 0.23/vLLM-Ascend、V4 checkpoint 和 `P2pHcclAFDConnector`。

**主要组合为 Graph/U2，以及 Graph/U2+DSpark；eager 只作辅助对照。** 每个主要组合都覆盖长上下文与目标 BS24，观察总输出吞吐、单位 NPU 吞吐及 TPOT；保留 HBM、Graph replay、U2 stage、DSpark acceptance rate 解释瓶颈。功能门禁仍按 C1–C5 的原顺序完成，性能矩阵优先级不豁免 C3/C5 的已知问题。

| 主要组合 | 原生基线 | AFD P2P 组 | 首要比较 |
|---|---|---|---|
| Graph，DSpark off | N-G：原生 target Decode Graph，DSpark off | B2：A4F2 target FULL_DECODE_ONLY/U2，DSpark off | 优化运行配置下的 AFD 总吞吐及每卡效率 |
| Graph+DSpark | N-GS：原生 target Decode Graph+DSpark，draft eager | B3：A4F2 target FULL_DECODE_ONLY/U2+DSpark，draft eager | 主流完整组合的吞吐及每卡效率，同时比较接受率 |

两侧使用同一模型/输入/KV 来源/生成预算，并记录原生是否支持与 AFD 相同的微批能力。若原生不支持 U2 或两侧 Graph 能力不同，主结果标为对应完整配置的部署收益，不能声称所有优化都已单变量对齐。C4 eager/U2+DSpark 与 C5 的配对可在代表点解释 Graph 贡献，不要求把所有 eager 模式扩成完整主矩阵。

#### 8.4.1 参考方法到 V4 的映射与前置检查

| 参考 recipe 的做法 | V4 的设计 |
|---|---|
| EP64 与 48A16F/64A16F 对比 | 原生 D4、可选 D8 与当前固定 A4F2 对比；不直接缩放成尚未支持的 A/F rank 数 |
| 分别选择原生/AFD 的 BS | 两侧独立扫描 BS、冻结各自满足相同资源/运行质量约束的最佳点；另保留同 BS 对照 |
| `请求数 = DP × BS × 6` | 原生用 D ranks，AFD 只用 A ranks；六波请求负责持续供给，不表示每 rank BS 或活跃并发扩大六倍 |
| `FFN BS = A ranks × A BS / F ranks` | A4F2 的 FFN 容量起点为 `2 × A BS`；U2/Graph 实际 stage 和 buffer 容量另核验 |
| 16K/32K 输入、512–1536 输出 | 主矩阵实际输入 8K/16K/32K/64K，目标每 A rank BS24；输出先固定 1024，再补 512–1536 分布。现有 4096 配置只作预检 |
| AISBench synthetic 饱和发压 | 以 AISBench 做主测，固定工具提交、数据、输出预算和随机种子，核对实际发出的请求及完成 token |
| 输出吞吐除以全部 die | V4 先按实际参与计算的 NPU 设备数归一化；报告 die 时先核对 A5 设备到物理 die 的对应关系 |

**主要组合优先使用真实语义输入及匹配的真实 KV。** 现有可执行入口是第 8.3 节真实 PD：其主结果计入 Prefill 成本，并保留 TTFT/TPOT。若单独测纯 Decode，须先验收将正确的 Prefill/KV 状态准备在计时窗口外的独立流程；当前文档不宣称已有此 launcher。参考 recipe 的合成 KV 可用于隔离性能的补充实验，不能替代 DSpark 的业务接受率和完整方案主结论。本次只补验证设计，不修改 KV connector、模型路由或启动脚本。

合成 Decode 实验另有以下适配前置项；这些前置项不代替真实 PD 的功能验收：

1. **合成 KV 路径未验收。** 当前 [decode_bench.py](../../tools/benchmarks/decode_bench.py) 的 `AFDDecodeBenchConnector.start_load_kv()` 直接返回，没有调用 worker 的 `start_fill_kv()`。因此仅传 `fill_mean=0.015/fill_std=0` 不能证明设备 KV 已填充。V4 还存在多 KV group、滑窗/indexer 等状态，不能仅凭类声明 `SupportsHMA` 判为兼容。先在固定 0.23 栈审查接口、分组/块步长及状态初始化；验证每个实际参与读取的缓存状态都可复现、值有限、容量合法，并确认没有真实 Prefill 计算混入。必要适配另做最小补丁和架构评审，验收前该项保持 pending。
2. **自然路由是 V4 首轮口径。** 参考 V3.2 使用强制 expert 均衡且改变输出；本地 `force_load_balance.py` 明确只 patch `AscendW8A8DynamicFusedMoEMethod`，不能推断它对 V4 本期的量化路径生效。V4 首轮不设置 `enable_force_load_balance` 或 `force_load_balance_topn_per_rank`，保留完整专家和真实路由。若另做强制均衡实验，须两侧均证明生效、明确专家范围，单独标为合成实验；不用于 GSM8K 或生产收益，也不声称模拟了更大逻辑规模。
3. **单独准备 benchmark launcher。** 当前 V4 Attention 脚本的 `ENABLE_PD=1` 固定构建 Mooncake，`ENABLE_PD=0` 不注入任何 KV connector；没有直接选择合成 connector 的开关。需为候选测试准备独立 launcher，保留模型/worker/A-F 路径，仅调整下表性能入口参数，并归档完整 argv；不能把 `ENABLE_PD=0` 当作已挂载合成 KV。共享 `tools.benchmarks` 不在 wheel 内，服务所有子进程都需能从本次仓库导入。
4. **软件/调度配置继续按 V4 冻结。** 不复制 V3.2 的 A3 镜像、W8A8 参数、`--async-scheduling`、16-die 节点参数或强制 `--ubatch-size 2`。当前 V4 recipe 使用 `--no-async-scheduling` 并由 AFD 的 U1/U2 配置管理微批，逐 rank 核验最终值。

合成实验使用初始化过但非真实 Prefill 得到的 KV，输出语义不作为模型质量结论。实际输出有 NaN/Inf、无效 token、错误终止或设备错误时仍不能作为有效性能样本。真实质量及远程 KV 正确性继续按第 6–8.3 节验证。

**DSpark 主结果不能依赖任意合成 KV。** 接受率由 target/draft 对同一真实上下文的预测一致性决定；非真实或未初始化的历史状态可能改变接受率和吞吐。DSpark 主组冻结代表业务的自然语言/代码等输入、chat template、真实历史 KV，并保证 draft 所需本地缓存状态完整。合成状态下即使 proposed/accepted 大于 0，也只记受控实验；真实 PD 主组按本轮 counter 增量、逐 rank draft/verify 和实际接受的输出 token 计算，不将草稿提议量当吞吐。

#### 8.4.2 两机候选拓扑、BS 扫描与六波负载

下面的设备分母表对应独立纯 Decode 容量实验：不启动计时窗口内的 Prefill 或 PD Proxy，客户端直连原生 Decode/Attention API，KV 来源按第 8.4.1 节单独验收。真实 PD 的主要组合沿用第 4.2.1/8.3 节两机 P2+D4/P2+A4F2，客户端经 Proxy，并把 Prefill 计入资源分母。两类结果分开记录；两台主机保留相同镜像、模型和网络条件，同一时刻只运行一组。

| 实验拓扑 | 机器 1 | 机器 2 | 归一化设备数 | 状态 |
|---|---|---|---:|---|
| D4 | 空闲 | 原生 DP4/TP1/EP4，设备 0–3 | 4 | 候选原生基线，先完成可运行性检查 |
| A4F2 | FFN，设备 6–7 | Attention，设备 0–3 | 6 | 固定角色数；合成 KV 和性能 launcher 待适配验收 |
| D8（可选） | 空闲 | 原生 DP8/TP1/EP8，设备 0–7 | 8 | 补充资源效率基线，需单独验证专家/显存及启动参数 |

当前没有已验收的原生 6 卡对照，因此本设计比较单位设备效率，不将 D4 或 D8 与 A4F2 称为严格等卡数实验。另列预留整机数/设备数：原生一机、AFD 两机的部署成本可能不同，不能只列忙碌设备的效率。

两个主要组合都运行以下输入长度矩阵。K 按 1024 token 计；这里的长度是服务端 tokenizer/template 处理后的实际输入长度，不是单独设置模型上限。

| 实际输入长度 | 目标每 rank 原生/A BS | 输出预算 | 目的 |
|---|---:|---:|---|
| 8K（8192） | 24 | 先固定 1024 | 较短上下文参照 |
| 16K（16384） | 24 | 先固定 1024 | 中等上下文吞吐 |
| 32K（32768） | 24 | 先固定 1024 | 中长上下文吞吐 |
| 64K（65536） | 先验证 8、16，再以 24 为目标 | 先固定 1024 | 长上下文容量边界与重点性能点 |

链路稳定后，两种主要组合分别补 512–1536 token 均匀输出预算，与参考 recipe 方法衔接。每个输入长度/功能组合下，原生和 AFD 可独立扫描 per-rank BS `8、16、24、32` 找最佳合法点；BS4 仅用于必要的小规模预检。BS24 是主要目标，不是已经支持的保证。相同 BS 的各组保留一份对照，容量结论另比较各自最佳合法点；基线 BS24 不能运行时，如实记录其可运行容量和差异，不强行凑数。

长上下文性能使用独立配置文件/RUN_ID，不能直接沿用第 6.1 节功能用例固定 `MAX_MODEL_LEN=4096` 的检查和启动流程。实际输入 65536、输出预算 1024 时，上限至少覆盖 66560 token；输出预算最大 1536 时至少覆盖 67072 token，实际模板 token 也计入输入。先确认模型/算子支持，再按新长度重新验证 target 与 draft 的 KV/工作区/Graph 内存、块表和设备容量，冻结新的性能配置，不把它写成本期 4096 功能配置已验证。`MAX_NUM_SEQS=24` 只是调度上限，主结果需证明实际每 A rank 接近 24 条活跃序列，且无抢占/重计算、OOM/timeout/残留，并满足本轮预设的延迟约束。

```text
D = 原生 Decode ranks；A = Attention ranks；F = FFN ranks
B_D = 原生每 rank BS；B_A = Attention 每 rank BS；W = 6
原生配置活跃容量 C_D = D × B_D
AFD 配置活跃容量 C_A = A × B_A
AFD FFN 每 rank 序列容量起点 B_F = A × B_A / F = 2 × B_A
原生总请求数 N_D = C_D × W
AFD 总请求数 N_A = C_A × W
```

| 拓扑 | 每 rank 原生/A BS | FFN 容量起点 | 配置活跃容量 | 六波请求数 |
|---|---:|---:|---:|---:|
| D4 | 8 / 16 / 24 / 32 | 不适用 | 32 / 64 / 96 / 128 | 192 / 384 / 576 / 768 |
| A4F2 | 8 / 16 / 24 / 32 | 16 / 32 / 48 / 64 | 32 / 64 / 96 / 128 | 192 / 384 / 576 / 768 |
| D8（可选） | 8 / 16 / 24 / 32 | 不适用 | 64 / 128 / 192 / 256 | 384 / 768 / 1152 / 1536 |

以 A4F2/BS24 为例，96 是配置同时处理容量，576 是正式提交请求总量；请求持续补充，不要求六批串行完成，也不是同时运行 576 条。六波不包含独立 warmup。DSpark 开启后仍按序列容量计算请求数，不能再乘 speculative token 数。客户端应有足够在途请求覆盖这些容量，不能沿用第 8.3 节固定并发 32；以实际 running/waiting、吞吐曲线和 trace 判断是否压满。若六波仍不足以形成稳定窗口，所有配对组统一提高 W 并冻结后重跑，不只延长某一组。

性能 launcher 的参数设计如下；每个 BS 点独立冷启动，保存 argv 和最终有效值：

| 参数 | 原生 D4/D8 | A4F2 Attention | A4F2 FFN |
|---|---|---|---|
| AFD additional config | 无 `afd` 键 | 原 role/connector/rank 配置 | 原 role/connector/rank 配置 |
| KV connector | 经前置检查验收的合成 Decode connector | 与原生同一 KV 来源/初始化方法 | 无 KV connector |
| `--max-num-seqs` | `B_D` | `B_A` | 至少覆盖 `B_F`，按实际 worker buffer 契约核验 |
| Decode token budget | 覆盖本组实际 target verify/draft 峰值 | 覆盖本组实际 target verify/draft 峰值 | 覆盖合并后的 target verify/draft 峰值及 padding；不能只按 `B_F` 算 |
| `--max-model-len` | 覆盖实际输入+最大输出预算并完成该长度验收 | 与配对原生相同上限 | 与 Attention 配套 |
| DSpark | N-G 关闭；N-GS 开启，draft eager | B2 关闭；B3 开启，draft eager | 启动开关始终为 0，执行 Attention 发来的配套 FFN 工作；不另起 drafter |
| async scheduling / prefix cache | 关闭 / 关闭 | 关闭 / 关闭 | async scheduling 关闭 |
| 主要模式 | 经验证的原生 target Decode Graph；记录原生微批能力 | target FULL_DECODE_ONLY/U2 | 与 Attention 配套的 Graph/U2 |
| 辅助模式 | eager | eager/U1，或代表点 eager/U2 | 与 Attention 配套 |

现有 V4 脚本中 Attention/FFN 都读取 `MAX_NUM_SEQS`，所以独立性能 launcher 或角色启动前缀应分别提供 `B_A/B_F`，不能全局设一个 BS 后假定 FFN 已按比例扩容。BS24 时 B_F=48 是无 draft 条件下的容量起点；DSpark 的一次 target verify/draft 可能包含每序列多个 token，需要根据 checkpoint 的 block size、实际 verify/draft 形状、U2 分段和 padding 重新核验 batch token budget、HCCL buffer 与 Graph capture key，不能把 48 当作所有阶段的 token 上限。

Graph 的实际 capture key 可能对应 U2 stage 的形状，不能把“capture size=24”直接当成 BS24/DSpark 已覆盖。用在线请求证明四个 A rank 的 target replay、stage 0/1 和两个 F rank 的协议/计算完成；分别记录实际 replay/fallback 比例，区分 capture/warmup 与业务窗口。draft eager 是本期预期配置，不能据此认定整个 Graph 场景回退。性能窗口只保留必要统计，详细 trace 和同步诊断另轮采集。

主要组合均覆盖上述输入长度和 BS24 目标，eager 辅助只选代表点。当前 rank 数固定 A4F2，不能用本轮结果声称已找到最优 A:F 比例；A/F 比例扫描需要另行支持并验收其他拓扑。

#### 8.4.3 AISBench 负载与执行步骤

在协议中冻结 AISBench 的提交/版本、生成器及原始请求文件 hash。配置字段以该版本实际源码为准；可参考 [AISBench 模型配置](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) 与 [synthetic 配置](https://github.com/AISBench/benchmark/blob/master/ais_bench/datasets/synthetic/synthetic_config.py)。本节不要求临时安装或升级工具。

主矩阵先使用冻结的真实输入母集，经 AISBench 或已验收的同类客户端发送，保留输入语义、长度、预算和每请求结果。第 8.3 节提供 PD 启动与采集方法，客户端并发按容量调整，不能继续固定 32。工具具体数据集配置以现场冻结版本支持的真实数据入口为准；未配置好该入口时，DSpark 主组保持 pending，不用下面的 synthetic 配置冒充。

下面仅示范参考 recipe 的 **synthetic 辅助压力配置**：输入 16K、D4 或 A4F2 的 BS24，故 `REQUEST_COUNT=4×24×6=576`；每个 BS/拓扑都按表重算。输出预算从 512 到 1536 均匀采样，不是所有请求都固定 1536。固定输出 1024 与分布输出的结果分开记录；两者都不自动等于真实业务负载。

`ais_bench/datasets/synthetic/synthetic_config.py` 候选配置：

```python
INPUT_LENGTH = 16384
REQUEST_COUNT = 576
synthetic_config = {
    "Type": "string",
    "RequestCount": REQUEST_COUNT,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": INPUT_LENGTH, "MaxValue": INPUT_LENGTH},
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 512, "MaxValue": 1536},
        },
    },
    # string 模式不使用本项；若切换 tokenid 模式，RequestSize 表示输入长度。
    "TokenIdConfig": {"RequestSize": INPUT_LENGTH, "PrefixLen": 0},
}
```

`ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py` 候选配置：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

REQUEST_COUNT = 576
models = [dict(
    attr="service",
    type=VLLMCustomAPIChat,
    abbr="dsv4-decode-capacity",
    path="/填写本组实际模型路径",
    model="dsv4-afd",
    stream=True,
    request_rate=0,
    retry=0,
    host_ip="192.0.2.11",  # 原生 Decode 或 Attention API，不是 PD Proxy。
    host_port=8910,
    max_out_len=1536,
    batch_size=REQUEST_COUNT,
    trust_remote_code=False,
    generation_kwargs=dict(temperature=0, seed=1024, ignore_eos=True),
)]
```

这里将参考配置的 retry 改为 0，避免重试掩盖失败和改变工作量；容量压力测试使用 `ignore_eos=True` 以尽量完成预设输出预算，接受率结论也限定在这个解码协议下，业务评估另保留自然 EOS 协议。先审查冻结 AISBench 版本是否将 synthetic 的每请求输出预算传入 API：`max_out_len=1536` 只是上限，不能证明每请求预算均匀分布。检查输出预算、原始请求和实际响应 token 数；若客户端只发送同一个上限，则先完成适配或标为固定输出实验。AISBench 的生成种子与模型解码 seed 分开冻结；每次重启客户端不保证数据相同，先保存同一输入/预算母集，各 BS 按 N 选固定前缀或确定性重复，禁止按运行组重新随机生成。

所有请求长度按 V4 tokenizer 和实际 chat template 计算，生成配置里的长度不自动等于服务端有效 token 数。归档服务端 usage/trace 和请求正文，确认无工具自动截断，实际输入加输出预算不超过该点新冻结的上限。8K–64K 主矩阵需要重新完成对应长度的 MAX_MODEL_LEN、target/draft HBM/KV、算子和 Graph 验收，不能在本期 4096 功能配置直接执行。

每个候选点按以下步骤执行：

1. 先按 C1–C5 原顺序关闭功能门禁，再完成本轮输入长度、Graph/U2、DSpark、内存和实际 launcher 的前置检查；登记真实输入/KV、自然路由和工具版本。主要组合优先运行第 8.3 节真实 PD；纯 Decode 和合成实验分别通过其 KV/launcher 检查后再测，互不冒充。
2. 停止其他组，按主要组合/拓扑/BS 独立冷启动，关闭诊断，确认全部 ranks ready。真实 PD 先 Prefill ready，再启动 FFN、紧接着 Attention，最后启动 Proxy；纯 Decode 不在窗口内运行 Prefill/Proxy，AFD 先 FFN、紧接着 Attention；原生均不启动 FFN。采集路径证据后以无诊断配置重启测量服务。N-G/B2 和 N-GS/B3 使用同一轮协议分别配对，draft eager 与 target Graph 分开核验。
3. 单独 warmup 一波 `D×B_D` 或 `A×B_A` 请求，等待请求归零。再提交六波正式请求，至少重复三轮，每轮单独输出目录并等待所有 rank 归零；同一配对轮次使用同一母集和预算分布。
4. 在冻结 AISBench checkout 中执行下面命令；每轮记录工具打印的实际输出目录并完整归档，不覆盖历史结果。所有服务节点按第 8.3.3 节采样 HBM，保存实际活跃数、preemption/recompute、Graph replay/fallback 和通信/计算 trace（诊断 trace 另轮采集）。
5. 核对实际完成数等于 N、失败和重试数为 0、实际输出预算/长度分布正确；确认六波覆盖稳定区间。Graph 主组检查在线 replay 和 U2 两个 stage；DSpark 主组另检查逐 rank proposed/accepted 增量、真实 draft/verify 和本轮接受率。OOM、hang、NaN/Inf、错误输出、未初始化缓存或无法正常清理的样本均无效；纯 Decode 窗口混入 Prefill 时重新分类，真实 PD 的正常 Prefill 不作为错误。
6. 两个主要组合分别保留 BS24 目标点和每个输入长度的最佳合法 BS，用新 RUN_ID/结果目录独立复测三轮，保留扫描全部原始点及选择依据，避免仅凭一次最快值下结论。真实 PD 与纯 Decode 各自成表；DSpark 业务收益使用真实输入/KV 的主要结果，eager 只作代表点归因。

```bash
# 本命令只运行上面的 synthetic 辅助配置；真实输入主测选择已验收的真实数据集配置。
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf
```

#### 8.4.4 结果指标与可声明的结论

```text
Q = 测量窗口中实际完成的输出 token 总数 / 对应墙钟时间
E_D4 = Q_D4 / 4；E_AF = Q_A4F2 / (4 + 2)；E_D8 = Q_D8 / 8
AFD 单位 NPU 收益 = (E_AF / E_native - 1) × 100%
总输出吞吐收益 = (Q_A4F2 / Q_native - 1) × 100%
```

Q 首先采用冻结工具明确界定的完整测量轮次吞吐，包含该轮排队/收尾，排除独立 warmup。若额外报告稳定区间吞吐，所有组使用同一预先冻结的窗口选择规则，确保输出 token 与时间来自同一窗口，另列完整轮次数据。不得用平均 TPOT 的倒数替代聚合吞吐，也不统计合成输入 token。

DSpark 的输出 token 只计算实际接受并最终返回的生成 token，不包含被拒绝的 draft、重复 target verify 或内部 padding。成对记录 proposed/accepted 的实际定义和增量、接受率、target verify/draft token budget、target replay/fallback 比例及 U2 生效证据。主结果按 Graph/U2、Graph/U2+DSpark 分表；跨 DSpark 开关的差值解释为该负载下投机解码收益，不能只凭更高 acceptance rate 断言吞吐更好。

Decode-only 的分母包含所有 A+F 计算设备，不包含未启动的 Prefill，也不只除 Attention 卡数。若改成真实 PD 链路，必须回到第 8.3 节端到端口径，另计 Prefill 和整机成本。NPU 设备到物理 die 的映射未核实前，图表用 `output tokens/s/NPU`，不直接写 `tokens/s/die`。

汇总表列：`主组合/测量入口/input_length/output_budget_distribution/输入集hash/KV来源与初始化/路由模式/target模式与U/draft模式/拓扑/D或A/F/P/物理NPU数/物理die数及映射/预留主机数/BS_D或BS_A/FFN容量/target_verify及draft预算/实际每rank活跃数/W/N/成功与失败/实际返回输出tokens/全轮Q/稳定区间Q(可选)/单位NPU吞吐/TTFT和TPOT分位数/HBM/preemption/proposed与accepted增量/接受率/target_replay与fallback比例/相对配对基线收益/结果目录`。分别画两个主组合在 8K/16K/32K/64K 的总吞吐、单位 NPU 吞吐和 BS 扫描曲线，突出 BS24 目标点；V3.2 的结果只作为方法来源，不作为 V4 的收益目标。

单位 NPU 吞吐提高但总吞吐下降、或资源增加后总吞吐提高但每卡效率下降，都如实展示。独立调优 BS 的结果是“各自配置下的容量/效率对比”；功能开关不匹配则标组合收益。合成 KV、强制均衡（若后续实现）和真实 PD 的数据各自成表，均不得替代正常输出/精度验收。

## 9. 【A5 手工执行】失败回传与最终交付表

任何点失败立即停止后续点。服务仍在时先采集本轮 metrics、NPU、进程、端口及角色完整日志，再按正常流程停服。保存最早错误和最后一个 PD/Attention/FFN 边界。
不要 reset、修改模型 config、同时改多个开关或降低门禁来绕过。需要关闭 Graph 多流定位时，使用新 run ID 保存完整开关配置和结果；回退通过不能自动写成默认全开组合通过。

每个节点归档本机 RUN_ROOT；运行根位于 repo 之外：

```bash
export ARCHIVE="$RUN_BASE/${RUN_ID}-$(hostname -s).tar.gz"
tar -czf "$ARCHIVE" -C "$RUN_BASE/$RUN_ID" "$(hostname -s)"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
```

回传三个角色/Proxy 日志、环境清单 `runtime.json`、site/node env 与 SHA256、完整终端输出、请求 JSON/SSE、before/after metrics、路径证据定位、停服过程及清理结果。
三机使用统一 run ID，但本机目录和配置分别记录。缺机器/权重/离线数据/资源文件时标记 `blocked` 并写明缺项，不用同机或 A3 结果代替 A5 跨机验收。
两机补充验证及第 8.3/8.4 节性能对照另存结果包，明确主机数、角色分布、原生基线状态、KV 来源及资源差异；不自动关闭下表三机验收项。第 8.4 节合成 KV 或性能 launcher 尚未验收时记录 pending，不能用端到端数据冒充 Decode-only 容量结果。

| 验收项 | 需要的结果 |
|---|---|
| M1 standalone 回归 | 新冻结 recipe 上 A4F2 Graph/U2 F0（PD/DSpark 不适用） |
| 同机 C1/C2/C3/C4/C5 | 五点 F0；C1/C5 两轮，其他至少一轮 |
| 三机 C1/C2/C3/C4/C5 | 五点 F0；C1/C5 两轮，C5 具备远程 KV 与远程 A/F 证据 |
| 同机 C5 GSM8K | 冻结协议、足量样本、固定分数门禁通过 |
| 三机 C5 GSM8K | 同一协议及分数门禁通过 |
| 两个 C5 性能 | 诊断关闭，可复现指标，完整请求成功率和局限说明 |
| 交付包 | 固定源码提交/tag、配置/操作入口、证据索引、SHA256、限制和回退步骤 |

只有上述功能与精度门禁完成、其他环境能够按本指导冷启动/验证/正常停止，才可宣布1030 交付完成。单个请求脚本通过或历史功能关闭均不等于本表全部完成。
