# DeepSeek-V4 AFD A5 1030 合并验证指导书

更新日期：2026-10-23（单机 K1–K4 已有测量附件，AFD 仍显著劣化；根据快慢阶段切换将性能筛选档收紧为 4×并发数、一轮，双机候选仍待测）。本文合并旧 A5 dSpark/PD 指导与[1030 执行计划](DEEPSEEK_V4_AFD_1030_EXECUTION_PLAN_ZH.md)，是本期现场操作入口。
执行计划规定交付范围和日期，本文规定现场命令、验证顺序、证据与失败处理。
本文是待执行指导，不代表组合已经通过 A5 验收。

**执行标记**：`【本地已完成】` 为代码/配置的离线检查；`【A5 手工执行】` 为需要你在 A5 实际环境完成的操作。首次执行从[手工交接清单](DEEPSEEK_V4_AFD_A5_MANUAL_HANDOFF_ZH.md)的 A5-01 开始。**2026-10-19 最新跨机核对**：10-16 日 cross-C5 full 92 项通过、90 个非流式响应全部非空且无输出 token ID `0`、四 DP 与 Proxy 请求归零。实际为两机 P2+A4+F2，P/F 在 `.18`、A 在 `.14`；缺 Attention 侧日志/metrics，site.env 与实际地址不符，P/F 末尾有 Killed。当前按[cross-C5 本轮结果第 5 节](DEEPSEEK_V4_AFD_CROSS_C5_20261016_RESULT_ZH.md)补材料，完整 F0、第二轮和三机 P8+A4+F2 尚不能据此关闭。此前各日期的单机状态和对照步骤保留作历史，不要求为确认本轮请求结果立即重跑 full。

## 0. 只做性能验证时，从哪里开始

**当前寻找性能收益时，必测场景以第 8.12 节为准。** 先在单机比较 A4F2/D4 的 BS24、BS48，再在两机比较 A4F4/D8 的相同总容量，核心共 8 个配置点；最后选择一对做 16K 确认。10-24 已收到 S1–S3 及 S2 runtime，S2 eager/U2 已出现约 250 ms 慢生成，**S4 暂不测试；当前只按第 8.12.7 节采集 S1/S2 timeline**，定位后再决定是否恢复拓扑筛选。增大 batch 是性能探索主线，A4F2 双机重测、A8、TP2 和 DSpark 为条件补测。第 8.2–8.10 节保留为操作方法和原四组复现流程，不要求再叠加执行其完整矩阵；第 8.12 节提供独立环境加载和场景选择命令。

按第 8.2–8.10 节顺序执行：**准备环境 → 设置 Synthetic 参数 → 选择一组配置 → 启动 → 检查 → 预热和测量 → 停服 → 换组/扩展长度 → 计算收益**。

**性能数据集统一使用 Synthetic，不使用 GSM8K。** benchmark 按固定 seed 自动生成合成输入，首轮输入 16K、输出 1024，同一 BS 的四组保持生成参数一致。GSM8K 仅用于第 8.1 节的独立精度验收。

先在**同一台 8 卡 A5** 上依次运行四组：

| 执行顺序 | 配置 | 用卡 |
|---|---|---:|
| 1 | 原生 Decode + Graph | P2 + D4，共 6 卡 |
| 2 | AFD P2P + Graph/U2 | P2 + A4 + F2，共 8 卡 |
| 3 | 原生 Decode + Graph + DSpark | P2 + D4，共 6 卡 |
| 4 | AFD P2P + Graph/U2 + DSpark | P2 + A4 + F2，共 8 卡 |

**1 对 2、3 对 4**，分别得到不开/开启 DSpark 时的收益。先完成四组 16K/BS8 预检，再测 16K/BS24，最后扩展 32K/64K。BS 指每个 Decode/Attention rank 的活跃序列容量。

第 1–7 节保留功能验收和故障处理；已知问题及实机结论不因本次文档重写而改变。跨机、eager、合成 KV 等补测统一放在第 8.11 节，首轮无需展开。

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

性能验证统一按第 8.2–8.10 节使用 Synthetic 数据，经过单机真实 Prefill/KV 和 PD 流程，分别比较 Graph、Graph+DSpark 两对原生/AFD 配置。C1–C5 都启用 AFD，彼此比较只能解释组合差异，AFD 部署收益必须与原生 Decode 配对测量。

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
export MAX_CUDAGRAPH_CAPTURE_SIZE=
export CUDAGRAPH_CAPTURE_SIZES=   # 留空，由当前 BS 和 speculative token 数推导
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

## 8. 【A5 手工执行】第 6 步：精度验收与 Synthetic 性能对照

### 8.1 M1 冻结精度协议，M5 执行

**本节 GSM8K 只用于精度验收。只做性能测试时，从第 8.2 节开始，使用 Synthetic；性能发压不读取本节数据。**

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

### 8.2 性能步骤 1：准备一台 A5，确认可以开始测试

**在哪里操作：A5 容器内的一个 Bash 协调终端。** 后续命令都在该终端按顺序执行，各服务在后台运行；换终端时重新加载环境。计时请求统一经过 `Proxy → Prefill → Decode/Attention`，结果称为“真实 PD 端到端性能”。

先完成三件事：

1. 按第 3 节准备并核对实际 `site.env`、`node.env`、软件提交、镜像、模型和 HIXL 资源。后续使用这套已冻结环境，不临时升级工具。
2. 按第 5–7 节完成相关功能、输出质量和正常停服验收。原文记录的 C3 输出/默认全开、C4 指标/停服、C5 待验证问题仍须关闭；原生 Graph/DSpark 命令也须通过下面的预检。未通过时只排查，不填正式收益。第 8.1 节的 GSM8K 单独安排，不作为性能数据准备步骤；完整交付仍须完成精度验收。
3. 清空本机旧服务，确认 8 卡可用。固定 P2 用卡 0、1；原生 D4 或 A4 用卡 2–5；只有 AFD 组启动 F2，用卡 6、7。Proxy 不额外占 NPU。

在协调终端先加载第 3.1 节的环境，然后固定单机拓扑：

```bash
set -euo pipefail
# SITE_FILE、NODE_FILE 使用第 3.1 节已经填写并验证过的实际文件。
source "$SITE_FILE"
source "$NODE_FILE"
source "$CANN_ROOT/set_env.sh"
source "$VENV_ROOT/bin/activate"
export PYTHON_BIN="$VENV_ROOT/bin/python" VLLM_BIN="$VENV_ROOT/bin/vllm"
export RECIPE="$AFD_PLUGIN_ROOT/recipe/npu/P2pHcclAFDConnector/deepseek_v4"
export TOPOLOGY=single PREFILL_DP_SIZE=2 PREFILL_TP_SIZE=1
export PREFILL_DEVICES=0,1 ATTENTION_DEVICES=2,3,4,5 FFN_DEVICES=6,7
export ATTENTION_RANKS=4 FFN_RANKS=2 TENSOR_PARALLEL_SIZE=1
export PREFILL_HOST_IP="$HCCL_IF_IP" ATTENTION_HOST_IP="$HCCL_IF_IP"
export FFN_HOST_IP="$HCCL_IF_IP" PROXY_HOST_IP="$HCCL_IF_IP" AFD_HOST="$HCCL_IF_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME" HCCL_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,$HCCL_IF_IP"
export no_proxy="$NO_PROXY"
export BASE_URL="http://$PROXY_HOST_IP:$PROXY_PORT"
export ATTENTION_URL="http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT"
export API_PORT="$ATTENTION_API_PORT"
export PERF_BASE="$RUN_BASE/performance"
mkdir -p "$PERF_BASE"
"$VLLM_BIN" bench serve --help > "$PERF_BASE/bench-help.txt"
```

**完成条件：** 版本/资源审计通过，旧服务已清理；本机 HIXL 至少覆盖 P 的 0、1 和 D/A 的 2–5。确认 CLI 支持后续 `random`、`--random-input-len`、`--random-output-len`、`--random-range-ratio`、`--random-prefix-len`、`--metadata`、`--save-detailed` 等参数。原生启动参数是待实机验收的候选配置。

### 8.3 性能步骤 2：设置 Synthetic 参数，由 benchmark 自动生成

**不需要手动生成数据文件，也不需要准备 JSONL。** 参考 recipe 的 AISBench 通过 `--datasets synthetic_gen` 自动生成合成请求。本指导已有的 `vllm bench serve` 同样支持自动生成，使用 `--dataset-name random`；`random` 就是它的 Synthetic 数据入口。

本步骤只确认参数；第 8.4 节设置变量，第 8.6 节调用 benchmark 时自动生成请求并发压：

| 参数 | 首轮取值 | 含义 |
|---|---|---|
| `--dataset-name` | `random` | 自动生成 Synthetic 文本 |
| `--random-input-len` / `ISL` | `16384` | 输入目标长度 16K；后续改为 32768、65536 |
| `--random-output-len` / `OSL` | `1024` | 每条请求输出预算，配合固定长度解码参数 |
| `--random-range-ratio` | `0.0` | 输入/输出长度不随机浮动 |
| `--random-prefix-len` | `0` | 不额外添加固定公共前缀 |
| `--seed` | `1024` | 合成请求生成 seed |
| `--num-prompts` | 由本轮 `count` 指定 | 预检/预热为 `CAPACITY`；本节原流程正式为 `6×CAPACITY`，第 8.12 节另设短测/确认档 |

四组统一使用冻结的 Flash tokenizer，并核对 DSpark checkpoint 的 tokenizer 文件 hash 相同。同一个长度/BS 的配对组保持 benchmark、NumPy、tokenizer 版本和上述参数一致；三轮正式测量也保持一致。不同请求数可能生成不同输入集合，不能只凭 seed 相同就认定输入逐条相同。

自动生成在 benchmark 的请求计时开始前完成。正式轮次的 JSON 通过 `--metadata` 保存生成参数，连同 tokenizer hash、实际命令和结果归档即可；预先落盘只在另有逐条审计需求时安排，不属于本流程。

**完成条件：** CLI 支持这些参数，冻结版本和 tokenizer 已核对；随后直接进入第 8.4 节。第 8.6 节预检还要核对实际输入/输出 token 数；自动生成器报告 token 长度不匹配时先排查，不把目标 16K 当成实际长度。

参考：[V3.2 recipe 的 AISBench 配置](../../recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md#aisbench-configuration)、[固定 vLLM 的 random 数据入口](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/benchmarks/datasets/datasets.py)。两种工具都能自动生成 Synthetic；本次四组统一使用下文的 vLLM 命令。

### 8.4 性能步骤 3：选择本轮配置，创建新目录

**换配置、输入长度、BS 或诊断模式时，都从本步骤重新开始。** 四组按表逐组执行：

| 顺序 / `PERF_CASE` | 模型 | Decode 路径 | DSpark |
|---|---|---|---|
| 1：`native_graph` | Flash | 原生 D4，target Graph | 关 |
| 2：`afd_graph` | 同一 Flash | A4F2，target Graph/U2（C3） | 关 |
| 3：`native_dspark` | DSpark checkpoint | 原生 D4，target Graph | draft eager |
| 4：`afd_dspark` | 同一 DSpark checkpoint | A4F2，target Graph/U2（C5） | draft eager |

原生首次候选为 Graph/U1，AFD 为 Graph/U2；保留此差异时，结论称为“两种部署配置的收益”。隔离 AFD 本身的贡献时，再核验原生 U2 并建立新对照；不用 eager 原生替代 Graph 主基线。

首次只改下面的组名，保留 `PHASE=check`、16K、BS8。后续怎样修改见第 8.9 节。

```bash
export PERF_CASE=native_graph     # 按表依次换组
export PHASE=check                # check：路径预检；measure：正式计时
export CASE="$PERF_CASE" CYCLE=1
export ISL=16384 OSL=1024 DECODE_BS=8
export MAX_MODEL_LEN=$((ISL + OSL))
export MAX_NUM_SEQS="$DECODE_BS"
export MAX_CUDAGRAPH_CAPTURE_SIZE= CUDAGRAPH_CAPTURE_SIZES=
export FFN_SEQ_CAPACITY=$((2 * DECODE_BS))
export CAPACITY=$((4 * DECODE_BS))
export REQUEST_COUNT=$((6 * CAPACITY))
export PERF_REPEATS=3 PERF_TIER=confirm
export PERF_REQUEST_RATE=inf
unset SLO_TTFT_MS SLO_TPOT_MS
export ENABLE_PD=1 ENABLE_MTP=0 ENABLE_DSPARK=0 USE_AFD=0
export MODEL_PATH="$FLASH_MODEL_PATH" EXECUTION_MODE=full-decode-only U_BATCHES=1
case "$PERF_CASE" in
  native_graph) ;;
  afd_graph) export USE_AFD=1 U_BATCHES=2 ;;
  native_dspark) export MODEL_PATH="$DSPARK_MODEL_PATH" ENABLE_DSPARK=1 ;;
  afd_dspark) export MODEL_PATH="$DSPARK_MODEL_PATH" ENABLE_DSPARK=1 USE_AFD=1 U_BATCHES=2 ;;
  *) printf 'Unknown PERF_CASE\n' >&2; exit 2 ;;
esac
case "$PHASE" in
  check) export AFD_HCCL_STAGE_DIAGNOSTICS=1 ;;
  measure) export AFD_HCCL_STAGE_DIAGNOSTICS=0 ;;
  *) printf 'PHASE must be check or measure\n' >&2; exit 2 ;;
esac
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0 VLLM_LOGGING_LEVEL=INFO
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=1 AFD_HCCL_GRAPH_U2_HYBRID_DAG=1
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1 AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-single-$PERF_CASE-i$ISL-b$DECODE_BS-$PHASE"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
cp "$SITE_FILE" "$RUN_ROOT/site.env"
cp "$NODE_FILE" "$RUN_ROOT/node.env"
```

**启动前核对：** 16K+1024 的上限是 17408，不能继续用 4096。保留已冻结的 `GPU_MEMORY_UTILIZATION=0.91`，重新检查 P/D/A/F 的 KV 和 HBM。`FFN_SEQ_CAPACITY=2×DECODE_BS` 仅是序列容量起点，DSpark 的 verify/draft token、buffer、padding 预算另核验。

不要沿用旧配置的固定 max=8。当前 recipe 在 capture 变量为空时，按每 rank、U2 切分前的 token 上限生成档位：普通 Decode 为 `DECODE_BS`，DSpark 为 `DECODE_BS × (dspark_block_size + 1)`。BS24 对应普通 Decode max=24、档位 `1 2 4 8 16 24`；DSpark block=5 对应 max=144、档位 `6 12 24 48 96 144`。U2 不会把 dispatcher 所需的 max 减半。显式覆盖值不足、最大档位不一致或超出角色 token budget 会在启动前报错。

每个新配置点重新加载上述空 capture 变量，避免复用上一组派生值。A/F 在各自进程内推导；F 无 speculative config，其实际远端 Graph 由 A 的 metadata 触发捕获。F 的接收/计算 buffer 仍须覆盖 A/F 聚合及 DSpark verify 的 token 数。扩大 BS 后重新验收 HBM、实际 replay/eager 比例和 A/F stage key，并保存各角色实际 argv/capture/token budget。自动覆盖上限不代表所有在线 shape 都已 replay。**未经预检的配置不进入 `measure`。**

按第 4.4 节保存 runtime，补充本轮 `PERF_CASE/PHASE/ISL/OSL/DECODE_BS/FFN_SEQ_CAPACITY`、各角色 token budget、capture sizes、Synthetic 生成参数和固定 tokenizer hash。`USE_AFD` 仅选择下文命令分支，不是插件开关。

### 8.5 性能步骤 4：启动服务，最后启动 Proxy

**4.1 先启动 P，等它 ready。** 每次最多等待 20 分钟；失败保存日志并排查，不继续后续角色。

```bash
wait_http() {
  local url="$1" deadline=$((SECONDS + 1200))
  until curl --noproxy '*' -fsS --max-time 5 "$url" >/dev/null; do
    ((SECONDS < deadline)) || return 1
    sleep 5
  done
}
MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-$MAX_NUM_SEQS}" \
  nohup bash "$RECIPE/afd_prefill.sh" > "$RUN_ROOT/prefill.log" 2>&1 &
echo $! > "$RUN_ROOT/prefill.pid"
wait_http "http://$PREFILL_HOST_IP:$PREFILL_API_PORT/health"
```

**4.2 根据本组启动 A/F 或原生 D。** AFD 先 F 后 A，两条紧接执行；原生不启动独立 F。开启 DSpark 时，P/A/F 都保持同一 DSpark checkpoint，只有 A/原生 D 加 drafter。

```bash
if [[ "$USE_AFD" == 1 ]]; then
  MAX_NUM_SEQS="$FFN_SEQ_CAPACITY" ENABLE_PD=0 ENABLE_DSPARK=0 \
    nohup bash "$RECIPE/afd_ffn.sh" > "$RUN_ROOT/ffn.log" 2>&1 &
  echo $! > "$RUN_ROOT/ffn.pid"
  nohup bash "$RECIPE/afd_attention.sh" > "$RUN_ROOT/attention.log" 2>&1 &
  echo $! > "$RUN_ROOT/attention.pid"
else
  source "$RECIPE/common.sh"
  export ASCEND_RT_VISIBLE_DEVICES="$ATTENTION_DEVICES" VLLM_HOST_IP="$HCCL_IF_IP"
  export HCCL_IF_BASE_PORT="$ATTENTION_HCCL_IF_BASE_PORT" VLLM_USE_V1=1
  KV_CONFIG="$(build_a5_pd_kv_config kv_consumer \
    "$DECODE_ENGINE_ID" "$DECODE_KV_PORT" "$ATTENTION_DEVICES")"
  printf '%s\n' "$KV_CONFIG" > "$RUN_ROOT/native-kv-config.json"
  configure_dspark
  MAX_NUM_BATCHED_TOKENS="$ATTENTION_MAX_NUM_BATCHED_TOKENS" configure_graph_capture
  read -r -a native_capture_sizes <<< "$CUDAGRAPH_CAPTURE_SIZES"
  nohup "$VLLM_BIN" serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$ATTENTION_API_PORT" \
    --api-server-count 1 --served-model-name dsv4-afd \
    --data-parallel-size "$((ATTENTION_RANKS / TENSOR_PARALLEL_SIZE))" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --enable-expert-parallel \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$DECODE_BS" \
    --max-num-batched-tokens "$ATTENTION_MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --seed 1024 \
    --tokenizer-mode deepseek_v4 --no-enable-prefix-caching \
    --safetensors-load-strategy prefetch --block-size 32 --kv-cache-dtype auto \
    --no-async-scheduling --kv-transfer-config "$KV_CONFIG" \
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE" \
    --cudagraph-capture-sizes "${native_capture_sizes[@]}" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    "${DSPARK_ARGS[@]}" > "$RUN_ROOT/decode.log" 2>&1 &
  echo $! > "$RUN_ROOT/decode.pid"
fi
wait_http "$ATTENTION_URL/health"
```

**完成条件：** 四个 D/A rank ready；AFD 另确认两个 F rank 进入 connector loop。原生日志/argv 无 `additional_config.afd`，AFD 有正确 role 和 A4F2 mapping。保存实际通信后端；Graph 在线 replay、DSpark 四个 drafter 和执行证据在下一步核对。

**4.3 后端就绪后启动 Proxy。**

```bash
nohup bash "$RECIPE/afd_proxy.sh" > "$RUN_ROOT/proxy.log" 2>&1 &
echo $! > "$RUN_ROOT/proxy.pid"
wait_http "$BASE_URL/healthcheck"
curl --noproxy '*' -fsS "$BASE_URL/healthcheck" > "$RUN_ROOT/proxy-ready.json"
"$PYTHON_BIN" - "$RUN_ROOT/proxy-ready.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1]))
assert state["status"] == "ok" and state["request_num"] == 0, state
assert state["prefill_instances"] == 1 and state["decode_instances"] == 1, state
PY
```

### 8.6 性能步骤 5：先检查请求和真实运行路径

先在当前终端定义下面的函数，预检、预热和正式轮次共用它。benchmark 按第 8.3 节参数自动生成 Synthetic 输入，固定输出预算；`ignore-eos/min_tokens` 用于固定长度压力协议。输出由模型实际生成，不预填输出 token，也不替换专家路由。

```bash
run_perf_batch() {
  local name="$1" count="$2"
  local -a goodput_args=()
  if [[ -n "${SLO_TTFT_MS:-}${SLO_TPOT_MS:-}" ]]; then
    : "${SLO_TTFT_MS:?Set both SLO_TTFT_MS and SLO_TPOT_MS}"
    : "${SLO_TPOT_MS:?Set both SLO_TTFT_MS and SLO_TPOT_MS}"
    goodput_args=(--goodput "ttft:$SLO_TTFT_MS" "tpot:$SLO_TPOT_MS")
  fi
  curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/$name.metrics-before.txt"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$RUN_ROOT/$name.wallclock-begin.txt"
  "$VLLM_BIN" bench serve \
    --backend openai --base-url "$BASE_URL" --endpoint /v1/completions \
    --model dsv4-afd --tokenizer "$FLASH_MODEL_PATH" --tokenizer-mode deepseek_v4 \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 0.0 --random-prefix-len 0 \
    --seed 1024 --num-prompts "$count" --max-concurrency "$CAPACITY" \
    --request-rate "${PERF_REQUEST_RATE:-inf}" "${goodput_args[@]}" \
    --ignore-eos --num-warmups 0 --ready-check-timeout-sec 0 \
    --extra-body "{\"temperature\":0,\"seed\":1024,\"min_tokens\":$OSL}" \
    --metadata dataset=Synthetic synthetic_seed=1024 input_len="$ISL" output_len="$OSL" \
      prefix_len=0 range_ratio=0.0 perf_tier="${PERF_TIER:-confirm}" \
      concurrency="$CAPACITY" num_prompts="$count" repeats="${PERF_REPEATS:-3}" \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99 \
    --save-result --save-detailed --result-dir "$RUN_ROOT" --result-filename "$name.json" \
    2>&1 | tee "$RUN_ROOT/$name.console.log"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$RUN_ROOT/$name.wallclock-end.txt"
  curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/$name.metrics-after.txt"
  "$PYTHON_BIN" - "$RUN_ROOT/$name.json" "$count" "$OSL" \
    "$BASE_URL" "$ATTENTION_URL" "$RUN_ROOT/$name.drain.json" "$ISL" <<'PY'
import json
import re
import sys
import time
from pathlib import Path

import httpx

result = json.loads(Path(sys.argv[1]).read_text())
assert result["completed"] == int(sys.argv[2]) and result["failed"] == 0, result
assert result["total_input_tokens"] == int(sys.argv[2]) * int(sys.argv[7]), result
assert result["total_output_tokens"] == int(sys.argv[2]) * int(sys.argv[3]), result
deadline = time.monotonic() + 60
with httpx.Client(trust_env=False, timeout=10) as client:
    while True:
        response = client.get(f"{sys.argv[4]}/healthcheck")
        response.raise_for_status()
        proxy = response.json()
        response = client.get(f"{sys.argv[5]}/metrics")
        response.raise_for_status()
        gauges = {}
        for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            values = re.findall(rf"^{re.escape(metric)}(?:\{{[^\n]*\}})? (\S+)",
                                response.text, re.MULTILINE)
            assert values, f"Missing metric: {metric}"
            gauges[metric] = [float(value) for value in values]
        snapshot = {"proxy": proxy, "decode_gauges": gauges}
        Path(sys.argv[6]).write_text(json.dumps(snapshot, indent=2))
        if proxy["status"] == "ok" and proxy["request_num"] == 0 and all(
                value == 0 for values in gauges.values() for value in values):
            break
        assert time.monotonic() < deadline, snapshot
        time.sleep(1)
PY
}
```

本例检查 `completed/failed/total_input_tokens/total_output_tokens`；字段不符时先核对冻结版本，不删除成功数/输入输出量检查。CLI 可对照固定 [vLLM benchmark 源码](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/benchmarks/serve.py)。无可信 token usage 时先核对统计路径，draft/rejected/verify/padding token 不计作输出。

**只有 `PHASE=check` 时执行预检：**

```bash
if [[ "$PHASE" == check ]]; then
  "$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
    --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" --suite smoke \
    --run-id "$RUN_ID" --output "$RUN_ROOT/smoke"
  run_perf_batch long-input-check "$CAPACITY"
  npu-smi info > "$RUN_ROOT/npu-after-check.txt"
fi
```

保存这些证据：真实 KV 收到且输出正常；四个 D/A rank 的实际输入长度、活跃数和 HBM/KV；target 在线 Graph replay；AFD 的 U2 stage 0/1 和两个 F rank 完成；DSpark 的真实 draft/verify、proposed/accepted 增量及 draft eager。原生 DSpark 兼容性以本轮证据为准。

**完成条件：** smoke 和 Synthetic 长输入通过，无异常 token 退化、OOM、hang、NaN/Inf，路径符合本组配置。Synthetic 输出不做 GSM8K 答案评分；smoke 不代替第 1–7、8.1 节完整验收。检查轮直接到第 8.8 节停服；四组 BS8 都通过后才开始正式轮次。

### 8.7 性能步骤 6：关闭诊断，预热后测量

从第 8.4 节或当前第 8.12 节的独立入口重新加载配置，两者择一，不交叉覆盖。设置 `PHASE=measure`，选择**已经通过预检**的长度/BS/capture/budget，以新 RUN_ID 冷启动。保持相同 Synthetic 参数/模型/执行开关，关闭诊断、DEBUG 和 profiler；不拿预检进程计时。

第 8.4 节原流程固定 `PERF_REPEATS=3`；第 8.12 节使用其独立配置，短测为 1 轮、正式确认为 3 轮。不要将筛选结果按正式三轮验收。

ready 且第 8.6 节函数已定义后执行：

```bash
if [[ "$PHASE" == measure ]]; then
  (
    while :; do date -Iseconds; npu-smi info; sleep 5; done
  ) > "$RUN_ROOT/npu-samples.txt" 2>&1 &
  echo $! > "$RUN_ROOT/npu-sampler.pid"
  run_perf_batch warmup "$CAPACITY"
  for ((repeat=1; repeat<=${PERF_REPEATS:-3}; repeat++)); do
    run_perf_batch "measure-$repeat" "$REQUEST_COUNT"
  done
fi
```

每个 batch 后等待 Proxy 请求数、D/A running/waiting 归零，再进入下一轮。失败按第 9 节保存现场，不继续推进。HBM 每 5 秒采样，报告采样最大值，不当作精确瞬时峰值。

| 每个 D/A rank 的 BS | F 序列容量起点 | 预热请求 | 每轮正式请求 | 客户端在途上限 |
|---:|---:|---:|---:|---:|
| 8 | 16 | 32 | 192 | 32 |
| 16 | 32 | 64 | 384 | 64 |
| 24 | 48 | 96 | 576 | 96 |
| 32（可选） | 64 | 128 | 768 | 128 |

“六波”表示 `正式请求数=4×BS×6`：BS24 共提交 576 条，客户端最多在途 96 条，完成后补充；不表示 576 条同时运行，也不保证每卡实际活跃数达到 24。检查 running/waiting、Prefill 供给和吞吐曲线；P2 或客户端饱和时如实记录瓶颈。窗口不稳定时，配对组统一增大请求数并重新冻结。

**完成条件：** 正式三轮的 `measure-*.json` 均完成 N 条、失败 0、输入/输出量符合预算；第 8.12 节的一轮短测仅作为筛选。DSpark 接受率按本轮有效 proposed/accepted 增量计算，注明分子分母/时间窗和 Synthetic 负载；指标异常保存缺项，不填 0。通过 Proxy 发压时单独采集 D/A `/metrics`，不假定 Proxy 暴露其指标。

### 8.8 性能步骤 7：请求归零后正常停服

停止新请求，确认最后一个 `*.drain.json` 中 Proxy 请求数和所有 D/A running/waiting 为 0，再执行：

```bash
kill -TERM "$(cat "$RUN_ROOT/proxy.pid")"
if [[ "$USE_AFD" == 1 ]]; then
  kill -TERM "$(cat "$RUN_ROOT/attention.pid")"
  # 暂停：人工确认两个不同 F rank 都收到 shutdown，再执行下一个代码块。
  grep 'AFD NPU FFN received Attention shutdown payload' "$RUN_ROOT/ffn.log"
else
  kill -TERM "$(cat "$RUN_ROOT/decode.pid")"
fi
```

AFD 的两个 F rank 都收到 shutdown 后执行；原生直接执行：

```bash
if [[ "$USE_AFD" == 1 ]]; then
  kill -TERM "$(cat "$RUN_ROOT/ffn.pid")"
fi
kill -TERM "$(cat "$RUN_ROOT/prefill.pid")"
if [[ -f "$RUN_ROOT/npu-sampler.pid" ]]; then
  kill -TERM "$(cat "$RUN_ROOT/npu-sampler.pid")"
fi
npu-smi info > "$RUN_ROOT/npu-after-stop.txt"
ss -ltnp > "$RUN_ROOT/ports-after-stop.txt"
```

**完成条件：** 本轮主/子进程退出、端口释放、NPU 清理完成，无 Segfault 或设备错误；AFD 按第 7 节完整核对正常 shutdown。停服失败先保存材料并修复，强制清理不记作通过，不继续下一组。

### 8.9 性能步骤 8：按顺序换组，再扩展长度和 BS

每行都按 `native_graph → afd_graph → native_dspark → afd_dspark` 执行，第 8.4–8.8 节组成一组的完整操作。清理完成后才换下一组。

| 阶段 | 输入 / 每卡 BS | 操作及通过条件 |
|---|---|---|
| 首次跑通 | 16K / BS8 | 四组各用 `check`；请求、路径、正常停服通过 |
| 首次正式比较 | 16K / BS24 | 先做 BS24 的 `check` 和 capture/budget/HBM 验收，再用 `measure` 冷启动，四组各三轮 |
| 扩展长输入 | 32K、64K / BS8→16→24 | 每个新长度/BS 先 `check`；合法点进入 `measure`，仍配对三轮 |
| 可选容量调优 | 各长度 / BS8、16、24、32 | 冻结双方各自最佳合法 BS，再各复测三轮，与同 BS 结果分表 |

32K/64K 在第 8.4 节把 `ISL` 分别设为 `32768`、`65536`；benchmark 自动生成对应长度的请求，无需准备数据文件。输入上限加 1024 输出对应 `MAX_MODEL_LEN` 至少为：16K **17408**、32K **33792**、64K **66560**。每个长度重新验证 KV/HBM、Graph、DSpark；BS24 是目标，不保证都能容纳。

某点 OOM/不支持时，记录实际支持的 BS，同 BS 收益记缺项；另表比较各自最佳合法配置。排队成功不证明达到目标活跃 BS。原生 Graph/DSpark 不可用时该配对保持 pending，不用 eager 补位。

### 8.10 性能步骤 9：计算收益，交付两张表

只取正式三轮，不混入 check/warmup 或其他长度。保留原始值，报告中位数和最小/最大值。Q 使用 `output_throughput`（实际返回输出 tokens/s），不使用 input+output 吞吐、平均 TPOT 倒数或单次峰值。

| 比较 | 计算方式 |
|---|---|
| 总输出吞吐收益 | `(Q_AFD / Q_native - 1) × 100%` |
| 实际用卡效率 | 原生 `Q_native/6`；AFD `Q_AFD/8`，均包含 P2 |
| 实际用卡效率收益 | `((Q_AFD/8) / (Q_native/6) - 1) × 100%` |
| 同整机成本 | 两组都预留 1 台 8 卡 A5，同时报告各自 `Q/8` |
| 延迟与资源 | TTFT/TPOT 均值和 P95/P99、各角色 HBM；DSpark 加接受率 |

两张表都标注“**Synthetic，seed=1024，固定输入/输出长度**”。不开 DSpark 一张表，开启 DSpark 一张表：

| 输入/输出 | D/A BS | F 容量 | 原生 Q | AFD Q | 总吞吐收益 | 每卡效率收益 | TPOT P95（原生/AFD） | TTFT P95（原生/AFD） | HBM/接受率/目录 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 16K/1024 | 24 | 48 起，填实际预算 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

附成功/失败数、实际输入/输出 tokens、请求数、Synthetic 生成参数、tokenizer/模型/runtime hash、卡位、target/draft 模式、原生微批差异和停服结果。归档各轮结果 JSON、生成参数、实际 argv、冻结版本和角色配置，不要求预先生成的数据文件。原生实际 6 卡、AFD 实际 8 卡，每卡归一化不等于严格等卡数对照；结论限定为本机这两种部署配置在 Synthetic 负载下的性能，DSpark 接受率不外推为业务数据接受率。另解释 DSpark 开关的收益时先核对 target 权重 hashes 一致。

**完成条件：** 两对都有三轮结果、质量/路径/清理证据，并说明瓶颈。负收益也如实报告，不新增最低收益比例门槛；按第 9 节归档。

### 8.11 主流程完成后，再选择补测

| 补测 | 何时做 / 口径 |
|---|---|
| 两机跨机 AFD | 单机主结果完成后，按第 4.2.1 节重跑配对，说明网络及 P/F 共机条件 |
| eager 归因 | 选代表点解释 Graph/U2、DSpark 贡献，不扩成首轮必测矩阵 |
| 8K、输出 512–1536、自然 EOS、固定 RPS | 另冻结数据/预算/延迟目标，与固定 1024、饱和吞吐分表 |
| 纯 Decode / 合成 KV / 可选 D8 | 先验收独立 launcher、KV 初始化、V4 多 KV group 和正确性；不纳入单机 PD 主流程。D8 加 P2 超过 8 卡，不能同时放入本布局 |

参考 [V3.2 CAMP2p recipe](../../recipe/npu/CAMP2pAFDConnector/deepseek_v3_2/README.md) 的 Synthetic 数据设计、BS 扫描、F 容量和六波供给；V4 主测固定输出 1024，使用 Synthetic 输入、真实 Prefill/KV 和自然专家路由。Synthetic 输入不等于合成 KV；参考中的强制均衡、逻辑专家规模模拟和合成 KV 不加入本节主测。

当前 `tools/benchmarks/decode_bench.py` 的 `start_load_kv()` 不调用其 KV 填充流程，设置 fill mean/std 不证明缓存已初始化。独立纯 Decode 验收后再测，其分母包含所有 D 或 A+F 设备；本节经过 Proxy 的结果始终包含 P 成本。NPU/die 映射未核实时，单位写 `tokens/s/NPU`。

### 8.12 【A5 手工执行】寻找 AFD 收益点：统一调整 BS、A/F 配比与测试负载

本节是 **2026-10-23 更新的待执行实验方案**。10-21 非 DSpark 测量中，AFD 的生成阶段约慢 11.6–11.9 倍，A/F 利用率也低；当前证据不足以将其归因于 F2 带宽或小 batch。先消除执行异常，再寻找吞吐、延迟和资源效率的折中点；不能预先保证某个拓扑有收益。

**10-23 单机反馈：** K1/K2/K4 各收到一轮、K3 三轮，均为 1024/1024、每轮 6×C。AFD BS24/48 为 622.26/850.96 tok/s，native 为 2881.22/4722.06 tok/s；Graph max 已为 24/48，仍有显著生成时延差距。K1 前两组各 C 条请求平均 TPOT 为 53.35/56.72 ms，第三组升至 248.65 ms，故原先 2×C 可能漏掉慢阶段，下面性能短测改为至少 4×C。优先定位快慢切换，暂缓 BS96 和完整双机矩阵；候选拓扑、精度及正常停服验收不能由请求完成推定通过。

**本轮范围：8 个核心配置点各做筛选，再对一组 AFD/native 做 16K 确认。** 所有核心点统一 TP1、关闭 DSpark、固定自然路由和 Graph 配置策略；不展开“全部拓扑×全部 BS×全部长度×DSpark 开关”。已经完成且软件/权重/执行路径/负载完全一致、证据有效的配置点直接复用；旧 max=8 或不同长度/版本的结果不能代替修正后基线。

#### 8.12.1 先明确每种调整解决什么问题

先关闭 DSpark，固定 TP1、自然专家路由、相同 U2 配置。记 A 侧 DP 数为 `D_A`，每 DP 实际参与该步的序列数为 `B`，则普通 Decode 全局真实 token 数约为 `N=D_A×B`。U2 均分时，每个 stage 约为 `N/2`；实际不均衡、dummy 和 padding 单列。

若该层有 `E=256` 个 routed experts、每 token 路由 `k` 个专家，均匀路由下每个专家每 stage 的平均 token 数约为 `N×k/(2×E)`。`k` 以实际模型配置为准，真实分布需采集，不能用均值宣称所有专家均已激活。

| 调整 | 全局 batch / 每专家 token | 资源和可能代价 |
|---|---|---|
| A4F2，BS24→48 | N：96→192，每专家 token 约翻倍 | 卡数不变；A 的计算、KV 和 F 的工作量增加，TPOT 可能上升 |
| A4F2→A4F4，同 BS | N 不变，每专家 token 不变 | F 的 EP2→EP4，均匀放置时每卡专家 128→64；增加计算/带宽容量，也改变 EP 通信 |
| A4F2 BS48→A8F2 BS24 | N 都为 192，每专家 token 近似相同 | A 卡翻倍、每 A 的 KV/序列压力降低；F2 容量未增加，需要额外资源及跨机 A/F |
| A4F4 TP1→TP2，同 N | A DP4→DP2，需要把每 DP BS 翻倍 | F 的 EP 仍为 4；Attention 增加 TP 通信，不能当作专家再切半 |

因此，**BS 与增加 F 是互补的两个变量**。优先比较 A4F2 BS24/48 和 A4F4 BS24/48。U2 的每专家小 batch 问题不因 F2→F4 自动改善；更大的 BS 也不保证延迟更好。

#### 8.12.2 第一阶段：先确认 Graph/U2 的实际执行路径

1. 各节点冻结相同插件提交、vLLM/vLLM-Ascend、CANN/驱动和模型/tokenizer hashes，原生也使用同一套环境。旧结果保留为历史；新旧版本分别成表。
2. 先完成下表 K3/K1：原生 D4 与 A4F2 在同一台 8 卡 A5、BS24、并发 96 下预检，输入/输出为 1024/1024。原来的 16384/1024 负载留到优胜配置的确认阶段。短输入仍走真实 Prefill/KV/Proxy，称为“Decode 占比较高的端到端测试”。
3. capture 留空由 recipe 生成：无 DSpark，BS24 max=24、BS48 max=48；DSpark block=5 时分别为 144、288。检查实际 replay/eager 次数、per-step 真实 token、两级 U2 stage key，以及 F 接收 buffer。仅有捕获日志不能证明在线命中。
4. 若仍出现数量级的生成时延差距和低 A/F 利用率，暂停全矩阵扩卡。先采集一个代表点约 5–10 秒的 A/F timeline，拆分 A 计算、A→F、F MoE/EP 通信、F→A、CPU launch 和等待，关联相同 step/stage，避免把重叠区间重复相加。
5. 必要时单独做双方 eager/U1 对照，再比较 AFD eager/U1→eager/U2→Graph/U2，控制 eager overlap 和五个 Graph 开关，按第 8.12.6 节执行。原生 eager 须移除第 8.5 节的 Graph 参数、改用 `--enforce-eager`；AFD 使用 `EXECUTION_MODE=eager U_BATCHES=1/2`。当前 recipe 的 Graph 路径要求 U2，不能直接构造 AFD Graph/U1。诊断结果与正式 Graph 部署对照分表，计时前关闭诊断/profiler。

无既有 replay 计数时，以受控短 profile 验证；不得将 `is_graph_capturing=False` 当作 eager 回退证据。运行异常尚未定位时，只做少量诊断点，不将“扩大 BS 后吞吐变高”视为异常已经修复。

#### 8.12.3 核心场景：单机增大 batch，再做 A4F4 等资源比较

**第一组 K1–K4：单机增大 batch，必测。** P2 用 0、1，A4/D4 用 2–5，只有 AFD 启动 F2、用 6、7。K1→K2 直接观察 A4F2 的 BS24→48；K3→K4 同时给原生增加 batch，避免只优化 AFD。P 的容量参数在各点固定。增加 BS 不增加卡数，但要重新验证 Graph、buffer、KV/HBM 和实际活跃数。

**第二组 K5–K8：双机 A4F4 与 native D8，必测。** P2+Proxy 独立一机，A4F4 或 D8 独占另一机，两者均为 10 张实际用卡。A4F4 也测 BS24→48；D8 对应 BS12→24，使每个配对的总序列容量和客户端并发保持 96/192。这一组直接判断 A4F4 的部署收益，无需先重跑 A4F2 双机矩阵。

| ID | 布局与 Decode 配置 | 每 A/D DP 的 BS | 总容量 / 客户端并发 C | 每轮短测 / 正式确认请求数 | 含 P2 用卡 | 直接比较 |
|---|---|---:|---:|---:|---:|---|
| K1 | 单机 P2+A4F2 | 24 | 96 | 384 / 1152 | 8 | 与 K2 比增 batch；与 K3 比部署 |
| K2 | 单机 P2+A4F2 | 48 | 192 | 768 / 1152 | 8 | 与 K4 比相同负载下的部署 |
| K3 | 单机 P2+D4 | 24 | 96 | 384 / 1152 | 6 | K1 的原生参照 |
| K4 | 单机 P2+D4 | 48 | 192 | 768 / 1152 | 6 | 原生也增加 batch，参照 K2 |
| K5 | 两机 P2 / A4F4 | 24 | 96 | 384 / 1152 | 10 | 与 K6 比增 batch；与 K7 比等资源 |
| K6 | 两机 P2 / A4F4 | 48 | 192 | 768 / 1152 | 10 | 增 batch 与 F4 的组合，与 K8 配对 |
| K7 | 两机 P2 / D8 | 12 | 96 | 384 / 1152 | 10 | K5 的等资源原生参照 |
| K8 | 两机 P2 / D8 | 24 | 192 | 768 / 1152 | 10 | K6 的等资源原生参照 |

建议执行顺序为 **K3→K1→K4→K2；K7→K5→K8→K6**，先确认小 batch 路径，再增加 batch；预检出现执行异常时按第 8.12.2 节定位，不继续跑完整矩阵。K1/K2 与 K3/K4 卡数不同，报告总 Q 和实际每卡 Q；K5/K7、K6/K8 才是相同用卡数的直接对照。

A/F 优先同机，P→D 的 KV 跨机在 K5–K8 中保持一致，P 机器不混放 F。双机 Decode 卡位示例为 A4：0–3、F4：4–7，或 D8：0–7；单机采用上面的 P2/A4/F2 卡位。按实际 NPU 互联、NUMA 和网卡亲和性选取并冻结，记录真实链路。单机 K1/K2 与双机 K5/K6 可作完整部署比较，不能把全部差值归因于 F2→F4；需要这种归因时才补同布局 A4F2。

上表的总容量**不等于实测每步 batch**。每个配对点先固定客户端并发 `C=96` 或 `192`，请求数和输入集合也一致，采集各 DP 实际 running/waiting 与 per-step token。P 供给或客户端不足时，C 不会全部变成生成中序列。需要增加 C 来填满时，双方另建相同 C 的对照，不能只给 AFD 增压。

BS24→48 的容量扫描同时将 C 从 96 提到 192；只改 `max_num_seqs` 而仍用 96 并发，不能填满 A DP4×BS48。跨档位的 Q/TPOT 变化用于观察扩展性，不称为同负载提速；同负载部署收益分别看 K1/K3、K2/K4、K5/K7、K6/K8。

**先用少量请求看趋势。** 本节每个点 `check` 通过后即可进入筛选，不要求旧流程的四组 BS8 全部重做。K1–K8 保持输入/输出 1024/1024，设 `PERF_TIER=screen`、`PHASE=measure`，关闭诊断后各做 `C` 条预热、`4×C` 条测量、1 轮。BS24/C96 测 384 条，BS48/C192 测 768 条；测量请求数比原来的 `6×C` 减少三分之一，不含预检、预热、加载和 Graph 编译，不能直接据此承诺总用时减少三分之一。该下限根据上面的 K1 快慢阶段反馈调整。

短测保持目标 BS 和并发不变；不将总请求数缩到 C 以下，否则客户端无法提供目标并发。只测 C 或 2×C 条可检查链路和早期趋势，不能据此宣布当前 AFD 满载性能已经改善。`4×C` 仍不保证稳态：必须查看 running、per-step token、Graph replay 和吞吐曲线，确认覆盖较快和较慢阶段；窗口不足时将该 AFD/native 配对统一增至 `6×C`。短测差距较小、波动或双方结论矛盾时不直接淘汰。先保留输出 1024，避免同时缩短输出改变 Prefill/Decode 占比。

短测用于回答“执行异常是否消失、增 batch 是否有改善趋势、哪个配对值得继续”。验证某次调整的前后效果时，要在相同 ISL/OSL、BS/C、请求数及测量开关下重跑对照，只改变待验证项；旧 16K 结果不能与新 1K 短测直接计算提速。

**正式确认只留给入选配对。** 设 `PERF_TIER=confirm`，每轮请求数为 `max(1152, 6×C)`、至少 3 轮；核心 C96/C192 均测 1152 条，条件补测 C384 时为 2304 条。双方冻结相同请求数和生成参数，核对输入集合，不能仅凭 seed 相同认定不同请求数的输入逐条一致。最终只选一组 AFD/native 将输入改为 16384、输出仍为 1024、沿用原 BS/C 正式确认；需要发布 1K 收益或短测结论不清时，仅对入选配对追加相同 1K 负载的正式测量，不默认要求两种长度都各三轮。不要求 8 点全部各三轮或全部再跑 16K。优先选 K5/K7 或 K6/K8；单机更合适时可选 K1/K3 或 K2/K4，并保留卡数差异说明。

短输入慢不自动否定长上下文价值；没有短输入优胜点时，也选一个 KV 容量有潜力的 AFD/native 配对做 16K 确认，再决定是否继续。BS48 预检若 OOM/预算不足，记录容量边界；可将该 AFD/native 配对降为 A 侧 BS32 对应的 C=128，D4 用 BS32、D8 用 BS16。不能把失败点当性能结果，也不只降低一侧的请求负载。

**执行入口。** 按第 3 节准备实际 site/node 环境、HIXL 设备资源和端口。每台机器的每个新终端都从下面加载环境，先将 `SITE_FILE`、`NODE_FILE` 改为本机实际文件路径，再选择一个 `SCENARIO`。本块同时支持上述单机/双机点，不再执行第 8.2/8.4 节的固定赋值。双机的 `site.env` 保持一致，其中 `PREFILL_HOST_IP` 指向机器 P、`ATTENTION_HOST_IP` 指向机器 D；`node.env` 使用各自本机值。单机点会将角色 IP 统一覆盖为本机 `HCCL_IF_IP`。不要加载旧 run 的归档 env。

```bash
set -euo pipefail
# 替换为第 3.1 节已填写并核对的本机实际文件路径。
export SITE_FILE=/data/validation/config/site.env
export NODE_FILE=/data/validation/config/node.env
source "$SITE_FILE"
source "$NODE_FILE"
test -f "$CANN_ROOT/set_env.sh"
source "$CANN_ROOT/set_env.sh"
source "$VENV_ROOT/bin/activate"
export PYTHON_BIN="$VENV_ROOT/bin/python" VLLM_BIN="$VENV_ROOT/bin/vllm"
export RECIPE="$AFD_PLUGIN_ROOT/recipe/npu/P2pHcclAFDConnector/deepseek_v4"

export SCENARIO=K2          # 核心点 K1..K8，按表选择
export ISL=1024 OSL=1024 PHASE=check
export PERF_TIER=screen     # screen：少请求、一轮；confirm：正式三轮
export SCREEN_WAVES=4       # K1 后续慢阶段不能由首 2×C 的结果代表
case "$SCENARIO" in
  K1) export TOPOLOGY=single POINT=A4F2 DECODE_BS=24 ;;
  K2) export TOPOLOGY=single POINT=A4F2 DECODE_BS=48 ;;
  K3) export TOPOLOGY=single POINT=D4 DECODE_BS=24 ;;
  K4) export TOPOLOGY=single POINT=D4 DECODE_BS=48 ;;
  K5) export TOPOLOGY=cross POINT=A4F4 DECODE_BS=24 ;;
  K6) export TOPOLOGY=cross POINT=A4F4 DECODE_BS=48 ;;
  K7) export TOPOLOGY=cross POINT=D8 DECODE_BS=12 ;;
  K8) export TOPOLOGY=cross POINT=D8 DECODE_BS=24 ;;
  *) printf 'Unknown SCENARIO\n' >&2; exit 2 ;;
esac
# 核心点保持上述 BS；只有条件补测时在此覆盖 DECODE_BS，随后容量自动重算。
# 例如 C=384：A4F2/A4F4/D4 用 DECODE_BS=96，D8 用 DECODE_BS=48。
export PREFILL_DP_SIZE=2 PREFILL_TP_SIZE=1
export PREFILL_DEVICES=0,1 PREFILL_MAX_NUM_SEQS=48
export PREFILL_MAX_NUM_BATCHED_TOKENS=4096
export TENSOR_PARALLEL_SIZE=1 ENABLE_PD=1 ENABLE_MTP=0 ENABLE_DSPARK=0
export MODEL_PATH="$FLASH_MODEL_PATH" EXECUTION_MODE=full-decode-only
export USE_AFD=0 U_BATCHES=1 ATTENTION_RANKS=4 FFN_RANKS=2
export ATTENTION_DEVICES=0,1,2,3 FFN_DEVICES=4,5
case "$POINT" in
  A4F2) export USE_AFD=1 U_BATCHES=2 ;;
  A4F4) export USE_AFD=1 U_BATCHES=2 FFN_RANKS=4 FFN_DEVICES=4,5,6,7 ;;
  D4) ;;
  D8) export ATTENTION_RANKS=8 ATTENTION_DEVICES=0,1,2,3,4,5,6,7 ;;
  *) printf 'Unknown POINT\n' >&2; exit 2 ;;
esac
if [[ "$TOPOLOGY" == single ]]; then
  export ATTENTION_DEVICES=2,3,4,5 FFN_DEVICES=6,7
  export PREFILL_HOST_IP="$HCCL_IF_IP" ATTENTION_HOST_IP="$HCCL_IF_IP"
fi
export CAPACITY=$((ATTENTION_RANKS / TENSOR_PARALLEL_SIZE * DECODE_BS))
export MAX_MODEL_LEN=$((ISL + OSL))
export MAX_NUM_SEQS="$DECODE_BS"
case "$PERF_TIER" in
  screen) export REQUEST_COUNT=$((SCREEN_WAVES * CAPACITY)) PERF_REPEATS=1 ;;
  confirm)
    export REQUEST_COUNT=$((6 * CAPACITY)) PERF_REPEATS=3
    if ((REQUEST_COUNT < 1152)); then export REQUEST_COUNT=1152; fi
    ;;
  *) printf 'Unknown PERF_TIER\n' >&2; exit 2 ;;
esac
# 短测窗口不足时，在此对 AFD/native 双方统一改为 6 * CAPACITY。
export FFN_SEQ_CAPACITY=$(((CAPACITY + FFN_RANKS - 1) / FFN_RANKS))
export ATTENTION_MAX_NUM_BATCHED_TOKENS=4096 FFN_MAX_NUM_BATCHED_TOKENS=8192
export MAX_CUDAGRAPH_CAPTURE_SIZE= CUDAGRAPH_CAPTURE_SIZES=
export FFN_HOST_IP="$ATTENTION_HOST_IP" AFD_HOST="$ATTENTION_HOST_IP"
export PROXY_HOST_IP="$PREFILL_HOST_IP" API_PORT="$ATTENTION_API_PORT"
export BASE_URL="http://$PROXY_HOST_IP:$PROXY_PORT"
export ATTENTION_URL="http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT"
export PERF_CASE="${SCENARIO}-${POINT}-b${DECODE_BS}-n${CAPACITY}-${PERF_TIER}"
export CASE="$PERF_CASE" CYCLE=1
export PERF_REQUEST_RATE=inf
unset SLO_TTFT_MS SLO_TPOT_MS
case "$PHASE" in
  check) export AFD_HCCL_STAGE_DIAGNOSTICS=1 ;;
  measure) export AFD_HCCL_STAGE_DIAGNOSTICS=0 ;;
  *) exit 2 ;;
esac
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0 VLLM_LOGGING_LEVEL=INFO
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=1 AFD_HCCL_GRAPH_U2_HYBRID_DAG=1
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1 AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
export GLOO_SOCKET_IFNAME="$NIC_NAME" HCCL_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,$PREFILL_HOST_IP,$ATTENTION_HOST_IP"
export no_proxy="$NO_PROXY"
```

`FFN_SEQ_CAPACITY=ceil(N/F)` 只是本表 TP1 普通 Decode 的接收序列容量起点，不是路由后每专家 token 数，也不替代 token/Graph buffer 检查。DSpark、padding 和 F>A 时按真实 shape 重新验收。P 的 max seqs/token budget 在各组固定；P2 无法供满时先标记供给受限，扩大 P 必须双方一起调整并重新计费。

单机在当前终端创建 run；双机使用相同 `SCENARIO/ISL/PHASE/PERF_TIER/RUN_ID`，由协调终端生成后复制到另一机。各机单独创建 `RUN_ROOT`，保存本机 env、runtime、日志和 PID。筛选使用 `PHASE=measure PERF_TIER=screen`；正式确认使用 `PHASE=measure PERF_TIER=confirm`，16K 时再将 `ISL` 改为 `16384`，重新创建 run。按第 8.7 节执行一次预热和 `PERF_REPEATS` 轮测量；切勿再手工额外循环三次。不要各机独立生成时间戳，也不要重新执行第 8.2/8.4 节的固定赋值。

```bash
# 在一个协调终端生成；将打印的完整 export 行复制到另一机执行。
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${TOPOLOGY}-$PERF_CASE-i$ISL-$PHASE"
printf 'export RUN_ID=%q\n' "$RUN_ID"
# 两机在 RUN_ID 一致后分别执行。
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
cp "$SITE_FILE" "$RUN_ROOT/site.env"
cp "$NODE_FILE" "$RUN_ROOT/node.env"
```

启动沿用第 8.5 节命令：K1–K4 的 P、A/F 或 D、Proxy 全部在本机依次启动；K5–K8 则 **P 仅在机器 P 启动，A/F 或 D 仅在机器 D 启动，最后回机器 P 启动 Proxy**。双机均定义 `wait_http`；只在机器 P 运行 benchmark，两机分别采集 NPU 数据，不要在两侧重复发压。第 8.5 节原生命令按 `ATTENTION_RANKS/TP` 推导 DP。ready 和 shutdown 的 rank 数按实际拓扑调整，禁止沿用固定“四个 A/D、两个 F”。停服分别在保存 PID 的本机执行；按第 9 节归档。

原生候选 Graph/U1、AFD Graph/U2 的差异保留记录，结论称为部署收益。要隔离 AFD 分离本身，另验收原生 U2 或双方 eager/U1 对照。D8/新 A/F 配比均为待实机验收候选，不能仅凭 CLI 合法写成已经支持或性能通过。

#### 8.12.4 条件补测：只有核心结果提出问题时才追加

下表不是第二份必测矩阵。优先完成上面的 batch 主线、等资源比较和一对 16K 确认。

| 观察结果 / 要回答的问题 | 最小追加场景 |
|---|---|
| BS48 的 Q 仍明显增长，TPOT、KV/HBM 有余量 | 在入选拓扑上追加 BS96，配对总容量 C=384；D4 用 BS96、D8 用 BS48，先预检后测；无需遍历 BS32/64/96 全部档位 |
| BS48 延迟过高、BS24 合格，需要找到折中 | 只补 A 侧 BS32、C=128 和对应原生点；不是默认必测 |
| 需要单独量化 F2→F4 的影响 | 只补一个选定 BS/C 的双机 A4F2，对比已有同布局 A4F4；无需完整重跑双机 A4F2/D4 |
| F 计算在关键路径，A 明显有余量 | 可选 A2F4，按同总容量验证资源重分配；先验收 F>A 路径 |
| A 时间/KV 成为限制，F 尚有余量 | 再测 A8F2 BS24 对 A4F2 BS48（N 都为 192），验证增加 A 的价值 |
| A、F 都接近容量上限 | 才考虑 A8F4；同样先做同 N 对照，再测自身容量上限 |
| F 等待 A，或低利用率且通信/同步占比高 | 优先处理执行路径、网络或重叠；继续增加 F 通常无助于该瓶颈 |
| 需要 DSpark 或 32K/更长上下文部署 | 只在入选拓扑和原生配对上追加，不重开 8 点全矩阵 |

可选 **A2F4 BS48/96** 对应 N96/192，与 A4F2 同为 6 张 Decode 卡，可探索把两张 A 转给 F 的效果。它使用 F>A 的 fan-out/切分路径，每 A 负担更重；先过请求、Graph/U2、padding、输出质量和正常停服预检，再做性能测试。该点不能由 A4F4 已跑通推定可用。

A8F2 不能与 P2 一起放在上述机器 D。若只有两机，改为机器一 A8，机器二 P2+F2；**此时 A4F2 BS48 也要按同样的跨机 A/F 布局重测**，才与 A8F2 BS24 配对，避免把跨机通信变化归因于 A 卡数量。三机可固定 P、A、F 各一机后扫配比。当前 recipe 每个角色最多 8 卡、各角色在单机启动，不直接支持一个角色跨多机扩成 A16/F16。

TP2 放到后续：A4F4 TP1 为 A DP4/F DP4，BS24 对应 N96；TP2 为 A DP2/F DP2，需要 BS48 才对应 N96，F 均为 EP4。比较 N192 时分别用 BS48/96。P 的 TP/KV 配置也须相容（当前 helper 要求 P TP≥A TP）；将 P2 从 DP2/TP1 改为 DP1/TP2 属于新的配对实验，双方同步调整。先证明 Attention 的 TP 加速能抵消通信，不能因为每卡专家相同就认为两种并行方式等价。

#### 8.12.5 正式测量与“有收益/不严重劣化”的判定

- 每个新点先 `check`；K1–K8 各做 C 条预热、4×C 条测量、1 轮筛选，须覆盖本批观察到的慢阶段，不据此写正式收益，不要求所有点先各三轮。入选 AFD/native 配对再按第 8.7 节做 `max(1152, 6×C)` 条/轮、至少 3 轮，交替安排配置并记录冷启动/同进程重复关系。筛选或正式确认没有稳定窗口时，双方一起增加请求数；保存统一的开始/结束墙钟、每轮请求集及波动范围。
- 先在同样输入/输出、C、请求数、DSpark 状态下比较 Q、TTFT/TPOT 的 P50/P95/P99、真实 batch、每角色 HBM/KV、通信/计算等待。一次短 profile 仅用于归因；正式测量关闭 profiler 和 stage 诊断。失败 0、真实 token 数正确、Proxy 和全部 D/A running/waiting 归零、正常停服是每个点的有效性条件；精度验收另行完成。
- **探索阶段可暂用**“同 C 下 Q≥原生的 90%，且 TPOT P95≤原生的 110%”筛出不严重劣化点，同时检查 TTFT 和业务 SLO；这是建议的筛选线，不是既有交付门槛，也不是用户已冻结的业务目标。正收益应超过重复测量的波动，并在相同 SLO/资源下成立。
- 用卡数同时报 Decode 和全链路：A4F2 为 6/8，A4F4 为 8/10，D4 为 4/6，D8 为 8/10，后一个数均含 P2。两机各预留整机时，还报 `Q/16` 和实际预留成本。A4F4 相对 D4，总 Q 至少达到 2 倍才持平 Decode 每卡效率；包含 P2 时阈值是 `10/6≈1.67` 倍。与 D8 相比才是相同实际用卡的直接比较。以上比例是核算关系，不是性能预测。D8 有 8 张卡参与专家计算，A4F4 只有 F4，故 A4F4 并不天然占优；应实测 KV 容量、A/F 重叠或共享 F 的价值能否补偿分离和通信成本。
- 最后冻结业务 TTFT/TPOT SLO，在同一请求集、相同有限 RPS 下逐级增压；双方各自调到合法最佳 BS/C 后再比较最大可持续流量。既要报告相同 C 的对照，也要单列各自最佳容量。不要要求 D8 保持 BS12/24 而限制它进一步调优。

第 8.6 节 `run_perf_batch` 支持 `PERF_REQUEST_RATE` 和可选的 `SLO_TTFT_MS/SLO_TPOT_MS`：吞吐筛选保持 `PERF_REQUEST_RATE=inf`；SLO 实验在新 run 中把它设为双方相同的有限 requests/s，并将两个 SLO 变量设为事先冻结的毫秒值，再调用同一个函数。固定相同 `CAPACITY/REQUEST_COUNT`，检查客户端限并发有无推迟发送，记录实际到达速率；限流导致到达达不到目标时，不称为目标 RPS 已通过。需要 E2E SLO 时再加入 `e2el` 条件。

`request_goodput` 是同时满足门槛的**成功请求数/总秒数**，不是 tokens/s；另报达标请求比例和失败率。官方参数定义见 [vLLM 0.23 benchmark 源码](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/benchmarks/serve.py)，现场以冻结 commit 的 `--help` 和结果字段为准。EP/DP/TP 的概念可核对 [vLLM EP 文档](https://github.com/vllm-project/vllm/blob/v0.23.0/docs/serving/expert_parallel_deployment.md)；其中 CUDA 通信后端建议不直接套用到 Ascend。

DSpark 只在筛出的 1–2 个拓扑上追加：双方冻结同一 target、draft 和 block，确认 A 的 draft eager、F 无 drafter，重算 capture/buffer，记录 proposed/accepted 增量。以真实输出 Q、TPOT 和 SLO goodput 判优，不能以接受率代替收益。最终至少交付“同负载比较、等资源比较、各自最佳 SLO 容量”三张表，并注明未找到收益点时的瓶颈与适用范围。

#### 8.12.6 10-23 后的下一步：三个诊断点，定位快慢阶段后再扩拓扑

**10-24 更新：本节保留上一轮 S1–S3 的执行定义。现场实际采用 BS48/C192，S2 runtime 已证实 overlap=0；S4 按用户要求暂不测试。当前操作转到第 8.12.7 节，不重新执行下面的 BS24 三点矩阵。**

**先做下面三个 A4F2 点，不重跑 K1–K4 全矩阵。** 固定同一版本、同一台 A5、P0–1/A2–5/F6–7、TP1、BS24、C96、输入/输出 1024/1024，关闭 DSpark/MTP。每点先 smoke，再预热 96 条、测量 384 条、一轮；没有覆盖慢阶段时将本次对照的请求数统一增至 576。每个点重新冷启动全部角色，保持相同的预检/预热顺序。

| 诊断点 | `EXECUTION_MODE` | `U_BATCHES` | eager overlap | 五个 Graph 优化开关 | 与上一点比较 |
|---|---|---:|---:|---|---|
| S1 | `eager` | 1 | 0 | 全 0 | 不拆 U2 的 AFD 参照 |
| S2 | `eager` | 2 | 0 | 全 0 | 只增加 U2 拆分 |
| S3 | `full-decode-only` | 2 | 0 | 全 0 | 启用 recipe 的 Graph 执行路径 |
| S4（条件追加） | `full-decode-only` | 2 | 0 | 全 1 | 与 S3 比较整组 Graph 多流优化 |

五个优化开关全 0 **仍然可以执行 Graph**。eager/U2 的 recipe 默认 overlap=1，故本表显式固定为 0；不能直接使用默认值来做 S2/S3 对照。S4 一次打开五个开关，只能判断整个组合的影响，不能归因于某一个开关。历史 K1 缺少完整运行开关/版本冻结，不代替本轮 S4 的受控对照。

**步骤 1：加载公共环境，再覆盖诊断变量。** 在同一个启动终端执行第 8.12.3 节从 `set -euo pipefail` 到 `export no_proxy` 的完整配置块，其中改成 `SCENARIO=K1`、`PHASE=measure`、`PERF_TIER=screen`、`SCREEN_WAVES=4`。该块包含实际 `SITE_FILE/NODE_FILE` 的 `source`、CANN 和 venv 加载。先不执行其后创建 RUN_ID 的代码，不再加载第 8.2/8.4 节的固定配置。随后执行：

```bash
# 每次新终端均先完成上面的公共环境加载；依次选择 S1、S2、S3。
export DIAG_POINT=S1
case "$DIAG_POINT" in
  S1) export EXECUTION_MODE=eager U_BATCHES=1 GRAPH_OPTIMIZATIONS=0 ;;
  S2) export EXECUTION_MODE=eager U_BATCHES=2 GRAPH_OPTIMIZATIONS=0 ;;
  S3) export EXECUTION_MODE=full-decode-only U_BATCHES=2 GRAPH_OPTIMIZATIONS=0 ;;
  S4) export EXECUTION_MODE=full-decode-only U_BATCHES=2 GRAPH_OPTIMIZATIONS=1 ;;
  *) printf 'Unknown DIAG_POINT\n' >&2; exit 2 ;;
esac
test "$SCENARIO:$USE_AFD:$DECODE_BS:$CAPACITY" = K1:1:24:96
export PHASE=measure PERF_TIER=diagnostic PERF_REPEATS=1
export REQUEST_COUNT=$((4 * CAPACITY))  # 窗口不足时，对照各点统一改为 6 * CAPACITY。
export DBO_DECODE_TOKEN_THRESHOLD=2 DBO_PREFILL_TOKEN_THRESHOLD=12
export AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP="$GRAPH_OPTIMIZATIONS"
export AFD_HCCL_GRAPH_U2_HYBRID_DAG="$GRAPH_OPTIMIZATIONS"
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM="$GRAPH_OPTIMIZATIONS"
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM="$GRAPH_OPTIMIZATIONS"
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER="$GRAPH_OPTIMIZATIONS"
export AFD_HCCL_STAGE_DIAGNOSTICS=0 AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export AFD_NPU_ATTENTION_PROFILER_ENABLE=0 AFD_NPU_FFN_PROFILER_ENABLE=0
export VLLM_LOGGING_LEVEL=INFO
export PERF_CASE="${SCENARIO}-${DIAG_POINT}-b${DECODE_BS}-n${CAPACITY}-${PERF_TIER}"
export CASE="$PERF_CASE"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-single-$PERF_CASE"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
cp "$SITE_FILE" "$RUN_ROOT/site.env"
cp "$NODE_FILE" "$RUN_ROOT/node.env"
```

按第 4.4 节保存本轮 runtime/版本/模型 hashes，另存源码 dirty diff 和本表最终变量。按第 8.5 节顺序启动 P→F+A→Proxy，并保存实际 argv；P/F 的命令前缀仍分别使用 `PREFILL_MAX_NUM_SEQS` 和 `FFN_SEQ_CAPACITY`，不能把 F 的容量改成 A 的 24。核对 S1/S2 实际 `enforce_eager=True`；S3/S4 A 的 Graph max=24、捕获桶包含 24，F 两个 U2 stage 的满载 key 为各 A rank 12 tokens。在线有非均匀 batch 时另记录真实/填充 shape，不能仅凭 max 正确认定全部 replay。

**步骤 2：每点按相同顺序发压。** 定义第 8.6 节的 `run_perf_batch`。本节为诊断筛选，服务 ready 后先执行以下 smoke，通过后执行第 8.7 节的采样、一次预热和一轮测量代码；不要额外跑旧四组矩阵或额外循环三次。

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" --suite smoke \
  --run-id "$RUN_ID" --output "$RUN_ROOT/smoke"
```

任一点出现设备错误、hang、失败请求或 token 数不符，先按第 9 节保存现场，不把失败点当作慢点继续计时。每次请求结束核对 drain；结束后按第 8.8 节正常停服并确认两个 F rank shutdown。S1–S3 是三个诊断点，尚不是 AFD/native 收益验收。K3 的历史稳定数据可作背景参照；版本变化或需要计算新收益时，必须补同版本、同请求数的 K3，不能拿旧 576 条直接与新 384 条算提速。

**步骤 3：检查是否仍有约 250 ms 的慢请求，不能只看总 Q。** 测量结束后执行下面的分组检查。`200 ms` 只是根据当前快慢簇设置的诊断阈值，不是业务 SLO；它统计每条请求的平均 TPOT，不能定位单次 token 的切换时刻。

```bash
"$PYTHON_BIN" - "$RUN_ROOT/measure-1.json" "$CAPACITY" \
  > "$RUN_ROOT/arrival-cohorts.txt" <<'PY'
import json
import statistics
import sys
from pathlib import Path

SLOW_TPOT_MS = 200.0
data = json.loads(Path(sys.argv[1]).read_text())
capacity = int(sys.argv[2])
count = data["completed"]
assert count > 0 and data["failed"] == 0
for key in ("start_times", "itls", "output_lens"):
    assert len(data[key]) == count, key
assert all(length > 1 for length in data["output_lens"])
tpot = [1000 * sum(itls) / (length - 1)
        for itls, length in zip(data["itls"], data["output_lens"])]
order = sorted(range(count), key=lambda index: data["start_times"][index])
origin = min(data["start_times"])
print(f'Q={data["output_throughput"]:.2f} tok/s; '
      f'mean/P95 TPOT={data["mean_tpot_ms"]:.2f}/{data["p95_tpot_ms"]:.2f} ms')
for offset in range(0, count, capacity):
    group = order[offset:offset + capacity]
    starts = [data["start_times"][index] - origin for index in group]
    values = [tpot[index] for index in group]
    slow = sum(value >= SLOW_TPOT_MS for value in values)
    print(f'arrivals {offset + 1}-{offset + len(group)}; '
          f'start={min(starts):.2f}..{max(starts):.2f}s; '
          f'mean TPOT={statistics.mean(values):.2f}ms; '
          f'>={SLOW_TPOT_MS:g}ms={slow}/{len(group)}')
PY
cat "$RUN_ROOT/arrival-cohorts.txt"
```

每 C 条是按请求到达顺序分组，不是独立轮次，也不是严格同步的波次。保存每点 Q、TPOT P50/P95/P99、分组均值/慢请求比例、真实 running/token、空白或异常输出。若 S3 较好，才追加 S4 复核默认优化组合；若 S1 已异常且 smoke 不通过，先停止后续点排查基本链路。

| 观察结果 | 下一步优先验证 |
|---|---|
| S1 无慢簇，S2 出现慢簇 | U2 拆分/控制面、两阶段通信与等待；按需另测 eager overlap=1，只与 S2 比较 |
| S2 无慢簇，S3 出现慢簇 | Graph 命中/回退、stage key/padding、图内通信与同步依赖 |
| S3 无慢簇，S4 出现慢簇 | Graph 多流优化组合；保留版本与负载，按依赖关系逐项恢复，`FFN_CROSS_LAYER=1` 要求 `FFN_RECV_STREAM=1` |
| 三点均有慢簇，或只是整体速度不同 | 先拆 F 专家算子/EP、A↔F、CPU 等待；不直接判为 F2 带宽不足 |
| 新点未复现旧 K1 慢簇 | 保留“未复现”；核对版本、运行开关和窗口，必要时补 S4/576 条，不宣布根因已修复 |

这些结果只用于选择下一处观测，不自动证明具体根因。S1 改变了 microbatch 大小，S2/S3 的 Graph 路径也包含编译/缓存等配套行为，不能将全部差值归给一个算子。

**步骤 4：只对有差异的路径单独采 profile。** 当前改用第 8.12.7 节的 S1/S2 配置和窗口检查。旧候选 `SKIP_FIRST=3500/ACTIVE=32` 不再作为当前操作入口；本轮各 rank 步数存在偏差，短窗口可能互相错开。

**步骤 5：按瓶颈选择一个后续拓扑点。** 若证据主要指向 Graph/U2，先修复或选定通过的配置，关闭 profiler/诊断后重跑同负载 K1/K3，再恢复 BS48 对照；不能只保留 S1 就宣布默认 Graph 性能通过。若证据主要指向 F 专家计算/权重读取，优先只做 K5/K7（A4F4 BS24 与 D8 BS12，均 C96），看增加 F 是否改善关键路径，随后才追加 BS48 的 K6/K8。三点均未定因时，先回传上述材料，不直接扩成完整拓扑矩阵。短输入没有收益但路径正常时，仍按第 8.12.3 节保留一对 16K 容量确认。

回传每点的 `runtime.json`、最终开关/实际 argv、四角色日志、`warmup/measure-1` 的 JSON/console/metrics/drain、`arrival-cohorts.txt`、NPU 采样和正常停服证据；profile 点另附完整 trace、触发参数及在线墙钟范围。上述命令仅完成本地代码/语法核对，A5 现场执行与性能结论仍待本轮实测。

#### 8.12.7 当前操作：暂停 S4，用 S1/S2 timeline 定位约两倍耗时

**只新增两个带 profiler 的诊断 run：S1/U1 与 S2/U2。** 已有无 profiler 的测量保留作性能结果，不再重跑 S1–S3 全矩阵；S4、扩 F 卡和增 BS 暂不执行。本节沿用实际测试的 BS48/C192、预热192、测量768、输入/输出1024/1024、P0–1/A2–5/F6–7、TP1。S1/S2 runtime 中的源码和包一致，除运行标识外，仅 U_BATCHES 不同；eager overlap 和五个 Graph 开关均为 0。目标是找出一个完整 Decode step 增加的时间落在哪里，不把两个 U2 stage 当成两份输出 token。

**1. 每个点先加载环境和 BS48 公共配置。** 每次新终端按第 8.12.3 节执行从 `set -euo pipefail` 到 `export no_proxy` 的完整代码块，填写本机实际 `SITE_FILE/NODE_FILE`，保留其中的 CANN/venv `source`。选择 `SCENARIO=K2`、`PHASE=measure`、`PERF_TIER=screen`、`SCREEN_WAVES=4`；K2 对应本轮实际 BS48，S1/S2 仍表示执行模式。先不执行旧的创建目录代码，也不执行第 8.12.6 节的 K1:24:96 检查。随后在同一终端、启动全部角色前执行：

```bash
export DIAG_POINT=S1       # 第二个 run 改为 S2，全部角色冷启动。
case "$DIAG_POINT" in
  S1) export U_BATCHES=1 ;;
  S2) export U_BATCHES=2 ;;
  *) printf 'Only S1/S2 profile runs are required\n' >&2; exit 2 ;;
esac
test "$SCENARIO:$TOPOLOGY:$USE_AFD:$DECODE_BS:$CAPACITY" = K2:single:1:48:192
test "$FFN_SEQ_CAPACITY:$MAX_MODEL_LEN:$ISL:$OSL" = 96:2048:1024:1024
export EXECUTION_MODE=eager PHASE=measure PERF_TIER=profile PERF_REPEATS=1
export REQUEST_COUNT=$((4 * CAPACITY))
export DBO_DECODE_TOKEN_THRESHOLD=2 DBO_PREFILL_TOKEN_THRESHOLD=12
export AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0 AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0 AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
export AFD_HCCL_STAGE_DIAGNOSTICS=0 AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export VLLM_LOGGING_LEVEL=INFO
export PERF_CASE="${SCENARIO}-${DIAG_POINT}-b${DECODE_BS}-n${CAPACITY}-${PERF_TIER}"
export CASE="$PERF_CASE" CYCLE=1
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${TOPOLOGY}-${PERF_CASE}"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
cp "$SITE_FILE" "$RUN_ROOT/site.env"
cp "$NODE_FILE" "$RUN_ROOT/node.env"

# 首轮留出足够窗口，随后只分析六个 rank 共同覆盖的 3–5 个完整 step。
export PROFILE_SKIP=1500 PROFILE_ACTIVE=256
for role in ATTENTION FFN; do
  export "AFD_NPU_${role}_PROFILER_ENABLE=1"
  export "AFD_NPU_${role}_PROFILER_RANKS=all"
  export "AFD_NPU_${role}_PROFILER_SKIP_FIRST=$PROFILE_SKIP"
  export "AFD_NPU_${role}_PROFILER_WAIT=0"
  export "AFD_NPU_${role}_PROFILER_WARMUP=1"
  export "AFD_NPU_${role}_PROFILER_ACTIVE=$PROFILE_ACTIVE"
  export "AFD_NPU_${role}_PROFILER_REPEAT=1"
  export "AFD_NPU_${role}_PROFILER_WITH_STACK=0"
done
export AFD_NPU_ATTENTION_PROFILER_DIR="$RUN_ROOT/profile/attention"
export AFD_NPU_FFN_PROFILER_DIR="$RUN_ROOT/profile/ffn"
mkdir -p "$AFD_NPU_ATTENTION_PROFILER_DIR" "$AFD_NPU_FFN_PROFILER_DIR"

# 补充 runtime.json 未覆盖的最终 profiler/诊断变量，限定输出键名。
for name in RUN_ID CASE SCENARIO DIAG_POINT TOPOLOGY EXECUTION_MODE U_BATCHES \
  DECODE_BS CAPACITY REQUEST_COUNT ISL OSL MAX_NUM_SEQS FFN_SEQ_CAPACITY \
  AFD_HCCL_EAGER_U2_STREAM_OVERLAP AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP \
  AFD_HCCL_GRAPH_U2_HYBRID_DAG AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM \
  AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER \
  AFD_HCCL_STAGE_DIAGNOSTICS AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS \
  ${!AFD_NPU_ATTENTION_PROFILER_@} ${!AFD_NPU_FFN_PROFILER_@}; do
  printf '%s=%q\n' "$name" "${!name}"
done > "$RUN_ROOT/profile.env"
```

当前 [profiler.py](../../afd_plugin/compat/npu/profiler.py)采 CPU+NPU、Level2、record_shapes=True，输出 Text 格式；没有开启 AI Core 带宽计数器。参数在 runner 初始化时读取，**必须在 A/F 启动前设置并重启进程**；这里不调用 `/start_profile`，也不向已启动的进程追加 export。`SKIP_FIRST/ACTIVE` 是各 runner 的 `profiler.step()` 调用数，不是请求数、stage 数或秒数；A 每次 execute_model 调一次，F 的入口位置不同，不能直接按相同 step 编号跨 rank 对齐。

本轮预热结束时，S1 四个 A 的 iteration 计数约 1246/1075/1075/1075，S2 约 1170/1127/1059/1061。这些只是选择窗口的参考，**不是 profiler.step 的精确计数**。1500 是先采测量早期的候选，256 步用于容纳已观察到的 rank 偏差，采集时间和文件大小会增加；按原始 TPOT 粗估约 34/64 秒，不能保证此时长。最终以六卡共同有效区为准，不保证同值自动同时触发。若完全未覆盖在线测量或交集不足，先记录每侧实际时间窗，只调整相应 A/F `SKIP_FIRST` 重采；不同时改变执行开关。不要恢复默认只采 rank0，也不把这里的 rank 当物理卡号。

**2. 启动、预热和发压。** 按第 4.4 节生成本点 `runtime.json`，按第 8.5 节 P→F+A→Proxy 启动；F 命令前缀仍用 `MAX_NUM_SEQS="$FFN_SEQ_CAPACITY"`、A 用48，P仍用48。检查 A/F 均 `enforce_eager=True`，日志有四个 A、两个 F 的 `profiler enabled`（以实际 rank/PID 为准）。服务 ready 后，按第 8.12.6 节步骤2执行同样的 smoke，通过后只定义第 8.6 节的 `run_perf_batch` 函数；然后执行第 8.7 节的采样/发压代码，得到一次 warmup192 和一次 measure768。不再执行旧 BS8 预检矩阵，不额外循环三轮。profile 点的 Q/TPOT 标为“带采集开销”，不替换现有无 profiler 的性能结果；若采集时快慢差异消失或发生设备错误，也按实际记录。

测量和 drain 完成后检查产物，再按第 8.8 节正常停服；保留完整原始 profiler 目录，不仅拷贝几个 CSV：

```bash
rg -n 'profiler enabled|enforce_eager|role=|non-default args:' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log" > "$RUN_ROOT/profile-startup-check.txt"
find "$RUN_ROOT/profile" -type f \
  \( -name 'trace_view.json' -o -name 'kernel_details.csv' \
     -o -name 'operator_details.csv' -o -name 'communication*.json' \
     -o -name 'profiler_info*.json' \) -print > "$RUN_ROOT/profile-files.txt"
cat "$RUN_ROOT/profile-files.txt"
```

预期有 A4/F2 对应的六组进程产物，通常各含 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`、kernel/operator CSV 和 profiler 元数据。用 PID、rank、Device ID 核对角色；只有目录存在、只有 CPU 事件或缺 NPU kernel 不能判为采集完成。若只有原始 PROF/FRAMEWORK 文件、尚无解析交付件，先检查解析错误并按现场 torch-npu 配套方法离线解析，不把空结果解释成设备空闲。

**3. 打开并选择可比较窗口。** 在 MindStudio Insight 的系统调优中导入本 run 的完整 profile 数据，查看 Timeline；保留 A/F 原目录和时间戳，按工具版本支持的追加导入方式查看对应进程，不假定 A/F 不同通信域会自动合并成一个完整集群。文件说明和界面操作见[官方系统调优指南](https://github.com/Ascend/msinsight/blob/master/docs/zh/user_guide/system_tuning.md)。首轮按以下顺序检查：

1. 对齐 `measure-1.wallclock-begin/end.txt` 与 profiler 元数据时间，排除启动、预热、trace 导出和最后排空。日志 running 是稀疏采样，还需结合 trace 的实际 token shapes；选择四个 A 持续接近48、无明显补充请求的区间。
2. 六个 rank 必须在同一墙钟范围内有实际事件。截取共同覆盖区内至少3–5个完整 Decode step；S1 计一个 stage，S2 计 stage0+stage1 的完整 step。不要把不同请求平均 TPOT 当作单个 kernel 时间，也不要给 S2 除以2。
3. 从 A 的计算→A→F send/recv→F MoE/EP→F→A→A 后续计算追一条依赖链，按通信对端、通信域、顺序和关联 ID 配对。`ProfilerStep#编号` 是本地编号；不能仅按相同编号或每卡各取第一个事件认定是同一步。跨流的 CPU/NPU flow 和发送/接收配对也要核对。
4. 若 trace 没有 layer/stage/真实 token 字段且无法可靠配对，先标记缺项、回传原始 trace。必要时再做单独的 eager 边界打点诊断；不要凭算子总表补出不存在的 stage 毫秒数，不先打开 DEBUG 或强制 compute sync 改变时序。

**4. 用新增时间归因，而不是只看哪一行最长。** 对每个完整 step 记录下表，在3–5步上报中位数和范围；S1/S2 使用同样的计时边界和采集设置。六卡总耗时、CPU父子事件以及重叠的通信/计算不能直接累加成一步墙钟时间。先用实际关键路径解释 S2 比 S1 增加的时间；原始无 profiler 的差值约117.82 ms，仅作背景。

| 项目 | 如何观察 S1 与 S2 | 何种证据支持该方向 |
|---|---|---|
| A 本地计算 | Attention、Gate、norm等 NPU 算子的次数、shape、实际执行时长 | F 在等 A，且 A 的两段计算增加能够解释关键路径差值；不能只检查 F |
| F 专家计算 | GroupedMatmul/量化/激活/重排等实际 NPU kernel，S2 两 stage 合起来与 S1 比 | 每段约半量 token，但每段专家算子时长仍接近 U1，且新增计算确实阻塞下一步；仅能定位到 FFN 执行，权重带宽是否饱和还需单独计数器/代表形状验证 |
| F 内部 EP | dispatch/combine 的通信域及 F0/F1 的对应任务，区别于 A↔F P2P | F 内 EP 的未重叠传输/等待显著增长，或另一张 F 是持续落后的生产者；融合算子含计算/通信时保留混合归因 |
| A↔F P2P | 按 A/F 对端识别 send/recv，核对传输量、次数、ready时间与未重叠区间 | 额外传输占据关键路径且对端已经 ready，才优先怀疑传输/消息粒度；recv 很长但发送方尚未完成计算，优先追发送方 |
| Host/同步等待 | CPU下发空隙、synchronize/event/stream wait，与同一时刻所有相关 NPU 流对照 | 所需依赖已完成仍延迟下发/放行，或显式 barrier 形成关键路径空泡；CPU synchronize 覆盖设备正在计算的区间只是等待表象，不能再把同一时间算一遍 |

建议回传一张表：`完整 step ms、A计算、F计算、F内部EP、A↔F、未解释空泡、重叠区间`，分别列 S1、S2、差值，以及每项的起止事件/依赖证据；这些是诊断统计项，未形成互斥关键路径划分前不强制相加等于总时长。`communication.json` 的传输与等待字段存在时辅助区分，字段缺失则明确记录；不能仅因 HCCL 总时长高就断定网络带宽差。现有 worker 每步有 `torch.npu.synchronize()`，S1/S2 都会执行，发现长 synchronize 本身不证明它就是新增开销的根因。

**5. 回传和后续。** 按第 9 节归档两个完整 RUN_ROOT，至少含完整 profile、runtime.json、profile.env、实际 argv/四角色日志、smoke、warmup/measure JSON/metrics/drain/wallclock、NPU采样、共同窗口范围和正常停服记录。先完成这对证据；若结果主要是缺少重叠，再只追加一个 eager/U2 overlap=1 点与 S2 比，不重开 S4/拓扑矩阵。判断修复或调参收益时，关闭两个 profiler、stage diagnostics 和 DEBUG 后，复测选定配置；timeline 归因不替代无 profiler 的性能确认。本节仅核对本地接口和脚本语法，尚未在现场 A5 执行采集。

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
第 8.2–8.10 节性能对照及第 8.11 节补测分别归档，明确主机数、角色分布、原生基线状态、KV 来源及资源差异；不自动关闭下表三机验收项。纯 Decode 流程未验收时记录 pending，不能用端到端数据冒充 Decode-only 结果。

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
