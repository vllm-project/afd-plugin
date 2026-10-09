# DeepSeek-V4 AFD A5 1030 合并验证指导书

更新日期：2026-10-09。本文合并旧 A5 dSpark/PD 指导与[1030 执行计划](DEEPSEEK_V4_AFD_1030_EXECUTION_PLAN_ZH.md)，是本期现场操作入口。
执行计划规定交付范围和日期，本文规定现场命令、验证顺序、证据与失败处理。
本文是待执行指导，不代表组合已经通过 A5 验收。

**执行标记**：`【本地已完成】` 为代码/配置的离线检查；`【A5 手工执行】` 为需要你在 A5 实际环境完成的操作。当前从[手工交接清单](DEEPSEEK_V4_AFD_A5_MANUAL_HANDOFF_ZH.md)的 A5-01 开始，先完成环境与 standalone 回归，不直接启动完整 C5。

## 1. 合并后的范围与门禁

| 项目 | 本期统一口径 |
|---|---|
| afd-plugin | 唯一交付分支 `feat/dsv4-afd-phase1-delivery`；`02d0779a2b11d29e3635f3dc9ec3ce5a606bf583` 是回归起点，正式运行冻结修正后的新提交 |
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，0.23 系列 |
| vLLM-Ascend | `11ee45653b199a097805b87011824a81ffa51b95`；不修改上游源码 |
| 环境 | A5 现场匹配的镜像、CANN/HCCL、torch/torch-npu；冻结精确版本、build 和镜像 digest |
| 同机 | 一台 8 卡 A5：Prefill DP2/NPU 0-1、Attention DP4/NPU 2-5、FFN DP2/NPU 6-7；TP1 |
| 跨机 | 三台 A5：Prefill DP8/NPU 0-7、Attention DP4/NPU 0-3、FFN DP2/NPU 0-1；TP1 |
| Graph | target Decode `FULL_DECODE_ONLY/U2`，五个 Graph 多流开关默认开启（1），支持显式关闭（0）；async scheduling off |
| DSpark | 只在 Attention 上运行 draft；draft eager，speculative token 数来自配套模型配置 |
| 最终精度 | 同机和跨机 C5 各做一次冻结口径的 GSM8K；不逐 token exact |
| 性能 | 两个 C5 记录吞吐、TTFT、TPOT、峰值 HBM、acceptance rate；不设置收益比例门槛 |

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
三个角色在 DSpark 点使用相同配套 checkpoint，只有 Attention 开启 speculative config。

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
export ATTENTION_MAX_NUM_BATCHED_TOKENS=4096
export FFN_MAX_NUM_BATCHED_TOKENS=8192
export PREFILL_MAX_NUM_BATCHED_TOKENS=4096
export MAX_CUDAGRAPH_CAPTURE_SIZE=8
export CUDAGRAPH_CAPTURE_SIZES='1 2 4 8'
export DBO_DECODE_TOKEN_THRESHOLD=2
export DBO_PREFILL_TOKEN_THRESHOLD=12
export ENABLE_MTP=0
```

`MAX_NUM_SEQS=32` 是本次请求覆盖配置，必须在 M1 冻结并实测 HBM；现有 recipe 默认 8。若现场只能使用 8，须明确区分“32 个客户端并发”和“32 个活跃序列”，并记录队列行为；不能用排队成功代替未捕获 shape 的 Graph fallback 证据。

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
test -d "/sys/class/net/$NIC_NAME"
ip -o -4 addr show dev "$NIC_NAME" scope global
```

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
rg -n 'kv-transfer|speculative|enable-dbo|no-async|cudagraph' /tmp/dsv4-1030-serve-help.txt
npu-smi info
```

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

### 4.3 设置 CASE 和轮次

每个角色终端执行同一 CASE/CYCLE/RUN_ID。RUN_ID 在协调终端生成一次，再复制到各节点，不能各节点独立生成时间戳。本机首次准备目录执行 `mkdir`，其他角色终端只加载已有目录。

```bash
export CASE=C1
export CYCLE=1
export RUN_ID=20261008T180000-single-C1-r1
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"

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
export VLLM_LOGGING_LEVEL=INFO
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP="${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP:-1}"
export AFD_HCCL_GRAPH_U2_HYBRID_DAG="${AFD_HCCL_GRAPH_U2_HYBRID_DAG:-1}"
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM="${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM:-1}"
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM="${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM:-1}"
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER="${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER:-1}"
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
PREFILL_KV_PORT DECODE_KV_PORT AFD_HOST AFD_PORT MAX_NUM_SEQS MAX_MODEL_LEN
EXECUTION_MODE U_BATCHES ENABLE_PD ENABLE_DSPARK AFD_HCCL_STAGE_DIAGNOSTICS
AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP
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

RUN_ID 示例每轮必须改成新的实际值，且与 CASE/TOPOLOGY/CYCLE 一致。
保存有效环境和运行清单，至少包括三个源码提交/dirty state、Python package/version/source、镜像、CANN/HCCL、角色地址/设备/端口、CASE/CYCLE、模型 hashes、最终开关。
只采集本次相关环境变量，避免把无关认证信息写入证据。上述 runtime 保存启动前意图值；角色脚本可能覆盖设置，启动后还须归档实际 argv 和逐 rank 的诊断配置日志，将二者与 runtime 对照，不能仅凭此 JSON 判定最终开关已生效。

## 5. 【A5 手工执行】第 3 步：先回归 standalone A4F2 Graph/U2

M1 修正后的新 recipe 必须先通过一次 standalone 回归，然后才能进入同机 C1。
在单台空闲 A5 使用新目录及 run ID，按前文配置环境，覆盖：

```bash
export ATTENTION_DEVICES=0,1,2,3
export FFN_DEVICES=4,5
export ATTENTION_HOST_IP="$HCCL_IF_IP"
export FFN_HOST_IP="$HCCL_IF_IP"
export AFD_HOST="$FFN_HOST_IP"
export MODEL_PATH="$FLASH_MODEL_PATH"
export ENABLE_PD=0 ENABLE_DSPARK=0
export EXECUTION_MODE=full-decode-only U_BATCHES=2
nohup bash "$RECIPE/afd_ffn.sh" > "$RUN_ROOT/ffn.log" 2>&1 &
echo $! > "$RUN_ROOT/ffn.pid"
nohup bash "$RECIPE/afd_attention.sh" > "$RUN_ROOT/attention.log" 2>&1 &
echo $! > "$RUN_ROOT/attention.pid"
```

使用下一节的 Attention readiness 检查、token pool 和请求脚本，添加 `--standalone`，`--base-url` 与 `--attention-url` 均指向 Attention。验证 capture、在线 replay、全部 Attention rank 的 stage 0/1、未捕获 shape fallback、正常停机及 NPU 清理。
没有 Prefill/Proxy，也不要求 PD/DSpark 证据。完成后全部停服，再恢复同机 P2+A4+F2 设备配置；不得将 standalone 的进程或日志复用到 C1。

## 6. 【A5 手工执行】第 4 步：每个 C1–C5 的启动、ready 和请求步骤

同机角色在本机执行；三机只在分配的主机执行其角色。每轮角色使用同一 RUN_ID。

### 6.1 启动 Prefill，然后立即相邻启动 FFN 与 Attention

Prefill 节点：

```bash
nohup bash "$RECIPE/afd_prefill.sh" > "$RUN_ROOT/prefill.log" 2>&1 &
echo $! > "$RUN_ROOT/prefill.pid"
```

FFN 节点，然后 Attention 节点；不要等待 FFN ready 后才启动 Attention，它们需要rendezvous。FFN 始终关闭 PD 和 DSpark：

```bash
# FFN 节点
ENABLE_PD=0 ENABLE_DSPARK=0 nohup bash "$RECIPE/afd_ffn.sh" \
  > "$RUN_ROOT/ffn.log" 2>&1 &
echo $! > "$RUN_ROOT/ffn.pid"
```

```bash
# Attention 节点
nohup bash "$RECIPE/afd_attention.sh" > "$RUN_ROOT/attention.log" 2>&1 &
echo $! > "$RUN_ROOT/attention.pid"
```

### 6.2 判断 ready 后再启动 Proxy

协调终端检查 Prefill/Attention，最多等待 20 分钟，失败保存日志并停止推进：

```bash
wait_http() {
  local url="$1" name="$2" deadline=$((SECONDS + 1200))
  until curl -fsS --max-time 5 "$url" > "$RUN_ROOT/$name-ready.txt"; do
    if ((SECONDS >= deadline)); then return 1; fi
    sleep 5
  done
}
wait_http "http://$PREFILL_HOST_IP:$PREFILL_API_PORT/health" prefill
wait_http "http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT/health" attention
curl -fsS "http://$ATTENTION_HOST_IP:$ATTENTION_API_PORT/v1/models" \
  > "$RUN_ROOT/models.json"
```

FFN 节点要求两个 rank 均进入 connector loop。计数只作初筛，最终核对 rank 身份，重复日志不能充当另一个 rank：

```bash
rg -n 'AFD FFN EngineCore started; workers run connector loop.' "$RUN_ROOT/ffn.log"
test "$(rg -c 'AFD FFN EngineCore started; workers run connector loop.' "$RUN_ROOT/ffn.log")" -ge 2
rg -n 'AFD HCCL diagnostics configuration:' "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
```

上一个 `rg` 在三机分别对本机日志执行。Graph 点逐 rank 核对最终五个 Graph 开关与本轮配置一致：默认均为 1，显式关闭项应为 0；配置与生效值不一致时返回 M1，不开始流量。`8911` 是 FFN 内部设置，不是 HTTP健康接口。
单个 Attention HTTP API 的 health 只作入口检查，另需核对四个 Attention rank ready、四个 drafter（DSpark 点）及两个 FFN rank ready。

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

### 6.3 保存 metrics-before，生成精确 token-ID 输入

业务请求经 Proxy；`/tokenize` 与 `/metrics` 是 Attention 管理调用，不计作 PD 请求。
使用 token-ID prompt，避免字符长度被误写为 KV block 边界长度。

```bash
curl -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-before.txt"
"$PYTHON_BIN" - "$RUN_ROOT/tokenize-request.json" <<'PY'
import json, sys
with open(sys.argv[1], 'w') as f:
    json.dump({'model': 'dsv4-afd', 'prompt': 'The sky is blue. ' * 2048,
               'add_special_tokens': False}, f)
PY
curl -fsS --max-time 60 "$ATTENTION_URL/tokenize" \
  -H 'Content-Type: application/json' \
  --data-binary "@$RUN_ROOT/tokenize-request.json" > "$RUN_ROOT/token-pool.json"
```

### 6.4 执行完整请求集

Proxy 主机或可达的验证客户端执行；token-pool 文件从 Attention 拷贝到该节点：

```bash
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --base-url "$BASE_URL" \
  --attention-url "$ATTENTION_URL" \
  --token-pool "$RUN_ROOT/token-pool.json" \
  --output "$RUN_ROOT/requests" \
  --run-id "$RUN_ID"
curl -fsS "$ATTENTION_URL/metrics" > "$RUN_ROOT/attention-metrics-after.txt"
```

standalone 回归将 `BASE_URL` 设为 `ATTENTION_URL`，同一条命令添加 `--standalone`。
脚本依赖环境已有的 `httpx`；输出目录必须不存在，避免覆盖旧证据。脚本依次检查：

1. 并发 1、8、32 的短输入和 1024 token 长输入，每条强制输出 64 token 并检查截断；
2. 31/32/33/63/64/65 token 输入，检查 PD 后 `usage.prompt_tokens` 没有改变；
3. 独立 EOS 请求，默认最多 1024 token，必须自然 `finish_reason=stop`；
4. 正常 SSE，必须有文本、结束原因和 `[DONE]`；
5. SSE 收到实际文本后关闭客户端连接，随后恢复请求成功；
6. 60 秒内 Proxy `request_num=0`，Attention 所有已暴露的 running/waiting gauge 为 0。

每个请求原始响应和 SSE 文件写入 `requests/`，结果为 `request_summary.json`。
EOS 用例若未自然结束，先冻结合适的 EOS prompt/预算再重跑，不可直接取消 EOS 门禁。
脚本返回 0 只表示请求子项通过，不自动判断首 token 内容、远程链路、Graph 或完整 F0。
取消后还要核对本次请求的上游 abort/cancel 处理及 Prefill KV/session 资源释放，不能仅用 Proxy 计数归零代替。当前 Proxy 自行生成上游 request ID，客户端发送的`X-Request-Id` 不保证原样转发；使用响应 ID、请求时间窗及 Proxy 的 ID 映射定位，缺少映射时补采诊断证据，不把客户端标签当作已经贯通的服务端 ID。

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
rg -n -i 'mooncake|session|transfer|first.token|kv' "$RUN_ROOT"/*.log \
  > "$RUN_ROOT/pd-evidence-candidates.txt" || true
rg -n -i 'stage=0|stage=1|stage 0|stage 1|replay|fallback|capture|draft|specul|verify' \
  "$RUN_ROOT"/*.log > "$RUN_ROOT/feature-evidence-candidates.txt" || true
rg -n -i 'Traceback|EngineCore.*fatal|out of memory|OOM|timeout|watchdog|507[0-9]{3}|ERR99999' \
  "$RUN_ROOT"/*.log > "$RUN_ROOT/error-candidates.txt" || true
```

DSpark metrics 使用 before/after counter 增量，逐 rank 核对，不使用进程累计值代替本轮业务生效。采集实际指标名、标签和 proposed/accepted 定义；acceptance rate 明确分子分母。
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
until [[ "$(rg -c 'AFD NPU FFN received Attention shutdown payload' "$RUN_ROOT/ffn.log" || true)" -ge 2 ]]; do
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

## 8. 【A5 手工执行】第 6 步：两个 C5 的 GSM8K 和性能记录

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
