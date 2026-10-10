# DeepSeek-V4 AFD A5 手工执行交接清单

更新日期：2026-10-09。执行顺序与[统一验证指导书](DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md)、[1030 计划](DEEPSEEK_V4_AFD_1030_EXECUTION_PLAN_ZH.md)一致。

`【本地已完成】` 表示源码修改及 CPU/mock 检查；`【A5 手工执行】` 表示需要你在A5 现场运行。当前先执行 **A5-01 → A5-02 → A5-03**，回传 M1 证据；M1 通过后再进入 A5-04。三机和最终精度留待前置阶段通过后执行。

## 1. 执行分工与当前状态

| ID | 工作 | 执行位置/人员 | 当前状态 |
|---|---|---|---|
| LOCAL-01 | 五个 Graph/U2 开关默认 1，保留显式 0 | 本地已完成 | 默认、关闭和混合覆盖测试通过 |
| LOCAL-02 | A5 Prefill/Attention HIXL JSON 写入 | 本地已完成 | 检查本机绝对目录和对应物理 NPU endpoint JSON；实际启动 argv 的 mock 测试通过 |
| LOCAL-03 | FFN/standalone 不加载 PD HIXL 配置 | 本地已完成 | launcher 参数检查通过；A3 配置不在本次修改范围 |
| LOCAL-04 | 请求脚本、文档和交付分支提交 | 本地已完成，现场更新待执行 | 使用 `feat/dsv4-afd-phase1-delivery` 的发布提交；离线包保留作备用 |
| A5-01 | 从 Git 更新交付分支、准备环境并核对安装位置 | **A5 手工执行：你** | 待执行 |
| A5-02 | 审计 HIXL、模型、版本、设备/IP/端口；冻结 GSM8K 口径 | **A5 手工执行：你** | 待执行 |
| A5-03 | standalone A4F2 Graph/U2 全开回归和清理 | **A5 手工执行：你** | 待执行，M1 实机门禁 |
| A5-04 | 同机 P2+A4+F2，C1 两轮 | **A5 手工执行：你** | M1 通过后执行 |
| A5-05 | 同机 C2→C3→C4→C5，C5 两轮 | **A5 手工执行：你** | 同机 C1 通过后执行 |
| A5-06 | 三机 P8/A4/F2，C1–C5 | **A5 手工执行：你** | 同机矩阵全部通过后执行 |
| A5-07 | 同机/三机 C5 GSM8K 与诊断关闭的性能采集 | **A5 手工执行：你** | 完整功能组合通过后执行 |
| LOCAL-05 | 审阅回传证据、定位问题、更新验收表及最终包 | 收到证据后本地继续 | 待现场结果 |

本地测试不会将 A5-01 到 A5-07 自动标记通过。默认 C3/C5 是五开关全开；关闭对照另用 run ID。资源文件内容与真实设备是否匹配、Mooncake session/HCCL 数据面以及Graph/DSpark 在线行为必须在 A5 上验证。

## 2. 【A5 手工执行】A5-01：从 Git 更新交付分支

当前优先从 `wenhow/afd-plugin` 的 `feat/dsv4-afd-phase1-delivery` 更新。
先正常停止本轮 A5 服务；原 checkout 有现场改动时先保留并处理改动，不强制覆盖。
检查 `origin` 确实指向该仓库，再在干净工作树执行：

```bash
set -euo pipefail
export AFD_PLUGIN_ROOT=/root/dsv4-afd-hccl/src/afd-plugin-phase1-delivery
git -C "$AFD_PLUGIN_ROOT" remote get-url origin
test -z "$(git -C "$AFD_PLUGIN_ROOT" status --porcelain)"
git -C "$AFD_PLUGIN_ROOT" fetch origin feat/dsv4-afd-phase1-delivery
git -C "$AFD_PLUGIN_ROOT" switch feat/dsv4-afd-phase1-delivery
git -C "$AFD_PLUGIN_ROOT" merge --ff-only origin/feat/dsv4-afd-phase1-delivery
export AFD_EXPECTED_COMMIT="$(git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD)"
printf 'AFD_EXPECTED_COMMIT=%s\n' "$AFD_EXPECTED_COMMIT"
```

核对输出与本次发布回执中的完整提交一致，写入 `site.env` 后冻结，不随每轮请求
重新取变化中的远端 HEAD。尚无 checkout 或 `origin` 是另一个仓库时，创建新目录：

```bash
export AFD_PLUGIN_ROOT=/root/dsv4-afd-hccl/src/afd-plugin-phase1-delivery
# 仅在新目录不存在时执行；已有目录按上一段更新。
test ! -e "$AFD_PLUGIN_ROOT"
git clone --branch feat/dsv4-afd-phase1-delivery \
  https://github.com/wenhow/afd-plugin.git "$AFD_PLUGIN_ROOT"
export AFD_EXPECTED_COMMIT="$(git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD)"
```

### 2.1 可选：无网络环境使用已有离线包

交付包包含离线 Git bundle、`SOURCE_COMMIT`、`delivery_manifest.json`、`SHA256SUMS`、`checks/` 及三份指导文档。使用新的解压目录和源码目录。
将本地交付的 tar.gz 与同名 `.sha256` 文件拷贝到 A5；在保存文件的目录先验证外层 hash，再解压。下面的 `DELIVERY_TAR` 替换为实际包名，`DELIVERY_DIR` 是新解压目录。

```bash
set -euo pipefail
export DELIVERY_TAR=/data/packages/dsv4-afd-a5-m1-20261009.tar.gz
export DELIVERY_DIR=/data/packages/dsv4-afd-a5-m1-20261009
cd "$(dirname "$DELIVERY_TAR")"
sha256sum -c "$DELIVERY_TAR.sha256"
test ! -e "$DELIVERY_DIR"
mkdir "$DELIVERY_DIR"
tar -xzf "$DELIVERY_TAR" -C "$DELIVERY_DIR" --strip-components=1
cd "$DELIVERY_DIR"
sha256sum -c SHA256SUMS
export AFD_PLUGIN_ROOT=/root/dsv4-afd-hccl/src/afd-plugin-phase1-delivery-m1-20261009
test ! -e "$AFD_PLUGIN_ROOT"
git clone "$DELIVERY_DIR/afd-plugin-a5-m1.bundle" "$AFD_PLUGIN_ROOT"
export AFD_EXPECTED_COMMIT="$(cat "$DELIVERY_DIR/SOURCE_COMMIT")"
test "$(git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD)" = "$AFD_EXPECTED_COMMIT"
test -z "$(git -C "$AFD_PLUGIN_ROOT" status --porcelain)"
```

离线 bundle 是交付分支工作内容的隔离冻结副本；commit 与本次包的 SOURCE_COMMIT一致。原本地交付分支中的未提交改动保留。不要把 `02d0779` 填作本次新提交；它是回归起点，不包含此次 HIXL 写入与请求入口。

### 2.2 Git 或离线更新后的安装位置确认

按统一指导书第 3 节填写 `site.env`、`node.env`，`AFD_PLUGIN_ROOT` 指向实际 checkout，
`AFD_EXPECTED_COMMIT` 填所选交付的完整提交。Git 交付与旧离线包的源码版本分别记录，
不能混用提交或文档。加载现场唯一 CANN 和已有 venv，检查：

```bash
export VENV_ROOT=/root/dsv4-afd-hccl/venv
export CANN_ROOT=/usr/local/Ascend/cann-9.2.0
test -f "$CANN_ROOT/set_env.sh"
source "$CANN_ROOT/set_env.sh"
source "$VENV_ROOT/bin/activate"
export PYTHON_BIN="$VENV_ROOT/bin/python"
export VLLM_BIN="$VENV_ROOT/bin/vllm"
"$PYTHON_BIN" -m pip show torch torch-npu vllm vllm-ascend vllm-afd-plugin
```

上述 CANN 仅是路径示例，必须改为冻结镜像里的实际版本；不要在已经加载另一版本CANN 的终端里叠加 source。已有安装如果仍指向旧 afd-plugin checkout，使用现场已有构建依赖离线安装本次源码，之后核对 import 的真实来源：

```bash
# 本期 P2pHccl 使用 HCCL send/recv，不构建 CAMP2P A2E/E2A 自定义算子。
AFD_BUILD_ASCEND_OPS=0 "$PYTHON_BIN" -m pip install \
  --no-index --no-deps --no-build-isolation -e "$AFD_PLUGIN_ROOT"
"$PYTHON_BIN" - "$AFD_PLUGIN_ROOT" <<'PY'
import sys
from pathlib import Path
from importlib.metadata import version
import afd_plugin
root = Path(sys.argv[1]).resolve()
source = Path(afd_plugin.__file__).resolve()
assert source.is_relative_to(root), (source, root)
print(version('vllm-afd-plugin'), source)
PY
```

安装仅针对 afd-plugin，不重装上游栈。构建依赖缺失则回传安装日志，不临时升级vLLM/torch/CANN；保留旧源码路径用于回退。此步骤是环境操作，不是功能验收通过。

## 3. 【A5 手工执行】A5-02：本机预检及 HIXL 配置验证

按统一指导书第 3–4 节完成以下现场检查，保存原始输出：

- 三个源码提交与 dirty state，Python 安装位置，镜像 digest，驱动/CANN/HCCL 精确 build。
- 空闲且健康的 NPU，容器可见 NIC/IP，所有角色地址、端口范围及路由。
- Flash/DSpark 配套原始 config、权重 manifest/hash，真实 block size 与 target layer IDs。
- DSpark 点在 Attention 节点按统一指导书第 3.3 节检查 `ascend_ops` 及两个 fused attention NPU kernel；使用服务的同一个 Python/venv，缺依赖时不启动 C2/C4/C5。
- A5 Prefill/Attention 的本机资源目录；对应物理设备的 endpoint JSON 与设备匹配。
- 离线 GSM8K/task 配置、题目数、prompt/fewshot、解码参数、评分键及最低分数；最低分必须由现场在运行前确定，本地不代填验收阈值。

同机 PD 的 P 设备为 0/1，A 设备为 2/3/4/5；三机 P 为 0–7，A 为 0–3。
资源文件名按物理设备编号，不按角色内 local rank 重编号。仅 PD 的 P/A 检查 HIXL，standalone 和 FFN 不需要此配置；A3 不调用 A5 专用生成函数。

下面是**同机 PD 的启动前配置预检**，不启动模型。先在当前终端完成统一指导书第 3.1 节的环境加载（含 `site.env`、`node.env`、CANN、venv），再执行下面整段。
`RUN_ROOT` 的初始化放在**本段开头、`export RECIPE=...` 之前**；预检 JSON 就保存在此目录。`RUN_BASE` 来自 `node.env`，本机路径按现场填写。Prefill/Attention 同机在一个终端执行：

```bash
: "${RUN_BASE:?先按统一指导书第 3.1 节加载 node.env 中的 RUN_BASE}"
export RUN_ID="$(date +%Y%m%dT%H%M%S)-single-precheck"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"

export RECIPE="$AFD_PLUGIN_ROOT/recipe/npu/P2pHcclAFDConnector/deepseek_v4"
export ASCEND_LOCAL_COMM_RES_PATH=/etc/hixlep
export PREFILL_DP_SIZE=2 PREFILL_TP_SIZE=1 ATTENTION_RANKS=4
source "$RECIPE/common.sh"
build_a5_pd_kv_config kv_producer dsv4-prefill 30000 0,1 \
  > "$RUN_ROOT/prefill-kv-precheck.json"
build_a5_pd_kv_config kv_consumer dsv4-decode 31000 2,3,4,5 \
  > "$RUN_ROOT/attention-kv-precheck.json"
"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ('prefill', 'attention'):
    config = json.loads((root / f'{name}-kv-precheck.json').read_text())
    value = config['kv_connector_extra_config']['ascend_local_comm_res_path']
    assert value == str(Path(os.environ['ASCEND_LOCAL_COMM_RES_PATH']))
    assert Path(value).is_absolute()
    print(name, config)
PY
```

三机预检改为 `PREFILL_DP_SIZE=8`，分别在 P 本机执行 producer/设备 `0,1,2,3,4,5,6,7`，在 A 本机执行 consumer/设备 `0,1,2,3`，使用各自真实的 HIXL 路径。不要只生成预检 JSON 就手工拼接 CLI；正式 launchers 已调用同一函数并打印生效 JSON。
三机预检的 `RUN_ID` 在协调终端生成一次，改用 `cross-precheck` 后缀，再复制同一值到 P/A；两机分别初始化本机 `RUN_ROOT`，不要各自生成时间戳。
文件语法可读不代表平台资源已验证；真实 session 要在 A5-04 证明。

## 4. 【A5 手工执行】A5-03：首先执行 standalone 回归

按统一指导书第 5 节，单台空闲 A5 执行：

第 5 节已将 `RUN_ROOT` 初始化放在设备配置及 `nohup` 启动命令之前。
在同一终端依次执行“第 3 节环境加载/版本检查 → 第 5 节目录及开关配置 → 第 4.4 节环境采集 → 第 5 节启动命令”。本轮使用新的 standalone RUN_ID，另建目录保存日志。

| 设置 | 本轮值 |
|---|---|
| Attention | DP4，NPU 0/1/2/3 |
| FFN | DP2，NPU 4/5 |
| PD / DSpark | 都为 0，不启动 Prefill/Proxy |
| Target / U | `full-decode-only` / 2 |
| 五个 Graph 开关 | 全部为 1，避免沿用上一终端的关闭值 |
| 回归轮数 | 至少一轮全新冷启动，完整请求验证和正常清理 |

在运行主启动命令前显式恢复本轮默认全开，随后使用指导书第 5 节启动 FFN/Attention：

```bash
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=1
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=1
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
```

Attention ready 且两个 FFN rank 进入 loop 后，先执行**统一指导书第 5 节列出的 5 请求 `--suite smoke` 命令**，无需生成 token pool。
smoke 通过后按第 6.3 节生成 token pool，再执行第 6.4 节的 `--suite full` 完整 92 请求集，添加 `--standalone`，两个 URL 均指向本机 Attention。
`--standalone` 是 `tools/validation/a5_1030_requests.py` 的参数，已放在该命令中；无需添加到 FFN/Attention 服务启动命令。
它跳过 Proxy `/healthcheck`/`request_num` 检查；`--base-url` 和 `--attention-url` 均填写 Attention 地址，其他请求验证保留。通过后按第 7 节正常停服；
收集所有角色/rank 的 capture、在线 replay、stage 0/1 和未捕获 shape eager fallback证据，以及无 fatal、shutdown、NPU/端口清理结果。

同机 HTTP 客户端使用 `http://127.0.0.1:$ATTENTION_API_PORT`；本机 IP 返回 504 时，按指导书第 3.1 节用 `curl --noproxy '*'` 诊断，不改 HCCL/AFD 通信地址。
EOS 失败时先用第 6.4 节的 `--suite eos` 单独发送 1 个 chat 请求定位，再用 `--suite tail` 补测剩余项目，避免反复发送已执行过的 88 个负载/边界请求。

本轮没有 PD/DSpark gate。`smoke`/`eos`/`tail` 成功只代表选定子项通过；M1 请求门禁要求 `full_request_suite_passed=true`，路径或 cleanup 未齐全时仍保持 pending。失败先回传此轮证据，不启动同机 C1。

## 5. 【A5 手工执行】A5-04 到 A5-07：后续顺序

1. **A5-04**：回归通过后恢复同机 P0–1/A2–5/F6–7，按第 4/6/7 节执行 C1 两轮。
   `eager/U1/DSpark off`，证明真实 Mooncake session、KV transfer、首 token 衔接、A2F/F2A 和正常停服。不要复用 standalone 的设备配置、进程或日志。
2. **A5-05**：同机 C2、C3、C4 至少一轮，C5 两轮。DSpark 只在 Attention；C3/C5 target Graph，draft eager，五开关默认全开。每点先通过再进入下一点。
3. **A5-06**：同机五点完成后，三台 A5 按 P8/A4/F2 重新执行 C1–C5，C1/C5 两轮，提供远程 KV 和远程 A/F 在线证据。缺机器或资源明确记录外部阻塞。
4. **A5-07**：只对同机和三机 C5 做冻结 GSM8K；另用新 run ID、关闭同步诊断后做性能采集。执行统一指导书第 8 节，保留完整分数及样本、吞吐/TTFT/TPOT/HBM/acceptance rate；功能错误或无法清理仍阻断。

## 6. 【A5 手工执行】当前请回传的 M1 证据

执行 A5-01 到 A5-03 后，按统一指导书第 9 节归档，回传一个 M1 evidence tar.gz及 SHA256。至少包括：

- Git 发布的完整提交（或离线包 SOURCE_COMMIT）、安装/审计输出、`runtime.json`、配置和模型 manifest/hash。
- 同机 PD 预检的两份 KV JSON、物理设备 endpoint 文件清单/hash及内容匹配确认。
- standalone 完整 FFN/Attention 日志、请求 JSON/SSE、metrics before/after。
- 逐 rank 的五开关生效值、在线 Graph replay/U2/fallback 的日志行号或 trace 时间。
- 停服顺序和两个 FFN shutdown 记录、NPU/端口/进程清理输出、退出结果。
- GSM8K 协议冻结情况，缺资源时明确列出缺项。

若失败，保留最早错误和最后进度，先采集再正常停服；同一份包回传即可。收到证据后继续本地审阅与定位，并更新 M1 状态，再进入同机 C1。当前未收到 A5 实机结果，不能把本地测试或离线交付包写成 M1 功能通过。
