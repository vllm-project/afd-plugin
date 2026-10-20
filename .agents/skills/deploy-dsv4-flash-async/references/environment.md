# 环境与验收

所有命令均在对应目标节点/容器运行；先替换占位值。相同部署的公共参数保持一致，路径可按节点实际挂载调整。

```bash
export DSV4_CODE_ROOT=/absolute/path/to/source-parent
export DSV4_SCRIPT_DIR="${DSV4_CODE_ROOT}/afd-plugin/.agents/skills/deploy-dsv4-flash-async/scripts"
export PYTHON=/absolute/path/to/python
export VLLM_CLI=/absolute/path/to/vllm
export PYTHON_VLLM_PATH="${DSV4_CODE_ROOT}/vllm"
export PYTHON_VLLM_ASCEND_PATH="${DSV4_CODE_ROOT}/vllm-ascend"
export PYTHON_AFD_PLUGIN_PATH="${DSV4_CODE_ROOT}/afd-plugin"
export DSV4_MODEL=/absolute/path/to/DeepSeek-V4-Flash/model
export SERVED_MODEL_NAME=dsv4
export P_NODE_IP=<primary-prefill-ip>
export NIC_NAME=<communication-interface>
export MAX_MODEL_LEN=65536
export PREFILL_MAX_NUM_BATCHED_TOKENS=8192
export PREFILL_MAX_NUM_SEQS=16
export FFN_MAX_NUM_SEQS=16
export PREFILL_GPU_MEMORY_UTILIZATION=0.7
export BLOCK_SIZE=128
export LOG_DIR=/absolute/path/to/logs/unique-deployment-id
export PID_DIR="${LOG_DIR}/pids"
```

上面的数值是初始配置示例，不是容量承诺。参考脚本允许 chunk 为 4096/8192/16384/32768/49152/65536；上下文、并发和 chunk 调大前检查容量。`legacy` 默认上下文为 1048576，其他列出的实验拓扑为 65536，因此建议显式设置，PD 的 P/D 也保持一致。

## 0.30 基线

本次迁移针对固定源码，不凭发行版名字推断接口兼容：

| 组件 | 源码版本 |
| --- | --- |
| AFD runtime 审阅基线 | `4b11d2affacef52aefaa78b71b68c5b41ecfdf25` |
| vLLM 0.30.0 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6` |

环境与历史实测见 [安装说明](../../../../README.md#ascend-npu-installation)。
记录实际 CANN、torch、torch-npu 和构建产物；已有 A3 记录不是任意版本组合的兼容承诺。
源码路径默认使用当前 AFD checkout 和同级 vLLM/Ascend，挂载或安装包不同则显式覆盖。

AFD A/F 显式设置 `VLLM_USE_V2_MODEL_RUNNER=1`，默认插件为 `ascend,afd`；
P/D 的 Attention 还会添加 `ascend_kv_connector`，以注册 `MooncakeHybridConnector`。
仅配置 `--kv-transfer-config` 不会自动加载被插件白名单排除的 connector。
V1 引擎变量不能代替 MRV2。原生 EP/D 记录实际 native runner，不强加 AFD 的运行器约束。
各模型入口显式设置 `--attention_config.indexer_kv_dtype int8`；
`DSV4_INDEXER_KV_DTYPE` 可选择在目标环境验证的其他值。
旧 `enable_force_load_balance` 改为 `enable_force_eplb`，移除上游不再接受的
`enable_dsv4_shared_compressor_workspace`；不靠过滤未知参数掩盖配置错误。

## 通信算子的安装与加载

CAM async 通信算子源码已集成到仓库 `csrc/npu/ascend_kernels/afd_async_*/`，随 AFD 的 pip 安装流程编译并安装，不需要单独安装外部 CAM wheel、vendor 或 libopapi.so。保留正常的 CANN 初始化环境及 Python/TBE 路径；启动脚本不再配置 CAM 专用路径或 LD_PRELOAD。

在已初始化 CANN、torch-npu 和构建依赖的 910C 环境中，从 AFD 仓库根目录安装：

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 \
  "${PYTHON:-python3}" -m pip install -e . -v --no-build-isolation
```

`setup.py` 在检测到 Ascend 环境时默认启用构建；示例显式设为 1，确保此次安装包含 NPU 扩展。正常完整安装不要设置 `AFD_BUILD_ASCEND_OPS=0` 或 `AFD_SKIP_ACLNN_BUILD=1`。后者仅用于已有匹配算子产物时跳过 ACLNN 重编译，不能补齐旧包缺少的通信算子。已安装匹配版本时无需每次启动重新编译。

先核对现有源码、CANN/Torch 和原生产物，再决定是否需要构建；只审查 skill 时无需为此加载模型。
无卡主机导入真实 CLI 时，Triton 可能因自动识别架构失败而退出。已知实际目标架构时可设置
`TRITON_ASCEND_ARCH`；本轮 910C 的无卡检查使用 `Ascend910_9392`，这不模拟设备执行。
无卡构建若因 Torch 的 CMake 路径探测混入设备诊断输出而失败，可显式指定实际安装位置的
`Torch_DIR` 和 `TORCH_NPU_PATH`。保留原失败日志，复用已经成功生成且匹配的 ACLNN 包；
不把仅完成绑定续建写成完整安装或真机通信已通过。

运行时 loader 配置插件打包的 `afd-plugin` vendor，并加载 `afd_plugin._C_ascend`。部署前可在与服务相同的 Python 环境中验证四个注册：

```bash
"${PYTHON:-python3}" - <<'PYTHON'
import torch
import torch_npu
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

ensure_afd_ascend_ops_loaded()
print(torch.ops.afd_ascend.afd_async_dispatch_send)
print(torch.ops.afd_ascend.afd_async_dispatch_recv)
print(torch.ops.afd_ascend.afd_async_combine_send)
print(torch.ops.afd_ascend.afd_async_combine_recv)
PYTHON
```

仅 `import afd_plugin` 或 loader 成功不证明旧安装包含这四个算子；注册检查也不代替实际通信验收。缺少算子时检查安装日志和实际导入路径，并重新安装匹配源码。P 的 Attention/FFN 节点均需匹配的算子版本。

仓库内构建细节见 `csrc/npu/README.md`，通信算子说明见 `docs/npu/CAM_ASYNC_ROUTED_OPS.md`。

| 用途 | 变量 / 默认值 |
| --- | --- |
| P API / FFN API | `PREFILL_PORT=7100` / `FFN_PORT=7101` |
| D API / proxy | `DECODE_PORT=7100` / `PROXY_PORT=8000` |
| CamAsync rendezvous | `AFD_HOST=P_NODE_IP` / `AFD_PORT=1239` |
| P / D DP RPC | `PREFILL_DP_RPC_PORT=12321` / `DECODE_DP_RPC_PORT=12321` |
| P / D KV | `PD_KV_PREFILL_PORT=30000` / `PD_KV_DECODE_PORT=30100` |

P/D 同号端口默认位于不同节点；合并角色到同机时需重新分配设备与端口。通信库可能需要额外端口，不把此表当完整防火墙配置。

AFD/full-model 主节点启动后检查：

```bash
curl -fsS --max-time 10 "http://${P_NODE_IP}:${PREFILL_PORT:-7100}/health"
curl -fsS --max-time 10 "http://${P_NODE_IP}:${PREFILL_PORT:-7100}/v1/models"
curl -fsS --max-time 120 "http://${P_NODE_IP}:${PREFILL_PORT:-7100}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Hello\",\"max_tokens\":1,\"temperature\":0}"
```

检查响应内容及服务日志。headless 节点不要求独立 HTTP API；FFN 不作为用户请求入口。AFD 还需确认 FFN 存活且通信初始化完成。

日志默认命名：`prefill_attention_nodeN.log`、`prefill_ffn_nodeN.log`、`prefill_<topology>_nodeN.log`、`decode.log`、`proxy.log`。源码默认 LOG_DIR 是脚本目录下 logs，但多轮部署应使用独立目录避免覆盖。

性能测试另行确认数据集及实际 chunk。入口只用于关闭 prefix cache、KV connector 的 prefill 服务，默认输出一个 token，并记录 `kv_connector=false`。`--chunk-size` 只作标签，不修改服务配置。保留失败请求与原始结果，不重跑挑选最好值。

内置固定数据集、校验规则和 RPS/MBT sweep 用法见 [固定负载压测](benchmark.md)。
