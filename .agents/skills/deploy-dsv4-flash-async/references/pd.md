# P/D 分离：可叠加 AFD 或非 AFD Prefill

PD 与 Prefill 是否使用 AFD 独立。以下先给出 AFD P 的两节点示例，再说明非 AFD P 的替换方式。每台有 16 个 910C 逻辑设备：

- P：`legacy`，A DP2TP4 占 0–7，F TP1/EP8 占 8–15。
- D：完整模型 DP16TP1 + EP，占 0–15。
- P Attention 与 D 通过 `MooncakeHybridConnector` 传递 KV；FFN 不参与 P/D KV connector。

两端加载公共环境，保持如下变量一致（路径和 NIC 按各机实际值设置）：

```bash
export P_NODE_IP=<prefill-ip>
export D_NODE_IP=<decode-ip>
export PREFILL_TOPOLOGY=legacy
export PREFILL_NODE_ID=0
export PREFILL_ENABLE_KV_CONNECTOR=1
export DECODE_DP_SIZE=16
export DECODE_TP_SIZE=1
export MAX_MODEL_LEN=65536
```

P 和 D 均通过 `VLLM_CLI` 选择 vLLM 可执行文件；proxy 使用 `PYTHON`。

```bash
# 在 D 节点提交
bash "${DSV4_SCRIPT_DIR}/run_decode.sh"
# 在 P 节点提交
bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"
```

两端均启动后检查 P/D 日志、进程及健康接口；不必在提交 P 之前等待 D 完成全部初始化。后端健康后，在通常为 P 的 proxy 节点运行：

```bash
curl -fsS --max-time 10 "http://${P_NODE_IP}:${PREFILL_PORT:-7100}/health"
curl -fsS --max-time 10 "http://${D_NODE_IP}:${DECODE_PORT:-7100}/health"
bash "${DSV4_SCRIPT_DIR}/run_proxy.sh"
PROXY_IP="${P_NODE_IP}" bash "${DSV4_SCRIPT_DIR}/curl_test.sh"
```

proxy 在其他节点运行时，最后一条使用该节点的真实 IP。默认 proxy 源码为 `${PYTHON_VLLM_ASCEND_PATH}/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`，可用 PROXY_SCRIPT 覆盖，先检查文件和参数契约。

## 非 AFD P + PD

两端将 `PREFILL_TOPOLOGY` 设置为同一个 full-model 拓扑（例如 `ep16`），保留 `PREFILL_ENABLE_KV_CONNECTOR=1`。D 仍执行 `run_decode.sh`，P 改为：

```bash
bash "${DSV4_SCRIPT_DIR}/run_prefill_full.sh"
```

P full-model 是 KV producer，无独立 FFN 服务；proxy 和验收步骤与上面一致。`ep16`/`ep16_dp4tp4` 的 P 为 DP4TP4，`ep16_dp8tp2` 为 DP8TP2，`ep16_dp2tp8` 为 DP2TP8，`ep32` 为 DP4TP8。公共环境在 P/D 两端解析同一拓扑，D consumer 使用对应全局 DP/TP。不要只在 P 的单条命令前临时设置拓扑而让 D 保留 legacy。

EP32 的 P 需要两个节点（各 local DP2），另外为 D 分配独立资源；两个 P 节点均需设置 KV=1，并以各自 PREFILL_NODE_ID 启动 full-model 入口。非 AFD PD 配置已补齐，尚未完成实际 NPU 链路验收。

## 参数核对与边界

- P KV：producer、engine_id=0、默认 kv_port=30000；D：consumer、engine_id=1、默认 kv_port=30100。
- P/D 的 `kv_connector_extra_config.prefill/decode` 必须分别填写真实全局 DP/TP。修改 P 拓扑也要同步 D 的 PREFILL_TOPOLOGY/DP/TP，不能沿用 D 进程的 legacy 默认值。
- D 脚本保留 `--async-scheduling`、MTP speculative token=1、`FULL_DECODE_ONLY`、`enable_npugraph_ex=true`，且 `VLLM_ASCEND_APPLY_DSV4_PATCH=1`。这些与 P 端 CamAsync/eager 是不同机制。
- 默认 D chunk=120、max seqs=60、显存比例=0.9；按容量验证，不推断所有环境均可启动。
- P 使用双机 AFD 拓扑时，它消耗两个 P 节点，默认完整 D 还需独立资源；不能在第二个 P 节点上照搬 16 卡 D 布局造成设备重叠。该扩展需验证，不能冒称是 README 的两机 1P1D。
- 验收必须通过 proxy 发实际生成请求，并查看 P/D 请求及 KV 传输日志。仅 P/D `/health` 成功不证明 PD 链路可用。
- Prefill 是否使用 AFD 不改变 PD 的 producer/consumer 分工；两种组合均以实际 proxy 请求及 KV 传输成功为验收标准。
