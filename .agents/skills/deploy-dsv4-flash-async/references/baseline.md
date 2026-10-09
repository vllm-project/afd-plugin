# 非 AFD prefill baseline

使用 full-model 入口，而不是给 `run_prefill.sh` 删掉一个 AFD 参数。

| PREFILL_TOPOLOGY | 全局布局 | 节点布局 |
| --- | --- | --- |
| `ep16` / `ep16_dp4tp4` | DP4TP4/EP16 | node0，设备 0–15 |
| `ep16_dp8tp2` | DP8TP2/EP16 | node0，设备 0–15 |
| `ep16_dp2tp8` | DP2TP8/EP16 | node0，设备 0–15 |
| `ep32` | DP4TP8/EP32 | 两节点各 local DP2；node0 start=0，node1 start=2 |

公共环境在新 shell 设置后，单机运行：

```bash
export PREFILL_TOPOLOGY=ep16
export PREFILL_ENABLE_KV_CONNECTOR=0
PREFILL_NODE_ID=0 bash "${DSV4_SCRIPT_DIR}/run_prefill_full.sh"
```

EP32 则两端共同设置 `PREFILL_TOPOLOGY=ep32` 和 `P_SECONDARY_NODE_IP`，分别以 PREFILL_NODE_ID=0/1 运行同一入口。node1 使用 headless，无独立 HTTP 入口。

`run_prefill_full_rank.sh` 过滤 AFD 源码 PYTHONPATH，并设置仅含 Ascend 插件的 `VLLM_PLUGINS`，同时保留 CANN Python/TBE 路径。核对日志未加载 AFD 插件；不要粗暴清空所有 PYTHONPATH。

参考配置启用 EP、chunked prefill、eager，关闭 prefix cache；FlashComm1 默认 1，shared expert DP 默认 true。与 AFD 对比时明确记录这些差异，并对齐模型、输入数据、chunk、上下文、并发和设备数量口径。

## 叠加 PD 分离

保持上述 full-model 拓扑，将 `PREFILL_ENABLE_KV_CONNECTOR=1`，入口仍为 `run_prefill_full.sh`。它给 full-model P 配置 MooncakeHybridConnector producer；D/proxy 的启动和验证见 [PD 分离](pd.md)。P/D 两端设置相同的 PREFILL_TOPOLOGY，自动获得一致的全局 DP/TP。非 AFD PD 的脚本配置已补齐，仍需在目标 NPU 环境验证 KV 传输和生成请求。
