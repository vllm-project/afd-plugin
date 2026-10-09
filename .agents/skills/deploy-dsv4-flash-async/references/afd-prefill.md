# AFD prefill_only

加载公共环境后，选择一个拓扑。以下是 `common_env.sh` 的默认逻辑设备布局；A 为 Attention，F 为 FFN。所有布局 F 均为 TP1、DP8/EP8。

| PREFILL_TOPOLOGY | 全局 A | node0 的 A | node1 的 A | F 所在节点 |
| --- | --- | --- | --- | --- |
| `legacy` | DP2TP4 | 0–7，DP2 | 无 | node0，8–15 |
| `afd_dp4tp2` | DP4TP2 | 0–7，DP4 | 无 | node0，8–15 |
| `afd_dp6tp4` | DP6TP4 | 0–15，DP4 | 0–7，DP2，start=4 | node1，8–15 |
| `afd_dp3tp8` | DP3TP8 | 0–15，DP2 | 0–7，DP1，start=2 | node1，8–15 |
| `afd_dp12tp2` | DP12TP2 | 0–15，DP8 | 0–7，DP4，start=8 | node1，8–15 |
| `afd_dp10tp2` | DP10TP2 | 0–11，DP6 | 0–7，DP4，start=6 | node1，8–15 |
| `afd_dp8tp2` | DP8TP2 | 0–7，DP4 | 0–7，DP4，start=4 | node1，8–15 |
| `afd_dp6tp2` | DP6TP2 | 0–7，DP4 | 0–3，DP2，start=4 | node1，8–15 |

单机示例：

```bash
export PREFILL_TOPOLOGY=afd_dp4tp2
export PREFILL_NODE_ID=0
export PREFILL_ENABLE_KV_CONNECTOR=0
bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"
```

双机示例：两端共同设置 `PREFILL_TOPOLOGY=afd_dp6tp4`、`P_SECONDARY_NODE_IP=<secondary-ip>`、`PREFILL_ENABLE_KV_CONNECTOR=0`，并共享模型配置和 P_NODE_IP；分别运行：

```bash
# node0
PREFILL_NODE_ID=0 bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"
# node1（在另一节点执行）
PREFILL_NODE_ID=1 bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"
```

node0 是 DP coordinator 和 HTTP 入口，node1 headless。两个节点启动命令均提交后再等待全局 ready，避免只启动一端就等待健康。脚本会在 F 所在节点先提交 FFN，再提交 Attention。

## 不变量

- 两个角色都使用 `CAMAsyncAFDConnector`、`async=true`、`compute_gate_on_attention=true`。
- `connector_extra_config` 保持 `dynamicQuant=1`、`async_moe_ubatching=true`、`async_moe_num_ubatches=2`、`async_moe_split=token`；不叠加原生 DBO flags。
- `num_attention_ranks = PREFILL_DP_SIZE * PREFILL_TP_SIZE`；`attn_ranks_per_dp = PREFILL_TP_SIZE`；`num_ffn_ranks = FFN_DP_SIZE = 8`（参考布局）。
- A/F 的 AFD_HOST、AFD_PORT、rank 总数、chunk 配置一致，跨节点 local DP 分段覆盖全局 DP，不重叠。
- A eager，F 默认 eager；A 默认 FlashComm1=1，TP1 的 F 强制为 0。
- 验证 AsyncCam layered FFN FULL 图时，仅 FFN 设
  `PREFILL_FFN_GRAPH_MODE=FULL` 和 `AFD_ASYNC_CAM_LAYERED_GMM=1`，
  并单独调用 `run_prefill_ffn.sh`；Attention 继续 eager，调用
  `run_prefill_attention.sh` 时取消继承 layered 开关。
  `PREFILL_FFN_GRAPH_MODE` 默认 `EAGER`，仅接受 `EAGER` 或 `FULL`。
  `FULL` 模式默认设置 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=60`，
  为图停止时最长 30 秒的 worker join 留出退出时间；显式设置该变量可覆盖。
  `EAGER` 模式不改动此超时默认值。
- `AFD_FORCE_BALANCED_TOPK_IDS=0`、`enable_force_load_balance=false`，不把强制均衡路由当真实模型配置。
- P-only 不需要 D_NODE_IP。只向主 P API 做单 token smoke/benchmark，不启动 D 或 proxy。
