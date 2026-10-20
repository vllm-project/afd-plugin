# AFD prefill 与执行模式

以下是内置布局，A 为 Attention，F 为 FFN。F 均为 TP1、DP8/EP8。
布局是命令模板，不代表这些组合都已完成 0.30 硬件验收。

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

单机例子（公共环境见 [环境](environment.md)）：

```bash
export PREFILL_TOPOLOGY=afd_dp4tp2
export PREFILL_NODE_ID=0
export PREFILL_ENABLE_KV_CONNECTOR=0
export PREFILL_ENABLE_DBO=1
bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"
```

双机使用相同拓扑、AFD_HOST/PORT、模型、MBT 和执行模式，设置
`P_SECONDARY_NODE_IP`，分别以 `PREFILL_NODE_ID=0/1` 启动。
node0 为 DP coordinator 和 HTTP 入口，node1 headless；两端均提交后再等待 ready。

## Attention 原生 DBO

两个角色显式选择 MRV2，使用 `async_dp=true`、Attention gating 和动态量化。
Attention 默认 `PREFILL_ENABLE_DBO=1`，使用 `--enable-dbo`；设为 `0` 进行对照。
默认保留上游 DBO 阈值。需要专门研究阈值时可设置
`PREFILL_DBO_PREFILL_TOKEN_THRESHOLD` / `PREFILL_DBO_DECODE_TOKEN_THRESHOLD`，
保留原生阈值的业务验证应独立进行。

DBO 只在 A 开启，固定两 microbatch，PP=PCP=DCP=1，无 speculative decoding。
低于阈值的步骤可不切分，不能仅凭并发或开关声称执行了 DBO。
不再传旧 `async_moe_split=request/token` 或把 native DBO flags 传给 F。
A 默认 FlashComm1=1，TP1 的 F 固定为 0；显式 shared-expert DP 和 finegrained TP
不属于当前 AFD MRV2 Async CAM 接受的配置。

## FFN eager、layered eager 与 FULL

| FFN 模式 | `PREFILL_FFN_GRAPH_MODE` | `AFD_ASYNC_CAM_LAYERED_GMM` |
| --- | --- | --- |
| 普通 eager | `EAGER`（默认） | `0` 或未设置 |
| W4A8 layered eager | `EAGER` | `1` |
| W4A8 layered FULL | `FULL` | `1` |

layered 路径的当前 checkpoint 目标是 910C、同构 DSV4 per-channel W4A8、
动态量化、静态专家和 Attention gating；per-group 算子测试不证明真实 per-group
checkpoint loader 可用。W8A8 仍使用普通 eager，不能由 A8W8 算子覆盖推断 FULL 支持。

FULL 只改变 F；A 保持 eager，可使用原生 DBO。F 省略 `--enforce-eager` 并显式指定
`cudagraph_mode=FULL`，不支持 piecewise/decode-only，也没有自动 eager fallback。
FULL 启动前必须明确设置 `BATCH_SIZE_FACTOR`：receive capacity 为
`262144 * BATCH_SIZE_FACTOR`，按实际 dispatch envelope、uint16 源索引限制和设备内存选择。
不能把小容量硬件 fixture 或 0.26 的窗口大小当成大 MBT 的容量证明。

```bash
export PREFILL_FFN_GRAPH_MODE=FULL
export AFD_ASYNC_CAM_LAYERED_GMM=1
export BATCH_SIZE_FACTOR=<capacity-factor-for-this-topology-and-MBT>
bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"
```

入口仅向 F 传递 layered 开关，A 子进程显式设为 0。F 的启动日志须证明
`actual=layered`，FULL 还需实际请求期间的 replay 证据。
切回普通 eager 时同时设 `PREFILL_FFN_GRAPH_MODE=EAGER` 和 layered=0，重启本次完整 worker 组。

所有 A/F 的 MBT 保持一致并能被 A TP 整除；rank 总数等于各角色 DP×TP，
`attn_ranks_per_dp=A TP`，跨节点 local DP 分段不重叠。
真实性能运行保持 `AFD_FORCE_BALANCED_TOPK_IDS=0`、`enable_force_eplb=false`，关闭 IO 调试日志。
验证路径、数学语义和停止边界见 [验证](validation.md) 与
[主线 FFN runtime](../../../../docs/npu/ASYNC_FFN_RUNTIME.md)。
