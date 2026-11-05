# E2E 场景矩阵与本次调整（RFC 附件）

状态：设计与覆盖变更提案，尚未修改测试或 CI。本文是 [E2E 框架 RFC](e2e_test_framework.md) 的正式附件，用于评审保留范围、替代关系和迁移验收；不是执行日志或设备通过报告。

基线：AFD `main@e778192d4c1cfd47e97be1881047532bfce7a7c0`（2026-11-05）。

## 阅读方式与统计口径

导航：[A 是现状基线](#current-matrix)；[精简条件是删除门槛](#deletion-gates)；[B 是条件满足后的目标变化](#target-matrix)，[完整目标清单折叠保留](#full-target-matrix)；[C 的独立入口、设备专项和主机检查](#other-entries)不计入 57 项服务场景。

> [!WARNING]
> **G17、G37：条件不满足就不能删；G02 本轮保留。**
> 当前删除门槛尚未满足；目标表不表示已实施删除，也不表示已取得替代验证结果。

- 当前服务矩阵按模型、后端、pytest 场景参数和 pod 布局区分，共 **57 项**。相同 node ID 的 GPU/NPU 实现分别计数；一个 multi-pod 场景不按 pod 数重复计数。
- 单独列出尚未接入 pytest 的设备脚本、算子验收、外部双流 driver，以及 **37 个主机检查**，它们不混入服务场景数量。
- 当前列描述代码要求执行的检查，不表示本轮已经取得设备结果。CI 列表示流水线选择；未选择不等于没有保留价值。
- 服务场景沿用当前存活、取消和清理规则。表内主要写出区分性的检查；NPU 清理例外及 FFN graph 不验收优雅关闭的限制另注明。
- 表中 `graph` 沿用当前角色的 CUDA/ACL Graph 配置，不能与 FFN `FULL` 混为同一场景。
- G/N/P 编号仅供本 RFC 两张矩阵对照，不是拟议 pytest ID。新场景名称仍需在实现时确认。

| 拟议动作 | 含义 | 当前行数 |
| --- | --- | ---: |
| 保留并迁移 | 保留对应行为；公共代码变化不减少验收 | 47 |
| 保留限定范围 | 保留功能 smoke 和已有路径检查，不宣称答案正确 | 2 |
| 增强 | 原场景保留，补当前缺失的路径证据 | 6 |
| 候选精简 | 拟移出固定回归组合；尚待确认替代、风险接受及执行条件 | 2 |

G02 保留两卡 pure-eager 的最小服务和低资源排障价值；它不在现有 CI 中，删除没有直接流水线收益。G17/G37 是两个有条件的精简候选，不代表完全等价替代，也不是精简数量上限；后续取舍依据实际价值与成本，不设删除数量目标。

<a id="current-matrix"></a>

## A. 当前全部服务场景及拟议调整

沿用现行 GSM8K profile：V2-Lite smoke 阈值为 0.25，较大样本及其他模型通常为 0.27；具体下限、生成参数、full run 和本地覆盖规则见 [现行契约](module/e2e_testing.md#accuracy-gate)。表内样本数为无调试覆盖时的配置。

### GPU 单机 / DeepSeek-V2-Lite / test_deepseek_v2_lite

入口：[tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py)，函数 `test_deepseek_v2_lite`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G01 / `baseline-graph` | 4 GPU；native DP4×TP1；graph | GSM8K 7题 | ready/merge | 保留并迁移 | 同模型原生非 AFD 对照。 |
| G02 / `afd-v2-eager-1a1f` | 2 GPU；A/F DP1×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | 两卡全程 eager 的最小完整服务及排障入口；G05 引入 graph、G03 需要四卡，均不等价。本轮保留，后续按使用和维护成本再评估。 |
| G03 / `afd-v2-eager-dp2` | 4 GPU；A/F DP2×TP1；eager | GSM8K 128题/12并发 | ready/merge | 保留并迁移 | pure eager、DP2 请求与收发；与 G08/G09 保持同一128题负载比较。 |
| G04 / `afd-v2-eager-tp2` | 4 GPU；A/F DP1×TP2；eager | GSM8K 7题 | ready/merge | 保留并迁移 | P2P 两角色 TP2 的纯 eager 接入。 |
| G05 / `afd-v2-graph-1a1f` | 2 GPU；A/F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | 最小 DP1/TP1 部署及实际 graph；保留最小进程/拓扑路径。 |
| G06 / `afd-v2-graph-dp2` | 4 GPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | ready/merge | 保留并迁移 | DP2 图重放及全部 A/F rank。 |
| G07 / `afd-v2-graph-tp2` | 4 GPU；A/F DP1×TP2；graph | GSM8K 7题；逐 rank graph | ready/merge | 保留并迁移 | 两角色 TP2 图重放。 |
| G08 / `afd-v2-eager-dbo-dp2` | 4 GPU；A/F DP2×TP1；eager＋DBO | GSM8K 128题/12并发；逐 rank 双阶段布局 | ready/merge | 保留并迁移 | 纯 eager 的 live DBO，不能由 graph+DBO 替代。 |
| G09 / `afd-v2-graph-dbo-dp2` | 4 GPU；A/F DP2×TP1；graph＋DBO | GSM8K 128题/12并发；逐 rank 连续 graph＋双阶段布局 | ready/merge | 保留并迁移 | DP2 下 graph 与两阶段布局结合。 |
| G10 / `afd-v2-graph-dbo-tp2` | 4 GPU；A/F DP1×TP2；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | DP1×TP2＋DBO，涉及无 DP 握手的 GPU ubatching 兼容路径。 |
| G11 / `afd-eager-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | 多个 Attention 合并到一个 FFN 的纯 eager 收发。 |
| G12 / `afd-graph-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | 同一非对称拓扑的普通单阶段 graph。 |
| G13 / `afd-graph-dbo-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | weekly | 保留并迁移 | 非对称拓扑与 graph+DBO 组合。 |

### GPU 单机 / DeepSeek-V2-Lite / test_deepseek_v2_lite_deepep_m2n

入口：[tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py)，函数 `test_deepseek_v2_lite_deepep_m2n`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G14 / `afd-deepep-m2n-eager-2a2f` | 4 GPU；A/F DP2×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | DeepSeek 共享专家与 Attention 路由的完整 pure-eager 接入。 |
| G15 / `afd-deepep-m2n-graph-2a2f` | 4 GPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | 同模型 DeepEP 普通图路径。 |
| G16 / `afd-deepep-m2n-graph-dbo-2a2f` | 4 GPU；A/F DP2×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | 同模型 DeepEP 的图/DBO 与共享专家接入。 |
| G17 / `afd-deepep-m2n-eager-tp2-2a2f` | 4 GPU；A DP1×TP2 / F DP2×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 候选精简 | 候选减少 TP2×eager 的完整模型组合；由 G14、G18 与 D01 专项分担。 见下文精简条件。 |
| G18 / `afd-deepep-m2n-graph-tp2-2a2f` | 4 GPU；A DP1×TP2 / F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | 模型共享专家及 TP token 切片后的 graph 接入。 |

### GPU 单机 / DeepSeek-V4-Flash / test_deepseek_v4_flash_gpu

入口：[tests/e2e/models/deepseek_v4_flash/test_gpu.py](../../tests/e2e/models/deepseek_v4_flash/test_gpu.py)，函数 `test_deepseek_v4_flash_gpu`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G19 / `baseline-dsv4-flash-eager` | 2 GPU；native DP2×TP1；eager | 两轮并发答案＋GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | 同一 DSV4 配置的 native eager 对照。 |
| G20 / `baseline-dsv4-flash-graph` | 2 GPU；native DP2×TP1；graph | 两轮并发答案＋GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | 同一 DSV4 配置的 native graph 对照。 |
| G21 / `afd-dsv4-flash-eager-2a2f` | 4 GPU；A/F DP2×TP1；eager | 两轮并发答案＋GSM8K 7题；Hash 配对/新鲜性 | 未被现有 CI 选择 | 保留并迁移 | DSV4 AFD eager 的 Hash 输入与结果一致性。 |
| G22 / `afd-dsv4-flash-graph-2a2f` | 4 GPU；A/F DP2×TP1；graph | 两轮并发答案＋GSM8K 7题；Hash 配对/新鲜性；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | DSV4 AFD graph 的 Hash 配对和跨轮新鲜性。 |

### GPU 单机 / DeepSeek-V4-Flash / test_deepseek_v4_flash_gpu_dbo

入口：[tests/e2e/models/deepseek_v4_flash/test_gpu.py](../../tests/e2e/models/deepseek_v4_flash/test_gpu.py)，函数 `test_deepseek_v4_flash_gpu_dbo`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G23 / `baseline-dsv4-flash-eager-dbo` | 2 GPU；native DP2×TP1；eager＋DBO | 两轮并发答案＋GSM8K 24题；逐 rank 双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | native eager DBO 对照及真实双阶段执行，不能由 graph DBO 替代。 |
| G24 / `baseline-dsv4-flash-graph-dbo` | 2 GPU；native DP2×TP1；graph＋DBO | 两轮并发答案＋GSM8K 24题；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | native graph DBO 对照与真实双子阶段证据。 |
| G25 / `afd-dsv4-flash-eager-dbo-2a2f` | 4 GPU；A/F DP2×TP1；eager＋DBO | 两轮并发答案＋GSM8K 24题；Hash 配对/新鲜性；逐 rank 双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | AFD pure eager DBO 下的 Hash 与双阶段执行。 |
| G26 / `afd-dsv4-flash-graph-dbo-2a2f` | 4 GPU；A/F DP2×TP1；graph＋DBO | 两轮并发答案＋GSM8K 24题；Hash 配对/新鲜性；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | AFD graph+DBO 的逐窗口 Hash 与连续重放。 |

### GPU 单机 / Qwen3.6 / test_qwen3_6

入口：[tests/e2e/models/qwen3_6/test_qwen3_6.py](../../tests/e2e/models/qwen3_6/test_qwen3_6.py)，函数 `test_qwen3_6`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G27 / `baseline-graph` | 4 GPU；native DP4×TP1；graph | GSM8K 7题 | weekly | 保留并迁移 | GDN 模型的 native 对照。 |
| G28 / `afd-eager-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | weekly | 保留并迁移 | GDN 模型 pure eager 状态路径。 |
| G29 / `afd-graph-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | weekly | 保留并迁移 | GDN 图重放状态路径；不恢复 GDN DBO。 |

### GPU 单机 / Qwen3 MoE / test_qwen3_moe

入口：[tests/e2e/models/qwen3_moe/test_qwen3_moe.py](../../tests/e2e/models/qwen3_moe/test_qwen3_moe.py)，函数 `test_qwen3_moe`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G30 / `baseline-graph` | 4 GPU；native DP4×TP1；graph | GSM8K 7题 | weekly | 保留并迁移 | Qwen3 模型的 native 对照。 |
| G31 / `afd-eager-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | weekly | 保留并迁移 | Qwen3 的 P2P pure-eager 模型接入。 |
| G32 / `afd-graph-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | weekly | 保留并迁移 | 同模型普通单阶段 P2P graph。 |
| G33 / `afd-graph-dbo-2a1f` | 3 GPU；A DP2×TP1 / F DP1×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | weekly | 保留并迁移 | 同模型 P2P graph+DBO。 |

### GPU 单机 / Qwen3 MoE / test_qwen3_moe_deepep_m2n

入口：[tests/e2e/models/qwen3_moe/test_qwen3_moe.py](../../tests/e2e/models/qwen3_moe/test_qwen3_moe.py)，函数 `test_qwen3_moe_deepep_m2n`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| G34 / `afd-deepep-m2n-eager-2a2f` | 4 GPU；A/F DP2×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | Qwen 无共享专家的 DeepEP pure-eager 接入，与 DeepSeek 接入不同。 |
| G35 / `afd-deepep-m2n-graph-2a2f` | 4 GPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | 同模型 DeepEP 单阶段 graph。 |
| G36 / `afd-deepep-m2n-graph-dbo-2a2f` | 4 GPU；A/F DP2×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | 同模型 DeepEP graph+DBO。 |
| G37 / `afd-deepep-m2n-eager-tp2-2a2f` | 4 GPU；A DP1×TP2 / F DP2×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 候选精简 | 候选减少 TP2×eager 的完整模型组合；由 G34、G38 与 D01 专项分担。 见下文精简条件。 |
| G38 / `afd-deepep-m2n-graph-tp2-2a2f` | 4 GPU；A DP1×TP2 / F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | Qwen 模型 TP2 分片和路由的 graph 接入。 |

### NPU 单机 / DeepSeek-V2-Lite / test_deepseek_v2_lite

入口：[tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py)，函数 `test_deepseek_v2_lite`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| N01 / `baseline-graph` | 4 NPU；native DP4×TP1；graph | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | 原生 Ascend 对照，不能用 CUDA baseline 替代。 |
| N02 / `afd-eager-2a2f` | 4 NPU；A/F DP2×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | Ascend CAMP2P pure-eager 服务生命周期。 |
| N03 / `afd-graph-2a2f` | 4 NPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | Ascend 普通单阶段 graph 与通信。 |
| N04 / `afd-graph-dbo-2a2f` | 4 NPU；A/F DP2×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | Ascend graph+DBO 与两阶段执行。 |
| N05 / `afd-eager-2a1f` | 3 NPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | 未被现有 CI 选择 | 保留并迁移 | NPU 非对称多 Attention 输入合并，pure eager。 |
| N06 / `afd-graph-2a1f` | 3 NPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 未被现有 CI 选择 | 保留并迁移 | NPU 非对称普通 graph。 |
| N07 / `afd-graph-dbo-2a1f` | 3 NPU；A DP2×TP1 / F DP1×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 未被现有 CI 选择 | 保留并迁移 | NPU 非对称 graph+DBO。 |

### NPU 单机 / DeepSeek-V4-Flash / test_deepseek_v4_flash_async_cam

入口：[tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py](../../tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py)，函数 `test_deepseek_v4_flash_async_cam`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| N08 / `afd-dsv4-flash-async-cam-dp2tp4-ep8` | 16 NPU；A DP2×TP4 / F DP8×TP1；A/F eager；A 配置 DBO，但不验收触发 | 10 并发答案、请求重叠；不验收 FFN FULL | 未被现有 CI 选择 | 保留并迁移 | 当前16 NPU异步部署和并发答案；与8 NPU FFN FULL测试的拓扑/目的不同。 |

### NPU 单机 / DeepSeek-V4-Flash / test_deepseek_v4_flash_sync_camp2p

入口：[tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py](../../tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py)，函数 `test_deepseek_v4_flash_sync_camp2p`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| N09 / `afd-dsv4-flash-sync-camp2p-2a2f` | A5，4 NPU；A/F DP2×TP1；graph，DBO 关闭 | 10 并发非空完成＋逐 rank graph；不验答案 | 未被现有 CI 选择 | 保留限定范围 | 保留 A5 graph 和并发管线；答案正确性仍未资格化。 |
| N10 / `afd-dsv4-flash-sync-camp2p-8a8f` | A3，16 NPU；A DP2×TP4 / F DP8×TP1；eager | 10 并发非空完成；不验答案 | 未被现有 CI 选择 | 保留限定范围 | 保留 A3 大拓扑并发管线；答案正确性仍未资格化。 |

### GPU multi-pod / DeepSeek-V2-Lite / test_multi_pod

入口：[tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py)，函数 `test_multi_pod`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| P01 / `afd-graph-2a2f-2pod-role-split` | 4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph | GSM8K、协调与清理；未验证 graph/DBO 实际执行 | 未被现有 CI 选择 | 增强 | 保留角色分离部署；补齐跨 pod graph 路径证明。 |
| P02 / `afd-graph-2a2f-2pod-interleaved` | 4 GPU；A DP2×TP1 / F DP2×TP1；交错布局；graph | GSM8K、协调与清理；未验证 graph/DBO 实际执行 | 未被现有 CI 选择 | 增强 | 保留交错部署及不同通信边界；补 graph 证明。 |
| P03 / `afd-graph-dbo-2a2f-2pod-role-split` | 4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph＋DBO | GSM8K、协调与清理；未验证 graph/DBO 实际执行 | 未被现有 CI 选择 | 增强 | 保留角色分离下双阶段执行；补跨 pod graph+DBO 证明。 |
| P04 / `afd-graph-dbo-2a2f-2pod-interleaved` | 4 GPU；A DP2×TP1 / F DP2×TP1；交错布局；graph＋DBO | GSM8K、协调与清理；未验证 graph/DBO 实际执行 | 未被现有 CI 选择 | 增强 | 保留交错下双阶段执行；补跨 pod graph+DBO 证明。 |

### GPU multi-pod / Qwen3 MoE / test_multi_pod_deepep_m2n

入口：[tests/e2e/models/qwen3_moe/test_qwen3_moe_multi_pod.py](../../tests/e2e/models/qwen3_moe/test_qwen3_moe_multi_pod.py)，函数 `test_multi_pod_deepep_m2n`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| P05 / `afd-deepep-m2n-eager-2a2f-2pod-role-split` | 4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；eager | GSM8K 默认100题、协调与清理；无需 graph/DBO 证明 | 未被现有 CI 选择 | 保留并迁移 | 保留 DeepEP 跨 pod pure-eager 服务。 |
| P06 / `afd-deepep-m2n-graph-2a2f-2pod-role-split` | 4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph | GSM8K 默认100题、协调与清理；未验证 graph/DBO 实际执行 | 未被现有 CI 选择 | 增强 | 保留 DeepEP 跨 pod graph；补逐 rank 证明。 |
| P07 / `afd-deepep-m2n-graph-dbo-2a2f-2pod-role-split` | 4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph＋DBO | GSM8K 默认100题、协调与清理；未验证 graph/DBO 实际执行 | 未被现有 CI 选择 | 增强 | 保留 DeepEP 跨 pod graph+DBO；补匹配窗口和布局证明。 |

### NPU multi-pod / DeepSeek-V4-Flash / test_deepseek_v4_flash_multi_pod

入口：[tests/e2e/models/deepseek_v4_flash/test_deepseek_v4_flash_multi_pod.py](../../tests/e2e/models/deepseek_v4_flash/test_deepseek_v4_flash_multi_pod.py)，函数 `test_deepseek_v4_flash_multi_pod`；下表“场景参数”拼入该函数的 pytest node ID。

| 编号 / 场景参数 | 部署与模式 | 当前负载及验收 | 现有 CI 选择 | 本次拟调整 | 独立价值、替代或前置条件 |
| --- | --- | --- | --- | --- | --- |
| P08 / `2pod-role-split` | 16 NPU；A DP2×TP4 / F DP8×TP1；按角色分 pod；A/F eager；A 配置 DBO，但不验收触发 | 10 并发答案、协调与清理；无 graph/DBO 路径证明 | 未被现有 CI 选择 | 保留并迁移 | 保留异步 CAM 跨 pod 角色分离及答案；不宣称 FFN FULL。 |
| P09 / `2pod-interleaved` | 16 NPU；A DP2×TP4 / F DP8×TP1；交错布局；A/F eager；A 配置 DBO，但不验收触发 | 10 并发答案、协调与清理；无 graph/DBO 路径证明 | 未被现有 CI 选择 | 保留并迁移 | 保留 TP 组局部化的交错放置及答案；不宣称 FFN FULL。 |

<a id="deletion-gates"></a>

### 精简候选的条件

G02 本轮保留，不由 G05 的 `FULL_DECODE_ONLY` 或 G03 的 DP2 eager 代替。下表只适用于 G17/G37。

| 候选 | 保留或替代项 | 实施精简前必须满足的条件 |
| --- | --- | --- |
| G17 | G14：同模型 DeepEP pure eager；G18：同模型 TP2 graph；D01：GPU TP2 专项 | G14/G18 保留验证通过；下述 `D01-TP2-4GPU` 任务实际取得两项通过、零 skip 记录，并明确后续执行归属。任一条件缺失就保留 G17；还须明确接受不再独立回归“V2-Lite＋DeepEP＋TP2＋全程 eager”的完整服务组合 |
| G37 | G34：同模型 DeepEP pure eager；G38：同模型 TP2 graph；D01：GPU TP2 专项 | G34/G38 保留验证通过；同一 `D01-TP2-4GPU` 任务取得有效零 skip 记录并明确执行归属。不能用 DeepSeek 的模型 E2E 替代 Qwen 自身接入，条件缺失就保留 G37；还须明确接受不再独立回归“Qwen3＋DeepEP＋TP2＋全程 eager”的完整服务组合 |

TP1 pure-eager 服务、TP2 graph 服务和 TP2 eager 专项分别通过，不证明 TP2 pure-eager 完整模型/调度/生命周期的组合完全等价。实施 PR 必须接受上述退出范围，不再声称保有该组合的独立完整服务覆盖；出现该组合专有回归，或替代项改弱、停止执行时，重新评估并按需要恢复对应服务场景。

D01 的 [TP 切片专项](../../tests/gpu/test_deepep_m2n_gpu.py)使用真实 GPU，覆盖 TP2、不同 token 数、空 DP 和单/双 lane；它不启动完整 vLLM 服务，必须与保留的模型 E2E 配合。当前 CI 未选择这些专项，硬件不足会 skip，所以“代码存在”不满足替代条件。

<a id="d01-manual-task"></a>

#### 手工任务 `D01-TP2-4GPU`

这是本 RFC 为 G17/G37 指定的替代验证任务定义，当前尚未执行、没有资格记录，也没有新增 Buildkite 任务。执行者在有效调度分配内取得单机四张 SM90+ GPU，按 [GPU 专项环境要求](../../tests/gpu/README.md#requirements)准备固定版本的运行环境；从仓库根目录运行，沿用分配提供的设备可见性。

固定选择 `test_attention_tp_peers_dispatch_disjoint_slices` 的 `2F-1DPx2TP` 子集：Attention DP1×TP2、FFN DP2×TP1，BF16 与 FP8 两项。六卡 `2F-2DPx2TP` 不混入该四卡任务。

```bash
set -e
export AFD_D01_RESULTS="$(mktemp -d /tmp/afd-d01-tp2-4gpu.XXXXXX)"
python -m pytest -sv -m gpu \
  "tests/gpu/test_deepep_m2n_gpu.py::test_attention_tp_peers_dispatch_disjoint_slices[2F-1DPx2TP-bf16]" \
  "tests/gpu/test_deepep_m2n_gpu.py::test_attention_tp_peers_dispatch_disjoint_slices[2F-1DPx2TP-fp8]" \
  --junitxml="$AFD_D01_RESULTS/junit.xml" \
  --basetemp="$AFD_D01_RESULTS/pytest"

python - <<'PY'
import os
import xml.etree.ElementTree as ET
from pathlib import Path

cases = ET.parse(Path(os.environ["AFD_D01_RESULTS"]) / "junit.xml").findall(".//testcase")
assert len(cases) == 2, "D01 must execute exactly the two selected GPU cases"
assert all(
    case.find(tag) is None
    for case in cases
    for tag in ("failure", "error", "skipped")
), "D01 requires two passes and zero failures, errors or skips"
PY
```

执行归属：精简实施 PR 的作者负责安排首次运行并提供记录；后续影响 DeepEP Attention TP 切片、dispatch/combine 或 payload 路径的 PR，由该 PR 作者负责重跑此任务并提供记录。无法落实执行资源和责任时，G17/G37 不退出。

记录所测 AFD/runtime 版本、设备、完整命令和退出状态，并保存 JUnit、控制台及 rank 日志到任务产物；记录应对应实施 PR 的代码和运行环境，链接写入 PR 描述。pytest 退出成功但存在 skip、仅收集、缺少依赖或没有运行记录均不满足删除门槛。上述校验不代表本轮已经运行设备。

上述条件尚未完成。实施前不能仅删除两个候选 node ID，也不能把产品的相应运行配置判定为不再支持。当前 required gate 没有选中这两个候选，本提案不删减现有 ready/merge 或 weekly 的必需集合。

<a id="target-matrix"></a>

## B. 条件满足后的目标变化（提案）

本表只列 2 项候选删除、3 个新增入口和 6 项现有验收增强。未列出的 49 项当前服务场景保留迁移，其中 N09/N10 继续限定 smoke 范围。所有删除动作受前述门槛约束；A 表始终保留完整现状基线。

G02 保留；若 G17/G37 均满足条件，57 个当前场景减少 2 个固定回归组合，再增加 3 个入口，目标为 58 项。某项条件不成立就补回对应原行，两项均不精简时为 60 项。数量是方案结果，不是目标。

保留 **15 个唯一 node ID（ready 和 merge 共用 7 个，weekly 8 个）**。新硬件任务的资源、频率和预算仍需确定；“拟定期/专项”不表示已有流水线或设备证据。目标分批实现，不要求新增能力与首批框架提取在同一 PR 完成。

| 动作 | 编号 / 场景 | 条件满足后的变化 | 生效条件 / 资源 |
| --- | --- | --- | --- |
| 候选删除 | G17 / V2-Lite DeepEP eager TP2 | 保留 G14/G18 模型接入，TP2 eager 通信由 D01 配合看护 | 同模型保留项通过，D01零 skip记录与执行归属齐全，并接受退出完整组合的风险，否则不删 |
| 候选删除 | G37 / Qwen3 DeepEP eager TP2 | 保留 G34/G38 模型接入，TP2 eager 通信由 D01 配合看护 | 同模型保留项通过，D01零 skip记录与执行归属齐全，并接受退出完整组合的风险，否则不删 |
| 新增 | G39 / V2-Lite graph 3A2F | 原生参考→3A2F graph 双部署；逐请求对照、不同长度/负载、各 FFN 组参与及清理 | 单机 5 GPU；定期/专项拟议，不扩入四卡 PR gate |
| 新增入口 | N11 / V2-Lite eager DBO 2A2F | 将既有手工配置纳入稳定 pytest 入口；保持 native 32/512 阈值，在指定 prefill-heavy 窗口证明 live 两阶段执行，不宣称 decode DBO 资格 | 单机 4 NPU；运行资源与频率待落实 |
| 新增 | N12 / DSV4 async CAM FFN FULL 4A4F | 自包含 eager/FULL 双部署；长度、重复输入、真实并发、部署级 capture、逐请求窗口 replay 增量、token 对照及两段优雅关闭 | 单机 8 NPU；采用 `cleanup_profile=strict`，关闭资格仍待验证 |
| 增强 | P01 / V2-Lite graph，按角色分 pod | 原结果验收外增加逐 A/F rank 的跨 pod graph 窗口证据 | 4 GPU、双 pod；先落实证据传输与协调 |
| 增强 | P02 / V2-Lite graph，交错布局 | 同上，保留交错放置的通信边界 | 4 GPU、双 pod；先落实证据传输与协调 |
| 增强 | P03 / V2-Lite graph＋DBO，按角色分 pod | 增加跨 pod 匹配双阶段布局及连续 graph 重放证据 | 4 GPU、双 pod；先落实证据传输与协调 |
| 增强 | P04 / V2-Lite graph＋DBO，交错布局 | 同上，保留交错放置的通信边界 | 4 GPU、双 pod；先落实证据传输与协调 |
| 增强 | P06 / Qwen3 DeepEP graph，按角色分 pod | 原 GSM8K 结果验收外增加逐 A/F rank 的跨 pod graph 证据 | 4 GPU、双 pod；先落实证据传输与协调 |
| 增强 | P07 / Qwen3 DeepEP graph＋DBO，按角色分 pod | 增加跨 pod 匹配双阶段布局及连续 graph 重放证据 | 4 GPU、双 pod；先落实证据传输与协调 |

<a id="full-target-matrix"></a>

### 完整目标清单（条件满足时为 58 项）

<details>
<summary>展开 GPU 单机 37 项、NPU 单机 12 项、multi-pod 9 项</summary>

以下保留 G02，并假设 G17/G37 删除门槛均满足；条件不成立时，对应候选从 A 表补回。

#### GPU 单机：37 项

| 编号 / 目标场景 | 模型、部署与模式 | 目标看护 | 调度建议 | 来源与变化 |
| --- | --- | --- | --- | --- |
| G01 / `baseline-graph` | DeepSeek-V2-Lite；4 GPU；native DP4×TP1；graph | GSM8K 7题 | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G02 / `afd-v2-eager-1a1f` | DeepSeek-V2-Lite；2 GPU；A/F DP1×TP1；eager | GSM8K 7题 | 手工/排障；受影响变更触发 | 保留两卡全程 eager 的最小完整服务 |
| G03 / `afd-v2-eager-dp2` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；eager | GSM8K 128题/12并发 | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G04 / `afd-v2-eager-tp2` | DeepSeek-V2-Lite；4 GPU；A/F DP1×TP2；eager | GSM8K 7题 | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G05 / `afd-v2-graph-1a1f` | DeepSeek-V2-Lite；2 GPU；A/F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G06 / `afd-v2-graph-dp2` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G07 / `afd-v2-graph-tp2` | DeepSeek-V2-Lite；4 GPU；A/F DP1×TP2；graph | GSM8K 7题；逐 rank graph | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G08 / `afd-v2-eager-dbo-dp2` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；eager＋DBO | GSM8K 128题/12并发；逐 rank 双阶段布局 | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G09 / `afd-v2-graph-dbo-dp2` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；graph＋DBO | GSM8K 128题/12并发；逐 rank 连续 graph＋双阶段布局 | 保留 ready/merge | 保留覆盖；迁移公共框架 |
| G10 / `afd-v2-graph-dbo-tp2` | DeepSeek-V2-Lite；4 GPU；A/F DP1×TP2；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G11 / `afd-eager-2a1f` | DeepSeek-V2-Lite；3 GPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G12 / `afd-graph-2a1f` | DeepSeek-V2-Lite；3 GPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G13 / `afd-graph-dbo-2a1f` | DeepSeek-V2-Lite；3 GPU；A DP2×TP1 / F DP1×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 保留 weekly | 保留覆盖；迁移公共框架 |
| G14 / `afd-deepep-m2n-eager-2a2f` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；eager | GSM8K 7题 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G15 / `afd-deepep-m2n-graph-2a2f` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G16 / `afd-deepep-m2n-graph-dbo-2a2f` | DeepSeek-V2-Lite；4 GPU；A/F DP2×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G18 / `afd-deepep-m2n-graph-tp2-2a2f` | DeepSeek-V2-Lite；4 GPU；A DP1×TP2 / F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G19 / `baseline-dsv4-flash-eager` | DeepSeek-V4-Flash；2 GPU；native DP2×TP1；eager | 两轮并发答案＋GSM8K 7题 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G20 / `baseline-dsv4-flash-graph` | DeepSeek-V4-Flash；2 GPU；native DP2×TP1；graph | 两轮并发答案＋GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G21 / `afd-dsv4-flash-eager-2a2f` | DeepSeek-V4-Flash；4 GPU；A/F DP2×TP1；eager | 两轮并发答案＋GSM8K 7题；Hash 配对/新鲜性 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G22 / `afd-dsv4-flash-graph-2a2f` | DeepSeek-V4-Flash；4 GPU；A/F DP2×TP1；graph | 两轮并发答案＋GSM8K 7题；Hash 配对/新鲜性；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G23 / `baseline-dsv4-flash-eager-dbo` | DeepSeek-V4-Flash；2 GPU；native DP2×TP1；eager＋DBO | 两轮并发答案＋GSM8K 24题；逐 rank 双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G24 / `baseline-dsv4-flash-graph-dbo` | DeepSeek-V4-Flash；2 GPU；native DP2×TP1；graph＋DBO | 两轮并发答案＋GSM8K 24题；逐 rank 连续 graph＋双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G25 / `afd-dsv4-flash-eager-dbo-2a2f` | DeepSeek-V4-Flash；4 GPU；A/F DP2×TP1；eager＋DBO | 两轮并发答案＋GSM8K 24题；Hash 配对/新鲜性；逐 rank 双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G26 / `afd-dsv4-flash-graph-dbo-2a2f` | DeepSeek-V4-Flash；4 GPU；A/F DP2×TP1；graph＋DBO | 两轮并发答案＋GSM8K 24题；Hash 配对/新鲜性；逐 rank 连续 graph＋双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G27 / `baseline-graph` | Qwen3.6；4 GPU；native DP4×TP1；graph | GSM8K 7题 | 保留 weekly | 保留覆盖；迁移公共框架 |
| G28 / `afd-eager-2a1f` | Qwen3.6；3 GPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | 保留 weekly | 保留覆盖；迁移公共框架 |
| G29 / `afd-graph-2a1f` | Qwen3.6；3 GPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 保留 weekly | 保留覆盖；迁移公共框架 |
| G30 / `baseline-graph` | Qwen3 MoE；4 GPU；native DP4×TP1；graph | GSM8K 7题 | 保留 weekly | 保留覆盖；迁移公共框架 |
| G31 / `afd-eager-2a1f` | Qwen3 MoE；3 GPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | 保留 weekly | 保留覆盖；迁移公共框架 |
| G32 / `afd-graph-2a1f` | Qwen3 MoE；3 GPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | 保留 weekly | 保留覆盖；迁移公共框架 |
| G33 / `afd-graph-dbo-2a1f` | Qwen3 MoE；3 GPU；A DP2×TP1 / F DP1×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 保留 weekly | 保留覆盖；迁移公共框架 |
| G34 / `afd-deepep-m2n-eager-2a2f` | Qwen3 MoE；4 GPU；A/F DP2×TP1；eager | GSM8K 7题 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G35 / `afd-deepep-m2n-graph-2a2f` | Qwen3 MoE；4 GPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G36 / `afd-deepep-m2n-graph-dbo-2a2f` | Qwen3 MoE；4 GPU；A/F DP2×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G38 / `afd-deepep-m2n-graph-tp2-2a2f` | Qwen3 MoE；4 GPU；A DP1×TP2 / F DP2×TP1；graph | GSM8K 7题；逐 rank graph | 专项/本地；暂不扩入 PR gate | 保留覆盖；迁移公共框架 |
| G39 / `afd-v2-graph-3a2f` | DeepSeek-V2-Lite；最多5 GPU；native DP4×TP1参考→A DP3×TP1 / F DP2×TP1 graph目标 | 本次原生参考逐请求对照；固定输入的串行/并发窗口；目标各 FFN 组与逐 rank graph；两段 strict 清理 | 5 GPU 定期/专项（拟） | 新增；覆盖同一运行内 FFN 子组大小不同。 |

#### NPU 单机：12 项

| 编号 / 目标场景 | 模型、部署与模式 | 目标看护 | 调度建议 | 来源与变化 |
| --- | --- | --- | --- | --- |
| N01 / `baseline-graph` | DeepSeek-V2-Lite；4 NPU；native DP4×TP1；graph | GSM8K 7题 | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N02 / `afd-eager-2a2f` | DeepSeek-V2-Lite；4 NPU；A/F DP2×TP1；eager | GSM8K 7题 | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N03 / `afd-graph-2a2f` | DeepSeek-V2-Lite；4 NPU；A/F DP2×TP1；graph | GSM8K 7题；逐 rank graph | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N04 / `afd-graph-dbo-2a2f` | DeepSeek-V2-Lite；4 NPU；A/F DP2×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N05 / `afd-eager-2a1f` | DeepSeek-V2-Lite；3 NPU；A DP2×TP1 / F DP1×TP1；eager | GSM8K 7题 | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N06 / `afd-graph-2a1f` | DeepSeek-V2-Lite；3 NPU；A DP2×TP1 / F DP1×TP1；graph | GSM8K 7题；逐 rank graph | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N07 / `afd-graph-dbo-2a1f` | DeepSeek-V2-Lite；3 NPU；A DP2×TP1 / F DP1×TP1；graph＋DBO | GSM8K 24题/12并发；逐 rank 连续 graph＋双阶段布局 | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N08 / `afd-dsv4-flash-async-cam-dp2tp4-ep8` | DeepSeek-V4-Flash；16 NPU；A DP2×TP4 / F DP8×TP1；A/F eager；A 配置 DBO，但不验收触发 | 10 并发答案、请求重叠；不验收 FFN FULL | NPU 专项；未有持续 CI | 保留覆盖；迁移公共框架 |
| N09 / `afd-dsv4-flash-sync-camp2p-2a2f` | DeepSeek-V4-Flash；A5，4 NPU；A/F DP2×TP1；graph，DBO 关闭 | 10 并发非空完成＋逐 rank graph；不验答案 | NPU 专项；未有持续 CI | 保留 smoke 边界；正确性待独立验收 |
| N10 / `afd-dsv4-flash-sync-camp2p-8a8f` | DeepSeek-V4-Flash；A3，16 NPU；A DP2×TP4 / F DP8×TP1；eager | 10 并发非空完成；不验答案 | NPU 专项；未有持续 CI | 保留 smoke 边界；正确性待独立验收 |
| N11 / `afd-eager-dbo-2a2f` | DeepSeek-V2-Lite；4 NPU；A/F DP2×TP1；pure eager＋DBO | GSM8K 24题/12并发、0.25；固定 prefill-heavy 窗口的 live 双非空阶段及 A/F rank 布局；不宣称 decode DBO；存活与清理 | NPU 专项（拟） | 现有手工配置增加稳定 pytest 入口；不降低阈值制造触发。 |
| N12 / `afd-dsv4-flash-async-cam-ffn-full-4a4f` | DeepSeek-V4-Flash；8 NPU；A DP1×TP4 / F DP4×TP1；A eager；F eager参考→FULL目标 | 多长度、同输入重复、真实 HTTP 并发；本次部署逐 FFN capture、逐请求窗口 replay 增量和模式；eager/FULL token 对照；两段 strict 关闭 | 8 NPU 定期/专项（拟） | 新增自包含 #509 比较测试；关闭项当前尚未资格化。 |

#### multi-pod：9 项

| 编号 / 目标场景 | 模型、部署与模式 | 目标看护 | 调度建议 | 来源与变化 |
| --- | --- | --- | --- | --- |
| P01 / `afd-graph-2a2f-2pod-role-split` | DeepSeek-V2-Lite；4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph | GSM8K、协调与清理；具名跨 pod 窗口、每个 A/F rank 的真实 graph 重放；存活/清理 | 多 pod 专项；需准备环境 | 增强原场景；跨 pod 传输证据，不新增重复配置 |
| P02 / `afd-graph-2a2f-2pod-interleaved` | DeepSeek-V2-Lite；4 GPU；A DP2×TP1 / F DP2×TP1；交错布局；graph | GSM8K、协调与清理；具名跨 pod 窗口、每个 A/F rank 的真实 graph 重放；存活/清理 | 多 pod 专项；需准备环境 | 增强原场景；跨 pod 传输证据，不新增重复配置 |
| P03 / `afd-graph-dbo-2a2f-2pod-role-split` | DeepSeek-V2-Lite；4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph＋DBO | GSM8K、协调与清理；具名跨 pod 窗口、每个 A/F rank 的匹配两阶段布局及连续 graph 重放；存活/清理 | 多 pod 专项；需准备环境 | 增强原场景；跨 pod 传输证据，不新增重复配置 |
| P04 / `afd-graph-dbo-2a2f-2pod-interleaved` | DeepSeek-V2-Lite；4 GPU；A DP2×TP1 / F DP2×TP1；交错布局；graph＋DBO | GSM8K、协调与清理；具名跨 pod 窗口、每个 A/F rank 的匹配两阶段布局及连续 graph 重放；存活/清理 | 多 pod 专项；需准备环境 | 增强原场景；跨 pod 传输证据，不新增重复配置 |
| P05 / `afd-deepep-m2n-eager-2a2f-2pod-role-split` | Qwen3 MoE；4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；eager | GSM8K 默认100题、协调与清理；无需 graph/DBO 证明 | 多 pod 专项；需准备环境 | 保留覆盖；迁移公共框架 |
| P06 / `afd-deepep-m2n-graph-2a2f-2pod-role-split` | Qwen3 MoE；4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph | GSM8K 默认100题、协调与清理；具名跨 pod 窗口、每个 A/F rank 的真实 graph 重放；存活/清理 | 多 pod 专项；需准备环境 | 增强原场景；跨 pod 传输证据，不新增重复配置 |
| P07 / `afd-deepep-m2n-graph-dbo-2a2f-2pod-role-split` | Qwen3 MoE；4 GPU；A DP2×TP1 / F DP2×TP1；按角色分 pod；graph＋DBO | GSM8K 默认100题、协调与清理；具名跨 pod 窗口、每个 A/F rank 的匹配两阶段布局及连续 graph 重放；存活/清理 | 多 pod 专项；需准备环境 | 增强原场景；跨 pod 传输证据，不新增重复配置 |
| P08 / `2pod-role-split` | DeepSeek-V4-Flash；16 NPU；A DP2×TP4 / F DP8×TP1；按角色分 pod；A/F eager；A 配置 DBO，但不验收触发 | 10 并发答案、协调与清理；无 graph/DBO 路径证明 | 多 pod 专项；需准备环境 | 保留覆盖；迁移公共框架 |
| P09 / `2pod-interleaved` | DeepSeek-V4-Flash；16 NPU；A DP2×TP4 / F DP8×TP1；交错布局；A/F eager；A 配置 DBO，但不验收触发 | 10 并发答案、协调与清理；无 graph/DBO 路径证明 | 多 pod 专项；需准备环境 | 保留覆盖；迁移公共框架 |

</details>

<a id="new-case-acceptance"></a>

### 新增场景的验收定义

以下是实现与资格验证的固定目标，当前没有新增设备通过记录。需要的任务、比较或证据能力先按正文规则独立扩展和验证，再验收仅增加组合声明的增量。

#### G39：原生参考与 3A2F graph 的逐请求比较

- **部署**：同一测试顺序运行原生 DP4×TP1 graph 参考（复用 G01 部署 profile，使用4卡）和 P2P 3A2F graph 目标（5卡）。两段独立目录，参考完成并 strict 关闭后再启动目标；参考不走 AFD 分组，避免用相同 connector 的 eager 错误给 graph 背书。
- **固定输入与生成**：GSM8K test 前10题，使用现有8-shot渲染，保存完整 prompt；依次执行串行窗口和10个真实 HTTP 客户端并发窗口。温度0、seed0、生成上限512，checkpoint、tokenizer、量化与生成参数固定；输入长度由这组固定题目决定并记录，不按运行结果临时挑题。
- **oracle**：按“窗口＋请求索引＋完整输入及生成参数”关联本次原生结果，逐条精确比较完整返回文本和终止原因。参考响应须完整、非空且在对应窗口内可区分；参考不完整或不可区分时资格失败，不能退化成 GSM8K 总分或部分匹配。它验收 AFD 相对原生的结果一致性，不新增模型准确率承诺。
- **路径与关闭**：目标拓扑须为 F0 接 A0/A1、F1 接 A2；每个目标窗口证明三个 A rank、两个 F rank 的真实请求执行及 graph replay，不能由其他窗口累计计数替代。各 F rank 参与加上逐请求 oracle 共同验收分组及返回；两段均 strict 关闭。公共任务或比较能力未具备时，不能把整个 G39 工作称作纯配置新增。

#### N11：指定 prefill-heavy 负载中的 pure-eager live DBO

- **部署与阈值**：4 NPU CAMP2P，A/F DP2×TP1，全程 eager，保留 native decode/prefill 阈值32/512。
- **窗口与负载**：`live-dbo` 使用4个互异的1024-token新输入、4个真实 HTTP 并发、生成上限64；固定输入在模型任务中声明，记录 tokenizer 后的实际长度，避免前缀缓存复用替代新输入。随后独立 `gsm8k` 窗口沿用 N04 的24题、12并发、8-shot、生成上限512及0.25阈值。
- **判定范围**：`live-dbo` 中每个要求的 A/F rank 均须有真实请求的两个非空 microbatch，双方存在匹配布局，实际执行保持 eager；capture/profile/warmup 不计。GSM8K 单独验结果，不能用它的累计计数代替前一窗口证据。
- **结论边界**：只证明该指定工作负载中发生 live DBO，不分别宣称 prefill DBO、decode DBO 均已资格化。并发4或12本身不算触发证明；未出现要求的布局就失败，不降低阈值取得通过。

#### N12：部署级 capture 与请求窗口 replay 分开验收

- **固定范围**：现有单流4A4F、8 NPU、容量128的 profile；A eager，F eager参考→FULL目标；生成8 token、seed1024。两段目录、参考身份及失败传播遵循正文 §6.3。
- **请求窗口**：输入长度16/128/129/257分别成窗口；128长度的同输入连续两次作为重复窗口；长度16与257的两个独立 HTTP 请求作为并发窗口，并检查实际重叠。两段重放相同输入和生成参数，逐请求精确比较 token IDs。
- **capture**：本次 FULL 部署的全部预期 FFN rank 建立预期图，沿用现有每rank一次 capture 的要求；不要求每个请求窗口重新 capture。
- **replay与模式**：每个上述窗口各取独立前后快照，所有要求 FFN rank 的真实请求 replay 有增量，实际模式为 FULL，无意外 fallback；Attention 保持预期 eager/MRV2 路径。前一窗口累计 replay 不给后一窗口背书。
- **关闭**：eager、FULL分别完成 strict 关闭与资源回收；任一段失败则整个用例失败。旧 L01 的强制清理通过不能代替此资格。4A8F、W8A8、双流仍单独资格化。

### 现有场景增强与范围限制

- **P01–P04、P06–P07**：共六个 graph/DBO 场景增强具名跨 pod 窗口和逐 rank 证据。当前 [multi-pod runner](../../tests/e2e/multi_pod/runner.py) 因 marker 文件不跨 pod 共享而关闭 MRV2 证据，需解决真实传输与协调。P05、P08、P09 保留其 eager 结果验收，不自动宣称 graph/DBO 看护。
- **N09、N10**：保留 smoke 的准确范围。N09 当前代码还要求 graph 证据；两者均未验收答案正确性。当前配置注释记录 A5 答案不可靠、A3 尚未对照答案 oracle，不能将两者都表述为已确认的同一缺陷。解决已知问题并完成各自正确性验收后升级现有场景，不另复制一个同配置弱 smoke。
- **双流 FFN**：若继续作为支持承诺，应形成单独可复现回归；暂列 L05 后续项，不由单流 N12 结果推导。4A8F/W8A8 同样需要独立目标设备资格，不自动进入 N12 默认范围。

<a id="other-entries"></a>

## C. 独立入口、设备专项和主机检查

这些入口不能因不属于 57 项服务参数化场景而遗漏，也不能将其数量混为模型 E2E 数量。

| 编号 / 当前入口 | 当前覆盖及限制 | 本次调整 / 目标去向 | 统计方式 |
| --- | --- | --- | --- |
| L01 / [ffn_graph.py](../../tests/e2e/models/deepseek_v4_flash/ffn_graph.py) | 单流；A eager、F eager/FULL；4A4F 或可选4A8F；多长度和双 prompt batch；token 对照及逐rank证据；不验收优雅关闭 | 复用到 N12 自包含对照；迁出重复部署代码，保留独有证明。4A8F 仍待设备资格；旧 CLI 在使用者迁移后再决定退出 | 一组独立 driver 配置；现无 pytest/CI 入口，不计入当前57 |
| L02 / [dspark_eager.py](../../tests/e2e/models/deepseek_v4_flash/dspark_eager.py) | 既有服务的长度、batch、真实并发、stop、流取消后再请求、MTP活跃度；可选FFN及逐层数值证明；含PD变体 | 保留为模型专用任务及检查，可由场景调用；当前CLI在部署入口迁移完成前保留。未选数值证明时不宣称已验证 | 不启动/关闭服务，作为负载工具单列 |
| L03 / [async_cam_roundtrip.py](../../tests/e2e/operators/async_cam_roundtrip.py) | TP1/2/4 × FP16/BF16 × quant0/1，共12个基础配置，每次覆盖边界、空rank和重复窗口 | 保留独立算子/通信数值验收；不重复扩成12个完整模型场景 | 算子/通信设备专项，不计入57 |
| L04 / [NPU eager DBO 手工配置](../../tests/e2e/runner.py) | 存在 afd-eager-dbo-2a2f 配置及native阈值，无对应pytest参数 | 由 N11 正式暴露和验证；不是因缺少pytest调用就直接判定死代码 | 新增的是自动化入口，部署模式已有 |
| L05 / [双流资格验证说明](../npu/ASYNC_FFN_RUNTIME.md#validation) | 当前真实资格依赖外部driver；仓库中的L01是单流 | 按支持承诺补可复现入口；单列后续工作，不混入N12 | 后续设备覆盖，暂不计目标58 |
| D01 / [DeepEP GPU 专项](../../tests/gpu/test_deepep_m2n_gpu.py) | 真实TP切片、传输、专家输出、eager/graph、双lane、mixed-mode和关闭；硬件不足会skip；当前CI未选择 | 按 [D01-TP2-4GPU 手工任务](#d01-manual-task)取得明确四卡子集的零 skip 记录并落实执行归属，才能精简 G17/G37 | 更轻的设备回归；不是完整模型服务E2E，也不是新写测试 |
| H01 / [DSpark证据校验器](../../tests/e2e/models/deepseek_v4_flash/test_dspark_graph_checker.py) | 19项主机检查，包括缺rank、缺层、数值证明等负面输入 | 保留拒绝错误证明的有效覆盖；不作为模型或GPU/NPU通过证据 | 主机检查19 |
| H02 / [部署脚本检查](../../tests/e2e/test_deploy_dsv4_flash_async_skill.py) | 16项CLI、环境、配置与结果处理的主机集成检查 | 保留有效契约，随对应部署接口迁移；不计作PD模型正确性验收 | 主机检查16 |
| H03 / [runner取消检查](../../tests/e2e/test_runner.py) | 2项真实本地主机进程、阻塞请求和SIGINT/SIGTERM取消 | 保留进程归属和取消回收的主机集成看护 | 主机检查2 |

H01 的负面检查与正常设备路径验证互补。不能因正常E2E能通过就删除校验器对错误证据的拒绝测试。

<a id="code-consolidation"></a>

## D. 维护代码的合并，不作为场景删除计数

| 当前重复 | 目标归属 | 需要保留的差异 |
| --- | --- | --- |
| 五份模型入口 _required_env 定义 | [environment.required_env](../../tests/e2e/environment.py) | 当前函数体一致，可删除重复定义 |
| V2-Lite、Qwen3、Qwen3.6 的公共资产准备与命令装配 | 模型场景配置＋公共入口/部署模块 | 模型、平台、设备布局、运行参数、GSM8K配置及覆盖优先级 |
| 专用driver的启动、等待、取消与清理 | 公共部署和已有进程工具 | NPU限定清理例外与“是否验收优雅关闭”的区别 |
| 相同的阶段/窗口组织流程 | 公共执行流程及主机侧检查 | MRV2、FFN FULL、DSpark 的不同探针和判定内容 |

合并同构参数表、减少函数和文件，不等于减少了设备执行组合或GPU时间。矩阵分别记录覆盖变化和维护代码变化，避免以删除行数替代收益。

<a id="maintenance"></a>

## E. 决策与持续维护

1. 先确认每行的保留、增强或候选精简，以及替代项实际执行的归属。不能把“建议”写成已验证结论。
2. 精简与替代验证一起评审；替代项不支持目标硬件、没有运行或发生skip时，原覆盖不能默默退出。
3. 当前gate、样本数、阈值、逐rank/窗口要求和清理失败语义保持有效；有意修改支持范围或验收标准需单独说明。
4. 新增硬件看护分别确定资源与频率，避免用框架重构隐含增加资源承诺。
5. 每批实现对照本附件检查实际去向。落地后稳定场景清单与运行说明归 E2E README，行为约束归模块契约；本附件保留该设计的变更映射，不写原始执行日志。

### 场景组的执行归属与触发

相关变更 PR 作者负责安排受影响验证并提交记录，按下表与对应模型、后端或 connector 维护者确定范围。已由 CI 验证的相同场景不重复手工运行；不要求每次跑完整矩阵，也不因列入此表就新增流水线。

| 场景组 | 维护归属 | 触发与所需范围 | 执行性质 |
| --- | --- | --- | --- |
| GPU P2P：G01–G13、G27–G33、G39 | GPU connector、对应模型及相关 PR 作者 | P2P 分组/收发、角色 TP、graph/DBO 或模型接入变化；选择受影响模式及模型。最小 eager 问题使用 G02，不等大分组使用 G39 | 既有15个唯一CI node ID中的对应项照常；其余受影响变更手工，G02也用于诊断 |
| DeepEP：G14–G18、G34–G38、D01 | DeepEP connector、对应模型及相关 PR 作者 | TP切片、dispatch/combine、payload、专家接入或依赖版本变化；D01按触发范围运行，配合受影响模型的保留服务场景 | 受影响变更手工；D01记录与删除门槛绑定 |
| DSV4 CUDA：G19–G26 | DSV4 GPU 模型及相关 PR 作者 | Hash输入、权重/量化、graph/DBO或运行时升级；选择相关原生/AFD对照及全部适用窗口 | 受影响变更手工；定期资源另落实 |
| NPU V2-Lite：N01–N07、N11 | NPU connector/runner、模型及相关 PR 作者 | CAMP2P、原生ubatching、graph、清理或依赖升级；选择受影响 eager/graph/DBO及拓扑 | 受影响变更手工；N11未资格化前保持待验证 |
| DSV4 NPU：N08–N10、N12、L01/L02 | NPU模型、async/sync connector及相关 PR 作者 | 模型、FFN模式、layered/DSpark、关闭或依赖升级；按实际受影响profile运行，N09/N10仍为限定smoke | 受影响变更手工；N12资格独立记录，不从旧driver通过推导 |
| multi-pod：P01–P09 | 协调/部署、对应connector及相关 PR 作者 | 地址、barrier、失败通知、放置或清理协议变化；覆盖受影响角色分离/交错布局及后端 | 受影响变更手工；六项路径增强另取得设备资格 |
| 其他独立设备入口与主机检查 | 对应工具/探针维护者及相关 PR 作者 | 相关协议、负载、校验器或脚本接口变化；按 C 表用途验证，不混算模型结果 | 受影响检查；已有主机集成可直接复用 |

公共 deployment 或清理协议变化横跨上述组时，按受影响 backend/connector 选择生命周期验证，不将一个后端结果外推到另一个。五卡、八卡或跨 pod 资源未取得时，标记对应项未验证并说明范围，不记作通过。

### 记录与覆盖复核

稳定 README 按场景组维护执行性质、触发、归属和资格记录链接。记录区分“已有定义”“已完成特定版本/硬件资格验证”“本次变更已验证”，包含代码/runtime版本、硬件、命令、规范与生效配置、结果和产物链接；原始日志保存在任务产物。减弱的调试配置、skip或其他平台结果不取得该场景资格。

替代场景的负载/检查改弱、停止执行，或出现被裁剪组合专有回归时，相关 PR 重新审查替代关系，必要时恢复服务场景。精简收益按实际减少的维护、执行或排障成本说明，不以删除行数或用例数代替。

## 依据

- AFD 当前入口、配置和CI选择：[E2E README](../../tests/e2e/README.md)、[ready](../../.buildkite/cuda/test-ready.yml)、[merge](../../.buildkite/cuda/test-merge.yml)、[weekly](../../.buildkite/cuda/test-weekly.yml)。
- [P2P 分组与角色rank](../../afd_plugin/distributed/topology.py)、[实际多peer输入与结果拆分](../../afd_plugin/connectors/gpu/p2p.py)：2A1F和非整比组的价值。
- [GPU ubatching兼容路径](../../afd_plugin/compat/patches/gpu_mrv2_ubatching.py)：G10涉及DP1＋DBO，不能因没有CI选择而视为重复。
- [DeepSeek Attention路由/共享专家](../../afd_plugin/model_executor/models/deepseek_v2.py)与[Qwen3接入](../../afd_plugin/model_executor/models/qwen3_moe.py)：保留每模型代表场景的依据。
- [当前 A5/A3 同步profile](../../tests/e2e/models/deepseek_v4_flash/config.py)与[runner](../../tests/e2e/runner.py)：N09/N10的答案检查和graph证据范围。
- [固定vLLM版本的图模式定义](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/compilation.py#L53)：FULL_DECODE_ONLY的混合模式为NONE；它不等于关闭全部torch.compile或覆盖所有pure-eager行为。
