# Issue #487 后续优化方案

日期：2026-11-02
依据：[issue487-report.md](issue487-report.md)（2026-10-30 双机复现 + 2026-10-31 原型验证）
目标环境：A3 910C 双机，0.30 serving（vLLM ced6857 / vllm-ascend 8fde6d4e3 / AFD 213ee17）

## 1. 范围与判定标准

**只优化 issue #487 中性能劣化严重的路径；跨机劣化不明显的项不做专门优化。**

判定标准：

| 类型 | 阈值 | 处理 |
| --- | --- | --- |
| 跨机/同机比值 | ≥ 1.5× | 优化 |
| GMM 并发导致的增量 | ≥ +30% | 优化 |
| 超额核调度 | 通信 AIV + 2×GMM AIC > 48 | 优化配额 |
| 跨机/同机比值 | < 1.2× | 不优化，仅记录 |
| 单算子小幅收益 | < 5% | 不作为主线 |

按此标准，**不纳入优化**的项：

- CS 48 AIV 跨机/同机 212.3/210.8 = **1.007×**
- DR 跨机/同机 1592.5/1569.5 = **1.015×**
- CS 4 AIV fan-in 1→2 基本平坦（1280.3→1281.7）
- DS 同机 fan-in 1→4 基本平坦（3363.5→3366.4）

## 2. 严重劣化项（优化对象）

| 编号 | 问题 | 证据（µs 或 %） | 性质 |
| --- | --- | --- | --- |
| S1 | **DS 跨机 completion** | 4 AIV 6297.4/3362.4 = **1.87×**；48 AIV 641.4/348.7 = **1.84×** | 跨机 |
| S2 | **DR + 本地 GMM 竞争** | 1569.5→2679.6（**+70.7%**）；跨机 1592.5→2725.6（+71.1%） | GMM 并发 |
| S3 | **高 AIV fan-in 8 共享资源** | 48 AIV：CS 667.7 vs 212.3（**3.1×**）；DS 682.3 vs 348.7（**1.96×**） | 源端共享 |
| S4 | **GMM 超额调度** | 8 通信 AIV + 22/44 = 52 AIV；CS 同机 522.5 vs 20/40 的 276.6（**+89%**） | 配额 |
| S5 | **CS 低 AIV 跨机 completion** | 4 AIV 1280.3/484.0 = **2.65×** | 低 AIV |

补充：S5 在 48 AIV 下已消失（1.007×），因此只有产品必须使用 <8 AIV 时才值得投入；否则以 48 AIV 作为规避手段。

## 3. 优化方案（按优先级）

### P0：DS payload 批量化（对应 S1）

**现状**：DS 每行 payload 立即 `MTE3_S`，1536 次/核；48 AIV 下跨机仍有 1.84×，而 index batch=64 在 48 AIV 基本中性（642.1→640.2），说明剩余瓶颈在 payload 事务数。

**改动**：

1. 每个 destination 在 UB 中暂存 N 个 token 的 int8 row（N=4~8，32–64 KiB）
2. 凑满或结束时一次 `DataCopy` 写出；保持 `moeTokenSendIdx` 递增与 token 去重 flag
3. 只在最后做一次 completion wait；status flag 在全部 payload 与 index 可见后发布
4. index 沿用 batch=64（已实现），与 payload 共用同一批次的完成点

**当前阻塞**：原型 batch=4/2 均触发 bisheng fatbin 生成失败。下一步二选一：

- 用 TQue 暂存替代裸 UB 指针，重试 batch=4
- 把 payload batching 拆成独立 kernel 变体（保留原 kernel 作为 fallback）

**验收**：

- 正确性：issue487 harness 的 DS 检查全部 sender 的 int8、scale=1、专家计数、source 去重
- 性能：4 AIV 跨机 6297→目标 ≤4500 µs；48 AIV 跨机 641→目标 ≤500 µs
- 覆盖 AIV 4/8/48，fan-in 1/8

### P1：DR token-major 泛化 + GMM 并发（对应 S2）

**现状**：token-major 在 4 AIV + GMM 22/44 下同/跨 −24.8%/−26.1%，但依赖“每个 AIV 连续 8 个 expert”；48 AIV 下退化为每 AIV 单 expert，收益消失甚至 +44.7%。

**改动**：

1. 在 48 AIV 下建立 token→4 个 expert slot 的逆表，允许跨 AIV 合并同一 token 的读取
2. 至少做 2–4 buffer 的 MTE2/MTE3 ping-pong，预取下一行、批量写 scale
3. 保留 4 AIV 的连续 expert 快路径作为特化

**验收**：

- 4 AIV + GMM 22/44：同/跨 ≤2050 µs（当前 2015.6/2013.2，保持）
- 48 AIV + GMM：不再出现 +44.7% 回归，目标 ≤22/44 baseline 的 0.85×
- 正确性：source/metadata 无遗漏/重复，scale=1，专家计数正确

### P2：高 AIV fan-in 8 的共享资源（对应 S3）

**现状**：48 AIV 下 fan-in 8 使 CS 3.1×、DS 1.96×；4 AIV 下 fan-in 平坦。说明瓶颈从“单 copy 完成”转为“源端/发送路径/目标入口共享资源”。

**改动/调查**：

1. 记录并固定源/目标 package 分组，对比不同 sender 板卡组合
2. HCCL buffer 512 MiB vs 1 GiB 对照，确认是否窗口反压
3. 评估多 stream/多 kernel 拆分发送路径，或把 8 sender 分批（2×4）
4. 用 detail probe 分离源端 DMA 队列、HCCL 入口、目标端 peer window

**验收**：48 AIV fan8 的 CS/DS 相对 fan1 的比值从 3.1×/1.96× 降到 ≤1.5×

### P3：GMM 配额（对应 S4）

**改动**：

1. 强制 `comm_AIV + 2×GMM_AIC ≤ 48`：8 通信 AIV 时 GMM 用 20/40，禁止 22/44 与 92 AIV
2. 默认配置改为 20/40；只有单算子基线验证过 22/44 的收益时才保留
3. 按算子分别验证：CS/DS 对配额敏感，DR 在 20/40 下反而 +11.7%，不能统一处理

**验收**：8 AIV + 20/40 的 CS 同机 ≤300 µs；DR 不劣于 22/44 的 1.1×；无 core-start spread >10 µs

### P4：CS 低 AIV completion（对应 S5，最低优先级）

**现状**：双缓冲 + blockDim=4 跨机仅 −0.7%；有效杠杆是提高 AIV（48 AIV 已 1.007×）。

**仅在必须使用 <8 AIV 时继续**：

1. 用 event/批量 fence 替代每 chunk 的 `MTE3_S`，把“写 status 前汇总等待”做到 kernel 级
2. 评估多 stream/多 kernel 拆分，把远端 store 完成与后续 chunk 解耦
3. 先用 detail probe 排除 HCCL 窗口反压，再改 kernel

**验收**：4 AIV 跨机从 1280 µs 降到 ≤900 µs；48 AIV 不回归

## 4. Serving 集成与端到端验证

0.30 跨机 serving 已可用（101 A 0-7 + 103 A 4-7 + 103 F 8-15，DBO=1 baseline Mean TTFT 1341 ms / Median 810 ms / P90 3050 / P99 6368，RPS=2，1536 请求）。

后续把 P0/P1 的 vendor 带进 0.30 插件：

1. 用 `build/variants` 的 csrc overlay（与 0.30 插件同为 213ee17）构建 0.30 的 `afd_ascend_C` 与 AFD vendor
2. serving 侧显式设置 `AFD_COMM_AIV`：
   - token-major 生效需要 4 AIV；若 serving 必须 48 AIV，则先完成 P1 的 48 AIV 泛化
3. 同 RPS=2 固定负载对比 baseline：TTFT Mean/P50/P90/P99、吞吐、失败数
4. 通过后再做 RPS=3/4 sweep，观察 DBO 与优化的交互
5. 记录 `AFD_COMM_AIV`、GMM 配额、DBO、chunk/MBT 与结果 RUN_ID

**端到端验收**：

- 1536/1536 成功、0 失败
- P90/P99 TTFT 相对 baseline 改善 ≥10%
- 吞吐不低于 baseline（固定 RPS 下至少持平）
- 至少一个严重项（DS 或 DR+GMM）在 serving 中可观测到收益

## 5. 里程碑

| 阶段 | 内容 | 交付 |
| --- | --- | --- |
| M1 | DS payload batch 编译通过 + 正确性 | DS 4/48 AIV 数据 |
| M2 | DR token-major 48 AIV 泛化 | 48 AIV + GMM 数据 |
| M3 | 0.30 serving 集成 + 端到端 bench | TTFT/吞吐对比 |
| M4 | fan-in 8 / GMM 配额 | S3/S4 结论 |
| M5（可选） | CS 低 AIV event/fence | S5 结论 |

## 6. 边界与风险

- issue #487 的原始环境是 CANN 9.1.0 / driver 25.5.1.1；当前 0.30 镜像是 CANN 9.2 / driver 26.0，CS target helper 在 0.30/CANN9.2 上初始化崩溃，因此算子级复现继续以 0.26/CANN9.1 为准
- 0.30 serving 的跨机拓扑使用 103 的 4-7（0-3 被其他用户占用），与 issue 的 8-15 物理分组不同；记录实际 ordinal
- 共享集群的卡占用会变化；每次测量记录 npu-smi 快照与 `oversubscribed` 标记
- DR token-major 的 4 AIV 快路径依赖连续 expert 布局；任意 routing 下需要逆表，收益可能下降
- DS payload batching 的 bisheng 工具链问题可能仍需独立 kernel 变体绕过
