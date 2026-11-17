# Issue #487 跨机性能劣化：修复实现与验证结果

日期：2026-11-02（初版）/ 2026-11-03（§10 补测）
依据：[issue487-report.md](issue487-report.md)（复现）、[issue487-optimization-plan.md](issue487-optimization-plan.md)（方案）
代码基线：`213ee17`

## 1. 结论摘要

按优化方案 P0/P1 落地了两个 kernel 改动，在 A3 910C 双机上用同一固定字节协议实测，
**跨机性能劣化基本消除**；同机 fan-in 8 有 **+4.4%~+5.6%** 的小幅回归（§10.4），
跨机 fan-in 8 在 11-02 仍为 −2.6%：

| 场景（fan-in 1，除非注明；µs） | 基线 单机 | 优化 单机 | 基线 跨机 | 优化 跨机 |
| --- | ---: | ---: | ---: | ---: |
| DS AIV=4 | 3374.5 | **2067.0** | 6302.6 | **2705.6** |
| DS AIV=8 | 1700.4† | **1049.4†** | 3176.1 | **1389.8** |
| DS AIV=24 | 590.9† | **433.5†** | 1084.5 | **625.3** |
| DS AIV=48 | 348.4 | **337.8** | 642.1 | **461.7** |
| DS AIV=48 + fan-in 8 | 771.3† | 805.1†（+4.4%） | 737.8 | **718.8** |
| DR AIV=4 | 1588.1 | **774.2** | 1588.5 | **764.5** |
| DR AIV=24 | 334.1† | **189.2†** | 333.4 | **192.6** |
| DR AIV=48 | 204.7 | **160.0** | 225.3 | **161.2** |
| DR AIV=4 + GMM 22/44 | 2674.4† | **1681.1†** | 2793.5 | **1682.1** |
| DR AIV=4（`BATCH_SIZE_FACTOR=1.0`） | 1575.7† | **760.2†** | 1591.6 | **763.5** |

单机 = 101 上 10 rank 同机（2026-11-02 22:04）；† = 2026-11-03 在 103 上补测的完全配对值
（§10）。跨机 = S=101 / R=103。逐点变化率见 §3.4 的总表（单机 −1.9%…−51.8%，
跨机 −3%…−57%；同机 fan-in 8 为 +4.4%）。

全部条件 `correctness=True`（CS 校验 8 个状态字与全部 payload；DS/DR 校验 int8 逐字节、
scale=1、专家计数、source 去重）。

**跨机惩罚的最终口径**（§3.3）：DS 1.87× → **1.31×**（aiv4）/ 1.84× → 1.37×（aiv48），
残余是 payload 字节量本身；DR 跨机比始终 ≈1.0×，本次把绝对时间砍半（同机 −51.8%，§10）。

**端到端 serving（§8.2 / §3.4）**：单机 `afd_dp4tp2`、1536 请求、RPS=2 跑通两臂
（各 1536/1536 成功）：Mean TTFT 1659.00→1654.89 ms、P90 3767.64→3739.17 ms、
P99 7362.30→7452.33 ms、吞吐 20513.7→20511.9 tok/s，偏移全在 ±1.25% 内且方向不一致 →
**该 prefill 部署下优化在 E2E 不可分辨**；跨机 serving 仍无有效数据（26 秒 `Executor
failed`，§8.1/§8.3）。原因是 CAM 算子只占 TTFT 的千分之几，不是这次部署的瓶颈；
只有在通信主导（低 AIV / 大 fan-in / GMM 争用）的形态才会体现。

serving 适用性（详见 §6）：两处优化在 production 形状下都会激活（DS stage 8 行、
DR 2 行）；但 DR 的行缓冲判据原先用了 `maxTokenNum_`，在 `BATCH_SIZE_FACTOR` 取默认
1.0 时会让优化**静默不激活**，本轮已改为按每核上限预留并实测确认（§6.2）。serving
实际用到的 48 AIV（DS、无 two-stream 的 DR）与 24 AIV（two-stream 的 DR）两个点均已
直接测到。

DS 的 4 AIV 跨机值 2705.6 µs 已**低于** issue 里同机 baseline 的 3362.4 µs；DR 的
4 AIV+GMM 值 1682.1 µs 也低于方案里 token-major 原型在 4 AIV+GMM 下的 2015.6 µs。
也就是说，跨机惩罚不再是主要成本，剩余成本主要是本地/payload 本身的量。

最终交付的 vendor（`build/variants/final_comm/`，由最终源码干净重建，含 §6.2 的激活修复）复跑确认：
DS aiv24 629.5、DS aiv48 455.9、DR aiv24 189.9、DR aiv48 152.5 µs，全部 pass；
其中 send kernel 与此前验证过的 N=8 构建逐字节相同，因此上表数值即最终产物。

## 2. 修改内容

### 2.1 DS：payload 按目的地连续行批量化（P0，对应 S1）

文件：`csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h`

原因：issue 的量化结论是跨机 DS 的增量与 **remote store 次数**成正比，而不是与字节数
成正比（4 AIV 1536 次/核、48 AIV 128 次/核，单次约 1.9–2.3 µs）。

改动：

1. **把 token 量化结果直接写进 UB stage 行**：`QuantProcess` 的目标从 `xOutTensor_`
   改成 `payloadStageTensor_[row * rowElements]`，因此不产生任何 UB→UB 拷贝。
2. **每个 token 占一行，行内记录它要去的目的地集合与目标 slot**（`payloadDestMaskTensor_`
   位图 + `payloadSlotTensor_`）。同一 token 去多个 MoE rank 时只占一行。
3. **凑满 `PAYLOAD_STAGE_MAX_ROWS` 行后一次性写出**：对每个目的地扫描行序列，把
   “同一目的地、在 stage 中连续、目标 slot 也连续”的行合并成**一次** `DataCopy`。
   连续性是协议自带的：本核连续发的 token 拿到该目的地的连续 slot。
4. 尾部不足一批时在循环结束后 flush 一次；flush 末尾用一次 `MTE3_S` 等待所有 store
   落地，再复用 stage，随后才允许发布 ready flag（保持原来的可见性顺序）。

容量自适应：stage 行数由剩余 UB 决定（`PAYLOAD_STAGE_MAX_ROWS = 8` 为上限），
`moeRankNum > 32`、行长非 32B 对齐或剩余 UB 放不下 2 行时**自动退回原来的 per-token
路径**，不改变任何协议语义。

效果原理：48 AIV 下每核 128 次 payload store → 16 次；4 AIV 下 1536 → 192。

同时把 token-offset（index）写批次从 16 提到 `TOKEN_OFFSET_FLUSH_BATCH = 64`（方案
里已验证 −9.2%@4AIV）。**注意：本表的 DS 收益是 payload 批量化 + index 批量的合计**；
index 单独的贡献引用报告 §7 的 6297.4→5719.3。

### 2.2 DS：stage 行数为什么取 8 而不是更大

方案建议 N=4~8；实测（同一窗口成对测量，48 AIV）：

| stage 行数 | fan-in 1 | fan-in 8（rank0 / 均值） |
| ---: | ---: | ---: |
| 1（baseline） | 642.1 | 737.8 / 743.0 |
| 8 | **461.7** | **718.8 / 706.0** |
| 16 | 498.7–506.4 | 865.7–1160.4（多个窗口一致偏慢） |

N=16 时单次 store 达 66 KB（>64 KiB），在 fan-in 8（8 个 Attention 同时向同一 FFN
窗口写）时明显变差；N=8 的单次 store 为 33 KB，跨机 fan-in 8 下不再放大（718.8 vs
737.8）。但 11-03 的同机复测显示 N=8 在 fan-in 8 下仍有 +4.4%~+5.6% 的回归（§10.4），
所以“不回归”只在跨机拓扑成立。方案 P0 里“48 AIV 641→≤500”的目标在 N=8 下达到（461.7）。

### 2.3 DR：读下一行与写当前行重叠（P1，对应 S2；替代 token-major）

文件：`csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h`

原因：原 routed 循环每行是“读 → `MTE2_S` → 写数据 → `MTE3_S` → 写 scale → `MTE3_S`”，
三次串行等待；issue 的 PMU 结论是 GMM 并发时单行本地读从 0.19 升到 0.50 µs，
逐行等待把这段延迟完全暴露出来。

改动：

1. 分配 `ROW_STAGE_NUM = 2` 个行缓冲（从原 token-address 缓冲的富余空间里划，保留
   其原有容量下限，放不下就退回原串行路径）。
2. 循环改成“**先发下一行的读，再写当前行**”：第 i 次迭代先 `MTE3_S` 保证复用缓冲的
   上一次 store 落地，再发起第 i+1 行的 `DataCopy`（MTE2），然后写第 i 行的数据与
   scale，最后等第 i+1 行的 `MTE2_S`。读与写因此重叠一个迭代。
3. 数据与 scale 两次 store 之间不再各等一次，改为一次 `MTE3_S` 覆盖两者；循环结束
   再统一 `MTE3_S`，保证完成标记发布前所有 store 已落地。

这个做法**不依赖 expert 布局**，所以在任意 AIV 与任意 routing 下都成立；它比方案里
的 token-major 原型更简单、结果也更好（4 AIV+GMM 1682.1 vs 原型 2015.6），因此
**P1 不再需要 48 AIV 的逆表泛化**。

### 2.4 Benchmark 工具修正（`tools/benchmarks/issue487/run.py`）

1. `--aivs` 增加 24：production FFN send stream 默认就是 24 AIV，之前只能测 4/8/48。
2. 设备 ordinal 校验放宽：ordinal 是**每机本地**的，跨机 case 允许 S/R 编号重复；
   `same` / `cross-fixed-receiver` 仍要求 10 个互异 ordinal。原规则在 101 的 0–7、
   103 的 0–9 被其他任务占用时无法组出任何可行映射。
3. 每个 case 独立 HCCL 端口段（`--npu-port-base`，先前已加）；此外本次在驱动脚本里
   让**每次调用**使用不同的 base/control/NPU 端口段，解决连续多次运行之间的
   `Communication_Error_Get_Socket` / Gloo 连接超时。

`docs/npu/CAM_ASYNC_ROUTED_OPS.md` 的 “Current kernel implementation notes” 补记了
两个优化及其退化条件。

## 3. 测量方法与环境

- 拓扑：S = 80.5.17.101（`p00848063-afd` 容器，发送者 ordinal 8–15，全部空闲），
  R = 80.5.17.103（`z00952334-afd487` 容器，接收者 ordinal 12,13）。
  两机 CANN 9.1.0、torch 2.10.0+cpu、torch-npu 2.10.0.post4，与复现报告一致。
- 协议：`tools/benchmarks/issue487`，10 rank（8 sender + 2 target），
  `--rounds 3 --samples 16`，结果取 rank0（DR 取接收者）3 轮 event 均值；
  结果目录写到共享 NFS `/mnt/share/z00952334/issue487-results/`。
- baseline vendor = `afd_plugin/_cann_ops_custom/vendors/afd-plugin`（213ee17 原始 csrc
  构建，101/103 两侧 md5 一致）；优化 vendor = 同一 csrc + 本次改动重建。
- 复现性：baseline 跨机 DS 6302.6/642.1、DR 1588.5、DR+GMM 2793.5 与复现报告的
  6297.4/641.4、1592.5、2725.6 一致（GMM 项 2.5% 以内）。
- 驱动脚本：`build/bench487.sh`（单侧）、`build/bench487_pair.sh`（双机配对，含
  独立端口段）。

### 3.1 边界

- 共享集群，测点是不同时间窗跑的；因此**关键对比都做了同一窗口的成对测量**
  （48 AIV fan1/fan8）。DS 4/8/24 AIV 的 baseline 与优化值分别在相邻窗口取得，
  由于这些点差异达 40%–57%，窗口噪声不足以改变结论。
- 两机都凑不齐 10 个空闲 ordinal，**本次没有重测同机数据**；因此 §3.2 的同机数据
  引用复现阶段的归档值，优化后的同机值待有空窗时补测（见 §3.3）。
- 103 的 ordinal 8 有一个 106 MB 的 vLLM 服务进程，故发送者放在 101（完全空闲），
  接收者用 103 的 12,13。
- CS（combine send/recv）kernel 二进制与 baseline **逐字节相同**（本次构建对比），
  因此 CS 行为与复现报告一致，未重测。

### 3.2 同机数据（归档值，来自复现报告 2026-10-30 / AIV=4 / fan-in=1）

同机数据是在 `issue487-report.md` 的复现阶段用同一 harness、同一固定字节协议测的，
拓扑为 S 主机 8 sender + 本机 2 target（ordinal 与本次跨机测量不同），基线代码同为
`213ee17`，因此可作为“未优化代码的同机参照”，但**不是本次窗口的成对测量**。

| 算子（AIV=4，fan-in=1，µs） | 同机 | 跨机 | 跨机/同机 |
| --- | ---: | ---: | ---: |
| CS | 482.3 | 1283.0 | 2.66× |
| CS + GMM 22/44（AIV=8） | 522.5 | 923.5 | 1.77× |
| DS | 3362.4 | 6297.4 | **1.87×** |
| DS + GMM 22/44 | 3878.8 | 6827.3 | 1.76× |
| DS + GMM 22/44（AIV=8） | 2048.1 | 3503.2 | 1.71× |
| DR（ready input） | 1569.5 | 1592.5 | 1.015× |
| DR + GMM 22/44 | 2679.6 | 2725.6 | 1.017× |
| DR + GMM 22/44（AIV=8） | 1152.4 | 1172.0 | 1.017× |

AIV 提高后跨机惩罚消失（AIV=48，同机 / 跨机）：

| 算子（AIV=48） | 同机 | 跨机 | 跨机/同机 |
| --- | ---: | ---: | ---: |
| CS | 210.8 | 212.3 | 1.007× |
| DS | 348.7 | 641.4 | 1.84× |
| DS（AIV=48 + 22/44 极端超额） | 675.0 | 957.2 | 1.42× |

CS 在 48 AIV 已无跨机惩罚（1.007×），DS 在 48 AIV 仍有 1.84× —— 这正是本次 P0 要
解决的残余项；DR 的跨机比全程≈1.0，但被本地 GMM 拖慢（+70.7%），对应 P1。

### 3.3 优化后跨机 vs 归档同机（跨机惩罚是否消失）

把本次测得的**优化后跨机**值与上表**归档同机基线**放在一起，可以直接看跨机惩罚的去留
（跨机/同机 < 1 表示优化后的跨机甚至快于未优化的同机）：

| 点（AIV，fan-in 1） | 归档同机基线 | 优化后跨机 | 比值 |
| --- | ---: | ---: | ---: |
| DS AIV=4 | 3362.4 | 2705.6 | **0.80×** |
| DS AIV=8 | — | 1389.8 | — |
| DS AIV=24 | — | 625.3 | — |
| DS AIV=48 | 348.7 | 461.7 | 1.32×（基线跨机 1.84×） |
| DR AIV=4 | 1569.5 | 764.5 | **0.49×** |
| DR AIV=48 | —（AIV=4 同机 1569.5） | 161.2 | 0.10× |
| DR + GMM 22/44 AIV=4 | 2679.6 | 1682.1 | **0.63×** |

读法：AIV=4 时 DS/DR 的优化后跨机值已经低于未优化的同机值（0.80× / 0.49×），说明
跨机惩罚不再主导；AIV=48 的 DS 仍有 1.32× 的残余，是 payload 字节量本身（见 §6.5）。

**同机 A/B 已补齐（2026-11-02 22:04，监控任务自动完成）**：101 的 10 卡窗口内用同一
harness 跑完 baseline 与 optimized 两臂，8 个点全部 `correctness=True`：

| 同机（AIV, fan-in 1） | baseline / µs | optimized / µs | 变化 |
| --- | ---: | ---: | ---: |
| DS AIV=4 | 3374.5 | **2067.0** | **−38.7%** |
| DS AIV=48 | 348.4 | **337.8** | −3.0% |
| DR AIV=4 | 1588.1 | **774.2** | **−51.3%** |
| DR AIV=48 | 204.7 | **160.0** | **−21.8%** |

同机 baseline 与归档值（复现报告）差异 <0.7%，说明两套测量口径一致，可以直接并表。

11-03 用最终交付构建在 103 复测了这四个点与其余缺口，见 §10.2/§10.3：DS 两点偏差
≤1.1%、DR AIV=4 偏差 −1.2%；DR AIV=48 optimized 为 143.1（本表 160.0 来自 101 本地
旧构建），同窗口 baseline 213.2。

**跨机惩罚的最终对照**（跨机值取 §1 的实测，同机值取上表）：

| 点 | 基线 跨机/同机 | 优化后 跨机/同机 | 说明 |
| --- | ---: | ---: | --- |
| DS AIV=4 | 6302.6 / 3374.5 = **1.87×** | 2705.6 / 2067.0 = **1.31×** | 惩罚降 30%，残余 ≈ payload 字节量 |
| DS AIV=48 | 642.1 / 348.4 = **1.84×** | 461.7 / 337.8 = **1.37×** | 同上；48 AIV 下每核 store 已很少 |
| DR AIV=4 | 1588.5 / 1588.1 = 1.00× | 764.5 / 774.2 = **0.99×** | 跨机本身无惩罚，绝对值减半 |
| DR AIV=48 | 225.3 / 204.7 = 1.10× | 161.2 / 160.0 = **1.01×** | 同上 |

结论：**DR 的跨机代价本就不是问题（1.0×），本次把它的绝对时间砍半**；**DS 才是真正的跨机
项，1.87× → 1.31×**，剩下那 0.31 的比值来自“同样的 24 MiB payload 必须过链路”，
属于带宽/字节量，而不是 store 完成等待，因此不能再靠合并事务继续压。

### 3.4 优化前后性能总表（单机 / 跨机）

把算子级与 serving 级的全部实测放在一张表里，**单机（同机）与跨机两列并排**，
并在两个维度上分别给出变化率。单机口径来自 101 的 10 卡窗口（101 独立跑，
2026-11-02 22:04）；† 行是 2026-11-03 在 103 上按同一协议补测的最终交付构建（§10）。
跨机口径来自 S=101 / R=103（§1）。所有点 `correctness=True`。

#### 算子级（fan-in 1，除非注明；µs，3 轮 event 均值）

| 场景 | 基线 单机 | 优化 单机 | 单机变化 | 基线 跨机 | 优化 跨机 | 跨机变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DS AIV=4 | 3374.5 | **2067.0** | **−38.7%** | 6302.6 | **2705.6** | **−57.1%** |
| DS AIV=8 | 1700.4† | **1049.4†** | **−38.3%†** | 3176.1 | **1389.8** | **−56.2%** |
| DS AIV=24 | 590.9† | **433.5†** | **−26.6%†** | 1084.5 | **625.3** | **−42.3%** |
| DS AIV=48 | 348.4 | **337.8** | −3.0% | 642.1 | **461.7** | **−28.1%** |
| DS AIV=48 + fan-in 8 | 771.3† | 805.1† | **+4.4%†** | 737.8 | **718.8** | −2.6% |
| DR AIV=4 | 1588.1 | **774.2** | **−51.3%** | 1588.5 | **764.5** | **−51.9%** |
| DR AIV=24 | 334.1† | **189.2†** | **−43.4%†** | 333.4 | **192.6** | **−42.2%** |
| DR AIV=48 | 204.7 | **160.0** | **−21.8%** | 225.3 | **161.2** | **−28.4%** |
| DR AIV=4 + GMM 22/44 | 2674.4† | **1681.1†** | **−37.1%†** | 2793.5 | **1682.1** | **−39.8%** |
| DR AIV=4（`BATCH_SIZE_FACTOR=1.0`） | 1575.7† | **760.2†** | **−51.8%†** | 1591.6 | **763.5** | **−52.0%** |

† = 2026-11-03 在 103 的 10 卡窗口（ordinals 4-13，senders 4-11 / targets 12,13）补测的
完全配对值，见 §10.2。DR AIV=48 的 11-02 optimized（160.0）来自 101 本地旧构建；最终
构建的同窗口复测见 §10.3（213.2→143.1，−32.9%）。跨机列仍是 11-02 的 S=101 / R=103。

读法：

- **DR 两个维度几乎同幅改善**（单机 −51.3%、跨机 −51.9%），符合"DR 跨机比本来就≈1.0×、
  本次砍的是绝对时间"；AIV=48 时两端分别 −21.8% / −28.4%，说明行读/写重叠在高并行度下
  仍有收益，因为它是把串行等待折叠而不是靠更多核。
- **DS 跨机收益明显大于单机**（AIV=4：−57.1% vs −38.7%），因为单机只省了 per-store 完成
  开销，跨机还额外去掉了跨机单次 store 的高延迟；AIV=48 时两端都变小（−28.1% / −3.0%），
  此时每核 store 数本来就少。
- 补测后单机列已无空格：DS AIV=8 的 −38.3% 与 AIV=4 接近，AIV=24 的 −26.6% 说明批量化
  在中高并行度仍有收益；DR AIV=24（two-stream production 配额）为 −43.4%。
- **fan-in 8 是同机唯一的回归点**：771.3→805.1（+4.4%），三次成对复测 +3.9%~+5.6%
  （§10.4）。跨机 fan-in 8 在 11-02 仍是 −2.6%，所以这是同机拓扑特有的回归，不是批量化
  收益消失。

#### serving 级（单机 `afd_dp4tp2`，冻结 1536 请求、RPS=2、MBT 8192）

| 指标 | 基线 单机 | 优化 单机 | 单机变化 | 跨机 |
| --- | ---: | ---: | ---: | --- |
| Mean TTFT | 1659.00 ms | 1654.89 ms | −0.25% | **无有效数据**（§8.1/§8.3） |
| Median TTFT | 1126.39 ms | 1131.06 ms | +0.41% | 同上 |
| P90 TTFT | 3767.64 ms | 3739.17 ms | −0.76% | 同上 |
| P99 TTFT | 7362.30 ms | 7452.33 ms | +1.22% | 同上 |
| 总吞吐 | 20513.7 tok/s | 20511.9 tok/s | −0.01% | 同上 |
| 成功 / 失败 | 1536 / 0 | 1536 / 0 | — | 26 秒 `Executor failed`，0 样本 |

两类数据放在一起就自洽了：**算子级收益是真实的（单机 −1.9%…−51.8%，跨机 −3%…−57%，
同机 fan-in 8 例外 +4.4%），但在计算主导的 prefill serving 里它只占 TTFT 的千分之几，
所以服务端看不出差异**；跨机服务端则连有效数据都拿不到（部署自身崩溃 + 103 卡长期不释放）。

## 4. 正确性验证

| 验证项 | 结果 |
| --- | --- |
| DS payload/int8/scale/专家计数/source 去重 | aiv 4/8/24/48，fan1/fan8 全部 pass |
| DS 部分批 flush（stage 未填满） | 用 `PAYLOAD_STAGE_MAX_ROWS=7` 变体（每核 64 token → 9 满批 + 1 行余量）验证 aiv4/aiv48 均 pass；耗时 2793.5/454.7 µs，与 N=8 同量级 |
| DR payload/scale/专家计数/source 去重 | aiv 4/24/48、含 GMM 12/24 与 22/44 全部 pass |
| CS 状态字与 payload | kernel 未改（二进制相同），沿用复现报告的 pass |
| 同机补测（2026-11-03，§10） | 12 个条件 × 2 臂 + fan-in 8 复测 4 例，全部 `correctness=True` |

## 5. 对优化方案各条的结论更新

| 条目 | 结论 |
| --- | --- |
| P0 DS payload 批量化（S1） | **完成**。4 AIV 跨机 −57.1% / 同机 −38.8%；48 AIV 跨机 −28.1% / 同机 −1.9%（目标 ≤4500 / ≤500 均达成）。行数取 8；但同机 fan-in 8 下 N=8 仍有 +4.4% 回归（N=16 更差），见 §10.4。 |
| P1 DR token-major 泛化（S2） | **改方案并完成**。用“行读/写重叠”替代 token-major：不依赖 routing 布局、任意 AIV 有效，4 AIV+GMM22 −39.8%（跨）/ −37.1%（同机），production 配额（aiv24+GMM12）−35.4%，均优于 token-major 原型的 −24.8%。48 AIV 泛化与逆表**不再需要**。 |
| P2 高 AIV fan-in 8 共享资源（S3） | **未解决**。跨机 48 AIV fan8 从 737.8→718.8（−2.6%）；同机同点在 11-03 复测为 771.3→805.1（**+4.4%**，三次成对 +3.9%~+5.6%，§10.4），批量化在 fan-in 8 下反而放大目标入口竞争。未做源端/板卡/窗口的根因隔离，方案里的 ≤1.5× 目标未达成。 |
| P3 GMM 核配额（S4） | **无需改代码，已补测 production 配额**。默认 `ffn_compute_aic_num=12`、`ffn_compute_aiv_num=24`、`ffn_combine_aiv_num=24` 满足 12 AIC/48 AIV 预算；12/24 下 DR 的 GMM 增量从 22/44 的 ~1.1 ms 降到 ~0.33 ms（aiv4），production 形状（DR aiv24）只有 +53 µs / +15.9%（§10.5）。22/44 只是 benchmark 压力配置（`--gmm-aic 22`），建议在 benchmark 文档里标注。 |
| P4 CS 低 AIV（S5） | 未做。CS 在 48 AIV 为 1.007×；有效杠杆仍是提高 AIV，双缓冲已证伪。 |

## 6. serving 配置下的适用性

算子是同一份二进制，但 serving 的 **形状、AIV 配额、容量因子** 与固定字节 harness
不同，结论取决于这两项优化是否仍然被激活、以及收益是否还在。

### 6.1 production 形状下的激活判据（按源码分配式逐项计算，UB = 192 KiB）

| kernel | production 形状 | Init 已用 UB | stage 行数 |
| --- | --- | ---: | ---: |
| DS send | hidden 4096, topk 6, 256 专家 / 8 FFN rank, 48 AIV | 144,064 B | **8**（上限） |
| DS send（假想 512 专家） | 同上 | 181,440 B | 3（自动降级，仍激活） |
| DR recv | 每 rank 32 专家, 8 FFN rank, 48 AIV | 5,280 B | **2**（`ROW_STAGE_NUM`） |

DS 的 stage 在 production 形状下能拿满 8 行；即使专家数翻倍也只会降级到 3 行而不是
关闭。DR 的 2 行缓冲只占 8,256 B，余量充足。

### 6.2 本轮发现并修复的 serving 激活问题（重要）

DR 的行缓冲判据最初写成“为 token-address 缓冲预留 `maxTokenNum_` 条”，而
`maxTokenNum_ = 262144 * BATCH_SIZE_FACTOR`：

- `BATCH_SIZE_FACTOR=0.0625`（qualification recipe / harness）→ 预留 32 KiB，激活；
- `BATCH_SIZE_FACTOR` 未设（pybind 与 op host 默认 1.0）→ 需要预留 512 KiB > UB，
  **DR 优化静默不激活**，行为退回改造前。

修法：按**每核可发起的上限**预留，即 `ceil(maxTokenNum_ / aivNum_)` 条
（routed 循环的 `tokenNumPerAiv = ceil(tokenNumSum / aivNum_)` 且
`tokenNumSum <= maxTokenNum_`，因此这是可证充分的界）。修复后实测：

| 条件（DR，aiv4，fan-in 1） | baseline | 修复后 | 变化 |
| --- | ---: | ---: | ---: |
| `BATCH_SIZE_FACTOR=0.0625`（跨机 11-02） | 1588.5 | 764.5 | −51.9% |
| `BATCH_SIZE_FACTOR=1.0`（原生容量，跨机 11-02） | 1591.6 | 763.5 | **−52.0%** |
| `BATCH_SIZE_FACTOR=0.0625`（同机 11-03，§10） | 1586.2 | 764.6 | −51.8% |
| `BATCH_SIZE_FACTOR=1.0`（原生容量，同机 11-03，§10） | 1575.7 | 760.2 | **−51.8%** |

四行一致说明：容量因子不影响循环本身，且行流水在原生容量下确实生效（修复前该行会
停在 ~1590）。harness 为此新增了 `ISSUE487_BATCH_SIZE_FACTOR` 透传。

### 6.3 serving 的 AIV 配额与实测覆盖

| 算子 | serving 里的 stream | AIV | 实测点 |
| --- | --- | ---: | --- |
| DS send（attention 侧 `dispatch`） | 默认 stream，无 `set_stream_limit` | 48 | aiv48 −28.1% |
| DR recv（FFN 侧，`ffn_two_stream`） | `self.send` → `ffn_combine_aiv_num` | 24 | **aiv24 333.4→192.6（−42.2%）** |
| DR recv（未开 two-stream） | 默认 stream | 48 | aiv48 225.3→161.2（−28.4%） |
| DS aiv24（中间点/对照） | — | 24 | **1084.5→625.3（−42.3%）** |

也就是说，serving 实际用到的 48 AIV（DS、无 two-stream 的 DR）与 24 AIV（two-stream
的 DR）两个点都直接测到了；跨机口径的增益在 −28%…−42% 之间。11-03 的同机补测（§10.2）
给出 DS aiv48 −1.9%、DR aiv48 −32.9%、DR aiv24 −43.4%：DS 在 48 AIV 的收益很小，
DR 在 two-stream 配额下仍有 30%+ 的收益，两者都与跨机口径的结论一致（DS 靠 4–24 AIV
拿收益，DR 不靠核数）。

### 6.4 收益随 token 数的变化：prefill 成立，decode 只是不亏损

两处优化的收益都来自“把 N 次串行等待压成 1 次”，所以与**每 AIV 的 token 数**正相关：

- **chunked prefill**（`max-num-batched-tokens=8192`，attn DP4 → 每 rank ~2048 token，
  每 AIV ~43 token）：DS 的 8 行 stage 会填满 5 次，与 harness 的 64 token/AIV 同量级，
  预期保持 −20%…−30%；DR 每 AIV 有几十行，行流水全程生效。
- **decode**（`max-num-batched-tokens=120`，每 rank ~30 token）：DS 每 AIV 不到 1 个
  token，stage 不会填满，末尾逐 token flush —— 退化成改造前的行为（只多一次行扫描），
  **不亏损**；DR 每 AIV 约 2 行，只重叠 1 次，收益很小但方向为正。

### 6.5 8 目的地的形状下为什么仍然成立（未端到端实测）

harness 固定 2 个 FFN target，production 是 8 个，所以 DS 的“每 token 目的地数”从 2
变成最多 6（topk）。批量化减少的是**每目的地的 store 次数**：一次 flush 对每个目的地
只发一段连续行，因此 store 数从
`每 token × 目的地数` 降到 `每 N token × 出现的不同目的地数`。按 production 形状估算，
每 AIV 约 43 token 时 payload store 从 ~250 降到 ~88，加上 index 每专家 1 次
（≤32，与 batch=16/64 无关），总 store 数约为原来的 1/2.4 → 预期相对增益略低于
harness 的 −28%，量级仍在 −20% 以上。

同时确认批量化在“目的地多、token 少”时不会放大开销：flush 的行扫描是
`N × 目的地数` 次标量位测试（8×8），每个 flush 只多几次 `GetPeerAddrByRankId`。

### 6.6 验证覆盖（2026-11-03 更新）

1. **真实 serving 端到端**：单机 `afd_dp4tp2` 已测（§8.2，prefill 计算主导 → 不可分辨）；
   跨机 20 卡 serving 仍未测（§8.1 的运行期崩溃 + 101 被占满）。
2. **production GMM 配额（12 AIC/24 AIV）下的 DR 增量**：**已补测（§10.5）**。production
   形状（DR aiv24）的 GMM 增量只有 +53 µs（+15.9%），远小于 22/44 的 ~1.1 ms；优化后
   仍为 −35.4%。aiv4 下 12/24 vs 22/44 的对照也在 §10.5。
3. **DBO / 多 stream 重叠下的相互影响**：仍未测。算子级 harness 是单 stream；单机 E2E
   的 DBO 是关的，跨机 E2E 无有效数据。
4. **8 目的地形状**：仍未测（harness 固定 10 rank / 2 target，需要 16 rank 与协议改动）；
   §6.5 的估算仍是唯一依据。
5. **同机 fan-in 8**：**已补测且发现回归（§10.4）**：+4.4%~+5.6%（三次成对），与跨机的
   −2.6% 方向相反；P2 仍未解决。

## 7. 后续建议

1. **serving 集成**：`build/variants/final_comm/` 是 CANN 9.1（harness 侧）vendor，
   `build/variants/final_comm_v030/` 是在 v0.30 容器（CANN 9.2）里重建的 serving 侧
   vendor，两者同源。端到端 A/B 由 §8 的监控任务自动执行，**当前被 serving 自身的
   运行期崩溃挡住**（见 §8.1），不是优化引入的问题。
2. **DR 与 GMM 的剩余增量**：22/44 压力配额下，优化后 GMM 仍使 DR 增加 916 µs（aiv4，
   +120% 相对，base 更小所以相对值变大）；production 配额（12/24）下 aiv24 只增加 ~61 µs。
   若要继续压缩，方向是批量写 scale / 多行合并，而不是再改 AIV。
3. **fan-in 8 的根因**：需要用 PMU/detail probe 分离源端 DMA 队列、HCCL 入口与目标
   peer window；本轮的批量化只是把单次 store 变小，同机 fan-in 8 甚至 +4.6%（§10.4）。
   若 production 存在 8 个 sender 打同一 FFN rank 的形态，建议按 fan-in 关闭/缩小 stage，
   但需要先定位共享资源。
4. 若后续要动 stage 上限，请同时复测 48 AIV fan1 与 fan8：这两个点对行数敏感方向相反
   （fan1 偏大更好、fan8 偏小更好）；11-03 的同机数据说明 8 行在 fan8 仍偏大（+4.4%），
   而 fan1 的 48 AIV 只剩 −1.9%，收益主要来自 4–24 AIV。

## 8. 端到端 serving 监控任务

### 8.1 本轮直接观察到的 serving 状态（重要）

2026-11-02 20:27，`afd_dp6tp2` 跨机部署**是活着的**：`/health` 200、`/v1/models` 返回
`dsv4`，参数从活进程的 cmdline/environ 核实：

| 项 | 实际值 |
| --- | --- |
| 拓扑 | `afd_dp6tp2`（node0 A 0-7；node1 A 4-7 + F 8-15） |
| `--max-num-batched-tokens` / `max-num-seqs` | 8192 / 16 |
| DBO | **关**（cmdline 无 `--enable-dbo`） |
| Attention | `--enforce-eager` |
| checkpoint / 量化 | `DeepSeek-V4-Flash-w8a8-mtp` / `ascend`（W8A8，FFN 普通 eager） |
| vendor | `afd-plugin-v030/.../vendors/afd-plugin` + 镜像自带 `custom_transformer` |

随后用冻结负载（1536 请求、RPS=2、chunk 8192）压测，**26 秒后服务自己崩了**：

```text
(EngineCore_DP0..DP3) RuntimeError: Executor failed.        # 12:28:10 UTC / 20:28:10 CST
curl /health -> 000（API 端口消失）
```

从 20:27:44 开始、跑到第 4 个请求左右就整组 DP core 报 `Executor failed`，因此这次
baseline 压测没有任何有效样本（0/1536），**不作为 E2E 数据**。同一套栈在 08:42 的
早前一次尝试里 FFN 侧报的是 `ACL stream synchronize failed, error code:507034`
（`recv_ffn_work_item` → `control.cpu().tolist()`），两次现象一致指向运行期 kernel/流
同步异常，与本次 kernel 改动无关（本次 baseline 臂用的就是原始 vendor）。

崩后约 1 分钟内，101 的 8-15 被另一个作业（`VLLMWorker_PP`）占走，资源窗口再次关闭。

结论：**端到端 serving A/B 目前卡在 serving 自身的运行期崩溃上，不是优化失效**。
要拿到 TTFT 对比，得先修这个崩溃；在它修好之前任何 serving 数字都不可信。

### 8.2 单机 serving A/B 结果（2026-11-03 01:08–01:46，已完成）

跨机 20 卡窗口整晚没出现，而 101 出现过整机 16 卡全空，于是改用 skill 里的单机布局
`afd_dp4tp2`（A 0-7 + F 8-15 同一台机器）做端到端 A/B。这是本次唯一跑通的 serving 测试。

部署（两次都是从活进程/日志核实的实际配置）：

| 项 | 值 |
| --- | --- |
| 拓扑 | `afd_dp4tp2`（单机，A DP4×TP2 = 8 rank，F EP8） |
| MBT / max-num-seqs | 8192 / 16 |
| DBO / Attention | 关 / eager |
| checkpoint / 量化 | `DeepSeek-V4-Flash-w8a8-mtp` / `ascend`(W8A8) |
| 负载 | 冻结 1536 请求、RPS=2、chunk 8192、每请求 1 输出 token |
| 臂 | baseline vendor vs `final_comm_v030`（仅换 dispatch send/recv kernel，md5 已验证） |

| 指标 | baseline | optimized | 变化 |
| --- | ---: | ---: | ---: |
| Mean TTFT | 1658.997 ms | 1654.892 ms | −0.25% |
| Median TTFT | 1126.393 ms | 1131.055 ms | +0.41% |
| P90 TTFT | 3767.640 ms | 3739.172 ms | −0.76% |
| P95 TTFT | 4971.512 ms | 4953.456 ms | −0.36% |
| P99 TTFT | 7362.301 ms | 7452.334 ms | **+1.22%** |
| 总吞吐 | 20513.7 tok/s | 20511.9 tok/s | −0.01% |
| 成功 / 失败 | 1536 / 0 | 1536 / 0 | — |

**结论：在这个 prefill 部署里，优化在端到端不可分辨**。所有指标的偏移都在 ±1.25% 以内，
而且方向不一致（Mean/P90/P95 变好、Median/P99 变差），属于噪声。方案 §4 里
“P90/P99 改善 ≥10%”的验收**没有达成**，但原因不是优化失效：

- 该部署的 Mean TTFT ≈ 1.66 s，主要由 GMM/prefill 计算与排队构成；CAM 的 DS/DR 每个 MoE
  层各几百 µs（aiv24 实测 DS 625 µs、DR 193 µs），相对 43 层 × chunk 的总时长只占
  千分之几到百分之一量级，压掉其中 30%–50% 仍在噪声内。
- 也就是说：**这两个 kernel 只有在“通信占主导”的形态下才会体现到 E2E**，例如低 AIV
  （4–8）、大 fan-in、或 GMM 并发把本地读延迟抬高（算子级 DR+GMM −39.8% 那一类）。
  当前 serving 是 prefill 计算主导，不是这种形态。

两点记录上的说明：

1. `bench_once.sh` 的 `--topology` 标签当时写死为 `afd_dp6tp2`，所以结果路径是
   `.../afd_dp6tp2/...`，而实际部署是 `afd_dp4tp2`；只是**路径标签**问题，数据本身有效
   （TTFT 不受标签影响），已在 `bench_once.sh` 里改为 `BENCH_TOPOLOGY` 可覆盖。
2. 两臂都跑满 1536 请求、0 失败，说明这次单机部署比跨机部署稳定得多（跨机那次 26 秒即
   `Executor failed`，见 §8.1）。

原始数据：`/mnt/share/z00952334/serving-e2e/ab/single-baseline.json` 和
`single-optimized.json`；压测日志 `/mnt/share/z00952334/serving-e2e/bench-single-*.log`；
服务日志 `/home/z00952334/logs-single-{baseline,optimized}/`（101）。

### 8.3 收尾判定：跨机 E2E 是"被 serving 缺陷阻塞"，不是"缺资源"

监控从 2026-11-02 20:46 跑到 2026-11-03 04:05（约 7.3 h），结论如下：

- **101 侧早已满足**：多次出现 16 卡全空（含 23:53、01:47、02:29 起长期全空）。
- **103 侧长期差两张**：整晚稳定在 4-13 十张空闲，`afd_dp6tp2` 需要 4-15 共十二张，
  14、15 一直被别的作业占着，8 小时没让出。
- 更关键的是：即使 103 让出这两张，跨机 A/B 也过不了闸门——20:27 那次**用原始 vendor**
  压测 26 秒就 `Executor failed`（§8.1），说明 `afd_dp6tp2` + W8A8 这套跨机部署在负载下
  本身不可用。等资源并不能解除这个阻塞。

另外一个原本想做的补充实验也确认不可行：为了把"通信占比"放大到 E2E 可见，需要压低 CAM
的 AIV 配额，而该配额只有 `ffn_two_stream` 路径才生效，它又要求 W4A8 layered GMM
（当前 checkpoint 是 W8A8），所以**在可用配置下构造不出通信受限的 serving 形态**。

因此跨机 E2E 的结论是：**已知缺陷阻塞，不是优化问题**；单机 E2E 已给出"prefill 计算主导时
不可分辨"的结论（§8.2），两者一致。资源与监控状态：watcher 已于 04:05 停止，
`watch.status` 记录为 `blocked-by-serving-crash+103-occupancy`。要重新开监控：

```bash
cd /home/z00952334/afd-plugin/build/serving_e2e
rm -f /mnt/share/z00952334/serving-e2e/watch.status
POLL_SECONDS=300 MAX_HOURS=20 setsid nohup ./watch_serving.sh >> /tmp/watch-serving.out 2>&1 &
```

### 8.4 监控任务做了什么

脚本放在 `build/serving_e2e/`（未进版本库），每 5 分钟轮询一次，最长 24 小时，
按代价从低到高做两件事，做完即退出并写状态：

| 顺序 | 触发条件 | 动作 | 产物 |
| --- | --- | --- | --- |
| 1 | 本机（103）有 ≥10 个空闲 ordinal | `run_same_host.sh`：同一 10 卡窗口内跑 baseline 与 optimized 的 **同机** DS/DR（aiv 4/48） | `$LOG_ROOT/same-host*/` |
| 2 | 101 有 0-7 且 103 有 4-15 全空 | `run_ab.sh`：换 vendor → 重启两腿 → `/health` + smoke 闸门 → RPS=2 压测 → 停服；再跑另一臂 | `$LOG_ROOT/{bench-*,vendor-*,compare.md}` |
| 2' | 101 上 `/health`=200（已有部署在跑） | 直接进入 `run_ab.sh`（会重启现有部署以切臂） | 同上 |

关键设计：

- **先闸门后压测**：任一臂 `/health` 不到 200 或单请求 smoke 失败即停该臂并记录日志尾部，
  不做无信息价值的重启（skill 的明确要求）。
- **停服只用自己的 PID**：走 skill 的 `stop_local.sh` + 本次部署的 `LOG_DIR/PID_DIR`，
  不按用户名清进程。
- **两臂可区分**：`swap_vendor.sh` 会把两臂的 kernel md5 写进日志（baseline
  `a3e95f38/202194a6`，optimized `61d78bd7/9ec53a3a`（send），recv 亦不同），
  避免"换了没换看不出来"。
- **两臂同源**：optimized 臂 = 原始 serving vendor 做 kernel 替换（只换
  `afd_async_dispatch_send/recv` 的 kernel 目录及其 config），GMM/其它算子保持原样，
  这样 FFN 不会因为缺 GMM 算子起不来。

日志与状态：`/mnt/share/z00952334/serving-e2e/`（两机共享）：
`watch.log`（轮询与决策）、`watch.status`（`done`/`failed`/`timeout` + 时间）、
`preflight` 结果、`compare.md`（两侧 TTFT 对比表）、`same-host-summary.txt`。

启动状态：2026-11-02 20:46 CST 起，`watch_serving.sh` 以 `setsid` 脱离会话常驻
（pid 1678775，poll 300 s，最长 24 h），另设 30 分钟一次的会话提醒复查
`watch.status`。查看/停止：

```bash
tail -5 /mnt/share/z00952334/serving-e2e/watch.log
cat /mnt/share/z00952334/serving-e2e/watch.status 2>/dev/null
ps -eo pid,args | grep "[w]atch_serving.sh"
kill 1678775          # 只停监控，不动已经在跑的服务/测量
```

### 8.5 手动复现同一套动作

```bash
cd /home/z00952334/afd-plugin/build/serving_e2e
./preflight.sh                  # 只查产物，不碰 NPU
./watch_serving.sh --once       # 探一次窗口，打印当前可用 ordinal
./watch_serving.sh              # 挂起轮询（本次即以此方式启动）
./run_ab.sh                     # 立即强制跑一次 A/B（会重启现有部署）
./run_same_host.sh "0 1 2 3 4 5 6 7 14 15"   # 强制补同机数据
```

## 9. 复现命令

```bash
# 1) 构建（103 的 CANN 9.1 容器内）；只编 4 个 async 算子，避开 GMM 的 bisheng 卡死
docker exec z00952334-afd487 bash -lc '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd /home/z00952334/afd-plugin
  python3 build/build_combine_variant.py --output build/variants/final_comm \
    --overlay build/overlays/empty \
    --ops afd_async_dispatch_send,afd_async_dispatch_recv,afd_async_combine_send,afd_async_combine_recv'

# 2) 两机各起一次（S 先起，因为它监听控制端口）。本次实测用的是 S=101 / R=103：
#    S: docker exec p00848063-afd bash -lc '<bench487.sh 内容>'   （node S）
#    R: docker exec z00952334-afd487   bash -lc '<bench487.sh 内容>'（node R）
# 等价的单侧命令（两侧参数必须完全一致，仅 --node 与 --output 不同）：
python3 tools/benchmarks/issue487/run.py \
  --sender-ip 80.5.17.101 --receiver-ip 80.5.17.103 --interface enp48s3u1u1 \
  --sender-devices 8,9,10,11,12,13,14,15 --target-devices 12,13 --placements cross \
  --node S --extension /home/z00952334/afd-plugin/afd_plugin/_C_ascend.cpython-312-aarch64-linux-gnu.so \
  --cs-helper /home/z00952334/afd-plugin/build/issue487-cs-helper/afd_plugin/_cann_ops_custom/vendors/afd-plugin \
  --vendor /home/z00952334/afd-plugin/build/variants/final_comm/afd_plugin/_cann_ops_custom/vendors/afd-plugin \
  --modes ds --aivs 48 --fan-ins 1 --gmm off --rounds 3 --samples 16 \
  --output /mnt/share/z00952334/issue487-results/ds48-S \
  --base-port 47000 --npu-port-base 32000 --control-port 28999

# 3) 汇总
python3 tools/benchmarks/issue487/run.py --summarize /mnt/share/z00952334/issue487-results/ds48-S
```

vendor 目录需在 101 上存在同一份（`scp -r` 到 `/home/z00952334/afd-plugin/build/variants/final_comm/`），
两机 `/home` 不共享。本次的便捷驱动脚本 `build/bench487.sh` / `build/bench487_pair.sh`
（`build/` 不进版本库）就是上面命令的封装：封装里每个 case 会算独立的
`--base-port/--npu-port-base/--control-port` 端口段，避免连续运行之间抢端口。

## 10. 未测项补齐（2026-11-03）

§3.4 与 §6.6 中标"未测"的同机单元格本轮已补齐：在 **103 的同一 10 卡窗口**（ordinals
4-13，8 sender + 2 target）用同一固定字节 harness、同一 `3 rounds × 16 samples` 协议，
对 baseline 与最终交付构建做**完全配对**的 A/B。跨机列不动（仍为 2026-11-02 的
S=101 / R=103）：本轮 10:22 之后 101 的 16 张卡被其他作业全部占满，无法重测。

### 10.1 环境、产物与判据

| 项 | 值 |
| --- | --- |
| 主机 | node-39-109（80.5.17.103），`z00952334-afd487`（CANN 9.1.0 / driver 26.0.rc2） |
| 运行时 | Python 3.12.13、torch 2.10.0+cpu、torch-npu 2.10.0.post4 |
| 设备 | Ascend910_9382，24 AIC / 48 AIV，L2 192 MB；senders 4-11，targets 12,13（同机 10 rank） |
| 协议 | `tools/benchmarks/issue487/run.py --placements same`，3 rounds、DR 每 round 16 ready groups；`BATCH_SIZE_FACTOR=0.0625`，除非注明 |
| baseline vendor | `afd_plugin/.../vendors/afd-plugin`：send `9f178cc1/da237d08`，recv `16ea3e4a/64660b77/d2b0f0bc/db886969` |
| optimized（GMM off） | `build/variants/final_comm`：send `2e65424a/3dccb14c`，recv `03a39d3b/08ecdbb3/41c27bce/cdd1124a` |
| optimized（GMM on） | `build/variants/final_comm_gmm`：baseline 的完整 op_api + `final_comm` 的 dispatch kernel（send/recv `.o` 与 `final_comm` 逐字节相同） |

`final_comm` 只打包四个 async 算子，不含 GMM kernel，所以 GMM 条件另做了一个
`final_comm_gmm`：以 baseline 完整 vendor 为底，只替换 `afd_async_dispatch_send/recv`
的 kernel 目录（`.o` 与同名 `.json`）；两处 config JSON 与 baseline 相同。harness 为此把
`--gmm-aic` 的取值扩到 12（production 配额），工具侧改动只有这一处。

12 个同机条件 × 2 臂 + 4 个 fan-in 8 复测条件全部 `correctness=True`（DS/DR 的 int8
逐字节、scale=1、专家计数、source 去重）。

### 10.2 同机完整矩阵（103，2026-11-03，µs）

| 场景（fan-in 1，除非注明） | baseline | optimized | 变化 |
| --- | ---: | ---: | ---: |
| DS AIV=4 | 3355.5 | 2055.0 | −38.8% |
| DS AIV=8 | 1700.4 | 1049.4 | −38.3% |
| DS AIV=24 | 590.9 | 433.5 | −26.6% |
| DS AIV=48 | 348.0 | 341.4 | −1.9% |
| DS AIV=48 + fan-in 8 | 771.3 | 805.1 | **+4.4%** |
| DR AIV=4 | 1586.2 | 764.6 | −51.8% |
| DR AIV=24 | 334.1 | 189.2 | −43.4% |
| DR AIV=48 | 213.2 | 143.1 | −32.9% |
| DR AIV=4 + GMM 22/44 | 2674.4 | 1681.1 | −37.1% |
| DR AIV=4 + GMM 12/24 | 1920.1 | 1056.0 | −45.0% |
| DR AIV=24 + GMM 12/24（production 形状） | 387.2 | 250.0 | −35.4% |
| DR AIV=4（`BATCH_SIZE_FACTOR=1.0`） | 1575.7 | 760.2 | −51.8% |

读法：

- **DS 的收益随 AIV 单调收敛**：4/8 AIV 都是 −38%，24 AIV −26.6%，48 AIV 只剩 −1.9%
  （每核 store 已经很少）。AIV=8 几乎与 AIV=4 同幅，说明 8 AIV 仍是"每核 copy 数主导"
  的形态；这也解释了为什么 serving 侧（48 AIV）看不出收益。
- **DR 在 24/48 AIV 仍有 −43.4% / −32.9%**：行读/写重叠折的是串行等待，不靠核数，
  所以高并行度下依然有效；AIV=24 正是 two-stream serving 的实际配额。
- 同机 DR 与跨机 DR 的绝对值几乎相同（aiv24：334.1/189.2 vs 跨机 333.4/192.6），再次
  说明 DR 读的是本地窗口，源端位置不影响它。

### 10.3 与 11-02 锚点的一致性

| 点（µs） | 11-03 同机（本节） | 11-02 22:04 同机 | 偏差 |
| --- | ---: | ---: | ---: |
| DS AIV=4 baseline | 3355.5 | 3374.5 | −0.6% |
| DS AIV=4 optimized | 2055.0 | 2067.0 | −0.6% |
| DS AIV=48 baseline | 348.0 | 348.4 | −0.1% |
| DS AIV=48 optimized | 341.4 | 337.8 | +1.1% |
| DR AIV=4 baseline | 1586.2 | 1588.1 | −0.1% |
| DR AIV=4 optimized | 764.6 | 774.2 | −1.2% |
| DR AIV=48 baseline | 213.2 | 204.7 | +4.2% |
| DR AIV=48 optimized | 143.1 | 160.0 | −10.6% |

除 DR AIV=48 外，两套窗口的锚点都在 1.2% 以内，说明 103 的 4-13 映射与 101 的 0-9
映射可以直接并表。DR AIV=48 的 optimized 差异来自**构建版本**：11-02 22:04 在 101 上
用的是当时 101 本地的 `final_comm`（recv `01b2340a/ab6367e2/f52f10b0/f84a6afa`，与最终
构建 `03a39d3b/...` 不同）；本轮统一用最终交付构建，且 baseline 在同窗口是 213.2
（+4.2%）。DR 的 4 AIV 两版只差 1.2%，说明该差异只体现在高 AIV 的短 kernel 上。因此
§3.4 的 DR AIV=48 同机行保留 11-02 值，但最终产物应看 143.1（−32.9%）。

### 10.4 fan-in 8 的可重复性（重要）

同机 DS AIV=48 fan-in 8 是本轮唯一出现**正向劣化**的点。为排除单次窗口噪声，在同样
10 张卡上又做了两轮 A/B（同脚本、同映射），共 3 组配对：

| 组 | baseline / µs | optimized / µs | 变化 |
| --- | ---: | ---: | ---: |
| 主矩阵 | 771.3 | 805.1 | +4.4% |
| 复测 1 | 755.4 | 797.9 | +5.6% |
| 复测 2 | 763.0 | 792.6 | +3.9% |
| 均值 | 763.2 | 798.5 | **+4.6%** |

这不是噪声：**同机 fan-in 8 下 payload 批量化带来 +4%~+6% 的回归**。它与 11-02 的跨机
fan-in 8（737.8→718.8，−2.6%）方向相反，说明 fan-in 8 的收益取决于拓扑：跨机时 store
完成延迟占主导，33 KB 的批量 store 仍能省等待；同机时完成延迟低，8 个 sender 同时向
同一目标窗口发 33 KB store，反而放大了目标入口/窗口竞争。这与复现报告 §9.5 的判断
一致——高 AIV fan-in 8 的瓶颈是共享资源，不是单次 store 延迟。

含义：P2 仍未解决，且**不能声称"没有引入 fan-in 回归"**。如果 production 存在 8 个
sender 同时打同一 FFN rank 的形态，需要按 fan-in 选择 stage 行数或关闭批量化；本轮
没有改代码规避，因为 fan-in 1 的收益（−38% 到 −52%）远大于 fan8 的 +5%，且 fan8 的
根因仍未定位。

### 10.5 production GMM 配额（12 AIC / 24 AIV）

production 的 FFN 侧是 `ffn_compute_aic_num=12` / `ffn_compute_aiv_num=24`（GMM）+
`ffn_combine_aiv_num=24`（DR recv stream）。harness 本轮新增 `--gmm-aic 12`，在同一
窗口测了 12/24 与 22/44 两个配额：

| 条件（DR，fan-in 1） | 无 GMM | GMM 12/24 | GMM 22/44 |
| --- | ---: | ---: | ---: |
| AIV=4 baseline | 1586.2 | 1920.1（+333.9 / +21.0%） | 2674.4（+1088.2 / +68.6%） |
| AIV=4 optimized | 764.6 | 1056.0（+291.4 / +38.1%） | 1681.1（+916.5 / +119.9%） |
| AIV=24 baseline（production 形状） | 334.1 | 387.2（+53.1 / +15.9%） | — |
| AIV=24 optimized（production 形状） | 189.2 | 250.0（+60.8 / +32.1%） | — |

结论：

- production 配额把 GMM 对 DR 的增量从 22/44 的 ~1.1 ms 压到 ~0.33 ms（aiv4，−70%），
  在 aiv24 只有 ~53–61 µs；§6.6 原先"预期增量更小但未测"的判断成立，且已量化。
- 12/24 的增量在**绝对 µs 上近似相加**（aiv4 ~300 µs、aiv24 ~55–60 µs），不随优化缩放；优化后
  base 变小，相对增量反而变大（aiv4 +38.1%、aiv24 +32.1%）。净时间仍是 −45.0% / −35.4%。
- 也就是说：production 配额下 GMM 与 DR 的相互作用不再是主要成本，DR 的剩余成本主要
  是本地读本身。

### 10.6 仍未覆盖的部分

1. **跨机重测**：本轮 10:22 起 101 的 16 张卡被其他作业全部占满，未能重测；表内跨机列
   仍是 2026-11-02 的 S=101 / R=103 数据。受影响最大的是 fan-in 8 的跨机行（仍为
   737.8→718.8），它与本轮同机的 +4.6% 不是同一拓扑，不能相互替代。
2. **8 目的地形状（§6.5）**：harness 固定 10 rank（8 sender + 2 target），production 是
   8 个 FFN rank；要直接测需要 16 rank 和协议改动，本轮仍未做。§6.5 的估算是目前唯一依据。
3. **DBO / 多 stream 重叠**：算子级 harness 是单 stream；单机 E2E（§8.2）DBO 是关的，
   跨机 E2E 没有有效数据。这一项仍无实测。
4. **跨机 serving**：仍被 §8.1 的运行期崩溃与 101 资源占用挡住。

复现命令（103，同机 10 rank；senders 4-11 / targets 12,13）：

```bash
docker exec z00952334-afd487 bash -lc '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd /home/z00952334/afd-plugin
  python3 tools/benchmarks/issue487/run.py \
    --sender-ip 127.0.0.1 --receiver-ip 127.0.0.2 --interface enp48s3u1u1 \
    --sender-devices 4,5,6,7,8,9,10,11 --target-devices 12,13 \
    --placements same --node S --modes dr --aivs 24 --fan-ins 1 \
    --gmm on --gmm-aic 12 --rounds 3 --samples 16 \
    --extension /home/z00952334/afd-plugin/afd_plugin/_C_ascend.cpython-312-aarch64-linux-gnu.so \
    --vendor /home/z00952334/afd-plugin/build/variants/final_comm_gmm/afd_plugin/_cann_ops_custom/vendors/afd-plugin \
    --output /mnt/share/z00952334/issue487-supplement/example
```

驱动脚本（`build/` 不进版本库）本轮放在 `/mnt/share/z00952334/issue487-supplement/`：
`run_arm.sh`（单臂矩阵）、`orchestrate.sh`（四段顺序执行）、`run_fan8_ab.sh`（fan-in 8
A/B/A/B）、`summarize.py`（按 case 汇总）。原始 case JSON（去掉 rank 日志）与
`summary.csv` 归档在 `build/results/issue487-supplement-2026-11-03/`。
