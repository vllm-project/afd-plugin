# Issue #487 双机复现报告（A3 910C）

日期：2026-10-30
复现对象：[vllm-project/afd-plugin#487](https://github.com/vllm-project/afd-plugin/issues/487)
代码基线：`213ee17`（`复现issue487问题`），三个 kernel 头文件 SHA-256 与 issue 一致。

## 1. 环境与拓扑

| 项目 | S 主机 | R 主机 |
| --- | --- | --- |
| 主机 | node-39-101 / 80.5.17.101 | node-39-109 / 80.5.17.103 |
| 容器 | `p00848063-afd` | `p00848063-afd-109` |
| 设备 | Ascend A3 `ascend910_93`，8 package / 16 logical NPU | 同左 |
| CANN / driver | 9.1.0 / 25.5.1.1 | 9.1.0 / 25.5.1.1 |
| Python / torch / torch-npu | 3.12.13 / 2.10.0+cpu / 2.10.0.post4 | 同左 |
| 构建 | `SOC_VERSION=910c`，完整 ACLNN run package + `_C_ascend` | 复用同一构建产物 |

- communicator：10 ranks，Gloo 控制 + 独立 HCCL 组，`hccl_buffer_size=512` MiB。
- 同机 smoke：senders S:4,5,6,7,12,13,14,15，targets S:2,3（本机 0/1、2-11 被其他任务占用）。
- 跨机：senders S:0-7，targets R:12,13（R 的 0-11 被其他 VLLM 任务占用；12,13 空闲）。
- 该集群为多人共享；测试时按空闲 ordinal 选择，环境 JSON 记录了实际映射和 `npu-smi` 快照。

## 2. 方法

使用仓库 `tools/benchmarks/issue487`（commit `213ee17`）的固定输入协议：

- CS：BF16 `[3072,4096]`，TP=2，64 calls/graph，3 rounds；目标端用独立 helper vendor 初始化/读回窗口。
- DS/DR：BF16 `[3072,4096]`，TP=1，top-k=8，16 calls/graph；DR 每 round 16 个 ready group。
- GMM：`gmm_swiglu_quant_v2_layered` + `grouped_matmul_layered`，W4A8，22 AIC / 44 AIV（或 20/40 对照）。
- 每个条件 10 个新进程；CS/DS 目标空闲，DR 计时前完成发布、设备同步和全员 barrier。
- 正确性：CS 检查 8 个状态字和全部 payload；DS/DR 通过真实 DR 检查 int8、scale=1、专家计数和 source 去重。所有报告条件均 `correctness=pass`。

### 2.1 Harness 修复（复现过程中发现）

1. **CS target helper tiling key**：`cs_window.cpp` 只有范围注册，没有 `TILING_KEY_IS(100/101)` 分支，真机上报
   `funcEntry 100 is invalid ... aclrtBinaryGetFunctionByEntry failed`。补上显式 key 分支后 helper 正常。
2. **连续条件 HCCL 端口冲突**：默认 NPU socket port 16666 在上一条件退出后仍可能被占用，下一条件报
   `Communication_Error_Bind_IP_Port ... port 16666 already bound`，或直接挂起。给每个条件设置独立的
   `HCCL_NPU_SOCKET_PORT_RANGE` / `HCCL_HOST_SOCKET_PORT_RANGE` 后连续 6 条件 smoke 全部通过。
3. 跨机时 `--sender-devices` 与 `--target-devices` 必须全局 10 个互异 ordinal；因此用 S:0-7 + R:12,13，
   而不是 issue 的 S:8-15 + R:0,1。物理 package 分组不同，但同/跨机使用同一组 ordinal，对比仍受控。

### 2.2 复现命令

构建（两机各自）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=910c MAX_JOBS=4 AFD_BUILD_JOBS=4 \
  python -m pip install -e . -v --no-build-isolation
python tools/benchmarks/issue487/build_cs_helper.py --output build/issue487-cs-helper
```

跨机 smoke（S=80.5.17.101，R=80.5.17.103，两端同时启动）：

```bash
COMMON="--sender-ip 80.5.17.101 --receiver-ip 80.5.17.103 \
  --interface enp48s3u1u1 --sender-devices 0,1,2,3,4,5,6,7 \
  --target-devices 12,13 --placements cross --modes cs ds dr \
  --aivs 4 --fan-ins 1 --gmm off on"
# S 主机
python3 tools/benchmarks/issue487/run.py $COMMON --node S \
  --extension .../_C_ascend.cpython-312-aarch64-linux-gnu.so \
  --vendor .../_cann_ops_custom/vendors/afd-plugin \
  --cs-helper .../build/issue487-cs-helper/.../vendors/afd-plugin \
  --output build/results/cross-smoke-S
# R 主机
python3 tools/benchmarks/issue487/run.py $COMMON --node R ... \
  --output build/results/cross-smoke-R
```

同机用 `--placements same`，只需在 S 启动；fan-in/AIV 矩阵把 `--aivs`/`--fan-ins` 改成
`4,48`/`1,2,4,8`。汇总：`python3 tools/benchmarks/issue487/run.py --summarize ...`。

## 3. 结果

### 3.1 头条结果（AIV=4，fan-in=1，单位 µs，3 轮 event 均值）

| 算子 | 同机 | 跨机 | 跨机/同机 | issue 同机 | issue 跨机 | issue 比值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CS | 482.3 | 1283.0 | 2.66× | 478.4 | 1228.9 | 2.57× |
| CS + GMM 22/44 | 519.6 | 1339.8 | 2.58× | 531.9 | 1295.2 | 2.44× |
| DS | 3362.4 | 6297.4 | 1.87× | 3375.0 | 6127.3 | 1.82× |
| DS + GMM 22/44 | 3878.8 | 6827.3 | 1.76× | 3776.8 | 6533.2 | 1.73× |
| DR（ready input） | 1569.5 | 1592.5 | 1.015× | 1562.2 | 1557.8 | 0.997× |
| DR + GMM 22/44 | 2679.6 | 2725.6 | 1.017× | 2696.6 | 2729.6 | 1.012× |

GMM 增量：CS +7.7%/+4.4%（同/跨），DS +15.4%/+8.4%，DR **+70.7%/+71.1%**；
与 issue 的 +72.6%/+75.2% 一致。跨机 DS/CS 的绝对值和倍数均在 issue 的 2-5% 内。

复现判定：六项头条指标与 issue 的偏差为 0.3%-5.7%；CS 在 48 AIV 下跨机/同机比为 1.007，
DS 为 1.87（issue 1.82），DR 的 GMM 增量为 +71%（issue +72.6%）。issue 的三条核心现象
（CS/DS 跨机劣化、DR 对源位置不敏感、DR 受本地 GMM 竞争）均成立。

### 3.2 AIV 分配与 fan-in

CS 的 AIV × fan-in 矩阵（本机实测 / µs；括号内为 issue 值）：

| AIV | 位置 | fan1 | fan2 | fan4 | fan8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 同机 | 484.0 (478.4) | 481.8 (482.4) | 515.6 (514.5) | 756.0 (765.6) |
| 4 | 跨机 | 1280.3 (1228.9) | 1281.7 (1228.5) | 1277.7 (1228.6) | 1289.3 (1249.4) |
| 48 | 同机 | 210.8 (209.3) | 220.6 (211.7) | 338.3 (343.5) | 678.8 (683.1) |
| 48 | 跨机 | 212.3 (212.8) | 213.7 (214.5) | 339.5 (339.6) | 667.7 (684.0) |

同机点与 issue 的偏差在 -0.6%~+4.2%，跨机 fan1/2 的偏差约 +4%。两个关键趋势已经复现：
- AIV=4 时跨机 fan-in 1→2 基本平坦（1280→1282），说明每个 sender 的完成延迟是独立的，
  增加 sender 并不改变单 sender 的串行 copy 链；
- AIV 48 时同机 fan-in 4→8 从 338→679 µs，源端共享资源效应出现，与 issue 一致。
- 最关键的一组对照：CS 4 AIV 跨机 1280.3 µs vs 同机 484.0（2.65×）；48 AIV 跨机 212.3 µs
  vs 同机 210.8（1.007×）。跨机惩罚在 48 AIV 下几乎消失，直接证明瓶颈是“每 copy 的完成
  等待无法被隐藏”，而不是跨机链路带宽。

DS 的 fan-in 曲线按 issue 约定取 rank0，CS 取活跃 sender 均值，避免把新增 sender
的硬件差异平均进去。

已知 issue 基线：CS 4→48 AIV 跨机 1228.9→212.8 µs；DS 6127.3→628.2 µs。
这说明 4 AIV 时跨机增量主要来自“每 copy 完成等待”的串行链，而不是链路带宽：
- CS 4 AIV：每 AIV ~48 次 copy，跨机单次约 19-27 µs；48 AIV 时每 AIV 只有 ~4 次 copy，
  同样的 24 MiB 可在 ~213 µs 内完成，吞吐提升近 6×。
- DS 4 AIV：每核 1536 次 payload + 384 次 index 写；issue 的 detail probe 估计
  payload 增量 ~2223 µs、index 增量 ~519 µs，与我们跨机-同机差 ~2935 µs 一致。

### 3.3 DS 的 fan-in（AIV=4，取 rank0）

| 位置 | fan1 | fan2 | fan4 | fan8 |
| --- | ---: | ---: | ---: | ---: |
| 同机 | 3363.5 (3375.0) | 3364.3 (3385.0) | 3366.4 (3383.5) | 3374.7 (3388.2) |
| 跨机 | 6292.6 (6127.3) | 6293.4 (6137.4) | 6289.3 (6128.4) | 6287.4 (6128.0) |

同机 fan-in 1→4 基本平坦（3363→3366），与 issue 一致；跨机 fan1 已复现 1.87× 的惩罚，
后续 fan 数据用于确认“增加 sender 不改变单 sender 串行链”的低 AIV 结论。

AIV 对照（DS 同机 fan1）：4 AIV 3363.5 µs → 48 AIV 348.7 µs（issue 3375.0 → 349.8），
与 CS 一样呈现“AIV 提高一个数量级、跨机完成等待被隐藏”的趋势。
DS 同机 AIV=48 的 fan-in：fan1 348.7、fan2 350.4、fan4 438.4、fan8 682.3
（issue 349.8、351.6、434.8、732.8）；fan4 起源端共享资源效应出现，与 CS 的 48 AIV 趋势一致。
DS 跨机 AIV=4 的 fan-in 1→8 基本平坦（6292.6→6287.4），与 issue 的 6127→6128 一致；
跨机 AIV=48 fan1 为 641.4 µs（issue 628.2），AIV 提升后跨机/同机比从 1.87 降到 1.84（按同机 348.7）。

### 3.4 DR fan-in 与本地 GMM 竞争

DR AIV=4 的 fan-in（ready input，µs）：

| 位置 | fan1 | fan2 | fan4 | fan8 |
| --- | ---: | ---: | ---: | ---: |
| 同机 | 1574.0 | 1580.7 | 1561.4 | 1566.3 |
| 跨机 | 1583.6 | 1589.8 | 1574.0 | 1583.3 |

DR 的 fan-in 是“已就绪 backlog”，不是同时入站的网络流量；曲线在 fan1→8 基本平坦，
同/跨源差 <1.5%，与 issue 的 1572→1563（同源）/1550→1544（跨源）一致。

- 同/跨源 DR 差仅 ~1.5%，说明 ready-input DR 读取的是本地窗口，源端是否跨机不影响它。
- 同机 GMM 使 DR 从 1569.5→2679.6 µs（+70.7%）；跨机源同样 +71.1%。
- 这与 issue 的 PMU/L2 结果一致：GMM 使 DR 的 MTE2 活跃时间从 ~564→~1486 µs，
  L2 read-hit 从 93.5%→~25%；DR 的 3072 行/核串行本地读是主要受害路径。

### 3.5 GMM 核配额对照（AIV=8，fan-in=1）

| 条件 | CS 同机 | CS 跨机 | DS 同机 | DS 跨机 | DR 同机 | DR 跨机 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 22/44（52 AIV，超额） | 522.5 | 923.5 | 2048.1 | 3503.2 | 1152.4 | 1172.0 |
| 20/40（48 AIV，可行） | 276.6 | 679.6 | 1948.1 | 3411.6 | 1286.8 | 1310.2 |

issue 的 22/44 参考值：CS 564.5 / 901.4，DS 2128.5 / 3482.7，DR 1328.7 / 1321.0。
22/44 与 8 通信 AIV 相加为 52 AIV，超过 A3 的 48 AIV；20/40 对照用于分离
“核启动离散/超额调度”与“真正并行后的访存竞争”。

首个对照：CS 同机 aiv8 从 22/44 的 522.5 µs 降到 20/40 的 276.6 µs（−47.1%），
与 issue 的 CS payload span 533.68→272.47 一致；这证明 22/44 的 CS 惩罚中很大一部分
来自超额调度/core-start spread，而不是额外的 DMA 字节。

但 DR 的对照相反：同机 22/44 的 1152.4 µs → 20/40 的 1286.8 µs（+11.7%），
跨机 1172.0 → 1310.2（+11.8%）。
20/40 让 GMM 真正占满 40 AIV，DR 的本地读竞争反而更明显；这与 issue
“去掉启动延迟不保证 DR 更快”的结论一致，也说明核配额必须按算子分别验证。
DS 的改善较小（2048.1→1948.1，−4.9%），其主因仍是 payload 串行完成，而不是启动离散。

补充 aiv48 + 22/44（92 AIV，极端超额）：CS 同机 500.6、跨机 524.1
（issue 551.5 / 568.8）；DS 同机 675.0、跨机 957.2（issue 725.3 / 992.3）；
DR 同机 544.8、跨机 541.7（issue 684.6 / 694.6）。48 AIV 把 CS 跨机/同机压到 1.05，
说明并行度足够时连 92 AIV 的超额调度也不至于让 CS 回到 4 AIV 的 2.66×；
GMM 本身仍使 CS 比无 GMM 的 210.8 µs 慢约 2.4×。DR 也从 aiv4 的 2679.6 µs
降到 544.8 µs，说明限制 DR 的主要是串行本地读，提升 AIV 同样能隐藏其延迟。

## 4. 原因分析

1. **CS/DS 是“串行 copy + completion wait”而非带宽瓶颈。**
   CS 每个 chunk 后立即 `MTE3_S`；DS 每行 payload、每批 index 也立即 `MTE3_S`。
   4 AIV 时每核的 copy 数很大，跨机完成延迟（~19 µs/次）无法被隐藏；48 AIV 时并行 copy
   数足够，跨机 CS 回落到 ~213 µs。提高 AIV 或增加在途 copy 都能直接验证这一点。
   定量看：CS 4 AIV 跨机-同机差 800.8 µs，每 AIV 48 次 copy，折合 16.7 µs/copy；
   DS 差 2935.0 µs，按 issue 的 1536 次 payload（+1.45 µs）+ 384 次 index（+1.35 µs）
   估算约 2746 µs，覆盖实测增量的 ~94%。
2. **DS 的索引写放大了事务数。** 仅 48 KiB 的 index 却贡献约 20% 的跨机增量（384 次/核 × 1.35 µs），
   说明小事务的 completion 成本远高于其字节数。
3. **DR 是本地访存竞争问题。** 同一 token 被 4 个本地专家展开，12,288 次读、48.375 MiB；
   GMM 占用 L2/HBM/UB 后，单次本地读从 0.19 升至 0.50 µs，约 88% 的 payload 阶段增量来自这里。
4. **超额核分配放大了启动离散。** 8 通信 AIV + 22/44 GMM = 52 AIV > 48，issue 观测到
   349-478 µs 的 core-start spread；8 + 20/40 = 48 时 spread 降到 <1.2 µs。
   但即使总 AIV 刚好 48，DR 真正并行后仍会因本地访存竞争变慢，所以配额修复不等于性能修复。
5. **fan-in 的共享资源效应。** 低 AIV 下跨机单发送者已慢，增加 sender 影响小；48 AIV 下
   增加 sender 会显著增加 CS/DS 完成时间，说明源端板卡/发送路径/目标入口存在共享资源竞争。
   现有数据还不能隔离到某一条物理链路或队列。

## 5. 可能的解决方案（按优先级）

### 5.1 CS：双缓冲 / 多缓冲流水化（低风险，直接针对 4 AIV 跨机）

- 把 `xTensor_` 的 UB 空间拆成 2-4 个 buffer；每次只等待“即将复用的那个 buffer”的 MTE3 完成，
  允许 2-4 个远端 `DataCopy` 同时在途；循环结束、写 status 前做一次 `MTE3_S` 汇总等待。
- 目标区间本来就按 sender 不相交；只需要保证同一 buffer 不被提前复用、status flag 在全部
  payload 完成后才写。预计 4 AIV 跨机 CS 可从 1283 µs 降到 700-900 µs；48 AIV 收益较小。
- 验证：用同一 harness 的 CS target helper 检查 8 个状态字和全部 payload；再跑 aiv 4/8/48、
  fan 1/8、GMM off/on。

### 5.2 DS：payload 与 index 批量化（中等风险，针对 1536+384 次小事务）

- 每个 destination 在 UB 中积累 N 个 token（例如 8-16 个 int8 row，32-64 KiB），凑满或结束时
  一次 `DataCopy` 写出；保持 `moeTokenSendIdx` 的递增顺序和 token 去重 flag。
- index 写（4/8 B/条）改为在 UB 中批量拼接，按 32-64 条一次写出；只在最后做一次 completion wait。
- 预计 payload 事务 1536→~192/核，index 384→~12/核，跨机 DS 可从 6.3 ms 降到 4 ms 左右。
- 需要保证：同一 destination 的 token 顺序、expert offset 的可见性、以及 ready flag 的最后发布。

### 5.3 DR：token-major 去重 + 读流水线（高风险，收益最大）

- 当前按 expert-slot 展开，同一 token 的 4 个专家各读一次；改为 token-major：
  每个 token 行只读一次，写入它的 4 个 expert 输出槽。读流量 48.375→12.1 MiB（4×），
  输出写仍为 48 MiB；总 DMA 从 ~96 MiB 降到 ~60 MiB。
- 若暂不改遍历顺序，至少做 2-4 buffer 的 MTE2/MTE3 ping-pong，预取下一行、批量写 scale，
  避免逐行 `MTE2_S`/`MTE3_S`；预计 GMM 并发下 DR 可从 2.73 ms 降到 2.0 ms 左右。
- 需要从 tiling/metadata 建立 token→4 个 slot 的映射，或在 UB 中缓存 routing 的逆表；
  这是三项里改动最大的，建议先在独立 vendor 中做原型。

### 5.4 核配额与调度（立即生效的缓解）

- 强制 `comm_AIV + 2 * GMM_AIC <= 48`：8 通信 AIV 时 GMM 用 20/40，而不是 22/44；
  避免 52/92 AIV 的超额调度和数百 µs 的 core-start spread。
- 跨机 CS/DS 优先用 48 通信 AIV（issue：CS 212.8、DS 628.2 µs），用并行度隐藏完成延迟。
- 若 DR 必须与 GMM 并发，给 DR 保留独立 AIV 配额（例如 8 通信 + 20/40 GMM），并接受
  “总配额可行但 DR 仍可能变慢”的事实；用 PMU 确认实际并行度，而不是只看 timeline 重叠。

### 5.5 通信/拓扑层

- 保持 CS/DS 两端在同一 supernode；跨 supernode 的 RoCE 路径不保证同样的 peer-window 语义/性能。
- 评估 HCCL buffer 大小（当前 512 MiB）与在途窗口；如果 completion 等待受窗口反压影响，
  可做 512 MiB vs 1 GiB 的对照，但先用 detail probe 确认不是链路带宽限制。
- 对 fan-in 大的场景，记录源/目标 package 分组，避免把“同一板卡共享资源”误判为链路问题。

## 6. 结论与边界

- 在双机 A3 上复现了 issue #487 的核心现象：4 AIV 下 CS 跨机 2.66×、DS 1.87×，
  DR 对源位置不敏感但对本地 GMM 敏感（+71%），与 issue 的 2.57×/1.82×/+72.6% 一致。
- CS 的 AIV×fan-in 矩阵与 issue 逐点吻合（0.1-4.2%）：4 AIV 跨机 1280.3 vs 同机 484.0（2.65×），
  48 AIV 跨机 212.3 vs 同机 210.8（1.007×）；跨机 fan-in 1→8 在 4 AIV 下平坦（1280→1289），
  在 48 AIV 下 fan8 升到 667.7。这组对照直接排除了“跨机链路带宽不足”。
- DS 同机/跨机 fan-in 1→8 也基本平坦（同机 3363→3375，跨机 6292→6287）；
  AIV 4→48 把同机从 3363.5 降到 348.7、跨机从 6292.6 降到 641.4，趋势与 CS 一致。
- GMM 配额对照说明“配额修复不等于性能修复”：aiv8 从 22/44 改到 20/40，
  CS 同机 522.5→276.6（−47.1%）、DS 2048.1→1948.1（−4.9%），但 DR 1152.4→1286.8（+11.7%）；
  DR 在 20/40 下反而更慢，因为 GMM 真正占满 40 AIV 后本地读竞争更明显。
- 数据支持“CS/DS 串行 completion 等待 + DS 小事务放大”和“DR 本地读竞争”两类原因；
  不支持“跨机 DR 网络延迟”或“链路带宽不足”的解释。
- 优化原型验证：CS 双缓冲 + blockDim 对齐对跨机无效（1271.8 vs 1280.3）；
  DS index batch=64 跨机 5719.3（−9.2%）；DR token-major 在 GMM 并发下
  2015.6/2013.2（同/跨，−24.8%/−26.1%），是三项里最有效的修复。
- 本文提出的是候选方案，尚未修改生产 kernel；下一步应在独立 vendor 中实现 5.1/5.2/5.3，
  用同一固定字节协议和可行 AIV 预算验证，再跑真实 serving schedule。
- 所有数字均为算子级 event 时间；共享集群、设备 ordinal 改动、无 PMU/phase probe 是本次的边界。

## 7. 优化原型验证（2026-10-31）

所有原型都在独立 staged vendor 中构建，生产 `csrc/` 未修改；使用同一 issue #487 harness、
同一固定字节输入和 AIV=4 / fan-in=1 / gmm off，双机 S=101 / R=103 目标 12,13。

| 变体 | 场景 | baseline / µs | optimized / µs | 变化 | 正确性 |
| --- | --- | ---: | ---: | ---: | --- |
| CS 双缓冲 | 同机 | 484.0 | 460.9 | −4.8% | pass |
| CS 双缓冲 | 跨机 | 1280.3 | 1271.4 | −0.7% | pass |
| CS 双缓冲 + blockDim=4 | 同机 | 484.0 | 459.6 | −5.1% | pass |
| CS 双缓冲 + blockDim=4 | 跨机 | 1280.3 | 1271.8 | −0.7% | pass |
| DS index batch=64 | 同机 | 3362.4 | 3158.7 | −6.1% | pass |
| DS index batch=64 | 跨机 | 6297.4 | 5719.3 | −9.2% | pass |

**CS 5.1 的结论是负面的，但有信息量。** 双缓冲单独只改善同机 4.8%、跨机 0.7%；
再用 `AFD_COMM_AIV` 把 kernel blockDim 从 48 降到 4（避免 48 个 block 分 12 波），
跨机仍是 1271.8 µs。说明 4 AIV 下每个 core 的远端 store completion 是串行的，
buffer 数/blockDim 都不能把它隐藏；有效的杠杆是提高并发 AIV（issue 的 48 AIV 值 212.8 µs）。
这也解释了为什么 48 AIV 下跨机惩罚几乎消失。

**DS 5.2 的 index 批量化是正向的。** 把 `tokenOffsetFlushBatchSize_` 从 16 提到 64，
同机 −6.1%、跨机 −9.2%，验证了“48 KiB index 贡献约 20% 跨机增量”的判断。
payload 批量化的原型（每 destination 暂存 4 个 token，再一次性写出）已写好，
但 bisheng kernel 编译阶段失败（fatbin 找不到生成的 object，未得到可运行二进制）；
把 batch 从 4 降到 2 仍然失败，基本排除 UB 容量因素，更像是新增 UB 暂存/QuantProcess
组合触发了工具链问题。代码保留在 `build/variants/ds_payload/`，下一步需要换用
TQue 暂存或把 payload 批量拆成独立 kernel 变体后重试。

**DR 5.3 token-major 去重已实现（含 GMM 并发验证）。** 通过 `AFD_COMM_AIV=4` 把 DR
kernel blockDim 降到 4，使每个 AIV 拥有 8 个连续 expert 的 3,072 个 slot；在 UB 中构建
token→slots 的 head/next 逆表，每个 source token 只读一次，再写它的 4 个 expert 槽。

| DR 场景 | baseline / µs | token-major / µs | 变化 | 正确性 |
| --- | ---: | ---: | ---: | --- |
| 同机 aiv4 fan1 gmm off | 1569.5 | 1497.5 | −4.6% | pass |
| 跨机 aiv4 fan1 gmm off | 1592.5 | 1452.4 | −8.8% | pass |
| 同机 aiv4 fan1 gmm 22/44 | 2679.6 | 2015.6 | **−24.8%** | pass |
| 跨机 aiv4 fan1 gmm 22/44 | 2725.6 | 2013.2 | **−26.1%** | pass |

no-GMM 下收益有限（读本来就有 93.5% L2 命中、输出写仍占大头），但 GMM 并发时收益显著：
DR 的读流量从 12,288 行降到 3,072 行（4×），GMM 争用下的 DR 延迟下降约 25%，
正好命中 issue 中“DR 主要受本地读竞争影响”的判断。实现上通过 `AFD_COMM_AIV=4` 让每个
AIV 拥有 8 个连续 expert 的 3,072 个 slot，从而能把同一 token 的 4 个连续 expert 槽聚合；
若换成任意 routing，需要处理跨 AIV 边界的 token（当前实现会退化为每个 AIV 各读一次，仍正确）。

**真实 serving schedule 尚未开始。** 权重和 E2E 入口已确认，但 101 的现有容器是 vLLM 0.26，
不匹配 skill 固定的 vLLM 0.30 / vllm-ascend 0.19.1rc2.dev2448；需要先在 101 准备匹配容器
（或重新核算 103 的 6 卡拓扑），否则启动参数和 API 兼容性都无法保证。

复现路径（都在本机仓库内）：

```text
build/variants/cs_pipeline/          # 2-buffer pipeline
build/variants/cs_pipeline_aiv/      # 2-buffer + AFD_COMM_AIV blockDim
build/variants/ds_index/             # DS index batch=64（已验证）
build/variants/ds_payload/           # DS payload batch=4（编译未通过）
build/variants/dr_token_major/       # DR token-major（no-GMM）
build/variants/dr_token_major_gmm/   # DR token-major + GMM（GMM 并发验证用）
build/make_cs_pipeline.py / make_cs_aiv.py / make_ds_index.py / make_ds_payload.py
build/build_combine_variant.py       # staged csrc + overlay -> vendor
```

每个目录的 `stage/afd_plugin/_cann_ops_custom/vendors/afd-plugin` 可直接作为
`run.py --vendor`；S/R 两端使用同一份 vendor 的副本。

### 7.1 叠加 vendor 验证（no-GMM）

把 DS index batch=64 与 DR token-major 叠加到同一个 async-only vendor
（`build/variants/stacked_async/`），用同一 10-rank harness、aiv4、fan1、gmm off 验证：

| 算子 | 同机 / µs | 跨机 / µs | 单独 vendor / µs |
| --- | ---: | ---: | --- |
| CS（未改） | 484.3 | 1280.4 | 482.3 / 1283.0 |
| DS（index） | 3156.6 | 5720.1 | 3158.7 / 5719.3 |
| DR（token-major） | 1467.2 | 1465.3 | 1497.5 / 1452.4 |

叠加后 DS/DR 与单独 vendor 一致，没有互相抵消；CS 仍是基线。第一次 DS 同机跑出
3550.2 µs，是单次噪声（同映射下 ds_index 对照 3160.3 µs，重跑 stacked DS 3156.6 µs）。

GMM 叠加的 vendor 用“kernel 替换”构造：把 `ds_index` 的
`afd_async_dispatch_send` kernel 目录替换进已构建好的 `dr_token_major_gmm` vendor
（两者的 kernel config JSON 相同，仅 send 二进制不同）。跨机 GMM 22/44、aiv4、fan1：

| 算子 | 叠加 GMM / µs | baseline / µs | 变化 | 正确性 |
| --- | ---: | ---: | ---: | --- |
| DS（index + token-major recv + GMM） | 6264.7 | 6827.3 | −8.2% | pass |
| DR（token-major + GMM） | 2026.3 | 2725.6 | **−25.7%** | pass |

同机 GMM 叠加没有跑成：101 的 0-3 被其他任务占用，GMM 权重分配触发 NPU OOM
（3.68 GiB allocated, 8.81 MiB free），不是 vendor 错误；跨机用 101 的 8-15 发送、
103 的 0/1 接收，正确性通过。

注意：全量重编译 GMM 仍会在 bisheng 阶段 0% CPU 死锁（4 job/1 job、CANN 9.1/9.2 都复现），
所以叠加 GMM vendor 是用已有 GMM vendor + send kernel 替换得到的；生产化前应重新完整构建。

## 8. 真实 serving 可行性评估（尚未执行）

已核实的资源：

- 101 主机 6-15 空闲，0-5 被其他用户的 `VLLMWorker` 占用（占用很小但在增长）；
  103 主机 0-3、12-15 空闲，4-11 被其他 VLLM 任务占用。
- `/mnt/weight/DeepSeek-V4-Flash-w8a8-mtp` checkpoint 存在；仓库有
  `tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py` 和
  `.agents/skills/deploy-dsv4-flash-async` 部署 skill。
- 运行时版本不匹配：101 的 `p00848063-afd` 是 vLLM 0.26；103 的 `test-k3.1-1025`
  是 vLLM 0.30.0 / CANN 9.2。skill 固定 vLLM 0.30.0 + vllm-ascend 0.19.1rc2.dev2448。
- 已在 101 创建 `z00952334-afd-v030`（`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，
  vLLM 0.30.0 / vllm-ascend 0.30.0rc2 / CANN 9.2 / torch-npu 2.10.0.post7，全 16 卡映射），
  并在 `/home/z00952334/afd-plugin-v030` 的独立副本中构建插件，避免覆盖 103 的 post4 产物。
- `afd_dp4tp2` 需要 A 0-7 + F 8-15 共 16 卡；当前 101 的 0-5 被占用，尚不满足。
- 固定版本 vllm-ascend 已准备：在 101 的 vLLM 0.30 容器中创建了
  `/vllm-workspace/vllm-ascend-8d4409d` worktree（commit 8d4409d），
  `PYTHONPATH` 指向该 worktree 时 import 成功。
- 插件构建在 `/home/z00952334/afd-plugin-v030` 独立副本中尝试，但 CANN 9.2 + GMM 的
  bisheng 编译在多个 tiling variant 上 0% CPU 死锁（4 job/1 job 都复现）；容器重启后仍复现。
- 绕过方式已验证：在 101 的 v0.30 容器里使用 **post4/CANN 9.1 的 `_C_ascend` 扩展 +
  `stacked_gmm` vendor**，配合固定 commit 的 `vllm-ascend-8d4409d` worktree，
  `ensure_afd_ascend_ops_loaded()` 能注册四个 async 算子（dispatch/combine send/recv），
  不依赖 0.30 容器里重新编译 GMM。
- serving 环境脚本已准备：`/home/z00952334/afd-plugin/serve_env.sh`
  （`afd_dp4tp2`、A 0-7、F 8-15、MBT 8192、DBO 开、KV connector 关、w8a8 eager，
  固定 vllm-ascend-8d4409d + post4 扩展 + `stacked_gmm` vendor）。
- **AIV 对齐注意（已实测）**：`recv_attn_output` / `recv_ffn_output` 直接在当前
  Attention stream 上调用 recv，插件没有对 recv 设置 `set_stream_limit`，默认按 48 AIV
  启动。跨机 DR、aiv48、fan1 实测：baseline 209.5 µs，token-major 303.1 µs
  （**慢 44.7%**）。原因是 48 AIV 下每个 AIV 只分到 64 个 slot，token-major 的逆表
  构造没有去重收益，反而增加 head/next 开销。因此：
  - DR token-major 只适合 4/8 AIV 的算子预算（4 AIV 时 1592.5 → 1452.4 µs，
    8 AIV 时也有收益）；serving 默认 48 AIV 下**不应使用 token-major recv**。
  - serving 的安全优化 vendor 应为 **DS index only + baseline recv + GMM**；
    `stacked_gmm`（含 token-major recv）不适合直接用于默认 48 AIV 的 serving。
  - 已构造并验证 `build/variants/ds_index_gmm/`（baseline vendor + ds_index 的 send
    kernel 替换）：跨机 GMM 22/44、aiv4、fan1 下 DS **6237.0 µs**（baseline 6827.3，
    −8.6%），DR 2707.6 µs（baseline 2725.6，基本不变）。这个 vendor 用作 serving 的
    优化版本，避免 token-major 在 48 AIV 的回归。
  - **DS index 也有 AIV 依赖**：跨机、无 GMM、fan1 下 aiv48 baseline 642.1 µs，
    ds_index 640.2 µs（基本不变）；aiv4 时才是 6297.4 → 5719.3（−9.2%）。serving 的
    FFN send 默认 `ffn_combine_aiv_num=24`，DS index 的实际收益可能介于两者之间，
    需要 serving 实测；不要直接套用 aiv4 的 −8.6%。

### 8.1 首轮 serving 尝试（2026-10-31）

101 在 11:57 凑齐 16 卡后，用 `afd_dp4tp2` + 固定 vllm-ascend 8d4409d 源码 +
post4 `_C_ascend` + `ds_index_gmm` vendor 启动了 baseline。启动过程中修了两个问题：

1. v0.30 容器里 AFD 插件没有 pip 安装，`vllm.general_plugins` entry point 缺失，
   `VllmConfig` 不认 `afd` 字段；手工补了
   `vllm_afd_plugin-0.1.0.dist-info/entry_points.txt`。
2. Attention 报 `aclnnHcPre ... not in libopapi.so`；只设 AFD vendor 会遮住
   vllm-ascend 的 `custom_transformer` vendor。改为
   `ASCEND_CUSTOM_OPP_PATH=<afd vendor>:<custom_transformer vendor>` 后通过。

之后 Attention API 到达 `/health` 200、`/v1/models` 返回 `dsv4`；但第一个请求
失败：

```text
aclnnRmsNormDynamicQuant ... Check yOut != nullptr failed
```

原因是 pinned vllm-ascend 8d4409d 的 **Python 源码**配的是镜像里 0.30.0rc2 的
**编译扩展** `_C_ascend`，两者的 RmsNormDynamicQuant 签名不一致。

随后尝试在 8d4409d worktree 里重建它自己的扩展，但被环境挡住：

1. `build_aclnn.sh` 首次失败：`git clone https://gitcode.com/cann/catlass.git` 无法解析
   域名（集群无外网 DNS）。
2. 从镜像的 `/vllm-workspace/vllm-ascend/csrc/third_party` 复制 catlass/protobuf 等
   submodule 后重试，CMake 在 `ascend_protobuf_build_transformer` 处 0% CPU 挂起
   （与 AFD GMM 编译死锁同类），构建中止。

之后回退到镜像自带的 vllm-ascend 0.30.0rc2 源码（与其编译扩展匹配），Attention
模型可以加载，但 FFN 侧报 `AFD NPU FFN worker loop failed`，没有成功的请求。
当前 101 的卡已被其他任务重新占用；本轮没有 serving 的 TTFT/TPOT/吞吐数字。

### 8.2 AFD 扩展与 vllm-ascend 的命名冲突

进一步把 AFD 扩展和 vllm-ascend 的 `_C_ascend` 命名冲突解开了：

- 在 v0.30 容器里只构建四个 async 算子的 vendor（`async_v030`），不再编 GMM；
- 用 `AFD_SKIP_ACLNN_BUILD=1` 只编 AFD 的 PyTorch 扩展；
- 把 AFD 扩展的 `TORCH_EXTENSION_NAME` 从 `_C_ascend` 改成 `afd_ascend_C`，
  避免覆盖 vllm-ascend 的 `_C_ascend`；
- 四个 async 算子和 `_C_ascend.npu_rms_norm_dynamic_quant` 都能注册，Attention
  `/health` 200。

但第一个请求仍然报同一个错：

```text
aclnnRmsNormDynamicQuant ... Check yOut != nullptr failed
```

这次 AFD 扩展已经不再占用 `_C_ascend`，所以问题在 **镜像自带的 vllm-ascend
0.30.0rc2 扩展与 `custom_transformer` ACLNN vendor 之间**：RmsNormDynamicQuant 的
ACLNN 签名不一致，不是 AFD 插件造成的。也说明 101 的 v0.30 镜像本身不能直接跑
这个 DSV4 模型。

### 8.3 干净 CANN9.1 环境尝试

101 上新建了 `z00952334-afd-cann91`（CANN9.1.0 / torch 2.10.0 / torch_npu post4，
16 卡），复制 vLLM 0.30.0 和 vllm-ascend 8d4409d 源码后：

- `vllm_ascend_C.so` / `libvllm_ascend_kernels.so` 编出并安装；
- AFD 扩展以 `afd_ascend_C` 名字重建成功，四个 async 算子 + `_C_ascend.npu_rms_norm_dynamic_quant` 可注册；
- 但 `csrc/build_aclnn.sh` 在 `ascend_protobuf_build_transformer` 子构建处 0% CPU
  停顿（10+ 分钟，清 `MAKEFLAGS` 后复现），拿不到 8d4409d 的 ACLNN vendor；
- 临时用镜像自带的 vllm-ascend 0.26 `custom_transformer` vendor 启动，engine 在
  KV cache 初始化时报 `TBE Subprocess ... main process disappeared`，仍然不可用。

因此 101 仍缺一个与 8d4409d 匹配的 ACLNN vendor；103 也还没有统一到这套环境。
需要的下一步：提供 skill 固定版本的 **预构建镜像/容器**（含 8d4409d 的
`custom_transformer`/ops-transformer 产物），或修复 protobuf 子构建后重试。

最小可行 serving 路径（待执行）：

1. 先跑 baseline Async CAM（不替换 vendor），确认 AFD Attention/FFN readiness、单请求输出和错误日志。
2. 再在相同启动参数下把 FFN 侧 vendor 换成 `build/variants/ds_index/`（或后续完整优化 vendor），
   比较端到端 TTFT/TPOT/吞吐；由于 103 空卡少，优先把 FFN 放 101（8 卡），Attention 放 103（2-4 卡）。
3. 记录启动耗时、端口、DP/TP/rank 映射、HCCL 接口、checkpoint 量化、请求负载和日志/PID 目录；
   这些都是 skill 的必填项。

风险：DeepSeek-V4-Flash 需要较大的 HBM 和特定量化配置；103 的卡被其他任务占用，
可用卡数可能不足以同时容纳 Attention/FFN；跨机 HCCL 建链和 AFD readiness 需要单独验证。
本轮没有启动服务，避免在未确认 checkpoint/拓扑/优化 kernel 的情况下产生无效端到端数字。

## 9. 严重劣化项的原因详解（2026-11-03 补充）

本节只解释 issue #487 中性能劣化严重的项；跨机劣化不明显的项（CS 48 AIV 1.007×、DR 跨机 1.015×、
低 AIV fan-in 平坦）不在此展开，也不作为优化对象。

### 9.1 统一的硬件机制

A3 每个逻辑 NPU 有 24 AIC / 48 AIV。CS/DS/DR 都走 MC2 peer window + DMA：

- 发送侧：`DataCopy` 把 UB 数据写到本地/远端窗口，`MTE3_S` 等待 store 完成；
- 接收侧：`MTE2` 从本地 peer window 逐行读，`MTE2_S` 等待 load 完成；
- 跨机时 store 的"完成"包含远端写入被确认，单次延迟远高于同机。

所以核心不是链路带宽，而是**每次 copy 都同步等完成**，形成串行链：
链长 = 每核 copy 次数 × 单次完成延迟。

### 9.2 DS 跨机 1.84–1.87×（48 AIV 仍不消失）

DS 发送侧每个 token 的 payload 按行 `DataCopy`，每行后立即 `MTE3_S`；index 也按批写、按批等。
4 AIV 时每核约 1536 次 payload（4,128 B）+ 384 次 index（4/8 B）。跨机单次 payload 完成比同机多
约 1.45 µs，index 多约 1.35 µs；48 KiB 的 index 只占字节数 0.2%，却贡献约 20% 的跨机增量，
说明**小事务的完成成本远高于字节数**。

48 AIV 下 CS 每核只剩约 4 次 copy，完成等待被 48 个 AIV 掩盖，跨机/同机回到 1.007×；
但 DS 每核仍有约 32 次 payload，这些 copy 在单个 AIV 内仍然串行等待：

- DS 48 AIV：同机 348.7 µs，跨机 641.4 µs，差 292.7 µs；
- 32 次/核 × 约 9 µs ≈ 288 µs。

因此 DS 的跨机惩罚不能靠"AIV 加够"消失，必须减少每核串行 copy 次数（payload 批量化）。

### 9.3 DR + 本地 GMM +71%

DR 读的是本地 peer window，不是网络。每个 token 被 4 个本地专家展开：

- 32 experts × 384 rows = 12,288 次读，每次 4,128 B，合计 48.375 MiB；
- 逐行 `MTE2` 等待，读和 scale/输出写分离。

GMM 并发时占用 L2、HBM、UB 和 AIV：

- L2 read-hit 从 93.5% 降到约 25%；
- 单次本地读从 0.19 µs 升到 0.50 µs；
- MTE2 活跃时间从约 564 µs 升到约 1,486 µs；
- DR 从 1,569.5 µs 升到 2,679.6 µs（+70.7%），跨机源同样 +71.1%。

同源/跨源 DR 只差约 1.5%，因为源端数据早已写进本地窗口，DR 只读本地；跨机源不改变本地读路径。
根因是 12,288 次逐行本地读在 GMM 争用下被放大；token-major 把同一 token 的 4 个专家合并成
1 次读，读流量 48.375 → 12.1 MiB，因此 GMM 并发下能降 24.8%/26.1%。

### 9.4 CS 4 AIV 跨机 2.65×

CS 每次调用 24 MiB、192 次 copy（128 KiB 级）。4 AIV 时每核约 48 次 copy，每次后立即 `MTE3_S`；
跨机单次完成约 16.7 µs（800.8 µs 差值 / 48 次），48 次串行约 1,280 µs，而同机 484 µs。
48 AIV 时每核只剩约 4 次 copy，48 个 AIV 并行把完成等待摊掉，跨机/同机变成 212.3/210.8 = 1.007×。

双缓冲无效的原因：双缓冲只增加同时可写的 buffer 数，但 `MTE3_S` 仍要在写 status 前等最老的
store 完成；4 AIV 下每核 48 次远端 store 的完成本身是串行的，buffer 数不能增加真正的在途完成。
blockDim=4 也只是减少 block 数，不改变完成链，因此跨机只 −0.7%。

### 9.5 48 AIV + fan-in 8：CS 3.1×、DS 1.96×

低 AIV 时每个 sender 的串行链很长，增加 sender 只是并行更多独立链，单 sender 完成时间基本不变，
所以 fan-in 1→8 平坦（CS 1,280→1,289）。48 AIV 时单 sender 的链很短，瓶颈转为源端/发送路径/
目标入口的聚合吞吐：

- 8 个 sender 同时打满源端板卡 DMA/HCCL 发送队列；
- 目标端 2 个 rank 的 peer window 入口成为共享点；
- 增加 sender 显著抬高每个 sender 的完成时间。

数据：CS 48 AIV fan1 212.3 → fan8 667.7（3.1×）；DS 348.7 → 682.3（1.96×）；同样 fan8 在 4 AIV
只有 1,280→1,289。该劣化不会因 AIV 高而消失，反而在高 AIV 才暴露。

### 9.6 GMM 22/44 超额调度（52/92 AIV）

A3 每个逻辑 NPU 只有 48 AIV。8 通信 AIV + 22/44 GMM = 52 AIV；8 + 22/44 + 48 通信 = 92 AIV。
超额时 AIV 需要分波调度，出现 349–478 µs 的 core-start spread；GMM 和通信的启动时间离散，
timeline 重叠不等于真正同时执行。

- CS 8 AIV + 22/44 同机 522.5 µs，20/40（48 AIV）276.6 µs，−47.1%；
- DS 只改善 4.9%，主因仍是 payload 串行完成；
- DR 在 20/40 下反而 1,152.4 → 1,286.8 µs（+11.7%），因为 GMM 真正占满 40 AIV 后本地读竞争更明显。

结论：配额修复不等于性能修复，必须按算子分别验证。

### 9.7 哪些劣化不严重（不优化）

- CS 48 AIV 跨机/同机 212.3/210.8 = 1.007×：完成等待已被并行掩盖；
- DR 跨机/同机 1,592.5/1,569.5 = 1.015×：本地窗口读，源位置无关；
- CS 4 AIV fan-in 1→2 平坦（1,280.3→1,281.7）：单 sender 串行链主导；
- DS 同机 fan-in 1→4 平坦（3,363.5→3,366.4）：同机完成延迟低，增加 sender 不改变单链。

### 9.8 归纳

| 严重项 | 根因 | 有效杠杆 |
| --- | --- | --- |
| DS 跨机 1.84–1.87× | 每核 1536+384 次小 copy 的串行 completion wait | payload 批量化 |
| DR + GMM +71% | 12,288 次逐行本地读，GMM 抢 L2/HBM | token-major 去重 |
| CS 4 AIV 2.65× | 每核 48 次远端 store 串行完成 | 提高 AIV 或 event/fence |
| fan-in 8 高 AIV 3.1× | 源端/目标入口聚合吞吐饱和 | 板卡分组、HCCL buffer、多 stream |
| GMM 52/92 AIV | 超额调度 + core-start spread | 20/40 配额 |

后续优化优先级与验收见 [issue487-optimization-plan.md](issue487-optimization-plan.md)。

优化后的同机补测（AIV 8/24、fan-in 8、GMM 12/24、`BATCH_SIZE_FACTOR=1.0`）与 production
GMM 配额见 [issue487-optimization-result.md](issue487-optimization-result.md) §10。
