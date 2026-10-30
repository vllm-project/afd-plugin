# Issue #487：A3 双机 Async CAM 算子复现

对应 [vllm-project/afd-plugin#487](https://github.com/vllm-project/afd-plugin/issues/487)
（2026-10-27 更新的正文）。这是 **CS / DS / DR 原生算子的固定输入实验**，
无需模型权重、vLLM server 或请求压测。默认用 10 个逻辑 NPU rank；非活跃发送者仍然
参加 communicator 初始化和 host barrier。不要在同一 NPU 上同时启动多个 rank。

## 可能的原因及证据边界

本地分析基线为 `39017a137d7809bd0ab7cda4a86697e2a6b7a522`。三个 kernel 头文件
统一为 LF 后的 SHA-256 与 issue 完全一致：

| 算子 | SHA-256 |
| --- | --- |
| CS | `e13773885b22d3e7d95336d0cf58b2864a51eb6b5278a9a1d3f83a4599e73892` |
| DS | `8e06441ddade4b54dd0e23dc856893b4d6c4244248f4d11be3b0d44d7acabc9f` |
| DR | `54b036ad0d9449eff63bb040864469284c12e0c20c7392e9752fe2aae4945739` |

这能对应源码，但不能补全 issue 未记录的原始二进制构建 commit。

1. **CS/DS 很可能受到串行远端写完成等待的限制。**
   [CS 的 `CombineSend`](../../../csrc/npu/ascend_kernels/afd_async_combine_send/op_kernel/afd_async_combine_send.h)
   每块 `DataCopy` 后立即 `MTE3_S`；
   [DS 的 `Process` / `FlushExpertTokenOffset`](../../../csrc/npu/ascend_kernels/afd_async_dispatch_send/op_kernel/afd_async_dispatch_send.h)
   对每行 payload、每批索引也等待 MTE3 完成。4 AIV 时，DS 每核 1,536 次 payload
   和 384 次索引写，issue 的单次完成时间差可以解释约 2,742 µs 增量，与详细探针
   观测的 2,753 µs 接近。仅 48 KiB 的索引就贡献约 20% 增量，说明事务数量和
   等待链很重要。CS 增加到 48 AIV 后，同/跨机单发送者时间接近，也符合提高并行度
   隐藏完成延迟的解释。不能把 copy+wait 当作网络 RTT，也不能据此认定链路带宽不足。
2. **ready-input DR 的主要问题更像本地访存竞争。**
   [DR 的 `DispatchRecvRouted`](../../../csrc/npu/ascend_kernels/afd_async_dispatch_recv/op_kernel/afd_async_dispatch_recv.h)
   通过 `GetPeerAddrByRankId(moeRankId_)` 读本地窗口，逐行等待 MTE2，然后分开写数据与
   scale。本例同一 token 被 4 个本地专家展开，12,288 次读取，共 48.375 MiB。
   issue 中加入 GMM 后，每次读取从 0.1917 增至 0.4998 µs；每核 3,072 行估计增加
   946 µs，约占详细探针 payload 阶段增量的 88%。L2 命中率与 MTE2 活跃时间的变化
   支持 cache/访存资源竞争。源主机改变对已就绪 DR 的影响很小，不能把它归为正在
   跨机接收的网络延迟；也还不能精确区分 L2、HBM、互联仲裁或某个硬件队列。
3. **8/48 通信 AIV 加 22 AIC / 44 AIV GMM 是超额调度。**
   总需求分别为 52/92 AIV，而 A3 每个逻辑 NPU 是 48 AIV。issue 记录的数百 µs
   核启动离散和 GMM2 completion 延长，与这个条件一致；20 AIC / 40 AIV + 8 AIV
   对照明显消除了启动离散。流限核是执行资源配置，不等于独占核或访存带宽隔离。
   即使总数等于 48，DR 也仍可能因真正并行后的访存竞争变慢。kernel 时间跨度有重叠
   不能证明所有核同时启动，更不能代表端到端收益。
4. **fan-in 大时还有共享资源效应。**
   低 AIV 下跨机单发送者已慢，增加发送者影响小；48 AIV 下增加发送者使 CS/DS 明显
   变慢。源端板卡、发送路径、目标端入口等都可能共享资源，现有证据没有隔离其中某一项。
   DR 的 fan-in 是预先就绪的 backlog，不是同时入站的网络流量。

值得验证的方向：CS 双缓冲/流水化；DS payload 与索引批量化、增加在途拷贝；DR 合并
读取和复用同一 token、减少逐行串行等待；按 `comm_AIV + 2 * GMM_AIC <= 48` 做可行
配额对照。修改任何 fence 都必须保证 UB 在 DMA 完成前不被重用，且 payload、索引和
计数全部可见后才能发布 ready flag。这些是候选方向，本工具没有修改生产 kernel，
没有把某个优化或硬件队列宣称为已确认根因。

## 环境与文件

- 两台 A3 / 910C，支持 `ascend910_93` 的 MC2 peer-window 路径。issue 的双机位于
  **同一超节点**；任意跨超节点/RoCE 两机不一定支持相同直访方式或有相同性能。
- 严格默认映射要求 S 可使用逻辑卡 `0,1,8,9,10,11,12,13,14,15`，R 可使用 `0,1`；
  固定 DR 接收者的补充对照还要求 R 的 `8..15`。设备 ordinal 可显式修改，但物理板卡
  分组变化会影响 fan-in 曲线，应记录拓扑。不要把逻辑卡 ordinal 当作物理板卡编号。
- 两机 CANN、驱动、torch/torch-npu、原生算子包应一致。issue 是 Python 3.12、
  torch 2.10、torch-npu 2.10.post4、CANN 9.1.0、驱动 25.5.1.1。
- CANN/torch-npu 应支持 `torch.npu.set_stream_limit`、NPUGraph、Gloo 与 HCCL。
  HCCL buffer 固定 512 MiB，DS/DR capacity 固定 16,384 行。
- `run.py`：两端启动、参数检查、CPU dry-run、日志及 CSV 汇总。
- `worker.py`：直接加载 `_C_ascend*.so`，不导入 vLLM，执行与校验真实 NPU 算子。
- `build_cs_helper.py` / `cs_window.cpp`：只供 CS 目标 rank 初始化、清零和读回窗口。
  独立构建目录及独立进程加载；发送 rank 始终用原始 vendor。不要安装或 source
  这个辅助包到通用运行环境。其 `comm_args` 是测试用写回缓冲，不能用于生产调用。

## 构建

两机先进入各自匹配的 Python 环境，source CANN 环境。原始算子若尚未安装，在仓库根目录：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=910c MAX_JOBS=4 AFD_BUILD_JOBS=4 \
  python -m pip install -e . -v --no-build-isolation
```

两机分别构建 CS 辅助包（目录必须不存在，已有目录不会被覆盖）：

```bash
python tools/benchmarks/issue487/build_cs_helper.py --output "$PWD/build/issue487-cs-helper"
```

原始扩展默认从本仓库 `afd_plugin/_C_ascend*.so` 查找。若使用别处已安装的扩展，显式加
`--extension /path/to/_C_ascend.cpython-...so --vendor /path/to/vendors/afd-plugin`。
每个进程启动前设置对应 vendor 的 `ASCEND_CUSTOM_OPP_PATH` 和绝对
`AFD_CUST_OPAPI_LIB_PATH`，不依赖已安装 vLLM-Ascend 的同名库。

## 双机执行

在两机仓库根目录设置相同的通信 IP 和矩阵参数；`IFACE` 是本机实际可用接口，可不同。
IP 必须可用于 HCCL 建链及 Gloo rendezvous；脚本不会修改网络配置。

```bash
S_IP=192.0.2.10       # 替换为发送主机通信地址
R_IP=192.0.2.11       # 替换为第二主机通信地址
IFACE=eth0           # 替换为本机接口
HELPER="$PWD/build/issue487-cs-helper/afd_plugin/_cann_ops_custom/vendors/afd-plugin"
COMMON=(--sender-ip "$S_IP" --receiver-ip "$R_IP" --interface "$IFACE" \
        --cs-helper "$HELPER")
```

先在任意 CPU 主机看默认的 12 个条件和所有 rank/device/主机映射，不启动 NPU：

```bash
python tools/benchmarks/issue487/run.py "${COMMON[@]}" --dry-run
```

然后分别启动以下命令。S/R 输出放在**各自本地目录**；不要让两个 launcher 写同一个共享目录。

```bash
# S 主机
python tools/benchmarks/issue487/run.py "${COMMON[@]}" --node S --output issue487-smoke-S

# R 主机（另一终端）
python tools/benchmarks/issue487/run.py "${COMMON[@]}" --node R --output issue487-smoke-R
```

默认依次执行 CS/DS/DR × same/cross × 4 AIV × fan-in 1 × GMM off/on，仍使用 issue 的
3 rounds、CS 64 次/graph、DS 16 次/graph、DR 每 round 16 个 ready groups，以及
128/256/64 个 GMM pairs。默认不是缩小输入的测试。

有 cross 条件时两端 launcher 必须同时运行；它们使用额外的 TCP 控制连接按条件同步，
避免 R 跳过 same-host 条件后提前进入 cross case 超时。只做 `--placements same`
时仅需在 S 启动。默认控制端口 29499，rendezvous 从 29500 每个条件加一，均可修改。
启动参数摘要不一致会报错。每个条件新建十个进程；失败/超时会终止该 launcher 启动的
子进程并保留日志，另一端通过控制连接或进程组超时退出，不会全机 `pkill`。

完整 144 条件矩阵，在上述两端命令中都追加以下参数，并换新的输出目录：

```bash
--aivs 4,8,48 --fan-ins 1,2,4,8
```

有目的的对照，同样在两端使用一致参数和新目录：

```bash
# 可行预算：8 通信 AIV + 20/40 GMM；与原 22/44 比较
--aivs 8 --fan-ins 1 --gmm on --gmm-aic 20

# DR 固定接收设备 S:0,1；cross-fixed-receiver 时发送者移到 R:8..15
--modes dr --placements same cross-fixed-receiver --aivs 4,8,48 --fan-ins 1,2,4,8

# 延迟主机读取 DR metadata 的提交扰动对照
--modes dr --aivs 8,48 --fan-ins 2,4 --gmm on --readback deferred

# 单独采集 PMU / L2；有 profiler 的结果不替代原始 event 时间
--aivs 4 --fan-ins 1 --profile pipe
--aivs 4 --fan-ins 1 --profile l2
```

## 协议、结果与判读

| 项目 | CS | DS | DR |
| --- | --- | --- | --- |
| 发送 ranks | 2..9；S:8..15 | 0..7；S:8..15 | 同 DS，计时前发布 |
| 目标 ranks | 0..1；same=S:0..1，cross=R:0..1 | 8..9；同设备规则 | 同 DS |
| 固定输入 | BF16 `[3072,4096]`；TP=2 | BF16 `[3072,4096]`；TP=1，top-k=8 | 每次处理一个 ready sender |
| logical payload bytes/call | 写 24 MiB | 写 24.1875 MiB，另有 48 KiB 索引 | 本地读 48.375 MiB，输出/scale 写 48.046875 MiB |
| GMM 位置 | 活跃发送者 | 活跃发送者 | 两个接收者 |

输入、路由、CS 71 项 metadata 与 issue 相同。CS 每 sender 的两个目标区间互不重叠；
DS 每 token 对每目标只传一次，DR 展开成 32 experts × 384 rows。代码保留图的输入、
权重、trace 和输出引用，每轮计时前恢复被 MSD 修改的 GMM int8 输入，并完成设备同步。
graph capture 和 replay 显式使用相应 stream。CS/DS 计时时目标空闲；DR 计时前 DS 完成、
设备同步和全员 barrier，计时期间 DS 空闲。

正确性检查：CS 检查八个状态字和所有活跃目标 payload，计时前清零防止旧结果假通过；
DS 使用真实 DR 检查全部 sender 的 int8 数据、scale=1、专家计数和 source 去重；DR
初始化/预热时校验所有来源的完整输出，计时组中读取 source/metadata 确认无遗漏/重复，
组结束后再校验最后一次输出。默认 `immediate` 在每次 DR 的 event 外读 metadata；
`deferred` 在通信流排入预分配的 metadata 快照，整个 ready group 完成后再主机读取。
这两种提交协议分开标记，不能混合聚合。

每个条件目录包含 `case.json`、`environment-rankN.json`、`rankN.log`、
成功完成测量才有的 `rankN.json`。JSON 保留
逐次 event 时间、source 顺序、版本、设备信息、源码和二进制 SHA-256、实际映射及
`oversubscribed` 标记。源码 hash 是本 checkout 的记录；二进制 hash 才标识实际加载文件，
两者不自动证明该二进制由此 checkout 构建。`profile-rankN/` 保存独立 PMU pass。

把两机结果目录收集到同一机器，汇总为 CSV（该命令不需要 torch）：

```bash
python tools/benchmarks/issue487/run.py \
  --summarize issue487-smoke-S issue487-smoke-R > issue487-smoke.csv
```

汇总要求每条件十个 rank 完整、正确性通过且无重复记录。`mean_rank_us` 等权平均活跃
测量 rank，`rank0_us` 用于对齐 issue 的 **DS fan-in 曲线**；CS fan-in 使用发送 rank
平均。主矩阵一发送者时两种 DS 聚合相同。不要把 profiler pass、deferred 对照和
unprofiled 主结果放进同一条曲线。

复现趋势看：4 AIV、fan-in=1 的 CS/DS 跨机增量；同/跨源 ready DR 的接近程度及本地
GMM 增量；48 AIV fan-in 增大时发送耗时上升；8+44 与 8+40 的启动差异。具体绝对值依赖
运行栈和物理拓扑。本脚本提供原始 event 与可选 Pipe/L2 profile；**没有实现 issue 的
逐核 phase/detail 时间戳探针，也不自动产出其阶段分解、逐次 DMA 时间或 GMM overlap
分类表**。用 profiler 先确认实际 blockDim、两流重叠和背景 GMM 覆盖范围；Pipe 取 CS
中间 48 / DS 中间 12 calls，DR 去掉首尾各两个 samples，再做跨 rank 聚合。不能直接
把 MTE 活跃时间相加当作 kernel elapsed。

## 本次验证范围

在用户指定的无 NPU 编译容器中，使用 Python 3.11、torch 2.8.0+cpu /
torch-npu 2.8.0.post4、CANN 9.0.0，已完成：

- 原始完整 `ascend910_93` ACLNN/kernel 包与隔离 CS 辅助包编译成功。
- 原始 PyTorch 扩展编译、加载成功，五个算子的真实 Meta 调用输出 shape/dtype 通过。
- 144 条件 dry-run 的 rank/设备映射、配额分类、无效参数拒绝、DS 聚合方式、缺失 rank
  拒绝，以及真实 socket 上的计划匹配/不匹配握手检查通过。
- Python 语法、Ruff 检查通过；未修改生产源码。

远程构建及日志保存在容器内 `/home/zhuguangyu/afd_issue487_20261030`，其中
`native-build.log`、`bindings-build.log`、`helper-build.log` 对应上述三项构建。
三个 `.exit` 文件均为 `0`。该环境与 issue 的 9.1.0/2.10 不同，
编译结果不构成双机运行、数值正确性、实际限核或性能复现证据。真正的 PASS 与趋势
必须在目标 A3 双机上运行后取得；本次不提供伪造的设备测量值。
