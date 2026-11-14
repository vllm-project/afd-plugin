---
title: Async GPU connector 单机与跨节点验证
kind: research
status: draft
primary_code_paths:
  - "csrc/gpu/"
  - "afd_plugin/connectors/gpu/async_gpu.py"
  - "afd_plugin/connectors/gpu/async_gpu_protocol.py"
  - "afd_plugin/connectors/gpu/nvshmem_rt.py"
  - "afd_plugin/connectors/gpu/symm_window.py"
  - "afd_plugin/v1/worker/async_ffn_model_runner.py"
depends_on:
  - "module/connector_contracts.md"
  - "module/ffn_runtime.md"
last_reviewed: 2026-11-14
---

# Async GPU connector 单机与跨节点验证

单机验证记录保留在第 1 节。第 2 节起记录 2026-11-14 实现的跨节点
NVSHMEM 传输、构建方法和两机验收；模型覆盖仍限于明确列出的 checkpoint。

## 1. 验证范围与环境

本次工作基于 AFD `e58fdec5876782b9c6a03683416611deea42af92`，工作分支为
`codex/async-gpu-mrv2`。单机参考实现是上游 `spike/AsyncGPUConnector` 的
`788d17dc17e5fa72638f09e7d16cc5e924f7417d`。
验收对象为该分支工作树，文件摘要保存在本地证据目录的 `qualified-files.sha256`。

| 项目 | 实测身份 |
| --- | --- |
| vLLM 源码目标 | 0.32.0，`5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM 运行产物 | PyPI `vllm==0.32.0` 官方 wheel，内置 commit `g5df63a720` |
| Python / PyTorch | 3.12.15 / 2.13.0+cu130 |
| PyTorch build CUDA / NCCL | 13.0 / 2.29.7 |
| PyTorch C++11 ABI | True |
| NVSHMEM / FlashInfer | 3.4.5 / 0.7.0.post1 |
| GPU / 驱动 | 4 × H20-3e，compute capability 9.0，580.105.08 |
| 连接 | 已分配四卡间 `nvidia-smi topo -m` 均为 NV18 |
| 模型 | DeepSeek-V2-Lite-Chat，BF16，无量化 |
| 模型 snapshot | `85864749cd611b4353ce1decdb286193298f64c7` |
| AFD 拓扑 | Attention DP2/TP1；FFN DP2/TP1/EP2；Attention 计算 gate |
| 执行 | 原生 ModelRunnerV2；图模式 FULL_DECODE_ONLY；编译 mode=0 |

模型使用 GPU 节点已有的 NVMe 缓存；数据集缓存位于用户的节点 scratch。
AFD 当前没有编译的 GPU 扩展。主机编译器版本未单独记录；未来 CUDA/RDMA
扩展的编译、链接和设备初始化需要独立验收。

### 传输和流程

[`tests/gpu/test_async_gpu.py`](../../tests/gpu/test_async_gpu.py) 的五个用例全部
通过，无跳过：双流双阶段 eager、连续图重放、路由权重精度、跨 FFN 抵消精度、
2A2F 独立推进。图用例也覆盖 int32 序号跨越有符号边界。

独立推进用例通过事件让第二个 Attention 暂停发送，要求第一个先完成两个阶段；
随后第二个执行多次改变输入和权重的图重放。每个结果同时检查 routed 和 shared
输出，最后验证所有生产者结束后 FFN 才退出。这里使用真实 CUDA/NVLink，专家
计算是可独立求解的仿射运算；真实专家内核另由模型验证覆盖。

### 真实模型

先在关闭 AFD 插件的原生单卡 TP1 vLLM 上完成真实模型推理。固定七组 token 前缀，
包括长 prefill，随后比较 819 个位置的概率输出。原生按批和逐条执行本身的
最大对数概率差为 0.482730。

验收同时要求：最大对数概率差不超过 0.5；原生 top-1 与 top-2 间隔大于
0.5 的 753 个位置，预测全部一致。图场景额外发送不同长度的 prefill，要求
每个 Attention、每个 FFN 都有真实重放；双阶段场景要求两阶段都有非空工作。

| 模式 | 最大对数概率差 | 高置信预测一致 | Attention 图重放 | FFN 图重放 |
| --- | ---: | ---: | --- | --- |
| eager | 0.484016 | 753/753 | 无 | 无 |
| CUDA Graph | 0.312845 | 753/753 | 80、84 | 8、13 |
| eager + DBO | 0.484016 | 753/753 | 无 | 无 |
| CUDA Graph + DBO | 0.429779 | 753/753 | 79、82 | 4、4 |

双阶段 eager 的两个 Attention 分别记录 51、52 次非空双阶段执行；图加 DBO
分别记录 87、82 次。两个 FFN 均处理了两个阶段及来自两个 Attention 的工作。
这些计数只在服务就绪后的请求窗口内记录。

BF16 生成并非逐字一致：六组自由生成中，完整文本一致数为 4–5 组。原生复跑
也存在后段文本分叉。因此文本一致数不作为数值相等或准确率的替代证明。

### GSM8K 对照

标准 E2E runner 使用同一组前 128 题、8-shot、greedy、最多 512 个生成 token；
strict-match 和 flexible-extract 结果相同。

| 场景 | 拓扑 | 正确数 | 准确率 | 结果 |
| --- | --- | ---: | ---: | --- |
| 原生 baseline-graph | DP4/TP1/EP4 | 80/128 | 62.5% | 通过 |
| GPU async graph + DBO | 2A2F，FFN EP2 | 84/128 | 65.625% | 通过 |

异步场景的标准执行探针确认每个 Attention 都有至少两次真实双阶段图重放。
AFD 评估并发为 12，以触发 DBO；原生 baseline 使用 runner 默认请求并发。
这组结果用于正确性验证，不用于比较吞吐或断言准确率提升，也不是全部 1319 题的评估。

静态检查和全量单测通过。JUnit 记录 1591 个事件，其中 46 个为跳过，失败和
错误均为零；计数包含 subtest。测试完成后无残留 GPU 进程，Slurm allocation
已释放。

本地证据位于 `.venv/validation/`：`transport-qualified.log`、四个模式的
`*-metrics.json` / `*-evidence.json`、`target-unit-final.xml`、
`native-gsm8k-driver.log`、`async-gsm8k-driver.log` 和两个 `gsm8k-*` 结果目录。
单机概率/传输用例显式关闭远程传输；标准 GSM8K runner 的远程传输变量未设置，
数据路径仍是本地 NVLink 映射，不能据此声明两机 RDMA 已验证。

### 本轮实测修正

1. NVSHMEM 清理绑定改用头文件实际导出的 `nvshmemx_hostlib_finalize`。
2. FFN 保留 FP32 路由权重并用 FP32 归并。权重 0.5001/0.4999 转成 BF16 后
   会相同；正负专家输出抵消的 GPU 用例在修复前出现约 17% 误差。
3. Attention 也用 FP32 累加，最后转回 payload dtype。跨 FFN 的正负回复可能
   抵消，在 BF16 中提前加入的小 shared 输出会完全丢失；该回归已由 GPU 用例覆盖。
4. FFN 图的 shared 容量按预期本地 routed 行数计算。对 2F、top-k=6 的均衡
   路由，768 routed 行对应 128 shared 行；原公式只分配 64 行，导致本可重放的
   工作回退 eager。修正后已观察到两个 FFN 的真实图重放。

DeepSeek V3/V4、量化、Attention TP2、其他 GPU 和跨节点均未在本轮验收。
V4 配方仍是候选，不能把 V2-Lite-Chat 的结果推广为 V4 支持声明。

### 重现入口

在同一 Slurm allocation 内使用指定的 vLLM 环境和本地模型；先按
[E2E 指南](../../tests/e2e/README.md)准备模型和 GSM8K 缓存。传输测试运行
`python -m pytest -s -v tests/gpu/test_async_gpu.py`，需要四张已分配 GPU，五项均须通过。

模型验收可直接使用现有 runner，无需临时探针脚本：

```bash
IFS=',' read -r -a AFD_TEST_GPUS <<< "$CUDA_VISIBLE_DEVICES"
AFD_GSM8K_LIMIT=128 OMP_NUM_THREADS=1 HF_ENDPOINT=https://huggingface.co \
python -m tests.e2e.runner \
  --model "$MODEL_PATH" --vllm-bin .venv/bin/vllm --device-backend gpu \
  --scenario afd-v2-graph-dbo-dp2 \
  --attention-devices "${AFD_TEST_GPUS[0]},${AFD_TEST_GPUS[1]}" \
  --ffn-devices "${AFD_TEST_GPUS[2]},${AFD_TEST_GPUS[3]}" \
  --afd-connector GpuAsyncAFDConnector --afd-async --compute-gate-on-attention \
  --api-port-base 18407 --afd-port 6281 \
  --gsm8k-output-path /path/to/results \
  --common-vllm-arg=--max-model-len=4096 \
  --common-vllm-arg=--max-num-batched-tokens=1024 \
  --common-vllm-arg=--max-num-seqs=8 \
  --common-vllm-arg=--gpu-memory-utilization=0.5 \
  --common-vllm-arg=--api-server-count=1 \
  --common-vllm-arg=--trust-remote-code \
  '--common-vllm-arg=--compilation-config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

原生对照的场景为 `baseline-graph`，Attention 使用全部四卡，并由 runner
构建不含 AFD 的原生命令。本轮完整调用记录保存在上述本地证据目录的
`run_accuracy.sh` 和两个 driver log。

## 2. 跨节点实现（2026-11-14）

实现保留 connector API、Attention 路由、MRV2 双阶段调度、FFN 本地专家和
producer-drain 协议，扩展集中于 `symm_window.py`、`nvshmem_rt.py` 和
[`csrc/gpu/`](../../csrc/gpu/README.md)。

### 数据路径和缓冲所有权

初始化后逐 PE 查询 `nvshmem_ptr`。能够映射的 PE 继续直接使用 Torch 写入；
不能映射的 PE 使用本进程对称 heap 中独立的 `(peer, region, ring)` outbox。
先完成 header、索引、FP32 权重及 BF16 routed/shared 字段的打包和类型转换，
再用一个连续 slot 的 `nvshmem_putmem_signal` 发布载荷和 64-bit 通知。
目标参数是本进程的对称 heap 地址加目标 slot offset，另传目标 PE，
不能使用 peer 的映射 VA。普通 Torch 源张量不会直接成为 RDMA 源。

AFD 逻辑 rank 保持 Attention-first。启动时在 AFD store 交换 hostname，
校验每节点 PE 数相同，再将 NVSHMEM PE 排成按节点连续的 block；直接指针查询
和 RMA 目标都通过同一映射转换。混合布局若直接使用 A-first PE，3.4.5 的
内置 same-local-rank team 初始化会停滞；重排后每节点 1A1F 的 2A2F 已通过
64 次双阶段图重放。AFD routing、header 和关闭握手继续使用逻辑 rank。

发送 kernel 最后执行 `nvshmem_quiet`，当前流完成前不得覆盖其 outbox。
两个阶段的缓冲独立；FFN routed/shared 回复继续分开存储，在 Attention
以 FP32 合并后转回 payload dtype。首版按静态 slot 容量发送，包含未使用字段；
尚未优化网络字节量或跨机吞吐。

### 通知、图执行和退出

协议从 v4 升为 v5，flag 扩为对齐的 64-bit signal；bootstrap 校验 header
版本、signal 宽度、扩展可用性以及所有几何参数，不能混用 v4/v5 worker。
CUDA 桥接 kernel 每次重放都在设备上读取 int32 dispatch counter，按与 Torch
相同的规则扩展为 64-bit；FFN 使用变化检测，因而包含 int32 全部回绕。
回复仍使用常量 ready=1，由 Attention 在消费后按流顺序清零。

扩展调用完整的 `nvshmemx_init_attr` 注册设备代码的 NVSHMEM 状态；仅调用
hostlib 初始化不足以使用设备 API。接收通过 NVSHMEM signal wait/test 获得
载荷可见性，host polling 保持在独立 CUDA 流上，不阻塞 FFN compute 流。
没有编译扩展的安装仍可执行同机映射路径；遇到不可映射的 PE 会报告构建要求。

关闭依旧先停止新请求、完成 producer/receiver 握手，排空流和 RMA，再 collective
free/finalize。图对象须在 heap 释放前停止使用。对端退出时采用整组 worker
失败并重启；首版不提供 PE 热替换或网络失败后的局部恢复。

这些选择依据固定版本 3.4.5 的实际头文件和以下官方语义：
[NVSHMEM put-with-signal](https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html)、
[wait/test 可见性](https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html)、
[对称地址与源注册](https://docs.nvidia.com/nvshmem/api/latest/faq.html)。
旧参考分支 `origin/AsyncGPU-CrossNode` 的 `20e0737` 使用 v3 header、共享发送
暂存及主机按值 signal，未直接移植这些部分。

## 3. 构建和两机复现

详见 [GPU 扩展构建说明](../../csrc/gpu/README.md)。默认 CPU/NPU 安装不构建
GPU 扩展；跨机安装显式设置 `AFD_BUILD_GPU_ASYNC_OPS=1`。
编译 CUDA toolkit 必须匹配 NVSHMEM 设备库，编译器的 C++ 运行库必须兼容节点。
本次登录节点默认 CUDA 实为 12.8，不能链接 CUDA 13 设备库；最终产物改在
H20 节点使用 CUDA 13.0/GCC 10.2.1 构建。源码不依赖 Torch C++ headers。

无需模型的测试入口：

```bash
sbatch tests/gpu/run_async_gpu_multinode.slurm
```

脚本申请两节点各两张 GPU，依次运行 eager 1A1F、64 次图重放 1A1F、权重精度
1A1F、跨 FFN 抵消精度 2A2F、Attention 独立推进 2A2F，以及本地/远端混合
2A2F 图重放（每节点 1A1F）。
同一节点的两个任务保持可见该作业分配的全部 GPU，并以 `SLURM_LOCALID`
选取不同的本地设备；全局 PE 序号与本地 CUDA index 分开。
本集群单 GPU mask 的 2A2F NCCL bootstrap 曾报 invalid device ordinal；
上述映射与真实 vLLM 本地多 worker 的设备可见性一致。
`--kill-on-bad-exit=1` 和 step 时限约束测试失败后的清理。
NVSHMEM 3.4.5 要求各节点的 PE 数相等；不均衡的 1A2F（2+1）布局会在
初始化前在 Python 层明确报错，当前不支持这种布局。

真实模型的两机 graph+DBO 配方：

```bash
export ATTENTION_MODEL_PATH=/node-local/DeepSeek-V2-Lite-Chat
export FFN_MODEL_PATH=/node-local/DeepSeek-V2-Lite-Chat
sbatch recipe/gpu/GpuAsyncAFDConnector/deepseek_v2_lite/2node_2a2f_async.slurm
```

第一节点放 2A，第二节点放 2F，AFD rendezvous 使用第一节点的可达 hostname，
`NVSHMEM_REMOTE_TRANSPORT=ibrc`。不要设成 loopback，也不要覆盖 Slurm 分配的
GPU mask。权重使用各节点 NVMe 或共享模型存储，不存入 home；两侧必须使用
同一 snapshot。服务日志记录每个 rank 的 `rma_peers` 和含 outbox 的总 heap 大小。
当前 V2-Lite、token cap 1024、两阶段的 2A2F 每 PE 约 120.9 MiB；
`NVSHMEM_SYMMETRIC_SIZE` 应覆盖完整 receive heap 与 outbox。

## 4. 本轮硬件证据

Slurm 作业 13839，`vllm-sgp-h20-01` 与 `vllm-sgp-h20-03` 各两张 H20。
运行时与第 1 节相同；网络为 NVSHMEM 3.4.5 IBRC / mlx5_bond RoCE。
两节点互相无法直接映射，各 rank 的对侧角色 PE 均进入 RMA 路径。

| 传输检查 | 结果 |
| --- | --- |
| 新扩展下单机五项回归 | 5 passed，零跳过 |
| 两机 eager 1A1F，双阶段与退出握手 | 通过 |
| 两机图 1A1F，64 次输入/权重变化，int32 越界 | 通过 |
| 两机 FP32 路由权重抵消 1A1F | 通过 |
| 两机 BF16 跨 FFN 抵消及 shared 精度 2A2F | 通过 |
| 两机 A1 等 A0 完成后才运行，双阶段加 64 次图重放 | 通过 |
| 两机每节点 1A1F 混合布局，同时使用本地/远端 FFN，64 次图重放 | 通过 |

传输 oracle 使用可独立计算的仿射专家，在真实 GPU 和网络上验证；不替代模型
内核或准确率。checkpoint 与第 1 节一致，节点 03 的四个复制分片已分别核对
源文件 SHA256。

两机真实模型对照使用同一个原生固定前缀基准，比较 819 个位置，并要求
753 个原生高置信位置的 top-1 全部一致：

| 模式 | 最大 logprob 差 | 高置信一致 | Attention / FFN 实际图重放 |
| --- | ---: | ---: | --- |
| eager | 0.484016 | 753/753 | 无 |
| Graph + DBO | 0.429779 | 753/753 | A: 80、84；F: 4、5 |

Graph+DBO 的两个 Attention 均记录到 83 次非空双阶段执行；两个 FFN 都处理了
来自两个 Attention、两个阶段的工作。探针只统计服务就绪后发出的真实请求。
六组自由生成各有 5 组文本一致；BF16 近似计算仍可能在近似并列的 token 上
分叉。eager 服务已通过正常 API 信号退出、producer drain 和 collective close。
首次 Graph+DBO 对照的 launcher 信号直接取消 Slurm step，因此其 teardown
不作为正常清理证据；传输测试和后续 E2E 独立检查退出。

标准 `tests.e2e.multi_pod.runner` 的 `afd-v2-graph-dbo-dp2`，布局
`2A0F,0A2F`，完成 128 题、8-shot、并发 12 的 GSM8K：strict-match 与
flexible-extract 在最终 PE 映射修复后复跑均为 **85/128 = 66.40625%**，
两侧 runner 均报告通过。重排前同配置另一次运行是 87/128；BF16 自回归输出
存在波动，不把这两次分数的差异解释为性能或准确率改进。
本轮另补齐了 multi-pod planner 的 GPU async rendezvous 登记，由 Attention
首 rank 托管 store。正常清理后两节点均无 GPU 进程残留，未触发强制清理。

原生基准沿用第 1 节同环境、同 checkpoint 的 80/128；本轮没有重跑原生模型
或全量 1319 题，也不据此宣称准确率或性能提升。multi-pod runner 当前不收集
跨 pod 的实时图计数；图/DBO 的运行证据来自上面的独立概率对照请求窗口，
不能把准确率通过本身当成图执行证明。

默认 CPU wheel 与可选 GPU wheel 均已构建，102 个 Python 源文件逐字节与
工作树一致；默认 wheel 不含 `.so`，且能在无 Torch/vLLM 环境导入。
GPU wheel 只额外包含 `_C_gpu_async`。打包检查发现已有 NPU patch 目录缺少
`__init__.py`，已补上无副作用的包标记，避免新构建漏掉其中四个兼容模块。
最终全量 CPU 单测记录 1584 个事件（45 个依赖/设备跳过），失败与错误均为零；
包括 PE 重排和不均衡节点的拒绝行为。包标记另由 package/connector 定向测试及
wheel 内容检查覆盖。

实际安装的最终 GPU wheel 已在两机复跑每节点 1A1F 的混合 2A2F、64 次图重放
并通过；本轮也执行过全远端 1A1F 图重放。实际安装的纯
Python wheel 未加载扩展，在同机两卡上复跑 eager、graph、精度三项，3 passed、
零跳过。两种 wheel 最终安装在节点本地的独立目录，并显式检查导入路径，避免
共享目录重装的可见性延迟导致从 checkout 意外导入被测代码。测试默认 Gloo
组使用独立 FileStore，避免多个临时端口探测重复选到同一端口。

本地证据目录为 `.venv/validation-cross-node/`，保留编译日志、单机回归、
两机各场景日志和模型文件校验。第 1 节 `.venv/validation/` 的单机历史证据未覆盖。
IBGDA、多 NIC rail 优化、DeepSeek V3/V4、量化和 Attention TP2 仍未验收。
