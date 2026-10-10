# 固定负载压测与 MBT 对比

仅在用户要求性能测试时加载本参考。`run_bench.sh` 执行一个 RPS/repeat，
`run_bench_sweep.sh` 对一个**已经部署**的 MBT 循环执行 RPS 和 repeat。
两者使用 `vllm bench serve`，直接请求 prefill API，输出一个 token。

## 内置数据集

skill 的 `assets/workloads/formal_0_1_2_vllm_bench.jsonl.gz` 保存当前
MoonConv/WildChat 的 formal_0/1/2 合并负载。解压后的字节和请求顺序保持不变：

| 属性 | 值 |
| --- | --- |
| 请求数 | 1536 |
| 总输入 tokens | 15,803,063 |
| 平均输入 tokens | 10,288.45 |
| 最大输入 tokens | 63,778 |
| 每请求输出 tokens | 1 |
| SHA256 | `1ebccbd149bc8f28568d3d5eced3911d2a9473fb015bdb829b898a26abf63d08` |

这是 vLLM custom JSONL，包含 `prompt`、`output_tokens` 和工作负载 metadata。
它是固定的文本适配文件，不是 random 数据，也不是按原 trace 时间戳回放的
token-ID 请求。开放负载的到达速率由 `--request-rate` 控制。

来源、冻结版本及压缩包哈希见 `assets/workloads/manifest.json`；数据库许可证
和第三方署名见同目录的 `LICENSE.md`、`THIRD_PARTY_NOTICES.md`。

不设置 `DSV4_BENCH_DATASET_PATH` 时，两个入口会自动解压并验证内置数据，
缓存到 `${REPO_ROOT}/bench_results/dsv4-flash/datasets/`。
可用 `DSV4_BENCH_DATASET_CACHE_DIR` 改变缓存目录。只解压、不压测时：

```bash
python3 "${DSV4_SCRIPT_DIR}/prepare_bench_dataset.py" \
  --output /absolute/cache/formal_0_1_2_vllm_bench.jsonl
```

已有缓存会重新校验；哈希错误的文件不会被静默覆盖。自备 custom 数据时同时
设置 `DSV4_BENCH_DATASET_PATH` 和真实 `DSV4_BENCH_DATASET_SHA256`。
单次入口可使用其他负载；sweep 的验收绑定上述固定 1536 请求契约，换用不同
规模/长度的数据时不能沿用该 sweep 的成功结论。

## 单次与单 MBT sweep

按环境参考提供 `DSV4_MODEL`、Python/CLI、源码路径和网络设置；服务需已健康。
默认每点预热 16 次，顺序固定，不 shuffle、不 oversample。默认不设置客户端
并发上限；`DSV4_BENCH_MAX_CONCURRENCY=0` 或空值均表示开放负载。

```bash
export DSV4_BENCH_HOST="${P_NODE_IP}"
export DSV4_BENCH_RESULT_ROOT=/absolute/results/dsv4-mbt

# 一个点；示例标签必须与实际部署相同。
bash "${DSV4_SCRIPT_DIR}/run_bench.sh" \
  --topology afd_dp4tp2 --chunk-size 8192 \
  --request-rate 2 --repeat 1

# 一个已部署 MBT 的 12 轮：4 个 RPS × 3 次重复。
bash "${DSV4_SCRIPT_DIR}/run_bench_sweep.sh" \
  --topology afd_dp4tp2 --chunk-size 8192 \
  --rates 2,3,4,5 --repeats 3
```

结果路径为 `RESULT_ROOT/TOPOLOGY/chunk_MBT/rps_RPS/repeat_N/result.json`，
包含逐请求 TTFT。保存 Mean、P25/P50/P90/P95/P99 TTFT、实际吞吐、输入/输出
长度和错误。sweep 验证请求数、零失败、总输入/输出 tokens、详细数组和点位
metadata，失败会停止并记录原因；已有结果保留，不能覆盖后挑选最好值。

## 五档 MBT/RPS 矩阵

当前矩阵为 MBT `8192/16384/32768/49152/65536`、RPS `2/3/4/5`，每点
3 次重复，共 60 轮。48k 为 **49152**。每轮 1536 请求，共 92,160 个正式请求，
另有预热和 benchmark ready probe。请求开放到达，服务端可以出现排队；同时
报告实际吞吐，不能把 offered RPS 当作 achieved RPS。

先按 [AFD 部署](afd-prefill.md) 或 [非 AFD 部署](baseline.md) 选择已有拓扑，
在对应节点启动服务，再将同一个拓扑名称传给压测入口。按实际设备容量设置
上下文、并发序列数、显存比例和通信窗口；最大上下文需覆盖 63,778-token 输入
及一个输出 token。压测客户端的模型和 tokenizer 必须与部署匹配。

AFD 脚本关闭 prefix cache，请求共享 compressor workspace，开启 2-stage token
ubatch。FFN 图开关和 layered 开关只传给 FFN；Attention 的 workspace 必须从
实际日志确认启用，而非只检查配置值。FFN 使用的图模式应随启动命令和验收日志保存。

**FULL 的运行时前提：**所用 AFD checkout 必须包含 Async CAM FFN 的专用
graph transaction 和多 DP 启动协调，例如 `capture_async_cam_ffn_graph`。
本 skill 只提供脚本和参数，不向发布分支添加该模型运行时能力。部署前核对
实际源码；FULL 验收要求所选拓扑的全部 FFN rank 出现 layered、通信预热、
capture complete 和 replay。启动或单 token smoke 通过不能替代
1536 请求压测，也不能作为模型精度验收。压测 metadata 不证明实际图执行。

**改变 MBT 必须重新部署：**完成一个 MBT 的 12 轮后，停止本次 A/F 服务并
确认 worker/端口/设备释放，更新两端 `PREFILL_MAX_NUM_BATCHED_TOKENS`，使用
新的 LOG_DIR/PID_DIR 重启，健康和实际请求通过后再执行对应 sweep。
`--chunk-size` 仅是压测标签，不会修改服务。每档的容量和窗口都要实际检查，
不能把 8k 启动成功推断成 64k 已验证。

分摊矩阵时按所选拓扑计算所需资源，确定不重叠的设备，并隔离 AFD rendezvous、
DP RPC、API 端口及结果目录。按 MBT/RPS 分别报告三个重复的中位数及范围，
不合并不同输入预算或速率的样本。
