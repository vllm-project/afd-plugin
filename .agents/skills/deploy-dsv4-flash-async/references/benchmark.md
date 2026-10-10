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

# 对一个已部署 MBT 执行 sweep；RPS 和重复次数按实际需求调整。
bash "${DSV4_SCRIPT_DIR}/run_bench_sweep.sh" \
  --topology afd_dp4tp2 --chunk-size 8192 \
  --rates 2,3,4,5 --repeats 3
```

压测轮数由 RPS 点位数和重复次数决定。改变 MBT 时需更新服务端
`PREFILL_MAX_NUM_BATCHED_TOKENS` 并重新部署，健康检查和实际请求通过后再压测。
`--chunk-size` 仅是压测标签，必须与实际部署一致，不会修改服务配置。

结果路径为 `RESULT_ROOT/TOPOLOGY/chunk_MBT/rps_RPS/repeat_N/result.json`，
包含逐请求 TTFT。保存 Mean、P25/P50/P90/P95/P99 TTFT、实际吞吐、输入/输出
长度和错误。sweep 验证请求数、零失败、总输入/输出 tokens、详细数组和点位
metadata，失败会停止并记录原因；已有结果保留，不能覆盖后挑选最好值。
