# C3 最新结果与 C4 Prefill 启动修正

更新日期：2026-10-13。**本页记录 C3 结果和 C4 首次启动失败的历史。后续 C4 Prefill 0.91 重试及 full 92 项已成功，当前剩余项和手工步骤以[C4 本轮核对结果](DEEPSEEK_V4_AFD_C4_20261013_RESULT_ZH.md)为准，无需再次执行本页的 Prefill 重试。C3 的 full 92 项请求检查通过，但实际五开关全关、部分输出仍有连续 token `0`，正常停服失败，不能关闭默认全开 C3 F0。**

## 1. 两个附件的结论

| 附件 | 已确认 | 尚未完成 |
|---|---|---|
| `48e1a5df943b46b2bc4bc77d7437d0b0.gz` | C3 定向 33 项和 full 92 项检查通过；真实 Graph/U2；批次后 A/Proxy 请求归零 | 五开关实际全 0；9 条响应含连续尾部 `0`；停服失败；golden 未检查 |
| `cf4f76620aea40d8aa3c8c77d0c252b2.gz` | C4 A/F 为 eager/U2；Attention 启用 DSpark，A/F 启动完成 | Prefill 4096 长度的 KV 容量检查失败；未进入 Proxy 请求验证；缺本轮 runtime/config 冻结 |

附件目录名和 `requests-c32-short-ropefix-allon` 的 `allon` 后缀不能代替实际参数。C3 的 `runtime.json` 和六个 A/F rank 的 diagnostics 行都为五开关全 `0`，本轮是关闭对照。

## 2. C3：请求检查通过仍有剩余问题

本轮 run ID 为 `20261013T063530-single-C3-r2`。runtime 记录插件 `f3c8399d80c0736b6d87d39396e1f6f1df609d8d`、vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、Ascend `11ee45653b199a097805b87011824a81ffa51b95`，三个工作树干净；包来源指向预期 checkout。torch/torch-npu 为 2.10.0/2.10.0.post2，CANN 路径 `/usr/local/Ascend/cann-9.2.0`；镜像 digest 和精确 CANN/HCCL build 尚未补齐。

Attention 使用 Flash、PD 开启、DSpark 关闭、`FULL_DECODE_ONLY`、U2；四 rank capture 完成。业务期四 Attention rank 有 `stage_count=2`，两个 FFN Worker 有 `stages=(0,1)`、`graph_capture=False`、`warmup=False`、`target_graph_replay=True`。

- `requests-c32-short-ropefix-allon/request_summary.json`：EOS+32，共 33 项通过；EOS 为 2 tokens/`stop`。
- `requests/request_summary.json`：full 92 项通过，覆盖 1/8/32 并发短/长、边界、SSE、取消及恢复；两次 summary 的四 DP running/waiting 和 Proxy request_num 均归零。
- 两个 summary 都是 `f0_passed=null`、`golden_checked=false`。full 的通过判据包括非空文本、token 数和协议行为，尚不验证生成 token 与 golden 一致。

检查全部响应 IDs 后发现，以下响应虽然文本非空，后半段仍退化为连续 token `0`。表中 DP 来自同一 response ID 的 KV 接收日志，不由客户端序号推测。

| suite | 请求 | 连续尾部 `0` 数 / 64 | Attention DP |
|---|---|---|---|
| 定向 33 | `c32-short-3` | 32 | DP1 |
| full | `c32-short-6` | 11 | DP1 |
| full | `c32-short-19` | 1 | DP1 |
| full | `c32-short-26` | 10 | DP1 |
| full | `c32-long-10` | 37 | DP3 |
| full | `c32-long-11` | 14 | DP3 |
| full | `c32-long-17` | 34 | DP3 |
| full | `c32-long-27` | 49 | DP3 |
| full | `c32-long-30` | 11 | DP3 |

例如 full `c32-long-27` 前 15 个 token 生成可见文字，之后 49 个全为 `0`；定向 `c32-short-3` 前 32 个 token 非零，之后 32 个全为 `0`。参考 tokenizer 中 `0` 为 BOS、默认过滤，故非空检查会通过；仍需用现场 Flash tokenizer 执行[C3 离线全量 token 核对第 3 节](DEEPSEEK_V4_AFD_C3_EMPTY_OUTPUT_REVIEW_ZH.md)。本轮不能宣称原输出退化已消失，且异常不再只限 DP3。没有 logits/逐层数值证据，不能据此断言 NaN 或某个算子的错误。

停服方面，`shutdown-failed.txt` 已记录失败，Attention 最后出现四次 `Segfault encountered`；FFN 的正常 shutdown payload 标记为 0，随后 Worker 死亡及 `KeyboardInterrupt: terminated`。最终 NPU 0–7 无模型进程，API 端口已释放，只能证明最终已清理，不能证明正常退出。附件没有完整操作终端记录，不推定具体强杀命令。

旧版 RoPE 标记使用 `afd_plugin.*` logger 的 INFO；固定 vLLM 的默认日志配置只设置 `vllm` logger 树，插件可能继承 root 的 WARNING，故缺 INFO 标记不能直接证明旧代码仍在运行。本次仅将这一启动标记改为 WARNING，便于下轮核验；RoPE 隔离/刷新逻辑未改变，也未声称它已修复实机输出。

【本地已完成】该日志调整的设计复核确认不修改上游方法、计算/通信路径或全局状态，只增加默认配置下可见的启动记录；沿用既有日志机制，未来统一插件日志配置后可恢复 INFO。现有 worker Graph/RoPE CPU/mock 回归 27 项通过，Ruff/格式、Python 与文档命令语法检查通过；A5 全开输出和正常停服仍须实机验证。

## 3. C4 Prefill 首错及原因

Prefill 在 07:01:17 UTC 的首次错误为：

```text
Available KV cache memory: 1.63 GiB
ValueError: ... max seq len (4096) ... 1.7 GiB KV cache is needed ...
available KV cache memory (1.63 GiB) ... estimated maximum model length is 3908
```

两个 Prefill Worker 都失败，另一 rank 的估计上限为 3912。实际 CLI 为 DSpark checkpoint、P2/TP1、eager、`max_model_len=4096`、`max_num_batched_tokens=4096`、`max_num_seqs=32`、`gpu_memory_utilization=0.90`。权重已加载（每卡约 78.3346 GiB），失败发生在 vLLM 的 `_check_enough_kv_cache_memory`，还未分配正式 KV blocks 或处理业务请求。后面的 Worker 退出、API Engine core initialization failed 和 shared_memory 提示是这次启动失败的后续结果。

固定 Ascend 使用“总显存 × utilization − profile 后的非 KV 使用量”计算 KV 预算。Flash 的 C3 Prefill 在相同 0.90 下仅余 1.79 GiB，DSpark checkpoint 与本轮 profile 使用量略高后变为 1.63 GiB，已低于单条 4096 请求最低需求。`ENABLE_DSPARK=0` 不会改变已选 checkpoint 的 target 权重；本轮 Prefill 的 `speculative_config=None`，并未加载 Attention drafter。

统一指导书第 3.1 节已经要求 `GPU_MEMORY_UTILIZATION=0.91`，但这次启动实际仍为 0.90。旧配置缺少 export、另一个终端没加载或后续赋值覆盖均可能导致这一差异，附件不能区分。recipe 未设置该变量时默认 0.90。历史 C2 成功日志实际为 0.91、Prefill KV 2.54 GiB，可作为调整依据；这轮修正后的具体容量仍须 A5 实测。

## 4. 【A5 手工执行·历史操作】修正 C4 Prefill 启动

**设置位置：启动 Prefill 的终端，加载 site/node 和 C4 配置之后、任何 Prefill nohup 之前。已经运行的进程不会受后续 export 影响。** 保留当前 C4 原日志；先检查失败 Prefill 的 NPU 0/1 没有残留模型进程和 8100/本轮 Prefill KV 端口已释放，不能启动第二份 Prefill。

```bash
npu-smi info
ss -ltnp
ps -eo pid,ppid,args | grep -E '71503|71571|71572|71646|71647|vllm' || true
# ss 不存在时按统一指导书第 7 节做 Python socket bind 检查。
# 确认旧 Prefill 已退出后，只重试 P；A/F 已启动时不再次拉起同一份 A/F。
export GPU_MEMORY_UTILIZATION=0.91
export MAX_MODEL_LEN=4096
export PREFILL_RETRY_LOG="$RUN_ROOT/prefill-retry-$(date +%Y%m%dT%H%M%S).log"
printf 'GPU_MEMORY_UTILIZATION=%s MAX_MODEL_LEN=%s LOG=%s\n' \
  "$GPU_MEMORY_UTILIZATION" "$MAX_MODEL_LEN" "$PREFILL_RETRY_LOG"
nohup bash "$RECIPE/afd_prefill.sh" > "$PREFILL_RETRY_LOG" 2>&1 &
echo $! > "$RUN_ROOT/prefill-retry.pid"
```

此命令适用于已加载环境且 `RUN_ROOT` 指向本次 C4 的终端。保留原 `prefill.log/prefill.pid`，新 PID 单独保存在 `prefill-retry.pid`；本次结束时停 Prefill 必须使用新 PID，不能继续读取旧 PID。当前重试只验证 Prefill 能否启动，不代表整个 C4 已冻结为统一 0.91。

```bash
grep -nE 'non-default args:|Available KV cache memory|GPU KV cache size|Maximum concurrency|Application startup complete|ERROR|ValueError' \
  "$PREFILL_RETRY_LOG"
curl --noproxy '*' -fsS --max-time 10 \
  "http://127.0.0.1:$PREFILL_API_PORT/health"
```

核对实际 CLI 为 0.91、4096；两个 P rank KV 初始化成功，HTTP health 返回成功。按实际模型容量报告判断，不保证调整后固定得到 2.54 GiB。若仍启动失败，保留新日志的首错和 NPU/进程/端口记录，停止扩大流量；不降 4096、不修改模型 config、不绕过 KV 容量检查。

将实际 `SITE_FILE` 中 `export GPU_MEMORY_UTILIZATION=...` **替换为** `export GPU_MEMORY_UTILIZATION=0.91`；缺此行则加在 `export MAX_NUM_SEQS=32` 后。同机各角色终端重新加载同一配置。正式冷启动点位前先核对值再做第 4.4 节 runtime/config 采集，避免采集 null 而子进程默认 0.90：

```bash
grep -nE '^export (GPU_MEMORY_UTILIZATION|MAX_MODEL_LEN|MAX_NUM_SEQS)=' "$SITE_FILE"
printf 'GPU_MEMORY_UTILIZATION=%s MAX_MODEL_LEN=%s\n' \
  "$GPU_MEMORY_UTILIZATION" "$MAX_MODEL_LEN"
test "$GPU_MEMORY_UTILIZATION" = 0.91
test "$MAX_MODEL_LEN" = 4096
```

正式 C1–C5 比较使用相同冻结预算，启动三个角色后还要核对各自 CLI，记录旧轮 0.90 与新轮 0.91 的差异。后续附件已完成 C4 Prefill 重试及 full，本节保留为故障修正入口；不把单角色重试写成 C4 F0，当前剩余项按最新 C4 结果处理。

## 5. 【A5 手工执行】C3 剩余验证入口

归档本轮关闭对照，不覆盖已通过的请求材料。下一轮按[RoPE 修复复测文档第 4 节](DEEPSEEK_V4_AFD_C3_EAGER_U2_ROPE_REVIEW_ZH.md)更新发布提交、写回 `SITE_FILE` 中完整 `AFD_EXPECTED_COMMIT`，使用全新 run ID 冷启动四角色：Flash、PD on、DSpark off、Graph/U2、五开关显式全 `1`，预算统一 0.91。

本次启动标记改为 WARNING 后，核对四 Attention Worker 的 RoPE 标记及六 A/F rank 的实际开关，再执行 EOS+32 的定向 suite。离线扫描全部输出，检查连续 BOS 退化，不仅检查非空；若再现，保存 token 报告和运行时输入/metadata 诊断，停止扩大流量，继续定位。通过后再执行 full、正常停服及完整功能门禁。输出正确性和停服未关闭前，不宣布 C3 F0，也不进入 C5/跨机/性能验收。

下一份回传包含：C4 原 Prefill 和 retry 日志/新 PID/ready 输出；新冷启动点位的 runtime、config hash、site/node、实际各角色 CLI；C3 全量 token 报告、原始响应、四角色日志、metrics 和完整停服终端记录。所有现场命令使用 grep。
