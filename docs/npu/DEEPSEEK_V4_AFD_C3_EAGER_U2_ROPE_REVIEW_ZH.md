# C3 eager/U2 对照结果与 RoPE 隔离修复复测

更新日期：2026-10-13。**本轮 eager/U2 定向 33 请求通过，未再出现 token `0`。已修复一处 CPU 可复现的 DSA RoPE 段间共享问题，尚未证明它消除了 A5 空输出。下一步更新修复提交、冷启动默认五开关全开的 Graph/U2，再执行同一 33 请求；不是进入 C4。**

## 1. 新附件结果

附件 `88aeb0977212430a9ff742b4a8ad69ef.gz` 的目录为 `requests-c32-short-eager-u2-alloff/`。A/F 实际启动于 05:52 UTC，均为 `enforce_eager=True`、`enable_dbo=True`、Graph mode `NONE`；Attention 为 Flash、PD 开启、`speculative_config=None`。四 Attention/两 FFN rank 均为 `stages=2`、五个 Graph 开关全 `0`；recipe 同时将 eager 专用 `eager_u2_stream_overlap` 设为 `1`，故不能把它解读为所有执行/流行为都与 Graph 相同。

| 项目 | 本轮证据 |
|---|---|
| EOS | `Hello`，2 tokens，`stop`，输出 IDs `[19923,1]` |
| 32 短输入 | 32/32 HTTP 200、非空、64 tokens、`length` |
| token 序列 | 全部 32 条均无 ID `0`，此前的全零/63 个零均未复现 |
| Attention DP | 每个 rank 8 个请求；DP3 本轮 8 条都没有零 token |
| 真实 U2 | FFN 在线 `stages=(0,1)`，`graph_capture=False`、`target_graph_replay=False` |
| 批次结束 | 四 DP running/waiting 均为 0，Proxy `status=ok`、`request_num=0` |
| 业务异常 | 四角色日志没有 ERROR/Traceback 或 HTTP 504 |

`request_checks_passed=true`；脚本仍明确 `full_request_suite_passed=false`、`f0_passed=null`、`golden_checked=false`。因此是一次定向请求通过，不能证明 full、精度或正常停服通过。

包内 run ID 仍沿用前一轮 `20261013T045000-single-C3-alloff-diag-r1-c32-short`；实际启动时间、PID 和输出目录均不同，已确认是新的 eager 服务，不将旧 ID 当作“没有切换”的证据。此次仍缺 runtime/source/config 冻结、现场 tokenizer 与退出材料，下轮另建 ID/目录并补采，不能据日志推定实际提交。

## 2. 对照结论与本地定位

| 路径 | 已观察结果 |
|---|---|
| Graph/U2，五开关全开 | DP3 两条全零空输出 |
| Graph/U2，五开关全关 | DP3 一条全零空输出，另一条首 token 非零、随后 63 个零 |
| eager/U2，五开关全关 | 本轮 32 条非空且无零 token |

证据支持优先排查 Graph 执行及其 U2 输入/metadata 生命周期，不能将问题只归为五项新增多流开关。单轮 eager 通过不证明 PD KV 内容在所有情况下都正确，也不证明 Graph 中某个算子产生 NaN；实际源码提交和动态 shape/调度也未完全冻结。

本地直接读取固定 Ascend `11ee45653b199a097805b87011824a81ffa51b95` 的 `rope_dsv4.py` 和 DSA builder：`get_cos_and_sin_dsa(...,use_cache=True)` 的同 config/group decode 缓冲没有 ubatch 区分。CPU 重放用可辨识的测试表及两个不同 position 序列构建 stage 0/1，得到如下确定性结果：stage 0 初始测试值为 `[5,6]`，构建 stage 1 后，**同一地址的 stage 0 值变成 `[10,11]`**。原插件仅隔离长度/ratio metadata，未隔离此 RoPE 缓冲。

这确认了一处可独立复现的 metadata 数据串用。它与 A5 空输出存在关联可能，**尚未证明是全零 token 的唯一根因**，不能提前宣布 C3 修好。

## 3. 【本地已完成】实现与设计复核

修复位于自有 `afd_plugin/v1/worker/npu/dsa_rope.py`、Attention runner 与 ubatch Graph wrapper：

1. 构建每个 stage 的首个 DSA decode metadata 时，快照 RoPE cos/sin；同 stage 的其他 attention group 复用该快照，并将 stage-local ratio cache 中的引用同步到快照。
2. 两个 stage 不再引用同一全局 runtime 缓冲。纯 prefill、使用普通 Tensor 的非 V4 metadata 不进入这一分支，非 U2 继续使用上游原路径。
3. 捕获保留各自的 tensor 地址。每次 replay 前原位刷新 cos/sin，同一共享地址每次只拷贝一次；保留现有长度更新及 replay 前同步，不以替换 tensor 地址更新捕获输入。

没有修改上游源码/接口签名、模型权重、PD KV、采样或请求验收，也没有新增可变全局状态。新增 snapshot/copy 是按 stage 和 RoPE config/group 的小张量拷贝，增加分配/拷贝成本；性能需要 A5 实测，不能声称零开销。直接依赖固定版本 `RopeDataProxy._data` 的结构，上游升级须复核其 ABI；长期应由上游为 ubatch 提供独立 decode RoPE 缓冲后移除此兼容处理。AFD 差异已用 PATCH START/END 标记。

CPU/mock 回归包括共享缓冲被后一段覆盖时前一段数据仍保留、两个连续 replay 更新 cos/sin 且捕获地址不变、pure prefill/普通 Tensor 不走新分支。另在 mock parent context 补齐当前 ABI 的 sinks/input_ids/eplb 字段，不修改运行时 context 行为。worker/请求组 46 项、独立 Proxy 组 4 项均通过，共 50 项；混合联跑在 Proxy 本地临时 socket fixture 失败，独立运行通过，未据此修改服务 Proxy。固定源码的实际 `RopeDataProxy/get_cos_and_sin_dsa` CPU 重放也验证隔离及 replay 原位更新；Ruff/格式检查通过。测试是本地机制验证，不是 A5 验收。

## 4. 【A5 手工执行】更新代码并恢复默认 C3

先归档当前 eager 目录，按统一指导书第 7 节停服并检查清理，保留退出日志；强杀仍记停服失败。按第 3.4 节更新交付分支、核对 dirty state。**这是代码修复，不能仅更新请求脚本或只在请求终端 export；四角色须冷启动。**

```bash
git -C "$AFD_PLUGIN_ROOT" status --short
git -C "$AFD_PLUGIN_ROOT" pull --ff-only origin feat/dsv4-afd-phase1-delivery
export AFD_EXPECTED_COMMIT="$(git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD)"
```

将刚取得的完整提交写回实际 `SITE_FILE` 中原有的 `export AFD_EXPECTED_COMMIT=...` 行。可以执行以下精确替换；替换失败时先核对配置文件，不能追加第二行掩盖旧值：

```bash
"$PYTHON_BIN" - "$SITE_FILE" "$AFD_EXPECTED_COMMIT" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
text, count = re.subn(r"^export AFD_EXPECTED_COMMIT=.*$",
                      f"export AFD_EXPECTED_COMMIT='{sys.argv[2]}'",
                      path.read_text(), flags=re.MULTILINE)
assert count == 1, "应替换且仅替换一行 export AFD_EXPECTED_COMMIT"
path.write_text(text)
PY
```

协调终端加载配置后，只生成一次新 run ID 并创建目录；将相同 ID 原样复制到全部角色和请求终端。其他终端赋值 `RUN_ROOT`，不重复创建目录：

```bash
export CASE=C3 CYCLE=1 TOPOLOGY=single
export RUN_ID="$(date +%Y%m%dT%H%M%S)-single-C3-ropefix-allon-r1"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
printf 'RUN_ID=%s\nRUN_ROOT=%s\n' "$RUN_ID" "$RUN_ROOT"
```

在 **Attention 和 FFN 启动终端**，加载 site/node、拓扑、第 4.3 节配置后，且在第 4.4 节采集和任何 nohup 之前显式设置，避免继承上一轮的五项 `0` 或 eager：

```bash
export MODEL_PATH="$FLASH_MODEL_PATH"
export EXECUTION_MODE=full-decode-only U_BATCHES=2
export ENABLE_PD=1 ENABLE_DSPARK=0
export AFD_HCCL_STAGE_DIAGNOSTICS=1
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=1
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=1
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1
```

各终端使用同一冻结配置。按第 4.4 节保存本轮 runtime、实际三源码提交/dirty state、config hash、site/node 与启动前设备记录；保留 A5 本机 HIXL 绝对路径 `/etc/hixlep`。按第 6.1/6.2 节冷启动 P、F/A、Proxy，F/A 同轮启动；FFN nohup 仍显式 `ENABLE_PD=0 ENABLE_DSPARK=0`。

就绪后用 grep 核对四个 Attention Worker 的新修复标记及六个 A/F rank 的开关：

```bash
grep -nF 'AFD DSA U2 RoPE isolation active:' "$RUN_ROOT/attention.log"
grep -nF 'AFD HCCL diagnostics configuration:' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
```

四 Worker 须有新标记；六 rank 须为 `stages=2`、五开关全 `1`。实际 A/F 为 Graph `FULL_DECODE_ONLY`，Attention DSpark 关闭、A/F capture/ready 完整。值不符或新标记缺失先核对源码/启动环境，不发流量。

## 5. 【A5 手工执行】先 33 请求，再判断是否继续 full

就绪后在请求终端执行，不加 `--standalone`：

```bash
export BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
export C3_DIAG_OUTPUT="$RUN_ROOT/requests-c32-short-ropefix-allon"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" \
  > "$RUN_ROOT/attention-metrics-before-c32.txt"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite c32-short \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --output "$C3_DIAG_OUTPUT" --run-id "$RUN_ID-c32-short"
```

只有 EOS + 32 短请求，无需 token-pool。结束后按[C3 核对文档第 3 节](DEEPSEEK_V4_AFD_C3_EMPTY_OUTPUT_REVIEW_ZH.md)采集后续 metrics/health 并用现场 tokenizer 离线扫描全部 32 条，保存 `output-token-report.json`，不能只看空文本。

若再次出现全零、长段 BOS 或其他异常：保持校验失败，保存新目录的四角色日志、响应与 token 报告，停止扩大流量；此修复尚不足以解决现场故障，继续定位，不能进入 C4/C5。

若 33 请求通过且没有此前退化序列：再按统一指导书第 6.3 节准备本轮模型的 token-pool，在**不同输出目录**执行原 `--suite full`，补齐长输入、边界、SSE/取消恢复；最后核对真实 Graph/U2、逐组 KV 与正常停服门禁。定向成功本身不关闭 C3 F0，C2 正常停服门禁也仍待关闭。正式默认五开关继续全开。
