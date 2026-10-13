# C3 并发 32 空输出核对与定向复测

更新日期：2026-10-13。结论：**默认五开关全开的 c32 定向复测仍失败，两个 DP3 请求各返回 64 个 token ID `0`、空文本。下一步执行第 4 节五开关全关的冷启动对照；底层原因仍未确定，C3 未通过，不进入 C4/C5。**

## 1. 本轮证据

### 1.1 最新 c32 定向复测：已取得 token ID

附件 `b68d20f6771c4f5d8a0e5e6a843f20dc.gz` 的新结果位于 `requests-c32-20261013T034300/`，不能用同包旧 `requests/` 目录替代。本轮 EOS 返回 `Hello`、2 tokens、`stop`；32 短请求全部收到 HTTP 200，并完整归档，30 条非空、2 条空文本。四个 DP 各处理 8 个请求，故障仍集中在 DP3。

| 请求 | response ID | 输出 token ID | Attention rank / KV 接收 |
|---|---|---|---|
| `c32-short-3` | `cmpl-d247811d-91f2-43aa-886f-5a19b1ecc3bf` | 64 个 `0`，`length`，空文本 | DP3，PID 54052；0.91 ms |
| `c32-short-11` | `cmpl-b015179a-0495-4a76-b4e3-6bfd58e53a31` | 64 个 `0`，`length`，空文本 | DP3，PID 54052；207.26 ms |

这次与上一轮失败编号不同，编号只是批次内从 0 开始的请求序号，并非错误码。新批次没有旧批次提前退出产生的两条 ReadError；收齐证据的客户端修改已生效，服务端空输出仍复现。30 条非空只满足该项检查，尚不能证明模型精度正确。

本地参考模型 `DeepSeek-V4-Flash-w8a8-mtp` 的 `tokenizer.json` 与 config 确认 `0` 为 BOS `<｜begin▁of▁sentence｜>`、`1` 为 EOS。离线解码 64 个 `0`，`skip_special_tokens=True` 得到空字符串；关闭过滤则显示重复 BOS。**该结果解释了空文本的形成方式，不能解释为何服务返回全零 token，也不能据此断言 logits 为 NaN、KV 内容错误或某个 Graph 开关有错。** A5 使用 `/home/models/DeepSeek-V4-Flash`，附件未包含其 tokenizer 文件；仍需用第 3 节离线命令确认现场 token 含义，不需要再发请求。EOS 自然停止与这两条 `length` 截断是不同结果，不能将空响应判为 EOS 成功。

实际启动日志为 03:39–03:41 UTC，Attention 四 rank 和 FFN 两 rank 的配置行均为 `stages=2`、五开关全 `1`；Attention 使用 `FULL_DECODE_ONLY`、capture sizes `[1,2,4,8]`、DSpark 关闭，A5 HIXL 路径为 `/etc/hixlep`。故障请求均有 KV 接收完成记录，但这不能验证 KV 内容正确。四角色日志未出现业务 ERROR/Traceback；后续 Attention/P 请求数回到 0、FFN 继续推进。

包内 `runtime.json`、`site.env` 和 run ID 仍沿用上一轮记录（预期/记录提交 `138202b…`，run ID 时间为 01:33），不能代表本次新启动进程的源码冻结证明。下一轮另建目录并重新执行指导书第 4.4 节采集；不要覆盖当前日志。此次没有正常停服证据，C2 的停服门禁也仍未关闭。

### 1.2 上一轮 full：当时没有 token ID

附件 `cbd77d8f02964963aa7fcdcfb962df95.gz` 中，`runtime.json` 与 `site.env` 的实际/预期插件提交均为 `138202b3797f932e5ea31650413d118b254b0588`，vLLM 为 `0fc695f…`、Ascend 为 `11ee456…`，Python 3.12、torch/torch-npu 2.10、CANN 本机目录 `/usr/local/Ascend/cann-9.2.0`。
C3 是 Flash 模型、PD + AFD、Graph `FULL_DECODE_ONLY`、U2、DSpark 关闭；五个 Graph/U2 开关均为 1。镜像 digest 和精确 CANN/HCCL build 在附件中仍为空。

| 请求阶段 | 结果 |
|---|---|
| EOS | `Hello`，2 tokens，`stop` |
| 并发 1 短/长、并发 8 短/长 | 18 个请求均返回非空文本、64 tokens、`length` |
| 并发 32 短输入 | 30 个响应归档：28 个非空、2 个空文本；另外 2 个有客户端 ReadError 记录 |
| 后续阶段 | 32 长输入、边界、SSE、取消恢复未执行 |

两个空响应分别为：

| 请求 | response ID | HTTP | tokens / finish | Attention rank |
|---|---|---|---|---|
| `c32-short-16` | `cmpl-073f57e7-d701-4627-91e7-87c9981f4d00` | 200 | 64 / length | DP3，PID 50885 |
| `c32-short-29` | `cmpl-ce3c2446-c2ea-4ce6-9b9c-7745eb044beb` | 200 | 64 / length | DP3，PID 50885 |

两者 `choices[0].text=""`、`token_ids=null`。Attention 日志分别记录 KV 接收成功，耗时 2.14 ms、0.92 ms。本轮业务期没有 504 或角色 ERROR/Traceback，FFN 有真实 `stages=(0, 1)`、`target_graph_replay=True` 记录。因此应从生成、detokenization、Graph/U2 及 PD 与回放的交互排查；不能继续将此故障归为此前 Proxy 环境代理的 504。

两个 ReadError 属于 `c32-short-5`、`c32-short-22`。时间与第一次校验异常后的客户端退出/Proxy cancelled 日志一致，优先视为批次提前结束的伴随现象，不当成已证实的服务超时根因。

原始响应没有 token ID，无法区分特殊 token 被过滤、无效生成序列或其他解码异常；现有日志也不能证明某一个多流开关造成了空输出。**不以“64 tokens”或 HTTP 200 替代非空输出校验，不将猜测写成 Graph 修复。**

## 2. 本地已完成的诊断改进

`a5_1030_requests.py` 的非流式请求现在发送 `return_token_ids=true`；保持原温度、EOS/截断参数和默认特殊 token 过滤，不通过显示特殊 token 将空输出改判成功。
失败另存 `<name>.validation.error.json`，包含文本是否为空、token 数、finish reason、response ID 和 token ID。

并发批次使用 `gather(return_exceptions=True)` 收齐同批结果，写 `batch-c32-short.summary.json` 等逐批汇总，随后判失败并停止后续负载。这样首个空输出不会提前取消其余请求，不会为了继续测试把失败忽略。
新增 `--suite c32-short`：**1 个 EOS + 32 个短请求，共 33 个，不需要 token-pool**；不重发并发 1/8、长输入、边界或 SSE。即使这 33 个都成功，`full_request_suite_passed=false`、`f0_passed=null`。

相关 CPU/mock 回归 23 项通过，包括两条空响应与两条延后响应同批返回的复现；全部 32 条证据保留，批次仍失败。这里修正了证据采集方式，尚未修复或验证 A5 的空输出原因。

## 3. 【A5 手工执行】默认全开 token 采集（本轮已完成请求，保留离线确认）

最新附件已完成下面的默认全开定向请求，**无需重复这一批**。若服务仍在同一终端中，可直接把 `C3_DIAG_OUTPUT` 指向已完成的目录，然后仅执行本节的离线 tokenizer 段：

```bash
export C3_DIAG_OUTPUT="$RUN_ROOT/requests-c32-20261013T034300"
test -f "$C3_DIAG_OUTPUT/batch-c32-short.summary.json"
```

以下启动/请求步骤留作后续复测参考；本轮下一步按第 4 节执行。

先保留本轮日志，按统一指导书更新交付分支、写回实际 `SITE_FILE` 中的完整 `AFD_EXPECTED_COMMIT`，重新冻结 runtime/config hash。正式本轮源码是 `138202b…`，若从远端更新，将包含随后提交的 Proxy 内部直连及此次请求诊断改进；不能继续使用旧脚本执行新 suite。

按指导书第 6 节以新 run ID 冷启动 C3 四角色，配置仍为 `full-decode-only/U2`、`ENABLE_DSPARK=0`、五开关全 1，保留 A5 本机 HIXL 绝对路径。A/F 同轮启动，确认 ready 后启动 Proxy；现场检索均使用 grep。
就绪后在请求终端运行：

```bash
export BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
export C3_DIAG_OUTPUT="$RUN_ROOT/requests-c32-short-$(date +%Y%m%dT%H%M%S)"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" \
  > "$RUN_ROOT/attention-metrics-before-c32.txt"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite c32-short \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --output "$C3_DIAG_OUTPUT" --run-id "$RUN_ID-c32-short"
```

发生校验异常时仍采集后续指标（若终端有 `set -e` 导致命令停止，在原终端单独执行以下段落），不要立即停服覆盖日志：

```bash
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" \
  > "$RUN_ROOT/attention-metrics-after-c32.txt"
curl --noproxy '*' -fsS "$BASE_URL/healthcheck" \
  > "$RUN_ROOT/proxy-health-after-c32.json"
grep -nEi 'KV cache transfer|step progress|send_dp_metadata|ERROR|Traceback' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log" \
  > "$RUN_ROOT/c32-evidence-candidates.txt" || true
```

在已安装现场 tokenizer 的 Python 环境离线解析空输出 token，不增加请求、不改变服务采样：

```bash
"$PYTHON_BIN" - "$C3_DIAG_OUTPUT" "$MODEL_PATH" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
from vllm.tokenizers import get_tokenizer

root = Path(sys.argv[1])
tokenizer = get_tokenizer(sys.argv[2], tokenizer_mode="deepseek_v4")
report = []
for path in sorted(root.glob("c32-short-*.json")):
    data = json.loads(path.read_text())
    if not data.get("choices"):
        continue
    choice = data["choices"][0]
    if choice.get("text"):
        continue
    ids = choice.get("token_ids")
    row = {"file": path.name, "response_id": data.get("id"),
           "usage": data.get("usage"), "token_ids": ids}
    if ids is not None:
        row.update(token_counts=dict(Counter(ids)),
                   decoded_keep_special=tokenizer.decode(ids, skip_special_tokens=False),
                   decoded_skip_special=tokenizer.decode(ids, skip_special_tokens=True))
    report.append(row)
(root / "empty-output-token-report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
```

同机使用上述 loopback；跨机客户端必须换为可达的对端地址。保留 `request_summary.json`、逐批 summary、32 条原始响应/错误文件、token 报告、metrics 与四角色日志。若定向测试不复现，不能宣布修复；再按原 full 的顺序采集相同 token 证据，保留失败前的 1/8 负载历史。

## 4. 【A5 手工执行】复现后做关闭对照

默认全开已经复现。先归档旧目录，按指导书第 7 节停止服务、保留退出日志并检查设备/端口清理结果；若再次需要强杀，记录为停服失败，不能算正常退出通过。

在协调终端加载环境后生成本轮 ID，仅生成一次；将打印的 ID 原样复制到 Prefill、Attention、FFN、Proxy/请求终端：

```bash
export CASE=C3 CYCLE=1 TOPOLOGY=single
export RUN_ID="$(date +%Y%m%dT%H%M%S)-single-C3-alloff-diag-r1"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
printf 'RUN_ID=%s\nRUN_ROOT=%s\n' "$RUN_ID" "$RUN_ROOT"
```

其他终端加载相同环境/拓扑、赋值同一 `RUN_ID` 与 `RUN_ROOT`，不要再次创建目录。按指导书第 4.3 节设置 C3：`EXECUTION_MODE=full-decode-only`、`U_BATCHES=2`、Attention 的 `ENABLE_PD=1`、`ENABLE_DSPARK=0`、Flash 模型。**在加载 site/node 配置和第 4.3 节配置之后、采集 runtime 与执行任何 nohup 之前**执行以下段落。各终端记录一致，尤其 Attention、FFN 启动终端都必须设置；在请求终端设置不会改变已启动服务：

```bash
# 仅用于隔离问题的对照；正式 C3 默认仍为五项 1。
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
```

其他模型、PD/HIXL、Graph/U2、DSpark、负载和软件版本保持与本次默认全开一致。按指导书第 4.4 节重新保存实际提交、dirty state、runtime/config hash，核对 `SITE_FILE` 的 `AFD_EXPECTED_COMMIT` 等于实际插件 HEAD；保留 A5 本机绝对路径 `/etc/hixlep`。此轮诊断无需为了文档更新改变运行代码版本。

随后按第 6.1/6.2 节冷启动四角色：Prefill，FFN 与 Attention 同轮启动（不等待 FFN ready 才启动 Attention），全部 ready 后启动 Proxy。FFN 启动保持 `ENABLE_PD=0 ENABLE_DSPARK=0`。就绪后，在同机协调终端检查：

```bash
grep -nF 'AFD HCCL diagnostics configuration:' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
```

要求四个 Attention rank、两个 FFN rank 的 `stages=2`，五个 `graph_u2_*` 值全部为 `0`；否则先修正启动环境，不发请求。不要把 Graph 本身改成 eager，也不要加 `--standalone`。确认后在请求终端只运行：

```bash
export BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
export C3_DIAG_OUTPUT="$RUN_ROOT/requests-c32-short-alloff"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite c32-short \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --output "$C3_DIAG_OUTPUT" --run-id "$RUN_ID-c32-short"
```

仍为 EOS + 32 短请求，共 33 个；不需要 token-pool，不先重跑 full。无论成功失败，保留新目录的 runtime/site/node/config hash、四角色日志、32 条响应及 batch/request summary、metrics、离线 token 报告和退出日志（可按第 3 节采集段保存指标）。回传时归档此新 `RUN_ROOT`，不要混入旧 `requests/`。

这是整体多流策略关闭对照，并非单变量定位；关闭后成功不能代替默认全开的 C3 F0。若两组都失败，继续排查 Graph/U2 基础回放、PD KV 和解码；若仅默认全开失败，再按依赖关系分组缩小开关范围，不能凭一次对照直接认定某个开关有错。正式验证的五开关默认仍为全 `1`。

## 5. 设计复核

本次仅修改自有 HTTP 验证客户端和诊断资料，不修改上游、模型、KV 数据面或 Graph 实现。token ID 增加响应归档大小但不改变采样/文本验收；并发批次仍最多 32、每请求超时仍为 300 秒，失败批次等待在途请求完成以保留证据，随后停止新增负载。没有新增可变全局状态；诊断 suite 不替代完整验收。后续 vLLM 升级须复核 chat/completion 的 `return_token_ids` 协议。
