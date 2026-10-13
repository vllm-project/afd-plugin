# C3 并发 32 空输出核对与定向复测

更新日期：2026-10-13。结论：**C3 full 未通过；两个请求生成计数为 64，但文本为空。当前尚未确定底层生成异常的原因，不关闭 C3，也不进入 C4/C5。**

## 1. 本轮证据

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

## 3. 【A5 手工执行】先采集默认全开下的 token ID

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

默认全开已复现后，正常停止服务并完成清理，创建新的诊断 run ID。在 **Attention 与 FFN 各自启动终端**、加载配置后、启动服务之前显式设置：

```bash
# 仅用于隔离问题的对照；正式 C3 默认仍为五项 1。
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
```

其他模型、PD/HIXL、Graph/U2、DSpark、负载和软件版本保持同一冻结配置，冷启动后核对各 rank 实际值，再执行同一 `c32-short`。这是整体多流策略关闭对照，并非单变量定位；关闭后成功不能代替默认全开的 C3 F0。若两组都失败，继续排查 Graph/U2 基础回放、PD KV 和解码；若仅默认全开失败，再按依赖关系分组缩小开关范围，不能凭一次对照直接认定某个开关有错。

## 5. 设计复核

本次仅修改自有 HTTP 验证客户端和诊断资料，不修改上游、模型、KV 数据面或 Graph 实现。token ID 增加响应归档大小但不改变采样/文本验收；并发批次仍最多 32、每请求超时仍为 300 秒，失败批次等待在途请求完成以保留证据，随后停止新增负载。没有新增可变全局状态；诊断 suite 不替代完整验收。后续 vLLM 升级须复核 chat/completion 的 `return_token_ids` 协议。
