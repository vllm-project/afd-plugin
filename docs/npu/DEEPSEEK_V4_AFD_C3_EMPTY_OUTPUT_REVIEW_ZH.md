# C3 并发 32 空输出核对与定向复测

更新日期：2026-10-13。**最新附件 `48e1a5df943b46b2bc4bc77d7437d0b0.gz` 的 Graph/U2 关闭对照在 `f3c8399…` 上完成定向 33 项/full 92 项请求检查，但仍有非空文字后连续 token `0`，且正常停服失败；五开关实际全 `0`，不能关闭默认全开 C3。当前操作见[C3/C4 最新结果与 Prefill 修正](DEEPSEEK_V4_AFD_C3_C4_20261013_REVIEW_ZH.md)。以下保留此前空输出、全关和 eager 对照历史及离线 token 核对命令，不以旧步骤代替最新执行入口。**

## 1. 本轮证据

### 1.1 此前 eager/U2：定向请求通过，修复待验证

附件 `88aeb0977212430a9ff742b4a8ad69ef.gz`：实际 eager/U2/PD，DSpark 关闭、五 Graph 开关全关；EOS 成功，32 短响应全部非空、64 tokens、`length`，全部无 ID `0`，四 DP 各 8 条。FFN 在线两 stage 且未 replay，批次结束 A/Proxy 请求归零。源码/runtime 冻结和退出材料仍缺，run ID 复用了上一轮，不能据此关闭 F0。详见[新结果、确定性 RoPE 复现及默认全开复测步骤](DEEPSEEK_V4_AFD_C3_EAGER_U2_ROPE_REVIEW_ZH.md)。

### 1.2 五开关全关对照：仍有异常

附件 `aab420537ef043ff9d6230165620a687.gz` 的新目录为 `requests-c32-short-alloff/`，run ID 为 `20261013T045000-single-C3-alloff-diag-r1-c32-short`。实际启动于 04:51 UTC；Attention 四 rank 和 FFN 两 rank 均记录 `stages=2`、五个 `graph_u2_*` 值为 `0`。Attention 仍是 `FULL_DECODE_ONLY`、U2、Flash、DSpark 关闭、PD 开启；本轮不是误用旧的全开服务。

EOS 返回 `Hello`、2 tokens、`stop`。32 短请求全部 HTTP 200、64 tokens、`length` 并完整归档，脚本按非空检查报告 31 条通过、1 条失败；四 DP 分配为 9/8/8/7。检查全部输出 token 后发现两条异常，均在 DP3、PID 57661：

| 请求 | response ID | 文本 / token ID | KV 接收 |
|---|---|---|---|
| `c32-short-19` | `cmpl-edc656ff-7419-4137-84d2-4e02ac78e7b2` | 空文本；64 个 `0` | 完成，0.96 ms |
| `c32-short-3` | `cmpl-05146075-6ba2-4c08-83cd-40a88446addd` | `" Use"`；`[6948] + 63 个 0` | 完成 |

因此 **31 条非空不等于 31 条生成正确**；只显示特殊 token 或只检查非空都不能修复底层生成。失败数量由 2 变为 1，不足以说明关闭策略改善了问题，因为新旧批次的 DP 分配和动态 shape 不完全一致。全开与全关均复现，不能把原因只归为开启这五项新增多流策略；Graph/U2 基础执行、padding/metadata、PD KV 内容及模型数值仍待定位。现有日志没有 logits/逐层数值，不能直接断言 NaN 或某个算子错误。

四角色业务日志无 ERROR/Traceback，Attention/P 请求数随后归零，FFN 继续推进。本包未含 `runtime.json`、site/node 配置、config hash、metrics、现场 tokenizer 报告及停服证据；可确认启动参数/开关和响应，不能确认这次实际源码提交、现场 token 映射或正常退出门禁。下一轮补采这些材料，不为补旧记录而重复本轮负载。

### 1.3 默认全开 c32：已取得 token ID

附件 `b68d20f6771c4f5d8a0e5e6a843f20dc.gz` 的新结果位于 `requests-c32-20261013T034300/`，不能用同包旧 `requests/` 目录替代。本轮 EOS 返回 `Hello`、2 tokens、`stop`；32 短请求全部收到 HTTP 200，并完整归档，30 条非空、2 条空文本。四个 DP 各处理 8 个请求，故障仍集中在 DP3。

| 请求 | response ID | 输出 token ID | Attention rank / KV 接收 |
|---|---|---|---|
| `c32-short-3` | `cmpl-d247811d-91f2-43aa-886f-5a19b1ecc3bf` | 64 个 `0`，`length`，空文本 | DP3，PID 54052；0.91 ms |
| `c32-short-11` | `cmpl-b015179a-0495-4a76-b4e3-6bfd58e53a31` | 64 个 `0`，`length`，空文本 | DP3，PID 54052；207.26 ms |

这次与上一轮失败编号不同，编号只是批次内从 0 开始的请求序号，并非错误码。新批次没有旧批次提前退出产生的两条 ReadError；收齐证据的客户端修改已生效，服务端空输出仍复现。30 条非空只满足该项检查，尚不能证明模型精度正确。

本地参考模型 `DeepSeek-V4-Flash-w8a8-mtp` 的 `tokenizer.json` 与 config 确认 `0` 为 BOS `<｜begin▁of▁sentence｜>`、`1` 为 EOS。离线解码 64 个 `0`，`skip_special_tokens=True` 得到空字符串；关闭过滤则显示重复 BOS。**该结果解释了空文本的形成方式，不能解释为何服务返回全零 token，也不能据此断言 logits 为 NaN、KV 内容错误或某个 Graph 开关有错。** A5 使用 `/home/models/DeepSeek-V4-Flash`，附件未包含其 tokenizer 文件；仍需用第 3 节离线命令确认现场 token 含义，不需要再发请求。EOS 自然停止与这两条 `length` 截断是不同结果，不能将空响应判为 EOS 成功。

实际启动日志为 03:39–03:41 UTC，Attention 四 rank 和 FFN 两 rank 的配置行均为 `stages=2`、五开关全 `1`；Attention 使用 `FULL_DECODE_ONLY`、capture sizes `[1,2,4,8]`、DSpark 关闭，A5 HIXL 路径为 `/etc/hixlep`。故障请求均有 KV 接收完成记录，但这不能验证 KV 内容正确。四角色日志未出现业务 ERROR/Traceback；后续 Attention/P 请求数回到 0、FFN 继续推进。

包内 `runtime.json`、`site.env` 和 run ID 仍沿用上一轮记录（预期/记录提交 `138202b…`，run ID 时间为 01:33），不能代表本次新启动进程的源码冻结证明。下一轮另建目录并重新执行指导书第 4.4 节采集；不要覆盖当前日志。此次没有正常停服证据，C2 的停服门禁也仍未关闭。

### 1.4 更早的 full：当时没有 token ID

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

## 3. 【A5 手工执行】token 离线确认与复测通用采集

默认全开、全关与 eager/U2 定向请求均已完成，**无需重复这两批**。若环境仍加载，可把 `C3_DIAG_OUTPUT` 指向最新 eager 目录，然后仅执行本节离线 tokenizer 段；它不发请求、不需要运行 NPU 服务：

```bash
export C3_DIAG_OUTPUT="$RUN_ROOT/requests-c32-short-eager-u2-alloff"
test -f "$C3_DIAG_OUTPUT/batch-c32-short.summary.json"
```

以下旧启动/请求步骤留作历史参考；本轮下一步按[新修复复测文档](DEEPSEEK_V4_AFD_C3_EAGER_U2_ROPE_REVIEW_ZH.md)第 4/5 节执行。

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

在已安装现场 tokenizer 的 Python 环境离线解析全部 c32 输出，不增加请求、不改变服务采样。不能只看空文本，因为 `c32-short-3` 的退化序列仍能返回一个非空词：

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
    ids = choice.get("token_ids")
    row = {"file": path.name, "response_id": data.get("id"),
           "text": choice.get("text"), "usage": data.get("usage"),
           "token_ids": ids, "bos_token_id": tokenizer.bos_token_id,
           "eos_token_id": tokenizer.eos_token_id}
    if ids is not None:
        row.update(token_counts=dict(Counter(ids)),
                   bos_count=ids.count(tokenizer.bos_token_id),
                   special_token_counts={str(i): ids.count(i)
                                         for i in tokenizer.all_special_ids if i in ids},
                   decoded_keep_special=tokenizer.decode(ids, skip_special_tokens=False),
                   decoded_skip_special=tokenizer.decode(ids, skip_special_tokens=True))
    report.append(row)
(root / "output-token-report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps([r for r in report if not r["text"] or r.get("bos_count", 0)],
                 ensure_ascii=False, indent=2))
PY
```

同机使用上述 loopback；跨机客户端必须换为可达的对端地址。保留 `request_summary.json`、逐批 summary、32 条原始响应/错误文件、token 报告、metrics 与四角色日志。若定向测试不复现，不能宣布修复；再按原 full 的顺序采集相同 token 证据，保留失败前的 1/8 负载历史。

## 4. 【A5 手工执行】五开关全关对照（已执行且失败，保留步骤）

本节已执行，结果见第 1.2 节；第 5 节 eager 对照也已执行，不继续重复切换这五个开关。

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

## 5. 【A5 手工执行】eager/U2 对照（已执行并通过，保留步骤）

本节定向请求已经通过，详细结果见第 1.1 节。下一步按[RoPE 修复复测文档](DEEPSEEK_V4_AFD_C3_EAGER_U2_ROPE_REVIEW_ZH.md)更新代码并恢复 Graph/U2、默认五开关全开；不用重跑本节。

以本轮五开关全关配置为对照，仅将 A/F 的 `EXECUTION_MODE` 从 `full-decode-only` 改为 `eager`。保持 Flash、`U_BATCHES=2`、Attention `ENABLE_PD=1 ENABLE_DSPARK=0`、五开关全关、相同设备/端口/容量和源码；FFN 仍关闭 PD 与 DSpark。**不要改成 C4：C4 会同时打开 DSpark，无法作为本次 Graph 单变量对照。**

先归档当前目录、按指导书第 7 节停止服务并核对设备/端口清理。协调终端加载环境与单机拓扑后只生成一次新 ID，再复制到各角色/请求终端：

```bash
export CASE=C3 CYCLE=1 TOPOLOGY=single
export RUN_ID="$(date +%Y%m%dT%H%M%S)-single-C3-eager-u2-alloff-diag-r1"
export RUN_ROOT="$RUN_BASE/$RUN_ID/$(hostname -s)"
mkdir -p "$RUN_BASE/$RUN_ID"
mkdir "$RUN_ROOT"
printf 'RUN_ID=%s\nRUN_ROOT=%s\n' "$RUN_ID" "$RUN_ROOT"
```

其他终端赋值同一 ID/目录、不重复 mkdir。各终端加载 site/node、拓扑和指导书第 4.3 节配置后，**在第 4.4 节采集之前、任何 nohup 之前**执行下面整段；尤其 A/F 两个启动终端均须执行。后面不要再次执行 C3 的 case 分支，它会把模式改回 Graph：

```bash
export MODEL_PATH="$FLASH_MODEL_PATH"
export EXECUTION_MODE=eager U_BATCHES=2
export ENABLE_PD=1 ENABLE_DSPARK=0
export AFD_HCCL_STAGE_DIAGNOSTICS=1
export AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS=0
export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0
export AFD_HCCL_GRAPH_U2_HYBRID_DAG=0
export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0
export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0
```

按第 4.4 节重新采集实际源码提交/dirty state、runtime/config hash、site/node 和启动前设备状态；`AFD_EXPECTED_COMMIT` 应等于本轮实际 HEAD，不能用旧 runtime 或旧提交替代。此轮无需修改运行代码，仅下载资料也不必为此变更服务版本。保留 A5 本机 HIXL 绝对路径。按第 6.1/6.2 节冷启动 P、F/A、Proxy（F/A 同轮启动）；FFN 的 nohup 仍显式 `ENABLE_PD=0 ENABLE_DSPARK=0`。

就绪后检查真实 argv 与六个 rank 的配置，A/F 应有 `--enforce-eager`、`--enable-dbo`，`stages=2`、五开关全 `0`，Attention `speculative_config=None`；保留 eager 的真实在线 U2 stage 0/1 证据。recipe 支持 eager/U2，不需手工改服务脚本：

```bash
grep -nE 'enforce_eager|enable_dbo|speculative_config|AFD HCCL diagnostics configuration:' \
  "$RUN_ROOT/attention.log" "$RUN_ROOT/ffn.log"
export BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ATTENTION_URL="http://127.0.0.1:$ATTENTION_API_PORT"
export C3_DIAG_OUTPUT="$RUN_ROOT/requests-c32-short-eager-u2-alloff"
curl --noproxy '*' -fsS "$ATTENTION_URL/metrics" \
  > "$RUN_ROOT/attention-metrics-before-c32.txt"
"$PYTHON_BIN" "$AFD_PLUGIN_ROOT/tools/validation/a5_1030_requests.py" \
  --suite c32-short \
  --base-url "$BASE_URL" --attention-url "$ATTENTION_URL" \
  --output "$C3_DIAG_OUTPUT" --run-id "$RUN_ID-c32-short"
```

仍只发 EOS + 32 短请求，不加 `--standalone`，不需要 token-pool。结束后按第 3 节采集后续 metrics/health 和离线 **全部输出** token 报告，随后保存停服日志，归档新 `RUN_ROOT`。

若 eager/U2 同样出现全零或长段 BOS，则下一步检查 U2 基础执行与 PD/数值路径；若只在 Graph/U2 出现，则优先核对回放输入、padding/DSA metadata 和缓存地址。单轮未复现不能证明修复，更不能证明某个算子或设备有错。此诊断点不计 C3/C4 F0；正式 C3 仍使用 Graph/U2、五开关默认全开。

## 6. 设计复核

早期证据采集修改仅涉及自有 HTTP 验证客户端和诊断资料；最新 DSA RoPE 隔离及 Graph 原位刷新实现、性能成本和上游 ABI 复核见[修复设计复核](DEEPSEEK_V4_AFD_C3_EAGER_U2_ROPE_REVIEW_ZH.md)第 3 节。token ID 增加响应归档大小但不改变采样/文本验收；并发批次仍最多 32、每请求超时仍为 300 秒，失败批次等待在途请求完成以保留证据，随后停止新增负载。没有新增可变全局状态；诊断 suite 不替代完整验收。后续 vLLM 升级须复核 chat/completion 的 `return_token_ids` 协议。
