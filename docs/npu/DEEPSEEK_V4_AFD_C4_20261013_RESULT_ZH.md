# C4 同机 eager/U2 + DSpark 本轮核对结果

更新日期：2026-10-13。附件 `6eb6a279f37c4040843516b1966b7354.gz` 确认 **C4 full 92 项请求检查通过，90 个非流式响应全部非空且无 token ID `0`，真实 DSpark/U2 在线运行。正常停服仍失败，完整 F0 尚未关闭。** 按现场要求，C5 暂缓分析；已确认 FFN 启动两次，本次不修改 C5 启动代码。

## 1. 已确认的配置与业务结果

附件目录 `20261013T034132-single-C4-r1/localhost` 保留了早期 Prefill 失败日志与后续 `prefill-retry-20261013T072805.log`。本轮成功结果来自后者和 07:30 UTC 新启动的 A/F，不能用旧 `prefill.log` 判定重试仍失败。以下时间均为日志中的 UTC。

| 项目 | 本轮证据 |
|---|---|
| 拓扑 | 同机 P2+A4+F2，TP1；P0/1、A2/3/4/5、F6/7 |
| 模型 | 三角色 `/home/models/DeepSeek-V4-Flash-DSpark` |
| Prefill | 0.91、4096、eager，两个 Worker 均有 2.54 GiB KV，ready 成功 |
| Attention | eager/U2、PD on、DSpark on，`num_spec_tokens=5`；四 Worker 加载完整 DSpark draft |
| FFN | eager/U2、`speculative_config=None`；只执行 target FFN，DSpark/PD 开关为 0 正确 |
| 开关 | 六个 A/F rank 五个 Graph 开关全 1；eager 专用 overlap=1；本点不要求 Graph replay |
| PD | P 两 rank/A 四 rank 有布局与 request tracking 标记；6→5 组逐层真实传输 |
| 请求 | `requests/request_summary.json`：full 92 项、自然 EOS、1/8/32 短/长、边界、SSE/取消/恢复全部通过 |
| 输出 IDs | 90 个非流式响应全部非空，所有已返回 token IDs 均无 `0`；流式结果按 SSE 规则检查 |
| 请求结束 | 四 DP running/waiting 为 0，Proxy `status=ok`、`request_num=0` |

请求时间窗为 07:38:05–07:39:09 UTC。本轮没有再出现此前 Graph 对照中的连续尾部 `0`；不能以 eager/DSpark 成功替代 C3 Graph/DSpark off 的输出验证，也未检查 golden。summary 仍明确 `f0_passed=null`、`golden_checked=false`。

## 2. DSpark 确实执行，但 metrics 证据仍需补齐

四个 Worker 日志均有 `Initializing the DeepSeek V4 DSpark draft model`。逐 engine 的业务日志含正的 Accepted/Drafted 值，第一条正值示例如下；归属由紧邻的 `Engine 000/001/002/003` 统计行核对：

| Attention engine | Accepted | Drafted |
|---|---:|---:|
| 000 | 9 | 100 |
| 001 | 10 | 85 |
| 002 | 10 | 85 |
| 003 | 10 | 85 |

四 Attention rank 在线 `stage_count=2`，两个 FFN Worker 在线 `stages=(0,1)`、`graph_capture=False`、`warmup=False`、`target_graph_replay=False`，证明本轮是真实 eager/U2 业务执行。

但附件 `attention-metrics-before.txt` 和 `attention-metrics-after.txt` 中，四 engine 的 `vllm:spec_decode_num_draft_tokens_total`、`vllm:spec_decode_num_accepted_tokens_total` 均为 0，不能把这些指标的增量填为大于 0。业务日志支持 DSpark proposal/accept 生效，**Prometheus 指标定义/导出路径尚未核实**；不能由计数为 0 反推 DSpark 没有执行，也不能直接把日志的区间计数当作 Prometheus counter。后续按同一软件版本核对 DSpark 实际指标来源及采集时间窗，记录正确 before/after 定义后关闭该证据项。

## 3. 正常停服仍失败

`stop-time.txt` 为 07:45:28 UTC。Attention 四个 EngineCore 收到 SIGTERM，日志末尾出现四次 `Segfault encountered`；FFN 正常 shutdown payload 标记为 0。07:48:34 FFN EngineCore 收到 SIGTERM，07:48:45 FFN DP1 在 `torch.npu.synchronize()` 报 `error code is 507899` / `ERR00100`，随后 Worker 死亡。

这些异常发生在请求完成约 6 分钟后的停服阶段，不把它们计作 full 请求失败；也不能因为 full 已通过就豁免停服门禁。现有 shutdown 标记是插件 INFO 日志，标记缺失本身不足以证明 payload 一定没有发送/接收；Segfault 和停服期 FFN 同步错误已经足以表明正常退出未通过。

本包缺完整停服终端记录、`npu-after.txt`、`processes-after.txt`、`ports-after.txt`，无法核实最终清理。也缺本轮 `runtime.json`、site/node 与 config hash，不能据其他轮的提交记录补填本轮实际源码冻结。

## 4. 当前记录与下一步

本轮记为 **C4 请求检查通过、DSpark/U2 业务证据存在、完整 F0 pending**；保留已通过的 92 项，不为确认它们再次发 full。

【A5 手工执行】在本机加载原 `node.env` 的 `RUN_BASE` 后，单独定位 **C4 原目录**；即使当前 `RUN_ROOT` 已切为 C5，也不要改用 C5 目录。下面只采集当前状态，不发送请求、不启动或停止服务：

```bash
set -euo pipefail
: "${RUN_BASE:?先加载本机 node.env 中的 RUN_BASE}"
export C4_RUN_ROOT="$RUN_BASE/20261013T034132-single-C4-r1/$(hostname -s)"
test -f "$C4_RUN_ROOT/requests/request_summary.json"
export C4_SUPPLEMENT_DIR="$(mktemp -d "$C4_RUN_ROOT/c4-supplement-$(date +%Y%m%dT%H%M%S)-XXXXXX")"
date -Iseconds > "$C4_SUPPLEMENT_DIR/collection-time.txt"
npu-smi info > "$C4_SUPPLEMENT_DIR/npu-current.txt" 2>&1
ps -eo pid,ppid,lstart,args > "$C4_SUPPLEMENT_DIR/processes-current.txt"
ss -ltnp > "$C4_SUPPLEMENT_DIR/ports-current.txt" 2>&1
printf 'C4 补采目录：%s\n' "$C4_SUPPLEMENT_DIR"
```

把当时实际启动/停服的终端记录保存在此补采目录；没有保留的材料标记缺失。新采集文件记录的是当前状态，若现场已有其他轮服务，只能用于当前资源核对，不能充当 C4 停服时的清理快照。现场没有 `ss` 时，按统一指导书第 7 节补充端口核对，不能因工具缺失填写清理通过。

下一次冷启动前，按统一指导书第 3.1、4.3、4.4 节使用新 run ID 重新冻结 runtime、配置 hash 和模型参数；不能用现在的 checkout 信息回填本轮启动时的实际源码。不覆盖现有目录，也不为补采而立即重启或重发 full。

【本地继续】正常停服链路和 DSpark 指标导出仍是待处理项；修复后用新 run ID 在 A5 验证，再关闭完整 F0。C3 默认全开、输出退化和正常停服的既有剩余项继续保留。C5 按用户当前要求暂停分析，不根据这次重复启动日志继续修改配置或实现。
