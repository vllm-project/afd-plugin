# cross-C5 两机验证结果（2026-10-16）

核对日期：2026-10-19。附件 `aa71744666ae49649fd7497a47ffbedc.gz` 确认 **本轮跨机 full 92 项请求检查全部通过，90 个非流式响应全部非空且无输出 token ID `0`；请求排空，FFN 未出现此前的 HCCL launcher 参数异常。** 本轮日志支持两机 P2+A4+F2、Prefill/FFN 共机的部署；三机 P8+A4+F2、Attention 侧完整路径证据、正常停服和第二轮冷启动尚不能据此关闭。

本报告核对跨机结果。归档文件只作为证据，没有执行其中的脚本、环境配置或指令，没有启动或停止现场服务。

## 1. 请求结果

原目录：`20261016T043524-cross-C5-r1/localhost`；请求目录：`requests-full-20261016T104408`。实际客户端时间为 **2026-10-16 北京时间 18:44:08.351–18:45:19.878**，由请求纳秒时间戳换算；服务日志为 UTC 10:44:08–10:45:19。该耗时不作为性能基准。

| 检查项 | 逐项核验结果 |
|---|---|
| 自然 EOS | HTTP 200，`Hello`，`finish_reason=stop`，2 个输出 token |
| 并发 1 短/长 | 各 1/1 通过 |
| 并发 8 短/长 | 各 8/8 通过 |
| 并发 32 短/长 | 各 32/32 通过 |
| KV block 边界 | 31/32/33/63/64/65 六项通过，各输出 32 token |
| 正常 SSE | 16 个 JSON 事件，有非空白文本、`length` 和 `[DONE]` |
| 取消与恢复 | 首片段到达后取消，未等到 `[DONE]`；随后 recovery HTTP 200、64 token |
| 排空 | summary 中四 Attention DP 的 running/waiting 均为 0，Proxy `request_num=0` |

已将 90 份 HTTP 元信息、原始 body、JSON 响应和 summary 交叉比对，全部 HTTP 200、文本非空、ID/usage/finish_reason 一致；5506 个输出 token ID 均不含 `0`。除 EOS 外，83 个非边界响应为 `length`/64 token，六个边界响应为 `length`/32 token。没有 transport.error 或缺失响应。

41 个并发长请求均为原 diverse 池前 1024 个 token。池长 1549，前 1024 个含 571 个不同 ID，SHA256 为 `b4040dba955525d816eabdf8bd89188f499980496b0629b20b3b943e765f5e79`，与 tokenize 原始响应及 metadata 一致。

取消文件只含一个空格片段；它满足当前脚本“收到 text 片段后断开”的判据，但不作为“生成非空白正文后取消”的额外覆盖证据。Proxy 的取消日志 `proxy.log:99` 与本次主动取消的 response ID 对齐，之后 recovery 成功，不计为服务故障。

summary 为 `request_count=92`、`request_checks_passed=true`、`full_request_suite_passed=true`，同时仍是 `f0_passed=null`、`golden_checked=false`。

## 2. 实际拓扑与配置冻结的差异

| 项目 | 本包支持的实际值 | 证据 |
|---|---|---|
| Prefill/Proxy 主机 | `7.246.125.18` | Prefill RPC/MQ 地址与 Proxy 访问日志 |
| FFN 主机 | `7.246.125.18`，NPU 6/7，DP2/TP1 | `ffn.log:2`、`:16`、`:83` |
| Attention 地址 | `7.246.125.14:8910` | token-pool metadata、`proxy.log:99`、runtime 地址一致 |
| Prefill 设备/规模 | NPU 0/1，DP2/TP1 | runtime、Prefill CLI `prefill.log:16`；非 DP8 |
| 模型与预算 | P/F 均 DSpark checkpoint、4096、32 seq、0.91 | P/F 实际 CLI；Attention 侧缺日志待核对 |
| FFN target | `enforce_eager=False`、`FULL_DECODE_ONLY`，capture sizes 1/2/4/8 | `ffn.log:82` |
| FFN U2/开关 | 两 rank 均 stages=2，五 Graph 开关全 1，eager overlap=0，compute sync diagnostics=0 | `ffn.log:129`、`:130` |
| A5 HIXL 路径 | Prefill 使用本机绝对目录 `/etc/hixlep` | `prefill.log:2`；Attention 的本机路径未提供 |

本包实际地址组合支持 **两机、P2/F2 在 .18、A4 在 .14**。它同时跨越 P→A 和 A→F 的主机边界；缺 Attention 端日志，尚不能逐 rank 复核两端 session、映射及传输完成记录。不能把这轮直接登记为计划的“三角色各一机、P8+A4+F2”完成。

归档的 `site.env` 却写 `ATTENTION_HOST_IP=7.246.125.12`、`FFN_HOST_IP=7.246.125.14`，与 runtime、FFN CLI、tokenize 地址及 Proxy 实际 decoder 不一致。`config.sha256` 与所附 site/node 文件字节相符，**文件 hash 正确不代表文件值就是进程实际配置**。本轮请求成功，不将这项冻结差异解释为已发生的路由故障；下一轮应在启动前统一实际配置并重新保存各节点清单。

本机 runtime 记录 afd-plugin `d704216de4eaad2b9416f1ff5f6319be93ee3b47`、vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、vLLM-Ascend `11ee45653b199a097805b87011824a81ffa51b95`，三个工作树干净；torch/torch_npu 为 2.10.0/2.10.0.post2，CANN 路径 `/usr/local/Ascend/cann-9.2.0`，image/build/HCCL package 未填写。该记录只有本机一份；`start-time.txt` 为 UTC 04:36:12，而此次 P/F 日志在 10:39 新启动，`npu-before.txt` 还是 PID 10565/10566/10773/10774 的旧进程。因此不能将这份快照扩写为两节点本次新进程均已完成一致的软件冻结，也不能据旧 NPU 快照认定新服务启动时仍有残留。

## 3. 在线执行与异常核对

FFN DP0 PID 14881、DP1 PID 14882，两 rank 的 `execute_begin` step 1–582 各出现一次且全部对应 `device_sync_complete`；最后同步完成为 UTC 10:45:20.633，见 `ffn.log:13488`、`:13489`。在客户端 full 时间窗内，每 rank 的 payload replay 标志为 True 391 步、False 154 步；其中单 stage 444 步、双 stage 101 步。单双 stage 与两类 replay 标志均有成功完成记录，在线 U2 在 FFN 端存在。

这些是协议 payload 标志和 Worker 完成记录，不是逐次底层 `graph.replay()` 调用计数，也不能替代四个 Attention rank 的 capture/replay/fallback 及 DSpark 验证证据。包中 **没有 `attention.log`、Attention metrics-before/after**，不能据 shell 中 `ENABLE_DSPARK=1` 就填入四 rank drafter 加载、drafted/accepted 增量或五开关实际生效。成功的 90 个响应和正常 SSE 支持服务完成了推理，暂未验证 GSM8K 或 golden。

P/F/Proxy 现存日志没有此前的 `execute kernel param invalid`、507011/507018、worker loop failure 或业务期 Traceback。FFN 启动的 capture 上限 8 小于潜在 token 数警告仍存在，但本轮 full 已通过，不能把该警告本身判为故障。

## 4. 正常停服仍未通过

| 文件/行号 | 日志 |
|---|---|
| `ffn.log:13490` | `common.sh: line 383: 14498 Killed setsid "$@"` |
| `prefill.log:707` | `common.sh: line 383: 14427 Killed setsid "$@"` |
| 两日志末尾 | leaked semaphore/shared_memory 警告 |

有被强杀的退出迹象，不能登记正常停服通过；缺停服终端和系统记录，不能直接认定是 OOM、手工强杀或 launcher 超时。FFN 最后一个有时间戳的业务同步在请求结束后完成，后续 Killed 行没有时间戳，不虚构强杀的确切时间。

包中也没有 shutdown/cleanup 时间、最终 NPU/进程/端口快照。FFN shutdown payload 标记未出现，该标记为 INFO，缺失本身不能证明协议一定未收到；Killed 与清理材料缺失已经足以保留退出门禁。本轮只标记 **两机 cross-C5 请求通过，完整 F0 pending**。

## 5. 【A5 手工执行】下一步

1. 保留本轮 full 和原始日志，不为核验已通过的 92 项立即重发。先从实际 Attention 主机 `.14` 补齐同轮 `attention.log`、本机 runtime/site/node/hash、原有 metrics-before/after（若当时有保存）；不要改附成 site.env 所写 `.12` 的其他轮日志。新采集状态注明时间，不能追溯冒充原轮 before/after。
2. 补两节点本轮停服终端记录和已有的最终 NPU/进程/端口材料；若没有，标记缺失。处理正常退出后，再在新轮验证，不以强杀后设备空闲代替正常停服。
3. 下一轮启动前，将各角色 IP、设备及运行清单统一到实际拓扑，重新冻结两个节点源码、CANN/toolkit/950 ops/HCCL build 与实际加载库。使用新 run ID，按指导书完成独立冷启动 full 和正常停服，累计所需的两轮证据。原轮无法补齐的版本/退出项不回填为 true。
4. 当前计划的主验收仍是三机 P8+A4+F2；该点需要相应三节点实际运行材料，本轮两机成功单独保留。功能证据关闭后按指导书第 8 节执行冻结口径 GSM8K 和关闭诊断后的性能采集；本包不代表上述项目已完成，也不自动补齐其他跨机 C1–C4 的验收状态。

本地复核产物位于 `/mnt/workspace/validation/cross_c5_20261019_aa717446/summary.json`，可用相邻 `verify_evidence.py` 重查。日志行号按原始 LF 分隔，CR 进度条不额外计行。附件 SHA256：`5ea8c98fc02db0208ba2bafaa767ab85d06a42eff6bc475b9fe958ac1fb0ab18`。
