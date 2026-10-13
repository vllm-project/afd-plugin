# C2 2026-10-13 回传结果核对

结论：**C2 请求验证通过；正常停服未通过，完整 F0 尚未关闭。下一步先处理停服链路，随后进入 C3。**

核对附件：`c34df56188ed459cb5bb5467ac8f3aa9.gz`，SHA256 为 `bb48f4df662b0c6263074a3153cba60b81b005e3dd7812d44de8d3456d6b50c1`。以下证据路径均相对于附件中的 `localhost/`。
日志实际发生于 2026-10-12；即使目录/run ID 沿用 `20261010T014609-single-C2-r1`，仍以本次请求时间、PID 和日志内容核对，不将旧 run ID 误判为旧服务。附件未提供 runtime 审计文件，不能仅从补丁标记反推出现场精确 Git HEAD。

## 1. 已通过的请求及真实路径

| 核对项 | 证据与结果 |
|---|---|
| EOS | 独立 EOS、smoke、full 均成功；full 自然结束，2 tokens、`finish_reason=stop` |
| full 完整覆盖 | `requests/request_summary.json`：92 项，`request_checks_passed=true`、`full_request_suite_passed=true` |
| 并发短/长输入 | 1/8/32 并发均执行；32 短、32 长各 32 项全部成功，均为 64 tokens/`length`；最大耗时分别约 4.805 秒、4.188 秒 |
| 边界、流式和恢复 | 6 个 boundary + stream + cancel + recovery 全部在 92 项记录中；90 个非流式响应均有 HTTP 200 归档 |
| 请求清空 | full 的 `after_recovery`：四 DP running/waiting 均为 0；Proxy `request_num=0` |
| PD 布局 | Prefill producer 6 组、Attention consumer 5 组；两端布局标记存在，Attention 有 98 条真实逐层传输记录 |
| 完成通知竞态 | Prefill 2 个、Attention 4 个 Worker 有 `request tracking active: register before enqueue`；本次日志无 `finish req not in reqs to process` |
| DSpark | 四个 Worker 加载 draft；09:18:30 UTC 的 Engine 000–003 业务窗口分别记录 accepted/drafted 为 220/245、100/100、95/95、60/60，均大于 0 |
| AFD | 两个 FFN Worker 均有在线执行记录；Attention/FFN 业务期未见 ERROR/Traceback |

full 的非流式记录时间为 09:17:52.950–09:18:39.944 UTC（北京时间 17:17:52.950–17:18:39.944）。本次请求期与后面的停服错误分开判定。
附件的 `attention-metrics-after-eos.txt` 只覆盖早期 EOS，`attention-metrics-before.txt` 随后采集；二者不能相减充当 full 的计数器增量。本轮 DSpark 在线证据来自各 Engine 的业务窗口日志；后续复测仍按指导书采集同一请求窗口的 before/after metrics。
请求 summary 自身的 `f0_passed=null`、`golden_checked=false` 也不能改写成完整功能或精度通过。

## 2. 剩余的停服失败

| UTC 时间 | 原始证据 |
|---|---|
| 09:22:22 | `stop-time.txt`；Attention API、EngineCore、Worker 同秒收到退出/TERM 信号 |
| 09:22:23 附近 | `attention.log` 尾部四次 `Segfault encountered`，并有 shared memory 泄漏提示 |
| 停 Attention 后 | `ffn.log` 中 `AFD NPU FFN received Attention shutdown payload` 数量为 0；`shutdown-failed.txt` 记录 gate failed |
| 09:25:40–09:25:46 | FFN 收到 TERM，随后 Worker died unexpectedly 和 EngineCore 清理 traceback |
| 09:26:00 | FFN API `Process manager: force killing remaining processes count=2` |
| 09:26:41 | `cleanup-time.txt`；进程表无本轮服务，服务端口已释放，八张 NPU 无运行进程 |

这不是仅凭历史残留 `shutdown-failed.txt` 判失败：本次日志时间线独立确认了异常退出和强杀。最终资源已释放，可保存为清理后的状态；**正常停服 gate 仍为 false**。

源码排查入口：`recipe/npu/P2pHcclAFDConnector/deepseek_v4/common.sh` 的 `run_role_service/stop_service_group` 在 launcher 收到 TERM 时向整个服务进程组发 TERM；`AFDNPUAttentionModelRunner.shutdown` 才发送 FFN shutdown payload。需要核对组信号传播、KV/HIXL 释放与 AFD 控制面释放的先后关系。这里是已确认的源码行为及排查方向，不能由 Python 日志单独确认底层 Segfault 原因。当前尚未交付停服修复，不能将重发 TERM 或增加等待时间写成已修复方案。

## 3. 下一步与 A5 手工边界

1. **【本地处理】** 修复并验证 launcher/Attention/FFN 的停服顺序，避免在 FFN 收到关闭通知之前并行拆除通信资源；提交具体修复及检查结果。
2. **【A5 手工执行：你；停服修复交付后】** 保留本次附件，更新并冻结修复提交，以新 run ID 冷启动 C2；保存真实提交、安装位置、PID/进程树和两端补丁标记。
3. **【A5 手工执行：你；停服修复交付后】** 先 EOS→smoke 检查退出前业务，采集同一窗口的 before/after metrics；请求归零后停 Proxy，再停 Attention，确认两个 FFN rank 收到 shutdown payload，再停 FFN launcher，最后 Prefill。采集退出结果、Segfault/错误时间线、进程/端口/NPU 状态。不得用强杀结果关闭正常退出 gate。
4. **【A5 手工执行：你；短退出复测成功后】** 在修复提交上完成正式 C2 full 和正常停服，补齐该轮 F0；本次已通过的 92 项证据保留，不为核对旧结果立即重跑。
5. **【A5 手工执行：你；C2 F0 关闭后】** 进入 C3：同机 P2+A4+F2，`EXECUTION_MODE=full-decode-only`、`U_BATCHES=2`、`ENABLE_DSPARK=0`，五个 Graph/U2 开关保持默认 1。按统一指导书第 6 节使用新 run ID 冷启动四角色，再 EOS→smoke→full，检查真实 Graph replay/U2 两个 stage 和正常停服。C3 完成后才进入 C4/C5。

C2→C3 同时关闭 DSpark、开启 Graph/U2；C3 对照 C1，不能描述为只改一个开关。A5 本机 HIXL 绝对路径配置继续保留，FFN 不加载 PD KV connector。
