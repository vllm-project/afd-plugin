# C2 DSpark PD KV 布局修复审查（2026-10-12）

现场证据：`cb15c79fe8614ade8b3fc654a8ec62d7.gz`，01:40–01:47 的本轮 PID/请求 ID。四个 DP scheduler 均已启用 10-10 补丁；EngineCore 在接收失败后继续运行，不将目录覆盖误判成旧进程。

## 问题与修复

同一 EOS 请求 Prefill 发出 6 组 KV block ID，Attention 只有 5 组。`11ee45653` 的 DSpark 分组会将 target SWA 与 draft SWA 合组；物理张量按页面大小重新排列，旧 Mooncake 按组号及注册地址数组下标配对。仅关闭异步 lookahead、按较短数组 zip 或移动组号，均不足以保证正确数据拷贝。上次 scheduler 补丁保留，但不代表 KV 传输已修复。

新补丁位于 `afd_plugin/compat/patches/npu/dspark_pd_layout.py`。Prefill/Attention 交换 target 缓存层的名称、实际组号、基础地址、块步长、块数和组件布局。Attention 按层名匹配两端，使用各自步长读入本地缓存；保留上游 SWA/state 的尾部 block 选择。DSpark 独有缓存不参与远端传输，实际两端组数可以不同。一次请求的全体目标、组件和 block 列表先验证，发现缺失层、语义/组件不同、远端块不足或空/越界块即拒绝，不启动部分拷贝。indexer 的重叠 K/scale/full 视图合并字节区间，避免重复拷贝；尾部 padding 不拷贝。

Proxy 的非流式后端失败返回 JSON HTTP 502，流式失败发出 SSE error；两者仍释放请求/KV。验证脚本保存原始 HTTP 状态、响应头和响应体再解析，不将失败算通过。

## 架构审查

- 固定 vLLM `0fc695fc6`（0.23）、Ascend `11ee45653`；检查方法参数和握手字段。两个复制方法的函数签名、返回标注及移除 PATCH 标记后的 AST 与冻结源码相同。
- 通过已有 Worker resolver 延迟加载：原生 Prefill 和 AFD Attention Worker 导入完成后、KV 缓存注册前安装。FFN 不触发布局安装。握手数据和接收状态在后台线程启动前准备，避免启动时抢先编码旧 metadata。
- 作用范围为 Mooncake producer 的 DSpark checkpoint，以及 AFD Attention + DSpark 的 Mooncake consumer。普通 Worker 保留原 msgspec schema/decoder；非 DSpark 传输委托捕获的原方法。该委托例外已在函数前说明，原算法较长且此场景需要整体替换。
- 扩展使用可导入、可 pickle 的模块级 dataclass，与原 msgspec map 格式兼容；包含协议版本。未更换上游 metadata 类。旧 Prefill 缺少扩展时，DSpark consumer 明确要求更新/重启，避免隐式位置配对。
- 不新增可变全局映射。远端布局缓存属于接收实例，复用上游有容量限制的容器和 metadata 锁，以 engine ID/端口区分 peer；控制面完成后只读布局。缓存与 session 信息同步检查。
- 不重新分配 NPU 缓存、不改 HBM 预算和分组策略、不增加 token 级设备同步。增加握手字节与请求级 host 匹配/地址构造，合并重叠区间并批量调用一次 Mooncake read；性能变化需要后续实测。
- 长期应在上游 Mooncake hybrid 握手支持异构分组/物理布局，并补齐原生 DSpark PD 分配/失败处理后移除此兼容补丁；本次不改变上游提交或第三方安装目录。

## 验证及现场入口

本地相关测试 72 项通过，包含真实 CPU 张量的 6→5 组拷贝、不同物理排列和步长、重叠 indexer 视图、非法布局拒绝、msgpack/pickle、普通 Worker 回退、Worker 导入顺序、代理错误及原始响应归档。Ruff 与 diff 检查通过。CPU 检查不验证 HIXL/NPU、模型语义、输出精度、DSpark acceptance 或 C2 F0。

`【A5 手工执行】` 完整更新/启动/请求命令在[合并指导书第 3.4 节](DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md)。更新同一插件提交，保留原失败证据，冷启动 Prefill/Attention/FFN/Proxy；确认 P 两个 Worker、A 四个 Worker 的 `AFD DSpark PD layer layout active:`。先 Proxy EOS 一请求，检查 `AFD DSpark PD layer KV transfer:` 的 regions/bytes 非零和自然 EOS。通过后 smoke，再 full；C2 F0 未通过不进入 C3。

## 32 并发通知丢失修复审查（同日追加）

附件 `78b041cf6b3b4f5cb30c3dddef3819de.gz`：EOS 正常 stop/2 tokens，KV 6→5 组成功且四 DP 排空。full 留下 44 个成功响应（EOS 1、1/8 短长 18、32 短 25）。另 7 个 KV 拷贝均成功（0.67–1.02 ms），紧随 `finish req not in reqs to process`，无 scheduler `Finished recving KV transfer`，反复 WAITING_FOR_REMOTE_KVS 直至 300 秒读取超时。未见上一轮 broadcast/传输失败。日志时间按现场日志原样引用，不用 RUN_ID 的旧日期判断是否旧服务。

`MooncakeConnectorWorker.start_load_kv` 的冻结源码先向接收线程入队全部请求，再执行 reqs_in_batch 登记。线程能在登记前完成，tracker 无法发布 finished 标记；之后仅增加 pending 请求，scheduler 永远收不到通知。修复把启用布局协议 Worker 的全批登记前移到任何入队前，并跳过原末尾登记；普通 Worker 保留原顺序。函数签名/返回标注匹配，复制体的差异全部 PATCH 标注。沿用 Worker 实例布局作用域，无新增全局状态、不改线程池、设备/网络/超时、KV 读写及 scheduler 门禁。请求级新增启动标记用于现场核实加载。

确定性测试在 add_request 返回前触发完成，固定上游顺序在 1/8/32 批次全丢通知，修复顺序全保留 finished 且 pending 归零。Prefill 的 delayed send 和不发送 rank 完成仍有登记。传输异常测试验证空 ReadTimeout 也会保存请求名和异常类型。本轮相关回归 77 项通过，Ruff/diff 检查通过，固定上游签名/返回标注和普通路径 AST 核对一致。本地回归不能代替 A5 并发成功及正常清理。上游长期应保证任务发布前登记并审查 tracker 生命周期；满足后移除本段兼容修复。

停止阶段的 Attention Segfault/强杀、FFN follower collective_rpc 错误发生在请求超时之后，不能用来解释此前 7 个遗漏通知，但仍属于未通过的清理门禁。本次先修复请求竞态，现场复测继续保存 shutdown 时间/日志和设备、端口状态。
