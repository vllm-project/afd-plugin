---
name: deploy-dsv4-flash-async
description: 部署或整理 DeepSeek-V4-Flash 在 Ascend NPU 上的 CamAsync AFD、prefill_only、非 AFD EP baseline 和 P/D 分离部署；基于 dsv4-flash 本地脚本生成配置、启动命令并检查服务。
---

# DSV4-Flash Async 部署

根据用户需求准备或执行部署。用户只要求方案、脚本或 skill 时，交付配置与命令即可；用户已要求部署时，完成环境检查、启动和验收，无须重复确认已授权步骤。

## 选择模式

Prefill 实现与 PD 分离是两个独立维度：先选 AFD 或非 AFD，再选 prefill_only 或 P/D 分离。`PREFILL_ENABLE_KV_CONNECTOR` 控制 P 侧是否启用 PD KV producer。

| Prefill 实现 | prefill_only（KV=0） | P/D 分离（KV=1） | Prefill 参考 |
| --- | --- | --- | --- |
| AFD / CamAsync | `run_prefill.sh` | D → `run_prefill.sh` → proxy | [AFD](references/afd-prefill.md) |
| 非 AFD / full-model EP | `run_prefill_full.sh` | D → `run_prefill_full.sh` → proxy | [非 AFD](references/baseline.md) |

两种 PD 组合都需同时阅读 [PD 分离](references/pd.md)，使用相同的 D/proxy 入口。AFD 拆分 P 内部的 Attention/FFN，PD 拆分 Prefill/Decode，两者可以叠加。

这里的 AFD async 是 `CAMAsyncAFDConnector` + `afd.async=true`，不是 vLLM 的 `--async-scheduling`。参考脚本仅在 D 端显式启用后者。非 AFD baseline 不应为了名称包含 async 而自动增加异步调度参数。

`prefill_only` 是本组脚本的工作负载约定：不启用 KV connector，不启动 D/proxy，直接向 P API 请求一个输出 token；不是额外的 CLI 开关，也不保证服务拒绝长输出。

## 找到脚本与环境

默认使用本 skill 内置的 `scripts/`（相对本文件），将其绝对路径设置为 `DSV4_SCRIPT_DIR`。用户显式指定其他脚本目录时使用指定目录并先核对差异。脚本清单和整理范围见 [执行脚本](references/scripts.md)。

先读内置 `common_env.sh` 和所选模式启动脚本。脚本源自 2026-09-28 的 `../scripts/dsv4-flash`，部署不依赖原目录；历史实验的暂停、RPS 矩阵和特定 OOM 结果不是通用部署限制。

整理以下信息，能够从环境获得的直接读取，缺少实际部署所必需的信息再询问：

- 目标节点/容器、910C 可用逻辑设备及占用、角色分布、节点间可达 IP 和 NIC。
- 模型目录、served model name、vLLM/vllm-ascend/AFD 路径与 commit、Python/CLI 路径、CANN 和随插件安装的源码通信算子。
- 拓扑、chunk、最大上下文、并发序列数、显存比例、端口及本次独立日志/PID 目录。

源 README 的环境基线为 vLLM/vllm-ascend v0.26、支持 DSV4 的 AFD、CANN 9.0.1、W8A8 checkpoint；这些不是对任意同名发行版兼容性的保证。部署前确认实际 checkout 的模型、CLI 和 connector 可用，记录版本，不自行升级依赖。使用 W4A8 AFD 时还要阅读 [W4A8 启动排障](references/w4a8-startup.md)；其中的 64K 配置来自一次 DP4TP2/EP8 部署，不是其他拓扑的容量保证。

公共环境模板见 [环境与验收](references/environment.md)。代码根目录 `DSV4_CODE_ROOT` 默认为 `/a3_inference/itask/workdir/jcz02615514/jcz-afd1`，vLLM、Ascend 和 AFD 源码默认分别使用其下的 `vllm`、`vllm-ascend`、`afd-plugin`。模型路径必须提供；代码与 CANN 路径按实际环境覆盖。

## 执行与交付

1. 按对应参考生成逐节点环境和启动命令，核对设备不重叠、全局 DP 与 local DP/start rank 一致，以及通信参数在各角色一致。
2. 启动前检查模型可读、所需库可加载、NPU/端口可用。AFD 按环境参考检查插件内置的四个 `afd_async_*` 算子；PD 额外检查 Mooncake 和 proxy 脚本。不要把 `source common_env.sh` 当纯只读检查，它会创建日志目录并修改环境。
3. 启动脚本使用 nohup；退出码为零和写出 PID 只表示提交成功。使用独立日志目录，按约定的启动超时监测进程、日志及健康接口；失败保留首个有效错误，不无限重启。
4. 按 [环境与验收](references/environment.md) 完成健康检查和实际请求。prefill 单 token smoke 不等于模型精度或性能验收。只有用户要求 benchmark 时才读取并运行 内置 `run_bench.sh`。
5. 报告模式、节点/设备/DP/TP/EP、有效配置、版本、命令、日志/PID、API 地址和验证结果。明确区分“配置已准备”“启动已提交”“健康且请求成功”；未上板验证的配置标为未验证。

需要停止时，仅处理本次部署的进程。`stop_local.sh` 遍历 PID_DIR 内所有 PID，且仅向所记 PID 发信号；先核对 PID 归属，停止后检查 worker、端口和 NPU 是否释放，不使用全局 pkill。模式切换使用新 shell，避免 baseline 的 VLLM_PLUGINS 等变量污染 AFD。
