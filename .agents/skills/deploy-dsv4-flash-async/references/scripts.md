# 内置执行脚本

脚本目录为本 skill 的 `scripts/`。从 AFD 仓库根目录使用：

```bash
export DSV4_SCRIPT_DIR="${PWD}/.agents/skills/deploy-dsv4-flash-async/scripts"
```

先按 [环境配置](environment.md) 设置模型、网络和路径，再按所选模式执行。

| 脚本 | 用途 |
| --- | --- |
| `common_env.sh` | 公共环境、拓扑映射及参数校验，由启动入口加载 |
| `run_prefill.sh` | AFD 入口，按节点角色启动 FFN 和 Attention |
| `run_prefill_attention.sh` / `run_prefill_ffn.sh` | AFD 两个角色的启动实现 |
| `run_prefill_full.sh` | 非 AFD EP16/EP32 入口 |
| `run_prefill_full_rank.sh` | full-model 子进程实现，由 full 入口传入布局变量 |
| `run_decode.sh` | PD 的完整 D 服务 |
| `run_proxy.sh` | 调用已安装 vllm-ascend 的 PD proxy |
| `curl_test.sh` | 向 proxy 发 chat smoke 请求，REQUEST_TIMEOUT 默认 120 秒 |
| `run_bench.sh` | 对已启动的 prefill API 做单次性能测试 |
| `stop_local.sh` | 向指定 PID_DIR 记录的本地进程发送终止信号 |

示例（已完成公共环境设置）：

```bash
# AFD 单机 prefill
PREFILL_TOPOLOGY=afd_dp4tp2 PREFILL_NODE_ID=0 \
  PREFILL_ENABLE_KV_CONNECTOR=0 bash "${DSV4_SCRIPT_DIR}/run_prefill.sh"

# 非 AFD 单机：在独立部署环境中执行，勿与上例占用同一组设备
PREFILL_TOPOLOGY=ep16 PREFILL_NODE_ID=0 \
  PREFILL_ENABLE_KV_CONNECTOR=0 bash "${DSV4_SCRIPT_DIR}/run_prefill_full.sh"

# 已启动 AFD prefill 的单次 benchmark；数据集由用户提供
DSV4_BENCH_HOST="${P_NODE_IP}" \
DSV4_BENCH_DATASET_PATH=/absolute/path/to/workload.jsonl \
  bash "${DSV4_SCRIPT_DIR}/run_bench.sh" \
  --topology afd_dp4tp2 --chunk-size 8192 --request-rate 4 --repeat 1

# 核对 PID 归属后，在每个部署节点使用启动时相同的 PID_DIR
bash "${DSV4_SCRIPT_DIR}/stop_local.sh"
```

benchmark 的数据集需要 vLLM custom JSONL 字段 `prompt` 和 `output_tokens`。
使用不同于原实验的数据集时，设置真实 `DSV4_BENCH_DATASET_SHA256`；单次脚本只记录该元数据，不主动计算或核验哈希。完整数据集默认全部执行，先确认请求规模。结果默认位于仓库 `bench_results/dsv4-flash`，可用 `DSV4_BENCH_RESULT_ROOT` 覆盖。

## 来源与本次整理

来自仓库同级目录 `../scripts/dsv4-flash` 的部署脚本快照，保留原有拓扑和模型运行参数。本次只作以下集成调整：

- REPO_ROOT 默认定位脚本所在的 AFD 仓库根目录；源码导入路径默认使用 `/a3_inference/itask/workdir/jcz02615514/jcz-afd1` 下的 `vllm`、`vllm-ascend`、`afd-plugin`，可用 DSV4_CODE_ROOT 或各 PYTHON_*_PATH 覆盖；Python 默认使用 PATH 中的 python3。
- 移除个人 checkpoint 默认路径，模型服务与 benchmark 要求显式 DSV4_MODEL。
- D 启动统一使用 VLLM_CLI；非 AFD 入口按 KV connector 开关配置 PD producer，公共环境向 P/D 提供一致的 full-model 全局 DP/TP。
- chat smoke 增加连接和请求超时。
- 移除外部 CAM vendor 路径和 libopapi 预加载配置；通信算子随插件 pip 安装编译，由运行时 loader 加载。

未复制历史实验 runner、打包/绘图/审计脚本、日志或数据集。原 `run_bench_sweep.sh` 绑定特定数据集规模和零失败验收，未纳入这个通用部署 skill。PD proxy 仍依赖所选 vllm-ascend checkout 的实现。

这些脚本无 dry-run 参数。静态检查使用 `bash -n`；命令生成验证用临时 mock CLI，不能据此声称 NPU 上服务已验证。
