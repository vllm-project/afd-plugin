# A5 上的同步 MegaFFN 实验后端

2026-11-16：`afd-plugin` 和 `megakernel` 的本地分支均为 `Dhq_AFD`。
本次实现已经通过单层、dummy 权重的双卡 HTTP 验证；这不代表真实模型精度或性能达标。

## 执行路径

```text
A：Attention + gate/top-k + shared experts
             |
             | CAMP2p / a2e：hidden_states、top-k IDs、已缩放的 routing weights
             v
F：MegaFFNBackend -> MegaFfnRunner.run_ffn()
   本地路由排序 -> GMM1 -> SwiGLU + 量化 -> GMM2 -> 本地加权还原
             |
             | CAMP2p / e2a：routed expert 输出
             v
A：合并 shared expert 输出，继续模型执行
```

MegaFFN 内部的路由排序和还原仍然执行。它们操作本卡 tensor，
不是 A/F 之间尚未接入的四个异步 dispatch/combine 通信算子。
本次保留原有 worker、model runner 和 CAMP2p 协议，在 F 侧模型计算入口替换 routed experts。

## 配置与边界

两侧 `additional-config` 的 `afd` 部分均配置：

```json
{
  "connector": "CAMP2pAFDConnector",
  "host": "127.0.0.1",
  "port": 40054,
  "num_attention_ranks": 1,
  "num_ffn_ranks": 1,
  "compute_gate_on_attention": true,
  "ffn_compute_backend": "mega_ffn",
  "connector_extra_config": {
    "hccl_buffer_size": 256,
    "quant_mode": 0
  }
}
```

另外分别设置 `role="attention"` / `role="ffn"`。
默认 `ffn_compute_backend="native"` 保留原有实现。

- 当前只接 DeepSeek V4、单 A/单 F、TP/DP/PP/CP 均为 1、eager、BF16 未量化源权重。
- 禁用 DBO、ubatching、speculative decoding。
- MegaFFN 暂无 V4 的 SwiGLU clipping：配置必须 `swiglu_limit=0`。
  真实模型如果要求 clipping，不能直接把这个值改成 0 后宣称语义等价。
- hidden size 和 FFN intermediate size 必须为 64 的倍数。
- vLLM 正常加载权重后，F 侧首次执行时一次性将 BF16 专家矩阵量化成 MXFP8/NZ。
  这是有损量化；没有把模型权重替换为 standalone demo 的空权重。
- 输入为 BF16 `[M,H]`、FP32 routing weights `[M,K]`、INT32/INT64 expert IDs `[M,K]`。
  kernel 固定输入 buffer 的 ID 类型是 INT32，copy 前检查范围。
- A 已经计算 routing scale，F 不重复 gate、不重复乘缩放因子、不执行 shared experts。
- 只使用同步 normal mode。copy、graph 执行、输出 clone 均完成后才返回 E2A。
  尚未使用 persistent/long-run，也未实现新的异步通信协议。
- 每层最多两张固定容量图：1 token 和 `max_num_batched_tokens`。
  其余 M 补零至容量，routing weights 补零、ID 补有效值 0，输出裁回 M。
  native graph 的资源随进程存在，不能通过 Python LRU 淘汰可靠释放，故不反复建图。
  此策略面向功能验证，有 padding、同步、额外内存开销。

## 代码位置

| 仓库 | 文件 | 作用 |
| --- | --- | --- |
| AFD | `afd_plugin/config.py` | 后端选择和协议组合校验 |
| AFD | `afd_plugin/model_executor/models/npu/mega_ffn_config.py` | 模型和并行范围校验 |
| AFD | `afd_plugin/model_executor/models/npu/deepseek_v4.py` | F 侧外部 routing 入口选择后端 |
| AFD | `afd_plugin/model_executor/models/npu/mega_ffn_backend.py` | 权重转换、固定容量图、输入输出适配 |
| AFD | `afd_plugin/compat/npu/feature_validation.py` | 启动时调用后端校验 |
| AFD | `afd_plugin/compat/npu/ops.py` | `AFD_CANN_VENDOR_PATH` 显式选择已有 A5 vendor |
| AFD | `afd_plugin/connectors/factory.py` | 修正 CAMP2p 模块注册路径 |
| Mega | `python/blockrt/models/mega_ffn/ffn_runtime.py` | 注入权重、检查输入、同步 `run_ffn()` |
| Mega | `python/blockrt/models/mega_ffn/{contracts,weight_loader}.py` | 输入合同、MXFP8/NZ 权重量化和检查 |
| Mega | `python/blockrt/models/mega_ffn/ffn_model.py` | INT32 IDs、F 侧不重复缩放/合并 shared experts |
| Mega | `python/blockrt/models/glm_5_2/moe.py` | 可选 finalize rounding，原有调用默认值保留 |
| Mega | `blade/runtime/core/megakernel.cpp` | user device ID 转 HAL logical ID，检查映射失败 |
| Mega | `examples/operators/moe_finalize_routing_v2/register/` | BF16 激活 + FP32 routing weights 选择兼容的 CutH tiling |

## 当前部署

操作容器：`dhq_afd`。源码：

```text
/mnt/share/d00893600/afd-plugin
/mnt/share/d00893600/megakernel
```

实验目录：`/data/libvirt/images/d00893600/afd-megaffn-20261116-01`。

- `attention.sh`：物理卡 2、HTTP 43526。
- `ffn.sh`：物理卡 3、HTTP 50998。
- `run_probe.py`：启动两侧、发两个请求、检查 logprobs 和 F 侧执行证据，然后关闭自己启动的进程。
- `model/config.json`：缩小到 1 层，H=4096、I=1024、E=8、K=2、`swiglu_limit=0`。
- 权重使用 `--load-format dummy`；tokenizer 指向 `/data/models/MiniCPM5-1B`。
  输出文字没有语义正确性要求。
- A5 a2e/e2a 使用之前构建的 vendor，通过脚本里的 `AFD_CANN_VENDOR_PATH` 选择。
  V4 和 A-side gate 的既有 A5 vendor 路径也显式写在两份脚本中。
- Mega 的 `libmega.so`、`ascendmk_rt` 已在自己的 build/output 中重新构建并部署。
  `build` / `output` 原有软链接保留；不要用清理构建命令删除它们。

容器内确认物理卡 2/3 空闲后，运行：

```bash
python /data/libvirt/images/d00893600/afd-megaffn-20261116-01/run_probe.py
```

也可以在两个终端分别运行 `bash .../ffn.sh` 和 `bash .../attention.sh`。
这两份实验脚本有 480 秒限时；不是常驻部署脚本。
probe 每次覆盖当前日志和 `result.json`，历史验证以备份日志和报告为准。

运行环境使用 `afd-env.sh` 指向自己的 CANN 9.2 SDK。
不要在这两个进程里再 source standalone Mega 的 deployment env：它会覆盖设备选择。
本次没有修改宿主机驱动、固件或 DeviceOS 安全配置。

设备日志曾显示通用 AICPU 包名已加载，但找不到 `libaicpu_manager.so`。
在自己的 SDK `/data/libvirt/images/d00893600/Ascend/cann-9.2.0` 内，
将当前自定义包以 `compat/Ascend-dhq-megaffn-20261116.tar.gz` 的独立名字注册到
`conf/ascend_package_load.ini` 后复测通过。旧 manifest 备份在实验目录
`ascend_package_load.ini.before-namespace`。这属于私有 SDK 部署状态，不在两个 Git 仓库内。
迁移环境时还需部署该包，不能只 checkout 分支。没有运行涉及 DeviceOS 配置的 install.sh。

## 验证结果与证据

- AFD 相关 CPU 单元测试：204 passed、1 skipped。
- Mega 合同/量化及实际 C++ finalize tiler 单元测试：8 passed。
- 实卡数值参考：M=128/8/1，每种两次 replay，gate/up 权重不同、每行输入和每次请求不同。
  独立 CPU 计算参考，检查输出不被下一次 replay 覆盖；全部 6 组通过。
  预先设定容差 `rtol=0.12, atol=0.002`，覆盖两次激活量化误差，没有放宽容差求通过。
- 双卡端到端：A=2/F=3；两个请求均 HTTP 200，各完成 8 token，logprobs 全有限。
  F 日志包含 `MegaFFN layer=0 completed tokens=...`，确认实际调用新后端。
  移除临时 tensor dump 后再次验证通过。退出码 -15 来自 probe 主动清理。
- 独立代码复核发现的图缓存 native 内存释放问题已用固定两种容量解决；
  数值参考也已补上不同 gate/up、行输入和 replay 输入，复核通过。

证据位于实验目录：`result.json`、`attention.log`、`ffn.log`、
`afd-unit-final.log`、`mega-unit-final.log`、`reference-varied-namespaced.log`。

没有验证：真实完整模型质量、端到端对 native 的数值等价性、吞吐/时延收益、
多 A/F rank、并发请求、persistent mode、四个异步 dispatch/combine 算子。
后续先补真实权重量化/精度及 V4 clipping，再扩展执行模式和通信路径。
