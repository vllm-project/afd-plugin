# W4A8 AFD 启动排障

以下经验来自 DeepSeek-V4-Flash W4A8、单机 Attention DP4TP2 + FFN EP8、`prefill_only`、MBT=65536 的部署。先核对模型、实际运行时与编译产物，再用日志中的首个有效错误定位；不能把这些数值直接当作其他拓扑的默认配置或性能验收。

## 启动前检查源码编译算子

当前 async CAM 算子由 AFD 源码构建，打包在 `afd_plugin/_cann_ops_custom/vendors/afd-plugin` 和 `_C_ascend` 中，不要求单独的 CAM vendor 库。不要以 `/usr/local/Ascend/.../vendors/CAM/op_api/lib/libopapi.so` 是否存在判定安装成败。检查安装日志的成功退出码、编译进程是否结束，并在与服务相同的 Python 环境中先导入 `torch`，再加载扩展；反过来先导入 `_C_ascend` 可能误报找不到 `libtorch.so`。

```python
import torch
import afd_plugin._C_ascend
from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

ensure_cam_async_ops_available()
print(torch.ops.afd_ascend.grouped_matmul_layered)
print(torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered)
```

算子注册成功只证明扩展可加载；还须通过服务 warmup 和实际请求。记录模型量化配置、所用 vLLM/vllm-ascend/AFD commit、CANN 版本及有效启动参数。两组性能对照须使用相同版本、模型和公共参数。

## MBT=64K 的 async dispatch 窗口

`csrc/npu/ascend_kernels/utils/op_host/ops_log.h` 中 `LCCL_BUFFER_SIZE` 的默认值为 408 MB。在上述 DP4TP2/EP8、MBT=65536 配置中，Attention dispatch tiling 需要约 1.61 GB；使用默认值曾在 warmup 报 `aclnnAfdAsyncDispatchSend failed`、`AfdAsyncDispatchSend do tiling failed, ret is -1`。该次部署在两个角色的进程启动前统一设置 `LCCL_BUFFER_SIZE=2048` 后越过此错误。遇到相同症状时，先核对实际 tiling 所需窗口和可用设备内存，再选取足够的 MB 值；不要仅凭 `ret is -1` 把错误归类为 OOM，也不要对所有 chunk 固定使用 2048。

## 动态算子配置路径

W4A8 layered GMM 若报 `aclnnGroupedMatmulSwigluQuantV2Layered` 后接 `Parse dynamic kernel config fail` 或 `ADD_TO_LAUNCHER_LIST_AICORE failed`，检查**服务进程启动时**的 `ASCEND_CUSTOM_OPP_PATH`。一次故障中它只指向已不存在的旧 CAM vendor，未包含插件打包的 `afd-plugin` vendor。`afd_plugin.compat.npu.ops` 的 loader 会添加插件路径，但 CANN 可能已在此之前读取了旧环境。可在启动 FFN 和 Attention 前，从实际导入的 AFD 包取得路径并显式配置：

```bash
AFD_VENDOR=$("${PYTHON:-python3}" - <<'PYTHON'
import torch
from afd_plugin.compat.npu.ops import get_afd_cann_vendor_path
print(get_afd_cann_vendor_path())
PYTHON
)
test -d "${AFD_VENDOR}" && test -f "${AFD_VENDOR}/op_api/lib/libcust_opapi.so"
export ASCEND_CUSTOM_OPP_PATH="${AFD_VENDOR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export AFD_CUST_OPAPI_LIB_PATH="${AFD_VENDOR}/op_api/lib/libcust_opapi.so"
export LD_LIBRARY_PATH="${AFD_VENDOR}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

核对 FFN 与 Attention 的 `/proc/<pid>/environ` 或等效启动证据，确认路径在进程创建时已生效；不要通过运行后的交互 shell 环境推断服务进程环境。该路径属于插件编译产物，不需要另装 CAM 库。

## layered GMM 开关只属于 FFN

`AFD_ASYNC_CAM_LAYERED_GMM=1` 只支持 DeepSeek V4 的 async CAM FFN 角色；传给 Attention 会在特性校验时报错。要比较关闭与开启 layered GMM 的部署：关闭侧在两个角色中都不设置该变量；开启侧分别调用角色脚本，只给 FFN 设置，Attention 显式取消继承：

```bash
AFD_ASYNC_CAM_LAYERED_GMM=1 bash "${DSV4_SCRIPT_DIR}/run_prefill_ffn.sh"
env -u AFD_ASYNC_CAM_LAYERED_GMM bash "${DSV4_SCRIPT_DIR}/run_prefill_attention.sh"
```

`run_prefill.sh` 会把同一环境传给两个角色，因此不能在其外层全局设置该开关。检查 FFN 运行日志中的 `AFD_ASYNC_CAM_LAYERED_GMM requested=... actual=layered/legacy`；只有环境变量或算子注册结果不足以证明实际执行了目标路径。路径日志缺失时标记为尚未确认，继续检查实际 FFN worker 日志，再解释性能结果。

每次定向重启使用新的日志和 PID 目录，保留失败启动的首个有效错误；先核对本次进程及其 worker 已退出、端口和 NPU 已释放，再启动下一次。`/health` 返回 200 后仍需完成单 token 请求验收。
