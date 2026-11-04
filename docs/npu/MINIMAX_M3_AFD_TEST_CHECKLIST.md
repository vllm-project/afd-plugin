# MiniMax-M3 AFD 验证方案（PR #530）

本文件是交给服务器 agent 执行的自包含 runbook：第 0 节是执行契约（怎么准备、
何时停止、交什么证据），第 1 节起是逐阶段命令与判据。**当前 PR 只完成了 CPU
契约层验证，A 组之后的所有项均未在 NPU 上执行过**，不要把"未执行"记为"通过"。

## 0. 执行契约（先读完再动手）

### 0.1 获取代码与安装

```bash
git clone https://github.com/wangyao-i/afd-plugin.git
cd afd-plugin
git checkout docs/minimax-m3-afd-test-checklist   # 含 PR #530 代码 + 本方案
git log --oneline -3                              # 代码提交应为 a2c29b9 及其上的 docs 提交

pip install -e .          # 在 NPU 机上会自动构建 CAMP2p 的 A2E/E2A CANN 算子
python -c "import afd_plugin; print(afd_plugin.__version__)"
python -c "import afd_plugin._cann_ops_custom" && echo CANN_OPS_OK
```

只验证代码、不需要本文件时，用 PR 分支 `feat/npu-minimax-m3-afd`（head `a2c29b9`）。
`pip install -e .` 必须成功：entry point `afd_plugin:register_afd` 与 CAMP2p 算子
都来自这一步，缺任何一项后面的 AFD 阶段都无意义。

### 0.2 硬性前提（任一不满足：停止并上报，不要降级执行）

- 单机 ≥ 4 张 NPU die（2 Attention + 2 FFN），同一 `$HOST` 可达；
- `torch_npu` 可导入，`vllm==0.32.0`，`vllm-ascend` 固定在
  `bd42eea01d8312c4df469fabaed45f77a12df4d3`；
- `$MODEL` 权重完整（`config.json` + `*.safetensors` + `model.safetensors.index.json`），
  磁盘余量 ≥ 2 倍模型体积；
- 端口 `6239`（AFD rendezvous）、`18000`（Attention HTTP）、`18001`（FFN HTTP）空闲；
- 环境干净：不要继承无关的 `HCCL_*`/`AFD_*`/`VLLM_*` 变量，每个阶段用独立 shell；
- 各阶段命令的 `--max-model-len`、量化配置、attention 后端必须前后一致，否则
  D/E 的逐字对比无意义。

### 0.3 角色边界（硬约束）

- 只执行验证，**不修改仓库代码、测试、权重和本方案**；发现问题记录并上报，
  本地改代码"让它过"会让这次验证失去意义；
- 不允许用 CPU 假实现、mock 或跳过项冒充 NPU 证据；被 skip 的用例记 `skipped`；
- 同一失败命令最多重试 1 次，仍失败就停止并上报，不要反复重启；
- 模型权重和完整日志不要上传到公网，只回传结论、错误片段和 `runs/` 目录清单。

### 0.4 阶段门禁与中止条件

| 阶段 | 通过条件 | 不通过时的动作 |
| --- | --- | --- |
| A（静态门禁） | A1–A5 全绿 | 停止。属环境/依赖问题，修好环境后从 A 重跑 |
| B（权重归属 dry-run） | 无 `unclassified`，计数自洽 | 停止并上报报错 key。属代码问题，不要绕过 |
| C（native 控制组） | 服务能起，16 条请求有输出 | 停止。说明权重/环境本身不可用，与 AFD 无关 |
| D（AFD eager 1A1F） | `mismatches: 0` | 记录 diff；先跑完 H，再停止，不进入 E/F/G |
| E（长 prompt） | 完成且与 native 一致 | 停止，附 connector/attention 日志尾部 |
| F（图模式 / DBO） | 与 native 一致 | 记录为 `failed`，可继续 G/H（F 属可选组） |
| G（VL text-only） | 启动成功且 0 mismatch | 记录并继续 H |
| H（负向契约 7 条） | 7/7 命中预期错误 | 记录未命中的用例，继续 I |
| I（角色隔离） | Attention HBM 明显低于 FFN | 记录数据即可，不阻塞 |
| J（收尾） | 两角色干净退出 | 若 hang，保留栈与日志后强制结束并上报 |

A/B/C/D 任一失败即视为"环境或代码不可用"，不要继续用后续阶段"补偿"。

### 0.5 证据与产物

统一放在 `runs/<YYYYmmdd-HHMM>/`，每阶段一个子目录：

```text
runs/<ts>/A_static/     pytest.log  pre_commit.log  registry.log  symbols.log  versions.log
runs/<ts>/B_weights/    counts.log  index.json(仅路径引用)
runs/<ts>/C_native/     native_eager.log  native_eager.json
runs/<ts>/D_afd_eager/  afd_attn.log  afd_ffn.log  afd_eager.json  diff.log
runs/<ts>/E_long/       long_prompt.log  long_prompt.json
runs/<ts>/F_graph_dbo/  attn.log  ffn.log  compare.log
runs/<ts>/G_vl/         attn.log  ffn.log  compare.log
runs/<ts>/H_negative/   h1.log ... h7.log
runs/<ts>/I_memory/     npu_smi_before.txt  npu_smi_after.txt
runs/<ts>/SUMMARY.md
```

每个阶段写一条 `result.json`：
`{"stage": "D", "command": "...", "exit_code": 0, "verdict": "pass|fail|skipped",
"detail": "...", "artifacts": ["afd_attn.log"]}`。`SUMMARY.md` 汇总 0.4 的表格。

### 0.6 上报模板（每条失败一段，不要合并）

```text
阶段: D（AFD eager 1A1F）
命令: <完整命令>
期望: mismatches: 0
实际: mismatches: 3 / 启动超时 / 报错文本
日志: runs/<ts>/D_afd_eager/<file>:<行号附近 10 行>
初步归因: 环境 | 权重 | 注册 | 权重归属 | connector 边界 | 未知
已排除: <做过的对照实验>
```

归因参考：`unclassified MiniMax-M3 checkpoint weight` → 权重归属；
启动即报 `supports Ascend NPU execution only` / `requires compute_gate_on_attention=false`
→ 注册或契约守卫被触发；ready 之后 hang 在某一层 → connector 边界；输出乱码但
进程正常 → 权重归属或残差边界。

---

## 固定上游版本

| 组件 | 版本 |
| --- | --- |
| vLLM | `0.32.0` (`5df63a72`) |
| vLLM-Ascend | `bd42eea01d8312c4df469fabaed45f77a12df4d3` |
| AFD 分支 | `feat/npu-minimax-m3-afd`，PR #530 head `a2c29b9` |

约定：`$MODEL` 为 MiniMax-M3 权重目录，`$HOST` 为本机可达 IP，工作目录固定为
`d:\code\afd-plugin`（NPU 机上为对应 clone 路径）。所有命令在同一 shell 会话内
先执行：

```bash
export MODEL=/path/to/minimax-m3
export HOST=127.0.0.1
export SERVED=minimax-m3
export AFD_PORT=6239
mkdir -p runs
```

---

## A. 静态与 CPU 门禁（无需 NPU，可先做）

### A1. 单元测试与注册表

```bash
python -m pytest tests/unit/model_executor/models/test_minimax_m3_weight_policy.py \
                 tests/unit/package/test_package.py -q
```

**预期**：全部通过。`test_minimax_m3_weight_policy.py` 在缺 `torch`/`vllm`/`vllm_ascend`
的机器上会整体 skip，这不算通过——必须在依赖齐全的环境里看到 `passed`。

### A2. pre-commit 等价门禁

```bash
pre-commit run --from-ref origin/main --to-ref HEAD
```

**预期**：除 `mixed-line-ending` 外全部通过。该 hook 在 Windows
（`core.autocrlf=true`）上会因工作树 CRLF 误报，Linux 上是干净的；本机已用
`git hash-object --no-filters` 逐文件确认 blob 为纯 LF。

### A3. 适配器注册可解析

```bash
python - <<'PY'
import importlib
import afd_plugin
for arch, path in afd_plugin._MINIMAX_MODEL_REGISTRATIONS.items():
    mod, cls = path.split(":")
    print(arch, "->", getattr(importlib.import_module(mod), cls))
PY
```

**预期**：两行输出分别解析到 `AFDMiniMaxM3SparseForCausalLM` 与
`AFDMiniMaxM3SparseForConditionalGeneration`。任何 `ImportError` /
`AttributeError` 都说明适配器引用的 `vllm_ascend` 符号与 pinned 提交不一致，先
核对 A4 再继续。

### A4. 上游符号对齐

```bash
python - <<'PY'
from vllm_ascend.models.minimax_m3 import minimax_m3 as n
from vllm_ascend.models.minimax_m3 import minimax_m3_vl as nvl
need = """MiniMaxM3DecoderLayer MiniMaxM3Model MiniMaxM3SparseForCausalLM
MiniMaxM3Attention MiniMaxM3SparseAttention MiniMaxM3MoE MiniMaxM3MLP
GemmaRMSNorm PPMissingLayer make_layers get_pp_group
make_pp_empty_intermediate_tensors make_empty_intermediate_tensors_factory
support_torch_compile _get_text_config maybe_prefix VocabParallelEmbedding
ParallelLMHead LogitsProcessor _get_max_position_embeddings
_get_rope_parameters _sparse_attention_layer_ids""".split()
vl_need = """MiniMaxM3VLModel MiniMaxM3SparseForConditionalGeneration
MiniMaxVLVisionModel PretrainedConfig maybe_prefix""".split()
print("native missing:", [x for x in need if not hasattr(n, x)])
print("vl missing:", [x for x in vl_need if not hasattr(nvl, x)])
PY
```

**预期**：两行都是空列表。

### A5. 版本前置

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
import torch_npu
print("torch_npu", torch_npu.__version__)
import vllm, vllm_ascend
print("vllm", vllm.__version__)
print("vllm_ascend", getattr(vllm_ascend, "__version__", "unknown"))
PY
```

**预期**：`torch_npu` 可导入，`vllm` 为 `0.32.0`，`vllm_ascend` 与固定提交一致。
注意 AFD 只注册 `AFD*` 别名类，原生 `MiniMaxM3*` 仍由 vLLM-Ascend 的注册表解析。

---

## B. 权重归属 dry-run（不加载权重）

先确认适配器的分类策略能覆盖真实 checkpoint 的全部 key。这一步能提前暴露
"checkpoint 里有未分类路径"这一最可能的失败点。

```bash
python - "$MODEL/model.safetensors.index.json" <<'PY'
import json, sys
from afd_plugin.model_executor.models.npu import minimax_m3 as a

index = json.load(open(sys.argv[1]))
counts = {"attention": 0, "ffn": 0, "both": 0, "dropped": 0}
for name in index["weight_map"]:
    roles = a._checkpoint_weight_roles(name)
    if not roles:
        counts["dropped"] += 1
    elif len(roles) == 2:
        counts["both"] += 1
    else:
        counts["attention" if "attention" in roles else "ffn"] += 1
print(counts, "total", len(index["weight_map"]))
assert sum(counts.values()) == len(index["weight_map"])
PY
```

**预期**：

- 不抛 `RuntimeError: unclassified MiniMax-M3 checkpoint weight`；
- `both` 等于 embedding / final norm / lm_head 的数量；
- 文本 checkpoint 的 `dropped` 为 0 或有 `mtp` 层时等于 MTP 参数数；
- VL checkpoint 的 `dropped` 等于 `vision_tower` + `multi_modal_projector` +
  `patch_merge_mlp` 参数数。

**不通过时**：把报错的 key 贴到 PR #530，需要扩展
`_checkpoint_weight_roles` 的 stage 分类，而不是放宽为"未分类即双侧加载"。

---

## C. native 控制组（原生 vLLM-Ascend，不经 AFD）

目的：证明权重、chat template、attention 后端本身可用，为后续对比提供基准。

### C1. 起服务（纯文本 checkpoint）

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_PLUGINS=ascend \
vllm serve "$MODEL" --trust-remote-code --served-model-name "$SERVED" \
  --host 127.0.0.1 --port 18000 \
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 \
  > runs/native_eager.log 2>&1 &
```

**注意**：`--language-model-only` 只对**多模态 checkpoint** 有效；文本 checkpoint
加上它会报错。VL checkpoint 见 G 组。

**预期**：启动成功，日志无 `unclassified`、无 OOM、无 IndexError；
`curl -s localhost:18000/health` 返回 200。

### C2. 固定负载

```bash
python - "$SERVED" 18000 runs/native_eager.json <<'PY'
import json, sys, urllib.request

served, port, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "请用一句话解释什么是张量并行：",
    "Q: 12 * 12 = ?\nA:",
] * 4
body = json.dumps({
    "model": served, "prompt": prompts, "temperature": 0,
    "max_tokens": 128, "seed": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", body,
    {"Content-Type": "application/json"})
data = json.load(urllib.request.urlopen(req, timeout=1800))
json.dump([c["text"] for c in data["choices"]], open(out, "w"))
print("saved", out, "prompt_tokens", data["usage"]["prompt_tokens"])
PY
```

**预期**：16 条 greedy 输出写入 `runs/native_eager.json`，`finish_reason` 为
`length` 或 `stop`，无空串。

---

## D. AFD eager 基线（1A1F，同步 CAMP2p，FFN 侧 gating）

拓扑：Attention 2 die（`0,1`）+ FFN 2 die（`2,3`），`compute_gate_on_attention=false`，
`quant_mode=0`。两个进程的 `host`/`port`/`num_attention_ranks`/`num_ffn_ranks` 必须
完全一致，只有 `role`、`ASCEND_RT_VISIBLE_DEVICES`、HTTP 端口不同。

### D1. 先起 FFN 角色

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 VLLM_PLUGINS=ascend,afd \
vllm serve "$MODEL" --trust-remote-code --served-model-name "$SERVED" \
  --host 127.0.0.1 --port 18001 \
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 \
  --additional-config '{"afd":{"role":"ffn","connector":"CAMP2pAFDConnector","host":"'"$HOST"'","port":'"$AFD_PORT"',"num_attention_ranks":2,"num_ffn_ranks":2,"compute_gate_on_attention":false,"connector_extra_config":{"ffn_core_num":8,"hccl_buffer_size":2048,"quant_mode":0}}}' \
  > runs/afd_ffn.log 2>&1 &
```

### D2. 再起 Attention 角色

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_PLUGINS=ascend,afd \
vllm serve "$MODEL" --trust-remote-code --served-model-name "$SERVED" \
  --host 127.0.0.1 --port 18000 \
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 8 \
  --additional-config '{"afd":{"role":"attention","connector":"CAMP2pAFDConnector","host":"'"$HOST"'","port":'"$AFD_PORT"',"num_attention_ranks":2,"num_ffn_ranks":2,"compute_gate_on_attention":false,"connector_extra_config":{"attn_core_num":8,"hccl_buffer_size":2048,"quant_mode":0}}}' \
  > runs/afd_attn.log 2>&1 &
```

**预期**：

- 两个进程都在 `--startup-timeout`（建议 `1800`s）内完成 ready；
- FFN 日志里构造了 `MiniMaxM3MoE`（sparse 层）与 `MiniMaxM3MLP`（dense 层）；
- Attention 日志里没有任何 `block_sparse_moe.experts` 权重加载记录；
- 两侧都不出现 `MiniMax-M3 checkpoint weight` 分类错误。

**典型失败**：FFN 端先起但 Attention 端一直卡在 rendezvous → 检查 `host` 是否
双方可达、`port` 是否被占用、两侧 rank 数是否一致。排查入口见
[TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

### D3. 同一负载，对比 greedy 输出

```bash
python - "$SERVED" 18000 runs/native_eager.json runs/afd_eager.json <<'PY'
import json, sys, urllib.request

served, port, baseline, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "请用一句话解释什么是张量并行：",
    "Q: 12 * 12 = ?\nA:",
] * 4
body = json.dumps({
    "model": served, "prompt": prompts, "temperature": 0,
    "max_tokens": 128, "seed": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", body,
    {"Content-Type": "application/json"})
data = json.load(urllib.request.urlopen(req, timeout=1800))
texts = [c["text"] for c in data["choices"]]
json.dump(texts, open(out, "w"))
ref = json.load(open(baseline))
bad = [(i, a, b) for i, (a, b) in enumerate(zip(ref, texts)) if a != b]
print("mismatches:", len(bad))
for i, a, b in bad[:4]:
    print(f"#{i}\n  native: {a[:80]!r}\n  afd   : {b[:80]!r}")
PY
```

**判定**：`mismatches: 0` 才算通过。若出现个别不一致，先确认两边 `max-model-len`、
量化配置、attention 后端完全一致，再怀疑 FFN 边界（`RemoteFFNProxy` 发送的是
post-attention norm 之后的 hidden states，残差仍在 Attention 侧）。

---

## E. 长 prompt / 批量（chunked prefill 与 stage 映射）

在 D 组服务不变的前提下跑一次长上下文，验证 `AFDTransferMetadata` 的
`layer_idx` / `stage_idx` / `seq_len` 在分块 prefill 下仍然对齐。

```bash
python - "$SERVED" 18000 <<'PY'
import json, sys, urllib.request
served, port = sys.argv[1], sys.argv[2]
long_prompt = "The quick brown fox jumps over the lazy dog. " * 400  # ~4k tokens
body = json.dumps({
    "model": served, "prompt": long_prompt,
    "temperature": 0, "max_tokens": 32, "seed": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", body,
    {"Content-Type": "application/json"})
data = json.load(urllib.request.urlopen(req, timeout=1800))
print("prompt_tokens", data["usage"]["prompt_tokens"], "text", data["choices"][0]["text"][:40])
PY
```

**预期**：请求完成不超时、不 hang；`prompt_tokens` 在 4000 上下；
与 native 服务（C1 改成同样长 prompt）逐字一致。

**典型失败**：hang 在某个 layer → connector 的 stage 数与 `num_ffn_ranks` 不匹配；
输出乱码 → 权重归属过滤把某个双侧参数误分类，回到 B 组核对。

---

## F. 图模式与 DBO（可选，D 组通过后再做）

在 D1/D2 两条命令上追加：

```bash
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}'
```

以及 DBO：

```bash
  --enable-dbo --dbo-decode-token-threshold 2 \
  --dbo-prefill-token-threshold 12 --ubatch-size 2
```

**预期**：图捕获阶段无 OOM、capture size 覆盖实际 batch；DBO 下 16 条 greedy
输出仍与 `runs/native_eager.json` 完全一致。若图模式失败但 eager 通过，先记录
失败阶段（capture vs replay）再判断是否需降级，不要直接改适配器代码。

---

## G. 多模态 checkpoint（`MiniMaxM3SparseForConditionalGeneration`）

仅对 VL 权重执行。关键是 `--language-model-only`：它只关闭多模态输入路径，
不切换架构类，AFD 注册的是同名 VL 架构。

在 D1/D2 两条命令的 `vllm serve` 参数里加 `--language-model-only`，其余不变。

**预期**：

- 两个角色都能启动，vision tower 被原生 staging 但不执行；
- `runs/afd_attn.log` 与 `runs/afd_ffn.log` 均不出现 vision 参数加载报错；
- D3 的 greedy 对比与文本 checkpoint 一样为 0 mismatch。

**负向**：去掉 `--language-model-only` 后必须**在构造期**失败，错误信息含
`AFD MiniMax-M3 currently supports text-only execution only; pass --language-model-only`。
若改成运行期才报错，说明 `_validate_text_only` 的位置被改动过。

---

## H. 负向契约矩阵

每个用例都是"改一个开关 → 期望在模型构造期抛错，而不是跑到一半"。用 D2 的
Attention 命令做基线，逐个替换 `--additional-config` / 追加参数，检查 stderr。

| # | 变更 | 期望错误片段 |
| --- | --- | --- |
| H1 | 在 CUDA 机器上跑同一条 AFD 命令 | `AFD MiniMax-M3 supports Ascend NPU execution only` |
| H2 | `"compute_gate_on_attention": true`（AFD 与 connector 两处） | `AFD MiniMax-M3 requires compute_gate_on_attention=false` |
| H3 | `--enable-eplb` | `AFD MiniMax-M3 does not support EPLB` |
| H4 | `--pipeline-parallel-size 2` | `AFD MiniMax-M3 does not support pipeline parallelism` |
| H5 | `--speculative-config '{"method":"eagle3","model":...}'` | `AFD MiniMax-M3 does not support speculative decoding` |
| H6 | `--enable-lora` + 任一 LoRA 参数 | `AFD MiniMax-M3 does not support LoRA` |
| H7 | SP MoE（需 `TP>1`、`DP>1`、`--enable-expert-parallel` 且 all2all backend 为 `allgather_reducescatter` 等） | `AFD MiniMax-M3 does not support sequence-parallel MoE` |

**判定**：7 条全部命中且进程退出码非 0。任何一条被放行，都是契约守卫失效。

H2 特别注意：connecter 内部 `connector_extra_config.compute_gate_on_attention`
与 AFD 顶层字段必须同为 `false`，否则也会被 connector 校验拦下。

---

## I. 角色隔离的间接证据

权重归属策略的收益是"另一侧不加载大模块"，用内存占用观测：

```bash
npu-smi info
```

在两个角色都 ready 后重跑一次，分别记录每个 device 的 HBM 占用。

**预期**：Attention 侧进程的 HBM 明显低于 FFN 侧（Mixtral 型 MoE 的专家权重只在
FFN 侧）。若两侧接近，说明 `load_weights` 过滤没有生效，回到 B/D 组排查
`_iter_role_weights` 是否被 `super().load_weights` 二次消费。

---

## J. 收尾与证据归档

1. 对两个角色分别 Ctrl-C，**预期 30s 内退出，无 ERROR 级 HCCL 报错**；
2. 归档到 `runs/`：

```text
runs/
  native_eager.log      native_eager.json
  afd_attn.log          afd_eager.json
  afd_ffn.log           npu_smi_before.txt / npu_smi_after.txt
```

1. 把以下结论回填 PR #530 的 Test Result：

| 项 | 结果 |
| --- | --- |
| A1–A5 静态门禁 | |
| B 权重归属 dry-run | |
| C native 控制组 | |
| D AFD eager 1A1F 输出一致性 | mismatches = ? |
| E 长 prompt | |
| F 图模式 / DBO | |
| G VL + `--language-model-only` | |
| H 负向契约 7/7 | |
| I 角色内存隔离 | |
| 环境版本（torch_npu / vllm / vllm-ascend commit） | |

**"跑通"的判定**：A、B 全绿；C 与 D 的 16 条 greedy 输出完全一致；E 长 prompt
完成且一致；H 全部按预期拒绝；J 干净退出。F、G 作为加分项，未执行需在 PR 里
明确标注为"未验证"，不得默认通过。
