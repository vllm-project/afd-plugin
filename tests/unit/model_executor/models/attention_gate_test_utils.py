# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolate the attention-gate FFN contract from optional NPU dependencies."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_attention_gate_moe_ffn(monkeypatch, apply_mlp):
    torch = pytest.importorskip("torch")
    # Importing the production module would initialize optional vLLM/Ascend
    # dependencies on CPU test hosts, so compile only the function under test.
    source_path = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py"
    )
    function = next(
        node
        for node in ast.parse(source_path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_attention_gate_moe_ffn"
    )
    namespace = {
        "torch": torch,
        "AFDF2ATransferPayload": SimpleNamespace,
        "_gmmswigluquant_fusion_enabled": lambda: False,
        "_dequantize_int8_activation": lambda states, scales, *, output_dtype: states,
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(source_path), "exec"), namespace)

    quant_type = SimpleNamespace(NONE="none", W8A8="w8a8", W4A8="w4a8")
    modules = {
        "vllm_ascend.ops.fused_moe.moe_mlp": SimpleNamespace(apply_moe_mlp=apply_mlp),
        "vllm_ascend.ops.fused_moe.dataclass.moe_mlp": SimpleNamespace(
            MoEMlpComputeInput=SimpleNamespace
        ),
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts": SimpleNamespace(
            MoEWeights=SimpleNamespace
        ),
        "vllm_ascend.ops.fused_moe.dataclass.moe_quant": SimpleNamespace(
            MoEQuantParams=SimpleNamespace
        ),
        "vllm_ascend.quantization.quant_type": SimpleNamespace(QuantType=quant_type),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return namespace["compute_attention_gate_moe_ffn"], quant_type
