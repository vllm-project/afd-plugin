# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""CPU registration of the routing-id op for tests that call the gate directly.

The op is normally registered by the AFD worker installer; gate unit tests and
standalone checks bypass that path, so they register the production op here
through real PyTorch.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest


@pytest.fixture
def registered_routing_ids_op(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.library import _scoped_library, infer_schema

    stub = types.ModuleType("vllm.utils.torch_utils")

    with contextlib.ExitStack() as stack:
        lib = stack.enter_context(_scoped_library("vllm", "FRAGMENT"))

        def registering_library(op_name, op_func, mutates_args, fake_impl):
            schema = infer_schema(op_func, mutates_args=mutates_args)
            lib.define(f"{op_name}{schema}")
            lib.impl(op_name, op_func, "CPU")
            if fake_impl is not None:
                lib.impl(op_name, fake_impl, "Meta")

        stub.__dict__["direct_register_custom_op"] = registering_library
        for name, module in (
            ("vllm", types.ModuleType("vllm")),
            ("vllm.utils", types.ModuleType("vllm.utils")),
            ("vllm.utils.torch_utils", stub),
        ):
            monkeypatch.setitem(sys.modules, name, module)

        import afd_plugin.compat.npu as afd_npu_package

        module_name = "afd_plugin.compat.npu.routing_ids"
        saved_module = sys.modules.get(module_name)
        had_attr = hasattr(afd_npu_package, "routing_ids")
        saved_attr = getattr(afd_npu_package, "routing_ids", None)

        def restore():
            if saved_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = saved_module
            if had_attr:
                afd_npu_package.routing_ids = saved_attr
            else:
                afd_npu_package.__dict__.pop("routing_ids", None)

        stack.callback(restore)
        sys.modules.pop(module_name, None)
        afd_npu_package.__dict__.pop("routing_ids", None)

        import afd_plugin.compat.npu.routing_ids  # noqa: F401  registers the op

        yield afd_plugin.compat.npu.routing_ids, torch.ops.vllm.afd_balance_routing_ids_
