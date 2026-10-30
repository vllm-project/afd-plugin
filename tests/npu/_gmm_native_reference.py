# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run the CANN A8W8 reference without AFD/Ascend custom API overrides.

Keep runtime imports inside the child entry point: op-plugin captures custom
vendor paths when its native libraries load, so clearing them in pytest is too
late and would also change the environment being qualified by the AFD path.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

_ND_FORMAT = 2
_REFERENCE_TIMEOUT_S = 300
_REFERENCE_APIS = (
    "aclnnGroupedMatmulSwigluQuantWeightNzV2",
    "aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize",
)
_EXCLUDED_MODULES = ("afd_plugin", "vllm", "vllm_ascend", "fla_npu")
_EXCLUDED_LIBRARY_PATHS = (
    "/afd_plugin/",
    "/vllm_ascend/",
    "/vllm-ascend/",
    "/fla_npu/",
    "/vendors/afd-plugin/",
    "/vendors/custom_transformer/",
    "/vendors/fla_npu_transformer/",
)


def capture_a8w8_request(
    torch_npu, x, x_scale, weight, weight_scale, group_list, group_list_type: int
) -> dict:
    """Copy the exact inputs, exporting NZ weight values in portable ND form."""
    return {
        "x": x.cpu(),
        "x_scale": x_scale.cpu(),
        "weight_nd": torch_npu.npu_format_cast(weight, _ND_FORMAT).cpu(),
        "weight_scale": weight_scale.cpu(),
        "group_list": group_list.cpu(),
        "group_list_type": group_list_type,
    }


def builtin_a8w8_references(
    torch, requests: list[dict], artifacts: Path, nz_format: int
):
    """Run one ordered batch per pytest case and retain all child diagnostics."""
    inputs_path = artifacts / "native-inputs.pt"
    outputs_path = artifacts / "native-outputs.pt"
    torch.save(
        {
            "requests": requests,
            "device": torch.npu.current_device(),
            "nz_format": nz_format,
        },
        inputs_path,
    )
    child_env = os.environ.copy()
    child_env["ASCEND_CUSTOM_OPP_PATH"] = ""
    child_env["VLLM_PLUGINS"] = ""
    # Use FLA's dedicated opt-out so its .pth hook cannot reinsert vendor paths.
    child_env["FLA_NPU_DISABLE_PTH"] = "1"
    print(f"CANN reference artifacts: {artifacts}", flush=True)
    with (
        (artifacts / "native-stdout.log").open("w") as stdout,
        (artifacts / "native-stderr.log").open("w") as stderr,
    ):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(artifacts)],
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            check=True,
            timeout=_REFERENCE_TIMEOUT_S,
        )
    outputs = torch.load(outputs_path, map_location="cpu", weights_only=True)
    assert len(outputs) == len(requests), "native reference batch is incomplete"
    report = json.loads((artifacts / "native-provider.json").read_text())
    print(f"CANN reference providers: {report['providers']}")
    return outputs


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("address", ctypes.c_void_p),
    ]


def _check_native_provider(torch_npu, report_path: Path) -> None:
    """Check addresses returned by op-plugin's actual native API resolver."""
    runtime_path = Path(torch_npu.__file__).resolve().parent / "lib" / "libtorch_npu.so"
    runtime = ctypes.CDLL(str(runtime_path), mode=os.RTLD_NOLOAD)
    # Exported by the pinned torch_npu runtime: const char * -> void *.
    resolve_api = runtime._Z16GetOpApiFuncAddrPKc
    resolve_api.argtypes = [ctypes.c_char_p]
    resolve_api.restype = ctypes.c_void_p
    loader = ctypes.CDLL("libdl.so.2")
    loader.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    loader.dladdr.restype = ctypes.c_int
    providers = {}
    for name in _REFERENCE_APIS:
        info = _DlInfo()
        address = resolve_api(name.encode())
        assert address, f"op-plugin could not resolve {name}"
        assert loader.dladdr(address, ctypes.byref(info)), name
        providers[name] = str(Path(os.fsdecode(info.filename)).resolve())
    shared_libraries: list[str] = sorted(
        {
            line.split(maxsplit=5)[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line
        }
    )
    excluded_modules_loaded: list[str] = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in _EXCLUDED_MODULES
    )
    report = {
        "resolver_library": str(runtime_path),
        "providers": providers,
        "shared_libraries": shared_libraries,
        "excluded_modules_loaded": excluded_modules_loaded,
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "vllm_plugins": os.environ.get("VLLM_PLUGINS"),
        "fla_npu_disable_pth": os.environ.get("FLA_NPU_DISABLE_PTH"),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    assert not excluded_modules_loaded, report
    assert not any(
        marker in path
        for path in shared_libraries
        for marker in _EXCLUDED_LIBRARY_PATHS
    ), report
    assert all(
        Path(path).name == "libopapi_transformer.so" for path in providers.values()
    ), report


def _main(artifacts: Path) -> None:
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == ""
    assert os.environ["VLLM_PLUGINS"] == ""

    # Import only the native torch_npu runtime, after the subprocess environment
    # has been sanitized. Do not import the test module or its pytest conftest.
    import torch
    import torch_npu

    payload = torch.load(
        artifacts / "native-inputs.pt", map_location="cpu", weights_only=True
    )
    torch.npu.set_device(payload["device"])
    torch.npu.config.allow_internal_format = True
    _check_native_provider(torch_npu, artifacts / "native-provider-before.json")
    outputs = []
    for request in payload["requests"]:
        x = request["x"].npu()
        x_scale = request["x_scale"].npu()
        weight = torch_npu.npu_format_cast(
            request["weight_nd"].contiguous().npu(), payload["nz_format"]
        )
        weight_scale = request["weight_scale"].npu()
        group_list = request["group_list"].npu()
        reference = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x.clone(),
            [weight],
            [weight_scale],
            x_scale,
            group_list,
            dequant_mode=0,
            quant_mode=0,
            group_list_type=request["group_list_type"],
        )
        torch.npu.synchronize()
        outputs.append(tuple(tensor.cpu() for tensor in reference))

    _check_native_provider(torch_npu, artifacts / "native-provider.json")
    torch.save(outputs, artifacts / "native-outputs.pt")


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
