# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Three DSpark acceptance cases, each with its own same-topology reference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tests.e2e.case import (
    CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    NPU_ASYNC_TERMINATION_TIMEOUT_S,
    NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S,
    NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S,
    Case,
    CaseEnvironment,
    Deployment,
    Phase,
    Role,
)
from tests.e2e.environment import add_case_environment_arguments, case_environment
from tests.e2e.models.deepseek_v4_flash import dspark_eager
from tests.e2e.runner import prepare_case, run_case

EAGER = "afd-dspark-eager"
GRAPH = "afd-dspark-graph"
PD_GRAPH = "pd-afd-dspark-graph"
CASE_IDS = (EAGER, GRAPH, PD_GRAPH)
ROOT = Path(__file__).resolve().parents[4]
EXTENSION = "tests.e2e.models.deepseek_v4_flash.dspark_core_evidence.DSparkCoreEvidence"


@dataclass(frozen=True)
class DSparkTask:
    control_url: str
    evidence_dir: str | None
    speculative: bool
    graph: bool
    pd: bool
    graph_numerics: bool

    def resolve(self, environment):
        return self, ()

    def execute(self, *, base_url, model_name, checkpoint, output_path):
        args = [
            "--url",
            base_url,
            "--control-url",
            self.control_url,
            "--model",
            model_name,
            "--output-dir",
            str(output_path),
            "--attention-dp",
            "2",
            "--ffn-ranks",
            "4",
            "--strict-deltas",
        ]
        if self.evidence_dir:
            args += ["--evidence-dir", self.evidence_dir]
        if not self.speculative:
            args += ["--no-speculation"]
        if self.graph and self.speculative:
            args += ["--observe-graph"]
        if self.pd:
            args += ["--pd"]
        if self.graph_numerics:
            assert self.evidence_dir is not None
            directory = Path(self.evidence_dir) / "graph-proof"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "enable").touch()
            args += ["--verify-graph-numerics"]
        dspark_eager.main(args)
        return json.loads((output_path / "result.json").read_text())


def build_case(
    case_id: str, environment: CaseEnvironment, *, speculative=True, instrumented=False
) -> Case:
    pd = case_id == PD_GRAPH
    graph = case_id != EAGER
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    port = environment.api_port_base
    extra = tuple(
        (key, False)
        for key in (
            "enable_cpu_binding",
            "enable_force_eplb",
            "enable_dsa_cp",
            "multistream_dsv4_dsa_overlap",
        )
    )

    def spec(role):
        if not speculative:
            return ()
        config = {
            "method": "dspark",
            "model": environment.model,
            "num_speculative_tokens": 5,
            "enforce_eager": not (graph and role == "attention"),
        }
        return ("--speculative-config", json.dumps(config, separators=(",", ":")))

    def kv(role):
        if not pd:
            return ()
        config = {
            "kv_connector": "MooncakeHybridConnector",
            "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
            "kv_port": port + (1100 if role == "prefill" else 1200),
            "kv_connector_extra_config": {
                "prefill": {"tp_size": 4, "dp_size": 1},
                "decode": {"tp_size": 1, "dp_size": 2},
            },
        }
        return ("--kv-transfer-config", json.dumps(config, separators=(",", ":")))

    attention = Role(
        2,
        1,
        execution,
        vllm_args=(
            "--all2all-backend",
            "allgather_reducescatter",
            *spec("attention"),
            *kv("attention"),
        ),
        additional_config=extra,
        data_parallel_rpc_port=port + 10000,
        compilation_config=(
            ("mode", 0),
            ("cudagraph_mode", "FULL_DECODE_ONLY"),
            ("cudagraph_capture_sizes", (1, 6, 20, 24)),
        )
        if graph
        else (),
    )
    ffn = Role(
        4,
        1,
        execution,
        vllm_args=("--all2all-backend", "flashinfer_all2allv", *spec("ffn")),
        additional_config=extra,
        data_parallel_rpc_port=port + 10001,
        compilation_config=(("cudagraph_mode", "FULL"),) if graph else (),
    )
    prefill = (
        Role(
            1,
            4,
            EXECUTION_EAGER,
            vllm_args=(*spec("prefill"), *kv("prefill")),
            additional_config=extra,
            data_parallel_rpc_port=port + 10002,
        )
        if pd
        else None
    )
    evidence = str(environment.output_path / "workers") if speculative else None
    return Case(
        id=case_id
        + ("-instrumented" if instrumented else "")
        + ("-reference" if not speculative else ""),
        model="DeepSeek-V4-Flash-W4A8",
        served_model_name_prefix="dsv4-dspark",
        deployment=Deployment(
            backend="npu",
            attention=attention,
            ffn=ffn,
            prefill=prefill,
            connector="CAMAsyncAFDConnector",
            compute_gate_on_attention=True,
            connector_extra_config=(("dynamicQuant", 1), ("attn_ranks_per_dp", 1)),
            pin_attention_address=True,
            cudagraph_capture_size=24 if graph else None,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            cleanup_profile=CLEANUP_PROFILE_NPU_ASYNC_CAM_LEGACY,
            shutdown_timeout_s=NPU_ASYNC_VLLM_SHUTDOWN_TIMEOUT_S,
            worker_shutdown_timeout_s=NPU_ASYNC_WORKER_SHUTDOWN_TIMEOUT_S,
            termination_timeout_s=NPU_ASYNC_TERMINATION_TIMEOUT_S,
        ),
        common_vllm_args=(
            "--api-server-count",
            "1",
            "--seed",
            "1024",
            "--max-model-len",
            "4096",
            "--block-size",
            "128",
            "--gpu-memory-utilization",
            "0.85",
            "--quantization",
            "ascend",
            "--attention_config.indexer_kv_dtype",
            "int8",
            "--tokenizer-mode",
            "deepseek_v4",
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--no-disable-hybrid-kv-cache-manager",
            "--model-loader-extra-config",
            '{"enable_multithread_load":true,"num_threads":4}',
        ),
        phases=(
            Phase(
                "requests",
                DSparkTask(
                    control_url=f"http://{environment.api_host}:{port}",
                    evidence_dir=evidence,
                    speculative=speculative,
                    graph=graph,
                    pd=pd,
                    graph_numerics=instrumented,
                ),
            ),
        ),
        worker_extension=EXTENSION if speculative else None,
        worker_environment=tuple(
            (key, value)
            for key, value in (
                ("VLLM_SERVER_DEV_MODE", "1"),
                ("AFD_DSPARK_EVIDENCE_DIR", evidence or ""),
                ("AFD_DSPARK_GRAPH_NUMERICS", "1" if instrumented else "0"),
                ("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"),
                ("BATCH_SIZE_FACTOR", "0.00390625"),
            )
        ),
    )


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_digest(root: Path, paths: set[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(file_digest(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def package_digest(origin: Path) -> str:
    root = origin.resolve().parent
    return files_digest(
        root,
        {
            item
            for item in root.rglob("*")
            if item.is_file()
            and item.suffix != ".pyc"
            and "__pycache__" not in item.parts
        },
    )


def identity(environment: CaseEnvironment) -> dict:
    """Bind reusable reference to source, native binaries, model and runtime."""
    model = Path(environment.model).resolve()
    plugin = ROOT / "afd_plugin"
    source_files = {p for p in plugin.rglob("*.py") if p.is_file()}
    source_files.update(p for p in plugin.rglob("*.so") if p.is_file())
    source_files.update(p for p in (ROOT / "tests/e2e").rglob("*.py") if p.is_file())
    # A CANN vendor package also contains device kernels and metadata that are
    # not linked into libcust_opapi.so. Hash the installed package itself.
    source_files.update(
        p
        for p in (plugin / "_cann_ops_custom").rglob("*")
        if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
    )
    model_files: dict[str, dict[str, str | int]] = {}
    for path in sorted(model.iterdir()):
        if path.is_file():
            stat = path.stat()
            model_files[path.name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if path.suffix == ".json":
                model_files[path.name]["sha256"] = file_digest(path)
    runtime: dict[str, dict[str, Any]] = {}
    for name in ("vllm", "vllm_ascend"):
        found = importlib.util.find_spec(name)
        if found and found.origin:
            path = Path(found.origin).resolve()
            try:
                revision = subprocess.check_output(
                    ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                changes = subprocess.check_output(
                    ["git", "-C", str(path.parent), "status", "--porcelain"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError):
                revision, changes = None, None
            runtime[name] = {
                "path": str(path),
                "revision": revision,
                "changes": changes,
                # Git status names dirty files, but does not identify their
                # contents or installed packages outside a Git checkout.
                "content_sha256": package_digest(path),
            }
            runtime[name]["native"] = {
                str(p): file_digest(p) for p in path.parent.glob("*.so")
            }
    executable = shutil.which(environment.vllm_bin)
    return {
        "source_sha256": files_digest(ROOT, source_files),
        "model": str(model),
        "model_files": model_files,
        "runtime": runtime,
        "python": {"executable": sys.executable, "version": sys.version},
        "service_executable": {
            "command": environment.vllm_bin,
            "path": str(Path(executable).resolve()) if executable else None,
            "sha256": file_digest(Path(executable)) if executable else None,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "numpy")
        },
        "devices": list(environment.devices[:6]),
    }


def service_runtime_identity(stage: Path) -> dict:
    """Fingerprint interpreters and package contents observed by A/F workers."""
    requests = stage / "requests"
    roles = {
        "attention": json.loads((requests / "attention.json").read_text())["results"],
        "ffn": list(json.loads((requests / "ffn-after.json").read_text()).values()),
    }
    workers: dict[str, dict[str, Any]] = {}
    executables: dict[str, str] = {}
    packages: dict[str, str] = {}
    required_modules = {
        "afd_plugin",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "torch_npu",
        "transformers",
        "vllm",
        "vllm_ascend",
    }
    for role, states in roles.items():
        if not states:
            raise ValueError(f"no {role} worker runtime identities")
        workers[role] = {}
        for state in states:
            runtime = state["runtime"]
            executable = runtime["executable"]
            modules = runtime["modules"]
            if not required_modules.issubset(modules):
                raise ValueError(f"{role} worker is missing runtime module identities")
            rank = str(state["rank"])
            if rank in workers[role]:
                raise ValueError(f"duplicate {role} worker rank {rank}")
            workers[role][rank] = runtime
            if executable not in executables:
                executables[executable] = file_digest(Path(executable))
            for origin in modules.values():
                if origin not in packages:
                    path = Path(origin)
                    if not path.is_file():
                        raise ValueError(f"worker runtime module is missing: {path}")
                    packages[origin] = package_digest(path)
    return {"workers": workers, "executables": executables, "packages": packages}


def compare(reference: Path, target: Path) -> None:
    old = json.loads((reference / "requests/summary.json").read_text())
    new = json.loads((target / "requests/summary.json").read_text())
    if old != new:
        raise AssertionError(f"serial token IDs differ: {reference} vs {target}")


def subrun(case_id, env, name, *, speculative=True, instrumented=False):
    # Retain previous artifacts without allowing them to qualify a new run.
    stage = replace(env, output_path=env.output_path / name / uuid.uuid4().hex)
    result = run_case(
        build_case(case_id, stage, speculative=speculative, instrumented=instrumented),
        stage,
    )
    if not result["acceptance_passed"]:
        raise AssertionError(f"{name} did not qualify")
    return stage.output_path


def execute(case_id: str, env: CaseEnvironment, eager_reference: Path | None = None):
    env.output_path.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case_id,
        "configuration": {"classification": "qualified"},
        "acceptance_passed": False,
        "identity": identity(env),
        "stages": {},
    }
    (env.output_path / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        if case_id == EAGER:
            reference = subrun(EAGER, env, "reference", speculative=False)
            target = subrun(EAGER, env, "target")
            compare(reference, target)
            record["stages"] = {"reference": str(reference), "target": str(target)}
            record["service_runtime"] = service_runtime_identity(target)
        elif case_id == GRAPH:
            if eager_reference is None:
                nested = replace(env, output_path=env.output_path / "eager-reference")
                execute(EAGER, nested)
                eager_reference = nested.output_path
            proof = json.loads((eager_reference / "run.json").read_text())
            if (
                proof.get("case_id") != EAGER
                or not proof.get("acceptance_passed")
                or proof.get("identity") != record["identity"]
                or not isinstance(proof.get("service_runtime"), dict)
            ):
                raise ValueError("eager reference is not a matching qualified run")
            reference = Path(proof["stages"]["target"])
            target = subrun(GRAPH, env, "target")
            compare(reference, target)
            target_runtime = service_runtime_identity(target)
            if proof.get("service_runtime") != target_runtime:
                raise ValueError("eager reference has a different service runtime")
            instrumented = subrun(GRAPH, env, "layer-evidence", instrumented=True)
            compare(target, instrumented)
            if service_runtime_identity(instrumented) != target_runtime:
                raise ValueError("graph deployments have different service runtimes")
            record["stages"] = {
                "reference": str(reference),
                "target": str(target),
                "layer_evidence": str(instrumented),
            }
            record["service_runtime"] = target_runtime
        else:
            reference = subrun(PD_GRAPH, env, "reference", speculative=False)
            target = subrun(PD_GRAPH, env, "target")
            compare(reference, target)
            target_runtime = service_runtime_identity(target)
            instrumented = subrun(PD_GRAPH, env, "layer-evidence", instrumented=True)
            compare(target, instrumented)
            if service_runtime_identity(instrumented) != target_runtime:
                raise ValueError("PD graph deployments have different service runtimes")
            record["stages"] = {
                "reference": str(reference),
                "target": str(target),
                "layer_evidence": str(instrumented),
            }
            record["service_runtime"] = target_runtime
        record["acceptance_passed"] = True
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (env.output_path / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_case_environment_arguments(parser, CASE_IDS)
    parser.add_argument("--eager-reference", type=Path)
    args = parser.parse_args()
    env = case_environment(args)
    if args.describe:
        _, record = prepare_case(build_case(args.scenario, env), env)
        print(json.dumps(record, indent=2))
    else:
        execute(args.scenario, env, args.eager_reference)


if __name__ == "__main__":
    main()
