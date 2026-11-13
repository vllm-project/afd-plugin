# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU smoke for the standalone FFN service (``--backend gpu-deepep-m2n``).

Four GPUs: two synthetic Attention ranks built with the DeepEP M2N harness
(fake vLLM config, no model) and two FFN service ranks launched by
``torchrun -m afd_plugin.ffn_faas serve`` from a synthetic Qwen3 MoE
checkpoint. The checkpoint's experts are the harness's deterministic expert
weights (Qwen3-30B-A3B sizes, two layers), so each Attention rank checks the
combined output of every layer against the harness's dense MoE reference:
that covers the FFN-only model, vLLM's loader and EP weight filter, expert
placement, the M2N handshake, eager steps and a captured CUDA graph.

Teardown: a ``/stop`` on an idle FFN service cancels its control receive;
the service releases its DeepEP lanes together with Attention (a lane's
``destroy`` is a barrier over every M2N rank) and exits 0. When Attention
dies first, the FFN service must fail and exit non-zero.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("safetensors")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from tests.gpu.deepep_m2n_harness import (  # noqa: E402
    Checker,
    available_gpus,
    free_ports,
    make_connector,
    make_expert_weights,
    make_inputs,
    run_attention_step,
    stage_seed,
)
from tests.unit.fakes.vllm_config import (  # noqa: E402
    M2N_HIDDEN_SIZE,
    M2N_INTERMEDIATE_SIZE,
    M2N_NUM_EXPERTS,
    M2N_TOP_K,
    M2NTopology,
)

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = M2NTopology(ffn_size=2, attn_size=2)
NUM_LAYERS = 2
CAPACITY = 256
BUCKET = 64
EAGER_STEPS = 2
GRAPH_REPLAYS = 3
HOST = "127.0.0.1"
EXTRA_CONFIG = {"capacity_tokens_per_rank": CAPACITY}
# Short DeepEP and rendezvous timeouts for the Attention-first failure case.
PEER_EXIT_EXTRA_CONFIG = {**EXTRA_CONFIG, "cpu_timeout_s": 10, "gpu_timeout_s": 10}
PEER_EXIT_AFD_TIMEOUT_S = 30
STARTUP_TIMEOUT_S = 900
STOP_TIMEOUT_S = 300
POLL_S = 0.5
ATTENTION_EXIT_CODE = 23
MANAGEMENT_PREFIX = '{"rank"'


def _skip_without_gpus() -> None:
    if available_gpus() < TOPOLOGY.world_size:
        pytest.skip(f"needs {TOPOLOGY.world_size} GPUs")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DeepEP M2N needs SM90+")


def write_qwen3_m2n_checkpoint(root: Path) -> Path:
    """A BF16 Qwen3 MoE checkpoint whose experts are the harness weights.

    Every layer holds the same experts, as the harness's dense reference
    assumes. HF names: ``experts.<e>.{gate,up,down}_proj``; the harness's
    ``w13`` is ``[gate; up]``. The MoE gate is zero: F builds it, never runs it.
    """
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    w13, w2 = make_expert_weights(range(M2N_NUM_EXPERTS))
    w13, w2 = w13.cpu(), w2.cpu()
    tensors: dict[str, Any] = {}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate.weight"] = torch.zeros(
            M2N_NUM_EXPERTS, M2N_HIDDEN_SIZE, dtype=torch.bfloat16
        )
        for expert in range(M2N_NUM_EXPERTS):
            gate, up = w13[expert].chunk(2, dim=0)
            name = f"{prefix}.experts.{expert}"
            tensors[f"{name}.gate_proj.weight"] = gate.clone()
            tensors[f"{name}.up_proj.weight"] = up.clone()
            tensors[f"{name}.down_proj.weight"] = w2[expert].clone()
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "hidden_size": M2N_HIDDEN_SIZE,
        "intermediate_size": 4 * M2N_HIDDEN_SIZE,
        "moe_intermediate_size": M2N_INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": M2N_NUM_EXPERTS,
        "num_experts_per_tok": M2N_TOP_K,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(config))
    return root


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """One synthetic checkpoint (about 2.3 GB) shared by this module's tests."""
    _skip_without_gpus()
    return write_qwen3_m2n_checkpoint(tmp_path_factory.mktemp("qwen3-m2n"))


def _send_control(connector: Any, tokens: tuple[int, ...], **flags: bool) -> None:
    """Send one step's control payload and plan it, as Attention's runner does."""
    from afd_plugin.connectors import AFDControlPayload, AFDDPMetadata

    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(num_tokens_across_dp_cpu=torch.tensor(tokens))
        },
        is_graph_capturing=flags.get("is_graph_capturing", False),
        is_warmup=flags.get("is_warmup", False),
        is_graph_replaying=flags.get("is_graph_replaying", False),
    )
    connector.control_plane.send_dp_metadata_list(payload)
    connector.control_plane.update_state_from_dp_metadata(payload)


def attention_scenario(connector: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Drive the FFN service like vLLM's Attention runner and check combines.

    Eager steps first; with ``graph``, then Attention's capture sequence: a
    warmup payload run eagerly, a capture payload captured into a CUDA graph
    on both sides, and replays with fresh inputs. Every step runs all
    ``NUM_LAYERS`` MoE layers, as the FFN service runs every layer per
    payload. Returns the checker's worst errors and the payload count.
    """
    checker = Checker(connector)
    tokens = (BUCKET,) * TOPOLOGY.attn_size
    rank = connector.role_rank
    payloads = 0

    def inputs(step: int, layer: int) -> tuple[Any, Any, Any]:
        return make_inputs(stage_seed(step * NUM_LAYERS + layer, 0), rank, BUCKET)

    def eager_step(step: int, **flags: bool) -> None:
        _send_control(connector, tokens, **flags)
        for layer in range(NUM_LAYERS):
            x, ids, w = inputs(step, layer)
            out = torch.empty_like(x)
            run_attention_step(connector, [(x, ids, w)], [out], layer)
            checker.check_combined(out, x, ids, w)

    step = 0
    for _ in range(args["eager_steps"]):
        eager_step(step)
        step += 1
        payloads += 1
    if args["graph"]:
        eager_step(step, is_warmup=True)
        step += 1
        payloads += 1
        statics = [inputs(step, layer) for layer in range(NUM_LAYERS)]
        outs = [torch.empty_like(x) for x, _, _ in statics]
        _send_control(connector, tokens, is_graph_capturing=True)
        payloads += 1
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for layer, (static, out) in enumerate(zip(statics, outs, strict=True)):
                run_attention_step(connector, [static], [out], layer)
        torch.cuda.synchronize()
        for _ in range(GRAPH_REPLAYS):
            step += 1
            for layer, static in enumerate(statics):
                for static_tensor, fresh in zip(
                    static, inputs(step, layer), strict=True
                ):
                    static_tensor.copy_(fresh)
            _send_control(connector, tokens, is_graph_replaying=True)
            graph.replay()
            torch.cuda.synchronize()
            payloads += 1
            for (x, ids, w), out in zip(statics, outs, strict=True):
                checker.check_combined(out, x, ids, w)
    return {**checker.finish(), "control_payloads": payloads}


def _attention_rank_main(
    role_rank: int,
    args: dict[str, Any],
    extra_config: dict[str, Any],
    ports: tuple[int, int],
    out_dir: str,
) -> None:
    """One synthetic Attention rank: connect, run the scenario, then release.

    The default process group spans only the Attention ranks (the FFN ranks
    are another launcher's world). After the scenario the rank writes a
    ``scenario`` marker, then either exits without cleanup (``exit_early``,
    simulating an Attention failure) or waits for the test's ``release``
    marker and closes its connector together with the FFN service.
    """
    out = Path(out_dir)
    result: dict[str, Any] = {"ok": True, "error": "", "metrics": {}}
    connector = None
    try:
        torch.cuda.set_device(role_rank)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://{HOST}:{ports[0]}",
            world_size=TOPOLOGY.attn_size,
            rank=role_rank,
        )
        connector = make_connector(
            TOPOLOGY,
            TOPOLOGY.ffn_size + role_rank,
            role_rank,
            extra_config,
            ports[1],
            HOST,
        )
        connector.init_afd_connector()
        result["metrics"] = attention_scenario(connector, args)
        (out / f"scenario{role_rank}").write_text(json.dumps(result["metrics"]))
        if args["exit_early"]:
            os._exit(ATTENTION_EXIT_CODE)
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while not (out / "release").exists() and time.monotonic() < deadline:
            time.sleep(POLL_S)
    except Exception:
        result["ok"] = False
        result["error"] = traceback.format_exc()
        (out / f"scenario{role_rank}").write_text("{}")
    finally:
        if connector is not None:
            try:
                connector.close()
            except Exception:
                result["ok"] = False
                result["error"] += traceback.format_exc()
        (out / f"rank{role_rank}.json").write_text(json.dumps(result, default=str))
        if dist.is_initialized():
            dist.destroy_process_group()


def _start_attention(
    args: dict[str, Any],
    extra_config: dict[str, Any],
    ports: tuple[int, int],
    out: Path,
) -> list[Any]:
    """Spawn the Attention ranks on the first two visible GPUs."""
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_attention_rank_main,
            args=(rank, args, extra_config, ports, str(out)),
        )
        for rank in range(TOPOLOGY.attn_size)
    ]
    for proc in procs:
        proc.start()
    return procs


def _ffn_devices() -> str:
    """The visible GPUs after the Attention ranks', as a CUDA_VISIBLE_DEVICES value."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = (
        [item.strip() for item in visible.split(",") if item.strip()]
        if visible
        else [str(index) for index in range(available_gpus())]
    )
    return ",".join(ids[TOPOLOGY.attn_size : TOPOLOGY.world_size])


def _start_ffn_service(
    checkpoint: Path,
    *,
    afd_port: int,
    master_port: int,
    status_file: Path,
    log_path: Path,
    graph: bool,
    extra_config: dict[str, Any],
    afd_timeout_s: int,
) -> subprocess.Popen[bytes]:
    """Launch both FFN service ranks with torchrun, as an operator would."""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": _ffn_devices(),
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_PLUGINS": "afd",
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH")))
        ),
    }
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={TOPOLOGY.ffn_size}",
        "--nnodes=1",
        f"--master-addr={HOST}",
        f"--master-port={master_port}",
        "-m",
        "afd_plugin.ffn_faas",
        "serve",
        str(checkpoint),
        "--backend",
        "gpu-deepep-m2n",
        "--data-parallel-size",
        str(TOPOLOGY.ffn_size),
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
        "--additional-config",
        json.dumps(
            {
                "afd": {
                    "role": "ffn",
                    "connector": "DeepEPM2NAFDConnector",
                    "host": HOST,
                    "port": afd_port,
                    "num_attention_ranks": TOPOLOGY.attn_size,
                    "num_ffn_ranks": TOPOLOGY.ffn_size,
                    "compute_gate_on_attention": True,
                    "afd_process_group_timeout_s": afd_timeout_s,
                    "connector_extra_config": extra_config,
                }
            }
        ),
        "--status-host",
        HOST,
        # Port 0: each rank binds a free port and prints it.
        "--status-port",
        "0",
        "--status-file",
        str(status_file),
        *(
            ["--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}']
            if graph
            else ["--enforce-eager"]
        ),
    ]
    with log_path.open("wb") as log:
        return subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _management_ports(log_path: Path) -> dict[int, int]:
    """Management port per FFN rank, from the object each rank prints at start.

    Both ranks share one log, so their announcements can land on one line.
    """
    text = log_path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    ports = {}
    start = text.find(MANAGEMENT_PREFIX)
    while start >= 0:
        announced, end = decoder.raw_decode(text, start)
        ports[int(announced["rank"])] = int(announced["management"][1])
        start = text.find(MANAGEMENT_PREFIX, end)
    return ports


def _wait_for(predicate: Any, timeout_s: float, what: str, *procs: Any) -> None:
    """Poll ``predicate`` until true; fail early if a watched process died."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        for proc in procs:
            code = proc.poll() if hasattr(proc, "poll") else proc.exitcode
            if code is not None:
                raise AssertionError(f"process exited with {code} while waiting {what}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting {what}")
        time.sleep(POLL_S)


def _status(port: int, path: str = "/status", method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(f"http://{HOST}:{port}{path}", method=method)
    with urllib.request.urlopen(request, timeout=5) as reply:
        return json.loads(reply.read())


def _status_files(status_file: Path) -> dict[int, dict[str, Any]]:
    files = {}
    for rank in range(TOPOLOGY.ffn_size):
        path = status_file.with_name(f"{status_file.stem}-rank-{rank}.json")
        if path.exists():
            files[rank] = json.loads(path.read_text())
    return files


def _descendants(pid: int) -> list[int]:
    """Every live descendant of ``pid``, read from ``/proc`` (Linux only).

    torchrun's workers are not in the launcher's process group, so killing
    the group alone leaves them holding GPUs.
    """
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            fields = stat.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    found, pending = [], [pid]
    while pending:
        for child in children.get(pending.pop(), []):
            found.append(child)
            pending.append(child)
    return found


def _kill(service: subprocess.Popen[bytes], attention: list[Any]) -> None:
    """Best-effort teardown so a failed test does not leak GPU processes."""
    if service.poll() is None:
        for pid in [*_descendants(service.pid), service.pid]:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        service.wait(timeout=30)
    for proc in attention:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=30)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_ffn_service_serves_and_stops_cleanly(  # long-test: two launchers
    checkpoint, tmp_path, graph
):
    attention_args = {"eager_steps": EAGER_STEPS, "graph": graph, "exit_early": False}
    group_port, afd_port, master_port = free_ports(3)
    status_file = tmp_path / "ffn.json"
    log_path = tmp_path / "ffn.log"
    service = _start_ffn_service(
        checkpoint,
        afd_port=afd_port,
        master_port=master_port,
        status_file=status_file,
        log_path=log_path,
        graph=graph,
        extra_config=EXTRA_CONFIG,
        afd_timeout_s=STARTUP_TIMEOUT_S,
    )
    attention = _start_attention(
        attention_args, EXTRA_CONFIG, (group_port, afd_port), tmp_path
    )
    try:
        _wait_for(
            lambda: all(
                (tmp_path / f"scenario{rank}").exists()
                for rank in range(TOPOLOGY.attn_size)
            ),
            STARTUP_TIMEOUT_S,
            "for the Attention scenario",
            service,
            *attention,
        )
        metrics = [
            json.loads((tmp_path / f"scenario{rank}").read_text())
            for rank in range(TOPOLOGY.attn_size)
        ]
        assert all("control_payloads" in rank_metrics for rank_metrics in metrics), (
            (tmp_path / "rank0.json").read_text()
            if (tmp_path / "rank0.json").exists()
            else metrics
        )
        payloads = metrics[0]["control_payloads"]
        ports = _management_ports(log_path)
        assert sorted(ports) == list(range(TOPOLOGY.ffn_size))
        # A step is counted after its device work completes.
        _wait_for(
            lambda: all(
                _status(port)["completed_batches"] == payloads
                for port in ports.values()
            ),
            STOP_TIMEOUT_S,
            "for the FFN service to count every step",
            service,
        )
        experts_per_rank = M2N_NUM_EXPERTS // TOPOLOGY.ffn_size
        for rank, port in ports.items():
            status = _status(port)
            assert status["phase"] == "ready"
            assert status["m2n_paired"] is True
            assert status["compute_ep_world_size"] == TOPOLOGY.ffn_size
            assert status["capacity_tokens_per_rank"] == CAPACITY
            assert status["attn_tp_size"] == 1
            assert (status["local_expert_first"], status["local_expert_last"]) == (
                rank * experts_per_rank,
                (rank + 1) * experts_per_rank - 1,
            )
            assert status["loaded_experts"] == status["total_experts"]
            assert status["total_experts"] == NUM_LAYERS * experts_per_rank

        for port in ports.values():
            _status(port, "/stop", "POST")
        (tmp_path / "release").write_text("")
        assert service.wait(timeout=STOP_TIMEOUT_S) == 0, log_path.read_text()[-4000:]
        for proc in attention:
            proc.join(timeout=STOP_TIMEOUT_S)
        results = [
            json.loads((tmp_path / f"rank{rank}.json").read_text())
            for rank in range(TOPOLOGY.attn_size)
        ]
        assert all(result["ok"] for result in results), results
        files = _status_files(status_file)
        assert sorted(files) == list(range(TOPOLOGY.ffn_size))
        for status in files.values():
            assert status["phase"] == "stopped"
            assert status["resources_released"] is True
    finally:
        _kill(service, attention)


def test_ffn_service_fails_when_attention_dies_first(checkpoint, tmp_path):
    attention_args = {"eager_steps": 1, "graph": False, "exit_early": True}
    group_port, afd_port, master_port = free_ports(3)
    status_file = tmp_path / "ffn.json"
    log_path = tmp_path / "ffn.log"
    service = _start_ffn_service(
        checkpoint,
        afd_port=afd_port,
        master_port=master_port,
        status_file=status_file,
        log_path=log_path,
        graph=False,
        extra_config=PEER_EXIT_EXTRA_CONFIG,
        afd_timeout_s=PEER_EXIT_AFD_TIMEOUT_S,
    )
    attention = _start_attention(
        attention_args, PEER_EXIT_EXTRA_CONFIG, (group_port, afd_port), tmp_path
    )
    try:
        for proc in attention:
            proc.join(timeout=STARTUP_TIMEOUT_S)
            assert proc.exitcode == ATTENTION_EXIT_CODE
        code = service.wait(timeout=STOP_TIMEOUT_S)
        assert code != 0, log_path.read_text()[-4000:]
        for status in _status_files(status_file).values():
            assert status["resources_released"] is False
    finally:
        _kill(service, attention)
