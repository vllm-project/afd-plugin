# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit one-NPU AFD integration test; MR837 local transport, real compute.

Run with MR837 test-output activated and its examples/models/mega_ffn on
PYTHONPATH. No production communication resources are constructed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from blockrt.models.mega_ffn.cam_service import CamServiceConfig
from ffn_test_support import mxfp8_reference as reference
from ffn_test_support import windows as local
from ffn_test_support.fixture import LOCAL_EXPERTS, TOPK, Geometry, source_fixture
from ffn_test_support.service_local import (
    Epoch,
    merge_publications,
    publish_epoch,
    reset_scratch,
    verify_epoch,
)

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.connectors.npu.async_cam_runtime.async_cam_startup import (
    resolve_ffn_mode,
)
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from tests.unit.fakes.vllm_config import npu_vllm_config


class LocalServiceConnector(CAMAsyncAFDConnector):
    """Replace only transport setup/publication/return for this explicit test."""

    def init_afd_connector(self):
        self.graph_startup.mode = resolve_ffn_mode([self.graph_startup.plan])
        self._initialized = True

    def recv_attn_output(self, *args, **kwargs):
        raise AssertionError("Service Graph must own DispatchRecv")

    def send_ffn_output(self, *args, **kwargs):
        raise AssertionError("Service Graph must own CombineSend")

    def publish(self, epoch):
        if self.pending is not None:
            raise RuntimeError("previous mock epoch has not been consumed")
        publish_epoch(torch, self.owner, epoch)
        self.pending = epoch

    def collect(self, runner, expected, geometry, budgets):
        epoch = self.pending
        if epoch is None:
            raise RuntimeError("no mock request is pending")
        status = runner.service_diagnostics(include_tasks=True)
        records, _ = verify_epoch(
            torch, runner, self.owner, epoch, expected, geometry, budgets, status
        )
        # verify_epoch checks every token/top-k return, source ACK and metadata.
        self.pending = None
        return dict(
            epoch=epoch.epoch,
            records=records,
            status=status,
            sources=[
                dict(
                    phase=f.phase,
                    layer=f.layer_id,
                    microbatch=f.microbatch_idx,
                    round=f.round_idx,
                )
                for f in epoch.fixtures
            ],
        )

    def close(self):
        if self.pending is not None:
            raise RuntimeError("cannot release unconsumed mock return windows")
        self.owner = None
        self._session.release()
        self._initialized = False


def loaded_model(layer_ids, hidden, intermediate, device):
    """Deterministic BF16 post-load matrices, distinct for every layer/expert."""
    layers = {}
    for layer in layer_ids:
        w13 = (
            torch.stack(
                [
                    reference.deterministic_weight(
                        layer, e, "gate_up", 2 * intermediate, hidden
                    ).T
                    for e in range(LOCAL_EXPERTS)
                ]
            )
            .contiguous()
            .to(device)
        )
        w2 = (
            torch.stack(
                [
                    reference.deterministic_weight(
                        layer, e, "down", hidden, intermediate
                    ).T
                    for e in range(LOCAL_EXPERTS)
                ]
            )
            .contiguous()
            .to(device)
        )
        layers[layer] = SimpleNamespace(
            experts=SimpleNamespace(
                routed_experts=SimpleNamespace(
                    w13_weight=w13,
                    w2_weight=w2,
                    dynamic_eplb=False,
                    activation=SimpleNamespace(value="silu"),
                ),
                _shared_experts=None,
                moe_config=SimpleNamespace(
                    swiglu_limit=0,
                    swiglu_alpha=1,
                    swiglu_beta=0,
                    activation_situ_beta=None,
                    activation_situ_linear_beta=None,
                ),
            )
        )
    return SimpleNamespace(
        get_experts_layer_indices=lambda: layer_ids,
        get_mxfp_expert_layers=lambda: layers,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    report = dict(
        passed=False,
        scope="AFD runner + local mock transport + real Device compute",
        real_cross_card=False,
        full_vllm_server=False,
        layers=args.layers,
    )
    runner = service = None
    try:
        manifest = local.verified_home(args.home, args.home / "ffn-local-manifest.json")
        report["manifest"] = manifest["sha256"]
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        device = torch.device("npu:0")
        layer_ids = tuple(range(args.layers))
        hidden, intermediate = 512, 256
        geometry = Geometry(
            hidden=hidden, capacity=32, block_dim=28, layer_ids=layer_ids
        )
        config = npu_vllm_config(
            role="ffn",
            connector="CAMAsyncAFDConnector",
            async_dp=True,
            compute_gate_on_attention=True,
            num_attention_ranks=2,
            num_ffn_ranks=1,
        )
        config.model_config.dtype = torch.bfloat16
        hf = config.model_config.hf_config
        hf.hidden_size, hf.n_routed_experts, hf.num_experts_per_tok = (
            hidden,
            LOCAL_EXPERTS,
            TOPK,
        )
        config.cache_config = SimpleNamespace(num_gpu_blocks=None)
        config.compilation_config.static_forward_context = {}
        connector = LocalServiceConnector(0, 0, config, parse_afd_config(config), 0)
        connector.owner = local.windows(torch, geometry, device)
        connector.pending = None
        runner = AsyncNPUFFNModelRunner(config, device, connector)
        model = loaded_model(layer_ids, hidden, intermediate, device)
        runner.attach_model(
            model,
            parameter_bytes=args.layers * LOCAL_EXPERTS * hidden * intermediate * 6,
        )
        service_config = CamServiceConfig(
            layer_ids=layer_ids,
            local_experts=LOCAL_EXPERTS,
            max_tokens=geometry.max_tokens,
            max_padded=geometry.max_rows,
            p_budget=3,
            d_budget=1,
            task_slots=192,
            block_dim=28,
            batch_strategy="budget",
            aicpu_core_counts=(2, 13, 13),
            batch_wait_ns=0,
        )
        runner.build_megaffn_service(connector.owner, service_config, "")
        service = runner._service_runner
        bank = service.model.weights
        local.validate_private_windows(connector.owner, bank)
        # Golden reads the original logical layer/expert, not Device-selected IDs.
        # Verify packing against original BF16 matrices separately from GMM.
        for slot, layer in enumerate(layer_ids):
            routed = model.get_mxfp_expert_layers()[layer].experts.routed_experts
            for e in range(LOCAL_EXPERTS):
                for packed, scales, source in (
                    (bank.w13, bank.w13_scale, routed.w13_weight),
                    (bank.w2, bank.w2_scale, routed.w2_weight),
                ):
                    decoded = reference.dequantize_mx(
                        reference.unpack_nz(packed[slot * LOCAL_EXPERTS + e]),
                        scales[slot * LOCAL_EXPERTS + e],
                    )
                    original = source[e].cpu().float().T
                    relative = float(
                        torch.linalg.vector_norm(decoded - original)
                        / torch.linalg.vector_norm(original)
                    )
                    if relative > 0.05:
                        raise AssertionError(
                            f"packed layer/expert mismatch: {layer}/{e}: {relative}"
                        )
        sequence = (0, 2, 1, 0) if args.layers == 3 else (0, 0)
        epochs = []
        for index, layer in enumerate(sequence):
            fixtures = tuple(
                source_fixture(
                    geometry,
                    phase=phase,
                    layer_slot=(layer if phase == 0 else (layer + 1) % args.layers),
                    txn_id=2 * index + phase + 1,
                    epoch=index,
                    tokens=((22, 5), (9, 1), (31, 7), (16, 3))[index][phase],
                )
                for phase in range(2)
            )
            payload, control = merge_publications(geometry, fixtures)
            epochs.append(Epoch(index + 1, fixtures, payload, control))
        if args.layers == 3:
            # A two-round P request interleaved with two independent D requests.
            # Return rows are round-local; the A-side CombineRecv is not mocked here.
            for round_idx in range(2):
                index = len(epochs)
                fixtures = (
                    source_fixture(
                        geometry,
                        phase=0,
                        layer_slot=2,
                        txn_id=9,
                        epoch=index,
                        tokens=(32, 7)[round_idx],
                        round_idx=round_idx,
                        num_comm_rounds=2,
                    ),
                    source_fixture(
                        geometry,
                        phase=1,
                        layer_slot=round_idx,
                        txn_id=10 + round_idx,
                        epoch=index,
                        tokens=round_idx + 1,
                    ),
                )
                payload, control = merge_publications(geometry, fixtures)
                epochs.append(Epoch(index + 1, fixtures, payload, control))
        expected = {}
        for epoch in epochs:
            for fixture in epoch.fixtures:
                key = (fixture.layer_slot, fixture.phase)
                # Union active experts for repeated layers with different token counts.
                table = expected.setdefault(key, {})
                table.update(local.golden(torch, reference, bank, fixture, device))
        torch.npu.current_stream().synchronize()
        pointers = tuple(t.data_ptr() for t in service._service_state["windows"])
        runner.initialize_afd_connector()
        results = []
        for epoch in epochs:
            connector.publish(epoch)
            reset_scratch(torch, service, connector.owner)
            if not runner.execute_connector_driven_step():
                raise AssertionError("AFD stopped before completing its epoch")
            results.append(connector.collect(service, expected, geometry, (3, 1)))
            if (
                tuple(t.data_ptr() for t in service._service_state["windows"])
                != pointers
            ):
                raise AssertionError(
                    "fixed buffers or WeightBank changed across requests"
                )
            print(
                f"AFD_SERVICE_PASS epoch={epoch.epoch} "
                f"layers={[f.layer_id for f in epoch.fixtures]}",
                flush=True,
            )
        if runner.execute_connector_driven_step(shutdown_requested=True):
            raise AssertionError("AFD ignored local shutdown")
        runner.drain()
        runner.session.finish_receiver()
        runner.mark_receiver_exited()
        runner.shutdown()
        life = service.service_lifecycle()
        assert life["persistent_starts"] == 1 and life["task_submissions"] == len(
            epochs
        )
        assert life["stops"] == 1 and life["stop_verified"] and not life["running"]
        assert not connector.is_initialized and connector.owner is None
        report.update(
            passed=True,
            epochs=results,
            lifecycle=life,
            weight_keys=bank.group_keys,
            fixed_addresses_reused=True,
            production_external_dr_cs_called=False,
        )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if service is not None and service.service_lifecycle()["running"]:
            try:
                service.stop_persistent()
                report["failure_stop"] = service.service_lifecycle()
            except BaseException as error:
                report["stop_error"] = str(error)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
