# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One real CAM A peer for a separately launched routed-only FFN service.

This finite functional check uses the original checkpoint on F and every main
MoE layer. It has no Attention model, KV cache, HTTP inference or performance
claim. Run an eager F service first with --output, then FULL/two-stream F with
--reference pointing to that output. Terminate the whole paired deployment
after this client: the production service has no EOS or cancellable DR.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available  # noqa: E402
from afd_plugin.config import parse_afd_config  # noqa: E402
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector  # noqa: E402
from afd_plugin.ffn_faas.settings import FFNConfig  # noqa: E402
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription  # noqa: E402


def attention_config(checkpoint, settings):
    additional = settings.additional_config()
    additional["afd"]["role"] = "attention"
    return SimpleNamespace(
        additional_config=additional,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=True,
            hf_config=SimpleNamespace(
                hidden_size=checkpoint.hidden_size,
                num_experts_per_tok=checkpoint.top_k,
                n_routed_experts=checkpoint.num_experts,
                num_hidden_layers=checkpoint.model_layers,
                architectures=["DeepseekV4ForCausalLM"],
                model_type="deepseek_v4",
            ),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=settings.max_num_batched_tokens
        ),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            use_ubatching=False,
            num_ubatches=2,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
        ),
        speculative_config=None,
        use_v2_model_runner=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ffn-ranks", type=int, default=4)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--hold", action="store_true", help="retain CAM owners until controller exit"
    )
    args = parser.parse_args()
    checkpoint = V4CheckpointDescription.read(args.model)
    settings = FFNConfig(
        args.model,
        attention_ranks=1,
        attention_tp=1,
        ffn_ranks=args.ffn_ranks,
        port=args.port,
        max_num_batched_tokens=32,
        max_num_seqs=8,
        hccl_buffer_size=64,
    )
    config = attention_config(checkpoint, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ensure_cam_async_ops_available()
    torch.npu.set_device(args.device)
    connector = CAMAsyncAFDConnector(
        0, args.device, config, parse_afd_config(config), role_rank=0
    )
    records: list[dict[str, int]] = []
    report = {
        "phase": "connecting",
        "layers": records,
        "performance_measured": False,
    }
    try:
        connector.init_afd_connector()
        report["mode"] = connector.graph_startup.mode
        references = (
            torch.load(args.reference, weights_only=True)
            if args.reference is not None
            else None
        )
        outputs = []
        # Every layer is selected by the existing device header. Alternating
        # routes include zero local rows on other F ranks and capacity tails.
        for index, layer in enumerate(
            (*checkpoint.layer_indices, checkpoint.layer_indices[0])
        ):
            tokens = (1, 7, 17)[layer % 3]
            inputs = (
                torch.linspace(-2, 2, checkpoint.hidden_size)
                .sin()
                .repeat(tokens, 1)
                .to(device=f"npu:{args.device}", dtype=torch.bfloat16)
            )
            stride = checkpoint.num_experts // checkpoint.top_k
            ids = torch.arange(checkpoint.top_k, dtype=torch.int32)
            if layer % 2:
                ids = (ids * stride + layer) % checkpoint.num_experts
            ids = ids.repeat(tokens, 1).to(inputs.device)
            weights = torch.full(
                (tokens, checkpoint.top_k),
                checkpoint.routed_scaling_factor / checkpoint.top_k,
                device=inputs.device,
            )
            context = AFDTransferContext(AFDTransferMetadata(layer, 0, [tokens]))
            with torch.inference_mode():
                connector.send_attn_output(
                    inputs, context, topk_ids=ids, topk_weights=weights
                )
                result = connector.recv_ffn_output(inputs, context=context)
                done = torch.npu.Event()
                done.record()
                connector.wait_graph_device(done)
                actual = result.cpu()
            assert torch.isfinite(actual).all(), f"nonfinite layer {layer}"
            if references is not None:
                torch.testing.assert_close(actual, references[layer], rtol=0, atol=0)
            if index < checkpoint.model_layers:
                outputs.append(actual)
                records.append({"layer": layer, "tokens": tokens})
            else:
                # This extra completed transaction proves every F entered its
                # next fixed-size batch. With no more A work, its remaining DRs
                # stay pending: stop must retain owners, rather than race a
                # legitimate exit at the preceding completed-batch boundary.
                report["pending_batch_probe"] = {"layer": layer, "tokens": tokens}
        if references is not None:
            assert len(references) == len(outputs)
        torch.save(outputs, args.output)
        report.update(phase="passed", exact_eager_match=references is not None)
    except BaseException as error:
        report.update(phase="failed", error=repr(error))
        raise
    finally:
        # Do not close CAM while F may already have its next DR queued. The
        # test controller tears down both roles together, just as deployment
        # process isolation must do after a pending production receive.
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    if args.hold:
        threading.Event().wait()


if __name__ == "__main__":
    main()
