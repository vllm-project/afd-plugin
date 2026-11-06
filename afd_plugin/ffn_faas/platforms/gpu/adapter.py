# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only launch adaptation for the GPU DeepEP M2N FFN service.

Imports neither torch, vLLM nor afdEP: every check here runs before a device
or process group exists, so a bad launch exits with a usage error and
allocates nothing. Native imports happen in the backend factory.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from functools import partial
from typing import TYPE_CHECKING

from afd_plugin.config import DEEPEP_M2N_CONNECTOR
from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.gpu.qwen3_model import Qwen3CheckpointDescription
from afd_plugin.ffn_faas.platforms.gpu.settings import (
    BF16_PAYLOAD,
    DEFAULT_AFD_PORT,
    DEFAULT_AFD_TIMEOUT_S,
    GPUFFNConfig,
    parse_connector_extra_config,
)
from afd_plugin.ffn_faas.platforms.launch import (
    read_ffn_launcher_environment,
    require_v2_model_runner,
)

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/adapter.py::create_backend
#   (bootstraps the Ascend OPP environment, then imports the backend).
# Reuse: PARTIAL - the deferred-import factory shape is the same; GPU needs
#   no environment bootstrap before importing torch.
def create_backend(
    checkpoint: Qwen3CheckpointDescription, settings: GPUFFNConfig
) -> FFNBackend:
    """Import the GPU backend only when the host loop constructs it.

    Called by ``FFNModelRunner.run`` after the management endpoint is up, so
    torch, vLLM and afdEP load inside the launched F process and an import
    failure is reported as a FAILED service rather than a usage error.
    """
    from afd_plugin.ffn_faas.platforms.gpu.backend import DeepEPM2NFFNBackend

    return DeepEPM2NFFNBackend(checkpoint, settings)


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/adapter.py::AscendCAMAdapter.
# Reuse: YES - the shared launch module owns the F-only torchrun and
#   ModelRunnerV2 checks. The option set still differs because GPU adopts
#   Attention TP, capacity and the bucket ladder from the M2N handshake.
class GPUDeepEPM2NAdapter:
    """Keep GPU, topology and connector options out of the host loop."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register the GPU backend's options on the shared ``serve`` parser.

        ``--host``/``--port`` must equal Attention's ``afd.host``/``afd.port``,
        ``--attention-ranks``/``--ffn-ranks`` its rank counts, ``--dbo`` its
        ``--enable-dbo`` and ``--connector-extra-config`` its must-match
        connector keys. There is no Attention TP, batch or capture-size flag:
        F learns them from the M2N handshake and control payloads.
        """
        parser.add_argument("--host", default="127.0.0.1", help="AFD rendezvous host")
        parser.add_argument(
            "--port", type=int, default=DEFAULT_AFD_PORT, help="AFD rendezvous port"
        )
        parser.add_argument("--attention-ranks", type=int, required=True)
        parser.add_argument("--ffn-ranks", type=int, required=True)
        parser.add_argument(
            "--graph",
            action="store_true",
            help="capture FULL_DECODE_ONLY CUDA graphs on F (default: eager)",
        )
        parser.add_argument(
            "--dbo",
            action="store_true",
            help="two DeepEP lanes; required when Attention runs --enable-dbo",
        )
        parser.add_argument(
            "--connector-extra-config",
            default="{}",
            help="DeepEPM2NAFDConnector options as a JSON object",
        )
        parser.add_argument(
            "--afd-timeout",
            type=int,
            default=DEFAULT_AFD_TIMEOUT_S,
            help="afd_process_group_timeout_s for the M2N rendezvous",
        )

    def configure(self, args: Namespace) -> FFNLaunch:
        """Validate the launch on the CPU and describe the backend to build.

        Reads this rank from the ``torchrun`` environment and requires the
        F-only world (``WORLD_SIZE == --ffn-ranks``) and
        ``VLLM_USE_V2_MODEL_RUNNER=1`` (Attention runs ModelRunnerV2 and the
        FFN runner validates the same deployment). Parses the connector JSON
        structurally, reads the checkpoint description, and checks that the
        experts divide across the F ranks. Raises ``ValueError`` or
        ``OSError``, which the entry point reports as a usage error.
        """
        launcher = read_ffn_launcher_environment(args.ffn_ranks)
        require_v2_model_runner()
        settings = GPUFFNConfig(
            model=args.model,
            attention_ranks=args.attention_ranks,
            ffn_ranks=args.ffn_ranks,
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            host=args.host,
            port=args.port,
            graph=args.graph,
            dbo=args.dbo,
            connector_extra_config=parse_connector_extra_config(
                args.connector_extra_config
            ),
            afd_timeout_s=args.afd_timeout,
            shutdown_timeout=args.shutdown_timeout,
        )
        checkpoint = Qwen3CheckpointDescription.read(args.model)
        checkpoint.local_experts(settings.ffn_ranks)
        return FFNLaunch(
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            model_path=str(checkpoint.root),
            backend_name=DEEPEP_M2N_CONNECTOR,
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, checkpoint, settings),
            initial_status={
                "payload_format": BF16_PAYLOAD,
                "num_lanes": settings.num_lanes,
                "graph_mode": settings.graph_mode,
                "dbo": settings.dbo,
                "m2n_paired": False,
            },
        )
