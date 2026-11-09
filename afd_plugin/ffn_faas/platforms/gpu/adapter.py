# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU DeepEP M2N CLI adaptation; importing this module remains CPU-only.

Imports neither torch, vLLM nor afdEP at module import: ``create_parser``
loads vLLM's CLI layer for the selected backend, and every check in
``configure`` runs before a device or process group exists, so a bad launch
exits with a usage error and allocates nothing. Native imports happen in the
backend factory.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from afd_plugin.config import DEEPEP_M2N_CONNECTOR
from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.gpu.settings import (
    BF16_PAYLOAD,
    FFN_GRAPH_MODE,
    GPUFFNConfig,
)
from afd_plugin.ffn_faas.platforms.launch import (
    configure_ffn_engine_args,
    create_ffn_engine_parser,
)

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/adapter.py::create_backend
#   (bootstraps the Ascend OPP environment, then imports the backend).
# Reuse: PARTIAL - the deferred-import factory shape is the same; GPU needs
#   no environment bootstrap before importing torch.
def create_backend(settings: GPUFFNConfig, args: Namespace) -> FFNBackend:
    """Import the GPU backend only when the host loop constructs it.

    Called by ``FFNModelRunner.run`` after the management endpoint is up, so
    torch, vLLM and afdEP load inside the launched F process and an import
    failure is reported as a FAILED service rather than a usage error.
    """
    from afd_plugin.ffn_faas.platforms.gpu.backend import DeepEPM2NFFNBackend

    return DeepEPM2NFFNBackend(settings, args)


def uses_cuda_graphs(args: Namespace) -> bool:
    """Return the qualified execution mode selected by the native engine CLI.

    ``--enforce-eager`` selects eager; otherwise the compilation config must
    name ``FULL_DECODE_ONLY``, the only graph mode the FFN graph cache
    serves. Any other mode, including vLLM's unset default, is a usage error.
    """
    if args.enforce_eager:
        return False
    compilation = args.compilation_config
    mode = (
        compilation.get("cudagraph_mode")
        if isinstance(compilation, dict)
        else getattr(compilation, "cudagraph_mode", None)
    )
    if getattr(mode, "name", mode) != FFN_GRAPH_MODE:
        raise ValueError(
            "GPU standalone FFN runs --enforce-eager or --compilation-config "
            f'\'{{"cudagraph_mode":"{FFN_GRAPH_MODE}"}}\''
        )
    return True


# NPU parallel: afd_plugin/ffn_faas/platforms/npu/adapter.py::AscendCAMAdapter.
# Reuse: YES - the shared launch module owns the native engine parser and the
#   F-only launcher checks. GPU additionally fixes TP 1 with expert parallel,
#   the execution mode and the lane count, and adopts Attention TP, capacity
#   and the bucket ladder from the M2N handshake.
class GPUDeepEPM2NAdapter:
    """Keep GPU, topology and connector options out of the host loop."""

    def create_parser(self) -> ArgumentParser:
        """Build vLLM's engine CLI with the F-only service's fixed defaults.

        F reads no request text, so the tokenizer is skipped, and non-local
        expert tensors are skipped before they are read, as the Worker-path
        recipes do.
        """
        parser = create_ffn_engine_parser()
        parser.set_defaults(skip_tokenizer_init=True, enable_ep_weight_filter=True)
        return parser

    def configure(self, args: Namespace) -> FFNLaunch:
        """Validate the launch on the CPU and describe the backend to build.

        ``additional_config["afd"]`` must select ``DeepEPM2NAFDConnector`` in
        the ``ffn`` role, and the ``torchrun`` ``WORLD_SIZE``,
        ``--data-parallel-size`` and ``afd.num_ffn_ranks`` must agree. TP is 1
        with expert parallelism, so vLLM's EP group is the F world. The model
        path is resolved without interpreting its architecture; vLLM and the
        AFD model registry own model selection. Raises ``ValueError`` or
        ``OSError``, which the entry point reports as a usage error.
        """
        afd, launcher = configure_ffn_engine_args(args, connector=DEEPEP_M2N_CONNECTOR)
        if args.tensor_parallel_size != 1 or not args.enable_expert_parallel:
            raise ValueError(
                "GPU standalone FFN requires --tensor-parallel-size 1 and "
                "--enable-expert-parallel"
            )
        model_path = Path(args.model).resolve(strict=True)
        args.model = str(model_path)
        settings = GPUFFNConfig(
            model=model_path,
            ffn_ranks=launcher.world_size,
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            graph=uses_cuda_graphs(args),
            dbo=bool(args.enable_dbo),
            connector_extra_config=MappingProxyType(dict(afd.connector_extra_config)),
            shutdown_timeout=args.receiver_shutdown_timeout,
        )
        return FFNLaunch(
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            model_path=str(model_path),
            backend_name=DEEPEP_M2N_CONNECTOR,
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, settings, args),
            initial_status={
                "payload_format": BF16_PAYLOAD,
                "num_lanes": settings.num_lanes,
                "graph_mode": settings.graph_mode,
                "dbo": settings.dbo,
                "m2n_paired": False,
            },
        )
