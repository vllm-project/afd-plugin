# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN backend: Qwen3 MoE routed experts over DeepEP M2N.

The adapter, settings and checkpoint description are CPU-only. The
environment, FFN-only model and backend import torch, vLLM and afdEP lazily,
after the adapter has validated the launch.
"""

# NPU parallel: afd_plugin/ffn_faas/platforms/npu/__init__.py (get_model
#   selects the one qualified NPU model).
# Reuse: NO - the GPU backend reads one checkpoint description directly and
#   needs no model-adapter selection step.

GPU_DEEPEP_M2N_BACKEND = "gpu-deepep-m2n"
