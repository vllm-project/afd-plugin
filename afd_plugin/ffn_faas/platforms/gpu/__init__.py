# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN backend for routed experts over DeepEP M2N.

The adapter and settings are CPU-only. The environment, FFN-only model and
backend import torch, vLLM and afdEP lazily, after the adapter has validated
the launch.
"""

# NPU parallel: afd_plugin/ffn_faas/platforms/npu/__init__.py (get_model
#   selects the one qualified NPU model).
# Reuse: NO - the GPU platform passes the model path to vLLM and lets the AFD
#   model registry select the FFN-only architecture.

GPU_DEEPEP_M2N_BACKEND = "gpu-deepep-m2n"
