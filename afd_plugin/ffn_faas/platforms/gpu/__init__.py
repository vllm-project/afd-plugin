# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN backends for DeepEP M2N and async NVSHMEM.

The adapters and settings are CPU-only. The environments, FFN models and
backends import torch, vLLM, afdEP and NVSHMEM lazily, after the adapter has
validated the launch.
"""

GPU_ASYNC_BACKEND = "gpu-async"
GPU_DEEPEP_M2N_BACKEND = "gpu-deepep-m2n"
