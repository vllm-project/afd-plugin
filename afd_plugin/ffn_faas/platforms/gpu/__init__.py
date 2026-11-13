# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU standalone FFN backend for routed experts over DeepEP M2N.

The adapter and settings are CPU-only. The environment, FFN-only model and
backend import torch, vLLM and afdEP lazily, after the adapter has validated
the launch.
"""

GPU_DEEPEP_M2N_BACKEND = "gpu-deepep-m2n"
