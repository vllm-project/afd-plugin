# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Executor-side state for CPU KV-cache configuration contracts."""

from copy import deepcopy


class KVCacheExecutor:
    max_concurrent_batches = 1

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.kv_cache_specs = []
        self.kv_cache_layout = None
        self.kv_cache_configs = []
        self.warmup_called = False

    def get_kv_cache_specs(self):
        return self.kv_cache_specs

    def get_supported_kv_cache_layouts(self):
        return ["layered"]

    def set_kv_cache_layout(self, layout_name):
        self.kv_cache_layout = layout_name

    def initialize_from_config(self, kv_cache_configs):
        # Workers receive a serialized configuration, not a live scheduler reference.
        self.kv_cache_configs = deepcopy(kv_cache_configs)

    def compile_or_warm_up_model(self):
        self.warmup_called = True

    def shutdown(self):
        self.shutdown_called = True
