# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import asyncio
import copy
from types import SimpleNamespace

from afd_plugin.config import parse_afd_config
from afd_plugin.elastic.config import ElasticTopology
from afd_plugin.elastic.ffn import FFNActorManager


def test_ffn_configs_are_independent_and_initialize_collectively(config):
    original = copy.deepcopy(config)
    manager = FFNActorManager(config)
    seen = []

    async def initialize(cfg):
        seen.append(cfg)
        # Collective setup must enter all ranks before any awaits completion.
        await asyncio.sleep(0)
        assert len(seen) == 2

    manager.actors = [
        SimpleNamespace(initialize_role=SimpleNamespace(remote=initialize))
        for _ in range(2)
    ]
    topology = ElasticTopology(4, 2)
    afd = topology.afd_config(parse_afd_config(config), port=32100, role="ffn")
    asyncio.run(manager.initialize_roles(topology, afd, 32101))
    assert config == original
    for rank, cfg in enumerate(seen):
        parallel = cfg.parallel_config
        assert parallel.data_parallel_size == 2
        assert parallel.data_parallel_rank == rank
        assert parallel.data_parallel_rank_local == 0
        assert not parallel.enable_elastic_ep
        assert parallel.placement_group is None
        assert parallel.assigned_physical_gpu_ids is None
        assert parallel._data_parallel_master_port_list == [32101]
        assert cfg.additional_config["afd"]["role"] == "ffn"
    seen[0].parallel_config._data_parallel_master_port_list.pop()
    assert seen[1].parallel_config._data_parallel_master_port_list == [32101]
