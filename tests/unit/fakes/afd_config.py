# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from afd_plugin.config import AFDConfig


def camp2p_config(*, role: str):
    return AFDConfig(
        connector="CAMP2pAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )
