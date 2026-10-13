# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep DSV4 RoPE inputs local to each ubatch and graph capture.

Pinned vllm-ascend 11ee456 uses one decode RoPE runtime buffer per config and
group. Building stage 1 overwrites stage 0 unless the runner snapshots stage
0 first. Graph replay must then refresh those capture-owned tensor addresses.
"""

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy

DSARopePair = tuple[RopeDataProxy, RopeDataProxy]


def isolate_dsa_rope_metadata(
    metadata: AscendDSAMetadata,
    stage_rope: DSARopePair | None,
) -> DSARopePair | None:
    """Snapshot a stage once; share its owned RoPE across attention groups."""
    if metadata.decode is None or not isinstance(metadata.cos, RopeDataProxy):
        return None
    if stage_rope is None:
        data = {
            config: {
                group: (cos.clone(), sin.clone())
                for group, (cos, sin) in groups.items()
            }
            for config, groups in metadata.cos._data.items()
        }
        stage_rope = (
            RopeDataProxy(data, is_cos=True),
            RopeDataProxy(data, is_cos=False),
        )
    metadata.cos, metadata.sin = stage_rope
    metadata.decode.cos = stage_rope[0][: metadata.num_decode_tokens]
    metadata.decode.sin = stage_rope[1][: metadata.num_decode_tokens]
    return stage_rope


def refresh_dsa_rope(
    captured: RopeDataProxy,
    live: RopeDataProxy,
    updated_tensors: set[int],
) -> None:
    """Copy live cos/sin into capture storage once per shared tensor address."""
    assert captured._data.keys() == live._data.keys()
    for config, groups in captured._data.items():
        assert groups.keys() == live._data[config].keys()
        for group, targets in groups.items():
            sources = live._data[config][group]
            for target, source in zip(targets, sources, strict=True):
                assert target.shape == source.shape
                assert target.dtype == source.dtype
                address = target.data_ptr()
                if address in updated_tensors:
                    continue
                updated_tensors.add(address)
                if address != source.data_ptr():
                    target.copy_(source)
