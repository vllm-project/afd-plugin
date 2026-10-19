# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek attention metadata ownership for staged NPU execution."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from vllm.v1.attention.backend import AttentionMetadata, AttentionMetadataBuilder
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy, get_cos_and_sin_dsa


def isolate_deepseek_attention_builder_inputs(
    builder: AttentionMetadataBuilder,
    metadata: AscendCommonAttentionMetadata,
) -> None:
    """Detach common metadata that a DeepSeek builder mutates in place.

    Async CAM builds the full batch before both execution stages.  Stage
    builders therefore must not write through views into the full-batch
    common metadata.  The copies below are deliberately backend-specific;
    the remaining common metadata fields are read-only builder inputs and
    retain their zero-copy views.
    """

    if isinstance(builder, AscendSFAMetadataBuilder):
        # SFA C8 reshape writes all three arrays through
        # store_kv_block_metadata().
        if metadata.group_len is not None:
            metadata.group_len = metadata.group_len.clone()
        if metadata.group_key_idx is not None:
            metadata.group_key_idx = metadata.group_key_idx.clone()
        if metadata.group_key_cache_idx is not None:
            metadata.group_key_cache_idx = metadata.group_key_cache_idx.clone()
    elif isinstance(builder, AscendDSAMetadataBuilder):
        # DSA clears padded request rows while building prefill/decode
        # metadata.  Keep those writes local to this stage and cache group.
        metadata.block_table_tensor = metadata.block_table_tensor.clone()


def materialize_deepseek_attention_metadata(
    metadata: AttentionMetadata,
    input_positions: torch.Tensor,
    num_input_tokens: int,
    *,
    rope_buffers: dict[str, torch.Tensor] | None = None,
    rope_capacity: int = 0,
) -> None:
    """Detach mutable backend workspaces from one metadata object.

    vLLM-Ascend metadata builders may return views into process-global RoPE
    runtime buffers. That is safe when one metadata object is live, but eager
    DBO and Async CAM build multiple metadata objects before executing all of
    them. A later build must not change an earlier metadata object.

    FULL graphs pass stage-owned persistent buffers: their captured pointers
    must observe later positions as well as isolate the two prepared stages.
    Eager callers retain fresh copies for independently live metadata objects.
    Only mutable runtime storage is materialized here.  Immutable RoPE tables,
    KV-cache tensors, masks, block tables, and other read-only inputs continue
    to be shared with the upstream runtime.
    """

    if isinstance(metadata, AscendSFAMetadata):
        # SFA always asks get_cos_and_sin_mla() for its reusable runtime
        # buffer, for both prefill and decode.
        metadata.cos = _own_rope(metadata.cos, "cos", rope_buffers, rope_capacity)
        metadata.sin = _own_rope(metadata.sin, "sin", rope_buffers, rope_capacity)
        return

    if isinstance(metadata, AscendMLAMetadata):
        # RoPE MLA prefill currently returns fresh indexed tensors, while
        # no-RoPE MLA prefill and every decode use reusable identity/runtime
        # buffers.  Materialize both paths so ownership does not depend on
        # that backend implementation detail.
        if metadata.prefill is not None:
            metadata.prefill.cos = _own_rope(
                metadata.prefill.cos, "prefill.cos", rope_buffers, rope_capacity
            )
            metadata.prefill.sin = _own_rope(
                metadata.prefill.sin, "prefill.sin", rope_buffers, rope_capacity
            )
        if metadata.decode is not None:
            metadata.decode.cos = _own_rope(
                metadata.decode.cos, "decode.cos", rope_buffers, rope_capacity
            )
            metadata.decode.sin = _own_rope(
                metadata.decode.sin, "decode.sin", rope_buffers, rope_capacity
            )
        return

    if isinstance(metadata, AscendDSAMetadata):
        # Target Ascend stores RoPE proxies on request metadata. Rebuild the
        # exact physical token span before another async stage reuses the
        # process-wide runtime RoPE buffers.
        assert metadata.req_metadata is not None
        cos, sin = get_cos_and_sin_dsa(
            input_positions[:num_input_tokens].long(),
            use_cache=False,
        )
        if rope_buffers is None:
            metadata.req_metadata.cos, metadata.req_metadata.sin = cos, sin
        else:
            # Pinned vllm-ascend ops/rope_dsv4.py at commit
            # 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6 shares _data between
            # the cos/sin proxies. Own each Tensor pair, preserving that API.
            owned_data: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
            for config_key, groups in cos._data.items():
                owned_data[config_key] = {}
                for group, (cos_value, sin_value) in groups.items():
                    field = repr((config_key, group))
                    owned_data[config_key][group] = (
                        _own_rope(
                            cos_value,
                            f"request.cos.{field}",
                            rope_buffers,
                            rope_capacity,
                        ),
                        _own_rope(
                            sin_value,
                            f"request.sin.{field}",
                            rope_buffers,
                            rope_capacity,
                        ),
                    )
            metadata.req_metadata.cos = RopeDataProxy(owned_data)
            metadata.req_metadata.sin = RopeDataProxy(owned_data, is_cos=False)


def _own_rope(
    value: torch.Tensor,
    field: str,
    buffers: dict[str, torch.Tensor] | None,
    capacity: int,
) -> torch.Tensor:
    if buffers is None:
        return value.clone()
    if field not in buffers:
        buffers[field] = torch.empty(
            (capacity, *value.shape[1:]), dtype=value.dtype, device=value.device
        )
    return buffers[field][: value.shape[0]].copy_(value)


def materialize_deepseek_attention_metadata_by_layer(
    metadata_by_layer: Mapping[str, AttentionMetadata],
    input_positions: torch.Tensor,
    num_input_tokens: int,
    *,
    rope_buffers: dict[str, dict[str, torch.Tensor]] | None = None,
    rope_capacity: int = 0,
) -> None:
    """Materialize each shared attention-group metadata object exactly once."""

    materialized_ids: set[int] = set()
    for layer, metadata in metadata_by_layer.items():
        metadata_id = id(metadata)
        if metadata_id in materialized_ids:
            continue
        materialize_deepseek_attention_metadata(
            metadata,
            input_positions,
            num_input_tokens,
            rope_buffers=rope_buffers.setdefault(layer, {})
            if rope_buffers is not None
            else None,
            rope_capacity=rope_capacity,
        )
        materialized_ids.add(metadata_id)


__all__ = [
    "isolate_deepseek_attention_builder_inputs",
    "materialize_deepseek_attention_metadata",
    "materialize_deepseek_attention_metadata_by_layer",
]
