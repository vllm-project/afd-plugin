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
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa


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
) -> None:
    """Detach mutable backend workspaces from one metadata object.

    vLLM-Ascend metadata builders may return views into process-global RoPE
    runtime buffers. That is safe when one metadata object is live, but eager
    DBO and Async CAM build multiple metadata objects before executing all of
    them. A later build must not change an earlier metadata object.

    Only mutable runtime storage is materialized here.  Immutable RoPE tables,
    KV-cache tensors, masks, block tables, and other read-only inputs continue
    to be shared with the upstream runtime.
    """

    if isinstance(metadata, AscendSFAMetadata):
        # SFA always asks get_cos_and_sin_mla() for its reusable runtime
        # buffer, for both prefill and decode.
        metadata.cos = metadata.cos.clone()
        metadata.sin = metadata.sin.clone()
        return

    if isinstance(metadata, AscendMLAMetadata):
        # RoPE MLA prefill currently returns fresh indexed tensors, while
        # no-RoPE MLA prefill and every decode use reusable identity/runtime
        # buffers.  Materialize both paths so ownership does not depend on
        # that backend implementation detail.
        if metadata.prefill is not None:
            metadata.prefill.cos = metadata.prefill.cos.clone()
            metadata.prefill.sin = metadata.prefill.sin.clone()
        if metadata.decode is not None:
            metadata.decode.cos = metadata.decode.cos.clone()
            metadata.decode.sin = metadata.decode.sin.clone()
        return

    if isinstance(metadata, AscendDSAMetadata):
        # Target Ascend stores RoPE proxies on request metadata. Rebuild the
        # exact physical token span before another async stage reuses the
        # process-wide runtime RoPE buffers.
        assert metadata.req_metadata is not None
        metadata.req_metadata.cos, metadata.req_metadata.sin = get_cos_and_sin_dsa(
            input_positions[:num_input_tokens].long(),
            use_cache=False,
        )


def materialize_deepseek_attention_metadata_by_layer(
    metadata_by_layer: Mapping[str, AttentionMetadata],
    input_positions: torch.Tensor,
    num_input_tokens: int,
) -> None:
    """Materialize each shared attention-group metadata object exactly once."""

    materialized_ids: set[int] = set()
    for metadata in metadata_by_layer.values():
        metadata_id = id(metadata)
        if metadata_id in materialized_ids:
            continue
        materialize_deepseek_attention_metadata(
            metadata, input_positions, num_input_tokens
        )
        materialized_ids.add(metadata_id)


__all__ = [
    "isolate_deepseek_attention_builder_inputs",
    "materialize_deepseek_attention_metadata",
    "materialize_deepseek_attention_metadata_by_layer",
]
