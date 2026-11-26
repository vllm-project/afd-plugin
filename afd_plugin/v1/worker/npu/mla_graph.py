# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bind Ascend graph tasks to the live metadata of their capture microbatch."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any

from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm_ascend.compilation.updatable_graph import _ACTIVE_GRAPH

if TYPE_CHECKING:
    from vllm_ascend.compilation.updatable_graph import (
        ParamProvider,
        Params,
        UpdatableGraph,
    )
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


@dataclass(frozen=True, slots=True)
class _UBatchParamProvider:
    ubatch_id: int
    native_provider: ParamProvider

    def resolve(self, metadata: Sequence[dict[str, Any] | None]) -> Params:
        return self.native_provider.resolve(metadata[self.ubatch_id])


@contextmanager
def _bind_graph_method(
    graph_or_manager: UpdatableGraph | ModelAclGraphManager,
    method: Callable[..., Any],
) -> Iterator[None]:
    name = method.__name__
    bindings = vars(graph_or_manager)
    has_override = name in bindings
    previous = bindings.get(name)
    bindings[name] = MethodType(method, graph_or_manager)
    try:
        yield
    finally:
        if has_override:
            bindings[name] = previous
        else:
            del bindings[name]


@contextmanager
def use_ubatch_capture_graph(graph: UpdatableGraph) -> Iterator[None]:
    """Expose the parent capture graph inside one existing microbatch thread."""
    token = _ACTIVE_GRAPH.set(graph)
    try:
        yield
    finally:
        _ACTIVE_GRAPH.reset(token)


@contextmanager
def use_ubatch_graph_registration(graph: UpdatableGraph) -> Iterator[None]:
    """Register distinct native providers for equal-shape microbatches."""
    native_register_task = graph.register_task

    # Upstream: Ascend 7dcdb690, compilation/updatable_graph.py.
    # Equal-sized stages can produce equal providers for the same layer. Tag
    # the provider before native registration groups handles by provider.
    # Signature matches UpdatableGraph.register_task. Delegate native capture
    # event/handle/resource ownership. Remove when providers carry stage IDs.
    def register_task(
        self,
        operation: Callable[..., Any],
        kwargs: dict[str, Any],
        provider: ParamProvider,
    ) -> None:
        # ### PATCH START: key graph tasks by their capture stage
        native_register_task(
            operation, kwargs, _UBatchParamProvider(dbo_current_ubatch_id(), provider)
        )
        # ### PATCH END: key graph tasks by their capture stage

    with _bind_graph_method(graph, register_task):
        yield


@contextmanager
def use_ubatch_replay_metadata(
    manager: ModelAclGraphManager, stages: Sequence[dict[str, Any] | None]
) -> Iterator[None]:
    """Supply all live stage metadata at the native task-resolution entry."""
    native_replay = manager._updatable_graph_replay

    # Upstream: Ascend 7dcdb690, worker/v2/aclgraph_utils.py.
    # The public manager replay handles graph selection and cleanup. Change
    # only the metadata passed to the native updatable-graph branch, preserving
    # its resolve/wait/replay/update sequence. Signature matches upstream.
    # Remove when the manager can resolve separate microbatch metadata itself.
    def _updatable_graph_replay(self, desc, attn_metadata):
        # ### PATCH START: resolve tasks with the current microbatch metadata
        return native_replay(desc, stages)
        # ### PATCH END: resolve tasks with the current microbatch metadata

    with _bind_graph_method(manager, _updatable_graph_replay):
        yield
