# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Layer-bank ordering and the A/F startup contract, without device execution."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.connectors.npu.async_mega.mega_startup import (  # noqa: E402
    agree_layer_selection,
)
from afd_plugin.connectors.npu.async_mega.mega_weights import (  # noqa: E402
    pack_layer_weights,
)


@pytest.fixture
def startup():
    def run(ffn_layers, error=None):
        store = torch.distributed.HashStore()
        layers = [[]] + ffn_layers
        with ThreadPoolExecutor(max_workers=len(layers)) as pool:
            futures = [
                pool.submit(
                    agree_layer_selection,
                    store,
                    1,
                    len(ffn_layers),
                    rank,
                    ids,
                    timeout=timedelta(seconds=5),
                    preparation_error=error if rank == len(ffn_layers) else None,
                )
                for rank, ids in enumerate(layers)
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=5))
                except (RuntimeError, ValueError) as exc:
                    results.append(exc)
        return results

    return run


def test_bank_order_matches_sparse_model_layer_mapping(startup):
    weights = {
        7: tuple([object()] for _ in range(4)),
        2: tuple([object()] for _ in range(4)),
    }
    bank = pack_layer_weights(weights)
    attention, ffn = startup([[7, 2]])
    assert attention == ffn == {2: 0, 7: 1}
    for layer, slot in ffn.items():
        for part in range(4):
            assert bank[part][slot] is weights[layer][part][0]


def test_disagreement_between_ffn_ranks_is_rejected(startup):
    outcomes = startup([[2, 7], [2, 8]])
    assert all(
        isinstance(exc, ValueError) and "must agree" in str(exc) for exc in outcomes
    )


@pytest.mark.parametrize("ids", [[], [-1], [2, 2]])
def test_invalid_layer_ids_rejected_before_dispatch(startup, ids):
    outcomes = startup([ids])
    assert all(
        isinstance(exc, ValueError) and "unique nonnegative" in str(exc)
        for exc in outcomes
    )


def test_weight_conversion_failure_reaches_attention_and_all_ffn_peers(startup):
    outcomes = startup([[2, 7], []], ValueError("invalid FP4 storage"))
    assert len(outcomes) == 3
    assert all(isinstance(exc, RuntimeError) for exc in outcomes)
    assert {str(exc) for exc in outcomes} == {
        "Mega weight preparation failed: rank 2: ValueError: invalid FP4 storage"
    }


def test_bank_rejects_per_expert_or_missing_projection_lists():
    with pytest.raises(ValueError, match="stacked expert tensors"):
        pack_layer_weights({0: ([object(), object()], [], [], [])})


def test_known_preparation_error_survives_peer_disconnect_during_ack():
    from unittest.mock import Mock

    store = Mock(spec=torch.distributed.Store)
    store.get.side_effect = [
        b'{"layers": [], "error": ""}',
        b'{"layers": [], "error": "ValueError: invalid FP4 storage"}',
    ]
    store.wait.side_effect = [None, OSError("Store peer exited")]
    with pytest.raises(RuntimeError, match="rank 1: ValueError: invalid FP4 storage"):
        agree_layer_selection(store, 1, 1, 0, [], timeout=timedelta(seconds=5))
