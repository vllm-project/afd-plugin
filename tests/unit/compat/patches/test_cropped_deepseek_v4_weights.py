# SPDX-License-Identifier: Apache-2.0
"""Exercise the real checkpoint filter without importing the Ascend runtime."""

import builtins
import importlib
import inspect
import sys
from collections.abc import Iterable
from types import ModuleType, SimpleNamespace

import pytest
import torch

from afd_plugin.compat.patches.npu import cropped_deepseek_v4_weights as patch


@pytest.fixture
def native_model(monkeypatch):
    class NativeModel:
        def __init__(self, depth):
            self.config = SimpleNamespace(num_hidden_layers=depth)
            self.received = []
            self.result = {"native-result"}
            self.error = None

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            self.received = list(weights)
            if self.error is not None:
                raise self.error
            return self.result

    module = ModuleType("vllm_ascend.models.deepseek_v4")
    module.AscendDeepseekV4ForCausalLM = NativeModel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return NativeModel


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize("depth", [1, 3])
def test_filter_keeps_configured_layers_and_all_non_decoder_weights(
    native_model, prefix, depth
):
    model = native_model(depth)
    original_signature = inspect.signature(native_model.load_weights)
    weights = [
        (f"{prefix}layers.{index}.ffn.experts.0.w1.weight", torch.tensor(index))
        for index in range(4)
    ]
    weights += [
        (name, torch.tensor(9))
        for name in (
            "embed.weight",
            "model.norm.weight",
            "head.weight",
            "mtp.0.layers.9.attn.weight",
            "layers.not_an_index.weight",
        )
    ]
    assert patch.apply_afd_cropped_deepseek_v4_weights_patch()
    result = model.load_weights(iter(weights))

    assert result is model.result
    assert model.received == weights[:depth] + weights[4:]
    assert inspect.signature(native_model.load_weights) == original_signature


def test_full_checkpoint_and_empty_iterator_preserve_native_behavior(native_model):
    model = native_model(2)
    weights = [
        ("layers.0.weight", torch.tensor(0)),
        ("layers.1.weight", torch.tensor(1)),
    ]
    assert patch.apply_afd_cropped_deepseek_v4_weights_patch()
    assert model.load_weights(iter(weights)) is model.result
    assert model.received == weights
    assert model.load_weights(iter(())) is model.result
    assert model.received == []


def test_iterator_is_consumed_once_and_native_errors_propagate(native_model):
    class OneShot:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1
            yield "layers.0.weight", torch.tensor(0)
            yield "layers.1.weight", torch.tensor(1)

    model = native_model(1)
    weights = OneShot()
    assert patch.apply_afd_cropped_deepseek_v4_weights_patch()
    model.load_weights(weights)
    assert weights.iterations == 1
    assert len(model.received) == 1
    model.error = ValueError("native failure")
    with pytest.raises(ValueError, match="native failure") as error:
        model.load_weights(iter(()))
    assert error.value is model.error


def test_installation_survives_module_reload_without_wrapping_again(native_model):
    original = native_model.load_weights
    assert patch.apply_afd_cropped_deepseek_v4_weights_patch()
    installed = native_model.load_weights
    assert installed is not original
    assert patch.apply_afd_cropped_deepseek_v4_weights_patch()
    assert native_model.load_weights is installed
    importlib.reload(patch)
    assert patch.apply_afd_cropped_deepseek_v4_weights_patch()
    assert native_model.load_weights is installed
    assert native_model.__dict__[patch._PATCH_ATTR] is original


@pytest.mark.parametrize("signature", ["extra", "default", "keyword_only"])
def test_incompatible_upstream_signature_is_left_unchanged(
    monkeypatch, native_model, signature
):
    def extra(self, weights, new_argument):
        return set()

    def default(self, weights=()):
        return set()

    def keyword_only(self, *, weights):
        return set()

    upstream = {"extra": extra, "default": default, "keyword_only": keyword_only}[
        signature
    ]
    monkeypatch.setattr(native_model, "load_weights", upstream)
    assert not patch.apply_afd_cropped_deepseek_v4_weights_patch()
    assert native_model.load_weights is upstream
    assert patch._PATCH_ATTR not in native_model.__dict__


def test_import_does_not_load_ascend_or_install_patch(monkeypatch, native_model):
    original = native_model.load_weights
    original_import = builtins.__import__

    def import_without_ascend(name, *args, **kwargs):
        if name.startswith("vllm_ascend"):
            raise AssertionError("Ascend imported before worker resolution")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ascend)
    importlib.reload(patch)
    assert native_model.load_weights is original
