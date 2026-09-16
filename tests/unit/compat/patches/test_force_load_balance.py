from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _QuantType:
    W8A8 = 1


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    class AscendW8A8DynamicFusedMoEMethod:
        quant_type = _QuantType.W8A8

    config = SimpleNamespace(
        additional_config={},
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(enable_eplb=False),
        model_config=SimpleNamespace(dtype=torch.float32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    ascend_config = SimpleNamespace(
        eplb_config=SimpleNamespace(dynamic_eplb=False), enable_force_eplb=False
    )
    context = SimpleNamespace(
        in_profile_run=False,
        moe_comm_method=SimpleNamespace(
            fused_experts=lambda fused_experts_input, quant_method: fused_experts_input
        ),
    )
    modules = {
        "vllm.config": {
            "VllmConfig": SimpleNamespace,
            "get_current_vllm_config": lambda: config,
        },
        "vllm.logger": {
            "logger": SimpleNamespace(
                warning_once=lambda *args: None,
                info=lambda *args: None,
                warning=lambda *args: None,
            ),
        },
        "vllm_ascend.ascend_config": {"get_ascend_config": lambda: ascend_config},
        "vllm_ascend.ascend_forward_context": {"_EXTRA_CTX": context},
        "vllm_ascend.distributed.parallel_state": {
            "get_mc2_group": lambda: SimpleNamespace(),
        },
        "vllm_ascend.ops.fused_moe.dataclass.fused_experts": {
            "build_fused_experts_input": lambda **kwargs: kwargs,
        },
        "vllm_ascend.ops.fused_moe.routed_experts": {
            "AscendRoutedExperts": SimpleNamespace,
        },
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic": {
            "AscendW8A8DynamicFusedMoEMethod": AscendW8A8DynamicFusedMoEMethod,
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def force_lb_mod(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _install_fake_modules(monkeypatch)
    module_name = "afd_plugin.compat.patches.npu.force_load_balance"
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod


def _new_layer() -> SimpleNamespace:
    return SimpleNamespace(
        mix_placement=False,
        n_shared_experts=0,
        ascend_expert_map=None,
        global_redundant_expert_num=0,
        ascend_mc2_mask=None,
        apply_router_weight_on_input=False,
        ascend_pertoken_scale=None,
    )


def _aggregate_target_rank_counts(
    force_lb_mod: types.ModuleType,
    *,
    n_routed_experts: int,
    ep_size: int,
    top_k: int,
    topn_per_rank: int,
    batch_tokens: int,
) -> torch.Tensor:
    local_routed_experts = n_routed_experts // ep_size
    expert_ids: list[torch.Tensor] = []
    for ep_rank in range(ep_size):
        config = force_lb_mod.ForceLoadBalanceConfig(
            n_routed_experts=n_routed_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            top_k=top_k,
            topn_per_rank=topn_per_rank,
        )
        expert_ids.append(
            force_lb_mod._build_topk_buffer(
                config,
                max_tokens=batch_tokens,
                device=torch.device("cpu"),
            ).flatten()
        )

    target_ranks = torch.cat(expert_ids) // local_routed_experts
    return torch.bincount(target_ranks.to(torch.int64), minlength=ep_size)


def test_force_load_balance_buffer_topn_per_rank(force_lb_mod: types.ModuleType):
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=8,
        ep_size=4,
        ep_rank=0,
        top_k=2,
        topn_per_rank=1,
    )

    force_lb_mod._init_force_lb_buffer(
        method,
        config,
        max_tokens=4,
        device=torch.device("cpu"),
    )

    expected = torch.tensor([[0, 2], [4, 6], [0, 2], [4, 6]], dtype=torch.int32)
    assert torch.equal(method.force_lb_fake_topk_buffer, expected)


def test_force_load_balance_buffer_uses_max_num_batched_tokens(
    force_lb_mod: types.ModuleType,
):
    max_tokens = force_lb_mod._get_force_lb_max_tokens(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=6))
    )
    assert max_tokens == 6

    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=4,
        ep_size=2,
        ep_rank=0,
        top_k=2,
        topn_per_rank=0,
    )

    force_lb_mod._init_force_lb_buffer(
        method,
        config,
        max_tokens=max_tokens,
        device=torch.device("cpu"),
    )

    assert method.force_lb_fake_topk_buffer.shape == (6, 2)


def test_force_load_balance_max_tokens_falls_back_when_not_int(
    force_lb_mod: types.ModuleType,
):
    max_tokens = force_lb_mod._get_force_lb_max_tokens(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=None))
    )
    assert max_tokens == 128


def test_force_load_balance_buffer_ids_within_routed_experts(
    force_lb_mod: types.ModuleType,
):
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=4,
        ep_size=2,
        ep_rank=0,
        top_k=2,
        topn_per_rank=2,
    )

    force_lb_mod._init_force_lb_buffer(
        method,
        config,
        max_tokens=2,
        device=torch.device("cpu"),
    )

    assert int(method.force_lb_fake_topk_buffer.max()) < config.n_routed_experts


def test_force_load_balance_full_expert_cycle_is_deterministic(
    force_lb_mod: types.ModuleType,
):
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=8,
        ep_size=4,
        ep_rank=0,
        top_k=2,
        topn_per_rank=0,
    )

    first = force_lb_mod._build_expert_cycle(config, torch.device("cpu"))
    second = force_lb_mod._build_expert_cycle(config, torch.device("cpu"))

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(8))


def test_force_load_balance_full_expert_cycle_generates_on_cpu(
    force_lb_mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
):
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=8,
        ep_size=4,
        ep_rank=0,
        top_k=2,
        topn_per_rank=0,
    )
    randperm_devices: list[torch.device | str | None] = []
    original_randperm = torch.randperm

    def recording_randperm(*args, **kwargs):
        randperm_devices.append(kwargs.get("device"))
        return original_randperm(*args, **kwargs)

    monkeypatch.setattr(torch, "randperm", recording_randperm)

    force_lb_mod._build_expert_cycle(config, torch.device("cpu"))

    assert randperm_devices == [torch.device("cpu")]


@pytest.mark.parametrize(
    ("ep_size", "batch_tokens"),
    [
        pytest.param(64, 16, id="ep64-bs16"),
        pytest.param(16, 42, id="ep16-bs42"),
        pytest.param(16, 56, id="ep16-bs56"),
    ],
)
def test_force_load_balance_aggregates_evenly_for_published_batches(
    force_lb_mod: types.ModuleType,
    ep_size: int,
    batch_tokens: int,
):
    top_k = 8
    counts = _aggregate_target_rank_counts(
        force_lb_mod,
        n_routed_experts=256,
        ep_size=ep_size,
        top_k=top_k,
        topn_per_rank=4,
        batch_tokens=batch_tokens,
    )

    assert torch.equal(
        counts,
        torch.full((ep_size,), batch_tokens * top_k, dtype=torch.int64),
    )


def test_force_load_balance_all_experts_aggregates_partial_cycle_evenly(
    force_lb_mod: types.ModuleType,
):
    ep_size = 4
    top_k = 2
    batch_tokens = 1
    counts = _aggregate_target_rank_counts(
        force_lb_mod,
        n_routed_experts=8,
        ep_size=ep_size,
        top_k=top_k,
        topn_per_rank=0,
        batch_tokens=batch_tokens,
    )

    assert torch.equal(
        counts,
        torch.full((ep_size,), batch_tokens * top_k, dtype=torch.int64),
    )


def test_force_load_balance_buffer_grows_for_large_batch(
    force_lb_mod: types.ModuleType,
):
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    config = force_lb_mod.ForceLoadBalanceConfig(
        n_routed_experts=4,
        ep_size=2,
        ep_rank=0,
        top_k=2,
        topn_per_rank=2,
    )

    force_lb_mod._init_force_lb_buffer(
        method,
        config,
        max_tokens=2,
        device=torch.device("cpu"),
    )
    topk_ids = force_lb_mod._get_force_lb_topk_ids(
        method,
        config,
        batch_tokens=5,
        device=torch.device("cpu"),
    )

    assert topk_ids.shape == (5, 2)
    assert method.force_lb_fake_topk_buffer.shape[0] >= 5


def test_w8a8_apply_lazily_builds_and_swaps_topk_ids(
    force_lb_mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
):
    vllm_config = force_lb_mod.get_current_vllm_config()
    vllm_config.additional_config = {
        "enable_force_load_balance": True,
        "force_load_balance_topn_per_rank": 1,
    }
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    assert method.enable_force_load_balance
    assert method.force_load_balance_topn_per_rank == 1
    assert method.force_lb_fake_topk_buffer is None
    monkeypatch.setattr(
        force_lb_mod,
        "get_current_vllm_config",
        lambda: (_ for _ in ()).throw(AssertionError("outside config context")),
    )

    layer = _new_layer()
    layer.mix_placement = False
    layer.moe_config = SimpleNamespace(
        ep_size=4,
        ep_rank=0,
        num_logical_experts=8,
    )
    layer.w13_weight = torch.empty(0)
    layer.w13_weight_scale_fp32 = torch.empty(0)
    layer.w2_weight = torch.empty(0)
    layer.w2_weight_scale = torch.empty(0)
    layer.swiglu_limit = None

    router_logits = torch.zeros((4, 8), dtype=torch.float32)
    out = method.apply(
        layer=layer,
        x=torch.empty((4, 1)),
        topk_weights=torch.ones((4, 2)),
        topk_ids=router_logits[:, :2].to(torch.int64),
        shared_experts=None,
        shared_experts_input=None,
    )

    expected = torch.tensor([[0, 2], [4, 6], [0, 2], [4, 6]])
    assert torch.equal(out["topk_ids"], expected)
    assert method.force_lb_fake_topk_buffer.shape == (8, 2)
    for field_name in ("n_routed_experts", "ep_size", "ep_rank", "top_k"):
        assert not hasattr(layer, field_name)


def test_w8a8_apply_passthrough_when_plugin_disabled(
    force_lb_mod: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
):
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    assert not method.enable_force_load_balance
    monkeypatch.setattr(
        force_lb_mod,
        "get_current_vllm_config",
        lambda: (_ for _ in ()).throw(AssertionError("outside config context")),
    )

    layer = _new_layer()
    layer.mix_placement = False
    layer.moe_config = SimpleNamespace(
        ep_size=1,
        ep_rank=0,
        num_logical_experts=2,
    )
    layer.w13_weight = torch.empty(0)
    layer.w13_weight_scale_fp32 = torch.empty(0)
    layer.w2_weight = torch.empty(0)
    layer.w2_weight_scale = torch.empty(0)
    layer.swiglu_limit = None

    router_logits = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    out = method.apply(
        layer=layer,
        x=torch.empty((4, 1)),
        topk_weights=torch.ones((4, 2)),
        topk_ids=router_logits.to(torch.int64),
        shared_experts=None,
        shared_experts_input=None,
    )

    assert torch.equal(out["topk_ids"], router_logits.to(torch.int64))
    assert method.force_lb_fake_topk_buffer is None


@pytest.mark.parametrize("native_policy", ["profile", "force_eplb"])
def test_native_routing_policy_takes_precedence(force_lb_mod, native_policy):
    force_lb_mod.get_current_vllm_config().additional_config = {
        "enable_force_load_balance": True,
    }
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    force_lb_mod._EXTRA_CTX.in_profile_run = native_policy == "profile"
    force_lb_mod.get_ascend_config().enable_force_eplb = native_policy == "force_eplb"
    layer = _new_layer()
    ids = torch.tensor([[3, 1]])
    weights = torch.tensor([[0.4, 0.6]])
    payload = method.apply(layer, torch.empty((1, 4)), weights, ids, None, None)
    assert payload["topk_ids"] is ids
    assert payload["topk_weights"] is weights
    assert method.force_lb_fake_topk_buffer is None


def test_mixed_shared_ids_and_weights_survive_routed_override(force_lb_mod):
    force_lb_mod.get_current_vllm_config().additional_config = {
        "enable_force_load_balance": True,
        "force_load_balance_topn_per_rank": 1,
    }
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    layer = _new_layer()
    layer.mix_placement = True
    layer.n_shared_experts = 1
    layer.moe_config = SimpleNamespace(num_logical_experts=8, ep_size=4, ep_rank=0)
    ids = torch.tensor([[7, 7, 8], [7, 7, 8]], dtype=torch.int64)
    weights = torch.rand((2, 3))
    payload = method.apply(layer, torch.empty((2, 4)), weights, ids, None, None)
    assert torch.equal(payload["topk_ids"], torch.tensor([[0, 2, 8], [4, 6, 8]]))
    assert payload["topk_weights"] is weights
    assert torch.equal(ids, torch.tensor([[7, 7, 8], [7, 7, 8]]))


@pytest.mark.parametrize(
    "v2,dynamic,enabled,expected",
    [(False, True, False, True), (True, True, False, False), (True, False, True, True)],
)
def test_target_expert_weight_list_policy(force_lb_mod, v2, dynamic, enabled, expected):
    config = force_lb_mod.get_current_vllm_config()
    config.use_v2_model_runner = v2
    config.parallel_config.enable_eplb = enabled
    force_lb_mod.get_ascend_config().eplb_config.dynamic_eplb = dynamic
    method = force_lb_mod.AscendW8A8DynamicFusedMoEMethod()
    assert method.use_expert_weight_list is expected
    assert method.dynamic_eplb is (dynamic and not v2)
