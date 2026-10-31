from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _QuantType:
    W8A8 = 1


def _fake_upstream_apply(
    self,
    layer,
    x,
    router_logits,
    top_k,
    renormalize,
    **kwargs,
):
    del self
    _, topk_ids = globals()["select_experts"](
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=kwargs.get("use_grouped_topk", False),
        renormalize=renormalize,
        topk_group=kwargs.get("topk_group"),
        num_expert_group=kwargs.get("num_expert_group"),
        custom_routing_function=kwargs.get("custom_routing_function"),
        scoring_func=kwargs.get("scoring_func", "softmax"),
        routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
        e_score_correction_bias=kwargs.get("e_score_correction_bias"),
        mix_placement=getattr(layer, "mix_placement", False),
        num_logical_experts=router_logits.shape[1],
        num_shared_experts=getattr(layer, "n_shared_experts", 0) or 0,
        num_experts=kwargs.get("num_experts", -1),
        tid2eid=kwargs.get("tid2eid"),
    )
    return topk_ids


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    def build_fused_experts_input(*args: object, **kwargs: object) -> torch.Tensor:
        """Fake builder: returns the possibly swapped topk_ids."""

        del args
        return kwargs["topk_ids"]

    class AscendW8A8DynamicFusedMoEMethod:
        quant_type = _QuantType.W8A8

        def __init__(self):
            self.upstream_initialized = True
            self.multistream_overlap_gate = False

    vllm = types.ModuleType("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.CompilationMode = SimpleNamespace(VLLM_COMPILE="vllm_compile")
    vllm_config.VllmConfig = object
    current_vllm_config = SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(mode="none"),
        model_config=SimpleNamespace(enforce_eager=True, dtype=torch.float32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )
    vllm_config.get_current_vllm_config = lambda: current_vllm_config
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger.logger = SimpleNamespace(warning_once=lambda *args, **kwargs: None)

    root = types.ModuleType("vllm_ascend")
    ascend_config_mod = types.ModuleType("vllm_ascend.ascend_config")
    ascend_config_mod.get_ascend_config = lambda: SimpleNamespace(
        enable_fused_mc2=0,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    ascend_forward_context_mod = types.ModuleType("vllm_ascend.ascend_forward_context")
    ascend_forward_context_mod.MoECommType = SimpleNamespace(FUSED_MC2="fused_mc2")
    ascend_forward_context_mod._MEGA_MOE_SUPPORTED = False
    ascend_forward_context_mod._EXTRA_CTX = SimpleNamespace(
        moe_comm_method=SimpleNamespace(
            fused_experts=lambda fused_experts_input: fused_experts_input
        ),
        moe_comm_type=None,
    )
    distributed = types.ModuleType("vllm_ascend.distributed")
    parallel_state_mod = types.ModuleType("vllm_ascend.distributed.parallel_state")
    parallel_state_mod.get_mc2_group = lambda: SimpleNamespace()
    ops = types.ModuleType("vllm_ascend.ops")
    fused_moe_pkg = types.ModuleType("vllm_ascend.ops.fused_moe")
    experts_selector_mod = types.ModuleType(
        "vllm_ascend.ops.fused_moe.experts_selector"
    )

    def select_experts(
        hidden_states,
        router_logits,
        top_k,
        use_grouped_topk,
        renormalize,
        topk_group,
        num_expert_group,
        custom_routing_function,
        scoring_func,
        routed_scaling_factor,
        e_score_correction_bias,
        mix_placement,
        num_logical_experts,
        num_shared_experts,
        num_experts,
        tid2eid,
    ):
        del (
            hidden_states,
            use_grouped_topk,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            mix_placement,
            num_logical_experts,
            num_shared_experts,
            num_experts,
            tid2eid,
        )
        topk_ids = router_logits[:, :top_k].to(torch.int64)
        return (
            torch.ones_like(topk_ids, dtype=torch.float32),
            topk_ids,
        )

    experts_selector_mod.select_experts = select_experts
    experts_selector_mod.zero_experts_compute = None
    upstream_apply = types.FunctionType(
        _fake_upstream_apply.__code__,
        {"select_experts": select_experts},
        name="apply",
    )
    AscendW8A8DynamicFusedMoEMethod.apply = upstream_apply
    fused_moe_mod = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    fused_moe_mod.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        info_once=lambda *args, **kwargs: None,
    )
    moe_runtime_args_mod = types.ModuleType(
        "vllm_ascend.ops.fused_moe.moe_runtime_args"
    )
    moe_runtime_args_mod.build_fused_experts_input = build_fused_experts_input

    quant = types.ModuleType("vllm_ascend.quantization")
    methods = types.ModuleType("vllm_ascend.quantization.methods")
    methods_base_mod = types.ModuleType("vllm_ascend.quantization.methods.base")

    def get_moe_num_logical_experts(
        layer,
        num_experts,
        global_redundant_expert_num=0,
        num_shared_experts=0,
    ):
        num_logical_experts = getattr(layer.moe_config, "num_logical_experts", None)
        if num_logical_experts is not None:
            return int(num_logical_experts)
        return int(num_experts - global_redundant_expert_num - num_shared_experts)

    methods_base_mod.get_moe_num_logical_experts = get_moe_num_logical_experts
    w8a8_mod = types.ModuleType("vllm_ascend.quantization.methods.w8a8_dynamic")
    w8a8_mod.AscendW8A8DynamicFusedMoEMethod = AscendW8A8DynamicFusedMoEMethod

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)
    monkeypatch.setitem(sys.modules, "vllm_ascend", root)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", ascend_config_mod)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        ascend_forward_context_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed",
        distributed,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.parallel_state",
        parallel_state_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fused_moe_pkg)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.experts_selector",
        experts_selector_mod,
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.fused_moe.fused_moe", fused_moe_mod
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_runtime_args",
        moe_runtime_args_mod,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quant)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.methods", methods)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.methods.base",
        methods_base_mod,
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.methods.w8a8_dynamic", w8a8_mod
    )
    return fused_moe_mod


@pytest.fixture
def force_lb_mod(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _install_fake_modules(monkeypatch)
    module_name = "afd_plugin.compat.patches.npu.force_load_balance"
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    return mod


def _new_layer() -> SimpleNamespace:
    return SimpleNamespace()


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
    assert method.upstream_initialized
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
        router_logits=router_logits,
        top_k=2,
        renormalize=True,
        num_experts=8,
    )

    expected = torch.tensor([[0, 2], [4, 6], [0, 2], [4, 6]])
    assert torch.equal(out, expected)
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
        router_logits=router_logits,
        top_k=2,
        renormalize=True,
        num_experts=2,
    )

    assert torch.equal(out, router_logits.to(torch.int64))
    assert method.force_lb_fake_topk_buffer is None
