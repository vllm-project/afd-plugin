from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from vllm.forward_context import get_forward_context as get_current_forward_context

from afd_plugin.async_moe import AsyncMoeStage
from afd_plugin.model_executor.models import (
    ASYNC_MOE_UBATCH_METADATA_KEY,
    AsyncMoeUbatchMetadata,
    get_afd_metadata_from_forward_context,
    get_async_moe_ubatch_metadata_from_forward_context,
)


def test_get_afd_metadata_from_additional_kwargs():
    forward_context = SimpleNamespace(
        additional_kwargs={"afd_metadata": {"stage": 0}},
        afd_metadata={"stage": 1},
    )

    assert get_afd_metadata_from_forward_context(forward_context) == {"stage": 0}


def test_get_afd_metadata_ignores_forward_context_attribute():
    forward_context = SimpleNamespace(
        additional_kwargs={},
        afd_metadata={"stage": 0},
    )

    assert get_afd_metadata_from_forward_context(forward_context) is None


def test_get_async_moe_ubatch_metadata_from_additional_kwargs():
    sidecar = {"ubatch_slices": ["stage0", "stage1"]}
    forward_context = SimpleNamespace(
        additional_kwargs={ASYNC_MOE_UBATCH_METADATA_KEY: sidecar},
    )

    assert (
        get_async_moe_ubatch_metadata_from_forward_context(forward_context) is sidecar
    )


def test_async_moe_execution_plan_rejects_inconsistent_stage_descriptions():
    with pytest.raises(ValueError, match="same non-empty stage count"):
        AsyncMoeUbatchMetadata(
            attn_metadata=[{}],
            stages=[
                AsyncMoeStage(slice(0, 1), slice(0, 2), input_tokens=2),
                AsyncMoeStage(slice(1, 2), slice(2, 4), input_tokens=2),
            ],
            parent_input_tokens=4,
            use_sequence_parallel=False,
        )

    with pytest.raises(ValueError, match="contiguous"):
        AsyncMoeUbatchMetadata(
            attn_metadata=[{}, {}],
            stages=[
                AsyncMoeStage(slice(0, 1), slice(0, 2), input_tokens=2),
                AsyncMoeStage(slice(1, 2), slice(3, 4), input_tokens=1),
            ],
            parent_input_tokens=4,
            use_sequence_parallel=False,
        )

    with pytest.raises(ValueError, match="must fit its physical extent"):
        AsyncMoeUbatchMetadata(
            attn_metadata=[{}, {}],
            stages=[
                AsyncMoeStage(slice(0, 1), slice(0, 2), input_tokens=2),
                AsyncMoeStage(slice(1, 2), slice(2, 5), input_tokens=2),
            ],
            parent_input_tokens=5,
            use_sequence_parallel=False,
        )


def test_deepseek_afd_wrapper_keeps_full_model_compile_enabled():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert "@native.support_torch_compile\nclass AFDDeepseekV2Model" in source
    assert "from __future__ import annotations" not in source
    assert "self.do_not_compile = True" not in source


def test_deepseek_afd_wrapper_treats_index_topk_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'self.is_v32 = hasattr(config, "index_topk")' in source
    assert "self.is_v32 = config.index_topk is not None" not in source
    assert "topk_tokens = config.index_topk" in source


def test_deepseek_afd_wrapper_treats_llama_4_scaling_as_optional():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'getattr(self.config, "llama_4_scaling", None)' in source
    assert "self.config.llama_4_scaling" not in source


def test_deepseek_afd_attention_path_can_compute_gate_before_send():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    forward_with_afd = source.split("    def forward_with_afd(", 1)[1].split(
        "    def forward_with_afd_v2(",
        1,
    )[0]
    forward_with_afd_v2 = source.split("    def forward_with_afd_v2(", 1)[1].split(
        "    def forward_with_afd_v3(",
        1,
    )[0]
    attention_gate_forward = executor_source.split(
        "def run_attention_gate_afd_forward(",
        1,
    )[1].split("def run_async_moe_ubatch_afd_forward(", 1)[0]

    assert 'if self.afd_role == "attention":' in source
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "from afd_plugin.model_executor.models.npu import (" in forward_with_afd_v2
    assert "deepseek_v2_async_cam_forward," in forward_with_afd_v2
    assert "def _forward_attention(" not in source
    assert "return self.forward_with_afd_v3(" in forward_with_afd
    assert "return self.forward_with_afd_v2(" in forward_with_afd
    assert (
        "return deepseek_v2_async_cam_forward.run_attention_gate_afd_forward("
        in forward_with_afd_v2
    )
    assert "layer.compute_attn_output(" not in forward_with_afd
    assert "layer.compute_attn_output(" in attention_gate_forward
    assert "pending_ffn_recv" in attention_gate_forward
    assert "topk_weights" in attention_gate_forward
    assert "topk_ids" in attention_gate_forward


def test_deepseek_afd_attention_gate_can_force_balanced_topk_ids():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    module_imports = source.split("logger = init_logger(__name__)", 1)[0]
    compute_attn_output = source.split("    def compute_attn_output(", 1)[1].split(
        "    def compute_ffn_output(",
        1,
    )[0]

    assert "compute_attention_gate_topk(" in compute_attn_output
    assert "afd_plugin.model_executor.models.npu" not in module_imports
    assert "from afd_plugin.model_executor.models.npu import (" in compute_attn_output
    assert "deepseek_v2_attention_gate," in compute_attn_output
    assert "force_balanced_topk_ids_enabled" in gate_source
    assert "def _force_balanced_topk_ids(" in gate_source
    assert "topk_ids.copy_(balanced_topk_ids)" in gate_source
    assert "topk_weights, topk_ids = afd_connector.select_experts(" in (gate_source)
    assert "if force_balanced_topk_ids_enabled():" in gate_source
    assert (
        gate_source.index(
            "topk_weights, topk_ids = afd_connector.select_experts(",
        )
        < gate_source.index("if force_balanced_topk_ids_enabled():")
        < gate_source.index("topk_weights = topk_weights.to(torch.float32)")
    )


def test_deepseek_afd_gate_on_attention_keeps_dense_layers_local():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    executor_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_async_cam_forward.py",
    ).read_text()

    assert "self.is_moe_layer = _is_moe_layer(config, layer_idx)" in source
    assert "self.compute_gate_on_attention and not self.is_moe_layer" in source
    assert "if not layer.is_moe_layer:" in executor_source
    assert "self.is_dense_mlp_weight(name)" in source


def test_deepseek_compute_gate_on_attention_is_npu_only():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()

    assert 'native.current_platform.device_type != "npu"' in source
    assert "DeepSeekV2 compute_gate_on_attention is supported only on NPU" in source
    assert "# NPU-only: non-NPU platforms are rejected before this branch." in source
    assert (
        "# NPU-only: Attention-side gate/topk is implemented in the NPU helper."
        in source
    )
    assert (
        "# NPU-only: gated MoE FFN compute consumes Attention-side topk payloads."
        in source
    )


def test_async_moe_dense_only_range_stays_on_full_batch_path(monkeypatch):
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward

    calls = []

    class DenseLayer:
        is_moe_layer = False

        def __init__(self, layer_idx):
            self.layer_idx = layer_idx

        def __call__(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            calls.append(
                (
                    self.layer_idx,
                    positions,
                    hidden_states,
                    llama_4_scaling,
                ),
            )
            return (
                f"{hidden_states}:dense{self.layer_idx}",
                f"residual:{self.layer_idx}",
            )

    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_forward_context",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "build_async_moe_stage_inputs",
        lambda *_args, **_kwargs: pytest.fail(
            "a dense-only layer range must not build stage inputs",
        ),
    )
    model = SimpleNamespace(
        start_layer=0,
        end_layer=2,
        layers=[DenseLayer(0), DenseLayer(1)],
    )
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 4), input_tokens=4),
            AsyncMoeStage(slice(1, 2), slice(4, 8), input_tokens=4),
        ],
        parent_input_tokens=8,
        use_sequence_parallel=False,
    )

    output, residual = deepseek_v2_async_cam_forward.run_async_moe_ubatch_afd_forward(
        model=model,
        hidden_states="full-hidden",
        residual=None,
        positions="full-positions",
        afd_metadata=SimpleNamespace(connector=object()),
        async_moe_ubatch_metadata=metadata,
        llama_4_scaling="full-scaling",
    )

    assert calls == [
        (0, "full-positions", "full-hidden", "full-scaling"),
        (1, "full-positions", "full-hidden:dense0", "full-scaling"),
    ]
    assert output == "full-hidden:dense0:dense1"
    assert residual == "residual:1"


def test_async_moe_single_layer_pipeline_preserves_stage_order(monkeypatch):
    from afd_plugin.connectors import AFDForwardContextMetadata
    from afd_plugin.model_executor.models.npu import deepseek_v2_async_cam_forward

    events = []
    forward_context = SimpleNamespace(
        attn_metadata={"layer": "full"},
        additional_kwargs={},
        ubatch_idx=0,
        num_ubatches=1,
        num_tokens=4,
        pad_size=0,
    )

    class FakeTensor:
        shape = (2, 8)

    class Connector:
        def send_attn_output(self, hidden_states, context, **_kwargs):
            events.append(("send", context.metadata.stage_idx, hidden_states))

        def recv_ffn_output(self, ref_tensor, ubatch_idx):
            events.append(("recv", ubatch_idx, ref_tensor))
            return ref_tensor

    class MoeLayer:
        is_moe_layer = True
        layer_idx = 0

        def compute_attn_output(
            self,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            stage_context = get_current_forward_context()
            events.append(
                (
                    "compute",
                    stage_context.ubatch_idx,
                    hidden_states,
                    positions,
                    llama_4_scaling,
                ),
            )
            return hidden_states, residual, FakeTensor(), FakeTensor(), None

    connector = Connector()
    parent_metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0],
        requests_start_loc=[0],
        stage_idx=0,
        connector=connector,
        tokens_lens=[4],
        num_stages=1,
        tokens_unpadded_lens=[4],
    )
    execution_plan = AsyncMoeUbatchMetadata(
        attn_metadata=[{"layer": "stage-0"}, {"layer": "stage-1"}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 2), input_tokens=2),
            AsyncMoeStage(slice(1, 2), slice(2, 4), input_tokens=2),
        ],
        parent_input_tokens=4,
        use_sequence_parallel=False,
    )
    stage_hidden_states = [FakeTensor(), FakeTensor()]
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "build_async_moe_stage_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            hidden_states=stage_hidden_states,
            residuals=[None, None],
            positions=["positions-0", "positions-1"],
            llama_4_scaling=["scaling-0", "scaling-1"],
        ),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "restore_async_moe_stage_outputs",
        lambda outputs, _metadata: tuple(outputs),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "prepare_cam_dispatch_payload",
        lambda hidden_states, topk_weights, topk_ids, router_logits, **_kwargs: (
            SimpleNamespace(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
                layout=object(),
            )
        ),
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "restore_cam_dispatch_output",
        lambda output, _layout: output,
    )

    output, residual = deepseek_v2_async_cam_forward.run_async_moe_ubatch_afd_forward(
        model=SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[MoeLayer()],
        ),
        hidden_states=FakeTensor(),
        residual=None,
        positions="full-positions",
        afd_metadata=parent_metadata,
        async_moe_ubatch_metadata=execution_plan,
        llama_4_scaling="full-scaling",
    )

    assert [(event[0], event[1]) for event in events] == [
        ("compute", 0),
        ("send", 0),
        ("compute", 1),
        ("recv", 0),
        ("send", 1),
        ("recv", 1),
    ]
    assert output == tuple(stage_hidden_states)
    assert residual is None
    assert forward_context.attn_metadata == {"layer": "full"}
    assert forward_context.num_tokens == 4


def test_plain_tp_attention_gate_dispatches_each_token_once(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.connectors import AFDForwardContextMetadata
    from afd_plugin.model_executor.models.npu import (
        async_moe_sp,
        deepseek_v2_async_cam_forward,
    )

    tp_rank = 1
    tp_tokens = 3
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    padded_ffn_output = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100
    sent_payloads = []

    monkeypatch.setattr(
        async_moe_sp,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=tp_rank),
    )
    monkeypatch.setattr(
        async_moe_sp,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: padded_ffn_output,
    )
    monkeypatch.setattr(
        deepseek_v2_async_cam_forward,
        "get_forward_context",
        lambda: SimpleNamespace(
            ubatch_idx=0,
            flash_comm_v1_enabled=False,
        ),
    )

    class Connector:
        def send_attn_output(self, hidden_states, context, **kwargs):
            sent_payloads.append((hidden_states, context, kwargs))

        def recv_ffn_output(self, ref_tensor, ubatch_idx):
            assert ref_tensor.shape[0] == tp_tokens
            assert ubatch_idx == 0
            return padded_ffn_output[tp_tokens:]

    class MoeLayer:
        is_moe_layer = True
        layer_idx = 0

        @staticmethod
        def compute_attn_output(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        ):
            return (
                hidden_states,
                residual,
                torch.ones(5, 2),
                torch.zeros(5, 2, dtype=torch.int32),
                torch.ones(5, 4),
            )

    connector = Connector()
    afd_metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0],
        requests_start_loc=[0],
        stage_idx=0,
        connector=connector,
        tokens_lens=[5],
        num_stages=1,
        tokens_unpadded_lens=[5],
    )
    output, residual = deepseek_v2_async_cam_forward.run_attention_gate_afd_forward(
        model=SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[MoeLayer()],
        ),
        hidden_states=hidden_states,
        residual=None,
        positions=torch.arange(5),
        afd_metadata=afd_metadata,
    )

    assert residual is None
    assert torch.equal(output, padded_ffn_output[:5])
    assert len(sent_payloads) == 1
    dispatched_hidden, context, dispatched_kwargs = sent_payloads[0]
    assert dispatched_hidden.shape[0] == tp_tokens
    assert context.metadata.total_tokens == tp_tokens
    assert dispatched_kwargs["topk_weights"].shape[0] == tp_tokens
    assert dispatched_kwargs["topk_ids"].shape[0] == tp_tokens
    assert dispatched_kwargs["router_logits"].shape[0] == tp_tokens
    assert torch.equal(dispatched_hidden[-1], torch.zeros(2))


def test_async_moe_sp_layout_transposes_full_shards_into_stage_shards(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_moe_sp

    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 10), input_tokens=10),
            AsyncMoeStage(slice(0, 1), slice(10, 15), input_tokens=6),
        ],
        parent_input_tokens=16,
        use_sequence_parallel=True,
    )
    global_hidden = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    global_residual = global_hidden + 100
    positions = torch.arange(16)
    scaling = torch.ones(2, 16)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_moe_sp, "get_tp_group", lambda: tp_group)

    for tp_rank, expected_positions in (
        (0, [[0, 1, 2, 3, 4], [10, 11, 12]]),
        (1, [[5, 6, 7, 8, 9], [13, 14, 0]]),
    ):
        tp_group.rank_in_group = tp_rank
        local_slice = slice(tp_rank * 8, (tp_rank + 1) * 8)

        def all_gather(tensor, token_dim):
            assert token_dim == 0
            assert tensor.shape[1] == 4
            return torch.cat((global_hidden, global_residual), dim=-1)

        monkeypatch.setattr(
            async_moe_sp,
            "tensor_model_parallel_all_gather",
            all_gather,
        )
        stage_inputs = async_moe_sp.build_async_moe_stage_inputs(
            global_hidden[local_slice],
            global_residual[local_slice],
            positions,
            scaling,
            metadata,
        )

        assert [stage.tolist() for stage in stage_inputs.positions] == (
            expected_positions
        )
        assert [int(stage.shape[0]) for stage in stage_inputs.hidden_states] == [
            5,
            3,
        ]
        assert [tuple(stage.shape) for stage in stage_inputs.llama_4_scaling] == [
            (2, 5),
            (2, 3),
        ]
        monkeypatch.setattr(
            async_moe_sp,
            "tensor_model_parallel_all_gather",
            lambda tensor, token_dim: (
                global_hidden[:10]
                if int(tensor.shape[token_dim]) == 5
                else torch.cat(
                    (
                        global_hidden[10:15],
                        global_hidden.new_zeros((1, 2)),
                    ),
                    dim=0,
                )
            ),
        )
        restored = async_moe_sp.restore_async_moe_stage_outputs(
            stage_inputs.hidden_states,
            metadata,
        )
        expected_restored = global_hidden.clone()
        expected_restored[15].zero_()
        assert torch.equal(restored, expected_restored[local_slice])


def test_async_moe_sp_layout_rejects_replicated_hidden_states(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_moe_sp

    monkeypatch.setattr(
        async_moe_sp,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 8), input_tokens=8),
            AsyncMoeStage(slice(0, 1), slice(8, 16), input_tokens=8),
        ],
        parent_input_tokens=16,
        use_sequence_parallel=True,
    )

    with pytest.raises(ValueError, match="TP-local"):
        async_moe_sp.build_async_moe_stage_inputs(
            torch.zeros(16, 2),
            None,
            torch.arange(16),
            None,
            metadata,
        )


def test_async_moe_replicated_layout_removes_and_restores_parent_padding():
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_moe_sp

    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 3), input_tokens=3),
            AsyncMoeStage(slice(0, 1), slice(3, 5), input_tokens=2),
        ],
        parent_input_tokens=8,
        use_sequence_parallel=False,
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    positions = torch.arange(8)

    stage_inputs = async_moe_sp.build_async_moe_stage_inputs(
        hidden_states,
        None,
        positions,
        None,
        metadata,
    )

    assert [stage[:, 0].tolist() for stage in stage_inputs.hidden_states] == [
        [0.0, 2.0, 4.0],
        [6.0, 8.0],
    ]
    assert [stage.tolist() for stage in stage_inputs.positions] == [
        [0, 1, 2],
        [3, 4],
    ]
    restored = async_moe_sp.restore_async_moe_stage_outputs(
        stage_inputs.hidden_states,
        metadata,
    )
    assert torch.equal(restored[:5], hidden_states[:5])
    assert torch.count_nonzero(restored[5:]) == 0


def test_plain_tp_cam_boundary_shards_and_restores_replicated_tokens(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_moe_sp

    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(async_moe_sp, "get_tp_group", lambda: tp_group)
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    topk_ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    router_logits = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    padded_output = torch.arange(12, dtype=torch.float32).reshape(6, 2) + 100

    monkeypatch.setattr(
        async_moe_sp,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: padded_output,
    )

    expected_hidden_rows = (
        hidden_states[:3],
        torch.cat((hidden_states[3:], hidden_states.new_zeros((1, 2)))),
    )
    for tp_rank in range(2):
        tp_group.rank_in_group = tp_rank
        payload = async_moe_sp.prepare_cam_dispatch_payload(
            hidden_states,
            topk_weights,
            topk_ids,
            router_logits,
            use_sequence_parallel=False,
        )

        assert torch.equal(payload.hidden_states, expected_hidden_rows[tp_rank])
        assert payload.hidden_states.shape[0] == 3
        assert payload.topk_weights.shape[0] == 3
        assert payload.topk_ids.shape[0] == 3
        assert payload.router_logits is not None
        assert payload.router_logits.shape[0] == 3
        assert payload.layout.parent_tokens == 5
        assert payload.layout.padded_tokens == 6
        assert payload.layout.requires_tp_all_gather is True

        local_output = padded_output[tp_rank * 3 : (tp_rank + 1) * 3]
        restored = async_moe_sp.restore_cam_dispatch_output(
            local_output,
            payload.layout,
        )
        assert torch.equal(restored, padded_output[:5])


def test_flashcomm_cam_boundary_keeps_existing_local_shard(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_moe_sp

    monkeypatch.setattr(
        async_moe_sp,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    topk_weights = torch.ones(3, 2)
    topk_ids = torch.zeros(3, 2, dtype=torch.int32)

    payload = async_moe_sp.prepare_cam_dispatch_payload(
        hidden_states,
        topk_weights,
        topk_ids,
        None,
        use_sequence_parallel=True,
    )

    assert payload.hidden_states is hidden_states
    assert payload.topk_weights is topk_weights
    assert payload.topk_ids is topk_ids
    assert payload.layout.requires_tp_all_gather is False
    assert (
        async_moe_sp.restore_cam_dispatch_output(
            hidden_states,
            payload.layout,
        )
        is hidden_states
    )


def test_async_moe_sp_layout_prefers_multi_axis_position_token_dim(monkeypatch):
    torch = pytest.importorskip("torch")

    from afd_plugin.model_executor.models.npu import async_moe_sp

    tp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    monkeypatch.setattr(async_moe_sp, "get_tp_group", lambda: tp_group)
    global_hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    monkeypatch.setattr(
        async_moe_sp,
        "tensor_model_parallel_all_gather",
        lambda tensor, token_dim: global_hidden,
    )
    metadata = AsyncMoeUbatchMetadata(
        attn_metadata=[{}, {}],
        stages=[
            AsyncMoeStage(slice(0, 1), slice(0, 2), input_tokens=2),
            AsyncMoeStage(slice(1, 2), slice(2, 4), input_tokens=2),
        ],
        parent_input_tokens=4,
        use_sequence_parallel=True,
    )
    positions = torch.arange(16).reshape(4, 4)
    scaling = positions.to(torch.float32).reshape(4, 4, 1, 1)

    stage_inputs = async_moe_sp.build_async_moe_stage_inputs(
        global_hidden[2:],
        None,
        positions,
        scaling,
        metadata,
    )

    assert [tuple(stage.shape) for stage in stage_inputs.positions] == [
        (4, 1),
        (4, 1),
    ]
    assert [stage[:, 0].tolist() for stage in stage_inputs.positions] == [
        positions[:, 1].tolist(),
        positions[:, 3].tolist(),
    ]
    assert [tuple(stage.shape) for stage in stage_inputs.llama_4_scaling] == [
        (4, 1, 1, 1),
        (4, 1, 1, 1),
    ]


def test_deepseek_afd_ffn_path_reuses_ascend_moe_mlp_after_attention_gate():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "compute_attention_gate_moe_ffn(" in compute_ffn_output
    assert "from afd_plugin.model_executor.models.npu import (" in compute_ffn_output
    assert "deepseek_v2_attention_gate," in compute_ffn_output
    assert "AFDF2ATransferPayload(" in compute_moe
    assert "MoEMlpComputeInput(" in compute_moe
    assert "unified_apply_mlp(" in compute_moe
    assert "quant_type == QuantType.W8A8" in compute_moe
    assert "w13_weight_scale_fp32" in compute_moe
    assert "w13_weight_scale_fp32_list" in compute_moe
    assert "w2_weight_scale_list" in compute_moe
    assert "MoEQuantParams(quant_type=quant_type)" in compute_moe
    assert "_gmmswigluquant_fusion_enabled()" in compute_moe
    assert "fusion=use_gmmswigluquant_fusion" in compute_moe
    assert "_compute_w8a8_shared_experts_from_int8(" in compute_moe
    assert "shared_input.dtype == torch.int8" in compute_moe
    assert "fusion=False" not in compute_moe


def test_deepseek_afd_ffn_compute_omits_stub_io_diagnostics():
    source = Path("afd_plugin/model_executor/models/deepseek_v2.py").read_text()
    gate_source = Path(
        "afd_plugin/model_executor/models/npu/deepseek_v2_attention_gate.py",
    ).read_text()
    compute_ffn_output = source.split(
        "    def compute_ffn_output(",
        1,
    )[1].split("\n\n@native.support_torch_compile", 1)[0]
    compute_moe = gate_source.split(
        "def compute_attention_gate_moe_ffn(",
        1,
    )[1].split("\ndef _dequantize_int8_activation(", 1)[0]

    assert "camp2p_stub_io_enabled()" not in source
    assert "_log_ffn_compute_step(" not in compute_ffn_output
    assert '"dense_mlp_begin"' not in compute_ffn_output
    assert '"dense_scaling_begin"' not in compute_ffn_output
    assert "_log_ffn_compute_step(" not in compute_moe
    assert '"routed_scaling_begin"' not in compute_moe
    assert '"shared_scaling_begin"' not in compute_moe
