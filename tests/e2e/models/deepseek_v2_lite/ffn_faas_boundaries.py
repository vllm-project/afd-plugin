# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Test-only first-prefill MoE observations, excluded from normal acceptance.

Hooks deliberately synchronize small tensors to CPU. They are diagnostic
instrumentation, never a performance path or a service management feature.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.ffn_faas.protocol import Mode
from tests.npu.ffn_faas_native import Work, combine


class FFNBoundaryWorker:
    model_runner: NPUModelRunner

    def arm_ffn_boundaries(
        self,
        directory: str,
        layer_index: int = 1,
        target_input_ids: list[int] | None = None,
    ) -> None:
        self._boundary_directory = Path(directory)
        self._boundary_tensors: dict[str, torch.Tensor] = {}
        self._boundary_handles = []
        self._boundary_active = target_input_ids is None
        self._boundary_expert_chunks: list[torch.Tensor] = []
        model = self.model_runner.model
        layer = model.model.layers[layer_index]
        gate = layer.mlp.gate
        shared = layer.mlp.shared_experts
        is_faas = type(layer.mlp).__name__ == "GateOnlyRemoteMoE"
        self._boundary_metadata = {
            "rank": get_tensor_model_parallel_rank(),
            "layer": layer_index,
            "is_faas": is_faas,
            "mlp_type": type(layer.mlp).__name__,
            "routed_scaling_factor": model.config.routed_scaling_factor,
            "norm_topk_prob": model.config.norm_topk_prob,
            "topk_method": model.config.topk_method,
            "gate_weight_dtype": str(gate.weight.dtype),
            "shared_gate_up_shape": list(shared.gate_up_proj.weight.shape),
            "target_input_ids": target_input_ids,
        }
        parameters = dict(gate.named_parameters(recurse=False))
        self._boundary_metadata["gate_weight_fp32_dtype"] = (
            str(parameters["weight_fp32"].dtype)
            if "weight_fp32" in parameters
            else "absent"
        )

        def capture(name, tensor):
            if (
                self._boundary_active
                and name not in self._boundary_tensors
                and isinstance(tensor, torch.Tensor)
            ):
                self._boundary_tensors[name] = tensor.detach().cpu().clone()

        def inputs(module, args, kwargs):
            del module
            tokens = args[0] if args else kwargs["input_ids"]
            if target_input_ids is not None:
                self._boundary_active = False
                if "input_ids" in self._boundary_tensors:
                    return
                actual_ids = tokens.detach().cpu().tolist()
                count = len(target_input_ids)
                for start in range(len(actual_ids) - count + 1):
                    if actual_ids[start : start + count] == target_input_ids:
                        self._boundary_active = True
                        self._boundary_metadata["target_row_start"] = start
                        break
            capture("input_ids", tokens)
            if len(args) > 1:
                capture("positions", args[1])
            else:
                capture("positions", kwargs["positions"])

        self._boundary_handles.append(
            model.register_forward_pre_hook(inputs, with_kwargs=True)
        )
        self._boundary_handles.append(
            model.model.norm.register_forward_hook(
                lambda module, args, output: capture("final_norm", output[0])
            )
        )
        self._boundary_handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                lambda module, args, output: capture("moe_input", output[0])
            )
        )
        self._boundary_handles.append(
            shared.register_forward_hook(
                lambda module, args, output: capture("shared_raw", output)
            )
        )
        self._boundary_handles.append(
            shared.gate_up_proj.register_forward_hook(
                lambda module, args, output: capture("shared_gate_up", output[0])
            )
        )
        self._boundary_handles.append(
            shared.act_fn.register_forward_hook(
                lambda module, args, output: capture("shared_activation", output)
            )
        )
        self._boundary_handles.append(
            shared.down_proj.register_forward_hook(
                lambda module, args, output: capture("shared_down", output[0])
            )
        )
        next_layer = model.model.layers[layer_index + 1]
        self._boundary_handles.append(
            next_layer.input_layernorm.register_forward_pre_hook(
                lambda module, args: capture("moe_after_add", args[0])
            )
        )
        if is_faas:
            router = layer.mlp.gate_router
            self._boundary_handles.append(
                gate.register_forward_hook(
                    lambda module, args, output: capture("gate_logits", output[0])
                )
            )
            self._boundary_handles.append(
                gate.register_forward_pre_hook(
                    lambda module, args: capture("gate_input", args[0])
                )
            )
            connector = self.model_runner.connector
            session = connector._session
            original_combine = session._combine

            def observe_combine(this, sequence, layer, x, ids, probabilities):
                if self._boundary_active and layer == layer_index:
                    experts = combine(
                        this.executor,
                        Work(
                            sequence,
                            layer,
                            x,
                            ids,
                            probabilities,
                            x,
                            ids,
                            probabilities,
                        ),
                        Mode.PEEK_RESULTS,
                    )
                    self._boundary_expert_chunks.append(experts)
                return original_combine(sequence, layer, x, ids, probabilities)

            session._combine = MethodType(observe_combine, session)
            original_send, original_recv = (
                connector.send_attn_output,
                connector.recv_ffn_output,
            )

            def send(this, hidden, context, **kwargs):
                if context.metadata.layer_idx == layer_index:
                    capture("dispatch_input", hidden)
                    capture("dispatch_ids", kwargs["topk_ids"])
                    capture("dispatch_probabilities", kwargs["topk_weights"])
                return original_send(hidden, context, **kwargs)

            def recv(this, ref_tensor, ubatch_idx=0, **kwargs):
                output = original_recv(ref_tensor, ubatch_idx, **kwargs)
                if kwargs["context"].metadata.layer_idx == layer_index:
                    capture("routed_local", output)
                    if self._boundary_active:
                        capture(
                            "routed_experts_local",
                            torch.cat(self._boundary_expert_chunks),
                        )
                return output

            connector.send_attn_output = MethodType(send, connector)
            connector.recv_ffn_output = MethodType(recv, connector)
        else:
            runner = layer.mlp.experts
            router = runner.router
            original_gate = runner._compute_router_logits
            original_scale = runner._maybe_apply_routed_scale_to_output

            def compute_gate(this, hidden, logits):
                capture("gate_input", hidden)
                capture("router_compute_argument", logits)
                result = original_gate(hidden, logits)
                capture("gate_logits", result)
                return result

            def scale(this, shared_output, routed_output):
                shared_result, routed_result = original_scale(
                    shared_output, routed_output
                )
                capture("shared_before_add", shared_result)
                capture("routed_before_add", routed_result)
                return shared_result, routed_result

            runner._compute_router_logits = MethodType(compute_gate, runner)
            runner._maybe_apply_routed_scale_to_output = MethodType(scale, runner)
            routed = runner.routed_experts
            original_routing = routed._select_experts

            def native_routing(
                this,
                hidden_states,
                router_logits,
                enable_force_load_balance,
                input_ids=None,
            ):
                weights, ids = original_routing(
                    hidden_states,
                    router_logits,
                    enable_force_load_balance,
                    input_ids,
                )
                capture("routing_input", hidden_states)
                capture("routing_logits", router_logits)
                capture("router_probabilities", weights)
                capture("router_ids", ids)
                return weights, ids

            routed._select_experts = MethodType(native_routing, routed)
            method = routed.quant_method
            original_gmm1, original_act, original_gmm2 = (
                method.apply_gmm1,
                method.apply_act_quant,
                method.apply_gmm2,
            )

            def gmm1(this, mlp_input):
                output = original_gmm1(mlp_input)
                capture("expert_input", mlp_input.hidden_states)
                capture("expert_group_list", mlp_input.group_list)
                capture("expert_map", routed.expert_map)
                capture("expert_topk_scales", mlp_input.topk_scales)
                capture("expert_gate_up", output)
                if self._boundary_active:
                    self._boundary_metadata["group_list_type"] = (
                        mlp_input.group_list_type
                    )
                    self._boundary_metadata["need_trans"] = mlp_input.need_trans
                return output

            def activation(this, mlp_input, hidden):
                capture("expert_activation", hidden)
                result = original_act(mlp_input, hidden)
                capture("expert_activation_scaled", result[0])
                return result

            def gmm2(this, mlp_input, hidden, scale):
                output = original_gmm2(mlp_input, hidden, scale)
                capture("expert_down", output)
                return output

            method.apply_gmm1 = MethodType(gmm1, method)
            method.apply_act_quant = MethodType(activation, method)
            method.apply_gmm2 = MethodType(gmm2, method)
        self._boundary_metadata["router_type"] = type(router).__name__
        original_select = router.select_experts

        def select(this, *args, **kwargs):
            weights, ids = original_select(*args, **kwargs)
            capture("router_probabilities", weights)
            capture("router_ids", ids)
            return weights, ids

        router.select_experts = MethodType(select, router)

    def flush_ffn_boundaries(self) -> dict[str, str | int]:
        for handle in self._boundary_handles:
            handle.remove()
        self._boundary_handles = []
        if "input_ids" not in self._boundary_tensors:
            raise RuntimeError("target prompt prefill was not observed")
        rank = get_tensor_model_parallel_rank()
        self._boundary_directory.mkdir(parents=True, exist_ok=True)
        path = self._boundary_directory / f"rank-{rank}.pt"
        torch.save(
            {"metadata": self._boundary_metadata, "tensors": self._boundary_tensors},
            path,
        )
        return {"path": str(path), "tensor_count": len(self._boundary_tensors)}
