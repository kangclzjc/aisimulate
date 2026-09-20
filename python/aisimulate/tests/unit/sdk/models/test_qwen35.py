# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 hybrid GDN + full-attention LM modeling contracts."""

import copy
import json

import pytest

from aisimulate.sdk import common, models
from aisimulate.sdk import config as sdk_config
from aisimulate.sdk.operations import CustomAllReduce, OverlapOp
from aisimulate_core.sdk import models as core_models

pytestmark = pytest.mark.unit


def _model_config(
    tp_size=2,
    *,
    moe_tp_size=None,
    moe_ep_size=1,
    attention_dp_size=1,
    moe_backend=None,
    enable_encoder_dp=True,
):
    return sdk_config.ModelConfig(
        tp_size=tp_size,
        pp_size=1,
        moe_tp_size=tp_size if moe_tp_size is None else moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=attention_dp_size,
        moe_backend=moe_backend,
        enable_encoder_dp=enable_encoder_dp,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    )


def _flatten_ops(phase_ops):
    for op in phase_ops:
        if isinstance(op, OverlapOp):
            yield from op._group_a
            yield from op._group_b
        else:
            yield op


@pytest.mark.parametrize("is_context", [True, False])
def test_sglang_attention_dp_prices_folded_tp_reduction_numerically(is_context):
    """Qwen omits its attention AR, so dispatch must retain BOTH collectives."""
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.operations import NCCL
    from aisimulate_core.sdk.perf_database import get_database_view

    model = models.get_model(
        "Qwen/Qwen3.5-397B-A17B",
        _model_config(tp_size=4, attention_dp_size=2, moe_tp_size=1, moe_ep_size=8),
        "sglang",
    )
    db = get_database_view("gb300", "sglang", "current", allow_missing_data=True, database_mode="SOL")
    ops = list(_flatten_ops(model.context_ops if is_context else model.generation_ops))
    phase = "context" if is_context else "generation"
    assert not any(op._name in (f"{phase}_gdn_ar", f"{phase}_full_ar") for op in ops)

    def cost(op, tokens):
        return float(_evaluate_single_op(db, op, is_context=is_context, batch_size=1, s=64, prefix=0, x=tokens))

    dispatches = [op for op in ops if op._name.endswith("_moe_pre_dispatch")]
    assert len(dispatches) == 2
    for op in dispatches:
        spec = json.loads(op._spec_json())["MoeDispatch"]
        assert spec["attn_ar_modeled"]
        scale, h = spec["scale_factor"], spec["hidden_size"]
        reduction = NCCL("reference", scale, "reduce_scatter", h, 4, common.CommQuantMode.half)
        gather = NCCL("reference", scale, "all_gather", h, 8, common.CommQuantMode.half)
        assert cost(op, 64) == pytest.approx(cost(reduction, 64) + cost(gather, 128), rel=1e-12)
        assert cost(op, 64) > cost(gather, 128)


@pytest.mark.parametrize("is_context", [True, False])
def test_trtllm_qwen_dispatch_retains_legacy_allreduce_cost(is_context):
    """The unqualified TRT-LLM path keeps its documented pre-existing behavior."""
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.perf_database import get_database_view

    model = models.get_model("Qwen/Qwen3.5-397B-A17B", _model_config(tp_size=8), "trtllm")
    db = get_database_view("gb200", "trtllm", "current", allow_missing_data=True, database_mode="SOL")

    def cost(op):
        return float(_evaluate_single_op(db, op, is_context=is_context, batch_size=1, s=64, prefix=0, x=64))

    ops = list(_flatten_ops(model.context_ops if is_context else model.generation_ops))
    dispatches = [op for op in ops if op._name.endswith("_moe_pre_dispatch")]
    assert len(dispatches) == 2
    for op in dispatches:
        spec = json.loads(op._spec_json())["MoeDispatch"]
        assert spec["flavor"] == "TrtllmAlltoall" and spec["attn_ar_modeled"]
        reference = CustomAllReduce("reference", spec["scale_factor"], spec["hidden_size"], 8)
        assert cost(op) == pytest.approx(cost(reference), rel=1e-12)
        assert cost(op) > 0


@pytest.mark.parametrize(
    ("model_name", "expected_out_hidden", "is_moe"),
    [
        ("Qwen/Qwen3.5-27B", 5120, False),
        ("Qwen/Qwen3.5-35B-A3B", 2048, True),
        ("Qwen/Qwen3.5-122B-A10B", 3072, True),
        ("Qwen/Qwen3.5-397B-A17B", 4096, True),
    ],
)
def test_qwen35_dense_and_moe_variants_preserve_language_and_build_vision_encoder(
    model_name, expected_out_hidden, is_moe
):
    model = models.get_model(model_name, _model_config(tp_size=2), "vllm")

    assert isinstance(model.extra_params, common.Qwen35Config)
    assert (model.extra_params.num_experts > 0) is is_moe
    assert model.encoder_config is model.extra_params.vision_config
    assert model.encoder_config.out_hidden_size == expected_out_hidden
    assert model.encoder_config.projector_dims == ((4608, 4608), (4608, expected_out_hidden))
    assert model.encoder_config.projector_n_instances == 1
    assert model.encoder_config.deepstack_visual_indexes == ()
    assert {"context_gdn_scan", "context_attention"} <= {op._name for op in model.context_ops}
    assert {"generation_gdn_recurrence", "generation_attention"} <= {op._name for op in model.generation_ops}
    assert {
        "encoder_patch_embed_gemm",
        "encoder_position_embed",
        "encoder_attention",
        "encoder_merger_norm",
        "encoder_projector_fc1_gemm",
        "encoder_rope_apply",
    } <= {op._name for op in model.encoder_ops}
    position_embed = next(op for op in model.encoder_ops if op._name == "encoder_position_embed")
    position_spec = json.loads(position_embed._spec_json())["Elementwise"]
    assert position_spec["bytes_per_token"] == 2 * (2 * 1152 + 1152)
    merger_norm = next(op for op in model.encoder_ops if op._name == "encoder_merger_norm")
    merger_spec = json.loads(merger_norm._spec_json())["Elementwise"]
    assert merger_spec["bytes_per_token"] == 2 * (1152 + 1152)


def test_qwen35_encoder_dp_replicates_vit_and_gathers_projected_tokens():
    model = models.get_model(
        "Qwen/Qwen3.5-27B",
        _model_config(tp_size=4, enable_encoder_dp=True),
        "vllm",
    )
    encoder_ops = {op._name: op for op in model.encoder_ops}

    assert encoder_ops["encoder_attention"]._n == 16
    assert encoder_ops["encoder_qkv_gemm"]._n == 3 * 1152
    assert "encoder_dp_all_gather" in encoder_ops


def test_qwen35_legacy_encoder_tp_shards_vit_and_projector():
    model = models.get_model(
        "Qwen/Qwen3.5-27B",
        _model_config(tp_size=4, enable_encoder_dp=False),
        "vllm",
    )
    encoder_ops = {op._name: op for op in model.encoder_ops}

    assert encoder_ops["encoder_attention"]._n == 4
    assert encoder_ops["encoder_qkv_gemm"]._n == 3 * 1152 // 4
    assert encoder_ops["encoder_projector_fc0_gemm"]._n == 4608 // 4
    assert encoder_ops["encoder_projector_fc1_gemm"]._k == 4608 // 4
    assert "encoder_dp_all_gather" not in encoder_ops


def test_bundled_qwen35_122b_is_not_added_to_default_support_matrix_roster():
    assert "Qwen/Qwen3.5-122B-A10B" not in common.DefaultHFModels
    assert "Qwen/Qwen3.5-122B-A10B" not in common.SupportMatrixHFModels


@pytest.mark.parametrize(
    (
        "model_name",
        "expected_k_heads",
        "expected_v_heads",
        "expected_in_proj_n",
        "expected_ba_n",
        "expected_out_proj_k",
    ),
    [
        ("Qwen/Qwen3.5-27B", 4, 12, 4096, 24, 1536),
        ("Qwen/Qwen3.5-35B-A3B", 4, 8, 3072, 16, 1024),
    ],
)
def test_qwen35_tp4_gdn_uses_local_heads_without_resharding_projection_gemms(
    model_name, expected_k_heads, expected_v_heads, expected_in_proj_n, expected_ba_n, expected_out_proj_k
):
    """GDN lookup heads are TP-local; qkvz and ba are separate per-rank GEMMs."""
    model = models.get_model(model_name, _model_config(tp_size=4), "sglang")

    context_ops = {op._name: op for op in model.context_ops}
    generation_ops = {op._name: op for op in model.generation_ops}

    for op_name in ("context_gdn_conv1d", "context_gdn_scan"):
        assert context_ops[op_name]._num_k_heads == expected_k_heads
        assert context_ops[op_name]._num_v_heads == expected_v_heads
    for op_name in ("generation_gdn_conv1d", "generation_gdn_recurrence"):
        assert generation_ops[op_name]._num_k_heads == expected_k_heads
        assert generation_ops[op_name]._num_v_heads == expected_v_heads

    assert context_ops["context_gdn_in_proj_gemm"]._n == expected_in_proj_n
    assert context_ops["context_gdn_in_proj_ba_gemm"]._n == expected_ba_n
    assert context_ops["context_gdn_out_proj_gemm"]._k == expected_out_proj_k
    assert generation_ops["generation_gdn_in_proj_gemm"]._n == expected_in_proj_n
    assert generation_ops["generation_gdn_in_proj_ba_gemm"]._n == expected_ba_n
    assert generation_ops["generation_gdn_out_proj_gemm"]._k == expected_out_proj_k


@pytest.mark.parametrize(
    (
        "tp_size",
        "expected_k_heads",
        "expected_v_heads",
        "expected_in_proj_n",
        "expected_ba_n",
        "expected_out_proj_k",
        "expected_qkv_n",
        "expected_logits_n",
    ),
    [
        (1, 16, 128, 36864, 256, 16384, 34816, 248320),
        (2, 8, 64, 18432, 128, 8192, 17408, 124160),
        (4, 4, 32, 9216, 64, 4096, 8704, 62080),
        (8, 2, 16, 4608, 32, 2048, 4608, 31040),
        (16, 1, 8, 2304, 16, 1024, 2560, 15520),
    ],
)
def test_qwen38max_gdn_ladder_and_gated_qkv_and_lm_head_widths(
    tp_size,
    expected_k_heads,
    expected_v_heads,
    expected_in_proj_n,
    expected_ba_n,
    expected_out_proj_k,
    expected_qkv_n,
    expected_logits_n,
):
    """Qwen3.8-Max (linear_num_key_heads=16, linear_num_value_heads=128,
    num_heads=64, num_kv_heads=4, vocab=248320) across the full GDN TP
    ladder: TP-local GDN kernel heads, qkvz/ba/out_proj GEMM widths, the
    output-gate-doubled full-attention qkv GEMM width, and the lm_head
    (logits) GEMM width -- all derived from the bundled config scalars via
    the op-construction formulas in qwen35.py.
    """
    model = models.get_model("Qwen/Qwen3.8-2.4T-A95B", _model_config(tp_size=tp_size), "sglang")

    context_ops = {op._name: op for op in model.context_ops}
    generation_ops = {op._name: op for op in model.generation_ops}

    for op_name in ("context_gdn_conv1d", "context_gdn_scan"):
        assert context_ops[op_name]._num_k_heads == expected_k_heads
        assert context_ops[op_name]._num_v_heads == expected_v_heads
    for op_name in ("generation_gdn_conv1d", "generation_gdn_recurrence"):
        assert generation_ops[op_name]._num_k_heads == expected_k_heads
        assert generation_ops[op_name]._num_v_heads == expected_v_heads

    assert context_ops["context_gdn_in_proj_gemm"]._n == expected_in_proj_n
    assert context_ops["context_gdn_in_proj_ba_gemm"]._n == expected_ba_n
    assert context_ops["context_gdn_out_proj_gemm"]._k == expected_out_proj_k
    assert generation_ops["generation_gdn_in_proj_gemm"]._n == expected_in_proj_n
    assert generation_ops["generation_gdn_in_proj_ba_gemm"]._n == expected_ba_n
    assert generation_ops["generation_gdn_out_proj_gemm"]._k == expected_out_proj_k

    # attn_output_gate=True doubles the query slice:
    # qkv_out = 2*n_q_per_tp*head_size + n_kv_per_tp*head_size*2.
    assert context_ops["context_qkv_gemm"]._n == expected_qkv_n
    assert generation_ops["generation_qkv_gemm"]._n == expected_qkv_n

    assert context_ops["context_logits_gemm"]._n == expected_logits_n
    assert context_ops["context_logits_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16
    assert generation_ops["generation_logits_gemm"]._n == expected_logits_n
    assert generation_ops["generation_logits_gemm"]._quant_mode == common.GEMMQuantMode.bfloat16


@pytest.mark.parametrize(
    ("model_name", "tp_size"),
    [
        ("Qwen/Qwen3.5-27B", 3),
        # Qwen3.8-Max: 16 GDN K heads is not divisible by 32 (128 V heads would be).
        ("Qwen/Qwen3.8-2.4T-A95B", 32),
    ],
)
def test_qwen35_rejects_tensor_parallel_size_that_cannot_shard_gdn_heads(model_name, tp_size):
    with pytest.raises(ValueError, match="GDN head counts must both be divisible"):
        models.get_model(model_name, _model_config(tp_size=tp_size), "sglang")


def test_qwen35_rejects_megamoe_backend():
    """MegaMoE is a DeepSeek-V4-only sglang module; modeling it here would
    double-count the attention AR through the non-DeepEP dispatch branch."""
    with pytest.raises(ValueError, match="megamoe"):
        models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(tp_size=4, moe_backend="megamoe"), "sglang")


@pytest.mark.parametrize(
    "model_config_kwargs",
    [
        {"tp_size": 8},  # pure TP (moe_tp follows tp)
        {"tp_size": 8, "moe_tp_size": 1, "moe_ep_size": 8},  # attention TP + EP
    ],
)
def test_qwen35_moe_prices_comm_through_dispatch_pair_for_all_topologies(model_config_kwargs):
    """Every topology emits the same dispatch chain (serial in context, routed
    and shared experts overlapped in generation); layout-specific collectives
    are resolved inside MoEDispatch, leaving one attention-side AR per layer
    plus embedding."""
    model = models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(**model_config_kwargs), "vllm")

    context_names = [op._name for op in model.context_ops]
    assert not any(name.endswith("_moe_final_ar") for name in context_names)
    for prefix in ("context_gdn", "context_full"):
        expected_order = [
            f"{prefix}_router_gemm",
            f"{prefix}_moe_pre_dispatch",
            f"{prefix}_moe",
            f"{prefix}_moe_post_dispatch",
            f"{prefix}_shared_expert_gate_gemm",
            f"{prefix}_shared_gate_up_gemm",
            f"{prefix}_shared_act_gate",
            f"{prefix}_shared_down_gemm",
            f"{prefix}_shared_expert_gate_mul",
            f"{prefix}_shared_merge",
        ]
        indices = [context_names.index(name) for name in expected_order]
        assert indices == sorted(indices)

    # Generation runs routed and shared experts on parallel CUDA streams
    # (OverlapOp); the merge and the post collective (all-reduce of the
    # merged sum) are serial after the join.
    generation_names = [op._name for op in model.generation_ops]
    generation_ops = {op._name: op for op in model.generation_ops}
    for prefix in ("generation_gdn", "generation_full"):
        overlap = generation_ops[f"{prefix}_moe_overlap"]
        assert [op._name for op in overlap._group_a] == [
            f"{prefix}_router_gemm",
            f"{prefix}_moe_pre_dispatch",
            f"{prefix}_moe",
        ]
        assert [op._name for op in overlap._group_b] == [
            f"{prefix}_shared_expert_gate_gemm",
            f"{prefix}_shared_gate_up_gemm",
            f"{prefix}_shared_act_gate",
            f"{prefix}_shared_down_gemm",
            f"{prefix}_shared_expert_gate_mul",
        ]
        assert generation_names.index(f"{prefix}_shared_merge") < generation_names.index(f"{prefix}_moe_post_dispatch")

    # Explicit CustomAllReduce ops: 40 attention-side + 1 embedding.
    for phase_ops in (model.context_ops, model.generation_ops):
        allreduce_ops = [op for op in phase_ops if isinstance(op, CustomAllReduce)]
        assert sum(op._scale_factor for op in allreduce_ops) == 41


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.8-2.4T-A95B"])
def test_qwen35_shared_expert_scalar_gate_uses_true_output_width(model_name):
    """The runtime ReplicatedLinear scalar gate is hidden_size -> 1."""
    model = models.get_model(
        model_name,
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        "vllm",
    )

    for phase_ops in (model.context_ops, model.generation_ops):
        scalar_gates = [op for op in _flatten_ops(phase_ops) if op._name.endswith("_shared_expert_gate_gemm")]
        assert len(scalar_gates) == 2
        assert {op._n for op in scalar_gates} == {1}


def test_qwen38max_router_and_moe_dims_match_config():
    """Router GEMM (num_experts=512 clears the >=128 gate) and MoE op dims
    are read straight off the bundled Qwen3.8-Max Qwen35Config: 512 experts,
    top-10 routing, 2048 routed-expert inter size, 2048 shared-expert inter
    size."""
    model = models.get_model("Qwen/Qwen3.8-2.4T-A95B", _model_config(tp_size=1), "vllm")
    cfg = model.extra_params
    assert (cfg.num_experts, cfg.topk, cfg.moe_inter_size, cfg.shared_expert_inter_size) == (512, 10, 2048, 2048)

    for phase_ops, prefixes in (
        (model.context_ops, ("context_gdn", "context_full")),
        (model.generation_ops, ("generation_gdn", "generation_full")),
    ):
        by_name = {op._name: op for op in _flatten_ops(phase_ops)}
        for prefix in prefixes:
            router = by_name[f"{prefix}_router_gemm"]
            assert router._n == 512
            assert router._quant_mode == common.GEMMQuantMode.bfloat16
            moe = by_name[f"{prefix}_moe"]
            assert (moe._inter_size, moe._topk, moe._num_experts) == (2048, 10, 512)
            # shared_gate_up_gemm n = 2 * shared_expert_inter_size // tp; tp=1 here.
            gate_up = by_name[f"{prefix}_shared_gate_up_gemm"]
            assert gate_up._n == 2 * 2048


def test_qwen35_sglang_standard_dispatcher_omits_nonexistent_pre_dispatch():
    """SGLang StandardDispatcher has no collective before routed experts; its
    CUDA-graph decode overlaps shared and routed experts, the scalar gate is
    fused into the post-join merge kernel, and the post all-reduce is serial
    (outside the overlap)."""
    model = models.get_model(
        "Qwen/Qwen3.5-397B-A17B",
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        "sglang",
    )

    for phase, phase_ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        op_names = [op._name for op in _flatten_ops(phase_ops)]
        assert any(name.endswith(("_gdn_ar", "_full_ar")) for name in op_names)
        for prefix in (f"{phase}_gdn", f"{phase}_full"):
            assert f"{prefix}_moe_pre_dispatch" not in op_names
            assert f"{prefix}_moe" in op_names
            assert f"{prefix}_moe_post_dispatch" in op_names
            assert f"{prefix}_shared_expert_gate_gemm" not in op_names
            assert f"{prefix}_shared_expert_gate_mul" not in op_names
            assert f"{prefix}_shared_merge" in op_names
    for op in model.generation_ops:
        if isinstance(op, OverlapOp):
            group_names = [inner._name for inner in _flatten_ops([op])]
            assert not any(name.endswith("_moe_post_dispatch") for name in group_names)
    assert any(isinstance(op, OverlapOp) for op in model.generation_ops)


def test_qwen35_sglang_deepep_prices_one_dispatch_and_replicates_shared_expert():
    """DeepEP rows hold the full dispatch+combine round trip, so one dispatch
    op prices it; sglang DeepEP replicates the shared expert (tp_size=1)
    instead of TP-sharding it."""
    model = models.get_model(
        "Qwen/Qwen3.5-35B-A3B",
        _model_config(tp_size=8, moe_tp_size=1, moe_ep_size=8, moe_backend="deepep_moe"),
        "sglang",
    )
    cfg = model.extra_params

    for phase, phase_ops in (("context", model.context_ops), ("generation", model.generation_ops)):
        op_names = [op._name for op in phase_ops]
        # DeepEP scatters: the attn-TP reduction is NOT folded into a gather.
        assert any(name.endswith(("_gdn_ar", "_full_ar")) for name in op_names)
        for prefix in (f"{phase}_gdn", f"{phase}_full"):
            assert f"{prefix}_moe_pre_dispatch" in op_names
            assert f"{prefix}_moe_post_dispatch" not in op_names
        gate_ups = [op for op in phase_ops if op._name.endswith("_shared_gate_up_gemm")]
        assert {op._n for op in gate_ups} == {2 * cfg.shared_expert_inter_size}
        # Context runs the shared expert on the attn-TP scattered token slice.
        expected_scale = 8 if phase == "context" else 1
        assert {op._scale_num_tokens for op in gate_ups} == {expected_scale}
    # DeepEP keeps shared and routed experts serial in generation.
    assert not any(isinstance(op, OverlapOp) for op in model.generation_ops)


def test_qwen35_sglang_default_moe_keeps_pre_dispatch_under_attention_dp():
    """With DP attention the LayerCommunicator pre-MLP gather is real and
    priced; the attn-TP partial-sum reduction folds into that gather, so the
    per-layer attention AR disappears."""
    model = models.get_model(
        "Qwen/Qwen3.5-35B-A3B",
        _model_config(tp_size=4, moe_tp_size=1, moe_ep_size=8, attention_dp_size=2),
        "sglang",
    )

    for phase_ops in (model.context_ops, model.generation_ops):
        op_names = [op._name for op in _flatten_ops(phase_ops)]
        assert any(name.endswith("_moe_pre_dispatch") for name in op_names)
        assert not any(name.endswith(("_gdn_ar", "_full_ar")) for name in op_names)


def test_qwen35_memory_charges_kv_on_full_layers_and_constant_gdn_state():
    """35B-A3B at tp4: 10 full layers hold per-token KV; 30 GDN layers hold a
    constant per-request state (fp32 SSM + bf16 conv window), TP-sharded."""
    model = models.get_model("Qwen/Qwen3.5-35B-A3B", _model_config(tp_size=4), "vllm")

    assert model.get_kvcache_elements_per_token() == 10 * 2 * 1 * 256
    expected_state = 30 * ((32 // 4) * 128 * 128 * 4 + (2 * 16 * 128 + 32 * 128) // 4 * 3 * 2)
    assert model._gdn_state_bytes_per_request() == expected_state
    per_token_bytes = 2 * model.get_kvcache_elements_per_token()
    assert model.get_kvcache_bytes_per_sequence(4096) == 4096 * per_token_bytes + expected_state
    assert model.get_kvcache_max_tokens(expected_state + 100 * per_token_bytes) == 100


def test_qwen35_bf16_config_threads_dtype_to_every_gdn_op(monkeypatch):
    model_name = "Qwen/Qwen3.5-35B-A3B"
    model_info = copy.deepcopy(core_models._get_model_info(model_name))
    raw_config = model_info["raw_config"]
    text_config = raw_config.get("text_config")
    if isinstance(text_config, dict):
        text_config["mamba_ssm_dtype"] = "bfloat16"
    else:
        raw_config["mamba_ssm_dtype"] = "bfloat16"
    monkeypatch.setattr(core_models, "_get_model_info", lambda _model_path: model_info)

    model = models.get_model(model_name, _model_config(tp_size=4), "sglang")
    gdn_ops = {
        op._name: op
        for op in (*model.context_ops, *model.generation_ops)
        if op._name
        in {
            "context_gdn_conv1d",
            "context_gdn_scan",
            "generation_gdn_conv1d",
            "generation_gdn_recurrence",
        }
    }

    assert set(gdn_ops) == {
        "context_gdn_conv1d",
        "context_gdn_scan",
        "generation_gdn_conv1d",
        "generation_gdn_recurrence",
    }
    assert {op._mamba_ssm_dtype for op in gdn_ops.values()} == {"bfloat16"}


@pytest.mark.parametrize(
    ("configured_dtype", "resolved_dtype", "ssm_element_bytes"),
    [("bfloat16", "bfloat16", 2), ("float8_e4m3", "float32", 4)],
)
def test_qwen35_custom_config_sizes_only_supported_ssm_dtypes(
    monkeypatch, configured_dtype, resolved_dtype, ssm_element_bytes
):
    model_name = "Qwen/Qwen3.5-35B-A3B"
    model_info = copy.deepcopy(core_models._get_model_info(model_name))
    raw_config = model_info["raw_config"]
    text_config = raw_config.get("text_config")
    if isinstance(text_config, dict):
        text_config["mamba_ssm_dtype"] = configured_dtype
    else:
        raw_config["mamba_ssm_dtype"] = configured_dtype
    monkeypatch.setattr(core_models, "_get_model_info", lambda _model_path: model_info)

    model = models.get_model(model_name, _model_config(tp_size=4), "sglang")

    cfg = model.extra_params
    n_gdn = cfg.layer_types.count("linear_attention")
    state_bytes = (
        (cfg.linear_num_value_heads // 4) * cfg.linear_key_head_dim * cfg.linear_value_head_dim * ssm_element_bytes
    )
    conv_bytes = (
        (
            2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim
        )
        // 4
        * (cfg.linear_conv_kernel_dim - 1)
        * 2
    )
    expected_state = n_gdn * (state_bytes + conv_bytes)

    assert model._mamba_ssm_dtype == resolved_dtype
    assert model._gdn_state_bytes_per_request() == expected_state
    per_token_bytes = 2 * model.get_kvcache_elements_per_token()
    assert model.get_kvcache_bytes_per_sequence(4096) == 4096 * per_token_bytes + expected_state
    assert model.get_kvcache_max_tokens(expected_state + 100 * per_token_bytes) == 100


@pytest.mark.parametrize(
    ("tp_size", "expected_elements_per_token", "expected_state_bytes"),
    [
        (1, 23 * 2 * 4 * 256, 69 * (128 * 128 * 128 * 4 + (2 * 16 * 128 + 128 * 128) * 3 * 2)),
        (
            8,
            23 * 2 * 1 * 256,
            69 * ((128 // 8) * 128 * 128 * 4 + (2 * 16 * 128 + 128 * 128) // 8 * 3 * 2),
        ),
    ],
)
def test_qwen38max_memory_charges_kv_on_full_layers_and_constant_gdn_state(
    tp_size, expected_elements_per_token, expected_state_bytes
):
    """Qwen3.8-Max: 23 full-attention layers hold per-token KV (num_kv_heads=4,
    ceil-sharded across tp); 69 GDN layers hold a constant per-request state
    (fp32 SSM state + bf16 conv window, linear_conv_kernel_dim=4), TP-sharded.
    tp=8 exercises ceil(4/8)=1, where per-token KV elements stop shrinking."""
    model = models.get_model("Qwen/Qwen3.8-2.4T-A95B", _model_config(tp_size=tp_size), "vllm")

    assert model.get_kvcache_elements_per_token() == expected_elements_per_token
    assert model._gdn_state_bytes_per_request() == expected_state_bytes

    per_token_bytes = 2 * expected_elements_per_token  # bf16 kvcache
    assert model.get_kvcache_bytes_per_sequence(4096) == 4096 * per_token_bytes + expected_state_bytes
    assert model.get_kvcache_max_tokens(expected_state_bytes + 100 * per_token_bytes) == 100


# Qwen/Qwen3.6-35B-A3B-FP8 ``quantization_config.modules_to_not_convert``
# (native fp8_block, no hf_quant_config), per-layer entries collapsed to
# layer 0 and the visual tower omitted: only norms, the router, the scalar
# shared-expert gate and the non-GEMM GDN parameters are kept in BF16.
_QWEN36_FP8_EXCLUSIONS = (
    "lm_head",
    "model.embed_tokens",
    "model.language_model.layers.0.input_layernorm",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.linear_attn.conv1d",
    "model.language_model.layers.0.linear_attn.dt_bias",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.in_proj_ba",
    "model.language_model.layers.0.linear_attn.norm",
    "model.language_model.layers.0.mlp.gate",
    "model.language_model.layers.0.mlp.shared_expert_gate",
    "model.language_model.layers.0.post_attention_layernorm",
    "model.language_model.layers.0.self_attn.k_norm",
    "model.language_model.layers.0.self_attn.q_norm",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
    "mtp.fc",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.self_attn.q_norm",
    "mtp.norm",
)

_FP8 = common.GEMMQuantMode.fp8_block
_BF16 = common.GEMMQuantMode.bfloat16


def _fp8_raw_config(exclusions, *, key="modules_to_not_convert"):
    return {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3", key: list(exclusions)}}


def _resolve_gemm_modes(exclusions, **kwargs):
    return core_models.qwen35._qwen35_mixed_precision_gemm_modes(
        _fp8_raw_config(exclusions, **kwargs), _FP8, allow_checkpoint_split=True
    )


def test_qwen36_fp8_native_exclusions_keep_every_gemm_category_quantized():
    # Regression: substring matching on ``linear_attn``/``self_attn``/``.mlp``
    # demoted every projection and shared-expert GEMM to BF16 although no GEMM
    # is excluded (the fp8_block lane was never used for this checkpoint).
    # in_proj_a / in_proj_b ARE excluded, so the GDN b/a GEMM stays BF16.
    assert _resolve_gemm_modes(_QWEN36_FP8_EXCLUSIONS) == (_FP8, _BF16, _FP8, _FP8)


@pytest.mark.parametrize(
    "exclusions",
    [
        ["model.language_model.layers.3.self_attn.q_proj"],
        ["model.layers.0.linear_attn.out_proj"],
        ["linear_attn.in_proj_qkvz"],
        ["self_attn"],
        ["model.language_model.layers.0.linear_attn"],
        ["model.language_model.layers.0.self_attn*"],
        ["layers.*.self_attn.q_proj"],
        ["re:.*linear_attn.*"],
    ],
    ids=["q_proj", "out_proj", "bare_in_proj", "bare_block", "block_path", "modelopt_glob", "layer_glob", "regex"],
)
def test_qwen35_projection_gemm_exclusion_demotes_only_projection(exclusions):
    # The GDN b/a GEMM follows the projection mode unless excluded on its own.
    assert _resolve_gemm_modes(exclusions) == (_BF16, _BF16, _FP8, _FP8)


def test_qwen35_gdn_ba_exclusion_demotes_only_the_ba_gemm():
    assert _resolve_gemm_modes(["model.layers.0.linear_attn.in_proj_ba"]) == (_FP8, _BF16, _FP8, _FP8)
    assert _resolve_gemm_modes(["linear_attn.in_proj_a", "linear_attn.in_proj_b"]) == (_FP8, _BF16, _FP8, _FP8)
    assert _resolve_gemm_modes([]) == (_FP8, _FP8, _FP8, _FP8)


@pytest.mark.parametrize(
    "exclusions",
    [
        ["model.layers.0.mlp.shared_expert.down_proj"],
        ["*.mlp.shared_expert.*"],
        ["model.language_model.layers.0.mlp.shared_expert*"],
    ],
    ids=["down_proj", "glob", "modelopt_glob"],
)
def test_qwen35_shared_expert_gemm_exclusion_demotes_only_shared_expert(exclusions):
    assert _resolve_gemm_modes(exclusions) == (_FP8, _FP8, _FP8, _BF16)


def test_qwen35_shared_expert_gate_and_router_exclusions_do_not_demote_gemms():
    # ``shared_expert_gate`` is the scalar gate, not a shared-expert GEMM;
    # ``mlp.gate`` is the router, not a dense-FFN GEMM.
    assert _resolve_gemm_modes(["model.layers.0.mlp.shared_expert_gate", "model.layers.0.mlp.gate"]) == (
        _FP8,
        _FP8,
        _FP8,
        _FP8,
    )


def test_qwen35_dense_ffn_exclusion_ignores_routed_experts():
    assert _resolve_gemm_modes(["model.layers.0.mlp.gate_proj"], key="ignore") == (_FP8, _FP8, _BF16, _FP8)
    assert _resolve_gemm_modes(["model.layers.0.mlp.experts.5.gate_proj", "re:.*mlp\\.experts.*"]) == (
        _FP8,
        _FP8,
        _FP8,
        _FP8,
    )
    # A whole-FFN exclusion covers the dense FFN and the shared expert alike.
    assert _resolve_gemm_modes(["model.layers.0.mlp"]) == (_FP8, _FP8, _BF16, _BF16)


@pytest.mark.parametrize(
    "exclusions",
    [["model.layers.0"], ["model.language_model.layers.7"], ["layers.0"], ["model.language_model.layers.0.*"]],
    ids=["flat_literal", "nested_literal", "bare_literal", "glob"],
)
def test_qwen35_whole_layer_exclusion_demotes_every_category(exclusions):
    assert _resolve_gemm_modes(exclusions) == (_BF16, _BF16, _BF16, _BF16)


def test_qwen35_regex_exclusions_anchor_at_the_module_path_start_like_compressed_tensors():
    # compressed-tensors applies ``re:`` patterns with ``re.match``: an
    # unanchored router regex only covers ``mlp.gate_proj`` when it starts
    # with ``.*`` (as it would in compressed-tensors itself).
    assert _resolve_gemm_modes(["re:mlp\\.gate"]) == (_FP8, _FP8, _FP8, _FP8)
    assert _resolve_gemm_modes(["re:.*mlp\\.gate$"]) == (_FP8, _FP8, _FP8, _FP8)
    assert _resolve_gemm_modes(["re:.*mlp\\.gate"]) == (_FP8, _FP8, _BF16, _FP8)
    assert _resolve_gemm_modes(["re:model\\.layers\\.\\d+\\.self_attn\\..*"]) == (_BF16, _BF16, _FP8, _FP8)


def test_qwen35_bare_substring_and_parameter_exclusions_are_not_honored():
    # Documented divergence from transformers' substring test.
    assert _resolve_gemm_modes(["attn", "proj", "model.layers.0.self_attn.q_proj.weight"]) == (
        _FP8,
        _FP8,
        _FP8,
        _FP8,
    )


def test_qwen35_unrecognized_or_foreign_exclusions_do_not_demote_gemms():
    assert _resolve_gemm_modes(
        ["foo.bar", "mtp*", "mtp.layers.0.self_attn.q_proj", "visual.blocks.0.attn.qkv_proj"]
    ) == (
        _FP8,
        _FP8,
        _FP8,
        _FP8,
    )
    assert _resolve_gemm_modes(["re:*["]) == (_FP8, _FP8, _FP8, _FP8)


def test_qwen35_explicit_quantized_layers_override_exclusions():
    raw_config = _fp8_raw_config(["model.language_model.layers.0.self_attn*", "*.mlp.shared_expert.*"])
    raw_config["hf_quant_config"] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                "model.language_model.layers.0.mlp.shared_expert.up_proj": {"quant_algo": "NVFP4"},
            },
        }
    }
    modes = core_models.qwen35._qwen35_mixed_precision_gemm_modes(
        raw_config, common.GEMMQuantMode.nvfp4, allow_checkpoint_split=True
    )
    assert modes == (
        common.GEMMQuantMode.fp8_static,
        common.GEMMQuantMode.fp8_static,
        common.GEMMQuantMode.nvfp4,
        common.GEMMQuantMode.nvfp4,
    )
    assert (
        core_models.qwen35._qwen35_mixed_precision_gemm_modes(
            raw_config, common.GEMMQuantMode.nvfp4, allow_checkpoint_split=False
        )
        == (common.GEMMQuantMode.nvfp4,) * 4
    )


def _mixed_precision_raw_config(quantized_layers, exclusions=()):
    raw_config = _fp8_raw_config(exclusions)
    raw_config["hf_quant_config"] = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {target: {"quant_algo": algo} for target, algo in quantized_layers.items()},
        }
    }
    return raw_config


_GDN_PROJECTIONS_FP8 = {
    "model.language_model.layers.0.linear_attn.in_proj_qkv": "FP8",
    "model.language_model.layers.0.linear_attn.in_proj_z": "FP8",
    "model.language_model.layers.0.linear_attn.out_proj": "FP8",
}


def test_qwen35_layer_map_that_enumerates_gdn_projections_without_in_proj_ab_keeps_the_ba_gemm_bf16():
    # ModelOpt ``quantized_layers`` lists every quantized module: naming the
    # GDN projections but neither in_proj_a/in_proj_b nor in_proj_ba means
    # those weights are stored in BF16 even without an exclusion entry.
    modes = core_models.qwen35._qwen35_mixed_precision_gemm_modes(
        _mixed_precision_raw_config(_GDN_PROJECTIONS_FP8), common.GEMMQuantMode.nvfp4, allow_checkpoint_split=True
    )
    assert modes == (
        common.GEMMQuantMode.fp8_static,
        common.GEMMQuantMode.bfloat16,
        common.GEMMQuantMode.nvfp4,
        common.GEMMQuantMode.nvfp4,
    )
    # A map that names only full-attention projections says nothing about the
    # GDN block, so the b/a GEMM keeps following the projection mode.
    modes = core_models.qwen35._qwen35_mixed_precision_gemm_modes(
        _mixed_precision_raw_config({"model.language_model.layers.3.self_attn.q_proj": "FP8"}),
        common.GEMMQuantMode.nvfp4,
        allow_checkpoint_split=True,
    )
    assert modes[:2] == (common.GEMMQuantMode.fp8_static, common.GEMMQuantMode.fp8_static)


@pytest.mark.parametrize("ba_target", ["linear_attn.in_proj_ba", "linear_attn.in_proj_a"])
def test_qwen35_explicit_gdn_ba_entry_is_authoritative_over_exclusions(ba_target):
    quantized_layers = {**_GDN_PROJECTIONS_FP8, f"model.language_model.layers.0.{ba_target}": "NVFP4"}
    exclusions = [
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
    ]
    modes = core_models.qwen35._qwen35_mixed_precision_gemm_modes(
        _mixed_precision_raw_config(quantized_layers, exclusions),
        common.GEMMQuantMode.nvfp4,
        allow_checkpoint_split=True,
    )
    assert modes[:2] == (common.GEMMQuantMode.fp8_static, common.GEMMQuantMode.nvfp4)


@pytest.mark.parametrize(
    "hf_id,default,expected",
    [
        ("Qwen/Qwen3.5-27B", _BF16, (_BF16, _BF16, _BF16, _BF16)),
        ("Qwen/Qwen3.5-35B-A3B", _BF16, (_BF16, _BF16, _BF16, _BF16)),
        ("Qwen/Qwen3.5-122B-A10B", _BF16, (_BF16, _BF16, _BF16, _BF16)),
        ("Qwen/Qwen3.5-397B-A17B", _BF16, (_BF16, _BF16, _BF16, _BF16)),
        # Whole-block ``layers.N.linear_attn*`` / ``self_attn*`` /
        # ``mlp.shared_expert*`` globs keep projections and shared experts BF16.
        (
            "nvidia/Qwen3.5-122B-A10B-NVFP4",
            common.GEMMQuantMode.nvfp4,
            (_BF16, _BF16, common.GEMMQuantMode.nvfp4, _BF16),
        ),
        (
            "nvidia/Qwen3.5-397B-A17B-NVFP4",
            common.GEMMQuantMode.nvfp4,
            (_BF16, _BF16, common.GEMMQuantMode.nvfp4, _BF16),
        ),
        # ModelOpt MIXED_PRECISION maps: FP8 projections, W4A16 NVFP4 FFN, and
        # in_proj_a/in_proj_b absent from ``quantized_layers`` (stored BF16).
        (
            "nvidia/Qwen3.6-27B-NVFP4",
            common.GEMMQuantMode.nvfp4,
            (common.GEMMQuantMode.fp8_static, _BF16, common.GEMMQuantMode.w4a16_nvfp4, common.GEMMQuantMode.nvfp4),
        ),
        (
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
            common.GEMMQuantMode.nvfp4,
            (common.GEMMQuantMode.fp8_static, _BF16, common.GEMMQuantMode.nvfp4, common.GEMMQuantMode.w4a16_nvfp4),
        ),
        ("Qwen/Qwen3.8-2.4T-A95B", _BF16, (_BF16, _BF16, _BF16, _BF16)),
        # Explicit q/k/v/o, in_proj_*, out_proj and shared_expert.* exclusions;
        # the only ``.mlp`` exclusions are the router and the shared expert, so
        # dense_ffn stays fp8_block (inert: dense FFN ops exist only for num_experts == 0).
        ("Qwen/Qwen3.8-2.4T-A95B-FP8", _FP8, (_BF16, _BF16, _FP8, _BF16)),
    ],
)
def test_bundled_qwen_configs_resolve_expected_gemm_modes(hf_id, default, expected):
    raw_config = core_models._get_model_info(hf_id)["raw_config"]
    assert (
        core_models.qwen35._qwen35_mixed_precision_gemm_modes(raw_config, default, allow_checkpoint_split=True)
        == expected
    )


@pytest.mark.parametrize("hf_id", ["nvidia/Qwen3.6-27B-NVFP4", "nvidia/Qwen3.6-35B-A3B-NVFP4"])
def test_qwen36_nvfp4_checkpoints_price_the_gdn_ba_gemm_in_bf16(hf_id):
    model_config = sdk_config.ModelConfig(
        tp_size=4,
        pp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        attention_dp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    model_config._gemm_quant_mode_is_explicit = False
    model = models.get_model(hf_id, model_config, "trtllm")
    by_name = {op._name: op for op in _flatten_ops(model.context_ops + model.generation_ops)}
    for name in ("context_gdn_in_proj_ba_gemm", "generation_gdn_in_proj_ba_gemm"):
        assert by_name[name]._quant_mode == _BF16, name
    for name in ("context_gdn_in_proj_gemm", "generation_gdn_out_proj_gemm"):
        assert by_name[name]._quant_mode == common.GEMMQuantMode.fp8_static, name


def test_qwen36_fp8_native_checkpoint_threads_fp8_block_to_projection_and_shared_gemms(monkeypatch):
    model_name = "Qwen/Qwen3.5-35B-A3B"
    model_info = copy.deepcopy(core_models._get_model_info(model_name))
    model_info["raw_config"]["quantization_config"] = _fp8_raw_config(_QWEN36_FP8_EXCLUSIONS)["quantization_config"]
    model_info["gemm_quant_mode_is_explicit"] = False
    monkeypatch.setattr(core_models, "_get_model_info", lambda _model_path: model_info)

    model_config = sdk_config.ModelConfig(
        tp_size=4,
        pp_size=1,
        moe_tp_size=4,
        moe_ep_size=1,
        attention_dp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    )
    model_config._gemm_quant_mode_is_explicit = False
    model = models.get_model(model_name, model_config, "sglang")
    by_name = {op._name: op for op in _flatten_ops(model.context_ops + model.generation_ops)}

    for name in (
        "context_gdn_in_proj_gemm",
        "context_gdn_out_proj_gemm",
        "context_qkv_gemm",
        "context_proj_gemm",
        "context_gdn_shared_gate_up_gemm",
        "context_gdn_shared_down_gemm",
        "generation_full_shared_down_gemm",
    ):
        assert by_name[name]._quant_mode == _FP8, name
    for name in ("context_gdn_in_proj_ba_gemm", "generation_gdn_in_proj_ba_gemm"):
        assert by_name[name]._quant_mode == _BF16, name
