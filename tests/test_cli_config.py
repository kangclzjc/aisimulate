# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from copy import deepcopy

import pytest
from pydantic import ValidationError

from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.config.common import split_config_sections
from aisimulate.config.engine import (
    EngineRecommendationConfig,
    WorkerPredictionConfig,
    WorkersPredictionConfig,
)
from aisimulate.recommend import _candidate_prediction, recommendation_to_sweeper
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.deploy import build_backend_deployment
from aisimulate.sweeper.parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import ReplaySpec
from aisimulate.sweeper.sample import unroll_sample


def _engine() -> dict:
    return {
        "model": "example/model",
        "hardware": "h200_sxm",
        "workers": {"aggregated": {}},
    }


@pytest.mark.parametrize(
    "load",
    [
        {"type": "concurrency", "concurrency": 32},
        {"type": "constant_rate", "requests_per_second": 10},
        {"type": "poisson", "requests_per_second": 10, "seed": 17},
    ],
)
@pytest.mark.parametrize("minimum", [9, 10])
def test_min_gpus_lowers_fixed_traffic_and_load_constraint(load, minimum):
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {**_engine(), "mode": "aggregated", "context_length": 4096},
            "traffic": {"source": {"type": "synthetic"}, "load": load, "stop": {"requests": 100}},
            "evaluation": {"sla": {"itl_ms": 30}},
            "optimization": {"target": "min_gpus", "constraints": {"min_goodput_rps": minimum}},
        }
    )
    lowered = recommendation_to_sweeper(config)
    assert lowered.goal.target.value == "min_gpus"
    assert lowered.goal.requires_aggregate_sla
    assert lowered.goal.min_goodput_rps == minimum
    assert lowered.goal.sla.itl_ms == 30
    assert lowered.workload.load_type == load["type"]
    if load["type"] == "concurrency":
        assert lowered.workload.concurrency == load["concurrency"]
        assert lowered.workload.request_rate is None
    else:
        assert lowered.workload.request_rate == load["requests_per_second"]
        assert lowered.workload.concurrency is None
        if load["type"] == "poisson":
            assert lowered.workload.arrival_seed == load["seed"]


@pytest.mark.parametrize(
    ("load", "minimum", "error"),
    [
        ({"type": "constant_rate", "requests_per_second": 10}, None, "requires constraints.min_goodput_rps"),
        ({"type": "constant_rate", "requests_per_second": 10}, 11, "cannot exceed"),
        ({"type": "concurrency", "concurrency": {"choices": [1, 32]}}, None, "fixed synthetic"),
        ({"type": "kv_capacity_fraction", "fraction": 0.5}, None, "fixed synthetic"),
    ],
)
def test_min_gpus_rejects_missing_capacity_target_or_variable_load(load, minimum, error):
    with pytest.raises(ValidationError, match=error):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {**_engine(), "mode": "aggregated"},
                "traffic": {"source": {"type": "synthetic"}, "load": load, "stop": {"requests": 100}},
                "evaluation": {"sla": {"itl_ms": 30}},
                "optimization": {"target": "min_gpus", "constraints": {"min_goodput_rps": minimum}},
            }
        )


def test_prediction_uses_reviewed_default_traffic() -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine()})

    assert config.traffic.source.type == "synthetic"
    assert config.traffic.load.type == "concurrency"
    assert config.traffic.load.concurrency == 10
    assert config.traffic.stop is not None
    assert config.traffic.stop.requests == 100


def test_prediction_scheduler_defaults_are_role_aware() -> None:
    aggregated = CorePredictionConfig.model_validate({"engine": _engine()})
    assert aggregated.engine.workers.aggregated is not None
    assert aggregated.engine.workers.aggregated.scheduler.max_batched_tokens == 8192
    assert aggregated.engine.workers.aggregated.scheduler.max_sequences == 256
    assert aggregated.engine.workers.aggregated.scheduler.prefill_schedule_interval == 1
    assert aggregated.engine.workers.aggregated.scheduler.prefill_decode_interval == 0

    disaggregated = CorePredictionConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "disaggregated",
                "workers": {"prefill": {}, "decode": {}},
            }
        }
    )
    assert disaggregated.engine.workers.prefill is not None
    assert disaggregated.engine.workers.decode is not None
    assert disaggregated.engine.workers.prefill.scheduler.max_batched_tokens == 8192
    assert disaggregated.engine.workers.prefill.scheduler.max_sequences == 1
    assert disaggregated.engine.workers.prefill.scheduler.prefill_schedule_interval == 1
    assert disaggregated.engine.workers.prefill.scheduler.prefill_decode_interval == 0
    assert disaggregated.engine.workers.decode.scheduler.max_batched_tokens == 8192
    assert disaggregated.engine.workers.decode.scheduler.max_sequences == 256
    assert disaggregated.engine.workers.decode.scheduler.prefill_schedule_interval == 1
    assert disaggregated.engine.workers.decode.scheduler.prefill_decode_interval == 0

    programmatic = WorkersPredictionConfig(prefill=WorkerPredictionConfig(), decode=WorkerPredictionConfig())
    assert programmatic.prefill is not None
    assert programmatic.decode is not None
    assert programmatic.prefill.scheduler.max_sequences == 1
    assert programmatic.decode.scheduler.max_sequences == 256


def test_prediction_rejects_nonpositive_prefill_schedule_interval() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"scheduler": {"prefill_schedule_interval": 0}}

    with pytest.raises(ValidationError, match="prefill_schedule_interval"):
        CorePredictionConfig.model_validate({"engine": engine})


@pytest.mark.parametrize("value", [0, 1, 20])
def test_prediction_accepts_sglang_prefill_decode_interval(value: int) -> None:
    engine = _engine()
    engine["backend"] = "sglang"
    engine["workers"]["aggregated"] = {"scheduler": {"prefill_decode_interval": value}}

    config = CorePredictionConfig.model_validate({"engine": engine})

    assert config.engine.workers.aggregated.scheduler.prefill_decode_interval == value
    assert config.engine.workers.aggregated.scheduler.prefill_schedule_interval == 1


@pytest.mark.parametrize("value", [-1, 1.5, True, "20"])
def test_prediction_rejects_invalid_prefill_decode_interval(value) -> None:
    engine = _engine()
    engine["backend"] = "sglang"
    engine["workers"]["aggregated"] = {"scheduler": {"prefill_decode_interval": value}}

    with pytest.raises(ValidationError, match="prefill_decode_interval"):
        CorePredictionConfig.model_validate({"engine": engine})


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_prediction_accepts_neutral_backend_scheduler_defaults(backend: str) -> None:
    engine = _engine()
    engine["backend"] = backend
    engine["workers"]["aggregated"] = {"scheduler": {"prefill_schedule_interval": 1, "prefill_decode_interval": 0}}

    CorePredictionConfig.model_validate({"engine": engine})


@pytest.mark.parametrize(
    "backend,field,value,required_backend",
    [
        ("vllm", "prefill_decode_interval", 1, "sglang"),
        ("trtllm", "prefill_decode_interval", 1, "sglang"),
        ("sglang", "prefill_schedule_interval", 2, "vllm"),
        ("trtllm", "prefill_schedule_interval", 2, "vllm"),
    ],
)
@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
def test_prediction_rejects_wrong_backend_scheduler_interval(backend, field, value, required_backend, role) -> None:
    engine = _engine()
    engine["backend"] = backend
    if role != "aggregated":
        engine["mode"] = "disaggregated"
        engine["workers"] = {"prefill": {}, "decode": {}}
    engine["workers"][role] = {"scheduler": {field: value}}

    with pytest.raises(
        ValidationError,
        match=rf"workers.{role}.scheduler.{field}.*backend={required_backend}",
    ):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_accepts_trtllm_disaggregated_dp1() -> None:
    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "disaggregated",
                "backend": "trtllm",
                "workers": {
                    "prefill": {"parallelism": {"attention_data": 1}},
                    "decode": {"parallelism": {"attention_data": 1}},
                },
            }
        }
    )

    assert config.engine.backend == "trtllm"
    assert config.engine.mode == "disaggregated"
    assert config.engine.workers.prefill is not None
    assert config.engine.workers.decode is not None


@pytest.mark.parametrize("value", [-1, (1 << 53) + 1])
def test_prediction_rejects_invalid_cuda_graph_reservation(value: int) -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {
        "kv_cache": {
            "capacity": {
                "type": "default",
                "cuda_graph_reserved_bytes": value,
            }
        }
    }

    with pytest.raises(ValidationError, match="cuda_graph_reserved_bytes"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_rejects_cuda_graph_reservation_with_fixed_capacity() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {
        "kv_cache": {
            "capacity": {
                "type": "fixed",
                "blocks": 128,
                "cuda_graph_reserved_bytes": 1 << 30,
            }
        }
    }

    with pytest.raises(ValidationError, match="fixed KV capacity rejects cuda_graph_reserved_bytes"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_rejects_recommendation_domain() -> None:
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "backend": {"choices": ["vllm", "sglang"]},
                }
            }
        )


def test_synthetic_session_rate_uses_session_units() -> None:
    config = CorePredictionConfig.model_validate(
        {
            "engine": _engine(),
            "traffic": {
                "source": {
                    "type": "synthetic-session",
                    "new_input_tokens_per_turn": 128,
                    "output_tokens_per_turn": 16,
                    "session": {"turns": 3},
                },
                "load": {
                    "type": "constant_rate",
                    "sessions_per_second": 4,
                },
                "stop": {"sessions": 20},
            },
        }
    )

    assert config.traffic.source.type == "synthetic-session"
    assert config.traffic.load.sessions_per_second == 4


def test_recommendation_rejects_unknown_nested_field() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {**_engine(), "mystery": 1},
                "optimization": {},
            }
        )


def test_recommendation_accepts_domains_and_parallel_preset() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": {"choices": ["aggregated", "disaggregated"]},
                "model": "example/model",
                "hardware": "auto",
                "backend": {"choices": ["vllm", "sglang"]},
                "workers": {
                    role: {"parallelism": {"preset": "default"}} for role in ("aggregated", "prefill", "decode")
                },
            },
            "optimization": {"hardware": "h200_sxm"},
        }
    )

    assert config.optimization.hardware == "h200_sxm"
    assert isinstance(config.engine, EngineRecommendationConfig)


def test_parallel_preset_modes_lower_without_conflating_semantics() -> None:
    independent = {
        "preset": False,
        "replicas": {"range": {"min": 1, "max": 2, "step": 1}},
    }
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {"aggregated": {"parallelism": independent}},
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_configs_by_mode == {}
    assert lowered.search_space.parallel_independent_by_mode["agg"]["replicas"] == [
        1,
        2,
    ]


def test_custom_parallel_preset_lowers_as_flat_atomic_choices() -> None:
    mapping = {
        "replicas": 1,
        "tensor": 1,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {"aggregated": {"parallelism": {"preset": [mapping]}}},
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.flat_parallel_modes == ["agg"]
    assert lowered.search_space.parallel_configs_by_mode["agg"] == [
        {
            "replicas": 1,
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
        }
    ]


def test_present_non_core_section_is_split_for_adapter_validation() -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
        },
        "optimization": {},
        "placement": {"policy": {"choices": ["first", "least_loaded"]}},
    }
    core, adapters = split_config_sections(base, command="recommend")
    config = CoreRecommendationConfig.model_validate(core)
    lowered = recommendation_to_sweeper(config, adapter_configs=adapters, stack="example")

    assert "placement" not in core
    assert adapters == {"placement": {"policy": {"choices": ["first", "least_loaded"]}}}
    assert lowered.adapters["example.placement"].search_space == adapters["placement"]


def test_core_models_reject_stack_owned_sections() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {**_engine(), "context_length": 4096},
                "placement": {"policy": "first"},
                "optimization": {},
            }
        )


@pytest.mark.parametrize("section", ["", "nested.section", 7])
def test_adapter_section_names_are_unambiguous(section) -> None:
    with pytest.raises(ValueError, match="section names without dots"):
        split_config_sections(
            {
                "engine": {**_engine(), "context_length": 4096},
                "optimization": {},
                section: {},
            },
            command="recommend",
        )


def test_parallel_custom_preset_requires_complete_strict_mapping() -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
            "workers": {"aggregated": {"parallelism": {"preset": [{"replicas": 2}]}}},
        },
        "optimization": {},
    }
    with pytest.raises(ValidationError, match="cover exactly all knobs"):
        CoreRecommendationConfig.model_validate(base)


@pytest.mark.parametrize("invalid", [1.9, True, "2"])
def test_parallel_custom_preset_rejects_coercible_leaves(invalid) -> None:
    mapping = {
        "replicas": 1,
        "tensor": invalid,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "mode": "aggregated",
                    "context_length": 4096,
                    "workers": {"aggregated": {"parallelism": {"preset": [mapping]}}},
                },
                "optimization": {},
            }
        )


def test_engine_scheduler_domains_replace_defaults_and_preserve_log_scale() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "aggregated",
                "backend_version": "0.19.0",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {"preset": "default"},
                        "scheduler": {
                            "max_batched_tokens": {"choices": [4096]},
                            "max_sequences": {"range": {"min": 1, "max": 8, "scale": "log"}},
                        },
                    }
                },
            },
            "optimization": {},
        }
    )
    lowered = recommendation_to_sweeper(config)
    assert lowered.search_space.agg_max_num_batched_tokens == [4096]
    assert lowered.search_space.agg_max_num_seqs == [1]
    assert lowered.search_space.engine_integer_log_ranges["agg_max_num_seqs"] == [
        1,
        8,
    ]
    assert lowered.search_space.backend_version == "0.19.0"


def test_large_integer_log_ranges_lower_as_compact_bounds() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "mode": "aggregated",
                "context_length": 4096,
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "preset": False,
                            "replicas": {"range": {"min": 1, "max": 1_000_000, "scale": "log"}},
                        }
                    }
                },
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_independent_by_mode["agg"]["replicas"] == [1]
    assert lowered.search_space.parallel_independent_log_ranges_by_mode["agg"]["replicas"] == [
        1,
        1_000_000,
    ]


def test_disagg_mixed_parallel_presets_preserve_each_role_semantics() -> None:
    mapping = {
        "replicas": 1,
        "tensor": 1,
        "pipeline": 1,
        "attention_data": 1,
        "moe_tensor": 1,
        "moe_expert": 1,
    }
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "disaggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "prefill": {"parallelism": {"preset": [mapping]}},
                    "decode": {
                        "parallelism": {
                            "preset": False,
                            "replicas": {"choices": [1, 2]},
                        }
                    },
                },
            },
            "optimization": {},
        }
    )

    lowered = recommendation_to_sweeper(config)

    assert lowered.search_space.parallel_custom_configs_by_mode["disagg"]["prefill"] == [
        {
            "replicas": 1,
            "tp": 1,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
        }
    ]
    assert lowered.search_space.parallel_independent_by_mode["disagg"]["decode_replicas"] == [
        1,
        2,
    ]
    assert not any(name.startswith("prefill_") for name in lowered.search_space.parallel_independent_by_mode["disagg"])


def test_backend_specific_block_size_validation() -> None:
    with pytest.raises(ValidationError, match="require block_size >= 2"):
        CorePredictionConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "backend": "vllm",
                    "workers": {"aggregated": {"kv_cache": {"block_size": 1}}},
                }
            }
        )

    config = CorePredictionConfig.model_validate(
        {
            "engine": {
                **_engine(),
                "backend": "sglang",
                "workers": {"aggregated": {"kv_cache": {"block_size": 1}}},
            }
        }
    )
    assert config.engine.workers.aggregated.kv_cache.block_size == 1


def test_prediction_rejects_auto_hardware_and_kv_relative_load() -> None:
    with pytest.raises(ValidationError, match="recommendation-only"):
        CorePredictionConfig.model_validate({"engine": {**_engine(), "hardware": "auto"}})
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(
            {
                "engine": _engine(),
                "traffic": {
                    "source": {"type": "synthetic"},
                    "load": {"type": "kv_capacity_fraction", "fraction": 1.2},
                    "stop": {"requests": 10},
                },
            }
        )


def test_traffic_and_optimizer_strict_defaults_and_finite_values() -> None:
    prediction = CorePredictionConfig.model_validate(
        {
            "engine": _engine(),
            "traffic": {
                "source": {"type": "synthetic"},
                "load": {},
                "stop": {"requests": 10},
            },
        }
    )
    assert prediction.traffic.load.concurrency == 10
    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "mode": "aggregated",
                    "context_length": 4096,
                },
                "optimization": {"hardware": "auto"},
            }
        )


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0), ("e2e_ms", 1000.0)])
@pytest.mark.parametrize("target", ["goodput", "goodput_per_gpu"])
def test_goodput_requires_at_least_one_sla_bound_in_typed_cli_config(target: str, field: str, bound: float) -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
        },
        "optimization": {"target": target},
    }
    with pytest.raises(ValidationError, match="requires an evaluation.sla bound"):
        CoreRecommendationConfig.model_validate(base)
    accepted = CoreRecommendationConfig.model_validate({**base, "evaluation": {"sla": {field: bound}}})
    assert getattr(accepted.evaluation.sla, field) == bound


def test_strict_sla_is_public_and_lowers_to_sweeper_goal() -> None:
    base = {
        "engine": {
            **_engine(),
            "mode": "aggregated",
            "context_length": 4096,
        },
        "optimization": {"target": "throughput", "strict_sla": True},
    }
    with pytest.raises(ValidationError, match="requires at least one"):
        CoreRecommendationConfig.model_validate(base)

    public = CoreRecommendationConfig.model_validate({**base, "evaluation": {"sla": {"itl_ms": 30}}})
    lowered = recommendation_to_sweeper(public)

    assert public.optimization.strict_sla is True
    assert lowered.goal.strict_sla is True
    assert lowered.goal.sla is not None
    assert lowered.goal.sla.itl_ms == 30

    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate({**base, "optimization": {"strict_sla": "true"}})


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_prediction_accepts_independent_token_sla_bounds(field: str, bound: float) -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine(), "evaluation": {"sla": {field: bound}}})

    assert getattr(config.evaluation.sla, field) == bound


@pytest.mark.parametrize(("field", "bound"), [("ttft_ms", 800.0), ("itl_ms", 30.0)])
def test_recommendation_accepts_independent_token_sla_bounds(field: str, bound: float) -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {**_engine(), "mode": "aggregated"},
            "evaluation": {"sla": {field: bound}},
            "optimization": {},
        }
    )

    assert getattr(config.evaluation.sla, field) == bound
    assert config.optimization.strict_sla is False


def test_kv_relative_load_choices_lower_as_generic_search_dimension() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "synthetic"},
                "load": {
                    "type": "kv_capacity_fraction",
                    "fraction": {"choices": [0.5, 1.5]},
                },
                "stop": {"requests": 10},
            },
            "engine": {
                **_engine(),
                "mode": "aggregated",
                "context_length": 4096,
            },
            "optimization": {},
        }
    )
    lowered = recommendation_to_sweeper(config)
    assert lowered.workload.load_search_field == "kv_load_ratio"
    assert lowered.workload.load_choices == [0.5, 1.5]


def test_trace_block_default_and_finite_rate_contract() -> None:
    config = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": ["trace.jsonl"],
                    "format": "mooncake",
                },
                "load": {"type": "trace_timestamps"},
            },
            "engine": _engine(),
        }
    )
    assert config.traffic.source.block_size == 512

    weka = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {"type": "trace", "paths": ["weka-corpus"], "format": "weka"},
                "load": {"type": "trace_timestamps", "agentic_lanes": 2},
            },
            "engine": _engine(),
        }
    )
    assert weka.traffic.source.block_size is None
    assert weka.traffic.source.nested_timestamp_basis is None
    assert weka.traffic.load.agentic_lanes == 2

    configured_weka = CorePredictionConfig.model_validate(
        {
            "traffic": {
                "source": {
                    "type": "trace",
                    "paths": ["weka-corpus"],
                    "format": "weka",
                    "nested_timestamp_basis": "relative",
                },
                "load": {"type": "trace_timestamps"},
            },
            "engine": _engine(),
        }
    )
    assert configured_weka.traffic.source.nested_timestamp_basis == "relative"

    with pytest.raises(ValidationError, match="only valid for trace format 'weka'"):
        CorePredictionConfig.model_validate(
            {
                "traffic": {
                    "source": {
                        "type": "trace",
                        "paths": ["trace.jsonl"],
                        "format": "mooncake",
                        "nested_timestamp_basis": "absolute",
                    },
                    "load": {"type": "trace_timestamps"},
                },
                "engine": _engine(),
            }
        )


@pytest.mark.parametrize(
    "traffic",
    [
        {
            "source": {"type": "trace", "paths": ["trace.jsonl"], "format": "mooncake"},
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
        {
            "source": {"type": "trace", "paths": ["weka-corpus"], "format": "weka"},
            "load": {"type": "concurrency", "concurrency": 1},
        },
        {
            "source": {"type": "trace", "paths": ["weka-corpus"], "format": "weka"},
            "load": {"type": "trace_timestamps", "agentic_lanes": 0},
        },
    ],
)
def test_agentic_lane_contract_rejects_unsupported_inputs(traffic: dict) -> None:
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate({"traffic": traffic, "engine": _engine()})


def test_weka_requires_aggregated_engine() -> None:
    with pytest.raises(ValidationError, match="weka requires aggregated"):
        CorePredictionConfig.model_validate(
            {
                "traffic": {
                    "source": {
                        "type": "trace",
                        "paths": ["weka-corpus"],
                        "format": "weka",
                    },
                    "load": {"type": "trace_timestamps"},
                },
                "engine": {
                    **_engine(),
                    "mode": "disaggregated",
                    "workers": {"prefill": {}, "decode": {}},
                },
            }
        )


def test_finite_rate_and_timeout_contract() -> None:
    with pytest.raises(ValidationError):
        CorePredictionConfig.model_validate(
            {
                "traffic": {
                    "source": {"type": "synthetic"},
                    "load": {
                        "type": "constant_rate",
                        "requests_per_second": float("inf"),
                    },
                    "stop": {"requests": 10},
                },
                "engine": _engine(),
            }
        )
    with pytest.raises(ValidationError):
        CoreRecommendationConfig.model_validate(
            {
                "engine": {
                    **_engine(),
                    "mode": "aggregated",
                    "context_length": 4096,
                },
                "optimization": {},
                "optimizer": {"candidate_timeout_seconds": float("inf")},
            }
        )


def test_prediction_timing_forward_model_defaults_to_op_level() -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine()})

    assert config.engine.workers.aggregated is not None
    assert config.engine.workers.aggregated.timing.type == "default"
    assert config.engine.workers.aggregated.timing.forward_model == "op_level"


def test_prediction_timing_accepts_fpm_forward_model_with_default_timing() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"timing": {"type": "default", "forward_model": "fpm"}}

    config = CorePredictionConfig.model_validate({"engine": engine})

    assert config.engine.workers.aggregated is not None
    assert config.engine.workers.aggregated.timing.forward_model == "fpm"


@pytest.mark.parametrize(
    "timing",
    [
        {"type": "fixed", "prefill_ms": 1, "decode_ms": 1, "forward_model": "fpm"},
        {"type": "polynomial", "forward_model": "fpm"},
    ],
)
def test_prediction_timing_rejects_fpm_forward_model_without_default_timing(
    timing: dict,
) -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"timing": timing}

    with pytest.raises(ValidationError, match="forward_model applies to default timing only"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_prediction_timing_rejects_unknown_forward_model() -> None:
    engine = _engine()
    engine["workers"]["aggregated"] = {"timing": {"forward_model": "layerwise"}}

    with pytest.raises(ValidationError, match="forward_model"):
        CorePredictionConfig.model_validate({"engine": engine})


def test_recommendation_timing_accepts_forward_model_per_role() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "disaggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "workers": {
                    "prefill": {"timing": {"type": "default"}},
                    "decode": {"timing": {"type": "default", "forward_model": "fpm"}},
                },
            },
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )

    assert config.engine.workers.prefill is not None
    assert config.engine.workers.decode is not None
    assert config.engine.workers.prefill.timing.forward_model == "op_level"
    assert config.engine.workers.decode.timing.forward_model == "fpm"


def _fpm_recommendation() -> CoreRecommendationConfig:
    return CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "aggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 2048,
                "workers": {
                    "aggregated": {
                        "parallelism": {
                            "preset": False,
                            "tensor": 1,
                            "moe_tensor": 1,
                            "moe_expert": 1,
                        },
                        "kv_cache": {"capacity": {"type": "fixed", "blocks": 256}},
                        "timing": {"type": "default", "forward_model": "fpm"},
                    }
                },
            },
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )


def test_recommendation_lowers_forward_model_per_role() -> None:
    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "mode": "disaggregated",
                "model": "example/model",
                "hardware": "h200_sxm",
                "backend": "vllm",
                "context_length": 4096,
                "workers": {
                    "prefill": {"timing": {"type": "default"}},
                    "decode": {"timing": {"type": "default", "forward_model": "fpm"}},
                },
            },
            "optimization": {"constraints": {"max_candidate_gpus": 8}},
        }
    )

    space = recommendation_to_sweeper(config).search_space

    assert space.prefill_forward_model == "op_level"
    assert space.decode_forward_model == "fpm"
    assert space.agg_forward_model == "op_level"


def test_recommendation_candidate_yaml_round_trips_forward_model() -> None:
    config = _fpm_recommendation()
    smart = recommendation_to_sweeper(config)
    assert smart.search_space.agg_forward_model == "fpm"

    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    assert sample["agg_forward_model"] == "fpm"
    deployment = build_backend_deployment(sample, backend_version="test")
    assert deployment.agg_engine_args["aic_forward_model"] == "fpm"

    prediction = _candidate_prediction(
        config,
        sample,
        ReplaySpec(backend_deployment=deployment, workload={}, goal={}),
        adapter_sections={},
    )

    assert prediction["engine"]["workers"]["aggregated"]["timing"]["estimation_mode"] == "fpm_interpolation"
    assert prediction["engine"]["workers"]["aggregated"]["timing"]["fallback_policy"] == "deny"
    CorePredictionConfig.model_validate(prediction)


@pytest.mark.parametrize("policy", [None, []])
def test_candidate_preserves_default_vs_disabled_transfers_and_pinned_capacity(policy):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.replay import ForwardPassEstimatorSpec
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    config = _fpm_recommendation()
    smart = recommendation_to_sweeper(config)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    resolved = ForwardPassPerfModelConfig(
        model="example/model",
        system="h200_sxm",
        backend="vllm",
        worker_type="aggregated",
        backend_version="test",
        estimation_mode="op_level",
        transfer_policy=policy,
    ).to_dict()
    deployment = build_backend_deployment(
        sample, backend_version="test", forward_pass_estimators={"agg": ForwardPassEstimatorSpec(config=resolved)}
    )
    assert "gpu_memory_utilization" not in deployment.agg_engine_args["timing_model"]["config"]
    prediction = _candidate_prediction(
        config, sample, ReplaySpec(backend_deployment=deployment, workload={}, goal={}), adapter_sections={}
    )
    replay = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction))
    timing = replay.backend_deployment.agg_engine_args["timing_model"]["config"]
    assert timing["transfer_policy"] == policy
    assert "gpu_memory_utilization" not in timing


def test_public_estimator_config_rejects_sol_full():
    from aisimulate.config.engine import TimingConfig
    from aisimulate.sweeper.config import SearchSpace

    for make in (
        lambda: CorePredictionConfig.model_validate({"engine": {**_engine(), "database_mode": "SOL_FULL"}}),
        lambda: SearchSpace(model_name="m", hardware_sku="h200_sxm", database_mode="SOL_FULL"),
        lambda: SearchSpace(
            model_name="m", hardware_sku="h200_sxm", role_estimator_controls={"agg": {"database_mode": "SOL_FULL"}}
        ),
        lambda: TimingConfig(database_mode="sol_full"),
    ):
        with pytest.raises(ValueError, match="database_mode"):
            make()


def test_recommendation_candidate_yaml_spells_out_op_level_like_other_defaults() -> None:
    config = _fpm_recommendation()
    raw = config.model_dump(mode="python", exclude_none=True)
    raw["engine"]["workers"]["aggregated"]["timing"] = {"type": "default"}
    config = CoreRecommendationConfig.model_validate(raw)
    smart = recommendation_to_sweeper(config)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        },
        parallel_config=ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1),
    )
    deployment = build_backend_deployment(sample, backend_version="test")

    prediction = _candidate_prediction(
        config,
        sample,
        ReplaySpec(backend_deployment=deployment, workload={}, goal={}),
        adapter_sections={},
    )

    # Normalization materializes every schema default into the candidate; forward_model is no exception.
    assert prediction["engine"]["workers"]["aggregated"]["timing"]["estimation_mode"] == "op_level"
    assert prediction["engine"]["workers"]["aggregated"]["timing"]["fallback_policy"] == "deny"


def _pd_hardware_config(*, recommend=False, **overrides):
    engine = {
        **_engine(),
        "mode": "disaggregated",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "context_length": 4096,
        "workers": {role: {"hardware": value} for role, value in overrides.items()},
    }
    for role in ("prefill", "decode"):
        engine["workers"].setdefault(role, {})
    raw = {"engine": engine}
    if recommend:
        raw["optimization"] = {"constraints": {"max_candidate_gpus": 4}}
    return raw


@pytest.mark.parametrize("recommend", [False, True])
@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "auto",
        " auto ",
        " gb200 ",
        "gb200 ",
        "\th200_sxm",
        {"choices": ["h200_sxm", "gb200"]},
    ],
)
def test_worker_hardware_requires_concrete_sku(recommend, role, value):
    schema = CoreRecommendationConfig if recommend else CorePredictionConfig
    with pytest.raises(ValidationError, match="hardware"):
        schema.model_validate(_pd_hardware_config(recommend=recommend, **{role: value}))


@pytest.mark.parametrize("recommend", [False, True])
@pytest.mark.parametrize("mode,role", [("aggregated", "aggregated"), ("afd", "decode")])
def test_worker_hardware_rejects_non_pd_modes(recommend, mode, role):
    schema = CoreRecommendationConfig if recommend else CorePredictionConfig
    raw = _pd_hardware_config(recommend=recommend)
    raw["engine"].update(mode=mode, workers={role: {"hardware": "gb200"}})
    with pytest.raises(ValidationError, match="hardware overrides require prefill/decode"):
        schema.model_validate(raw)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, ("h200_sxm", "h200_sxm")),
        ({"decode": "gb200"}, ("h200_sxm", "gb200")),
        ({"prefill": "gb200"}, ("gb200", "h200_sxm")),
        ({"prefill": "gb200", "decode": "gb200"}, ("gb200", "gb200")),
    ],
)
def test_pd_hardware_survives_search_candidate_yaml_and_predict(overrides, expected):
    import yaml

    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.parallel_enum import DisaggParallelConfig

    source = CoreRecommendationConfig.model_validate(_pd_hardware_config(recommend=True, **overrides))
    smart = recommendation_to_sweeper(source)
    assert tuple(smart.search_space.hardware_sku_for(role) for role in ("prefill", "decode")) == expected
    parallel = ReplicaParallelConfig(ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1)
    sample = unroll_sample(
        search_space=smart.search_space,
        selection={
            "deployment_mode": "disagg",
            "backend": "vllm",
            "prefill_max_num_batched_tokens": 8192,
            "prefill_max_num_seqs": 1,
            "decode_max_num_batched_tokens": 8192,
            "decode_max_num_seqs": 256,
        },
        parallel_config=DisaggParallelConfig(prefill=parallel, decode=parallel),
    )
    sample["backend_version"] = "0.24.0"
    deployment = build_backend_deployment(sample, backend_version="0.24.0")
    mapping = _candidate_prediction(
        source,
        sample,
        ReplaySpec(backend_deployment=deployment, workload={}, goal={}),
        adapter_sections={},
    )
    concrete = CorePredictionConfig.model_validate(yaml.safe_load(yaml.safe_dump(mapping)))
    compiled = prediction_to_replay_spec(concrete).backend_deployment
    assert concrete.engine.hardware == "h200_sxm"
    for role, hardware in zip(("prefill", "decode"), expected, strict=True):
        assert ("hardware" in mapping["engine"]["workers"][role]) == (role in overrides)
        assert getattr(compiled, f"{role}_engine_args")["timing_model"]["config"]["system"] == hardware
        assert compiled.performance_model_metadata[role]["config"]["system"] == hardware
        assert getattr(deployment, f"{role}_engine_args")["aic_system"] == hardware


def test_pd_hardware_mixed_modes_and_auto_fallback():
    raw = _pd_hardware_config(recommend=True, decode="gb200")
    raw["engine"]["mode"] = {"choices": ["aggregated", "disaggregated"]}
    raw["engine"]["workers"]["aggregated"] = {}
    raw["engine"]["hardware"] = "auto"
    raw["optimization"]["hardware"] = "h200_sxm"
    space = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)).search_space
    assert space.hardware_sku_for("agg") == "h200_sxm"
    assert space.hardware_sku_for("prefill") == "h200_sxm"
    assert space.hardware_sku_for("decode") == "gb200"


@pytest.mark.parametrize("same_version", [False, True])
def test_pd_predict_requires_shared_implicit_backend_version(monkeypatch, same_version):
    from aisimulate.compiler import prediction_to_replay_spec

    calls = []

    def resolve(hardware, backend, **kwargs):
        calls.append((hardware, backend))
        return "0.24.0" if same_version or hardware == "h200_sxm" else "0.23.0"

    monkeypatch.setattr("aisimulate.compiler.resolve_backend_version", resolve)
    raw = _pd_hardware_config(decode="gb200")
    del raw["engine"]["backend_version"]
    config = CorePredictionConfig.model_validate(raw)
    if same_version:
        deployment = prediction_to_replay_spec(config).backend_deployment
        assert deployment.backend_version == "0.24.0"
        for role in ("prefill", "decode"):
            assert getattr(deployment, f"{role}_engine_args")["timing_model"]["config"]["backend_version"] == "0.24.0"
    else:
        with pytest.raises(ValueError, match="Set engine.backend_version"):
            prediction_to_replay_spec(config)
    assert set(calls) == {("h200_sxm", "vllm"), ("gb200", "vllm")}


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("backend_version", [None, "0.24.0"])
def test_pd_predict_rejects_unknown_worker_hardware_before_runtime(role, backend_version):
    from aisimulate.compiler import prediction_to_replay_spec

    raw = _pd_hardware_config(**{role: "nonexistent_worker_sku"})
    raw["engine"]["backend_version"] = backend_version
    config = CorePredictionConfig.model_validate(raw)
    with pytest.raises(ValueError, match=rf"unknown workers\.{role}\.hardware.*nonexistent_worker_sku"):
        prediction_to_replay_spec(config)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_pd_predict_accepts_worker_hardware_from_configured_system_paths(monkeypatch, tmp_path, role):
    import yaml

    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate_core.sdk import perf_database

    system_paths = perf_database.get_systems_paths()
    spec = perf_database.load_system_spec("gb200")
    (tmp_path / "custom_worker_sku.yaml").write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(perf_database, "get_systems_paths", lambda: [str(tmp_path), *system_paths])
    raw = _pd_hardware_config(**{role: "custom_worker_sku"})
    config = CorePredictionConfig.model_validate(raw)
    deployment = prediction_to_replay_spec(config).backend_deployment
    assert getattr(deployment, f"{role}_engine_args")["timing_model"]["config"]["system"] == "custom_worker_sku"


@pytest.mark.parametrize("override", [False, True])
def test_pd_predict_keeps_legacy_version_defaults_without_hardware_override(monkeypatch, override):
    from aisimulate.compiler import prediction_to_replay_spec

    def resolve(hardware, backend, **kwargs):
        assert hardware == "gb200"
        return "0.24.0"

    monkeypatch.setattr("aisimulate.compiler.resolve_backend_version", resolve)
    raw = _pd_hardware_config(**({"prefill": "gb200", "decode": "gb200"} if override else {}))
    del raw["engine"]["backend_version"]
    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    assert deployment.backend_version == ("0.24.0" if override else "")


@pytest.mark.parametrize("router_hardware", ["h200_sxm", "gb200", None])
def test_pd_predict_checks_effective_prefill_hardware_in_router_hook(router_hardware):
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec

    config = CorePredictionConfig.model_validate(_pd_hardware_config(prefill="gb200", decode="gb200"))
    spec = AdapterReplaySpec(
        runtime_hooks=(
            RuntimeHookSpec(
                provider="dynamo.router",
                kind="placement_policy",
                api_version=1,
                config={
                    "router_config": {"router_prefill_load_model": "aic"},
                    "aic_perf_config": {"aic_system": router_hardware},
                },
            ),
        ),
    )
    if router_hardware == "gb200":
        prediction_to_replay_spec(config, adapter_specs={"dynamo.router": spec})
    else:
        with pytest.raises(ValueError, match="does not match effective prefill_hardware_sku"):
            prediction_to_replay_spec(config, adapter_specs={"dynamo.router": spec})


def test_new_default_selection_survives_serialization_without_becoming_legacy_op_level():
    config = CorePredictionConfig.model_validate({"engine": _engine()})
    serialized = config.model_dump(mode="json", exclude_none=True)
    timing = serialized["engine"]["workers"]["aggregated"]["timing"]
    assert "forward_model" not in timing
    reloaded = CorePredictionConfig.model_validate(serialized)
    assert reloaded.engine.estimation_mode == "auto"
    assert reloaded.engine.workers.aggregated.timing.estimation_mode is None


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_role_systems_roots_reach_prediction_and_search_preflight(tmp_path, mode):
    from copy import deepcopy
    from importlib.resources import files
    from pathlib import Path

    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.search_space import enumerate_branches

    packaged = Path(str(files("aisimulate_core") / "systems"))
    roles = ("aggregated",) if mode == "aggregated" else ("prefill", "decode")
    workers = {}
    for role in roles:
        root = tmp_path / role
        root.mkdir()
        for entry in packaged.iterdir():
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        (root / "role_only_gpu.yaml").write_text((packaged / "h200_sxm.yaml").read_text())
        workers[role] = {
            "parallelism": {"tensor": 2},
            "kv_cache": {"capacity": {"type": "fixed", "blocks": 256}},
            "timing": {"systems_paths": [str(root)]},
        }
        if mode == "disaggregated":
            workers[role]["hardware"] = "role_only_gpu"
    engine = {
        "model": "Qwen/Qwen3-32B",
        "hardware": "role_only_gpu" if mode == "aggregated" else "h200_sxm",
        "backend": "vllm",
        "mode": mode,
        "context_length": 4096,
        "workers": workers,
    }
    prediction = CorePredictionConfig.model_validate({"engine": engine})
    deployment = prediction_to_replay_spec(prediction).backend_deployment
    for role in roles:
        args = getattr(deployment, f"{'agg' if role == 'aggregated' else role}_engine_args")
        assert args["timing_model"]["config"]["systems_paths"] == [str(tmp_path / role)]
        assert args["timing_model"]["config"]["system"] == "role_only_gpu"
    recommendation_engine = deepcopy(engine)
    for worker in recommendation_engine["workers"].values():
        worker.pop("parallelism")
        worker["kv_cache"] = {"capacity": {"memory_fraction": 0.9}}
    recommendation = CoreRecommendationConfig.model_validate(
        {
            "engine": recommendation_engine,
            "optimization": {
                "target": "throughput",
                "constraints": {"max_candidate_gpus": 4},
            },
        }
    )
    smart = recommendation_to_sweeper(recommendation)
    (branch,) = enumerate_branches(smart, max_seq_len=4096)
    assert branch.parallel_configs
    assert branch.deployment_mode == ("agg" if mode == "aggregated" else "disagg")


_KV_DTYPE_MODEL = "Qwen/Qwen3-32B-FP8"  # fp8_block weights, no kv_cache_quant_algo: FP8 KV is inferred unless pinned
_KV_DTYPE_TRANSFER = {"bytes_per_token": "auto", "bandwidth_gb_per_second": 400.0}
_KV_DTYPE_MODES = ("aggregated", "disaggregated")
_KV_DTYPE_ROLES = {"aggregated": ("agg",), "disaggregated": ("prefill", "decode")}
# (dtype, canonical kvcache_quant_mode, canonical fmha_quant_mode) for the concrete pins.
_KV_DTYPE_PINS = [("bfloat16", "bfloat16", "bfloat16"), ("fp8", "fp8", None)]
_KV_DTYPE_ASSUMED = "assuming an FP8 KV cache"


def _worker_name(role: str) -> str:
    return "aggregated" if role == "agg" else role


def _kv_dtype_engine(mode: str, dtype, *, recommend: bool, kv_cache_extra: dict | None, worker_overrides: dict) -> dict:
    parallelism = (
        {"preset": False, "tensor": 1, "attention_data": 1, "moe_tensor": 1, "moe_expert": 1}
        if recommend
        else {"tensor": 2}
    )
    worker = {"parallelism": parallelism, "kv_cache": {"dtype": dtype, **(kv_cache_extra or {})}, **worker_overrides}
    engine = {
        "model": _KV_DTYPE_MODEL,
        "hardware": "h200_sxm",
        "backend": "vllm",
        "mode": mode,
        "context_length": 4096,
        "workers": {_worker_name(role): deepcopy(worker) for role in _KV_DTYPE_ROLES[mode]},
    }
    if mode == "disaggregated":
        engine["kv_transfer"] = deepcopy(_KV_DTYPE_TRANSFER)
    return engine


def _kv_dtype_prediction(
    dtype, mode: str = "aggregated", kv_cache_extra: dict | None = None, **worker_overrides
) -> dict:
    return {
        "engine": _kv_dtype_engine(
            mode, dtype, recommend=False, kv_cache_extra=kv_cache_extra, worker_overrides=worker_overrides
        )
    }


def _kv_dtype_recommendation(
    dtype, mode: str = "aggregated", kv_cache_extra: dict | None = None, **worker_overrides
) -> dict:
    return {
        "engine": _kv_dtype_engine(
            mode, dtype, recommend=True, kv_cache_extra=kv_cache_extra, worker_overrides=worker_overrides
        ),
        "optimization": {"constraints": {"max_candidate_gpus": 8}},
    }


_KV_DTYPE_SCHEMAS = [(CorePredictionConfig, _kv_dtype_prediction), (CoreRecommendationConfig, _kv_dtype_recommendation)]


def _kv_dtype_sample(search_space, mode: str) -> dict:
    shape = ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1)
    if mode == "aggregated":
        selection = {
            "deployment_mode": "agg",
            "backend": "vllm",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        }
        parallel = ReplicaParallelConfig(shape, replicas=1)
    else:
        selection = {
            "deployment_mode": "disagg",
            "backend": "vllm",
            "prefill_max_num_batched_tokens": 8192,
            "prefill_max_num_seqs": 4,
            "decode_max_num_batched_tokens": 8192,
            "decode_max_num_seqs": 256,
        }
        parallel = DisaggParallelConfig(
            prefill=ReplicaParallelConfig(shape, replicas=1), decode=ReplicaParallelConfig(shape, replicas=1)
        )
    return unroll_sample(search_space=search_space, selection=selection, parallel_config=parallel)


def _kv_dtype_estimators(search_space, sample: dict, roles: tuple[str, ...]) -> dict:
    """Resolved canonical estimators carrying exactly the modes the resolver requested per role."""
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver
    from aisimulate.sweeper.replay import ForwardPassEstimatorSpec
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    resolver = ForwardPassEstimatorResolver(search_space)
    estimators = {}
    for role in roles:
        request = resolver._request(sample, role)
        estimators[role] = ForwardPassEstimatorSpec(
            config=ForwardPassPerfModelConfig(
                model=_KV_DTYPE_MODEL,
                system="h200_sxm",
                backend="vllm",
                worker_type=_worker_name(role),
                backend_version="test",
                estimation_mode="op_level",
                kvcache_quant_mode=request.kvcache_quant_mode,
                fmha_quant_mode=request.fmha_quant_mode,
            ).to_dict()
        )
    return estimators


def _compiled_engine_args(raw: dict) -> dict[str, dict]:
    from aisimulate.compiler import prediction_to_replay_spec

    deployment = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw)).backend_deployment
    return {role: getattr(deployment, f"{role}_engine_args") for role in _KV_DTYPE_ROLES[raw["engine"]["mode"]]}


@pytest.mark.parametrize("mode", _KV_DTYPE_MODES)
@pytest.mark.parametrize("dtype", ["auto", "bfloat16", "fp8"])
def test_kv_cache_dtype_accepts_supported_values_in_predict_and_recommend(mode: str, dtype: str) -> None:
    for schema, build in _KV_DTYPE_SCHEMAS:
        config = schema.model_validate(build(dtype, mode))
        for role in _KV_DTYPE_ROLES[mode]:
            assert getattr(config.engine.workers, _worker_name(role)).kv_cache.dtype == dtype


def test_kv_cache_dtype_defaults_to_auto() -> None:
    config = CorePredictionConfig.model_validate({"engine": _engine()})
    assert config.engine.workers.aggregated.kv_cache.dtype == "auto"


@pytest.mark.parametrize("value", ["int8", "FP8", {"choices": ["fp8", "bfloat16"]}, True])
def test_kv_cache_dtype_rejects_unsupported_values(value) -> None:
    with pytest.raises(ValidationError, match="kv_cache.dtype"):
        CorePredictionConfig.model_validate(_kv_dtype_prediction(value))
    with pytest.raises(ValidationError, match="kv_cache.dtype"):
        CoreRecommendationConfig.model_validate(_kv_dtype_recommendation(value))


@pytest.mark.parametrize(("schema", "build"), _KV_DTYPE_SCHEMAS)
@pytest.mark.parametrize("timing", [{"type": "fixed", "prefill_ms": 1, "decode_ms": 1}, {"type": "polynomial"}])
def test_kv_cache_dtype_requires_default_timing(schema, build, timing: dict) -> None:
    with pytest.raises(ValidationError, match="kv_cache.dtype requires default timing"):
        schema.model_validate(build("bfloat16", timing=timing))


def test_kv_cache_dtype_rejected_for_afd_and_encoder_pools() -> None:
    # AFD decode combined with a regular prefill worker: the only AFD layout that carries a language worker.
    afd = {
        "engine": {
            "mode": "afd",
            "model": "Qwen/Qwen3-32B",
            "hardware": "h200_sxm",
            "backend": "trtllm",
            "afd": {
                "phase": "decode",
                "combined_with_pd": True,
                "n_a_nodes": 1,
                "n_f_nodes": 1,
                "tp_a": 8,
                "a_batch_size": 8,
            },
            "workers": {"prefill": {"kv_cache": {"dtype": "bfloat16"}}},
        }
    }
    with pytest.raises(ValidationError, match="kv_cache.dtype requires default timing"):
        CorePredictionConfig.model_validate(afd)
    afd["engine"]["workers"]["prefill"]["kv_cache"]["dtype"] = "auto"
    CorePredictionConfig.model_validate(afd)

    encoder = _kv_dtype_prediction("bfloat16")
    encoder["engine"]["workers"]["encoder"] = {"tensor": 1, "replicas": 1, "batch_size": 1}
    with pytest.raises(ValidationError, match="kv_cache.dtype requires default timing"):
        CorePredictionConfig.model_validate(encoder)


@pytest.mark.parametrize(("schema", "build"), _KV_DTYPE_SCHEMAS)
@pytest.mark.parametrize("field", ["kvcache_quant_mode", "fmha_quant_mode"])
def test_kv_cache_dtype_rejects_engine_wide_quant_mode_conflict(schema, build, field: str) -> None:
    raw = build("bfloat16")
    raw["engine"][field] = "fp8"
    with pytest.raises(ValidationError, match="kv_cache.dtype conflicts"):
        schema.model_validate(raw)
    # ``auto`` is not a pin: the engine-wide spelling stays usable on its own.
    raw = build("auto")
    raw["engine"][field] = "fp8"
    schema.model_validate(raw)


@pytest.mark.parametrize(("schema", "build"), _KV_DTYPE_SCHEMAS)
def test_kv_cache_dtype_must_match_across_prefill_and_decode(schema, build) -> None:
    raw = build("bfloat16", "disaggregated")
    raw["engine"]["workers"]["decode"]["kv_cache"]["dtype"] = "fp8"
    with pytest.raises(ValidationError, match="must match across prefill and decode"):
        schema.model_validate(raw)
    raw["engine"]["workers"]["decode"]["kv_cache"]["dtype"] = "auto"
    workers = schema.model_validate(raw).engine.workers
    assert (workers.prefill.kv_cache.dtype, workers.decode.kv_cache.dtype) == ("bfloat16", "auto")


def test_search_space_kv_cache_dtype_validators() -> None:
    base = {"model_name": "example/model", "hardware_sku": "h200_sxm", "gpu_budget": 8}
    with pytest.raises(ValidationError, match="kv_cache_dtype conflicts"):
        SearchSpace(**base, agg_kv_cache_dtype="bfloat16", kvcache_quant_mode="fp8")
    with pytest.raises(ValidationError, match="must match"):
        SearchSpace(**base, prefill_kv_cache_dtype="bfloat16", decode_kv_cache_dtype="fp8")
    with pytest.raises(ValidationError, match="require default timing"):
        SearchSpace(
            **base,
            decode_kv_cache_dtype="bfloat16",
            agg_timing_model={"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
        )
    assert (
        SearchSpace(**base, prefill_kv_cache_dtype="fp8", decode_kv_cache_dtype="auto").prefill_kv_cache_dtype == "fp8"
    )


@pytest.mark.parametrize("mode", _KV_DTYPE_MODES)
@pytest.mark.parametrize(("dtype", "expected_kv", "expected_fmha"), [("auto", None, None), *_KV_DTYPE_PINS])
def test_kv_cache_dtype_lowers_to_canonical_timing_config(mode, dtype, expected_kv, expected_fmha) -> None:
    from aisimulate.compiler import prediction_to_replay_spec

    config = CorePredictionConfig.model_validate(_kv_dtype_prediction(dtype, mode))
    deployment = prediction_to_replay_spec(config).backend_deployment
    for role in _KV_DTYPE_ROLES[mode]:
        args = getattr(deployment, f"{role}_engine_args")
        timing = args["timing_model"]["config"]
        assert (timing["kvcache_quant_mode"], timing["fmha_quant_mode"]) == (expected_kv, expected_fmha)
        # The worker's perf-model identity must agree with the timing config it was lowered into.
        metadata = deployment.performance_model_metadata[_worker_name(role)]["config"]
        assert (metadata.get("kvcache_quant_mode"), metadata.get("fmha_quant_mode")) == (expected_kv, expected_fmha)
        # The canonical estimator config is the only carrier; no flat rank alias is emitted.
        assert not {"kv_cache_dtype", "fmha_dtype", "aic_kv_cache_dtype", "aic_fmha_dtype"} & set(args)


def test_kv_cache_dtype_steers_predict_kv_geometry() -> None:
    """Host-offload and P/D-transfer bytes per token follow the pinned KV dtype (FP8 halves BF16)."""
    offload = {"host_offload": {"num_host_blocks": 1024}}
    agg = {
        dtype: _compiled_engine_args(_kv_dtype_prediction(dtype, kv_cache_extra=offload))["agg"][
            "kv_cache_bytes_per_token"
        ]
        for dtype in ("bfloat16", "fp8")
    }
    assert agg["bfloat16"] > 0
    assert agg["fp8"] * 2 == agg["bfloat16"]

    transfer = {
        dtype: {
            role: args["kv_transfer_bytes_per_token"]
            for role, args in _compiled_engine_args(_kv_dtype_prediction(dtype, "disaggregated")).items()
        }
        for dtype in ("bfloat16", "fp8")
    }
    assert transfer["bfloat16"]["prefill"] == transfer["bfloat16"]["decode"] > 0
    assert transfer["fp8"]["prefill"] == transfer["fp8"]["decode"]
    assert transfer["fp8"]["prefill"] * 2 == transfer["bfloat16"]["prefill"]


def test_kv_cache_dtype_steers_sweeper_kv_geometry() -> None:
    """build_backend_deployment sizes host offload and P/D transfer from the role-pinned dtype."""

    def deployment_for(dtype: str, mode: str, kv_cache_extra: dict | None = None):
        config = CoreRecommendationConfig.model_validate(_kv_dtype_recommendation(dtype, mode, kv_cache_extra))
        search_space = recommendation_to_sweeper(config).search_space
        sample = _kv_dtype_sample(search_space, mode)
        return build_backend_deployment(
            sample,
            backend_version="test",
            forward_pass_estimators=_kv_dtype_estimators(search_space, sample, _KV_DTYPE_ROLES[mode]),
        )

    offload = {"host_offload": {"num_host_blocks": 1024}}
    agg = {
        dtype: deployment_for(dtype, "aggregated", offload).agg_engine_args["kv_cache_bytes_per_token"]
        for dtype in ("bfloat16", "fp8")
    }
    assert agg["bfloat16"] > 0
    assert agg["fp8"] * 2 == agg["bfloat16"]

    transfer = {}
    for dtype in ("bfloat16", "fp8"):
        deployment = deployment_for(dtype, "disaggregated")
        prefill = deployment.prefill_engine_args["kv_transfer_bytes_per_token"]
        assert deployment.decode_engine_args["kv_transfer_bytes_per_token"] == prefill
        transfer[dtype] = prefill
    assert transfer["bfloat16"] > 0
    assert transfer["fp8"] * 2 == transfer["bfloat16"]


def test_kv_cache_dtype_requires_resolved_estimator_in_sweeper_payload() -> None:
    smart = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(_kv_dtype_recommendation("bfloat16")))
    sample = _kv_dtype_sample(smart.search_space, "aggregated")
    with pytest.raises(ValueError, match="resolved canonical forward-pass estimator"):
        build_backend_deployment(sample, backend_version="test")


@pytest.mark.parametrize("mode", _KV_DTYPE_MODES)
@pytest.mark.parametrize(("dtype", "expected_kv", "expected_fmha"), _KV_DTYPE_PINS)
def test_kv_cache_dtype_round_trips_through_recommendation_candidate(mode, dtype, expected_kv, expected_fmha) -> None:
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolver

    roles = _KV_DTYPE_ROLES[mode]
    config = CoreRecommendationConfig.model_validate(_kv_dtype_recommendation(dtype, mode))
    smart = recommendation_to_sweeper(config)
    assert smart.search_space.kvcache_quant_mode is None
    assert smart.search_space.fmha_quant_mode is None
    for role in roles:
        assert getattr(smart.search_space, f"{role}_kv_cache_dtype") == dtype

    sample = _kv_dtype_sample(smart.search_space, mode)
    resolver = ForwardPassEstimatorResolver(smart.search_space)
    for role in roles:
        assert sample[f"{role}_kv_cache_dtype"] == dtype
        request = resolver._request(sample, role)
        assert (request.kvcache_quant_mode, request.fmha_quant_mode) == (expected_kv, expected_fmha)

    deployment = build_backend_deployment(
        sample, backend_version="test", forward_pass_estimators=_kv_dtype_estimators(smart.search_space, sample, roles)
    )
    prediction = _candidate_prediction(
        config, sample, ReplaySpec(backend_deployment=deployment, workload={}, goal={}), adapter_sections={}
    )
    assert "kvcache_quant_mode" not in prediction["engine"]
    assert "fmha_quant_mode" not in prediction["engine"]
    replay = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction)).backend_deployment
    for role in roles:
        assert prediction["engine"]["workers"][_worker_name(role)]["kv_cache"]["dtype"] == dtype
        assert getattr(deployment, f"{role}_engine_args")["timing_model"]["config"]["kvcache_quant_mode"] == expected_kv
        timing = getattr(replay, f"{role}_engine_args")["timing_model"]["config"]
        assert (timing["kvcache_quant_mode"], timing["fmha_quant_mode"]) == (expected_kv, expected_fmha)


@pytest.mark.parametrize("mode", _KV_DTYPE_MODES)
def test_kv_cache_dtype_pin_reaches_parallel_feasibility_prefilter(mode: str, monkeypatch, caplog) -> None:
    """The per-role pin sizes the KV pre-filter like the engine-wide spelling and silences the inferred-FP8 warning."""
    from aisimulate.sweeper.search_space import _engine_memory_kwargs, enumerate_branches
    from aisimulate_core.sdk.models import helpers as model_helpers

    roles = _KV_DTYPE_ROLES[mode]
    pinned = {"kvcache_quant_mode": "bfloat16", "fmha_quant_mode": "bfloat16"}
    per_role = recommendation_to_sweeper(
        CoreRecommendationConfig.model_validate(_kv_dtype_recommendation("bfloat16", mode))
    )
    engine_wide_raw = _kv_dtype_recommendation("auto", mode)
    engine_wide_raw["engine"].update(pinned)
    engine_wide = recommendation_to_sweeper(CoreRecommendationConfig.model_validate(engine_wide_raw))

    role_map = dict(zip(roles, roles, strict=True))
    assert _engine_memory_kwargs(per_role.search_space, role_map) == {
        "role_model_controls": dict.fromkeys(roles, pinned)
    }
    assert _engine_memory_kwargs(engine_wide.search_space, role_map) == {"model_controls": pinned}

    monkeypatch.setattr(model_helpers, "_INFERRED_FP8_KV_WARNED", set())
    with caplog.at_level(logging.WARNING, logger="aisimulate_core.sdk.models.helpers"):
        (per_role_branch,) = enumerate_branches(per_role, max_seq_len=4096)
    assert _KV_DTYPE_ASSUMED not in caplog.text
    (engine_wide_branch,) = enumerate_branches(engine_wide, max_seq_len=4096)
    assert per_role_branch.parallel_configs
    assert per_role_branch.parallel_configs == engine_wide_branch.parallel_configs
