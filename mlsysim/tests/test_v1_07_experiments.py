import json
import math

import pytest

from mlsysim.engine.v1_07_experiments import (
    OPERATION_GRAPH,
    MODEL_KEY,
    TRACKS,
    compare_dispatch,
    evaluate_compilation,
    evaluate_dispatch,
    evaluate_fusion,
    evaluate_operator_support,
    evaluate_recomputation,
    replay,
)


def test_every_track_dispatch_result_has_fixed_work_and_positive_timing():
    for track_id in TRACKS:
        coarse = evaluate_dispatch(track_id, 1)
        fine = evaluate_dispatch(track_id, len(OPERATION_GRAPH))

        assert coarse["flops"] == fine["flops"]
        assert coarse["payload_mb"] == fine["payload_mb"]
        assert fine["dispatch_time_us"] > coarse["dispatch_time_us"]
        assert fine["total_time_us"] > coarse["total_time_us"] > 0


def test_dispatch_comparison_uses_identical_baseline_work():
    comparison = compare_dispatch("cloud", 6, 1)

    assert comparison["baseline"]["flops"] == comparison["intervention"]["flops"]
    assert comparison["baseline"]["payload_mb"] == comparison["intervention"]["payload_mb"]
    assert comparison["speedup"] > 1
    assert comparison["saved_time_us"] > 0


def test_compilation_repayment_depends_on_repetitions_and_recompilations():
    short = evaluate_compilation("edge", 1)
    long = evaluate_compilation("edge", 100)
    unstable = evaluate_compilation("edge", 100, recompilations=5)

    assert not short["compilation_repaid"]
    assert long["compilation_repaid"]
    assert unstable["compiled_total_ms"] > long["compiled_total_ms"]
    assert unstable["eager_total_ms"] == long["eager_total_ms"]
    assert unstable["break_even_executions"] > long["break_even_executions"]


def test_fusion_eliminates_launches_and_intermediate_traffic():
    unfused = evaluate_fusion("mobile", graph_breaks=len(OPERATION_GRAPH) - 1)
    fused = evaluate_fusion("mobile", graph_breaks=0)
    broken = evaluate_fusion("mobile", graph_breaks=2)

    assert unfused["launches"] == len(OPERATION_GRAPH)
    assert unfused["total_traffic_mb"] == pytest.approx(unfused["unfused_traffic_mb"])
    assert fused["launches"] == 1
    assert fused["total_traffic_mb"] < broken["total_traffic_mb"] < unfused["total_traffic_mb"]
    assert fused["total_time_us"] < unfused["total_time_us"]


def test_unsupported_operator_has_explicit_fallback_copies_and_cost():
    native = evaluate_fusion("mobile", unsupported_operator="relu")
    fallback = evaluate_fusion("mobile", unsupported_operator="dynamic_slice")

    assert native["fallback"]["status"] == "native"
    assert native["fallback_copy_mb"] == 0
    assert fallback["fallback"]["status"] == "fallback"
    assert fallback["fallback_copy_mb"] > 0
    assert fallback["boundary_count"] == native["boundary_count"] + 2
    assert fallback["launches"] == native["launches"] + 2
    assert fallback["total_time_us"] > native["total_time_us"]


def test_recomputation_reduces_storage_but_adds_operations_and_step_time():
    retained = evaluate_recomputation("cloud", "retain_all")
    alternate = evaluate_recomputation("cloud", "alternate")
    minimal = evaluate_recomputation("cloud", "input_only")

    assert retained["retained_activation_mb"] > alternate["retained_activation_mb"] > minimal["retained_activation_mb"]
    assert retained["repeated_flops"] < alternate["repeated_flops"] < minimal["repeated_flops"]
    assert retained["step_time_ms"] < alternate["step_time_ms"] < minimal["step_time_ms"]


def test_tiny_mobile_and_edge_training_use_development_hosts():
    tiny = evaluate_recomputation("tinyml", "retain_all")
    mobile = evaluate_recomputation("mobile", "retain_all")
    edge = evaluate_recomputation("edge", "retain_all")

    assert tiny["training_host"] == TRACKS["tinyml"].development_host.name
    assert mobile["training_host"] == TRACKS["mobile"].development_host.name
    assert edge["training_host"] == TRACKS["edge"].development_host.name
    assert tiny["training_host"] != TRACKS["tinyml"].target.name
    assert mobile["training_host"] != TRACKS["mobile"].target.name
    assert edge["training_host"] != TRACKS["edge"].target.name




def test_all_tracks_have_one_native_case_and_a_failed_or_fallback_case():
    expected_non_native = {
        "tinyml": "unsupported",
        "mobile": "fallback",
        "edge": "fallback",
        "cloud": "fallback",
    }
    for track_id in TRACKS:
        native = evaluate_operator_support(track_id, "relu", "static")
        adverse = evaluate_operator_support(track_id, "host_callback", "static")

        assert native["status"] == "native", track_id
        assert adverse["status"] == expected_non_native[track_id], track_id
        assert not adverse["operator_supported"]
        assert adverse["shape_supported"]


def test_operator_identity_is_noncausal_when_both_paths_are_supported_equally():
    relu = evaluate_operator_support("cloud", "relu", "static", tensor_elements=1_000)
    dynamic_slice = evaluate_operator_support("cloud", "dynamic_slice", "static", tensor_elements=1_000)

    assert relu["status"] == dynamic_slice["status"] == "native"
    assert relu["total_time_us"] == pytest.approx(dynamic_slice["total_time_us"])


def test_large_batch_can_fail_activation_memory_and_recomputation_can_restore_fit():
    retained = evaluate_recomputation("tinyml", "retain_all", tensor_elements=2_000_000_000, batch_size=4)
    minimal = evaluate_recomputation("tinyml", "input_only", tensor_elements=2_000_000_000, batch_size=4)

    assert not retained["memory_feasible"]
    assert minimal["memory_feasible"]
    assert minimal["step_time_ms"] > retained["step_time_ms"]


@pytest.mark.parametrize(
    "call,args,kwargs",
    [
        (evaluate_dispatch, ("unknown", 1), {}),
        (evaluate_dispatch, ("edge", 0), {}),
        (evaluate_compilation, ("edge", 0), {}),
        (evaluate_compilation, ("edge", 10), {"recompilations": -1}),
        (evaluate_compilation, ("edge", 5), {"recompilations": 5}),
        (evaluate_fusion, ("edge",), {"graph_breaks": len(OPERATION_GRAPH)}),
        (evaluate_operator_support, ("edge", "magic", "static"), {}),
        (evaluate_operator_support, ("edge", "relu", "ragged"), {}),
        (evaluate_recomputation, ("edge", "magic"), {}),
        (evaluate_recomputation, ("edge", "retain_all"), {"batch_size": 0}),
    ],
)
def test_invalid_inputs_fail_explicitly(call, args, kwargs):
    with pytest.raises(ValueError):
        call(*args, **kwargs)


def test_results_are_finite_for_all_default_track_endpoints():
    for track_id in TRACKS:
        results = (
            evaluate_dispatch(track_id, 6),
            evaluate_compilation(track_id, 50),
            evaluate_fusion(track_id),
            evaluate_operator_support(track_id, "relu", "static"),
            evaluate_recomputation(track_id, "alternate"),
        )
        for result in results:
            for value in result.values():
                if isinstance(value, float):
                    assert math.isfinite(value), (track_id, result)


def test_every_experiment_result_carries_complete_replay_inputs():
    results = (
        evaluate_dispatch("mobile", 3, tensor_elements=2_000),
        evaluate_compilation("mobile", 20, recompilations=2, tensor_elements=2_000),
        evaluate_fusion(
            "mobile",
            graph_breaks=1,
            unsupported_operator="host_callback",
            shape_mode="bounded",
            tensor_elements=2_000,
        ),
        evaluate_operator_support("mobile", "host_callback", "bounded", tensor_elements=2_000),
        evaluate_recomputation("mobile", "alternate", tensor_elements=2_000, batch_size=4),
    )

    for result in results:
        assert result["model_key"] == "v1_07_experiments"
        assert result["inputs"]["track_id"] == "mobile"
        assert result["inputs"]["tensor_elements"] == 2_000


@pytest.mark.parametrize("track_id", TRACKS)
def test_actual_notebook_evidence_arms_survive_json_roundtrip_and_replay(track_id):
    dispatch = compare_dispatch(track_id, len(OPERATION_GRAPH), 1)
    arms = (
        dispatch["baseline"],
        dispatch["intervention"],
        evaluate_compilation(track_id, 1, recompilations=0),
        evaluate_compilation(track_id, 50, recompilations=0),
        evaluate_fusion(track_id, graph_breaks=len(OPERATION_GRAPH) - 1),
        evaluate_fusion(track_id, graph_breaks=0),
        evaluate_recomputation(track_id, "retain_all", batch_size=8),
        evaluate_recomputation(track_id, "alternate", batch_size=8),
        evaluate_operator_support(track_id, "relu", "static"),
        evaluate_operator_support(track_id, "host_callback", "static"),
        # Part E's hold decision records this native baseline as chosen_result.
        evaluate_operator_support(track_id, "relu", "static"),
    )

    restored_arms = json.loads(json.dumps(arms))
    for original, restored in zip(arms, restored_arms, strict=True):
        assert replay(restored["model_key"], restored["inputs"]) == original

    if track_id == "tinyml":
        # This is the actual no-feasible-path capture produced by Part E.
        assert restored_arms[9]["status"] == "unsupported"
        assert restored_arms[9]["total_time_us"] is None


def test_replay_dispatch_is_explicit_and_rejects_unknown_schemas():
    comparison = compare_dispatch("mobile", len(OPERATION_GRAPH), 1)
    assert replay(MODEL_KEY, comparison["inputs"]) == comparison

    with pytest.raises(ValueError, match="unknown Chapter 7 model_key"):
        replay("v1_07_inferred", comparison["inputs"])
    with pytest.raises(ValueError, match="inputs must be a mapping"):
        replay(MODEL_KEY, [])
    with pytest.raises(ValueError, match="known Chapter 7 evaluator schema"):
        replay(MODEL_KEY, {"track_id": "mobile", "dispatches": 1})
