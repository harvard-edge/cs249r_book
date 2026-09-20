import json
import math

import pytest

from mlsysim.core.units import Q_, ureg
from mlsysim.engine.v1_05_experiments import (
    EVALUATE_MODEL_KEY,
    NUMERICAL_MODEL_KEY,
    TRACKS,
    analyze_network,
    compare_track,
    default_batch,
    evaluate_track,
    growth_comparison,
    repair_comparison,
    replay,
    run_numerical_case,
    snapshot,
)


def _bytes(quantity) -> float:
    return quantity.to(ureg.byte).magnitude


def test_dense_network_counts_every_layer_and_declares_mac_conversion():
    result = analyze_network((2, 3, 1), batch_size=4, precision="fp16")

    assert result["weight_parameters"] == 9
    assert result["bias_parameters"] == 4
    assert result["parameters"] == 13
    assert result["batch_macs"] == 36
    assert result["flops_per_mac"] == 2.0
    assert result["bytes_per_element"] == 2
    assert result["optimizer_bytes_per_element"] == 4
    assert result["mac_convention_flops"] == 72
    assert result["bias_additions"] == 16
    assert result["nonlinearity_evaluations"] == 16
    assert [layer["parameters"] for layer in result["layers"]] == [9, 4]
    assert [_bytes(layer["minimum_forward_traffic"]) for layer in result["layers"]] == [58, 40]
    assert _bytes(result["minimum_forward_traffic"]) == 98


def test_inference_and_training_activation_schedules_and_state_are_exact():
    inference = analyze_network((2, 3, 1), batch_size=4, precision="fp16")
    training = analyze_network(
        (2, 3, 1), batch_size=4, precision="fp16", phase="training", optimizer="adam"
    )

    assert inference["inference_live_activation_elements"] == 20
    assert inference["activation_elements"] == 20
    assert _bytes(inference["weight_memory"]) == 26
    assert _bytes(inference["activation_memory"]) == 40
    assert _bytes(inference["total_state_memory"]) == 66

    assert training["retained_training_activation_elements"] == 24
    assert training["activation_elements"] == 24
    assert _bytes(training["gradient_memory"]) == 26
    assert training["optimizer_slots"] == 2
    assert _bytes(training["optimizer_memory"]) == 104
    assert _bytes(training["total_state_memory"]) == 204


def test_batch_changes_total_work_and_activations_but_not_parameters():
    one = analyze_network((8, 4, 2), batch_size=1)
    eight = analyze_network((8, 4, 2), batch_size=8)

    assert eight["parameters"] == one["parameters"]
    assert eight["weight_memory"] == one["weight_memory"]
    assert eight["batch_macs"] == 8 * one["batch_macs"]
    assert eight["activation_elements"] == 8 * one["activation_elements"]
    assert eight["minimum_forward_traffic"] > one["minimum_forward_traffic"]
    assert eight["minimum_forward_traffic_per_sample"] < one["minimum_forward_traffic_per_sample"]


def test_format_changes_storage_and_traffic_without_changing_tensor_counts():
    fp32 = analyze_network((16, 8, 2), batch_size=3, precision="fp32")
    fp16 = analyze_network((16, 8, 2), batch_size=3, precision="fp16")
    int8 = analyze_network((16, 8, 2), batch_size=3, precision="int8")

    for key in ("parameters", "batch_macs", "activation_elements"):
        assert fp32[key] == fp16[key] == int8[key]
    assert _bytes(fp32["total_state_memory"]) == 2 * _bytes(fp16["total_state_memory"])
    assert _bytes(fp16["total_state_memory"]) == 2 * _bytes(int8["total_state_memory"])
    assert _bytes(fp32["minimum_forward_traffic"]) == 4 * _bytes(int8["minimum_forward_traffic"])


def test_optimizer_is_noncausal_for_inference_and_only_changes_training_state():
    inference_none = analyze_network((8, 4, 2), optimizer="none")
    inference_adam = analyze_network((8, 4, 2), optimizer="adam")
    training_sgd = analyze_network((8, 4, 2), phase="training", optimizer="sgd")
    training_adam = analyze_network((8, 4, 2), phase="training", optimizer="adam")

    for key in (
        "parameters",
        "batch_macs",
        "activation_memory",
        "total_state_memory",
        "minimum_forward_traffic",
    ):
        assert inference_none[key] == inference_adam[key]
    assert training_sgd["batch_macs"] == training_adam["batch_macs"]
    assert training_sgd["gradient_memory"] == training_adam["gradient_memory"]
    assert training_sgd["optimizer_memory"] < training_adam["optimizer_memory"]
    assert training_sgd["total_state_memory"] < training_adam["total_state_memory"]


def test_every_track_has_feasible_inference_and_failed_adam_training():
    for track_id in TRACKS:
        inference = evaluate_track(track_id)
        training = evaluate_track(track_id, phase="training", optimizer="adam")

        assert inference["fits_memory"], track_id
        assert inference["violations"] == []
        assert not training["fits_memory"], track_id
        assert training["violations"] == ["memory"]
        assert training["total_state_memory"] > inference["total_state_memory"]


def test_batching_trades_higher_batch_latency_and_state_for_amortized_throughput():
    for track_id in TRACKS:
        one = evaluate_track(track_id, batch_size=1)
        eight = evaluate_track(track_id, batch_size=8)

        assert eight["batch_latency"] > one["batch_latency"], track_id
        assert eight["activation_memory"] > one["activation_memory"], track_id
        assert eight["throughput"] > one["throughput"], track_id
        assert eight["amortized_service_time"] < one["amortized_service_time"], track_id
        assert (
            eight["minimum_forward_traffic_per_sample"]
            < one["minimum_forward_traffic_per_sample"]
        ), track_id


def test_large_batch_throughput_approaches_the_flat_asymptote():
    large = evaluate_track("cloud", batch_size=4096)
    larger = evaluate_track("cloud", batch_size=8192)

    gain = (larger["throughput"] / large["throughput"]).to("").magnitude
    assert 1.0 < gain < 1.01


def test_execution_terms_respond_only_to_their_causal_rates():
    baseline = evaluate_track("mobile")
    faster_compute = evaluate_track("mobile", effective_compute_rate=Q_(40, "GFLOP / second"))
    faster_memory = evaluate_track("mobile", effective_bandwidth=Q_(40, "GB / second"))
    lower_launch = evaluate_track("mobile", launch_overhead=Q_(0.05, "millisecond"))

    assert faster_compute["compute_time"] < baseline["compute_time"]
    assert faster_compute["movement_time"] == baseline["movement_time"]
    assert faster_compute["launch_time"] == baseline["launch_time"]
    assert faster_memory["movement_time"] < baseline["movement_time"]
    assert faster_memory["compute_time"] == baseline["compute_time"]
    assert lower_launch["launch_time"] < baseline["launch_time"]
    assert lower_launch["compute_time"] == baseline["compute_time"]


def test_growth_instruments_change_only_the_selected_network_dimension():
    input_growth = growth_comparison("edge", "input", 2)
    hidden_growth = growth_comparison("edge", "hidden", 2)
    batch_growth = growth_comparison("edge", "batch", 2)

    assert input_growth["result"]["layer_dims"][0] == 2 * input_growth["baseline"]["layer_dims"][0]
    assert input_growth["result"]["batch_size"] == input_growth["baseline"]["batch_size"]
    assert hidden_growth["result"]["layer_dims"][1] == 2 * hidden_growth["baseline"]["layer_dims"][1]
    assert batch_growth["result"]["layer_dims"] == batch_growth["baseline"]["layer_dims"]
    assert batch_growth["result"]["batch_size"] == 2 * batch_growth["baseline"]["batch_size"]


def test_repair_actions_share_one_failed_baseline_and_change_distinct_causes():
    comparisons = {action: repair_comparison("mobile", action) for action in ("tensor", "batch", "format")}
    baselines = [item["baseline"]["inputs"] for item in comparisons.values()]

    assert baselines[0] == baselines[1] == baselines[2]
    assert not comparisons["tensor"]["baseline"]["fits_memory"]
    assert comparisons["tensor"]["result"]["parameters"] < comparisons["tensor"]["baseline"]["parameters"]
    assert comparisons["batch"]["result"]["batch_size"] < comparisons["batch"]["baseline"]["batch_size"]
    assert comparisons["format"]["result"]["precision"] == "fp16"


def test_held_out_repair_outcomes_are_defined_for_every_track():
    outcomes = {}
    for track_id in TRACKS:
        comparisons = {
            action: repair_comparison(track_id, action) for action in ("tensor", "batch", "format")
        }
        baseline_inputs = [item["baseline"]["inputs"] for item in comparisons.values()]
        assert baseline_inputs[0] == baseline_inputs[1] == baseline_inputs[2], track_id
        assert all(not item["baseline"]["fits_memory"] for item in comparisons.values()), track_id
        outcomes[track_id] = {
            action: comparison["result"]["fits_memory"] for action, comparison in comparisons.items()
        }

    assert outcomes["cloud"]["tensor"]
    assert not any(outcomes[track][action] for track in ("tinyml", "mobile", "edge") for action in outcomes[track])


def test_each_track_memory_requirement_can_fail_and_recover_at_the_boundary():
    for track_id in TRACKS:
        baseline = evaluate_track(track_id)
        required = baseline["total_state_memory"]
        failed = evaluate_track(track_id, memory_limit=required - Q_(1, "byte"))
        passed = evaluate_track(track_id, memory_limit=required)

        assert not failed["fits_memory"], track_id
        assert _bytes(failed["memory_margin"]) == -1
        assert passed["fits_memory"], track_id
        assert _bytes(passed["memory_margin"]) == 0


def test_memory_limit_is_noncausal_for_computation_and_only_changes_feasibility():
    tight = evaluate_track("edge", memory_limit=Q_(1, "byte"))
    loose = evaluate_track("edge", memory_limit=Q_(1, "GiB"))

    for key in (
        "parameters",
        "batch_macs",
        "mac_convention_flops",
        "activation_memory",
        "total_state_memory",
        "minimum_forward_traffic",
    ):
        assert tight[key] == loose[key]
    assert not tight["fits_memory"]
    assert loose["fits_memory"]


def test_compare_preserves_supplied_baseline_and_reports_physical_deltas():
    baseline = evaluate_track("mobile", batch_size=2)
    comparison = compare_track(
        "mobile",
        baseline=baseline,
        baseline_options={"batch_size": 99},
        intervention_options={"batch_size": 4},
    )

    assert comparison["baseline"]["batch_size"] == 2
    assert comparison["result"]["batch_size"] == 4
    assert comparison["delta"]["parameters"] == 0
    assert comparison["delta"]["batch_macs"] > 0
    assert comparison["delta"]["activation_memory"].is_compatible_with(ureg.byte)
    assert comparison["delta"]["minimum_forward_traffic"].is_compatible_with(ureg.byte)


def test_evaluations_carry_complete_replay_inputs_and_snapshot_as_json():
    result = evaluate_track(
        "tinyml",
        layer_dims=(10, 5, 2),
        batch_size=3,
        precision="fp16",
        phase="training",
        optimizer="momentum",
        optimizer_precision="fp32",
        memory_limit=Q_(32, "KiB"),
    )

    assert result["inputs"] == {
        "track_id": "tinyml",
        "layer_dims": [10, 5, 2],
        "batch_size": 3,
        "precision": "fp16",
        "phase": "training",
        "optimizer": "momentum",
        "optimizer_precision": "fp32",
        "memory_limit": {"magnitude": 32768.0, "unit": "byte"},
        "effective_compute_rate": {"magnitude": 50000000.0, "unit": "flop / second"},
        "effective_bandwidth": {"magnitude": 100000000.0, "unit": "byte / second"},
        "launch_overhead": {"magnitude": 0.05, "unit": "millisecond"},
    }
    frozen = snapshot(result)
    assert frozen["total_state_memory"]["unit"] == "byte"
    assert frozen["throughput"]["unit"] == "1 / second"
    assert frozen["effective_compute_rate"]["unit"] == "flop / second"
    assert frozen["effective_bandwidth"]["unit"] == "byte / second"
    assert json.loads(json.dumps(frozen))["inputs"] == result["inputs"]


@pytest.mark.parametrize("numeric_format", ["fp32", "fp16", "int8"])
def test_rounding_case_executes_real_arrays_for_each_format(numeric_format):
    result = run_numerical_case("rounding", numeric_format)

    assert len(result["represented_inputs"]) == len(result["weights"]) == len(result["products"]) == 2
    assert result["bytes_per_element"] in {1, 2, 4}
    assert math.isfinite(result["reference_output"])
    assert result["inputs"] == {"case_id": "rounding", "numeric_format": numeric_format}
    assert "quality" not in result


def test_fp16_case_detects_nonfinite_arithmetic_overflow():
    fp32 = run_numerical_case("float_overflow", "fp32")
    fp16 = run_numerical_case("float_overflow", "fp16")

    assert not fp32["arithmetic_overflow"]
    assert math.isfinite(fp32["output"])
    assert fp16["arithmetic_overflow"]
    assert fp16["output_nonfinite"]
    assert math.isinf(fp16["output"])
    frozen = snapshot(fp16)
    assert frozen["output"] is None
    assert frozen["output_status"] == "nonfinite_overflow"
    assert frozen["products"] == [None, None]


def test_int8_case_detects_unit_scale_operand_clipping_with_int32_accumulation():
    result = run_numerical_case("integer_clipping", "int8")

    assert result["quantization_scale"] == 1.0
    assert result["accumulator_format"] == "int32"
    assert result["input_clipped"]
    assert not result["arithmetic_overflow"]
    assert result["output_changed"]
    assert result["reference_output"] == 400.0
    assert result["output"] == 254
    assert result["output"] != result["reference_output"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_dims": (4,)},
        {"layer_dims": (4, 0)},
        {"layer_dims": (4, 2), "batch_size": 0},
        {"layer_dims": (4, 2), "precision": "bf16"},
        {"layer_dims": (4, 2), "phase": "backward"},
        {"layer_dims": (4, 2), "optimizer": "unknown"},
    ],
)
def test_network_inputs_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        analyze_network(**kwargs)


def test_default_batch_provides_contrast_for_all_tracks():
    for track_id in TRACKS:
        batch = default_batch(track_id)
        assert batch >= 2
        assert batch >= TRACKS[track_id]["batch_size"]


def test_track_and_numerical_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        evaluate_track("serverless")
    with pytest.raises(ValueError):
        evaluate_track("tinyml", memory_limit=Q_(1, "millisecond"))
    with pytest.raises(ValueError):
        default_batch("serverless")
    with pytest.raises(ValueError):
        run_numerical_case("unknown", "fp16")
    with pytest.raises(ValueError):
        run_numerical_case("rounding", "fp8")


def _assert_capture_arms_replay(capture):
    for field in ("baseline", "result", "chosen_result"):
        saved = capture.get(field)
        if saved is not None:
            assert snapshot(replay(capture["model_key"], saved["inputs"])) == saved


def test_actual_notebook_a_through_d_capture_shapes_replay_every_arm():
    growth = growth_comparison("tinyml", "hidden", 2)
    phase_inference = evaluate_track("mobile", precision="fp16", phase="inference")
    phase_training = evaluate_track(
        "mobile", precision="fp16", phase="training", optimizer="momentum"
    )
    batch_one = evaluate_track("edge", batch_size=1)
    batch_eight = evaluate_track("edge", batch_size=8)
    numeric_fp32 = run_numerical_case("float_overflow", "fp32")
    numeric_fp16 = run_numerical_case("float_overflow", "fp16")
    captures = (
        {
            "part": "A",
            "model_key": EVALUATE_MODEL_KEY,
            "baseline": snapshot(growth["baseline"]),
            "result": snapshot(growth["result"]),
            "chosen_result": None,
        },
        {
            "part": "B",
            "model_key": EVALUATE_MODEL_KEY,
            "baseline": snapshot(phase_inference),
            "result": snapshot(phase_training),
            "chosen_result": None,
        },
        {
            "part": "C",
            "model_key": EVALUATE_MODEL_KEY,
            "baseline": snapshot(batch_one),
            "result": snapshot(batch_eight),
            "chosen_result": None,
        },
        {
            "part": "D",
            "model_key": NUMERICAL_MODEL_KEY,
            "baseline": snapshot(numeric_fp32),
            "result": snapshot(numeric_fp16),
            "chosen_result": None,
        },
    )

    for capture in captures:
        _assert_capture_arms_replay(capture)


def test_actual_notebook_e_chosen_and_no_change_capture_shapes_replay():
    tensor = repair_comparison("cloud", "tensor")
    batch = repair_comparison("tinyml", "batch")
    chosen_capture = {
        "part": "E",
        "model_key": EVALUATE_MODEL_KEY,
        "baseline": snapshot(tensor["baseline"]),
        "result": snapshot(tensor["result"]),
        "chosen_result": snapshot(tensor["result"]),
        "result_role": "chosen intervention",
    }
    no_change_capture = {
        "part": "E",
        "model_key": EVALUATE_MODEL_KEY,
        "baseline": snapshot(batch["baseline"]),
        "result": snapshot(batch["result"]),
        "chosen_result": snapshot(batch["baseline"]),
        "result_role": "rejected alternative",
    }

    _assert_capture_arms_replay(chosen_capture)
    _assert_capture_arms_replay(no_change_capture)


def test_replay_rejects_ambiguous_keys_and_incomplete_or_extra_inputs():
    evaluated = snapshot(evaluate_track("tinyml"))
    missing = dict(evaluated["inputs"])
    missing.pop("batch_size")
    extra = {**evaluated["inputs"], "unknown": 1}

    with pytest.raises(ValueError):
        replay("v1_05_experiments", evaluated["inputs"])
    with pytest.raises(ValueError):
        replay(EVALUATE_MODEL_KEY, missing)
    with pytest.raises(ValueError):
        replay(EVALUATE_MODEL_KEY, extra)
    with pytest.raises(ValueError):
        replay(NUMERICAL_MODEL_KEY, {"case_id": "rounding"})
