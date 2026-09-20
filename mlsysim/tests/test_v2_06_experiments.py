import json
import math

import pytest

from mlsysim.engine.v2_06_experiments import (
    TRACK_IDS,
    algorithm_cases,
    algorithm_crossover,
    compression_comparison,
    overlap_timeline,
    overlap_bucket_options,
    routing_cases,
    semantic_exchange,
    topology_comparison,
    track_profile,
)


def test_tracks_use_honest_fleet_participants_and_keep_endpoint_traffic_distinct():
    profiles = {track_id: track_profile(track_id) for track_id in TRACK_IDS}

    assert "gateway" in profiles["tinyml"]["participant_kind"]
    assert "backend" in profiles["mobile"]["participant_kind"]
    assert "accelerator" in profiles["edge"]["participant_kind"]
    assert "accelerator" in profiles["cloud"]["participant_kind"]
    assert all(not profile["endpoint_traffic_is_collective"] for profile in profiles.values())
    assert profiles["tinyml"]["endpoint_count"] > 0
    assert profiles["cloud"]["endpoint_count"] == 0


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_every_experiment_has_a_valid_endpoint_for_every_track(track_id):
    profile = track_profile(track_id)
    routed = semantic_exchange(track_id)
    timeline = overlap_timeline(track_id, bucket_mb=profile["payload_mb"] / 4)
    compression = compression_comparison(track_id, method="fp8")

    assert routed["reduction"]["time_ms"] > 0
    assert routed["routed"]["time_ms"] > 0
    assert timeline["events"]
    assert 0 <= timeline["overlap_fraction"] <= 1
    assert compression["quality_target_reached"]
    assert compression["compressed_payload_mb"] < compression["payload_mb"]


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_ring_tree_ranking_crosses_on_both_sides_for_every_track(track_id):
    boundary = algorithm_crossover(track_id, payload_mb=0)
    crossover_mb = boundary["crossover_mb"]

    small = algorithm_crossover(track_id, payload_mb=crossover_mb / 10)
    large = algorithm_crossover(track_id, payload_mb=crossover_mb * 10)

    assert small["winner"] == "tree"
    assert large["winner"] == "ring"
    assert small["ring"]["startup_steps"] > small["tree"]["startup_steps"]
    assert large["ring"]["transfer_mb"] < large["tree"]["transfer_mb"]
    assert small["ring"]["total_ms"] > small["tree"]["total_ms"]
    assert large["ring"]["total_ms"] < large["tree"]["total_ms"]

    choices = algorithm_cases(track_id)
    by_id = {item["case_id"]: item for item in choices["cases"]}
    assert by_id["below"]["payload_mb"] < choices["crossover_mb"]
    assert by_id["above"]["payload_mb"] > choices["crossover_mb"]


def test_reduction_and_routed_exchange_have_different_payload_semantics():
    balanced = semantic_exchange("cloud", payload_mb=8)
    skewed = semantic_exchange("cloud", payload_mb=8, hotspot_fraction=0.25)

    assert balanced["reduction"]["output_is_identical_at_each_participant"]
    assert not balanced["routed"]["output_is_identical_at_each_participant"]
    assert balanced["reduction"]["time_ms"] == skewed["reduction"]["time_ms"]
    assert skewed["routed"]["time_ms"] > balanced["routed"]["time_ms"]
    assert skewed["routed"]["limiting_transfer_bound"] == "hottest_receiver"
    choices = routing_cases("cloud")
    assert choices["cases"][0]["hotspot_fraction"] == pytest.approx(1 / 128)
    assert choices["cases"][-1]["hotspot_fraction"] > choices["cases"][0]["hotspot_fraction"]


def test_oversubscription_changes_routed_bisection_not_reduction():
    baseline = semantic_exchange("edge", hotspot_fraction=1 / 64)
    constrained = semantic_exchange("edge", hotspot_fraction=1 / 64, oversubscription=4)

    assert baseline["reduction"] == constrained["reduction"]
    assert constrained["routed"]["bisection_bandwidth_gb_s"] < baseline["routed"]["bisection_bandwidth_gb_s"]
    assert constrained["routed"]["time_ms"] > baseline["routed"]["time_ms"]


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_topology_comparison_is_available_and_calibration_is_explicit(track_id):
    result = topology_comparison(track_id)

    assert result["flat"]["analytical_ms"] > 0
    assert result["hierarchical"]["analytical_ms"] > 0
    assert result["fixture"]["track_id"] == track_id
    assert "illustrative" in result["fixture"]["provenance"].lower()
    assert "independent inter-group path" in result["assumption"]


def test_matched_calibration_can_reverse_the_analytical_topology_choice():
    original = topology_comparison("cloud")
    fixture = dict(original["fixture"])
    fixture["flat_observed_ms"] = 2.0
    fixture["hierarchical_observed_ms"] = 6.0
    fixture["provenance"] = "Supplied illustrative reversal fixture."

    calibrated = topology_comparison("cloud", calibration_fixture=fixture)

    assert calibrated["analytical_winner"] == "hierarchical"
    assert calibrated["calibrated_winner"] == "flat"


def test_calibration_rejects_wrong_process_group_context():
    fixture = dict(topology_comparison("mobile")["fixture"])
    fixture["participants"] += 1

    with pytest.raises(ValueError, match="process-group"):
        topology_comparison("mobile", calibration_fixture=fixture)


def test_overlap_comes_from_layer_readiness_and_a_serialized_network():
    layers = [
        {"layer": "late", "ready_ms": 90, "gradient_mb": 12},
        {"layer": "early", "ready_ms": 20, "gradient_mb": 12},
        {"layer": "middle", "ready_ms": 55, "gradient_mb": 12},
    ]
    result = overlap_timeline("edge", bucket_mb=12, layers=layers)

    assert [event["layers"] for event in result["events"]] == [["early"], ["middle"], ["late"]]
    assert result["events"][1]["start_ms"] >= result["events"][0]["end_ms"]
    assert result["hidden_communication_ms"] > 0
    assert result["exposed_communication_ms"] < result["total_communication_ms"]


def test_bucket_size_trades_earlier_readiness_against_startup_count():
    small = overlap_timeline("mobile", bucket_mb=1)
    fused = overlap_timeline("mobile", bucket_mb=100)

    assert len(small["events"]) > len(fused["events"])
    assert small["events"][0]["start_ms"] < fused["events"][0]["start_ms"]
    assert small["total_communication_ms"] > fused["total_communication_ms"]
    assert small["hidden_communication_ms"] > fused["hidden_communication_ms"]
    choices = overlap_bucket_options("mobile")
    assert choices["options"][0]["bucket_mb"] < choices["options"][-1]["bucket_mb"]


def test_compression_counts_codec_work_and_supplied_time_to_target_evidence():
    baseline = compression_comparison("cloud", method="none")
    fp8 = compression_comparison("cloud", method="fp8")
    error_feedback = compression_comparison("cloud", method="topk_error_feedback")
    naive = compression_comparison("cloud", method="topk_naive")

    assert fp8["communication_ms"] < baseline["communication_ms"]
    assert fp8["encode_ms"] > 0 and fp8["decode_ms"] > 0
    assert fp8["steps_to_target"] > baseline["steps_to_target"]
    assert error_feedback["error_feedback"]
    assert error_feedback["quality_target_reached"]
    assert not naive["quality_target_reached"]
    assert naive["time_to_target_ms"] is None
    assert naive["quality_gap_pp"] > 0
    assert "not a universal" in naive["evidence_provenance"]
    assert fp8["baseline"]["inputs"]["method"] == "none"
    assert fp8["result"]["inputs"] == fp8["inputs"]


def test_codec_can_cost_more_than_bytes_saved_on_a_fast_small_exchange():
    baseline = compression_comparison("cloud", method="none", payload_mb=0.01, baseline_compute_ms=0)
    compressed = compression_comparison("cloud", method="topk_error_feedback", payload_mb=0.01, baseline_compute_ms=0)

    assert compressed["communication_ms"] < baseline["communication_ms"]
    assert compressed["step_ms"] > baseline["step_ms"]


def test_results_are_unit_labeled_json_and_preserve_exact_effective_inputs():
    results = [
        track_profile("tinyml"),
        algorithm_crossover("mobile"),
        semantic_exchange("edge", hotspot_fraction=0.25, oversubscription=2),
        topology_comparison("cloud"),
        overlap_timeline("mobile", bucket_mb=2),
        compression_comparison("cloud", method="fp8"),
    ]

    for result in results:
        json.dumps(result)
        assert result["inputs"]["track_id"] == result["track_id"]

    crossover = results[1]
    assert crossover["ring"]["inputs"]["payload_mb"] == crossover["payload_mb"]
    assert crossover["tree"]["inputs"]["participants"] == crossover["participants"]
    topology = results[3]
    assert topology["flat"]["inputs"]["calibration_fixture"] == topology["fixture"]
    assert topology["hierarchical"]["inputs"]["payload_mb"] == topology["payload_mb"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: track_profile("unknown"),
        lambda: algorithm_crossover("unknown"),
        lambda: algorithm_cases("unknown"),
        lambda: semantic_exchange("unknown"),
        lambda: routing_cases("unknown"),
        lambda: topology_comparison("unknown"),
        lambda: overlap_timeline("unknown", bucket_mb=1),
        lambda: overlap_bucket_options("unknown"),
        lambda: compression_comparison("unknown", method="none"),
        lambda: algorithm_crossover("edge", participants=2),
        lambda: algorithm_crossover("edge", payload_mb=math.inf),
        lambda: semantic_exchange("edge", hotspot_fraction=0),
        lambda: semantic_exchange("edge", oversubscription=0.5),
        lambda: overlap_timeline("edge", bucket_mb=0),
        lambda: overlap_timeline("edge", bucket_mb=1, layers=[]),
        lambda: compression_comparison("edge", method="invented"),
    ],
)
def test_invalid_or_inapplicable_inputs_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()


def test_track_profiles_distinguish_aggregation_vs_backward_pass():
    tinyml = track_profile("tinyml")
    mobile = track_profile("mobile")
    edge = track_profile("edge")
    cloud = track_profile("cloud")

    assert tinyml["compute_phase"] == "gateway aggregation"
    assert mobile["compute_phase"] == "backend aggregation"
    assert edge["compute_phase"] == "backward pass"
    assert cloud["compute_phase"] == "backward pass"


def test_semantic_exchange_supports_startup_as_limiting_bound():
    tinyml_small = semantic_exchange("tinyml", payload_mb=0.01)
    assert tinyml_small["routed"]["limiting_bound"] == "startup"

    cloud_hotspot = semantic_exchange("cloud", payload_mb=64, hotspot_fraction=0.3)
    assert cloud_hotspot["routed"]["limiting_bound"] == "hottest_receiver"
    assert cloud_hotspot["routed"]["limiting_transfer_bound"] == "hottest_receiver"


def test_overlap_timeline_evaluates_prediction_outcomes():
    mostly_hidden = overlap_timeline("cloud", bucket_mb=1)
    assert mostly_hidden["outcome"] == "mostly_hidden"


def test_compression_comparison_evaluates_all_prediction_outcomes():
    fp8 = compression_comparison("cloud", method="fp8")
    assert fp8["outcome"] in ("faster", "slower")
    assert fp8["result"]["outcome"] == fp8["outcome"]
    naive = compression_comparison("cloud", method="topk_naive")
    assert naive["outcome"] == "misses"
    assert naive["result"]["outcome"] == "misses"
