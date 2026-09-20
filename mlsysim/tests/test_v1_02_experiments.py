import json
from pathlib import Path
import sys

import pytest

LABS_ROOT = Path(__file__).resolve().parents[2] / "labs"
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

from mlsysim.engine.v1_02_experiments import (
    FILTERING_ALTERNATIVES,
    MODEL_KEY,
    TRACKS,
    compare_placements,
    placement_accounting,
    replay,
    sustained_operation,
    workload_feasibility,
)
from mlsysbook_labs.experiment_evidence import capture_evidence


def test_every_track_has_valid_baselines_and_reachable_hard_failures():
    for track_id in TRACKS:
        fit = workload_feasibility(track_id)
        memory_failure = workload_feasibility(track_id, memory_scale=3.0)
        execution_failure = workload_feasibility(track_id, execution_scale=20.0)
        sustained = sustained_operation(track_id)
        continuous = sustained_operation(track_id, duty_cycle=1.0)
        insufficient_observation = sustained_operation(track_id, duty_cycle=0.0)

        assert fit["feasible"], track_id
        assert memory_failure["violations"] == ["memory"], track_id
        assert execution_failure["violations"] == ["execution_time"], track_id
        assert sustained["sustained_feasible"], track_id
        assert not continuous["sustained_feasible"], track_id
        assert continuous["violations"] == ["average_power"], track_id
        assert not insufficient_observation["sustained_feasible"], track_id
        assert insufficient_observation["violations"] == ["observation_time"], track_id


def test_remote_latency_is_the_sum_of_explicit_path_terms():
    result = placement_accounting("mobile", filter_id="event_filter")

    assert result["remote_latency_ms"] == pytest.approx(
        result["filter_preprocess_ms"]
        + result["request_upload_ms"]
        + result["propagation_rtt_ms"]
        + result["remote_execution_ms"]
        + result["response_download_ms"],
        abs=2e-6,
    )


def test_payload_and_connection_rate_move_only_transfer_related_terms():
    baseline = placement_accounting("edge", payload_mb=4.0, connection_mbps=50.0)
    larger_payload = placement_accounting(
        "edge", payload_mb=8.0, connection_mbps=50.0
    )
    faster_link = placement_accounting(
        "edge", payload_mb=4.0, connection_mbps=100.0
    )

    assert larger_payload["request_upload_ms"] == pytest.approx(
        2 * baseline["request_upload_ms"]
    )
    assert faster_link["request_upload_ms"] == pytest.approx(
        baseline["request_upload_ms"] / 2
    )
    for result in (larger_payload, faster_link):
        assert result["remote_execution_ms"] == baseline["remote_execution_ms"]
        assert result["propagation_rtt_ms"] == baseline["propagation_rtt_ms"]


def test_distance_changes_propagation_without_changing_serialization_or_compute():
    near = placement_accounting("cloud", distance_km=100.0)
    far = placement_accounting("cloud", distance_km=1_000.0)

    assert far["propagation_rtt_ms"] > near["propagation_rtt_ms"]
    assert far["request_upload_ms"] == near["request_upload_ms"]
    assert far["response_download_ms"] == near["response_download_ms"]
    assert far["remote_execution_ms"] == near["remote_execution_ms"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_connectivity_is_a_hard_gate_and_does_not_change_local_outcomes(track_id):
    online = compare_placements(track_id, connectivity_available=True)
    offline = compare_placements(track_id, connectivity_available=False)

    assert offline["local"] == online["local"]
    for result in offline["remote"].values():
        assert result["latency_ms"] is None
        assert not result["feasible"]
        assert "connectivity" in result["violations"]


def test_memory_and_execution_controls_are_causally_separate():
    baseline = workload_feasibility("mobile")
    memory_only = workload_feasibility("mobile", memory_scale=1.2)
    execution_only = workload_feasibility("mobile", execution_scale=1.2)

    assert memory_only["execution_ms"] == baseline["execution_ms"]
    assert execution_only["required_memory_mb"] == baseline["required_memory_mb"]
    assert memory_only["required_memory_mb"] > baseline["required_memory_mb"]
    assert execution_only["execution_ms"] > baseline["execution_ms"]


def test_average_duty_cycle_energy_uses_active_and_idle_power():
    result = sustained_operation("edge", duty_cycle=0.25, horizon_hours=10.0)
    expected_power_w = 25.0 * 0.25 + 4.0 * 0.75

    assert result["average_power_w"] == pytest.approx(expected_power_w)
    assert result["energy_wh"] == pytest.approx(expected_power_w * 10.0)
    assert result["active_observation_hours"] == pytest.approx(2.5)


@pytest.mark.parametrize("track_id", TRACKS)
def test_lower_duty_saves_energy_but_can_fail_the_mission_requirement(track_id):
    track = TRACKS[track_id]
    required = track["minimum_observation_fraction"]
    meets = sustained_operation(track_id, duty_cycle=required)
    misses = sustained_operation(track_id, duty_cycle=required / 2)

    assert misses["energy_wh"] < meets["energy_wh"]
    assert meets["service_requirement_met"]
    assert not misses["service_requirement_met"]
    assert "observation_time" in misses["violations"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_filtering_alternatives_have_opposite_side_tradeoffs(track_id):
    results = compare_placements(track_id)["remote"]
    raw = results["raw"]
    event = results["event_filter"]
    feature = results["feature_summary"]

    assert raw["uploaded_payload_mb"] > event["uploaded_payload_mb"] > feature[
        "uploaded_payload_mb"
    ]
    assert raw["retained_information_pct"] > event[
        "retained_information_pct"
    ] > feature["retained_information_pct"]
    assert event["feasible"]
    assert not raw["feasible"]
    assert {"deadline", "upload"}.intersection(raw["violations"])
    assert not feature["feasible"]
    assert "retained_information" in feature["violations"]


def test_filtering_does_not_change_remote_compute_or_propagation():
    results = [
        placement_accounting("mobile", filter_id=filter_id)
        for filter_id in FILTERING_ALTERNATIVES
    ]

    assert len({result["remote_execution_ms"] for result in results}) == 1
    assert len({result["propagation_rtt_ms"] for result in results}) == 1
    assert len({result["response_download_ms"] for result in results}) == 1


def test_filter_preprocessing_power_is_track_specific_and_within_active_envelope():
    tiny = placement_accounting("tinyml", filter_id="event_filter")
    cloud = placement_accounting("cloud", filter_id="event_filter")

    assert tiny["filter_preprocess_power_w"] != cloud["filter_preprocess_power_w"]
    for track_id, track in TRACKS.items():
        for filter_id in FILTERING_ALTERNATIVES:
            result = placement_accounting(track_id, filter_id=filter_id)
            assert result["filter_preprocess_power_w"] <= track["active_power_w"]


def test_deadline_boundary_predicts_opposite_sides_of_the_deadline():
    baseline = placement_accounting("mobile", filter_id="event_filter")
    boundary = baseline["deadline_boundary_payload_mb"]
    assert boundary is not None

    below = placement_accounting(
        "mobile", payload_mb=boundary * 0.99, filter_id="event_filter"
    )
    above = placement_accounting(
        "mobile", payload_mb=boundary * 1.01, filter_id="event_filter"
    )

    assert below["remote_latency_ms"] < below["deadline_ms"]
    assert above["remote_latency_ms"] > above["deadline_ms"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"track_id": "unknown"}, "track_id"),
        ({"track_id": "edge", "payload_mb": 0}, "payload_mb"),
        ({"track_id": "edge", "connection_mbps": 0}, "connection_mbps"),
        ({"track_id": "edge", "connectivity_available": 1}, "Boolean"),
        ({"track_id": "edge", "filter_id": "learned"}, "filter_id"),
    ],
)
def test_invalid_placement_inputs_fail_clearly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        placement_accounting(**kwargs)


def test_invalid_duty_cycle_is_rejected():
    with pytest.raises(ValueError, match="at most 1"):
        sustained_operation("tinyml", duty_cycle=1.01)


@pytest.mark.parametrize("track_id", TRACKS)
def test_actual_notebook_capture_shape_replays_after_json_roundtrip(track_id):
    profile = TRACKS[track_id]
    a_base = placement_accounting(
        track_id,
        payload_mb=profile["payload_options_mb"][0],
        connection_mbps=profile["connection_options_mbps"][-1],
    )
    a_result = placement_accounting(
        track_id,
        payload_mb=profile["payload_options_mb"][-1],
        connection_mbps=profile["connection_mbps"],
    )
    b_base = workload_feasibility(track_id)
    b_result = workload_feasibility(
        track_id, memory_scale=2.0, execution_scale=2.0
    )
    c_base = sustained_operation(track_id)
    c_result = sustained_operation(track_id, duty_cycle=0.75)
    c_low = sustained_operation(track_id, duty_cycle=0.0)
    placements = compare_placements(
        track_id,
        payload_mb=profile["payload_options_mb"][-1],
        connection_mbps=profile["connection_mbps"],
    )
    candidates = {
        "local": placements["local"],
        "raw": placements["remote"]["raw"],
        "event_filter": placements["remote"]["event_filter"],
        "feature_summary": placements["remote"]["feature_summary"],
    }
    captures = {
        "A": capture_evidence(
            track=track_id,
            part="A",
            prediction="fail",
            inputs={"payload_mb": a_result["inputs"]["payload_mb"]},
            baseline=a_base,
            result=a_result,
            model_key=MODEL_KEY,
        ),
        "B": capture_evidence(
            track=track_id,
            part="B",
            prediction="memory",
            inputs={"memory_scale": 2.0, "execution_scale": 2.0},
            baseline=b_base,
            result=b_result,
            model_key=MODEL_KEY,
        ),
        "C": capture_evidence(
            track=track_id,
            part="C",
            prediction="average_power",
            inputs={"duty_cycle": 0.75},
            baseline=c_base,
            result=c_result,
            alternatives=(c_low,),
            model_key=MODEL_KEY,
        ),
        "D": capture_evidence(
            track=track_id,
            part="D",
            prediction="event_filter",
            inputs={"choice": "event_filter", "rejected": "raw"},
            baseline=candidates["raw"],
            result=candidates["event_filter"],
            alternatives=tuple(candidates.values()),
            decision="event_filter",
            chosen_result=candidates["event_filter"],
            result_role="chosen placement",
            model_key=MODEL_KEY,
        ),
    }

    snapshots = json.loads(
        json.dumps(
            {part: capture.to_dict() for part, capture in captures.items()},
            allow_nan=False,
        )
    )
    for snapshot in snapshots.values():
        arms = [snapshot["baseline"], snapshot["result"]]
        arms.extend(snapshot["alternatives"])
        if snapshot["chosen_result"] is not None:
            arms.append(snapshot["chosen_result"])
        for arm in arms:
            assert replay(snapshot["model_key"], arm["inputs"]) == arm


def test_replay_rejects_unknown_models_and_input_shapes():
    with pytest.raises(ValueError, match="model_key"):
        replay("other", {"track_id": "edge", "placement": "local"})
    with pytest.raises(ValueError, match="evaluation shape"):
        replay(MODEL_KEY, {"track_id": "edge", "arbitrary": "function"})


def test_duty_cycle_step_allows_selecting_baseline_and_minimum():
    for track_id, track in TRACKS.items():
        step = track["duty_cycle_step"]
        baseline = track["baseline_duty_cycle"]
        minimum = track["minimum_observation_fraction"]
        assert round(baseline / step, 6).is_integer(), (
            f"{track_id}: baseline {baseline} not selectable with step {step}"
        )
        assert round(minimum / step, 6).is_integer(), (
            f"{track_id}: minimum {minimum} not selectable with step {step}"
        )
