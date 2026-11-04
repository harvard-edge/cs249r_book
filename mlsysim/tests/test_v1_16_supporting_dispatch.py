"""Supporting studies must be recomputed independently of primary evidence."""

from copy import deepcopy

import pytest

from mlsysim.labs.experiments import v1_03_experiments as lifecycle
from mlsysim.labs.experiments import v1_05_experiments as compute
from mlsysim.labs.experiments import v1_06_experiments as architecture
from mlsysim.labs.experiments import v1_15_experiments as responsibility
from mlsysim.labs.experiments.v1_16_experiments import replay_evidence_arm, replay_capture


def _primary(reference):
    base = lifecycle.trace_requirement("tinyml", "energy")
    return {**base, "supplemental_reference": reference}


def _refs():
    for runner, kwargs in (
        (compute.run_dependency_trace, {}),
        (compute.run_nonlinearity_case, {}),
        (architecture.attention_state_comparison, {"track_id": "tinyml"}),
        (architecture.language_phase_reference, {}),
        (architecture.embedding_lookup_reference, {}),
        (lifecycle.release_gate_trial, {"track_id": "tinyml", "check_plan": "combined"}),
    ):
        ref = compute.snapshot(runner(**kwargs))
        ref.setdefault("model_key", f"{runner.__module__.split('.')[-1]}.{runner.__name__}")
        yield ref


@pytest.mark.parametrize("reference", list(_refs()))
def test_supporting_result_replays_with_primary_return_preserved(reference):
    primary = _primary(reference)
    expected = lifecycle.replay("v1_03_experiments.trace_requirement", primary["inputs"])
    result = replay_evidence_arm(
        "v1_03_experiments", "v1_03_experiments.trace_requirement", primary
    )
    assert result == expected


@pytest.mark.parametrize("reference", list(_refs()))
def test_tampered_supporting_result_is_rejected(reference):
    changed = deepcopy(reference)
    changed["fabricated_result"] = "not produced by the simulator"
    with pytest.raises(ValueError, match="differs from exact replay"):
        replay_evidence_arm(
            "v1_03_experiments", "v1_03_experiments.trace_requirement", _primary(changed)
        )


def test_unknown_supporting_method_is_rejected():
    with pytest.raises(ValueError):
        replay_evidence_arm(
            "v1_03_experiments",
            "v1_03_experiments.trace_requirement",
            _primary({"model_key": "v1_06_experiments.invented", "inputs": {}}),
        )


@pytest.mark.parametrize(
    "runner", (responsibility.fairness_criteria, responsibility.selective_feedback, responsibility.lifecycle_policy, responsibility.proxy_alignment, responsibility.feedback_replay)
)
def test_new_responsibility_methods_dispatch(runner):
    result = runner("tinyml")
    assert (
        replay_evidence_arm("v1_15_experiments", f"v1_15_experiments.{runner.__name__}", result)
        == result
    )


def test_supporting_reference_in_alternative_is_verified():
    reference = next(_refs())
    arm = _primary(reference)
    snapshot = {
        "model_key": "v1_03_experiments.trace_requirement",
        "baseline": arm,
        "result": arm,
        "alternatives": [reference],
    }
    replay_capture("v1_03_experiments", snapshot)
    snapshot["alternatives"][0]["fabricated"] = True
    with pytest.raises(ValueError, match="differs from exact replay"):
        replay_capture("v1_03_experiments", snapshot)


def test_dataflow_packet_and_calibration_support_replay():
    from mlsysim.labs.experiments import v1_11_experiments as accelerator
    from mlsysim.labs.experiments import v1_12_experiments as benchmarking

    values = {
        "m": 8,
        "n": 8,
        "k": 8,
        "tile_m": 4,
        "tile_n": 4,
        "tile_k": 4,
        "precision": "fp16",
        "stationary": "weight",
    }
    packet = accelerator.make_replay_packet(
        track="tinyml", experiment="stationary_dataflow", inputs=values
    )
    packet["outputs"] = accelerator.to_jsonable(accelerator.replay_experiment(packet))
    calibration = {
        "model_key": "v1_12_experiments.compare_calibration",
        **benchmarking.to_jsonable(benchmarking.compare_calibration("tinyml")),
    }
    arm = _primary(next(_refs()))
    capture = {
        "model_key": "v1_03_experiments.trace_requirement",
        "track": "tinyml",
        "baseline": arm,
        "result": arm,
        "alternatives": [packet, calibration],
    }
    replay_capture("v1_03_experiments", capture)
    packet["outputs"]["fabricated"] = True
    with pytest.raises(ValueError, match="differs from exact replay"):
        replay_capture("v1_03_experiments", capture)


def test_separate_language_serving_references_replay_and_reject_tampering():
    from mlsysim.labs.experiments import v1_13_experiments as serving

    schedule = serving.language_scheduling_to_snapshot(
        serving.language_scheduling_reference(chunks=4)
    )
    capacity = serving.state_capacity_to_snapshot(
        serving.state_capacity(serving.get_language_serving_reference(pressure="baseline")),
        pressure="baseline",
    )
    arm = _primary(next(_refs()))
    reference = {
        "supplemental_reference": {
            "selected": {
                "model_key": "v1_13.language_scheduling",
                "inputs": {"chunks": 4},
                "result": schedule,
            },
            "capacity": {
                "model_key": "v1_13.language_state_capacity",
                "inputs": {"pressure": "baseline"},
                "result": capacity,
            },
        }
    }
    capture = {
        "model_key": "v1_03_experiments.trace_requirement",
        "track": "tinyml",
        "baseline": arm,
        "result": arm,
        "alternatives": [reference],
    }
    replay_capture("v1_03_experiments", capture)
    schedule["prefill_work_ms"] += 1.0
    with pytest.raises(ValueError, match="differs from exact replay"):
        replay_capture("v1_03_experiments", capture)


def test_continuous_batching_supporting_reference_replays():
    from mlsysim.labs.experiments.v1_13_experiments import language_batching_reference
    reference = language_batching_reference(policy="continuous", length_mix="mixed", arrival_spacing_ms=16, context_scale=1)
    arm = {**reference, "model_key": "v1_13.language_batching"}
    assert replay_evidence_arm("v1_13_experiments", "v1_13.language_batching", arm) == reference
