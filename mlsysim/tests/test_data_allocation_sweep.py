"""Controlled allocation accounting, fitted evidence limits, and exact replay."""

import json

import pytest

from mlsysim import Literature
from mlsysim.labs.experiments.v1_09_experiments import (
    ALLOCATION_MODEL_SCALES, TRACK_IDS, compute_allocation, replay, serialize_value,
)


@pytest.mark.parametrize("track", TRACK_IDS)
def test_all_model_allocations_spend_identical_compute(track):
    sweep = compute_allocation(track)
    budget = sweep["compute_budget"].to("flop").magnitude
    for row in sweep["allocations"]:
        params = row["parameters"].to("count").magnitude
        tokens = row["processed_tokens"].to("count").magnitude
        assert row["training_compute"].to("flop").magnitude == pytest.approx(budget)
        assert Literature.Chinchilla.ComputeConstant * params * tokens == pytest.approx(budget)
        assert row["repeated_tokens"].magnitude == 0
    baseline = next(row for row in sweep["allocations"] if row["model_scale"] == 1)
    larger = next(row for row in sweep["allocations"] if row["model_scale"] == 2)
    assert larger["parameters"] == 2 * baseline["parameters"]
    assert larger["processed_tokens"] * 2 == baseline["processed_tokens"]
    assert larger["model_loss_term"] < baseline["model_loss_term"]
    assert larger["data_loss_term"] > baseline["data_loss_term"]


def test_fitted_curve_has_tradeoff_and_never_calls_largest_universal_winner():
    sweep = compute_allocation("cloud")
    rows = sweep["allocations"]
    best = min(rows, key=lambda row: row["predicted_loss_nats_per_token"])
    assert best["model_scale"] not in (min(ALLOCATION_MODEL_SCALES), max(ALLOCATION_MODEL_SCALES))
    assert sweep["best_supported_model_scale"] == best["model_scale"]
    fit = Literature.Chinchilla
    for row in rows:
        expected = (fit.ParametricLossE
                    + fit.ParametricLossA / row["parameters"].magnitude ** fit.ParametricLossAlpha
                    + fit.ParametricLossB / row["processed_tokens"].magnitude ** fit.ParametricLossBeta)
        assert row["predicted_loss_nats_per_token"] == pytest.approx(expected)


@pytest.mark.parametrize("track", TRACK_IDS)
def test_unique_data_cap_changes_evidence_not_invented_repeat_quality(track):
    fresh = compute_allocation(track, unique_data_scale=4)
    capped = compute_allocation(track, unique_data_scale=0.25)
    for original, limited in zip(fresh["allocations"], capped["allocations"], strict=True):
        assert limited["parameters"] == original["parameters"]
        assert limited["processed_tokens"] == original["processed_tokens"]
        assert limited["training_compute"] == original["training_compute"]
        assert limited["unique_tokens_used"] + limited["repeated_tokens"] == limited["processed_tokens"]
        if limited["requires_repetition"]:
            assert limited["predicted_loss_nats_per_token"] is None
            assert limited["data_loss_term"] is None
        else:
            assert limited["predicted_loss_nats_per_token"] == original["predicted_loss_nats_per_token"]
    assert capped["selected"]["predicted_loss_nats_per_token"] is None
    assert "not proven globally optimal" in " ".join(capped["assumptions"])


@pytest.mark.parametrize("scale", ALLOCATION_MODEL_SCALES)
@pytest.mark.parametrize("data", (0.25, 1.0, 4.0))
def test_all_allocation_controls_replay_exactly(scale, data):
    result = compute_allocation("tinyml", model_scale=scale, unique_data_scale=data)
    inputs = json.loads(json.dumps(serialize_value(result["inputs"])))
    assert replay(result["model_key"], inputs) == result


@pytest.mark.parametrize("kwargs", ({"track_id": "unknown"}, {"model_scale": True},
                                   {"model_scale": 3}, {"unique_data_scale": float("nan")},
                                   {"unique_data_scale": False}))
def test_allocation_rejects_unsupported_inputs(kwargs):
    with pytest.raises(ValueError):
        compute_allocation(**({"track_id": "tinyml"} | kwargs))
