"""Finite goal and selected-label replay invariants, not empirical forecasts."""

import json
import pytest
from mlsysim.labs.experiments.v1_15_experiments import TRACKS, proxy_alignment, feedback_replay


@pytest.mark.parametrize("track", TRACKS)
def test_proxy_optimization_can_reverse_independent_goal(track):
    proxy = proxy_alignment(track)
    goal = proxy_alignment(track, policy="benefit_first")
    assert proxy["slots"] == goal["slots"]
    assert proxy["proxy_clicks"] > goal["proxy_clicks"]
    assert proxy["independent_benefits"] < goal["independent_benefits"]
    assert proxy["benefit_gate"] == "fail" and goal["benefit_gate"] == "pass"
    unseen = proxy_alignment(track, policy="benefit_first", outcome_audit="none")
    assert unseen["proxy_clicks"] == goal["proxy_clicks"]
    assert unseen["independent_benefits"] is None and unseen["benefit_gate"] == "insufficient_evidence"
    assert unseen["independent_followups"] == 0


@pytest.mark.parametrize("track", TRACKS)
def test_labels_selected_now_change_only_later_rounds(track):
    routine = feedback_replay(track)
    audit = feedback_replay(track, observation_policy="independent_sample")
    assert [r["groups"]["affected"]["exposed"] for r in routine["history"]] == [0, 0, 0]
    assert [r["groups"]["affected"]["exposed"] for r in audit["history"]] == [0, 0, 4]
    assert (
        routine["history"][0]["groups"]["affected"]["estimate_before"]
        == audit["history"][0]["groups"]["affected"]["estimate_before"]
    )
    assert audit["total_followups"] > 0
    assert audit["total_displaced_slots"] == 0
    assert audit["final_exposure_gap"] < routine["final_exposure_gap"]
    assert (
        feedback_replay(track, observation_policy="independent_sample", rounds=1)["final_exposure_gap"]
        == routine["history"][0]["exposure_gap"]
    )


def test_exploration_has_a_distinct_serving_cost_and_no_omniscient_labels():
    result = feedback_replay("cloud", observation_policy="exploration")
    assert result["total_displaced_slots"] > 0 and result["total_followups"] == 0
    for interval in result["history"]:
        for row in interval["records"]:
            assert row["observed"] == row["exposed"]
            assert (row["label"] is not None) == row["observed"]
    assert result["final_exposure_gap"] == 0


def test_audits_do_not_change_decisions_in_the_same_round_and_complete_identifies_labels():
    routine = feedback_replay("edge")
    full = feedback_replay("edge", observation_policy="complete")
    assert [r["exposed"] for r in routine["history"][0]["records"]] == [
        r["exposed"] for r in full["history"][0]["records"]
    ]
    assert all(row["observed"] for interval in full["history"] for row in interval["records"])
    assert all(g["unobserved"] == 0 for interval in full["history"] for g in interval["groups"].values())


@pytest.mark.parametrize(
    "runner,kwargs",
    [
        (proxy_alignment, {}),
        (proxy_alignment, {"outcome_audit": "none"}),
        (feedback_replay, {}),
        (feedback_replay, {"observation_policy": "exploration"}),
    ],
)
def test_replay_json_is_exact(runner, kwargs):
    result = runner("tinyml", **kwargs)
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert runner(**restored["inputs"]) == restored


@pytest.mark.parametrize(
    "runner,kwargs",
    [
        (proxy_alignment, {"policy": "invented"}),
        (proxy_alignment, {"outcome_audit": "guessed"}),
        (feedback_replay, {"rounds": 4}),
        (feedback_replay, {"rounds": True}),
        (feedback_replay, {"observation_policy": "invented"}),
    ],
)
def test_unsupported_controls_fail(runner, kwargs):
    with pytest.raises(ValueError):
        runner("tinyml", **kwargs)


@pytest.mark.parametrize("policy", ("routine", "independent_sample", "complete", "exploration"))
def test_next_estimate_is_reconstructed_from_only_observed_prior_labels(policy):
    result = feedback_replay("cloud", observation_policy=policy)
    counters = {g: [seed["successes"], seed["count"]] for g, seed in result["seed_labels"].items()}
    for interval in result["history"]:
        for group, report in interval["groups"].items():
            assert report["estimate_before"] == counters[group][0] / counters[group][1]
            for record in interval["records"]:
                if record["population"] == group and record["observed"]:
                    counters[group][0] += record["label"]
                    counters[group][1] += 1
            assert report["estimate_after"] == counters[group][0] / counters[group][1]
