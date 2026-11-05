"""Boundary checks for scenario wiring and deadline-limited policy accounting.

Inputs are constructed scenarios. Enumerated policy paths provide expectations
independent of the solver's aggregate expressions; no hardware measurement or
empirical success-rate claim is made.
"""

import math

import pytest

from mlsysim import Q_, Scenarios
from mlsysim.engine.results import GripperTransferResult
from mlsysim.engine.scenario_types import GripperTransferScenario
from mlsysim.physics.agents import calc_short_circuit_cost
from mlsysim.solvers import DeadlinePolicyModel


def test_gripper_result_uses_geometry_and_force_fields():
    scenario = GripperTransferScenario(
        name="constructed gripper transfer", description="unit and clearance boundary test",
        command=0.5, donor_stroke=Q_(0.12, "meter"), target_stroke=Q_(80, "mm"),
        part_width=Q_(40, "mm"), donor_hold_force=Q_(18, "N"), target_force_rating=Q_(6, "N"),
    )
    result = scenario.result
    assert isinstance(result, GripperTransferResult)
    assert result.donor_opening.m_as("mm") == 0.5 * 0.12 * 1000
    assert result.target_opening.m_as("mm") == 0.5 * 80
    assert result.stroke_excess.m_as("mm") == 0.5 * 0.12 * 1000 - 80
    assert result.force_ratio == 18 / 6
    assert result.donor_clears is True
    # Equal opening and width has no positive clearance.
    assert result.target_clears is False


def test_registered_gripper_snapshot_preserves_existing_scalar_properties():
    scenario = Scenarios.GripperStrokeTransfer
    result = scenario.result
    for name in GripperTransferResult.model_fields:
        if hasattr(scenario, name):
            assert getattr(result, name) == getattr(scenario, name)


def solve_policy(deadline, cap=3, initial=0.4, revision=0.5):
    return DeadlinePolicyModel().solve(
        decode_per_token=Q_(1, "second"), prompt=10, output=8, diagnostic=2,
        breadth=3, max_rounds=cap, deadline=Q_(deadline, "second"), check_time=Q_(2, "second"),
        initial_pass=initial, revision_pass=revision,
        input_price=Q_(3, "USD/token"), output_price=Q_(5, "USD/token"),
    )


@pytest.mark.parametrize("deadline,cap,admitted", [
    (0, 3, 0), (9, 3, 0), (10, 3, 1), (19, 3, 1), (20, 3, 2),
    (30, 3, 2), (20, 1, 1), (20, 2, 2),
])
def test_deadline_feedback_aggregates_only_admitted_execution_paths(deadline, cap, admitted):
    p, revision = 0.4, 0.5
    result = solve_policy(deadline, cap, p, revision)
    # Leaves: probability, success, input tokens, output tokens, rounds.
    if admitted == 0:
        paths = [(1.0, False, 0, 0, 0)]
    elif admitted == 1:
        paths = [(p, True, 10, 8, 1), (1 - p, False, 10, 8, 1)]
    else:
        paths = [
            (p, True, 10, 8, 1),
            ((1 - p) * revision, True, 10 + 10 + 8 + 2, 8 + 8, 2),
            ((1 - p) * (1 - revision), False, 10 + 10 + 8 + 2, 8 + 8, 2),
        ]
    success = sum(probability for probability, passed, *_ in paths if passed)
    inputs = sum(probability * tokens for probability, _, tokens, _, _ in paths)
    outputs = sum(probability * tokens for probability, _, _, tokens, _ in paths)
    expected_rounds = sum(probability * rounds for probability, _, _, _, rounds in paths)
    assert result.rounds == admitted
    assert result.c_pass == pytest.approx(success)
    assert result.c_input == pytest.approx(inputs)
    assert result.c_output == pytest.approx(outputs)
    assert result.c_rounds == pytest.approx(expected_rounds)
    assert result.c_cost.m_as("USD") == pytest.approx(inputs * 3 + outputs * 5)
    assert result.c_time.m_as("second") == pytest.approx(expected_rounds * (8 + 2))
    assert result.c_time <= result.max_c_time <= Q_(deadline, "second")
    if success:
        assert result.c_cost_per_pass.m_as("USD") == pytest.approx((inputs * 3 + outputs * 5) / success)
    else:
        assert math.isinf(result.c_cost_per_pass.m_as("USD"))
        assert math.isinf(result.c_output_per_pass)


@pytest.mark.parametrize("deadline", [0, 9])
def test_deadline_rejects_all_policies_when_no_round_fits(deadline):
    result = solve_policy(deadline)
    for policy in ("a", "b", "c"):
        assert getattr(result, f"{policy}_pass") == 0
        assert getattr(result, f"{policy}_input") == 0
        assert getattr(result, f"{policy}_output") == 0
        assert getattr(result, f"{policy}_cost").m_as("USD") == 0
        assert math.isinf(getattr(result, f"{policy}_cost_per_pass").m_as("USD"))
    assert math.isnan(result.b_over_a)


def test_zero_initial_success_can_still_succeed_after_admitted_revision():
    result = solve_policy(20, initial=0, revision=0.5)
    assert result.a_pass == result.b_pass == 0
    assert result.c_pass == 0.5
    assert result.c_rounds == 2
    assert math.isinf(result.a_cost_per_pass.m_as("USD"))
    assert math.isfinite(result.c_cost_per_pass.m_as("USD"))
    assert math.isnan(result.b_over_a)


@pytest.mark.parametrize("arguments", [
    {"deadline": -1}, {"deadline": 20, "cap": 0},
    {"deadline": 20, "initial": -0.1}, {"deadline": 20, "revision": 1.1},
])
def test_deadline_policy_rejects_invalid_budgets_and_probabilities(arguments):
    with pytest.raises(ValueError):
        solve_policy(**arguments)


@pytest.mark.parametrize("probabilities,costs", [([0.5], [2, 4]), ([0.5, 0.5], [2]), ([], [])])
def test_short_circuit_requires_one_cost_for_each_probability(probabilities, costs):
    with pytest.raises(ValueError):
        calc_short_circuit_cost(probabilities, costs)


def test_short_circuit_cost_matches_path_enumeration_with_units():
    # First check passes half of requests; only those pay the second cost.
    cost, survival = calc_short_circuit_cost([0.5, 0.25], [Q_(2, "ms"), Q_(0.004, "second")])
    paths = [(0.5, 2, False), (0.5 * 0.75, 2 + 4, False), (0.5 * 0.25, 2 + 4, True)]
    assert cost.m_as("ms") == sum(probability * duration for probability, duration, _ in paths)
    assert survival == sum(probability for probability, _, passed in paths if passed)


def test_short_circuit_accepts_equal_length_numpy_sequences():
    import numpy as np

    cost, survival = calc_short_circuit_cost(np.array([0.5, 0.25]), np.array([2, 4]))
    assert cost == 2 + 0.5 * 4
    assert survival == 0.5 * 0.25


@pytest.mark.parametrize("unit,scale", [("second", 1), ("ms", 1000), ("us", 1000000)])
@pytest.mark.parametrize("deadline_seconds,expected", [(0.3, 1), (0.6, 2)])
def test_decimal_deadline_boundaries_are_unit_invariant(unit, scale, deadline_seconds, expected):
    result = DeadlinePolicyModel().solve(
        Q_(0.1 * scale, unit), 10, 3, 2, 3, 3,
        Q_(deadline_seconds * scale, unit), Q_(0, unit),
        0.4, 0.5, Q_(3, "USD/token"), Q_(5, "USD/token"),
    )
    assert result.rounds == expected
    assert result.max_c_time.m_as("second") == pytest.approx(deadline_seconds)


@pytest.mark.parametrize("deadline_seconds,expected", [(0.3, 0), (0.6, 1)])
def test_decimal_deadline_does_not_admit_work_for_next_float_below_boundary(deadline_seconds, expected):
    deadline = math.nextafter(deadline_seconds, -math.inf)
    result = DeadlinePolicyModel().solve(
        Q_(100, "ms"), 10, 3, 2, 3, 3,
        Q_(deadline, "second"), Q_(0, "second"),
        0.4, 0.5, Q_(3, "USD/token"), Q_(5, "USD/token"),
    )
    assert result.rounds == expected


@pytest.mark.parametrize("check,deadline", [(Q_(100, "ms"), Q_(400, "ms")), (Q_(0.1, "s"), Q_(0.4, "s"))])
def test_decimal_deadline_includes_check_time_before_admission(check, deadline):
    result = DeadlinePolicyModel().solve(
        Q_(100, "ms"), 10, 3, 2, 3, 3,
        deadline, check, 0.4, 0.5, Q_(3, "USD/token"), Q_(5, "USD/token"),
    )
    assert result.rounds == 1


@pytest.mark.parametrize("probability", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_short_circuit_rejects_invalid_conditional_probabilities(probability):
    with pytest.raises(ValueError):
        calc_short_circuit_cost([1.0, probability], [Q_(0, "ms"), Q_(1, "ms")])


@pytest.mark.parametrize("cost", [-1, math.nan, math.inf, Q_(-1, "ms"), Q_(math.nan, "ms"), Q_(math.inf, "ms")])
def test_short_circuit_rejects_invalid_costs_even_after_zero_reach(cost):
    with pytest.raises(ValueError):
        calc_short_circuit_cost([0.0, 0.5], [0 * cost, cost])


def test_short_circuit_permits_zero_cost_and_zero_survival():
    cost, probability = calc_short_circuit_cost([0.0, 1.0], [Q_(0, "ms"), Q_(1, "ms")])
    assert cost == Q_(0, "ms")
    assert probability == 0.0
