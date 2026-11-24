"""Invariant and worked-scenario tests for TemporalShiftModel (carbon-aware time shifting)."""

import itertools
import random

import pytest

from mlsysim import Q_, Scenarios
from mlsysim.solvers import TemporalShiftModel


def _signal(values):
    return [Q_(v, "g/kWh") for v in values]


def _solve(values, slack_h, duration_h=2, power="1 MW", **kw):
    return TemporalShiftModel().solve(_signal(values), Q_(power), Q_(duration_h, "hour"), Q_(slack_h, "hour"), **kw)


def _oracle(values, power_kw, duration, slack, deferral_kwh=0.0, standby_kw=0.0, free=None):
    """Independent hourly brute force written without the solver's helpers (grams CO2e)."""
    free = [True] * len(values) if free is None else free
    best = None
    for start in range(slack + 1):
        if not all(free[start + i] for i in range(duration)):
            continue
        grams = 0.0
        for hour in range(start, start + duration):
            grams += power_kw * values[hour]
        if start > 0:
            grams += deferral_kwh * values[0]
            for hour in range(start):
                grams += standby_kw * values[hour]
        if best is None or grams < best[1]:
            best = (start, grams)
    return best


def test_zero_slack_starts_immediately_and_saves_nothing():
    r = _solve([600, 500, 100, 100], slack_h=0)
    assert not r.shifted
    assert r.best_start.m_as("hour") == 0
    assert r.expected_completion.m_as("hour") == 2
    assert r.net_saving.m_as("kg") == 0
    assert r.candidate_starts == 1
    assert r.min_beneficial_wait is None


def test_flat_signal_never_shifts_even_without_overhead():
    r = _solve([400] * 10, slack_h=6)
    assert not r.shifted
    assert r.net_saving.m_as("kg") == 0


def test_flat_signal_with_overhead_makes_every_wait_worse():
    r = _solve([400] * 10, slack_h=6, deferral_energy=Q_("10 kWh"), standby_power=Q_("50 kW"))
    assert not r.shifted
    immediate = r.net_emissions_by_start[0].m_as("kg")
    assert all(v.m_as("kg") > immediate for v in r.net_emissions_by_start[1:])


def test_immediate_emissions_are_energy_times_intensity():
    # 1 MW for 2 h at 600 then 500 g/kWh -> 1000 kWh * 600 + 1000 kWh * 500 = 1,100 kg
    r = _solve([600, 500, 100, 100], slack_h=2)
    assert r.immediate_emissions.m_as("kg") == pytest.approx(1100.0)


def test_waits_for_the_clean_window_when_waiting_is_free():
    r = _solve([600, 500, 100, 100], slack_h=2)
    assert r.shifted and r.best_start.m_as("hour") == 2
    assert r.expected_completion.m_as("hour") == 4
    assert r.best_net_emissions.m_as("kg") == pytest.approx(200.0)
    assert r.saving_fraction == pytest.approx(900 / 1100)


def test_chooses_lowest_integrated_window_not_lowest_single_hour():
    # Hour 1 is the cleanest single hour, but a 2-hour run starting there hits the
    # 900 g/kWh spike; the run starting at hour 3 has the lower total.
    r = _solve([500, 50, 900, 200, 200], slack_h=3)
    assert r.best_start.m_as("hour") == 3


def test_waiting_overhead_can_erase_the_gross_saving():
    # Gross saving of waiting one hour is 1 MWh * (600 - 100) = 500 kg; a 1 MWh
    # deferral cost charged at the 600 g/kWh release hour costs 600 kg.
    r = TemporalShiftModel().solve(
        _signal([600, 100]), Q_("1 MW"), Q_("1 hour"), Q_("1 hour"), deferral_energy=Q_("1 MWh")
    )
    assert not r.shifted
    assert r.net_emissions_by_start[1].m_as("kg") == pytest.approx(100 + 600)


def test_overhead_is_charged_only_when_waiting():
    r = _solve([600, 500, 100, 100], slack_h=2, deferral_energy=Q_("100 kWh"), standby_power=Q_("200 kW"))
    # Overhead at wait 2: 100 kWh * 600 + 200 kWh * (600 + 500) = 60 + 220 = 280 kg
    assert r.shift_overhead_emissions.m_as("kg") == pytest.approx(280.0)
    assert r.best_net_emissions.m_as("kg") == pytest.approx(200.0 + 280.0)


def test_more_slack_never_increases_best_net_emissions():
    rng = random.Random(7)
    values = [rng.uniform(50, 800) for _ in range(30)]
    previous = None
    for slack in range(0, 25):
        r = _solve(values, slack_h=slack, duration_h=4, deferral_energy=Q_("20 kWh"), standby_power=Q_("50 kW"))
        best = r.best_net_emissions.m_as("kg")
        if previous is not None:
            assert best <= previous + 1e-9
        previous = best


def test_matches_independent_brute_force_oracle():
    rng = random.Random(11)
    for _ in range(200):
        duration = rng.randint(1, 4)
        slack = rng.randint(0, 8)
        values = [rng.choice([0, 50, 120, 300, 450, 800]) for _ in range(duration + slack)]
        free = [rng.random() > 0.2 for _ in values]
        deferral_kwh = rng.choice([0.0, 5.0, 40.0])
        standby_kw = rng.choice([0.0, 10.0, 80.0])
        expected = _oracle(values, 100.0, duration, slack, deferral_kwh, standby_kw, free)
        kwargs = dict(
            deferral_energy=Q_(deferral_kwh, "kWh"), standby_power=Q_(standby_kw, "kW"), capacity_available=free
        )
        if expected is None:
            with pytest.raises(ValueError, match="capacity"):
                _solve(values, slack, duration, power="100 kW", **kwargs)
            continue
        r = _solve(values, slack, duration, power="100 kW", **kwargs)
        assert r.best_start.m_as("hour") == expected[0]
        assert r.best_net_emissions.m_as("g") == pytest.approx(expected[1])


def test_net_saving_is_never_negative_when_capacity_is_unconstrained():
    rng = random.Random(5)
    for _ in range(50):
        values = [rng.uniform(0, 900) for _ in range(20)]
        r = _solve(values, slack_h=8, duration_h=3, deferral_energy=Q_("30 kWh"), standby_power=Q_("80 kW"))
        nets = [v.m_as("kg") for v in r.net_emissions_by_start]
        assert r.net_saving.m_as("kg") >= 0
        assert r.best_net_emissions.m_as("kg") == pytest.approx(min(nets))


def test_ties_choose_the_earliest_start():
    r = _solve([500, 100, 500, 100, 100], slack_h=3, duration_h=1)
    assert r.best_start.m_as("hour") == 1


def test_min_beneficial_wait_is_the_first_wait_that_beats_starting_now():
    r = _solve([500, 600, 450, 100], slack_h=3, duration_h=1)
    assert r.min_beneficial_wait.m_as("hour") == 2
    assert r.best_start.m_as("hour") == 3


def test_sub_hour_steps():
    signal = _signal([400, 400, 100, 100, 100, 100])
    r = TemporalShiftModel().solve(signal, Q_("1 MW"), Q_("30 minute"), Q_("1 hour"), step=Q_("15 minute"))
    assert r.best_start.m_as("hour") == pytest.approx(0.5)
    # 1 MW for 15 minutes is 250 kWh per step; two steps at 400 g/kWh -> 200 kg
    assert r.immediate_emissions.m_as("kg") == pytest.approx(200.0)


def test_capacity_excludes_windows_that_overlap_booked_steps():
    # Hour 2 is booked, so windows starting at hours 1 and 2 are infeasible and the
    # cleanest remaining 2-hour window starts at hour 3.
    r = _solve([600, 500, 100, 100, 200, 200], slack_h=4, capacity_available=[True, True, False, True, True, True])
    assert r.best_start.m_as("hour") == 3
    assert r.net_emissions_by_start[1] is None and r.net_emissions_by_start[2] is None
    assert r.candidate_starts == 3


def test_capacity_can_force_a_start_dirtier_than_starting_now():
    r = _solve([100, 100, 900, 900], slack_h=2, capacity_available=[False, True, True, True])
    assert not r.immediate_feasible
    assert r.best_start.m_as("hour") == 1
    assert r.net_saving.m_as("kg") < 0
    assert r.min_beneficial_wait is None


def test_saving_fraction_is_undefined_when_the_start_now_reference_is_zero():
    # Start-now emits nothing but has no capacity; the only feasible start emits 90 kg.
    r = _solve([0, 90], slack_h=1, duration_h=1, power="1 MW", capacity_available=[False, True])
    assert not r.immediate_feasible
    assert r.immediate_emissions.m_as("kg") == 0
    assert r.net_saving.m_as("kg") == pytest.approx(-90.0)
    assert r.saving_fraction is None


def test_saving_fraction_is_undefined_for_an_all_zero_signal():
    r = _solve([0, 0, 0], slack_h=1, duration_h=1)
    assert r.saving_fraction is None and not r.shifted


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_intensity_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        _solve([400, bad, 400], slack_h=1, duration_h=1)


@pytest.mark.parametrize("name, unit", [("power", "kW"), ("deferral_energy", "kWh"), ("standby_power", "kW")])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_scalars_raise(name, unit, bad):
    args = {"intensities": _signal([400] * 4), "power": Q_("1 MW"), "duration": Q_(1, "hour"), "slack": Q_(1, "hour")}
    args[name] = Q_(bad, unit)
    with pytest.raises(ValueError, match="finite"):
        TemporalShiftModel().solve(**args)


@pytest.mark.parametrize("name", ["duration", "slack", "step"])
def test_non_finite_time_spans_raise(name):
    args = {"intensities": _signal([400] * 4), "power": Q_("1 MW"), "duration": Q_(1, "hour"), "slack": Q_(1, "hour")}
    args[name] = Q_(float("nan"), "hour")
    with pytest.raises(ValueError, match="finite"):
        TemporalShiftModel().solve(**args)


@pytest.mark.parametrize("bad", ["False", "no", 0, 1, None])
def test_capacity_flags_must_be_bool(bad):
    # The string "False" is truthy in Python, so it must be rejected.
    with pytest.raises(ValueError, match="bool"):
        _solve([400, 400, 400], slack_h=1, duration_h=1, capacity_available=[bad, True, True])


def test_extreme_finite_inputs_that_overflow_raise():
    with pytest.raises(ValueError, match="overflow"):
        TemporalShiftModel().solve([Q_(1e308, "g/kWh")] * 2, Q_(1e308, "kW"), Q_(1, "hour"), Q_(1, "hour"))


def test_no_feasible_start_raises():
    with pytest.raises(ValueError, match="capacity"):
        _solve([400] * 4, slack_h=2, capacity_available=[False, False, False, True])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"duration": Q_(0, "hour")}, "at least one step"),
        ({"duration": Q_(1.5, "hour")}, "whole number"),
        ({"slack": Q_(-1, "hour")}, "whole number"),
        ({"slack": Q_(10, "hour")}, "cover duration"),
        ({"power": Q_(-1, "kW")}, "nonnegative"),
        ({"standby_power": Q_(-1, "kW")}, "nonnegative"),
        ({"capacity_available": [True]}, "capacity_available must cover"),
    ],
)
def test_invalid_inputs_raise(kwargs, message):
    args = {"intensities": _signal([400] * 4), "power": Q_("1 MW"), "duration": Q_(1, "hour"), "slack": Q_(1, "hour")}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TemporalShiftModel().solve(**args)


def test_negative_intensity_raises():
    with pytest.raises(ValueError, match="nonnegative"):
        _solve([400, -1, 400], slack_h=1, duration_h=1)


def test_evening_release_scenarios():
    full = Scenarios.EveningReleaseTimeShift.result
    short = Scenarios.EveningReleaseShortSlack.result
    none = Scenarios.EveningReleaseNoSlack.result
    booked = Scenarios.EveningReleaseCapacityLimited.result
    costly = Scenarios.EveningReleaseCostlyStandby.result

    # Start now: 1 MWh per hour over 520, 560, 600, 620, 580, 520 g/kWh.
    assert full.immediate_emissions.m_as("kg") == pytest.approx(3400.0)
    # Waiting 16 h: run over 320..220 g/kWh = 1,320 kg, plus 100 kW standby over
    # the first 16 hours (7,130 g/kWh-h) = 713 kg.
    assert full.shifted and full.best_start.m_as("hour") == 16
    assert full.expected_completion.m_as("hour") == 22
    assert full.shift_overhead_emissions.m_as("kg") == pytest.approx(713.0)
    assert full.net_saving.m_as("kg") == pytest.approx(1367.0)

    assert short.best_start.m_as("hour") == 3
    assert none.net_saving.m_as("kg") == 0 and not none.shifted
    assert none.net_saving <= short.net_saving <= full.net_saving

    # Booked capacity blocks every window that reaches the clean morning hours.
    assert booked.best_start.m_as("hour") == 3
    assert booked.net_saving < full.net_saving

    # 400 kW standby: no wait saves carbon.
    assert not costly.shifted and costly.min_beneficial_wait is None
    assert costly.net_saving.m_as("kg") == 0
    # Cleaner run windows still exist on the same signal.
    assert full.best_run_emissions < costly.immediate_emissions


@pytest.mark.parametrize("values", list(itertools.product([100, 500], repeat=3)))
def test_small_exhaustive_signals_match_oracle(values):
    values = list(values)
    expected = _oracle(values, 1000.0, 1, 2)
    r = _solve(values, slack_h=2, duration_h=1)
    assert r.best_start.m_as("hour") == expected[0]
    assert r.best_net_emissions.m_as("g") == pytest.approx(expected[1])
