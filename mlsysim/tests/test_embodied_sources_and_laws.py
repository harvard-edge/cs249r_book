"""Sourced embodied registry facts and the physical laws promoted with them (2026-09-29).

Each registry value is pinned to the number its provenance record quotes, and
each promoted formula is checked against a hand-derivable case.
"""

from __future__ import annotations

import math

import pytest

from mlsysim import Embodied, Hardware
from mlsysim.core.units import ureg
from mlsysim.physics.robotics import (
    calc_free_space_expiry,
    calc_linear_invalidation_horizon,
    calc_quasistatic_spill_acceleration,
    calc_reduced_mass,
    calc_scurve_rest_to_rest_time,
    calc_transient_contact_speed_limit,
    calc_transient_impact_force,
)

AISLE = Embodied.Scenario.WarehouseAisle
RM = AISLE.machine


# ── ISO/TS 15066 hand body model ────────────────────────────────────────────
def test_iso_hand_body_model_values_and_provenance():
    assert AISLE.k_contact_human.to("N/mm").magnitude == pytest.approx(75.0)
    assert AISLE.m_contact_human.to("kg").magnitude == pytest.approx(0.6)
    for leaf in (AISLE.k_contact_human, AISLE.m_contact_human):
        assert leaf.provenance.id == "prov:iso-ts-15066-hand-body-model"
        assert "2203.02706" in leaf.provenance.url


def test_reduced_mass_limits():
    m = calc_reduced_mass(12.0 * ureg.kg, 0.6 * ureg.kg)
    assert m.to("kg").magnitude == pytest.approx(12.0 * 0.6 / 12.6)
    # A very heavy second body (a clamped contact) leaves the first mass.
    assert calc_reduced_mass(12.0 * ureg.kg, 1e9 * ureg.kg).to("kg").magnitude == pytest.approx(12.0)


def test_transient_force_free_vs_clamped():
    v, m, k, mh = 0.10 * ureg("m/s"), 12.0 * ureg.kg, 75.0 * ureg("N/mm"), 0.6 * ureg.kg
    clamped = calc_transient_impact_force(v, m, k)
    free = calc_transient_impact_force(v, m, k, human_mass=mh)
    assert clamped.to("N").magnitude == pytest.approx(0.10 * math.sqrt(75_000 * 12.0))
    assert free.to("N").magnitude == pytest.approx(0.10 * math.sqrt(75_000 * 12.0 * 0.6 / 12.6))
    assert free < clamped


def test_contact_speed_limit_inverts_force():
    m, k, mh, f = 12.0 * ureg.kg, 75.0 * ureg("N/mm"), 0.6 * ureg.kg, 50.0 * ureg.N
    for human in (None, mh):
        v = calc_transient_contact_speed_limit(f, m, k, human_mass=human)
        back = calc_transient_impact_force(v, m, k, human_mass=human)
        assert back.to("N").magnitude == pytest.approx(50.0)


# ── Invalidation horizons ───────────────────────────────────────────────────
def test_linear_horizon_and_floor():
    t = calc_linear_invalidation_horizon(0.1 * ureg.mm, 10 * ureg("mm/s"), 1.0 * ureg.mm)
    assert t.to("ms").magnitude == pytest.approx(90.0)
    assert calc_linear_invalidation_horizon(2 * ureg.mm, 10 * ureg("mm/s"), 1 * ureg.mm).magnitude == 0.0


def test_free_space_expiry_matches_law():
    t = calc_free_space_expiry(1.10 * ureg.m, 50 * ureg.mm, 100 * ureg.mm, 1.6 * ureg("m/s"))
    assert t.to("ms").magnitude == pytest.approx((1.10 - 0.05 - 0.10) / 1.6 * 1000)
    assert calc_free_space_expiry(0.1 * ureg.m, 50 * ureg.mm, 100 * ureg.mm, 1.6 * ureg("m/s")).magnitude == 0.0


# ── Jerk-limited rest-to-rest move and quasi-static spill ───────────────────
def test_scurve_rest_to_rest_limits():
    d, a = 0.5 * ureg.radian, 7.5 * ureg("rad/s**2")
    # Very large jerk -> bang-bang: T = 2 sqrt(D/a).
    fast = calc_scurve_rest_to_rest_time(d, a, 1e9 * ureg("rad/s**3"))
    assert fast["duration"].to("s").magnitude == pytest.approx(2 * math.sqrt(0.5 / 7.5), rel=1e-6)
    # Plateau case satisfies D = a x (x + a/j), T = 2 (x + a/j).
    j = 604.8 * ureg("rad/s**3")
    res = calc_scurve_rest_to_rest_time(d, a, j)
    tj = 7.5 / 604.8
    x = res["duration"].to("s").magnitude / 2 - tj
    assert res["reaches_max_acceleration"]
    assert 7.5 * x * (x + tj) == pytest.approx(0.5, rel=1e-9)
    # No-plateau case satisfies D = 2 j tau^3, T = 4 tau.
    small = calc_scurve_rest_to_rest_time(1e-4 * ureg.radian, a, j)
    tau = small["duration"].to("s").magnitude / 4
    assert not small["reaches_max_acceleration"]
    assert 2 * 604.8 * tau**3 == pytest.approx(1e-4, rel=1e-9)


def test_quasistatic_spill_acceleration():
    a = calc_quasistatic_spill_acceleration(40 * ureg.mm, 10 * ureg.mm)
    assert a.to("m/s**2").magnitude == pytest.approx(9.80665 * 10 / 40)


# ── Franka FER / Panda limits ───────────────────────────────────────────────
def test_panda_kinematics_and_limits():
    arm = RM.arm
    assert arm.upper_arm_length.to("m").magnitude == pytest.approx(0.316)
    assert arm.forearm_length.to("m").magnitude == pytest.approx(0.384)
    assert [q.to("rad/s").magnitude for q in arm.joint_velocity_limits] == pytest.approx(
        [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
    assert [q.to("rad/s**2").magnitude for q in arm.joint_acceleration_limits] == pytest.approx(
        [15, 7.5, 10, 12.5, 15, 20, 20])
    assert [q.to("rad/s**3").magnitude for q in arm.joint_jerk_limits] == pytest.approx(
        [7500, 3750, 5000, 6250, 7500, 10000, 10000])
    assert arm.joint_torque_rate_limit.to("N*m/s").magnitude == pytest.approx(1000.0)
    assert arm.cartesian_velocity_limit.to("m/s").magnitude == pytest.approx(1.7)
    assert arm.upper_arm_length.provenance.id == "prov:franka-fer-control-limits"


# ── Orin memory interface and LPDDR5-6400 timings ───────────────────────────
def test_orin_bus_matches_bandwidth():
    mem = Hardware.Edge.JetsonAGXOrin.memory
    assert mem.bus_width.to("bit").magnitude == pytest.approx(256)
    assert mem.pin_data_rate.to("Mbit/s").magnitude == pytest.approx(6400)
    bw = (mem.bus_width * mem.pin_data_rate / ureg.bit).to("GB/s").magnitude
    assert bw == pytest.approx(204.8)
    assert abs(bw / mem.bandwidth.to("GB/s").magnitude - 1) < 0.005


def test_lpddr5_6400_timings():
    t = Hardware.Tech.Memory.LPDDR5_6400
    assert t.t_ck.to("ns").magnitude == pytest.approx(1.25)
    assert t.t_rcd.to("ns").magnitude == pytest.approx(15 * 1.25)
    assert t.t_rp.to("ns").magnitude == pytest.approx(15 * 1.25)
    assert t.metadata.provenance.id == "prov:lpddr5-6400-timings"


# ── Partial-slowdown stretch (Amdahl read as a slowdown) ────────────────────
def test_partial_slowdown_stretch_and_breakeven():
    from mlsysim.physics import calc_amdahls_speedup, calc_partial_slowdown_stretch, calc_stall_share_breakeven

    assert calc_partial_slowdown_stretch(0.1, 21.0) == pytest.approx(3.0)
    assert calc_partial_slowdown_stretch(0.1, 21.0) == pytest.approx(1 / calc_amdahls_speedup(0.1, 1 / 21.0))
    f = calc_stall_share_breakeven(250 / 135, 21.0)
    assert f == pytest.approx((250 / 135 - 1) / 20)
    assert calc_partial_slowdown_stretch(f, 21.0) == pytest.approx(250 / 135)
    assert calc_stall_share_breakeven(2.0, 1.0) == 1.0
    assert calc_stall_share_breakeven(50.0, 2.0) == 1.0  # capped: the whole task may be slowed


# ── Importance sampling and the cross-entropy method (exponential tail) ─────
def test_monte_carlo_sample_size():
    from mlsysim.physics import calc_monte_carlo_sample_size

    assert calc_monte_carlo_sample_size(0.01, 0.1) == pytest.approx(0.99 / (0.01 * 0.01))


def test_importance_sampling_exponential_reduces_to_plain_mc():
    from mlsysim.physics import calc_importance_sampling_exponential

    # Proposal equal to the operational model: w = 1, ESS per draw 1, and the
    # relative variance is the Bernoulli (1 - p) / p.
    r = calc_importance_sampling_exponential(20.0, 5.0, 5.0)
    p = math.exp(-4.0)
    assert r["failure_probability"] == pytest.approx(p)
    assert r["ess_per_draw"] == pytest.approx(1.0)
    assert r["relative_variance"] == pytest.approx((1 - p) / p)
    assert r["proposal_failure_share"] == pytest.approx(p)
    with pytest.raises(ValueError):
        calc_importance_sampling_exponential(20.0, 5.0, 2.0)  # eta >= 2 lam: infinite variance


def test_cross_entropy_and_optimal_exponential_proposal():
    from mlsysim.core.units import ureg
    from mlsysim.physics import (
        calc_cross_entropy_exponential_proposal,
        calc_importance_sampling_exponential,
        calc_optimal_exponential_proposal_mean,
    )

    gamma, m0 = 100.775 * ureg.ms, 5.0 * ureg.ms
    ce = calc_cross_entropy_exponential_proposal(gamma, m0, 0.1)
    assert ce["n_levels"] == 4
    assert ce["levels"][0].to("ms").magnitude == pytest.approx(-5.0 * math.log(0.1))
    assert ce["proposal_mean"].to("ms").magnitude == pytest.approx(105.775)
    opt = calc_optimal_exponential_proposal_mean(gamma, m0)
    rv_ce = calc_importance_sampling_exponential(gamma, m0, ce["proposal_mean"])["relative_variance"]
    rv_opt = calc_importance_sampling_exponential(gamma, m0, opt)["relative_variance"]
    assert rv_opt <= rv_ce < rv_opt * 1.001
    # The optimum is a stationary point: nearby proposals do no better.
    for bump in (0.98, 1.02):
        assert calc_importance_sampling_exponential(gamma, m0, opt * bump)["relative_variance"] > rv_opt


# ── Confidence in a structured safety argument ──────────────────────────────
def test_argument_confidence():
    from mlsysim.physics import calc_conjunctive_confidence, calc_multilegged_confidence

    assert calc_conjunctive_confidence([0.99] * 10) == pytest.approx(0.99**10)
    assert calc_conjunctive_confidence([0.99] * 10) < 0.99
    assert calc_multilegged_confidence([0.9, 0.9]) == pytest.approx(0.99)
    assert calc_multilegged_confidence([0.9]) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        calc_conjunctive_confidence([])


def test_insulation_class_f_limit_is_iec60085_155c():
    """IEC 60085:2007 thermal class 155 (F); read by the vol4 body and training chapters."""
    import mlsysim
    from mlsysim.core.units import ureg

    limit = mlsysim.INSULATION_CLASS_F_LIMIT
    assert abs(limit.to(ureg.degC).magnitude - 155.0) < 1e-9
    assert abs(limit.to(ureg.kelvin).magnitude - 428.15) < 1e-9
