"""Tests for the braking-drive physics and the sourced drivetrain, bus, and joint entries (vol4 M6)."""

from __future__ import annotations

import math

import pytest

from mlsysim import Actuators, Embodied
from mlsysim.core.units import ureg
from mlsysim.physics.drivetrain import (
    calc_braking_limits,
    calc_drive_torque_deceleration_limit,
    calc_drive_voltage_headroom,
    calc_max_cg_height_ratio,
    calc_required_friction,
    calc_tipover_deceleration_limit,
    calc_traction_deceleration_limit,
)

G = 9.80665
MPS2 = ureg.meter / ureg.second**2


# ── Formulas ──────────────────────────────────────────────────────────────────

def test_traction_limit_is_mu_g_phi():
    assert calc_traction_deceleration_limit(0.6).to(MPS2).magnitude == pytest.approx(0.6 * G)
    assert calc_traction_deceleration_limit(0.5, 0.8).to(MPS2).magnitude == pytest.approx(0.4 * G)
    with pytest.raises(ValueError):
        calc_traction_deceleration_limit(0.5, 0.0)


def test_required_friction_inverts_traction_limit():
    a = 2.0 * MPS2
    mu = calc_required_friction(a, 0.8)
    assert calc_traction_deceleration_limit(mu, 0.8).to(MPS2).magnitude == pytest.approx(2.0)


def test_tipover_limit_and_its_inverse():
    a_tip = calc_tipover_deceleration_limit(1.0 * ureg.meter, 0.25 * ureg.meter)
    assert a_tip.to(MPS2).magnitude == pytest.approx(G / 4)
    # At the inverse ratio the machine sits exactly on the tipping edge.
    ratio = calc_max_cg_height_ratio(a_tip)
    assert ratio == pytest.approx(4.0)


def test_drive_torque_limit_counts_rotating_mass():
    out = calc_drive_torque_deceleration_limit(60 * ureg.newton * ureg.meter, 0.133 * ureg.meter, 2, 368 * ureg.kg)
    assert out["force"].to(ureg.newton).magnitude == pytest.approx(2 * 60 / 0.133)
    with_rot = calc_drive_torque_deceleration_limit(
        60 * ureg.newton * ureg.meter, 0.133 * ureg.meter, 2, 368 * ureg.kg, 19 * ureg.kg
    )
    assert with_rot["m_eff"].to(ureg.kg).magnitude == pytest.approx(387.0)
    assert with_rot["a_max"] < out["a_max"]


def test_voltage_headroom_terminal_equation():
    out = calc_drive_voltage_headroom(
        0.262 * ureg.volt / (ureg.revolution / ureg.minute), 1.5 * ureg.meter / ureg.second,
        0.133 * ureg.meter, 42 * ureg.volt, 25 * ureg.ampere, 0.7 * ureg.ohm,
    )
    rpm = 1.5 / 0.133 * 60 / (2 * math.pi)
    assert out["omega"].to(ureg.revolution / ureg.minute).magnitude == pytest.approx(rpm)
    assert out["e_bemf"].to(ureg.volt).magnitude == pytest.approx(0.262 * rpm)
    assert out["v_drive"].to(ureg.volt).magnitude == pytest.approx(0.262 * rpm + 17.5)
    assert out["v_brake"].to(ureg.volt).magnitude == pytest.approx(0.262 * rpm - 17.5)


def test_braking_limits_requires_tip_geometry_to_pass():
    kw = dict(wheel_torque_max=60 * ureg.newton * ureg.meter, wheel_radius=0.133 * ureg.meter,
              n_driven_wheels=2, friction_coefficient=0.6)
    no_geom = calc_braking_limits(368 * ureg.kg, 2.0 * MPS2, **kw)
    assert no_geom["a_tip"] is None and no_geom["passes"] is False
    with_geom = calc_braking_limits(368 * ureg.kg, 2.0 * MPS2, cg_height=0.8 * ureg.meter,
                                    tip_arm=0.4 * ureg.meter, **kw)
    assert with_geom["passes"] is True
    tall = calc_braking_limits(368 * ureg.kg, 2.0 * MPS2, cg_height=1.2 * ureg.meter,
                               tip_arm=0.2 * ureg.meter, **kw)
    assert tall["binding"] == "tip-over" and tall["passes"] is False


# ── Registry entries ──────────────────────────────────────────────────────────

def test_zltech_hub_motor_datasheet_values():
    h = Actuators.HubMotor.ZLTECH_ZLLG10ASM800
    assert h.wheel_radius.to(ureg.meter).magnitude == pytest.approx(0.133)
    assert h.torque_constant.to(ureg.newton * ureg.meter / ureg.ampere).magnitude == pytest.approx(2.4)
    assert h.rotor_inertia is None  # needs source
    assert h.rotating_mass_bound.to(ureg.kg).magnitude == pytest.approx(9.5)
    # The line-to-line back-EMF constant reaches the rated bus at the datasheet peak speed.
    e_max = (h.back_emf_constant * h.max_speed).to(ureg.volt).magnitude
    assert e_max == pytest.approx(47.16, abs=0.01)
    assert e_max < h.rated_voltage.to(ureg.volt).magnitude


def test_panda_class_joint_is_sourced_sum():
    j = Actuators.PandaClassJoint.CSG_25_50_ILM70x18
    m = Actuators.FramelessMotor.TQ_ILM70x18
    assert j.rotor_inertia.to(ureg.kg * ureg.meter**2).magnitude == pytest.approx(0.734e-4)
    assert j.reflected_inertia.to(ureg.kg * ureg.meter**2).magnitude == pytest.approx(0.1835)
    # TQ's rated copper loss (22.2 W) confirms the amplitude-invariant current convention.
    assert m.rated_copper_loss.to(ureg.watt).magnitude == pytest.approx(22.2, abs=0.05)
    assert (m.torque_constant * m.rated_current).to(ureg.newton * ureg.meter).magnitude == pytest.approx(1.25, rel=0.01)
    # The motor's peak through 50:1 covers the gear's repeatable peak (the pairing rule).
    assert (m.peak_torque * j.gear_ratio) >= j.peak_torque
    # Panda axes 1-4 peak torque sits within the gear's repeatable peak.
    panda = Embodied.Manipulator.Panda
    assert panda.max_torque <= j.peak_torque


def test_running_machine_drive_bus_and_braking():
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    aisle = Embodied.Scenario.WarehouseAisle
    assert rm.drive_bus_battery_low < rm.drive_bus_nominal < rm.drive_bus_full_charge < rm.drive_bus_clamp
    assert rm.drive_bus_clamp <= rm.drive_bus_converter_max
    assert rm.drive_bus_switch_rating >= 1.5 * rm.drive_bus_clamp
    assert rm.drive_bus_uvlo < rm.drive_bus_battery_low
    # Geometry the datasheets do not supply stays unset.
    assert rm.cg_height is None and rm.tip_arm_stowed is None and rm.drive_wheel_load_share is None
    phi_max = rm.drive_wheel_load_share_max
    assert phi_max == pytest.approx(300 / 368)
    h = rm.drive_unit
    res = calc_braking_limits(
        rm.loaded_mass, aisle.a_brake,
        wheel_torque_max=h.torque_constant * rm.drive_current_limit,
        wheel_radius=h.wheel_radius, n_driven_wheels=rm.drive_wheel_count,
        friction_coefficient=aisle.mu_dry, drive_load_share=phi_max,
        rotating_equivalent_mass=rm.drive_wheel_count * h.rotating_mass_bound,
    )
    # Drive torque at the motor peak current clears 2.0 m/s^2 even with the rotating-mass bound.
    assert res["a_torque"].to(MPS2).magnitude == pytest.approx(902.2556 / 387, rel=1e-4)
    assert res["a_torque"] > aisle.a_brake
    # With the load rating capping phi, the traction premise needs mu >= 0.25, not 0.204.
    assert res["mu_required"] == pytest.approx(0.2502, abs=1e-4)
    # The inspected floor cannot support the credible deceleration.
    assert calc_traction_deceleration_limit(aisle.mu_inspected_floor) < aisle.a_brake
    # Tip-over is unverified, so the check cannot pass yet.
    assert res["passes"] is False
