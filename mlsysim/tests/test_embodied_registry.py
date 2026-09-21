"""Unit tests for mlsysim.embodied registry and EmbodiedPlatform models."""

from __future__ import annotations

import pytest
from mlsysim import Embodied, Hardware, Sensors, Actuators, Models, Datasets
from mlsysim.embodied.types import EmbodiedPlatform
from mlsysim.sensors.types import CameraSensor, LiDARSensor, IMUSensor
from mlsysim.actuators.types import ActuatorJoint
from mlsysim.models.types import EmbodiedWorkload
from mlsysim.datasets.types import DatasetProfile


def test_embodied_registry_platforms_exist():
    assert Embodied.Quadruped.Spot.name == "Boston Dynamics Spot"
    assert Embodied.Humanoid.Atlas.name == "Boston Dynamics Atlas"
    assert Embodied.Humanoid.Unitree_H1.name == "Unitree H1"
    assert Embodied.Manipulator.Panda.name == "Franka Emika Panda"
    assert Embodied.Manipulator.ALOHA_Bimanual.name == "ALOHA Bimanual Manipulator"
    assert Embodied.Drone.DJI_Matrice.name == "DJI Matrice 350 RTK"
    assert Embodied.AMR.LogisticsAMR.name == "Industrial Logistics AMR"
    assert Embodied.Vehicle.Robotaxi.name == "Autonomous Vehicle / Robotaxi"
    assert Embodied.Vehicle.UberATG_VolvoXC90.name == "Uber ATG Volvo XC90"
    assert Embodied.ContinuousProcess.DED_MeltPool.name == "Direct Energy Deposition (DED) Melt Pool"


def test_embodied_registry_units():
    spot = Embodied.Quadruped.Spot
    assert spot.mass.to("kg").magnitude == pytest.approx(32.7)
    assert spot.payload_capacity.to("kg").magnitude == pytest.approx(14.0)
    assert spot.max_velocity.to("m/s").magnitude == pytest.approx(1.6)
    assert spot.control_frequency.to("Hz").magnitude == pytest.approx(1000.0)

    amr = Embodied.AMR.LogisticsAMR
    assert amr.mass.to("kg").magnitude == pytest.approx(150.0)
    assert amr.max_velocity.to("m/s").magnitude == pytest.approx(1.8)
    assert amr.max_acceleration.to("m/s^2").magnitude == pytest.approx(2.5)

    volvo = Embodied.Vehicle.UberATG_VolvoXC90
    assert volvo.mass.to("kg").magnitude == pytest.approx(2100.0)
    assert volvo.max_acceleration.to("m/s^2").magnitude == pytest.approx(8.0)


def test_embodied_registry_provenance():
    for platform in [
        Embodied.Quadruped.Spot,
        Embodied.Humanoid.Atlas,
        Embodied.Humanoid.Unitree_H1,
        Embodied.Manipulator.Panda,
        Embodied.Manipulator.ALOHA_Bimanual,
        Embodied.Drone.DJI_Matrice,
        Embodied.AMR.LogisticsAMR,
        Embodied.Vehicle.Robotaxi,
        Embodied.Vehicle.UberATG_VolvoXC90,
        Embodied.ContinuousProcess.DED_MeltPool,
    ]:
        assert isinstance(platform, EmbodiedPlatform)
        assert platform.metadata.provenance is not None
        assert platform.metadata.provenance.ref != ""


def test_hardware_embodied_drive_thor():
    thor = Hardware.Embodied.DriveThor
    assert thor.name == "NVIDIA DRIVE Thor"
    assert thor.tdp.to("W").magnitude == pytest.approx(100.0)


def test_sensors_registry():
    cam477 = Sensors.Camera.Sony_IMX477
    assert isinstance(cam477, CameraSensor)
    assert cam477.shutter_type == "rolling"
    assert cam477.resolution_width == 4056
    assert "4-lane" in cam477.interface
    assert cam477.nominal_latency.to("ms").magnitude == pytest.approx(33.3, rel=0.01)

    cam296 = Sensors.Camera.Sony_IMX296
    assert cam296.shutter_type == "global"
    assert cam296.nominal_latency.to("ms").magnitude == pytest.approx(10.0)

    lidar = Sensors.LiDAR.Ouster_OS1_64
    assert isinstance(lidar, LiDARSensor)
    assert lidar.channels == 64
    assert lidar.scan_rate.to("Hz").magnitude == pytest.approx(20.0)

    imu = Sensors.IMU.Bosch_BMI088
    assert isinstance(imu, IMUSensor)
    assert imu.sample_rate.to("Hz").magnitude == pytest.approx(1000.0)


def test_actuators_registry():
    csg17 = Actuators.HarmonicDrive.CSG_17_50
    assert isinstance(csg17, ActuatorJoint)
    assert csg17.gear_ratio == 50.0
    assert csg17.rated_torque.to("N*m").magnitude == pytest.approx(21.0)
    # Reflected inertia computed or verified: 50^2 * 0.19e-4 = 0.0475 kg*m^2
    assert csg17.reflected_inertia.to("kg*m^2").magnitude == pytest.approx(0.0475, rel=0.01)

    csg25 = Actuators.HarmonicDrive.CSG_25_50
    assert csg25.rated_torque.to("N*m").magnitude == pytest.approx(51.0)
    assert csg25.peak_torque.to("N*m").magnitude == pytest.approx(127.0)

    m107 = Actuators.QuasiDirectDrive.Unitree_M107
    assert isinstance(m107, ActuatorJoint)
    assert m107.peak_torque.to("N*m").magnitude == pytest.approx(360.0)
    assert m107.reflected_inertia.to("kg*m^2").magnitude == pytest.approx(0.012, rel=0.01)


def test_models_embodied_registry():
    openvla = Models.Embodied.OpenVLA_7B
    assert isinstance(openvla, EmbodiedWorkload)
    assert openvla.parameters.to("count").magnitude == pytest.approx(7e9)
    assert openvla.action_chunk_horizon == 1
    # Check lowering to computation graph
    cg = openvla.lower()
    assert cg.parameter_count.to("count").magnitude == pytest.approx(7e9)

    act = Models.Embodied.ACT_ALOHA
    assert act.action_chunk_horizon == 100
    assert act.action_dim == 14


def test_datasets_embodied_registry():
    oxe = Datasets.OpenXEmbodiment
    assert isinstance(oxe, DatasetProfile)
    assert oxe.episodes.to("count").magnitude == pytest.approx(1e6)
    assert oxe.embodiments == 22

    droid = Datasets.DROID
    assert droid.episodes.to("count").magnitude == pytest.approx(76000)
    assert droid.total_duration.to("hour").magnitude == pytest.approx(350.0)



# ---------------------------------------------------------------------------
# Warehouse mobile manipulator guard values (added 2026-09-21, add-only).
# Every value below is one the running machine's worked examples print. A
# registry edit that moves any of them fails here before it reaches a chapter.
# ---------------------------------------------------------------------------

from mlsysim import ureg
from mlsysim.physics.robotics import (
    calc_action_chunk_streaming_amortization,
    calc_c2_stop_suffix,
    calc_contact_yield_deadline,
    calc_max_permitted_velocity,
    calc_target_evidence_horizon,
)

_RM = Embodied.MobileManipulator.WarehouseMobileManipulator
_AISLE = Embodied.Scenario.WarehouseAisle
_MPS = ureg.meter / ureg.second
_G = 9.80665 * ureg.meter / ureg.second**2


def _mm(q):
    return q.to(ureg.millimeter).magnitude


def _v(x):
    return x * _MPS


def _const_brake(v):
    return v**2 / (2 * _AISLE.a_brake)


def _c2(v):
    return calc_c2_stop_suffix(v, _AISLE.a_brake)


def _full_budget(v):
    """Finished budget: reaction over tau_delay, C2 stop, all three overheads."""
    return v * _AISLE.tau_delay + _c2(v)["distance"] + _AISLE.fixed_overhead


def _ceiling(extra_delay=0 * ureg.millisecond, decel=None):
    """Speed at which the finished budget exactly fills d_clear (C2 stop, a_eff = a / 1.5)."""
    decel = _AISLE.a_brake if decel is None else decel
    return calc_max_permitted_velocity(
        _AISLE.d_clear, _AISLE.tau_delay + extra_delay, decel / 1.5,
        _AISLE.delta_loc + _AISLE.eps_track, _AISLE.delta_margin,
    ).to(_MPS).magnitude


def _running_totals(v):
    """Rows 1-7 of the stopping-budget table at speed v, in mm."""
    ch2 = _const_brake(v) + _AISLE.delta_loc + _AISLE.delta_margin + v * _AISLE.t_brake_onset
    ch4 = ch2 + v * (_AISLE.t_lease + _AISLE.t_tick + _AISLE.t_bus)
    ch8 = ch4 + v * _AISLE.perception_age
    ch11 = ch8 + (_c2(v)["distance"] - _const_brake(v))
    ch12 = ch11 + _AISLE.eps_track
    return {k: _mm(q) for k, q in
            {"ch2": ch2, "ch4": ch4, "ch8": ch8, "ch11": ch11, "ch12": ch12}.items()}


def _walking_person_ceiling():
    """Largest v with v*tau + 0.75 v^2/a + v_h (tau + 1.5 v/a) + overheads <= d_clear."""
    a = _AISLE.a_brake.to(ureg.meter / ureg.second**2).magnitude
    tau = _AISLE.tau_delay.to(ureg.second).magnitude
    vh = _AISLE.v_human_approach.to(_MPS).magnitude
    d = (_AISLE.d_clear - _AISLE.fixed_overhead).to(ureg.meter).magnitude
    # 0.75/a v^2 + (tau + 1.5 vh / a) v + (vh tau - d) = 0, positive root
    qa, qb, qc = 0.75 / a, tau + 1.5 * vh / a, vh * tau - d
    return (-qb + (qb**2 - 4 * qa * qc) ** 0.5) / (2 * qa)


def _weight_sweep(model):
    weights = model.parameters.magnitude * 2 * ureg.byte  # FP16
    return calc_action_chunk_streaming_amortization(
        weights, _RM.brain_soc.memory.bandwidth, _RM.chunk_horizon, _RM.dram_efficiency,
    )["t_stream"]


_QUANTITY_GUARDS = [
    # (id, getter, expected, unit, abs tolerance)
    ("m0", lambda: _RM.unloaded_mass, 318.0, "kg", 1e-9),
    ("m1", lambda: _RM.loaded_mass, 368.0, "kg", 1e-9),
    ("m_eff", lambda: _RM.arm_effective_contact_mass, 12.0, "kg", 1e-9),
    ("t_age", lambda: _AISLE.perception_age, 51.6, "ms", 1e-9),
    ("tau_delay", lambda: _AISLE.tau_delay, 133.6, "ms", 1e-9),
    ("c2_T_1.5", lambda: _c2(_v(1.5))["duration"], 1125.0, "ms", 1e-9),
    ("c2_d_1.5", lambda: _c2(_v(1.5))["distance"], 843.75, "mm", 1e-9),
    ("c2_T_1.3", lambda: _c2(_v(1.3))["duration"], 975.0, "ms", 1e-9),
    ("c2_d_1.3", lambda: _c2(_v(1.3))["distance"], 633.75, "mm", 1e-9),
    ("budget_1.3", lambda: _full_budget(_AISLE.v_aisle), 997.43, "mm", 5e-3),
    ("spare_1.3", lambda: _AISLE.d_clear - _full_budget(_AISLE.v_aisle), 102.57, "mm", 5e-3),
    ("reaction_1.3", lambda: _AISLE.v_aisle * _AISLE.tau_delay, 173.68, "mm", 5e-3),
    ("T_latest_1.3", lambda: (_AISLE.d_clear - _full_budget(_AISLE.v_aisle)) / _AISLE.v_aisle,
     78.9, "ms", 5e-3),
    ("prebrake_ceiling_1.5",
     lambda: (_AISLE.d_clear - _AISLE.delta_loc - _AISLE.delta_margin - _const_brake(_v(1.5))) / _v(1.5),
     258.33, "ms", 5e-3),
    ("renewal_to_onset", lambda: _AISLE.t_lease + _AISLE.t_tick + _AISLE.t_bus + _AISLE.t_brake_onset,
     82.0, "ms", 1e-9),
    ("missed_tick_1.3", lambda: _AISLE.v_aisle * _AISLE.t_tick, 1.3, "mm", 1e-9),
    ("occlusion_belief", lambda: (_AISLE.delta_margin - _AISLE.delta_loc) / _AISLE.v_human_approach,
     31.25, "ms", 1e-9),
    ("E_k", lambda: 0.5 * _RM.loaded_mass * _v(1.5) ** 2, 414.0, "J", 1e-9),
    ("intent_sweep", lambda: _weight_sweep(_RM.intent_model), 98.04, "ms", 5e-3),
    ("chunk_sweep", lambda: _weight_sweep(_RM.chunk_model), 0.700, "ms", 5e-4),
    ("F_cmd", lambda: _AISLE.f_latch_tripwire
     + _AISLE.k_latch * _AISLE.v_latch_approach * _AISLE.t_contact_response, 39.0, "N", 1e-9),
    ("yield_0.03", lambda: calc_contact_yield_deadline(
        _AISLE.f_latch_limit, _AISLE.k_latch, _AISLE.v_latch_approach), 8.333, "ms", 5e-4),
    ("yield_0.10", lambda: calc_contact_yield_deadline(
        _AISLE.f_latch_limit, _AISLE.k_latch, _AISLE.v_latch_high), 2.5, "ms", 1e-9),
    ("v_hand_ceiling", lambda: _AISLE.f_contact_criterion
     / (_AISLE.k_contact_human * _RM.arm_effective_contact_mass) ** 0.5, 0.1179, "m/s", 5e-5),
    ("F_handover", lambda: _AISLE.v_handover
     * (_AISLE.k_contact_human * _RM.arm_effective_contact_mass) ** 0.5, 42.4, "N", 5e-2),
    ("tau_ev_untracked", lambda: calc_target_evidence_horizon(
        _AISLE.grasp_tolerance, _AISLE.grasp_initial_error, _AISLE.v_conveyor), 60.0, "ms", 1e-9),
    ("tau_ev_tracked", lambda: (2 * _AISLE.grasp_tolerance / _AISLE.a_conveyor_slip) ** 0.5,
     244.95, "ms", 5e-3),
    ("rail_sag", lambda: _RM.control_rail_battery_low
     - _AISLE.i_rail_transient * _RM.control_rail_resistance, 17.3, "V", 1e-9),
    ("enforcer_cycles", lambda: _AISLE.enforcer_unloaded * _RM.permission_mcu.compute.clock_rate,
     54_000.0, "dimensionless", 1e-6),
]


@pytest.mark.parametrize(
    "getter, want, unit, tol",
    [g[1:] for g in _QUANTITY_GUARDS],
    ids=[g[0] for g in _QUANTITY_GUARDS],
)
def test_running_machine_quantity_guard(getter, want, unit, tol):
    assert getter().to(unit).magnitude == pytest.approx(want, abs=tol)


@pytest.mark.parametrize("row, want", [
    ("ch2", 742.5), ("ch4", 835.5), ("ch8", 912.9), ("ch11", 1194.15), ("ch12", 1234.15),
])
def test_running_machine_budget_rows_at_drive_limit(row, want):
    v = _RM.base.max_velocity
    assert v.to(_MPS).magnitude == pytest.approx(1.5)
    assert _running_totals(v)[row] == pytest.approx(want, abs=5e-3)


@pytest.mark.parametrize("name, got, want, tol", [
    ("ceiling", lambda: _ceiling(), 1.3898, 5e-5),
    ("body_only", lambda: calc_max_permitted_velocity(
        _AISLE.d_clear, _AISLE.t_brake_onset, _AISLE.a_brake,
        _AISLE.delta_loc, _AISLE.delta_margin).to(_MPS).magnitude, 1.9098, 5e-5),
    ("in_loop_takeover", lambda: _ceiling(_AISLE.t_takeover_in_loop), 1.2249, 5e-5),
    ("out_of_loop_takeover", lambda: _ceiling(_AISLE.t_takeover_out_of_loop), 0.3986, 5e-5),
    ("inspected_floor", lambda: _ceiling(decel=_AISLE.mu_inspected_floor * _G), 1.0947, 5e-5),
    ("walking_person", _walking_person_ceiling, 0.4620, 5e-5),
    ("mu_min", lambda: (_AISLE.a_brake / _G).to("dimensionless").magnitude, 0.204, 5e-4),
])
def test_running_machine_ceiling_guard(name, got, want, tol):
    assert got() == pytest.approx(want, abs=tol)


def test_running_machine_chosen_speeds_sit_under_their_ceilings():
    """Each chosen speed is below the derived ceiling it was chosen against."""
    assert _AISLE.v_aisle.to(_MPS).magnitude < _ceiling()
    assert _AISLE.v_takeover_in_loop.to(_MPS).magnitude < _ceiling(_AISLE.t_takeover_in_loop)
    assert _AISLE.v_takeover_out_of_loop.to(_MPS).magnitude < _ceiling(_AISLE.t_takeover_out_of_loop)
    hand_ceiling = (_AISLE.f_contact_criterion
                    / (_AISLE.k_contact_human * _RM.arm_effective_contact_mass) ** 0.5)
    assert _AISLE.v_handover < hand_ceiling
    assert _AISLE.enforcer_wcet < _AISLE.enforcer_deadline < _AISLE.t_tick
    assert _RM.control_rail_battery_low < _RM.control_rail_nominal


def test_lockstep_mcu_clock_rate():
    mcu = Hardware.Tiny.LockstepSafetyMCU_Reference
    assert mcu.compute.clock_rate.to("MHz").magnitude == pytest.approx(400.0)


def test_clock_rate_rejects_flops():
    from mlsysim.hardware.types import ComputeCore
    with pytest.raises(ValueError):
        ComputeCore(peak_flops="1 TFLOPs / s", clock_rate="1 TFLOPs / s")
    assert ComputeCore(peak_flops="1 TFLOPs / s").clock_rate is None
