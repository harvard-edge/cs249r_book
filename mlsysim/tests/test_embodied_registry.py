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
    ("chunk_sweep", lambda: _weight_sweep(_RM.chunk_model), 1.1204, "ms", 5e-4),  # ACT ~80M (2026-09-28)
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
    ("in_loop_takeover", lambda: _scurve_ceiling(_AISLE.t_takeover_in_loop), 0.8362, 5e-5),
    ("out_of_loop_takeover", lambda: _scurve_ceiling(_AISLE.t_takeover_out_of_loop), 0.1911, 5e-5),
    ("inspected_floor", lambda: _ceiling(decel=_AISLE.mu_inspected_floor * _G), 1.0947, 5e-5),
    ("walking_person", _walking_person_ceiling, 0.4620, 5e-5),
    ("mu_min", lambda: (_AISLE.a_brake / _G).to("dimensionless").magnitude, 0.204, 5e-4),
])
def test_running_machine_ceiling_guard(name, got, want, tol):
    assert got() == pytest.approx(want, abs=tol)


def test_running_machine_chosen_speeds_sit_under_their_ceilings():
    """Each chosen speed is below the derived ceiling it was chosen against."""
    assert _AISLE.v_aisle.to(_MPS).magnitude < _ceiling()
    assert _AISLE.v_takeover_in_loop.to(_MPS).magnitude < _scurve_ceiling(_AISLE.t_takeover_in_loop)
    assert _AISLE.v_takeover_out_of_loop.to(_MPS).magnitude < _scurve_ceiling(_AISLE.t_takeover_out_of_loop)
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


# ---------------------------------------------------------------------------
# Permission-path rail guard values (added 2026-09-22, round-3 D15, add-only).
# The rail must ride through from an undervoltage or feed-loss flag until the
# longest stop the enforcer can command inside the envelope has finished and
# the spring brakes have set at standstill.
# ---------------------------------------------------------------------------


def _inspected_floor_stop_duration():
    """C2 stop duration from the inspected-floor ceiling on mu_inspected_floor."""
    decel = _AISLE.mu_inspected_floor * _G
    return calc_c2_stop_suffix(_v(_ceiling(decel=decel)), decel)["duration"]


def _flag_to_brakes_set():
    """Tick + bus + brake onset, the inspected-floor C2 stop, then the slow spring-brake engage."""
    return (_AISLE.t_tick + _AISLE.t_bus + _AISLE.t_brake_onset
            + _inspected_floor_stop_duration() + _RM.spring_brake_engage_max)


_PERMISSION_RAIL_GUARDS = [
    # (id, getter, expected, unit, abs tolerance)
    ("holdup_capacitance", lambda: _RM.permission_rail_holdup_capacitance, 2.025, "F", 5e-4),
    ("holdup_energy", lambda: _RM.permission_rail_load * _RM.permission_rail_holdup, 140.0, "J", 1e-9),
    ("flag_to_onset", lambda: _AISLE.t_tick + _AISLE.t_bus + _AISLE.t_brake_onset, 22.0, "ms", 1e-9),
    ("inspected_floor_stop", _inspected_floor_stop_duration, 1395.35, "ms", 5e-3),
    ("flag_to_brakes_set", _flag_to_brakes_set, 1497.35, "ms", 5e-3),
    ("holdup_margin", lambda: _RM.permission_rail_holdup - _flag_to_brakes_set(), 502.65, "ms", 5e-3),
]


@pytest.mark.parametrize(
    "getter, want, unit, tol",
    [g[1:] for g in _PERMISSION_RAIL_GUARDS],
    ids=[g[0] for g in _PERMISSION_RAIL_GUARDS],
)
def test_permission_rail_guard(getter, want, unit, tol):
    assert getter().to(unit).magnitude == pytest.approx(want, abs=tol)


def test_permission_rail_rides_through_the_envelope_stop():
    """The hold-up covers the envelope's longest stop; an oil-film stop lies outside it."""
    assert _flag_to_brakes_set() < _RM.permission_rail_holdup
    assert _RM.spring_brake_engage_min < _RM.spring_brake_engage_max
    assert _RM.control_rail_dropout < _RM.control_rail_battery_low
    oil = _AISLE.mu_oil_film * _G
    oil_stop = calc_c2_stop_suffix(_v(_ceiling(decel=_AISLE.mu_inspected_floor * _G)), oil)["duration"]
    assert (_AISLE.t_tick + _AISLE.t_bus + _AISLE.t_brake_onset + oil_stop
            + _RM.spring_brake_engage_max) > _RM.permission_rail_holdup


# ---------------------------------------------------------------------------
# Intent-model latency (added 2026-09-28, vol4 decision M2). The latency is
# computed from a compute-bound prefill and bandwidth-bound decode sweeps, and
# the stated rate must fit it.

def test_intent_latency_is_computed_from_both_terms():
    b = _RM.intent_latency_breakdown
    assert b["prefill_bound"] == "compute"
    assert b["prefill_tokens"] == 288 and b["decode_steps"] == 6
    assert b["t_prefill"].to("ms").magnitude == pytest.approx(188.55, abs=5e-3)
    assert b["t_weight_sweep"].to("ms").magnitude == pytest.approx(24.51, abs=5e-3)
    assert b["t_decode"].to("ms").magnitude == pytest.approx(147.06, abs=5e-3)
    assert _RM.intent_inference_latency.to("ms").magnitude == pytest.approx(335.61, abs=5e-3)


def test_intent_rate_fits_computed_latency():
    period = (1 / _RM.intent_rate).to("ms")
    assert period >= _RM.intent_inference_latency
    # 2 Hz is the largest whole-hertz rate whose period covers the latency.
    assert (1 / (_RM.intent_rate + 1 * ureg.Hz)).to("ms") < _RM.intent_inference_latency


def test_intent_latency_agrees_with_published_measurement():
    # Jetson AI Lab, OpenVLA-7B INT4 on Jetson AGX Orin 64GB: 336 ms per frame.
    ref = _RM.intent_reference_latency
    assert ref.provenance.id == "prov:jetson-ai-lab-openvla-orin"
    rel = abs((_RM.intent_inference_latency - ref) / ref).to("dimensionless").magnitude
    assert rel < 0.05


def test_intent_rate_rejects_unachievable_rate():
    with pytest.raises(ValueError, match="shorter than the computed intent latency"):
        _RM.model_validate({**_RM.__dict__, "intent_rate": 5.0 * ureg.Hz})


# ---------------------------------------------------------------------------
# Remote-takeover response terms (added 2026-09-28, M4): sourced human times,
# design totals, S-curve ceilings, and the shared pre-trigger window.
# ---------------------------------------------------------------------------

def _scurve_ceiling(extra_delay=0 * ureg.millisecond):
    """Speed at which the finished budget with the jerk-limited stop fills d_clear."""
    from mlsysim.physics.robotics import calc_c2_stop_suffix, calc_scurve_max_permitted_velocity
    j = calc_c2_stop_suffix(_RM.base.max_velocity, _AISLE.a_brake)["peak_jerk"]
    return calc_scurve_max_permitted_velocity(
        _AISLE.d_clear, _AISLE.tau_delay + extra_delay, _AISLE.a_brake, j,
        _AISLE.delta_loc + _AISLE.eps_track, _AISLE.delta_margin,
    ).to(_MPS).magnitude


def test_takeover_human_terms_carry_provenance():
    for field, prov in [
        ("t_human_perceptual", "prov:card-moran-newell-1983-mhp"),
        ("t_human_cognitive_slow", "prov:card-moran-newell-1983-mhp"),
        ("t_brake_response_surprise", "prov:green-2000-perception-brake"),
        ("t_takeover_driving_p90", "prov:zhang-2019-takeover-meta-analysis"),
        ("t_video_glass_to_glass", "prov:schimpe-2022-teleop-glass-to-glass"),
    ]:
        assert getattr(_AISLE, field).provenance.id == prov


@pytest.mark.parametrize("getter, want, unit", [
    (lambda: _AISLE.t_human_simple_reaction, 240.0, "ms"),
    (lambda: _AISLE.t_human_simple_reaction_slow, 470.0, "ms"),
    (lambda: _AISLE.t_handshake, 31.0, "ms"),
    (lambda: _AISLE.t_takeover_transport, 135.0, "ms"),
    (lambda: _AISLE.t_takeover_in_loop, 605.0, "ms"),
    (lambda: _AISLE.t_takeover_out_of_loop, 4.465, "s"),
    (lambda: _AISLE.pre_trigger_window, 5.0, "s"),
], ids=["mhp_nominal", "mhp_slow", "handshake", "transport", "in_loop", "out_of_loop", "pre_trigger"])
def test_takeover_design_totals(getter, want, unit):
    assert getter().to(unit).magnitude == pytest.approx(want, abs=1e-9)


def test_takeover_response_ordering():
    """Expectation and engagement order the sourced human times."""
    assert _AISLE.t_human_simple_reaction < _AISLE.t_human_simple_reaction_slow
    assert _AISLE.t_human_simple_reaction_slow < _AISLE.t_brake_response_expected
    assert (_AISLE.t_brake_response_expected < _AISLE.t_brake_response_unexpected
            < _AISLE.t_brake_response_surprise)
    assert (_AISLE.t_takeover_driving_p50 < _AISLE.t_takeover_driving_mean
            < _AISLE.t_takeover_driving_p90 < _AISLE.t_takeover_driving_p99)
    assert _AISLE.pre_trigger_window >= _AISLE.t_takeover_out_of_loop


def test_takeover_redesign_rejects_the_former_speeds():
    """With sourced delays, the former 1.2 m/s and 0.3 m/s request speeds exceed their ceilings."""
    assert _scurve_ceiling(_AISLE.t_takeover_in_loop) < 1.2
    assert _scurve_ceiling(_AISLE.t_takeover_out_of_loop) < 0.3
    # Ceilings fall monotonically with the response percentile.
    ceilings = [_scurve_ceiling(_AISLE.t_takeover_transport + t) for t in (
        _AISLE.t_takeover_driving_p50, _AISLE.t_takeover_driving_mean,
        _AISLE.t_takeover_driving_p90, _AISLE.t_takeover_driving_p99)]
    assert ceilings == sorted(ceilings, reverse=True)


# ---------------------------------------------------------------------------
# Chunk-policy renewal on the setpoint grid (added 2026-09-28, M5). The
# renewal period must be a whole number of setpoint steps so that temporal
# ensembling indexes chunks by setpoint; ACT is the one deployed chunk policy.
# ---------------------------------------------------------------------------


def test_chunk_policy_is_act():
    assert _RM.chunk_model is Models.Embodied.ACT_ALOHA


def test_chunk_schedule_on_setpoint_grid():
    s = _RM.chunk_schedule
    assert s["renewal_period"].to("ms").magnitude == pytest.approx(50.0)
    assert _RM.chunk_step.to("ms").magnitude == pytest.approx(25.0)
    assert s["setpoints_per_renewal"] == 2
    assert s["open_loop_span"].to("ms").magnitude == pytest.approx(90.0)
    assert s["min_supply_setpoints"] == 4
    assert s["chunk_span"].to("ms").magnitude == pytest.approx(400.0)
    assert s["overlapping_chunks"] == 8
    assert s["renewal_slack"].to("ms").magnitude == pytest.approx(10.0)
    assert _RM.chunk_period == s["renewal_period"]


def test_chunk_schedule_keeps_lease_and_evidence_arguments():
    """The renewal still fits the 60 ms lease with 5 ms jitter; the chunk outlasts the conveyor horizon."""
    assert _RM.chunk_period + 5 * ureg.millisecond < _AISLE.t_lease
    # tracked evidence horizon under residual slip: sqrt(2 (r_tol - e_0) / a_slip)
    tau_ev = ((2 * (_AISLE.grasp_tolerance - _AISLE.grasp_initial_error)
               / _AISLE.a_conveyor_slip) ** 0.5).to("ms")
    assert _RM.chunk_schedule["open_loop_span"] < tau_ev < _RM.chunk_schedule["chunk_span"]


def test_chunk_schedule_rejects_off_grid_step():
    with pytest.raises(ValueError, match="not a whole number"):
        _RM.model_validate({**_RM.__dict__, "chunk_step": 20.0 * ureg.millisecond})
