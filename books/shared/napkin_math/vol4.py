"""Napkin Math, Volume IV: the action (one machine's command crossing the causal boundary).

Every number below is computed here. The machine is the book's running machine,
the warehouse mobile manipulator (Embodied.MobileManipulator.WarehouseMobileManipulator,
in the site Embodied.Scenario.WarehouseAisle), and worked examples reuse the
book's own mlsysim scenarios, so the sheet and the chapters agree. Many of those
inputs are illustrative or chosen in the book, not measured; the data sheet
(mlsysim.embodied.datasheet) records each one's category. Each result is
recomputed here from the printed formula and checked against the scenario's own
solver, so a drift in either shows up as a build error. Inputs chosen only for
this sheet are listed in SCENARIO and tagged with assumed() on the page.
"""
from __future__ import annotations

import math

from mlsysim import Embodied
from mlsysim.core.units import ureg
from mlsysim.embodied.datasheet import data_sheet_lookup
from mlsysim.engine.scenarios import Scenarios as SC
from mlsysim.physics.constants import COPPER_COEFFICIENT_REFERENCE_TEMPERATURE, COPPER_TEMPERATURE_COEFFICIENT
from mlsysim.physics.robotics import (calc_reduced_mass, calc_scurve_stop_distance,
                                      calc_transient_contact_speed_limit,
                                      calc_transient_impact_force)

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, sig, v)

EXTERNAL: dict = {}  # nothing outside the registry (g is pint's standard_gravity)
# Chosen only for this sheet: the speed axis of the stopping chart.
SCENARIO = dict(chart_speeds=(0.05, 2.0))


def _check(mine: float, book: float, what: str, rel: float = 1e-6) -> float:
    """Fail the build if the sheet's formula and the book's solver disagree."""
    if not math.isclose(mine, book, rel_tol=rel, abs_tol=1e-12):
        raise AssertionError(f"{what}: sheet {mine!r} vs book {book!r}")
    return mine


def _m(q, unit: str) -> float:
    return float(q.to(unit).magnitude)


def _dn(x: float, d: int = 0) -> str:
    """Round an upper bound down (safe direction)."""
    return f"{math.floor(x * 10 ** d + 1e-9) / 10 ** d:,.{d}f}"


def _up(x: float, d: int = 0) -> str:
    """Round a lower bound up (safe direction)."""
    return f"{math.ceil(x * 10 ** d - 1e-9) / 10 ** d:,.{d}f}"


def build() -> Sheet:
    S = SCENARIO
    A = TAG = assumed()
    RM = Embodied.MobileManipulator.WarehouseMobileManipulator
    SITE = Embodied.Scenario.WarehouseAisle
    DS = data_sheet_lookup()
    G = _m(1 * ureg.standard_gravity, "m/s**2")

    # ================================================================ stopping budget (Ch 1, 2, 12)
    v_drive = _m(RM.base.max_velocity, "m/s")
    v_aisle = _m(SITE.v_aisle, "m/s")
    a_br = _m(SITE.a_brake, "m/s**2")
    j_br = _m(SITE.j_brake, "m/s**3")
    d_clear = _m(SITE.d_clear, "m")
    d_loc, d_mar, e_trk = _m(SITE.delta_loc, "m"), _m(SITE.delta_margin, "m"), _m(SITE.eps_track, "m")
    m_load = _m(RM.loaded_mass, "kg")
    t_age = _m(SITE.perception_age, "s")
    t_lease, t_tick, t_bus, t_act = (_m(SITE.t_lease, "s"), _m(SITE.t_tick, "s"),
                                     _m(SITE.t_bus, "s"), _m(SITE.t_brake_onset, "s"))
    tau = _check(t_age + t_lease + t_tick + t_bus + t_act, _m(SITE.tau_delay, "s"), "tau_delay")
    t_backbone = _m(SITE.t_backbone, "s")

    full = SC.WarehouseBaseStopFullDelay.result
    d_delay = _check(v_drive * tau, _m(full.delay_travel, "m"), "v tau")
    d_brake = _check(v_drive ** 2 / (2 * a_br), _m(full.braking_travel, "m"), "v^2/2a")
    d_req = _check(d_delay + d_brake + d_loc + d_mar, _m(full.required_distance, "m"), "d_stop")
    spare = d_clear - d_req
    e_k = 0.5 * m_load * v_drive ** 2

    def v_ceiling(delay, room):
        return -a_br * delay + math.sqrt((a_br * delay) ** 2 + 2 * a_br * room)

    v_max_rigid = _check(v_ceiling(tau, d_clear - d_loc - d_mar), _m(full.speed_ceiling, "m/s"), "v_max rigid")
    fin = SC.WarehouseFinishedStopAtDriveLimit.result
    fin_aisle = SC.WarehouseFinishedStopAtAisleSpeed.result
    v_max_fin = _check(v_ceiling(tau + a_br / (2 * j_br), d_clear - d_loc - d_mar - e_trk),
                       _m(fin.speed_ceiling, "m/s"), "v_max finished")
    def v_br_s(vv):
        return _m(calc_scurve_stop_distance(vv * ureg("m/s"), a_br * ureg("m/s**2"), j_br * ureg("m/s**3")), "m")

    ramp = a_br / (2 * j_br)
    sfx_drive = _check(v_drive ** 2 / (2 * a_br) + v_drive * ramp, _m(fin.scurve_travel, "m"), "S-curve")
    _check(sfx_drive, v_br_s(v_drive), "S-curve fn")
    fin_req_drive = _m(fin.required_distance, "m")
    fin_req_aisle = _check(v_aisle * tau + v_aisle ** 2 / (2 * a_br) + v_aisle * ramp + d_loc + d_mar + e_trk,
                           _m(fin_aisle.required_distance, "m"), "finished at aisle")
    fin_spare_aisle = d_clear - fin_req_aisle

    # friction: dry-floor premise and the oil film (Ch 2, 18)
    mu_req = _check(a_br / G, float(DS["mu_req"].value), "mu required")
    mu_oil = float(SITE.mu_oil_film)
    a_oil = mu_oil * G
    oil = SC.WarehouseFinishedStopOilFilm.result
    _check(a_oil, _m(oil.deceleration, "m/s**2"), "a on oil")
    oil_req = _m(oil.required_distance, "m")
    oil_vmax = _m(oil.speed_ceiling, "m/s")

    # drive torque ceiling (Ch 2)
    brk = SC.WarehouseBaseBrakingCeilings.result
    r_wheel = _m(RM.drive_unit.wheel_diameter, "m") / 2
    n_wheel = RM.drive_wheel_count
    tq_peak = _m(RM.drive_unit.peak_torque, "N*m")
    m_eff_base = _m(brk.m_eff, "kg")
    a_torque = _check(n_wheel * tq_peak / r_wheel / m_eff_base, _m(brk.a_torque, "m/s**2"), "a torque")
    a_torque_rated = _m(brk.a_torque_rated, "m/s**2")

    # ================================================================ sensing (Ch 2, 5, 8, 9)
    cam = RM.nav_camera
    age_sc = SC.WarehouseNavCameraAge.result
    stages = [(s[0], _m(s[1], "ms")) for s in age_sc.stages]
    _check(sum(x for _, x in stages) / 1e3, t_age, "age stages")
    age_mm = t_age * v_drive * 1e3
    grasp = SC.WristCameraGraspAgeProfile.result
    r_tol, e0 = _m(SITE.grasp_tolerance, "mm"), _m(SITE.grasp_initial_error, "mm")
    v_conv = _m(SITE.v_conveyor, "m/s")
    t_fresh = _check((r_tol - e0) / 1e3 / v_conv, _m(grasp.deadline, "s"), "freshness")
    p50, p999 = _m(grasp.stage_sum_p50, "ms"), _m(SC.WristCameraGraspAgeProfile.system_p999, "ms")
    rs = SC.WarehouseNavCameraRollingShutter.result
    t_ro = _m(cam.readout_time, "s")
    skew = _check(v_drive * t_ro, _m(rs.skew, "m"), "skew")
    st = SC.WarehouseStereoRigDepthError
    sr = st.result
    b_st, f_px, sd = _m(st.baseline, "m"), st.focal_length_px, st.disparity_sigma_px
    sig_z = lambda z: z * z * sd / (f_px * b_st)  # noqa: E731
    z_near, z_far = _m(st.near_range, "m"), _m(st.far_range, "m")
    _check(sig_z(z_near), _m(sr.sigma_near, "m"), "sigma near")
    _check(sig_z(z_far), _m(sr.sigma_far, "m"), "sigma far")
    z_loc = _check(math.sqrt(d_loc * f_px * b_st / sd), _m(sr.range_at_loc_allowance, "m"), "stereo crossover")
    ing = SC.WarehouseTeleopIngestion
    nav_rate = _check(cam.resolution_width * cam.resolution_height * ing.rgb_bytes_per_pixel * _m(cam.frame_rate, "Hz"),
                      _m(ing.nav_rate, "B/s"), "nav rate")
    sata = _m(ing.slow_drive_bandwidth, "B/s")
    tot_rate, per_hour = _m(ing.total_rate, "B/s"), _m(ing.storage_per_hour, "TB")
    kin_rate = _m(ing.kinematic_rate, "B/s")
    fs = SC.WarehouseRackEndFreeSpace.result
    v_h = _m(SITE.v_human_approach, "m/s")
    side = _m(SITE.side_clearance, "m")
    t_inv_ahead = _check((d_clear - d_loc - d_mar) / v_h, _m(fs.horizon_ahead, "s"), "free-space ahead")
    t_inv_side = max(0.0, (side - d_loc - d_mar) / v_h)
    _check(t_inv_side, _m(fs.horizon_abreast, "s"), "free-space abreast")
    t_left = _m(fs.time_left_at_dispatch, "s")

    # ================================================================ brain and nervous system (Ch 3, 4)
    soc = RM.brain_soc
    bw_peak = _m(soc.memory.bandwidth, "B/s")
    eta_dram = float(RM.dram_efficiency)
    pt = SC.WarehouseProposerTraffic.result
    bw = _check(bw_peak * eta_dram, _m(pt.sustained_bandwidth, "B/s"), "sustained bw")
    n_par = _m(RM.intent_model.parameters, "param")
    w_int4 = _check(n_par * _m(RM.intent_weight_bytes_per_param, "B/param"), _m(pt.intent_weights_deployed, "B"), "int4 bytes")
    w_fp16 = 2 * n_par
    rate_int4 = _check(bw / w_int4, _m(pt.intent_sweep_rate_deployed, "Hz"), "sweep rate int4")
    rate_fp16 = _check(bw / w_fp16, _m(pt.intent_sweep_rate_fp16, "Hz"), "sweep rate fp16")
    ev = SC.IntentModelEvaluationInt4.result
    ev16 = SC.IntentModelEvaluationFp16.result
    o_pre = _m(ev.prefill_flops, "flop")
    r_ach = _m(ev.achieved_rate, "flop/s")
    peak16 = _m(soc.compute.precision_flops["fp16"], "flop/s")
    eta_c = float(RM.intent_compute_utilization)
    _check(peak16 * eta_c, r_ach, "eta R")
    t_pre = o_pre / r_ach
    t_sweep = w_int4 / bw
    s_out = RM.intent_model.action_dim
    t_eval = _check(max(t_pre, t_sweep) + (s_out - 1) * t_sweep, _m(ev.t_total, "s"), "t_infer int4")
    t_eval16 = _m(ev16.t_total, "s")
    f_intent = _m(RM.intent_rate, "Hz")
    cm = SC.ConveyorMugEvidenceHorizon.result
    t_chunk = _m(RM.chunk_step, "s") * 2
    _check(t_chunk, 1 / _m(RM.chunk_rate, "Hz"), "renewal period")
    dt_step = _m(RM.chunk_step, "s")
    t_inf_chunk = _m(RM.chunk_inference_latency, "s")
    h_min = _check(math.ceil((t_chunk + t_inf_chunk) / dt_step - 1e-9), cm.min_supply_setpoints, "h_min")
    a_slip = _m(SITE.a_conveyor_slip, "m/s**2")
    tau_ev = _check(math.sqrt(2 * (r_tol - e0) / 1e3 / a_slip), _m(cm.horizon, "s"), "evidence horizon")
    horizon = RM.chunk_horizon
    lease = SC.WarehouseChunkLease.result
    jitter = _m(SC.WarehouseChunkLease.renewal_jitter, "s")
    lease_floor = _check(t_chunk + jitter, _m(lease.lease_floor, "s"), "lease floor")
    lease_ceil = _check((d_clear - d_loc - d_mar - v_drive ** 2 / (2 * a_br)) / v_drive - t_tick - t_bus - t_act - t_age,
                        _m(lease.lease_ceiling_after_age, "s"), "lease ceiling")
    rta = SC.WarehousePermissionTickResponse.result
    c_i, b_i, i_i, d_i = (_m(rta.wcet, "us"), _m(rta.blocking, "us"), _m(rta.interference, "us"), _m(rta.deadline, "us"))
    r_i = _check(c_i + b_i + i_i, _m(rta.response, "us"), "RTA")
    can = SC.WarehouseServoCanFdOneBus.result
    frame_us = _m(can.frame_time, "us")
    occ = _check(can.frames * frame_us, _m(can.occupancy, "us"), "CAN occupancy")
    cycle_us = _m(SC.WarehouseServoCanFdOneBus.cycle, "us")

    # ================================================================ evidence (Ch 5, 6, 7)
    tc = SC.LatchTrialCampaign
    n_tr = tc.trials
    t_roll, t_auto, t_man = _m(tc.rollout, "s"), _m(tc.automated_reset, "s"), _m(tc.manual_reset, "s")
    wall_auto = _check(n_tr * (t_roll + t_auto), _m(tc.automated_time, "s"), "wall auto") / 86400
    wall_man = _check(n_tr * (t_roll + t_man), _m(tc.manual_time, "s"), "wall manual") / 86400
    wall_farm = _check(n_tr * (t_roll + t_man) / tc.farm_rigs / 86400, _m(tc.farm_manual_time, "d"), "wall farm")
    sim_min = _m(tc.simulated_time, "min")
    steps_trial = tc.decisions_per_trial
    _check(steps_trial, t_roll / _m(tc.decision_step, "s"), "steps per trial")
    s2 = steps_trial ** 2

    ec = SC.LatchEvaluationCampaign
    alpha = ec.alpha
    n_clean = ec.clean_runs
    p_low = _check(alpha ** (1 / n_clean), ec.lower_bound, "p_lower")
    ln_a = math.log(1 / alpha)
    n_999 = _check(math.ceil(ln_a / math.log(1 / ec.target)), ec.target_runs, "n for 0.999")
    cyc = _m(ec.cycle, "s")
    hours_999 = n_999 * cyc / 3600
    hz = SC.HazardExposureWall
    lam_list = [_m(q, "1/hour") for q in hz.rates]
    lam6 = lam_list[1]  # the book's 1e-6 per hour target
    t6 = ln_a / lam6
    rig_h = _m(hz.rig_time, "hour")
    rig_rate = _check(ln_a / rig_h, _m(hz.rig_rate, "1/hour"), "rig rate")
    rig_exp = math.floor(math.log10(rig_rate))
    dc = SC.WarehouseDiagnosticCoverageHundredFit.result
    cov = _check(1 - SC.WarehouseDiagnosticCoverageHundredFit.target / dc.raw_rate, dc.coverage, "coverage")
    inj = _check(math.ceil(ln_a / -math.log(cov)), dc.injections, "injections")

    # ================================================================ body (Ch 2, 12, 13)
    bus = SC.WarehouseDriveBusStop.result
    i_tr, r_bus = _m(SITE.i_rail_transient, "A"), _m(RM.drive_bus_resistance, "ohm")
    sag = _check(i_tr * r_bus, _m(bus.sag, "V"), "bus sag")
    v_low, v_uvlo = _m(RM.drive_bus_battery_low, "V"), _m(RM.drive_bus_uvlo, "V")
    _check(e_k, _m(bus.kinetic_energy, "J"), "kinetic energy")
    regen = _m(bus.regenerated, "J")
    regen_frac = SC.WarehouseDriveBusStop.regen_fraction
    c_link = _m(SC.WarehouseDriveBusStop.link_capacitance, "F")
    v_full = _m(RM.drive_bus_full_charge, "V")
    v_cap = _check(math.sqrt(v_full ** 2 + 2 * regen / c_link), _m(bus.capacitor_only_endpoint, "V"), "cap endpoint")
    v_clamp = _m(RM.drive_bus_clamp, "V")

    th = SC.PandaClassJointBurst
    thr = th.result
    r_th, c_th = _m(th.thermal_resistance, "K/W"), _m(th.thermal_capacitance, "J/K")
    tau_th = _check(r_th * c_th, _m(thr.time_constant, "s"), "tau_th")
    t_amb = _m(th.ambient, "degC")
    p_rated = _m(thr.rated_loss, "W")
    t_eq = _check(t_amb + p_rated * r_th, _m(thr.rated_equilibrium, "degC"), "rated equilibrium")
    p_burst = _m(thr.burst_loss, "W")
    t_ceiling, t_limit = _m(th.ceiling, "degC"), _m(th.limit, "degC")
    hor_cold, hor_hot = _m(thr.horizon_cold, "s"), _m(thr.horizon_hot, "s")
    _check(t_amb + p_burst * r_th * (1 - math.exp(-hor_cold / tau_th)), t_limit, "cold burst horizon", rel=1e-4)
    duty = thr.duty

    m_arm_eff = _m(RM.arm_effective_contact_mass, "kg")
    m_hand, k_hand = _m(SITE.m_contact_human, "kg"), _m(SITE.k_contact_human, "N/m")
    m_red = _check(1 / (1 / m_hand + 1 / m_arm_eff), _m(calc_reduced_mass(RM.arm_effective_contact_mass, SITE.m_contact_human), "kg"), "m_red")
    f_crit = _m(SITE.f_contact_criterion, "N")
    tcp = _m(RM.tcp_speed_limit, "m/s")
    f_tcp = _check(tcp * math.sqrt(k_hand * m_red),
                   _m(calc_transient_impact_force(RM.tcp_speed_limit, RM.arm_effective_contact_mass, SITE.k_contact_human,
                                                  human_mass=SITE.m_contact_human), "N"), "F at TCP limit")
    v_free = _check(f_crit / math.sqrt(k_hand * m_red),
                    _m(calc_transient_contact_speed_limit(SITE.f_contact_criterion, RM.arm_effective_contact_mass,
                                                          SITE.k_contact_human, human_mass=SITE.m_contact_human), "m/s"), "v free")
    v_clamped = _check(f_crit / math.sqrt(k_hand * m_arm_eff),
                       _m(calc_transient_contact_speed_limit(SITE.f_contact_criterion, RM.arm_effective_contact_mass,
                                                             SITE.k_contact_human), "m/s"), "v clamped")
    f_trip, k_latch = _m(SITE.f_latch_tripwire, "N"), _m(SITE.k_latch, "N/m")
    v_latch, t_c = _m(SITE.v_latch_approach, "m/s"), _m(SITE.t_contact_response, "s")
    f_cmd = f_trip + k_latch * v_latch * t_c
    f_lim = _m(SITE.f_latch_limit, "N")
    v_latch_hi = _m(SITE.v_latch_high, "m/s")
    f_cmd_hi = f_trip + k_latch * v_latch_hi * t_c

    # ================================================================ people (Ch 14, 17)
    t_perc, t_cog, t_mot = (_m(SITE.t_human_perceptual_slow, "s"), _m(SITE.t_human_cognitive_slow, "s"),
                            _m(SITE.t_human_motor_slow, "s"))
    t_vid, t_hs = _m(SITE.t_video_glass_to_glass, "s"), _m(SITE.t_handshake, "s")
    t_sig = _check(t_perc + t_cog + t_mot + t_vid + t_hs, _m(SITE.t_takeover_in_loop, "s"), "T_sigma")
    t_sig_out = _m(SITE.t_takeover_out_of_loop, "s")
    t_p90, t_console = _m(SITE.t_takeover_driving_p90, "s"), _m(SITE.t_takeover_transport, "s")
    _check(t_p90 + t_console, t_sig_out, "out-of-loop takeover")
    to = SC.WarehouseInLoopTakeoverAtAisleSpeed.result
    to_travel = _check(v_aisle * t_sig, _m(to.takeover_travel, "m"), "takeover travel")
    to_over_spare = _check(to_travel / fin_spare_aisle, to.takeover_over_spare, "takeover over spare")
    t_da = _m(SITE.t_dispatch_to_admission, "s")
    t_full = tau + t_da + t_sig + ramp
    d_fixed = d_loc + d_mar + e_trk
    v_human = _check(a_br * (math.sqrt(t_full ** 2 + 2 * (d_clear - d_fixed) / a_br) - t_full),
                     _m(SITE.v_takeover_in_loop_ceiling, "m/s"), "speed a human buys")
    q = SC.WarehouseRemoteAssistantQueue
    qr = q.result
    lam_q = _m(q.request_rate, "1/hour")
    t_s = _m(q.service_time, "hour")
    rho = _check(q.machines * lam_q * t_s, qr.utilization, "rho")
    w_q = _check(t_s * rho / (1 - rho) * 3600, _m(qr.mean_wait, "s"), "mean wait")
    p99_w = _m(qr.p99_wait, "s")
    q32 = SC.WarehouseRemoteAssistantQueueThirtyTwoMachines.result
    ds = SC.WarehouseGroupDetectionShortfall
    dsr = ds.result
    p_miss = ds.all_miss_probability

    def frames(p):
        return math.ceil(math.log(p_miss) / math.log(1 - p) - 1e-9)

    k_ref = _check(frames(ds.reference), dsr.reference_frames, "frames ref")
    row = dsr.rows[1]
    p_lo = row.per_frame_detection
    k_lo = _check(frames(p_lo), row.frames, "frames low")
    t_frame = _m(dsr.frame_period, "s")
    lost = _check((k_lo - k_ref) * t_frame * v_aisle, _m(row.distance_lost, "m"), "distance lost")
    v_imp = _m(row.impact_speed, "m/s")
    e_imp = _check(0.5 * m_load * v_imp ** 2, _m(row.impact_energy, "J"), "impact energy")

    # ---- audit additions: finished-budget lease, traction share, regen inputs, impact speed, rounding
    from decimal import Decimal, ROUND_HALF_UP

    def half_up(x, d):
        return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))

    tau_max_aisle = _check((d_clear - d_fixed - v_br_s(v_aisle)) / v_aisle, _m(fin_aisle.delay_ceiling, "s"), "tau_max finished 1.3")
    tau_max_drive = _check((d_clear - d_fixed - v_br_s(v_drive)) / v_drive, _m(fin.delay_ceiling, "s"), "tau_max finished 1.5")
    t_rest = t_tick + t_bus + t_act + t_age
    lease_ceil_aisle = tau_max_aisle - t_rest
    lease_ceil_drive = tau_max_drive - t_tick - t_bus - t_act - t_age
    phi = float(DS["load_share_max"].value)
    _check(phi, brk.drive_load_share, "driven-wheel share")
    mu_rated = _check(a_br / (G * phi), float(DS["mu_req_rated"].value), "mu at rated share", rel=1e-3)
    v_switch = _m(RM.drive_bus_switch_rating, "V")
    _check(regen, regen_frac * e_k, "regenerated energy")
    f_max = _check(1 / t_eval, _m(ev.f_max, "Hz"), "max intent rate")
    _check(math.sqrt(2 * a_br * lost), v_imp, "impact speed")
    cam_fps = _m(cam.frame_rate, "Hz")
    t_react = _m(SITE.t_human_simple_reaction, "s")
    t_react_slow = _m(SITE.t_human_simple_reaction_slow, "s")
    t_lat = _check(t_s * 3600 / (1 - rho), w_q + t_s * 3600, "T_lat = wait + service")
    p99_lat = math.log(100) * t_lat
    t_lat32 = _m(SC.WarehouseRemoteAssistantQueueThirtyTwoMachines.service_time, "s") / (1 - q32.utilization)
    oil_rigid = v_aisle * tau + v_aisle ** 2 / (2 * a_oil) + d_loc + d_mar
    rot_mass = _check(m_eff_base - m_load, _m(brk.rotating_mass, "kg"), "rotating mass")
    wrist_rate, ext_rate = _m(ing.wrist_rate, "B/s"), _m(ing.external_rate, "B/s")
    _check(nav_rate + wrist_rate + ext_rate + kin_rate, tot_rate, "stream sum")
    _check(tot_rate * 3600 / 1e12, per_hour, "TB per hour")
    short = lambda name, k=1: " ".join(name.split()[-k:])  # noqa: E731  (drop the vendor name)
    drive_force = _check(n_wheel * tq_peak / r_wheel, _m(brk.force, "N"), "drive braking force")
    wrist = RM.wrist_camera


    # ================================================================ pages 3-4 (Ch 2, 3, 7, 8, 9, 11, 12, 13, 15)
    GR = SC.PandaClassJointGearRatios
    grr = GR.result
    jt, mot = GR.joint, GR.motor
    k_tq = _m(mot.torque_constant, "N*m/A")
    r_j = _m(jt.internal_resistance, "ohm")
    j_in = _m(jt.rotor_inertia, "kg*m**2")
    eta_g = GR.gear_efficiency
    tau_j = _m(GR.output_torque, "N*m")
    k_mot = _check(k_tq / math.sqrt(r_j), _m(grr.motor_constant, "N*m/W**0.5"), "motor constant")
    gear_rows = []
    for nr in GR.ratios:
        tm = _check(tau_j / (eta_g * nr), _m(grr.motor_torque[nr], "N*m"), f"motor torque {nr}")
        pl = _check((tm / k_mot) ** 2, _m(grr.copper_loss[nr], "W"), f"copper loss {nr}")
        jr = _check(nr ** 2 * j_in, _m(grr.reflected_inertia[nr], "kg*m**2"), f"reflected inertia {nr}")
        gear_rows.append((nr, tm, pl, jr))
    n_joint = GR.ratios[0]
    tau_rated_j, tau_peak_j = _m(jt.rated_torque, "N*m"), _m(jt.peak_torque, "N*m")

    ST = SC.PandaClassElbowStall
    stl = ST.result
    rth_s = _m(ST.thermal_resistance, "K/W")
    tamb_s, tlim_s = _m(ST.ambient, "degC"), _m(ST.limit, "degC")
    alpha_cu = float(COPPER_TEMPERATURE_COEFFICIENT.to("1/K").magnitude)
    t0_cu = _m(COPPER_COEFFICIENT_REFERENCE_TEMPERATURE, "degC")
    i_stall = _check(tau_j / (eta_g * n_joint) / k_tq, _m(stl.stall_current, "A"), "stall current")
    t_wind = _m(stl.winding_temperature, "degC")
    p_max_s = _check((tlim_s - tamb_s) / rth_s, _m(stl.sustainable_loss, "W"), "sustainable loss")
    r_hot = _check(r_j * (1 + alpha_cu * (tlim_s - t0_cu)), _m(stl.hot_resistance, "ohm"), "hot resistance")
    i_max_s = _check(math.sqrt(p_max_s / r_hot), _m(stl.sustainable_current, "A"), "sustainable current")
    tau_hold_s = _check(i_max_s * k_tq * eta_g * n_joint, _m(stl.sustainable_hold, "N*m"), "sustainable hold")

    GS = SC.LoadedArmJointGravityStop
    j_gs, lev, w0 = _m(GS.inertia, "kg*m**2"), _m(GS.lever, "m"), _m(GS.joint_speed, "1/s")
    tq_gs = _m(GS.torque_limit, "N*m")
    bias_gs = _check(GS.bias_fraction * tq_gs, _m(GS.bias, "N*m"), "gravity bias")
    al_free = _check(tq_gs / j_gs, _m(GS.alpha_free, "1/s**2"), "alpha free")
    al_down = _check((tq_gs - bias_gs) / j_gs, _m(GS.alpha_descending, "1/s**2"), "alpha descending")
    d_free = _check(lev * w0 ** 2 / (2 * al_free), _m(GS.tool_stop_free, "m"), "tool stop free")
    d_down = _check(lev * w0 ** 2 / (2 * al_down), _m(GS.tool_stop_descending, "m"), "tool stop descending")

    SW = SC.WarehouseArmShoulderSwing
    swr = SW.result
    dq = _m(SW.swing, "rad")
    a_bind = _m(swr.binding_acceleration, "rad/s**2")
    t_bang = _check(2 * math.sqrt(dq / a_bind), _m(swr.bang_bang_time, "s"), "bang-bang time")
    w_peak = _check(a_bind * t_bang / 2, _m(swr.peak_speed, "rad/s"), "peak joint speed")
    w_lim = _m(RM.arm.joint_velocity_limits[1], "rad/s")
    t_jerk_sw = _m(swr.jerk_limited_time, "s")
    a_up, a_dn = _m(swr.accel_up, "rad/s**2"), _m(swr.accel_down, "rad/s**2")

    SB = SC.PandaClassJointSeamBridge
    sbr = SB.result
    dw, t_tk, t_bl = _m(SB.velocity_gap, "rad/s"), _m(SB.tick, "s"), _m(SB.bridge_duration, "s")
    j_ref = _m(sbr.reflected_inertia, "kg*m**2")
    tq_step = _check(j_ref * dw / t_tk, _m(sbr.step_torque, "N*m"), "seam step torque")
    tq_bridge = _check(1.5 * j_ref * dw / t_bl, _m(sbr.peak_torque, "N*m"), "bridge peak torque")

    TB = SC.ThrottledNavigationBackbone
    tbr = TB.result
    t_bb_thr = _check(t_backbone / TB.clock_ratio, _m(tbr.throttled, "s"), "throttled backbone")
    bb_travel = _check((t_bb_thr - t_backbone) * _m(TB.speed, "m/s"), _m(tbr.added_travel, "m"), "throttle travel")

    TH = SC.WarehouseSharedDieThermalThrottle
    thr2 = TH.result
    th_amb, th_theta = _m(TH.ambient, "degC"), _m(TH.theta, "K/W")
    p_quiet, p_loaded = _m(TH.quiet_power, "W"), _m(TH.loaded_power, "W")
    th_trip, th_clear, th_tau = _m(TH.trip, "degC"), _m(TH.clear, "degC"), _m(TH.time_constant, "s")
    tj_quiet = _check(th_amb + p_quiet * th_theta, _m(thr2.junction_quiet, "degC"), "junction quiet")
    tj_loaded = _check(th_amb + p_loaded * th_theta, _m(thr2.junction_loaded, "degC"), "junction loaded")
    t_trip = _check(-th_tau * math.log(1 - (th_trip - tj_quiet) / (tj_loaded - tj_quiet)), _m(thr2.time_to_trip, "s"), "time to trip")
    t_recov = _check(th_tau * math.log((th_trip - tj_quiet) / (th_clear - tj_quiet)), _m(thr2.recovery, "s"), "recovery")
    clk_ratio = _m(TH.nominal_clock, "Hz") / _m(TH.throttled_clock, "Hz")
    exec_thr = _check(c_i * clk_ratio, _m(thr2.execution_throttled, "us"), "throttled execution")
    resp_thr = _check(exec_thr + b_i + i_i, _m(thr2.response_throttled, "us"), "throttled response")
    ticks_over = _check(math.floor(t_recov / t_tick), thr2.ticks_exposed, "ticks exposed")

    chunk_b = _m(SITE.chunk_wire_bytes, "B")
    f_link = _m(SITE.board_link_clock, "Hz")
    t_link = _check(chunk_b * 8 / f_link, _m(SITE.t_board_link, "s"), "board link")
    t_deser, t_val, t_actv = _m(SITE.t_chunk_deserialize, "s"), _m(SITE.t_plan_validate, "s"), _m(SITE.t_plan_activate, "s")
    _check(t_inf_chunk + t_link + t_deser + t_val + t_actv + t_tick, t_da, "dispatch to admission")
    disp = SC.WarehouseFinishedStopWithDispatch.result
    disp_req = _check(fin_req_aisle + v_aisle * t_da, _m(disp.required_distance, "m"), "stop with dispatch")
    disp_spare = d_clear - disp_req

    p_tdp = _m(soc.tdp, "W")
    e_prop = p_tdp * t_eval
    p_parts = [("base", _m(RM.base.nominal_power, "W")), ("arm", _m(RM.arm.nominal_power, "W")), ("SoC", p_tdp),
               ("lidar", _m(RM.lidar.nominal_power, "W")),
               ("cameras", _m(cam.nominal_power, "W") + _m(RM.wrist_camera.nominal_power, "W")),
               ("MCU", _m(RM.permission_mcu.tdp, "W"))]
    p_total = sum(x for _, x in p_parts)

    HM = SC.WorkcellHostMapSizing
    hmr = HM.result
    ext = [_m(x, "m") for x in HM.extent]
    pitch = _m(HM.pitch, "m")
    vb = _m(HM.voxel_bytes, "B")
    v_ws = ext[0] * ext[1] * ext[2]
    m_dense = _check(v_ws / pitch ** 3 * vb, _m(hmr.dense_bytes, "B"), "dense map")
    a_surf, hw_band = _m(HM.surface_area, "m**2"), _m(HM.band_half_width, "m")
    m_band = _check(a_surf * 2 * hw_band / pitch ** 3 * vb, _m(hmr.hashed_bytes, "B"), "band map")

    CF = SC.ClearanceInsetConformalRank
    cfr = CF.result if hasattr(CF, "result") else CF
    k_conf = _check(math.ceil((CF.calibration_count + 1) * (1 - CF.alpha) - 1e-9), cfr.rank, "conformal rank")
    n_min_conf = _check(math.ceil(1 / CF.alpha - 1 - 1e-9), cfr.min_count, "conformal minimum count")

    MI = SC.GenericReplannerMissionExposure
    mir = MI.result
    k_ho = _check(round(_m(MI.mission_time, "s") * _m(MI.replan_rate, "Hz")), mir.handoffs, "handoffs")
    p_all_ok = _check((1 - MI.late_probability) ** k_ho, mir.all_on_time, "all on time", rel=1e-6)
    eps_ho = _check(MI.mission_budget / k_ho, mir.per_handoff_budget, "per-handoff budget")

    # audit-4 additions
    a_oil_rated = mu_oil * G * phi
    oil_req_rated = v_aisle * tau + _m(calc_scurve_stop_distance(v_aisle * ureg("m/s"), a_oil_rated * ureg("m/s**2"), j_br * ureg("m/s**3")), "m") + d_fixed
    taud = _m(RM.arm.joint_torque_rate_limit, "N*m/s")
    j_sw = _m(swr.effective_inertia, "kg*m**2")
    jerk_sw = _check(taud / j_sw, _m(swr.jerk_bound, "rad/s**3"), "swing jerk bound")
    i_cont, r_cont = _m(th.motor.rated_current, "A"), _m(th.joint.internal_resistance, "ohm")
    _check(i_cont ** 2 * r_cont, p_rated, "rated loss")
    p_cold_s = _m(stl.cold_loss, "W")
    gain_s = _check(p_cold_s * alpha_cu * rth_s, stl.feedback_gain, "stall feedback gain")
    const_s = _check(p_cold_s * rth_s * (1 + alpha_cu * (tamb_s - t0_cu)), _m(stl.feedback_constant, "K"), "stall feedback constant")
    _check(tamb_s + const_s / (1 - gain_s), t_wind, "stall winding temperature")
    hold_print = math.floor(tau_hold_s + 1e-9)
    t_stop_total = tau + _m(fin_aisle.stop_time, "s")
    tq_rated_wheel = _m(RM.drive_unit.rated_torque, "N*m")
    _check(n_wheel * tq_rated_wheel / r_wheel / m_eff_base, a_torque_rated, "rated torque decel")
    orders = math.log10(MI.late_probability / eps_ho)
    p_rail, t_hold = _m(RM.permission_rail_load, "W"), _m(RM.permission_rail_holdup, "s")
    PL = SC.LatchPairingLag
    pl_lag, pl_clr = _m(PL.lag, "s"), _m(PL.clearance, "m")
    pl_vt, pl_vf, pl_vg = _m(PL.transit_speed, "m/s"), _m(PL.fast_speed, "m/s"), _m(PL.guarded_speed, "m/s")
    pl_dt = _check(pl_vt * pl_lag, _m(PL.displacement_transit, "m"), "pairing transit")
    pl_df = _check(pl_vf * pl_lag, _m(PL.displacement_fast, "m"), "pairing fast")
    pl_dg = _check(pl_vg * pl_lag, _m(PL.displacement_guarded, "m"), "pairing guarded")
    _check(pl_clr / pl_vf, _m(PL.lag_ceiling_fast, "s"), "lag ceiling fast")

    # ================================================================ charts
    sp = Plot((0, 2.0), (0, 2.0), W=300, H=120, bottom=24, right=8, left=34)
    sp.grid([0, 0.5, 1.0, 1.5, 2.0], [0, 0.5, 1.0, 1.5, 2.0], yfmt=lambda y: f"{y:g}")
    lo_v, hi_v = S["chart_speeds"]
    vs = [lo_v + k * (hi_v - lo_v) / 60 for k in range(61)]
    rigid = lambda x: x * tau + x * x / (2 * a_br) + d_loc + d_mar  # noqa: E731
    finished = lambda x: x * tau + v_br_s(x) + d_fixed  # noqa: E731
    human = lambda x: x * (tau + t_da + t_sig) + v_br_s(x) + d_fixed  # noqa: E731
    sp.line([(x, rigid(x)) for x in vs], "line muted")
    sp.line([(x, finished(x)) for x in vs], "line")
    sp.line([(x, human(x)) for x in vs if human(x) < 2.05], "line cat3")
    sp.parts.append(f'<line x1="{sp.X(0):.1f}" x2="{sp.X(2.0):.1f}" y1="{sp.Y(d_clear):.1f}" y2="{sp.Y(d_clear):.1f}" class="ridge"/>')
    sp.text(sp.X(0.04), sp.Y(d_clear) - 4, f"clear {d_clear:.2f} m")
    sp.point(v_aisle, fin_req_aisle)
    sp.point(v_drive, fin_req_drive)
    sp.text(sp.X(1.98), sp.Y(finished(1.98)) - 8, "finished", "end")
    sp.text(sp.X(1.98), sp.Y(0.55), "rigid", "end", "lab ital")
    sp.text(sp.X(1.06), sp.Y(1.82), "+ takeover", "end")
    sp.axes("speed at detection v (m/s)", "meters to rest")

    ts_min = _m(q.service_time, "min")
    per_machine = _m(q.request_rate, "1/min") * ts_min
    qp = Plot((0, 40), (0, 75), W=300, H=110, bottom=24, right=8, left=34)
    qp.grid([0, 10, 20, 30, 40], [0, 25, 50, 75])
    for mult, cls in ((1.0, "line"), (math.log(100), "line cat3")):
        pts = []
        for k in range(0, 400):
            nm = k * 0.1
            r = nm * per_machine
            if r >= 0.99:
                break
            val = ts_min * mult / (1 - r)
            pts.append((nm, val))
            if val > 75:
                break
        qp.line(pts, cls)
    q_caps = [(nm, ts_min / (1 - nm * per_machine)) for nm in (q.machines, q32.machines)]
    for nm, tl in q_caps:
        qp.point(nm, tl)
        qp.point(nm, math.log(100) * tl)
    qp.text(qp.X(20) - 4, qp.Y(math.log(100) * q_caps[0][1]) - 6, "p99", "end", "lab ital")
    qp.text(qp.X(28), qp.Y(ts_min / (1 - 28 * per_machine)) - 6, "mean", "end", "lab ital")
    qp.axes("machines per assistant", "minutes")

    # thermal hangover: junction temperature rising to the trip, then cooling
    hp = Plot((0, 30), (40, 100), W=300, H=88, bottom=24, right=8, left=34)
    hp.grid([0, 10, 20, 30], [40, 60, 80, 100])
    rise = [(t / 4, tj_quiet + (tj_loaded - tj_quiet) * (1 - math.exp(-(t / 4) / th_tau))) for t in range(0, int(t_trip * 4) + 1)]
    rise.append((t_trip, th_trip))
    cool = [(t_trip + t / 4, tj_quiet + (th_trip - tj_quiet) * math.exp(-(t / 4) / th_tau)) for t in range(0, int((30 - t_trip) * 4) + 1)]
    hp.line(rise, "line")
    hp.line(cool, "line cat3")
    for yy in (th_trip, th_clear):
        hp.parts.append(f'<line x1="{hp.X(0):.1f}" x2="{hp.X(30):.1f}" y1="{hp.Y(yy):.1f}" y2="{hp.Y(yy):.1f}" class="ridge"/>')
    hp.text(hp.X(11), hp.Y(th_trip) - 3, f"trip {fmt(th_trip)} °C")
    hp.text(hp.X(0.5), hp.Y(th_clear) + 9, f"clear {fmt(th_clear)} °C")
    hp.text(hp.X(29.5), hp.Y(62), "load stops", "end", "lab ital")
    hp.axes("seconds of continuous inference", "junction °C")

    # union bound: chance every handoff is on time against mission length
    up = Plot((1, 1e4), (1e-14, 1), xlog=True, ylog=True, W=300, H=78, bottom=24, right=8, left=40)
    up.grid([1, 10, 100, 1000, 1e4], [1e-12, 1e-6, 1], xfmt=lambda x: f"{x:,.0f}",
            yfmt=lambda y: "1" if y == 1 else f"1e{round(math.log10(y))}")
    kk = [10 ** (k / 20) for k in range(0, 81)]
    up.line([(x, (1 - MI.late_probability) ** x) for x in kk if (1 - MI.late_probability) ** x > 1e-14], "line")
    up.point(k_ho, max(p_all_ok, 1e-14), f"{fmt(k_ho)} handoffs", "end", -6, -4)
    up.axes("handoffs in the mission, K", "P(all ok)")

    stage_short = {"exposure midpoint": "exp/2", "readout": "readout", "link transport": "link", "DMA and queue": "DMA",
                   "ISP": "ISP", "vision backbone": "backbone", "dispatch": "dispatch"}
    heads = ["stage"] + [stage_short.get(nm, nm) for nm, _ in stages] + ["<b>sum</b>"]
    row_ms = ["ms"] + [f"{ms:g}" for _, ms in stages] + [f"{t_age * 1e3:.1f}"]
    row_mm = ["mm"] + [f"{ms * v_drive:.1f}" for _, ms in stages] + [f"{age_mm:.1f}"]
    mm_up = lambda x: _up(x * 1e3)  # noqa: E731  (a hazard distance in mm, rounded up)

    # ================================================================ page 1
    band = (f'<div class="eqbig">{v("d", "stop")} = {v("v")}{v("τ", "delay")} + {frac(v("v", sup="2"), "2" + v("a", "brake"))} + {v("δ", "loc")} + {v("δ", "margin")} ≤ {v("D", "clear")}'
            f'<span class="lbl">travel while deciding + braking + allowances</span></div>'
            f'<div class="txt"><div><b>Rigid</b> stop: as written. <b>Finished</b> (Ch 12): + jerk ramp {v("v")}{v("a", "brake")}/2{v("j", "max")} ({v("j", "max")} = {j_br:.2f} m/s³), allowance δ.</div>'
            f'<div>{v("τ", "delay")} = {fmt(tau * 1e3, 1)} ms, {v("D", "clear")} = {fmt(d_clear, 2)} m, {fmt(v_drive, 1)} m/s: rigid {fmt(d_delay, 4)} + {fmt(d_brake, 4)} + {fmt(d_loc + d_mar, 2)} = '
            f'{n(_up(d_req, 3) + " m")}; finished {n(_up(fin_req_drive, 3) + " m", True)} &gt; {fmt(d_clear, 2)} m.</div></div>'
            f'<div class="side"><div><b>Speed ceiling</b> {v("v", "max")} (positive root)</div>'
            f'<div>−{v("a", "brake")}{v("τ", "delay")} + √(({v("a", "brake")}{v("τ", "delay")})² + 2{v("a", "brake")}({v("D", "clear")} − {v("δ", "loc")} − {v("δ", "margin")}))</div>'
            f'<div>rigid {n(_dn(v_max_rigid, 2) + " m/s")}; finished {n(_dn(v_max_fin, 2) + " m/s")}</div></div>')

    p1 = Page(
        question="Can it stop in time?",
        sub="Read left to right: how far the machine travels while it decides, how old its evidence is, and whether its brain and nerves keep up.",
        meta="the running machine: warehouse mobile manipulator",
        notation=[
            ("Allowances", f'{v("δ", "loc")} localization {fmt(d_loc * 1e3)} mm + {v("δ", "margin")} margin {fmt(d_mar * 1e3)} mm + {v("ε", "track")} tracking {fmt(e_trk * 1e3)} mm = δ = {fmt(d_fixed * 1e3)} mm'),
            ("Motion", f'{v("v")} speed · {v("d", "stop")} stopping distance · {v("τ", "delay")} pre-brake delay · {v("a", "brake")} credible deceleration · {v("j", "max")} jerk limit · '
                       f'{v("D", "clear")} clear distance · {v("t", "age")} observation age · {v("T", "lease")} chunk lease'),
        ],
        know=[
            Know(f"1 m/s = {fmt(_m(1 * ureg('m/s') * ureg.ms, 'mm'))} mm/ms", "speed × delay is distance"),
            Know(f"{G:.2f} m/s²", "g, standard gravity"),
            Know(f"{fmt(a_br, 1)} m/s²", f"≈ {a_br / G:.1f} g: credible braking, loaded base, dry floor"),
            Know(f"{fmt(v_h, 1)} m/s", "human approach speed (ISO 13855)"),
            Know(f"≈ {fmt(t_react * 1e3)} ms", f"human simple reaction; slow bound {fmt(t_react_slow * 1e3)} ms"),
            Know(f"{1e3 / cam_fps:.1f} ms", f"frame period at {fmt(cam_fps)} fps"),
            Know(f"{ln_a:.0f} / n", f"{ec.confidence:.0%} bound on failure rate after n clean runs"),
        ],
        band=band,
        cols=[
            Col("1 How far does it travel?", [
                Chart(sp.svg("Stopping distance versus speed"), caption=f"Dots: the {fmt(v_aisle, 1)} m/s aisle speed fits; the {fmt(v_drive, 1)} m/s drive limit does not."),
                Card("Delay is distance", [f'{v("d")} = {v("v")} · {v("t")}', f'{v("τ", "delay")} = {v("t", "age")} + {v("T", "lease")} + {v("T", "tick")} + {v("T", "bus")} + {v("T", "act")}'], chapter="Ch 1, 4",
                     ex=f'age {fmt(t_age * 1e3, 1)} + lease {fmt(t_lease * 1e3)} + tick {fmt(t_tick * 1e3)} + bus {fmt(t_bus * 1e3)} + brake onset {fmt(t_act * 1e3)} = {fmt(tau * 1e3, 1)} ms; '
                        f'× {fmt(v_drive, 1)} m/s = {n(mm_up(d_delay) + " mm")}.',
                     so=f'the learned vision backbone is {fmt(t_backbone / tau * 100)}% of the delay; the lease is the largest term.'),
                Card("Jerk-limited stop", f'{v("d", "suffix")} = {frac(v("v", sup="2"), "2" + v("a", "brake"))} + {v("v")} {frac(v("a", "brake"), "2" + v("j", "max"))}', chapter="Ch 11, 12",
                     ex=f'{v("d", "suffix")}: brake command to rest, for {v("v")} ≥ {v("a", "brake", "2")}/{v("j", "max")}. {fmt(v_drive, 1)} m/s: {fmt(d_brake, 4)} + {fmt(v_drive * ramp, 3)} = {n(_up(sfx_drive, 3) + " m")}.',
                     so=f'the ramp is worth ≈ {fmt(ramp * 1e3)} ms of extra delay.'),
            ], "Ch 1, 4, 11, 12"),
            Col("2 How old is what it sees?", [
                Table(heads, [row_ms, row_mm],
                      caption=f'{v("t", "age")} by stage, {cam.name} navigation camera, {fmt(cam_fps)} fps; mm traveled at {fmt(v_drive, 1)} m/s.'),
                Card("Freshness deadline", [f'{v("Δt", "fresh")} ≤ {v("e", "max")} / {v("v")}, &nbsp; {v("e", "max")} = {v("r", "tol")} − {v("e", "0")}'], chapter="Ch 2",
                     ex=f'{v("r", "tol")} grasp tolerance, {v("e", "0")} initial error. {A}Mug on a {fmt(v_conv, 1)} m/s belt: ({fmt(r_tol)} − {fmt(e0)}) mm / {fmt(v_conv, 1)} m/s = {n(_dn(t_fresh * 1e3) + " ms")}. '
                        f'{wrist.name} wrist camera: median {fmt(p50)} ms, p99.9 {n(fmt(p999, 1) + " ms", True)}.',
                     so='the tail age, not the median, decides whether the grasp lands.'),
                Card("Raw sensor data rate", 'bytes/s = width × height × bytes/pixel × fps', chapter="Ch 5",
                     ex=f'Nav camera, RGB: {cam.resolution_width} × {cam.resolution_height} × {ing.rgb_bytes_per_pixel} B × {fmt(cam_fps)}/s = {n(f"{nav_rate / 1e9:.2f} GB/s")}; '
                        f'+ wrist {wrist_rate / 1e9:.2f} + two RGB-D {ext_rate / 1e9:.2f} = {tot_rate / 1e9:.2f} GB/s → {per_hour:.1f} TB/h.',
                     so=f'one raw camera is {nav_rate / sata:.1f}× a {fmt(sata / 1e6)} MB/s SATA SSD; log compressed or crop.'),
                Card("Free space expires", f'{v("T", "inv")} = ({v("d", "free")} − {v("δ", "loc")} − {v("δ", "margin")}) / {v("v", "h")}', chapter="Ch 9",
                     ex=f'{v("d", "free")} depth seen empty, {v("v", "h")} = {fmt(v_h, 1)} m/s person. Ahead: ({fmt(d_clear, 2)} − {fmt(d_loc + d_mar, 2)}) m / {fmt(v_h, 1)} m/s = {n(_dn(t_inv_ahead * 1e3) + " ms")}. '
                        f'Beside the rack end ({fmt(side * 1e3)} mm): {n(fmt(t_inv_side * 1e3) + " ms", True)}.',
                     so='an occluded gap needs a slower approach, not a fresher frame.'),
            ], "Ch 2, 5, 9"),
            Col("3 Can the brain and nerves keep up?", [
                Card("A proposer's memory wall", [f'sweep = {v("M", "weights")} / BW, &nbsp; prefill = {v("O")} / ({v("η", "hw")} {v("R", "peak")})',
                                                  f'{v("t", "propose")} = prefill + ({v("S", "out")} − 1) · sweep'], chapter="Ch 3",
                     ex=f'{v("M", "weights")} weight bytes, BW sustained, {v("S", "out")} tokens out. Sweep {fmt(w_int4 / 1e9, 1)} GB / {fmt(bw / 1e9, 1)} GB/s = {t_sweep * 1e3:.1f} ms; '
                        f'prefill {o_pre / 1e12:.3f} TFLOP / ({A}{v("η", "hw")} {eta_c:.0%} × {fmt(peak16 / 1e12)} TFLOP/s) = {fmt(t_pre * 1e3)} ms; '
                        f'{fmt(t_pre * 1e3)} + {s_out - 1} × {t_sweep * 1e3:.1f} = {n(fmt(t_eval * 1e3) + " ms")} ⇒ ≤ {_dn(f_max, 2)} Hz.',
                     so=f'the book plans {fmt(f_intent)} Hz; FP16 ({fmt(w_fp16 / 1e9)} GB) would take {fmt(t_eval16 * 1e3)} ms.'),
                Card("Lease rule (finished model)", f'{v("T", "chunk")} + {v("T", "jitter")} &lt; {v("T", "lease")} ≤ {v("τ", "max")} − {v("T", "rest")}', chapter="Ch 4",
                     ex=f'{v("T", "chunk")} renewal period {fmt(t_chunk * 1e3)} ms, jitter {fmt(jitter * 1e3)} ms; {v("τ", "max")} = ({v("D", "clear")} − δ − {v("d", "suffix")})/{v("v")}; '
                        f'{v("T", "rest")} = tick + bus + onset + age = {fmt(t_rest * 1e3, 1)} ms. At {fmt(v_aisle, 1)} m/s: {fmt(lease_floor * 1e3)} &lt; {v("T", "lease")} = {n(fmt(t_lease * 1e3) + " ms")} ≤ {_dn(lease_ceil_aisle * 1e3)} ms: fits. '
                        f'At {fmt(v_drive, 1)} m/s the ceiling is {n(_dn(lease_ceil_drive * 1e3) + " ms", True)}: no lease fits.',
                     so='slow down; a shorter lease cannot buy back the speed.'),
                Card("Response time: execution + blocking + interference", f'{v("R", "i")} = {v("C", "i")} + {v("B", "i")} + {v("I", "i")} ≤ {v("D", "i")}', chapter="Ch 4",
                     ex=f'Permission tick: {A}{v("C", "i")} {fmt(c_i)} + {v("B", "i")} {fmt(b_i)} + {v("I", "i")} {fmt(i_i)} = {n(fmt(r_i) + " µs")} ≤ {v("D", "i")} {fmt(d_i)} µs.<br>'
                        f'Fieldbus: {SC.WarehouseServoCanFdOneBus.num_axes} axes on CAN FD, {can.frames} frames × {frame_us:.1f} µs = {n(fmt(occ) + " µs", True)} per {fmt(cycle_us)} µs cycle.',
                     so='an unbounded blocking term (a mutex without priority inheritance) voids the bound.'),
            ], "Ch 3, 4"),
        ],
    )

    # ================================================================ page 2
    p2 = Page(
        question="Has it earned permission?",
        sub="Left: how much evidence a claim needs. Middle: electrical, thermal, and contact limits. Right: how fast people answer.",
        meta=f"warehouse mobile manipulator · {RM.arm.name} arm",
        notation=[
            ("Shared", f'{v("α", "sig")} = 1 − confidence · {v("p")} success per run · {v("n")} clean runs · {v("T", "Σ")} takeover latency · {v("ρ")} utilization · '
                       f'{v("t", "s")} service time · {v("T", "lat")} time in system · motion symbols as on page 1'),
        ],
        cols=[
            Col("4 How much evidence?", [
                Card("Physical trials take wall-clock time", f'{v("T", "wall")} = {v("N")} ({v("t", "rollout")} + {v("t", "reset")}) / rigs', chapter="Ch 6",
                     ex=f'{v("N")} trials of {v("t", "rollout")} each, reset {v("t", "reset")}. {A}{fmt(n_tr)} × ({fmt(t_roll)} + {fmt(t_auto)}) s = {n(fmt(wall_auto) + " days")}; '
                        f'manual {fmt(t_man)} s resets: {n(fmt(wall_man) + " days")}; {tc.farm_rigs} rigs: {fmt(wall_farm)} days.',
                     so='resets, not rollouts, set the calendar: automate resets before adding rigs.'),
                Card("A clean record bounds success", [f'{v("p")} ≥ {v("α", "sig")}<sup>1/{v("n")}</sup>', f'{v("n")} = ⌈ln {v("α", "sig")} / ln {v("p")}⌉ ≈ {ln_a:.0f} / (1 − {v("p")})'], chapter="Ch 7",
                     ex=f'{A}{n_clean} clean runs, {ec.confidence:.0%}: {alpha:.2f}<sup>1/{n_clean}</sup> ⇒ {v("p")} ≥ {n(_dn(p_low, 3))}. For {v("p")} = {ec.target:g}: '
                        f'⌈ln {alpha:.2f} / ln {ec.target:g}⌉ = {n(fmt(n_999) + " runs")} × {fmt(cyc)} s = {fmt(hours_999, 1)} h.',
                     so='each extra nine costs ten times the runs; every aborted run stays in the count.'),
                Card("The exposure wall", f'failure-free hours ≥ ln(1/{v("α", "sig")}) / {v("λ")} ≈ {ln_a:.0f} / {v("λ")}', chapter="Ch 7",
                     ex=f'{v("λ")} hazard rate per hour. {A}{v("λ")} = 10<sup>{str(round(math.log10(lam6))).replace("-", "−")}</sup>/h: {ln_a:.2f} / 10<sup>{str(round(math.log10(lam6))).replace("-", "−")}</sup> = {n(f"{t6 / 1e6:.1f}M h")} ({fmt(t6 / rig_h)} years). '
                        f'One rig-year ({fmt(rig_h)} h) shows only {v("λ")} ≤ {_up(rig_rate / 10 ** rig_exp, 1)} × 10<sup>{str(rig_exp).replace("-", "−")}</sup>/h.',
                     so='rare hazards need fleets, simulation, or argument, not test rigs.'),
                Note("An evidence number:", 'every aborted and intercepted run counted? Runs independent and drawn from the target conditions? The 3/n rule on the right n? '
                     'A calibration measured under the conditions it will be used in?'),
                Note("Units:", 'm/s × ms = mm · N·m / m = N · W × s = J · K/W × W = K · 1 FIT = 10<sup>−9</sup>/h · GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B.'),
            ], "Ch 6, 7"),
            Col("5 Will the body hold?", [
                Card("Stopping energy and the bus", [f'{v("E", "k")} = ½ {v("m")} {v("v", sup="2")}, &nbsp; Δ{v("V")} = {v("I")} {v("R", "bus")}', f'{v("V", "final")} = √({v("V", "initial", "2")} + 2{v("E", "regen")} / {v("C", "bus")})'], chapter="Ch 2, 15",
                     ex=f'½ × {fmt(m_load)} kg × ({fmt(v_drive, 1)} m/s)² = {n(fmt(e_k) + " J")}. {A}{regen_frac:.0%} ({fmt(regen, 1)} J) into {fmt(c_link * 1e3)} mF from {fmt(v_full)} V: {n(_up(v_cap) + " V", True)} &gt; {fmt(v_clamp)} V clamp. '
                        f'{A}{fmt(i_tr)} A × {fmt(r_bus * 1e3)} mΩ = {n(fmt(sag, 1) + " V")}: {fmt(v_low)} → {fmt(v_low - sag, 1)} V, above the {fmt(v_uvlo)} V lockout.',
                     so='regeneration needs a brake resistor; sag is safe even on a depleted pack.'),
                Card("Motor heating", [f'{v("T", "eq")} = {v("T", "amb")} + {v("P", "loss")} {v("R", "th")}, &nbsp; {v("τ", "th")} = {v("R", "th")} {v("C", "th")}'], chapter="Ch 2",
                     ex=f'{v("P", "loss")} = {v("I", sup="2")} × winding resistance; {v("R", "th")}, {v("C", "th")} thermal resistance, capacity. Joint: {r_th:g} K/W × {fmt(c_th)} J/K = {n(fmt(tau_th) + " s")}; '
                        f'rated ({fmt(i_cont, 1)} A)² × {r_cont:g} Ω = {p_rated:.1f} W: {fmt(t_amb)} °C + {p_rated:.1f} × {r_th:g} = {t_eq:.1f} °C. {A}{th.burst_factor:g}× current → {th.burst_factor ** 2:g}× loss = {fmt(p_burst)} W: {fmt(t_limit)} °C in {n(_dn(hor_cold) + " s")} cold, {_dn(hor_hot)} s hot.',
                     so='bursts are budgets in seconds; resistance is taken constant here, though it rises with heat.'),
                Card("Contact force on a person", [f'{v("F")} = {v("v")} √({v("k", "H")} {v("m", "red")}), &nbsp; {v("m", "red")} = (1/{v("m", "H")} + 1/{v("m", "arm")})<sup>−1</sup>'], chapter="Ch 12",
                     ex=f'Hand {v("m", "H")} = {m_hand:g} kg, {v("k", "H")} = {fmt(k_hand / 1e3)} N/mm, arm {v("m", "arm")} = {fmt(m_arm_eff)} kg: at {fmt(tcp, 1)} m/s, {n(fmt(f_tcp) + " N", True)}. '
                        f'{A}{fmt(f_crit)} N limit ⇒ {n(_dn(v_free, 2) + " m/s")} free; pinned hand ({v("m", "red")} → {v("m", "arm")}): {n(_dn(v_clamped, 2) + " m/s")}.',
                     so='near people, contact sets the speed, not stopping.'),
                Note("A body number:", 'peak or rated torque? Resistance at the hot temperature? Regeneration with somewhere to go? Contact with a pinned or a free body part?'),
            ], "Ch 2, 12, 15"),
            Col("6 Who answers, and how fast?", [
                Card("Human takeover latency", f'{v("T", "Σ")} = reaction + video + handshake', chapter="Ch 14",
                     ex=f'Slow reaction {fmt(t_perc * 1e3)} + {fmt(t_cog * 1e3)} + {fmt(t_mot * 1e3)} ms + video {fmt(t_vid * 1e3)} + handshake {fmt(t_hs * 1e3)} = {n(fmt(t_sig * 1e3) + " ms")}; '
                        f'out of the loop: p90 {t_p90:g} s + {fmt(t_console * 1e3)} ms = {n(half_up(t_sig_out, 2) + " s")}. Attending {fmt(t_sig * 1e3)} ms × {fmt(v_aisle, 1)} m/s = {n(mm_up(to_travel) + " mm", True)}.',
                     so=f'{to_over_spare:.0f}× the {fmt(fin_spare_aisle * 1e3)} mm finished spare: a human cannot be the brake.'),
                Card("The speed a human buys (finished model)", f'in {v("v", "max")}: {v("τ", "delay")} → {v("τ", "delay")} + {v("Δ", "da")} + {v("T", "Σ")} + {v("a", "brake")}/2{v("j", "max")}', chapter="Ch 14",
                     ex=f'{v("Δ", "da")} dispatch to admission (chunk inference, link, validation, tick) = {fmt(t_da * 1e3, 1)} ms; room {v("D", "clear")} − δ = {fmt(d_clear - d_fixed, 2)} m ⇒ {n(_dn(v_human, 2) + " m/s", True)} &lt; {fmt(v_aisle, 1)} m/s aisle speed.',
                     so='slow down before asking for help.'),
                Card("One assistant, many machines (M/M/1)", [f'{v("ρ")} = machines × requests/h × {v("t", "s")}', f'{v("T", "lat")} = {v("t", "s")} / (1 − {v("ρ")}), &nbsp; p99 ≈ ln(100) · {v("T", "lat")}'], chapter="Ch 17",
                     ex=f'{A}{q.machines} × {lam_q:g}/h × {fmt(ts_min)} min = {rho:g}: {fmt(t_s * 3600)} s / (1 − {rho:g}) = {n(fmt(t_lat) + " s")}, p99 {fmt(p99_lat)} s. '
                        f'{q32.machines} machines: {fmt(q_caps[1][1])} and {fmt(math.log(100) * q_caps[1][1])} min (upper dots).',
                     so=f'keep {v("ρ")} ≤ {q.utilization_cap:g}; the book quotes the wait, {v("T", "lat")} − {v("t", "s")} = {fmt(w_q)} s.'),
                Chart(qp.svg("Time in system versus machines per assistant")),
            ], "Ch 14, 17"),
        ],
        bottom=Table(["The running machine (Ch 1)", "device", "rate", "timing", "capacity or limit", "power"], [
            ["Base", "warehouse AMR, 2 hub motors", f"{fmt(v_drive, 1)} m/s limit, {fmt(v_aisle, 1)} aisle", f"{fmt(a_br, 1)} m/s², {fmt(t_act * 1e3)} ms brake onset",
             f"{fmt(m_load)} kg loaded", f"{fmt(_m(RM.base.nominal_power, 'W'))} W"],
            ["Arm", f"{RM.arm.name}, {RM.arm.dofs} joints", f"{fmt(_m(RM.arm.control_frequency, 'Hz'))} Hz control", f"{fmt(tcp, 1)} m/s TCP limit",
             f"{fmt(_m(RM.arm.payload_capacity, 'kg'))} kg, {fmt(_m(RM.arm.max_torque, 'N*m'))} N·m", f"{fmt(_m(RM.arm.nominal_power, 'W'))} W"],
            ["Application processor", soc.name, f"{fmt(peak16 / 1e12)} TFLOP/s", f"{fmt(f_intent)} Hz intent, {fmt(_m(RM.chunk_rate, 'Hz'))} Hz chunks",
             f"{fmt(bw_peak / 1e9)} GB/s, {fmt(_m(soc.memory.capacity, 'GiB'))} GiB", f"{fmt(_m(soc.tdp_min, 'W'))}–{fmt(_m(soc.tdp, 'W'))} W"],
            ["Permission path", "lockstep safety MCU", f"{fmt(_m(RM.permission_rate, 'Hz'))} Hz tick", f"{fmt(t_lease * 1e3)} ms lease, {fmt(tau * 1e3, 1)} ms τ<sub>delay</sub>",
             f"{fmt(_m(RM.permission_mcu.memory.sram_capacity, 'MiB'))} MiB SRAM", f"{fmt(_m(RM.permission_mcu.tdp, 'W'))} W"],
            ["Sensors", f"{short(cam.name)} camera, {short(RM.lidar.name)} lidar, {short(RM.safety_scanner.name)} scanner", f"{fmt(cam_fps)} fps, {fmt(_m(RM.lidar.scan_rate, 'Hz'))} Hz",
             f"{fmt(t_age * 1e3, 1)} ms age; scanner {fmt(_m(RM.safety_scanner.response_time, 'ms'))} ms",
             f"{cam.resolution_width}×{cam.resolution_height}; {RM.lidar.channels} ch", f"{_m(cam.nominal_power, 'W'):g} W, {fmt(_m(RM.lidar.nominal_power, 'W'))} W"],
            ["Power", f"{fmt(_m(RM.drive_bus_nominal, 'V'), 1)} V drive bus", f"{fmt(v_low)}–{fmt(v_full)} V pack", f"{fmt(_m(RM.permission_rail_holdup, 's'))} s rail hold-up",
             f"lockout {fmt(v_uvlo)} V", f"{fmt(_m(RM.permission_rail_load, 'W'))} W rail"],
        ]),
    )

    # ================================================================ page 3
    gear_tab = Table(["ratio N", "motor N·m", "copper loss", "reflected J"],
                     [[f"{nr}:1", f"{tm:.2f}", f"{fmt(pl)} W", f"{jr:.4g} kg·m²"] for nr, tm, pl, jr in sorted(gear_rows)])
    p3 = Page(
        question="Will the hardware hold up?",
        sub="Left: what gears, held loads, and power cost. Middle: what limits speed. Right: frames, heat, and links against deadlines.",
        meta=f"Panda-class stand-in joint · {soc.name}",
        notation=[
            ("Joint", f'{v("N")} gear ratio · {v("η")} gear efficiency · {v("k", "t")} torque constant · {v("R")} winding resistance · {v("J")} inertia · {v("ω")} joint speed · '
                      f'{v("τ", "out")} joint torque · {v("R", "th")}, {v("τ", "th")} thermal resistance, time constant · page 1 symbols'),
        ],
        cols=[
            Col("7 What do joints and power cost?", [
                Card("Gear ratio: heat against inertia", [f'motor torque = {v("τ", "out")} / ({v("η")} {v("N")}), &nbsp; loss = (torque / {v("K", "m")})²', f'reflected inertia = {v("N", sup="2")} {v("J", "in")}'], chapter="Ch 2",
                     ex=f'{v("K", "m")} = {v("k", "t")}/√{v("R")} = {k_tq:g} / √{r_j:g} = {k_mot:.3f} N·m/√W; {v("J", "in")} rotor + gear input {j_in * 1e5:.2f} × 10<sup>−5</sup> kg·m². Holding {fmt(tau_j)} N·m, {v("η")} = {eta_g:g}:',
                     so='loss × reflected inertia is the same at every ratio: pick which one the task can afford.'),
                gear_tab,
                Card("Holding a load at stall", [f'{v("P", "max")} = ({v("T", "limit")} − {v("T", "amb")}) / {v("R", "th")}', f'{v("R", "hot")} = {v("R")}(1 + {v("α", "Cu")}({v("T", "limit")} − {fmt(t0_cu)} °C))',
                                                  f'{v("I", "max")} = √({v("P", "max")} / {v("R", "hot")}), &nbsp; hold = {v("I", "max")} {v("k", "t")} {v("η")} {v("N")}'], chapter="Ch 2",
                     ex=f'{A}{v("R", "th")} = {rth_s:g} K/W: ({fmt(tlim_s)} − {fmt(tamb_s)}) / {rth_s:g} = {p_max_s:.1f} W; {v("R", "hot")} = {r_hot:.3f} Ω ({v("α", "Cu")} = {alpha_cu * 100:.3f}%/K); '
                        f'{v("I", "max")} = {i_max_s:.2f} A ⇒ {n(fmt(hold_print) + " N·m")} of {fmt(tau_j)}. Holding {fmt(tau_j)} N·m ({i_stall:.1f} A, {fmt(p_cold_s, 1)} W cold): Δ{v("T")} = {const_s:.1f} / (1 − {gain_s:.3f}) ⇒ {n(fmt(t_wind) + " °C", True)} &gt; {fmt(tlim_s)}.',
                     so=f'a stalled joint derates {fmt((1 - hold_print / tau_j) * 100)}%; hot copper resists more.'),
                Card("Power draw and energy per proposal", ['P = Σ nominal draws, &nbsp; E = P<sub>SoC</sub> · t'], chapter="Ch 2, 3",
                     ex=f'{" + ".join(f"{nm} {fmt(x) if x >= 1 else f'{x:g}'}" for nm, x in p_parts)} = {n(fmt(p_total) + " W")}. One 7B proposal at full {fmt(p_tdp)} W TDP: {fmt(p_tdp)} × {t_eval:.3f} s ≤ {n(_up(e_prop, 1) + " J")}.',
                     so='motion, not inference, drains the pack. Battery runtime = pack Wh / draw; the registry gives no pack capacity.'),
                Note("Peak is not continuous:", f'the stand-in joint\'s {fmt(tau_peak_j)} N·m peak is {tau_peak_j / tau_rated_j:.1f}× its {fmt(tau_rated_j)} N·m rated torque; '
                     f'size continuous holds and duty cycles on rated, and bursts on peak with a thermal budget.'),
            ], "Ch 2, 3"),
            Col("8 How fast may it move?", [
                Card("Time-optimal swing of one joint", [f'{v("T", "swing")} = 2√(Δ{v("q")} / {v("α", "bind")}), &nbsp; {v("ω", "peak")} = {v("α", "bind")} {v("T", "swing")} / 2'], chapter="Ch 11",
                     ex=f'{v("α", "bind")}: the smallest of torque-limited ({a_up:.0f}, {a_dn:.0f}) and the joint\'s {a_bind:g} rad/s² limit. {A}Δ{v("q")} = {dq:g} rad: 2√({dq:g}/{a_bind:g}) = {n(fmt(t_bang * 1e3) + " ms")}; '
                        f'{v("ω", "peak")} = {w_peak:.2f} &lt; {w_lim:g} rad/s. The torque-rate limit {fmt(taud)} N·m/s ÷ {j_sw:.2f} kg·m² = {fmt(jerk_sw)} rad/s³ jerk adds {fmt((t_jerk_sw - t_bang) * 1e3)} ms.',
                     so='the datasheet acceleration limit, not the motor, binds this swing.'),
                Card("The seam spike", [f'one tick: {v("τ")} = {v("J", "ref")} Δ{v("ω")} / {v("T", "tick")}', f'smooth bridge: {v("τ", "peak")} = 1.5 {v("J", "ref")} Δ{v("ω")} / {v("T", "blend")}'], chapter="Ch 11",
                     ex=f'{v("J", "ref")} = {v("N", sup="2")}{v("J", "in")} = {j_ref:.4g} kg·m² (rotor only). {A}Δ{v("ω")} = {dw:g} rad/s between chunks: in {fmt(t_tk * 1e3)} ms, {n(f"{tq_step:.1f} N·m")}; '
                        f'over {fmt(t_bl * 1e3)} ms, {tq_bridge:.2f} N·m.',
                     so=f'blending seams cuts the torque spike {tq_step / tq_bridge:.0f}×; never hand over a velocity step.'),
                Card("Setpoints per chunk", [f'{v("h", "min")} = ⌈({v("T", "chunk")} + {v("t", "infer")}) / {v("Δt", "step")}⌉', f'{v("τ", "ev")} = √(2({v("r", "tol")} − {v("e", "0")}) / {v("a", "slip")})'], chapter="Ch 3",
                     ex=f'{v("T", "chunk")} {fmt(t_chunk * 1e3)}, {v("t", "infer")} {fmt(t_inf_chunk * 1e3)}, step {fmt(dt_step * 1e3)} ms: ⌈({fmt(t_chunk * 1e3)} + {fmt(t_inf_chunk * 1e3)}) / {fmt(dt_step * 1e3)}⌉ = {n(str(h_min))}. '
                        f'{A}Belt slip {v("a", "slip")} = {a_slip:g} m/s²: √(2 × {fmt(r_tol - e0)} mm / {a_slip:g}) = {n(_dn(tau_ev * 1e3) + " ms")} of evidence vs a {horizon} × {fmt(dt_step * 1e3)} = {fmt(horizon * dt_step * 1e3)} ms chunk.',
                     so='a chunk is a buffer, not permission.'),
                Card("Pairing lag: speed × clock error", [f'Δ{v("x")} = {v("v")} · Δ{v("t", "lag")} ≤ clearance'], chapter="Ch 5",
                     ex=f'Δ{v("t", "lag")}: time between a frame and the action logged with it. {A}{fmt(pl_lag * 1e3)} ms: transit {fmt(pl_vt, 1)} m/s → {n(fmt(pl_dt * 1e3, 1) + " mm", True)}; '
                        f'latch approach {pl_vf:g} m/s → {n(fmt(pl_df * 1e3, 1) + " mm", True)} vs a {fmt(pl_clr * 1e3)} mm clearance; {pl_vg:g} m/s → {fmt(pl_dg * 1e3, 2)} mm.',
                     so=f'the lag must stay under {fmt(pl_clr * 1e3)} mm / speed: {_dn(pl_clr / pl_vf * 1e3)} ms at {pl_vf:g} m/s.'),
                Note("Joint 2 limits:", f'{v("ω")} ≤ {w_lim:g} rad/s, acceleration ≤ {a_bind:g} rad/s², datasheet jerk ≤ {fmt(_m(RM.arm.joint_jerk_limits[1], "rad/s**3"))} rad/s³ (torque rate binds first); tool ≤ {fmt(tcp, 1)} m/s.'),
            ], "Ch 3, 5, 11"),
            Col("9 What breaks the deadlines?", [
                Card("The thermal hangover", [f'{v("T", "j")} = {v("T", "amb")} + {v("P")} {v("R", "th")}, &nbsp; rise 1 − e<sup>−<i>t</i>/τ<sub>th</sub></sup>, fall e<sup>−<i>t</i>/τ<sub>th</sub></sup>'], chapter="Ch 13",
                     ex=f'{A}Shared die, {v("R", "th")} = {th_theta:g} K/W, {v("τ", "th")} = {fmt(th_tau)} s. Idle {fmt(p_quiet)} W: {fmt(th_amb)} + {fmt(p_quiet)} × {th_theta:g} = {fmt(tj_quiet)} °C start; '
                        f'loaded {fmt(p_loaded)} W: {n(fmt(tj_loaded) + " °C", True)} &gt; {fmt(th_trip)} °C, tripped at {fmt(th_tau)} ln({fmt(tj_loaded - tj_quiet)}/{fmt(tj_loaded - th_trip)}) = {t_trip:.1f} s. '
                        f'Half clock: response {n(fmt(resp_thr) + " µs", True)} &gt; {fmt(d_i)}. Back to {fmt(th_clear)} °C: {fmt(th_tau)} ln({fmt(th_trip - tj_quiet)}/{fmt(th_clear - tj_quiet)}) = {t_recov:.3f} s = {fmt(ticks_over)} late ticks.',
                     so='keep the permission path off the inference die.'),
                Chart(hp.svg("Junction temperature under load and after")),
                Card("The board-link toll", [f'{v("t", "link")} = bytes × 8 / clock', f'{v("Δ", "da")} = {v("t", "infer")} + {v("t", "link")} + {v("t", "sw")} + {v("T", "tick")}'], chapter="Ch 13",
                     ex=f'{v("t", "sw")} = decode + validate + activate. {A}{fmt(chunk_b)} B over {fmt(f_link / 1e6)} MHz SPI = {t_link * 1e3:.2f} ms; {fmt(t_inf_chunk * 1e3)} + {t_link * 1e3:.2f} + {fmt(t_deser * 1e3)} + {fmt(t_val * 1e3)} + {fmt(t_actv * 1e3)} + {fmt(t_tick * 1e3)} = '
                        f'{v("Δ", "da")} {fmt(t_da * 1e3, 1)} ms × {fmt(v_aisle, 1)} m/s → finished stop + {v("Δ", "da")} {_up(disp_req, 3)} m, {n(_dn(disp_spare * 1e3) + " mm")} spare.',
                     so='validation, not the wire, dominates the toll.'),
                Card("Frame budget: throughput, not age", [f'frames/s ≤ 1 / slowest stage', f'stages in flight = ⌈{v("t", "stage")} / {v("T", "frame")}⌉', f'throttled: {v("t")}′ = {v("t")} / clock ratio'], chapter="Ch 2, 8",
                     ex=f'{v("T", "frame")} = 1/{fmt(cam_fps)} s = {1e3 / cam_fps:.1f} ms; backbone {fmt(t_backbone * 1e3)} ms: 1 / {t_backbone:g} s = {n(_dn(1 / t_backbone) + " frames/s", True)} &lt; {fmt(cam_fps)}; '
                        f'⌈{fmt(t_backbone * 1e3)} / {1e3 / cam_fps:.1f}⌉ = {math.ceil(t_backbone * cam_fps)} in flight. {A}At {TB.clock_ratio:.2f} clock: {fmt(t_bb_thr * 1e3)} ms, +{fmt((t_bb_thr - t_backbone) * 1e3)} ms × {_m(TB.speed, "m/s"):g} m/s = {mm_up(bb_travel)} mm blind.',
                     so='pipelining keeps up with the camera but does not shorten the observation age.'),
            ], "Ch 2, 8, 13"),
        ],
        bottom=Table(["Stand-in joint (Ch 2)", "gear ratio", "torque constant", "winding R", "input inertia", "rated / peak torque", "gear η"], [
            [f"{jt.name}", f"{fmt(n_joint)}:1", f"{k_tq:g} N·m/A", f"{r_j:g} Ω", f"{j_in * 1e5:.2f} × 10<sup>−5</sup> kg·m²", f"{fmt(tau_rated_j)} / {fmt(tau_peak_j)} N·m", f"{eta_g:g} (assumed)"],
        ]),
    )

    # ================================================================ page 4
    p4 = Page(
        question="Can the evidence be trusted?",
        sub="Left: what maps, depth, calibration, and detection cost. Middle: how small errors add up. Right: what can quietly steal the deceleration.",
        meta="world model · mission · release",
        notation=[
            ("Evidence", f'{v("α")} allowed miss rate · {v("n")} calibration or test count · {v("K")} handoffs per mission · {v("ε")} error per step or per handoff · page 1 and 2 symbols as before'),
        ],
        cols=[
            Col("10 What does perception cost?", [
                Card("Dense grid or a band around surfaces", ['dense = volume / pitch³ × bytes per voxel', 'band = surface × 2 · half-width / pitch³ × bytes'], chapter="Ch 9",
                     ex=f'{A}{fmt(ext[0])} × {fmt(ext[1])} × {fmt(ext[2])} m at {fmt(pitch * 100)} cm, {fmt(vb)} B/voxel: {fmt(v_ws)} m³ / 10<sup>−6</sup> m³ × {fmt(vb)} B = {n(f"{m_dense / 1e9:.1f} GB")}. '
                        f'{fmt(a_surf)} m² × 2 × {fmt(hw_band * 100)} cm band = {a_surf * 2 * hw_band:.1f} m³ → {n(fmt(m_band / 1e6, 1) + " MB")}.',
                     so=f'{m_dense / m_band:.0f}× less memory: most of a dense grid stores empty air.'),
                Card("A calibrated clearance inset", [f'rank {v("k")} = ⌈({v("n")} + 1)(1 − {v("α")})⌉, &nbsp; {v("n")} ≥ 1/{v("α")} − 1'], chapter="Ch 8",
                     ex=f'Sort {v("n")} measured position residuals; the inset is the {v("k")}-th smallest. {A}{v("n")} = {fmt(CF.calibration_count)}, {v("α")} = {CF.alpha:g}: '
                        f'⌈{fmt(CF.calibration_count + 1)} × {1 - CF.alpha:g}⌉ = {n(fmt(k_conf))}, the {CF.calibration_count - k_conf + 1}th largest. A {CF.alpha:.0%} claim needs {v("n")} ≥ {fmt(n_min_conf)}.',
                     so='the inset is a measured quantile, valid only under the conditions it was measured in.'),
                Card("Stereo depth error", f'{v("σ", "z")} = {v("Z", "c", "2")} {v("σ", "d")} / ({v("f")} {v("b")})', chapter="Ch 8",
                     ex=f'Range {v("Z", "c")}, baseline {v("b")}, focal {v("f")} (px), disparity noise {v("σ", "d")}. {A}{fmt(b_st * 100)} cm, {fmt(f_px)} px, {sd:g} px: {fmt(z_far)}² × {sd:g} / ({fmt(f_px)} × {b_st:g}) = '
                        f'{n(fmt(sig_z(z_far) * 1e3, 1) + " mm")} at {fmt(z_far)} m; it equals {v("δ", "loc")} at {_dn(z_loc, 1)} m.',
                     so=f'{fmt(z_far / z_near)}× the range costs {fmt((z_far / z_near) ** 2)}× the depth noise.'),
                Card("Frames to detect a person", [f'{v("N", "frames")} = ⌈ln {v("p", "miss")} / ln(1 − {v("p", "frame")})⌉', f'Δ{v("d")} = {v("v")} {v("T", "frame")} Δ{v("N", "frames")}, &nbsp; {v("v", "impact")} = √(2{v("a", "brake")} Δ{v("d")})'], chapter="Ch 17",
                     ex=f'{A}{v("p", "miss")} = {p_miss:g}: {v("p", "frame")} = {ds.reference:g} needs {k_ref} frames, {p_lo:g} needs {k_lo}. Δ{v("d")} = ({k_lo} − {k_ref}) × {1e3 / cam_fps:.1f} ms × {fmt(v_aisle, 1)} m/s = '
                        f'{mm_up(lost)} mm ⇒ {v("v", "impact")} = {n(f"{v_imp:.2f} m/s", True)} where the stop should end at rest; ½ × {fmt(m_load)} kg × {v_imp:.2f}² = {fmt(e_imp)} J.',
                     so='a group the detector sees less often is hit by a stop sized for others.'),
            ], "Ch 8, 9, 17"),
            Col("11 How do small errors add up?", [
                Card("Every handoff on time", [f'P(all on time) = (1 − {v("p", "late")})<sup>{v("K")}</sup>', f'per-handoff {v("ε")} ≤ {v("ε", "mission")} / {v("K")}'], chapter="Ch 11",
                     ex=f'{A}{fmt(_m(MI.mission_time, "min"))} min at {fmt(_m(MI.replan_rate, "Hz"))} Hz = {fmt(k_ho)} handoffs; p99 budget, {v("p", "late")} = {MI.late_probability:g}: '
                        f'{1 - MI.late_probability:g}<sup>{fmt(k_ho)}</sup> = {n(f"{p_all_ok:.0e}")}. Mission {v("ε", "mission")} = {MI.mission_budget:g}: {MI.mission_budget:g} / {fmt(k_ho)} = {n(f"{eps_ho:.1e}")}.',
                     so=f'design each handoff {orders:.1f} orders of magnitude deeper than p99.'),
                Chart(up.svg("Probability every handoff is on time versus handoff count")),
                Card("Errors compound over an episode", [f'imitation: {v("J")} − {v("J")}* ≤ {v("S", sup="2")}{v("ε")}', f'with expert corrections (DAgger): ≤ {v("u")} {v("S")} {v("ε", "N")} + O(1)'], chapter="Ch 5",
                     ex=f'{v("J")} task cost, {v("S")} steps, {v("ε")} step error; {v("ε", "N")} loss on the learner\'s own states, {v("u")} cost of one wrong action. '
                        f'{fmt(t_roll)} s at {fmt(_m(tc.decision_step, "ms"))} ms: {v("S")} = {steps_trial}, so {v("S", sup="2")} = {n(fmt(s2))}.',
                     so='collect corrections where the policy goes, not more demos.'),
                Card("Fault-injection count", ['coverage = 1 − target / raw rate', 'injections = ⌈ln(1/α<sub>sig</sub>) / −ln coverage⌉'], chapter="Ch 15",
                     ex=f'FIT = failures per 10<sup>9</sup> h. {A}Raw {fmt(dc.raw_rate)} FIT, target {fmt(SC.WarehouseDiagnosticCoverageHundredFit.target)} FIT: coverage {cov * 100:.2f}%; '
                        f'⌈{ln_a:.3f} / {-math.log(cov):.5f}⌉ = {n(fmt(inj) + " zero-miss injections")}.',
                     so='sample injections in proportion to the fault rates.'),
            ], "Ch 5, 11, 15"),
            Col("12 What can steal the deceleration?", [
                Card("Traction caps deceleration", f'{v("a", "brake")} ≤ {v("μ")} {v("g")} {v("φ", "drive")}', chapter="Ch 2, 18",
                     ex=f'{v("μ")} friction, {v("φ", "drive")} driven-wheel weight share. {fmt(a_br, 1)} m/s² needs {v("μ")} ≥ {_up(mu_req, 3)} ({v("φ", "drive")} = 1) or {_up(mu_rated, 3)} ({v("φ", "drive")} = {phi:.3f}). '
                        f'{A}Oil, {v("μ")} = {mu_oil:g}, finished stop at {fmt(v_aisle, 1)} m/s: {v("φ", "drive")} = {phi:.3f}: {a_oil_rated:.2f} m/s² ⇒ {n(_up(oil_req_rated, 2) + " m", True)} ({v("φ", "drive")} = 1, optimistic: {a_oil:.2f} m/s², {_up(oil_req, 2)} m).',
                     so='a floor contaminant more than doubles the stop; tip-over is separate.'),
                Card("Can the drives deliver it?", f'{v("a", "brake")} ≤ wheel force / ({v("m")} + rotating mass)', chapter="Ch 2",
                     ex=f'{n_wheel} × {fmt(tq_peak)} N·m / {fmt(r_wheel * 1e3)} mm = {fmt(drive_force)} N; ({fmt(m_load)} + {fmt(rot_mass)} rotating) kg = {fmt(m_eff_base)} kg ⇒ {n(_dn(a_torque, 2) + " m/s²")}.',
                     so=f'peak torque covers {fmt(a_br, 1)} m/s²; rated {fmt(tq_rated_wheel)} N·m per wheel gives only {_dn(a_torque_rated, 2)}.'),
                Card("Gravity steals braking torque", [f'deceleration = ({v("τ", "max")} − gravity bias) / inertia', f'tool travel = {v("L")} {v("ω", sup="2")} / (2 · deceleration)'], chapter="Ch 12",
                     ex=f'{A}Inertia {fmt(j_gs)} kg·m², lever {v("L")} = {lev:g} m, {v("ω")} = {w0:g} rad/s, gravity holds {GS.bias_fraction:.0%} of {fmt(tq_gs)} N·m. '
                        f'Free: {fmt(tq_gs)}/{fmt(j_gs)} = {al_free:g} rad/s² → {mm_up(d_free)} mm. Descending: {al_down:g} rad/s² → {n(mm_up(d_down) + " mm")}.',
                     so=f'descending needs {d_down / d_free:.0f}× the room; budget the worst pose.'),
                Note("A stopping number:", 'is every delay multiplied by speed, the human included? Is the deceleration backed by friction, drive torque, and the worst pose? '
                     'Rigid or finished model, and the same one throughout? Allowances δ included?'),
                Note("A timing number:", 'tail ages, not medians? Blocking and interference bounded? A throttle that stretches it? '
                     'A frame budget (throughput) confused with observation age (latency)?'),
            ], "Ch 2, 12, 18"),
        ],
    )
    p4.bottom = Table(["Event timeline at the aisle speed: finished stop + Δ<sub>da</sub> (Ch 4, 13)", "interval", "ms", f"mm at {fmt(v_aisle, 1)} m/s"], [
        ["capture → dispatch", "observation age t<sub>age</sub>", f"{t_age * 1e3:.1f}", mm_up(v_aisle * t_age)],
        ["dispatch → admission", "Δ<sub>da</sub>: inference, link, decode, validate, activate, tick", f"{t_da * 1e3:.1f}", mm_up(v_aisle * t_da)],
        ["admission → brake onset (stall)", "lease + tick + bus + onset", f"{(t_lease + t_tick + t_bus + t_act) * 1e3:.0f}", mm_up(v_aisle * (t_lease + t_tick + t_bus + t_act))],
        ["capture → brake onset", "sum", f"{_check((t_age + t_da + t_lease + t_tick + t_bus + t_act), _m(SITE.t_capture_to_brake_onset, 's'), 'capture to onset') * 1e3:.1f}",
         mm_up(v_aisle * (t_age + t_da + t_lease + t_tick + t_bus + t_act))],
    ])
    return Sheet(volume=4, book="Physical AI", scope="The Action", pages=[p1, p2, p3, p4],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · g = {G:g} m/s² · {TAG}= example input the book chooses · '
                         f'values from the running machine and the book\'s scenarios (mlsysim), many illustrative')
