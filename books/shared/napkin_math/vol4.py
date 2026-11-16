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

    # ================================================================ charts
    sp = Plot((0, 2.0), (0, 2.0), W=300, H=90, bottom=24, right=8, left=34)
    sp.grid([0, 0.5, 1.0, 1.5, 2.0], [0, 0.5, 1.0, 1.5, 2.0], yfmt=lambda y: f"{y:g}")
    lo_v, hi_v = S["chart_speeds"]
    vs = [lo_v + k * (hi_v - lo_v) / 60 for k in range(61)]
    rigid = lambda x: x * tau + x * x / (2 * a_br) + d_loc + d_mar  # noqa: E731
    finished = lambda x: x * tau + v_br_s(x) + d_fixed  # noqa: E731
    human = lambda x: x * (tau + t_da + t_sig) + v_br_s(x) + d_fixed  # noqa: E731
    sp.line([(x, rigid(x)) for x in vs], "line muted")
    sp.line([(x, finished(x)) for x in vs], "line")
    sp.line([(x, human(x)) for x in vs if human(x) < 2.05], "line warn")
    sp.parts.append(f'<line x1="{sp.X(0):.1f}" x2="{sp.X(2.0):.1f}" y1="{sp.Y(d_clear):.1f}" y2="{sp.Y(d_clear):.1f}" class="ridge"/>')
    sp.text(sp.X(0.04), sp.Y(d_clear) - 4, f"clear {d_clear:.2f} m")
    sp.point(v_aisle, fin_req_aisle)
    sp.point(v_drive, fin_req_drive)
    sp.text(sp.X(1.98), sp.Y(finished(1.98)) - 8, "finished", "end")
    sp.text(sp.X(1.98), sp.Y(0.55), "rigid", "end", "lab ital")
    sp.text(sp.X(1.06), sp.Y(1.82), "+ takeover", "end")
    sp.axes("speed at detection v (m/s)", "meters to rest")

    qs = SC.WarehouseRemoteAssistantQueue
    ts_min = _m(qs.service_time, "min")
    per_machine = _m(qs.request_rate, "1/min") * ts_min          # utilization one machine adds
    qp = Plot((0, 40), (0, 75), W=300, H=72, bottom=24, right=8, left=34)
    qp.grid([0, 10, 20, 30, 40], [0, 25, 50, 75])
    for mult, cls in ((1.0, "line"), (math.log(100), "line warn")):
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
    for nm in (q.machines, q32.machines):
        r = nm * per_machine
        qp.point(nm, ts_min / (1 - r))
        qp.point(nm, math.log(100) * ts_min / (1 - r))
    qp.text(qp.X(20) - 4, qp.Y(math.log(100) * ts_min / (1 - q.machines * per_machine)) - 6, "p99", "end", "lab ital")
    qp.text(qp.X(28), qp.Y(ts_min / (1 - 28 * per_machine)) - 6, "mean", "end", "lab ital")
    qp.axes("machines per assistant", "minutes")
    q_caps = [(nm, ts_min / (1 - nm * per_machine)) for nm in (q.machines, q32.machines)]

    # ================================================================ page 1
    allow = (f'{v("δ", "loc")} = {fmt(d_loc * 1e3)} mm, {v("δ", "margin")} = {fmt(d_mar * 1e3)} mm, {v("ε", "track")} = {fmt(e_trk * 1e3)} mm. '
             f'δ = all three = {fmt(d_fixed * 1e3)} mm · <b>rigid</b> stop: constant {v("a", "brake")} with {v("δ", "loc")} + {v("δ", "margin")} · <b>finished</b>: + jerk ramp {v("v")}{v("a", "brake")}/2{v("j", "max")} ({v("j", "max")} = {j_br:.2f} m/s³) with δ')
    band = (f'<div class="eqbig">{v("d", "stop")} = {v("v")}{v("τ", "delay")} + {frac(v("v", sup="2"), "2" + v("a", "brake"))} + {v("δ", "loc")} + {v("δ", "margin")} ≤ {v("D", "clear")}'
            f'<span class="lbl">travel while deciding + braking + allowances</span></div>'
            f'<div class="txt"><div><b>Stopping budget (Ch 1, 2, 12).</b> Running machine: {v("τ", "delay")} = {fmt(tau * 1e3, 1)} ms, {v("D", "clear")} = {fmt(d_clear, 2)} m.</div>'
            f'<div>At {fmt(v_drive, 1)} m/s: rigid {fmt(d_delay, 3)} + {fmt(d_brake, 4)} + {fmt(d_loc + d_mar, 2)} = {n(fmt(d_req, 3) + " m")}; finished {n(fmt(fin_req_drive, 3) + " m", True)}.</div></div>'
            f'<div class="side"><div><b>Speed ceiling</b> {v("v", "max")}, the positive root:</div>'
            f'<div>−{v("a", "brake")}{v("τ", "delay")} + √(({v("a", "brake")}{v("τ", "delay")})² + 2{v("a", "brake")}({v("D", "clear")} − {v("δ", "loc")} − {v("δ", "margin")}))</div>'
            f'<div>rigid {n(_dn(v_max_rigid, 2) + " m/s")}; finished ({v("τ", "delay")} + {v("a", "brake")}/2{v("j", "max")}, δ) {n(_dn(v_max_fin, 2) + " m/s")}</div></div>')

    stage_short = {"exposure midpoint": "exp/2", "readout": "readout", "link transport": "link", "DMA and queue": "DMA",
                   "ISP": "ISP", "vision backbone": "backbone", "dispatch": "dispatch"}
    heads = ["stage"] + [stage_short.get(nm, nm) for nm, _ in stages] + ["<b>sum</b>"]
    row_ms = ["ms"] + [f"{ms:g}" for _, ms in stages] + [f"{t_age * 1e3:.1f}"]
    row_mm = ["mm"] + [f"{ms * v_drive:.1f}" for _, ms in stages] + [f"{age_mm:.1f}"]

    p1 = Page(
        question="Can it stop in time?",
        sub="Read left to right: how far it travels while deciding, how old its evidence is, and whether its brain and nerves keep up.",
        meta="the running machine: warehouse mobile manipulator",
        notation=[
            ("Allowances", allow),
            ("Motion", f'{v("v")} speed · {v("d", "stop")} stopping distance · {v("τ", "delay")} pre-brake delay · {v("a", "brake")} credible deceleration · {v("j", "max")} jerk limit · '
                       f'{v("D", "clear")} clear distance · {v("v", "max")} speed ceiling · {v("d", "suffix")} brake command to rest'),
            ("Timing, task", f'{v("t", "age")} observation age · {v("T", "lease")} chunk lease · {v("T", "tick")}, {v("T", "bus")}, {v("T", "act")} tick, bus cycle, brake onset · '
                             f'{v("T", "chunk")} renewal period · {v("t", "infer")} inference time · {v("r", "tol")}, {v("e", "0")} task tolerance, initial error · {A}= example input'),
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
                     ex=f'{fmt(t_age * 1e3, 1)} + {fmt(t_lease * 1e3)} + {fmt(t_tick * 1e3)} + {fmt(t_bus * 1e3)} + {fmt(t_act * 1e3)} = {n(fmt(tau * 1e3, 1) + " ms")} → {n(fmt(d_delay * 1e3) + " mm")} at {fmt(v_drive, 1)} m/s; '
                        f'vision backbone {fmt(t_backbone / tau * 100)}% of it.'),
                Card("Jerk-limited stop", f'{v("d", "suffix")} = {frac(v("v", sup="2"), "2" + v("a", "brake"))} + {v("v")} {frac(v("a", "brake"), "2" + v("j", "max"))}', chapter="Ch 11, 12",
                     ex=f'For {v("v")} ≥ {v("a", "brake", "2")}/{v("j", "max")}. {fmt(v_drive, 1)} m/s: {fmt(d_brake, 3)} + {fmt(v_drive * ramp, 3)} = {n(fmt(sfx_drive, 3) + " m")}; the ramp adds ≈ {fmt(ramp * 1e3)} ms of delay.'),
                Card("Traction caps deceleration", f'{v("a", "brake")} ≤ {v("μ")} {v("g")} {v("φ", "drive")}', chapter="Ch 2, 18",
                     ex=f'{v("μ")} friction, {v("φ", "drive")} driven-wheel weight share. {fmt(a_br, 1)} m/s²: {v("μ")} ≥ {_up(mu_req, 3)} at {v("φ", "drive")} = 1, {_up(mu_rated, 3)} at rated {phi:.3f}; tip-over is separate. '
                        f'{A}Oil, {v("μ")} = {mu_oil:g}: {a_oil:.2f} m/s²; {fmt(v_aisle, 1)} m/s: rigid {fmt(oil_rigid, 2)}, finished {n(fmt(oil_req, 2) + " m", True)}.'),
            ], "Ch 1, 2, 12"),
            Col("2 How old is what it sees?", [
                Table(heads, [row_ms, row_mm],
                      caption=f'{v("t", "age")} by stage, {cam.name} navigation camera, {fmt(cam_fps)} fps; mm at {fmt(v_drive, 1)} m/s.'),
                Card("Freshness deadline", f'{v("Δt", "fresh")} ≤ {v("e", "max")} / {v("v")}, &nbsp; {v("e", "max")} = {v("r", "tol")} − {v("e", "0")}', chapter="Ch 2",
                     ex=f'{A}Mug on a {fmt(v_conv, 1)} m/s belt, {v("e", "max")} = {fmt(r_tol)} − {fmt(e0)} mm: {n(fmt(t_fresh * 1e3) + " ms")}. '
                        f'{wrist.name} wrist camera: median {fmt(p50)} ms passes, p99.9 {fmt(p999, 1)} ms fails.'),
                f'<p class="holds">Rolling shutter (Ch 8): {fmt(t_ro * 1e3, 1)} ms readout × {fmt(v_drive, 1)} m/s skews the frame by {n(fmt(skew * 1e3) + " mm")}, {fmt(skew / d_loc * 100)}% of {v("δ", "loc")}.</p>',
                Card("Stereo depth error", f'{v("σ", "z")} = {v("Z", "c", "2")} {v("σ", "d")} / ({v("f")} {v("b")})', chapter="Ch 8",
                     ex=f'Range {v("Z", "c")}, baseline {v("b")}, focal {v("f")} (px), disparity noise {v("σ", "d")}. {A}{fmt(b_st * 100)} cm, {fmt(f_px)} px, {sd:g} px: '
                        f'{fmt(sig_z(z_near) * 1e3, 1)} mm at {fmt(z_near)} m, {n(fmt(sig_z(z_far) * 1e3, 1) + " mm")} at {fmt(z_far)} m; = {v("δ", "loc")} at {fmt(z_loc, 1)} m.'),
                Card("Raw sensor data rate", 'bytes/s = width × height × bytes/pixel × fps', chapter="Ch 5",
                     ex=f'Nav camera, RGB: {cam.resolution_width}×{cam.resolution_height}×{ing.rgb_bytes_per_pixel}×{fmt(cam_fps)} = {n(f"{nav_rate / 1e9:.2f} GB/s")}. '
                        f'+ wrist {wrist_rate / 1e9:.2f} + two RGB-D {ext_rate / 1e9:.2f} GB/s = {tot_rate / 1e9:.2f} GB/s → {per_hour:.1f} TB/h.'),
                Card("Free space expires", f'{v("T", "inv")} = ({v("d", "free")} − {v("δ", "loc")} − {v("δ", "margin")}) / {v("v", "h")}', chapter="Ch 9",
                     ex=f'{v("d", "free")} seen-empty depth, {v("v", "h")} = {fmt(v_h, 1)} m/s. Ahead ({fmt(d_clear, 2)} m): {n(_dn(t_inv_ahead * 1e3) + " ms")}; '
                        f'beside the rack end ({fmt(side * 1e3)} mm): {n(fmt(t_inv_side * 1e3) + " ms", True)}.'),
            ], "Ch 2, 5, 8, 9"),
            Col("3 Can the brain and nerves keep up?", [
                Card("A proposer's memory wall", [f'weight sweeps/s ≤ BW / {v("M", "weights")}',
                                                  f'{v("t", "infer")} ≈ max({v("O")}/{v("η", "hw")}{v("R", "peak")}, sweep) + ({v("S", "out")} − 1) · sweep'], chapter="Ch 3",
                     ex=f'BW = {fmt(bw_peak / 1e9)} × {eta_dram:.2f} = {fmt(bw / 1e9, 1)} GB/s; INT4 7B, {fmt(w_int4 / 1e9, 1)} GB: sweep {t_sweep * 1e3:.1f} ms. {o_pre / 1e12:.3f} TFLOP prefill, '
                        f'{A}{v("η", "hw")} = {eta_c:.0%} of {fmt(peak16 / 1e12)} TFLOP/s: {fmt(t_pre * 1e3)} ms; {v("S", "out")} = {s_out}: {n(fmt(t_eval * 1e3) + " ms")}, ≤ {_dn(f_max, 2)} Hz (book: {fmt(f_intent)}).'),
                Card("Setpoints per chunk", [f'{v("h", "min")} = ⌈({v("T", "chunk")} + {v("t", "infer")}) / {v("Δt", "step")}⌉', f'{v("τ", "ev")} = √(2({v("r", "tol")} − {v("e", "0")}) / {v("a", "slip")})'], chapter="Ch 3",
                     ex=f'Step {v("Δt", "step")} = {fmt(dt_step * 1e3)} ms: ⌈({fmt(t_chunk * 1e3)} + {fmt(t_inf_chunk * 1e3)}) / {fmt(dt_step * 1e3)}⌉ = {n(str(h_min))}. {A}Belt slip {v("a", "slip")} = {a_slip:g} m/s²: '
                        f'{n(_dn(tau_ev * 1e3) + " ms")} of evidence, under a {horizon}-step chunk ({fmt(horizon * dt_step * 1e3)} ms).'),
                Card("Lease rule (finished model)", f'{v("T", "chunk")} + {v("T", "jitter")} &lt; {v("T", "lease")} ≤ {v("τ", "max")} − {v("T", "rest")}', chapter="Ch 4",
                     ex=f'{v("T", "rest")} = {v("T", "tick")} + {v("T", "bus")} + {v("T", "act")} + {v("t", "age")} = {fmt(t_rest * 1e3, 1)} ms; {v("τ", "max")} = ({v("D", "clear")} − δ − {v("d", "suffix")})/{v("v")}; jitter {fmt(jitter * 1e3)} ms. '
                        f'{fmt(v_aisle, 1)} m/s: {fmt(lease_floor * 1e3)} &lt; {n(fmt(t_lease * 1e3) + " ms")} ≤ {_dn(lease_ceil_aisle * 1e3)} ms; {fmt(v_drive, 1)} m/s: {n(_dn(lease_ceil_drive * 1e3) + " ms", True)}, none fits.'),
                Card("Response time: execution + blocking + interference", f'{v("R", "i")} = {v("C", "i")} + {v("B", "i")} + {v("I", "i")} ≤ {v("D", "i")}', chapter="Ch 4",
                     ex=f'Tick: {A}{fmt(c_i)} + {fmt(b_i)} + {fmt(i_i)} = {n(fmt(r_i) + " µs")} ≤ {fmt(d_i)} µs deadline {v("D", "i")}. '
                        f'{SC.WarehouseServoCanFdOneBus.num_axes} CAN FD axes: {can.frames} frames × {frame_us:.1f} µs = {n(fmt(occ) + " µs", True)} per {fmt(cycle_us)} µs.'),
            ], "Ch 3, 4"),
        ],
    )

    # ================================================================ page 2
    p2 = Page(
        question="Has it earned permission?",
        sub="How much evidence a claim needs, what the body's electrical, thermal, and contact limits allow, and how fast people answer.",
        meta=f"warehouse mobile manipulator · {RM.arm.name} arm",
        notation=[
            ("Shared", f'{v("α", "sig")} = 1 − confidence · {v("p")} success per run · {v("n")} clean runs · {v("T", "Σ")} takeover latency · {v("ρ")} utilization · '
                       f'{v("t", "s")} service time · {v("T", "lat")} time in system · {v("F")} force · {v("m")} mass · motion symbols as on page 1 · {A}= example input'),
        ],
        cols=[
            Col("4 How much evidence?", [
                Card("Physical trials take wall-clock time", f'{v("T", "wall")} = {v("N")} ({v("t", "rollout")} + {v("t", "reset")}) / rigs', chapter="Ch 6",
                     ex=f'{A}{v("N")} = {fmt(n_tr)} trials, {v("t", "rollout")} = {fmt(t_roll)} s: {v("t", "reset")} = {fmt(t_auto)} s → {fmt(wall_auto)} days; {fmt(t_man)} s → {n(fmt(wall_man) + " days", True)}; '
                        f'{tc.farm_rigs} rigs → {fmt(wall_farm)} days.'),
                Card("Errors compound over an episode", f'cloning ∝ {v("S", sup="2")}{v("ε")}; with corrections (DAgger) ∝ {v("S")}', chapter="Ch 5",
                     ex=f'{v("ε")} error per step, {v("S")} steps. {fmt(t_roll)} s at {fmt(_m(tc.decision_step, "ms"))} ms: {v("S")} = {steps_trial}, {v("S", sup="2")} = {n(fmt(s2), True)}.'),
                Card("A clean record bounds success", [f'{v("p")} ≥ {v("α", "sig")}<sup>1/{v("n")}</sup>', f'{v("n")} = ⌈ln {v("α", "sig")} / ln {v("p")}⌉ ≈ {ln_a:.0f} / (1 − {v("p")})'], chapter="Ch 7",
                     ex=f'Every run counts, aborts included. {A}{n_clean} clean, {ec.confidence:.0%}: {v("p")} ≥ {n(_dn(p_low, 3))}. {v("p")} = {ec.target:g}: {n(fmt(n_999) + " runs")}, '
                        f'× {fmt(cyc)} s = {fmt(hours_999, 1)} h. Each extra nine costs ten times the runs.'),
                Card("The exposure wall", f'failure-free hours ≥ ln(1/{v("α", "sig")}) / {v("λ")} ≈ {ln_a:.0f} / {v("λ")}', chapter="Ch 7",
                     ex=f'{v("λ")} hazard rate per hour. {A}10<sup>{str(round(math.log10(lam6))).replace("-", "−")}</sup>/h: {n(f"{t6 / 1e6:.1f}M h")} ({fmt(t6 / rig_h)} years). '
                        f'One rig-year ({fmt(rig_h)} h): {v("λ")} ≤ {_up(rig_rate / 10 ** rig_exp, 1)} × 10<sup>{str(rig_exp).replace("-", "−")}</sup>/h.'),
                Card("Fault-injection count", 'injections = ⌈ln(1/α<sub>sig</sub>) / −ln(coverage)⌉', chapter="Ch 15",
                     ex=f'Coverage = 1 − target / raw rate. {A}Raw {fmt(dc.raw_rate)} FIT, target {fmt(SC.WarehouseDiagnosticCoverageHundredFit.target)} FIT (1 FIT = 1 failure per 10<sup>9</sup> h): coverage {cov:.1%}, {n(fmt(inj) + " zero-miss injections")}.'),
            ], "Ch 5, 6, 7, 15"),
            Col("5 Will the body hold?", [
                Card("Stopping energy and the bus", [f'{v("E", "k")} = ½ {v("m")} {v("v", sup="2")}, &nbsp; Δ{v("V")} = {v("I")} {v("R", "bus")}', f'{v("V", "final")} = √({v("V", "initial", "2")} + 2{v("E", "regen")} / {v("C", "bus")})'], chapter="Ch 2, 15",
                     ex=f'{fmt(m_load)} kg, {fmt(v_drive, 1)} m/s: {n(fmt(e_k) + " J")}; {A}{regen_frac:.0%} ({fmt(regen, 1)} J) into {fmt(c_link * 1e3)} mF from {fmt(v_full)} V: '
                        f'{n(fmt(v_cap) + " V", True)}, over the {fmt(v_clamp)} V clamp, {fmt(v_switch)} V switches: needs a resistor. '
                        f'{A}{fmt(i_tr)} A × {fmt(r_bus * 1e3)} mΩ = {n(fmt(sag, 1) + " V")}: {fmt(v_low)} → {fmt(v_low - sag, 1)} V, {fmt(v_low - sag - v_uvlo, 1)} V over the {fmt(v_uvlo)} V UVLO.'),
                Card("Can the drives deliver it?", f'{v("a", "brake")} ≤ wheel force / ({v("m")} + rotating mass)', chapter="Ch 2",
                     ex=f'{n_wheel} × {fmt(tq_peak)} N·m / {fmt(r_wheel * 1e3)} mm = {fmt(drive_force)} N; {fmt(m_load)} + {fmt(rot_mass)} kg rotating = {fmt(m_eff_base)} kg: '
                        f'{n(_dn(a_torque, 2) + " m/s²")}; rated {_dn(a_torque_rated, 2)}.'),
                Card("Motor heating", [f'{v("T", "eq")} = {v("T", "amb")} + {v("P", "loss")} {v("R", "th")}, &nbsp; {v("τ", "th")} = {v("R", "th")} {v("C", "th")}'], chapter="Ch 2",
                     ex=f'{v("P", "loss")} = {v("I", sup="2")} × resistance (constant here; it rises with heat). '
                        f'Joint: {v("R", "th")} = {r_th:g} K/W, {v("C", "th")} = {fmt(c_th)} J/K: {n(fmt(tau_th) + " s")}; {fmt(t_amb)} °C + {p_rated:.1f} W × {r_th:g} = {fmt(t_eq)} °C. '
                        f'{A}{th.burst_factor:g}× current ({fmt(p_burst)} W): {fmt(t_limit)} °C in {n(fmt(hor_cold) + " s")} cold, {fmt(hor_hot)} s hot.'),
                Card("Contact force on a person", [f'{v("F")} = {v("v")} √({v("k", "H")} {v("m", "red")}), &nbsp; {v("m", "red")} = (1/{v("m", "H")} + 1/{v("m", "arm")})<sup>−1</sup>'], chapter="Ch 12",
                     ex=f'Hand {v("m", "H")} = {m_hand:g} kg, {v("k", "H")} = {fmt(k_hand / 1e3)} N/mm, arm {v("m", "arm")} = {fmt(m_arm_eff)} kg: {n(fmt(f_tcp) + " N", True)} at {fmt(tcp, 1)} m/s; '
                        f'{A}{fmt(f_crit)} N: {n(_dn(v_free, 2) + " m/s")}, clamped {n(_dn(v_clamped, 2) + " m/s")} ({v("m", "red")} → {v("m", "arm")}).'),
                Card("Force when the brake command issues", f'{v("F", "cmd")} = {v("F", "trip")} + {v("k", "latch")} {v("v")} {v("T", "c")}', chapter="Ch 10, 12",
                     ex=f'Trip {fmt(f_trip)} N + {fmt(k_latch / 1e3)} N/mm × {A}{fmt(v_latch * 1e3)} mm/s × loop {v("T", "c")} = {fmt(t_c * 1e3)} ms = {n(fmt(f_cmd) + " N")} of {fmt(f_lim)} N; '
                        f'{n(fmt(f_cmd_hi) + " N", True)} at {fmt(v_latch_hi * 1e3)} mm/s.'),
            ], "Ch 2, 10, 12, 15"),
            Col("6 Who answers, and how fast?", [
                Card("Human takeover latency", f'{v("T", "Σ")} = reaction + video + handshake', chapter="Ch 14",
                     ex=f'Slow reaction {fmt(t_perc * 1e3)} + {fmt(t_cog * 1e3)} + {fmt(t_mot * 1e3)} ms + video {fmt(t_vid * 1e3)} + handshake {fmt(t_hs * 1e3)} = {n(fmt(t_sig * 1e3) + " ms")}; '
                        f'out of the loop, p90 {t_p90:g} + {fmt(t_console * 1e3)} ms = {n(half_up(t_sig_out, 2) + " s", True)}. At {fmt(v_aisle, 1)} m/s: {n(fmt(to_travel * 1e3) + " mm", True)}, {to_over_spare:.0f}× the {fmt(fin_spare_aisle * 1e3)} mm finished spare.'),
                Card("The speed a human buys (finished model)", f'in {v("v", "max")}: {v("τ", "delay")} → {v("τ", "delay")} + {v("Δ", "da")} + {v("T", "Σ")} + {v("a", "brake")}/2{v("j", "max")}', chapter="Ch 14",
                     ex=f'Dispatch to admission {v("Δ", "da")} = {fmt(t_da * 1e3, 1)} ms, room {v("D", "clear")} − δ: {n(_dn(v_human, 2) + " m/s")} &lt; {fmt(v_aisle, 1)} m/s aisle speed. Slow down before asking.'),
                Card("One assistant, many machines (M/M/1)", [f'{v("ρ")} = machines × requests/h × {v("t", "s")}', f'{v("T", "lat")} = {v("t", "s")} / (1 − {v("ρ")}), &nbsp; p99 ≈ ln(100) · {v("T", "lat")}'], chapter="Ch 17",
                     ex=f'{A}{q.machines} × {lam_q:g}/h × {fmt(ts_min)} min: {v("ρ")} = {rho:g}, {v("T", "lat")} = {n(fmt(t_lat) + " s")}, p99 {fmt(p99_lat)} s. Dots: {q32.machines} machines, {fmt(q_caps[1][1])} and {fmt(math.log(100) * q_caps[1][1])} min. '
                        f'Book: wait {v("T", "lat")} − {v("t", "s")} = {fmt(w_q)} s.'),
                Chart(qp.svg("Time in system versus machines per assistant")),
                Card("Frames to detect a person", [f'{v("N", "frames")} = ⌈ln {v("p", "miss")} / ln(1 − {v("p", "frame")})⌉', f'Δ{v("d")} = {v("v")} {v("T", "frame")} Δ{v("N", "frames")}, &nbsp; {v("v", "impact")} = √(2{v("a", "brake")} Δ{v("d")})'], chapter="Ch 17",
                     ex=f'{A}{v("p", "miss")} = {p_miss:g}; {v("T", "frame")} = {1e3 / cam_fps:.1f} ms, {fmt(v_aisle, 1)} m/s. Detection per frame {v("p", "frame")} = {ds.reference:g}: {k_ref} frames; {p_lo:g}: {k_lo}. '
                        f'Δ{v("d")} = {fmt(lost * 1e3)} mm → {n(f"{v_imp:.2f} m/s", True)} ({fmt(e_imp)} J) at the person.'),
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
    return Sheet(volume=4, book="Physical AI", scope="The Action", pages=[p1, p2],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · g = {G:g} m/s² · {TAG}= example input the book chooses · '
                         f'values from the running machine and the book\'s scenarios (mlsysim), many illustrative')
