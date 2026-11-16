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
    short = lambda name, k=1: " ".join(name.split()[-k:])  # noqa: E731  (drop the vendor name)
    drive_force = _check(n_wheel * tq_peak / r_wheel, _m(brk.force, "N"), "drive braking force")
    wrist = RM.wrist_camera

    # ================================================================ charts
    sup = lambda k: "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(c)] if c.isdigit() else "⁻" for c in str(k))  # noqa: E731
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

    npl = Plot((1e-4, 0.6), (3, 1e5), xlog=True, ylog=True, W=300, H=72, bottom=24, right=8, left=34)
    npl.grid([1e-4, 1e-3, 1e-2, 1e-1], [10, 1000, 1e5],
             xfmt=lambda x: "10" + sup(round(math.log10(x))), yfmt=lambda y: "10" + sup(round(math.log10(y))))
    xs = [10 ** (-4 + k * (math.log10(0.5) + 4) / 50) for k in range(51)]
    npl.line([(x, ln_a / -math.log(1 - x)) for x in xs], "line")
    for tgt in (0.9, 0.99, 0.999):
        nn = math.ceil(ln_a / math.log(1 / tgt))
        npl.point(1 - tgt, nn, f"p = {tgt:g}: {fmt(nn)}", "start", 6, -4)
    npl.axes("allowed failure rate per run, 1 − p", "runs n")

    # ================================================================ page 1
    band = (f'<div class="eqbig">{v("d", "stop")} = {v("v")}{v("τ", "delay")} + {frac(v("v", sup="2"), "2" + v("a", "brake"))} + {v("δ", "loc")} + {v("δ", "margin")} ≤ {v("D", "clear")}'
            f'<span class="lbl">travel while deciding + braking + allowances</span></div>'
            f'<div class="txt"><div><b>Stopping budget (Ch 1, 2).</b> At {fmt(v_drive, 1)} m/s: {fmt(d_delay, 3)} + {fmt(d_brake, 4)} + {fmt(d_loc + d_mar, 2)} = {n(fmt(d_req, 3) + " m")} of {fmt(d_clear, 2)} m.</div>'
            f'<div><b>Finished (Ch 12):</b> + {v("v")}{v("a", "brake")}/2{v("j", "max")} + {v("ε", "track")} ({fmt(e_trk * 1e3)} mm) = {n(fmt(fin_req_drive, 3) + " m", True)}; '
            f'{fmt(v_aisle, 1)} m/s leaves {fmt(fin_spare_aisle * 1e3)} mm.</div></div>'
            f'<div class="side"><div><b>Speed ceiling</b> {v("v", "max")} (positive root)</div>'
            f'<div>−{v("a", "brake")}{v("τ", "delay")} + √(({v("a", "brake")}{v("τ", "delay")})² + 2{v("a", "brake")}({v("D", "clear")} − {v("δ", "loc")} − {v("δ", "margin")}))</div>'
            f'<div>{n(f"{v_max_rigid:.2f} m/s")} rigid; with {v("τ", "delay")} + {v("a", "brake")}/2{v("j", "max")} and δ: {n(f"{v_max_fin:.2f} m/s")}</div></div>')

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
            ("Motion", f'{v("v")} speed · {v("d", "stop")} stopping distance · {v("τ", "delay")} pre-brake delay · {v("a", "brake")} credible deceleration · {v("j", "max")} jerk limit · '
                       f'{v("D", "clear")} clear distance · {v("δ", "loc")}, {v("δ", "margin")}, {v("ε", "track")} allowances, δ = their sum · {v("v", "max")} speed ceiling'),
            ("Timing", f'{v("t", "age")} observation age · {v("T", "lease")} chunk lease · {v("T", "tick")}, {v("T", "bus")}, {v("T", "act")} tick, bus cycle, brake onset · '
                       f'{v("T", "chunk")}, {v("T", "jitter")} renewal period, jitter · {v("Δt", "step")} setpoint step · {v("t", "infer")} inference'),
            ("Sensing, compute", f'{v("r", "tol")}, {v("e", "0")} tolerance, initial error · {v("Z", "c")} range · {v("b")} baseline · {v("f")} focal length (px) · {v("σ", "d")} disparity noise · '
                                 f'{v("BW", "mem")} sustained bandwidth · {v("M", "weights")} weight bytes · {v("O")} FLOPs · {v("R", "peak")} peak FLOP/s · {v("η")} fraction sustained'),
        ],
        know=[
            Know(f"{fmt(v_drive, 1)} mm/ms", f"travel per ms at the {fmt(v_drive, 1)} m/s drive limit"),
            Know(f"{fmt(tau * 1e3, 1)} ms", "pre-brake delay of the running machine"),
            Know(f"{fmt(a_br, 1)} m/s²", f"credible deceleration, dry floor (μ ≥ {mu_req:.2f} to {mu_rated:.2f})"),
            Know(f"{fmt(t_age * 1e3, 1)} ms", "camera observation age at dispatch"),
            Know(f"{fmt(v_h, 1)} m/s", "human approach speed (ISO 13855)"),
            Know(f"{ln_a:.0f} / n", f"{ec.confidence:.0%} bound on failure rate after n clean runs (≈ ln {1 / alpha:.0f})"),
            Know(f"{fmt(t_sig * 1e3)} ms", "takeover by an attending operator"),
        ],
        band=band,
        cols=[
            Col("1 How far does it travel?", [
                Chart(sp.svg("Stopping distance versus speed"), caption=f"Dots: the {fmt(v_aisle, 1)} m/s aisle speed fits; the {fmt(v_drive, 1)} m/s drive limit does not."),
                Card("Delay is distance", [f'{v("d")} = {v("v")} · {v("t")}', f'{v("τ", "delay")} = {v("t", "age")} + {v("T", "lease")} + {v("T", "tick")} + {v("T", "bus")} + {v("T", "act")}'], chapter="Ch 1, 4",
                     ex=f'{fmt(t_age * 1e3, 1)} + {fmt(t_lease * 1e3)} + {fmt(t_tick * 1e3)} + {fmt(t_bus * 1e3)} + {fmt(t_act * 1e3)} = {fmt(tau * 1e3, 1)} ms → {n(fmt(d_delay * 1e3) + " mm")} at {fmt(v_drive, 1)} m/s; '
                        f'the vision backbone is {fmt(t_backbone / tau * 100)}%.'),
                Card("Jerk-limited stop", f'{v("d", "suffix")} = {frac(v("v", sup="2"), "2" + v("a", "brake"))} + {v("v")} {frac(v("a", "brake"), "2" + v("j", "max"))}', chapter="Ch 11, 12",
                     ex=f'Brake command to rest, for {v("v")} ≥ {v("a", "brake", "2")}/{v("j", "max")}. {j_br:.2f} m/s³, {fmt(v_drive, 1)} m/s: {fmt(d_brake, 3)} + {fmt(v_drive * ramp, 3)} = {n(fmt(sfx_drive, 3) + " m")}; '
                        f'the ramps act as {fmt(ramp * 1e3)} ms more delay.'),
                Card("Traction caps deceleration", f'{v("a", "brake")} ≤ {v("μ")} {v("g")} {v("φ", "drive")}', chapter="Ch 2, 18",
                     ex=f'{v("μ")} friction, {v("φ", "drive")} weight share on driven wheels. {fmt(a_br, 1)} m/s² needs {v("μ")} ≥ {mu_req:.3f} at {v("φ", "drive")} = 1, {mu_rated:.3f} at the rated {phi:.3f} '
                        f'(tip-over is separate). Oil, {v("μ")} = {mu_oil:g}: {a_oil:.2f} m/s²; {fmt(v_aisle, 1)} m/s needs {n(fmt(oil_req, 2) + " m", True)}.'),
            ], "Ch 1, 2, 12"),
            Col("2 How old is what it sees?", [
                Table(heads, [row_ms, row_mm],
                      caption=f'{v("t", "age")} by stage, {cam.name} navigation camera, {fmt(cam_fps)} fps; mm at {fmt(v_drive, 1)} m/s.'),
                Card("Freshness deadline", f'{v("Δt", "fresh")} ≤ {v("e", "max")} / {v("v")}, &nbsp; {v("e", "max")} = {v("r", "tol")} − {v("e", "0")}', chapter="Ch 2",
                     ex=f'Mug on a {fmt(v_conv, 1)} m/s belt, {v("e", "max")} = {fmt(r_tol - e0)} mm: {n(fmt(t_fresh * 1e3) + " ms")}. '
                        f'Grasp path ({wrist.name} wrist camera): median {fmt(p50)} ms passes, p99.9 {fmt(p999, 1)} ms fails.'),
                f'<p class="holds">Rolling shutter (Ch 8): {fmt(t_ro * 1e3, 1)} ms readout skews the frame by {n(fmt(skew * 1e3) + " mm")}, {fmt(skew / d_loc * 100)}% of {v("δ", "loc")}.</p>',
                Card("Stereo depth error", f'{v("σ", "z")} = {v("Z", "c", "2")} {v("σ", "d")} / ({v("f")} {v("b")})', chapter="Ch 8",
                     ex=f'{v("σ", "z")} depth noise; {fmt(b_st * 100)} cm, {fmt(f_px)} px, {sd:g} px: {fmt(sig_z(z_near) * 1e3, 1)} mm at {fmt(z_near)} m, '
                        f'{n(fmt(sig_z(z_far) * 1e3, 1) + " mm")} at {fmt(z_far)} m; it fills {v("δ", "loc")} by {fmt(z_loc, 1)} m.'),
                Card("Raw sensor data rate", 'bytes/s = width × height × bytes/pixel × fps', chapter="Ch 5",
                     ex=f'Navigation camera, RGB: {cam.resolution_width}×{cam.resolution_height}×{ing.rgb_bytes_per_pixel}×{fmt(cam_fps)} = {n(f"{nav_rate / 1e9:.2f} GB/s")}, '
                        f'{nav_rate / sata:.1f}× a SATA SSD. All teleoperation streams: {per_hour:.1f} TB/h.'),
                Card("Free space expires", f'{v("T", "inv")} = ({v("d", "free")} − {v("δ", "loc")} − {v("δ", "margin")}) / {v("v", "h")}', chapter="Ch 9",
                     ex=f'{v("d", "free")} seen-empty depth, {v("v", "h")} = {fmt(v_h, 1)} m/s. Ahead: {n(fmt(t_inv_ahead * 1e3) + " ms")}; '
                        f'beside the rack end ({fmt(side * 1e3)} mm): {n(fmt(t_inv_side * 1e3) + " ms", True)}.'),
            ], "Ch 2, 5, 8, 9"),
            Col("3 Can the brain and nerves keep up?", [
                Card("A proposer's memory wall", [f'weight sweeps/s ≤ {v("BW", "mem")} / {v("M", "weights")}',
                                                  f'{v("t", "infer")} ≈ max({v("O")}/{v("η")}{v("R", "peak")}, sweep) + ({v("S", "out")} − 1) · sweep'], chapter="Ch 3",
                     ex=f'{fmt(bw / 1e9, 1)} GB/s ÷ {fmt(w_int4 / 1e9, 1)} GB (INT4): {t_sweep * 1e3:.1f} ms. {o_pre / 1e12:.2f} TFLOP prefill at {r_ach / 1e12:g} TFLOP/s: '
                        f'{fmt(t_pre * 1e3)} ms, + {s_out - 1} sweeps ({v("S", "out")} = {s_out}): {n(fmt(t_eval * 1e3) + " ms")}, ≤ {f_max:.2f} Hz; book: {fmt(f_intent)} Hz.'),
                Card("Setpoints per chunk", [f'{v("h", "min")} = ⌈({v("T", "chunk")} + {v("t", "infer")}) / {v("Δt", "step")}⌉', f'{v("τ", "ev")} = √(2({v("r", "tol")} − {v("e", "0")}) / {v("a", "slip")})'], chapter="Ch 3",
                     ex=f'⌈({fmt(t_chunk * 1e3)} + {fmt(t_inf_chunk * 1e3)}) / {fmt(dt_step * 1e3)}⌉ = {n(str(h_min))}; {horizon} cover {fmt(horizon * dt_step * 1e3)} ms, '
                        f'but {a_slip:g} m/s² slip ends the evidence at {n(fmt(tau_ev * 1e3) + " ms")}: a buffer, not permission.'),
                Card("Lease rule", [f'{v("T", "chunk")} + {v("T", "jitter")} &lt; {v("T", "lease")} ≤ {v("τ", "max")} − {v("T", "rest")}'], chapter="Ch 4",
                     ex=f'{v("T", "rest")} = {v("T", "tick")} + {v("T", "bus")} + {v("T", "act")} + {v("t", "age")} = {fmt(t_rest * 1e3, 1)} ms; {v("τ", "max")} = ({v("D", "clear")} − δ − {v("d", "suffix")})/{v("v")} (finished budget). {fmt(v_aisle, 1)} m/s, {v("T", "jitter")} = {fmt(jitter * 1e3)} ms: '
                        f'{fmt(lease_floor * 1e3)} &lt; {n(fmt(t_lease * 1e3) + " ms")} ≤ {fmt(lease_ceil_aisle * 1e3)} ms. At {fmt(v_drive, 1)} m/s: {n(fmt(lease_ceil_drive * 1e3) + " ms", True)}, no lease fits.'),
                Card("Response time: execution + blocking + interference", f'{v("R", "i")} = {v("C", "i")} + {v("B", "i")} + {v("I", "i")} ≤ {v("D", "i")}', chapter="Ch 4",
                     ex=f'Tick: {fmt(c_i)} + {fmt(b_i)} + {fmt(i_i)} = {n(fmt(r_i) + " µs")} ≤ {fmt(d_i)} µs ({v("R", "i")} response, {v("D", "i")} deadline). '
                        f'{SC.WarehouseServoCanFdOneBus.num_axes} CAN FD axes: {n(fmt(occ) + " µs", True)} per {fmt(cycle_us)} µs.'),
            ], "Ch 3, 4"),
        ],
    )

    # ================================================================ page 2
    p2 = Page(
        question="Has it earned permission?",
        sub="How much evidence a claim needs, what the body's electrical, thermal, and contact limits allow, and how fast people answer.",
        meta=f"warehouse mobile manipulator · {RM.arm.name} arm",
        notation=[
            ("Evidence", f'{v("n")} clean runs · {v("α", "sig")} = 1 − confidence · {v("p")} success per run · {v("λ")} hazard rate per hour · '
                         f'{v("N")} trials · {v("t", "rollout")}, {v("t", "reset")} time per trial · {v("S")} steps per episode · {v("J")} task cost'),
            ("Body", f'{v("m")} mass · {v("E")} energy · {v("V")}, {v("I")} bus voltage, current · {v("R", "bus")}, {v("C", "bus")} bus resistance, capacitance · {v("P", "loss")} loss · '
                     f'{v("T", "eq")}, {v("T", "amb")} winding, ambient temperature · {v("R", "th")}, {v("C", "th")}, {v("τ", "th")} thermal resistance, capacity, time constant · {v("F")} force'),
            ("People", f'{v("T", "Σ")} takeover latency · {v("Δ", "da")} dispatch to admission · {v("ρ")} utilization · {v("N", "frames")} frames · {v("p", "frame")}, {v("p", "miss")} per-frame detection, allowed miss · page 1 motion symbols'),
        ],
        cols=[
            Col("4 How much evidence?", [
                Card("Physical trials take wall-clock time", f'{v("T", "wall")} = {v("N")} ({v("t", "rollout")} + {v("t", "reset")}) / rigs', chapter="Ch 6",
                     ex=f'{fmt(n_tr)} latch openings of {fmt(t_roll)} s: {fmt(wall_auto)} days with {fmt(t_auto)} s automatic resets, {n(fmt(wall_man) + " days", True)} with {fmt(t_man)} s manual ones, '
                        f'{fmt(wall_farm)} on {tc.farm_rigs} rigs.'),
                Card("Errors compound over an episode", [f'cloning: {v("J")} − {v("J")}* ≤ {v("S", sup="2")}{v("ε")}', f'DAgger: {v("J")} − {v("J")}* ≤ {v("u")} {v("S")} {v("ε", "N")} + O(1)'], chapter="Ch 5",
                     ex=f'{v("ε")} step error; {v("ε", "N")} loss on aggregated learner states; {v("u")} recoverability. {v("S")} = {steps_trial} ({fmt(t_roll)} s at {fmt(_m(tc.decision_step, "ms"))} ms): {v("S", sup="2")} = {n(fmt(s2), True)}.'),
                Card("A clean record bounds success", [f'{v("p")} ≥ {v("α", "sig")}<sup>1/{v("n")}</sup>', f'{v("n")} = ⌈ln {v("α", "sig")} / ln {v("p")}⌉ ≈ {ln_a:.0f} / (1 − {v("p")})'], chapter="Ch 7",
                     ex=f'Aborts count. {n_clean} clean runs at {ec.confidence:.0%}: {v("p")} ≥ {n(f"{p_low:.3f}")}; {ec.target:g} takes {n(fmt(n_999) + " runs")}, {fmt(hours_999, 1)} h at {fmt(cyc)} s.'),
                Chart(npl.svg("Clean runs needed versus allowed failure rate")),
                Card("The exposure wall", f'failure-free hours ≥ ln(1/{v("α", "sig")}) / {v("λ")} ≈ {ln_a:.0f} / {v("λ")}', chapter="Ch 7",
                     ex=f'{v("λ")} = 10<sup>{str(round(math.log10(lam6))).replace("-", "−")}</sup>/h: {n(f"{t6 / 1e6:.1f}M h")} ({fmt(t6 / rig_h)} years). A rig-year bounds {rig_rate / 10 ** rig_exp:.1f} × 10<sup>{str(rig_exp).replace("-", "−")}</sup>/h.'),
            ], "Ch 5, 6, 7"),
            Col("5 Will the body hold?", [
                Card("Stopping energy and the bus", [f'{v("E", "k")} = ½ {v("m")} {v("v", sup="2")}, &nbsp; Δ{v("V")} = {v("I")} {v("R", "bus")}', f'{v("V", "final")} = √({v("V", "initial", "2")} + 2{v("E", "regen")} / {v("C", "bus")})'], chapter="Ch 2, 15",
                     ex=f'{fmt(m_load)} kg at {fmt(v_drive, 1)} m/s: {n(fmt(e_k) + " J")}. Full {fmt(v_full)} V pack, {regen_frac:.0%} ({fmt(regen, 1)} J) into {fmt(c_link * 1e3)} mF: '
                        f'{n(fmt(v_cap) + " V", True)}, over the {fmt(v_clamp)} V clamp, {fmt(v_switch)} V switches: add a brake resistor. '
                        f'At {fmt(v_low)} V, {fmt(i_tr)} A × {fmt(r_bus * 1e3)} mΩ = {n(fmt(sag, 1) + " V")} sag leaves {fmt(v_low - sag - v_uvlo, 1)} V over the {fmt(v_uvlo)} V lockout.'),
                Card("Can the drives deliver it?", f'{v("a", "brake")} ≤ wheel force / mass with rotors', chapter="Ch 2",
                     ex=f'{n_wheel} × {fmt(tq_peak)} N·m / {fmt(r_wheel * 1e3)} mm = {fmt(drive_force)} N; / {fmt(m_eff_base)} kg = {n(f"{a_torque:.2f} m/s²")} (rated: {a_torque_rated:.2f}).'),
                Card("Motor heating", [f'{v("T", "eq")} = {v("T", "amb")} + {v("P", "loss")} {v("R", "th")}, &nbsp; {v("τ", "th")} = {v("R", "th")} {v("C", "th")}'], chapter="Ch 2",
                     ex=f'{v("P", "loss")} = {v("I", sup="2")} × resistance, taken constant (it rises with heat). Joint, {v("T", "amb")} = {fmt(t_amb)} °C: {n(fmt(tau_th) + " s")}; '
                        f'rated {p_rated:.1f} W → {fmt(t_eq)} °C; a {th.burst_factor:g}× burst ({fmt(p_burst)} W) hits {fmt(t_limit)} °C in {n(fmt(hor_cold) + " s")} cold, {fmt(hor_hot)} s hot.'),
                Card("Contact force on a person", [f'{v("F")} = {v("v")} √({v("k", "H")} {v("m", "red")}), &nbsp; {v("m", "red")} = (1/{v("m", "H")} + 1/{v("m", "arm")})<sup>−1</sup>'], chapter="Ch 12",
                     ex=f'Hand {v("m", "H")} = {m_hand:g} kg, {v("k", "H")} = {fmt(k_hand / 1e3)} N/mm; arm {v("m", "arm")} = {fmt(m_arm_eff)} kg: {n(fmt(f_tcp) + " N", True)} at {fmt(tcp, 1)} m/s. '
                        f'{fmt(f_crit)} N allows {n(f"{v_free:.2f} m/s")} free, {n(f"{v_clamped:.2f} m/s")} clamped.'),
                Card("Force when the brake command issues", f'{v("F", "cmd")} = {v("F", "trip")} + {v("k", "latch")} {v("v")} {v("T", "c")}', chapter="Ch 10, 12",
                     ex=f'Tripwire {v("F", "trip")} = {fmt(f_trip)} N, plate {fmt(k_latch / 1e3)} N/mm, {fmt(v_latch * 1e3)} mm/s, loop {v("T", "c")} = {fmt(t_c * 1e3)} ms: {n(fmt(f_cmd) + " N")} of a {fmt(f_lim)} N limit; '
                        f'{n(fmt(f_cmd_hi) + " N", True)} at {fmt(v_latch_hi * 1e3)} mm/s.'),
            ], "Ch 2, 10, 12, 15"),
            Col("6 Who answers, and how fast?", [
                Card("Human takeover latency", f'{v("T", "Σ")} = reaction + video + handshake', chapter="Ch 14",
                     ex=f'Perceive, decide, move ({fmt(t_perc * 1e3)} + {fmt(t_cog * 1e3)} + {fmt(t_mot * 1e3)} ms, slow bounds) + {fmt(t_vid * 1e3)} + {fmt(t_hs * 1e3)} = {n(fmt(t_sig * 1e3) + " ms")} attending; '
                        f'{n(half_up(t_sig_out, 2) + " s", True)} out of the loop. At {fmt(v_aisle, 1)} m/s attending alone covers {n(fmt(to_travel * 1e3) + " mm", True)}, '
                        f'{to_over_spare:.0f}× the {fmt(fin_spare_aisle * 1e3)} mm spare.'),
                Card("The speed a human buys", f'in {v("v", "max")}: {v("τ", "delay")} → {v("τ", "delay")} + {v("Δ", "da")} + {v("T", "Σ")} + {v("a", "brake")}/2{v("j", "max")}', chapter="Ch 14",
                     ex=f'{v("Δ", "da")} = {fmt(t_da * 1e3, 1)} ms, room {v("D", "clear")} − δ: {n(f"{v_human:.2f} m/s")}, below the {fmt(v_aisle, 1)} m/s aisle speed. Slow down before asking.'),
                Card("One assistant, many machines", holds=f'M/M/1: random requests, one assistant. Waits explode as {v("ρ")} → 1.', eq=[f'{v("ρ")} = machines × requests/h × service', f'mean wait = service × {v("ρ")} / (1 − {v("ρ")})'], chapter="Ch 17",
                     ex=f'{q.machines} × {lam_q:g}/h × {fmt(t_s * 60)} min: {v("ρ")} = {rho:g}, wait {n(fmt(w_q) + " s")} (p99 {fmt(p99_w)} s). '
                        f'{q32.machines} machines: {v("ρ")} = {q32.utilization:g}, {n(fmt(_m(q32.mean_wait, "s")) + " s", True)}.'),
                Card("Frames to detect a person", holds=f'{v("T", "frame")} frame period (1/fps), Δ{v("d")} travel lost before braking, {v("v", "impact")} speed on reaching the person.', eq=[f'{v("N", "frames")} = ⌈ln {v("p", "miss")} / ln(1 − {v("p", "frame")})⌉', f'Δ{v("d")} = {v("v")} {v("T", "frame")} Δ{v("N", "frames")}, &nbsp; {v("v", "impact")} = √(2{v("a", "brake")} Δ{v("d")})'], chapter="Ch 17",
                     ex=f'{v("p", "miss")} = {p_miss:g}, {fmt(cam_fps)} fps, {fmt(v_aisle, 1)} m/s: {v("p", "frame")} = {ds.reference:g} needs {k_ref} frames, {p_lo:g} needs {k_lo}. '
                        f'Δ{v("d")} = {fmt(lost * 1e3)} mm, so a stop sized for the first group hits the second at {n(f"{v_imp:.2f} m/s", True)} ({fmt(e_imp)} J).'),
            ], "Ch 14, 17"),
        ],
        bottom=Table(["The running machine (Ch 1)", "device", "rate", "timing", "capacity or limit", "power"], [
            ["Base", "warehouse AMR, 2 hub motors", f"{fmt(v_drive, 1)} m/s limit", f"{fmt(a_br, 1)} m/s², {fmt(t_act * 1e3)} ms brake onset",
             f"{fmt(m_load)} kg loaded", f"{fmt(_m(RM.base.nominal_power, 'W'))} W"],
            ["Arm", f"{RM.arm.name}, {RM.arm.dofs} joints", f"{fmt(_m(RM.arm.control_frequency, 'Hz'))} Hz control", f"{fmt(tcp, 1)} m/s TCP limit",
             f"{fmt(_m(RM.arm.payload_capacity, 'kg'))} kg, {fmt(_m(RM.arm.max_torque, 'N*m'))} N·m", f"{fmt(_m(RM.arm.nominal_power, 'W'))} W"],
            ["Application processor", soc.name, f"{fmt(peak16 / 1e12)} TFLOP/s", f"{fmt(f_intent)} Hz intent, {fmt(_m(RM.chunk_rate, 'Hz'))} Hz chunks",
             f"{fmt(bw_peak / 1e9)} GB/s, {fmt(_m(soc.memory.capacity, 'GiB'))} GiB", f"{fmt(_m(soc.tdp_min, 'W'))}–{fmt(_m(soc.tdp, 'W'))} W"],
            ["Permission path", "lockstep safety MCU", f"{fmt(_m(RM.permission_rate, 'Hz'))} Hz tick", f"{fmt(t_lease * 1e3)} ms lease",
             f"{fmt(_m(RM.permission_mcu.memory.sram_capacity, 'MiB'))} MiB SRAM", f"{fmt(_m(RM.permission_mcu.tdp, 'W'))} W"],
            ["Sensors", f"{short(cam.name)} camera, {short(RM.lidar.name)} lidar, {short(RM.safety_scanner.name)} scanner", f"{fmt(cam_fps)} fps, {fmt(_m(RM.lidar.scan_rate, 'Hz'))} Hz",
             f"{fmt(t_ro * 1e3, 1)} ms readout; scanner {fmt(_m(RM.safety_scanner.response_time, 'ms'))} ms",
             f"{cam.resolution_width}×{cam.resolution_height}; {RM.lidar.channels} ch", f"{_m(cam.nominal_power, 'W'):g} W, {fmt(_m(RM.lidar.nominal_power, 'W'))} W"],
            ["Power", f"{fmt(_m(RM.drive_bus_nominal, 'V'), 1)} V drive bus", f"{fmt(v_low)}–{fmt(v_full)} V pack", f"{fmt(_m(RM.permission_rail_holdup, 's'))} s rail hold-up",
             f"lockout {fmt(v_uvlo)} V", f"{fmt(_m(RM.permission_rail_load, 'W'))} W rail"],
        ]),
    )
    return Sheet(volume=4, book="Physical AI", scope="The Action", pages=[p1, p2],
                 sources=f'Values from the mlsysim registry: the running machine and the book\'s own scenarios, many illustrative or chosen in the book, not measured · '
                         f'g = {G:g} m/s² (standard gravity)')
