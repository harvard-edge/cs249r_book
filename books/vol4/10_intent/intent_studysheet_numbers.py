"""Numbers for the Grounded Intent study sheet (Volume IV, Chapter 10).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(IntentLeaseArchetypes, ConformalAmbiguity, MugLeaseExpiry), and written as \\setnum{key}{value}
to intent_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios
from mlsysim.physics.robotics import calc_latch_contact_stopping_profile, calc_target_evidence_horizon


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    cad = Scenarios.ConveyorMugIntentCadence.result

    # ---- Front: evidence horizons of the conveyor mug (chapter: IntentLeaseArchetypes)
    rtol = aisle.grasp_tolerance.to("mm").magnitude
    e0 = aisle.grasp_initial_error.to("mm").magnitude
    vbelt = aisle.v_conveyor.to("mm/s").magnitude
    aslip = aisle.a_conveyor_slip.to("m/s^2").magnitude
    tau_u = cad.untracked_horizon.to("ms").magnitude
    tau_t = cad.tracked_horizon.to("ms").magnitude
    assert abs(tau_u - calc_target_evidence_horizon(aisle.grasp_tolerance, aisle.grasp_initial_error,
                                                     aisle.v_conveyor).to("ms").magnitude) < 1e-9
    assert abs(tau_u - (rtol - e0) / vbelt * 1e3) < 1e-9
    assert abs(tau_t - math.sqrt(2 * (rtol - e0) * 1e-3 / aslip) * 1e3) < 1e-6
    infer = rm.intent_inference_latency.to("ms").magnitude
    period = cad.intent_period.to("ms").magnitude
    n.update(rtol=f(rtol, 0), ezero=f(e0, 0), budget=f(rtol - e0, 0), vbelt=f(vbelt, 0), aslip=f(aslip, 1),
             tauu=f(tau_u, 0), taut=f(tau_t, 0), infer=f(infer, 0), period=f(period, 0),
             frate=f(rm.intent_rate.to("Hz").magnitude, 0))
    # picture: error (mm) against time since capture (ms); 0.5 pt/ms, 3 pt/mm
    xs, ys = 0.5, 3.0
    tmax = 380.0
    pts_u = [(0.0, e0), ((30 - e0) / vbelt * 1e3, 30.0)]
    pts_t = [(t, e0 + 0.5 * aslip * 1e3 * (t * 1e-3) ** 2) for t in range(0, 341, 10)]
    pts_t = [(t, e) for t, e in pts_t if e <= 30.0]
    t30 = math.sqrt(2 * (30.0 - e0) * 1e-3 / aslip) * 1e3   # where the tracked curve reaches the 30 mm top
    pts_t.append((t30, 30.0))
    n.update(curveu="".join(f"({t * xs:.2f},{e * ys:.2f})" for t, e in pts_u),
             curvet="".join(f"({t * xs:.2f},{e * ys:.2f})" for t, e in pts_t),
             ptauu=f(tau_u * xs, 2), ptaut=f(tau_t * xs, 2), pinfer=f(infer * xs, 2), prtol=f(rtol * ys, 2),
             pezero=f(e0 * ys, 2), paxis=f(tmax * xs, 2), ptop=f(30 * ys, 2))

    # ---- Worked example: the mug lease expiry, untracked (chapter: MugLeaseExpiry)
    exp = Scenarios.ConveyorMugIntentLeaseExpiry
    er = exp.result
    tadmin = exp.administrative_lease.to("ms").magnitude
    dclear = exp.clearance.to("mm").magnitude
    v = rm.tcp_speed_limit.to("m/s").magnitude
    trev = cad.stop_onset.to("ms").magnitude
    abrake = er.credible_deceleration.to("m/s^2").magnitude
    ab = round(abrake, 2)                       # printed value, used for the chain
    dreact = v * trev                            # mm (m/s x ms)
    dbrake = v ** 2 / (2 * ab) * 1e3             # mm
    dstop = round(dreact) + round(dbrake, 1)
    tstop = (dclear - round(dstop)) / v          # ms (mm / (m/s))
    assert abs(dstop - er.stop_distance.to("mm").magnitude) < 0.5
    assert abs(tstop - er.stopping_deadline.to("ms").magnitude) < 0.5
    texp = min(tau_u, tadmin, tstop)
    assert texp == er.expiry_untracked.to("ms").magnitude
    n.update(tadmin=f(tadmin, 0), dclear=f(dclear, 0), vtcp=f(v, 1), trev=f(trev, 0), abrake=f(ab, 2),
             dreact=f(dreact, 0), dbrake=f(dbrake, 0), dstop=f(dstop, 0), tstop=f(tstop, 0),
             vcart=f(rm.arm.cartesian_velocity_limit.to("m/s").magnitude, 1),
             testop=f(rm.arm.emergency_stop_time.to("ms").magnitude, 0))
    # picture: stop strip to scale against the clearance, 0.6 pt/mm
    sc = 0.6
    n.update(sreact=f(dreact * sc, 2), sbrake=f(dbrake * sc, 2), sclear=f(dclear * sc, 2),
             sspare=f(dclear - round(dstop), 0))

    # ---- Your turn: same lease under a qualified tracker (slip only)
    texp_t = min(round(tau_t), tadmin, round(tstop))
    assert texp_t == tadmin and er.binding_tracked == "administrative"
    n.update(yexp=f(texp_t, 0), yover=f(round(tau_t) - tadmin, 0))
    stall = cad.stall.to("ms").magnitude
    n.update(stall=f(stall, 0), erru=f(cad.stall_error_untracked.to("mm").magnitude, 0),
             errt=f(cad.stall_error_tracked.to("mm").magnitude, 0))

    # ---- Mistake: effort ceiling at the cage-door latch (chapter: latch ceilings notebook)
    ftrip = aisle.f_latch_tripwire.to("N").magnitude
    klatch = aisle.k_latch.to("N/mm").magnitude
    vlatch = aisle.v_latch_approach.to("m/s").magnitude
    tcontact = aisle.t_contact_response.to("ms").magnitude
    fbrake = ftrip + klatch * vlatch * tcontact          # N/mm x (m/s x ms = mm)
    assert abs(fbrake - calc_latch_contact_stopping_profile(scenario=aisle)["f_at_brake_onset"].to("N").magnitude) < 1e-9
    assert fbrake < aisle.f_latch_limit.to("N").magnitude
    xrem = (aisle.f_latch_limit.to("N").magnitude - fbrake) / klatch    # mm left before the limit
    assert abs(xrem - calc_latch_contact_stopping_profile(scenario=aisle)["x_remaining"].to("mm").magnitude) < 1e-9
    n.update(xrem=f(xrem, 2))
    n.update(ftrip=f(ftrip, 0), klatch=f(klatch, 0), vlatch=f(vlatch, 2), tcontact=f(tcontact, 0),
             fbrake=f(fbrake, 0), flimit=f(aisle.f_latch_limit.to("N").magnitude, 0))

    # ---- Mistake: per-box scores (chapter: ConformalAmbiguity, MugJarReferentSet)
    beside = Scenarios.MugJarReferentSet
    br = beside.result
    n.update(smug=f(beside.scores[0], 2), sjar=f(beside.scores[1], 2), cmug=f(br.normalized[0], 2),
             cjar=f(br.normalized[1], 2), qhat=f(beside.threshold_score, 2), cthr=f(br.threshold, 2),
             ncal=f(beside.calibration_count, 0), eps=f(beside.alpha, 2))
    assert br.set_size == 2

    # ---- Q1: ASSUMED error box (9, 9, 9) mm against the mug's 15 mm tolerance on every axis
    ebox = 9.0
    ratio = round(ebox / rtol, 2)
    norm = math.sqrt(3 * ratio ** 2)
    n.update(qbox=f(ebox, 0), qratio=f(ratio, 2), qnorm=f(norm, 2))
    assert ebox < rtol and norm > 1

    # ---- Q2: ASSUMED detector scores, same calibration (threshold 1 - q_hat)
    s_m, s_j = 0.72, 0.24
    c_m, c_j = s_m / (s_m + s_j), s_j / (s_m + s_j)
    thr = 1 - beside.threshold_score
    assert c_m >= thr > c_j
    n.update(qsmug=f(s_m, 2), qsjar=f(s_j, 2), qcmug=f(c_m, 2), qcjar=f(c_j, 2))

    # ---- Q3: ASSUMED handle 80 mm away at admission; TCP speed ceiling; the two lease expiries
    # found in the worked example (untracked) and your turn (tracked)
    reach = Scenarios.ConveyorMugDynamicReach
    d3 = 80.0
    t3 = d3 / reach.speed_ceiling.to("mm/ms").magnitude
    assert texp < t3 < texp_t
    n.update(qdist=f(d3, 0), qtransit=f(t3, 0), qreachu=f(v * texp, 0),
             chapdist=f(reach.target_distance.to("mm").magnitude, 0),
             chaptransit=f(reach.result.transit_floor.to("ms").magnitude, 0))

    # ---- Q4: ASSUMED coworker track 40 ms old, screened over the 100 ms administrative lease
    occ = Scenarios.AisleCoworkerForwardOccupancy
    vh = aisle.v_human_approach.to("m/s").magnitude
    age = 40.0
    grow = vh * (age + tadmin)                   # mm (m/s x ms)
    long_ = occ.forecast.to("s").magnitude
    n.update(vh=f(vh, 1), qage=f(age, 0), qwin=f(age + tadmin, 0), qgrow=f(grow, 0),
             tlong=f(long_, 0), glong=f(vh * long_, 1), qglong=f(vh * (age * 1e-3 + long_), 2), aisle=f(aisle.aisle_width.to("m").magnitude, 2),
             tspan=f(aisle.aisle_width.to("m").magnitude / vh, 2))
    assert abs(vh * long_ - occ.result.growth_forecast.to("m").magnitude) < 1e-9
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
