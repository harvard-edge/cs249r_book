"""Numbers for the Trajectory Planning study sheet (Volume IV, Chapter 11).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to planning_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Actuators, Embodied, Scenarios


def f(x, nd):
    return f"{x:,.{nd}f}"


def r(x, nd):
    return round(x, nd)


def sci(x, nd, down=False):
    exp = math.floor(math.log10(x))
    mant = x / 10 ** exp
    if down:  # a bound printed rounded must not be weaker than the computed bound
        mant = math.floor(mant * 10 ** nd) / 10 ** nd
    return f"{mant:.{nd}f}\\times10^{{{exp}}}"


def up(x, nd):
    return math.ceil(x * 10 ** nd) / 10 ** nd


def down(x, nd):
    return math.floor(x * 10 ** nd) / 10 ** nd


def numbers() -> dict:
    n = {}
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    aisle = Embodied.Scenario.WarehouseAisle

    # ---- Front: replacement latency (chapter: PlanningReplacementBudget, PlanningLeaseRule)
    age = aisle.perception_age.to("ms").magnitude
    admit = aisle.t_dispatch_to_admission.to("ms").magnitude
    lat = Scenarios.WarehouseArmChunkReplacement.result.latency.to("ms").magnitude
    assert abs(lat - (age + admit)) < 1e-9
    renew = (1 / rm.chunk_rate).to("ms").magnitude
    lease = aisle.t_lease.to("ms").magnitude
    inflight = math.ceil(r(lat, 1) / renew)
    assert inflight == Scenarios.WarehouseArmChunkReplacement.result.requests_in_flight
    n.update(agems=f(age, 1), admitms=f(admit, 1), latms=f(lat, 1), renewms=f(renew, 0),
             leasems=f(lease, 0), inflight=f(inflight, 0))
    # picture: to-scale bar, 1 ms = 1.9 pt
    s = 1.9
    n.update(pxage=f(age * s, 2), pxadmit=f(admit * s, 2), pxrenew=f(renew * s, 2), pxlease=f(lease * s, 2))

    # ---- Front: joint rate near a singularity (chapter: PlanningTwoLinkArm, PlanningDLSStep)
    good = Scenarios.WarehouseArmRightAngleElbow
    bad = Scenarios.WarehouseArmNearStraightElbow
    vtcp = rm.tcp_speed_limit.to("m/s").magnitude
    qdmax = rm.arm.joint_velocity_limits[1].to("rad/s").magnitude
    lam = good.damping.to("m").magnitude
    sig_bad = bad.result.sigma_min.to("m").magnitude
    n.update(vtcp=f(vtcp, 1), qdmax=f(qdmax, 3), lambdamm=f(lam * 1e3, 0),
             elbowgood=f(good.elbow_angle.to("degree").magnitude, 0),
             elbowbad=f(bad.elbow_angle.to("degree").magnitude, 0),
             rategood=f(good.result.joint_rate_norm.to("rad/s").magnitude, 1),
             ratebad=f(bad.result.joint_rate_norm.to("rad/s").magnitude, 1),
             ceiling=f(good.result.damped_ceiling.to("rad/s").magnitude, 0),
             sigmabad=f(sig_bad * 1e3, 1))
    assert abs(vtcp / (2 * lam) - good.result.damped_ceiling.to("rad/s").magnitude) < 1e-9

    # ---- Q2: ASSUMED 0.5 m/s request along the arm at the near-straight elbow
    v2 = 0.5
    sig2 = r(sig_bad * 1e3, 1) / 1e3
    und = v2 / sig2
    damped = sig2 / (sig2 ** 2 + lam ** 2) * v2
    assert abs(sig_bad / (sig_bad ** 2 + lam ** 2) * vtcp - bad.result.damped_joint_rate.to("rad/s").magnitude) < 1e-9
    n.update(qtool=f(v2, 1), qundamped=f(und, 1), qundone=f(r(und, 1) / math.sqrt(2), 1),
             qdamped=f(damped, 2), qdampone=f(r(damped, 2) / math.sqrt(2), 2), qceiling=f(v2 / (2 * lam), 1))

    # ---- Front + mistake: seam torque and the quintic bridge (chapter: PlanningSeamInertia, PlanningQuinticBridge)
    seam = Scenarios.PandaClassJointSeamBridge
    joint = Actuators.PandaClassJoint.CSG_25_50_ILM70x18
    jref = joint.reflected_inertia.to("kg*m^2").magnitude
    dw = seam.velocity_gap.to("rad/s").magnitude
    tick = seam.tick.to("s").magnitude
    tb = seam.bridge_duration.to("s").magnitude
    taudmax = rm.arm.joint_torque_rate_limit.to("N*m/s").magnitude
    res = seam.result
    step = jref * dw / tick
    assert abs(step - res.step_torque.to("N*m").magnitude) < 1e-9
    step_rate = step / tick
    n.update(jeff=f(jref, 4), domega=f(dw, 1), tickms=f(tick * 1e3, 0), stepnm=f(step, 1),
             steprate=f(step_rate, 0), stepratio=f(step_rate / taudmax, 1),
             rated=f(joint.rated_torque.to("N*m").magnitude, 0), taudmax=f(taudmax, 0),
             bridgems=f(tb * 1e3, 0), bridgepeak=f(jref * 1.5 * dw / tb, 1),
             bridgerate=f(jref * 6 * dw / tb ** 2, 1))
    assert abs(jref * 1.5 * dw / tb - res.peak_torque.to("N*m").magnitude) < 1e-3
    assert abs(jref * 6 * dw / tb ** 2 - res.torque_rate.to("N*m/s").magnitude) < 1e-6

    # ---- Q1: ASSUMED late admission compresses the same bridge to 12 ms
    t1 = 0.012
    rate1 = jref * 6 * dw / t1 ** 2
    tmin = math.sqrt(6 * jref * dw / taudmax)
    assert abs(tmin * 1e3 - res.shortest_bridge.to("ms").magnitude) < 1e-9
    n.update(qbridge=f(t1 * 1e3, 0), qacc=f(1.5 * dw / t1, 0), qpeak=f(jref * 1.5 * dw / t1, 2),
             qjerk=f(6 * dw / t1 ** 2, 0), qrate=f(rate1, 0), qratio=f(rate1 / taudmax, 2),
             qshortest=f(up(tmin * 1e3, 1), 1))

    # ---- Worked example: the stall runway (chapter: PlanningLeaseRule, PlanningStallRunway)
    stall = Scenarios.WarehouseArmStallStop
    st = stall.result
    v = stall.speed.to("m/s").magnitude
    a = st.deceleration.to("m/s^2").magnitude
    j = st.jerk.to("m/s^3").magnitude
    tickw, bus, onset = (x.to("ms").magnitude for x in (aisle.t_tick, aisle.t_bus, aisle.t_brake_onset))
    resp = lease + tickw + bus + onset
    assert abs(resp - st.delay.to("ms").magnitude) < 1e-9
    j_print = r(j, 1)
    brake = v ** 2 / (2 * a) * 1e3
    ramp = v * a / (2 * j_print) * 1e3
    stop = r(brake, 1) + r(ramp, 1)
    travel = v * resp
    total = travel + stop
    clear = st.clear_distance.to("mm").magnitude
    assert abs(stop - st.scurve_travel.to("mm").magnitude) < 0.05
    n.update(wv=f(v, 1), wa=f(a, 1), wj=f(j_print, 1), wthresh=f(a ** 2 / j_print, 3),
             wtick=f(tickw, 0), wbus=f(bus, 0), wonset=f(onset, 0), wresp=f(resp, 0),
             wtravel=f(travel, 0), wbrake=f(brake, 1), wramp=f(ramp, 1), wstop=f(stop, 1),
             wtotal=f(total, 1), wclear=f(clear, 0), wmargin=f(clear - total, 1),
             wstopms=f((v / a + a / j_print) * 1e3, 0))
    # picture: 1 mm = 0.62 pt
    k = 0.62
    n.update(ptravel=f(travel * k, 2), pstop=f(stop * k, 2), pclear=f(clear * k, 2))

    # ---- Mistake: the alternate longer-lease schedule's commitment and admission cutoff (chapter: PlanningLeaseRule)
    sched = Scenarios.WarehouseArmCommitmentSchedule
    commit = sched.commit.to("ms").magnitude
    respc = sched.response.to("ms").magnitude
    assert abs(commit - respc - sched.result.admission_cutoff.to("ms").magnitude) < 1e-9
    n.update(commitms=f(commit, 0), commitresp=f(respc, 0), cutoffms=f(commit - respc, 0))

    # ---- Your turn: same stall carrying an open mug (chapter: PlanningMugSpill)
    mug = Scenarios.WarehouseArmOpenMugStop.result
    am = r(mug.spill_deceleration.to("m/s^2").magnitude, 3)
    jm = r(mug.stop.jerk.to("m/s^3").magnitude, 2)
    bm = v ** 2 / (2 * am) * 1e3
    rmp = v * am / (2 * jm) * 1e3
    stopm = r(bm, 1) + r(rmp, 1)
    totalm = travel + stopm
    assert abs(totalm - mug.stop.required_distance.to("mm").magnitude) < 0.1
    # highest speed whose response travel and stop fit the clearance, from the printed a and j
    tr = resp / 1e3
    qa, qb, qc = 1 / (2 * am), tr + am / (2 * jm), -clear / 1e3
    vfit = (-qb + math.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa)
    assert abs(vfit - mug.carry_speed.to("m/s").magnitude) < 2e-3
    assert vfit >= am ** 2 / jm
    n.update(ma=f(am, 3), mj=f(jm, 2), mthresh=f(am ** 2 / jm, 3), mbrake=f(bm, 1), mramp=f(rmp, 1),
             mstop=f(stopm, 1), mtotal=f(totalm, 1), mover=f(totalm - clear, 1), mfit=f(down(vfit, 2), 2))

    # ---- Mistake: constant deceleration on the base (chapter: PlanningC2Stop)
    base = Scenarios.WarehouseBaseJerkLimitedStop
    b = base.result
    vb = base.speed.to("m/s").magnitude
    ab = b.deceleration.to("m/s^2").magnitude
    jb = r(b.jerk.to("m/s^3").magnitude, 3)
    const = b.constant_required.to("mm").magnitude
    rampb = vb * ab / (2 * jb) * 1e3
    totb = r(const, 1) + r(rampb, 1)
    clearb = b.clear_distance.to("mm").magnitude
    assert abs(totb - b.required_distance.to("mm").magnitude) < 0.1
    n.update(bbrake=f(b.braking_travel.to("mm").magnitude, 1),
             bother=f(const - b.braking_travel.to("mm").magnitude, 1))
    n.update(bv=f(vb, 1), ba=f(ab, 1), bj=f(jb, 3), bconst=f(const, 1), bramp=f(rampb, 1),
             btotal=f(totb, 1), bclear=f(clearb / 1e3, 2), bover=f(totb - clearb, 1))

    # ---- Q3: ASSUMED 30-minute shift at the running renewal rate, mission target from the chapter
    shift = 30 * 60
    rate = rm.chunk_rate.to("Hz").magnitude
    eps_m = Scenarios.GenericReplannerMissionExposure.mission_budget
    handoffs = shift * rate
    eps = eps_m / handoffs
    n.update(qshiftmin=f(shift / 60, 0), qhz=f(rate, 0), qmission=sci(eps_m, 0),
             qhandoffs=f(handoffs, 0), qeps=sci(eps, 1, down=True))

    # ---- Q4: ASSUMED two seam logs on the seam joint
    w4, d4, acc4 = 0.6, 70.0, 4.0
    n.update(lspeed=f(w4, 1), ldelay=f(d4, 0), lgap=f(w4 * d4 / 1e3, 3), ljump=f(acc4, 0),
             ltorque=f(jref * acc4, 2))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
