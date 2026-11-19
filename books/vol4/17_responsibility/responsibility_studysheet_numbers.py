"""Numbers for the Responsible Physical AI study sheet (Volume IV, Chapter 17).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to responsibility_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Chained results are computed from the rounded values the sheet prints beside them, so a
student who works from the printed numbers gets the printed answer.
"""
import math
from pathlib import Path

from mlsysim import Embodied, Scenarios
from mlsysim.core.units import Q_


def f(x, nd):
    return f"{x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    aisle = Embodied.Scenario.WarehouseAisle

    # ---- Front: time test (chapter: OversightTime)
    over = Scenarios.WarehouseRackEndOversightTime
    v = over.budget.speed.to("m/s").magnitude
    dclear = aisle.d_clear.to("m").magnitude
    window = round(dclear / v, 2)
    assert abs(window - over.hazard_window.to("s").magnitude) < 5e-3
    take = round(over.budget.takeover.to("s").magnitude, 2)
    travel = round(v * take, 1)
    assert abs(travel - over.takeover_travel.to("m").magnitude) < 0.05
    n.update(v=f(v, 1), dclear=f(dclear, 1), window=f(window, 2), takeover=f(take, 2), travel=f(travel, 1))

    # ---- Mistake 1: chain of six links at 0.90 (chapter: DecisionChain)
    chain = Scenarios.AdmittedCommandDecisionChain
    p_poor = chain.link_probabilities[2]
    c = chain.completeness(p_poor)
    assert abs(c - p_poor ** chain.links) < 1e-12
    n.update(links=f"{chain.links}", plink=f(p_poor, 2), chain=f(c, 2))

    # ---- Mistake 3: exposure concentration (chapter: ExposureConcentration)
    cx = Scenarios.WarehouseCrossingExposure
    n.update(encshare=f(cx.encounter_share * 100, 0), usershare=f(cx.user_share * 100, 0),
             conc=f(cx.concentration, 0), rest=f(cx.rest_ratio, 2))

    # ---- Front card and worked example: detection gap to impact (chapter: DetectionShortfall,
    # nbk-responsibility-detection-shortfall, middle row)
    det = Scenarios.WarehouseGroupDetectionShortfall
    r = det.result
    tf = round(det.frame_period.to("ms").magnitude, 1)
    fps = (1 / det.frame_period).to("1/s").magnitude
    a = det.site.a_brake.to("m/s^2").magnitude
    m = det.site.machine.loaded_mass.to("kg").magnitude
    pmiss_exp = round(math.log10(det.all_miss_probability))
    lnmiss = round(math.log(1 / det.all_miss_probability), 2)

    def frames(p):
        lnp = round(math.log(1 / (1 - p)), 2)
        return lnp, math.ceil(lnmiss / lnp), lnmiss / lnp

    lnref, nref, qref = frames(det.reference)
    assert nref == r.reference_frames
    row = r.rows[1]
    plow = row.per_frame_detection
    lnlow, nlow, qlow = frames(plow)
    assert nlow == row.frames
    dn = nlow - nref
    dd = round(v * tf * dn, 1)                    # mm, from printed v and frame period
    vimp = round(math.sqrt(2 * a * dd / 1e3), 2)
    e = round(m * a * dd / 1e3, 0)
    assert abs(dd / 1e3 - row.distance_lost.to("m").magnitude) < 5e-4
    assert abs(vimp - row.impact_speed.to("m/s").magnitude) < 6e-3
    assert abs(e - row.impact_energy.to("J").magnitude) < 0.6
    n.update(tframe=f(tf, 1), fps=f(fps, 0), abrake=f(a, 0), mass=f(m, 0), pmissexp=f"{pmiss_exp}",
             lnmiss=f(lnmiss, 2), pref=f(det.reference, 2), lnref=f(lnref, 2), qref=f(qref, 2), nref=f"{nref}",
             plow=f(plow, 2), lnlow=f(lnlow, 2), qlow=f(qlow, 2), nlow=f"{nlow}", dn=f"{dn}",
             dd=f(dd, 1), vimp=f(vimp, 2), eimp=f(e, 0),
             twoa=f(2 * a, 0), dbrake=f(r.braking_travel.to("m").magnitude, 2))

    # picture: speed against distance, 1 m = 260 pt, 1 m/s = 84 pt
    sx, sy = 260, 84
    xb = r.reference_brake_onset.to("m").magnitude
    xp = r.reference_stop.to("m").magnitude
    n.update(pxb=f(xb * sx, 2), pxbl=f((xb + row.distance_lost.to("m").magnitude) * sx, 2),
             pxp=f(xp * sx, 2), pv=f(v * sy, 2), pvimp=f(row.impact_speed.to("m/s").magnitude * sy, 2),
             pk=f(2 * a * sy ** 2 / sx, 4))

    # ---- Your turn: the same stop, a group seen in 0.60 of frames (chapter table, last row)
    yrow = r.rows[2]
    py = yrow.per_frame_detection
    lny, ny, qy = frames(py)
    assert ny == yrow.frames
    ydn = ny - nref
    ydd = round(v * tf * ydn, 1)
    yv = round(math.sqrt(2 * a * ydd / 1e3), 2)
    ye = round(m * a * ydd / 1e3, 0)
    assert abs(ydd / 1e3 - yrow.distance_lost.to("m").magnitude) < 5e-4
    assert abs(yv - yrow.impact_speed.to("m/s").magnitude) < 6e-3
    assert abs(ye - yrow.impact_energy.to("J").magnitude) < 0.6
    n.update(py=f(py, 2), lny=f(lny, 2), qy=f(qy, 2), ny=f"{ny}", ydn=f"{ydn}", ydd=f(ydd, 1),
             yv=f(yv, 2), ye=f(ye, 0), yratio=f(ye / e, 1))

    # ---- Front card: the queue (chapter: SupervisorLoad); Q1 uses ASSUMED inputs
    q = Scenarios.WarehouseRemoteAssistantQueue
    n.update(qmach=f"{q.machines}", qrate=f(q.request_rate.to("1/hour").magnitude, 1),
             qserv=f(q.service_time.to("minute").magnitude, 0), qrho=f(q.result.utilization, 2),
             qwait=f(q.result.mean_wait.to("minute").magnitude, 0), qcap=f(q.utilization_cap, 1))
    q1 = q.variant(machines=16, request_rate=Q_(0.6, "1/hour"), service_time=Q_(4.0, "minute"))  # ASSUMED
    lam1 = round(q1.machines * q1.request_rate.to("1/hour").magnitude, 1)
    rho1 = round(lam1 * q1.service_time.to("hour").magnitude, 2)
    w1 = round(rho1 * 4.0 / (1 - rho1), 1)
    assert abs(rho1 - q1.result.utilization) < 5e-3
    assert abs(w1 - q1.result.mean_wait.to("minute").magnitude) < 0.1
    nmax1 = q1.max_machines
    n.update(aqmach=f"{q1.machines}", aqrate=f(0.6, 1), aqserv=f(4.0, 0), aqlam=f(lam1, 1),
             aqrho=f(rho1, 2), aqlammin=f(lam1 / 60, 2), aqidle=f(1 - rho1, 2), aqwait=f(w1, 1), aqper=f(0.6 * 4.0 / 60, 2), aqmax=f"{nmax1}")

    # ---- Front card: leaving a stage (chapter: StageExit); Q2 uses ASSUMED inputs
    st = Scenarios.WarehouseSupervisedStageExit
    alpha = 1 - st.confidence
    lnalpha = round(math.log(1 / alpha), 2)
    target = round(1 / st.rates[0].to("1/hour").magnitude)
    hours = st.exposure(st.rates[0]).to("hour").magnitude
    assert abs(lnalpha * target - hours) < 5
    n.update(alpha=f(alpha, 2), lnalpha=f(lnalpha, 2), conf=f(st.confidence * 100, 0),
             starget=f(target, 0), shours=f(lnalpha * target, 0), smach=f"{st.rigs}",
             sday=f(st.hours_per_day.to("hour/day").magnitude, 0),
             sdays=f(math.ceil(st.calendar_days(st.rates[0]).to("day").magnitude), 0))
    st2 = st.variant(rates=[Q_(1 / 2000, "1/hour")], rigs=8)  # ASSUMED
    t2 = round(lnalpha * 2000, 0)
    days2 = math.ceil(t2 / (8 * 16))
    assert abs(t2 - st2.exposure(st2.rates[0]).to("hour").magnitude) < 20
    assert days2 == math.ceil(st2.calendar_days(st2.rates[0]).to("day").magnitude)
    n.update(atarget=f(2000, 0), amach=f"{8}", aday=f(16, 0), afleet=f(8 * 16, 0),
             ahours=f(t2, 0), adays=f"{days2}", aship=f"{30}")  # ASSUMED day the new build ships

    # ---- Front line and Q3: per-group sample (chapter: SubgroupIntervals); Q3 uses ASSUMED share
    s = Scenarios.PedestrianGroupEvaluationSample
    n.update(sa=f(s.rate_a, 2), sb=f(s.rate_b, 2), spower=f(s.power * 100, 0), sneed=f(s.per_group_needed, 0),
             stotal=f(s.total, 0))
    s3 = s.variant(small_share=0.03)  # ASSUMED
    lo3, hi3 = s3.small_interval
    assert lo3 < s.rate_b < s.rate_a < hi3
    n.update(a3share=f(s3.small_share * 100, 0), a3n=f(s3.small, 0), a3lo=f(lo3, 3), a3hi=f(hi3, 3),
             a3pct=f(s.per_group_needed / s.total * 100, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
