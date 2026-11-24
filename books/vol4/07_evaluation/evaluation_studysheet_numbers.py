"""Numbers for the Closed-Loop Evaluation study sheet (Volume IV, Chapter 7).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to evaluation_studysheet_numbers.tex next to this file.
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


def pct(x, nd):
    return f"{100 * x:,.{nd}f}"


def numbers() -> dict:
    n = {}
    camp = Scenarios.LatchEvaluationCampaign
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator

    # ---- Front: zero-failure card and ladder (chapter: ClopperPearsonZeroFailure, DoorCycle)
    alpha = camp.alpha
    ln_inv = camp.ln_inv_alpha
    n20 = camp.clean_runs
    p20 = camp.lower_bound
    n.update(alpha=f(alpha, 2), conf=pct(camp.confidence, 0), lninv=f(ln_inv, 3), nclean=f(n20, 0),
             plowpct=pct(p20, 1), failuppct=pct(camp.failure_upper, 1), rulethree=f(camp.rule_of_three, 2))
    t90, t99, t999 = camp.nine_targets
    runs = {k: camp.runs_for(t) for k, t in (("ninety", t90), ("ninetynine", t99), ("threenines", t999))}
    assert runs["threenines"] == camp.target_runs
    n.update(runsninety=f(runs["ninety"], 0), runsninetynine=f(runs["ninetynine"], 0),
             runsthreenines=f(runs["threenines"], 0), tgtninety=pct(t90, 0), tgtninetynine=pct(t99, 0),
             tgtthreenines=pct(t999, 1))
    cycle = camp.cycle.to("s").magnitude
    n.update(runtime=f(camp.run.to("s").magnitude, 0), treset=f(camp.reset.to("s").magnitude, 0), tcycle=f(cycle, 0),
             hthreenines=f(camp.campaign_time(runs["threenines"]).to("hour").magnitude, 1))
    # ladder picture: x = 60 * log10(runs) pt, offset so 10 runs sits at x = 0
    for key, cnt in (("lxclean", n20), ("lxninety", runs["ninety"]), ("lxninetynine", runs["ninetynine"]),
                     ("lxthreenines", runs["threenines"])):
        n[key] = f(60 * (math.log10(cnt) - 1), 2)
    n.update(tickten=f(10, 0), tickhundred=f(100, 0), tickthousand=f(1000, 0))
    # with failures: exact binomial (Clopper-Pearson) lower limit, chapter's BoundWithFailures
    n.update(nwithfail=f(camp.clean_runs - camp.recorded_failures, 0), pfailpct=pct(camp.lower_bound_with_failures, 1))
    assert camp.lower_bound_with_failures < p20

    # ---- Worked example: what twenty clean runs support, and what a 99 percent claim costs
    expo20 = round(ln_inv, 3) / n20
    n.update(wexpo=f(expo20, 4), wlnnn=f(-math.log(t99), 5),
             wratio=f(round(ln_inv, 3) / round(-math.log(t99), 5), 1),
             wsec=f(runs["ninetynine"] * cycle, 0),
             whours=f(runs["ninetynine"] * cycle / 3600, 1))
    assert abs(math.exp(-expo20) - p20) < 5e-4

    # ---- Your turn: same campaign, 99.5 percent target (the chapter's checkpoint target)
    t995 = camp.estimate_target
    n995 = camp.runs_for(t995)
    n.update(ytgt=pct(t995, 1), yrule=f(camp.estimate_rule_of_three_runs, 0), ylnn=f(-math.log(t995), 6),
             yruns=f(n995, 0), ysec=f(n995 * cycle, 0), yhours=f(n995 * cycle / 3600, 1))
    assert math.ceil(round(ln_inv, 3) / round(-math.log(t995), 6)) == n995

    # ---- Mistake: offline score vs closed-loop drift (chapter: AisleDrift)
    drift = Scenarios.AisleHeadingDrift
    n.update(headdeg=f(drift.heading_offset.to("degree").magnitude, 1),
             msemath=f(drift.angular_mse.magnitude / 1e-4, 2) + r"\times 10^{-4}",
             vdrive=f(drift.speed.to("m/s").magnitude, 1), sideclear=f(aisle.side_clearance.to("m").magnitude, 2),
             timpact=f(drift.time_to_clearance.to("s").magnitude, 1),
             ximpact=f(drift.distance_to_clearance.to("m").magnitude, 1),
             aislelen=f(aisle.aisle_length.to("m").magnitude, 0))

    # ---- Mistake: a thousand clean runs (chapter fallacy)
    n.update(nthousand=f(camp.fallacy_runs, 0), pthousandpct=pct(camp.fallacy_lower_bound, 2))

    # ---- Mistake: twenty against sixteen of twenty (chapter: PolicyComparison)
    cmp_ = Scenarios.LatchCheckpointComparison
    n.update(armruns=f(cmp_.arm_runs, 0), armb=f(cmp_.successes_b, 0), fisherp=f(cmp_.fisher_p, 3),
             phigh=pct(cmp_.p_high, 0), plow=pct(cmp_.p_low, 0), powerpct=pct(cmp_.power, 0),
             perarm=f(cmp_.runs_per_arm, 0))

    # ---- Front: SPRT card values (chapter: Sprt)
    sprt = Scenarios.LatchSequentialTest
    inc_s = sprt.success_increment
    inc_f = abs(sprt.failure_increment)
    up = sprt.accept
    lo = sprt.reject
    n.update(pzero=f(sprt.p_unacceptable, 2), pone=f(sprt.p_acceptable, 2), salpha=f(sprt.alpha, 2),
             sbeta=f(sprt.beta, 2), incs=f(inc_s, 4), incf=f(inc_f, 3), bup=f(up, 3), blo=f(abs(lo), 3),
             naccept=f(sprt.runs_to_accept, 0))

    # ---- Q3: ASSUMED trace, 12 clean runs then one failure; worked from the printed increments
    m_clean = 12  # ASSUMED
    s_r, f_r, up_r, lo_r = round(inc_s, 4), round(inc_f, 3), round(up, 3), round(lo, 3)
    score = m_clean * s_r - f_r
    assert lo_r < score < up_r
    more = math.ceil((up_r - score) / s_r)
    n.update(qclean=f(m_clean, 0), qcleanp=f(m_clean + 1, 0), qscore=f(abs(score), 3), qgap=f(up_r - score, 3),
             qratio=f((up_r - score) / s_r, 1), qmore=f(more, 0), qfirst=f(f_r, 3))

    # ---- Q4: ASSUMED site hazard rate 1e-5 per hour and 10 rigs; exposure wall (chapter: ExposureWall)
    wall = Scenarios.HazardExposureWall
    lam_exp = 5  # ASSUMED: lambda = 1e-5 per hour
    rigs = 10  # ASSUMED
    rate = Q_(10.0 ** -lam_exp, "1/hour")
    t_hours = wall.exposure(rate).to("hour").magnitude
    assert abs(t_hours - wall.ln_inv_alpha / 10.0 ** -lam_exp) < 1e-6
    assert abs(round(t_hours) - round(round(wall.ln_inv_alpha, 3) / 10.0 ** -lam_exp)) < 50
    rig_year = wall.rig_time.to("hour").magnitude
    t_print = round(round(wall.ln_inv_alpha, 3) / 10.0 ** -lam_exp)  # from the printed 2.996, as the student works it
    n.update(qlamexp=f(lam_exp, 0), qlammath=f"10^{{-{lam_exp}}}", qhours=f(t_print, 0), rigyear=f(rig_year, 0),
             qyears=f(t_print / rig_year, 1), qrigs=f(rigs, 0), qcal=f(t_print / rig_year / rigs, 1),
             wallmantnine=f(wall.exposure(wall.rates[2]).to("hour").magnitude / 1e9, 1))

    # ---- Q1: ASSUMED 50-run campaign with every outcome channel (one event per intervened run)
    runs_q, clean_q, clip_q, assist_q, stop_q = 50, 41, 5, 2, 2  # ASSUMED
    assert clean_q + clip_q + assist_q + stop_q == runs_q
    done = clean_q + clip_q + assist_q
    events = clip_q + assist_q + stop_q
    t_active_min = runs_q * camp.run.to("s").magnitude / 60
    n.update(qruns=f(runs_q, 0), qunass=f(clean_q, 0), qclip=f(clip_q, 0), qassist=f(assist_q, 0), qstop=f(stop_q, 0),
             qdone=f(done, 0), qdonepct=pct(done / runs_q, 0), qacrpct=pct(clean_q / runs_q, 0), qevents=f(events, 0),
             qactive=f(t_active_min, 1), qmtbi=f(round(t_active_min, 1) / events, 1))

    # ---- Q2: chunk timing quoted from the registry (chapter: EvalMachine)
    n.update(chunkinfer=f(rm.chunk_inference_latency.to("ms").magnitude, 0),
             chunkperiod=f((1 / rm.chunk_rate).to("ms").magnitude, 0))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
