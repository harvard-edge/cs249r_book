"""Numbers for the Trajectory Curation study sheet (Volume III, Chapter 15).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(VerificationCascadeEconomicsThroughput, MixedCurationTokenBudgetAllocation,
InterventionOrderTradeoff), and written as \\setnum{key}{value} to
trajectory_curation_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC
from mlsysim.core.units import USD, second, token


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))   # round(.., 9) drops float noise


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def comma(n):
    return f"{int(n):,}".replace(",", "{,}")


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def cascade(sc):
    """Pass rates and slot times (s) of a chapter admission-order scenario, in record/use/process/outcome/target order."""
    p = {k: v[0] for k, v in sc.checks.items()}
    t = {k: v[1].to(second).magnitude for k, v in sc.checks.items()}
    return p, t


def expected_time(p, t, order):
    """Short-circuit expected slot time per attempted record (the chapter's cascade equation)."""
    total, reach = 0.0, 1.0
    for c in order:
        total += reach * t[c]
        reach *= p[c]
    return total


def numbers() -> dict:
    n = {}

    # ---- Worked example: which of process audit and outcome check runs first
    #      (chapter: VerificationCascadeEconomicsThroughput, Scenarios.DemonstrationAdmissionOrder)
    sc = SC.DemonstrationAdmissionOrder
    res = sc.result
    p, t = cascade(sc)
    pfirst, ofirst = [list(o) for o in sc.orders]
    assert pfirst.index("process") < pfirst.index("outcome") and ofirst.index("outcome") < ofirst.index("process")
    rej_p = t["process"] / (1 - p["process"])
    rej_o = t["outcome"] / (1 - p["outcome"])
    near(rej_p, res.cost_per_rejection["process"].to(second).magnitude, 1e-9)
    near(rej_o, res.cost_per_rejection["outcome"].to(second).magnitude, 1e-9)
    assert rej_p < rej_o
    reach = p["record"] * p["use"]
    gain = r(reach, 3) * (t["outcome"] * (1 - p["process"]) - t["process"] * (1 - p["outcome"]))
    e_p, e_o = expected_time(p, t, pfirst), expected_time(p, t, ofirst)
    near(e_p, res.time_per_attempted[0].to(second).magnitude, 1e-9)
    near(e_o, res.time_per_attempted[1].to(second).magnitude, 1e-9)
    near(gain, e_o - e_p, 1e-9)
    y = 1.0
    for c in p:
        y *= p[c]
    near(y, res.route_yield, 1e-12)
    save = r(gain, 2) / r(e_o, 2)
    near(save, res.saving, 0.001)
    per_adm_p, per_adm_o = r(e_p, 2) / r(y, 4), r(e_o, 2) / r(y, 4)
    near(per_adm_p, e_p / y, 0.05)
    n.update(precp=pct(p["record"]), prect=fx(t["record"], 2), pusep=pct(p["use"]), puset=fx(t["use"], 2),
             pprocp=pct(p["process"]), pproct=fx(t["process"], 1), poutp=pct(p["outcome"]), poutt=fx(t["outcome"], 0),
             ptargp=pct(p["target"]), ptargt=fx(t["target"], 0),
             pprocfrac=fx(p["process"], 1), poutfrac=fx(p["outcome"], 1),
             procmiss=fx(1 - p["process"], 1), outmiss=fx(1 - p["outcome"], 1),
             rejproc=fx(rej_p, 0), rejout=fx(rej_o, 1), reach=fx(reach, 3), gain=fx(gain, 2),
             eproc=fx(e_p, 2), eout=fx(e_o, 2), savepct=pct(save, 1), routeyield=fx(y, 4), routeyieldpct=pct(y, 1),
             admproc=fx(per_adm_p, 1), admout=fx(per_adm_o, 1))
    # paired bars (s per attempted record)
    n.update(barproc=fx(e_p, 2), barout=fx(e_o, 2))
    # the cascade sum term by term, as the sheet prints it: reach of each later check, rounded to 3 places
    head = t["record"] + p["record"] * t["use"]
    r_proc, r_out, r_both = r(reach * p["process"], 3), r(reach * p["outcome"], 3), r(reach * p["process"] * p["outcome"], 3)
    sum_p = r(head, 3) + r(reach, 3) * t["process"] + r_proc * t["outcome"] + r_both * t["target"]
    sum_o = r(head, 3) + r(reach, 3) * t["outcome"] + r_out * t["process"] + r_both * t["target"]
    assert fx(sum_p, 2) == fx(e_p, 2) and fx(sum_o, 2) == fx(e_o, 2), (sum_p, e_p, sum_o, e_o)
    n.update(precfrac=fx(p["record"], 2), headtime=fx(head, 3), reachproc=fx(reach * p["process"], 3),
             reachout=fx(reach * p["outcome"], 3), reachboth=fx(reach * p["process"] * p["outcome"], 3),
             savewhole=pct((r(e_o, 2) - r(e_p, 2)) / r(e_o, 2), 0))
    assert n["savewhole"] == pct(save, 0)

    # ---- Your turn: the chapter's slower, less selective process audit
    #      (chapter checkpoint: Scenarios.DemonstrationAdmissionOrderSlowAudit)
    sv = SC.DemonstrationAdmissionOrderSlowAudit
    vres = sv.result
    pv, tv = cascade(sv)
    assert all(pv[c] == p[c] and tv[c] == t[c] for c in p if c != "process")   # one condition changed
    rej_pv = tv["process"] / (1 - pv["process"])
    near(rej_pv, vres.cost_per_rejection["process"].to(second).magnitude, 1e-9)
    assert rej_pv > rej_o                                             # order flips: outcome first
    gainv = r(reach, 3) * (tv["process"] * (1 - pv["outcome"]) - tv["outcome"] * (1 - pv["process"]))
    ev_p, ev_o = expected_time(pv, tv, pfirst), expected_time(pv, tv, ofirst)
    near(ev_p, vres.time_per_attempted[0].to(second).magnitude, 1e-9)
    near(gainv, ev_p - ev_o, 1e-9)
    yv = 1.0
    for c in pv:
        yv *= pv[c]
    near(yv, vres.route_yield, 1e-12)
    adm_v = r(ev_o, 2) / r(yv, 4)
    near(adm_v, ev_o / yv, 0.05)
    n.update(vprocp=pct(pv["process"]), vproct=fx(tv["process"], 0), vprocmiss=fx(1 - pv["process"], 2),
             vrejproc=fx(rej_pv, 0), vgain=fx(gainv, 3), veproc=fx(ev_p, 2), veout=fx(ev_o, 2),
             vsavepct=pct(r(gainv, 3) / r(ev_p, 2), 1), vyield=fx(yv, 4), vadm=fx(adm_v, 1))
    near(r(gainv, 3) / r(ev_p, 2), vres.saving, 0.001)
    # Finish: the outcome-first cascade sum, term by term from rounded reaches
    rv_out, rv_both = r(reach * pv["outcome"], 3), r(reach * pv["outcome"] * pv["process"], 3)
    sumv_o = r(head, 3) + r(reach, 3) * tv["outcome"] + rv_out * tv["process"] + rv_both * tv["target"]
    assert fx(sumv_o, 2) == fx(ev_o, 2), (sumv_o, ev_o)
    n.update(vreachboth=fx(reach * pv["outcome"] * pv["process"], 3),
             vsaveround=pct((r(ev_p, 2) - r(ev_o, 2)) / r(ev_p, 2), 1))
    assert n["vsaveround"] == n["vsavepct"]

    # ---- Mistake: staging sized from average verifier utilization
    #      (chapter: VerifierBurstExhaustion, Scenarios.VerifierTimeoutStagingBurst)
    vb = SC.VerifierTimeoutStagingBurst
    n.update(burstrho=pct(vb.result.utilization), burstfull=fx(vb.result.time_to_full.to("minute").magnitude, 1),
             bursttimeout=fx(vb.timeout.to(second).magnitude, 0))

    # ---- Mistake: accepted run, every token a target (chapter: RecoveryPoolDenominators)
    pool = SC.RecoveryPoolDenominators
    _, l_rec, u_rec = pool.classes["recovery"]
    l_rec, u_rec = l_rec.to(token).magnitude, u_rec.to(token).magnitude
    near(l_rec / u_rec, pool.result.context_per_target["recovery"], 1e-12)
    n.update(recstored=comma(l_rec), rectarget=comma(u_rec), rectargetpct=pct(u_rec / l_rec, 0))

    # ---- Mistake: fine-tune on every failure (chapter: InterventionOrderTradeoff, Scenarios.RepairAgentFailureTriage)
    tri = SC.RepairAgentFailureTriage.result
    n.update(failall=comma(tri.failures), failsystem=comma(tri.system_failures), failresidual=comma(tri.residual),
             systempct=pct(tri.system_share), costall=fx(tri.cost.to(USD).magnitude, 0),
             costres=fx(tri.residual_cost.to(USD).magnitude, 0))

    # ---- Mistake: drop target review to cut checking time (worked-example route, same inputs)
    y_drop = y / p["target"]
    e_drop = e_p - reach * p["process"] * p["outcome"] * t["target"]
    n.update(dropyield=fx(y_drop, 3), dropadm=fx(r(e_drop, 2) / r(y_drop, 3), 1), dropet=fx(e_drop, 2))
    near(r(e_drop, 2) / r(y_drop, 3), e_drop / y_drop, 0.05)

    # ---- Front: straddle curve at the chapter's split (chapter: RandomSplitLeakage, Scenarios.EightAttemptRunSplit)
    es = SC.EightAttemptRunSplit
    fch, kch = es.train_fraction, es.attempts
    near(1 - fch ** kch - (1 - fch) ** kch, es.result.straddle, 1e-12)
    n.update(chf=fx(fch, 1), chk=f"{kch}", chstraddle=pct(es.result.straddle, 0))

    # ---- Check yourself
    # Q1 ASSUMED: traces split at random, f_train = 0.9; three and then six attempts per task
    f, k1, k2 = 0.9, 3, 6
    s1 = 1 - f ** k1 - (1 - f) ** k1
    s2 = 1 - f ** k2 - (1 - f) ** k2
    n.update(qf=fx(f, 1), qfpct=pct(f), qtestpct=pct(1 - f), qkone=f"{k1}", qktwo=f"{k2}",
             qfkone=fx(f ** k1, 3), qmkone=fx((1 - f) ** k1, 3), qsone=pct(s1, 0),
             qfktwo=fx(f ** k2, 3), qsktwo=pct(s2, 0))
    near(1 - r(f ** k2, 3) - (1 - f) ** k2, s2, 0.001)

    # Q3 ASSUMED: four attempts per fixture, fixtures solved half the time and nine times in ten
    k3, pm, pe = 4, 0.5, 0.9
    ym, mm = 1 - (1 - pm) ** k3, 1 - (1 - pm) ** k3 - pm ** k3
    ye, me = 1 - (1 - pe) ** k3, 1 - (1 - pe) ** k3 - pe ** k3
    n.update(qkthree=f"{k3}", qpmid=fx(pm, 1), qpeasy=fx(pe, 1), qpmidpct=pct(pm), qpeasypct=pct(pe),
             qmidpow=fx(pm ** k3, 4), qeasypow=fx(pe ** k3, 4), qeasymisspow=fx((1 - pe) ** k3, 4),
             qymid=pct(ym, 1), qmmid=pct(mm, 1), qyeasy=pct(ye, 2), qmeasy=pct(me, 1))

    # Q4 ASSUMED: a demonstration pool of direct and recovery records
    nd, sd, wd = 9000, 5000, 1000
    nr, sr, wr = 3000, 20000, 2000
    vd, vr, td, tr = nd * sd, nr * sr, nd * wd, nr * wr
    n.update(qnd=comma(nd), qsd=comma(sd), qwd=comma(wd), qnr=comma(nr), qsr=comma(sr), qwr=comma(wr),
             qrecshare=pct(nr / (nd + nr)), qvd=fx(vd / 1e6, 0), qvr=fx(vr / 1e6, 0), qtd=fx(td / 1e6, 0),
             qtr=fx(tr / 1e6, 0), qstoredshare=pct(vr / (vd + vr), 0), qtargetshare=pct(tr / (td + tr)),
             qratiod=fx(sd / wd, 0), qratior=fx(sr / wr, 0),
             qntot=comma(nd + nr), qvtot=fx((vd + vr) / 1e6, 0), qttot=fx((td + tr) / 1e6, 0))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
