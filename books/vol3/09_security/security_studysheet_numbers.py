"""Numbers for the Agent Security study sheet (Volume III, Chapter 9).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(InjectionExposure, PrefixProbe, SecurityPolicyComparison), and written as \\setnum{key}{value}
to security_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC
from mlsysim.core.units import ms


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))   # round(.., 9) drops float noise


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def exposure(q, n):
    return 1 - (1 - q) ** n


def numbers() -> dict:
    n = {}
    n.update(one="1", three="3")                                     # formula constants printed in answers

    # ---- Front: exposure card and curve (chapter: InjectionExposure, Scenarios.InjectionExposureByReadVolume)
    ie = SC.InjectionExposureByReadVolume
    ires = ie.result
    q, miss, cohort = ie.payload_fraction, ie.miss_rate, ie.cohort
    reads_repair, reads_research = ie.reads["repair"], ie.reads["research"]
    exp_repair, exp_research = ires.exposure["repair"], ires.exposure["research"]
    near(exp_repair, exposure(q, reads_repair), 1e-12)
    near(exp_research, exposure(q, reads_research), 1e-12)
    screened_research = ires.screened["research"]
    near(screened_research, exposure(q * miss, reads_research), 1e-12)
    n.update(payload=pct(q), payloadfrac=fx(q, 2), readsrepair=f"{reads_repair}", readsresearch=f"{reads_research}",
             exprepair=pct(exp_repair, 1), expresearch=pct(exp_research, 1),
             lnclean=f"{math.log(1 - q):.7f}",                         # for the plotted curve only
             missrate=pct(miss), screenedresearch=pct(screened_research, 1),
             screenedcohort=fx(ires.screened_per_cohort["research"], 0), cohort=f"{cohort:,}")

    # ---- Key idea: shared prefix cache timing (chapter: PrefixProbe, Scenarios.PrefixCacheProbe2K)
    probe = SC.PrefixCacheProbe2K
    t_miss = probe.result.rebuild_time.to(ms).magnitude
    t_hit = probe.result.weight_read_time.to(ms).magnitude
    assert t_hit < t_miss / 10
    n.update(probeprefix=f"{probe.prefix_tokens:,}", probemiss=fx(t_miss, 0), probehit=fx(t_hit, 0))

    # ---- Worked example: does the release check pay? (chapter: SecurityPolicyComparison,
    #      Scenarios.FixtureReleasePolicyComparison, nbk-09-security-policy-comparison)
    pc = SC.FixtureReleasePolicyComparison
    pres = pc.result
    att, ben = pc.attacked, pc.benign
    a, b = pc.baseline, pc.candidate
    lo_a, lo_b = a["confirmed"] / att, b["confirmed"] / att
    up_a, up_b = (a["confirmed"] + a["unresolved"]) / att, (b["confirmed"] + b["unresolved"]) / att
    near(lo_a, pres.baseline_lower, 1e-12); near(lo_b, pres.candidate_lower, 1e-12)
    near(up_a, pres.baseline_upper, 1e-12); near(up_b, pres.candidate_upper, 1e-12)
    red_obs = 1 - lo_b / lo_a
    red_up = 1 - up_b / up_a
    red_lf = 1 - up_b / lo_a
    near(red_obs, pres.reduction_observed, 1e-12)
    near(red_up, pres.reduction_upper, 1e-12)
    near(red_lf, pres.reduction_least_favorable, 1e-12)
    acc_a, acc_b = a["accepted_benign"] / ben, b["accepted_benign"] / ben
    attempt = r(pc.attempt_cost, 2)
    near(attempt, pc.attempt_cost, 1e-9)
    ceff_a, ceff_b = attempt / acc_a, attempt / acc_b
    near(ceff_a, pres.baseline_cost_per_accepted, 1e-9)
    near(ceff_b, pres.candidate_cost_per_accepted, 1e-9)
    ratio = acc_a / acc_b
    near(ratio - 1, pres.cost_rise, 1e-12)
    n.update(attacked=f"{att}", benign=f"{ben}",
             confa=f"{a['confirmed']}", unresa=f"{a['unresolved']}", benigna=f"{a['accepted_benign']}",
             confb=f"{b['confirmed']}", unresb=f"{b['unresolved']}", benignb=f"{b['accepted_benign']}",
             falseblocka=f"{a['false_block']}", falseblockb=f"{b['false_block']}",
             lowa=pct(lo_a), lowb=pct(lo_b), upa=pct(up_a), upb=pct(up_b),
             upsuma=f"{a['confirmed'] + a['unresolved']}", upsumb=f"{b['confirmed'] + b['unresolved']}",
             redobserved=pct(red_obs), redupper=pct(red_up), redleast=pct(red_lf),
             attempt=fx(attempt, 2), acca=fx(acc_a, 2), accb=fx(acc_b, 2),
             ceffa=fx(ceff_a, 2), ceffb=fx(ceff_b, 2), costratio=fx(ratio, 3), costrise=pct(ratio - 1, 1))

    # ---- Your turn. ASSUMED: better endpoint logs leave only 2 of B's attacked episodes unresolved,
    #      with no new confirmed disclosure; A and both benign cohorts unchanged.
    b_unres_new = 2
    v = pc.variant(candidate=dict(b, unresolved=b_unres_new))
    vres = v.result
    up_b2 = (b["confirmed"] + b_unres_new) / att
    near(up_b2, vres.candidate_upper, 1e-12)
    red_up2, red_lf2 = 1 - up_b2 / up_a, 1 - up_b2 / lo_a
    near(red_up2, vres.reduction_upper, 1e-12)
    near(red_lf2, vres.reduction_least_favorable, 1e-12)
    near(vres.cost_rise, pres.cost_rise, 1e-12)
    n.update(unresnew=f"{b_unres_new}", upsumnew=f"{b['confirmed'] + b_unres_new}", upbnew=pct(up_b2),
             redupnew=pct(red_up2), redleastnew=pct(red_lf2))

    # ---- Mistake: zero breaches (chapter: SecurityPolicyComparison, Scenarios.CleanAttackSuite100)
    zs = SC.CleanAttackSuite100
    conf, zero_n = zs.confidence, zs.trials
    alpha = 1 - conf
    zero_ub = 1 - alpha ** (1 / zero_n)
    near(zero_ub, zs.result.rate_upper_exact, 1e-12)
    n.update(conf=pct(conf), alpha=fx(alpha, 2), zeron=f"{zero_n}", zeroub=pct(zero_ub, 1),
             zerorule=pct(3 / zero_n, 1))

    # ---- Check yourself
    # Q1 ASSUMED: an on-call agent reads 30 alerts and log lines at a 0.5 percent payload rate;
    #    a screen misses 10 percent of payloads.
    q1, n1, m1 = 0.005, 30, 0.10
    e1 = exposure(q1, n1)
    e1s = exposure(q1 * m1, n1)
    n.update(qpayload=pct(q1, 1), qpayloadfrac=fx(q1, 3), qcleanfrac=fx(1 - q1, 3), qreads=f"{n1}",
             qexp=pct(e1, 1), qlinear=pct(q1 * n1, 1), qmiss=pct(m1), qscreenfrac=fx(q1 * m1, 4),
             qscreencleanfrac=fx(1 - q1 * m1, 4), qexpscreen=pct(e1s, 1))

    # Q4 ASSUMED: zero disclosures in 120 attacked episodes; a release target of 2 percent;
    #    4 unresolved episodes the team had dropped.
    q4n, q4target, q4drop = 120, 0.02, 4
    ub4 = 1 - alpha ** (1 / q4n)
    need_exact = math.ceil(math.log(alpha) / math.log(1 - q4target))
    assert 1 - alpha ** (1 / need_exact) <= q4target < 1 - alpha ** (1 / (need_exact - 1))
    need_rule = 3 / q4target
    assert ub4 > q4target
    restored = q4drop / (q4n + q4drop)
    n.update(qtrials=f"{q4n}", qtarget=pct(q4target), qbound=pct(ub4, 1), qrule=pct(3 / q4n, 1),
             qneed=f"{need_exact}", qneedrule=fx(need_rule, 0), qdropped=f"{q4drop}",
             qrestoredn=f"{q4n + q4drop}", qrestored=pct(restored, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
