"""Numbers for the Responsible AI study sheet (Volume II, Chapter 16).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to responsible_ai_studysheet_numbers.tex next to this
file. Inputs the sheet chooses for its own questions are marked ASSUMED (built as variants of
the chapter's scenarios). A printed result is computed from the rounded values the sheet prints
beside it, and checked against the unrounded model value.

Units: GB are decimal (10^9 bytes); dollars are the chapter's scenario prices.
"""
import math
from pathlib import Path

from mlsysim import Scenarios as SC
from mlsysim.core.units import ureg


def fx(x, nd):
    return f"{x:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}

    # ---- Worked example + criteria table (chapter: FairnessWorkedExample, PpvGap,
    # FairnessMetricsTable, FairnessZTest; Scenarios.LoanScreeningTwoGroups, LoanApprovalGapTest)
    s = SC.LoanScreeningTwoGroups
    ga, gb = s.groups["A"], s.groups["B"]
    for g, c in (("a", ga), ("b", gb)):
        n.update({f"tp{g}": f"{c['tp']:.0f}", f"fp{g}": f"{c['fp']:.0f}",
                  f"fn{g}": f"{c['fn']:.0f}", f"tn{g}": f"{c['tn']:.0f}"})
    n.update(sizea=f"{s.size('A'):.0f}", sizeb=f"{s.size('B'):.0f}",
             appra=f"{s.positives('A'):.0f}", apprb=f"{s.positives('B'):.0f}",
             qualA=f"{ga['tp'] + ga['fn']:.0f}", qualB=f"{gb['tp'] + gb['fn']:.0f}",
             unqualA=f"{ga['fp'] + ga['tn']:.0f}", unqualB=f"{gb['fp'] + gb['tn']:.0f}")
    pa, pb = s.positive_rate("A"), s.positive_rate("B")
    tpra, tprb = s.tpr("A"), s.tpr("B")
    fpra, fprb = s.fpr("A"), s.fpr("B")
    ppva, ppvb = s.ppv("A"), s.ppv("B")
    di = min(pa, pb) / max(pa, pb)
    near(di, s.disparate_impact, 1e-12)
    four_fifths = 0.8                    # chapter: four-fifths screening convention (FairnessMetricsTable)
    assert di < four_fifths
    n.update(rateA=fx(pa, 2), rateB=fx(pb, 2), gapAB=fx(pa - pb, 2), di=fx(di, 2), fourfifths=fx(four_fifths, 1),
             tprA=fx(tpra, 2), tprB=fx(tprb, 2), tprgap=fx(round(tpra, 2) - round(tprb, 2), 2),
             fprA=fx(fpra, 2), fprB=fx(fprb, 2), ppvA=fx(ppva, 2), ppvB=fx(ppvb, 2))

    t = SC.LoanApprovalGapTest
    assert (t.rate_a, t.rate_b, t.n_per_group) == (pa, pb, s.size("A"))
    se = math.sqrt(t.pooled_rate * (1 - t.pooled_rate) * (2 / t.n_per_group))
    near(t.gap / se, t.z, 1e-9)
    near(round(t.gap, 2) / round(se, 4), round(t.z, 2), 0.005)   # printed 0.30 / 0.0704 -> printed 4.26
    n.update(pooled=fx(t.pooled_rate, 2), pooledc=fx(1 - t.pooled_rate, 2), nbig=f"{t.n_per_group}",
             inv=fx(2 / t.n_per_group, 2), se=fx(se, 4), z=fx(t.z, 2), zcrit=fx(t.z_critical, 2),
             signifpct=f"{100 * t.significance:.0f}")

    # z versus per-group n for this gap (front picture): z = coef * sqrt(n)
    coef = t.gap / math.sqrt(t.pooled_rate * (1 - t.pooled_rate) * 2)
    near(coef * math.sqrt(t.n_per_group), t.z, 1e-9)
    ncrit = t.n_at_critical
    near(coef * math.sqrt(ncrit), t.z_critical, 1e-6)
    n.update(zcoef=fx(coef, 5), ncrit=f"{math.ceil(ncrit)}")

    # ---- Your turn (chapter: "a 30-point approval gap observed on 20 loan applications";
    # LoanApprovalGapTest.small_n_per_group)
    ns = t.small_n_per_group
    se_s = math.sqrt(t.pooled_rate * (1 - t.pooled_rate) * (2 / ns))
    zs = t.z_at(ns)
    near(t.gap / se_s, zs, 1e-9)
    near(round(t.gap, 2) / round(se_s, 3), round(zs, 2), 0.005)
    assert zs < t.z_critical
    n.update(nsmall=f"{ns}", invsmall=fx(2 / ns, 1), sesmall=fx(se_s, 3), zsmall=fx(zs, 2))

    # ---- Mistake: one criterion met (chapter: tbl-responsible-ai-fairness-summary, FairnessGapCalc;
    # the four per-group percentages are the chapter's own typed table inputs)
    m = dict(a_appr=55, b_appr=40, a_tpr=90, b_tpr=60, a_fpr=20, b_fpr=20)
    assert m["a_fpr"] == m["b_fpr"]
    n.update(mfpr=f"{m['a_fpr']}", mapprgap=f"{m['a_appr'] - m['b_appr']}", mtprgap=f"{m['a_tpr'] - m['b_tpr']}")

    # ---- Mistake: automation bias (chapter: AutomationBias, Scenarios.RadiologyAutomationBias)
    r = SC.RadiologyAutomationBias
    miss = r.ai_miss_rate
    acc = r.miss_acceptance
    fail = r.accepted_misses()
    near(round(miss, 2) * round(acc, 1), fail, 1e-12)
    near(1 - fail, r.team_sensitivity_bound, 1e-12)
    n.update(human=f"{100 * r.human_sensitivity:.0f}", ai=f"{100 * r.ai_sensitivity:.0f}",
             accept=f"{100 * acc:.0f}", aimiss=fx(miss, 2), acceptfrac=fx(acc, 1), fail=fx(fail, 3),
             team=fx(100 * r.team_sensitivity_bound, 1), gainpts=fx(100 * r.gain_over_ai, 1))

    # ---- Capacity table, monitoring row (chapter: MonitoringScale, Scenarios.FairnessMonitor10K)
    f = SC.FairnessMonitor10K
    rec = f.record_size.to("byte").magnitude
    log_gb = f.log_per_day.to("GB").magnitude
    near(f.decisions_per_day * f.sample_share * rec / 1e9, log_gb, 1e-9)
    near(f.metrics * f.false_alarm_rate, f.false_alerts_per_interval, 1e-12)
    n.update(monrate=f"{f.request_rate.to('1/s').magnitude:,.0f}", mondaym=f"{f.decisions_per_day / 1e6:.0f}",
             monshare=f"{100 * f.sample_share:.0f}", monrec=f"{rec:.0f}", monlog=fx(log_gb, 1),
             mongroups=f"{f.groups}", monper=f"{f.metrics_per_group}", monmetrics=f"{f.metrics}",
             monfa=f"{100 * f.false_alarm_rate:.0f}", monalerts=fx(f.false_alerts_per_interval, 1))

    # Check yourself Q1, ASSUMED: twice the request rate, 2 percent logged, 80 groups
    q1 = f.variant(name="ASSUMED: larger decision service", description="study sheet question",
                   request_rate=20000 / ureg.second, sample_share=0.02, groups=80)
    q1rec = q1.record_size.to("byte").magnitude
    q1log = q1.log_per_day.to("GB").magnitude
    near(q1.decisions_per_day * q1.sample_share * q1rec / 1e9, q1log, 1e-9)
    near(q1.metrics * q1.false_alarm_rate, q1.false_alerts_per_interval, 1e-12)
    n.update(qrate=f"{q1.request_rate.to('1/s').magnitude:,.0f}", qshare=f"{100 * q1.sample_share:.0f}",
             qgroups=f"{q1.groups}", qdaym=f"{q1.decisions_per_day / 1e6:,.0f}",
             qrecm=fx(q1.decisions_per_day * q1.sample_share / 1e6, 1), qlog=fx(q1log, 1),
             qmetrics=f"{q1.metrics}", qalerts=f"{q1.false_alerts_per_interval:.0f}")

    # ---- Capacity table, unlearning row + mistake (chapter: UnlearningCostAnalysis,
    # Scenarios.Gpt3ShardedUnlearning)
    u = SC.Gpt3ShardedUnlearning
    full = u.full_cost.to("USD").magnitude
    near(full / u.shards, u.shard_cost.to("USD").magnitude, 1e-6)
    near(u.planning_days * 24 / u.shards, u.shard_time.to("hour").magnitude, 1e-9)
    n.update(fullm=fx(full / 1e6, 1), fulldays=f"{u.planning_days}", shards=f"{u.shards}",
             shardk=f"{u.shard_cost.to('USD').magnitude / 1e3:.0f}", shardh=fx(u.shard_time.to('hour').magnitude, 1),
             ugpus=f"{u.n_accelerators:,}", ureq=f"{u.requests_per_day:,}",
             utouched=f"{u.shards_touched_per_day:.0f}")

    # Check yourself Q2, ASSUMED: the same run split into 50 shards
    q2 = u.variant(name="ASSUMED: 50 shards", description="study sheet question", shards=50)
    near(round(full / 1e6, 1) * 1e6 / q2.shards, q2.shard_cost.to("USD").magnitude, 1e-6)
    near(q2.planning_days * 24 / q2.shards, q2.shard_time.to("hour").magnitude, 1e-9)
    assert q2.shards_touched_per_day > q2.shards - 0.5            # a day's batch touches every shard
    n.update(qshards=f"{q2.shards}", qshardk=f"{q2.shard_cost.to('USD').magnitude / 1e3:.0f}",
             qshardh=fx(q2.shard_time.to('hour').magnitude, 1), qdayh=f"{q2.planning_days * 24}",
             qbatchm=fx(q2.shards * q2.shard_cost.to('USD').magnitude / 1e6, 1))

    # ---- Mistake: explanations retrofitted (chapter: ExplainabilityRetrofitCost, Scenarios.InlineShapRetrofit)
    x = SC.InlineShapRetrofit
    base, lo, hi = (x.base_latency.to("ms").magnitude, x.added_low.to("ms").magnitude, x.added_high.to("ms").magnitude)
    near(base + lo, x.latency_low.to("ms").magnitude, 1e-9)
    near(base + hi, x.latency_high.to("ms").magnitude, 1e-9)
    near((base + lo) / base, x.capacity_low, 1e-9)       # capacity grows with latency at fixed utilization
    near((base + hi) / base, x.capacity_high, 1e-9)
    n.update(shapbase=f"{base:.0f}", shaplo=f"{lo:.0f}", shaphi=f"{hi:.0f}",
             shaprate=f"{x.request_rate.to('1/s').magnitude:,.0f}",
             shaptotlo=f"{base + lo:.0f}", shaptothi=f"{base + hi:.0f}",
             capplo=fx(x.capacity_low, 1), capphi=f"{x.capacity_high:.0f}")

    # ---- Capacity table, explanation row (chapter: exact Shapley subset enumeration, 2^n)
    feats = 20                                   # chapter's example: "2^20 ~ 1 million for twenty features"
    n.update(feats=f"{feats}", subsetsm=f"{2 ** feats / 1e6:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
