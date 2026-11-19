"""Numbers for the ML Operations at Scale study sheet (Volume II, Chapter 12).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to ops_scale_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED (built as variants of the
chapter's scenarios). A printed result is computed from the rounded values the sheet prints
beside it, and checked against the unrounded model value.
"""
from pathlib import Path

from mlsysim import Infrastructure
from mlsysim import Scenarios as SC
from mlsysim.core.units import USD, hour, minute


def fx(x, nd):
    return f"{x:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}

    # ---- Experiment card + worked example (chapter: AbSampleSize, CtrLiftExperiment;
    # CanaryDuration, CanaryGuardrailSchedule)
    e = SC.CtrLiftExperiment
    g = SC.CanaryGuardrailSchedule
    rate = g.request_rate.to("1/hour").magnitude
    zsq = (e.z_alpha + e.z_beta) ** 2
    near(zsq, e.z_sum_squared, 1e-12)
    near(zsq * e.two_variance / e.lift ** 2, e.per_variant, 1e-6)
    near(e.per_variant, 29_792, 0.5)                       # the chapter prints 29,792
    share_low, share_mid = g.stages[0][0], g.stages[1][0]   # 1 and 5 percent stages
    t_mid = e.per_variant / (rate * share_mid)              # hours
    t_guard = g.stage_minimum(share_mid).to("minute").magnitude
    near(t_guard, 12, 1e-9)                                 # the chapter prints 12 minutes
    near(g.stage_minimum(share_low).to("hour").magnitude, 1, 1e-9)
    n.update(zalpha=fx(e.z_alpha, 2), zbeta=fx(e.z_beta, 2), zsquared=fx(zsq, 2),
             baseline=fx(e.baseline_rate, 2), baselinepct=f"{100 * e.baseline_rate:.0f}",
             lift=fx(e.lift, 3), liftpoints=fx(100 * e.lift, 1),
             liftrel=f"{100 * e.lift / e.baseline_rate:.0f}",
             twovar=fx(e.two_variance, 3), liftsq=f"{e.lift ** 2:.6f}",
             samples=f"{e.per_variant:,.0f}",
             ratemillion=f"{rate / 1e6:.0f}", stagelowpct=f"{100 * share_low:.0f}",
             stagemidpct=f"{100 * share_mid:.0f}", stagemidrate=f"{rate * share_mid:,.0f}",
             stagemidh=fx(t_mid, 2), stagemidmin=f"{t_mid * 60:.0f}",
             guardsamples=f"{g.samples_per_variant:,.0f}", guardmin=f"{t_guard:.0f}",
             guardratio=fx(e.per_variant / g.samples_per_variant, 1))

    # ---- Problem to finish: the chapter's second case (1 percent baseline, 5 percent relative lift)
    f = e.variant(name="chapter case: rare outcome, small lift", description="study sheet problem",
                  baseline_rate=0.01, lift=0.0005)
    near(f.per_variant, 620_928, 0.5)                       # the chapter prints 620,928
    numer = zsq * f.two_variance
    f_mid = f.per_variant / (rate * share_mid)
    f_all = 2 * f.per_variant / rate                    # both arms on an even split of all traffic
    near(round(numer, 6) / f.lift ** 2, f.per_variant, 1e-6)
    n.update(fbaseline=fx(f.baseline_rate, 2), fbaselinepct=f"{100 * f.baseline_rate:.0f}",
             flift=fx(f.lift, 4), fliftrel=f"{100 * f.lift / f.baseline_rate:.0f}",
             ftwovar=fx(f.two_variance, 4), fnumer=fx(numer, 6), fliftsq=f"{f.lift ** 2:.8f}",
             fsamples=f"{f.per_variant:,.0f}", fmidh=fx(f_mid, 1), fallmin=f"{f_all * 60:.0f}",
             fratio=f"{f.per_variant / e.per_variant:.0f}")

    # ---- False-alert card, curve (chapter: FalseAlarmTax, ThreeSigmaModelFleetAlerts)
    a3 = SC.ThreeSigmaModelFleetAlerts
    near(a3.monitors * a3.checks_per_day * a3.false_positive_rate, a3.daily_false_alerts, 1e-9)
    any3 = a3.any_false_alert(a3.monitors)
    n.update(noalert=fx(1 - a3.false_positive_rate, 3), fprate=fx(a3.false_positive_rate, 3), monitors=f"{a3.monitors:,}",
             anyround=f"{100 * any3:.0f}", anyroundfrac=fx(any3, 4))

    # Check yourself Q1, ASSUMED: a smaller fleet, checked less often, same 3-sigma thresholds
    q1 = a3.variant(name="ASSUMED: 50 models x 8 metrics every 10 minutes", description="study sheet question",
                    models=50, metrics_per_model=8, check_interval=10 * minute)
    q1any = q1.any_false_alert(q1.monitors)
    near(q1.monitors * q1.checks_per_day * a3.false_positive_rate, q1.daily_false_alerts, 1e-9)
    n.update(qmodels=f"{q1.models}", qmetrics=f"{q1.metrics_per_model}",
             qinterval=f"{q1.check_interval.to('minute').magnitude:.0f}", qmonitors=f"{q1.monitors}",
             qchecks=f"{q1.checks_per_day:.0f}", qdaily=f"{q1.daily_false_alerts:.0f}",
             qany=f"{100 * q1any:.0f}", qnone=f"{100 * (1 - q1any):.0f}")

    # ---- Mistake: loosen to 2 sigma (chapter: TwoSigmaAlerts, TwoSigmaModelFleetAlerts)
    a2 = SC.TwoSigmaModelFleetAlerts
    n.update(twomodels=f"{a2.models}", twometrics=f"{a2.metrics_per_model}",
             twointerval=f"{a2.check_interval.to('minute').magnitude:.0f}",
             twofprate=fx(a2.false_positive_rate, 2), twodaily=f"{a2.daily_false_alerts:,.0f}",
             dedup=f"{100 * a2.dedup_share:.0f}", twoleft=f"{a2.remaining_after_dedup:.0f}",
             threedaily=f"{a3.daily_false_alerts:.0f}")

    # ---- Platform card (chapter: PlatformRoi, SharedPlatformBenefit50 / 20)
    p50, p20 = SC.SharedPlatformBenefit50, SC.SharedPlatformBenefit20
    hrs = p50.hours_saved.to("hour/month").magnitude
    eng = p50.engineer_rate.to(USD / hour).magnitude
    cost = p50.platform_cost.to("USD/month").magnitude
    near(p50.models * hrs * eng / cost, p50.benefit_cost_ratio, 1e-12)
    breakeven = cost / (hrs * eng)
    n.update(pmodels=f"{p50.models}", phours=f"{hrs:.0f}", prate=f"{eng:.0f}", pcostk=f"{cost / 1e3:.0f}",
             pbcr=fx(p50.benefit_cost_ratio, 2), plowmodels=f"{p20.models}", plowbcr=fx(p20.benefit_cost_ratio, 2),
             pbreakeven=f"{breakeven:.0f}")

    # Check yourself Q2, ASSUMED: the same platform at 30 models
    q2 = p50.variant(name="ASSUMED: the same platform at 30 models", description="study sheet question", models=30)
    q2benefit = q2.gross_benefit.to("USD/month").magnitude
    near(q2benefit / cost, q2.benefit_cost_ratio, 1e-12)
    perteam = cost / q2.models / eng                       # hours per team per month to break even at 30 models
    n.update(qpmodels=f"{q2.models}", qpbenefitk=f"{q2benefit / 1e3:.0f}", qpbcr=fx(q2.benefit_cost_ratio, 2),
             qploss=f"{100 * (1 - q2.benefit_cost_ratio):.0f}", qpperteam=fx(perteam, 1),
             qpmonthlyk=f"{hrs * eng / 1e3:.0f}")

    # ---- Mistake: load grows linearly (chapter: PlatformEconomics fallacy inputs; pairs N(N-1)/2)
    hours_per_model_month, fallacy_models = 40, 40          # chapter LEGO inputs (PlatformEconomics)
    rate_labor = Infrastructure.Pricing.Labor.MlEngineerLoadedHourly.rate.to(USD / hour).magnitude
    monthly = fallacy_models * hours_per_model_month
    annual = monthly * 12 * rate_labor
    pair_models = 100
    n.update(linmodels=f"{fallacy_models}", linhours=f"{hours_per_model_month}", linmonthly=f"{monthly:,}",
             linannualm=fx(annual / 1e6, 2), pairmodels=f"{pair_models}",
             pairs=f"{pair_models * (pair_models - 1) // 2:,}", tenpairs=f"{10 * 9 // 2}")

    # ---- Mistake: owning is always cheaper at scale (chapter: TcoScenario, OwnVsRentDgxH100Node;
    # TenKGpuClusterTco, OwnVsRentH100Cluster10K)
    own1, own10k = SC.OwnVsRentDgxH100Node, SC.OwnVsRentH100Cluster10K
    assert own1.rent_rate == own10k.rent_rate
    n.update(rentrate=f"{own1.rent_rate.to(USD / hour).magnitude:.0f}",
             nodebreak=f"{100 * own1.breakeven_utilization:.1f}", bigcluster=f"{own10k.gpus:,}",
             bigbreak=f"{100 * own10k.breakeven_utilization:.0f}")

    # ---- Budget vector card picture (chapter: fig-ops-scale-budget-vector, illustrative values)
    vec = {"avail": 0.40, "latency": 0.60, "fresh": 1.30, "quality": 0.50}
    mean = sum(vec.values()) / len(vec)
    near(mean, 0.70, 1e-12)                                 # the caption prints 0.70
    n.update({f"b{k}": fx(v, 2) for k, v in vec.items()})
    n.update(bmean=fx(mean, 2), bmax=fx(max(vec.values()), 2))

    # Check yourself Q3, ASSUMED: a vector whose quality budget is overspent
    allowed = {"avail": 0.001, "latency": 0.01, "fresh": 0.02, "quality": 0.05}   # allowed violating fractions
    observed = {"avail": 0.0002, "latency": 0.009, "fresh": 0.008, "quality": 0.06}
    used = {k: observed[k] / allowed[k] for k in allowed}
    qmean = sum(used.values()) / len(used)
    n.update({f"qv{k}": f"{100 * observed[k]:g}" for k in observed})
    n.update({f"qa{k}": f"{100 * allowed[k]:g}" for k in allowed})
    n.update({f"qb{k}": fx(v, 2) for k, v in used.items()})
    n.update(qbmean=fx(round(qmean, 6), 3))
    assert max(used, key=used.get) == "quality" and used["quality"] > 1 > qmean
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
