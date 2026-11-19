"""Numbers for the ML Operations study sheet (Volume I, Chapter 14).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(OpsRunningCases, RankingDriftScenario, RetrainingInterval, RetrainContention,
ReleaseControlFootprint, DriftDetectionDelay), and written as \\setnum{key}{value} to
ml_ops_studysheet_numbers.tex next to this file. Inputs the sheet chooses for its own examples
and questions are marked ASSUMED. Results are computed from the rounded values the sheet prints
beside them, so a student can reproduce each one.
"""
import math
from pathlib import Path

from mlsysim import Ops, Scenarios


def c(x):
    return f"{x:,.0f}"


def numbers() -> dict:
    n = {}

    # ---- Ranking service drift (chapter: RankingDriftScenario): A(t) = A0 exp(-gamma t)
    drift = Scenarios.RankingServiceAccuracyDrift
    a0, a28 = drift.initial_accuracy, drift.later_accuracy
    days = drift.elapsed.to("day").magnitude
    gamma = math.log(a0 / a28) / days
    gamma_p = float(f"{gamma:.3g}")                      # printed value; later steps use it
    n.update(accstart=f"{a0:.2f}", accweek=f"{a28:.2f}", slidedays=c(days),
             ratio=f"{a0 / a28:.4f}", lnratio=f"{math.log(a0 / a28):.4f}", gamma=f"{gamma_p}",
             gammapct=f"{gamma_p * 100:.2f}")

    # ---- Optimal interval (chapter: RetrainingInterval): T* = sqrt(2C / (Q V A0 gamma))
    iv = Scenarios.RankingRetrainingInterval
    run = iv.run_cost.to("USD").magnitude
    q = iv.query_rate.to("1/day").magnitude
    v = iv.value_per_query.to("USD").magnitude
    slope = q * v * float(f"{a0:.2f}") * gamma_p      # $ per day^2 of staleness growth
    slope_p = round(slope)
    tstar = math.sqrt(2 * run / slope_p)
    n.update(queries=c(q), qmillion=f"{q / 1e6:.0f}", value=f"{v:.2f}", runcost=c(run),
             slope=c(slope_p), tstar=f"{tstar:.1f}", tstarfour=f"{math.sqrt(2 * run / (4 * slope_p)):.1f}",
             costatstar=c(run / tstar), totalatstar=c(2 * run / tstar))
    # plot inputs (plain numbers for TikZ)
    n.update(two=c(2))                                   # the 2 in T* = sqrt(2C / ...)
    n.update(plotslope=f"{slope_p}", plotrun=f"{run:.0f}", plottstar=f"{tstar:.3f}")

    # ---- Your turn: accelerator share of the run (chapter: RetrainContention)
    rk = Scenarios.DlrmReleaseFootprintH100
    accels = rk.retrain_accelerators
    hours = rk.retrain_time.to("hour").magnitude
    price = iv.accelerator_price.to("USD/hour").magnitude
    compute = accels * hours * price
    run_a = run - compute / 2                            # option A: training twice as fast
    run_b = 1250.0                                       # ASSUMED option B: automated validation
    n.update(runaccels=c(accels), runhours=c(hours), accelhours=c(accels * hours), price=c(price),
             compute=c(compute), computepct=f"{compute / run * 100:.1f}", halfsave=c(compute / 2),
             runa=c(run_a), runb=c(run_b), tstara=f"{math.sqrt(2 * run_a / slope_p):.2f}",
             tstarb=f"{math.sqrt(2 * run_b / slope_p):.1f}", runratio=f"{run / run_b:.0f}",
             tratio=f"{math.sqrt(run / run_b):.0f}")

    # ---- Release memory (chapter: OpsRunningCases, ReleaseControlFootprint)
    copy_gb = rk.copy_bytes.to("GB").magnitude
    hbm_gib = rk.device.memory.capacity.to("GiB").magnitude
    hbm_gb = rk.device.memory.capacity.to("GB").magnitude
    per_copy = rk.accelerators_per_copy
    node = rk.node_accelerators
    free = node - per_copy - accels
    n.update(tablegb=c(copy_gb), hbmgib=c(hbm_gib), hbmgb=f"{hbm_gb:.1f}", percopy=c(per_copy),
             twocopies=c(rk.accelerators_for(2)), canarypct=c(rk.canary_share * 100),
             canaryshare=c(1 / rk.canary_share), node=c(node), freeaccels=c(free),
             freecopies=c(free // per_copy), busyaccels=c(per_copy + accels))

    # ---- Detection delay (chapter: DriftDetectionDelay): time = evidence budget / labeled rate
    conf = Scenarios.LabeledOutcomeDriftConfirmation
    budget = conf.evidence_budget
    hi = conf.label_rates["high"].to("1/s").magnitude
    lo = conf.label_rates["low"].to("1/day").magnitude
    fraud_rate = 40                                      # ASSUMED Q1: chargeback labels per day
    n.update(budget=c(budget), ratehigh=c(hi), minuteshigh=f"{budget / hi / 60:.0f}",
             ratelow=c(lo), dayslow=c(budget / lo), fraudrate=c(fraud_rate),
             frauddays=c(budget / fraud_rate))

    # ---- PSI bands (chapter: Ops.Monitoring thresholds)
    n.update(psiwarn=f"{Ops.Monitoring.PsiWarnThreshold:.1f}", psicrit=f"{Ops.Monitoring.PsiCriticalThreshold:.2f}")

    # ---- Chapter case-study constants (typed in the chapter's prose, not in mlsysim)
    n.update(sessiontrain=c(12), sessionserve=c(45),     # training-serving skew case study (minutes)
             ctrdrop=c(12))                              # incident-response example (percent CTR drop)

    # ---- Q4: ASSUMED ML Test Score suite scores (each out of 7)
    suites = {"data": 5, "model": 7, "infra": 4, "mon": 1}
    n.update(suitemax=c(7), **{f"suite{k}": c(s) for k, s in suites.items()},
             suitemin=c(min(suites.values())))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
