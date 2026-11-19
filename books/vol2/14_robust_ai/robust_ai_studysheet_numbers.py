"""Numbers for the Robust AI study sheet (Volume II, Chapter 14).

Every value the sheet prints is computed here from mlsysim scenarios and the chapter's data CSV, as
in the chapter's LEGO cells, and written as \\setnum{key}{value} to robust_ai_studysheet_numbers.tex
next to this file. Inputs the sheet chooses for its own questions are marked ASSUMED (built as
variants of the chapter's scenarios where one exists). A printed result is computed from the
rounded values the sheet prints beside it, and checked against the unrounded model value.
"""
import math
from pathlib import Path

from mlsysim import Ops, load_series
from mlsysim import Scenarios as SC
from mlsysim.physics import calc_ks_coefficient


def fx(x, nd):
    return f"{x:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}

    # ---- PSI card + worked example (chapter: CountryPSIDecomposition, Scenarios.FraudCountryPsi)
    p = SC.FraudCountryPsi
    usa_b, usa_c = p.shares("USA")
    oth_b, oth_c = p.shares("Other")
    t_usa, t_oth = p.term("USA"), p.term("Other")
    t_rest = p.psi - t_usa - t_oth
    near(t_usa + t_oth + t_rest, p.psi, 1e-12)
    # printed terms reproduce the printed total
    near(round(t_oth, 3) + round(t_usa, 3) + round(t_rest, 3), round(p.psi, 3), 1e-9)
    near((oth_c - oth_b) * round(math.log(oth_c / oth_b), 3), t_oth, 1e-4)
    warn, crit = float(Ops.Monitoring.PsiWarnThreshold), float(Ops.Monitoring.PsiCriticalThreshold)
    assert p.psi < warn
    n.update(usab=fx(usa_b, 2), usac=fx(usa_c, 2), othb=fx(oth_b, 2), othc=fx(oth_c, 2),
             othdiff=fx(oth_c - oth_b, 2), othlog=fx(math.log(oth_c / oth_b), 3), othterm=fx(t_oth, 3),
             usaterm=fx(t_usa, 3), restterm=fx(t_rest, 3), psitotal=fx(p.psi, 3),
             othgrowth=fx(oth_c / oth_b, 1), othshare=f"{100 * t_oth / p.psi:.0f}",
             psiwarn=fx(warn, 2), psicrit=fx(crit, 2))

    # ---- KS card + worked example (chapter: ContinuousDriftAnalysis, Scenarios.FraudAmountKs)
    k = SC.FraudAmountKs
    near(k.coefficient, calc_ks_coefficient(k.significance), 1e-12)
    near(k.coefficient * k.scale, k.critical_value, 1e-12)
    near(round(k.coefficient, 2) * round(k.scale, 4), k.critical_value, 5e-4)
    assert k.observed_statistic > k.critical_value
    n.update(ksalpha=fx(k.significance, 2), kscoef=fx(k.coefficient, 2), ksn=f"{k.n_reference:,}",
             ksscale=fx(k.scale, 4), kscrit=fx(k.critical_value, 3), ksd=fx(k.observed_statistic, 3),
             ksratio=fx(k.observed_statistic / k.critical_value, 1),
             kscoeffive=fx(float(Ops.Monitoring.KsTestCoefficient), 2))

    # Problem to finish, ASSUMED: a daily window of 1,000 against 1,000 with a smaller observed gap
    y = k.variant(name="ASSUMED: daily window", description="study sheet problem",
                  n_reference=1000, n_current=1000, observed_statistic=0.060)
    near(round(y.coefficient, 2) * round(y.scale, 4), y.critical_value, 5e-4)
    assert y.observed_statistic < y.critical_value
    n.update(yn=f"{y.n_reference:,}", yd=fx(y.observed_statistic, 3), yscale=fx(y.scale, 4),
             ycrit=fx(y.critical_value, 3))

    # ---- Defense-cost card (chapter: AdversarialPayback, Scenarios.PgdSevenTraining; FGSM listing)
    g = SC.PgdSevenTraining
    assert g.passes == g.attack_passes + g.update_passes
    fgsm_passes = 1 + 2 * g.update_passes        # one attack pass, then one update over a doubled batch
    n.update(pgdk=f"{g.attack_passes}", pgdpasses=f"{g.passes}", fgsmpasses=f"{fgsm_passes}")
    # Check yourself Q3a, ASSUMED: the team moves to PGD-10 on a run that took 5,000 accelerator-hours
    g10 = g.variant(name="ASSUMED: PGD-10", description="study sheet question", attack_passes=10)
    base_hours = 5000
    n.update(qk=f"{g10.attack_passes}", qpasses=f"{g10.passes}", qbase=f"{base_hours:,}",
             qhours=f"{g10.passes * base_hours:,}")

    # ---- Certified radius (chapter: RandomizedSmoothingRadius, Scenarios.SmoothingRadiusCifar10)
    s = SC.SmoothingRadiusCifar10
    near(s.noise_sd * s.quantile, s.radius, 1e-12)
    near(round(s.noise_sd, 1) * round(s.quantile, 2), s.radius, 5e-3)
    near(s.linf_budget * math.sqrt(s.coordinates), s.box_corner_l2, 1e-12)
    near(round(s.linf_budget, 4) * round(math.sqrt(s.coordinates), 1), s.box_corner_l2, 5e-3)
    assert not s.covers_box and s.radius < s.box_corner_l2
    n.update(rsp=fx(s.top_class_probability, 3), rssigma=fx(s.noise_sd, 1), rsq=fx(s.quantile, 2),
             rsr=fx(s.radius, 2), rslevels=f"{s.linf_levels}", rspix=f"{s.pixel_levels - 1}",
             rseps=fx(s.linf_budget, 4), rsd=f"{s.coordinates:,}", rssqrtd=fx(math.sqrt(s.coordinates), 1),
             rscorner=fx(s.box_corner_l2, 2))

    # ---- Mistake: robustness penalty (chapter: RobustnessTaxAnalysis; data/imagenet_testbed_shift.csv)
    rows = {r["model"]: r for r in load_series("vol2/14_robust_ai/data/imagenet_testbed_shift.csv").entries}
    std_clean = rows["resnet50"]["val"]
    rob_clean = rows["resnet50_linf_eps4_robust"]["val"]   # same testbed: l-infinity 4/255 adversarially trained
    rob_attacked = 50.0                                     # chapter's scenario input (RobustnessTaxAnalysis)
    n.update(robeps="4/255",                              # budget of the testbed row (linf_eps4) and the chapter's scenario
             stdclean=f"{std_clean:.0f}", robattacked=f"{rob_attacked:.0f}",
             crossgap=f"{round(std_clean) - round(rob_attacked):.0f}", robclean=f"{rob_clean:.0f}",
             cleanpen=f"{round(std_clean) - round(rob_clean):.0f}")

    # ---- Check yourself Q1, ASSUMED: fallback load at two thresholds against fallback capacity
    lam, cov_lo, cov_hi, cap = 2000, 0.95, 0.90, 150    # requests/s, coverages, fallback requests/s
    load_lo, load_hi = (1 - cov_lo) * lam, (1 - cov_hi) * lam
    assert load_lo <= cap < load_hi
    n.update(qlam=f"{lam:,}", qcovlo=f"{100 * cov_lo:.0f}", qcovhi=f"{100 * cov_hi:.0f}", qcap=f"{cap}",
             qloadlo=f"{load_lo:.0f}", qloadhi=f"{load_hi:.0f}", qover=f"{load_hi - cap:.0f}",
             qrejlo=f"{100 * (1 - cov_lo):.0f}", qrejhi=f"{100 * (1 - cov_hi):.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
