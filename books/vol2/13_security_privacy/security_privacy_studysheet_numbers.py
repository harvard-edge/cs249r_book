"""Numbers for the Security and Privacy study sheet (Volume II, Chapter 13).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to security_privacy_studysheet_numbers.tex next to this
file. Inputs the sheet chooses for its own questions are marked ASSUMED (built as variants of
the chapter's scenarios). A printed result is computed from the rounded values the sheet prints
beside it, and checked against the unrounded model value.

Units: prices in USD; the chapter's scenario inputs are constructed (it says so), not measured.
"""
import math
from pathlib import Path

from mlsysim import Scenarios as SC


def fx(x, nd):
    return f"{x:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def mag(q, unit):
    return q.to(unit).magnitude


def numbers() -> dict:
    n = {}

    # ---- Worked example: copying a ResNet-50 API under three billing regimes
    # (chapter: DefenseOverheadAnalysis, ExtractionBudgetRegimes; Scenarios.ResNet50ApiExtractionBudget)
    b = SC.ResNet50ApiExtractionBudget
    train = mag(b.training_cost, "USD")
    queries = b.extraction_queries
    price = mag(b.price_per_query, "USD")
    tier = mag(b.tier_price, "USD")
    quota = b.tier_daily_quota
    period = mag(b.billing_period, "day")
    allow = b.lifetime_allowance
    paygo = queries * price
    near(paygo, mag(b.full_set_pay_per_query, "USD"), 1e-9)
    near(paygo, train, 1e-9)                                     # pay-as-you-go is break-even
    days = queries / quota
    near(days, b.collection_days, 1e-9)
    tiercost = tier * days / period
    near(tiercost, mag(b.full_set_tier, "USD"), 1e-9)
    overage = (queries - allow) * price
    near(overage, mag(b.overage_cost, "USD"), 1e-9)
    allowfee = tier * (allow / quota) / period                   # tier fee while the allowance is collected
    near(allowfee, mag(b.allowance_fees, "USD"), 1e-9)
    life = overage + allowfee
    near(life, mag(b.full_set_lifetime, "USD"), 1e-9)
    near(round(overage) + round(allowfee), round(life), 1)       # printed parts add to the printed total
    n.update(wtrain=f"{train:,.0f}", wqueriesM=f"{queries / 1e6:.0f}", wfidelity=f"{100 * b.target_fidelity:.0f}",
             wprice=fx(price, 3), wtier=f"{tier:.0f}", wquota=f"{quota:,.0f}", wperiod=f"{period:.0f}",
             wallowK=f"{allow / 1e3:.0f}", wpaygo=f"{paygo:,.0f}", wdays=f"{days:.0f}",
             wtiercost=f"{tiercost:,.0f}", wtierpct=f"{100 * tiercost / train:.0f}",
             woverage=f"{overage:,.0f}", wallowfee=f"{allowfee:.0f}", wlife=f"{life:,.0f}",
             wlifepct=f"{100 * life / train:.0f}", wallowdays=f"{allow / quota:.0f}",
             # thousands of dollars, for the to-scale bars
             drawtrain=fx(train / 1e3, 3), drawpaygo=fx(paygo / 1e3, 3), drawtier=fx(tiercost / 1e3, 3),
             drawlife=fx(life / 1e3, 3), wtrainK=f"{train / 1e3:.0f}", wpaygoK=f"{paygo / 1e3:.1f}",
             wtierK=f"{tiercost / 1e3:.2f}", wlifeK=f"{life / 1e3:.2f}")

    # ---- Problem to finish: the chapter's general-purpose API (ExtractionDefenseReference:
    # $5M training, 100M extraction queries, basic tier 100,000 queries/day; chapter LOAD inputs).
    big_train, big_queries, basic_quota = 5e6, 100e6, 100_000
    big_break = big_train / big_queries
    near(big_break, 0.05, 1e-12)                                 # chapter guard: $0.05 per query
    basic_fee = 500                                              # ASSUMED: basic tier's flat fee, USD/month
    fdays = big_queries / basic_quota
    fbill = basic_fee * fdays / period
    n.update(ftrainM=f"{big_train / 1e6:.0f}", fqueriesM=f"{big_queries / 1e6:.0f}", fquota=f"{basic_quota:,}",
             fbreak=fx(big_break, 2), ffee=f"{basic_fee}", fdays=f"{fdays:,.0f}", fbill=f"{fbill:,.0f}",
             fpct=fx(100 * fbill / big_train, 1))

    # ---- Laplace card (chapter: DPCostAnalysis, Scenarios.SalaryMeanLaplaceRelease)
    s = SC.SalaryMeanLaplaceRelease
    sens = mag(s.sensitivity, "USD")
    scale = sens / s.epsilon
    near(scale, mag(s.noise_scale, "USD"), 1e-9)
    err = scale / s.group_size
    near(err, mag(s.mean_error(), "USD"), 1e-9)
    n.update(salrange=f"{sens:,.0f}", saleps=f"{s.epsilon:.0f}", salgroup=f"{s.group_size:,}",
             salscale=f"{scale:,.0f}", salerr=f"{err:,.0f}")

    # Check yourself Q1, ASSUMED: half the budget for half the group, released monthly for a year
    q1 = s.variant(name="ASSUMED: salary mean at epsilon 0.5 for 500 people", description="study sheet question",
                   epsilon=0.5, group_size=500)
    qscale = sens / q1.epsilon
    qerr = qscale / q1.group_size
    near(qerr, mag(q1.mean_error(), "USD"), 1e-9)
    months = 12                                                  # ASSUMED: one release a month for a year
    total = months * q1.epsilon                                  # basic composition
    n.update(qeps=fx(q1.epsilon, 1), qgroup=f"{q1.group_size}", qscale=f"{qscale:,.0f}", qerr=f"{qerr:,.0f}",
             qratio=f"{qerr / err:.0f}", qmonths=f"{months}", qtotal=f"{total:.0f}")

    # ---- Membership card and Q3 (chapter: MiaAuditDpBound, Scenarios.MembershipAuditAtEpsilonOne)
    m = SC.MembershipAuditAtEpsilonOne
    eps = m.epsilon
    knee = 1 / (1 + math.exp(eps))                               # where the two ceiling branches meet
    near(m.ceiling(knee), math.exp(eps) * knee, 1e-9)
    near(m.ceiling(knee), 1 - math.exp(-eps) * (1 - knee), 1e-9)
    fpr, tpr = 0.10, 0.30                                        # ASSUMED: an audit's best operating point
    ceil = m.ceiling(fpr)
    near(ceil, math.exp(eps) * fpr, 1e-12)                       # the e^eps branch binds below the knee
    assert tpr > ceil and m.audit(fpr) <= ceil
    near(round(math.exp(eps), 2) * fpr * 100, round(100 * ceil, 1), 0.05)   # printed 2.72 x 10% -> printed 27.2%
    n.update(epsone=f"{eps:.0f}", eone=fx(math.exp(eps), 2), knee=fx(knee, 2),
             auditfpr=f"{100 * fpr:.0f}", audittpr=f"{100 * tpr:.0f}", ceilpct=fx(100 * ceil, 1),
             # drawing: audit curve shape a in TPR = FPR + a FPR (1 - FPR)
             auditshape=fx(m.audit_shape, 2), expeps=fx(math.exp(eps), 4))

    # ---- C3 price table and Q2 (chapter: ReferenceClusterAttackSurface, Scenarios.ReferenceJobAttackSurface)
    r = SC.ReferenceJobAttackSurface
    dp = r.dp_degree
    ring = 2 * (dp - 1) / dp
    gather = dp - 1
    near(ring, r.allreduce_factor, 1e-12)
    near(gather, r.gather_factor, 1e-12)
    near(gather / ring, r.gather_traffic_ratio, 1e-12)
    near(gather / ring, dp / 2, 1e-12)
    q2 = r.variant(name="ASSUMED: the same job with 16 data-parallel replicas", description="study sheet question",
                   dp_degree=16)
    near(q2.gather_traffic_ratio, q2.dp_degree / 2, 1e-12)
    n.update(qdp=f"{q2.dp_degree}", qgatherratio=f"{q2.gather_traffic_ratio:.0f}")
    cipher_share = 0.5                                           # chapter: a cipher that sustains half the link rate
    n.update(gpus=f"{r.accelerators:,}", nodes=f"{r.nodes}", dp=f"{dp}", gradgb=f"{mag(r.gradient_bytes, 'GB'):.0f}",
             ringfactor=fx(ring, 2), gatherfactor=f"{gather}", gatherratio=f"{r.gather_traffic_ratio:.0f}",
             cipherfactor=f"{1 / cipher_share:.0f}")

    # ---- Q1(c): the chapter's DP-SGD run (PracticalDPBudget, Scenarios.Cifar10DpSgdRun) states its target epsilon
    run = SC.Cifar10DpSgdRun
    n.update(sgdeps=f"{run.target_epsilon:.0f}")

    # ---- Mistake: federated learning (chapter: secure aggregation needs at least two noncolluding clients)
    n.update(noncolluding="two")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
