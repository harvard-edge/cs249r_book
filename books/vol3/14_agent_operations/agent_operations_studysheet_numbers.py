"""Numbers for the Operating Agents study sheet (Volume III, Chapter 14).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(ObservedCriticalPath, HeadSamplingMiss, DiagnosticCounts, SizingCanaryTrafficCircuitBreaker,
CanaryExcessFailures), and written as \\setnum{key}{value} to
agent_operations_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from statistics import NormalDist
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC


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


def numbers() -> dict:
    n = {}

    # ---- Front picture: spans and the critical path (chapter: ObservedCriticalPath)
    tr = SC.RepairTrialSpanTimeline
    res = tr.result
    iv = {k: (a.to("s").magnitude, b.to("s").magnitude) for k, (a, b) in tr.intervals.items()}
    dur = {k: b - a for k, (a, b) in iv.items()}
    gap = iv["synthesize"][0] - max(iv["tests"][1], iv["lint"][1])
    leaf = sum(dur.values())
    crit = dur["retrieve"] + dur["propose"] + max(dur["tests"], dur["lint"]) + dur["synthesize"]
    wall = iv["synthesize"][1] - iv["retrieve"][0]
    near(leaf, res.leaf_sum.to("s").magnitude, 1e-9)
    near(crit, res.critical_path.to("s").magnitude, 1e-9)
    near(gap, res.unexplained.to("s").magnitude, 1e-9)
    near(wall, res.elapsed.to("s").magnitude, 1e-9)
    fast = tr.intervention["tests"].to("s").magnitude
    saving = max(dur["tests"], dur["lint"]) - max(fast, dur["lint"])
    near(saving, res.saving.to("s").magnitude, 1e-9)
    n.update({f"d{k}": fx(v, 1) for k, v in dur.items()})
    n.update({f"s{k}": f"{iv[k][0]:.2f}" for k in iv})           # picture positions only
    n.update(gap=fx(gap, 1), leaf=fx(leaf, 1), crit=fx(crit, 1), wall=fx(wall, 1),
             fasttests=fx(fast, 0), testcut=fx(dur["tests"] - fast, 0), saving=fx(saving, 0))

    # ---- Mistake: head sampling and rare failures (chapter: HeadSamplingMiss)
    rare, common = SC.RareFailureUnderHeadSampling, SC.RecurringFailureUnderHeadSampling
    theta = rare.retention_probability
    miss_rare = (1 - theta) ** rare.occurrences
    near(miss_rare, rare.miss, 1e-12)
    miss_common = (1 - theta) ** common.occurrences
    near(miss_common, common.miss, 1e-12)
    n.update(theta=pct(theta), thetafrac=fx(1 - theta, 1), rarecount=f"{rare.occurrences}",
             missrare=pct(miss_rare), commoncount=f"{common.occurrences}", misscommon=pct(miss_common, 1))

    # ---- Mistake: a clean diagnostic run (chapter: DiagnosticCounts, Wilson upper bound)
    br = SC.InterfaceRepairBranches
    runs = br.runs("paginated")
    assert br.accepted("paginated") == runs
    conf = 2 * NormalDist().cdf(br.z) - 1                     # two-sided level of the Wilson bound
    n.update(conf=pct(conf), cleanruns=f"{runs}", wilsonup=pct(round(br.failure_upper("paginated"), 2)))

    # ---- Worked example: canary sizing (chapter: SizingCanaryTrafficCircuitBreaker, CanaryExcessFailures)
    mon = SC.AcceptanceCanaryMonitor
    sp = mon.result
    p0, p1, a, b = mon.p_base, mon.p_degraded, mon.alpha, mon.beta
    lam = mon.arrival_rate.to("1/hour").magnitude
    rho = mon.fraction
    d = mon.label_delay.to("hour").magnitude
    u = math.log((1 - b) / a)
    lo = math.log(b / (1 - a))
    m1 = p1 * math.log(p1 / p0) + (1 - p1) * math.log((1 - p1) / (1 - p0))
    en = ((1 - b) * u + b * lo) / m1
    near(u, sp.upper, 1e-9)
    near(lo, sp.lower, 1e-9)
    near(m1, sp.drift, 1e-9)
    near(en, sp.expected_samples, 1e-9)
    # as the sheet computes it, from printed values
    u_p, lo_p, m1_p = r(u, 3), r(lo, 3), r(m1, 5)
    en_p = r(((1 - b) * u_p + b * lo_p) / m1_p, 1)
    assert fx(en_p, 1) == fx(en, 1), (en_p, en)
    rate = rho * lam
    near(rate, sp.candidate_rate.to("1/hour").magnitude, 1e-9)
    t_dec = en_p / rate + d
    assert fx(t_dec, 1) == fx(sp.decision_time.to("hour").magnitude, 1)
    pend = rate * d
    near(pend, sp.pending, 1e-9)
    gapacc = p0 - p1
    exc = (en_p + pend) * gapacc
    assert fx(exc, 1) == fx(sp.excess_failures, 1)
    pend_full = lam * d
    exc_full = (en_p + pend_full) * gapacc
    assert fx(exc_full, 1) == fx(sp.full_excess_failures, 1)
    n.update(pzero=fx(p0, 2), pone=fx(p1, 2), alpha=fx(a, 2), beta=fx(b, 2), betac=fx(1 - b, 2),
             alphac=fx(1 - a, 2), ponec=fx(1 - p1, 2), pzeroc=fx(1 - p0, 2), arrivals=fx(lam, 0), rho=pct(rho), rhofrac=fx(rho, 2), delay=fx(d, 1),
             upper=fx(u, 3), lower=fx(lo, 3), lowerabs=fx(-lo, 3), drift=fx(m1, 5), en=fx(en_p, 1),
             rate=fx(rate, 0), tdec=fx(t_dec, 1), pend=fx(pend, 1), gapacc=fx(gapacc, 2),
             excess=fx(exc, 1), pendfull=fx(pend_full, 0), excessfull=fx(exc_full, 1))

    # ---- Your turn: the same monitor at twice the allocation (chapter: AcceptanceCanaryMonitorDoubled)
    dbl = SC.AcceptanceCanaryMonitorDoubled
    rho2 = dbl.fraction
    rate2 = rho2 * lam
    t2 = en_p / rate2 + d
    assert fx(t2, 1) == fx(dbl.result.decision_time.to("hour").magnitude, 1)
    pend2 = rate2 * d
    exc2 = (en_p + pend2) * gapacc
    assert fx(exc2, 1) == fx(dbl.result.excess_failures, 1)
    n.update(rhotwo=pct(rho2), ratetwo=fx(rate2, 0), tdectwo=fx(t2, 1), pendtwo=fx(pend2, 1),
             excesstwo=fx(exc2, 1), admittwo=fx(en_p / rate2, 1), admit=fx(en_p / rate, 1))

    # ---- Q1 ASSUMED: one host clock; retrieve 0.5 s, propose 2.5 s, tests 7.0 s alongside lint 4.0 s,
    #      a 1.0 s unexplained wait, synthesize 3.0 s; a cache cuts the tests to 3.0 s
    qr, qp, qt, ql, qg, qs, qcut = 0.5, 2.5, 7.0, 4.0, 1.0, 3.0, 3.0
    qwall = qr + qp + max(qt, ql) + qg + qs
    qleaf = qr + qp + qt + ql + qs
    qcrit = qr + qp + max(qt, ql) + qs
    qsave = max(qt, ql) - max(qcut, ql)
    assert qsave < qt - qcut
    n.update(qret=fx(qr, 1), qprop=fx(qp, 1), qtests=fx(qt, 1), qlint=fx(ql, 1), qgap=fx(qg, 1),
             qsyn=fx(qs, 1), qwall=fx(qwall, 1), qleaf=fx(qleaf, 1), qcrit=fx(qcrit, 1),
             qcut=fx(qcut, 1), qsave=fx(qsave, 1), qtestcut=fx(qt - qcut, 1))

    # ---- Q2 ASSUMED: 2,000 trials an hour (60 failed, 20 pending, the rest ordinary), 2 MB bodies,
    #      head sampling at 5 percent, tail keeps failed + pending + a 1 percent control sample;
    #      a failure mode seen 12 times
    tot, fail, pend_q, size, head, ctrl, occ = 2000, 60, 20, 2.0, 0.05, 0.01, 12
    ordin = tot - fail - pend_q
    miss = (1 - head) ** occ
    head_n = tot * head
    tail_n = fail + pend_q + ctrl * ordin
    share = fail / tail_n
    n.update(qtrials=f"{tot:,}", qfailed=f"{fail}", qpending=f"{pend_q}", qordinary=f"{ordin:,}",
             qsize=fx(size, 0), qhead=pct(head), qheadc=fx(1 - head, 2), qctrl=pct(ctrl), qocc=f"{occ}",
             qmiss=pct(miss), qheadn=fx(head_n, 0), qheadmb=fx(head_n * size, 0),
             qctrln=fx(ctrl * ordin, 1), qtailn=fx(tail_n, 1), qtailmb=fx(tail_n * size, 1),
             qshare=pct(share), qrate=pct(fail / tot))

    # ---- Q4 ASSUMED: paired lower limit -3 points against a 2-point tolerance; goodput +8 percent,
    #      relative tolerance 5 percent; zero violations; tool drift above its alert threshold
    q_l, q_tol, q_g, q_eps = -3, 2, 0.08, 0.05
    assert not (q_l > -q_tol) and (1 + q_g) >= (1 - q_eps)
    n.update(qlow=f"{-q_l}", qtol=f"{q_tol}", qgood=pct(q_g), qgeps=pct(q_eps))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
