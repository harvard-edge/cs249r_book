"""Numbers for the Agent Evaluation study sheet (Volume III, Chapter 13).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(PairedAcceptance, SampleSizeScale, PassPowK, KappaChance, SelectionMetrics), and written as
\\setnum{key}{value} to evaluation_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
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


def signed(x, nd):
    """A signed value with a typeset minus sign for negatives."""
    s = fx(x, nd)
    return ("$-$" + s[1:]) if s.startswith("-") else ("+" + s)


def numbers() -> dict:
    n = {}

    # ---- Confidence quantile (chapter: FiftyTaskIntervals.z_critical, 95 percent two-sided)
    iv = SC.FiftyTaskIntervals
    z = iv.result.z_critical
    zr = r(z, 2)
    n.update(zcrit=fx(z, 2), zsq=fx(zr ** 2, 2))

    # ---- Worked example: paired gate (chapter: PairedAcceptance, Scenarios.RepairCandidatePairedComparison)
    pc = SC.RepairCandidatePairedComparison
    pres = pc.result
    N, b, c = pres.pairs, pc.baseline_only, pc.candidate_only
    tol = pc.tolerance
    dhat = (c - b) / N
    near(dhat, pres.difference, 1e-12)
    psi = (b + c) / N
    near(psi, pres.discordant_fraction, 1e-12)
    var = (psi - dhat ** 2) / N
    near(var * N, pres.paired_variance, 1e-12)
    se = r(math.sqrt(var), 4)                                         # printed 0.0364
    hw = zr * se                                                      # 1.96 x 0.0364, as the sheet computes it
    near(hw, pres.paired_half_width, 0.0005)
    hw_pt = r(hw * 100, 1)
    lo_pt, hi_pt = r(dhat * 100, 1) - hw_pt, r(dhat * 100, 1) + hw_pt
    assert lo_pt < -tol * 100 and lo_pt < 0 < hi_pt                   # gate fails, not superior
    assert not pres.noninferior and not pres.superior
    n.update(pairs=f"{N}", baselineonly=f"{b}", candidateonly=f"{c}", both=f"{pc.both}", neither=f"{pc.neither}",
             disc=f"{b + c}", tolpt=fx(tol * 100, 0),
             dhat=fx(dhat, 3), dhatpt=fx(dhat * 100, 1), psi=fx(psi, 2),
             dhatsq=fx(dhat ** 2, 4), var=fx(var, 6), se=fx(se, 4), hw=fx(hw, 3), hwpt=fx(hw_pt, 1),
             lopt=signed(lo_pt, 1), hipt=signed(hi_pt, 1),
             lomten=signed(pres.lower * 100, 1), himten=signed(pres.upper * 100, 1),
             pairsgain=f"{pres.pairs_if_gain_holds}")
    # for the picture (points, plain decimals)
    n.update(lonum=fx(lo_pt, 1), hinum=fx(hi_pt, 1), tolnum=fx(-tol * 100, 0))

    # ---- Your turn. ASSUMED: the same discordance pattern on twice as many pairs
    N2, b2, c2 = 2 * N, 2 * b, 2 * c
    var2 = (psi - dhat ** 2) / N2
    se2 = r(math.sqrt(var2), 4)
    hw2_pt = r(zr * se2 * 100, 1)
    lo2, hi2 = r(dhat * 100, 1) - hw2_pt, r(dhat * 100, 1) + hw2_pt
    assert -tol * 100 < lo2 < 0 < hi2                                 # gate passes, still not superior
    n.update(pairstwo=f"{N2}", baselineonlytwo=f"{b2}", candidateonlytwo=f"{c2}", vartwo=fx(var2, 6), setwo=fx(se2, 4),
             hwtwopt=fx(hw2_pt, 1), lotwopt=signed(lo2, 1), hitwopt=signed(hi2, 1))

    # ---- Front: pairing share of variance (chapter: SampleSizeScale.paired_share, independent_multiple)
    n.update(pairedshare=pct(round(pres.paired_share, 2)), indepmult=fx(pres.independent_multiple, 1))

    # ---- Front + mistake: repeated-run consistency (chapter: PassPowK, Scenarios.RecurringTaskConsistency)
    rc = SC.RecurringTaskConsistency
    rres = rc.result
    p, k, tgt = rc.step_success, rc.long_horizon, rc.target_success
    powk = rres.success[k]
    near(powk, p ** k, 1e-12)
    pneed = tgt ** (1 / k)
    near(pneed, 1 - rres.required_step_error, 1e-12)
    n.update(prun=pct(p), prunfrac=fx(p, 1), krec=f"{k}", powk=pct(powk), target=pct(tgt),
             pneed=pct(pneed, 1), failcut=fx(rres.error_ratio, 1),
             atk=pct(1 - (1 - p) ** k, 0),
             lnp=f"{math.log(p):.7f}", lnq=f"{math.log(1 - p):.7f}")   # for the plotted curves only

    # ---- Mistake: agreement is not truth (chapter: KappaChance, Scenarios.IndependentJudgesNinetyPercent)
    kj = SC.IndependentJudgesNinetyPercent
    shares = kj.class_shares
    po = sum(s * s for s in shares)                                   # independent raters: observed = chance
    pe = sum(s * s for s in shares)
    kappa = (po - pe) / (1 - pe)
    near(po, kj.agreement, 1e-12)
    near(kappa, kj.kappa, 1e-12)
    n.update(judgepass=pct(shares[0]), judgeagree=pct(po), kappa=fx(kappa, 0))

    # ---- Front: selection cost (chapter: SelectionMetrics, Scenarios.CappedCandidateSelection)
    cs = SC.CappedCandidateSelection
    sres = cs.result
    n.update(batchpremium=pct(round(sres.batch_premium, 2)))

    # ---- Check yourself
    # Q1 ASSUMED: zero violations in 80 trials; either 80 independent tasks, or 20 tasks run 4 times
    q_n, q_tasks, q_runs = 80, 20, 4
    assert q_tasks * q_runs == q_n
    u80 = zr ** 2 / (q_n + zr ** 2)
    u20 = zr ** 2 / (q_tasks + zr ** 2)
    near(u80, z ** 2 / (q_n + z ** 2), 1e-4)
    n.update(qn=f"{q_n}", qtasks=f"{q_tasks}", qruns=f"{q_runs}", qneighty=fx(q_n + zr ** 2, 2),
             qntwenty=fx(q_tasks + zr ** 2, 2), quhigh=pct(u20, 1), qulow=pct(u80, 1))

    # Q2 ASSUMED: a support answer correct on 96 percent of single runs, asked in 6 sessions; pool of 3
    q2p, q2k, q2pool, q2t = 0.96, 6, 3, 0.90
    q2pow = q2p ** q2k
    q2at = 1 - (1 - q2p) ** q2pool
    q2need = q2t ** (1 / q2k)
    n.update(qtwop=pct(q2p), qtwopfrac=fx(q2p, 2), qtwok=f"{q2k}", qtwopool=f"{q2pool}", qtwotarget=pct(q2t),
             qtwotargetfrac=fx(q2t, 1), qtwopow=pct(q2pow, 1), qtwoat=pct(q2at, 2), qtwomissfrac=fx(1 - q2p, 2),
             qtwoneed=pct(q2need, 1))

    # Q3 ASSUMED: candidates pass with 0.3, cap 4, $0.06 to generate and $0.03 to check each
    q3p, q3k, q3g, q3v = 0.3, 4, 0.06, 0.03
    unit = q3g + q3v
    seq = unit / q3p
    q3q = 1 - (1 - q3p) ** q3k
    q3qr = r(q3q, 3)
    batch = r(q3k * unit, 2) / q3qr                                   # as the sheet computes it, printed q
    near(batch, q3k * unit / q3q, 0.005)
    assert fx(batch, 2) == fx(q3k * unit / q3q, 2)
    n.update(qthreep=fx(q3p, 1), qthreek=f"{q3k}", qthreeg=fx(q3g, 2), qthreev=fx(q3v, 2),
             qthreeunit=fx(unit, 2), qthreeseq=fx(seq, 2), qthreeq=fx(q3qr, 3),
             qthreemiss=fx(1 - q3p, 1), qthreebatchcost=fx(q3k * unit, 2), qthreebatch=fx(batch, 2))

    # Q4 ASSUMED: an audit of 100 judge-accepted reports finds 12 that fail reference assessment
    q4n, q4f = 100, 12
    n.update(qfourn=f"{q4n}", qfourf=f"{q4f}", qfourfrac=pct(q4f / q4n))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
