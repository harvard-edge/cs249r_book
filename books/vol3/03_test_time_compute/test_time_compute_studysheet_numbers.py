"""Numbers for the Test-Time Compute study sheet (Volume III, Chapter 3).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(ProbeAllocation, AllocationAxes, CoverageVsSelection, CheckYield, ArgmaxSelection,
StrategyComparison), and written as \\setnum{key}{value} to test_time_compute_studysheet_numbers.tex
next to this file. Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC
from mlsysim import load_series


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

    # ---- Front: the three axes under one output ceiling (chapter: AllocationAxes)
    ax = SC.OutputCeilingThreeAllocations
    axr = ax.result
    n.update(axprompt=f"{ax.prompt:,}", axceiling=f"{ax.ceiling:,}", axcandidates=f"{ax.candidates}",
             axrounds=f"{ax.rounds}",
             axdepthtime=fx(axr.depth_time.to("s").magnitude, 1),
             axbreadthtime=fx(axr.breadth_time.to("s").magnitude, 1),
             axfeedtime=fx(axr.feedback_time.to("s").magnitude, 1),
             axdepthbill=fx(axr.depth_cost.to("USD").magnitude, 3),
             axbreadthbill=fx(axr.breadth_cost.to("USD").magnitude, 3),
             axfeedbill=fx(axr.feedback_cost.to("USD").magnitude, 3))

    # ---- Front picture: coverage vs majority vote, 8B model on MATH (chapter: CoverageVsSelection,
    # measured, recomputed from released per-sample labels; data/repeated_sampling_majority.csv)
    rows = load_series("vol3/03_test_time_compute/data/repeated_sampling_majority.csv").entries
    pts = sorted((e for e in rows if e["model"] == "Llama-3-8B-Instruct" and e["dataset"] == "MATH"),
                 key=lambda e: e["k"])
    width, height, kmax = 196.0, 60.0, 10_000
    def xy(k, v):
        return f"({math.log10(k) / math.log10(kmax) * width:.1f},{v * height:.1f})"
    n["covcurve"] = " ".join(xy(e["k"], e["coverage"]) for e in pts)          # positions for the plot only
    n["votecurve"] = " ".join(xy(e["k"], e["majority_vote_accuracy"]) for e in pts)
    last = pts[-1]
    assert last["k"] == kmax
    n.update(kmax=f"{kmax:,}", covmax=pct(last["coverage"]), votemax=pct(last["majority_vote_accuracy"]),
             covone=pct(pts[0]["coverage"]),
             covmaxy=f"{last['coverage'] * height:.1f}", votemaxy=f"{last['majority_vote_accuracy'] * height:.1f}")

    # ---- Worked example: sparse valid pool, first passing candidate returned (chapter: CheckYield)
    cy = SC.SparseValidPoolFirstPass
    cr = cy.result
    pv, fr, fa, pool = cy.valid_rate, cy.false_reject_rate, cy.false_accept_rate, cy.pool_size
    vpass = r(pv * (1 - fr), 4)
    ipass = r((1 - pv) * fa, 4)
    q = vpass + ipass
    near(q, cr.pass_probability, 1e-9)
    prec = vpass / q
    near(prec, cr.precision, 1e-9)
    anysel = 1 - (1 - q) ** pool
    near(anysel, cr.any_selection, 1e-9)
    wrong = r(anysel, 3) * r(1 - prec, 3)                           # from the printed rounded values
    near(wrong, cr.wrong_selection, 0.001)
    n.update(validrate=pct(pv), falsereject=pct(fr), falseaccept=pct(fa), pool=f"{pool}",
             validfrac=fx(pv, 2), validmiss=fx(1 - pv, 2), frkeep=fx(1 - fr, 2), fafrac=fx(fa, 2),
             validpass=fx(vpass, 3), invalidpass=fx(ipass, 3), passprob=fx(q, 3),
             precision=pct(prec, 1), defect=pct(1 - prec, 1), anyselect=pct(anysel, 1),
             anyselectfrac=fx(anysel, 3), defectfrac=fx(1 - prec, 3), wrongreturn=pct(wrong, 1),
             validcount=fx(pool * vpass, 2), invalidcount=fx(pool * ipass, 2),
             passcount=fx(pool * q, 2), anyfalse=pct(cr.any_false_pass, 0))

    # ---- Your turn: halve the false-acceptance rate, or double the valid rate (chapter's sensitivity scenarios)
    hf = SC.SparseValidPoolHalvedFalseAccept
    dv = SC.SparseValidPoolDoubledValidRate
    h_ipass = r((1 - hf.valid_rate) * hf.false_accept_rate, 4)
    h_prec = vpass / (vpass + h_ipass)
    near(h_prec, hf.result.precision, 1e-9)
    d_vpass = r(dv.valid_rate * (1 - dv.false_reject_rate), 4)
    d_ipass = r((1 - dv.valid_rate) * dv.false_accept_rate, 4)
    d_prec = d_vpass / (d_vpass + d_ipass)
    near(d_prec, dv.result.precision, 1e-9)
    n.update(halfaccept=pct(hf.false_accept_rate), halffrac=fx(hf.false_accept_rate, 2),
             halfinvalid=fx(h_ipass, 4), halfprecision=pct(h_prec, 1), halfpassprob=fx(vpass + h_ipass, 4),
             doublevalid=pct(dv.valid_rate), doublefrac=fx(dv.valid_rate, 2), doublemiss=fx(1 - dv.valid_rate, 2),
             doublevalidpass=fx(d_vpass, 3), doubleinvalid=fx(d_ipass, 3),
             doublepassprob=fx(d_vpass + d_ipass, 3), doubleprecision=pct(d_prec, 1))

    # ---- Mistake: more tokens vs a probe (chapter: ProbeAllocation)
    pa = SC.ProbeVsGeneratedExplanation
    par = pa.result
    tok_ms = r(par.token_time.to("ms").magnitude, 2)
    ans_s = r(pa.answer_tokens * tok_ms / 1000, 1)
    near(ans_s, par.answer_time.to("s").magnitude, 0.05)
    probe_ms = pa.probe_time.to("ms").magnitude
    n.update(answertokens=f"{pa.answer_tokens:,}", tokenms=fx(tok_ms, 2), answersec=fx(ans_s, 1),
             probemillis=fx(probe_ms, 0), proberatio=fx(ans_s * 1000 / probe_ms, 0))

    # ---- Mistake: breadth reached the highest pass rate (chapter: StrategyComparison)
    sc = SC.DeadlineBoundRepairPolicies
    s = sc.result
    a_bill = r(s.a_cost_per_pass.to("USD").magnitude, 4)
    b_bill = r(s.b_cost_per_pass.to("USD").magnitude, 4)
    c_bill = r(s.c_cost_per_pass.to("USD").magnitude, 4)
    n.update(apass=pct(s.a_pass), bpass=pct(s.b_pass, 0), cpass=pct(s.c_pass), breadthcount=f"{sc.breadth}",
             roundsec=fx(s.round_time.to("s").magnitude, 0), deadlinesec=fx(sc.deadline.to("s").magnitude, 0),
             feedrounds=f"{s.rounds}",
             abill=fx(a_bill, 4), bbill=fx(b_bill, 4), cbill=fx(c_bill, 4), bratio=fx(b_bill / a_bill, 1))
    near(b_bill / a_bill, s.b_over_a, 0.01)

    # ---- Mistake: the top score (chapter: ArgmaxSelection)
    am = SC.ArgmaxScoreNoise
    amr = am.result
    big = am.large_pool
    n.update(argpool=f"{big}", argshift=fx(amr.expected_max[big], 1), exploitrate=pct(am.exploit_rate, 1),
             exploitpresent=pct(amr.exploit_presence, 0))
    near(1 - (1 - am.exploit_rate) ** big, amr.exploit_presence, 1e-9)

    # ---- Check yourself
    # Q2 ASSUMED: one task at p = 0.25, pools of 8; mixed workload half at 0.5, half never passing
    q2_p, q2_n, q2_easy = 0.25, 8, 0.50
    q2_pass = 1 - (1 - q2_p) ** q2_n
    q2_true = 0.5 * (1 - (1 - q2_easy) ** q2_n)
    n.update(qpass=pct(q2_p), qpassfrac=fx(q2_p, 2), qmissfrac=fx(1 - q2_p, 2), qpool=f"{q2_n}",
             qpoolpass=pct(q2_pass, 1), qeasy=pct(q2_easy), qeasyfrac=fx(q2_easy, 1), qtrue=pct(q2_true, 1))
    # Q3 ASSUMED: 6 children per node, a step check rejecting 75 percent, depth 3
    q3_b, q3_rej, q3_d = 6, 0.75, 3
    q3_eff = q3_b * (1 - q3_rej)
    q3_nodes = sum(q3_b * q3_eff ** (j - 1) for j in range(1, q3_d + 1))
    q3_full = sum(q3_b ** j for j in range(1, q3_d + 1))
    n.update(qkids=f"{q3_b}", qreject=pct(q3_rej), qrejectfrac=fx(q3_rej, 2), qdepth=f"{q3_d}",
             qeff=fx(q3_eff, 1), qeffsq=fx(q3_eff ** 2, 2), qcritical=pct(1 - 1 / q3_b),
             qnodes=fx(q3_nodes, 1), qfull=f"{q3_full}")
    n.update(qkeepfrac=fx(1 - q3_rej, 2), qlevelone=f"{q3_b}", qleveltwo=fx(q3_b * q3_eff, 1),
             qlevelthree=fx(q3_b * q3_eff ** 2, 1))
    near(r(q3_b * q3_eff, 1) + r(q3_b * q3_eff ** 2, 1) + q3_b, q3_nodes, 1e-9)
    # Q1: choose a policy (chapter's DeadlineBoundRepairPolicies spends and pass rates; ASSUMED 60 percent target)
    target = 0.60
    spend = {k: r(getattr(s, f"{k}_cost").to("USD").magnitude, 5) for k in ("a", "b", "c")}
    pacc = {"a": r(s.a_pass, 3), "b": r(s.b_pass, 3), "c": r(s.c_pass, 3)}
    ceff = {k: spend[k] / pacc[k] for k in spend}
    for k in ceff:
        near(ceff[k], getattr(s, f"{k}_cost_per_pass").to("USD").magnitude, 0.0002)
    assert pacc["a"] < target <= pacc["c"] < pacc["b"] and ceff["c"] < ceff["b"]
    n.update(qtarget=pct(target), qaspend=fx(spend["a"], 5), qbspend=fx(spend["b"], 5), qcspend=fx(spend["c"], 5),
             qapass=pct(pacc["a"], 0), qbpass=pct(pacc["b"], 1), qcpass=pct(pacc["c"], 0),
             qbpassfrac=fx(pacc["b"], 3), qapassfrac=fx(pacc["a"], 2), qcpassfrac=fx(pacc["c"], 2),
             qaeff=fx(ceff["a"], 4), qbeff=fx(ceff["b"], 4), qceff=fx(ceff["c"], 4),
             qcabstain=pct(1 - pacc["c"]), qbovercs=fx(r(ceff["b"], 4) / r(ceff["c"], 4), 1),
             qmaxsec=fx(s.max_c_time.to("s").magnitude, 0))
    # Q4 ASSUMED: the top candidate's learned score
    n.update(qscore=fx(0.97, 2))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
