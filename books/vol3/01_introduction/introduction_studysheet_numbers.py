"""Numbers for the Introduction study sheet (Volume III, Chapter 1).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(HorizonCompounding, CostPerAccepted, IntroFallacies), and written as \\setnum{key}{value} to
introduction_studysheet_numbers.tex next to this file.
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


def numbers() -> dict:
    n = {}

    # ---- Front: horizon compounding (chapter: HorizonCompounding, Scenarios.UncheckedRepairHorizon)
    hz = SC.UncheckedRepairHorizon
    hres = hz.result
    step_ok, h_short, h_long = hz.step_success, hz.short_horizon, hz.long_horizon
    p_short, p_long = hres.success[h_short], hres.success[h_long]
    near(p_short, step_ok ** h_short, 1e-12)
    near(p_long, step_ok ** h_long, 1e-12)
    n.update(stepok=pct(step_ok), stepokfrac=fx(step_ok, 2), hshort=f"{h_short}", hlong=f"{h_long}",
             pshort=pct(p_short, 1), plong=pct(p_long, 1),
             lnstep=f"{math.log(step_ok):.7f}",                       # for the plotted curve only
             target=pct(hz.target_success), epsneeded=pct(hres.required_step_error, 2),
             epsratio=fx(hres.error_ratio, 1))

    # ---- Worked example: cheaper tokens or more acceptances (chapter: CostPerAccepted, nbk-01-cost-per-accepted)
    cs = SC.RepairAttemptCostLevers
    cres = cs.result
    c_model = cs.model_cost.to("USD").magnitude
    c_other = cs.other_cost.to("USD").magnitude
    p_acc, cut, p_better = cs.acceptance, cs.token_price_cut, cs.improved_acceptance
    c_attempt = r(c_model + c_other, 2)
    near(c_attempt, cres.attempt_cost, 1e-9)
    attempts = 1 / p_acc
    c_eff = c_attempt / p_acc
    near(c_eff, cres.cost_per_accepted, 1e-9)
    c_cheap = r(c_model * (1 - cut) + c_other, 2)
    near(c_cheap, cres.cheaper_attempt_cost, 1e-9)
    c_eff_cheap = c_cheap / p_acc
    near(c_eff_cheap, cres.cost_per_accepted_cheaper_tokens, 1e-9)
    c_eff_better = c_attempt / p_better
    near(c_eff_better, cres.cost_per_accepted_improved, 1e-9)
    share = r(c_model / c_attempt, 2)                                # printed tokens' share, 2 places
    p_be = p_acc / (1 - cut * share)
    near(p_be, cres.break_even_acceptance, 0.001)
    p_be = c_attempt / r(c_eff_cheap, 2)                             # as the sheet computes it: printed $2.10 / $3.63
    near(p_be, cres.break_even_acceptance, 0.005)
    assert c_eff_better < c_eff_cheap < c_eff
    n.update(cmodel=fx(c_model, 2), cother=fx(c_other, 2), cattempt=fx(c_attempt, 2), pacc=pct(p_acc),
             paccfrac=fx(p_acc, 2), attempts=fx(attempts, 1), ceff=fx(c_eff, 2), cutpct=pct(cut),
             cheapmodel=fx(c_model * (1 - cut), 2), ccheap=fx(c_cheap, 2), ceffcheap=fx(c_eff_cheap, 2),
             pbetter=pct(p_better), pbetterfrac=fx(p_better, 2), ceffbetter=fx(c_eff_better, 2),
             share=pct(share), sharefrac=fx(share, 2), cutfrac=fx(cut, 1),
             pbreakeven=pct(p_be), pbreakevenfrac=fx(p_be, 2))

    # ---- Your turn. ASSUMED: a sealed-test stage adds $0.60 per attempt and raises acceptance to 50 percent
    add, p_mech = 0.60, 0.50
    c_mech = r(c_attempt + add, 2)
    c_eff_mech = c_mech / p_mech
    p_need = c_mech / r(c_eff, 2)                                    # acceptance at which C'/p' equals C/p
    assert c_eff_mech > c_eff and p_need > p_mech
    n.update(mechadd=fx(add, 2), pmech=pct(p_mech), pmechfrac=fx(p_mech, 2), cmech=fx(c_mech, 2),
             ceffmech=fx(c_eff_mech, 2), pneed=pct(p_need), pneedfrac=fx(p_need, 3))

    # ---- Mistakes: trust-the-report and winning-attempt-only accounting (chapter's fallacies)
    n.update(understate=fx(c_eff / c_attempt, 1),
             wrongshare=pct(1 - r(p_long, 3)))                       # share of trusted acceptances that are wrong

    # ---- Mistake: a larger window (chapter: IntroFallacies, Scenarios.AppendOnlyRepairContext)
    ctx = SC.AppendOnlyRepairContext
    xres = ctx.result
    turns = ctx.turns
    s0 = ctx.initial_tokens.to("token").magnitude
    delta = ctx.tokens_per_turn.to("token").magnitude
    window = ctx.context_window.to("token").magnitude
    last = s0 + delta * (turns - 1)
    cum = sum(s0 + delta * (t - 1) for t in range(1, turns + 1))
    near(last, xres.last_call_tokens.to("token").magnitude, 1e-6)
    near(cum, xres.cumulative_input_tokens.to("token").magnitude, 1e-6)
    n.update(ctxturns=f"{turns}", ctxfirst=f"{s0:,.0f}", ctxper=f"{delta:,.0f}", ctxlast=f"{last:,.0f}",
             ctxwindow=f"{window / 1000:.0f}K", ctxshare=pct(last / window), ctxcum=f"{cum:,.0f}",
             ctxreread=fx(cum / last, 0))

    # ---- Check yourself
    # Q1 ASSUMED: 98 percent per step, horizons of 30 and 60 turns, a target of 80 percent at 30 turns
    q_ok, q_h, q_h2, q_target = 0.98, 30, 60, 0.80
    q_p = q_ok ** q_h
    q_p2 = q_ok ** q_h2
    near(q_p2, q_p ** 2, 1e-12)
    q_eps = 1 - q_target ** (1 / q_h)
    n.update(qok=pct(q_ok), qokfrac=fx(q_ok, 2), qh=f"{q_h}", qhtwo=f"{q_h2}", qtarget=pct(q_target),
             qtargetfrac=fx(q_target, 1), qp=pct(q_p), qpfrac=fx(q_p, 3), qptwo=pct(q_p2),
             qeps=pct(q_eps, 2), qepsh=fx(-math.log(q_target), 3))

    # Q4 ASSUMED: acceptance 30 percent per attempt, a budget of 4 attempts
    q_pa, q_k = 0.30, 4
    q_any = 1 - (1 - q_pa) ** q_k
    n.update(qpa=pct(q_pa), qpafrac=fx(q_pa, 1), qmiss=fx(1 - q_pa, 1), qk=f"{q_k}", qallfail=fx((1 - q_pa) ** q_k, 2), qany=pct(q_any))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
