"""Numbers for the Conclusion study sheet (Volume III, Chapter 20).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(IncidentPrice, FallacyPrices), and written as \\setnum{key}{value} to
conclusion_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Agents, USD
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

    # ---- Worked example: price one incident attempt (chapter: IncidentPrice, Agents.Remediation.OnCall)
    remed = Agents.Remediation.OnCall
    price = SC.RemediationIncidentAttempt.result
    pivot = SC.RemediationCanaryPivotLoss.result
    c_model = remed.model_cost_per_attempt.to(USD).magnitude
    c_env = remed.environment_cost_per_attempt.to(USD).magnitude
    c_assess = remed.assessment_cost_per_attempt.to(USD).magnitude
    c_approval = remed.approval_cost_per_attempt.to(USD).magnitude
    review_min = remed.approver_review_time.to("minute").magnitude
    rate = remed.approver_hourly_rate.to("USD/hour").magnitude
    near(c_approval, review_min / 60 * rate, 1e-9)
    p_acc = remed.acceptance_probability
    c_attempt = r(c_model + c_env + c_assess + c_approval, 2)
    near(c_attempt, price.attempt_cost, 1e-9)
    c_eff = c_attempt / p_acc
    near(c_eff, price.cost_per_accepted, 1e-9)
    model_share = c_model / c_attempt
    near(model_share, price.token_share, 1e-12)
    approval_share = c_approval / c_attempt
    near(approval_share, remed.approval_share, 1e-12)
    # doubling the model calls (chapter: "raises attempt cost by its existing model share")
    c_grown = r(c_attempt + c_model, 2)
    near(c_grown, price.grown_attempt_cost, 1e-9)
    p_need = c_grown / r(c_eff, 2)                                   # as the sheet computes it: $20.60 / $29.21
    near(p_need, price.grown_break_even_acceptance, 0.001)          # 70.52% from printed values, 70.51% unrounded
    # expected irreversible loss of a wrong change with no check ahead of the write (rho = 0)
    p_bad = remed.wrong_change_probability
    impact = remed.wrong_change_impact.to(USD).magnitude
    loss = p_bad * impact * (1 - 0)
    near(loss, pivot.loss_without_check.to(USD).magnitude, 1e-9)
    loss_mult = loss / c_attempt
    assert loss_mult > 20                                             # chapter guard: over twenty attempts' worth
    n.update(cmodel=fx(c_model, 2), cenv=fx(c_env, 2), cassess=fx(c_assess, 2), capproval=fx(c_approval, 0),
             reviewmin=fx(review_min, 0), reviewrate=fx(rate, 0),
             cattempt=fx(c_attempt, 2), pacc=pct(p_acc), paccfrac=fx(p_acc, 2), ceff=fx(c_eff, 2),
             modelshare=pct(model_share, 1), approvalshare=pct(approval_share, 1),
             othershare=pct((c_env + c_assess) / c_attempt, 1),
             cgrown=fx(c_grown, 2), pneed=pct(p_need, 1),
             pbad=pct(p_bad), impact=f"{impact:,.0f}", loss=fx(loss, 0), lossmult=fx(loss_mult, 0))

    # ---- Your turn. ASSUMED: a second reviewer signs each approval, adding one more approver charge
    c_two = r(c_attempt + c_approval, 2)
    c_eff_two = c_two / p_acc
    p_need_two = c_two / r(c_eff, 2)
    assert p_need_two > 1                                             # no acceptance can repay it on cost alone
    n.update(ctwo=fx(c_two, 2), cefftwo=fx(c_eff_two, 2), pneedtwo=pct(p_need_two))

    # ---- Mistakes (chapter: FallacyPrices)
    res = SC.ResidualViolationsFiftyTurns
    eps, horizon, per_day = res.step_violation, res.horizon, res.trajectories_per_day
    p_any = 1 - (1 - eps) ** horizon
    p_any_r = r(p_any, 4)                                             # printed to two places of percent
    near(p_any_r * per_day, p_any * per_day, 1.0)
    n.update(epsviolate=pct(eps, 1), horizonturns=f"{horizon}", trajday=f"{per_day:,}",
             pany=pct(p_any, 2), violday=fx(p_any_r * per_day, 0))

    ps = SC.ThirtyPercentValidPassingSet
    p_valid, fa, va = 1 - ps.defect_rate, ps.false_accept_rate, ps.valid_accept_rate
    invalid = (1 - p_valid) * fa / (p_valid * va + (1 - p_valid) * fa)
    near(invalid, ps.result.posterior, 1e-12)
    n.update(pvalid=pct(p_valid), pmiss=pct(1 - va), pfa=pct(fa), invalid=pct(invalid, 1))

    win = SC.HalvedHealthWindow
    widen = 1 / math.sqrt(win.window_share)
    n.update(windowshare=fx(win.window_share, 1), widen=fx(widen, 2))

    # ---- Check yourself
    # Q1 ASSUMED: the same check (misses 10 percent of valid, passes 5 percent of invalid), 10 percent valid;
    # target: at most 5 percent of the passing set invalid
    q_valid, q_target = 0.10, 0.05
    q_inv = (1 - q_valid) * fa / (q_valid * va + (1 - q_valid) * fa)
    # (1-p) a / (p v + (1-p) a) <= t  ->  a <= t p v / ((1-p)(1-t))
    q_fa_max = q_target * q_valid * va / ((1 - q_valid) * (1 - q_target))
    near((1 - q_valid) * q_fa_max / (q_valid * va + (1 - q_valid) * q_fa_max), q_target, 1e-12)
    n.update(qvalid=pct(q_valid), qinvalid=pct(q_inv, 1), qtarget=pct(q_target),
             qfamax=pct(q_fa_max, 3), qfafold=fx(fa / q_fa_max, 1),
             qpassvalid=fx(q_valid * va, 3), qpassinvalid=fx((1 - q_valid) * fa, 3))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
