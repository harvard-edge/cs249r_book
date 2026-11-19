"""Numbers for the RLVR study sheet (Volume III, Chapter 17).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(OracleGaming, GRPOGroup, DifficultyAbsorption, LengthCoefficient, ZeroContrast, RolloutBudget,
RolloutTime), and written as \\setnum{key}{value} to rlvr_studysheet_numbers.tex next to this file.
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


def contrast(p, g):
    return 1 - p ** g - (1 - p) ** g


def numbers() -> dict:
    n = {}

    # ---- Front: GRPO card, the chapter's [0, 1, 0] group (GRPOGroup) and the lone pass of 16 (DifficultyAbsorption)
    grp = SC.RepairGroupOnePassOfThree.result
    near(grp.coefficients[1], math.sqrt(2), 1e-6)
    lone = SC.LonePassInGroupOfSixteen.result
    near(lone.coefficients[0], math.sqrt(lone.group - 1), 1e-9)
    n.update(grpopos=fx(grp.coefficients[1], 2), grponeg=fx(grp.coefficients[0], 2),
             lonegroup=f"{lone.group}", lonecoef=fx(lone.coefficients[0], 2), lonefail=fx(-lone.coefficients[1], 2))

    # ---- Front curve and worked example (ZeroContrast, RolloutBudget)
    hard, big, tgt = SC.HardTaskRolloutStep, SC.HardTaskRolloutStepDoubledGroup, SC.TargetedTaskRolloutStep
    g, gbig, p_hard, p_tgt = hard.group, big.group, hard.pass_rate, tgt.pass_rate
    near(contrast(p_hard, g), hard.result.contrast, 1e-12)
    near(contrast(p_hard, gbig), big.result.contrast, 1e-12)
    near(contrast(p_tgt, g), tgt.result.contrast, 1e-12)
    allfail = r((1 - p_hard) ** g, 4)                     # all-pass term p^G is negligible here
    c_hard = r(1 - allfail, 4)
    near(c_hard, hard.result.contrast, 1e-4)
    per_hard = g / c_hard
    near(per_hard, hard.result.rollouts_per_informative_group, 0.05)
    allfail_big = r((1 - p_hard) ** gbig, 4)
    c_big = r(1 - allfail_big, 4)
    per_big = gbig / c_big
    near(per_big, big.result.rollouts_per_informative_group, 0.05)
    tie_tgt = r(2 * p_tgt ** g, 4)                        # all-pass + all-fail at p = 0.5
    c_tgt = r(1 - tie_tgt, 4)
    per_tgt = g / c_tgt
    near(per_tgt, tgt.result.rollouts_per_informative_group, 0.05)
    saving = r(per_hard, 1) / r(per_tgt, 1)
    near(saving, hard.result.rollouts_per_informative_group / tgt.result.rollouts_per_informative_group, 0.05)
    assert per_tgt < per_hard < per_big
    n.update(group=f"{g}", groupbig=f"{gbig}", phard=pct(p_hard), phardfrac=fx(p_hard, 2),
             ptarget=pct(p_tgt), ptargetfrac=fx(p_tgt, 1),
             missfrac=fx(1 - p_hard, 2), allfail=fx(allfail, 4), contrasthard=fx(c_hard, 4),
             perhard=fx(per_hard, 1), allfailbig=fx(allfail_big, 4), contrastbig=fx(c_big, 4), perbig=fx(per_big, 1),
             tietarget=fx(tie_tgt, 4), contrasttarget=fx(c_tgt, 4), pertarget=fx(per_tgt, 1),
             saving=fx(saving, 1),
             contrasthardshort=fx(hard.result.contrast, 2), contrasttargetshort=fx(tgt.result.contrast, 2),
             silenthard=pct(hard.result.silent_share))

    # ---- Your turn: the chapter's 20 percent variant (ModerateTaskRolloutStep); ASSUMED: groups of 16 at that rate
    mod = SC.ModerateTaskRolloutStep
    p_mod = mod.pass_rate
    allfail_mod = r((1 - p_mod) ** g, 4)
    c_mod = r(1 - allfail_mod - p_mod ** g, 4)
    near(c_mod, mod.result.contrast, 1e-4)
    per_mod = g / c_mod
    near(per_mod, mod.result.rollouts_per_informative_group, 0.05)
    allfail_mod_big = r((1 - p_mod) ** gbig, 4)
    c_mod_big = r(1 - allfail_mod_big, 4)
    per_mod_big = gbig / c_mod_big
    near(c_mod_big, contrast(p_mod, gbig), 1e-4)
    assert per_mod < per_mod_big
    n.update(pmod=pct(p_mod), pmodfrac=fx(p_mod, 1), missmodfrac=fx(1 - p_mod, 1),
             allfailmod=fx(allfail_mod, 4), contrastmod=fx(c_mod, 4), permod=fx(per_mod, 1),
             allfailmodbig=fx(allfail_mod_big, 4), contrastmodbig=fx(c_mod_big, 4), permodbig=fx(per_mod_big, 1),
             modvstarget=fx(r(per_mod, 1) / r(per_tgt, 1), 1))

    # ---- Mistake: false accepts (OracleGaming, and the chapter's 5 percent checkpoint variant)
    fa10, fa5 = SC.VerifierFalseAcceptsAtTenPercentPass, SC.VerifierFalseAcceptsAtFivePercentPass
    eps, p10, p5 = fa10.false_accept_rate, fa10.pass_rate, fa5.pass_rate
    assert fa10.valid_accept_rate == 1.0
    num10 = r((1 - p10) * eps, 4)
    share10 = num10 / (p10 + num10)
    near(share10, fa10.result.false_share, 1e-9)
    num5 = r((1 - p5) * eps, 5)
    share5 = num5 / (p5 + num5)
    near(share5, fa5.result.false_share, 1e-6)
    n.update(faeps=pct(eps, 1), faepsfrac=fx(eps, 3), fapass=pct(p10), fapassfrac=fx(p10, 1), fanum=fx(num10, 4),
             fashare=pct(share10, 1), falowpass=pct(p5), falowshare=pct(share5, 1))

    # ---- Mistake: length bias (LengthCoefficient)
    lc = SC.ShortAndLongFailureLengths
    ratio = lc.long_tokens / lc.short_tokens
    near(ratio, lc.result.per_token_ratio, 1e-12)
    n.update(shorttokens=f"{lc.short_tokens:,}", longtokens=f"{lc.long_tokens:,}", lengthratio=fx(ratio, 0))

    # ---- Mistake: reset vs capacity (RolloutTime, Scenarios.RepairRolloutSlotResets)
    rs = SC.RepairRolloutSlotResets
    rres = rs.result
    h = rs.decisions
    t_model = rs.model_time.to("ms").magnitude
    t_tool = rs.tool_time.to("ms").magnitude
    t_assess = rs.assessment_time.to("s").magnitude
    cold = rs.cold_reset.to("s").magnitude
    snap = rs.snapshot_reset.to("ms").magnitude
    work = h * (t_model + t_tool) / 1000 + t_assess
    near(work, rres.work.to("s").magnitude, 1e-9)
    cold_cycle = work + cold
    snap_cycle = work + snap / 1000
    near(cold_cycle, rres.cold_cycle.to("s").magnitude, 1e-9)
    near(snap_cycle, rres.snapshot_cycle.to("s").magnitude, 1e-9)
    reset_ratio = cold / (snap / 1000)
    near(reset_ratio, rres.reset_ratio, 1e-9)
    cap_ratio = cold_cycle / r(snap_cycle, 3)
    near(cap_ratio, rres.capacity_ratio, 1e-9)
    n.update(slotturns=f"{h}", slotmodel=fx(t_model, 0), slottool=fx(t_tool, 0), slotassess=fx(t_assess, 1),
             slotcold=fx(cold, 1), slotsnap=fx(snap, 0), slotwork=fx(work, 1), slotcoldcycle=fx(cold_cycle, 1),
             slotsnapcycle=fx(snap_cycle, 3), resetratio=fx(reset_ratio, 0), capacityratio=fx(cap_ratio, 2))

    # ---- Check yourself
    # Q1 ASSUMED: a group of five rollouts with one pass; population dispersion, stabilizer ignored
    rewards = [0, 0, 1, 0, 0]
    q_g = len(rewards)
    q_mean = sum(rewards) / q_g
    q_std = math.sqrt(sum((x - q_mean) ** 2 for x in rewards) / q_g)
    q_pass = (1 - q_mean) / q_std
    q_fail = (0 - q_mean) / q_std
    near(q_pass, math.sqrt(q_g - 1), 1e-12)
    near(q_pass + (q_g - 1) * q_fail, 0, 1e-12)
    n.update(qgroup=f"{q_g}", qmean=fx(q_mean, 1), qvar=fx(q_mean * (1 - q_mean), 2), qstd=fx(q_std, 1),
             qcenteredpass=fx(1 - q_mean, 1), qcenteredfail=fx(-q_mean, 1),
             qpass=fx(q_pass, 1), qfail=fx(q_fail, 1), qroot=f"{q_g - 1}")

    # Q2 ASSUMED: format weight 0.3, length weight 0.8
    w_f, w_len = 0.3, 0.8
    low_pass = 1 - w_len
    assert low_pass < w_f
    n.update(wformat=fx(w_f, 1), wlength=fx(w_len, 1), lowpass=fx(low_pass, 1), wlengthmax=fx(1 - w_f, 1))

    # Q3 ASSUMED: training reward climbs from 30 to 55 percent while held-out acceptance stays flat
    n.update(rewardstart="30", rewardend="55")

    # Q4 ASSUMED: a short-horizon support-agent check: 5 turns of 0.4 s model + 0.2 s tool, 0.5 s assessment,
    # 6.5 s cold reset against a 0.5 s snapshot restore
    s_h, s_model, s_tool, s_assess, s_cold, s_snap = 5, 0.4, 0.2, 0.5, 6.5, 0.5
    s_work = s_h * (s_model + s_tool) + s_assess
    s_cold_cycle = s_work + s_cold
    s_snap_cycle = s_work + s_snap
    s_gain = s_cold_cycle / s_snap_cycle
    s_share = s_cold / s_cold_cycle
    repair_share = cold / cold_cycle
    assert s_share > 0.5 > repair_share and s_gain > cap_ratio
    n.update(supturns=f"{s_h}", supmodel=fx(s_model, 1), suptool=fx(s_tool, 1), supassess=fx(s_assess, 1),
             supcold=fx(s_cold, 1), supsnap=fx(s_snap, 1), supwork=fx(s_work, 1), supcoldcycle=fx(s_cold_cycle, 1),
             supsnapcycle=fx(s_snap_cycle, 1), supgain=fx(s_gain, 1), supshare=pct(s_share),
             repairshare=pct(repair_share))

    # Q4(b) ASSUMED: the trainer steps every 30 s; rollouts admitted 45 s and 130 s after their first
    # token; lag bound 2 (chapter: eq-vol3-rlvr-staleness, lag = floor(admission age / step time))
    l_step, l_short, l_long, l_bound = 30, 45, 130, 2
    lag_short, lag_long = math.floor(l_short / l_step), math.floor(l_long / l_step)
    assert lag_short <= l_bound < lag_long
    n.update(lagstep=f"{l_step}", lagshortage=f"{l_short}", laglongage=f"{l_long}", lagbound=f"{l_bound}",
             lagshort=f"{lag_short}", laglong=f"{lag_long}")

    # Front, key idea 4: the continuation standard error bound 1/(2 sqrt K) halves at 4x the continuations
    n.update(mchalve=fx((1 / 0.5) ** 2, 0))
    return n



if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
