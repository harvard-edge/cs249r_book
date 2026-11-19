"""Numbers for the Context Engineering study sheet (Volume III, Chapter 4).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(UsableContextBudget, SoftmaxDilution, ContextLayoutPrice, CompactionCadence,
ContextInvalidationTokens, ConstraintRetention), and written as \\setnum{key}{value}
to context_engineering_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value. No accelerator memory appears on this sheet (all sizes are tokens).
"""
import math
from pathlib import Path

from mlsysim import Scenarios as SC


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def c(x):
    """Integer with thousands separators, in LaTeX ({,} keeps math spacing right)."""
    return f"{x:,.0f}".replace(",", "{,}")


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}"


def numbers() -> dict:
    n = {}

    # ---- Input budget of the reference repair request (chapter: UsableContextBudget)
    b = SC.RepairContextBudget
    r = b.result
    window, reserve, turns = b.window, b.output_reserve, b.turns
    system, tools, state = b.system_tokens, b.tool_tokens, b.state_tokens
    perturn, selected = b.observation_per_turn, b.selected_observation
    limit = window - reserve
    fixed = system + tools + state
    budget = limit - fixed
    fit = budget // perturn
    digest = budget // turns
    assert (limit, budget, fit, digest) == (r.input_limit, r.observation_budget, r.turns_fit,
                                            r.max_observation_per_turn)
    assert fit + 1 == r.overflow_turn and fixed == r.fixed_tokens
    first = fixed + perturn                       # call 1 already carries one result
    append_bill = turns * first + perturn * turns * (turns - 1) // 2
    select_bill = turns * (fixed + selected)
    assert append_bill == r.append_bill and select_bill == r.select_bill
    final_need = fixed + turns * perturn + reserve
    assert final_need == r.final_append_with_reserve
    n.update(window=c(window), reserve=c(reserve), turns=f"{turns}", turnsless=f"{turns - 1}",
             system=c(system), tools=c(tools), state=c(state), perturn=c(perturn),
             selected=c(selected), limit=c(limit), fixed=c(fixed), budget=c(budget),
             fit=f"{fit}", overflow=f"{fit + 1}", digest=c(digest), first=c(first),
             appendbill=c(append_bill), selectbill=c(select_bill),
             billratio=f"{append_bill / select_bill:.1f}", finalneed=c(final_need),
             pairs=c(turns * (turns - 1) // 2), turnsfirst=c(turns * first),
             appendpart=c(perturn * turns * (turns - 1) // 2), requestsize=c(fixed + selected),
             # drawing: token counts in thousands
             kwindow=f"{window / 1e3:.0f}", ksystem=f"{system / 1e3:.0f}", ktools=f"{tools / 1e3:.0f}",
             kstate=f"{state / 1e3:.0f}", kbudget=f"{budget / 1e3:.0f}", kreserve=f"{reserve / 1e3:.0f}",
             kperturn=f"{perturn / 1e3:.0f}", kappendbill=f"{append_bill / 1e6:.2f}",
             kselectbill=f"{select_bill / 1e6:.2f}")
    # cumulative bills for the worked-example picture (millions of tokens), turn 1..30
    pts = list(enumerate(r.append_cumulative, 1))
    n["curveappend"] = " ".join(f"({t},{x / 1e6:.4f})" for t, x in pts if t <= fit + 1)
    n["curveappendlate"] = " ".join(f"({t},{x / 1e6:.4f})" for t, x in pts if t >= fit + 1)
    n["koverflowbill"] = f"{r.append_cumulative[fit] / 1e6:.4f}"
    n["curveselect"] = " ".join(f"({t},{x / 1e6:.4f})" for t, x in enumerate(r.select_cumulative, 1))

    # ---- Your turn. ASSUMED: a larger tool catalog, 20,000 tokens of declarations; all else as above.
    ytools = 20_000
    yfixed = system + ytools + state
    ybudget = limit - yfixed
    yfit = ybudget // perturn
    ydigest = ybudget // turns
    assert yfit < turns and ydigest < perturn
    n.update(ytools=c(ytools), yfixed=c(yfixed), ybudget=c(ybudget), yfit=f"{yfit}",
             yoverflow=f"{yfit + 1}", ydigest=c(ydigest), ydigestfit=c(ydigest * turns))

    # ---- Attention dilution, toy head (chapter: SoftmaxDilution)
    d = SC.ToyHeadAttentionDilution
    zt, zb = d.target_logit, d.background_logit
    short_w = math.exp(zt) / (math.exp(zt) + (d.short_keys - 1) * math.exp(zb))
    long_w = math.exp(zt) / (math.exp(zt) + (d.long_keys - 1) * math.exp(zb))
    near(short_w, d.result.short_weight, 1e-12)
    near(long_w, d.result.long_weight, 1e-12)
    n.update(ztarget=f"{zt:.0f}", zback=f"{zb:.0f}", shortkeys=c(d.short_keys), longkeys=c(d.long_keys),
             shortweight=pct(short_w), longweight=pct(long_w))

    # ---- Prefix layout (chapter: ContextLayoutPrice); Q1 asks where reuse ends.
    lp = SC.RepairPrefixLayoutBill
    lr = lp.result
    root, trunk, leaf, rewrites = lp.root_tokens, lp.trunk_tokens, lp.leaf_tokens, lp.trunk_rewrites
    req = root + trunk + leaf
    assert req == fixed + selected == lr.request_tokens
    warm = lp.turns - 1 - rewrites
    total = lp.turns * req
    unstable = total                                               # ASSUMED in Q1: request ID first
    after_root = req + (lp.turns - 1) * (trunk + leaf)             # ASSUMED in Q1: request ID after the root
    stable = req + rewrites * (trunk + leaf) + warm * leaf         # ID in the leaf (chapter layout)
    assert stable == lr.uncached_tokens and total == lr.total_input and warm == lr.warm_calls
    near(total / stable, lr.prefill_ratio, 1e-9)
    n.update(root=c(root), trunk=c(trunk), leaf=c(leaf), rewrites=f"{rewrites}", warm=f"{warm}",
             request=c(req), suffix=c(trunk + leaf), total=c(total), afterroot=c(after_root),
             stable=c(stable), callsless=f"{lp.turns - 1}",
             unstableratio=f"{unstable / stable:.1f}", afterratio=f"{after_root / stable:.1f}",
             hitrate=pct(1 - stable / total))

    # ---- Compaction spacing (chapter: CompactionCadence); Q2 changes the loss rate.
    cc = SC.RepairCompactionWatermarks
    cr = cc.result
    high, low, alt, arrive = cc.trigger, cc.target, cc.alternate_target, cc.tokens_per_turn
    cyc = math.ceil((high - low) / arrive)
    alt_cyc = math.ceil((high - alt) / arrive)
    trig = low + cyc * arrive
    assert cyc == cr.cycle_turns and alt_cyc == cr.alternate_cycle_turns and trig == cr.trigger_length
    passes, alt_passes = cc.turns // cyc, cc.turns // alt_cyc
    assert passes == cr.passes and alt_passes == cr.alternate_passes
    assert trig + cc.output_reserve <= window
    n.update(high=c(high), low=c(low), alt=c(alt), gap=c(high - low), altgap=c(high - alt),
             cycle=f"{cyc}", altcycle=f"{alt_cyc}", triglen=c(trig), overshoot=c(trig - high),
             trigreserve=c(trig + cc.output_reserve), passes=f"{passes}", altpasses=f"{alt_passes}",
             passratio=f"{alt_passes / passes:.0f}")

    # ---- Survival across compactions (chapter: ConstraintRetention)
    cs = SC.CompactionConstraintRetention
    rho = 1 - cs.step_success
    s_short, s_long = cs.step_success ** cs.short_horizon, cs.step_success ** cs.long_horizon
    near(s_short, cs.result.success[cs.short_horizon], 1e-12)
    near(s_long, cs.result.success[cs.long_horizon], 1e-12)
    below = math.ceil(math.log(0.5) / math.log(cs.step_success))
    assert below == cs.result.first_horizon_below_target
    n.update(rho=f"{rho * 100:.0f}", keep=f"{cs.step_success:.2f}", roundsshort=f"{cs.short_horizon}",
             roundslong=f"{cs.long_horizon}", surviveshort=pct(s_short), survivelong=pct(s_long),
             evenodds=f"{below}")
    # Q2 ASSUMED: a better summarizer loses each constraint with probability 0.03 per pass
    qrho = 0.03
    qkeep = 1 - qrho
    n.update(qrho=f"{qrho * 100:.0f}", qkeep=f"{qkeep:.2f}", qwide=pct(qkeep ** passes),
             qnarrow=pct(qkeep ** alt_passes))

    # ---- Invalidation (chapter: ContextInvalidationTokens)
    iv = SC.SupersededExcerptInvalidation
    ir = iv.result
    replaced_suffix = iv.request_tokens - iv.superseded_tokens + iv.marker_tokens - iv.prefix_tokens
    assert replaced_suffix == ir.replaced_suffix and iv.notice_tokens == ir.appended_suffix
    assert iv.prefix_tokens == root and iv.request_tokens == req
    n.update(excerpt=c(iv.superseded_tokens), marker=c(iv.marker_tokens), notice=c(iv.notice_tokens),
             replsuffix=c(replaced_suffix), suffixratio=f"{round(replaced_suffix / iv.notice_tokens)}")

    # ---- Q5 ASSUMED: two complete policies on the same 100 repair trials, every read, compaction and
    # correction call charged to its policy. A prunes hard; B selects, audits, keeps handles.
    trials = 100
    cost_a, acc_a = 1.20, 50          # USD per attempt, accepted trials
    cost_b, acc_b = 1.60, 80
    eff_a, eff_b = cost_a / (acc_a / trials), cost_b / (acc_b / trials)
    assert cost_a < cost_b and eff_b < eff_a
    n.update(qtrials=f"{trials}", qcosta=f"{cost_a:.2f}", qcostb=f"{cost_b:.2f}", qacca=f"{acc_a}",
             qaccb=f"{acc_b}", qpa=f"{acc_a / trials:.2f}", qpb=f"{acc_b / trials:.2f}",
             qeffa=f"{eff_a:.2f}", qeffb=f"{eff_b:.2f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
