"""Numbers for the Failure Recovery study sheet (Volume III, Chapter 12).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(RepairVsUnwind, PivotLoss, LoopCost, RetryLayers, ContainmentCapacity), and written as
\\setnum{key}{value} to failure_recovery_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
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


def comma(x):
    return f"{x:,.0f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def resend(first, grow, turns):
    """Append-only resend: S_sent = H * S_base + dS * H (H - 1) / 2 (chapter: eq-vol3-trajectory-resend)."""
    return turns * first + grow * turns * (turns - 1) // 2


def numbers() -> dict:
    n = {"one": "1"}

    # ---- Worked example: repair vs unwind at matched readiness (chapter: RepairVsUnwind)
    sc = SC.TestConfigRepairVsRebuild
    res = sc.result
    unwind = sc.unwind_time.to("second").magnitude
    rebuild = sc.rebuild_time.to("second").magnitude
    pin = sc.input_price.to("USD / Mtoken").magnitude
    pout = sc.output_price.to("USD / Mtoken").magnitude
    reb_in, reb_out = sc.rebuild_input, sc.rebuild_output
    rep_in, rep_out = sc.repair_input, sc.repair_output
    startup = sc.startup_time.to("second").magnitude
    tps = sc.output_rate.to("token / second").magnitude
    valid = sc.validation_time.to("second").magnitude
    p_assume = sc.assumed_success

    fb_s = unwind + rebuild
    near(fb_s, res.fallback_time.to("second").magnitude, 1e-9)
    fb_usd = (reb_in * pin + reb_out * pout) / 1e6
    near(fb_usd, res.fallback_cost.to("USD").magnitude, 1e-12)
    rep_s = startup + rep_out / tps + valid
    near(rep_s, res.repair_time.to("second").magnitude, 1e-9)
    rep_usd = (rep_in * pin + rep_out * pout) / 1e6
    near(rep_usd, res.repair_cost.to("USD").magnitude, 1e-12)
    # printed values: 86.6 s, $0.162, 4.09 s, $0.00987
    pstar_s = r(rep_s, 2) / r(fb_s, 1)
    pstar_usd = r(rep_usd, 5) / r(fb_usd, 3)
    near(pstar_s, res.break_even_time, 1e-6)
    near(pstar_usd, res.break_even_cost, 1e-6)
    exp_s = r(rep_s, 2) + (1 - p_assume) * r(fb_s, 1)
    exp_usd = r(rep_usd, 5) + (1 - p_assume) * r(fb_usd, 3)
    near(exp_s, res.expected_time.to("second").magnitude, 1e-9)
    near(exp_usd, res.expected_cost.to("USD").magnitude, 1e-9)
    save = 1 - exp_s / r(fb_s, 1)
    near(save, res.time_saved_fraction, 1e-9)
    n.update(unwind=fx(unwind, 1), rebuild=fx(rebuild, 0), fbs=fx(fb_s, 1),
             rebin=comma(reb_in), rebout=comma(reb_out), pin=fx(pin, 0), pout=fx(pout, 0),
             fbusd=fx(fb_usd, 3), repin=comma(rep_in), repout=comma(rep_out), startup=fx(startup, 2),
             tps=fx(tps, 0), valid=fx(valid, 1), reps=fx(rep_s, 2), repusd=fx(rep_usd, 5),
             pstars=pct(pstar_s, 1), pstarusd=pct(pstar_usd, 1), passume=pct(p_assume),
             passumefrac=fx(p_assume, 1), compassume=fx(1 - p_assume, 1),
             exps=fx(exp_s, 1), expusd=fx(exp_usd, 4), saves=pct(save))

    # ---- Your turn. ASSUMED: same repair, but this failure class's measured success is 5 percent
    p_turn = 0.05
    t_exp_s = r(rep_s, 2) + (1 - p_turn) * r(fb_s, 1)
    t_exp_usd = r(rep_usd, 5) + (1 - p_turn) * r(fb_usd, 3)
    assert pstar_s < p_turn < pstar_usd                                # time favors repair, dollars do not
    assert t_exp_s < r(fb_s, 1) and t_exp_usd > r(fb_usd, 3)
    n.update(pturn=pct(p_turn), pturnfrac=fx(p_turn, 2), compturn=fx(1 - p_turn, 2),
             turnexps=fx(t_exp_s, 2), turnexpusd=fx(t_exp_usd, 5),
             turnsaves=fx(r(fb_s, 1) - t_exp_s, 2), turnlossusd=fx(t_exp_usd - r(fb_usd, 3), 5),
             turnworst=fx(r(rep_s, 2) + r(fb_s, 1), 2))                 # failed repair, then the fallback

    # ---- Front: expected irreversible loss around a pivot (chapter: PivotLoss)
    pv = SC.RemediationCanaryPivotLoss
    pres = pv.result
    wl = pv.workload
    p_bad, impact = wl.wrong_change_probability, wl.wrong_change_impact.to("USD").magnitude
    v, rho = wl.canary_catch_rate, wl.canary_recovery_fraction
    l_first = p_bad * impact
    l_canary = p_bad * impact * (1 - v * rho)
    near(l_first, pres.loss_without_check.to("USD").magnitude, 1e-9)
    near(l_canary, pres.loss_with_check.to("USD").magnitude, 1e-9)
    missed = p_bad * (1 - v) * impact
    near(missed / l_canary, pres.missed_share, 1e-12)
    n.update(pbad=pct(p_bad), impact=comma(impact), catch=pct(v), revert=pct(rho),
             lfirst=comma(l_first), lcanary=fx(l_canary, 2),
             lreduce=fx(l_first / l_canary, 1), missedshare=pct(missed / l_canary, 1))

    # ---- Front + mistake: cost of an undetected loop (chapter: LoopCost)
    lp = SC.RepairLoopToTurnCeiling
    lres = lp.result
    first, grow = lp.first_input, lp.output_per_turn + lp.observation_per_turn
    cap, stop = lp.ceiling, lp.stop_after
    in_cap, in_stop = resend(first, grow, cap), resend(first, grow, stop)
    assert (in_cap, in_stop, grow) == (lres.ceiling_input, lres.stop_input, lres.appended_per_turn)
    usd_cap = lres.ceiling_cost.to("USD").magnitude
    usd_stop = lres.stop_cost.to("USD").magnitude
    near(usd_cap, (in_cap * 3.0 + cap * lp.output_per_turn * 15.0) / 1e6, 1e-9)        # prices from the scenario
    near(lp.input_price.to("USD / Mtoken").magnitude, 3.0, 1e-12)
    near(lp.output_price.to("USD / Mtoken").magnitude, 15.0, 1e-12)
    ratio = r(usd_cap, 2) / r(usd_stop, 4)
    near(ratio, lres.cost_ratio, 0.1)               # printed ratio rounds to the same whole number
    n.update(loopfirst=comma(first), loopgrow=comma(grow), loopcap=f"{cap}", loopstop=f"{stop}",
             loopincap=comma(in_cap), loopinstop=comma(in_stop), loopinratio=fx(in_cap / in_stop, 0),
             loopusdcap=fx(usd_cap, 2), loopusdstop=fx(usd_stop, 4), loopratio=fx(ratio, 0),
             loopcapk=fx(in_cap / 1e6, 1))                         # millions of tokens, curve label
    # curve drawing values (thousands of tokens, scaled; pgfmath stays far below 16,384)
    n.update(curvefirst=fx(first / 1000, 1), curvegrow=fx(grow / 1000, 1),
             curvecap=fx(in_cap / 1000, 1), curvestop=fx(in_stop / 1000, 1))

    # ---- Front table + mistake: retry layers (chapter: RetryLayers) and bulkhead (ContainmentCapacity)
    rl = SC.ThreeLayerRetryAmplification
    layers, att = len(rl.layer_names), rl.attempts_per_layer
    assert att ** layers == rl.result.layered and att == rl.result.coordinated
    n.update(layers=f"{layers}", attempts=f"{att}", layered=f"{att ** layers}", coordinated=f"{att}")
    bh = SC.BlockingCallBulkhead
    bres = bh.result
    assert bh.workers - bh.worker_cap == bres.protected_workers
    assert bh.workers - bh.arrivals == bres.uncapped_free
    n.update(bhworkers=f"{bh.workers}", bharrivals=f"{bh.arrivals}", bhcap=f"{bh.worker_cap}",
             bhfree=f"{bres.uncapped_free}", bhprotected=f"{bres.protected_workers}")
    jt = SC.RetryCohortJitter
    jres = jt.result
    n.update(jitsync=comma(jres.synchronized_rate.to("1/second").magnitude),
             jitmean=comma(jres.jittered_mean_rate.to("1/second").magnitude))

    # ---- Check yourself
    # Q1 ASSUMED: a pivot plan with 5% wrong changes, $8,000 impact, 70% canary catch, 90% revert
    q_bad, q_imp, q_v, q_rho = 0.05, 8000, 0.70, 0.90
    q_first = q_bad * q_imp
    q_vr = r(q_v * q_rho, 2)
    q_canary = q_first * (1 - q_vr)
    q_missed = q_bad * (1 - q_v) * q_imp
    near(q_canary, q_bad * q_imp * (1 - q_v * q_rho), 1e-9)
    n.update(qbad=pct(q_bad), qbadfrac=fx(q_bad, 2), qimpact=comma(q_imp), qcatch=pct(q_v),
             qcatchfrac=fx(q_v, 1), qrevert=pct(q_rho), qrevertfrac=fx(q_rho, 1),
             qvr=fx(q_vr, 2), qcompvr=fx(1 - q_vr, 2), qfirst=comma(q_first), qcanary=comma(q_canary),
             qreduce=fx(q_first / q_canary, 1), qmisscompfrac=fx(1 - q_v, 1), qmissed=comma(q_missed),
             qmissshare=pct(q_missed / q_canary))

    # Q2 ASSUMED: first input 1,500 tokens, 1,000 appended per turn, 30-turn ceiling, stop after turn 5
    l_first, l_grow, l_cap, l_stop = 1500, 1000, 30, 5
    l_stop2 = 2 * l_stop
    s_cap, s_stop, s_stop2 = (resend(l_first, l_grow, t) for t in (l_cap, l_stop, l_stop2))
    n.update(lfirstq=comma(l_first), lgrowq=comma(l_grow), lcapq=f"{l_cap}", lcapmq=f"{l_cap - 1}",
             lstopq=f"{l_stop}", lstopmq=f"{l_stop - 1}", lstoptwo=f"{l_stop2}", lstoptwom=f"{l_stop2 - 1}",
             scapq=comma(s_cap), sstopq=comma(s_stop), sstoptwo=comma(s_stop2),
             sratioq=fx(s_cap / s_stop, 1), stworatio=fx(s_stop2 / s_stop, 1),
             two="2")

    # Q4 ASSUMED: two layers, four attempts each; 200 blocking calls on a 256-worker pool, cap 48
    q_layers, q_att = 2, 4
    q_pool, q_calls, q_cap = 256, 200, 48
    n.update(qlayers=f"{q_layers}", qatt=f"{q_att}", qlayered=f"{q_att ** q_layers}",
             qpool=f"{q_pool}", qcalls=f"{q_calls}", qcap=f"{q_cap}",
             qfree=f"{q_pool - q_calls}", qprotected=f"{q_pool - q_cap}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
