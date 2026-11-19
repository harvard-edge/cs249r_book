"""Numbers for the Agent Economics study sheet (Volume III, Chapter 19).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(TokenPriceFallacy, CriticalPath, PrefixCacheCost, QueuePrice, CascadeCost, EconomicsFallacies),
and written as \\setnum{key}{value} to agent_economics_studysheet_numbers.tex next to this file.
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


def usd(q):
    return q.to("USD").magnitude


def numbers() -> dict:
    n = {}

    # ---- Worked example: the token-price fallacy cohort (chapter: TokenPriceFallacy)
    cmp_ = SC.RepairCohortTokenPriceComparison
    a, b = cmp_.baseline, cmp_.alternative
    ra, rb = cmp_.result.baseline, cmp_.result.alternative
    tasks = a.tasks
    assert b.tasks == tasks
    env_turn, assess = usd(a.environment_per_turn), usd(a.assessment_per_attempt)
    assert usd(b.environment_per_turn) == env_turn and usd(b.assessment_per_attempt) == assess

    def model_cost(s):
        per_turn = (s.input_tokens_per_turn * s.input_price.to("USD / Mtoken").magnitude
                    + s.output_tokens_per_turn * s.output_price.to("USD / Mtoken").magnitude) / 1e6
        return s.turns * per_turn

    a_model, b_model = model_cost(a), model_cost(b)
    near(a_model, ra.model_cost, 1e-9)
    near(b_model, rb.model_cost, 1e-9)
    a_env, b_env = a.turns * env_turn, b.turns * env_turn
    a_att = r(a_model, 2) + r(a_env, 2) + assess                       # printed parts, summed
    b_att = r(b_model, 3) + r(b_env, 2) + assess
    near(a_att, ra.attempt_cost, 1e-9)
    near(b_att, rb.attempt_cost, 0.0005)
    a_ceff = r(a_att, 2) * tasks / a.accepted
    b_ceff = r(b_att, 3) * tasks / b.accepted
    near(a_ceff, ra.cost_per_accepted, 0.0005)
    near(b_ceff, rb.cost_per_accepted, 0.0006)
    ratio = r(b_ceff, 3) / r(a_ceff, 3)
    near(ratio, cmp_.result.cost_ratio, 0.01)
    n.update(tickets=f"{tasks}", aturns=f"{a.turns}", bturns=f"{b.turns}",
             ain=f"{a.input_tokens_per_turn:,}", aout=f"{a.output_tokens_per_turn:,}",
             bin=f"{b.input_tokens_per_turn:,}", bout=f"{b.output_tokens_per_turn:,}",
             ainprice=fx(a.input_price.to("USD / Mtoken").magnitude, 0),
             aoutprice=fx(a.output_price.to("USD / Mtoken").magnitude, 0),
             binprice=fx(b.input_price.to("USD / Mtoken").magnitude, 2),
             boutprice=fx(b.output_price.to("USD / Mtoken").magnitude, 2),
             aaccepted=f"{a.accepted}", baccepted=f"{b.accepted}",
             envturn=fx(env_turn, 2), assess=fx(assess, 2),
             amodel=fx(a_model, 2), aenv=fx(a_env, 2), aattempt=fx(a_att, 2),
             bmodel=fx(b_model, 3), benv=fx(b_env, 2), battempt=fx(b_att, 3),
             acohort=fx(r(a_att, 2) * tasks, 2), bcohort=fx(r(b_att, 3) * tasks, 2),
             aceff=fx(a_ceff, 3), bceff=fx(b_ceff, 3), ceffratio=fx(ratio, 1),
             modelcut=pct(cmp_.result.model_cut), attemptcut=pct(cmp_.result.attempt_cut),
             acceptcut=pct(cmp_.result.acceptance_cut),
             arejper=fx(ra.rejected_per_accepted, 2), brejper=fx(rb.rejected_per_accepted, 0))
    # For the to-scale bars: each attempt term divided by acceptance (dollars per accepted task)
    pa, pb = a.accepted / tasks, b.accepted / tasks
    n.update(barAmodel=f"{a_model / pa:.4f}", barAenv=f"{a_env / pa:.4f}", barAassess=f"{assess / pa:.4f}",
             barBmodel=f"{b_model / pb:.4f}", barBenv=f"{b_env / pb:.4f}", barBassess=f"{assess / pb:.4f}")

    # ---- Your turn. ASSUMED: a tool fix lifts the low-price configuration to 50 accepted, all else equal
    fix_acc = 50
    b_coh = r(b_att, 3) * tasks
    fix_ceff = b_coh / fix_acc
    need = b_coh / r(a_ceff, 3)
    need_whole = math.ceil(need)
    assert fix_ceff > a_ceff and b_coh / need_whole <= ra.cost_per_accepted < b_coh / (need_whole - 1)
    n.update(fixaccepted=f"{fix_acc}", fixceff=fx(fix_ceff, 3), needacc=fx(need, 1), needwhole=f"{need_whole}")

    # ---- Front: critical-path ceiling (chapter: CriticalPath, Scenarios.SerialRepairEpisode)
    ep = SC.SerialRepairEpisode
    er = ep.result
    sec = lambda q: q.to("second").magnitude
    t_model, t_tests, t_life, t_rt = sec(er.model_time), sec(er.tests_time), sec(er.lifecycle_time), sec(er.runtime_time)
    t_total, t_other = sec(er.total_time), sec(er.other_time)
    g = ep.model_speedup
    t_new = r(t_other, 2) + r(t_model, 2) / g                          # printed parts at two decimals
    near(t_new, sec(er.changed_time), 0.006)
    near(r(t_tests, 2) + r(t_life, 2) + r(t_rt, 2), r(t_other, 2), 1e-9)
    near(r(t_other, 2) + r(t_model, 2), r(t_total, 2), 1e-9)
    n.update(cptests=fx(t_tests, 2), cpsandbox=fx(t_life, 2), cpmodel=fx(t_model, 2), cpruntime=fx(t_rt, 2),
             cptotal=fx(t_total, 2), cpother=fx(t_other, 2), cpfrac=pct(er.model_fraction, 1),
             cpgain=fx(g, 0), cpnew=fx(t_new, 2), cpspeed=fx(r(t_total, 2) / r(t_new, 2), 2),
             cpcap=fx(r(t_total, 2) / r(t_other, 2), 2), cptesthalf=fx(t_tests / 2, 1),
             cpmodelsave=fx(r(t_total, 2) - r(t_new, 2), 2))
    near(r(t_total, 2) / r(t_new, 2), er.speedup, 0.005)
    near(r(t_total, 2) / r(t_other, 2), er.limiting_speedup, 0.005)
    for k, v in (("barTests", t_tests), ("barSandbox", t_life), ("barModel", t_model), ("barRuntime", t_rt)):
        n[k] = f"{v:.3f}"

    # ---- Mistake: caching makes context free (chapter: PrefixCacheCost, EconomicsFallacies)
    cb = SC.RepairResendCacheBill
    cr = cb.result
    n.update(cacheturns=f"{cb.trajectory.turns}", billfresh=fx(usd(cr.total_uncached), 2),
             billcached=fx(usd(cr.total_cached), 2), cacheratio=fx(cr.cache_ratio, 1),
             rereadshare=pct(cr.reread_share), missdrop=pct(cb.miss_fraction), missrise=pct(cr.miss_share))
    near(r(usd(cr.total_uncached), 2) / r(usd(cr.total_cached), 2), cr.cache_ratio, 0.05)

    # ---- Mistake: compaction always saves (chapter: PrefixCacheCost, Scenarios.RepairCompactionPayback,
    # ...PaybackHalfCachedRate). U = c_out K_sum + c_in S_comp; s = c_cache (S - S_comp); N_min = 1 + ceil(U / s)
    cp, cph = SC.RepairCompactionPayback, SC.RepairCompactionPaybackHalfCachedRate
    c_in = cp.bill.input_price.to("USD / Mtoken").magnitude
    c_out = cp.bill.output_price.to("USD / Mtoken").magnitude
    c_cache = c_in * cp.bill.cached_fraction
    ctx, comp, ksum = cp.result.context_at_compaction, cp.compact_to, cp.summary_tokens
    up = (c_out * ksum + c_in * comp) / 1e6
    save = c_cache * (ctx - comp) / 1e6
    near(up, usd(cp.result.upfront), 1e-12)
    near(save, usd(cp.result.per_call_saving), 1e-12)
    pay = 1 + math.ceil(r(up, 4) / r(save, 5))                       # from the printed, rounded U and s
    assert pay == cp.result.breakeven_calls <= cp.result.remaining_calls
    save_half = c_in * cph.bill.cached_fraction * (ctx - comp) / 1e6
    pay_half = 1 + math.ceil(r(up, 4) / r(save_half, 6))
    assert pay_half == cph.result.breakeven_calls > cp.result.remaining_calls
    n.update(cpin=fx(c_in, 0), cpout=fx(c_out, 0), cpcache=fx(c_cache, 2), cpctx=f"{ctx:,}", cpto=f"{comp:,}",
             cpsum=f"{ksum:,}", cpup=fx(up, 4), cpsave=fx(save, 5), cpratio=fx(up / save, 1), cppay=f"{pay}",
             cprest=f"{cp.result.remaining_calls}", cppayhalf=f"{pay_half}")

    # ---- Mistake: running hot (chapter: QueuePrice, Scenarios.BurstyModelServerQueuePrice)
    qp = SC.BurstyModelServerQueuePrice
    qr = qp.result
    w_lo, w_hi = sec(qr.wait_low), sec(qr.wait_high)
    near(w_lo, qr.variability * qp.low_utilization / (1 - qp.low_utilization) * sec(qp.service), 1e-9)
    n.update(rholow=pct(qp.low_utilization), rhohigh=pct(qp.high_utilization),
             thrgain=pct(qr.throughput_gain), waitlow=fx(w_lo, 0), waithigh=fx(w_hi, 0),
             waitratio=fx(r(w_hi, 0) / r(w_lo, 0), 1))
    near(r(w_hi, 0) / r(w_lo, 0), qr.wait_ratio, 0.05)

    # ---- Mistake: cascade gate (chapter: CascadeCost, EconomicsFallacies)
    cc = SC.CodeEditThreeTierCascade
    ccr = cc.result
    harm = cc.false_accept_rate * cc.accepted_by_tier[0] * usd(cc.harm_per_false_accept)
    near(harm, ccr.harm_cost, 1e-6)
    with_harm = (ccr.spend + harm) / ccr.accepted
    near(with_harm, ccr.per_accepted_with_harm, 1e-9)
    assert with_harm > ccr.direct_per_accepted
    n.update(cascspend=f"{ccr.spend:,.0f}", cascacc=f"{ccr.accepted:,}", cascceff=fx(ccr.per_accepted, 3),
             directceff=fx(ccr.direct_per_accepted, 3), fapct=pct(cc.false_accept_rate, 1),
             tieronexits=f"{cc.accepted_by_tier[0]:,}", harmeach=fx(usd(cc.harm_per_false_accept), 0),
             harmtotal=f"{harm:,.0f}", cascharm=fx(with_harm, 3))

    # ---- Check yourself
    # Q1 ASSUMED: model calls 60 s, tests 150 s, sandbox 50 s on the critical path; a 3x faster model
    qm, qt, qs, qg = 60, 150, 50, 3
    qT = qm + qt + qs
    q_new = qt + qs + qm / qg
    n.update(qmodel=f"{qm}", qtests=f"{qt}", qsandbox=f"{qs}", qtotal=f"{qT}", qgain=f"{qg}",
             qfrac=pct(qm / qT), qrest=f"{qt + qs}", qmodelnew=f"{qm / qg:.0f}", qnew=f"{q_new:.0f}",
             qspeed=fx(qT / q_new, 2), qcap=fx(qT / (qt + qs), 2), qmodelsave=f"{qm - qm / qg:.0f}",
             qtestsave=f"{qt / 2:.0f}")

    # Q2 ASSUMED: support goodput 20 accepted/min at alpha 0.8; sessions 3 min, 6 serial calls;
    # one endpoint, 0.3 s mean service, (ca^2 + cs^2)/2 = 1.5; acceptance then falls to 0.7
    G, al, stay, calls = 20, 0.8, 3, 6
    vt, svc, al2 = 1.5, 0.3, 0.7
    lam = G / al
    call_rate_s = lam * calls / 60                                       # calls per second
    rho = call_rate_s * svc
    rho2 = (G / al2) * calls / 60 * svc
    near(rho2, rho * al / al2, 1e-12)
    near(rho, 0.75, 1e-12)
    w1 = vt * rho / (1 - rho) * svc
    w2 = vt * rho2 / (1 - rho2) * svc
    near(rho2 / (1 - rho2), 6.0, 1e-9)
    n.update(sgood=f"{G}", salpha=fx(al, 1), sstay=f"{stay}", scalls=f"{calls}", sadmit=f"{lam:.0f}",
             sconc=f"{lam * stay:.0f}", scallrate=f"{lam * calls:.0f}", scallsec=fx(call_rate_s, 1),
             svar=fx(vt, 1), srho=fx(rho, 2), ssvc=fx(svc, 1), sratio=fx(rho / (1 - rho), 0), swait=fx(w1, 2),
             salphatwo=fx(al2, 1), srhotwo=fx(rho2, 3), sratiotwo=fx(rho2 / (1 - rho2), 0), swaittwo=fx(w2, 1),
             sloadrise=pct(al / al2 - 1), swaitx=fx(w2 / w1, 0))

    # Q3 ASSUMED: loop at $0.40 per turn; conditional acceptance 0.5, 0.3, 0.1, 0.02; a $5 engineer resolves all
    turn, qs_, human, human2 = 0.40, (0.5, 0.3, 0.1, 0.02), 5.0, 2.0
    gains = [q * human for q in qs_]
    cap = max(i + 1 for i, v in enumerate(gains) if v > turn)
    gains2 = [q * human2 for q in qs_]
    cap2 = max(i + 1 for i, v in enumerate(gains2) if v > turn)
    assert all(v > turn for v in gains[:cap]) and all(v > turn for v in gains2[:cap2])
    n.update(lturn=fx(turn, 2), lqone=fx(qs_[0], 1), lqtwo=fx(qs_[1], 1), lqthree=fx(qs_[2], 1),
             lqfour=fx(qs_[3], 2), lhuman=fx(human, 0), lhumantwo=fx(human2, 0),
             lgone=fx(gains[0], 2), lgtwo=fx(gains[1], 2), lgthree=fx(gains[2], 2), lgfour=fx(gains[3], 2),
             lcap=f"{cap}", lhone=fx(gains2[0], 2), lhtwo=fx(gains2[1], 2), lhthree=fx(gains2[2], 2),
             lcaptwo=f"{cap2}")
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    n.update(lcapword=words[cap], lcaptwoword=words[cap2])

    # Q4 ASSUMED: child allocation $4.00, $1.10 settled, a remote job bounded at $1.50 at local stop; bills $0.90 late
    alloc, settled, bound, late = 4.00, 1.10, 1.50, 0.90
    assert late <= bound
    n.update(balloc=fx(alloc, 2), bsettled=fx(settled, 2), bbound=fx(bound, 2), blate=fx(late, 2),
             bsafe=fx(alloc - settled - bound, 2), bnaive=fx(alloc - settled, 2),
             bclose=fx(alloc - settled - late, 2), bmore=fx(bound - late, 2))
    near(alloc - settled - bound + (bound - late), alloc - settled - late, 1e-9)
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
