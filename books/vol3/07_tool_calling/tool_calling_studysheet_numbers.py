"""Numbers for the Tool Calling study sheet (Volume III, Chapter 7).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(SettlementBudget, PollingBudget, ToolkitCatalogBudget), and written as \\setnum{key}{value} to
tool_calling_studysheet_numbers.tex next to this file.
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


def ints(x):
    return f"{round(x):,}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def p_timeout(q_req, q_rep):
    return 1 - (1 - q_req) * (1 - q_rep)


def p_duplicate(q_req, q_rep):
    """Original delivered, its reply lost, the one blind retry delivered (chapter's notebook)."""
    return (1 - q_req) * q_rep * (1 - q_req)


def numbers() -> dict:
    n = {}

    # ---- Worked example: which change removes duplicates? (chapter: SettlementBudget,
    #      Scenarios.BlindRetryTwoPercentLoss, nbk-07-settlement-budget)
    sb = SC.BlindRetryTwoPercentLoss
    sres = sb.result
    q_req, q_rep, workers, per_worker = sb.request_loss, sb.reply_loss, sb.workers, sb.calls_per_worker
    calls = workers * per_worker
    assert calls == sres.calls
    t = p_timeout(q_req, q_rep)
    d = p_duplicate(q_req, q_rep)
    near(t, sres.timeout_probability, 1e-12)
    near(d, sres.duplicate_probability, 1e-12)
    dup_day = calls * d
    near(dup_day, sres.duplicates, 1e-6)
    d_half_rep = p_duplicate(q_req, q_rep / 2)               # halve reply loss
    d_half_req = p_duplicate(q_req / 2, q_rep)               # halve request loss
    near(calls * d_half_rep, dup_day / 2, 1e-6)
    assert calls * d_half_req > dup_day                      # halving request loss adds duplicates
    n.update(wqreq=pct(q_req), wqrep=pct(q_rep), wqreqfrac=fx(q_req, 2), wqrepfrac=fx(q_rep, 2),
             wkeep=fx(1 - q_req, 2), workers=ints(workers), perworker=ints(per_worker), wcalls=ints(calls),
             wtimeout=pct(t, 2), wdup=fx(d, 6), wdupday=ints(dup_day),
             whalfrep=fx(q_rep / 2, 2), whalfreq=fx(q_req / 2, 2), wkeephalf=fx(1 - q_req / 2, 2),
             wduphalfrep=fx(d_half_rep, 6), wduphalfrepday=ints(calls * d_half_rep),
             wduphalfreq=fx(d_half_req, 6), wduphalfreqday=ints(calls * d_half_req),
             wdupratio=fx(d / t, 2), wdupsharepct=pct(d / t))
    # bar lengths for the picture (plain numbers)
    n.update(barbase=f"{dup_day:.1f}", barhalfreq=f"{calls * d_half_req:.1f}",
             barhalfrep=f"{calls * d_half_rep:.1f}")

    # ---- Your turn. ASSUMED: the same 200,000 mutating calls a day with the loss split changed:
    #      request loss 0.5 percent, reply loss 4 percent, one blind retry per timeout.
    y_req, y_rep, y_calls = 0.005, 0.04, calls
    y_t = p_timeout(y_req, y_rep)
    y_d = p_duplicate(y_req, y_rep)
    y_d_half = p_duplicate(y_req, y_rep / 2)
    n.update(yqreq=fx(y_req * 100, 1), yqrep=pct(y_rep), yqreqfrac=fx(y_req, 3), yqrepfrac=fx(y_rep, 2),
             ykeep=fx(1 - y_req, 3), ykeeprep=fx(1 - y_rep, 2), ycalls=ints(y_calls), ytimeout=pct(y_t, 2),
             ydup=fx(y_d, 6), ydupday=ints(y_calls * y_d), yhalfrep=fx(y_rep / 2, 2),
             yduphalf=fx(y_d_half, 6), yduphalfday=ints(y_calls * y_d_half),
             ysharepct=pct(y_d / y_t))

    # ---- Front card + Q1: polling cost (chapter: PollingBudget, Scenarios.ModelPollingEveryFiveSeconds)
    pb = SC.ModelPollingEveryFiveSeconds
    pres = pb.result
    job = pb.job_duration.to("s").magnitude
    every = pb.poll_interval.to("s").magnitude
    ctx = pb.context_tokens
    out_tok = pb.output_tokens
    c_in = pres.blended_input_price.to("USD/Mtoken").magnitude          # cache-blended, $ per million
    c_unc = pb.input_price.to("USD/Mtoken").magnitude
    c_cache = pres.cached_price.to("USD/Mtoken").magnitude
    c_out = pb.output_price.to("USD/Mtoken").magnitude
    near(c_in, pb.cache_hit * c_cache + (1 - pb.cache_hit) * c_unc, 1e-12)
    polls = math.ceil(job / every) - 1
    assert polls == pres.polls
    per_call = (ctx * c_in + out_tok * c_out) / 1e6
    near(polls * per_call, pres.cost.to("USD").magnitude, 1e-9)
    n.update(pjob=f"{job:.0f}", pevery=f"{every:.0f}", ppolls=f"{polls}", pctx=ints(ctx), pout=f"{out_tok}",
             pcin=fx(c_in, 3), pcout=fx(c_out, 0), pcache=pct(pb.cache_hit), pcunc=fx(c_unc, 0),
             pccached=fx(c_cache, 2), pcost=fx(polls * per_call, 2))

    # Q1 ASSUMED: a 600 s job polled every 5 s, then every 30 s; same context, output and prices.
    q_job, q_every, q_every2 = 600, 5, 30
    q_polls = math.ceil(q_job / q_every) - 1
    q_polls2 = math.ceil(q_job / q_every2) - 1
    per_call_r = r(per_call, 5)                                        # printed per-call cost
    n.update(qjob=f"{q_job}", qevery=f"{q_every}", qeverytwo=f"{q_every2}", qpolls=f"{q_polls}",
             qpollstwo=f"{q_polls2}", qpercall=fx(per_call, 5),
             qinpart=fx(ctx * c_in / 1e6, 5), qoutpart=fx(out_tok * c_out / 1e6, 5),
             qcost=fx(q_polls * per_call_r, 2), qcosttwo=fx(q_polls2 * per_call_r, 2))
    near(q_polls * per_call_r, q_polls * per_call, 0.005)
    near(r(ctx * c_in / 1e6, 5) + r(out_tok * c_out / 1e6, 5), per_call_r, 1e-9)

    # ---- Q3 ASSUMED: head-and-tail budget of 8,192 bytes, 192-byte notice, 2,048-byte head; a 3,000,000-byte
    #      failing log whose 1,200-byte panic is followed by 4,000 bytes of teardown and summary.
    m_obs, m_notice, m_head = 8_192, 192, 2_048
    m_tail = m_obs - m_notice - m_head
    log, panic, after = 3_000_000, 1_200, 4_000
    reach = panic + after                                              # panic starts this far from the end
    assert m_tail >= reach
    omitted = log - m_head - m_tail
    after2 = 6_000                                                     # ASSUMED: teardown grows
    reach2 = panic + after2
    assert m_tail < reach2
    head_max2 = m_obs - m_notice - reach2
    n.update(hobs=ints(m_obs), hnotice=ints(m_notice), hhead=ints(m_head), htail=ints(m_tail),
             hlog=ints(log), hpanic=ints(panic), hafter=ints(after), hreach=ints(reach),
             homit=ints(omitted), hafterlong=ints(after2), hreachlong=ints(reach2), hheadmax=ints(head_max2))

    # ---- Mistake: stage every tool (chapter: ToolkitCatalogBudget, Scenarios.FiftyToolCatalogSixSelected)
    cb = SC.FiftyToolCatalogSixSelected
    cres = cb.result
    full = cb.registered * cb.descriptor_tokens
    sel = cb.selected * cb.descriptor_tokens + cb.selection_overhead
    assert full == cres.full_tokens and sel == cres.selected_tokens
    n.update(creg=f"{cb.registered}", csel=f"{cb.selected}", cdesc=f"{cb.descriptor_tokens}",
             cover=f"{cb.selection_overhead}", cfull=ints(full), cselected=ints(sel),
             creleased=ints(full - sel))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
