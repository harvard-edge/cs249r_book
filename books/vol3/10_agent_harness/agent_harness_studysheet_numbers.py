"""Numbers for The Agent Harness study sheet (Volume III, Chapter 10).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(InstructionHold, BoundaryWait, EscrowByWorkload, DrrShares, Admission), and written as
\\setnum{key}{value} to agent_harness_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
Memory follows the registry: the workspace lease is MiB, so totals print as GiB (GB).
"""
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def grp(x, nd=0):
    """Thousands separator, as the chapter prints counts."""
    return f"{r(x, nd):,.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


MIB_PER_GIB = 1024
BYTES_PER_MIB = 2 ** 20
PER_MILLION = 1e6


def numbers() -> dict:
    n = {}

    # ---- Key idea: a rule kept per turn, over the horizon (chapter: InstructionHold)
    rule = SC.FreezeRuleOverStepBudget
    turns, follow = rule.short_horizon, rule.step_success
    hold = rule.result.success[turns]
    near(hold, follow ** turns, 1e-12)
    n.update(follow=pct(follow), ruleturns=f"{turns}", hold=pct(hold), brk=pct(1 - hold))

    # ---- Turn-boundary card (chapter: BoundaryWait, Scenarios.CanceledStreamDecodeSavings)
    call = SC.CanceledStreamDecodeSavings
    kmax = call.output_allowance
    tdec_ms = call.floors.result.memory_time.to("ms").magnitude
    twait = call.allowance_decode_time.to("s").magnitude
    twait_printed = kmax * r(tdec_ms, 2) / 1000                    # from the printed 21.07 ms
    near(r(twait_printed, 1), r(twait, 1), 1e-9)
    n.update(kmax=grp(kmax), tdec=fx(tdec_ms, 2), twait=fx(twait, 1))

    # ---- Little's law card (chapter: EscrowByWorkload, Scenarios.RepairRemediationApprovalEscrow)
    esc = SC.RepairRemediationApprovalEscrow
    lam = esc.repair_rate.to("1/hour").magnitude
    wait_h = esc.repair_wait.to("hour").magnitude
    lease_mib = esc.lease.to("MiB").magnitude
    held = lam * wait_h
    near(held, esc.result.held, 1e-9)
    ret_gib = held * lease_mib / MIB_PER_GIB
    near(ret_gib, esc.result.retained_memory.to("GiB").magnitude, 1e-9)
    ret_gb = esc.result.retained_memory.to("GB").magnitude
    n.update(lam=fx(lam, 0), waith=fx(wait_h, 0), held=fx(held, 0), lease=fx(lease_mib, 0),
             retgib=fx(ret_gib, 0), retgb=fx(ret_gb, 1))

    # ---- DRR card (chapter: DrrShares, Scenarios.InteractiveMaintenanceDrr)
    drr = SC.InteractiveMaintenanceDrr
    dres = drr.result
    q = drr.quantum
    ci, cm = drr.classes["interactive"], drr.classes["maintenance"]
    chg_i, chg_m = ci["input"] + ci["output"], cm["input"] + cm["output"]
    near(chg_i, dres.charge_per_call["interactive"], 1e-9)
    near(chg_m, dres.charge_per_call["maintenance"], 1e-9)
    calls_i = ci["weight"] * q / chg_i
    rounds_m = chg_m / (cm["weight"] * q)
    near(calls_i, dres.calls_per_round["interactive"], 1e-9)
    near(rounds_m, dres.rounds_per_call["maintenance"], 1e-9)
    share_i = ci["weight"] / (ci["weight"] + cm["weight"])
    near(share_i, dres.charge_share["interactive"], 1e-9)
    ratio = calls_i * rounds_m
    near(ratio, dres.call_ratio, 1e-9)
    n.update(wint=f"{ci['weight']}", wmnt=f"{cm['weight']}", quantum=grp(q), chgint=grp(chg_i),
             chgmnt=grp(chg_m), callsint=fx(calls_i, 1), roundsmnt=fx(rounds_m, 2),
             shareint=pct(share_i), callratio=fx(ratio, 0))

    # ---- Worked example: the next call (chapter: Admission, Scenarios.RepairNextCallAdmission)
    adm = SC.RepairNextCallAdmission
    ares = adm.result
    allow = adm.allowance.to("USD").magnitude
    settled = adm.settled.to("USD").magnitude
    reserved = adm.reserved.to("USD").magnitude
    fin = adm.finalization_reserve.to("USD").magnitude
    inputtokens, kout = adm.input_tokens, adm.output_allowance
    pin = adm.input_price.to("USD / Mtoken").magnitude
    pout = adm.output_price.to("USD / Mtoken").magnitude
    c_in = r(inputtokens * pin / PER_MILLION, 2)
    c_out = r(kout * pout / PER_MILLION, 4)
    liab = c_in + c_out
    near(liab, ares.liability, 5e-5)
    spend_head = r(allow - settled, 2)
    near(spend_head, ares.spend_headroom, 1e-9)
    head = r(spend_head - reserved - fin, 2)
    near(head, ares.headroom, 1e-9)
    assert head < liab < spend_head
    short = liab - head
    near(short, ares.finalization_shortfall, 5e-5)
    nmax = (head - c_out) / pin * PER_MILLION
    near(nmax, ares.max_input_tokens, 50)
    resshare = reserved / allow
    near(resshare, ares.reservation_share, 1e-12)
    n.update(allow=fx(allow, 2), settled=fx(settled, 2), reserved=fx(reserved, 2), fin=fx(fin, 2),
             inputtokens=grp(inputtokens), kout=grp(kout), pin=fx(pin, 0), pout=fx(pout, 0), cin=fx(c_in, 2),
             cout=fx(c_out, 4), liab=fx(liab, 4), spendhead=fx(spend_head, 2), head=fx(head, 2),
             short=fx(short, 4), nmax=f"{round(nmax, -2):,.0f}",
             resshare=pct(resshare))

    # Bar picture: segments scaled to the allowance (pt per dollar chosen to fit the column)
    n.update(barsettled=fx(settled, 2), barreserved=fx(reserved, 2), barfin=fx(fin, 2))

    # ---- Your turn. ASSUMED: the test job settles with a $0.20 charge (its $0.50 reservation released).
    t_charge = 0.20
    settled2 = r(settled + t_charge, 2)
    head2 = r(allow - settled2 - fin, 2)
    assert liab <= head2
    nmax2 = (head2 - c_out) / pin * PER_MILLION
    n.update(tcharge=fx(t_charge, 2), settledtwo=fx(settled2, 2), headtwo=fx(head2, 2),
             nmaxtwo=f"{round(nmax2, -2):,.0f}", spare=fx(head2 - liab, 4))

    # ---- Check yourself
    # Q1 ASSUMED: 25 ms per output token, 6,000-token allowance, pause target 20 s.
    q1_t, q1_k, q1_target = 25.0, 6000, 20.0
    q1_wait = q1_k * q1_t / 1000
    q1_kmax = q1_target * 1000 / q1_t
    n.update(qtdec=fx(q1_t, 0), qk=grp(q1_k), qtarget=fx(q1_target, 0), qwait=fx(q1_wait, 0),
             qkmax=grp(q1_kmax))

    # Q2 ASSUMED: 9 review requests per hour, 40-minute mean decision; the chapter's 512 MiB lease.
    q2_lam, q2_w_min = 9, 40
    q2_held = q2_lam * q2_w_min / 60
    near(r(q2_lam / 60, 2) * q2_w_min, q2_held, 1e-9)                 # printed per-minute rate reproduces it
    q2_gib = q2_held * lease_mib / MIB_PER_GIB
    q2_gb = q2_held * lease_mib * BYTES_PER_MIB / 1e9
    q2_held_fast = q2_held / 2
    n.update(qlam=f"{q2_lam}", qlammin=fx(q2_lam / 60, 2), qwmin=f"{q2_w_min}", qheld=fx(q2_held, 0), qgib=fx(q2_gib, 0), qgb=fx(q2_gb, 1), qheldfast=fx(q2_held_fast, 0),
             qgibfast=fx(q2_held_fast * lease_mib / MIB_PER_GIB, 1))

    # Q3 ASSUMED: weights 2 and 1, quantum 10,000; support call 3,000 + 500, batch call 30,000 + 4,000;
    #    a 20,000-token cap on carried credit.
    q3_q, q3_ws, q3_wb = 10000, 2, 1
    q3_cs, q3_cb = 3000 + 500, 30000 + 4000
    q3_cap = 20000
    q3_calls_s = q3_ws * q3_q / q3_cs
    q3_rounds_b = q3_cb / (q3_wb * q3_q)
    q3_share_s = q3_ws / (q3_ws + q3_wb)
    q3_ratio = r(q3_calls_s, 1) * r(q3_rounds_b, 1)                   # from the printed values
    assert q3_cb > q3_cap > q3_cs
    n.update(qq=grp(q3_q), qws=f"{q3_ws}", qwb=f"{q3_wb}", qsin=grp(3000), qsout=grp(500),
             qbin=grp(30000), qbout=grp(4000), qcs=grp(q3_cs), qcb=grp(q3_cb), qcap=grp(q3_cap),
             qcreds=grp(q3_ws * q3_q), qcalls=fx(q3_calls_s, 1), qcallsbatch=fx(q3_wb * q3_q / q3_cb, 2), qshare=pct(q3_share_s), qsharebatch=pct(1 - q3_share_s), qratio=fx(q3_ratio, 0))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
