"""Numbers for The Foundation Model study sheet (Volume III, Chapter 2).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(CallPrice, InvocationTokenBudget, CandidateCheckerPosterior), and written as \\setnum{key}{value}
to foundation_model_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
from pathlib import Path

from mlsysim import Scenarios as SC


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def c(x):
    """Integer with thousands separators, in LaTeX ({,} keeps math spacing right)."""
    return f"{x:,.0f}".replace(",", "{,}")


def numbers() -> dict:
    n = {}

    # ---- Per-token floors (chapter: CallPrice, Scenarios.Llama3_70B_Int8DecodeH100)
    floors = SC.Llama3_70B_Int8DecodeH100
    fr = floors.result
    params_b = floors.workload.parameters.to("param").magnitude / 1e9
    peak = floors.hardware.compute.precision_flops["fp8"].to("TFLOP/s").magnitude
    bw = floors.hardware.memory.bandwidth.to("TB/s").magnitude
    weights_gb = fr.weight_bytes.to("GB").magnitude            # model weight bytes (decimal GB), not device memory
    t_pre_us = fr.compute_time.to("microsecond").magnitude
    t_dec_ms = fr.memory_time.to("millisecond").magnitude
    ratio = fr.bound_ratio
    # printed (rounded) inputs used by every later calculation
    t_pre_ = round(t_pre_us, 2)          # 71.35 us
    t_dec_ = round(t_dec_ms, 3)          # 21.075 ms
    near(t_pre_, 2 * params_b * 1e9 / (peak * 1e12) * 1e6, 0.01)
    near(t_dec_, weights_gb / (bw * 1e3) * 1e3, 0.001)
    near(t_dec_ / (t_pre_ / 1000), ratio, 1.0)
    n.update(params=f"{params_b:.1f}", peak=c(peak), bw=f"{bw:.2f}", weightsgb=f"{weights_gb:.1f}",
             tpre=f"{t_pre_:.2f}", tdec=f"{t_dec_:.3f}", ratio=f"{ratio:.0f}")

    # ---- Worked example: whole file vs localized edit under a 15 s deadline (chapter: CallPrice)
    full, diff = SC.CodingAgentWholeFileEdit, SC.CodingAgentLocalizedEdit
    deadline = full.deadline.to("second").magnitude
    pf, kf = full.prompt_tokens + full.instruction_tokens, full.output_tokens
    pd, kd = diff.prompt_tokens + diff.instruction_tokens, diff.output_tokens
    f_pre, f_dec = pf * t_pre_ / 1e6, kf * t_dec_ / 1e3
    d_pre, d_dec = pd * t_pre_ / 1e6, kd * t_dec_ / 1e3
    f_tot, d_tot = round(f_pre, 2) + round(f_dec, 2), round(d_pre, 2) + round(d_dec, 2)
    near(f_tot, full.result.total_time.to("second").magnitude, 0.01)
    near(d_tot, diff.result.total_time.to("second").magnitude, 0.01)
    assert f_tot > deadline > d_tot
    applies, passes = diff.stage_success["applies"], diff.stage_success["passes tests"]
    yld = applies * passes
    near(yld, diff.result.acceptance, 1e-9)
    n.update(deadline=f"{deadline:.0f}", base=c(full.prompt_tokens),
             fullinstr=c(full.instruction_tokens), diffinstr=c(diff.instruction_tokens),
             fullprompt=c(pf), fullout=c(kf), diffprompt=c(pd), diffout=c(kd),
             fullpre=f"{f_pre:.2f}", fulldec=f"{f_dec:.2f}", fulltot=f"{f_tot:.1f}",
             diffpre=f"{d_pre:.2f}", diffdec=f"{d_dec:.2f}", difftot=f"{d_tot:.1f}",
             fulloutshare=f"{f_dec / f_tot * 100:.0f}",
             applies=f"{applies * 100:.0f}", passes=f"{passes * 100:.0f}", yieldpct=f"{yld * 100:.0f}",
             # bar lengths for the drawing, in seconds
             barfullpre=f"{f_pre:.4f}", barfulldec=f"{f_dec:.4f}", bardiffpre=f"{d_pre:.4f}",
             bardiffdec=f"{d_dec:.4f}")

    # ---- Your turn ASSUMED: the repository context grows to 120,000 prompt tokens (one condition changed);
    # format instructions, output lengths, floors and the deadline stay as in the worked example.
    ybase = 120_000
    yfp, ydp = ybase + full.instruction_tokens, ybase + diff.instruction_tokens
    yf_pre, yd_pre = round(yfp * t_pre_ / 1e6, 2), round(ydp * t_pre_ / 1e6, 2)
    yf_tot, yd_tot = yf_pre + round(f_dec, 2), yd_pre + round(d_dec, 2)
    assert yf_tot > deadline > yd_tot
    n.update(ytbase=c(ybase), ytfullprompt=c(yfp), ytdiffprompt=c(ydp), ytfullpre=f"{yf_pre:.2f}",
             ytdiffpre=f"{yd_pre:.2f}", ytfulltot=f"{yf_tot:.1f}", ytdifftot=f"{yd_tot:.1f}",
             ytdiffpreshare=f"{yd_pre / yd_tot * 100:.0f}", ytmargin=f"{deadline - yd_tot:.1f}")

    # ---- Front curve and Your turn: append-only resend (chapter: CodingAgentResend30Turns / 60Turns)
    r30, r60 = SC.CodingAgentResend30Turns, SC.CodingAgentResend60Turns
    s0, delta, kturn = r30.first_prompt, r30.appended, r30.output_per_turn
    assert (r60.first_prompt, r60.appended, r60.output_per_turn) == (s0, delta, kturn)

    def sent(h):
        return h * s0 + delta * h * (h - 1) // 2

    def fresh(h):
        return s0 + (h - 1) * delta
    h30, h60 = r30.turns, r60.turns
    assert sent(h30) == r30.result.resent_tokens and fresh(h30) == r30.result.reused_prefix_tokens
    assert sent(h60) == r60.result.resent_tokens and fresh(h60) == r60.result.reused_prefix_tokens
    pre_sent60, pre_fresh60 = sent(h60) * t_pre_ / 1e6, fresh(h60) * t_pre_ / 1e6
    dec60 = h60 * kturn * t_dec_ / 1e3
    near(pre_sent60, r60.result.prefill_time.to("second").magnitude, 0.05)
    near(pre_fresh60, r60.result.reused_prefill_time.to("second").magnitude, 0.01)
    near(dec60, r60.result.decode_time.to("second").magnitude, 0.05)
    n.update(sbase=c(s0), delta=c(delta), kturn=c(kturn),
             hshort=f"{h30}", hlong=f"{h60}", hlongless=f"{h60 - 1}",
             sentshort=c(sent(h30)), freshshort=c(fresh(h30)),
             sentshortk=f"{sent(h30) / 1e3:.1f}", freshshortk=f"{fresh(h30) / 1e3:.1f}",
             sentlong=c(sent(h60)), freshlong=c(fresh(h60)),
             presentlong=f"{pre_sent60:.0f}", prefreshlong=f"{pre_fresh60:.1f}", declong=f"{dec60:.0f}",
             growthsent=f"{sent(h60) / sent(h30):.1f}", growthfresh=f"{fresh(h60) / fresh(h30):.1f}")
    # curve samples for the front picture (thousands of tokens), turn h = 1..30
    n["curvesent"] = " ".join(f"({h},{sent(h) / 1e3:.2f})" for h in range(1, h30 + 1))
    n["curvefresh"] = " ".join(f"({h},{fresh(h) / 1e3:.2f})" for h in range(1, h30 + 1))

    # ---- Check yourself
    # Q1 ASSUMED: a 65,536-token window with 4,096 reserved for output, the front curve's growth
    # (first prompt s0, delta appended per turn), and a 20-turn trajectory for the resend count.
    win, res, hq = 65_536, 4_096, 20
    ceil = win - res
    last = (ceil - s0) // delta + 1                       # last turn t with s0 + (t-1)*delta <= ceil
    assert s0 + (last - 1) * delta <= ceil < s0 + last * delta
    half = res // 2
    ceil_half = win - half
    last_half = (ceil_half - s0) // delta + 1
    assert s0 + (last_half - 1) * delta <= ceil_half < s0 + last_half * delta
    n.update(qwin=c(win), qres=c(res), qceil=c(ceil), qlast=f"{last}", qlastprompt=c(s0 + (last - 1) * delta),
             qnext=f"{last + 1}", qnextprompt=c(s0 + last * delta), qhalf=c(half), qceilhalf=c(ceil_half),
             qlasthalf=f"{last_half}", qnexthalf=f"{last_half + 1}", qturns=f"{hq}", qsent=c(sent(hq)), qfresh=c(fresh(hq)),
             qreuse=f"{sent(hq) / fresh(hq):.1f}")

    # Q2 ASSUMED: next-token probabilities at unit temperature; top-p 0.90
    probs, topp = [0.62, 0.25, 0.09, 0.04], 0.90
    cum, keep = 0.0, []
    for q in probs:
        keep.append(q)
        cum += q
        if cum >= topp - 1e-12:
            break
    assert len(keep) == 3
    mass = sum(keep)
    n.update(qpa=f"{probs[0]:.2f}", qpb=f"{probs[1]:.2f}", qpc=f"{probs[2]:.2f}", qpd=f"{probs[3]:.2f}",
             qtopp=f"{topp:.2f}", qtwo=f"{probs[0] + probs[1]:.2f}", qmass=f"{mass:.2f}",
             qra=f"{probs[0] / mass:.2f}", qrb=f"{probs[1] / mass:.2f}", qrc=f"{probs[2] / mass:.2f}")

    # Q2(b) ASSUMED: the same four tokens at temperature 0.5 (chapter's sharp case, eq-temperature-softmax).
    # Pr is proportional to exp(z/tau) and probs are proportional to exp(z), so tau = 0.5 squares each weight.
    tau_q = 0.5
    sq = [q ** (1 / tau_q) for q in probs]
    sq_r = [float(f"{w:.4f}") for w in sq]                  # printed weights, 4 places
    sq_sum = sum(sq_r)
    top_sharp = sq_r[0] / sq_sum                             # as the sheet computes it from printed values
    near(top_sharp, sq[0] / sum(sq), 5e-4)
    n.update(qtau=f"{tau_q:.1f}", qsqa=f"{sq_r[0]:.4f}", qsqsum=f"{sq_sum:.4f}", qtopsharp=f"{top_sharp:.2f}")

    # Q4 ASSUMED: 20 percent of patches defective; the check passes defective patches with 0.5, valid with 0.9.
    eps, fa, va = 0.20, 0.50, 0.90
    post = fa * eps / (fa * eps + va * (1 - eps))
    n.update(qdef=f"{eps * 100:.0f}", qfa=f"{fa:.1f}", qva=f"{va:.1f}",
             qnum=f"{fa * eps:.2f}", qvalid=f"{va * (1 - eps):.2f}", qpost=f"{post * 100:.1f}",
             qkeep=f"{1 - eps:.1f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
