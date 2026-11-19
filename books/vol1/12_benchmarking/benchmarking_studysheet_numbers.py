"""Numbers for the Benchmarking study sheet (Volume I, Chapter 12).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to benchmarking_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
"""
from pathlib import Path

from mlsysim.engine.scenarios import Scenarios


def g(x, nd=3):
    return f"{x:.{nd}g}"


def c(x):
    return f"{x:,.0f}"


def pct(x, nd=0):
    return f"{x * 100:.{nd}f}"


def numbers() -> dict:
    n = {}

    # ---- Amdahl card: the chapter's preprocess-bound vision pipeline (AmdahlBenchmarkCalc)
    p = Scenarios.PreprocessBoundVisionPipeline
    pre = p.stages["preprocess"].to("ms").magnitude
    inf = p.stages["inference"].to("ms").magnitude
    sp = p.speedup
    tot, opt_inf = pre + inf, inf / sp
    opt_tot = pre + opt_inf
    frac = inf / tot
    n.update(ampre=g(pre), aminf=g(inf), amspeed=g(sp), amtotal=g(tot), amoptinf=g(opt_inf),
             amopttotal=g(opt_tot), amgain=f"{tot / opt_tot:.1f}", amfrac=f"{frac:.2f}",
             amceiling=f"{1 / (1 - frac):.2f}")
    assert abs(tot / opt_tot - p.overall_speedup) < 1e-9 and abs(1 / (1 - frac) - p.limit) < 1e-9

    # ---- Paired test: 95 % vs 94 % on the same 1,000 images (StatisticalConfidenceTrap, McNemarThreshold)
    cmp_ = Scenarios.NinetyFiveAgainstNinetyFourOnSameImages
    gap = int(cmp_.net_gap)
    crit = cmp_.chi2_critical
    n.update(mcimages=c(cmp_.n_samples), mcbase=pct(cmp_.baseline_accuracy), mccand=pct(cmp_.candidate_accuracy),
             mcbaseerr=c(cmp_.baseline_errors), mccanderr=c(cmp_.candidate_errors), mcgap=c(gap),
             mccrit=f"{crit:.2f}", mcmax=c(cmp_.max_discordant))
    assert int(gap ** 2 // crit) == cmp_.max_discordant
    # Worked example: the compressed model repeats the baseline's errors and adds `gap` new ones (chapter's case)
    wb, wc = gap, 0
    n.update(wkbase=c(wb), wkcomp=c(wc), wkstat=f"{(wb - wc) ** 2 / (wb + wc):.0f}")
    # Your turn: ASSUMED same gap, but 40 disagreements
    disagree = 40
    yb, yc = (disagree + gap) // 2, (disagree - gap) // 2
    # MLPerf Training ResNet-50 quality target as the chapter prints it (typed in its prose; not in mlsysim)
    n.update(target="75.9", ytsq=c((yb - yc) ** 2))
    n.update(ytdis=c(disagree), ytbase=c(yb), ytcomp=c(yc), ytstat=f"{(yb - yc) ** 2 / (yb + yc):.1f}")

    # ---- Duty-cycle card: keyword spotting on an ESP32-S3-class board (KwsDutyCycle)
    k = Scenarios.KeywordSpottingOnEsp32S3
    pa = k.active_power.to("mW").magnitude
    pi = k.idle_power.to("mW").magnitude
    duty = k.duty
    pavg = k.average_power.to("mW").magnitude
    n.update(kwsactive=f"{pa / 1000:.1f}", kwsidle=c(pi), kwsms=f"{k.inference_time.to('ms').magnitude:.1f}",
             kwsduty=pct(duty, 2), kwsavg=f"{pavg:.1f}", kwsidleshare=pct(k.idle_share),
             kwsactshare=pct(1 - k.idle_share), kwsidlemw=f"{(1 - duty) * pi:.1f}", kwsactmw=f"{duty * pa:.1f}")
    assert abs(duty * pa + (1 - duty) * pi - pavg) < 1e-9

    # ---- Training card: scaling efficiency (ScalingEfficiencyCalc) and MFU (ResNet50H100Sustained)
    s = Scenarios.ImageClassifierEightGpuScaling
    t1, nd, tn = s.base_time.to("hour").magnitude, s.scaled_devices, s.scaled_time.to("hour").magnitude
    n.update(sct1=g(t1), scn=c(nd), sctn=g(tn), sceff=pct(t1 / (nd * tn)))
    r = Scenarios.ResNet50H100TrainingCeiling
    rate = r.sustained_rate.to("1/s").magnitude
    fpi = r.training_flops_per_sample.to("GFLOP").magnitude
    peak = r.hardware.compute.peak_flops.to("TFLOP/s").magnitude
    ach = rate * fpi / 1000
    n.update(mfurate=c(rate), mfugflop=f"{fpi:.1f}", mfuach=f"{ach:.0f}", mfupeak=c(peak), mfu=pct(ach / peak))
    assert abs(ach / peak - r.sustained_fraction) < 1e-9

    # ---- Mistakes
    t = Scenarios.ModelOnlyMeanVersusProductionP99
    mean, lo, hi = (x.to("ms").magnitude for x in (t.mean, t.tail_low, t.tail_high))
    n.update(tailmean=g(mean), taillow=g(lo), tailhigh=g(hi), tailxlow=g(lo / mean), tailxhigh=f"{hi / mean:.1f}")
    q = Scenarios.MobileNetV2Int8CameraTrapGates
    n.update(qtopf=f"{q.float_top1:.1f}", qtopi=f"{q.int_top1:.1f}", qdrop=f"{q.top1_loss:.1f}",
             qgate=g(q.accuracy_loss_limit), qecef=f"{q.float_ece:.3f}", qecei=f"{q.int_ece:.3f}",
             qedgef=f"{q.float_edge:.1f}", qedgei=f"{q.int_edge:.1f}", qedgedrop=f"{q.edge_drop:.1f}")
    d = Scenarios.DatasheetVersusSustainedEdgeChip
    dt, dw = d.claimed_rate.to("TOPS").magnitude, d.claimed_power.to("W").magnitude
    st, sw = d.sustained_rate.to("TOPS").magnitude, d.sustained_power.to("W").magnitude
    n.update(dstops=g(dt), dswatt=g(dw), dseff=g(dt / dw), sutops=g(st), suwatt=g(sw), sueff=g(st / sw),
             effgap=f"{(dt / dw) / (st / sw):.1f}")

    # ---- Q1: ASSUMED camera-trap frame path; the frame budget is the chapter board's (30 FPS)
    board = Scenarios.MobileNetV2CameraTrapBoard
    q_pre, q_host, q_acc, q_link = 5, 20, 2.5, 1.5            # ms, ASSUMED
    host_e2e, acc_e2e = q_pre + q_host, q_pre + q_acc + q_link
    # Ceiling: the model's time goes to zero; preprocessing and the new link copy remain
    n.update(qpre=g(q_pre), qhost=g(q_host), qacc=g(q_acc), qlink=g(q_link), qiso=g(q_host / q_acc),
             qhoste2e=g(host_e2e), qacce2e=g(acc_e2e), qe2e=f"{host_e2e / acc_e2e:.1f}",
             qrest=g(q_pre + q_link), qcap=f"{host_e2e / (q_pre + q_link):.1f}", qbudget=c(board.frame_budget.to("ms").magnitude), qfps=c(board.frame_rate.to("1/s").magnitude))

    # ---- Q2: ASSUMED wake-word board: 0.5 W active, 20 mW idle, one 10 ms inference per 1 s window
    w_act, w_idle, w_ms, w_win = 500.0, 20.0, 10.0, 1000.0    # mW, mW, ms, ms
    def avg(duty_, idle_):
        return duty_ * w_act + (1 - duty_) * idle_
    wd = w_ms / w_win
    base = avg(wd, w_idle)
    n.update(wact=f"{w_act / 1000:.1f}", widle=g(w_idle), wms=g(w_ms), wduty=pct(wd), wavg=f"{base:.1f}",
             widleshare=pct((1 - wd) * w_idle / base), wfast=f"{avg(wd / 2, w_idle):.1f}",
             wlowidle=f"{avg(wd, w_idle / 2):.1f}", wfastduty=f"{wd / 2 * 100:.1f}",
             wlowidlemw=g(w_idle / 2))

    # ---- Q4: ASSUMED 1 accelerator 30 h, 4 accelerators 9 h
    f1, fn, fdev = 30, 9, 4
    n.update(ft1=c(f1), ftn=c(fn), fdev=c(fdev), fideal=f"{f1 / fdev:.1f}", feff=pct(f1 / (fdev * fn)))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
