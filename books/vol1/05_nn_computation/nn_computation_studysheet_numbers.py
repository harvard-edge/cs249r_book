"""Numbers for the Neural Computation study sheet (Volume I, Chapter 5).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(MNISTInference, MnistFlopsCalc, MnistTrainingMemoryCalc, EpochTimeEnergy, CrossEntropyExample),
and written as \\setnum{key}{value} to nn_computation_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.

Where a printed result is a sum or ratio of printed parts, it is computed from the rounded parts
the sheet prints, and checked against the unrounded value, so a student's arithmetic reproduces it.
"""
import math
from pathlib import Path

from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def kb1(x_bytes):
    """Bytes -> KB (1 KB = 1,000 bytes), one decimal, as the chapter's memory table prints."""
    return round(x_bytes / 1e3, 1)


def dense_counts(widths):
    """Weights, biases, MACs per example for a dense network (chapter eq-depth-width-cost)."""
    w = [a * b for a, b in zip(widths[:-1], widths[1:])]
    b = list(widths[1:])
    return w, b


def numbers() -> dict:
    n = {}
    net = Scenarios.MnistDenseNetwork          # 784 -> 128 -> 64 -> 10, batch 32, FP32, Adam
    r = net.result
    bpv = int(net.bytes_per_value.to("byte").magnitude)
    widths = list(net.layer_widths)
    batch = net.batch_size

    # ---- Front: the running network (chapter: MNISTInference, tbl-nn-computation-mnist-params)
    w1, w2, w3 = r.weights_per_layer
    n.update(inputs=c(widths[0]), nh1=c(widths[1]), nh2=c(widths[2]), nout=c(widths[3]),
             w1=c(w1), w2=c(w2), w3=c(w3), weights=c(r.weights), biases=c(r.biases),
             params=c(r.parameters), macs=c(r.macs_per_example), flops=c(r.flops_per_example),
             l1share=f"{r.first_layer_share * 100:.0f}", bpv=c(bpv), batch=c(batch))
    # raw values for drawing the to-scale bar
    n.update(f1=f"{w1 / r.weights:.4f}", f2=f"{w2 / r.weights:.4f}", f3=f"{w3 / r.weights:.4f}")

    # Intensity under the chapter's traffic model (MnistFlopsCalc): batch 1 and batch 32
    i1 = net.variant(batch_size=1).result.intensity.to("flop/byte").magnitude
    i32 = r.intensity.to("flop/byte").magnitude
    n.update(i1=f"{i1:.1f}", i32=f"{i32:.0f}")
    # Mistake 3: the same traffic model in the sheet's terms (intensity card): work = B x O with
    # O = 2 x MACs per example; traffic = parameters once + B x (input + output + 2 x hidden) values.
    # Printed work and traffic are rounded to whole KFLOP / KB; the printed intensities are
    # recomputed from those rounded values and checked against mlsysim's (which adds bias/ReLU FLOPs).
    hidden = sum(widths[1:-1])
    for b, key in ((1, "1"), (batch, "32")):
        work_m = round(2 * r.macs_per_example * b / 1e6, 2 if b == 1 else 1)   # MFLOP, as printed
        traffic_kb = round(bpv * (r.parameters + b * (widths[0] + widths[-1] + 2 * hidden)) / 1e3)
        ref = net.variant(batch_size=b).result
        assert abs(traffic_kb - ref.traffic_bytes.to("byte").magnitude / 1e3) < 0.5
        printed = float(n["i" + key])
        ratio = work_m * 1e3 / traffic_kb
        assert abs(ratio - printed) < 0.06 if key == "1" else round(ratio) == int(printed)
        n.update({f"work{key}": f"{work_m:.2f}" if b == 1 else f"{work_m:.1f}", f"traffic{key}": c(traffic_kb)})
    # the printed batch-32 work reproduces from the printed batch-1 work x 32
    assert round(float(n["work1"]) * batch, 1) == float(n["work32"])
    n.update(trafficgrow=f"{float(n['traffic32'].replace(',', '')) / float(n['traffic1'].replace(',', '')):.1f}")

    # ---- Worked example: training vs inference memory at batch 32 (MnistTrainingMemoryCalc)
    wkb = kb1(r.parameter_bytes.to("byte").magnitude)
    # Gradients match the parameters; Adam keeps two values per parameter (memory card).
    # Computed from the printed parameter KB so a student's 2 x 437.5 reproduces the sheet,
    # and checked against mlsysim's unrounded optimizer bytes.
    gkb = wkb
    okb = round(2 * wkb, 1)
    assert abs(gkb - r.gradient_bytes.to("byte").magnitude / 1e3) < 0.1
    assert abs(okb - r.optimizer_bytes.to("byte").magnitude / 1e3) < 0.1
    act_vals = r.saved_activation_values                    # B x (784+128+64+10)
    akb = kb1(r.saved_activation_bytes.to("byte").magnitude)
    inf_vals = r.inference_activation_values                # B x largest adjacent pair
    ikb = kb1(r.inference_activation_bytes.to("byte").magnitude)
    statekb = round(wkb + gkb + okb, 1)
    train_kb = statekb + akb
    infer_kb = wkb + ikb
    true_train = r.training_memory.to("byte").magnitude / 1e3
    true_infer = r.inference_memory.to("byte").magnitude / 1e3
    assert abs(train_kb - true_train) < 0.5 and abs(infer_kb - true_infer) < 0.5
    ratio = round(train_kb) / round(infer_kb)
    assert abs(ratio - r.training_memory_ratio) < 0.05
    pair = widths[0] + widths[1]                            # largest adjacent pair of widths
    assert inf_vals == batch * pair
    n.update(wkb=f"{wkb:.1f}", gkb=f"{gkb:.1f}", okb=f"{okb:.1f}", statekb=f"{statekb:,.1f}",
             actvals=c(act_vals), actsum=c(sum(widths)), akb=f"{akb:.1f}", pairw=c(pair),
             infvals=c(inf_vals), ikb=f"{ikb:.1f}", trainkb=c(train_kb), inferkb=c(infer_kb),
             memratio=f"{ratio:.1f}", statebpp=c((r.parameter_bytes + r.gradient_bytes + r.optimizer_bytes)
                                                 .to("byte").magnitude / r.parameters))
    # raw values for the to-scale stacked bars
    n.update(wkbraw=f"{wkb}", gkbraw=f"{gkb}", okbraw=f"{okb}", akbraw=f"{akb}", ikbraw=f"{ikb}", trainkbraw=f"{train_kb:.1f}")

    # ---- Your turn: the same network at batch 1,024 (Scenarios.MnistDenseNetworkLargeBatch)
    big = Scenarios.MnistDenseNetworkLargeBatch
    rb = big.result
    bb = big.batch_size
    bakb = kb1(rb.saved_activation_bytes.to("byte").magnitude)
    bikb = kb1(rb.inference_activation_bytes.to("byte").magnitude)
    btrain = round(statekb + bakb, 1)
    binfer = round(wkb + bikb, 1)
    assert abs(btrain - rb.training_memory.to("byte").magnitude / 1e3) < 0.5
    assert abs(binfer - rb.inference_memory.to("byte").magnitude / 1e3) < 0.5
    n.update(bbatch=c(bb), bgrow=c(bb / batch), bactvals=c(rb.saved_activation_values), bakb=f"{bakb:,.1f}",
             binfvals=c(rb.inference_activation_values), bikb=f"{bikb:,.1f}", btrain=c(btrain),
             binfer=c(binfer), bratio=f"{round(btrain) / round(binfer):.1f}",
             bactshare=f"{bakb / btrain * 100:.0f}")

    # ---- Mistake 1: one epoch on a desktop CPU and a data-center accelerator (EpochTimeEnergy)
    ep = Scenarios.MnistEpochCpuVersusH100
    er = ep.result
    ms = lambda q: q.to("millisecond").magnitude
    n.update(epsteps=c(r.steps_per_epoch), epops=c(er.ops_per_step),
             epworkcpu=f"{ms(er.work_time['host']):.1f}", epworkacc=f"{ms(er.work_time['accelerator']):.2f}",
             eptotcpu=f"{ms(er.total_time['host']):.0f}", eptotacc=f"{ms(er.total_time['accelerator']):.0f}",
             epovh=f"{er.overhead_share_accelerator * 100:.1f}", epspeed=f"{er.work_speedup:.0f}")

    # ---- Mistake 2: a training step against the forward pass (MnistTrainingArithmetic)
    n.update(trainratio=f"{r.training_flops_ratio:.1f}")

    # ---- Mistake 3: the 1989 ZIP-code reader, as reported in the chapter (tbl-usps-numbers, LeCun 1989)
    zip_err, zip_rej = 0.01, 0.121           # reported: 1% error on accepted digits, 12.1% rejected
    n.update(ziperr=f"{zip_err * 100:.0f}", ziprej=f"{zip_rej * 100:.1f}",
             zipacc=f"{(1 - zip_rej) * 100:.1f}", zipwrong=f"{(1 - zip_rej) * zip_err * 100:.2f}")

    # ---- Q1: ASSUMED a wider network 784 -> 512 -> 256 -> 10, FP32
    qw = [widths[0], 512, 256, widths[-1]]
    ws, bs = dense_counts(qw)
    qp = sum(ws) + sum(bs)
    n.update(q1h1=c(qw[1]), q1h2=c(qw[2]), q1w1=c(ws[0]), q1w2=c(ws[1]), q1w3=c(ws[2]), q1w=c(sum(ws)),
             q1b=c(sum(bs)), q1p=c(qp), q1mb=f"{qp * bpv / 1e6:.2f}", q1share=f"{ws[0] / sum(ws) * 100:.0f}",
             q1x=f"{sum(ws) / r.macs_per_example:.1f}")

    # ---- Q2: cross-entropy (Scenarios.DigitCrossEntropyExamples); ASSUMED confident mistake p = 0.05
    ce = Scenarios.DigitCrossEntropyExamples
    p_ok = ce.probabilities["correct"]
    p_bad = 0.05                              # ASSUMED
    p_wrong = ce.misread["wrong"]
    n.update(q2pok=f"{p_ok:.1f}", q2pbad=f"{p_bad:.2f}", q2pwrong=f"{p_wrong:.1f}",
             q2lok=f"{-math.log(p_ok):.2f}", q2lbad=f"{-math.log(p_bad):.2f}",
             q2uni=f"{ce.uniform_loss:.1f}", q2classes=c(ce.classes),
             q2target="1", q2gtrue=f"{p_bad - 1:.2f}", q2gwrong=f"{p_wrong:.1f}",
             q2lx=f"{round(-math.log(p_bad), 2) / round(-math.log(p_ok), 2):.0f}")   # from the printed losses

    # ---- Q3: 20 sigmoid layers (Scenarios.TwentySigmoidLayers)
    sg = Scenarios.TwentySigmoidLayers
    mant, exp = f"{sg.surviving_fraction:.1e}".split("e")
    n.update(sigslope=f"{sg.peak_slope:.2f}", sigdepth=c(sg.depth), sigsurv=f"{mant}\\times10^{{{int(exp)}}}")

    # ---- Q4: ASSUMED two thresholds and costs, per 1,000 arriving items
    per = 1000
    cr, ce_ = 0.20, 5.00                      # ASSUMED: $ per review, $ per wrong accepted item
    thr = {"a": (0.04, 0.03), "b": (0.12, 0.01)}  # ASSUMED: (review fraction, error among accepted)
    for k, (rv, ea) in thr.items():
        wrong = (1 - rv) * ea * per
        reviews = rv * per
        n.update({f"q4{k}rev": f"{rv * 100:.0f}", f"q4{k}err": f"{ea * 100:.0f}",
                  f"q4{k}n": c(reviews), f"q4{k}w": f"{wrong:.1f}",
                  f"q4{k}cost": f"{reviews * cr + wrong * ce_:.0f}"})
    n.update(q4cr=f"{cr:.2f}", q4ce=f"{ce_:.0f}", q4per=c(per))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
