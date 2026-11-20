"""Numbers for the ML Frameworks study sheet (Volume I, Chapter 7).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to frameworks_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED. Where a result
is printed beside its inputs, it is computed from the inputs as printed (rounded), so a student
can reproduce it.
"""
from pathlib import Path

from mlsysim import Scenarios


def c(x):
    return f"{x:,.0f}"


def r(x, nd):
    """Round to the precision the sheet prints, so later steps use the printed value."""
    return round(x, nd)


def numbers() -> dict:
    n = {}

    # ---- Front picture + mistake 1: GPT-2 XL batch-one FP16 decode step (chapter: GPT2DecodeDispatch)
    decode = Scenarios.Gpt2XlFp16DecodeOnH100
    disp = Scenarios.Gpt2XlEagerDecodeDispatch
    py_us = disp.dispatch.to("microsecond").magnitude
    ln_us = disp.launch.to("microsecond").magnitude
    kernels = int(disp.launches)
    py_ms = r(kernels * py_us / 1e3, 2)
    ln_ms = r(kernels * ln_us / 1e3, 2)
    host_ms = r(py_ms + ln_ms, 2)
    read_ms = r(disp.device_time.to("millisecond").magnitude, 2)
    n.update(pyus=c(py_us), lnus=c(ln_us), opus=c(py_us + ln_us), dkern=c(kernels),
             dlayers=c(decode.workload.layers), dperblock=c(disp.ops_per_block),
             dpy=f"{py_ms:.2f}", dln=f"{ln_ms:.2f}", dhost=f"{host_ms:.2f}", dread=f"{read_ms:.2f}",
             dgb=f"{decode.result.weight_bytes.to('GB').magnitude:.0f}",
             dbw=f"{decode.hardware.memory.bandwidth.to('TB/s').magnitude:.2f}",
             dratio=f"{host_ms / read_ms:.1f}")

    # ---- Worked example: the chapter's two-layer MLP training step (MnistTwoLayerTrainingStepOnH100)
    step = Scenarios.MnistTwoLayerTrainingStepOnH100
    mflop = r(step.total_flops.to("flop").magnitude / 1e6, 1)
    mb = r(step.total_bytes.to("MB").magnitude, 2)
    peak = step.peak_rate.to("TFLOPs/s").magnitude
    bw = step.hardware.memory.bandwidth.to("TB/s").magnitude
    entries, launches = int(step.host.operations), int(step.host.launches)
    t_cmp = r(mflop * 1e6 / (peak * 1e12) * 1e6, 2)   # us
    t_mem = r(mb * 1e6 / (bw * 1e12) * 1e6, 2)        # us
    t_dev = r(t_cmp + t_mem, 2)
    t_host = entries * py_us + launches * ln_us        # us
    ratio = t_host / t_dev
    n.update(win=c(step.input_width), whid=c(step.hidden_width), wout=c(step.output_width),
             wbatch=c(step.batch_size), wmflop=f"{mflop:.1f}", wmb=f"{mb:.2f}",
             wpeak=f"{peak:.0f}", wbw=f"{bw:.2f}", wentries=c(entries), wlaunch=c(launches),
             wfwd=c(entries - 1), wtc=f"{t_cmp:.2f}", wtm=f"{t_mem:.2f}", wtdev=f"{t_dev:.2f}",
             wthost=c(t_host), wratio=c(ratio), wshare=f"{100 * ratio / (1 + ratio):.0f}")

    # ---- Your turn: ASSUMED compiled step = 1 Python entry + 3 fused kernels; device time unchanged
    y_entries, y_launches = 1, 3
    y_host = y_entries * py_us + y_launches * ln_us
    y_ratio = y_host / t_dev
    n.update(yentries=c(y_entries), ylaunch=c(y_launches), ythost=c(y_host), yratio=c(y_ratio),
             yshare=f"{100 * y_ratio / (1 + y_ratio):.0f}",
             ybatch=c(step.batch_size * round(y_ratio)))

    # ---- Mistake 2: compilation overhead (chapter: CompilationBenchmark / CompilationOverheadCalc)
    be = Scenarios.ResNet50CompileBreakeven
    n.update(m2img=c(be.experiment_items), m2rec=c(be.recompiles),
             m2comp=c(be.compile_time.to("second").magnitude),
             m2eager=f"{be.experiment_eager.to('second').magnitude:.1f}",
             m2compiled=f"{be.experiment_compiled_total.to('second').magnitude:.1f}",
             m2slow=f"{be.experiment_slowdown:.0f}",
             m2be=c(round(be.breakeven_items, -3)))

    # ---- Mistake 3: the chapter's hypothetical embedded project (prose values, Choosing a Framework)
    budget_ms, budget_mb = 20, 256
    tfl_ms, tfl_mb, onnx_ms, onnx_mb = 15, 90, 18, 140
    n.update(m3ms=c(budget_ms), m3mb=c(budget_mb), m3tflms=c(tfl_ms), m3tflmb=c(tfl_mb),
             m3onnxms=c(onnx_ms), m3onnxmb=c(onnx_mb))

    # ---- Q1: ASSUMED bias add -> dropout -> ReLU on 16M 16-bit values, same device memory bandwidth
    q1n, q1k, q1b = 16e6, 3, 2
    unf = 2 * q1k * q1n * q1b
    fus = 2 * q1n * q1b
    n.update(q1n=f"{q1n / 1e6:.0f}M", q1k=c(q1k), q1bytes=c(q1b), q1rw=c(2),  # q1rw: one read + one write per operation
             q1unf=c(unf / 1e6), q1fus=c(fus / 1e6),
             q1tunf=c(unf / (bw * 1e12) * 1e6), q1tfus=c(fus / (bw * 1e12) * 1e6),
             q1saved=c((q1k - 1) * (py_us + ln_us)))

    # ---- Q2: ASSUMED serving model: 8 ms eager, 5 ms compiled, 60 s compile; 50 req/s; 200 runs per edit
    te, tc, tcomp, rps, nb = 8.0, 5.0, 60, 50, 200
    q2be = tcomp / ((te - tc) / 1e3)
    n.update(q2te=f"{te:.0f}", q2tc=f"{tc:.0f}", q2comp=c(tcomp), q2rps=c(rps), q2nb=c(nb),
             q2be=c(q2be), q2min=f"{q2be / rps / 60:.1f}")

    # ---- Q3: ASSUMED transformer training: B=8, N_L=24, S=1024, d=2048, 16 saved tensors, 2 bytes; 20 GB free
    B, L, S, d, k, b, free = 8, 24, 1024, 2048, 16, 2, 20
    m = B * L * S * d * k * b / 1e9
    n.update(q3b=c(B), q3l=c(L), q3s=c(S), q3d=c(d), q3k=c(k), q3bytes=c(b), q3free=c(free),
             q3gb=f"{m:.1f}", q3b2=c(2 * B), q3gb2=f"{2 * m:.1f}")

    # ---- Q4: chapter's 64-image FP32 batch over the host link (ImageNetBatch64HostFeed); ASSUMED 0.4 ms compute
    feed = Scenarios.ImageNetBatch64HostFeed
    link = decode.hardware.interconnect.bandwidth.to("GB/s").magnitude
    q4mb = r(feed.batch_bytes.to("MB").magnitude, 1)
    q4t = r(q4mb / 1e3 / link * 1e3, 1)   # ms, from the printed MB and GB/s
    q4c = 0.4
    n.update(q4batch=c(feed.batch_size), q4mb=f"{q4mb:.1f}", q4link=c(link), q4t=f"{q4t:.1f}",
             q4c=f"{q4c:.1f}", q4serial=f"{q4t + q4c:.1f}", q4ovl=f"{max(q4t, q4c):.1f}",
             q4u8=f"{q4mb / 4:.1f}", q4tu8=f"{q4mb / 4 / 1e3 / link * 1e3:.2f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
