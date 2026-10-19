#!/usr/bin/env python3
"""Every derived number in Lecture 1 (Introduction), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports the data loaders below so figures and
slides share one computation.

Run from this folder with the MLSysBook venv (it imports mlsysim):
  /Users/VJ/GitHub/MLSysBook/.venv/bin/python calcs.py
"""

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parents[3]                     # repository root (holds mlsysim/)


def show(label, value):
    print(f"  {label}: {value}")


# ---------------------------------------------------------------------------
# Data loaders shared with make_figs.py
# ---------------------------------------------------------------------------
def load_gft_cdc():
    """Weekly GFT estimate (ILI per 100,000 visits -> percent) and CDC wILI.
    Source: Delphi Epidata API (CMU), retrieved 2026-10-16, data/."""
    gft = {r["epiweek"]: r["num"] / 1000.0
           for r in json.load(open(DATA / "gft_nat.json"))["rows"]}
    cdc = {r["epiweek"]: r["wili"]
           for r in json.load(open(DATA / "cdc_ilinet_nat.json"))["rows"]}
    return gft, cdc


def gft_season_ratios(gft, cdc):
    """Ratio of summed weekly values, weeks 40 to 20, seasons 2003-04 to 2012-13."""
    out = []
    for y in range(2003, 2013):
        ws = [w for w in gft if w in cdc and ((w // 100 == y and w % 100 >= 40)
                                              or (w // 100 == y + 1 and w % 100 <= 20))]
        out.append((f"{y % 100:02d}-{(y + 1) % 100:02d}",
                    sum(gft[w] for w in ws) / sum(cdc[w] for w in ws)))
    return out


def load_recht():
    """ImageNet top-1, original validation set against ImageNetV2
    MatchedFrequency, one row per model. Source: Recht, Roelofs, Schmidt,
    Shankar, ICML 2019, appendix tables (arXiv 1902.10811 source,
    tables/imagenetv2-b-33_top1_model_results_table.{1,2}.tex), copied to
    data/recht2019/."""
    rows = []
    pat = re.compile(r"\\model\{([^}]*)\}\s*&\s*([\d.]+)\s*\{.*?\}\s*&\s*([\d.]+)\s*\{.*?\}\s*&\s*([\d.]+)")
    for part in (1, 2):
        for line in open(DATA / "recht2019" / f"imagenetv2-b-33_top1_model_results_table.{part}.tex"):
            m = pat.search(line)
            if m:
                rows.append((m.group(1).replace("\\_", "_"), float(m.group(2)),
                             float(m.group(3)), float(m.group(4))))
    return rows


def load_mlperf():
    """MLPerf Inference v4.0, closed division, NVIDIA, Llama 2 70B (99.9),
    Offline scenario, 8 accelerators, Tokens per second from each
    mlperf_log_summary.txt (github.com/mlcommons/inference_results_v4.0,
    closed/NVIDIA/results/<system>/llama2-70b-99.9/Offline/performance/run_1/),
    copied to data/mlperf_v4.0/."""
    out = {}
    for key, sysname in (("H100", "DGX-H100_H100-SXM-80GBx8_TRT"),
                         ("H200", "H200-SXM-141GBx8_TRT"),
                         ("H200 1000 W", "H200-SXM-141GB-CTSx8_TRT")):
        txt = (DATA / "mlperf_v4.0" / f"{sysname}_llama2-70b-99.9_Offline_mlperf_log_summary.txt").read_text()
        out[key] = float(re.search(r"Tokens per second:\s*([\d.]+)", txt).group(1))
    return out


def mlsysim_terms():
    """Iron-law terms from MLSysIM SingleNodeModel on the H100, FP16, efficiency 0.5."""
    sys.path.insert(0, str(ROOT / "mlsysim"))
    from mlsysim import Hardware, Models
    from mlsysim.solvers import SingleNodeModel
    h = Hardware.Cloud.H100
    sn = SingleNodeModel()
    rows = {}
    for key, m, b in (("ResNet-50, batch 64", Models.Vision.ResNet50, 64),
                      ("GPT-2 XL decode, batch 1", Models.Language.GPT2, 1)):
        p = sn.solve(m, h, batch_size=b, precision="fp16", efficiency=0.5)
        rows[key] = dict(
            mem=float(p.latency_memory.to("millisecond").magnitude),
            comp=float(p.latency_compute.to("millisecond").magnitude),
            over=float(p.latency_overhead.to("millisecond").magnitude),
            overlapped=float(p.latency.to("millisecond").magnitude),
            ai=float(p.arithmetic_intensity.magnitude),
            bw=float(h.memory.bandwidth.to("byte/second").magnitude))
    return rows


def hardware_bw():
    """H100 and H200 peak memory bandwidth and FP16 dense peak, MLSysIM registry."""
    sys.path.insert(0, str(ROOT / "mlsysim"))
    from mlsysim import Hardware
    return {n: (float(getattr(Hardware.Cloud, n).memory.bandwidth.to("TB/s").magnitude),
                float(getattr(Hardware.Cloud, n).compute.peak_flops.to("TFLOPs/s").magnitude))
            for n in ("H100", "H200")}


# ---------------------------------------------------------------------------
def main():
    print("Lecture 1 calcs")

    print("\n[Recall] GPT-3 training time (Lecture 0)")
    O_gpt3 = 3.14e23            # FLOPs, chapter (Brown et al. 2020)
    n_gpu, r_peak, eta = 1024, 312e12, 0.45   # A100 FP16 peak 312 TFLOP/s; utilization 0.45 (Lecture 0)
    days = O_gpt3 / (n_gpu * r_peak * eta) / 86400
    show("GPT-3 days at eta 0.45", f"{days:.1f} (slide: about 25)")
    show("GPT-3 days at eta 0.60", f"{O_gpt3 / (n_gpu * r_peak * 0.60) / 86400:.1f} (note: 19)")
    show("6 x 175e9 x 300e9", f"{6 * 175e9 * 300e9:.3g} (note: about 3.14e23)")

    print("\n[No test set covers the inputs]")
    pixels = 224 * 224 * 3
    digits = math.floor(pixels * math.log10(256)) + 1
    show("bytes per 224x224 RGB image", f"{pixels:,}")
    show("digits of 256^150,528", f"{digits:,}")

    print("\n[AlexNet]")
    params, bytes_per = 60e6, 4            # chapter Sec. 1.3.3
    show("weights", f"{params * bytes_per / 1e6:.0f} MB")
    show("GTX 580 memory / weights", f"{3e9 / (params * bytes_per):.1f}x (3 GB card)")

    print("\n[GPU-days]")
    show("AlexNet GPU-days (2 GPUs x 6 days)", 2 * 6)
    show("GPT-3 GPU-days (405 V100 GPU-years)", f"{405 * 365:,} (about 148,000)")
    show("GPT-4-class: 25,000 GPUs x 100 days", f"{25_000 * 100:,} (about 2.5 million)")

    print("\n[Degradation, pedestrian example]")
    r0, r1 = 0.95, 0.85                   # chapter Sec. 1.6
    show("miss rate before, after", f"{1 - r0:.2f}, {1 - r1:.2f}; ratio {(1 - r1) / (1 - r0):.1f}x (note: triples)")

    print("\n[ImageNetV2, Recht et al. 2019]")
    rec = load_recht()
    gaps = [g for _, _, _, g in rec]
    xs = [o for _, o, _, _ in rec]
    ys = [n for _, _, n, _ in rec]
    n = len(rec)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    icpt = my - slope * mx
    show("models", n)
    show("gap range, all models (points)", f"{min(gaps):.1f} to {max(gaps):.1f}")
    nn = [g for name, _, _, g in rec if not name.startswith("fv_")]   # fv_ = Fisher-vector baselines
    show("neural networks / Fisher-vector baselines", f"{len(nn)} / {n - len(nn)}")
    show("gap range, neural networks (points)", f"{min(nn):.1f} to {max(nn):.1f}")
    show("mean gap (points)", f"{sum(gaps) / n:.1f}")
    show("least-squares fit", f"new = {slope:.2f} x orig {icpt:+.1f}")
    show("gap of the top model", f"{rec[0][3]:.1f} ({rec[0][0]})")
    top = [g for _, o, _, g in rec if o >= 75]
    low = [g for _, o, _, g in rec if o < 65]
    show("mean gap, orig >= 75 / orig < 65", f"{sum(top) / len(top):.1f} / {sum(low) / len(low):.1f}")

    print("\n[Drift: Flu Trends season ratios]")
    gft, cdc = load_gft_cdc()
    seas = gft_season_ratios(gft, cdc)
    for s, r in seas:
        show(f"season {s}", f"{r:.2f}")
    early = [r for s, r in seas[:-2]]
    show("largest deviation from 1 before 2011-12", f"{max(abs(r - 1) for r in early) * 100:.0f}%")
    ws = [w for w in gft if w in cdc and 201240 <= w <= 201320]
    pk_g, pk_c = max(gft[w] for w in ws), max(cdc[w] for w in ws)
    show("2012-13 peaks GFT / CDC", f"{pk_g:.1f} / {pk_c:.1f}; ratio {pk_g / pk_c:.2f}")
    show("2012-13 largest single-week ratio", f"{max(gft[w] / cdc[w] for w in ws):.2f}")

    print("\n[Skew]")
    show("412 days in seconds", f"{412 * 86400:,}")

    print("\n[Napkin Math 1.1]")
    mem, comp, over = 2e9 / 100e9 * 1e3, 1e9 / 100e9 * 1e3, 5.0
    t = mem + comp + over
    t2 = mem / 2 + comp + over
    show("memory, compute, total (ms)", f"{mem:.0f}, {comp:.0f}, {t:.0f}")
    show("2x bandwidth total (ms)", f"{t2:.0f}; naive halving {t / 2:.1f}; cut {(t - t2) / t * 100:.0f}%")
    show("overlapped before, after (ms)", f"{max(mem, comp) + over:.0f}, {max(mem / 2, comp) + over:.0f}")
    show("chapter: 2 + 8 + 1 s, 2x compute", f"{2 + 8 / 2 + 1:.0f} s; naive {(2 + 8 + 1) / 2:.1f} s")

    print("\n[Iron-law bars, MLSysIM on H100]")
    T = mlsysim_terms()
    for k, v in T.items():
        tot = v["mem"] + v["comp"] + v["over"]
        tot2 = v["mem"] / 2 + v["comp"] + v["over"]
        show(k, f"mem {v['mem']:.3f}, comp {v['comp']:.3f}, over {v['over']:.3f}, "
                f"sum {tot:.2f} ms; 2x BW {tot2:.2f} ms ({(tot2 - tot) / tot * 100:+.0f}%); "
                f"overlapped {v['overlapped']:.2f} ms; bytes {v['mem'] * 1e-3 * v['bw'] / 1e9:.2f} GB; "
                f"intensity {v['ai']:.1f} FLOP/byte")

    print("\n[MLPerf Inference v4.0, Llama 2 70B Offline, 8 GPUs]")
    mp = load_mlperf()
    hw = hardware_bw()
    for k, v in mp.items():
        show(f"{k} tokens/s", f"{v:,.0f}; per GPU {v / 8:,.0f}; vs H100 {v / mp['H100']:.2f}x")
    show("bandwidth H200 / H100", f"{hw['H200'][0]:.2f} / {hw['H100'][0]:.2f} = {hw['H200'][0] / hw['H100'][0]:.2f}x")
    show("FP16 dense peak H200 / H100", f"{hw['H200'][1]:.0f} / {hw['H100'][1]:.0f} = {hw['H200'][1] / hw['H100'][1]:.2f}x")
    # Memory capacity: 80 GB (H100 SXM) and 141 GB (H200 SXM), from the
    # MLPerf system names (H100-SXM-80GB, H200-SXM-141GB) and NVIDIA datasheets.
    show("memory capacity H200 / H100", f"141 / 80 = {141 / 80:.2f}x")

    print("\n[GPT-2 decode profile, data/gpt2_profile.txt]")
    # GPT-2 small config (Hugging Face 'gpt2' config.json): n_layer 12,
    # n_embd 768, vocab_size 50257, FP32 = 4 bytes per weight.
    L, d, V, B = 12, 768, 50257, 4
    block_w = L * (d * 3 * d + d * d + d * 4 * d + 4 * d * d)   # c_attn, c_proj, c_fc, mlp c_proj
    head_w = V * d                                               # lm_head (tied to wte)
    prof = (DATA / "gpt2_profile.txt").read_text()

    def self_ms(op):
        m = re.search(rf"^\s*{re.escape(op)}\s+[\d.]+%\s+([\d.]+)(ms|us)", prof, re.M)
        return float(m.group(1)) / (1 if m.group(2) == "ms" else 1000)
    t_addmm, t_mm = self_ms("aten::addmm"), self_ms("aten::mm")
    t_total = float(re.search(r"Self CPU time total:\s*([\d.]+)ms", prof).group(1))
    t_step = self_ms("decode_step")
    show("block weight bytes (48 addmm)", f"{block_w * B / 1e6:.0f} MB")
    show("lm_head weight bytes (1 mm)", f"{head_w * B / 1e6:.0f} MB")
    show("addmm bytes / time", f"{block_w * B / (t_addmm * 1e-3) / 1e9:.0f} GB/s ({t_addmm} ms)")
    show("mm bytes / time", f"{head_w * B / (t_mm * 1e-3) / 1e9:.0f} GB/s ({t_mm} ms)")
    show("matrix multiplies share of step", f"{(t_addmm + t_mm) / t_total * 100:.0f}%")
    show("Python and launch overhead (decode_step self) share", f"{t_step / t_total * 100:.0f}%")
    show("FLOPs per token (2 per weight) / matmul time",
         f"{2 * (block_w + head_w) / ((t_addmm + t_mm) * 1e-3) / 1e9:.0f} GFLOP/s")

    print("\n[Amdahl]")
    pre, model, post, k = 60, 45, 25, 3       # chapter Sec. 1.11
    before, after = pre + model + post, pre + model / k + post
    show("before, after (ms)", f"{before}, {after:.0f}")
    show("end-to-end cut", f"{(before - after) / before * 100:.0f}% ; model cut {(1 - 1 / k) * 100:.0f}%")
    show("overall speedup", f"{before / after:.2f}x")
    show("model share; bound with infinitely fast model", f"{model / before * 100:.0f}%; {before / (pre + post):.2f}x")

    print("\n[Efficiency]")
    show("utilization 0.45 -> 0.60, days", f"{days:.1f} -> {days * 0.45 / 0.60:.1f}")


if __name__ == "__main__":
    main()
