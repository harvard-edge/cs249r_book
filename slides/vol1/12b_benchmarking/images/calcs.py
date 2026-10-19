#!/usr/bin/env python3
"""Every derived number in Lecture 12b (Benchmarking, Part 2), one block per slide.

Inputs carry their source. "Chapter" is the rendered Benchmarking chapter
(Volume I). Every printed value is what the slide or its note shows, with the
label used there. make_figs.py imports the run statistics from here.

Run from this folder with the MLSysBook venv (needs scipy):
  python3 calcs.py
"""

import csv
import math
import statistics as st
from pathlib import Path

from scipy import stats

HERE = Path(__file__).parent


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Run data: mlperf_v40_resnet_runs.csv, one row per result_N.txt of
# mlcommons/training_results_v4.0 (closed, resnet), read from the MLLOG lines
# run_start, run_stop, last eval_accuracy, status, and seed (SOURCES.yml).
# =============================================================================
RUNS = list(csv.DictReader(open(HERE / "mlperf_v40_resnet_runs.csv")))
SCALE = 1.0042232277526395            # scaling.json in every ResNet folder
SYSTEMS = [  # key, row label, color
    ("eos_ngc24.04_mxnet", "Eos, 8x H100", "memory"),
    ("XE9680x8H100-SXM-80GB", "Dell, 8x H100", "compute"),
    ("h200_ngc24.04_mxnet", "NVIDIA, 8x H200", "data"),
    ("eos_n8_ngc24.04_mxnet", "Eos, 64x H100", "network"),
]
EOS, DELL, H200, N8 = (s[0] for s in SYSTEMS)
TARGET = 75.9                          # MLPerf ResNet-50 quality target, top-1 %
VAL_IMAGES = 50000                     # ImageNet validation set


def runs(system, converged=True):
    return [float(r["minutes"]) for r in RUNS if r["system"] == system
            and (r["status"] == "success" or not converged)]


def trimmed(system):
    """MLPerf run rule: a failed run counts as the slowest; drop both ends."""
    ok = sorted(runs(system))
    failed = len(runs(system, converged=False)) - len(ok)
    v = ok + [math.inf] * failed
    return st.mean(v[1:-1])


def welch(a, b):
    """Difference mean(a) - mean(b) with its Welch 95 percent interval."""
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df)
    d = st.mean(a) - st.mean(b)
    return dict(d=d, se=se, df=df, t=t, lo=d - t * se, hi=d + t * se)


SCORE = {s: trimmed(s) for s, _, _ in SYSTEMS}
PUB = {s: SCORE[s] * SCALE for s in SCORE}
CV = {s: st.stdev(runs(s)) / st.mean(runs(s)) * 100 for s, _, _ in SYSTEMS}
EOS_MEAN = st.mean(runs(EOS))
EOS_SD = st.stdev(runs(EOS))
DELL_MEAN = st.mean(runs(DELL))
DELL_SD = st.stdev(runs(DELL))
W_DELL = welch(runs(EOS), runs(DELL))
W_DELL_ALL = welch(runs(EOS), runs(DELL, converged=False))
W_H200 = welch(runs(EOS), runs(H200))
ACC = [float(r["top1"]) for r in RUNS]
P_ACC = st.mean(ACC) / 100
BINOM_HW = 1.96 * math.sqrt(P_ACC * (1 - P_ACC) / VAL_IMAGES) * 100
Z95, Z99 = stats.norm.ppf(0.975), stats.norm.ppf(0.995)
# n = z^2 p (1 - p) / m^2 (chapter Napkin Math 12.5; MLPerf Inference rules)
N_ACC = Z95 ** 2 * 0.95 * 0.05 / 0.01 ** 2
N_P90 = Z99 ** 2 * 0.90 * 0.10 / 0.005 ** 2
N_P99 = Z99 ** 2 * 0.99 * 0.01 / 0.0005 ** 2


def pct(x):
    """Minutes as a percent of Eos's mean converged run."""
    return 100 * x / EOS_MEAN


def runs_to_resolve(diff_pct, power=0.80):
    """Runs per system for a two-sided 95% test at the given power, Eos SD."""
    d = diff_pct / 100 * EOS_MEAN
    return 2 * (Z95 + stats.norm.ppf(power)) ** 2 * EOS_SD ** 2 / d ** 2


def half_width_pct(n):
    """95% half-width of a mean of n runs, percent of Eos's mean, Eos SD."""
    return 100 * stats.t.ppf(0.975, n - 1) * EOS_SD / math.sqrt(n) / EOS_MEAN


# =============================================================================
# Per slide, in deck order
# =============================================================================
def report():
    # --- Hook --------------------------------------------------------------
    head("Hook: published scores and the runs behind them")
    show("Dell published score (min)", f"{PUB[DELL]:.2f}")
    show("Eos published score (min)", f"{PUB[EOS]:.2f}")
    show("Eos score over Dell score, percent", f"{100 * (PUB[EOS] / PUB[DELL] - 1):.1f}")
    show("Dell converged runs range (min)", f"{min(runs(DELL)):.3f} to {max(runs(DELL)):.3f}, "
         f"spread {max(runs(DELL)) - min(runs(DELL)):.3f}")
    show("Eos runs range (min)", f"{min(runs(EOS)):.3f} to {max(runs(EOS)):.3f}, "
         f"spread {max(runs(EOS)) - min(runs(EOS)):.3f}")

    # --- Recall (Lecture 12's hook, chapter Sec. 12.7, illustrative) --------
    head("Recall thumbnail: TOPS per watt")
    show("datasheet 10 TOPS / 0.5 W", f"{10 / 0.5:.0f}")
    show("sustained 3 TOPS / 2 W", f"{3 / 2:.1f}")

    # --- The log: Eos result_0.txt (inputs typed from the MLLOG lines) ------
    # run_start time_ms 1714042844981 (line 194); run_stop time_ms
    # 1714043643353 (line 347), status success; last eval_accuracy 0.75936.
    head("Log excerpt: Eos result_0")
    t0, t1 = 1714042844981, 1714043643353
    show("run_stop - run_start (ms)", f"{t1 - t0:,}")
    show("run_stop - run_start (min)", f"{(t1 - t0) / 60000:.4f}")
    # Dell result_4.txt: run_start 1714566113431 (line 243); run_stop
    # 1714566950073 (line 411), status aborted, epoch 36; last eval 0.75894.
    show("Dell result_4 run_stop - run_start (min)", f"{(1714566950073 - 1714566113431) / 60000:.3f}")
    show("evaluations between the first and the last (epochs 3, 7, ..., 35)",
         f"{len(range(3, 36, 4)) - 2}")
    show("CSV minutes for this run", next(r["minutes"] for r in RUNS
                                         if r["system"] == EOS and r["run"] == "0"))
    show("last eval_accuracy as percent", f"{0.75936 * 100:.3f}")

    # --- Noise strip ---------------------------------------------------------
    head("Noise: 20 runs relative to each system's median")
    for s, lab, _ in SYSTEMS:
        med = st.median(runs(s))
        span = 100 * (max(runs(s)) - min(runs(s))) / med
        show(f"{lab}: CV, span of converged runs (% of median)", f"{CV[s]:.2f}, {span:.2f}")
        for r in RUNS:
            if r["system"] == s and r["status"] != "success":
                show(f"{lab}: failed run vs median, percent", f"+{100 * (float(r['minutes']) / med - 1):.1f}")
    show("Eos CV over Dell CV", f"{CV[EOS] / CV[DELL]:.1f}")

    # --- Seeds and accuracy -------------------------------------------------
    head("Seeds: top-1 at the last evaluation, 20 runs")
    show("top-1 range (%)", f"{min(ACC):.2f} to {max(ACC):.2f}")
    show("spread, all 20 (points)", f"{max(ACC) - min(ACC):.2f}")
    eos_acc = [float(r["top1"]) for r in RUNS if r["system"] == EOS]
    show("spread, Eos's five (points)", f"{max(eos_acc) - min(eos_acc):.2f}")
    show("mean top-1 (%)", f"{st.mean(ACC):.2f}")
    show("binomial 95% half-width at the mean, 50,000 images (points)", f"{BINOM_HW:.2f}")

    # --- Run rule (Dell) ----------------------------------------------------
    head("MLPerf run rule on Dell's five runs")
    show("mean of middle three (min)", f"{SCORE[DELL]:.3f}")
    show("x 1.0042 = published score (min)", f"{SCORE[DELL] * SCALE:.2f}")
    show("scaling factor, rounded", f"{SCALE:.4f}")

    # --- Sample sizes (Napkin Math 12.5; MLPerf Inference rules) ------------
    head("Samples a number needs: n = z^2 p (1-p) / m^2")
    n_acc, n_p90, n_p99 = N_ACC, N_P90, N_P99
    show("z at 95%, 99%", f"{Z95:.2f}, {Z99:.3f}")
    show("accuracy 95% +/- 1 point at 95%", f"{n_acc:,.0f}")
    show("p90 +/- 0.5 points at 99%", f"{n_p90:,.0f}")
    show("p99 +/- 0.05 points at 99%", f"{n_p99:,.0f}")
    show("p99 over accuracy", f"{n_p99 / n_acc:.0f}")

    # --- Welch interval on the hook ----------------------------------------
    head("Interval on the hook's difference (Welch, 95%)")
    show("Eos converged: n, mean, SD", f"{len(runs(EOS))}, {EOS_MEAN:.3f}, {EOS_SD:.3f}")
    show("Dell converged: n, mean, SD", f"{len(runs(DELL))}, {DELL_MEAN:.3f}, {DELL_SD:.4f}")
    for name, w in (("Eos - Dell", W_DELL), ("Eos - Dell, all five", W_DELL_ALL),
                    ("Eos - H200", W_H200)):
        show(name, f"d {w['d']:.3f}  SE {w['se']:.4f}  df {w['df']:.1f}  t {w['t']:.2f}  "
             f"CI {w['lo']:.3f} to {w['hi']:.3f} min  = {pct(w['d']):.1f}% "
             f"({pct(w['lo']):.1f} to {pct(w['hi']):.1f}%)")
    # "If Dell were as noisy as Eos": Dell's mean and n with Eos's SD.
    se_eq = EOS_SD * math.sqrt(1 / len(runs(EOS)) + 1 / len(runs(DELL)))
    df_eq = (EOS_SD ** 2 / 5 + EOS_SD ** 2 / 4) ** 2 / (
        (EOS_SD ** 2 / 5) ** 2 / 4 + (EOS_SD ** 2 / 4) ** 2 / 3)
    show("if Dell had Eos's SD: half-width (min)", f"{stats.t.ppf(0.975, df_eq) * se_eq:.2f}")

    # --- Runs against interval width --------------------------------------
    head("Half-width of a mean against runs (Eos SD)")
    for n in (5, 10, 20):
        show(f"n = {n}: +/- percent of {EOS_MEAN:.2f} min", f"{half_width_pct(n):.2f}")
    show("runs per system to resolve 0.5% at 80% power", f"{runs_to_resolve(0.5):.1f}")
    show("runs per system to resolve 1% at 80% power", f"{runs_to_resolve(1.0):.1f}")

    # --- Vote: Napkin Math 12.5 --------------------------------------------
    head("Vote: 95% against 94% on the same 1,000 images")
    n = 1000
    show("errors", f"{round(n * 0.05)} and {round(n * 0.06)}")
    se_ind = math.sqrt(0.95 * 0.05 / n + 0.94 * 0.06 / n)
    show("independent-sets interval (points)", f"{100 * (0.01 - 1.96 * se_ind):.1f} to "
         f"{100 * (0.01 + 1.96 * se_ind):.1f}")
    show("b - c", f"{round(n * 0.06) - round(n * 0.05)}")
    show("McNemar: b + c at most (10^2 / 3.84)", f"{10 ** 2 / 3.84:.1f}")

    # --- McNemar on two Eos runs --------------------------------------------
    head("McNemar on two Eos seeds")
    eos_runs = [r for r in RUNS if r["system"] == EOS]
    best = max(eos_runs, key=lambda r: float(r["top1"]))
    worst = min(eos_runs, key=lambda r: float(r["top1"]))
    gap = round((float(best["top1"]) - float(worst["top1"])) / 100 * VAL_IMAGES)
    show("runs", f"result_{best['run']} {best['top1']}%, result_{worst['run']} {worst['top1']}%")
    show("gap b - c (images)", f"{gap}")
    show("b + c at most (gap^2 / 3.84)", f"{gap ** 2 / 3.84:,.0f}")
    show("as percent of 50,000", f"{100 * gap ** 2 / 3.84 / VAL_IMAGES:.0f}")

    # --- System descriptions -------------------------------------------------
    # NVIDIA H200 datasheet: 4.8 TB/s, 1,979 TFLOP/s FP16 with sparsity;
    # H100 SXM: 3.35 TB/s, 989 TFLOP/s dense FP16 (chapter Sec. 12.3).
    # Containers: NGC 23.09 (September 2023) against 24.04 (April 2024).
    head("System descriptions: Eos, Dell, H200")
    show("H200 bandwidth over H100, percent more", f"{100 * (4.8 / 3.35 - 1):.0f}")
    show("H200 dense FP16 (sparse / 2)", f"{1979 / 2:.0f}")
    show("container age gap (months)", f"{(2024 - 2023) * 12 + (4 - 9)}")

    # --- Run order (Lee et al. 2019, Fig. 8 endpoints, approximate) ----------
    head("Run order on a warming device")
    blocked, inter = order_effects()
    show("blocked: B looks slower by (percent)", f"{blocked:.0f}")
    show("interleaved: B looks slower by (percent)", f"{inter:.0f}")

    # --- Real against useful ------------------------------------------------
    head("Real against useful: percent faster than Eos")
    for name, w in (("Dell, all five", W_DELL_ALL), ("Dell, converged", W_DELL), ("H200", W_H200)):
        show(name, f"{pct(w['lo']):.1f} to {pct(w['hi']):.1f}")

    # --- Beam (Example 12.1, illustrative) ----------------------------------
    head("Beam search: 50 ms to 200 ms, SLO 100 ms")
    show("slowdown", f"{200 / 50:.0f}")
    show("over the SLO", f"{200 / 100:.0f}")

    # --- Fresh test sets: Recht et al. 2019, Table (ImageNetV2 MatchedFrequency,
    # top-1), copied to recht2019_imagenetv2_top1.csv from the arXiv source.
    head("Recht et al.: original against new ImageNet top-1")
    rows = recht_rows()
    gaps = [o - n for _, o, n in rows]
    show("models plotted (Fisher-vector baselines excluded, as in the paper's Figure 1)", f"{len(rows)}")
    show("gap range (points)", f"{min(gaps):.1f} to {max(gaps):.1f}")
    rho = stats.spearmanr([o for _, o, _ in rows], [n for _, _, n in rows]).statistic
    show("Spearman rank correlation, original against new", f"{rho:.3f}")

    # --- Galaxy S4 (AnandTech table: 5-run averages, clocks) -----------------
    head("Galaxy S4 triangle throughput")
    show("GLBenchmark over GFXBench, percent", f"{100 * (43.2 / 37.9 - 1):.0f}")
    show("532 over 480 MHz, percent", f"{100 * (532 / 480 - 1):.1f}")

    # --- ECE (constructed bins) --------------------------------------------
    head("ECE by hand")
    bins = [(20, 0.65, 0.60), (30, 0.85, 0.80), (50, 0.97, 0.84)]
    show("gaps", ", ".join(f"{c - a:.2f}" for _, c, a in bins))
    show("ECE", f"{ece(bins):.4f}")

    # --- Guo et al. 2017 ------------------------------------------------------
    head("Guo et al.: LeNet 5 against ResNet-110, CIFAR-100")
    show("ECE ratio 16.53 / 4.85", f"{16.53 / 4.85:.1f}")

    # --- WILDS (Koh et al. Table 4) -------------------------------------------
    head("WILDS iWildCam: ID against OOD")
    show("average accuracy drop (points)", f"{75.7 - 71.6:.1f}")
    show("macro F1 drop (points)", f"{47.0 - 31.0:.1f}")
    show("Camelyon17 drop (points), chapter", f"{93.2 - 70.3:.1f}")

    # --- Diagnose (chapter Lighthouse 12.3, Table 18, assumed values) -------
    head("Acceptance gates for the INT8 camera-trap model")
    show("top-1 loss (points)", f"{71.8 - 70.9:.1f}")
    show("top-5 loss (points)", f"{91.0 - 90.4:.1f}")
    show("ECE ratio 0.089 / 0.031", f"{0.089 / 0.031:.1f}")
    show("edge-case drop (points)", f"{68.2 - 61.4:.1f}")
    n_edge = round(0.05 * 1000)
    show("edge cases in 1,000 images at 5%", f"{n_edge}")
    show("binomial 95% half-width at 68% on 50 (points)",
         f"{100 * 1.96 * math.sqrt(0.682 * 0.318 / n_edge):.0f}")

    # --- Verification gap -----------------------------------------------------
    # A 224 x 224 RGB image with 8 bits per channel; ImageNet validation set.
    head("Verification gap: inputs a test set samples")
    values = 224 * 224 * 3
    log10_space = values * math.log10(256)
    show("8-bit values per image (224 x 224 x 3)", f"{values:,}")
    show("possible images 256^values = 10^", f"{log10_space:,.0f}")
    show("log10 of 50,000", f"{math.log10(VAL_IMAGES):.1f}")
    show("fraction tested = 10^", f"{math.log10(VAL_IMAGES) - log10_space:,.0f}")

    # --- Close ----------------------------------------------------------------
    head("Close: the three verdicts")
    show("Dell interval (percent)", f"{pct(W_DELL['lo']):.1f} to {pct(W_DELL['hi']):.1f}")
    show("H200 (percent)", f"{pct(W_H200['d']):.1f}, {pct(W_H200['lo']):.1f} to {pct(W_H200['hi']):.1f}")


def recht_rows():
    rows = []
    for r in csv.DictReader(open(HERE / "recht2019_imagenetv2_top1.csv")):
        if not r["model"].startswith("fv_"):
            rows.append((r["model"], float(r["orig_top1"]), float(r["new_top1"])))
    return rows


def drift(x):
    """Smooth curve through Lee et al. 2019, Fig. 8 endpoints: 2.4 to 3.9 ms."""
    return 2.4 + 1.5 * (1 - math.exp(-x / 70))


ORDER_TIMES = [10 + 18 * i for i in range(10)]
ORDER_BLOCKED = ["A"] * 5 + ["B"] * 5
ORDER_INTER = ["A", "B"] * 5


def order_effects():
    out = []
    for labels in (ORDER_BLOCKED, ORDER_INTER):
        a = st.mean(drift(x) for x, lab in zip(ORDER_TIMES, labels) if lab == "A")
        b = st.mean(drift(x) for x, lab in zip(ORDER_TIMES, labels) if lab == "B")
        out.append(100 * (b / a - 1))
    return out


def ece(bins):
    total = sum(n for n, _, _ in bins)
    return sum(n / total * abs(c - a) for n, c, a in bins)


if __name__ == "__main__":
    report()
