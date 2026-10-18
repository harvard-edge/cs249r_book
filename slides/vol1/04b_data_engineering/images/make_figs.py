#!/usr/bin/env python3
"""Data plots for Lecture 4b (Data Engineering, Part 2: Delivery and Drift).

Every value is sourced from the rendered chapter (Data Engineering) unless
marked otherwise:
  - Transfer time: 1 PB over 100 Gb/s takes 80,000 s, about 22 h; over
    10 Gb/s about 9.3 days (Sec. 4.9.1, Napkin Math 4.7). The 20 h training
    run is the chapter's self-check scenario.
  - Storage price per TB, 1956 to 2023, constant 2020 US$: Our World in Data,
    via MLSysIM Literature.Series.StoragePricePerTB (the book figure's data).
  - Storage latency: DRAM 100 ns, local NVMe 100 us, remote SSD 2 ms, object
    store 20 ms against a 1 ms line (Sec. 4.10 margin figure).
  - Sequential bandwidth by tier (table 8): in-memory cache 20 to 50 GB/s,
    local NVMe 5 to 7 GB/s, object store 100 to 500 MB/s per connection,
    archive 10 to 50 MB/s after retrieval.
  - ResNet-50 on one A100: 312 TFLOP/s / 24.6 GFLOP = 12,682 img/s ceiling,
    3,200 img/s sustained, 150 KB JPEG, 602 KB decoded FP32, 480 MB/s and
    1.9 GB/s demand, 250 MB/s attached volume (Sec. 4.9.3).
  - Storage supply: object store about 100 MB/s per connection, SATA SSD
    500 MB/s, NVMe 5 to 7 GB/s; eight accelerators 3.8 GB/s compressed,
    15.4 GB/s decoded (Napkin Math 4.8, Sec. 4.10.1).
  - Dataloader: 375 img/s per worker, 3,200 img/s accelerator rate; the
    solid line is training throughput, min(375 w, 3,200); the thin dashed
    line is loader capacity past the crossing, up to the book figure's
    illustrative 3,500 img/s cap (Sec. 4.10.4).
  - Format: 20 of 100 equal-size columns (Napkin Math 4.9).
  - Codecs: 100 GB per epoch, gzip 120 MB/s gives 13.9 min, Snappy 500 MB/s
    gives 3.3 min (Sec. 4.10.5).
  - Stragglers: seven workers at 12 ms, one at 180 ms (Sec. 4.10.5).
  - Recht et al. 2019 (ICML), abstract: accuracy drops of 3 to 15 points on
    CIFAR-10 and 11 to 14 points on ImageNet.
  - K-S: n = 1,000 per sample, critical value 0.061, observed D = 0.08
    (Napkin Math 4.5); the critical value against n uses the same formula.
    The two CDF curves are illustrative.
  - PSI: baseline quintiles at 20 percent each against recent shares of
    30/25/20/15/10 percent are illustrative; the formula and the bands
    (0.1, 0.25) are the chapter's (Sec. 4.12.1).

Diagrams (drift types) are SVG written by this script.

Run from this folder: python3 make_figs.py
"""

import sys
from math import erf, log, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
import numpy as np  # noqa: E402

out = Path(__file__).parent
C = sp.C
plt = sp.plt


def bars_lines(path, rows, step=None, lines=(), log_x=True, xlim=None,
               xlabel=None, size=sp.SIZE, note=None, xticks=None, fs=13,
               tick_fs=None):
    """Horizontal bars with direct value labels and vertical reference lines.

    rows: (label, value, color, value_text[, "in"]); a value may be a
    (low, high) pair, drawn as a range bar.
    lines: (x, text, color, text_y[, visible]); hidden lines keep their
    label in the layout (alpha 0) so the figure size does not change.
    fs sets bar and value labels; tick_fs, when given, sets the tick labels,
    the axis label, and the reference-line labels (for plots drawn small)."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        label, v, ck, txt = r[:4]
        inside = len(r) > 4 and r[4] == "in"
        lo, hi = (v if isinstance(v, tuple) else (xlim[0] if log_x else 0, v))
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.62,
                alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=fs, color=C["text"],
                fontweight="bold")
        if inside:
            ax.text(hi / 1.25 if log_x else hi * 0.97, y, txt, va="center",
                    ha="right", fontsize=fs, color="white", fontweight="bold",
                    alpha=1.0 if shown else 0.0)
        else:
            ax.text(hi * 1.18 if log_x else hi + (xlim[1] if xlim else hi) * 0.015,
                    y, txt, va="center", fontsize=fs, color=C[ck],
                    fontweight="bold", alpha=1.0 if shown else 0.0)
    for ln in lines:
        x, text, ck, ty = ln[:4]
        vis = ln[4] if len(ln) > 4 else True
        ax.axvline(x, color=C[ck], lw=2, ls="--", alpha=1.0 if vis else 0.0)
        ax.text(x * (1.12 if log_x else 1.0) if log_x else x + (xlim[1] * 0.01),
                ty, text, color=C[ck], fontsize=tick_fs or 11, fontweight="bold",
                va="center", transform=ax.get_xaxis_transform(),
                alpha=1.0 if vis else 0.0)
    ax.set_yticks([])
    if log_x:
        ax.set_xscale("log")
        ax.minorticks_off()
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=tick_fs) if tick_fs else ax.set_xlabel(xlabel)
    if tick_fs:
        ax.tick_params(axis="x", labelsize=tick_fs)
    if note:
        ax.text(1.0, -0.42, note, transform=ax.transAxes, ha="right",
                fontsize=9, color=C["neutral"])
    sp.save(fig, path)


def steps(stem, rows, **kw):
    for k in range(1, len(rows) + 1):
        bars_lines(out / f"{stem}_step{k}.svg", rows, step=k, **kw)


# 1. Moving a petabyte: transfer time at two link speeds against a 20 h run
bars_lines(out / "transfer_time.svg", [
    ("10 Gb/s link", 9.3 * 24, "danger", "9.3 days"),
    ("100 Gb/s link", 22.2, "neutral", "22 h"),
], lines=[(20, "20 h training run", "data", 1.06)], log_x=False,
    xlim=(0, 330), size=(4.0, 1.35), xlabel="Time to move 1 PB (hours)")


# 2. Storage price per TB by medium, 1956 to 2023 (variant of the book figure)
def price_trend():
    root = str(ROOT / "mlsysim")
    if root not in sys.path:
        sys.path.insert(0, root)
    from mlsysim import Literature
    rows = Literature.Series.StoragePricePerTB.entries
    series = [("dram", "DRAM", C["danger"]), ("ssd", "Solid-state drive", C["data"]),
              ("hdd", "Hard disk", C["memory"])]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        label_y = {"dram": 1e4, "ssd": 1.5e2, "hdd": 1.5}
        for key, name, col in series:
            pts = [(r["year"], r[key]) for r in rows if r[key] is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", lw=2.4, color=col)
            ax.text(pts[-1][0] + 1.5, label_y[key], name, color=col, fontsize=11,
                    fontweight="bold", va="center", ha="left")
        first = next(r for r in rows if r["hdd"] is not None)
        last = rows[-1]
        a = 1.0 if k == 2 else 0.0
        ax.annotate(f"\\${first['hdd'] / 1e9:.0f}B per TB, {first['year']:.0f}",
                    xy=(first["year"], first["hdd"]), xytext=(1960, 3e5),
                    fontsize=11, fontweight="bold", color=C["memory"], alpha=a,
                    arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1, alpha=a))
        ax.annotate(f"\\${last['hdd']:.1f} per TB, {last['year']:.0f}",
                    xy=(last["year"], last["hdd"]), xytext=(1988, 1e-1),
                    fontsize=11, fontweight="bold", color=C["memory"], alpha=a,
                    arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1, alpha=a))
        ax.text(1957, 3e2, f"2023: DRAM {last['dram'] / last['ssd']:.0f}× SSD,\n"
                f"SSD {last['ssd'] / last['hdd']:.1f}× hard disk",
                fontsize=11, fontweight="bold", color=C["text"], alpha=a, va="center")
        ax.set_yscale("log")
        ax.set_xlim(1955, 2048)
        ax.set_ylim(1e-2, 1e17)
        ax.set_xticks([1960, 1980, 2000, 2020])
        ax.set_yticks([1, 1e5, 1e10, 1e15])
        ax.set_yticklabels(["\\$1", "\\$100K", "\\$10B", "\\$1,000T"])
        ax.minorticks_off()
        ax.grid(axis="y", color=C["grid"], linewidth=0.6)
        ax.set_ylabel("Price per TB\n(2020 US\\$, log)", fontsize=11)
        ax.text(1.0, -0.2, "Data: Our World in Data (McCallum price records)",
                transform=ax.transAxes, ha="right", fontsize=9, color=C["neutral"])
        sp.save(fig, out / f"price_trend_step{k}.svg")


price_trend()

# 3. Latency ladder against a 1 ms line (values in ms), final state only
bars_lines(out / "latency_ladder.svg", [
    ("DRAM", 1e-4, "memory", "100 ns"),
    ("Local NVMe", 0.1, "memory", "100 µs", "in"),
    ("Remote SSD", 2, "danger", "2 ms"),
    ("Object store", 20, "danger", "20 ms"),
], lines=[(1, "1 ms", "danger", 1.07)], xlim=(3e-5, 2e3), size=(1.75, 1.45),
    xlabel="Access latency (log)", fs=12, tick_fs=12,
    xticks=[(1e-4, "100 ns"), (100, "100 ms")])

# 4. Sequential bandwidth by tier (table 8), ranges in MB/s
bars_lines(out / "tier_bandwidth.svg", [
    ("Memory cache", (20000, 50000), "memory", "20–50 GB/s"),
    ("Local NVMe", (5000, 7000), "memory", "5–7 GB/s"),
    ("Object store", (100, 500), "danger", "0.1–0.5 GB/s"),
    ("Archive", (10, 50), "danger", "10–50 MB/s"),
], xlim=(5, 3e6), size=(1.55, 1.45), xlabel="Bandwidth (MB/s, log)",
    fs=12, tick_fs=12,
    xticks=[(10, "10"), (1e4, "10,000")])

# 5. The accelerator's appetite: ceiling against sustained rate
bars_lines(out / "appetite.svg", [
    ("Arithmetic ceiling", 12682, "compute", "12,682 img/s"),
    ("Sustained", 3200, "compute", "3,200 img/s  (25%)"),
], log_x=False, xlim=(0, 19000), size=(5.6, 1.5), xlabel="Images per second")

# 6. Which bytes count: compressed against decoded demand, 250 MB/s supply
steps("decoded_bytes", [
    ("JPEG, 150 KB", 480, "compute", "480 MB/s  (48% lost)"),
    ("Decoded FP32, 602 KB", 1900, "danger", "1.9 GB/s  (87% lost)"),
], lines=[(250, "Supply: 250 MB/s", "neutral", 1.07)], log_x=False,
    xlim=(0, 3000), size=(5.6, 1.75), xlabel="Storage bandwidth needed at 3,200 img/s (MB/s)")

# 7. Storage supply against demand (MB/s): today, the ceiling, eight A100s
supply_rows = [
    ("Needed at 3,200 img/s", 480, "compute", "480 MB/s"),
    ("Object store, 1 connection", 100, "neutral", "100 MB/s"),
    ("Attached block volume", 250, "neutral", "250 MB/s"),
    ("SATA SSD", 500, "memory", "500 MB/s"),
    ("Local NVMe SSD", 5000, "memory", "5–7 GB/s", "in"),
    ("Needed at the ceiling", 1900, "danger", "1.9 GB/s"),
    ("Needed, eight A100s", 3800, "danger", "3.8 GB/s", "in"),
]
for k in (1, 2, 3):
    bars_lines(out / f"storage_supply_step{k}.svg", supply_rows, step=4 + k,
               lines=[(15400, "decoded:\n15.4 GB/s", "danger", 1.06, k == 3)],
               xlim=(30, 1e5), size=(5.8, 3.0),
               xlabel="Sequential read bandwidth (MB/s, log scale)",
               xticks=[(100, "100"), (1000, "1,000"), (10000, "10,000")])


# 8. Dataloader choke point (variant of the book figure)
def choke():
    # Solid line: training throughput, min(375 w, 3,200). Thin dashed line:
    # what the loader alone could supply past the crossing, up to the book
    # figure's illustrative 3,500 img/s cap.
    w = np.arange(1, 17)
    loader = np.minimum(375 * w, 3500)
    train = np.minimum(375 * w, 3200)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(np.concatenate(([8.53], w[w >= 9])),
            np.concatenate(([3200], loader[w >= 9])),
            "--", color=C["neutral"], lw=1.3)
    ax.text(12.2, 3680, "loader capacity", color=C["neutral"], fontsize=11,
            fontweight="bold", va="center")
    ax.plot(w, train, "o-", color=C["memory"], lw=2.4, ms=6, zorder=4)
    ax.axhline(3200, color=C["danger"], lw=2, ls="--", zorder=3)
    ax.text(16.8, 2750, "Accelerator\n3,200 img/s", color=C["danger"], fontsize=11,
            fontweight="bold", va="center")
    ax.fill_between(w, 0, train, color=C["memory_l"], alpha=0.5)
    ax.axvline(8.53, color=C["text"], lw=1, ls=":")
    ax.text(8.75, 600, "about 9 workers", fontsize=11, fontweight="bold", color=C["text"])
    ax.text(1.0, 2600, "Host decode binds", fontsize=11, va="center",
            fontweight="bold", color=C["memory"])
    ax.text(4.5, 1150, "375 img/s\nper worker", fontsize=11, va="center",
            fontweight="bold", color=C["memory"])
    ax.text(11, 2300, "Accelerator binds", fontsize=11, fontweight="bold", color=C["danger"])
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 4000)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(["0", "1,000", "2,000", "3,000"])
    ax.set_xlabel("Loader worker processes")
    ax.set_ylabel("Images per second", fontsize=11)
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "dataloader.svg")


choke()

# 9. Bytes read for 20 of 100 columns
steps("format_bytes", [
    ("Row format (CSV)", 100, "neutral", "reads 100 of 100 columns"),
    ("Columnar (Parquet)", 20, "data", "reads 20 columns: 5× less"),
], log_x=False, xlim=(0, 175), size=(5.6, 1.5),
    xlabel="Share of the table read from disk (%)")

# 10. CPU decompression per epoch for 100 GB
bars_lines(out / "codec_minutes.svg", [
    ("gzip", 13.9, "danger", "13.9 min"),
    ("Snappy", 3.3, "data", "3.3 min"),
], log_x=False, xlim=(0, 24), size=(3.4, 1.4),
    xlabel="CPU decompression, 100 GB (min)")

# 11. Stragglers: eight loader workers, one reading an oversized partition
bars_lines(out / "stragglers.svg", [
    ("Workers 1 to 7", 12, "data", "12 ms each"),
    ("Worker 8", 180, "danger", "180 ms"),
], log_x=False, xlim=(0, 260), size=(3.6, 1.4), xlabel="Time to load one batch (ms)")

# 12. Recht et al. 2019: accuracy drop on re-collected test sets (ranges)
bars_lines(out / "recht_ranges.svg", [
    ("CIFAR-10", (3, 15), "danger", "3 to 15 points"),
    ("ImageNet", (11, 14), "danger", "11 to 14 points"),
], log_x=False, xlim=(0, 22), size=(5.6, 1.5),
    xlabel="Accuracy drop on the new test set, across models (points)",
    xticks=[(0, "0"), (5, "5"), (10, "10"), (15, "15")])

# 13. K-S test on clip loudness (illustrative curves, chapter numbers)
xs = np.linspace(-50, -10, 400)
cdf = np.vectorize(lambda x, m: 0.5 * (1 + erf((x - m) / (6 * sqrt(2)))))
train = cdf(xs, -30.0)
recent = cdf(xs, -31.2)   # a shift of 0.2 sigma gives a largest gap of about 0.08
i = int(np.argmax(recent - train))
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(xs, train, color=C["memory"], lw=3)
    ax.text(-25.5, 0.45, "Training clips", color=C["memory"], fontsize=12,
            fontweight="bold", ha="left")
    if k >= 2:
        ax.plot(xs, recent, color=C["danger"], lw=3)
        ax.text(-47.5, 0.42, "Recent clips", color=C["danger"], fontsize=12,
                fontweight="bold", ha="left")
    if k >= 3:
        ax.plot([xs[i], xs[i]], [train[i], recent[i]], color=C["text"], lw=3.5,
                solid_capstyle="butt", zorder=6)
        ax.annotate("D = 0.08, the largest gap", xy=(xs[i], (train[i] + recent[i]) / 2),
                    xytext=(-24, 0.12), fontsize=12, fontweight="bold",
                    color=C["text"], arrowprops=dict(arrowstyle="-", color=C["text"], lw=1))
    ax.set_xlim(-50, -10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Clip loudness (dBFS)")
    ax.set_ylabel("Share of clips\nat most this loud", fontsize=11)
    sp.save(fig, out / f"ks_cdf_step{k}.svg")


# 14. K-S critical value against sample size (same formula as Napkin Math 4.5)
def ks_vs_n():
    n = np.logspace(2, 5, 200)
    crit = 1.36 * np.sqrt(2 / n)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(n, crit, color=C["memory"], lw=3)
    ax.axhline(0.08, color=C["danger"], lw=2, ls="--")
    ax.text(1.4e4, 0.088, "observed D = 0.08", color=C["danger"], fontsize=11,
            fontweight="bold")
    for nn, lab, tx, ty in ((1000, "0.061 at n = 1,000", 1400, 0.091),
                            (100000, "0.006 at n = 100,000", 9000, 0.04)):
        c = 1.36 * sqrt(2 / nn)
        ax.plot([nn], [c], "o", color=C["text"], ms=7, zorder=5)
        ax.annotate(lab, xy=(nn, c), xytext=(tx, ty),
                    fontsize=11, fontweight="bold", color=C["text"],
                    arrowprops=dict(arrowstyle="-", color=C["text"], lw=1))
    ax.text(3e2, 0.17, "Flag when D is above the curve", color=C["memory"],
            fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(100, 1.2e5)
    ax.set_ylim(0, 0.2)
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_xlabel("Clips in each sample, n")
    ax.set_ylabel("Critical value", fontsize=11)
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "ks_vs_n.svg")


ks_vs_n()


# 15. PSI over five loudness bins (illustrative shares; chapter formula)
def psi_bins():
    base = [0.20] * 5
    rec = [0.30, 0.25, 0.20, 0.15, 0.10]
    terms = [(r - b) * log(r / b) for r, b in zip(rec, base)]
    names = ["Quietest", "2", "3", "4", "Loudest"]
    x = np.arange(5)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        ax.bar(x - 0.2, [100 * b for b in base], 0.38, color=C["memory"])
        ax.bar(x + 0.2, [100 * r for r in rec], 0.38, color=C["danger"])
        ax.text(4.55, 21.5, "Training", color=C["memory"], fontsize=11, fontweight="bold")
        ax.text(4.55, 15.5, "Recent", color=C["danger"], fontsize=11, fontweight="bold")
        for j, t in enumerate(terms):
            ax.text(j, 33.5, f"{t:.3f}", ha="center", fontsize=12, fontweight="bold",
                    color=C["text"], alpha=1.0 if k >= 2 else 0.0)
        ax.text(4.55, 33.5, "PSI terms", ha="left", fontsize=11, color=C["neutral"],
                alpha=1.0 if k >= 2 else 0.0)
        ax.text(2, 40.5, f"PSI = {sum(terms):.3f}: moderate band (0.1 to 0.25)",
                ha="center", fontsize=12.5, fontweight="bold", color=C["danger"],
                alpha=1.0 if k >= 3 else 0.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlim(-0.8, 5.6)
        ax.set_ylim(0, 44)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel("Share of clips (%)", fontsize=11)
        ax.set_xlabel("Loudness bin (training quintiles)")
        ax.grid(axis="y", color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f"psi_bins_step{k}.svg")


psi_bins()

# ---------------------------------------------------------------------------
# SVG diagram: drift types, three panels revealed one per step (900 x 400),
# the same drawing as Part 1
# ---------------------------------------------------------------------------
FONT = 'font-family="Helvetica, Arial, sans-serif"'
HEX = {"text": "#1A1A2E"}


def svg(path, body, w=900, h=500):
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + body + "\n</svg>\n")


def txt(x, y, s, size=20, bold=False, color="#1A1A2E", anchor="middle"):
    b = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}"{b} fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


def box(x, y, w, h, fill, stroke, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')


def drift(k):
    panels = [
        ("Covariate shift", "p(x) changes", "p(y | x) stays",
         "New microphone,\nsame words", "Compare input\nfeatures", C["memory"], C["memory_l"]),
        ("Label shift", "p(y) changes", "p(x | y) stays",
         "Flu season: more\npositive cases", "Compare the\nprediction mix", C["compute"], C["compute_l"]),
        ("Concept drift", "p(y | x) changes", "the mapping moves",
         "Fraudsters change\ntactics", "Needs fresh\nlabels", C["danger"], C["danger_l"]),
    ]
    s = ""
    for j, (name, what, keep, ex, det, col, light) in enumerate(panels):
        x = 10 + j * 297
        if j >= k:
            s += box(x, 10, 285, 380, "#FAFAFA", "#DDDDDD")
            continue
        s += box(x, 10, 285, 380, "white", col)
        s += f'<rect x="{x}" y="10" width="285" height="58" rx="8" fill="{col}"/>'
        s += txt(x + 142, 39, name, 27, True, "white")
        s += txt(x + 142, 100, what, 27, True, col)
        s += txt(x + 142, 135, keep, 22, False, "#555555")
        for m, ln in enumerate(ex.split("\n")):
            s += txt(x + 142, 185 + m * 30, ln, 23)
        s += box(x + 18, 270, 249, 100, light, col)
        dl = det.split("\n")
        for m, ln in enumerate(dl):
            s += txt(x + 142, 320 - (len(dl) - 1) * 15 + m * 30, ln, 23, True, col)
    svg(out / f"drift_step{k}.svg", s, h=400)


for k in (1, 2, 3):
    drift(k)

print("figures written")
