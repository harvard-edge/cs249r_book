#!/usr/bin/env python3
"""Data plots for Lecture 4b (Data Engineering, Part 2: Delivery and Drift).

Every plotted value and every number in a label comes from calcs.py, which
reads MLSysIM (the delivery and drift Scenarios.*, Systems.Storage, Ops,
Literature); labels print the recorded strings with val(). Bars are drawn at
the scenario's exact value, so a label and its bar cannot disagree.

Typed here, because MLSysIM does not yet hold them (the chapter types them
too): the latency ladder's DRAM 100 ns, remote SSD 2 ms, and object store
20 ms against a 1 ms line (the chapter's Sec. 4.10 margin figure); the tier
bandwidth ranges for the in-memory cache (20 to 50 GB/s), the object store's
high end (500 MB/s per connection), and archive (10 to 50 MB/s) from table 8;
and the straggler batch times, seven workers at 12 ms and one at 180 ms
(Sec. 4.10.5). Axis limits and label positions are layout.

Diagrams (drift types) are SVG written by this script.

Run from this folder: python3 make_figs.py
"""

import sys
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
import numpy as np  # noqa: E402
from deckvals import val, num  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402
from calcs import m  # noqa: E402

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
    (f"{val('slow_gbps')} Gb/s link", m(K.MOVE.transfer_time("10 Gb/s"), "hour"), "danger", f"{val('slow_days')} days"),
    (f"{val('link_gbps')} Gb/s link", m(K.MOVE.transfer_time("100 Gb/s"), "hour"), "neutral", f"{val('xfer_h')} h"),
], lines=[(m(K.MOVE.job_time, "hour"), f"{val('run_h')} h training run", "data", 1.06)], log_x=False,
    xlim=(0, 330), size=(4.0, 1.35), xlabel=f"Time to move {val('pb')} PB (hours)")


# 2. Storage price per TB by medium, 1956 to 2023 (variant of the book figure)
def price_trend():
    rows = K.Literature.Series.StoragePricePerTB.entries
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
        ax.annotate(f"\\${val('hdd_first_b')}B per TB, {val('hdd_first_year')}",
                    xy=(first["year"], first["hdd"]), xytext=(1960, 3e5),
                    fontsize=11, fontweight="bold", color=C["memory"], alpha=a,
                    arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1, alpha=a))
        ax.annotate(f"\\${val('hdd_last')} per TB, {val('hdd_last_year')}",
                    xy=(last["year"], last["hdd"]), xytext=(1988, 1e-1),
                    fontsize=11, fontweight="bold", color=C["memory"], alpha=a,
                    arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1, alpha=a))
        ax.text(1957, 3e2, f"{val('hdd_last_year')}: DRAM {val('dram_over_ssd')}× SSD,\n"
                f"SSD {val('ssd_over_hdd')}× hard disk",
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
    ("DRAM", num("dram_lat_ns") * 1e-6, "memory", f"{val('dram_lat_ns')} ns"),  # ns to ms
    ("Local NVMe", m(K.STORE.LocalNvmeGen4.latency, "ms"), "memory", f"{val('nvme_lat_us')} µs", "in"),
    ("Remote SSD", 2, "danger", "2 ms"),
    ("Object store", 20, "danger", "20 ms"),
], lines=[(1, "1 ms", "danger", 1.07)], xlim=(3e-5, 2e3), size=(1.75, 1.45),
    xlabel="Access latency (log)", fs=12, tick_fs=12,
    xticks=[(1e-4, "100 ns"), (100, "100 ms")])

# 4. Sequential bandwidth by tier (table 8), ranges in MB/s
bars_lines(out / "tier_bandwidth.svg", [
    ("Memory cache", (20000, 50000), "memory", "20–50 GB/s"),
    ("Local NVMe", (m(K.STORE.LocalNvmeTrainingTypical.bandwidth, "MB/s"), m(K.STORE.LocalNvmeGen4.bandwidth, "MB/s")),
     "memory", f"{val('nvme_lo')}–{val('nvme_hi')} GB/s"),
    ("Object store", (m(K.STORE.ObjectStoreSingleStream.bandwidth, "MB/s"), 500), "danger",
     f"{m(K.STORE.ObjectStoreSingleStream.bandwidth, 'GB/s'):.1f}–0.5 GB/s"),
    ("Archive", (10, 50), "danger", "10–50 MB/s"),
], xlim=(5, 3e6), size=(1.55, 1.45), xlabel="Bandwidth (MB/s, log)",
    fs=12, tick_fs=12,
    xticks=[(10, "10"), (1e4, "10,000")])

# 5. The accelerator's appetite: ceiling against sustained rate
# The ceiling is the chapter's printed whole-image value (calcs.py).
bars_lines(out / "appetite.svg", [
    ("Arithmetic ceiling", K.ceiling, "compute", f"{val('ceiling_img')} img/s"),
    ("Sustained (chapter)", m(K.FEED.sustained_rate, "count/second"), "compute",
     f"{val('img_per_s')} img/s  ({val('sustained_pct')}%)"),
    ("Measured (NVIDIA)", K.nv1, "neutral",
     f"{val('nv_img')} img/s  ({val('nv_pct_ceiling')}%)"),
], log_x=False, xlim=(0, 19000), size=(5.6, 1.75), xlabel="Images per second")

# 6. Which bytes count: compressed against decoded demand, 250 MB/s supply
steps("decoded_bytes", [
    (f"JPEG, {val('jpeg_kb')} KB", m(K.FEED.demand(), "MB/s"), "compute",
     f"{val('need_mbs')} MB/s  ({val('loss_pct')}% lost)"),
    (f"Decoded FP32, {val('decoded_kb')} KB", m(K.FEED.demand(decoded=True), "MB/s"), "danger",
     f"{val('decoded_gbs')} GB/s  ({val('tensor_loss_pct')}% lost)"),
], lines=[(m(K.FEED.supply.bandwidth, "MB/s"), f"Supply: {val('supply_mbs')} MB/s", "neutral", 1.07)], log_x=False,
    xlim=(0, 3000), size=(6.4, 1.3), fs=15, tick_fs=14,
    xlabel=f"Storage bandwidth needed at {val('img_per_s')} img/s (MB/s)")

# 7. Storage supply against demand (MB/s): today, the ceiling, eight A100s
ceiling_rate = K.ceiling * K.FEED.sustained_rate.units
supply_rows = [
    (f"Needed at {val('img_per_s')} img/s", m(K.FEED.demand(), "MB/s"), "compute", f"{val('need_mbs')} MB/s"),
    ("Object store, 1 connection", m(K.STORE.ObjectStoreSingleStream.bandwidth, "MB/s"), "neutral",
     f"{val('obj_mbs')} MB/s"),
    ("Attached block volume", m(K.FEED.supply.bandwidth, "MB/s"), "neutral", f"{val('supply_mbs')} MB/s"),
    ("SATA SSD", m(K.STORE.SataSsdEffective.bandwidth, "MB/s"), "memory", f"{val('sata_mbs')} MB/s"),
    ("Local NVMe SSD", m(K.STORE.LocalNvmeTrainingTypical.bandwidth, "MB/s"), "memory",
     f"{val('nvme_lo')}–{val('nvme_hi')} GB/s", "in"),
    ("Needed at the ceiling", m(K.FEED.demand(rate=ceiling_rate), "MB/s"), "danger", f"{val('ceiling_gbs')} GB/s"),
    ("Needed, eight A100s", m(K.SERVER.demand(), "MB/s"), "danger", f"{val('eight_gbs')} GB/s", "in"),
]
for k in (1, 2, 3):
    bars_lines(out / f"storage_supply_step{k}.svg", supply_rows, step=4 + k,
               lines=[(m(K.SERVER.demand(decoded=True), "MB/s"), f"decoded:\n{val('eight_decoded_gbs')} GB/s",
                       "danger", 1.06, k == 3)],
               xlim=(30, 1e5), size=(5.8, 3.0),
               xlabel="Sequential read bandwidth (MB/s, log scale)",
               xticks=[(100, "100"), (1000, "1,000"), (10000, "10,000")])


# 8. Dataloader choke point (variant of the book figure)
def choke():
    # Solid line: training throughput, min(375 w, 3,200). Thin dashed line:
    # what the loader alone could supply past the crossing, up to the book
    # figure's illustrative 3,500 img/s cap.
    per_worker = m(K.FEED.worker_rate, "count/second")
    rate = m(K.FEED.sustained_rate, "count/second")
    crossing, whole = K.FEED.worker_crossing, K.FEED.workers_per_accelerator
    w = np.arange(1, 17)   # plotted range of worker counts
    loader = np.minimum(per_worker * w, m(K.FEED.host_decode_cap, "count/second"))
    train = np.minimum(per_worker * w, rate)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(np.concatenate(([crossing], w[w >= whole])),
            np.concatenate(([rate], loader[w >= whole])),
            "--", color=C["neutral"], lw=1.3)
    ax.text(12.2, 3680, "loader capacity", color=C["neutral"], fontsize=11,
            fontweight="bold", va="center")
    ax.plot(w, train, "o-", color=C["memory"], lw=2.4, ms=6, zorder=4)
    ax.axhline(rate, color=C["danger"], lw=2, ls="--", zorder=3)
    ax.text(16.8, 2750, f"Accelerator\n{val('img_per_s')} img/s", color=C["danger"], fontsize=11,
            fontweight="bold", va="center")
    ax.fill_between(w, 0, train, color=C["memory_l"], alpha=0.5)
    ax.axvline(crossing, color=C["text"], lw=1, ls=":")
    ax.text(8.75, 600, f"about {whole} workers", fontsize=11, fontweight="bold", color=C["text"])
    ax.text(1.0, 2600, "Host decode binds", fontsize=11, va="center",
            fontweight="bold", color=C["memory"])
    ax.text(4.5, 1150, f"{val('worker_rate')} img/s\nper worker", fontsize=11, va="center",
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
    ("Row format (CSV)", 100, "neutral", f"reads {val('cols_all')} of {val('cols_all')} columns"),
    ("Columnar (Parquet)", 100 * K.COLS.read_fraction, "data",
     f"reads {val('cols_used')} columns: {val('col_speedup')}× less"),
], log_x=False, xlim=(0, 175), size=(6.0, 1.1), fs=15, tick_fs=14,
    xlabel="Share of the table read from disk (%)")

# 9b. The profiler's step breakdown, PyTorch tutorial (published values in calcs.py)
def profiler_bars():
    P = K.PIPE
    rows = [("Default", {"step": K.step_a, "DataLoader": K.dl_a, "dl_pct": float(P.ProfilerDefaultDataLoaderPct)}),
            (f"{val('prof_workers')} workers",
             {"step": K.step_b, "DataLoader": K.dl_b, "dl_pct": float(P.ProfilerFourWorkersDataLoaderPct)})]
    # Drawn for its column slot (about 3.5 by 1.1 in), with type to match.
    fig, ax = plt.subplots(figsize=(4.6, 1.1))
    for i, (lab, v) in enumerate(rows):
        y = 1 - i
        dl, rest = v["DataLoader"] / K.us_per_ms, (v["step"] - v["DataLoader"]) / K.us_per_ms
        ax.barh(y, dl, color=C["danger"], height=0.6)
        ax.barh(y, rest, left=dl, color=C["neutral_l"], height=0.6)
        ax.text(-3, y, lab, ha="right", va="center", fontsize=15, color=C["text"])
        ax.text(dl + rest + 2, y, f"{v['step'] / K.us_per_ms:.1f} ms", va="center",
                fontsize=15, fontweight="bold", color=C["text"])
        if dl > 20:
            ax.text(dl / 2, y, f"DataLoader {v['dl_pct']:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")
            ax.text(dl + rest / 2, y, "rest of step", ha="center", va="center",
                    fontsize=14, color=C["text"])
        else:
            ax.text(dl + 2, y, f"DataLoader {v['dl_pct']:.1f}%",
                    ha="left", va="center", fontsize=14, color=C["text"])
    ax.set_xlim(0, 160)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150])
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Average step time (ms)", fontsize=14)
    sp.save(fig, out / "profiler_steps.svg")


profiler_bars()

# 10. CPU decompression per epoch for 100 GB
bars_lines(out / "codec_minutes.svg", [
    ("gzip", m(K.CODECS.decode_time("gzip"), "minute"), "danger", f"{val('gzip_min')} min"),
    ("Snappy", m(K.CODECS.decode_time("Snappy"), "minute"), "data", f"{val('snappy_min')} min"),
], log_x=False, xlim=(0, 24), size=(3.4, 1.4),
    xlabel=f"CPU decompression, {val('codec_gb')} GB (min)")

# 11. Stragglers: eight loader workers, one reading an oversized partition
# (typed, as the chapter types them in Sec. 4.10.5; see the docstring)
bars_lines(out / "stragglers.svg", [
    ("Workers 1 to 7", 12, "data", "12 ms each"),
    ("Worker 8", 180, "danger", "180 ms"),
], log_x=False, xlim=(0, 260), size=(3.6, 1.4), xlabel="Time to load one batch (ms)")

# 13. K-S test on clip loudness (illustrative curves, chapter numbers)
xs = np.linspace(-50, -10, 400)
SH = K.SHIFT
cdf = np.vectorize(lambda x, mu: 0.5 * (1 + erf((x - mu) / (SH.feature_sd * sqrt(2)))))
train = cdf(xs, SH.reference_mean)
recent = cdf(xs, SH.current_mean)   # the shift whose largest CDF gap matches the observed D
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
        ax.annotate(f"D = {val('ks_d')}, the largest gap", xy=(xs[i], (train[i] + recent[i]) / 2),
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
    crit = K.KS.coefficient * np.sqrt(2 / n)   # equal samples of n each
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(n, crit, color=C["memory"], lw=3)
    ax.axhline(K.KS.observed_statistic, color=C["danger"], lw=2, ls="--")
    ax.text(1.4e4, 0.088, f"observed D = {val('ks_d')}", color=C["danger"], fontsize=11,
            fontweight="bold")
    for sc, key, tx, ty in ((K.KS, "", 1400, 0.091), (K.KS_BIG, "_big", 9000, 0.04)):
        nn, c = sc.n_reference, sc.critical_value
        lab = f"{val('ks_crit' + key)} at n = {val('ks_n' + key)}"
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
    base, rec, terms = K.BINS.baseline_shares, K.BINS.current_shares, K.BINS.terms
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
        ax.text(2, 40.5, f"PSI = {val('psi')}: moderate band ({val('psi_warn')} to {val('psi_crit')})",
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
