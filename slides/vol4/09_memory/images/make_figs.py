#!/usr/bin/env python3
"""Figures for Lecture 9 (Spatial Memory), Volume IV.

Plots use the shared slideplot helper; diagrams are written as SVG with text
kept as text. Every number drawn comes from calcs.py (derived values) or from
the sourced inputs it holds. Each figure is drawn at the size of its slot:
full-width plots 5.8 x 2.4 in, column figures about 5.4 x 3.6 in, full-width
SVG diagrams on a 900 x 500 canvas with labels of 20 units or more.

Run from this folder: python3 make_figs.py   (then make deck V=4 D=09_memory)
"""
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
from slideplot import C, plt  # noqa: E402

import calcs  # noqa: E402

COL = (5.4, 3.6)
FONT = "Helvetica, Arial, sans-serif"
# Volume IV palette roles (rules/volumes/vol4/palette.md)
NAVY, BLUE, GREEN, RED, PETROL, SLATE, AMBER = (
    "#1F407A", "#2563EB", "#059669", "#DC2626", "#0F766E", "#334155", "#D97706")


def out(name):
    return str(HERE / name)


def svg(name, body, w=900, h=500):
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w / 100:.2f}in" '
           f'height="{h / 100:.2f}in" viewBox="0 0 {w} {h}" '
           f'font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="white"/>\n'
           f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker>'
           f'<marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" '
           f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker>'
           f'<pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" '
           f'patternTransform="rotate(45)"><rect width="10" height="10" fill="#F1F5F9"/>'
           f'<line x1="0" y1="0" x2="0" y2="10" stroke="#94A3B8" stroke-width="3"/>'
           f'</pattern></defs>\n{body}\n</svg>\n')
    Path(out(name)).write_text(doc)


def t(x, y, s, size=22, fill=SLATE, anchor="middle", weight="normal", halo=False):
    h = ' stroke="white" stroke-width="7" paint-order="stroke"' if halo else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{h}>{s}</text>')


# ---------------------------------------------------------------------------
# Segment 1: estimation
# ---------------------------------------------------------------------------
def kalman():
    k = calcs.kalman()
    m = calcs.misdated(k)
    cols = [("Prior", 0, k["sig_pred"] * 0 + calcs.SIG0, "neutral"),
            ("Predict", 0, k["sig_pred"], "neutral"),
            ("+ lidar", 0, k["sig_l"], "memory"),
            ("+ stereo", 0, k["sig_s"], "data"),
            ("Misdated\nstereo", m["shift"], k["sig_s"], "danger")]
    meas = {2: calcs.SIG_LIDAR, 3: calcs.SIG_STEREO}
    for step in range(1, 5):
        n = {1: 2, 2: 3, 3: 4, 4: 5}[step]
        fig, ax = plt.subplots(figsize=COL)
        for i, (lab, mu, s, ck) in enumerate(cols):
            vis = i < n
            a = 1.0 if vis else 0.0
            ax.plot([i, i], [mu - s, mu + s], color=C[ck], lw=9, alpha=a,
                    solid_capstyle="butt")
            ax.plot([i - 0.22, i + 0.22], [mu, mu], color=C[ck], lw=2.5, alpha=a)
            ax.text(i, mu + s + 3, f"±{s:.1f}", ha="center", va="bottom",
                    fontsize=15, color=C[ck], fontweight="bold", alpha=a)
            if i in meas:
                ax.plot([i + 0.3, i + 0.3], [-meas[i], meas[i]], color=C["neutral_l"],
                        lw=5, alpha=a, solid_capstyle="butt", zorder=0)
        ax.axhline(0, color=C["text"], lw=1, ls=":")
        ax.text(4.45, -2, "true range", fontsize=13, color=C["text"], ha="right",
                va="top")
        if step == 4:
            ax.annotate(f"{m['shift']:.1f} mm off,\nsame ±{k['sig_s']:.1f}",
                        (4, m["shift"]), xytext=(2.85, 52), fontsize=14,
                        color=C["danger"], fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="->", color=C["danger"]))
        ax.set_xticks(range(5))
        ax.set_xticklabels([c[0] for c in cols], fontsize=14)
        for i, lab in enumerate(ax.get_xticklabels()):
            lab.set_alpha(1.0 if i < n else 0.0)
        ax.set_xlim(-0.5, 4.6)
        ax.set_ylim(-35, 70)
        ax.set_ylabel("Range error (mm)", fontsize=15)
        ax.tick_params(axis="y", labelsize=14)
        sp.save(fig, out(f"kalman_step{step}.svg"))


def gates():
    g = calcs.gates(calcs.kalman())
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    rows = [("Gate 3.84 (95%)", g["p95"]), ("Gate 6.63 (99%)", g["p99"])]
    for i, (lab, v) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C["memory"], height=0.6)
        ax.text(v + 0.015, y, f"{v:.2f} caught", va="center", fontsize=13,
                color=C["memory"], fontweight="bold")
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, color=C["text"], fontweight="bold")
    ax.axvline(1.0, color=C["neutral"], ls="--", lw=1.5)
    ax.text(0.99, 1.45, "every one caught", ha="right", fontsize=11, color=C["neutral"])
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of single misdated updates the gate rejects")
    sp.save(fig, out("gates.svg"))


def imu():
    v = calcs.imu()
    rows = [("Accel. bias", v["acc_bias"], "danger"),
            ("Gyro bias at 1.1 m", v["lateral"], "compute"),
            ("Accel. noise", v["acc_noise"], "neutral")]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (lab, val, ck) in enumerate(rows):
        y = 2 - i
        ax.barh(y, val, color=C[ck], height=0.6)
        if val > 5:
            ax.text(val / 1.12, y, f"{val:.1f} mm", va="center", ha="right", fontsize=13,
                    color="white", fontweight="bold")
        else:
            ax.text(val * 1.2, y, f"{val:.1f} mm", va="center", fontsize=13,
                    color=C[ck], fontweight="bold")
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, color=C["text"], fontweight="bold")
    ax.axvline(50, color=C["danger"], ls="--", lw=1.5)
    ax.text(55, 2.45, "50 mm bound", color=C["danger"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2000)
    ax.set_ylim(-0.5, 2.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Position error after 1 s without a fix (mm, log scale)")
    sp.save(fig, out("imu.svg"))


def sort_timeline():
    # Six camera frames; the person is seen in frames 1 to 4, hidden in 5 and 6.
    x0, dx, y, bh = 250, 105, [62, 132, 202], 52
    b = []
    for f in range(6):
        b.append(t(x0 + f * dx + dx / 2, 40, f"{f + 1}", 28, SLATE, weight="bold"))
    b.append(t(x0 - 14, 40, "Frame", 28, SLATE, "end", "bold"))
    for r, lab in enumerate(["In view", "SORT out", "Occupancy"]):
        b.append(t(x0 - 14, y[r] + 37, lab, 28, SLATE, "end", "bold"))
    for f in range(6):
        xx = x0 + f * dx + 6
        ww = dx - 12
        seen = f < 4
        b.append(f'<rect x="{xx}" y="{y[0]}" width="{ww}" height="{bh}" rx="5" '
                 f'fill="{PETROL if seen else "white"}" stroke="{PETROL if seen else "#CBD5E1"}" stroke-width="3"/>')
        out_ok = f in (2, 3)
        dash = "" if out_ok else ' stroke-dasharray="7,5"'
        b.append(f'<rect x="{xx}" y="{y[1]}" width="{ww}" height="{bh}" rx="5" '
                 f'fill="{BLUE if out_ok else "white"}" stroke="{BLUE if out_ok else "#CBD5E1"}" stroke-width="3"{dash}/>')
    b.append(f'<rect x="{x0 + 6}" y="{y[2]}" width="{4 * dx - 12}" height="{bh}" rx="5" fill="{RED}" fill-opacity="0.85" stroke="{RED}" stroke-width="3"/>')
    b.append(t(x0 + 2 * dx, y[2] + 37, "occupied", 28, "white", weight="bold"))
    b.append(f'<rect x="{x0 + 4 * dx + 6}" y="{y[2]}" width="{2 * dx - 12}" height="{bh}" rx="5" fill="url(#hatch)" stroke="#64748B" stroke-width="3"/>')
    b.append(t(x0 + 5 * dx, y[2] + 37, "unknown", 28, SLATE, weight="bold"))
    b.append(t(x0 + dx, 300, "min_hits = 3:", 26, BLUE, weight="bold"))
    b.append(t(x0 + dx, 330, "frames 1 and 2 held back", 26, BLUE))
    b.append(t(x0 + 4.6 * dx, 300, "max_age = 1:", 26, BLUE, weight="bold"))
    b.append(t(x0 + 4.6 * dx, 330, "hidden track not output", 26, BLUE))
    svg("sort_timeline.svg", "\n".join(b), w=920, h=345)


# ---------------------------------------------------------------------------
# Segment 2: records and frames
# ---------------------------------------------------------------------------
def epochs():
    b = []
    y = 150
    b.append(f'<line x1="40" y1="{y}" x2="610" y2="{y}" stroke="{SLATE}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(t(600, y + 45, "time", 21, SLATE, "end"))
    pts = [(90, "acquisition", "photons strike", PETROL),
           (330, "estimate", "instant it describes", NAVY),
           (520, "publication", "message sent", "#64748B")]
    for x, name, sub, col in pts:
        b.append(f'<line x1="{x}" y1="{y - 34}" x2="{x}" y2="{y + 34}" stroke="{col}" stroke-width="5"/>')
        b.append(t(x, y - 92, name, 23, col, weight="bold"))
        b.append(t(x, y - 60, sub, 20, col))
    b.append(f'<line x1="90" y1="{y + 70}" x2="560" y2="{y + 70}" stroke="{RED}" stroke-width="3" marker-end="url(#arrR)"/>')
    b.append(t(320, y + 108, "evidence age: measured from acquisition", 21, RED, weight="bold"))
    b.append(t(320, y + 160, "Prediction moves the estimate epoch;", 21, SLATE))
    b.append(t(320, y + 190, "only a new measurement moves acquisition.", 21, SLATE))
    svg("epochs.svg", "\n".join(b), w=640, h=370)


def lever():
    v = calcs.lever()
    one = calcs.latch()["one"]  # common-epoch error over one frame (mm)
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    rows = [("Stale edge\n+ current wrist", [calcs.E0_SLOT, v["e_tr"], v["e_rot"]]),
            ("Same-epoch\nchain", [calcs.E0_SLOT, v["e_tr"], one - calcs.E0_SLOT - v["e_tr"]])]
    cols = [C["neutral"], C["memory"], C["compute"]]
    for i, (lab, parts) in enumerate(rows):
        yy = 1 - i
        left = 0
        for j, p in enumerate(parts):
            ax.barh(yy, p, left=left, color=cols[j], height=0.55)
            left += p
        ax.text(left + 0.02, yy, f"{left:.2f} mm", va="center", fontsize=13,
                fontweight="bold", color=C["danger"] if left > calcs.R_TOL else C["data"])
        ax.text(-0.02, yy, lab, ha="right", va="center", fontsize=12,
                color=C["text"], fontweight="bold")
    key = [(0.0, f"initial {calcs.E0_SLOT:.2f}", cols[0]),
           (0.30, f"drift {v['e_tr']:.2f}", cols[1]),
           (0.58, f"lever arm {v['e_rot']:.2f}", cols[2])]
    for x, s, c in key:
        ax.text(x, 1.45, s, fontsize=11.5, color=c, fontweight="bold", ha="left")
    ax.axvline(calcs.R_TOL, color=C["danger"], ls="--", lw=1.5)
    ax.text(calcs.R_TOL + 0.02, -0.5, "1 mm slot\ntolerance", color=C["danger"],
            fontsize=11, ha="left", va="center", fontweight="bold")
    ax.set_xlim(0, 1.35)
    ax.set_ylim(-0.75, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Finger error after one 16.7 ms frame (mm)")
    sp.save(fig, out("lever.svg"))


def umbmark():
    u = calcs.umbmark()
    x = np.linspace(0, 60, 300)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        ax.plot(x, u["r_unc"] * x * 1000, color=C["danger"], lw=2.5)
        ax.text(12.5, 225, f"uncalibrated: {u['r_unc'] * 100:.1f}% of distance",
                color=C["danger"], fontsize=12, fontweight="bold", va="center")
        a = 1.0 if step == 2 else 0.0
        ax.plot(x, u["r_cal"] * x * 1000, color=C["data"], lw=2.5, alpha=a)
        ax.text(44, 105, f"calibrated:\n{u['r_cal'] * 100:.2f}%", color=C["data"],
                fontsize=12, fontweight="bold", alpha=a)
        ax.axhline(50, color=C["neutral"], ls="--", lw=1.5)
        ax.text(14, 56, "50 mm bound", ha="left", va="bottom", fontsize=11,
                color=C["neutral"])
        ax.axvline(calcs.AISLE_M, color=C["neutral"], ls=":", lw=1.5)
        ax.text(calcs.AISLE_M - 0.7, 170, "one aisle, 30 m", fontsize=11,
                color=C["neutral"], ha="right")
        ax.plot([u["d_unc"]], [50], "o", color=C["danger"], ms=8)
        ax.annotate(f"{u['d_unc']:.1f} m", (u["d_unc"], 50), xytext=(8, 92),
                    fontsize=12, color=C["danger"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["danger"]))
        ax.plot([u["d_cal"]], [50], "o", color=C["data"], ms=8, alpha=a)
        ax.annotate(f"{u['d_cal']:.1f} m", (u["d_cal"], 50), xytext=(46, 12),
                    fontsize=12, color=C["data"], fontweight="bold", alpha=a,
                    arrowprops=dict(arrowstyle="->", color=C["data"], alpha=a))
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 250)
        ax.set_xlabel("Distance driven (m)")
        ax.set_ylabel("Odometry error (mm)")
        sp.save(fig, out(f"umbmark_step{step}.svg"))


def kitti():
    rows = calcs.kitti()
    dl = calcs.DELTA_LOC
    groups = [("stereo", "Stereo", C["memory"]), ("laser", "Lidar", C["data"]),
              ("other", "Other", C["neutral"])]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    rng = np.random.default_rng(0)  # vertical jitter only
    for i, (key, lab, col) in enumerate(groups):
        v = np.array([r["translation_err_pct"] for r in rows
                      if (r["sensor_group"] if r["sensor_group"] in ("stereo", "laser") else "other") == key])
        ax.scatter(v, 2 - i + rng.uniform(-0.2, 0.2, len(v)), s=22, color=col,
                   alpha=0.85, zorder=3)
    thr = 100 * dl / calcs.AISLE_M
    ax.axvspan(thr, 40, color=C["danger_l"], alpha=0.6, lw=0, zorder=1)
    ax.axvline(thr, color=C["danger"], lw=1.5, ls="--")
    ax.text(38, -0.72, "red: bound spent within one 30 m aisle", fontsize=12,
            color=C["danger"], fontweight="bold", ha="right")
    best = min(r["translation_err_pct"] for r in rows)
    ax.annotate(f"lowest {best:.2f}%", (best, 2), xytext=(0.19, 2.45), fontsize=13,
                color=C["text"], arrowprops=dict(arrowstyle="->", color=C["text"]))
    ax.set_xscale("log")
    ax.set_xlim(0.1, 40)
    ax.set_ylim(-0.9, 2.75)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([g[1] for g in groups], fontsize=12, fontweight="bold")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_xlabel("Translation error (% of distance, log)")
    top = ax.secondary_xaxis("top", functions=(lambda e: 100 * dl / np.maximum(e, 1e-9),
                                               lambda d: 100 * dl / np.maximum(d, 1e-9)))
    top.set_xticks([30, 10, 3, 1, 0.3])
    top.set_xticklabels(["30", "10", "3", "1", "0.3"])
    top.set_xlabel("Meters until 50 mm is spent")
    sp.save(fig, out("kitti.svg"))


# ---------------------------------------------------------------------------
# Segment 3: maps
# ---------------------------------------------------------------------------
def reads():
    b = calcs.budget()
    r = calcs.reads()
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    names = [("occ", "occupancy", C["memory"]), ("oct", "octree", C["network"]),
             ("esdf", "ESDF", C["data"])]
    w = 0.27
    for j, (key, lab, col) in enumerate(names):
        xs = [i + (j - 1) * w for i in range(3)]
        vals = [r[cm][key] for cm in calcs.PITCHES_CM]
        ax.bar(xs, vals, width=w * 0.92, color=col)
        for xx, v in zip(xs, vals):
            ax.text(xx, v * 1.3, f"{v:,.0f}", ha="center", fontsize=9.5, color=col,
                    fontweight="bold")
        ax.text(2.97, 2e6 / 6**j, lab, ha="right", va="top", fontsize=11.5,
                color=col, fontweight="bold")
    ax.axhline(b["reads"], color=C["danger"], ls="--", lw=1.5)
    ax.text(2.97, b["reads"] * 1.2, f"budget\n{b['reads']:.0f} reads", color=C["danger"],
            fontsize=11, fontweight="bold", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(2.5, 3e6)
    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{cm} cm voxels" for cm in calcs.PITCHES_CM], fontsize=12)
    ax.set_ylabel("Reads per hull query")
    sp.save(fig, out("reads.svg"))


def nvblox():
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    items = list(calcs.ESDF_MS.items())
    for i, (k, v) in enumerate(items):
        y = len(items) - 1 - i
        col = C["data"] if "nvblox" in k else C["neutral"]
        ax.barh(y, v, color=col, height=0.42)
        ax.text(v * 1.15, y, f"{v:g} ms", va="center", fontsize=16, color=col,
                fontweight="bold")
        ax.text(1.2, y + 0.32, k, fontsize=15, color=C["text"], fontweight="bold",
                va="bottom")
    ax.axvline(1.0, color=C["danger"], ls="--", lw=2)
    ax.text(0.95, 2.95, "1 ms tick", fontsize=15, color=C["danger"], fontweight="bold",
            ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 400)
    ax.set_ylim(-0.45, 3.25)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("ESDF update, average (ms)", fontsize=15)
    sp.save(fig, out("nvblox_esdf.svg"))


def snapshot():
    r = calcs.reads()
    rows = [(f"{cm} cm pitch", r[cm]["esdf_kb"], "danger" if r[cm]["esdf_kb"] > 1050 else "data",
             f"{r[cm]['esdf_kb']:,.0f} kB") for cm in calcs.PITCHES_CM]
    sp.hbars(out("snapshot.svg"), rows, "kB", log=True, size=(5.8, 1.8),
             vlines=[(calcs.SRAM_CAP_MB * 1000, "1.05 MB SRAM", "danger")],
             xlabel="ESDF snapshot of the stopping corridor (kB, log)")


def ray_grid():
    b = []
    x0, y0, cs = 130, 60, 56
    ncol, nrow = 12, 4
    ray_row = 2
    for r in range(nrow):
        for c in range(ncol):
            x, y = x0 + c * cs, y0 + r * cs
            fill = "white"
            if r == ray_row and c < 7:
                fill = "#DCFCE7"
            if r == ray_row and c == 7:
                fill = "#334155"
            if r == ray_row and c > 7:
                fill = "url(#hatch)"
            if r == 0 and c in (9, 10):
                fill = "#FEE2E2"
            b.append(f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" fill="{fill}" stroke="#CBD5E1" stroke-width="2"/>')
    yr = y0 + ray_row * cs + cs / 2
    b.append(f'<circle cx="75" cy="{yr}" r="24" fill="{PETROL}"/>')
    b.append(t(75, yr + 62, "sensor", 27, PETROL, weight="bold"))
    b.append(f'<line x1="99" y1="{yr}" x2="{x0 + 7 * cs + 8}" y2="{yr}" stroke="{PETROL}" stroke-width="4" marker-end="url(#arr)"/>')
    b.append(f'<rect x="{x0 + 3 * cs + 12}" y="{y0 + ray_row * cs + 12}" width="{cs - 24}" height="{cs - 24}" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="6,4"/>')
    b.append(t(x0 + 3.5 * cs, y0 + ray_row * cs - 12, "phantom retired", 26, RED, weight="bold"))
    b.append(t(x0 + 3.5 * cs, y0 + 4 * cs + 34, "free, epoch t1", 27, GREEN, weight="bold"))
    b.append(t(x0 + 7.5 * cs, y0 + 4 * cs + 34, "return", 27, SLATE, weight="bold"))
    b.append(t(x0 + 10.2 * cs, y0 + 4 * cs + 34, "behind: still t0", 27, SLATE, weight="bold"))
    b.append(t(x0 + 9.8 * cs, y0 - 14, "occupied, t0", 26, "#991B1B", weight="bold"))
    lx, ly = 40, 385
    leg = [("#FEE2E2", "occupied"), ("#DCFCE7", "free"), ("url(#hatch)", "unknown"),
           ("#E2E8F0", "expired")]
    for i, (f, lab) in enumerate(leg):
        xx = lx + i * 215
        b.append(f'<rect x="{xx}" y="{ly}" width="40" height="40" fill="{f}" stroke="#64748B" stroke-width="2"/>')
        b.append(t(xx + 52, ly + 31, lab, 28, SLATE, "start"))
    b.append(t(450, 475, "Unknown and expired cells count as obstacles.", 28, SLATE, weight="bold"))
    svg("ray_grid.svg", "\n".join(b), w=900, h=495)


# ---------------------------------------------------------------------------
# Segment 4: expiry
# ---------------------------------------------------------------------------
def rack_top():
    # Top view of the aisle; 1 m = 250 units. The robot drives to the right.
    s = 250
    b = []
    cy = 190
    hull_x, hull_w, hull_h = 40, 130, 105
    front = hull_x + hull_w
    hull_y = cy + calcs.SIDE_CLEAR * s
    rack_end = front + calcs.D_CLEAR * s
    W, H = 720, 500
    r = (calcs.D_CLEAR - calcs.DELTA_LOC - calcs.DELTA_MARGIN) * s
    low = hull_y + hull_h + 45
    b.append(f'<rect x="{rack_end}" y="0" width="{W - rack_end}" height="{cy}" fill="url(#hatch)"/>')
    b.append(f'<circle cx="{rack_end}" cy="{cy}" r="{r}" fill="{RED}" fill-opacity="0.05" stroke="{RED}" stroke-width="3" stroke-dasharray="9,6"/>')
    b.append(f'<rect x="0" y="{cy - 75}" width="{rack_end}" height="75" fill="#E2E8F0" stroke="{SLATE}" stroke-width="2"/>')
    b.append(t(70, cy - 27, "rack", 30, SLATE, weight="bold"))
    b.append(f'<rect x="0" y="{low}" width="{W}" height="55" fill="#E2E8F0" stroke="{SLATE}" stroke-width="2"/>')
    b.append(t(70, low + 38, "rack", 30, SLATE, weight="bold"))
    b.append(f'<rect x="{rack_end + 22}" y="18" width="215" height="84" fill="white" opacity="0.88"/>')
    b.append(t(rack_end + 130, 52, "hidden", 30, SLATE, weight="bold"))
    b.append(t(rack_end + 130, 88, "cross aisle", 30, SLATE, weight="bold"))
    b.append(f'<rect x="{hull_x}" y="{hull_y}" width="{hull_w}" height="{hull_h}" fill="#F8FAFC" stroke="{SLATE}" stroke-width="3"/>')
    b.append(t(hull_x + hull_w / 2, hull_y + hull_h / 2 + 10, "robot", 30, SLATE, weight="bold"))
    yy = hull_y + hull_h / 2
    b.append(f'<line x1="{front}" y1="{yy + 18}" x2="{rack_end}" y2="{yy + 18}" stroke="{NAVY}" stroke-width="4" marker-start="url(#arr)" marker-end="url(#arr)"/>')
    b.append(t((front + rack_end) / 2, yy - 2, "ahead: 1.10 m", 30, NAVY, weight="bold", halo=True))
    b.append(f'<line x1="{hull_x + 25}" y1="{hull_y}" x2="{hull_x + 25}" y2="{cy}" stroke="{RED}" stroke-width="5"/>')
    b.append(t(hull_x + 36, cy + 31, "abreast: 150 mm", 28, RED, "start", "bold", halo=True))
    b.append(t(10, H - 14, "Dashed: a hidden walker's reach after 594 ms", 28, RED, "start", "bold"))
    svg("rack_top.svg", "\n".join(b), w=W, h=H)


def erosion():
    e = calcs.erosion()
    tt = np.linspace(0, 700, 300)
    d0 = (calcs.D_CLEAR - calcs.DELTA_LOC - calcs.DELTA_MARGIN) * 1000
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        y = np.clip(d0 - calcs.V_WALK * tt, 0, None)
        ax.plot(tt, y, color=C["memory"], lw=3)
        ax.plot([e["t_ah"]], [0], "o", color=C["danger"], ms=9, zorder=5)
        ax.annotate(f"ahead: {e['t_ah']:.2f} ms", (e["t_ah"], 0), xytext=(470, 330),
                    fontsize=12, color=C["memory"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["memory"]))
        a2 = 1.0 if step >= 2 else 0.0
        ax.axvspan(0, calcs.T_AGE_MS, color=C["neutral_l"], alpha=0.8 * a2, lw=0)
        ax.text(calcs.T_AGE_MS + 8, 300, f"frame age {calcs.T_AGE_MS} ms",
                fontsize=11, color=C["neutral"], alpha=a2, va="center")
        a3 = 1.0 if step >= 3 else 0.0
        ax.plot([0, 700], [0, 0], color=C["danger"], lw=3, ls="--", alpha=a3)
        ax.plot([0], [0], "o", color=C["danger"], ms=9, alpha=a3, zorder=6)
        ax.text(60, 70, "abreast of the rack end: 0 ms", fontsize=12,
                color=C["danger"], fontweight="bold", alpha=a3)
        ax.set_xlim(-10, 700)
        ax.set_ylim(-40, 1000)
        ax.set_xlabel("Time since the frame was captured (ms)")
        ax.set_ylabel("Certified depth (mm)")
        sp.save(fig, out(f"erosion_step{step}.svg"))


def latch():
    v = calcs.latch()
    tt = np.linspace(0, 0.1, 300)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=COL)
        e = calcs.E0_SLOT + calcs.V_DRIFT * tt + calcs.A_DIST / 2 * tt**2
        lin = calcs.E0_SLOT + calcs.V_DRIFT * tt
        ax.plot(tt * 1000, e, color=C["danger"], lw=3)
        ax.axhline(calcs.R_TOL, color=C["text"], ls="--", lw=1.5)
        ax.text(2, calcs.R_TOL + 0.03, "slot tolerance 1 mm", fontsize=14, color=C["text"])
        ax.axvspan(v["t"], 100, color=C["danger_l"], alpha=0.6, lw=0)
        ax.text(v["t"] - 1, 0.25, f"{v['t']:.0f} ms", fontsize=16, color=C["danger"],
                fontweight="bold", ha="right")
        ax.text(5, 0.55, "bounded error,\nwith 50 mm/s²", fontsize=14, color=C["danger"])
        a = 1.0 if step == 2 else 0.0
        ax.plot(tt * 1000, lin, color=C["neutral"], lw=2.5, ls="--", alpha=a)
        ax.plot([v["t_lin"]], [v["e_at"]], "o", color=C["danger"], ms=10, alpha=a)
        ax.annotate(f"{v['t_lin']:.0f} ms: error {v['e_at']:.2f} mm",
                    (v["t_lin"], v["e_at"]), xytext=(14, 1.32), fontsize=14,
                    color=C["text"], fontweight="bold", alpha=a,
                    arrowprops=dict(arrowstyle="->", color=C["text"], alpha=a))
        ax.text(80, 0.72, "constant\nvelocity", fontsize=14, color=C["neutral"], alpha=a)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("Time since the finger hid the slot (ms)", fontsize=15)
        ax.set_ylabel("Error bound (mm)", fontsize=15)
        ax.tick_params(labelsize=14)
        sp.save(fig, out(f"latch_step{step}.svg"))


def horizons():
    lat = calcs.latch()
    ero = calcs.erosion()
    cw = (calcs.TOL_COWORKER - calcs.E0_COWORKER) / calcs.V_WALK
    rows = [("Free space abreast", 0.0, "danger", "0 ms"),
            ("Latch-slot finger", lat["t"], "compute", f"{lat['t']:.0f} ms"),
            ("Coworker by the arm", cw, "network", f"{cw:.0f} ms"),
            ("Free space ahead", ero["t_ah"], "memory", f"{ero['t_ah']:.2f} ms")]
    sp.hbars(out("horizons.svg"), rows, "ms", log=False, size=(5.8, 2.2),
             xlabel="Invalidation horizon: time from capture to expiry (ms)")


def lifecycle():
    b = []
    boxes = {"VALID": (20, 150, GREEN, "#F0FDF4", "measured"),
             "DEGRADED": (330, 150, AMBER, "#FFFBEB", "predicted"),
             "EXPIRED": (640, 150, RED, "#FEF2F2", "no motion"),
             "UNOBSERVABLE": (330, 320, SLATE, "#F1F5F9", "rank-deficient")}
    w, h = 240, 95
    for name, (x, y, col, fill, sub) in boxes.items():
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{col}" stroke-width="3"/>')
        b.append(t(x + w / 2, y + 42, name, 28, col, weight="bold"))
        b.append(t(x + w / 2, y + 78, sub, 26, col))
    b.append(f'<line x1="262" y1="197" x2="326" y2="197" stroke="{SLATE}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(f'<line x1="572" y1="197" x2="636" y2="197" stroke="{SLATE}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(t(604, 285, "E = tolerance", 25, SLATE))
    b.append(t(294, 285, "occlusion", 25, SLATE))
    b.append(f'<path d="M 760 146 L 760 85 L 140 85 L 140 146" fill="none" stroke="{GREEN}" stroke-width="3" marker-end="url(#arr)"/>')
    b.append(f'<line x1="450" y1="85" x2="450" y2="146" stroke="{GREEN}" stroke-width="3"/>')
    b.append(t(450, 60, "back to VALID only with fresh evidence", 28, GREEN, weight="bold"))
    b.append(f'<path d="M 140 249 L 140 367 L 326 367" fill="none" stroke="{SLATE}" stroke-width="3" stroke-dasharray="7,5" marker-end="url(#arr)"/>')
    b.append(t(20, 445, "observability lost (any state)", 25, SLATE, "start"))
    b.append(t(590, 360, "leave by restoring", 25, SLATE, "start"))
    b.append(t(590, 390, "the constraint", 25, SLATE, "start"))
    svg("lifecycle.svg", "\n".join(b), w=900, h=460)


def force():
    tt = np.linspace(0, 15, 100)
    fig, ax = plt.subplots(figsize=COL)
    ax.fill_between(tt, 5 - 0.05, 5 + 0.05, color=C["memory_l"])
    ax.plot(tt, np.full_like(tt, 5.0), color=C["memory"], lw=3)
    ax.text(0.3, 5.25, "reported 5 N ± 0.05", fontsize=15, color=C["memory"],
            fontweight="bold")
    ax.plot(tt, 5 - calcs.BIAS_RATE * tt, color=C["danger"], lw=3)
    ax.plot([15], [2], "o", color=C["danger"], ms=10)
    ax.text(14.8, 1.45, "true 2 N", fontsize=15, color=C["danger"], fontweight="bold",
            ha="right")
    ax.text(0.3, 0.45, "innovation: 0 at every sample", fontsize=14, color=C["text"])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Minutes of thermal drift", fontsize=15)
    ax.set_ylabel("Grip force (N)", fontsize=15)
    ax.tick_params(labelsize=14)
    sp.save(fig, out("force.svg"))


if __name__ == "__main__":
    kalman()
    gates()
    imu()
    sort_timeline()
    epochs()
    lever()
    umbmark()
    kitti()
    reads()
    nvblox()
    snapshot()
    ray_grid()
    rack_top()
    erosion()
    latch()
    horizons()
    lifecycle()
    force()
