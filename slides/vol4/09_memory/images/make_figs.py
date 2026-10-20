#!/usr/bin/env python3
"""Figures for Lecture 9 (Spatial Memory), Volume IV.

Plots use the shared slideplot helper; diagrams are written as SVG with text
kept as text. Every number drawn comes from calcs.py (imported as K), which
reads MLSysIM, and every label prints val(key), the string the slide prints.
Each figure is drawn at the size of its slot: full-width plots 5.8 x 2.4 in,
column figures about 5.4 x 3.6 in, full-width SVG diagrams on a 900 x 500
canvas with labels of 20 units or more.

Run from this folder: python3 make_figs.py   (then make deck V=4 D=09_memory)
"""
import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import slideplot as sp  # noqa: E402
from slideplot import C, plt  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every derived value comes from calcs.py
from deckvals import val, num  # noqa: E402

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
    f = K.FUSED  # geometry at full precision; labels print the slide's strings
    shift, sig_s = K.m(f.misdated_shift, "mm"), K.m(f.fused_sigma, "mm")
    cols = [("Prior", 0, K.m(K.FUSION.prior_sigma, "mm"), "neutral", val("sig_prior_fig")),
            ("Predict", 0, K.m(f.predicted_sigma, "mm"), "neutral", val("sig_pred")),
            ("+ lidar", 0, K.m(f.first_sigma, "mm"), "memory", val("sig_l")),
            ("+ stereo", 0, sig_s, "data", val("sig_s")),
            ("Misdated\nstereo", shift, sig_s, "danger", val("sig_s"))]
    meas = {2: K.m(K.RM.lidar.range_precision_max, "mm"), 3: K.m(f.second_sensor_sigma, "mm")}
    for step in range(1, 5):
        n = step + 1  # columns shown at this build step
        fig, ax = plt.subplots(figsize=COL)
        for i, (lab, mu, s, ck, s_lab) in enumerate(cols):
            vis = i < n
            a = 1.0 if vis else 0.0
            ax.plot([i, i], [mu - s, mu + s], color=C[ck], lw=9, alpha=a,
                    solid_capstyle="butt")
            ax.plot([i - 0.22, i + 0.22], [mu, mu], color=C[ck], lw=2.5, alpha=a)
            ax.text(i, mu + s + 3, f"±{s_lab}", ha="center", va="bottom",
                    fontsize=15, color=C[ck], fontweight="bold", alpha=a)
            if i in meas:
                ax.plot([i + 0.3, i + 0.3], [-meas[i], meas[i]], color=C["neutral_l"],
                        lw=5, alpha=a, solid_capstyle="butt", zorder=0)
        ax.axhline(0, color=C["text"], lw=1, ls=":")
        ax.text(4.45, -2, "true range", fontsize=13, color=C["text"], ha="right",
                va="top")
        if step == 4:
            ax.annotate(f"{val('shift')} mm off,\nsame ±{val('sig_s')}",
                        (4, shift), xytext=(2.85, 52), fontsize=14,
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
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    rows = [(f"Gate {val('gate95')} ({val('gate95_pct')}%)", "p95", K.cons.detection[0]),
            (f"Gate {val('gate99')} ({val('gate99_pct')}%)", "p99", K.cons.detection[1])]
    for i, (lab, key, v) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C["memory"], height=0.6)
        ax.text(v + 0.015, y, f"{val(key)} caught", va="center", fontsize=13,
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
    rows = [("Accel. bias", "acc_bias_mm", "danger", K.imu.position_bias),
            (f"Gyro bias at {val('d_clear_fig')} m", "imu_lateral", "compute", K.imu.lateral_bias),
            ("Accel. noise", "acc_noise_mm", "neutral", K.imu.position_noise)]
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, (lab, key, ck, q) in enumerate(rows):
        y = 2 - i
        v = K.m(q, "mm")
        ax.barh(y, v, color=C[ck], height=0.6)
        if v > 5:
            ax.text(v / 1.12, y, f"{val(key)} mm", va="center", ha="right", fontsize=13,
                    color="white", fontweight="bold")
        else:
            ax.text(v * 1.2, y, f"{val(key)} mm", va="center", fontsize=13,
                    color=C[ck], fontweight="bold")
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, color=C["text"], fontweight="bold")
    bound = num("delta_loc")
    ax.axvline(bound, color=C["danger"], ls="--", lw=1.5)
    ax.text(bound * 1.1, 2.45, f"{val('delta_loc')} mm bound", color=C["danger"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2000)
    ax.set_ylim(-0.5, 2.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Position error after {val('t_coast')} s without a fix (mm, log scale)")
    sp.save(fig, out("imu.svg"))


def sort_timeline():
    # Six camera frames; the person is seen in frames 1 to 4, hidden in 5 and 6.
    hits = int(num("min_hits"))
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
        out_ok = seen and f >= hits - 1  # output once the hit streak reaches min_hits
        dash = "" if out_ok else ' stroke-dasharray="7,5"'
        b.append(f'<rect x="{xx}" y="{y[1]}" width="{ww}" height="{bh}" rx="5" '
                 f'fill="{BLUE if out_ok else "white"}" stroke="{BLUE if out_ok else "#CBD5E1"}" stroke-width="3"{dash}/>')
    b.append(f'<rect x="{x0 + 6}" y="{y[2]}" width="{4 * dx - 12}" height="{bh}" rx="5" fill="{RED}" fill-opacity="0.85" stroke="{RED}" stroke-width="3"/>')
    b.append(t(x0 + 2 * dx, y[2] + 37, "occupied", 28, "white", weight="bold"))
    b.append(f'<rect x="{x0 + 4 * dx + 6}" y="{y[2]}" width="{2 * dx - 12}" height="{bh}" rx="5" fill="url(#hatch)" stroke="#64748B" stroke-width="3"/>')
    b.append(t(x0 + 5 * dx, y[2] + 37, "unknown", 28, SLATE, weight="bold"))
    held = " and ".join(str(f + 1) for f in range(hits - 1))
    b.append(t(x0 + dx, 300, f"min_hits = {val('min_hits')}:", 26, BLUE, weight="bold"))
    b.append(t(x0 + dx, 330, f"frames {held} held back", 26, BLUE))
    b.append(t(x0 + 4.6 * dx, 300, f"max_age = {val('max_age')}:", 26, BLUE, weight="bold"))
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
    st = K.stale
    e0, drift, swing, same = (K.m(q, "mm") for q in (K.LATCH.initial_error, st.drift_term, st.lever_term, st.same_epoch_error))
    r_tol = num("r_tol")
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    rows = [("Stale edge\n+ current wrist", [e0, drift, swing], "e_total"),
            ("Same-epoch\nchain", [e0, drift, same - e0 - drift], "e_same")]
    cols = [C["neutral"], C["memory"], C["compute"]]
    for i, (lab, parts, key) in enumerate(rows):
        yy = 1 - i
        left = 0
        for j, p in enumerate(parts):
            ax.barh(yy, p, left=left, color=cols[j], height=0.55)
            left += p
        ax.text(left + 0.02, yy, f"{val(key)} mm", va="center", fontsize=13,
                fontweight="bold", color=C["danger"] if left > r_tol else C["data"])
        ax.text(-0.02, yy, lab, ha="right", va="center", fontsize=12,
                color=C["text"], fontweight="bold")
    key = [(0.0, f"initial {val('e0_slot_fig')}", cols[0]),
           (0.30, f"drift {val('e_drift')}", cols[1]),
           (0.58, f"lever arm {val('e_lever')}", cols[2])]
    for x, s, c in key:
        ax.text(x, 1.45, s, fontsize=11.5, color=c, fontweight="bold", ha="left")
    ax.axvline(r_tol, color=C["danger"], ls="--", lw=1.5)
    ax.text(r_tol + 0.02, -0.5, f"{val('r_tol')} mm slot\ntolerance", color=C["danger"],
            fontsize=11, ha="left", va="center", fontweight="bold")
    ax.set_xlim(0, 1.35)
    ax.set_ylim(-0.75, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Finger error after one {val('t_frame')} ms frame (mm)")
    sp.save(fig, out("lever.svg"))


def umbmark():
    u = K.umb
    bound, aisle = num("delta_loc"), num("aisle_m")
    x = np.linspace(0, 60, 300)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        ax.plot(x, u.rate_uncalibrated * x * 1000, color=C["danger"], lw=2.5)
        ax.text(12.5, 225, f"uncalibrated: {val('r_unc_pct')}% of distance",
                color=C["danger"], fontsize=12, fontweight="bold", va="center")
        a = 1.0 if step == 2 else 0.0
        ax.plot(x, u.rate_calibrated * x * 1000, color=C["data"], lw=2.5, alpha=a)
        ax.text(44, 105, f"calibrated:\n{val('r_cal_pct')}%", color=C["data"],
                fontsize=12, fontweight="bold", alpha=a)
        ax.axhline(bound, color=C["neutral"], ls="--", lw=1.5)
        ax.text(14, bound + 6, f"{val('delta_loc')} mm bound", ha="left", va="bottom", fontsize=11,
                color=C["neutral"])
        ax.axvline(aisle, color=C["neutral"], ls=":", lw=1.5)
        ax.text(aisle - 0.7, 170, f"one aisle, {val('aisle_m')} m", fontsize=11,
                color=C["neutral"], ha="right")
        d_unc, d_cal = K.m(u.distance_uncalibrated, "m"), K.m(u.distance_calibrated, "m")
        ax.plot([d_unc], [bound], "o", color=C["danger"], ms=8)
        ax.annotate(f"{val('d_unc')} m", (d_unc, bound), xytext=(8, 92),
                    fontsize=12, color=C["danger"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["danger"]))
        ax.plot([d_cal], [bound], "o", color=C["data"], ms=8, alpha=a)
        ax.annotate(f"{val('d_cal')} m", (d_cal, bound), xytext=(46, 12),
                    fontsize=12, color=C["data"], fontweight="bold", alpha=a,
                    arrowprops=dict(arrowstyle="->", color=C["data"], alpha=a))
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 250)
        ax.set_xlabel("Distance driven (m)")
        ax.set_ylabel("Odometry error (mm)")
        sp.save(fig, out(f"umbmark_step{step}.svg"))


def kitti():
    rows = K.KITTI.entries
    dl = K.m(K.AISLE.delta_loc, "m")
    groups = [("stereo", "Stereo", C["memory"]), ("laser", "Lidar", C["data"]),
              ("other", "Other", C["neutral"])]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    rng = np.random.default_rng(0)  # vertical jitter only
    for i, (key, lab, col) in enumerate(groups):
        v = np.array([r["translation_err_pct"] for r in rows
                      if (r["sensor_group"] if r["sensor_group"] in ("stereo", "laser") else "other") == key])
        ax.scatter(v, 2 - i + rng.uniform(-0.2, 0.2, len(v)), s=22, color=col,
                   alpha=0.85, zorder=3)
    thr = 100 * dl / num("aisle_m")
    ax.axvspan(thr, 40, color=C["danger_l"], alpha=0.6, lw=0, zorder=1)
    ax.axvline(thr, color=C["danger"], lw=1.5, ls="--")
    ax.text(38, -0.72, f"red: bound spent within one {val('aisle_m')} m aisle", fontsize=12,
            color=C["danger"], fontweight="bold", ha="right")
    best = K.kit.best_pct
    ax.annotate(f"lowest {val('kitti_best')}%", (best, 2), xytext=(0.19, 2.45), fontsize=13,
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
    top.set_xlabel(f"Meters until {val('delta_loc')} mm is spent")
    sp.save(fig, out("kitti.svg"))


# ---------------------------------------------------------------------------
# Segment 3: maps
# ---------------------------------------------------------------------------
def reads():
    pitches = list(K.q.rows)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    names = [("occ", "occupancy", C["memory"]), ("oct", "octree", C["network"]),
             ("esdf", "ESDF", C["data"])]
    w = 0.27
    for j, (key, lab, col) in enumerate(names):
        xs = [i + (j - 1) * w for i in range(3)]
        keys = [f"{key}{cm}" if key != "esdf" else "trilinear" for cm in pitches]
        field = {"occ": "occupancy_reads", "oct": "octree_reads", "esdf": "esdf_reads"}[key]
        vals = [getattr(K.q.rows[cm], field) for cm in pitches]
        ax.bar(xs, vals, width=w * 0.92, color=col)
        for xx, v, k in zip(xs, vals, keys):
            ax.text(xx, v * 1.3, val(k), ha="center", fontsize=9.5, color=col,
                    fontweight="bold")
        ax.text(2.97, 2e6 / 6**j, lab, ha="right", va="top", fontsize=11.5,
                color=col, fontweight="bold")
    budget = num("reads")
    ax.axhline(budget, color=C["danger"], ls="--", lw=1.5)
    ax.text(2.97, budget * 1.2, f"budget\n{val('reads')} reads", color=C["danger"],
            fontsize=11, fontweight="bold", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(2.5, 3e6)
    ax.set_xlim(-0.5, 3.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{val(f'pitch{cm}')} cm voxels" for cm in pitches], fontsize=12)
    ax.set_ylabel("Reads per hull query")
    sp.save(fig, out("reads.svg"))


def nvblox():
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    items = [("nvblox (GPU)", "esdf_nvblox"), ("Fiesta (CPU)", "esdf_fiesta"),
             ("voxblox (CPU)", "esdf_voxblox")]
    for i, (k, key) in enumerate(items):
        y = len(items) - 1 - i
        v = num(key)
        col = C["data"] if "nvblox" in k else C["neutral"]
        ax.barh(y, v, color=col, height=0.42)
        ax.text(v * 1.15, y, f"{val(key)} ms", va="center", fontsize=16, color=col,
                fontweight="bold")
        ax.text(1.2, y + 0.32, k, fontsize=15, color=C["text"], fontweight="bold",
                va="bottom")
    tick = num("t_tick_ms")
    ax.axvline(tick, color=C["danger"], ls="--", lw=2)
    ax.text(tick * 0.95, 2.95, f"{val('t_tick_ms')} ms tick", fontsize=15, color=C["danger"], fontweight="bold",
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
    sram_kb = K.m(K.RM.permission_mcu.memory.sram_capacity, "kB")
    rows = [(f"{val(f'pitch{cm}')} cm pitch", K.m(r.esdf_bytes, "kB"),
             "danger" if K.m(r.esdf_bytes, "kB") > sram_kb else "data",
             f"{val(f'esdf{cm}_kb')} kB") for cm, r in K.q.rows.items()]
    sp.hbars(out("snapshot.svg"), rows, "kB", log=True, size=(5.8, 1.8),
             vlines=[(num("sram_mb") * 1000, f"{val('sram_mb')} MB SRAM", "danger")],
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
    hull_y = cy + num("side_clearance_m") * s
    rack_end = front + num("d_clear") * s
    W, H = 720, 500
    r = num("depth_ahead_m") * s
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
    b.append(t((front + rack_end) / 2, yy - 2, f"ahead: {val('d_clear')} m", 30, NAVY, weight="bold", halo=True))
    b.append(f'<line x1="{hull_x + 25}" y1="{hull_y}" x2="{hull_x + 25}" y2="{cy}" stroke="{RED}" stroke-width="5"/>')
    b.append(t(hull_x + 36, cy + 31, f"abreast: {val('side_clearance')} mm", 28, RED, "start", "bold", halo=True))
    b.append(t(10, H - 14, f"Dashed: a hidden walker's reach after {val('t_ahead_whole')} ms", 28, RED, "start", "bold"))
    svg("rack_top.svg", "\n".join(b), w=W, h=H)


def erosion():
    t_ah, t_age = num("t_ahead"), num("t_age")
    tt = np.linspace(0, 700, 300)
    d0 = num("depth_ahead_mm")
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        y = np.clip(d0 - num("v_walk") * tt, 0, None)  # m/s is mm per ms
        ax.plot(tt, y, color=C["memory"], lw=3)
        ax.plot([t_ah], [0], "o", color=C["danger"], ms=9, zorder=5)
        ax.annotate(f"ahead: {val('t_ahead')} ms", (t_ah, 0), xytext=(470, 330),
                    fontsize=12, color=C["memory"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["memory"]))
        a2 = 1.0 if step >= 2 else 0.0
        ax.axvspan(0, t_age, color=C["neutral_l"], alpha=0.8 * a2, lw=0)
        ax.text(t_age + 8, 300, f"frame age {val('t_age')} ms",
                fontsize=11, color=C["neutral"], alpha=a2, va="center")
        a3 = 1.0 if step >= 3 else 0.0
        ax.plot([0, 700], [0, 0], color=C["danger"], lw=3, ls="--", alpha=a3)
        ax.plot([0], [num("t_abreast")], "o", color=C["danger"], ms=9, alpha=a3, zorder=6)
        ax.text(60, 70, f"abreast of the rack end: {val('t_abreast')} ms", fontsize=12,
                color=C["danger"], fontweight="bold", alpha=a3)
        ax.set_xlim(-10, 700)
        ax.set_ylim(-40, 1000)
        ax.set_xlabel("Time since the frame was captured (ms)")
        ax.set_ylabel("Certified depth (mm)")
        sp.save(fig, out(f"erosion_step{step}.svg"))


def latch():
    lat = K.lat
    t_lin, e_at, r_tol = num("t_lin"), num("e_at_lin"), num("r_tol")
    tt = np.linspace(0, 0.1, 300)
    e0, v0, a_half = num("e0_slot"), num("v_drift"), K.m(lat.quadratic_coefficient, "mm/s**2")
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=COL)
        e = e0 + v0 * tt + a_half * tt**2
        lin = e0 + v0 * tt
        ax.plot(tt * 1000, e, color=C["danger"], lw=3)
        ax.axhline(r_tol, color=C["text"], ls="--", lw=1.5)
        ax.text(2, r_tol + 0.03, f"slot tolerance {val('r_tol')} mm", fontsize=14, color=C["text"])
        ax.axvspan(K.m(lat.horizon, "ms"), 100, color=C["danger_l"], alpha=0.6, lw=0)
        ax.text(K.m(lat.horizon, "ms") - 1, 0.25, f"{val('t_latch')} ms", fontsize=16, color=C["danger"],
                fontweight="bold", ha="right")
        ax.text(5, 0.55, f"bounded error,\nwith {val('a_dist')} mm/s²", fontsize=14, color=C["danger"])
        a = 1.0 if step == 2 else 0.0
        ax.plot(tt * 1000, lin, color=C["neutral"], lw=2.5, ls="--", alpha=a)
        ax.plot([t_lin], [e_at], "o", color=C["danger"], ms=10, alpha=a)
        ax.annotate(f"{val('t_lin')} ms: error {val('e_at_lin')} mm",
                    (t_lin, e_at), xytext=(14, 1.32), fontsize=14,
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
    rows = [("Free space abreast", num("t_abreast"), "danger", f"{val('t_abreast')} ms"),
            ("Latch-slot finger", K.m(K.lat.horizon, "ms"), "compute", f"{val('t_latch')} ms"),
            ("Coworker by the arm", K.m(K.COWORKER.result.horizon, "ms"), "network", f"{val('t_cw')} ms"),
            ("Free space ahead", num("t_ahead"), "memory", f"{val('t_ahead')} ms")]
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
    f, sig, span = num("f_read"), num("f_sigma"), num("bias_span")
    tt = np.linspace(0, span, 100)
    fig, ax = plt.subplots(figsize=COL)
    ax.fill_between(tt, f - sig, f + sig, color=C["memory_l"])
    ax.plot(tt, np.full_like(tt, f), color=C["memory"], lw=3)
    ax.text(0.3, 5.25, f"reported {val('f_read')} N ± {val('f_sigma')}", fontsize=15, color=C["memory"],
            fontweight="bold")
    ax.plot(tt, f - num("bias_rate") * tt, color=C["danger"], lw=3)
    ax.plot([span], [num("f_true")], "o", color=C["danger"], ms=10)
    ax.text(14.8, 1.45, f"true {val('f_true')} N", fontsize=15, color=C["danger"], fontweight="bold",
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
