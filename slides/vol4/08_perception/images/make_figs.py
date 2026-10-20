#!/usr/bin/env python3
"""Figures for Lecture 8 (Sensor Perception), Volume IV.

Every figure is drawn at the size of its slot on the 16 x 9 cm slide, so the
slide places it at scale 1 and the type keeps its size (10 to 12 pt here).
Every number comes from calcs.py (imported as K), which reads MLSysIM, and is
labeled with val(key), the string the slide prints. Sources:

  - MLSysIM: the running machine and its site record, and the Scenarios.*
    the chapter, Sensor Perception, reads (Napkin Math 8.1 to 8.3, tables 3
    to 5, §8.3, 8.4, 8.6, 8.7).
  - KITTI raw data, drive 2011_09_26_0001 and its calibration (Geiger et al.,
    IJRR 2013; CC BY-NC-SA 3.0), in data/kitti: host-clock timestamp logs,
    Velodyne HDL-64E sweep 0, color camera image_02 frame 0, OXTS frame 0.
  - MLPerf Inference v3.1, closed, NVIDIA Jetson AGX Orin, RetinaNet INT8,
    SingleStream (data/).
  - Ouster OS1 datasheet, rev7, firmware 3.1 (data/): range precision
    5 to 30 mm (1 sigma beyond 1 m, 1024 x 10 Hz).

Volume IV roles (palette.md): sensing petrol, learned proposal blue,
permission path green, veto and hazard red, latency amber, plant slate.

Run from this folder: python3 make_figs.py  (then make deck converts SVG to PDF)
"""

import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every derived value comes from calcs.py
from deckvals import val, num  # noqa: E402
from mlsysim.physics.robotics.sensing import calc_stereo_depth_sigma  # noqa: E402

from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

plt = sp.plt
C = dict(sp.C)
C.update({
    "sense": "#007A87", "sense_l": "#D9EFF1",
    "propose": "#2563EB", "propose_l": "#DBE7FD",
    "cog": "#1F407A",
    "permit": "#059669", "permit_l": "#D5F2E7",
    "veto": "#DC2626", "veto_l": "#FBDADA",
    "amber": "#D97706", "amber_l": "#FBE8CC",
    "slate": "#334155", "slate_l": "#E2E8F0",
})
plt.rcParams.update({"font.size": 11, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})
OUT = Path(__file__).parent
DATA = OUT / "data"


def cm(w, h):
    return (w / 2.54, h / 2.54)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, fc, ec, text=None, fs=11, tc=None, bold=False, ls="-",
        r=0.02, lw=1.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or C["text"], fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color, lw=1.8, ls="-"):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=12))


# =============================================================================
# Aisle scene (frame: a correct detection can still arrive too late)
# =============================================================================
def aisle():
    fig, ax = plt.subplots(figsize=cm(8.4, 4.6))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(-0.75, 0.75)
    bare(ax)
    ax.add_patch(Rectangle((-0.5, 0.45), 1.6, 0.16, fc=C["slate_l"], ec=C["slate"], lw=1.2))
    ax.add_patch(Rectangle((-0.5, -0.61), 1.6, 0.16, fc=C["slate_l"], ec=C["slate"], lw=1.2))
    ax.text(0.3, 0.53, "rack", ha="center", va="center", fontsize=10, color=C["slate"])
    ax.text(0.3, -0.53, "rack", ha="center", va="center", fontsize=10, color=C["slate"])
    ax.add_patch(Rectangle((-0.42, -0.15), 0.42, 0.3, fc=C["permit_l"], ec=C["slate"], lw=1.6))
    ax.text(-0.21, 0.0, "base", ha="center", va="center", fontsize=10.5, fontweight="bold")
    arrow(ax, 0.03, 0.0, 0.30, 0.0, C["slate"], lw=2.2)
    ax.text(0.33, 0.0, f"{val('v_drive')} m/s", ha="left", va="center", fontsize=10.5, fontweight="bold")
    ax.add_patch(plt.Circle((1.30, 0.28), 0.07, fc=C["veto_l"], ec=C["veto"], lw=1.8))
    ax.text(1.40, 0.28, "worker", ha="left", va="center", fontsize=10.5, color=C["veto"],
            fontweight="bold")
    arrow(ax, 1.30, 0.20, 1.30, 0.05, C["veto"], lw=1.6)
    dc = num("d_clear")
    ax.plot([dc, dc], [-0.40, 0.40], color=C["text"], lw=1, ls=":")
    ax.text(dc + 0.02, -0.35, "rack end", ha="left", va="center", fontsize=10)
    ax.annotate("", (dc, -0.30), (0.0, -0.30),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.3))
    ax.text(0.55, -0.26, f"clear distance {val('d_clear')} m", ha="center", va="bottom", fontsize=10.5)
    ax.text(0.6, 0.67, f"each ms of delay: {val('d_per_ms')} mm of travel", ha="center", va="bottom",
            fontsize=10.5, color=C["amber"], fontweight="bold")
    save(fig, "aisle")


# =============================================================================
def kitti_calib():
    v2c = {}
    for line in (DATA / "kitti" / "calib_velo_to_cam.txt").read_text().splitlines():
        k, _, v = line.partition(":")
        v2c[k] = v
    c2c = {}
    for line in (DATA / "kitti" / "calib_cam_to_cam.txt").read_text().splitlines():
        k, _, v = line.partition(":")
        c2c[k] = v
    R = np.array(v2c["R"].split(), float).reshape(3, 3)
    T = np.array(v2c["T"].split(), float).reshape(3, 1)
    Rr = np.array(c2c["R_rect_00"].split(), float).reshape(3, 3)
    P = np.array(c2c["P_rect_02"].split(), float).reshape(3, 4)
    return R, T, Rr, P


def project(yaw_err_deg=0.0):
    pts = K.PTS[:, :3]
    pts = pts[pts[:, 0] > 2.0]                       # in front of the car
    R, T, Rr, P = kitti_calib()
    if yaw_err_deg:
        a = math.radians(yaw_err_deg)                # error about the lidar's vertical axis
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        pts = pts @ Rz.T
    cam = Rr @ (R @ pts.T + T)
    keep = cam[2] > 0.5
    cam = cam[:, keep]
    uv = P @ np.vstack([cam, np.ones(cam.shape[1])])
    u, v = uv[0] / uv[2], uv[1] / uv[2]
    depth = cam[2]
    img = np.asarray(Image.open(DATA / "kitti" / "sync" / "image_02_0000000000.png"))
    h, w = img.shape[:2]
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return img, u[m], v[m], depth[m]


def kitti_overlay():
    img, u, v, d = project()
    h, w = img.shape[:2]
    fig = plt.figure(figsize=cm(14.6, 4.6))
    ax = fig.add_axes([0.0, 0.0, 0.85, 1.0])
    ax.imshow(img)
    sc = ax.scatter(u, v, c=np.clip(d, 0, 40), s=0.6, cmap="turbo", vmin=0, vmax=40,
                    linewidths=0, rasterized=True)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    bare(ax)
    cax = fig.add_axes([0.87, 0.08, 0.025, 0.84])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([0, 10, 20, 30, 40])
    cb.ax.tick_params(labelsize=10)
    cb.set_label("lidar range (m)", fontsize=10)
    save(fig, "kitti_overlay")


# =============================================================================
# Who guards the rack end (camera proposes, lidar admits, scanner stops)
# =============================================================================
def guard():
    fig, ax = plt.subplots(figsize=cm(9.6, 5.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    bare(ax)
    # Brain lane
    box(ax, 0.1, 4.45, 2.6, 1.15, C["sense_l"], C["sense"], "navigation\ncamera", fs=10.5)
    box(ax, 3.4, 4.45, 3.3, 1.15, C["propose_l"], C["propose"], "learned proposer\nclass, heading", fs=10.5, ls="--")
    arrow(ax, 2.7, 5.02, 3.4, 5.02, C["sense"])
    ax.text(7.0, 5.02, "proposes", fontsize=11, fontweight="bold", color=C["propose"],
            va="center")
    # proposal boundary
    ax.plot([0.1, 9.9], [4.05, 4.05], color=C["slate"], lw=1.4, ls=(0, (6, 4)))
    ax.text(9.9, 3.82, "proposal boundary", fontsize=10, color=C["slate"], ha="right", va="center")
    # permission lane
    box(ax, 0.1, 2.2, 2.6, 1.15, C["sense_l"], C["sense"], "spinning\nlidar", fs=10.5)
    box(ax, 3.4, 2.2, 3.3, 1.15, C["permit_l"], C["permit"], "clearance check\non safety MCU", fs=10.5)
    arrow(ax, 2.7, 2.77, 3.4, 2.77, C["sense"])
    arrow(ax, 5.05, 4.45, 5.05, 3.35, C["propose"], ls="--")
    ax.text(7.0, 2.77, "admits or\nrefuses", fontsize=11, fontweight="bold", color=C["permit"],
            va="center")
    # certified floor
    box(ax, 0.1, 0.15, 2.6, 1.15, C["sense_l"], C["sense"], "safety laser\nscanner", fs=10.5)
    box(ax, 3.4, 0.15, 3.3, 1.15, C["veto_l"], C["veto"], "certified stop\n(PL d, no classes)", fs=10)
    arrow(ax, 2.7, 0.72, 3.4, 0.72, C["sense"])
    ax.text(7.0, 0.72, "stops", fontsize=11, fontweight="bold", color=C["veto"], va="center")
    save(fig, "guard")


# =============================================================================
# Capture epoch: midpoint, arrival stamp, trigger stamp
# =============================================================================
def capture(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 4.6))
    fig.subplots_adjust(0.02, 0.25, 0.98, 0.80)
    ax.set_xlim(-27, 34)
    ax.set_ylim(-0.3, 2.9)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([-8, 0, 8, 16, 24, 32])
    ax.set_xlabel("time from the exposure midpoint (ms)", fontsize=10.5)
    # exposure
    ax.barh(1.9, K.T_EXP, left=-K.T_EXP / 2, height=0.5, color=C["sense"])
    ax.text(0, 1.9, f"exposure {val('t_exp')} ms", color="white", ha="center", va="center", fontsize=10,
            fontweight="bold")
    ax.axvline(0, color=C["permit"], lw=2)
    ax.text(0.4, 2.5, "capture epoch t₀ (midpoint)", color=C["permit"], fontsize=10.5,
            fontweight="bold", va="center")
    if step >= 2:
        ax.barh(1.1, K.T_READ, left=K.T_EXP / 2, height=0.5, color=C["neutral"])
        ax.text(K.T_EXP / 2 + K.T_READ / 2, 1.1, f"readout {val('t_readout')}", color="white", ha="center", va="center", fontsize=10,
                fontweight="bold")
        ax.barh(1.1, num("t_link_dma"), left=K.T_CAPTURE, height=0.5, color=C["amber"])
        ax.text(K.T_ARRIVAL + 0.4, 1.1, "link + DMA", color=C["amber"], ha="left", va="center", fontsize=10)
        ax.axvline(K.T_ARRIVAL, color=C["veto"], lw=2, ls="--")
        ax.text(K.T_ARRIVAL - 0.4, 0.25, f"arrival stamp: {val('t_arrival')} ms late\n= {val('d_arrival')} mm",
                color=C["veto"], fontsize=10.5, fontweight="bold", ha="right", va="center")
    if step >= 3:
        ax.axvline(-K.T_EXP / 2, color=C["amber"], lw=2, ls="--")
        ax.text(-K.T_EXP / 2 - 0.6, 1.1, f"trigger edge:\n{val('t_trigger')} ms early = {val('d_trigger')} mm", color=C["amber"],
                fontsize=10, fontweight="bold", ha="right", va="center")
    # travel axis on top
    top = ax.secondary_xaxis("top", functions=(lambda t: t * K.V, lambda d: d / K.V))
    top.set_xticks([-36, -24, -12, 0, 12, 24, 36, 48])
    top.set_xlabel(f"base travel at {val('v_drive')} m/s (mm)", fontsize=10)
    save(fig, f"capture_step{step}")


# =============================================================================
# KITTI: one trigger, four host stamps (real log)
# =============================================================================
def kitti_strip():
    i = 20                                    # one logged sweep, chosen from the middle
    t0 = K.K_VS[i]
    vs, vf, ve = 0.0, K.K_VF[i] - t0, K.K_VE[i] - t0
    cams = {}
    for c, t in K.K_CAM.items():
        j = np.argmin(abs(t - K.K_VF[i]))
        cams[c] = t[j] - t0
    fig, ax = plt.subplots(figsize=cm(8.6, 4.4))
    fig.subplots_adjust(0.03, 0.27, 0.97, 0.98)
    ax.set_xlim(-4, 132)
    ax.set_ylim(-0.3, 2.3)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"ms from sweep start (sweep {i})", fontsize=10.5)
    ax.barh(1.85, ve - vs, left=vs, height=0.35, color=C["sense_l"], ec=C["sense"], lw=1.2)
    ax.text(2, 1.85, "lidar sweep", fontsize=10, color=C["sense"], va="center", fontweight="bold")
    ax.axvline(vf, color=C["permit"], lw=2)
    ax.text(vf - 1.5, 1.32, "faces forward:\ntriggers cameras", fontsize=10, color=C["permit"],
            ha="right", va="center", fontweight="bold")
    for c, y, col in [("image_02", 0.85, C["propose"]), ("image_03", 0.85, C["propose"]),
                      ("image_00", 0.3, C["neutral"]), ("image_01", 0.3, C["neutral"])]:
        ax.plot([cams[c]], [y], "o", ms=7, color=col, mec="white", mew=1)
    ax.text(cams["image_02"] + 2.5, 0.85, f"color stamps +{cams['image_02'] - vf:.1f}", fontsize=10, color=C["propose"],
            va="center", fontweight="bold")
    ax.text(cams["image_00"] + 2.5, 0.3, f"gray stamps +{cams['image_00'] - vf:.1f}", fontsize=10, color=C["neutral"],
            va="center", fontweight="bold")
    save(fig, "kitti_strip")


# =============================================================================
# Rolling-shutter rows (Napkin Math 8.1)
# =============================================================================
def rows():
    fig, ax = plt.subplots(figsize=cm(7.4, 5.0))
    fig.subplots_adjust(0.22, 0.20, 0.96, 0.80)
    r = np.array([0, K.ROWS])
    x = K.V * r * K.T_READ / K.ROWS
    blur = num("d_exposure")
    ax.fill_betweenx(r, x - blur / 2, x + blur / 2,
                     color=C["sense_l"], lw=0)
    ax.plot(x, r, color=C["sense"], lw=2.6)
    ax.axvline(K.D_LOC, color=C["veto"], lw=1.8, ls="--")
    ax.text(K.D_LOC - 1, 300, f"{val('delta_loc')} mm\nallowance", color=C["veto"], fontsize=10, ha="right",
            va="top", fontweight="bold")
    ax.text(num("d_skew") + 1.5, K.ROWS - 120, f"{val('d_skew')} mm", color=C["sense"], fontsize=11,
            fontweight="bold", va="center")
    ax.set_ylim(K.ROWS + 150, -150)
    ax.set_xlim(-14, 56)
    ax.set_yticks([0, 1000, 2000, K.ROWS])
    ax.set_yticklabels(["0", "1,000", "2,000", val("rows")])
    ax.set_ylabel("image row")
    ax.set_xlabel("camera position (mm)")
    top = ax.secondary_xaxis("top", functions=(lambda d: d / K.V, lambda t: t * K.V))
    top.set_xticks([0, K.T_EXP / 2, K.T_READ, 33])
    top.set_xticklabels(["0", val("t_trigger"), val("t_readout"), "33"])
    top.set_xlabel("row read time (ms)", fontsize=10)
    save(fig, "rows")


# =============================================================================
# KITTI: one Velodyne sweep, colored by firing time (real data)
# =============================================================================
def kitti_sweep():
    p = K.PTS
    x, y = p[:, 0], p[:, 1]
    m = (abs(x) < 32) & (abs(y) < 20)
    x, y = x[m], y[m]
    # KITTI stores no per-point times. The sweep starts and ends facing
    # backward and faces forward (the camera trigger) halfway (timestamps.txt),
    # so a point's time from the trigger is its angle from the front as a
    # share of the sweep, whichever way the head spins.
    az = np.degrees(np.arctan2(y, x))           # 0 = forward
    t = abs(az) / 360 * K.SWEEP0
    fig, ax = plt.subplots(figsize=cm(8.0, 5.4))
    fig.subplots_adjust(0.02, 0.02, 0.80, 0.98)
    sc = ax.scatter(x, y, c=t, s=0.25, cmap="viridis", vmin=0, vmax=K.SWEEP0 / 2,
                    linewidths=0, rasterized=True)
    ax.add_patch(Rectangle((-2.2, -0.9), 4.4, 1.8, fc="white", ec=C["veto"], lw=1.6))
    arrow(ax, 2.4, 0, 6.5, 0, C["veto"], lw=2)
    ax.text(4.4, 2.0, f"car {val('kitti_v')} m/s", color=C["veto"], fontsize=10.5, ha="center",
            fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.plot([-3, -31], [0, 0], color=C["amber"], lw=2, ls="--")
    ax.text(-17, -6, "seam: start and end\nof one sweep", color=C["amber"], fontsize=10,
            ha="center", va="top", fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xlim(-32, 32)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    bare(ax)
    cax = fig.add_axes([0.83, 0.12, 0.035, 0.76])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([0, 25, 50])
    cb.ax.tick_params(labelsize=10)
    cb.set_label("ms from the trigger (±)", fontsize=10)
    save(fig, "kitti_sweep")


# =============================================================================
# Coordinate frame tree (SP figure 4, redrawn at slot size)
# =============================================================================
def frames():
    fig, ax = plt.subplots(figsize=cm(6.2, 5.6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 5.6)
    bare(ax)
    names = ["earth", "map", "odom", "base_link", "sensor_link"]
    notes = ["fixed datum, if present", "may jump at relocalization",
             "continuous, drifts", "calibrated mount"]
    ys = [5.0, 3.85, 2.7, 1.55, 0.4]
    for i, (n, y) in enumerate(zip(names, ys)):
        col = C["sense"] if n == "sensor_link" else C["slate"]
        fc = C["sense_l"] if n == "sensor_link" else C["slate_l"]
        box(ax, 0.15, y - 0.3, 2.3, 0.6, fc, col, n, fs=11, ls="--" if n == "earth" else "-")
        if i < 4:
            arrow(ax, 1.3, y - 0.3, 1.3, ys[i + 1] + 0.3, C["slate"], lw=1.5)
            ax.text(1.5, (y + ys[i + 1]) / 2, notes[i], fontsize=9.5, va="center", color=C["text"])
    save(fig, "frames")


# =============================================================================
# Error sources against the allowances (SP figure, §8.3)
# =============================================================================
def errbars():
    fig, ax = plt.subplots(figsize=cm(14.6, 4.6))
    fig.subplots_adjust(0.27, 0.22, 0.97, 0.88)
    bars = [  # (label, value key), top to bottom
        (f"{val('angle_err')}° mount error at {val('mount_range')} m", "d_deg_far"),
        ("Dispatch age", "d_age"),
        ("Arrival stamp", "d_arrival"),
        ("Row skew", "d_skew"),
        ("Trigger stamp", "d_trigger"),
    ][::-1]
    labs = [n for n, _ in bars]
    keys = [k for _, k in bars]
    raw = {"d_deg_far": K.m(K.MOUNT.result.angle_lateral, "mm"), "d_age": K.m(K.AGE.age_travel, "mm"),
           "d_arrival": K.m(K.AGE.arrival_travel, "mm"), "d_skew": K.m(K.SHUTTER.skew, "mm"),
           "d_trigger": K.m(K.AGE.trigger_travel, "mm")}
    vals = [raw[k] for k in keys]
    cols = [C["amber"], C["sense"], C["amber"], C["amber"], C["slate"]]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v, k in zip(y, vals, keys):
        ax.text(v + 4, yi, f"{val(k)} mm", va="center", fontsize=10.5,
                fontweight="bold", color=C["text"],
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=10.5)
    ax.axvline(K.D_LOC, color=C["slate"], lw=1.6, ls=":")
    ax.text(K.D_LOC + 3, 4.55, f"{val('delta_loc')} mm localization", fontsize=10, color=C["slate"], va="bottom")
    ax.axvline(K.SIDE, color=C["veto"], lw=1.8, ls="--")
    ax.text(K.SIDE + 3, 4.55, f"{val('side_clearance')} mm side clearance", fontsize=10, color=C["veto"], va="bottom")
    ax.set_xlim(0, 320)
    ax.set_ylim(-0.5, 5.0)
    ax.set_xlabel(f"position error at the {val('v_drive')} m/s drive limit (mm)", fontsize=10.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "errbars")


# =============================================================================
# The latency waterfall (Napkin Math 8.2), revealed in steps
# =============================================================================
def waterfall(step):
    fig, ax = plt.subplots(figsize=cm(14.6, 5.0))
    fig.subplots_adjust(0.20, 0.17, 0.97, 0.80)
    names = ["Exposure midpoint", "Rolling readout", "Link transport", "DMA and queue", "ISP",
             "Vision backbone", "Dispatch"]
    n = len(K.STAGES)
    shown = {1: 0, 2: 2, 3: 5, 4: 7}[step]
    cols = [C["sense"], C["sense"], C["amber"], C["amber"], C["amber"], C["propose"], C["amber"]]
    for i, ((_, t, cum), name) in enumerate(zip(K.STAGES, names)):
        y = n - 1 - i
        vis = i < shown
        ax.barh(y, t, left=cum - t, height=0.66, color=cols[i], alpha=1.0 if vis else 0.0)
        ax.text(-0.8, y, name, ha="right", va="center", fontsize=10.5,
                color=C["text"] if vis else "white")
        ax.text(cum + 0.6, y, f"{t:g}", ha="left", va="center", fontsize=10,
                color=C["text"] if vis else "white", fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("age since the exposure midpoint (ms)", fontsize=10.5)
    top = ax.secondary_xaxis("top", functions=(lambda t: t * K.V, lambda d: d / K.V))
    top.set_xticks([0, 15, 30, 45, 60, 75, 90])
    top.set_xlabel(f"base travel at {val('v_drive')} m/s (mm)", fontsize=10)
    if step == 1:
        ax.axvline(K.T_AGE, color=C["veto"], lw=2, ls="--")
        ax.text(K.T_AGE - 0.6, 3.0, f"dispatch at {val('t_age')} ms:\nwhich stage\ndominates?",
                color=C["veto"], fontsize=11, fontweight="bold", ha="right", va="center")
    if step >= 2:
        ax.axvline(K.T_CAPTURE, color=C["sense"], lw=1.4, ls=":")
    if step == 4:
        ax.axvline(K.T_AGE, color=C["veto"], lw=2, ls="--")
        ax.text(K.T_AGE - 0.5, 2.9, f"{val('t_age')} ms\n{val('d_age')} mm", color=C["veto"],
                fontsize=11.5, fontweight="bold", ha="right", va="center")
    if step >= 2:
        ax.text(K.T_CAPTURE - 0.5, 0.4, f"{val('t_capture')} ms before\nany computation",
                color=C["sense"], fontsize=10, fontweight="bold", ha="right", va="center")
    save(fig, f"waterfall_step{step}")


# =============================================================================
# A measured detector against the allocations (MLPerf v3.1, Orin)
# =============================================================================
def mlperf():
    rows_ = [
        ("RetinaNet p99,\nmeasured", "rn_p99_short", C["propose"]),
        ("backbone\nallocation", "t_st_backbone", C["propose_l"]),
        ("exposure +\nreadout", "t_capture", C["sense"]),
        ("age at\ndispatch", "t_age", C["amber"]),
    ]
    fig, ax = plt.subplots(figsize=cm(8.4, 4.8))
    fig.subplots_adjust(0.30, 0.20, 0.95, 0.97)
    for i, (lab, key, col) in enumerate(rows_):
        y = len(rows_) - 1 - i
        v = {"rn_p99_short": K.RN_P99, "t_st_backbone": K.m(K.AISLE.t_backbone, "ms"),
             "t_capture": K.T_CAPTURE, "t_age": K.T_AGE}[key]
        ax.barh(y, v, color=col, height=0.62, ec=C["propose"] if col == C["propose_l"] else col)
        ax.text(v + 1, y, val(key), va="center", fontsize=10.5, fontweight="bold")
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 62)
    ax.set_xlabel("milliseconds", fontsize=10.5)
    save(fig, "mlperf")


# =============================================================================
# Current evidence age keeps growing after dispatch
# =============================================================================
def agegrow():
    fig, ax = plt.subplots(figsize=cm(14.6, 3.9))
    fig.subplots_adjust(0.02, 0.28, 0.98, 0.95)
    ax.set_xlim(-4, 150)
    ax.set_ylim(-0.2, 2.75)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 25, K.T_AGE, 75, 100, 125, 150])
    ax.set_xticklabels(["0", "25", val("t_age"), "75", "100", "125", "150"])
    ax.set_xlabel("ms since capture t₀, read on the consumer's clock", fontsize=10.5)
    ax.barh(1.5, K.T_AGE, left=0, height=0.45, color=C["amber"])
    ax.text(K.T_AGE / 2, 1.5, "observation age at dispatch", color="white", fontsize=10,
            ha="center", va="center", fontweight="bold")
    ax.barh(1.5, 36, left=K.T_AGE, height=0.45, color="white", ec=C["amber"], lw=1.4, hatch="//")
    ax.text(K.T_AGE + 18, 1.95, "dispatch to admission\n(not bounded here)", color=C["amber"],
            fontsize=9.5, ha="center", va="bottom")
    ax.barh(1.5, 150 - K.T_AGE - 36, left=K.T_AGE + 36, height=0.45, color=C["amber_l"])
    ax.text(K.T_AGE + 36 + 30, 1.5, "reuse at every tick", color=C["amber"], fontsize=10,
            ha="center", va="center")
    ax.annotate("", (148, 0.75), (0, 0.75), arrowprops=dict(arrowstyle="-|>", color=C["text"], lw=1.6))
    ax.text(100, 0.33, "current evidence age = now − t₀ (keeps growing)", ha="center", va="center",
            fontsize=10.5, fontweight="bold")
    ax.axvline(K.T_AGE + K.SYNC_BOUND, color=C["veto"], lw=1.8, ls="--", ymax=0.93)
    ax.text(K.T_AGE + 1.5, 0.95, f"dispatch limit {val('t_dispatch_limit')}", color=C["veto"],
            fontsize=9.5, ha="left", va="center", fontweight="bold")
    save(fig, "agegrow")


# =============================================================================
# KITTI: stamp periods, lidar against cameras (real data)
# =============================================================================
def kitti_jitter():
    series = [
        ("lidar", np.diff(K.K_VF), C["sense"]),
        ("color cam", np.diff(K.K_CAM[K.COLOR]), C["propose"]),
        ("gray cam", np.diff(K.K_CAM[K.GRAY]), C["neutral"]),
    ]
    fig, ax = plt.subplots(figsize=cm(8.4, 4.8))
    fig.subplots_adjust(0.25, 0.20, 0.96, 0.97)
    rng = np.random.default_rng(0)
    for i, (lab, d, col) in enumerate(series):
        y = len(series) - 1 - i
        ax.scatter(d, y + rng.uniform(-0.18, 0.18, len(d)), s=9, color=col, alpha=0.75, lw=0)
        ax.text(99.6, y, lab, ha="right", va="center", fontsize=10.5, color=col, fontweight="bold")
        ax.text(d.max() + 0.15, y + 0.28, f"{d.min():.1f} to {d.max():.1f}", fontsize=9.5,
                color=col, va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(99.7, 107.6)
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlabel("ms between consecutive stamps", fontsize=10.5)
    save(fig, "kitti_jitter")


# =============================================================================
# Delay spends phase margin (SP §8.4.4)
# =============================================================================
def phase():
    fig, ax = plt.subplots(figsize=cm(8.0, 4.6))
    fig.subplots_adjust(0.18, 0.22, 0.96, 0.95)
    t = np.linspace(0, 100, 200)
    pm0, wc = K.m(K.PHASE.phase_margin, "deg"), K.m(K.PHASE.crossover, "rad/s")
    t_pm, t_used = K.m(K.PHASE.result.delay_ceiling, "ms"), K.m(K.PHASE.delay, "ms")
    pm = pm0 - np.degrees(wc * t / 1000)
    ok = t <= t_pm
    ax.plot(t[ok], pm[ok], color=C["permit"], lw=2.4)
    ax.plot(t[~ok], pm[~ok], color=C["veto"], lw=2.4)
    ax.axhspan(-15, 0, color=C["veto_l"], lw=0)
    ax.axhline(0, color=C["text"], lw=1, ls="--")
    used_pm = pm0 - math.degrees(wc * t_used / 1000)
    ax.plot([t_used], [used_pm], "o", color=C["amber"], ms=8)
    ax.text(t_used + 2, used_pm + 2.5, f"{val('t_pm_used')} ms", color=C["amber"], fontsize=10.5,
            fontweight="bold", ha="left")
    ax.plot([t_pm], [0], "o", color=C["veto"], ms=8)
    ax.text(t_pm + 2, 4, f"{val('t_pm')} ms", color=C["veto"], fontsize=10.5, fontweight="bold")
    ax.text(60, -10, "unstable", color=C["veto"], fontsize=10.5, ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(-15, 50)
    ax.set_xlabel("observation age in the loop (ms)")
    ax.set_ylabel("phase margin (°)")
    save(fig, "phase")


# =============================================================================
# Stereo range wall against lidar precision (Napkin Math 8.3, datasheet)
# =============================================================================
def stereo(step):
    fig, ax = plt.subplots(figsize=cm(9.4, 5.3))
    fig.subplots_adjust(0.17, 0.19, 0.97, 0.96)
    rig = K.STEREO

    def sigma_mm(zm):
        return calc_stereo_depth_sigma(K.Q_(zm, "m"), rig.focal_length_px, rig.baseline,
                                       rig.disparity_sigma_px).to("mm").magnitude

    z = np.linspace(0.3, 10, 300)
    ax.plot(z, [sigma_mm(v) for v in z], color=C["propose"], lw=2.6)
    ax.text(7.6, 285, "stereo σz grows as z²", color=C["propose"], fontsize=10.5,
            fontweight="bold", ha="right")
    ax.plot([K.m(K.AISLE.d_clear, "m")], [K.m(rig.result.sigma_target, "mm")], "o", color=C["propose"], ms=7)
    ax.text(0.4, 205, f"rack end ({val('d_clear')} m): {val('st_rack')} mm", fontsize=10, color=C["propose"])
    if step >= 2:
        ax.axhspan(num("os1_prec_min"), num("os1_prec_max"), color=C["sense_l"], lw=0)
        ax.text(9.9, 15, "OS1 lidar, datasheet 1σ", color=C["sense"], fontsize=9.5,
                ha="right", va="center", fontweight="bold")
    ax.axhline(K.D_LOC, color=C["slate"], lw=1.4, ls=":")
    ax.axhline(K.SIDE, color=C["veto"], lw=1.6, ls="--")
    ax.text(9.9, K.D_LOC + 6, f"{val('delta_loc')} mm localization", fontsize=10, color=C["slate"], ha="right")
    ax.text(0.4, K.SIDE + 5, f"{val('side_clearance')} mm side clearance", fontsize=10, color=C["veto"])
    if step >= 3:
        for c, key, col in [(K.D_LOC, "st_z_loc", C["slate"]), (K.SIDE, "st_z_side", C["veto"])]:
            zc = K.m({"st_z_loc": rig.result.range_at_loc_allowance,
                      "st_z_side": rig.result.range_at_side_clearance}[key], "m")
            ax.plot([zc], [c], "o", ms=8, color=col)
            if c < K.SIDE:
                ax.text(zc - 0.2, c + 4, f"{val(key)} m", color=col, fontsize=11, fontweight="bold", ha="right", va="bottom")
            else:
                ax.text(zc + 0.15, c - 12, f"{val(key)} m", color=col, fontsize=11, fontweight="bold", ha="left", va="top")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 330)
    ax.set_xlabel("range z (m)")
    ax.set_ylabel("1σ depth error (mm)")
    save(fig, f"stereo_step{step}")


# =============================================================================
# The clearance inset rests on a measured tail (SP eq. 9, figure 6b)
# =============================================================================
def inset(step):
    fig, ax = plt.subplots(figsize=cm(9.2, 4.9))
    fig.subplots_adjust(0.03, 0.20, 0.97, 0.97)
    x = np.linspace(-120, 420, 600)
    c_safe, q = 150.0, 120.0
    est = c_safe + q
    # heavy-tailed residual density centered on the estimate (schematic)
    from math import gamma
    nu, s = 2.0, 40.0
    dens = (1 + ((x - est) / s) ** 2 / nu) ** (-(nu + 1) / 2)
    dens = dens / dens.max()
    ax.plot(x, dens, color=C["propose"], lw=2.2)
    tail = x <= c_safe
    if step >= 2:
        ax.fill_between(x[tail], 0, dens[tail], color=C["veto"], alpha=0.55, lw=0)
        ax.text(60, 0.16, "residual tail\nand bias", color=C["veto"], fontsize=10, ha="center",
                fontweight="bold")
    ax.axvline(0, color=C["slate"], lw=2.4)
    ax.text(4, 1.02, "body", fontsize=10.5, color=C["slate"], fontweight="bold", va="bottom")
    ax.axvline(c_safe, color=C["veto"], lw=1.6, ls="--")
    ax.text(c_safe + 3, 1.02, "C_safe", fontsize=10.5, color=C["veto"], va="bottom", fontweight="bold")
    ax.axvline(est, color=C["permit"], lw=1.8, ls="--")
    ax.text(est + 3, 1.02, "C_inset", fontsize=10.5, color=C["permit"],
            va="bottom", fontweight="bold")
    if step >= 2:
        ax.annotate("", (est, 0.86), (c_safe, 0.86),
                    arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.4))
        ax.text(c_safe - 6, 0.80, "measured p99\nresidual", ha="right", va="top", fontsize=10)
    ax.set_xlim(-30, 420)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    ax.set_xticks([])
    for s_ in ("left", "top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel("distance from the machine's envelope (schematic)", fontsize=10.5)
    save(fig, f"inset_step{step}")


# =============================================================================
# Metric veto: a 2 Hz label against 20 Hz range evidence (SP Systems Perspective 8.3)
# =============================================================================
def veto():
    fig, ax = plt.subplots(figsize=cm(14.6, 4.4))
    fig.subplots_adjust(0.17, 0.24, 0.98, 0.97)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.4, 2.6)
    ax.set_yticks([0.3, 1.3, 2.2])
    ax.set_yticklabels(["permission\npath", f"lidar\n({val('f_lidar')} Hz)", f"vision-language\nmodel ({val('f_vlm')} Hz)"],
                       fontsize=10)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time (ms)", fontsize=10.5)
    for t in np.arange(0, 1000, K.T_VLM):
        ax.barh(2.2, K.T_VLM - 30, left=t + 15, height=0.45, color=C["propose_l"], ec=C["propose"], ls="--")
        ax.text(t + K.T_VLM / 2, 2.2, '"clear aisle"', ha="center", va="center", fontsize=10.5,
                color=C["propose"], fontweight="bold")
    hit = 650
    for t in np.arange(0, 1000, K.T_SWEEP):
        col = C["veto"] if t >= hit else C["sense"]
        ax.plot([t + K.T_SWEEP / 2], [1.3], "o", ms=6.5, color=col)
    ax.text(hit + 60, 1.60, "return inside the envelope", color=C["veto"], fontsize=10,
            fontweight="bold", va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", pad=0.5))
    ax.barh(0.3, hit + 50, left=0, height=0.4, color=C["permit_l"], ec=C["permit"])
    ax.text(330, 0.3, "admits", ha="center", va="center", fontsize=10.5, color=C["permit"],
            fontweight="bold")
    ax.barh(0.3, 1000 - hit - 50, left=hit + 50, height=0.4, color=C["veto"])
    ax.text((hit + 50 + 1000) / 2, 0.3, "halts", ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold")
    arrow(ax, hit + 25, 1.12, hit + 50, 0.52, C["veto"])
    save(fig, "veto")


# =============================================================================
# A lost frame against a misdated frame (SP §8.7)
# =============================================================================
def lostmis(step=2):
    fig, axs = plt.subplots(1, 2, figsize=cm(14.6, 4.4))
    fig.subplots_adjust(0.02, 0.24, 0.98, 0.84, wspace=0.10)
    x = np.linspace(-80, 120, 400)
    true = 0.0

    def g(mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    for ax, title, col in [(axs[0], "lost frame: flag raised", C["permit"]),
                           (axs[1], "misdated frame: no flag", C["veto"])]:
        ax.set_yticks([])
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-80, 120)
        ax.set_ylim(0, 1.25)
        ax.axvline(true, color=C["text"], lw=1.6)
        ax.text(true - 2, 1.12, "true position", ha="right", fontsize=10)
        ax.set_title(title, fontsize=11, color=col, fontweight="bold", pad=4)
        ax.set_xlabel("obstacle position (mm)", fontsize=10.5)
    # left: estimate before and after a missing update: spread grows, honest
    axs[0].plot(x, g(4, 12), color=C["neutral"], lw=1.5, ls="--")
    axs[0].plot(x, 0.62 * g(4, 26), color=C["permit"], lw=2.4)
    axs[0].text(40, 0.5, f"spread grows;\n{val('t_frame')} ms older", color=C["permit"], fontsize=10)
    # right: misdated by 26.6 ms: narrow, centered 39.9 mm off
    off = K.m(K.AGE.arrival_travel, "mm")
    axs[1].plot(x, g(4, 12), color=C["neutral"], lw=1.5, ls="--")
    axs[1].plot(x, g(off, 8), color=C["veto"], lw=2.4)
    axs[1].annotate("", (off, 0.35), (0, 0.35), arrowprops=dict(arrowstyle="<->", color=C["veto"], lw=1.4))
    axs[1].text(off + 14, 0.65, f"spread shrinks;\n{val('d_arrival')} mm off", color=C["veto"], fontsize=10)
    save(fig, "lostmis")


# =============================================================================
# KITTI: a saturation monitor on a real frame
# =============================================================================
def kitti_sat():
    img = np.asarray(Image.open(DATA / "kitti" / "sync" / "image_02_0000000000.png").convert("RGB"))
    gray = np.asarray(Image.open(DATA / "kitti" / "sync" / "image_02_0000000000.png").convert("L"))
    full = num("pix_max")
    sat = gray >= full
    over = img.copy().astype(float) / 255
    over[sat] = [0.86, 0.15, 0.15]
    fig = plt.figure(figsize=cm(14.6, 4.6))
    ax = fig.add_axes([0.0, 0.08, 0.66, 0.90])
    ax.imshow(over, interpolation="bilinear")
    bare(ax)
    ax.text(10, 360, f"red: pixels clipped at {val('pix_max')}", color="white", fontsize=10.5, fontweight="bold",
            va="bottom", bbox=dict(fc=C["veto"], ec="none", pad=1.5))
    ax2 = fig.add_axes([0.73, 0.27, 0.25, 0.68])
    h = np.bincount(gray.ravel(), minlength=256) / gray.size * 100
    ax2.bar(np.arange(255), h[:255], width=1.0, color=C["neutral"], lw=0)
    ax2.bar([254], [h[255]], width=4.0, color=C["veto"], lw=0)
    ax2.set_xlim(0, 262)
    ax2.set_xticks([0, 128, 255])
    ax2.set_yticks([0, 4, 8])
    ax2.set_xlabel("pixel value", fontsize=10)
    ax2.set_ylabel("% of pixels", fontsize=10)
    ax2.text(250, h[255] - 0.3, f"{val('sat_pct')}%", color=C["veto"], fontsize=10.5,
             fontweight="bold", ha="right", va="top")
    save(fig, "kitti_sat")


if __name__ == "__main__":
    aisle()
    kitti_overlay()
    guard()
    for s in (1, 2, 3):
        capture(s)
    kitti_strip()
    rows()
    kitti_sweep()
    frames()
    errbars()
    for s in (1, 2, 3, 4):
        waterfall(s)
    mlperf()
    agegrow()
    kitti_jitter()
    phase()
    for s in (1, 2, 3):
        stereo(s)
    for s in (1, 2):
        inset(s)
    veto()
    lostmis()
    kitti_sat()
    print("figures written to", OUT)
