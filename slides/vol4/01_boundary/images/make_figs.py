#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 1 (The Causal Boundary), Volume IV.

Every figure is drawn at the size of its slot on the 16:9 slide (the slide
body is 15 cm, 425 pt, wide), so nothing is scaled when it is placed:
SVG diagrams use point units (viewBox in pt, width and height in pt) and
matplotlib plots use their display size in inches. Labels are 8.5 to 10 pt.

Sources (numbers derived in calcs.py, which this file imports):
  - Running machine: 1.5 m/s, 2 m/s^2, 1.10 m, pre-brake delay 51.6 + 60 +
    1 + 1 + 20 = 133.6 ms with a 22 ms backbone (The Causal Boundary,
    §1.4.2 table 2, Napkin Math 1.1).
  - Tempe: NTSB HAR-19/03, the table of selected parameters recorded by the
    ADS (pp. 15-16) and footnote 33 (0.71 g ADS cap, 0.92 g road).
  - Training compute: Epoch AI via the mlsysim registry
    (Literature.ModelsByDomain), as the chapter's figure 2.
  - Scope test, command chain, five levels, record chain: the chapter's
    §§1.4.1, 1.6, 1.7, 1.11 (figures 3, 6, 7, 9), redrawn at slide size.
Colors follow the Volume IV causal-boundary palette (volumes/vol4/palette.md):
navy and blue for learned proposals, green for the deterministic permission
path, slate for the plant, petrol for sensing, red for hazards and limits.

Run from this folder: python3 make_figs.py
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

import calcs  # noqa: E402

plt = sp.plt
OUT = Path(__file__).parent

NAVY, BLUE, GREEN, RED = "#1F407A", "#2563EB", "#059669", "#DC2626"
PETROL, SLATE, AMBER, INK = "#007A87", "#334155", "#D97706", "#1A1A2E"
GRAY, LGRAY, ORANGE = "#666666", "#B8B8B8", "#E67817"
FILL = {"navy": "#EFF6FF", "green": "#ECFDF5", "red": "#FEF2F2",
        "slate": "#F1F5F9", "petrol": "#F0FDFA", "amber": "#FFF7ED", "gray": "#F3F4F6"}
FONT = "Helvetica, Arial, sans-serif"

plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "axes.edgecolor": "#888888"})


# ---------------------------------------------------------------------------
# A small SVG writer in point units
# ---------------------------------------------------------------------------
class Svg:
    def __init__(self, w, h, scale=1.0):
        self.w, self.h, self.parts, self.k = w, h, [], scale

    def rect(self, x, y, w, h, fill="white", stroke=SLATE, sw=1.0, rx=3, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

    def text(self, x, y, s, size=9.5, color=INK, anchor="middle", weight="normal", italic=False):
        st = ' font-style="italic"' if italic else ""
        for i, line in enumerate(s.split("\n")):
            self.parts.append(f'<text x="{x:.1f}" y="{y + i * size * 1.18:.1f}" font-family="{FONT}" '
                              f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                              f'font-weight="{weight}"{st}>{line}</text>')

    def line(self, x1, y1, x2, y2, color=SLATE, sw=1.2, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, color=SLATE, sw=1.2, dash=None, arrow=False, fill="none"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}"{da}{m}/>')

    def circle(self, cx, cy, r, fill, stroke, sw=1.2, opacity=1):
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
                          f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')

    def save(self, name):
        cols = {SLATE, RED, GREEN, BLUE, NAVY, AMBER, GRAY, PETROL, INK}
        markers = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in cols)
        W, H = self.w * self.k, self.h * self.k
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.1f}pt" height="{H:.1f}pt" '
               f'viewBox="0 0 {W:.1f} {H:.1f}"><defs>{markers}</defs>'
               f'<rect x="0" y="0" width="{W:.1f}" height="{H:.1f}" fill="white"/>'
               f'<g transform="scale({self.k})">' + "".join(self.parts) + "</g></svg>")
        (OUT / f"{name}.svg").write_text(svg)


def save_plot(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg", bbox_inches=None, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Command-to-motion chain, three steps (full width, 404 x 124 pt)
# ---------------------------------------------------------------------------
def chain():
    boxes = [("tensor\n(a proposal)", NAVY, "navy", True),
             ("permission\npath admits", GREEN, "green", False),
             ("position and\ncurrent loops", GREEN, "green", False),
             ("PWM register\nwrite", GREEN, "green", False),
             ("current in\nthe winding", SLATE, "slate", False),
             ("torque moves\nmass", SLATE, "slate", False)]
    W, bw, y0, bh = 404, 60, 40, 40
    gap = (W - 6 * bw) / 5
    for step in (1, 2, 3):
        s = Svg(W, 124)
        n = 4 if step == 1 else 6
        for i in range(n):
            label, stroke, fill, dashed = boxes[i]
            x = i * (bw + gap)
            s.rect(x, y0, bw, bh, fill=FILL[fill], stroke=stroke, sw=1.3,
                   rx=0 if fill == "slate" else 3, dash="4,2" if dashed else None)
            s.text(x + bw / 2, y0 + 16, label, size=9)
            if i < n - 1:
                s.line(x + bw + 1, y0 + bh / 2, x + bw + gap - 1, y0 + bh / 2, color=SLATE, arrow=True)
        xb = 4 * (bw + gap) - gap / 2
        # what a change costs on each side
        s.path(f"M2,{y0 - 8} L2,{y0 - 13} L{xb - 6},{y0 - 13} L{xb - 6},{y0 - 8}", color=GREEN, sw=1.1)
        s.text((xb - 4) / 2, y0 - 18, "rewriting a command here costs no physical work", size=9, color=GREEN)
        if step >= 2:
            s.path(f"M{xb + 6},{y0 - 8} L{xb + 6},{y0 - 13} L{W - 2},{y0 - 13} L{W - 2},{y0 - 8}", color=RED, sw=1.1)
            s.text((xb + W) / 2, y0 - 18, "changing it here takes work", size=9, color=RED)
        if step >= 3:
            s.line(xb, y0 - 6, xb, y0 + bh + 22, color=RED, sw=1.8, dash="4,2")
            s.text(xb, y0 + bh + 34, "causal boundary", size=10, color=RED, weight="bold")
            s.text(xb - 6, y0 + bh + 16, "digital", size=9, color=GRAY, anchor="end")
            s.text(xb + 6, y0 + bh + 16, "physical", size=9, color=GRAY, anchor="start")
        s.save(f"chain_step{step}")


# ---------------------------------------------------------------------------
# 2. Recovery: rollback against a new trajectory (column, 262 x 138 pt)
# ---------------------------------------------------------------------------
def recovery():
    W = 262
    for step in (1, 2):
        s = Svg(W, 138)
        def row(y, title, states, colors, arrows, note, ncol):
            s.text(0, y - 6, title, size=9.5, anchor="start", weight="bold", color=INK)
            xs = [2, 104, 206]
            for i, (lab, c) in enumerate(zip(states, colors)):
                s.rect(xs[i], y, 54, 26, fill=FILL[c], stroke={"green": GREEN, "red": RED, "slate": SLATE}[c],
                       rx=0, sw=1.3)
                s.text(xs[i] + 27, y + 17, lab, size=9.5)
            for i, a in enumerate(arrows):
                s.line(xs[i] + 55, y + 13, xs[i + 1] - 1, y + 13, color=SLATE, arrow=True)
                s.text((xs[i] + 55 + xs[i + 1]) / 2, y + 9, a, size=8.5, color=GRAY)
            s.text(W - 2, y + 40, note, size=9, color=ncol, anchor="end")
        row(16, "A database transaction", ["state S0", "state S1", "state S0"],
            ["slate", "slate", "green"], ["write", "abort"], "the same state again", GREEN)
        if step >= 2:
            row(86, "A machine that moved", ["state S0", "state S1", "state S2"],
                ["slate", "slate", "red"], ["torque", "brake"], "a new state: heat, wear", RED)
        s.save(f"recovery_step{step}")


# ---------------------------------------------------------------------------
# 3. Pre-brake delay bar, plus its scanner-path variant (full width)
# ---------------------------------------------------------------------------
def delay_bar(name, scanner=False):
    fig, ax = plt.subplots(figsize=(5.6, 1.15))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.70, bottom=0.30)
    rest = calcs.AGE - calcs.BACKBONE
    segs = [(calcs.BACKBONE, ORANGE, "backbone\n22"), (rest, LGRAY, "rest of age 29.6"),
            (calcs.LEASE, SLATE, "chunk lease 60"), (calcs.TICK + calcs.BUS, GREEN, ""),
            (calcs.ONSET, "#8C8C8C", "brake\nonset 20")]
    x = 0
    for i, (w, col, lab) in enumerate(segs):
        gone = scanner and i in (0, 2)
        ax.barh(0, w, left=x, color="white" if gone else col, edgecolor=col if gone else "white",
                hatch="////" if gone else None, height=0.62, lw=1.2)
        if lab:
            tc = INK if (col in (LGRAY,) or gone) else "white"
            ax.text(x + w / 2, 0, lab, ha="center", va="center", color=tc, fontsize=9, fontweight="bold", linespacing=0.95,
                    bbox=dict(fc="white", ec="none", pad=1.0) if gone else None)
        x += w
    tau = calcs.tau_s() * 1000
    ax.annotate("tick 1 + bus 1", xy=(calcs.AGE + calcs.LEASE + 1, -0.32),
                xytext=(calcs.AGE + calcs.LEASE + 1, -0.62), ha="center", va="top", fontsize=9,
                color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0), annotation_clip=False)
    if scanner:
        leave = calcs.BACKBONE + calcs.LEASE
        ax.text(0, 0.42, f"with a scanner stop, backbone and lease leave: {leave:.0f} ms, "
                f"{calcs.V * leave:.0f} mm at 1.5 m/s", ha="left", va="bottom", fontsize=9.5,
                color=RED, fontweight="bold")
    else:
        dd, _ = calcs.stop_mm()
        ax.text(0, 0.42, f"pre-brake delay {tau:.1f} ms = {dd:.1f} mm of travel at 1.5 m/s",
                ha="left", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlim(0, 136)
    ax.set_ylim(-0.4, 0.4)
    ax.axis("off")
    save_plot(fig, name)


# ---------------------------------------------------------------------------
# 4. Delay travel against braking travel, three steps (column)
# ---------------------------------------------------------------------------
def crossover():
    t = calcs.tau_s()
    v = np.linspace(0, 2.0, 401)
    delay, brake = v * t * 1000, v ** 2 / (2 * calcs.A_BRAKE) * 1000
    vs = 2 * calcs.A_BRAKE * t
    dd, db = calcs.stop_mm()
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(3.75, 2.25))
        fig.subplots_adjust(left=0.19, right=0.80, top=0.95, bottom=0.21)
        ax.plot(v, delay, color=AMBER, lw=2.2)
        ax.text(2.03, delay[-1], "delay\ntravel", color=AMBER, fontsize=9, fontweight="bold", va="center")
        if step >= 2:
            ax.plot(v, brake, color=SLATE, lw=2.2)
            ax.text(2.03, brake[-1], "braking\ntravel", color=SLATE, fontsize=9, fontweight="bold", va="center")
            ax.plot([vs], [vs * t * 1000], "o", ms=6, mfc="white", mec=INK, mew=1.5, zorder=5)
            ax.annotate(f"crossover {vs:.2f} m/s", xy=(vs, vs * t * 1000), xytext=(0.08, 520),
                        fontsize=9, fontweight="bold", color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
        if step >= 3:
            ax.plot(v, delay + brake, color=INK, lw=2.4)
            ax.text(2.03, delay[-1] + brake[-1], "sum", color=INK, fontsize=9, fontweight="bold",
                    va="center", clip_on=False)
            ax.axhline(calcs.D_CLEAR, color=RED, lw=1.6, ls="--")
            ax.text(0.04, calcs.D_CLEAR + 30, "clear distance 1,100 mm", color=RED, fontsize=9,
                    fontweight="bold", va="bottom")
            ax.plot([calcs.V], [dd + db], "o", ms=7, color=INK, zorder=5)
            ax.text(calcs.V - 0.05, dd + db + 40, f"{dd + db:.1f} mm", fontsize=9, fontweight="bold",
                    ha="right", va="bottom")
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 1600)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticks([0, 400, 800, 1200, 1600])
        ax.set_xlabel("speed when the worker appears (m/s)")
        ax.set_ylabel("distance (mm)")
        save_plot(fig, f"crossover_step{step}")


# ---------------------------------------------------------------------------
# 5. Tempe through the budget (full width bars, NTSB inputs)
# ---------------------------------------------------------------------------
def tempe_bars():
    v1, v2 = calcs.V_EMERG_MPH * calcs.MPH, calcs.V_SUPP_END_MPH * calcs.MPH
    supp = (calcs.T_EMERG - calcs.T_SUPP_END) * (v1 + v2) / 2
    road, _ = calcs.stopping_distance(v1, calcs.G_ROAD)
    ads, _ = calcs.stopping_distance(v1, calcs.G_ADS)
    rows = [("traveled during the 1 s suppression", supp, RED),
            ("full stop at the road's 0.92 g", road, SLATE),
            ("full stop at the ADS cap, 0.71 g", ads, GRAY)]
    fig, ax = plt.subplots(figsize=(5.6, 1.55))
    fig.subplots_adjust(left=0.43, right=0.90, top=0.97, bottom=0.27)
    for i, (lab, val, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, val, color=col, height=0.62)
        ax.text(-0.6, y, lab, ha="right", va="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(val + 0.4, y, f"{val:.1f} m", ha="left", va="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("meters, from 43.2 mph (19.3 m/s)")
    ax.spines["left"].set_visible(False)
    save_plot(fig, "tempe_bars")


# ---------------------------------------------------------------------------
# 6. Training compute by domain (Epoch AI via mlsysim), column
# ---------------------------------------------------------------------------
def epoch():
    rows = [r for r in calcs.epoch_rows() if str(r["publication_date"]) >= "2018" and r["training_flop"]]
    date = lambda r: dt.datetime.fromisoformat(str(r["publication_date"])[:10])
    lang = [r for r in rows if r["domain_group"] == "Language"]
    robo = [r for r in rows if r["domain_group"] == "Robotics" and "Cosmos" not in r["model"]]
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.15)
    ax.scatter([date(r) for r in lang], [r["training_flop"] for r in lang], s=7, color=GRAY, alpha=0.35, lw=0)
    ax.scatter([date(r) for r in robo], [r["training_flop"] for r in robo], s=26, color=GREEN, zorder=4)
    ax.text(dt.datetime(2019, 3, 1), 3e25, "language models", color=GRAY, fontsize=9, fontweight="bold")
    ax.text(dt.datetime(2018, 3, 1), 4e16, "robotics models", color=GREEN, fontsize=9, fontweight="bold")
    for name, (lab, (dx, dy)) in {"OpenVLA": ("OpenVLA", (-48, 14)), "GR00T N1 2B": ("GR00T N1", (-12, -26))}.items():
        r = next(r for r in robo if r["model"] == name)
        ax.annotate(lab, (date(r), r["training_flop"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=GREEN,
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0))
    ax.set_yscale("log")
    ax.set_ylim(1e15, 1e28)
    ax.set_yticks([1e16, 1e19, 1e22, 1e25, 1e28])
    ax.set_ylabel("training compute (FLOP)")
    ax.set_xlim(dt.datetime(2018, 1, 1), dt.datetime(2026, 6, 1))
    ax.set_xticks([dt.datetime(y, 1, 1) for y in (2018, 2020, 2022, 2024, 2026)],
                  ["2018", "2020", "2022", "2024", "2026"])
    save_plot(fig, "epoch_compute")


# ---------------------------------------------------------------------------
# 7. Scope test Venn, three steps (column, 256 x 210 drawn, placed at 0.85: 218 x 179 pt)
# ---------------------------------------------------------------------------
def venn():
    W, H = 256, 210
    L, P, D, r = (96, 82), (160, 82), (128, 136), 60
    for step in (1, 2, 3):
        s = Svg(W, H, scale=0.85)
        s.circle(*L, r, FILL["navy"], NAVY, opacity=0.8)
        s.circle(*P, r, FILL["slate"], SLATE, opacity=0.8)
        s.circle(*D, r, FILL["green"], GREEN, opacity=0.8)
        s.text(70, 62, "learned\nmodel", size=9.5, color=NAVY, weight="bold")
        s.text(186, 62, "physical\nfeedback", size=9.5, color=SLATE, weight="bold")
        s.text(128, 176, "delegated\nauthority", size=9.5, color=GREEN, weight="bold")
        if step >= 2:
            s.text(128, 50, "advisory", size=9.5, color=INK)
            s.text(90, 120, "digital\nautonomy", size=9.5, color=INK)
            s.text(166, 120, "classical\ncontrol", size=9.5, color=INK)
        if step >= 3:
            s.rect(104, 82, 48, 24, fill="white", stroke=RED, sw=1.4)
            s.text(128, 92, "physical", size=9, color=RED, weight="bold")
            s.text(128, 102.5, "AI", size=9, color=RED, weight="bold")
        s.save(f"venn_step{step}")


# ---------------------------------------------------------------------------
# 8. Tempe track: classification and predicted path, two steps (full width)
# ---------------------------------------------------------------------------
def tempe_track():
    # NTSB HAR-19/03 table (pp. 15-16): (start, end, label) in seconds before impact.
    cls = [(-5.6, -5.2, "vehicle", PETROL), (-5.2, -4.2, "other", LGRAY), (-4.2, -3.8, "vehicle", PETROL),
           (-3.8, -2.6, "vehicle, other\nalternating", None), (-2.6, -1.5, "bicycle", AMBER),
           (-1.5, -1.2, "other", LGRAY), (-1.2, 0.0, "bicycle", AMBER)]
    above = {0: -5.4, 2: -4.0, 5: -1.35}         # narrow segments: label above the bar
    changes = [-5.2, -4.2, -2.6, -1.5, -1.2]      # changed class, so the track history was dropped
    path = [(-5.6, -5.2, "none", "#E5E7EB", INK), (-5.2, -3.9, "static", "#E5E7EB", INK),
            (-3.9, -2.6, "next lane\nor static", "#E5E7EB", INK),
            (-2.6, -1.5, "static, then\nnext lane", "#E5E7EB", INK),
            (-1.5, -1.2, "", "#FDE68A", INK), (-1.2, 0.0, "in the\nSUV's lane", "#FECACA", RED)]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.75, 2.1))
        fig.subplots_adjust(left=0.13, right=0.985, top=0.89, bottom=0.23)
        for i, (a, b, lab, col) in enumerate(cls):
            if col is None:
                ax.barh(1, b - a, left=a, height=0.5, color="white", edgecolor=PETROL, hatch="//", lw=1.0)
            else:
                ax.barh(1, b - a, left=a, height=0.5, color=col, edgecolor="white")
            if i in above:
                ax.text(above[i], 1.36, lab, ha="center", va="bottom", fontsize=8.5, color=INK)
            else:
                ax.text((a + b) / 2, 1, lab, ha="center", va="center", fontsize=8.5, linespacing=0.95,
                        color="white" if col in (PETROL, AMBER) else INK, fontweight="bold",
                        bbox=dict(fc="white", ec="none", pad=0.6) if col is None else None)
        for d in changes:
            ax.plot([d, d], [0.68, 1.32], color=RED, lw=1.8, solid_capstyle="butt")
        if step >= 2:
            for a, b, lab, col, tc in path:
                ax.barh(0, b - a, left=a, height=0.5, color=col, edgecolor="white")
                if lab:
                    ax.text((a + b) / 2, 0, lab, ha="center", va="center", fontsize=8.5, color=tc,
                            fontweight="bold", linespacing=0.95)
            ax.annotate("", xy=(-1.2, -0.42), xytext=(-0.2, -0.42),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3), annotation_clip=False)
            ax.text(-1.25, -0.42, "suppression, 1 s", ha="right", va="center", fontsize=8.5, color=RED,
                    fontweight="bold")
        ax.set_yticks([0, 1], ["predicted\npath", "class"] if step >= 2 else ["", "class"])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.55, 1.62)
        ax.set_xlim(-5.6, 0.0)
        ax.set_xticks([-5, -4, -3, -2, -1, 0])
        ax.set_xlabel("seconds before impact")
        ax.spines["left"].set_visible(False)
        save_plot(fig, f"tempe_track_step{step}")


# ---------------------------------------------------------------------------
# 9. Convergence: three threats, one permission path, three demands
# ---------------------------------------------------------------------------
def convergence():
    W, H = 310, 150
    threats = [("slick floor", "the traction premise fails"), ("spoofed bus packet", "a false measurement"),
               ("hallucinated free path", "a confident wrong proposal")]
    demands = [("fresh, authenticated", "physical state"), ("authority to refuse,", "with a feasible fallback"),
               ("a synchronized record", "of what was admitted")]
    for step in (1, 2):
        s = Svg(W, H)
        for i, (a, b) in enumerate(threats):
            y = 8 + i * 48
            s.rect(0, y, 108, 36, fill=FILL["red"], stroke=RED, rx=0)
            s.text(54, y + 15, a, size=9, color=RED, weight="bold")
            s.text(54, y + 27, b, size=8.5, color=INK)
            s.line(109, y + 18, 124, 75 + (i - 1) * 10, color=SLATE, sw=1.0, arrow=True)
        s.rect(126, 50, 58, 50, fill=FILL["green"], stroke=GREEN, sw=1.5)
        s.text(155, 71, "permission\npath", size=9, color=GREEN, weight="bold")
        if step >= 2:
            for i, (a, b) in enumerate(demands):
                y = 8 + i * 48
                s.line(185, 75 + (i - 1) * 10, 200, y + 18, color=GREEN, sw=1.0, arrow=True)
                s.rect(202, y, 108, 36, fill=FILL["green"], stroke=GREEN)
                s.text(256, y + 15, a, size=9, color=GREEN, weight="bold")
                s.text(256, y + 27, b, size=8.5, color=INK)
        s.save(f"convergence_step{step}")


# ---------------------------------------------------------------------------
# 10. The record chain (full width, 404 x 172 pt), chapter figure 9 redrawn
# ---------------------------------------------------------------------------
def record_chain():
    W, H = 404, 172
    bw, bh, cw = 48, 32, 50.5
    yA, yB, yC = 6, 56, 124
    off = (cw - bw) / 2
    s = Svg(W, H)
    bands = [(2 * cw - 1, yA - 4, 3 * cw + 2, bh + 8, "petrol"),
             (0, yB - 4, 2 * cw - 2, bh + 8, "slate"),
             (2 * cw - 1, yB - 4, 6 * cw + 1 - 2 * cw, bh + 8, "navy"),
             (3 * cw - 1, yC - 4, 5 * cw + 1, bh + 8, "green")]
    for x, y, w, h, f in bands:
        s.rect(x, y, w, h, fill=FILL[f], stroke="none", sw=0, rx=2)
    def box(col, y, num, name, color=SLATE):
        x = col * cw + off
        s.rect(x, y, bw, bh, fill="white", stroke=color, sw=1.1)
        s.text(x + bw / 2, y + 11, name, size=8, color=INK)
        s.text(x + bw - 3, y + bh - 3, f"L{num}", size=7, color=GRAY, anchor="end")
    box(0, yB, 2, "limit\nrecord")
    box(1, yB, 4, "handoff\nrecord")
    for i, (n, nm) in enumerate([(5, "provenance\nrecord"), (6, "policy\nmanifest"), (7, "evaluation\nrecord")]):
        box(2 + i, yA, n, nm, PETROL)
    for i, (n, nm) in enumerate([(8, "observation\ncontract"), (9, "belief\nrecord"), (10, "intent\nlease"),
                                 (11, "trajectory\nrecord")]):
        box(2 + i, yB, n, nm, NAVY)
    box(6, yB, 12, "enforcement\nrecord", NAVY)
    box(7, yB, 13, "placement\nrecord", NAVY)
    for i, (n, nm) in enumerate([(14, "authority\nrecord"), (15, "fault\nrecords"), (16, "release\nmanifest"),
                                 (17, "approval\nrecord")]):
        box(3 + i, yC, n, nm, GREEN)
    box(7, yC, 18, "residual-\nclaims", RED)
    def right(col, y):
        return col * cw + off + bw, y + bh / 2
    def left(col):
        return col * cw + off
    for row, cols in ((yB, (0, 1, 2, 3, 4, 5, 6)), (yA, (2, 3)), (yC, (3, 4, 5, 6))):
        for c in cols:
            if row == yB and c == 1:
                continue
            x1, y1 = right(c, row)
            s.line(x1, y1, left(c + 1) - 1, y1, arrow=True, sw=1.0)
    # the handoff record forks into Part II and Part III
    hx, hy = right(1, yB)
    s.line(hx, hy, left(2) - 1, hy, arrow=True, sw=1.0)
    fx = 1 * cw + cw / 2
    s.path(f"M{fx},{yB} L{fx},{yA + bh / 2} L{left(2) - 1},{yA + bh / 2}", sw=1.0, arrow=True)
    # evaluation joins enforcement
    ex, ey = right(4, yA)
    s.path(f"M{ex},{ey} L{6 * cw + cw / 2},{ey} L{6 * cw + cw / 2},{yB - 1}", sw=1.0, arrow=True)
    # proposer limits (Lecture 3) enter the handoff record from below
    s.line(1 * cw + cw / 2, yB + bh + 13, 1 * cw + cw / 2, yB + bh + 1, color=GRAY, sw=1.0, dash="2,2", arrow=True)
    s.text(1 * cw + cw / 2, yB + bh + 22, "L3: proposer limits", size=7.5, color=GRAY)
    # placement down to the authority record
    px, py = 7 * cw + cw / 2, yB + bh
    s.path(f"M{px},{py} L{px},{yC - 11} L{3 * cw + cw / 2},{yC - 11} L{3 * cw + cw / 2},{yC - 1}",
           sw=1.0, arrow=True)
    s.text(5 * cw + 4, yA + 9, "Part II: endogenous data", size=8, color=PETROL, anchor="start", weight="bold")
    s.text(0, yB - 17, "Part I:", size=8, color=SLATE, anchor="start", weight="bold")
    s.text(0, yB - 7.5, "irreversibility", size=8, color=SLATE, anchor="start", weight="bold")
    s.text(6 * cw - 2, yB + bh + 12, "Part III: proposal is not permission", size=8, color=NAVY,
           anchor="end", weight="bold")
    s.text(3 * cw + 1, yC + bh + 12, "Part IV: permission must be earned", size=8, color=GREEN,
           anchor="start", weight="bold")
    s.save("record_chain")


def main():
    chain()
    recovery()
    delay_bar("delay_bar")
    delay_bar("delay_bar_scanner", scanner=True)
    crossover()
    tempe_bars()
    epoch()
    venn()
    tempe_track()
    convergence()
    record_chain()
    print("ok")


if __name__ == "__main__":
    main()
