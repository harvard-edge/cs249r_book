#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 10 (Grounded Intent), Volume IV.

Every derived value comes from calcs.py (inputs from the rendered chapter,
Grounded Intent, and the MLSysIM registry the chapter prints from). Values
read from papers carry their table or figure in the comment beside them.
Each figure is drawn at the size of its slot (full width about 5.8 in wide;
a column about 5.4 by 3.6 in) so its type stays legible on the slide.
A figure revealed in steps is written as name_step1.svg ... name_stepN.svg
with every element drawn in every step (hidden ones transparent), so the
canvas and the axes never move between steps.

Colors follow the Volume IV causal-boundary palette (volumes/vol4/palette.md):
navy for cognition, dashed blue for unprivileged proposals, green for the
deterministic permission side, red only for vetoes, hazards, and limits,
petrol for sensing, slate for the physical plant.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402,F401  (sets the shared rcParams)

import calcs  # noqa: E402

plt = sp.plt
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch, Rectangle, Wedge)

V = calcs.VALUES
OUT = Path(__file__).parent

P = {
    "navy": "#1F407A", "navy_l": "#EFF6FF",
    "blue": "#2563EB", "blue_l": "#DBEAFE",
    "green": "#059669", "green_l": "#DCFCE7", "green_d": "#047857",
    "red": "#DC2626", "red_l": "#FEE2E2", "red_d": "#991B1B",
    "petrol": "#0D9488", "petrol_l": "#CCFBF1",
    "slate": "#334155", "slate_l": "#F1F5F9",
    "amber": "#D97706", "amber_l": "#FEF3C7",
    "gray": "#64748B", "gray_l": "#E2E8F0",
    "text": "#1A1A2E",
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", format="svg", facecolor="white")
    plt.close(fig)


def a(shown):
    return 1.0 if shown else 0.0


def box(ax, x, y, w, h, text, fc, ec, shown=True, fs=13, tc=None, dashed=False,
        bold=True, rad=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                                fc=fc, ec=ec, lw=1.8, ls="--" if dashed else "-",
                                alpha=a(shown)))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or P["text"], fontweight="bold" if bold else "normal",
            alpha=a(shown))


def arrow(ax, x0, y0, x1, y1, color, shown=True, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw, ls=ls,
                                 alpha=a(shown)))


def canvas(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Hook: AutoRT task filter (Ahn et al. 2024, quoted in War Story 10.1)
# ---------------------------------------------------------------------------
def autort():
    for step in (1, 2):
        fig = plt.figure(figsize=(5.8, 2.4))
        ax1 = fig.add_axes([0.2, 0.62, 0.77, 0.3])
        ax2 = fig.add_axes([0.2, 0.2, 0.77, 0.3])
        acc = calcs.AUTORT_ACCEPTABLE
        uns = calcs.AUTORT_PROPOSED - acc
        ax1.barh(0, acc, color=P["gray"], height=0.7)
        ax1.barh(0, uns, left=acc, color=P["amber"], height=0.7)
        ax1.text(acc / 2, 0, f"{acc} acceptable", ha="center", va="center", fontsize=12.5,
                 color="white", fontweight="bold")
        ax1.text(acc + uns / 2, 0.62, f"{uns} unsuitable", ha="center", va="bottom",
                 fontsize=12.5, color=P["amber"], fontweight="bold")
        ax1.set_xlim(0, 270)
        ax1.set_xticks([0, 100, 200, 259])
        ax1.text(-6, 0, f"{calcs.AUTORT_PROPOSED}\nproposed", ha="right", va="center",
                 fontsize=12.5, fontweight="bold", color=P["text"])
        s2 = step >= 2
        rej = calcs.AUTORT_REJECTED_UNSUITABLE
        ax2.barh(0, rej, color=P["green"], height=0.7, alpha=a(s2))
        ax2.barh(0, uns - rej, left=rej, color=P["red"], height=0.7, alpha=a(s2))
        ax2.text(rej / 2, 0, f"{rej} rejected", ha="center", va="center", fontsize=12.5,
                 color="white", fontweight="bold", alpha=a(s2))
        ax2.text(rej + (uns - rej) / 2, 0, f"{uns - rej} passed", ha="center", va="center",
                 fontsize=12.5, color="white", fontweight="bold", alpha=a(s2))
        ax2.set_xlim(0, uns)
        ax2.set_xticks([0, rej, uns])
        ax2.text(-0.7, 0, f"{uns}\nunsuitable", ha="right", va="center", fontsize=12.5,
                 fontweight="bold", color=P["text"], alpha=a(s2))
        ax2.set_xlabel("Tasks in the AutoRT sample (human-judged)", fontsize=11.5)
        for ax in (ax1, ax2):
            ax.set_ylim(-0.5, 0.9)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        for t in ax2.get_xticklabels():
            t.set_alpha(a(s2))
        ax2.spines["bottom"].set_alpha(a(s2))
        save(fig, f"autort_step{step}")


# ---------------------------------------------------------------------------
# Grounded affordance on a mug (Sec. 10.1, Definition 10.1)
# ---------------------------------------------------------------------------
def mug_affordance():
    for step in (1, 2, 3):
        fig, ax = canvas(5.8, 3.4, (0, 11.6), (0, 6.8))
        # mug body (slate, the physical object) and handle
        ax.add_patch(Rectangle((1.2, 0.8), 3.4, 4.2, fc=P["slate_l"], ec=P["slate"], lw=2.2))
        ax.add_patch(Ellipse((2.9, 5.0), 3.4, 0.7, fc="white", ec=P["slate"], lw=2.2))
        ax.add_patch(Wedge((4.6, 2.9), 1.15, -80, 80, width=0.38, fc=P["slate_l"],
                           ec=P["slate"], lw=2.2))
        # affordance sites: rim, wall, handle
        sites = [((2.9, 5.0), "rim", (0.3, 6.05)), ((1.2, 2.2), "wall", (0.05, 0.25)),
                 ((5.55, 2.9), "handle", (5.9, 4.35))]
        for (x, y), name, (tx, ty) in sites:
            ax.add_patch(Circle((x, y), 0.16, fc=P["navy"], ec="white", lw=1.2))
            ax.text(tx, ty, name, fontsize=14, color=P["navy"], fontweight="bold")
        # step 2: handle frame, approach direction, aperture
        s2 = step >= 2
        arrow(ax, 8.6, 2.9, 5.95, 2.9, P["blue"], shown=s2, lw=2.6)
        ax.text(7.1, 3.25, "approach", fontsize=13, color=P["blue"], ha="center",
                fontweight="bold", alpha=a(s2))
        ax.plot([6.15, 6.15], [2.35, 3.45], color=P["navy"], lw=2.5, alpha=a(s2))
        ax.text(6.35, 1.95, "aperture", fontsize=13, color=P["navy"], fontweight="bold",
                alpha=a(s2))
        # step 3: wrench ceiling and evidence epoch
        s3 = step >= 3
        box(ax, 8.0, 4.7, 3.4, 1.6, "wrench ceiling\n(force, torque)", P["red_l"], P["red"],
            shown=s3, fs=12, tc=P["red_d"])
        box(ax, 8.0, 0.4, 3.4, 1.6, "evidence epoch\n(capture time)", P["petrol_l"],
            P["petrol"], shown=s3, fs=12, tc=P["slate"])
        save(fig, f"mug_affordance_step{step}")


# ---------------------------------------------------------------------------
# Goal set is a specification (Sec. 10.1, equation 1): one slice of the
# ellipsoid with all semi-axes at 15 mm; offsets of 10 and 20 mm.
# ---------------------------------------------------------------------------
def goalset():
    r = V["r_tol"]
    for step in (1, 2):
        fig = plt.figure(figsize=(5.4, 3.6))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.set_aspect("equal")
        ax.set_xlim(-24, 34)
        ax.set_ylim(-19, 19)
        ax.axis("off")
        ax.add_patch(Circle((0, 0), r, fc=P["green_l"], ec=P["green"], lw=2.5))
        ax.text(0, r + 1.0, "goal set: 15 mm (task profile)", ha="center", va="bottom",
                fontsize=14, color=P["green_d"], fontweight="bold")
        ax.plot([0], [0], "+", color=P["slate"], ms=16, mew=2.5)
        ax.text(-1, -2.4, "target", ha="right", va="top", fontsize=13, color=P["slate"])
        ax.plot([V["norm_in"] * r], [0], "o", ms=12, color=P["navy"])
        ax.text(V["norm_in"] * r, -2.6, f"10 mm\n{V['norm_in']:.2f}", ha="center",
                va="top", fontsize=13, color=P["navy"], fontweight="bold")
        s2 = step >= 2
        ax.plot([V["norm_out"] * r], [0], "o", ms=12, color=P["red"], alpha=a(s2))
        ax.text(V["norm_out"] * r, 2.0, f"20 mm\n{V['norm_out']:.2f}", ha="center",
                va="bottom", fontsize=13, color=P["red_d"], fontweight="bold", alpha=a(s2))
        ax.text(V["norm_out"] * r + 2.5, -1.5, "outside:\nrefuse", ha="left", va="top",
                fontsize=13, color=P["red_d"], alpha=a(s2))
        save(fig, f"goalset_step{step}")


# ---------------------------------------------------------------------------
# Admission pipeline across two processors (Sec. 10.3.1 figure 1; Sec. 10.4)
# ---------------------------------------------------------------------------
def ap_mcu():
    for step in (1, 2, 3):
        fig, ax = canvas(5.9, 2.9, (0, 11.8), (0, 5.8))
        # domains
        ax.text(3.0, 5.45, "Application processor: proposes", ha="center",
                fontsize=11.5, color=P["navy"], fontweight="bold")
        ax.text(9.2, 5.45, "Safety MCU: commands", ha="center", fontsize=11.5,
                color=P["green_d"], fontweight="bold")
        ax.plot([6.15, 6.15], [0.2, 5.2], color=P["slate"], lw=1.6, ls=(0, (6, 4)))
        box(ax, 0.2, 3.3, 2.4, 1.3, "intent model\n336 ms, 2 Hz", P["navy_l"], P["navy"],
            fs=11.5)
        box(ax, 3.0, 3.3, 2.7, 1.3, "unique\nreferent?", P["navy_l"], P["navy"], fs=11.5)
        arrow(ax, 2.6, 3.95, 3.0, 3.95, P["navy"])
        box(ax, 3.0, 0.5, 2.7, 1.2, "refuse:\nambiguous", P["red_l"], P["red"], fs=11.5,
            tc=P["red_d"])
        arrow(ax, 4.35, 3.3, 4.35, 1.7, P["red"])
        # step 2: board link and mailbox
        s2 = step >= 2
        arrow(ax, 5.7, 3.95, 6.6, 3.95, P["blue"], shown=s2, ls="--")
        ax.text(7.6, 2.95, "board link:\nCRC, sequence", ha="center", va="top",
                fontsize=10.5, color=P["slate"], alpha=a(s2))
        box(ax, 6.6, 3.3, 2.0, 1.3, "mailbox\n(SRAM)", P["slate_l"], P["slate"],
            shown=s2, fs=11.5, dashed=True)
        # step 3: permission path, lease timer, drives
        s3 = step >= 3
        box(ax, 9.0, 3.3, 2.6, 1.3, "permission\npath, 1 kHz", P["green_l"], P["green"],
            shown=s3, fs=11.5)
        arrow(ax, 8.6, 3.95, 9.0, 3.95, P["green"], shown=s3)
        box(ax, 9.0, 0.5, 2.6, 1.2, "joint drives", "white", P["slate"], shown=s3,
            fs=11.5, rad=0.0)
        arrow(ax, 10.3, 3.3, 10.3, 1.7, P["green"], shown=s3)
        box(ax, 6.6, 0.5, 2.0, 1.2, "lease\ntimer", P["red_l"], P["red"], shown=s3, fs=11.5,
            tc=P["red_d"])
        arrow(ax, 8.6, 1.1, 9.0, 1.1, P["red"], shown=s3, ls="--")
        save(fig, f"ap_mcu_step{step}")


# ---------------------------------------------------------------------------
# Conformal refusal (Sec. 10.3.1): normalized scores against 1 - qhat
# ---------------------------------------------------------------------------
def conformal():
    cases = [("Jar beside mug\n(raw 0.62, 0.48)", V["p_mug"], V["p_jar"], "set size 2: refuse"),
             ("Jar scores 0.10\n(raw 0.62, 0.10)", V["p_mug_low"], V["p_jar_low"],
              "set size 1: proceed")]
    for step in (1, 2):
        fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), sharey=True)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.3, top=0.8, wspace=0.18)
        for i, (ax, (title, pm, pj, verdict)) in enumerate(zip(axes, cases)):
            shown = i < step
            for x, (name, p) in enumerate((("mug", pm), ("jar", pj))):
                keep = p >= V["thresh"]
                c = P["navy"] if keep else P["gray_l"]
                ax.bar(x, p, color=c, width=0.6, alpha=a(shown))
                ax.text(x, p + 0.03, f"{p:.2f}", ha="center", fontsize=13,
                        fontweight="bold", color=P["text"], alpha=a(shown))
            ax.axhline(V["thresh"], color=P["red"], ls="--", lw=1.6, alpha=a(shown))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["mug", "jar"], fontsize=13)
            ax.set_ylim(0, 1.1)
            ax.set_title(title, fontsize=12, alpha=a(shown))
            ax.text(0.5, -0.32, verdict, transform=ax.transAxes, ha="center", va="top",
                    fontsize=12.5, fontweight="bold",
                    color=P["red_d"] if "refuse" in verdict else P["green_d"],
                    alpha=a(shown))
            for t in ax.get_xticklabels():
                t.set_alpha(a(shown))
        axes[0].set_ylabel("normalized score", fontsize=12)
        axes[0].text(0.5, V["thresh"] + 0.03, "0.40", color=P["red"], fontsize=11,
                     fontweight="bold", ha="center")
        save(fig, f"conformal_step{step}")


# ---------------------------------------------------------------------------
# Imprecise estimate vs wrong object (Sec. 10.3.3)
# ---------------------------------------------------------------------------
def imprecise():
    sx, sy, sz = V["semi"]
    r = V["r_tol"]
    fig = plt.figure(figsize=(3.0, 2.8))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_aspect("equal")
    ax.set_xlim(-46, 40)
    ax.set_ylim(-40, 40)
    ax.axis("off")
    ax.add_patch(Ellipse((0, 0), 2 * sx, 2 * sz, fc=P["blue_l"], ec=P["blue"], lw=2.2))
    ax.add_patch(Rectangle((-r, -r), 2 * r, 2 * r, fc="none", ec=P["green"], lw=2.4,
                           ls="--"))
    ax.plot([0], [0], "o", color=P["navy"], ms=7)
    ax.annotate("", xy=(sx + 9, sz), xytext=(sx + 9, 0),
                arrowprops=dict(arrowstyle="<|-|>", color=P["red"], lw=2))
    ax.text(sx + 11, sz / 2, f"{sz:.1f}\nmm", fontsize=14, color=P["red_d"],
            fontweight="bold", va="center")
    ax.annotate("", xy=(-sx, -r - 7), xytext=(sx * 0, -r - 7),
                arrowprops=dict(arrowstyle="<|-|>", color=P["navy"], lw=1.6))
    ax.text(-sx - 2, -r - 7, f"{sx:.1f} mm", fontsize=13, color=P["navy"], ha="right",
            va="center", fontweight="bold")
    ax.text(r + 2, -r + 1, "15 mm\ntolerance", fontsize=12.5, color=P["green_d"],
            fontweight="bold", ha="left", va="bottom")
    save(fig, "imprecise")


def wrong_object():
    fig, ax = canvas(4.6, 2.0, (0, 10.8), (0, 4.7))
    labels = [("confident\nwrong object", P["amber"]), ("passes every\ngeometric check", P["slate"]),
              ("mug-rated\nsqueeze crushes\nthe jar", P["red"])]
    xs = [1.5, 5.4, 9.2]
    for (t, c), x in zip(labels, xs):
        ax.add_patch(Circle((x, 3.3), 0.42, fc=c, ec="white"))
        ax.text(x, 2.6, t, ha="center", va="top", fontsize=12.5, color=P["text"],
                fontweight="bold")
    for x0, x1 in ((1.9, 4.9), (5.9, 8.9)):
        arrow(ax, x0, 3.3, x1, 3.3, P["slate"])
    save(fig, "wrong_object")


# ---------------------------------------------------------------------------
# Latch ceilings (Napkin Math 10.1)
# ---------------------------------------------------------------------------
def latch():
    rows = [("tripwire", V["f_trip"], P["amber"]), ("at brake", V["f_brake"], P["amber"]),
            ("200 ms stall,\nno local trip", V["f_stall"], P["red"])]
    for step in (1, 2, 3):
        fig = plt.figure(figsize=(5.8, 2.4))
        ax = fig.add_axes([0.25, 0.22, 0.72, 0.72])
        n = len(rows)
        for i, (lab, v, c) in enumerate(rows):
            y = n - 1 - i
            shown = i < step
            ax.barh(y, v, color=c, height=0.6, alpha=a(shown))
            ax.text(v * 1.12, y, f"{v:,.0f} N", va="center", fontsize=13, color=c,
                    fontweight="bold", alpha=a(shown))
            ax.text(8.5, y, lab, ha="right", va="center", fontsize=13,
                    fontweight="bold", color=P["text"], alpha=a(shown))
        ax.axvline(V["f_limit"], color=P["red"], ls="--", lw=1.8)
        ax.text(V["f_limit"] * 1.06, n - 0.42, "100 N latch limit", color=P["red_d"],
                fontsize=11.5, fontweight="bold", va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(10, 2e4)
        ax.set_ylim(-0.6, n - 0.1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Contact force on the latch (N, log scale)", fontsize=12)
        save(fig, f"latch_step{step}")


# ---------------------------------------------------------------------------
# Cadence gap: target error and leases on one clock (Napkin Math 10.2,
# Sec. 10.5 figure 5)
# ---------------------------------------------------------------------------
def cadence():
    t = np.linspace(0, 560, 600)
    e_u = V["e0"] + V["v_belt"] * t / 1000
    e_t = V["e0"] + 0.5 * V["a_slip"] * (t / 1000) ** 2
    for step in (1, 2, 3):
        fig = plt.figure(figsize=(5.8, 3.2))
        ax1 = fig.add_axes([0.13, 0.52, 0.84, 0.44])
        ax2 = fig.add_axes([0.13, 0.13, 0.84, 0.33], sharex=ax1)
        ax1.plot(t, e_u, color=P["amber"], lw=2.5, ls="--")
        ax1.plot(t, e_t, color=P["petrol"], lw=2.5, alpha=a(step >= 3))
        ax1.axhline(V["r_tol"], color=P["red"], lw=1.6, ls="--")
        ax1.text(565, V["r_tol"] + 1, "15 mm", color=P["red_d"], fontsize=11.5,
                 ha="right", va="bottom", fontweight="bold")
        ax1.text(5, 22, f"untracked\n{V['tau_u']:.0f} ms", color=P["amber"],
                 fontsize=12, fontweight="bold")
        ax1.text(V["tau_t"] + 12, 6.5, f"tracked: {V['tau_t']:.0f} ms", color=P["petrol"],
                 fontsize=12, fontweight="bold", alpha=a(step >= 3))
        ax1.set_ylim(0, 32)
        ax1.set_ylabel("target error\n(mm)", fontsize=11)
        ax1.set_yticks([0, 15, 30])
        plt.setp(ax1.get_xticklabels(), visible=False)
        # lanes
        s2, s3 = step >= 2, step >= 3
        ax2.barh(2, V["t_infer"], color=P["navy"], height=0.6, alpha=a(s2))
        ax2.text(V["t_infer"] / 2, 2, f"inference {V['t_infer']:.0f} ms", va="center",
                 ha="center", fontsize=11.5, color="white", fontweight="bold", alpha=a(s2))
        ax2.plot([V["t_period"]], [2], "v", color=P["navy"], ms=10, alpha=a(s2))
        ax2.text(V["t_period"] + 14, 2, "next\nproposal", fontsize=10.5, color=P["navy"],
                 ha="left", va="center", alpha=a(s2))
        for y, tau, c in ((1, V["tau_u"], P["amber"]), (0, V["tau_t"], P["petrol"])):
            ax2.barh(y, tau, color=c, height=0.6, alpha=a(s3))
            ax2.barh(y, V["onset"], left=tau, color="white", edgecolor=P["red"],
                     hatch="///", height=0.6, alpha=a(s3))
            ax2.text(tau + V["onset"] + 8, y, "+22 ms stop onset", va="center",
                     fontsize=10.5, color=P["red_d"], alpha=a(s3))
        ax2.set_yticks([2, 1, 0])
        ax2.set_yticklabels(["model", "lease,\nuntracked", "lease,\ntracked"], fontsize=10.5)
        ax2.set_ylim(-0.5, 2.9)
        ax2.set_xlim(0, 565)
        ax2.set_xlabel("time since capture (ms)", fontsize=11.5)
        for lab in ax2.get_yticklabels()[1:]:
            lab.set_alpha(a(s3))
        lab0 = ax2.get_yticklabels()[0]
        lab0.set_alpha(a(s2))
        save(fig, f"cadence_step{step}")


# ---------------------------------------------------------------------------
# Dynamic reachability (Sec. 10.6, Definition 10.2)
# ---------------------------------------------------------------------------
def transit():
    for step in (1, 2):
        fig = plt.figure(figsize=(5.8, 2.0))
        ax = fig.add_axes([0.22, 0.3, 0.75, 0.66])
        ax.barh(1, V["t_transit"], color=P["navy"], height=0.55)
        ax.text(V["t_transit"] + 6, 1, f"{V['t_transit']:.0f} ms at 1 m/s", va="center",
                fontsize=12.5, color=P["navy"], fontweight="bold")
        ax.axvline(V["tau_t"], color=P["red"], ls="--", lw=1.8)
        ax.text(V["tau_t"] - 5, 1.55, f"evidence {V['tau_t']:.0f} ms", color=P["red_d"],
                fontsize=11.5, ha="right", va="center", fontweight="bold")
        s2 = step >= 2
        half = V["t_transit"] / 2
        for k in range(2):
            ax.barh(0, half - 6, left=k * half, color=P["green"], height=0.55, alpha=a(s2))
            ax.text(k * half + half / 2, 0, f"capture {k + 1}", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold", alpha=a(s2))
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["one terminal\nmove", "two guarded\nsegments"], fontsize=12)
        ax.get_yticklabels()[1].set_alpha(a(s2))
        ax.set_xlim(0, 470)
        ax.set_ylim(-0.5, 1.8)
        ax.set_xlabel("time (ms)", fontsize=11.5)
        save(fig, f"transit_step{step}")


# ---------------------------------------------------------------------------
# Forward occupancy on the aisle (Napkin Math 10.3, Sec. 10.6.1 figure 7)
# ---------------------------------------------------------------------------
def aisle():
    w = V["aisle_w"] * 1000  # mm
    for step in (1, 2, 3):
        fig = plt.figure(figsize=(5.8, 2.6))
        ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
        ax.set_aspect("equal")
        ax.set_xlim(-200, 6200)
        ax.set_ylim(-700, w + 700)
        ax.axis("off")
        ax.add_patch(Rectangle((-200, w), 6400, 420, fc=P["gray_l"], ec=P["slate"], lw=1.4))
        ax.add_patch(Rectangle((-200, -420), 2600, 420, fc=P["gray_l"], ec=P["slate"], lw=1.4))
        ax.add_patch(Rectangle((3300, -420), 2900, 420, fc=P["gray_l"], ec=P["slate"], lw=1.4))

        ax.text(2850, -560, "cross aisle", ha="center", va="top", fontsize=11.5,
                color=P["slate"])
        s3 = step >= 3
        ax.add_patch(Rectangle((-200, 0), 6400, w, fc=P["red_l"], ec="none", alpha=0.6 * a(s3)))
        ax.text(3000, w + 210, f"2 s set: {V['d_long']:.1f} m spans the {V['aisle_w']:.2f} m aisle",
                ha="center", va="center", fontsize=11.5, color=P["red_d"], fontweight="bold",
                alpha=a(s3))
        # robot path
        y0 = w * 0.55
        ax.add_patch(Rectangle((150, y0 - 220), 440, 440, fc=P["slate"], ec=P["slate"]))
        ax.text(370, y0 - 330, "robot", ha="center", va="top", fontsize=11.5, color=P["slate"])
        ax.plot([590, 5600], [y0, y0], color=P["navy"], lw=2.5)
        ax.plot([5600], [y0], "*", ms=18, color=P["navy"])
        ax.text(5600, y0 - 170, "goal", ha="center", va="top", fontsize=11.5, color=P["navy"])
        # coworker
        cx, cy = 2850, w * 0.55 - 260
        ax.plot([cx], [cy], "o", ms=11, color=P["red"])
        ax.text(cx, cy - V["d_t"] - 40, "coworker", fontsize=11.5, color=P["red_d"], va="top",
                ha="center")
        ax.add_patch(Circle((cx, cy), V["d_u"], fc="none", ec=P["red"], lw=1.8, ls="--"))
        s2 = step >= 2
        ax.add_patch(Circle((cx, cy), V["d_t"], fc="none", ec=P["red"], lw=2.2, ls="--",
                            alpha=a(s2)))
        dx = np.sqrt(max(V["d_t"] ** 2 - (y0 - cy) ** 2, 0))
        ax.plot([cx - dx, cx + dx], [y0, y0], color=P["red"], lw=5, alpha=a(s2))
        ax.text(4650, -210, f"{V['d_u']:.0f} mm at {V['tau_u']:.0f} ms\n"
                f"{V['d_t']:.0f} mm at {V['tau_t']:.0f} ms",
                ha="center", va="center", fontsize=10.5, color=P["red_d"], fontweight="bold",
                alpha=a(s2))
        save(fig, f"aisle_step{step}")


# ---------------------------------------------------------------------------
# Lease expiry: three countdowns on the MCU clock (equation 9). Evidence
# terms are the chapter's 60 and 219 ms; the administrative and stopping
# terms are drawn at illustrative positions (the chapter gives no value).
# ---------------------------------------------------------------------------
def lease_terms():
    rows = [("evidence\nt0 + horizon", V["tau_t"], P["petrol"], f"{V['tau_t']:.0f} ms tracked"),
            ("host silence\nadmission + lease", 300, P["navy"], "requested, clamped"),
            ("stopping\ndeadline", 170, P["red"], "from clearance left")]
    for step in (1, 2, 3, 4):
        fig = plt.figure(figsize=(5.8, 2.5))
        ax = fig.add_axes([0.27, 0.22, 0.7, 0.74])
        n = len(rows)
        for i, (lab, v, c, txt) in enumerate(rows):
            y = n - 1 - i
            shown = i < step
            ax.barh(y, v, color=c, height=0.55, alpha=0.9 * a(shown))
            ax.text(v + 6, y, txt, va="center", fontsize=11.5, color=c, fontweight="bold",
                    alpha=a(shown))
            ax.text(-8, y, lab, ha="right", va="center", fontsize=11.5, color=P["text"],
                    fontweight="bold", alpha=a(shown))
        s4 = step >= 4
        ax.axvline(170, color=P["red"], lw=2.2, alpha=a(s4))
        ax.text(176, n - 0.45, "expiry = earliest", ha="left", va="center", fontsize=11.5,
                color=P["red_d"], fontweight="bold", alpha=a(s4))
        ax.set_xlim(0, 480)
        ax.set_ylim(-0.6, n - 0.2)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("MCU monotonic clock (illustrative positions)", fontsize=11)
        save(fig, f"lease_terms_step{step}")


# ---------------------------------------------------------------------------
# Lease lifecycle (Sec. 10.8 figure 8)
# ---------------------------------------------------------------------------
def lifecycle():
    for step in (1, 2):
        fig, ax = canvas(5.9, 3.2, (0, 11.8), (0, 6.4))
        box(ax, 0.2, 3.6, 1.9, 1.0, "Pending", P["navy_l"], P["navy"], fs=13.5)
        ax.add_patch(FancyBboxPatch((2.8, 2.6), 5.6, 3.1,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    fc=P["green_l"], ec=P["green"], lw=1.8))
        ax.text(5.6, 5.85, "lease held, MCU timer armed", ha="center", fontsize=11.5,
                color=P["green_d"], fontweight="bold")
        box(ax, 3.2, 3.2, 1.9, 1.0, "Active", "white", P["green"], fs=13.5)
        box(ax, 6.1, 3.2, 1.9, 1.0, "Renewed", "white", P["green"], fs=13.5)
        arrow(ax, 2.1, 4.1, 3.2, 3.9, P["green"])
        arrow(ax, 5.1, 3.7, 6.1, 3.7, P["green"])
        ax.text(5.6, 2.75, "newer capture", ha="center", fontsize=12, color=P["green_d"])
        arrow(ax, 1.15, 3.6, 1.15, 2.0, P["slate"], ls="--")
        ax.text(1.15, 1.85, "refused:\nno authority", fontsize=12, color=P["slate"],
                ha="center", va="top")
        box(ax, 8.9, 4.3, 2.8, 1.2, "goal\ncomplete", "white", P["slate"], fs=12.5)
        arrow(ax, 8.4, 4.9, 8.9, 4.9, P["slate"])
        box(ax, 3.2, 0.3, 1.9, 1.0, "Expired", P["red_l"], P["red"], fs=13, tc=P["red_d"])
        box(ax, 6.1, 0.3, 1.9, 1.0, "Aborted", P["red_l"], P["red"], fs=13, tc=P["red_d"])
        box(ax, 9.2, 0.3, 2.4, 1.0, "fallback\n(MCU picks)", "white", P["red"], fs=12.5,
            tc=P["red_d"])
        arrow(ax, 4.15, 2.6, 4.15, 1.3, P["red"])
        arrow(ax, 7.05, 2.6, 7.05, 1.3, P["red"])
        ax.text(4.35, 1.75, "timer\nexpires", ha="left", fontsize=12,
                color=P["red_d"])
        ax.text(7.25, 1.75, "contact limit or\nevidence lost", ha="left", fontsize=12,
                color=P["red_d"])
        arrow(ax, 8.0, 0.8, 9.2, 0.8, P["red"])
        s2 = step >= 2
        ax.add_patch(FancyArrowPatch((3.55, 4.2), (4.75, 4.2), connectionstyle="arc3,rad=-1.0",
                                     arrowstyle="-|>", mutation_scale=14, color=P["red"],
                                     lw=2, alpha=a(s2)))
        ax.text(5.0, 4.95, "same capture:\nrejected, timer runs", ha="center", fontsize=11.5,
                color=P["red_d"], fontweight="bold", alpha=a(s2))
        save(fig, f"lifecycle_step{step}")


# ---------------------------------------------------------------------------
# OK-Robot measured outcomes (Liu et al. 2024, Fig. 4 counts, 375 trials)
# ---------------------------------------------------------------------------
def okrobot_fail():
    o = V["okr"]
    rows = [("succeeded", o["success"], P["green"]),
            ("wrong or no\nobject found", o["nav_wrong_object"], P["red"]),
            ("grasp model\nfailed", o["grasp_model"], P["amber"]),
            ("hardware", o["hardware"], P["slate"]),
            ("other", o["nav_other"] + o["placing"], P["gray"])]
    fig = plt.figure(figsize=(5.4, 3.6))
    ax = fig.add_axes([0.3, 0.17, 0.66, 0.8])
    n = len(rows)
    for i, (lab, v, c) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=c, height=0.62)
        ax.text(v + 5, y, f"{v}", va="center", fontsize=15, color=c, fontweight="bold")
        ax.text(-8, y, lab, ha="right", va="center", fontsize=14, color=P["text"])
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.set_xlabel(f"Trials of {o['trials']} in real homes", fontsize=14)
    save(fig, "okrobot_fail")


# ---------------------------------------------------------------------------
# Published action-model inference periods against the mug's horizons
# ---------------------------------------------------------------------------
def vla_rates():
    rows = V["vla"]
    fig = plt.figure(figsize=(5.8, 2.6))
    ax = fig.add_axes([0.2, 0.2, 0.77, 0.66])
    n = len(rows)
    for i, (lab, lo, hi, _) in enumerate(rows):
        y = n - 1 - i
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=P["navy"], lw=7, solid_capstyle="butt")
        ax.plot([lo], [y], "o", ms=10, color=P["navy"])
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold", color=P["text"])
    for x, lab, c in ((V["tau_u"], "60 ms\nuntracked", P["amber"]),
                      (V["tau_t"], "219 ms\ntracked", P["petrol"])):
        ax.axvline(x, color=c, ls="--", lw=1.8)
        ax.text(x, n - 0.35, lab, color=c, fontsize=11, fontweight="bold", ha="center",
                va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(40, 1500)
    ax.set_xticks([50, 100, 200, 500, 1000])
    ax.set_xticklabels(["50", "100", "200", "500", "1,000"])
    ax.minorticks_off()
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time per inference (ms, log scale)", fontsize=12)
    save(fig, "vla_rates")


# ---------------------------------------------------------------------------
# Measured walking speeds against the ISO 13855 design value
# ---------------------------------------------------------------------------
def walking():
    rows = V["walk"]
    fig = plt.figure(figsize=(5.4, 3.6))
    ax = fig.add_axes([0.14, 0.24, 0.84, 0.72])
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [c / 100 for _, c, _ in rows], width=0.38, color=P["gray"])
    ax.bar(x + 0.2, [m / 100 for _, _, m in rows], width=0.38, color=P["navy"])
    ax.axhline(V["v_human"], color=P["red"], ls="--", lw=2)
    ax.text(-0.45, 2.78, "dashed: ISO 13855 design value, 1.6 m/s", ha="left",
            va="center", fontsize=14, color=P["red_d"], fontweight="bold")
    ax.text(-0.2, 1.39 / 2, "usual", rotation=90, ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")
    ax.text(0.2, 2.53 / 2, "fastest", rotation=90, ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" ", "\n") for g, _, _ in rows], fontsize=15)
    ax.set_ylabel("walking speed (m/s)", fontsize=15)
    ax.set_ylim(0, 2.95)
    ax.tick_params(axis="y", labelsize=14)
    save(fig, "walking")


def main():
    autort()
    mug_affordance()
    goalset()
    ap_mcu()
    conformal()
    imprecise()
    wrong_object()
    latch()
    cadence()
    transit()
    aisle()
    lease_terms()
    lifecycle()
    okrobot_fail()
    vla_rates()
    walking()


if __name__ == "__main__":
    main()
