#!/usr/bin/env python3
"""Figures for Lecture 16 (Deployment Release).

Every value is sourced. "DR" is the rendered chapter, Deployment Release
(Volume IV, Chapter 16). Every value comes from calcs.py (imported as K;
labels through deckvals.val), which reads MLSysIM: the running machine, the
Challenger O-ring record (Literature.ChallengerORingDistressRecord, Dalal,
Fowlkes, and Hoadley 1989 via UCI) with the Rogers Commission's temperatures
(Literature.SpaceflightAnomalies), and Waymo's rider-only crash rates by
outcome (Literature.WaymoSafetyImpact).
Diagrams follow the Volume IV causal-boundary palette (rules/volumes/vol4/
palette.md): green for verified or certified, blue for learned proposals,
slate for plant and structure, amber for latency and warnings, red only for
veto, hazard, breach, and refusal.

Every figure is drawn at the size of its slot on the slide, so the slide
places it at scale 1 (plots: fixed figure size, no tight bounding box;
diagrams: an SVG viewBox with the slot's aspect ratio).

Run from this folder: python3 make_figs.py
"""

import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import calcs as K  # noqa: E402  every derived value comes from calcs.py

from deckvals import val  # noqa: E402
from mlsysim import Literature  # noqa: E402  (calcs.py put MLSysIM on the path)

C = sp.C
NAVY, BLUE, GREEN, RED, AMBER, SLATE, PETROL, GRAY = (
    "#1F407A", "#2563EB", "#059669", "#DC2626", "#D97706", "#334155", "#0F766E", "#6B7280")
GREEN_F, BLUE_F, RED_F, SLATE_F, AMBER_F, GRAY_F = (
    "#DCFCE7", "#EFF6FF", "#FEE2E2", "#F8FAFC", "#FEF3C7", "#F3F4F6")
TXT = "#1A1A2E"
FONT = "Helvetica, Arial, sans-serif"


def cm(w, h):
    """Figure size for a slot of w by h centimeters."""
    return (w / 2.54, h / 2.54)


def save_slot(fig, name):
    """Save at exactly the figure size, so the slide places it at scale 1."""
    fig.savefig(HERE / f"{name}.svg", format="svg", facecolor="white")
    plt.close(fig)


# =============================================================================
# SVG helpers for diagrams
# =============================================================================
DEFS = f"""<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker>
<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>
<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker>
</defs>"""


def svg(name, w, h, body):
    (HERE / f"{name}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n{DEFS}\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + "\n".join(body) + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, dash=None, rx=6, sw=2.2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=20, fill=TXT, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{esc(s)}</text>')


def lines(x, y, rows, size=20, fill=TXT, anchor="middle", weight="normal", gap=None):
    gap = gap or size * 1.22
    return [text(x, y + i * gap, r, size, fill, anchor, weight) for i, r in enumerate(rows)]


def arrow(x1, y1, x2, y2, color=SLATE, sw=2.4, dash=None, marker="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def line(x1, y1, x2, y2, color=SLATE, sw=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'


# =============================================================================
# 1. Challenger flight record (hook and segment 1). Slot 14.6 x 5.0 cm.
# =============================================================================
def oring(k):
    fig = plt.figure(figsize=cm(14.6, 5.0))
    ax = fig.add_axes([0.10, 0.22, 0.88, 0.75])
    seen = {}
    for _, t, d in K.ORING:
        if d > 0:
            ax.plot(t, d, "o", ms=8, color=RED, mec="white", mew=1.0, zorder=5)
        else:
            n = seen.get(t, 0)
            seen[t] = n + 1
            ax.plot(t, 0.13 * n, "o", ms=7, color=SLATE, mec="white", mew=1.0, zorder=4,
                    alpha=1.0 if k >= 2 else 0.0)
    ax.text(84, 2.85, "Flights with O-ring distress only" if k == 1 else
            f"All {val('n_flights')} flights in the record", ha="right", va="center", fontsize=11,
            fontweight="bold", color=TXT)
    ax.text(84, 2.42, f"{val('cold_max_f')} °F or colder: {val('cold_hit')} of {val('cold_n')} had distress\n"
            f"{val('warm_min_f')} °F or warmer: {val('warm_hit')} of {val('warm_n')}",
            ha="right", va="center", fontsize=10, color=SLATE, alpha=1.0 if k >= 2 else 0.0,
            linespacing=1.15)
    a3 = 1.0 if k >= 3 else 0.0
    ax.axvspan(15, K.T_THIOKOL, color=GRAY_F, zorder=0, alpha=a3)
    ax.axvline(K.T_THIOKOL, color=SLATE, lw=1.6, ls="--", alpha=a3)
    ax.text(K.T_THIOKOL - 1.5, 2.6, f"no flight evidence\nbelow {val('coldest_f')} °F", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=SLATE, alpha=a3, linespacing=1.1)
    ax.errorbar(K.T_JOINT_51L, 1.0, xerr=K.T_JOINT_51L_ERR, fmt="D", ms=9, color=RED,
                ecolor=RED, elinewidth=2, capsize=4, alpha=a3, zorder=6)
    ax.text(K.T_JOINT_51L, 1.22, f"Challenger joint, about {val('joint_f')} °F", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=RED, alpha=a3)
    ax.text(K.T_JOINT_51L, 0.75, f"(air {val('air_f')} °F)", ha="center", va="top",
            fontsize=10, color=RED, alpha=a3)
    ax.set_xlim(15, 85)
    ax.set_ylim(-0.12, 3.05)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel("Distressed O-rings", fontsize=10.5)
    ax.set_xlabel("Calculated joint temperature at launch (°F)", fontsize=10.5)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save_slot(fig, f"oring_step{k}")


# The aisle stopping budget is the course kit's bar (assets/kit/vol4,
# budget_bar_aisle_step7 and step8); thumb_budget_bar_aisle_step8.svg is the
# kit bar run through tools/greek_thumbs.py.


# =============================================================================
# 3. The safety case for one claim (CAE / GSN). viewBox 900 x 330, slot 14.6 x 5.35 cm.
# =============================================================================
def cae(k):
    W, H = 900, 330
    b = []
    # context (left)
    b.append(box(8, 8, 222, 118, SLATE_F, SLATE, dash="7,5"))
    b += lines(119, 34, ["Context: envelope"], 19, weight="bold")
    b += lines(119, 60, [f"payload ≤ {val('payload')} kg", f"floor μ ≥ {val('mu_min')}", "person stands still"], 18)
    # claim
    b.append(box(262, 8, 400, 66, GREEN_F, GREEN, sw=2.6))
    b += lines(462, 34, [f"Claim C1a: the base stops inside {val('d_clear_m')} m"], 20, weight="bold")
    b += lines(462, 60, [f"at {val('v_aisle')} m/s, loaded (d_stop ≤ D_clear)"], 18)
    b.append(line(230, 41, 262, 41, SLATE, 2, dash="5,4"))
    # warrant
    b.append(box(262, 104, 400, 66, "white", SLATE, sw=2.2))
    b += lines(462, 130, ["Warrant: the stopping budget bounds"], 19, weight="bold")
    b += lines(462, 156, ["travel if every term is bounded"], 18)
    b.append(arrow(462, 104, 462, 77))
    # evidence leaves
    leaves = [("Loaded stop trials", f"a_brake ≥ {val('a_brake')} m/s²"), ("Brake-onset trials", f"≤ {val('t_onset')} ms"),
              ("End-to-end age trace", f"{val('t_age')} ms"), ("Site survey", f"D_clear {val('d_clear_m')} m")]
    for i, (t1, t2) in enumerate(leaves):
        x = 98 + i * 182
        b.append(box(x, 230, 176, 62, "white", GREEN, sw=2.0))
        b += lines(x + 88, 255, [t1], 16, weight="bold")
        b += lines(x + 88, 279, [t2], 16)
        b.append(arrow(x + 88, 230, 462 + (i - 1.5) * 50, 173, GREEN, 2, marker="ahg"))
    b.append(text(450, 318, "Evidence: calibrated records with stated uncertainty", 17, GREEN, weight="bold"))
    if k >= 2:
        b.append(box(8, 150, 222, 56, GRAY_F, GRAY, dash="3,4"))
        b += lines(119, 172, ["1,000 clean runs"], 18, weight="bold", fill=GRAY)
        b += lines(119, 194, ["a demonstration"], 17, fill=GRAY)
        b.append(text(119, 222, "not evidence for new conditions", 15, RED, weight="bold"))
    if k >= 3:
        b.append(box(704, 100, 188, 74, RED_F, RED, sw=2.4))
        b += lines(798, 128, ["Defeater: clear", f"oil film, μ ≈ {val('mu_oil')}"], 18, RED, weight="bold")
        b.append(arrow(704, 137, 666, 137, RED, 2.6, marker="ahr"))
    svg(f"cae_step{k}", W, H, b)


# =============================================================================
# 4. Required premises and independent legs. viewBox 600 x 340, slot 9.0 x 5.1 cm.
# =============================================================================
def legs(k):
    W, H = 600, 340
    b = []
    b.append(text(8, 26, "Conjunctive case: every premise must hold", 20, TXT, "start", "bold"))
    for i in range(K.ARG.premises):
        x = 8 + i * 47
        b.append(box(x, 44, 40, 40, GREEN_F, GREEN, rx=4, sw=1.8))
        b.append(text(x + 20, 70, val("p_premise"), 16, TXT, weight="bold"))
        if i:
            b.append(line(x - 7, 64, x, 64, SLATE, 2))
    b.append(text(488, 72, f"= {val('p_conj')}", 22, RED, "start", "bold"))
    b.append(text(8, 116, f"{val('n_premises_words').capitalize()} premises at {val('p_premise')} hold together at {val('p_conj')},",
                  17, SLATE, "start"))
    b.append(text(8, 138, "below every single premise.", 17, SLATE, "start"))
    if k >= 2:
        b.append(text(8, 184, "Diverse legs: either one suffices", 20, TXT, "start", "bold"))
        b.append(line(20, 260, 70, 260, SLATE, 2))
        b.append(line(70, 222, 70, 298, SLATE, 2))
        for y, lab in ((200, f"Leg A  {val('p_leg')}"), (276, f"Leg B  {val('p_leg')}")):
            b.append(line(70, y + 22, 110, y + 22, SLATE, 2))
            b.append(box(110, y, 170, 44, BLUE_F, NAVY, rx=4, sw=1.8))
            b.append(text(195, y + 29, lab, 18, TXT))
            b.append(line(280, y + 22, 320, y + 22, SLATE, 2))
        b.append(line(320, 222, 320, 298, SLATE, 2))
        b.append(line(320, 260, 360, 260, SLATE, 2))
        b.append(text(368, 268, f"= 1 − {val('p_leg_miss2')}² = {val('p_legs')}", 20, GREEN, "start", "bold"))
        b.append(text(368, 300, "only if they fail", 17, RED, "start", "bold"))
        b.append(text(368, 322, "independently", 17, RED, "start", "bold"))
    svg(f"legs_step{k}", W, H, b)


# =============================================================================
# 5. The door branch: arm deceleration needed (log ladder). Slot 8.4 x 3.6 cm.
# =============================================================================
def door(k):
    a_lo, a_hi = K.A_LATCH_LO, K.A_LATCH_HI
    rows = [(f"{val('v_latch_hi')} m/s", a_hi, RED, f"{val('a_arm_hi')} m/s²", 2),
            ("1 g", K.G, GRAY, f"{val('g_one')} (1 g)", 1),
            (f"{val('v_latch')} m/s", a_lo, GREEN, f"about {val('a_arm')} m/s²", 1)]
    fig = plt.figure(figsize=cm(8.4, 3.6))
    ax = fig.add_axes([0.23, 0.33, 0.68, 0.64])
    for i, (lab, v, col, txt, step) in enumerate(rows):
        a = 1.0 if k >= step else 0.0
        ax.barh(i, v, color=col, height=0.6, alpha=a)
        ax.text(v * 1.25, i, txt, va="center", fontsize=10.5, fontweight="bold", color=col, alpha=a)
        ax.text(0.8, i, lab, ha="right", va="center", fontsize=10.5, fontweight="bold",
                color=TXT, alpha=a)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Arm deceleration needed (m/s²)", fontsize=10)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.tick_params(labelsize=10)
    save_slot(fig, f"door_step{k}")


# =============================================================================
# 6. The mug-delivery case, every branch. viewBox 900 x 360, slot 14.8 x 5.9 cm.
# =============================================================================
TREE = [  # claim, leaf, maturity
    ("C1a  Base stop", f"{val('budget')} of {val('d_clear_mm')} mm; a_brake {val('a_brake')} m/s²", "ill"),
    ("C1b  Door latch", f"{val('f_onset')} N at onset; arm stops in {val('x_rem')} mm", "unk"),
    ("C1c  Mug handover", f"{val('f_handover')} N free hand; no clamping", "unk"),
    ("C2  Permission timing", f"onset ≤ {val('t_silent')} ms; WCET {val('wcet')} of {val('deadline')} µs", "dec"),
    ("C3  Braking traction", f"floor μ ≥ {val('mu_min')} (screen)", "ill"),
    ("C4  Observation age", f"dispatch age {val('t_age')} ms", "ill"),
    ("C5  Independence", "placement test report", "unk"),
    ("C6  Authority transfer", "PASS records for A1 to A4", "unk"),
]
DASH = {"ill": "8,5", "dec": "10,4,2,4", "unk": "2,4"}
MAT = {"ill": "illustrative", "dec": "declared", "unk": "unknown"}


def tree(k):
    W, H = 900, 360
    b = []
    b.append(box(4, 100, 150, 82, GREEN_F, GREEN, sw=2.4))
    b += lines(79, 128, ["C0  Restricted", "mug delivery", "(all must hold)"], 17, weight="bold", gap=21)
    for i, (c, leaf, m) in enumerate(TREE):
        y = 8 + i * 44
        b.append(line(154, 141, 176, y + 17, SLATE, 1.6))
        b.append(box(176, y, 205, 34, "white", SLATE, sw=1.8))
        b.append(text(186, y + 23, c, 17, TXT, "start", "bold"))
        b.append(line(381, y + 17, 394, y + 17, SLATE, 1.6))
        b.append(box(394, y, 300, 34, "white", GRAY if k == 1 else SLATE, dash=DASH[m], sw=2.0))
        b.append(text(402, y + 23, leaf, 16, TXT, "start"))
        if k >= 3:
            b.append(box(702, y + 2, 88, 30, RED_F, RED, rx=4, sw=1.8))
            b.append(text(746, y + 23, "REFUSE", 16, RED, weight="bold"))
    if k >= 2:
        y3 = 8 + 4 * 44
        b.append(box(798, y3 - 6, 98, 46, RED_F, RED, sw=2.2))
        b += lines(847, y3 + 13, ["clear oil", "film"], 16, RED, weight="bold", gap=18)
        b.append(arrow(798, y3 + 17, 790 if k >= 3 else 698, y3 + 17, RED, 2.2, marker="ahr"))
        b.append(text(847, y3 - 14, "defeater", 16, RED, weight="bold"))
        for j, mm in enumerate(("ill", "dec", "unk")):
            yy = 214 + j * 32
            b.append(box(8, yy, 36, 22, "white", SLATE, dash=DASH[mm], sw=2.0, rx=3))
            b.append(text(52, yy + 17, MAT[mm], 16, TXT, "start"))
        b.append(text(8, 330, "none measured", 16, RED, "start", "bold"))
    svg(f"tree_step{k}", W, H, b)


# =============================================================================
# 7. Waymo rider-only crash rates by outcome. Slot 14.6 x 4.9 cm.
# =============================================================================
LABELS = {
    "Police-reported": "Police-reported crash",
    "Any-injury-reported": "Any injury reported",
    "Any Airbag Deployment": "Any airbag deployment",
    "Ego Vehicle Airbag Deployment": "Airbag in the robotaxi",
    "Suspected Serious Injury+": "Serious injury or worse",
    "Any Fatal": "Fatal",
}


def waymo(k):
    rows = Literature.WaymoSafetyImpact.entries
    last = max(r["data_cut"] for r in rows)
    latest = [r for r in rows if r["data_cut"] == last and r["outcome"] in LABELS]
    latest.sort(key=lambda r: -r["waymo_count"])
    fig = plt.figure(figsize=cm(10.8, 4.7))
    ax = fig.add_axes([0.40, 0.31, 0.46, 0.66])
    for i, r in enumerate(latest):
        rare = r["waymo_count"] < 20
        a = 1.0 if (not rare or k >= 2) else 0.0
        col = AMBER if rare else BLUE
        p, lo, hi = (-100 * r["pct_reduction"], -100 * r["pct_reduction_ci_hi"],
                     -100 * r["pct_reduction_ci_lo"])
        ax.plot([lo, hi], [i, i], color=col, lw=3.2, solid_capstyle="butt", alpha=a)
        ax.plot(p, i, "o", color=col, ms=7, alpha=a)
        ax.text(-0.02, i, LABELS[r["outcome"]], ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=TXT, alpha=a, transform=ax.get_yaxis_transform())
        ax.text(1.02, i, f"n = {r['waymo_count']:.0f}", ha="left", va="center", fontsize=10.5,
                color=col, fontweight="bold", alpha=a, transform=ax.get_yaxis_transform())
        if r["outcome"] == "Any Fatal":
            ax.text(lo + 1, i - 0.42, f"{lo:.0f} to {hi:.0f}%", ha="left", va="bottom",
                    fontsize=10, color=AMBER, fontweight="bold", alpha=a)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(latest) - 0.4, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Crash-rate reduction against local\nhuman drivers (%, 95% interval)", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    save_slot(fig, f"waymo_step{k}")


# =============================================================================
# 8. The silent-proposer window (Napkin Math 16.2). Slot 14.6 x 3.4 cm.
# =============================================================================
def silent(k):
    fig = plt.figure(figsize=cm(14.6, 3.7))
    ax = fig.add_axes([0.30, 0.27, 0.48, 0.71])
    rows = [(2, "Proposer falls silent", 0, K.T_LEASE, f"lease runs out: {val('t_lease')} ms", BLUE_F, BLUE, 1),
            (1, "Premise flags at once", 0, 0, "", AMBER_F, AMBER, 1),
            (0, f"Premise flags at {val('t_crossover')} ms", 0, K.T_SILENT - K.T_FLAGGED,
             f"flag latency: {val('t_crossover')} ms", AMBER_F, AMBER, 2)]
    for y, lab, x0, pre, pre_lab, fc, ec, step in rows:
        a = 1.0 if k >= step else 0.0
        if pre:
            ax.barh(y, pre, left=x0, color=fc, edgecolor=ec, lw=1.3, height=0.6, alpha=a,
                    hatch="///" if ec == AMBER else None)
            ax.text(pre / 2, y, pre_lab, ha="center", va="center", fontsize=10,
                    color=NAVY if ec == BLUE else "#92400E", fontweight="bold", alpha=a)
        ax.barh(y, K.T_TICK + K.T_BUS, left=pre, color=SLATE, height=0.6, alpha=a)
        ax.barh(y, K.T_ONSET, left=pre + K.T_TICK + K.T_BUS, color=RED, height=0.6, alpha=a)
        ax.text(pre + K.T_TICK + K.T_BUS + K.T_ONSET / 2, y, "onset", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold", alpha=a)
        end = pre + K.T_FLAGGED
        label = (f"{val('t_silent')} ms: {val('d_silent')} mm" if pre else
                 f"{val('t_flagged')} ms: {val('d_flagged')} mm")
        ax.text(end + 1.5, y, label, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color=TXT, alpha=a)
        ax.text(-2, y, lab, ha="right", va="center", fontsize=10.5, fontweight="bold", color=TXT,
                alpha=a)
    ax.set_xlim(0, 85)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Milliseconds from the failure to brake onset, base at {val('v_aisle')} m/s", fontsize=10)
    ax.tick_params(labelsize=10)
    save_slot(fig, f"silent_step{k}")


# =============================================================================
# 9. The envelope trichotomy. viewBox 900 x 210, slot 14.6 x 3.4 cm.
# =============================================================================
def tri():
    W, H = 900, 210
    b = []
    cols = [("Known-true", GREEN_F, GREEN, ["qualified state set inside", "the envelope; fresh,", "verified evidence"], "permit = 1: track"),
            ("Unknown", AMBER_F, AMBER, ["sensor dropout, CRC", "mismatch, or estimator", "divergence"], "permit = 0"),
            ("Known-false", RED_F, RED, ["measured breach, such as", "tracking error above", f"its {val('d_track')} mm bound"], "permit = 0")]
    for i, (t, f, s, body, out) in enumerate(cols):
        x = 8 + i * 300
        b.append(box(x, 8, 284, 138, f, s, sw=2.6))
        b.append(text(x + 142, 38, t, 22, s if s != AMBER else "#92400E", weight="bold"))
        b += lines(x + 142, 66, body, 18)
        b.append(text(x + 142, 176, out, 19, TXT, weight="bold"))
    b.append(line(316, 194, 884, 194, SLATE, 2))
    b.append(line(316, 186, 316, 194, SLATE, 2))
    b.append(line(884, 186, 884, 194, SLATE, 2))
    b.append(text(600, 206, "same response (assessed stop or hold); causes logged apart", 16, SLATE))
    svg("tri", W, H, b)


# =============================================================================
# 10. O-ring distress by flight order (normalization of deviance). Slot 8.6 x 4.2 cm.
# =============================================================================
def erosion():
    fig = plt.figure(figsize=cm(8.6, 4.2))
    ax = fig.add_axes([0.13, 0.26, 0.84, 0.64])
    for order, _, d in K.ORING:
        if d:
            ax.bar(order, d, color=RED, width=0.7)
        else:
            ax.plot(order, 0.06, "o", ms=4, color=SLATE)
    ax.set_xlim(0.3, 24)
    ax.set_ylim(0, 2.6)
    ax.set_yticks([0, 1, 2])
    ax.set_xticks([1, 5, 10, 15, 20, 23])
    ax.set_xlabel("Flight, in launch order", fontsize=10)
    ax.set_ylabel("O-rings with distress", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.text(23.5, 2.45, f"{val('n_distressed')} of {val('n_flights')} flights; each one returned", ha="right", va="top",
            fontsize=10, color=RED, fontweight="bold")
    save_slot(fig, "erosion")


# =============================================================================
# 11. The release manifest. viewBox 900 x 370, slot 14.8 x 6.1 cm.
# =============================================================================
GROUPS = [
    ("Plant and timing", [("limit record", "L2"), ("handoff record", "L4")]),
    ("Data and policy", [("provenance", "L5"), ("policy manifest", "L6"), ("evaluation", "L7")]),
    ("Perception, planning", [("observation contract", "L8"), ("belief record", "L9"),
                              ("intent lease", "L10"), ("trajectory record", "L11")]),
    ("Containment", [("enforcement", "L12"), ("placement", "L13"), ("authority", "L14"),
                     ("fault records", "L15")]),
]


def manifest(k):
    W, H = 900, 370
    b = []
    y = 6
    for g, recs in GROUPS:
        b.append(text(8, y + 18, g, 17, SLATE, "start", "bold"))
        y += 24
        for j, (r, lec) in enumerate(recs):
            x = 8 + (j % 2) * 176
            if j and j % 2 == 0:
                y += 30
            b.append(box(x, y, 172, 26, SLATE_F, SLATE, rx=4, sw=1.4))
            b.append(text(x + 5, y + 19, r, 15, TXT, "start"))
            b.append(text(x + 167, y + 19, lec, 15, GRAY, "end", "bold"))
        y += 36
    b.append(text(8, 342, "Digest: a hash that changes if any byte does.", 15, GRAY, "start"))
    b.append(text(8, 362, "Merkle root: one digest over all digests.", 15, GRAY, "start"))
    a2 = k >= 2
    if a2:
        b.append(arrow(360, 185, 384, 185, SLATE, 2.4))
        b.append(box(388, 6, 236, 358, GREEN_F, GREEN, sw=2.6))
        b.append(text(506, 34, "Release manifest", 21, GREEN, weight="bold"))
        fields = ["record digests → Merkle root", "root claim, argument digest", "binary hashes: policy,",
                  "   enforcer, RTOS kernel", "calibration tables", "standing conditions",
                  "wear budget (hours, cycles)", "expiry date", "adjudicator signature (HSM)"]
        for i, f in enumerate(fields):
            b.append(text(398, 68 + i * 32, f, 15.5, TXT, "start"))
    if k >= 3:
        b.append(arrow(624, 185, 646, 185, SLATE, 2.4))
        b.append(box(650, 120, 112, 130, "white", SLATE, sw=2.2))
        b += lines(706, 150, ["Gate:", "authenticate,", "then check", "standing"], 16, gap=24)
        b.append(arrow(762, 160, 782, 110, GREEN, 2.2, marker="ahg"))
        b.append(arrow(762, 210, 782, 260, RED, 2.2, marker="ahr"))
        b.append(box(784, 54, 112, 84, GREEN_F, GREEN, sw=2.2))
        b += lines(840, 80, ["UNCOND or", "COND:", "permit on"], 16, GREEN, weight="bold", gap=20)
        b.append(box(784, 232, 112, 84, RED_F, RED, sw=2.2))
        b += lines(840, 258, ["REFUSE:", "permit off;", "stop or hold"], 16, RED, weight="bold", gap=20)
    svg(f"manifest_step{k}", W, H, b)


# =============================================================================
# 12. What an update reopens. viewBox 900 x 300, slot 14.6 x 4.9 cm.
# After DR 16.9.1 (backbone and bus contention), 16.7 (third and fourth
# probes: C4 age chain, OS reflash and calibration), 16.8 (manifest
# invalidation), and 16.5 (wear with unchanged software).
# =============================================================================
REOPEN_COLS = ["Task", "Stop", "Contact", "Timing", "Age", "Indep."]
REOPEN_ROWS = [
    ("Retrained policy, same backbone", [1, 0, 1, 0, 0, 0]),
    ("Larger backbone, same enforcer binary", [1, 0, 0, 1, 1, 1]),
    ("Driver or thread-priority change", [0, 0, 0, 1, 1, 1]),
    ("OS reflash that resets calibration", [0, 1, 1, 0, 0, 0]),
    ("Worn brake lining, no software change", [0, 1, 0, 0, 0, 0]),
]


def reopen():
    W, H = 900, 300
    b = []
    x0, dx = 440, 76
    for j, c in enumerate(REOPEN_COLS):
        b.append(text(x0 + j * dx, 26, c, 18, SLATE, weight="bold"))
    b.append(text(x0 + 2.5 * dx, 292, "Task: capability evidence. The other columns: C1 stop and contact, C2, C4, C5.", 15, GRAY))
    for i, (r, marks) in enumerate(REOPEN_ROWS):
        y = 66 + i * 46
        b.append(text(8, y + 6, r, 18, TXT, "start", "bold" if i in (1, 4) else "normal"))
        b.append(line(8, y + 22, 892, y + 22, "#E5E7EB", 1))
        for j, m in enumerate(marks):
            cx = x0 + j * dx
            if m:
                b.append(f'<circle cx="{cx}" cy="{y}" r="12" fill="{AMBER}"/>')
            else:
                b.append(f'<circle cx="{cx}" cy="{y}" r="11" fill="white" stroke="#D1D5DB" stroke-width="2"/>')
    svg("reopen", W, H, b)


if __name__ == "__main__":
    for k in (1, 2, 3):
        oring(k)
        cae(k)
        tree(k)
        manifest(k)
    for k in (1, 2):
        legs(k)
        door(k)
        waymo(k)
        silent(k)
    tri()
    erosion()
    reopen()
    print("figures written to", HERE)
