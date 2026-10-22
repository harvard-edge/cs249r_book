#!/usr/bin/env python3
"""Figures for Lecture 14 (Supervisory Intervention).

Every value comes from MLSysIM through calcs.py (imported as K); printed
numbers use the formatted strings calcs.py records (deckvals.val). "SI" is the
rendered chapter, Supervisory Intervention (Volume IV, Chapter 14). Data:
  - California DMV disengagement filings, Waymo 2015 to 2025
    (Literature.CaliforniaDisengagements; SI 14.1 figure 1).
  - Zhang et al. 2019, 520 mean take-over times (Literature.TakeoverTimeMeta).
  - Lufthansa 2904 timeline (Literature.IncidentReports; SI War Story 14.1).
Diagram colors follow the Volume IV causal-boundary palette
(rules/volumes/vol4/palette.md); red only for veto, hazard, and limits.
Figures are drawn at the size of their slot: full-width SVG 900 px wide with
text 18 px or larger; column SVG about 560 px wide with text 20 px or larger.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402

import calcs as K  # noqa: E402  (prints its table on import)
from deckvals import val, num  # noqa: E402

# Volume IV roles
NAVY, BLUE, GREEN, RED, AMBER, SLATE, PETROL = (
    "#1F407A", "#2563EB", "#059669", "#DC2626", "#D97706", "#334155", "#0F766E")
GREEN_F, BLUE_F, RED_F, SLATE_F, AMBER_F = "#DCFCE7", "#EFF6FF", "#FEE2E2", "#F8FAFC", "#FEF3C7"
GRAY, LGRAY = "#94A3B8", "#E2E8F0"
TXT = "#1A1A2E"
FONT = "Helvetica, Arial, sans-serif"


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", format="svg", bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    plt.close(fig)


# =============================================================================
# SVG helpers
# =============================================================================
def _marker(mid, color):
    return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')


DEFS = "<defs>" + "".join(_marker(m, c) for m, c in [
    ("ah", SLATE), ("ahg", GREEN), ("ahb", BLUE), ("ahr", RED), ("ahn", NAVY), ("aha", AMBER)]) + (
    '<pattern id="hatch" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)">'
    f'<rect width="10" height="10" fill="{RED_F}"/><line x1="0" y1="0" x2="0" y2="10" stroke="{RED}" stroke-width="3"/></pattern>'
    "</defs>")


def svg(name, w, h, body):
    (HERE / f"{name}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n{DEFS}\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + "\n".join(body) + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, dash=None, rx=8, sw=2.5):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=20, fill=TXT, anchor="middle", weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}>{s}</text>')


def lines(x, y, rows, size=20, fill=TXT, anchor="middle", weight="normal", gap=None):
    gap = gap or size * 1.25
    return [text(x, y + i * gap, r, size, fill, anchor, weight) for i, r in enumerate(rows)]


def arrow(x1, y1, x2, y2, color=SLATE, sw=3, dash=None, marker="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def line(x1, y1, x2, y2, color=SLATE, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'


def bracket(x1, x2, y, label, color=SLATE, size=19, up=True, weight="bold"):
    t = -8 if up else 8
    ty = y - 12 if up else y + 28
    return [f'<path d="M{x1},{y - t} L{x1},{y} L{x2},{y} L{x2},{y - t}" fill="none" stroke="{color}" stroke-width="2.5"/>',
            text((x1 + x2) / 2, ty, label, size, color, weight=weight)]


# =============================================================================
# Hook: California DMV filings, Waymo (SI 14.1 figure 1, panel b)
# =============================================================================
def dmv():
    rows = K.DMV
    sd = sorted((r["report_year"], r["miles_per_disengagement"]) for r in rows
                if r["permit_type"] == K.DRV and r["miles_per_disengagement"])
    dl = sorted((r["report_year"], r["miles_per_disengagement"]) for r in rows
                if r["permit_type"] == K.DL and r["miles_per_disengagement"])
    last = K.DMV_LAST
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.6, 2.55))
        pts = [p for p in sd if step == 2 or p[0] <= 2020]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=sp.C["memory"], lw=2.5, ms=6)
        ax.text(sd[0][0] + 0.25, sd[0][1] * 0.82, f"{sd[0][1]:,.0f}", color=sp.C["memory"],
                fontsize=12, va="top", fontweight="bold")
        y2020 = dict(sd)[2020]
        ax.text(2020, y2020 * 1.35, f"{y2020:,.0f}", color=sp.C["memory"], fontsize=12,
                ha="center", fontweight="bold")
        ax.text(2016.6, 2.2e4, "with a safety driver", color=sp.C["memory"], fontsize=12,
                fontweight="bold", ha="center")
        if step == 1:
            ax.text(2023.5, 1.1e3, "2021 to 2025?", color=sp.C["neutral"], fontsize=13,
                    ha="center", fontweight="bold")
        else:
            y2021 = dict(sd)[2021]
            ax.text(2021.15, y2021 * 0.62, f"{y2021:,.0f}", color=sp.C["memory"], fontsize=12,
                    ha="center", va="top", fontweight="bold")
            ax.text(2025.25, sd[-1][1] * 0.9, f"{sd[-1][1]:,.0f}\n{last['disengagements']:.0f} takeovers",
                    color=sp.C["memory"], fontsize=12, va="top", fontweight="bold")
            ax.plot([p[0] for p in dl], [p[1] for p in dl], "s-", color=sp.C["data"], lw=2.5, ms=7)
            ax.text(2025.25, dl[-1][1] * 1.05, f"{dl[-1][1]:,.0f}", color=sp.C["data"], fontsize=12,
                    va="bottom", fontweight="bold")
            ax.text(2022.6, 1.25e5, "driverless testing", color=sp.C["data"], fontsize=12,
                    fontweight="bold", ha="right")
        ax.set_yscale("log")
        ax.set_ylim(600, 2.5e5)
        ax.set_yticks([1e3, 1e4, 1e5])
        ax.set_yticklabels(["1,000", "10,000", "100,000"])
        ax.minorticks_off()
        ax.set_xlim(2014.5, 2027.7)
        ax.set_xticks([2015, 2017, 2019, 2021, 2023, 2025])
        ax.set_ylabel("Miles per disengagement")
        ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
        save(fig, f"dmv_step{step}")


# =============================================================================
# Intervention changes the proposer; enforcement changes the proposal (SI 14.2)
# =============================================================================
def propose_admit():
    b = []
    for col, (x0, title) in enumerate([(0, "Enforcement"), (455, "Intervention")]):
        b.append(text(x0 + 222, 34, title, 28, TXT, weight="bold"))
        if col == 0:
            b.append(box(x0 + 15, 56, 205, 80, BLUE_F, NAVY))
            b += lines(x0 + 117, 90, ["Learned", "policy"], 22, NAVY, weight="bold", gap=26)
            b.append(box(x0 + 235, 56, 195, 80, "white", GRAY, dash="6,4"))
            b += lines(x0 + 332, 90, ["no", "operator"], 22, GRAY, gap=26)
            b.append(arrow(x0 + 117, 138, x0 + 117, 188, BLUE, dash="7,5", marker="ahb"))
            b.append(text(x0 + 135, 172, "proposal", 21, BLUE, anchor="start", italic=True))
        else:
            b.append(box(x0 + 15, 56, 205, 80, "white", GRAY, dash="6,4"))
            b += lines(x0 + 117, 90, ["Policy:", "token revoked"], 22, GRAY, gap=26)
            b.append(box(x0 + 235, 56, 195, 80, BLUE_F, BLUE))
            b += lines(x0 + 332, 90, ["Authenticated", "operator"], 22, BLUE, weight="bold", gap=26)
            b.append(arrow(x0 + 332, 138, x0 + 332, 188, BLUE, dash="7,5", marker="ahb"))
            b.append(text(x0 + 314, 172, "proposal", 21, BLUE, anchor="end", italic=True))
        b.append(box(x0 + 22, 190, 400, 80, GREEN_F, GREEN))
        b += lines(x0 + 222, 224, ["Enforcer on the safety MCU", "admits, corrects, or refuses"], 22, "#047857",
                   weight="bold", gap=27)
        b.append(arrow(x0 + 222, 272, x0 + 222, 306, GREEN, marker="ahg"))
        b.append(box(x0 + 117, 308, 210, 52, SLATE_F, SLATE, rx=0))
        b.append(text(x0 + 222, 342, "Drives", 22, SLATE, weight="bold"))
        cap = ["changes the proposal;", "the proposer stays"] if col == 0 else ["changes the proposer;", "the check stays"]
        b += lines(x0 + 222, 396, cap, 23, TXT, weight="bold", gap=27)
    b.append(line(450, 20, 450, 420, LGRAY, 2))
    svg("propose_admit", 900, 432, b)


# =============================================================================
# Takeover timing strip (SI 14.2.2, figure 2)
# =============================================================================
def strip():
    x0, x1, tmax = 200, 880, 650.0
    X = lambda ms: x0 + (x1 - x0) * ms / tmax  # noqa: E731
    lanes = {"Console": 72, "Operator": 150, "Permission": 228}
    tv = K.T_VIDEO * 1e3
    tp, tc, tm = K.T_PERC * 1e3, K.T_COG * 1e3, K.T_MOT * 1e3
    t_act_end = tv + tp + tc + tm
    t_in = K.T_IN * 1e3
    spare_ms = K.SPARE / K.V_AISLE * 1e3
    for step in (1, 2, 3):
        b = []
        for name, y in lanes.items():
            b.append(text(x0 - 12, y + 36, name, 23, TXT, anchor="end", weight="bold"))
            b.append(line(x0, y + 54, x1, y + 54, LGRAY, 1.5))
        ay = 288
        b.append(line(x0, ay, x1, ay, SLATE, 2))
        for ms in range(0, 700, 100):
            b.append(line(X(ms), ay, X(ms), ay + 8, SLATE, 2))
            b.append(text(X(ms), ay + 32, f"{ms}", 22, SLATE))
        b.append(text(x0 - 12, ay + 32, "ms", 22, SLATE, anchor="end"))
        y = lanes["Console"]
        b.append(box(X(0), y + 8, X(tv) - X(0), 44, AMBER_F, AMBER, rx=3))
        b.append(text(X(tv) + 10, y + 38, f"video {tv:.0f} ms", 23, "#92400E", anchor="start", weight="bold"))
        ty = 366
        b.append(text(x0 - 12, ty + 30, "Travel", 23, TXT, anchor="end", weight="bold"))
        b.append(box(X(0), ty + 6, X(spare_ms) - X(0), 34, RED_F, RED, rx=2))
        if step < 3:
            b.append(text(X(spare_ms) + 10, ty + 32, f"spare {K.SPARE * 1e3:.0f} mm, gone at {spare_ms:.0f} ms",
                          23, RED, anchor="start", weight="bold"))
        if step >= 2:
            y = lanes["Operator"]
            for t0, dur, lab in [(tv, tp, "perceive"), (tv + tp, tc, "decide"), (tv + tp + tc, tm, "act")]:
                b.append(box(X(t0), y + 8, X(t0 + dur) - X(t0), 44, BLUE_F, BLUE, rx=3))
                b.append(text((X(t0) + X(t0 + dur)) / 2, y + 38, f"{lab} {dur:.0f}" if lab != "act" else "act",
                              22, "#1D4ED8", weight="bold"))
        if step >= 3:
            y = lanes["Permission"]
            b.append(box(X(t_act_end), y + 8, X(t_in) - X(t_act_end), 44, GREEN_F, GREEN, rx=3))
            b.append(text(X(t_act_end) - 10, y + 38, f"handshake {K.T_HS * 1e3:.0f} ms", 23, "#047857",
                          anchor="end", weight="bold"))
            b.append(line(X(t_in), 44, X(t_in), ay, GREEN, 2.5, dash="6,4"))
            b += bracket(X(0), X(t_in), 36, f"in-loop takeover, {t_in:.0f} ms, before any command changes", GREEN, 23)
            b.append(box(X(spare_ms), ty + 6, X(t_in) - X(spare_ms), 34, "url(#hatch)", RED, rx=2))
            b.append(text(X(0), ty - 8, f"spare {val('spare0')} mm, then {val('travel_in')} mm in all",
                          22, RED, anchor="start", weight="bold"))
        svg(f"strip_step{step}", 900, 412, b)


# =============================================================================
# The takeover tax (SI Napkin Math 14.1)
# =============================================================================
# The takeover-tax bars are the course kit's stopping-budget bar
# (assets/kit/vol4/budget_bar_aisle_step7) with the takeover stated in text.


def zhang():
    vals = np.array(K.MTOT)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    bins = np.arange(0, 10.01, 0.25)
    ax.hist(vals[vals <= 10], bins=bins, color=sp.C["memory_l"], edgecolor=sp.C["memory"], lw=0.8)
    marks = [("median", K.TOT["p50"], "p50", 123),
             ("mean", K.TOT["mean"], "mean", 123),
             ("90th pct.", K.TOT["p90"], "p90", 80),
             ("99th pct.", K.TOT["p99"], "p99", 80)]
    for lab, x, key, y in marks:
        col = RED if key == "p90" else SLATE  # the design percentile
        ax.axvline(x, color=col, lw=2, ls="--")
        ha = "right" if lab == "median" else "left"
        dx = -0.08 if ha == "right" else 0.08
        ax.text(x + dx, y, f"{lab} {x:.2f} s\n{K.CEIL[key]:.2f} m/s", color=col, fontsize=11,
                fontweight="bold", ha=ha, va="top")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 126)
    ax.set_xlabel("Mean take-over time of one study condition (s)", fontsize=12)
    ax.set_ylabel("Conditions", fontsize=12)
    ax.text(9.95, 36, f"{val('n_off_scale')} condition at\n{val('tot_max')} s off scale", ha="right", fontsize=10.5, color=sp.C["neutral"])
    save(fig, "zhang_hist")


# =============================================================================
# The fallback deadline against speed (SI eq. 4, Napkin Math 14.3)
# =============================================================================
def deadline():
    v = np.linspace(0.12, 1.42, 400)
    dl = np.array([K.deadline(x) for x in v]) * 1e3
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.6))
        ax.plot(v, dl, color=GREEN, lw=2.8)
        ax.text(0.22, 42, "longest wait before\nthe stop must begin", color="#047857", fontsize=11.5,
                fontweight="bold")
        ax.plot([K.V_AISLE], [K.DL_AISLE * 1e3], "o", color=RED, ms=8)
        ax.text(K.V_AISLE - 0.02, K.DL_AISLE * 1e3 * 0.62, f"{val('v_aisle')} m/s: {val('deadline')} ms",
                color=RED, fontsize=11.5, fontweight="bold", ha="right", va="top")
        if step >= 2:
            ax.axhline(K.T_IN * 1e3, color=AMBER, lw=2, ls="--")
            ax.text(0.22, K.T_IN * 1e3 * 0.88, f"in-loop takeover {val('t_in')} ms", color="#B45309", fontsize=11,
                    fontweight="bold", ha="left", va="top")
            ax.plot([K.V_SLOW], [K.DL_SLOW * 1e3], "o", color=NAVY, ms=8)
            ax.text(K.V_SLOW + 0.03, K.DL_SLOW * 1e3 * 1.25, f"{val('v_slow')} m/s: {val('dl_slow')} ms",
                    color=NAVY, fontsize=11.5, fontweight="bold", va="bottom")
        if step >= 3:
            ax.axhline(K.T_OOL * 1e3, color=AMBER, lw=2, ls=":")
            ax.text(1.42, K.T_OOL * 1e3 * 1.12, f"out-of-loop takeover {val('t_ool_s')} s", color="#B45309", fontsize=11,
                    fontweight="bold", ha="right", va="bottom")
            ax.plot([K.V_CRAWL], [K.DL_CRAWL * 1e3], "o", color=NAVY, ms=8)
            ax.text(K.V_CRAWL + 0.02, K.DL_CRAWL * 1e3 * 1.3, f"{val('v_crawl')} m/s: {val('dl_crawl_s')} s",
                    color=NAVY, fontsize=11.5, fontweight="bold", va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(30, 20000)
        ax.set_yticks([100, 1000, 10000])
        ax.set_yticklabels(["100 ms", "1 s", "10 s"])
        ax.minorticks_off()
        ax.set_xlim(0.1, 1.45)
        ax.set_xlabel("Speed when the takeover is requested (m/s)", fontsize=12)
        ax.set_ylabel("Fallback deadline", fontsize=12)
        ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
        save(fig, f"deadline_step{step}")


# =============================================================================
# Two leases on the safety MCU's clock (SI 14.3.1, 14.3.4)
# =============================================================================
def leases():
    x0, x1, tmax = 20, 548, 272.0
    X = lambda ms: x0 + (x1 - x0) * ms / tmax  # noqa: E731
    b = []
    lease, manual, period = num("t_lease"), num("t_manual"), num("chunk_period")
    specs = [(f"Autonomous: chunk lease {val('t_lease')} ms", 34, [0, period, period + period], lease, NAVY, "renewal"),
             (f"Manual: lease {val('t_manual')} ms", 206, [0, 25, 50, 75, 100], manual, BLUE, "heartbeat")]
    for title, y, ev, lease, col, lab in specs:
        b.append(text(x0, y, title, 23, col, anchor="start", weight="bold"))
        b.append(line(x0, y + 66, x1, y + 66, SLATE, 2))
        for t in ev:
            b.append(line(X(t), y + 34, X(t), y + 66, col, 4))
        last = ev[-1]
        b.append(text(X(last), y + 28, f"last {lab}", 21, col, anchor="middle"))
        b.append(box(X(last), y + 72, X(last + lease) - X(last), 22, BLUE_F, col, rx=3, sw=2))
        b.append(text((X(last) + X(last + lease)) / 2, y + 120, f"{lease:.0f} ms, MCU clock", 21, col))
        b.append(line(X(last + lease), y + 30, X(last + lease), y + 98, RED, 3))
        b.append(box(X(last + lease) + 6, y + 34, 122, 32, RED_F, RED, rx=4, sw=2))
        b.append(text(X(last + lease) + 67, y + 58, "FALLBACK", 20, RED, weight="bold"))
    b.append(text(x0, 372, "Silence ends authority on the MCU clock,", 21, TXT, anchor="start", weight="bold"))
    b.append(text(x0, 398, "and the machine holds or stops.", 21, TXT, anchor="start", weight="bold"))
    svg("leases", 570, 410, b)


# =============================================================================
# Authority states (SI 14.3.2, figure 5)
# =============================================================================
def fsm():
    W, H = 168, 84
    pos = {"AUTO": (12, 74), "PEND": (244, 74), "BLEND": (476, 74), "MAN": (708, 74),
           "FALLBACK": (290, 250), "PROTECTIVE": (708, 290)}
    names = {"AUTO": "Autonomous", "PEND": "Handover-", "BLEND": "Blend", "MAN": "Human-",
             "FALLBACK": "Fallback", "PROTECTIVE": "Protective"}
    names2 = {"PEND": "Pending", "MAN": "Manual"}
    token = {"AUTO": "policy", "PEND": "policy", "BLEND": "no one",
             "MAN": "operator", "FALLBACK": "no one", "PROTECTIVE": "no one"}
    col = {"AUTO": (BLUE_F, NAVY), "PEND": (BLUE_F, NAVY), "BLEND": (GREEN_F, GREEN),
           "MAN": (BLUE_F, BLUE), "FALLBACK": (RED_F, RED), "PROTECTIVE": (RED_F, RED)}

    def node(k):
        x, y = pos[k]
        f, s = col[k]
        out = [box(x, y, W, H, f, s)]
        if k in names2:
            out += lines(x + W / 2, y + 27, [names[k], names2[k]], 22, s, weight="bold", gap=24)
            out.append(text(x + W / 2, y + 76, token[k], 20, SLATE, italic=True))
        else:
            out.append(text(x + W / 2, y + 36, names[k], 22, s, weight="bold"))
            out.append(text(x + W / 2, y + 68, token[k], 20, SLATE, italic=True))
        return out

    for step in (1, 2, 3):
        b = [f'<path d="M4,22 H896 V236 H690 V392 H4 Z" fill="none" stroke="{GRAY}" stroke-width="2" '
             f'stroke-dasharray="8,6"/>']
        b.append(text(14, 382, "Italic: who holds the proposal token", 20, SLATE, anchor="start", italic=True))
        for k in ["AUTO", "PEND", "BLEND", "MAN"]:
            b += node(k)
        cy = 74 + H / 2
        for (a, c, lab, colr, mk) in [("AUTO", "PEND", "request", NAVY, "ahn"),
                                      ("PEND", "BLEND", "commit", GREEN, "ahg"),
                                      ("BLEND", "MAN", "done", BLUE, "ahb")]:
            xa = pos[a][0] + W
            xc = pos[c][0]
            b.append(arrow(xa + 2, cy, xc - 2, cy, colr, marker=mk))
            b.append(text((xa + xc) / 2, 62, lab, 20, colr, weight="bold"))
        if step >= 2:
            b += node("FALLBACK")
            fx, fy = pos["FALLBACK"]
            for k, tx in [("AUTO", fx + 20), ("PEND", fx + 60), ("BLEND", fx + 110), ("MAN", fx + 150)]:
                x, y = pos[k]
                b.append(arrow(x + W / 2, y + H, tx, fy - 2, RED, sw=2.5, marker="ahr"))
            b.append(text(fx + W + 14, fy + 36, "lease lapse,", 20, RED, anchor="start"))
            b.append(text(fx + W + 14, fy + 60, "timeout, bad state", 20, RED, anchor="start"))
            b.append(f'<path d="M{fx},{fy + H / 2} C{150},{fy + H / 2} {50},{fy + 10} {50},{74 + H + 4}" fill="none" '
                     f'stroke="{SLATE}" stroke-width="2.5" stroke-dasharray="7,5" marker-end="url(#ah)"/>')
            b.append(text(140, fy + H / 2 + 48, "reset from rest only", 20, SLATE))
        if step >= 3:
            b += node("PROTECTIVE")
            px, py = pos["PROTECTIVE"]
            b.append(arrow(px + W / 2, 236, px + W / 2, py - 2, RED, sw=3, marker="ahr"))
            b.append(text(px + W / 2 + 10, 262, "hardwired", 20, RED, anchor="start"))
        svg(f"fsm_step{step}", 900, 400, b)


# =============================================================================
# The four-phase handshake (SI 14.3.3, table 5)
# =============================================================================
def handshake():
    x0, x1, tmax = 150, 880, 34.0
    X = lambda ms: x0 + (x1 - x0) * ms / tmax  # noqa: E731
    lanes = {"Console": 66, "MCU": 166, "Drives": 266}
    b = []
    for name, y in lanes.items():
        b.append(text(x0 - 12, y + 34, name, 23, TXT, anchor="end", weight="bold"))
        b.append(line(x0, y + 52, x1, y + 52, LGRAY, 1.5))
    t = 0.0
    starts = {}
    for k, d in K.HS.items():
        starts[k] = (t, t + d * 1e3)
        t += d * 1e3
    r0, r1 = starts["Request"]
    b.append(box(X(r0), 72, X(r1) - X(r0), 40, BLUE_F, BLUE, rx=3))
    b.append(text((X(r0) + X(r1)) / 2, 100, f"Request {val('hs_req')}", 22, "#1D4ED8", weight="bold"))
    b.append(arrow(X(r1) - 4, 114, X(r1) + 6, 172, BLUE, marker="ahb"))
    a0, a1 = starts["Acknowledge"]
    b.append(box(X(a0), 174, X(a1) - X(a0), 40, GREEN_F, GREEN, rx=3))
    b.append(text((X(a0) + X(a1)) / 2, 240, f"Ack {val('hs_ack')}", 22, "#047857", weight="bold"))
    c0, c1 = starts["Commit"]
    b.append(box(X(c0), 168, X(c1) - X(c0) + 2, 52, RED_F, RED, rx=2))
    b.append(line(X(c0), 44, X(c0), 326, RED, 2.5, dash="6,4"))
    b.append(text(X(c0), 36, f"Commit on one {val('tick')} ms tick changes the owner", 22, RED, weight="bold"))
    f0, f1 = starts["Confirm"]
    b.append(box(X(f0) + 2, 174, X(f1) - X(f0) - 2, 40, GREEN_F, GREEN, rx=3))
    b.append(text((X(f0) + X(f1)) / 2, 202, f"Confirm {val('hs_conf')}", 22, "#047857", weight="bold"))
    b.append(arrow(X(f1) - 6, 172, X(f1) + 4, 114, GREEN, marker="ahg"))
    b.append(text(X(f1) - 8, 100, "if lost, alarm only", 22, RED, anchor="end", italic=True))
    b.append(box(X(0), 274, X(c0) - X(0), 40, BLUE_F, NAVY, rx=3))
    b.append(text((X(0) + X(c0)) / 2, 302, "policy setpoints", 22, NAVY))
    b.append(box(X(c1), 274, X(tmax) - X(c1), 40, GREEN_F, GREEN, rx=3))
    b.append(text((X(c1) + X(tmax)) / 2, 302, "enforcer blend", 22, "#047857"))
    ay = 344
    b.append(line(x0, ay, x1, ay, SLATE, 2))
    for ms, lab in [(0, "0"), (num("hs_req"), val("hs_req")), (num("hs_acked"), val("hs_acked")),
                    (num("t_hs"), f"{val('t_hs')} ms")]:
        b.append(line(X(ms), ay, X(ms), ay + 8, SLATE, 2))
        b.append(text(X(ms), ay + 32, lab, 22, SLATE))
    svg("handshake", 900, 385, b)


# =============================================================================
# Step against quintic (SI 14.4, eq. 6 and 7)
# =============================================================================
def quintic():
    W = K.W_BASE
    t = np.linspace(-0.15, 0.75, 2000)
    s = np.clip(t / W, 0, 1)
    u = K.DA_BASE * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    du = np.where((t > 0) & (t < W), K.DA_BASE * 30 * s ** 2 * (1 - s) ** 2 / W, 0.0)
    for step in (1, 2):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.9), sharex=True,
                                     gridspec_kw=dict(hspace=0.25))
        step_u = np.where(t >= 0, K.DA_BASE, 0.0)
        a1.plot(t * 1e3, step_u, color=RED, lw=2.4)
        a1.text(20, 1.55, "step", color=RED, fontsize=12, fontweight="bold")
        a2.annotate("", xy=(0, 9.6), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
        a2.text(14, 9.0, f"step: {K.DA_BASE / K.TICK:,.0f} m/s³ for one tick (off scale)", color=RED,
                fontsize=11.5, fontweight="bold", va="center")
        if step == 2:
            a1.plot(t * 1e3, u, color=GREEN, lw=2.8)
            a1.text(W * 1e3 * 0.62, 0.55, f"quintic over {W * 1e3:.0f} ms", color="#047857",
                    fontsize=12, fontweight="bold")
            a2.plot(t * 1e3, du, color=GREEN, lw=2.8)
            a2.text(W * 1e3 * 0.5 + 170, du.max() - 1.0, f"peak {du.max():.2f} m/s³", color="#047857",
                    fontsize=11.5, fontweight="bold", ha="left")
        a1.set_ylabel("m/s²", fontsize=11)
        a1.set_ylim(-0.2, 2.4)
        a1.set_yticks([0, 1, 2])
        a2.set_ylabel("m/s³", fontsize=11)
        a2.set_ylim(0, 10.5)
        a2.set_yticks([0, 5, 10])
        a2.set_xlabel("Time from commit (ms)", fontsize=12)
        a1.set_title("Acceleration command", fontsize=12, loc="left")
        a2.set_title("Its rate (jerk)", fontsize=12, loc="left")
        a2.set_xlim(-150, 750)
        save(fig, f"quintic_step{step}")


# =============================================================================
# The remote-assistant session at 0.75 m/s (SI table 7)
# =============================================================================
def session():
    rows = [(f"Spare at {val('v_slow')} m/s", K.S0, 0.0),
            (f"− takeover {val('t_in')} ms", K.S1, K.S0 - K.S1),
            ("− dispatch to admission", K.S2, K.S1 - K.S2),
            (f"− manual lease, {val('lease_ext')} ms longer", K.S3, K.S2 - K.S3)]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    n = len(rows) + 1
    for i, (lab, left, spent) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, left * 1e3, color=GREEN, height=0.6)
        if spent:
            ax.barh(y, spent * 1e3, left=left * 1e3, color=AMBER_F, edgecolor=AMBER, height=0.6, lw=1.2)
        ax.text(left * 1e3 + (spent * 1e3 if spent else 0) + 10, y, f"{left * 1e3:.1f} mm left",
                va="center", fontsize=12, fontweight="bold", color="#047857")
    ax.barh(0, K.BLEND_RESERVE * 1e3, color=RED_F, edgecolor=RED, hatch="//", height=0.6, lw=1.2)
    ax.text(K.BLEND_RESERVE * 1e3 + 10, 0, f"braking blend needs up to {val('blend_reserve')} mm",
            va="center", fontsize=12, fontweight="bold", color=RED)
    ax.axvline(K.S3 * 1e3, color=RED, lw=1.6, ls="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels(["− base braking blend"] + [r[0] for r in rows][::-1], fontsize=12)
    ax.set_xlim(0, 700)
    ax.set_xlabel("Clearance left at the rack end (mm)", fontsize=12)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "session")


# =============================================================================
# Shared autonomy: averaging two good commands (SI 14.3)
# =============================================================================
def pallet():
    b = []
    b.append(box(195, 120, 170, 90, "#FDE68A", "#B45309", rx=4))
    b.append(text(280, 174, "pallet", 26, "#92400E", weight="bold"))
    b.append(box(238, 322, 84, 52, SLATE_F, SLATE, rx=0))
    b.append(text(280, 356, "robot", 24, SLATE))
    b.append(f'<path d="M280,320 C280,260 140,250 140,170 C140,120 200,80 280,56" fill="none" stroke="{BLUE}" '
             f'stroke-width="4" marker-end="url(#ahb)"/>')
    b.append(f'<path d="M280,320 C280,260 420,250 420,170 C420,120 360,80 280,56" fill="none" stroke="{NAVY}" '
             f'stroke-width="4" stroke-dasharray="9,6" marker-end="url(#ahn)"/>')
    b.append(arrow(280, 320, 280, 216, RED, sw=4, marker="ahr"))
    b += lines(12, 250, ["operator", "goes left"], 25, BLUE, anchor="start", weight="bold", gap=30)
    b += lines(438, 250, ["policy", "goes right"], 25, NAVY, anchor="start", weight="bold", gap=30)
    b += lines(280, 408, ["their average hits the pallet"], 26, RED, weight="bold")
    svg("pallet", 570, 425, b)


# =============================================================================
# Verifier flooding (SI 14.9 pitfall)
# =============================================================================
def flood():
    n = np.arange(0, 61)
    fin = (n + 1) * K.T_ED * 1e3
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.plot(n, fin, color=SLATE, lw=2.6)
    for budget, lab, col, nn in [(K.DL_AISLE * 1e3, f"deadline {val('deadline')} ms", RED, K.N_MISS),
                                 ((K.DL_AISLE - K.T_DA) * 1e3, f"after admission {val('deadline_da')} ms", AMBER, K.N_MISS_DA)]:
        ax.axhline(budget, color=col, lw=2, ls="--")
        ax.text(1 if nn > 30 else 30, budget + 3, lab, color=col, fontsize=11.5, fontweight="bold", va="bottom")
        ax.plot([nn], [(nn + 1) * K.T_ED * 1e3], "o", color=col, ms=8)
        ax.text(nn + 1.5, (nn + 1) * K.T_ED * 1e3 - 4, f"{nn} forged", color=col, fontsize=11.5,
                fontweight="bold", va="top")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Forged signed packets queued ahead of the veto", fontsize=12)
    ax.set_ylabel("Veto verified at (ms)", fontsize=12)
    ax.text(58, 8, f"{val('t_ed')} ms per Ed25519 check\n(illustrative)", ha="right", fontsize=10.5, color=sp.C["neutral"])
    save(fig, "flood")


# =============================================================================
# Mode confusion on split channels (SI 14.6)
# =============================================================================
def channels():
    x0, x1, tmax = 170, 880, 610.0
    X = lambda ms: x0 + (x1 - x0) * ms / tmax  # noqa: E731
    t_in = K.T_IN * 1e3
    for step in (1, 2):
        b = []
        b.append(text(x0, 34, f"Takeover window {t_in:.0f} ms; the spare at 1.3 m/s is {K.SPARE * 1e3:.0f} mm",
                      22, SLATE, anchor="start"))
        for lab, y in [("Arm", 96), ("Base", 182), ("Console", 268)]:
            b.append(text(x0 - 12, y, lab, 24, TXT, anchor="end", weight="bold"))
        b.append(box(X(0), 66, X(t_in) - X(0), 46, BLUE_F, BLUE, rx=3))
        b.append(text((X(0) + X(t_in)) / 2, 97, "operator aligns the gripper (MAN)", 23, "#1D4ED8", weight="bold"))
        b.append(box(X(0), 152, X(t_in) - X(0), 46, RED_F, RED, rx=3))
        b.append(text((X(0) + X(t_in)) / 2, 183, f"policy still drives at {val('v_aisle')} m/s, {val('travel_in')} mm",
                      23, RED, weight="bold"))
        if step == 1:
            b.append(box(X(0), 238, X(t_in) - X(0), 46, SLATE_F, GRAY, rx=3))
            b.append(text((X(0) + X(t_in)) / 2, 269, "ARM: MAN   (base not shown)", 23, SLATE, italic=True))
        else:
            b.append(box(X(0), 238, X(t_in) - X(0), 46, GREEN_F, GREEN, rx=3))
            b.append(text((X(0) + X(t_in)) / 2, 269, "ARM: MAN   |   BASE: AUTO, MOVING", 23, "#047857",
                          weight="bold"))
            b.append(text(20, 326, "Announce every channel's owner, or commit the untransferred", 23, "#047857",
                          anchor="start", weight="bold"))
            b.append(text(20, 356, "channel into a certified hold with the brake on.", 23, "#047857",
                          anchor="start", weight="bold"))
        svg(f"channels_step{step}", 900, 370, b)


# =============================================================================
# Lufthansa 2904 (SI War Story 14.1)
# =============================================================================
def lh2904():
    x0, x1, tmax = 270, 880, 15.0
    X = lambda s: x0 + (x1 - x0) * s / tmax  # noqa: E731
    G, B = num("lh_gear"), num("lh_brake")  # left gear contact; wheel brakes act
    rows = [("Right gear", 40), ("Left gear", 102), ("Crew", 164), ("Spoilers, reversers", 226), ("Wheel brakes", 288)]
    b = []
    for name, y in rows:
        b.append(text(x0 - 12, y + 30, name, 24, TXT, anchor="end", weight="bold"))
    b.append(box(X(0), 44, X(15) - X(0), 36, GREEN_F, GREEN, rx=3))
    b.append(text(X(7.5), 71, "on the runway", 23, "#047857"))
    b.append(box(X(0), 106, X(G) - X(0), 36, "white", GRAY, dash="6,4", rx=3))
    b.append(text(X(4.5), 133, "not compressed", 23, SLATE, italic=True))
    b.append(box(X(G), 106, X(15) - X(G), 36, GREEN_F, GREEN, rx=3))
    b.append(box(X(0), 168, X(15) - X(0), 36, BLUE_F, BLUE, rx=3))
    b.append(text(X(7.5), 195, "brakes and reversers commanded", 23, "#1D4ED8", weight="bold"))
    b.append(box(X(0), 230, X(G) - X(0), 36, RED_F, RED, rx=3))
    b.append(text(X(4.5), 257, "locked out", 23, RED, weight="bold"))
    b.append(box(X(G), 230, X(15) - X(G), 36, GREEN_F, GREEN, rx=3))
    b.append(box(X(0), 292, X(B) - X(0), 36, RED_F, RED, rx=3))
    b.append(text(X(6.5), 319, "no braking, aquaplaning", 23, RED, weight="bold"))
    b.append(box(X(B), 292, X(15) - X(B), 36, GREEN_F, GREEN, rx=3))
    ay = 352
    b.append(line(x0, ay, x1, ay, SLATE, 2))
    for s, lab in [(0, "0"), (3, "3"), (6, "6"), (G, val("lh_gear")), (B, val("lh_brake")), (15, "15 s")]:
        b.append(line(X(s), ay, X(s), ay + 8, SLATE, 2))
        b.append(text(X(s), ay + 32, lab, 23, RED if s == B else SLATE, weight="bold" if s == B else "normal"))
    b.append(line(X(G), 30, X(G), ay, RED, 2, dash="5,4"))
    b.append(line(X(B), 286, X(B), ay, RED, 2, dash="5,4"))
    b.append(text(X(G), 24, f"{val('lh_gear')} s", 23, RED, weight="bold"))
    svg("lh2904", 900, 392, b)


# =============================================================================
# Causal intervention masks (SI 14.8.3)
# =============================================================================
def masks():
    cells = [("drift", "policy drifts", f"{val('drift_s')} s", "M = 0", RED_F, RED),
             ("blend", "enforcer spline", f"{val('arm_ticks')} ms", "M = 0", GREEN_F, GREEN),
             ("re-orient", "operator hunts", "until settled", "M = 0", AMBER_F, AMBER),
             ("stable", "operator steady", "after a dwell", "M = 1", BLUE_F, BLUE)]
    xs = [10, 233, 456, 679]
    w = 211
    stamps = ["divergence", "commit", "blend end", "stabilized"]
    for step in (1, 2):
        b = []
        b.append(text(10, 30, "MCU timestamps split one takeover into four intervals", 23, TXT, anchor="start",
                      weight="bold"))
        for (name, who, dur, m, f, s), x in zip(cells, xs):
            b.append(box(x, 96, w, 106, f if name == "stable" else "white", s, sw=3))
            b.append(text(x + w / 2, 132, name, 25, s, weight="bold"))
            b.append(text(x + w / 2, 162, who, 21, SLATE))
            b.append(text(x + w / 2, 188, dur, 21, SLATE, italic=True))
            b.append(text(x + w / 2, 236, m, 25, TXT if m == "M = 1" else SLATE, weight="bold"))
        for x, st in zip(xs, stamps):
            xx = x if x == 10 else x - 6
            b.append(line(xx, 70, xx, 214, SLATE, 2, dash="4,4"))
            b.append(text(xx, 64, st, 21, SLATE, anchor="start" if x == 10 else "middle"))
        b += bracket(679, 890, 262, "trains", BLUE, 22, up=False)
        if step == 2:
            wx = 10 + w * (K.T_OOL - num("sirius_s")) / K.T_OOL
            b.append(box(wx, 194, 10 + w - wx, 12, AMBER, AMBER, rx=2, sw=1))
            b.append(text(10 + w - 6, 188, f"{val('sirius_s')} s", 21, "#92400E", anchor="end", weight="bold"))
            b.append(text(10, 326, f"A fixed {val('sirius_s')} s window (Sirius) masks the last {val('sirius_s')} s of a {val('drift_s')} s drift;", 22,
                          "#92400E", anchor="start"))
            b.append(text(10, 354, f"the first {val('sirius_left_s')} s stay in the data as if they were good.", 22,
                          "#92400E", anchor="start"))
        svg(f"masks_step{step}", 900, 365, b)


def main():
    dmv()
    propose_admit()
    strip()
    zhang()
    deadline()
    leases()
    fsm()
    handshake()
    quintic()
    session()
    pallet()
    flood()
    channels()
    lh2904()
    masks()
    print("\nfigures written to", HERE)


if __name__ == "__main__":
    main()
