#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 7 (Closed-Loop Evaluation), Volume IV.

Every value comes from MLSysIM through calcs.py: the scenario objects for the
plotted curves and val()/num() for every printed number (the chapter, Closed-
Loop Evaluation, reads the same scenarios; Mandlekar et al. 2021, RAND RR-1478
Table 1, and Kusano et al. 2025 are Literature entries). Diagrams are SVG
drawn at the size of their slot (full width 900 x 500, labels 20 px or more);
plots use the shared slideplot helper. Red marks only limits and hazards.

Run from this folder: python3 make_figs.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

CAMPAIGN, DRIFT, CONTACT, SPRT = calcs.CAMPAIGN, calcs.DRIFT, calcs.CONTACT, calcs.SPRT
WALL, ROAD, TAIL, AISLE, LIT = calcs.WALL, calcs.ROAD, calcs.TAIL, calcs.AISLE, calcs.LIT
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def per_hour_label(exponent_key):
    """'10⁻⁴ per h' from a recorded power of ten."""
    return f"10{val(exponent_key).translate(SUPERSCRIPT)} per h"

plt = sp.plt
C = sp.C
OUT = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"

# Volume IV causal-boundary roles (volumes/vol4/palette.md)
PROPOSAL = "#2563EB"
NERVOUS = "#059669"
NERVOUS_L = "#DCFCE7"
PLANT = "#334155"
HAZARD = "#DC2626"
HAZARD_L = "#FEE2E2"
GRAY = "#94A3B8"
GRAY_L = "#F1F5F9"
TEXT = "#1A1A2E"


def svg(path, w, h, body):
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, size=20, fill=TEXT, anchor="middle", weight="normal", opacity=1.0):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" opacity="{opacity}">{s}</text>')


def rect(x, y, w, h, fill, stroke, sw=2, rx=6, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')


def line(x1, y1, x2, y2, stroke, sw=2.5, dash=None, marker=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def arrow_defs(color):
    return (f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{color}"/></marker></defs>')


# 1. Scenario coverage of latch-bc-01 (ch §7.8): plate positions x approach speeds
#    (Scenarios.LatchScenarioTable), one cell with demonstrations and the clean runs.
def coverage(step):
    w, h = 600, 470
    b = []
    cols = ["Plate −", "Nominal", "Plate +"]
    rows = [f"{calcs.m(sp_, 'm/s'):.2f} m/s" for sp_ in calcs.TABLE.speeds]
    x0, y0, cw, ch = 140, 52, 152, 128
    for j, c in enumerate(cols):
        b.append(text(x0 + j * cw + cw / 2, y0 - 14, c, 26, weight="bold"))
    for i, r in enumerate(rows):
        b.append(text(x0 - 14, y0 + i * ch + ch / 2 + 9, r, 26, anchor="end", weight="bold"))
        for j in range(3):
            supported = (i == 0 and j == 1)
            fill = NERVOUS_L if supported else GRAY_L
            stroke = NERVOUS if supported else GRAY
            b.append(rect(x0 + j * cw + 5, y0 + i * ch + 5, cw - 10, ch - 10, fill, stroke, 2.5))
            if supported:
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + 50, "demos", 26, NERVOUS, weight="bold"))
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + 90, f"{val('n_clean')} runs", 26, TEXT, weight="bold",
                              opacity=1.0 if step >= 2 else 0.0))
            else:
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + 73, "no data", 24, GRAY))
    a = 1.0 if step >= 2 else 0.0
    b.append(text(w / 2, 370, "Base stopped, empty rack, Unit 01", 26, TEXT, opacity=a))
    b.append(text(w / 2, 420, f"{val('absences_words').capitalize()} cells and the loaded", 26, HAZARD, weight="bold", opacity=a))
    b.append(text(w / 2, 455, "machine: no evidence", 26, HAZARD, weight="bold", opacity=a))
    svg(OUT / f"coverage_step{step}.svg", w, h, b)


# 2. Offline illusion, published: Mandlekar et al. 2021, Appendix G
def mandlekar():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    tasks = ["Square", "Transport"]
    xs = np.arange(len(tasks))
    best = [num("sq_best"), num("tr_best")]
    low = [num("sq_low"), num("tr_low")]
    ax.bar(xs - 0.2, best, 0.38, color=C["data"])
    ax.bar(xs + 0.2, low, 0.38, color=C["neutral"])
    for x, v in zip(xs, best):
        ax.text(x - 0.2, v + 2, f"{v:g}%", ha="center", fontsize=13, fontweight="bold", color=C["data"])
    for x, v in zip(xs, low):
        ax.text(x + 0.2, v + 2, f"{v:g}%", ha="center", fontsize=13, fontweight="bold", color=C["neutral"])
    ax.set_xticks(xs)
    ax.set_xticklabels(tasks, fontsize=13)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.text(-0.45, 114, "Top checkpoint", color=C["data"], fontsize=12, fontweight="bold")
    ax.text(-0.45, 103, "Lowest validation loss", color=C["neutral"], fontsize=12, fontweight="bold")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / "mandlekar_bars.svg")


# 3. Aisle drift (ch §7.3.1, equation 2)
def aisle(step):
    rad = DRIFT.heading
    d_hit = calcs.m(DRIFT.distance_to_clearance, "m")
    aisle_m = calcs.m(AISLE.aisle_length, "m")
    clear_cm = calcs.m(AISLE.side_clearance, "cm")
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    d = np.linspace(0, aisle_m, 300)
    ax.axhline(clear_cm, color=C["danger"], lw=2, ls="--")
    ax.text(29.6, clear_cm + 0.6, f"rack: {val('clear_cm')} cm side clearance", ha="right",
            color=C["danger"], fontsize=12, fontweight="bold")
    ax.plot(d, np.zeros_like(d), color=C["memory"], lw=3)
    ax.text(29.6, 1.6, f"replay: reset each step, MSE {val('mse_m')}×10{val('mse_e').translate(SUPERSCRIPT)} rad²",
            ha="right", color=C["memory"], fontsize=12, fontweight="bold")
    a = 1.0 if step >= 2 else 0.0
    dd = d[d <= d_hit]
    ax.plot(dd, dd * math.tan(rad) * 100, color=C["compute"], lw=3, alpha=a)
    ax.plot([d_hit], [clear_cm], "o", ms=10, color=C["danger"], alpha=a)
    ax.text(d_hit + 0.6, 11.2, f"closed loop: {val('t_hit')} s, {val('d_hit')} m", color=C["compute"],
            fontsize=12, fontweight="bold", alpha=a)
    ax.set_xlim(0, aisle_m)
    ax.set_ylim(-1, 19)
    ax.set_xlabel(f"Distance along the {val('aisle_m')} m aisle (m)")
    ax.set_ylabel("Lateral offset (cm)")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / f"aisle_step{step}.svg")


# 4. Contact: F = k * delta (ch §7.3.2)
def contact():
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    k = calcs.m(CONTACT.stiffness, "N/m")
    dm = np.logspace(-3, math.log10(3), 100)  # mm
    ax.plot(dm, k * dm / 1000, color=C["compute"], lw=3)
    pts = [(num("f_trip"), f"{val('f_trip')} N tripwire at {val('trip_pen_mm')} mm", C["danger"]),
           (num("f_lim"), f"{val('f_lim')} N latch limit at {val('lim_pen_mm')} mm", C["danger"]),
           (num("residual_n"), f"{val('residual_mm')} mm residual: {val('residual_n')} N", C["compute"])]
    for f, lab, col in pts:
        x = f / k * 1000
        ax.plot([x], [f], "o", ms=9, color=col)
        ax.annotate(lab, (x, f),
                    xytext=(10, -16) if f < 1000 else (-10, 6),
                    textcoords="offset points", ha="left" if f < 1000 else "right", fontsize=12, fontweight="bold", color=col)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 4)
    ax.set_ylim(0.4, 4000)
    ax.set_xlabel("Penetration into the latch (mm)")
    ax.set_ylabel("Reaction force (N)")
    ax.text(0.0012, 1500, f"stiffness {val('k_contact_m')}×10{val('k_contact_e').translate(SUPERSCRIPT)} N/m",
            fontsize=12, color=C["neutral"])
    ax.grid(color=C["grid"], lw=0.5)
    sp.save(fig, OUT / "contact_force.svg")


# 5. Four evidence regimes (ch §7.4, table 2), built one quadrant per step
REGIMES = [  # (col, row, name, shows, blind)
    (0, 1, "Recorded replay", "Latency, memory, numerics", "Drift: errors never feed back"),
    (1, 1, "Closed-loop simulation", "Recovery, reachability, scale", "Real contact, slip, sensors"),
    (0, 0, "Shadow operation", "Real buses, heat, sensor noise", "Recovery: baseline drives"),
    (1, 0, "Constrained physical test", "Contact, friction, compliance", "Beyond the safety boundary"),
]


def regimes(step):
    w, h = 900, 420
    b = [arrow_defs(PLANT)]
    x0, y0, cw, ch = 130, 8, 375, 172
    b.append(line(x0 - 30, y0 + 2 * ch + 10, x0 - 30, y0 + 5, PLANT, 2.5, marker=True))
    b.append(line(x0 - 30, y0 + 2 * ch + 14, x0 + 2 * cw + 10, y0 + 2 * ch + 14, PLANT, 2.5, marker=True))
    b.append(text(x0 + cw, y0 + 2 * ch + 46, "Causal feedback: open loop  →  closed loop", 21, PLANT, weight="bold"))
    b.append(f'<g transform="translate({x0 - 52},{y0 + ch}) rotate(-90)">'
             + text(0, 0, "Physical realism: logged → live", 21, PLANT, weight="bold") + "</g>")
    for k, (c, r, name, shows, blind) in enumerate(REGIMES):
        shown = k < step
        x, y = x0 + c * cw + 8, y0 + r * ch + 8
        live = r == 0
        b.append(rect(x, y, cw - 16, ch - 16, "#F8FAFC" if live else "white",
                      PLANT if shown else "#E2E8F0", 2.5, rx=8,
                      dash="8,5" if (shown and k == 3) else None))
        if shown:
            if k == 3:
                b.append(rect(x + 3, y + 3, cw - 22, ch - 22, "none", HAZARD, 2.5, rx=8, dash="8,6"))
            b.append(text(x + (cw - 16) / 2, y + 32, name, 24, TEXT, weight="bold"))
            b.append(text(x + 18, y + 64, "Shows:", 21, NERVOUS, anchor="start", weight="bold"))
            b.append(text(x + 18, y + 88, shows, 21, TEXT, anchor="start"))
            b.append(text(x + 18, y + 116, "Blind to:", 21, HAZARD, anchor="start", weight="bold"))
            b.append(text(x + 18, y + 140, blind, 21, TEXT, anchor="start"))
    svg(OUT / f"regimes_step{step}.svg", w, h, b)


# 6. Inverting a clean record (ch §7.6, figure 3)
def inversion(step):
    lb, n, alpha = CAMPAIGN.lower_bound, CAMPAIGN.clean_runs, CAMPAIGN.alpha
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    p = np.linspace(0.70, 1.0, 400)
    ax.plot(p, p ** n, color=C["memory"], lw=3)
    ax.axhline(alpha, color=C["neutral"], lw=1.5, ls="--")
    ax.text(0.702, alpha * 1.25, r"$\alpha_{\rm sig}$ = " + val("alpha"), color=C["neutral"], fontsize=12, fontweight="bold")
    a = 1.0 if step >= 2 else 0.0
    ax.axvspan(0.70, lb, color=C["neutral_l"], alpha=0.6 * a, lw=0)
    ax.plot([lb], [alpha], "o", ms=10, color=C["danger"], alpha=a)
    ax.text(0.705, 0.25, f"ruled out: p < {val('lb')}", ha="left", fontsize=12,
            fontweight="bold", color=C["danger"], alpha=a)
    q = CAMPAIGN.weak_policy_success
    ax.plot([q], [q ** n], "o", ms=10, color=C["compute"], alpha=a)
    ax.annotate(f"fails 1 run in {CAMPAIGN.weak_policy_one_in}: clean record in {val('weak_clean_pct')}% of campaigns",
                (q, q ** n), xytext=(0.882, 0.0016), textcoords="data", fontsize=12,
                arrowprops=dict(arrowstyle="-", color=C["compute"], lw=1.2, alpha=a),
                fontweight="bold", color=C["compute"], alpha=a)
    ax.set_yscale("log")
    ax.set_xlim(0.70, 1.0)
    ax.set_ylim(5e-4, 1.5)
    ax.set_xlabel("True per-run success probability p")
    ax.set_ylabel(f"P({val('n_clean')} clean runs)")
    ax.grid(color=C["grid"], lw=0.5)
    sp.save(fig, OUT / f"inversion_step{step}.svg")


# 7. The nines ladder (ch Napkin Math 7.1, 7.2; table 3)
def nines():
    rows = [(f"{val('lb_pct')}%", CAMPAIGN.clean_runs, "neutral",
             f"{val('n_clean')} runs · {val('opening_min')} min")]
    for target, key in zip(CAMPAIGN.nine_targets, ("90", "99", "999")):
        n = CAMPAIGN.runs_for(target)
        hrs = calcs.m(CAMPAIGN.campaign_time(n), "hour")
        tl = f"{hrs * 60:.1f} min" if hrs < 1 else f"{hrs:.1f} h"
        rows.append((f"{val(f't{key}_pct')}%", n, "memory", f"{val(f'n{key}')} runs · {tl}"))
    sp.hbars_steps(str(OUT / "nines"), rows, "runs", xlabel="Clean runs needed (log scale)",
                   size=(5.8, 2.3))


# 8. Sequential test corridor (ch §7.7.3)
def sprt():
    up, down, a, bnd = SPRT.success_increment, SPRT.failure_increment, SPRT.accept, SPRT.reject
    n_acc, traced = SPRT.runs_to_accept, SPRT.trace_score
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    n = np.arange(0, n_acc + 1)
    ax.step(n, n * up, where="post", color=C["data"], lw=3)
    tr = [k * up for k in range(SPRT.trace_clean_runs + 1)] + [traced]
    ax.plot([len(tr) - 2, len(tr) - 1], tr[-2:], color=C["compute"], lw=2.5)
    ax.plot([len(tr) - 1], [traced], "o", ms=8, color=C["compute"])
    ax.axhline(a, color=C["data"], lw=1.8, ls="--")
    ax.axhline(bnd, color=C["danger"], lw=1.8, ls="--")
    ax.text(0.5, a + 0.25, f"accept at {val('sprt_accept')}", color=C["data"], fontsize=12, fontweight="bold")
    ax.text(0.5, bnd + 0.25, f"reject at -{val('sprt_reject')}", color=C["danger"], fontsize=12, fontweight="bold")
    ax.annotate(f"{val('sprt_n_accept')} clean runs", (n_acc, n_acc * up), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=12, fontweight="bold", color=C["data"])
    ax.annotate(f"{val('sprt_trace_n')} clean, 1 failure: {val('sprt_trace')}", (len(tr) - 1, traced), xytext=(8, -18),
                textcoords="offset points", fontsize=12, fontweight="bold", color=C["compute"])
    ax.set_xlim(0, 34)
    ax.set_ylim(-3, 3.8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Log-likelihood ratio")
    ax.grid(color=C["grid"], lw=0.5)
    sp.save(fig, OUT / "sprt.svg")


# 9. The exposure wall (ch Napkin Math 7.3, figure in §7.9.1)
def exposure():
    door = calcs.m(CAMPAIGN.campaign_time(CAMPAIGN.target_runs), "hour")
    hours = [calcs.m(WALL.exposure(r), "hour") for r in WALL.rates]
    rows = [(f"Door, {val('t999_pct')}%", door, "neutral", f"{val('h999')} h"),
            (per_hour_label("site_e"), hours[0], "memory", f"{val('site_h')} h"),
            (per_hour_label("e6_e"), hours[1], "memory", f"{val('e6_mh')} million h"),
            (per_hour_label("e9_e"), hours[2], "danger", f"{val('e9_bh')} billion h")]
    sp.hbars_steps(str(OUT / "exposure"), rows, "h", xlabel="Failure-free hours needed (log scale)",
                   size=(5.8, 2.3), vlines=[(calcs.m(WALL.rig_time, "hour"), "one rig-year", "data")])


# 10. RAND requirements against Waymo's rider-only miles
def rand():
    rows = [("Fatalities", float(ROAD.zero_failure["fatalities"]), "danger", f"{val('rand_fatal_m')}M"),
            ("Injuries", float(ROAD.zero_failure["injuries"]), "memory", f"{val('rand_injury_m')}M"),
            ("Crashes", float(ROAD.zero_failure["crashes"]), "memory", f"{val('rand_crash_m')}M")]
    sp.hbars(str(OUT / "rand_waymo.svg"), rows, "M", xlabel="Zero-event miles to bound the human rate (millions)",
             size=(4.8, 2.1), vlines=[(float(ROAD.fleet_miles), f"Waymo {val('waymo_m')}M", "data")])


# 11. Mean against peak latch force (ch §7.9.2)
def mean_peak():
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    rows = [("Mean", num("f_mean"), "compute"), ("99th percentile", num("f_p99"), "danger")]
    for i, (lab, v, ck) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=C[ck], height=0.6)
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=C["text"])
        ax.text(v + 2 if v > 50 else v + 5.5, y, f"{v:g} N", va="center",
                fontsize=13, fontweight="bold", color=C[ck])
    for x, lab, ck in [(num("f_trip"), f"{val('f_trip')} N tripwire", "neutral"),
                       (num("f_lim"), f"{val('f_lim')} N limit", "danger")]:
        ax.axvline(x, color=C[ck], lw=1.5, ls="--")
        ax.text(x + 1, 1.45, lab, color=C[ck], fontsize=11, fontweight="bold", va="bottom")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 140)
    ax.set_ylim(-0.5, 1.8)
    ax.set_xlabel(f"Latch contact force (N), {TAIL.trials} trials")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / "mean_peak.svg")


# 12. The evaluation record and its runtime obligations (ch §7.8, table 5; §7.9.3)
FIELDS = [("target_odd_boundary", f"nominal plate, {val('v_guard')} m/s, base stopped"),
          ("component_hashes", "manifest digest, weights, Unit 01"),
          ("sampling_protocol", "draws, seed, reset, independence"),
          ("complete_outcome_log", "every run: task, autonomy, protection"),
          ("statistical_evidence", f"{val('n_clean')} clean runs: failure ≤ {val('fail_up_pct')}% ({val('conf_pct')}%)"),
          ("unsupported_gap_catalog", f"{calcs.TABLE.absences} cells, payload, other units"),
          ("runtime_monitor_specs", "signal, threshold, tested response")]
MONITORS = ["Input support", "Latency and freshness", "Constraint margin",
            "Intervention rate", "Environmental assumption"]


def record(step):
    w, h = 900, 470
    b = [arrow_defs(NERVOUS)]
    x, y, cw = 6, 6, 548
    b.append(rect(x, y, cw, 458, "white", PLANT, 2.5, rx=8))
    b.append(text(x + cw / 2, y + 34, "Evaluation record: latch-bc-01", 25, TEXT, weight="bold"))
    for i, (f, v) in enumerate(FIELDS):
        yy = y + 50 + i * 57
        hot = step >= 2 and i >= 5
        b.append(rect(x + 10, yy, cw - 20, 52, HAZARD_L if hot else GRAY_L,
                      HAZARD if hot else "#CBD5E1", 1.5, rx=5))
        b.append(text(x + 20, yy + 23, f.replace("_", " "), 22, TEXT, anchor="start", weight="bold"))
        b.append(text(x + 20, yy + 46, v, 21, "#334155", anchor="start"))
    a = 1.0 if step >= 2 else 0.0
    mx, mw = 612, 282
    b.append(f'<g opacity="{a}">')
    b.append(text(mx + mw / 2, 30, "Runtime monitors", 24, NERVOUS, weight="bold"))
    for i, m in enumerate(MONITORS):
        yy = 46 + i * 64
        b.append(rect(mx, yy, mw, 52, NERVOUS_L, NERVOUS, 2, rx=6))
        b.append(text(mx + mw / 2, yy + 34, m, 22, TEXT, weight="bold"))
    b.append(line(x + cw + 4, 410, mx - 6, 220, NERVOUS, 2.5, marker=True))
    b.append(rect(mx, 382, mw, 80, HAZARD_L, HAZARD, 2, rx=6))
    b.append(text(mx + mw / 2, 414, "Cannot be sensed:", 22, HAZARD, weight="bold"))
    b.append(text(mx + mw / 2, 444, "hard limit or barrier", 22, TEXT))
    b.append("</g>")
    svg(OUT / f"record_step{step}.svg", w, h, b)


def main():
    for s in (1, 2):
        coverage(s)
        aisle(s)
        inversion(s)
        record(s)
    for s in (1, 2, 3, 4):
        regimes(s)
    mandlekar()
    contact()
    nines()
    sprt()
    exposure()
    rand()
    mean_peak()
    print("  figures written to", OUT)


if __name__ == "__main__":
    main()
