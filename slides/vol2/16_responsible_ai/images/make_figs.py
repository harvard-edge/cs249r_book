#!/usr/bin/env python3
"""Figures for Lecture 16 (Volume II): Responsible AI.

Every number drawn comes from calcs.py (MLSysIM, via deckvals) or is a sourced
input listed here. "Ch." is the rendered chapter (Volume II, Chapter 16).

  - Ofqual, Grading gaps in summer 2020 (July 2021), Table 1 text: A level
    entries with the calculated grade equal to the CAG 59%, lower 39%,
    higher "just over 2%" (n = 457,420 entries).
  - Loan model (Ch. Napkin Math 16.1, 16.2; rates in calcs.py).
  - COMPAS: recomputed from ProPublica's compas-scores-two-years.csv with
    ProPublica's filters (counts and rates in calcs.py).
  - Dual thresholds: illustrative scores in calcs.py (after Ch. figure 5).
  - Obermeyer et al. 2019 (Ch. footnote 7): 17.7 and 46.5 percent.
  - z-test: Ch. 16.5.2 formula, computed in calcs.py.
  - Gender Shades: mlsysim literature series gender_shades_audit
    (Buolamwini and Gebru 2018, Table 4; Raji and Buolamwini 2019, Table 1).
  - Monitoring scenario: Ch. 16.5.5 (calcs.py).
  - Overheads: Ch. table 5 (scenario ranges, not measurements).
  - SHAP subsets: Ch. 16.7.2; 1 us per subset from the Ch. 16.7 self-check.
  - DP-SGD accuracy: Abadi et al., CCS 2016, Sec. 5.2 (MNIST) and Fig. 6
    (CIFAR-10), delta = 1e-5.
  - Unlearning: Ch. Napkin Math 16.6 (calcs.py).
  - Automation bias: Ch. Napkin Math 16.7 (calcs.py).
  - Waymo: mlsysim series waymo_rider_only_miles and
    waymo_vs_human_benchmark (Waymo Safety Impact Data Hub, through June 2026).

Run from this folder: python3 make_figs.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import contextlib  # noqa: E402
import io  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import slideplot as sp  # noqa: E402

C = sp.C
# Volume II C3 roles and structural colors (volumes/vol2/palette.md)
ETH = "#1F407A"
ETH_L = "#EAF1F8"
AMBER = "#d97706"
AMBER_L = "#fef3c7"
VIOLET = "#7c3aed"
VIOLET_L = "#f3e8ff"
GREEN = "#16a34a"
GREEN_L = "#dcfce7"
RED = "#dc2626"
RED_L = "#fee2e2"
FRAME = "#94a3b8"
FRAME_L = "#f8fafc"
INK = "#0f172a"
GRAY = "#64748b"
FONT = "Helvetica, Arial, sans-serif"


def out(name):
    return str(HERE / name)


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------
def svg_doc(body, w=900, h=500):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<defs>{arrow_defs()}</defs>\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n{body}\n</svg>\n')


def arrow_defs():
    s = ""
    for nm, col in (("blue", ETH), ("amber", AMBER), ("purple", VIOLET), ("green", GREEN),
                    ("red", RED), ("gray", GRAY)):
        s += (f'<marker id="arr-{nm}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
              f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
              f'fill="{col}"/></marker>')
    return s


def box(x, y, w, h, fill, stroke, lines, size=20, color=INK, bold_first=True, rx=6, dash=None,
        sw=2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="{sw}"{d}/>\n')
    n = len(lines)
    lh = size * 1.25
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, t in enumerate(lines):
        wt = "bold" if (i == 0 and bold_first) else "normal"
        s += (f'<text x="{x + w / 2}" y="{y0 + i * lh:.1f}" text-anchor="middle" font-size="{size}" '
              f'font-weight="{wt}" fill="{color}">{t}</text>\n')
    return s


def text(x, y, t, size=20, color=INK, anchor="middle", weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}"{st}>{t}</text>\n')


def arrow(x1, y1, x2, y2, col="gray", w=2.5, dash=None):
    colhex = {"blue": ETH, "amber": AMBER, "purple": VIOLET, "green": GREEN, "red": RED,
              "gray": GRAY}[col]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colhex}" stroke-width="{w}" '
            f'marker-end="url(#arr-{col})"{d}/>\n')


def path_arrow(dpath, col="gray", w=2.5, dash=None):
    colhex = {"blue": ETH, "amber": AMBER, "purple": VIOLET, "green": GREEN, "red": RED,
              "gray": GRAY}[col]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{dpath}" fill="none" stroke="{colhex}" stroke-width="{w}" '
            f'marker-end="url(#arr-{col})"{d}/>\n')


def write(name, body, w=900, h=500):
    Path(out(name)).write_text(svg_doc(body, w, h))


plt.rcParams.update({"xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "axes.labelsize": 13})


# ---------------------------------------------------------------------------
# Hook: Ofqual entry-level outcome (Grading gaps, Table 1 text)
# ---------------------------------------------------------------------------
def hook():
    rows = [("Lower than teacher grade", num("ofq_lower_pct"), "danger", f"{val('ofq_lower_pct')}%"),
            ("Same as teacher grade", num("ofq_same_pct"), "neutral", f"{val('ofq_same_pct')}%"),
            ("Higher than teacher grade", num("ofq_higher_pct"), "data", f"about {val('ofq_higher_pct')}%")]
    for k, step in ((1, 0), (2, 3)):
        sp.hbars(out(f"hook_step{k}.svg"), rows, "%", log=False, step=step,
                 xlabel=f"Share of {val('ofq_entries')} A level entries (%)", size=(5.2, 1.8))


def ofqual_flows():
    """Ofqual Figure 1 redrawn from its Table 1: for each teacher grade (CAG),
    the share of entries whose calculated grade was higher, the same, one
    grade lower, or two or more lower. Column slot about 7 x 5.3 cm."""
    split = calcs.OFQ_SPLIT
    grades = calcs.OFQ_GRADES
    cols = [("higher", C["data"]), ("same", C["neutral_l"]), ("one lower", RED_L),
            ("two or more lower", RED)]
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    for i, s in enumerate(split):
        y = len(split) - 1 - i
        left = 0
        for (name, col), v in zip(cols, s):
            ax.barh(y, v, left=left, height=0.7, color=col, edgecolor="white", lw=0.8)
            left += v
        lower = s[2] + s[3]
        ax.text(101.5, y, f"{lower:.0f}%", va="center", ha="left", fontsize=12,
                fontweight="bold", color=RED if lower else GRAY)
    ax.text(101.5, len(split) - 0.35, "lower", va="bottom", ha="left", fontsize=12,
            color=RED)
    top = len(split) - 0.35
    ax.text(0, top, "higher", ha="left", va="bottom", fontsize=12, color=C["data"])
    ax.text(36, top, "same", ha="center", va="bottom", fontsize=12, color=GRAY)
    ax.text(80, top, "one lower", ha="center", va="bottom", fontsize=12, color=RED,
            fontweight="bold")
    ax.set_yticks(range(len(split)))
    ax.set_yticklabels(list(reversed(grades)), fontsize=12)
    ax.set_ylabel("Teacher grade", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(split) + 0.3)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Calculated grade, share of entries (%)", fontsize=12)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    sp.save(fig, out("ofqual_flows.svg"))


def ofqual_candidates():
    """Ofqual Table 2 as bars: candidates by the number of A levels lowered."""
    n = calcs.OFQ_T2
    pct = [100 * v / calcs.OFQ.total_candidates for v in n]
    fig, ax = plt.subplots(figsize=(4.4, 1.8))
    for k, p in enumerate(pct):
        ax.bar(k, p, width=0.7, color=GRAY if k == 0 else RED)
        ax.text(k, p + 1.5, f"{p:.0f}%" if p >= 0.5 else "<0.5%", ha="center",
                va="bottom", fontsize=12, color=GRAY if k == 0 else RED, fontweight="bold")
    at_least = num("ofq_any_pct_1")
    ax.plot([0.7, 4.3], [52, 52], color=RED, lw=1.5)
    for x in (0.7, 4.3):
        ax.plot([x, x], [49, 52], color=RED, lw=1.5)
    ax.text(2.5, 54, f"{at_least:.0f}% had at least one lowered", ha="center",
            va="bottom", fontsize=12, fontweight="bold", color=RED)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=12)
    ax.set_xlabel("A levels lowered per student", fontsize=12)
    ax.set_ylim(0, 66)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    sp.save(fig, out("ofqual_candidates.svg"))


# ---------------------------------------------------------------------------
# Segment 1: the assurance case, in four steps (canvas 900 x 430)
# ---------------------------------------------------------------------------
def assurance():
    def frame(k):
        b = text(450, 30, "One stated use, one operating envelope", 22, GRAY, weight="bold")
        ev = ["Subgroup results", "Audit reports", "Decision logs", "Documentation"]
        for i, t in enumerate(ev):
            b += box(10, 52 + i * 58, 200, 48, GREEN_L, GREEN, [t], size=22, bold_first=False)
        b += text(110, 300, "Evidence", 23, GREEN, weight="bold")
        b += path_arrow("M210,170 L238,170", "green")
        b += box(240, 120, 180, 100, FRAME_L, FRAME, ["Argument", "why the", "evidence holds"],
                 size=22)
        b += path_arrow("M420,170 L448,170", "blue")
        b += box(450, 110, 190, 120, ETH_L, ETH, ["Claim", "loan screening", "meets criteria"],
                 size=22)
        if k >= 2:
            b += box(240, 262, 400, 64, "white", GRAY, ["Assumptions:", "population, setting, suppliers"],
                     size=22, bold_first=True, dash="7,5")
        if k >= 3:
            b += path_arrow("M640,150 L676,110", "purple")
            b += box(678, 52, 212, 80, VIOLET_L, VIOLET, ["Owner", "can block launch"], size=22)
            b += path_arrow("M784,132 L784,170", "purple")
            b += box(678, 172, 212, 100, VIOLET_L, VIOLET, ["Decision", "launch, restrict,", "or reject"],
                     size=22)
        if k >= 4:
            b += box(240, 352, 200, 64, "white", GRAY, ["Material change"], size=22)
            b += path_arrow("M440,384 L478,384", "red")
            b += box(480, 352, 200, 64, RED_L, RED, ["Invalidation rule"], size=22)
            b += path_arrow("M680,384 C760,384 790,340 784,276", "red", dash="8,5")
            b += text(800, 330, "approval", 22, RED, anchor="start", weight="bold")
            b += text(800, 356, "expires", 22, RED, anchor="start", weight="bold")
        write(f"assure_step{k}.svg", b, 900, 430)
    for k in range(1, 5):
        frame(k)


def lifecycle():
    stages = ["Context", "Risk", "Controls", "Evaluate", "Launch", "Operate", "Change", "Retire"]
    b = ""
    w, gap, x0, y = 100, 12, 6, 30
    for i, s in enumerate(stages):
        x = x0 + i * (w + gap)
        fill, st = ((VIOLET_L, VIOLET) if i == 4 else (RED_L, RED) if i == 6 else
                    (GREEN_L, GREEN) if i == 7 else (ETH_L, ETH))
        b += box(x, y, w, 60, fill, st, [s], size=19, sw=3 if i in (4, 6, 7) else 2)
        if i < 7:
            b += path_arrow(f"M{x + w},{y + 30} L{x + w + gap - 1},{y + 30}", "gray", w=2)
    xm = x0 + 6 * (w + gap) + w / 2
    xr = x0 + 1 * (w + gap) + w / 2
    b += path_arrow(f"M{xm},{y + 60} C{xm},{y + 125} {xr},{y + 125} {xr},{y + 64}", "red", dash="8,5")
    b += text((xm + xr) / 2, y + 150, "new model, data, corpus, region, or use: re-open the risk review",
              21, RED)
    lx = x0 + 4 * (w + gap) + w / 2
    b += text(lx, y + 195, "Launch: who may authorize, and who may stop it?", 21, VIOLET, weight="bold")
    rx = x0 + 7 * (w + gap) + w / 2
    b += text(rx - 26, y + 242, "Retire: an appeal owner remains after shutdown", 21, GREEN,
              anchor="end", weight="bold")
    b += path_arrow(f"M{rx - 20},{y + 236} C{rx + 4},{y + 230} {rx},{y + 150} {rx},{y + 64}", "green",
                    w=2)
    write("lifecycle.svg", b, 900, 290)


# ---------------------------------------------------------------------------
# Segment 2
# ---------------------------------------------------------------------------
def loan_metrics():
    import numpy as np
    names = ["Approval", "TPR", "FPR", "PPV"]
    keys = ["approval", "tpr", "fpr", "ppv"]
    L = calcs.LOAN
    a = [L.positive_rate("A"), L.tpr("A"), L.fpr("A"), L.ppv("A")]
    bb = [L.positive_rate("B"), L.tpr("B"), L.fpr("B"), L.ppv("B")]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    x = np.arange(4)
    ax.bar(x - 0.2, a, 0.38, color=ETH)
    ax.bar(x + 0.2, bb, 0.38, color=AMBER)
    for i in range(4):
        ax.text(x[i] - 0.2, a[i] + 0.03, f"{a[i]:.2f}", ha="center", fontsize=12.5, color=ETH,
                fontweight="bold")
        ax.text(x[i] + 0.2, bb[i] + 0.03, f"{bb[i]:.2f}", ha="center", fontsize=12.5, color=AMBER,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(-0.45, 1.12, "Group A", color=ETH, fontsize=13, fontweight="bold")
    ax.text(0.75, 1.12, "Group B", color=AMBER, fontsize=13, fontweight="bold")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("loan_metrics.svg"))


def compas():
    import numpy as np
    names = ["Base rate", "FPR", "PPV"]
    CP = calcs.COMPAS
    blk = [100 * f(calcs.BLACK) for f in (CP.base_rate, CP.fpr, CP.ppv)]
    wht = [100 * f(calcs.WHITE) for f in (CP.base_rate, CP.fpr, CP.ppv)]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(3)
    ax.bar(x - 0.2, blk, 0.38, color=ETH)
    ax.bar(x + 0.2, wht, 0.38, color=C["neutral"])
    for i in range(3):
        ax.text(x[i] - 0.2, blk[i] + 1.5, f"{blk[i]:.1f}", ha="center", fontsize=12.5, color=ETH,
                fontweight="bold")
        ax.text(x[i] + 0.2, wht[i] + 1.5, f"{wht[i]:.1f}", ha="center", fontsize=12.5,
                color=C["neutral"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Percent")
    ax.text(-0.45, 82, "Black", color=ETH, fontsize=13, fontweight="bold")
    ax.text(0.2, 82, "White defendants", color=C["neutral"], fontsize=13, fontweight="bold")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("compas.svg"))


def dual_thresholds():
    D = calcs.DUAL

    def frame(k):
        b = ""
        x0, x1 = 70, 880

        def X(s):
            return x0 + s * (x1 - x0)
        bands = (("A", 36, ETH_L, ETH), ("B", 182, AMBER_L, AMBER))
        for g, y, fill, st in bands:
            b += f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="100" fill="{fill}" stroke="{st}" stroke-width="1.5"/>\n'
            b += text(32, y + 62, g, 34, st, weight="bold")
            t = D.uniform_threshold if k == 1 else D.group_thresholds[g]
            b += f'<rect x="{X(t)}" y="{y}" width="{x1 - X(t)}" height="100" fill="{st}" fill-opacity="0.18"/>\n'
            b += (f'<line x1="{X(t)}" y1="{y - 4}" x2="{X(t)}" y2="{y + 104}" stroke="{INK}" '
                  f'stroke-width="3" stroke-dasharray="8,5"/>\n')
            b += text(X(t) - 8, y - 8, f"cutoff {t:.2f}", 22, INK, anchor="end", weight="bold")
            for s in D.scores[g]["qualified"]:
                b += text(X(s), y + 46, "+", 40, st, weight="bold")
            for s in D.scores[g]["default"]:
                b += (f'<circle cx="{X(s)}" cy="{y + 72}" r="10" fill="white" stroke="{st}" '
                      f'stroke-width="3"/>\n')
            tpr = D.tpr(g, t)
            b += text(x1 - 8, y + 94, f"TPR {100 * tpr:.0f}%", 24, INK, anchor="end", weight="bold")
        b += f'<line x1="{x0}" y1="300" x2="{x1}" y2="300" stroke="{GRAY}" stroke-width="1.5"/>\n'
        for s in (0, 0.25, 0.5, 0.75, 1.0):
            b += text(X(s), 326, f"{s:g}", 22, GRAY)
        if k == 2:
            same = num("dual_same_score")
            b += text(X(same), 370, f"Two applicants at {val('dual_same_score')}: A denied, B approved", 24, RED,
                      weight="bold")
            b += path_arrow(f"M{X(same) - 150},350 C{X(same) - 200},300 {X(same) - 40},140 {X(same) - 6},96",
                            "red", w=2.2)
            b += path_arrow(f"M{X(same) + 30},350 L{X(same) + 4},{182 + 46}", "red", w=2.2)
        write(f"dual_step{k}.svg", b, 900, 385)
    frame(1)
    frame(2)


def obermeyer():
    rows = [("Before (cost label)", num("ober_before"), "neutral", f"{val('ober_before')}%"),
            ("After (simulated fix)", num("ober_after"), "memory", f"{val('ober_after')}%")]
    sp.hbars(out("obermeyer.svg"), rows, "%", log=False,
             xlabel="Black patients among those auto-identified (%)", size=(4.6, 1.5))


# ---------------------------------------------------------------------------
# Segment 3
# ---------------------------------------------------------------------------
def ztest():
    import numpy as np
    ns = np.linspace(4, 130, 300)
    zs = [calcs.z_for(n) for n in ns]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.plot(ns, zs, color=ETH, lw=3)
    ax.axhline(round(calcs.ZT.z_critical, 2), color=RED, lw=1.8, ls="--")
    ax.text(129, 1.45, f"z = {val('z_crit')} (5%, two-sided)", color=RED, fontsize=13, ha="right",
            fontweight="bold")
    for n in (calcs.ZT.small_n_per_group, calcs.ZT.n_per_group):
        z = calcs.ZT.z_at(n)
        col = AMBER if n == calcs.ZT.small_n_per_group else ETH
        ax.plot([n], [z], "o", ms=11, color=col, mec="white", mew=1.5, zorder=5)
        if n == calcs.ZT.small_n_per_group:
            ax.annotate(f"{n} per group: z = {z:.2f}", (n, z), xytext=(12, -24),
                        textcoords="offset points", ha="left", fontsize=13, fontweight="bold",
                        color=col)
        else:
            ax.annotate(f"{n} per group: z = {z:.2f}", (n, z), xytext=(-10, 10),
                        textcoords="offset points", ha="right", fontsize=13, fontweight="bold",
                        color=col)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 5.4)
    ax.set_xlabel(f"Applicants per group ({val('z_rate_a')} against {val('z_rate_b')} approved)")
    ax.set_ylabel("z statistic")
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("ztest.svg"))


def gs_rows():
    return calcs.GS


def gender_grid():
    acc = calcs.IBM_ACC
    b = text(150, 22, "IBM, 2017 accuracy", 22, GRAY, weight="bold")
    cells = [("lighter male", 6, 36, "#e5e7eb"), ("lighter female", 152, 36, "#d1d5db"),
             ("darker male", 6, 152, "#9ca3af"), ("darker female", 152, 152, RED_L)]
    for k, x, y, fill in cells:
        dk = k == "darker female"
        b += box(x, y, 142, 110, fill, RED if dk else GRAY, [f"{acc[k]:.1f}%"] + k.split(" "),
                 size=22, color=RED if dk else INK, sw=3 if dk else 1.5)
    write("gs_grid.svg", b, 300, 268)


def gender_dumbbell():
    rows = gs_rows()
    comps = ["IBM", "Face++", "Microsoft", "Amazon", "Kairos"]
    df = {r["company"]: r for r in rows if r["subgroup"] == "darker female"}
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(4.2, 2.9))
        for i, co in enumerate(comps):
            y = len(comps) - 1 - i
            a = df[co]["error_may2017_pct"]
            bb = float(df[co]["error_aug2018_pct"])
            if a:
                a = float(a)
                if k == 2:
                    ax.plot([bb, a], [y, y], color=C["grid"], lw=3, zorder=1)
                ax.plot([a], [y], "o", ms=11, color=RED, zorder=3)
                ax.text(a + 1.4, y, f"{a:.1f}", color=RED, fontsize=13, fontweight="bold",
                        va="center")
            if k == 2:
                ax.plot([bb], [y], "o", ms=11, color=ETH, zorder=3)
                if a:
                    ax.text(bb, y - 0.45, f"{bb:.1f}", color=ETH, fontsize=13, fontweight="bold",
                            ha="center", va="center")
                else:
                    ax.text(bb + 1.4, y, f"{bb:.1f}", color=ETH, fontsize=13, fontweight="bold",
                            va="center")
            ax.text(-1.5, y, co, ha="right", va="center", fontsize=13.5, fontweight="bold")
        ax.set_ylim(-0.8, len(comps) + 0.7)
        ax.set_xlim(0, 42)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Error on darker-skinned women (%)")
        ax.text(41, 5.25, "2017 audit", color=RED, fontsize=13, fontweight="bold", ha="right")
        ax.text(41, 4.7, "2018 re-audit", color=ETH, fontsize=13, fontweight="bold", ha="right",
                alpha=1.0 if k == 2 else 0.0)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out(f"gs_step{k}.svg"))


def architecture():
    def frame(k):
        b = text(10, 28, "Request path (inside the serving SLO)", 22, ETH, anchor="start", weight="bold")
        for x, w, t in ((10, 150, "Request"), (190, 210, "Access control"), (430, 160, "Inference"),
                        (740, 150, "Response")):
            b += box(x, 45, w, 60, ETH_L, ETH, [t], size=24)
        b += path_arrow("M160,75 L188,75", "blue")
        b += path_arrow("M400,75 L428,75", "blue")
        b += path_arrow("M590,75 L738,75", "blue")
        if k >= 2:
            b += path_arrow("M510,105 L510,143", "green")
            b += box(360, 145, 300, 50, GREEN_L, GREEN, ["Evidence log"], size=24)
            for i in range(8):
                known = i < 5
                dash = "" if known else ' stroke-dasharray="4,3"'
                b += (f'<rect x="{366 + i * 37}" y="208" width="30" height="28" rx="3" '
                      f'fill="{GREEN if known else "white"}" fill-opacity="{0.5 if known else 1}" '
                      f'stroke="{GREEN}" stroke-width="1.5"{dash}/>\n')
            b += text(440, 262, "known", 19, GREEN)
            b += text(602, 262, "pending", 19, GRAY)
        if k >= 3:
            b += text(10, 440, "Governance tier (asynchronous)", 22, VIOLET, anchor="start", weight="bold")
            b += box(10, 320, 200, 90, VIOLET_L, VIOLET, ["Parity", "updates now"], size=22)
            b += box(230, 320, 240, 90, VIOLET_L, VIOLET, ["Error rates", "when labels land"], size=22)
            b += box(490, 320, 230, 90, "white", RED, ["Thin group", "insufficient, not 0"], size=22,
                     color=RED, dash="6,4")
            b += box(745, 320, 150, 90, VIOLET_L, VIOLET, ["Release", "gate: owner"], size=22)
            b += path_arrow("M365,198 L110,318", "purple", w=2)
            b += path_arrow("M380,240 L330,318", "purple", w=2)
            b += path_arrow("M720,365 L743,365", "purple", w=2)
            b += path_arrow("M820,318 L820,128 L560,128 L560,108", "purple", w=2, dash="8,5")
            b += text(812, 162, "approved update", 19, VIOLET, anchor="end")
        write(f"arch_step{k}.svg", b, 900, 452)
    for k in (1, 2, 3):
        frame(k)


def alerts():
    n = int(num("mon_metrics"))
    exp = num("mon_alerts")
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    cols, rows_ = 25, 6
    red = int(exp)
    for i in range(n):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c, rows_ - 1 - r), 0.86, 0.86,
                                   color=RED if i < red else C["neutral_l"]))
    ax.add_patch(plt.Rectangle((red % cols, rows_ - 1 - red // cols), 0.43, 0.86, color=RED))
    ax.set_xlim(-0.2, cols)
    ax.set_ylim(-0.2, rows_)
    ax.set_aspect("equal")
    ax.axis("off")
    sp.save(fig, out("alerts.svg"))


# ---------------------------------------------------------------------------
# Segment 4
# ---------------------------------------------------------------------------
def overhead():
    rows = calcs.OVERHEAD
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, (lab, lo, hi, path) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=AMBER if path == "train" else ETH, lw=11,
                solid_capstyle="butt")
        ax.text(0.85 * lo, y, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["+10%", "+100%", "+1,000%", "+10⁴%", "+10⁵%"])
    ax.text(3e5, 5.3, "training path", color=AMBER, fontsize=13, fontweight="bold", ha="right")
    ax.text(3e5, 1.3, "serving path", color=ETH, fontsize=13, fontweight="bold", ha="right")
    ax.set_xlabel("Extra work over the unprotected path (log)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("overhead.svg"))


def shap_ladder():
    rows = []
    for f, ck in zip(calcs.SHAP.feature_counts, ("data", "compute", "danger")):
        ms = calcs.SHAP.time(f).to("ms").magnitude
        sub = calcs.SHAP.subsets(f)
        lab = f"{ms:.3f} ms" if ms < 0.1 else (f"{ms:.2f} ms" if ms < 1000 else f"{ms / 1000:.2f} s")
        rows.append((f"{f} features: {sub:,.0f}", ms, ck, lab))
    sp.hbars(out("shap.svg"), rows, "ms", log=True, size=(5.0, 1.8),
             xlabel=f"Exact SHAP time at {val('shap_us')} µs per subset (ms, log)",
             vlines=[(num("shap_budget_ms"), f"{val('shap_budget_ms')} ms budget", "danger")])


def dp_accuracy():
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    me, ma = [r["epsilon"] for r in calcs.MNIST], [r["accuracy_pct"] for r in calcs.MNIST]
    ce, ca = [r["epsilon"] for r in calcs.CIFAR], [r["accuracy_pct"] for r in calcs.CIFAR]
    ax.plot(me, ma, "o-", color=ETH, lw=3, ms=9)
    ax.plot(ce, ca, "o-", color=AMBER, lw=3, ms=9)
    ax.axhline(num("dp_mnist0"), color=ETH, lw=1.5, ls="--")
    ax.axhline(num("dp_cifar0"), color=AMBER, lw=1.5, ls="--")
    ax.text(0.37, 99.3, f"MNIST, no privacy: {val('dp_mnist0')}%", color=ETH, fontsize=12.5, ha="right")
    ax.text(0.37, 81.3, f"CIFAR-10, no privacy: about {val('dp_cifar0')}%", color=AMBER, fontsize=12.5, ha="right")
    for e, a in zip(me, ma):
        ax.text(e, a - 5.0, f"{a:.0f}%", color=ETH, fontsize=13, ha="center", fontweight="bold")
    for e, a in zip(ce, ca):
        ax.text(e, a + 2.0, f"{a:.0f}%", color=AMBER, fontsize=13, ha="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(12, 0.35)
    ax.set_xticks([8, 4, 2, 1, 0.5])
    ax.set_xticklabels(["8", "4", "2", "1", "0.5"])
    ax.minorticks_off()
    ax.set_ylim(60, 104)
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (smaller is stronger)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("dp_accuracy.svg"))


def privacy_boundary():
    def frame(k):
        b = ""
        for i in range(3):
            y = 20 + i * 105
            b += box(10, y, 160, 88, ETH_L, ETH, [f"Phone {i + 1}", "raw data"], size=22)
            b += path_arrow(f"M170,{y + 44} L226,{180 + (i - 1) * 25}", "blue", w=2)
        b += text(10, 350, "1: data stays local", 22, ETH, anchor="start", weight="bold")
        if k >= 2:
            b += box(228, 105, 220, 150, AMBER_L, AMBER, ["Secure", "aggregation", "hides each", "update"],
                     size=22)
            b += path_arrow("M448,180 L486,180", "amber")
            b += box(488, 105, 200, 150, VIOLET_L, VIOLET, ["Curator", "adds DP", "noise, counts ε"],
                     size=22)
            b += path_arrow("M688,180 L738,180", "purple")
            b += box(740, 130, 150, 100, GREEN_L, GREEN, ["Released", "model"], size=22)
            b += f'<line x1="714" y1="20" x2="714" y2="290" stroke="{GRAY}" stroke-width="2" stroke-dasharray="6,5"/>\n'
            b += text(720, 40, "public", 20, GRAY, anchor="start")
            b += text(338, 282, "2: updates hidden", 21, AMBER, weight="bold")
            b += text(588, 282, "3: model bounded", 21, VIOLET, weight="bold")
        if k >= 3:
            b += box(330, 320, 330, 80, RED_L, RED, ["Gateway log: raw prompt", "no DP claim covers it"],
                     size=22, color=RED)
            b += path_arrow("M815,232 C815,360 750,362 664,360", "red", dash="7,5")
        write(f"privacy_step{k}.svg", b, 900, 410)
    for k in (1, 2, 3):
        frame(k)


def unlearn():
    U = calcs.SISA
    rows = [("Full retraining", U.full_cost.to("USD").magnitude, "danger",
             f"\\${val('sisa_full_m')}M, {val('sisa_days')} days"),
            ("SISA, 1 request", U.shard_cost.to("USD").magnitude, "data",
             f"\\${val('sisa_shard_cost')}, {val('sisa_shard_h1')} h"),
            (f"SISA, {val('sisa_req')}/day", U.daily_batch_cost.to("USD").magnitude,
             "compute", f"about \\${val('sisa_batch_m')}M a day")]
    for k, step in ((1, 2), (2, 3)):
        sp.hbars(out(f"unlearn_step{k}.svg"), rows, "$", log=True, step=step, size=(4.4, 1.6),
                 xlabel=f"Cost to forget, {val('sisa_params_b')}B model ($, log)")


def genai():
    b = box(330, 120, 240, 110, ETH_L, ETH, ["Model weights", "aligned by RLHF"], size=24)
    items = [(10, 10, "System prompt", "versioned per product"),
             (10, 250, "Retrieval index", "untrusted context"),
             (630, 10, "Reward model", "whose raters?"),
             (630, 250, "Output filter", "runtime policy check")]
    for x, y, t1, t2 in items:
        b += box(x, y, 260, 90, VIOLET_L, VIOLET, [t1, t2], size=23)
    b += path_arrow("M270,70 L328,140", "purple", w=2.5)
    b += path_arrow("M270,290 L328,212", "purple", w=2.5)
    b += path_arrow("M630,70 L572,140", "purple", w=2.5)
    b += path_arrow("M572,212 L628,290", "purple", w=2.5)
    write("genai.svg", b, 900, 350)


# ---------------------------------------------------------------------------
# Segment 5
# ---------------------------------------------------------------------------
def contest():
    def frame(k):
        b = ""
        st = [("Decision provenance", "inputs, model, policy version", ETH_L, ETH),
              ("Explanation", "reasons for this audience", FRAME_L, FRAME),
              ("Appeal routing", "reviewer with authority", FRAME_L, FRAME),
              ("Outcome tracking", "upheld or overturned", FRAME_L, FRAME)]
        for i, (t1, t2, f, s) in enumerate(st):
            y = 10 + i * 108
            if k == 1 and i > 0:
                continue
            b += box(100, y, 390, 84, f, s, [t1, t2], size=24)
            if i < 3 and k == 2:
                b += path_arrow(f"M295,{y + 84} L295,{y + 106}", "gray", w=2)
        if k == 2:
            b += path_arrow("M100,376 C50,376 50,52 96,52", "green", dash="8,5")
            b += (f'<text x="22" y="214" text-anchor="middle" font-size="22" font-weight="bold" '
                  f'fill="{GREEN}" transform="rotate(-90 22 214)">retrain</text>\n')
        write(f"contest_step{k}.svg", b, 500, 440)
    frame(1)
    frame(2)


def loop():
    def frame(k):
        b = ""
        nodes = [(315, 15, ["Biased training data"], False), (620, 165, ["Biased predictions"], False),
                 (315, 315, ["Disparate outcomes"], True), (10, 158, ["Reinforced bias", "in new data"], False)]
        for x, y, t, red in nodes:
            h = 84 if len(t) > 1 else 70
            b += box(x, y, 270, h, RED_L if red else ETH_L, RED if red else ETH, t, size=24,
                     color=RED if red else INK, bold_first=True)
        b += path_arrow("M585,50 C660,55 740,90 755,163", "blue")
        b += path_arrow("M755,235 C740,300 660,345 587,350", "blue")
        b += path_arrow("M315,350 C220,345 150,300 145,244", "blue")
        b += path_arrow("M145,158 C150,90 230,55 313,50", "blue")
        b += text(450, 195, "retrain on", 22, GRAY, italic=True)
        b += text(450, 222, "its own outcomes", 22, GRAY, italic=True)
        if k == 2:
            for x, y in ((196, 84), (702, 86), (702, 316)):
                b += f'<circle cx="{x}" cy="{y}" r="12" fill="{GREEN}"/>\n'
            b += box(10, 10, 250, 46, GREEN_L, GREEN, ["Preprocess: reweight"], size=22, bold_first=False)
            b += box(615, 10, 280, 46, GREEN_L, GREEN, ["In-process: constraints"], size=22,
                     bold_first=False)
            b += box(615, 340, 280, 46, GREEN_L, GREEN, ["Post-process: thresholds"], size=22,
                     bold_first=False)
        write(f"loop_step{k}.svg", b, 900, 395)
    frame(1)
    frame(2)


def autobias():
    best = num("ab_team")
    rows = [("Radiologist alone", num("ab_human"), "neutral", f"{val('ab_human')}%"),
            ("AI alone", num("ab_ai"), "memory", f"{val('ab_ai')}%"),
            ("Team, best case", best, "compute", f"at most {best:.1f}%")]
    sp.hbars(out("autobias.svg"), rows, "%", log=False, size=(4.4, 1.7),
             xlabel="Tumors found among patients with tumors (%)")


def waymo():
    import datetime as dt
    import matplotlib.dates as mdates
    rows = calcs.MILES
    xs = [dt.date(int(r["data_through"][:4]), int(r["data_through"][5:]), 1) for r in rows]
    ys = [float(r["cumulative_ro_miles_millions"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    ax.plot(xs, ys, "o-", color=ETH, lw=3, ms=7)
    ax.annotate(f"{ys[0]:.0f}M", (xs[0], ys[0]), xytext=(8, 12), textcoords="offset points",
                ha="left", fontsize=13, fontweight="bold", color=ETH)
    ax.annotate(f"{ys[-1]:.0f}M", (xs[-1], ys[-1]), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", fontsize=13, fontweight="bold", color=ETH)
    ax.text(xs[1], 200, f"{val('waymo_growth')}× in two years", fontsize=13,
            fontweight="bold", color=INK)
    ax.set_ylabel("Rider-only miles (M)")
    ax.set_ylim(0, 300)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("Jun %Y"))
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out("waymo_miles.svg"))


def main():
    ofqual_flows()
    ofqual_candidates()
    hook()
    assurance()
    lifecycle()
    loan_metrics()
    compas()
    dual_thresholds()
    obermeyer()
    ztest()
    gender_grid()
    gender_dumbbell()
    architecture()
    alerts()
    overhead()
    shap_ladder()
    dp_accuracy()
    privacy_boundary()
    unlearn()
    genai()
    contest()
    loop()
    autobias()
    waymo()


if __name__ == "__main__":
    main()
