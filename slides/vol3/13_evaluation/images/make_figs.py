"""Lecture 13 (Agent Evaluation): every figure the deck draws.

Every value comes from calcs.py, which reads MLSysIM (and the deck's own run
in data/); this script imports it and prints the recorded values (val, num),
so figures and slides agree. Plots use the shared slideplot helper; diagrams
are SVG drawn at the size of their slot.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=13_evaluation`
converts them to PDF).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402
import calcs  # noqa: E402
from deckvals import val, num  # noqa: E402
from mlsysim.physics import calc_independent_success, calc_unchecked_success  # noqa: E402


# Volume III roles (volumes/vol3/palette.md)
BLUE, BLUE_L = "#0284c7", "#e0f2fe"
GREEN, GREEN_L = "#16a34a", "#dcfce7"
AMBER, AMBER_L = "#d97706", "#fef3c7"
PURPLE, PURPLE_L = "#7c3aed", "#f3e8ff"
RED, RED_L = "#dc2626", "#fee2e2"
GRAY, GRAY_L = "#64748b", "#f1f5f9"
INK = "#0f172a"
FONT = "Helvetica, Arial, sans-serif"


def axis_type(ax, size):
    """Set tick and axis-label type for a plot drawn at its slot size."""
    ax.tick_params(labelsize=size)
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)


# ---------------------------------------------------------------- SVG helpers
def svg(path, w, h, body):
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n<defs>'
        + "".join(f'<marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
                  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>'
                  for n, c in [("g", GRAY), ("p", PURPLE), ("a", AMBER), ("r", RED), ("b", BLUE)])
        + f'</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, rx=10, dash=False, sw=2.5):
    d = ' stroke-dasharray="8 6"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def text(x, y, s, size=22, color=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def arrow(x1, y1, x2, y2, m="g", color=GRAY, dash=False, sw=2.5):
    d = ' stroke-dasharray="7 5"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
            f'marker-end="url(#a{m})"/>')


# ---------------------------------------------------------------- plots
def hook():
    """pass^k against pass@k at 90 percent per run (NM 13.2)."""
    p, k = calcs.HOOK_S.step_success, calcs.HOOK_S.long_horizon
    ks = list(range(1, k + 1))
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.6, 2.9))
        ax.plot(ks, [100 * calc_independent_success(p, j) for j in ks], color=BLUE, lw=3,
                alpha=1 if step == 2 else 0)
        ax.plot(ks, [100 * calc_unchecked_success(p, j) for j in ks], color=PURPLE, lw=3,
                alpha=1 if step == 2 else 0)
        ax.plot([1], [num("hook_p_pct")], "o", color=INK, ms=9)
        ax.annotate(f"one run: {val('hook_p_pct')}%", (1, num("hook_p_pct")), xytext=(1.6, 50), fontsize=13,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=1))
        if step == 1:
            ax.text(k, 50, f"all {val('hook_k')} pass?", fontsize=15, color=PURPLE, ha="right", fontweight="bold")
        else:
            pow_k = 100 * calcs.HOOK.success[k]
            ax.plot([k], [pow_k], "o", color=PURPLE, ms=10)
            ax.text(7.8, pow_k - 12, f"pass$^{val('hook_k')}$ = {val('hook_pow_pct')}%", fontsize=14,
                    color=PURPLE, ha="right", fontweight="bold")
            ax.text(7.8, 104, f"pass@{val('hook_k')}: at least one of {val('hook_k')}", fontsize=13, color=BLUE,
                    ha="right")
            ax.text(4.6, 74, "all of k pass", fontsize=13, color=PURPLE)
        ax.set_xlim(0.7, 8.4)
        ax.set_ylim(0, 112)
        ax.set_xticks(ks)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Runs of the same task, k")
        ax.set_ylabel("Probability (%)")
        ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
        sp.save(fig, HERE / f"hook_step{step}.svg")


def scaffold():
    """Terminal-Bench 2.0: same model, many scaffolds (MLSysIM)."""
    by = {}
    for r in calcs.SC_S.rows:
        by.setdefault(r["model"], []).append(100 * r["accuracy"])
    by = {m: v for m, v in by.items() if m in calcs.SC.spreads}
    order = sorted(by, key=lambda m: sorted(by[m])[len(by[m]) // 2])
    fig, ax = plt.subplots(figsize=(6.2, 2.75))
    for y, m in enumerate(order):
        v = by[m]
        hi = m == calcs.SC.widest
        ax.plot([min(v), max(v)], [y, y], color=PURPLE if hi else BLUE_L, lw=4 if hi else 3.5,
                solid_capstyle="round", alpha=0.5 if hi else 1)
        ax.plot(v, [y] * len(v), "o", color=PURPLE if hi else BLUE, ms=4.5)
        ax.text(101, y, f"{100 * calcs.SC.spreads[m]:.0f} pts, n={len(v)}", va="center", fontsize=10,
                color=PURPLE if hi else GRAY, fontweight="bold" if hi else "normal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    for t, m in zip(ax.get_yticklabels(), order):
        if m == calcs.SC.widest:
            t.set_color(PURPLE)
            t.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Terminal-Bench 2.0 accuracy (%), one dot per scaffold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.tick_params(axis="y", length=0)
    axis_type(ax, 10)
    sp.save(fig, HERE / "scaffold.svg")


def verdicts():
    """Dropping unresolved trials inflates the rate (illustrative, 60/25/15)."""
    r = calcs.VERD
    rows = [("Reported, default rule", "vd_rate", r.default_rate, PURPLE),
            ("If every open trial passes", "vd_upper", r.upper_bound, GRAY),
            ("Open trials dropped", "vd_drop", r.dropped_rate, RED)]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(2.6, 1.3))
        for i, (lab, key, rate, c) in enumerate(rows):
            show = step == 2 or i < 2
            v = 100 * rate
            ax.barh(2 - i, v, color=c, height=0.6, alpha=1 if show else 0)
            ax.text(v + 1.5, 2 - i, f"{val(key)}%", va="center", fontsize=9.4,
                    color=c, fontweight="bold", alpha=1 if show else 0)
            ax.text(-1.5, 2 - i, lab, ha="right", va="center", fontsize=8.6, color=INK)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"Acceptance rate of {val('vd_n')} trials (%)")
        axis_type(ax, 8.5)
        sp.save(fig, HERE / f"verdicts_step{step}.svg")


def coverage():
    """Our run: visible test pass vs sealed test pass, per configuration."""
    rows = [(calcs.BASE, "b"), (calcs.CAND, "c")]
    fig, ax = plt.subplots(figsize=(2.3, 1.25))
    for i, (lab, tag) in enumerate(rows):
        y = 1 - i
        vis = 100 * calcs.RUN.visible_pass[lab] / calcs.RUN.trials[lab]
        seal = 100 * calcs.RUN.rate[lab]
        ax.barh(y + 0.18, vis, height=0.34, color=GRAY_L, edgecolor=GRAY)
        ax.barh(y - 0.18, seal, height=0.34, color=PURPLE)
        ax.text(vis + 1.5, y + 0.18, f"visible test {val(f'run_{tag}_vis_pct')}%", va="center", fontsize=8.6,
                color=GRAY)
        ax.text(seal - 1.5, y - 0.18, f"sealed {val(f'run_{tag}_pct')}%", va="center", ha="right", fontsize=8.6,
                color="white", fontweight="bold")
        ax.text(-2, y, lab, ha="right", va="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Trials passing (%), {val('run_trials')} per configuration")
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "coverage.svg")


def zheng():
    """Zheng et al. 2023, Table 2, default prompt: position bias."""
    rows = [(j, r["consistency_pct"], r["biased_first_pct"], r["biased_second_pct"], r["error_pct"])
            for j, r in ((j, calcs.ZHENG[j]) for j in ("GPT-4", "GPT-3.5", "Claude-v1"))]
    fig, ax = plt.subplots(figsize=(5.4, 2.1))
    for i, (j, cons, first, second, err) in enumerate(rows):
        y = 2 - i
        left = 0
        for v, c in ((cons, GREEN), (first, RED), (second, AMBER), (err, GRAY)):
            ax.barh(y, v, left=left, color=c, height=0.62)
            if v >= 12:
                ax.text(left + v / 2, y, f"{v:.1f}%", ha="center", va="center", fontsize=11.5,
                        color="white", fontweight="bold")
            left += v
        ax.text(-2, y, j, ha="right", va="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(1, 2.52, "same verdict after swap", fontsize=11, color=GREEN, fontweight="bold")
    ax.text(66, 2.52, "favors first position", fontsize=11, color=RED, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Answer pairs judged twice, order swapped (%)")
    sp.save(fig, HERE / "zheng.svg")


def designs():
    """Tasks or repeats: standard error of 600-run designs (exercise 1)."""
    tr = calcs.TR
    free = [(f"{tr.free_tasks[r]} tasks x {r}", 100 * tr.free_se[r]) for r in calcs.TR_S.free_runs]
    shown = (1, 2, tr.best_runs, tr.best_runs + 1)  # the cheapest designs and the best one
    paid = [(f"{tr.paid_tasks[r]} tasks x {r}", 100 * tr.paid_se[r]) for r in shown]
    for step, rows, name in ((1, free, "designs_step1"), (2, paid, "designs_step2")):
        fig, ax = plt.subplots(figsize=(5.0, 1.9))
        best = min(v for _, v in rows)
        for i, (lab, v) in enumerate(rows):
            y = len(rows) - 1 - i
            c = PURPLE if v == best else GRAY
            ax.barh(y, v, color=c, height=0.62)
            ax.text(v + 0.12, y, f"{v:.2f}", va="center", fontsize=13, color=c, fontweight="bold")
            ax.text(-0.12, y, lab.replace(" x ", " × "), ha="right", va="center", fontsize=13, color=INK)
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Standard error (points)")
        axis_type(ax, 12)
        sp.save(fig, HERE / f"{name}.svg")


def wilson_fig():
    """Wald against Wilson at 50 tasks (§13.5.2, figure 4 panel B)."""
    n = calcs.WL_S.trials
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.axvspan(-8, 0, color=RED_L)
    ax.text(-4, 2.62, "impossible", ha="center", fontsize=11, color=RED)
    for i, x in enumerate(calcs.WL_S.successes):
        y = 2 - i
        wl, wu = calcs.WL.wald[x]
        sl, su = calcs.WL.wilson[x]
        ax.plot([100 * wl, 100 * wu], [y + 0.17] * 2, color=GRAY, lw=5, solid_capstyle="butt")
        if x == 0:
            ax.plot([0], [y + 0.17], "|", color=GRAY, ms=14, mew=3)
        ax.plot([100 * sl, 100 * su], [y - 0.17] * 2, color=PURPLE, lw=5, solid_capstyle="butt")
        ax.text(-9, y, f"{x} of {n}", ha="right", va="center", fontsize=12.5, fontweight="bold", color=INK)
        ax.text(100 * max(wu, su) + 2, y + 0.17, f"Wald {100*wl:.2f} to {100*wu:.2f}", va="center",
                fontsize=11, color=GRAY)
        ax.text(100 * max(wu, su) + 2, y - 0.17, f"Wilson {100*sl:.2f} to {100*su:.2f}", va="center",
                fontsize=11, color=PURPLE, fontweight="bold")
    ax.set_xlim(-8, 100)
    ax.set_ylim(-0.6, 2.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Acceptance rate (%), {val('conf_pct')} percent interval")
    sp.save(fig, HERE / "wilson.svg")


def ranks():
    """SWE-bench Verified, Epoch AI runs, top 25 by score (MLSysIM)."""
    rows = sorted(calcs.RK_S.rows, key=lambda r: -r["mean_score"])[:calcs.RK.top]
    z = calcs.WL.z_critical
    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    xs = range(1, len(rows) + 1)
    ax.errorbar(list(xs), [100 * r["mean_score"] for r in rows],
                yerr=[100 * z * r["stderr"] for r in rows], fmt="o", color=BLUE, ecolor=BLUE,
                elinewidth=2, capsize=2.5, ms=5)
    a, b = rows[0], rows[1]
    ax.annotate("ranks 1 and 2 overlap", (1.5, 100 * a["mean_score"] + 4), xytext=(5, 89),
                fontsize=12, color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.2))
    ax.set_xticks([1, 5, 10, 15, 20, calcs.RK.top])
    ax.set_xlim(0, 26)
    ax.set_ylim(60, 92)
    ax.set_xlabel("Leaderboard rank (one evaluator, one scaffold)")
    ax.set_ylabel("Resolved (%)")
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    sp.save(fig, HERE / "ranks.svg")


def run_curves():
    """Our run: pass@k and pass^k from 8 graded runs per task."""
    run = calcs.RUN
    runs = run.runs
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for cfg, ls in ((calcs.BASE, "--"), (calcs.CAND, "-")):
        ks = list(range(1, runs + 1))
        at = [100 * run.pass_at[cfg][k] for k in ks]
        pw = [100 * run.pass_pow[cfg][k] for k in ks]
        ax.plot(ks, at, ls, color=BLUE, lw=3)
        ax.plot(ks, pw, ls, color=PURPLE, lw=3)
        if cfg == calcs.CAND:
            ax.text(8.15, at[-1] + 1, f"pass@{runs}\n{val('run_c_at_max')}%", va="center", fontsize=13, color=BLUE,
                    fontweight="bold")
            ax.text(8.15, pw[-1] + 2, f"pass$^{runs}$\n{val('run_pow_max')}%", va="center", fontsize=13,
                    color=PURPLE, fontweight="bold")
        else:
            ax.text(8.15, at[-1] - 6, f"{val('run_b_at_max')}%", va="center", fontsize=13, color=BLUE)
    ax.text(1.2, 36, "solid: candidate_v2\ndashed: baseline_v1", fontsize=12.5, color=GRAY)
    ax.set_xlim(0.7, runs + 0.1)
    ax.set_ylim(30, 100)
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.set_xlabel("k (graded runs per task)")
    ax.set_ylabel("Tasks (%)")
    axis_type(ax, 13)
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    sp.save(fig, HERE / "run_curves.svg")


def kapoor():
    """Kapoor et al. 2024, Table A1: HumanEval accuracy against total cost."""
    role_color = {"published agent": GRAY, "author baseline": PURPLE, "single call": BLUE}
    label = {"GPT-4 (baseline)": "GPT-4, one call", "GPT-3.5 (baseline)": "GPT-3.5, one call"}
    pts = [(label.get(a, a), r["accuracy_pct"], r["total_cost_usd"],
            RED if a == "LATS (GPT-4)" else role_color[r["role"]]) for a, r in calcs.KAPOOR.items()]
    offs = {"LATS (GPT-4)": (-8, 7, "right"), "LATS (GPT-3.5)": (6, -14, "left"),
            "LDB (GPT-4)": (6, 4, "left"), "Reflexion (GPT-4)": (6, -12, "left"),
            "Retry (GPT-4)": (-8, -6, "right"), "Warming (GPT-4)": (-8, 6, "right"),
            "GPT-4, one call": (-8, -6, "right"), "Escalation": (6, -12, "left"),
            "GPT-3.5, one call": (6, 2, "left")}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for lab, acc, cost, c in pts:
        ax.plot([cost], [acc], "o", color=c, ms=8)
        dx, dy, ha = offs[lab]
        ax.annotate(lab, (cost, acc), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=10.5, color=c, fontweight="bold" if c != GRAY else "normal")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 400)
    ax.set_ylim(70, 97)
    ax.set_xlabel(f"Total cost of the {val('ha_problems')} problems (US dollars, log)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(color=sp.C["grid"], lw=0.6)
    sp.save(fig, HERE / "kapoor.svg")


def batch():
    """NM 13.3: cost per accepted task, batch vs sequential."""
    b, s = calcs.usd(calcs.SEL.batch_per_accepted), calcs.usd(calcs.SEL.sequential_per_accepted)
    fig, ax = plt.subplots(figsize=(4.4, 1.6))
    ax.barh(1, b, color=GRAY, height=0.6)
    ax.barh(0, s, color=PURPLE, height=0.6)
    ax.text(b + 0.004, 1, f"${val('sel_batch_acc')}", va="center", fontsize=13, color=GRAY, fontweight="bold")
    ax.text(s + 0.004, 0, f"${val('sel_seq_acc')}", va="center", fontsize=13, color=PURPLE, fontweight="bold")
    ax.text(-0.003, 1, f"batch of {val('sel_cap')}", ha="right", va="center", fontsize=12.5)
    ax.text(-0.003, 0, "sequential", ha="right", va="center", fontsize=12.5)
    ax.set_xlim(0, 0.2)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xticklabels(["$0", "$0.05", "$0.10", "$0.15"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cost per accepted task")
    sp.save(fig, HERE / "batch.svg")


# ---------------------------------------------------------------- diagrams
def dims():
    """Four evidence dimensions (table 1), then the criteria they supply."""
    cols = [("Proposal", "validity", "schema rejected,", "then repaired", BLUE, BLUE_L),
            ("Execution", "behavior", "timeout leaves", "a test unresolved", AMBER, AMBER_L),
            ("Task", "outcome", "checks on the", "exact ws_r19 bytes", GREEN, GREEN_L),
            ("Constraint", "compliance", "authority, budget,", "cleanup owners", RED, RED_L)]
    for step in (1, 2):
        b = []
        for i, (t1, t2, e1, e2, s, f) in enumerate(cols):
            x = 20 + i * 220
            b.append(box(x, 20, 200, 190, f, s))
            b.append(text(x + 100, 68, t1, 26, s, weight="bold"))
            b.append(text(x + 100, 100, t2, 26, s, weight="bold"))
            b.append(text(x + 100, 150, e1, 19, INK))
            b.append(text(x + 100, 176, e2, 19, INK))
        if step == 2:
            b.append(text(450, 262, "The accepted-task criteria they supply", 22, GRAY, weight="bold"))
            spans = [(20, 420, "process", PURPLE), (460, 640, "outcome", PURPLE),
                     (670, 880, "resource, residual duty", PURPLE)]
            for x1, x2, lab, c in spans:
                b.append(box(x1, 280, x2 - x1, 54, PURPLE_L, c, rx=8))
                b.append(text((x1 + x2) / 2, 316, lab, 22 if len(lab) < 12 else 17, c, weight="bold"))
            b.append(text(450, 378, "Exposure each audits", 22, GRAY, weight="bold"))
            b.append(text(220, 414, "authority exercised", 21, INK))
            b.append(text(550, 414, "state the result rests on", 21, INK))
            b.append(text(770, 414, "horizon consumed", 21, INK))
        svg(HERE / f"dims_step{step}.svg", 900, 430, "\n".join(b))


def gym():
    """Trial sandbox and grading boundary (after figure 2, §13.3.4)."""
    for step in (1, 2, 3):
        b = []
        b.append(box(15, 15, 420, 450, "#fffbeb", AMBER))
        b.append(text(225, 50, "Task sandbox, one per trial", 22, AMBER, weight="bold"))
        b.append(box(45, 75, 360, 70, PURPLE_L, PURPLE))
        b.append(text(225, 108, "agent under test", 22, PURPLE, weight="bold"))
        b.append(text(225, 134, "candidate_v2", 19, INK))
        b.append(arrow(225, 145, 225, 190, "g"))
        b.append(box(45, 195, 360, 52, GREEN_L, GREEN))
        b.append(text(225, 229, "copy-on-write layer (this trial)", 20, INK))
        b.append(box(45, 252, 360, 52, GRAY_L, GRAY))
        b.append(text(225, 286, "base image, read-only, shared", 20, INK))
        b.append(box(45, 325, 220, 52, BLUE_L, BLUE))
        b.append(text(155, 358, "service fixtures", 20, INK))
        b.append(box(285, 325, 120, 52, RED_L, RED))
        b.append(text(345, 349, "egress", 19, RED, weight="bold"))
        b.append(text(345, 370, "blocked", 19, RED, weight="bold"))
        b.append(text(225, 425, "reset: kill processes, revoke", 19, GRAY))
        b.append(text(225, 450, "credentials, discard the layer", 19, GRAY))
        if step >= 2:
            b.append(arrow(405, 221, 495, 221, "a", AMBER, sw=3.5))
            b.append(text(450, 205, "ws_r19", 19, AMBER, weight="bold"))
            b.append(box(500, 15, 385, 450, "#faf5ff", PURPLE))
            b.append(text(692, 50, "Grading boundary", 22, PURPLE, weight="bold"))
            b.append(box(530, 75, 325, 150, "white", PURPLE))
            b.append(text(692, 106, "trusted controller", 21, PURPLE, weight="bold"))
            for x, w, chip in ((540, 130, "hidden tests"), (678, 85, "criteria"), (771, 78, "results")):
                b.append(box(x, 124, w, 44, PURPLE_L, PURPLE, rx=6, sw=1.5))
                b.append(text(x + w / 2, 152, chip, 19, INK))
            b.append(text(692, 202, "outside the candidate's reach", 20, GRAY))
        if step >= 3:
            b.append(arrow(640, 225, 640, 275, "p", PURPLE))
            b.append(text(632, 258, "selected tests", 19, PURPLE, anchor="end"))
            b.append(box(530, 280, 325, 70, RED_L, RED, dash=True))
            b.append(text(692, 309, "candidate execution, limited", 19, RED, weight="bold"))
            b.append(text(692, 336, "the remaining attack surface", 20, INK))
            b.append(arrow(760, 280, 760, 230, "r", RED, dash=True))
            b.append(text(772, 258, "untrusted", 19, RED, anchor="start"))
            b.append(box(530, 375, 325, 75, "white", PURPLE))
            b.append(text(692, 404, "pass, fail, pending, unknown", 19, INK, weight="bold"))
            b.append(text(692, 434, "each bound to sha256(ws_r19)", 20, PURPLE))
        svg(HERE / f"gym_step{step}.svg", 900, 480, "\n".join(b))


def confusion():
    """Grader error rates condition on different populations (exercise 3)."""
    cells = {("fails", "accepts"): val("gr_acc_fail"), ("fails", "rejects"): val("gr_rej_fail"),
             ("passes", "accepts"): val("gr_acc_pass"), ("passes", "rejects"): val("gr_rej_pass")}
    for step in (1, 2):
        b = []
        x0, y0, cw, ch = 240, 95, 180, 110
        b.append(text(x0 + cw, 40, "Grader under study", 22, GRAY, weight="bold"))
        b.append(text(x0 + cw / 2, 78, "accepts", 22, INK, weight="bold"))
        b.append(text(x0 + 1.5 * cw, 78, "rejects", 22, INK, weight="bold"))
        b.append(text(80, y0 + 40, "Reference", 21, GRAY, weight="bold", anchor="middle"))
        b.append(text(80, y0 + 66, "assessment", 21, GRAY, weight="bold", anchor="middle"))
        b.append(text(x0 - 22, y0 + ch / 2 + 8, "fails", 22, INK, anchor="end", weight="bold"))
        b.append(text(x0 - 22, y0 + 1.5 * ch + 8, "passes", 22, INK, anchor="end", weight="bold"))
        for i, ref in enumerate(("fails", "passes")):
            for j, gr in enumerate(("accepts", "rejects")):
                hit = (ref, gr) == ("fails", "accepts")
                b.append(box(x0 + j * cw, y0 + i * ch, cw, ch, RED_L if hit else "white", GRAY, rx=0, sw=1.5))
                b.append(text(x0 + j * cw + cw / 2, y0 + i * ch + ch / 2 + 12, str(cells[(ref, gr)]), 34,
                              RED if hit else INK, weight="bold"))
        # step 1: row of reference failures
        b.append(f'<rect x="{x0-6}" y="{y0-6}" width="{2*cw+12}" height="{ch+12}" fill="none" '
                 f'stroke="{PURPLE}" stroke-width="4" rx="6"/>')
        b.append(text(x0 + 2 * cw + 25, y0 + 45, "false-acceptance rate", 21, PURPLE, anchor="start", weight="bold"))
        b.append(text(x0 + 2 * cw + 25, y0 + 75, f"{val('gr_acc_fail')} / {val('gr_ref_fail')} = {val('gr_far')}%", 21, PURPLE, anchor="start"))
        if step == 2:
            b.append(f'<rect x="{x0-12}" y="{y0-12}" width="{cw+24}" height="{2*ch+24}" fill="none" '
                     f'stroke="{BLUE}" stroke-width="4" stroke-dasharray="10 6" rx="6"/>')
            b.append(text(x0 + 2 * cw + 25, y0 + ch + 60, "failure among accepted", 21, BLUE, anchor="start", weight="bold"))
            b.append(text(x0 + 2 * cw + 25, y0 + ch + 90, f"{val('gr_acc_fail')} / {val('gr_accepted')} = {val('gr_fam')}%", 21, BLUE, anchor="start"))
        svg(HERE / f"confusion_step{step}.svg", 900, 340, "\n".join(b))


def kappa():
    """Two judges that each pass 90 percent at random (§13.4.3)."""
    b = []
    x0, y0, c = 140, 70, 150
    b.append(text(x0 + c, 30, "Judge B", 30, GRAY, weight="bold"))
    b.append(text(x0 + c / 2, 60, "pass", 28, INK))
    b.append(text(x0 + 1.5 * c, 60, "fail", 28, INK))
    b.append(text(52, y0 + c - 10, "Judge", 30, GRAY, weight="bold"))
    b.append(text(52, y0 + c + 22, "A", 30, GRAY, weight="bold"))
    b.append(text(x0 - 12, y0 + c / 2 + 8, "pass", 28, INK, anchor="end"))
    b.append(text(x0 - 12, y0 + 1.5 * c + 8, "fail", 28, INK, anchor="end"))
    vals = [[val("kc_both_pass_pct"), val("kc_split_pct")], [val("kc_split_pct"), val("kc_both_fail_pct")]]
    for i in range(2):
        for j in range(2):
            agree = i == j
            b.append(box(x0 + j * c, y0 + i * c, c, c, GREEN_L if agree else "white", GRAY, rx=0, sw=1.5))
            b.append(text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 14, f"{vals[i][j]}%", 38,
                          GREEN if agree else INK, weight="bold"))
    svg(HERE / "kappa.svg", 470, 390, "\n".join(b))


def recall_thumbs():
    """Text-free cues: a saga with compensations (Lecture 12) and a staged
    release (Lecture 14). Shapes only; the slide text carries the words."""
    b = []
    for i in range(4):
        x = 20 + i * 120
        b.append(box(x, 40, 80, 50, GREEN_L if i < 3 else RED_L, GREEN if i < 3 else RED))
        if i < 3:
            b.append(arrow(x + 80, 65, x + 118, 65, "g"))
    for i in range(2, 0, -1):
        x = 20 + i * 120
        b.append(f'<path d="M{x+40},92 C{x+30},140 {x-50},140 {x-80+40},92" fill="none" stroke="{PURPLE}" '
                 f'stroke-width="3" stroke-dasharray="7 5" marker-end="url(#ap)"/>')
    svg(HERE / "thumb_saga.svg", 500, 150, "\n".join(b))
    b = [box(20, 30, 460, 40, GRAY_L, GRAY, rx=6), box(20, 30, 46, 40, PURPLE, PURPLE, rx=6),
         box(20, 90, 460, 40, GRAY_L, GRAY, rx=6), box(20, 90, 230, 40, PURPLE, PURPLE, rx=6),
         arrow(250, 80, 470, 80, "p", PURPLE)]
    svg(HERE / "thumb_canary.svg", 500, 150, "\n".join(b))


def thumbs_src():
    """Text-free sources for goals and takeaways thumbnails are made by
    tools/greek_thumbs.py from these figures."""
    return ["dims_step2.svg", "gym_step3.svg", "confusion_step2.svg", "hook_step2.svg", "batch.svg"]


# ------------------------------------------- visual-review additions (2026-10-21)
def run_hsa():
    """One trial of the running case, tied to its three exposures."""
    b = [box(170, 168, 220, 96, PURPLE_L, PURPLE),
         text(280, 206, "trial ws_r19", 27, PURPLE, weight="bold"),
         text(280, 240, "pool_repair_01", 23, INK)]
    spokes = [("H", "horizon", "many turns: tokens,", "time, dollars", 20, 20, BLUE, BLUE_L, "b"),
              ("S", "state", "workspace ws_r19,", "test environment", 20, 304, AMBER, AMBER_L, "a"),
              ("A", "authority", "edits the repo,", "runs the tests", 300, 304, RED, RED_L, "r")]
    for k, name, l1, l2, x, y, c, cl, m in spokes:
        b.append(box(x, y, 240, 110, cl, c))
        b.append(text(x + 120, y + 34, f"{k}: {name}", 27, c, weight="bold"))
        b.append(text(x + 120, y + 68, l1, 23))
        b.append(text(x + 120, y + 96, l2, 23))
    b.append(arrow(230, 168, 170, 136, "b", BLUE, sw=3))
    b.append(arrow(220, 264, 170, 300, "a", AMBER, sw=3))
    b.append(arrow(340, 264, 390, 300, "r", RED, sw=3))
    b.append(text(420, 50, "one run exercises", 23, GRAY))
    b.append(text(420, 78, "all three at once", 23, GRAY))
    svg(HERE / "run_hsa.svg", 560, 424, "\n".join(b))


def seq_batch():
    """Batch generates and checks every candidate; sequential stops at a pass."""
    n = int(num("sel_cap"))
    b = [text(6, 26, f"Batch: all {val('sel_cap')}, every trial", 25, GRAY, "start", "bold")]
    for i in range(n):
        b.append(box(6 + i * 146, 38, 126, 46, GRAY_L, GRAY))
        b.append(text(69 + i * 146, 70, f"cand. {i + 1}", 24))
    b.append(text(6, 122, "Sequential: stop at a pass", 25, PURPLE, "start", "bold"))
    for i in range(n):
        b.append(box(6 + i * 146, 134, 126, 46, PURPLE_L, PURPLE, dash=i > 0))
        b.append(text(69 + i * 146, 166, f"cand. {i + 1}", 24, PURPLE))
        if i < n - 1:
            b.append(arrow(132 + i * 146, 157, 150 + i * 146, 157, "p", PURPLE))
    b.append(text(6, 212, f"{val('sel_en')} candidates on average", 23, INK, "start"))
    svg(HERE / "seq_batch.svg", 440, 222, "\n".join(b))


if __name__ == "__main__":
    for f in (hook, scaffold, verdicts, coverage, zheng, designs, wilson_fig, ranks,
              run_curves, kapoor, batch, dims, gym, confusion, kappa, recall_thumbs,
              run_hsa, seq_batch):
        f()
        print("wrote", f.__name__)
