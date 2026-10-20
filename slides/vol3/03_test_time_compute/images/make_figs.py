#!/usr/bin/env python3
"""Figures for Lecture 3 (Test-Time Compute), Volume III.

Every number drawn comes from calcs.py (derived values) or from a named
source. "TTC" is the rendered chapter, Test-Time Compute; "NM" a Napkin Math
notebook in it. Chapter scenario inputs are hypothetical, so the figures
that draw them say "illustrative".

  - hook: NM 3.1 (1,024 tokens at 21.075 ms; 40 ms test), illustrative.
  - axes: the three allocation axes (TTC 3.2, figure 2), redrawn at slide size.
  - alloc: NM 3.2, table 2, illustrative.
  - passk, poolcost: NM 3.3, illustrative.
  - coverage: Brown et al. 2024 (Large Language Monkeys), recomputed per-sample
    labels via MLSysIM Literature.RepeatedSamplingMajority / Coverage.
  - precision, loss: NM 3.4 and TTC equation 3, illustrative rates.
  - selection: TTC equation 4; exact Gaussian expected maximum (calcs.py).
  - topo: TTC figure 7 (where each search structure prunes), redrawn.
  - prune: TTC 3.6.2 (b = 5, r = 0.68, depth 3), illustrative.
  - plan: TTC table 5 and figure 5 (invalidation and scoped regeneration).
  - accounts: NM 3.6, illustrative.
  - stop: TTC figure 8 (stopping precedence), redrawn.
  - policy: NM 3.7, illustrative.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# Volume III agentic-stack roles (rules volumes/vol3/palette.md)
V3 = {
    "think": ("#e0f2fe", "#0284c7", "#0369a1"),   # compute / deliberation
    "mem": ("#dcfce7", "#16a34a", "#15803d"),     # memory / context
    "tool": ("#fef3c7", "#d97706", "#b45309"),    # actuation / tools (checks run here)
    "sys": ("#f3e8ff", "#7c3aed", "#6d28d9"),     # storage / system
    "bad": ("#fee2e2", "#dc2626", "#b91c1c"),     # faults
    "neutral": ("#f8fafc", "#94a3b8", "#0f172a"),
}


class plain_log_ticks:
    """Write log-axis ticks as plain numbers (1, 10, 100), not 10 with a
    superscript, whose exponent is too small to read in a column."""

    def __init__(self, every=1):
        self.every = every  # label every n-th decade, so wide ranges do not collide

    def __enter__(self):
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
        every = self.every
        self.save = sp.save

        def save(fig, path):
            for ax in fig.axes:
                if ax.get_xscale() == "log":
                    if every > 1:
                        ax.xaxis.set_major_locator(LogLocator(base=10.0 ** every))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
                    ax.xaxis.set_minor_formatter(NullFormatter())
            self.save(fig, path)
        sp.save = save

    def __exit__(self, *exc):
        sp.save = self.save


def tag(ax, text, x=0.0, y=-0.22):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=C["neutral"],
            ha="left", va="top")


def canvas(w, h, xmax=100.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax * h / w)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, role="neutral", fs=12, dashed=False, alpha=1.0, bold=False,
        lw=1.6):
    fill, stroke, txt = V3[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=stroke, lw=lw, ls="--" if dashed else "-",
                                alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=txt, alpha=alpha, zorder=3, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color="#64748b", ls="-", lw=1.6, alpha=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, ls=ls, alpha=alpha, zorder=1,
                                 shrinkA=0, shrinkB=0))


# --------------------------------------------------------------------------
# Hook: an explanation against a test (NM 3.1)
# --------------------------------------------------------------------------
def hook():
    h = calcs.hook()
    rows = [("Explain in 1,024 tokens", h["gen_s"] * 1000, "compute", f"{h['gen_s']:.2f} s"),
            ("Run the targeted test", 40, "data", "40 ms")]
    sp.hbars_steps(str(out / "hook"), rows, "ms", log=True, size=(5.8, 1.7),
                   xlabel="Time (ms, log scale)")
    # The same two bars drawn for a half-width column (the closing retrieval slide)
    with plain_log_ticks(every=2):
        sp.hbars(str(out / "hook_col.svg"), rows, "ms", log=True, size=(2.6, 0.9),
                 xlabel="Time (ms, log scale)")


# --------------------------------------------------------------------------
# Three allocation axes (TTC figure 2), one panel per step
# --------------------------------------------------------------------------
def axes():
    """Drawn 9 in wide and shown about 5.6 in wide: 15 pt text lands near 9.5 pt."""
    heads = [("Depth", "one longer chain"), ("Breadth", "sample, then select"),
             ("Feedback", "check, then revise")]
    for k in (1, 2, 3):
        fig, ax = canvas(9.0, 3.6)
        for i, (t, sub) in enumerate(heads):
            a = 1.0 if i < k else 0.0
            ax.text(16.5 + i * 33.5, 36.5, t, ha="center", fontsize=21, fontweight="bold",
                    color="#0f172a", alpha=a)
            ax.text(16.5 + i * 33.5, 32.3, sub, ha="center", fontsize=16, color="#475569", alpha=a)
        if k >= 1:
            box(ax, 1, 22, 13, 6.5, "Input", "neutral", fs=15)
            for j in range(4):
                box(ax, 2 + j * 7.5, 12, 5.5, 5.5, "", "think")
            arrow(ax, 5, 22, 4.6, 17.7)
            for j in range(3):
                arrow(ax, 7.5 + j * 7.5, 14.75, 9.5 + j * 7.5, 14.75)
            box(ax, 13, 1, 18, 7, "Candidate", "think", fs=15)
            arrow(ax, 27, 12, 24, 8.2)
        if k >= 2:
            x0 = 34.5
            box(ax, x0, 22, 13, 6.5, "Input", "neutral", fs=15)
            for j, y in enumerate((24, 17, 10)):
                box(ax, x0 + 16, y, 5, 5, "", "think")
                box(ax, x0 + 23, y, 5, 5, "", "think")
                arrow(ax, x0 + 13, 25, x0 + 16, y + 2.5)
                arrow(ax, x0 + 21, y + 2.5, x0 + 23, y + 2.5)
                arrow(ax, x0 + 28, y + 2.5, x0 + 29.5, 8.4)
            box(ax, x0 + 18, 1, 14, 7, "Select", "sys", fs=15)
        if k >= 3:
            x0 = 68
            box(ax, x0, 22, 16, 6.5, "Candidate", "think", fs=15)
            ax.add_patch(FancyBboxPatch((x0 + 18, 13), 13.5, 17.5, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc="none", ec=V3["tool"][1], lw=1.4, ls="--", zorder=1))
            ax.text(x0 + 24.75, 27.3, "sandbox", ha="center", fontsize=14, color=V3["tool"][2])
            box(ax, x0 + 19, 16.5, 11.5, 6.5, "Check", "tool", fs=15)
            arrow(ax, x0 + 16, 24, x0 + 19, 21)
            box(ax, x0 + 14, 1, 17.5, 7, "Observation", "tool", fs=14, bold=True)
            arrow(ax, x0 + 22.75, 16.5, x0 + 22.75, 8.2)
            box(ax, x0, 1, 11, 7, "Revised", "think", fs=15)
            arrow(ax, x0 + 14, 4.5, x0 + 11, 4.5)
            arrow(ax, x0 + 5.5, 22, x0 + 5.5, 8.2, ls="--")
        sp.save(fig, out / f"axes_step{k}.svg")


# --------------------------------------------------------------------------
# NM 3.2: one ceiling, three latencies and bills
# --------------------------------------------------------------------------
def alloc():
    a = calcs.axes()
    names = ["Depth", "Breadth", "Feedback"]
    keys = ["depth", "breadth", "feedback"]
    cols = [V3["think"][1], V3["think"][1], V3["tool"][1]]
    for k in (1, 2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 1.8))
        for ax, field, label, fmt, show in (
                (ax1, "time", "Generation and checks (s)", "{:.1f} s", True),
                (ax2, "bill", "Model-token bill ($)", "${:.3f}", k >= 2)):
            vals = [a[key][field] for key in keys]
            ys = [2, 1, 0]
            ax.barh(ys, vals if show else [0, 0, 0], color=cols, height=0.6)
            for y, v in zip(ys, vals):
                ax.text(v + max(vals) * 0.03, y, fmt.format(v), va="center", fontsize=12,
                        fontweight="bold", color=C["text"], alpha=1 if show else 0)
            ax.set_xlim(0, max(vals) * 1.55)
            ax.set_yticks(ys)
            ax.set_yticklabels(names if ax is ax1 else ["", "", ""], fontsize=12)
            ax.set_xlabel(label, fontsize=11)
            ax.set_xticks([])
            ax.tick_params(left=False)
            ax.spines["left"].set_visible(False)
        fig.subplots_adjust(wspace=0.12)
        sp.save(fig, out / f"alloc_step{k}.svg")


# --------------------------------------------------------------------------
# NM 3.3: pass@k bars and the cost of a parallel pool
# --------------------------------------------------------------------------
def passk():
    p = calcs.pool()
    rows = [(f"{k} candidate{'s' if k > 1 else ''}", v * 100, "data", f"{v * 100:.1f}%")
            for k, v in sorted(p["avail"].items())]
    sp.hbars_steps(str(out / "passk"), rows, "%", log=False, size=(3.6, 1.9),
                   xlabel="Pool holds a pass (%)")
    rows2 = [("Pool of 8", p["par_per_acc"], "compute", f"{p['par_per_acc']:.1f}"),
             ("Retry to a pass", p["seq_per_acc"], "data", f"{p['seq_per_acc']:.1f}")]
    sp.hbars(str(out / "poolcost.svg"), rows2, "", log=False, size=(3.6, 0.75),
             xlabel="Candidates per accepted repair")


# --------------------------------------------------------------------------
# Brown et al. 2024: coverage against majority vote (MLSysIM)
# --------------------------------------------------------------------------
def coverage():
    from mlsysim import Literature
    maj = Literature.RepeatedSamplingMajority.entries
    cov = Literature.RepeatedSamplingCoverage.entries
    series = (("Llama-3-8B-Instruct", "8B", C["memory"]), ("Llama-3-70B-Instruct", "70B", C["network"]))
    rep = sorted([e for e in cov if e["dataset"] == "SWE-bench Lite (all 300)"], key=lambda e: e["k"])
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.2))
        for model, lab, col in series:
            r = sorted([e for e in maj if e["model"] == model and e["dataset"] == "MATH"], key=lambda e: e["k"])
            ks = [e["k"] for e in r]
            ax.plot(ks, [100 * e["coverage"] for e in r], "-", color=col, lw=2.4)
            ax.text(ks[-1] * 1.25, 100 * r[-1]["coverage"] + (4 if lab == "70B" else -7),
                    f"{lab} coverage", color=col, fontsize=11, fontweight="bold", va="center")  # noqa
            if k >= 2:
                ax.plot(ks, [100 * e["majority_vote_accuracy"] for e in r], "--", color=col, lw=2.4)
                ax.text(ks[-1] * 1.25, 100 * r[-1]["majority_vote_accuracy"],
                        f"{lab} majority vote", color=col, fontsize=11, fontweight="bold", va="center")
        if k >= 3:
            ax.plot([e["k"] for e in rep], [100 * e["coverage"] for e in rep], ":", color=C["data"], lw=2.8)
            ax.text(14, 3, "Repair agent coverage, 1 to 250 samples (dotted)",
                    color=C["data"], fontsize=11, fontweight="bold", va="bottom", ha="left")
        ax.set_xscale("log")
        ax.set_xlim(1, 10_000)
        ax.set_ylim(0, 104)
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
        ax.set_xlabel("Samples per problem (log scale)")
        ax.set_ylabel("Problems (%)")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        fig.subplots_adjust(right=0.70)
        sp.save(fig, out / f"coverage_step{k}.svg")


# --------------------------------------------------------------------------
# NM 3.4: precision against the valid-candidate rate
# --------------------------------------------------------------------------
def precision():
    import numpy as np
    v = calcs.verifier()
    xs = np.logspace(-3, 0, 300)
    ys = [100 * calcs.precision(x, 0.10, 0.02) for x in xs]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.0))
        ax.plot(xs * 100, ys, color=C["text"], lw=2.4)
        ax.axvspan(0.1, 5, color=C["danger_l"], alpha=0.6, lw=0)
        if k >= 2:
            ax.plot([5], [v["prec"] * 100], "o", ms=11, color=C["danger"], mec="white", mew=1.5, zorder=5)
            ax.annotate(f"5% valid: {v['prec'] * 100:.1f}% of passes valid",
                        (5, v["prec"] * 100), xytext=(12, -24), textcoords="offset points",
                        ha="left", va="top", fontsize=12, fontweight="bold", color=C["danger"])
        if k >= 3:
            ax.plot([5], [v["prec_half_fa"] * 100], "o", ms=10, color=C["data"], mec="white", mew=1.5, zorder=5)
            ax.annotate(f"false acceptance halved: {v['prec_half_fa'] * 100:.1f}%", (5, v["prec_half_fa"] * 100),
                        xytext=(-12, 6), textcoords="offset points", ha="right", va="bottom",
                        fontsize=11, fontweight="bold", color=C["data"])
            ax.plot([10], [v["prec_double_v"] * 100], "s", ms=9, color=C["memory"], mec="white", mew=1.5, zorder=5)
            ax.annotate(f"valid rate doubled: {v['prec_double_v'] * 100:.1f}%", (10, v["prec_double_v"] * 100),
                        xytext=(10, -6), textcoords="offset points", ha="left", va="top",
                        fontsize=11, fontweight="bold", color=C["memory"])
        ax.set_xscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_ylim(0, 105)
        ax.set_xticks([0.1, 1, 10, 100])
        ax.set_xticklabels(["0.1", "1", "10", "100"])
        ax.set_xlabel("Valid candidates in the pool (%, log scale)")
        ax.set_ylabel("Passes that are valid (%)")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f"precision_step{k}.svg")
    rows = [("Incident loss", v["loss_attempt"], "danger", f"${v['loss_attempt']:.2f}"),
            ("Output bill", v["out_bill"], "neutral", f"${v['out_bill']:.2f}")]
    with plain_log_ticks():
        sp.hbars(str(out / "loss.svg"), rows, "", log=True, size=(3.4, 0.95),
                 xlabel="Dollars per attempt (log)")


# --------------------------------------------------------------------------
# Equation 4: the winner's scoring error
# --------------------------------------------------------------------------
def selection():
    import numpy as np
    s = calcs.selection()
    ns = sorted(s["exact"])
    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    ax.plot(ns, [s["exact"][n] for n in ns], "o-", color=C["danger"], lw=2.4, ms=7)
    na = np.array(sorted(s["asym"]))
    ax.plot(na, [s["asym"][n] for n in na], "--", color=C["neutral"], lw=1.8)
    ax.text(24, 2.75, "asymptote √(2 ln N)", color=C["neutral"], fontsize=12, ha="right", va="bottom")
    for n in (4, 64):
        ax.annotate(f"{s['exact'][n]:.1f}σ at N = {n}", (n, s["exact"][n]),
                    xytext=(6, -18) if n == 4 else (-4, -40), textcoords="offset points", fontsize=12,
                    fontweight="bold", color=C["danger"], ha="left" if n == 4 else "right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(-0.2, 3.2)
    ax.set_xlabel("Candidates scored, N")
    ax.set_ylabel("Winner's scoring error (σ)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "selection.svg")


# --------------------------------------------------------------------------
# Where each search structure prunes (TTC figure 7), one panel per step
# --------------------------------------------------------------------------
def node(ax, x, y, kind, r=1.8):
    styles = {"prompt": ("#e5e7eb", "#64748b", "-"), "gen": (V3["think"][0], V3["think"][1], "-"),
              "keep": (V3["mem"][0], V3["mem"][1], "-"), "pruned": ("#f1f5f9", "#94a3b8", "-"),
              "open": ("white", "#94a3b8", "--")}
    fc, ec, ls = styles[kind]
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec, lw=1.8, ls=ls, zorder=3))
    if kind == "pruned":
        for s in (1, -1):
            ax.plot([x - r * 0.6, x + r * 0.6], [y - s * r * 0.6, y + s * r * 0.6], color="#64748b",
                    lw=1.5, zorder=4)


def edge(ax, x0, y0, x1, y1, color="#94a3b8", lw=1.6):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1)


def topo():
    heads = [("Best-of-N", "prunes only at the end"), ("Beam", "prunes at each step"),
             ("Tree search", "grows one favored leaf")]
    for k in (1, 2, 3):
        fig, ax = canvas(9.0, 3.6)
        for i, (t, sub) in enumerate(heads):
            a = 1 if i < k else 0
            ax.text(16.5 + i * 33.5, 36.5, t, ha="center", fontsize=21, fontweight="bold", color="#0f172a", alpha=a)
            ax.text(16.5 + i * 33.5, 32.3, sub, ha="center", fontsize=16, color="#475569", alpha=a)
        if k >= 1:
            node(ax, 2.5, 16, "prompt")
            for j, y in enumerate([27, 20, 13, 6]):
                edge(ax, 2.5, 16, 8, y)
                for c in range(3):
                    node(ax, 8 + c * 5.5, y, "gen")
                    if c:
                        edge(ax, 8 + (c - 1) * 5.5, y, 8 + c * 5.5, y)
                edge(ax, 19, y, 27.5, y)
                node(ax, 27.5, y, "keep" if j == 2 else "pruned")
            ax.plot([24, 24], [3.5, 29.5], ls="--", color="#475569", lw=1.4)
            ax.text(24, 0.2, "select", ha="center", fontsize=15, color="#475569")
        if k >= 2:
            x0 = 35
            node(ax, x0 + 1.5, 16, "prompt")
            for y in (22, 10):
                edge(ax, x0 + 1.5, 16, x0 + 8, y)
                node(ax, x0 + 8, y, "gen")
            kids = [(27, "keep"), (19.5, "pruned"), (12.5, "keep"), (5, "pruned")]
            for (y, kind), par in zip(kids, [22, 22, 10, 10]):
                edge(ax, x0 + 8, par, x0 + 15, y)
                node(ax, x0 + 15, y, kind)
            ax.plot([x0 + 19] * 2, [3.5, 29.5], ls="--", color="#475569", lw=1.4)
            ax.text(x0 + 19, 0.2, "score, keep 2", ha="center", fontsize=15, color="#475569")
            for y, par in ((29, 27), (23.5, 27), (15, 12.5), (9.5, 12.5)):
                edge(ax, x0 + 15, par, x0 + 26, y)
                node(ax, x0 + 26, y, "gen")
        if k >= 3:
            x0 = 68.5
            node(ax, x0 + 1.5, 17, "prompt")
            pts = {"a": (x0 + 8.5, 26), "b": (x0 + 8.5, 17), "c": (x0 + 8.5, 8),
                   "d": (x0 + 15.5, 21), "e": (x0 + 15.5, 12), "f": (x0 + 22.5, 26), "g": (x0 + 22.5, 18)}
            parent = {"a": (x0 + 1.5, 17), "b": (x0 + 1.5, 17), "c": (x0 + 1.5, 17), "d": pts["b"],
                      "e": pts["b"], "f": pts["d"], "g": pts["d"]}
            for n_, (x, y) in pts.items():
                hot = n_ in ("b", "d", "g")
                edge(ax, parent[n_][0], parent[n_][1], x, y, color=V3["think"][1] if hot else "#94a3b8",
                     lw=3.5 if hot else 1.6)
            kinds = {"a": "gen", "b": "gen", "c": "open", "d": "gen", "e": "open", "f": "open", "g": "keep"}
            for n_, (x, y) in pts.items():
                node(ax, x, y, kinds[n_])
            ax.text(x0 + 22.5, 13.8, "new leaf\nscored", ha="center", va="top", fontsize=14,
                    color=V3["mem"][2], fontweight="bold")
            ax.text(x0 + 16, 0.2, "bold path: visits updated", ha="center", fontsize=15,
                    color=V3["think"][2])
        sp.save(fig, out / f"topo_step{k}.svg")


# --------------------------------------------------------------------------
# Step pruning bounds tree growth (TTC 3.6.2)
# --------------------------------------------------------------------------
def prune():
    p = calcs.pruning()
    rows = [("No pruning", p["unpruned"], "neutral", f"{p['unpruned']} nodes"),
            ("A flat 68% cut", p["flat"], "memory", f"{p['flat']:.1f} nodes (reference)"),
            ("Step check rejects 68%", p["pruned"], "compute", f"{p['pruned']:.1f} nodes")]
    sp.hbars_steps(str(out / "prune"), rows, "", log=False, size=(5.8, 1.6),
                   xlabel="Nodes generated through depth 3 (5 children per node)")


# --------------------------------------------------------------------------
# Revisable candidate plan (TTC table 5, figure 5)
# --------------------------------------------------------------------------
def plan():
    for k in (1, 2, 3):
        fig, ax = canvas(9.0, 3.5)
        inputs = [("Source excerpt", 31), ("Failure log", 23), ("Waiter-return test", 13),
                  ("API declaration", 2.5)]
        for name, y in inputs:
            role = "tool" if "test" in name else "neutral"
            a = 0.0 if ("test" in name and k == 1) else 1.0
            box(ax, 0.5, y, 23, 6.5, name, role, fs=14, alpha=a)
        dx = {1: "Diagnosis:\ntimeout too short", 2: "Diagnosis:\ntimeout too short",
              3: "Diagnosis':\nmissed notify"}[k]
        drole = {1: "think", 2: "bad", 3: "think"}[k]
        prole = "think" if k != 2 else "bad"
        box(ax, 27.5, 23.5, 23.5, 10, dx, drole, fs=15)
        box(ax, 54, 23.5, 21, 10, "Patch':\nnotify on return" if k == 3 else "Patch:\nraise timeout",
            prole, fs=14, dashed=(k == 2))
        box(ax, 78, 23.5, 21.5, 10, "Check':\nordering test" if k == 3 else "Check\nselection",
            prole, fs=15, dashed=(k == 2))
        box(ax, 27.5, 2.5, 23.5, 6.5, "API notes", "mem", fs=15)
        arrow(ax, 23.5, 33, 27.5, 30.5)
        arrow(ax, 23.5, 26.5, 27.5, 27.5)
        arrow(ax, 51, 28.5, 54, 28.5)
        arrow(ax, 75, 28.5, 78, 28.5)
        arrow(ax, 23.5, 5.75, 27.5, 5.75)
        arrow(ax, 23.5, 36.5, 60, 33.6, alpha=0.5)
        if k >= 2:
            arrow(ax, 23.5, 16, 29, 23.5, color=V3["bad"][1] if k == 2 else "#64748b",
                  ls="--" if k == 2 else "-")
        if k == 2:
            ax.text(27.5, 14.5, "contradicts", fontsize=15, color=V3["bad"][2], fontweight="bold")
            ax.text(77, 17, "Desc(diagnosis): needs review", fontsize=15, color=V3["bad"][2],
                    fontweight="bold", ha="center")
            ax.text(54, 5.75, "support kept", fontsize=15, color=V3["mem"][2], fontweight="bold", va="center")
        if k == 3:
            ax.add_patch(FancyBboxPatch((26.5, 22), 74, 13, boxstyle="round,pad=0,rounding_size=1.5",
                                        fc="none", ec=V3["think"][1], lw=1.5, ls="--"))
            ax.text(77, 17, "regenerated: 3 artifacts", fontsize=15, color=V3["think"][2],
                    fontweight="bold", ha="center")
            ax.text(54, 5.75, "retained", fontsize=15, color=V3["mem"][2], fontweight="bold", va="center")
        sp.save(fig, out / f"plan_step{k}.svg")


# --------------------------------------------------------------------------
# NM 3.6: four accounts for flat sampling and beam search
# --------------------------------------------------------------------------
def accounts():
    b = calcs.beam()
    fields = [("out", "Output tokens", "{:,.0f}"), ("inp", "Input tokens", "{:,.0f}"),
              ("time", "Time (s)", "{:.1f}"), ("bill", "Token bill ($)", "{:.3f}")]
    fig, axs = plt.subplots(1, 4, figsize=(5.8, 1.7))
    for ax, (f, lab, fmt) in zip(axs, fields):
        vals = [b["flat"][f], b["beam"][f]]
        ax.bar([0, 1], vals, color=[C["neutral"], C["compute"]], width=0.62)
        for x, v in zip([0, 1], vals):
            ax.text(x, v * 1.03, fmt.format(v), ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                    color=C["text"])
        ax.set_ylim(0, max(vals) * 1.32)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Flat", "Beam"], fontsize=11)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(lab, fontsize=12)
    fig.subplots_adjust(wspace=0.35)
    sp.save(fig, out / "accounts.svg")


# --------------------------------------------------------------------------
# Stopping precedence (TTC figure 8)
# --------------------------------------------------------------------------
def diamond(ax, x, y, w, h, text, role="neutral", alpha=1.0):
    fill, stroke, txt = V3[role]
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y)],
                         closed=True, fc=fill, ec=stroke, lw=1.8, alpha=alpha, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=15, color=txt,
            alpha=alpha, zorder=3)


def stop():
    steps = [("1. Ceiling\nreached?", "Stop admitting;\nkeep a pass,\nelse escalate", "bad"),
             ("2. Candidate\naccepted?", "Stop search;\nsend it to the\nacceptance gate", "mem"),
             ("3. Gain per\ndollar too low?", "Stop early;\nreport incomplete\nor escalate", "tool")]
    for k in (1, 2, 3):
        fig, ax = canvas(9.0, 3.4)
        box(ax, 0.3, 19, 14.5, 12, "Before\nadmitting\nwork", "neutral", fs=15)
        for i, (q, outcome, role) in enumerate(steps):
            a = 1.0 if i < k else 0.0
            x = 18.5 + i * 27.5
            diamond(ax, x, 17, 25, 16, q, "neutral", alpha=a)
            box(ax, x, 0.3, 25, 12.5, outcome, role, fs=14, alpha=a)
            ax.add_patch(FancyArrowPatch((x + 12.5, 17), (x + 12.5, 12.9), arrowstyle="-|>",
                                         mutation_scale=13, color="#64748b", lw=1.8, alpha=a))
            ax.text(x + 13.5, 14.2, "yes", fontsize=14, color="#475569", alpha=a)
            prev_x = 14.8 if i == 0 else x - 2.5
            ax.add_patch(FancyArrowPatch((prev_x, 25), (x, 25), arrowstyle="-|>", mutation_scale=13,
                                         color="#64748b", lw=1.8, alpha=a))
            if i:
                ax.text(x - 1.3, 26.3, "no", fontsize=14, color="#475569", ha="center", alpha=a)
        ax.text(99.5, 35.5, "no to all three: admit the next round", fontsize=14, color="#475569",
                ha="right", va="center", alpha=1.0 if k == 3 else 0.0)
        sp.save(fig, out / f"stop_step{k}.svg")


# --------------------------------------------------------------------------
# NM 3.7: three policies under one deadline
# --------------------------------------------------------------------------
def policy():
    p = calcs.policies()
    pts = [("A", "A: one candidate", C["neutral"]), ("C", "C: feedback, 2 rounds", C["data"]),
           ("B", "B: 8 in parallel", C["compute"])]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.0))
        for i, (key, lab, col) in enumerate(pts):
            if i >= k:
                continue
            d = p[key]
            ax.plot([d["per_acc"]], [d["pass_"] * 100], "o", ms=13, color=col, mec="white", mew=1.5, zorder=5)
            t = d["time"]
            ax.annotate(f"{lab}\n{d['pass_'] * 100:.0f}% pass, {t:.1f} s",
                        (d["per_acc"], d["pass_"] * 100),
                        xytext=(12, -4) if key != "B" else (-12, -4), textcoords="offset points",
                        ha="left" if key != "B" else "right", va="center", fontsize=11,
                        fontweight="bold", color=col)
        ax.set_xlim(0.03, 0.2)
        ax.set_ylim(20, 110)
        ax.set_xlabel("Model bill per accepted repair ($)")
        ax.set_ylabel("Repairs accepted (%)")
        ax.grid(color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f"policy_step{k}.svg")


# --------------------------------------------------------------------------
# Real artifacts typeset as code panels: exact excerpts, elisions marked
# --------------------------------------------------------------------------
MONO = ["Menlo", "DejaVu Sans Mono", "Courier New", "monospace"]


def code_panel(path, lines, width, fs=10.5, title=None, colors=None):
    """lines: list of strings; colors: {index: color} for highlighted lines."""
    colors = colors or {}
    # SVG collapses leading spaces; keep indentation with no-break spaces, and
    # size the panel from the longest line (Menlo advance is 0.602 em).
    lines = [ln.replace(" ", "\u00a0") for ln in lines]
    longest = max(len(ln) for ln in lines + ([title] if title else []))
    width = max(width, longest * fs * 0.602 / 72.0 + 0.35)
    n = len(lines) + (1 if title else 0)
    lh = fs * 1.45 / 72.0
    fig = plt.figure(figsize=(width, n * lh + 0.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.98, boxstyle="round,pad=0,rounding_size=0.02",
                                fc="#f8fafc", ec="#cbd5e1", lw=1.0, transform=ax.transAxes))
    y = 1 - 0.12 / (n * lh + 0.25)
    step = lh / (n * lh + 0.25)
    if title:
        ax.text(0.02, y, title, fontsize=fs, family="sans-serif", color="#475569", va="top",
                fontweight="bold", transform=ax.transAxes)
        y -= step
    for i, line in enumerate(lines):
        col = colors.get(i, "#0f172a")
        ax.text(0.02, y, line, fontsize=fs, family=MONO, color=col, va="top", transform=ax.transAxes,
                fontweight="bold" if i in colors else "normal", usetex=False, parse_math=False)
        y -= step
    sp.save(fig, path)


def panels():
    red, green, gray = "#b91c1c", "#15803d", "#64748b"
    # The real reproducer run (pool_trace.txt), split into its three cases
    runs = [blk.strip().splitlines() for blk in (out / "pool_trace.txt").read_text().strip().split("\n\n")]
    # Only the first case is drawn; the patch runs appear as text on the predict slide.
    for name, lines in zip(("trace_bug",), runs):
        cols = {i: (green if "PASS" in ln else red) for i, ln in enumerate(lines)
                if ln.startswith("probe") or ln.startswith("B blocked")}
        cols[0] = gray
        code_panel(out / f"{name}.svg", lines, 4.0, fs=11, colors=cols)
    # CPython 3.12 Lib/queue.py (branch 3.12 @ 58ed60b), exact lines, dedented 8 spaces
    q = ["def get(self, block=True, timeout=None):",
         "    with self.not_empty:",
         "        ...",
         "            while not self._qsize():",
         "                self.not_empty.wait()",
         "",
         "def put(self, item, block=True, timeout=None):",
         "    with self.not_full:",
         "        ...",
         "        self._put(item)",
         "        self.unfinished_tasks += 1",
         "        self.not_empty.notify()       # line 152"]
    code_panel(out / "queue_code.svg", q, 4.6, fs=10.5, colors={4: "#0369a1", 11: red})
    # openai/human-eval human_eval/evaluation.py @ b9453e8, lines 22 to 28, dedented 4 spaces
    he = ["def estimator(n: int, c: int, k: int) -> float:",
          '    """',
          "    Calculates 1 - comb(n - c, k) / comb(n, k).",
          '    """',
          "    if n - c < k:",
          "        return 1.0",
          "    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))"]
    code_panel(out / "humaneval_code.svg", he, 5.8, fs=11, colors={2: "#0369a1"})
    # openai/prm800k README annotated line @ 7ecc794 (exact fields; long strings cut at "...")
    prm = ['"problem": "...greatest common factor of $20 !$ and $200,\\\\!000$? ..."',
           '"pre_generated_steps": [',
           '  "I want to find the largest positive integer ...",',
           '  "One way to do this is to factor both numbers ...",',
           '  "I know that $200,\\\\!000 = 2^5\\\\cdot 10^4 = 2^9\\\\cdot 5^4$.",',
           '  ... 13 more steps ...',
           '"pre_generated_answer": "320,000",',
           '"ground_truth_answer": "40,\\\\!000",',
           '"pre_generated_verifier_score": 0.010779580529581414',
           '# label for step 3: "rating": -1   (scale -1, 0, +1)']
    code_panel(out / "prm800k.svg", prm, 5.8, fs=10, colors={4: red, 6: red, 8: "#0369a1", 9: red})


if __name__ == "__main__":
    panels()
    hook()
    axes()
    alloc()
    passk()
    coverage()
    precision()
    selection()
    topo()
    prune()
    plan()
    accounts()
    stop()
    policy()
    print("figures written to", out)
