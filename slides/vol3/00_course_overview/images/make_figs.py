#!/usr/bin/env python3
"""Figures for Lecture 0 (Volume III course overview).

Every value comes from calcs.py, which reads its sources:
  - METR Time Horizon 1.1 (metr_benchmark_results_1_1.yaml; Kwa et al. 2025)
  - 500 SWE-bench Verified runs of mini-SWE-agent 1.0.0 with Claude 4 Sonnet
    (swebench_runs.csv, swebench_run_django12050.csv; swebench_data.py)
  - Introduction NM 1.1 and 1.2, Foundation Model NM 2.2, Agent Economics
    NM 19.2, Agent Harness Sec. 10.10 (scenario values, labeled illustrative)
Diagrams redraw two Introduction figures at slide size: figure 1 (who
closes the loop) and figure 3 (the H·S·A space), in the Volume III role
palette (volumes/vol3/palette.md).

Run from this folder: python3 make_figs.py   (then make deck converts SVG to PDF)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402

plt = sp.plt
C = sp.C
OUT = Path(__file__).parent

# Volume III role palette (palette.md)
V3 = dict(
    compute=("#e0f2fe", "#0284c7", "#0369a1"),
    memory=("#dcfce7", "#16a34a", "#15803d"),
    tools=("#fef3c7", "#d97706", "#b45309"),
    system=("#f3e8ff", "#7c3aed", "#6d28d9"),
    fault=("#fee2e2", "#dc2626", "#b91c1c"),
    struct=("#f8fafc", "#94a3b8", "#0f172a"),
)
GRAY = "#64748b"
FONT = "Helvetica, Arial, sans-serif"


# =============================================================================
# SVG helpers (diagrams drawn at 900 x 500, labels 18 px or larger)
# =============================================================================
def svg_doc(body, w=900, h=500):
    markers = "".join(
        f'<marker id="arr-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for k, c in [("gray", GRAY), ("purple", V3["system"][1]), ("blue", V3["compute"][1]),
                     ("amber", V3["tools"][1])])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}"><defs>{markers}</defs>'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>')


def box(cx, cy, w, h, role, title, sub=None, tsize=21, ssize=18):
    fill, stroke, text = V3[role]
    s = (f'<rect x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}" rx="5" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>')
    if sub:
        s += (f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="{tsize}" '
              f'font-weight="bold" fill="#0f172a">{title}</text>'
              f'<text x="{cx}" y="{cy + 21}" text-anchor="middle" font-size="{ssize}" fill="{text}">{sub}</text>')
    else:
        s += (f'<text x="{cx}" y="{cy + 7}" text-anchor="middle" font-size="{tsize}" '
              f'font-weight="bold" fill="#0f172a">{title}</text>')
    return s


def arrow(x1, y1, x2, y2, color="gray", width=2.2):
    c = {"gray": GRAY, "purple": V3["system"][1], "blue": V3["compute"][1], "amber": V3["tools"][1]}[color]
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{width}" '
            f'marker-end="url(#arr-{color})"/>')


def label(x, y, s, size=18, anchor="middle", color="#334155", weight="normal", style="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" font-style="{style}">{s}</text>')


def write_svg(name, body, w=900, h=500):
    (OUT / f"{name}.svg").write_text(svg_doc(body, w, h))


# =============================================================================
# Who closes the loop (after Introduction figure 1), two steps
# =============================================================================
def loop_compare():
    """Canvas 900 x 420 so the figure fills a 6.2 cm slot; labels 20 px and up."""
    H = 420

    def panel_a():
        cx = 215
        s = label(10, 30, "A. A person closes the loop", 25, "start", "#0f172a", "bold")
        s += box(cx, 92, 290, 70, "compute", "Model endpoint", "answers only", 24, 21)
        s += box(cx, 228, 290, 70, "struct", "Developer", "carries everything", 24, 21)
        s += box(cx, 364, 290, 70, "tools", "Local terminal", "make test", 24, 21)
        s += arrow(cx - 60, 192, cx - 60, 128) + label(cx - 70, 168, "prompt + error", 21, "end")
        s += arrow(cx + 60, 128, cx + 60, 192) + label(cx + 70, 168, "patch", 21, "start")
        s += arrow(cx - 60, 264, cx - 60, 328) + label(cx - 70, 304, "runs", 21, "end")
        s += arrow(cx + 60, 328, cx + 60, 264) + label(cx + 70, 304, "error", 21, "start")
        return s

    def panel_b():
        s = '<line x1="440" y1="10" x2="440" y2="410" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6,6"/>'
        s += label(458, 30, "B. A runtime closes the loop", 25, "start", "#0f172a", "bold")
        s += box(672, 92, 300, 70, "compute", "Foundation model", "holds no authority", 24, 21)
        s += box(672, 228, 300, 70, "system", "Agent runtime", "grants each action", 24, 21)
        s += box(560, 364, 190, 70, "tools", "Sandbox", "isolated", 24, 21)
        s += box(790, 364, 190, 70, "tools", "Verifiers", "tests", 24, 21)
        s += arrow(615, 192, 615, 128) + label(605, 168, "context", 21, "end")
        s += arrow(730, 128, 730, 192) + label(740, 168, "proposal", 21, "start")
        s += arrow(520, 264, 520, 328, "purple") + label(510, 304, "dispatch", 21, "end", V3["system"][2])
        s += arrow(600, 328, 600, 264) + label(610, 304, "observation", 21, "start")
        s += arrow(790, 328, 785, 264) + label(797, 304, "evidence", 21, "start")
        s += arrow(656, 364, 694, 364)
        return s

    write_svg("loop_compare_step1", panel_a(), h=H)
    write_svg("loop_compare_step2", panel_a() + panel_b(), h=H)


# =============================================================================
# H·S·A space with the four lighthouses (after Introduction figure 3), 5 steps
# =============================================================================
def sub(base, idx):
    return f'{base}<tspan font-size="14" dy="5">{idx}</tspan><tspan dy="-5"> </tspan>'


def hsa():
    x0, x1 = 250, 870            # horizon axis, log10 1 .. 3000
    yb = 360                     # baseline (A0 bottom)
    band = 76                    # height of one authority band

    def hx(h):
        return x0 + (x1 - x0) * np.log10(h) / np.log10(3000)

    def ay(a):                   # center of authority band a
        return yb - band * (a + 0.5)

    base = ""
    names = ["read only", "workspace", "compensable", "irreversible"]
    for a in range(4):
        fill = "#fef2f2" if a == 3 else ("#f1f5f9" if a % 2 == 0 else "#f8fafc")
        base += f'<rect x="{x0}" y="{yb - band * (a + 1)}" width="{x1 - x0}" height="{band}" fill="{fill}" stroke="#e2e8f0"/>'
        color = V3["fault"][2] if a == 3 else "#334155"
        base += (f'<text x="{x0 - 12}" y="{ay(a) + 8}" text-anchor="end" font-size="22" fill="{color}">'
                 f'A<tspan font-size="16" dy="5">{a}</tspan><tspan dy="-5" dx="6">{names[a]}</tspan></text>')
    base += arrow(x0, yb, x1 + 20, yb) + arrow(x0, yb, x0, yb - 4 * band - 22)
    for h in [1, 10, 100, 1000]:
        base += label(hx(h), yb + 30, f"{h:,}", 21)
    base += label(x1 + 20, yb + 58, "Horizon H (turns)", 22, "end", "#0f172a", "bold")
    base += label(x0, 22, "Authority", 22, "middle", "#0f172a", "bold")
    base += label(x1, 22, "S = carried state, levels 0 to 3", 21, "end", "#475569")
    base += f'<circle cx="{x0}" cy="{yb}" r="7" fill="#0f172a"/>'
    base += label(x0 + 14, yb - 12, "one model call", 20, "start", "#475569", style="italic")

    pts = [  # name, H, state level, authority level, label anchor, dx, dy, color
        ("Repair agent", 100, 2, 1, "start", 16, -6, "#1e293b"),
        ("Research agent", 1000, 3, 0, "end", -16, -16, "#1e293b"),
        ("Remediation agent", 20, 3, 3, "start", 16, -6, V3["fault"][2]),
        ("Support agent", 3, 1, 0, "start", 16, -16, "#1e293b"),
    ]
    layers = [base]
    for name, h, s_lvl, a_lvl, anc, dx, dy, col in pts:
        x, y = hx(h), ay(a_lvl)
        hs = {100: "H ≈ 100", 1000: "H ≥ 1,000", 20: "H ≈ 20", 3: "H ≤ 3"}[h]
        g = f'<circle cx="{x}" cy="{y}" r="11" fill="{col}"/>'
        g += label(x + dx, y + dy, name, 24, anc, col, "bold")
        g += (f'<text x="{x + dx}" y="{y + dy + 26}" text-anchor="{anc}" font-size="21" fill="#334155">'
              f'{hs} · S<tspan font-size="16" dy="5">{s_lvl}</tspan><tspan dy="-5" dx="6">· A</tspan>'
              f'<tspan font-size="16" dy="5">{a_lvl}</tspan></text>')
        layers.append(g)
    for k in range(1, 6):
        write_svg(f"hsa_step{k}", "".join(layers[:k]), h=430)


# =============================================================================
# Plots (slideplot style, drawn at slot size)
# =============================================================================
def metr_plot():
    m = calcs.metr()
    import datetime as dt
    pts = [(dt.date.fromisoformat(d), v, s) for d, v, s in m["points"]]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.75))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cols = [C["compute"] if p[2] else C["neutral_l"] for p in pts]
        ax.scatter(xs, ys, s=46, c=cols, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_yscale("log")
        ticks = [1 / 60, 1, 60, 16 * 60]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["1 s", "1 min", "1 h", "16 h"])
        ax.minorticks_off()
        ax.set_ylim(1 / 600, 4000)
        ax.set_ylabel("Task length at\n50% success")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        first, last = pts[0], max(pts, key=lambda p: p[1])
        alpha = 1.0 if step == 2 else 0.0
        ax.annotate(f"{first[0].year}: {m['lo_s']:.0f} s", (first[0], first[1]), xytext=(10, -2),
                    textcoords="offset points", fontsize=12, fontweight="bold", color=C["compute"],
                    alpha=alpha, va="center")
        ax.annotate(f"{last[0].year}: {m['hi_h']:.0f} h", (last[0], last[1]), xytext=(-10, 4),
                    textcoords="offset points", fontsize=12, fontweight="bold", color=C["compute"],
                    alpha=alpha, ha="right")
        ax.text(0.02, 0.95, f"Doubles every {m['dbl_all']:.1f} months;\n{m['dbl_2023']:.1f} months since 2023",
                transform=ax.transAxes, va="top", fontsize=12, color=C["text"], alpha=alpha)
        ax.text(0.99, 0.02, "orange: frontier model at release", transform=ax.transAxes, ha="right",
                fontsize=10, color=C["neutral"])
        sp.save(fig, OUT / f"metr_horizon_step{step}.svg")


def horizon_curve():
    h = calcs.horizon()
    t = np.arange(1, 101)
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        ax.plot(t, 100 * calcs.STEP_OK ** t, color=C["danger"], lw=3)
        ax.text(60, 100 * calcs.STEP_OK ** 60 + 6, "99% per step", color=C["danger"], fontsize=12,
                fontweight="bold")
        ax.plot([10], [100 * h["p10"]], "o", ms=9, color=C["danger"])
        ax.text(2, 58, f"{h['p10']:.1%} at 10 turns", fontsize=12, fontweight="bold", color=C["danger"])
        a2 = 1.0 if step >= 2 else 0.0
        ax.plot([100], [100 * h["p100"]], "o", ms=9, color=C["danger"], alpha=a2)
        ax.annotate(f"{h['p100']:.1%} at 100", (100, 100 * h["p100"]), xytext=(-8, -22),
                    textcoords="offset points", ha="right", fontsize=12, fontweight="bold",
                    color=C["danger"], alpha=a2)
        a3 = 1.0 if step >= 3 else 0.0
        ax.plot(t, 100 * (1 - h["eps_needed"]) ** t, color=C["data"], lw=2.5, ls="--", alpha=a3)
        ax.text(2, 8, f"dashed: {h['eps_needed']:.2%} error per step holds 90%", color=C["data"],
                fontsize=12, fontweight="bold", alpha=a3, va="bottom", ha="left")
        ax.set_xlim(0, 103)
        ax.set_ylim(0, 108)
        ax.set_xlabel("Horizon (turns)")
        ax.set_ylabel("Task success (%)")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, OUT / f"horizon_curve_step{step}.svg")


def runs_by_calls():
    r = calcs.runs()
    labels, vals, ns = [], [], []
    for lo, hi, n, a in r["bins"]:
        labels.append(f"{lo}-{hi - 1}" if hi < 1000 else f"{lo}+")
        vals.append(100 * a)
        ns.append(n)
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    x = np.arange(len(vals))
    colors = [C["data"] if v >= 70 else (C["compute"] if v >= 55 else C["danger"]) for v in vals]
    ax.bar(x, vals, color=colors, width=0.62)
    for xi, v, n in zip(x, vals, ns):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(xi, 4, f"n={n}", ha="center", fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Model calls in the run")
    ax.set_ylabel("Resolved (%)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / "runs_by_calls.svg")


def resend_tokens():
    turns = np.arange(1, calcs.TURNS + 1)
    sent = [calcs.resent_after(int(k)) / 1e3 for k in turns]
    fresh = [calcs.fresh_after(int(k)) / 1e3 for k in turns]
    b = calcs.bill()
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(4.3, 3.3))
        ax.plot(turns, sent, color=C["danger"], lw=3)
        ax.text(17, 780, f"resent\n{b['sent'] / 1e3:,.1f}k", ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["danger"])
        a = 1.0 if step == 2 else 0.0
        ax.plot(turns, fresh, color=C["data"], lw=3, alpha=a)
        ax.annotate(f"fresh {b['fresh'] / 1e3:,.1f}k", (turns[-1], fresh[-1]), xytext=(-4, 10),
                    textcoords="offset points", ha="right", fontsize=13, fontweight="bold",
                    color=C["data"], alpha=a)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Input tokens (thousands)")
        ax.set_xlim(0, 31)
        ax.set_ylim(0, 1000)
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, OUT / f"resend_tokens_step{step}.svg")


def real_resend():
    o = calcs.one_run()
    calls = [int(r["call"]) for r in o["rows"]]
    sent = [int(r["cum_sent_chars"]) / 1e3 for r in o["rows"]]
    new = [int(r["cum_new_chars"]) / 1e3 for r in o["rows"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(calls, sent, color=C["danger"], lw=3, marker="o", ms=4)
    ax.plot(calls, new, color=C["data"], lw=3, marker="o", ms=4)
    ax.text(1, 330, f"sent to the model\n{o['sent'] / 1e3:,.0f}k characters", ha="left", va="center",
            fontsize=12, fontweight="bold", color=C["danger"])
    ax.annotate(f"new text {o['new'] / 1e3:,.0f}k", (calls[-1], new[-1]), xytext=(-4, 10),
                textcoords="offset points", ha="right", fontsize=12, fontweight="bold", color=C["data"])
    ax.set_xlabel("Model call")
    ax.set_ylabel("Characters (thousands)")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 430)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "real_resend.svg")


def bill_bars():
    b = calcs.bill()
    sp.hbars(OUT / "bill_bars.svg", [
        ("No reuse", b["no_reuse"], "danger", f"${b['no_reuse']:.2f}"),
        ("Ideal prefix reuse", b["reuse"], "data", f"${b['reuse']:.2f}"),
    ], "$", log=False, xlabel="Token bill for one 30-turn trajectory ($, illustrative prices)",
        size=sp.SHORT)


def real_cost():
    r = calcs.runs()
    sp.hbars_steps(OUT / "real_cost", [
        ("Per attempt", r["per_attempt"], "neutral", f"${r['per_attempt']:.2f}"),
        ("Per accepted repair", r["per_accepted"], "compute", f"${r['per_accepted']:.2f}"),
    ], "$", log=False, xlabel="Dollars, 500 SWE-bench Verified tasks", size=(4.4, 1.9))


def freeze_curve():
    h = calcs.horizon()
    t = np.arange(0, calcs.H_FREEZE + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(t, 100 * calcs.STEP_OK ** t, color=C["danger"], lw=3)
    ax.plot([30], [100 * h["p30"]], "o", ms=10, color=C["danger"])
    ax.annotate(f"{h['p30']:.0%} after 30 turns", (30, 100 * h["p30"]), xytext=(-8, -24),
                textcoords="offset points", ha="right", fontsize=13, fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 31.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Instruction held\non every turn (%)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, OUT / "freeze_curve.svg")


def cpa_bars():
    c = calcs.cost_per_accepted()
    sp.hbars(OUT / "cpa_bars.svg", [
        ("Baseline", c["base"], "neutral", f"${c['base']:.2f}"),
        ("Halve token price", c["cheaper"], "compute", f"${c['cheaper']:.2f}"),
        ("Acceptance 40% to 60%", c["better"], "data", f"${c['better']:.2f}"),
    ], "$", log=False, xlabel="Cost per accepted task ($, illustrative)", size=(3.0, 1.5))


if __name__ == "__main__":
    loop_compare()
    hsa()
    metr_plot()
    horizon_curve()
    runs_by_calls()
    resend_tokens()
    real_resend()
    bill_bars()
    real_cost()
    freeze_curve()
    cpa_bars()
    print("figures written")
