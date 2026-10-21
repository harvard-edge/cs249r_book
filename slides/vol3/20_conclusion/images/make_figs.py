"""Lecture 20 (Conclusion): every figure the deck draws.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=20_conclusion`
converts them to PDF). Plots use the shared slideplot helper at the size of
their slot; diagrams are SVG drawn at the size of their slot. Derived values
come from MLSysIM through calcs.py (its scenarios and the values it records,
read here with val and num); layout coordinates are typed. The one exception
is recall_bars, whose Lecture 19 inputs wait for that lecture's Agent
Economics scenario. Text-free thumbnails come from tools/greek_thumbs.py.
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = Path(__file__).resolve().parents[3] / "assets" / "kit"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(HERE))

import calcs  # noqa: E402
import slideplot as sp  # noqa: E402
from deckvals import val, num  # noqa: E402
from mlsysim.physics.agents import calc_unchecked_success  # noqa: E402
from slideplot import plt  # noqa: E402

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
                  for n, c in [("g", GRAY), ("p", PURPLE), ("a", AMBER), ("r", RED), ("b", BLUE),
                               ("n", GREEN), ("k", INK)])
        + f'</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n' + "\n".join(body) + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, rx=10, dash=False, sw=2.5):
    d = ' stroke-dasharray="8 6"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=22, color=INK, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def arrow(x1, y1, x2, y2, m="g", color=GRAY, dash=False, sw=2.5):
    d = ' stroke-dasharray="7 5"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
            f'marker-end="url(#a{m})"/>')


def card(x, y, w, h, title, sub, fill, stroke, tcolor=INK, size=21, dash=False):
    out = [box(x, y, w, h, fill, stroke, dash=dash)]
    if sub:
        out.append(text(x + w / 2, y + h / 2 - 4, title, size, tcolor, weight="bold"))
        out.append(text(x + w / 2, y + h / 2 + 22, sub, size - 2, stroke))
    else:
        out.append(text(x + w / 2, y + h / 2 + 7, title, size, tcolor, weight="bold"))
    return out


# ---------------------------------------------------------------- diagrams
def capstone():
    """The incident crosses Parts I to IV; Parts V and VI never enter (Ch 20 figure 1)."""
    W, H = 900, 400
    for step in (1, 2, 3):
        b = []
        # harness around every step (Part IV)
        b.append(box(150, 70, 600, 236, PURPLE_L, PURPLE, rx=14))
        b.append(text(170, 100, "Harness around every step", 21, PURPLE, "start", "bold"))
        b.append(text(730, 100, "Part IV", 19, PURPLE, "end"))
        b += card(190, 120, 230, 66, "Context assembly", "Part II", GREEN_L, GREEN)
        b += card(480, 120, 230, 66, "Model call", "Part I", BLUE_L, BLUE)
        b += card(480, 222, 230, 66, "Tool call", "Part III", AMBER_L, AMBER)
        b += card(190, 222, 230, 66, "Sandbox", "Part III", AMBER_L, AMBER)
        b.append(arrow(420, 153, 478, 153))
        b.append(arrow(595, 186, 595, 220))
        b.append(text(605, 210, "proposal", 18, GRAY, "start"))
        b.append(arrow(480, 255, 422, 255, "a", AMBER))
        b.append(arrow(305, 222, 305, 188, "a", AMBER, dash=True))
        b.append(text(295, 210, "observation", 18, AMBER, "end"))
        if step >= 2:
            b += card(300, 4, 300, 50, "Task contract, adopted", None, GRAY_L, INK, size=20)
            b.append(arrow(450, 54, 450, 68, "k", INK))
            b += card(300, 330, 300, 56, "Acceptance on evidence", None, GRAY_L, INK, size=20)
            b.append(arrow(450, 306, 450, 328, "k", INK))
            b.append(text(462, 322, "settled outcome", 17, GRAY, "start"))
        if step >= 3:
            b += card(10, 330, 270, 56, "Learning: not needed", None, "white", GRAY, GRAY, 19, dash=True)
            b += card(620, 330, 270, 56, "Delegation: not needed", None, "white", GRAY, GRAY, 19, dash=True)
            b.append(text(20, 318, "Part V", 17, GRAY, "start"))
            b.append(text(880, 318, "Part VI", 17, GRAY, "end"))
        svg(HERE / f"capstone_step{step}.svg", W, H, b)


def lostreply():
    """op_42: intent before dispatch, the lost reply, settlement by status query."""
    W, H = 900, 300
    T, E = 46, 206          # top of the runtime boxes, top of the endpoint boxes
    for step in (1, 2, 3):
        b = []
        b.append(text(6, 30, "Runtime", 21, INK, "start", "bold"))
        b.append(text(6, 296, "Endpoint", 21, INK, "start", "bold"))
        b.append('<line x1="100" y1="24" x2="890" y2="24" stroke="#cbd5e1" stroke-width="2"/>')
        b.append('<line x1="110" y1="290" x2="890" y2="290" stroke="#cbd5e1" stroke-width="2"/>')
        b.append(box(100, T, 220, 74, PURPLE_L, PURPLE))
        b.append(text(210, T + 30, "1. Intent for op_42", 21, PURPLE, weight="bold"))
        b.append(text(210, T + 58, "and its owner, on disk", 20, INK))
        b.append(arrow(300, T + 74, 372, E - 2, "k", INK))
        b.append(text(104, 154, "2. Dispatch op_42,", 20, INK, "start"))
        b.append(text(104, 180, "expects config_v7", 20, INK, "start"))
        if step >= 2:
            b.append(box(330, E, 190, 74, GREEN_L, GREEN))
            b.append(text(425, E + 30, "3. Applies", 21, GREEN, weight="bold"))
            b.append(text(425, E + 58, "config_v8", 20, INK))
            b.append(arrow(478, E - 2, 532, T + 78, "r", RED, dash=True))
            b.append(text(476, 150, "reply", 20, RED, "end", "bold"))
            b.append(text(476, 174, "lost", 20, RED, "end", "bold"))
            b.append(box(520, T, 170, 74, RED_L, RED))
            b.append(text(605, T + 30, "4. op_42 is", 21, RED, weight="bold"))
            b.append(text(605, T + 58, "unknown", 21, RED, weight="bold"))
        if step >= 3:
            b.append(arrow(612, T + 76, 650, E - 2, "k", INK))
            b.append(text(652, 152, "5. Status?", 20, INK, "start"))
            b.append(text(652, 178, "(read-only)", 20, INK, "start"))
            b.append(box(600, E, 210, 74, GREEN_L, GREEN))
            b.append(text(705, E + 30, "Applied:", 21, GREEN, weight="bold"))
            b.append(text(705, E + 58, "config_v8", 20, INK))
            b.append(arrow(812, E - 2, 830, T + 76, "n", GREEN))
            b.append(box(718, T, 172, 74, GREEN_L, GREEN))
            b.append(text(804, T + 30, "6. Settled,", 21, GREEN, weight="bold"))
            b.append(text(804, T + 58, "no repeat", 21, GREEN, weight="bold"))
        svg(HERE / f"lostreply_step{step}.svg", W, H, b)


def barriers():
    """An injected instruction meets three independent barriers (§20.2.2)."""
    W, H = 480, 340
    rows = [("Action gate", "scope names", "no such host"),
            ("Egress policy", "socket to unlisted", "host dropped"),
            ("Flow label", "credential may not", "reach recipient")]
    for step in (1, 2, 3):
        b = [box(6, 4, 468, 64, RED_L, RED)]
        b.append(text(240, 30, "Injected log line: send the", 21, RED, weight="bold"))
        b.append(text(240, 56, "database credential out", 21, RED, weight="bold"))
        for i, (t, l1, l2) in enumerate(rows):
            y = 90 + i * 84
            op = "1" if i < step else "0"
            c, cl = (AMBER, AMBER_L) if i == 1 else (PURPLE, PURPLE_L)
            b.append(f'<g opacity="{op}">')
            b.append(arrow(240, y - 20, 240, y - 2, "k", INK))
            b.append(box(6, y, 468, 66, cl, c))
            b.append(text(20, y + 41, t, 22, c, "start", "bold"))
            b.append(text(462, y + 28, l1, 20, INK, "end"))
            b.append(text(462, y + 54, l2, 20, INK, "end"))
            b.append("</g>")
        svg(HERE / f"barriers_step{step}.svg", W, H, b)


def records():
    """Five records after op_42 commits; a cache hit is not freshness (§20.4)."""
    W, H = 500, 360
    rows = [("Source record", "config_v8", GREEN, GREEN_L),
            ("Event log", "op_42 settled", PURPLE, PURPLE_L),
            ("Agent memory", "says config_v7", RED, RED_L),
            ("Derived index", "says config_v7", RED, RED_L),
            ("KV cache", "valid for its tokens", BLUE, BLUE_L)]
    before = ["config_v7", "op_42 pending", "config_v7", "config_v7", "valid for its tokens"]
    for step in (1, 2):
        b = []
        for i, (role, state, c, cl) in enumerate(rows):
            y = 4 + i * 71
            b.append(box(6, y, 488, 62, cl if step == 2 else GRAY_L, c if step == 2 else GRAY))
            b.append(text(20, y + 39, role, 22, INK, "start", "bold"))
            b.append(text(480, y + 39, state if step == 2 else before[i], 21,
                          c if step == 2 else INK, "end", "bold"))
        svg(HERE / f"records_step{step}.svg", W, H, b)


def measured():
    """Our run: four runtimes, 25,000 incidents each (images/data/settle_run.json)."""
    m = calcs.measured_run()["runtimes"]
    rows = [("Retry blindly", m["retry"]["duplicates"], RED, "duplicated changes"),
            ("Assume success", m["assume"]["false_closures"], AMBER, "closed, never applied"),
            ("Halt", m["halt"]["unowned_unknowns"], GRAY, "left with no owner"),
            ("Settle by status", m["settle"]["duplicates"] + m["settle"]["false_closures"]
             + m["settle"]["unowned_unknowns"], GREEN, "failures")]
    fig, ax = plt.subplots(figsize=(1.9, 1.45))
    for i, (lab, v, c, what) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=c, height=0.6)
        ax.text(v + 3, y, f"{v} {what}", va="center", fontsize=8.5, color=c, fontweight="bold")
        ax.text(-3, y, lab, ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xlim(0, 260)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Bad outcomes in {val('ms_n')} incidents (measured)")
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "measured.svg")


def recall_l19():
    """NM 19.1, Ch 19 table 1, read from calcs.py (Scenarios.RepairCohortTokenPriceComparison)."""
    return dict(rej_per_acc_frontier=num("rc_rej_fr"), rej_per_acc_cheap=num("rc_rej_lp"))


def recall_bars():
    """L19 recall: rejected attempts carried by each accepted repair (NM 19.1)."""
    r = recall_l19()
    fig, ax = plt.subplots(figsize=(3.0, 1.4))
    ax.barh(1, r["rej_per_acc_frontier"], color=BLUE, height=0.55)
    ax.barh(0, r["rej_per_acc_cheap"], color=RED, height=0.55)
    ax.set_yticks([])
    ax.set_xlim(0, 3.5)
    ax.spines["left"].set_visible(False)
    sp.save(fig, HERE / "recall_bars.svg")


def bill():
    """incident_42: where one attempt's money goes, and the unchecked loss (§20.3.2)."""
    rows = [("Model calls", num("bill_model"), BLUE, f"${val('bill_model')}"),
            ("Telemetry and change tools", num("bill_tools"), AMBER, f"${val('bill_tools')}"),
            ("Health-window check", num("bill_health"), GREEN, f"${val('bill_health')}"),
            ("On-call approval", num("bill_approval2"), PURPLE, f"${val('bill_approval2')}"),
            ("Wrong change, no check", num("bill_loss"), RED, f"${val('bill_loss')} expected loss")]
    fig, ax = plt.subplots(figsize=(3.9, 1.15))
    for i, (lab, v, c, s) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, color=c, height=0.62)
        ax.text(v * 1.15, y, s, va="center", fontsize=8.5, color=c, fontweight="bold")
        ax.text(0.042, y, lab, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 8000)
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["$0.10", "$1", "$10", "$100", "$1,000"])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Dollars per attempt (log scale; illustrative profile)")
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "bill.svg")


def swe_agent():
    """SWE-agent ablations, SWE-bench Lite (Yang et al. 2024, Tables 1 and 3)."""
    swe = calcs.SWE
    rows = [("SWE-agent interface", swe.SweBenchLiteSweAgentResolved),
            ("Same interface, Claude 3 Opus", swe.SweAgentClaude3OpusLiteResolved),
            ("Viewer shows whole file", swe.SweAgentFullFileViewerResolved),
            ("Iterative search", swe.SweAgentIterativeSearchResolved),
            ("Shell only", swe.SweBenchLiteShellResolved),
            ("No edit command", swe.SweAgentNoEditCommandResolved)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (lab, v) in enumerate(rows):
        y = len(rows) - 1 - i
        model = "Opus" in lab
        c = BLUE if model else (PURPLE if i == 0 else AMBER)
        ax.barh(y, v, color=c, height=0.62)
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=12, color=c, fontweight="bold")
        ax.text(-0.3, y, lab, ha="right", va="center", fontsize=12, color=INK)
    ax.set_xlim(0, 22)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Issues resolved, SWE-bench Lite (%)")
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 11)
    sp.save(fig, HERE / "swe_agent.svg")


def passing():
    """Stacked bars: candidates against the passing set (§20.10, eq. 3.3)."""
    inv0 = num("ps_invalid")
    inv1 = num("ps_after")
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        ax.bar(0, 100 - inv0, color=GREEN, width=0.55)
        ax.bar(0, inv0, bottom=100 - inv0, color=RED, width=0.55)
        ax.text(0, 100 - inv0 / 2, f"{val('ps_invalid')}%\ninvalid", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        ax.text(0, (100 - inv0) / 2, f"{val('ps_valid')}%\nvalid", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")
        ax.annotate("", xy=(0.62, 55), xytext=(0.38, 55),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
        ax.text(0.5, 60, "check", ha="center", fontsize=10, color=INK)
        if step == 1:
            ax.bar(1, 100, color=GRAY_L, width=0.55, edgecolor=GRAY)
            ax.text(1, 50, "???", ha="center", va="center", fontsize=16, color=RED, fontweight="bold")
        else:
            ax.bar(1, 100 - inv1, color=GREEN, width=0.55)
            ax.bar(1, inv1, bottom=100 - inv1, color=RED, width=0.55)
            ax.text(1.32, 100 - inv1 / 2, f"{val('ps_after')}%\ninvalid", va="center", fontsize=10,
                    color=RED, fontweight="bold")
            ax.text(1, 45, "valid", ha="center", va="center", color="white", fontsize=10,
                    fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Candidates", "Passing set"])
        ax.set_xlim(-0.45, 1.75)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Share (%)")
        axis_type(ax, 9.5)
        sp.save(fig, HERE / f"passing_step{step}.svg")


def swebp():
    """SWE-bench+ Table 1: what the 251 passing patches were (Aleithan et al. 2024)."""
    a = calcs.AUDIT
    parts = [("Correct fix", a.correct, GREEN), ("Weak tests passed it", a.weak_tests, AMBER),
             ("Answer was in the issue", a.solution_leak, RED)]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    bottom = 0
    for lab, v, c in parts:
        ax.bar(0, v, bottom=bottom, color=c, width=0.5)
        ax.text(0.32, bottom + v / 2, f"{lab}: {v:.0f}", va="center", fontsize=9.5, color=c,
                fontweight="bold")
        bottom += v
    ax.set_xlim(-0.35, 1.9)
    ax.set_ylim(0, 260)
    ax.set_xticks([0])
    ax.set_xticklabels([f"{val('sbp_total')} passing patches"])
    ax.set_ylabel("Patches")
    axis_type(ax, 9.5)
    sp.save(fig, HERE / "swebp.svg")


def metr_hack():
    """Part V evidence: reward hacking with the scorer in reach (Ch 17 WS 17.1)."""
    rows = [("RE-Bench", "metr_rebench", RED), ("HCAST", "metr_hcast", GRAY)]
    fig, ax = plt.subplots(figsize=(1.3, 0.48))
    for i, (lab, k, c) in enumerate(rows):
        y = 1 - i
        v = num(f"{k}_pct")
        ax.barh(y, v, color=c, height=0.6)
        ax.text(v + 1, y, f"{val(k + '_hacks')} of {val(k + '_runs')} runs, {val(k + '_pct')}%", va="center",
                fontsize=8.5, color=c, fontweight="bold")
        ax.text(-1, y, lab, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "metr_hack.svg")


def attribution():
    """Counterfactual search: 4^50 paths against one checked 5-step window (§20.8)."""
    a = calcs.BLAME
    fig, ax = plt.subplots(figsize=(2.1, 0.8))
    rows = [(f"No checks, {val('at_h')} steps", float(a.full_paths), RED,
             f"{val('at_b')}$^{{{val('at_h')}}}$ ≈ 10$^{{{val('at_exp')}}}$ paths"),
            (f"Checked every {val('at_w')} steps", float(a.window_paths), GREEN,
             f"{val('at_b')}$^{val('at_w')}$ = {val('at_window')} paths")]
    for i, (lab, v, c, s) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=c, height=0.6)
        if i == 0:
            ax.text(v / 1e3, y, s, va="center", ha="right", fontsize=8.5, color="white",
                    fontweight="bold")
        else:
            ax.text(v * 3, y, s, va="center", ha="left", fontsize=8.5, color=c, fontweight="bold")
        ax.text(0.5, y, lab, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e31)
    ax.set_xticks([1, 1e10, 1e20, 1e30])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Replay paths to search (log scale)")
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "attribution.svg")


def horizon():
    """eq. 1.1 at 99% per step (§20.11 figure)."""
    sc = calcs.HORIZON
    short, long_ = sc.short_horizon, sc.long_horizon
    xs = list(range(0, long_ + 1))
    ys = [100 * calc_unchecked_success(sc.step_success, x) for x in xs]
    fig, ax = plt.subplots(figsize=(2.3, 1.45))
    ax.plot(xs, ys, color=PURPLE, lw=2.5)
    ax.plot([short], [num("hz_p_short")], "o", color=GREEN, ms=8)
    ax.plot([long_], [num("hz_p_long")], "o", color=RED, ms=8)
    ax.text(14, num("hz_p_short") + 2, f"{val('hz_p_short')}% at {val('hz_short')} turns", fontsize=8.5,
            color=GREEN, fontweight="bold")
    ax.text(97, num("hz_p_long") - 16, f"{val('hz_p_long')}% at {val('hz_long')} turns", fontsize=8.5,
            color=RED, fontweight="bold", ha="right")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Horizon (turns)")
    ax.set_ylabel("Unchecked success (%)")
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "horizon.svg")


def metr_p80():
    """METR Time Horizon 1.1: p50 against p80 horizon, frontier models since 2024."""
    rows = [(str(r["release_date"]), r["model_key"], r["p50_min"], r["p80_min"]) for r in calcs.GAP.rows]
    fig, ax = plt.subplots(figsize=(2.2, 1.45))
    for i, (date, name, p50, p80) in enumerate(rows):
        ax.plot([i, i], [p80, p50], color=GRAY_L, lw=3, zorder=1)
        ax.plot([i], [p50], "o", color=PURPLE, ms=5.5, zorder=2)
        ax.plot([i], [p80], "o", color=RED, ms=5.5, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3000)
    ax.set_yticks([1, 10, 60, 600])
    ax.set_yticklabels(["1 min", "10 min", "1 h", "10 h"])
    ax.set_xticks([0, len(rows) - 1])
    ax.set_xticklabels([rows[0][0][:7], rows[-1][0][:7]])
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.text(0.2, 900, "50% success", color=PURPLE, fontsize=8.5, fontweight="bold")
    ax.text(len(rows) - 1, 1.5, "80% success", color=RED, fontsize=8.5, fontweight="bold",
            ha="right")
    ax.set_xlabel("Frontier models by release date")
    ax.set_ylabel("Task length (human time)")
    ax.grid(axis="y", color=sp.C["grid"], lw=0.6)
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "metr_p80.svg")


def cost():
    """NM 1.2: cost per accepted task, three bars (§20.11 figure)."""
    rows = [(f"Baseline, {val('c_acc')}% accepted", "c_eff", GRAY),
            ("Half the token price", "c_cheap", AMBER),
            (f"{val('c_better_acc')}% accepted", "c_better", PURPLE)]
    fig, ax = plt.subplots(figsize=(2.6, 1.0))
    for i, (lab, k, col) in enumerate(rows):
        y = 2 - i
        v = num(k)
        ax.barh(y, v, color=col, height=0.6)
        ax.text(v + 0.08, y, f"${val(k)}", va="center", fontsize=8.5, color=col, fontweight="bold")
        ax.text(-0.08, y, lab, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 6.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Dollars per accepted task")
    axis_type(ax, 8.5)
    sp.save(fig, HERE / "cost.svg")


# ---------------------------------------------------------------- kit copies, thumbnails
KIT_COPIES = ["stack_model", "stack_memory", "stack_tools", "stack_runtime", "stack_learning",
              "stack_scale"]
THUMBS = ["capstone_step3", "lostreply_step3", "records_step2", "swe_agent", "passing_step2",
          "recall_bars"] + KIT_COPIES


def thumbs():
    for k in KIT_COPIES:
        shutil.copy(KIT / "vol3" / f"{k}.svg", HERE / f"{k}.svg")
    root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, str(root / "tools" / "greek_thumbs.py")]
                   + [str(HERE / f"{t}.svg") for t in THUMBS], check=True)


# ------------------------------------------- visual-review additions (2026-10-21)
SEAMS = [("World", "Copies", "config_v7 shown", "after config_v8"),
         ("Proposal", "Effect", "lost reply:", "op_42 repeated"),
         ("Recovery", "Authority", "old plan used", "as permission"),
         ("Record", "Learning", "every accepted", "step a target"),
         ("Worker", "Worker", "one allowance", "copied twice")]


def seams():
    """Five seams; each side meets its own spec, the boundary fails."""
    shown = {1: 2, 2: 3, 3: 5}
    for step, n in shown.items():
        b = []
        for i, (l, r, f1, f2) in enumerate(SEAMS):
            x = 6 + i * 178
            body = (card(x - 2, 10, 84, 64, l, None, BLUE_L, BLUE, size=17)
                    + card(x + 86, 10, 84, 64, r, None, BLUE_L, BLUE, size=17))
            body.append(f'<line x1="{x + 83}" y1="0" x2="{x + 83}" y2="84" stroke="{RED}" '
                        f'stroke-width="4" stroke-dasharray="7 5"/>')
            body.append(text(x + 83, 120, f1, 20, RED, weight="bold"))
            body.append(text(x + 83, 146, f2, 20, RED, weight="bold"))
            b.append(f'<g opacity="{1 if i < n else 0}">' + "".join(body) + "</g>")
        svg(HERE / f"seams_step{step}.svg", 900, 160, b)


def release_curve():
    """Tasks per arm grow as one over the detectable difference squared."""
    import numpy as np
    wide, narrow = calcs.PLAN.differences
    d = np.linspace(narrow * 0.8, wide * 1.4, 200)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(d * 100, (wide / d) ** 2, color=PURPLE, lw=3)
    for x, lab in ((wide, "1×"), (narrow, f"{val('rel_halve')}×")):
        y = (wide / x) ** 2
        ax.plot([x * 100], [y], "o", color=RED, ms=9)
        ax.text(x * 100 + 0.4, y + 0.25, lab, fontsize=15, color=RED, fontweight="bold")
    ax.set_xlabel("Detectable difference (points)")
    ax.set_ylabel("Tasks per arm\n(relative)")
    ax.grid(color=sp.C["grid"], lw=0.6)
    axis_type(ax, 13)
    sp.save(fig, HERE / "release_curve.svg")


OWNERS = [("Task admitted", "runtime, by contract", "contract, read scope"),
          ("Evidence selected", "context manager", "context manifest"),
          ("Change approved", "operator decides", "change request, approval"),
          ("Change dispatched", "action gate, journal", "durable intent, op_42"),
          ("Lost reply settled", "settlement process", "status: config_v8"),
          ("Incident accepted", "accepting principal", "acceptance, health window")]


def owners():
    """Six transitions, each with an owner and a record."""
    for step in (1, 2, 3):
        b = []
        for i, (t, o, r) in enumerate(OWNERS):
            x, y = 6 + (i % 3) * 298, 6 + (i // 3) * 170
            b += card(x, y, 282, 56, t, None, PURPLE_L, PURPLE, size=21)
            if i % 3 < 2:
                b.append(arrow(x + 282, y + 28, x + 296, y + 28, "p", PURPLE))
            b.append(f'<g opacity="{1 if step >= 2 else 0}">'
                     + text(x + 141, y + 92, o, 21, INK, weight="bold") + "</g>")
            b.append(f'<g opacity="{1 if step >= 3 else 0}">'
                     + text(x + 141, y + 124, r, 20, GRAY) + "</g>")
        svg(HERE / f"owners_step{step}.svg", 900, 310, b)


if __name__ == "__main__":
    capstone()
    lostreply()
    barriers()
    records()
    measured()
    recall_bars()
    bill()
    swe_agent()
    passing()
    swebp()
    metr_hack()
    attribution()
    horizon()
    metr_p80()
    cost()
    seams()
    release_curve()
    owners()
    thumbs()
