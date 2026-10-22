"""Lecture 17 (Reinforcement Learning from Verifiable Rewards): every figure the deck draws.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=17_rlvr`
converts them to PDF). Plots use the shared slideplot helper and are drawn at
the size of their slot; diagrams are SVG at 900 px wide for a full-width slot.
Every derived value comes from calcs.py, which reads MLSysIM.
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

# Volume III roles (volumes/vol3/palette.md)
BLUE, BLUE_L = "#0284c7", "#e0f2fe"
GREEN, GREEN_L = "#16a34a", "#dcfce7"
AMBER, AMBER_L = "#d97706", "#fef3c7"
PURPLE, PURPLE_L = "#7c3aed", "#f3e8ff"
RED, RED_L = "#dc2626", "#fee2e2"
GRAY, GRAY_L = "#64748b", "#f1f5f9"
INK = "#0f172a"
FONT = "Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- SVG helpers
def svg(path, w, h, body):
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n<defs>'
        + "".join(f'<marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
                  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>'
                  for n, c in [("g", GRAY), ("p", PURPLE), ("a", AMBER), ("r", RED), ("b", BLUE),
                               ("n", GREEN)])
        + f'</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, rx=10, dash=False, sw=2.5, op=1.0):
    d = ' stroke-dasharray="8 6"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d} opacity="{op}"/>')


def text(x, y, s, size=22, color=INK, anchor="middle", weight="normal", op=1.0):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" opacity="{op}">{s}</text>')


def arrow(x1, y1, x2, y2, m="g", color=GRAY, dash=False, sw=2.5, op=1.0):
    d = ' stroke-dasharray="7 5"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
            f'marker-end="url(#a{m})" opacity="{op}"/>')


def axis_type(ax, size):
    ax.tick_params(labelsize=size)
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)


# ---------------------------------------------------------------- hook: METR
def metr_tasks():
    """Share of o3 runs that reward hacked, per task (METR task table)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    rows = calcs.METR_TASK_ROWS
    colors = {"task": "compute", "rebench": "danger", "hcast": "neutral"}
    for i, (label, h, n, kind, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ck = colors[kind]
        ax.barh(y, v, color=sp.C[ck], height=0.62)
        ax.text(-3, y, label, ha="right", va="center", fontsize=16, color=sp.C["text"], fontweight="bold")
        ax.text(v + 3, y, f"{h:.0f}/{n:,.0f}", va="center", fontsize=16, color=sp.C[ck], fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Runs that reward-hacked (%)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 15)
    sp.save(fig, HERE / "metr_tasks.svg")


def metr_prompts():
    """Hacking rate with an added instruction (METR prompt table, 20 runs each)."""
    rows = calcs.METR_PROMPTS
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, (label, v) in enumerate(rows):
            y = len(rows) - 1 - i
            shown = step == 2 or i == 0
            ck = "neutral" if i == 0 else "danger"
            ax.barh(y, v, color=sp.C[ck], height=0.62, alpha=1 if shown else 0)
            ax.text(-3, y, label, ha="right", va="center", fontsize=16, color=sp.C["text"], fontweight="bold")
            ax.text(v + 3, y, f"{v:.0f}%", va="center", fontsize=16, color=sp.C[ck], fontweight="bold",
                    alpha=1 if shown else 0)
            if not shown:
                ax.text(5, y, "?", va="center", fontsize=18, color=sp.C["danger"], fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Runs that planned a hack (%)")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        axis_type(ax, 15)
        sp.save(fig, HERE / f"metr_prompts_step{step}.svg")


# ---------------------------------------------------------------- the RLVR loop
def loop():
    """Policy samples a group; sandboxes run it; a verifier scores; update."""
    W, H = 900, 470
    for step in (1, 2, 3):
        b = []
        # policy
        b.append(box(30, 170, 170, 120, BLUE_L, BLUE))
        b.append(text(115, 222, "policy", 26, BLUE, weight="bold"))
        b.append(text(115, 254, "weights θ", 22, INK))
        # rollouts
        b.append(arrow(200, 230, 262, 230, "b", BLUE))
        for i in range(4):
            y = 70 + i * 90
            b.append(box(268, y, 210, 66, AMBER_L, AMBER, rx=8))
            b.append(text(373, y + 42, f"rollout {i+1}", 22, AMBER, weight="bold"))
        b.append(text(373, 52, "sandboxes run each attempt", 23, AMBER))
        if step >= 2:
            # verifier
            b.append(box(540, 150, 160, 160, RED_L, RED))
            b.append(text(620, 214, "verifier", 26, RED, weight="bold"))
            b.append(text(620, 246, "tests,", 23, INK))
            b.append(text(620, 272, "compiler", 23, INK))
            for i in range(4):
                y = 103 + i * 90
                b.append(arrow(478, y, 538, 230 - 30 + i * 20, "r", RED, sw=2))
            for i, r in enumerate([0, 1, 0, 0]):
                b.append(text(735, 152 + i * 50, f"r = {r}", 22, GREEN if r else GRAY, "start", "bold"))
        if step >= 3:
            b.append(box(712, 365, 170, 70, PURPLE_L, PURPLE))
            b.append(text(797, 396, "update", 24, PURPLE, weight="bold"))
            b.append(text(797, 424, "raise the pass", 22, INK))
            b.append(f'<path d="M712,400 C420,470 160,420 115,292" fill="none" stroke="{PURPLE}" '
                     f'stroke-width="2.5" marker-end="url(#ap)"/>')
            b.append(arrow(797, 340, 797, 362, "p", PURPLE))
        svg(HERE / f"loop_step{step}.svg", W, H, "\n".join(b))


# ---------------------------------------------------------------- contrast groups
def groups():
    """Three groups of 8: never solved, sometimes solved, always solved."""
    W, H = 900, 400
    rows = [("never solved", [0] * 8, GRAY), ("solved 2 of 8", [0, 1, 0, 0, 0, 1, 0, 0], GREEN),
            ("always solved", [1] * 8, GRAY)]
    b = [text(150, 40, "task", 22, GRAY, weight="bold"), text(470, 40, "rewards of 8 rollouts", 22, GRAY,
                                                               weight="bold"),
         text(790, 40, "update signal", 22, GRAY, weight="bold")]
    for k, (name, rs, col) in enumerate(rows):
        y = 80 + k * 110
        coef = calcs.coefficients(rs)
        b.append(text(150, y + 42, name, 24, INK, weight="bold"))
        for i, r in enumerate(rs):
            x = 300 + i * 44
            b.append(box(x, y + 12, 36, 40, GREEN_L if r else GRAY_L, GREEN if r else "#94a3b8", rx=5, sw=2))
            b.append(text(x + 18, y + 40, str(r), 22, GREEN if r else GRAY, weight="bold"))
            # coefficient bar under the cell
            hpx = coef[i] * 14
            if abs(hpx) > 0.5:
                yb = y + 76
                b.append(f'<rect x="{x+8}" y="{yb - max(hpx, 0)}" width="20" height="{abs(hpx)}" '
                         f'fill="{GREEN if hpx > 0 else RED}"/>')
        b.append(f'<line x1="300" y1="{y+76}" x2="648" y2="{y+76}" stroke="#94a3b8" stroke-width="1.5"/>')
        sig = "none: all 0" if col == GRAY else "2 up, 6 down"
        b.append(text(790, y + 42, sig, 22, RED if col == GRAY else GREEN, weight="bold"))
    svg(HERE / "groups.svg", W, H, "\n".join(b))


# ---------------------------------------------------------------- false accepts
def false_share():
    import numpy as np
    ps = np.linspace(0.01, 0.5, 200)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(100 * ps, [100 * calcs.false_share(p) for p in ps], color=RED, lw=3)
        x10, y10 = num("fa_pass"), calcs.pct(calcs.FA_S.result.false_share)
        ax.plot([x10], [y10], "o", color=RED, ms=10)
        ax.annotate(f"pass rate {val('fa_pass')}%:\n{val('fa_share')}% of rewards\ngo to failures", (x10, y10),
                    xytext=(17, 48),
                    fontsize=15, color=RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
        x5, y5 = num("fa5_pass"), calcs.pct(calcs.FA5_S.result.false_share)
        ax.plot([x5], [y5], "o", color=PURPLE, ms=10, alpha=1 if step == 2 else 0)
        ax.text(7.5, y5 + 6, f"{val('fa5_pass')}%: {y5:.0f}%" if step == 2 else f"{val('fa5_pass')}%: ?", fontsize=15,
                color=PURPLE, fontweight="bold")
        ax.set_xlim(0, 52)
        ax.set_ylim(0, 75)
        ax.set_xlabel("Policy's true pass rate (%)")
        ax.set_ylabel("Rewarded rollouts\nthat fail (%)")
        ax.grid(color=sp.C["grid"], lw=0.6)
        axis_type(ax, 15)
        sp.save(fig, HERE / f"false_share_step{step}.svg")


# ---------------------------------------------------------------- shaping
def shaping():
    """Number line: failure scores [-lambda, alpha], pass scores [1-lambda, 1+alpha]."""
    W, H = 900, 330
    x0, x1 = 90, 830   # maps score -1 .. 1.5

    def X(v):
        return x0 + (v + 1.0) / 2.5 * (x1 - x0)

    cases = [(1, calcs.SHAPE_OK_S), (2, calcs.SHAPE_S)]
    for step, sc in cases:
        a, lam, res = sc.format_weight, sc.length_weight, sc.result
        b = [f'<line x1="{x0}" y1="250" x2="{x1}" y2="250" stroke="{GRAY}" stroke-width="2"/>']
        for v in (-1, -0.5, 0, 0.5, 1, 1.5):
            b.append(f'<line x1="{X(v)}" y1="244" x2="{X(v)}" y2="256" stroke="{GRAY}" stroke-width="2"/>')
            b.append(text(X(v), 286, f"{v:g}", 20, GRAY))
        b.append(text(X(-1) - 10, 320, "score", 20, GRAY, "start"))
        # failure band
        b.append(box(X(-lam), 150, X(a) - X(-lam), 50, RED_L, RED, rx=6))
        b.append(text((X(-lam) + X(a)) / 2, 182, "any failure", 22, RED, weight="bold"))
        # pass band
        b.append(box(X(1 - lam), 70, X(1 + a) - X(1 - lam), 50, GREEN_L, GREEN, rx=6))
        b.append(text((X(1 - lam) + X(1 + a)) / 2, 102, "any pass", 22, GREEN, weight="bold"))
        ok = res.order_holds
        b.append(text(30, 40, f"α = {a}, λ = {lam}:  α + λ = {res.weight_sum:.1f}", 24, INK, "start",
                      "bold"))
        b.append(text(870, 40, "every pass outranks every failure" if ok else "a short failure can beat a long pass",
                      22, GREEN if ok else RED, "end", "bold"))
        if not ok:
            b.append(box(X(1 - lam), 60, X(a) - X(1 - lam), 150, "none", RED, rx=4, dash=True, sw=3))
        svg(HERE / f"shaping_step{step}.svg", W, H, "\n".join(b))


# ---------------------------------------------------------------- credit strip
def credit():
    """Five turns of the repair; one terminal failure; equal blame; forks."""
    W, H = 900, 400
    turns = ["reproduce", "read pool.py", "debugger", "bad edit", "tests fail"]
    fail = val("credit_fail").replace("-", "−")
    for step in (1, 2, 3):
        b = []
        for i, t in enumerate(turns):
            x = 10 + i * 178
            fill, stroke = (RED_L, RED) if i >= 3 else (BLUE_L, BLUE)
            b.append(box(x, 44, 160, 70, fill, stroke))
            b.append(text(x + 80, 88, t, 22, INK, weight="bold"))
            if i < 4:
                b.append(arrow(x + 160, 79, x + 176, 79, "g"))
            b.append(text(x + 80, 32, f"turn {i+1}" if i < 4 else "reward 0", 23, GRAY if i < 4 else RED,
                          weight="bold"))
            if step >= 2 and i < 4:
                b.append(f'<rect x="{x + 60}" y="132" width="40" height="48" fill="{RED}"/>')
                b.append(text(x + 80, 212, fail, 26, RED, weight="bold"))
        if step >= 3:
            xd = 10 + 2 * 178
            b.append(arrow(xd + 80, 224, xd + 80, 262, "p", PURPLE, dash=True))
            b.append(box(60, 264, 780, 130, PURPLE_L, PURPLE, rx=10))
            k = val("fork_k")
            b.append(text(450, 300, f"fork {k} continuations before and after turn 3", 25, PURPLE, weight="bold"))
            b.append(text(450, 340, f"before: {val('fork_before')} of {k} pass = {val('fork_v0')}    "
                                    f"after: {val('fork_after')} of {k} = {val('fork_v1')}", 25, INK))
            b.append(text(450, 380, f"debugger turn: +{val('fork_diff')}  (noise bound {val('fork_bound')})", 26, GREEN,
                          weight="bold"))
        svg(HERE / f"credit_step{step}.svg", W, H, "\n".join(b))


# ---------------------------------------------------------------- GRPO
def lone_pass():
    import numpy as np
    g_small, g_big = calcs.GRPO_S.result.group, int(num("lone_g"))
    Gs = np.arange(2, g_big + 1)
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(Gs, [calcs.lone_pass(int(g)) for g in Gs], color=GREEN, lw=3, marker="o", ms=5)
    for g in (g_small, g_big):
        v = calcs.lone_pass(g)
        ax.plot([g], [v], "o", color=PURPLE, ms=11)
        ax.text(g + (0.6 if g == g_small else -0.4), v + (-0.75 if g == g_small else 0.25),
                f"G = {g}: {v:.2f}", fontsize=16, color=PURPLE, fontweight="bold",
                ha="left" if g == g_small else "right")
    ax.set_xlim(1, 17)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("Group size G")
    ax.set_ylabel("Lone-pass\ncoefficient")
    ax.grid(color=sp.C["grid"], lw=0.6)
    axis_type(ax, 16)
    sp.save(fig, HERE / "lone_pass.svg")


def length_bias():
    """Token count and per-token push for two failures with one coefficient (column)."""
    W, H = 500, 400
    ratio = num("len_ratio")
    full = 440
    b = []
    for k, (label, toks, push) in enumerate([(f"failure, {val('len_short')} tokens", full / ratio, full),
                                             (f"failure, {val('len_long')} tokens", full, full / ratio)]):
        y = 34 + k * 190
        b.append(text(10, y, label, 27, INK, "start", "bold"))
        b.append(box(10, y + 14, toks, 36, RED_L, RED, rx=3, sw=2))
        b.append(text(10 + toks + 10, y + 42, "tokens", 22, GRAY, "start") if toks < 100 else "")
        b.append(f'<rect x="10" y="{y + 72}" width="{push:.2f}" height="36" fill="{RED}"/>')
        b.append(text(10 + push + 10 if push < 100 else 20, y + 99,
                      f"push per token: {val('len_ratio')}" if k == 0 else "push per token: 1",
                      24, "white" if push >= 100 else RED, "start", "bold"))
    svg(HERE / "length_bias.svg", W, H, "\n".join(b))


def arch():
    import numpy as np
    ps = np.linspace(0, 1, 201)
    g, g_big = int(num("arch_g")), int(num("arch_big_g"))
    x_hard, x_tgt = num("arch_hard_pct"), num("arch_tgt_pct")
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(ps * 100, [calcs.contrast(p, g) for p in ps], color=BLUE, lw=3)
        ax.text(50, 1.04, f"G = {g}", fontsize=15, color=BLUE, fontweight="bold", ha="center")
        ax.plot(ps * 100, [calcs.contrast(p, g_big) for p in ps], color=AMBER, lw=2.5, ls="--",
                alpha=1 if step == 2 else 0)
        if step == 2:
            ax.text(3, 1.03, f"G = {g_big}", fontsize=15, color=AMBER, fontweight="bold")
        c5, c50 = calcs.HARD_S.result.contrast, calcs.TGT_S.result.contrast
        ax.plot([x_hard], [c5], "o", color=RED, ms=11)
        ax.text(x_hard + 2, c5 - 0.1, f"{val('arch_hard_pct')}%: {val('arch_hard_c')}", fontsize=15, color=RED,
                fontweight="bold")
        ax.plot([x_tgt], [c50], "o", color=GREEN, ms=11)
        ax.text(x_tgt + 2, c50 - 0.15, f"{val('arch_tgt_pct')}%: {val('arch_tgt_c')}", fontsize=15, color=GREEN,
                fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Task pass rate p (%)")
        ax.set_ylabel("Groups with\nreward contrast")
        ax.grid(color=sp.C["grid"], lw=0.6)
        axis_type(ax, 15)
        sp.save(fig, HERE / f"arch_step{step}.svg")


def informative():
    rows = [(f"G {val('arch_g')}, p {val('arch_hard_pct')}%", calcs.HARD_S, "budget_hard_r", "danger"),
            (f"G {val('arch_big_g')}, p {val('arch_hard_pct')}%", calcs.BIG_S, "budget_big_r", "danger"),
            (f"G {val('arch_g')}, p {val('arch_tgt_pct')}%", calcs.TGT_S, "budget_tgt_r", "data")]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for i, (lab, sc, key, ck) in enumerate(rows):
        y = len(rows) - 1 - i
        r = sc.result.rollouts_per_informative_group
        ax.barh(y, r, color=sp.C[ck], height=0.6)
        ax.text(-0.6, y, lab, ha="right", va="center", fontsize=16, fontweight="bold", color=sp.C["text"])
        ax.text(r + 0.6, y, val(key), va="center", fontsize=16, fontweight="bold", color=sp.C[ck])
    ax.set_yticks([])
    ax.set_xlim(0, 36)
    ax.set_xlabel("Rollouts per informative group")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 15)
    sp.save(fig, HERE / "informative.svg")


# ---------------------------------------------------------------- records
def records():
    """Turn 2 was conditioned on the full log; the final transcript holds a summary."""
    W, H = 900, 470
    for step in (1, 2):
        b = [text(30, 36, "what the policy saw at turn 2", 22, BLUE, "start", "bold")]
        b.append(box(30, 50, 380, 60, GREEN_L, GREEN, rx=6))
        b.append(text(220, 88, "full failure log", 23, INK))
        b.append(arrow(410, 80, 470, 80, "b", BLUE))
        b.append(box(476, 50, 240, 60, BLUE_L, BLUE, rx=6))
        b.append(text(596, 88, "turn 2: the patch", 23, BLUE, weight="bold"))
        b.append(text(30, 166, "the transcript after compaction at turn 4", 22, GRAY, "start", "bold"))
        b.append(box(30, 180, 160, 60, GREEN_L, GREEN, rx=6, dash=True))
        b.append(text(110, 218, "summary", 23, INK))
        b.append(arrow(190, 210, 250, 210, "r", RED))
        b.append(box(256, 180, 240, 60, BLUE_L, BLUE, rx=6))
        b.append(text(376, 218, "turn 2: the patch", 23, BLUE, weight="bold"))
        b.append(text(520, 218, "scored against a context", 23, RED, "start", "bold"))
        b.append(text(520, 244, "the policy never saw", 23, RED, "start", "bold"))
        if step >= 2:
            b.append(box(30, 290, 840, 160, PURPLE_L, PURPLE, rx=10))
            b.append(text(50, 326, "Record per invocation", 23, PURPLE, "start", "bold"))
            items = ["delivered prefix token IDs", "sampled token IDs", "behavior log probability per token",
                     "checkpoint, sampling settings", "verdict and its versions"]
            for i, it in enumerate(items):
                col, row = i % 2, i // 2
                b.append(text(70 + col * 420, 366 + row * 32, "• " + it, 23, INK, "start"))
        svg(HERE / f"records_step{step}.svg", W, H, "\n".join(b))


def states():
    """Four ways a rollout stops, and what each does to the reward (column)."""
    W, H = 520, 440
    rows = [("budget exhausted", "fail: reward 0", RED),
            ("tool error, recovered", "an observation", BLUE),
            ("verifier crash, retry", "pending: wait", AMBER),
            ("preempted, no retry", "unknown: no reward", AMBER)]
    b = []
    for i, (s, t, c) in enumerate(rows):
        y = 10 + i * 108
        b.append(text(10, y + 30, s, 26, INK, "start", "bold"))
        b.append(arrow(30, y + 42, 30, y + 64, "g"))
        b.append(box(50, y + 46, 440, 46, {RED: RED_L, BLUE: BLUE_L, AMBER: AMBER_L}[c], c, rx=6))
        b.append(text(66, y + 79, t, 26, c, "start", "bold"))
    svg(HERE / "states.svg", W, H, "\n".join(b))


def isolation():
    """Exploration side and assessment side, with the handoff between them."""
    W, H = 900, 470
    for step in (1, 2):
        b = []
        b.append(box(20, 30, 360, 420, AMBER_L, AMBER, rx=12))
        b.append(text(200, 66, "exploration side", 24, AMBER, weight="bold"))
        for i, (f, bad) in enumerate([("src/pool.py", False), ("tests/test_pool.py", False),
                                      ("conftest.py (added)", True)]):
            y = 100 + i * 70
            b.append(box(50, y, 300, 50, "white", RED if bad else "#94a3b8", rx=6, sw=2.5 if bad else 2))
            b.append(text(200, y + 33, f, 23, RED if bad else INK, weight="bold" if bad else "normal"))
        b.append(text(200, 340, "candidate's runner:", 23, INK))
        b.append(text(200, 372, "1 passed", 26, RED, weight="bold"))
        if step >= 2:
            b.append(box(520, 30, 360, 420, PURPLE_L, PURPLE, rx=12))
            b.append(text(700, 66, "assessment side", 24, PURPLE, weight="bold"))
            b.append(box(550, 100, 300, 50, "white", "#94a3b8", rx=6, sw=2))
            b.append(text(700, 133, "fresh fixture + src/ diff", 23, INK))
            b.append(box(550, 170, 300, 50, "white", "#94a3b8", rx=6, sw=2))
            b.append(text(700, 203, "controller's own tests", 23, INK))
            b.append(text(700, 340, "controller's runner:", 23, INK))
            b.append(text(700, 372, "1 failed", 26, GREEN, weight="bold"))
            b.append(arrow(380, 125, 516, 125, "p", PURPLE, sw=3))
            b.append(text(450, 112, "admit src/", 22, PURPLE, weight="bold"))
            b.append(arrow(380, 265, 450, 265, "r", RED, sw=3))
            b.append(text(450, 300, "refused", 23, RED, weight="bold"))
            b.append(f'<line x1="438" y1="250" x2="462" y2="280" stroke="{RED}" stroke-width="4"/>')
            b.append(f'<line x1="462" y1="250" x2="438" y2="280" stroke="{RED}" stroke-width="4"/>')
        svg(HERE / f"isolation_step{step}.svg", W, H, "\n".join(b))


def allpass():
    ns = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    curves = [calcs.all_pass(n) for n in ns]
    ax.plot(ns, [100 * good for good, _ in curves], color=GREEN, lw=3, marker="o", ms=7)
    ax.plot(ns, [100 * bad for _, bad in curves], color=RED, lw=3, marker="o", ms=7)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 300)
    ax.text(1.1, 150, f"correct ({val('ap_good')} per trial)", fontsize=15, color=GREEN, fontweight="bold")
    ax.text(3.15, 1.5, f"{val('ap_trials')} trials: {val('ap_bad_acc')}%", fontsize=16, color=RED, fontweight="bold")
    ax.text(1.1, 0.025, f"wrong\n({val('ap_bad')} per trial)", fontsize=15, color=RED, fontweight="bold")
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xticks(ns)
    ax.set_xlabel("Trials that must all pass")
    ax.set_ylabel("Accepted (%)")
    ax.grid(color=sp.C["grid"], lw=0.6)
    axis_type(ax, 16)
    sp.save(fig, HERE / "allpass.svg")


# ---------------------------------------------------------------- slot cycle
def slot():
    d = calcs.SLOT_S
    parts = [("model", calcs.sec(d.decisions * d.model_time), sp.C["memory"]),
             ("tools", calcs.sec(d.decisions * d.tool_time), sp.C["compute"]),
             ("assess", calcs.sec(d.assessment_time), sp.C["network"])]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        for y, (name, reset, cap, cyc) in enumerate([
                ("snapshot", calcs.sec(d.snapshot_reset), val("slot_snap_rate"), f"{val('slot_snap_cycle')} s"),
                ("cold", calcs.sec(d.cold_reset), val("slot_cold_rate"), f"{val('slot_cold_cycle1')} s")]):
            left = 0
            vis = step == 2 or name == "cold"
            for lab, v, c in parts:
                ax.barh(y, v, left=left, color=c, height=0.55, alpha=1 if vis else 0)
                if y == 1:
                    ax.text(left + v / 2, 1.42 if lab != "assess" else 0.5, lab, ha="center",
                            va="bottom" if lab != "assess" else "center", fontsize=14, color=c,
                            fontweight="bold")
                left += v
            ax.barh(y, reset, left=left, color=sp.C["danger"], height=0.55, alpha=1 if vis else 0)
            if y == 1:
                ax.text(left + reset / 2, 1.42, "reset", ha="center", va="bottom", fontsize=14,
                        color=sp.C["danger"], fontweight="bold")
            ax.text(left + reset + 0.15, y, f"{cyc}: {cap}/h", va="center", fontsize=14,
                    fontweight="bold", color=sp.C["text"], alpha=1 if vis else 0)
            ax.text(-0.15, y, name, ha="right", va="center", fontsize=14, fontweight="bold",
                    color=sp.C["text"])
        ax.set_yticks([])
        ax.set_ylim(-0.5, 1.9)
        ax.set_xlim(0, 13)
        ax.set_xlabel("Seconds per episode in one slot")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=sp.C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        axis_type(ax, 14)
        sp.save(fig, HERE / f"slot_step{step}.svg")


# ---------------------------------------------------------------- Yue et al.
def yue():
    xs = [r["step"] for r in calcs.YUE]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(xs, [r["pass_at_1"] for r in calcs.YUE], color=GREEN, lw=3, marker="o", ms=8)
    ax.plot(xs, [r["pass_at_256"] for r in calcs.YUE], color=BLUE, lw=3, marker="o", ms=8)
    x_end = xs[-1] + 10
    ax.text(10, 77, "pass@256: one of 256 samples", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(x_end, num("yue_pk_end"), val("yue_pk_end"), fontsize=14, color=BLUE, fontweight="bold", va="center")
    ax.text(-15, num("yue_pk_base"), val("yue_pk_base"), fontsize=14, color=BLUE, fontweight="bold", va="center",
            ha="right")
    ax.text(200, 10, "pass@1: one sample", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(-15, num("yue_p1_base"), val("yue_p1_base"), fontsize=14, color=GREEN, fontweight="bold", va="center",
            ha="right")
    ax.text(x_end, num("yue_p1_end"), val("yue_p1_end"), fontsize=14, color=GREEN, fontweight="bold", va="center")
    ax.set_xlim(-90, 540)
    ax.set_ylim(0, 90)
    ax.set_xticks(xs)
    ax.set_xticklabels(["base"] + [f"{x:.0f}" for x in xs[1:]])
    ax.set_xlabel("GRPO training steps")
    ax.set_ylabel("Problems solved (%)")
    ax.grid(color=sp.C["grid"], lw=0.6)
    axis_type(ax, 15)
    sp.save(fig, HERE / "yue.svg")


# ---------------------------------------------------------------- thumbnails drawn here
def thumbs():
    """Text-free pictures for the recall and next-lecture slides."""
    # recall: a stored trajectory, mostly observations (green), a sliver of targets (dark)
    b = [box(10, 40, 600, 60, GREEN_L, GREEN, rx=4),
         f'<rect x="473" y="40" width="107" height="60" fill="{GRAY_L}" stroke="#94a3b8" stroke-width="2"/>',
         f'<rect x="580" y="40" width="30" height="60" fill="{INK}"/>']
    svg(HERE / "thumb_masking.svg", 620, 140, "\n".join(b))
    # next lecture: one controller delegating to four workers
    b = [f'<circle cx="300" cy="50" r="34" fill="{BLUE_L}" stroke="{BLUE}" stroke-width="4"/>']
    for i in range(4):
        x = 90 + i * 140
        b.append(f'<line x1="300" y1="84" x2="{x}" y2="166" stroke="{GRAY}" stroke-width="4"/>')
        b.append(f'<circle cx="{x}" cy="190" r="26" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="4"/>')
    svg(HERE / "thumb_multiagent.svg", 600, 230, "\n".join(b))


# ------------------------------------------- visual-review additions (2026-10-21)
def forks():
    """Continuation forks at every boundary against two chosen turns (NM 17.2)."""
    W, H = 440, 300
    n = int(num("forks_turns"))
    b = []
    for row, (label, every, mult, col) in enumerate(
            (("every boundary", True, "forks_all_mult", RED),
             (f"turns {val('forks_two_first')} and {val('forks_two_second')}", False,
              "forks_two_mult", GREEN))):
        y = 70 + row * 150
        b.append(text(8, y - 34, label, 29, col, "start", "bold"))
        b.append(f'<line x1="8" y1="{y}" x2="300" y2="{y}" stroke="{GRAY}" stroke-width="5"/>')
        picks = range(1, n) if every else (int(num("forks_two_first")), int(num("forks_two_second")))
        for t in picks:
            x = 8 + 292 * t / n
            for d in (-1, 1):
                b.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x + 12:.1f}" y2="{y + d * 22}" '
                         f'stroke="{col}" stroke-width="2.4"/>')
        bar = 300 * num(mult) / num("forks_all_mult")
        b.append(box(8, y + 34, max(bar, 8), 34, col, col, rx=3))
        b.append(text(max(bar, 8) + 20, y + 62, f"{val(mult)}×", 32, col, "start", "bold"))
    svg(HERE / "forks.svg", W, H, "\n".join(b))


def loop_limits():
    """Three limits on the rollout loop: barrier, staleness, verifier pool."""
    W, H = 900, 330
    b = []
    heads = ["Group barrier", "Staleness", "Verifier pool"]
    for i, h in enumerate(heads):
        x = 8 + i * 300
        b.append(box(x, 6, 284, 318, "white", PURPLE, rx=10))
        b.append(text(x + 142, 40, h, 25, PURPLE, weight="bold"))
    lat = [q.to("second").magnitude for q in calcs.BAR_S.latencies]
    top = max(lat)
    for k, v in enumerate(sorted(lat)):
        y = 70 + k * 44
        b.append(box(26, y, 248, 30, GRAY_L, GRAY, rx=3, dash=True))
        b.append(box(26, y, max(248 * v / top, 4), 30, AMBER if v == top else BLUE, AMBER if v == top else BLUE, rx=3))
    b.append(text(150, 272, "the step waits for the slowest", 20))
    b.append(text(150, 304, f"{val('bar_idle')}% of slot time idle", 22, RED, weight="bold"))
    for k in range(4):
        x = 330 + k * 64
        b.append(box(x, 70, 54, 40, PURPLE_L, PURPLE, rx=4))
        b.append(text(x + 27, 97, f"v{k + 1}", 21, PURPLE, weight="bold"))
    b.append(box(330, 134, 246, 34, AMBER_L, AMBER, rx=4))
    b.append(text(453, 158, "one long rollout", 20))
    b.append(box(330, 180, 54, 34, BLUE_L, BLUE, rx=4))
    b.append(text(357, 204, "short", 18))
    b.append(text(458, 262, "long rollouts go stale", 21))
    b.append(text(458, 296, "lag = versions passed", 22, RED, weight="bold"))
    b.append(text(758, 86, f"{val('pool_rate')} trials/s arrive", 21))
    slots = 10
    busy = round(slots * num("pool_target") / 100)
    for k in range(slots):
        x = 630 + (k % 5) * 52
        y = 110 + (k // 5) * 52
        b.append(box(x, y, 42, 42, BLUE if k < busy else GRAY_L, BLUE if k < busy else GRAY, rx=4))
    b.append(text(758, 252, f"{val('pool_busy')} busy of {val('pool_slots')} slots", 21))
    b.append(text(758, 296, f"{val('pool_target')}% target", 22, RED, weight="bold"))
    svg(HERE / "loop_limits.svg", W, H, "\n".join(b))


if __name__ == "__main__":
    metr_tasks()
    metr_prompts()
    loop()
    groups()
    false_share()
    shaping()
    credit()
    lone_pass()
    length_bias()
    arch()
    informative()
    records()
    states()
    isolation()
    allpass()
    slot()
    yue()
    thumbs()
    forks()
    loop_limits()
    print("figures written")
