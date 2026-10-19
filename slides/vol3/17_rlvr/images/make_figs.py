"""Lecture 17 (Reinforcement Learning from Verifiable Rewards): every figure the deck draws.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=17_rlvr`
converts them to PDF). Plots use the shared slideplot helper and are drawn at
the size of their slot; diagrams are SVG at 900 px wide for a full-width slot.
Every derived value comes from calcs.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))

import slideplot as sp  # noqa: E402
from slideplot import plt  # noqa: E402
import calcs  # noqa: E402

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
    short = ["LLM Foundry", "Rust scaffold", "Kernel", "RE-Bench, all", "HCAST, all"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    rows = list(zip(short, calcs.METR_TASKS, calcs.metr()))
    for i, (label, (_, h, n), (_, v)) in enumerate(rows):
        y = len(rows) - 1 - i
        ck = "neutral" if label.startswith("HCAST") else ("danger" if label.startswith("RE-Bench") else "compute")
        ax.barh(y, v, color=sp.C[ck], height=0.62)
        ax.text(-3, y, label, ha="right", va="center", fontsize=16, color=sp.C["text"], fontweight="bold")
        ax.text(v + 3, y, f"{h}/{n:,}", va="center", fontsize=16, color=sp.C[ck], fontweight="bold")
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
    short = ["Original prompt", "Designer's intent", "Do not cheat", "Do not reward hack", "Alzheimer's stakes"]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, (label, (_, v)) in enumerate(zip(short, calcs.METR_PROMPTS)):
            y = len(short) - 1 - i
            shown = step == 2 or i == 0
            ck = "neutral" if i == 0 else "danger"
            ax.barh(y, v, color=sp.C[ck], height=0.62, alpha=1 if shown else 0)
            ax.text(-3, y, label, ha="right", va="center", fontsize=16, color=sp.C["text"], fontweight="bold")
            ax.text(v + 3, y, f"{v}%", va="center", fontsize=16, color=sp.C[ck], fontweight="bold",
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
        _, _, coef = calcs.grpo(rs, eps=1e-8)
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
        y10 = 100 * calcs.false_share(0.10)
        ax.plot([10], [y10], "o", color=RED, ms=10)
        ax.annotate(f"pass rate 10%:\n{y10:.1f}% of rewards\ngo to failures", (10, y10), xytext=(17, 48),
                    fontsize=15, color=RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
        y5 = 100 * calcs.false_share(0.05)
        ax.plot([5], [y5], "o", color=PURPLE, ms=10, alpha=1 if step == 2 else 0)
        ax.text(7.5, y5 + 6, f"5%: {y5:.0f}%" if step == 2 else "5%: ?", fontsize=15,
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

    cases = [(1, 0.2, 0.3), (2, 0.4, 0.7)]
    for step, a, lam in cases:
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
        ok = a + lam < 1
        b.append(text(30, 40, f"α = {a}, λ = {lam}:  α + λ = {a + lam:.1f}", 24, INK, "start",
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
    v0, v1, d, se, sed, thr = calcs.forks()
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
                b.append(text(x + 80, 212, "−0.4", 26, RED, weight="bold"))
        if step >= 3:
            xd = 10 + 2 * 178
            b.append(arrow(xd + 80, 224, xd + 80, 262, "p", PURPLE, dash=True))
            b.append(box(60, 264, 780, 130, PURPLE_L, PURPLE, rx=10))
            b.append(text(450, 300, "fork 20 continuations before and after turn 3", 25, PURPLE, weight="bold"))
            b.append(text(450, 340, f"before: 1 of 20 pass = {v0:.2f}    after: 12 of 20 = {v1:.2f}", 25, INK))
            b.append(text(450, 380, f"debugger turn: +{d:.2f}  (noise bound {thr:.2f})", 26, GREEN,
                          weight="bold"))
        svg(HERE / f"credit_step{step}.svg", W, H, "\n".join(b))


# ---------------------------------------------------------------- GRPO
def lone_pass():
    import numpy as np
    Gs = np.arange(2, 17)
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(Gs, [calcs.lone_pass(g) for g in Gs], color=GREEN, lw=3, marker="o", ms=5)
    for g in (3, 16):
        v = calcs.lone_pass(g)
        ax.plot([g], [v], "o", color=PURPLE, ms=11)
        ax.text(g + (0.6 if g == 3 else -0.4), v + (-0.75 if g == 3 else 0.25),
                f"G = {g}: {v:.2f}", fontsize=16, color=PURPLE, fontweight="bold",
                ha="left" if g == 3 else "right")
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
    ratio = calcs.length_ratio()
    full = 440
    b = []
    for k, (label, toks, push) in enumerate([("failure, 128 tokens", full / ratio, full),
                                             ("failure, 4,096 tokens", full, full / ratio)]):
        y = 34 + k * 190
        b.append(text(10, y, label, 27, INK, "start", "bold"))
        b.append(box(10, y + 14, toks, 36, RED_L, RED, rx=3, sw=2))
        b.append(text(10 + toks + 10, y + 42, "tokens", 22, GRAY, "start") if toks < 100 else "")
        b.append(f'<rect x="10" y="{y + 72}" width="{push:.2f}" height="36" fill="{RED}"/>')
        b.append(text(10 + push + 10 if push < 100 else 20, y + 99,
                      f"push per token: {ratio:.0f}" if k == 0 else "push per token: 1",
                      24, "white" if push >= 100 else RED, "start", "bold"))
    svg(HERE / "length_bias.svg", W, H, "\n".join(b))


def arch():
    import numpy as np
    ps = np.linspace(0, 1, 201)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(ps * 100, [calcs.contrast(p, 8) for p in ps], color=BLUE, lw=3)
        ax.text(50, 1.04, "G = 8", fontsize=15, color=BLUE, fontweight="bold", ha="center")
        ax.plot(ps * 100, [calcs.contrast(p, 16) for p in ps], color=AMBER, lw=2.5, ls="--",
                alpha=1 if step == 2 else 0)
        if step == 2:
            ax.text(3, 1.03, "G = 16", fontsize=15, color=AMBER, fontweight="bold")
        c5, c50 = calcs.contrast(0.05, 8), calcs.contrast(0.5, 8)
        ax.plot([5], [c5], "o", color=RED, ms=11)
        ax.text(7, c5 - 0.1, f"5%: {c5:.2f}", fontsize=15, color=RED, fontweight="bold")
        ax.plot([50], [c50], "o", color=GREEN, ms=11)
        ax.text(52, c50 - 0.15, f"50%: {c50:.2f}", fontsize=15, color=GREEN, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("Task pass rate p (%)")
        ax.set_ylabel("Groups with\nreward contrast")
        ax.grid(color=sp.C["grid"], lw=0.6)
        axis_type(ax, 15)
        sp.save(fig, HERE / f"arch_step{step}.svg")


def informative():
    _, rows = calcs.nm174()
    labels = ["G 8, p 5%", "G 16, p 5%", "G 8, p 50%"]
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for i, (lab, (_, G, p, c, r)) in enumerate(zip(labels, rows)):
        y = 2 - i
        ck = "danger" if r > 20 else "data"
        ax.barh(y, r, color=sp.C[ck], height=0.6)
        ax.text(-0.6, y, lab, ha="right", va="center", fontsize=16, fontweight="bold", color=sp.C["text"])
        ax.text(r + 0.6, y, f"{r:.1f}", va="center", fontsize=16, fontweight="bold", color=sp.C[ck])
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
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(ns, [100 * 0.98 ** n for n in ns], color=GREEN, lw=3, marker="o", ms=7)
    ax.plot(ns, [100 * 0.20 ** n for n in ns], color=RED, lw=3, marker="o", ms=7)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 300)
    ax.text(1.1, 150, "correct patch (0.98 per trial)", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(3.15, 1.5, f"3 trials: {100*0.2**3:.1f}%", fontsize=15, color=RED, fontweight="bold")
    ax.text(1.1, 0.03, "incorrect patch\n(0.20 per trial)", fontsize=14, color=RED, fontweight="bold")
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.set_xticks(ns)
    ax.set_xlabel("Trials that must all pass")
    ax.set_ylabel("Accepted (%)")
    ax.grid(color=sp.C["grid"], lw=0.6)
    axis_type(ax, 15)
    sp.save(fig, HERE / "allpass.svg")


# ---------------------------------------------------------------- slot cycle
def slot():
    w, cc, cs, kc, ks, ratio, rs, frac = calcs.nm176()
    parts = [("model", 20 * 0.12, sp.C["memory"]), ("tools", 20 * 0.08, sp.C["compute"]),
             ("assess", 0.5, sp.C["network"])]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        for y, (name, reset, cap, cyc) in enumerate([("snapshot", 0.015, ks, f"{cs:.3f} s"),
                                                     ("cold", 3.5, kc, f"{cc:.1f} s")]):
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
            ax.text(left + reset + 0.15, y, f"{cyc}: {cap:,.0f}/h", va="center", fontsize=14,
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
    xs = [s for s, _, _ in calcs.YUE]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(xs, [a for _, a, _ in calcs.YUE], color=GREEN, lw=3, marker="o", ms=8)
    ax.plot(xs, [b for _, _, b in calcs.YUE], color=BLUE, lw=3, marker="o", ms=8)
    ax.text(10, 77, "pass@256: one of 256 samples", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(460, 64.3, "64.3", fontsize=14, color=BLUE, fontweight="bold", va="center")
    ax.text(-15, 67.2, "67.2", fontsize=14, color=BLUE, fontweight="bold", va="center", ha="right")
    ax.text(200, 10, "pass@1: one sample", fontsize=14, color=GREEN, fontweight="bold")
    ax.text(-15, 9.9, "9.9", fontsize=14, color=GREEN, fontweight="bold", va="center", ha="right")
    ax.text(460, 42.5, "42.5", fontsize=14, color=GREEN, fontweight="bold", va="center")
    ax.set_xlim(-90, 540)
    ax.set_ylim(0, 90)
    ax.set_xticks(xs)
    ax.set_xticklabels(["base", "150", "300", "450"])
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
    print("figures written")
