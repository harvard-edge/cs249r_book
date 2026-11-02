#!/usr/bin/env python3
"""Lecture 17 (Reinforcement Learning from Verifiable Rewards): every figure the deck draws.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=17_rlvr`
converts them to PDF). Plots use the shared slideplot helper and are drawn at
the size of their slot; diagrams are SVG at 900 px wide for a full-width slot.
Every derived value comes from calcs.py, which reads MLSysIM.
"""
from types import SimpleNamespace
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from slideplot import plt
BLUE, BLUE_L = ('#0284c7', '#e0f2fe')
GREEN, GREEN_L = ('#16a34a', '#dcfce7')
AMBER, AMBER_L = ('#d97706', '#fef3c7')
PURPLE, PURPLE_L = ('#7c3aed', '#f3e8ff')
RED, RED_L = ('#dc2626', '#fee2e2')
GRAY, GRAY_L = ('#64748b', '#f1f5f9')
INK = '#0f172a'
FONT = 'Helvetica, Arial, sans-serif'

def svg(path, w, h, body):
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">\n<defs>' + ''.join((f'<marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>' for n, c in [('g', GRAY), ('p', PURPLE), ('a', AMBER), ('r', RED), ('b', BLUE), ('n', GREEN)])) + f'</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + '\n</svg>\n')

def box(x, y, w, h, fill, stroke, rx=10, dash=False, sw=2.5, op=1.0):
    d = ' stroke-dasharray="8 6"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{op}"/>'

def text(x, y, s, size=22, color=INK, anchor='middle', weight='normal', op=1.0):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" opacity="{op}">{s}</text>'

def arrow(x1, y1, x2, y2, m='g', color=GRAY, dash=False, sw=2.5, op=1.0):
    d = ' stroke-dasharray="7 5"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#a{m})" opacity="{op}"/>'

def axis_type(ax, size):
    ax.tick_params(labelsize=size)
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)

def metr_tasks():
    """Share of o3 runs that reward hacked, per task (METR task table)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    rows = [('LLM Foundry', 21.0, 21.0, 'task', 100.0), ('Rust scaffold', 12.0, 28.0, 'task', 42.85714286), ('Kernel', 6.0, 24.0, 'task', 25.0), ('RE-Bench, all', 39.0, 128.0, 'rebench', 30.46875), ('HCAST, all', 8.0, 1087.0, 'hcast', 0.7359705612)]
    colors = {'task': 'compute', 'rebench': 'danger', 'hcast': 'neutral'}
    for i, (label, h, n, kind, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ck = colors[kind]
        ax.barh(y, v, color=sp.C[ck], height=0.62)
        ax.text(-3, y, label, ha='right', va='center', fontsize=16, color=sp.C['text'], fontweight='bold')
        ax.text(v + 3, y, f'{h:.0f}/{n:,.0f}', va='center', fontsize=16, color=sp.C[ck], fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel('Runs that reward-hacked (%)')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=sp.C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 15)
    sp.save(fig, HERE / 'metr_tasks.svg')

def metr_prompts():
    """Hacking rate with an added instruction (METR prompt table, 20 runs each)."""
    rows = [('Original prompt', 80.0), ("Designer's intent", 95.0), ('Do not cheat', 80.0), ('Do not reward hack', 70.0), ("Alzheimer's stakes", 70.0)]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, (label, v) in enumerate(rows):
            y = len(rows) - 1 - i
            shown = step == 2 or i == 0
            ck = 'neutral' if i == 0 else 'danger'
            ax.barh(y, v, color=sp.C[ck], height=0.62, alpha=1 if shown else 0)
            ax.text(-3, y, label, ha='right', va='center', fontsize=16, color=sp.C['text'], fontweight='bold')
            ax.text(v + 3, y, f'{v:.0f}%', va='center', fontsize=16, color=sp.C[ck], fontweight='bold', alpha=1 if shown else 0)
            if not shown:
                ax.text(5, y, '?', va='center', fontsize=18, color=sp.C['danger'], fontweight='bold')
        ax.set_yticks([])
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel('Runs that planned a hack (%)')
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=sp.C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        axis_type(ax, 15)
        sp.save(fig, HERE / f'metr_prompts_step{step}.svg')

def loop():
    """Policy samples a group; sandboxes run it; a verifier scores; update."""
    W, H = (900, 470)
    for step in (1, 2, 3):
        b = []
        b.append(box(30, 170, 170, 120, BLUE_L, BLUE))
        b.append(text(115, 222, 'policy', 26, BLUE, weight='bold'))
        b.append(text(115, 254, 'weights θ', 22, INK))
        b.append(arrow(200, 230, 262, 230, 'b', BLUE))
        for i in range(4):
            y = 70 + i * 90
            b.append(box(268, y, 210, 66, AMBER_L, AMBER, rx=8))
            b.append(text(373, y + 42, f'rollout {i + 1}', 22, AMBER, weight='bold'))
        b.append(text(373, 52, 'sandboxes run each attempt', 23, AMBER))
        if step >= 2:
            b.append(box(540, 150, 160, 160, RED_L, RED))
            b.append(text(620, 214, 'verifier', 26, RED, weight='bold'))
            b.append(text(620, 246, 'tests,', 23, INK))
            b.append(text(620, 272, 'compiler', 23, INK))
            for i in range(4):
                y = 103 + i * 90
                b.append(arrow(478, y, 538, 230 - 30 + i * 20, 'r', RED, sw=2))
            for i, r in enumerate([0, 1, 0, 0]):
                b.append(text(735, 152 + i * 50, f'r = {r}', 22, GREEN if r else GRAY, 'start', 'bold'))
        if step >= 3:
            b.append(box(712, 365, 170, 70, PURPLE_L, PURPLE))
            b.append(text(797, 396, 'update', 24, PURPLE, weight='bold'))
            b.append(text(797, 424, 'raise the pass', 22, INK))
            b.append(f'<path d="M712,400 C420,470 160,420 115,292" fill="none" stroke="{PURPLE}" stroke-width="2.5" marker-end="url(#ap)"/>')
            b.append(arrow(797, 340, 797, 362, 'p', PURPLE))
        svg(HERE / f'loop_step{step}.svg', W, H, '\n'.join(b))

def groups():
    """Three groups of 8: never solved, sometimes solved, always solved."""
    W, H = (900, 400)
    rows = [('never solved', [0] * 8, GRAY), ('solved 2 of 8', [0, 1, 0, 0, 0, 1, 0, 0], GREEN), ('always solved', [1] * 8, GRAY)]
    b = [text(150, 40, 'task', 22, GRAY, weight='bold'), text(470, 40, 'rewards of 8 rollouts', 22, GRAY, weight='bold'), text(790, 40, 'update signal', 22, GRAY, weight='bold')]
    for k, (name, rs, col) in enumerate(rows):
        y = 80 + k * 110
        coef = {0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1: [-0.5773502559, 1.732050768, -0.5773502559, -0.5773502559, -0.5773502559, 1.732050768, -0.5773502559, -0.5773502559], 2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}[k]
        b.append(text(150, y + 42, name, 24, INK, weight='bold'))
        for i, r in enumerate(rs):
            x = 300 + i * 44
            b.append(box(x, y + 12, 36, 40, GREEN_L if r else GRAY_L, GREEN if r else '#94a3b8', rx=5, sw=2))
            b.append(text(x + 18, y + 40, str(r), 22, GREEN if r else GRAY, weight='bold'))
            hpx = coef[i] * 14
            if abs(hpx) > 0.5:
                yb = y + 76
                b.append(f'<rect x="{x + 8}" y="{yb - max(hpx, 0)}" width="20" height="{abs(hpx)}" fill="{(GREEN if hpx > 0 else RED)}"/>')
        b.append(f'<line x1="300" y1="{y + 76}" x2="648" y2="{y + 76}" stroke="#94a3b8" stroke-width="1.5"/>')
        sig = 'none: all 0' if col == GRAY else '2 up, 6 down'
        b.append(text(790, y + 42, sig, 22, RED if col == GRAY else GREEN, weight='bold'))
    svg(HERE / 'groups.svg', W, H, '\n'.join(b))

def false_share():
    import numpy as np
    ps = np.linspace(0.01, 0.5, 200)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(100 * ps, [71.22302158, 66.4547545, 62.26534955, 58.55542912, 55.24713213, 52.27858985, 49.60002045, 47.17091476, 44.95797341, 42.93357111, 41.07459751, 39.36157101, 37.77795347, 36.30961451, 34.94440871, 33.67183894, 32.48278586, 31.36928909, 30.32436875, 29.34187901, 28.41638712, 27.54307293, 26.71764486, 25.93626944, 25.19551173, 24.49228485, 23.823807, 23.18756466, 22.58128103, 22.00288878, 21.45050652, 20.92241836, 20.41705606, 19.93298355, 19.46888325, 19.02354409, 18.595851, 18.18477554, 17.78936766, 17.40874841, 17.04210334, 16.68867675, 16.34776646, 16.01871916, 15.70092622, 15.39381995, 15.09687021, 14.80958137, 14.5314895, 14.26215993, 14.00118496, 13.74818177, 13.50279061, 13.26467304, 13.03351037, 12.8090023, 12.59086555, 12.37883269, 12.17265107, 11.97208176, 11.7768987, 11.58688779, 11.40184614, 11.22158131, 11.04591071, 10.87466091, 10.7076671, 10.54477257, 10.38582822, 10.23069208, 10.0792289, 9.931309767, 9.786811737, 9.645617477, 9.507614964, 9.37269718, 9.240761841, 9.111711135, 8.985451479, 8.861893299, 8.740950809, 8.622541819, 8.506587547, 8.393012444, 8.281744028, 8.172712735, 8.065851766, 7.961096959, 7.858386654, 7.757661576, 7.658864719, 7.561941239, 7.466838352, 7.373505239, 7.281892954, 7.191954342, 7.103643951, 7.016917964, 6.93173412, 6.848051649, 6.765831204, 6.685034803, 6.605625767, 6.527568666, 6.450829266, 6.375374481, 6.301172321, 6.228191851, 6.156403146, 6.08577725, 6.016286138, 5.947902678, 5.880600597, 5.814354444, 5.749139564, 5.68493206, 5.621708772, 5.559447239, 5.498125684, 5.437722978, 5.378218623, 5.319592726, 5.261825976, 5.204899625, 5.148795465, 5.093495813, 5.038983486, 4.985241791, 4.932254498, 4.880005836, 4.828480465, 4.77766347, 4.727540343, 4.678096968, 4.629319612, 4.581194906, 4.533709838, 4.486851742, 4.44060828, 4.394967438, 4.349917514, 4.305447104, 4.261545099, 4.218200669, 4.175403259, 4.13314258, 4.091408596, 4.050191524, 4.009481817, 3.969270166, 3.929547486, 3.890304912, 3.851533792, 3.813225681, 3.775372335, 3.737965702, 3.700997922, 3.664461317, 3.628348387, 3.592651805, 3.557364413, 3.522479215, 3.487989373, 3.453888204, 3.420169176, 3.3868259, 3.353852131, 3.32124176, 3.288988813, 3.257087445, 3.225531939, 3.194316701, 3.163436256, 3.132885247, 3.10265843, 3.072750672, 3.043156949, 3.013872339, 2.984892026, 2.956211289, 2.927825509, 2.899730158, 2.871920802, 2.844393095, 2.817142782, 2.790165689, 2.763457729, 2.737014893, 2.710833254, 2.68490896, 2.659238236, 2.63381738, 2.608642761, 2.583710818, 2.55901806, 2.534561061, 2.51033646, 2.486340961, 2.462571329, 2.43902439], color=RED, lw=3)
        x10, y10 = (10.0, 18.36734694)
        ax.plot([x10], [y10], 'o', color=RED, ms=10)
        ax.annotate('pass rate 10%:\n18.4% of rewards\ngo to failures', (x10, y10), xytext=(17, 48), fontsize=15, color=RED, fontweight='bold', arrowprops=dict(arrowstyle='-', color=RED, lw=1.2))
        x5, y5 = (5.0, 32.20338983)
        ax.plot([x5], [y5], 'o', color=PURPLE, ms=10, alpha=1 if step == 2 else 0)
        ax.text(7.5, y5 + 6, f'5%: {y5:.0f}%' if step == 2 else '5%: ?', fontsize=15, color=PURPLE, fontweight='bold')
        ax.set_xlim(0, 52)
        ax.set_ylim(0, 75)
        ax.set_xlabel("Policy's true pass rate (%)")
        ax.set_ylabel('Rewarded rollouts\nthat fail (%)')
        ax.grid(color=sp.C['grid'], lw=0.6)
        axis_type(ax, 15)
        sp.save(fig, HERE / f'false_share_step{step}.svg')

def shaping():
    """Number line: failure scores [-lambda, alpha], pass scores [1-lambda, 1+alpha]."""
    W, H = (900, 330)
    x0, x1 = (90, 830)

    def X(v):
        return x0 + (v + 1.0) / 2.5 * (x1 - x0)
    for step, sc in [(1, SimpleNamespace(failure_format=1.0, failure_tokens=400, format_weight=0.2, length_cap=8000, length_weight=0.3, name='The same shaped score with format weight 0.2 and length weight 0.3', pass_format=0.5, pass_tokens=7200, result=SimpleNamespace(example_failure=0.185, example_pass=0.83, highest_failure=0.2, lowest_pass=0.7, order_holds=True, weight_sum=0.5))), (2, SimpleNamespace(failure_format=1.0, failure_tokens=400, format_weight=0.4, length_cap=8000, length_weight=0.7, name='A shaped terminal score with format weight 0.4 and length weight 0.7 under an 8,000-token cap', pass_format=0.5, pass_tokens=7200, result=SimpleNamespace(example_failure=0.365, example_pass=0.57, highest_failure=0.4, lowest_pass=0.3, order_holds=False, weight_sum=1.1)))]:
        a, lam, res = ({1: 0.2, 2: 0.4}[step], {1: 0.3, 2: 0.7}[step], {1: SimpleNamespace(example_failure=0.185, example_pass=0.83, highest_failure=0.2, lowest_pass=0.7, order_holds=True, weight_sum=0.5), 2: SimpleNamespace(example_failure=0.365, example_pass=0.57, highest_failure=0.4, lowest_pass=0.3, order_holds=False, weight_sum=1.1)}[step])
        b = [f'<line x1="{x0}" y1="250" x2="{x1}" y2="250" stroke="{GRAY}" stroke-width="2"/>']
        for v in (-1, -0.5, 0, 0.5, 1, 1.5):
            b.append(f'<line x1="{X(v)}" y1="244" x2="{X(v)}" y2="256" stroke="{GRAY}" stroke-width="2"/>')
            b.append(text(X(v), 286, f'{v:g}', 20, GRAY))
        b.append(text(X(-1) - 10, 320, 'score', 20, GRAY, 'start'))
        b.append(box(X(-lam), 150, X(a) - X(-lam), 50, RED_L, RED, rx=6))
        b.append(text((X(-lam) + X(a)) / 2, 182, 'any failure', 22, RED, weight='bold'))
        b.append(box(X(1 - lam), 70, X(1 + a) - X(1 - lam), 50, GREEN_L, GREEN, rx=6))
        b.append(text((X(1 - lam) + X(1 + a)) / 2, 102, 'any pass', 22, GREEN, weight='bold'))
        ok = res.order_holds
        b.append(text(30, 40, f'α = {a}, λ = {lam}:  α + λ = {res.weight_sum:.1f}', 24, INK, 'start', 'bold'))
        b.append(text(870, 40, 'every pass outranks every failure' if ok else 'a short failure can beat a long pass', 22, GREEN if ok else RED, 'end', 'bold'))
        if not ok:
            b.append(box(X(1 - lam), 60, X(a) - X(1 - lam), 150, 'none', RED, rx=4, dash=True, sw=3))
        svg(HERE / f'shaping_step{step}.svg', W, H, '\n'.join(b))

def credit():
    """Five turns of the repair; one terminal failure; equal blame; forks."""
    W, H = (900, 400)
    turns = ['reproduce', 'read pool.py', 'debugger', 'bad edit', 'tests fail']
    fail = '−0.4'
    for step in (1, 2, 3):
        b = []
        for i, t in enumerate(turns):
            x = 10 + i * 178
            fill, stroke = (RED_L, RED) if i >= 3 else (BLUE_L, BLUE)
            b.append(box(x, 44, 160, 70, fill, stroke))
            b.append(text(x + 80, 88, t, 22, INK, weight='bold'))
            if i < 4:
                b.append(arrow(x + 160, 79, x + 176, 79, 'g'))
            b.append(text(x + 80, 32, f'turn {i + 1}' if i < 4 else 'reward 0', 23, GRAY if i < 4 else RED, weight='bold'))
            if step >= 2 and i < 4:
                b.append(f'<rect x="{x + 60}" y="132" width="40" height="48" fill="{RED}"/>')
                b.append(text(x + 80, 212, fail, 26, RED, weight='bold'))
        if step >= 3:
            xd = 10 + 2 * 178
            b.append(arrow(xd + 80, 224, xd + 80, 262, 'p', PURPLE, dash=True))
            b.append(box(60, 264, 780, 130, PURPLE_L, PURPLE, rx=10))
            k = '20'
            b.append(text(450, 300, f'fork {k} continuations before and after turn 3', 25, PURPLE, weight='bold'))
            b.append(text(450, 340, f'before: 1 of {k} pass = 0.05    after: 12 of {k} = 0.60', 25, INK))
            b.append(text(450, 380, 'debugger turn: +0.55  (noise bound 0.32)', 26, GREEN, weight='bold'))
        svg(HERE / f'credit_step{step}.svg', W, H, '\n'.join(b))

def lone_pass():
    import numpy as np
    g_small, g_big = (3, int(16.0))
    Gs = np.arange(2, g_big + 1)
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(Gs, [1.0, 1.414213562, 1.732050808, 2.0, 2.236067977, 2.449489743, 2.645751311, 2.828427125, 3.0, 3.16227766, 3.31662479, 3.464101615, 3.605551275, 3.741657387, 3.872983346], color=GREEN, lw=3, marker='o', ms=5)
    for g in (g_small, g_big):
        v = {3: 1.414213562, 16: 3.872983346}[g]
        ax.plot([g], [v], 'o', color=PURPLE, ms=11)
        ax.text(g + (0.6 if g == g_small else -0.4), v + (-0.75 if g == g_small else 0.25), f'G = {g}: {v:.2f}', fontsize=16, color=PURPLE, fontweight='bold', ha='left' if g == g_small else 'right')
    ax.set_xlim(1, 17)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel('Group size G')
    ax.set_ylabel('Lone-pass\ncoefficient')
    ax.grid(color=sp.C['grid'], lw=0.6)
    axis_type(ax, 16)
    sp.save(fig, HERE / 'lone_pass.svg')

def length_bias():
    """Token count and per-token push for two failures with one coefficient (column)."""
    W, H = (500, 400)
    ratio = 32.0
    full = 440
    b = []
    for k, (label, toks, push) in enumerate([('failure, 128 tokens', full / ratio, full), ('failure, 4,096 tokens', full, full / ratio)]):
        y = 34 + k * 190
        b.append(text(10, y, label, 27, INK, 'start', 'bold'))
        b.append(box(10, y + 14, toks, 36, RED_L, RED, rx=3, sw=2))
        b.append(text(10 + toks + 10, y + 42, 'tokens', 22, GRAY, 'start') if toks < 100 else '')
        b.append(f'<rect x="10" y="{y + 72}" width="{push:.2f}" height="36" fill="{RED}"/>')
        b.append(text(10 + push + 10 if push < 100 else 20, y + 99, 'push per token: 32' if k == 0 else 'push per token: 1', 24, 'white' if push >= 100 else RED, 'start', 'bold'))
    svg(HERE / 'length_bias.svg', W, H, '\n'.join(b))

def arch():
    import numpy as np
    ps = np.linspace(0, 1, 201)
    g, g_big = (int(8.0), int(16.0))
    x_hard, x_tgt = (5.0, 50.0)
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(ps * 100, [0.0, 0.03930695642, 0.07725530557, 0.1138854985, 0.1492369774, 0.1833481963, 0.2162566406, 0.2479988465, 0.278610421, 0.3081260603, 0.3365795687, 0.3640038771, 0.3904310613, 0.4158923594, 0.4404181898, 0.4640381683, 0.4867811252, 0.5086751221, 0.5297474681, 0.5500247364, 0.56953278, 0.5882967474, 0.606341098, 0.6236896174, 0.6403654322, 0.6563910246, 0.6717882469, 0.6865783356, 0.7007819253, 0.7144190625, 0.7275092187, 0.7400713039, 0.7521236794, 0.7636841706, 0.7747700792, 0.7853981959, 0.7955848121, 0.8053457317, 0.8146962828, 0.8236513288, 0.83222528, 0.840432104, 0.8482853367, 0.8557980927, 0.8629830753, 0.8698525873, 0.8764185397, 0.8826924627, 0.8886855137, 0.8944084879, 0.8998718262, 0.9050856247, 0.9100596433, 0.9148033136, 0.9193257479, 0.9236357465, 0.9277418062, 0.9316521275, 0.9353746222, 0.938916921, 0.94228638, 0.9454900881, 0.9485348735, 0.9514273103, 0.9541737249, 0.9567802021, 0.9592525914, 0.9615965122, 0.96381736, 0.9659203118, 0.9679103312, 0.9697921736, 0.9715703913, 0.9732493382, 0.974833174, 0.9763258696, 0.9777312102, 0.9790528005, 0.9802940678, 0.9814582661, 0.98254848, 0.9835676275, 0.9845184637, 0.985403584, 0.9862254266, 0.9869862758, 0.9876882643, 0.9883333755, 0.9889234465, 0.9894601693, 0.9899450937, 0.9903796282, 0.9907650426, 0.9911024688, 0.9913929023, 0.9916372036, 0.9918360991, 0.9919901819, 0.9920999125, 0.9921656195, 0.9921875, 0.9921656195, 0.9920999125, 0.9919901819, 0.9918360991, 0.9916372036, 0.9913929023, 0.9911024688, 0.9907650426, 0.9903796282, 0.9899450937, 0.9894601693, 0.9889234465, 0.9883333755, 0.9876882643, 0.9869862758, 0.9862254266, 0.985403584, 0.9845184637, 0.9835676275, 0.98254848, 0.9814582661, 0.9802940678, 0.9790528005, 0.9777312102, 0.9763258696, 0.974833174, 0.9732493382, 0.9715703913, 0.9697921736, 0.9679103312, 0.9659203118, 0.96381736, 0.9615965122, 0.9592525914, 0.9567802021, 0.9541737249, 0.9514273103, 0.9485348735, 0.9454900881, 0.94228638, 0.938916921, 0.9353746222, 0.9316521275, 0.9277418062, 0.9236357465, 0.9193257479, 0.9148033136, 0.9100596433, 0.9050856247, 0.8998718262, 0.8944084879, 0.8886855137, 0.8826924627, 0.8764185397, 0.8698525873, 0.8629830753, 0.8557980927, 0.8482853367, 0.840432104, 0.83222528, 0.8236513288, 0.8146962828, 0.8053457317, 0.7955848121, 0.7853981959, 0.7747700792, 0.7636841706, 0.7521236794, 0.7400713039, 0.7275092187, 0.7144190625, 0.7007819253, 0.6865783356, 0.6717882469, 0.6563910246, 0.6403654322, 0.6236896174, 0.606341098, 0.5882967474, 0.56953278, 0.5500247364, 0.5297474681, 0.5086751221, 0.4867811252, 0.4640381683, 0.4404181898, 0.4158923594, 0.3904310613, 0.3640038771, 0.3365795687, 0.3081260603, 0.278610421, 0.2479988465, 0.2162566406, 0.1833481963, 0.1492369774, 0.1138854985, 0.07725530557, 0.03930695642, 0.0], color=BLUE, lw=3)
        ax.text(50, 1.04, f'G = {g}', fontsize=15, color=BLUE, fontweight='bold', ha='center')
        ax.plot(ps * 100, [0.0, 0.07706887603, 0.1485422289, 0.2148010902, 0.2762022794, 0.3330798316, 0.3857463465, 0.4344942652, 0.4795970753, 0.5213104516, 0.5598733313, 0.5955089318, 0.6284257092, 0.6588182645, 0.6868681983, 0.712744916, 0.7366063883, 0.758599867, 0.7788625603, 0.7975222681, 0.8146979811, 0.8305004439, 0.8450326857, 0.858390519, 0.8706630086, 0.881932913, 0.8922770987, 0.9017669294, 0.9104686321, 0.9184436397, 0.9257489138, 0.9324372461, 0.9385575426, 0.9441550884, 0.949271797, 0.9539464432, 0.9582148815, 0.96211025, 0.9656631618, 0.9689018835, 0.9718525023, 0.974539082, 0.9769838086, 0.9792071266, 0.9812278661, 0.9830633606, 0.9847295581, 0.9862411237, 0.9876115355, 0.9888531737, 0.989977404, 0.9909946545, 0.9919144878, 0.9927456674, 0.9934962204, 0.9941734942, 0.9947842104, 0.9953345145, 0.9958300213, 0.9962758578, 0.9966767026, 0.9970368223, 0.997360105, 0.9976500919, 0.9979100056, 0.9981427773, 0.9983510706, 0.9985373049, 0.9987036757, 0.998852174, 0.9989846041, 0.9991025997, 0.9992076388, 0.9993010578, 0.9993840637, 0.999457746, 0.9995230869, 0.9995809716, 0.9996321966, 0.9996774781, 0.9997174595, 0.9997527179, 0.9997837706, 0.9998110804, 0.9998350609, 0.9998560811, 0.9998744693, 0.9998905174, 0.9999044841, 0.9999165976, 0.9999270588, 0.9999360437, 0.9999437052, 0.9999501755, 0.9999555675, 0.9999599762, 0.9999634799, 0.9999661414, 0.9999680087, 0.9999691157, 0.9999694824, 0.9999691157, 0.9999680087, 0.9999661414, 0.9999634799, 0.9999599762, 0.9999555675, 0.9999501755, 0.9999437052, 0.9999360437, 0.9999270588, 0.9999165976, 0.9999044841, 0.9998905174, 0.9998744693, 0.9998560811, 0.9998350609, 0.9998110804, 0.9997837706, 0.9997527179, 0.9997174595, 0.9996774781, 0.9996321966, 0.9995809716, 0.9995230869, 0.999457746, 0.9993840637, 0.9993010578, 0.9992076388, 0.9991025997, 0.9989846041, 0.998852174, 0.9987036757, 0.9985373049, 0.9983510706, 0.9981427773, 0.9979100056, 0.9976500919, 0.997360105, 0.9970368223, 0.9966767026, 0.9962758578, 0.9958300213, 0.9953345145, 0.9947842104, 0.9941734942, 0.9934962204, 0.9927456674, 0.9919144878, 0.9909946545, 0.989977404, 0.9888531737, 0.9876115355, 0.9862411237, 0.9847295581, 0.9830633606, 0.9812278661, 0.9792071266, 0.9769838086, 0.974539082, 0.9718525023, 0.9689018835, 0.9656631618, 0.96211025, 0.9582148815, 0.9539464432, 0.949271797, 0.9441550884, 0.9385575426, 0.9324372461, 0.9257489138, 0.9184436397, 0.9104686321, 0.9017669294, 0.8922770987, 0.881932913, 0.8706630086, 0.858390519, 0.8450326857, 0.8305004439, 0.8146979811, 0.7975222681, 0.7788625603, 0.758599867, 0.7366063883, 0.712744916, 0.6868681983, 0.6588182645, 0.6284257092, 0.5955089318, 0.5598733313, 0.5213104516, 0.4795970753, 0.4344942652, 0.3857463465, 0.3330798316, 0.2762022794, 0.2148010902, 0.1485422289, 0.07706887603, 0.0], color=AMBER, lw=2.5, ls='--', alpha=1 if step == 2 else 0)
        if step == 2:
            ax.text(3, 1.03, f'G = {g_big}', fontsize=15, color=AMBER, fontweight='bold')
        c5, c50 = (0.3365795687, 0.9921875)
        ax.plot([x_hard], [c5], 'o', color=RED, ms=11)
        ax.text(x_hard + 2, c5 - 0.1, '5%: 0.34', fontsize=15, color=RED, fontweight='bold')
        ax.plot([x_tgt], [c50], 'o', color=GREEN, ms=11)
        ax.text(x_tgt + 2, c50 - 0.15, '50%: 0.99', fontsize=15, color=GREEN, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel('Task pass rate p (%)')
        ax.set_ylabel('Groups with\nreward contrast')
        ax.grid(color=sp.C['grid'], lw=0.6)
        axis_type(ax, 15)
        sp.save(fig, HERE / f'arch_step{step}.svg')

def informative():
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for i, (lab, sc, key, ck) in enumerate([('G 8, p 5%', SimpleNamespace(group=8, name='One update of 512 tasks with groups of 8 on a mix the policy passes 5 percent', pass_rate=0.05, result=SimpleNamespace(contrast=0.3365795687, rollout_tokens=15360, rollouts=4096, rollouts_per_informative_group=23.76852532, silent_share=0.6634204313, step_tokens=62914560), tasks=512, tokens_per_turn=512, turns=30), 'budget_hard_r', 'danger'), ('G 16, p 5%', SimpleNamespace(group=16, name='The same update with groups of 16', pass_rate=0.05, result=SimpleNamespace(contrast=0.5598733313, rollout_tokens=15360, rollouts=8192, rollouts_per_informative_group=28.57789272, silent_share=0.4401266687, step_tokens=125829120), tasks=512, tokens_per_turn=512, turns=30), 'budget_big_r', 'danger'), ('G 8, p 50%', SimpleNamespace(group=8, name='The same update on tasks the policy passes about half the time', pass_rate=0.5, result=SimpleNamespace(contrast=0.9921875, rollout_tokens=15360, rollouts=4096, rollouts_per_informative_group=8.062992126, silent_share=0.0078125, step_tokens=62914560), tasks=512, tokens_per_turn=512, turns=30), 'budget_tgt_r', 'data')]):
        y = len([('G 8, p 5%', SimpleNamespace(group=8, name='One update of 512 tasks with groups of 8 on a mix the policy passes 5 percent', pass_rate=0.05, result=SimpleNamespace(contrast=0.3365795687, rollout_tokens=15360, rollouts=4096, rollouts_per_informative_group=23.76852532, silent_share=0.6634204313, step_tokens=62914560), tasks=512, tokens_per_turn=512, turns=30), 'budget_hard_r', 'danger'), ('G 16, p 5%', SimpleNamespace(group=16, name='The same update with groups of 16', pass_rate=0.05, result=SimpleNamespace(contrast=0.5598733313, rollout_tokens=15360, rollouts=8192, rollouts_per_informative_group=28.57789272, silent_share=0.4401266687, step_tokens=125829120), tasks=512, tokens_per_turn=512, turns=30), 'budget_big_r', 'danger'), ('G 8, p 50%', SimpleNamespace(group=8, name='The same update on tasks the policy passes about half the time', pass_rate=0.5, result=SimpleNamespace(contrast=0.9921875, rollout_tokens=15360, rollouts=4096, rollouts_per_informative_group=8.062992126, silent_share=0.0078125, step_tokens=62914560), tasks=512, tokens_per_turn=512, turns=30), 'budget_tgt_r', 'data')]) - 1 - i
        r = {'budget_hard_r': 23.76852532, 'budget_big_r': 28.57789272, 'budget_tgt_r': 8.062992126}[key]
        ax.barh(y, r, color=sp.C[ck], height=0.6)
        ax.text(-0.6, y, lab, ha='right', va='center', fontsize=16, fontweight='bold', color=sp.C['text'])
        ax.text(r + 0.6, y, {'budget_hard_r': '23.8', 'budget_big_r': '28.6', 'budget_tgt_r': '8.1'}[key], va='center', fontsize=16, fontweight='bold', color=sp.C[ck])
    ax.set_yticks([])
    ax.set_xlim(0, 36)
    ax.set_xlabel('Rollouts per informative group')
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=sp.C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    axis_type(ax, 15)
    sp.save(fig, HERE / 'informative.svg')

def records():
    """Turn 2 was conditioned on the full log; the final transcript holds a summary."""
    W, H = (900, 470)
    for step in (1, 2):
        b = [text(30, 36, 'what the policy saw at turn 2', 22, BLUE, 'start', 'bold')]
        b.append(box(30, 50, 380, 60, GREEN_L, GREEN, rx=6))
        b.append(text(220, 88, 'full failure log', 23, INK))
        b.append(arrow(410, 80, 470, 80, 'b', BLUE))
        b.append(box(476, 50, 240, 60, BLUE_L, BLUE, rx=6))
        b.append(text(596, 88, 'turn 2: the patch', 23, BLUE, weight='bold'))
        b.append(text(30, 166, 'the transcript after compaction at turn 4', 22, GRAY, 'start', 'bold'))
        b.append(box(30, 180, 160, 60, GREEN_L, GREEN, rx=6, dash=True))
        b.append(text(110, 218, 'summary', 23, INK))
        b.append(arrow(190, 210, 250, 210, 'r', RED))
        b.append(box(256, 180, 240, 60, BLUE_L, BLUE, rx=6))
        b.append(text(376, 218, 'turn 2: the patch', 23, BLUE, weight='bold'))
        b.append(text(520, 218, 'scored against a context', 23, RED, 'start', 'bold'))
        b.append(text(520, 244, 'the policy never saw', 23, RED, 'start', 'bold'))
        if step >= 2:
            b.append(box(30, 290, 840, 160, PURPLE_L, PURPLE, rx=10))
            b.append(text(50, 326, 'Record per invocation', 23, PURPLE, 'start', 'bold'))
            items = ['delivered prefix token IDs', 'sampled token IDs', 'behavior log probability per token', 'checkpoint, sampling settings', 'verdict and its versions']
            for i, it in enumerate(items):
                col, row = (i % 2, i // 2)
                b.append(text(70 + col * 420, 366 + row * 32, '• ' + it, 23, INK, 'start'))
        svg(HERE / f'records_step{step}.svg', W, H, '\n'.join(b))

def states():
    """Four ways a rollout stops, and what each does to the reward (column)."""
    W, H = (520, 440)
    rows = [('budget exhausted', 'fail: reward 0', RED), ('tool error, recovered', 'an observation', BLUE), ('verifier crash, retry', 'pending: wait', AMBER), ('preempted, no retry', 'unknown: no reward', AMBER)]
    b = []
    for i, (s, t, c) in enumerate(rows):
        y = 10 + i * 108
        b.append(text(10, y + 30, s, 26, INK, 'start', 'bold'))
        b.append(arrow(30, y + 42, 30, y + 64, 'g'))
        b.append(box(50, y + 46, 440, 46, {RED: RED_L, BLUE: BLUE_L, AMBER: AMBER_L}[c], c, rx=6))
        b.append(text(66, y + 79, t, 26, c, 'start', 'bold'))
    svg(HERE / 'states.svg', W, H, '\n'.join(b))

def isolation():
    """Exploration side and assessment side, with the handoff between them."""
    W, H = (900, 470)
    for step in (1, 2):
        b = []
        b.append(box(20, 30, 360, 420, AMBER_L, AMBER, rx=12))
        b.append(text(200, 66, 'exploration side', 24, AMBER, weight='bold'))
        for i, (f, bad) in enumerate([('src/pool.py', False), ('tests/test_pool.py', False), ('conftest.py (added)', True)]):
            y = 100 + i * 70
            b.append(box(50, y, 300, 50, 'white', RED if bad else '#94a3b8', rx=6, sw=2.5 if bad else 2))
            b.append(text(200, y + 33, f, 23, RED if bad else INK, weight='bold' if bad else 'normal'))
        b.append(text(200, 340, "candidate's runner:", 23, INK))
        b.append(text(200, 372, '1 passed', 26, RED, weight='bold'))
        if step >= 2:
            b.append(box(520, 30, 360, 420, PURPLE_L, PURPLE, rx=12))
            b.append(text(700, 66, 'assessment side', 24, PURPLE, weight='bold'))
            b.append(box(550, 100, 300, 50, 'white', '#94a3b8', rx=6, sw=2))
            b.append(text(700, 133, 'fresh fixture + src/ diff', 23, INK))
            b.append(box(550, 170, 300, 50, 'white', '#94a3b8', rx=6, sw=2))
            b.append(text(700, 203, "controller's own tests", 23, INK))
            b.append(text(700, 340, "controller's runner:", 23, INK))
            b.append(text(700, 372, '1 failed', 26, GREEN, weight='bold'))
            b.append(arrow(380, 125, 516, 125, 'p', PURPLE, sw=3))
            b.append(text(450, 112, 'admit src/', 22, PURPLE, weight='bold'))
            b.append(arrow(380, 265, 450, 265, 'r', RED, sw=3))
            b.append(text(450, 300, 'refused', 23, RED, weight='bold'))
            b.append(f'<line x1="438" y1="250" x2="462" y2="280" stroke="{RED}" stroke-width="4"/>')
            b.append(f'<line x1="462" y1="250" x2="438" y2="280" stroke="{RED}" stroke-width="4"/>')
        svg(HERE / f'isolation_step{step}.svg', W, H, '\n'.join(b))

def allpass():
    ns = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    curves = [(0.98, 0.2), (0.9604, 0.04), (0.941192, 0.008), (0.92236816, 0.0016), (0.9039207968, 0.00032)]
    ax.plot(ns, [100 * good for good, _ in curves], color=GREEN, lw=3, marker='o', ms=7)
    ax.plot(ns, [100 * bad for _, bad in curves], color=RED, lw=3, marker='o', ms=7)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 300)
    ax.text(1.1, 150, 'correct (0.98 per trial)', fontsize=15, color=GREEN, fontweight='bold')
    ax.text(3.15, 1.5, '3 trials: 0.8%', fontsize=16, color=RED, fontweight='bold')
    ax.text(1.1, 0.025, 'wrong\n(0.20 per trial)', fontsize=15, color=RED, fontweight='bold')
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
    ax.set_xticks(ns)
    ax.set_xlabel('Trials that must all pass')
    ax.set_ylabel('Accepted (%)')
    ax.grid(color=sp.C['grid'], lw=0.6)
    axis_type(ax, 16)
    sp.save(fig, HERE / 'allpass.svg')

def slot():
    parts = [('model', 2.4, sp.C['memory']), ('tools', 1.6, sp.C['compute']), ('assess', 0.5, sp.C['network'])]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        for y, (name, reset, cap, cyc) in enumerate([('snapshot', 0.015, '204,120', '4.515 s'), ('cold', 3.5, '115,200', '8.0 s')]):
            left = 0
            vis = step == 2 or name == 'cold'
            for lab, v, c in parts:
                ax.barh(y, v, left=left, color=c, height=0.55, alpha=1 if vis else 0)
                if y == 1:
                    ax.text(left + v / 2, 1.42 if lab != 'assess' else 0.5, lab, ha='center', va='bottom' if lab != 'assess' else 'center', fontsize=14, color=c, fontweight='bold')
                left += v
            ax.barh(y, reset, left=left, color=sp.C['danger'], height=0.55, alpha=1 if vis else 0)
            if y == 1:
                ax.text(left + reset / 2, 1.42, 'reset', ha='center', va='bottom', fontsize=14, color=sp.C['danger'], fontweight='bold')
            ax.text(left + reset + 0.15, y, f'{cyc}: {cap}/h', va='center', fontsize=14, fontweight='bold', color=sp.C['text'], alpha=1 if vis else 0)
            ax.text(-0.15, y, name, ha='right', va='center', fontsize=14, fontweight='bold', color=sp.C['text'])
        ax.set_yticks([])
        ax.set_ylim(-0.5, 1.9)
        ax.set_xlim(0, 13)
        ax.set_xlabel('Seconds per episode in one slot')
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=sp.C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        axis_type(ax, 14)
        sp.save(fig, HERE / f'slot_step{step}.svg')

def yue():
    xs = [0.0, 150.0, 300.0, 450.0]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(xs, [9.9, 26.1, 33.6, 42.5], color=GREEN, lw=3, marker='o', ms=8)
    ax.plot(xs, [67.2, 66.3, 65.3, 64.3], color=BLUE, lw=3, marker='o', ms=8)
    x_end = xs[-1] + 10
    ax.text(10, 77, 'pass@256: one of 256 samples', fontsize=14, color=BLUE, fontweight='bold')
    ax.text(x_end, 64.3, '64.3', fontsize=14, color=BLUE, fontweight='bold', va='center')
    ax.text(-15, 67.2, '67.2', fontsize=14, color=BLUE, fontweight='bold', va='center', ha='right')
    ax.text(200, 10, 'pass@1: one sample', fontsize=14, color=GREEN, fontweight='bold')
    ax.text(-15, 9.9, '9.9', fontsize=14, color=GREEN, fontweight='bold', va='center', ha='right')
    ax.text(x_end, 42.5, '42.5', fontsize=14, color=GREEN, fontweight='bold', va='center')
    ax.set_xlim(-90, 540)
    ax.set_ylim(0, 90)
    ax.set_xticks(xs)
    ax.set_xticklabels(['base'] + [f'{x:.0f}' for x in xs[1:]])
    ax.set_xlabel('GRPO training steps')
    ax.set_ylabel('Problems solved (%)')
    ax.grid(color=sp.C['grid'], lw=0.6)
    axis_type(ax, 15)
    sp.save(fig, HERE / 'yue.svg')

def thumbs():
    """Text-free pictures for the recall and next-lecture slides."""
    b = [box(10, 40, 600, 60, GREEN_L, GREEN, rx=4), f'<rect x="473" y="40" width="107" height="60" fill="{GRAY_L}" stroke="#94a3b8" stroke-width="2"/>', f'<rect x="580" y="40" width="30" height="60" fill="{INK}"/>']
    svg(HERE / 'thumb_masking.svg', 620, 140, '\n'.join(b))
    b = [f'<circle cx="300" cy="50" r="34" fill="{BLUE_L}" stroke="{BLUE}" stroke-width="4"/>']
    for i in range(4):
        x = 90 + i * 140
        b.append(f'<line x1="300" y1="84" x2="{x}" y2="166" stroke="{GRAY}" stroke-width="4"/>')
        b.append(f'<circle cx="{x}" cy="190" r="26" fill="{AMBER_L}" stroke="{AMBER}" stroke-width="4"/>')
    svg(HERE / 'thumb_multiagent.svg', 600, 230, '\n'.join(b))

def forks():
    """Continuation forks at every boundary against two chosen turns (NM 17.2)."""
    W, H = (440, 300)
    n = int(30.0)
    b = []
    for row, (label, every, mult, col) in enumerate((('every boundary', True, 'forks_all_mult', RED), ('turns 10 and 20', False, 'forks_two_mult', GREEN))):
        y = 70 + row * 150
        b.append(text(8, y - 34, label, 29, col, 'start', 'bold'))
        b.append(f'<line x1="8" y1="{y}" x2="300" y2="{y}" stroke="{GRAY}" stroke-width="5"/>')
        picks = range(1, n) if every else (int(10.0), int(20.0))
        for t in picks:
            x = 8 + 292 * t / n
            for d in (-1, 1):
                b.append(f'<line x1="{x:.1f}" y1="{y}" x2="{x + 12:.1f}" y2="{y + d * 22}" stroke="{col}" stroke-width="2.4"/>')
        bar = 300 * {'every boundary': 117.0, 'turns 10 and 20': 9.0}[label] / 117.0
        b.append(box(8, y + 34, max(bar, 8), 34, col, col, rx=3))
        b.append(text(max(bar, 8) + 20, y + 62, f"{ {'every boundary': '117', 'turns 10 and 20': '9'}[label]}×", 32, col, 'start', 'bold'))
    svg(HERE / 'forks.svg', W, H, '\n'.join(b))

def loop_limits():
    """Three limits on the rollout loop: barrier, staleness, verifier pool."""
    W, H = (900, 330)
    b = []
    heads = ['Group barrier', 'Staleness', 'Verifier pool']
    for i, h in enumerate(heads):
        x = 8 + i * 300
        b.append(box(x, 6, 284, 318, 'white', PURPLE, rx=10))
        b.append(text(x + 142, 40, h, 25, PURPLE, weight='bold'))
    lat = [0.2, 2.1, 4.2, 16.8]
    top = max(lat)
    for k, v in enumerate(sorted(lat)):
        y = 70 + k * 44
        b.append(box(26, y, 248, 30, GRAY_L, GRAY, rx=3, dash=True))
        b.append(box(26, y, max(248 * v / top, 4), 30, AMBER if v == top else BLUE, AMBER if v == top else BLUE, rx=3))
    b.append(text(150, 272, 'the step waits for the slowest', 20))
    b.append(text(150, 304, '65% of slot time idle', 22, RED, weight='bold'))
    for k in range(4):
        x = 330 + k * 64
        b.append(box(x, 70, 54, 40, PURPLE_L, PURPLE, rx=4))
        b.append(text(x + 27, 97, f'v{k + 1}', 21, PURPLE, weight='bold'))
    b.append(box(330, 134, 246, 34, AMBER_L, AMBER, rx=4))
    b.append(text(453, 158, 'one long rollout', 20))
    b.append(box(330, 180, 54, 34, BLUE_L, BLUE, rx=4))
    b.append(text(357, 204, 'short', 18))
    b.append(text(458, 262, 'long rollouts go stale', 21))
    b.append(text(458, 296, 'lag = versions passed', 22, RED, weight='bold'))
    b.append(text(758, 86, '32 trials/s arrive', 21))
    slots = 10
    busy = round(slots * 80.0 / 100)
    for k in range(slots):
        x = 630 + k % 5 * 52
        y = 110 + k // 5 * 52
        b.append(box(x, y, 42, 42, BLUE if k < busy else GRAY_L, BLUE if k < busy else GRAY, rx=4))
    b.append(text(758, 252, '768 busy of 960 slots', 21))
    b.append(text(758, 296, '80% target', 22, RED, weight='bold'))
    svg(HERE / 'loop_limits.svg', W, H, '\n'.join(b))
if __name__ == '__main__':
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
    print('figures written')
