#!/usr/bin/env python3
"""Figures for Lecture 2 (The Foundation Model). Run from this folder: python3 make_figs.py

Every plotted value is either an input with its source (comment) or comes from calcs.py's
inputs and data/model_runs.json (the real model run). Diagrams use the Volume III stack
palette (rules/volumes/vol3/palette.md): sky blue = model and generation, green = context,
amber = tools and actions, violet = runtime records, red = faults.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.patches as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = HERE
RUN = json.loads((HERE / "data" / "model_runs.json").read_text())

# Volume III role colors (palette.md)
V = {
    "model": "#0284c7", "model_l": "#e0f2fe", "model_t": "#0369a1",
    "ctx": "#16a34a", "ctx_l": "#dcfce7", "ctx_t": "#15803d",
    "tool": "#d97706", "tool_l": "#fef3c7", "tool_t": "#b45309",
    "rt": "#7c3aed", "rt_l": "#f3e8ff", "rt_t": "#6d28d9",
    "bad": "#dc2626", "bad_l": "#fee2e2", "bad_t": "#b91c1c",
    "st": "#94a3b8", "st_l": "#f8fafc", "ink": "#0f172a", "gray": "#64748b",
}

# Reference floors, Ch. 2.8.2: 70.6B params, 8-bit, 1,979 TFLOP/s, 3.35 TB/s
P, B, R, BW = 70.6e9, 1, 1979e12, 3.35e12
TP, TD = 2 * P / R, P * B / BW


def canvas(w=9.0, h=4.6, phys=None):
    """900 units wide; phys is the drawn width in inches (the slot size)."""
    phys = phys or w
    fig = plt.figure(figsize=(phys, h * phys / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 900)
    ax.set_ylim(0, h / w * 900)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fill, edge, text="", tc=None, fs=17, bold=True, ls="-", lw=2.2, alpha=1.0):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                   fc=fill, ec=edge, lw=lw, ls=ls, alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or V["ink"], fontweight="bold" if bold else "normal", alpha=alpha,
                linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color, lw=2.4, ls="-", alpha=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                mutation_scale=20, alpha=alpha))


def save(fig, name):
    sp.save(fig, out / name)


# =============================================================================
# 1. The proposal boundary (Ch. 2.2, figure 1, Definition 2.1), three steps
# =============================================================================
def boundary(k):
    fig, ax = canvas(9.0, 4.4, phys=6.4)
    H = 440
    box(ax, 15, 270, 220, 95, V["model_l"], V["model"], "Foundation\nmodel", V["model_t"], fs=14)
    box(ax, 15, 140, 220, 95, V["model_l"], V["model"], "Inference\nservice", V["model_t"], fs=14)
    ax.text(125, 385, "Generates text", ha="center", fontsize=13, color=V["model_t"], fontweight="bold")
    box(ax, 255, 210, 195, 90, "white", V["tool"], 'read_file(\n"src/pool.py")', V["tool_t"],
        fs=11.5, ls="--")
    ax.text(352, 198, "proposal:\nno authority", ha="center", va="top", fontsize=12, color=V["tool_t"])
    arrow(ax, 235, 215, 255, 235, V["model"])
    a2 = 1.0 if k >= 2 else 0.0
    a3 = 1.0 if k >= 3 else 0.0
    ax.add_patch(mp.FancyBboxPatch((470, 40), 420, 365, boxstyle="round,pad=0,rounding_size=10",
                                   fc=V["rt_l"], ec=V["rt"], lw=2.2, alpha=a2))
    ax.text(680, 375, "Agent runtime decides", ha="center", fontsize=14, color=V["rt_t"],
            fontweight="bold", alpha=a2)
    box(ax, 485, 260, 185, 85, "white", V["rt"], "Validate\nschema", V["rt_t"], fs=13, alpha=a2)
    box(ax, 690, 260, 190, 85, "white", V["rt"], "Authorize\nthe path", V["rt_t"], fs=13, alpha=a2)
    arrow(ax, 450, 255, 485, 295, V["tool"], alpha=a2)
    arrow(ax, 670, 302, 690, 302, V["rt"], alpha=a2)
    box(ax, 690, 140, 190, 85, V["tool_l"], V["tool"], "Execute in\nsandbox", V["tool_t"], fs=13, alpha=a3)
    box(ax, 485, 140, 185, 85, V["ctx_l"], V["ctx"], "Observation:\nfile excerpt", V["ctx_t"], fs=13, alpha=a3)
    arrow(ax, 785, 260, 785, 225, V["rt"], alpha=a3)
    arrow(ax, 690, 182, 670, 182, V["tool"], alpha=a3)
    ax.plot([577, 577], [140, 95], color=V["ctx"], lw=2.4, ls="--", alpha=a3)
    arrow(ax, 577, 95, 125, 95, V["ctx"], ls="--", alpha=a3)
    arrow(ax, 125, 95, 125, 140, V["ctx"], ls="--", alpha=a3)
    ax.text(270, 62, "observation joins the next call", ha="center", fontsize=12,
            color=V["ctx_t"], alpha=a3)
    ax.set_ylim(30, H)
    save(fig, f"boundary_step{k}.svg")


for k in (1, 2, 3):
    boundary(k)


# =============================================================================
# 2. Real tokenizer output, cl100k_base (run_model.py via tiktoken)
# =============================================================================
def token_row(ax, y, label, pieces, ids, color, fill):
    ax.text(10, y + 40, label, fontsize=13, color=V["ink"], fontweight="bold", va="center")
    x = 140
    for pc, i in zip(pieces, ids):
        shown = pc.replace(" ", "␣")
        w = max(62, 22 * len(shown) + 26)
        box(ax, x, y + 15, w, 52, fill, color, shown, color, fs=14, bold=False, lw=1.8)
        ax.text(x + w / 2, y + 8, str(i), ha="center", va="top", fontsize=12, color=V["gray"])
        x += w + 10
    return x


fig, ax = canvas(9.0, 2.8, phys=6.4)
f, ind = RUN["cl100k_flat"], RUN["cl100k_indented"]
token_row(ax, 175, "flush", f["pieces"], f["ids"], V["model"], V["model_l"])
token_row(ax, 60, "indented", ind["pieces"], ind["ids"], V["tool"], V["tool_l"])
ax.text(880, 15, "␣ = space.  Gray: token id.", ha="right", fontsize=13, color=V["gray"])
ax.set_ylim(0, 280)
save(fig, "tokens.svg")


# =============================================================================
# 3. Admission: S <= S_max - K_max (Ch. 2.3.2, 2.3.3), three steps
# =============================================================================
def admit(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    rows = [("Text only", [(6000, "6,000", C["memory"]), (1024, "1,024", C["neutral"])]),
            ("+ image", [(6000, "", C["memory"]), (2048, "2,048", C["compute"]),
                         (1024, "1,024", C["neutral"])]),
            ("+ image,\nhalf reserve", [(6000, "", C["memory"]), (2048, "2,048", C["compute"]),
                                        (512, "512", C["neutral"])])]
    for i, (lab, segs) in enumerate(rows):
        y = 2 - i
        vis = i < k
        x = 0
        for w, t, c in segs:
            ax.barh(y, w, left=x, color=c, height=0.6, alpha=1 if vis else 0)
            if t:
                ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=10.5, color="white",
                        fontweight="bold", alpha=1 if vis else 0)
            x += w
        ax.text(-150, y, lab, ha="right", va="center", fontsize=12, fontweight="bold",
                color=C["text"], alpha=1)
        over = x - 8192
        msg = f"{-over:,} to spare" if over < 0 else f"{over:,} over"
        ax.text(max(x, 8192) + 150, y, msg, va="center", fontsize=12, fontweight="bold",
                color=C["data"] if over < 0 else C["danger"], alpha=1 if vis else 0)
    ax.axvline(8192, color=C["danger"], lw=2, ls="--")
    ax.text(8192, 2.55, "window 8,192", ha="center", fontsize=11, color=C["danger"], fontweight="bold")
    ax.text(0, 2.55, "prompt", fontsize=11, color=C["memory"], fontweight="bold")
    ax.text(1500, 2.55, "image", fontsize=11, color=C["compute"], fontweight="bold")
    ax.text(2900, 2.55, "output reserve", fontsize=11, color=C["neutral"], fontweight="bold")
    ax.set_xlim(0, 11600)
    ax.set_ylim(-0.5, 2.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Tokens (scenario values, Ch. 2.3)")
    sp.save(fig, out / f"admit_step{k}.svg")


for k in (1, 2, 3):
    admit(k)


# =============================================================================
# 4. Real next-token probabilities at the path argument (run), three steps
# =============================================================================
pl = RUN["path_logits"]


def probs(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    toks = [t if t.strip() else repr(t) for t in pl["tokens"]]
    xs = range(len(toks))
    w = 0.38
    for i, (t, a, b) in enumerate(zip(toks, pl["p_T1"], pl["p_T05"])):
        right = t == "src"
        c1 = C["data"] if right else C["memory"]
        ax.bar(i - (w / 2 if k >= 2 else 0), a, width=w, color=c1)
        ax.text(i - (w / 2 if k >= 2 else 0), a + 0.015, f"{100 * a:.0f}" if a >= 0.01 else "<1", ha="center",
                fontsize=10, color=c1, fontweight="bold")
        ax.bar(i + w / 2, b if k >= 2 else 0, width=w, color=C["compute"], alpha=1 if k >= 2 else 0)
        ax.text(i + w / 2, b + 0.015, f"{100 * b:.0f}" if b >= 0.01 else "<1", ha="center", fontsize=10,
                color=C["compute"], fontweight="bold", alpha=1 if k >= 2 else 0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(toks, fontsize=10.5)
    ax.get_xticklabels()[toks.index("src")].set_color(C["data"])
    ax.get_xticklabels()[toks.index("src")].set_fontweight("bold")
    ax.set_ylim(0, 0.82)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.99, 0.95, "T = 1 (blue)", transform=ax.transAxes, ha="right", fontsize=12,
            color=C["memory"], fontweight="bold")
    ax.text(0.99, 0.82, "T = 0.5 (orange)", transform=ax.transAxes, ha="right", fontsize=12,
            color=C["compute"], fontweight="bold", alpha=1 if k >= 2 else 0)
    ax.text(0.99, 0.69, f"top-p 0.90 at T = 1 keeps {pl['nucleus_090_size']} tokens",
            transform=ax.transAxes, ha="right", fontsize=12, color=C["text"], fontweight="bold",
            alpha=1 if k >= 3 else 0)
    ax.annotate("the file the\nprompt names", (toks.index("src") - 0.2, pl["p_T1"][2] + 0.06),
                xytext=(3.4, 0.33), fontsize=11, color=C["data"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C["data"], lw=1.4))
    ax.set_ylabel("Probability (%)", fontsize=11)
    sp.save(fig, out / f"probs_step{k}.svg")


for k in (1, 2, 3):
    probs(k)


# =============================================================================
# 5. Napkin Math 2.1 (Ch. 2.4.2): logits [12, 10.5, 9, 6, 3], two steps
# =============================================================================
import math  # noqa: E402

z = [12, 10.5, 9, 6, 3]


def sm(v, T):
    e = [math.exp(x / T) for x in v]
    return [x / sum(e) for x in e]


def nm21(k):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    p1, p5 = sm(z, 1), sm(z, 0.5)
    w = 0.38
    for i in range(5):
        ax.bar(i - w / 2, p1[i], width=w, color=C["memory"])
        ax.bar(i + w / 2, p5[i] if k >= 2 else 0, width=w, color=C["compute"], alpha=1 if k >= 2 else 0)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"v{i + 1}\n{z[i]:g}" for i in range(5)], fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100"], fontsize=12)
    ax.text(4.5, 1.0, f"T = 1: v1 {100 * p1[0]:.1f}%", ha="right", fontsize=14, color=C["memory"],
            fontweight="bold")
    ax.text(4.5, 0.84, f"T = 0.5: v1 {100 * p5[0]:.1f}%", ha="right", fontsize=14, color=C["compute"],
            fontweight="bold", alpha=1 if k >= 2 else 0)
    ax.set_ylabel("Probability (%)", fontsize=13)
    sp.save(fig, out / f"nm21_step{k}.svg")


for k in (1, 2):
    nm21(k)


# =============================================================================
# 6. A truncated command deletes more (Ch. 2.5.2)
# =============================================================================
fig, ax = canvas(9.0, 3.6, phys=6.0)
box(ax, 330, 270, 240, 55, V["st_l"], V["st"], "/tmp/scratch/", V["ink"], fs=14)
box(ax, 120, 140, 230, 55, V["st_l"], V["st"], "cache_dir/", V["ink"], fs=14)
box(ax, 400, 140, 160, 55, V["st_l"], V["st"], "build/", V["ink"], fs=14)
box(ax, 610, 140, 190, 55, V["st_l"], V["st"], "results.db", V["ink"], fs=14)
for x in (235, 480, 705):
    ax.plot([450, x], [270, 195], color=V["st"], lw=2)
ax.add_patch(mp.FancyBboxPatch((105, 127), 260, 81, boxstyle="round,pad=0,rounding_size=10",
                               fc="none", ec=V["ctx"], lw=3, ls="--"))
ax.add_patch(mp.FancyBboxPatch((80, 112), 745, 230, boxstyle="round,pad=0,rounding_size=12",
                               fc="none", ec=V["bad"], lw=3))
ax.text(235, 95, "COMPLETED\nrm -rf /tmp/scratch/cache_dir", ha="center", va="top", fontsize=12,
        color=V["ctx_t"], fontweight="bold", family="monospace", linespacing=1.4)
ax.text(640, 95, "TRUNCATED\nrm -rf /tmp/scratch", ha="center", va="top", fontsize=12,
        color=V["bad_t"], fontweight="bold", family="monospace", linespacing=1.4)
ax.set_ylim(15, 360)
save(fig, "rmrf.svg")


# =============================================================================
# 7. Streaming rejection and cancel (Ch. 2.5.3): reject at step 96 of 4,096
# =============================================================================
fig, ax = plt.subplots(figsize=(5.8, 1.9))
K, rej = 4096, 96
ax.barh(0, rej, color=C["danger"], height=0.5)
ax.barh(0, K - rej, left=rej, color=C["neutral_l"], height=0.5, hatch="//", edgecolor=C["neutral"])
ax.text(rej + 60, 0.45, f"rejected at step {rej}", fontsize=11, color=C["danger"], fontweight="bold")
ax.text(rej + (K - rej) / 2 + 60, 0, f"{K - rej:,} steps ({(K - rej) * TD:.1f} s at 21.075 ms) "
        "skipped if the cancel is honored", ha="center", va="center", fontsize=10,
        color=C["text"], fontweight="bold", bbox=dict(fc="white", ec="none", pad=2))
ax.set_xlim(0, K)
ax.set_ylim(-0.5, 0.8)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Decode step (output allowance 4,096)")
sp.save(fig, out / "stream.svg")


# =============================================================================
# 8. What each check certifies (Ch. Table 3), four steps
# =============================================================================
layers = [("Syntax and schema", "parser, validator", "fields and types", "permission, values"),
          ("Authorization", "grant check", "target in the grant", "serves the goal"),
          ("Static analysis", "compiler, linter", "analyzed properties", "runtime behavior"),
          ("Behavioral tests", "sandboxed run", "exercised inputs", "untested inputs")]


def ladder(k):
    fig, ax = canvas(9.0, 4.2, phys=6.4)
    ax.text(390, 395, "Certifies", ha="center", fontsize=14, color=V["ctx_t"], fontweight="bold")
    ax.text(702, 395, "Leaves open", ha="center", fontsize=14, color=V["bad_t"], fontweight="bold")
    for i, (name, mech, yes, no) in enumerate(layers):
        y = 300 - i * 88
        a = 1.0 if i < k else 0.0
        box(ax, 5, y, 250, 76, V["rt_l"], V["rt"], "", alpha=a)
        ax.text(130, y + 52, name, ha="center", va="center", fontsize=12.5, color=V["rt_t"],
                fontweight="bold", alpha=a)
        ax.text(130, y + 22, mech, ha="center", va="center", fontsize=11, color=V["gray"], alpha=a)
        box(ax, 270, y + 8, 240, 60, V["ctx_l"], V["ctx"], yes, V["ctx_t"], fs=12, bold=False, alpha=a)
        ax.text(548, y + 38, "not", ha="center", va="center", fontsize=12, color=V["gray"], alpha=a)
        box(ax, 585, y + 8, 235, 60, V["bad_l"], V["bad"], no, V["bad_t"], fs=12, bold=False, alpha=a)
    ax.text(895, 300 + 52, "cheap", ha="right", fontsize=11, color=V["gray"])
    ax.text(895, 300 - 3 * 88 + 2, "costly", ha="right", fontsize=11, color=V["gray"])
    ax.annotate("", xy=(865, 300 - 3 * 88 + 28), xytext=(865, 300 + 38),
                arrowprops=dict(arrowstyle="-|>", color=V["gray"], lw=1.8))
    ax.set_ylim(25, 420)
    save(fig, f"ladder_step{k}.svg")


for k in (1, 2, 3, 4):
    ladder(k)


# =============================================================================
# 9. Self-review without new evidence (Huang et al. 2024, Tables 2 and 3, GPT-4)
# =============================================================================
huang = [("GSM8K", 95.5, 89.0, 97.5), ("CommonSenseQA", 82.0, 80.0, 85.5), ("HotpotQA", 49.0, 43.0, 59.0)]


def huang_fig(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    w = 0.27
    for i, (name, std, sc, orc) in enumerate(huang):
        for j, (v, c, show) in enumerate([(std, C["neutral"], True), (sc, C["danger"], True),
                                          (orc, C["data"], k >= 2)]):
            x = i + (j - 1) * w
            ax.bar(x, v if show else 0, width=w * 0.95, color=c)
            ax.text(x, v + 2, f"{v:.1f}", ha="center", fontsize=10.5, color=c, fontweight="bold",
                    alpha=1 if show else 0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([h[0] for h in huang], fontsize=12)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlim(-0.5, 2.6)
    ax.text(-0.42, 116, "one answer", color=C["neutral"], fontsize=10.5, fontweight="bold")
    ax.text(0.42, 116, "2 rounds of self-review", color=C["danger"], fontsize=10.5, fontweight="bold")
    ax.text(1.66, 116, "stopped by true label", color=C["data"], fontsize=11, fontweight="bold",
            alpha=1 if k >= 2 else 0)
    sp.save(fig, out / f"huang_step{k}.svg")


for k in (1, 2):
    huang_fig(k)


# =============================================================================
# 10. Residual defects after a fallible check (Ch. Systems Perspective 2.1), three steps
# =============================================================================
def resid(d, fa, pv):
    return fa * d / (fa * d + pv * (1 - d))


def bayes(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    rows = [("Generated", 0.18), ("Passed the check", resid(0.18, 0.65, 0.85)),
            ("Passed, FA halved", resid(0.18, 0.325, 0.85))]
    for i, (lab, d) in enumerate(rows):
        y = 2 - i
        a = 1 if i < k else 0
        ax.barh(y, d, color=C["danger"], height=0.55, alpha=a)
        ax.barh(y, 1 - d, left=d, color=C["data_l"], height=0.55, alpha=a)
        ax.text(d + 0.01, y, f"{100 * d:.1f}% defective" if i else "18% defective",
                va="center", fontsize=12, color=C["danger"], fontweight="bold", alpha=a)
        ax.text(-0.01, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    sp.save(fig, out / f"bayes_step{k}.svg")


for k in (1, 2, 3):
    bayes(k)


# =============================================================================
# 11. A grammar mask at the path position (real candidates from the run)
# =============================================================================
fig, ax = canvas(9.0, 3.4, phys=6.4)
ax.text(10, 305, 'Parser state: inside the string value of "path"', fontsize=13, color=V["rt_t"],
        fontweight="bold")
x = 10
for t, p in zip(pl["tokens"], pl["p_T1"]):
    w = max(72, 15 * len(t) + 30)
    box(ax, x, 185, w, 78, V["ctx_l"], V["ctx"], f'{t}\n{100 * p:.0f}%', V["ctx_t"], fs=12)
    x += w + 8
ax.text(10, 150, "All eight stay legal inside a string, so the mask keeps them.",
        fontsize=13, color=V["ctx_t"], va="top")
for i, t in enumerate(["newline", "tab", "\\q"]):
    box(ax, 10 + i * 125, 30, 112, 60, V["bad_l"], V["bad"], t, V["bad_t"], fs=13)
ax.text(395, 60, "Illegal here: logit \u2212\u221e", fontsize=13, color=V["bad_t"], va="center")
ax.set_ylim(15, 340)
save(fig, "mask.svg")


# =============================================================================
# 12. Measured on a laptop and in a real trace (run; Azure 2023 code trace)
# =============================================================================
# Log ticks as plain numbers (1, 10, 100): a 10 with a superscript exponent
# is too small to read in a half-width column.
_sp_save = sp.save


def _plain_ticks_save(fig, path):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    _sp_save(fig, path)


sp.save = _plain_ticks_save
tm = RUN["timing"]
pre_ms = tm["prefill_s_median"] / tm["prompt_tokens"] * 1e3
dec_ms = tm["decode_s_per_token_median"] * 1e3
sp.hbars(out / "laptop.svg", [("prompt", pre_ms, "memory", f"{pre_ms:.2f} ms"),
                              ("output", dec_ms, "compute", f"{dec_ms:.1f} ms")],
         "ms", log=True, size=(3.6, 1.9), xlabel="ms per token (log scale)")

rows = list(csv.DictReader(open(HERE / "data" / "azure_llm_code_2023.csv")))
ctx = [int(r["ContextTokens"]) for r in rows]
gen = [int(r["GeneratedTokens"]) for r in rows]
sp.hbars(out / "azure.svg", [("prompt", statistics.median(ctx), "memory", f"{statistics.median(ctx):,.0f}"),
                             ("output", statistics.median(gen), "compute", f"{statistics.median(gen):,.0f}")],
         "tokens", log=True, size=(3.6, 1.9), xlabel="median tokens per request (log)")
sp.save = _sp_save


# =============================================================================
# 14. Re-sent input against fresh prefill (Ch. NM 2.2), three steps
# =============================================================================
def resend(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    H, S0, D = 30, 8000, 1500
    t = list(range(1, H + 1))
    cum_resent, cum_fresh, a, b = [], [], 0, 0
    for j in range(H):
        a += S0 + j * D
        b += S0 if j == 0 else D
        cum_resent.append(a / 1e3)
        cum_fresh.append(b / 1e3)
    ax.plot(t, cum_resent, color=C["danger"], lw=3, alpha=1 if k >= 2 else 0)
    ax.plot(t, cum_fresh, color=C["data"], lw=3, alpha=1 if k >= 3 else 0)
    ax.text(29.5, cum_resent[-1], f"re-sent {cum_resent[-1]:,.1f}k", ha="right", va="bottom",
            fontsize=12, color=C["danger"], fontweight="bold", alpha=1 if k >= 2 else 0)
    ax.text(29.5, cum_fresh[-1] + 40, f"fresh prefill {cum_fresh[-1]:,.1f}k", ha="right", va="bottom",
            fontsize=12, color=C["data"], fontweight="bold", alpha=1 if k >= 3 else 0)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 1000)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Cumulative input\n(thousand tokens)", fontsize=11)
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, out / f"resend_step{k}.svg")


for k in (1, 2, 3):
    resend(k)


# =============================================================================
# 15. Two output formats against a 15 s deadline (Ch. NM 2.3), two steps
# =============================================================================
def deadline(k):
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    rows = [("Whole file", (8000 + 350) * TP, 1850 * TD), ("Local edit", (8000 + 120) * TP, 80 * TD)]
    for i, (lab, a, b) in enumerate(rows):
        y = 1 - i
        al = 1 if i < k else 0
        ax.barh(y, a, color=C["memory_l"], height=0.55, alpha=al, edgecolor=C["memory"])
        ax.barh(y, b, left=a, color=C["memory"], height=0.55, alpha=al)
        ax.text(a + b + 0.6, y, f"{a + b:.1f} s", va="center", fontsize=13, color=C["memory"],
                fontweight="bold", alpha=al)
        ax.text(-0.5, y, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.axvline(15, color=C["danger"], lw=2, ls="--")
    ax.axvspan(15, 46, color=C["danger_l"], alpha=0.5, zorder=0)
    ax.text(15.4, 1.55, "15 s deadline", fontsize=11, color=C["danger"], fontweight="bold")
    ax.text(45, -0.42, "light: prompt   dark: output", ha="right", fontsize=10, color=C["memory"])
    ax.set_xlim(0, 46)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Service time (s), reference floors")
    sp.save(fig, out / f"deadline_step{k}.svg")


for k in (1, 2):
    deadline(k)

# =============================================================================
# 16. Hook: the three serving bugs of War Story 2.1, redrawn from the postmortem's dates
# =============================================================================
# Source: Anthropic, "A postmortem of three recent issues" (2025): routing bug from Aug 5
# (0.8% of one model's requests), load-balancing change Aug 29, worst hour Aug 31 (16%),
# fix deployed Sep 4; TPU output corruption Aug 25 to Sep 2; approximate top-k
# miscompilation deployed Aug 26 (timeline figure), rolled back Sep 12.
import datetime as _dt  # noqa: E402


def day(m, d):
    return (_dt.date(2025, m, d) - _dt.date(2025, 8, 1)).days


def hook_timeline(k, size=(6.0, 2.4), fs=13.5, labels=True):
    fig, ax = plt.subplots(figsize=size)
    lanes = [("Routing to\nlong-context servers", day(8, 5), day(9, 4), C["memory"]),
             ("Corrupted output\n(misconfigured servers)", day(8, 25), day(9, 2), C["compute"]),
             ("Miscompiled\napproximate top-k", day(8, 26), day(9, 12), C["network"])]
    for i, (lab, a, b, c) in enumerate(lanes):
        y = 2 - i
        al = 1 if i < k else 0
        ax.plot([a, b], [y, y], color=c, lw=6, solid_capstyle="butt", alpha=al)
        ax.plot([a], [y], "o", ms=9, color=c, alpha=al)
        ax.plot([b], [y], "o", ms=9, color=C["data"], alpha=al)
        if labels:
            ax.text(-1, y, lab, ha="right", va="center", fontsize=fs - 1, color=c,
                    fontweight="bold", alpha=1)
    ax.plot([day(8, 31)], [2], "o", ms=12, color=C["danger"], alpha=1 if k >= 1 else 0)
    if labels:
        ax.annotate("16% of requests\nat the worst hour", (day(8, 31), 2), xytext=(day(8, 31), 2.55),
                    ha="center", fontsize=fs - 1, color=C["danger"], fontweight="bold")
        ax.text(day(8, 5), 2.3, "0.8% at first", ha="left", fontsize=fs - 1, color=C["memory"])
    ticks = [(8, 5), (8, 25), (9, 2), (9, 12)]
    ax.set_xticks([day(m, d) for m, d in ticks])
    ax.set_xticklabels(["Aug 5", "Aug 25", "Sep 2", "Sep 12"], fontsize=fs - 1)
    ax.set_xlim(day(8, 1), day(9, 15))
    ax.set_ylim(-0.6, 3.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    sp.save(fig, out / "hook_timeline.svg")


hook_timeline(3)

print("figures written")
