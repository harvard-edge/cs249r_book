#!/usr/bin/env python3
"""Figures for Volume II, Lecture 13 (Security and Privacy).

Every number a figure shows comes from images/calcs.py, which reads it from
MLSysIM (registry, Literature, and Scenarios.*); figures read the recorded
values through val() and num(). Values computed there say "derived" on the
figure or on its slide; scenario inputs say "illustrative" or "scenario".
Shapes that carry no number (the Laplace densities, the breakdown plot's
spread around the honest value, layout coordinates) are drawn here.

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402  (records every value; figures read them through val/num)
from deckvals import val, num  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# Volume II C3 roles (volumes/vol2/palette.md)
V2 = {
    "c1": "#1F407A", "c1_l": "#EAF1F8",
    "c2": "#d97706", "c2_l": "#fef3c7",
    "c3": "#7c3aed", "c3_l": "#f3e8ff",
    "ok": "#16a34a", "ok_l": "#dcfce7",
    "risk": "#dc2626", "risk_l": "#fee2e2",
    "frame": "#94a3b8", "frame_l": "#f8fafc",
    "text": "#0f172a", "gray": "#64748b",
}


def canvas(w, h):
    """A diagram canvas in inches, so font sizes read as printed sizes."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fill, stroke, fs=11, color=None, bold=True, alpha=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec=stroke, lw=1.6, alpha=alpha, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs * 1.08,
            color=color or V2["text"], fontweight="bold" if bold else "normal",
            alpha=alpha, linespacing=1.15)


def arrow(ax, p, q, color, lw=1.8, alpha=1.0, ls="-", both=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>",
                                 mutation_scale=13, color=color, lw=lw, alpha=alpha,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def marker(ax, x, y, n, alpha=1.0):
    ax.add_patch(plt.Circle((x, y), 0.13, fc=V2["risk"], ec="white", lw=1.2, alpha=alpha, zorder=6))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=10, color="white",
            fontweight="bold", alpha=alpha, zorder=7)


def save(fig, name):
    sp.save(fig, out / name)


# ------------------------------------------------------------ recall canary
sp.hbars(out / "recall_canary.svg", [
    ("Cutover", num("cutover_pct"), "danger", f"{val('cutover_pct')}%"),
    ("Canary", num("canary_pct"), "compute", f"{val('canary_pct')}%"),
], "", log=False, size=(3.6, 1.0), xlabel="Traffic exposed until rollback")


# ------------------------------------------------------------ fleet interface map
def fleetmap(path, step):
    W, H = 5.8, 2.6
    fig, ax = canvas(W, H)
    # components
    box(ax, 0.05, 1.45, 0.95, 0.75, "Data feed\nweb, users,\nlabelers", V2["ok_l"], V2["ok"], fs=9.8)
    ax.add_patch(FancyBboxPatch((1.35, 0.95), 1.75, 1.45, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=V2["c1_l"], ec=V2["c1"], lw=1.6))
    ax.text(2.225, 2.22, "Training cluster", ha="center", va="center", fontsize=11, fontweight="bold",
            color=V2["text"])
    ax.text(2.225, 1.98, f"{val('ref_gpus')} GPUs, {val('ref_nodes')} nodes", ha="center", va="center", fontsize=10,
            color=V2["text"])
    # ring of nodes
    for k in range(8):
        a = 2 * math.pi * k / 8
        cx, cy = 2.225 + 0.42 * math.cos(a), 1.5 + 0.26 * math.sin(a)
        ax.add_patch(Rectangle((cx - 0.08, cy - 0.06), 0.16, 0.12, fc="white", ec=V2["c1"], lw=1.2, zorder=3))
    ax.add_patch(plt.matplotlib.patches.Ellipse((2.225, 1.5), 0.84, 0.52, fill=False, ec=V2["c2"], lw=2.0, zorder=2))
    box(ax, 1.35, 0.1, 1.75, 0.55, "Node admission\nattestation, keys", V2["c3_l"], V2["c3"], fs=10.5)
    box(ax, 3.4, 1.45, 0.95, 0.75, "Registry\nrelease\nrecord", V2["c3_l"], V2["c3"], fs=10.5)
    box(ax, 4.6, 1.45, 1.15, 0.75, "Serving API\nshared GPUs", V2["c1_l"], V2["c1"], fs=10.5)
    box(ax, 4.6, 0.1, 1.15, 0.55, "Users,\ntenants", V2["frame_l"], V2["frame"], fs=10.5)
    arrow(ax, (1.0, 1.82), (1.35, 1.82), V2["ok"])
    arrow(ax, (3.1, 1.82), (3.4, 1.82), V2["gray"])
    arrow(ax, (4.35, 1.82), (4.6, 1.82), V2["gray"])
    arrow(ax, (5.175, 0.65), (5.175, 1.45), V2["gray"], both=True)
    arrow(ax, (2.225, 0.65), (2.225, 0.95), V2["c3"], both=True)
    if step >= 2:
        a = 1.0
        marker(ax, 1.17, 2.0, 1, a)
        ax.text(1.17, 2.43, "records", ha="center", fontsize=10, color=V2["risk"], fontweight="bold")
        marker(ax, 2.225, 1.5, 2, a)
        ax.text(2.225, 1.06, "gradients", ha="center", va="center", fontsize=10, color=V2["risk"],
                fontweight="bold")
        marker(ax, 2.5, 0.8, 3, a)
        ax.text(2.68, 0.8, "evidence, keys", ha="left", va="center", fontsize=10, color=V2["risk"],
                fontweight="bold")
        marker(ax, 3.25, 2.0, 4, a)
        ax.text(3.25, 2.43, "weights", ha="center", fontsize=10, color=V2["risk"], fontweight="bold")
        marker(ax, 5.35, 1.05, 5, a)
        ax.text(5.5, 1.05, "queries,\nanswers", ha="left", va="center", fontsize=10, color=V2["risk"],
                fontweight="bold")
        marker(ax, 4.6, 1.45, 6, a)
        ax.text(4.42, 1.2, "co-tenant\ntiming", ha="right", va="center", fontsize=10, color=V2["risk"],
                fontweight="bold")
    ax.set_xlim(0, W + 0.25)
    save(fig, path)


fleetmap("fleetmap_step1.svg", 1)
fleetmap("fleetmap_step2.svg", 2)


# ------------------------------------------------------------ threat model, federated round
def threatmodel(path):
    W, H = 5.5, 1.3
    fig, ax = canvas(W, H)
    box(ax, 0.05, 0.3, 1.55, 0.72, "Enrolled client\n(compromised)", V2["risk_l"], V2["risk"], fs=11)
    box(ax, 1.98, 0.3, 1.55, 0.72, "Coordinator\n(reads messages)", V2["c3_l"], V2["c3"], fs=11)
    box(ax, 3.9, 0.3, 1.55, 0.72, "Global model\n(every replica)", V2["c1_l"], V2["c1"], fs=11)
    arrow(ax, (1.6, 0.66), (1.98, 0.66), V2["risk"], lw=2.2)
    arrow(ax, (3.53, 0.66), (3.9, 0.66), V2["risk"], lw=2.2)
    ax.text(1.79, 1.1, "update", ha="center", fontsize=10.5, color=V2["risk"], fontweight="bold")
    for x, lab in ((1.79, "A"), (3.715, "B")):
        ax.plot([x, x], [0.02, 0.25], color=V2["text"], lw=1.4, ls="--")
        ax.text(x + 0.06, 0.1, f"boundary {lab}", ha="left", va="center", fontsize=10.5, fontweight="bold",
                color=V2["text"])
    save(fig, path)


threatmodel("threatmodel.svg")


# ------------------------------------------------------------ LLM control and data in one stream
def llm_stream(path):
    W, H = 5.8, 1.9
    fig, ax = canvas(W, H)
    segs = [("System\ninstructions", V2["c3_l"], V2["c3"], 1.05),
            ("User\ntext", V2["frame_l"], V2["frame"], 0.75),
            ("Retrieved page:\n\u201cignore previous\u2026\u201d", V2["risk_l"], V2["risk"], 1.55)]
    x = 0.05
    for t, f, s, w in segs:
        box(ax, x, 1.0, w, 0.75, t, f, s, fs=10.5)
        x += w + 0.05
    ax.text(0.05, 0.82, "one token stream: control and data share it", ha="left", va="top",
            fontsize=10.5, color=V2["gray"])
    box(ax, 3.62, 1.0, 0.68, 0.75, "Model", V2["c1_l"], V2["c1"], fs=11)
    arrow(ax, (3.52, 1.375), (3.62, 1.375), V2["gray"])
    box(ax, 4.55, 1.0, 1.2, 0.75, "Policy check\nprincipal,\npermissions", V2["ok_l"], V2["ok"], fs=10)
    arrow(ax, (4.3, 1.375), (4.55, 1.375), V2["gray"])
    box(ax, 4.55, 0.05, 1.2, 0.55, "Tools, data", V2["frame_l"], V2["frame"], fs=10.5)
    arrow(ax, (5.15, 1.0), (5.15, 0.6), V2["ok"])
    ax.text(4.45, 0.33, "output is untrusted input", ha="right", va="center", fontsize=10.5,
            color=V2["ok"], fontweight="bold")
    save(fig, path)


llm_stream("llm_stream.svg")


# ------------------------------------------------------------ two kinds of theft
def theft(path):
    W, H = 5.8, 2.0
    fig, ax = canvas(W, H)
    ax.text(1.4, 1.88, "Exact artifact theft", ha="center", va="top", fontsize=12, fontweight="bold",
            color=V2["risk"])
    ax.text(4.4, 1.88, "Approximate behavioral theft", ha="center", va="top", fontsize=12,
            fontweight="bold", color=V2["risk"])
    box(ax, 0.1, 0.65, 1.1, 0.8, "Bucket,\nregistry,\ndevice", V2["c3_l"], V2["c3"], fs=10.5)
    box(ax, 1.65, 0.65, 1.05, 0.8, "Same\nweights", V2["risk_l"], V2["risk"], fs=10.5)
    arrow(ax, (1.2, 1.05), (1.65, 1.05), V2["risk"])
    ax.text(1.4, 0.38, "needs file access", ha="center", va="center", fontsize=10.5, color=V2["text"])
    ax.plot([2.95, 2.95], [0.15, 1.8], color=V2["frame"], lw=1)
    box(ax, 3.1, 0.65, 0.95, 0.8, "Deployed\nAPI", V2["c1_l"], V2["c1"], fs=10.5)
    box(ax, 4.65, 0.6, 1.12, 0.9, "Surrogate:\nnew weights,\nsimilar\nanswers", V2["risk_l"], V2["risk"], fs=10)
    arrow(ax, (4.05, 1.2), (4.65, 1.2), V2["risk"])
    arrow(ax, (4.65, 0.9), (4.05, 0.9), V2["gray"])
    ax.text(4.35, 1.3, "queries", ha="center", va="bottom", fontsize=9.5, color=V2["risk"])
    ax.text(4.35, 0.8, "scores", ha="center", va="top", fontsize=9.5, color=V2["gray"])
    ax.text(4.4, 0.38, "needs only an account", ha="center", va="center", fontsize=10.5, color=V2["text"])
    save(fig, path)


theft("theft_types.svg")

# ------------------------------------------------------------ Carlini et al. 2024 query counts
fig, ax = plt.subplots(figsize=(5.8, 1.45))
rows = [("ada", num("steal_ada_q_m") * 1e6, "compute", f"< {val('steal_ada_q_m')}M queries: weight matrix"),
        ("babbage", num("steal_babbage_q_m") * 1e6, "compute", f"< {val('steal_babbage_q_m')}M queries: weight matrix"),
        ("gpt-3.5-turbo", num("steal_gpt35_q_m") * 1e6, "neutral", f"< {val('steal_gpt35_q_m')}M queries: hidden width")]
for i, (lab, v, ck, txt) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, v, color=C[ck], height=0.6)
    ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=13, fontweight="bold", color=C["text"])
    ax.text(v * 1.15, y, txt, va="center", fontsize=12.5, color=C[ck], fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(1e5, 3e9)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Queries to the public API (log scale)")
ax.grid(axis="x", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "carlini_queries.svg")


# ------------------------------------------------------------ cumulative bill (Napkin Math 13.1)
BUDGET = calcs.BUDGET
QUERIES, TRAIN = BUDGET.extraction_queries, BUDGET.training_cost.to("USD").magnitude


def pay_cost(q):
    return BUDGET.pay_per_query_cost(q).to("USD").magnitude


def tier_cost(q):
    return BUDGET.tier_cost(q).to("USD").magnitude


def life_cost(q):
    return BUDGET.lifetime_cost(q).to("USD").magnitude


def bill(path, step):
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    q = np.linspace(0, QUERIES, 400)
    ax.axhline(TRAIN / 1e3, color=C["neutral"], ls="--", lw=1.4)
    ax.text(0.05, TRAIN / 1e3 + 0.12, f"training cost, \\${val('ext_train_k')}K", fontsize=11, color=C["neutral"], va="bottom")
    ax.plot(q / 1e6, [pay_cost(v) / 1e3 for v in q], color=C["neutral"], lw=1.6, ls=":")
    ax.text(3.3, 4.3, "pay per query", fontsize=11, color=C["neutral"], ha="right")
    a2 = 1.0 if step >= 2 else 0.0
    ax.plot(q / 1e6, [tier_cost(v) / 1e3 for v in q], color=C["danger"], lw=2.6, alpha=a2)
    ax.text(4.95, tier_cost(QUERIES) / 1e3 + 0.2, f"daily quota: \\${val('ext_tier_cost')}\n{val('ext_days')} days", ha="right",
            va="bottom", fontsize=11, color=C["danger"], fontweight="bold", alpha=a2)
    a3 = 1.0 if step >= 3 else 0.0
    ax.plot(q / 1e6, [life_cost(v) / 1e3 for v in q], color=C["memory"], lw=2.6, alpha=a3)
    ax.text(2.3, 3.7, f"lifetime budget:\n\\${val('ext_life')}", ha="right", va="center", fontsize=11,
            color=C["memory"], fontweight="bold", alpha=a3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6.2)
    ax.set_xlabel("Queries collected (millions)")
    ax.set_ylabel("Attacker's bill (\\$K)")
    sp.save(fig, out / path)


for k in (1, 2, 3):
    bill(f"bill_step{k}.svg", k)


# ------------------------------------------------------------ bilevel poisoning loop
def bilevel(path):
    W, H = 5.8, 2.45
    fig, ax = canvas(W, H)
    box(ax, 0.05, 1.45, 2.3, 0.9, "Attacker (outer loop)\nchoose poison P to raise\ntarget loss",
        V2["risk_l"], V2["risk"], fs=10.5)
    # data grid
    gx, gy = 0.35, 0.15
    for r in range(3):
        for c in range(6):
            red = (r, c) in ((0, 1), (1, 4), (2, 2))
            ax.add_patch(Rectangle((gx + c * 0.24, gy + r * 0.24), 0.2, 0.2,
                                   fc=V2["risk"] if red else V2["ok_l"],
                                   ec=V2["risk"] if red else V2["ok"], lw=1))
    ax.text(gx + 0.72, gy + 0.85, "training data: clean S plus poison P", ha="center", fontsize=10.5,
            color=V2["text"], fontweight="bold")
    arrow(ax, (1.2, 1.45), (1.05, 1.1), V2["risk"], lw=2)
    ax.text(1.28, 1.27, "inject", fontsize=10.5, color=V2["risk"], fontweight="bold")
    box(ax, 2.68, 0.15, 1.8, 0.9, "Defender (inner loop)\ntrains: minimize\ntraining loss",
        V2["c1_l"], V2["c1"], fs=10.5)
    arrow(ax, (1.85, 0.5), (2.68, 0.5), V2["c1"], lw=2)
    box(ax, 4.6, 0.15, 1.15, 0.9, "Trained model\nscored on\ntarget loss", V2["frame_l"], V2["frame"], fs=10.5)
    arrow(ax, (4.48, 0.6), (4.6, 0.6), V2["c1"], lw=2)
    ax.add_patch(FancyArrowPatch((5.2, 1.05), (2.35, 1.95), connectionstyle="arc3,rad=0.35",
                                 arrowstyle="-|>", mutation_scale=13, color=V2["risk"], lw=2))
    ax.text(3.75, 1.4, "feedback: did the\nmodel learn it?", ha="center", fontsize=10.5,
            color=V2["risk"], fontweight="bold")
    save(fig, path)


bilevel("bilevel.svg")


# ------------------------------------------------------------ Nightshade grid, relabeled
def nightshade(path):
    """The paper's grid: the first four prompts, four doses, relabeled at slide size."""
    from PIL import Image
    im = Image.open(out / "poisoning_example.png").convert("RGB")
    crop = np.asarray(im.crop((96, 22, 96 + 4 * 72, 313)))
    fig = plt.figure(figsize=(3.7, 3.0))
    ax = fig.add_axes([0.2, 0.11, 0.79, 0.79])
    ax.imshow(crop, aspect="auto")
    ax.axis("off")
    for i, r in enumerate(["clean", val("ns_low"), val("ns_mid"), val("ns_high")]):
        fig.text(0.19, 0.11 + 0.79 * (1 - (i + 0.5) / 4), r, ha="right", va="center", fontsize=13,
                 fontweight="bold", color=C["danger"] if i else C["text"])
    fig.text(0.0, 0.5, "poison images", ha="left", va="center", fontsize=11, color=C["neutral"],
             rotation=90)
    for j, (a_, b_) in enumerate(zip(["Dog", "Car", "Handbag", "Hat"], ["Cat", "Cow", "Toaster", "Cake"])):
        x = 0.2 + 0.79 * (j + 0.5) / 4
        fig.text(x, 0.91, a_, ha="center", va="bottom", fontsize=12, fontweight="bold")
        fig.text(x, 0.10, b_, ha="center", va="top", fontsize=12, color=C["danger"], fontweight="bold")
    sp.save(fig, out / path)


nightshade("nightshade.svg")


# ------------------------------------------------------------ Stuxnet chain and its ML analog
def supply(path, step):
    W, H = 5.8, 2.2
    fig, ax = canvas(W, H)
    rows = [("Stuxnet, 2010", ["USB media", "Windows\nhosts", "Siemens\nStep7", "PLC logic"],
             "operators\nsee normal", 1.25, 1.0),
            ("ML supply chain", ["public\nartifact", "build\nhost", "training\njob", "model\nartifact"],
             "health checks\npass", 0.15, 1.0 if step >= 2 else 0.0)]
    for title, stages, mon, y, a in rows:
        ax.text(0.02, y + 0.78, title, fontsize=11, fontweight="bold", color=V2["text"], alpha=a)
        for i, s in enumerate(stages):
            x = 0.05 + i * 1.12
            box(ax, x, y, 0.92, 0.65, s, V2["frame_l"], V2["frame"], fs=10.5, alpha=a)
            ax.add_patch(plt.Circle((x + 0.8, y + 0.53), 0.06, fc=V2["risk"], alpha=a, zorder=5))
            if i < 3:
                arrow(ax, (x + 0.92, y + 0.33), (x + 1.12, y + 0.33), V2["risk"], alpha=a)
        box(ax, 4.62, y, 1.15, 0.65, mon, "white", V2["gray"], fs=9.5, ls="--", alpha=a)
        arrow(ax, (4.53, y + 0.33), (4.65, y + 0.33), V2["gray"], alpha=a)
    save(fig, path)


supply("supply_step1.svg", 1)
supply("supply_step2.svg", 2)


# ------------------------------------------------------------ breakdown point of the mean
rng = np.random.default_rng(13)
n_workers = int(num("bd_workers"))
honest = rng.normal(num("bd_honest"), 0.1, int(num("bd_honest_n")))  # spread around the honest value: drawing only
vs = np.logspace(0, math.log10(num("bd_attack")), 200)
means = [(honest.sum() + v) / n_workers for v in vs]
meds = [np.median(np.append(honest, v)) for v in vs]
fig, ax = plt.subplots(figsize=(3.6, 2.2))
ax.plot(vs, means, color=C["danger"], lw=2.6)
ax.plot(vs, meds, color=C["data"], lw=2.6)
ax.text(1.3, 120, f"mean of {val('bd_workers')}\n(what AllReduce\naverages)", color=C["danger"], fontsize=11,
        fontweight="bold", ha="left", va="top")
ax.text(1.2e3, 0.55, f"median of {val('bd_workers')}", color=C["data"], fontsize=11, fontweight="bold", ha="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.4, 300)
ax.set_xlabel("Value one worker sends")
ax.set_ylabel("Aggregate")
ax.set_xticks([1, 10, 100, 1000, 10000])
ax.set_xticklabels(["1", "10", "100", "1K", "10K"])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1", "10", "100"])
ax.minorticks_off()
sp.save(fig, out / "breakdown.svg")

# ------------------------------------------------------------ gather traffic, N_DP = 64
sp.hbars(out / "gather_traffic.svg", [
    ("Ring AllReduce (sum)", calcs.FLEET.allreduce_factor, "memory", f"{val('ring')} S"),
    ("Gather every gradient", num("gather"), "danger", f"{val('gather')} S  ({val('gather_ratio')} times)"),
], "", log=False, size=(5.8, 1.0), xlabel="Bytes each worker receives per step, S = gradient size (derived)")

# ------------------------------------------------------------ enclave page cache
fig, ax = plt.subplots(figsize=(3.0, 2.0))
labels = ["ResNet-18\nINT8", "ResNet-50\nINT8", "ResNet-50\nFP32"]
keys = ["r18_int8_mb", "r50_int8_mb", "r50_fp32_mb"]
vals = [num(k) for k in keys]
ax.bar(range(3), vals, color=[C["memory"], C["memory"], C["compute"]], width=0.62)
for i, k in enumerate(keys):
    ax.text(i, vals[i] + 3, f"{val(k)} MB", ha="center", fontsize=11, fontweight="bold")
ax.axhline(num("epc_mb"), color=C["danger"], ls="--", lw=1.8)
ax.text(2.4, num("epc_mb") + 3.8, f"EPC {val('epc_mib')} MiB = {val('epc_mb')} MB", ha="right", va="bottom", fontsize=11, color=C["danger"],
        fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 165)
ax.set_ylabel("Weights only (MB)")
sp.save(fig, out / "epc.svg")

# ------------------------------------------------------------ trusted compute latency
fig, ax = plt.subplots(figsize=(3.0, 2.0))
labels = ["Plaintext", "AES", "FHE"]
vals = [num("fhe_base_ms"), num("fhe_aes_total_ms"), num("fhe_s") * 1e3]
ax.bar(range(3), vals, color=[C["neutral"], C["neutral"], C["danger"]], width=0.62)
for i, (v, t) in enumerate(zip(vals, [f"{val('fhe_base_ms')} ms", f"{val('fhe_aes_total_ms')} ms",
                                      f"{val('fhe_s')} s"])):
    ax.text(i, v * 1.4, t, ha="center", fontsize=11, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(5, 2e6)
ax.set_yticks([10, 1e3, 1e5])
ax.set_yticklabels(["10 ms", "1 s", "100 s"])
ax.minorticks_off()
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Inference latency")
sp.save(fig, out / "fhe.svg")

# ------------------------------------------------------------ GPT-2 extraction funnel
sp.hbars_steps(str(out / "gpt2_funnel"), [
    ("Generated", num("gpt2_gen"), "neutral", val("gpt2_gen")),
    ("Inspected", num("gpt2_insp"), "compute", val("gpt2_insp")),
    ("Memorized", num("gpt2_mem"), "danger", val("gpt2_mem")),
], "", log=True, size=(3.3, 1.2), xlabel="GPT-2 samples (log scale)")


# ------------------------------------------------------------ audit lower bound, DP ceiling
AUDIT = calcs.AUDIT
fpr = np.linspace(0, 1, 301)
ceil = np.array([AUDIT.ceiling(f) for f in fpr])
aud = np.array([AUDIT.audit(f) for f in fpr])
assert np.all(aud <= ceil + 1e-12)
fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.fill_between(fpr, aud, ceil, color=C["compute_l"], alpha=0.9)
ax.plot(fpr, fpr, color=C["neutral"], ls="--", lw=1.4)
ax.plot(fpr, ceil, color=C["text"], lw=2.6)
ax.plot(fpr, aud, color=C["memory"], lw=2.6)
ax.text(0.04, 1.0, f"DP ceiling, $\\epsilon$ = {val('audit_eps')}\n(upper bound)", fontsize=12, color=C["text"],
        fontweight="bold", va="top", ha="left")
ax.text(0.62, 0.55, "audit\n(lower bound)", fontsize=12, color=C["memory"], fontweight="bold", va="top")
ax.text(0.8, 0.70, "chance", fontsize=10.5, color=C["neutral"], ha="left", va="top")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("Nonmembers flagged (FPR)")
ax.set_ylabel("Members flagged (TPR)")
sp.save(fig, out / "audit_roc.svg")


# ------------------------------------------------------------ DP output densities
def dens(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    x = np.linspace(-6, 7, 800)
    b = 1.0
    p = np.exp(-np.abs(x) / b) / (2 * b)
    q = np.exp(-np.abs(x - 1.0) / b) / (2 * b)
    ax.plot(x, p, color=C["memory"], lw=2.6)
    ax.plot(x, q, color=C["compute"], lw=2.6)
    ax.text(-1.25, 0.42, "release on D", color=C["memory"], fontsize=12, fontweight="bold", ha="right")
    ax.text(2.2, 0.42, "release on D\u2032\n(one person changed)", color=C["compute"], fontsize=12,
            fontweight="bold", ha="left", va="center")
    if step >= 2:
        m = (x >= 2.0) & (x <= 4.0)
        ax.fill_between(x[m], 0, p[m], color=C["memory_l"], alpha=0.9)
        ax.fill_between(x[m], 0, q[m], color=C["compute_l"], alpha=0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["f(D)", "f(D\u2032)"], fontsize=12)
    ax.set_xlim(-6, 7)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Released value (illustrative Laplace noise)")
    sp.save(fig, out / path)


dens("dp_step1.svg", 1)
dens("dp_step2.svg", 2)

# ------------------------------------------------------------ salary mean error
sp.hbars(out / "salary_err.svg", [
    (f"{val('sal_n')} people", num("sal_err"), "memory", f"\\${val('sal_err')}"),
    (f"{val('sal_small_n')} people", num("sal_small_err"), "danger", f"\\${val('sal_small_err')}"),
], "", log=False, size=(3.4, 1.0), xlabel="Expected error of the mean")

# ------------------------------------------------------------ accountant against composition
fig, ax = plt.subplots(figsize=(3.3, 1.8))
groups = [(f"{val('ab_short')} epochs", "ab_acc_short", "ab_strong_short"),
          (f"{val('ab_long')} epochs", "ab_acc_long", "ab_strong_long")]
for i, (g, mom_key, strong_key) in enumerate(groups):
    mom, strong = num(mom_key), num(strong_key)
    y = 1 - i
    ax.barh(y + 0.18, strong, height=0.34, color=C["neutral"])
    ax.barh(y - 0.18, mom, height=0.34, color=C["memory"])
    ax.text(strong + 0.4, y + 0.18, val(strong_key), va="center", fontsize=11, color=C["neutral"],
            fontweight="bold")
    ax.text(mom + 0.4, y - 0.18, val(mom_key), va="center", fontsize=11, color=C["memory"],
            fontweight="bold")
    ax.text(-0.5, y, g, ha="right", va="center", fontsize=11, fontweight="bold")
ax.text(13.6, 1.18, "strong composition", fontsize=10.5, color=C["neutral"], va="center")
ax.text(5.4, 0.82, "moments accountant", fontsize=10.5, color=C["memory"], va="center")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 28)
ax.set_xlabel("$\\epsilon$ reported for the same run")
sp.save(fig, out / "accountant.svg")

print("figures written to", out)
