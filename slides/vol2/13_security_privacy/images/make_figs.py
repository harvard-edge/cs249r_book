#!/usr/bin/env python3
"""Figures for Volume II, Lecture 13 (Security and Privacy).

Every value is sourced. "SP" is the rendered chapter, Security & Privacy
(Volume II, Chapter 13). Values computed here say "derived" on the figure
or on its slide; scenario inputs say "illustrative" or "scenario".

  - Recall (Lecture 12 chapter, Napkin Math 12.5): a 5 percent canary
    against a 100 percent cutover, 20 times less exposure.
  - Fleet interface map: the reference cluster of 1,024 GPUs in 128 nodes
    of 8 (SP Sec. 13.6), its data feed, gradient exchange, node admission,
    release path, query API, and shared accelerators (SP Sec. 13.1, 13.6).
  - Extraction (SP Sec. 13.5.1, table 4, after Carlini et al. 2024):
    ada 1,024 dims, < 2e6 queries; babbage 2,048, < 4e6; babbage-002
    1,536, < 4e6; gpt-3.5-turbo-instruct and -1106, < 4e7 (dimension only,
    costs estimated). Abstract: under $20 for the full projection matrix of
    ada and babbage; under $2,000 estimated for gpt-3.5-turbo.
  - Cumulative bill (SP Napkin Math 13.1, figure 6 inputs; scenario):
    5M queries for 90 percent fidelity, training cost $5K, $0.001 per
    query, paid tier $50 per 30 days for 10K queries/day, lifetime
    allowance 100K queries. Daily-quota bill (5M / 10K) x 50 / 30 = $833;
    lifetime bill 10 x 50 / 30 + 4.9M x 0.001 = $4,917 (as in the book).
  - Nightshade dose-response (SP figure 12, Shan et al. 2024): the paper's
    image grid, cropped; labels redrawn at slide size.
  - Breakdown point (illustrative): 63 honest values near 1.0 and one
    value v from a compromised worker; mean and median of the 64.
  - Gather traffic (SP Sec. 13.6.1): ring AllReduce 2(N-1)/N x S per
    worker; gathering every gradient (N-1) x S; N_DP = 64 gives 1.97 S
    against 63 S, 32 times (derived).
  - Enclave page cache (SP Sec. 13.8.1): 128 MiB = 134.2 MB; ResNet-18
    INT8 11.7 MB, ResNet-50 INT8 25.6 MB, ResNet-50 FP32 102.4 MB.
  - Trusted compute (SP Napkin Math 13.2, assumed overheads): plaintext
    20 ms, AES 20.5 ms, FHE 200 s.
  - GPT-2 extraction (SP Sec. 13.5.2, Carlini et al. 2021): 600,000
    samples generated, 1,800 inspected, 604 confirmed memorized.
  - Audit against the DP ceiling (SP figure 7): ceiling TPR <=
    min(e^eps FPR, 1 - e^-eps (1 - FPR)) at eps = 1; audit curve
    TPR = FPR + 0.6 FPR (1 - FPR), illustrative, as in the book.
  - DP densities (illustrative): Laplace densities of scale b around f(D)
    and f(D') one sensitivity apart.
  - Salary mean (SP Napkin Math 13.3): sensitivity $200,000, eps = 1,
    expected |error| $200 for 1,000 people, $2,000 for 100.
  - Accountant (SP Sec. 13.9.2, Abadi et al. 2016): q = 0.01, sigma = 4,
    delta = 1e-5: moments accountant eps 1.26 (100 epochs), 2.55 (400);
    strong composition 9.34 and 24.22.

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
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
    ("Cutover", 100, "danger", "100%"),
    ("Canary", 5, "compute", "5%"),
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
    ax.text(2.225, 1.98, "1,024 GPUs, 128 nodes", ha="center", va="center", fontsize=10,
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
rows = [("ada", 2e6, "compute", "< 2M queries: weight matrix"),
        ("babbage", 4e6, "compute", "< 4M queries: weight matrix"),
        ("gpt-3.5-turbo", 4e7, "neutral", "< 40M queries: hidden width")]
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
QUERIES, PRICE, TRAIN = 5e6, 0.001, 5000.0
TIER_Q, TIER_USD, DAYS = 1e4, 50.0, 30.0
ALLOW = 1e5


def tier_cost(q):
    return q / TIER_Q * TIER_USD / DAYS


def life_cost(q):
    return min(q, ALLOW) / TIER_Q * TIER_USD / DAYS + max(q - ALLOW, 0) * PRICE


assert abs(tier_cost(QUERIES) - 833.3) < 1 and abs(life_cost(QUERIES) - 4916.7) < 1


def bill(path, step):
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    q = np.linspace(0, QUERIES, 400)
    ax.axhline(TRAIN / 1e3, color=C["neutral"], ls="--", lw=1.4)
    ax.text(0.05, TRAIN / 1e3 + 0.12, "training cost, \\$5K", fontsize=11, color=C["neutral"], va="bottom")
    ax.plot(q / 1e6, q * PRICE / 1e3, color=C["neutral"], lw=1.6, ls=":")
    ax.text(3.3, 4.3, "pay per query", fontsize=11, color=C["neutral"], ha="right")
    a2 = 1.0 if step >= 2 else 0.0
    ax.plot(q / 1e6, [tier_cost(v) / 1e3 for v in q], color=C["danger"], lw=2.6, alpha=a2)
    ax.text(4.95, tier_cost(QUERIES) / 1e3 + 0.2, "daily quota: \\$833\n500 days", ha="right",
            va="bottom", fontsize=11, color=C["danger"], fontweight="bold", alpha=a2)
    a3 = 1.0 if step >= 3 else 0.0
    ax.plot(q / 1e6, [life_cost(v) / 1e3 for v in q], color=C["memory"], lw=2.6, alpha=a3)
    ax.text(2.3, 3.7, "lifetime budget:\n\\$4,917", ha="right", va="center", fontsize=11,
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
    for i, r in enumerate(["clean", "50", "100", "300"]):
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
honest = rng.normal(1.0, 0.1, 63)
vs = np.logspace(0, 4, 200)
means = [(honest.sum() + v) / 64 for v in vs]
meds = [np.median(np.append(honest, v)) for v in vs]
fig, ax = plt.subplots(figsize=(3.6, 2.2))
ax.plot(vs, means, color=C["danger"], lw=2.6)
ax.plot(vs, meds, color=C["data"], lw=2.6)
ax.text(1.3, 120, "mean of 64\n(what AllReduce\naverages)", color=C["danger"], fontsize=11,
        fontweight="bold", ha="left", va="top")
ax.text(1.2e3, 0.55, "median of 64", color=C["data"], fontsize=11, fontweight="bold", ha="center")
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
N = 64
ring = 2 * (N - 1) / N
gather = N - 1
assert abs(gather / ring - 32) < 0.01
sp.hbars(out / "gather_traffic.svg", [
    ("Ring AllReduce (sum)", ring, "memory", f"{ring:.2f} S"),
    ("Gather every gradient", gather, "danger", f"{gather} S  (32 times)"),
], "", log=False, size=(5.8, 1.0), xlabel="Bytes each worker receives per step, S = gradient size (derived)")

# ------------------------------------------------------------ enclave page cache
fig, ax = plt.subplots(figsize=(3.0, 2.0))
labels = ["ResNet-18\nINT8", "ResNet-50\nINT8", "ResNet-50\nFP32"]
vals = [11.7, 25.6, 102.4]
ax.bar(range(3), vals, color=[C["memory"], C["memory"], C["compute"]], width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 3, f"{v:g} MB", ha="center", fontsize=11, fontweight="bold")
ax.axhline(134.2, color=C["danger"], ls="--", lw=1.8)
ax.text(2.4, 138, "EPC 128 MiB = 134.2 MB", ha="right", va="bottom", fontsize=11, color=C["danger"],
        fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 165)
ax.set_ylabel("Weights only (MB)")
sp.save(fig, out / "epc.svg")

# ------------------------------------------------------------ trusted compute latency
fig, ax = plt.subplots(figsize=(3.0, 2.0))
labels = ["Plaintext", "AES", "FHE"]
vals = [20, 20.5, 200000]
ax.bar(range(3), vals, color=[C["neutral"], C["neutral"], C["danger"]], width=0.62)
for i, (v, t) in enumerate(zip(vals, ["20 ms", "20.5 ms", "200 s"])):
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
    ("Generated", 600000, "neutral", "600,000"),
    ("Inspected", 1800, "compute", "1,800"),
    ("Memorized", 604, "danger", "604"),
], "", log=True, size=(3.3, 1.2), xlabel="GPT-2 samples (log scale)")


# ------------------------------------------------------------ audit lower bound, DP ceiling
def ceiling(f, eps=1.0):
    return min(math.exp(eps) * f, 1 - math.exp(-eps) * (1 - f))


fpr = np.linspace(0, 1, 301)
ceil = np.array([ceiling(f) for f in fpr])
aud = fpr + 0.6 * fpr * (1 - fpr)
assert np.all(aud <= ceil + 1e-12)
fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.fill_between(fpr, aud, ceil, color=C["compute_l"], alpha=0.9)
ax.plot(fpr, fpr, color=C["neutral"], ls="--", lw=1.4)
ax.plot(fpr, ceil, color=C["text"], lw=2.6)
ax.plot(fpr, aud, color=C["memory"], lw=2.6)
ax.text(0.04, 1.0, "DP ceiling, $\\epsilon$ = 1\n(upper bound)", fontsize=12, color=C["text"],
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
    ("1,000 people", 200, "memory", "\\$200"),
    ("100 people", 2000, "danger", "\\$2,000"),
], "", log=False, size=(3.4, 1.0), xlabel="Expected error of the mean")

# ------------------------------------------------------------ accountant against composition
fig, ax = plt.subplots(figsize=(3.3, 1.8))
groups = [("100 epochs", 1.26, 9.34), ("400 epochs", 2.55, 24.22)]
for i, (g, mom, strong) in enumerate(groups):
    y = 1 - i
    ax.barh(y + 0.18, strong, height=0.34, color=C["neutral"])
    ax.barh(y - 0.18, mom, height=0.34, color=C["memory"])
    ax.text(strong + 0.4, y + 0.18, f"{strong:g}", va="center", fontsize=11, color=C["neutral"],
            fontweight="bold")
    ax.text(mom + 0.4, y - 0.18, f"{mom:g}", va="center", fontsize=11, color=C["memory"],
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
