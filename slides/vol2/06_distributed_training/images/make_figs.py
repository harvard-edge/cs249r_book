#!/usr/bin/env python3
"""Plots and diagrams for Volume II, Lecture 6 (Distributed Training).

Every value is read from MLSysIM through calcs.py (the same Scenarios.* the
chapter reads) or from a cited Literature series:

  - Hook and batch sweep: slides/vol2/06_distributed_training/images/data/goyal_resnet50_large_minibatch.csv
    (Goyal et al. 2017, Table 1 and Figure 3).
  - 175B data-parallel step and the hybrid scoreboard: Scenarios.Gpt3DataParallel1K,
    Gpt3TensorDataParallel1K, Gpt3ThreeDParallel1K.
  - GPT-2 inside and across nodes: Scenarios.Gpt2DataParallelOneNode and
    Gpt2DataParallelFourNodes10GbE.
  - MLPerf Training: Scenarios.MLPerfLlmScalingSeries.
  - Straggler: Scenarios.StragglerFreeStep1K (a^N, calc_all_up_availability).
  - Critical batch: gradient-noise model of McCandlish et al. 2018 as the
    chapter plots it, steps 1 + Bcrit/B and samples 1 + B/Bcrit (dimensionless).
  - Memory wall: Scenarios.NotableModelMemoryWall.
  - ZeRO: Scenarios.Llama2_7B_ZeroStages64.
  - Pipeline: Scenarios.NaiveFourStagePipeline, GPipeFourStageWalkthrough,
    Pipeline8Stage32Micro, Pipeline16Stage32Micro (calc_pipeline_bubble).
  - Tensor parallelism: Scenarios.Gpt3TensorParallelLayerTp8.
  - Megatron-LM weak scaling: slides/vol2/06_distributed_training/images/data/megatron_lm_weak_scaling.csv.
  - The top half of the scaling-wall figure and the timelines are schematic.

Run from this folder: python make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
import slideplot as sp  # noqa: E402
import calcs as K  # noqa: E402  (records every value; figures read them)
from deckvals import val, num  # noqa: E402
from mlsysim.physics import calc_all_up_availability, calc_pipeline_bubble  # noqa: E402

plt = sp.plt
C = sp.C
out = HERE

# Volume II C3 roles for schematics (volumes/vol2/palette.md)
V = {
    "c1": "#1F407A", "c1_l": "#EAF1F8",      # compute
    "c2": "#d97706", "c2_l": "#fef3c7",      # communication
    "c3": "#7c3aed", "c3_l": "#f3e8ff",      # coordination
    "ok": "#16a34a", "ok_l": "#dcfce7",
    "bad": "#dc2626", "bad_l": "#fee2e2",
    "frame": "#94a3b8", "frame_l": "#f8fafc", "text": "#0f172a", "gray": "#64748b",
}


def label(ax, x, y, text, color, shown=True, **kw):
    """Text drawn in every step, invisible until shown, so steps align."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)


# ---------------------------------------------------------------------------
# A tiny SVG writer for diagrams (text stays text).
# ---------------------------------------------------------------------------
class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill, stroke="none", sw=2, rx=4, op=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')

    def text(self, x, y, s, size=22, color=V["text"], weight="bold", anchor="middle", op=1.0):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.items.append(
            f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" opacity="{op}">{s}</text>')

    def line(self, x1, y1, x2, y2, color, sw=3, arrow=False, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" opacity="{op}"{d}{m}/>')

    def save(self, path):
        cols = {V[k] for k in V} | {"#000000"}
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
            f'markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in cols)
        body = "\n".join(self.items)
        Path(path).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}"><defs>{defs}</defs>'
            f'<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')


# ---------------------------------------------------------------------------
# 1. Hook: Goyal et al. 2017, Table 1 (step 1 baseline only, then the 8k runs)
# ---------------------------------------------------------------------------
k8 = f"{val('goyal_batch_k')}k"
rows = [(f"{val('goyal_base_gpus')} GPUs, batch {val('goyal_base_batch')}", "goyal_err_base", "neutral"),
        (f"{val('goyal_gpus')} GPUs, batch {k8}, no warmup", "goyal_err_nowarm", "danger"),
        (f"{val('goyal_gpus')} GPUs, batch {k8}, gradual warmup", "goyal_err_warm", "data")]
def hook_warmup(path, k, size, fs, names):
    fig, ax = plt.subplots(figsize=size)
    for i, (name, key, ck) in enumerate(rows):
        y = 2 - i
        shown = i < k
        v = num(key)
        ax.barh(y, v - 23.0, left=23.0, color=C[ck], height=0.6, alpha=1 if shown else 0)
        ax.text(22.97, y, names[i], ha="right", va="center", fontsize=fs, fontweight="bold",
                color=C["text"])
        label(ax, v + 0.04, y, f"{val(key)}%", C[ck], shown, va="center", fontsize=fs + 0.5)
        if not shown:
            ax.text(23.1, y, "?", va="center", fontsize=fs + 3, fontweight="bold", color=C["neutral"])
    ax.set_xlim(23.0, 25.6)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([23, 24, 25])
    ax.tick_params(labelsize=fs - 1)
    ax.set_xlabel("top-1 validation error (%)", fontsize=fs - 0.5)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 3):
    hook_warmup(out / f"hook_warmup_step{k}.svg", k, (5.8, 1.6), 12.5, [r[0] for r in rows])
hook_warmup(out / "hook_warmup_col.svg", 3, (5.4, 3.0), 15,
            [f"batch {val('goyal_base_batch')}", f"{k8}, no warmup", f"{k8}, warmup"])

# ---------------------------------------------------------------------------
# 2. Data-parallel flow (diagram, 3 steps): replicas, local gradients, AllReduce
# ---------------------------------------------------------------------------
for k in (1, 2, 3):
    s = SVG(900, 330)
    xs = [40, 255, 470, 685]
    for i, x in enumerate(xs):
        s.rect(x, 20, 175, 70, V["frame_l"], V["frame"], rx=6)
        s.text(x + 87, 42, f"GPU {i}", 24, V["text"])
        s.text(x + 87, 70, "full model copy", 21, V["gray"], weight="normal")
        s.rect(x + 10, 108, 155, 42, V["ok_l"], V["ok"], rx=4)
        s.text(x + 87, 129, f"batch slice {i}", 21, V["ok"])
        op = 1 if k >= 2 else 0
        s.rect(x + 10, 172, 155, 42, V["c1_l"], V["c1"], rx=4, op=op)
        s.text(x + 87, 193, f"gradient g{i}", 21, V["c1"], op=op)
        s.line(x + 87, 150, x + 87, 168, V["c1"], 3, arrow=True, op=op)
        op3 = 1 if k >= 3 else 0
        s.line(x + 87, 214, x + 87, 240, V["c2"], 3, arrow=True, op=op3)
    op3 = 1 if k >= 3 else 0
    s.rect(40, 244, 820, 40, V["c2_l"], V["c2"], rx=6, op=op3)
    s.text(450, 264, "AllReduce: every GPU receives g = (g0 + g1 + g2 + g3) / 4", 24, "#b45309", op=op3)
    s.text(450, 311, "Each GPU applies the same update to its own copy", 24, V["text"], op=op3)
    s.save(out / f"dp_flow_step{k}.svg")

# ---------------------------------------------------------------------------
# 3. Scaling wall: schematic few/many bars (top), then the 175B step (bottom)
# ---------------------------------------------------------------------------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.3), gridspec_kw={"height_ratios": [2, 1.25]})
    for y, comp, name in [(1, 4.0, "Few GPUs"), (0, 1.0, "4× the GPUs")]:
        a1.barh(y, comp, color=V["c1"], height=0.6)
        a1.barh(y, 1.6, left=comp, color=V["c2"], height=0.6)
        a1.text(-0.08, y, name, ha="right", va="center", fontsize=13.5, fontweight="bold")
    a1.text(2.0, 1, "compute", color="white", ha="center", va="center", fontsize=12, fontweight="bold")
    a1.text(4.8, 1, "comm.", color="white", ha="center", va="center", fontsize=12, fontweight="bold")
    a1.text(1.8, 0, "comm.", color="white", ha="center", va="center", fontsize=12, fontweight="bold")
    a1.text(2.8, 0, "same comm., now longer than compute", va="center", fontsize=12.5,
            fontweight="bold", color=V["c2"])
    a1.text(8.3, 1.55, "illustrative", ha="right", fontsize=10, color=C["neutral"], style="italic")
    a1.set_xlim(0, 8.3)
    a1.set_ylim(-0.5, 1.7)
    a1.axis("off")
    sh = k >= 2
    tc, te = num("t_compute_s"), num("exposed_a_s")
    a2.barh(0, tc, color=V["c1"], height=0.6, alpha=1 if sh else 0)
    a2.barh(0, te, left=tc, color=V["c2"], height=0.6, alpha=1 if sh else 0)
    a2.text(-0.15, 0, f"{val('params_b')}B, {val('ref_gpus')} GPUs", ha="right", va="center", fontsize=13.5,
            fontweight="bold", alpha=1 if sh else 0)
    label(a2, tc / 2, 0, f"{val('t_compute_s')} s", "white", sh, ha="center", va="center", fontsize=11)
    label(a2, 8.0, 0, f"{val('exposed_a_s')} s exposed AllReduce", "white", sh, ha="center",
          va="center", fontsize=13)
    label(a2, tc + te + 0.2, 0, f"{val('eff_a_pct')}%", C["danger"], sh, va="center", fontsize=15)
    a2.set_xlim(0, 16.5)
    a2.set_ylim(-0.5, 0.5)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("seconds per step", alpha=1 if sh else 0)
    for t in a2.get_xticklabels():
        t.set_alpha(1 if sh else 0)
    a2.tick_params(axis="x", colors=C["text"] if sh else "white")
    a2.spines["bottom"].set_alpha(1 if sh else 0)
    fig.subplots_adjust(hspace=0.35)
    sp.save(fig, out / f"eta_bars_step{k}.svg")

# ---------------------------------------------------------------------------
# 4. GPT-2: one NVLink node against four 10 GbE nodes (Napkin Math 6.1, 6.2)
# ---------------------------------------------------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    data = [(f"{val('gpt2_gpus_1')} GPUs, one node\n(NVLink)", num("gpt2_step_ms"), num("gpt2_comm1_ms"),
             f"{val('gpt2_eff1_pct')}%", val("gpt2_step1_ms")),
            (f"{val('gpt2_gpus_4')} GPUs, four nodes\n({val('eth10_gbps')} Gb Ethernet)", num("gpt2_step_ms"),
             num("gpt2_comm4_ms"), f"{val('gpt2_eff4_pct')}%", val("gpt2_step4_round_ms"))]
    for i, (name, comp, comm, eff, total) in enumerate(data):
        y = 1 - i
        shown = i == 0 or k >= 2
        a = 1 if shown else 0
        ax.barh(y, comp, color=V["c1"], height=0.6, alpha=a)
        ax.barh(y, comm, left=comp, color=V["c2"], height=0.6, alpha=a)
        ax.text(-60, y, name, ha="right", va="center", fontsize=12, fontweight="bold")
        label(ax, comp + comm + 80, y, f"{total} ms   {eff}", C["data"] if i == 0 else C["danger"],
              shown, va="center", fontsize=12.5)
    label(ax, 900, 1, "compute", "white", True, ha="center", va="center", fontsize=11)
    label(ax, 900, 0, "compute", "white", k >= 2, ha="center", va="center", fontsize=11)
    label(ax, 4100, 0, f"communication {val('gpt2_comm4_round_ms')} ms", "white", k >= 2, ha="center",
          va="center", fontsize=11)
    ax.set_xlim(0, 8500)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("milliseconds per step")
    sp.save(fig, out / f"gpt2_step{k}.svg")

# ---------------------------------------------------------------------------
# 5. MLPerf Training: measured time to train against accelerators (real data)
# ---------------------------------------------------------------------------
# Short legend names for the scenario's five series, in its order.
names_ml = [("GPT-3, H100 (NVIDIA Eos)", C["memory"]), ("GPT-3, TPU v5p", C["data"]),
            ("GPT-3, TPU Trillium", C["compute"]), ("Llama 3.1 405B, H100", C["network"]),
            ("Llama 3.1 405B, GB200", C["danger"])]
ml = K.MLPERF.result
fig, ax = plt.subplots(figsize=(5.8, 2.4))
for (name, col), pts, eff, (_, (_, _, _, acc)) in zip(names_ml, ml.series, ml.efficiencies, K.MLPERF.specs):
    n = [r["accelerators"] for r in pts]
    t = [r["score_minutes"] for r in pts]
    ax.plot(n, t, "-o", ms=5, lw=2.2, color=col, zorder=3)
    ax.plot([n[0], n[-1]], [t[0], t[0] * n[0] / n[-1]], "--", lw=1.2, color=col, alpha=0.7)
    dy = -11 if acc == "TPU-trillium" else (5 if acc == "Blackwell" else 0)
    ax.annotate(f"{eff:.0%}", (n[-1], t[-1]), xytext=(6, dy), textcoords="offset points",
                fontsize=11, fontweight="bold", color=col, va="center")
    ax.plot([], [], "-o", color=col, lw=2.2, ms=5, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="lower left", fontsize=9.5, frameon=False, handlelength=1.6)
ax.set_xlim(150, 16000)
ax.set_ylim(1.2, 300)
ax.set_xticks([512, 2048, 8192])
ax.set_xticklabels(["512", "2,048", "8,192"])
ax.set_yticks([5, 10, 30, 100])
ax.set_yticklabels(["5", "10", "30", "100"])
ax.minorticks_off()
ax.set_xlabel("accelerators (log)")
ax.set_ylabel("minutes to train (log)")
ax.text(15500, 250, "dashed: ideal from each series' first point;\n% = efficiency of the largest run",
        ha="right", va="top", fontsize=9.5, color=C["neutral"])
sp.save(fig, out / "mlperf_scaling.svg")

# ---------------------------------------------------------------------------
# 6. Straggler: chance that no GPU is late, 0.99^N (derived)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
N = np.logspace(0, 4, 200)
on_time = np.vectorize(lambda n: 100 * calc_all_up_availability(K.STRAG.availability, n))
ax.plot(N, on_time(N), color=V["bad"], lw=3)
for n_, lab, dy in [(num("gpus_per_node"), f"{val('gpus_per_node')} GPUs: {val('strag_node_pct')}%", 12),
                    (100, f"100: {val('strag_100_pct')}%", 12),
                    (num("strag_n"), f"{val('strag_n')}: {val('strag_1k_pct')}%", 14)]:
    v = on_time(n_)
    ax.plot([n_], [max(v, 0.4)], "o", ms=9, color=V["bad"], mec="white", mew=1.5)
    ax.annotate(lab, (n_, max(v, 0.4)), xytext=(8, dy), textcoords="offset points", fontsize=15,
                fontweight="bold", color=V["text"])
ax.set_xscale("log")
ax.set_xlim(1, 10000)
ax.set_ylim(0, 110)
ax.set_xlabel("GPUs in the synchronous group", fontsize=15)
ax.set_ylabel("% of steps with\nno late GPU", fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / "straggler.svg")

# ---------------------------------------------------------------------------
# 7. BSP, SSP, ASP timelines (after the chapter's figure 5), 3 steps
# ---------------------------------------------------------------------------
def sync_panel(s, x0, title, kind, op):
    unit = 24  # px per time unit
    s.text(x0 + 135, 22, title, 24, V["text"], op=op)
    for w in range(4):
        y = 50 + w * 52
        s.text(x0 - 2, y + 18, f"W{w}", 20, V["gray"], anchor="end", op=op)
        slow = w == 3
        t = 0.0
        it = 0
        while t < 10.0 and it < 6:
            dur = 2.6 if slow else 1.6
            if t + dur > 10.0:
                break
            s.rect(x0 + 8 + t * unit, y, dur * unit - 3, 36, V["bad_l"] if slow else V["c1_l"],
                   V["bad"] if slow else V["c1"], 2, 3, op)
            t += dur
            it += 1
            if kind == "bsp" and not slow:
                wait = 1.0
                s.rect(x0 + 8 + t * unit, y + 12, wait * unit - 3, 12, "#e2e8f0", "none", 0, 2, op)
                t += wait
            elif kind == "ssp" and not slow and it % 2 == 0:
                s.rect(x0 + 8 + t * unit, y + 12, 1.0 * unit - 3, 12, "#e2e8f0", "none", 0, 2, op)
                t += 1.0
    if kind == "bsp":
        for b in (2.6, 5.2, 7.8):
            s.line(x0 + 8 + b * unit, 44, x0 + 8 + b * unit, 252, V["c3"], 2.5, dash="6,4", op=op)
    s.text(x0 + 135, 278, {"bsp": "barrier every step", "ssp": "lead of at most s steps",
                           "asp": "no barrier; stale updates"}[kind], 20, V["gray"], weight="normal", op=op)


for k in (1, 2, 3):
    s = SVG(900, 295)
    for i, (x0, title, kind) in enumerate([(40, "BSP", "bsp"), (340, "SSP (s = 1)", "ssp"),
                                            (640, "ASP", "asp")]):
        sync_panel(s, x0, title, kind, 1.0 if i < k else 0.0)
    s.save(out / f"sync_step{k}.svg")

# ---------------------------------------------------------------------------
# 8. Critical batch: gradient-noise model (McCandlish et al. 2018), as in figure 10
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
r = np.logspace(-2, 2, 300)
ax.plot(r, 1 + 1 / r, color=V["c1"], lw=3)
ax.plot(r, 1 + r, color=V["bad"], lw=3)
ax.text(12, 1.4, "optimizer steps", color=V["c1"], fontsize=15, fontweight="bold")
ax.text(4, 25, "samples processed", color=V["bad"], fontsize=15, fontweight="bold", ha="right")
ax.axvline(1, color=V["gray"], lw=1.5, ls="--")
ax.plot([1], [2], "o", ms=10, color=V["text"])
ax.annotate(r"at $B = B_{\mathrm{crit}}$, both are" + "\n2× their minimum", (1, 2), xytext=(5, 2.3),
            fontsize=13, fontweight="bold", color=V["text"],
            arrowprops=dict(arrowstyle="->", color=V["text"], lw=1.2))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.01, 100)
ax.set_ylim(0.8, 120)
ax.set_xticks([0.01, 0.1, 1, 10, 100])
ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1", "10", "100"])
ax.minorticks_off()
ax.set_xlabel(r"batch $B$ / critical batch $B_{\mathrm{crit}}$", fontsize=15)
ax.set_ylabel("cost / minimum cost", fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / "critical_batch.svg")

# ---------------------------------------------------------------------------
# 9. Goyal et al. 2017 batch sweep (Figure 3 legends), real data
# ---------------------------------------------------------------------------
sweep = sorted((r["global_batch"], r["top1_error_pct"]) for r in K.GOYAL.entries if r["source"] == "figure3")
bs = [b for b, _ in sweep]
err = [e for _, e in sweep]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.axvspan(11000, 90000, color=C["danger_l"], zorder=0)
ax.plot(bs, err, "-o", color=C["memory"], lw=3, ms=8, zorder=3)
ax.annotate(f"{k8}: {val('goyal_err_warm')}%", (num("goyal_batch"), num("goyal_err_warm")), xytext=(-110, 22),
            textcoords="offset points", fontsize=15, fontweight="bold", color=C["memory"])
ax.annotate(f"{val('goyal_up3_k')}k: {val('goyal_up3_err')}%", (bs[-1], err[-1]), xytext=(-125, -6),
            textcoords="offset points", fontsize=15, fontweight="bold", color=C["danger"])
ax.text(14000, 31, f"past\n{k8}", fontsize=15, fontweight="bold", color=C["danger"])
ax.set_xscale("log", base=2)
ax.set_xticks([128, 512, 2048, 8192, 32768])
ax.set_xticklabels(["128", "512", "2k", "8k", "32k"])
ax.minorticks_off()
ax.set_ylim(22, 36)
ax.set_xlabel("global batch (images)", fontsize=15)
ax.set_ylabel("top-1 val. error (%)", fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / "goyal_batch.svg")

# ---------------------------------------------------------------------------
# 10. Memory wall (Epoch AI notable models x 16 B/param; vendor capacities)
# ---------------------------------------------------------------------------
# The scenario's models, with the deck's shorter labels; ResNet-152 is not drawn here.
deck_label = {"AlexNet": "AlexNet", "VGG-16": "", "BERT-Large": "", "GPT-2": "GPT-2", "GPT-3": "GPT-3",
              "PaLM": "PaLM", "Llama 3 (405B)": "Llama 3 405B"}
wall_rows = K.WALL.result.rows
mx, my, ml = [], [], []
for r_ in wall_rows:
    if r_["Type"] == "Model" and r_["Name"] in deck_label:
        d = r_["Date"]
        mx.append(int(d[:4]) + (int(d[5:7]) - 1) / 12)
        my.append(r_["Memory_GB"])
        ml.append(deck_label[r_["Name"]])
gx, gy, gl = [], [], []
for r_ in wall_rows:
    if r_["Type"] == "GPU":
        gx.append(r_["Year"])
        gy.append(r_["Memory_GB"])
        gl.append(r_["Name"])
gpt3 = next(r_ for r_ in wall_rows if r_["Name"] == "GPT-3")
fig, ax = plt.subplots(figsize=(5.8, 2.35))
ax.plot(gx, gy, "--s", color=C["memory"], lw=2.5, ms=7)
ax.plot(mx, my, "-o", color=C["danger"], lw=2.5, ms=7)
for x, y, lab in zip(gx, gy, gl):
    ax.annotate(lab, (x, y), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=10.5,
                fontweight="bold", color=C["memory"])
offs = {"AlexNet": (4, 8), "": (0, 0), "GPT-2": (8, -12),
        "GPT-3": (-10, 8), "PaLM": (0, 9), "Llama 3 405B": (0, -18)}
for x, y, lab in zip(mx, my, ml):
    ax.annotate(lab, (x, y), xytext=offs[lab], textcoords="offset points",
                ha="right" if lab == "GPT-3" else ("left" if lab == "GPT-2" else "center"),
                fontsize=10.5, fontweight="bold", color=C["danger"])
ax.annotate(f"GPT-3: {val('state_tb')} TB of state,\n{val('wall_gap')}× one {val('a100_gb')} GB GPU",
            (2020.4, gpt3["Memory_GB"]), xytext=(2013.6, 1500),
            fontsize=11, fontweight="bold", color=C["text"],
            arrowprops=dict(arrowstyle="->", color=C["text"], lw=1.2))
ax.text(2012.3, 70, "one accelerator's memory", fontsize=11, fontweight="bold", color=C["memory"])
ax.text(2015.6, 0.35, f"training state, {val('state_bpp')} B per parameter", fontsize=11, fontweight="bold",
        color=C["danger"])
ax.set_yscale("log")
ax.set_ylim(0.2, 30000)
ax.set_xlim(2011.8, 2025.2)
ax.set_ylabel("GB (log)")
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
ax.minorticks_off()
sp.save(fig, out / "memory_wall.svg")

# ---------------------------------------------------------------------------
# 11. ZeRO stages, 7B model on 64 GPUs: per-GPU state (4 steps)
# ---------------------------------------------------------------------------
z7 = K.ZERO7.result
Ng = K.ZERO7.n_devices
W, G, O = (q.to("GB").magnitude for q in (z7.weights, z7.gradients, z7.optimizer_state))
stage_key = {"DDP": "z7_ddp_gb", "ZeRO-1": "z7_z1_gb", "ZeRO-2": "z7_z2_gb", "ZeRO-3": "z7_z3_gb"}
stages = [("DDP", W, G, O), ("ZeRO-1", W, G, O / Ng), ("ZeRO-2", W, G / Ng, O / Ng),
          ("ZeRO-3", W / Ng, G / Ng, O / Ng)]
cap = num("a100_gb")
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    for i, (name, wv, gv, ov) in enumerate(stages):
        a = 1 if i < k else 0
        traffic = f"{z7.traffic_multiple[name]:g}×"
        ax.bar(i, wv, 0.58, color=C["memory"], alpha=a)
        ax.bar(i, gv, 0.58, bottom=wv, color=C["network"], alpha=a)
        ax.bar(i, ov, 0.58, bottom=wv + gv, color=C["compute"], alpha=a)
        tot = wv + gv + ov
        label(ax, i, tot + 4, f"{val(stage_key[name])} GB",
              C["danger"] if tot > cap else C["data"], i < k, ha="center", fontsize=13)
        label(ax, i, -26, f"traffic {traffic}", C["neutral"], i < k, ha="center", fontsize=11)
    ax.axhline(cap, color=C["danger"], lw=2, ls="--")
    ax.text(3.45, cap + 2, f"{val('a100_gb')} GB GPU", color=C["danger"], fontsize=11, fontweight="bold", ha="right",
            va="bottom")
    ax.text(1.55, 120, "weights", color=C["memory"], fontsize=11, fontweight="bold", ha="left")
    ax.text(1.55, 108, "gradients", color=C["network"], fontsize=11, fontweight="bold", ha="left")
    ax.text(1.55, 96, "optimizer (FP32)", color=C["compute"], fontsize=11, fontweight="bold", ha="left")
    ax.set_xticks(range(4), [s_[0] for s_ in stages], fontsize=12)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 128)
    ax.set_ylabel("GB per GPU")
    ax.tick_params(axis="x", length=0, pad=4)
    sp.save(fig, out / f"zero_step{k}.svg")

# ---------------------------------------------------------------------------
# 12. Pipeline: naive layer split (step 1), then GPipe p = m = 4 (step 2)
# ---------------------------------------------------------------------------
def pipe_grid(s, y0, slots, title, op, cw=46):
    s.text(20, y0 - 18, title, 23, V["text"], anchor="start", op=op)
    for st in range(4):
        y = y0 + st * 34
        s.text(62, y + 16, f"GPU {st}", 20, V["gray"], anchor="end", op=op)
        for t in range(len(slots[st])):
            x = 70 + t * cw
            cell = slots[st][t]
            if cell is None:
                s.rect(x, y, cw - 4, 30, V["bad_l"], "none", 0, 3, op)
            else:
                fwd = cell[0] == "F"
                s.rect(x, y, cw - 4, 30, V["c1_l"] if fwd else V["c2_l"],
                       V["c1"] if fwd else V["c2"], 2, 3, op)
                s.text(x + (cw - 4) / 2, y + 16, cell, 20, V["c1"] if fwd else "#b45309", op=op)


def naive(p=K.NAIVE.stages):
    T = K.NAIVE.slots
    g = [[None] * T for _ in range(p)]
    for st in range(p):
        g[st][st] = "F"
        g[st][T - 1 - st] = "B"
    return g


def gpipe(p=K.GPIPE.stages, m=K.GPIPE.microbatches):
    T = 2 * (m + p - 1)
    g = [[None] * T for _ in range(p)]
    for st in range(p):
        for i in range(m):
            g[st][st + i] = f"F{i + 1}"
            g[st][(m + p - 1) + (p - 1 - st) + i] = f"B{m - i}"
    return g


for k in (1, 2):
    s = SVG(900, 345)
    pipe_grid(s, 30, naive(), f"One batch through {val('pipe_p')} stages: each GPU idles "
              f"{val('naive_idle')} of {val('naive_slots')} slots", 1.0)
    pipe_grid(s, 210, gpipe(), f"Four microbatches (GPipe): each GPU idles {val('pipe_idle')} of "
              f"{val('pipe_slots')} slots, {val('pipe_bubble_pct')}%", 1.0 if k >= 2 else 0.0, cw=55)
    s.save(out / f"pipe_step{k}.svg")

# ---------------------------------------------------------------------------
# 13. Bubble fraction against microbatch count (2 steps: p=4, 8; then p=16)
# ---------------------------------------------------------------------------
m = np.arange(1, 257)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    curves = [(K.GPIPE.stages, C["memory"], True, 90), (K.P8M32.stages, C["compute"], True, 79),
              (K.P16M32.stages, C["network"], k >= 2, 68)]
    for p, col, show, ly in curves:
        f = np.array([calc_pipeline_bubble(p, int(mm)) for mm in m])
        ax.plot(m, 100 * f, color=col, lw=2.6, alpha=1 if show else 0)
        label(ax, 240, ly, f"p = {p} stages", col, show, fontsize=15, ha="right")
    pts = [(K.GPIPE, C["memory"], f"p={val('pipe_p')}, m={val('pipe_p')}: {val('pipe_bubble_pct')}%", None, True),
           (K.P8M32, C["compute"], f"p={val('p8m32_p')}, m={val('p8m32_m')}: {val('p8m32_bubble_pct')}%",
            (-10, -22), True),
           (K.P16M32, C["network"], f"m={val('p16m32_m')}: {val('p16m32_bubble_round_pct')}%", (8, 8), k >= 2)]
    for sc, col, lab, off, show in pts:
        p, mm, v = sc.stages, sc.microbatches, 100 * sc.bubble
        ax.plot([mm], [v], "o", ms=9, color=col, mec="white", mew=1.5, alpha=1 if show else 0)
        if p in (4, 8):
            ax.annotate(lab, (mm, v), xytext=(1.15, 19 if p == 4 else 3), fontsize=15, fontweight="bold",
                        color=col, arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
            continue
        ax.annotate(lab, (mm, v), xytext=off, textcoords="offset points", fontsize=15,
                    fontweight="bold", color=col, alpha=1 if show else 0)
    ax.axhline(num("bubble_target_pct"), color=C["neutral"], lw=1.4, ls="--")
    ax.text(1.05, num("bubble_target_pct") + 2, f"{val('bubble_target_pct')}%", color=C["neutral"], fontsize=14,
            fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256])
    ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlim(1, 256)
    ax.set_xlabel("microbatches per step, m", fontsize=15)
    ax.set_ylabel("idle fraction (%)", fontsize=15)
    ax.tick_params(labelsize=14)
    sp.save(fig, out / f"bubble_step{k}.svg")

# ---------------------------------------------------------------------------
# 14. Megatron column-then-row split on two GPUs (3 steps)
# ---------------------------------------------------------------------------
for k in (1, 2, 3):
    s = SVG(900, 300)
    for r_, gpu in enumerate((0, 1)):
        y = 40 + r_ * 120
        s.text(40, y + 30, f"GPU {gpu}", 23, V["gray"], anchor="start")
        s.rect(130, y + 8, 110, 46, V["ok_l"], V["ok"])
        s.text(185, y + 31, "X (copy)", 22, V["ok"])
        s.line(240, y + 31, 278, y + 31, V["gray"], 3, arrow=True)
        s.rect(280, y + 8, 110, 46, V["c1_l"], V["c1"])
        s.text(335, y + 31, f"A{gpu + 1}: cols", 22, V["c1"])
        s.line(390, y + 31, 428, y + 31, V["gray"], 3, arrow=True)
        s.rect(430, y + 8, 90, 46, V["c1_l"], V["c1"])
        s.text(475, y + 31, f"Y{gpu + 1}", 23, V["c1"])
        op2 = 1 if k >= 2 else 0
        s.line(520, y + 31, 558, y + 31, V["gray"], 3, arrow=True, op=op2)
        s.rect(560, y + 8, 110, 46, V["c1_l"], V["c1"], op=op2)
        s.text(615, y + 31, f"B{gpu + 1}: rows", 22, V["c1"], op=op2)
        s.line(670, y + 31, 698, y + 31, V["gray"], 3, arrow=True, op=op2)
        s.rect(700, y + 8, 70, 46, "#f1f5f9", V["gray"], op=op2)
        s.text(735, y + 31, f"Z{gpu + 1}", 23, V["text"], op=op2)
    s.text(475, 132, "no exchange: Y stays local", 21, V["c1"], weight="normal")
    op2 = 1 if k >= 2 else 0
    s.text(700, 132, "partial sums", 21, V["gray"], weight="normal", op=op2)
    op3 = 1 if k >= 3 else 0
    s.rect(790, 40, 90, 206, V["c2_l"], V["c2"], op=op3)
    s.text(835, 128, "All-", 23, "#b45309", op=op3)
    s.text(835, 154, "Reduce", 23, "#b45309", op=op3)
    s.line(770, 71, 788, 71, V["c2"], 3, arrow=True, op=op3)
    s.line(770, 191, 788, 191, V["c2"], 3, arrow=True, op=op3)
    s.text(835, 274, "Z = Z1 + Z2", 23, V["text"], op=op3)
    s.text(30, 274, "Megatron: QKV, then attention out; FFN-1, then FFN-2", 21, V["gray"], weight="normal",
           anchor="start", op=op3)
    s.save(out / f"tp_step{k}.svg")

# ---------------------------------------------------------------------------
# 15. Tensor-parallel feasibility per layer (column figure)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4))
bars = [("compute", "tp_compute_ms", V["c1"]), (f"TP over NVLink\n{val('nvlink_gbs')} GB/s", "tp_nvlink_ms", V["ok"]),
        (f"TP over one NDR\nport, {val('ndr_gbs')} GB/s", "tp_ndr_ms", V["bad"])]
for i, (n_, key, col) in enumerate(bars):
    y = 2 - i
    v = num(key)
    ax.barh(y, v, color=col, height=0.6)
    ax.text(-0.08, y, n_, ha="right", va="center", fontsize=15, fontweight="bold")
    ax.text(v + 0.06, y, f"{val(key)} ms", va="center", fontsize=16, fontweight="bold", color=col)
ax.axvline(num("tp_compute_ms"), color=V["c1"], lw=1.5, ls="--")
ax.set_xlim(0, 4.6)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("ms per GPT-3 layer, forward", fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / "tp_tiers.svg")

# ---------------------------------------------------------------------------
# 16. Megatron-LM weak scaling (Shoeybi et al. 2019, Figure 5)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 1.95))
meg = sorted(K.MEGATRON.entries, key=lambda r: (r["mode"] != "model", r["gpus"]))
xs_ = [f"{r['gpus']:.0f}" for r in meg]
vals = [r["efficiency_pct"] for r in meg]
cols = [C["memory"] if r["mode"] == "model" else C["compute"] for r in meg]
for i, (x_, v, c_) in enumerate(zip(xs_, vals, cols)):
    xi = i + (0.6 if i >= 4 else 0)
    ax.bar(xi, v, 0.62, color=c_)
    ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold", color=c_)
ax.set_xticks([i + (0.6 if i >= 4 else 0) for i in range(8)], xs_)
ax.text(1.5, 122, "tensor parallel, one server", ha="center", fontsize=11.5, fontweight="bold",
        color=C["memory"])
ax.text(6.1, 122, f"{max(r['gpus'] for r in meg if r['mode'] == 'model'):.0f}-way TP × data parallel", ha="center", fontsize=11.5, fontweight="bold",
        color=C["compute"])
ax.set_ylim(0, 130)
ax.set_yticks([0, 50, 100])
ax.set_xlabel("GPUs (V100, DGX-2H)")
ax.set_ylabel("weak-scaling\nefficiency (%)")
sp.save(fig, out / "megatron_scaling.svg")

# ---------------------------------------------------------------------------
# 17. Hybrid scoreboard for the 175B job on 1,024 GPUs (3 steps)
# ---------------------------------------------------------------------------
cfg = [(f"A: DP-{val('dp_a')}", "eff_a_pct", "state_gb", None),
       (f"B: TP{val('tp_b')} × DP{val('dp_b')}", "eff_b_pct", "state_b_gb", None),
       (f"C: TP{val('tp_b')} × PP{val('pp_c')} × DP{val('dp_c')}", "eff_c_pct", "state_c_gb", "zero1_c_gb")]
cap = num("hbm_gb")
for k in (1, 2, 3):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.9, 2.15), gridspec_kw={"width_ratios": [1, 1]})
    for i, (name, eff_key, mem_key, z1_key) in enumerate(cfg):
        y = 2 - i
        sh = i < k
        eff, mem = num(eff_key), num(mem_key)
        z1 = num(z1_key) if z1_key else None
        a1.barh(y, eff, color=C["data"] if eff > 50 else C["danger"], height=0.6, alpha=1 if sh else 0)
        a1.text(-3, y, name, ha="right", va="center", fontsize=11.5, fontweight="bold")
        label(a1, eff + 2, y, f"{val(eff_key)}%", C["data"] if eff > 50 else C["danger"], sh, va="center",
              fontsize=12.5)
        a2.barh(y, mem, color=C["danger"] if mem > cap else C["data"], height=0.6, alpha=1 if sh else 0)
        label(a2, mem * 1.15, y, f"{val(mem_key)} GB", C["danger"], sh, va="center", fontsize=12.5)
        if z1:
            a2.barh(y - 0.36, z1, color=C["data"], height=0.28, alpha=1 if sh else 0)
            label(a2, 160, y - 0.36, f"with ZeRO-1: {val(z1_key)} GB", C["data"], sh, va="center", fontsize=11)
    a1.set_xlim(0, 110)
    a1.set_ylim(-0.7, 2.6)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("scaling efficiency (%)")
    a2.set_xscale("log")
    a2.set_xlim(10, 20000)
    a2.set_ylim(-0.7, 2.6)
    a2.axvline(cap, color=C["danger"], lw=1.6, ls="--")
    a2.text(cap, 2.55, f" {val('hbm_gb')} GB", color=C["danger"], fontsize=10.5, fontweight="bold", va="bottom")
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xticks([10, 100, 1000, 10000])
    a2.set_xticklabels(["10", "100", "1k", "10k"])
    a2.minorticks_off()
    a2.set_xlabel("training state per GPU (GB, log)")
    fig.subplots_adjust(wspace=0.08)
    sp.save(fig, out / f"hybrid_step{k}.svg")

# ---------------------------------------------------------------------------
# 18. One 3D layout on the tiers: TP8 x PP4 x DP32 on 1,024 GPUs
# ---------------------------------------------------------------------------
s = SVG(900, 330)
for rrow, (y, lab) in enumerate([(30, "replica 1"), (130, "replica 2")]):
    s.text(18, y + 40, lab, 21, V["gray"], anchor="start")
    for n_ in range(K.C.pp_size):
        x = 120 + n_ * 190
        s.rect(x, y, 160, 80, V["c1_l"], V["c1"], 2.5, 8)
        for gi in range(K.C.tp_size):
            gx_ = x + 10 + gi * 18.4
            s.rect(gx_, y + 10, 14, 26, "white", V["c1"], 1.5, 2)
        s.text(x + 80, y + 60, f"stage {n_ + 1}", 23, V["c1"])
        if n_ < K.C.pp_size - 1:
            s.line(x + 162, y + 40, x + 186, y + 40, V["c3"], 3.5, arrow=True)
s.text(450, 228, f"... {val('dp_c')} replicas in all", 22, V["gray"])
for n_ in range(K.C.pp_size):
    x = 120 + n_ * 190 + 80
    s.line(x, 112, x, 128, V["c2"], 3.5, arrow=True)
s.rect(20, 250, 280, 66, V["c1_l"], V["c1"], 2, 6)
s.text(160, 270, f"TP = {val('tp_b')} inside a node", 23, V["c1"])
s.text(160, 296, f"NVLink, {val('nvlink_gbs')} GB/s", 21, V["c1"], weight="normal")
s.rect(310, 250, 280, 66, V["c3_l"], V["c3"], 2, 6)
s.text(450, 270, f"PP = {val('pp_c')} across nodes", 23, "#6d28d9")
s.text(450, 296, "activations, point to point", 21, "#6d28d9", weight="normal")
s.rect(600, 250, 280, 66, V["c2_l"], V["c2"], 2, 6)
s.text(740, 270, f"DP = {val('dp_c')} across replicas", 23, "#b45309")
s.text(740, 296, f"{val('shard_c_gb')} GB gradient shard", 21, "#b45309", weight="normal")
s.save(out / "grid3d.svg")

print("figures written to", out)

# ---------------------------------------------------------------------------
# PPO against DPO: static state per model, stacked (frame "PPO alignment
# holds four models"). Values from calcs.py (Distributed Training, table 7).
# Added 2026-10-21 so the footprints compare at a glance.
# ---------------------------------------------------------------------------
_pol, _ref = num("rlhf_policy_gb"), num("rlhf_ref_gb")
_rew, _valg = num("rlhf_reward_gb"), num("rlhf_value_gb")
fig, ax = plt.subplots(figsize=(5.8, 1.1))
for y, parts, tot in ((1, [(_pol, C["memory"], "policy"), (_ref, C["neutral"], "ref."),
                           (_rew, C["data"], ""), (_valg, C["compute"], "value")],
                       f"PPO {val('rlhf_ppo_gb')} GB"),
                      (0, [(_pol, C["memory"], "policy"), (_ref, C["neutral"], "ref.")],
                       f"DPO {val('rlhf_dpo_gb')} GB")):
    left = 0
    for w, col, lab in parts:
        ax.barh(y, w, 0.62, left=left, color=col, edgecolor="white", lw=1)
        if lab:
            ax.text(left + w / 2, y, lab, ha="center", va="center", fontsize=11.5,
                    fontweight="bold", color="white")
        left += w
    ax.text(left + 15, y, tot, va="center", fontsize=12.5, fontweight="bold", color=C["text"])
_x = _pol + _ref + _rew / 2
ax.annotate("reward", (_x, 1.31), xytext=(_x - 25, 1.62), ha="right", va="center",
            fontsize=11, fontweight="bold", color=C["data"],
            arrowprops=dict(arrowstyle="-", color=C["data"], lw=1.2))
ax.set_xlim(0, (_pol + _ref + _rew + _valg) * 1.32)
ax.set_ylim(-0.4, 1.8)
ax.axis("off")
sp.save(fig, out / "rlhf_state.svg")
