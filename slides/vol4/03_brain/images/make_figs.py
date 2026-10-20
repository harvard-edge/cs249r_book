#!/usr/bin/env python3
"""Figures for Lecture 3 (Volume IV, The Cognitive Brain).

Every number drawn here comes from calcs.py, which reads MLSysIM (the
Embodied.* machine and site, Literature.*, and the Scenarios.* the chapter also
reads): plotted geometry uses the scenario values (namespace K below), and
labels use the formatted strings the slides print (deckvals val). Axis ranges,
tick positions, and label coordinates are layout. Figures are drawn at the size of their slot on the 16 x 9 cm
page and saved at that size (save_slot), so the slide places them at scale 1
and every label keeps its drawn size (8 to 9.5 pt).

  - MLPerf Inference edge, single stream, p90, Jetson AGX Orin 64G (TensorRT),
    latest round per model: Scenarios.OrinDevKitSingleStream, as the chapter's
    figure 9 reads it.
  - Memory wall, roofline, evaluation bars, decoder ladder:
    Scenarios.IntentModelEvaluation*, IntentModelDecodeRoofline,
    IntentBackboneDecoderFamilies, WarehouseProposerTraffic.
  - Published two-speed rates: Literature.VlaSystems (Figure AI 2025; Bjorck et al. 2025).
  - Evidence horizon and timing strip: Scenarios.ConveyorMugEvidenceHorizon.
  - KV ladder: Scenarios.ThreeViewKvHistory. Confidence and success:
    Scenarios.ProposerConfidenceRegimes (stipulated, illustrative).
  - Volume IV schematic colors (palette.md): cognition navy #1F407A, proposal
    blue #2563EB (dashed), permission green #059669, veto red #DC2626, plant
    slate #334155. Resource plots keep the base contract (memory blue,
    compute orange).

Run from this folder: ../../../../.venv/bin/python make_figs.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import calcs as c  # noqa: E402  every value comes from calcs.py (MLSysIM)
from deckvals import val  # noqa: E402
from types import SimpleNamespace  # noqa: E402

m = c.m
_f16, _i4, _tr, _fam, _ev = c.f16, c.i4, c.tr, c.fam, c.ev
_rm = c.RM
# Plotted magnitudes, in the units the figure axes use (ms, Hz, bytes, mm, GB, MB).
K = SimpleNamespace(
    MLPERF=[(task, model, c.lat[b], c.MLPERF.latest_round(b)) for b, task, model in c.MLPERF.benchmarks],
    T_CHUNK=m(_rm.chunk_period, "ms"), T_INTENT=m(1 / _rm.intent_rate, "ms"),
    BW_SUS=m(_tr.sustained_bandwidth, "byte/s"), BW_PEAK=m(_rm.brain_soc.memory.bandwidth, "byte/s"),
    PEAK_FP16=m(_rm.brain_soc.compute.precision_flops["fp16"], "flop/s"),
    BPP16=m(c.FP16.bytes_per_param, "byte/param"), BPP4=m(c.INT4.bytes_per_param, "byte/param"),
    TICK_RATE=m(_rm.permission_rate, "Hz"), CHUNK_RATE=m(_rm.chunk_rate, "Hz"), INTENT_RATE=m(_rm.intent_rate, "Hz"),
    CHUNK_PARAMS=m(_rm.chunk_model.parameters, "param"), N_PARAMS=m(_rm.intent_model.parameters, "param"),
    RIDGE=m(_f16.ridge_intensity, "flop/byte"), DECODE_I=m(_f16.decode_intensity, "flop/byte"),
    PREFILL_I=m(_f16.prefill_intensity, "flop/byte"), S_IN=c.INT4.tokens_in,
    DEC16=m(_f16.t_decode, "ms"), DEC4=m(_i4.t_decode, "ms"),
    T_FP16=m(_f16.t_total, "ms"), T_INT4=m(_i4.t_total, "ms"), T_PREFILL=m(_i4.t_prefill, "ms"),
    DEMAND_SINGLE=m(_tr.single_step_demand, "byte/s"), DEMAND_CHUNK=m(_tr.chunked_demand, "byte/s"),
    H=_rm.chunk_horizon, CHUNK_INFER=m(_rm.chunk_inference_latency, "ms"), CHUNK_STEP=m(_rm.chunk_step, "ms"),
    T_BINNED=m(_fam.t_binned, "ms"), T_FAST=m(_fam.t_fast, "ms"), T_PAR=m(_fam.t_parallel, "ms"),
    T_EXPERT=m(_fam.t_expert, "ms"), T_PRE_ONLY=m(_fam.t_prefill, "ms"), N_BINNED=_fam.binned_tokens,
    DUAL=c.DUAL,
    MARGIN=m(_ev.margin, "mm"), V_CONV=m(c.AISLE.v_conveyor, "m/s"), A_SLIP=m(c.AISLE.a_conveyor_slip, "m/s**2"),
    T_UNTRACKED=m(_ev.untracked_deadline, "ms"), TAU_EV=m(_ev.horizon, "ms"), REACT=m(_ev.age_at_replacement, "ms"),
    DRIFT_90=m(_ev.drift_at_replacement, "mm"), SPAN=m(_ev.chunk_span, "ms"), DRIFT_SPAN=m(_ev.drift_at_chunk_end, "mm"),
    SOC_MEM=m(c.KV.module_memory, "GB"), KV_5S=m(c.KV.history, "GB"), W_FP16=m(_f16.weight_memory, "GB"),
    KV_STEP=m(c.KV.kv_per_step, "MB"), KV_TOKEN=m(c.TOKENS.kv_per_token, "GB"),
    REGIMES=c.REGIMES.regimes,
)

C = sp.C
plt = sp.plt
out = Path(__file__).parent

NAVY, PROP, GREEN, RED, SLATE, AMBER = "#1F407A", "#2563EB", "#059669", "#DC2626", "#334155", "#D97706"
FS, TFS, SFS = 9, 8.5, 8   # label, tick, small tag sizes in pt (page size = drawn size)


def cm(w, h):
    return (w / 2.54, h / 2.54)


def save_slot(fig, path):
    """Save at exactly the figure size, so the slide places it at scale 1."""
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)


def tag(ax, text, x=0.0, y=-0.30, fs=SFS):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, color=C["neutral"], ha="left", va="top")


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def ms_label(v):
    if v < 10:
        return f"{v:.2f} ms"
    if v < 100:
        return f"{v:.1f} ms"
    if v < 1000:
        return f"{v:.0f} ms"
    return f"{v / 1000:.1f} s"


# ---------------------------------------------------------------- MLPerf on Orin
def mlperf_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.9))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.90, bottom=0.20)
    rows = K.MLPERF
    for i, (task, model, v, rnd) in enumerate(rows):
        col = C["memory"] if v <= K.T_CHUNK else (C["compute"] if v <= K.T_INTENT else C["neutral"])
        shown = k >= 2
        ax.barh(i, v, color=col, height=0.62, alpha=1.0 if shown else 0.0)
        ax.text(v * 1.18, i, ms_label(v), va="center", fontsize=FS, color=col, zorder=6,
                fontweight="bold", alpha=1.0 if shown else 0.0,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=1.0 if shown else 0.0))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} ({r[1]})" for r in rows], fontsize=FS)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3e5)
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    ax.tick_params(axis="y", length=0)
    for t, lab, ha, dx in ((K.T_CHUNK, f"chunk period {val('t_chunk')} ms", "right", 0.9),
                           (K.T_INTENT, f"intent period {val('t_intent')} ms", "left", 1.1)):
        ax.axvline(t, color=C["danger"], ls="--", lw=1.5, zorder=1)
        ax.text(t * dx, -0.85, lab, fontsize=FS, color=C["danger"], ha=ha, va="center", fontweight="bold")
    ax.set_ylim(len(rows) - 0.5, -1.25)
    ax.set_xlabel("90th-percentile latency of one query (ms, log scale)", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"mlperf_step{k}.svg")


for k in (1, 2):
    mlperf_fig(k)


# ---------------------------------------------------------------- cadence plot
def cadence_fig():
    fig, ax = plt.subplots(figsize=cm(14.2, 4.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.21)
    n = np.logspace(7, 11, 200)
    for bpp, lab, ls in ((K.BPP16, "FP16 ceiling", "-"), (K.BPP4, "INT4 ceiling", "--")):
        ax.plot(n, K.BW_SUS / (n * bpp), color=C["memory"], lw=2, ls=ls)
    ax.text(3e7, K.BW_SUS / (3e7 * K.BPP16) * 0.55, "FP16 ceiling", color=C["memory"], fontsize=FS,
            fontweight="bold", rotation=-17, ha="left", va="top")
    ax.text(4e9, K.BW_SUS / (4e9 * K.BPP4) * 1.35, "INT4 ceiling", color=C["memory"], fontsize=FS,
            fontweight="bold", rotation=-17, ha="left", va="bottom")
    for r, lab, x, ha in ((K.TICK_RATE, f"permission tick {val('tick_khz')} kHz", 1.1e11, "right"),
                          (K.CHUNK_RATE, f"chunk renewal {val('chunk_hz')} Hz", 1.1e7, "left"),
                          (K.INTENT_RATE, f"intent revision {val('intent_hz')} Hz", 1.1e7, "left")):
        ax.axhline(r, color=C["neutral"], lw=1, ls=":")
        ax.text(x, r * 1.12, lab, fontsize=SFS, color=C["neutral"], ha=ha, va="bottom")
    pts = [(K.CHUNK_PARAMS, K.CHUNK_RATE, f"chunk policy {val('chunk_params_m')}M", (6, 4), "left"),
           (K.N_PARAMS, K.INTENT_RATE, f"intent model {val('params_b')}B", (-6, 6), "right")]
    for x, y, lab, off, ha in pts:
        ax.plot([x], [y], "o", ms=8, color=NAVY, mec="white", mew=1.2, zorder=5)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=FS,
                    fontweight="bold", color=NAVY, ha=ha, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e7, 1.2e11)
    ax.set_ylim(0.5, 3e4)
    ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11])
    ax.set_xticklabels(["10M", "100M", "1B", "10B", "100B"])
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.tick_params(labelsize=TFS, pad=1)
    ax.set_xlabel("Parameters", fontsize=FS, labelpad=1)
    ax.set_ylabel("Fresh evaluations per s", fontsize=FS, labelpad=1)
    save_slot(fig, out / "cadence.svg")


cadence_fig()


# ---------------------------------------------------------------- roofline
def roofline_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.8))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.95, bottom=0.21)
    peak = K.PEAK_FP16 / 1e12
    bw = K.BW_PEAK / 1e12
    xs = np.logspace(-1, 4, 300)
    roof = np.minimum(peak, bw * xs)
    m = xs <= K.RIDGE
    ax.plot(xs[m], roof[m], color=C["memory"], lw=2.5)
    ax.plot(xs[~m], roof[~m], color=C["compute"], lw=2.5)
    ax.fill_between(xs[m], 1e-3, roof[m], color=C["memory_l"], alpha=0.6)
    ax.fill_between(xs[~m], 1e-3, roof[~m], color=C["compute_l"], alpha=0.6)
    ax.axvline(K.RIDGE, color=C["neutral"], lw=1, ls="--")
    ax.text(K.RIDGE * 1.1, 0.012, f"ridge {val('ridge')} FLOP/byte", fontsize=SFS, color=C["neutral"])
    ax.text(0.15, peak * 1.25, f"dense FP16 peak {val('peak_fp16')} TFLOP/s", fontsize=SFS, color=C["compute"])
    ax.text(0.9, 0.012, f"slope: {val('bw_peak')} GB/s", fontsize=SFS, color=C["memory"])
    if k >= 2:
        x, y = K.DECODE_I, bw * K.DECODE_I
        ax.plot([x], [y], "o", ms=9, color=C["memory"], mec="white", mew=1.4, zorder=5)
        ax.annotate(f"decode, one token: {val('decode_i')} FLOP/byte\n{val('decode_tflops')} TFLOP/s, "
                    f"{val('decode_pct')}% of peak", (x, y), xytext=(0.13, 2.2),
                    textcoords="data", ha="left", va="bottom", fontsize=FS, color=C["memory"],
                    fontweight="bold", arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1))
    if k >= 3:
        x, y = K.PREFILL_I, peak
        ax.plot([x], [y], "o", ms=9, color=C["compute"], mec="white", mew=1.4, zorder=5)
        ax.annotate(f"prefill, {val('s_in')} tokens:\n{val('prefill_i')} FLOP/byte", (x, y), xytext=(10, -26),
                    textcoords="offset points", ha="left", fontsize=FS, color=C["compute"],
                    fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(0.008, 150)
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.tick_params(labelsize=TFS, pad=1)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte from off-chip memory)", fontsize=FS, labelpad=1)
    ax.set_ylabel("TFLOP/s", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"roofline_step{k}.svg")


for k in (1, 2, 3):
    roofline_fig(k)


# ---------------------------------------------------------------- one evaluation, FP16 vs INT4
def eval_fig(k, size=(14.2, 2.6), name="eval"):
    fig, ax = plt.subplots(figsize=cm(*size))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.36)
    rows = [("FP16", K.DEC16, K.T_FP16, val("decode16"), val("t_fp16")),
            ("INT4", K.DEC4, K.T_INT4, val("decode4"), val("t_int4"))]
    for i, (lab, dec, tot, dec_txt, tot_txt) in enumerate(rows):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.barh(i, K.T_PREFILL, color=C["compute"], height=0.6, alpha=a)
        ax.barh(i, dec, left=K.T_PREFILL, color=C["memory"], height=0.6, alpha=a)
        ax.text(K.T_PREFILL / 2, i, f"prefill {val('prefill')}", ha="center", va="center",
                color="white", fontsize=FS, fontweight="bold", alpha=a)
        ax.text(K.T_PREFILL + dec / 2, i, f"decode {dec_txt}", ha="center", va="center",
                color="white", fontsize=FS, fontweight="bold", alpha=a)
        ax.text(tot + 12, i, f"{tot_txt} ms", va="center", fontsize=FS, fontweight="bold",
                color=C["text"], alpha=a)
    ax.axvline(K.T_INTENT, color=C["danger"], ls="--", lw=1.5)
    ax.text(K.T_INTENT + 8, 1.55, f"{val('t_intent')} ms period ({val('intent_hz')} Hz)", color=C["danger"],
            fontsize=SFS, fontweight="bold", va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["FP16", "INT4"], fontsize=FS, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_ylim(1.8, -0.5)
    ax.set_xlim(0, 900)
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    ax.set_xlabel("ms for one fresh evaluation (computed, Napkin Math 3.1)", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"{name}_step{k}.svg")


for k in (1, 2):
    eval_fig(k)


# ---------------------------------------------------------------- chunk supply
def supply_fig(k):
    fig, ax = plt.subplots(figsize=cm(8.4, 3.7))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.82, bottom=0.30)
    rows = [("one setpoint\nper pass", K.DEMAND_SINGLE / 1e9, C["danger"], val("demand")),
            (f"chunk of {val('horizon')}", K.DEMAND_CHUNK / 1e9, C["data"], val("chunked"))]
    for i, (lab, v, col, txt) in enumerate(rows):
        a = 1.0 if i < k else 0.0
        ax.barh(i, v, color=col, height=0.6, alpha=a)
        ax.text(v + 8, i, f"{txt} GB/s", va="center", fontsize=FS, fontweight="bold", color=col, alpha=a)
    ax.axvline(K.BW_SUS / 1e9, color=C["memory"], ls="--", lw=1.5)
    ax.text(K.BW_SUS / 1e9, -0.62, f"sustained {val('b_sus')} GB/s", color=C["memory"],
            fontsize=SFS, fontweight="bold", ha="center", va="bottom")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_ylim(1.5, -0.65)
    ax.set_xlim(0, 700)
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    ax.set_xlabel(f"FP16 weight traffic at {val('setpoint_hz')} setpoints/s", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"supply_step{k}.svg")


for k in (1, 2):
    supply_fig(k)


# ---------------------------------------------------------------- decoder ladder
def decoder_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.4))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.22)
    rows = [("Binned tokens", K.T_BINNED, f"{val('binned_tok')} sequential", f"{val('t_binned_s')} s"),
            ("FAST tokens", K.T_FAST, f"{val('fast_tok')} sequential", f"{val('t_fast')} ms"),
            ("Parallel head", K.T_PAR, "one longer prefill", f"{val('t_par')} ms"),
            ("Flow expert", K.T_EXPERT, f"{val('expert_steps')} expert steps", f"{val('t_expert')} ms")]
    for i, (lab, v, why, txt) in enumerate(rows):
        a = 1.0 if k >= 2 else 0.0
        ax.barh(i, v, color=NAVY, height=0.6, alpha=a)
        ax.text(v * 1.08, i, f"{txt}   ({why})", va="center", fontsize=FS,
                fontweight="bold", color=NAVY, alpha=a)
    ax.axvline(K.T_PRE_ONLY, color=C["compute"], ls=":", lw=1.8)
    ax.text(K.T_PRE_ONLY * 1.05, -0.55, f"shared prefill {val('prefill')} ms", color=C["compute"],
            fontsize=SFS, fontweight="bold", ha="left", va="center")
    ax.axvline(K.T_CHUNK, color=C["danger"], ls="--", lw=1.8)
    ax.text(K.T_CHUNK * 1.05, -1.0, f"renewal {val('t_chunk')} ms", color=C["danger"],
            fontsize=SFS, fontweight="bold", ha="left", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.set_xscale("log")
    ax.set_xlim(20, 3e4)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 3000, 10000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1,000", "3,000", "10,000"])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    ax.set_xlabel(f"ms for one fresh {val('horizon')}-step chunk, INT4 {val('params_b')}B backbone on the Orin (computed)", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"decoder_step{k}.svg")


for k in (1, 2):
    decoder_fig(k)


# ---------------------------------------------------------------- published two-speed rates
def dual_fig():
    fig, ax = plt.subplots(figsize=cm(9.0, 4.4))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.25)
    for i, (name, lo, hi, fast) in enumerate(K.DUAL):
        y = i
        ax.plot([hi, fast], [y, y], color=C["neutral_l"], lw=4, zorder=1)
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=NAVY, lw=8, solid_capstyle="round", zorder=3)
        else:
            ax.plot([lo], [y], "o", ms=9, color=NAVY, zorder=3)
        ax.plot([fast], [y], "o", ms=9, color=PROP, zorder=3)
        slow_txt = f"{lo:.0f} to {hi:.0f} Hz" if hi > lo else f"{lo:.0f} Hz"
        ax.text(lo / 1.25, y - 0.32, slow_txt, ha="right", va="center", fontsize=SFS, color=NAVY, fontweight="bold")
        ax.text(fast * 1.25, y - 0.32, f"{fast:.0f} Hz", ha="left", va="center", fontsize=SFS, color=PROP, fontweight="bold")
    ax.text(1.1, -0.95, "slow model", color=NAVY, fontsize=FS, fontweight="bold", ha="left")
    ax.text(900, -0.95, "fast policy", color=PROP, fontsize=FS, fontweight="bold", ha="right")
    ax.set_yticks(range(len(K.DUAL)))
    ax.set_yticklabels([d[0] for d in K.DUAL], fontsize=FS, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(K.DUAL) - 0.5, -1.3)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 900)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    ax.set_xlabel("Rate (Hz, log scale)", fontsize=FS, labelpad=1)
    save_slot(fig, out / "dual_rates.svg")


dual_fig()


# ---------------------------------------------------------------- two-speed diagram
def box(ax, x, y, w, h, txt, ec, fc, ls="-", fs=FS, bold=True, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                                ec=ec, fc=fc, lw=1.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc or ec)


def arrow(ax, p, q, col, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=col, lw=lw, ls=ls))


def twospeed_fig():
    fig, ax = plt.subplots(figsize=cm(6.8, 5.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 5.0)
    hide(ax)
    box(ax, 0.1, 3.75, 3.6, 1.0, f"Intent model, {val('params_b')}B INT4\n{val('intent_hz')} Hz, {val('t_int4')} ms", NAVY, "#EFF6FF")
    box(ax, 0.1, 2.15, 3.6, 1.0, f"Chunk policy, {val('chunk_params_m')}M\n{val('chunk_hz')} Hz, {val('chunk_infer')} ms", NAVY, "#EFF6FF")
    box(ax, 0.1, 0.25, 3.6, 1.15, f"{val('horizon')} setpoints,\n{val('step_ms')} ms apart", PROP, "#F8FAFC", ls="--")
    arrow(ax, (1.9, 3.75), (1.9, 3.15), NAVY)
    ax.text(2.05, 3.45, "target +\ncapture time", fontsize=SFS, color=NAVY, va="center")
    arrow(ax, (1.9, 2.15), (1.9, 1.40), PROP, ls="--")
    ax.plot([4.25, 4.25], [0.1, 4.9], color="#475569", ls=(0, (4, 3)), lw=1.3)
    ax.text(4.35, 4.75, "proposal\nboundary", fontsize=SFS, color="#475569", va="top")
    box(ax, 4.55, 1.85, 2.15, 1.3, f"Permission\npath\n{val('tick_khz')} kHz", GREEN, "#F0FDF4")
    box(ax, 4.55, 0.25, 2.15, 1.0, "Drives", SLATE, "#FFFFFF")
    arrow(ax, (3.7, 0.85), (4.55, 2.2), PROP, ls="--")
    arrow(ax, (5.62, 1.85), (5.62, 1.25), GREEN)
    save_slot(fig, out / "twospeed.svg")


twospeed_fig()


# ---------------------------------------------------------------- evidence horizon drift
def drift_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.7))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.22)
    t = np.linspace(0, 420, 400)
    ax.axhline(K.MARGIN, color=C["danger"], ls="--", lw=1.6)
    ax.text(415, K.MARGIN - 1.0, f"grasp margin {val('margin_mm')} mm",
            color=C["danger"], fontsize=SFS, fontweight="bold", ha="right", va="top")
    belt = K.V_CONV * t                      # mm, since m/s x ms = mm
    ax.plot(t[t <= 100], belt[t <= 100], color=C["neutral"], lw=2.2)
    ax.plot([K.T_UNTRACKED], [K.MARGIN], "o", ms=7, color=C["neutral"], zorder=5)
    ax.text(6, 44, f"untracked, belt drift:\nmargin gone at {val('untracked_ms')} ms", ha="left", va="top",
            fontsize=FS, color=C["neutral"], fontweight="bold")
    if k >= 2:
        slip = 0.5 * K.A_SLIP * (t / 1000) ** 2 * 1000
        sel = t <= (K.SPAN if k >= 3 else K.TAU_EV + 15)
        ax.plot(t[sel], slip[sel], color=C["data"], lw=2.4)
        ax.plot([K.TAU_EV], [K.MARGIN], "o", ms=8, color=C["data"], zorder=5)
        ax.annotate(f"tracked, slip only:\nevidence horizon {val('tau_ev')} ms", (K.TAU_EV, K.MARGIN),
                    xytext=(125, 30), textcoords="data", ha="left", va="center", fontsize=FS,
                    color=C["data"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C["data"], lw=1))
        ax.plot([K.REACT], [K.DRIFT_90], "o", ms=7, color=NAVY, zorder=5)
        ax.annotate(f"replaced at {val('react')} ms: {val('drift_react')} mm", (K.REACT, K.DRIFT_90),
                    xytext=(135, 0.8), textcoords="data", ha="left", va="bottom", fontsize=SFS, color=NAVY,
                    arrowprops=dict(arrowstyle="-", color=NAVY, lw=1),
                    fontweight="bold")
    if k >= 3:
        ax.plot([K.SPAN], [K.DRIFT_SPAN], "o", ms=8, color=C["danger"], zorder=5)
        ax.annotate(f"chunk end {val('span_ms')} ms:\n{val('drift_span')} mm", (K.SPAN, K.DRIFT_SPAN),
                    xytext=(-8, 4), textcoords="offset points", ha="right", va="bottom", fontsize=FS,
                    color=C["danger"], fontweight="bold")
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 52)
    ax.set_xticks([0, K.T_UNTRACKED, K.REACT, 150, K.TAU_EV, 300, K.SPAN])
    ax.set_xticklabels(["0", val("untracked_ms"), val("react"), "150", val("tau_ev"), "300", val("span_ms")])
    ax.tick_params(labelsize=TFS, pad=1)
    ax.set_xlabel("Evidence age since capture (ms)", fontsize=FS, labelpad=1)
    ax.set_ylabel("Mug drift (mm)", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"drift_step{k}.svg")


for k in (1, 2, 3):
    drift_fig(k)


# ---------------------------------------------------------------- timing strip
def strip_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.6))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.20)
    lanes = {"capture": 4, "inference": 3, "in force": 2, "chunk 0": 1, "setpoints": 0}
    T = K.T_CHUNK
    for cap in np.arange(0, K.SPAN, T):
        ax.plot([cap, cap], [lanes["capture"] - 0.3, lanes["capture"] + 0.3], color=SLATE, lw=2)
        ax.add_patch(Rectangle((cap, lanes["inference"] - 0.28), K.CHUNK_INFER, 0.56, fc=NAVY, ec="none", alpha=0.85))
    for j, cap in enumerate(np.arange(K.CHUNK_INFER, K.SPAN, T)):
        col = GREEN if j == 0 else "#A7F3D0"
        ax.add_patch(Rectangle((cap, lanes["in force"] - 0.28), T, 0.56, fc=col, ec="white", lw=1))
        if cap + T <= K.SPAN:
            ax.text(cap + T / 2, lanes["in force"], f"{j}", ha="center", va="center", fontsize=SFS,
                    color="white" if j == 0 else SLATE, fontweight="bold")
    for s in np.arange(0, K.SPAN + 1, K.CHUNK_STEP):
        ax.plot([s, s], [lanes["setpoints"] - 0.18, lanes["setpoints"] + 0.18], color=PROP, lw=1.3)
    y = lanes["chunk 0"]
    ax.add_patch(Rectangle((0, y - 0.3), K.CHUNK_INFER, 0.6, fc=C["neutral_l"], ec="none"))
    ax.add_patch(Rectangle((K.CHUNK_INFER, y - 0.3), K.REACT - K.CHUNK_INFER, 0.6, fc=GREEN, ec="none"))
    ax.text(K.CHUNK_INFER + (K.REACT - K.CHUNK_INFER) / 2, y, "runs", ha="center", va="center",
            fontsize=SFS, color="white", fontweight="bold")
    if k >= 2:
        ax.add_patch(Rectangle((K.REACT, y - 0.3), K.TAU_EV - K.REACT, 0.6, fc="#D1FAE5", ec="none"))
        ax.text((K.REACT + K.TAU_EV) / 2, y, "evidence still valid", ha="center", va="center",
                fontsize=SFS, color=SLATE, fontweight="bold")
        ax.plot([K.TAU_EV, K.TAU_EV], [-0.5, 4.5], color=C["danger"], ls="--", lw=1.6)
        ax.text(K.TAU_EV + 3, 4.55, f"evidence horizon {val('tau_ev')} ms", color=C["danger"],
                fontsize=SFS, fontweight="bold", va="center")
    if k >= 3:
        ax.add_patch(Rectangle((K.TAU_EV, y - 0.3), K.SPAN - K.TAU_EV, 0.6, fc="white", ec=C["danger"],
                               hatch="///", lw=1.0))
        ax.text((K.TAU_EV + K.SPAN) / 2, y, f"buffer past the evidence ({val('outlast')} ms)",
                ha="center", va="center", fontsize=SFS, color=C["danger"], fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(list(lanes.keys()), fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlim(0, 405)
    ax.set_xticks([0, K.CHUNK_INFER, K.REACT, 150, K.TAU_EV, 300, K.SPAN])
    ax.set_xticklabels(["0", val("chunk_infer"), val("react"), "150", val("tau_ev"), "300", val("span_ms")])
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("ms since the capture behind chunk 0", fontsize=FS, labelpad=1)
    save_slot(fig, out / f"strip_step{k}.svg")


for k in (1, 2, 3):
    strip_fig(k)


# ---------------------------------------------------------------- ensembling blends modes
def blend_fig():
    fig, ax = plt.subplots(figsize=cm(7.6, 2.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 2.9)
    hide(ax)
    ax.add_patch(Circle((3.8, 1.45), 0.42, fc=C["danger_l"], ec=RED, lw=1.6))
    s_ = np.linspace(0, 1, 60)
    x = 0.3 + 7.0 * s_
    left = 1.45 + 0.95 * np.sin(np.pi * s_)
    right = 1.45 - 0.95 * np.sin(np.pi * s_)
    ax.plot(x, left, color=NAVY, lw=2)
    ax.plot(x, right, color=PROP, lw=2)
    ax.plot(x, (left + right) / 2, color=RED, lw=2.2, ls="--")
    ax.text(3.8, 2.55, "older chunk: left", ha="center", va="center", fontsize=SFS, color=NAVY, fontweight="bold")
    ax.text(3.8, 0.33, "newer chunk: right", ha="center", va="center", fontsize=SFS, color=PROP, fontweight="bold")
    ax.text(5.6, 1.55, "average", ha="left", va="bottom", fontsize=SFS, color=RED, fontweight="bold")
    ax.text(3.8, 0.95, "obstacle", ha="center", va="top", fontsize=SFS, color=RED, fontweight="bold")
    save_slot(fig, out / "blend.svg")


blend_fig()


# ---------------------------------------------------------------- KV ladder
def kv_fig():
    rows = [("Module memory", K.SOC_MEM, C["neutral"], f"{val('soc_gb')} GB"),
            (f"{val('hist_s')} s of history", K.KV_5S, C["danger"], f"{val('kv_hist_gb')} GB"),
            ("FP16 weights", K.W_FP16, C["memory"], f"{val('w_fp16_gb')} GB"),
            (f"One step, {val('ctx')} tokens", K.KV_STEP / 1000, C["data"], f"{val('kv_step_mb')} MB"),
            ("One token", K.KV_TOKEN, C["data"], f"{val('kv_tok_mb')} MB")]
    fig, ax = plt.subplots(figsize=cm(12.4, 4.0))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.25)
    for i, (lab, v, col, txt) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.6)
        ax.text(v * 1.2, i, txt, va="center", fontsize=FS, fontweight="bold", color=col)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e3)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=TFS, pad=1)
    ax.set_xlabel(f"GB (log scale), full-width FP16 keys and values, {val('layers')} layers", fontsize=FS, labelpad=1)
    save_slot(fig, out / "kv_ladder.svg")


kv_fig()


# ---------------------------------------------------------------- printed mug: depth check
def depth_fig():
    fig, axs = plt.subplots(1, 2, figsize=cm(7.4, 4.0), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.27, wspace=0.12)
    x = np.linspace(0, 100, 200)
    real = 30 * np.exp(-((x - 50) / 14) ** 2)
    for ax, y, title, col in ((axs[0], real, "mug handle", GREEN), (axs[1], 0 * x, "printed photo", RED)):
        ax.plot(x, y, color=col, lw=2.4)
        ax.set_title(title, fontsize=FS, color=col, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_ylim(-4, 36)
        ax.tick_params(labelsize=TFS, pad=1)
        ax.axvspan(40, 60, color=C["neutral_l"], alpha=0.6)
    axs[0].set_ylabel("Relief (mm)", fontsize=FS, labelpad=1)
    axs[0].set_xlabel("along the grasp", fontsize=SFS, labelpad=1)
    axs[1].set_xlabel("along the grasp", fontsize=SFS, labelpad=1)
    axs[1].text(50, 8, "flat: refuse", ha="center", fontsize=FS, color=RED, fontweight="bold")
    save_slot(fig, out / "depth_check.svg")


depth_fig()


# ---------------------------------------------------------------- confidence vs success
def conf_fig():
    fig, ax = plt.subplots(figsize=cm(8.6, 4.3))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.27)
    names = [r[0] for r in K.REGIMES]
    xs = np.arange(len(names))
    conf = [r[1] for r in K.REGIMES]
    acc = [r[2] for r in K.REGIMES]
    ax.bar(xs - 0.2, conf, 0.38, color=NAVY)
    ax.bar(xs + 0.2, acc, 0.38, color=C["data"])
    for x, cf, a in zip(xs, conf, acc):
        ax.text(x - 0.2, cf + 0.02, f"{cf:.2f}", ha="center", fontsize=SFS, color=NAVY, fontweight="bold")
        ax.text(x + 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=SFS, color=C["data"], fontweight="bold")
    ax.text(-0.45, 1.1, "confidence", color=NAVY, fontsize=FS, fontweight="bold", ha="left")
    ax.text(1.35, 1.1, "task success", color=C["data"], fontsize=FS, fontweight="bold", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels(["In\ndistribution", "Near\nOOD", "Sensor\ndegraded", "Far\nOOD"], fontsize=SFS)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=TFS, pad=1)
    save_slot(fig, out / "conf_success.svg")


conf_fig()


# ---------------------------------------------------------------- synthesis: the hook answered
def synth_fig():
    fig, ax = plt.subplots(figsize=cm(14.2, 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    hide(ax)
    ax.set_xlim(-95, 412)
    ax.set_ylim(-0.25, 3.45)
    ax.text(-8, 2.55, "intent model", ha="right", va="center", fontsize=FS, color=C["text"], fontweight="bold")
    ax.text(-8, 1.4, "chunk policy", ha="right", va="center", fontsize=FS, color=NAVY, fontweight="bold")
    ax.add_patch(Rectangle((0, 2.2), K.T_PREFILL, 0.7, fc=C["compute"], ec="white"))
    ax.add_patch(Rectangle((K.T_PREFILL, 2.2), K.T_INT4 - K.T_PREFILL, 0.7, fc=C["memory"], ec="white"))
    ax.text(K.T_PREFILL / 2, 2.55, f"prefill {val('prefill')} ms (arithmetic)", ha="center", va="center",
            fontsize=FS, color="white", fontweight="bold")
    ax.text(K.T_INT4 - 4, 2.55, f"decode {val('decode4_syn')} ms",
            ha="right", va="center", fontsize=FS, color="white", fontweight="bold")
    for cap in np.arange(0, K.SPAN, K.T_CHUNK):
        ax.add_patch(Rectangle((cap, 1.15), K.CHUNK_INFER, 0.5, fc=NAVY, ec="none", alpha=0.85))
    ax.plot([K.TAU_EV, K.TAU_EV], [0.9, 3.05], color=C["danger"], ls="--", lw=1.6)
    ax.text(K.TAU_EV + 3, 3.08, f"evidence horizon {val('tau_ev')} ms", color=C["danger"], fontsize=SFS,
            fontweight="bold", va="bottom")
    ax.plot([0, 400], [0.35, 0.35], color=SLATE, lw=1.2)
    for t in (0, 100, 200, 300, 400):
        ax.plot([t, t], [0.25, 0.45], color=SLATE, lw=1.2)
        ax.text(t, 0.05, f"{t} ms", ha="center", va="top", fontsize=TFS, color=SLATE)
    save_slot(fig, out / "synthesis.svg")


synth_fig()

print("figures written")
