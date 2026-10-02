#!/usr/bin/env python3
"""generate_margin_figures.py - the Volume IV margin-figure generator.

Every Volume IV margin figure is a committed vector SVG at
books/vol4/<chapter>/images/svg/margin_<slug>.svg, drawn by one registered
function below with the shared device kit (binder/tools/figures/margin/devices.py:
ladder, knee, budget_envelope, sequence_strip, causal_chain, ...).

The rule this file enforces (margin-figures.md, "Quantitative source of truth"):
a margin figure never hand-types a physical value. Each function reads its
numbers from the chapter's LEGO cell (or MLSysIM) through ``lego(chapter, name)``,
which runs the chapter's ``{python}`` cells in document order, so the figure is
drawn from the same objects the prose prints. When a chapter value changes,
regenerate and the figure follows.

Every function returns ``(fig, alt_tokens)``. ``alt_tokens`` are strings
formatted from the same LEGO values that the figure's hand-written ``fig-alt``
text must contain (fig-alt cannot hold inline Python). ``--check`` verifies
them, so a value change that the alt text did not follow fails loudly.

Usage (from the repository root, with the MLSysIM venv on PATH):

    python3 binder/tools/scripts/margin_figures/generate_margin_figures.py            # all
    python3 binder/tools/scripts/margin_figures/generate_margin_figures.py --chapter 04_nervous
    python3 binder/tools/scripts/margin_figures/generate_margin_figures.py --only margin_fieldbus_cycle_budget_envelope
    python3 binder/tools/scripts/margin_figures/generate_margin_figures.py --preview /tmp/margins   # + PNGs to inspect
    python3 binder/tools/scripts/margin_figures/generate_margin_figures.py --check     # stale SVG / alt drift / unowned refs
    python3 binder/tools/scripts/margin_figures/generate_margin_figures.py --list

Status of a registered figure:
    placed     referenced from a chapter .column-margin block
    preface    shown only in the preface gallery (frontmatter/about.qmd)
    ready      regenerated from LEGO and recommended for placement, not yet placed
    schematic  placed, but draws no physical values (shape only)
"""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mlsysim"))

from binder.tools.figures.margin.devices import (  # noqa: E402
    MEM, COMP, DATA, NET, RED, REDFILL, GRID, INK, TIME, MECHANICAL,
    new_fig, ladder, sparkline, budget_envelope, sequence_strip, save, _clean,
)

VOL4 = ROOT / "books" / "vol4"
NEUTRAL = "#4A5568"   # Vol IV softink: a boundary that is not a hazard


# ─────────────────────────────────────────────────────────────────────────────
# LEGO access: run a chapter's cells once, in order, and hand back its objects
# ─────────────────────────────────────────────────────────────────────────────

_CHAPTER_STATE: dict = {}
_CELL_RE = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```", re.S | re.M)


def lego(chap: str, name: str):
    """Return ``name`` (a LEGO class, or any module-level value) from chapter ``chap``.

    Runs the chapter's ``{python}`` cells in document order inside the chapter
    directory, stopping as soon as ``name`` is defined. The namespace is cached
    per chapter, so a second lookup continues from where the first stopped
    instead of re-running the chapter.
    """
    from binder.tools.audit.fmt.cell_exec import exec_cell_code, make_exec_namespace

    state = _CHAPTER_STATE.get(chap)
    qmd = VOL4 / chap / f"{chap}.qmd"
    if state is None:
        state = {"ns": make_exec_namespace(), "cells": _CELL_RE.findall(qmd.read_text()), "next": 0}
        _CHAPTER_STATE[chap] = state
    ns = state["ns"]
    if name not in ns:
        cwd = os.getcwd()
        os.chdir(qmd.parent)
        try:
            while state["next"] < len(state["cells"]) and name not in ns:
                exec_cell_code(state["cells"][state["next"]], ns)
                state["next"] += 1
        finally:
            os.chdir(cwd)
            plt.close("all")
    if name not in ns:
        raise KeyError(f"LEGO object {name!r} not found in {qmd}")
    return ns[name]


def mag(q, unit: str) -> float:
    """Magnitude of a Pint quantity in ``unit``."""
    return float(q.to(unit).magnitude)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarginFigure:
    name: str
    chap: str
    device: str
    status: str
    sources: tuple
    fn: Callable = field(repr=False)

    @property
    def path(self) -> Path:
        return VOL4 / self.chap / "images" / "svg" / f"{self.name}.svg"


REGISTRY: dict[str, MarginFigure] = {}


def margin(chap: str, name: str, device: str, status: str = "placed", sources=()):
    """Register a margin-figure function. ``sources`` names the LEGO/MLSysIM
    objects it reads, for ``--list`` and for the next editor."""
    def deco(fn):
        REGISTRY[name] = MarginFigure(name, chap, device, status, tuple(sources), fn)
        return fn
    return deco


# Unplaced SVGs this generator deliberately does NOT own: stale, invented, or
# contradicted by the chapter (footnotes-margins.md, section 5). Never place one.
DROPPED = {
    "margin_timescale_separation_ladder": "01: stale tiers (20 us PWM, 10 s governance); rebuild as the Ch4 cadence ladder instead",
    "margin_covariate_shift_divergence": "01: drawn with an invented t**2.2 exponent; redundant with fig-05/fig-06 compounding error",
    "margin_pdn_voltage_droop_envelope": "02: 12.8 V vs 7 V headroom from a pre-48 V bus; subsumed by the voltage-window envelope",
    "margin_mechanical_fatigue_ladder": "06: gearhead/cable lives not in the chapter; contradicts fn-hw-bearing-life",
    "margin_dma_ingress_slack_envelope": "08: replaced by margin_renewal_jitter_lease_envelope (FIG-P3)",
    "margin_active_negative_evidence_chain": "09: replaced by @fig-09-ray-update-epochs (FIG-P3)",
    "margin_operating_frequency_ladder": "10: stale (0.5 Hz VLA, 10 Hz chunks, a nonexistent 100 Hz MPC)",
    "margin_quadratic_stopping_clearance_envelope": "12: stale; redundant with @tbl/@fig-enforcement-stopping-budget",
    "margin_execution_determinism_ladder": "13: invented domain bounds; replaced by margin_coupling_timescale_ladder",
    "margin_memory_contention_tail_hockey_stick_knee": "13: invented ms-scale tails where the chapter's reads are in us",
    "margin_supervisory_escalation_ladder": "14: contradicts the chapter (no universal escalation ladder)",
    "margin_poisson_exposure_wall_scale_anchor_knee": "15: no data; the bound is deferred to Ch7",
    "margin_fault_injection_coverage_ladder": "15: draws the cumulative rung hierarchy the chapter denies; no data (cut 2026-09-29)",
    "margin_extreme_value_shock_sparkline": "07: arbitrary tail curves; replaced by margin_latch_peak_force_envelope",
    "margin_action_chunk_latency_tail_strip": "11: late-handoff counts; replaced by margin_handoff_exceedance_ladder",
    "margin_reflected_rotor_inertia_torque_knee": "11: redundant with @fig-11-action-chunk-seam-continuity",
    "margin_asil_decomposition_taxonomy": "16: ASIL is not the chapter's certification route",
    "margin_cryptographic_seal_strip": "16: names Ed25519 the chapter never uses; gate order disagrees",
    "margin_battery_diminishing_returns_knee": "17: battery endurance no longer in the chapter; hand-typed inputs",
    "margin_observational_indistinguishability_chain": "17: ends on inevitable fracture, contradicting the section",
}


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 01: The Causal Boundary
# ─────────────────────────────────────────────────────────────────────────────

@margin("01_boundary", "margin_prebrake_delay_composition_bar", "iron-law-bar",
        sources=("StoppingBudget.tau_delay", ".t_age", ".t_backbone", ".t_lease", ".t_tick", ".t_bus", ".t_onset"))
def ch01_delay_composition_bar():
    # One stacked bar of the pre-brake delay. The learned backbone (orange,
    # compute) sits inside the observation age; the chunk lease (slate, time)
    # is the largest single term. Everything else is gray.
    S = lego("01_boundary", "StoppingBudget")
    ms = lambda q: mag(q, "millisecond")
    backbone, age, lease = ms(S.t_backbone), ms(S.t_age), ms(S.t_lease)
    tickbus, onset, total = ms(S.t_tick) + ms(S.t_bus), ms(S.t_onset), ms(S.tau_delay)
    segs = [("model", backbone, COMP, "white"), ("rest of age", age - backbone, GRID, INK),
            ("lease", lease, TIME, "white"), ("", tickbus, GRID, INK), ("brake", onset, GRID, INK)]
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.70)
    y, h, left = 0.46, 0.22, 0.0
    for i, (label, v, color, txt) in enumerate(segs):
        ax.add_patch(plt.Rectangle((left, y), v, h, facecolor=color, edgecolor="white", lw=0.4))
        if label and v > 14:
            ax.text(left + v / 2, y + h / 2, f"{v:g}", ha="center", va="center", color=txt, fontsize=4.7)
        if label:
            lvl = 1 if label == "rest of age" else 0
            ax.text(left + v / 2, y - 0.07 - 0.14 * lvl, label, ha="center", va="top",
                    color=COMP if label == "model" else INK, fontsize=4.6)
        left += v
    ax.plot([0, total], [y + h + 0.10] * 2, color=TIME, lw=0.8)
    for xe in (0, total):
        ax.plot([xe, xe], [y + h + 0.06, y + h + 0.14], color=TIME, lw=0.6)
    ax.text(total / 2, y + h + 0.20, f"{total:.1f} ms before braking", ha="center", va="bottom",
            color=TIME, fontsize=4.8, fontweight="bold")
    ax.set_xlim(-1, total + 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{total:.1f} milliseconds", f"{backbone:.0f} milliseconds", f"{lease:.0f} milliseconds"]


@margin("01_boundary", "margin_tempe_suppression_strip", "sequence-strip",
        sources=("IncidentRecord.t_detect", ".t_recognize", ".t_suppress", ".t_supp_end",
                 ".t_takeover", ".d_suppress"))
def ch01_tempe_suppression_strip():
    # Order is the concept: detection, recognition, suppression, slowdown, and
    # takeover were separate events. Cells are equal width (spacing is not to
    # scale); the red cell is the second in which emergency braking was withheld.
    R = lego("01_boundary", "IncidentRecord")
    s = lambda q: mag(q, "second")
    t_det, t_rec, t_end, t_to = s(R.t_detect), s(R.t_recognize), s(R.t_supp_end), s(R.t_takeover)
    d_sup = mag(R.d_suppress, "meter")
    steps = [(f"{t_det:.1f}", GRID), (f"{t_rec:.1f}", TIME), ("", RED),
             (f"{t_end:.1f}", TIME), (f"{t_to:.2f}", TIME)]
    fig, ax = new_fig("sequence-strip")
    fig.set_size_inches(1.32, 0.90)
    sequence_strip(ax, steps, bracket=(2, 2),
                   bracket_label=f"{d_sup:.1f} m, brake withheld",
                   bracket_color=RED,
                   groups=[(0, 0, "detected", INK, 0), (1, 1, "emergency", INK, 1),
                           (2, 2, "suppressed", RED, 0), (3, 3, "slowdown", INK, 1),
                           (4, 4, "takeover", INK, 0)])
    ax.set_ylim(-0.16, 1)
    ax.text(0.5, -0.16, "seconds before impact", ha="center", va="bottom", color=TIME, fontsize=4.6,
            fontstyle="italic")
    return fig, [f"{t_det:.1f} seconds", f"{t_rec:.1f} seconds", f"{d_sup:.1f} meters",
                 f"{t_end:.1f} seconds", f"{t_to:.2f} seconds"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 02: The Physical Body
# ─────────────────────────────────────────────────────────────────────────────

@margin("02_body", "margin_reflected_rotor_inertia_knee", "scale-anchor",
        sources=("ReflectedInertia.j_load", ".j_rotor", ".n_star", ".n_gear"))
def ch02_rotor_inertia_knee():
    # Joint acceleration per unit motor torque, N / (J_load + N^2 J_rotor),
    # normalized to its peak at N* = sqrt(J_load / J_rotor). The rising branch
    # is neutral; red marks only the post-knee regime where gearing costs
    # acceleration.
    RI = lego("02_body", "ReflectedInertia")
    j_load = mag(RI.j_load, "kilogram * meter ** 2")
    j_rotor = mag(RI.j_rotor, "kilogram * meter ** 2")
    n_star, n_gear = float(RI.n_star), float(RI.n_gear)
    n_hi = 1.1 * n_gear
    fig, ax = new_fig("scale-anchor")
    N = np.linspace(5, n_hi, 300)
    acc = N / (j_load + N**2 * j_rotor)
    acc = acc / acc.max()
    m = N <= n_star
    ax.axvspan(n_star, n_hi, color=REDFILL, alpha=0.25, lw=0)
    ax.plot(N[m], acc[m], color=TIME, lw=1.35)
    ax.plot(N[~m], acc[~m], color=RED, lw=1.35)
    ax.plot(n_star, 1.0, "o", color=TIME, ms=3.5, zorder=5)
    ax.axvline(n_star, color=GRID, ls="--", lw=0.6)
    ax.text(n_star + 3, 0.84, f"N* ≈ {n_star:.0f}", color=INK, fontsize=4.8, fontweight="bold",
            ha="left", va="top")
    share = n_gear / (j_load + n_gear**2 * j_rotor) / (n_star / (j_load + n_star**2 * j_rotor))
    ax.plot(n_gear, share, "o", color=RED, ms=3.0, zorder=5)
    ax.text(n_gear, share + 0.07, f"{n_gear:.0f}:1", color=RED, fontsize=4.6, ha="center", va="bottom")
    ax.text(0.5 * (n_star + n_hi) + 5, 0.30, "rotor\ndominated", color=RED, fontsize=4.6,
            ha="center", multialignment="center")
    ax.set_xlim(5, n_hi)
    ax.set_ylim(0, 1.12)
    _clean(ax, keep=("left", "bottom"))
    return fig, [f"about {n_star:.0f}", f"{n_gear:.0f} to 1"]


@margin("02_body", "margin_braking_ceilings_envelope", "budget-envelope",
        sources=("BrakingLimits.a_req", ".a_trac_dry", ".a_torque", ".a_insp", ".a_torque_rated"))
def ch02_braking_ceilings_envelope():
    # Each ceiling on the loaded base's deceleration is a bar; the credible
    # deceleration is a dashed demand line. Bars that clear it are mechanical
    # (amber); red marks only the ceilings that fall short. Tip-over has no
    # measured center of gravity, so it gets a row and no bar.
    B = lego("02_body", "BrakingLimits")
    u = "meter / second ** 2"
    req = mag(B.a_req, u)
    rows = [("dry floor", mag(B.a_trac_dry, u)), ("peak current", mag(B.a_torque, u)),
            ("inspected floor", mag(B.a_insp, u)), ("rated current", mag(B.a_torque_rated, u))]
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 1.0)
    x0, w, h = 0.42, 0.40, 0.095
    xmax = max(v for _, v in rows) * 1.05
    ys = np.linspace(0.74, 0.18, len(rows) + 1)
    for (label, v), y in zip(rows, ys):
        c = MECHANICAL if v >= req else RED
        ax.text(x0 - 0.03, y, label, ha="right", va="center", color=INK, fontsize=4.6)
        ax.add_patch(plt.Rectangle((x0, y - h / 2), w * v / xmax, h, facecolor=c, edgecolor="white", lw=0.35))
        ax.text(0.99, y, f"{v:.2f}", ha="right", va="center", color=c, fontsize=4.5)
    ax.text(x0 - 0.03, ys[-1], "tip-over", ha="right", va="center", color=INK, fontsize=4.6)
    req_x = x0 + w * req / xmax
    ax.text(req_x + 0.03, ys[-1], "? unmeasured", ha="left", va="center", color=RED, fontsize=4.4,
            style="italic")
    ax.plot([req_x, req_x], [ys[-1] - 0.08, ys[0] + 0.08], color=INK, lw=0.7, ls="--")
    ax.text(req_x, 0.93, f"needs {req:.1f} m/s²", ha="center", va="center", color=INK, fontsize=4.6,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{rows[0][1]:.2f}", f"{rows[1][1]:.2f}", f"{rows[2][1]:.2f}", f"{rows[3][1]:.2f}",
                 f"{req:.1f}"]


@margin("02_body", "margin_drive_bus_voltage_window", "hierarchy-ladder",
        sources=("DriveBusIntegrity.v_uvlo", ".v_sag_low", ".v_battery_low", ".v_full", ".v_clamp",
                 ".v_conv_max", ".v_breakdown", ".v_final"))
def ch02_drive_bus_voltage_window():
    # One linear voltage scale for the 48 V drive bus. The sustained operating
    # band (depleted sag to full charge) is neutral; red marks the lockout below
    # it, the switch rating above it, and the unclamped regenerative endpoint.
    D = lego("02_body", "DriveBusIntegrity")
    V = {k: mag(getattr(D, k), "volt") for k in
         ("v_uvlo", "v_sag_low", "v_battery_low", "v_full", "v_clamp", "v_conv_max", "v_breakdown", "v_final")}
    marks = [
        (V["v_final"], f"{V['v_final']:.1f} V unclamped stop", RED),
        (V["v_breakdown"], f"{V['v_breakdown']:.0f} V switch rating", RED),
        (V["v_conv_max"], f"{V['v_conv_max']:.1f} V converter max", NEUTRAL),
        (V["v_clamp"], f"{V['v_clamp']:.0f} V chopper clamp", NEUTRAL),
        (V["v_full"], f"{V['v_full']:.0f} V full pack", INK),
        (V["v_battery_low"], f"{V['v_battery_low']:.0f} V depleted", INK),
        (V["v_sag_low"], f"{V['v_sag_low']:.1f} V depleted, loaded", INK),
        (V["v_uvlo"], f"{V['v_uvlo']:.0f} V converter lockout", RED),
    ]
    lo, hi = 20.0, V["v_final"] * 1.04
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.32, 1.6)
    yv = lambda v: 0.04 + 0.92 * (v - lo) / (hi - lo)
    x_axis = 0.16
    ax.add_patch(plt.Rectangle((x_axis - 0.035, yv(V["v_sag_low"])), 0.07,
                               yv(V["v_full"]) - yv(V["v_sag_low"]), facecolor=GRID, edgecolor="none"))
    ax.add_patch(plt.Rectangle((x_axis - 0.035, yv(lo)), 0.07, yv(V["v_uvlo"]) - yv(lo),
                               facecolor=REDFILL, edgecolor="none", alpha=0.6))
    ax.add_patch(plt.Rectangle((x_axis - 0.035, yv(V["v_breakdown"])), 0.07, yv(hi) - yv(V["v_breakdown"]),
                               facecolor=REDFILL, edgecolor="none", alpha=0.6))
    ax.plot([x_axis, x_axis], [yv(lo), yv(hi)], color=GRID, lw=0.6)
    ax.annotate("", xy=(x_axis + 0.06, yv(V["v_final"])), xytext=(x_axis + 0.06, yv(V["v_full"])),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.6, ls=(0, (1.5, 1.2)), mutation_scale=4))
    # spread labels so neighbours sit at least `gap` apart, keeping their order
    gap = 0.062
    ys = [yv(v) for v, _, _ in marks]
    lab = list(ys)
    for i in range(1, len(lab)):
        lab[i] = min(lab[i], lab[i - 1] - gap)
    shift = max(0.0, 0.03 - lab[-1])
    lab = [y + shift for y in lab]
    for i in range(len(lab) - 2, -1, -1):
        lab[i] = max(lab[i], lab[i + 1] + gap)
    x_lab = 0.40
    for (v, text, c), y, ly in zip(marks, ys, lab):
        ax.plot([x_axis - 0.035, x_axis + 0.035], [y, y], color=c, lw=0.7)
        ax.plot([x_axis + 0.035, x_lab - 0.08, x_lab - 0.02], [y, ly, ly], color=GRID, lw=0.35)
        ax.text(x_lab, ly, text, ha="left", va="center", color=c, fontsize=4.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{V['v_uvlo']:.0f} volts", f"{V['v_sag_low']:.1f} volts", f"{V['v_full']:.0f} volts",
                 f"{V['v_clamp']:.0f} volts", f"{V['v_breakdown']:.0f} volts", f"{V['v_final']:.1f} volts"]


@margin("02_body", "margin_phase_short_braking_knee", "scale-anchor",
        sources=("PhaseShortPeak.omega_m_peak", ".omega_drive"))
def ch02_phase_short_knee():
    # Phase-short braking torque per unit, tau ~ x / (1 + x^2) with x the wheel
    # speed over the peak speed (P_n omega L = R). The base's speed range sits
    # below the peak (neutral); red marks the fading branch past it.
    P = lego("02_body", "PhaseShortPeak")
    w_pk, w_dr = mag(P.omega_m_peak, "rpm"), mag(P.omega_drive, "rpm")
    w = np.linspace(0, 3.2 * w_pk, 300)
    x = w / w_pk
    tq = 2 * x / (1 + x**2)   # normalized to 1 at the peak
    fig, ax = new_fig("scale-anchor")
    m = w <= w_pk
    ax.axvspan(w_pk, w[-1], color=REDFILL, alpha=0.25, lw=0)
    ax.axvspan(0, w_dr, color=GRID, alpha=0.35, lw=0)
    ax.plot(w[m], tq[m], color=TIME, lw=1.35)
    ax.plot(w[~m], tq[~m], color=RED, lw=1.35)
    ax.plot(w_pk, 1.0, "o", color=TIME, ms=3.2, zorder=5)
    ax.text(w_pk + 0.08 * w_pk, 1.04, f"peak {w_pk:.0f} rpm", color=INK, fontsize=4.7, fontweight="bold",
            ha="left", va="bottom")
    tq_dr = 2 * (w_dr / w_pk) / (1 + (w_dr / w_pk) ** 2)
    ax.plot(w_dr, tq_dr, "o", color=MECHANICAL, ms=3.0, zorder=5)
    ax.text(w_dr * 0.5, 0.18, f"base up to\n{w_dr:.0f} rpm", color=INK, fontsize=4.5, ha="center",
            multialignment="center")
    ax.text(2.4 * w_pk, 0.42, "fades with\nspeed", color=RED, fontsize=4.5, ha="center", multialignment="center")
    ax.set_xlim(0, w[-1])
    ax.set_ylim(0, 1.25)
    _clean(ax, keep=("left", "bottom"))
    ax.text(w[-1], -0.05, "wheel speed", ha="right", va="top", fontsize=3.9, color="#777777")
    return fig, [f"{w_pk:.0f}", f"{w_dr:.0f}"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 03: The Cognitive Brain
# ─────────────────────────────────────────────────────────────────────────────

@margin("03_brain", "margin_action_chunk_windows_strip", "sequence-strip",
        sources=("ChunkSchedule.horizon", ".step", ".n_frozen", ".n_per_renewal", ".t_replan"))
def ch03_chunk_windows_strip():
    # One chunk of H setpoints. The first n_frozen fall inside the inference
    # time, so they are already past when the chunk arrives; the next
    # n_per_renewal run before the successor takes over one renewal period
    # later; the rest is buffer that later, overlapping chunks cover.
    S = lego("03_brain", "ChunkSchedule")
    H, past, run = int(S.horizon), int(S.n_frozen), int(S.n_per_renewal)
    buf = H - past - run
    step_ms, renew_ms = mag(S.step, "millisecond"), mag(S.t_replan, "millisecond")
    steps = [("", GRID)] * past + [("", COMP)] * run + [("", "none")] * buf
    fig, ax = new_fig("sequence-strip")
    fig.set_size_inches(1.32, 0.95)
    sequence_strip(ax, steps, bracket=(past, past + run - 1),
                   bracket_label=f"runs {renew_ms:.0f} ms", bracket_color=TIME,
                   groups=[(0, past - 1, "past", INK, 1), (past, past + run - 1, "run", COMP, 0),
                           (past + run, H - 1, f"{buf} steps of buffer", INK, 0)])
    ax.text(0.92, 0.97, f"H = {H} × {step_ms:.0f} ms", ha="right", va="top", color=INK, fontsize=4.8)
    return fig, [f"{H} steps", f"{step_ms:.0f} ms", f"{renew_ms:.0f} ms", f"remaining {buf}"]


@margin("03_brain", "margin_intent_prefill_decode_envelope", "budget-envelope",
        sources=("IntentModelLatency.t_prefill", ".fp16", ".int4", ".t_period"))
def ch03_prefill_decode_envelope():
    # One intent-model evaluation as prefill (compute-bound, orange) plus
    # decode (bandwidth-bound, violet) against the 2 Hz renewal period.
    # Weight-only INT4 shrinks only the decode segment. The period is a rate
    # budget, not a physical hazard, so it is drawn in neutral ink.
    L = lego("03_brain", "IntentModelLatency")
    pre = mag(L.t_prefill, "millisecond")
    dec16, dec4 = mag(L.fp16["t_decode"], "millisecond"), mag(L.int4["t_decode"], "millisecond")
    per = mag(L.t_period, "millisecond")
    tot16, tot4 = pre + dec16, pre + dec4
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.80)
    xmax = tot16 * 1.08
    x0, w, h = 0.20, 0.62, 0.15
    sx = lambda v: x0 + w * v / xmax
    for y, lab, dec, tot in ((0.62, "FP16", dec16, tot16), (0.34, "INT4", dec4, tot4)):
        ax.text(x0 - 0.03, y, lab, ha="right", va="center", color=INK, fontsize=4.8)
        ax.add_patch(plt.Rectangle((x0, y - h / 2), sx(pre) - x0, h, facecolor=COMP, edgecolor="white", lw=0.35))
        ax.add_patch(plt.Rectangle((sx(pre), y - h / 2), sx(tot) - sx(pre), h, facecolor=NET,
                                   edgecolor="white", lw=0.35))
        ax.text(sx(tot) + 0.02, y, f"{tot:.0f} ms", ha="left", va="center", color=INK, fontsize=4.7)
    ax.text((x0 + sx(pre)) / 2, 0.62, "prefill", ha="center", va="center", color="white", fontsize=4.0)
    ax.text((sx(pre) + sx(tot16)) / 2, 0.62, "decode", ha="center", va="center", color="white", fontsize=4.3)
    # the period line breaks around the INT4 total so the label is not struck through
    for y0, y1 in ((0.45, 0.76), (0.18, 0.24)):
        ax.plot([sx(per), sx(per)], [y0, y1], color=NEUTRAL, lw=0.7, ls="--")
    ax.text(sx(per), 0.86, f"{per:.0f} ms period", ha="center", va="center", color=NEUTRAL, fontsize=4.7,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.text(x0, 0.10, f"prefill {pre:.0f} ms in both", ha="left", va="center", color=COMP, fontsize=4.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{pre:.0f} milliseconds", f"{dec16:.0f} milliseconds", f"{dec4:.0f} milliseconds",
                 f"{tot16:.0f}", f"{tot4:.0f}", f"{per:.0f} millisecond"]


@margin("03_brain", "margin_decoder_family_latency_ladder", "hierarchy-ladder",
        sources=("DecoderFamilies.t_binned", ".t_fast", ".t_parallel", ".t_expert", ".t_prefill", ".t_period"))
def ch03_decoder_family_ladder():
    # One fresh chunk from the intent-sized backbone, per decoder family, on a
    # log time axis. Every bar is at least the shared prefill (dotted ink);
    # the red line is the chunk policy's renewal period, which all of them miss.
    D = lego("03_brain", "DecoderFamilies")
    rows = [("Binned tokens", mag(D.t_binned, "millisecond")), ("FAST tokens", mag(D.t_fast, "millisecond")),
            ("Parallel head", mag(D.t_parallel, "millisecond")), ("Flow expert", mag(D.t_expert, "millisecond"))]
    pre, per = mag(D.t_prefill, "millisecond"), mag(D.t_period, "millisecond")
    fmt_t = lambda v: f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.30, 1.40)
    xmin, xmax = per * 0.55, rows[0][1] * 2.4
    ax.set_xscale("log"); ax.set_xlim(xmin, xmax)
    n = len(rows)
    ax.set_ylim(-1.15, n - 0.05)
    # reference lines sit behind the bars; each family's name sits above its bar
    ax.plot([per, per], [-0.45, n - 0.25], color=RED, lw=0.75, ls="--", zorder=1)
    ax.plot([pre, pre], [-0.45, n - 0.25], color=INK, lw=0.6, ls=(0, (1.2, 1.2)), zorder=1)
    for i, (lab, v) in enumerate(rows):
        yy = n - 1 - i
        ax.barh(yy - 0.12, v - xmin, left=xmin, height=0.36, color=TIME, alpha=0.95, zorder=2)
        ax.text(xmin * 1.05, yy + 0.26, lab, ha="left", va="center", color=INK, fontsize=4.6, zorder=3,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
        ax.text(v * 1.10, yy - 0.12, fmt_t(v), ha="left", va="center", color=INK, fontsize=4.6, zorder=3)
    ax.text(per * 0.95, -0.72, f"{per:.0f} ms\nrenewal", ha="left", va="center", color=RED, fontsize=4.4,
            linespacing=1.0)
    ax.text(pre * 1.06, -0.72, f"prefill\n{pre:.0f} ms", ha="left", va="center", color=INK, fontsize=4.4,
            linespacing=1.0)
    _clean(ax, keep=("bottom",))
    ax.tick_params(axis="x", which="both", length=0)
    return fig, [fmt_t(v).replace(" s", " seconds").replace(" ms", " milliseconds") for _, v in rows] + \
        [f"{per:.0f} milliseconds", f"{pre:.0f} milliseconds"]


@margin("03_brain", "margin_kv_history_memory_ladder", "hierarchy-ladder",
        sources=("ActionChunkingLPDDR5.kv_per_token", ".kv_per_step", ".kv_history", ".weight_mem", ".soc_mem"))
def ch03_kv_history_ladder():
    # Retained key-value state across five orders of magnitude, beside the
    # weights it competes with and the processor's memory. Capacity is bytes,
    # so the ladder takes the memory hue.
    A = lego("03_brain", "ActionChunkingLPDDR5")
    tok, step = mag(A.kv_per_token, "MB"), mag(A.kv_per_step, "MB")
    hist, wts, soc = mag(A.kv_history, "GB"), mag(A.weight_mem, "GB"), mag(A.soc_mem, "GB")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.45)
    ladder(ax, [(f"Processor\n{soc:.1f} GB", soc * 1000), (f"5 s history\n{hist:.0f} GB", hist * 1000),
                (f"Weights\n{wts:.0f} GB", wts * 1000), (f"One step\n{step:.0f} MB", step),
                (f"One token\n{tok:.2f} MB", tok)], domain="memory")
    return fig, [f"{soc:.1f} GB", f"{hist:.0f} GB", f"{wts:.0f} GB", f"{step:.0f} MB", f"{tok:.2f} MB"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 04: The Real-Time Nervous System
# ─────────────────────────────────────────────────────────────────────────────

@margin("04_nervous", "margin_fieldbus_cycle_budget_envelope", "budget-envelope",
        sources=("FieldbusSchedulability.t_total_ethercat", ".t_total_arm", ".t_total_one", ".cycle_time"))
def ch04_fieldbus_envelope():
    # All buses are network (violet); red marks only the cycle limit and the
    # occupancy past it. The arm-bus row (2026-09-29) shows the per-limb remedy
    # beside the single-bus failure.
    F = lego("04_nervous", "FieldbusSchedulability")
    t_ecat = round(mag(F.t_total_ethercat, "microsecond"), 1)
    t_arm = round(mag(F.t_total_arm, "microsecond"), 1)
    t_can = round(mag(F.t_total_one, "microsecond"), 1)
    t_cycle = mag(F.cycle_time, "microsecond")
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("EtherCAT", t_ecat, NET), ("CAN FD arm", t_arm, NET), ("CAN FD all", t_can, NET)],
                    limit=t_cycle, style="burn", limit_label=f"{t_cycle:g} µs cycle",
                    limit_label_offset=-0.03, value_fmt="{:.1f}")
    return fig, [f"{t_can:.1f} microseconds", f"{t_arm:.1f} microseconds", f"{t_ecat:.1f} microseconds"]


@margin("04_nervous", "margin_substrate_blast_radius", "blast-radius",
        sources=("PermissionRail.v_bus_nom", ".v_uvlo"))
def ch04_substrate_blast_radius():
    # A transient on the shared drive bus reaches every load behind the shared
    # converter; the permission rail is isolated and held up, so the fan stops.
    P = lego("04_nervous", "PermissionRail")
    v_bus, v_uvlo = mag(P.v_bus_nom, "volt"), mag(P.v_uvlo, "volt")
    fig, ax = new_fig("blast-radius")
    ax.plot(0.08, 0.52, "s", color=RED, ms=10)
    ax.text(0.02, 1.0, f"{v_bus:.0f} V bus\ntransient", color=RED, fontsize=4.6,
            fontweight="bold", ha="left", va="top")
    victims = [f"host lockout {v_uvlo:.0f} V", "thermal coupling", "DMA starvation", "clock jitter"]
    for y, label in zip(np.linspace(0.30, 0.86, 4), victims):
        ax.annotate("", xy=(0.56, y), xytext=(0.14, 0.52),
                    arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.65))
        ax.plot(0.58, y, "o", color=RED, ms=4.5)
        ax.text(0.66, y, label, color=INK, fontsize=4.4, va="center", ha="left")
    ax.annotate("", xy=(0.40, 0.08), xytext=(0.14, 0.50),
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.65, ls=(0, (2, 1.5))))
    ax.plot([0.40, 0.40], [0.02, 0.14], color=DATA, lw=1.4)
    ax.plot(0.58, 0.08, "o", color=DATA, ms=4.5)
    ax.text(0.66, 0.08, "permission rail", color=INK, fontsize=4.4, va="center", ha="left")
    ax.set_xlim(0, 1.35)
    ax.set_ylim(-0.02, 1.02)
    _clean(ax)
    return fig, [f"{v_bus:.0f} V bus", f"{v_uvlo:.0f} V"]


@margin("04_nervous", "margin_cadence_tier_ladder", "hierarchy-ladder",
        sources=("MachineTiers.f_intent", "MachineTiers.t_chunk_period", "TickVocabulary.t_period",
                 "MachineTiers.t_current"))
def ch04_cadence_tier_ladder():
    # The four tiers of the chapter's tier list, as loop periods on one log
    # axis (replaces the stale Ch1 timescale ladder). The current-loop rung
    # reads MachineTiers.t_current (RM.current_loop_rate), added 2026-09-29.
    T = lego("04_nervous", "TickVocabulary")
    M = lego("04_nervous", "MachineTiers")
    t_tick = mag(T.t_period, "millisecond")
    t_chunk = mag(M.t_chunk_period, "millisecond")
    t_intent = 1000.0 / mag(M.f_intent, "hertz")
    t_cur = mag(M.t_current, "microsecond")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.45)
    ladder(ax, [(f"Intent\n{t_intent:.0f} ms", t_intent), (f"Chunk\n{t_chunk:.0f} ms", t_chunk),
                (f"Tick\n{t_tick:g} ms", t_tick), (f"Current\n{t_cur:.0f} µs", t_cur / 1000.0)],
           domain="time")
    return fig, [f"{t_intent:.0f} milliseconds", f"{t_chunk:.0f} milliseconds",
                 f"{t_tick:g} millisecond", f"{t_cur:.0f} microseconds"]


@margin("04_nervous", "margin_tick_phase_budget_envelope", "budget-envelope",
        sources=("TickDelayBudget.phi_zoh", ".phi_comp", ".phi_bus", ".phi_total", ".phi_allow", ".f_c"))
def ch04_phase_budget_envelope():
    # Phase lag of a joint loop closed through the tick and the bus, term by
    # term, against the erosion allowance (the chapter's pivotal decision:
    # the joint loops close in the drives). Red marks only the allowance.
    B = lego("04_nervous", "TickDelayBudget")
    zoh, comp, bus, tot = (round(float(x), 1) for x in (B.phi_zoh, B.phi_comp, B.phi_bus, B.phi_total))
    allow = float(B.phi_allow.magnitude)
    f_c = mag(B.f_c, "hertz")
    # One stacked bar (hold + compute + bus = total) on a shared degree axis,
    # so the eye reads which term breaks the allowance. Drawn here rather
    # than with budget_envelope, whose four-row layout overlaps at margin size.
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.72)
    xmax = tot * 1.08
    x0, w, y, h = 0.04, 0.92, 0.40, 0.20
    sx = lambda d: x0 + w * d / xmax
    ax.axvspan(sx(allow), x0 + w, color=REDFILL, alpha=0.34, lw=0)
    left = 0.0
    shades = (TIME, "#8A969D", "#3F4A50")
    for (lab, d), c, lv in zip((("hold", zoh), ("compute", comp), ("bus", bus)), shades, (0, 1, 0)):
        ax.add_patch(plt.Rectangle((sx(left), y), sx(left + d) - sx(left), h, facecolor=c,
                                   edgecolor="white", lw=0.4))
        ax.text((sx(left) + sx(left + d)) / 2, y - 0.07 - 0.15 * lv, f"{lab} {d:.1f}°", ha="center", va="top",
                color=INK, fontsize=4.6, zorder=3,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
        left += d
    ax.plot([sx(allow)] * 2, [y - 0.04, 0.97], color=RED, lw=0.7, ls="--", zorder=1)
    ax.text(sx(allow) - 0.02, 0.86, f"{allow:g}° allowed", ha="right", va="center", color=RED, fontsize=4.8)
    ax.text(sx(tot), 0.86, f"{tot:.1f}°", ha="right", va="center", color=INK, fontsize=4.8, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{f_c:.0f} hertz", f"{allow:g} degrees", f"{zoh:.1f} degrees", f"{comp:.1f} degrees",
                 f"{bus:.1f} degrees", f"{tot:.1f} degrees"]


@margin("04_nervous", "margin_lease_rule_envelope", "budget-envelope",
        sources=("PartOneCoupledBudget.t_renewal_need", ".t_lease", ".t_lease_upper_age"))
def ch04_lease_rule_envelope():
    # The lease rule on one time axis: the renewal floor below, the chosen
    # lease, and the ceiling left after tick, bus, onset, and evidence age.
    # Red marks only the ceiling (past it, the machine cannot stop in time).
    P = lego("04_nervous", "PartOneCoupledBudget")
    floor_ = mag(P.t_renewal_need, "millisecond")
    lease = mag(P.t_lease, "millisecond")
    ceil_ = round(mag(P.t_lease_upper_age, "millisecond"), 1)
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("renewal floor", floor_, TIME), ("lease", lease, DATA)],
                    limit=ceil_, limit_label=f"{ceil_:.1f} ms ceiling", limit_label_offset=-0.03,
                    value_fmt="{:.0f}")
    return fig, [f"{floor_:.0f} milliseconds", f"{lease:.0f} milliseconds", f"{ceil_:.1f} milliseconds"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 05: Physical Data Systems
# ─────────────────────────────────────────────────────────────────────────────

@margin("05_data", "margin_sensory_ingestion_bandwidth_ladder", "hierarchy-ladder",
        sources=("MultiCameraIngestionBudget.bw_nvme", ".bw_total", ".bw_nav", ".bw_compressed", ".bw_sata"))
def ch05_ingestion_ladder():
    # Raw rungs cross the SATA write ceiling; the 8x-compressed session is the
    # only rung under it (compressed rung added 2026-09-29, ch05 margins pass).
    B = lego("05_data", "MultiCameraIngestionBudget")
    bw_nvme, bw_total = mag(B.bw_nvme, "MB/s"), mag(B.bw_total, "MB/s")
    bw_nav, bw_sata = mag(B.bw_nav, "MB/s"), mag(B.bw_sata, "MB/s")
    bw_comp = mag(B.bw_compressed, "MB/s")
    fig, ax = new_fig("hierarchy-ladder")
    tiers = [(f"NVMe Gen4\n{bw_nvme:,.0f} MB/s", bw_nvme),
             (f"All streams\n{bw_total:,.0f} MB/s", bw_total),
             (f"Nav camera\n{bw_nav:,.0f} MB/s", bw_nav),
             (f"Compressed\n{bw_comp:,.0f} MB/s", bw_comp)]
    ladder(ax, tiers, domain="bandwidth", wall=False)
    for t in ax.texts:   # keep the short rung's label clear of the SATA line
        if t.get_text().startswith("Compressed"):
            t.set_x(bw_sata + 0.05 * bw_nvme)
    # The SATA write ceiling: every raw rung crosses it, the compressed one does not.
    n = len(tiers)
    ax.plot([bw_sata, bw_sata], [-0.4, n - 0.4], color=RED, lw=0.75, ls="--", zorder=3)
    ax.text(bw_sata, n - 0.22, f"SATA SSD write {bw_sata:,.0f} MB/s", color=RED,
            fontsize=5.0, ha="left", va="center")
    ax.set_ylim(-0.4, n + 0.02)
    return fig, [f"{bw_comp:,.0f}", f"{bw_sata:,.0f}"]


@margin("05_data", "margin_state_grid_coverage_ladder", "hierarchy-ladder",
        sources=("StateSpaceCoverage.n_cells_fine", ".n_cells", ".n_records", ".bins", ".bins_fine", ".n_fleet"))
def ch05_state_grid_ladder():
    # Counts on one log axis: the fine and coarse grids against every record a
    # 1,000-robot fleet could log at 1 kHz for the age of the universe.
    C = lego("05_data", "StateSpaceCoverage")
    e_fine, e_coarse = int(round(np.log10(C.n_cells_fine))), int(round(np.log10(C.n_cells)))
    m, e = f"{C.n_records:.2e}".split("e")
    e_rec = int(e)
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.30, 1.10)
    # Rungs are drawn as exponents so 10^23 and 10^56 share one readable axis.
    tiers = [(f"{C.bins_fine} bins/axis\n$10^{{{e_fine}}}$ cells", e_fine),
             (f"{C.bins} bins/axis\n$10^{{{e_coarse}}}$ cells", e_coarse),
             (f"Universe-long log\n{m} × $10^{{{e_rec}}}$ records", np.log10(C.n_records))]
    ladder(ax, tiers, domain="memory", wall=False)
    ax.text(0.0, -0.62, "log10 of count", ha="left", va="center", fontsize=3.9, color="#777777",
            transform=ax.get_yaxis_transform())
    return fig, [f"10 to the {e_fine}", f"10 to the {e_coarse}", f"{m} times 10 to the {e_rec}",
                 f"{C.n_fleet:,} robots"]


@margin("05_data", "margin_teleop_loop_latency_strip", "budget-envelope",
        sources=("TeleopLoopLatency.t_disp", ".t_hum", ".t_cmd", ".t_mech", ".t_loop"))
def ch05_teleop_loop_strip():
    # The four stages of one teleoperated correction as one proportional bar,
    # so the eye reads that the operator's reaction is most of the loop.
    # Nothing here is a hazard, so no red.
    L = lego("05_data", "TeleopLoopLatency")
    ms = lambda q: mag(q, "millisecond")
    stages = [("display", ms(L.t_disp), TIME), ("human", ms(L.t_hum), COMP),
              ("cmd", ms(L.t_cmd), "#8A969D"), ("mechanics", ms(L.t_mech), "#3F4A50")]
    tot = ms(L.t_loop)
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.72)
    x0, w, y, h = 0.04, 0.92, 0.48, 0.20
    sx = lambda d: x0 + w * d / tot
    left = 0.0
    for i, (lab, d, c) in enumerate(stages):
        ax.add_patch(plt.Rectangle((sx(left), y), sx(left + d) - sx(left), h, facecolor=c,
                                   edgecolor="white", lw=0.4))
        level = 1 if lab == "cmd" else 0
        ax.text((sx(left) + sx(left + d)) / 2, y - 0.07 - 0.31 * level, f"{lab}\n{d:.0f} ms" if level == 0
                else f"{lab} {d:.0f} ms", ha="center", va="top", color=INK, fontsize=4.5,
                multialignment="center", zorder=3, bbox=dict(boxstyle="square,pad=0.06", fc="white", ec="none"))
        left += d
    top = y + h + 0.10
    ax.plot([sx(0), sx(tot)], [top, top], color=TIME, lw=0.9)
    for xx in (sx(0), sx(tot)):
        ax.plot([xx, xx], [top - 0.04, top + 0.04], color=TIME, lw=0.7)
    ax.text(sx(tot / 2), top + 0.10, f"{tot:.0f} ms loop", ha="center", va="center", color=TIME,
            fontsize=4.8, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{ms(L.t_disp):.0f}", f"{ms(L.t_hum):.0f}", f"{ms(L.t_cmd):.0f}",
                 f"{ms(L.t_mech):.0f}", f"{tot:.0f} milliseconds"]


@margin("05_data", "margin_facility_yield_burndown_envelope", "budget-envelope",
        sources=("DemonstrationYieldFatigue.t_sched", ".t_motion", ".t_yield"))
def ch05_yield_envelope():
    # The session end is a schedule boundary, not a hazard, so it is slate.
    Y = lego("05_data", "DemonstrationYieldFatigue")
    t_sched, t_motion = mag(Y.t_sched, "minute"), mag(Y.t_motion, "minute")
    t_yield = round(mag(Y.t_yield, "minute"), 1)
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("Scheduled", t_sched, TIME), ("Motion", t_motion, TIME),
                              ("Retained", t_yield, DATA)],
                    limit=t_sched, style="burn", limit_label=f"{t_sched:.0f} min session",
                    limit_color=TIME, limit_label_offset=-0.02)
    return fig, []


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 06: Simulation & Policy Training
# ─────────────────────────────────────────────────────────────────────────────

@margin("06_training", "margin_generative_inference_deadlines_envelope", "budget-envelope",
        sources=("DecoderBudget.t_act", ".t_flow", ".t_ddim", ".k_flow", ".k_ddim", ".t_step", ".t_renew"))
def ch06_decoder_deadlines_envelope():
    # Decoder latency against the synchronous setpoint deadline (red: a missed
    # setpoint) and the asynchronous renewal period (slate reference).
    D = lego("06_training", "DecoderBudget")
    t_act, t_flow, t_ddim = (mag(D.t_act, "millisecond"), mag(D.t_flow, "millisecond"),
                             mag(D.t_ddim, "millisecond"))
    t_step, t_renew = mag(D.t_step, "millisecond"), mag(D.t_renew, "millisecond")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.85)
    budget_envelope(ax, rows=[("ACT", t_act, COMP), (f"Flow, {D.k_flow} steps", t_flow, COMP),
                              (f"DDIM, {D.k_ddim} steps", t_ddim, COMP)],
                    limit=t_step, style="burn", limit_label=f"{t_step:.0f} ms setpoint",
                    marks=[(t_renew, f"{t_renew:.0f} ms renewal", TIME)], value_fmt="{:.0f}")
    # Placed in #sec-training-decoder-costs (2026-09-29); also the preface gallery sample.
    return fig, [f"{t_act:.0f} milliseconds", f"{t_flow:.0f} milliseconds", f"{t_ddim:.0f} milliseconds",
                 f"{t_step:.0f} millisecond", f"{t_renew:.0f} millisecond"]


@margin("06_training", "margin_contact_stiffness_cliff_knee", "scale-anchor",
        sources=("LatchContactExploit.m_task", ".k_plate", ".v_fast", ".f_lim", ".k_robot_star", ".f_rigid",
                 "calc_transient_impact_force"))
def ch06_contact_stiffness_knee():
    # Peak latch force against robot stiffness, through k_eff = series(k_robot,
    # k_plate): it crosses the latch limit at the break-even stiffness and
    # saturates at the rigid-robot bound.
    X = lego("06_training", "LatchContactExploit")
    impact = lego("06_training", "calc_transient_impact_force")
    ureg = lego("06_training", "ureg")
    k_plate = mag(X.k_plate, "newton / meter")
    k_star = mag(X.k_robot_star, "newton / meter")
    f_lim, f_rigid = mag(X.f_lim, "newton"), mag(X.f_rigid, "newton")
    k = np.logspace(np.log10(k_star / 30), np.log10(k_plate * 30), 200)
    k_eff = 1.0 / (1.0 / k + 1.0 / k_plate)
    f = np.array([mag(impact(X.v_fast, X.m_task, ke * ureg.newton / ureg.meter), "newton") for ke in k_eff])
    fig, ax = new_fig("scale-anchor")
    ax.set_xscale("log")
    m = f <= f_lim
    ax.axvspan(k_star, k[-1], color=REDFILL, alpha=0.25, lw=0)
    ax.plot(k[m], f[m], color=MECHANICAL, lw=1.35)
    ax.plot(k[~m], f[~m], color=RED, lw=1.35)
    ax.axhline(f_lim, color=RED, ls="--", lw=0.6)
    ax.axhline(f_rigid, color=GRID, ls=":", lw=0.6)
    ax.plot(k_star, f_lim, "o", color=RED, ms=3.2, zorder=5)
    ax.text(k[0] * 1.2, f_lim + 6, f"limit {f_lim:.0f} N", color=RED, fontsize=4.6, va="bottom")
    ax.text(k[0] * 1.2, f_rigid + 6, f"rigid arm {f_rigid:.0f} N", color=INK, fontsize=4.6, va="bottom")
    ax.text(k_star * 1.4, f_lim - 10, f"k* ≈ {k_star / 1e3:.0f} kN/m", color=RED, fontsize=4.6,
            fontweight="bold", va="top")
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0, f_rigid * 1.2)
    _clean(ax, keep=("left", "bottom"))
    ax.minorticks_off()
    ax.set_xticks([])
    ax.text(k[-1], -0.04 * f_rigid, "arm stiffness (log)", ha="right", va="top", fontsize=3.9, color="#777777")
    return fig, [f"{f_lim:.0f} newtons", f"{f_rigid:.0f} newtons", f"{k_star / 1e3:.0f} kilonewtons per meter"]


@margin("06_training", "margin_latch_trial_calendar_ladder", "hierarchy-ladder",
        sources=("TrialWallClock.t_manual", ".t_auto", ".t_exec", "ParallelSimBudget.t_latch_sim"))
def ch06_trial_calendar_ladder():
    # The same 10^6 latch trials priced four ways: on the machine with manual
    # resets, with an automated fixture, execution alone, and in simulation at
    # the reported (locomotion) throughput. Log time axis; no hazard, so no red.
    W = lego("06_training", "TrialWallClock")
    P = lego("06_training", "ParallelSimBudget")
    d_man, d_auto, d_exec = (mag(W.t_manual, "day"), mag(W.t_auto, "day"), mag(W.t_exec, "day"))
    m_sim = mag(P.t_latch_sim, "minute")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.45)
    ladder(ax, [(f"Manual reset\n{d_man:.0f} d", d_man), (f"Auto reset\n{d_auto:.0f} d", d_auto),
                (f"Execution only\n{d_exec:.0f} d", d_exec), (f"Simulated\n{m_sim:.0f} min", m_sim / 1440.0)],
           domain="time")
    return fig, [f"{d_man:.0f} days", f"{d_auto:.0f} days", f"{d_exec:.0f} days", f"{m_sim:.0f} minutes"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 07: Empirical Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@margin("07_evaluation", "margin_nonasymptotic_reliability_ladder", "hierarchy-ladder",
        sources=("ClopperPearsonZeroFailure.n_req_999", ".n_req_99", ".n_req_90", ".n_runs", ".p_lower"))
def ch07_reliability_ladder():
    Z = lego("07_evaluation", "ClopperPearsonZeroFailure")
    fig, ax = new_fig("hierarchy-ladder")
    tiers = [(f"{Z.n_req_999:,} runs\n≥{Z.p_target_999:.1%}", Z.n_req_999),
             (f"{Z.n_req_99:,} runs\n≥{Z.p_target_99:.1%}", Z.n_req_99),
             (f"{Z.n_req_90:,} runs\n≥{Z.p_target_90:.1%}", Z.n_req_90),
             (f"{Z.n_runs:,} runs\n≥{Z.p_lower:.1%}", Z.n_runs)]
    ladder(ax, tiers, domain="rate", style="bars", color=DATA)
    return fig, [f"{Z.n_runs:,} runs", f"{Z.n_req_90:,} runs", f"{Z.n_req_99:,} runs", f"{Z.n_req_999:,} runs",
                 f"{100 * Z.p_lower:.1f} percent", f"{100 * Z.p_target_999:.1f} percent"]


@margin("07_evaluation", "margin_exposure_wall_ladder", "hierarchy-ladder",
        sources=("DoorCycle.t_target", "MtbfCampaign.t_continuous", "ExposureWall.t_e6", ".t_e9"))
def ch07_exposure_wall_ladder():
    # Failure-free exposure a zero-event test needs, from the door campaign's
    # 99.9 percent trial count (in test-cell hours) to the 1e-9 per hour wall
    # (2026-09-29 margins pass; replaces the dropped Ch15 exposure-wall knee).
    D = lego("07_evaluation", "DoorCycle")
    M = lego("07_evaluation", "MtbfCampaign")
    E = lego("07_evaluation", "ExposureWall")
    h_door = mag(D.t_target, "hour")
    h_mtbf = mag(M.t_continuous, "hour")
    h_e6, h_e9 = mag(E.t_e6, "hour"), mag(E.t_e9, "hour")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.5)
    ladder(ax, [(f"$10^{{-9}}$/h\n{h_e9 / 1e9:.1f}" + r"$\times10^{9}$ h", h_e9),
                (f"$10^{{-6}}$/h\n{h_e6 / 1e6:.1f}" + r"$\times10^{6}$ h", h_e6),
                (f"$10^{{-4}}$/h\n{h_mtbf:,.0f} h", h_mtbf), (f"door 99.9%\n{h_door:.1f} h", h_door)],
           domain="time")
    return fig, [f"{h_door:.1f} hours", f"{h_mtbf:,.0f} hours", f"{h_e6 / 1e6:.1f} million hours",
                 f"{h_e9 / 1e9:.1f} billion hours"]


@margin("07_evaluation", "margin_sprt_corridor_trace", "sparkline-trend",
        sources=("Sprt.A", ".B", ".inc_success", ".inc_failure", ".n_accept", ".m_before_fail"))
def ch07_sprt_corridor():
    # Wald's corridor: a clean record climbs to the acceptance boundary in
    # n_accept runs; the traced campaign drops by one failure's increment at
    # run m_before_fail + 1 and stays inside. Red marks only rejection.
    S = lego("07_evaluation", "Sprt")
    A, B = float(S.A), float(S.B)
    up, down = float(S.inc_success), float(S.inc_failure)
    n_acc, m_f = int(S.n_accept), int(S.m_before_fail)
    fig, ax = new_fig("sparkline-trend")
    fig.set_size_inches(1.32, 0.95)
    x_end = n_acc + 3
    ax.axhline(A, color=DATA, lw=0.7, ls="--")
    ax.axhline(B, color=RED, lw=0.7, ls="--")
    ax.axhspan(B - 1.0, B, color=REDFILL, alpha=0.34, lw=0)
    ax.text(0.5, A + 0.18, rf"accept $\Lambda_{{\mathrm{{up}}}}$ = {A:.2f}", color=DATA, fontsize=4.6, va="bottom")
    ax.text(0.5, B - 0.2, rf"reject $\Lambda_{{\mathrm{{lo}}}}$ = −{abs(B):.2f}", color=RED, fontsize=4.6, va="top")
    m = np.arange(0, n_acc + 1)
    ax.plot(m[m >= m_f], m[m >= m_f] * up, color=GRID, lw=1.0)
    ax.plot(n_acc, n_acc * up, "o", color=DATA, ms=2.6, zorder=5)
    ax.text(n_acc + 1.2, n_acc * up - 0.3, f"{n_acc} clean runs", color=INK, fontsize=4.4, ha="left", va="top")
    trace_x = list(range(0, m_f + 1)) + [m_f + 1]
    trace_y = [k * up for k in range(0, m_f + 1)] + [m_f * up + down]
    ax.plot(trace_x, trace_y, color=TIME, lw=1.2, drawstyle="default")
    ax.plot(trace_x[-1], trace_y[-1], "o", color=TIME, ms=2.6, zorder=5)
    ax.text(trace_x[-1] + 1.4, trace_y[-1], f"1 failure: −{abs(down):.2f}", color=INK, fontsize=4.4,
            va="center", ha="left")
    ax.set_xlim(0, x_end + 8)
    ax.set_ylim(B - 1.0, A + 0.9)
    _clean(ax)
    return fig, [f"{A:.2f}", f"minus {abs(B):.2f}", f"{n_acc} clean runs", f"{abs(down):.2f}"]


@margin("07_evaluation", "margin_latch_peak_force_envelope", "budget-envelope",
        sources=("MeanVsPeak.f_mean", ".f_p99", "EvalMachine.f_latch_limit", ".f_trip"))
def ch07_latch_peak_envelope():
    # Replaces the schematic tail-shape sparkline (2026-09-29): the chapter's
    # own mean and 99th-percentile latch forces against the latch limit (red),
    # with the contact tripwire as a slate reference.
    P = lego("07_evaluation", "MeanVsPeak")
    E = lego("07_evaluation", "EvalMachine")
    f_mean, f_p99 = mag(P.f_mean, "newton"), mag(P.f_p99, "newton")
    f_lim, f_trip = mag(E.f_latch_limit, "newton"), mag(E.f_trip, "newton")
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("mean", f_mean, MECHANICAL), ("99th pct peak", f_p99, MECHANICAL)],
                    limit=f_lim, style="burn", limit_label=f"{f_lim:.0f} N latch limit",
                    limit_label_offset=-0.03, marks=[(f_trip, f"{f_trip:.0f} N tripwire", TIME)],
                    value_fmt="{:.0f}")
    # The mean's value would print on the tripwire mark; set it just past the mark.
    x_trip = 0.34 + 0.58 * f_trip / (max(f_lim, f_p99, f_trip) * 1.16)
    for t in ax.texts:
        if t.get_text() == f"{f_mean:.0f}":
            t.set_x(x_trip + 0.025)
    return fig, [f"{f_mean:.0f} newtons", f"{f_p99:.0f} newtons", f"{f_lim:.0f} newtons", f"{f_trip:.0f} newtons"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 08: Sensor Perception
# ─────────────────────────────────────────────────────────────────────────────

@margin("08_perception", "margin_phase_margin_erosion_knee", "scale-anchor",
        sources=("PhaseMarginAge.tau_max", ".t_used", ".pm_nom_deg"))
def ch08_phase_margin_knee():
    PM = lego("08_perception", "PhaseMarginAge")
    tau, used, pm0 = mag(PM.tau_max, "millisecond"), mag(PM.t_used, "millisecond"), float(PM.pm_nom_deg)
    fig, ax = new_fig("sparkline-trend")
    fig.set_size_inches(1.3, 0.95)
    t = np.linspace(0, 1.3 * tau, 200)
    pm = pm0 * (1 - t / tau)
    ax.axvspan(tau, 1.3 * tau, color=REDFILL, alpha=0.34, lw=0)
    ax.axhline(0, color=GRID, lw=0.6, ls="--")
    m = t <= tau
    ax.plot(t[m], pm[m], color=DATA, lw=1.3)
    ax.plot(t[~m], pm[~m], color=RED, lw=1.3)
    ax.plot([used], [pm0 * (1 - used / tau)], "o", color=INK, ms=2.2)
    ax.text(used, pm0 * (1 - used / tau) + 0.12 * pm0, f"{used:.0f} ms", ha="center", va="bottom",
            fontsize=4.7, color=INK)
    ax.plot([tau], [0], "o", color=RED, ms=2.6)
    ax.text(0.97 * tau, -0.2 * pm0, f"{tau:.1f} ms", ha="right", va="center", fontsize=5.0, color=RED,
            fontweight="bold")
    ax.text(1.28 * tau, 0.55 * pm0, "chatter", ha="right", va="center", fontsize=4.9, color=RED)
    ax.text(0.03 * tau, pm0 * 1.02, f"PM {pm0:.0f}°", ha="left", va="bottom", fontsize=4.9, color=DATA)
    ax.text(1.3 * tau, -0.62 * pm0, "age", ha="right", va="center", fontsize=4.6, color=INK)
    ax.set_xlim(0, 1.3 * tau)
    ax.set_ylim(-0.72 * pm0, 1.25 * pm0)
    _clean(ax, keep=("left",))
    return fig, [f"PM {pm0:.0f} degrees", f"{used:.0f} milliseconds", f"{tau:.1f} milliseconds"]


@margin("08_perception", "margin_renewal_jitter_lease_envelope", "budget-envelope",
        sources=("HostContentionAllowance.t_period", ".t_jitter", ".t_lease"))
def ch08_renewal_jitter_envelope():
    H = lego("08_perception", "HostContentionAllowance")
    per, jit, lease = mag(H.t_period, "millisecond"), mag(H.t_jitter, "millisecond"), mag(H.t_lease, "millisecond")
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=(("renewal due", per, TIME), ("latest", per + jit, TIME)),
                    limit=lease, limit_label=f"lease {lease:.0f} ms", limit_label_offset=-0.03)
    return fig, [f"due at {per:.0f}", f"latest at {per + jit:.0f}", f"lease {lease:.0f} milliseconds"]


@margin("08_perception", "margin_clearance_slip_ladder", "hierarchy-ladder",
        sources=("CaptureTimingScale.dx_trigger", "RollingShutterSkew.skew", "ArrivalStampOffset.dx_recv",
                 "RackEndOpeningAge.dx", "ExtrinsicErrorBudget.dy_frame", ".delta_loc", ".side_clearance"))
def ch08_clearance_slip_ladder():
    # Each timing or mounting slip of @sec-perception-transduction-calibration
    # and @sec-perception-handoff priced in millimeters of clearance, against
    # the along-travel allowance delta_loc (slate) and the side clearance the
    # 1-degree mount error exceeds (red). 2026-09-29 margins pass.
    T = lego("08_perception", "CaptureTimingScale")
    R = lego("08_perception", "RollingShutterSkew")
    A = lego("08_perception", "ArrivalStampOffset")
    O = lego("08_perception", "RackEndOpeningAge")
    E = lego("08_perception", "ExtrinsicErrorBudget")
    trig, skew = mag(T.dx_trigger, "millimeter"), mag(R.skew, "millimeter")
    recv, age = mag(A.dx_recv, "millimeter"), mag(O.dx, "millimeter")
    mount = mag(E.dy_frame, "millimeter")
    dloc, side = mag(E.delta_loc, "millimeter"), mag(E.side_clearance, "millimeter")
    th = f"{float(E.theta_frame.to('degree').magnitude):.0f}°"
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.6)
    tiers = [(f"{th} mount\n{mount:.0f} mm", mount), (f"Full age\n{age:.1f} mm", age),
             (f"Arrival stamp\n{recv:.1f} mm", recv), (f"Row skew\n{skew:.1f} mm", skew),
             (f"Trigger stamp\n{trig:.0f} mm", trig)]
    ladder(ax, tiers, domain="distance")
    n = len(tiers)
    # Rung labels sit on a white ground above the reference lines, so a line
    # never runs through a label (it breaks behind it instead).
    for t in ax.texts:
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.25))
        t.set_zorder(4)
        if t.get_color() == "white":   # in-bar labels keep the bar as their ground
            t.set_bbox(None)
    for x, lab, col, ls in ((dloc, r"$\delta_{\mathrm{loc}}$" + f" {dloc:.0f}", TIME, ":"),
                            (side, f"side {side:.0f}", RED, "--")):
        ax.plot([x, x], [-0.4, n - 0.4], color=col, lw=0.75, ls=ls, zorder=2)
        ax.text(x, n - 0.22, lab, color=col, fontsize=4.6, ha="center", va="center")
    ax.set_ylim(-0.4, n + 0.02)
    return fig, [f"{trig:.0f} millimeters", f"{skew:.1f}", f"{recv:.1f}", f"{age:.1f}", f"{mount:.0f} millimeters",
                 f"{dloc:.0f} millimeter", f"{side:.0f} millimeter"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 09: Spatial Memory
# ─────────────────────────────────────────────────────────────────────────────

@margin("09_memory", "margin_volumetric_raycasting_bandwidth_ladder", "budget-envelope",
        sources=("RaymarchBandwidth.soc_bw", ".bw", ".share"))
def ch09_raycasting_bandwidth():
    R = lego("09_memory", "RaymarchBandwidth")
    peak, bw = mag(R.soc_bw, "GB/s"), mag(R.bw, "GB/s")
    share = f"{100 * float(R.share):.1f}"
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.80)
    ax.set_xlim(0, peak * 1.05)
    ax.set_ylim(-0.25, 1.85)
    ax.barh(1.5, peak, height=0.5, color="white", edgecolor=GRID, lw=0.6)
    ax.plot([peak, peak], [-0.1, 1.85], color=RED, lw=0.8)
    ax.text(peak * 0.97, 1.5, f"SoC peak {peak:.0f} GB/s", ha="right", va="center", fontsize=5.2, color=INK)
    ax.barh(0.5, bw, height=0.5, color=MEM, alpha=0.92)
    ax.text(bw + peak * 0.03, 0.5, f"Ray-march updates\n{bw:.2f} GB/s, {share}%", ha="left",
            va="center", fontsize=5.2, color=INK, linespacing=1.1)
    _clean(ax, keep=("bottom",))
    ax.tick_params(axis="x", which="both", length=0)
    return fig, [f"{peak:.0f} GB/s", f"{bw:.2f} GB/s", f"{share} percent"]


@margin("09_memory", "margin_misdated_update_gate_envelope", "budget-envelope",
        sources=("EstimatorConsistency.p_catch_95", ".p_catch_99", ".nis_95", ".nis_99"))
def ch09_misdated_gate_envelope():
    # Probability that one misdated stereo update trips a per-update NIS gate,
    # against certain detection (a neutral reference, not a hazard limit).
    E = lego("09_memory", "EstimatorConsistency")
    p95, p99 = float(E.p_catch_95), float(E.p_catch_99)
    g95, g99 = float(E.nis_95), float(E.nis_99)
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[(f"gate {g95:.2f}", p95, MEM), (f"gate {g99:.2f}", p99, MEM)],
                    limit=1.0, limit_color=NEUTRAL, limit_label="always caught",
                    limit_label_offset=-0.03, value_fmt="{:.2f}")
    return fig, [f"{g95:.2f}", f"{g99:.2f}", f"{p95:.2f}", f"{p99:.2f}"]


@margin("09_memory", "margin_bias_drift_confidence_sparkline", "sparkline-trend",
        sources=("MemoryBiasDrift.y_read", ".sigma_claim", ".x_true", ".span"))
def ch09_bias_drift_sparkline():
    # The reported force stays at the reading with a tight claimed band while the
    # true force slides along the unobservable direction. Red marks the true
    # force only: it is the hazard the band hides.
    B = lego("09_memory", "MemoryBiasDrift")
    y, sig = mag(B.y_read, "newton"), mag(B.sigma_claim, "newton")
    x_end, span = mag(B.x_true, "newton"), mag(B.span, "minute")
    fig, ax = new_fig("sparkline-trend")
    fig.set_size_inches(1.32, 0.95)
    t = np.linspace(0, span, 100)
    ax.fill_between(t, y - 4 * sig, y + 4 * sig, color=MEM, alpha=0.25, lw=0)
    ax.plot(t, np.full_like(t, y), color=MEM, lw=1.2)
    ax.plot(t, y + (x_end - y) * t / span, color=RED, lw=1.2)
    ax.text(0.02 * span, y + 0.35, f"reported {y:.0f} N ± {sig:.2f}", ha="left", va="bottom",
            fontsize=4.6, color=MEM)
    ax.plot([span], [x_end], "o", color=RED, ms=2.4)
    ax.text(0.97 * span, x_end - 0.35, f"true {x_end:.0f} N", ha="right", va="top", fontsize=4.8,
            color=RED, fontweight="bold")
    ax.text(span, -0.45, f"{span:.0f} min", ha="right", va="top", fontsize=4.5, color=INK)
    ax.text(0, -0.45, "0", ha="left", va="top", fontsize=4.5, color=INK)
    ax.set_xlim(0, span * 1.04)
    ax.set_ylim(-1.1, y + 1.3)
    _clean(ax, keep=("bottom",))
    return fig, [f"{y:.0f} newtons", f"{x_end:.0f} newtons", f"{span:.0f} minutes"]


@margin("09_memory", "margin_frozen_transform_chain", "causal-chain",
        sources=("MemoryFrozenTransform.t_loss", ".f_repub", ".tau_max", ".err", ".t_window"))
def ch09_frozen_transform_chain():
    from binder.tools.figures.margin.devices import causal_chain
    F = lego("09_memory", "MemoryFrozenTransform")
    t_loss, f_rep = mag(F.t_loss, "millisecond"), mag(F.f_repub, "hertz")
    tau, err, win = mag(F.tau_max, "millisecond"), mag(F.err, "millimeter"), mag(F.t_window, "millisecond")
    fig, ax = new_fig("causal-chain")
    causal_chain(ax, labels=(f"camera\nlost\n{t_loss:.0f} ms", f"restamp\n{f_rep:,.0f}\nHz",
                             f"check\n< {tau:.0f} ms\npasses", f"{err:.0f} mm\nin\n{win:.0f} ms"),
                 colors=[TIME, TIME, TIME, RED])
    return fig, [f"{t_loss:.0f} milliseconds", f"{f_rep:,.0f} hertz", f"{tau:.0f} milliseconds",
                 f"{err:.0f} millimeters", f"{win:.0f} milliseconds"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 10: Grounded Intent
# ─────────────────────────────────────────────────────────────────────────────

@margin("10_intent", "margin_dynamic_reachability_shortfall_envelope", "budget-envelope",
        sources=("t_transit_floor", "tau_ev_tracked", "n_captures_min"))
def ch10_reachability_envelope():
    # The transit floor for the offset proposal against one tracked evidence
    # window: the reach outlasts the window, so it needs more captures.
    t_floor = mag(lego("10_intent", "t_transit_floor"), "millisecond")
    tau = mag(lego("10_intent", "tau_ev_tracked"), "millisecond")
    n_cap = int(lego("10_intent", "n_captures_min"))
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("transit floor", t_floor, TIME)], limit=tau,
                    limit_label=f"evidence {tau:.0f} ms", limit_label_offset=-0.03, value_fmt="{:.0f} ms")
    ax.text(0.02, 0.24, f"needs {n_cap} captures", ha="left", va="center", color=INK, fontsize=4.7)
    return fig, [f"{t_floor:.0f} milliseconds", f"{tau:.0f} milliseconds", f"{n_cap} captures"]


@margin("10_intent", "margin_latch_stall_force_ladder", "hierarchy-ladder",
        sources=("AISLE.f_latch_tripwire", "f_at_brake_onset", "AISLE.f_latch_limit", "f_unbounded"))
def ch10_latch_stall_ladder():
    # The cage-door latch notebook: the tripwire, the force already reached at
    # brake command, and the force after an unbraked stall shorter than one
    # intent period, on a log axis against the structural latch limit (red).
    A = lego("10_intent", "AISLE")
    f_trip = mag(A.f_latch_tripwire, "newton")
    f_brake = mag(lego("10_intent", "f_at_brake_onset"), "newton")
    f_lim = mag(A.f_latch_limit, "newton")
    f_stall = mag(lego("10_intent", "f_unbounded"), "newton")
    t_stall = mag(lego("10_intent", "delta_t_stall_arm"), "millisecond")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.3, 1.15)
    ladder(ax, [(f"{t_stall:.0f} ms stall\n{f_stall:,.0f} N", f_stall),
                (f"at brake\n{f_brake:.0f} N", f_brake),
                (f"tripwire\n{f_trip:.0f} N", f_trip)], domain="force")
    ax.axvline(f_lim, color=RED, lw=0.8, ls="--")
    # The two rungs below the limit print their labels past the limit line,
    # so the line never strikes through the text.
    for t in ax.texts:
        if t.get_color() == INK and "N" in t.get_text():
            t.set_x(f_lim * 1.3)
    ax.text(f_lim * 0.9, 2.62, f"{f_lim:.0f} N limit", ha="right", va="center", color=RED, fontsize=4.7)
    ax.set_ylim(-0.65, 2.85)
    return fig, [f"{f_trip:.0f} newtons", f"{f_brake:.0f} newtons", f"{f_lim:.0f} newtons", f"{f_stall:,.0f} newtons"]


@margin("10_intent", "margin_wrong_object_chain", "causal-chain", status="schematic")
def ch10_wrong_object_chain():
    # The wrong-object premise: a confident grounding to the wrong entity passes
    # every geometric check and fails only at closure. Shape only, no values.
    from binder.tools.figures.margin.devices import causal_chain
    fig, ax = new_fig("causal-chain")
    causal_chain(ax, labels=("confident\nwrong\nobject", "passes\ngeometric\nchecks", "mug-rated\nsqueeze"),
                 colors=[COMP, NEUTRAL, RED])
    for t in ax.texts:
        t.set_fontsize(4.4)
    return fig, []


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 11: Trajectory Planning
# ─────────────────────────────────────────────────────────────────────────────

@margin("11_planning", "margin_dls_joint_rate_knee", "scale-anchor",
        sources=("_two_link_sigma_min", "PlanningTwoLinkArm.l1", ".l2", ".v_tool", ".qd_norm_good", ".qd_norm_bad",
                 "PlanningDLSStep.lam", ".qd_dls_good", ".qd_dls_bad", ".qd_bound"))
def ch11_dls_knee():
    # Weakest-direction joint rate for the TCP limit against elbow angle:
    # exact inverse v / sigma_min (red past the damping ceiling) and damped
    # least squares v sigma / (sigma^2 + lambda^2), capped at v / (2 lambda).
    # 2026-09-29 margins pass (replaces the dropped torque envelope).
    TL = lego("11_planning", "PlanningTwoLinkArm")
    D = lego("11_planning", "PlanningDLSStep")
    smin = lego("11_planning", "_two_link_sigma_min")
    l1, l2 = mag(TL.l1, "meter"), mag(TL.l2, "meter")
    v, lam = mag(TL.v_tool, "meter / second"), mag(D.lam, "meter")
    cap = mag(D.qd_bound, "radian / second")
    q_good, q_bad = float(TL.q2_good_deg), float(TL.q2_bad_deg)
    pts = [(q_good, mag(TL.qd_norm_good, "radian / second"), mag(D.qd_dls_good, "radian / second")),
           (q_bad, mag(TL.qd_norm_bad, "radian / second"), mag(D.qd_dls_bad, "radian / second"))]
    q = np.linspace(4, 95, 300)
    sig = np.array([smin(l1, l2, a) for a in q])
    exact, damped = v / sig, v * sig / (sig**2 + lam**2)
    fig, ax = new_fig("scale-anchor")
    fig.set_size_inches(1.32, 1.05)
    ax.set_yscale("log")
    ax.axhline(cap, color=NEUTRAL, lw=0.6, ls="--")
    ax.text(94, cap * 1.12, f"cap {cap:.0f} rad/s", color=NEUTRAL, fontsize=4.5, ha="right", va="bottom")
    ax.plot(q, exact, color=RED, lw=1.2)
    ax.plot(q, damped, color=TIME, lw=1.2)
    for qq, e, d in pts:
        ax.plot(qq, e, "o", color=RED, ms=2.4, zorder=5)
        ax.plot(qq, d, "o", color=TIME, ms=2.4, zorder=5)
    ax.text(q_bad + 4, pts[1][1], f"exact {pts[1][1]:.1f}", color=RED, fontsize=4.5, ha="right", va="center")
    ax.text(q_bad + 2, pts[1][2] * 0.66, f"damped {pts[1][2]:.1f}", color=TIME, fontsize=4.5, ha="right", va="top")
    ax.text(q_good, pts[0][1] * 1.35, f"{pts[0][1]:.1f}", color=RED, fontsize=4.5, ha="center", va="bottom")
    ax.text(q_good, pts[0][2] * 0.72, f"{pts[0][2]:.1f}", color=TIME, fontsize=4.5, ha="center", va="top")
    ax.text(93, 1.05, f"elbow {q_good:.0f}°", color=INK, fontsize=4.3, ha="left", va="bottom")
    ax.text(5, 1.05, "straight", color=INK, fontsize=4.3, ha="right", va="bottom")
    ax.set_xlim(95, 3)  # straightening elbow reads left to right
    ax.set_ylim(1.0, 80)
    _clean(ax, keep=("bottom",))
    from matplotlib.ticker import NullLocator
    ax.yaxis.set_minor_locator(NullLocator())
    return fig, [f"{pts[0][1]:.1f}", f"{pts[1][1]:.1f}", f"{pts[0][2]:.1f}", f"{pts[1][2]:.1f}",
                 f"{cap:.0f} radians per second"]


@margin("11_planning", "margin_replacement_latency_envelope", "budget-envelope",
        sources=("PlanningReplacementLedger.t_age", ".t_sum", "PlanningLeaseRule.t_renew", ".t_lease"))
def ch11_replacement_latency_envelope():
    # Capture-to-dispatch age and capture-to-admission L against the renewal
    # period (slate limit) and the lease (mark). Not a hazard, so no red:
    # overlap of requests, not a stop, answers L > renewal.
    R = lego("11_planning", "PlanningReplacementLedger")
    LR = lego("11_planning", "PlanningLeaseRule")
    age, L = mag(R.t_age, "millisecond"), mag(R.t_sum, "millisecond")
    renew, lease = mag(LR.t_renew, "millisecond"), mag(LR.t_lease, "millisecond")
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("to dispatch", age, TIME), ("to admission", L, TIME)], limit=renew,
                    limit_label=f"renewal {renew:.0f} ms", limit_label_offset=-0.03, limit_color=NEUTRAL,
                    marks=[(lease, f"lease {lease:.0f}", NEUTRAL)], value_fmt="{:.1f}")
    return fig, [f"{age:.1f}", f"{L:.1f}", f"{renew:.0f} millisecond renewal", f"{lease:.0f} millisecond lease"]


@margin("11_planning", "margin_handoff_exceedance_ladder", "hierarchy-ladder",
        sources=("PlanningMissionExposure.p_late", ".eps_mission", ".eps_handoff"))
def ch11_exceedance_ladder():
    # Exceedance probability per handoff: the p99 budget, the mission target,
    # and the union-bound per-handoff target (replaces the late-count strip).
    P = lego("11_planning", "PlanningMissionExposure")
    p99, eps_m, eps_h = float(P.p_late), float(P.eps_mission), float(P.eps_handoff)
    mant, ex = f"{eps_h:.1e}".split("e")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.3)
    # The leading U+200B keeps ladder()'s sentence-casing from turning the
    # upright lowercase percentile p99 (series core) into "P99" (2026-10-01).
    ladder(ax, [(f"\u200bp99 budget\n{p99:g}", p99), (f"mission\n{eps_m:g}", eps_m),
                (f"per handoff\n{mant}" + rf"$\times10^{{{int(ex)}}}$", eps_h)], domain="rate", color=TIME)
    return fig, [f"{p99:g}", f"{eps_m:g}", f"{mant} times 10 to the minus {abs(int(ex))}"]


@margin("11_planning", "margin_mug_carry_clearance_envelope", "budget-envelope",
        sources=("PlanningLeaseRule.lease_total", ".d_arm_clear", "PlanningMugSpill.d_mug_total"))
def ch11_mug_clearance_envelope():
    # Lease path plus stop at the TCP limit: empty gripper (free-space stop)
    # against an open mug (spill-limited stop), both against the clearance.
    LR = lego("11_planning", "PlanningLeaseRule")
    M = lego("11_planning", "PlanningMugSpill")
    empty, mug = mag(LR.lease_total, "millimeter"), mag(M.d_mug_total, "millimeter")
    clear = mag(LR.d_arm_clear, "millimeter")
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=[("empty gripper", empty, MECHANICAL), ("open mug", mug, MECHANICAL)],
                    limit=clear, limit_label=f"{clear:.0f} mm clearance", limit_label_offset=-0.03,
                    value_fmt="{:.1f}")
    return fig, [f"{empty:.1f} millimeters", f"{mug:.1f} millimeters", f"{clear:.0f} millimeter"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 12: Safety Enforcement
# ─────────────────────────────────────────────────────────────────────────────

@margin("12_enforcement", "margin_cbf_qp_cycle_budget_envelope", "budget-envelope",
        sources=("EnforcementCycleBudget.qp", ".wcet", ".reserve", ".unloaded", ".deadline"))
def ch12_cycle_budget_envelope():
    # The declared WCET, its QP share and its reserve, against the deadline;
    # the unloaded run is the measured reference.
    # 2026-09-29 margins pass: placed at #sec-enforcement-loop-budget.
    E = lego("12_enforcement", "EnforcementCycleBudget")
    us = lambda q: mag(q, "microsecond")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.85)
    budget_envelope(ax, rows=[("QP solve", us(E.qp), COMP), ("Declared WCET", us(E.wcet), TIME),
                              ("Reserve", us(E.reserve), TIME)],
                    limit=us(E.deadline), limit_label=f"{us(E.deadline):.0f} µs deadline",
                    limit_label_offset=-0.03, value_fmt="{:.0f}",
                    marks=[(us(E.unloaded), f"unloaded {us(E.unloaded):.0f}", NEUTRAL)])
    return fig, [f"{us(E.qp):.0f}", f"{us(E.wcet):.0f}", f"{us(E.reserve):.0f}",
                 f"{us(E.deadline):.0f} microsecond deadline", f"{us(E.unloaded):.0f}"]


@margin("12_enforcement", "margin_barrier_braking_demand_knee", "scale-anchor",
        sources=("KineticBarrierArm.inertia", ".bias", ".torque_limit", ".speed", ".radius", ".gamma",
                 ".alpha", ".clear_far", ".clear_zero", ".clear_near", ".ceiling_far", ".ceiling_near"))
def ch12_barrier_braking_demand_knee():
    # Braking torque the kinetic barrier (eq-enforce-kinetic-torque) demands
    # against tool clearance: -tau_ceiling = bias + J alpha (1 - gamma h / omega),
    # with h = d / l - omega^2 / (2 alpha). Red only past the joint's peak torque.
    # 2026-09-29 margins pass.
    K = lego("12_enforcement", "KineticBarrierArm")
    J, bias = mag(K.inertia, "kilogram * meter ** 2"), mag(K.bias, "newton * meter")
    t_max, w = mag(K.torque_limit, "newton * meter"), mag(K.speed, "1 / second")
    ell, gam, alpha = mag(K.radius, "meter"), mag(K.gamma, "1 / second"), mag(K.alpha, "1 / second ** 2")
    d_far, d_zero, d_near = (mag(K.clear_far, "millimeter"), mag(K.clear_zero, "millimeter"),
                             mag(K.clear_near, "millimeter"))
    b_far, b_near = -mag(K.ceiling_far, "newton * meter"), -mag(K.ceiling_near, "newton * meter")
    stop = w**2 / (2 * alpha)
    d = np.linspace(d_near - 10, d_far + 20, 300)
    demand = bias + J * alpha * (1 - gam * (d / 1000 / ell - stop) / w)
    fig, ax = new_fig("scale-anchor")
    fig.set_size_inches(1.32, 1.05)
    over = d <= d_zero
    ax.axvspan(d[0], d_zero, color=REDFILL, alpha=0.25, lw=0)
    ax.axhline(t_max, color=RED, lw=0.7, ls="--")
    ax.text(d[-1], t_max + 0.6, f"peak {t_max:.0f} N·m", color=RED, fontsize=4.5, ha="right", va="bottom")
    ax.plot(d[~over], demand[~over], color=MECHANICAL, lw=1.3)
    ax.plot(d[over], demand[over], color=RED, lw=1.3)
    ax.plot(d_far, b_far, "o", color=MECHANICAL, ms=2.6, zorder=5)
    ax.text(d_far - 7, b_far - 1.0, f"{d_far:.0f} mm: {b_far:.1f}", color=INK, fontsize=4.5, ha="center", va="top")
    ax.plot(d_zero, t_max, "o", color=RED, ms=2.6, zorder=5)
    ax.text(d_zero - 1.5, t_max - 1.5, f"h = 0\n{d_zero:.0f} mm", color=RED, fontsize=4.5, ha="right", va="top",
            multialignment="right")
    ax.plot(d_near, b_near, "o", color=RED, ms=2.6, zorder=5)
    ax.text(d_near + 2.5, b_near + 0.8, f"{d_near:.0f} mm: {b_near:.1f}", color=RED, fontsize=4.5, ha="left",
            va="bottom")
    ax.text(0.5 * (d[0] + d[-1]), ax.get_ylim()[0], "tool clearance (mm)", color=INK, fontsize=4.3,
            ha="center", va="bottom")
    ax.set_xlim(d[0], d[-1])
    ax.set_ylim(min(demand) - 3, max(demand) + 2)
    _clean(ax, keep=("bottom",))
    return fig, [f"{b_far:.1f}", f"{t_max:.0f} newton meter", f"{d_far:.0f} millimeters",
                 f"{d_near:.0f} millimeters", f"{b_near:.1f}"]


@margin("12_enforcement", "margin_handover_clamp_force_envelope", "budget-envelope",
        sources=("EnforcementContactForce.f_handover", ".f_handover_clamped", ".f_contact", ".v_handover"))
def ch12_handover_clamp_envelope():
    # One handover speed, two contacts: a free hand that recoils against a hand
    # clamped against an upright, against the chosen force criterion.
    # 2026-09-29 margins pass (content moved in from Ch16 by coherence fix-E).
    C = lego("12_enforcement", "EnforcementContactForce")
    free, clamped = mag(C.f_handover, "newton"), mag(C.f_handover_clamped, "newton")
    crit = mag(C.f_contact, "newton")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.8)
    budget_envelope(ax, rows=[("free hand", free, MECHANICAL), ("clamped hand", clamped, RED)],
                    limit=crit, limit_label=f"{crit:.0f} N criterion", limit_label_offset=-0.03,
                    value_fmt="{:.1f}")
    return fig, [f"{free:.1f} newtons", f"{clamped:.1f} newtons", f"{crit:.0f} newton"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 13: Silicon Placement
# ─────────────────────────────────────────────────────────────────────────────

@margin("13_placement", "margin_coupling_timescale_ladder", "hierarchy-ladder", status="ready",
        sources=("InductivePDNVoltageDroop.t_rise", "PlacementDramTiming.t_conflict",
                 "PlacementLoadedRead.queue", "PlacementMachine.tick", "RejectedPlacementThrottle.t_recov"))
def ch13_coupling_timescale_ladder():
    # Coupling mechanisms on one chassis, by the timescale each acts on.
    # 2026-09-29 margins pass: the 1.88 us rung is one FIFO backlog's wait
    # (PlacementLoadedRead.queue), not the whole read, so it is labeled so.
    s = lambda q: mag(q, "second")
    rungs = [("Thermal recovery", s(lego("13_placement", "RejectedPlacementThrottle").t_recov), "s"),
             ("Permission tick", s(lego("13_placement", "PlacementMachine").tick), "ms"),
             ("DRAM backlog wait", s(lego("13_placement", "PlacementLoadedRead").queue), "µs"),
             ("Bank conflict", s(lego("13_placement", "PlacementDramTiming").t_conflict), "ns"),
             ("Droop rise", s(lego("13_placement", "InductivePDNVoltageDroop").t_rise), "ns")]
    scale = {"s": 1.0, "ms": 1e3, "µs": 1e6, "ns": 1e9}
    fmt = lambda v, u: f"{v * scale[u]:.3g} {u}"
    words = {"s": "seconds", "ms": "millisecond", "µs": "microseconds", "ns": "nanoseconds"}
    fig, ax = new_fig("hierarchy-ladder")
    ladder(ax, [(f"{name}\n{fmt(v, u)}", v) for name, v, u in rungs], domain="time")
    return fig, [f"{v * scale[u]:.3g} {words[u]}" for _, v, u in rungs]


@margin("13_placement", "margin_stall_share_breakeven_knee", "scale-anchor",
        sources=("PlacementInterference.stretch_wcet", ".stretch_deadline", ".s_mshr", ".s_wb",
                 ".f_mshr", ".f_wb", "calc_stall_share_breakeven"))
def ch13_stall_share_knee():
    # The miss-stall share of the tick the declared WCET (and the deadline)
    # tolerates, f = (C/t - 1)/(S - 1), against the interference slowdown S on
    # log-log axes. Markers at the two measured slowdowns the chapter cites.
    # Red shades the region above the WCET curve (bound refuted).
    # 2026-09-29 margins pass; curve from MLSysIM calc_stall_share_breakeven.
    from mlsysim.physics import calc_stall_share_breakeven
    P = lego("13_placement", "PlacementInterference")
    k_w, k_d = float(P.stretch_wcet), float(P.stretch_deadline)
    s_m, s_b = float(P.s_mshr), float(P.s_wb)
    f_m, f_b = 100 * float(P.f_mshr), 100 * float(P.f_wb)
    S = np.logspace(np.log10(2.0), np.log10(1.6 * s_b), 300)
    fw = np.array([100 * calc_stall_share_breakeven(k_w, x) for x in S])
    fd = np.array([100 * calc_stall_share_breakeven(k_d, x) for x in S])
    fig, ax = new_fig("scale-anchor")
    fig.set_size_inches(1.32, 1.05)
    ax.set_xscale("log")
    ax.set_yscale("log")
    top = 100.0
    ax.fill_between(S, fw, top, color=REDFILL, alpha=0.3, lw=0)
    ax.plot(S, fd, color=NEUTRAL, lw=0.9, ls=(0, (2.5, 1.5)))
    ax.plot(S, fw, color=RED, lw=1.3)
    for x, y in ((s_m, f_m), (s_b, f_b)):
        ax.plot(x, y, "o", color=RED, ms=2.6, zorder=5)
    ax.text(s_m * 0.8, f_m * 0.55, f"{s_m:.0f}×: {f_m:.1f}%", color=RED, fontsize=4.5, ha="right", va="top")
    ax.text(s_b * 0.8, f_b * 0.55, f"{s_b:.0f}×: {f_b:.2f}%", color=RED, fontsize=4.5, ha="right", va="top")
    ax.text(2.4, 16, "WCET", color=RED, fontsize=4.4, ha="left", va="top")
    ax.text(S[-1], fd[-1] * 5.0, "deadline", color=NEUTRAL, fontsize=4.4, ha="right", va="bottom")  # 2026-09-30: lifted clear of both curves after the deadline curve moved to the full response budget (310/135)
    ax.text(40, 30, "bound refuted", color=RED, fontsize=4.5, ha="center", va="center")
    ax.text(np.sqrt(S[0] * S[-1]), 0.035, "slowdown S (log)", color=INK, fontsize=4.3, ha="center", va="bottom")
    ax.text(2.12, 0.5, "stall share (log)", color=INK, fontsize=4.3, ha="left", va="center", rotation=90)
    ax.set_xlim(S[0], S[-1])
    ax.set_ylim(0.03, top)
    _clean(ax, keep=("left", "bottom"))
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    return fig, [f"{s_m:.0f}", f"{f_m:.1f} percent", f"{s_b:.0f}", f"{f_b:.2f} percent"]


@margin("13_placement", "margin_dispatch_admission_budget_envelope", "budget-envelope",
        sources=("PlacementTransport.t_infer", ".t_link", ".delta_da", ".t_ceiling"))
def ch13_dispatch_admission_envelope():
    # The discrete placement's dispatch-to-admission allocation against the
    # ceiling the stopping budget's spare can pay for. The board link, the
    # price of the separation, is a sliver beside inference and validation.
    # Not a hazard line: exceeding the ceiling exhausts the reserve, so the
    # limit is drawn neutral. 2026-09-29 margins pass.
    T = lego("13_placement", "PlacementTransport")
    ms = lambda q: mag(q, "millisecond")
    infer, link = ms(T.t_infer), ms(T.t_link)
    da, ceil = ms(T.delta_da), ms(T.t_ceiling)
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.9)
    budget_envelope(ax, rows=[("inference", infer, COMP), ("board link", link, NET),
                              (r"$\Delta_{\mathrm{da}}$ total", da, TIME)],
                    limit=ceil, limit_label=f"{ceil:.1f} ms ceiling", limit_label_offset=-0.03,
                    limit_color=NEUTRAL, value_fmt="{:.3g}")
    return fig, [f"{infer:.0f} milliseconds", f"{link:.2f} milliseconds",
                 f"{da:.1f} milliseconds", f"{ceil:.1f} milliseconds"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 14: Supervisory Intervention
# ─────────────────────────────────────────────────────────────────────────────

@margin("14_intervention", "margin_four_phase_handshake_strip", "sequence-strip", status="ready",
        sources=("InterventionTimeoutBudget.t_request", ".t_acknowledge", ".tick", ".t_confirm", ".t_handshake"))
def ch14_handshake_strip():
    # Commit is one control tick (AISLE.t_handshake sums the four phases).
    # 2026-09-29 margins pass: cut from the chapter (redundant with
    # @tbl-14-handshake-protocol); preface gallery swapped it for the takeover-allowance envelope in the final sweep; kept ready for reuse.
    B = lego("14_intervention", "InterventionTimeoutBudget")
    ms = lambda q: mag(q, "millisecond")
    phases = [("Request", ms(B.t_request)), ("Ack", ms(B.t_acknowledge)),
              ("Commit", ms(B.tick)), ("Confirm", ms(B.t_confirm))]
    total = ms(B.t_handshake)
    fig, ax = new_fig("sequence-strip")
    fig.set_size_inches(1.32, 0.80)
    sequence_strip(ax, [(f"{v:g} ms", TIME) for _, v in phases], bracket=(0, 3),
                   bracket_label=f"{total:g} ms in all", bracket_color=TIME,
                   groups=[(i, i, p, INK, i % 2) for i, (p, _) in enumerate(phases)])
    return fig, [f"{total:g}-millisecond"]


@margin("14_intervention", "margin_takeover_allowance_envelope", "budget-envelope",
        sources=("InterventionTimeoutBudget.t_handshake", ".t_in", ".t_latest", ".t_latest_in", ".v", ".v_in"))
def ch14_takeover_allowance_envelope():
    # The protocol exchange and the whole in-loop takeover against the
    # waiting allowance at the aisle speed (red: the takeover overruns it) and
    # at the in-loop slow-down (neutral mark: it fits). Partial ledger, before
    # Delta_da, as in the paragraph it sits beside. 2026-09-29 margins pass.
    T = lego("14_intervention", "InterventionTimeoutBudget")
    ms = lambda q: mag(q, "millisecond")
    hs, t_in = ms(T.t_handshake), ms(T.t_in)
    tl, tl_in = ms(T.t_latest), ms(T.t_latest_in)
    v, v_in = mag(T.v, "meter / second"), mag(T.v_in, "meter / second")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.9)
    budget_envelope(ax, rows=[(f"handshake {hs:.0f} ms", hs, NET), (f"takeover {t_in:.0f} ms", t_in, TIME)],
                    limit=tl, limit_label=f"{tl:.1f} ms at {v:g} m/s", limit_label_offset=0.02,
                    marks=[(tl_in, f"{tl_in:.1f} ms at {v_in:g} m/s", NEUTRAL)], value_fmt="")
    return fig, [f"{hs:.0f} milliseconds", f"{t_in:.0f} milliseconds",
                 f"{tl:.1f} milliseconds", f"{tl_in:.1f} milliseconds"]


@margin("14_intervention", "margin_blended_stop_distance_envelope", "budget-envelope",
        sources=("InterventionBlendBudget.d_ideal", ".d_res", ".d_total", ".v0"))
def ch14_blended_stop_envelope():
    # Three stops from the in-loop takeover speed under one deceleration: the
    # idealized step (gray, not executable), the jerk-limited resident stop,
    # and the smoothstep blend followed by braking. The resident stop is the
    # reference line, drawn neutral: longer is not a hazard by itself, only a
    # spend of reserve. 2026-09-29 margins pass.
    B = lego("14_intervention", "InterventionBlendBudget")
    mm = lambda q: mag(q, "millimeter")
    ideal, res, total = mm(B.d_ideal), mm(B.d_res), mm(B.d_total)
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.9)
    budget_envelope(ax, rows=[("ideal step", ideal, GRID), ("resident stop", res, MECHANICAL),
                              ("blend, then brake", total, TIME)],
                    limit=res, limit_label=f"resident {res:.1f} mm", limit_color=NEUTRAL,
                    value_fmt="{:.0f}")
    return fig, [f"{ideal:.1f} millimeters", f"{res:.1f} millimeters", f"{total:.1f} millimeters"]


@margin("14_intervention", "margin_intervention_owner_strip", "sequence-strip", status="schematic")
def ch14_owner_strip():
    # The four ownership intervals of one intervention, unscaled (the drift
    # lasts seconds, the blend tens of milliseconds, re-orientation up to tens
    # of seconds), with the causal mask of each. Only the stabilized interval
    # trains the policy. Order is the concept; no physical values drawn.
    fig, ax = new_fig("sequence-strip")
    fig.set_size_inches(1.32, 0.80)
    sequence_strip(ax, [("M=0", "none"), ("M=0", "none"), ("M=0", "none"), ("M=1", DATA)],
                   bracket=(3, 3), bracket_label="trains", bracket_color=DATA,
                   groups=[(0, 0, "drift", INK, 0), (1, 1, "blend", INK, 1),
                           (2, 2, "re-orient", INK, 0), (3, 3, "stable", DATA, 1)])
    return fig, []


@margin("14_intervention", "margin_fixed_window_stop_envelope", "budget-envelope",
        sources=("InterventionFixedWindowPitfall.d_stop", ".d_stop_crawl", ".d_clear", ".window", ".v", ".v_crawl"))
def ch14_fixed_window_envelope():
    # One configured acknowledgment window added to the stopping budget at
    # the aisle speed and at the out-of-loop crawl, against the rack-end clear
    # distance (red: past it is contact). 2026-09-29 margins pass.
    F = lego("14_intervention", "InterventionFixedWindowPitfall")
    mm = lambda q: mag(q, "millimeter")
    d_fast, d_crawl, d_clear = mm(F.d_stop), mm(F.d_stop_crawl), mm(F.d_clear)
    v, v_c = mag(F.v, "meter / second"), mag(F.v_crawl, "meter / second")
    w = mag(F.window, "millisecond")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.9)
    budget_envelope(ax, rows=[(f"{v:g} m/s", d_fast, TIME), (f"{v_c:g} m/s", d_crawl, TIME)],
                    limit=d_clear, limit_label=f"{d_clear:.0f} mm clear",
                    marks=[], value_fmt="{:.0f}")
    return fig, [f"{w:.0f}-millisecond", f"{d_fast:.0f} millimeters", f"{d_crawl:.0f} millimeters",
                 f"{d_clear:.0f} millimeters"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 15: Adversarial Verification
# ─────────────────────────────────────────────────────────────────────────────

@margin("15_verification", "margin_renewal_onset_deadline_bar", "iron-law-bar",
        sources=("RackEndVerification.t_lease", ".t_tick", ".t_bus", ".t_act", ".onset_bound"))
def ch15_onset_deadline_bar():
    # The F1 onset oracle, window by window from the last valid renewal: the
    # lease dominates, tick and bus are slivers, then drive onset. One stacked
    # bar to scale (2026-09-29 margins pass; replaces the coverage ladder).
    R = lego("15_verification", "RackEndVerification")
    ms = lambda q: mag(q, "millisecond")
    lease, tick, bus, act, bound = ms(R.t_lease), ms(R.t_tick), ms(R.t_bus), ms(R.t_act), ms(R.onset_bound)
    segs = [("lease", lease, TIME, "white"), ("", tick + bus, GRID, INK), ("onset", act, GRID, INK)]
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.70)
    y, h, left = 0.46, 0.22, 0.0
    for label, v, color, txt in segs:
        ax.add_patch(plt.Rectangle((left, y), v, h, facecolor=color, edgecolor="white", lw=0.4))
        if label:
            ax.text(left + v / 2, y + h / 2, f"{v:g}", ha="center", va="center", color=txt, fontsize=4.7)
            ax.text(left + v / 2, y - 0.07, label, ha="center", va="top", color=INK, fontsize=4.6)
        left += v
    ax.text(lease + (tick + bus) / 2, y - 0.21, f"tick + bus {tick + bus:g}", ha="center", va="top",
            color=INK, fontsize=4.4)
    ax.plot([0, bound], [y + h + 0.10] * 2, color=TIME, lw=0.8)
    for xe in (0, bound):
        ax.plot([xe, xe], [y + h + 0.06, y + h + 0.14], color=TIME, lw=0.6)
    ax.text(bound / 2, y + h + 0.20, f"{bound:.0f} ms from last renewal", ha="center", va="bottom",
            color=TIME, fontsize=4.8, fontweight="bold")
    ax.set_xlim(-1, bound + 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{lease:.0f} milliseconds", f"{tick:g} millisecond", f"{act:.0f} milliseconds",
                 f"{bound:.0f} milliseconds"]


@margin("15_verification", "margin_floor_friction_ceiling_knee", "scale-anchor",
        sources=("RackEndVerification.d_clear", ".tau_delay", ".a_brake", ".j_brake", ".delta_loc",
                 ".delta_margin", ".eps_track", ".g", ".mu_min", ".mu_insp", ".mu_dry", ".ceiling_dry",
                 ".ceiling_insp", ".v_aisle", "calc_scurve_max_permitted_velocity"))
def ch15_friction_ceiling_knee():
    # Speed ceiling against floor friction: flat where the brake sets the
    # deceleration (mu >= mu_min), falling where the floor does (a = mu g).
    # The aisle speed line is red where it exceeds the ceiling. The curve is
    # the chapter's own ceiling function over a mu sweep (2026-09-29).
    from mlsysim.physics.robotics import calc_scurve_max_permitted_velocity
    R = lego("15_verification", "RackEndVerification")
    mu_min, mu_insp, mu_dry = float(R.mu_min), float(R.mu_insp), float(R.mu_dry)
    v_aisle = mag(R.v_aisle, "meter / second")
    c_dry, c_insp = mag(R.ceiling_dry, "meter / second"), mag(R.ceiling_insp, "meter / second")
    mus = np.linspace(0.6 * mu_insp, 1.1 * mu_dry, 160)

    def ceiling(mu):
        a = min(R.a_brake, (mu * R.g).to(R.a_brake.units))
        return mag(calc_scurve_max_permitted_velocity(
            R.d_clear, R.tau_delay, a, R.j_brake, loc_margin=R.delta_loc,
            physical_margin=R.delta_margin + R.eps_track), "meter / second")

    v = np.array([ceiling(m) for m in mus])
    fig, ax = new_fig("scale-anchor")
    fig.set_size_inches(1.32, 1.05)
    over = v < v_aisle
    ax.fill_between(mus, v, v_aisle, where=over, color=REDFILL, alpha=0.35, lw=0)
    ax.plot(mus, v, color=MECHANICAL, lw=1.3)
    ax.plot(mus[~over], [v_aisle] * int((~over).sum()), color=NEUTRAL, lw=0.7, ls=(0, (2, 1.5)))
    ax.plot(mus[over], [v_aisle] * int(over.sum()), color=RED, lw=0.9, ls=(0, (2, 1.5)))
    ax.text(mus[-1], v_aisle - 0.02, f"aisle {v_aisle:.1f} m/s", color=NEUTRAL, fontsize=4.5,
            ha="right", va="top")
    ax.plot(mu_min, c_dry, "o", color=MECHANICAL, ms=2.6, zorder=5)
    ax.text(mu_min + 0.02, c_dry + 0.025, f"μ {mu_min:.3f}: {c_dry:.2f} m/s", color=INK, fontsize=4.5,
            ha="left", va="bottom")
    ax.plot(mu_insp, c_insp, "o", color=RED, ms=2.6, zorder=5)
    ax.text(mu_insp + 0.02, c_insp - 0.03, f"μ {mu_insp:.2f}: {c_insp:.2f} m/s", color=RED, fontsize=4.5,
            ha="left", va="top")
    ax.set_xlim(mus[0], mus[-1])
    ax.set_ylim(v.min() - 0.12, c_dry + 0.14)
    ax.text(0.5 * (mus[0] + mus[-1]), v.min() - 0.11, "floor friction μ", color=INK, fontsize=4.3,
            ha="center", va="bottom")
    _clean(ax, keep=("bottom",))
    return fig, [f"{mu_min:.3f}", f"{c_dry:.2f} meters per second", f"{mu_insp:.2f}",
                 f"{c_insp:.2f} meters per second", f"{v_aisle:.1f} meters per second"]


@margin("15_verification", "margin_rare_event_runs_ladder", "hierarchy-ladder",
        sources=("RareEventOvershoot.n_mc", ".n_zero", ".n_is"))
def ch15_rare_event_runs_ladder():
    # Runs one rare-overshoot question needs, by method, under the chapter's
    # illustrative delay model: plain Monte Carlo at 10 percent error, a
    # zero-failure campaign, and the cross-entropy importance sampler.
    E = lego("15_verification", "RareEventOvershoot")
    n_mc, n_zero, n_is = float(E.n_mc), float(E.n_zero), float(np.ceil(E.n_is))
    sci = lambda x: (lambda m, e: f"{float(m):.1f}" + rf"$\times10^{{{int(e)}}}$")(*f"{x:.1e}".split("e"))
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.3)
    ladder(ax, [(f"plain MC\n{sci(n_mc)}", n_mc), (f"zero-failure\n{sci(n_zero)}", n_zero),
                (f"importance\n{n_is:,.0f}", n_is)], domain="count")
    word = lambda x: (lambda m, e: f"{float(m):.1f} times 10 to the {int(e)}")(*f"{x:.1e}".split("e"))
    return fig, [word(n_mc), word(n_zero), f"{n_is:,.0f}"]


@margin("15_verification", "margin_sync_blackout_drift_envelope", "budget-envelope",
        sources=("ClockDriftVerification.drift_blackout", ".one_sigma", ".drift_fifth",
                 "RackEndVerification.eps_sync"))
def ch15_sync_blackout_envelope():
    # Clock-conversion error spent by missed sync frames: a four-period
    # blackout, the one-sigma scenario, and a fifth period,
    # against the 0.2 ms bound F2's threshold rests on (red: the bound).
    C = lego("15_verification", "ClockDriftVerification")
    us = lambda q: mag(q, "microsecond")
    p4, sig, p5, bound = us(C.drift_blackout), us(C.one_sigma), us(C.drift_fifth), us(C.eps_sync)
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.85)
    budget_envelope(ax, rows=[("4 periods", p4, TIME), ("+1 sigma", sig, TIME), ("5 periods", p5, TIME)],
                    limit=bound, limit_label=f"{bound:.0f} µs bound", limit_label_offset=-0.03,
                    value_fmt="{:.0f}")
    return fig, [f"{p4:.0f} microseconds", f"{sig:.0f} microseconds",
                 f"{p5:.0f} microseconds", f"{bound:.0f} microsecond"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 16: Deployment Release
# ─────────────────────────────────────────────────────────────────────────────

@margin("16_release", "margin_aisle_budget_spare_envelope", "budget-envelope",
        sources=("BiomechanicalImpactEnvelope.d_reaction", ".d_braking", ".d_loc", ".d_track",
                 ".d_margin", ".d_clear", ".budget", ".spare"))
def ch16_budget_spare_envelope():
    # The base's partial stopping budget at aisle speed as one stacked bar
    # against the clear distance: braking dominates, and the spare the chapter
    # spends five times is the dashed remainder. The clear distance is the
    # person at the rack end, so it is the one red line. 2026-09-29 margins pass.
    B = lego("16_release", "BiomechanicalImpactEnvelope")
    mm = lambda q: mag(q, "millimeter")
    react, brake = mm(B.d_reaction), mm(B.d_braking)
    loc, trk, marg = mm(B.d_loc), mm(B.d_track), mm(B.d_margin)
    clear, budget, spare = mm(B.d_clear), mm(B.budget), mm(B.spare)
    allow = loc + trk + marg
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.78)
    y, h, left = 0.44, 0.22, 0.0
    for label, v, color, txt in (("reaction", react, TIME, "white"), ("braking", brake, MECHANICAL, "white"),
                                 ("", loc, GRID, INK), ("", trk, GRID, INK), ("", marg, GRID, INK)):
        ax.add_patch(plt.Rectangle((left, y), v, h, facecolor=color, edgecolor="white", lw=0.4))
        if label:
            ax.text(left + v / 2, y + h / 2, f"{v:.0f}", ha="center", va="center", color=txt, fontsize=4.7)
            ax.text(left + v / 2, y - 0.06, label, ha="center", va="top", color=INK, fontsize=4.5)
        left += v
    ax.text(react + brake + allow / 2, y + h / 2, f"{allow:.0f}", ha="center", va="center", color=INK,
            fontsize=4.7)
    ax.text(react + brake + allow / 2, y - 0.06, "allow.", ha="center", va="top", color=INK, fontsize=4.5)
    ax.add_patch(plt.Rectangle((budget, y), spare, h, facecolor="white", edgecolor=NEUTRAL, lw=0.5,
                               ls=(0, (1.5, 1.0))))
    ax.text(budget + spare / 2, y + h / 2, f"{spare:.0f}", ha="center", va="center", color=NEUTRAL,
            fontsize=4.7, fontweight="bold")
    ax.text(budget + spare / 2, y - 0.19, "spare", ha="center", va="top", color=NEUTRAL, fontsize=4.5,
            fontweight="bold")
    ax.plot([clear, clear], [y - 0.04, y + h + 0.10], color=RED, lw=0.8)
    ax.text(clear, y + h + 0.13, f"{clear / 1000:.2f} m clear", ha="right", va="bottom", color=RED,
            fontsize=4.8, fontweight="bold")
    ax.text(0, y + h + 0.13, "mm", ha="left", va="bottom", color=INK, fontsize=4.4)
    ax.set_xlim(-8, clear + 12); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{react:.1f} millimeters", f"{brake:.1f} millimeters", f"{loc:.0f}", f"{trk:.0f}",
                 f"{marg:.0f} millimeters", f"{spare:.1f} millimeters", f"{clear / 1000:.2f} meters"]


@margin("16_release", "margin_arm_stop_demand_ladder", "hierarchy-ladder",
        sources=("ReleaseWorkedVerdict.a_arm_lo", ".a_arm_hi", ".g", ".v_latch", ".v_latch_high"))
def ch16_arm_stop_ladder():
    # The arm stop the latch demands at the guarded and the unsupported
    # approach, on a log ladder with 1 g as a neutral reference rung: the
    # guarded stop is a fraction of g, the unsupported one tens of g.
    # 2026-09-29 margins pass (Ch12 declined it beside @fig-10-latch-force-budget).
    W = lego("16_release", "ReleaseWorkedVerdict")
    acc = lambda q: mag(q, "meter / second ** 2")
    lo, hi, g = acc(W.a_arm_lo), acc(W.a_arm_hi), acc(W.g)
    v_lo, v_hi = mag(W.v_latch, "meter / second"), mag(W.v_latch_high, "meter / second")
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.25, 1.05)
    ladder(ax, [(f"{v_hi:.2f} m/s: {hi:.0f}", hi), (f"1 g: {g:.1f}", g), (f"{v_lo:.2f} m/s: {lo:.0f}", lo)],
           color=MECHANICAL)
    ax.patches[1].set_color(GRID)
    ax.text(ax.get_xlim()[0], 2.62, "arm stop needed, m/s²", ha="left", va="bottom", color=INK, fontsize=4.5)
    ax.set_ylim(-0.65, 3.0)
    return fig, [f"{v_lo:.2f} meters per second", f"{lo:.0f} meters per second squared",
                 f"{v_hi:.2f} meters per second", f"{hi:.0f} meters per second squared", f"{g:.1f}"]


@margin("16_release", "margin_silent_proposer_response_strip", "sequence-strip",
        sources=("ReleaseBlindTravel.t_lease", ".t_tick", ".t_bus", ".t_onset", ".t_silent", ".t_monitored"))
def ch16_silent_proposer_strip():
    # Two responses on one time axis: a proposer that stops renewing is caught
    # only when its lease expires, then tick, bus, and brake onset; a premise
    # the enforcer monitors itself skips the lease. Callback to the Ch15 onset
    # bar with the monitored case beside it. 2026-09-29 margins pass.
    T = lego("16_release", "ReleaseBlindTravel")
    ms = lambda q: mag(q, "millisecond")
    lease, tick, bus, onset = ms(T.t_lease), ms(T.t_tick), ms(T.t_bus), ms(T.t_onset)
    silent, monitored = ms(T.t_silent), ms(T.t_monitored)
    fig, ax = new_fig("sequence-strip")
    fig.set_size_inches(1.32, 0.80)
    h = 0.16
    rows = ((0.64, "silent proposer", (("lease", lease, TIME, "white"), ("", tick + bus, GRID, INK),
                                       ("onset", onset, GRID, INK)), silent),
            (0.16, "monitored premise", (("", tick + bus, GRID, INK), ("onset", onset, GRID, INK)), monitored))
    for y, name, segs, total in rows:
        left = 0.0
        ax.text(0, y + h + 0.03, name, ha="left", va="bottom", color=INK, fontsize=4.6)
        for label, v, color, txt in segs:
            ax.add_patch(plt.Rectangle((left, y), v, h, facecolor=color, edgecolor="white", lw=0.4))
            if label:
                ax.text(left + v / 2, y + h / 2, f"{v:g}", ha="center", va="center", color=txt, fontsize=4.5)
                if y > 0.5:
                    ax.text(left + v / 2, y - 0.03, label, ha="center", va="top", color=INK, fontsize=4.3)
            left += v
        ax.text(left + 1.5, y + h / 2, f"{total:.0f} ms", ha="left", va="center", color=TIME, fontsize=4.9,
                fontweight="bold")
    ax.set_xlim(-1, silent + 20); ax.set_ylim(0.1, 1.0); _clean(ax)
    return fig, [f"{lease:.0f}-millisecond", f"{tick + bus:.0f} milliseconds", f"{onset:.0f} milliseconds",
                 f"{silent:.0f} milliseconds", f"{monitored:.0f} milliseconds"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 17: Engineering Physical AI
# (2026-09-29 margins pass: replaces the dropped battery knee and the
#  indistinguishability chain; the packaging-cell stop envelope was not added
#  because @fig-17-packaging-cell panel (c) already draws both stops.)
# ─────────────────────────────────────────────────────────────────────────────

@margin("17_frontier", "margin_friction_floor_deceleration_knee", "scale-anchor",
        sources=("OilFilm.a_brake", ".g", ".mu_min", ".mu_dry", ".mu_insp", ".mu_oil",
                 ".a_dry", ".a_insp", ".a_oil"))
def ch17_friction_floor_knee():
    # Credible deceleration a = min(a_brake, mu g) against floor friction:
    # flat where the brake binds, a straight line through the origin where the
    # floor binds. Dots at the three floors the chapter names; the film is the
    # hazard (red), the dry floor sits on the flat part.
    O = lego("17_frontier", "OilFilm")
    acc = lambda q: mag(q, "meter / second ** 2")
    a_b, g = acc(O.a_brake), acc(O.g)
    mu_min, mu_dry, mu_insp, mu_oil = float(O.mu_min), float(O.mu_dry), float(O.mu_insp), float(O.mu_oil)
    a_dry, a_insp, a_oil = acc(O.a_dry), acc(O.a_insp), acc(O.a_oil)
    mus = np.linspace(0.0, 1.12 * mu_dry, 200)
    a = np.minimum(a_b, mus * g)
    fig, ax = new_fig("scale-anchor")
    fig.set_size_inches(1.32, 1.05)
    ax.axvspan(0.0, mu_min, color=REDFILL, alpha=0.30, lw=0)
    ax.plot(mus, a, color=MECHANICAL, lw=1.3)
    ax.plot([mu_min, mu_min], [0, a_b], color=NEUTRAL, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(mu_min, -0.06, f"μ {mu_min:.3f}", color=NEUTRAL, fontsize=4.5, ha="center", va="top")
    ax.text(0.5 * (mu_min + mus[-1]), a_b + 0.07, f"brake binds, {a_b:.0f} m/s²", color=INK, fontsize=4.4,
            ha="center", va="bottom")
    ax.text(mu_min * 0.5, a_b + 0.07, "floor binds", color=RED, fontsize=4.4, ha="center", va="bottom")
    pts = ((mu_dry, a_dry, MECHANICAL), (mu_insp, a_insp, MECHANICAL), (mu_oil, a_oil, RED))
    for x, y, c in pts:
        ax.plot(x, y, "o", color=c, ms=2.6, zorder=5)
    ax.text(mu_dry, a_dry - 0.12, f"dry μ {mu_dry:.2f}", color=INK, fontsize=4.5, ha="center", va="top")
    ax.text(mu_min + 0.02, a_insp, f"inspected {a_insp:.2f}", color=INK, fontsize=4.5, ha="left", va="center")
    ax.text(mu_min + 0.02, a_oil, f"film {a_oil:.2f}", color=RED, fontsize=4.5, ha="left", va="center")
    ax.text(mus[-1], -0.06, "floor friction μ", color=INK, fontsize=4.3, ha="right", va="top")
    ax.set_xlim(0, mus[-1])
    ax.set_ylim(-0.42, a_b + 0.34)
    _clean(ax, keep=("bottom",))
    ax.spines["bottom"].set_bounds(0, mus[-1])
    ax.spines["bottom"].set_position(("data", 0))
    return fig, [f"{mu_min:.3f}", f"{a_b:.0f} meters per second squared", f"{mu_dry:.2f}",
                 f"{mu_insp:.2f}", f"{a_insp:.2f}", f"{mu_oil:.2f}", f"{a_oil:.2f}"]


@margin("17_frontier", "margin_stop_time_overreserve_envelope", "budget-envelope",
        sources=("OilFilm.t_brake_aisle", ".t_brake_equiv", ".t_overreserve", ".t_latest"))
def ch17_stop_time_overreserve():
    # The jerk-limited stop from the aisle speed: how long it lasts against the
    # time at unchanged speed its distance represents. The difference, what a
    # time-form deadline over-reserves (red), set beside the whole timing
    # allowance the spare distance buys (neutral).
    O = lego("17_frontier", "OilFilm")
    ms = lambda q: mag(q, "millisecond")
    t_b, t_eq, t_over, t_sp = ms(O.t_brake_aisle), ms(O.t_brake_equiv), ms(O.t_overreserve), ms(O.t_latest)
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.95)
    x0, w, h = 0.30, 0.64, 0.13
    sc = w / (1.02 * t_b)
    rows = ((0.72, f"stop {t_b:.1f} ms", t_b), (0.44, "distance ÷ v", t_eq), (0.16, "whole spare", t_sp))
    for y, label, v in rows:
        ax.text(x0 - 0.03, y, label, ha="right", va="center", color=INK, fontsize=4.6)
    # stop duration: distance-equivalent part slate, over-reserved part red
    ax.add_patch(plt.Rectangle((x0, 0.72 - h / 2), t_eq * sc, h, facecolor=TIME, edgecolor="white", lw=0.35))
    ax.add_patch(plt.Rectangle((x0 + t_eq * sc, 0.72 - h / 2), t_over * sc, h, facecolor=RED,
                               edgecolor="white", lw=0.35))
    ax.text(x0 + t_eq * sc / 2, 0.72, f"{t_eq:.1f}", ha="center", va="center", color="white", fontsize=4.6)
    ax.text(x0 + (t_eq + t_over / 2) * sc, 0.72 + h / 2 + 0.03, f"+{t_over:.1f} over", ha="center",
            va="bottom", color=RED, fontsize=4.5)
    ax.add_patch(plt.Rectangle((x0, 0.44 - h / 2), t_eq * sc, h, facecolor=TIME, edgecolor="white", lw=0.35))
    ax.text(x0 + t_eq * sc / 2, 0.44, f"{t_eq:.1f}", ha="center", va="center", color="white", fontsize=4.6)
    ax.add_patch(plt.Rectangle((x0, 0.16 - h / 2), t_sp * sc, h, facecolor=GRID, edgecolor="white", lw=0.35))
    ax.text(x0 + t_sp * sc + 0.02, 0.16, f"{t_sp:.1f} ms", ha="left", va="center", color=INK, fontsize=4.6)
    ax.plot([x0 + t_eq * sc] * 2, [0.34, 0.79], color=NEUTRAL, lw=0.5, ls=(0, (1.5, 1.2)), zorder=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _clean(ax)
    return fig, [f"{t_b:.1f} milliseconds", f"{t_eq:.1f} milliseconds", f"{t_over:.1f} milliseconds",
                 f"{t_sp:.1f} milliseconds"]


@margin("17_frontier", "margin_latch_coasting_peak_envelope", "budget-envelope",
        sources=("LatchCompliance.f_coast_rigid_guarded", ".f_coast_rigid_fast",
                 ".f_coast_compliant_fast", ".f_floor", ".v_guarded", ".v_fast"))
def ch17_latch_coasting_envelope():
    # Peak latch force if drive torque is removed at the tripwire: the rigid
    # plate at the guarded and the fast approach, and the series element at the
    # fast approach, against the declared force floor (red: the floor is the
    # limit a weakened latch may not see exceeded).
    L = lego("17_frontier", "LatchCompliance")
    N = lambda q: mag(q, "newton")
    g_r, f_r, f_c, floor = N(L.f_coast_rigid_guarded), N(L.f_coast_rigid_fast), N(L.f_coast_compliant_fast), N(L.f_floor)
    vg, vf = mag(L.v_guarded, "meter / second"), mag(L.v_fast, "meter / second")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.9)
    budget_envelope(ax, rows=[(f"rigid, {vg:g} m/s", g_r, MECHANICAL), (f"rigid, {vf:g} m/s", f_r, MECHANICAL),
                              (f"spring, {vf:g} m/s", f_c, MECHANICAL)],
                    limit=floor, limit_label=f"{floor:.0f} N floor", value_fmt="{:.0f}")
    return fig, [f"{g_r:.0f} newtons", f"{f_r:.0f} newtons", f"{f_c:.0f} newtons", f"{floor:.0f} newton",
                 f"{vg:g} m/s", f"{vf:g} m/s"]


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _render(spec: MarginFigure, out_svg: Path) -> list:
    fig, alt = spec.fn()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(out_svg))
    return alt


def _preview(svg: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{svg.stem}.png"
    subprocess.run(["rsvg-convert", "-w", "700", "-b", "white", str(svg), "-o", str(png)], check=True)
    return png


def _qmd_alts(name: str) -> list:
    """(file, fig-alt) for every vol4 qmd reference to ``name``.svg."""
    pat = re.compile(re.escape(name) + r'\.svg\)\{[^}]*fig-alt="([^"]*)"')
    hits = []
    for qmd in sorted(VOL4.rglob("*.qmd")):
        for alt in pat.findall(qmd.read_text()):
            hits.append((qmd.relative_to(ROOT), alt))
    return hits


def _referenced_margin_svgs() -> set:
    names = set()
    for qmd in VOL4.rglob("*.qmd"):
        names.update(re.findall(r"(margin_[a-z0-9_]+)\.svg", qmd.read_text()))
    return names


def check(specs) -> int:
    """Fail on a stale committed SVG, an alt text that lost a LEGO value, or a
    placed margin SVG this generator does not own."""
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        for spec in specs:
            fresh = Path(tmp) / f"{spec.name}.svg"
            alt_tokens = _render(spec, fresh)
            if not spec.path.exists():
                problems.append(f"MISSING  {spec.path.relative_to(ROOT)}")
            elif not filecmp.cmp(fresh, spec.path, shallow=False):
                problems.append(f"STALE    {spec.path.relative_to(ROOT)} (regenerate)")
            if spec.status in ("placed", "preface", "schematic"):
                refs = _qmd_alts(spec.name)
                if not refs:
                    problems.append(f"UNPLACED {spec.name} has status {spec.status!r} but no qmd reference")
                for qmd, alt in refs:
                    if "Margin visual example" in alt:
                        continue  # preface gallery alt text describes the form, not values
                    for tok in alt_tokens:
                        if tok not in alt:
                            problems.append(f"ALT      {qmd}: fig-alt of {spec.name} lacks {tok!r}")
    for name in sorted(_referenced_margin_svgs() - set(REGISTRY)):
        why = DROPPED.get(name, "not registered")
        problems.append(f"UNOWNED  {name}.svg is referenced in a vol4 qmd but {why}")
    for p in problems:
        print(p)
    print(f"margin-figure check: {len(specs)} figures, {len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chapter", help="only this chapter directory, e.g. 04_nervous")
    ap.add_argument("--only", action="append", help="only this figure name (repeatable)")
    ap.add_argument("--preview", type=Path, help="also write inspection PNGs (rsvg-convert) here")
    ap.add_argument("--check", action="store_true", help="verify committed SVGs and fig-alt values; write nothing")
    ap.add_argument("--list", action="store_true", help="list registered figures and their sources")
    args = ap.parse_args(argv)

    specs = [s for s in REGISTRY.values()
             if (not args.chapter or s.chap == args.chapter) and (not args.only or s.name in args.only)]
    if args.list:
        for s in specs:
            print(f"{s.chap:16} {s.status:9} {s.device:16} {s.name}\n{'':43}<- {', '.join(s.sources) or 'no values'}")
        return 0
    if args.check:
        return check(specs)
    for s in specs:
        _render(s, s.path)
        line = f"wrote {s.path.relative_to(ROOT)}"
        if args.preview:
            line += f"  preview {_preview(s.path, args.preview)}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
