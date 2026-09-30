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
        sources=("MultiCameraIngestionBudget.bw_nvme", ".bw_total", ".bw_nav", ".bw_sata"))
def ch05_ingestion_ladder():
    B = lego("05_data", "MultiCameraIngestionBudget")
    bw_nvme, bw_total = mag(B.bw_nvme, "MB/s"), mag(B.bw_total, "MB/s")
    bw_nav, bw_sata = mag(B.bw_nav, "MB/s"), mag(B.bw_sata, "MB/s")
    fig, ax = new_fig("hierarchy-ladder")
    tiers = [(f"NVMe Gen4\n{bw_nvme:,.0f} MB/s", bw_nvme),
             (f"All streams\n{bw_total:,.0f} MB/s", bw_total),
             (f"Nav camera\n{bw_nav:,.0f} MB/s", bw_nav)]
    ladder(ax, tiers, domain="bandwidth", wall=False)
    # The SATA write ceiling is the wall every camera rung crosses.
    n = len(tiers)
    ax.plot([bw_sata, bw_sata], [-0.4, n - 0.4], color=RED, lw=0.75, ls="--", zorder=3)
    ax.text(bw_sata, n - 0.22, f"SATA SSD write {bw_sata:,.0f} MB/s", color=RED,
            fontsize=5.0, ha="left", va="center")
    ax.set_ylim(-0.4, n + 0.02)
    return fig, []


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
        status="preface",
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
    return fig, []


@margin("06_training", "margin_contact_stiffness_cliff_knee", "scale-anchor", status="ready",
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
    return fig, []


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
    return fig, [f"about {100 * Z.p_lower:.0f} percent", f"{100 * Z.p_target_999:.1f} percent"]


@margin("07_evaluation", "margin_extreme_value_shock_sparkline", "sparkline-trend", status="schematic")
def ch07_extreme_value_sparkline():
    # Shape only: the caption says neither curve is the campaign's tail family.
    fig, ax = new_fig("sparkline-trend")
    t = np.linspace(0, 1, 100)
    p_gauss = np.exp(-8 * t)
    p_frechet = 1.0 / (1.0 + 3.0 * t) ** 1.8
    ax.plot(t, p_frechet, color=RED, lw=1.35)
    ax.plot(t, p_gauss, color=TIME, lw=1.35)
    ax.fill_between(t, p_gauss, p_frechet, color=REDFILL, alpha=0.35)
    ax.text(0.95, 0.82, "Heavy tail (Fréchet)", color=RED, fontsize=5.0, fontweight="bold", ha="right")
    ax.text(0.95, 0.62, "Light tail (Gumbel domain)", color=TIME, fontsize=4.8, ha="right")
    ax.set_xlim(-0.04, 1.0)
    ax.set_ylim(-0.04, 1.1)
    _clean(ax)
    return fig, []


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


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 10: Grounded Intent
# ─────────────────────────────────────────────────────────────────────────────

@margin("10_intent", "margin_dynamic_reachability_shortfall_envelope", "budget-envelope", status="ready",
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
    ax.text(0.02, 0.14, f"needs {n_cap} captures", ha="left", va="center", color=INK, fontsize=4.7)
    return fig, []


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 11: Trajectory Planning
# ─────────────────────────────────────────────────────────────────────────────

@margin("11_planning", "margin_action_chunk_latency_tail_strip", "hierarchy-ladder",
        sources=("PlanningMissionExposure.n_late", ".n_handoff", ".eps_handoff", ".q_pct"))
def ch11_tail_ladder():
    # Name kept for the existing placement; the device is a two-rung ladder.
    P = lego("11_planning", "PlanningMissionExposure")
    n_late = float(P.n_late)
    n_design = float(P.n_handoff * P.eps_handoff)
    q = f"{P.q_pct:.5f}"
    fig, ax = new_fig("hierarchy-ladder")
    fig.set_size_inches(1.2, 1.1)
    ladder(ax, [(f"P99: {n_late:.0f}", n_late), (f"P{q}: {n_design:g}", n_design)], color=TIME)
    return fig, [f"P99 {n_late:.0f}", f"P{q} {n_design:g}"]


@margin("11_planning", "margin_reflected_rotor_inertia_torque_knee", "budget-envelope",
        sources=("PlanningSeamInertia.tau_step", ".tau_blend_peak", ".tau_rated", ".t_blend", ".dt_tick"))
def ch11_torque_envelope():
    SI = lego("11_planning", "PlanningSeamInertia")
    step = round(mag(SI.tau_step, "newton * meter"), 1)
    blend = round(mag(SI.tau_blend_peak, "newton * meter"), 1)
    rated = round(mag(SI.tau_rated, "newton * meter"))
    tb, tk = round(mag(SI.t_blend, "millisecond")), round(mag(SI.dt_tick, "millisecond"))
    fig, ax = new_fig("budget-envelope")
    budget_envelope(ax, rows=((f"{tk} ms step", step, MECHANICAL), (f"{tb} ms C² blend", blend, MECHANICAL)),
                    limit=rated, limit_label=f"rated {rated} N·m", limit_label_offset=-0.025,
                    limit_color=NEUTRAL)
    return fig, [f"{tk}-millisecond step at {step:.1f}", f"{tb}-millisecond", f"blend at {blend:.1f}",
                 f"{rated} newton-meter"]


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 12: Safety Enforcement
# ─────────────────────────────────────────────────────────────────────────────

@margin("12_enforcement", "margin_cbf_qp_cycle_budget_envelope", "budget-envelope", status="ready",
        sources=("EnforcementCycleBudget.qp", ".wcet", ".reserve", ".unloaded", ".deadline"))
def ch12_cycle_budget_envelope():
    # The declared WCET, its QP share and its reserve, against the deadline;
    # the unloaded run is the measured reference.
    E = lego("12_enforcement", "EnforcementCycleBudget")
    us = lambda q: mag(q, "microsecond")
    fig, ax = new_fig("budget-envelope")
    fig.set_size_inches(1.32, 0.85)
    budget_envelope(ax, rows=[("QP solve", us(E.qp), COMP), ("Declared WCET", us(E.wcet), TIME),
                              ("Reserve", us(E.reserve), TIME)],
                    limit=us(E.deadline), limit_label=f"{us(E.deadline):.0f} µs deadline",
                    limit_label_offset=-0.03, value_fmt="{:.0f}",
                    marks=[(us(E.unloaded), f"unloaded {us(E.unloaded):.0f}", NEUTRAL)])
    return fig, []


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 13: Silicon Placement
# ─────────────────────────────────────────────────────────────────────────────

@margin("13_placement", "margin_coupling_timescale_ladder", "hierarchy-ladder", status="ready",
        sources=("InductivePDNVoltageDroop.t_rise", "PlacementDramTiming.t_conflict",
                 "PlacementLoadedRead.queue", "PlacementMachine.tick", "RejectedPlacementThrottle.t_recov"))
def ch13_coupling_timescale_ladder():
    # Coupling mechanisms on one chassis, by the timescale each acts on.
    s = lambda q: mag(q, "second")
    rungs = [("Thermal recovery", s(lego("13_placement", "RejectedPlacementThrottle").t_recov), "s"),
             ("Permission tick", s(lego("13_placement", "PlacementMachine").tick), "ms"),
             ("Loaded DRAM read", s(lego("13_placement", "PlacementLoadedRead").queue), "µs"),
             ("Bank conflict", s(lego("13_placement", "PlacementDramTiming").t_conflict), "ns"),
             ("Droop rise", s(lego("13_placement", "InductivePDNVoltageDroop").t_rise), "ns")]
    scale = {"s": 1.0, "ms": 1e3, "µs": 1e6, "ns": 1e9}
    fmt = lambda v, u: f"{v * scale[u]:.3g} {u}"
    fig, ax = new_fig("hierarchy-ladder")
    ladder(ax, [(f"{name}\n{fmt(v, u)}", v) for name, v, u in rungs], domain="time")
    return fig, []


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 14: Supervisory Intervention
# ─────────────────────────────────────────────────────────────────────────────

@margin("14_intervention", "margin_four_phase_handshake_strip", "sequence-strip",
        sources=("InterventionTimeoutBudget.t_request", ".t_acknowledge", ".tick", ".t_confirm", ".t_handshake"))
def ch14_handshake_strip():
    # Commit is one control tick (AISLE.t_handshake sums the four phases).
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


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 15: Adversarial Verification
# ─────────────────────────────────────────────────────────────────────────────

@margin("15_verification", "margin_fault_injection_coverage_ladder", "hierarchy-ladder", status="schematic")
def ch15_fault_injection_ladder():
    # Ordinal staircase, no physical values. footnotes-margins.md recommends
    # cutting it (it draws a cumulative hierarchy the chapter denies); kept
    # here only because it is still placed.
    fig, ax = new_fig("hierarchy-ladder")
    ladder(ax, [("4: Dyno Stall Clamp", 4), ("3: Power Rail Droop", 3),
                ("2: CAN Babbler Attack", 2), ("1: SRAM Bit-Flips", 1)],
           domain="compute", style="staircase")
    return fig, []


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
