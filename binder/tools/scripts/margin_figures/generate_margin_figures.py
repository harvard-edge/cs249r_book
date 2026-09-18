#!/usr/bin/env python3
"""generate_margin_figures.py - Generate the canonical margin visual figures for Volume IV.

Outputs production vector SVGs into books/vol4/<chapter>/images/svg/margin_<slug>.svg.
Uses the canonical margin devices in binder/tools/figures/margin/devices.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from binder.tools.figures.margin.devices import (
    C, MEM, COMP, DATA, NET, RED, REDFILL, GRID, INK, SEL, TIME, MECHANICAL,
    new_fig, ladder, knee, sparkline, roofline, ironbar, taxonomy, blast,
    budget_envelope, sequence_strip, causal_chain, save
)

CONTENTS = ROOT / "books" / "vol4"

def out_path(chap: str, name: str) -> str:
    return str(CONTENTS / chap / "images" / "svg" / f"{name}.svg")

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 01: The Causal Boundary
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch01():
    chap = "01_boundary"
    # Fig 01-M1: Timescale Separation Ladder
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("10 s: Governance", 10.0),
        ("1 s: Deliberation", 1.0),
        ("100 ms: Trajectory", 0.1),
        ("1 ms: Real-Time Reflex", 0.001),
        ("20 us: PWM Switching", 0.00002)
    ]
    ladder(ax, tiers, domain='time', style='bars')
    save(fig, out_path(chap, "margin_timescale_separation_ladder"))

    # Fig 01-M2: Compounding Covariate Shift Sparkline
    fig, ax = new_fig('sparkline-trend')
    t = np.linspace(0, 1, 100)
    open_loop = 0.05 + 0.35 * t
    closed_loop = 0.05 + 0.90 * (t ** 2.2)
    ax.plot(t, closed_loop, color=RED, lw=1.35)
    ax.plot(t, open_loop, color=DATA, lw=1.35)
    ax.fill_between(t, open_loop, closed_loop, color=REDFILL, alpha=0.35)
    ax.text(0.95, closed_loop[-1], "O(T^2)", color=RED, fontsize=5.0, fontweight="bold", ha="right", va="bottom")
    ax.text(0.95, open_loop[-1] - 0.08, "O(T)", color=DATA, fontsize=5.0, fontweight="bold", ha="right", va="top")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_covariate_shift_divergence"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 02: The Physical Body
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch02():
    chap = "02_body"
    # Fig 02-M1: Reflected Rotor Inertia Knee (N*)
    fig, ax = new_fig('scale-anchor')
    N = np.linspace(5, 100, 200)
    tau_m = 6.5
    J_rotor = 1.2e-4
    J_load = 0.18
    alpha = (N * tau_m) / (J_load + (N**2) * J_rotor)
    N_star = np.sqrt(J_load / J_rotor) # 38.73
    peak_alpha = (N_star * tau_m) / (J_load + (N_star**2) * J_rotor) # ~699 rad/s^2
    
    m = N <= N_star
    ax.plot(N[m], alpha[m], color=MECHANICAL, lw=1.35)
    ax.plot(N[~m], alpha[~m], color=RED, lw=1.35)
    ax.axvspan(N_star, 100, color=REDFILL, alpha=0.25)
    ax.plot(N_star, peak_alpha, "o", color=MECHANICAL, ms=3.5)
    ax.axvline(N_star, color=GRID, ls="--", lw=0.6)
    ax.text(N_star - 2, 715, "N*=39\nPeak", color=MECHANICAL, fontsize=4.8, fontweight="bold", ha="right", multialignment="right")
    ax.text(75, 420, "rotor\ndominated", color=RED, fontsize=4.6, ha="center", multialignment="center")
    ax.set_xlim(5, 100); ax.set_ylim(0, 800)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_reflected_rotor_inertia_knee"))

    # Fig 02-M2: PDN Voltage Droop Envelope (Droop vs Headroom)
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("Idle Droop", 1.5, COMP),
        ("60A Surge", 12.8, RED)
    ]
    budget_envelope(ax, rows=rows, limit=7.0, style='burn', limit_label="Headroom 7V")
    save(fig, out_path(chap, "margin_pdn_voltage_droop_envelope"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 03: The Cognitive Brain
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch03():
    chap = "03_brain"
    # Fig 03-M1: Action Chunk Receding Windows & Blending
    fig, ax = new_fig('sequence-strip')
    steps = [
        ("H=32", COMP),
        ("K=12", COMP),
        ("Blend", DATA)
    ]
    sequence_strip(ax, steps=steps, bracket=(0, 2), bracket_label="Temporal Blend")
    save(fig, out_path(chap, "margin_action_chunk_windows_strip"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 04: The Real-Time Nervous System
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch04():
    chap = "04_nervous"
    # Fig 04-M1: Fieldbus 1.0 ms Control Cycle Budget Envelope
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("EtherCAT", 30.4, DATA),
        ("CAN-FD", 1123.2, RED)
    ]
    budget_envelope(ax, rows=rows, limit=1000.0, style='burn', limit_label="1.0 ms Tick")
    save(fig, out_path(chap, "margin_fieldbus_cycle_budget_envelope"))

    # Fig 04-M2: Substrate Fault Blast Radius
    fig, ax = new_fig('blast-radius')
    ax.plot(0.08, 0.52, "s", color=RED, ms=10)
    ax.text(0.08, 0.28, "NPU 50A", color=RED, fontsize=4.8, fontweight="bold", ha="center")
    victims = ["Voltage droop", "Thermal sag", "DMA starvation", "Clock jitter"]
    ys = np.linspace(0.12, 0.90, 4)
    for y, label in zip(ys, victims):
        ax.annotate("", xy=(0.58, y), xytext=(0.14, 0.52),
                    arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.65))
        ax.plot(0.58, y, "o", color=MEM, ms=5)
        ax.text(0.66, y, label, color=INK, fontsize=4.4, va="center", ha="left")
    ax.set_xlim(0, 1.35); ax.set_ylim(0, 1.0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_substrate_blast_radius"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 05: Physical Data Systems
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch05():
    chap = "05_data"
    # Fig 05-M1: Sensory Ingestion Bandwidth Cliff
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("NVMe PCIe: 3500 MB/s", 3500),
        ("4x Cam Raw: 968 MB/s", 968),
        ("SATA SSD: 550 MB/s", 550),
        ("USB 3.0: 400 MB/s", 400)
    ]
    ladder(ax, tiers, domain='bandwidth', wall=True)
    save(fig, out_path(chap, "margin_sensory_ingestion_bandwidth_ladder"))

    # Fig 05-M2: Facility Demonstration Yield Burn-Down
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("Usable Data", 14.7, DATA),
        ("Shift Time", 60.0, TIME)
    ]
    budget_envelope(ax, rows=rows, limit=60.0, style='burn', limit_label="60 min Shift")
    save(fig, out_path(chap, "margin_facility_yield_burndown_envelope"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 06: Simulation & Policy Training
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch06():
    chap = "06_training"
    # Fig 06-M1: Multi-Rate Generative Inference Deadlines
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("Flow Match 2-step", 17.3, DATA),
        ("DDIM 16-step", 40.4, RED)
    ]
    budget_envelope(ax, rows=rows, limit=20.0, style='burn', limit_label="20 ms (50 Hz)")
    save(fig, out_path(chap, "margin_generative_inference_deadlines_envelope"))

    # Fig 06-M2: Mechanical Fatigue Lifespan in RL
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("RL Trials: 4.0M Cycles", 4000),
        ("Gearhead L10: 1.5M", 1500),
        ("Tendon Cable: 0.2M", 200)
    ]
    ladder(ax, tiers, domain='rate', wall=True, color=MECHANICAL)
    save(fig, out_path(chap, "margin_mechanical_fatigue_ladder"))

    # Fig 06-M3: Contact Stiffness Impact Force Cliff
    fig, ax = new_fig('scale-anchor')
    x = np.linspace(0, 1.0, 100)
    f_sim = 94 * (x ** 0.8)
    f_real = 94 + 3676 * (x ** 3.5)
    ax.plot(x, f_real, color=RED, lw=1.35)
    ax.plot(x, f_sim, color=DATA, lw=1.35)
    ax.axhline(250, color=GRID, ls="--", lw=0.6)
    ax.text(0.12, 500, "250N limit", color=INK, fontsize=4.6)
    ax.text(0.12, 1400, "Sim: 94N", color=DATA, fontsize=5.0, fontweight="bold")
    ax.text(0.95, 3850, "Real: 3770N", color=RED, fontsize=5.0, fontweight="bold", ha="right")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 4200)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_contact_stiffness_cliff_knee"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 07: Empirical Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch07():
    chap = "07_evaluation"
    # Fig 07-M1: Non-Asymptotic Zero-Failure Reliability Ladder
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("2995 runs: 99.9%", 2995),
        ("299 runs: 99.0%", 299),
        ("59 runs: 95.0%", 59),
        ("20 runs: 86.1%", 20)
    ]
    ladder(ax, tiers, domain='rate', style='bars', color=DATA)
    save(fig, out_path(chap, "margin_nonasymptotic_reliability_ladder"))

    # Fig 07-M2: Extreme Value Shock vs Gaussian
    fig, ax = new_fig('sparkline-trend')
    t = np.linspace(0, 1, 100)
    p_gauss = np.exp(-8 * t)
    p_frechet = 1.0 / (1.0 + 3.0 * t) ** 1.8
    ax.plot(t, p_frechet, color=RED, lw=1.35)
    ax.plot(t, p_gauss, color=TIME, lw=1.35)
    ax.fill_between(t, p_gauss, p_frechet, color=REDFILL, alpha=0.35)
    ax.text(0.95, p_frechet[-1] + 0.08, "Heavy tail", color=RED, fontsize=5.0, fontweight="bold", ha="right", va="bottom")
    ax.text(0.75, 0.18, "Gaussian", color=TIME, fontsize=4.8, ha="right", va="bottom")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.1)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_extreme_value_shock_sparkline"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 08: Sensor Perception
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch08():
    chap = "08_perception"
    # Fig 08-M1: Phase Margin Erosion vs Perception Age
    fig, ax = new_fig('scale-anchor')
    tau = np.linspace(0, 100, 200)
    pm = 45.0 - 0.573 * tau
    tau_crit = 78.5
    
    m = tau <= tau_crit
    ax.plot(tau[m], pm[m], color=DATA, lw=1.35)
    ax.plot(tau[~m], pm[~m], color=RED, lw=1.35)
    ax.axhline(0, color=GRID, ls="--", lw=0.6)
    ax.axvline(tau_crit, color=RED, ls=":", lw=0.65)
    ax.plot(tau_crit, 0, "o", color=RED, ms=3.3)
    ax.text(tau_crit - 3, 7, "78.5 ms", color=RED, fontsize=4.8, fontweight="bold", ha="right")
    ax.text(20, 32, "Safe PM", color=DATA, fontsize=4.8)
    ax.text(88, -7, "Chatter", color=RED, fontsize=4.8, ha="center")
    ax.set_xlim(0, 100); ax.set_ylim(-18, 50)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_phase_margin_erosion_knee"))

    # Fig 08-M2: Sensory DMA Ingress Burst Slack
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("4x 1080p DMA", 622.0, DATA),
        ("4x 4K DMA", 2488.0, RED)
    ]
    budget_envelope(ax, rows=rows, limit=850.0, style='burn', limit_label="850 us Slack")
    save(fig, out_path(chap, "margin_dma_ingress_slack_envelope"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 09: Spatial Memory
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch09():
    chap = "09_memory"
    # Fig 09-M1: Active Negative Evidence Raycasting
    fig, ax = new_fig('causal-chain')
    labels = ["Ray\ncast", "Free\nspace", "Hit\nsurf", "Evict\nghost"]
    causal_chain(ax, labels=labels, colors=[NET, DATA, DATA, RED])
    save(fig, out_path(chap, "margin_active_negative_evidence_chain"))

    # Fig 09-M2: Volumetric Raycasting Bandwidth Explosion
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("SoC Bus Limit: 51.2 GB/s", 51.2),
        ("Dense TSDF: 44.2 GB/s", 44.2),
        ("Sparse VDB: 0.044 GB/s", 0.044)
    ]
    ladder(ax, tiers, domain='bandwidth', wall=True)
    save(fig, out_path(chap, "margin_volumetric_raycasting_bandwidth_ladder"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 10: Grounded Intent
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch10():
    chap = "10_intent"
    # Fig 10-M1: Operating Frequency Execution Ladder
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("20 kHz: PWM Drives", 20000),
        ("1 kHz: Fieldbus Loop", 1000),
        ("100 Hz: Whole-Body MPC", 100),
        ("10 Hz: Policy Chunks", 10),
        ("0.5 Hz: VLA Deliberation", 0.5)
    ]
    ladder(ax, tiers, domain='rate', style='bars')
    save(fig, out_path(chap, "margin_operating_frequency_ladder"))

    # Fig 10-M2: Dynamic Reachability Time Shortfall
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("Transit Time", 900.0, RED)
    ]
    budget_envelope(ax, rows=rows, limit=450.0, style='burn', limit_label="Lease 450 ms")
    save(fig, out_path(chap, "margin_dynamic_reachability_shortfall_envelope"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 11: Trajectory Planning
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch11():
    chap = "11_planning"
    # Fig 11-M1: Reflected Rotor Inertia Seam Torque Spike Knee
    fig, ax = new_fig('scale-anchor')
    t_blend = np.linspace(1, 60, 200) # ms
    tau_peak = 50.0 / t_blend
    m = t_blend >= 20.0
    ax.plot(t_blend[m], tau_peak[m], color=DATA, lw=1.35)
    ax.plot(t_blend[~m], tau_peak[~m], color=RED, lw=1.35)
    ax.axhline(5.0, color=GRID, ls="--", lw=0.6)
    ax.text(45, 7.5, "Limit: 5 Nm", color=INK, fontsize=4.6)
    ax.plot(1.0, 50.0, "o", color=RED, ms=3.3)
    ax.text(4.0, 48.0, "1 ms: 50 Nm", color=RED, fontsize=4.8, fontweight="bold")
    ax.text(48.0, 14.0, "50 ms: 1 Nm", color=DATA, fontsize=4.8, fontweight="bold", ha="center")
    ax.set_xlim(-1, 65); ax.set_ylim(0, 58)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, out_path(chap, "margin_reflected_rotor_inertia_torque_knee"))

    # Fig 11-M2: Action Chunk Latency Tail Strip
    fig, ax = new_fig('sequence-strip')
    steps = [
        ("Chunk T", COMP),
        ("P99 Gap", RED),
        ("P99.99997", DATA)
    ]
    sequence_strip(ax, steps=steps, bracket=(1, 2), bracket_label="Replenish")
    save(fig, out_path(chap, "margin_action_chunk_latency_tail_strip"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 12: Safety Enforcement
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch12():
    chap = "12_enforcement"
    # Fig 12-M1: Quadratic Stopping Clearance Burn-Down
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("v = 1.2 m/s", 388.0, DATA),
        ("v = 2.4 m/s", 1312.0, RED)
    ]
    budget_envelope(ax, rows=rows, limit=1200.0, style='burn', limit_label="Sightline 1.2m")
    save(fig, out_path(chap, "margin_quadratic_stopping_clearance_envelope"))

    # Fig 12-M2: Embedded CBF-QP Cycle Budget
    fig, ax = new_fig('budget-envelope')
    rows = [
        ("QP Active-Set", 175.0, DATA)
    ]
    budget_envelope(ax, rows=rows, limit=1000.0, style='burn', limit_label="1000 us Tick")
    save(fig, out_path(chap, "margin_cbf_qp_cycle_budget_envelope"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 13: Silicon Placement
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch13():
    chap = "13_placement"
    # Fig 13-M1: Execution Determinism Across Silicon Domains
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("Cloud Host (>150 ms)", 150000000),
        ("Linux RT (20 ms)", 20000000),
        ("Cortex-M7 (5 us)", 5000),
        ("Lockstep MCU (50 ns)", 50)
    ]
    ladder(ax, tiers, domain='time', style='bars')
    save(fig, out_path(chap, "margin_execution_determinism_ladder"))

    # Fig 13-M2: Memory Bus Contention Tail Latency
    fig, ax = new_fig('scale-anchor')
    knee(ax, knee_frac=0.75, style='dashed', pct_label="9.0 ms")
    ax.text(18, 4.0, "P50: 4.6 ms", ha="center", va="center", color=DATA, fontsize=5.0)
    ax.text(80, 26.0, "P99.9: 9.7 ms", ha="left", va="center", color=RED, fontsize=4.8, fontweight="bold")
    save(fig, out_path(chap, "margin_memory_contention_tail_hockey_stick_knee"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 14: Supervisory Intervention
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch14():
    chap = "14_intervention"
    # Fig 14-M1: Supervisory Escalation Ladder (Linear Authority Tiers)
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("4: Galvanic STO Cutoff", 4),
        ("3: State Token Revocation", 3),
        ("2: Software Envelope Clamp", 2),
        ("1: Advisory Haptic Chime", 1)
    ]
    ladder(ax, tiers, domain='certified', style='staircase')
    save(fig, out_path(chap, "margin_supervisory_escalation_ladder"))

    # Fig 14-M2: Four-Phase Handshake Window
    fig, ax = new_fig('sequence-strip')
    steps = [
        ("Req\n10ms", TIME),
        ("Ack\n5ms", TIME),
        ("Cmt\n1ms", TIME),
        ("Cnf\n15ms", TIME)
    ]
    sequence_strip(ax, steps=steps, bracket=(0, 3), bracket_label="31 ms Total")
    save(fig, out_path(chap, "margin_four_phase_handshake_strip"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 15: Adversarial Verification
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch15():
    chap = "15_verification"
    # Fig 15-M1: Cross-Layer Fault Injection Coverage Ladder
    fig, ax = new_fig('hierarchy-ladder')
    tiers = [
        ("4: Dyno Stall Clamp", 4),
        ("3: Power Rail Droop", 3),
        ("2: CAN Babbler Attack", 2),
        ("1: SRAM Bit-Flips", 1)
    ]
    ladder(ax, tiers, domain='compute', style='staircase')
    save(fig, out_path(chap, "margin_fault_injection_coverage_ladder"))

    # Fig 15-M2: Butler & Finelli Poisson Exposure Wall
    fig, ax = new_fig('scale-anchor')
    knee(ax, knee_frac=0.60, style='shaded', pct_label="10^-9/h")
    ax.text(25, 4.0, "Drone: 3000 h", ha="center", va="center", color=DATA, fontsize=4.8)
    ax.text(78, 22.0, "AV: 342k years", ha="center", va="center", color=RED, fontsize=4.8, fontweight="bold")
    save(fig, out_path(chap, "margin_poisson_exposure_wall_scale_anchor_knee"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 16: Deployment Release
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch16():
    chap = "16_release"
    # Fig 16-M1: ASIL Decomposition Privilege Boundary
    fig, ax = new_fig('taxonomy-mini')
    taxonomy(ax, hot=3, style='quadrant')
    ax.text(0.46, 0.46, "QM(D)\nNeural", ha="center", va="center", color=INK, fontsize=4.8, fontweight="bold")
    ax.text(1.46, 1.46, "ASIL D\nEnforcer", ha="center", va="center", color="white", fontsize=4.8, fontweight="bold")
    ax.text(0.46, 1.46, "Isolated", ha="center", va="center", color="#888888", fontsize=4.4)
    ax.text(1.46, 0.46, "Veto", ha="center", va="center", color="#888888", fontsize=4.4)
    save(fig, out_path(chap, "margin_asil_decomposition_taxonomy"))

    # Fig 16-M2: Cryptographic Hardware Seal
    fig, ax = new_fig('sequence-strip')
    steps = [
        ("eFuse", NET),
        ("Ed25519", DATA),
        ("Audit", TIME),
        ("Power", COMP)
    ]
    sequence_strip(ax, steps=steps, bracket=(0, 3), bracket_label="Root of Trust")
    save(fig, out_path(chap, "margin_cryptographic_seal_strip"))

# ─────────────────────────────────────────────────────────────────────────────
# Chapter 17: The Epistemic Frontier
# ─────────────────────────────────────────────────────────────────────────────

def gen_ch17():
    chap = "17_frontier"
    # Fig 17-M1: Observational Indistinguishability Causal Split
    fig, ax = new_fig('causal-chain')
    labels = ["State\nsplit", "Identical\ntelemetry", "Deadline\npassed", "Physical\nfracture"]
    causal_chain(ax, labels=labels, colors=[TIME, TIME, RED, RED])
    save(fig, out_path(chap, "margin_observational_indistinguishability_chain"))

    # Fig 17-M2: Battery Specific Energy Diminishing Returns
    fig, ax = new_fig('scale-anchor')
    m_batt = np.linspace(0, 40, 100)
    # Rational transport equation from §17.5.1: e_spec*m_b / (P_comp + P_mech*(m_dry + m_b))
    t_endurance = (160.0 * m_batt) / (150.0 + 42.0 * (15.0 + m_batt))
    ax.plot(m_batt, t_endurance, color=COMP, lw=1.35)
    ax.axhline(3.8, color=RED, ls="--", lw=0.65)
    ax.text(38, 3.95, "Ceiling: 3.8 h", color=RED, fontsize=5.0, fontweight="bold", ha="right")
    ax.plot(12.0, 2.3, "o", color=COMP, ms=3.3)
    ax.text(14.0, 2.0, "12 kg: 2.3 h", color=COMP, fontsize=4.8)
    ax.set_xlim(0, 42); ax.set_ylim(0, 4.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.55)
    ax.set_xticks([0, 20, 40])
    ax.set_xticklabels(["0", "20", "40 kg"], fontsize=4.0, color="#555555")
    ax.set_yticks([0, 2, 4])
    ax.set_yticklabels(["0", "2", "4 h"], fontsize=4.0, color="#555555")
    save(fig, out_path(chap, "margin_battery_diminishing_returns_knee"))

def main():
    print("Generating Volume IV Margin Visual Figures (Refined)...")
    gen_ch01()
    gen_ch02()
    gen_ch03()
    gen_ch04()
    gen_ch05()
    gen_ch06()
    gen_ch07()
    gen_ch08()
    gen_ch09()
    gen_ch10()
    gen_ch11()
    gen_ch12()
    gen_ch13()
    gen_ch14()
    gen_ch15()
    gen_ch16()
    gen_ch17()
    print("All 33 Volume IV margin visual figures generated successfully!")

if __name__ == "__main__":
    main()
