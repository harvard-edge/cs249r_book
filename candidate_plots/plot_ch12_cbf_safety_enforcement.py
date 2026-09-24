#!/usr/bin/env python3
"""
Chapter 12: Control Barrier Function (CBF-QP) Runtime Safety Filtering
and Real-Time Execution Jitter Contention.

Demonstrates:
(a) Phase-space trajectory deflection: Invariant safe set C, kinetic barrier boundary,
    unshielded neural policy crash, and CBF-QP active projection along boundary.
(b) Control acceleration input timeline u(t): Nominal proposal vs. CBF override.
(c) 1 kHz real-time loop execution latency: PREEMPT_RT determinism vs. standard Linux jitter.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add script directory to sys.path to find plot_style
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_style import PALETTE


def generate_ch12_cbf_plot():
    # -------------------------------------------------------------------------
    # 1. Physical Simulation Parameters
    # -------------------------------------------------------------------------
    dt = 0.001  # 1 kHz control loop (1 ms)
    t_max = 1.6
    a_max = 2.0  # max available braking deceleration (m/s^2)
    d_min = 0.05  # safe standoff distance margin (m)
    gamma = 4.0  # CBF class-K gain

    # Trajectory 1: Unshielded Neural Policy
    d_un = 1.0
    v_un = 0.5
    u_nom_val = 1.2  # m/s^2 forward acceleration
    traj_un_t, traj_un_d, traj_un_v, traj_un_u = [], [], [], []

    for t in np.arange(0, t_max, dt):
        traj_un_t.append(t)
        traj_un_d.append(d_un)
        traj_un_v.append(v_un)
        traj_un_u.append(u_nom_val)
        if d_un <= 0:
            break
        v_un += u_nom_val * dt
        d_un -= v_un * dt

    # Trajectory 2: CBF-QP Filtered Policy
    d_cbf = 1.0
    v_cbf = 0.5
    gamma = 28.0  # high fidelity boundary barrier stiffness
    traj_cbf_t, traj_cbf_d, traj_cbf_v, traj_cbf_u, traj_cbf_unom = [], [], [], [], []

    for t in np.arange(0, t_max, dt):
        # Kinetic barrier function: h(d, v) = d - v^2 / (2 * a_max) - d_min
        h = d_cbf - (v_cbf**2) / (2.0 * a_max) - d_min
        
        # Nominal unshielded command
        u_nom = 1.2 if (v_cbf > 0.02 and d_cbf > d_min) else 0.0
        
        # CBF constraint: dot_h >= -gamma * h
        # dot_h = -v - (v / a_max) * u >= -gamma * h
        # u <= -a_max * (1.0 - gamma * h / max(0.05, v))
        u_cbf_bound = -a_max * (1.0 - gamma * max(0.0, h) / max(0.08, v_cbf))
        u_filtered = min(u_nom, u_cbf_bound)
        u_act = np.clip(u_filtered, -a_max, u_nom)
        
        if v_cbf <= 0.015 and d_cbf <= d_min + 0.025:
            u_act = 0.0
            v_cbf = 0.0

        traj_cbf_t.append(t)
        traj_cbf_d.append(d_cbf)
        traj_cbf_v.append(v_cbf)
        traj_cbf_u.append(u_act)
        traj_cbf_unom.append(u_nom)

        v_cbf = max(0.0, v_cbf + u_act * dt)
        d_cbf -= v_cbf * dt

    traj_un_t = np.array(traj_un_t)
    traj_un_d = np.array(traj_un_d)
    traj_un_v = np.array(traj_un_v)
    traj_un_u = np.array(traj_un_u)

    traj_cbf_t = np.array(traj_cbf_t)
    traj_cbf_d = np.array(traj_cbf_d)
    traj_cbf_v = np.array(traj_cbf_v)
    traj_cbf_u = np.array(traj_cbf_u)
    traj_cbf_unom = np.array(traj_cbf_unom)

    # -------------------------------------------------------------------------
    # 2. Timing / Jitter Simulation (1000 us loop)
    # -------------------------------------------------------------------------
    np.random.seed(42)
    n_samples = len(traj_cbf_t)
    
    # PREEMPT_RT / MCU: tightly bounded around 95 us with rare small jitter (< 180 us)
    rt_latencies = np.random.normal(loc=95.0, scale=12.0, size=n_samples)
    rt_latencies = np.clip(rt_latencies, 65.0, 185.0)

    # Standard Linux (CFS scheduler, non-RT): median ~140 us, but heavy tails and preemption spikes
    non_rt_latencies = np.random.normal(loc=135.0, scale=28.0, size=n_samples)
    non_rt_latencies = np.clip(non_rt_latencies, 80.0, 450.0)
    # Inject scheduling spikes, page faults, and multi-thread contention
    spike_indices = [65, 180, 310, 490, 720, 930, 1120]
    spike_magnitudes = [1850.0, 2400.0, 1450.0, 3100.0, 2200.0, 1750.0, 2900.0]
    for idx, mag in zip(spike_indices, spike_magnitudes):
        if idx < n_samples:
            non_rt_latencies[idx] = mag
            if idx + 1 < n_samples:
                non_rt_latencies[idx + 1] = mag * 0.45

    # -------------------------------------------------------------------------
    # 3. Canvas & Layout Setup
    # -------------------------------------------------------------------------
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.9

    fig = plt.figure(figsize=(15.2, 7.2), dpi=300, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0],
                           wspace=0.25, hspace=0.36)

    ax1 = fig.add_subplot(gs[:, 0])  # Phase Portrait
    ax2 = fig.add_subplot(gs[0, 1])  # Control Acceleration u(t)
    ax3 = fig.add_subplot(gs[1, 1])  # Loop Execution Latency Timeline

    # -------------------------------------------------------------------------
    # Panel (a): Phase Space Portrait (d vs. v)
    # -------------------------------------------------------------------------
    v_grid = np.linspace(0.0, 1.8, 400)
    d_barrier = (v_grid**2) / (2.0 * a_max) + d_min

    # Shading
    ax1.fill_betweenx(v_grid, d_barrier, 1.25, color="#F0FDF4", alpha=0.85, label="Invariant Safe Set $\\mathcal{C}$ ($h \\geq 0$)")
    ax1.fill_betweenx(v_grid, 0, d_barrier, color="#FEF2F2", alpha=0.75, label="Inevitable Collision Set ($h < 0$)")

    # Invariant Boundary line
    ax1.plot(d_barrier, v_grid, color=PALETTE["crimson"], linewidth=2.4, linestyle="-",
             label=r"Kinetic Boundary $\partial\mathcal{C}: d = \frac{v^2}{2a_{\max}} + d_{\min}$")

    # Unshielded Trajectory
    ax1.plot(traj_un_d, traj_un_v, color="#DC2626", linewidth=2.2, linestyle="--",
             label="Unshielded Neural Policy (Collision Breach)", zorder=4)
    # Unshielded crash point
    ax1.scatter([traj_un_d[-1]], [traj_un_v[-1]], color="#B91C1C", s=90, marker="X", zorder=6)
    ax1.text(traj_un_d[-1] + 0.04, traj_un_v[-1] + 0.04,
             f"Collision Impact\n($v={traj_un_v[-1]:.2f}$ m/s at $d=0$)",
             fontsize=8.5, fontweight="bold", color="#B91C1C", va="bottom")

    # Filtered CBF Trajectory
    ax1.plot(traj_cbf_d, traj_cbf_v, color=PALETTE["dark_blue"], linewidth=2.6, linestyle="-",
             label="CBF-QP Filtered Policy (Safe Invariance)", zorder=5)
    # Start point
    ax1.scatter([traj_cbf_d[0]], [traj_cbf_v[0]], color=PALETTE["dark_blue"], s=70, marker="o", zorder=6)
    ax1.text(traj_cbf_d[0] + 0.02, traj_cbf_v[0] - 0.05, "Initial State $(d_0, v_0)$",
             fontsize=8.5, fontweight="bold", color=PALETTE["dark_blue"], ha="left")

    # Safe stop point
    ax1.scatter([traj_cbf_d[-1]], [traj_cbf_v[-1]], color=PALETTE["forest_green"], s=85, marker="o", zorder=6)
    ax1.text(traj_cbf_d[-1] + 0.03, traj_cbf_v[-1] + 0.06,
             f"CBF Safe Rest\n($d={traj_cbf_d[-1]:.2f}$ m, $v=0$)",
             fontsize=8.5, fontweight="bold", color=PALETTE["forest_green"])

    # Vector intervention callout at engagement point
    t_eng_candidates = np.where(traj_cbf_u < 0.0)[0]
    t_eng_idx = t_eng_candidates[0] if len(t_eng_candidates) > 0 else 0
    d_eng = traj_cbf_d[t_eng_idx]
    v_eng = traj_cbf_v[t_eng_idx]
    ax1.scatter([d_eng], [v_eng], color=PALETTE["amber"], s=80, zorder=6)
    
    # Vector arrows showing projection
    ax1.annotate("", xy=(d_eng - 0.14, v_eng + 0.16), xytext=(d_eng, v_eng),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.8))
    ax1.text(d_eng - 0.15, v_eng + 0.19, r"Nominal $\mathbf{u}_{\mathrm{nom}}$",
             fontsize=8.2, fontweight="bold", color="#DC2626")

    ax1.annotate("", xy=(d_eng - 0.13, v_eng - 0.16), xytext=(d_eng, v_eng),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["forest_green"], lw=1.8))
    ax1.text(d_eng - 0.22, v_eng - 0.18, r"Projected $\mathbf{u}^*$ (Braking)",
             fontsize=8.2, fontweight="bold", color=PALETTE["forest_green"])

    # Place CBF Tangential Deflection explanation box in clear space at bottom-center (d=0.45, v=0.40)
    ax1.annotate("CBF Tangential Deflection:\n" + r"$\mathbf{u}^* = \arg\min \|\mathbf{u} - \mathbf{u}_{\mathrm{nom}}\|^2$" + "\n" + r"s.t. $\dot{h} + \gamma h \geq 0$",
                 xy=(d_eng, v_eng), xytext=(0.42, 0.40),
                 arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.2, linestyle="--"),
                 fontsize=8.5, fontweight="bold", color=PALETTE["charcoal"],
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.8))

    ax1.set_xlim(-0.02, 1.2)
    ax1.set_ylim(-0.05, 1.85)
    ax1.set_xlabel("Clearance Distance to Constraint $d$ (meters)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Approach Velocity $v$ (m/s)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax1.set_title(r"(a) Kinetic barrier invariant set and safe trajectory deflection in phase space $(d, v)$",
                  fontsize=11.2, fontweight="bold", color="#1E293B", pad=10)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=PALETTE["grid"])
    ax1.legend(loc="upper right", fontsize=8.4, framealpha=0.95, edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # Panel (b): Control Acceleration Input u(t)
    # -------------------------------------------------------------------------
    ax2.plot(traj_un_t, traj_un_u, color="#DC2626", linestyle="--", linewidth=1.8,
             label=r"Unconstrained Proposal $u_{\mathrm{nom}}(t)$")
    ax2.plot(traj_cbf_t, traj_cbf_u, color=PALETTE["dark_blue"], linestyle="-", linewidth=2.3,
             label=r"Enforced Actuation $u^*(t)$")

    # Shaded area of intervention
    ax2.fill_between(traj_cbf_t, traj_cbf_u, traj_cbf_unom, where=(traj_cbf_u < traj_cbf_unom),
                     color="#FEF3C7", alpha=0.7, label=r"CBF-QP Intervention $\Delta u$")

    # Markings for engagement and stop
    ax2.axvline(traj_cbf_t[t_eng_idx], color=PALETTE["amber"], linestyle=":", linewidth=1.4)
    ax2.text(traj_cbf_t[t_eng_idx] + 0.02, -0.6, f"Barrier Engagement\n($t={traj_cbf_t[t_eng_idx]:.2f}$ s)",
             fontsize=8.0, fontweight="bold", color=PALETTE["amber"])

    t_stop_cands = np.where(traj_cbf_v <= 0.02)[0]
    t_stop_idx = t_stop_cands[0] if len(t_stop_cands) > 0 else -1
    ax2.axvline(traj_cbf_t[t_stop_idx], color=PALETTE["forest_green"], linestyle=":", linewidth=1.4)
    ax2.text(traj_cbf_t[t_stop_idx] - 0.02, 0.4, f"Zero Velocity Rest\n($t={traj_cbf_t[t_stop_idx]:.2f}$ s)",
             fontsize=8.0, fontweight="bold", color=PALETTE["forest_green"], ha="right")

    ax2.set_xlim(0.0, 1.52)
    ax2.set_ylim(-2.4, 2.2)
    ax2.set_ylabel(r"Acceleration $u(t)$ ($\mathrm{m/s^2}$)", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax2.set_title("(b) Control input intervention: nominal proposal vs. active barrier clamp",
                  fontsize=11.0, fontweight="bold", color="#1E293B", pad=10)
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=PALETTE["grid"])
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.95, edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # Panel (c): 1 kHz Loop Latency & Jitter Timeline
    # -------------------------------------------------------------------------
    ax3.plot(traj_cbf_t, non_rt_latencies, color=PALETTE["amber"], alpha=0.85, linewidth=1.1,
             label="Standard Linux (Non-RT Generic Kernel)")
    ax3.plot(traj_cbf_t, rt_latencies, color=PALETTE["teal"], alpha=0.95, linewidth=1.4,
             label="Bare-Metal MCU / PREEMPT_RT Linux")

    # 1 kHz Loop Deadline line
    ax3.axhline(1000.0, color="#DC2626", linestyle="--", linewidth=1.6, zorder=5)
    ax3.text(0.02, 1120.0, "1 kHz Actuation Loop Deadline ($1000\\,\\mu\\mathrm{s}$ / $1.0\\,\\mathrm{ms}$)",
             fontsize=8.4, fontweight="bold", color="#DC2626")

    # Annotate scheduling spike overrun
    ax3.annotate("Kernel Jitter Spike:\nLoop Deadline Overrun\n(Triggers Watchdog Fault)",
                 xy=(traj_cbf_t[490], non_rt_latencies[490]), xytext=(traj_cbf_t[490] + 0.08, 2500),
                 arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.2),
                 fontsize=8.0, fontweight="bold", color="#B91C1C",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#FECDD3", lw=0.8))

    # Annotate MCU RT headroom
    ax3.text(0.85, 260.0, "PREEMPT_RT / MCU Headroom:\nWCET $185\\,\\mu\\mathrm{s}$ ($>80\\%$ margin)",
             fontsize=8.0, fontweight="bold", color=PALETTE["teal"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=0.8))

    ax3.set_xlim(0.0, 1.52)
    ax3.set_ylim(0, 3600)
    ax3.set_xlabel("Elapsed Time $t$ (seconds)", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax3.set_ylabel(r"Cycle Execution Time ($\mu\mathrm{s}$)", fontsize=10.0, fontweight="bold", color="#1E293B")
    ax3.set_title(r"(c) Real-time 1 kHz loop execution: PREEMPT_RT determinism vs. standard Linux jitter",
                  fontsize=11.0, fontweight="bold", color="#1E293B", pad=10)
    ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color=PALETTE["grid"])
    ax3.legend(loc="upper right", fontsize=8.2, framealpha=0.95, edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # Save & Export
    # -------------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "ch12_cbf_safety_enforcement.png")
    plt.savefig(out_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")

    repo_root = os.path.abspath(os.path.join(out_dir, ".."))
    book_png_dir = os.path.join(repo_root, "books/vol4/12_enforcement/images/png")
    os.makedirs(book_png_dir, exist_ok=True)
    book_path = os.path.join(book_png_dir, "fig12_cbf_safety_enforcement.png")
    plt.savefig(book_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close()

    print(f"Generated Chapter 12 plot:")
    print(f"  Candidate: {out_path}")
    print(f"  Book PNG:  {book_path}")
    return out_path


if __name__ == "__main__":
    generate_ch12_cbf_plot()
