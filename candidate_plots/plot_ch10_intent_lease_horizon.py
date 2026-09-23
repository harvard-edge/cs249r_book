#!/usr/bin/env python3
"""
Chapter 10: Grounded Intent
Plot: The Target-Evidence Horizon and Dynamic Reachability Filter
Two-panel publication-grade systems plot:
- Panel (a): Target-Evidence Validity Horizon (tau_ev) vs. Target Drift Velocity (v_drift)
  Showing the hyperbola tau_ev <= (r_tol - e_0) / v_drift across grasp tolerances (5mm, 15mm, 30mm)
  contrasted against VLA deliberation latencies (650 ms high-tail inference, 180 ms nominal, 20 ms local tracker).
- Panel (b): Dynamic Reachability Transit Time Floor vs. Commanded Distance
  Showing transit floor t_transit = d / v_tcp vs. remaining lease window (tau_lease),
  illustrating the kinematic & dynamic admission boundary (ERR_DYNAMIC_TIME_INSUFFICIENT).
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Style helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import PALETTE

def add_clean_badge(ax, text, xy, xytext, color="#1E293B", fontsize=8.4, arrow=True, arrow_color="#94A3B8", ha="center"):
    """Adds a clean rounded badge with an arrow without crossing lines."""
    arrowprops = dict(
        arrowstyle="->",
        color=arrow_color,
        linewidth=0.9,
        shrinkA=3,
        shrinkB=3,
    ) if arrow else None
    
    bbox = dict(
        boxstyle="round,pad=0.35",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        alpha=0.95,
        linewidth=0.8,
    )
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontweight="medium",
        color=color,
        bbox=bbox,
        arrowprops=arrowprops,
        va="center",
        ha=ha,
        zorder=10,
    )

def generate_ch10_plot():
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 9.6), dpi=300, facecolor="#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")
    
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=PALETTE["grid"])
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")

    # ----------------------------------------------------
    # PANEL (a): Target-Evidence Validity Horizon vs. Drift Speed
    # Equation: tau_ev = (r_tol - e_0) / v_drift
    # e_0 = 3 mm initial measurement error
    # ----------------------------------------------------
    e_0 = 3.0  # mm
    v_drift = np.linspace(5.0, 350.0, 500)  # mm/s (from slow conveyor/breeze to fast motion)
    
    tolerances = [
        {"r_tol": 8.0,  "label": r"Precision Insertion ($r_{\mathrm{tol}} = 8\,\mathrm{mm}$)", "color": PALETTE["crimson"], "ls": "-"},
        {"r_tol": 15.0, "label": r"Standard Mug Grasp ($r_{\mathrm{tol}} = 15\,\mathrm{mm}$)", "color": PALETTE["dark_blue"], "ls": "-"},
        {"r_tol": 30.0, "label": r"Coarse Tray Pick ($r_{\mathrm{tol}} = 30\,\mathrm{mm}$)", "color": PALETTE["forest_green"], "ls": "-"},
    ]
    
    for t in tolerances:
        tau_ev = (t["r_tol"] - e_0) / v_drift * 1000.0  # in ms
        ax1.plot(v_drift, tau_ev, label=t["label"], color=t["color"], linewidth=2.3, linestyle=t["ls"])

    # Highlight specific conveyor points from textbook example
    # Fast conveyor: 200 mm/s, r_tol=15mm -> tau = (15-3)/200 = 60 ms
    # Slow conveyor: 20 mm/s, r_tol=15mm -> tau = (15-3)/20 = 600 ms
    ax1.scatter([20.0, 200.0], [600.0, 60.0], color=PALETTE["dark_blue"], s=60, zorder=5)
    
    # Reasoning Model Inference Latency Baselines
    ax1.axhline(650.0, color="#DC2626", linestyle=":", linewidth=1.5, alpha=0.85)
    ax1.text(345.0, 675.0, "VLA High-Tail Inference Horizon (650 ms)", color="#DC2626",
             fontsize=8.5, fontweight="bold", ha="right")

    ax1.axhline(180.0, color=PALETTE["amber"], linestyle="--", linewidth=1.4, alpha=0.85)
    ax1.text(345.0, 195.0, "Nominal Edge VLA Inference (180 ms)", color=PALETTE["amber"],
             fontsize=8.5, fontweight="bold", ha="right")

    ax1.axhline(20.0, color="#059669", linestyle="-.", linewidth=1.4, alpha=0.85)
    ax1.text(345.0, 24.0, "Local Visual Tracker Loop (20 ms, 50 Hz)", color="#059669",
             fontsize=8.5, fontweight="bold", ha="right",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=0.7, alpha=0.95))

    # Inadmissible hazard zone shading (tau_ev < 180 ms)
    ax1.axhspan(0, 180, color="#FEF2F2", alpha=0.55, zorder=0)
    ax1.text(105.0, 125.0, "VLA Stale Evidence Hazard Zone\nTarget drifts beyond tolerance during inference",
             ha="center", va="center", fontsize=8.4, fontweight="bold", color="#B91C1C",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#FECDD3", lw=0.8))

    add_clean_badge(ax1, "Fast Conveyor (200 mm/s):\nEvidence valid for only 60 ms;\n650 ms VLA proposal is stale\nbefore execution begins!",
                    xy=(200, 60), xytext=(225, 360),
                    color=PALETTE["crimson"], fontsize=8.2, arrow=True, arrow_color=PALETTE["crimson"])

    add_clean_badge(ax1, "Slow Conveyor (20 mm/s):\nEvidence valid for 600 ms;\nfits 180 ms nominal VLA inference,\nbut vulnerable to tail jitter.",
                    xy=(20, 600), xytext=(90, 800),
                    color=PALETTE["dark_blue"], fontsize=8.2, arrow=True, arrow_color=PALETTE["dark_blue"])

    ax1.set_xlim(5, 350)
    ax1.set_ylim(0, 1100)
    ax1.set_xlabel(r"Target Physical Drift Velocity $v_{\mathrm{drift}}$ (mm/s)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax1.set_ylabel(r"Target-Evidence Horizon $\tau_{\mathrm{ev}}$ (ms)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax1.set_title(r"(a) Target-evidence validity horizon $\tau_{\mathrm{ev}} \leq (r_{\mathrm{tol}} - e_0) / v_{\mathrm{drift}}$ under physical drift",
                  fontsize=11.5, fontweight="bold", color="#1E293B", pad=12)
    ax1.legend(loc="upper right", fontsize=8.8, framealpha=0.95, edgecolor="#CBD5E1")

    # ----------------------------------------------------
    # PANEL (b): Dynamic Reachability Transit Time Floor vs. Distance
    # Minimum transit time: t_transit >= d / v_max
    # Actuator limits: v_tcp = 0.50 m/s (500 mm/s), a_max = 2.0 m/s^2
    # ----------------------------------------------------
    d_mm = np.linspace(10, 450, 400)  # distance to target (mm)
    v_tcp = 500.0  # mm/s (0.5 m/s)
    a_tcp = 2000.0  # mm/s^2 (2.0 m/s^2)
    
    # Exact trapezoidal / triangular minimum motion time:
    # d_crit for reaching max velocity: d_crit = v_max^2 / a_max
    d_crit = (v_tcp**2) / a_tcp  # 125 mm
    t_min_s = np.where(
        d_mm <= d_crit,
        2.0 * np.sqrt(d_mm / a_tcp),  # triangular profile
        (d_mm / v_tcp) + (v_tcp / a_tcp)  # trapezoidal profile
    )
    t_min_ms = t_min_s * 1000.0

    # Theoretical linear floor (infinite acceleration): d / v_max
    t_linear_ms = (d_mm / v_tcp) * 1000.0

    ax2.plot(d_mm, t_min_ms, label=r"Dynamic Transit Floor (Trapezoidal: $a_{\max} = 2\,\mathrm{m/s}^2, v_{\max} = 0.5\,\mathrm{m/s}$)",
             color=PALETTE["crimson"], linewidth=2.4)
    ax2.plot(d_mm, t_linear_ms, label=r"Kinematic Velocity Limit ($d / v_{\max}$ with infinite acceleration)",
             color="#64748B", linewidth=1.6, linestyle="--")

    # Remaining lease horizons
    lease_horizons = [
        {"tau": 150.0, "name": r"Untracked Short Lease ($\tau_{\mathrm{lease}} = 150\,\mathrm{ms}$)", "color": PALETTE["amber"]},
        {"tau": 450.0, "name": r"Tracked Extended Lease ($\tau_{\mathrm{lease}} = 450\,\mathrm{ms}$)", "color": PALETTE["forest_green"]},
    ]

    for lh in lease_horizons:
        ax2.axhline(lh["tau"], color=lh["color"], linestyle="-.", linewidth=1.5, alpha=0.85, label=lh["name"])
        # Find intersection distance
        # solve t_min_ms = lh["tau"]
        idx = np.argmin(np.abs(t_min_ms - lh["tau"]))
        d_intersect = d_mm[idx]
        ax2.scatter(d_intersect, lh["tau"], color=lh["color"], s=75, zorder=5)

    # Shaded dynamically inadmissible region for 150 ms lease
    d_admit_150 = d_mm[np.argmin(np.abs(t_min_ms - 150.0))]  # ~35 mm
    ax2.axvspan(d_admit_150, 450, color="#FEF2F2", alpha=0.50, zorder=0)
    ax2.text(280.0, 80.0, "Dynamically Inadmissible for 150 ms Lease\n" + r"$\frac{d}{v_{\max}} > \tau_{\mathrm{lease}} \to$ ERR_DYNAMIC_TIME_INSUFFICIENT",
             ha="center", va="center", fontsize=8.5, fontweight="bold", color="#B91C1C",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#FECDD3", lw=0.8))

    add_clean_badge(ax2, f"Short Lease Reach Limit ({d_admit_150:.0f} mm):\nUntracked lease admits only\nfinal approach maneuvers (<{d_admit_150/10:.1f} cm).",
                    xy=(d_admit_150, 150.0), xytext=(85, 260),
                    color=PALETTE["amber"], fontsize=8.2, arrow=True, arrow_color=PALETTE["amber"])

    d_admit_450 = d_mm[np.argmin(np.abs(t_min_ms - 450.0))]  # ~102 mm
    add_clean_badge(ax2, f"Tracked Lease Reach Limit ({d_admit_450:.0f} mm):\nActive visual tracking extends\nreach manifold to {d_admit_450/10:.1f} cm.",
                    xy=(d_admit_450, 450.0), xytext=(280, 580),
                    color=PALETTE["forest_green"], fontsize=8.2, arrow=True, arrow_color=PALETTE["forest_green"])

    ax2.set_xlim(10, 450)
    ax2.set_ylim(0, 950)
    ax2.set_xlabel("Required Tool-Center-Point (TCP) Travel Distance $d$ (mm)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax2.set_ylabel(r"Minimum Motion Duration $t_{\mathrm{transit}}$ (ms)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax2.set_title(r"(b) Dynamic reachability filter: motion duration floor vs. remaining lease window $\tau_{\mathrm{lease}}$",
                  fontsize=11.5, fontweight="bold", color="#1E293B", pad=12)
    ax2.legend(loc="upper left", fontsize=8.6, framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "ch10_intent_lease_horizon.png")
    plt.savefig(out_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    
    repo_root = os.path.abspath(os.path.join(out_dir, ".."))
    book_png_dir = os.path.join(repo_root, "books/vol4/10_intent/images/png")
    os.makedirs(book_png_dir, exist_ok=True)
    book_path = os.path.join(book_png_dir, "fig10_intent_lease_horizon.png")
    plt.savefig(book_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close()
    
    print(f"Generated Chapter 10 plot:")
    print(f"  Candidate: {out_path}")
    print(f"  Book PNG:  {book_path}")
    return out_path

if __name__ == "__main__":
    generate_ch10_plot()
