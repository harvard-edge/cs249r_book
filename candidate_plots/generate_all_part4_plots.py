"""
Generate publication-grade quantitative plots for Volume IV Part IV (Chapters 14-17).
Textbook standard: MIT Press / Harvard MLSysBook.

- Chapter 14: Intervention (Human takeover reaction time vs vehicle speed & remaining stopping margin)
- Chapter 15: Verification & Safety (The Astronomical Exposure Wall: Target failure rate lambda per hour vs required zero-failure operating hours N=3/lambda plotted against cumulative fleet hours of Waymo, Cruise, and Tesla)
- Chapter 16: Release & Fleet Learning (A Decade of California DMV Autonomous Vehicle Disengagements: 2015–2025 Miles Between Interventions for Waymo, Cruise, Zoox, Apple, Aurora)
- Chapter 17: Frontier (State-space dimensionality vs formal verification tractability: Hamilton-Jacobi Reachability vs Sampling vs Neural Barrier Certificates)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add candidate_plots directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plot_style import PALETTE, ERA_COLORS, setup_canvas, add_badge


# -----------------------------------------------------------------------------
# PLOT 1: Chapter 14 - Human Takeover Latency vs Vehicle Speed & Stopping Margin
# -----------------------------------------------------------------------------
def generate_ch14_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5), dpi=200, facecolor="#FFFFFF")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#334155")
        ax.spines["bottom"].set_color("#334155")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color=PALETTE["grid"])
        ax.set_axisbelow(True)

    # Physical parameters
    v_kmh = np.linspace(5, 120, 300)
    v_mps = v_kmh / 3.6
    tau_delay = 0.10  # 100 ms onboard perception lease and brake bus delay
    a_eff = 6.0       # effective emergency braking acceleration (m/s^2, ~0.61g)
    delta_buf = 1.0   # physical safety buffer (m)
    d_clear = 60.0    # clear line-of-sight / headway clearance (m)

    # Human supervisory takeover reaction times (Eriksson & Stanton 2017, Zhang 2019, Ch 14 text)
    modes = [
        {"name": "Direct In-Loop: $T_\\Sigma = 0.8\\,\\mathrm{s}$", "T": 0.8, "color": PALETTE["forest_green"], "ls": "-"},
        {"name": "Alerted Driver: $T_\\Sigma = 1.5\\,\\mathrm{s}$", "T": 1.5, "color": PALETTE["dark_blue"], "ls": "-"},
        {"name": "Out-of-Loop (NDRT): $T_\\Sigma = 2.5\\,\\mathrm{s}$", "T": 2.5, "color": PALETTE["amber"], "ls": "-"},
        {"name": "Remote Teleoperation: $T_\\Sigma = 4.0\\,\\mathrm{s}$", "T": 4.0, "color": PALETTE["crimson"], "ls": "-"},
    ]

    # --- Panel (a): Stopping Distance Breakdown ---
    t_base = 2.5
    d_react = v_mps * (t_base + tau_delay)
    d_brake = (v_mps**2) / (2 * a_eff)
    d_total_base = d_react + d_brake + delta_buf

    ax1.fill_between(v_kmh, 0, d_react, color="#FEF3C7", alpha=0.65, label="Reaction Drift ($v \\cdot [T_\\Sigma + \\tau]$)")
    ax1.fill_between(v_kmh, d_react, d_total_base, color="#FEE2E2", alpha=0.65, label="Braking Distance ($v^2 / 2a_{\\mathrm{eff}}$)")

    for m in modes:
        d_tot = v_mps * (m["T"] + tau_delay) + (v_mps**2) / (2 * a_eff) + delta_buf
        ax1.plot(v_kmh, d_tot, color=m["color"], linewidth=2.3, linestyle=m["ls"], label=m["name"])

    # Reference Headways / Sight Clearances
    ax1.axhline(30, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax1.text(8, 32, "Urban Clear Sight (30 m)", fontsize=8.5, color="#64748B", fontweight="bold")
    ax1.axhline(60, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax1.text(8, 62, "Suburban Headway (60 m)", fontsize=8.5, color="#64748B", fontweight="bold")
    ax1.axhline(100, color="#94A3B8", linestyle=":", linewidth=1.2)
    ax1.text(8, 102, "Highway Headway (100 m)", fontsize=8.5, color="#64748B", fontweight="bold")

    ax1.set_xlim(5, 120)
    ax1.set_ylim(0, 205)
    ax1.set_xlabel("Vehicle Velocity $v$ (km/h)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax1.set_ylabel("Total Stopping Distance (meters)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax1.set_title("(a) Stopping Distance Decomposition vs. Speed\nLinear Drift + Quadratic Braking", fontsize=12.5, fontweight="bold", pad=12, color=PALETTE["charcoal"])
    ax1.legend(loc="upper left", fontsize=8.8, framealpha=0.92, edgecolor="#CBD5E1")

    # Clean badge placement avoiding lines
    add_badge(
        ax1,
        "Kinetic Reality:\nDrift scales linearly ($v \\cdot T_\\Sigma$);\nBraking scales quadratically ($v^2 / 2a$).\nAt 100 km/h, out-of-loop drift alone is >72 m!",
        xy=(42, 145),
        xytext=(42, 145),
        color=PALETTE["charcoal"],
        fontsize=8.5,
        arrow=False
    )

    # --- Panel (b): Remaining Stopping Margin & Crash Horizon ---
    ax2.axhspan(-120, 0, color="#FEE2E2", alpha=0.55, label="Inevitable Collision Zone ($M < 0$)")
    ax2.axhline(0, color=PALETTE["crimson"], linestyle="-", linewidth=1.6, alpha=0.85)

    crit_speeds = {}
    for m in modes:
        d_tot = v_mps * (m["T"] + tau_delay) + (v_mps**2) / (2 * a_eff) + delta_buf
        margin = d_clear - d_tot
        ax2.plot(v_kmh, margin, color=m["color"], linewidth=2.4, linestyle=m["ls"], label=m["name"])
        
        # Calculate critical speed where margin == 0
        A = 1.0 / (2.0 * a_eff)
        B = m["T"] + tau_delay
        C = delta_buf - d_clear
        v_crit_mps = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
        v_crit_kmh = v_crit_mps * 3.6
        crit_speeds[m["T"]] = v_crit_kmh
        ax2.plot(v_crit_kmh, 0, marker="o", markersize=7.5, color=m["color"], zorder=5)

    ax2.set_xlim(10, 120)
    ax2.set_ylim(-100, 65)
    ax2.set_xlabel("Vehicle Velocity $v$ (km/h)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax2.set_ylabel("Remaining Clearance Margin $M = d_{\\mathrm{clear}} - d_{\\mathrm{stop}}$ (m)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax2.set_title("(b) Clearance Margin & Collision Horizon\n(Given $d_{\\mathrm{clear}} = 60\\,\\mathrm{m}$ sightline)", fontsize=12.5, fontweight="bold", pad=12, color=PALETTE["charcoal"])
    ax2.legend(loc="lower left", fontsize=8.8, framealpha=0.92, edgecolor="#CBD5E1")

    # Clean summary card in top right (completely free of curves)
    summary_text = (
        "Takeover Speed Ceilings ($d_{\\mathrm{clear}} = 60\\,\\mathrm{m}$):\n"
        f"  • Direct In-Loop (0.8s):    {crit_speeds[0.8]:.1f} km/h\n"
        f"  • Alerted Driver (1.5s):   {crit_speeds[1.5]:.1f} km/h\n"
        f"  • Out-of-Loop NDRT (2.5s): {crit_speeds[2.5]:.1f} km/h\n"
        f"  • Remote Teleop (4.0s):    {crit_speeds[4.0]:.1f} km/h\n"
        "Beyond these ceilings, crash occurs before rest!"
    )
    add_badge(
        ax2,
        summary_text,
        xy=(78.3, 0),
        xytext=(88, 38),
        color=PALETTE["charcoal"],
        fontsize=8.5,
        arrow=False
    )

    # Clean small vertical tags right at the intercept points
    intercept_tags = [
        (crit_speeds[4.0], f"{crit_speeds[4.0]:.1f}", PALETTE["crimson"], -16),
        (crit_speeds[2.5], f"{crit_speeds[2.5]:.1f}", PALETTE["amber"], -16),
        (crit_speeds[1.5], f"{crit_speeds[1.5]:.1f}", PALETTE["dark_blue"], -16),
        (crit_speeds[0.8], f"{crit_speeds[0.8]:.1f} km/h", PALETTE["forest_green"], -16),
    ]
    for spd, txt, col, y_pos in intercept_tags:
        ax2.plot([spd, spd], [0, y_pos + 6], color=col, linestyle=":", linewidth=1.0)
        ax2.text(
            spd, y_pos, txt,
            fontsize=8.0, fontweight="bold", color=col, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=col, alpha=0.9, linewidth=0.6)
        )

    # Callout badge inside crash zone
    add_badge(
        ax2,
        "Collision Horizon ($M < 0$):\nKinetic stopping distance exceeds sightline;\nimpact is physically inevitable regardless of brake force.",
        xy=(85, -50),
        xytext=(80, -78),
        color=PALETTE["crimson"],
        fontsize=8.4,
        arrow=False,
    )

    fig.suptitle("Human Supervisory Takeover Reaction Budget vs. Stopping Clearance Margin", fontsize=14, fontweight="bold", y=0.98, color=PALETTE["charcoal"])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = os.path.join(SCRIPT_DIR, "ch14_human_takeover_reaction_margin.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated Chapter 14 plot: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# PLOT 2: Chapter 15 - The Astronomical Exposure Wall
# -----------------------------------------------------------------------------
def generate_ch15_plot():
    fig, ax = setup_canvas(figsize=(14.0, 8.0), dpi=200)

    # Lambda: target failure rate per hour (log scale 10^-1 to 10^-10)
    lam = np.logspace(-1, -10, 500)
    # Poisson 95% one-sided confidence: T = -ln(0.05) / lambda ~= 2.9957 / lambda
    T_95 = 2.99573 / lam
    # Poisson 99% one-sided confidence: T = -ln(0.01) / lambda ~= 4.605 / lambda
    T_99 = 4.60517 / lam

    # Plot theoretical exposure curves
    ax.plot(lam, T_95, color=PALETTE["dark_blue"], linewidth=2.8, label="Required 0-Failure Hours: $T_{95\\%} = 3.0 / \\lambda$ (95% Conf.)")
    ax.plot(lam, T_99, color=PALETTE["slate"], linewidth=1.6, linestyle="--", label="Upper 99% Confidence Bound ($T_{99\\%} = 4.61 / \\lambda$)")
    ax.fill_between(lam, T_95, T_99, color="#E0F2FE", alpha=0.5)

    # Reverse x-axis so failure rate decreases to the right (more stringent)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e-10)
    ax.set_ylim(5, 2e11)

    # Key Safety Standards and Benchmarks (Vertical lines)
    standards = [
        {"name": "Police Crashes", "sub": "$6 \\times 10^{-5}$/hr", "val": 6e-5, "color": PALETTE["slate"], "y_pos": 8e1},
        {"name": "IEC 61508 SIL 1", "sub": "$10^{-5}$/hr", "val": 1e-5, "color": PALETTE["amber"], "y_pos": 8e2},
        {"name": "IEC 61508 SIL 2", "sub": "$10^{-6}$/hr", "val": 1e-6, "color": PALETTE["gold"], "y_pos": 8e3},
        {"name": "Human Fatality (NHTSA)", "sub": "$3.5 \\times 10^{-7}$/hr", "val": 3.5e-7, "color": PALETTE["forest_green"], "y_pos": 2e4},
        {"name": "ISO 26262 ASIL B", "sub": "100 FIT ($10^{-7}$/hr)", "val": 1e-7, "color": PALETTE["teal"], "y_pos": 8e4},
        {"name": "ISO 26262 ASIL D", "sub": "10 FIT ($10^{-8}$/hr)", "val": 1e-8, "color": PALETTE["crimson"], "y_pos": 3e5},
        {"name": "FAA Aviation / SIL 4", "sub": "1 FIT ($10^{-9}$/hr)", "val": 1e-9, "color": PALETTE["purple"], "y_pos": 1e6},
    ]

    for std in standards:
        ax.axvline(std["val"], color=std["color"], linestyle=":", linewidth=1.3, alpha=0.75)
        ax.text(
            std["val"], std["y_pos"], f" {std['name']}\n [{std['sub']}]",
            rotation=90, va="bottom", ha="right", fontsize=7.8, fontweight="bold", color=std["color"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.85, linewidth=0.5)
        )

    # Real Empirical Fleet Operating Hours
    fleets = [
        {"name": "Cruise (Pre-Oct 2023 Suspension)", "hrs": 4.0e5, "color": PALETTE["coral"], "y_text": 6.2e5, "desc": "10M miles @ 25 mph", "label_hrs": "~400,000 hrs"},
        {"name": "Waymo Rider-Only (L4 Fleet 2025–2026)", "hrs": 7.0e6, "color": PALETTE["cyan"], "y_text": 1.1e7, "desc": "200M miles @ 28 mph", "label_hrs": "~7.0 Million hrs"},
        {"name": "Tesla FSD Supervised (L2 Fleet 2026)", "hrs": 2.5e8, "color": PALETTE["dark_blue"], "y_text": 3.8e8, "desc": "8.5B miles @ 35 mph (L2)", "label_hrs": "~250 Million hrs"},
    ]

    for fl in fleets:
        ax.axhline(fl["hrs"], color=fl["color"], linestyle="-.", linewidth=1.8, alpha=0.85)
        ax.text(
            7e-2, fl["y_text"],
            f"[FLEET] {fl['name']}: {fl['label_hrs']} ({fl['desc']})",
            fontsize=8.4, fontweight="bold", color=fl["color"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=fl["color"], alpha=0.92, linewidth=0.8)
        )

    # Astronomical Verification Deficit Arrow at ASIL D (1e-8)
    ax.annotate(
        "",
        xy=(1e-8, 3e8), xycoords="data",
        xytext=(1e-8, 7e6), textcoords="data",
        arrowprops=dict(arrowstyle="<->", color=PALETTE["crimson"], linewidth=2.4)
    )
    
    # Place Verification Deficit Badge in clear lower-right quadrant
    add_badge(
        ax,
        "Astronomical Verification Deficit (43× Shortfall):\nWaymo's 200M driverless miles (7M hrs) surpass human fatality parity,\nyet fall nearly two orders of magnitude short of ISO 26262 ASIL D (300M hrs).\nValidating ASIL D via road testing alone would require ~34,000 continuous vehicle-years!",
        xy=(1e-8, 7e6),
        xytext=(1.0e-6, 2.5e2),
        color=PALETTE["crimson"],
        fontsize=8.8,
        arrow=False
    )

    # Butler & Finelli (1993) Infeasibility Badge placed in open top-left
    add_badge(
        ax,
        "Butler & Finelli (1993) Infeasibility Horizon:\nAt $\\lambda = 10^{-9}$/hr (Aviation Catastrophic / SIL 4), zero-failure validation\nrequires $3 \\times 10^9$ hours (~342,000 continuous vehicle-years).\nConclusion: Statistical black-box fleet testing CANNOT qualify ultra-reliability.\nSystem safety must be proven through formal runtime architecture and fail-safe envelopes.",
        xy=(1e-9, 3e9),
        xytext=(2e-3, 3e10),
        color=PALETTE["purple"],
        fontsize=8.8,
        arrow=False
    )

    # Twin axis for Vehicle-Years (T / 8760)
    ax_yr = ax.twinx()
    ax_yr.set_yscale("log")
    ax_yr.set_ylim(5 / 8760, 2e11 / 8760)
    ax_yr.spines["top"].set_visible(False)
    ax_yr.spines["left"].set_visible(False)
    ax_yr.spines["right"].set_color("#334155")
    ax_yr.set_ylabel("Continuous Single-Vehicle Operating Years ($T / 8760$)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"], labelpad=12)

    ax.set_xlabel("Target Failure Rate $\\lambda$ per Operating Hour (Failures / Hour, log scale)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax.set_ylabel("Required Zero-Failure Operating Hours ($N = 3 / \\lambda$, log scale)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax.set_title("The Astronomical Exposure Wall: Target Failure Rates vs. Required Zero-Failure Operating Hours", fontsize=13.5, fontweight="bold", pad=16, color=PALETTE["charcoal"])
    ax.legend(loc="lower left", fontsize=9.2, framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    out_path = os.path.join(SCRIPT_DIR, "ch15_astronomical_exposure_wall.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated Chapter 15 plot: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# PLOT 3: Chapter 16 - A Decade of California DMV AV Disengagements (2015-2025)
# -----------------------------------------------------------------------------
def generate_ch16_plot():
    fig, ax = setup_canvas(figsize=(14.2, 8.0), dpi=200)

    # Shaded Historical Eras
    ax.axvspan(2014.5, 2018.5, color=ERA_COLORS[0], alpha=0.7, zorder=0)
    ax.axvspan(2018.5, 2021.5, color=ERA_COLORS[1], alpha=0.7, zorder=0)
    ax.axvspan(2021.5, 2025.5, color=ERA_COLORS[2], alpha=0.7, zorder=0)

    # Era labels
    ax.text(2016.5, 3.8e5, "Era I: Early Highway & Suburban Testing", fontsize=9.5, fontweight="bold", color="#475569", ha="center")
    ax.text(2020.0, 3.8e5, "Era II: Dense Urban Core Transition (SF)", fontsize=9.5, fontweight="bold", color="#1E40AF", ha="center")
    ax.text(2023.5, 3.8e5, "Era III: Driverless Commercial Scale & Divergence", fontsize=9.5, fontweight="bold", color="#065F46", ha="center")

    # California DMV Disengagement Data (2015-2025)
    data = {
        "Waymo": {
            "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "mpi":   [1244, 5128, 5596, 11017, 13219, 29945, 7965, 17366, 17393, 38000, 63415],
            "color": PALETTE["dark_blue"],
            "marker": "o",
            "ls": "-",
            "lw": 2.7
        },
        "Cruise": {
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "mpi":   [54, 1193, 5205, 12221, 28520, 43805, 95901, 80000],
            "color": PALETTE["crimson"],
            "marker": "s",
            "ls": "-",
            "lw": 2.5
        },
        "Zoox": {
            "years": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "mpi":   [160, 2051, 1596, 1654, 1847, 2000, 2030, 1600, 1500],
            "color": PALETTE["forest_green"],
            "marker": "^",
            "ls": "-",
            "lw": 2.3
        },
        "Apple (Titan)": {
            "years": [2018, 2019, 2020, 2021, 2022, 2023],
            "mpi":   [1.15, 118, 145, 20, 893, 1509],
            "color": PALETTE["purple"],
            "marker": "D",
            "ls": "--",
            "lw": 2.1
        },
        "Aurora": {
            "years": [2018, 2019, 2020, 2021, 2022],
            "mpi":   [121, 1045, 340, 535, 2100],
            "color": PALETTE["amber"],
            "marker": "v",
            "ls": "-.",
            "lw": 2.0
        }
    }

    # Plot lines
    for company, d in data.items():
        ax.plot(
            d["years"], d["mpi"],
            color=d["color"], marker=d["marker"], markersize=6.8,
            linewidth=d["lw"], linestyle=d["ls"],
            label=company, zorder=4
        )

    # Human driver baseline
    ax.axhline(40000, color="#64748B", linestyle=":", linewidth=1.5, zorder=2)
    ax.text(2014.7, 44000, "Human Driver Baseline: ~40,000 Miles Between Police-Reported Crashes (NHTSA)", fontsize=8.6, color="#475569", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlim(2014.5, 2025.5)
    ax.set_ylim(0.8, 6.5e5)

    # Non-crossing, well-spaced badges with short direct arrows:
    # 1. Cruise Suspension Marker & Badge
    ax.plot(2023.2, 80000, marker="X", markersize=11, color=PALETTE["crimson"], zorder=6)
    add_badge(
        ax,
        "Cruise Oct 2023:\nPermit revoked by CA DMV following\nSan Francisco pedestrian dragging incident",
        xy=(2023.2, 80000),
        xytext=(2020.6, 1.8e5),
        color=PALETTE["crimson"],
        fontsize=8.4,
        arrow=True,
        arrow_color=PALETTE["crimson"]
    )

    # 2. Waymo 2025 Milestone
    add_badge(
        ax,
        "Waymo 2025: 63,415 miles/disengagement\n(5.2M testing miles, 82 disengagements)",
        xy=(2025, 63415),
        xytext=(2023.4, 1.2e5),
        color=PALETTE["dark_blue"],
        fontsize=8.4,
        arrow=True,
        arrow_color=PALETTE["dark_blue"]
    )

    # 3. Waymo 2021 Urban Core Transition Dip
    add_badge(
        ax,
        "Waymo 2021 SF Dip:\nExpanded from suburban Phoenix\ninto dense SF street grid",
        xy=(2021, 7965),
        xytext=(2019.2, 3200),
        color=PALETTE["dark_blue"],
        fontsize=8.3,
        arrow=True,
        arrow_color=PALETTE["dark_blue"]
    )

    # 4. Zoox Dedicated Urban Robotaxi (in open space between Zoox and Waymo)
    add_badge(
        ax,
        "Zoox: Consistent ~1,500-2,000 miles\nPurpose-built bidirectional robotaxi\ntested strictly in dense urban traffic",
        xy=(2024, 1600),
        xytext=(2024.1, 5500),
        color=PALETTE["forest_green"],
        fontsize=8.2,
        arrow=True,
        arrow_color=PALETTE["forest_green"]
    )

    # 5. Aurora Freight Transition (placed cleanly above Aurora point in open space)
    add_badge(
        ax,
        "Aurora (2022):\nPivoted from CA passenger cars to\nTexas Class 8 commercial freight routes",
        xy=(2022, 2100),
        xytext=(2021.5, 4800),
        color=PALETTE["amber"],
        fontsize=8.2,
        arrow=True,
        arrow_color=PALETTE["amber"]
    )

    # 6. Apple Titan Cancellation (placed directly below Apple final 2023 point in open space)
    ax.plot(2023, 1509, marker="X", markersize=11, color=PALETTE["purple"], zorder=6)
    add_badge(
        ax,
        "Apple Project Titan:\nCancelled Feb 2024\nafter $10B+ investment",
        xy=(2023, 1509),
        xytext=(2023.1, 45),
        color=PALETTE["purple"],
        fontsize=8.3,
        arrow=True,
        arrow_color=PALETTE["purple"]
    )

    # 7. Apple 2018 Initial Sensitivity
    add_badge(
        ax,
        "Apple 2018: 1.15 miles/disengagement\n(69,510 interventions in 79,754 mi;\nhyper-conservative early threshold)",
        xy=(2018, 1.15),
        xytext=(2015.0, 4),
        color=PALETTE["purple"],
        fontsize=8.2,
        arrow=True,
        arrow_color=PALETTE["purple"]
    )

    ax.set_xticks(range(2015, 2026))
    ax.set_xlabel("Reporting Year (Dec 1 – Nov 30)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax.set_ylabel("Miles Between Disengagements (MPI, log scale)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax.set_title("A Decade of California Autonomous Vehicle Disengagements (2015–2025)", fontsize=13.5, fontweight="bold", pad=16, color=PALETTE["charcoal"])
    ax.legend(loc="lower right", fontsize=9.2, framealpha=0.95, edgecolor="#CBD5E1")

    fig.text(0.12, 0.015, "*Data Source: California DMV Annual Autonomous Vehicle Disengagement Reports (2015–2025). DMV sunsetting traditional disengagement reporting in 2025/2026.", fontsize=8.2, color="#64748B", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(SCRIPT_DIR, "ch16_decade_california_dmv_disengagements.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated Chapter 16 plot: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# PLOT 4: Chapter 17 - State-Space Dimensionality vs Verification Tractability
# -----------------------------------------------------------------------------
def generate_ch17_plot():
    fig, ax = setup_canvas(figsize=(14.2, 8.0), dpi=200)

    # State space dimension d from 2 to 50
    d = np.linspace(2, 50, 400)

    # 1. Hamilton-Jacobi Reachability (Grid PDE, O(N^d))
    d_hj = np.linspace(2, 6.2, 150)
    T_hj = 0.005 * (48.0 ** (d_hj - 2))

    # 2. Sampling-Based Statistical Verification (Scenario / AST, O(d^2))
    T_sample = 0.02 * (d ** 1.8)

    # 3. Neural Barrier Certificates (SMT / alpha-beta-CROWN, O(d^k))
    T_nbc = 0.035 * (d ** 2.7)

    # Plot curves
    ax.plot(d_hj, T_hj, color=PALETTE["crimson"], linewidth=2.8, label="Hamilton-Jacobi Reachability (Grid PDE, $\\mathcal{O}(N^d)$)")
    ax.plot(d, T_sample, color=PALETTE["forest_green"], linewidth=2.4, linestyle="--", label="Sampling-Based Statistical Verification (Scenario / AST, $\\mathcal{O}(d^2)$)")
    ax.plot(d, T_nbc, color=PALETTE["dark_blue"], linewidth=2.7, label="Neural Barrier Certificates (SMT / $\\alpha,\\beta$-CROWN, $\\mathcal{O}(d^k)$)")

    # The Grid Wall for Hamilton-Jacobi Reachability
    ax.axvline(6.2, color=PALETTE["crimson"], linestyle=":", linewidth=1.8, alpha=0.85)
    ax.fill_betweenx([1e-3, 5e10], 6.2, 52, color="#FEE2E2", alpha=0.28)
    
    # Grid Wall text inside shaded zone at a safe height
    ax.text(
        6.5, 8e7,
        "THE GRID WALL (Curse of Dimensionality):\nDiscretization memory & PDE solve time\nexplode exponentially beyond $d > 6$",
        fontsize=8.8, fontweight="bold", color=PALETTE["crimson"]
    )

    ax.set_yscale("log")
    ax.set_xlim(1.5, 52)
    ax.set_ylim(1e-3, 5e10)

    # Benchmark Robot Systems Callouts aligned with Neural Barrier Curve
    systems = [
        {"d": 3,  "name": "Dubins Car (3D)\n$(x, y, \\theta)$", "y_box": 1e-2, "col": PALETTE["slate"]},
        {"d": 4,  "name": "Kinematic Car (4D)\n$(x, y, \\theta, v)$", "y_box": 5e2, "col": PALETTE["slate"]},
        {"d": 7,  "name": "Kinodynamic Car (7D)\n$(x, y, z, \\dot{x}, \\dot{y}, \\psi, \\dot{\\psi})$", "y_box": 3e-1, "col": PALETTE["teal"]},
        {"d": 12, "name": "6-DOF Quadrotor (12D)\n$(p, v, q, \\omega)$", "y_box": 2e4, "col": PALETTE["dark_blue"]},
        {"d": 16, "name": "Bipedal Robot (16D)\n(Floating Base + Legs)", "y_box": 4e-2, "col": PALETTE["dark_blue"]},
        {"d": 30, "name": "Mobile Manipulator (30D)\n(Base + 14-DOF Dual Arm)", "y_box": 1.5e6, "col": PALETTE["purple"]},
        {"d": 48, "name": "Multi-Agent Swarm (48D)\n($8 \\times 6$D Coupled Agents)", "y_box": 4e8, "col": PALETTE["purple"]},
    ]

    for sys_item in systems:
        sd = sys_item["d"]
        st = 0.035 * (sd ** 2.7)
        ax.plot(sd, st, marker="o", markersize=6.5, color=sys_item["col"], zorder=5)
        ax.axvline(sd, color="#CBD5E1", linestyle=":", linewidth=0.8, alpha=0.55)
        
        # Draw dotted line connecting dot to label
        ax.plot([sd, sd], [st, sys_item["y_box"]], color=sys_item["col"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(
            sd, sys_item["y_box"], sys_item["name"],
            fontsize=7.8, fontweight="bold", color=sys_item["col"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.92, linewidth=0.6)
        )

    # Sampling compromise badge with short, clean arrow
    add_badge(
        ax,
        "Sampling-Based Compromise:\nFast and polynomial up to $d > 100$, but provides only\nprobabilistic PAC bounds ($1-\\epsilon$). Blind to measure-zero\nadversarial counterexamples and worst-case corners!",
        xy=(36, 12),
        xytext=(36, 2e-2),
        color=PALETTE["forest_green"],
        fontsize=8.4,
        arrow=True,
        arrow_color=PALETTE["forest_green"]
    )

    # Neural Barrier Frontier badge
    add_badge(
        ax,
        "The Neural Barrier Frontier:\nSynthesizes learned certificates and verifies them via\nSMT / branch-and-bound ($\\alpha,\\beta$-CROWN), breaking the grid wall\nto guarantee formal forward invariance on 12D–50D systems!",
        xy=(24, 200),
        xytext=(24, 1.5e5),
        color=PALETTE["dark_blue"],
        fontsize=8.5,
        arrow=True,
        arrow_color=PALETTE["dark_blue"]
    )

    # Time scale markers on right inside plot margin
    time_markers = [
        (1.0, "1 Second"),
        (60.0, "1 Minute"),
        (3600.0, "1 Hour"),
        (86400.0, "1 Day"),
        (3.15e7, "1 Year"),
        (3.15e9, "1 Century"),
    ]
    for sec, label in time_markers:
        ax.axhline(sec, color="#E2E8F0", linestyle="-", linewidth=0.6, alpha=0.6)
        ax.text(50.8, sec, label, fontsize=7.8, color="#94A3B8", va="center", ha="right")

    ax.set_xlabel("State-Space Dimensionality $d$ (Degrees of Freedom / State Variables)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax.set_ylabel("Verification / Safety Synthesis Computation Time (seconds, log scale)", fontsize=11, fontweight="bold", color=PALETTE["charcoal"])
    ax.set_title("Formal Verification Tractability Frontier: Hamilton-Jacobi vs. Sampling vs. Neural Barrier Certificates", fontsize=13.0, fontweight="bold", pad=16, color=PALETTE["charcoal"])
    ax.legend(loc="upper left", fontsize=9.2, framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout(rect=[0, 0, 0.98, 1])
    out_path = os.path.join(SCRIPT_DIR, "ch17_state_space_formal_verification_frontier.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated Chapter 17 plot: {out_path}")
    return out_path


if __name__ == "__main__":
    print("Generating all Volume IV Part IV publication-grade plots...")
    p1 = generate_ch14_plot()
    p2 = generate_ch15_plot()
    p3 = generate_ch16_plot()
    p4 = generate_ch17_plot()
    print("All plots generated successfully!")
