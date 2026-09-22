#!/usr/bin/env python3
"""
Fifty Years of Microprocessor Scaling and Physical AI Compute (1975-2025).

Adapted from the landmark Karl Rupp / Hennessy & Patterson microprocessor trend data
specifically for robotics and physical artificial intelligence systems.

Key Historical Insights:
1. 1975-2004: Classical scaling (Moore's law + Dennard scaling). Frequency and compute
   scaled exponentially, allowing robotics to transition from offboard mainframes
   (Shakey) to onboard VMEbus/x86 controllers (Navlab, Stanley).
2. 2004: The Dennard Scaling Wall (~3.8 GHz). Single-thread performance stalled,
   forcing the architectural split between 1 kHz deterministic microcontrollers
   (Nervous System) and throughput-oriented multi-core CPUs.
3. 2012-2020: The Deep Learning & Edge NPU Pivot. Server GPUs (300-700W) violate mobile
   robot battery/thermal payloads (15-60W ceiling). Edge NPUs (Jetson Xavier/Orin) deliver
   a 1,000x surge in onboard TOPS under the strict mobile power envelope.
4. 2023-2025: The VLA Cadence Frontier. Autoregressive VLAs hit the Cognitive Deliberation
   Wall (1-3 Hz), overcome by Diffusion/Flow Matching action chunking reaching 30-50 Hz.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0

# Create figure with 2 vertically stacked subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 11), sharex=True, dpi=200, facecolor="#FFFFFF",
    gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.14}
)

# -----------------------------------------------------------------------------
# TOP PANEL: 50-Year Microprocessor Scaling & Mobile Robot Thermal Envelope
# -----------------------------------------------------------------------------
ax1.set_facecolor("#FFFFFF")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#E2E8F0")

# Shaded Eras
ax1.axvspan(1974, 2004, color="#F8FAFC", alpha=0.9, zorder=0)
ax1.axvspan(2004, 2012, color="#EFF6FF", alpha=0.7, zorder=0)
ax1.axvspan(2012, 2020, color="#ECFDF5", alpha=0.7, zorder=0)
ax1.axvspan(2020, 2026, color="#FFFBEB", alpha=0.7, zorder=0)

# Era labels at the very top with background boxes
era_style = dict(boxstyle="square,pad=0.25", facecolor="#FFFFFF", edgecolor="none", alpha=0.9)
ax1.text(1989, 1.2e12, "DENNARD SCALING ERA\n(Clock frequency & IPC scaling)",
         ha="center", va="top", fontsize=9, fontweight="bold", color="#475569", bbox=era_style)
ax1.text(2008, 1.2e12, "MULTICORE GAP\n(Thermal wall)",
         ha="center", va="top", fontsize=9, fontweight="bold", color="#1E40AF", bbox=era_style)
ax1.text(2016, 1.2e12, "EMBEDDED GPU\n(Deep learning)",
         ha="center", va="top", fontsize=9, fontweight="bold", color="#065F46", bbox=era_style)
ax1.text(2023, 1.2e12, "PHYSICAL AI NPUs\n(Heterogeneous SoCs)",
         ha="center", va="top", fontsize=9, fontweight="bold", color="#92400E", bbox=era_style)

# Microprocessor Data (Karl Rupp / Stanford / IEEE dataset milestones)
years = [1975, 1978, 1982, 1985, 1989, 1993, 1997, 2000, 2004, 2006, 2008, 2010, 2012, 2015, 2018, 2020, 2022, 2024]

# Transistors per die (count)
transistors = [
    5e3, 2.9e4, 1.34e5, 2.75e5, 1.2e6, 3.1e6, 7.5e6, 4.2e7, 1.25e8, 2.9e8,
    7.3e8, 1.17e9, 2.6e9, 5.5e9, 1.2e10, 2.8e10, 5.4e10, 1.5e11
]

# Frequency (MHz)
freq_years = [1975, 1978, 1982, 1986, 1989, 1993, 1997, 2000, 2004, 2006, 2010, 2015, 2020, 2024]
frequency_mhz = [
    2, 5, 8, 16, 25, 66, 200, 1500, 3800, 2930, 3330, 4000, 4200, 4500
]

# Desktop/Server TDP (Watts) vs Mobile Robot Payload Power Ceiling (15-60W)
tdp_years = [1975, 1982, 1989, 1993, 1997, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
server_tdp = [1.5, 3, 5, 15, 35, 75, 130, 125, 150, 250, 400, 700]

# Plot Transistors
ax1.plot(years, transistors, color="#004B87", linewidth=2.4, marker="o", markersize=5,
         label="Transistors per Die (Moore's Law)")

# Plot Frequency
ax1.plot(freq_years, frequency_mhz, color="#A51C30", linewidth=2.2, linestyle="-.", marker="s", markersize=4.5,
         label="Clock Frequency [MHz] (Dennard Wall at 3.8 GHz in 2004)")

# Plot Server TDP
ax1.plot(tdp_years, server_tdp, color="#D97706", linewidth=2.0, linestyle="--", marker="^", markersize=4.5,
         label="Server/Desktop Peak TDP [W] (Prohibitive for mobile robots)")

# Mobile Robot Power Envelope Band (15W - 60W)
ax1.axhspan(15, 60, color="#10B981", alpha=0.18, label="Mobile Robot Thermal & Battery Budget (15–60 W)")

# Dennard Scaling breakdown vertical marker
ax1.axvline(2004, color="#A51C30", linestyle=":", linewidth=1.5, alpha=0.7)
ax1.annotate(
    "2004: Dennard Scaling Breakdown\nClock speed saturates (~3.8 GHz)\nRobotics loops decouple to MCUs",
    xy=(2004, 3800), xytext=(1982, 1e7),
    arrowprops=dict(arrowstyle="->", color="#A51C30", lw=1.2),
    fontsize=9.5, fontweight="bold", color="#A51C30",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF1F2", edgecolor="#FECDD3", lw=0.8)
)

ax1.set_yscale("log")
ax1.set_ylim(0.2, 2e12)
ax1.set_ylabel("Quantity / Frequency / Power\n(Logarithmic Scale)", fontsize=11, fontweight="bold", color="#1E293B")
ax1.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)
ax1.set_title("Fifty Years of Microprocessor Scaling & Physical AI Compute (1975–2025)",
              fontsize=14, fontweight="bold", color="#0F172A", pad=14)

# -----------------------------------------------------------------------------
# BOTTOM PANEL: Embodied Compute Density (TOPS) vs Closed-Loop Control Cadence
# -----------------------------------------------------------------------------
ax2.set_facecolor("#FFFFFF")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#E2E8F0")

# Shaded Eras (matching top)
ax2.axvspan(1974, 2004, color="#F8FAFC", alpha=0.9, zorder=0)
ax2.axvspan(2004, 2012, color="#EFF6FF", alpha=0.7, zorder=0)
ax2.axvspan(2012, 2020, color="#ECFDF5", alpha=0.7, zorder=0)
ax2.axvspan(2020, 2026, color="#FFFBEB", alpha=0.7, zorder=0)

# Physical AI Onboard Compute Density in Mobile Budget (Peak GFLOPS / TOPS at <= 60W)
robot_compute_years = [1975, 1986, 1995, 2005, 2012, 2018, 2022, 2025]
robot_compute_gflops = [1e-3, 1e-2, 0.13, 6.0, 150.0, 3.2e4, 2.75e5, 1.5e6]

# Closed-Loop Autonomy & Deliberation Rate (Hz)
cadence_years = [1975, 1986, 2005, 2015, 2023, 2025]
vision_cadence_hz = [0.001, 0.5, 10.0, 30.0, 2.0, 50.0]

# Plot Onboard Mobile Compute Density
ax2.plot(robot_compute_years, robot_compute_gflops, color="#1E793C", linewidth=2.5, marker="o", markersize=6,
         label="Onboard Compute at ≤60 W Payload [GFLOPS / TOPS-equiv] (10⁹× expansion)")

# Plot Perception / Deliberation Action Loop Rate
ax2.plot(cadence_years, vision_cadence_hz, color="#6B21A8", linewidth=2.2, linestyle="--", marker="D", markersize=5.5,
         label="Cognitive Perception-to-Action Cadence [Hz] (Deliberation loop)")

# Constant 1 kHz Fieldbus deterministic servo line
ax2.axhline(1000, color="#475569", linestyle=":", linewidth=1.4, alpha=0.8)
ax2.text(1975.5, 1500, "1,000 Hz Deterministic Fieldbus Motor Servo Loop (CAN / EtherCAT)",
         fontsize=9, fontweight="bold", color="#475569")

# Annotate Milestones on Bottom Plot with carefully adjusted coordinates to prevent any overlap
milestones = [
    (1975, 1e-3, "1975: Shakey Robot\nOffboard PDP-10 tether\n~10⁻³ Hz closed-loop", (1976.5, 1e-4)),
    (1986, 1e-2, "1986: CMU Navlab 1\nOnboard 68020 VMEbus\nALVINN neural steering (0.5 Hz)", (1978, 5e1)),
    (2005, 6.0, "2005: Stanley (DARPA)\nIntel Pentium M\n10 Hz vision-in-the-loop", (1995, 3e1)),
    (2018, 3.2e4, "2018: Jetson Xavier\n30W Edge NPU (32 TOPS)", (2007, 2e5)),
    (2023, 2.0, "2023: RT-2 / OpenVLA\nAutoregressive VLA wall (1–3 Hz)", (2014, 0.05)),
    (2025, 1.5e6, "2024–2025: Jetson Orin / Thor\n275–1,000 TOPS at 60W\nπ₀ Flow Matching (50 Hz chunks)", (2012, 5e6)),
]

for x, y, text, xytext in milestones:
    ax2.annotate(
        text, xy=(x, y), xytext=xytext,
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=0.9),
        fontsize=8.5, fontweight="medium", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.8, alpha=0.95),
        zorder=10
    )

ax2.set_yscale("log")
ax2.set_ylim(2e-5, 2e7)
ax2.set_xlim(1974, 2026)
ax2.set_xlabel("Year (1975–2025)", fontsize=11, fontweight="bold", color="#1E293B")
ax2.set_ylabel("Compute [GFLOPS] / Rate [Hz]\n(Logarithmic Scale)", fontsize=11, fontweight="bold", color="#1E293B")
ax2.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

# Formatting ticks
ax2.set_xticks(range(1975, 2026, 5))
ax2.set_xticklabels([str(y) for y in range(1975, 2026, 5)], fontsize=10)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch13_50yr_microprocessor_robotics_trend.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Plot saved successfully to {output_path}")
