"""
Generate Chapter 4 Plot: The Nervous System
Fieldbus synchronization: Bandwidth vs worst-case clock synchronization jitter.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Add current dir to path to import plot_style
sys.path.append(os.path.dirname(__file__))
from plot_style import PALETTE, ERA_COLORS, setup_canvas, add_badge

def generate_plot():
    fig, ax = setup_canvas(figsize=(14, 8.5), dpi=300)

    # ---------------------------------------------------------
    # 1. Background Shaded Regimes
    # ---------------------------------------------------------
    # Regime I: Non-Deterministic / Soft Real-Time (Jitter > 1000 µs = 1 ms)
    ax.axhspan(1000.0, 60000.0, color="#FEF2F2", alpha=0.55, zorder=0)
    # Regime II: Hard Real-Time Joint Servo Motion (1 µs to 1000 µs)
    ax.axhspan(1.0, 1000.0, color="#FFFBEB", alpha=0.45, zorder=0)
    # Regime III: Sub-Microsecond Inverter / Distributed Clocks (< 1 µs)
    ax.axhspan(0.008, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)

    # Regime Explanatory Labels - placed in clear open zones
    # Regime I label placed in open mid-upper zone
    ax.text(2.5, 25000.0, "NON-DETERMINISTIC / SOFT REAL-TIME REGIME\nCSMA/CR arbitration, unmanaged switches, non-preemptive packet delays\nSuitable for supervisory diagnostics, BMS telemetry, and human interfaces",
            fontsize=8.2, fontweight="bold", color="#B91C1C", alpha=0.9, va="center")

    # Regime II label placed in open lower-left of the yellow band
    ax.text(0.07, 35.0, "COORDINATED MULTI-AXIS JOINT SERVO REGIME (1 kHz Control Loops)\nHardware-scheduled TDMA or priority-bypassed Layer 2 industrial Ethernet\nRequired for smooth multi-axis trajectory interpolation, impedance control, and haptics",
            fontsize=8.2, fontweight="bold", color="#B45309", alpha=0.9, va="center")

    # Regime III label placed in open lower-left of the green band
    ax.text(0.07, 0.022, "SUB-MICROSECOND INVERTER & PHYSICAL AI SYNCHRONIZATION FRONTIER\nHardware Distributed Clocks (EtherCAT DC) & IEEE 802.1AS gPTP + Time-Aware Shaper (802.1Qbv)\nEnables 20–40 kHz PWM Field-Oriented Control (FOC) across 50+ actuators without torque ripple",
            fontsize=8.2, fontweight="bold", color="#047857", alpha=0.95, va="center")

    # ---------------------------------------------------------
    # 2. Key Physical Frequency & Actuation Deadlines
    # ---------------------------------------------------------
    deadlines = [
        (1000.0, "1,000 µs (1 ms): Soft Real-Time Supervisory & Telemetry Limit", PALETTE["slate"], "-."),
        (100.0,  "100 µs: 1 kHz Coordinated Multi-Axis Joint Servo Jitter Boundary", PALETTE["amber"], ":"),
        (1.0,    "1.0 µs: Motor Inverter PWM Commutation Deadline (FOC 20–40 kHz)", PALETTE["crimson"], "--"),
        (0.1,    "100 ns: Ultra-Precision Distributed Clocks Boundary (Beckhoff DC / IEEE 1588)", PALETTE["forest_green"], ":"),
    ]
    for jitter, lbl, col, ls in deadlines:
        ax.axhline(jitter, color=col, linestyle=ls, linewidth=1.2, alpha=0.75, zorder=1)
        ax.text(22000.0, jitter * 1.15, lbl, fontsize=7.5, fontweight="bold", color=col, ha="right", va="bottom")

    # ---------------------------------------------------------
    # 3. Physical AI Actuation Frontier Line & Shaded Area
    # ---------------------------------------------------------
    fx = [100.0, 1000.0, 10000.0]
    fy = [0.05, 0.035, 0.02]
    ax.plot(fx, fy, color=PALETTE["forest_green"], lw=2.2, ls="-", zorder=3, alpha=0.85)
    
    # Fill under frontier to emphasize high-performance envelope
    ax.fill_between([80.0, 100.0, 1000.0, 10000.0, 23000.0],
                    [0.05, 0.05, 0.035, 0.02, 0.02],
                    0.008, color="#A7F3D0", alpha=0.25, zorder=1)

    # Clean multi-line annotation placed at x=220, completely clear of PROFINET IRT and margins
    ax.text(220.0, 0.28, "Physical AI Actuation Frontier\nEtherCAT · EtherCAT G · IEEE TSN (802.1AS/Qbv)\nSub-100 ns sync for 50+ coordinated actuators",
            fontsize=8.0, fontweight="bold", color=PALETTE["forest_green"], va="center")

    # ---------------------------------------------------------
    # 4. Protocol Data Points & Badges
    # ---------------------------------------------------------
    protocols = [
        # Asynchronous / CSMA-CD Serial Buses
        {
            "name": "Modbus RTU",
            "bw": 0.115, "jitter": 20000.0, "year": 1979,
            "color": PALETTE["slate"], "marker": "s",
            "badge_xytext": (0.07, 42000.0),
            "text": "1979 · Modbus RTU (RS-485)\n115.2 kbps · ~20 ms jitter\nSerial polling master-slave",
        },
        {
            "name": "CAN 2.0B",
            "bw": 1.0, "jitter": 2500.0, "year": 1986,
            "color": PALETTE["crimson"], "marker": "o",
            "badge_xytext": (0.20, 2500.0),
            "text": "1986 · Classic CAN 2.0B\n1 Mbps · 1,000–5,000 µs jitter\nCSMA/CR priority arbitration",
        },
        {
            "name": "CAN-FD",
            "bw": 5.0, "jitter": 350.0, "year": 2012,
            "color": PALETTE["coral"], "marker": "o",
            "badge_xytext": (8.5, 350.0),
            "text": "2012 · CAN-FD\n5 Mbps · 200–500 µs jitter\n64-byte payload, dual bitrate",
        },

        # Time-Triggered TDMA Bus
        {
            "name": "FlexRay",
            "bw": 10.0, "jitter": 1.5, "year": 2005,
            "color": PALETTE["teal"], "marker": "^",
            "badge_xytext": (3.5, 4.5),
            "text": "2005 · FlexRay\n10 Mbps · 1–2 µs jitter\nDeterministic static TDMA cycle",
        },

        # Switched / Isochronous Industrial Ethernet
        {
            "name": "PROFINET RT",
            "bw": 100.0, "jitter": 1000.0, "year": 2003,
            "color": PALETTE["amber"], "marker": "D",
            "badge_xytext": (32.0, 2200.0),
            "text": "2003 · PROFINET RT\n100 Mbps · ~1 ms jitter\nSoftware Layer 2 bypass",
        },
        {
            "name": "PROFINET IRT",
            "bw": 100.0, "jitter": 0.5, "year": 2006,
            "color": PALETTE["gold"], "marker": "D",
            "badge_xytext": (25.0, 0.45),
            "text": "2006 · PROFINET IRT\n100 Mbps · 0.2–0.5 µs jitter\nHardware TDMA ASIC + PTCP sync",
        },
        {
            "name": "EtherCAT (100BASE-TX)",
            "bw": 100.0, "jitter": 0.05, "year": 2003,
            "color": PALETTE["forest_green"], "marker": "*",
            "badge_xytext": (20.0, 0.045),
            "text": "2003 · EtherCAT (Beckhoff)\n100 Mbps · 20–80 ns jitter\nProcessing-on-the-fly, Distributed Clocks",
        },

        # Gigabit Real-Time & TSN Frontiers
        {
            "name": "EtherCAT G",
            "bw": 1000.0, "jitter": 0.035, "year": 2018,
            "color": PALETTE["dark_blue"], "marker": "*",
            "badge_xytext": (380.0, 0.016),
            "text": "2018 · EtherCAT G\n1 Gbps · 30–50 ns jitter\nBranch controller gigabit fabric",
        },
        {
            "name": "IEEE TSN (802.1AS/Qbv)",
            "bw": 10000.0, "jitter": 0.02, "year": 2024,
            "color": PALETTE["purple"], "marker": "*",
            "badge_xytext": (3200.0, 0.035),
            "text": "2024 · IEEE TSN (802.1AS + 802.1Qbv)\n1–10 Gbps · 10–30 ns jitter\nConverged IT/OT Time-Aware Shaper",
        },
    ]

    # Plot points
    for p in protocols:
        s_size = 220 if p["marker"] == "*" else (110 if p["marker"] == "D" else 100)
        ax.scatter(p["bw"], p["jitter"], color=p["color"], s=s_size,
                   marker=p["marker"], edgecolor="#FFFFFF", linewidth=1.5, zorder=10)
        add_badge(ax, p["text"], xy=(p["bw"], p["jitter"]), xytext=p["badge_xytext"],
                  color=PALETTE["charcoal"], fontsize=7.8, arrow=True, arrow_color=p["color"])

    # ---------------------------------------------------------
    # 5. Axes, Scales, Titles, and Legends
    # ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(0.05, 25000.0)
    ax.set_ylim(0.008, 70000.0)

    # Custom Ticks
    ax.set_xticks([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f"{int(x*1000)} kbps" if x < 1.0 else (f"{int(x)} Mbps" if x < 1000.0 else f"{int(x/1000)} Gbps")
    ))

    ax.set_yticks([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(
        lambda y, p: f"{int(y*1000)} ns" if y < 1.0 else (f"{int(y)} µs" if y < 1000.0 else f"{int(y/1000)} ms")
    ))

    ax.set_xlabel("Fieldbus Raw Bandwidth / Bitrate (log scale)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Worst-Case Clock Synchronization Jitter (log scale)  [ $\\downarrow$ Lower Jitter = Higher Determinism ]",
                  fontsize=10.5, fontweight="bold", labelpad=8)

    # Title block
    fig.suptitle("The Nervous System Frontier: Fieldbus Bandwidth vs Synchronization Jitter",
                 fontsize=14, fontweight="bold", x=0.08, y=0.98, ha="left", color=PALETTE["charcoal"])
    ax.set_title("From asynchronous CSMA arbitration to gigabit sub-microsecond determinism: enabling 20–40 kHz distributed motor control",
                 fontsize=9.5, color="#475569", loc="left", pad=14)

    # Legend - placed at top right
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=PALETTE['slate'], markersize=8, label='Serial Master-Slave (Modbus RS-485)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['crimson'], markersize=8, label='Asynchronous CSMA/CR (CAN, CAN-FD)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=PALETTE['teal'], markersize=8, label='Time-Triggered Static TDMA (FlexRay)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=PALETTE['gold'], markersize=8, label='Switched / Isochronous Industrial Ethernet (PROFINET RT/IRT)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=PALETTE['forest_green'], markersize=12, label='On-the-Fly & Converged TSN (EtherCAT, TSN 802.1AS/Qbv)'),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.96,
                       edgecolor="#CBD5E1", fontsize=7.8, title="Industrial Fieldbus Architectures",
                       title_fontsize=8.2)
    legend.get_title().set_fontweight("bold")
    legend.set_zorder(15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.94)

    out_path = os.path.join(os.path.dirname(__file__), "ch04_fieldbus_jitter.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_plot()
