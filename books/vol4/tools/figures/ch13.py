"""
book/tools/figures/ch13.py
Figures for Chapter 13: Placement (Heterogeneous SoC Bus Contention, PDN Droop, Lock-Free Boundary Contract).
Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import os
import subprocess
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch13_soc_contention_droop():
    W = 960
    H = 580
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">HETEROGENEOUS SOC RESOURCE CONTENTION &amp; POWER-RAIL INTERFERENCE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Shared Memory Crossbar Burst Starvation and Transient PDN Voltage Droop on Shared Silicon</text>')

    # -------------------------------------------------------------
    # PANEL A: UNIFIED MEMORY CROSSBAR & DRAM ARBITRATION CONTENTION
    # -------------------------------------------------------------
    p1_x = 24
    p1_y = 62
    p1_w = 445
    p1_h = 500
    svg.append(f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="28" rx="8" fill="{NAVY}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p1_x+14}" y="{p1_y+19}" font-size="11" font-weight="700" fill="{NAVY}">(a) Memory Bus &amp; Cache Contention</text>')
    svg.append(f'<text x="{p1_x+p1_w-14}" y="{p1_y+19}" font-size="9" font-weight="600" fill="{MUTED}" text-anchor="end">64-bit LPDDR4 · 12.8 GB/s</text>')

    # Compute Masters
    # 1. Neural Processing Unit (Host / NPU)
    npu_x = p1_x + 14
    npu_y = p1_y + 38
    npu_w = 200
    npu_h = 76
    svg.append(f'<rect x="{npu_x}" y="{npu_y}" width="{npu_w}" height="{npu_h}" rx="6" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{npu_x}" y="{npu_y}" width="{npu_w}" height="20" rx="6" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{npu_x+npu_w/2}" y="{npu_y+14}" font-size="9" font-weight="700" fill="{BLUE}" text-anchor="middle">PROPOSAL ENGINE (NPU / MPU)</text>')
    svg.append(f'<text x="{npu_x+8}" y="{npu_y+34}" font-size="8.5" font-weight="600" fill="{INK}">50 Hz Policy Inference</text>')
    svg.append(f'<text x="{npu_x+8}" y="{npu_y+47}" font-size="8" fill="{SLATE}">• 180 MB weights &amp; activations</text>')
    svg.append(f'<text x="{npu_x+8}" y="{npu_y+60}" font-size="8" font-weight="700" fill="{CORAL}">• 256 kB burst @ 12.0 GB/s (94% bus)</text>')

    # 2. Real-Time MCU (Safety Enforcer)
    mcu_x = p1_x + p1_w - 214
    mcu_y = p1_y + 38
    mcu_w = 200
    mcu_h = 76
    svg.append(f'<rect x="{mcu_x}" y="{mcu_y}" width="{mcu_w}" height="{mcu_h}" rx="6" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{mcu_x}" y="{mcu_y}" width="{mcu_w}" height="20" rx="6" fill="{PETROL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{mcu_x+mcu_w/2}" y="{mcu_y+14}" font-size="9" font-weight="700" fill="{PETROL}" text-anchor="middle">SAFETY ENFORCER (RT Core)</text>')
    svg.append(f'<text x="{mcu_x+8}" y="{npu_y+34}" font-size="8.5" font-weight="600" fill="{INK}">1000 Hz Hard Real-Time Loop</text>')
    svg.append(f'<text x="{mcu_x+8}" y="{npu_y+47}" font-size="8" fill="{SLATE}">• 64 kB sensor history &amp; invariants</text>')
    svg.append(f'<text x="{mcu_x+8}" y="{npu_y+60}" font-size="8" font-weight="700" fill="{PETROL}">• Uncontended read budget: 5.0 µs</text>')

    # Shared Interconnect / L3 Cache
    ic_x = p1_x + 14
    ic_y = p1_y + 124
    ic_w = p1_w - 28
    ic_h = 60
    svg.append(f'<rect x="{ic_x}" y="{ic_y}" width="{ic_w}" height="{ic_h}" rx="6" fill="{BG_WHITE}" stroke="{PURPLE}" stroke-width="1.3" stroke-dasharray="4,2"/>')
    svg.append(f'<text x="{ic_x+ic_w/2}" y="{ic_y+16}" font-size="9.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">SHARED SYSTEM CROSSBAR (AXI4/5 QoS) &amp; 8MB LAST-LEVEL CACHE</text>')
    svg.append(f'<text x="{ic_x+10}" y="{ic_y+34}" font-size="8" fill="{CORAL}">⚡ Matrix tile burst evicts enforcer sensor tables from L3 (t_refill = 20ns → 350ns)</text>')
    svg.append(f'<text x="{ic_x+10}" y="{ic_y+48}" font-size="8" fill="{SLATE}">• Non-preemptive packet-atomic bursts (ARLEN ≤ 256 beats cannot be truncated)</text>')

    # Arrows to Interconnect
    svg.append(f'<line x1="{npu_x+npu_w/2}" y1="{npu_y+npu_h}" x2="{npu_x+npu_w/2}" y2="{ic_y}" stroke="{BLUE}" stroke-width="1.8" marker-end="url(#arr-blue)"/>')
    svg.append(f'<line x1="{mcu_x+mcu_w/2}" y1="{mcu_y+mcu_h}" x2="{mcu_x+mcu_w/2}" y2="{ic_y}" stroke="{PETROL}" stroke-width="1.8" marker-end="url(#arr-petrol)"/>')

    # DRAM Controller & Arbiter Queue
    dq_x = p1_x + 14
    dq_y = p1_y + 194
    dq_w = p1_w - 28
    dq_h = 110
    svg.append(f'<rect x="{dq_x}" y="{dq_y}" width="{dq_w}" height="{dq_h}" rx="6" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<text x="{dq_x+10}" y="{dq_y+16}" font-size="9.5" font-weight="700" fill="{NAVY}">DRAM CONTROLLER QUEUE &amp; ARBITER (Head-of-Line Blocking)</text>')

    # Queue Slots
    slot_y = dq_y + 26
    # Slot 1: In-flight NPU burst
    svg.append(f'<rect x="{dq_x+10}" y="{slot_y}" width="125" height="34" rx="4" fill="{CORAL}" fill-opacity="0.15" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{dq_x+72}" y="{slot_y+14}" font-size="8" font-weight="700" fill="{CORAL}" text-anchor="middle">In-Flight NPU Burst</text>')
    svg.append(f'<text x="{dq_x+72}" y="{slot_y+26}" font-size="7.5" fill="{INK}" text-anchor="middle">256 kB Tile (20.0 µs)</text>')

    # Slot 2: Sensor DMA Burst
    svg.append(f'<rect x="{dq_x+140}" y="{slot_y}" width="100" height="34" rx="4" fill="{AMBER}" fill-opacity="0.15" stroke="{AMBER}" stroke-width="1.1"/>')
    svg.append(f'<text x="{dq_x+190}" y="{slot_y+14}" font-size="8" font-weight="700" fill="{AMBER}" text-anchor="middle">Sensor DMA Ingest</text>')
    svg.append(f'<text x="{dq_x+190}" y="{slot_y+26}" font-size="7.5" fill="{INK}" text-anchor="middle">64 kB (5.0 µs)</text>')

    # Slot 3: Logging DMA Burst
    svg.append(f'<rect x="{dq_x+245}" y="{slot_y}" width="80" height="34" rx="4" fill="{SLATE}" fill-opacity="0.15" stroke="{SLATE}" stroke-width="1.1"/>')
    svg.append(f'<text x="{dq_x+285}" y="{slot_y+14}" font-size="8" font-weight="700" fill="{SLATE}" text-anchor="middle">Log Dump</text>')
    svg.append(f'<text x="{dq_x+285}" y="{slot_y+26}" font-size="7.5" fill="{INK}" text-anchor="middle">32 kB (2.5 µs)</text>')

    # Slot 4: Enforcer Read (BLOCKED)
    svg.append(f'<rect x="{dq_x+330}" y="{slot_y}" width="78" height="34" rx="4" fill="{PETROL}" fill-opacity="0.2" stroke="{PETROL}" stroke-width="1.4" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{dq_x+369}" y="{slot_y+14}" font-size="8" font-weight="700" fill="{PETROL}" text-anchor="middle">RT Enforcer</text>')
    svg.append(f'<text x="{dq_x+369}" y="{slot_y+26}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">WAITS in Q</text>')

    svg.append(f'<line x1="{dq_x+ic_w/2}" y1="{ic_y+ic_h}" x2="{dq_x+ic_w/2}" y2="{dq_y}" stroke="{PURPLE}" stroke-width="1.5" marker-end="url(#arr-purple)"/>')

    # Queue Annotation
    svg.append(f'<text x="{dq_x+10}" y="{dq_y+74}" font-size="8" fill="{SLATE}">Total Queuing Delay: t_wait = 20.0 + 5.0 + 2.5 + 0.1 (bank conflict) = 27.6 µs</text>')
    svg.append(f'<text x="{dq_x+10}" y="{dq_y+88}" font-size="8" font-weight="600" fill="{CORAL}">DRAM Page Reordering: Arbiter favors open row locality, starving real-time closed-bank reads.</text>')
    svg.append(f'<text x="{dq_x+10}" y="{dq_y+101}" font-size="7.5" fill="{MUTED}">Result: Enforcer read latency explodes from nominal 5.0 µs to 27.6–80.0 µs under load.</text>')

    # Latency Waterfall Comparison
    wf_y = p1_y + 314
    wf_w = p1_w - 28
    wf_h = 176
    svg.append(f'<rect x="{p1_x+14}" y="{wf_y}" width="{wf_w}" height="{wf_h}" rx="6" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<text x="{p1_x+24}" y="{wf_y+16}" font-size="9" font-weight="700" fill="{INK}">ENFORCEMENT TIMING SLACK COLLAPSE (1.0 ms Control Cycle)</text>')

    # Timeline 1: Uncontended Baseline
    svg.append(f'<text x="{p1_x+24}" y="{wf_y+36}" font-size="8" font-weight="700" fill="{PETROL}">1. Uncontended (Cold / Idle): Total = 100.0 µs (Slack = +20.0 µs ✓)</text>')
    t1_y = wf_y + 44
    svg.append(f'<rect x="{p1_x+24}" y="{t1_y}" width="20" height="18" rx="2" fill="{PETROL}" fill-opacity="0.8"/>')
    svg.append(f'<text x="{p1_x+34}" y="{t1_y+12}" font-size="7" fill="#FFF" text-anchor="middle">5µs</text>')
    svg.append(f'<rect x="{p1_x+46}" y="{t1_y}" width="160" height="18" rx="2" fill="{BLUE}" fill-opacity="0.8"/>')
    svg.append(f'<text x="{p1_x+126}" y="{t1_y+12}" font-size="7" fill="#FFF" text-anchor="middle">Invariant Evaluation: 80 µs</text>')
    svg.append(f'<rect x="{p1_x+208}" y="{t1_y}" width="30" height="18" rx="2" fill="{NAVY}" fill-opacity="0.8"/>')
    svg.append(f'<text x="{p1_x+223}" y="{t1_y+12}" font-size="7" fill="#FFF" text-anchor="middle">15µs</text>')
    svg.append(f'<rect x="{p1_x+240}" y="{t1_y}" width="40" height="18" rx="2" fill="{TEAL}" fill-opacity="0.2" stroke="{TEAL}" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{p1_x+260}" y="{t1_y+12}" font-size="7" font-weight="700" fill="{TEAL}" text-anchor="middle">+20µs</text>')

    # Budget Line
    svg.append(f'<line x1="{p1_x+280}" y1="{wf_y+28}" x2="{p1_x+280}" y2="{wf_y+130}" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{p1_x+282}" y="{wf_y+34}" font-size="7" font-weight="700" fill="{CORAL}">Max Compute Budget (120 µs)</text>')

    # Timeline 2: Contended Under NPU Burst
    svg.append(f'<text x="{p1_x+24}" y="{wf_y+78}" font-size="8" font-weight="700" fill="{CORAL}">2. Contended Under NPU Burst: Total = 127.6 µs (Slack = -7.6 µs ✕)</text>')
    t2_y = wf_y + 86
    # Stall segment
    svg.append(f'<rect x="{p1_x+24}" y="{t2_y}" width="55" height="18" rx="2" fill="{CORAL}" fill-opacity="0.7"/>')
    svg.append(f'<text x="{p1_x+51}" y="{t2_y+12}" font-size="6.5" font-weight="700" fill="#FFF" text-anchor="middle">Stall 27.6µs</text>')
    # Read segment
    svg.append(f'<rect x="{p1_x+81}" y="{t2_y}" width="20" height="18" rx="2" fill="{PETROL}" fill-opacity="0.8"/>')
    svg.append(f'<text x="{p1_x+91}" y="{t2_y+12}" font-size="7" fill="#FFF" text-anchor="middle">5µs</text>')
    # Eval segment
    svg.append(f'<rect x="{p1_x+103}" y="{t2_y}" width="160" height="18" rx="2" fill="{BLUE}" fill-opacity="0.8"/>')
    svg.append(f'<text x="{p1_x+183}" y="{t2_y+12}" font-size="7" fill="#FFF" text-anchor="middle">Invariant Evaluation: 80 µs</text>')
    # Write segment
    svg.append(f'<rect x="{p1_x+265}" y="{t2_y}" width="30" height="18" rx="2" fill="{NAVY}" fill-opacity="0.8"/>')
    svg.append(f'<text x="{p1_x+280}" y="{t2_y+12}" font-size="7" fill="#FFF" text-anchor="middle">15µs</text>')

    # Overrun Callout
    svg.append(f'<rect x="{p1_x+24}" y="{wf_y+114}" width="{wf_w-20}" height="48" rx="4" fill="{CORAL}" fill-opacity="0.08" stroke="{CORAL}" stroke-width="1"/>')
    svg.append(f'<text x="{p1_x+32}" y="{wf_y+128}" font-size="8" font-weight="700" fill="{CORAL}">DEADLINE BREACH: 7.6 µs Overrun into Valve Transit Margin</text>')
    svg.append(f'<text x="{p1_x+32}" y="{wf_y+142}" font-size="7.5" fill="{SLATE}">Although average channel utilization is only 25%, instantaneous burst collisions</text>')
    svg.append(f'<text x="{p1_x+32}" y="{wf_y+154}" font-size="7.5" fill="{SLATE}">exhaust timing margins and delay high-pressure shutoff command.</text>')

    # -------------------------------------------------------------
    # PANEL B: POWER DISTRIBUTION NETWORK (PDN) DROOP & VOLTAGE COLLAPSE
    # -------------------------------------------------------------
    p2_x = 485
    p2_y = 62
    p2_w = 450
    p2_h = 500
    svg.append(f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="28" rx="8" fill="{CRIMSON}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p2_x+14}" y="{p2_y+19}" font-size="11" font-weight="700" fill="{CRIMSON}">(b) PDN Transient Voltage Droop &amp; Clock Jitter</text>')
    svg.append(f'<text x="{p2_x+p2_w-14}" y="{p2_y+19}" font-size="9" font-weight="600" fill="{MUTED}" text-anchor="end">ΔV = L(di/dt) + I·R</text>')

    # PDN Schematic Circuit Model
    sch_x = p2_x + 14
    sch_y = p2_y + 38
    sch_w = p2_w - 28
    sch_h = 100
    svg.append(f'<rect x="{sch_x}" y="{sch_y}" width="{sch_w}" height="{sch_h}" rx="6" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{sch_x+10}" y="{sch_y+15}" font-size="9" font-weight="700" fill="{INK}">POWER DISTRIBUTION NETWORK (PDN) EQUIVALENT CIRCUIT</text>')

    # VRM block
    svg.append(f'<rect x="{sch_x+10}" y="{sch_y+26}" width="70" height="42" rx="4" fill="{NAVY}" fill-opacity="0.1" stroke="{NAVY}" stroke-width="1.2"/>')
    svg.append(f'<text x="{sch_x+45}" y="{sch_y+42}" font-size="8" font-weight="700" fill="{NAVY}" text-anchor="middle">Off-Chip VRM</text>')
    svg.append(f'<text x="{sch_x+45}" y="{sch_y+54}" font-size="7" fill="{MUTED}" text-anchor="middle">0.85V (τ=2µs)</text>')

    # Inductance & Resistance parasitics
    svg.append(f'<line x1="{sch_x+80}" y1="{sch_y+47}" x2="{sch_x+115}" y2="{sch_y+47}" stroke="{INK}" stroke-width="1.5"/>')
    # Resistor R_pkg
    svg.append(f'<rect x="{sch_x+115}" y="{sch_y+38}" width="40" height="18" rx="2" fill="{BG_WHITE}" stroke="{INK}" stroke-width="1.2"/>')
    svg.append(f'<text x="{sch_x+135}" y="{sch_y+50}" font-size="7.5" font-weight="700" fill="{INK}" text-anchor="middle">R=8mΩ</text>')
    svg.append(f'<line x1="{sch_x+155}" y1="{sch_y+47}" x2="{sch_x+180}" y2="{sch_y+47}" stroke="{INK}" stroke-width="1.5"/>')
    # Inductor L_pkg
    svg.append(f'<rect x="{sch_x+180}" y="{sch_y+38}" width="42" height="18" rx="2" fill="{BG_WHITE}" stroke="{INK}" stroke-width="1.2"/>')
    svg.append(f'<text x="{sch_x+201}" y="{sch_y+50}" font-size="7.5" font-weight="700" fill="{INK}" text-anchor="middle">L=30pH</text>')
    svg.append(f'<line x1="{sch_x+222}" y1="{sch_y+47}" x2="{sch_x+255}" y2="{sch_y+47}" stroke="{INK}" stroke-width="1.5"/>')

    # Shared Internal Rail
    svg.append(f'<line x1="{sch_x+255}" y1="{sch_y+26}" x2="{sch_x+255}" y2="{sch_y+84}" stroke="{CORAL}" stroke-width="2.5"/>')
    svg.append(f'<text x="{sch_x+260}" y="{sch_y+34}" font-size="8" font-weight="700" fill="{CORAL}">V_core Rail</text>')

    # Decoupling Cap (DTC)
    svg.append(f'<line x1="{sch_x+255}" y1="{sch_y+58}" x2="{sch_x+285}" y2="{sch_y+58}" stroke="{INK}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{sch_x+285}" y="{sch_y+50}" width="36" height="16" rx="2" fill="{TEAL}" fill-opacity="0.15" stroke="{TEAL}" stroke-width="1"/>')
    svg.append(f'<text x="{sch_x+303}" y="{sch_y+61}" font-size="7" font-weight="600" fill="{TEAL}" text-anchor="middle">C_die</text>')

    # Load 1: NPU Step Current
    svg.append(f'<line x1="{sch_x+255}" y1="{sch_y+38}" x2="{sch_x+345}" y2="{sch_y+38}" stroke="{BLUE}" stroke-width="1.5" marker-end="url(#arr-blue)"/>')
    svg.append(f'<rect x="{sch_x+345}" y="{sch_y+26}" width="70" height="24" rx="3" fill="{BLUE}" fill-opacity="0.12" stroke="{BLUE}" stroke-width="1"/>')
    svg.append(f'<text x="{sch_x+380}" y="{sch_y+38}" font-size="7.5" font-weight="700" fill="{BLUE}" text-anchor="middle">NPU Load</text>')
    svg.append(f'<text x="{sch_x+380}" y="{sch_y+47}" font-size="6.5" fill="{CORAL}" text-anchor="middle">ΔI=6A (3A/ns)</text>')

    # Load 2: RT MCU Victim
    svg.append(f'<line x1="{sch_x+255}" y1="{sch_y+74}" x2="{sch_x+345}" y2="{sch_y+74}" stroke="{PETROL}" stroke-width="1.5" marker-end="url(#arr-petrol)"/>')
    svg.append(f'<rect x="{sch_x+345}" y="{sch_y+64}" width="70" height="24" rx="3" fill="{PETROL}" fill-opacity="0.12" stroke="{PETROL}" stroke-width="1"/>')
    svg.append(f'<text x="{sch_x+380}" y="{sch_y+76}" font-size="7.5" font-weight="700" fill="{PETROL}" text-anchor="middle">RT Core Victim</text>')
    svg.append(f'<text x="{sch_x+380}" y="{sch_y+85}" font-size="6.5" fill="{CORAL}" text-anchor="middle">Suffers Droop</text>')

    svg.append(f'<text x="{sch_x+10}" y="{sch_y+94}" font-size="7.5" fill="{SLATE}">Droop calculation: ΔV = L(di/dt) + I·R = 30pH·3A/ns + 6A·8mΩ = 90mV + 48mV = 138mV</text>')

    # Oscilloscope Waveform Display
    osc_x = p2_x + 14
    osc_y = p2_y + 148
    osc_w = p2_w - 28
    osc_h = 342
    svg.append(f'<rect x="{osc_x}" y="{osc_y}" width="{osc_w}" height="{osc_h}" rx="6" fill="#0F172A" stroke="{BORDER_DARK}" stroke-width="1.4"/>')

    # Grid lines
    for gx in range(osc_x+30, osc_x+osc_w, 48):
        svg.append(f'<line x1="{gx}" y1="{osc_y+10}" x2="{gx}" y2="{osc_y+osc_h-15}" stroke="#1E293B" stroke-width="1"/>')
    for gy in range(osc_y+25, osc_y+osc_h-15, 32):
        svg.append(f'<line x1="{osc_x+10}" y1="{gy}" x2="{osc_x+osc_w-10}" y2="{gy}" stroke="#1E293B" stroke-width="1"/>')

    # Event line: NPU activation
    t_burst = osc_x + 90
    svg.append(f'<line x1="{t_burst}" y1="{osc_y+10}" x2="{t_burst}" y2="{osc_y+osc_h-20}" stroke="{AMBER}" stroke-width="1.3" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{t_burst}" y="{osc_y+20}" font-size="7.5" font-weight="700" fill="{AMBER}" text-anchor="middle">NPU Step (t₀)</text>')

    # Waveform 1: I_NPU(t)
    w1_y = osc_y + 55
    svg.append(f'<text x="{osc_x+12}" y="{w1_y-14}" font-size="8.5" font-weight="700" fill="{BLUE}">CH1: NPU Current Draw I_NPU(t)</text>')
    svg.append(f'<text x="{osc_x+osc_w-12}" y="{w1_y-14}" font-size="7.5" fill="#94A3B8" text-anchor="end">ΔI = +6.0 A in 2.0 ns</text>')
    svg.append(f'<line x1="{osc_x+15}" y1="{w1_y}" x2="{t_burst}" y2="{w1_y}" stroke="{BLUE}" stroke-width="2"/>')
    svg.append(f'<line x1="{t_burst}" y1="{w1_y}" x2="{t_burst+6}" y2="{w1_y-24}" stroke="{BLUE}" stroke-width="2"/>')
    svg.append(f'<line x1="{t_burst+6}" y1="{w1_y-24}" x2="{osc_x+osc_w-15}" y2="{w1_y-24}" stroke="{BLUE}" stroke-width="2"/>')

    # Waveform 2: V_core(t) Droop
    w2_y = osc_y + 130
    svg.append(f'<text x="{osc_x+12}" y="{w2_y-22}" font-size="8.5" font-weight="700" fill="{CORAL}">CH2: Supply Voltage V_core(t) on Shared Rail</text>')
    svg.append(f'<text x="{osc_x+osc_w-12}" y="{w2_y-22}" font-size="7.5" fill="{CORAL}" text-anchor="end">16.2% Collapse (0.712 V &lt; V_min)</text>')

    # Nominal level line (0.85V)
    svg.append(f'<line x1="{osc_x+15}" y1="{w2_y-12}" x2="{osc_x+osc_w-15}" y2="{w2_y-12}" stroke="#64748B" stroke-width="1" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{osc_x+16}" y="{w2_y-14}" font-size="6.5" fill="#94A3B8">Nominal 0.85V</text>')

    # V_min limit line (0.78V)
    svg.append(f'<line x1="{osc_x+15}" y1="{w2_y+8}" x2="{osc_x+osc_w-15}" y2="{w2_y+8}" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{osc_x+16}" y="{w2_y+6}" font-size="6.5" font-weight="700" fill="{CORAL}">V_min = 0.78V (Timing Fault / Brownout Limit)</text>')

    # Droop curve
    droop_path = f"M {osc_x+15} {w2_y-12} L {t_burst} {w2_y-12} C {t_burst+4} {w2_y+26}, {t_burst+12} {w2_y+28}, {t_burst+35} {w2_y+16} C {t_burst+80} {w2_y+8}, {t_burst+140} {w2_y-2}, {osc_x+osc_w-15} {w2_y-4}"
    svg.append(f'<path d="{droop_path}" fill="none" stroke="{CORAL}" stroke-width="2.2"/>')

    # Droop annotation arrow
    svg.append(f'<line x1="{t_burst+14}" y1="{w2_y-12}" x2="{t_burst+14}" y2="{w2_y+26}" stroke="{AMBER}" stroke-width="1.2"/>')
    svg.append(f'<text x="{t_burst+20}" y="{w2_y+22}" font-size="7" font-weight="700" fill="{AMBER}">ΔV = -138 mV</text>')

    # Waveform 3: RT Core Clock & Logic Delay
    w3_y = osc_y + 220
    svg.append(f'<text x="{osc_x+12}" y="{w3_y-16}" font-size="8.5" font-weight="700" fill="{PETROL}">CH3: RT Core Propagation Delay &amp; Clock Jitter</text>')
    svg.append(f'<text x="{osc_x+osc_w-12}" y="{w3_y-16}" font-size="7.5" fill="{CORAL}" text-anchor="end">Gate delay t_prop stretches +38%</text>')

    # Clock square pulses before and during droop
    # Before droop: sharp, fast clock (1.2 GHz)
    cx_pos = osc_x + 15
    while cx_pos < t_burst:
        svg.append(f'<line x1="{cx_pos}" y1="{w3_y}" x2="{cx_pos+6}" y2="{w3_y}" stroke="{PETROL}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx_pos+6}" y1="{w3_y}" x2="{cx_pos+6}" y2="{w3_y-16}" stroke="{PETROL}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx_pos+6}" y1="{w3_y-16}" x2="{cx_pos+12}" y2="{w3_y-16}" stroke="{PETROL}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx_pos+12}" y1="{w3_y-16}" x2="{cx_pos+12}" y2="{w3_y}" stroke="{PETROL}" stroke-width="1.5"/>')
        cx_pos += 12

    # During droop: stretched clock / jitter / throttled to 600 MHz
    while cx_pos < osc_x + osc_w - 20:
        svg.append(f'<line x1="{cx_pos}" y1="{w3_y}" x2="{cx_pos+12}" y2="{w3_y}" stroke="{CORAL}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx_pos+12}" y1="{w3_y}" x2="{cx_pos+12}" y2="{w3_y-16}" stroke="{CORAL}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx_pos+12}" y1="{w3_y-16}" x2="{cx_pos+24}" y2="{w3_y-16}" stroke="{CORAL}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx_pos+24}" y1="{w3_y-16}" x2="{cx_pos+24}" y2="{w3_y}" stroke="{CORAL}" stroke-width="1.5"/>')
        cx_pos += 24

    # Explanatory bottom summary box
    svg.append(f'<rect x="{osc_x+10}" y="{osc_y+osc_h-86}" width="{osc_w-20}" height="72" rx="4" fill="#1E293B" stroke="{BORDER_DARK}" stroke-width="1"/>')
    svg.append(f'<text x="{osc_x+18}" y="{osc_y+osc_h-70}" font-size="8" font-weight="700" fill="{CORAL}">PHYSICAL CROSS-DOMAIN FAILURE COUPLING</text>')
    svg.append(f'<text x="{osc_x+18}" y="{osc_y+osc_h-56}" font-size="7.5" fill="#E2E8F0">• VRM loop bandwidth (&lt;1 MHz) is 3 orders of magnitude too slow for nanosecond di/dt steps.</text>')
    svg.append(f'<text x="{osc_x+18}" y="{osc_y+osc_h-44}" font-size="7.5" fill="#E2E8F0">• Logic propagation delay t_prop proportional to V_DD/(V_DD-V_th)^2 causes setup-time violations.</text>')
    svg.append(f'<text x="{osc_x+18}" y="{osc_y+osc_h-32}" font-size="7.5" fill="#E2E8F0">• DVFS emergency throttle cuts clock to 600 MHz: t_exec = 0.70ms → 1.40ms &gt; 1.0ms deadline.</text>')
    svg.append(f'<text x="{osc_x+18}" y="{osc_y+osc_h-20}" font-size="7.5" font-weight="700" fill="{AMBER}">Verdict: Software isolation is defeated by power grid impedance and thermal substrate coupling.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/13-placement/figures/fig13_soc_contention_droop.svg", "\n".join(svg))


def gen_ch13_lockfree_boundary_contract():
    W = 960
    H = 530
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">LOCK-FREE SEQLOCK BOUNDARY CONTRACT &amp; PREEMPTION RECOVERY</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Asynchronous Inter-Core State Exchange with Zero Wait States and Deterministic Fallback Hierarchy</text>')

    # -------------------------------------------------------------
    # PANEL A: WRITER AND READER MEMORY BARRIER PROTOCOL
    # -------------------------------------------------------------
    p1_x = 24
    p1_y = 62
    p1_w = 460
    p1_h = 450
    svg.append(f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="28" rx="8" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p1_x+14}" y="{p1_y+19}" font-size="11" font-weight="700" fill="{BLUE}">(a) Lock-Free Seqlock Publication Protocol</text>')
    svg.append(f'<text x="{p1_x+p1_w-14}" y="{p1_y+19}" font-size="9" font-weight="600" fill="{MUTED}" text-anchor="end">Atomic Barriers · Zero Mutex</text>')

    # Writer Column (Untrusted Host / NPU)
    w_x = p1_x + 14
    w_y = p1_y + 36
    w_w = 205
    w_h = 240
    svg.append(f'<rect x="{w_x}" y="{w_y}" width="{w_w}" height="{w_h}" rx="6" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{w_x}" y="{w_y}" width="{w_w}" height="22" rx="6" fill="{BLUE}" fill-opacity="0.1"/>')
    svg.append(f'<text x="{w_x+w_w/2}" y="{w_y+15}" font-size="9.5" font-weight="700" fill="{BLUE}" text-anchor="middle">UNTRUSTED WRITER (NPU/MPU)</text>')

    w_steps = [
        ("1. atomic_store_release(S, 2k+1)", "Marks sequence counter ODD", CORAL),
        ("2. Store-Release Barrier", "Flushes counter visibility", PURPLE),
        ("3. Write Payload Fields", "m_cmd, k, t_prod, Δt_valid, CRC", INK),
        ("4. Store-Release Barrier", "Ensures payload commits first", PURPLE),
        ("5. atomic_store_release(S, 2k+2)", "Marks sequence counter EVEN", TEAL)
    ]
    for idx, (st, desc, col) in enumerate(w_steps):
        sy = w_y + 30 + idx * 41
        svg.append(f'<rect x="{w_x+8}" y="{sy}" width="{w_w-16}" height="35" rx="4" fill="{col}" fill-opacity="0.08" stroke="{col}" stroke-width="0.9"/>')
        svg.append(f'<text x="{w_x+12}" y="{sy+14}" font-size="8" font-weight="700" fill="{col}">{st}</text>')
        svg.append(f'<text x="{w_x+12}" y="{sy+27}" font-size="7.5" fill="{SLATE}">{desc}</text>')

    # Reader Column (Deterministic Real-Time Enforcer)
    r_x = p1_x + p1_w - 219
    r_y = p1_y + 36
    r_w = 205
    r_h = 240
    svg.append(f'<rect x="{r_x}" y="{r_y}" width="{r_w}" height="{r_h}" rx="6" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{r_x}" y="{r_y}" width="{r_w}" height="22" rx="6" fill="{PETROL}" fill-opacity="0.1"/>')
    svg.append(f'<text x="{r_x+r_w/2}" y="{r_y+15}" font-size="9.5" font-weight="700" fill="{PETROL}" text-anchor="middle">SAFETY READER (Real-Time MCU)</text>')

    r_steps = [
        ("1. S1 = atomic_load_acquire(S)", "Sample initial sequence", PETROL),
        ("2. if (S1 & 1) reject();", "Odd → Write in progress! Zero wait", CORAL),
        ("3. Load-Acquire &amp; Copy", "Copy payload to private stack", INK),
        ("4. S2 = atomic_load_acquire(S)", "Sample second sequence", PETROL),
        ("5. if (S1 != S2) reject_torn();", "Torn write trap (zero blocking)", AMBER)
    ]
    for idx, (st, desc, col) in enumerate(r_steps):
        sy = r_y + 30 + idx * 41
        svg.append(f'<rect x="{r_x+8}" y="{sy}" width="{r_w-16}" height="35" rx="4" fill="{col}" fill-opacity="0.08" stroke="{col}" stroke-width="0.9"/>')
        svg.append(f'<text x="{r_x+12}" y="{sy+14}" font-size="8" font-weight="700" fill="{col}">{st}</text>')
        svg.append(f'<text x="{r_x+12}" y="{sy+27}" font-size="7.5" fill="{SLATE}">{desc}</text>')

    # Exchange Contract Details Box (Bottom of Panel A)
    con_y = p1_y + 284
    con_w = p1_w - 28
    con_h = 154
    svg.append(f'<rect x="{p1_x+14}" y="{con_y}" width="{con_w}" height="{con_h}" rx="6" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{p1_x+24}" y="{con_y+16}" font-size="9" font-weight="700" fill="{NAVY}">SELF-CONTAINED BOUNDARY CONTRACT PAYLOAD (256 bytes)</text>')

    fields = [
        ("dot_m_cmd", "float32", "Commanded mass flow target (physical limits [0, 45 kg/s])"),
        ("seq_id (k)", "uint64", "Monotonic proposal index (detects dropped/skipped frames)"),
        ("t_prod", "uint64", "Monotonic hardware timestamp of inference completion"),
        ("Δt_valid", "uint32", "Lease expiration duration (e.g. 45.0 ms maximum validity)"),
        ("crc32_hash", "uint32", "Payload integrity checksum (detects corrupted bitflips)")
    ]
    for idx, (fn, ft, fd) in enumerate(fields):
        fy = con_y + 28 + idx * 24
        svg.append(f'<rect x="{p1_x+24}" y="{fy}" width="80" height="20" rx="3" fill="{NAVY}" fill-opacity="0.08"/>')
        svg.append(f'<text x="{p1_x+28}" y="{fy+13}" class="code-text">{fn}</text>')
        svg.append(f'<text x="{p1_x+112}" y="{fy+13}" font-size="7.5" font-weight="700" fill="{PURPLE}">[{ft}]</text>')
        svg.append(f'<text x="{p1_x+165}" y="{fy+13}" font-size="7.5" fill="{SLATE}">{fd}</text>')

    # -------------------------------------------------------------
    # PANEL B: PREEMPTION FAULT MATRIX & DETERMINISTIC FALLBACKS
    # -------------------------------------------------------------
    p2_x = 495
    p2_y = 62
    p2_w = 440
    p2_h = 450
    svg.append(f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="28" rx="8" fill="{PURPLE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p2_x+14}" y="{p2_y+19}" font-size="11" font-weight="700" fill="{PURPLE}">(b) Preemption Points &amp; Deterministic Fallbacks</text>')
    svg.append(f'<text x="{p2_x+p2_w-14}" y="{p2_y+19}" font-size="9" font-weight="600" fill="{MUTED}" text-anchor="end">Guaranteed Bounded Response</text>')

    # 4 Writer Preemption Scenarios
    svg.append(f'<text x="{p2_x+14}" y="{p2_y+44}" font-size="9" font-weight="700" fill="{INK}">WRITER PREEMPTION / CRASH POINT ANALYSIS</text>')

    preempt_cases = [
        ("Point A · Crash Before Step 1", "S is even (2k)", "Reader sees intact prior record (2k==2k). Zero corruption. ✓", TEAL),
        ("Point B · Preempted During Write", "S is odd (2k+1)", "Reader tests (S1 &amp; 1) != 0, drops buffer in &lt;5 µs. Zero blocking. ✓", AMBER),
        ("Point C · Crashed Mid-Publication", "S stays odd", "Reader detects persistent odd sequence, trips fallback clamp. ✓", CORAL),
        ("Point D · Write Fully Committed", "S is even (2k+2)", "Reader validates S1==S2==2k+2 and CRC. Accepts fresh packet. ✓", TEAL)
    ]
    for idx, (pt, state, res, col) in enumerate(preempt_cases):
        py = p2_y + 54 + idx * 42
        svg.append(f'<rect x="{p2_x+12}" y="{py}" width="{p2_w-24}" height="37" rx="5" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.1"/>')
        svg.append(f'<rect x="{p2_x+12}" y="{py}" width="4" height="37" rx="2" fill="{col}"/>')
        svg.append(f'<text x="{p2_x+22}" y="{py+14}" font-size="8.5" font-weight="700" fill="{col}">{pt}</text>')
        svg.append(f'<text x="{p2_x+200}" y="{py+14}" font-size="7.5" font-weight="600" fill="{MUTED}">[{state}]</text>')
        svg.append(f'<text x="{p2_x+22}" y="{py+28}" font-size="7.5" fill="{SLATE}">{res}</text>')

    # 4 Failure Classes & Deterministic Fallback Hierarchy
    svg.append(f'<text x="{p2_x+14}" y="{p2_y+238}" font-size="9" font-weight="700" fill="{CRIMSON}">4 BOUNDARY FAILURE CLASSES &amp; DETERMINISTIC RESPONSES</text>')

    fallbacks = [
        ("1. MALFORMED", "CRC mismatch / Out-of-bounds dot_m_cmd", "Clamp valve immediately to last verified safe orifice position.", CORAL),
        ("2. STALE", "t_now > t_prod + Δt_valid (Lease expired)", "Hold position for τ_hold = 10 ms; if persists, begin ramp-down.", AMBER),
        ("3. MISSING", "Sequence gap (k_new > k_prev + 1)", "Flag queue-overrun; apply state extrapolation with rate limit.", PURPLE),
        ("4. TORN / OVERRUN", "Seqlock mismatch (S1 != S2)", "Reject torn read instantly (&lt;5 µs); continue 1 kHz control tick.", BLUE)
    ]
    for idx, (fc, fsub, fresp, col) in enumerate(fallbacks):
        fby = p2_y + 248 + idx * 48
        svg.append(f'<rect x="{p2_x+12}" y="{fby}" width="{p2_w-24}" height="42" rx="5" fill="{col}" fill-opacity="0.06" stroke="{col}" stroke-width="1"/>')
        svg.append(f'<text x="{p2_x+22}" y="{fby+14}" font-size="8.5" font-weight="700" fill="{col}">{fc}: <tspan font-weight="500" fill="{INK}">{fsub}</tspan></text>')
        svg.append(f'<text x="{p2_x+22}" y="{fby+27}" font-size="7.5" font-weight="600" fill="{NAVY}">Fallback Action: <tspan font-weight="400" fill="{SLATE}">{fresp}</tspan></text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/13-placement/figures/fig13_lockfree_boundary_contract.svg", "\n".join(svg))


def run_all():
    gen_ch13_soc_contention_droop()
    gen_ch13_lockfree_boundary_contract()

if __name__ == "__main__":
    run_all()
