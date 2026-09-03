#!/usr/bin/env python3
"""
Generate publication-grade SVG and PDF figures for:
1. Dual-Brain Architecture (fig00_dual_brain_architecture)
2. Modular Curriculum Blueprint (fig03_modular_blueprint)
"""
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ASSETS_FIG_DIR = os.path.join(BOOK_DIR, "assets", "figures")
CH03_FIG_DIR = os.path.join(BOOK_DIR, "chapters", "03-workflow", "figures")

os.makedirs(ASSETS_FIG_DIR, exist_ok=True)
os.makedirs(CH03_FIG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Dual-Brain Architecture Figure
# -----------------------------------------------------------------------------
dual_brain_svg = """<svg width="960" height="580" viewBox="0 0 960 580" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 18px; fill: #1F407A; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-style: italic; font-size: 12.5px; fill: #666666; }
      .card-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 14px; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 12px; fill: #2D3748; }
      .badge-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 10.5px; }
      .label-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; font-weight: 600; }
    </style>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#007A87"/>
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
    </marker>
  </defs>

  <!-- Container Box -->
  <rect x="15" y="15" width="930" height="550" rx="8" fill="#FAFCFE" stroke="#E2E8F0" stroke-width="1.2"/>

  <!-- Titles -->
  <text x="480" y="45" text-anchor="middle" class="title">THE PHYSICAL AI DUAL-BRAIN ARCHITECTURE</text>
  <text x="480" y="65" text-anchor="middle" class="subtitle">Privilege Separation &amp; Safety Contracts on Heterogeneous Silicon</text>

  <!-- 1. UNTRUSTED PROPOSAL ENGINE -->
  <g transform="translate(45, 85)">
    <rect x="0" y="0" width="870" height="125" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.4"/>
    <rect x="0" y="0" width="8" height="125" rx="3" fill="#1F407A"/>
    
    <text x="24" y="26" class="card-title" fill="#1F407A">UNTRUSTED PROPOSAL ENGINE (Linux MPU / Edge NPU)</text>
    
    <!-- Badge -->
    <rect x="680" y="10" width="170" height="22" rx="4" fill="#1F407A"/>
    <text x="765" y="25" text-anchor="middle" class="badge-text" fill="#FFFFFF">UNTRUSTED PROPOSALS</text>

    <text x="24" y="52" class="body-text">• Multi-Modal Foundation Models (VLMs), Vision Encoders (ViT / DINOv2), Action Chunk Decoders (Diffusion / ACT)</text>
    <text x="24" y="74" class="body-text">• Asynchronous Deliberation: 1 Hz Semantic Intent → 20–50 Hz Action Chunking (Amortizes Inference Delay Across Time)</text>
    <text x="24" y="98" class="label-text" fill="#007A87">Emits: Expiring Intent Leases  pt = ⟨SE(3) Target, Workspace Bounding Volume, t_expire, Monotonic Counter⟩</text>
  </g>

  <!-- Connector 1 -->
  <line x1="480" y1="210" x2="480" y2="255" stroke="#007A87" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow-blue)"/>
  <rect x="330" y="222" width="300" height="22" rx="4" fill="#FFFFFF" stroke="#007A87" stroke-width="1"/>
  <text x="480" y="237" text-anchor="middle" class="badge-text" fill="#007A87">Shared SRAM Mailbox · Expiring Proposal pt (TTL ≤ 100 ms)</text>

  <!-- 2. TRUSTED PERMISSION AUTHORITY -->
  <g transform="translate(45, 255)">
    <rect x="0" y="0" width="870" height="125" rx="6" fill="#FBF2F3" stroke="#A51C30" stroke-width="1.4"/>
    <rect x="0" y="0" width="8" height="125" rx="3" fill="#A51C30"/>

    <text x="24" y="26" class="card-title" fill="#A51C30">TRUSTED PERMISSION AUTHORITY (Real-Time Bare-Metal MCU)</text>

    <!-- Badge -->
    <rect x="680" y="10" width="170" height="22" rx="4" fill="#A51C30"/>
    <text x="765" y="25" text-anchor="middle" class="badge-text" fill="#FFFFFF">SOLE PERMISSION VETO</text>

    <text x="24" y="52" class="body-text">• Dedicated 1000 Hz Timing Loop: Zero Dynamic Memory Heap (Static SRAM allocation, no malloc / GC jitter)</text>
    <text x="24" y="74" class="body-text">• Minimal-Intervention Control Barrier Functions (CBF: h(x) ≥ 0) + Dynamic Stopping Clearance Check (d_stop ≤ d_clear)</text>
    <text x="24" y="98" class="label-text" fill="#A51C30">Emergency Interlock (IEC 60204-1): Heartbeat Timeout (&gt; 20 ms) → SS1 Dynamic Braking → Safe Torque Off (STO)</text>
  </g>

  <!-- Connector 2 -->
  <line x1="480" y1="380" x2="480" y2="425" stroke="#A51C30" stroke-width="2" marker-end="url(#arrow-red)"/>
  <rect x="330" y="392" width="300" height="22" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="1"/>
  <text x="480" y="407" text-anchor="middle" class="badge-text" fill="#A51C30">Permitted Phase Current Setpoints ut = permit(pt) (20 kHz PWM)</text>

  <!-- 3. PHYSICAL WORLD -->
  <g transform="translate(45, 425)">
    <rect x="0" y="0" width="870" height="95" rx="6" fill="#FDF8F0" stroke="#B87333" stroke-width="1.4"/>
    <rect x="0" y="0" width="8" height="95" rx="3" fill="#B87333"/>

    <text x="24" y="26" class="card-title" fill="#B87333">THE PHYSICAL WORLD (Irreversible State Mutation: Wt ──► Wt+1)</text>

    <!-- Badge -->
    <rect x="680" y="10" width="170" height="22" rx="4" fill="#B87333"/>
    <text x="765" y="25" text-anchor="middle" class="badge-text" fill="#FFFFFF">NON-REVERSIBLE PHYSICS</text>

    <text x="24" y="52" class="body-text">• 3-Phase MOSFET Inverter Bridge · Stator Magnetic Flux · Kinetic Momentum (p = mv) · Joule Heating (I²R)</text>
    <text x="24" y="74" class="body-text" style="font-style: italic;">• Endogenous Feedback: Physical Mutation Wt+1 Instantly Shapes All Future Sensory Observations Ot+1</text>
  </g>
</svg>
"""

# -----------------------------------------------------------------------------
# 2. Modular Curriculum Blueprint Figure
# -----------------------------------------------------------------------------
modular_blueprint_svg = """<svg width="960" height="600" viewBox="0 0 960 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 18px; fill: #1F407A; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-style: italic; font-size: 12.5px; fill: #666666; }
      .part-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 13.5px; }
      .ch-num { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; font-weight: 700; }
      .ch-name { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 12px; }
      .ch-desc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 11px; fill: #4A5568; }
      .badge-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 10px; }
    </style>
  </defs>

  <!-- Container Box -->
  <rect x="15" y="15" width="930" height="570" rx="8" fill="#FAFCFE" stroke="#E2E8F0" stroke-width="1.2"/>

  <!-- Title -->
  <text x="480" y="44" text-anchor="middle" class="title">THE PHYSICAL AI CURRICULUM BLUEPRINT &amp; ROADMAP</text>
  <text x="480" y="64" text-anchor="middle" class="subtitle">Systematic Deconstruction: From Causal Foundations to Pipeline Organs &amp; Real-World Assurance</text>

  <!-- PART 1: THE FOUNDATIONAL TRIAD -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="890" height="135" rx="6" fill="#FDF8F0" stroke="#B87333" stroke-width="1.2"/>
    <rect x="0" y="0" width="890" height="28" rx="6" fill="#F6ECE0"/>
    <text x="16" y="19" class="part-title" fill="#B87333">PART 1: THE FOUNDATIONAL TRIAD (THE REALM OF PHYSICS &amp; LATENCY)</text>

    <!-- Ch 1 -->
    <g transform="translate(16, 38)">
      <rect x="0" y="0" width="270" height="85" rx="4" fill="#FFFFFF" stroke="#B87333" stroke-width="0.8"/>
      <text x="10" y="20" class="ch-num" fill="#B87333">CHAPTER 1</text>
      <text x="10" y="38" class="ch-name" fill="#1F407A">Physical Causality</text>
      <text x="10" y="56" class="ch-desc">Causal boundary, 3 criteria,</text>
      <text x="10" y="72" class="ch-desc">irreversible mutation (Wt → Wt+1)</text>
    </g>

    <!-- Ch 2 -->
    <g transform="translate(306, 38)">
      <rect x="0" y="0" width="270" height="85" rx="4" fill="#FFFFFF" stroke="#B87333" stroke-width="0.8"/>
      <text x="10" y="20" class="ch-num" fill="#B87333">CHAPTER 2</text>
      <text x="10" y="38" class="ch-name" fill="#1F407A">Time &amp; Latency Metrology</text>
      <text x="10" y="56" class="ch-desc">7-stage latency ledger, tail P99,</text>
      <text x="10" y="72" class="ch-desc">freshness wall &amp; stopping distance</text>
    </g>

    <!-- Ch 3 -->
    <g transform="translate(596, 38)">
      <rect x="0" y="0" width="278" height="85" rx="4" fill="#FFFFFF" stroke="#B87333" stroke-width="0.8"/>
      <text x="10" y="20" class="ch-num" fill="#B87333">CHAPTER 3</text>
      <text x="10" y="38" class="ch-name" fill="#1F407A">The Agent Workflow</text>
      <text x="10" y="56" class="ch-desc">Great Tug-of-War, 9-station lifecycle,</text>
      <text x="10" y="72" class="ch-desc">3 cadences &amp; multi-rate contracts</text>
    </g>
  </g>

  <!-- PART 2: THE 7 CANONICAL PIPELINE ORGANS -->
  <g transform="translate(35, 230)">
    <rect x="0" y="0" width="890" height="175" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.2"/>
    <rect x="0" y="0" width="890" height="28" rx="6" fill="#E2EBF6"/>
    <text x="16" y="19" class="part-title" fill="#1F407A">PART 2: THE 7 CANONICAL PIPELINE ORGANS (BUILDING THE WORKFLOW)</text>

    <!-- Ch 4 -->
    <g transform="translate(16, 38)">
      <rect x="0" y="0" width="162" height="120" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#1F407A">CHAPTER 4</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Perception</text>
      <text x="10" y="54" class="ch-desc">MIPI DMA stream</text>
      <text x="10" y="70" class="ch-desc">ViT / DINOv2 tokens</text>
      <text x="10" y="86" class="ch-desc">3D SE(3) affordances</text>
      <text x="10" y="104" class="ch-desc" style="font-weight:600; fill:#007A87;">[Stations 2 &amp; 3]</text>
    </g>

    <!-- Ch 5 -->
    <g transform="translate(190, 38)">
      <rect x="0" y="0" width="162" height="120" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#1F407A">CHAPTER 5</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">World Models</text>
      <text x="10" y="54" class="ch-desc">Latent JEPAs</text>
      <text x="10" y="70" class="ch-desc">Coordinate trees</text>
      <text x="10" y="86" class="ch-desc">Uncertainty bounds</text>
      <text x="10" y="104" class="ch-desc" style="font-weight:600; fill:#007A87;">[Station 4]</text>
    </g>

    <!-- Ch 6 -->
    <g transform="translate(364, 38)">
      <rect x="0" y="0" width="162" height="120" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#1F407A">CHAPTER 6</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Semantic Intent</text>
      <text x="10" y="54" class="ch-desc">1 Hz VLMs</text>
      <text x="10" y="70" class="ch-desc">Open-world goals</text>
      <text x="10" y="86" class="ch-desc">Expiring leases (TTL)</text>
      <text x="10" y="104" class="ch-desc" style="font-weight:600; fill:#007A87;">[Stations 1 &amp; 5]</text>
    </g>

    <!-- Ch 7 -->
    <g transform="translate(538, 38)">
      <rect x="0" y="0" width="162" height="120" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#1F407A">CHAPTER 7</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Action Chunking</text>
      <text x="10" y="54" class="ch-desc">Diffusion / ACT</text>
      <text x="10" y="70" class="ch-desc">Delay amortization</text>
      <text x="10" y="86" class="ch-desc">C² jerk continuous</text>
      <text x="10" y="104" class="ch-desc" style="font-weight:600; fill:#007A87;">[Station 6]</text>
    </g>

    <!-- Ch 8 -->
    <g transform="translate(712, 38)">
      <rect x="0" y="0" width="162" height="120" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#A51C30">CHAPTER 8</text>
      <text x="10" y="36" class="ch-name" fill="#A51C30">Real-Time Reflex</text>
      <text x="10" y="54" class="ch-desc">1 kHz CBF safety QP</text>
      <text x="10" y="70" class="ch-desc">Proposal-permission</text>
      <text x="10" y="86" class="ch-desc">IEC dynamic halts</text>
      <text x="10" y="104" class="ch-desc" style="font-weight:600; fill:#A51C30;">[Station 7]</text>
    </g>
  </g>

  <!-- PART 3: INTEGRATION, GOVERNANCE & DEPLOYMENT -->
  <g transform="translate(35, 420)">
    <rect x="0" y="0" width="890" height="145" rx="6" fill="#FBF2F3" stroke="#A51C30" stroke-width="1.2"/>
    <rect x="0" y="0" width="890" height="28" rx="6" fill="#F6E4E6"/>
    <text x="16" y="19" class="part-title" fill="#A51C30">PART 3: INTEGRATION, GOVERNANCE &amp; DEPLOYMENT (SYSTEM QUALIFICATION)</text>

    <!-- Ch 9 -->
    <g transform="translate(16, 38)">
      <rect x="0" y="0" width="205" height="90" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#A51C30">CHAPTER 9</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Workload Placement</text>
      <text x="10" y="54" class="ch-desc">MPU vs MCU vs NPU</text>
      <text x="10" y="70" class="ch-desc">UMA memory contention &amp; IPC</text>
    </g>

    <!-- Ch 10 -->
    <g transform="translate(233, 38)">
      <rect x="0" y="0" width="205" height="90" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#A51C30">CHAPTER 10</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Human Governance</text>
      <text x="10" y="54" class="ch-desc">Bumpless manual takeover</text>
      <text x="10" y="70" class="ch-desc">Telemetry logs &amp; intervention tags</text>
    </g>

    <!-- Ch 11 -->
    <g transform="translate(450, 38)">
      <rect x="0" y="0" width="205" height="90" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#A51C30">CHAPTER 11</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Assurance &amp; Release</text>
      <text x="10" y="54" class="ch-desc">STPA hazard mitigation</text>
      <text x="10" y="70" class="ch-desc">Claim-Argument-Evidence case</text>
    </g>

    <!-- Ch 12 -->
    <g transform="translate(667, 38)">
      <rect x="0" y="0" width="207" height="90" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="0.8"/>
      <text x="10" y="18" class="ch-num" fill="#A51C30">CHAPTER 12</text>
      <text x="10" y="36" class="ch-name" fill="#1F407A">Capstone Deployment</text>
      <text x="10" y="54" class="ch-desc">Full dual-brain release</text>
      <text x="10" y="70" class="ch-desc">Arduino UNO Q bench qualification</text>
    </g>
  </g>
</svg>
"""

# Save Dual-Brain SVG
dual_brain_svg_path = os.path.join(ASSETS_FIG_DIR, "fig00_dual_brain_architecture.svg")
with open(dual_brain_svg_path, "w", encoding="utf-8") as f:
    f.write(dual_brain_svg)
print(f"Saved: {dual_brain_svg_path}")

# Save Modular Blueprint SVG
blueprint_svg_path = os.path.join(CH03_FIG_DIR, "fig03_modular_blueprint.svg")
with open(blueprint_svg_path, "w", encoding="utf-8") as f:
    f.write(modular_blueprint_svg)
print(f"Saved: {blueprint_svg_path}")

# Rasterize both to PDF and PNG
for svg_p in [dual_brain_svg_path, blueprint_svg_path]:
    pdf_p = svg_p.replace(".svg", ".pdf")
    png_p = svg_p.replace(".svg", ".png")
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "pdf", "-o", pdf_p, svg_p], check=True)
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "png", "-w", "1600", "-o", png_p, svg_p], check=True)
    print(f"Compiled {os.path.basename(svg_p)} -> PDF & PNG")

print("All figures built successfully.")
