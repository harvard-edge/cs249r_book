#!/usr/bin/env python3
"""
Generate a Crisp Architectural Vector SVG of the Dual-Brain Architecture.
"""
import os

svg_content = """<svg width="960" height="620" viewBox="0 0 960 620" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 19px; fill: #1F407A; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-style: italic; font-size: 13px; fill: #666666; }
      .card-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 14.5px; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 12.5px; fill: #2D3748; }
      .badge-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 11px; }
      .label-text { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11.5px; font-weight: 600; }
    </style>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#007A87"/>
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
    </marker>
  </defs>

  <!-- Container Box -->
  <rect x="15" y="15" width="930" height="590" rx="8" fill="#FAFCFE" stroke="#E2E8F0" stroke-width="1.2"/>

  <!-- Titles -->
  <text x="480" y="48" text-anchor="middle" class="title">THE PHYSICAL AI DUAL-BRAIN ARCHITECTURE</text>
  <text x="480" y="70" text-anchor="middle" class="subtitle">Privilege Separation &amp; Safety Contracts on Heterogeneous Silicon</text>

  <!-- 1. UNTRUSTED PROPOSAL ENGINE -->
  <g transform="translate(45, 95)">
    <rect x="0" y="0" width="870" height="135" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.4"/>
    <rect x="0" y="0" width="8" height="135" rx="3" fill="#1F407A"/>
    
    <text x="24" y="28" class="card-title" fill="#1F407A">UNTRUSTED PROPOSAL ENGINE (Linux MPU / Edge NPU)</text>
    
    <!-- Badge -->
    <rect x="680" y="12" width="170" height="22" rx="4" fill="#1F407A"/>
    <text x="765" y="27" text-anchor="middle" class="badge-text" fill="#FFFFFF">UNTRUSTED PROPOSALS</text>

    <text x="24" y="58" class="body-text">• Multi-Modal Foundation Models (VLMs), Vision Encoders (ViT / DINOv2), Action Chunk Decoders (Diffusion / ACT)</text>
    <text x="24" y="82" class="body-text">• Asynchronous Deliberation: 1 Hz Semantic Intent → 20–50 Hz Action Chunking (Amortizes Inference Delay Across Time)</text>
    <text x="24" y="108" class="label-text" fill="#007A87">Emits: Expiring Intent Leases  pt = ⟨SE(3) Target, Workspace Bounding Volume, t_expire, Monotonic Counter⟩</text>
  </g>

  <!-- Connector 1 -->
  <line x1="480" y1="230" x2="480" y2="280" stroke="#007A87" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow-blue)"/>
  <rect x="330" y="243" width="300" height="24" rx="4" fill="#FFFFFF" stroke="#007A87" stroke-width="1"/>
  <text x="480" y="259" text-anchor="middle" class="badge-text" fill="#007A87">Shared SRAM Mailbox · Expiring Proposal pt (TTL ≤ 100 ms)</text>

  <!-- 2. TRUSTED PERMISSION AUTHORITY -->
  <g transform="translate(45, 285)">
    <rect x="0" y="0" width="870" height="135" rx="6" fill="#FBF2F3" stroke="#A51C30" stroke-width="1.4"/>
    <rect x="0" y="0" width="8" height="135" rx="3" fill="#A51C30"/>

    <text x="24" y="28" class="card-title" fill="#A51C30">TRUSTED PERMISSION AUTHORITY (Real-Time Bare-Metal MCU)</text>

    <!-- Badge -->
    <rect x="680" y="12" width="170" height="22" rx="4" fill="#A51C30"/>
    <text x="765" y="27" text-anchor="middle" class="badge-text" fill="#FFFFFF">SOLE PERMISSION VETO</text>

    <text x="24" y="58" class="body-text">• Dedicated 1000 Hz Timing Loop: Zero Dynamic Memory Heap (Static SRAM allocation, no malloc / GC jitter)</text>
    <text x="24" y="82" class="body-text">• Minimal-Intervention Control Barrier Functions (CBF: h(x) ≥ 0) + Dynamic Stopping Clearance Check (d_stop ≤ d_clear)</text>
    <text x="24" y="108" class="label-text" fill="#A51C30">Emergency Interlock (IEC 60204-1): Heartbeat Timeout (&gt; 20 ms) → SS1 Dynamic Braking → Safe Torque Off (STO)</text>
  </g>

  <!-- Connector 2 -->
  <line x1="480" y1="420" x2="480" y2="470" stroke="#A51C30" stroke-width="2" marker-end="url(#arrow-red)"/>
  <rect x="330" y="433" width="300" height="24" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="1"/>
  <text x="480" y="449" text-anchor="middle" class="badge-text" fill="#A51C30">Permitted Phase Current Setpoints ut = permit(pt) (20 kHz PWM)</text>

  <!-- 3. PHYSICAL WORLD -->
  <g transform="translate(45, 475)">
    <rect x="0" y="0" width="870" height="100" rx="6" fill="#FDF8F0" stroke="#B87333" stroke-width="1.4"/>
    <rect x="0" y="0" width="8" height="100" rx="3" fill="#B87333"/>

    <text x="24" y="28" class="card-title" fill="#B87333">THE PHYSICAL WORLD (Irreversible State Mutation: Wt ──► Wt+1)</text>

    <!-- Badge -->
    <rect x="680" y="12" width="170" height="22" rx="4" fill="#B87333"/>
    <text x="765" y="27" text-anchor="middle" class="badge-text" fill="#FFFFFF">NON-REVERSIBLE PHYSICS</text>

    <text x="24" y="58" class="body-text">• 3-Phase MOSFET Inverter Bridge · Stator Magnetic Flux · Kinetic Momentum (p = mv) · Joule Heating (I²R)</text>
    <text x="24" y="82" class="body-text" style="font-style: italic;">• Endogenous Feedback: Physical Mutation Wt+1 Instantly Shapes All Future Sensory Observations Ot+1</text>
  </g>
</svg>
"""

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_crisp_vector.svg"))
with open(out_path, "w", encoding="utf-8") as f:
    f.write(svg_content)

print("Saved Crisp Vector SVG to:", out_path)
