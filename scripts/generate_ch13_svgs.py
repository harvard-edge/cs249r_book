import os

svg_dir = "books/vol3/14_fine_tuning/images/svg"

# 1. ch13-block-diagonal-mask.svg
mask_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 480" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #334155; stroke-width: 1.5; rx: 6; ry: 6; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .panel-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; text-anchor: middle; }
      .axis-lbl { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 600; fill: #334155; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 11px; fill: #0f172a; text-anchor: middle; }
      .mono-red { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 11px; font-weight: 700; fill: #dc2626; text-anchor: middle; }
      .mono-green { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 11px; font-weight: 700; fill: #166534; text-anchor: middle; }
      .grid-cell { stroke: #94a3b8; stroke-width: 1; }
      .attn-active { fill: #dbeafe; stroke: #2563eb; stroke-width: 1; }
      .attn-leak { fill: #fee2e2; stroke: #dc2626; stroke-width: 1; }
      .attn-masked { fill: #f1f5f9; stroke: #cbd5e1; stroke-width: 1; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; }
    </style>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="460" y="30" class="title-text">Sequence Packing Attention Isolation: Standard Causal vs. 2D Block-Diagonal</text>
  <text x="460" y="50" class="sub-text">Preventing Cross-Task Information Leakage Across Concatenated Trajectories in Dense Batches</text>

  <!-- Left Panel: Unisolated Causal Attention -->
  <g transform="translate(50, 75)">
    <rect width="380" height="370" class="box" stroke="#dc2626" />
    <rect x="15" y="12" width="350" height="28" fill="#fef2f2" rx="4" />
    <text x="190" y="31" class="panel-hdr" fill="#991b1b">A. Unisolated Causal Attention (Contaminated)</text>

    <!-- Matrix Layout -->
    <!-- Column headers (Keys): tau_1 [50..180], tau_2 [190..320] -->
    <text x="125" y="68" class="axis-lbl" text-anchor="middle">Keys: Trajectory τ₁</text>
    <text x="265" y="68" class="axis-lbl" text-anchor="middle">Keys: Trajectory τ₂</text>

    <!-- Row headers (Queries): tau_1 [80..210], tau_2 [220..350] -->
    <text x="25" y="150" class="axis-lbl" transform="rotate(-90 25 150)" text-anchor="middle">Queries τ₁</text>
    <text x="25" y="290" class="axis-lbl" transform="rotate(-90 25 290)" text-anchor="middle">Queries τ₂</text>

    <!-- Block 1,1: tau_1 x tau_1 (Lower triangular active) -->
    <path d="M 60 80 L 190 210 L 60 210 Z" class="attn-active" />
    <path d="M 60 80 L 190 80 L 190 210 Z" class="attn-masked" />
    <text x="105" y="170" class="mono">τ₁ Attention</text>
    <text x="145" y="120" class="mono" fill="#94a3b8">0 (Future)</text>

    <!-- Block 1,2: tau_1 x tau_2 (Zero, future) -->
    <rect x="200" y="80" width="130" height="130" class="attn-masked" />
    <text x="265" y="150" class="mono" fill="#94a3b8">0 (Future)</text>

    <!-- Block 2,1: tau_2 x tau_1 (UNMASKED LEAKAGE!) -->
    <rect x="60" y="220" width="130" height="130" class="attn-leak" />
    <text x="125" y="275" class="mono-red">CROSS-TALK</text>
    <text x="125" y="295" class="mono-red">LEAKAGE</text>
    <text x="125" y="315" class="body-text" text-anchor="middle" fill="#dc2626">Attends to τ₁ state!</text>

    <!-- Block 2,2: tau_2 x tau_2 (Lower triangular active) -->
    <path d="M 200 220 L 330 350 L 200 350 Z" class="attn-active" />
    <path d="M 200 220 L 330 220 L 330 350 Z" class="attn-masked" />
    <text x="245" y="310" class="mono">τ₂ Attention</text>
    <text x="285" y="260" class="mono" fill="#94a3b8">0 (Future)</text>

    <!-- Defect note -->
    <text x="190" y="360" class="body-text" text-anchor="middle" font-weight="600" fill="#dc2626">Hazard: Policy learns spurious inter-task dependencies</text>
  </g>

  <!-- Right Panel: 2D Block-Diagonal Attention -->
  <g transform="translate(490, 75)">
    <rect width="380" height="370" class="box" stroke="#16a34a" />
    <rect x="15" y="12" width="350" height="28" fill="#f0fdf4" rx="4" />
    <text x="190" y="31" class="panel-hdr" fill="#166534">B. 2D Block-Diagonal Attention (Strictly Isolated)</text>

    <!-- Column headers (Keys) -->
    <text x="125" y="68" class="axis-lbl" text-anchor="middle">Keys: Trajectory τ₁</text>
    <text x="265" y="68" class="axis-lbl" text-anchor="middle">Keys: Trajectory τ₂</text>

    <!-- Row headers (Queries) -->
    <text x="25" y="150" class="axis-lbl" transform="rotate(-90 25 150)" text-anchor="middle">Queries τ₁</text>
    <text x="25" y="290" class="axis-lbl" transform="rotate(-90 25 290)" text-anchor="middle">Queries τ₂</text>

    <!-- Block 1,1: tau_1 x tau_1 (Lower triangular active) -->
    <path d="M 60 80 L 190 210 L 60 210 Z" class="attn-active" />
    <path d="M 60 80 L 190 80 L 190 210 Z" class="attn-masked" />
    <text x="105" y="170" class="mono">τ₁ Attention</text>
    <text x="145" y="120" class="mono" fill="#94a3b8">-∞</text>

    <!-- Block 1,2: tau_1 x tau_2 (Masked) -->
    <rect x="200" y="80" width="130" height="130" class="attn-masked" />
    <text x="265" y="150" class="mono" fill="#94a3b8">-∞</text>

    <!-- Block 2,1: tau_2 x tau_1 (MASKED: NO LEAKAGE) -->
    <rect x="60" y="220" width="130" height="130" class="attn-masked" />
    <text x="125" y="275" class="mono-green">MASKED</text>
    <text x="125" y="295" class="mono-green">A_ij = -∞</text>
    <text x="125" y="315" class="body-text" text-anchor="middle" fill="#166534">Zero attention weight</text>

    <!-- Block 2,2: tau_2 x tau_2 (Lower triangular active) -->
    <path d="M 200 220 L 330 350 L 200 350 Z" class="attn-active" />
    <path d="M 200 220 L 330 220 L 330 350 Z" class="attn-masked" />
    <text x="245" y="310" class="mono">τ₂ Attention</text>
    <text x="285" y="260" class="mono" fill="#94a3b8">-∞</text>

    <!-- Kernel implementation note -->
    <text x="190" y="360" class="mono" font-weight="600" fill="#166534">cu_seqlens = [0, c₁, c₂] (FlashAttention-2)</text>
  </g>

</svg>
"""

# 2. ch13-sft-evaluation-radar.svg
radar_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 520" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .axis-line { stroke: #cbd5e1; stroke-width: 1.5; }
      .ring-line { stroke: #e2e8f0; stroke-width: 1; fill: none; }
      .ring-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10px; fill: #94a3b8; }
      .axis-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 700; fill: #1e293b; text-anchor: middle; }
      .poly-base { fill: #94a3b8; fill-opacity: 0.15; stroke: #64748b; stroke-width: 2; stroke-dasharray: 4,4; }
      .poly-chka { fill: #f87171; fill-opacity: 0.2; stroke: #dc2626; stroke-width: 2.2; }
      .poly-chkb { fill: #60a5fa; fill-opacity: 0.25; stroke: #2563eb; stroke-width: 2.5; }
      .legend-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #1e293b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; }
    </style>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="460" y="30" class="title-text">Multi-Metric Evaluation Radar for Supervised Policy Checkpoints</text>
  <text x="460" y="50" class="sub-text">Comparative Acceptance Profile Across Task Completion, Syntax, Efficiency, and Retention</text>

  <!-- Radar Center: (cx = 380, cy = 270), Radius R = 170 -->
  <!-- 4 Axes:
       Top (0 deg, North): y = 270 - 170 = 100 -> Task Completion (Pass@1)
       Right (90 deg, East): x = 380 + 170 = 550 -> Syntactic Tool Validity
       Bottom (180 deg, South): y = 270 + 170 = 440 -> Trajectory Efficiency
       Left (270 deg, West): x = 380 - 170 = 210 -> General Capability Retention
  -->

  <!-- Concentric Rings: 25%, 50%, 75%, 100% -->
  <!-- R = 42.5 (25%), 85 (50%), 127.5 (75%), 170 (100%) -->
  <polygon points="380,227.5 422.5,270 380,312.5 337.5,270" class="ring-line" />
  <text x="384" y="232" class="ring-text">25%</text>

  <polygon points="380,185 465,270 380,355 295,270" class="ring-line" />
  <text x="384" y="189" class="ring-text">50%</text>

  <polygon points="380,142.5 507.5,270 380,397.5 252.5,270" class="ring-line" />
  <text x="384" y="146" class="ring-text">75%</text>

  <polygon points="380,100 550,270 380,440 210,270" class="ring-line" stroke="#94a3b8" stroke-width="1.2" />
  <text x="384" y="104" class="ring-text">100%</text>

  <!-- 4 Axis Lines -->
  <line x1="380" y1="270" x2="380" y2="90" class="axis-line" />
  <line x1="380" y1="270" x2="560" y2="270" class="axis-line" />
  <line x1="380" y1="270" x2="380" y2="450" class="axis-line" />
  <line x1="380" y1="270" x2="200" y2="270" class="axis-line" />

  <!-- Axis Titles -->
  <text x="380" y="80" class="axis-title">Task Completion (Pass@1 %)</text>
  <text x="575" y="274" class="axis-title" text-anchor="start">Syntactic Tool Validity (%)</text>
  <text x="380" y="470" class="axis-title">Trajectory Efficiency (1 / Tokens)</text>
  <text x="185" y="274" class="axis-title" text-anchor="end">General Capability Retention (%)</text>

  <!-- Model 1: Unadapted Base Model
       Task: 28% -> r = 47.6 -> y = 270 - 48 = 222
       Syntax: 42% -> r = 71.4 -> x = 380 + 71 = 451
       Efficiency: 35% -> r = 59.5 -> y = 270 + 60 = 330
       General: 95% -> r = 161.5 -> x = 380 - 162 = 218
  -->
  <polygon points="380,222 451,270 380,330 218,270" class="poly-base" />

  <!-- Model 2: Checkpoint A (Overfitted Surface SFT)
       Task: 52% -> r = 88.4 -> y = 270 - 88 = 182
       Syntax: 96% -> r = 163.2 -> x = 380 + 163 = 543
       Efficiency: 40% -> r = 68.0 -> y = 270 + 68 = 338
       General: 45% -> r = 76.5 -> x = 380 - 77 = 303 (Catastrophic Forgetting!)
  -->
  <polygon points="380,182 543,270 380,338 303,270" class="poly-chka" />

  <!-- Model 3: Checkpoint B (Balanced SFT with Action Masking & Dynamic Schema)
       Task: 78% -> r = 132.6 -> y = 270 - 133 = 137
       Syntax: 92% -> r = 156.4 -> x = 380 + 156 = 536
       Efficiency: 82% -> r = 139.4 -> y = 270 + 139 = 409
       General: 88% -> r = 149.6 -> x = 380 - 150 = 230
  -->
  <polygon points="380,137 536,270 380,409 230,270" class="poly-chkb" />

  <!-- Legend & Diagnostic Box (Right Side: x = 650 to 880) -->
  <g transform="translate(640, 110)">
    <rect width="250" height="290" fill="#f8fafc" stroke="#64748b" stroke-width="1.2" rx="6" />
    <text x="20" y="28" class="axis-title" text-anchor="start">Evaluation Scorecard</text>
    <line x1="20" y1="38" x2="230" y2="38" stroke="#cbd5e1" />

    <!-- Checkpoint B -->
    <line x1="20" y1="60" x2="50" y2="60" stroke="#2563eb" stroke-width="3" />
    <text x="60" y="64" class="legend-text" font-weight="700" fill="#1e40af">Checkpoint B (Balanced SFT)</text>
    <text x="20" y="82" class="body-text">• Action-targeted loss masking</text>
    <text x="20" y="98" class="body-text">• Dynamic schema regularization</text>
    <text x="20" y="114" class="body-text" font-weight="600" fill="#166534">Optimal production candidate</text>

    <!-- Checkpoint A -->
    <line x1="20" y1="140" x2="50" y2="140" stroke="#dc2626" stroke-width="2.5" />
    <text x="60" y="144" class="legend-text" font-weight="700" fill="#991b1b">Checkpoint A (Overfitted)</text>
    <text x="20" y="162" class="body-text">• Unmasked full-sequence SFT</text>
    <text x="20" y="178" class="body-text">• Fixed schema memorization</text>
    <text x="20" y="194" class="body-text" fill="#dc2626">Severe capability forgetting</text>

    <!-- Base Model -->
    <line x1="20" y1="220" x2="50" y2="220" stroke="#64748b" stroke-width="2" stroke-dasharray="4,4" />
    <text x="60" y="224" class="legend-text">Base Model (Unadapted)</text>
    <text x="20" y="242" class="body-text">• High general reasoning</text>
    <text x="20" y="258" class="body-text">• Poor agentic tool dispatch</text>
  </g>

</svg>
"""

# Replace symlinks with real files
files = [
    ("ch13-block-diagonal-mask.svg", mask_svg),
    ("ch13-sft-evaluation-radar.svg", radar_svg),
]

for filename, content in files:
    filepath = os.path.join(svg_dir, filename)
    if os.path.islink(filepath) or os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Wrote {filename}")

