#!/usr/bin/env python3
"""
Generate 3 publication-grade textbook SVGs for Chapter 16 (Observability & Evaluation):
1. sample_size_confidence_intervals.svg
2. tail-sampling-pipeline.svg
3. observability-synthesis-architecture.svg

Style: Classic Hennessy & Patterson / Saltzer & Kaashoek / CS:APP computer systems textbook.
Clean rectangular functional blocks, clear dataflow, formal interfaces, minimal text in graphics.
"""

import os
import math

OUT_DIR = "books/vol3/16_observability/images/svg"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# 1. SAMPLE SIZE SCALING & WILSON SCORE CONFIDENCE BOUNDS
# ==============================================================================
def generate_sample_size_svg():
    def x_map(n):
        return 80 + (n / 2500.0) * 470.0

    def y_map(margin_pct):
        return 420 - (margin_pct / 25.0) * 310.0

    n_samples = [25, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    
    def wilson_margin(p_target, n, z=1.96):
        x = round(p_target * n)
        p_tilde = (x + z**2 / 2) / (n + z**2)
        margin = (z / (n + z**2)) * math.sqrt(x * (n - x) / n + z**2 / 4)
        return margin * 100.0

    pts_50 = [f"{x_map(n):.1f},{y_map(wilson_margin(0.50, n)):.1f}" for n in n_samples]
    pts_20 = [f"{x_map(n):.1f},{y_map(wilson_margin(0.20, n)):.1f}" for n in n_samples]
    pts_05 = [f"{x_map(n):.1f},{y_map(wilson_margin(0.05, n)):.1f}" for n in n_samples]

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 620" width="100%" height="100%">
  <defs>
    <style>
      .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }}
      .subtitle {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }}
      .panel-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12.5px; fill: #0f172a; }}
      .axis-label {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 600; font-size: 10.5px; fill: #334155; }}
      .tick-label {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #64748b; }}
      .legend-text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 600; font-size: 9.5px; fill: #1e293b; }}
      .card-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }}
      .card-body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }}
      .mono-text {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9.5px; }}
    </style>
  </defs>

  <!-- Outer Canvas Frame -->
  <rect x="0" y="0" width="1150" height="620" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="590" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">STATISTICAL SAMPLE SIZE SCALING &amp; WILSON SCORE CONFIDENCE BOUNDS</text>
  <text x="35" y="63" class="subtitle">Resolution of Agent Architectural Regressions: Error Margin Contraction vs. Boundary Catastrophes</text>

  <!-- PANEL A: ERROR MARGIN CONTRACTION CURVES -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="585" height="495" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="585" height="30" fill="#f1f5f9" rx="4" />
    <text x="16" y="20" class="panel-title">PANEL A: 95% Wilson Score Margin of Error (&#177;&#916;p) vs. Benchmark Sample Size N</text>

    <!-- Grid lines (Horizontal) -->
'''
    for pct in [0, 5, 10, 15, 20, 25]:
        y = y_map(pct)
        svg += f'''    <line x1="65" y1="{y:.1f}" x2="560" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1" />
    <text x="55" y="{y+3:.1f}" class="tick-label" text-anchor="end">&#177;{pct}%</text>
'''

    for n in [250, 500, 1000, 1500, 2000, 2500]:
        x = x_map(n)
        svg += f'''    <line x1="{x:.1f}" y1="{y_map(25):.1f}" x2="{x:.1f}" y2="{y_map(0):.1f}" stroke="#e2e8f0" stroke-width="1" />
    <text x="{x:.1f}" y="{y_map(0)+16:.1f}" class="tick-label" text-anchor="middle">N={n}</text>
'''

    svg += f'''
    <!-- Axes lines -->
    <line x1="65" y1="{y_map(0):.1f}" x2="560" y2="{y_map(0):.1f}" stroke="#64748b" stroke-width="1.5" />
    <line x1="65" y1="{y_map(0):.1f}" x2="65" y2="{y_map(25):.1f}" stroke="#64748b" stroke-width="1.5" />
    <text x="312" y="{y_map(0)+32:.1f}" class="axis-label" text-anchor="middle">Evaluation Harness Sample Size N (Independent Tasks)</text>
    <text x="18" y="{y_map(12.5):.1f}" class="axis-label" text-anchor="middle" transform="rotate(-90 18 {y_map(12.5):.1f})">Error Margin (&#177;&#916;p at 95% Confidence)</text>

    <!-- Horizontal Canary Threshold Line at +/- 2% -->
    <line x1="65" y1="{y_map(2.0):.1f}" x2="560" y2="{y_map(2.0):.1f}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 3" />
    <!-- Threshold Label placed safely below line on left side in clear region -->
    <text x="75" y="{y_map(2.0)+14:.1f}" class="mono-text" fill="#dc2626" font-weight="700">Canary Resolution Threshold: &#177;2.0%</text>

    <!-- Data Curves -->
    <!-- Curve 1: p = 0.50 (Max Variance) -->
    <polyline points="{' '.join(pts_50)}" fill="none" stroke="#2563eb" stroke-width="2.5" />
    <!-- Curve 2: p = 0.20 (Hard Agent Benchmark) -->
    <polyline points="{' '.join(pts_20)}" fill="none" stroke="#059669" stroke-width="2.5" />
    <!-- Curve 3: p = 0.05 (Rare-Event / Early Prototype) -->
    <polyline points="{' '.join(pts_05)}" fill="none" stroke="#d97706" stroke-width="2.5" />

    <!-- Specific Callout Dots & Annotations -->
    <!-- At N=100, p=0.50: +/- 9.62% -->
    <circle cx="{x_map(100):.1f}" cy="{y_map(9.62):.1f}" r="4" fill="#2563eb" />
    <rect x="{x_map(100)+10:.1f}" y="{y_map(9.62)-18:.1f}" width="115" height="20" fill="#ffffff" stroke="#2563eb" stroke-width="1" rx="2" />
    <text x="{x_map(100)+15:.1f}" y="{y_map(9.62)-4:.1f}" class="mono-text" fill="#2563eb" font-weight="600">N=100: &#177;9.6% span</text>

    <!-- At N=2400, p=0.50: +/- 2.0% -->
    <circle cx="{x_map(2400):.1f}" cy="{y_map(2.0):.1f}" r="4" fill="#dc2626" />
    <rect x="{x_map(2400)-165:.1f}" y="{y_map(2.0)-32:.1f}" width="160" height="20" fill="#ffffff" stroke="#dc2626" stroke-width="1" rx="2" />
    <text x="{x_map(2400)-85:.1f}" y="{y_map(2.0)-18:.1f}" class="mono-text" fill="#dc2626" font-weight="600" text-anchor="middle">N &#8805; 2,400 to resolve &#177;2.0%</text>

    <!-- Legend Box -->
    <g transform="translate(235, 42)">
      <rect x="0" y="0" width="315" height="90" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <line x1="12" y1="16" x2="35" y2="16" stroke="#2563eb" stroke-width="2.5" />
      <text x="42" y="19" class="legend-text">p = 0.50 (Max Variance, p(1-p)=0.25)</text>

      <line x1="12" y1="34" x2="35" y2="34" stroke="#059669" stroke-width="2.5" />
      <text x="42" y="37" class="legend-text">p = 0.20 (Hard Agent Benchmark, e.g. SWE-bench)</text>

      <line x1="12" y1="52" x2="35" y2="52" stroke="#d97706" stroke-width="2.5" />
      <text x="42" y="55" class="legend-text">p = 0.05 (Rare-Event Regime, Asymmetric Skew)</text>

      <line x1="12" y1="70" x2="35" y2="70" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 3" />
      <text x="42" y="73" class="legend-text" fill="#dc2626">Target Resolution (&#177;2.0% Canary Threshold)</text>
    </g>

    <!-- Bottom Takeaway Bar -->
    <rect x="15" y="456" width="555" height="28" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1" rx="3" />
    <text x="25" y="474" class="mono-text" fill="#1e40af">Key Insight: Error margin scales as 1/&#8730;N. Halving uncertainty requires 4&#215; evaluation scale.</text>
  </g>

  <!-- PANEL B: WALD BREAKDOWN VS. WILSON REGULARIZATION -->
  <g transform="translate(640, 80)">
    <rect x="0" y="0" width="475" height="495" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="475" height="30" fill="#f1f5f9" rx="4" />
    <text x="16" y="20" class="panel-title">PANEL B: Wald Breakdown vs. Wilson Regularization (N = 50 Tasks)</text>

    <!-- Card 1: Boundary Catastrophe (X = 0 / 50) -->
    <g transform="translate(15, 42)">
      <rect x="0" y="0" width="445" height="115" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="445" height="24" fill="#fee2e2" rx="3" />
      <text x="10" y="16" class="card-title" fill="#991b1b">REGIME 1: BOUNDARY CATASTROPHE (X = 0 / 50, Empirical Rate = 0.0%)</text>

      <rect x="10" y="32" width="208" height="50" fill="#fff1f2" stroke="#fca5a5" stroke-width="0.8" rx="2" />
      <text x="18" y="47" class="mono-text" fill="#b91c1c" font-weight="700">Wald Normal Approx:</text>
      <text x="18" y="62" class="mono-text" fill="#991b1b">CI = [0.0%, 0.0%] (&#177;0.0%)</text>
      <text x="18" y="74" class="card-body" fill="#b91c1c">&#9888; Degenerate: claims 100% certainty of 0%</text>

      <rect x="227" y="32" width="208" height="50" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8" rx="2" />
      <text x="235" y="47" class="mono-text" fill="#047857" font-weight="700">Wilson Score Interval:</text>
      <text x="235" y="62" class="mono-text" fill="#065f46">CI = [0.0%, 7.1%] (p_center = 3.6%)</text>
      <text x="235" y="74" class="card-body" fill="#047857">&#10004; Regularized: true ceiling reaches 7.1%</text>

      <text x="10" y="100" class="card-body" fill="#475569">Agresti-Coull prior adds +2 virtual successes and +2 virtual failures to anchor variance.</text>
    </g>

    <!-- Card 2: Negative Probability Trap (X = 2 / 50) -->
    <g transform="translate(15, 172)">
      <rect x="0" y="0" width="445" height="115" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="445" height="24" fill="#fef3c7" rx="3" />
      <text x="10" y="16" class="card-title" fill="#92400e">REGIME 2: NEGATIVE PROBABILITY TRAP (X = 2 / 50, Empirical Rate = 4.0%)</text>

      <rect x="10" y="32" width="208" height="50" fill="#fff1f2" stroke="#fca5a5" stroke-width="0.8" rx="2" />
      <text x="18" y="47" class="mono-text" fill="#b91c1c" font-weight="700">Wald Normal Approx:</text>
      <text x="18" y="62" class="mono-text" fill="#991b1b">CI = [-1.4%, 9.4%] (&#177;5.4%)</text>
      <text x="18" y="74" class="card-body" fill="#b91c1c">&#9888; Physically impossible negative bound</text>

      <rect x="227" y="32" width="208" height="50" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8" rx="2" />
      <text x="235" y="47" class="mono-text" fill="#047857" font-weight="700">Wilson Score Interval:</text>
      <text x="235" y="62" class="mono-text" fill="#065f46">CI = [1.1%, 13.5%] (p_center = 7.3%)</text>
      <text x="235" y="74" class="card-body" fill="#047857">&#10004; Asymmetric, strictly positive lower bound</text>

      <text x="10" y="100" class="card-body" fill="#475569">Inverts null hypothesis test; evaluates variance at hypothesized p rather than noisy point p_hat.</text>
    </g>

    <!-- Card 3: Symmetric Central Regime (X = 25 / 50) -->
    <g transform="translate(15, 302)">
      <rect x="0" y="0" width="445" height="100" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="445" height="24" fill="#f1f5f9" rx="3" />
      <text x="10" y="16" class="card-title" fill="#334155">REGIME 3: MAXIMUM VARIANCE (X = 25 / 50, Empirical Rate = 50.0%)</text>

      <rect x="10" y="32" width="208" height="42" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8" rx="2" />
      <text x="18" y="47" class="mono-text" fill="#334155">Wald: [36.1%, 63.9%] (&#177;13.9%)</text>
      <text x="18" y="61" class="card-body" fill="#64748b">Symmetric, wide uncertainty span</text>

      <rect x="227" y="32" width="208" height="42" fill="#eff6ff" stroke="#bfdbfe" stroke-width="0.8" rx="2" />
      <text x="235" y="47" class="mono-text" fill="#1e40af">Wilson: [36.6%, 63.4%] (&#177;13.4%)</text>
      <text x="235" y="61" class="card-body" fill="#1e40af">Slight shrinkage prior pull</text>

      <text x="10" y="88" class="card-body" fill="#475569">Both methods agree closely near p=0.50, but huge N=50 uncertainty (&#177;13%) remains.</text>
    </g>

    <!-- Engineering Decision Rule -->
    <g transform="translate(15, 415)">
      <rect x="0" y="0" width="445" height="67" fill="#f8fafc" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 2" rx="3" />
      <text x="12" y="18" class="card-title" fill="#0f172a">&#9679; SRE RELEASE RULE OF THUMB</text>
      <text x="12" y="34" class="card-body" fill="#334155">1. Never claim deployment superiority based on small benchmarks (N &lt; 200).</text>
      <text x="12" y="48" class="card-body" fill="#334155">2. Discard Wald confidence intervals for agent evaluation; enforce Wilson score bounds.</text>
      <text x="12" y="60" class="card-body" fill="#334155">3. Resolving a true &#916;p = 2.0% advantage requires N &gt; 2,000 independent eval tasks.</text>
    </g>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "sample_size_confidence_intervals.svg"), "w") as f:
        f.write(svg)
    print("Updated sample_size_confidence_intervals.svg")

# ==============================================================================
# 2. TAIL-BASED TELEMETRY SAMPLING PIPELINE
# ==============================================================================
def generate_tail_sampling_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 640" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .col-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
      .box-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 9.5px; fill: #64748b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }
      .mono-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #0f172a; }
      .arrow { stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#arrow-slate); }
      .arrow-blue { stroke: #2563eb; stroke-width: 1.5; fill: none; marker-end: url(#arrow-blue-marker); }
      .arrow-red { stroke: #dc2626; stroke-width: 1.5; fill: none; marker-end: url(#arrow-red-marker); }
      .arrow-green { stroke: #059669; stroke-width: 1.5; fill: none; marker-end: url(#arrow-green-marker); }
    </style>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arrow-blue-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
    </marker>
    <marker id="arrow-red-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#dc2626" />
    </marker>
    <marker id="arrow-green-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#059669" />
    </marker>
  </defs>

  <!-- Outer Canvas Frame -->
  <rect x="0" y="0" width="1150" height="640" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="610" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">TAIL-BASED TELEMETRY SAMPLING PIPELINE FOR AGENT RUNTIMES</text>
  <text x="35" y="63" class="subtitle">In-Memory Trace Escrow Buffering, Deterministic Anomaly Cascades, and Prioritized Storage Reduction</text>

  <!-- COLUMN 1: INGESTION & TRACE ESCROW BUFFER (Left) -->
  <g transform="translate(35, 85)">
    <rect x="0" y="0" width="295" height="430" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="295" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="col-title">1. IN-FLIGHT TRACE ESCROW</text>

    <!-- Concurrent Agent Fleet -->
    <rect x="15" y="40" width="265" height="85" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="25" y="58" class="box-title">Agent Execution Fleet (C = 128)</text>
    <text x="25" y="73" class="mono-text" fill="#475569">&#946; = S_trace / T_task (streaming load)</text>
    <text x="25" y="87" class="body-text">&#8226; GenAI Spans: Prefill/decode tokens, logits</text>
    <text x="25" y="101" class="body-text">&#8226; Tool Spans: Shell stdout/stderr, API diffs</text>
    <text x="25" y="115" class="mono-text" fill="#64748b">Raw Ingestion Load: 336 GB / day</text>

    <!-- Downward Arrow -->
    <line x1="147" y1="130" x2="147" y2="155" class="arrow" />

    <!-- Trace Escrow Ring Buffer -->
    <rect x="15" y="160" width="265" height="155" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.2" rx="3" />
    <text x="25" y="180" class="box-title" fill="#1e40af">In-Memory Trace Escrow Buffer</text>
    <text x="25" y="195" class="mono-text" fill="#2563eb">M_escrow &#8805; C &#183; S_trace &#8776; 2.0 GB RAM</text>
    <line x1="25" y1="202" x2="270" y2="202" stroke="#bfdbfe" stroke-width="1" />

    <!-- Escrow Slots -->
    <g transform="translate(25, 210)">
      <rect x="0" y="0" width="245" height="24" fill="#ffffff" stroke="#93c5fd" stroke-width="0.8" rx="2" />
      <text x="8" y="16" class="mono-text">Trace_0x4bf9 [turn 1..14] (In-Flight...)</text>
      
      <rect x="0" y="30" width="245" height="24" fill="#ffffff" stroke="#93c5fd" stroke-width="0.8" rx="2" />
      <text x="8" y="46" class="mono-text">Trace_0x7ea2 [turn 1..3]  (In-Flight...)</text>

      <rect x="0" y="60" width="245" height="24" fill="#ffffff" stroke="#93c5fd" stroke-width="0.8" rx="2" />
      <text x="8" y="76" class="mono-text">Trace_0x11c8 [turn 1..42] (Terminated)</text>
    </g>

    <text x="25" y="303" class="body-text" fill="#1e40af">&#9679; Holds full span DAG until terminal status event</text>

    <!-- Streaming Sanitizer Box -->
    <rect x="15" y="330" width="265" height="85" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="25" y="348" class="box-title">Pre-Egress Secret Sanitizer</text>
    <text x="25" y="364" class="mono-text" fill="#dc2626">&#8226; Regex Mask: API keys, JWTs, AWS tokens</text>
    <text x="25" y="378" class="body-text">&#8226; Named Entity Redaction: PII, email, IPs</text>
    <text x="25" y="394" class="body-text">&#8226; Principle: Redact BEFORE crossing node boundary</text>
  </g>

  <!-- Arrow from Col 1 to Col 2 with plenty of room -->
  <line x1="330" y1="280" x2="385" y2="280" class="arrow-blue" />
  <text x="357" y="268" class="mono-text" fill="#2563eb" text-anchor="middle">Trace DAG</text>

  <!-- COLUMN 2: DETERMINISTIC ANOMALY CLASSIFICATION CASCADE (Center) -->
  <g transform="translate(385, 85)">
    <rect x="0" y="0" width="380" height="430" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="380" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="col-title">2. DETERMINISTIC ANOMALY CASCADE</text>

    <!-- Priority 1: Failures & Violations -->
    <g transform="translate(15, 36)">
      <rect x="0" y="0" width="350" height="82" fill="#fef2f2" stroke="#fca5a5" stroke-width="1" rx="3" />
      <text x="12" y="20" class="box-title" fill="#991b1b">P1: Failures &amp; Policy Violations</text>
      <rect x="245" y="8" width="95" height="18" fill="#dc2626" rx="2" />
      <text x="292" y="21" class="mono-text" fill="#ffffff" font-weight="700" text-anchor="middle">100% RETAIN</text>

      <text x="12" y="42" class="mono-text" fill="#b91c1c">&#9679; Exit code &#8800; 0 | Schema error | Security trip</text>
      <text x="12" y="57" class="body-text">&#8226; Ingestion: 400 tasks/hr &#215; 1.4 MB = 560 MB/hr</text>
      <text x="12" y="72" class="body-text" fill="#7f1d1d">Forensics: Automated repro fixtures &amp; regression tests</text>
    </g>

    <!-- Priority 2: Latency & Token Outliers -->
    <g transform="translate(15, 124)">
      <rect x="0" y="0" width="350" height="82" fill="#fffbeb" stroke="#fde68a" stroke-width="1" rx="3" />
      <text x="12" y="20" class="box-title" fill="#92400e">P2: Latency &amp; Token Stragglers</text>
      <rect x="245" y="8" width="95" height="18" fill="#d97706" rx="2" />
      <text x="292" y="21" class="mono-text" fill="#ffffff" font-weight="700" text-anchor="middle">100% RETAIN</text>

      <text x="12" y="42" class="mono-text" fill="#b45309">&#9679; Latency &gt; p99 | Tokens &gt; p95 | Turns &gt; 3&#215; med</text>
      <text x="12" y="57" class="body-text">&#8226; Ingestion: 100 tasks/hr &#215; 1.4 MB = 140 MB/hr</text>
      <text x="12" y="72" class="body-text" fill="#78350f">Diagnostics: Autoregressive loops &amp; context bloat</text>
    </g>

    <!-- Priority 3: Human Interventions -->
    <g transform="translate(15, 212)">
      <rect x="0" y="0" width="350" height="82" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1" rx="3" />
      <text x="12" y="20" class="box-title" fill="#5b21b6">P3: Human Interventions &amp; Overrides</text>
      <rect x="245" y="8" width="95" height="18" fill="#7c3aed" rx="2" />
      <text x="292" y="21" class="mono-text" fill="#ffffff" font-weight="700" text-anchor="middle">100% RETAIN</text>

      <text x="12" y="42" class="mono-text" fill="#6d28d9">&#9679; User abort | Permission elevation | Correction</text>
      <text x="12" y="57" class="body-text">&#8226; Ingestion: 20 tasks/hr &#215; 1.4 MB = 28 MB/hr</text>
      <text x="12" y="72" class="body-text" fill="#4c1d95">Alignment: Expert demonstration &amp; DPO curation</text>
    </g>

    <!-- Priority 4: Routine Nominal Successes -->
    <g transform="translate(15, 300)">
      <rect x="0" y="0" width="350" height="82" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" rx="3" />
      <text x="12" y="20" class="box-title" fill="#065f46">P4: Routine Nominal Successes</text>
      <rect x="245" y="8" width="95" height="18" fill="#059669" rx="2" />
      <text x="292" y="21" class="mono-text" fill="#ffffff" font-weight="700" text-anchor="middle">2% RETAIN</text>

      <text x="12" y="42" class="mono-text" fill="#047857">&#9679; Objective verified (r = 1.0) | Latency &#8804; p90</text>
      <text x="12" y="57" class="body-text">&#8226; Ingestion: 9,500 &#215; 0.02 = 190 tasks/hr = 266 MB/hr</text>
      <text x="12" y="72" class="body-text" fill="#064e3b">98% Ephemeral Drop: Discarded from memory ring</text>
    </g>

    <text x="15" y="396" class="mono-text" fill="#475569">Deterministic Cascade: First matching predicate wins.</text>
    <text x="15" y="413" class="mono-text" fill="#475569">Sets root span attribute: `sampling.priority = {1, 0.02}`</text>
  </g>

  <!-- Arrows from Col 2 to Col 3 -->
  <line x1="765" y1="160" x2="800" y2="160" class="arrow-red" />
  <line x1="765" y1="360" x2="800" y2="360" class="arrow-green" />

  <!-- COLUMN 3: STORAGE TIERS & TELEMETRY ECONOMICS (Right) -->
  <g transform="translate(800, 85)">
    <rect x="0" y="0" width="315" height="430" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="315" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="col-title">3. DESTINATION TIERS &amp; COST</text>

    <!-- Top Tier: Forensic Replay Queue -->
    <rect x="15" y="38" width="285" height="110" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="15" y="38" width="285" height="22" fill="#fee2e2" rx="3" />
    <text x="25" y="53" class="box-title" fill="#991b1b">Forensic Incident &amp; Replay Queue</text>
    <text x="25" y="75" class="mono-text" fill="#b91c1c">&#9679; 100% of Failures, Stragglers &amp; Overrides</text>
    <text x="25" y="90" class="body-text">&#8226; Feeds hermetic counterfactual replay</text>
    <text x="25" y="104" class="body-text">&#8226; Generates new regression test fixtures</text>
    <text x="25" y="118" class="body-text">&#8226; Dispatches P1 alerts to SRE watchdogs</text>
    <text x="25" y="135" class="mono-text" fill="#64748b">Volume: 728 MB / hour (17.5 GB / day)</text>

    <!-- Middle Tier: OLAP Columnar Store -->
    <rect x="15" y="160" width="285" height="100" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="15" y="160" width="285" height="22" fill="#dcfce7" rx="3" />
    <text x="25" y="175" class="box-title" fill="#166534">High-Cardinality Columnar Store</text>
    <text x="25" y="196" class="mono-text" fill="#15803d">&#9679; ClickHouse / BigQuery / Parquet on S3</text>
    <text x="25" y="210" class="body-text">&#8226; Longitudinal drift monitoring</text>
    <text x="25" y="224" class="body-text">&#8226; Token economics &amp; latency percentiles</text>
    <text x="25" y="242" class="mono-text" fill="#64748b">Volume: 266 MB / hour (5.7 GB / day)</text>

    <!-- Bottom Tier: Storage Economics Synthesis -->
    <rect x="15" y="272" width="285" height="142" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.2" rx="3" />
    <text x="25" y="292" class="box-title" fill="#1e40af">Telemetry Storage Accounting</text>
    <line x1="25" y1="298" x2="290" y2="298" stroke="#bfdbfe" stroke-width="1" />

    <text x="25" y="315" class="mono-text" fill="#475569">Full Logging:      336.0 GB / day</text>
    <text x="25" y="332" class="mono-text" fill="#2563eb">Tail-Sampled:       23.2 GB / day</text>
    
    <rect x="25" y="342" width="265" height="26" fill="#1e3a8a" rx="2" />
    <text x="157" y="359" class="mono-text" fill="#ffffff" font-weight="700" text-anchor="middle">STORAGE REDUCTION: 93.1% (14.5&#215;)</text>

    <text x="25" y="385" class="body-text" fill="#1e40af">&#10004; Zero loss of anomalous signals</text>
    <text x="25" y="400" class="body-text" fill="#1e40af">&#10004; Eliminates linear telemetry tax on fleet scale</text>
  </g>

  <!-- BOTTOM ARCHITECTURAL INVARIANT BAR -->
  <g transform="translate(35, 530)">
    <rect x="0" y="0" width="1080" height="75" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <text x="20" y="22" class="box-title">SYSTEM INVARIANT: ZERO LOSS OF ANOMALOUS TRACE EVIDENCE</text>
    <text x="20" y="40" class="body-text">Head-based random sampling (e.g. uniform 5% retain) loses 95% of rare operational errors and edge cases. Tail-based sampling defers retention decisions until task completion,</text>
    <text x="20" y="55" class="body-text">guaranteeing that 100% of execution faults, invariant violations, and latency outliers are permanently preserved for offline post-mortems and automated regression synthesis.</text>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "tail-sampling-pipeline.svg"), "w") as f:
        f.write(svg)
    print("Updated tail-sampling-pipeline.svg")

# ==============================================================================
# 3. UNIFIED EMPIRICAL OBSERVABILITY CONTROL-PLANE ARCHITECTURE
# ==============================================================================
def generate_observability_synthesis_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 720" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .layer-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }
      .layer-num { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 10px; font-weight: 700; fill: #2563eb; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
      .box-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 9.5px; fill: #64748b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }
      .mono-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #0f172a; }
      .arrow { stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#arrow-slate); }
      .arrow-blue { stroke: #2563eb; stroke-width: 1.5; fill: none; marker-end: url(#arrow-blue-marker); }
      .arrow-green { stroke: #059669; stroke-width: 1.5; fill: none; marker-end: url(#arrow-green-marker); }
      .arrow-red { stroke: #dc2626; stroke-width: 1.5; fill: none; marker-end: url(#arrow-red-marker); }
    </style>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arrow-blue-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
    </marker>
    <marker id="arrow-green-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#059669" />
    </marker>
    <marker id="arrow-red-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#dc2626" />
    </marker>
  </defs>

  <!-- Outer Canvas Frame -->
  <rect x="0" y="0" width="1150" height="720" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="690" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">UNIFIED EMPIRICAL OBSERVABILITY &amp; EVALUATION CONTROL-PLANE ARCHITECTURE</text>
  <text x="35" y="63" class="subtitle">Closed-Loop Topology: Execution Telemetry, Streaming Sanitization, Offline Verification, and Canary Release Gating</text>

  <!-- 1. EXECUTION LAYER (Top Wide Box) -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="1080" height="95" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="26" fill="#f1f5f9" rx="4" />
    <text x="15" y="18" class="layer-title">1. EXECUTION LAYER: AGENT WORKER &amp; ISOLATION INFRASTRUCTURE</text>
    <text x="1060" y="18" class="layer-num" text-anchor="end">SLA: Overhead &#8804; 2%</text>

    <!-- Sub-box 1: Host Supervisor -->
    <g transform="translate(15, 33)">
      <rect x="0" y="0" width="330" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Host Agent Supervisor</text>
      <text x="12" y="32" class="body-text">&#8226; Generates root trace_id &amp; causal headers</text>
      <text x="12" y="45" class="mono-text" fill="#475569">W3C TraceContext Propagation</text>
    </g>

    <!-- Sub-box 2: Model Gateway -->
    <g transform="translate(365, 33)">
      <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Foundation Model Gateway</text>
      <text x="12" y="32" class="body-text">&#8226; Captures prompt/decode tokens, TTFT, logits</text>
      <text x="12" y="45" class="mono-text" fill="#475569">OpenTelemetry GenAI Semantic Conventions</text>
    </g>

    <!-- Sub-box 3: Sandboxes -->
    <g transform="translate(725, 33)">
      <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Isolated Execution Sandboxes</text>
      <text x="12" y="32" class="body-text">&#8226; MicroVM / Container POSIX stdout/stderr capture</text>
      <text x="12" y="45" class="mono-text" fill="#475569">Zero Ambient Authority (A = 0) Invariant</text>
    </g>
  </g>

  <!-- Downward Flow Arrow from Layer 1 to Layer 2 -->
  <line x1="575" y1="175" x2="575" y2="205" class="arrow-blue" />
  <text x="585" y="195" class="mono-text" fill="#2563eb">Raw OpenTelemetry Span Streams &amp; Context Carrier</text>

  <!-- 2. TELEMETRY COLLECTOR (Second Layer) -->
  <g transform="translate(35, 205)">
    <rect x="0" y="0" width="1080" height="95" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="26" fill="#f1f5f9" rx="4" />
    <text x="15" y="18" class="layer-title">2. TELEMETRY COLLECTOR: STREAMING SANITIZATION &amp; TAIL SAMPLING</text>
    <text x="1060" y="18" class="layer-num" text-anchor="end">SLA: Ingestion Lag &#8804; 200 ms</text>

    <!-- Sub-box 1: Streaming Redaction -->
    <g transform="translate(15, 33)">
      <rect x="0" y="0" width="515" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Streaming PII &amp; Secret Sanitizer</text>
      <text x="12" y="32" class="body-text">&#8226; Inline Regex &amp; NER masking of API keys, tokens, auth headers, and PII</text>
      <text x="12" y="45" class="mono-text" fill="#dc2626">Redaction Invariant: Secrets never cross node isolation wall</text>
    </g>

    <!-- Sub-box 2: Tail-based filter -->
    <g transform="translate(550, 33)">
      <rect x="0" y="0" width="515" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">In-Memory Escrow &amp; Tail-Based Filter</text>
      <text x="12" y="32" class="body-text">&#8226; Holds in-flight traces; retains 100% faults/stragglers, downsamples routine to 2%</text>
      <text x="12" y="45" class="mono-text" fill="#059669">Telemetry Reduction: 93.1% (14.5&#215; ingestion compression)</text>
    </g>
  </g>

  <!-- Flow Arrows from Collector downwards -->
  <line x1="290" y1="300" x2="290" y2="345" class="arrow-red" />
  <text x="180" y="328" class="mono-text" fill="#dc2626">100% Anomaly Traces</text>

  <line x1="860" y1="300" x2="860" y2="345" class="arrow-blue" />
  <text x="870" y="328" class="mono-text" fill="#2563eb">Canary Telemetry &amp; Metrics</text>

  <!-- 4. FORENSIC DIAGNOSTIC SERVICE (Left Column) -->
  <g transform="translate(35, 345)">
    <rect x="0" y="0" width="515" height="155" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="515" height="26" fill="#fee2e2" rx="4" />
    <text x="14" y="18" class="layer-title" fill="#991b1b">4. FORENSIC DIAGNOSTIC SERVICE</text>
    <text x="500" y="18" class="layer-num" fill="#991b1b" text-anchor="end">Root-Cause</text>

    <!-- Sub-box 1 -->
    <g transform="translate(15, 35)">
      <rect x="0" y="0" width="235" height="105" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="10" y="18" class="box-title">Deterministic Replay Engine</text>
      <text x="10" y="33" class="body-text">&#8226; Replays cached observations o_t</text>
      <text x="10" y="48" class="body-text">&#8226; Fixes PRNG seed &amp; env clock</text>
      <text x="10" y="63" class="body-text">&#8226; Isolated sandbox container</text>
      <text x="10" y="80" class="mono-text" fill="#64748b">Bit-accurate execution</text>
      <text x="10" y="95" class="mono-text" fill="#059669">&#10004; Zero network cross-talk</text>
    </g>

    <!-- Sub-box 2 -->
    <g transform="translate(265, 35)">
      <rect x="0" y="0" width="235" height="105" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="10" y="18" class="box-title">Counterfactual Perturbation</text>
      <text x="10" y="33" class="body-text">&#8226; System prompt ablation</text>
      <text x="10" y="48" class="body-text">&#8226; Tool availability masking</text>
      <text x="10" y="63" class="body-text">&#8226; Divergence localization</text>
      <text x="10" y="80" class="mono-text" fill="#64748b">Causal attribution</text>
      <text x="10" y="95" class="mono-text" fill="#dc2626">&#9679; Isolates tripping step</text>
    </g>
  </g>

  <!-- 5. RELEASE GATEWAY (Right Column) -->
  <g transform="translate(600, 345)">
    <rect x="0" y="0" width="515" height="155" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="515" height="26" fill="#eff6ff" rx="4" />
    <text x="14" y="18" class="layer-title" fill="#1e40af">5. RELEASE GATEWAY &amp; TRAFFIC ORCHESTRATOR</text>
    <text x="500" y="18" class="layer-num" text-anchor="end">MTTR &lt; 10 s</text>

    <!-- Sub-box 1 -->
    <g transform="translate(15, 35)">
      <rect x="0" y="0" width="235" height="105" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="10" y="18" class="box-title">Staged Traffic Routing</text>
      <text x="10" y="33" class="body-text">&#8226; Stage 1: Hermetic Gyms</text>
      <text x="10" y="48" class="body-text">&#8226; Stage 2: Shadow Escrow (0% auth)</text>
      <text x="10" y="63" class="body-text">&#8226; Stage 3: Canary (1% &#8594; 5% &#8594; 25%)</text>
      <text x="10" y="80" class="mono-text" fill="#2563eb">Progressive Autonomy</text>
      <text x="10" y="95" class="mono-text" fill="#15803d">&#10004; Zero user disruption</text>
    </g>

    <!-- Sub-box 2 -->
    <g transform="translate(265, 35)">
      <rect x="0" y="0" width="235" height="105" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="10" y="18" class="box-title">Automated Circuit Breakers</text>
      <text x="10" y="33" class="body-text">&#8226; Wald SPRT sequential testing</text>
      <text x="10" y="48" class="body-text">&#8226; Goodput drop &gt; 2% trips abort</text>
      <text x="10" y="63" class="body-text">&#8226; Instant clamp to 0% traffic</text>
      <text x="10" y="80" class="mono-text" fill="#dc2626">Hard tripwire switch</text>
      <text x="10" y="95" class="mono-text" fill="#b91c1c">&#9679; Instant rollback to baseline</text>
    </g>
  </g>

  <!-- Connective Arrows between Diagnostic / Release and Evaluation Layer -->
  <line x1="290" y1="500" x2="290" y2="535" class="arrow" />
  <line x1="860" y1="535" x2="860" y2="500" class="arrow-green" />
  <text x="870" y="522" class="mono-text" fill="#059669">Verified Gate Assertion</text>

  <!-- 3. EVALUATION ENGINE (Bottom Wide Box) -->
  <g transform="translate(35, 535)">
    <rect x="0" y="0" width="1080" height="95" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="26" fill="#f1f5f9" rx="4" />
    <text x="15" y="18" class="layer-title">3. EVALUATION ENGINE: OFFLINE HERMETIC BENCHMARKING</text>
    <text x="1060" y="18" class="layer-num" text-anchor="end">Gate 1 Standard: &#945; = 0.05</text>

    <!-- Sub-box 1: Hermetic Gyms -->
    <g transform="translate(15, 33)">
      <rect x="0" y="0" width="330" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Hermetic Evaluation Gyms</text>
      <text x="12" y="32" class="body-text">&#8226; MicroVM jailers, eBPF TAP, CoW storage</text>
      <text x="12" y="45" class="mono-text" fill="#475569">Zero network leakage; zero cross-talk</text>
    </g>

    <!-- Sub-box 2: Statistical Rigor -->
    <g transform="translate(365, 33)">
      <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Statistical Confidence Engine</text>
      <text x="12" y="32" class="body-text">&#8226; Wilson score interval bounds; sample scale N &#8805; 2000</text>
      <text x="12" y="45" class="mono-text" fill="#2563eb">Prevents Wald boundary catastrophes</text>
    </g>

    <!-- Sub-box 3: Oracle Matrix -->
    <g transform="translate(725, 33)">
      <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="box-title">Deterministic Ground-Truth Oracles</text>
      <text x="12" y="32" class="body-text">&#8226; Out-of-band test suites, AST diff validation</text>
      <text x="12" y="45" class="mono-text" fill="#059669">Strict binary contract V(&#964;) &#8712; {0, 1}</text>
    </g>
  </g>

  <!-- Closed-Loop Feedback Arrows -->
  <!-- Left Side: From Diagnostic up to Gyms (Bug regression fixtures) -->
  <path d="M 35 420 L 22 420 L 22 580 L 35 580" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="3 2" marker-end="url(#arrow-red-marker)" />
  <text x="15" y="495" class="mono-text" fill="#dc2626" transform="rotate(-90 15 495)" text-anchor="middle">Regress Fixtures</text>

  <!-- Right Side: Release Gateway Control loop back to Execution Layer -->
  <path d="M 1115 420 L 1128 420 L 1128 130 L 1115 130" fill="none" stroke="#2563eb" stroke-width="1.5" stroke-dasharray="3 2" marker-end="url(#arrow-blue-marker)" />
  <text x="1138" y="275" class="mono-text" fill="#2563eb" transform="rotate(90 1138 275)" text-anchor="middle">Traffic Weight &amp; Rollback Bus</text>

  <!-- Bottom Global Invariant -->
  <g transform="translate(35, 642)">
    <rect x="0" y="0" width="1080" height="50" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="15" y="20" class="box-title" fill="#0f172a">CONTROL-PLANE CLOSURE INVARIANT</text>
    <text x="15" y="38" class="body-text">Production telemetry directly drives offline test expansion; offline verification gates production traffic shifting; automated watchdogs enforce MTTR &lt; 10s rollback upon divergence.</text>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "observability-synthesis-architecture.svg"), "w") as f:
        f.write(svg)
    print("Updated observability-synthesis-architecture.svg")

if __name__ == "__main__":
    generate_sample_size_svg()
    generate_tail_sampling_svg()
    generate_observability_synthesis_svg()

