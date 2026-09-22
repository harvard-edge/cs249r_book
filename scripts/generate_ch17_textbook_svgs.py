#!/usr/bin/env python3
"""
Generate 5 publication-grade textbook SVGs for Chapter 17 (Serving Economics):
1. fig-vol3-task-cost-treemap.svg
2. fig-vol3-critical-path-waterfall.svg
3. hierarchical-budget-ledger.svg
4. fig-vol3-architectural-radar-chart.svg
5. fig-vol3-fleet-economics-architecture.svg

Style: Classic Hennessy & Patterson / Saltzer & Kaashoek / CS:APP computer systems textbook.
Clean rectangular functional blocks, clear dataflow, formal interfaces, minimal text in graphics.
"""

import os
import math

OUT_DIR = "books/vol3/17_agent_economics/images/svg"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# 1. WHOLE-TRAJECTORY COST ACCOUNTING & EXPENDITURE BREAKDOWN
# ==============================================================================
def generate_task_cost_treemap_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 640" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .panel-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }
      .mono-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #0f172a; }
      .card-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
      .card-body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }
    </style>
  </defs>

  <!-- Outer Canvas Frame -->
  <rect x="0" y="0" width="1150" height="640" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="610" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">WHOLE-TRAJECTORY COST ACCOUNTING &amp; EXPENDITURE BREAKDOWN</text>
  <text x="35" y="63" class="subtitle">Physical Subsystem Cost Decomposition vs. The Token-Price Fallacy in Autonomous Trajectories</text>

  <!-- PANEL A: PHYSICAL SUBSYSTEM COST STACK -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="560" height="515" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="560" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="panel-title">PANEL A: Trajectory Cost Composition Stack (C_task = $0.420 Baseline)</text>

    <!-- Formula Box -->
    <g transform="translate(15, 38)">
      <rect x="0" y="0" width="530" height="36" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="22" class="mono-text" fill="#1e40af" font-weight="700">C_task = &#8721; (C_prefill + C_decode + C_tools + C_sandbox) + C_verify + C_human</text>
    </g>

    <!-- Stacked Cost Components (Treemap Bars) -->
    <g transform="translate(15, 84)">
      <!-- 1. Model Inference (Prefill + Decode) -->
      <rect x="0" y="0" width="530" height="82" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="530" height="22" fill="#dbeafe" rx="3" />
      <text x="10" y="15" class="box-title" fill="#1e40af">1. FOUNDATION MODEL INFERENCE: $0.174 (41.4% of Total)</text>
      <text x="10" y="38" class="mono-text" fill="#1e40af">&#8226; Prompt Prefill (C_prefill): $0.112 (26.7%)</text>
      <text x="10" y="52" class="body-text">Evaluates expanding context history across 10 turns (superlinear token growth).</text>
      <text x="10" y="68" class="mono-text" fill="#1e40af">&#8226; Autoregressive Decode (C_decode): $0.062 (14.7%) &#8212; Scratchpad reasoning &amp; tool parameters.</text>

      <!-- 2. Isolated Sandbox Hosting -->
      <g transform="translate(0, 92)">
        <rect x="0" y="0" width="530" height="74" fill="#f0fdf4" stroke="#86efac" stroke-width="1" rx="3" />
        <rect x="0" y="0" width="530" height="22" fill="#dcfce7" rx="3" />
        <text x="10" y="15" class="box-title" fill="#166534">2. ISOLATED SANDBOX RUNTIME: $0.110 (26.2% of Total)</text>
        <text x="10" y="38" class="mono-text" fill="#15803d">&#8226; MicroVM / Container Alloc (C_sandbox): 120 s lifetime &#215; 2 vCPU / 4 GB RAM</text>
        <text x="10" y="52" class="body-text">Storage I/O, ephemeral overlay mounts, and compiler memory footprints.</text>
        <text x="10" y="66" class="body-text">Note: Paid even when model is paused waiting for I/O or inter-process locks.</text>
      </g>

      <!-- 3. Tool API Egress & Verification -->
      <g transform="translate(0, 176)">
        <rect x="0" y="0" width="530" height="74" fill="#fffbeb" stroke="#fde68a" stroke-width="1" rx="3" />
        <rect x="0" y="0" width="530" height="22" fill="#fef3c7" rx="3" />
        <text x="10" y="15" class="box-title" fill="#92400e">3. EXTERNAL TOOLS &amp; VERIFIERS: $0.108 (25.7% of Total)</text>
        <text x="10" y="38" class="mono-text" fill="#b45309">&#8226; Tool API Calls (C_tools): $0.058 (13.8%) &#8212; Vector search indices &amp; web RPCs</text>
        <text x="10" y="52" class="mono-text" fill="#b45309">&#8226; Mechanical Verifiers (C_verify): $0.050 (11.9%) &#8212; Pytest suites, linters, AST diffs</text>
        <text x="10" y="66" class="body-text">Essential to enforce invariant closure before committing candidate modifications.</text>
      </g>

      <!-- 4. Amortized Human Escalation -->
      <g transform="translate(0, 260)">
        <rect x="0" y="0" width="530" height="66" fill="#fef2f2" stroke="#fca5a5" stroke-width="1" rx="3" />
        <rect x="0" y="0" width="530" height="22" fill="#fee2e2" rx="3" />
        <text x="10" y="15" class="box-title" fill="#991b1b">4. AMORTIZED HUMAN ESCALATION: $0.028 (6.7% of Total)</text>
        <text x="10" y="38" class="mono-text" fill="#b91c1c">&#8226; Operator Intervention (C_human): P(escalate) &#215; $3.50 triage labor</text>
        <text x="10" y="54" class="body-text">Unrecoverable deadlocks and permission elevations handled by human-in-the-loop.</text>
      </g>
    </g>

    <!-- Bottom Takeaway Bar -->
    <rect x="15" y="470" width="530" height="32" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1" rx="3" />
    <text x="25" y="490" class="mono-text" fill="#1e40af">Critical Takeaway: Non-model physical execution accounts for 58.6% of task expense!</text>
  </g>

  <!-- PANEL B: THE TOKEN-PRICE FALLACY -->
  <g transform="translate(620, 80)">
    <rect x="0" y="0" width="495" height="515" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="495" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="panel-title">PANEL B: The Token-Price Fallacy (Nominal Tariff vs. Effective Task Cost)</text>

    <!-- Effective Cost Formula -->
    <g transform="translate(15, 38)">
      <rect x="0" y="0" width="465" height="42" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="card-title" fill="#0f172a">Effective Cost per Accepted Task:</text>
      <text x="12" y="32" class="mono-text" fill="#2563eb">C_effective = E[C_attempt] / P(success) = &#8721; Fleet Spend / N_acceptable</text>
    </g>

    <!-- Strategy A Card -->
    <g transform="translate(15, 90)">
      <rect x="0" y="0" width="465" height="155" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="465" height="24" fill="#fee2e2" rx="3" />
      <text x="12" y="16" class="card-title" fill="#991b1b">STRATEGY A: NAIVE LOW-COST MODEL (Nominal Tariff = $0.20 / 1M)</text>

      <text x="12" y="42" class="mono-text" fill="#475569">&#8226; Nominal Token Fee per turn: $0.002 (10&#215; cheaper tariff!)</text>
      <text x="12" y="58" class="mono-text" fill="#dc2626">&#8226; Single-turn tool success rate: p = 0.65</text>
      <text x="12" y="74" class="mono-text" fill="#dc2626">&#8226; 8-turn task success probability: P(success) = 0.65^8 = 3.18%</text>
      <text x="12" y="90" class="body-text">&#8226; Result: 96.8% of trajectories fail in infinite loops or broken tests.</text>
      <text x="12" y="104" class="body-text">&#8226; Average attempts required to produce 1 accepted task: 31.4 rollouts</text>
      <text x="12" y="118" class="body-text">&#8226; Wasted sandbox hosting &amp; tool egress per failed attempt: $0.055</text>

      <rect x="12" y="126" width="440" height="22" fill="#fff1f2" stroke="#fca5a5" stroke-width="0.8" rx="2" />
      <text x="20" y="141" class="mono-text" fill="#991b1b" font-weight="700">EFFECTIVE COST PER ACCEPTED TASK: $1.85 (4.4&#215; MORE EXPENSIVE!)</text>
    </g>

    <!-- Strategy B Card -->
    <g transform="translate(15, 255)">
      <rect x="0" y="0" width="465" height="155" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="465" height="24" fill="#ecfdf5" rx="3" />
      <text x="12" y="16" class="card-title" fill="#065f46">STRATEGY B: FRONTIER REASONING CORE (Nominal Tariff = $15.00 / 1M)</text>

      <text x="12" y="42" class="mono-text" fill="#475569">&#8226; Nominal Token Fee per turn: $0.025 (12&#215; higher nominal tariff)</text>
      <text x="12" y="58" class="mono-text" fill="#059669">&#8226; Single-turn tool success rate: p = 0.96</text>
      <text x="12" y="74" class="mono-text" fill="#059669">&#8226; 8-turn task success probability: P(success) = 0.96^8 = 72.1%</text>
      <text x="12" y="90" class="body-text">&#8226; Result: 72% of trajectories complete cleanly on first attempt.</text>
      <text x="12" y="104" class="body-text">&#8226; Average attempts required to produce 1 accepted task: 1.38 rollouts</text>
      <text x="12" y="118" class="body-text">&#8226; Minimal sandbox thrashing; zero redundant compiler retries.</text>

      <rect x="12" y="126" width="440" height="22" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8" rx="2" />
      <text x="20" y="141" class="mono-text" fill="#065f46" font-weight="700">EFFECTIVE COST PER ACCEPTED TASK: $0.42 (OPTIMAL PRODUCTIVITY)</text>
    </g>

    <!-- Bottom Principle Box -->
    <g transform="translate(15, 420)">
      <rect x="0" y="0" width="465" height="82" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 2" rx="3" />
      <text x="12" y="18" class="card-title" fill="#0f172a">&#9679; THE LAW OF AGENTIC FLEET ECONOMICS</text>
      <text x="12" y="34" class="card-body" fill="#334155">In closed-loop agent systems, selecting an inferior model because of nominal token discounts</text>
      <text x="12" y="48" class="card-body" fill="#334155">multiplies turn counts, saturates sandbox capacity, and burns verification budgets.</text>
      <text x="12" y="64" class="mono-text" fill="#1e40af">Always optimize C_effective (cost per accepted task), never raw $/token.</text>
    </g>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "fig-vol3-task-cost-treemap.svg"), "w") as f:
        f.write(svg)
    print("Generated fig-vol3-task-cost-treemap.svg")

# ==============================================================================
# 2. WATERFALL TIMELINE DECOMPOSITION & AMDAHL BOUNDS
# ==============================================================================
def generate_critical_path_waterfall_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 640" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .panel-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }
      .mono-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #0f172a; }
      .card-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
      .card-body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }
      .arrow { stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#arrow-slate); }
    </style>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
  </defs>

  <!-- Outer Canvas Frame -->
  <rect x="0" y="0" width="1150" height="640" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="610" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">WATERFALL TIMELINE DECOMPOSITION OF AN AUTONOMOUS AGENT TURN</text>
  <text x="35" y="63" class="subtitle">Critical-Path Phase Breakdown: Prefill, Autoregressive Decode, Host Mediation, Sandbox Actuation, and Tool I/O</text>

  <!-- PANEL A: WATERFALL TIMELINE DECOMPOSITION -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="1080" height="280" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="panel-title">PANEL A: Turn Critical Path Decomposition (Total Turn Latency T_turn = 4.25 s)</text>

    <!-- Timeline Coordinate Frame -->
    <g transform="translate(20, 42)">
      <!-- Axis Ruler at top -->
      <line x1="280" y1="15" x2="1030" y2="15" stroke="#64748b" stroke-width="1.2" />
      <text x="280" y="10" class="mono-text" fill="#64748b" text-anchor="middle">0.0 s</text>
      <text x="456" y="10" class="mono-text" fill="#64748b" text-anchor="middle">1.0 s</text>
      <text x="633" y="10" class="mono-text" fill="#64748b" text-anchor="middle">2.0 s</text>
      <text x="809" y="10" class="mono-text" fill="#64748b" text-anchor="middle">3.0 s</text>
      <text x="986" y="10" class="mono-text" fill="#64748b" text-anchor="middle">4.0 s</text>
      <text x="1030" y="10" class="mono-text" fill="#64748b" text-anchor="middle">4.25 s</text>

      <!-- Gridlines -->
      <line x1="280" y1="15" x2="280" y2="215" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2 2" />
      <line x1="456" y1="15" x2="456" y2="215" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2 2" />
      <line x1="633" y1="15" x2="633" y2="215" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2 2" />
      <line x1="809" y1="15" x2="809" y2="215" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2 2" />
      <line x1="986" y1="15" x2="986" y2="215" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="2 2" />
      <line x1="1030" y1="15" x2="1030" y2="215" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="2 2" />

      <!-- Row 1: Prompt Prefill -->
      <text x="10" y="42" class="box-title">1. Prompt Prefill (T_prefill)</text>
      <text x="10" y="55" class="mono-text" fill="#2563eb">GEMM Tensor Core Compute-Bound</text>
      <rect x="280" y="30" width="63" height="30" fill="#bfdbfe" stroke="#2563eb" stroke-width="1" rx="2" />
      <text x="311" y="49" class="mono-text" fill="#1e40af" font-weight="700" text-anchor="middle">380 ms</text>

      <!-- Row 2: Autoregressive Decode -->
      <text x="10" y="82" class="box-title">2. Autoregressive Decode (T_decode)</text>
      <text x="10" y="95" class="mono-text" fill="#7c3aed">GEMV Memory-Bandwidth Bound</text>
      <rect x="343" y="70" width="185" height="30" fill="#ddd6fe" stroke="#7c3aed" stroke-width="1" rx="2" />
      <text x="435" y="89" class="mono-text" fill="#4c1d95" font-weight="700" text-anchor="middle">1,120 ms (48 tokens @ 43 tok/s)</text>

      <!-- Row 3: Host Supervisor Mediation -->
      <text x="10" y="122" class="box-title">3. Supervisor Mediation (T_runtime)</text>
      <text x="10" y="135" class="mono-text" fill="#475569">JSON parse, AST checks, WAL append</text>
      <rect x="528" y="110" width="25" height="30" fill="#e2e8f0" stroke="#64748b" stroke-width="1" rx="2" />
      <text x="560" y="129" class="mono-text" fill="#334155" font-weight="600">150 ms</text>

      <!-- Row 4: Isolated Sandbox Actuation -->
      <text x="10" y="162" class="box-title">4. Sandbox Actuation (T_tool)</text>
      <text x="10" y="175" class="mono-text" fill="#059669">MicroVM fork/exec, compiler, pytest</text>
      <rect x="553" y="150" width="321" height="30" fill="#bbf7d0" stroke="#059669" stroke-width="1" rx="2" />
      <text x="713" y="169" class="mono-text" fill="#065f46" font-weight="700" text-anchor="middle">1,950 ms (pytest suite execution)</text>

      <!-- Row 5: External Network RPC / Wait -->
      <text x="10" y="202" class="box-title">5. External Network Wait (T_wait)</text>
      <text x="10" y="215" class="mono-text" fill="#d97706">Remote Git fetch &amp; package resolution</text>
      <rect x="874" y="190" width="156" height="30" fill="#fde68a" stroke="#d97706" stroke-width="1" rx="2" />
      <text x="952" y="209" class="mono-text" fill="#78350f" font-weight="700" text-anchor="middle">650 ms (Network RPC)</text>
    </g>
  </g>

  <!-- PANEL B: AMDAHL'S LAW AND ACCELERATION ASYMPTOTES -->
  <g transform="translate(35, 380)">
    <rect x="0" y="0" width="1080" height="225" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="panel-title">PANEL B: Amdahl's Law Acceleration Asymptotes: S = 1 / [ (1 - f) + f / s ]</text>

    <!-- Card 1: Accelerating Model Inference (Left) -->
    <g transform="translate(15, 38)">
      <rect x="0" y="0" width="515" height="170" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="515" height="24" fill="#eff6ff" rx="3" />
      <text x="12" y="16" class="card-title" fill="#1e40af">CASE 1: ACCELERATING MODEL INFERENCE (f_model = 0.353, 35.3%)</text>

      <text x="12" y="42" class="body-text">Total Model Duration = T_prefill (0.38s) + T_decode (1.12s) = 1.50 s (35.3% of 4.25 s)</text>
      <text x="12" y="58" class="mono-text" fill="#475569">&#8226; Speculative Decoding (s = 2.0&#215;): S = 1 / [ 0.647 + 0.353/2 ] = 1.21&#215; (17.4% speedup)</text>
      <text x="12" y="74" class="mono-text" fill="#475569">&#8226; Massive Cluster TP=8 (s = 5.0&#215;): S = 1 / [ 0.647 + 0.353/5 ] = 1.39&#215; (28.1% speedup)</text>
      
      <rect x="12" y="86" width="490" height="38" fill="#fef2f2" stroke="#fca5a5" stroke-width="0.8" rx="2" />
      <text x="20" y="101" class="mono-text" fill="#991b1b" font-weight="700">INFINITE GPU ACCELERATION ASYMPTOTE (s_model &#8594; &#8734;):</text>
      <text x="20" y="116" class="mono-text" fill="#b91c1c">S_max = 1 / (1 - 0.353) = 1.545&#215; (Cannot exceed 35% total reduction!)</text>

      <text x="12" y="144" class="card-body" fill="#475569">Even with instantaneous zero-latency neural generation, 2.75s of sandbox/tool time remains.</text>
    </g>

    <!-- Card 2: Accelerating Environmental Execution (Right) -->
    <g transform="translate(550, 38)">
      <rect x="0" y="0" width="515" height="170" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="515" height="24" fill="#f0fdf4" rx="3" />
      <text x="12" y="16" class="card-title" fill="#166534">CASE 2: ACCELERATING ENVIRONMENT &amp; TOOLS (f_env = 0.647, 64.7%)</text>

      <text x="12" y="42" class="body-text">Total Environment Time = T_tool (1.95s) + T_wait (0.65s) + T_rt (0.15s) = 2.75 s (64.7%)</text>
      <text x="12" y="58" class="mono-text" fill="#15803d">&#8226; In-Memory tmpfs RAM-Disk + Test Cache (s = 2.5&#215;): S = 1.64&#215; (39.0% speedup)</text>
      <text x="12" y="74" class="mono-text" fill="#15803d">&#8226; Hermetic MicroVM Snapshotting (s = 4.0&#215;): S = 1 / [ 0.353 + 0.647/4 ] = 1.94&#215;</text>

      <rect x="12" y="86" width="490" height="38" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8" rx="2" />
      <text x="20" y="101" class="mono-text" fill="#065f46" font-weight="700">ENVIRONMENT OPTIMIZATION LEVERAGE:</text>
      <text x="20" y="116" class="mono-text" fill="#047857">A 2.5&#215; tool speedup produces MORE overall gain than an INFINITE GPU speedup!</text>

      <text x="12" y="144" class="card-body" fill="#475569">Systems Lesson: Optimize the dominant 65% non-neural critical path before upgrading GPUs.</text>
    </g>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "fig-vol3-critical-path-waterfall.svg"), "w") as f:
        f.write(svg)
    print("Generated fig-vol3-critical-path-waterfall.svg")

# ==============================================================================
# 3. HIERARCHICAL BUDGET RESERVATION LEDGER
# ==============================================================================
def generate_hierarchical_budget_ledger_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 640" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
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
  <rect x="0" y="0" width="1150" height="640" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="610" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">HIERARCHICAL BUDGET RESERVATION LEDGER &amp; ESCROW PROTOCOL</text>
  <text x="35" y="63" class="subtitle">Monotonic Financial Governance: Non-Overlapping Child Escrows, Conservation Invariants, and Terminal Refunds</text>

  <!-- TOP BLOCK: ROOT AGENT TASK LEDGER -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="1080" height="120" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="26" fill="#eff6ff" rx="4" />
    <text x="14" y="18" class="box-title" fill="#1e40af">ROOT AGENT TASK LEDGER: &#120027;_root = &#9001; Budget B, Settled E, Escrow R, Free F &#9002;</text>
    <text x="1060" y="18" class="mono-text" fill="#1e40af" text-anchor="end">Conservation Invariant: B = E + R + F</text>

    <!-- State Variable Cards with explicit local groups -->
    <g transform="translate(15, 36)">
      <!-- Budget B -->
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="245" height="70" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Allocated Budget (B_root)</text>
        <text x="12" y="40" class="mono-text" fill="#0f172a" font-size="16px" font-weight="700">$10.00</text>
        <text x="12" y="58" class="body-text" fill="#64748b">Static financial ceiling for task</text>
      </g>

      <!-- Settled Spend E -->
      <g transform="translate(265, 0)">
        <rect x="0" y="0" width="245" height="70" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title" fill="#dc2626">Settled Spend (E_root)</text>
        <text x="12" y="40" class="mono-text" fill="#dc2626" font-size="16px" font-weight="700">$0.40</text>
        <text x="12" y="58" class="body-text" fill="#64748b">Direct root planning inferences</text>
      </g>

      <!-- Escrow Reservation R -->
      <g transform="translate(530, 0)">
        <rect x="0" y="0" width="245" height="70" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title" fill="#2563eb">Active Escrow (R_root)</text>
        <text x="12" y="40" class="mono-text" fill="#2563eb" font-size="16px" font-weight="700">$6.00</text>
        <text x="12" y="58" class="body-text" fill="#64748b">Committed: Subagent 1 ($2) + 2 ($4)</text>
      </g>

      <!-- Free Balance F -->
      <g transform="translate(795, 0)">
        <rect x="0" y="0" width="255" height="70" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title" fill="#059669">Free Capital (F_root)</text>
        <text x="12" y="40" class="mono-text" fill="#059669" font-size="16px" font-weight="700">$3.60</text>
        <text x="12" y="58" class="body-text" fill="#64748b">Available for new child delegations</text>
      </g>
    </g>
  </g>

  <!-- DELEGATION ARROWS (Down from Root to Children) -->
  <line x1="295" y1="200" x2="295" y2="245" class="arrow-blue" />
  <text x="205" y="225" class="mono-text" fill="#2563eb">Reserve $2.00 Escrow</text>

  <line x1="855" y1="200" x2="855" y2="245" class="arrow-blue" />
  <text x="865" y="225" class="mono-text" fill="#2563eb">Reserve $4.00 Escrow</text>

  <!-- MIDDLE ROW: TWO PARALLEL CHILD AGENT LEDGERS -->
  <!-- CHILD 1: SEARCH SUBAGENT -->
  <g transform="translate(35, 245)">
    <rect x="0" y="0" width="520" height="180" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="520" height="26" fill="#f1f5f9" rx="4" />
    <text x="14" y="18" class="box-title">SUBAGENT 1: REPOSITORY SEARCHER (B_1 = $2.00 Allocated)</text>
    <text x="505" y="18" class="mono-text" fill="#059669" text-anchor="end">STATUS: COMPLETED</text>

    <!-- Ledger Details -->
    <g transform="translate(15, 35)">
      <rect x="0" y="0" width="490" height="42" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="mono-text" fill="#475569">Settled Spend: E_1 = $0.50 (15 Vector queries + 2 SLM parses)</text>
      <text x="12" y="32" class="mono-text" fill="#059669" font-weight="700">Unexpended Balance: F_1 = B_1 - E_1 = $2.00 - $0.50 = $1.50</text>

      <rect x="0" y="52" width="490" height="42" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8" rx="3" />
      <text x="12" y="68" class="box-title" fill="#065f46">&#10004; TERMINAL RETURN PROTOCOL:</text>
      <text x="12" y="84" class="mono-text" fill="#047857">Releases $1.50 unused reservation back to Root Free Capital F_root.</text>

      <text x="0" y="112" class="body-text" fill="#64748b">Local Draw Invariant: Child draws strictly from local escrow; cannot overdraw root.</text>
    </g>
  </g>

  <!-- CHILD 2: REFACTOR SUBAGENT -->
  <g transform="translate(595, 245)">
    <rect x="0" y="0" width="520" height="180" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="520" height="26" fill="#f1f5f9" rx="4" />
    <text x="14" y="18" class="box-title">SUBAGENT 2: CODE PATCH SYNTHESIZER (B_2 = $4.00 Allocated)</text>
    <text x="505" y="18" class="mono-text" fill="#059669" text-anchor="end">STATUS: COMPLETED</text>

    <g transform="translate(15, 35)">
      <rect x="0" y="0" width="490" height="42" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <text x="12" y="18" class="mono-text" fill="#475569">Settled Spend: E_2 = $3.10 (4 Model turns + 2 Compiler invocations)</text>
      <text x="12" y="32" class="mono-text" fill="#059669" font-weight="700">Unexpended Balance: F_2 = B_2 - E_2 = $4.00 - $3.10 = $0.90</text>

      <rect x="0" y="52" width="490" height="42" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8" rx="3" />
      <text x="12" y="68" class="box-title" fill="#065f46">&#10004; TERMINAL RETURN PROTOCOL:</text>
      <text x="12" y="84" class="mono-text" fill="#047857">Releases $0.90 unused reservation back to Root Free Capital F_root.</text>

      <text x="0" y="112" class="body-text" fill="#64748b">Fast-Trip Circuit Breaker: Auto-terminates if burn velocity &gt; $1.50/min.</text>
    </g>
  </g>

  <!-- REFUND ARROWS (Upward back to Reconciled Root) -->
  <line x1="295" y1="425" x2="295" y2="465" class="arrow-green" />
  <line x1="855" y1="425" x2="855" y2="465" class="arrow-green" />

  <!-- BOTTOM BLOCK: RECONCILED ROOT LEDGER & GOVERNANCE INVARIANTS -->
  <g transform="translate(35, 465)">
    <rect x="0" y="0" width="1080" height="140" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="26" fill="#f1f5f9" rx="4" />
    <text x="14" y="18" class="box-title">RECONCILED ROOT LEDGER &amp; ECONOMIC SETTLEMENT</text>
    <text x="1060" y="18" class="mono-text" fill="#15803d" text-anchor="end">&#10004; TASK VERIFIED &amp; COMMITTED</text>

    <g transform="translate(15, 36)">
      <!-- Left: Settlement Accounting -->
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="515" height="85" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title" fill="#1e40af">Final Settled Ledger Balance:</text>
        <text x="12" y="34" class="mono-text" fill="#475569">&#8226; Total Settled Spend: E_final = $0.40 (Root) + $0.50 (Child 1) + $3.10 (Child 2) = $4.00</text>
        <text x="12" y="50" class="mono-text" fill="#059669">&#8226; Total Released Refunds: R_refund = $1.50 + $0.90 = $2.40</text>
        <text x="12" y="66" class="mono-text" fill="#0f172a" font-weight="700">&#8226; Final Free Capital Restored: F_final = $10.00 - $4.00 = $6.00 (Zero Leakage!)</text>
      </g>

      <!-- Right: Three Governance Invariants -->
      <g transform="translate(535, 0)">
        <rect x="0" y="0" width="515" height="85" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title" fill="#0f172a">Three Formal Governance Invariants:</text>
        <text x="12" y="34" class="body-text">1. <tspan font-weight="700">Monotonicity:</tspan> Settled spend E_k is non-decreasing across all ticks (dE/dt &#8805; 0).</text>
        <text x="12" y="50" class="body-text">2. <tspan font-weight="700">Non-Overlapping Escrows:</tspan> &#8721; R_child &#8804; R_parent (Prevents concurrent double-spending).</text>
        <text x="12" y="66" class="body-text">3. <tspan font-weight="700">Hard Tripwire Clamping:</tspan> If F_k &lt; C_min, runtime traps OutOfBudgetException instantly.</text>
      </g>
    </g>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "hierarchical-budget-ledger.svg"), "w") as f:
        f.write(svg)
    print("Updated hierarchical-budget-ledger.svg")

# ==============================================================================
# 4. DIMENSIONAL TRADE-OFFS ACROSS ARCHITECTURAL AUTONOMY SPECTRUM
# ==============================================================================
def generate_architectural_radar_chart_svg():
    cx = 265
    cy = 310
    max_r = 145
    angles = [i * (2 * math.pi / 6) - math.pi/2 for i in range(6)]
    dim_names = [
      "1. Problem Flexibility\\n&amp; Epistemic Reach",
      "2. Execution Speed\\n&amp; Low Latency",
      "3. Financial Predictability\\n&amp; Low Cost",
      "4. Operational\\nDeterminism",
      "5. Verification\\nSimplicity",
      "6. Safety &amp; Low\\nBlast Radius"
    ]

    scores = {
      "sw1": [1.0, 10.0, 10.0, 10.0, 10.0, 9.5],
      "sw2": [4.0, 7.5, 8.0, 7.0, 8.0, 8.0],
      "bounded": [8.0, 4.0, 5.0, 5.0, 6.0, 6.0],
      "fleet": [10.0, 1.5, 2.0, 2.0, 2.5, 2.0]
    }

    def get_coords(score_list):
        pts = []
        for s, a in zip(score_list, angles):
            r = (s / 10.0) * max_r
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 620" width="100%" height="100%">
  <defs>
    <style>
      .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }}
      .subtitle {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }}
      .panel-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }}
      .box-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }}
      .body-text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }}
      .mono-text {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9px; fill: #0f172a; }}
      .card-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }}
      .card-body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 9.5px; fill: #334155; }}
      .radar-axis {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 600; font-size: 9.5px; fill: #334155; }}
    </style>
  </defs>

  <!-- Outer Canvas Frame -->
  <rect x="0" y="0" width="1150" height="620" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="590" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Main Header -->
  <text x="35" y="45" class="title">DIMENSIONAL TRADE-OFFS ACROSS THE ARCHITECTURAL AUTONOMY SPECTRUM</text>
  <text x="35" y="63" class="subtitle">Comparing Software 1.0, Software 2.0, Bounded Agentic Workflows, and Autonomous Fleets Across 6 System Dimensions</text>

  <!-- PANEL A: RADAR CHART (Left) -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="530" height="495" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="530" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="panel-title">PANEL A: 6-Dimensional Radar Comparison</text>

    <!-- Radar Background Webs -->
'''
    for r_level in [0.2, 0.4, 0.6, 0.8, 1.0]:
        r = r_level * max_r
        pts = [f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}" for a in angles]
        svg += f'    <polygon points="{" ".join(pts)}" fill="none" stroke="#e2e8f0" stroke-width="1" />\n'

    # Radar Spokes & Labels
    for i, a in enumerate(angles):
        x_end = cx + max_r * math.cos(a)
        y_end = cy + max_r * math.sin(a)
        svg += f'    <line x1="{cx}" y1="{cy}" x2="{x_end:.1f}" y2="{y_end:.1f}" stroke="#cbd5e1" stroke-width="1" />\n'
        x_lbl = cx + (max_r + 24) * math.cos(a)
        y_lbl = cy + (max_r + 16) * math.sin(a)
        anchor = "middle"
        if math.cos(a) > 0.2:
            anchor = "start"
        elif math.cos(a) < -0.2:
            anchor = "end"
        lbl_parts = dim_names[i].split("\\n")
        svg += f'    <text x="{x_lbl:.1f}" y="{y_lbl-4:.1f}" class="radar-axis" text-anchor="{anchor}">{lbl_parts[0]}</text>\n'
        if len(lbl_parts) > 1:
            svg += f'    <text x="{x_lbl:.1f}" y="{y_lbl+8:.1f}" class="radar-axis" text-anchor="{anchor}">{lbl_parts[1]}</text>\n'

    # Polygons
    svg += f'    <polygon points="{get_coords(scores["sw1"])}" fill="#059669" fill-opacity="0.15" stroke="#059669" stroke-width="2" />\n'
    svg += f'    <polygon points="{get_coords(scores["sw2"])}" fill="#2563eb" fill-opacity="0.15" stroke="#2563eb" stroke-width="2" />\n'
    svg += f'    <polygon points="{get_coords(scores["bounded"])}" fill="#d97706" fill-opacity="0.15" stroke="#d97706" stroke-width="2" />\n'
    svg += f'    <polygon points="{get_coords(scores["fleet"])}" fill="#dc2626" fill-opacity="0.15" stroke="#dc2626" stroke-width="2" />\n'

    # Legend Box placed cleanly at top left with ample room
    svg += '''    <g transform="translate(15, 38)">
      <rect x="0" y="0" width="205" height="74" fill="#ffffff" stroke="#cbd5e1" stroke-width="0.8" rx="2" />
      <line x1="10" y1="14" x2="28" y2="14" stroke="#059669" stroke-width="2.5" />
      <text x="35" y="17" class="mono-text" fill="#065f46" font-weight="700">Software 1.0 (Procedural)</text>

      <line x1="10" y1="30" x2="28" y2="30" stroke="#2563eb" stroke-width="2.5" />
      <text x="35" y="33" class="mono-text" fill="#1e40af" font-weight="700">Software 2.0 (Direct Model)</text>

      <line x1="10" y1="46" x2="28" y2="46" stroke="#d97706" stroke-width="2.5" />
      <text x="35" y="49" class="mono-text" fill="#92400e" font-weight="700">Bounded Agentic (K &#8804; K_max)</text>

      <line x1="10" y1="62" x2="28" y2="62" stroke="#dc2626" stroke-width="2.5" />
      <text x="35" y="65" class="mono-text" fill="#991b1b" font-weight="700">Autonomous Fleet (Emergent)</text>
    </g>
  </g>
'''

    # PANEL B: ARCHITECTURAL SELECTION RUBRIC (Right)
    svg += '''  <g transform="translate(585, 80)">
    <rect x="0" y="0" width="530" height="495" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="530" height="28" fill="#f1f5f9" rx="4" />
    <text x="14" y="19" class="panel-title">PANEL B: Production Architectural Selection Rubric</text>

    <!-- 1. SW 1.0 Card -->
    <g transform="translate(15, 36)">
      <rect x="0" y="0" width="500" height="98" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="500" height="22" fill="#ecfdf5" rx="3" />
      <text x="12" y="15" class="card-title" fill="#065f46">1. SOFTWARE 1.0 (DETERMINISTIC PROCEDURAL)</text>
      <text x="12" y="38" class="mono-text" fill="#047857">&#8226; Latency: &#956;s &#8211; ms | Cost: $0.00 | Variance: Zero</text>
      <text x="12" y="52" class="body-text">&#8226; Selection Rule: Use when the domain has formal grammars, relational schemas, or</text>
      <text x="12" y="66" class="body-text">  exact ASTs. Zero epistemic ambiguity. Fails completely on unstructured language.</text>
      <text x="12" y="82" class="mono-text" fill="#15803d">Target Workload: Compilers, static linters, SQL parsers, financial arithmetic.</text>
    </g>

    <!-- 2. SW 2.0 Card -->
    <g transform="translate(15, 142)">
      <rect x="0" y="0" width="500" height="98" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="500" height="22" fill="#eff6ff" rx="3" />
      <text x="12" y="15" class="card-title" fill="#1e40af">2. SOFTWARE 2.0 (DIRECT MODEL INVOCATION)</text>
      <text x="12" y="38" class="mono-text" fill="#2563eb">&#8226; Latency: 100 ms &#8211; 2 s | Cost: $10^-5 &#8211; $10^-3 | Variance: Bounded</text>
      <text x="12" y="52" class="body-text">&#8226; Selection Rule: Use for single-shot perceptual/translational tasks under grammar</text>
      <text x="12" y="66" class="body-text">  masking. Control flow remains strictly external; zero tool looping or self-healing.</text>
      <text x="12" y="82" class="mono-text" fill="#1e40af">Target Workload: Sentiment analysis, structured entity extraction, summarization.</text>
    </g>

    <!-- 3. Bounded Agentic Card -->
    <g transform="translate(15, 248)">
      <rect x="0" y="0" width="500" height="108" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="500" height="22" fill="#fffbeb" rx="3" />
      <text x="12" y="15" class="card-title" fill="#92400e">3. BOUNDED AGENTIC WORKFLOW (K &#8804; K_max, A = 0)</text>
      <text x="12" y="38" class="mono-text" fill="#b45309">&#8226; Latency: 5 s &#8211; 120 s | Cost: $0.01 &#8211; $0.50 | Invariants: Enforced</text>
      <text x="12" y="52" class="body-text">&#8226; Selection Rule: SWEET SPOT for software engineering tasks. Multi-turn feedback</text>
      <text x="12" y="66" class="body-text">  loop anchored by deterministic verification oracles (test suites, linters, builds).</text>
      <text x="12" y="80" class="body-text">  Authority A=0; hard turn limit K_max prevents runaway cost accumulation.</text>
      <text x="12" y="96" class="mono-text" fill="#78350f">Target Workload: Repository bug fixing, incident remediation, API integrations.</text>
    </g>

    <!-- 4. Autonomous Fleet Card -->
    <g transform="translate(15, 364)">
      <rect x="0" y="0" width="500" height="118" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
      <rect x="0" y="0" width="500" height="22" fill="#fee2e2" rx="3" />
      <text x="12" y="15" class="card-title" fill="#991b1b">4. AUTONOMOUS MULTI-AGENT FLEET (EMERGENT TOPOLOGY)</text>
      <text x="12" y="38" class="mono-text" fill="#dc2626">&#8226; Latency: Minutes &#8211; Hours | Cost: $1.00 &#8211; $50.00+ | Risk: Critical</text>
      <text x="12" y="52" class="body-text">&#8226; Selection Rule: MECHANISM OF LAST RESORT. Required only for vast search</text>
      <text x="12" y="66" class="body-text">  spaces with high epistemic uncertainty where tasks cannot be decomposed upfront.</text>
      <text x="12" y="80" class="body-text">  Demands hierarchical budget ledgers, microVM sandboxes, and human oversight.</text>
      <text x="12" y="96" class="mono-text" fill="#991b1b">Target Workload: Greenfield codebase synthesis, automated red-teaming audits.</text>
    </g>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "fig-vol3-architectural-radar-chart.svg"), "w") as f:
        f.write(svg)
    print("Updated fig-vol3-architectural-radar-chart.svg")

# ==============================================================================
# 5. FLEET ECONOMICS CONTROL LOOP ARCHITECTURE
# ==============================================================================
def generate_fleet_economics_architecture_svg():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 720" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .stage-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }
      .stage-num { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 10px; font-weight: 700; fill: #2563eb; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11px; fill: #0f172a; }
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
  <text x="35" y="45" class="title">THE FLEET ECONOMICS CONTROL LOOP ARCHITECTURE</text>
  <text x="35" y="63" class="subtitle">Five-Stage Closed-Loop Optimization Engine: Profiling, Scheduling, Cascading, Speculative Decoding, and Ledger Escrow</text>

  <!-- STAGE 1: ADMISSION WORKLOAD PROFILER -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="1080" height="92" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="24" fill="#f1f5f9" rx="4" />
    <text x="14" y="17" class="stage-title">STAGE 1: ADMISSION WORKLOAD PROFILER &amp; ESCROW SIZING</text>
    <text x="1060" y="17" class="stage-num" text-anchor="end">SLA: Latency &lt; 5 ms</text>

    <g transform="translate(15, 30)">
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="330" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Task Intent &amp; SLA Classifier</text>
        <text x="12" y="32" class="body-text">&#8226; Inspects prompt, deadline SLA, historical tokens</text>
        <text x="12" y="45" class="mono-text" fill="#2563eb">Partitions: Interactive vs Trajectory vs Batch</text>
      </g>

      <g transform="translate(350, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Resource Reservation Vector</text>
        <text x="12" y="32" class="body-text">&#8226; Computes initial token budget, memory ceiling</text>
        <text x="12" y="45" class="mono-text" fill="#475569">Limits: B_max, T_max, Sandboxed container class</text>
      </g>

      <g transform="translate(710, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Initial Escrow Allocation</text>
        <text x="12" y="32" class="body-text">&#8226; Earmarks funds in root task ledger</text>
        <text x="12" y="45" class="mono-text" fill="#059669">Asserts: R_root + F_root &#8804; B_tenant</text>
      </g>
    </g>
  </g>

  <!-- Flow Arrow -->
  <line x1="575" y1="172" x2="575" y2="198" class="arrow-blue" />

  <!-- STAGE 2: MLFQ SERVING SCHEDULER -->
  <g transform="translate(35, 198)">
    <rect x="0" y="0" width="1080" height="92" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="24" fill="#f1f5f9" rx="4" />
    <text x="14" y="17" class="stage-title">STAGE 2: MULTI-LEVEL FEEDBACK QUEUE (MLFQ) SERVING SCHEDULER</text>
    <text x="1060" y="17" class="stage-num" text-anchor="end">Kingman Clamping: &#961; &#8804; 0.70</text>

    <g transform="translate(15, 30)">
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="330" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Queue 0: Interactive &amp; Probes</text>
        <text x="12" y="32" class="body-text">&#8226; Quantum &#916;_0 = 64 tokens | Highest priority</text>
        <text x="12" y="45" class="mono-text" fill="#2563eb">Non-preemptible; minimizes TTFT</text>
      </g>

      <g transform="translate(350, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Queue 1: Multi-Turn Reasoning</text>
        <text x="12" y="32" class="body-text">&#8226; Quantum &#916;_1 = 512 tokens | Medium priority</text>
        <text x="12" y="45" class="mono-text" fill="#475569">Preemptible by Q_0; swaps KV on tool wait</text>
      </g>

      <g transform="translate(710, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Queue 2: Autonomous Batch Fleets</text>
        <text x="12" y="32" class="body-text">&#8226; Quanta &#916;_2 &#8805; 4,096 tokens | Lowest priority</text>
        <text x="12" y="45" class="mono-text" fill="#059669">High-throughput continuous batching</text>
      </g>
    </g>
  </g>

  <!-- Flow Arrow -->
  <line x1="575" y1="290" x2="575" y2="316" class="arrow-blue" />

  <!-- STAGE 3: TIERED MODEL ROUTING CASCADE -->
  <g transform="translate(35, 316)">
    <rect x="0" y="0" width="1080" height="92" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="24" fill="#f1f5f9" rx="4" />
    <text x="14" y="17" class="stage-title">STAGE 3: TIERED MODEL ROUTING CASCADE (FRUGALGPT STATE MACHINE)</text>
    <text x="1060" y="17" class="stage-num" text-anchor="end">Cost Reduction: 92.1%</text>

    <g transform="translate(15, 30)">
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="330" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Tier 1: SLM / Deterministic Script</text>
        <text x="12" y="32" class="body-text">&#8226; $0.05 / 1M | Syntax validation, AST parsing</text>
        <text x="12" y="45" class="mono-text" fill="#059669">65% Tasks Exit Here ($0.00005/call)</text>
      </g>

      <g transform="translate(350, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Tier 2: Mid-Tier Generalist</text>
        <text x="12" y="32" class="body-text">&#8226; $0.40 / 1M | Routine tool dispatch &amp; editing</text>
        <text x="12" y="45" class="mono-text" fill="#2563eb">28% Tasks Exit Here ($0.0004/call)</text>
      </g>

      <g transform="translate(710, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Tier 3: Frontier Reasoning Core</text>
        <text x="12" y="32" class="body-text">&#8226; $15.00 / 1M | Subtle error recovery, synthesis</text>
        <text x="12" y="45" class="mono-text" fill="#dc2626">7% Escalation Only ($0.015/call)</text>
      </g>
    </g>
  </g>

  <!-- Flow Arrow -->
  <line x1="575" y1="408" x2="575" y2="434" class="arrow-blue" />

  <!-- STAGE 4: SPECULATIVE DECODING ENGINE -->
  <g transform="translate(35, 434)">
    <rect x="0" y="0" width="1080" height="92" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="24" fill="#f1f5f9" rx="4" />
    <text x="14" y="17" class="stage-title">STAGE 4: SPECULATIVE DECODING &amp; AMDAHL CRITICAL-PATH ACCELERATION</text>
    <text x="1060" y="17" class="stage-num" text-anchor="end">Speedup: 2.2&#215; Decode Latency</text>

    <g transform="translate(15, 30)">
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="515" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Draft Model Speculation (&#947; = 3..5 Tokens)</text>
        <text x="12" y="32" class="body-text">&#8226; Fast small draft model emits candidate token sequence at low arithmetic intensity</text>
        <text x="12" y="45" class="mono-text" fill="#7c3aed">Transforms memory-bandwidth bound decode into parallel GEMM verification</text>
      </g>

      <g transform="translate(550, 0)">
        <rect x="0" y="0" width="515" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Target Model Parallel Verification Pass</text>
        <text x="12" y="32" class="body-text">&#8226; Target model verifies block in 1 forward pass; preserves exact output distribution</text>
        <text x="12" y="45" class="mono-text" fill="#059669">Zero Epistemic Drift; mathematical equivalence to pure target generation</text>
      </g>
    </g>
  </g>

  <!-- Flow Arrow -->
  <line x1="575" y1="526" x2="575" y2="552" class="arrow-blue" />

  <!-- STAGE 5: HIERARCHICAL BUDGET LEDGER & INVARIANT ESCROW -->
  <g transform="translate(35, 552)">
    <rect x="0" y="0" width="1080" height="92" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4" />
    <rect x="0" y="0" width="1080" height="24" fill="#f1f5f9" rx="4" />
    <text x="14" y="17" class="stage-title">STAGE 5: HIERARCHICAL BUDGET LEDGER &amp; VERIFICATION ORACLES</text>
    <text x="1060" y="17" class="stage-num" text-anchor="end">Invariant Closure: A = 0</text>

    <g transform="translate(15, 30)">
      <g transform="translate(0, 0)">
        <rect x="0" y="0" width="330" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Hierarchical Escrow Ledger</text>
        <text x="12" y="32" class="body-text">&#8226; Non-overlapping reservations to subagents</text>
        <text x="12" y="45" class="mono-text" fill="#1e40af">Conservation Invariant: B = E + R + F</text>
      </g>

      <g transform="translate(350, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Automated Circuit Breakers</text>
        <text x="12" y="32" class="body-text">&#8226; Trips on burn velocity &gt; &#952; or loop detection</text>
        <text x="12" y="45" class="mono-text" fill="#dc2626">Hard clamp: MTTR &lt; 10s rollback</text>
      </g>

      <g transform="translate(710, 0)">
        <rect x="0" y="0" width="340" height="52" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
        <text x="12" y="18" class="box-title">Deterministic Mechanical Oracle</text>
        <text x="12" y="32" class="body-text">&#8226; Sandboxed pytest, AST diffs, linters</text>
        <text x="12" y="45" class="mono-text" fill="#059669">Verified Commit: Output y* accepted</text>
      </g>
    </g>
  </g>

  <!-- BOTTOM CLOSED FEEDBACK BUS (Looping back from Stage 5 to Stage 1) -->
  <path d="M 35 598 L 22 598 L 22 126 L 35 126" fill="none" stroke="#059669" stroke-width="1.5" stroke-dasharray="3 2" marker-end="url(#arrow-green-marker)" />
  <text x="15" y="362" class="mono-text" fill="#059669" transform="rotate(-90 15 362)" text-anchor="middle">Telemetry Feedback: Goodput/Dollar (GP_$) Calibration Loop</text>

  <!-- Bottom Global Invariant -->
  <g transform="translate(35, 654)">
    <rect x="0" y="0" width="1080" height="40" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1" rx="3" />
    <text x="15" y="24" class="mono-text" fill="#1e40af">SYSTEM OPTIMIZATION INVARIANT: Maximize Goodput per Dollar (GP_$ = N_acceptable / &#8721; C_task) while satisfying p95 Latency &lt; SLA.</text>
  </g>
</svg>'''
    with open(os.path.join(OUT_DIR, "fig-vol3-fleet-economics-architecture.svg"), "w") as f:
        f.write(svg)
    print("Updated fig-vol3-fleet-economics-architecture.svg")

if __name__ == "__main__":
    generate_task_cost_treemap_svg()
    generate_critical_path_waterfall_svg()
    generate_hierarchical_budget_ledger_svg()
    generate_architectural_radar_chart_svg()
    generate_fleet_economics_architecture_svg()
