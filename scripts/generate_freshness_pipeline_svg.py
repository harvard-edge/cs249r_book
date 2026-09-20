#!/usr/bin/env python3
"""
Generate publication-grade textbook SVG for asynchronous-freshness-pipeline.svg
Illustrates:
(a) Execution Timeline: Synchronous Lockstep (Straggler-Bound) vs Asynchronous Streaming (Decoupled)
(b) Freshness-Bounded Ingestion Architecture: Rollout fleet, Circular Replay Buffer, Version Admission Gate, Truncated Importance Sampling, and Training Engine.
Textbook style: Hennessy & Patterson / Saltzer & Kaashoek systems engineering.
"""

import os

svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 720" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 600; font-size: 12px; fill: #475569; }
      .panel-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 13px; fill: #0f172a; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #1e293b; }
      .box-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 10.5px; fill: #64748b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 11px; fill: #334155; }
      .code-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 10.5px; fill: #0f172a; }
      .badge-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 10px; }
      .arrow { stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#arrow-slate); }
      .arrow-blue { stroke: #2563eb; stroke-width: 1.5; fill: none; marker-end: url(#arrow-blue-marker); }
      .arrow-emerald { stroke: #059669; stroke-width: 1.5; fill: none; marker-end: url(#arrow-emerald-marker); }
      .arrow-rose { stroke: #e11d48; stroke-width: 1.5; fill: none; marker-end: url(#arrow-rose-marker); }
    </style>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arrow-blue-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
    </marker>
    <marker id="arrow-emerald-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#059669" />
    </marker>
    <marker id="arrow-rose-marker" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e11d48" />
    </marker>
  </defs>

  <!-- Canvas Background -->
  <rect x="0" y="0" width="1150" height="720" fill="#ffffff" />

  <!-- Outer Frame -->
  <rect x="15" y="15" width="1120" height="690" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- TOP PANEL: (a) TIMELINE COMPARISON: LOCKSTEP VS ASYNCHRONOUS -->
  <rect x="30" y="30" width="1090" height="280" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="4" />
  <text x="45" y="52" class="panel-label">(a) Execution Timelines: Synchronous Lockstep vs. Asynchronous Streaming</text>

  <!-- Left Side: Synchronous Lockstep (Straggler-Bound) -->
  <g transform="translate(45, 65)">
    <rect x="0" y="0" width="515" height="230" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="15" y="24" class="box-title">Synchronous Lockstep Pipeline (Straggler-Bound)</text>
    <text x="15" y="40" class="box-sub">Global barrier forces training cluster to idle awaiting slowest rollout</text>

    <!-- Timeline Tracks -->
    <!-- Trainer Track -->
    <text x="15" y="68" class="code-text" font-weight="bold">Trainer:</text>
    <rect x="90" y="55" width="80" height="20" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1" rx="2" />
    <text x="130" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v</text>

    <!-- Idle bubble -->
    <rect x="175" y="55" width="225" height="20" fill="#fef2f2" stroke="#fca5a5" stroke-dasharray="3,3" stroke-width="1" rx="2" />
    <text x="287" y="69" class="code-text" text-anchor="middle" fill="#dc2626" font-size="9.5">IDLE WAITING BUBBLE (MFU ~18%)</text>

    <rect x="405" y="55" width="85" height="20" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1" rx="2" />
    <text x="447" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v+1</text>

    <!-- Rollout Workers Track -->
    <text x="15" y="105" class="code-text" font-weight="bold">Worker 1:</text>
    <rect x="90" y="92" width="60" height="18" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" rx="2" />
    <text x="120" y="105" class="code-text" text-anchor="middle" font-size="9">120 tokens</text>
    <rect x="155" y="92" width="245" height="18" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="2,2" rx="2" />
    <text x="277" y="105" class="code-text" text-anchor="middle" fill="#94a3b8" font-size="9">idle wait at barrier</text>

    <text x="15" y="133" class="code-text" font-weight="bold">Worker 2:</text>
    <rect x="90" y="120" width="160" height="18" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" rx="2" />
    <text x="170" y="133" class="code-text" text-anchor="middle" font-size="9">850 tokens (tool use)</text>
    <rect x="255" y="120" width="145" height="18" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="2,2" rx="2" />
    <text x="327" y="133" class="code-text" text-anchor="middle" fill="#94a3b8" font-size="9">idle wait</text>

    <text x="15" y="161" class="code-text" font-weight="bold">Worker G:</text>
    <rect x="90" y="148" width="305" height="18" fill="#fef3c7" stroke="#f59e0b" stroke-width="1" rx="2" />
    <text x="242" y="161" class="code-text" text-anchor="middle" font-size="9">4,096 tokens (Max Horizon Straggler)</text>

    <!-- Barrier line -->
    <line x1="400" y1="50" x2="400" y2="175" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,3" />
    <text x="400" y="190" class="code-text" text-anchor="middle" fill="#dc2626" font-weight="bold" font-size="9">Global Sync Barrier</text>

    <!-- Bottom summary line -->
    <rect x="15" y="200" width="485" height="20" fill="#f1f5f9" stroke="#e2e8f0" rx="2" />
    <text x="25" y="214" class="body-text" font-size="10">Properties: Strictly on-policy (Δv = 0), but step time T_step = max(T_exec) + T_train</text>
  </g>

  <!-- Right Side: Asynchronous Streaming (Decoupled with Buffer) -->
  <g transform="translate(585, 65)">
    <rect x="0" y="0" width="515" height="230" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="15" y="24" class="box-title">Asynchronous Streaming Pipeline (Freshness-Bound)</text>
    <text x="15" y="40" class="box-sub">Workers stream rollouts to circular replay queue; trainer computes at saturation</text>

    <!-- Continuous Trainer Steps -->
    <text x="15" y="68" class="code-text" font-weight="bold">Trainer:</text>
    <rect x="90" y="55" width="75" height="20" fill="#dcfce7" stroke="#86efac" stroke-width="1" rx="2" />
    <text x="127" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v</text>

    <rect x="170" y="55" width="75" height="20" fill="#dcfce7" stroke="#86efac" stroke-width="1" rx="2" />
    <text x="207" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v+1</text>

    <rect x="250" y="55" width="75" height="20" fill="#dcfce7" stroke="#86efac" stroke-width="1" rx="2" />
    <text x="287" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v+2</text>

    <rect x="330" y="55" width="75" height="20" fill="#dcfce7" stroke="#86efac" stroke-width="1" rx="2" />
    <text x="367" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v+3</text>

    <rect x="410" y="55" width="75" height="20" fill="#dcfce7" stroke="#86efac" stroke-width="1" rx="2" />
    <text x="447" y="69" class="code-text" text-anchor="middle" font-size="9.5">Step v+4</text>

    <!-- Rollout Streams with Lag Labels -->
    <text x="15" y="105" class="code-text" font-weight="bold">Worker 1:</text>
    <rect x="90" y="92" width="75" height="18" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" rx="2" />
    <text x="127" y="105" class="code-text" text-anchor="middle" font-size="9">Gen (θ_v)</text>
    <path d="M 165 98 L 205 78" class="arrow-emerald" />
    <rect x="205" y="85" width="92" height="15" fill="#ecfdf5" stroke="#a7f3d0" rx="2" />
    <text x="251" y="96" class="code-text" text-anchor="middle" fill="#065f46" font-size="8">Ingest: Δv = 1 (OK)</text>

    <text x="15" y="133" class="code-text" font-weight="bold">Worker 2:</text>
    <rect x="90" y="120" width="150" height="18" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" rx="2" />
    <text x="165" y="133" class="code-text" text-anchor="middle" font-size="9">Gen (θ_v, 850t)</text>
    <path d="M 240 125 L 285 78" class="arrow-emerald" />
    <rect x="285" y="112" width="92" height="15" fill="#ecfdf5" stroke="#a7f3d0" rx="2" />
    <text x="331" y="123" class="code-text" text-anchor="middle" fill="#065f46" font-size="8">Ingest: Δv = 2 (OK)</text>

    <text x="15" y="161" class="code-text" font-weight="bold">Worker G:</text>
    <rect x="90" y="148" width="315" height="18" fill="#fef2f2" stroke="#fca5a5" stroke-width="1" rx="2" />
    <text x="247" y="161" class="code-text" text-anchor="middle" font-size="9">Gen (θ_v, 4096t)</text>
    <path d="M 405 155 L 438 120 L 438 80" class="arrow-rose" />
    <rect x="365" y="128" width="135" height="15" fill="#fff1f2" stroke="#fecdd3" rx="2" />
    <text x="432" y="139" class="code-text" text-anchor="middle" fill="#9f1239" font-size="8">Δv = 4 > Δv_max (REJECTED)</text>

    <!-- Bottom summary line -->
    <rect x="15" y="200" width="485" height="20" fill="#f1f5f9" stroke="#e2e8f0" rx="2" />
    <text x="25" y="214" class="body-text" font-size="10">Properties: Continuous GPU saturation (MFU ~75%), off-policy bounded by Δv_max</text>
  </g>

  <!-- BOTTOM PANEL: (b) ARCHITECTURE: FRESHNESS GATE & STREAMING INGESTION -->
  <rect x="30" y="325" width="1090" height="365" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" rx="4" />
  <text x="45" y="347" class="panel-label">(b) Host-Managed Replay Architecture &amp; Admission Gating Protocol</text>

  <!-- Block 1: Rollout Fleet (Inference Cluster) -->
  <g transform="translate(45, 360)">
    <rect x="0" y="0" width="220" height="295" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="3" />
    <rect x="0" y="0" width="220" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">1. ROLLOUT FLEET</text>

    <text x="15" y="48" class="body-text" font-weight="600">Disaggregated Inference Ranks</text>
    <text x="15" y="65" class="code-text" font-size="10">vLLM / SGLang Engine</text>

    <!-- Sub-boxes for Workers -->
    <rect x="12" y="80" width="196" height="55" fill="#f8fafc" stroke="#cbd5e1" rx="2" />
    <text x="20" y="98" class="code-text" font-weight="bold">Worker Pod 1..M</text>
    <text x="20" y="114" class="code-text" font-size="9.5">Local Checkpoint: θ_rollout</text>
    <text x="20" y="128" class="code-text" font-size="9.5">Radix KV Prefix Cache</text>

    <rect x="12" y="145" width="196" height="65" fill="#f8fafc" stroke="#cbd5e1" rx="2" />
    <text x="20" y="163" class="code-text" font-weight="bold">MicroVM Enclave</text>
    <text x="20" y="179" class="code-text" font-size="9.5">Hermetic Tool Sandbox (A=0)</text>
    <text x="20" y="194" class="code-text" font-size="9.5">Deterministic Test Oracles</text>
    <text x="20" y="205" class="code-text" font-size="9" fill="#059669">R_outcome ∈ {0, 1}</text>

    <rect x="12" y="220" width="196" height="60" fill="#f8fafc" stroke="#cbd5e1" rx="2" />
    <text x="20" y="238" class="code-text" font-weight="bold">Trajectory Envelope</text>
    <text x="20" y="254" class="code-text" font-size="9">UUID, v_rollout, logprobs</text>
    <text x="20" y="268" class="code-text" font-size="9">Actions, States, Rewards</text>
  </g>

  <!-- RDMA Stream Arrow -->
  <path d="M 265 500 L 305 500" class="arrow-blue" />
  <text x="285" y="485" class="code-text" text-anchor="middle" font-size="9" fill="#2563eb">RDMA</text>
  <text x="285" y="495" class="code-text" text-anchor="middle" font-size="8" fill="#64748b">Transit</text>

  <!-- Block 2: Circular Trajectory Replay Buffer -->
  <g transform="translate(310, 360)">
    <rect x="0" y="0" width="235" height="295" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="3" />
    <rect x="0" y="0" width="235" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">2. CIRCULAR REPLAY BUFFER</text>

    <text x="15" y="48" class="body-text" font-weight="600">Pinned Host DRAM Ring</text>
    <text x="15" y="65" class="code-text" font-size="10">Asynchronous FIFO Stage</text>

    <!-- Ring Buffer Slots visualization -->
    <g transform="translate(15, 80)">
      <rect x="0" y="0" width="205" height="25" fill="#ecfdf5" stroke="#6ee7b7" rx="2" />
      <text x="8" y="16" class="code-text" font-size="9">Slot 0: τ_101 [v_rollout = 48]</text>

      <rect x="0" y="32" width="205" height="25" fill="#ecfdf5" stroke="#6ee7b7" rx="2" />
      <text x="8" y="48" class="code-text" font-size="9">Slot 1: τ_102 [v_rollout = 49]</text>

      <rect x="0" y="64" width="205" height="25" fill="#ecfdf5" stroke="#6ee7b7" rx="2" />
      <text x="8" y="80" class="code-text" font-size="9">Slot 2: τ_103 [v_rollout = 49]</text>

      <rect x="0" y="96" width="205" height="25" fill="#fff1f2" stroke="#fca5a5" rx="2" />
      <text x="8" y="112" class="code-text" font-size="8.5" fill="#9f1239">Slot 3: τ_104 [v_r=46] (STALE)</text>

      <rect x="0" y="128" width="205" height="25" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="2,2" rx="2" />
      <text x="8" y="144" class="code-text" font-size="9" fill="#94a3b8">Slot 4..K: In-flight / Writing</text>
    </g>

    <rect x="15" y="245" width="205" height="38" fill="#f1f5f9" stroke="#e2e8f0" rx="2" />
    <text x="22" y="260" class="code-text" font-size="9">Capacity: K_buffer = 4,096</text>
    <text x="22" y="274" class="code-text" font-size="8.5" fill="#64748b">Zero-copy host memory pinned</text>
  </g>

  <!-- Queue to Gate Arrow -->
  <path d="M 545 500 L 585 500" class="arrow" />

  <!-- Block 3: Trajectory Admission Gate -->
  <g transform="translate(590, 360)">
    <rect x="0" y="0" width="240" height="295" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="3" />
    <rect x="0" y="0" width="240" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">3. FRESHNESS ADMISSION GATE</text>

    <text x="15" y="48" class="body-text" font-weight="600">Staleness Computation:</text>
    <rect x="15" y="55" width="210" height="26" fill="#eff6ff" stroke="#bfdbfe" rx="2" />
    <text x="120" y="72" class="code-text" text-anchor="middle" font-weight="bold">Δv = v_trainer - v_rollout</text>

    <text x="15" y="102" class="body-text" font-weight="600">Admission Policy Invariant:</text>
    <rect x="15" y="108" width="210" height="42" fill="#f8fafc" stroke="#cbd5e1" rx="2" />
    <text x="22" y="125" class="code-text" font-size="9.5">Admit(τ) = TRUE  if Δv ≤ Δv_max</text>
    <text x="22" y="140" class="code-text" font-size="9.5">Admit(τ) = FALSE if Δv > Δv_max</text>

    <!-- Branch Outcomes -->
    <!-- Admitted path -->
    <path d="M 120 150 L 120 175" class="arrow-emerald" />
    <rect x="15" y="175" width="210" height="45" fill="#ecfdf5" stroke="#6ee7b7" rx="2" />
    <text x="22" y="192" class="code-text" font-weight="bold" fill="#065f46">PASS: Δv ≤ 2 (Admitted)</text>
    <text x="22" y="208" class="code-text" font-size="9">Forward to Truncated IS</text>

    <!-- Rejected path -->
    <path d="M 120 220 L 120 238" class="arrow-rose" />
    <rect x="15" y="238" width="210" height="45" fill="#fff1f2" stroke="#fca5a5" rx="2" />
    <text x="22" y="254" class="code-text" font-weight="bold" fill="#9f1239">FAIL: Δv > 2 (Quarantined)</text>
    <text x="22" y="270" class="code-text" font-size="8.5" fill="#9f1239">Recycle Buffer; Drop Trajectory</text>
  </g>

  <!-- Admitted Arrow to Training Engine -->
  <path d="M 830 500 L 870 500" class="arrow-emerald" stroke-width="2" />
  <text x="850" y="490" class="code-text" text-anchor="middle" font-size="9" fill="#059669">Admit</text>

  <!-- Block 4: Training Fleet & Weight Broadcast -->
  <g transform="translate(875, 360)">
    <rect x="0" y="0" width="230" height="295" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" rx="3" />
    <rect x="0" y="0" width="230" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">4. TRAINING ENGINE</text>

    <text x="15" y="48" class="body-text" font-weight="600">Truncated Importance Ratio:</text>
    <rect x="10" y="55" width="210" height="38" fill="#f8fafc" stroke="#cbd5e1" rx="2" />
    <text x="15" y="70" class="code-text" font-size="8">ρ_t = min(ρ̄, π_train / π_rollout)</text>
    <text x="15" y="84" class="code-text" font-size="8" fill="#64748b">Bounds variance on policy drift</text>

    <text x="15" y="110" class="body-text" font-weight="600">GRPO Gradient Step:</text>
    <rect x="10" y="116" width="210" height="52" fill="#eff6ff" stroke="#bfdbfe" rx="2" />
    <text x="15" y="132" class="code-text" font-size="8">ĝ = E[ ∇_θ log π_θ · Â_i ]</text>
    <text x="15" y="148" class="code-text" font-size="8">Active counter: v_trainer++</text>
    <text x="15" y="161" class="code-text" font-size="8" fill="#2563eb">Backward GEMM Saturation</text>

    <text x="15" y="186" class="body-text" font-weight="600">Checkpoint Release Gate:</text>
    <rect x="10" y="192" width="210" height="42" fill="#f1f5f9" stroke="#cbd5e1" rx="2" />
    <text x="15" y="207" class="code-text" font-size="8">Eval Sub-cluster Canary Test</text>
    <text x="15" y="222" class="code-text" font-size="8" fill="#059669">Passed: Promote θ_(v+1) to rollout</text>

    <!-- Async Broadcast Back to Workers -->
    <rect x="10" y="244" width="210" height="38" fill="#f0fdf4" stroke="#86efac" rx="2" />
    <text x="15" y="259" class="code-text" font-weight="bold" fill="#15803d">Asynchronous Broadcast</text>
    <text x="15" y="272" class="code-text" font-size="8" fill="#15803d">Pushes θ_(v+1) to Rollout Fleet</text>
  </g>

  <!-- Global Feedback Loop from Checkpoint Broadcast to Rollout Fleet -->
  <path d="M 978 680 L 978 695 L 155 695 L 155 655" fill="none" stroke="#15803d" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrow-emerald-marker)" />
  <text x="560" y="690" class="code-text" text-anchor="middle" font-size="9" fill="#15803d">Asynchronous Checkpoint Dissemination Bus (θ_v → θ_v+1 update)</text>

</svg>'''

target_svg = "/Users/VJ/GitHub/MLSysBook-vol3/books/vol3/14_rlvr/images/svg/asynchronous-freshness-pipeline.svg"

# Remove symlink if exists
if os.path.islink(target_svg) or os.path.exists(target_svg):
    os.remove(target_svg)

with open(target_svg, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"Written {len(svg_content)} bytes to {target_svg}")
