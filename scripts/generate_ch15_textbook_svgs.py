#!/usr/bin/env python3
"""
Generate 3 publication-grade textbook SVGs for Chapter 15 (Multi-Agent Systems):
1. fig-vol3-occ-worktrees.svg
2. fig-vol3-correlated-ensemble-failure.svg
3. capability-attenuation-tree.svg

Style: Classic Hennessy & Patterson / Saltzer & Kaashoek computer systems textbook.
Clean rectangular functional blocks, clear dataflow, formal interfaces, minimal text in graphics.
"""

import os

# ==============================================================================
# 1. OPTIMISTIC CONCURRENCY CONTROL WITH ISOLATED WORKTREES
# ==============================================================================
occ_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 700" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #0f172a; }
      .box-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 10px; fill: #64748b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 10px; fill: #334155; }
      .code-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9.5px; fill: #0f172a; }
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

  <!-- Outer Frame -->
  <rect x="0" y="0" width="1150" height="700" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="670" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Header Block -->
  <text x="35" y="45" class="title">OPTIMISTIC CONCURRENCY CONTROL (OCC) WITH ISOLATED GIT WORKTREES</text>
  <text x="35" y="63" class="subtitle">Four-Phase Concurrency Protocol: Read Snapshotting, Sandboxed Execution, Validation, and Atomic Reconciliation</text>

  <!-- 4 Formal Phase Columns -->
  <!-- PHASE 1: READ SNAPSHOT -->
  <g transform="translate(35, 85)">
    <rect x="0" y="0" width="245" height="420" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="0" y="0" width="245" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">PHASE 1: READ SNAPSHOT</text>

    <!-- Primary Repository Box -->
    <rect x="12" y="45" width="220" height="85" fill="#ffffff" stroke="#94a3b8" rx="2" />
    <text x="22" y="65" class="box-title" font-size="11">Shared Primary Repository</text>
    <text x="22" y="80" class="code-text">Head Commit: C_base</text>
    <text x="22" y="95" class="code-text" fill="#64748b">Ref: refs/heads/main</text>
    <text x="22" y="115" class="body-text">Object Store: .git/objects (Immutable)</text>

    <!-- Fast Worktree Fork -->
    <path d="M 122 130 L 122 170" class="arrow-blue" />
    <text x="132" y="155" class="code-text" font-size="9" fill="#2563eb">git worktree add</text>

    <!-- Ephemeral Worktrees -->
    <rect x="12" y="175" width="220" height="100" fill="#eff6ff" stroke="#bfdbfe" rx="2" />
    <text x="22" y="195" class="box-title" font-size="11" fill="#1e40af">Private Worktree B_i</text>
    <text x="22" y="212" class="code-text">Branch: task-{uuid}</text>
    <text x="22" y="228" class="code-text">Hardlinked object store</text>
    <text x="22" y="244" class="code-text">Private index &amp; tree</text>
    <text x="22" y="260" class="body-text" fill="#1e40af">Creation: ~14 ms (Zero data copy)</text>

    <!-- Protocol Properties -->
    <rect x="12" y="295" width="220" height="110" fill="#ffffff" stroke="#e2e8f0" rx="2" />
    <text x="22" y="315" class="box-title" font-size="10.5">Read Phase Invariant:</text>
    <text x="22" y="333" class="code-text">C_base = SHA256(RepoHead)</text>
    <text x="22" y="352" class="body-text">• No global read locks acquired</text>
    <text x="22" y="369" class="body-text">• Shared symbol table populated</text>
    <text x="22" y="386" class="body-text">• Read set R_i initialized</text>
  </g>

  <!-- Flow Arrow Phase 1 -> 2 -->
  <path d="M 280 295 L 310 295" class="arrow" />

  <!-- PHASE 2: SANDBOXED EXECUTE -->
  <g transform="translate(310, 85)">
    <rect x="0" y="0" width="250" height="420" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="0" y="0" width="250" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">PHASE 2: EXECUTE (ISOLATED)</text>

    <!-- Worker Agent Execution Enclave -->
    <rect x="15" y="45" width="220" height="135" fill="#ffffff" stroke="#94a3b8" rx="2" />
    <text x="25" y="65" class="box-title" font-size="11">Worker Agent W_i Enclave</text>
    <text x="25" y="82" class="body-text">Untrusted autoregressive generation:</text>
    <text x="25" y="98" class="code-text">• Local file edits (patch/write)</text>
    <text x="25" y="112" class="code-text">• Hermetic test runs (pytest/cargo)</text>
    <text x="25" y="126" class="code-text">• Epistemic diagnostics (grep/ast)</text>
    <text x="25" y="145" class="code-text" fill="#059669">Exit status: PASS (R_outcome = 1)</text>
    <text x="25" y="165" class="code-text">Terminal commit: C_i = SHA256(...)</text>

    <!-- Mutation Set Capture -->
    <path d="M 125 180 L 125 215" class="arrow" />
    <rect x="15" y="215" width="220" height="75" fill="#ecfdf5" stroke="#6ee7b7" rx="2" />
    <text x="25" y="235" class="box-title" font-size="11" fill="#065f46">TaskReceipt Envelope</text>
    <text x="25" y="252" class="code-text">Read Set:  R_i = {f_1, f_2, ...}</text>
    <text x="25" y="268" class="code-text">Write Set: W_i = {f_auth.py}</text>

    <!-- Execution Invariants -->
    <rect x="15" y="305" width="220" height="100" fill="#ffffff" stroke="#e2e8f0" rx="2" />
    <text x="25" y="325" class="box-title" font-size="10.5">Execution Invariant:</text>
    <text x="25" y="343" class="body-text">• Zero dirty state leakage to main</text>
    <text x="25" y="360" class="body-text">• Concurrent workers execute parallel</text>
    <text x="25" y="377" class="body-text">• CPU/IO quota strictly bounded</text>
  </g>

  <!-- Flow Arrow Phase 2 -> 3 -->
  <path d="M 560 295 L 590 295" class="arrow" />

  <!-- PHASE 3: VALIDATION -->
  <g transform="translate(590, 85)">
    <rect x="0" y="0" width="255" height="420" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="0" y="0" width="255" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">PHASE 3: VALIDATION</text>

    <!-- Head State Comparator -->
    <rect x="15" y="45" width="225" height="95" fill="#ffffff" stroke="#94a3b8" rx="2" />
    <text x="25" y="65" class="box-title" font-size="11">Branch Head Interrogation</text>
    <text x="25" y="82" class="code-text">C_head = repo.rev_parse("HEAD")</text>
    <text x="25" y="100" class="code-text">Delta_m = C_head \\ C_base</text>
    <text x="25" y="118" class="body-text">Detect intervening merges</text>

    <!-- Disjoint Set Check -->
    <path d="M 127 140 L 127 175" class="arrow" />
    <rect x="15" y="175" width="225" height="90" fill="#eff6ff" stroke="#bfdbfe" rx="2" />
    <text x="25" y="195" class="box-title" font-size="11" fill="#1e40af">OCC Disjointness Invariant</text>
    <rect x="25" y="205" width="205" height="26" fill="#ffffff" stroke="#93c5fd" rx="2" />
    <text x="127" y="222" class="code-text" text-anchor="middle" font-weight="bold">W_i ∩ W_merged == ∅</text>
    <text x="25" y="248" class="body-text">Intervening writes from C_k ∈ Delta_m</text>

    <!-- Validation Branching -->
    <rect x="15" y="280" width="225" height="125" fill="#ffffff" stroke="#e2e8f0" rx="2" />
    <text x="25" y="300" class="box-title" font-size="10.5">Validation Outcomes:</text>
    <text x="25" y="320" class="code-text" fill="#059669">1. C_head == C_base (No drift)</text>
    <text x="25" y="338" class="code-text" fill="#059669">2. Disjoint: W_i ∩ W_m == ∅</text>
    <text x="25" y="354" class="body-text" fill="#059669">   (Clean Disjoint Writes)</text>
    <text x="25" y="374" class="code-text" fill="#dc2626">3. Overlap: W_i ∩ W_m != ∅</text>
    <text x="25" y="390" class="body-text" fill="#dc2626">   (Write Collision Hazard)</text>
  </g>

  <!-- Flow Arrow Phase 3 -> 4 -->
  <path d="M 845 295 L 875 295" class="arrow" />

  <!-- PHASE 4: COMMIT & RECONCILIATION -->
  <g transform="translate(875, 85)">
    <rect x="0" y="0" width="240" height="420" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="0" y="0" width="240" height="28" fill="#f1f5f9" rx="3" />
    <text x="15" y="19" class="box-title">PHASE 4: COMMIT / RECONCILE</text>

    <!-- Outcome 1: Fast-Forward -->
    <rect x="12" y="45" width="216" height="75" fill="#ecfdf5" stroke="#6ee7b7" rx="2" />
    <text x="20" y="65" class="box-title" font-size="10.5" fill="#065f46">A. Fast-Forward Commit (CAS)</text>
    <text x="20" y="80" class="code-text" font-size="8.5">git update-ref refs/heads/main C_i</text>
    <text x="20" y="95" class="code-text" font-size="8.5">old_val = C_base (Atomic swap)</text>
    <text x="20" y="110" class="code-text" font-size="8.5" fill="#059669">Status: COMMITTED_FAST_FORWARD</text>

    <!-- Outcome 2: Clean 3-Way Merge -->
    <rect x="12" y="130" width="216" height="85" fill="#f0fdf4" stroke="#86efac" rx="2" />
    <text x="20" y="150" class="box-title" font-size="10.5" fill="#15803d">B. Disjoint 3-Way Merge</text>
    <text x="20" y="165" class="code-text" font-size="8.5">C_new = Merge(C_base, C_head, C_i)</text>
    <text x="20" y="180" class="code-text" font-size="8.5">Run regression suite in enclave</text>
    <text x="20" y="195" class="code-text" font-size="8.5">git update-ref old_val = C_head</text>
    <text x="20" y="208" class="code-text" font-size="8.5" fill="#15803d">Status: COMMITTED_MERGE</text>

    <!-- Outcome 3: Conflict & Reconciliation -->
    <rect x="12" y="225" width="216" height="95" fill="#fff1f2" stroke="#fca5a5" rx="2" />
    <text x="20" y="245" class="box-title" font-size="10.5" fill="#9f1239">C. Conflicting Merge Resolution</text>
    <text x="20" y="260" class="code-text" font-size="8.5">Conflict markers emitted (&lt;&lt;&lt;, &gt;&gt;&gt;)</text>
    <text x="20" y="275" class="code-text" font-size="8.5">Reconciliation Subtask: Agent A_r</text>
    <text x="20" y="290" class="body-text" font-size="8.5">Re-synthesizes AST; runs pytest</text>
    <text x="20" y="305" class="code-text" font-size="8.5" fill="#dc2626">Fallback: Abort transaction (Saga)</text>

    <!-- Cleanup / Pruning -->
    <rect x="12" y="330" width="216" height="75" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="20" y="348" class="box-title" font-size="10">Worktree Garbage Collection:</text>
    <text x="20" y="365" class="code-text" font-size="8.5">git worktree remove --force</text>
    <text x="20" y="380" class="code-text" font-size="8.5">Delete branch task-{uuid}</text>
    <text x="20" y="395" class="body-text" font-size="8.5" fill="#64748b">Prunes directory in &lt;5 ms</text>
  </g>

  <!-- BOTTOM COMPARISON SUMMARY TABLE -->
  <g transform="translate(35, 520)">
    <rect x="0" y="0" width="1080" height="150" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <rect x="0" y="0" width="1080" height="26" fill="#f1f5f9" rx="3" />
    <text x="15" y="18" class="box-title">SYSTEMS COMPARISON: NAIVE DIRECT MUTATION VS. OPTIMISTIC WORKTREE PROTOCOL</text>

    <!-- Table Grid -->
    <line x1="0" y1="55" x2="1080" y2="55" stroke="#e2e8f0" stroke-width="1" />
    <line x1="0" y1="85" x2="1080" y2="85" stroke="#e2e8f0" stroke-width="1" />
    <line x1="0" y1="115" x2="1080" y2="115" stroke="#e2e8f0" stroke-width="1" />
    
    <line x1="220" y1="0" x2="220" y2="150" stroke="#e2e8f0" stroke-width="1" />
    <line x1="650" y1="0" x2="650" y2="150" stroke="#e2e8f0" stroke-width="1" />

    <!-- Headers -->
    <text x="15" y="42" class="box-title" font-size="10">DIMENSION</text>
    <text x="235" y="42" class="box-title" font-size="10" fill="#dc2626">NAIVE RECURSIVE COPY / SHARED WORKSPACE</text>
    <text x="665" y="42" class="box-title" font-size="10" fill="#059669">OCC GIT WORKTREE PROTOCOL</text>

    <!-- Row 1 -->
    <text x="15" y="73" class="code-text">Provisioning Latency</text>
    <text x="235" y="73" class="body-text">480 s under 32 workers (inode lock thrashing, 57.6 GB I/O)</text>
    <text x="665" y="73" class="body-text" font-weight="bold">14 ms per worker (shared .git/objects, 134 MB total I/O; 99.8% reduction)</text>

    <!-- Row 2 -->
    <text x="15" y="103" class="code-text">Concurrency Hazard</text>
    <text x="235" y="103" class="body-text">Lost updates, clobbered dirty files, unrepeatable test runs</text>
    <text x="665" y="103" class="body-text" font-weight="bold">Zero ambient interference; formal OCC validation via read/write sets</text>

    <!-- Row 3 -->
    <text x="15" y="133" class="code-text">Conflict Resolution</text>
    <text x="235" y="133" class="body-text">Fatal crash or uncoordinated file overwrites by last-writing agent</text>
    <text x="665" y="133" class="body-text" font-weight="bold">Atomic CAS fast-forward, clean 3-way merge, or AST agentic reconciliation</text>
  </g>

</svg>'''

# ==============================================================================
# 2. CORRELATED ENSEMBLE FAILURES VS INDEPENDENT ERROR SCALING
# ==============================================================================
ensemble_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 680" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .panel-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 13px; fill: #0f172a; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 11.5px; fill: #1e293b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 10.5px; fill: #334155; }
      .code-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 10px; fill: #0f172a; }
      .axis-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 600; font-size: 11px; fill: #475569; }
      .grid-line { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 3,3; }
      .axis-line { stroke: #475569; stroke-width: 1.5; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
  </defs>

  <!-- Outer Frame -->
  <rect x="0" y="0" width="1150" height="680" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="650" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Title -->
  <text x="35" y="45" class="title">CORRELATED ENSEMBLE FAILURES VS. INDEPENDENT CONDORCET ERROR SCALING</text>
  <text x="35" y="63" class="subtitle">Theoretical Exponential Scaling under Independent Trials vs. Irreducible Error Floors under Latent Pre-training Correlation (ρ = 0.40)</text>

  <!-- LEFT PANEL: MATHEMATICAL CHART -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="620" height="565" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="20" y="25" class="panel-label">Ensemble Majority Failure Probability P(Majority Fail) vs. Fleet Size M</text>

    <!-- Axes: Y-axis from 0% to 30% (Origin at x=75, y=480, Top at y=70) -->
    <line x1="75" y1="480" x2="550" y2="480" class="axis-line" marker-end="url(#arrow)" />
    <line x1="75" y1="480" x2="75" y2="70" class="axis-line" marker-end="url(#arrow)" />

    <text x="490" y="505" class="axis-label">Voter Pool Size (M)</text>
    <text x="40" y="60" class="axis-label" text-anchor="middle">P(Fail)</text>

    <!-- Y-axis Grid and Ticks -->
    <text x="65" y="484" class="code-text" text-anchor="end">0%</text>
    
    <line x1="75" y1="415" x2="530" y2="415" class="grid-line" />
    <text x="65" y="419" class="code-text" text-anchor="end">5%</text>

    <line x1="75" y1="350" x2="530" y2="350" class="grid-line" />
    <text x="65" y="354" class="code-text" text-anchor="end">10%</text>

    <line x1="75" y1="285" x2="530" y2="285" class="grid-line" />
    <text x="65" y="289" class="code-text" text-anchor="end">15%</text>

    <!-- 20% (Baseline Individual Error p=0.20) -->
    <line x1="75" y1="220" x2="530" y2="220" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,2" />
    <text x="65" y="224" class="code-text" text-anchor="end" font-weight="bold">20%</text>
    <text x="320" y="212" class="code-text" fill="#64748b" font-size="8.5">Baseline Single-Agent Error (p = 0.20)</text>

    <line x1="75" y1="155" x2="530" y2="155" class="grid-line" />
    <text x="65" y="159" class="code-text" text-anchor="end">25%</text>

    <!-- X-axis Ticks for M = 1, 3, 5, 7, 9, 11, 13, 15 -->
    <!-- M=1: x=95, M=3: x=155, M=5: x=215, M=7: x=275, M=9: x=335, M=11: x=395, M=13: x=455, M=15: x=515 -->
    <text x="95" y="498" class="code-text" text-anchor="middle">M=1</text>
    <text x="155" y="498" class="code-text" text-anchor="middle">M=3</text>
    <text x="215" y="498" class="code-text" text-anchor="middle">M=5</text>
    <text x="275" y="498" class="code-text" text-anchor="middle">M=7</text>
    <text x="335" y="498" class="code-text" text-anchor="middle">M=9</text>
    <text x="395" y="498" class="code-text" text-anchor="middle">M=11</text>
    <text x="455" y="498" class="code-text" text-anchor="middle">M=13</text>
    <text x="515" y="498" class="code-text" text-anchor="middle">M=15</text>

    <!-- CURVE 1: INDEPENDENT CONDORCET SCALING (rho = 0.0) -->
    <!-- M=1: 20% (y=220) | M=3: 10.4% (y=345) | M=5: 5.79% (y=405) | M=7: 3.3% (y=437) | M=9: 1.9% (y=455) | M=11: 1.1% (y=466) | M=13: 0.6% (y=472) | M=15: 0.3% (y=476) -->
    <!-- Strictly monotonic polyline / smooth path -->
    <path d="M 95 220 C 125 300, 140 335, 155 345 C 185 385, 200 400, 215 405 C 245 425, 260 433, 275 437 C 305 448, 320 452, 335 455 C 365 462, 380 464, 395 466 C 425 470, 440 471, 455 472 C 485 474, 500 475, 515 476" 
          fill="none" stroke="#2563eb" stroke-width="2.5" />
    
    <circle cx="95" cy="220" r="3.5" fill="#2563eb" />
    <circle cx="155" cy="345" r="3.5" fill="#2563eb" />
    <circle cx="215" cy="405" r="3.5" fill="#2563eb" />
    <circle cx="275" cy="437" r="3.5" fill="#2563eb" />
    <circle cx="335" cy="455" r="3.5" fill="#2563eb" />
    <circle cx="395" cy="466" r="3.5" fill="#2563eb" />
    <circle cx="455" cy="472" r="3.5" fill="#2563eb" />
    <circle cx="515" cy="476" r="3.5" fill="#2563eb" />

    <!-- CURVE 2: CORRELATED HOMOGENEOUS FLEET (rho = 0.40) -->
    <!-- M=1: 20% (y=220) | M=3: 16.2% (y=269) | M=5: 14.8% (y=288) | M=7: 14.2% (y=295) | M=9: 13.9% (y=299) | M=11: 13.7% (y=302) | M=13: 13.6% (y=303) | M=15: 13.5% (y=304) -->
    <path d="M 95 220 C 125 250, 140 264, 155 269 C 185 281, 200 285, 215 288 C 245 292, 260 294, 275 295 C 305 297, 320 298, 335 299 C 365 300, 380 301, 395 302 C 425 302.5, 440 303, 455 303 C 485 303.5, 500 304, 515 304" 
          fill="none" stroke="#dc2626" stroke-width="2.5" />
    
    <circle cx="95" cy="220" r="3.5" fill="#dc2626" />
    <circle cx="155" cy="269" r="3.5" fill="#dc2626" />
    <circle cx="215" cy="288" r="3.5" fill="#dc2626" />
    <circle cx="275" cy="295" r="3.5" fill="#dc2626" />
    <circle cx="335" cy="299" r="3.5" fill="#dc2626" />
    <circle cx="395" cy="302" r="3.5" fill="#dc2626" />
    <circle cx="455" cy="303" r="3.5" fill="#dc2626" />
    <circle cx="515" cy="304" r="3.5" fill="#dc2626" />

    <!-- Irreducible Floor Line -->
    <line x1="75" y1="305" x2="530" y2="305" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" />
    <rect x="220" y="312" width="280" height="22" fill="#fff1f2" stroke="#fecdd3" rx="2" />
    <text x="360" y="327" class="code-text" text-anchor="middle" fill="#9f1239" font-weight="bold">Irreducible Correlated Error Floor: ~13.5%</text>

    <!-- Curve Annotations -->
    <rect x="235" y="415" width="190" height="28" fill="#eff6ff" stroke="#bfdbfe" rx="2" />
    <text x="245" y="428" class="code-text" fill="#1e40af" font-weight="bold">Independent Condorcet (ρ = 0.0)</text>
    <text x="245" y="439" class="code-text" fill="#2563eb" font-size="8">Exponential decay P(fail) → 0</text>

    <rect x="175" y="240" width="180" height="28" fill="#fff1f2" stroke="#fecdd3" rx="2" />
    <text x="185" y="253" class="code-text" fill="#9f1239" font-weight="bold">Correlated LLM Fleet (ρ = 0.40)</text>
    <text x="185" y="264" class="code-text" fill="#dc2626" font-size="8">Plateaus rapidly at M ≥ 3..5</text>

    <!-- Bottom Math Callout in Chart -->
    <rect x="15" y="520" width="590" height="35" fill="#ffffff" stroke="#e2e8f0" rx="2" />
    <text x="25" y="534" class="body-text">Mathematical Invariant: Under positive pairwise correlation ρ &gt; 0, lim_{M→∞} P(Majority Fail) = P(Z &lt; 0) &gt; 0.</text>
    <text x="25" y="548" class="body-text" font-size="9" fill="#64748b">Homogeneous scaling burns inference tokens without reducing systematic alignment and training-set blindspots.</text>
  </g>

  <!-- RIGHT PANEL: SYSTEMS ANALYSIS & STRUCTURAL CONSEQUENCES -->
  <g transform="translate(675, 80)">
    <rect x="0" y="0" width="440" height="565" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="20" y="25" class="panel-label">Systems Implications &amp; Worked Example (M = 5)</text>

    <!-- Block 1: Worked Example Card -->
    <rect x="15" y="45" width="410" height="150" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="25" y="65" class="box-title" font-size="11">Numerical Breakdown: M = 5 Voters (p = 0.20)</text>
    
    <text x="25" y="85" class="code-text" font-weight="bold" fill="#2563eb">1. Independent Errors (ρ = 0.0, Binomial):</text>
    <text x="35" y="100" class="code-text" font-size="9">P(k ≥ 3) = 10(0.2)³(0.8)² + 5(0.2)⁴(0.8) + (0.2)⁵</text>
    <text x="35" y="115" class="code-text" font-size="9.5" font-weight="bold">P(majority fail) = 5.79% (3.45× error reduction)</text>

    <text x="25" y="135" class="code-text" font-weight="bold" fill="#dc2626">2. Correlated Errors (ρ = 0.40, Beta-Binomial):</text>
    <text x="35" y="150" class="code-text" font-size="9">P(k ≥ 3) evaluated with dispersion (α+β) = 1.5</text>
    <text x="35" y="165" class="code-text" font-size="9.5" font-weight="bold" fill="#dc2626">P(majority fail) = 14.80% (Only 1.35× reduction!)</text>
    <text x="35" y="180" class="code-text" font-size="8.5" fill="#9f1239">Unanimous 5-way failure rate jumps 180× (0.03% → 5.71%)</text>

    <!-- Block 2: Root Causes of LLM Error Correlation -->
    <rect x="15" y="210" width="410" height="150" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="25" y="230" class="box-title" font-size="11">Why Homogeneous LLMs Fail Correlatedly:</text>

    <text x="25" y="250" class="body-text" font-weight="bold">1. Common Training Distribution Ingestion:</text>
    <text x="35" y="265" class="body-text">• Identical web corpora (Common Crawl, GitHub, StackOverflow)</text>
    <text x="35" y="278" class="body-text">• Shared tokenizer vocabularies and BPE segmentation boundaries</text>

    <text x="25" y="298" class="body-text" font-weight="bold">2. Latent Weight Tensor Alignment:</text>
    <text x="35" y="313" class="body-text">• Sampling temperature T &gt; 0 alters token path, NOT underlying priors</text>
    <text x="35" y="326" class="body-text">• Models share deceptive heuristics and off-by-one blindspots</text>

    <text x="25" y="346" class="body-text" fill="#dc2626" font-weight="bold">Result: 5 wrong agents vote unanimously for flawed code.</text>

    <!-- Block 3: Production System Remedies -->
    <rect x="15" y="375" width="410" height="175" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="25" y="395" class="box-title" font-size="11" fill="#065f46">Architectural Countermeasures in Production:</text>

    <text x="25" y="415" class="body-text" font-weight="bold">A. Replace Voting with Mechanical Oracles:</text>
    <text x="35" y="430" class="body-text">• Gate merges on deterministic compilers, pytests, and linters</text>
    <text x="35" y="443" class="body-text">• Mathematical ground truth breaks statistical groupthink</text>

    <text x="25" y="463" class="body-text" font-weight="bold">B. Enforce Heterogeneous Architectural Lineage:</text>
    <text x="35" y="478" class="body-text">• Ensemble diverse model families (e.g., Claude + GPT + Gemini)</text>
    <text x="35" y="491" class="body-text">• Lowers pairwise correlation ρ from 0.45 down to &lt;0.12</text>

    <text x="25" y="511" class="body-text" font-weight="bold">C. Cap Majority Ensembles at M = 3..5:</text>
    <text x="35" y="526" class="body-text">• Beyond M=5, marginal accuracy gain is &lt;0.5% while cost explodes</text>
    <text x="35" y="539" class="body-text">• Reallocate token budget to deeper test-time deliberation</text>
  </g>

</svg>'''

# ==============================================================================
# 3. HIERARCHICAL CAPABILITY ATTENUATION TREE
# ==============================================================================
capability_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 700" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 15px; fill: #1e293b; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 500; font-size: 11.5px; fill: #475569; }
      .panel-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 13px; fill: #0f172a; }
      .box-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; font-size: 12px; fill: #1e293b; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 400; font-size: 10px; fill: #334155; }
      .code-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; font-size: 9.5px; fill: #0f172a; }
      .arrow-blue { stroke: #2563eb; stroke-width: 2; fill: none; marker-end: url(#arrow-blue-m); }
      .arrow-purple { stroke: #7c3aed; stroke-width: 2; fill: none; marker-end: url(#arrow-purple-m); }
      .arrow-green { stroke: #059669; stroke-width: 2; fill: none; marker-end: url(#arrow-green-m); }
      .arrow { stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#arrow-slate); }
    </style>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arrow-blue-m" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
    </marker>
    <marker id="arrow-purple-m" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c3aed" />
    </marker>
    <marker id="arrow-green-m" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#059669" />
    </marker>
  </defs>

  <!-- Outer Frame -->
  <rect x="0" y="0" width="1150" height="700" fill="#ffffff" />
  <rect x="15" y="15" width="1120" height="670" fill="none" stroke="#cbd5e1" stroke-width="1.5" rx="4" />

  <!-- Title -->
  <text x="35" y="45" class="title">HIERARCHICAL CAPABILITY ATTENUATION &amp; MACAROON DELEGATION LATTICE</text>
  <text x="35" y="63" class="subtitle">Monotonic Capability Attenuation: Cryptographic HMAC Chaining Bounding Child Agent Authority across Tools, Filesystem, TTL, and Budgets</text>

  <!-- LEFT: 3-TIER DELEGATION LATTICE -->
  <g transform="translate(35, 80)">
    <rect x="0" y="0" width="650" height="585" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="20" y="25" class="panel-label">Cryptographic Token Attenuation Flow (C_2 ⊆ C_1 ⊆ C_0)</text>

    <!-- NODE 0: ROOT SUPERVISOR (A_0) -->
    <g transform="translate(25, 45)">
      <rect x="0" y="0" width="600" height="110" fill="#ffffff" stroke="#2563eb" stroke-width="1.5" rx="3" />
      <rect x="0" y="0" width="600" height="26" fill="#eff6ff" rx="3" />
      <text x="15" y="18" class="box-title" fill="#1e40af">TIER 0: ROOT ORCHESTRATOR (Agent A_0)</text>
      <text x="585" y="18" class="code-text" text-anchor="end" fill="#2563eb" font-weight="bold">Master Secret K_0</text>

      <text x="15" y="45" class="code-text" font-weight="bold">Capability Envelope C_0 (Global Authority):</text>
      <text x="25" y="62" class="code-text">• Tools (T_0): {read, write, bash, git, deploy, rm, grep, find}</text>
      <text x="25" y="78" class="code-text">• Scope (F_0): / (Root global repository access) | Network: Egress allowed</text>
      <text x="25" y="94" class="code-text">• Budget: B_0 = $50.00 (500k tokens) | TTL: tau_max = 3600 seconds</text>
    </g>

    <!-- ATTENUATION STEP 1: CAVEAT INJECTION -->
    <path d="M 325 155 L 325 205" class="arrow-blue" />
    <rect x="180" y="165" width="290" height="30" fill="#eff6ff" stroke="#bfdbfe" rx="2" />
    <text x="325" y="178" class="code-text" text-anchor="middle" font-weight="bold" fill="#1e40af">Caveat 1: Attenuate to Worker Scope</text>
    <text x="325" y="190" class="code-text" text-anchor="middle" font-size="8.5">HMAC Tag: sigma_1 = HMAC(K_0, Caveats_1)</text>

    <!-- NODE 1: WORKER AGENT (A_1) -->
    <g transform="translate(25, 205)">
      <rect x="0" y="0" width="600" height="120" fill="#ffffff" stroke="#7c3aed" stroke-width="1.5" rx="3" />
      <rect x="0" y="0" width="600" height="26" fill="#f5f3ff" rx="3" />
      <text x="15" y="18" class="box-title" fill="#6d28d9">TIER 1: WORKER AGENT (Agent A_1) - Scoped Subtask</text>
      <text x="585" y="18" class="code-text" text-anchor="end" fill="#7c3aed" font-weight="bold">Macaroon M_1 [sigma_1]</text>

      <text x="15" y="45" class="code-text" font-weight="bold">Attenuated Capability C_1 (C_1 ⊆ C_0):</text>
      <text x="25" y="62" class="code-text">• Tools (T_1 ⊂ T_0): {read, write, grep, test} (deploy, rm STRIPPED)</text>
      <text x="25" y="78" class="code-text">• Scope (F_1 ⊂ F_0): /tmp/workspaces/task-102/ (Path restricted)</text>
      <text x="25" y="94" class="code-text">• Network: Hard blocked (No WAN egress) | TTL: tau_max &lt;= 600 s</text>
      <text x="25" y="110" class="code-text">• Budget: B_1 = $5.00 (50k tokens)</text>
    </g>

    <!-- ATTENUATION STEP 2: SUB-WORKER CAVEATS -->
    <path d="M 325 325 L 325 375" class="arrow-purple" />
    <rect x="175" y="335" width="300" height="30" fill="#f5f3ff" stroke="#ddd6fe" rx="2" />
    <text x="325" y="348" class="code-text" text-anchor="middle" font-weight="bold" fill="#6d28d9">Caveat 2: Attenuate to Build Sandbox</text>
    <text x="325" y="360" class="code-text" text-anchor="middle" font-size="8.5">Chained HMAC: sigma_2 = HMAC(sigma_1, Caveats_2)</text>

    <!-- NODE 2: COMPILER WORKER (A_2) -->
    <g transform="translate(25, 375)">
      <rect x="0" y="0" width="600" height="115" fill="#ffffff" stroke="#059669" stroke-width="1.5" rx="3" />
      <rect x="0" y="0" width="600" height="26" fill="#ecfdf5" rx="3" />
      <text x="15" y="18" class="box-title" fill="#065f46">TIER 2: COMPILER AGENT (Agent A_2) - Build Specialist</text>
      <text x="585" y="18" class="code-text" text-anchor="end" fill="#059669" font-weight="bold">Macaroon M_2 [sigma_2]</text>

      <text x="15" y="45" class="code-text" font-weight="bold">Leaf Capability C_2 (C_2 ⊆ C_1 ⊆ C_0):</text>
      <text x="25" y="62" class="code-text">• Tools (T_2 ⊂ T_1): {compile} strictly (read, write, grep STRIPPED)</text>
      <text x="25" y="78" class="code-text">• Scope (F_2 ⊂ F_1): /tmp/workspaces/task-102/build/ only</text>
      <text x="25" y="94" class="code-text">• TTL: tau_max &lt;= 60 seconds | Budget: B_2 = $0.50 (5k tokens)</text>
      <text x="25" y="108" class="code-text" fill="#065f46">Result: Leaf agent cannot tamper with source tree or parent context</text>
    </g>

    <!-- Monotonic Attenuation Math Callout -->
    <g transform="translate(25, 505)">
      <rect x="0" y="0" width="600" height="65" fill="#ffffff" stroke="#e2e8f0" rx="2" />
      <text x="15" y="20" class="box-title" font-size="10.5">Monotonic Attenuation Invariant:</text>
      <text x="15" y="38" class="code-text" font-size="9">C_{i+1} ⊆ C_i  &lt;=&gt;  (T_{i+1} ⊆ T_i) and (F_{i+1} ⊆ F_i) and (N_{i+1} ⊆ N_i) and (B_{i+1} &lt;= B_i)</text>
      <text x="15" y="54" class="body-text" font-size="9" fill="#64748b">Cryptographic guarantee: A child agent cannot remove caveats because it lacks intermediate secret sigma_{i-1}.</text>
    </g>
  </g>

  <!-- RIGHT: HOST RUNTIME VERIFICATION GATEWAY -->
  <g transform="translate(710, 80)">
    <rect x="0" y="0" width="405" height="585" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="3" />
    <text x="20" y="25" class="panel-label">Host Runtime Verification Gateway (A = 0)</text>

    <!-- Step 1: Request Ingestion -->
    <rect x="15" y="45" width="375" height="85" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="25" y="65" class="box-title" font-size="11">1. Protected RPC Ingestion</text>
    <text x="25" y="82" class="body-text">Agent dispatches RPC request:</text>
    <text x="25" y="98" class="code-text">request = {tool: "deploy", token: M_2}</text>
    <text x="25" y="115" class="body-text" fill="#dc2626">Notice: Agent A_2 attempting unauthorized tool</text>

    <!-- Step 2: Gateway Signature Verification -->
    <path d="M 202 130 L 202 160" class="arrow" />
    <rect x="15" y="160" width="375" height="115" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="25" y="180" class="box-title" font-size="11">2. Cryptographic Chain Recomputation</text>
    <text x="25" y="198" class="body-text">Gateway holds root secret K_0:</text>
    <text x="25" y="214" class="code-text" font-size="8.5">sigma_0' = K_0</text>
    <text x="25" y="228" class="code-text" font-size="8.5">sigma_1' = HMAC(sigma_0', Caveats_1)</text>
    <text x="25" y="242" class="code-text" font-size="8.5">sigma_2' = HMAC(sigma_1', Caveats_2)</text>
    <text x="25" y="260" class="code-text" font-size="8.5" fill="#059669">Assert: sigma_2' == M_2.signature (Tamper-proof)</text>

    <!-- Step 3: Caveat Evaluation Engine -->
    <path d="M 202 275 L 202 305" class="arrow" />
    <rect x="15" y="305" width="375" height="125" fill="#ffffff" stroke="#cbd5e1" rx="2" />
    <text x="25" y="325" class="box-title" font-size="11">3. Caveat Predicate Evaluation</text>
    <text x="25" y="345" class="code-text" font-size="8.5">Check 1: tool ∈ T_2?  ("deploy" ∈ {compile}?) → FALSE</text>
    <text x="25" y="362" class="code-text" font-size="8.5">Check 2: path ∈ F_2?  (/tmp/.../build)        → PASS</text>
    <text x="25" y="379" class="code-text" font-size="8.5">Check 3: epoch ≤ tau_max? (now ≤ epoch)       → PASS</text>
    <text x="25" y="396" class="code-text" font-size="8.5">Check 4: budget ≤ B_2?  (cost ≤ $0.50)        → PASS</text>
    <text x="25" y="415" class="code-text" fill="#dc2626" font-weight="bold">Gate Verdict: VIOLATION DETECTED</text>

    <!-- Step 4: Defense Enforcement Outcome -->
    <path d="M 202 430 L 202 460" class="arrow" />
    <rect x="15" y="460" width="375" height="110" fill="#fff1f2" stroke="#fca5a5" rx="2" />
    <text x="25" y="480" class="box-title" font-size="11" fill="#9f1239">4. Hard Isolation Enforcement (A = 0)</text>
    <text x="25" y="498" class="code-text" font-size="9" fill="#9f1239">• Immediate EPERM / PermissionDenied (HTTP 403)</text>
    <text x="25" y="513" class="code-text" font-size="9" fill="#9f1239">• Execution aborted; no host filesystem modified</text>
    <text x="25" y="528" class="code-text" font-size="9" fill="#9f1239">• Security exception logged to Supervisor A_0 WAL</text>
    <text x="25" y="548" class="body-text" font-size="9" font-weight="bold">Zero Ambient Authority prevents Confused Deputy attacks.</text>
  </g>

</svg>'''

# Paths
occ_path = str(Path(__file__).resolve().parent.parent) + "/books/vol3/15_multi_agent/images/svg/fig-vol3-occ-worktrees.svg"
ensemble_path1 = str(Path(__file__).resolve().parent.parent) + "/books/vol3/15_multi_agent/images/svg/vol3/ch15/fig-vol3-correlated-ensemble-failure.svg"
ensemble_path2 = str(Path(__file__).resolve().parent.parent) + "/books/vol3/15_multi_agent/images/svg/fig-vol3-correlated-ensemble-failure.svg"
capability_path = str(Path(__file__).resolve().parent.parent) + "/books/vol3/15_multi_agent/images/svg/capability-attenuation-tree.svg"

# Write OCC SVG
with open(occ_path, "w", encoding="utf-8") as f:
    f.write(occ_svg)
print(f"Written {len(occ_svg)} bytes to {occ_path}")

# Write Ensemble SVG
with open(ensemble_path1, "w", encoding="utf-8") as f:
    f.write(ensemble_svg)
with open(ensemble_path2, "w", encoding="utf-8") as f:
    f.write(ensemble_svg)
print(f"Written {len(ensemble_svg)} bytes to {ensemble_path1}")

# Write Capability SVG
with open(capability_path, "w", encoding="utf-8") as f:
    f.write(capability_svg)
print(f"Written {len(capability_svg)} bytes to {capability_path}")

print("Refined SVGs generated successfully.")
