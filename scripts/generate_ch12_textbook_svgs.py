import os

# Ensure symlinks are replaced with actual SVG files
svg_dir = "books/vol3/12_data_flywheel/images/svg"

# 1. fig-vol3-failure-diagnostic-tree.svg
diag_tree_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 560" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #334155; stroke-width: 1.5; rx: 6; ry: 6; }
      .box-gate { fill: #eff6ff; stroke: #2563eb; stroke-width: 1.8; rx: 6; ry: 6; }
      .box-infra { fill: #fef2f2; stroke: #dc2626; stroke-width: 1.8; rx: 6; ry: 6; }
      .box-syntax { fill: #fefce8; stroke: #ca8a04; stroke-width: 1.8; rx: 6; ry: 6; }
      .box-sem { fill: #f0fdf4; stroke: #16a34a; stroke-width: 1.8; rx: 6; ry: 6; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .node-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; text-anchor: middle; }
      .edge { stroke: #64748b; stroke-width: 1.5; fill: none; }
      .edge-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #1e293b; }
      .pill { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 4; ry: 4; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; fill: #0f172a; text-anchor: middle; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">Diagnostic Decision Tree for Agent Execution Failures</text>
  <text x="480" y="50" class="sub-text">Hierarchical Fault Classification Across Infrastructure, Syntax, and Semantic Substrates</text>

  <!-- Root: Intercepted Execution Anomaly -->
  <g transform="translate(360, 75)">
    <rect width="240" height="50" class="box" />
    <text x="120" y="22" class="node-title" fill="#0f172a">Trajectory Anomaly Detected</text>
    <text x="120" y="38" class="body-text">Intercepted via Supervisor or WAL</text>
  </g>

  <path d="M 480 125 L 480 160" class="edge" marker-end="url(#arrow)" />

  <!-- Level 1: Substrate vs Model Fault -->
  <g transform="translate(340, 160)">
    <rect width="280" height="54" class="box-gate" />
    <text x="140" y="22" class="node-title" fill="#1e40af">Gate 1: Substrate vs. Model Fault</text>
    <text x="140" y="40" class="body-text">Did execution engine or sandbox crash?</text>
  </g>

  <!-- Branch 1A: Infrastructure Crash -->
  <path d="M 340 187 L 180 187 L 180 240" class="edge" marker-end="url(#arrow)" />
  <rect x="195" y="165" width="130" height="20" class="pill" />
  <text x="260" y="179" class="edge-label" text-anchor="middle">Substrate / Host Fault</text>

  <g transform="translate(70, 240)">
    <rect width="220" height="85" class="box-infra" />
    <text x="110" y="22" class="node-title" fill="#991b1b">INFRASTRUCTURE FAULT</text>
    <text x="110" y="40" class="body-text">• OOM kill (exit 137), cgroup freeze</text>
    <text x="110" y="56" class="body-text">• Network RST, veth interface leak</text>
    <text x="110" y="72" class="mono">Action: Restart Sandbox / Purge</text>
  </g>

  <!-- Branch 1B: Model Action Emitted -->
  <path d="M 480 214 L 480 255" class="edge" marker-end="url(#arrow)" />
  <rect x="495" y="224" width="125" height="20" class="pill" />
  <text x="557" y="238" class="edge-label" text-anchor="middle">Model Payload Received</text>

  <!-- Level 2: Syntactic vs Semantic Gate -->
  <g transform="translate(340, 255)">
    <rect width="280" height="54" class="box-gate" />
    <text x="140" y="22" class="node-title" fill="#1e40af">Gate 2: Syntactic Validation</text>
    <text x="140" y="40" class="body-text">Does output conform to JSON/Tool Schema?</text>
  </g>

  <!-- Branch 2A: Syntax Failure -->
  <path d="M 620 282 L 780 282 L 780 340" class="edge" marker-end="url(#arrow)" />
  <rect x="635" y="260" width="130" height="20" class="pill" />
  <text x="700" y="274" class="edge-label" text-anchor="middle">Schema / Grammar Reject</text>

  <g transform="translate(670, 340)">
    <rect width="220" height="85" class="box-syntax" />
    <text x="110" y="22" class="node-title" fill="#854d0e">SYNTACTIC DEFECT</text>
    <text x="110" y="40" class="body-text">• Malformed JSON / unbalanced quotes</text>
    <text x="110" y="56" class="body-text">• Missing required schema fields</text>
    <text x="110" y="72" class="mono">Action: Logit Mask / Repair Prompt</text>
  </g>

  <!-- Branch 2B: Syntax Valid -> Gate 3 (Semantic) -->
  <path d="M 480 309 L 480 350" class="edge" marker-end="url(#arrow)" />
  <rect x="495" y="318" width="105" height="20" class="pill" />
  <text x="547" y="332" class="edge-label" text-anchor="middle">Syntax Valid</text>

  <!-- Level 3: Semantic Gate -->
  <g transform="translate(340, 350)">
    <rect width="280" height="54" class="box-gate" />
    <text x="140" y="22" class="node-title" fill="#1e40af">Gate 3: Semantic Progress Analysis</text>
    <text x="140" y="40" class="body-text">Evaluate state transitions and test diffs</text>
  </g>

  <!-- Branch 3A: State Oscillation Loop -->
  <path d="M 340 377 L 180 377 L 180 445" class="edge" marker-end="url(#arrow)" />
  <rect x="195" y="355" width="130" height="20" class="pill" />
  <text x="260" y="369" class="edge-label" text-anchor="middle">Hash Match: h(s_t) == h(s_t-k)</text>

  <g transform="translate(70, 445)">
    <rect width="220" height="85" class="box-sem" />
    <text x="110" y="22" class="node-title" fill="#166534">SEMANTIC OSCILLATION</text>
    <text x="110" y="40" class="body-text">• Non-advancing repair spin loop</text>
    <text x="110" y="56" class="body-text">• In-context autoregressive trap</text>
    <text x="110" y="72" class="mono">Action: Watchdog Jolt / Backtrack</text>
  </g>

  <!-- Branch 3B: Terminal Divergence / Failure -->
  <path d="M 480 404 L 480 445" class="edge" marker-end="url(#arrow)" />
  <rect x="495" y="414" width="130" height="20" class="pill" />
  <text x="560" y="428" class="edge-label" text-anchor="middle">Irrecoverable Divergence</text>

  <g transform="translate(370, 445)">
    <rect width="220" height="85" class="box-infra" />
    <text x="110" y="22" class="node-title" fill="#991b1b">TERMINAL FAILURE</text>
    <text x="110" y="40" class="body-text">• Test assertions permanently failing</text>
    <text x="110" y="56" class="body-text">• Budget exhaustion or pivot crossed</text>
    <text x="110" y="72" class="mono">Action: Mine as Hard Negative (τ^-)</text>
  </g>

</svg>
"""

# 2. fig-vol3-verifier-cascade.svg
# The Five-Stage Trajectory Acceptance Funnel
funnel_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 520" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .stage-box { fill: #f8fafc; stroke: #334155; stroke-width: 1.6; rx: 6; ry: 6; }
      .stage-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .metric-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 600; fill: #2563eb; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; text-anchor: middle; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .edge { stroke: #64748b; stroke-width: 1.8; fill: none; }
      .edge-drop { stroke: #dc2626; stroke-width: 1.5; stroke-dasharray: 4,4; fill: none; }
      .discard-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10px; fill: #dc2626; text-anchor: middle; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10.5px; fill: #0f172a; text-anchor: middle; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">The Five-Stage Trajectory Acceptance Verifier Funnel</text>
  <text x="480" y="50" class="sub-text">Progressive Filtering Pipeline Protecting Expensive Execution Tiers from Compute Exhaustion</text>

  <!-- 5 Sequential Funnel Stages -->
  <!-- Stage 1 -->
  <g transform="translate(30, 80)">
    <rect width="160" height="280" class="stage-box" />
    <text x="80" y="26" class="stage-hdr">Stage 1: Static AST</text>
    <text x="80" y="44" class="metric-hdr">Latency: &lt; 1 ms • $0</text>
    <line x1="15" y1="55" x2="145" y2="55" stroke="#cbd5e1" />
    <text x="80" y="78" class="body-text">• Tool schema check</text>
    <text x="80" y="96" class="body-text">• JSON parse validity</text>
    <text x="80" y="114" class="body-text">• Static lint &amp; syntax</text>
    <text x="80" y="132" class="body-text">• Zero sandbox compute</text>
    
    <text x="80" y="180" class="mono">Pass: ~70%</text>
    <text x="80" y="196" class="discard-text">Discard: 30% malformed</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <path d="M 190 220 L 218 220" class="edge" marker-end="url(#arrow)" />
  <path d="M 110 360 L 110 420" class="edge-drop" marker-end="url(#arrow-red)" />
  <text x="110" y="438" class="discard-text">Syntax Reject Sink</text>

  <!-- Stage 2 -->
  <g transform="translate(220, 80)">
    <rect width="160" height="280" class="stage-box" />
    <text x="80" y="26" class="stage-hdr">Stage 2: Hermetic Diff</text>
    <text x="80" y="44" class="metric-hdr">Latency: ~100 ms • Low</text>
    <line x1="15" y1="55" x2="145" y2="55" stroke="#cbd5e1" />
    <text x="80" y="78" class="body-text">• Git diff cleanliness</text>
    <text x="80" y="96" class="body-text">• Hermetic file sandbox</text>
    <text x="80" y="114" class="body-text">• Compile &amp; build pass</text>
    <text x="80" y="132" class="body-text">• Fast unit tests</text>
    
    <text x="80" y="180" class="mono">Pass: ~45%</text>
    <text x="80" y="196" class="discard-text">Discard: 25% compile fail</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <path d="M 380 220 L 408 220" class="edge" marker-end="url(#arrow)" />
  <path d="M 300 360 L 300 420" class="edge-drop" marker-end="url(#arrow-red)" />
  <text x="300" y="438" class="discard-text">Build Failure Sink</text>

  <!-- Stage 3 -->
  <g transform="translate(410, 80)">
    <rect width="160" height="280" class="stage-box" />
    <text x="80" y="26" class="stage-hdr">Stage 3: Full Test Suite</text>
    <text x="80" y="44" class="metric-hdr">Latency: ~2–10 s • Med</text>
    <line x1="15" y1="55" x2="145" y2="55" stroke="#cbd5e1" />
    <text x="80" y="78" class="body-text">• Pytest / Cargo test</text>
    <text x="80" y="96" class="body-text">• Integration assertions</text>
    <text x="80" y="114" class="body-text">• Multi-target regression</text>
    <text x="80" y="132" class="body-text">• Exit code = 0</text>
    
    <text x="80" y="180" class="mono">Pass: ~25%</text>
    <text x="80" y="196" class="discard-text">Discard: 20% test fail</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <path d="M 570 220 L 598 220" class="edge" marker-end="url(#arrow)" />
  <path d="M 490 360 L 490 420" class="edge-drop" marker-end="url(#arrow-red)" />
  <text x="490" y="438" class="discard-text">Hard Negative Mining (τ^-)</text>

  <!-- Stage 4 -->
  <g transform="translate(600, 80)">
    <rect width="160" height="280" class="stage-box" />
    <text x="80" y="26" class="stage-hdr">Stage 4: Anti-Tautology</text>
    <text x="80" y="44" class="metric-hdr">Latency: ~5 s • Med</text>
    <line x1="15" y1="55" x2="145" y2="55" stroke="#cbd5e1" />
    <text x="80" y="78" class="body-text">• Trivial assert check</text>
    <text x="80" y="96" class="body-text">• Mutation testing check</text>
    <text x="80" y="114" class="body-text">• Test deletion detector</text>
    <text x="80" y="132" class="body-text">• Zero reward hacking</text>
    
    <text x="80" y="180" class="mono">Pass: ~20%</text>
    <text x="80" y="196" class="discard-text">Discard: 5% reward hacks</text>
  </g>

  <!-- Arrow 4 -> 5 -->
  <path d="M 760 220 L 788 220" class="edge" marker-end="url(#arrow)" />
  <path d="M 680 360 L 680 420" class="edge-drop" marker-end="url(#arrow-red)" />
  <text x="680" y="438" class="discard-text">Tautology Discard Sink</text>

  <!-- Stage 5 -->
  <g transform="translate(790, 80)">
    <rect width="140" height="280" fill="#f0fdf4" stroke="#16a34a" stroke-width="2" rx="6" ry="6" />
    <text x="70" y="26" class="stage-hdr" fill="#166534">Stage 5: Final Gate</text>
    <text x="70" y="44" class="metric-hdr" fill="#166534">Golden Dataset</text>
    <line x1="15" y1="55" x2="125" y2="55" stroke="#bbf7d0" />
    <text x="70" y="85" class="body-text">• Cryptographic seal</text>
    <text x="70" y="105" class="body-text">• Role annotation</text>
    <text x="70" y="125" class="body-text">• Pristine vs. Recovery</text>
    <text x="70" y="150" class="mono" fill="#166534">Yield: ~18–20%</text>
    
    <rect x="15" y="190" width="110" height="40" fill="#ffffff" stroke="#16a34a" rx="4" />
    <text x="70" y="215" class="mono" font-weight="700" fill="#166534">τ ∈ D_train</text>
  </g>

  <!-- Summary bar at bottom -->
  <rect x="30" y="465" width="900" height="40" fill="#f8fafc" stroke="#64748b" stroke-width="1.2" rx="6" />
  <text x="480" y="489" class="body-text">Funnel Economics: 80%+ compute saved by executing sub-millisecond static gates before spinning up full microVM test environments.</text>

</svg>
"""

# 3. vol3-trajectory-typology.svg
# The Tripartite Trajectory Typology
typology_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 520" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .card { fill: #f8fafc; stroke: #475569; stroke-width: 1.5; rx: 6; ry: 6; }
      .hdr-pristine { fill: #f0fdf4; stroke: #16a34a; stroke-width: 1.5; rx: 4; ry: 4; }
      .hdr-recovery { fill: #fffbeb; stroke: #d97706; stroke-width: 1.5; rx: 4; ry: 4; }
      .hdr-negative { fill: #fef2f2; stroke: #dc2626; stroke-width: 1.5; rx: 4; ry: 4; }
      .step-box { fill: #ffffff; stroke: #94a3b8; stroke-width: 1.2; rx: 4; ry: 4; }
      .step-ok { fill: #f0fdf4; stroke: #16a34a; stroke-width: 1.5; rx: 4; ry: 4; }
      .step-err { fill: #fef2f2; stroke: #dc2626; stroke-width: 1.5; rx: 4; ry: 4; }
      .step-diag { fill: #eff6ff; stroke: #2563eb; stroke-width: 1.5; rx: 4; ry: 4; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .card-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; fill: #0f172a; text-anchor: middle; }
      .edge { stroke: #64748b; stroke-width: 1.5; fill: none; }
      .edge-green { stroke: #16a34a; stroke-width: 1.8; fill: none; }
      .edge-red { stroke: #dc2626; stroke-width: 1.8; fill: none; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="470" y="30" class="title-text">The Tripartite Trajectory Typology</text>
  <text x="470" y="50" class="sub-text">Structural Decomposition of Supervised and Preference Data Archetypes for Agent Policy Optimization</text>

  <!-- Archetype 1: Pristine Forward Path -->
  <g transform="translate(30, 80)">
    <rect width="270" height="420" class="card" />
    <rect x="15" y="15" width="240" height="34" class="hdr-pristine" />
    <text x="135" y="37" class="card-title" fill="#166534">Type 1: Pristine Path (τ_pristine)</text>
    
    <text x="20" y="70" class="body-text">• Direct, optimal execution graph</text>
    <text x="20" y="86" class="body-text">• Zero test or syntax failures</text>
    <text x="20" y="102" class="body-text">• Target: Standard SFT demonstration</text>

    <!-- Sequence Flow -->
    <rect x="25" y="125" width="220" height="40" class="step-ok" />
    <text x="135" y="145" class="mono">s_0 ──► a_1 (inspect repo)</text>

    <path d="M 135 165 L 135 190" class="edge-green" marker-end="url(#arrow-green)" />

    <rect x="25" y="195" width="220" height="40" class="step-ok" />
    <text x="135" y="215" class="mono">s_1 ──► a_2 (edit patch)</text>

    <path d="M 135 235 L 135 260" class="edge-green" marker-end="url(#arrow-green)" />

    <rect x="25" y="265" width="220" height="40" class="step-ok" />
    <text x="135" y="285" class="mono">s_2 ──► a_3 (pytest PASS)</text>

    <path d="M 135 305 L 135 330" class="edge-green" marker-end="url(#arrow-green)" />

    <rect x="25" y="335" width="220" height="40" class="step-ok" fill="#dcfce7" />
    <text x="135" y="355" class="mono" font-weight="700" fill="#166534">Goal Reached: Exit 0</text>

    <text x="20" y="402" class="body-text" font-weight="600" fill="#166534">Outcome: Fast convergence on clean tasks</text>
  </g>

  <!-- Archetype 2: Perturbed Recovery Path -->
  <g transform="translate(335, 80)">
    <rect width="270" height="420" class="card" />
    <rect x="15" y="15" width="240" height="34" class="hdr-recovery" />
    <text x="135" y="37" class="card-title" fill="#b45309">Type 2: Recovery Path (τ_recovery)</text>
    
    <text x="20" y="70" class="body-text">• Contains execution failure &amp; error</text>
    <text x="20" y="86" class="body-text">• Models self-diagnosis and repair</text>
    <text x="20" y="102" class="body-text">• Critical: Teaches systemic resilience</text>

    <!-- Sequence Flow -->
    <rect x="25" y="125" width="220" height="40" class="step-ok" />
    <text x="135" y="145" class="mono">s_0 ──► a_1 (initial patch)</text>

    <path d="M 135 165 L 135 185" class="edge" marker-end="url(#arrow)" />

    <rect x="25" y="188" width="220" height="42" class="step-err" />
    <text x="135" y="206" class="mono" fill="#dc2626">pytest: AssertionError (s_err)</text>
    <text x="135" y="222" class="mono" fill="#dc2626">Non-fatal failure intercepted</text>

    <path d="M 135 230 L 135 248" class="edge" marker-end="url(#arrow)" />

    <rect x="25" y="250" width="220" height="42" class="step-diag" />
    <text x="135" y="268" class="mono" fill="#1e40af">a_diag: inspect stack trace</text>
    <text x="135" y="284" class="mono" fill="#1e40af">Hypothesize root cause</text>

    <path d="M 135 292 L 135 310" class="edge-green" marker-end="url(#arrow-green)" />

    <rect x="25" y="312" width="220" height="40" class="step-ok" />
    <text x="135" y="332" class="mono">a_repair ──► pytest PASS</text>

    <path d="M 135 352 L 135 368" class="edge-green" marker-end="url(#arrow-green)" />

    <rect x="25" y="370" width="220" height="34" class="step-ok" fill="#fef3c7" />
    <text x="135" y="391" class="mono" font-weight="700" fill="#b45309">Self-Healed Terminal State</text>
  </g>

  <!-- Archetype 3: Hard Negative -->
  <g transform="translate(640, 80)">
    <rect width="270" height="420" class="card" />
    <rect x="15" y="15" width="240" height="34" class="hdr-negative" />
    <text x="135" y="37" class="card-title" fill="#dc2626">Type 3: Hard Negative (τ_negative)</text>
    
    <text x="20" y="70" class="body-text">• Unrecoverable error / spin loop</text>
    <text x="20" y="86" class="body-text">• Tagged at divergence point t*</text>
    <text x="20" y="102" class="body-text">• Target: DPO, PPO unlikelihood loss</text>

    <!-- Sequence Flow -->
    <rect x="25" y="125" width="220" height="40" class="step-ok" />
    <text x="135" y="145" class="mono">s_0 ──► a_1 (valid step)</text>

    <path d="M 135 165 L 135 185" class="edge" marker-end="url(#arrow)" />

    <rect x="25" y="188" width="220" height="42" class="step-err" stroke-dasharray="3,3" />
    <text x="135" y="206" class="mono" fill="#dc2626">Divergence Point t*</text>
    <text x="135" y="222" class="mono" fill="#dc2626">Hallucinated parameter / loop</text>

    <path d="M 135 230 L 135 250" class="edge-red" marker-end="url(#arrow-red)" />

    <rect x="25" y="252" width="220" height="42" class="step-err" />
    <text x="135" y="270" class="mono" fill="#dc2626">a_bad: repeat broken command</text>
    <text x="135" y="286" class="mono" fill="#dc2626">Watchdog hash collision</text>

    <path d="M 135 294 L 135 315" class="edge-red" marker-end="url(#arrow-red)" />

    <rect x="25" y="318" width="220" height="42" class="step-err" />
    <text x="135" y="336" class="mono" fill="#dc2626">Context window pollution</text>
    <text x="135" y="352" class="mono" fill="#dc2626">Compounding delirium</text>

    <path d="M 135 360 L 135 378" class="edge-red" marker-end="url(#arrow-red)" />

    <rect x="25" y="380" width="220" height="30" class="step-err" fill="#fee2e2" />
    <text x="135" y="399" class="mono" font-weight="700" fill="#dc2626">TERMINAL ABORT: τ^-</text>
  </g>

</svg>
"""

# 4. vol3-collection-pipeline.svg
# Distributed Trajectory Harvesting Pipeline Architecture
collection_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 520" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #334155; stroke-width: 1.5; rx: 6; ry: 6; }
      .box-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; }
      .body-center { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; text-anchor: middle; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .edge { stroke: #475569; stroke-width: 1.6; fill: none; }
      .edge-fb { stroke: #2563eb; stroke-width: 1.5; stroke-dasharray: 4,4; fill: none; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; fill: #0f172a; }
      .pill { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 3; ry: 3; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">Distributed Trajectory Harvesting Pipeline Architecture</text>
  <text x="480" y="50" class="sub-text">Asynchronous Coordination Between Task Dispatch, GPU Rollouts, MicroVM Sandboxes, and Sinks</text>

  <!-- Subsystem 1: Task Dispatcher -->
  <g transform="translate(40, 80)">
    <rect width="180" height="180" class="box" />
    <text x="90" y="24" class="box-hdr">1. Task Dispatcher</text>
    <line x1="15" y1="34" x2="165" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• Seed leasing engine</text>
    <text x="15" y="72" class="body-text">• Priority task queue</text>
    <text x="15" y="90" class="body-text">• Repo fixture manager</text>
    <text x="15" y="108" class="body-text">• Concurrency throttler</text>
    <text x="15" y="132" class="mono">lease_ttl = 600s</text>
  </g>

  <!-- Arrow 1 -> 2 & 3 -->
  <path d="M 220 140 L 268 140" class="edge" marker-end="url(#arrow)" />
  <text x="245" y="132" class="mono" text-anchor="middle">Task</text>

  <!-- Subsystem 2: Model Inference Fleet (Top) -->
  <g transform="translate(270, 80)">
    <rect width="210" height="115" class="box" fill="#eff6ff" stroke="#2563eb" />
    <text x="105" y="24" class="box-hdr" fill="#1e40af">2. Inference Fleet (vLLM)</text>
    <line x1="15" y1="34" x2="195" y2="34" stroke="#bfdbfe" />
    <text x="15" y="52" class="body-text">• Autoregressive token rollout</text>
    <text x="15" y="70" class="body-text">• Radix tree prefix caching</text>
    <text x="15" y="88" class="body-text">• Continuous batching engine</text>
  </g>

  <!-- Subsystem 3: MicroVM Execution Fleet (Bottom) -->
  <g transform="translate(270, 220)">
    <rect width="210" height="125" class="box" fill="#fefce8" stroke="#ca8a04" />
    <text x="105" y="24" class="box-hdr" fill="#854d0e">3. Sandbox Pool (Firecracker)</text>
    <line x1="15" y1="34" x2="195" y2="34" stroke="#fef08a" />
    <text x="15" y="52" class="body-text">• Warm microVM snapshot pool</text>
    <text x="15" y="70" class="body-text">• CoW ephemeral rootfs overlays</text>
    <text x="15" y="88" class="body-text">• Isolated network namespace</text>
    <text x="15" y="106" class="mono">Reset: ~15 ms via snapshot</text>
  </g>

  <!-- Tool execution bi-directional loop -->
  <path d="M 375 195 L 375 214" class="edge" marker-end="url(#arrow)" />
  <text x="415" y="210" class="mono" fill="#64748b">Tool Call</text>

  <!-- Arrow to Streaming Message Bus -->
  <path d="M 480 140 L 538 140" class="edge" marker-end="url(#arrow)" />
  <path d="M 480 280 L 510 280 L 510 160 L 538 160" class="edge" marker-end="url(#arrow)" />

  <!-- Subsystem 4: Streaming Broker & Staging -->
  <g transform="translate(540, 80)">
    <rect width="180" height="180" class="box" />
    <text x="90" y="24" class="box-hdr">4. Stream Ingestion</text>
    <line x1="15" y1="34" x2="165" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• Kafka / NATS queue</text>
    <text x="15" y="72" class="body-text">• Backpressure feedback</text>
    <text x="15" y="90" class="body-text">• WAL trace buffering</text>
    <text x="15" y="108" class="body-text">• In-memory staging</text>
    <text x="15" y="132" class="mono">Rate: 50k tokens/s</text>
  </g>

  <!-- Arrow 4 -> 5 -->
  <path d="M 720 170 L 768 170" class="edge" marker-end="url(#arrow)" />

  <!-- Subsystem 5: Verifier & Sinks -->
  <g transform="translate(770, 80)">
    <rect width="160" height="265" class="box" fill="#f0fdf4" stroke="#16a34a" />
    <text x="80" y="24" class="box-hdr" fill="#166534">5. Verifier &amp; Sinks</text>
    <line x1="15" y1="34" x2="145" y2="34" stroke="#bbf7d0" />
    <text x="15" y="54" class="body-text">• 5-stage filter cascade</text>
    <text x="15" y="74" class="body-text">• Taint sanitization</text>
    <text x="15" y="94" class="body-text">• Lineage manifest</text>
    <text x="15" y="114" class="body-text">• Columnar Parquet</text>
    
    <rect x="15" y="145" width="130" height="40" fill="#ffffff" stroke="#16a34a" rx="4" />
    <text x="80" y="170" class="mono" font-weight="700" fill="#166534">Object Store Sink</text>

    <rect x="15" y="200" width="130" height="40" fill="#ffffff" stroke="#dc2626" rx="4" />
    <text x="80" y="225" class="mono" font-weight="700" fill="#dc2626">Quarantine Deadletter</text>
  </g>

  <!-- Closed-loop Backpressure Arrow from Stream Ingestion to Dispatcher -->
  <path d="M 630 260 L 630 400 L 130 400 L 130 266" class="edge-fb" marker-end="url(#arrow-blue)" />
  <rect x="300" y="388" width="220" height="24" class="pill" />
  <text x="410" y="404" class="body-center" fill="#2563eb" font-weight="600">Closed-Loop Backpressure Feedback</text>

  <!-- Footer description -->
  <rect x="40" y="440" width="890" height="50" fill="#f8fafc" stroke="#64748b" stroke-width="1.2" rx="6" />
  <text x="485" y="462" class="body-center">Decoupled Architecture Invariant: Inference generation is never stalled by slow disk I/O; ephemeral microVMs are reclaimed asynchronously.</text>
  <text x="485" y="478" class="body-center">Queue depth governs token lease emission: backpressure halts dispatch before memory saturation triggers host thrashing.</text>

</svg>
"""

# 5. vol3-provenance-envelope.svg
# The Trajectory Lineage Envelope and Sanitization Flow
envelope_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 540" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #334155; stroke-width: 1.5; rx: 6; ry: 6; }
      .box-env { fill: #eff6ff; stroke: #2563eb; stroke-width: 1.8; rx: 6; ry: 6; }
      .box-pass { fill: #ffffff; stroke: #64748b; stroke-width: 1.4; rx: 6; ry: 6; }
      .box-sink { fill: #f0fdf4; stroke: #16a34a; stroke-width: 1.8; rx: 6; ry: 6; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .hdr-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #334155; }
      .edge { stroke: #475569; stroke-width: 1.6; fill: none; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; fill: #0f172a; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">Trajectory Lineage Envelope &amp; 3-Pass Sanitization Pipeline</text>
  <text x="480" y="50" class="sub-text">Cryptographic Lineage Attestation Paired with Syntactically Invariant Secret Masking</text>

  <!-- Left Side: Immutable Provenance Envelope Structure -->
  <g transform="translate(40, 80)">
    <rect width="360" height="420" class="box-env" />
    <text x="20" y="28" class="hdr-text" fill="#1e40af">CRYPTOGRAPHIC LINEAGE ENVELOPE</text>
    <line x1="20" y1="38" x2="340" y2="38" stroke="#bfdbfe" />

    <!-- Section 1: Provenance Metadata -->
    <text x="20" y="60" class="mono" font-weight="700">1. Header Manifest (Immutable):</text>
    <text x="30" y="78" class="mono">• trajectory_id: UUIDv7</text>
    <text x="30" y="94" class="mono">• base_checkpoint: LLaMA-3-70B-v1.2</text>
    <text x="30" y="110" class="mono">• prompt_template_sha256: 0x4f8a...</text>
    <text x="30" y="126" class="mono">• sandbox_rootfs_sha256: 0x9b12...</text>
    <text x="30" y="142" class="mono">• host_kernel: 6.8.0-custom-ebpf</text>

    <line x1="20" y1="160" x2="340" y2="160" stroke="#bfdbfe" stroke-dasharray="3,3" />

    <!-- Section 2: Execution Trace Payload -->
    <text x="20" y="180" class="mono" font-weight="700">2. Token &amp; State Payload:</text>
    <text x="30" y="198" class="mono">• actions: [tool_call_1, tool_call_2, ...]</text>
    <text x="30" y="214" class="mono">• observations: [stdout, stderr, exit_code]</text>
    <text x="30" y="230" class="mono">• state_hash_ring: [H_0, H_1, ..., H_n]</text>
    <text x="30" y="246" class="mono">• diff_patch: Unified Diff (+12, -4)</text>

    <line x1="20" y1="265" x2="340" y2="265" stroke="#bfdbfe" stroke-dasharray="3,3" />

    <!-- Section 3: Verification Certificate -->
    <text x="20" y="285" class="mono" font-weight="700">3. Verification Proof:</text>
    <text x="30" y="303" class="mono">• verifier_version: v2.4.1</text>
    <text x="30" y="319" class="mono">• hermetic_eval_exit: 0 (PASS)</text>
    <text x="30" y="335" class="mono">• role_label: RECOVERY_DEMONSTRATION</text>

    <line x1="20" y1="355" x2="340" y2="355" stroke="#bfdbfe" />

    <!-- Section 4: Merkle Seal -->
    <rect x="20" y="365" width="320" height="40" fill="#ffffff" stroke="#2563eb" rx="4" />
    <text x="180" y="390" class="mono" text-anchor="middle" font-weight="700" fill="#1e40af">MERKLE_ROOT: SHA256(H || P || V)</text>
  </g>

  <!-- Arrow from Envelope to Sanitization Pipeline -->
  <path d="M 400 290 L 452 290" class="edge" marker-end="url(#arrow)" />

  <!-- Right Side: 3-Pass Sanitization Pipeline -->
  <g transform="translate(460, 80)">
    <rect width="460" height="420" class="box" />
    <text x="20" y="28" class="hdr-text">THREE-PASS SANITIZATION PIPELINE</text>
    <line x1="20" y1="38" x2="440" y2="38" stroke="#cbd5e1" />

    <!-- Pass 1 -->
    <g transform="translate(20, 50)">
      <rect width="420" height="70" class="box-pass" />
      <text x="15" y="22" class="mono" font-weight="700">PASS 1: REGEX &amp; HIGH-ENTROPY SCANNER</text>
      <text x="15" y="40" class="body-text">• Scan AWS keys (AKIA...), GitHub tokens (ghp_...), SSH headers</text>
      <text x="15" y="56" class="body-text">• Sliding-window Shannon entropy threshold: H(S) &gt;= 4.5 bits/char</text>
    </g>

    <path d="M 230 120 L 230 142" class="edge" marker-end="url(#arrow)" />

    <!-- Pass 2 -->
    <g transform="translate(20, 145)">
      <rect width="420" height="70" class="box-pass" />
      <text x="15" y="22" class="mono" font-weight="700">PASS 2: CONTEXT-AWARE NER &amp; IP DETECTOR</text>
      <text x="15" y="40" class="body-text">• Extract company names, employee emails, internal domain URLs</text>
      <text x="15" y="56" class="body-text">• Private RFC 1918 IPv4/IPv6 address masking (10.0/8, 192.168/16)</text>
    </g>

    <path d="M 230 215 L 230 237" class="edge" marker-end="url(#arrow)" />

    <!-- Pass 3 -->
    <g transform="translate(20, 240)">
      <rect width="420" height="75" class="box-pass" />
      <text x="15" y="22" class="mono" font-weight="700">PASS 3: SYNTACTICALLY INVARIANT TOKEN MASKING</text>
      <text x="15" y="40" class="body-text">• Parse JSON AST: preserve quotes, delimiters, and key structures</text>
      <text x="15" y="56" class="body-text">• Substitute secret values with deterministic salt pseudonyms</text>
      <text x="15" y="70" class="mono" fill="#16a34a">Token: &lt;SECRET:HMAC_SHA256(k_salt, secret)[:10]&gt;</text>
    </g>

    <path d="M 230 315 L 230 337" class="edge" marker-end="url(#arrow)" />

    <!-- Sanitized Sink Output -->
    <g transform="translate(20, 340)">
      <rect width="420" height="65" class="box-sink" />
      <text x="210" y="25" class="hdr-text" fill="#166534" text-anchor="middle">SANITIZED TRAINING CORPUS SINK</text>
      <text x="210" y="45" class="body-text" text-anchor="middle">Preserves 100% syntactic AST integrity with zero credential leakage</text>
    </g>

  </g>

</svg>
"""

# 6. vol3-data-composition-ablation.svg
# Downstream Task Success vs Trajectory Corpus Composition
ablation_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 520" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .grid { stroke: #e2e8f0; stroke-width: 1; }
      .axis { stroke: #334155; stroke-width: 1.5; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .axis-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; font-weight: 600; fill: #1e293b; text-anchor: middle; }
      .tick-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #475569; }
      .curve-easy { stroke: #16a34a; stroke-width: 2.5; fill: none; }
      .curve-complex { stroke: #2563eb; stroke-width: 3; fill: none; }
      .curve-ood { stroke: #dc2626; stroke-width: 2.5; stroke-dasharray: 5,4; fill: none; }
      .legend-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #1e293b; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10.5px; fill: #0f172a; }
      .peak-marker { fill: #2563eb; stroke: #ffffff; stroke-width: 2; }
    </style>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="460" y="30" class="title-text">Downstream Task Success vs. Trajectory Corpus Composition</text>
  <text x="460" y="50" class="sub-text">Ablation of Recovery Trajectory Ratio (τ_recovery) Under Fixed Training Token Budget</text>

  <!-- Plot Area: x from 120 to 820 (width 700), y from 90 to 420 (height 330) -->
  <!-- Horizontal Grid Lines (Success Rate: 0%, 20%, 40%, 60%, 80%, 100%) -->
  <!-- y = 420 (0%), 354 (20%), 288 (40%), 222 (60%), 156 (80%), 90 (100%) -->
  <line x1="120" y1="90" x2="820" y2="90" class="grid" />
  <text x="105" y="94" class="tick-text" text-anchor="end">100%</text>

  <line x1="120" y1="156" x2="820" y2="156" class="grid" />
  <text x="105" y="160" class="tick-text" text-anchor="end">80%</text>

  <line x1="120" y1="222" x2="820" y2="222" class="grid" />
  <text x="105" y="226" class="tick-text" text-anchor="end">60%</text>

  <line x1="120" y1="288" x2="820" y2="288" class="grid" />
  <text x="105" y="292" class="tick-text" text-anchor="end">40%</text>

  <line x1="120" y1="354" x2="820" y2="354" class="grid" />
  <text x="105" y="358" class="tick-text" text-anchor="end">20%</text>

  <line x1="120" y1="420" x2="820" y2="420" class="grid" />
  <text x="105" y="424" class="tick-text" text-anchor="end">0%</text>

  <!-- Vertical Grid Lines (% Recovery Data: 0%, 10%, 20%, 30%, 40%, 50%, 60%) -->
  <!-- dx = 700 / 6 = 116.66 px per 10% -->
  <!-- x = 120 (0%), 236.6 (10%), 353.3 (20%), 470 (30%), 586.6 (40%), 703.3 (50%), 820 (60%) -->
  <line x1="120" y1="90" x2="120" y2="420" class="axis" />
  <text x="120" y="440" class="tick-text" text-anchor="middle">0% (All Pristine)</text>

  <line x1="237" y1="90" x2="237" y2="420" class="grid" />
  <text x="237" y="440" class="tick-text" text-anchor="middle">10%</text>

  <line x1="353" y1="90" x2="353" y2="420" class="grid" />
  <text x="353" y="440" class="tick-text" text-anchor="middle">20%</text>

  <line x1="470" y1="90" x2="470" y2="420" class="axis" stroke-dasharray="3,3" />
  <text x="470" y="440" class="tick-text" text-anchor="middle" font-weight="700" fill="#2563eb">30% (Optimal)</text>

  <line x1="587" y1="90" x2="587" y2="420" class="grid" />
  <text x="587" y="440" class="tick-text" text-anchor="middle">40%</text>

  <line x1="703" y1="90" x2="703" y2="420" class="grid" />
  <text x="703" y="440" class="tick-text" text-anchor="middle">50%</text>

  <line x1="820" y1="90" x2="820" y2="420" class="grid" />
  <text x="820" y="440" class="tick-text" text-anchor="middle">60%</text>

  <!-- Bottom Axis Line -->
  <line x1="120" y1="420" x2="820" y2="420" class="axis" />

  <!-- Curve A: Easy / Short Tasks (Green) -->
  <!-- 0%: 82% (y=150), 10%: 84% (y=143), 20%: 83% (y=146), 30%: 80% (y=156), 40%: 76% (y=169), 50%: 70% (y=189), 60%: 64% (y=209) -->
  <path d="M 120 150 C 200 140 300 144 470 156 C 587 169 703 189 820 209" class="curve-easy" />

  <!-- Curve B: Complex / Multi-Step Tasks (Blue) -->
  <!-- 0%: 41% (y=285), 10%: 52% (y=248), 20%: 64% (y=209), 30%: 73% (y=179, PEAK), 40%: 68% (y=195), 50%: 59% (y=225), 60%: 50% (y=255) -->
  <path d="M 120 285 C 237 240 353 195 470 179 C 587 195 703 225 820 255" class="curve-complex" />
  <circle cx="470" cy="179" r="6" class="peak-marker" />
  
  <rect x="420" y="145" width="100" height="24" fill="#eff6ff" stroke="#2563eb" rx="4" />
  <text x="470" y="161" class="mono" font-weight="700" fill="#1e40af" text-anchor="middle">Peak: 73.2%</text>

  <!-- Curve C: Out-of-Distribution Error Recovery (Red dashed) -->
  <!-- 0%: 15% (y=370), 10%: 28% (y=327), 20%: 42% (y=281), 30%: 56% (y=235), 40%: 61% (y=219), 50%: 62% (y=216), 60%: 60% (y=222) -->
  <path d="M 120 370 C 237 320 353 260 470 235 C 587 219 703 216 820 222" class="curve-ood" />

  <!-- Axis Titles -->
  <text x="470" y="475" class="axis-label">Corpus Composition Ratio: % Recovery Demonstrations (Remainder Pristine)</text>
  <text x="45" y="255" class="axis-label" transform="rotate(-90 45 255)">Benchmark Task Success Rate (%)</text>

  <!-- Legend Box -->
  <g transform="translate(560, 100)">
    <rect width="250" height="85" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" rx="6" />
    <line x1="15" y1="22" x2="45" y2="22" class="curve-complex" />
    <text x="55" y="26" class="legend-text">Complex Multi-Step Repos</text>

    <line x1="15" y1="44" x2="45" y2="44" class="curve-easy" />
    <text x="55" y="48" class="legend-text">Single-Turn Unit Tasks</text>

    <line x1="15" y1="66" x2="45" y2="66" class="curve-ood" />
    <text x="55" y="70" class="legend-text">Perturbed Environmental Faults</text>
  </g>

</svg>
"""

# 7. vol3-harvesting-synthesis.svg
# End-to-End Trajectory Harvesting Architecture (6-Stage Pipeline)
synthesis_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 520" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .stage-box { fill: #f8fafc; stroke: #334155; stroke-width: 1.6; rx: 6; ry: 6; }
      .stage-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12.5px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10.5px; fill: #334155; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .edge { stroke: #475569; stroke-width: 1.6; fill: none; }
      .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 9.5px; fill: #0f172a; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">End-to-End Trajectory Harvesting Pipeline Architecture</text>
  <text x="480" y="50" class="sub-text">Lifecycle Progression from Runtime Execution Capture to Cryptographically Sealed Training Partitions</text>

  <!-- Top Row: Stages 1, 2, 3 -->
  <!-- Stage 1 -->
  <g transform="translate(40, 80)">
    <rect width="260" height="175" class="stage-box" />
    <text x="130" y="24" class="stage-hdr">Stage 1: Raw Execution Capture</text>
    <line x1="15" y1="34" x2="245" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• Agent Runtime WAL / ACB hooks</text>
    <text x="15" y="72" class="body-text">• Stdio, stdin, and tool RPC capture</text>
    <text x="15" y="90" class="body-text">• Isolated microVM execution traces</text>
    <text x="15" y="108" class="body-text">• Ephemeral CoW filesystem diffs</text>
    <text x="15" y="136" class="mono">Output: Raw Session Envelope τ_raw</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <path d="M 300 167 L 348 167" class="edge" marker-end="url(#arrow)" />

  <!-- Stage 2 -->
  <g transform="translate(350, 80)">
    <rect width="260" height="175" class="stage-box" />
    <text x="130" y="24" class="stage-hdr">Stage 2: Staged Verifier Cascade</text>
    <line x1="15" y1="34" x2="245" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• V1: Static schema &amp; AST lint (&lt;1ms)</text>
    <text x="15" y="72" class="body-text">• V2: Hermetic test suite runner</text>
    <text x="15" y="90" class="body-text">• V3: Flakiness &amp; seed perturbation</text>
    <text x="15" y="108" class="body-text">• V4: Tautology &amp; reward hack filter</text>
    <text x="15" y="136" class="mono">Output: Verified Trajectory Stream</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <path d="M 610 167 L 658 167" class="edge" marker-end="url(#arrow)" />

  <!-- Stage 3 -->
  <g transform="translate(660, 80)">
    <rect width="260" height="175" class="stage-box" />
    <text x="130" y="24" class="stage-hdr">Stage 3: Role Annotation</text>
    <line x1="15" y1="34" x2="245" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• Pristine path classification (τ_pristine)</text>
    <text x="15" y="72" class="body-text">• Recovery path tag with error trace</text>
    <text x="15" y="90" class="body-text">• Hard negative mining (τ^- at step t*)</text>
    <text x="15" y="108" class="body-text">• Task difficulty &amp; step-count scoring</text>
    <text x="15" y="136" class="mono">Output: Annotated Archetype Bundle</text>
  </g>

  <!-- Turnaround Arrow from Stage 3 to Stage 4 -->
  <path d="M 790 255 L 790 295 L 790 315" class="edge" marker-end="url(#arrow)" />

  <!-- Bottom Row: Stages 4, 5, 6 (Right to Left flow) -->
  <!-- Stage 4 -->
  <g transform="translate(660, 315)">
    <rect width="260" height="175" class="stage-box" />
    <text x="130" y="24" class="stage-hdr">Stage 4: Multi-Pass Sanitization</text>
    <line x1="15" y1="34" x2="245" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• Regex scanning &amp; Shannon entropy</text>
    <text x="15" y="72" class="body-text">• Context-aware NER &amp; RFC 1918 IPs</text>
    <text x="15" y="90" class="body-text">• AST-preserving pseudonym salt tokens</text>
    <text x="15" y="108" class="body-text">• Differential privacy verification</text>
    <text x="15" y="136" class="mono">Output: Redacted Safe Payload</text>
  </g>

  <!-- Arrow 4 -> 5 -->
  <path d="M 660 402 L 618 402" class="edge" marker-end="url(#arrow)" />

  <!-- Stage 5 -->
  <g transform="translate(350, 315)">
    <rect width="260" height="175" class="stage-box" />
    <text x="130" y="24" class="stage-hdr">Stage 5: Cryptographic Lineage</text>
    <line x1="15" y1="34" x2="245" y2="34" stroke="#cbd5e1" />
    <text x="15" y="54" class="body-text">• SHA-256 Merkle tree manifest</text>
    <text x="15" y="72" class="body-text">• Checkpoint, rootfs, &amp; tool digest seal</text>
    <text x="15" y="90" class="body-text">• Immutable provenance signature</text>
    <text x="15" y="108" class="body-text">• Auditable compliance certificate</text>
    <text x="15" y="136" class="mono">Output: Sealed Provenance Envelope</text>
  </g>

  <!-- Arrow 5 -> 6 -->
  <path d="M 350 402 L 308 402" class="edge" marker-end="url(#arrow)" />

  <!-- Stage 6 -->
  <g transform="translate(40, 315)">
    <rect width="260" height="175" class="stage-box" fill="#f0fdf4" stroke="#16a34a" />
    <text x="130" y="24" class="stage-hdr" fill="#166534">Stage 6: Disjoint Split Partition</text>
    <line x1="15" y1="34" x2="245" y2="34" stroke="#bbf7d0" />
    <text x="15" y="54" class="body-text">• Family-tree disjoint train/eval split</text>
    <text x="15" y="72" class="body-text">• Zero repository overlap boundary</text>
    <text x="15" y="90" class="body-text">• Leakage-proof evaluation holdout</text>
    <text x="15" y="108" class="body-text">• Columnar Parquet / Arrow tables</text>
    <text x="15" y="136" class="mono" font-weight="700" fill="#166534">Final: Versioned Training Corpus</text>
  </g>

</svg>
"""

# Write all 7 files, replacing any symlinks
files = [
    ("fig-vol3-failure-diagnostic-tree.svg", diag_tree_svg),
    ("fig-vol3-verifier-cascade.svg", funnel_svg),
    ("vol3-trajectory-typology.svg", typology_svg),
    ("vol3-collection-pipeline.svg", collection_svg),
    ("vol3-provenance-envelope.svg", envelope_svg),
    ("vol3-data-composition-ablation.svg", ablation_svg),
    ("vol3-harvesting-synthesis.svg", synthesis_svg),
]

for filename, content in files:
    filepath = os.path.join(svg_dir, filename)
    if os.path.islink(filepath) or os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Wrote {filename}")

