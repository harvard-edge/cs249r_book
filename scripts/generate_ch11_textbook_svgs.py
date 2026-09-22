import os

# 1. circuit_breaker_state_machine.svg
fsm_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 490" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #334155; stroke-width: 1.8; rx: 8; ry: 8; }
      .box-active { fill: #f0fdf4; stroke: #16a34a; stroke-width: 2; rx: 8; ry: 8; }
      .box-alert { fill: #fef2f2; stroke: #dc2626; stroke-width: 2; rx: 8; ry: 8; }
      .box-probe { fill: #fefce8; stroke: #ca8a04; stroke-width: 2; rx: 8; ry: 8; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .state-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 700; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #334155; text-anchor: middle; }
      .label-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #1e293b; }
      .edge { stroke: #64748b; stroke-width: 1.6; fill: none; }
      .edge-alert { stroke: #dc2626; stroke-width: 1.8; fill: none; }
      .edge-ok { stroke: #16a34a; stroke-width: 1.8; fill: none; }
      .code-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10.5px; text-anchor: middle; }
      .pill { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 4; ry: 4; }
      .pill-alert { fill: #ffffff; stroke: #fca5a5; stroke-width: 1; rx: 4; ry: 4; }
      .pill-ok { fill: #ffffff; stroke: #86efac; stroke-width: 1; rx: 4; ry: 4; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="470" y="32" class="title-text">Tri-State Tool Circuit Breaker Finite State Machine</text>
  <text x="470" y="52" class="sub-text">Deterministic Fault Containment Boundary for External API &amp; Sandbox Invocations</text>

  <!-- State 1: CLOSED -->
  <g transform="translate(80, 140)">
    <rect width="230" height="150" class="box-active" />
    <text x="115" y="30" class="state-title" fill="#166534">STATE: CLOSED</text>
    <line x1="20" y1="42" x2="210" y2="42" stroke="#bbf7d0" stroke-width="1" />
    <text x="115" y="66" class="body-text">• Normal tool dispatch</text>
    <text x="115" y="86" class="body-text">• Invocations routed to host/API</text>
    <text x="115" y="106" class="body-text">• Sliding error counter: E / N</text>
    <text x="115" y="130" class="code-text" fill="#166534">pass_through() = True</text>
  </g>

  <!-- State 2: OPEN -->
  <g transform="translate(630, 140)">
    <rect width="230" height="150" class="box-alert" />
    <text x="115" y="30" class="state-title" fill="#991b1b">STATE: OPEN</text>
    <line x1="20" y1="42" x2="210" y2="42" stroke="#fecaca" stroke-width="1" />
    <text x="115" y="66" class="body-text">• Fast-fail containment</text>
    <text x="115" y="86" class="body-text">• Invocations rejected locally</text>
    <text x="115" y="106" class="body-text">• Zero socket / token waste</text>
    <text x="115" y="130" class="code-text" fill="#991b1b">raise CircuitOpenError()</text>
  </g>

  <!-- State 3: HALF-OPEN -->
  <g transform="translate(355, 320)">
    <rect width="230" height="145" class="box-probe" />
    <text x="115" y="30" class="state-title" fill="#854d0e">STATE: HALF-OPEN</text>
    <line x1="20" y1="42" x2="210" y2="42" stroke="#fef08a" stroke-width="1" />
    <text x="115" y="66" class="body-text">• Trial probe canary phase</text>
    <text x="115" y="86" class="body-text">• Allow 1 trial request through</text>
    <text x="115" y="106" class="body-text">• Concurrency throttled to 1</text>
    <text x="115" y="128" class="code-text" fill="#854d0e">canary_in_flight = 1</text>
  </g>

  <!-- Transitions -->
  <!-- 1. Self loop CLOSED: Success -->
  <path d="M 150 140 C 150 75 240 75 240 134" class="edge-ok" marker-end="url(#arrow-green)" />
  <rect x="135" y="65" width="160" height="22" class="pill-ok" />
  <text x="215" y="80" class="label-text" text-anchor="middle" fill="#166534">Success: Reset failure count</text>

  <!-- 2. CLOSED -> OPEN: Failure Rate Exceeded -->
  <path d="M 310 195 L 622 195" class="edge-alert" marker-end="url(#arrow-red)" />
  <rect x="385" y="172" width="175" height="24" class="pill-alert" />
  <text x="472" y="188" class="label-text" text-anchor="middle" fill="#dc2626">Failure rate &gt; τ_fail (in window W)</text>

  <!-- 3. OPEN -> HALF-OPEN: Cooldown Expired (curves down from right of OPEN) -->
  <path d="M 800 290 C 800 400 630 405 593 405" class="edge" marker-end="url(#arrow)" />
  <rect x="650" y="393" width="170" height="24" class="pill" />
  <text x="735" y="409" class="label-text" text-anchor="middle">Cooldown T_cooldown elapsed</text>

  <!-- 4. HALF-OPEN -> CLOSED: Canary Succeeded -->
  <path d="M 355 395 C 240 395 190 350 195 296" class="edge-ok" marker-end="url(#arrow-green)" />
  <rect x="150" y="390" width="155" height="24" class="pill-ok" />
  <text x="227" y="406" class="label-text" text-anchor="middle" fill="#166534">Canary probe succeeds</text>

  <!-- 5. HALF-OPEN -> OPEN: Canary Failed -->
  <path d="M 520 320 C 550 250 610 245 624 245" class="edge-alert" marker-end="url(#arrow-red)" />
  <rect x="475" y="240" width="135" height="22" class="pill-alert" />
  <text x="542" y="255" class="label-text" text-anchor="middle" fill="#dc2626">Canary probe fails</text>

</svg>
"""

with open("books/vol3/11_failure_recovery/images/svg/circuit_breaker_state_machine.svg", "w") as f:
    f.write(fsm_svg)

# 2. semantic-watchdog-oscillation.svg
watchdog_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 480" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #64748b; stroke-width: 1.5; rx: 6; ry: 6; }
      .box-highlight { fill: #fef2f2; stroke: #dc2626; stroke-width: 2; rx: 6; ry: 6; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .step-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 600; fill: #1e293b; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #334155; text-anchor: middle; }
      .policy-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12.5px; font-weight: 700; fill: #0f172a; text-anchor: start; }
      .policy-body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #334155; text-anchor: start; }
      .mono-text { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; fill: #0f172a; text-anchor: middle; }
      .mono-alert { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12px; font-weight: 700; fill: #dc2626; text-anchor: middle; }
      .edge { stroke: #64748b; stroke-width: 1.5; fill: none; }
      .edge-alert { stroke: #dc2626; stroke-width: 2; stroke-dasharray: 5,4; fill: none; }
      .annot-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #475569; }
      .pill-alert { fill: #ffffff; stroke: #fca5a5; stroke-width: 1.5; rx: 6; ry: 6; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="470" y="32" class="title-text">Semantic Watchdog: State Hash Sliding Window Oscillation Detection</text>
  <text x="470" y="52" class="sub-text">Tracking Environmental State Digest Invariants to Terminate Non-Advancing Cyclic Repair Loops</text>

  <!-- Step columns: t-3, t-2, t-1, t -->
  <!-- Col 1: t-3 -->
  <g transform="translate(60, 80)">
    <text x="80" y="20" class="step-hdr">Step t - 3</text>
    <rect x="10" y="35" width="140" height="48" class="box" />
    <text x="80" y="64" class="body-text">State s_{t-3}</text>
    
    <path d="M 80 83 L 80 123" class="edge" marker-end="url(#arrow)" />
    
    <rect x="10" y="125" width="140" height="38" class="box" />
    <text x="80" y="149" class="mono-text">Hash: 0x8F4A</text>
  </g>

  <!-- Col 2: t-2 (Cycle Origin) -->
  <g transform="translate(280, 80)">
    <text x="80" y="20" class="step-hdr">Step t - 2</text>
    <rect x="10" y="35" width="140" height="48" class="box-highlight" />
    <text x="80" y="64" class="body-text" fill="#991b1b">State s_{t-2}</text>
    
    <path d="M 80 83 L 80 123" class="edge" marker-end="url(#arrow)" />
    
    <rect x="10" y="125" width="140" height="38" class="box-highlight" />
    <text x="80" y="149" class="mono-alert">Hash: 0xA19C</text>
  </g>

  <!-- Col 3: t-1 -->
  <g transform="translate(500, 80)">
    <text x="80" y="20" class="step-hdr">Step t - 1</text>
    <rect x="10" y="35" width="140" height="48" class="box" />
    <text x="80" y="64" class="body-text">State s_{t-1}</text>
    
    <path d="M 80 83 L 80 123" class="edge" marker-end="url(#arrow)" />
    
    <rect x="10" y="125" width="140" height="38" class="box" />
    <text x="80" y="149" class="mono-text">Hash: 0xC3E2</text>
  </g>

  <!-- Col 4: t (Cycle Duplicate) -->
  <g transform="translate(720, 80)">
    <text x="80" y="20" class="step-hdr">Step t (Current)</text>
    <rect x="10" y="35" width="140" height="48" class="box-highlight" />
    <text x="80" y="64" class="body-text" fill="#991b1b">State s_{t}</text>
    
    <path d="M 80 83 L 80 123" class="edge" marker-end="url(#arrow)" />
    
    <rect x="10" y="125" width="140" height="38" class="box-highlight" />
    <text x="80" y="149" class="mono-alert">Hash: 0xA19C</text>
  </g>

  <!-- Horizontal progression arrows -->
  <path d="M 210 138 L 274 138" class="edge" marker-end="url(#arrow)" />
  <path d="M 430 138 L 494 138" class="edge" marker-end="url(#arrow)" />
  <path d="M 650 138 L 714 138" class="edge" marker-end="url(#arrow)" />

  <!-- Match Arc curving down -->
  <path d="M 360 245 C 360 300 800 300 800 251" class="edge-alert" marker-end="url(#arrow-red)" />
  
  <rect x="430" y="295" width="280" height="48" class="pill-alert" />
  <text x="570" y="315" class="mono-alert">HASH COLLISION: h(s_t) == h(s_{t-2})</text>
  <text x="570" y="331" class="annot-text" text-anchor="middle" fill="#dc2626">Environment reverted without forward progress (Orbit = 2)</text>

  <!-- Supervisor Action Box -->
  <g transform="translate(70, 375)">
    <rect width="800" height="78" fill="#f8fafc" stroke="#334155" stroke-width="1.5" rx="6" />
    <text x="24" y="26" class="policy-title">Supervisory Watchdog Intervention Policy:</text>
    <text x="24" y="46" class="policy-body">• Iteration 1–2: Inject synthetic alert into context: [STATE_OSCILLATION_DETECTED: State hash matches step t-2]</text>
    <text x="24" y="64" class="policy-body">• Iteration &gt; 2: Halt forward execution, freeze environment sandbox, trigger Saga backward compensation or escalation.</text>
  </g>

</svg>
"""

with open("books/vol3/11_failure_recovery/images/svg/semantic-watchdog-oscillation.svg", "w") as f:
    f.write(watchdog_svg)

# 3. recovery-decision-tree.svg
decision_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 580" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .box { fill: #f8fafc; stroke: #334155; stroke-width: 1.5; rx: 6; ry: 6; }
      .box-gate { fill: #eff6ff; stroke: #2563eb; stroke-width: 1.8; rx: 6; ry: 6; }
      .box-term-green { fill: #f0fdf4; stroke: #16a34a; stroke-width: 2; rx: 6; ry: 6; }
      .box-term-amber { fill: #fffbeb; stroke: #d97706; stroke-width: 2; rx: 6; ry: 6; }
      .box-term-red { fill: #fef2f2; stroke: #dc2626; stroke-width: 2; rx: 6; ry: 6; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .gate-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; font-weight: 700; fill: #1e40af; text-anchor: middle; }
      .term-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13.5px; font-weight: 700; text-anchor: middle; }
      .body-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #334155; text-anchor: middle; }
      .edge { stroke: #64748b; stroke-width: 1.6; fill: none; }
      .edge-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #1e293b; }
      .code-mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10.5px; fill: #0f172a; text-anchor: middle; }
      .pill { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; rx: 4; ry: 4; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">Runtime Recovery Decision Architecture</text>
  <text x="480" y="50" class="sub-text">Multi-Tier Supervisory Arbitration Between Forward Self-Healing, Saga Compensation, and Human Escalation</text>

  <!-- Root: Execution Fault Intercepted -->
  <g transform="translate(360, 75)">
    <rect width="240" height="52" class="box" />
    <text x="120" y="23" class="gate-title" fill="#0f172a">Fault Intercepted: Ω_t</text>
    <text x="120" y="41" class="code-mono">(s_t, T_t, o_t^err, B_t, Σ_t)</text>
  </g>

  <!-- Downward arrow to Gate 1 -->
  <path d="M 480 127 L 480 155" class="edge" marker-end="url(#arrow)" />

  <!-- Gate 1: Error Taxonomy -->
  <g transform="translate(340, 157)">
    <rect width="280" height="56" class="box-gate" />
    <text x="140" y="24" class="gate-title">Gate 1: Error Taxonomy Signature</text>
    <text x="140" y="43" class="body-text">Classify error payload o_t^err</text>
  </g>

  <!-- Branch 1A: Transient Fault -> Retry -->
  <path d="M 340 185 L 180 185 L 180 238" class="edge" marker-end="url(#arrow)" />
  <rect x="195" y="162" width="130" height="20" class="pill" />
  <text x="260" y="176" class="edge-label" text-anchor="middle">Transient (429, Timeout)</text>

  <g transform="translate(70, 240)">
    <rect width="220" height="74" class="box-term-green" />
    <text x="110" y="24" class="term-title" fill="#166534">TRANSIENT RETRY</text>
    <text x="110" y="43" class="body-text">Exponential backoff + jitter</text>
    <text x="110" y="60" class="code-mono">t_wait = 2^k * t_base ± δ</text>
  </g>

  <!-- Branch 1B: Fatal Invariant -> Direct to Escalate (drops into center of OPERATOR_ESCALATE at x=780) -->
  <path d="M 620 185 L 780 185 L 780 438" class="edge" marker-end="url(#arrow)" />
  <rect x="635" y="162" width="130" height="20" class="pill" />
  <text x="700" y="176" class="edge-label" text-anchor="middle">Fatal (Auth, OOM, Sec)</text>

  <!-- Branch 1C: Deterministic Semantic -> Gate 2 -->
  <path d="M 480 213 L 480 252" class="edge" marker-end="url(#arrow)" />
  <rect x="495" y="222" width="145" height="20" class="pill" />
  <text x="567" y="236" class="edge-label" text-anchor="middle">Deterministic (Syntax, Schema)</text>

  <!-- Gate 2: Resource Budget Headroom -->
  <g transform="translate(340, 254)">
    <rect width="280" height="56" class="box-gate" />
    <text x="140" y="24" class="gate-title">Gate 2: Resource Budget Headroom</text>
    <text x="140" y="43" class="body-text">Tokens, steps, and dollar limits: B_t</text>
  </g>

  <!-- Branch 2A: Budget Exhausted -> Join Escalate Line at x=780 -->
  <path d="M 620 282 L 780 282" class="edge" />
  <rect x="640" y="260" width="115" height="20" class="pill" />
  <text x="697" y="274" class="edge-label" text-anchor="middle">Budget Exhausted</text>

  <!-- Branch 2B: Budget OK -> Gate 3 -->
  <path d="M 480 310 L 480 348" class="edge" marker-end="url(#arrow)" />
  <rect x="495" y="318" width="75" height="20" class="pill" />
  <text x="532" y="332" class="edge-label" text-anchor="middle">Budget OK</text>

  <!-- Gate 3: Amortization & Invariants -->
  <g transform="translate(340, 350)">
    <rect width="280" height="56" class="box-gate" />
    <text x="140" y="24" class="gate-title">Gate 3: Economic Amortization &amp; Safety</text>
    <text x="140" y="43" class="body-text">Repair cost vs. Sunk cost + Reversibility</text>
  </g>

  <!-- Terminal 1: FORWARD REPAIR -->
  <path d="M 340 378 L 180 378 L 180 438" class="edge" marker-end="url(#arrow)" />
  <rect x="195" y="356" width="130" height="20" class="pill" />
  <text x="260" y="370" class="edge-label" text-anchor="middle">Cost(Repair) &lt;&lt; Unwind</text>

  <g transform="translate(70, 440)">
    <rect width="220" height="90" class="box-term-green" />
    <text x="110" y="24" class="term-title" fill="#166534">FORWARD_REPAIR</text>
    <text x="110" y="43" class="body-text">• Synthesize self-healing prompt</text>
    <text x="110" y="60" class="body-text">• Ingest error trace in context</text>
    <text x="110" y="77" class="code-mono">a_repair ~ P(• | s_t, o_t^err)</text>
  </g>

  <!-- Terminal 2: BACKWARD ROLLBACK (SAGA) -->
  <path d="M 480 406 L 480 438" class="edge" marker-end="url(#arrow)" />
  <rect x="495" y="412" width="90" height="20" class="pill" />
  <text x="540" y="426" class="edge-label" text-anchor="middle">Unwind Viable</text>

  <g transform="translate(370, 440)">
    <rect width="220" height="90" class="box-term-amber" />
    <text x="110" y="24" class="term-title" fill="#b45309">BACKWARD_ROLLBACK</text>
    <text x="110" y="43" class="body-text">• Execute compensating handlers</text>
    <text x="110" y="60" class="body-text">• Reverse LIFO unwinding: C_k → C_1</text>
    <text x="110" y="77" class="code-mono">for c in reversed(Σ): c()</text>
  </g>

  <!-- Terminal 3: OPERATOR ESCALATE (centered at x=780) -->
  <g transform="translate(670, 440)">
    <rect width="220" height="90" class="box-term-red" />
    <text x="110" y="24" class="term-title" fill="#b91c1c">OPERATOR_ESCALATE</text>
    <text x="110" y="43" class="body-text">• Irreversible pivot crossed</text>
    <text x="110" y="60" class="body-text">• Compensator execution failed</text>
    <text x="110" y="77" class="code-mono">freeze() &amp; notify_human()</text>
  </g>

</svg>
"""

with open("books/vol3/11_failure_recovery/images/svg/recovery-decision-tree.svg", "w") as f:
    f.write(decision_svg)

# 4. fig-vol3-fault-tolerant-synthesis.svg
synthesis_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 560" width="100%" height="100%">
  <defs>
    <style>
      .bg { fill: #ffffff; }
      .participant { fill: #f8fafc; stroke: #334155; stroke-width: 1.8; rx: 6; ry: 6; }
      .header-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .lifeline { stroke: #cbd5e1; stroke-width: 1.5; stroke-dasharray: 4,4; }
      .title-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; fill: #0f172a; text-anchor: middle; }
      .sub-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #64748b; text-anchor: middle; }
      .msg-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; fill: #1e293b; }
      .msg-mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; fill: #0f172a; }
      .edge { stroke: #475569; stroke-width: 1.5; fill: none; }
      .edge-red { stroke: #dc2626; stroke-width: 1.8; fill: none; }
      .edge-amber { stroke: #d97706; stroke-width: 1.8; fill: none; }
      .edge-green { stroke: #16a34a; stroke-width: 1.5; fill: none; }
      .phase-bar { fill: #f1f5f9; stroke: #cbd5e1; stroke-width: 1; rx: 4; ry: 4; }
      .phase-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 600; fill: #475569; }
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#d97706" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#16a34a" />
    </marker>
  </defs>

  <rect width="100%" height="100%" class="bg" />

  <text x="480" y="30" class="title-text">Fault-Tolerant Trajectory Lifecycle &amp; Recovery Trace</text>
  <text x="480" y="50" class="sub-text">Coordinated Execution Sequence Across Control Plane, Write-Ahead Log, Saga Coordinator, and Sandbox</text>

  <!-- 4 Participant Columns -->
  <!-- Col 1: Agent Control Plane (x = 120) -->
  <rect x="30" y="70" width="180" height="42" class="participant" />
  <text x="120" y="96" class="header-text">1. Control Plane</text>
  <line x1="120" y1="112" x2="120" y2="530" class="lifeline" />

  <!-- Col 2: Write-Ahead Log (x = 360) -->
  <rect x="270" y="70" width="180" height="42" class="participant" />
  <text x="360" y="96" class="header-text">2. Write-Ahead Log</text>
  <line x1="360" y1="112" x2="360" y2="530" class="lifeline" />

  <!-- Col 3: Saga Coordinator (x = 600) -->
  <rect x="510" y="70" width="180" height="42" class="participant" />
  <text x="600" y="96" class="header-text">3. Saga Coordinator</text>
  <line x1="600" y1="112" x2="600" y2="530" class="lifeline" />

  <!-- Col 4: Execution Sandbox (x = 840) -->
  <rect x="750" y="70" width="180" height="42" class="participant" />
  <text x="840" y="96" class="header-text">4. Execution Sandbox</text>
  <line x1="840" y1="112" x2="840" y2="530" class="lifeline" />

  <!-- Phase 1: Forward Intent & Dispatch -->
  <rect x="30" y="125" width="900" height="20" class="phase-bar" />
  <text x="40" y="139" class="phase-text">PHASE 1: FORWARD SUB-TRANSACTION &amp; COMPENSATOR REGISTRATION (T₁)</text>

  <!-- Step 1: Write intent to WAL -->
  <path d="M 120 160 L 352 160" class="edge" marker-end="url(#arrow)" />
  <text x="236" y="154" class="msg-mono" text-anchor="middle">1. write_intent(T1, k1, fsync)</text>

  <!-- Step 2: Register compensator C1 -->
  <path d="M 360 178 L 592 178" class="edge" marker-end="url(#arrow)" />
  <text x="480" y="172" class="msg-mono" text-anchor="middle">2. register_compensator(C1)</text>

  <!-- Step 3: Dispatch to Sandbox -->
  <path d="M 120 200 L 832 200" class="edge" marker-end="url(#arrow)" />
  <text x="480" y="195" class="msg-text" text-anchor="middle">3. dispatch_tool("git checkout -b worktree-v1")</text>

  <!-- Step 4: Sandbox ACK -->
  <path d="M 840 220 L 128 220" class="edge-green" marker-end="url(#arrow-green)" />
  <text x="480" y="215" class="msg-mono" text-anchor="middle" fill="#166534">4. Exit 0: Branch created, workspace clean</text>

  <!-- Phase 2: Fault Injection & Cyclic Failure -->
  <rect x="30" y="245" width="900" height="20" class="phase-bar" />
  <text x="40" y="259" class="phase-text" fill="#dc2626">PHASE 2: FAULT INTERCEPTION &amp; WATCHDOG OSCILLATION (T₃ → T₄)</text>

  <!-- Step 5: Execute broken test -->
  <path d="M 120 280 L 832 280" class="edge" marker-end="url(#arrow)" />
  <text x="480" y="275" class="msg-mono" text-anchor="middle">5. dispatch_tool("pytest test_auth.py")</text>

  <!-- Step 6: Tool fails -->
  <path d="M 840 305 L 128 305" class="edge-red" marker-end="url(#arrow-red)" />
  <text x="480" y="300" class="msg-mono" text-anchor="middle" fill="#dc2626">6. Exit 1: SyntaxError in auth.py</text>

  <!-- Step 7: Cyclic retry emits identical state -->
  <path d="M 120 330 L 592 330" class="edge-red" marker-end="url(#arrow-red)" />
  <text x="360" y="324" class="msg-mono" text-anchor="middle" fill="#dc2626">7. check_watchdog(h(s_t)) == h(s_t-2)</text>

  <!-- Step 8: Watchdog interrupts -->
  <path d="M 600 355 L 128 355" class="edge-red" marker-end="url(#arrow-red)" />
  <text x="360" y="349" class="msg-mono" text-anchor="middle" fill="#dc2626">8. TRAP: OSCILLATION_DETECTED (Breaker Trips)</text>

  <!-- Phase 3: Saga Backward Compensation Unwind -->
  <rect x="30" y="380" width="900" height="20" class="phase-bar" />
  <text x="40" y="394" class="phase-text" fill="#b45309">PHASE 3: SAGA BACKWARD COMPENSATION UNWIND (C₂ → C₁)</text>

  <!-- Step 9: Trigger unwind -->
  <path d="M 120 415 L 592 415" class="edge-amber" marker-end="url(#arrow-amber)" />
  <text x="360" y="409" class="msg-text" text-anchor="middle" fill="#b45309">9. initiate_backward_rollback()</text>

  <!-- Step 10: Exec C2 -->
  <path d="M 600 440 L 832 440" class="edge-amber" marker-end="url(#arrow-amber)" />
  <text x="720" y="433" class="msg-mono" text-anchor="middle" fill="#b45309">10. exec_compensator(C2: git checkout auth.py)</text>

  <!-- Step 11: Exec C1 -->
  <path d="M 600 468 L 832 468" class="edge-amber" marker-end="url(#arrow-amber)" />
  <text x="720" y="461" class="msg-mono" text-anchor="middle" fill="#b45309">11. exec_compensator(C1: git worktree remove -f)</text>

  <!-- Step 12: Write ROLLED_BACK to WAL -->
  <path d="M 600 496 L 368 496" class="edge" marker-end="url(#arrow)" />
  <text x="480" y="490" class="msg-mono" text-anchor="middle">12. append_wal_entry([ROLLED_BACK], fsync)</text>

  <!-- Step 13: Report clean termination to Control Plane -->
  <path d="M 600 520 L 128 520" class="edge-green" marker-end="url(#arrow-green)" />
  <text x="360" y="514" class="msg-text" text-anchor="middle" fill="#166534">13. rollback_complete(status=CLEAN_ABORT)</text>

</svg>
"""

with open("books/vol3/11_failure_recovery/images/svg/fig-vol3-fault-tolerant-synthesis.svg", "w") as f:
    f.write(synthesis_svg)

print("Regenerated all 4 SVGs with clean layouts.")
