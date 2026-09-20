#!/usr/bin/env python3
"""
Generate publication-grade classic textbook SVGs for Chapter 18 (System / Architectural Synthesis).
Design Standard: Hennessy & Patterson / Saltzer & Kaashoek / CS:APP style.
- Clean rectangular functional blocks, clear dataflow, formal interfaces
- Muted textbook color palette (slate, navy, blue, emerald, amber, red)
- Proper XML escaping (&amp;, &lt;, &gt;)
- Zero overlapping text, clean translations per sub-box
"""

import os
import shutil

OUTPUT_DIR = "books/vol3/18_conclusion/images/svg"
CH18_SUBDIR = os.path.join(OUTPUT_DIR, "ch18")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CH18_SUBDIR, exist_ok=True)

def generate_synthesized_execution_pipeline():
    """Figure: Six-Stage Deterministic Mediation Harness (Trajectory Execution Pipeline)."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 640" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 18px; fill: #0f172a; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #475569; }
      .section-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 13px; fill: #0f172a; }
      .label-bold { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 600; font-size: 12px; fill: #1e293b; }
      .text-regular { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #334155; }
      .text-code { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-size: 10.5px; fill: #0f172a; }
      .metric-tag { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 10px; fill: #2563eb; }
      .invariant-tag { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 10.5px; fill: #059669; }
      .box-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.2; }
      .stage-hdr-bg { fill: #f1f5f9; stroke: #cbd5e1; stroke-width: 1; }
      .stage-sub-bg { fill: #f8fafc; stroke: #e2e8f0; stroke-width: 1; }
      .arrow { stroke: #2563eb; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead); }
      .arrow-sec { stroke: #64748b; stroke-width: 1.5; fill: none; marker-end: url(#arrowhead-sec); }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2563eb" />
    </marker>
    <marker id="arrowhead-sec" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#64748b" />
    </marker>
  </defs>

  <!-- Canvas Background -->
  <rect width="1200" height="640" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Main Title Block -->
  <g transform="translate(30, 36)">
    <text class="title" x="0" y="0">THE SYNTHESIZED TRAJECTORY EXECUTION PIPELINE</text>
    <text class="subtitle" x="0" y="20">Six-Stage Deterministic Mediation Harness Between Unprivileged Model Proposals and External Systems</text>
  </g>

  <!-- Top Ingress Bar -->
  <g transform="translate(30, 72)">
    <rect width="1140" height="34" rx="4" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1"/>
    <text class="label-bold" x="16" y="22">UNPRIVILEGED INPUT:</text>
    <text class="text-code" x="165" y="22">Prompt c_t ~ [Task Contract &lt;G, E, A, O, K&gt; | Working Memory | Prior Observations]</text>
    <text class="metric-tag" x="980" y="22">Authority A = 0</text>
  </g>

  <!-- 6 Pipelined Stages Layout: 2 rows of 3 stages -->
  <!-- ROW 1: Stages 1, 2, 3 -->
  <!-- Stage 1 -->
  <g transform="translate(30, 126)">
    <rect class="box-bg" width="360" height="210" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 30 L 0 30 Z" class="stage-hdr-bg"/>
    <text class="section-hdr" x="14" y="20">STAGE 1: MODEL PROPOSAL &amp; GRAMMAR MASK</text>
    <g transform="translate(12, 42)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Logit-Level Pushdown Automaton</text>
      <text class="text-regular" x="10" y="34">Applies bitmask M_t to logits prior to softmax</text>
      <text class="metric-tag" x="10" y="46">Zero Syntax Errors: Output in L(G)</text>
    </g>
    <g transform="translate(12, 102)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Deterministic Action Frame Generation</text>
      <text class="text-code" x="10" y="34">Emits: {tool: "fs_write", path: "src/lock.c"}</text>
      <text class="text-regular" x="10" y="46">Held in unprivileged memory escrow</text>
    </g>
    <text class="invariant-tag" x="14" y="194">&#x2713; Invariant: Schema conformance verified at decode</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="390" y1="231" x2="418" y2="231" class="arrow"/>

  <!-- Stage 2 -->
  <g transform="translate(420, 126)">
    <rect class="box-bg" width="360" height="210" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 30 L 0 30 Z" class="stage-hdr-bg"/>
    <text class="section-hdr" x="14" y="20">STAGE 2: CAPABILITY &amp; AUTHORITY GATE</text>
    <g transform="translate(12, 42)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Cryptographic Capability Token Check</text>
      <text class="text-regular" x="10" y="34">Validates macaroons / HMAC grants against tool ID</text>
      <text class="metric-tag" x="10" y="46">Attenuated Permissions: Read/Write Mask</text>
    </g>
    <g transform="translate(12, 102)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Budget &amp; Escrow Reservation Check</text>
      <text class="text-code" x="10" y="34">Assert: C(t) + R_turn &lt;= B_task (Monotonic Ledger)</text>
      <text class="text-regular" x="10" y="46">Rejects action if balance depleted</text>
    </g>
    <text class="invariant-tag" x="14" y="194">&#x2713; Invariant: Complete Mediation Invariant (Saltzer)</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="780" y1="231" x2="808" y2="231" class="arrow"/>

  <!-- Stage 3 -->
  <g transform="translate(810, 126)">
    <rect class="box-bg" width="360" height="210" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 30 L 0 30 Z" class="stage-hdr-bg"/>
    <text class="section-hdr" x="14" y="20">STAGE 3: WRITE-AHEAD INTENT LOG (WAL)</text>
    <g transform="translate(12, 42)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Synchronous Pre-Execution fsync</text>
      <text class="text-regular" x="10" y="34">Appends intent record to disk before network dispatch</text>
      <text class="metric-tag" x="10" y="46">Monotonic LSN: LSN_(t+1) = LSN_t + 1</text>
    </g>
    <g transform="translate(12, 102)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Crash-Consistent Trajectory State</text>
      <text class="text-code" x="10" y="34">Payload: &lt;LSN, turn_t, action_id, args, prev_hash&gt;</text>
      <text class="text-regular" x="10" y="46">Enables exact point-in-time replay</text>
    </g>
    <text class="invariant-tag" x="14" y="194">&#x2713; Invariant: Persistence-Before-Side-Effect Invariant</text>
  </g>

  <!-- Connecting Snake Arrow: Stage 3 (right) down to Stage 4 (left) -->
  <path d="M 990 336 L 990 356 L 210 356 L 210 374" class="arrow"/>

  <!-- ROW 2: Stages 4, 5, 6 -->
  <!-- Stage 4 -->
  <g transform="translate(30, 380)">
    <rect class="box-bg" width="360" height="210" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 30 L 0 30 Z" class="stage-hdr-bg"/>
    <text class="section-hdr" x="14" y="20">STAGE 4: ISOLATED SANDBOX ACTUATION</text>
    <g transform="translate(12, 42)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Hypervisor / MicroVM Containment</text>
      <text class="text-regular" x="10" y="34">Firecracker microVM / gVisor with seccomp-bpf filter</text>
      <text class="metric-tag" x="10" y="46">Network: Dedicated netns | Rootfs: Read-Only Overlay</text>
    </g>
    <g transform="translate(12, 102)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Deterministic Execution Quotas</text>
      <text class="text-code" x="10" y="34">Enforces: T_tool &lt;= 30s, RAM &lt;= 4GB, CPU &lt;= 2 cores</text>
      <text class="text-regular" x="10" y="46">SIGKILL emitted on quota overrun</text>
    </g>
    <text class="invariant-tag" x="14" y="194">&#x2713; Invariant: Zero Host Privilege Escalation</text>
  </g>

  <!-- Arrow 4 -> 5 -->
  <line x1="390" y1="485" x2="418" y2="485" class="arrow"/>

  <!-- Stage 5 -->
  <g transform="translate(420, 380)">
    <rect class="box-bg" width="360" height="210" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 30 L 0 30 Z" class="stage-hdr-bg"/>
    <text class="section-hdr" x="14" y="20">STAGE 5: OBSERVATION SANITIZATION</text>
    <g transform="translate(12, 42)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Payload Truncation &amp; ANSI Stripping</text>
      <text class="text-regular" x="10" y="34">Clamps stdout/stderr to S_obs &lt;= K_max tokens</text>
      <text class="metric-tag" x="10" y="46">Context Protection: Prevents OOM &amp; Buffer Churn</text>
    </g>
    <g transform="translate(12, 102)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">W &#x2295; X Prompt Injection Firewall</text>
      <text class="text-code" x="10" y="34">Wraps raw output in typed status envelope: [OK | ERR]</text>
      <text class="text-regular" x="10" y="46">External text never treated as executable prompt</text>
    </g>
    <text class="invariant-tag" x="14" y="194">&#x2713; Invariant: W &#x2295; X Protection Barrier</text>
  </g>

  <!-- Arrow 5 -> 6 -->
  <line x1="780" y1="485" x2="808" y2="485" class="arrow"/>

  <!-- Stage 6 -->
  <g transform="translate(810, 380)">
    <rect class="box-bg" width="360" height="210" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 30 L 0 30 Z" class="stage-hdr-bg"/>
    <text class="section-hdr" x="14" y="20">STAGE 6: COMPENSATING SAGA REGISTRATION</text>
    <g transform="translate(12, 42)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Dual Forward &amp; Compensating Ledger</text>
      <text class="text-regular" x="10" y="34">Every forward action A_i registers inverse action C_i</text>
      <text class="metric-tag" x="10" y="46">Example: fs_write(X) -&gt; fs_restore(backup_X)</text>
    </g>
    <g transform="translate(12, 102)">
      <rect class="stage-sub-bg" width="336" height="52" rx="3"/>
      <text class="label-bold" x="10" y="18">Verified Transaction Commit / Rollback</text>
      <text class="text-code" x="10" y="34">If oracle succeeds: Commit; Else: Execute C_k..C_1</text>
      <text class="text-regular" x="10" y="46">Guarantees state consistency upon abort</text>
    </g>
    <text class="invariant-tag" x="14" y="194">&#x2713; Invariant: Semantic Atomicity Invariant (Garcia-Molina)</text>
  </g>

  <!-- Bottom Summary Strip -->
  <g transform="translate(30, 600)">
    <rect width="1140" height="26" rx="3" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
    <text class="text-code" x="16" y="17">GOVERNING INVARIANT: Unprivileged neural proposals cross deterministic hardware and logging boundaries before committing physical state mutations.</text>
  </g>
</svg>'''
    filepath = os.path.join(OUTPUT_DIR, "fig-vol3-synthesized-execution-pipeline.svg")
    if os.path.islink(filepath) or os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        f.write(svg)
    print("Generated fig-vol3-synthesized-execution-pipeline.svg")


def generate_safety_case_pyramid():
    """Figure: Multi-Tier Verification Pyramid and Safety Case Architecture."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 620" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 18px; fill: #0f172a; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #475569; }
      .section-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 13px; fill: #0f172a; }
      .label-bold { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 600; font-size: 12px; fill: #1e293b; }
      .text-regular { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #334155; }
      .text-code { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-size: 10px; fill: #0f172a; }
      .metric-tag { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 10px; fill: #2563eb; }
      .card-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.2; }
      .panel-bg { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; }
      .tier-top { fill: #fef2f2; stroke: #f87171; stroke-width: 1.2; }
      .tier-mid { fill: #fefce8; stroke: #facc15; stroke-width: 1.2; }
      .tier-base { fill: #f0fdf4; stroke: #4ade80; stroke-width: 1.2; }
      .arrow { stroke: #2563eb; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2563eb" />
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="620" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Title Block -->
  <g transform="translate(30, 36)">
    <text class="title" x="0" y="0">THE MULTI-TIER AGENT SAFETY CASE &amp; VERIFICATION PYRAMID</text>
    <text class="subtitle" x="0" y="20">Structuring Assurance: Claims Grounded by Mechanistic Arguments Backed by Three Progressive Evidence Tiers</text>
  </g>

  <!-- Left Panel: 3-Tier Verification Pyramid -->
  <g transform="translate(30, 75)">
    <rect class="panel-bg" width="560" height="520" rx="4"/>
    <text class="section-hdr" x="16" y="24">PANEL A: The Three-Tier Evidence Pyramid</text>

    <!-- Top Tier -->
    <g transform="translate(25, 46)">
      <rect class="tier-top" width="510" height="112" rx="4"/>
      <text class="label-bold" x="14" y="20" fill="#991b1b">TOP TIER: RUNTIME CONTAINMENT &amp; CANARY TELEMETRY</text>
      <text class="text-regular" x="14" y="38">&#x2022; Automated Hardware Circuit Breakers: Instant trip if burn velocity &gt; threshold</text>
      <text class="text-regular" x="14" y="54">&#x2022; Canary Deployment Rings: Progressive rollouts (1% -&gt; 5% -&gt; 100%) with SPRT stopping</text>
      <text class="text-regular" x="14" y="70">&#x2022; Distributed OpenTelemetry Tracing: Real-time span waterfall &amp; anomaly detection</text>
      <text class="metric-tag" x="14" y="94">Latency: &lt; 100 ms | Coverage: Dynamic In-Flight</text>
      <text class="text-code" x="380" y="94" fill="#991b1b">T_react &lt; 50 ms</text>
    </g>

    <!-- Arrow Top -> Mid -->
    <line x1="280" y1="162" x2="280" y2="182" class="arrow"/>

    <!-- Middle Tier -->
    <g transform="translate(25, 186)">
      <rect class="tier-mid" width="510" height="126" rx="4"/>
      <text class="label-bold" x="14" y="20" fill="#854d0e">MIDDLE TIER: STATISTICAL SANDBOX EVALUATION</text>
      <text class="text-regular" x="14" y="38">&#x2022; Hermetic Interactive Gyms: Synthetic &amp; historical benchmark replay</text>
      <text class="text-regular" x="14" y="54">&#x2022; Wilson Score Margin-of-Error Bounds: Rigorous sample dimensioning (N &gt;= 2,400)</text>
      <text class="text-regular" x="14" y="70">&#x2022; Compounding Trajectory Reliability: Empirical pass-rate modeling across H turns (p^H)</text>
      <text class="text-regular" x="14" y="86">&#x2022; Dirty Environmental Fixtures: Network jitter, partial disk crashes, malformed mocks</text>
      <text class="metric-tag" x="14" y="110">Latency: 10 s - 10 min | Statistical</text>
      <text class="text-code" x="340" y="110" fill="#854d0e">&#916;p &lt;= 2.0% (95% CI)</text>
    </g>

    <!-- Arrow Mid -> Base -->
    <line x1="280" y1="316" x2="280" y2="336" class="arrow"/>

    <!-- Base Tier -->
    <g transform="translate(25, 340)">
      <rect class="tier-base" width="510" height="136" rx="4"/>
      <text class="label-bold" x="14" y="20" fill="#166534">BASE TIER: DETERMINISTIC MECHANICAL VERIFIERS</text>
      <text class="text-regular" x="14" y="38">&#x2022; Abstract Syntax Tree (AST) Linters &amp; Compilers: Zero-tolerance syntax barriers</text>
      <text class="text-regular" x="14" y="54">&#x2022; Decode-Time Pushdown Automata: FSM logit masking enforces strict JSON schema</text>
      <text class="text-regular" x="14" y="70">&#x2022; OS Sandbox Capability Enclaves: seccomp-bpf syscall traps, unshare netns/pidns</text>
      <text class="text-regular" x="14" y="86">&#x2022; Static Taint Analysis: Dataflow boundary checks on prompt injection &amp; credentials</text>
      <text class="metric-tag" x="14" y="112">Latency: &lt; 1 ms | Formal Grammars</text>
      <text class="text-code" x="310" y="112" fill="#166534">Error Rate = 0 (Relative to G)</text>
    </g>
    <text class="text-code" x="25" y="500" fill="#475569">FOUNDATION: Non-learned mechanical filters intercept &gt; 95% of invalid states.</text>
  </g>

  <!-- Right Panel: The Formal Assurance Structure (Claims-Argument-Evidence) -->
  <g transform="translate(610, 75)">
    <rect class="panel-bg" width="560" height="520" rx="4"/>
    <text class="section-hdr" x="16" y="24">PANEL B: Formal Assurance Case Architecture (Claims &#x2190; Argument &#x2190; Evidence)</text>

    <!-- Case Study 1 -->
    <g transform="translate(18, 42)">
      <rect class="card-bg" width="524" height="142" rx="4"/>
      <path d="M 0 0 L 524 0 L 524 26 L 0 26 Z" fill="#eff6ff"/>
      <text class="label-bold" x="12" y="18" fill="#1e40af">ASSURANCE CLAIM 1: ZERO HOST PRIVILEGE ESCALATION</text>
      <text class="label-bold" x="12" y="44">Causal Mechanistic Argument:</text>
      <text class="text-regular" x="20" y="60">Neural agent proposals run inside microVMs with seccomp-bpf filters.</text>
      <text class="text-regular" x="20" y="74">Privileged syscalls (clone, ptrace, bpf) are dropped at the kernel boundary.</text>
      <text class="label-bold" x="12" y="94">Empirical Multi-Tier Evidence:</text>
      <text class="text-code" x="20" y="110">&#x2022; Base Tier: seccomp profile passes static syscall audit (0 violations).</text>
      <text class="text-code" x="20" y="126">&#x2022; Mid Tier: 10,000 automated breakout fuzzing tests show zero escapes.</text>
    </g>

    <!-- Case Study 2 -->
    <g transform="translate(18, 196)">
      <rect class="card-bg" width="524" height="142" rx="4"/>
      <path d="M 0 0 L 524 0 L 524 26 L 0 26 Z" fill="#eff6ff"/>
      <text class="label-bold" x="12" y="18" fill="#1e40af">ASSURANCE CLAIM 2: BOUNDED FINANCIAL EXPENDITURE (C_task &lt;= B_max)</text>
      <text class="label-bold" x="12" y="44">Causal Mechanistic Argument:</text>
      <text class="text-regular" x="20" y="60">Monotonic spending ledger wraps all inferences inside non-overlapping escrows.</text>
      <text class="text-regular" x="20" y="74">If free capital F_root &lt; min_cost, supervisor clamps with OutOfBudgetException.</text>
      <text class="label-bold" x="12" y="94">Empirical Multi-Tier Evidence:</text>
      <text class="text-code" x="20" y="110">&#x2022; Base Tier: Conservation invariant B = E + R + F holds across all trees.</text>
      <text class="text-code" x="20" y="126">&#x2022; Top Tier: Financial velocity circuit breaker trips at $2.00/min threshold.</text>
    </g>

    <!-- Case Study 3 -->
    <g transform="translate(18, 350)">
      <rect class="card-bg" width="524" height="142" rx="4"/>
      <path d="M 0 0 L 524 0 L 524 26 L 0 26 Z" fill="#eff6ff"/>
      <text class="label-bold" x="12" y="18" fill="#1e40af">ASSURANCE CLAIM 3: DETERMINISTIC RECOVERY UPON FAULT</text>
      <text class="label-bold" x="12" y="44">Causal Mechanistic Argument:</text>
      <text class="text-regular" x="20" y="60">Saga coordinator logs compensating rollback actions C_i to WAL before mutation.</text>
      <text class="text-regular" x="20" y="74">Upon unhandled tool error or crash, compensation executes in reverse order.</text>
      <text class="label-bold" x="12" y="94">Empirical Multi-Tier Evidence:</text>
      <text class="text-code" x="20" y="110">&#x2022; Mid Tier: Chaos injection confirms 100% clean rollback on 500 fault seeds.</text>
      <text class="text-code" x="20" y="126">&#x2022; Top Tier: Mean Time to Recovery (MTTR) &lt; 3.2 seconds in canary cluster.</text>
    </g>

    <text class="text-code" x="18" y="506" fill="#475569">ASSURANCE RULE: Claims without mechanical arguments and evidence are untrusted.</text>
  </g>
</svg>'''
    filepath = os.path.join(OUTPUT_DIR, "fig-vol3-safety-case-pyramid.svg")
    if os.path.islink(filepath) or os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        f.write(svg)
    print("Generated fig-vol3-safety-case-pyramid.svg")


def generate_digital_vs_physical_boundary():
    """Figure: Digital Trajectory Assumptions vs. Physical Embodied Constraints."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 620" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 18px; fill: #0f172a; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #475569; }
      .section-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 13px; fill: #0f172a; }
      .label-bold { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 600; font-size: 12px; fill: #1e293b; }
      .text-regular { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #334155; }
      .text-code { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-size: 10px; fill: #0f172a; }
      .metric-tag { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 10px; fill: #2563eb; }
      .panel-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.2; }
      .box-digital { fill: #f0fdf4; stroke: #86efac; stroke-width: 1; }
      .box-physical { fill: #fff1f2; stroke: #fca5a5; stroke-width: 1; }
      .card-bg { fill: #f8fafc; stroke: #e2e8f0; stroke-width: 1; }
      .arrow { stroke: #2563eb; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2563eb" />
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="620" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Title Block -->
  <g transform="translate(30, 36)">
    <text class="title" x="0" y="0">ARCHITECTURAL BOUNDARY: DIGITAL TRAJECTORIES VS. PHYSICAL EMBODIED AGENCY</text>
    <text class="subtitle" x="0" y="20">Transition from Resettable Software Sandboxes (Volume III) to Irreversible Thermodynamic Environments (Volume IV)</text>
  </g>

  <!-- Left Panel: Digital Trajectory Domain -->
  <g transform="translate(30, 75)">
    <rect class="panel-bg" width="560" height="520" rx="4"/>
    <path d="M 0 0 L 560 0 L 560 32 L 0 32 Z" fill="#ecfdf5"/>
    <text class="section-hdr" x="16" y="21" fill="#065f46">PANEL A: DIGITAL TRAJECTORY REGIME (VOLUME III)</text>
    <text class="metric-tag" x="420" y="21" fill="#047857">Reversible State Space</text>

    <!-- Sub-boxes -->
    <g transform="translate(18, 48)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">1. Thermodynamic State &amp; Invertibility</text>
      <text class="text-regular" x="12" y="38">&#x2022; Finite Reversal Time: T_reversal &lt; &#x221E; (Instantaneous software reset)</text>
      <text class="text-regular" x="12" y="54">&#x2022; Zero Physical Entropy: Erroneous tokens consume only electricity, leaving zero residue</text>
      <text class="text-code" x="12" y="70" fill="#047857">State Invariant: Speculative actions are 100% discardable</text>
    </g>

    <g transform="translate(18, 134)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">2. Environmental Containment Boundary</text>
      <text class="text-regular" x="12" y="38">&#x2022; Ephemeral Sandboxes: Firecracker microVMs, Docker containers, chroot namespaces</text>
      <text class="text-regular" x="12" y="54">&#x2022; Storage Isolation: Copy-on-Write (CoW) overlays, temporary Git worktree branches</text>
      <text class="text-code" x="12" y="70" fill="#047857">Complete Mediation: OS kernel blocks unauthorized host side-effects</text>
    </g>

    <g transform="translate(18, 220)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">3. Rollback &amp; Compensation Mechanisms</text>
      <text class="text-regular" x="12" y="38">&#x2022; Deterministic Compensation: Saga ledger executes inverse actions (C_k ... C_1)</text>
      <text class="text-regular" x="12" y="54">&#x2022; Snapshot Restoration: git reset --hard, ZFS snapshot rollback, database ROLLBACK</text>
      <text class="text-code" x="12" y="70" fill="#047857">Recovery Guarantee: Trajectory abort cleanly restores clean baseline</text>
    </g>

    <g transform="translate(18, 306)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">4. Verification Oracle &amp; Exploration Policy</text>
      <text class="text-regular" x="12" y="38">&#x2022; Fast Mechanical Oracles: Compilers, type-checkers, unit test suites (pytest, cargo test)</text>
      <text class="text-regular" x="12" y="54">&#x2022; Aggressive Trial-and-Error: Best-of-N sampling, Monte Carlo Tree Search, rollouts</text>
      <text class="text-code" x="12" y="70" fill="#047857">Safety Barrier: Software oracles evaluate candidate prior to final commit</text>
    </g>

    <g transform="translate(18, 392)">
      <rect class="box-digital" width="524" height="110" rx="3"/>
      <text class="label-bold" x="12" y="20" fill="#065f46">THE DIGITAL AGENT SYSTEMS CONTRACT</text>
      <text class="text-regular" x="12" y="38">&#x2022; Fault Model: Fail-Plausible (Model emits syntactically plausible but invalid code)</text>
      <text class="text-regular" x="12" y="54">&#x2022; Host Defense: Wrap model in strict type checking and sandbox test runners</text>
      <text class="text-regular" x="12" y="70">&#x2022; Primary Bottleneck: Context window drift, memory bus bandwidth, serving economics</text>
      <text class="text-code" x="12" y="94" fill="#065f46">VOLUME III FOCUS: Constructing the accountable digital software runtime.</text>
    </g>
  </g>

  <!-- Right Panel: Physical Embodied Domain -->
  <g transform="translate(610, 75)">
    <rect class="panel-bg" width="560" height="520" rx="4"/>
    <path d="M 0 0 L 560 0 L 560 32 L 0 32 Z" fill="#fff1f2"/>
    <text class="section-hdr" x="16" y="21" fill="#9f1239">PANEL B: PHYSICAL EMBODIED REGIME (VOLUME IV)</text>
    <text class="metric-tag" x="420" y="21" fill="#be123c">Irreversible State Space</text>

    <!-- Sub-boxes -->
    <g transform="translate(18, 48)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">1. Thermodynamic State &amp; Irreversibility</text>
      <text class="text-regular" x="12" y="38">&#x2022; Infinite Reversal Time: T_reversal = &#x221E; (Thermodynamics forbids entropy rollback)</text>
      <text class="text-regular" x="12" y="54">&#x2022; Physical Damage: Broken pipettes, shattered glass, chemical explosions, robotic collision</text>
      <text class="text-code" x="12" y="70" fill="#9f1239">Physical Reality: Once voltage reaches actuator, action is committed forever</text>
    </g>

    <g transform="translate(18, 134)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">2. Out-of-Band Hardware Interlock Boundary</text>
      <text class="text-regular" x="12" y="38">&#x2022; Hard Physical Interlocks: Optical light curtains, limit switches, current-limiting relays</text>
      <text class="text-regular" x="12" y="54">&#x2022; Isolated Microcontrollers: Formal safety state machines executed on verified PLCs</text>
      <text class="text-code" x="12" y="70" fill="#9f1239">Air-Gap Defense: Safety logic runs completely out-of-band of AI neural software</text>
    </g>

    <g transform="translate(18, 220)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">3. Absence of Post-Hoc Rollbacks</text>
      <text class="text-regular" x="12" y="38">&#x2022; No Undo Primitives: You cannot "un-aspirate" toxic acid or "un-drop" expensive samples</text>
      <text class="text-regular" x="12" y="54">&#x2022; Real-Time Emergency Stop: Hardware ESTOP power cut-off is the sole failure mitigation</text>
      <text class="text-code" x="12" y="70" fill="#9f1239">Recovery Reality: Failures require physical repair and decontamination</text>
    </g>

    <g transform="translate(18, 306)">
      <rect class="card-bg" width="524" height="76" rx="3"/>
      <text class="label-bold" x="12" y="20">4. Kinematic Safety Envelopes &amp; Clamping</text>
      <text class="text-regular" x="12" y="38">&#x2022; Deterministic Envelope Clamping: Neural trajectories clamped to safe velocity/force bounds</text>
      <text class="text-regular" x="12" y="54">&#x2022; Zero Trial-and-Error in Production: Exploration strictly confined to high-fidelity physics sims</text>
      <text class="text-code" x="12" y="70" fill="#9f1239">Safety Invariant: Drives check commands against invariant geometry</text>
    </g>

    <g transform="translate(18, 392)">
      <rect class="box-physical" width="524" height="110" rx="3"/>
      <text class="label-bold" x="12" y="20" fill="#9f1239">THE EMBODIED PHYSICAL AI CONTRACT</text>
      <text class="text-regular" x="12" y="38">&#x2022; Fault Model: Fail-Safe Emergency Stop (Drop power; lock mechanical brakes)</text>
      <text class="text-regular" x="12" y="54">&#x2022; System Defense: Independent PLC / FPGA safety circuit clamps model torques</text>
      <text class="text-regular" x="12" y="70">&#x2022; Primary Bottleneck: Sensorimotor latency jitter, physical wear, non-invertibility</text>
      <text class="text-code" x="12" y="94" fill="#9f1239">VOLUME IV PREVIEW: Transitioning from digital software to physical agency.</text>
    </g>
  </g>
</svg>'''
    filepath = os.path.join(OUTPUT_DIR, "vol3-digital-vs-physical-boundary.svg")
    if os.path.islink(filepath) or os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        f.write(svg)
    print("Generated vol3-digital-vs-physical-boundary.svg")


def generate_brooks_essential_complexity():
    """Figure: Accidental vs. Essential Complexity across Software 1.0, 2.0, and 3.0."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 620" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 18px; fill: #0f172a; }
      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #475569; }
      .section-hdr { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 700; font-size: 12.5px; fill: #0f172a; }
      .label-bold { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-weight: 600; font-size: 12px; fill: #1e293b; }
      .text-regular { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #334155; }
      .text-code { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-size: 10px; fill: #0f172a; }
      .metric-tag { font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-weight: 600; font-size: 9.5px; fill: #2563eb; }
      .box-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1.2; }
      .accidental-bar { fill: #fecaca; stroke: #f87171; stroke-width: 1; }
      .essential-bar { fill: #bfdbfe; stroke: #60a5fa; stroke-width: 1; }
      .card-sub { fill: #f8fafc; stroke: #e2e8f0; stroke-width: 1; }
      .arrow { stroke: #2563eb; stroke-width: 1.8; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2563eb" />
    </marker>
  </defs>

  <!-- Background -->
  <rect width="1200" height="620" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>

  <!-- Title Block -->
  <g transform="translate(30, 36)">
    <text class="title" x="0" y="0">EVOLUTION OF SOFTWARE COMPLEXITY ACROSS PARADIGMS</text>
    <text class="subtitle" x="0" y="20">Brooks' No Silver Bullet: Accidental Implementation Friction Compresses Toward Zero; Essential Invariant Specification Remains Invariant</text>
  </g>

  <!-- 3 Comparative Paradigm Columns -->
  <!-- Column 1: Software 1.0 -->
  <g transform="translate(30, 75)">
    <rect class="box-bg" width="360" height="490" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 36 L 0 36 Z" fill="#f1f5f9"/>
    <text class="section-hdr" x="14" y="22">SOFTWARE 1.0 (PROCEDURAL)</text>
    <text class="metric-tag" text-anchor="end" x="346" y="22">Manual Code</text>

    <!-- Visual Complexity Stack Bar -->
    <g transform="translate(20, 52)">
      <text class="label-bold" x="0" y="14">Effort Allocation Stack:</text>
      <!-- Accidental bar (80%) -->
      <rect x="0" y="24" width="320" height="96" rx="3" class="accidental-bar"/>
      <text class="label-bold" x="12" y="52" fill="#991b1b">Accidental Complexity: 80%</text>
      <text class="text-regular" x="12" y="70" fill="#7f1d1d">&#x2022; Syntax formatting, compiler errors</text>
      <text class="text-regular" x="12" y="86" fill="#7f1d1d">&#x2022; Manual memory management (malloc/free)</text>
      <text class="text-regular" x="12" y="102" fill="#7f1d1d">&#x2022; Serialization, boilerplate, build configs</text>

      <!-- Essential bar (20%) -->
      <rect x="0" y="126" width="320" height="36" rx="3" class="essential-bar"/>
      <text class="label-bold" x="12" y="148" fill="#1e40af">Essential Complexity: 20%</text>
    </g>

    <!-- Description Details -->
    <g transform="translate(20, 232)">
      <rect class="card-sub" width="320" height="110" rx="3"/>
      <text class="label-bold" x="10" y="18">The Engineer's Daily Burden:</text>
      <text class="text-regular" x="10" y="36">Engineers spend the majority of cognitive labor</text>
      <text class="text-regular" x="10" y="52">translating mental models into rigid syntax,</text>
      <text class="text-regular" x="10" y="68">debugging segfaults, plumbing networking RPCs,</text>
      <text class="text-regular" x="10" y="84">and fighting compiler type-checkers.</text>
      <text class="text-code" x="10" y="102" fill="#475569">Productivity: 10-50 lines of tested code/day</text>
    </g>

    <g transform="translate(20, 354)">
      <rect class="card-sub" width="320" height="110" rx="3"/>
      <text class="label-bold" x="10" y="18">Primary Failure Mode:</text>
      <text class="text-regular" x="10" y="36">&#x2022; Buffer overflows &amp; memory corruption</text>
      <text class="text-regular" x="10" y="52">&#x2022; Race conditions &amp; distributed deadlocks</text>
      <text class="text-regular" x="10" y="68">&#x2022; Brittle monolithic dependencies</text>
      <text class="text-code" x="10" y="96" fill="#dc2626">Fault Model: Fail-Stop (Crash)</text>
    </g>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="390" y1="320" x2="418" y2="320" class="arrow"/>

  <!-- Column 2: Software 2.0 -->
  <g transform="translate(420, 75)">
    <rect class="box-bg" width="360" height="490" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 36 L 0 36 Z" fill="#f1f5f9"/>
    <text class="section-hdr" x="14" y="22">SOFTWARE 2.0 (MODELS)</text>
    <text class="metric-tag" text-anchor="end" x="346" y="22">Learned Weights</text>

    <!-- Visual Complexity Stack Bar -->
    <g transform="translate(20, 52)">
      <text class="label-bold" x="0" y="14">Effort Allocation Stack:</text>
      <!-- Accidental bar (45%) -->
      <rect x="0" y="24" width="320" height="60" rx="3" class="accidental-bar"/>
      <text class="label-bold" x="12" y="46" fill="#991b1b">Accidental Complexity: 45%</text>
      <text class="text-regular" x="12" y="62" fill="#7f1d1d">&#x2022; Feature pipelines, CUDA tensor alignment</text>
      <text class="text-regular" x="12" y="76" fill="#7f1d1d">&#x2022; Hyperparameter tuning, GPU clustering</text>

      <!-- Essential bar (55%) -->
      <rect x="0" y="90" width="320" height="72" rx="3" class="essential-bar"/>
      <text class="label-bold" x="12" y="112" fill="#1e40af">Essential Complexity: 55%</text>
      <text class="text-regular" x="12" y="128" fill="#1e3a8a">&#x2022; Objective loss formulation (MSE, CE)</text>
      <text class="text-regular" x="12" y="144" fill="#1e3a8a">&#x2022; Dataset curation &amp; evaluation schema</text>
    </g>

    <!-- Description Details -->
    <g transform="translate(20, 232)">
      <rect class="card-sub" width="320" height="110" rx="3"/>
      <text class="label-bold" x="10" y="18">The Engineer's Shift:</text>
      <text class="text-regular" x="10" y="36">Replaces explicit rule heuristics with neural</text>
      <text class="text-regular" x="10" y="52">approximators across perceptual tasks.</text>
      <text class="text-regular" x="10" y="68">Eliminates manual feature engineering,</text>
      <text class="text-regular" x="10" y="84">introducing dataset curation &amp; training loops.</text>
      <text class="text-code" x="10" y="102" fill="#475569">Unit: Parameter Weights &amp; Tensors</text>
    </g>

    <g transform="translate(20, 354)">
      <rect class="card-sub" width="320" height="110" rx="3"/>
      <text class="label-bold" x="10" y="18">Primary Failure Mode:</text>
      <text class="text-regular" x="10" y="36">&#x2022; Distribution shift &amp; out-of-distribution drift</text>
      <text class="text-regular" x="10" y="52">&#x2022; Gradient explosion &amp; loss collapse</text>
      <text class="text-regular" x="10" y="68">&#x2022; Adversarial label perturbation</text>
      <text class="text-code" x="10" y="96" fill="#d97706">Fault Model: Statistical Degradation</text>
    </g>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="780" y1="320" x2="808" y2="320" class="arrow"/>

  <!-- Column 3: Software 3.0 -->
  <g transform="translate(810, 75)">
    <rect class="box-bg" width="360" height="490" rx="4"/>
    <path d="M 0 0 L 360 0 L 360 36 L 0 36 Z" fill="#eff6ff"/>
    <text class="section-hdr" x="14" y="22" fill="#1d4ed8">SOFTWARE 3.0 (AGENTS)</text>
    <text class="metric-tag" text-anchor="end" x="346" y="22" fill="#1e40af">Autonomous Loops</text>

    <!-- Visual Complexity Stack Bar -->
    <g transform="translate(20, 52)">
      <text class="label-bold" x="0" y="14">Effort Allocation Stack:</text>
      <!-- Accidental bar (5%) -->
      <rect x="0" y="24" width="320" height="18" rx="3" class="accidental-bar"/>
      <text class="label-bold" x="8" y="37" fill="#991b1b" font-size="10px">Accidental: &lt; 5% (Automated)</text>

      <!-- Essential bar (95%) -->
      <rect x="0" y="48" width="320" height="114" rx="3" class="essential-bar"/>
      <text class="label-bold" x="12" y="70" fill="#1e40af">Essential Complexity: &gt; 95% (Human Core)</text>
      <text class="text-regular" x="12" y="88" fill="#1e3a8a">&#x2022; Formal invariant &amp; contract formulation</text>
      <text class="text-regular" x="12" y="104" fill="#1e3a8a">&#x2022; Threat modeling &amp; capability boundaries</text>
      <text class="text-regular" x="12" y="120" fill="#1e3a8a">&#x2022; Verification harnesses &amp; deterministic oracles</text>
      <text class="text-regular" x="12" y="136" fill="#1e3a8a">&#x2022; Financial budget ceilings &amp; circuit breakers</text>
      <text class="text-regular" x="12" y="152" fill="#1e3a8a">&#x2022; Systemic architectural decomposition</text>
    </g>

    <!-- Description Details -->
    <g transform="translate(20, 232)">
      <rect class="card-sub" width="320" height="110" rx="3"/>
      <text class="label-bold" x="10" y="18">The Engineer as System Architect:</text>
      <text class="text-regular" x="10" y="36">Agents generate syntax, tests, diffs, and glue code</text>
      <text class="text-regular" x="10" y="52">in milliseconds. The human engineer is freed from</text>
      <text class="text-regular" x="10" y="68">accidental typing to focus entirely on specifying</text>
      <text class="text-regular" x="10" y="84">system boundaries, invariants, and safety cases.</text>
      <text class="text-code" x="10" y="102" fill="#1d4ed8">Role: System Architect &amp; Verifier</text>
    </g>

    <g transform="translate(20, 354)">
      <rect class="card-sub" width="320" height="110" rx="3"/>
      <text class="label-bold" x="10" y="18">Primary Failure Mode:</text>
      <text class="text-regular" x="10" y="36">&#x2022; Semantic divergence across multi-turn loops</text>
      <text class="text-regular" x="10" y="52">&#x2022; Silent regression past unverified boundaries</text>
      <text class="text-regular" x="10" y="68">&#x2022; Runaway budget exhaustion &amp; deadlocks</text>
      <text class="text-code" x="10" y="96" fill="#b91c1c">Fault Model: Fail-Plausible (Stochastic)</text>
    </g>
  </g>

  <!-- Bottom Synthesis Banner -->
  <g transform="translate(30, 580)">
    <rect width="1140" height="28" rx="3" fill="#eff6ff" stroke="#93c5fd" stroke-width="1"/>
    <text class="label-bold" x="16" y="18" fill="#1e40af">THE LAW OF ACCIDENTAL ACCELERATION:</text>
    <text class="text-regular" x="305" y="18">Agents compress accidental coding latency toward zero; the essential complexity of architectural modeling and invariant specification remains irreducible.</text>
  </g>
</svg>'''
    filepath = os.path.join(CH18_SUBDIR, "brooks_essential_complexity.svg")
    if os.path.islink(filepath) or os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w") as f:
        f.write(svg)
    print("Generated ch18/brooks_essential_complexity.svg")


def fix_ladder_symlink():
    """Ensure fig-vol3-intervention-ladder-flowchart.svg is a solid file copied from autonomous_improvement_ladder_v2.svg."""
    src = os.path.join(OUTPUT_DIR, "autonomous_improvement_ladder_v2.svg")
    dst = os.path.join(OUTPUT_DIR, "fig-vol3-intervention-ladder-flowchart.svg")
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    shutil.copyfile(src, dst)
    print("Updated fig-vol3-intervention-ladder-flowchart.svg from autonomous_improvement_ladder_v2.svg")


if __name__ == "__main__":
    generate_synthesized_execution_pipeline()
    generate_safety_case_pyramid()
    generate_digital_vs_physical_boundary()
    generate_brooks_essential_complexity()
    fix_ladder_symlink()
    print("All Chapter 18 SVGs successfully generated and finalized.")
