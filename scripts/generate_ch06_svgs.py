#!/usr/bin/env python3
"""
Generate publication-grade SVGs for Chapter 06 (Episodic & Persistent Memory):
1. vol3-lexical-vs-ast.svg: Lexical BM25 Vocabulary Mismatch & AST Structural Blindness
2. vol3_ch06_hnsw_rrf.svg: Multi-Stage Hybrid Retrieval & Reciprocal Rank Fusion Pipeline
3. cpg-multi-hop-traversal.svg: Code Property Graph & Seeded Neighborhood Expansion
4. retrieval_access_control_prefilter.svg: Access-Controlled Storage & Bitset Pre-Filtering
5. distractor_dilemma_success_curve.svg: The Distractor Dilemma (Task Success vs Retrieval Depth k)
"""

import os
import subprocess

from pathlib import Path
TARGET_DIR = str(Path(__file__).resolve().parent.parent / "books/vol3/06_long_term_memory/images/svg")
os.makedirs(TARGET_DIR, exist_ok=True)

# 1. vol3-lexical-vs-ast.svg
svg_lexical_vs_ast = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 460" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 15px; fill: #0f172a; }
      .sub-note { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #64748b; }
      .panel-title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 12px; }
      .box-lbl { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 11px; }
      .box-sub { font-family: system-ui, -apple-system, sans-serif; font-size: 10px; fill: #475569; }
      .mono { font-family: ui-monospace, monospace; font-size: 10.5px; }
      .card-text { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #334155; }
    </style>
    <marker id="arr-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626"/>
    </marker>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
  </defs>

  <rect width="1080" height="460" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- Top Banner -->
  <rect x="20" y="16" width="1040" height="40" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="41" class="title">VOCABULARY MISMATCH AND STRUCTURAL BLINDNESS PATHOLOGIES</text>
  <text x="1045" y="41" class="sub-note" text-anchor="end">Unimodal Retrieval Failure Modes Under Unassisted Agent Queries</text>

  <!-- Query Box -->
  <rect x="200" y="70" width="680" height="42" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="215" y="87" class="box-lbl" fill="#1e40af">Agent Query:</text>
  <text x="300" y="87" class="mono" font-weight="700" fill="#1e3a8a">"find socket connection rate limiter algorithm"</text>
  <text x="215" y="103" class="box-sub">Intent: Locate traffic shaping primitive controlling ingress packet flow</text>

  <!-- Split Connectors -->
  <path d="M 400 112 L 280 148" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr-blue)"/>
  <path d="M 680 112 L 800 148" fill="none" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr-blue)"/>

  <!-- Left Panel: Lexical BM25 -->
  <g transform="translate(45, 150)">
    <rect width="475" height="215" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>
    <rect x="15" y="12" width="445" height="28" rx="4" fill="#ffffff" stroke="#fecaca"/>
    <text x="25" y="31" class="panel-title" fill="#991b1b">PATHOLOGY 1: LEXICAL BM25 VOCABULARY MISMATCH</text>

    <text x="20" y="62" class="box-lbl" fill="#0f172a">Parsed Query Postings:</text>
    <text x="175" y="62" class="mono" fill="#b91c1c">{socket, connection, rate, limiter}</text>

    <rect x="20" y="76" width="435" height="52" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="93" class="box-lbl" fill="#334155">Target Implementation (C++):</text>
    <text x="30" y="112" class="mono" fill="#0f172a">struct TokenBucketDrain { uint64_t drain_rate; ... };</text>

    <path d="M 237 132 L 237 155" fill="none" stroke="#dc2626" stroke-width="1.5" marker-end="url(#arr-red)"/>

    <rect x="20" y="160" width="435" height="42" rx="4" fill="#fee2e2" stroke="#ef4444"/>
    <text x="237" y="177" class="box-lbl" fill="#991b1b" text-anchor="middle">Retrieval Collapse: Postings Match = 0 Hits</text>
    <text x="237" y="193" class="box-sub" fill="#b91c1c" text-anchor="middle">Zero literal overlap; BM25 cannot infer synonymy with leaky_bucket</text>
  </g>

  <!-- Right Panel: AST Symbol Graph -->
  <g transform="translate(560, 150)">
    <rect width="475" height="215" rx="8" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>
    <rect x="15" y="12" width="445" height="28" rx="4" fill="#ffffff" stroke="#fed7aa"/>
    <text x="25" y="31" class="panel-title" fill="#9a3412">PATHOLOGY 2: AST GRAPH STRUCTURAL BLINDNESS</text>

    <text x="20" y="62" class="box-lbl" fill="#0f172a">Static AST Symbol Table:</text>
    <text x="175" y="62" class="mono" fill="#c2410c">Exact match: TokenBucketDrain</text>

    <rect x="20" y="76" width="435" height="52" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="93" class="box-lbl" fill="#334155">Multi-Language &amp; Dynamic Dispatch Boundary:</text>
    <text x="30" y="112" class="mono" fill="#0f172a">channel.invoke("RateLimiterService", payload)</text>

    <path d="M 237 132 L 237 155" fill="none" stroke="#dc2626" stroke-width="1.5" marker-end="url(#arr-red)"/>

    <rect x="20" y="160" width="435" height="42" rx="4" fill="#ffedd5" stroke="#f97316"/>
    <text x="237" y="177" class="box-lbl" fill="#9a3412" text-anchor="middle">Structural Blindness: Dynamic Edge Severed</text>
    <text x="237" y="193" class="box-sub" fill="#c2410c" text-anchor="middle">Static AST parser cannot traverse RPC, reflection, or macro bounds</text>
  </g>

  <!-- Bottom Invariant Card -->
  <rect x="20" y="380" width="1040" height="64" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="405" class="card-text" font-weight="700" fill="#0f172a">The Architectural Dilemma:</text>
  <text x="195" y="405" class="card-text">Unimodal retrieval fails symmetrically: lexical BM25 is brittle to synonym variations, while static AST graphs</text>
  <text x="195" y="425" class="card-text">are blind to dynamic boundaries. Production agent runtimes require fused hybrid retrieval (BM25 + HNSW + CPG).</text>
</svg>"""

with open(f"{TARGET_DIR}/vol3-lexical-vs-ast.svg", "w") as f:
    f.write(svg_lexical_vs_ast)


# 2. vol3_ch06_hnsw_rrf.svg
svg_hnsw_rrf = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 500" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 15px; fill: #0f172a; }
      .sub-note { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #64748b; }
      .stage-hdr { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 11.5px; }
      .box-lbl { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 11px; text-anchor: middle; }
      .box-sub { font-family: system-ui, -apple-system, sans-serif; font-size: 10px; text-anchor: middle; fill: #475569; }
      .mono { font-family: ui-monospace, monospace; font-size: 10px; }
      .card-text { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #334155; }
    </style>
    <marker id="arr-m-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
  </defs>

  <rect width="1080" height="500" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- Top Banner -->
  <rect x="20" y="16" width="1040" height="40" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="41" class="title">MULTI-STAGE HYBRID RETRIEVAL AND RECIPROCAL RANK FUSION (RRF)</text>
  <text x="1045" y="41" class="sub-note" text-anchor="end">Sub-Linear Candidate Filtering with Full-Attention Cross-Encoder Precision</text>

  <!-- Query Box -->
  <rect x="360" y="70" width="360" height="40" rx="5" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="540" y="94" class="box-lbl" fill="#1e40af">Agent Query Q: "reclaim orphaned file descriptors"</text>

  <!-- Branching Arrows -->
  <path d="M 450 110 L 300 138" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-m-blue)"/>
  <path d="M 630 110 L 780 138" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-m-blue)"/>

  <!-- Stage 1: Dual Engines -->
  <!-- Sparse BM25 -->
  <g transform="translate(150, 140)">
    <rect width="300" height="75" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>
    <text x="150" y="24" class="stage-hdr" fill="#0f172a" text-anchor="middle">Sparse Engine (BM25 Postings)</text>
    <text x="150" y="42" class="box-sub">Exact Token IDF Matching &amp; Term Frequencies</text>
    <rect x="40" y="50" width="220" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="150" y="63" class="mono" fill="#0284c7" text-anchor="middle">Top-100 Candidates (Rank r_sparse)</text>
  </g>

  <!-- Dense HNSW -->
  <g transform="translate(630, 140)">
    <rect width="300" height="75" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>
    <text x="150" y="24" class="stage-hdr" fill="#0f172a" text-anchor="middle">Dense Engine (HNSW / IVF-PQ)</text>
    <text x="150" y="42" class="box-sub">Approximate Cosine / L2 Nearest Neighbors</text>
    <rect x="40" y="50" width="220" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="150" y="63" class="mono" fill="#7c3aed" text-anchor="middle">Top-100 Candidates (Rank r_dense)</text>
  </g>

  <!-- Converging Arrows into RRF -->
  <path d="M 300 215 L 470 248" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-m-blue)"/>
  <path d="M 780 215 L 610 248" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-m-blue)"/>

  <!-- Stage 2: RRF Merge -->
  <g transform="translate(320, 250)">
    <rect width="440" height="70" rx="6" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="1.5"/>
    <text x="220" y="24" class="stage-hdr" fill="#5b21b6" text-anchor="middle">Stage 2: Reciprocal Rank Fusion (RRF)</text>
    <text x="220" y="42" class="mono" font-weight="700" fill="#6d28d9" text-anchor="middle">RRF(d) = Σ 1 / (60 + r_m(d))  [Parameter-Free Ordinal Merge]</text>
    <text x="220" y="58" class="box-sub" fill="#5b21b6">Resolves score incommensurability; produces unified Top-50 candidate pool</text>
  </g>

  <!-- Down Arrow to Cross-Encoder -->
  <path d="M 540 320 L 540 342" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-m-blue)"/>

  <!-- Stage 3: Cross-Encoder Re-Ranking -->
  <g transform="translate(260, 345)">
    <rect width="560" height="65" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5"/>
    <text x="280" y="24" class="stage-hdr" fill="#065f46" text-anchor="middle">Stage 3: Cross-Encoder Full-Attention Re-Ranking</text>
    <text x="280" y="41" class="mono" fill="#047857" text-anchor="middle">Score(Q, D) = Softmax( Q · D^T / √d_k )  [All-to-All Token Cross-Attention]</text>
    <text x="280" y="55" class="box-sub" fill="#047857">Evaluates prepositional logic, syntactic negation, and subtle code invariants over 50 items</text>
  </g>

  <!-- Final Top-k Staged Result -->
  <!-- Down Arrow to Final Output -->
  <path d="M 540 410 L 540 422" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-m-blue)"/>
  <rect x="360" y="425" width="360" height="30" rx="4" fill="#0f172a" stroke="#0f172a"/>
  <text x="540" y="445" font-family="system-ui, sans-serif" font-weight="700" font-size="11.5px" fill="#38bdf8" text-anchor="middle">High-Assurance Staged Context: Top-k* (k* in [3, 5])</text>

  <!-- Bottom Invariant Card -->
  <rect x="20" y="465" width="1040" height="28" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="483" class="card-text"><b>Pipeline Invariant:</b> Decouple high-recall candidate generation (O(log N)) from high-precision cross-attention re-ranking (O(k·L²)).</text>
</svg>"""

with open(f"{TARGET_DIR}/vol3_ch06_hnsw_rrf.svg", "w") as f:
    f.write(svg_hnsw_rrf)


# 3. cpg-multi-hop-traversal.svg
svg_cpg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 470" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 15px; fill: #0f172a; }
      .sub-note { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #64748b; }
      .panel-title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 12px; }
      .box-lbl { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 11px; }
      .box-sub { font-family: system-ui, -apple-system, sans-serif; font-size: 10px; fill: #475569; }
      .mono { font-family: ui-monospace, monospace; font-size: 10px; }
      .card-text { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #334155; }
    </style>
    <marker id="arr-cpg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb"/>
    </marker>
    <marker id="arr-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed"/>
    </marker>
  </defs>

  <rect width="1080" height="470" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- Top Banner -->
  <rect x="20" y="16" width="1040" height="40" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="41" class="title">CODE PROPERTY GRAPH (CPG) &amp; SEEDED NEIGHBORHOOD EXPANSION</text>
  <text x="1045" y="41" class="sub-note" text-anchor="end">Multi-Hop Dependency Traversal Across Translation Units</text>

  <!-- Left: Panel A CPG Traversal -->
  <g transform="translate(30, 75)">
    <rect width="495" height="300" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <rect x="15" y="12" width="465" height="28" rx="4" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="25" y="31" class="panel-title" fill="#1e40af">PANEL A: STRUCTURAL MULTI-HOP PATH TRAVERSAL</text>

    <!-- Node 1 -->
    <rect x="30" y="60" width="180" height="45" rx="5" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="120" y="80" class="box-lbl" fill="#1e3a8a" text-anchor="middle">process_transaction()</text>
    <text x="120" y="95" class="box-sub" text-anchor="middle">File: tx_handler.c</text>

    <!-- Arrow 1 to 2 -->
    <path d="M 210 82.5 L 285 82.5" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-cpg)"/>
    <text x="247" y="75" class="mono" font-weight="700" fill="#2563eb" text-anchor="middle">CALLS</text>

    <!-- Node 2 -->
    <rect x="295" y="60" width="170" height="45" rx="5" fill="#ffffff" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="380" y="80" class="box-lbl" fill="#1e3a8a" text-anchor="middle">validate_signature()</text>
    <text x="380" y="95" class="box-sub" text-anchor="middle">File: sec_crypto.c</text>

    <!-- Arrow 2 to 3 -->
    <path d="M 380 105 L 380 155" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-cpg)"/>
    <text x="415" y="135" class="mono" font-weight="700" fill="#2563eb" text-anchor="middle">READS</text>

    <!-- Node 3 -->
    <rect x="295" y="160" width="170" height="45" rx="5" fill="#ffffff" stroke="#10b981" stroke-width="1.5"/>
    <text x="380" y="180" class="box-lbl" fill="#065f46" text-anchor="middle">HardwareSecModule</text>
    <text x="380" y="195" class="box-sub" text-anchor="middle">Global Struct: hsm_key</text>

    <!-- Arrow 1 to 4 -->
    <path d="M 120 105 L 120 155" fill="none" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr-cpg)"/>
    <text x="155" y="135" class="mono" font-weight="700" fill="#2563eb" text-anchor="middle">MODIFIES</text>

    <!-- Node 4 -->
    <rect x="30" y="160" width="180" height="45" rx="5" fill="#ffffff" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="120" y="180" class="box-lbl" fill="#b45309" text-anchor="middle">TxLedgerRecord</text>
    <text x="120" y="195" class="box-sub" text-anchor="middle">State Variable: pending_bit</text>

    <!-- Linearized Callpath Box -->
    <rect x="20" y="225" width="455" height="60" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="242" class="box-lbl" fill="#0f172a">Serialized Callpath Representation:</text>
    <text x="30" y="259" class="mono" fill="#475569">[PATH] tx_handler.c:process_transaction()</text>
    <text x="30" y="274" class="mono" fill="#2563eb">  -&gt; CALLS sec_crypto.c:validate_signature() -&gt; READS hsm_key</text>
  </g>

  <!-- Right: Panel B Seeded Neighborhood Expansion -->
  <g transform="translate(555, 75)">
    <rect width="495" height="300" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <rect x="15" y="12" width="465" height="28" rx="4" fill="#f5f3ff" stroke="#ddd6fe"/>
    <text x="25" y="31" class="panel-title" fill="#6d28d9">PANEL B: SEEDED NEIGHBORHOOD EXPANSION &amp; PPR PRUNING</text>

    <!-- Step 1 -->
    <rect x="20" y="55" width="455" height="45" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="73" class="box-lbl" fill="#0f172a">1. Hybrid Seed Selection:</text>
    <text x="30" y="90" class="mono" fill="#6d28d9">V_0 = Top-k_0 by RRF(Q)  [k_0 = 5 initial anchor functions]</text>

    <!-- Arrow down -->
    <path d="M 247 100 L 247 115" fill="none" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arr-purple)"/>

    <!-- Step 2 -->
    <rect x="20" y="120" width="455" height="60" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="138" class="box-lbl" fill="#0f172a">2. Topological Expansion Dilemma:</text>
    <text x="30" y="154" class="box-sub" fill="#dc2626">• Bounded BFS (H=3): 1,310 functions = 235k tokens (28.8× Context Overflow!)</text>
    <text x="30" y="170" class="box-sub" fill="#059669">• Personalized PageRank (PPR): Biased walk flows along CALLS/MODIFIES edges</text>

    <!-- Arrow down -->
    <path d="M 247 180 L 247 195" fill="none" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arr-purple)"/>

    <!-- Step 3 -->
    <rect x="20" y="200" width="455" height="85" rx="4" fill="#ecfdf5" stroke="#10b981"/>
    <text x="30" y="220" class="box-lbl" fill="#065f46">3. Compact Induced Subgraph G_sub (38 functions):</text>
    <text x="30" y="238" class="mono" fill="#047857">Tokens consumed: 6,840 / 8,192 budget (1,352 tokens headroom)</text>
    <text x="30" y="256" class="box-sub" fill="#065f46">Stationary distribution p = (1 - α)s + α P^T p with cutoff ε_PPR = 1.2e-3</text>
    <text x="30" y="272" class="box-sub" fill="#047857">Guarantees zero context exhaustion while retaining full multi-hop causality</text>
  </g>

  <!-- Bottom Invariant Card -->
  <rect x="20" y="390" width="1040" height="64" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="415" class="card-text" font-weight="700" fill="#0f172a">Graph-Structured Retrieval Invariant:</text>
  <text x="270" y="415" class="card-text">Never execute unconstrained BFS across production call graphs. Seed topological expansion via</text>
  <text x="270" y="435" class="card-text">hybrid search, and prune candidate walks via Personalized PageRank to enforce context budget bounds.</text>
</svg>"""

with open(f"{TARGET_DIR}/cpg-multi-hop-traversal.svg", "w") as f:
    f.write(svg_cpg)


# 4. retrieval_access_control_prefilter.svg
svg_access_control = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 450" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 15px; fill: #0f172a; }
      .sub-note { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #64748b; }
      .panel-title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 12px; }
      .box-lbl { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 11px; }
      .box-sub { font-family: system-ui, -apple-system, sans-serif; font-size: 10px; fill: #475569; }
      .mono { font-family: ui-monospace, monospace; font-size: 10px; }
      .card-text { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #334155; }
    </style>
    <marker id="arr-sec" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284c7"/>
    </marker>
  </defs>

  <rect width="1080" height="450" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- Top Banner -->
  <rect x="20" y="16" width="1040" height="40" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="41" class="title">ACCESS-CONTROLLED EPISODIC STORAGE AND BITSET PRE-FILTERING</text>
  <text x="1045" y="41" class="sub-note" text-anchor="end">Eliminating Indirect Injection and Post-Filtering Recall Collapse</text>

  <!-- Left Side: Supervisor & Pre-filtering -->
  <g transform="translate(35, 75)">
    <rect width="520" height="285" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
    <rect x="15" y="12" width="490" height="28" rx="4" fill="#e0f2fe" stroke="#bae6fd"/>
    <text x="25" y="31" class="panel-title" fill="#0369a1">SECURE PRE-FILTERING PROTOCOL (IN-TRAVERSAL)</text>

    <!-- Query Step -->
    <rect x="20" y="55" width="480" height="42" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="73" class="box-lbl" fill="#0f172a">1. Authenticated Query Ingress:</text>
    <text x="30" y="88" class="mono" fill="#0284c7">Query Q + Security Principal Token U (Active Role / Scopes)</text>

    <!-- Arrow down -->
    <path d="M 260 97 L 260 112" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arr-sec)"/>

    <!-- Bitset Step -->
    <rect x="20" y="115" width="480" height="45" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="133" class="box-lbl" fill="#0f172a">2. Dynamic Bitset Mask Generation:</text>
    <text x="30" y="148" class="mono" fill="#0369a1">Bitset B = { doc_id : Principal U in ACL(doc_id) }</text>

    <!-- Arrow down -->
    <path d="M 260 160 L 260 175" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arr-sec)"/>

    <!-- Traversal Step -->
    <rect x="20" y="178" width="480" height="48" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="196" class="box-lbl" fill="#0f172a">3. Constrained Index Traversal:</text>
    <text x="30" y="213" class="mono" fill="#0f172a">• HNSW: Step to neighbor v iff B[v] == 1  |  • BM25: Postings ∩ B</text>

    <!-- Arrow down -->
    <path d="M 260 226 L 260 238" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arr-sec)"/>

    <!-- Result -->
    <rect x="20" y="240" width="480" height="35" rx="4" fill="#ecfdf5" stroke="#10b981"/>
    <text x="260" y="262" class="box-lbl" fill="#065f46" text-anchor="middle">Guaranteed Authorized Result Set R_k (Zero Latency Variance)</text>
  </g>

  <!-- Right Side: Post-Filtering Pathology -->
  <g transform="translate(585, 75)">
    <rect width="460" height="285" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5"/>
    <rect x="15" y="12" width="430" height="28" rx="4" fill="#ffffff" stroke="#fecaca"/>
    <text x="25" y="31" class="panel-title" fill="#991b1b">POST-FILTERING FAILURE MODES</text>

    <rect x="20" y="55" width="420" height="50" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="73" class="box-lbl" fill="#991b1b">Pathology 1: Recall Collapse</text>
    <text x="30" y="92" class="box-sub">If top-M candidates are unauthorized, result set is empty</text>

    <rect x="20" y="115" width="420" height="50" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="133" class="box-lbl" fill="#991b1b">Pathology 2: Unpredictable P99 Tail Latency</text>
    <text x="30" y="152" class="box-sub">Iteratively expanding search radius M explodes query deadlines</text>

    <rect x="20" y="175" width="420" height="85" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
    <text x="30" y="195" class="box-lbl" fill="#0f172a">Ingestion Quarantine Invariant:</text>
    <text x="30" y="213" class="box-sub">• Secret Scrubbing: Regex + Entropy scanner strips credentials</text>
    <text x="30" y="229" class="box-sub">• XML / Nonce Fencing: Wraps evidence in non-executable tags</text>
    <text x="30" y="245" class="box-sub">• Neutralizes Indirect Prompt Injection before context entry</text>
  </g>

  <!-- Bottom Invariant Card -->
  <rect x="20" y="375" width="1040" height="60" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="400" class="card-text" font-weight="700" fill="#0f172a">Zero Ambient Authority Invariant in Storage:</text>
  <text x="320" y="400" class="card-text">Enforce access controls during index traversal via bitset masking (pre-filtering).</text>
  <text x="320" y="420" class="card-text">Never execute unconstrained similarity search followed by post-query rejection.</text>
</svg>"""

with open(f"{TARGET_DIR}/retrieval_access_control_prefilter.svg", "w") as f:
    f.write(svg_access_control)


# 5. distractor_dilemma_success_curve.svg
svg_distractor = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 460" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: system-ui, -apple-system, sans-serif; font-weight: 700; font-size: 15px; fill: #0f172a; }
      .sub-note { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #64748b; }
      .axis-lbl { font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 700; fill: #1e293b; }
      .tick-lbl { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #64748b; }
      .legend-lbl { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; font-weight: 600; }
      .card-text { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #334155; }
    </style>
    <marker id="arr-axis-d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#1e293b"/>
    </marker>
  </defs>

  <rect width="1080" height="460" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>

  <!-- Top Title Banner -->
  <rect x="20" y="16" width="1040" height="40" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="41" class="title">THE DISTRACTOR DILEMMA: TASK SUCCESS VS. RETRIEVAL DEPTH (k)</text>
  <text x="1045" y="41" class="sub-note" text-anchor="end">Non-Monotonic Task Performance Under Attention Dilution</text>

  <!-- Plot Background Regions -->
  <!-- Under-retrieval -->
  <rect x="110" y="75" width="220" height="230" fill="#fef2f2" opacity="0.6"/>
  <text x="220" y="98" font-family="system-ui, sans-serif" font-weight="700" font-size="11px" fill="#b91c1c" text-anchor="middle">UNDER-RETRIEVAL REGIME</text>
  <text x="220" y="114" font-family="system-ui, sans-serif" font-size="10px" fill="#b91c1c" text-anchor="middle">(Missing Invariant / Evidence)</text>

  <!-- Optimal Window -->
  <rect x="330" y="75" width="180" height="230" fill="#f0fdf4" opacity="0.6"/>
  <text x="420" y="98" font-family="system-ui, sans-serif" font-weight="700" font-size="11px" fill="#15803d" text-anchor="middle">OPTIMAL OPERATING WINDOW</text>
  <text x="420" y="114" font-family="system-ui, sans-serif" font-size="10px" fill="#15803d" text-anchor="middle">(k* in [3, 5] Passages)</text>

  <!-- Dilution Regime -->
  <rect x="510" y="75" width="500" height="230" fill="#fff7ed" opacity="0.6"/>
  <text x="760" y="98" font-family="system-ui, sans-serif" font-weight="700" font-size="11px" fill="#c2410c" text-anchor="middle">ATTENTION DILUTION &amp; DISTRACTOR INTERFERENCE</text>
  <text x="760" y="114" font-family="system-ui, sans-serif" font-size="10px" fill="#c2410c" text-anchor="middle">(Lost-in-the-Middle + Distractor Hallucinations)</text>

  <!-- Grid lines -->
  <path d="M 110 245 L 1010 245" stroke="#e2e8f0" stroke-width="1"/>
  <path d="M 110 185 L 1010 185" stroke="#e2e8f0" stroke-width="1"/>
  <path d="M 110 125 L 1010 125" stroke="#e2e8f0" stroke-width="1"/>

  <!-- Recall@k Curve (Monotonically Increasing Gray Dashed Line) -->
  <path d="M 110 290 C 220 220, 350 160, 510 140 C 700 125, 880 118, 1010 115" 
        fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4"/>

  <!-- End-to-End Task Success Rate Curve (Green to Orange/Red) -->
  <path d="M 110 280 
           C 200 230, 280 160, 360 130 
           C 410 115, 430 115, 470 130 
           C 560 170, 700 240, 850 260 
           C 920 270, 970 275, 1010 280" 
        fill="none" stroke="#2563eb" stroke-width="3.5"/>

  <!-- Peak Marker at k* -->
  <circle cx="420" cy="118" r="6" fill="#10b981" stroke="#ffffff" stroke-width="2"/>
  <rect x="360" y="135" width="120" height="24" rx="4" fill="#ffffff" stroke="#10b981" stroke-width="1.5"/>
  <text x="420" y="151" font-family="system-ui, sans-serif" font-weight="700" font-size="11px" fill="#047857" text-anchor="middle">Peak SR: 72% (k*)</text>

  <!-- Axes -->
  <path d="M 110 305 L 1030 305" stroke="#1e293b" stroke-width="1.5" marker-end="url(#arr-axis-d)"/>
  <path d="M 110 305 L 110 70" stroke="#1e293b" stroke-width="1.5" marker-end="url(#arr-axis-d)"/>

  <!-- Y-Axis Labels -->
  <text x="45" y="190" class="axis-lbl" transform="rotate(-90 45,190)" text-anchor="middle">Task Success Rate (%)</text>
  <text x="100" y="309" class="tick-lbl" text-anchor="end">0%</text>
  <text x="100" y="249" class="tick-lbl" text-anchor="end">25%</text>
  <text x="100" y="189" class="tick-lbl" text-anchor="end">50%</text>
  <text x="100" y="129" class="tick-lbl" text-anchor="end">75%</text>

  <!-- X-Axis Labels -->
  <text x="560" y="348" class="axis-lbl" text-anchor="middle">Retrieved Context Passages (k)</text>
  <text x="110" y="325" class="tick-lbl" text-anchor="middle">0</text>
  <text x="220" y="325" class="tick-lbl" text-anchor="middle">2</text>
  <text x="330" y="325" class="tick-lbl" text-anchor="middle">3</text>
  <text x="420" y="325" class="tick-lbl" text-anchor="middle" font-weight="700" fill="#15803d">5 (k*)</text>
  <text x="510" y="325" class="tick-lbl" text-anchor="middle">10</text>
  <text x="680" y="325" class="tick-lbl" text-anchor="middle">20</text>
  <text x="850" y="325" class="tick-lbl" text-anchor="middle">50</text>
  <text x="1010" y="325" class="tick-lbl" text-anchor="middle">100</text>

  <!-- Legend -->
  <g transform="translate(250, 365)">
    <line x1="0" y1="10" x2="30" y2="10" stroke="#2563eb" stroke-width="3"/>
    <text x="38" y="14" class="legend-lbl" fill="#1e40af">End-to-End Task Success Rate (SR %)</text>

    <line x1="300" y1="10" x2="330" y2="10" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4"/>
    <text x="338" y="14" class="legend-lbl" fill="#64748b">Raw Retrieval Recall@k (Monotonic)</text>
  </g>

  <!-- Bottom Invariant Card -->
  <rect x="20" y="395" width="1040" height="52" rx="6" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="35" y="418" class="card-text" font-weight="700" fill="#0f172a">The End-to-End Verification Argument:</text>
  <text x="270" y="418" class="card-text">Maximizing Recall@k does not maximize agent task completion. Staging excess passages dilutes self-attention</text>
  <text x="270" y="434" class="card-text">mass across distractor keys, triggering hallucinations and reducing task success from 72% down to 28%.</text>
</svg>"""

with open(f"{TARGET_DIR}/distractor_dilemma_success_curve.svg", "w") as f:
    f.write(svg_distractor)

print("Generated all 5 Chapter 06 SVGs successfully")
