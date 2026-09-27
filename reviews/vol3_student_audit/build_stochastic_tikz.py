import subprocess

TIKZ_CODE = r"""\documentclass[tikz,border=20pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb}
\usetikzlibrary{calc,positioning,fit,backgrounds,shapes.geometric,arrows.meta}

% =============================================================================
% CANONICAL ACADEMIC SYSTEMS PALETTE (MLSysBook Standard)
% =============================================================================
\definecolor{oxfordnavy}{HTML}{1E293B}    % Primary Architecture / Borders / Titles
\definecolor{slatesteel}{HTML}{334155}    % Secondary Datapath / Outlines
\definecolor{slateteal}{HTML}{0F766E}     % Memory Hierarchy / Caches
\definecolor{deepburgundy}{HTML}{991B1B}  % Control / Preemption / Traps
\definecolor{warmochre}{HTML}{92400E}     % Actuation / Tool Interconnect
\definecolor{forestsage}{HTML}{166534}    % Verification / State Gates
\definecolor{amethyst}{HTML}{581C87}      % Volume III Theme Accent
\definecolor{coolparchment}{HTML}{F8FAFC} % Neutral Box Fill
\definecolor{paleslate}{HTML}{F1F5F9}     % Sub-block Accent Fill
\definecolor{slaterim}{HTML}{CBD5E1}      % Structural Stroke Rule
\definecolor{charcoal}{HTML}{0F172A}      % Primary Labels
\definecolor{mutediron}{HTML}{475569}     % Secondary Specs / Annotations

\tikzset{
    >=Latex,
    headerbar/.style={
        fill=#1, text=white, font=\bfseries\footnotesize, rounded corners=2pt,
        inner sep=4.5pt
    },
    containerbox/.style={
        draw=slaterim, line width=0.9pt, fill=coolparchment, rounded corners=4pt
    },
    block/.style={
        draw=#1, line width=0.9pt, fill=white, rounded corners=2.5pt,
        inner sep=4.5pt
    },
    flowarrow/.style={
        ->, line width=1.1pt, draw=#1
    },
    bidirarrow/.style={
        <->, line width=1.2pt, draw=#1
    }
}

\begin{document}
\begin{tikzpicture}[x=1cm, y=1cm]

% =============================================================================
% 1. TOP TIER: STOCHASTIC PROCESSOR & CONTEXT MEMORY HIERARCHY (y: 7.7 to 15.0)
% =============================================================================

% --- LEFT CONTAINER: STOCHASTIC PROCESSOR (x: 0 to 8.6, y: 7.7 to 15.0) ---
\draw[containerbox] (0, 7.7) rectangle (8.6, 15.0);
\node[anchor=north west, fill=oxfordnavy, text=white, font=\bfseries\footnotesize, rounded corners=2pt, minimum width=8.6cm, inner sep=4.5pt] 
    at (0, 15.0) {\ \ STOCHASTIC PROCESSOR (Compute Core) \hfill \textnormal{\scriptsize Matrix Inference Substrate}\ \ };

% Context Staging Register (y: 14.3 to 13.6)
\node[block=slatesteel, anchor=north, minimum width=8.0cm] (reg_c) at (4.3, 14.3) {
    \begin{minipage}{7.7cm}
        \textbf{\scriptsize Context Staging Register} $\mathbf{[c_t]}$ \hfill \texttt{\scriptsize [Prompt $\mid$ Trajectory $\tau_t$ $\mid$ Observation $o_t$]}
    \end{minipage}
};

% Neural Parameter Weights (y: 13.25 to 12.3)
\node[block=amethyst, anchor=north, minimum width=8.0cm] (core_w) at (4.3, 13.25) {
    \begin{minipage}{7.7cm}
        \textbf{\scriptsize Neural Parameter Weights} $\mathbf{[W_\theta]}$ \textbf{\& Autoregressive Core}\\
        {\tiny $z_t = \mathrm{Softmax}(f_\theta(c_t)/T) \in \mathbb{R}^{|\mathcal{V}|} \quad \cdot \quad \text{Prefill (GEMM)} \leftrightarrow \text{Decode (GEMV)}$}
    \end{minipage}
};

% Deliberation & Grammar Masking (Side-by-Side) (y: 11.95 to 10.55)
\node[block=slatesteel, anchor=north west, minimum width=3.85cm, minimum height=1.35cm] (spec_delib) at (0.3, 11.95) {
    \begin{minipage}{3.65cm}
        \textbf{\scriptsize Speculative Deliberator}\\
        {\tiny MCTS / PRM Verifier Search\\Test-Time Deliberation Scaling\\Rollout Pruning \& Escrow}
    \end{minipage}
};
\node[block=forestsage, anchor=north east, minimum width=3.85cm, minimum height=1.35cm] (gram_mask) at (8.3, 11.95) {
    \begin{minipage}{3.65cm}
        \textbf{\scriptsize Grammar Logit Mask (FSM)}\\
        {\tiny Constrained Decoding Engine\\Enforces Valid JSON/Tool Schema $\mathcal{G}$\\Zero Syntactic Parsing Defects}
    \end{minipage}
};

% Action Proposal Register (y: 10.15 to 9.45)
\node[block=deepburgundy, anchor=north, minimum width=8.0cm] (reg_a) at (4.3, 10.15) {
    \begin{minipage}{7.7cm}
        \textbf{\scriptsize Action Proposal Register} $\mathbf{[a_{\mathrm{prop}}]}$ \hfill \texttt{\scriptsize Status: [PROPOSE $\mid$ TRUNCATE $\mid$ REFUSE]}
    \end{minipage}
};

\node[font=\tiny, text=mutediron] at (4.3, 8.4) {Zero Ambient Authority $\cdot$ Memory Bandwidth Roofline Governed};

% Internal arrows inside Processor
\draw[flowarrow=amethyst] (reg_c.south) -- (core_w.north);
\draw[flowarrow=amethyst] ($(core_w.south)-(2.0,0)$) -- (spec_delib.north);
\draw[flowarrow=amethyst] ($(core_w.south)+(2.0,0)$) -- (gram_mask.north);
\draw[flowarrow=slatesteel] (spec_delib.south) -- ($(reg_a.north)-(2.0,0)$);
\draw[flowarrow=forestsage] (gram_mask.south) -- ($(reg_a.north)+(2.0,0)$);

% --- RIGHT CONTAINER: CONTEXT & MEMORY HIERARCHY (x: 9.6 to 18.2, y: 7.7 to 15.0) ---
\draw[containerbox] (9.6, 7.7) rectangle (18.2, 15.0);
\node[anchor=north west, fill=slateteal, text=white, font=\bfseries\footnotesize, rounded corners=2pt, minimum width=8.6cm, inner sep=4.5pt] 
    at (9.6, 15.0) {\ \ CONTEXT \& MEMORY HIERARCHY \hfill \textnormal{\scriptsize Hierarchical State Tiers}\ \ };

% L1 Active Working Context (y: 14.3 to 13.1)
\node[block=slateteal, anchor=north, minimum width=8.0cm] (l1_mem) at (13.9, 14.3) {
    \begin{minipage}{7.7cm}
        \textbf{\scriptsize L1: Active Working Context Window} \hfill \textbf{\scriptsize [Accelerator HBM]}\\
        {\tiny Attention Sinks $\cdot$ Sliding Scratchpad $\cdot$ Working-Set Budget ($128\mathrm{K}\text{--}1\mathrm{M}\text{ tokens}$)}
    \end{minipage}
};

% L2 Virtual Paged KV-Cache (y: 12.7 to 11.5)
\node[block=slateteal, anchor=north, minimum width=8.0cm] (l2_mem) at (13.9, 12.7) {
    \begin{minipage}{7.7cm}
        \textbf{\scriptsize L2: Virtual Paged KV-Cache} \hfill \textbf{\scriptsize [Host DRAM / CXL]}\\
        {\tiny PagedAttention Block Tables $\cdot$ Radix Prefix Tree $\cdot$ Async Host Paging ($64\mathrm{\,GB/s}$)}
    \end{minipage}
};

% L3 Persistent Episodic Store (y: 11.1 to 9.9)
\node[block=slateteal, anchor=north, minimum width=8.0cm] (l3_mem) at (13.9, 11.1) {
    \begin{minipage}{7.7cm}
        \textbf{\scriptsize L3: Persistent Episodic Storage} \hfill \textbf{\scriptsize [Host NVMe / Object Store]}\\
        {\tiny HNSW Vector Index $\cdot$ Relational Session State $\cdot$ Write-Ahead Log (WAL) Replay}
    \end{minipage}
};

\node[font=\tiny, text=mutediron] at (13.9, 8.4) {Two-Phase Context Commit $\cdot$ LRU Cache Eviction $\cdot$ Prefix Invalidation};

% Memory Inter-Tier Arrows
\draw[bidirarrow=slateteal] (l1_mem.south) -- node[right, font=\tiny, text=slateteal] {Host-Device Paging} (l2_mem.north);
\draw[bidirarrow=slateteal] (l2_mem.south) -- node[right, font=\tiny, text=slateteal] {Vector / WAL Hydration} (l3_mem.north);

% High-Bandwidth Attention Bus between Processor & Memory
\draw[bidirarrow=slateteal, line width=1.6pt] (8.6, 13.0) -- 
    node[above, font=\bfseries\tiny, text=slateteal] {ATTENTION BUS} 
    node[below, font=\tiny, text=mutediron] {HBM3e ($3.4\mathrm{\,TB/s}$)} (9.6, 13.0);

% =============================================================================
% 2. MIDDLE TIER: SYSTEM BUS INTERCONNECT (y: 6.6 to 7.3)
% =============================================================================
\draw[draw=oxfordnavy, fill=oxfordnavy, rounded corners=2.5pt] (0, 6.6) rectangle (18.2, 7.3);
\node[text=white, font=\bfseries\scriptsize] at (9.1, 6.95) {
    ADDRESS / KV BUS \texttt{[/64]} \quad $\bullet$ \quad
    DATA / TOKEN BUS $\mathbf{[B \times d_{\mathrm{model}}]}$: Observations $o_t$ \& Proposals $a_{\mathrm{prop}}$ \quad $\bullet$ \quad
    CONTROL \& IRQ BUS \texttt{[TRAP $\mid$ ESCROW $\mid$ COMMIT]}
};

% Direct Connections from Top Tier to Bus
\draw[flowarrow=deepburgundy, line width=1.3pt] (reg_a.south) -- (4.3, 7.3);
\draw[bidirarrow=slateteal, line width=1.3pt] (13.9, 7.3) -- (13.9, 7.7);

% =============================================================================
% 3. AGENT OS CONTROL PLANE (HOST SUPERVISOR) (y: 1.8 to 6.2)
% =============================================================================
\draw[containerbox] (0, 1.8) rectangle (18.2, 6.2);
\node[anchor=north west, fill=oxfordnavy, text=white, font=\bfseries\footnotesize, rounded corners=2pt, minimum width=18.2cm, inner sep=4.5pt] 
    at (0, 6.2) {\ \ AGENT OS CONTROL PLANE \hfill \textnormal{\scriptsize Deterministic Host Control Plane (Supervisor Daemon)}\ \ };

% Four Supervisory Blocks (y: 5.6 to 3.7)
\node[block=oxfordnavy, anchor=north west, minimum width=4.15cm, minimum height=1.9cm] (acb_box) at (0.3, 5.6) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Agent Control Block (ACB)}\\
        {\tiny Trajectory PID $\cdot$ Program Counter\\Budget: $B_{\mathrm{consumed}} / B_{\max}$\\Lifecycle: \texttt{INIT$\to$RUN$\to$SUSP$\to$HALT}\\Context Stack Pointer $\mathrm{SP}_t$}
    \end{minipage}
};

\node[block=deepburgundy, anchor=north west, minimum width=4.15cm, minimum height=1.9cm] (sig_box) at (4.75, 5.6) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Signal \& Trap Reactor}\\
        {\tiny POSIX Signals: \texttt{SIGINT}, \texttt{SIGALRM}\\Preemption: Loop \& Budget Watchdog\\Human Authorization Escrow Gate\\Cascading Signal Escalation}
    \end{minipage}
};

\node[block=forestsage, anchor=north west, minimum width=4.15cm, minimum height=1.9cm] (verif_box) at (9.2, 5.6) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Verification Enclave}\\
        {\tiny Invariant Closure Watchdogs $\mathcal{I}(s_t)$\\Automated Test / Linter Runners\\Rejects Fail-Plausible Model Reports\\Deterministic Evidence Capture}
    \end{minipage}
};

\node[block=warmochre, anchor=north west, minimum width=4.15cm, minimum height=1.9cm] (saga_box) at (13.65, 5.6) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Saga Fault Coordinator}\\
        {\tiny Reversibility Boundary Manager\\Compensating Actions $\{T_i \to C_i\}$\\WAL Event-Sourced Hydration\\Circuit Breaker Trip Logic}
    \end{minipage}
};

% Capability Authorization Filter (y: 3.3 to 2.2)
\node[block=oxfordnavy, anchor=north, minimum width=17.6cm, fill=white] (auth_filter) at (9.1, 3.3) {
    \begin{minipage}{17.2cm}
        \centering
        \textbf{\scriptsize CAPABILITY AUTHORIZATION FILTER:}\quad
        Unverified Proposal $[a_{\mathrm{prop}}]$ $\xrightarrow{\quad\text{Schema Validation \& Path Canonicalization}\quad}$ Authorized Action $[a_{\mathrm{perm}}]$ \quad $\bullet$ \quad Violation $\to \bot$ Abort
    \end{minipage}
};

\draw[flowarrow=oxfordnavy, line width=1.3pt] (9.1, 6.6) -- (9.1, 6.2);

% =============================================================================
% 4. SANDBOXED ACTUATION SUBSYSTEM (I/O & TOOLS) (y: -2.7 to 1.4)
% =============================================================================
\draw[containerbox] (0, -2.7) rectangle (18.2, 1.4);
\node[anchor=north west, fill=warmochre, text=white, font=\bfseries\footnotesize, rounded corners=2pt, minimum width=18.2cm, inner sep=4.5pt] 
    at (0, 1.4) {\ \ I/O \& PERIPHERAL ACTUATION SUBSYSTEM \hfill \textnormal{\scriptsize Sandboxed Execution Boundary (Principle of Least Privilege)}\ \ };

% Three Actuation Blocks (y: 0.85 to -0.65)
\node[block=warmochre, anchor=north west, minimum width=5.6cm, minimum height=1.5cm] (mcp_bus) at (0.3, 0.85) {
    \begin{minipage}{5.35cm}
        \textbf{\scriptsize Model Context Protocol (MCP) Bus}\\
        {\tiny JSON-RPC 2.0 Client/Server $\cdot$ Schema Registry\\Async Tool Polling $\cdot$ Non-Blocking Dispatch\\Connection Heartbeats \& Health Probes}
    \end{minipage}
};

\node[block=deepburgundy, anchor=north, minimum width=5.6cm, minimum height=1.5cm] (sandbox_box) at (9.1, 0.85) {
    \begin{minipage}{5.35cm}
        \textbf{\scriptsize Copy-on-Write Execution Sandboxes}\\
        {\tiny Firecracker MicroVMs $\cdot$ gVisor $\cdot$ WASM\\$W \oplus X$ Isolation $\cdot$ NetNS Egress Proxy\\Memory Quota \& Watchdog Kill Limits}
    \end{minipage}
};

\node[block=forestsage, anchor=north east, minimum width=5.6cm, minimum height=1.5cm] (obs_norm) at (17.9, 0.85) {
    \begin{minipage}{5.35cm}
        \textbf{\scriptsize Observation Normalization Pipeline}\\
        {\tiny ANSI Escape Stripper $\cdot$ Deterministic Exit Codes\\Traceback Distillation $\to$ Structured $o_{t+1}$\\Prevents Byte-Pair Tokenizer Flooding}
    \end{minipage}
};

% External Actuation Ports (y: -1.05 to -2.15)
\node[draw=slaterim, fill=paleslate, minimum width=4.15cm, minimum height=0.9cm, rounded corners=2pt, font=\scriptsize, anchor=north west] at (0.3, -1.05) {\centering Bash Shell / Terminal};
\node[draw=slaterim, fill=paleslate, minimum width=4.15cm, minimum height=0.9cm, rounded corners=2pt, font=\scriptsize, anchor=north west] at (4.75, -1.05) {\centering Workspace File Tree};
\node[draw=slaterim, fill=paleslate, minimum width=4.15cm, minimum height=0.9cm, rounded corners=2pt, font=\scriptsize, anchor=north west] at (9.2, -1.05) {\centering REST / RPC Web APIs};
\node[draw=slaterim, fill=paleslate, minimum width=4.15cm, minimum height=0.9cm, rounded corners=2pt, font=\scriptsize, anchor=north west] at (13.65, -1.05) {\centering Human Review Terminal};

\draw[flowarrow=warmochre, line width=1.3pt] (auth_filter.south) -- (sandbox_box.north);
\draw[flowarrow=forestsage, line width=1.2pt] (sandbox_box.east) -- (obs_norm.west);

% Clean observation return around the right margin (x: 18.9)
\draw[flowarrow=forestsage, line width=1.2pt] 
    (obs_norm.east) -- ++(0.9, 0) -- ++(0, 6.85) -- 
    node[pos=0.5, above=2pt, font=\bfseries\tiny, text=forestsage] {Observation Return $o_{t+1}$} (18.2, 6.95);

% =============================================================================
% 5. BOTTOM TIER: TRAJECTORY LEARNING ENGINE (ACROSS TASKS) (y: -5.7 to -3.2)
% =============================================================================
\draw[containerbox] (0, -5.7) rectangle (18.2, -3.2);
\node[anchor=north west, fill=forestsage, text=white, font=\bfseries\footnotesize, rounded corners=2pt, minimum width=18.2cm, inner sep=4.5pt] 
    at (0, -3.2) {\ \ TRAJECTORY LEARNING ENGINE \hfill \textnormal{\scriptsize Offline Policy Optimization \& Recompilation (Across Tasks)}\ \ };

% Pipeline stages (y: -3.8 to -5.2)
\node[block=forestsage, anchor=north west, minimum width=4.15cm, minimum height=1.4cm] (t_harv) at (0.3, -3.8) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Telemetry Harvester}\\
        {\tiny Causal DAG $\cdot$ Span Extraction\\Turn Loss Masking $M_t$}
    \end{minipage}
};

\node[block=forestsage, anchor=north west, minimum width=4.15cm, minimum height=1.4cm] (t_filter) at (4.75, -3.8) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Curation \& Repair Filter}\\
        {\tiny Decontamination Sieve\\Counterfactual Sandboxes}
    \end{minipage}
};

\node[block=amethyst, anchor=north west, minimum width=4.15cm, minimum height=1.4cm] (t_sft) at (9.2, -3.8) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Trajectory Post-Training}\\
        {\tiny LoRA Adapter Swapping\\Action-Masked Supervised Loss}
    \end{minipage}
};

\node[block=forestsage, anchor=north west, minimum width=4.15cm, minimum height=1.4cm] (t_rlvr) at (13.65, -3.8) {
    \begin{minipage}{3.95cm}
        \textbf{\scriptsize Verifiable RL (RLVR / GRPO)}\\
        {\tiny Deterministic Reward Oracles $R(\tau)$\\Parameter Updates $\Delta\theta_{k+1}$}
    \end{minipage}
};

\draw[flowarrow=forestsage] (t_harv.east) -- (t_filter.west);
\draw[flowarrow=forestsage] (t_filter.east) -- (t_sft.west);
\draw[flowarrow=amethyst] (t_sft.east) -- (t_rlvr.west);

% Closed Feedback Loop back to Stochastic Processor (Routed around the left margin at x: -1.3)
\draw[flowarrow=amethyst, dashed, line width=1.4pt] 
    (t_harv.west) -- ++(-1.3, 0) -- ++(0, 16.7) node[midway, left=4pt, font=\bfseries\scriptsize, text=amethyst, rotate=90] {$\Delta\theta$ WEIGHT UPDATE BUS (Offline Policy Recompilation)}
    -| (core_w.west);

\end{tikzpicture}
\end{document}
"""

repo_root = Path(__file__).resolve().parents[2]
tex_path = repo_root / "books/vol3/tex/stochastic_computer_architecture.tex"

with open(tex_path, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-interaction=nonstopmode", str(tex_path)], cwd="/tmp", capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/stochastic_computer_architecture.pdf", "/tmp/stochastic_computer_arch_render_v6"], cwd="/tmp")
    subprocess.run(["pdftocairo", "-svg", "/tmp/stochastic_computer_architecture.pdf", "/tmp/stochastic_computer_architecture.svg"], cwd="/tmp")
    print("Rendered v6 to /tmp/stochastic_computer_arch_render_v6-1.png")
else:
    print(res.stdout[-1000:])
