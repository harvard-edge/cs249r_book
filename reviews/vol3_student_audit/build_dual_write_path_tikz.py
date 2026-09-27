#!/usr/bin/env python3
"""
Build and render Dual Write Path & Write Admission Quarantine Pipeline Diagram for Chapter 06.
Follows strict MLSysBook publication standards:
- Canonical color palette (DarkSlate, BlueLine, GreenLine, OrangeLine, RedLine, etc.)
- Visual inspection via pdftoppm + view_file
- Zero collisions or clipping
"""

import subprocess
from pathlib import Path

TIKZ_CODE = r"""\documentclass[tikz,border=22pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{tcolorbox}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc,positioning,arrows.meta,backgrounds,decorations.pathreplacing,shapes.geometric,fit}

% =============================================================================
% CANONICAL MLSYSBOOK PALETTE
% =============================================================================
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{slatesteel}{HTML}{334155}
\definecolor{GrayLine}{HTML}{64748B}
\definecolor{coolparchment}{HTML}{F8FAFC}
\definecolor{GridGray}{HTML}{E2E8F0}

\definecolor{BlueLine}{HTML}{006395}
\definecolor{BlueL}{RGB}{209,243,255}
\definecolor{BlueL0}{HTML}{F0F9FF}

\definecolor{GreenLine}{HTML}{008F45}
\definecolor{GreenL}{RGB}{219,253,166}
\definecolor{GreenL0}{HTML}{EBF8F2}
\definecolor{forestsage}{HTML}{166534}

\definecolor{OrangeLine}{HTML}{E67817}
\definecolor{OrangeL}{RGB}{254,237,222}
\definecolor{OrangeL0}{HTML}{FFFBEB}
\definecolor{warmochre}{HTML}{92400E}

\definecolor{RedLine}{HTML}{D9534F}
\definecolor{RedL}{RGB}{253,226,240}
\definecolor{RedL0}{HTML}{FEF2F2}
\definecolor{deepburgundy}{HTML}{991B1B}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth,
  node distance=0.5cm
]

% -------------------------------------------------------------
% TIER 0: MASTER TITLE BANNER
% -------------------------------------------------------------
\node[anchor=north west, inner sep=0pt] (title) at (0, 22.2) {
  \begin{minipage}{25.0cm}
    {\fontsize{18}{22}\selectfont \textbf{\textcolor{DarkSlate}{The Dual Write Path \& Write Admission Quarantine Pipeline}}}\\[4pt]
    {\fontsize{10.5}{14}\selectfont \textcolor{slatesteel}{Decoupling interactive critical-path logging ($< 1\text{ ms}$) from asynchronous surprisal-gated episodic consolidation}}
  \end{minipage}
};

% -------------------------------------------------------------
% TIER 1: INTERACTIVE AGENT LOOP & TURN EVENT EMISSION
% -------------------------------------------------------------
\node[anchor=north west, fill=coolparchment, draw=DarkSlate, line width=1.1pt, rounded corners=5pt, inner sep=8pt] 
  (agent_box) at (0, 20.8) {
  \begin{minipage}{24.2cm}
    \begin{tabular}{@{}p{7.2cm} p{11.2cm} p{5.4cm}@{}}
      \textbf{\textcolor{DarkSlate}{\fontsize{11}{13}\selectfont Agent Deliberation Loop (Turn $t$)}} & 
      \fontsize{9}{12}\selectfont \textbf{Event Tuple Emitted:} $e_t = \langle \text{Turn } t, \text{Task } \tau, a_t, O_t, \Delta_t, \text{ExitCode } 0 \rangle$ &
      \raggedleft \fontsize{8.5}{11}\selectfont \textbf{Budget Constraint:} $T_{\text{interactive}} \le 25\text{ ms}$
    \end{tabular}
  \end{minipage}
};

% Split Dispatch Arrows with labels
\draw[->, line width=1.5pt, BlueLine] (4.9, 19.65) -- (4.9, 18.95);
\node[anchor=west] at (5.0, 19.3) {\fontsize{7.5}{9}\selectfont \textcolor{BlueLine}{\textbf{Sync Zero-Copy WAL Append}}};

\draw[->, line width=1.5pt, OrangeLine] (17.5, 19.65) -- (17.5, 18.95);
\node[anchor=west] at (17.6, 19.3) {\fontsize{7.5}{9}\selectfont \textcolor{OrangeLine}{\textbf{Async Quarantine Pipeline}}};

% -------------------------------------------------------------
% TIER 2: DUAL WRITE PATHS (SIDE BY SIDE)
% -------------------------------------------------------------

% --- PATH 1: SYNCHRONOUS FAST APPEND PATH (LEFT COLUMN) ---
\node[anchor=north west, fill=BlueL0, draw=BlueLine, line width=1.2pt, rounded corners=5pt, inner sep=8pt] 
  (path1_box) at (0, 18.95) {
  \begin{minipage}{9.8cm}
    {\fontsize{11}{13}\selectfont \textbf{\textcolor{BlueLine}{PATH 1: Synchronous Append-Only WAL}}}\\[2pt]
    {\fontsize{8}{10}\selectfont \textcolor{slatesteel}{Interactive Critical Path $\cdot$ Ultra-low Latency Budget ($< 1\text{ ms}$)}}

    \vspace{7pt}
    % Sub-box 1: Zero-Compute Serialization
    \begin{tcolorbox}[colback=white, colframe=BlueLine!40, arc=3pt, boxrule=0.8pt, left=6pt, right=6pt, top=5pt, bottom=5pt]
      {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{DarkSlate}{1. Deterministic POSIX Stream Append}}}\\[2pt]
      {\fontsize{7.5}{9.5}\selectfont
      \textbullet\ Zero embedding inference $\cdot$ Zero vector search\\
      \textbullet\ Zero LLM reflection $\cdot$ Zero mutex lock contention\\
      \textbullet\ Append-only sequence framing: \texttt{write(wal\_fd, e\_t, len)}
      }
    \end{tcolorbox}

    \vspace{4pt}
    % Sub-box 2: In-Memory Ring Buffer & NVMe Commit
    \begin{tcolorbox}[colback=white, colframe=BlueLine!40, arc=3pt, boxrule=0.8pt, left=6pt, right=6pt, top=5pt, bottom=5pt]
      {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{DarkSlate}{2. Host DRAM Ring Buffer \& NVMe Commit}}}\\[2pt]
      {\fontsize{7.5}{9.5}\selectfont
      \textbullet\ Memory-mapped circular buffer (\texttt{mmap})\\
      \textbullet\ Direct async page flush to local NVMe SSD ($< 450\ \mu\text{s}$)\\
      \textbullet\ Authoritative causal spine $\mathcal{S}_{\text{causal}}$ for deterministic replay
      }
    \end{tcolorbox}

    \vspace{6pt}
    % Result Badge: Critical Path Latency
    \begin{tcolorbox}[colback=GreenL0, colframe=GreenLine, arc=3pt, boxrule=1.0pt, left=6pt, right=6pt, top=5pt, bottom=5pt]
      \centering
      {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{forestsage}{\checkmark\ Control Returns to Model in $0.45\text{ ms}$ ($< 1\text{ ms}$)}}\\[2pt]
      \fontsize{7.5}{9.5}\selectfont Zero pipeline stall $\implies$ Next token decode begins immediately}
    \end{tcolorbox}
  \end{minipage}
};

% --- PATH 2: ASYNCHRONOUS REFLECTION & QUARANTINE GATE (RIGHT COLUMN) ---
\node[anchor=north west, fill=OrangeL0, draw=OrangeLine, line width=1.2pt, rounded corners=5pt, inner sep=8pt] 
  (path2_box) at (10.4, 18.95) {
  \begin{minipage}{14.2cm}
    {\fontsize{11}{13}\selectfont \textbf{\textcolor{warmochre}{PATH 2: Asynchronous Write Admission Quarantine Gate}}}\\[2pt]
    {\fontsize{8}{10}\selectfont \textcolor{slatesteel}{Out-of-band Background Daemon $\cdot$ Triggered during tool execution waits or session idle}}

    \vspace{7pt}
    % Grid of 4 Gating Filters
    \begin{minipage}{6.8cm}
      % Gate 1: Principal Scoping
      \begin{tcolorbox}[colback=white, colframe=OrangeLine!50, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8}{10}\selectfont \textbf{\textcolor{DarkSlate}{Gate 1: Principal Tenant Scoping}}}\\[2pt]
        {\fontsize{7}{8.5}\selectfont Enforces tenant namespace isolation ($\mathcal{S}_{\text{tenant}}$). Rejects cross-tenant memory leakage.}
      \end{tcolorbox}
      
      \vspace{3pt}
      % Gate 3: Provenance Anchoring
      \begin{tcolorbox}[colback=white, colframe=OrangeLine!50, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8}{10}\selectfont \textbf{\textcolor{DarkSlate}{Gate 3: Provenance Anchoring}}}\\[2pt]
        {\fontsize{7}{8.5}\selectfont Binds immutable SHA-256 pointer to turn $t$ in WAL ($\text{id}_{\text{traj}}$). Verifiable audit trail.}
      \end{tcolorbox}
    \end{minipage}
    \hfill
    \begin{minipage}{6.8cm}
      % Gate 2: Secret Scrubber
      \begin{tcolorbox}[colback=white, colframe=OrangeLine!50, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8}{10}\selectfont \textbf{\textcolor{DarkSlate}{Gate 2: Secret \& PII Scrubber}}}\\[2pt]
        {\fontsize{7}{8.5}\selectfont Regex \& Shannon entropy filters strip API keys, private certs, and session tokens before ingestion.}
      \end{tcolorbox}

      \vspace{3pt}
      % Gate 4: Verification Closure
      \begin{tcolorbox}[colback=white, colframe=OrangeLine!50, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8}{10}\selectfont \textbf{\textcolor{DarkSlate}{Gate 4: Verification Closure Gate}}}\\[2pt]
        {\fontsize{7}{8.5}\selectfont Asserts deterministic oracle proof (exit code 0, test pass). Discards ungrounded model self-reports.}
      \end{tcolorbox}
    \end{minipage}

    \vspace{5pt}
    % Surprisal Novelty Filter (Gate 5)
    \begin{tcolorbox}[colback=white, colframe=deepburgundy, arc=3pt, boxrule=1.0pt, left=6pt, right=6pt, top=4pt, bottom=4pt]
      \begin{minipage}{8.8cm}
        {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{deepburgundy}{Gate 5: Geometric Surprisal \& Novelty Filter}}}\\[2pt]
        {\fontsize{7.5}{9.5}\selectfont
        $\mathcal{N}(e) = 1 - \max_{m \in \mathcal{M}_{<t}} \cos\bigl(E_D(e), E_D(m)\bigr) > \tau_{\text{novel}}$\\[2pt]
        Evaluates angular distance against existing memory manifold $\mathcal{M}_{<t}$.
        }
      \end{minipage}
      \hfill
      \begin{minipage}{4.4cm}
        \raggedleft
        \begin{tcolorbox}[colback=RedL0, colframe=RedLine, arc=2pt, boxrule=0.6pt, left=4pt, right=4pt, top=2pt, bottom=2pt]
          \centering
          {\fontsize{6.8}{8}\selectfont \textcolor{deepburgundy}{\textbf{If $\le \tau_{\text{novel}}$: Discard (Redundant)}}}
        \end{tcolorbox}
        \vspace{1pt}
        \begin{tcolorbox}[colback=GreenL0, colframe=GreenLine, arc=2pt, boxrule=0.6pt, left=4pt, right=4pt, top=2pt, bottom=2pt]
          \centering
          {\fontsize{6.8}{8}\selectfont \textcolor{forestsage}{\textbf{If $> \tau_{\text{novel}}$: Admit to Index}}}
        \end{tcolorbox}
      \end{minipage}
    \end{tcolorbox}
  \end{minipage}
};

% -------------------------------------------------------------
% TIER 3: OFFLINE REFLECTION, CONSOLIDATION & MULTI-TIER INDEXING
% Dynamically anchored below the lower of path1_box and path2_box
% -------------------------------------------------------------
\node[anchor=north west, fill=coolparchment, draw=DarkSlate, line width=1.2pt, rounded corners=5pt, inner sep=8pt] 
  (consolidation_box) at ([yshift=-1.2cm] 0, 0 |- path2_box.south) {
  \begin{minipage}{24.2cm}
    {\fontsize{11}{13}\selectfont \textbf{\textcolor{DarkSlate}{TIER 3: Offline Episodic Consolidation \& Multi-Substrate Materialization}}}\\[2pt]
    {\fontsize{8}{10}\selectfont \textcolor{slatesteel}{Transforms raw admitted traces into high-salience procedural lessons, compressed vector clusters, and active tombstone bitsets}}

    \vspace{7pt}
    \begin{minipage}{7.8cm}
      % Sub-panel A: LLM Reflection Engine
      \begin{tcolorbox}[colback=white, colframe=DarkSlate!40, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{DarkSlate}{A. Deliberative LLM Reflection}}}\\[2pt]
        {\fontsize{7.5}{9.5}\selectfont
        \textbullet\ Evaluates multi-turn trajectory in batch\\
        \textbullet\ Synthesizes generalized schema $S_k$: ``Root cause: cursor timeout, not connection exhaustion''\\
        \textbullet\ Compresses 4,000-token trace to 85-token lesson
        }
      \end{tcolorbox}
    \end{minipage}
    \hfill
    \begin{minipage}{7.6cm}
      % Sub-panel B: DBSCAN Clustering
      \begin{tcolorbox}[colback=white, colframe=DarkSlate!40, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{DarkSlate}{B. DBSCAN Sieve \& Dedup}}}\\[2pt]
        {\fontsize{7.5}{9.5}\selectfont
        \textbullet\ Manifold density clustering ($\varepsilon = 0.12$)\\
        \textbullet\ Merges near-duplicate operational records\\
        \textbullet\ Elevates access frequency weights\\
        \textbullet\ Eliminates distractor clustering in top-$K$
        }
      \end{tcolorbox}
    \end{minipage}
    \hfill
    \begin{minipage}{8.2cm}
      % Sub-panel C: Multi-Substrate Store
      \begin{tcolorbox}[colback=white, colframe=DarkSlate!40, arc=3pt, boxrule=0.8pt, left=5pt, right=5pt, top=4pt, bottom=4pt]
        {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{DarkSlate}{C. Persistent Multi-Index Placement}}}\\[2pt]
        {\fontsize{7.5}{9.5}\selectfont
        \textbullet\ \textbf{Dense ANN:} IVF-PQ / HNSW graph insertion\\
        \textbullet\ \textbf{Code Property Graph:} SCIP/Tree-sitter edge update\\
        \textbullet\ \textbf{In-Memory Tombstones:} $400\text{ ns}$ hash bitset update, instantly suppressing stale file versions
        }
      \end{tcolorbox}
    \end{minipage}
  \end{minipage}
};

% Connecting Arrow from Path 2 to Tier 3 (strictly between borders)
\draw[->, line width=1.6pt, forestsage] (path2_box.south -| 17.5, 0) -- (consolidation_box.north -| 17.5, 0)
  node[midway, right=4pt] {\fontsize{8}{10}\selectfont \textcolor{forestsage}{\textbf{Admitted Episodic Candidate}}};

% Connecting Arrow from Path 1 WAL to Tier 3 (asynchronous feed)
\draw[->, line width=1.4pt, dashed, BlueLine] (path1_box.south -| 4.9, 0) -- (consolidation_box.north -| 4.9, 0)
  node[midway, right=4pt] {\fontsize{7.5}{9.5}\selectfont \textcolor{BlueLine}{\textbf{Async WAL Log Feeder}}};

% -------------------------------------------------------------
% TIER 4: COMPARATIVE ARCHITECTURAL AUDIT CARDS
% Dynamically anchored below consolidation_box
% -------------------------------------------------------------
\node[anchor=north west] (audit_title) at ([yshift=-0.6cm] 0, 0 |- consolidation_box.south) {
  {\fontsize{10.5}{13}\selectfont \textbf{\textcolor{DarkSlate}{Architectural Contrast: Naive Synchronous Coupling vs. Dual Write Path Architecture}}}
};

% Card Left: Naive Synchronous Architecture (The Antipattern)
\node[anchor=north west, fill=RedL0, draw=RedLine, line width=1.0pt, rounded corners=4pt, inner sep=6pt] 
  (card_naive) at ([yshift=-0.35cm] 0, 0 |- audit_title.south) {
  \begin{minipage}{11.7cm}
    {\fontsize{9.5}{12}\selectfont \textbf{\textcolor{deepburgundy}{\texttimes\ Naive Synchronous Memory Coupling (Antipattern)}}}\\[3pt]
    {\fontsize{8}{10.5}\selectfont
    \textbf{Mechanism:} Mutates vector indexes and runs LLM reflection inside turn loop.\\[3pt]
    \textbf{Catastrophic Systems Consequences:}
    \begin{itemize}\setlength{\itemsep}{1.5pt}\setlength{\leftmargin}{10pt}
      \item \textbf{Severe Latency Wall:} $T_{\text{turn}} = T_{\text{infer}} + T_{\text{embed}} + T_{\text{HNSW}} + T_{\text{reflect}} > \mathbf{5.2\text{ seconds}}$.
      \item \textbf{Serving Queue Starvation:} Locks accelerator batch slots during memory RPCs.
      \item \textbf{Memory Self-Poisoning:} Routine command outputs, syntax retries, and duplicate polling pollute dense vector space, diluting top-$K$ precision.
      \item \textbf{Credential Leakage:} Ingests secrets and API keys directly into searchable embeddings.
    \end{itemize}
    }
  \end{minipage}
};

% Card Right: Dual Write Path Architecture (MLSysBook Standard)
\node[anchor=north west, fill=GreenL0, draw=GreenLine, line width=1.1pt, rounded corners=4pt, inner sep=6pt] 
  (card_dual) at ([yshift=-0.35cm] 12.4, 0 |- audit_title.south) {
  \begin{minipage}{11.8cm}
    {\fontsize{9.5}{12}\selectfont \textbf{\textcolor{forestsage}{\checkmark\ Dual Write Path \& Admission Quarantine (Production Standard)}}}\\[3pt]
    {\fontsize{8}{10.5}\selectfont
    \textbf{Mechanism:} Bifurcates append-only WAL logging from out-of-band reflection.\\[3pt]
    \textbf{Architectural Systems Guarantees:}
    \begin{itemize}\setlength{\itemsep}{1.5pt}\setlength{\leftmargin}{10pt}
      \item \textbf{Sub-Millisecond Critical Path:} WAL append finishes in $\mathbf{0.45\text{ ms}}$ ($< 1\text{ ms}$). Zero GPU stalls.
      \item \textbf{Surprisal Sieve:} Filters $99.4\%$ of repetitive traces, eliminating distractor bloat.
      \item \textbf{Hardware Zero-Staleness:} In-memory tombstoning bitset ($400\text{ ns}$) ensures stale code chunks are ignored immediately.
      \item \textbf{Strict Invariant Closure:} Only deterministically verified lessons (exit code 0, test pass) enter durable long-term memory.
    \end{itemize}
    }
  \end{minipage}
};

\end{tikzpicture}
\end{document}
"""

repo_root = Path(__file__).resolve().parents[2]
tex_file = repo_root / "books/vol3/tex/dual_write_path_quarantine.tex"

with open(tex_file, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-halt-on-error", "-output-directory=/tmp", str(tex_file)], capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/dual_write_path_quarantine.pdf", "/tmp/dual_write_path_render"])
    subprocess.run(["pdftocairo", "-svg", "/tmp/dual_write_path_quarantine.pdf", "/tmp/dual_write_path_quarantine.svg"])
    print("Successfully compiled and rendered Dual Write Path Quarantine Diagram!")
else:
    print(res.stdout[-1500:])
