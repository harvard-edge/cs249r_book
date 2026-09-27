#!/usr/bin/env python3
"""
Build and render Bernstein Hazard Dependency DAG Scheduling Diagram for Chapter 07.
Follows strict MLSysBook publication standards:
- Beautiful multi-tier layout with mathematical rigor
- Visual inspection via pdftoppm + view_file
- Zero collisions or clipping
"""

import subprocess
from pathlib import Path

TIKZ_CODE = r"""\documentclass[tikz,border=22pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb}
\usetikzlibrary{calc,positioning,fit,backgrounds,shapes.geometric,shapes.symbols,arrows.meta,decorations.pathreplacing}

% =============================================================================
% CANONICAL MLSYSBOOK PALETTE
% =============================================================================
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{slatesteel}{HTML}{334155}
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
\definecolor{VioletLine}{HTML}{7E317B}
\definecolor{VioletL}{RGB}{247,215,247}
\definecolor{amethyst}{HTML}{581C87}
\definecolor{amethystL0}{HTML}{FAF5FF}
\definecolor{GrayLine}{HTML}{64748B}
\definecolor{coolparchment}{HTML}{F8FAFC}
\definecolor{CardBorder}{HTML}{CBD5E1}
\definecolor{ChipBlack}{HTML}{181F2C}
\definecolor{GoldPin}{HTML}{EAB308}

\tikzset{
  >=Latex,
  font=\sffamily,
  flow_arrow/.style={->, >=Latex, line width=1.1pt},
  flow_arrow_thick/.style={->, >=Latex, line width=1.5pt},
  hazard_arrow/.style={->, >=Latex, line width=1.1pt, draw=deepburgundy, dashed},
}

\begin{document}
\begin{tikzpicture}[x=1cm, y=1cm]

% =============================================================================
% TOP TITLE BANNER (y: 20.30 to 21.70)
% =============================================================================
\draw[fill=DarkSlate, draw=none, rounded corners=4pt] (0, 20.30) rectangle (24.0, 21.70);
\node[anchor=west, font=\bfseries\large, text=white] at (0.45, 21.15) {
  BERNSTEIN HAZARD ANALYSIS \& DEPENDENCY DAG DISPATCH SCHEDULING
};
\node[anchor=east, font=\footnotesize, text=GoldPin] at (23.55, 21.15) {
  Chapter 07: Tool Calling $\cdot$ Section 7.1
};
\node[anchor=west, font=\scriptsize, text=white!85] at (0.45, 20.65) {
  Compiling Stochastic Multi-Tool Token Emissions into Deterministic, Race-Free Concurrent Execution Waves
};

% =============================================================================
% TIER 1: EMISSION BATCH & FORMAL RESOURCE SET EXTRACTION (y: 15.20 to 20.05)
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.0pt, rounded corners=4pt] (0, 15.20) rectangle (24.0, 20.05);

\fill[DarkSlate, rounded corners=2pt] (0.2, 19.55) rectangle (23.8, 19.90);
\node[font=\bfseries\fontsize{6.2pt}{7.2pt}\selectfont, text=white] at (12.0, 19.72) {
  1. UNPRIVILEGED MULTI-TOOL TOKEN EMISSION \& DETERMINISTIC RESOURCE SET EXTRACTION
};

% Left Panel: Emitted Batch from Model (x: 0.35 to 8.90)
\draw[draw=CardBorder, fill=white, rounded corners=3pt] (0.35, 15.35) rectangle (8.90, 19.40);
\node[anchor=north west, font=\bfseries\fontsize{5.2pt}{6.2pt}\selectfont, text=BlueLine] at (0.50, 19.25) {Single-Turn Multi-Tool Emission Batch $\mathcal{T} = \{t_1, t_2, t_3, t_4, t_5\}$};
\node[anchor=north west, font=\fontsize{4.0pt}{5.0pt}\selectfont, text=slatesteel] at (0.50, 18.82) {Model outputs 5 function proposals in sequential autoregressive token order:};

% 5 Tool Proposal Rows (Safe vertical spacing, top starts at 18.25)
\draw[draw=BlueLine, fill=BlueL0, rounded corners=1.5pt] (0.50, 17.75) rectangle (8.75, 18.25);
\node[anchor=west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=BlueLine] at (0.60, 18.00) {$t_1$: \texttt{fs\_read(path="/src/auth.py")}};
\node[anchor=east, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (8.65, 18.00) {Inspect auth config};

\draw[draw=BlueLine, fill=BlueL0, rounded corners=1.5pt] (0.50, 17.20) rectangle (8.75, 17.70);
\node[anchor=west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=BlueLine] at (0.60, 17.45) {$t_2$: \texttt{db\_query(table="users", uid=42)}};
\node[anchor=east, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (8.65, 17.45) {Fetch role credentials};

\draw[draw=warmochre, fill=OrangeL0, rounded corners=1.5pt] (0.50, 16.65) rectangle (8.75, 17.15);
\node[anchor=west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=warmochre] at (0.60, 16.90) {$t_3$: \texttt{fs\_write(path="/src/auth.py", ...)}};
\node[anchor=east, font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=warmochre] at (8.65, 16.90) {MUTATION: Patch code};

\draw[draw=forestsage, fill=GreenL0, rounded corners=1.5pt] (0.50, 16.10) rectangle (8.75, 16.60);
\node[anchor=west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=forestsage] at (0.60, 16.35) {$t_4$: \texttt{slack\_notify(ch="\#ops", ...)}};
\node[anchor=east, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (8.65, 16.35) {Broadcast alert};

\draw[draw=amethyst, fill=amethystL0, rounded corners=1.5pt] (0.50, 15.55) rectangle (8.75, 16.05);
\node[anchor=west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=amethyst] at (0.60, 15.80) {$t_5$: \texttt{test\_run(target="/tests/auth\_test.py")}};
\node[anchor=east, font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=amethyst] at (8.65, 15.80) {Run unit test suite};

% Arrow connecting Left to Right
\draw[flow_arrow, draw=slatesteel] (8.90, 17.35) -- (9.25, 17.35);

% Right Panel: Semantic Validation & Resource Set Extraction (x: 9.25 to 23.65)
\draw[draw=CardBorder, fill=white, rounded corners=3pt] (9.25, 15.35) rectangle (23.65, 19.40);
\node[anchor=north west, font=\bfseries\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (9.40, 19.25) {Host Semantic Parser: Deterministic Read / Write Set Extraction};
\node[anchor=north west, font=\fontsize{4.0pt}{5.0pt}\selectfont, text=slatesteel] at (9.40, 18.82) {Extracts explicit filesystem paths, database keys, and external endpoints from tool argument ASTs:};

% Resource Set Table Cards
% Row 1: t1, t2, t4 (starts at 18.25 down to 17.05)
\draw[draw=BlueLine, fill=BlueL0, rounded corners=2pt] (9.40, 17.05) rectangle (13.90, 18.25);
\node[anchor=north west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=BlueLine] at (9.50, 18.12) {$t_1$: \texttt{fs\_read} [READ]};
\node[anchor=north west, font=\fontsize{3.9pt}{4.9pt}\selectfont, text=DarkSlate] at (9.50, 17.72) {$\mathcal{I}(t_1) = \{\mathbf{path: \text{/src/auth.py}}\}$};
\node[anchor=north west, font=\fontsize{3.7pt}{4.7pt}\selectfont, text=slatesteel] at (9.50, 17.32) {$\mathcal{O}(t_1) = \emptyset$ (Side-effect free)};

\draw[draw=BlueLine, fill=BlueL0, rounded corners=2pt] (14.10, 17.05) rectangle (18.60, 18.25);
\node[anchor=north west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=BlueLine] at (14.20, 18.12) {$t_2$: \texttt{db\_query} [READ]};
\node[anchor=north west, font=\fontsize{3.9pt}{4.9pt}\selectfont, text=DarkSlate] at (14.20, 17.72) {$\mathcal{I}(t_2) = \{\mathbf{db: \text{users[42]}}\}$};
\node[anchor=north west, font=\fontsize{3.7pt}{4.7pt}\selectfont, text=slatesteel] at (14.20, 17.32) {$\mathcal{O}(t_2) = \emptyset$ (Pure read)};

\draw[draw=forestsage, fill=GreenL0, rounded corners=2pt] (18.80, 17.05) rectangle (23.50, 18.25);
\node[anchor=north west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=forestsage] at (18.90, 18.12) {$t_4$: \texttt{slack\_notify} [EXTERNAL]};
\node[anchor=north west, font=\fontsize{3.9pt}{4.9pt}\selectfont, text=DarkSlate] at (18.90, 17.72) {$\mathcal{I}(t_4) = \emptyset$};
\node[anchor=north west, font=\fontsize{3.7pt}{4.7pt}\selectfont, text=forestsage] at (18.90, 17.32) {$\mathcal{O}(t_4) = \{\mathbf{slack: \text{\#ops}}\}$ (Disjoint sink)};

% Row 2: t3 and t5 (starts at 16.90 down to 15.36, with tuned line steps)
\draw[draw=warmochre, fill=OrangeL0, rounded corners=2pt] (9.40, 15.36) rectangle (16.30, 16.90);
\node[anchor=north west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=warmochre] at (9.50, 16.78) {$t_3$: \texttt{fs\_write} [MUTATION]};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (9.50, 16.42) {$\mathcal{I}(t_3) = \emptyset$};
\node[anchor=north west, font=\fontsize{3.9pt}{4.9pt}\selectfont, text=deepburgundy] at (9.50, 16.08) {$\mathcal{O}(t_3) = \{\mathbf{path: \text{/src/auth.py}}\}$};
\node[anchor=north west, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, text=deepburgundy] at (9.50, 15.74) {Collides with $t_1$ (WAR Hazard) and $t_5$ (RAW Hazard)!};

\draw[draw=amethyst, fill=amethystL0, rounded corners=2pt] (16.50, 15.36) rectangle (23.50, 16.90);
\node[anchor=north west, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, text=amethyst] at (16.60, 16.78) {$t_5$: \texttt{test\_run} [DEPENDENT VERIFICATION]};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (16.60, 16.42) {$\mathcal{I}(t_5) = \{\mathbf{path: \text{/src/auth.py}}, \mathbf{path: \text{/tests/auth\_test.py}}\}$};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (16.60, 16.08) {$\mathcal{O}(t_5) = \{\mathbf{env: \text{test\_exit\_code}}\}$};
\node[anchor=north west, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, text=amethyst] at (16.60, 15.74) {Must execute strictly AFTER $t_3$ commits patch!};

% =============================================================================
% TIER 2: BERNSTEIN HAZARD MATRIX & CONDITION VERIFICATION (y: 10.30 to 14.95)
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.0pt, rounded corners=4pt] (0, 10.30) rectangle (24.0, 14.95);

\fill[DarkSlate, rounded corners=2pt] (0.2, 14.50) rectangle (23.8, 14.85);
\node[font=\bfseries\fontsize{6.2pt}{7.2pt}\selectfont, text=white] at (12.0, 14.67) {
  2. CLASSICAL BERNSTEIN HAZARD THEOREM \& CONCURRENCY INTERSECTION ANALYSIS
};

% Left Box: Mathematical Conditions (x: 0.35 to 8.90)
\draw[draw=CardBorder, fill=white, rounded corners=3pt] (0.35, 10.45) rectangle (8.90, 14.35);
\node[anchor=north west, font=\bfseries\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (0.50, 14.20) {Bernstein's Concurrency Equivalence Conditions};
\node[anchor=north west, font=\fontsize{3.9pt}{4.9pt}\selectfont, text=slatesteel] at (0.50, 13.80) {Operations $t_j, t_k$ ($j < k$) are concurrent ($t_j \parallel t_k$) iff all 3 hold:};

% 3 Condition Sub-Blocks (Starts at 13.35 with generous clearance)
\draw[draw=CardBorder, fill=coolparchment, rounded corners=1.5pt] (0.50, 12.45) rectangle (8.75, 13.35);
\node[anchor=north west, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=deepburgundy] at (0.60, 13.23) {1. Write-After-Read (WAR) Invariant:};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (0.60, 12.83) {$\mathcal{I}(t_j) \cap \mathcal{O}(t_k) = \emptyset \quad \implies \text{Prevents writes from corrupting concurrent reads}$};

\draw[draw=CardBorder, fill=coolparchment, rounded corners=1.5pt] (0.50, 11.45) rectangle (8.75, 12.35);
\node[anchor=north west, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=deepburgundy] at (0.60, 12.23) {2. Read-After-Write (RAW) Invariant:};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (0.60, 11.83) {$\mathcal{O}(t_j) \cap \mathcal{I}(t_k) = \emptyset \quad \implies \text{Prevents reads from observing stale, uncommitted states}$};

\draw[draw=CardBorder, fill=coolparchment, rounded corners=1.5pt] (0.50, 10.50) rectangle (8.75, 11.35);
\node[anchor=north west, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=deepburgundy] at (0.60, 11.23) {3. Write-After-Write (WAW) Invariant:};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (0.60, 10.83) {$\mathcal{O}(t_j) \cap \mathcal{O}(t_k) = \emptyset \quad \implies \text{Prevents blind overwrites and non-deterministic final state}$};

% Right Box: Pairwise Hazard Evaluation Matrix (x: 9.25 to 23.65)
\draw[draw=CardBorder, fill=white, rounded corners=3pt] (9.25, 10.45) rectangle (23.65, 14.35);
\node[anchor=north west, font=\bfseries\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (9.40, 14.20) {Pairwise Batch Hazard Resolution Matrix};
\node[anchor=north west, font=\fontsize{3.9pt}{4.9pt}\selectfont, text=slatesteel] at (9.40, 13.80) {Automated intersection mapping establishes causal execution edges in the dispatch DAG:};

% 4 Hazard Outcome Cards (Starts at 13.35 with generous clearance)
% Card 1: WAR Hazard
\draw[draw=deepburgundy, fill=RedL0, rounded corners=2pt] (9.40, 10.55) rectangle (12.75, 13.35);
\node[draw=none, fill=deepburgundy, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.2pt}{5.2pt}\selectfont, inner sep=1.2pt] at (11.07, 13.15) {WAR HAZARD: $t_1 \to t_3$};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (9.50, 12.85) {
  \begin{minipage}{3.05cm}\raggedright
    $\mathcal{I}(t_1) \cap \mathcal{O}(t_3) = \{\text{\texttt{/src/auth.py}}\}$\\
    \vspace{0.15cm}
    \textbf{Conflict:} $t_3$ mutates what $t_1$ is actively reading!\\
    \vspace{0.15cm}
    \textbf{Resolution:} $t_1$ MUST finish before $t_3$ dispatches.
  \end{minipage}
};

% Card 2: RAW Hazard
\draw[draw=deepburgundy, fill=RedL0, rounded corners=2pt] (12.95, 10.55) rectangle (16.30, 13.35);
\node[draw=none, fill=deepburgundy, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.2pt}{5.2pt}\selectfont, inner sep=1.2pt] at (14.62, 13.15) {RAW HAZARD: $t_3 \to t_5$};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (13.05, 12.85) {
  \begin{minipage}{3.05cm}\raggedright
    $\mathcal{O}(t_3) \cap \mathcal{I}(t_5) = \{\text{\texttt{/src/auth.py}}\}$\\
    \vspace{0.15cm}
    \textbf{Conflict:} $t_5$ tests code patched by $t_3$!\\
    \vspace{0.15cm}
    \textbf{Resolution:} $t_3$ MUST commit before $t_5$ runs.
  \end{minipage}
};

% Card 3: Disjoint / Independent
\draw[draw=forestsage, fill=GreenL0, rounded corners=2pt] (16.50, 10.55) rectangle (19.90, 13.35);
\node[draw=none, fill=forestsage, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.2pt}{5.2pt}\selectfont, inner sep=1.2pt] at (18.20, 13.15) {INDEPENDENT: $t_1 \parallel t_2$};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (16.60, 12.85) {
  \begin{minipage}{3.10cm}\raggedright
    $\mathcal{I}(t_1) \cap \mathcal{O}(t_2) = \emptyset$\\
    $\mathcal{O}(t_1) \cap \mathcal{I}(t_2) = \emptyset$\\
    $\mathcal{O}(t_1) \cap \mathcal{O}(t_2) = \emptyset$\\
    \vspace{0.10cm}
    \textbf{Resolution:} Pure reads on disjoint targets $\implies$ Concurrent!
  \end{minipage}
};

% Card 4: Disjoint Asynchronous
\draw[draw=forestsage, fill=GreenL0, rounded corners=2pt] (20.10, 10.55) rectangle (23.50, 13.35);
\node[draw=none, fill=forestsage, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.2pt}{5.2pt}\selectfont, inner sep=1.2pt] at (21.80, 13.15) {DISJOINT: $t_4 \parallel \{t_1, t_2, t_3, t_5\}$};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (20.20, 12.85) {
  \begin{minipage}{3.10cm}\raggedright
    $\mathcal{I}(t_4) \cap \mathcal{O}(t_k) = \emptyset$\\
    $\mathcal{O}(t_4) \cap \mathcal{I}(t_k) = \emptyset$\\
    \vspace{0.15cm}
    \textbf{Resolution:} Slack API is isolated external sink $\implies$ Concurrent!
  \end{minipage}
};

% =============================================================================
% TIER 3: EXECUTION DATAPATH: NAIVE CONCURRENCY vs. COMPILED DAG (y: 5.10 to 10.05)
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.0pt, rounded corners=4pt] (0, 5.10) rectangle (24.0, 10.05);

\fill[DarkSlate, rounded corners=2pt] (0.2, 9.60) rectangle (23.8, 9.95);
\node[font=\bfseries\fontsize{6.2pt}{7.2pt}\selectfont, text=white] at (12.0, 9.77) {
  3. EXECUTION ENGINE DATAPATH: NAIVE \texttt{asyncio.gather} DATA RACES vs. COMPILED DAG BARRIER SCHEDULING
};

% Left Panel: Naive Async Dispatch (x: 0.35 to 9.80)
\draw[draw=deepburgundy, fill=white, rounded corners=3pt, line width=0.8pt] (0.35, 5.25) rectangle (9.80, 9.45);
\node[draw=none, fill=deepburgundy, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, inner sep=1.5pt] at (5.07, 9.25) {
  NAIVE ASYNCHRONOUS GATHER (\texttt{asyncio.gather(*tools)})
};

% Naive Timeline Box
\draw[->, line width=0.7pt, draw=slatesteel] (0.55, 5.50) -- (9.60, 5.50);
\node[anchor=west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (9.30, 5.35) {Time $t$};

% 5 tools launched concurrently at t=0
\draw[draw=BlueLine, fill=BlueL0, rounded corners=1.5pt] (0.60, 8.35) rectangle (3.20, 8.85);
\node[font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=BlueLine] at (1.90, 8.60) {$t_1$: \texttt{fs\_read} ($60\text{ms}$)};

\draw[draw=BlueLine, fill=BlueL0, rounded corners=1.5pt] (0.60, 7.75) rectangle (4.40, 8.25);
\node[font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=BlueLine] at (2.50, 8.00) {$t_2$: \texttt{db\_query} ($110\text{ms}$)};

\draw[draw=deepburgundy, fill=RedL0, rounded corners=1.5pt, line width=0.8pt] (0.60, 7.15) rectangle (3.90, 7.65);
\node[font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=deepburgundy] at (2.25, 7.40) {$t_3$: \texttt{fs\_write} ($95\text{ms}$)};

\draw[draw=forestsage, fill=GreenL0, rounded corners=1.5pt] (0.60, 6.55) rectangle (2.80, 7.05);
\node[font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=forestsage] at (1.70, 6.80) {$t_4$: \texttt{slack} ($50\text{ms}$)};

\draw[draw=deepburgundy, fill=RedL0, rounded corners=1.5pt, line width=0.8pt] (0.60, 5.95) rectangle (5.20, 6.45);
\node[font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=deepburgundy] at (2.90, 6.20) {$t_5$: \texttt{test\_run} ($160\text{ms}$)};

% Callout Box for Naive Data Races on the right of tasks
\draw[draw=deepburgundy, fill=RedL0, rounded corners=2pt] (4.80, 6.80) rectangle (9.60, 8.85);
\node[anchor=north west, font=\bfseries\fontsize{4.2pt}{5.2pt}\selectfont, text=deepburgundy] at (4.90, 8.70) {FATAL CONCURRENCY RACES:};
\node[anchor=north west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (4.90, 8.35) {
  \begin{minipage}{4.4cm}\raggedright
    $\bullet$ \textbf{WAR Race ($t_1 \times t_3$):} $t_3$ mutates \texttt{auth.py} while $t_1$ is reading $\implies$ torn buffer / syntax crash!\\
    $\bullet$ \textbf{RAW Race ($t_3 \times t_5$):} $t_5$ tests stale code before $t_3$ commits patch $\implies$ flaky false failure!\\
    $\bullet$ \textbf{Result:} Non-deterministic failures!
  \end{minipage}
};

% Connecting bracket indicators on the left
\draw[draw=deepburgundy, line width=0.8pt] (4.00, 7.40) -- (4.65, 7.40);
\draw[draw=deepburgundy, line width=0.8pt] (3.30, 8.60) -- (4.65, 8.60);
\draw[draw=deepburgundy, line width=0.8pt] (4.65, 7.40) -- (4.65, 8.60);
\node[anchor=east, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, text=deepburgundy] at (4.60, 8.00) {WAR};

% Right Panel: Dynamic DAG Scheduling (x: 10.10 to 23.65)
\draw[draw=forestsage, fill=white, rounded corners=3pt, line width=0.8pt] (10.10, 5.25) rectangle (23.65, 9.45);
\node[draw=none, fill=forestsage, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.4pt}{5.4pt}\selectfont, inner sep=1.5pt] at (16.87, 9.25) {
  COMPILED DEPENDENCY DAG WITH SYNCHRONIZATION BARRIERS ($\mathcal{G}_{\text{dispatch}}$)
};

% Sub-DAG Diagram (Left side of right panel, x: 10.25 to 14.80)
\node[draw=BlueLine, fill=BlueL0, rounded corners=2pt, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=BlueLine, inner sep=2pt] (dag_t1) at (10.90, 8.50) {$t_1$: \texttt{fs\_read}};
\node[draw=BlueLine, fill=BlueL0, rounded corners=2pt, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=BlueLine, inner sep=2pt] (dag_t2) at (10.90, 7.40) {$t_2$: \texttt{db\_query}};
\node[draw=forestsage, fill=GreenL0, rounded corners=2pt, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=forestsage, inner sep=2pt] (dag_t4) at (10.90, 6.30) {$t_4$: \texttt{slack}};

\node[draw=warmochre, fill=OrangeL0, rounded corners=2pt, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=warmochre, inner sep=2pt] (dag_t3) at (12.70, 8.50) {$t_3$: \texttt{fs\_write}};
\node[draw=amethyst, fill=amethystL0, rounded corners=2pt, font=\bfseries\fontsize{4.0pt}{5.0pt}\selectfont, text=amethyst, inner sep=2pt] (dag_t5) at (14.25, 8.50) {$t_5$: \texttt{test\_run}};

% DAG Edges
\draw[flow_arrow, draw=deepburgundy] (dag_t1) -- node[above, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, text=deepburgundy] {WAR} (dag_t3);
\draw[flow_arrow, draw=deepburgundy] (dag_t3) -- node[above, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, text=deepburgundy] {RAW} (dag_t5);

% Timeline Execution with Barriers (Right side of right panel, x: 15.00 to 23.50)
\draw[->, line width=0.7pt, draw=slatesteel] (15.10, 5.50) -- (23.40, 5.50);
\node[anchor=west, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=slatesteel] at (23.10, 5.35) {Time $t$};

% --- BARRIER 1 & BARRIER 2 DRAWN FIRST (So t2 naturally passes over Barrier 1!) ---
% Barrier 1: WAR Synchronization Fence
\draw[draw=deepburgundy, line width=0.9pt, dashed] (17.30, 5.50) -- (17.30, 8.70);
\node[draw=none, fill=deepburgundy, rounded corners=1.2pt, text=white, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, inner sep=1.2pt] at (17.30, 8.90) {BARRIER 1: Wait $t_1$};

% Barrier 2: RAW Commit Fence
\draw[draw=forestsage, line width=0.9pt, dashed] (20.00, 5.50) -- (20.00, 8.70);
\node[draw=none, fill=forestsage, rounded corners=1.2pt, text=white, font=\bfseries\fontsize{3.6pt}{4.6pt}\selectfont, inner sep=1.2pt] at (20.00, 8.90) {BARRIER 2: Flush $t_3$};

% Wave 0 (t = 0 to 60ms)
\draw[draw=BlueLine, fill=BlueL0, rounded corners=1.5pt] (15.20, 8.00) rectangle (17.10, 8.55);
\node[font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=BlueLine] at (16.15, 8.28) {$t_1$: \texttt{fs\_read}};

% t2 spans across Barrier 1! Drawn on top so BlueL0 cleanly overlays the barrier line!
\draw[draw=BlueLine, fill=BlueL0, rounded corners=1.5pt] (15.20, 7.30) rectangle (18.10, 7.85);
\node[font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=BlueLine] at (16.65, 7.58) {$t_2$: \texttt{db\_query} ($110\text{ms}$) [Non-blocking]};

\draw[draw=forestsage, fill=GreenL0, rounded corners=1.5pt] (15.20, 6.60) rectangle (17.00, 7.15);
\node[font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=forestsage] at (16.10, 6.88) {$t_4$: \texttt{slack}};

% Wave 1: Exclusive Mutation (t3, 95ms)
\draw[draw=warmochre, fill=OrangeL0, rounded corners=1.5pt] (17.40, 8.00) rectangle (19.90, 8.55);
\node[font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=warmochre] at (18.65, 8.28) {$t_3$: \texttt{fs\_write} ($95\text{ms}$)};

% Wave 2: Dependent Verification (t5, 160ms)
\draw[draw=amethyst, fill=amethystL0, rounded corners=1.5pt] (20.10, 8.00) rectangle (23.30, 8.55);
\node[font=\bfseries\fontsize{3.8pt}{4.8pt}\selectfont, text=amethyst] at (21.70, 8.28) {$t_5$: \texttt{test\_run} ($160\text{ms}$)};

% Guaranteed Correctness Badge on bottom right
\node[draw=none, fill=forestsage, rounded corners=1.5pt, text=white, font=\bfseries\fontsize{4.2pt}{5.2pt}\selectfont, inner sep=1.8pt] at (21.20, 7.15) {
  $100\%$ RACE-FREE DISPATCH
};
\node[anchor=north, font=\fontsize{3.8pt}{4.8pt}\selectfont, text=DarkSlate] at (21.20, 6.85) {
  \begin{minipage}{3.8cm}\centering
    Optimal Makespan: $315\text{ ms}$\\
    Zero Stale Reads $\cdot$ Deterministic
  \end{minipage}
};

% =============================================================================
% TIER 4: SYSTEMS ARCHITECTURAL COMPARISON (y: 0.50 to 4.80)
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.1pt, rounded corners=4pt] (0, 0.50) rectangle (24.0, 4.80);

\fill[DarkSlate, rounded corners=2pt] (0.2, 4.35) rectangle (23.8, 4.70);
\node[font=\bfseries\fontsize{6.8pt}{8.2pt}\selectfont, text=white] at (12.0, 4.52) {
  THE SYSTEMS ARCHITECTURAL CONTRAST: UNCHECKED CONCURRENCY vs. BERNSTEIN DAG MEDIATION
};

% Left Box: Unconstrained Concurrency (x: 0.35 to 11.65)
\draw[draw=deepburgundy, fill=RedL0, rounded corners=3pt, line width=0.8pt] (0.35, 0.70) rectangle (11.65, 4.20);
\node[draw=none, fill=deepburgundy, rounded corners=2pt, inner xsep=3pt, inner ysep=1.5pt, font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=white] at (3.5, 3.98) {
  UNCONSTRAINED ASYNC DISPATCH (\texttt{asyncio.gather})
};

\node[anchor=north west, font=\fontsize{4.8pt}{6.1pt}\selectfont, text=DarkSlate] at (0.5, 3.70) {
  \begin{minipage}{10.8cm}\raggedright
    $\bullet$ \textbf{Data Race Inevitability:} Dispatches all emitted calls concurrently at $t=0$, causing fatal RAW, WAR, and WAW memory and filesystem corruption.\\
    $\bullet$ \textbf{Torn Buffer Reads:} Concurrent read/write collisions yield partially modified files, leading to syntax errors and unhandled deserialization crashes.\\
    $\bullet$ \textbf{Flaky Epistemic Falsification:} Validation tools (linters, test suites) execute against stale pre-mutation state, misinforming subsequent reasoning steps.\\
    $\bullet$ \textbf{No Fault Containment:} An unhandled exception in one coroutine often cancels the entire batch, discarding successful sibling observations.
  \end{minipage}
};

% Right Box: Bernstein DAG Dispatch Engine (x: 12.00 to 23.65)
\draw[draw=forestsage, fill=GreenL0, rounded corners=3pt, line width=0.8pt] (12.0, 0.70) rectangle (23.65, 4.20);
\node[draw=none, fill=forestsage, rounded corners=2pt, inner xsep=3pt, inner ysep=1.5pt, font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=white] at (15.5, 3.98) {
  BERNSTEIN-MEDIATED DAG SCHEDULER (HOST SUPERVISOR)
};

\node[anchor=north west, font=\fontsize{4.8pt}{6.1pt}\selectfont, text=DarkSlate] at (12.15, 3.70) {
  \begin{minipage}{11.1cm}\raggedright
    $\bullet$ \textbf{Deterministic Hazard Extraction:} Formally evaluates read/write set intersections $\mathcal{I}(t_j) \cap \mathcal{O}(t_k)$ before dispatching any physical kernel syscall.\\
    $\bullet$ \textbf{Optimal Wave Parallelization:} Dynamically compiles disjoint operations ($t_1 \parallel t_2 \parallel t_4$) into concurrent I/O waves while serializing conflicting mutations.\\
    $\bullet$ \textbf{Zero Critical-Path Penalty:} Independent reads and side effects run concurrently in Wave 0, maximizing system goodput without violating serial semantics.\\
    $\bullet$ \textbf{Per-Call Fault Containment:} Each outcome binds to its unique call ID ($t_k$); individual timeouts or exceptions are isolated without aborting sibling DAG branches.
  \end{minipage}
};

\end{tikzpicture}
\end{document}
"""

repo_root = Path(__file__).resolve().parents[2]
tex_file = repo_root / "books/vol3/tex/tool_dependency_dag_scheduling.tex"

with open(tex_file, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-halt-on-error", "-output-directory=/tmp", str(tex_file)], capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/tool_dependency_dag_scheduling.pdf", "/tmp/tool_dag_render"])
    subprocess.run(["pdftocairo", "-svg", "/tmp/tool_dependency_dag_scheduling.pdf", "/tmp/tool_dependency_dag_scheduling.svg"])
    print("Successfully compiled and rendered Tool DAG Scheduling Diagram!")
else:
    print(res.stdout[-1500:])
