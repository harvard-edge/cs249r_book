#!/usr/bin/env python3
"""
Build publication-grade schematic for Chapter 19:
The Unified Trajectory Spine & Specification Boundary (fig-vol3-trajectory-lifecycle)

Standards:
- Canonical academic palette (OxfordNavy, SlateSteel, MutedTeal, Burgundy, WarmOchre, ForestSage, Amethyst)
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy (8.0pt badges, 6.5pt components, 5.0pt metrics/signals)
- Authentic IEEE/ACM computer systems architecture glyphs:
  - Specification Boundary & Task Contract Descriptor (C = <G, E, T_allow, Sigma_obs, V_done>)
  - Monotonic Capability Quota Registers (H <= H_max, T <= T_max, M <= M_max)
  - 5-Stage Live Trajectory Execution Pipeline (Context -> Model -> Gate -> Sandbox -> Sagas)
  - Memory Hierarchy with Radix Tree Prefix Cache & Taint Tracking
  - GPU Pushdown Automata Grammars & Action Escrow (a_t)
  - Macaroon Cryptographic Lease & Sync WAL fsync Barrier
  - MicroVM Sandbox with seccomp-bpf, netns isolation, and Ephemeral CoW upperdir
  - Passive Observation Return Bus (o_{t+1})
  - Sealed 4-Layer Verification Oracle Ladder (Schema -> Exit Code 0 -> State Delta -> SLA)
  - Offline Trajectory Recompiler (Part V) with Action-Targeted Loss & SPRT Canary Rollout
  - Fleet Economic Governance (Part VI) with Cost-per-Accepted-Task Frontier
  - Architectural Synthesis Invariants (H-S-A Conservation Laws)
- Zero bullet points, zero paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys
import shutil
import re

TEX_PATH = "books/vol3/tex/unified_trajectory_spine.tex"
PNG_DIR = "books/vol3/19_conclusion/images/png"
SVG_DIR = "books/vol3/19_conclusion/images/svg"
PDF_DIR = "books/vol3/19_conclusion/images/pdf"

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

TIKZ_CONTENT = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,fit,arrows.meta,shapes.geometric}

% Canonical Academic Palette
\definecolor{OxfordNavy}{HTML}{1E293B}
\definecolor{SlateSteel}{HTML}{334155}
\definecolor{MutedTeal}{HTML}{0F766E}
\definecolor{TealL0}{HTML}{F0FDFA}
\definecolor{Burgundy}{HTML}{991B1B}
\definecolor{BurgundyL0}{HTML}{FEF2F2}
\definecolor{WarmOchre}{HTML}{92400E}
\definecolor{OchreL0}{HTML}{FFFBEB}
\definecolor{ForestSage}{HTML}{166534}
\definecolor{SageL0}{HTML}{F0FDF4}
\definecolor{CoolParchment}{HTML}{F8FAFC}
\definecolor{SlateRim}{HTML}{CBD5E1}
\definecolor{GridGray}{HTML}{E2E8F0}
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{Amethyst}{HTML}{6B21A8}
\definecolor{AmethystL0}{HTML}{FAF5FF}
\definecolor{CobaltBlue}{HTML}{1D4ED8}
\definecolor{CobaltL0}{HTML}{EFF6FF}

\begin{document}
\begin{tikzpicture}[x=1cm, y=1cm, >=Latex]

% =============================================================================
% TIER 1: THE SPECIFICATION BOUNDARY & TASK CONTRACT (TOP)
% (x: 0 to 24.0, y: 13.9 to 16.8) - Height 2.9 cm
% =============================================================================
\begin{scope}[shift={(0, 13.9)}]
  \draw[fill=CoolParchment, draw=Burgundy, line width=0.9pt, rounded corners=3pt] (0, 0) rectangle (24.0, 2.9);

  % Badge
  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 2.75) {1. THE SPECIFICATION BOUNDARY: FORMAL TASK CONTRACT ($\mathcal{C}$)};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=Burgundy]
    at (23.8, 2.70) {Part VII: Essential Complexity $\cdot$ Authored by Human Systems Engineer};

  % Contract Tuple Descriptor Box (Left: x 0.3 to 11.6, y 0.20 to 2.35)
  \draw[fill=white, draw=Burgundy!40, line width=0.5pt, rounded corners=2pt] (0.3, 0.20) rectangle (11.6, 2.35);
  \node[anchor=north west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.45, 2.18)
    {Formal Specification Tuple: $\mathcal{C} = \langle \mathcal{G}, \mathcal{E}, \mathcal{T}_{\mathrm{allow}}, \Sigma_{\mathrm{obs}}, \mathcal{V}_{\mathrm{done}} \rangle$};

  % Sub-fields
  \draw[fill=BurgundyL0, draw=Burgundy!20, line width=0.3pt, rounded corners=1pt] (0.45, 1.00) rectangle (5.85, 1.80);
  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=DarkSlate] at (0.55, 1.70) {Goal \& Environment:};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\ttfamily, text=SlateSteel] at (0.55, 1.48) {$\mathcal{G} \in \mathcal{S}_{\mathrm{target}} \mid \mathcal{E} \subset \text{Production}$};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\ttfamily, text=ForestSage] at (0.55, 1.25) {$\mathcal{T}_{\mathrm{allow}} = \{\text{read\_log}, \text{query\_metrics}\}$};

  \draw[fill=BurgundyL0, draw=Burgundy!20, line width=0.3pt, rounded corners=1pt] (6.05, 1.00) rectangle (11.45, 1.80);
  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=DarkSlate] at (6.15, 1.70) {Observation \& Gate:};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\ttfamily, text=SlateSteel] at (6.15, 1.48) {$\Sigma_{\mathrm{obs}} \colon \text{JSON Schema (RFC 8259)}$};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\bfseries, text=Burgundy] at (6.15, 1.25) {$\mathcal{V}_{\mathrm{done}} \colon \Delta\mathcal{S}_{\mathrm{ext}} \implies \text{Exit Code } 0$};

  \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont\itshape, text=SlateSteel] at (0.45, 0.55)
    {Conservation of Complexity: Machine automates implementation ($\Delta\text{Accidental} \to 0$); Human defines purpose ($\text{Essential}$).};

  % Quota Registers & Capability Macaroon Box (Center: x 11.9 to 18.8, y 0.20 to 2.35)
  \draw[fill=white, draw=SlateRim, line width=0.5pt, rounded corners=2pt] (11.9, 0.20) rectangle (18.8, 2.35);
  \node[anchor=north west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (12.05, 2.18)
    {Monotonic Quota Registers \& Macaroon Lease Envelope};

  % Registers strip
  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (12.05, 1.10) rectangle (14.15, 1.80);
  \node[anchor=center, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=SlateSteel] at (13.1, 1.60) {TURN CEILING};
  \node[anchor=center, font=\fontsize{3.5}{4.2}\selectfont\ttfamily\bfseries, text=DarkSlate] at (13.1, 1.32) {$H \le H_{\max}$ (25)};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (14.30, 1.10) rectangle (16.40, 1.80);
  \node[anchor=center, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=SlateSteel] at (15.35, 1.60) {WALL-CLOCK};
  \node[anchor=center, font=\fontsize{3.5}{4.2}\selectfont\ttfamily\bfseries, text=DarkSlate] at (15.35, 1.32) {$T \le 300\text{ s}$};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (16.55, 1.10) rectangle (18.65, 1.80);
  \node[anchor=center, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=SlateSteel] at (17.6, 1.60) {TOKEN BUDGET};
  \node[anchor=center, font=\fontsize{3.5}{4.2}\selectfont\ttfamily\bfseries, text=DarkSlate] at (17.6, 1.32) {$M \le 128\text{k tok}$};

  \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont\ttfamily, text=Burgundy] at (12.05, 0.55)
    {Macaroon: $\text{HMAC}_{K_{\mathrm{root}}}(\text{TaskID} \parallel \text{Epoch} \parallel \mathcal{T}_{\mathrm{allow}} \parallel \text{ReadOnly})$};

  % Task Admission Engine (Right: x 19.1 to 23.7, y 0.20 to 2.35)
  \draw[fill=OxfordNavy, draw=none, rounded corners=2pt] (19.1, 0.20) rectangle (23.7, 2.35);
  \node[anchor=north west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=white] at (19.25, 2.18)
    {TASK ADMISSION};
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont, text=TealL0] at (19.25, 1.70)
    {1. Bind cryptographic contract $\mathcal{C}$};
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont, text=TealL0] at (19.25, 1.30)
    {2. Allocate Agent Control Block (ACB)};
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont, text=TealL0] at (19.25, 0.90)
    {3. Sync WAL initialization $\to \tau_0$};
\end{scope}

% Pipeline Dispatch Arrow & Centered Label
\draw[->, draw=OxfordNavy, line width=1.1pt] (21.4, 13.9) -- (21.4, 13.5) -- (1.5, 13.5) -- (1.5, 13.1);
\node[anchor=south, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=OxfordNavy] at (11.5, 13.55)
  {Admit Task $\implies$ Instantiate 256-Byte ACB $\implies$ Dispatch Context Engine};

% =============================================================================
% TIER 2: LIVE EXECUTION LOOP: 5-STAGE TRAJECTORY DATAPATH & HARNESS
% (x: 0 to 24.0, y: 6.7 to 13.1) - Height 6.4 cm
% =============================================================================
\begin{scope}[shift={(0, 6.7)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=0.9pt, rounded corners=3pt] (0, 0) rectangle (24.0, 6.4);

  % Header Badge
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 6.25) {2. LIVE EXECUTION LOOP: 5-STAGE TRAJECTORY STATE BUS ($\tau_t$)};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=DarkSlate]
    at (23.8, 6.20) {Operational Horizon: $10^1\text{--}10^4\text{ s} \cdot$ Zero Ambient Authority $\cdot$ Invariant Closure};

  % ---------------------------------------------------------------------------
  % STAGE 1: CONTEXT MEMORY CONTROLLER (x: 0.4 to 4.6, y: 1.8 to 5.7) - Width 4.2
  % ---------------------------------------------------------------------------
  \draw[fill=white, draw=ForestSage, line width=0.7pt, rounded corners=2pt] (0.4, 1.8) rectangle (4.6, 5.7);
  \node[anchor=north west, fill=ForestSage, text=white, font=\fontsize{3.8}{4.5}\selectfont\bfseries, rounded corners=1pt, inner sep=1.5pt]
    at (0.5, 5.6) {STAGE 1: CONTEXT MEMORY};
  \node[anchor=north east, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=ForestSage] at (4.5, 5.6) {Part II};

  \draw[fill=SageL0, draw=ForestSage!30, line width=0.3pt, rounded corners=1pt] (0.55, 4.40) rectangle (4.45, 5.08);
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (0.65, 5.00) {Radix Prefix Tree Hit:};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\ttfamily, text=ForestSage] at (0.65, 4.68) {$>85\%$ sharing, zero HBM copy};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (0.55, 3.50) rectangle (4.45, 4.30);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (0.65, 4.22) {Static Prefix Leftmost};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (0.65, 3.96) {Prevents Radix page table bust};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=ForestSage] at (0.65, 3.72) {$\Delta\text{TTFT} < 10\text{ ms}$ (vs. $1.7\text{ s}$ stall)};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (0.55, 2.60) rectangle (4.45, 3.40);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (0.65, 3.32) {Provenance Ledger};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=DarkSlate] at (0.65, 3.06) {$\mathcal{P} = \langle \text{src\_hash}, \tau_{\mathrm{trust}} \rangle$};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=Burgundy] at (0.65, 2.82) {Flag untrusted external spans};

  \draw[fill=SageL0, draw=ForestSage!30, line width=0.3pt] (0.55, 1.90) rectangle (4.45, 2.50);
  \node[anchor=north west, font=\fontsize{3.2}{3.8}\selectfont\bfseries, text=ForestSage] at (0.65, 2.42) {Causal Spine Offload:};
  \node[anchor=north west, font=\fontsize{2.9}{3.5}\selectfont\ttfamily, text=DarkSlate] at (0.65, 2.14) {Payloads $>64\text{ KB} \to$ BlobStore};

  % Signal 1 -> 2: Context Buffer c_t
  \draw[->, draw=ForestSage, line width=1.1pt] (4.6, 3.75) -- (5.15, 3.75);
  \node[anchor=south, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=ForestSage] at (4.87, 3.82) {$c_t$};

  % ---------------------------------------------------------------------------
  % STAGE 2: UNPRIVILEGED INFERENCE & DELIBERATION (x: 5.15 to 9.35) - Width 4.2
  % ---------------------------------------------------------------------------
  \draw[fill=white, draw=MutedTeal, line width=0.7pt, rounded corners=2pt] (5.15, 1.8) rectangle (9.35, 5.7);
  \node[anchor=north west, fill=MutedTeal, text=white, font=\fontsize{3.8}{4.5}\selectfont\bfseries, rounded corners=1pt, inner sep=1.5pt]
    at (5.25, 5.6) {STAGE 2: MODEL DELIBERATION};
  \node[anchor=north east, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=MutedTeal] at (9.25, 5.6) {Part I};

  \draw[fill=TealL0, draw=MutedTeal!30, line width=0.3pt, rounded corners=1pt] (5.30, 4.40) rectangle (9.20, 5.08);
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (5.40, 5.00) {Zero Ambient Authority};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (5.40, 4.68) {Stochastic token sampler $\pi_\theta(a \mid c_t)$};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (5.30, 3.50) rectangle (9.20, 4.30);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (5.40, 4.22) {Pushdown Automata (PDA)};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (5.40, 3.96) {GPU decode kernel token mask};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=Burgundy] at (5.40, 3.72) {Schema violation $\implies -\infty$ logit};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (5.30, 2.60) rectangle (9.20, 3.40);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (5.40, 3.32) {Test-Time Deliberation};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (5.40, 3.06) {Speculative tree search rollouts};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=MutedTeal] at (5.40, 2.82) {Prunes invalid candidate paths};

  \draw[fill=TealL0, draw=MutedTeal!30, line width=0.3pt] (5.30, 1.90) rectangle (9.20, 2.50);
  \node[anchor=north west, font=\fontsize{3.2}{3.8}\selectfont\bfseries, text=MutedTeal] at (5.40, 2.42) {Proposal Port (Escrow):};
  \node[anchor=north west, font=\fontsize{2.9}{3.5}\selectfont\ttfamily, text=DarkSlate] at (5.40, 2.14) {Emits proposal $a_t \in \text{Escrow}$};

  % Signal 2 -> 3: Proposal a_t
  \draw[->, draw=MutedTeal, line width=1.1pt] (9.35, 3.75) -- (9.90, 3.75);
  \node[anchor=south, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=MutedTeal] at (9.62, 3.82) {$a_t$};

  % ---------------------------------------------------------------------------
  % STAGE 3: SUPERVISOR CAPABILITY GATE (x: 9.9 to 14.1) - Width 4.2
  % ---------------------------------------------------------------------------
  \draw[fill=white, draw=WarmOchre, line width=0.7pt, rounded corners=2pt] (9.9, 1.8) rectangle (14.1, 5.7);
  \node[anchor=north west, fill=WarmOchre, text=white, font=\fontsize{3.8}{4.5}\selectfont\bfseries, rounded corners=1pt, inner sep=1.5pt]
    at (10.0, 5.6) {STAGE 3: CAPABILITY GATE};
  \node[anchor=north east, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=WarmOchre] at (14.0, 5.6) {Part III/IV};

  \draw[fill=OchreL0, draw=WarmOchre!30, line width=0.3pt, rounded corners=1pt] (10.05, 4.40) rectangle (13.95, 5.08);
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (10.15, 5.00) {Macaroon Lease Check};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (10.15, 4.68) {Attenuations: path, verb, epoch};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (10.05, 3.50) rectangle (13.95, 4.30);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (10.15, 4.22) {Intent-Before-Effect (WAL)};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (10.15, 3.96) {Synchronous \texttt{fsync} to NVMe WAL};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=ForestSage] at (10.15, 3.72) {Crash-consistent state recovery};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (10.05, 2.60) rectangle (13.95, 3.40);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (10.15, 3.32) {Active Idempotency Bridge};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=DarkSlate] at (10.15, 3.06) {$K_{\mathrm{idemp}} = \text{HMAC}(k_{\mathrm{seed}}, \tau \parallel a_t)$};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=WarmOchre] at (10.15, 2.82) {Guarantees exactly-once effects};

  \draw[fill=BurgundyL0, draw=Burgundy!30, line width=0.3pt] (10.05, 1.90) rectangle (13.95, 2.50);
  \node[anchor=north west, font=\fontsize{3.2}{3.8}\selectfont\bfseries, text=Burgundy] at (10.15, 2.42) {Pre-Pivot Trap ($T_{\mathrm{pivot}}$):};
  \node[anchor=north west, font=\fontsize{2.9}{3.5}\selectfont, text=Burgundy] at (10.15, 2.14) {Irreversible action $\implies$ human sig!};

  % Signal 3 -> 4: Grant
  \draw[->, draw=WarmOchre, line width=1.1pt] (14.1, 3.75) -- (14.65, 3.75);
  \node[anchor=south, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=WarmOchre] at (14.37, 3.82) {Grant};

  % ---------------------------------------------------------------------------
  % STAGE 4: ISOLATED MICROVM SANDBOX (x: 14.65 to 18.85) - Width 4.2
  % ---------------------------------------------------------------------------
  \draw[fill=white, draw=Burgundy, line width=0.7pt, rounded corners=2pt] (14.65, 1.8) rectangle (18.85, 5.7);
  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{3.8}{4.5}\selectfont\bfseries, rounded corners=1pt, inner sep=1.5pt]
    at (14.75, 5.6) {STAGE 4: MICROVM SANDBOX};
  \node[anchor=north east, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=Burgundy] at (18.75, 5.6) {Part III};

  \draw[fill=BurgundyL0, draw=Burgundy!30, line width=0.3pt, rounded corners=1pt] (14.80, 4.40) rectangle (18.70, 5.08);
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (14.90, 5.00) {$W \oplus X$ Isolation Barrier};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (14.90, 4.68) {KVM hypervisor containment};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (14.80, 3.50) rectangle (18.70, 4.30);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (14.90, 4.22) {Kernel \& Egress Filters};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=SlateSteel] at (14.90, 3.96) {seccomp-bpf (syscall whitelist)};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=Burgundy] at (14.90, 3.72) {Netns default-deny exfiltration};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (14.80, 2.60) rectangle (18.70, 3.40);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (14.90, 3.32) {Ephemeral Copy-on-Write};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (14.90, 3.06) {OverlayFS upperdir on ramdisk};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=ForestSage] at (14.90, 2.82) {Fault $\implies$ millisecond rootfs discard};

  \draw[fill=BurgundyL0, draw=Burgundy!30, line width=0.3pt] (14.80, 1.90) rectangle (18.70, 2.50);
  \node[anchor=north west, font=\fontsize{3.2}{3.8}\selectfont\bfseries, text=Burgundy] at (14.90, 2.42) {Actuation Execution:};
  \node[anchor=north west, font=\fontsize{2.9}{3.5}\selectfont\ttfamily, text=DarkSlate] at (14.90, 2.14) {Run tool $\to$ capture (exit\_code, out)};

  % Signal 4 -> 5: Raw Output
  \draw[->, draw=Burgundy, line width=1.1pt] (18.85, 3.75) -- (19.40, 3.75);
  \node[anchor=south, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=Burgundy] at (19.12, 3.82) {$o_{\mathrm{raw}}$};

  % ---------------------------------------------------------------------------
  % STAGE 5: SANITIZATION & SAGAS (x: 19.4 to 23.6) - Width 4.2
  % ---------------------------------------------------------------------------
  \draw[fill=white, draw=Amethyst, line width=0.7pt, rounded corners=2pt] (19.4, 1.8) rectangle (23.6, 5.7);
  \node[anchor=north west, fill=Amethyst, text=white, font=\fontsize{3.8}{4.5}\selectfont\bfseries, rounded corners=1pt, inner sep=1.5pt]
    at (19.5, 5.6) {STAGE 5: SANITIZER \& SAGAS};
  \node[anchor=north east, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=Amethyst] at (23.5, 5.6) {Part III/IV};

  \draw[fill=AmethystL0, draw=Amethyst!30, line width=0.3pt, rounded corners=1pt] (19.55, 4.40) rectangle (23.45, 5.08);
  \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (19.65, 5.00) {Observation Sanitizer};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (19.65, 4.68) {Validate against $\Sigma_{\mathrm{obs}}$};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (19.55, 3.50) rectangle (23.45, 4.30);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (19.65, 4.22) {Compensating Saga Log};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=DarkSlate] at (19.65, 3.96) {$\mathcal{A}^{-1} \colon \text{compensate}(a_t)$};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=Amethyst] at (19.65, 3.72) {Backward recovery on abort};

  \draw[fill=CoolParchment, draw=SlateRim, line width=0.3pt] (19.55, 2.60) rectangle (23.45, 3.40);
  \node[anchor=north west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=DarkSlate] at (19.65, 3.32) {Indirect Injection Defense};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (19.65, 3.06) {Escape delimiters, mark tainted};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=Burgundy] at (19.65, 2.82) {Neutralizes payload attacks};

  \draw[fill=AmethystL0, draw=Amethyst!30, line width=0.3pt] (19.55, 1.90) rectangle (23.45, 2.50);
  \node[anchor=north west, font=\fontsize{3.2}{3.8}\selectfont\bfseries, text=Amethyst] at (19.65, 2.42) {State Delta Commit:};
  \node[anchor=north west, font=\fontsize{2.9}{3.5}\selectfont\ttfamily, text=DarkSlate] at (19.65, 2.14) {WAL sync $\implies o_{t+1}$ appends to $\tau$};

  % ---------------------------------------------------------------------------
  % BOTTOM SUB-TIER: OBSERVATION RETURN BUS & SUPERVISORY HARNESS (y: 0.20 to 1.55)
  % ---------------------------------------------------------------------------
  \draw[fill=white, draw=SlateRim, line width=0.5pt, rounded corners=2pt] (0.4, 0.20) rectangle (23.6, 1.55);

  % Observation Feedback Bus
  \draw[->, draw=Amethyst, line width=1.1pt, dashed] (21.5, 1.8) -- (21.5, 1.35) -- (2.5, 1.35) -- (2.5, 1.8);
  \node[anchor=south, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=Amethyst] at (12.0, 1.37)
    {$\longleftarrow$ PASSIVE OBSERVATION RETURN BUS: $o_{t+1}$ returns strictly as PASSIVE DATA into Context $c_{t+1}$ $\longleftarrow$};

  % Supervisory Control Plane Status Indicators
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=OxfordNavy] at (0.55, 0.65)
    {SUPERVISOR (ACB):};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (3.6, 0.65)
    {Monotonic Clock ($T \le T_{\max}$)};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateRim] at (7.6, 0.65) {$\mid$};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (8.0, 0.65)
    {Three-Tier Storage WAL (NVMe $\to$ Postgres $\to$ S3)};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateRim] at (14.5, 0.65) {$\mid$};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (14.9, 0.65)
    {Crash Recovery Vector ($\gamma$ Fencing Tokens)};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateRim] at (20.2, 0.65) {$\mid$};
  \node[anchor=west, font=\fontsize{3.1}{3.7}\selectfont\bfseries, text=ForestSage] at (20.6, 0.65)
    {Verified Invariant Closure};
\end{scope}

% Downward Arrow to Acceptance Gate
\draw[->, draw=OxfordNavy, line width=1.1pt] (12.0, 6.7) -- (12.0, 6.30);
\node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=OxfordNavy] at (12.2, 6.50)
  {Candidate Solution Emitted $\implies$ Route to Sealed Verification Oracle};

% =============================================================================
% TIER 3: ACCEPTANCE GATE & LIFECYCLE INFRASTRUCTURE (BOTTOM)
% (x: 0 to 24.0, y: 0.0 to 6.2) - Height 6.2 cm
% =============================================================================

% -----------------------------------------------------------------------------
% PANEL 3A: SEALED EXTERNAL VERIFICATION ORACLE (Bottom-Left)
% (x: 0 to 7.8, y: 0 to 6.2) - Width 7.8 cm
% -----------------------------------------------------------------------------
\begin{scope}[shift={(0, 0)}]
  \draw[fill=CoolParchment, draw=ForestSage, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (7.8, 6.2);

  \node[anchor=north west, fill=ForestSage, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 6.05) {3A. SEALED VERIFICATION ORACLE};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=ForestSage] at (7.6, 6.00) {Part IV/VII};

  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\itshape, text=Burgundy] at (0.3, 5.60)
    {The Fallacy of Green Dashboards: Never trust model self-reports!};

  % 4-Layer Verification Ladder (2 lines per layer, clean spacing)
  \draw[fill=white, draw=ForestSage!40, line width=0.4pt, rounded corners=1pt] (0.3, 4.35) rectangle (7.5, 5.20);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 5.07)
    {Layer 4: Performance \& SLA Invariant};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=ForestSage] at (0.45, 4.70)
    {Query latency $< 20\text{ ms}$ sustained for 5 min $\implies$ SLO intact};

  \draw[fill=white, draw=ForestSage!40, line width=0.4pt, rounded corners=1pt] (0.3, 3.40) rectangle (7.5, 4.25);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 4.12)
    {Layer 3: External State Delta Assertions ($\Delta\mathcal{S}_{\mathrm{ext}}$)};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=Burgundy] at (0.45, 3.75)
    {Direct DB inspection: 0 NULL rows, FK constraints validated};

  \draw[fill=white, draw=ForestSage!40, line width=0.4pt, rounded corners=1pt] (0.3, 2.45) rectangle (7.5, 3.30);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 3.17)
    {Layer 2: Deterministic Process Exit Code 0};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateSteel] at (0.45, 2.80)
    {MicroVM test runner exit status evaluated by host hypervisor};

  \draw[fill=white, draw=ForestSage!40, line width=0.4pt, rounded corners=1pt] (0.3, 1.50) rectangle (7.5, 2.35);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 2.22)
    {Layer 1: Structural Schema Conformance};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateSteel] at (0.45, 1.85)
    {Deliverable JSON payload validated against $\Sigma_{\mathrm{obs}}$ schema};

  % Verdict Strip
  \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=2pt] (0.3, 0.25) rectangle (7.5, 1.35);
  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=ForestSage] at (0.45, 1.20)
    {Mechanical Acceptance: Exit Code 0 across ALL 4 Layers};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (0.45, 0.88)
    {Failure $\implies$ Abort trajectory, trigger Saga compensate, quarantine};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\bfseries, text=ForestSage] at (0.45, 0.55)
    {Success $\implies$ Commit trajectory to Golden Dataset for post-training};
\end{scope}

% Arrow from Acceptance to Across-Task Learning
\draw[->, draw=ForestSage, line width=1.1pt] (7.5, 0.55) -- (8.1, 0.55);

% -----------------------------------------------------------------------------
% PANEL 3B: ACROSS-TASK LEARNING INFRASTRUCTURE (Bottom-Center)
% (x: 8.1 to 15.9, y: 0 to 6.2) - Width 7.8 cm
% -----------------------------------------------------------------------------
\begin{scope}[shift={(8.1, 0)}]
  \draw[fill=CoolParchment, draw=CobaltBlue, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (7.8, 6.2);

  \node[anchor=north west, fill=CobaltBlue, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 6.05) {3B. ACROSS-TASK LEARNING ENGINE};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=CobaltBlue] at (7.6, 6.00) {Part V: Adaptation};

  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\itshape, text=DarkSlate] at (0.3, 5.60)
    {Operational Horizon: Hours to Days $\cdot$ Distilling Deliberation into Weights};

  % Module 1: Trajectory Curation & Filtering
  \draw[fill=white, draw=CobaltBlue!40, line width=0.4pt, rounded corners=1pt] (0.3, 3.80) rectangle (7.5, 5.20);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 5.05)
    {1. Trajectory Curation \& Sieve Cascade};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateSteel] at (0.45, 4.70)
    {Prunes 74\% of candidate rollouts via syntax \& AST filters};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (0.45, 4.35)
    {Action-Targeted Loss: $\mathcal{L}_{\mathrm{SFT}} = -\sum_{t \in \mathcal{A}} \ln \pi_\theta(a_t \mid c_t)$};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=CobaltBlue] at (0.45, 4.02)
    {Backprop only through action tokens; environment masked};

  % Module 2: RLVR & Dual-Sandbox Enclave
  \draw[fill=white, draw=CobaltBlue!40, line width=0.4pt, rounded corners=1pt] (0.3, 2.30) rectangle (7.5, 3.65);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 3.50)
    {2. RLVR Policy Optimization (GRPO)};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateSteel] at (0.45, 3.15)
    {Evaluated against sealed holdout test chambers};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (0.45, 2.82)
    {Constant-Denominator Penalty: Fixes reasoning bloat};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=Burgundy] at (0.45, 2.50)
    {Scalar verdict $R_i \in \{0, 1\}$ updates surrogate policy};

  % Module 3: Wald SPRT Canary Certification
  \draw[fill=CobaltL0, draw=CobaltBlue!40, line width=0.4pt, rounded corners=1pt] (0.3, 0.25) rectangle (7.5, 2.15);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 2.00)
    {3. Wald SPRT Canary Certification};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=SlateSteel] at (0.45, 1.68)
    {Sequential probability ratio test on live traffic:};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\ttfamily, text=DarkSlate] at (0.45, 1.30)
    {$\Lambda_n = \sum_{i=1}^n \ln \frac{f(x_i \mid p_1)}{f(x_i \mid p_0)} \gtrless [\ln B, \ln A]$};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=ForestSage] at (0.45, 0.90)
    {Clamps traffic $\rho \to 0$ instantly upon defect detection};
  \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\bfseries, text=CobaltBlue] at (0.45, 0.50)
    {Certified Checkpoint Deploys to Core Serving Engine};
\end{scope}

% Feedback loop from Learning Engine back up to Core Inference (Dashed Cobalt Line)
\draw[->, draw=CobaltBlue, line width=1.0pt, dashed] (8.1, 0.4) -- (7.9, 0.4) -- (7.9, 6.7);
\node[anchor=east, font=\fontsize{3.2}{3.8}\selectfont\bfseries, text=CobaltBlue, rotate=90] at (7.82, 3.5)
  {Certified Weights $\theta^*$ Deployment Arc};

% -----------------------------------------------------------------------------
% PANEL 3C: FLEET SCALE & ARCHITECTURAL SYNTHESIS INVARIANTS (Bottom-Right)
% (x: 16.2 to 24.0, y: 0 to 6.2) - Width 7.8 cm
% -----------------------------------------------------------------------------
\begin{scope}[shift={(16.2, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (7.8, 6.2);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 6.05) {3C. FLEET SCALE \& THE H$\cdot$S$\cdot$A LAWS};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=DarkSlate] at (7.6, 6.00) {Part VI/VII};

  % Fleet Economics
  \draw[fill=white, draw=SlateRim, line width=0.4pt, rounded corners=1pt] (0.3, 3.75) rectangle (7.5, 5.40);
  \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 5.25)
    {Fleet Economics \& Effective Cost Frontier};
  \node[anchor=north west, font=\fontsize{3.2}{3.8}\selectfont\ttfamily, text=Burgundy] at (0.45, 4.80)
    {$C_{\mathrm{eff}} = C_{\mathrm{succ}} + \frac{1-\alpha}{\alpha} C_{\mathrm{fail}} \quad (\alpha = p^K)$};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=DarkSlate] at (0.45, 4.38)
    {The Token-Price Fallacy: 75x cheaper tokens cost 4.4x MORE!};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=ForestSage] at (0.45, 4.02)
    {Bayes Routing: $\tau^* = \Delta C / (V_{\mathrm{success}} \cdot \Delta P)$};

  % The Three H-S-A Conservation Laws
  \draw[fill=white, draw=ForestSage!40, line width=0.4pt, rounded corners=1pt] (0.3, 2.55) rectangle (7.5, 3.65);
  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=ForestSage] at (0.45, 3.52)
    {Law 1: Invariant Closure Principle (H$\cdot$S$\cdot$A)};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=DarkSlate] at (0.45, 3.18)
    {Stochastic core cannot close its own bounds. Host};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=ForestSage] at (0.45, 2.85)
    {mechanically bounds Horizon ($H$), State ($S$), Authority ($\mathcal{M}$).};

  \draw[fill=white, draw=Burgundy!40, line width=0.4pt, rounded corners=1pt] (0.3, 1.40) rectangle (7.5, 2.45);
  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=Burgundy] at (0.45, 2.32)
    {Law 2: The Human Specification Boundary};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=DarkSlate] at (0.45, 1.98)
    {Brooks' Essential Complexity cannot be automated away.};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=Burgundy] at (0.45, 1.65)
    {Engineer retains exclusive ownership of task contract $\mathcal{C}$.};

  \draw[fill=white, draw=Amethyst!40, line width=0.4pt, rounded corners=1pt] (0.3, 0.25) rectangle (7.5, 1.30);
  \node[anchor=north west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=Amethyst] at (0.45, 1.17)
    {Law 3: End-to-End Verification Argument};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont, text=DarkSlate] at (0.45, 0.83)
    {Verification must be external, deterministic, asymmetric.};
  \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=Amethyst] at (0.45, 0.50)
    {Task acceptance requires state delta ($\Delta\mathcal{S}_{\mathrm{ext}}$), not prose.};
\end{scope}

\end{tikzpicture}
\end{document}
"""

def main():
    print(f"Writing TikZ source to {TEX_PATH}...")
    with open(TEX_PATH, "w") as f:
        f.write(TIKZ_CONTENT.strip() + "\n")

    build_dir = "/tmp/build_unified_trajectory_spine"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    tex_build_file = os.path.join(build_dir, "unified_trajectory_spine.tex")
    with open(tex_build_file, "w") as f:
        f.write(TIKZ_CONTENT.strip() + "\n")

    print(f"Compiling LaTeX: pdflatex in {build_dir}...")
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory=" + build_dir, tex_build_file],
        capture_output=True,
        text=True
    )
    if res.returncode != 0:
        print("LaTeX compilation failed!")
        for line in res.stdout.splitlines()[-30:]:
            print(line)
        sys.exit(1)

    pdf_out = os.path.join(build_dir, "unified_trajectory_spine.pdf")
    png_prefix = os.path.join(build_dir, "unified_trajectory_spine_page")
    print(f"Rendering PDF to PNG at 250 DPI via pdftoppm...")
    subprocess.run(["pdftoppm", "-png", "-r", "250", pdf_out, png_prefix], check=True)

    png_out = png_prefix + "-1.png"
    if not os.path.exists(png_out):
        alt = png_prefix + ".png"
        if os.path.exists(alt):
            png_out = alt

    # Convert PDF to SVG via inkscape
    svg_out = os.path.join(build_dir, "unified_trajectory_spine.svg")
    print("Converting PDF to SVG via Inkscape...")
    subprocess.run(["inkscape", pdf_out, "--export-filename=" + svg_out, "--export-plain-svg"], check=True)

    # Sanitize SVG C0 control characters
    with open(svg_out, "rb") as f:
        svg_data = f.read()
    svg_sanitized = re.sub(b"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", svg_data)
    with open(svg_out, "wb") as f:
        f.write(svg_sanitized)

    # Copy artifacts to canonical repo locations
    targets = [
        # (src, dst)
        (png_out, os.path.join(PNG_DIR, "fig-vol3-trajectory-lifecycle.png")),
        (svg_out, os.path.join(SVG_DIR, "fig-vol3-trajectory-lifecycle.svg")),
        (pdf_out, os.path.join(PDF_DIR, "fig-vol3-trajectory-lifecycle.pdf")),
    ]

    for src, dst in targets:
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")

    print("Successfully built and deployed Chapter 19 Unified Trajectory Spine schematic!")

if __name__ == "__main__":
    main()
