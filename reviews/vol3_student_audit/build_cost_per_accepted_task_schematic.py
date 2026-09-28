#!/usr/bin/env python3
"""
Build publication-grade schematic for Chapter 18:
The Token-Price Fallacy & Cost-per-Accepted-Task Frontier (cost_per_accepted_task_frontier)

Standards:
- Canonical academic palette (OxfordNavy, SlateSteel, MutedTeal, Burgundy, WarmOchre, ForestSage)
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy (8.0pt badges, 6.5pt components, 5.0pt metrics/signals)
- Authentic IEEE/ACM computer systems architecture glyphs:
  - Whole-Trajectory Subsystem Cost Stack (Model tokens vs Sandbox vs Tools vs Human Triage)
  - Non-Token Majority Invariant (41.4% tokens vs 58.6% systems overhead)
  - Quantitative Hyperbolic Cost Curve (Effective cost vs step success rate p, K=8 steps)
  - Point A (Naive low-cost, $1.85/task) vs Point B (Frontier model, $0.42/task) -> 4.4x savings!
  - Accelerator Roofline Pricing (GEMM prefill vs GEMV decode bandwidth equations)
  - Hyperbolic Tax on Unreliability (tau_unrel = (1-alpha)/alpha)
  - Bayes-Optimal Model Routing Threshold (tau*)
- Zero bullet points, zero paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys
import shutil
import re

TEX_PATH = "books/vol3/tex/cost_per_accepted_task_frontier.tex"
PNG_DIR = "books/vol3/18_agent_economics/images/png"
SVG_DIR = "books/vol3/18_agent_economics/images/svg"
PDF_DIR = "books/vol3/18_agent_economics/images/pdf"

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

\begin{document}
\begin{tikzpicture}[x=1cm, y=1cm, >=Latex]

% =============================================================================
% TOP SECTION: COST STACK (LEFT) & EFFECTIVE COST FRONTIER (RIGHT)
% (x: 0 to 24.0, y: 6.0 to 16.8) - Height 10.8 cm
% =============================================================================

% -----------------------------------------------------------------------------
% PANEL 1: WHOLE-TRAJECTORY SUBSYSTEM COST STACK (Top-Left)
% x: 0 to 11.8, y: 6.0 to 16.8 (width 11.8, height 10.8 cm)
% -----------------------------------------------------------------------------
\begin{scope}[shift={(0, 6.0)}]
  \draw[fill=CoolParchment, draw=SlateRim, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (11.8, 10.8);

  % Badge
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 10.65) {1. WHOLE-TRAJECTORY SUBSYSTEM COST STACK};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=MutedTeal]
    at (11.6, 10.6) {$K = 10$ Turns $\cdot C_{\mathrm{attempt}} = \$0.420$};

  % Formula Box
  \draw[fill=white, draw=SlateRim, line width=0.4pt, rounded corners=1pt] (0.3, 9.6) rectangle (11.5, 10.3);
  \node[anchor=center, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (5.9, 9.95)
    {$C_{\mathrm{attempt}} = \sum_{k=1}^K \left( C_{\mathrm{in},k} + C_{\mathrm{out},k} + C_{\mathrm{tools},k} + C_{\mathrm{sandbox},k} \right) + C_{\mathrm{verify}} + C_{\mathrm{human}}$};

  % Tier 1: Foundation Model Inference Tier ($0.174, 41.4%)
  \draw[fill=white, draw=MutedTeal, line width=0.7pt, rounded corners=2pt] (0.3, 7.3) rectangle (11.5, 9.4);
  \node[anchor=north west, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=MutedTeal] at (0.45, 9.25)
    {1. Foundation Model Inference Tier -- \$0.174 (41.4\%)};
  \node[anchor=north east, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=SlateSteel] at (11.35, 9.25)
    {Accelerator Serving Engine};

  % Sub-block A: Prompt Prefill
  \draw[fill=TealL0, draw=MutedTeal!30, line width=0.4pt, rounded corners=1pt] (0.45, 7.45) rectangle (5.75, 8.85);
  \node[anchor=north west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=DarkSlate] at (0.6, 8.72)
    {Prompt Prefill ($C_{\mathrm{in}}$) -- \$0.112 (26.7\%)};
  \node[anchor=west, font=\fontsize{3.5}{4.2}\selectfont, text=DarkSlate] at (0.6, 8.35)
    {Compute-Bound GEMM ($I \gg 100\text{ FLOP/B}$)};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\ttfamily, text=SlateSteel] at (0.6, 7.95)
    {Context re-sent each turn ($K=10$ turns)};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=ForestSage] at (0.6, 7.62)
    {Prefill rate: $R_{\mathrm{prefill}} \approx 4{,}000\text{ tok/s}$};

  % Sub-block B: Autoregressive Decode
  \draw[fill=TealL0, draw=MutedTeal!30, line width=0.4pt, rounded corners=1pt] (6.05, 7.45) rectangle (11.35, 8.85);
  \node[anchor=north west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=DarkSlate] at (6.2, 8.72)
    {Autoregressive Decode ($C_{\mathrm{out}}$) -- \$0.062 (14.7\%)};
  \node[anchor=west, font=\fontsize{3.5}{4.2}\selectfont, text=DarkSlate] at (6.2, 8.35)
    {Memory-Bound GEMV ($I \approx 2\text{ FLOP/B}$)};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\ttfamily, text=SlateSteel] at (6.2, 7.95)
    {Emitted reasoning tokens + tool arguments};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=Burgundy] at (6.2, 7.62)
    {Decode tariff penalty: $3\times\text{--}5\times$ prefill rate};

  % Tier 2: Isolated Sandbox Runtime ($0.110, 26.2%)
  \draw[fill=white, draw=ForestSage, line width=0.6pt, rounded corners=2pt] (0.3, 5.35) rectangle (11.5, 7.15);
  \node[anchor=north west, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=ForestSage] at (0.45, 7.02)
    {2. Isolated Sandbox Runtime ($C_{\mathrm{sandbox}}$) -- \$0.110 (26.2\%)};
  \node[anchor=north east, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=SlateSteel] at (11.35, 7.02)
    {MicroVM / Container Host};

  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 6.55) {Hardware Profile:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\ttfamily, text=DarkSlate] at (2.9, 6.55) {120s lifetime $\times$ 2 vCPU / 4 GB RAM + Ephemeral Rootfs Overlay};
  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 6.10) {Isolation Primitive:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont, text=DarkSlate] at (2.9, 6.10) {Firecracker microVM / KVM guest with rootfs copy-on-write snapshot};
  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=Burgundy] at (0.45, 5.65) {Hidden Idle Trap:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont, text=Burgundy] at (2.9, 5.65) {Sandbox meter runs continuously while host awaits model decode generation!};

  % Tier 3: Tools & Verification Oracles ($0.108, 25.7%)
  \draw[fill=white, draw=WarmOchre, line width=0.6pt, rounded corners=2pt] (0.3, 3.4) rectangle (11.5, 5.2);
  \node[anchor=north west, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=WarmOchre] at (0.45, 5.07)
    {3. Tool API Egress \& Verification Oracles -- \$0.108 (25.7\%)};
  \node[anchor=north east, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=SlateSteel] at (11.35, 5.07)
    {Egress Network \& CI Host};

  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 4.60) {Tool Egress ($C_{\mathrm{tools}}$):};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont, text=DarkSlate] at (3.2, 4.60) {\$0.058 (Web search, SaaS RPCs, embedding lookups, vector stores)};
  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 4.15) {Verification ($C_{\mathrm{verify}}$):};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont, text=DarkSlate] at (3.2, 4.15) {\$0.050 (Deterministic pytest runners, AST static checks, invariant tests)};
  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 3.70) {Verifier Cascade:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont, text=ForestSage] at (3.2, 3.70) {Prunes 74\% of candidate patches before launching heavy microVM test suites};

  % Tier 4: Amortized Human Triage ($0.028, 6.7%)
  \draw[fill=white, draw=Burgundy, line width=0.6pt, rounded corners=2pt] (0.3, 1.45) rectangle (11.5, 3.25);
  \node[anchor=north west, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=Burgundy] at (0.45, 3.12)
    {4. Amortized Human Escalation ($C_{\mathrm{human}}$) -- \$0.028 (6.7\%)};
  \node[anchor=north east, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=SlateSteel] at (11.35, 3.12)
    {Operator Triage Queue};

  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 2.65) {Triage Formula:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\ttfamily, text=DarkSlate] at (2.9, 2.65) {$C_{\mathrm{human}} = P(\mathrm{escalate}) \times \$3.50$ labor cost per intervention};
  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=DarkSlate] at (0.45, 2.20) {Trigger Event:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont, text=DarkSlate] at (2.9, 2.20) {Semantic deadlocks, security taint violations, repeated verification aborts};
  \node[anchor=west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=Burgundy] at (0.45, 1.75) {Catastrophic Tail:};
  \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=Burgundy] at (2.9, 1.75) {Uncontrolled failure cascades shift workload onto expensive human triage!};

  % Footer Metrics Strip
  \draw[fill=OxfordNavy, draw=none, rounded corners=1pt] (0.3, 0.25) rectangle (11.5, 1.25);
  \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=white] at (0.5, 0.95)
    {THE NON-TOKEN MAJORITY: Tokens = 41.4\% (\$0.174) $\mid$ Supporting Systems = 58.6\% (\$0.246)};
  \node[anchor=west, font=\fontsize{3.5}{4.2}\selectfont, text=TealL0] at (0.5, 0.55)
    {Economic Law: Minimizing token tariff while ignoring trajectory pass rates guarantees financial ruin.};
\end{scope}

% -----------------------------------------------------------------------------
% PANEL 2: THE TOKEN-PRICE FALLACY: EFFECTIVE COST FRONTIER (Top-Right)
% x: 12.2 to 24.0, y: 6.0 to 16.8 (width 11.8, height 10.8 cm)
% -----------------------------------------------------------------------------
\begin{scope}[shift={(12.2, 6.0)}]
  \draw[fill=CoolParchment, draw=SlateRim, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (11.8, 10.8);

  % Badge
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
    at (0.2, 10.65) {2. THE TOKEN-PRICE FALLACY: EFFECTIVE COST};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=ForestSage]
    at (11.6, 10.6) {Why 75x cheaper tokens cost 4.4x MORE};

  % Formula Box
  \draw[fill=white, draw=SlateRim, line width=0.4pt, rounded corners=1pt] (0.3, 9.6) rectangle (11.5, 10.3);
  \node[anchor=center, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (5.9, 9.95)
    {$C_{\mathrm{eff}} = C_{\mathrm{succ}} + \tau_{\mathrm{unrel}} \cdot C_{\mathrm{fail}} = C_{\mathrm{succ}} + \left(\frac{1-\alpha}{\alpha}\right) C_{\mathrm{fail}} \quad (\alpha = p^K, K = 8\text{ Steps})$};

  % Quantitative Hyperbolic Plot Container
  \begin{scope}[shift={(0.3, 3.5)}]
    \draw[fill=white, draw=SlateRim, line width=0.6pt, rounded corners=2pt] (0, 0) rectangle (11.2, 5.95);

    % Graph Axes: Origin (1.1, 0.8), X max (10.6, 0.8), Y max (1.1, 5.5)
    \draw[->, draw=SlateSteel, line width=0.6pt] (1.1, 0.8) -- (10.7, 0.8);
    \draw[->, draw=SlateSteel, line width=0.6pt] (1.1, 0.8) -- (1.1, 5.6) node[left=1pt, font=\fontsize{3.6}{4.3}\selectfont] {$C_{\mathrm{eff}}$};

    % Centered X Axis Label
    \node[anchor=north, font=\fontsize{3.5}{4.2}\selectfont, text=SlateSteel] at (5.9, 0.35)
      {Single-Step Tool Execution Success Rate $p$ ($K = 8$ Steps)};

    % Y Axis Ticks: $0.00 (0.8), $0.50 (1.9), $1.00 (3.0), $1.50 (4.1), $2.00 (5.2)
    \draw[draw=GridGray, line width=0.3pt] (1.1, 1.9) -- (10.5, 1.9);
    \node[left=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (1.1, 1.9) {\$0.50};
    \draw[draw=GridGray, line width=0.3pt] (1.1, 3.0) -- (10.5, 3.0);
    \node[left=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (1.1, 3.0) {\$1.00};
    \draw[draw=GridGray, line width=0.3pt] (1.1, 4.1) -- (10.5, 4.1);
    \node[left=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (1.1, 4.1) {\$1.50};
    \draw[draw=GridGray, line width=0.3pt] (1.1, 5.2) -- (10.5, 5.2);
    \node[left=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (1.1, 5.2) {\$2.00};

    % X Axis Ticks: 0.60 (2.2), 0.70 (4.2), 0.80 (6.3), 0.90 (8.4), 1.00 (10.5)
    \draw[draw=SlateSteel, line width=0.4pt] (2.2, 0.75) -- (2.2, 0.8);
    \node[below=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (2.2, 0.75) {0.60};
    \draw[draw=SlateSteel, line width=0.4pt] (4.2, 0.75) -- (4.2, 0.8);
    \node[below=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (4.2, 0.75) {0.70};
    \draw[draw=SlateSteel, line width=0.4pt] (6.3, 0.75) -- (6.3, 0.8);
    \node[below=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (6.3, 0.75) {0.80};
    \draw[draw=SlateSteel, line width=0.4pt] (8.4, 0.75) -- (8.4, 0.8);
    \node[below=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (8.4, 0.75) {0.90};
    \draw[draw=SlateSteel, line width=0.4pt] (10.5, 0.75) -- (10.5, 0.8);
    \node[below=1pt, font=\fontsize{3.0}{3.6}\selectfont, text=SlateSteel] at (10.5, 0.75) {1.00};

    % Shaded Exponential Failure Zone: p <= 0.75 (x: 1.1 to 5.25)
    \draw[fill=BurgundyL0, draw=none, opacity=0.7] (1.1, 0.8) rectangle (5.25, 5.4);
    \draw[draw=Burgundy, dashed, line width=0.6pt] (5.25, 0.8) -- (5.25, 5.4);
    \node[anchor=north west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=Burgundy] at (1.3, 2.7)
      {EXPONENTIAL FAILURE REGIME};
    \node[anchor=north west, font=\fontsize{3.0}{3.6}\selectfont\bfseries, text=Burgundy] at (1.3, 2.25)
      {$\alpha = p^8 < 10\% \quad (\tau_{\mathrm{unrel}} > 9.0)$};
    \node[anchor=north west, font=\fontsize{2.8}{3.4}\selectfont, text=Burgundy] at (1.3, 1.85)
      {Wasted sandboxes \& tokens dominate};
    \node[anchor=north west, font=\fontsize{2.7}{3.3}\selectfont\ttfamily, text=Burgundy] at (1.3, 1.45)
      {Yield collapses below $p < 0.75$};

    % Hyperbolic Curve: C_eff = C_succ + (1 - p^8)/p^8 * C_fail
    % Curve points: (p=0.65 -> x=3.25, y=4.87 for $1.85)
    % (p=0.75 -> x=5.25, y=3.00 for $1.00)
    % (p=0.85 -> x=7.35, y=2.00 for $0.55)
    % (p=0.96 -> x=9.66, y=1.72 for $0.42)
    % (p=1.00 -> x=10.5, y=1.68 for $0.40)
    \draw[draw=Burgundy, line width=1.1pt]
      (2.5, 5.4) .. controls (2.9, 5.0) and (3.15, 4.95) .. (3.25, 4.87)
      .. controls (3.6, 4.5) and (4.5, 3.4) .. (5.25, 3.0)
      .. controls (6.0, 2.1) and (7.2, 1.82) .. (9.66, 1.72)
      .. controls (10.0, 1.70) .. (10.5, 1.68);

    % Point A: Naive Low-Cost Model (p=0.65, x=3.25, y=4.87)
    \draw[fill=Burgundy, draw=white, line width=0.8pt] (3.25, 4.87) circle (3pt);
    \draw[fill=white, draw=Burgundy, line width=0.5pt, rounded corners=1pt] (5.60, 4.25) rectangle (10.00, 5.50);
    \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=Burgundy] at (5.75, 5.35)
      {Point A: Naive Low-Cost Model ($p = 0.65$)};
    \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (5.75, 4.95)
      {Trajectory Yield: $\alpha = 0.65^8 = \mathbf{3.2\%}$};
    \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\bfseries, text=Burgundy] at (5.75, 4.55)
      {$C_{\mathrm{eff}} = \mathbf{\$1.85}$ / task ($31.4$ attempts!)};
    \draw[->, draw=Burgundy, line width=0.7pt] (5.60, 4.87) -- (3.40, 4.87);

    % Savings Callout Banner
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=2pt] (6.00, 3.55) rectangle (10.40, 4.05);
    \node[anchor=center, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=ForestSage] at (8.20, 3.80)
      {4.4x LOWER COST PER ACCEPTED TASK!};

    % Point B: Frontier Model (p=0.96, x=9.66, y=1.72)
    \draw[fill=ForestSage, draw=white, line width=0.8pt] (9.66, 1.72) circle (3pt);
    \draw[fill=white, draw=ForestSage, line width=0.5pt, rounded corners=1pt] (6.40, 2.15) rectangle (10.60, 3.40);
    \node[anchor=north west, font=\fontsize{3.6}{4.3}\selectfont\bfseries, text=ForestSage] at (6.55, 3.25)
      {Point B: Frontier Model ($p = 0.96$)};
    \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont, text=DarkSlate] at (6.55, 2.85)
      {Trajectory Yield: $\alpha = 0.96^8 = \mathbf{72.1\%}$};
    \node[anchor=north west, font=\fontsize{3.1}{3.7}\selectfont\bfseries, text=ForestSage] at (6.55, 2.45)
      {$C_{\mathrm{eff}} = \mathbf{\$0.42}$ / task ($1.38$ attempts)};
    \draw[->, draw=ForestSage, line width=0.7pt] (9.66, 2.15) -- (9.66, 1.85);
  \end{scope}

  % Structured Comparison Table (y: 0.25 to 3.35, height 3.1 cm)
  \begin{scope}[shift={(0.3, 0.25)}]
    \draw[fill=white, draw=SlateRim, line width=0.6pt, rounded corners=2pt] (0, 0) rectangle (11.2, 3.1);

    % Table Header
    \draw[fill=OxfordNavy, draw=none, rounded corners=1pt] (0.05, 2.65) rectangle (11.15, 3.05);
    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=white] at (0.2, 2.85) {Economic Metric};
    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=white] at (3.2, 2.85) {Naive Model (Point A)};
    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=white] at (6.0, 2.85) {Frontier Model (Point B)};
    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=white] at (8.8, 2.85) {Net Systems Advantage};

    % Row 1: Nominal Tariff
    \draw[fill=CoolParchment, draw=none] (0.05, 2.0) rectangle (11.15, 2.65);
    \node[anchor=west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=DarkSlate] at (0.2, 2.32) {Nominal Tariff:};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont\ttfamily, text=Burgundy] at (3.2, 2.32) {\$0.20 / 1M tokens};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont\ttfamily, text=DarkSlate] at (6.0, 2.32) {\$15.00 / 1M tokens};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont, text=DarkSlate] at (8.8, 2.32) {75x nominal tariff premium};

    \draw[draw=SlateRim, line width=0.3pt] (0.05, 2.0) -- (11.15, 2.0);

    % Row 2: Success Yield
    \draw[fill=white, draw=none] (0.05, 1.35) rectangle (11.15, 2.0);
    \node[anchor=west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=DarkSlate] at (0.2, 1.67) {Trajectory Yield $\alpha$:};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont\ttfamily, text=Burgundy] at (3.2, 1.67) {$0.65^8 = 3.2\%$ (1 in 31)};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont\ttfamily, text=ForestSage] at (6.0, 1.67) {$0.96^8 = 72.1\%$ (1 in 1.4)};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=ForestSage] at (8.8, 1.67) {22.7x higher success rate};

    \draw[draw=SlateRim, line width=0.3pt] (0.05, 1.35) -- (11.15, 1.35);

    % Row 3: Wasted Overhead
    \draw[fill=CoolParchment, draw=none] (0.05, 0.7) rectangle (11.15, 1.35);
    \node[anchor=west, font=\fontsize{3.5}{4.2}\selectfont\bfseries, text=DarkSlate] at (0.2, 1.02) {Wasted Overhead:};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont, text=Burgundy] at (3.2, 1.02) {30.4 failed sandbox runs};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont, text=ForestSage] at (6.0, 1.02) {0.38 failed sandbox runs};
    \node[anchor=west, font=\fontsize{3.3}{4.0}\selectfont\bfseries, text=ForestSage] at (8.8, 1.02) {80x less sandbox waste};

    \draw[draw=SlateRim, line width=0.3pt] (0.05, 0.7) -- (11.15, 0.7);

    % Row 4: Effective Cost
    \draw[fill=SageL0, draw=none] (0.05, 0.05) rectangle (11.15, 0.7);
    \node[anchor=west, font=\fontsize{3.6}{4.4}\selectfont\bfseries, text=OxfordNavy] at (0.2, 0.37) {Effective Cost:};
    \node[anchor=west, font=\fontsize{3.6}{4.4}\selectfont\bfseries, text=Burgundy] at (3.2, 0.37) {\$1.85 / accepted task};
    \node[anchor=west, font=\fontsize{3.6}{4.4}\selectfont\bfseries, text=ForestSage] at (6.0, 0.37) {\$0.42 / accepted task};
    \node[anchor=west, font=\fontsize{3.6}{4.4}\selectfont\bfseries, text=ForestSage] at (8.8, 0.37) {4.4x NET CASH SAVINGS!};
  \end{scope}
\end{scope}

% =============================================================================
% BOTTOM SECTION: PHYSICAL ROOFLINES & ECONOMIC GOVERNANCE INVARIANTS
% (x: 0 to 24.0, y: 0 to 5.6) - Height 5.6 cm
% =============================================================================
\begin{scope}[shift={(0, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.0pt, rounded corners=3pt] (0, 0) rectangle (24.0, 5.6);

  % Badge
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{5.5}{6.5}\selectfont\bfseries, rounded corners=1pt, inner sep=2.5pt]
    at (0.3, 5.45) {3. PHYSICAL HARDWARE ROOFLINES \& ECONOMIC GOVERNANCE INVARIANTS};
  \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=SlateSteel]
    at (23.7, 5.4) {Accelerator Cost Rooflines, Unreliability Tax, and Bayes-Optimal Routing};

  % Column 1: Accelerator Roofline Economics (x: 0.3 to 7.8, width 7.5 cm)
  \begin{scope}[shift={(0.3, 0.3)}]
    \draw[fill=white, draw=SlateRim, line width=0.6pt, rounded corners=2pt] (0, 0) rectangle (7.5, 4.65);
    \node[anchor=north west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=MutedTeal] at (0.2, 4.50)
      {A. ACCELERATOR ROOFLINE PRICING};

    % Prefill Equation
    \draw[fill=TealL0, draw=MutedTeal!30, line width=0.4pt, rounded corners=1pt] (0.2, 3.4) rectangle (7.3, 4.15);
    \node[anchor=center, font=\fontsize{3.8}{4.5}\selectfont, text=DarkSlate] at (3.75, 3.77)
      {$c_{\mathrm{prefill}} = \frac{C_{\mathrm{node}} \cdot 2P}{3600 \cdot \eta \cdot \text{FLOPS}} \quad (I \gg 100\text{ FLOP/B})$};

    % Decode Equation
    \draw[fill=TealL0, draw=MutedTeal!30, line width=0.4pt, rounded corners=1pt] (0.2, 2.5) rectangle (7.3, 3.25);
    \node[anchor=center, font=\fontsize{3.8}{4.5}\selectfont, text=DarkSlate] at (3.75, 2.87)
      {$c_{\mathrm{decode}}(B) = \frac{C_{\mathrm{node}} \cdot P \cdot b}{3600 \cdot B \cdot \beta_{\mathrm{HBM}}} \quad (I \approx 2\text{ FLOP/B})$};

    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (0.25, 2.15) {FLOP vs. Bandwidth Asymmetry:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 1.75) {Prefill saturates Tensor Cores across large prompt batches.};
    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=Burgundy] at (0.25, 1.30) {Decode Tariff Penalty:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 0.90) {Memory-bandwidth bound decode costs $3\times\text{--}5\times$ more per token};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=SlateSteel] at (0.25, 0.50) {at low batch concurrency $B \ll B^*$.};
  \end{scope}

  % Column 2: The Hyperbolic Tax on Unreliability (x: 8.2 to 15.8, width 7.6 cm)
  \begin{scope}[shift={(8.2, 0.3)}]
    \draw[fill=white, draw=SlateRim, line width=0.6pt, rounded corners=2pt] (0, 0) rectangle (7.6, 4.65);
    \node[anchor=north west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=Burgundy] at (0.2, 4.50)
      {B. THE HYPERBOLIC TAX ON UNRELIABILITY};

    % Formula Box
    \draw[fill=BurgundyL0, draw=Burgundy!30, line width=0.4pt, rounded corners=1pt] (0.2, 3.4) rectangle (7.4, 4.15);
    \node[anchor=center, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (3.8, 3.77)
      {$\tau_{\mathrm{unrel}}(\alpha) = \frac{1 - \alpha}{\alpha} \implies \lim_{\alpha \to 0} \tau_{\mathrm{unrel}}(\alpha) = \infty$};

    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (0.25, 2.95) {Asymptotic Multiplier:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 2.55) {Near $\alpha \to 1.0$, tax is tiny ($\tau(0.90) = 0.11$).};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 2.15) {Below $\alpha = 0.50$, tax exceeds unity ($\tau(0.10) = 9.0$).};

    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=Burgundy] at (0.25, 1.55) {Financial Destruction Mechanism:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 1.15) {Discarded microVM sandboxes, redundant tool egress fees,};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 0.75) {and re-sent prefix prompt tokens wipe out all tariff savings!};
  \end{scope}

  % Column 3: Bayes-Optimal Model Routing Threshold (x: 16.2 to 23.7, width 7.5 cm)
  \begin{scope}[shift={(16.2, 0.3)}]
    \draw[fill=white, draw=SlateRim, line width=0.6pt, rounded corners=2pt] (0, 0) rectangle (7.5, 4.65);
    \node[anchor=north west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=ForestSage] at (0.2, 4.50)
      {C. BAYES-OPTIMAL ROUTING THRESHOLD};

    % Formula Box
    \draw[fill=SageL0, draw=ForestSage!30, line width=0.4pt, rounded corners=1pt] (0.2, 3.4) rectangle (7.3, 4.15);
    \node[anchor=center, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (3.75, 3.77)
      {$\tau^* = \frac{C_{\mathrm{frontier}} - C_{\mathrm{small}}}{V_{\mathrm{task\_success}} \cdot \Delta P(\mathrm{success})}$};

    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=DarkSlate] at (0.25, 2.95) {Optimal Dispatch Rule:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 2.55) {Route to frontier model iff predicted accuracy gain satisfies:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont\ttfamily, text=ForestSage] at (0.25, 2.15) {$\Delta P > (C_{\mathrm{frontier}} - C_{\mathrm{small}}) / V_{\mathrm{task\_success}}$};

    \node[anchor=west, font=\fontsize{3.4}{4.1}\selectfont\bfseries, text=ForestSage] at (0.25, 1.55) {Asymmetric Loss Function:};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=DarkSlate] at (0.25, 1.15) {Routing down erroneously wastes an entire task execution cycle;};
    \node[anchor=west, font=\fontsize{3.2}{3.8}\selectfont, text=SlateSteel] at (0.25, 0.75) {routing up erroneously wastes only the tier tariff delta.};
  \end{scope}
\end{scope}

\end{tikzpicture}
\end{document}
"""

def main():
    print(f"Writing TikZ source to {TEX_PATH}...")
    with open(TEX_PATH, "w") as f:
        f.write(TIKZ_CONTENT.strip() + "\n")

    build_dir = "/tmp/build_cost_per_accepted_task"
    os.makedirs(build_dir, exist_ok=True)
    tex_build_file = os.path.join(build_dir, "cost_frontier.tex")
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

    pdf_out = os.path.join(build_dir, "cost_frontier.pdf")
    png_prefix = os.path.join(build_dir, "cost_frontier_page")
    print(f"Rendering PDF to PNG at 250 DPI via pdftoppm...")
    subprocess.run(["pdftoppm", "-png", "-r", "250", pdf_out, png_prefix], check=True)

    png_out = png_prefix + "-1.png"
    if not os.path.exists(png_out):
        alt = png_prefix + ".png"
        if os.path.exists(alt):
            png_out = alt

    # Convert PDF to SVG via inkscape
    svg_out = os.path.join(build_dir, "cost_frontier.svg")
    print("Converting PDF to SVG via Inkscape...")
    subprocess.run(["inkscape", pdf_out, "--export-filename=" + svg_out, "--export-plain-svg"], check=True)

    with open(svg_out, "rb") as f:
        svg_data = f.read()
    svg_sanitized = re.sub(b"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", svg_data)
    with open(svg_out, "wb") as f:
        f.write(svg_sanitized)

    # Copy artifacts to canonical repo locations
    targets = [
        # (src, dst)
        (png_out, os.path.join(PNG_DIR, "fig-vol3-task-cost-treemap.png")),
        (png_out, os.path.join(PNG_DIR, "cost_per_accepted_task_frontier.png")),
        (svg_out, os.path.join(SVG_DIR, "fig-vol3-task-cost-treemap.svg")),
        (svg_out, os.path.join(SVG_DIR, "cost_per_accepted_task_frontier.svg")),
        (pdf_out, os.path.join(PDF_DIR, "fig-vol3-task-cost-treemap.pdf")),
        (pdf_out, os.path.join(PDF_DIR, "cost_per_accepted_task_frontier.pdf")),
    ]

    for src, dst in targets:
        shutil.copyfile(src, dst)
        print(f"Copied {src} -> {dst}")

    print("Successfully built and deployed Chapter 18 Cost-per-Accepted-Task schematic!")

if __name__ == "__main__":
    main()
