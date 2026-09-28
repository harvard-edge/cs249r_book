#!/usr/bin/env python3
"""
Build publication-grade schematic for Chapter 15:
The Three Homes of Agent Knowledge & The HSA Boundary (three_homes_of_agent_knowledge)

Standards:
- Canonical academic palette (OxfordNavy, SlateSteel, MutedTeal, Burgundy, WarmOchre, ForestSage)
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy (8.0pt badges, 6.5pt components, 5.0pt metrics/signals)
- Authentic IEEE/ACM computer systems architecture glyphs:
  - 3 Homes Substrates: Weights (Procedure), Context (Contracts), Runtime (Enforcement)
  - Hardware-Software-Agent (HSA) physical isolation boundary
  - Inter-home runtime control loop (Proposal -> Invariant Interception -> Isolated Actuation -> Context Feed)
  - Fine-Tuning Jurisdiction Matrix & Non-Fungibility Laws
- Zero bullet points, zero paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys
import re

TEX_PATH = "books/vol3/tex/three_homes_of_agent_knowledge.tex"
PNG_DIR = "books/vol3/15_fine_tuning/images/png"
SVG_DIR = "books/vol3/15_fine_tuning/images/svg"
PDF_DIR = "books/vol3/15_fine_tuning/images/pdf"

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
% PANEL 1: THE THREE HOMES OF AGENT KNOWLEDGE (TOP)
% (x: 0 to 24.0, y: 7.6 to 16.8) - Height 9.2 cm
% =============================================================================
\begin{scope}[shift={(0, 7.6)}]
  % Base Canvas Container
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (24.0, 9.2);

  % Sub-realm indicator banner (y: 8.05 to 8.45)
  \draw[fill=AmethystL0, draw=Amethyst!40, line width=0.5pt, rounded corners=1.5pt] (0.4, 8.05) rectangle (15.1, 8.45);
  \node[anchor=center, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=Amethyst] at (7.75, 8.25)
    {STOCHASTIC PROPOSAL REALM (Uncertainty $\cdot$ Prediction $\cdot$ Probability Distribution $P_\theta(a_t \mid c_t) \in [0, 1]$)};

  \draw[fill=SageL0, draw=ForestSage!40, line width=0.5pt, rounded corners=1.5pt] (15.7, 8.05) rectangle (23.6, 8.45);
  \node[anchor=center, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=ForestSage] at (19.65, 8.25)
    {DETERMINISTIC INVARIANT REALM (Binary Invariants $\{0, 1\}$)};

  % --- HOME 1: NEURAL WEIGHTS (x: 0.4 to 7.5, width 7.1 cm, y: 0.4 to 7.9) ---
  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=Amethyst, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (7.1, 7.5);

    \node[anchor=north west, fill=Amethyst, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
      at (0.2, 7.38) {HOME 1: NEURAL WEIGHTS [$W_\theta$]};
    \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=Amethyst]
      at (6.9, 7.35) {Procedure (Frozen)};

    % Block 1.1: Physical Substrate & Scope
    \draw[fill=AmethystL0, draw=Amethyst!30, line width=0.4pt, rounded corners=1pt] (0.2, 5.0) rectangle (6.9, 6.75);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=Amethyst] at (0.3, 6.62) {Hardware Substrate \& Scope:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 6.20) {Storage:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (1.8, 6.20) {HBM parameter tensors (FP8 / BF16), LoRA $\Delta W$};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.75) {Content:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (1.8, 5.75) {Tool selection priors, stack trace parsing, error recovery};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.30) {Latency:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=SlateSteel] at (1.8, 5.30) {$O(1)$ memory bus bandwidth (GEMV decode $40\text{--}80\,\text{ms}$)};

    % Block 1.2: Training & Update Horizon
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.4pt, rounded corners=1pt] (0.2, 3.1) rectangle (6.9, 4.85);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=OxfordNavy] at (0.3, 4.72) {Lifecycle \& Update Horizon:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 4.30) {Update Cadence:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (2.4, 4.30) {Offline batch training ($\sim 10^3\text{--}10^5$ steps)};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 3.85) {Loss Objective:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (2.4, 3.85) {Masked trajectory cross-entropy $\mathcal{L}_{\mathrm{SFT}}(\theta)$};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 3.40) {Serving Cost:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=ForestSage] at (2.4, 3.40) {Zero per-turn compilation or prompt overhead};

    % Block 1.3: Failure Mode
    \draw[fill=BurgundyL0, draw=Burgundy!40, line width=0.5pt, rounded corners=1pt] (0.2, 1.2) rectangle (6.9, 2.95);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=Burgundy] at (0.3, 2.82) {Pathological Failure: Schema Memorization};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 2.40) {Stale Parameters:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (2.4, 2.40) {Model memorizes API arguments; ignores live schema in $c_t$};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 1.95) {Secret Leakage:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (2.4, 1.95) {Hardcoding credentials or API keys directly into weights};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=SlateSteel] at (0.3, 1.50) {Breakage on next API version bump is guaranteed};

    % Footer
    \draw[fill=Amethyst, draw=none, rounded corners=1pt] (0.2, 0.15) rectangle (6.9, 0.95);
    \node[anchor=center, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=white] at (3.55, 0.68)
      {SFT Effect: Reshapes proposal logits $P_\theta(a_t \mid c_t)$};
    \node[anchor=center, font=\fontsize{3.8}{4.5}\selectfont, text=AmethystL0] at (3.55, 0.38)
      {Increases likelihood of canonical habits; leaves runtime untouched};
  \end{scope}

  % --- HOME 2: CONTEXT REGISTER (x: 7.9 to 15.0, width 7.1 cm, y: 0.4 to 7.9) ---
  \begin{scope}[shift={(7.9, 0.4)}]
    \draw[fill=white, draw=MutedTeal, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (7.1, 7.5);

    \node[anchor=north west, fill=MutedTeal, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
      at (0.2, 7.38) {HOME 2: CONTEXT REGISTER [$c_t$]};
    \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=MutedTeal]
      at (6.9, 7.35) {Contracts (Dynamic)};

    % Block 2.1: Physical Substrate & Scope
    \draw[fill=TealL0, draw=MutedTeal!30, line width=0.4pt, rounded corners=1pt] (0.2, 5.0) rectangle (6.9, 6.75);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=MutedTeal] at (0.3, 6.62) {Hardware Substrate \& Scope:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 6.20) {Storage:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (1.8, 6.20) {Radix prefix tree, PagedAttention KV cache, host RAM};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.75) {Content:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (1.8, 5.75) {Live OpenAPI schemas, user goal, granted scope, past turns};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.30) {Latency:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=SlateSteel] at (1.8, 5.30) {$O(\text{seqlen})$ attention prefill / sub-millisecond Radix hit};

    % Block 2.2: Training & Update Horizon
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.4pt, rounded corners=1pt] (0.2, 3.1) rectangle (6.9, 4.85);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=OxfordNavy] at (0.3, 4.72) {Lifecycle \& Update Horizon:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 4.30) {Update Cadence:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (2.4, 4.30) {Immediate per-turn dynamic assembly ($< 1\,\text{ms}$)};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 3.85) {Retention Limit:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (2.4, 3.85) {Bounded by hardware context ceiling $C_{\mathrm{ctx}}$};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 3.40) {Serving Cost:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=WarmOchre] at (2.4, 3.40) {KV cache footprint grows quadratically without compaction};

    % Block 2.3: Failure Mode
    \draw[fill=BurgundyL0, draw=Burgundy!40, line width=0.5pt, rounded corners=1pt] (0.2, 1.2) rectangle (6.9, 2.95);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=Burgundy] at (0.3, 2.82) {Pathological Failure: Context Flooding};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 2.40) {Bloated Prompts:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (2.4, 2.40) {Stuffing static examples into prompt crowds out tool observations};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 1.95) {Raw Dumps:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (2.4, 1.95) {Inserting 500-line compiler dumps induces attention dispersion};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=SlateSteel] at (0.3, 1.50) {Context rot corrupts reasoning within 10 turns};

    % Footer
    \draw[fill=MutedTeal, draw=none, rounded corners=1pt] (0.2, 0.15) rectangle (6.9, 0.95);
    \node[anchor=center, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=white] at (3.55, 0.68)
      {SFT Effect: Teaches model to CONDITION ON contracts};
    \node[anchor=center, font=\fontsize{3.8}{4.5}\selectfont, text=TealL0] at (3.55, 0.38)
      {Compresses instructions out of prompt into weights, freeing context};
  \end{scope}

  % --- THE HSA BOUNDARY (Demarcation Line at x: 15.35) ---
  \draw[draw=Burgundy, line width=1.5pt, dashed] (15.35, 0.4) -- (15.35, 8.5);
  \node[draw=Burgundy, fill=white, line width=0.8pt, rounded corners=2pt, inner sep=2pt, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=Burgundy, rotate=90]
    at (15.35, 4.45) {THE HSA ARCHITECTURAL BOUNDARY};

  % --- HOME 3: DETERMINISTIC RUNTIME (x: 16.5 to 23.6, width 7.1 cm, y: 0.4 to 7.9) ---
  \begin{scope}[shift={(16.5, 0.4)}]
    \draw[fill=white, draw=ForestSage, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (7.1, 7.5);

    \node[anchor=north west, fill=ForestSage, text=white, font=\fontsize{5.2}{6.2}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
      at (0.2, 7.38) {HOME 3: HOST RUNTIME [$E_{\mathrm{host}}$]};
    \node[anchor=north east, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=ForestSage]
      at (6.9, 7.35) {Enforcement (Fixed)};

    % Block 3.1: Physical Substrate & Scope
    \draw[fill=SageL0, draw=ForestSage!30, line width=0.4pt, rounded corners=1pt] (0.2, 5.0) rectangle (6.9, 6.75);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=ForestSage] at (0.3, 6.62) {Hardware Substrate \& Scope:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 6.20) {Substrate:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (1.8, 6.20) {Host Linux kernel, seccomp-BPF, KVM / Firecracker microVM};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.75) {Content:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (1.8, 5.75) {Grammar masks, capability airlocks, sealed unit tests, diff audits};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.30) {Latency:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=ForestSage] at (1.8, 5.30) {Sub-millisecond host syscall interception ($< 10\,\mu\text{s}$)};

    % Block 3.2: Training & Update Horizon
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.4pt, rounded corners=1pt] (0.2, 3.1) rectangle (6.9, 4.85);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=OxfordNavy] at (0.3, 4.72) {Lifecycle \& Update Horizon:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 4.30) {Guarantee Model:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=ForestSage] at (2.4, 4.30) {Mathematically closed binary invariants ($\{0, 1\}$)};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 3.85) {Authority Model:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (2.4, 3.85) {Zero ambient authority (\texttt{CLONE\_NEWUSER})};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 3.40) {Sovereignty:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=ForestSage] at (2.4, 3.40) {Survives arbitrary adversarial model hallucination};

    % Block 3.3: Failure Mode
    \draw[fill=BurgundyL0, draw=Burgundy!40, line width=0.5pt, rounded corners=1pt] (0.2, 1.2) rectangle (6.9, 2.95);
    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=Burgundy] at (0.3, 2.82) {Pathological Failure: Prompt-Injection Fallacy};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 2.40) {The Fatal Myth:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (2.4, 2.40) {Believing SFT or prompt guardrails can replace sandboxes};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=DarkSlate] at (0.3, 1.95) {Vulnerability:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (2.4, 1.95) {Adversarial jailbreaks easily bypass weights-only alignment};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=SlateSteel] at (0.3, 1.50) {Only hardware/kernel boundaries guarantee invariant closure};

    % Footer
    \draw[fill=ForestSage, draw=none, rounded corners=1pt] (0.2, 0.15) rectangle (6.9, 0.95);
    \node[anchor=center, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=white] at (3.55, 0.68)
      {SFT Effect: LEAVES RUNTIME COMPLETELY UNTOUCHED};
    \node[anchor=center, font=\fontsize{3.8}{4.5}\selectfont, text=SageL0] at (3.55, 0.38)
      {Runtime intercepts fine-tuned proposals unconditionally};
  \end{scope}

  % Header Badges Drawn on Top of Base
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{6.8}{8.0}\selectfont\bfseries, rounded corners=2pt, inner xsep=4pt, inner ysep=2.5pt]
    at (0.4, 9.08) {1. THE THREE HOMES OF AGENT KNOWLEDGE \& THE HSA BOUNDARY};
  \node[anchor=north east, font=\fontsize{5.2}{6.2}\selectfont\bfseries, text=SlateSteel]
    at (23.6, 9.02) {Tripartite Law: Weights $\longleftrightarrow$ Context $\longleftrightarrow$ Runtime};
\end{scope}

% =============================================================================
% PANEL 2: INTER-HOME RUNTIME CONTROL LOOP (BOTTOM LEFT)
% (x: 0 to 14.6, y: 0 to 7.2) - Width 14.6 cm, Height 7.2 cm
% =============================================================================
\begin{scope}[shift={(0, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (14.6, 7.2);

  \begin{scope}[shift={(0.3, 0.35)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (14.0, 6.0);

    % Step 1: Model Proposal Generation (Home 1 + Home 2)
    \draw[fill=AmethystL0, draw=Amethyst!60, line width=0.5pt, rounded corners=1.5pt] (0.2, 3.25) rectangle (4.2, 5.80);
    \node[anchor=north west, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=Amethyst] at (0.3, 5.65) {1. PROPOSAL GENERATION};
    \node[anchor=north east, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=SlateSteel] at (4.1, 5.65) {Homes 1 \& 2};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.15) {Neural Tensor Core:};
    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont, text=DarkSlate] at (0.3, 4.75) {$a_{\mathrm{prop}} \sim P_\theta(\cdot \mid c_t)$};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=SlateSteel] at (0.3, 4.35) {Unverified action token candidate};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=SlateSteel] at (0.3, 3.95) {Conditioned on prompt schema $c_t$};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=Amethyst] at (0.3, 3.55) {Emits proposal packet $a_{\mathrm{prop}} \longrightarrow$};

    % Connector 1 -> 2
    \draw[->, line width=1.4pt, draw=OxfordNavy] (4.2, 4.5) -- node[above=1pt, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] {$a_{\mathrm{prop}}$} (4.9, 4.5);

    % Step 2: Invariant Interception (Home 3 Entry)
    \draw[fill=OchreL0, draw=WarmOchre!60, line width=0.5pt, rounded corners=1.5pt] (4.9, 3.25) rectangle (8.9, 5.80);
    \node[anchor=north west, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=WarmOchre] at (5.0, 5.65) {2. INVARIANT INTERCEPTION};
    \node[anchor=north east, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=ForestSage] at (8.8, 5.65) {Home 3 Gate};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (5.0, 5.15) {Deterministic Airlock:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (5.0, 4.75) {JSON Grammar Mask: $\mathcal{G}(a_{\mathrm{prop}}) = 1$};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (5.0, 4.35) {HMAC Capability: $\mathrm{Auth}(a_{\mathrm{prop}}) = \mathrm{OK}$};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=Burgundy] at (5.0, 3.95) {Rejects on syntax breach or scope violation};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=WarmOchre] at (5.0, 3.55) {Clears authorized action $a_{\mathrm{auth}} \longrightarrow$};

    % Connector 2 -> 3
    \draw[->, line width=1.4pt, draw=ForestSage] (8.9, 4.5) -- node[above=1pt, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=ForestSage] {$a_{\mathrm{auth}}$} (9.6, 4.5);

    % Step 3: Sandboxed Execution (Home 3 Actuation)
    \draw[fill=SageL0, draw=ForestSage!60, line width=0.5pt, rounded corners=1.5pt] (9.6, 3.25) rectangle (13.8, 5.80);
    \node[anchor=north west, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=ForestSage] at (9.7, 5.65) {3. ISOLATED ACTUATION};
    \node[anchor=north east, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=ForestSage] at (13.7, 5.65) {MicroVM};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (9.7, 5.15) {Physical Execution:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (9.7, 4.75) {Linux cgroup / Firecracker sandbox};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (9.7, 4.35) {Captures stdout, stderr, process return};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=ForestSage] at (9.7, 3.95) {Produces observation tuple $o_{t+1}$};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=MutedTeal] at (9.7, 3.55) {Feeds Observation Return $\hookleftarrow$};

    % Return Loop Bar (Bottom: 0.2 to 2.85, height 2.65 cm)
    \draw[fill=TealL0, draw=MutedTeal!40, line width=0.5pt, rounded corners=1.5pt] (0.2, 0.2) rectangle (13.8, 2.85);
    \node[anchor=north west, font=\fontsize{4.6}{5.4}\selectfont\bfseries, text=MutedTeal] at (0.3, 2.70)
      {FEEDBACK PIPELINE: OBSERVATION RETURN ($o_{t+1} \hookrightarrow c_{t+1}$)};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 2.25) {Dynamic KV Append:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (2.8, 2.25)
      {Observation $o_{t+1}$ serialized into token blocks; appended to Radix KV prefix tree};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 1.80) {Turn Assembly:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (2.8, 1.80)
      {Next-turn prompt $c_{t+1} = [c_t, a_t, o_{t+1}]$ assembled without modifying weights $W_\theta$};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=OxfordNavy] at (0.3, 1.35) {Memory Integrity:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont\bfseries, text=ForestSage] at (2.8, 1.35)
      {Zero parameter drift; prompt changes instantly reflect new environmental observations};

    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=SlateSteel] at (0.3, 0.85)
      {Control Loop Cycle: Generates proposal $a_{t+1}$ under refreshed context $c_{t+1}$ in turn $t+1$};
  \end{scope}

  % Header Badges Drawn on Top of Base
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{6.8}{8.0}\selectfont\bfseries, rounded corners=2pt, inner xsep=4pt, inner ysep=2.5pt]
    at (0.4, 7.08) {2. INTER-HOME RUNTIME CONTROL LOOP};
  \node[anchor=north east, font=\fontsize{5.2}{6.2}\selectfont\bfseries, text=MutedTeal]
    at (14.2, 7.02) {Invariant Closure: $\mathcal{G}(a_{\mathrm{prop}}) = 1 \implies \mathrm{Execute}$};
\end{scope}

% =============================================================================
% PANEL 3: JURISDICTION MATRIX & LAWS (BOTTOM RIGHT)
% (x: 15.2 to 24.0, y: 0 to 7.2) - Width 8.8 cm, Height 7.2 cm
% =============================================================================
\begin{scope}[shift={(15.2, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (8.8, 7.2);

  \begin{scope}[shift={(0.3, 0.35)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (8.2, 6.0);

    % Component 1: Jurisdiction Matrix (Top: 3.45 to 5.80, height 2.35 cm)
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.5pt, rounded corners=1.5pt] (0.2, 3.45) rectangle (8.0, 5.80);
    \node[anchor=north west, font=\fontsize{4.8}{5.8}\selectfont\bfseries, text=OxfordNavy] at (0.3, 5.65)
      {FINE-TUNING JURISDICTION MATRIX:};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=Amethyst] at (0.3, 5.15) {Weights $W_\theta$:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (2.2, 5.15) {Procedure (Which tool, how to localize stack traces)};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=MutedTeal] at (0.3, 4.65) {Context $c_t$:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (2.2, 4.65) {Contracts (Live OpenAPI schemas, user goals, scopes)};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=ForestSage] at (0.3, 4.15) {Runtime $E_{\mathrm{host}}$:};
    \node[anchor=west, font=\fontsize{4.0}{4.8}\selectfont, text=DarkSlate] at (2.2, 4.15) {Enforcement (Grammar masks, microVMs, sealed tests)};

    \node[anchor=west, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=Burgundy] at (0.3, 3.70)
      {Non-Fungible Principle: No home can substitute for another};

    % Component 2: 3 Invariable Laws (Bottom: 0.20 to 3.25, height 3.05 cm)
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.5pt, rounded corners=1.5pt] (0.2, 0.20) rectangle (8.0, 3.25);
    \node[anchor=north west, font=\fontsize{4.8}{5.8}\selectfont\bfseries, text=OxfordNavy] at (0.3, 3.10)
      {INVARIABLE LAWS OF AGENT ARCHITECTURE:};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=Burgundy] at (0.3, 2.65)
      {1. The Invariant Boundary Law:};
    \node[anchor=west, font=\fontsize{3.8}{4.5}\selectfont, text=DarkSlate] at (0.3, 2.35)
      {What must hold when the model is wrong cannot depend on the model.};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=WarmOchre] at (0.3, 1.95)
      {2. The Stale-Schema Law:};
    \node[anchor=west, font=\fontsize{3.8}{4.5}\selectfont, text=DarkSlate] at (0.3, 1.65)
      {Hardcoding schemas in $W_\theta$ guarantees breakage upon API version bumps.};

    \node[anchor=west, font=\fontsize{4.2}{5.0}\selectfont\bfseries, text=ForestSage] at (0.3, 1.25)
      {3. The Authority Decoupling Law:};
    \node[anchor=west, font=\fontsize{3.8}{4.5}\selectfont, text=DarkSlate] at (0.3, 0.95)
      {SFT raises proposal quality; it confers zero host authority.};

    \node[anchor=west, font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=SlateSteel] at (0.3, 0.55)
      {Invariant: $\mathrm{Auth}(E_{\mathrm{host}}) \cap W_\theta = \emptyset$ (Host always supervenes)};
  \end{scope}

  % Header Badges Drawn on Top of Base
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{6.8}{8.0}\selectfont\bfseries, rounded corners=2pt, inner xsep=4pt, inner ysep=2.5pt]
    at (0.4, 7.08) {3. JURISDICTION MATRIX \& LAWS};
  \node[anchor=north east, font=\fontsize{5.2}{6.2}\selectfont\bfseries, text=SlateSteel]
    at (8.4, 7.02) {Authority: $\mathrm{Auth} \cap W_\theta = \emptyset$};
\end{scope}

\end{tikzpicture}
\end{document}
"""

with open(TEX_PATH, "w") as f:
    f.write(TIKZ_CONTENT)

print(f"Wrote TikZ source to {TEX_PATH}")

build_dir = "/tmp/build_three_homes"
os.makedirs(build_dir, exist_ok=True)

cmd_latex = f"pdflatex -interaction=nonstopmode -output-directory={build_dir} {TEX_PATH}"
print(f"Compiling LaTeX: {cmd_latex}")
res = subprocess.run(cmd_latex, shell=True, capture_output=True, text=True)
if res.returncode != 0:
    print("LaTeX compilation failed:")
    print(res.stdout[-1500:])
    sys.exit(1)

pdf_built = os.path.join(build_dir, "three_homes_of_agent_knowledge.pdf")

# Convert PDF to high-res PNG (250 DPI) using pdftoppm
png_prefix = os.path.join(build_dir, "three_homes_page")
cmd_pdftoppm = f"pdftoppm -png -r 250 {pdf_built} {png_prefix}"
subprocess.run(cmd_pdftoppm, shell=True, check=True)
png_built = f"{png_prefix}-1.png"

# Convert PDF to SVG using inkscape
svg_built = os.path.join(build_dir, "three_homes_of_agent_knowledge.svg")
cmd_inkscape = f"inkscape {pdf_built} --export-plain-svg={svg_built}"
subprocess.run(cmd_inkscape, shell=True, check=True)

# Sanitize SVG if needed
with open(svg_built, "rb") as f:
    svg_data = f.read()
svg_sanitized = re.sub(b"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", svg_data)
with open(svg_built, "wb") as f:
    f.write(svg_sanitized)

# Copy to final destinations
targets = [
    (png_built, os.path.join(PNG_DIR, "fig-vol3-three-homes-agent-knowledge.png")),
    (png_built, os.path.join(PNG_DIR, "three_homes_of_agent_knowledge.png")),
    (svg_built, os.path.join(SVG_DIR, "fig-vol3-three-homes-agent-knowledge.svg")),
    (svg_built, os.path.join(SVG_DIR, "three_homes_of_agent_knowledge.svg")),
    (pdf_built, os.path.join(PDF_DIR, "fig-vol3-three-homes-agent-knowledge.pdf")),
    (pdf_built, os.path.join(PDF_DIR, "three_homes_of_agent_knowledge.pdf")),
]

import shutil
for src, dst in targets:
    shutil.copyfile(src, dst)
    print(f"Copied {src} -> {dst}")

print("Successfully built and deployed Chapter 15 Three Homes schematic!")
