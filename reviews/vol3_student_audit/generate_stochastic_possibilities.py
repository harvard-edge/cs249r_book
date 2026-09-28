#!/usr/bin/env python3
"""
Generate 3 distinct publication-grade architectural possibilities for
The Stochastic Computer Architecture (Chapter 1, Section 1.2).

Standards:
- Canonical academic palette (Oxford Navy, Slate Steel, Muted Teal, Deep Burgundy, Warm Ochre, Forest Sage)
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy (8.5pt titles, 7.5pt components, 6.2pt monospace signals)
- Authentic IEEE/ACM computer architecture glyphs (ALU trapezoids, 3D HBM stacks, DIMM pins, clock waveforms, privilege rings)
- ZERO bullet points, ZERO paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys

TEX_DIR = "/tmp/stochastic_options"
os.makedirs(TEX_DIR, exist_ok=True)

# =============================================================================
# OPTION A: CLASSIC COMPUTER ARCHITECTURE SCHEMATIC (HENNESSY & PATTERSON)
# =============================================================================
TIKZ_OPTION_A = r"""\documentclass[tikz,border=10pt]{standalone}
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
\definecolor{GoldPin}{HTML}{D97706}
\definecolor{DarkSlate}{HTML}{0F172A}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth,
  node distance=0.5cm
]

% -------------------------------------------------------------
% TOP TIER: STOCHASTIC PROCESSOR & CONTEXT MEMORY
% -------------------------------------------------------------

% 1. STOCHASTIC PROCESSING UNIT (SPU)
\begin{scope}[shift={(0, 7.6)}]
  % Enclosing boundary
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt] 
    (0, 0) rectangle (11.8, 6.6);
  
  % Header badge
  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.3) {STOCHASTIC PROCESSING UNIT (RING 3 CORE)};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 6.2) {Unprivileged Matrix Engine};

  % Unit 1A: Context Register File & Program Counter
  \begin{scope}[shift={(0.5, 3.4)}]
    \draw[fill=white, draw=SlateSteel, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (3.3, 2.3);
    \node[anchor=north west, text=OxfordNavy, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (0.2, 2.1) {Context Register File};
    
    % Register cells [c_t]
    \foreach \i/\label in {0/Prompt, 1/History, 2/Ret $o_t$} {
      \draw[fill=CoolParchment, draw=SlateRim, line width=0.5pt] (0.2, 0.95 - \i*0.42) rectangle (3.1, 1.3 - \i*0.42);
      \node[anchor=west, text=DarkSlate, font=\fontsize{5.8}{7}\selectfont\ttfamily] at (0.3, 1.12 - \i*0.42) {R\i: \label};
    }
    
    % Program counter
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1pt] (0.2, 0.15) rectangle (3.1, 0.45);
    \node[anchor=west, text=WarmOchre, font=\fontsize{5.8}{7}\selectfont\ttfamily\bfseries] at (0.3, 0.3) {PC: Step $t = 14$};
  \end{scope}

  % Unit 1B: Systolic Tensor Core Array (W_\theta)
  \begin{scope}[shift={(4.2, 3.4)}]
    \draw[fill=white, draw=OxfordNavy, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (3.4, 2.3);
    \node[anchor=north, text=OxfordNavy, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (1.7, 2.1) {Systolic Tensor Array};
    \node[anchor=north, text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont] at (1.7, 1.7) {Dense Weights ($W_\theta$)};
    
    % 3x3 PE matrix
    \foreach \r in {0, 1, 2} {
      \foreach \c in {0, 1, 2} {
        \draw[fill=OxfordNavy!15, draw=OxfordNavy!60, line width=0.4pt, rounded corners=0.8pt]
          (0.35 + \c*0.9, 0.2 + \r*0.42) rectangle (0.35 + \c*0.9 + 0.8, 0.2 + \r*0.42 + 0.35);
        \node[text=OxfordNavy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0.35 + \c*0.9 + 0.4, 0.2 + \r*0.42 + 0.175) {PE};
      }
    }
  \end{scope}

  % Unit 1C: Trapezoidal Softmax Sampler & Grammar Mask
  \begin{scope}[shift={(8.0, 3.4)}]
    \draw[fill=white, draw=Burgundy, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (3.3, 2.3);
    \node[anchor=north, text=Burgundy, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (1.65, 2.1) {Sampling \& Logit Mask};
    
    % Trapezoidal ALU-like Sampler
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt] 
      (0.4, 1.7) -- (2.9, 1.7) -- (2.3, 0.8) -- (1.0, 0.8) -- cycle;
    \node[text=Burgundy, font=\fontsize{6.0}{7}\selectfont\bfseries] at (1.65, 1.35) {Softmax Simplex};
    \node[text=DarkSlate, font=\fontsize{5.0}{6}\selectfont] at (1.65, 1.05) {$p = \mathrm{softmax}(z/T)$};

    % Hardware PDA Filter
    \draw[fill=ForestSage!15, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 0.15) rectangle (3.0, 0.65);
    \node[text=ForestSage, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (1.65, 0.4) {PDA Grammar Clamping};
  \end{scope}

  % Inter-block signals inside SPU
  \draw[->, draw=SlateSteel, line width=0.9pt] (3.8, 4.55) -- (4.2, 4.55);
  \draw[->, draw=OxfordNavy, line width=0.9pt] (7.6, 4.55) -- (8.0, 4.55)
    node[midway, above=1pt, font=\fontsize{5.2}{6}\selectfont\ttfamily] {Logits $z_t$};

  % SPU Output Proposal Pin
  \draw[fill=Burgundy, draw=Burgundy, rounded corners=1.5pt] (3.8, 0.4) rectangle (8.0, 1.2);
  \node[text=white, font=\fontsize{6.8}{8}\selectfont\bfseries] at (5.9, 0.8) {PROPOSAL PORT ($a_{\mathrm{prop}}$)};
  \node[text=Burgundy, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (5.9, 0.15) {Zero Ambient Authority (Escrow Gate)};
  
  \draw[->, draw=Burgundy, line width=1.1pt] (9.65, 3.4) |- (5.9, 1.2);
\end{scope}


% 2. CONTEXT MEMORY HIERARCHY
\begin{scope}[shift={(12.6, 7.6)}]
  \draw[fill=CoolParchment, draw=MutedTeal, line width=1.1pt, rounded corners=3pt] 
    (0, 0) rectangle (11.8, 6.6);
  
  \node[anchor=north west, fill=MutedTeal, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.3) {CONTEXT MEMORY HIERARCHY};
  \node[anchor=north east, text=MutedTeal, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 6.2) {Multi-Tiered Staging};

  % Tier 1: L1 HBM3e 3D Stack
  \begin{scope}[shift={(0.5, 4.2)}]
    \draw[fill=white, draw=MutedTeal, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 1.7);
    \node[anchor=west, text=MutedTeal, font=\fontsize{7.0}{8}\selectfont\bfseries] at (0.3, 1.4) {L1: Active KV-Cache (HBM3e)};
    \node[anchor=east, text=DarkSlate, font=\fontsize{6.0}{7}\selectfont\ttfamily] at (10.5, 1.4) {3.2 TB/s $\cdot$ PagedAttention};
    
    % 3D Stack Graphic
    \foreach \l in {0, 1, 2, 3} {
      \draw[fill=TealL0, draw=MutedTeal, line width=0.4pt] (0.3 + \l*2.6, 0.3) rectangle (0.3 + \l*2.6 + 2.4, 0.95);
      \node[text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.3 + \l*2.6 + 1.2, 0.62) {DRAM Die \l};
      % Micro TSVs
      \fill[GoldPin] (0.5 + \l*2.6, 0.15) rectangle (0.6 + \l*2.6, 0.3);
      \fill[GoldPin] (2.4 + \l*2.6, 0.15) rectangle (2.5 + \l*2.6, 0.3);
    }
  \end{scope}

  % Tier 2: L2 Episodic Host RAM (DDR5 DIMM)
  \begin{scope}[shift={(0.5, 2.2)}]
    \draw[fill=white, draw=MutedTeal, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 1.7);
    \node[anchor=west, text=MutedTeal, font=\fontsize{7.0}{8}\selectfont\bfseries] at (0.3, 1.4) {L2: Episodic Session RAM (DDR5 / CXL)};
    \node[anchor=east, text=DarkSlate, font=\fontsize{6.0}{7}\selectfont\ttfamily] at (10.5, 1.4) {64 GB/s $\cdot$ Radix Prefix Tree};
    
    % DIMM PCB graphic
    \draw[fill=ForestSage!15, draw=ForestSage, line width=0.6pt] (0.3, 0.3) rectangle (10.5, 1.0);
    \foreach \x in {0.6, 2.0, 3.4, 4.8, 6.4, 7.8, 9.2} {
      \draw[fill=DarkSlate, draw=black] (\x, 0.45) rectangle (\x+1.0, 0.9);
      \node[text=white, font=\fontsize{4.8}{5.5}\selectfont] at (\x+0.5, 0.67) {DRAM};
    }
    % Gold fingers
    \foreach \p in {0.5, 0.8, ..., 10.2} {
      \fill[GoldPin] (\p, 0.18) rectangle (\p+0.15, 0.3);
    }
  \end{scope}

  % Tier 3: L3 Persistent Store (NVMe & Vector Index)
  \begin{scope}[shift={(0.5, 0.3)}]
    \draw[fill=white, draw=MutedTeal, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 1.6);
    \node[anchor=west, text=MutedTeal, font=\fontsize{7.0}{8}\selectfont\bfseries] at (0.3, 1.35) {L3: Persistent Store (NVMe \& Vector Index)};
    \node[anchor=east, text=DarkSlate, font=\fontsize{6.0}{7}\selectfont\ttfamily] at (10.5, 1.35) {7 GB/s $\cdot$ WAL Log \& HNSW};
    
    % NVMe Drive
    \draw[fill=DarkSlate, draw=black, rounded corners=1pt] (0.4, 0.25) rectangle (4.8, 0.95);
    \fill[GoldPin] (0.4, 0.4) rectangle (0.6, 0.8);
    \node[text=white, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (2.6, 0.6) {NVMe Write-Ahead Log (WAL)};
    
    % Vector Platters
    \begin{scope}[shift={(7.5, 0.6)}]
      \foreach \y in {-0.2, 0.0, 0.2} {
        \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt] (0, \y) ellipse (1.6cm and 0.15cm);
      }
      \node[text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (0, 0.0) {Vector Index (HNSW)};
    \end{scope}
  \end{scope}
\end{scope}

% High-Bandwidth Attention Interconnect between SPU and HBM3e
\draw[<->, draw=MutedTeal, line width=2.0pt] (11.8, 12.0) -- (12.6, 12.0);
\node[fill=white, text=MutedTeal, font=\fontsize{5.5}{6.5}\selectfont\bfseries, inner sep=1pt] at (12.2, 12.0) {3.2 TB/s};


% -------------------------------------------------------------
% MIDDLE: SYSTEM TRAJECTORY BUS (\tau)
% -------------------------------------------------------------
\draw[fill=OxfordNavy, draw=GoldPin, line width=1.0pt, rounded corners=2pt]
  (0, 6.2) rectangle (24.4, 6.9);
\node[text=white, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (12.2, 6.55) {
  SYSTEM TRAJECTORY BUS $\cdot$ [Tokens $x_t$ $\mid$ KV-Pointers $\mid$ Action Proposals $a_t$ $\mid$ Observation Feed $o_t$]
};

% SPU Proposal link to Bus
\draw[->, draw=Burgundy, line width=1.5pt] (5.9, 7.6) -- (5.9, 6.9);
% Memory link to Bus
\draw[<->, draw=MutedTeal, line width=1.5pt] (18.5, 7.6) -- (18.5, 6.9);


% -------------------------------------------------------------
% BOTTOM TIER: SUPERVISOR & SANDBOX AIRLOCK
% -------------------------------------------------------------

% 3. HOST SUPERVISOR PLANE (RING 0 KERNEL)
\begin{scope}[shift={(0, 0.2)}]
  \draw[fill=CoolParchment, draw=WarmOchre, line width=1.1pt, rounded corners=3pt] 
    (0, 0) rectangle (11.8, 5.4);
  
  \node[anchor=north west, fill=WarmOchre, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 5.1) {HOST SUPERVISOR (RING 0 KERNEL)};
  \node[anchor=north east, text=WarmOchre, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 5.0) {Deterministic Governor};

  % Unit 3A: Agent Control Block (ACB)
  \begin{scope}[shift={(0.5, 0.5)}]
    \draw[fill=white, draw=WarmOchre, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.2, 4.0);
    \node[anchor=north west, text=WarmOchre, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (0.3, 3.75) {Agent Control Block (ACB)};

    % Monotonic Deadline Clock Pulse
    \begin{scope}[shift={(0.4, 2.3)}]
      \draw[fill=CoolParchment, draw=SlateRim, rounded corners=1pt] (0, 0) rectangle (4.4, 0.9);
      % Square wave glyph
      \draw[draw=WarmOchre, line width=0.9pt] 
        (0.2, 0.2) -- (0.5, 0.2) -- (0.5, 0.7) -- (0.8, 0.7) -- (0.8, 0.2) -- (1.1, 0.2) -- (1.1, 0.7) -- (1.4, 0.7);
      \node[anchor=west, text=DarkSlate, font=\fontsize{5.8}{7}\selectfont\ttfamily] at (1.6, 0.45) {Deadline $T \le T_{\max}$};
    \end{scope}

    % Token Fuel Gauge
    \begin{scope}[shift={(0.4, 1.2)}]
      \draw[fill=CoolParchment, draw=SlateRim, rounded corners=1pt] (0, 0) rectangle (4.4, 0.9);
      \draw[fill=ForestSage!70, draw=none] (0.2, 0.3) rectangle (3.0, 0.65);
      \draw[draw=SlateSteel, line width=0.5pt] (0.2, 0.3) rectangle (4.2, 0.65);
      \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.2, 0.15) {Token Fuel: $65\%$ consumed};
    \end{scope}

    % Watchdog Interrupt
    \begin{scope}[shift={(0.4, 0.2)}]
      \draw[fill=BurgundyL0, draw=Burgundy, rounded corners=1pt] (0, 0) rectangle (4.4, 0.8);
      \draw[draw=Burgundy, line width=0.8pt] (0.3, 0.4) -- (1.0, 0.4);
      \draw[fill=white, draw=Burgundy, line width=0.8pt] (1.1, 0.4) circle (1.8pt);
      \node[anchor=west, text=Burgundy, font=\fontsize{5.8}{7}\selectfont\ttfamily\bfseries] at (1.4, 0.4) {SIG\_PREEMPT (Watchdog)};
    \end{scope}
  \end{scope}

  % Unit 3B: Privilege Protection Rings & Capability MMU
  \begin{scope}[shift={(6.1, 0.5)}]
    \draw[fill=white, draw=OxfordNavy, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.2, 4.0);
    \node[anchor=north west, text=OxfordNavy, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (0.3, 3.75) {Protection Rings};

    % Concentric Rings
    \begin{scope}[shift={(2.6, 2.1)}]
      \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt] (0, 0) circle (1.3cm);
      \draw[fill=SageL0, draw=ForestSage, line width=0.8pt] (0, 0) circle (0.85cm);
      \draw[fill=white, draw=OxfordNavy, line width=0.9pt] (0, 0) circle (0.42cm);
      \node[text=OxfordNavy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0) {Ring 0};
      \node[text=ForestSage, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0.62) {Ring 1};
      \node[text=Burgundy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 1.05) {Ring 3};
    \end{scope}

    % Capability Lease Output
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1pt] (0.4, 0.2) rectangle (4.8, 0.8);
    \node[text=WarmOchre, font=\fontsize{6.0}{7}\selectfont\bfseries] at (2.6, 0.5) {Signed Capability Lease};
  \end{scope}
\end{scope}


% 4. SANDBOX AIRLOCK & PERIPHERAL CHAMBER
\begin{scope}[shift={(12.6, 0.2)}]
  % Grounded double-walled boundary stroke
  \draw[fill=CoolParchment, draw=Burgundy, double, double distance=1.5pt, line width=0.8pt, rounded corners=3pt] 
    (0, 0) rectangle (11.8, 5.4);
  
  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 5.1) {SANDBOX AIRLOCK CHAMBER ($W \oplus X$)};
  \node[anchor=north east, text=Burgundy, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 5.0) {Isolation Perimeter};

  % Ingress Gate Valve
  \begin{scope}[shift={(0.5, 0.6)}]
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (1.8, 3.8);
    \node[text=WarmOchre, font=\fontsize{6.5}{7.5}\selectfont\bfseries, align=center] at (0.9, 2.5) {INGRESS\\VALVE};
    \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont, align=center] at (0.9, 1.2) {Policy\\Check\\Schema};
  \end{scope}

  % Isolated MicroVM Enclave
  \begin{scope}[shift={(2.8, 0.6)}]
    \draw[fill=DarkSlate, draw=black, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (6.2, 3.8);
    \node[anchor=north, text=white, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (3.1, 3.6) {ISOLATED MICROVM ENCLAVE};

    % Terminal Sub-box
    \begin{scope}[shift={(0.3, 0.4)}]
      \draw[fill=black!90, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (2.6, 2.7);
      \fill[Burgundy] (0.2, 2.45) circle (1.2pt);
      \fill[GoldPin] (0.4, 2.45) circle (1.2pt);
      \fill[ForestSage] (0.6, 2.45) circle (1.2pt);
      \node[anchor=north west, text=ForestSage, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.1, 2.2) {\$ pytest};
      \node[anchor=north west, text=white, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.1, 1.6) {5 passed};
      \node[anchor=north west, text=GoldPin, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.1, 1.0) {\$ exit 0};
    \end{scope}

    % Web DOM / Browser Sub-box
    \begin{scope}[shift={(3.2, 0.4)}]
      \draw[fill=white, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (2.6, 2.7);
      \fill[SlateRim!40] (0, 2.3) rectangle (2.6, 2.7);
      \node[text=OxfordNavy, font=\fontsize{4.5}{5.5}\selectfont\ttfamily] at (1.3, 2.5) {https://api.db};
      \node[text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont\bfseries, align=center] at (1.3, 1.3) {Web DOM\\MCP RPC};
    \end{scope}
  \end{scope}

  % Egress Filter Valve
  \begin{scope}[shift={(9.5, 0.6)}]
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (1.8, 3.8);
    \node[text=ForestSage, font=\fontsize{6.5}{7.5}\selectfont\bfseries, align=center] at (0.9, 2.5) {EGRESS\\VALVE};
    \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont, align=center] at (0.9, 1.2) {Output\\Sanitizer\\$o_{t+1}$};
  \end{scope}
\end{scope}

% Bus connection down to Supervisor
\draw[->, draw=OxfordNavy, line width=1.5pt] (5.9, 6.2) -- (5.9, 5.6);
% Supervisor capability dispatch to Ingress Valve
\draw[->, draw=WarmOchre, line width=1.3pt] (11.8, 2.6) -- (13.1, 2.6)
  node[midway, above=1pt, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=WarmOchre] {Dispatch};

% Egress Observation Return loop back to Trajectory Bus
\draw[->, draw=ForestSage, line width=1.5pt] (23.9, 4.6) -- (23.9, 6.2)
  node[midway, right=2pt, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] {Sanitized $o_{t+1}$};

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# OPTION B: PHYSICAL SILICON & MOTHERBOARD DIE BLUEPRINT
# =============================================================================
TIKZ_OPTION_B = r"""\documentclass[tikz,border=10pt]{standalone}
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
\definecolor{GoldPin}{HTML}{D97706}
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{SiliconCarrier}{HTML}{0F172A}
\definecolor{InterposerTrace}{HTML}{F59E0B}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth
]

% -------------------------------------------------------------
% TOP HALF: CO-PACKAGED MULTI-CHIP MODULE (SPU + HBM ON INTERPOSER)
% -------------------------------------------------------------

% Physical Multi-Chip Carrier (Ceramic BGA Package)
\draw[fill=CoolParchment, draw=SlateSteel, line width=1.2pt, rounded corners=4pt]
  (0, 7.2) rectangle (24.4, 14.8);

\node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
  at (0.4, 14.4) {1. SPU NEURAL ACCELERATOR PACKAGE (CO-PACKAGED MULTI-CHIP MODULE)};
\node[anchor=north east, text=SlateSteel, font=\fontsize{6.5}{8}\selectfont\bfseries]
  at (24.0, 14.3) {Silicon Interposer Architecture};

% Ceramic Substrate Body (Dark Slate)
\draw[fill=SiliconCarrier, draw=black, line width=1.0pt, rounded corners=3pt]
  (0.6, 7.8) rectangle (15.6, 13.8);

% Peripheral Gold BGA Micro-Pins
\foreach \x in {1.0, 1.6, ..., 15.2} {
  \fill[GoldPin] (\x, 7.65) rectangle (\x+0.35, 7.8);
  \fill[GoldPin] (\x, 13.8) rectangle (\x+0.35, 13.95);
}
\foreach \y in {8.2, 8.8, ..., 13.4} {
  \fill[GoldPin] (0.45, \y) rectangle (0.6, \y+0.35);
  \fill[GoldPin] (15.6, \y) rectangle (15.75, \y+0.35);
}

% SPU Compute Die (Logic Die)
\draw[fill=OxfordNavy, draw=SlateRim, line width=0.8pt, rounded corners=2pt]
  (1.0, 8.2) rectangle (10.0, 13.4);
\node[anchor=north, text=white, font=\fontsize{7.5}{9}\selectfont\bfseries] at (5.5, 13.2) {NEURAL ACCELERATOR CORE DIE ($W_\theta$)};

% Inside Logic Die: Left Systolic Grid
\begin{scope}[shift={(1.4, 8.6)}]
  \draw[fill=SlateSteel!60, draw=SlateRim, rounded corners=1.5pt] (0, 0) rectangle (3.8, 3.8);
  \node[anchor=north, text=white, font=\fontsize{6.8}{8}\selectfont\bfseries] at (1.9, 3.65) {Systolic Tensor Grid};
  \foreach \r in {0, 1, 2, 3} {
    \foreach \c in {0, 1, 2, 3} {
      \draw[fill=OxfordNavy, draw=SlateRim!60, line width=0.3pt]
        (0.25 + \c*0.85, 0.25 + \r*0.75) rectangle (0.25 + \c*0.85 + 0.72, 0.25 + \r*0.75 + 0.6);
      \node[text=white, font=\fontsize{4.2}{5}\selectfont\bfseries] at (0.25 + \c*0.85 + 0.36, 0.25 + \r*0.75 + 0.3) {PE};
    }
  }
\end{scope}

% Inside Logic Die: Right Sampler & Registers
\begin{scope}[shift={(5.6, 8.6)}]
  \draw[fill=SlateSteel!60, draw=SlateRim, rounded corners=1.5pt] (0, 0) rectangle (4.0, 3.8);
  \node[anchor=north, text=white, font=\fontsize{6.8}{8}\selectfont\bfseries] at (2.0, 3.65) {Sampling \& Control Logic};

  % Context Registers
  \draw[fill=OxfordNavy, draw=SlateRim, rounded corners=1pt] (0.3, 2.3) rectangle (3.7, 3.1);
  \node[anchor=west, text=white, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (0.4, 2.85) {Context Regs $[c_t]$};
  \node[anchor=west, text=InterposerTrace, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.4, 2.5) {PC: Step $t = 14$};

  % Softmax Sampler Curve
  \draw[fill=BurgundyL0, draw=Burgundy, rounded corners=1pt] (0.3, 1.2) rectangle (3.7, 2.1);
  \draw[draw=Burgundy, line width=0.9pt]
    (0.5, 1.4) .. controls (1.2, 1.4) and (1.7, 1.95) .. (2.0, 1.95)
    .. controls (2.3, 1.95) and (2.8, 1.4) .. (3.5, 1.4);
  \node[anchor=north, text=Burgundy, font=\fontsize{5.2}{6}\selectfont\bfseries] at (2.0, 1.4) {Softmax Sampling: $x_{t+1} \sim P$};

  % PDA Mask
  \draw[fill=ForestSage!20, draw=ForestSage, rounded corners=1pt] (0.3, 0.3) rectangle (3.7, 1.0);
  \node[text=ForestSage, font=\fontsize{5.8}{7}\selectfont\bfseries] at (2.0, 0.65) {Hardware PDA Mask};
\end{scope}

% Silicon Interposer Trace Bridge
\draw[fill=SiliconCarrier, draw=InterposerTrace, line width=0.8pt] (10.0, 9.8) rectangle (11.0, 11.8);
\node[text=InterposerTrace, font=\fontsize{5.0}{6}\selectfont\bfseries, align=center] at (10.5, 10.8) {TSV\\Bus};

% Co-Packaged 3D HBM3e Die Stack
\begin{scope}[shift={(11.0, 8.4)}]
  \draw[fill=MutedTeal!25, draw=MutedTeal, line width=0.9pt, rounded corners=2pt] (0, 0) rectangle (4.2, 4.8);
  \node[anchor=north, text=MutedTeal, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (2.1, 4.6) {L1: HBM3e 3D STACK};
  \node[anchor=north, text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont] at (2.1, 4.2) {PagedAttention KV-Cache};

  % 4 DRAM layers
  \foreach \y [count=\idx] in {0.8, 1.6, 2.4, 3.2} {
    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt, rounded corners=1pt] (0.3, \y) rectangle (3.9, \y+0.6);
    \node[text=DarkSlate, font=\fontsize{5.8}{7}\selectfont\bfseries] at (2.1, \y+0.3) {DRAM Layer \idx \ ($16\text{ GB}$)};
    \fill[GoldPin] (0.6, \y-0.2) rectangle (0.75, \y);
    \fill[GoldPin] (3.45, \y-0.2) rectangle (3.6, \y);
  }
  % Base Logic Die
  \draw[fill=MutedTeal!60, draw=MutedTeal, line width=0.7pt, rounded corners=1pt] (0.3, 0.15) rectangle (3.9, 0.65);
  \node[text=white, font=\fontsize{5.8}{7}\selectfont\bfseries] at (2.1, 0.4) {Base Logic Die (Radix Tree)};
\end{scope}

% External Memory Subsystem on Motherboard (L2 & L3)
\begin{scope}[shift={(16.2, 7.8)}]
  \draw[fill=white, draw=SlateRim, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (7.6, 6.0);
  \node[anchor=north west, text=MutedTeal, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 5.7) {HOST MEMORY BUS (L2 \& L3)};

  % DDR5 Module
  \begin{scope}[shift={(0.4, 3.0)}]
    \draw[fill=ForestSage!15, draw=ForestSage, line width=0.7pt, rounded corners=1.5pt] (0, 0) rectangle (6.8, 2.2);
    \node[anchor=north west, text=ForestSage, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (0.2, 2.05) {L2: Host DRAM (DDR5/CXL)};
    \node[anchor=north east, text=SlateSteel, font=\fontsize{5.2}{6}\selectfont] at (6.6, 2.05) {64 GB/s};
    \foreach \x in {0.4, 2.0, 3.6, 5.2} {
      \draw[fill=DarkSlate, draw=black] (\x, 0.5) rectangle (\x+1.2, 1.4);
      \node[text=white, font=\fontsize{5.0}{6}\selectfont] at (\x+0.6, 0.95) {DRAM};
    }
    \foreach \p in {0.4, 0.7, ..., 6.5} {
      \fill[GoldPin] (\p, 0.15) rectangle (\p+0.18, 0.35);
    }
  \end{scope}

  % NVMe M.2 Drive
  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=DarkSlate, draw=black, rounded corners=1.5pt] (0, 0) rectangle (6.8, 2.2);
    \node[anchor=north west, text=white, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (0.3, 2.05) {L3: NVMe M.2 (PCIe 5.0)};
    \node[anchor=north east, text=InterposerTrace, font=\fontsize{5.2}{6}\selectfont] at (6.5, 2.05) {7 GB/s};
    \draw[fill=SlateSteel, draw=SlateRim] (0.5, 0.4) rectangle (3.0, 1.5);
    \node[text=white, font=\fontsize{5.5}{6.5}\selectfont\ttfamily, align=center] at (1.75, 0.95) {WAL\\Log};
    \draw[fill=MutedTeal!70, draw=SlateRim] (3.5, 0.4) rectangle (6.3, 1.5);
    \node[text=white, font=\fontsize{5.5}{6.5}\selectfont\bfseries, align=center] at (4.9, 0.95) {Vector Index\\(HNSW)};
  \end{scope}
\end{scope}

% Proposal Pin at bottom of SPU
\draw[fill=Burgundy, draw=Burgundy, rounded corners=1.5pt] (5.5, 7.3) rectangle (10.5, 7.9);
\node[text=white, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (8.0, 7.6) {PROPOSAL PORT (ZERO AMBIENT AUTHORITY)};


% -------------------------------------------------------------
% MIDDLE: MOTHERBOARD TRAJECTORY BUS
% -------------------------------------------------------------
\draw[fill=OxfordNavy, draw=InterposerTrace, line width=1.1pt, rounded corners=2pt]
  (0, 6.0) rectangle (24.4, 6.7);
\node[text=white, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (12.2, 6.35) {
  SYSTEM TRAJECTORY BUS $\cdot$ [Tokens $x_t$ $\mid$ KV-Pointers $\mid$ Action Proposals $a_t$ $\mid$ Hardware Traps]
};

\draw[->, draw=Burgundy, line width=1.6pt] (8.0, 7.3) -- (8.0, 6.7);
\draw[<->, draw=MutedTeal, line width=1.6pt] (18.5, 7.8) -- (18.5, 6.7);


% -------------------------------------------------------------
% BOTTOM HALF: SUPERVISOR & ISOLATION AIRLOCK
% -------------------------------------------------------------

% 3. ROOT-OF-TRUST SUPERVISOR
\begin{scope}[shift={(0, 0.2)}]
  \draw[fill=CoolParchment, draw=WarmOchre, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (11.8, 5.4);
  \node[anchor=north west, fill=WarmOchre, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 5.1) {3. HOST SUPERVISOR \& ROOT-OF-TRUST (RING 0 KERNEL)};

  % ACB
  \begin{scope}[shift={(0.5, 0.5)}]
    \draw[fill=white, draw=WarmOchre, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.2, 4.0);
    \node[anchor=north west, text=WarmOchre, font=\fontsize{7.0}{8}\selectfont\bfseries] at (0.3, 3.75) {Agent Control Block (ACB)};

    % Square wave
    \begin{scope}[shift={(0.3, 2.3)}]
      \draw[fill=CoolParchment, draw=SlateRim] (0, 0) rectangle (4.6, 0.9);
      \draw[draw=WarmOchre, line width=0.9pt]
        (0.2, 0.2) -- (0.5, 0.2) -- (0.5, 0.7) -- (0.8, 0.7) -- (0.8, 0.2) -- (1.1, 0.2) -- (1.1, 0.7) -- (1.4, 0.7);
      \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (1.6, 0.45) {Deadline $T \le T_{\max}$};
    \end{scope}

    % Fuel Gauge
    \begin{scope}[shift={(0.3, 1.2)}]
      \draw[fill=CoolParchment, draw=SlateRim] (0, 0) rectangle (4.6, 0.9);
      \fill[ForestSage!70] (0.2, 0.3) rectangle (3.0, 0.65);
      \draw[draw=SlateSteel, line width=0.5pt] (0.2, 0.3) rectangle (4.4, 0.65);
      \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.2, 0.15) {Token Fuel: $65\%$ consumed};
    \end{scope}

    % Interrupt
    \begin{scope}[shift={(0.3, 0.2)}]
      \draw[fill=BurgundyL0, draw=Burgundy] (0, 0) rectangle (4.6, 0.8);
      \draw[draw=Burgundy, line width=0.8pt] (0.3, 0.4) -- (1.0, 0.4);
      \draw[fill=white, draw=Burgundy, line width=0.8pt] (1.1, 0.4) circle (1.8pt);
      \node[anchor=west, text=Burgundy, font=\fontsize{5.5}{6.5}\selectfont\ttfamily\bfseries] at (1.4, 0.4) {SIG\_PREEMPT (Loop Watchdog)};
    \end{scope}
  \end{scope}

  % Rings
  \begin{scope}[shift={(6.1, 0.5)}]
    \draw[fill=white, draw=OxfordNavy, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.2, 4.0);
    \node[anchor=north west, text=OxfordNavy, font=\fontsize{7.0}{8}\selectfont\bfseries] at (0.3, 3.75) {Privilege Separation};
    \begin{scope}[shift={(2.6, 2.1)}]
      \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt] (0, 0) circle (1.3cm);
      \draw[fill=SageL0, draw=ForestSage, line width=0.8pt] (0, 0) circle (0.85cm);
      \draw[fill=white, draw=OxfordNavy, line width=0.9pt] (0, 0) circle (0.42cm);
      \node[text=OxfordNavy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0) {Ring 0};
      \node[text=ForestSage, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0.62) {Ring 1};
      \node[text=Burgundy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 1.05) {Ring 3};
    \end{scope}
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt] (0.4, 0.2) rectangle (4.8, 0.8);
    \node[text=WarmOchre, font=\fontsize{5.8}{7}\selectfont\bfseries] at (2.6, 0.5) {Signed Capability Token};
  \end{scope}
\end{scope}

% 4. AIRLOCK ENCLAVE
\begin{scope}[shift={(12.6, 0.2)}]
  \draw[fill=CoolParchment, draw=Burgundy, double, double distance=1.5pt, line width=0.8pt, rounded corners=3pt]
    (0, 0) rectangle (11.8, 5.4);
  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 5.1) {4. ISOLATED PERIPHERAL AIRLOCK ($W \oplus X$)};

  % Ingress
  \begin{scope}[shift={(0.5, 0.6)}]
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (1.8, 3.8);
    \node[text=WarmOchre, font=\fontsize{6.5}{7.5}\selectfont\bfseries, align=center] at (0.9, 2.5) {INGRESS\\VALVE};
    \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont, align=center] at (0.9, 1.2) {Policy\\Check};
  \end{scope}

  % MicroVM
  \begin{scope}[shift={(2.8, 0.6)}]
    \draw[fill=DarkSlate, draw=black, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (6.2, 3.8);
    \node[anchor=north, text=white, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (3.1, 3.6) {ISOLATED MICROVM ENCLAVE};
    \begin{scope}[shift={(0.3, 0.4)}]
      \draw[fill=black!90, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (2.6, 2.7);
      \fill[Burgundy] (0.2, 2.45) circle (1.2pt);
      \fill[GoldPin] (0.4, 2.45) circle (1.2pt);
      \fill[ForestSage] (0.6, 2.45) circle (1.2pt);
      \node[anchor=north west, text=ForestSage, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.1, 2.2) {\$ pytest};
      \node[anchor=north west, text=white, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.1, 1.6) {5 passed};
      \node[anchor=north west, text=GoldPin, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.1, 1.0) {\$ exit 0};
    \end{scope}
    \begin{scope}[shift={(3.2, 0.4)}]
      \draw[fill=white, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (2.6, 2.7);
      \fill[SlateRim!40] (0, 2.3) rectangle (2.6, 2.7);
      \node[text=OxfordNavy, font=\fontsize{4.5}{5.5}\selectfont\ttfamily] at (1.3, 2.5) {https://api.db};
      \node[text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont\bfseries, align=center] at (1.3, 1.3) {Web DOM\\MCP RPC};
    \end{scope}
  \end{scope}

  % Egress
  \begin{scope}[shift={(9.5, 0.6)}]
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (1.8, 3.8);
    \node[text=ForestSage, font=\fontsize{6.5}{7.5}\selectfont\bfseries, align=center] at (0.9, 2.5) {EGRESS\\VALVE};
    \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont, align=center] at (0.9, 1.2) {Sanitizer\\$o_{t+1}$};
  \end{scope}
\end{scope}

\draw[->, draw=OxfordNavy, line width=1.5pt] (5.9, 6.0) -- (5.9, 5.6);
\draw[->, draw=WarmOchre, line width=1.3pt] (11.8, 2.6) -- (13.1, 2.6)
  node[midway, above=1pt, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=WarmOchre] {Dispatch};
\draw[->, draw=ForestSage, line width=1.5pt] (23.9, 4.6) -- (23.9, 6.0)
  node[midway, right=2pt, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] {Sanitized $o_{t+1}$};

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# OPTION C: CLOSED-LOOP TRAJECTORY CONTROL DATAPATH (DUAL-PLANE ARCHITECTURE)
# =============================================================================
TIKZ_OPTION_C = r"""\documentclass[tikz,border=10pt]{standalone}
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
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{GoldPin}{HTML}{D97706}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth
]

% -------------------------------------------------------------
% TOP PLANE: UNPRIVILEGED STOCHASTIC INFERENCE (RING 3)
% -------------------------------------------------------------
\draw[fill=CoolParchment, draw=OxfordNavy, line width=1.2pt, rounded corners=4pt]
  (0, 8.2) rectangle (24.4, 13.8);

\node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
  at (0.4, 13.4) {UNPRIVILEGED STOCHASTIC INFERENCE PLANE (RING 3)};
\node[anchor=north east, text=OxfordNavy, font=\fontsize{6.5}{8}\selectfont\bfseries]
  at (24.0, 13.3) {Stateless Tensor Engine};

% Context Assembly Register
\begin{scope}[shift={(0.6, 8.8)}]
  \draw[fill=white, draw=SlateSteel, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (6.2, 3.8);
  \node[anchor=north west, text=OxfordNavy, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 3.5) {Context Buffer $[c_t]$};
  
  \draw[fill=CoolParchment, draw=SlateRim] (0.3, 2.3) rectangle (5.9, 2.9);
  \node[anchor=west, text=DarkSlate, font=\fontsize{6.0}{7}\selectfont\ttfamily] at (0.4, 2.6) {Task Goal $g$ $\cdot$ System Prompt};
  
  \draw[fill=CoolParchment, draw=SlateRim] (0.3, 1.4) rectangle (5.9, 2.0);
  \node[anchor=west, text=DarkSlate, font=\fontsize{6.0}{7}\selectfont\ttfamily] at (0.4, 1.7) {Trajectory History $\tau_t = (s, a, o)_{1..t-1}$};

  \draw[fill=SageL0, draw=ForestSage] (0.3, 0.4) rectangle (5.9, 1.1);
  \node[anchor=west, text=ForestSage, font=\fontsize{6.0}{7}\selectfont\ttfamily\bfseries] at (0.4, 0.75) {Latest Feedback: Observation $o_t$};
\end{scope}

% KV-Cache Attention Sinks
\begin{scope}[shift={(7.3, 8.8)}]
  \draw[fill=white, draw=MutedTeal, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (7.0, 3.8);
  \node[anchor=north west, text=MutedTeal, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 3.5) {PagedAttention KV-Cache (L1/L2)};

  \foreach \i in {0, 1, 2} {
    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt] (0.3, 2.0 - \i*0.8) rectangle (6.7, 2.6 - \i*0.8);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.8}{7}\selectfont\ttfamily] at (0.4, 2.3 - \i*0.8) {Block \i: Keys $[K_t]$ $\cdot$ Values $[V_t]$};
  }
\end{scope}

% Systolic Tensor Model & Softmax
\begin{scope}[shift={(14.8, 8.8)}]
  \draw[fill=white, draw=Burgundy, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (9.0, 3.8);
  \node[anchor=north west, text=Burgundy, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 3.5) {Stochastic Policy Core ($\pi_\theta$)};

  % Trapezoidal Sampler
  \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt]
    (0.4, 2.6) -- (4.2, 2.6) -- (3.5, 1.4) -- (1.1, 1.4) -- cycle;
  \node[text=Burgundy, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (2.3, 2.0) {Softmax Simplex};
  \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont] at (2.3, 1.6) {$p = \mathrm{softmax}(z/T)$};

  % Grammar Mask
  \begin{scope}[shift={(4.6, 1.4)}]
    \draw[fill=ForestSage!15, draw=ForestSage, rounded corners=1.5pt] (0, 0) rectangle (4.0, 1.2);
    \node[text=ForestSage, font=\fontsize{6.0}{7}\selectfont\bfseries, align=center] at (2.0, 0.6) {Grammar DFA Mask\\$\mathcal{L}(G)$ JSON Clamping};
  \end{scope}

  % Output proposal
  \draw[fill=Burgundy, draw=Burgundy, rounded corners=1.5pt] (1.0, 0.3) rectangle (8.0, 0.9);
  \node[text=white, font=\fontsize{6.2}{7.5}\selectfont\bfseries] at (4.5, 0.6) {ACTION PROPOSAL $a_{\mathrm{prop}} \sim \pi_\theta(\cdot \mid c_t)$};
\end{scope}

% Signal connections in top plane
\draw[->, draw=SlateSteel, line width=1.1pt] (6.8, 10.7) -- (7.3, 10.7);
\draw[->, draw=MutedTeal, line width=1.1pt] (14.3, 10.7) -- (14.8, 10.7);


% -------------------------------------------------------------
% HORIZONTAL FIREWALL: ZERO AMBIENT AUTHORITY BOUNDARY
% -------------------------------------------------------------
\draw[draw=Burgundy, line width=1.4pt, dashed] (0, 7.3) -- (24.4, 7.3);
\node[fill=Burgundy, text=white, font=\fontsize{7.0}{8.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3pt]
  at (12.2, 7.3) {ZERO AMBIENT AUTHORITY \& CAPABILITY ESCROW BOUNDARY};

% Proposal crossing the boundary
\draw[->, draw=Burgundy, line width=2.0pt] (19.3, 8.8) -- (19.3, 5.8)
  node[midway, right=3pt, font=\fontsize{6.0}{7}\selectfont\bfseries, text=Burgundy] {Escrow Check};


% -------------------------------------------------------------
% BOTTOM PLANE: DETERMINISTIC HOST SUPERVISOR & ACTUATION
% -------------------------------------------------------------
\draw[fill=CoolParchment, draw=ForestSage, line width=1.2pt, rounded corners=4pt]
  (0, 0.4) rectangle (24.4, 6.4);

\node[anchor=north west, fill=ForestSage, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
  at (0.4, 6.0) {DETERMINISTIC HOST SUPERVISOR PLANE (RING 0 KERNEL)};
\node[anchor=north east, text=ForestSage, font=\fontsize{6.5}{8}\selectfont\bfseries]
  at (24.0, 5.9) {Root of Trust};

% Host Supervisor Control Box (ACB)
\begin{scope}[shift={(0.6, 0.8)}]
  \draw[fill=white, draw=WarmOchre, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (7.0, 4.6);
  \node[anchor=north west, text=WarmOchre, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 4.3) {Agent Control Block (ACB)};

  % Clock
  \begin{scope}[shift={(0.3, 3.0)}]
    \draw[fill=CoolParchment, draw=SlateRim] (0, 0) rectangle (6.4, 0.9);
    \draw[draw=WarmOchre, line width=0.9pt]
      (0.2, 0.2) -- (0.5, 0.2) -- (0.5, 0.7) -- (0.8, 0.7) -- (0.8, 0.2) -- (1.1, 0.2) -- (1.1, 0.7) -- (1.4, 0.7);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (1.6, 0.45) {Deadline Clock: $T \le T_{\max}$};
  \end{scope}

  % Fuel
  \begin{scope}[shift={(0.3, 1.8)}]
    \draw[fill=CoolParchment, draw=SlateRim] (0, 0) rectangle (6.4, 0.9);
    \fill[ForestSage!70] (0.2, 0.3) rectangle (4.2, 0.65);
    \draw[draw=SlateSteel, line width=0.5pt] (0.2, 0.3) rectangle (6.0, 0.65);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.2, 0.15) {Token Spend: $M \le M_{\max}$};
  \end{scope}

  % WAL
  \begin{scope}[shift={(0.3, 0.6)}]
    \draw[fill=CoolParchment, draw=SlateSteel] (0, 0) rectangle (6.4, 0.9);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily\bfseries] at (0.3, 0.45) {WAL: Saga Compensating Log};
  \end{scope}
\end{scope}

% Capability Escrow Gate
\begin{scope}[shift={(8.0, 0.8)}]
  \draw[fill=white, draw=Burgundy, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (5.0, 4.6);
  \node[anchor=north west, text=Burgundy, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 4.3) {Capability Escrow Gate};

  % Privilege Rings
  \begin{scope}[shift={(2.5, 2.5)}]
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt] (0, 0) circle (1.2cm);
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt] (0, 0) circle (0.8cm);
    \draw[fill=white, draw=OxfordNavy, line width=0.9pt] (0, 0) circle (0.4cm);
    \node[text=OxfordNavy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0) {Ring 0};
    \node[text=ForestSage, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0.58) {Ring 1};
    \node[text=Burgundy, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0, 0.98) {Ring 3};
  \end{scope}

  \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt] (0.3, 0.5) rectangle (4.7, 1.1);
  \node[text=WarmOchre, font=\fontsize{6.0}{7}\selectfont\bfseries] at (2.5, 0.8) {Attenuated Lease Token};
\end{scope}

% Isolated Sandbox Enclave
\begin{scope}[shift={(13.4, 0.8)}]
  \draw[fill=white, draw=Burgundy, double, double distance=1.5pt, line width=0.8pt, rounded corners=2pt]
    (0, 0) rectangle (6.2, 4.6);
  \node[anchor=north west, text=Burgundy, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 4.3) {Isolated MicroVM ($W \oplus X$)};

  % Terminal
  \begin{scope}[shift={(0.3, 2.0)}]
    \draw[fill=black!90, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (5.6, 2.0);
    \fill[Burgundy] (0.2, 1.75) circle (1.2pt);
    \fill[GoldPin] (0.4, 1.75) circle (1.2pt);
    \fill[ForestSage] (0.6, 1.75) circle (1.2pt);
    \node[anchor=north west, text=ForestSage, font=\fontsize{5.2}{6}\selectfont\ttfamily] at (0.1, 1.55) {\$ pytest tests/};
    \node[anchor=north west, text=white, font=\fontsize{5.2}{6}\selectfont\ttfamily] at (0.1, 1.05) {5 passed [100\%]};
  \end{scope}

  % Sanitizer
  \begin{scope}[shift={(0.3, 0.4)}]
    \draw[fill=SageL0, draw=ForestSage] (0, 0) rectangle (5.6, 1.3);
    \node[anchor=north west, text=ForestSage, font=\fontsize{6.0}{7}\selectfont\bfseries] at (0.2, 1.1) {Egress Sanitizer};
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.2}{6}\selectfont\ttfamily] at (0.2, 0.5) {Observation: $o_{t+1} = (\text{out, exit 0})$};
  \end{scope}
\end{scope}

% Independent Verification Oracle
\begin{scope}[shift={(20.0, 0.8)}]
  \draw[fill=white, draw=ForestSage, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (3.8, 4.6);
  \node[anchor=north west, text=ForestSage, font=\fontsize{7.0}{8}\selectfont\bfseries] at (0.2, 4.3) {Verifier Oracle};

  \draw[fill=SageL0, draw=ForestSage, rounded corners=1pt] (0.3, 2.2) rectangle (3.5, 3.6);
  \node[text=ForestSage, font=\fontsize{6.0}{7}\selectfont\bfseries, align=center] at (1.9, 2.9) {Deterministic\\Invariant Gate\\$v_{t+1} \in \{0, 1\}$};

  \draw[fill=CoolParchment, draw=SlateRim] (0.3, 0.5) rectangle (3.5, 1.6);
  \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont, align=center] at (1.9, 1.05) {Dijkstra Proof:\\Rejects Model\\Self-Reports};
\end{scope}

% Signals in bottom plane
\draw[->, draw=WarmOchre, line width=1.3pt] (13.0, 1.4) -- (13.4, 1.4);
\draw[->, draw=ForestSage, line width=1.3pt] (19.6, 2.0) -- (20.0, 2.0);

% Observation Return Loop crossing from Bottom Plane to Top Plane!
\draw[->, draw=ForestSage, line width=1.8pt, rounded corners=5pt]
  (16.5, 0.8) -- (16.5, 0.1) -- (-0.4, 0.1) -- (-0.4, 10.7) -- (0.6, 10.7);
\node[fill=ForestSage, text=white, font=\fontsize{5.5}{6.5}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
  at (-0.4, 5.4) {Sanitized Observation Feedback ($o_{t+1}$)};

\end{tikzpicture}
\end{document}
"""

def compile_tikz(tikz_code, output_base):
    tex_path = f"{output_base}.tex"
    with open(tex_path, "w") as f:
        f.write(tikz_code.strip() + "\n")
    print(f"[*] Wrote {tex_path}")
    
    cmd_pdf = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={TEX_DIR}", tex_path]
    res_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
    if res_pdf.returncode != 0:
        print(f"[!] Compilation failed for {output_base}:\n", res_pdf.stdout[-1500:])
        return False
        
    pdf_path = f"{output_base}.pdf"
    png_prefix = f"{output_base}_render"
    cmd_png = ["pdftoppm", "-png", "-r", "200", pdf_path, png_prefix]
    res_png = subprocess.run(cmd_png, capture_output=True, text=True)
    if res_png.returncode != 0:
        print(f"[!] pdftoppm failed for {output_base}:\n", res_png.stderr)
        return False
        
    print(f"[+] Successfully rendered {png_prefix}-1.png")
    return f"{png_prefix}-1.png"

def main():
    out_a = os.path.join(TEX_DIR, "option_a_classic_microarchitecture")
    res_a = compile_tikz(TIKZ_OPTION_A, out_a)
    
    out_b = os.path.join(TEX_DIR, "option_b_silicon_hardware_schematic")
    res_b = compile_tikz(TIKZ_OPTION_B, out_b)
    
    out_c = os.path.join(TEX_DIR, "option_c_dual_plane_control_loop")
    res_c = compile_tikz(TIKZ_OPTION_C, out_c)
    
    print("\n=== COMPILATION SUMMARY ===")
    print("Option A:", res_a)
    print("Option B:", res_b)
    print("Option C:", res_c)

if __name__ == "__main__":
    main()

