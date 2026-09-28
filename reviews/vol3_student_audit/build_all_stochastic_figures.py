#!/usr/bin/env python3
"""
Build all three publication-grade variants of the Stochastic Computer for Chapter 1:
1. Option A: Classic Microarchitecture (stochastic_computer_architecture)
2. Option B: Silicon Interposer & MCM Hardware (stochastic_computer_silicon_hardware)
3. Option C: Dual-Plane Control Loop Datapath (stochastic_computer_dual_plane_datapath)

Standards:
- Canonical academic palette
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy
- Authentic IEEE/ACM computer architecture glyphs
- Zero bullet points, zero paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys

TEX_DIR = "books/vol3/tex"
PNG_DIR = "books/vol3/01_introduction/images/png"
SVG_DIR = "books/vol3/01_introduction/images/svg"
PDF_DIR = "books/vol3/01_introduction/images/pdf"

os.makedirs(TEX_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

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
    \foreach \i/\label in {0/Prompt, 1/History} {
      \draw[fill=CoolParchment, draw=SlateRim, line width=0.5pt] (0.2, 1.15 - \i*0.48) rectangle (3.1, 1.55 - \i*0.48);
      \node[anchor=west, text=DarkSlate, font=\fontsize{5.8}{7}\selectfont\ttfamily] at (0.3, 1.35 - \i*0.48) {R\i: \label};
    }
    
    % Program counter
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1pt] (0.2, 0.15) rectangle (3.1, 0.55);
    \node[anchor=west, text=WarmOchre, font=\fontsize{5.8}{7}\selectfont\ttfamily\bfseries] at (0.3, 0.35) {PC: Step $t = 14$};
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

  % Internal datapath arrows in SPU
  \draw[->, line width=1.0pt, draw=OxfordNavy] (3.8, 4.55) -- (4.2, 4.55);
  \draw[->, line width=1.0pt, draw=OxfordNavy] (7.6, 4.55) -- (8.0, 4.55);
  \node[anchor=south, font=\fontsize{5.2}{6}\selectfont\ttfamily, text=SlateSteel] at (7.8, 4.55) {Logits $z_t$};

  % Unit 1D: Proposal Port & Zero Ambient Authority Escrow
  \begin{scope}[shift={(3.8, 0.7)}]
    \draw[fill=Burgundy, draw=Burgundy, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (4.2, 0.9);
    \node[text=white, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (2.1, 0.45) {PROPOSAL PORT ($a_{\mathrm{prop}}$)};
  \end{scope}
  \node[anchor=north, text=Burgundy, font=\fontsize{5.8}{7}\selectfont\bfseries] at (5.9, 0.6) {Zero Ambient Authority (Escrow Gate)};

  % Datapath from Sampler down to Proposal Port
  \draw[->, line width=1.1pt, draw=Burgundy] (9.65, 3.4) |- (8.0, 1.15);

\end{scope}

% 2. CONTEXT MEMORY HIERARCHY (M_ctx)
\begin{scope}[shift={(12.6, 7.6)}]
  \draw[fill=CoolParchment, draw=MutedTeal, line width=1.1pt, rounded corners=3pt] 
    (0, 0) rectangle (11.8, 6.6);

  \node[anchor=north west, fill=MutedTeal, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.3) {CONTEXT MEMORY HIERARCHY};
  \node[anchor=north east, text=MutedTeal, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 6.2) {Multi-Tiered Staging};

  % Tier L1: Active KV-Cache (3D HBM3e Stack)
  \begin{scope}[shift={(0.5, 4.2)}]
    \draw[fill=white, draw=MutedTeal, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 1.7);
    \node[anchor=north west, text=MutedTeal, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.2, 1.55) {L1: Active KV-Cache (HBM3e)};
    \node[anchor=north east, text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (10.6, 1.55) {3.2 TB/s $\cdot$ PagedAttention};

    % 4 Stacked Dies (Horizontal Physical Representation)
    \foreach \d in {0, 1, 2, 3} {
      \draw[fill=TealL0, draw=MutedTeal!80, line width=0.5pt, rounded corners=1pt]
        (0.3 + \d*2.6, 0.4) rectangle (0.3 + \d*2.6 + 2.4, 1.05);
      \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont\ttfamily] at (0.3 + \d*2.6 + 1.2, 0.72) {DRAM Die \d};
      % TSV Micro-bumps
      \fill[GoldPin] (0.5 + \d*2.6, 0.28) rectangle (0.7 + \d*2.6, 0.4);
      \fill[GoldPin] (2.3 + \d*2.6, 0.28) rectangle (2.5 + \d*2.6, 0.4);
    }
  \end{scope}

  % Tier L2: Episodic Session RAM (DDR5 / CXL DIMM)
  \begin{scope}[shift={(0.5, 2.2)}]
    \draw[fill=white, draw=SlateSteel, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 1.6);
    \node[anchor=north west, text=SlateSteel, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.2, 1.45) {L2: Episodic Session RAM (DDR5 / CXL)};
    \node[anchor=north east, text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (10.6, 1.45) {64 GB/s $\cdot$ Radix Prefix Tree};

    % DIMM PCB Strip
    \draw[fill=ForestSage!15, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 0.3) rectangle (10.5, 1.0);
    % DRAM Packages on DIMM
    \foreach \k in {0, 1, 2, 3, 4, 5, 6} {
      \draw[fill=DarkSlate, draw=black, line width=0.3pt] (0.5 + \k*1.42, 0.42) rectangle (0.5 + \k*1.42 + 1.05, 0.88);
      \node[text=white, font=\fontsize{4.0}{5}\selectfont] at (0.5 + \k*1.42 + 0.525, 0.65) {DRAM};
    }
    % Gold Edge Connector Pins
    \foreach \p in {0.5, 0.8, ..., 10.3} {
      \fill[GoldPin] (\p, 0.18) rectangle (\p+0.16, 0.3);
    }
  \end{scope}

  % Tier L3: Persistent Store (NVMe SSD & Vector Index)
  \begin{scope}[shift={(0.5, 0.4)}]
    \draw[fill=white, draw=SlateSteel, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 1.5);
    \node[anchor=north west, text=SlateSteel, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.2, 1.35) {L3: Persistent Store (NVMe \& Vector Index)};
    \node[anchor=north east, text=SlateSteel, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (10.6, 1.35) {7 GB/s $\cdot$ WAL Log \& HNSW};

    % NVMe Block
    \draw[fill=DarkSlate, draw=black, rounded corners=1pt] (0.4, 0.25) rectangle (4.6, 0.95);
    \fill[GoldPin] (0.4, 0.4) rectangle (0.6, 0.8);
    \node[text=white, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (2.6, 0.6) {NVMe Write-Ahead Log (WAL)};

    % Vector Index Platters Glyph
    \begin{scope}[shift={(7.2, 0.25)}]
      \draw[draw=MutedTeal, fill=TealL0, line width=0.6pt] (0, 0.45) ellipse (1.6cm and 0.18cm);
      \draw[draw=MutedTeal, fill=TealL0, line width=0.6pt] (0, 0.25) ellipse (1.6cm and 0.18cm);
      \draw[draw=MutedTeal, fill=TealL0, line width=0.6pt] (0, 0.05) ellipse (1.6cm and 0.18cm);
      \node[text=MutedTeal, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (0, 0.25) {Vector Index (HNSW)};
    \end{scope}
  \end{scope}

\end{scope}

% High-speed interconnect between SPU and L1 Memory
\draw[<->, draw=MutedTeal, line width=1.5pt] (11.8, 11.8) -- (12.6, 11.8)
  node[midway, above, font=\fontsize{5.5}{6.5}\selectfont\bfseries] {3.2 TB/s};


% -------------------------------------------------------------
% MIDDLE: SYSTEM TRAJECTORY BUS
% -------------------------------------------------------------
\begin{scope}[shift={(0, 6.2)}]
  \draw[fill=DarkSlate, draw=GoldPin, line width=1.0pt, rounded corners=2pt] (0, 0) rectangle (24.4, 0.7);
  \node[text=white, font=\fontsize{7.5}{9}\selectfont\bfseries] at (12.2, 0.35) {
    SYSTEM TRAJECTORY BUS $\cdot$ [Tokens $x_t$ $\mid$ KV-Pointers $\mid$ Action Proposals $a_t$ $\mid$ Observation Feed $o_t$]
  };
\end{scope}

% Bus connection arrows
\draw[->, draw=Burgundy, line width=1.6pt] (5.9, 7.6) -- (5.9, 6.9);
\draw[<->, draw=MutedTeal, line width=1.4pt] (18.5, 7.6) -- (18.5, 6.9);
\draw[->, draw=DarkSlate, line width=1.4pt] (5.9, 6.2) -- (5.9, 5.5);


% -------------------------------------------------------------
% BOTTOM TIER: DETERMINISTIC HOST SUPERVISOR & SANDBOX AIRLOCK
% -------------------------------------------------------------

% 3. HOST SUPERVISOR & AGENT RUNTIME KERNEL (RING 0)
\begin{scope}[shift={(0, 0.2)}]
  \draw[fill=CoolParchment, draw=WarmOchre, line width=1.1pt, rounded corners=3pt] 
    (0, 0) rectangle (11.8, 5.3);

  \node[anchor=north west, fill=WarmOchre, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 5.0) {HOST SUPERVISOR (RING 0 KERNEL)};
  \node[anchor=north east, text=WarmOchre, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 4.9) {Deterministic Governor};

  % Unit 3A: Agent Control Block (ACB)
  \begin{scope}[shift={(0.5, 0.5)}]
    \draw[fill=white, draw=WarmOchre, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.0, 3.9);
    \node[anchor=north west, text=WarmOchre, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.2, 3.7) {Agent Control Block (ACB)};

    % Hardware Deadline Clock (Square wave pulse)
    \draw[fill=CoolParchment, draw=SlateRim, rounded corners=1pt] (0.3, 2.5) rectangle (4.7, 3.2);
    \draw[draw=WarmOchre, line width=0.8pt] 
      (0.5, 2.65) -- (0.8, 2.65) -- (0.8, 3.05) -- (1.1, 3.05) -- (1.1, 2.65) -- (1.4, 2.65) -- (1.4, 3.05) -- (1.7, 3.05);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.2}{6}\selectfont\ttfamily] at (1.9, 2.85) {Deadline $T \le T_{\max}$};

    % Fuel Gauge (Token & Cost Budget Bar)
    \draw[fill=CoolParchment, draw=SlateRim, rounded corners=1pt] (0.3, 1.4) rectangle (4.7, 2.2);
    \draw[fill=ForestSage!70, draw=ForestSage, line width=0.4pt] (0.5, 1.65) rectangle (3.3, 1.95);
    \draw[draw=SlateSteel, line width=0.4pt] (0.5, 1.65) rectangle (4.5, 1.95);
    \node[anchor=north west, text=DarkSlate, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (0.5, 1.58) {Token Fuel: 65\% consumed};

    % Watchdog Interrupt (Active-low signal bubble)
    \draw[fill=BurgundyL0, draw=Burgundy, rounded corners=1pt] (0.3, 0.3) rectangle (4.7, 1.1);
    \draw[draw=Burgundy, line width=0.8pt] (0.6, 0.7) -- (1.3, 0.7);
    \draw[draw=Burgundy, fill=white, line width=0.8pt] (1.4, 0.7) circle (0.1);
    \node[anchor=west, text=Burgundy, font=\fontsize{5.2}{6}\selectfont\ttfamily\bfseries] at (1.6, 0.7) {SIG\_PREEMPT (Watchdog)};
  \end{scope}

  % Unit 3B: Capability Ring Engine
  \begin{scope}[shift={(6.0, 0.5)}]
    \draw[fill=white, draw=SlateSteel, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.3, 3.9);
    \node[anchor=north west, text=DarkSlate, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.4, 3.7) {Protection Rings};

    % Concentric Privilege Rings Glyph
    \begin{scope}[shift={(2.65, 2.1)}]
      \draw[draw=Burgundy, fill=BurgundyL0, line width=0.9pt] (0, 0) circle (1.35);
      \node[anchor=south, font=\fontsize{4.8}{5.5}\selectfont\bfseries, text=Burgundy] at (0, 0.95) {Ring 3};
      
      \draw[draw=ForestSage, fill=white, line width=0.8pt] (0, 0) circle (0.95);
      \node[anchor=south, font=\fontsize{4.8}{5.5}\selectfont\bfseries, text=ForestSage] at (0, 0.55) {Ring 1};

      \draw[draw=MutedTeal, fill=TealL0, line width=0.8pt] (0, 0) circle (0.55);
      \node[font=\fontsize{5.2}{6}\selectfont\bfseries, text=DarkSlate] at (0, 0) {Ring 0};
    \end{scope}

    % Capability Token Badge
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1pt] (0.4, 0.3) rectangle (4.9, 0.75);
    \node[font=\fontsize{5.8}{7}\selectfont\bfseries, text=WarmOchre] at (2.65, 0.52) {Signed Capability Lease};
  \end{scope}

\end{scope}

% Dispatch arrow from Host Supervisor to Sandbox Airlock
\draw[->, draw=WarmOchre, line width=1.3pt] (11.8, 2.85) -- (12.6, 2.85)
  node[midway, above, font=\fontsize{5.5}{6.5}\selectfont\bfseries] {Dispatch};


% 4. SANDBOX AIRLOCK CHAMBER (W xor X)
\begin{scope}[shift={(12.6, 0.2)}]
  % Grounded double-walled isolation boundary
  \draw[fill=white, draw=Burgundy, line width=1.1pt, rounded corners=3pt] (0, 0) rectangle (11.8, 5.3);
  \draw[draw=Burgundy!60, line width=0.5pt, rounded corners=2pt] (0.08, 0.08) rectangle (11.72, 5.22);

  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 5.0) {SANDBOX AIRLOCK CHAMBER ($W \oplus X$)};
  \node[anchor=north east, text=Burgundy, font=\fontsize{6.5}{8}\selectfont\bfseries]
    at (11.5, 4.9) {Isolation Perimeter};

  % Valve 1: Ingress Policy Valve
  \begin{scope}[shift={(0.5, 0.5)}]
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.7pt, rounded corners=1.5pt] (0, 0) rectangle (1.8, 3.9);
    \node[font=\fontsize{6.5}{7.5}\selectfont\bfseries, text=WarmOchre, align=center] at (0.9, 2.6) {INGRESS\\VALVE};
    \node[font=\fontsize{4.8}{5.5}\selectfont, text=DarkSlate, align=center] at (0.9, 1.2) {Policy\\Check\\Schema};
  \end{scope}

  % Center: Isolated MicroVM & Effectors
  \begin{scope}[shift={(2.8, 0.5)}]
    \draw[fill=DarkSlate, draw=black, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (6.1, 3.9);
    \node[anchor=north, font=\fontsize{6.2}{7.5}\selectfont\bfseries, text=white] at (3.05, 3.75) {ISOLATED MICROVM ENCLAVE};

    % Sub-unit: Execution Terminal
    \begin{scope}[shift={(0.3, 0.4)}]
      \draw[fill=black!85, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (2.6, 2.9);
      % Traffic lights
      \fill[Burgundy] (0.2, 2.7) circle (0.05);
      \fill[GoldPin] (0.38, 2.7) circle (0.05);
      \fill[ForestSage] (0.56, 2.7) circle (0.05);
      % Terminal output lines
      \node[anchor=north west, font=\fontsize{4.2}{5}\selectfont\ttfamily, text=ForestSage] at (0.15, 2.45) {\$ pytest};
      \node[anchor=north west, font=\fontsize{4.0}{5}\selectfont\ttfamily, text=white!85] at (0.15, 1.95) {5 passed};
      \node[anchor=north west, font=\fontsize{4.2}{5}\selectfont\ttfamily, text=GoldPin] at (0.15, 1.15) {\$ exit 0};
    \end{scope}

    % Sub-unit: Network MCP RPC Window
    \begin{scope}[shift={(3.2, 0.4)}]
      \draw[fill=white, draw=SlateSteel, rounded corners=1pt] (0, 0) rectangle (2.6, 2.9);
      \draw[fill=CoolParchment, draw=SlateRim] (0, 2.45) rectangle (2.6, 2.9);
      \node[font=\fontsize{4.0}{5}\selectfont\ttfamily, text=DarkSlate] at (1.3, 2.68) {https://api.db};
      \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=DarkSlate, align=center] at (1.3, 1.5) {Web DOM\\MCP RPC};
    \end{scope}
  \end{scope}

  % Valve 2: Egress Sanitizer Valve
  \begin{scope}[shift={(9.4, 0.5)}]
    \draw[fill=SageL0, draw=ForestSage, line width=0.7pt, rounded corners=1.5pt] (0, 0) rectangle (1.8, 3.9);
    \node[font=\fontsize{6.5}{7.5}\selectfont\bfseries, text=ForestSage, align=center] at (0.9, 2.6) {EGRESS\\VALVE};
    \node[font=\fontsize{4.8}{5.5}\selectfont, text=DarkSlate, align=center] at (0.9, 1.2) {Output\\Sanitizer\\$o_{t+1}$};
  \end{scope}

  % Internal flow through valves
  \draw[->, draw=WarmOchre, line width=0.9pt] (2.3, 2.45) -- (2.8, 2.45);
  \draw[->, draw=ForestSage, line width=0.9pt] (8.9, 2.45) -- (9.4, 2.45);

\end{scope}

% Feedback loop from Egress Valve back to Trajectory Bus
\draw[->, draw=ForestSage, line width=1.5pt] (23.8, 4.35) -- (23.8, 6.2)
  node[midway, right=1pt, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] {Sanitized $o_{t+1}$};

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# OPTION B: SILICON HARDWARE & DIE-LEVEL SCHEMATIC
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
\definecolor{SiliconCarrier}{HTML}{1E222D}
\definecolor{InterposerTrace}{HTML}{0284C7}

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
  (0, 7.0) rectangle (24.4, 14.8);

\node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
  at (0.4, 14.4) {1. SPU NEURAL ACCELERATOR PACKAGE (CO-PACKAGED MULTI-CHIP MODULE)};
\node[anchor=north east, text=SlateSteel, font=\fontsize{6.5}{8}\selectfont\bfseries]
  at (24.0, 14.3) {Silicon Interposer Architecture};

% Ceramic Substrate Body (Dark Slate)
\draw[fill=SiliconCarrier, draw=black, line width=1.0pt, rounded corners=3pt]
  (0.6, 7.8) rectangle (15.6, 13.8);

% Peripheral Gold BGA Micro-Pins (with pin cutout at bottom center for Proposal Port)
\foreach \x in {1.0, 1.6, ..., 15.2} {
  \fill[GoldPin] (\x, 13.8) rectangle (\x+0.35, 13.95);
}
\foreach \x in {1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 11.6, 12.2, 12.8, 13.4, 14.0, 14.6, 15.2} {
  \fill[GoldPin] (\x, 7.65) rectangle (\x+0.35, 7.8);
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

% Proposal Socket cleanly seated in pin cutout at bottom of SPU package
\draw[fill=Burgundy, draw=Burgundy, rounded corners=1.5pt] (5.4, 7.15) rectangle (10.6, 7.75);
\node[text=white, font=\fontsize{6.2}{7.5}\selectfont\bfseries] at (8.0, 7.45) {PROPOSAL PORT (ZERO AMBIENT AUTHORITY)};


% -------------------------------------------------------------
% MIDDLE: MOTHERBOARD TRAJECTORY BUS
% -------------------------------------------------------------
\draw[fill=OxfordNavy, draw=InterposerTrace, line width=1.1pt, rounded corners=2pt]
  (0, 5.8) rectangle (24.4, 6.5);
\node[text=white, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (12.2, 6.15) {
  SYSTEM TRAJECTORY BUS $\cdot$ [Tokens $x_t$ $\mid$ KV-Pointers $\mid$ Action Proposals $a_t$ $\mid$ Hardware Traps]
};

% Motherboard PCB via traces
\draw[->, draw=Burgundy, line width=1.6pt] (8.0, 7.15) -- (8.0, 6.5);
\draw[<->, draw=MutedTeal, line width=1.4pt] (18.5, 7.8) -- (18.5, 6.5);
\draw[->, draw=OxfordNavy, line width=1.4pt] (6.0, 5.8) -- (6.0, 5.0);


% -------------------------------------------------------------
% BOTTOM HALF: HOST ROOT-OF-TRUST & ISOLATED AIRLOCK
% -------------------------------------------------------------

% Host Supervisor Kernel
\begin{scope}[shift={(0, 0.2)}]
  \draw[fill=CoolParchment, draw=WarmOchre, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (11.0, 4.8);
  \node[anchor=north west, fill=WarmOchre, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 4.5) {3. HOST SUPERVISOR \& ROOT-OF-TRUST (RING 0 KERNEL)};

  % ACB
  \begin{scope}[shift={(0.5, 0.4)}]
    \draw[fill=white, draw=WarmOchre, rounded corners=2pt] (0, 0) rectangle (4.8, 3.6);
    \node[anchor=north west, text=WarmOchre, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.2, 3.4) {Agent Control Block (ACB)};

    % Clock waveform
    \draw[fill=CoolParchment, draw=SlateRim] (0.3, 2.3) rectangle (4.5, 2.95);
    \draw[draw=WarmOchre, line width=0.8pt] 
      (0.5, 2.45) -- (0.8, 2.45) -- (0.8, 2.8) -- (1.1, 2.8) -- (1.1, 2.45) -- (1.4, 2.45) -- (1.4, 2.8) -- (1.7, 2.8);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (1.8, 2.62) {Deadline $T \le T_{\max}$};

    % Fuel progress bar
    \draw[fill=CoolParchment, draw=SlateRim] (0.3, 1.3) rectangle (4.5, 1.95);
    \fill[ForestSage!70] (0.5, 1.45) rectangle (3.2, 1.75);
    \draw[draw=SlateSteel, line width=0.3pt] (0.5, 1.45) rectangle (4.3, 1.75);
    \node[anchor=north west, text=DarkSlate, font=\fontsize{4.8}{5.5}\selectfont\ttfamily] at (0.5, 1.4) {Token Fuel: 65\% consumed};

    % Watchdog
    \draw[fill=BurgundyL0, draw=Burgundy] (0.3, 0.3) rectangle (4.5, 0.95);
    \draw[draw=Burgundy, line width=0.8pt] (0.6, 0.62) -- (1.3, 0.62);
    \draw[draw=Burgundy, fill=white] (1.4, 0.62) circle (0.08);
    \node[anchor=west, text=Burgundy, font=\fontsize{5.0}{6}\selectfont\ttfamily\bfseries] at (1.6, 0.62) {SIG\_PREEMPT (Loop Watchdog)};
  \end{scope}

  % Privilege Rings
  \begin{scope}[shift={(5.6, 0.4)}]
    \draw[fill=white, draw=SlateSteel, rounded corners=2pt] (0, 0) rectangle (4.8, 3.6);
    \node[anchor=north west, text=DarkSlate, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.3, 3.4) {Privilege Separation};

    \begin{scope}[shift={(2.4, 1.9)}]
      \draw[draw=Burgundy, fill=BurgundyL0, line width=0.8pt] (0, 0) circle (1.2);
      \node[anchor=south, font=\fontsize{4.8}{5.5}\selectfont\bfseries, text=Burgundy] at (0, 0.85) {Ring 3};
      
      \draw[draw=ForestSage, fill=white, line width=0.7pt] (0, 0) circle (0.85);
      \node[anchor=south, font=\fontsize{4.8}{5.5}\selectfont\bfseries, text=ForestSage] at (0, 0.5) {Ring 1};

      \draw[draw=MutedTeal, fill=TealL0, line width=0.7pt] (0, 0) circle (0.5);
      \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=DarkSlate] at (0, 0) {Ring 0};
    \end{scope}

    \draw[fill=OchreL0, draw=WarmOchre, rounded corners=1pt] (0.4, 0.3) rectangle (4.4, 0.7);
    \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=WarmOchre] at (2.4, 0.5) {Signed Capability Token};
  \end{scope}
\end{scope}

% Dispatch Trace
\draw[->, draw=WarmOchre, line width=1.3pt] (11.0, 2.6) -- (11.8, 2.6)
  node[midway, above, font=\fontsize{5.5}{6.5}\selectfont\bfseries] {Dispatch};

% Isolated Airlock Peripheral
\begin{scope}[shift={(11.8, 0.2)}]
  \draw[fill=white, draw=Burgundy, line width=1.1pt, rounded corners=3pt] (0, 0) rectangle (12.6, 4.8);
  \draw[draw=Burgundy!60, line width=0.5pt, rounded corners=2pt] (0.08, 0.08) rectangle (12.52, 4.72);

  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 4.5) {4. ISOLATED PERIPHERAL AIRLOCK ($W \oplus X$)};

  % Ingress
  \begin{scope}[shift={(0.5, 0.4)}]
    \draw[fill=OchreL0, draw=WarmOchre, rounded corners=1.5pt] (0, 0) rectangle (1.7, 3.6);
    \node[font=\fontsize{6.5}{7.5}\selectfont\bfseries, text=WarmOchre, align=center] at (0.85, 2.6) {INGRESS\\VALVE};
    \node[font=\fontsize{4.8}{5.5}\selectfont, text=DarkSlate, align=center] at (0.85, 1.2) {Policy\\Check};
  \end{scope}

  % MicroVM
  \begin{scope}[shift={(2.6, 0.4)}]
    \draw[fill=DarkSlate, draw=black, rounded corners=2pt] (0, 0) rectangle (6.5, 3.6);
    \node[anchor=north, font=\fontsize{6.2}{7.5}\selectfont\bfseries, text=white] at (3.25, 3.45) {ISOLATED MICROVM ENCLAVE};

    \begin{scope}[shift={(0.3, 0.35)}]
      \draw[fill=black!90, draw=SlateSteel] (0, 0) rectangle (2.8, 2.7);
      \fill[Burgundy] (0.2, 2.5) circle (0.05);
      \fill[GoldPin] (0.38, 2.5) circle (0.05);
      \fill[ForestSage] (0.56, 2.5) circle (0.05);
      \node[anchor=north west, font=\fontsize{4.2}{5}\selectfont\ttfamily, text=ForestSage] at (0.15, 2.25) {\$ pytest};
      \node[anchor=north west, font=\fontsize{4.0}{5}\selectfont\ttfamily, text=white!85] at (0.15, 1.75) {5 passed};
      \node[anchor=north west, font=\fontsize{4.2}{5}\selectfont\ttfamily, text=GoldPin] at (0.15, 1.05) {\$ exit 0};
    \end{scope}

    \begin{scope}[shift={(3.4, 0.35)}]
      \draw[fill=white, draw=SlateSteel] (0, 0) rectangle (2.8, 2.7);
      \draw[fill=CoolParchment] (0, 2.25) rectangle (2.8, 2.7);
      \node[font=\fontsize{4.0}{5}\selectfont\ttfamily, text=DarkSlate] at (1.4, 2.48) {https://api.db};
      \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=DarkSlate, align=center] at (1.4, 1.35) {Web DOM\\MCP RPC};
    \end{scope}
  \end{scope}

  % Egress
  \begin{scope}[shift={(9.5, 0.4)}]
    \draw[fill=SageL0, draw=ForestSage, rounded corners=1.5pt] (0, 0) rectangle (1.8, 3.6);
    \node[font=\fontsize{6.5}{7.5}\selectfont\bfseries, text=ForestSage, align=center] at (0.9, 2.6) {EGRESS\\VALVE};
    \node[font=\fontsize{4.8}{5.5}\selectfont, text=DarkSlate, align=center] at (0.9, 1.2) {Sanitizer\\$o_{t+1}$};
  \end{scope}
\end{scope}

% Feedback loop to Trajectory Bus
\draw[->, draw=ForestSage, line width=1.5pt] (24.0, 3.9) -- (24.0, 5.8)
  node[midway, right=1pt, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] {Sanitized $o_{t+1}$};

\end{tikzpicture}
\end{document}
"""

# =============================================================================
# OPTION C: DUAL-PLANE CONTROL LOOP (RING 3 VS RING 0 CLOSED DATAPATH)
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
\definecolor{GoldPin}{HTML}{D97706}
\definecolor{DarkSlate}{HTML}{0F172A}

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
  (0, 8.2) rectangle (24.4, 14.2);

\node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.5}{10}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
  at (0.4, 13.8) {UNPRIVILEGED STOCHASTIC INFERENCE PLANE (RING 3)};
\node[anchor=north east, text=SlateSteel, font=\fontsize{6.5}{8}\selectfont\bfseries]
  at (24.0, 13.7) {Stateless Tensor Engine};

% Context Buffer
\begin{scope}[shift={(0.8, 8.8)}]
  \draw[fill=white, draw=SlateSteel, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (6.0, 4.4);
  \node[anchor=north west, text=OxfordNavy, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.3, 4.1) {Context Buffer $[c_t]$};

  \draw[fill=CoolParchment, draw=SlateRim] (0.3, 2.7) rectangle (5.7, 3.6);
  \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.5, 3.15) {Task Goal $g$ $\cdot$ System Prompt};

  \draw[fill=CoolParchment, draw=SlateRim] (0.3, 1.5) rectangle (5.7, 2.4);
  \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.5, 1.95) {Trajectory History $\tau_t = (s, a, o)_{1..t-1}$};

  \draw[fill=SageL0, draw=ForestSage] (0.3, 0.3) rectangle (5.7, 1.2);
  \node[anchor=west, text=ForestSage, font=\fontsize{5.5}{6.5}\selectfont\ttfamily\bfseries] at (0.5, 0.75) {Latest Feedback: Observation $o_t$};
\end{scope}

% PagedAttention KV-Cache
\begin{scope}[shift={(7.3, 8.8)}]
  \draw[fill=white, draw=MutedTeal, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (7.0, 4.4);
  \node[anchor=north west, text=MutedTeal, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.4, 4.1) {PagedAttention KV-Cache (L1/L2)};

  \foreach \b in {0, 1, 2} {
    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt, rounded corners=1pt] (0.3, 2.7 - \b*1.1) rectangle (6.7, 3.5 - \b*1.1);
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\ttfamily] at (0.5, 3.1 - \b*1.1) {Block \b: Keys $[K_t]$ $\cdot$ Values $[V_t]$};
  }
\end{scope}

% Stochastic Policy Core
\begin{scope}[shift={(14.8, 8.8)}]
  \draw[fill=white, draw=Burgundy, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (9.0, 4.4);
  \node[anchor=north west, text=Burgundy, font=\fontsize{7.5}{9}\selectfont\bfseries] at (0.4, 4.1) {Stochastic Policy Core ($\pi_\theta$)};

  % Trapezoidal Sampler
  \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt] 
    (0.4, 2.6) -- (3.8, 2.6) -- (3.3, 1.4) -- (0.9, 1.4) -- cycle;
  \node[text=Burgundy, font=\fontsize{6.2}{7.5}\selectfont\bfseries] at (2.1, 2.1) {Softmax Simplex};
  \node[text=DarkSlate, font=\fontsize{5.2}{6}\selectfont] at (2.1, 1.7) {$p = \mathrm{softmax}(z/T)$};

  % DFA Grammar
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

% Proposal crossing the boundary (clean white background mask on Escrow Check label)
\draw[->, draw=Burgundy, line width=2.0pt] (19.3, 8.8) -- (19.3, 5.8)
  node[midway, fill=white, inner sep=2pt, rounded corners=1pt, right=4pt, font=\fontsize{6.0}{7}\selectfont\bfseries, text=Burgundy] {Escrow Check};


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

  \begin{scope}[shift={(2.5, 2.5)}]
    \draw[draw=Burgundy, fill=BurgundyL0, line width=0.8pt] (0, 0) circle (1.4);
    \node[anchor=south, font=\fontsize{5.0}{6}\selectfont\bfseries, text=Burgundy] at (0, 1.05) {Ring 3};

    \draw[draw=ForestSage, fill=white, line width=0.7pt] (0, 0) circle (1.0);
    \node[anchor=south, font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage] at (0, 0.65) {Ring 1};

    \draw[draw=MutedTeal, fill=TealL0, line width=0.7pt] (0, 0) circle (0.6);
    \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=DarkSlate] at (0, 0) {Ring 0};
  \end{scope}

  \draw[fill=OchreL0, draw=WarmOchre, rounded corners=1pt] (0.3, 0.3) rectangle (4.7, 0.8);
  \node[font=\fontsize{6.0}{7}\selectfont\bfseries, text=WarmOchre] at (2.5, 0.55) {Attenuated Lease Token};
\end{scope}

\draw[->, draw=Burgundy, line width=1.3pt] (13.0, 3.1) -- (13.7, 3.1);

% Isolated MicroVM
\begin{scope}[shift={(13.7, 0.7)}]
  \draw[fill=white, draw=Burgundy, line width=0.9pt, rounded corners=2pt] (0, 0) rectangle (6.2, 4.8);
  \draw[draw=Burgundy!50, line width=0.4pt] (0.06, 0.06) rectangle (6.14, 4.74);
  \node[anchor=north west, text=Burgundy, font=\fontsize{7.2}{8.5}\selectfont\bfseries] at (0.3, 4.5) {Isolated MicroVM ($W \oplus X$)};

  % Terminal
  \begin{scope}[shift={(0.3, 1.8)}]
    \draw[fill=DarkSlate, draw=black, rounded corners=1.5pt] (0, 0) rectangle (5.6, 2.3);
    \fill[Burgundy] (0.25, 2.1) circle (0.05);
    \fill[GoldPin] (0.45, 2.1) circle (0.05);
    \fill[ForestSage] (0.65, 2.1) circle (0.05);
    \node[anchor=north west, font=\fontsize{4.6}{5.5}\selectfont\ttfamily, text=ForestSage] at (0.2, 1.85) {\$ pytest tests/};
    \node[anchor=north west, font=\fontsize{4.4}{5.5}\selectfont\ttfamily, text=white!90] at (0.2, 1.35) {5 passed [100\%]};
  \end{scope}

  % Egress Sanitizer
  \begin{scope}[shift={(0.3, 0.3)}]
    \draw[fill=SageL0, draw=ForestSage, rounded corners=1.5pt] (0, 0) rectangle (5.6, 1.2);
    \node[anchor=north west, text=ForestSage, font=\fontsize{6.0}{7}\selectfont\bfseries] at (0.2, 1.05) {Egress Sanitizer};
    \node[anchor=north west, text=DarkSlate, font=\fontsize{4.6}{5.5}\selectfont\ttfamily] at (0.2, 0.6) {Observation: $o_{t+1} = (\text{out}, \text{exit } 0)$};
  \end{scope}
\end{scope}

% Verifier Oracle
\begin{scope}[shift={(20.3, 0.8)}]
  \draw[fill=white, draw=ForestSage, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (3.8, 4.6);
  \node[anchor=north west, text=ForestSage, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (0.3, 4.3) {Verifier Oracle};

  \draw[fill=SageL0, draw=ForestSage, rounded corners=1.5pt] (0.3, 2.1) rectangle (3.5, 3.6);
  \node[font=\fontsize{5.8}{7}\selectfont\bfseries, text=ForestSage, align=center] at (1.9, 3.05) {Deterministic\\Invariant Gate};
  \node[font=\fontsize{5.0}{6}\selectfont\ttfamily, text=DarkSlate] at (1.9, 2.45) {$v_{t+1} \in \{0, 1\}$};

  \draw[fill=CoolParchment, draw=SlateRim] (0.3, 0.4) rectangle (3.5, 1.7);
  \node[font=\fontsize{4.8}{5.8}\selectfont, text=DarkSlate, align=center] at (1.9, 1.05) {Dijkstra Proof:\\Rejects Model\\Self-Reports};
\end{scope}

\draw[->, draw=ForestSage, line width=1.3pt] (19.9, 3.1) -- (20.3, 3.1);

% Closed Feedback Loop back to Context Buffer in Ring 3
\draw[->, draw=ForestSage, line width=1.6pt]
  (16.8, 0.7) -- (16.8, 0.1) -- (-0.4, 0.1) -- (-0.4, 11.0) -- (0.8, 11.0);

\node[anchor=south west, fill=ForestSage, text=white, font=\fontsize{5.5}{6.5}\selectfont\bfseries, rounded corners=1pt, inner sep=2pt]
  at (-2.2, 9.8) {Sanitized Observation Feedback ($o_{t+1}$)};

\end{tikzpicture}
\end{document}
"""

CONFIGS = [
    {
        "name": "Option A: Classic Microarchitecture",
        "tex_filename": "stochastic_computer_architecture.tex",
        "extra_tex_filename": "stochastic_computer_classic_microarchitecture.tex",
        "png_filename": "stochastic_computer_architecture.png",
        "svg_filename": "stochastic_computer_architecture.svg",
        "pdf_filename": "stochastic_computer_architecture.pdf",
        "tikz_code": TIKZ_OPTION_A
    },
    {
        "name": "Option B: Silicon Interposer & MCM Hardware",
        "tex_filename": "stochastic_computer_silicon_hardware.tex",
        "extra_tex_filename": None,
        "png_filename": "stochastic_computer_silicon_hardware.png",
        "svg_filename": "stochastic_computer_silicon_hardware.svg",
        "pdf_filename": "stochastic_computer_silicon_hardware.pdf",
        "tikz_code": TIKZ_OPTION_B
    },
    {
        "name": "Option C: Dual-Plane Control Loop Datapath",
        "tex_filename": "stochastic_computer_dual_plane_datapath.tex",
        "extra_tex_filename": None,
        "png_filename": "stochastic_computer_dual_plane_datapath.png",
        "svg_filename": "stochastic_computer_dual_plane_datapath.svg",
        "pdf_filename": "stochastic_computer_dual_plane_datapath.pdf",
        "tikz_code": TIKZ_OPTION_C
    }
]

def compile_all():
    inkscape_bin = "/opt/homebrew/bin/inkscape" if os.path.exists("/opt/homebrew/bin/inkscape") else "inkscape"
    
    for cfg in CONFIGS:
        print(f"\n=======================================================")
        print(f"Compiling: {cfg['name']}")
        print(f"=======================================================")
        
        # Write primary TeX
        tex_path = os.path.join(TEX_DIR, cfg["tex_filename"])
        with open(tex_path, "w") as f:
            f.write(cfg["tikz_code"].strip() + "\n")
        print(f"Wrote TeX to {tex_path}")
        
        if cfg["extra_tex_filename"]:
            extra_path = os.path.join(TEX_DIR, cfg["extra_tex_filename"])
            with open(extra_path, "w") as f:
                f.write(cfg["tikz_code"].strip() + "\n")
            print(f"Wrote alias TeX to {extra_path}")
            
        # Compile via pdflatex
        base_name = os.path.splitext(cfg["tex_filename"])[0]
        cmd_pdf = ["pdflatex", "-interaction=nonstopmode", f"-output-directory=/tmp", tex_path]
        res_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
        if res_pdf.returncode != 0:
            print(f"pdflatex error for {base_name}:")
            print(res_pdf.stdout[-1500:])
            sys.exit(1)
        
        tmp_pdf = f"/tmp/{base_name}.pdf"
        target_pdf = os.path.join(PDF_DIR, cfg["pdf_filename"])
        subprocess.run(["cp", tmp_pdf, target_pdf], check=True)
        print(f"Saved PDF to {target_pdf}")
        
        # Render high-res PNG via pdftoppm (250 DPI)
        png_prefix = f"/tmp/{base_name}_ppm"
        cmd_png = ["pdftoppm", "-png", "-r", "250", tmp_pdf, png_prefix]
        subprocess.run(cmd_png, check=True)
        rendered_png = f"{png_prefix}-1.png"
        
        target_png = os.path.join(PNG_DIR, cfg["png_filename"])
        subprocess.run(["cp", rendered_png, target_png], check=True)
        print(f"Saved PNG to {target_png}")
        
        # Render clean vector SVG via Inkscape
        target_svg = os.path.join(SVG_DIR, cfg["svg_filename"])
        cmd_svg = [inkscape_bin, "--export-type=svg", f"--export-filename={target_svg}", tmp_pdf]
        res_svg = subprocess.run(cmd_svg, capture_output=True, text=True)
        if res_svg.returncode != 0:
            print(f"Inkscape error for {base_name}: {res_svg.stderr}")
            sys.exit(1)
        print(f"Saved SVG to {target_svg}")

    print("\nAll 3 Stochastic Computer figures successfully built, compiled, and deployed!")

if __name__ == "__main__":
    compile_all()
