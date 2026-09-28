#!/usr/bin/env python3
"""
Prototype build script for the Redesigned, Highly-Illustrative
Stochastic Computer Architecture Diagram.

Focus: Hardware iconography, processor die, memory interposer,
sandbox airlock, privilege rings, minimal text, maximum visual intuition.
"""

import os
import subprocess
import sys

TIKZ_CONTENT = r"""\documentclass[tikz,border=12pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{calc,positioning,fit,arrows.meta,shapes.geometric,shapes.misc,decorations.pathmorphing}

% Canonical MLSys Palette
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{slatesteel}{HTML}{475569}
\definecolor{coolparchment}{HTML}{F8FAFC}
\definecolor{GridGray}{HTML}{E2E8F0}

\definecolor{BlueLine}{HTML}{0284C7}
\definecolor{BlueL0}{HTML}{F0F9FF}
\definecolor{BlueL}{HTML}{BAE6FD}
\definecolor{BlueDD}{HTML}{0C4A6E}
\definecolor{deepcyan}{HTML}{0369A1}

\definecolor{GreenLine}{HTML}{16A34A}
\definecolor{GreenL0}{HTML}{F0FDF4}
\definecolor{forestsage}{HTML}{15803D}

\definecolor{OrangeLine}{HTML}{EA580C}
\definecolor{OrangeL0}{HTML}{FFFBEB}
\definecolor{warmochre}{HTML}{B45309}

\definecolor{RedLine}{HTML}{DC2626}
\definecolor{RedL0}{HTML}{FEF2F2}
\definecolor{deepburgundy}{HTML}{991B1B}

\definecolor{PurpleLine}{HTML}{7C3AED}
\definecolor{PurpleL0}{HTML}{FAF5FF}
\definecolor{PurpleL}{HTML}{E9D5FF}
\definecolor{deeproyal}{HTML}{5B21B6}

\definecolor{GoldPin}{HTML}{F59E0B}
\definecolor{SiliconGray}{HTML}{1E293B}
\definecolor{TraceCopper}{HTML}{D97706}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth
]

% =============================================================
% CANVAS & MASTER TITLE BANNER
% =============================================================
\node[anchor=north west, inner sep=0pt] at (0, 16.6) {
  \begin{minipage}{26.0cm}
    \begin{tabular}{@{}p{18.5cm} r@{}}
      {\fontsize{13}{16}\selectfont \textbf{\textcolor{DarkSlate}{THE STOCHASTIC COMPUTER: HARDWARE MICROARCHITECTURE}}} &
      \fontsize{9}{11.5}\selectfont \textcolor{slatesteel}{\textbf{Chapter 1: Systems Foundations · Section 1.2}} \\[2pt]
      {\fontsize{8.5}{11}\selectfont \textcolor{slatesteel}{Physical Microarchitecture: Unprivileged Stochastic Core, Tiered Memory, Host Supervisor, and Sandbox Airlock}} &
      \fontsize{8}{10}\selectfont \textcolor{deepcyan}{\textbf{The von Neumann Mapping}} \\
    \end{tabular}
  \end{minipage}
};

% =============================================================
% 1. THE UNPRIVILEGED STOCHASTIC CORE (SILICON DIE & HBM PACKAGE)
% =============================================================

% Package Substrate (Ceramic BGA Carrier)
\draw[fill=coolparchment, draw=BlueLine!80, line width=1.4pt, rounded corners=8pt]
  (0, 7.2) rectangle (12.2, 15.6);

% Substrate Header Badge
\node[anchor=north west, fill=BlueLine, text=white, font=\fontsize{7.5}{9}\selectfont\bfseries, rounded corners=3pt, inner sep=4pt]
  at (0.4, 15.3) {1. UNPRIVILEGED STOCHASTIC CORE (RING 3)};

% SPU Chip Package (Dark Silicon Carrier with Gold Pins)
\draw[fill=SiliconGray, draw=DarkSlate, line width=1.2pt, rounded corners=5pt]
  (0.5, 7.6) rectangle (8.0, 14.6);

% Gold Pins around the SPU Package (Microprocessor leads)
\foreach \y in {8.0, 8.5, ..., 14.2} {
  \fill[GoldPin] (0.32, \y) rectangle (0.5, \y+0.25);
  \fill[GoldPin] (8.0, \y) rectangle (8.18, \y+0.25);
}
\foreach \x in {1.0, 1.6, ..., 7.5} {
  \fill[GoldPin] (\x, 7.42) rectangle (\x+0.35, 7.6);
  \fill[GoldPin] (\x, 14.6) rectangle (\x+0.35, 14.78);
}

% Central Silicon Die (Accelerator Compute Logic)
\draw[fill=DarkSlate, draw=BlueLine!60, line width=0.9pt, rounded corners=3pt]
  (0.8, 8.0) rectangle (7.7, 14.2);
\node[anchor=north, text=BlueL, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (4.25, 14.1) {NEURAL ACCELERATOR DIE ($W_\theta$)};

% Inside Die: Unit A - Systolic Matrix Core Grid (Tensor ALU)
\begin{scope}[shift={(1.1, 10.3)}]
  \draw[fill=SiliconGray!80, draw=BlueLine!40, rounded corners=2pt] (0, 0) rectangle (3.1, 3.2);
  \node[anchor=north, text=white, font=\fontsize{6.0}{7.2}\selectfont\bfseries] at (1.55, 3.1) {Systolic Tensor Grid};
  
  % 3x3 Matrix Units
  \foreach \r in {0, 1, 2} {
    \foreach \c in {0, 1, 2} {
      \draw[fill=BlueLine!70, draw=BlueL, line width=0.4pt, rounded corners=1pt]
        (0.2 + \c*0.9, 0.2 + \r*0.85) rectangle (0.2 + \c*0.9 + 0.75, 0.2 + \r*0.85 + 0.7);
      \node[text=white, font=\fontsize{4.8}{5.5}\selectfont\bfseries] at (0.2 + \c*0.9 + 0.375, 0.2 + \r*0.85 + 0.35) {GEMM};
    }
  }
\end{scope}

% Inside Die: Unit B - Stochastic Autoregressive Sampler (Softmax + Temperature)
\begin{scope}[shift={(4.5, 10.3)}]
  \draw[fill=SiliconGray!80, draw=OrangeLine!60, rounded corners=2pt] (0, 0) rectangle (2.9, 3.2);
  \node[anchor=north, text=OrangeLine, font=\fontsize{6.0}{7.2}\selectfont\bfseries] at (1.45, 3.1) {Stochastic Token Sampler};
  
  % Distribution curve
  \draw[draw=OrangeLine, line width=1.2pt]
    (0.3, 0.6) .. controls (0.8, 0.6) and (1.2, 2.2) .. (1.5, 2.2)
    .. controls (1.8, 2.2) and (2.2, 0.6) .. (2.7, 0.6);
  
  % Sampled token spike
  \draw[dashed, draw=RedLine, line width=0.8pt] (1.5, 0.6) -- (1.5, 2.2);
  \fill[RedLine] (1.5, 2.2) circle (2pt);
  \node[anchor=south, text=RedLine, font=\fontsize{5.0}{6.0}\selectfont\bfseries] at (1.5, 2.25) {Sample $x_{t+1} \sim P$};
  \node[anchor=north, text=white!80, font=\fontsize{5.0}{6.0}\selectfont] at (1.45, 0.5) {$P(x_{t+1}|c_t) = \text{Softmax}(z_t / T)$};
\end{scope}

% Inside Die: Unit C - Context Register File & Turn PC
\begin{scope}[shift={(1.1, 8.3)}]
  \draw[fill=BlueDD!70, draw=BlueLine!50, rounded corners=2pt] (0, 0) rectangle (3.1, 1.6);
  \node[anchor=north, text=BlueL, font=\fontsize{5.2}{6.2}\selectfont\bfseries] at (1.55, 1.52) {Context Registers ($R_t$)};
  \node[anchor=north, text=GoldPin, font=\fontsize{4.8}{5.8}\selectfont\bfseries] at (1.55, 1.1) {Program Counter: $t=14$};
  
  % Register cells
  \foreach \i in {0, 1, 2, 3} {
    \draw[fill=DarkSlate, draw=BlueLine!40, line width=0.4pt] (0.2 + \i*0.68, 0.15) rectangle (0.2 + \i*0.68 + 0.62, 0.65);
    \node[text=white, font=\fontsize{4.8}{5.5}\selectfont] at (0.2 + \i*0.68 + 0.31, 0.4) {$t_{\i}$};
  }
\end{scope}

% Inside Die: Unit D - Hardware Pushdown Automaton (PDA Logit Filter)
\begin{scope}[shift={(4.5, 8.3)}]
  \draw[fill=GreenLine!15, draw=GreenLine, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (2.9, 1.6);
  \node[anchor=north, text=forestsage, font=\fontsize{5.2}{6.2}\selectfont\bfseries] at (1.45, 1.52) {Hardware PDA Mask};
  \node[text=DarkSlate, font=\fontsize{5.0}{6.2}\selectfont, align=center] at (1.45, 0.72) {
    Grammar DFA Clamping\\
    \textcolor{deepburgundy}{\textbf{Illegal Logits $\to -\infty$}}\\
    JSON / Schema Enforced
  };
\end{scope}

% Interconnect inside Die: Tensor -> Sampler -> PDA -> Proposal
\draw[->, draw=BlueL, line width=1.1pt] (4.2, 11.9) -- (4.5, 11.9);
\draw[->, draw=OrangeLine, line width=1.1pt] (5.95, 10.3) -- (5.95, 9.9);

% SPU Output Pin: Unprivileged Proposal Port
\node[anchor=north, fill=RedLine, text=white, font=\fontsize{6.2}{7.5}\selectfont\bfseries, rounded corners=2pt, inner sep=3pt]
  (prop_port) at (5.95, 7.4) {PROPOSAL PORT (ZERO AMBIENT AUTHORITY)};

% Co-Packaged High-Bandwidth Memory (HBM3e 3D Stack on Silicon Interposer)
\begin{scope}[shift={(8.7, 9.2)}]
  \draw[fill=coolparchment, draw=BlueLine!80, line width=0.8pt, rounded corners=4pt] (-0.3, -1.4) rectangle (3.3, 5.0);
  \node[anchor=north, text=BlueLine, font=\fontsize{6.8}{8.0}\selectfont\bfseries] at (1.5, 4.8) {L1: HBM3e 3D STACK};
  \node[anchor=north, text=slatesteel, font=\fontsize{5.2}{6.2}\selectfont] at (1.5, 4.3) {PagedAttention KV Cache};
  
  % 4-layer DRAM Die Stack
  \foreach \y [count=\idx] in {0.8, 1.6, 2.4, 3.2} {
    \draw[fill=BlueL0, draw=BlueLine!60, line width=0.6pt, rounded corners=1.5pt] (0, \y) rectangle (3.0, \y+0.6);
    \node[text=DarkSlate, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (1.5, \y+0.3) {DRAM Layer \idx \ ($16\text{ GB}$)};
    % TSV vertical interconnects
    \fill[GoldPin] (0.3, \y-0.2) rectangle (0.42, \y);
    \fill[GoldPin] (2.6, \y-0.2) rectangle (2.72, \y);
  }
  
  % Base Logic Die & Silicon Interposer
  \draw[fill=BlueLine!20, draw=BlueLine, line width=0.8pt, rounded corners=1.5pt] (0, 0) rectangle (3.0, 0.6);
  \node[text=BlueLine, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (1.5, 0.3) {Base Logic Die (Radix Tree)};
  
  % Silicon Interposer micro-bumps
  \foreach \x in {0.2, 0.6, ..., 2.8} {
    \fill[GoldPin] (\x, -0.15) rectangle (\x+0.15, 0);
  }
  \node[text=slatesteel, font=\fontsize{5.0}{6.0}\selectfont\bfseries] at (1.5, -0.6) {Ultra-Fast 3.2 TB/s Bus};
\end{scope}

% Silicon Interposer Bridge connecting SPU to HBM
\draw[fill=SiliconGray, draw=GoldPin, line width=0.6pt] (7.7, 11.5) rectangle (8.7, 12.1);
\node[text=GoldPin, font=\fontsize{4.5}{5.5}\selectfont\bfseries, align=center] at (8.2, 11.8) {TSV\\Bus};


% =============================================================
% 2. THE MULTI-TIER MEMORY SUBSYSTEM (L2 & L3)
% =============================================================

% Subsystem Box
\draw[fill=coolparchment, draw=GreenLine!80, line width=1.4pt, rounded corners=8pt]
  (12.7, 7.2) rectangle (25.8, 15.6);
\node[anchor=north west, fill=GreenLine, text=white, font=\fontsize{7.5}{9}\selectfont\bfseries, rounded corners=3pt, inner sep=4pt]
  at (13.1, 15.3) {2. CONTEXT MEMORY HIERARCHY};

% L2 Episodic RAM (DDR5 DIMM Module)
\begin{scope}[shift={(13.2, 11.4)}]
  \draw[fill=GreenL0, draw=GreenLine!60, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (12.1, 2.7);
  \node[anchor=north west, text=forestsage, font=\fontsize{6.8}{8.0}\selectfont\bfseries] at (0.3, 2.5) {L2: Host Episodic RAM (DDR5 / CXL)};
  \node[anchor=north east, text=slatesteel, font=\fontsize{5.2}{6.2}\selectfont] at (11.8, 2.5) {Latency: $\approx 10\text{ ms}$};
  
  % PCB DIMM Stick Graphic
  \draw[fill=GreenLine!25, draw=forestsage, line width=0.8pt, rounded corners=2pt] (0.4, 0.7) rectangle (11.6, 1.9);
  % DRAM chips on stick
  \foreach \x in {0.8, 2.2, 3.6, 5.0, 7.0, 8.4, 9.8} {
    \draw[fill=DarkSlate, draw=black, rounded corners=1pt] (\x, 0.9) rectangle (\x+1.1, 1.7);
    \node[text=white, font=\fontsize{4.5}{5.5}\selectfont] at (\x+0.55, 1.3) {DRAM};
  }
  % DIMM gold fingers
  \foreach \x in {0.6, 0.9, ..., 11.2} {
    \fill[GoldPin] (\x, 0.55) rectangle (\x+0.18, 0.7);
  }
  \node[text=forestsage, font=\fontsize{5.5}{6.5}\selectfont\bfseries] at (6.0, 0.28) {Sliding Window Context $\cdot$ Rollout Branch Scratchpads};
\end{scope}

% L3 Persistent Store (NVMe SSD & Write-Ahead Log)
\begin{scope}[shift={(13.2, 7.5)}]
  \draw[fill=GreenL0, draw=GreenLine!60, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (12.1, 3.5);
  \node[anchor=north west, text=forestsage, font=\fontsize{6.8}{8.0}\selectfont\bfseries] at (0.3, 3.3) {L3: Persistent Store \& Causal Spine};
  \node[anchor=north east, text=slatesteel, font=\fontsize{5.2}{6.2}\selectfont] at (11.8, 3.3) {Throughput: $7\text{ GB/s}$ NVMe};
  
  % NVMe M.2 Drive Graphic
  \begin{scope}[shift={(0.5, 0.45)}]
    \draw[fill=DarkSlate, draw=black, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (5.2, 2.2);
    \fill[GoldPin] (0, 0.4) rectangle (0.3, 1.8); % M.2 edge pins
    % Subtitle inside NVMe
    \node[anchor=north, text=GoldPin, font=\fontsize{5.2}{6.2}\selectfont\bfseries] at (2.6, 2.05) {Write-Ahead Log (WAL)};
    % NAND Flash chips
    \draw[fill=SiliconGray, draw=slatesteel] (0.5, 0.25) rectangle (2.0, 1.45);
    \node[text=white, font=\fontsize{4.8}{5.8}\selectfont, align=center] at (1.25, 0.85) {NVMe\\Flash};
    \draw[fill=SiliconGray, draw=slatesteel] (2.2, 0.25) rectangle (3.7, 1.45);
    \node[text=white, font=\fontsize{4.8}{5.8}\selectfont, align=center] at (2.95, 0.85) {WAL\\Log};
    \draw[fill=forestsage!80, draw=white] (3.9, 0.35) rectangle (4.9, 1.35);
    \node[text=white, font=\fontsize{4.5}{5.5}\selectfont, rotate=90] at (4.4, 0.85) {PCIe 5};
  \end{scope}
  
  % Vector DB / Disk Platters Graphic
  \begin{scope}[shift={(6.4, 0.45)}]
    \draw[fill=white, draw=forestsage, line width=0.8pt, rounded corners=2pt] (0, 0) rectangle (5.2, 2.2);
    % 3 stacked disk platters
    \foreach \y in {0.4, 0.8, 1.2} {
      \draw[fill=GreenLine!20, draw=forestsage, line width=0.6pt] (1.0, \y) ellipse (0.75cm and 0.18cm);
    }
    \node[anchor=west, text=DarkSlate, font=\fontsize{5.2}{6.2}\selectfont, align=left] at (2.0, 1.1) {
      \textbf{Vector Index (HNSW)}\\
      Long-term memory store\\
      Causal artifacts $>64\text{ KB}$
    };
  \end{scope}
\end{scope}


% =============================================================
% 3. THE HIGH-SPEED SYSTEM INTERCONNECT BUS (COPPER/GOLD TRACES)
% =============================================================
\draw[fill=DarkSlate, draw=GoldPin, line width=1.2pt, rounded corners=3pt]
  (0, 6.0) rectangle (25.8, 6.8);

\node[text=white, font=\fontsize{7.0}{8.5}\selectfont\bfseries] at (12.9, 6.4) {
  SYSTEM TRAJECTORY BUS $\cdot$ [Tokens $x_t$ $\mid$ KV-Pointers $\mid$ Action Proposals $a_t$ $\mid$ Hardware Traps]
};

% Bus connections
\draw[->, draw=RedLine, line width=2.0pt] (5.95, 7.3) -- (5.95, 6.8);
\draw[<->, draw=BlueLine, line width=2.0pt] (10.2, 7.2) -- (10.2, 6.8);
\draw[<->, draw=GreenLine, line width=2.0pt] (19.2, 7.2) -- (19.2, 6.8);


% =============================================================
% 4. THE HOST CONTROL PLANE (SUPERVISOR / RING 0 KERNEL)
% =============================================================
\draw[fill=coolparchment, draw=OrangeLine!90, line width=1.4pt, rounded corners=8pt]
  (0, 0.4) rectangle (12.2, 5.6);
\node[anchor=north west, fill=OrangeLine, text=white, font=\fontsize{7.5}{9}\selectfont\bfseries, rounded corners=3pt, inner sep=4pt]
  at (0.4, 5.3) {3. HOST SUPERVISOR PLANE (RING 0 KERNEL)};

% Unit 4A: Agent Control Block (ACB) & Quota Timers
\begin{scope}[shift={(0.5, 0.7)}]
  \draw[fill=white, draw=OrangeLine!60, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (5.3, 4.0);
  \node[anchor=north west, text=warmochre, font=\fontsize{6.8}{8.0}\selectfont\bfseries] at (0.2, 3.8) {Agent Control Block (ACB)};
  
  % Monotonic Stopwatch graphic
  \draw[fill=OrangeLine!15, draw=OrangeLine, line width=0.6pt] (0.8, 2.7) circle (0.45cm);
  \draw[line width=0.8pt, draw=OrangeLine] (0.8, 2.7) -- (0.8, 3.0);
  \draw[line width=0.8pt, draw=OrangeLine] (0.8, 2.7) -- (1.0, 2.7);
  \node[anchor=west, text=DarkSlate, font=\fontsize{5.2}{6.5}\selectfont, align=left] at (1.4, 2.7) {
    \textbf{Monotonic Deadline}\\
    $T \le T_{\max}$ (Wall-clock)
  };

  % Token Fuel Gauge graphic
  \draw[fill=GridGray, draw=slatesteel, line width=0.5pt, rounded corners=1pt] (0.4, 1.6) rectangle (4.9, 1.95);
  \fill[GreenLine] (0.4, 1.6) rectangle (3.3, 1.95);
  \node[anchor=north west, text=DarkSlate, font=\fontsize{5.2}{6.2}\selectfont] at (0.4, 1.5) {Token Fuel: $65\%$ consumed ($M \le M_{\max}$)};

  % Turn PC Register
  \draw[fill=coolparchment, draw=slatesteel, line width=0.5pt] (0.4, 0.2) rectangle (4.9, 0.7);
  \node[text=DarkSlate, font=\fontsize{5.2}{6.2}\selectfont] at (2.65, 0.45) {Turn Counter: $H = 14 \le H_{\max}$};
\end{scope}

% Unit 4B: Hardware Protection Rings & Capability MMU
\begin{scope}[shift={(6.2, 0.7)}]
  \draw[fill=white, draw=RedLine!70, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (5.5, 4.0);
  \node[anchor=north west, text=deepburgundy, font=\fontsize{6.8}{8.0}\selectfont\bfseries] at (0.2, 3.8) {Hardware Protection Rings};
  
  % Concentric Privilege Rings
  \begin{scope}[shift={(1.6, 2.0)}]
    \draw[fill=OrangeLine!15, draw=OrangeLine, line width=0.8pt] (0,0) circle (1.2cm);
    \draw[fill=GreenLine!15, draw=GreenLine, line width=0.8pt] (0,0) circle (0.8cm);
    \draw[fill=RedLine!20, draw=RedLine, line width=0.9pt] (0,0) circle (0.42cm);
    \node[text=RedLine, font=\fontsize{4.2}{5.2}\selectfont\bfseries] at (0, 0) {RING 0};
    \node[text=GreenLine, font=\fontsize{4.2}{5.2}\selectfont\bfseries] at (0, 0.6) {RING 1};
    \node[text=OrangeLine, font=\fontsize{4.2}{5.2}\selectfont\bfseries] at (0, 1.0) {RING 3};
  \end{scope}
  
  \node[anchor=west, text=DarkSlate, font=\fontsize{5.2}{6.2}\selectfont, align=left] at (3.0, 2.0) {
    \textbf{Ring 0: Host OS}\\
    \textbf{Ring 1: Schema Gate}\\
    \textbf{Ring 3: SPU Model}\\
    \textcolor{deepburgundy}{Zero Ambient Authority}
  };
  
  \node[anchor=north, text=deepburgundy, font=\fontsize{5.2}{6.2}\selectfont\bfseries, align=center] at (2.75, 0.6) {
    Macaroon Capability Lease\\
    Sync \texttt{fsync} to WAL before Dispatch
  };
\end{scope}


% =============================================================
% 5. THE SANDBOX AIRLOCK & I/O SUBSYSTEM (PERIPHERAL EFFECTORS)
% =============================================================
\draw[fill=coolparchment, draw=PurpleLine!90, line width=1.4pt, rounded corners=8pt]
  (12.7, 0.4) rectangle (25.8, 5.6);
\node[anchor=north west, fill=PurpleLine, text=white, font=\fontsize{7.5}{9}\selectfont\bfseries, rounded corners=3pt, inner sep=4pt]
  at (13.1, 5.3) {4. SANDBOX AIRLOCK \& PERIPHERAL CHAMBER};

% Physical Airlock Container Graphic
\begin{scope}[shift={(13.2, 0.7)}]
  \draw[fill=white, draw=PurpleLine!60, line width=0.8pt, rounded corners=3pt] (0, 0) rectangle (12.1, 4.0);
  
  % Left Airlock Valve (Ingress Capability Gate)
  \draw[fill=OrangeLine!20, draw=OrangeLine, line width=0.8pt] (0.4, 0.4) rectangle (2.2, 3.5);
  \node[text=warmochre, font=\fontsize{6.0}{7.0}\selectfont\bfseries, align=center] at (1.3, 2.0) {INGRESS\\GATE\\VALVE\\[4pt]\fontsize{4.5}{5.5}\selectfont Policy\\Check};

  % Central MicroVM Quarantine Chamber
  \draw[fill=SiliconGray, draw=PurpleLine, line width=1.2pt, rounded corners=2pt] (2.6, 0.4) rectangle (9.3, 3.5);
  \node[anchor=north, text=PurpleL, font=\fontsize{6.5}{7.8}\selectfont\bfseries] at (5.95, 3.35) {ISOLATED MICROVM ENCLAVE ($W \oplus X$)};
  
  % Tool A: Sandboxed Bash Terminal
  \begin{scope}[shift={(2.9, 0.7)}]
    \draw[fill=DarkSlate, draw=black, rounded corners=1.5pt] (0, 0) rectangle (2.8, 2.2);
    \fill[black!85] (0, 1.8) rectangle (2.8, 2.2);
    \fill[RedLine] (0.2, 2.0) circle (1.2pt);
    \fill[GoldPin] (0.4, 2.0) circle (1.2pt);
    \fill[GreenLine] (0.6, 2.0) circle (1.2pt);
    \node[anchor=north west, text=GreenLine, font=\fontsize{4.8}{5.8}\selectfont\ttfamily] at (0.1, 1.7) {\$ pytest tests/};
    \node[anchor=north west, text=white, font=\fontsize{4.8}{5.8}\selectfont\ttfamily] at (0.1, 1.2) {5 passed};
    \node[anchor=north west, text=GoldPin, font=\fontsize{4.8}{5.8}\selectfont\ttfamily] at (0.1, 0.7) {\$ exit 0};
  \end{scope}

  % Tool B: Sandboxed Web / MCP Browser
  \begin{scope}[shift={(6.1, 0.7)}]
    \draw[fill=white, draw=BlueLine!60, rounded corners=1.5pt] (0, 0) rectangle (2.8, 2.2);
    \fill[BlueL!30] (0, 1.8) rectangle (2.8, 2.2);
    \fill[RedLine] (0.2, 2.0) circle (1.2pt);
    \fill[GoldPin] (0.4, 2.0) circle (1.2pt);
    \fill[GreenLine] (0.6, 2.0) circle (1.2pt);
    \draw[fill=white, draw=slatesteel] (0.8, 1.85) rectangle (2.6, 2.15);
    \node[text=deepcyan, font=\fontsize{4.2}{5.2}\selectfont\ttfamily] at (1.7, 2.0) {https://api.db};
    \node[text=slatesteel, font=\fontsize{5.0}{6.0}\selectfont, align=center] at (1.4, 0.9) {Web DOM /\\MCP Tool RPC};
  \end{scope}

  % Right Airlock Valve (Egress Sanitizer)
  \draw[fill=GreenLine!20, draw=GreenLine, line width=0.8pt] (9.7, 0.4) rectangle (11.7, 3.5);
  \node[text=forestsage, font=\fontsize{6.0}{7.0}\selectfont\bfseries, align=center] at (10.7, 2.0) {EGRESS\\FILTER\\VALVE\\[4pt]\fontsize{4.5}{5.5}\selectfont Sanitizer\\$o_{t+1}$};
\end{scope}

% Bus connections to Supervisor and Airlock
\draw[->, draw=OrangeLine, line width=2.0pt] (5.95, 6.0) -- (5.95, 5.6);
\draw[->, draw=OrangeLine, line width=1.5pt] (12.2, 2.8) -- (13.6, 2.8);
\node[fill=coolparchment, inner sep=1.5pt, font=\fontsize{5.0}{6.0}\selectfont\bfseries, text=warmochre] at (12.9, 2.8) {RPC Lease};

% Return trace from Egress Filter to Memory Hierarchy Bus
\draw[->, draw=GreenLine, line width=1.8pt] (24.9, 4.4) -- (24.9, 6.0)
  node[midway, right=2pt, font=\fontsize{5.2}{6.5}\selectfont\bfseries, text=forestsage] {Sanitized $o_{t+1}$};

\end{tikzpicture}
\end{document}
"""

def main():
    tex_dir = "/tmp/stochastic_prototype"
    os.makedirs(tex_dir, exist_ok=True)
    tex_path = os.path.join(tex_dir, "prototype_stochastic_computer.tex")
    
    with open(tex_path, "w") as f:
        f.write(TIKZ_CONTENT.strip() + "\n")
    print(f"Wrote TikZ source to {tex_path}")
    
    # Run pdflatex
    cmd_pdf = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={tex_dir}", tex_path]
    res_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
    print(f"pdflatex return code: {res_pdf.returncode}")
    if res_pdf.returncode != 0:
        print("pdflatex stdout:\n", res_pdf.stdout[-1500:])
        print("pdflatex stderr:\n", res_pdf.stderr)
        sys.exit(1)
        
    # Convert PDF to PNG via pdftoppm
    pdf_path = os.path.join(tex_dir, "prototype_stochastic_computer.pdf")
    png_prefix = os.path.join(tex_dir, "prototype_stochastic_computer_render")
    cmd_png = ["pdftoppm", "-png", "-r", "200", pdf_path, png_prefix]
    res_png = subprocess.run(cmd_png, capture_output=True, text=True)
    if res_png.returncode != 0:
        print("pdftoppm failed:\n", res_png.stderr)
        sys.exit(1)
        
    print(f"Successfully compiled and rendered prototype to {png_prefix}-1.png")

if __name__ == "__main__":
    main()
