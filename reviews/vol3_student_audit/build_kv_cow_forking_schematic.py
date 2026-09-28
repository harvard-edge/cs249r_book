#!/usr/bin/env python3
"""
Build publication-grade schematic for Chapter 5:
Copy-on-Write Deliberation Forking & Block Reclamation Lifecycle (kv_cow_forking_lifecycle)

Standards:
- Canonical academic palette (OxfordNavy, SlateSteel, MutedTeal, Burgundy, WarmOchre, ForestSage)
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy (8.0pt badges, 6.5pt components, 5.0pt metrics/signals)
- Authentic IEEE/ACM computer architecture glyphs:
  - 4-phase microarchitectural lifecycle datapath
  - Virtual-to-physical PagedAttention translation table & circular ring buffer
  - Physical HBM memory map comparison (contiguous static reservation vs virtualized CoW)
  - Quantitative hardware metrics panel
- Zero bullet points, zero paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys
import re

TEX_PATH = "books/vol3/tex/kv_cow_forking_lifecycle.tex"
PNG_DIR = "books/vol3/05_kv_cache/images/png"
SVG_DIR = "books/vol3/05_kv_cache/images/svg"
PDF_DIR = "books/vol3/05_kv_cache/images/pdf"

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

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth,
  node distance=0.5cm
]

% =============================================================================
% TOP TIER: 4-PHASE COW DELIBERATION LIFECYCLE (4 Columns)
% =============================================================================
\begin{scope}[shift={(0, 8.4)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (25.4, 8.4);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{7.5}{9.0}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 8.1) {1. FOUR-PHASE COPY-ON-WRITE DELIBERATION LIFECYCLE};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{5.8}{7}\selectfont\bfseries]
    at (25.1, 8.0) {Virtualized PagedAttention Memory};

  % Column 1: Phase 1 (Ingestion)
  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.8, 7.3);
    \node[anchor=north, fill=OxfordNavy, text=white, font=\fontsize{6.0}{7}\selectfont\bfseries, rounded corners=1pt, inner sep=2.5pt]
      at (2.9, 7.1) {PHASE 1: COMMON PREFIX};
    \node[font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=SlateSteel] at (2.9, 6.45) {Turn 0: Prefill ($S=8\text{k}$)};

    % Block Table
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1pt] (0.3, 4.3) rectangle (5.5, 6.1);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=MutedTeal] at (0.4, 5.95) {Logical Block Table:};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.4, 5.3) {Seq 0: L0$\to$B12, L1$\to$B15, L2$\to$B19};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=ForestSage] at (0.4, 4.7) {Ref Count: $\text{ref}=1$ per block};

    % Arrow to Physical
    \draw[->, draw=SlateSteel, line width=0.8pt] (2.9, 4.3) -- (2.9, 3.7);

    % Physical Blocks
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.6pt, rounded corners=1pt] (0.3, 1.8) rectangle (5.5, 3.7);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=DarkSlate] at (0.4, 3.55) {Physical HBM Pool:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 3.0) {Blk 12, 15, 19 ($16$ tok/blk)};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=MutedTeal] at (0.4, 2.3) {Allocated from Ring Buffer};

    % Status Badge
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1pt] (0.3, 0.3) rectangle (5.5, 1.4);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=MutedTeal, align=center] at (2.9, 0.85) {HBM Traffic: $2.56\text{ GiB}$\\Normal Initial Prefill};
  \end{scope}

  % Column 2: Phase 2 (Zero-Copy Fork)
  \begin{scope}[shift={(6.6, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.8, 7.3);
    \node[anchor=north, fill=OxfordNavy, text=white, font=\fontsize{6.0}{7}\selectfont\bfseries, rounded corners=1pt, inner sep=2.5pt]
      at (2.9, 7.1) {PHASE 2: ZERO-COPY FORK};
    \node[font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=SlateSteel] at (2.9, 6.45) {Turn 1: Fork $K=3$ Candidates};

    % Block Table
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1pt] (0.3, 4.3) rectangle (5.5, 6.1);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=MutedTeal] at (0.4, 5.95) {Cloned Block Tables:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 5.4) {Seq A: L0..L2 $\to$ B12, B15, B19};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 4.9) {Seq B: L0..L2 $\to$ B12, B15, B19};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 4.4) {Seq C: L0..L2 $\to$ B12, B15, B19};

    % Arrow to Physical
    \draw[->, draw=SlateSteel, line width=0.8pt] (2.9, 4.3) -- (2.9, 3.7);

    % Physical Blocks
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 1.8) rectangle (5.5, 3.7);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=ForestSage] at (0.4, 3.55) {Shared Prefix Blocks:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 3.0) {Blk 12, 15, 19: $\mathbf{ref\_count = 3}$};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=ForestSage] at (0.4, 2.3) {Zero Physical Duplication};

    % Status Badge
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 0.3) rectangle (5.5, 1.4);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage, align=center] at (2.9, 0.85) {HBM Traffic: $\mathbf{0\text{ BYTES}}$\\$O(1)$ Host Pointer Clone $< 5\,\mu\text{s}$};
  \end{scope}

  % Column 3: Phase 3 (Lazy CoW Divergence)
  \begin{scope}[shift={(12.8, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.8, 7.3);
    \node[anchor=north, fill=OxfordNavy, text=white, font=\fontsize{6.0}{7}\selectfont\bfseries, rounded corners=1pt, inner sep=2.5pt]
      at (2.9, 7.1) {PHASE 3: LAZY COW DIVERGENCE};
    \node[font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=SlateSteel] at (2.9, 6.45) {Turn 2: Tail Token Emission};

    % Block Table
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1pt] (0.3, 4.3) rectangle (5.5, 6.1);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=WarmOchre] at (0.4, 5.95) {Divergent Block Tables:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 5.4) {Seq A: L2 $\to$ \textbf{Blk 28} (Private)};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 4.9) {Seq B: L2 $\to$ \textbf{Blk 31} (Private)};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 4.4) {Seq C: L2 $\to$ \textbf{Blk 44} (Private)};

    % Arrow to Physical
    \draw[->, draw=SlateSteel, line width=0.8pt] (2.9, 4.3) -- (2.9, 3.7);

    % Physical Blocks
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.6pt, rounded corners=1pt] (0.3, 1.8) rectangle (5.5, 3.7);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=DarkSlate] at (0.4, 3.55) {Private Tail Blocks Popped:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=DarkSlate] at (0.4, 3.0) {Blk 12, 15: $\text{ref}=3$ (Immutable)};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=WarmOchre] at (0.4, 2.3) {Blk 28, 31, 44: $\text{ref}=1$};

    % Status Badge
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1pt] (0.3, 0.3) rectangle (5.5, 1.4);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=WarmOchre, align=center] at (2.9, 0.85) {HBM Traffic: $15.36\text{ MiB}$\\$\mathbf{99.85\%\text{ Bandwidth Saved}}$};
  \end{scope}

  % Column 4: Phase 4 (Reclaim)
  \begin{scope}[shift={(19.0, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (5.8, 7.3);
    \node[anchor=north, fill=OxfordNavy, text=white, font=\fontsize{6.0}{7}\selectfont\bfseries, rounded corners=1pt, inner sep=2.5pt]
      at (2.9, 7.1) {PHASE 4: RECLAIM TO RING};
    \node[font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=SlateSteel] at (2.9, 6.45) {Turn 3: Verifier Pruning};

    % Block Table
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.6pt, rounded corners=1pt] (0.3, 4.3) rectangle (5.5, 6.1);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=Burgundy] at (0.4, 5.95) {Pruned Candidates:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=ForestSage] at (0.4, 5.4) {Seq A: PASS $\implies$ Retain Blk 28};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=Burgundy] at (0.4, 4.9) {Seq B: REJECT $\implies \text{ref} \to 0$};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=Burgundy] at (0.4, 4.4) {Seq C: REJECT $\implies \text{ref} \to 0$};

    % Arrow to Physical
    \draw[->, draw=Burgundy, line width=0.8pt] (2.9, 4.3) -- (2.9, 3.7);

    % Physical Blocks
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 1.8) rectangle (5.5, 3.7);
    \node[anchor=north west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=ForestSage] at (0.4, 3.55) {Ring Buffer Push:};
    \node[anchor=west, font=\fontsize{4.6}{5.3}\selectfont\ttfamily, text=ForestSage] at (0.4, 3.0) {\texttt{+ Blk 31, + Blk 44 Reclaimed}};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=DarkSlate] at (0.4, 2.3) {Zero memory fragmentation};

    % Status Badge
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 0.3) rectangle (5.5, 1.4);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage, align=center] at (2.9, 0.85) {Reclamation: $\mathbf{O(1) < 1\,\mu\text{s}}$\\Zero GPU Compaction Stall};
  \end{scope}

\end{scope}

% =============================================================================
% BOTTOM TIER: MEMORY MAP COMPARISON & HARDWARE RING BUFFER DATAPATH
% =============================================================================

% Left Panel: Physical HBM Memory Map (Contiguous vs Paged CoW)
\begin{scope}[shift={(0, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (14.2, 7.8);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.0}{9.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 7.5) {2. PHYSICAL MEMORY MAP: CONTIGUOUS vs PAGED COW};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{6.0}{7}\selectfont\bfseries]
    at (13.9, 7.4) {Accelerator HBM Address Space};

  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (13.4, 6.7);

    % Sub-panel A: Contiguous Pre-PagedAttention (Pathological)
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.6pt, rounded corners=1.5pt] (0.3, 3.5) rectangle (13.1, 6.4);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=Burgundy] at (0.5, 6.2) {A. Contiguous Static Allocation (Naive): Duplicate Prefixes \& Jagged Holes};

    % Contiguous memory bar
    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt] (0.5, 4.3) rectangle (3.8, 5.3);
    \node[font=\fontsize{4.6}{5.3}\selectfont\bfseries, text=MutedTeal] at (2.15, 4.95) {Seq A Prefix ($2.56\text{ GB}$)};
    \node[font=\fontsize{4.0}{4.6}\selectfont\ttfamily, text=DarkSlate] at (2.15, 4.55) {Physical Copy 1};

    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt] (4.0, 4.3) rectangle (7.3, 5.3);
    \node[font=\fontsize{4.6}{5.3}\selectfont\bfseries, text=MutedTeal] at (5.65, 4.95) {Seq B Prefix ($2.56\text{ GB}$)};
    \node[font=\fontsize{4.0}{4.6}\selectfont\ttfamily, text=Burgundy] at (5.65, 4.55) {Duplicate Copy 2};

    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt] (7.5, 4.3) rectangle (10.8, 5.3);
    \node[font=\fontsize{4.6}{5.3}\selectfont\bfseries, text=MutedTeal] at (9.15, 4.95) {Seq C Prefix ($2.56\text{ GB}$)};
    \node[font=\fontsize{4.0}{4.6}\selectfont\ttfamily, text=Burgundy] at (9.15, 4.55) {Duplicate Copy 3};

    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.5pt, dashed] (11.0, 4.3) rectangle (12.9, 5.3);
    \node[font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=Burgundy, align=center] at (11.95, 4.8) {Hole\\(OOM)};

    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=Burgundy] at (0.5, 4.1) {Total Footprint: $3 \times 2.56 = 7.68\text{ GiB}$ duplicate bus traffic $\cdot$ External fragmentation prevents new prefills};


    % Sub-panel B: PagedAttention Virtualized CoW (Optimal)
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt, rounded corners=1.5pt] (0.3, 0.3) rectangle (13.1, 3.2);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] at (0.5, 3.0) {B. PagedAttention Virtualized CoW: Unified Fixed-Size 16-Token Block Frames};

    % Shared Prefix Block Bar
    \draw[fill=TealL0, draw=MutedTeal, line width=0.7pt] (0.5, 1.1) rectangle (4.2, 2.1);
    \node[font=\fontsize{4.8}{5.5}\selectfont\bfseries, text=MutedTeal] at (2.35, 1.75) {Shared Prefix: Blk 12, 15};
    \node[font=\fontsize{4.2}{5.0}\selectfont\ttfamily, text=ForestSage] at (2.35, 1.35) {$\text{ref}=3 \implies 2.56\text{ GB}$ ONCE};

    % Private Tails
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt] (4.4, 1.1) rectangle (6.2, 2.1);
    \node[font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=ForestSage] at (5.3, 1.75) {Blk 28 (Seq A)};
    \node[font=\fontsize{3.8}{4.5}\selectfont\ttfamily, text=DarkSlate] at (5.3, 1.35) {Retained};

    \draw[fill=CoolParchment, draw=SlateRim, line width=0.5pt, dashed] (6.4, 1.1) rectangle (8.2, 2.1);
    \node[font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=SlateSteel] at (7.3, 1.75) {Blk 31 (Seq B)};
    \node[font=\fontsize{3.8}{4.5}\selectfont\ttfamily, text=ForestSage] at (7.3, 1.35) {Reclaimed};

    \draw[fill=CoolParchment, draw=SlateRim, line width=0.5pt, dashed] (8.4, 1.1) rectangle (10.2, 2.1);
    \node[font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=SlateSteel] at (9.3, 1.75) {Blk 44 (Seq C)};
    \node[font=\fontsize{3.8}{4.5}\selectfont\ttfamily, text=ForestSage] at (9.3, 1.35) {Reclaimed};

    % Free Pool
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt] (10.4, 1.1) rectangle (12.9, 2.1);
    \node[font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=ForestSage] at (11.65, 1.75) {Free Block Pool};
    \node[font=\fontsize{3.8}{4.5}\selectfont\ttfamily, text=ForestSage] at (11.65, 1.35) {Ring Buffer ($O(1)$)};

    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=ForestSage] at (0.5, 0.9) {Total Footprint: $2.56\text{ GiB} + 15.36\text{ MiB}$ $\implies$ Zero duplication, zero holes, zero defragmentation};
  \end{scope}
\end{scope}

% Right Panel: Virtual Translation Table & Circular Free-List Ring Buffer Datapath
\begin{scope}[shift={(14.8, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (10.6, 7.8);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.0}{9.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 7.5) {3. HARDWARE FREE-LIST RING BUFFER DATAPATH};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{6.0}{7}\selectfont\bfseries]
    at (10.3, 7.4) {Page Allocator};

  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (9.8, 6.7);

    % Ring buffer diagram
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.6pt, rounded corners=1.5pt] (0.3, 3.4) rectangle (9.5, 6.4);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=DarkSlate] at (0.5, 6.2) {Circular Ring Buffer Register ($N=98{,}304$ blocks):};

    % Ring blocks
    \foreach \i/\blk in {0/{Blk 52}, 1/{Blk 53}, 2/{Blk 54}, 3/{Blk 31}, 4/{Blk 44}, 5/{...}} {
      \pgfmathsetmacro{\rx}{0.6 + \i * 1.45}
      \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt, rounded corners=1pt] (\rx, 4.4) rectangle ++(1.3, 0.9);
      \node[font=\fontsize{4.6}{5.3}\selectfont\ttfamily\bfseries, text=DarkSlate] at (\rx+0.65, 4.85) {\blk};
    }

    % Pointers
    \draw[<-, draw=ForestSage, line width=1.0pt] (0.6+0.65, 4.4) -- ++(0, -0.5)
      node[below, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=ForestSage] {HEAD (Pop)};
    \draw[<-, draw=Burgundy, line width=1.0pt] (0.6+4*1.45+0.65, 4.4) -- ++(0, -0.5)
      node[below, font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=Burgundy] {TAIL (Push)};

    \node[anchor=north west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=DarkSlate] at (0.5, 3.65) {Push/Pop: $(\text{ptr} + 1) \pmod N$ in $O(1) < 1\,\mu\text{s}$};


    % Microarchitectural Comparison Matrix Box
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.6pt, rounded corners=1.5pt] (0.3, 0.3) rectangle (9.5, 3.1);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=DarkSlate] at (0.5, 2.9) {Systems Performance Comparison:};

    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily, text=DarkSlate] at (0.5, 2.3) {\textbf{Fork Bus Traffic:} $10.24\text{ GiB} \to \mathbf{0\text{ Bytes}}$ ($100\%$ zero-copy)};
    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily, text=DarkSlate] at (0.5, 1.8) {\textbf{Fork Kernel Stall:} $5.1\text{ ms} \to \mathbf{4.2\,\mu\text{s}}$ ($1{,}200\times$ faster)};
    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily, text=DarkSlate] at (0.5, 1.3) {\textbf{HBM Peak Capacity:} $30.72\text{ GiB} \to \mathbf{10.26\text{ GiB}}$ ($3\times$ savings)};
    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily\bfseries, text=ForestSage] at (0.5, 0.75) {\textbf{Serving Density:} $4\text{--}8 \to \mathbf{>48\text{ Deliberative Agents}}$};

  \end{scope}
\end{scope}

\end{tikzpicture}
\end{document}
"""

def main():
    with open(TEX_PATH, "w") as f:
        f.write(TIKZ_CONTENT.strip() + "\n")
    print(f"Wrote TikZ source to {TEX_PATH}")

    # Compile via pdflatex
    cmd_pdf = ["pdflatex", "-interaction=nonstopmode", "-output-directory=/tmp", TEX_PATH]
    res_pdf = subprocess.run(cmd_pdf, capture_output=True, text=True)
    if res_pdf.returncode != 0:
        print("pdflatex error:\n", res_pdf.stdout[-1500:])
        sys.exit(1)

    tmp_pdf = "/tmp/kv_cow_forking_lifecycle.pdf"
    
    # Save PDF
    target_pdf = os.path.join(PDF_DIR, "kv_cow_forking_lifecycle.pdf")
    subprocess.run(["cp", tmp_pdf, target_pdf], check=True)

    # Render PNG via pdftoppm (250 DPI)
    png_prefix = "/tmp/kv_cow_render"
    subprocess.run(["pdftoppm", "-png", "-r", "250", tmp_pdf, png_prefix], check=True)
    rendered_png = f"{png_prefix}-1.png"

    # Save to both target PNG locations
    for p in ["kv_cow_forking_lifecycle.png", "fig-vol3-kv-cow-forking-lifecycle.png"]:
        t = os.path.join(PNG_DIR, p)
        subprocess.run(["cp", rendered_png, t], check=True)
        print(f"Saved PNG to {t}")

    # Render SVG via Inkscape
    inkscape_bin = "/opt/homebrew/bin/inkscape" if os.path.exists("/opt/homebrew/bin/inkscape") else "inkscape"
    for s in ["kv_cow_forking_lifecycle.svg", "fig-vol3-kv-cow-forking-lifecycle.svg"]:
        t = os.path.join(SVG_DIR, s)
        cmd_svg = [inkscape_bin, "--export-type=svg", f"--export-filename={t}", tmp_pdf]
        subprocess.run(cmd_svg, check=True)
        # Sanitize any C0 control characters
        with open(t, "rb") as f:
            svg_data = f.read()
        clean_svg = re.sub(b"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", svg_data)
        if clean_svg != svg_data:
            with open(t, "wb") as f:
                f.write(clean_svg)
        print(f"Saved & sanitized SVG to {t}")

    print("\nChapter 5 KV CoW Forking Lifecycle figure successfully built and deployed!")

if __name__ == "__main__":
    main()
