#!/usr/bin/env python3
"""
Build publication-grade schematic for Chapter 6:
Dual Write Path & Write Admission Quarantine Pipeline (dual_write_path_quarantine)

Standards:
- Canonical academic palette (OxfordNavy, SlateSteel, MutedTeal, Burgundy, WarmOchre, ForestSage)
- Pure Helvetica typography (helvet + T1 fontenc)
- Strict 3-level font hierarchy (8.0pt badges, 6.5pt components, 5.0pt metrics/signals)
- Authentic IEEE/ACM computer architecture glyphs:
  - Dual-path event bifurcation datapath
  - 5-stage write admission quarantine pipeline
  - Execution latency waterfall (Synchronous Coupling stall vs Decoupled Dual-Path)
  - Multi-substrate storage materialization & atomic tombstone bitset
- Zero bullet points, zero paragraphs of prose
- No burned-in banners / titles (clean canvas)
"""

import os
import subprocess
import sys
import re

TEX_PATH = "books/vol3/tex/dual_write_path_quarantine.tex"
PNG_DIR = "books/vol3/06_long_term_memory/images/png"
SVG_DIR = "books/vol3/06_long_term_memory/images/svg"
PDF_DIR = "books/vol3/06_long_term_memory/images/pdf"

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
% TOP-LEFT (PANEL 1): DUAL WRITE PATH EVENT BIFURCATION DATAPATH
% =============================================================================
\begin{scope}[shift={(0, 8.4)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (11.8, 8.0);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{7.5}{9.0}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 7.7) {1. DUAL WRITE PATH EVENT BIFURCATION};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{5.8}{7}\selectfont\bfseries]
    at (11.5, 7.6) {Interactive Critical-Path Decoupling};

  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (11.0, 6.9);

    % Agent Emission Node
    \draw[fill=CoolParchment, draw=SlateSteel, line width=0.8pt, rounded corners=2pt] (0.5, 5.4) rectangle (10.5, 6.5);
    \node[anchor=north west, font=\fontsize{5.8}{7}\selectfont\bfseries, text=DarkSlate] at (0.7, 6.3) {Agent Turn Event Tuple $e_t$:};
    \node[anchor=west, font=\fontsize{5.0}{5.8}\selectfont\ttfamily, text=SlateSteel] at (0.7, 5.75) {$e_t = \langle \text{Turn } t, \text{Task } \tau, a_t, O_t, \Delta_t, \text{ExitCode } 0 \rangle$};

    % Bifurcation Splitter
    \draw[->, draw=MutedTeal, line width=1.3pt] (3.0, 5.4) -- (3.0, 4.7);
    \draw[->, draw=WarmOchre, line width=1.3pt] (8.0, 5.4) -- (8.0, 4.7);

    % PATH 1: Synchronous WAL Append (Left Branch)
    \draw[fill=TealL0, draw=MutedTeal, line width=0.9pt, rounded corners=2pt] (0.5, 1.8) rectangle (5.3, 4.7);
    \node[anchor=north, font=\fontsize{6.2}{7.2}\selectfont\bfseries, text=MutedTeal] at (2.9, 4.5) {PATH 1: Synchronous Append WAL};
    \node[font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate, align=center] at (2.9, 3.65) {Deterministic POSIX stream\\Non-blocking Ring Buffer\\Direct async NVMe SSD flush};
    
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.7, 2.0) rectangle (5.1, 2.7);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage] at (2.9, 2.35) {Critical Path: $T_{\mathrm{wal}} < 0.45\text{ ms}$};

    % PATH 2: Asynchronous Admission Queue (Right Branch)
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.9pt, rounded corners=2pt] (5.7, 1.8) rectangle (10.5, 4.7);
    \node[anchor=north, font=\fontsize{6.2}{7.2}\selectfont\bfseries, text=WarmOchre] at (8.1, 4.5) {PATH 2: Async Quarantine Queue};
    \node[font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate, align=center] at (8.1, 3.65) {Out-of-band background queue\\Decoupled from model tokens\\Zero serving queue locks};

    \draw[fill=CoolParchment, draw=SlateRim, line width=0.6pt, rounded corners=1pt] (5.9, 2.0) rectangle (10.3, 2.7);
    \node[font=\fontsize{5.0}{6}\selectfont\ttfamily, text=DarkSlate] at (8.1, 2.35) {Dispatched during tool/idle};

    % Fast Return Arrow to Core
    \draw[->, draw=ForestSage, line width=1.1pt] (2.9, 1.8) -- (2.9, 1.1) -- (0.8, 1.1) -- (0.8, 0.4);
    \node[anchor=north west, font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage] at (1.0, 0.95) {Zero Pipeline Stall: Control Returns in $0.45\text{ ms}$};

  \end{scope}
\end{scope}

% =============================================================================
% TOP-RIGHT (PANEL 2): 5-STAGE WRITE ADMISSION QUARANTINE SIEVE
% =============================================================================
\begin{scope}[shift={(12.4, 8.4)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (13.6, 8.0);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{7.5}{9.0}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 7.7) {2. FIVE-STAGE WRITE ADMISSION QUARANTINE SIEVE};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{5.8}{7}\selectfont\bfseries]
    at (13.3, 7.6) {Memory Hygiene Firewall};

  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (12.8, 6.9);

    % Stage 1: Tenant Isolation
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1.5pt] (0.3, 5.5) rectangle (12.5, 6.5);
    \node[anchor=west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=MutedTeal] at (0.5, 6.15) {Stage 1: Tenant Principal Scoping};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.5, 5.75) {Enforces namespace $\mathcal{S}_{\mathrm{tenant}}$ $\cdot$ Rejects cross-tenant taint};
    \node[anchor=east, font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage] at (12.3, 6.0) {ISOLATE};

    % Stage 2: Secret & PII Filter
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1.5pt] (0.3, 4.3) rectangle (12.5, 5.3);
    \node[anchor=west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=WarmOchre] at (0.5, 4.95) {Stage 2: Secret \& Entropy Scrubber};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.5, 4.55) {Regex + Shannon entropy: strips API tokens ($\mathcal{H} > 4.5$) \& credentials};
    \node[anchor=east, font=\fontsize{5.0}{6}\selectfont\bfseries, text=WarmOchre] at (12.3, 4.8) {SCRUB};

    % Stage 3: Provenance Pointer
    \draw[fill=CoolParchment, draw=SlateRim, line width=0.6pt, rounded corners=1.5pt] (0.3, 3.1) rectangle (12.5, 4.1);
    \node[anchor=west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=DarkSlate] at (0.5, 3.75) {Stage 3: SHA-256 Provenance Anchor};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.5, 3.35) {Cryptographic hash binds entry to immutable WAL record $\text{id}_{\mathrm{traj}}$};
    \node[anchor=east, font=\fontsize{5.0}{6}\selectfont\ttfamily\bfseries, text=SlateSteel] at (12.3, 3.6) {ANCHOR};

    % Stage 4: Invariant Verification Closure
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1.5pt] (0.3, 1.9) rectangle (12.5, 2.9);
    \node[anchor=west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=MutedTeal] at (0.5, 2.55) {Stage 4: Verification Closure Oracle};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.5, 2.15) {Asserts deterministic proof ($\text{ExitCode}=0$, unit test pass)};
    \node[anchor=east, font=\fontsize{5.0}{6}\selectfont\bfseries, text=ForestSage] at (12.3, 2.4) {VERIFY};

    % Stage 5: Surprisal / Novelty Filter
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt, rounded corners=1.5pt] (0.3, 0.4) rectangle (12.5, 1.6);
    \node[anchor=west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=Burgundy] at (0.5, 1.25) {Stage 5: Geometric Surprisal / Novelty Filter};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.5, 0.75) {$\mathcal{N}(e) = 1 - \max \cos(E_D(e), E_D(m)) > \tau_{\mathrm{novel}}$};
    \node[anchor=east, font=\fontsize{5.0}{6}\selectfont\bfseries, text=Burgundy] at (12.3, 1.0) {SURPRISAL};

  \end{scope}
\end{scope}

% =============================================================================
% BOTTOM-LEFT (PANEL 3): LATENCY WATERFALL: SYNC COUPLING vs DUAL-PATH
% =============================================================================
\begin{scope}[shift={(0, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (14.2, 7.8);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{7.5}{9.0}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 7.5) {3. LATENCY WATERFALL: SYNC COUPLING vs DUAL-PATH};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{5.8}{7}\selectfont\bfseries]
    at (13.9, 7.4) {Execution Wall-Time Comparison};

  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (13.4, 6.7);

    % Naive Synchronous Coupling Waterfall (Top)
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.6pt, rounded corners=1.5pt] (0.3, 3.5) rectangle (13.1, 6.4);
    \node[anchor=north west, font=\fontsize{5.8}{7}\selectfont\bfseries, text=Burgundy] at (0.5, 6.2) {A. Synchronous Ingestion Coupling (Antipattern): $5.37\text{ s}$ STALL};

    % Waterfall segments
    % Segment 1: Model Token Decode (40 ms)
    \draw[fill=TealL0, draw=MutedTeal, line width=0.5pt] (0.5, 4.8) rectangle (1.5, 5.6);
    \node[font=\fontsize{4.2}{5}\selectfont\bfseries, text=MutedTeal] at (1.0, 5.2) {Decode};

    % Segment 2: Embedding Inference (180 ms)
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.5pt] (1.6, 4.8) rectangle (3.0, 5.6);
    \node[font=\fontsize{4.2}{5}\selectfont\bfseries, text=WarmOchre] at (2.3, 5.2) {Embed};

    % Segment 3: Vector Index Lock (350 ms)
    \draw[fill=CoolParchment, draw=SlateSteel, line width=0.5pt] (3.1, 4.8) rectangle (4.8, 5.6);
    \node[font=\fontsize{4.2}{5}\selectfont\bfseries, text=SlateSteel] at (3.95, 5.2) {HNSW Lock};

    % Segment 4: LLM Reflection Engine (4,800 ms)
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt] (4.9, 4.8) rectangle (12.8, 5.6);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=Burgundy] at (8.85, 5.2) {LLM Trajectory Reflection \& Consolidation ($4{,}800\text{ ms}$)};

    \node[anchor=north west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=Burgundy] at (0.5, 4.3) {Total Interactive Stall: $T_{\mathrm{turn}} = 40 + 180 + 350 + 4800 = \mathbf{5{,}370\text{ ms}}$ per turn!};
    \node[anchor=north west, font=\fontsize{4.4}{5.2}\selectfont\ttfamily, text=DarkSlate] at (0.5, 3.85) {Blocks serving engine worker thread $\cdot$ Locks batch slots $\cdot$ Severe queue starvation};


    % Decoupled Dual-Path Waterfall (Bottom)
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt, rounded corners=1.5pt] (0.3, 0.3) rectangle (13.1, 3.2);
    \node[anchor=north west, font=\fontsize{5.8}{7}\selectfont\bfseries, text=ForestSage] at (0.5, 3.0) {B. Production Dual-Path Architecture: Sub-Millisecond Return ($0.45\text{ ms}$)};

    % Turn t Critical Path
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt] (0.5, 1.8) rectangle (1.5, 2.5);
    \node[font=\fontsize{4.2}{5}\selectfont\bfseries, text=MutedTeal] at (1.0, 2.15) {Decode};

    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt] (1.55, 1.8) rectangle (2.1, 2.5);
    \node[font=\fontsize{3.8}{4.5}\selectfont\bfseries, text=ForestSage] at (1.82, 2.15) {WAL};

    % Turn t+1 begins immediately!
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt] (2.2, 1.8) rectangle (3.2, 2.5);
    \node[font=\fontsize{4.2}{5}\selectfont\bfseries, text=MutedTeal] at (2.7, 2.15) {Turn $t+1$};

    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt] (3.3, 1.8) rectangle (4.3, 2.5);
    \node[font=\fontsize{4.2}{5}\selectfont\bfseries, text=MutedTeal] at (3.8, 2.15) {Turn $t+2$};

    % Background Reflection Daemon
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, dashed] (2.2, 0.9) rectangle (12.8, 1.5);
    \node[font=\fontsize{4.5}{5.2}\selectfont\bfseries, text=WarmOchre] at (7.5, 1.2) {Background Quarantine \& Reflection Daemon (100\% Overlapped Execution)};

    \node[anchor=north west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily\bfseries, text=ForestSage] at (0.5, 0.7) {Interactive Delay: $T_{\mathrm{wal}} = \mathbf{0.45\text{ ms}}$ $\implies \mathbf{11{,}900\times\text{ Faster Responsive Turn Release}}$};

  \end{scope}
\end{scope}

% =============================================================================
% BOTTOM-RIGHT (PANEL 4): MULTI-SUBSTRATE MATERIALIZATION & TOMBSTONES
% =============================================================================
\begin{scope}[shift={(14.8, 0)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (11.2, 7.8);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{7.5}{9.0}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 7.5) {4. MULTI-SUBSTRATE MEMORY MATERIALIZATION};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{5.8}{7}\selectfont\bfseries]
    at (10.9, 7.4) {Durable Learning Engine};

  \begin{scope}[shift={(0.4, 0.4)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.4, 6.7);

    % Substrate 1: Dense ANN Vector Store
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1.5pt] (0.3, 4.6) rectangle (10.1, 6.4);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=MutedTeal] at (0.5, 6.2) {Substrate A: Dense ANN Vector Store (IVF-PQ)};
    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily, text=DarkSlate] at (0.5, 5.5) {Admitted novel reflection embeddings $\cdot$ Centroid clustering};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=SlateSteel] at (0.5, 4.9) {Quantized codebooks $\cdot$ Sub-vector ADC distance tables};

    % Substrate 2: Structural Code Property Graph
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.6pt, rounded corners=1.5pt] (0.3, 2.5) rectangle (10.1, 4.3);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=WarmOchre] at (0.5, 4.1) {Substrate B: Structural Property Graph (SCIP)};
    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily, text=DarkSlate] at (0.5, 3.4) {Tree-sitter AST nodes $\cdot$ Def-use / call graph dependencies};
    \node[anchor=west, font=\fontsize{4.5}{5.2}\selectfont\ttfamily, text=SlateSteel] at (0.5, 2.8) {Exact deterministic symbol resolution for tools};

    % Substrate 3: In-Memory Tombstone Bitset
    \draw[fill=SageL0, draw=ForestSage, line width=0.7pt, rounded corners=1.5pt] (0.3, 0.4) rectangle (10.1, 2.2);
    \node[anchor=north west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] at (0.5, 2.0) {Substrate C: In-Memory Fast Tombstone Bitset};
    \node[anchor=west, font=\fontsize{4.6}{5.4}\selectfont\ttfamily, text=DarkSlate] at (0.5, 1.35) {Atomic 64-bit word masks $\cdot$ Suppresses stale memory chunks};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily\bfseries, text=ForestSage] at (0.5, 0.75) {Invalidation Latency: $\mathbf{400\text{ ns}}$ Zero-Lock Bitwise Mask};

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

    tmp_pdf = "/tmp/dual_write_path_quarantine.pdf"
    
    # Save PDF
    target_pdf = os.path.join(PDF_DIR, "dual_write_path_quarantine.pdf")
    subprocess.run(["cp", tmp_pdf, target_pdf], check=True)

    # Render PNG via pdftoppm (250 DPI)
    png_prefix = "/tmp/dual_write_render"
    subprocess.run(["pdftoppm", "-png", "-r", "250", tmp_pdf, png_prefix], check=True)
    rendered_png = f"{png_prefix}-1.png"

    # Save to both target PNG locations
    for p in ["dual_write_path_quarantine.png", "fig-vol3-dual-write-path-quarantine.png"]:
        t = os.path.join(PNG_DIR, p)
        subprocess.run(["cp", rendered_png, t], check=True)
        print(f"Saved PNG to {t}")

    # Render SVG via Inkscape
    inkscape_bin = "/opt/homebrew/bin/inkscape" if os.path.exists("/opt/homebrew/bin/inkscape") else "inkscape"
    for s in ["dual_write_path_quarantine.svg", "fig-vol3-dual-write-path-quarantine.svg"]:
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

    print("\nChapter 6 Dual Write Path figure successfully built and deployed!")

if __name__ == "__main__":
    main()
