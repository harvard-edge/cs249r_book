#!/usr/bin/env python3
"""
Build publication-grade schematic for Chapter 3:
Speculative Verification Hardware Pipeline Timeline (speculative_verification_timeline)

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
import re

TEX_PATH = "books/vol3/tex/speculative_verification_timeline.tex"
PNG_DIR = "books/vol3/03_test_time_compute/images/png"
SVG_DIR = "books/vol3/03_test_time_compute/images/svg"
PDF_DIR = "books/vol3/03_test_time_compute/images/pdf"

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
\definecolor{GoldPin}{HTML}{D97706}
\definecolor{DarkSlate}{HTML}{0F172A}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth,
  node distance=0.5cm
]

% =============================================================================
% TOP TIER: TIMELINE COMPARISON (SERIAL BASELINE vs SPECULATIVE PIPELINE)
% =============================================================================

% Left Unit: Serial Lockstep Baseline
\begin{scope}[shift={(0, 7.6)}]
  \draw[fill=CoolParchment, draw=Burgundy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (9.6, 6.8);

  \node[anchor=north west, fill=Burgundy, text=white, font=\fontsize{8.0}{9.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.5) {1. SERIAL LOCKSTEP BASELINE (NAIVE)};
  \node[anchor=north east, text=Burgundy, font=\fontsize{6.0}{7}\selectfont\bfseries]
    at (9.3, 6.4) {$T_{\mathrm{step}} = 1{,}786\text{ ms}$};

  % Timeline canvas
  \begin{scope}[shift={(0.4, 0.5)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (8.8, 5.4);

    % Axis
    \draw[->, draw=SlateSteel, line width=0.8pt] (0.5, 0.7) -- (8.3, 0.7)
      node[right, font=\fontsize{5.2}{6}\selectfont\ttfamily, text=SlateSteel] {Time $t$};

    % Step 1: Draft Decode
    \draw[fill=TealL0, draw=MutedTeal, line width=0.8pt, rounded corners=1.5pt] (0.6, 3.2) rectangle (2.6, 4.8);
    \node[anchor=north, text=MutedTeal, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (1.6, 4.6) {8B Draft Decode};
    \node[text=DarkSlate, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (1.6, 3.9) {128 tok $\times 2.39\text{ ms}$};
    \node[text=MutedTeal, font=\fontsize{5.2}{6}\selectfont\bfseries] at (1.6, 3.45) {$T_{\mathrm{draft}} = 306\text{ ms}$};

    % ALU Idle Bubble below Draft Decode
    \draw[fill=BurgundyL0, draw=Burgundy, dashed, line width=0.7pt, rounded corners=1pt] (0.6, 1.3) rectangle (2.6, 2.7);
    \node[text=Burgundy, font=\fontsize{5.8}{7}\selectfont\bfseries] at (1.6, 2.15) {ALU Idle Bubble};
    \node[text=Burgundy, font=\fontsize{4.8}{5.5}\selectfont] at (1.6, 1.65) {Tensor Cores $< 5\%$ Active};

    % Step 2: 70B Verifier Scoring
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt, rounded corners=1.5pt] (2.9, 1.3) rectangle (8.2, 2.9);
    \node[anchor=north, text=WarmOchre, font=\fontsize{6.5}{7.5}\selectfont\bfseries] at (5.55, 2.7) {70B Verifier Scoring (Prefill)};
    \node[text=DarkSlate, font=\fontsize{5.0}{6}\selectfont\ttfamily] at (5.55, 2.0) {4,224 tokens @ 400 TFLOP/s FP8};
    \node[text=WarmOchre, font=\fontsize{5.2}{6}\selectfont\bfseries] at (5.55, 1.55) {$T_{\mathrm{verif}} = 1{,}480\text{ ms}$};

    % Memory Bus Idle Bubble above Verifier Prefill
    \draw[fill=BurgundyL0, draw=Burgundy, dashed, line width=0.7pt, rounded corners=1pt] (2.9, 3.4) rectangle (8.2, 4.8);
    \node[text=Burgundy, font=\fontsize{5.8}{7}\selectfont\bfseries] at (5.55, 4.25) {Memory Bus Idle Bubble};
    \node[text=Burgundy, font=\fontsize{4.8}{5.5}\selectfont] at (5.55, 3.75) {HBM Bandwidth $< 8\%$ Utilized};
  \end{scope}

\end{scope}


% Right Unit: Speculative Verification Pipeline
\begin{scope}[shift={(10.2, 7.6)}]
  \draw[fill=CoolParchment, draw=ForestSage, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (14.2, 6.8);

  \node[anchor=north west, fill=ForestSage, text=white, font=\fontsize{8.0}{9.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.5) {2. SPECULATIVE PIPELINE EXECUTION};
  \node[anchor=north east, text=ForestSage, font=\fontsize{6.0}{7}\selectfont\bfseries]
    at (13.9, 6.4) {Critical Path: $\max(T_{\mathrm{draft}}, T_{\mathrm{verif}}) = 1{,}480\text{ ms}$};

  % Pipeline Canvas
  \begin{scope}[shift={(0.4, 0.5)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (13.4, 5.4);

    % Upper Lane: 8B Draft Decoder
    \node[anchor=west, text=MutedTeal, font=\fontsize{6.2}{7.5}\selectfont\bfseries] at (0.3, 4.7) {8B Draft:};

    % Step 1
    \draw[fill=TealL0, draw=MutedTeal, line width=0.6pt, rounded corners=1pt] (1.8, 4.3) rectangle (3.8, 5.1);
    \node[font=\fontsize{5.2}{6}\selectfont\bfseries, text=MutedTeal] at (2.8, 4.7) {Draft Step 1};

    % Step 2
    \draw[fill=TealL0, draw=MutedTeal, line width=0.8pt, rounded corners=1pt] (4.1, 4.3) rectangle (6.5, 5.1);
    \node[font=\fontsize{5.2}{6}\selectfont\bfseries, text=MutedTeal] at (5.3, 4.8) {Draft Step 2};
    \node[font=\fontsize{4.4}{5.2}\selectfont\ttfamily, text=ForestSage] at (5.3, 4.45) {[Speculative]};

    % Step 3
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.6pt, dashed, rounded corners=1pt] (7.2, 4.3) rectangle (9.6, 5.1);
    \node[font=\fontsize{5.2}{6}\selectfont\bfseries, text=Burgundy] at (8.4, 4.8) {Draft Step 3};
    \node[font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=Burgundy] at (8.4, 4.45) {[ABORTED]};

    % Lower Lane: 70B Verifier
    \node[anchor=west, text=WarmOchre, font=\fontsize{6.2}{7.5}\selectfont\bfseries] at (0.3, 2.7) {70B Verifier:};

    % Verifier Scoring Step 1
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt, rounded corners=1pt] (4.1, 2.3) rectangle (6.5, 3.1);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=WarmOchre, align=center] at (5.3, 2.7) {Verify Step 1\\(Compute-Bound)};

    % Verifier Scoring Step 2
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt, rounded corners=1pt] (7.2, 2.3) rectangle (9.6, 3.1);
    \node[font=\fontsize{5.0}{6}\selectfont\bfseries, text=WarmOchre, align=center] at (8.4, 2.7) {Verify Step 2\\(Failed Invariant)};

    % Resolution Gates
    % Gate 1: ACCEPT
    \draw[->, draw=ForestSage, line width=1.3pt] (5.3, 2.3) -- (5.3, 1.4);
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt, rounded corners=1pt] (4.1, 0.7) rectangle (6.5, 1.4);
    \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] at (5.3, 1.05) {ACCEPT $\implies$ Commit};

    % Gate 2: REJECT
    \draw[->, draw=Burgundy, line width=1.3pt] (8.4, 2.3) -- (8.4, 1.4);
    \draw[fill=BurgundyL0, draw=Burgundy, line width=0.8pt, rounded corners=1pt] (7.2, 0.7) rectangle (9.6, 1.4);
    \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=Burgundy] at (8.4, 1.05) {REJECT $\implies$ Rollback};

    % Inter-lane datapath / pipeline dependency arrows
    \draw[->, draw=MutedTeal, line width=0.8pt] (3.8, 4.7) -| (4.1, 3.1);
    \draw[->, draw=MutedTeal, line width=0.8pt] (6.5, 4.7) -| (7.2, 3.1);

    % Abort feedback trigger from Failed Invariant to Draft Step 3
    \draw[->, draw=Burgundy, line width=0.9pt, dashed] (8.4, 3.1) -- (8.4, 4.3)
      node[midway, right, font=\fontsize{4.4}{5.2}\selectfont\bfseries, text=Burgundy] {Abort Trigger};

    % Fast Rollback Action Box
    \draw[fill=CoolParchment, draw=Burgundy, line width=0.7pt, rounded corners=1.5pt] (10.1, 2.2) rectangle (13.1, 3.8);
    \node[anchor=north, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=Burgundy] at (11.6, 3.6) {Fast Rollback: $T_{\mathrm{rb}} < 35\,\mu\text{s}$};
    \node[text=DarkSlate, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, align=center] at (11.6, 2.8) {PagedAttention\\Block Reclaim};

    % Rollback arrow
    \draw[->, draw=Burgundy, line width=1.1pt] (9.6, 1.05) -| (11.6, 2.2);

    % Wall-clock time axis
    \draw[->, draw=SlateSteel, line width=0.8pt] (0.5, 0.3) -- (12.8, 0.3)
      node[right, font=\fontsize{5.2}{6}\selectfont\ttfamily, text=SlateSteel] {Wall Time $t$};
  \end{scope}

\end{scope}


% =============================================================================
% BOTTOM TIER: MEMORY CO-LOCATION & ASYMMETRIC ROOFLINE ABSORPTION
% =============================================================================

% Left Sub-box: 80 GB HBM3 Co-Location
\begin{scope}[shift={(0, 0.3)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (11.8, 6.7);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.0}{9.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.4) {3. SINGLE-GPU MEMORY CO-LOCATION};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{6.0}{7}\selectfont\bfseries]
    at (11.5, 6.3) {80 GB H100 $\cdot$ Zero Swapping};

  \begin{scope}[shift={(0.5, 0.6)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 5.3);

    % Memory Allocation Bar
    \node[anchor=north west, font=\fontsize{6.8}{8}\selectfont\bfseries, text=DarkSlate] at (0.3, 4.9) {Physical 80 GB HBM3 Allocation Map:};

    % 70B Policy (70 GB)
    \draw[fill=TealL0, draw=MutedTeal, line width=0.8pt] (0.3, 2.7) rectangle (8.4, 4.2);
    \node[font=\fontsize{6.5}{7.5}\selectfont\bfseries, text=MutedTeal] at (4.35, 3.7) {Primary 70B Generation Policy (70.0 GB in FP8)};
    \node[font=\fontsize{5.2}{6}\selectfont\ttfamily, text=DarkSlate] at (4.35, 3.1) {Occupancy: $87.5\%$ $\cdot$ Memory-Bandwidth Decode Bound};

    % 8B PRM Verifier (8 GB)
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt] (8.4, 2.7) rectangle (9.8, 4.2);
    \node[font=\fontsize{5.2}{6}\selectfont\bfseries, text=WarmOchre, align=center] at (9.1, 3.65) {8B PRM\\8.0 GB};
    \node[font=\fontsize{4.4}{5.2}\selectfont\ttfamily, text=DarkSlate] at (9.1, 2.95) {10.0\%};

    % Paged KV Cache (2 GB)
    \draw[fill=SageL0, draw=ForestSage, line width=0.8pt] (9.8, 2.7) rectangle (10.5, 4.2);
    \node[font=\fontsize{4.8}{5.5}\selectfont\bfseries, text=ForestSage, align=center] at (10.15, 3.65) {KV\\2 GB};
    \node[font=\fontsize{4.2}{5}\selectfont\ttfamily, text=DarkSlate] at (10.15, 2.95) {2.5\%};

    % Systems Metrics Panel (clean rows without bullet points)
    \draw[fill=CoolParchment, draw=SlateRim, rounded corners=1.5pt] (0.3, 0.4) rectangle (10.5, 2.2);
    \node[anchor=west, font=\fontsize{5.2}{6.2}\selectfont\ttfamily, text=DarkSlate] at (0.5, 1.75) {\textbf{Resident Weights:} 78.0 GB footprint (87.5\% 70B Gen + 10\% 8B PRM + 2.5\% KV)};
    \node[anchor=west, font=\fontsize{5.2}{6.2}\selectfont\ttfamily, text=DarkSlate] at (0.5, 1.25) {\textbf{Zero Weight Swapping:} Saves 210 GB PCIe Gen5 transfers ($3.28\text{ s}$ per step)};
    \node[anchor=west, font=\fontsize{5.2}{6.2}\selectfont\ttfamily\bfseries, text=ForestSage] at (0.5, 0.75) {\textbf{Cluster Economy:} Single-GPU self-contained serving vs 4-GPU tensor cluster};
  \end{scope}

\end{scope}


% Right Sub-box: Asymmetric Roofline Latency Absorption
\begin{scope}[shift={(12.6, 0.3)}]
  \draw[fill=CoolParchment, draw=OxfordNavy, line width=1.1pt, rounded corners=3pt]
    (0, 0) rectangle (11.8, 6.7);

  \node[anchor=north west, fill=OxfordNavy, text=white, font=\fontsize{8.0}{9.5}\selectfont\bfseries, rounded corners=1.5pt, inner sep=3.5pt]
    at (0.3, 6.4) {4. ASYMMETRIC ROOFLINE LATENCY ABSORPTION};
  \node[anchor=north east, text=SlateSteel, font=\fontsize{6.0}{7}\selectfont\bfseries]
    at (11.5, 6.3) {100\% Hidden Behind Shuttles};

  \begin{scope}[shift={(0.5, 0.6)}]
    \draw[fill=white, draw=SlateRim, line width=0.7pt, rounded corners=2pt] (0, 0) rectangle (10.8, 5.3);

    % Roofline Bars
    \node[anchor=north west, font=\fontsize{6.8}{8}\selectfont\bfseries, text=DarkSlate] at (0.3, 4.9) {Execution Latency Profile:};

    % 70B Generation Bar (Long)
    \draw[fill=TealL0, draw=MutedTeal, line width=0.8pt, rounded corners=1pt] (0.3, 3.2) rectangle (10.5, 4.2);
    \node[anchor=west, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=MutedTeal] at (0.5, 3.85) {70B Generator Decode: 128 tokens};
    \node[anchor=east, font=\fontsize{5.5}{6.5}\selectfont\ttfamily\bfseries, text=MutedTeal] at (10.3, 3.85) {$T_{\mathrm{decode}} = 2.68\text{ s}$};
    \node[anchor=west, font=\fontsize{4.8}{5.5}\selectfont\ttfamily, text=DarkSlate] at (0.5, 3.45) {Streams 70 GB weights across bus 128 times $\cdot$ Arithmetic Intensity $I = 2\text{ FLOP/B}$};

    % 8B Verifier Prefill Bar (Short, nested)
    \draw[fill=OchreL0, draw=WarmOchre, line width=0.8pt, rounded corners=1pt] (0.3, 1.8) rectangle (3.2, 2.7);
    \node[anchor=west, font=\fontsize{5.2}{6}\selectfont\bfseries, text=WarmOchre] at (0.4, 2.4) {8B Verifier Prefill};
    \node[anchor=west, font=\fontsize{5.0}{6}\selectfont\ttfamily\bfseries, text=WarmOchre] at (0.4, 2.05) {$T_{\mathrm{verif}} = 0.164\text{ s}$};

    % Headroom bracket
    \draw[draw=ForestSage, line width=1.0pt] (3.4, 2.0) -- (10.5, 2.0);
    \node[anchor=north, font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage] at (6.95, 1.85) {$2.52\text{ s}$ Headroom: Verifier is 100\% Hidden Behind Decode Memory Shuttles};

    % Takeaway box
    \draw[fill=SageL0, draw=ForestSage, line width=0.6pt, rounded corners=1pt] (0.3, 0.4) rectangle (10.5, 1.15);
    \node[font=\fontsize{5.5}{6.5}\selectfont\bfseries, text=ForestSage, align=center] at (5.4, 0.78) {Zero Critical-Path Latency Penalty: Interleaves compute-bound matrix mult into idle ALU cycles};

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

    tmp_pdf = "/tmp/speculative_verification_timeline.pdf"
    
    # Save PDF
    target_pdf = os.path.join(PDF_DIR, "speculative_verification_timeline.pdf")
    subprocess.run(["cp", tmp_pdf, target_pdf], check=True)

    # Render PNG via pdftoppm (250 DPI)
    png_prefix = "/tmp/spec_verif_render"
    subprocess.run(["pdftoppm", "-png", "-r", "250", tmp_pdf, png_prefix], check=True)
    rendered_png = f"{png_prefix}-1.png"

    # Save to both target PNG locations
    for p in ["speculative_verification_timeline.png", "fig-vol3-speculative-verification-timeline.png"]:
        t = os.path.join(PNG_DIR, p)
        subprocess.run(["cp", rendered_png, t], check=True)
        print(f"Saved PNG to {t}")

    # Render SVG via Inkscape
    inkscape_bin = "/opt/homebrew/bin/inkscape" if os.path.exists("/opt/homebrew/bin/inkscape") else "inkscape"
    for s in ["speculative_verification_timeline.svg", "fig-vol3-speculative-verification-timeline.svg"]:
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

    print("\nChapter 3 Speculative Verification figure successfully built and deployed!")

if __name__ == "__main__":
    main()
