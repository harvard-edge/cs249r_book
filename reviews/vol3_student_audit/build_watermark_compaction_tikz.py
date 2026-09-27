#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

TIKZ_CODE = r"""\documentclass[tikz,border=22pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc,positioning,arrows.meta,backgrounds,decorations.pathreplacing,shapes.geometric,fit}

% =============================================================================
% CANONICAL MLSYSBOOK PALETTE
% =============================================================================
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{slatesteel}{HTML}{334155}
\definecolor{GrayLine}{HTML}{64748B}
\definecolor{coolparchment}{HTML}{F8FAFC}
\definecolor{GridGray}{HTML}{E2E8F0}

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

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth,
  node distance=0.5cm
]

% -------------------------------------------------------------
% TIER 0: MASTER TITLE BANNER
% -------------------------------------------------------------
\node[anchor=north west, inner sep=0pt] (title) at (0, 21.8) {
  \begin{minipage}{25.0cm}
    {\fontsize{18}{22}\selectfont \textbf{\textcolor{DarkSlate}{Hysteretic Watermark Compaction \& Radix Cache Amortization Timeline}}}\\[4pt]
    {\fontsize{10.5}{14}\selectfont \textcolor{slatesteel}{Closed-loop working-set regulation: amortizing cold prefix-bust prefill penalties across high-frequency warm turns}}
  \end{minipage}
};

% -------------------------------------------------------------
% TIER 1: CONTEXT TOPOLOGY STRIP (Root, Trunk, Leaf)
% -------------------------------------------------------------
\node[anchor=north west, inner sep=0pt] (topo_title) at (0, 20.3) {
  {\fontsize{10}{13}\selectfont \textbf{\textcolor{DarkSlate}{Three-Zone Logical Context Hierarchy \& Physical Prefix-Cache Boundary}}}
};

% Root Zone
\draw[rounded corners=3pt, fill=coolparchment, draw=GrayLine, line width=0.8pt] 
  (0, 18.9) rectangle (4.5, 19.9);
\node[anchor=center] at (2.25, 19.55) {\fontsize{9}{11}\selectfont \textbf{\textcolor{DarkSlate}{Root Zone $[0, L_{\text{root}}]$}}};
\node[anchor=center] at (2.25, 19.18) {\fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{Tier 0 Supervisor $\cdot$ Pinned $\text{TTL}=\infty$}};

% Trunk Zone (Active Eviction Target)
\draw[rounded corners=3pt, fill=OrangeL0, draw=OrangeLine, line width=1.1pt] 
  (4.7, 18.9) rectangle (19.0, 19.9);
\node[anchor=center] at (11.85, 19.55) {\fontsize{9.5}{11.5}\selectfont \textbf{\textcolor{warmochre}{Trunk Zone $[L_{\text{root}}, L_{\text{leaf}}]$ --- Dynamic Compaction \& Digest Target}}};
\node[anchor=center] at (11.85, 19.18) {\fontsize{8}{10}\selectfont \textcolor{OrangeLine}{\textbf{Tier 1 Verified State $\cdot$ Invalidation rewrites prefix tree from $L_{\text{root}}$}}};

% Leaf Zone
\draw[rounded corners=3pt, fill=GreenL0, draw=GreenLine, line width=0.8pt] 
  (19.2, 18.9) rectangle (25.0, 19.9);
\node[anchor=center] at (22.1, 19.55) {\fontsize{9}{11}\selectfont \textbf{\textcolor{forestsage}{Leaf Zone $[L_{\text{leaf}}, M^*]$}}};
\node[anchor=center] at (22.1, 19.18) {\fontsize{7.5}{9.5}\selectfont \textcolor{forestsage}{Tier 2 Raw Obs $\cdot$ Rapid Turnover}};

% -------------------------------------------------------------
% TIER 2: PRIMARY PLOT - TOKEN ACCUMULATION SAWTOOTH CURVE
% -------------------------------------------------------------
\node[anchor=north west] (plot_title) at (0, 18.2) {
  {\fontsize{10.5}{13}\selectfont \textbf{\textcolor{DarkSlate}{Trajectory Token Footprint Sawtooth Curve across Hysteresis Band $[\tau_{\text{low}}, \tau_{\text{high}}]$}}}
};

% Coordinate frame for sawtooth plot
% X: 0 to 20 turns -> mapped to x = 1.8 to 24.8 cm (scale: 1 turn = 1.15 cm)
% Y: 0 to 35k tokens -> mapped to y = 9.8 to 16.8 cm (scale: 5k tokens = 1.0 cm)

\begin{scope}[shift={(1.8, 10.0)}]
  % Background Shading: Lost-in-the-Middle Danger Zone (above M* = 32k)
  \fill[fill=RedL0, opacity=0.85] (0, 6.4) rectangle (23.0, 7.1);
  \node[anchor=east] at (22.8, 6.75) {\fontsize{8}{10}\selectfont \textcolor{deepburgundy}{\textbf{Phase III: Lost-in-the-Middle Saturation ($M > M^*$, Softmax Dilution $33\times$, Quadratic Attention FLOPs)}}};

  % Background Shading: Safe Hysteretic Operating Window
  \fill[fill=BlueL0, opacity=0.55] (0, 0) rectangle (23.0, 5.12);

  % Grid Lines
  \foreach \y/\tok in {1.0/{5\text{k}}, 2.0/{10\text{k}}, 3.2/{16\text{k}}, 4.0/{20\text{k}}, 5.12/{25.6\text{k}}, 6.4/{32\text{k}}} {
    \draw[GridGray, line width=0.5pt] (0, \y) -- (23.0, \y);
    \node[anchor=east] at (-0.15, \y) {\fontsize{8}{10}\selectfont \textcolor{slatesteel}{\tok}};
  }

  % Axes
  \draw[->, line width=1.0pt, DarkSlate] (0,0) -- (23.4, 0);
  \draw[->, line width=1.0pt, DarkSlate] (0,0) -- (0, 7.3);

  % Rotated Y-axis title placed cleanly to the left
  \node[anchor=south, rotate=90] at (-1.1, 3.6) {
    \fontsize{9}{11}\selectfont \textbf{\textcolor{DarkSlate}{Active Context Tokens ($M$)}}
  };

  % Centered X-axis title
  \node[anchor=north] at (11.5, -0.60) {
    \fontsize{9.5}{12}\selectfont \textbf{\textcolor{DarkSlate}{Trajectory Execution Turn $t$}}
  };

  % X-axis ticks (Turns 0 to 20)
  \foreach \t in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} {
    \pgfmathsetmacro{\xpos}{\t * 1.15}
    \draw[DarkSlate, line width=0.7pt] (\xpos, 0) -- (\xpos, -0.12);
    \node[anchor=north] at (\xpos, -0.15) {\fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{\t}};
  }

  % Threshold Lines:
  % tau_high = 25.6k -> y = 5.12
  \draw[OrangeLine, line width=1.3pt, dashed] (0, 5.12) -- (23.0, 5.12);
  \node[anchor=south west, fill=white, inner sep=2.5pt, rounded corners=2pt, draw=OrangeLine, line width=0.6pt, align=left] at (0.2, 5.20) {
    \fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{warmochre}{High Watermark $\tau_{\text{high}} = 25.6\text{k}$}}\\[1pt]
    \fontsize{7}{8.5}\selectfont \textcolor{warmochre}{Trigger Threshold ($0.80 M^*$)}
  };

  % tau_low = 16.0k -> y = 3.20 (Compact 2-line badge placed underneath dashed line)
  \draw[BlueLine, line width=1.3pt, dashed] (0, 3.20) -- (23.0, 3.20);
  \node[anchor=north west, fill=white, inner sep=2.5pt, rounded corners=2pt, draw=BlueLine, line width=0.6pt, align=left] at (0.2, 3.08) {
    \fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{BlueLine}{Low Watermark $\tau_{\text{low}} = 16.0\text{k}$}}\\[1pt]
    \fontsize{7}{8.5}\selectfont \textcolor{BlueLine}{Compaction Target ($0.50 M^*$)}
  };

  % M* frontier = 32k -> y = 6.40
  \draw[RedLine, line width=1.4pt, dash pattern=on 4pt off 3pt] (0, 6.40) -- (23.0, 6.40);
  \node[anchor=south west, fill=white, inner sep=2.5pt, rounded corners=2pt, draw=RedLine, line width=0.6pt] at (0.2, 6.48) {
    \fontsize{8}{10}\selectfont \textcolor{deepburgundy}{\textbf{Effective Context Frontier $M^* = 32\text{k}$ (Attentional Cliff)}}
  };

  % Hysteresis Bracket on Right Side
  \draw[decorate, decoration={brace, amplitude=6pt, mirror}, line width=1.0pt, DarkSlate] 
    (23.1, 3.20) -- (23.1, 5.12);
  \node[anchor=west, align=left] at (23.4, 4.16) {
    \fontsize{8}{10}\selectfont \textbf{\textcolor{DarkSlate}{Hysteresis Band $\Delta \tau = 9.6\text{k}$}}\\[2pt]
    \fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{$\Delta T = \frac{\tau_{\text{high}} - \tau_{\text{low}}}{\Delta L_{\text{turn}}} = \frac{9.6\text{k}}{0.96\text{k}} \approx 10\text{ turns}$}
  };

  % The Sawtooth Token Curve:
  % Cycle 1: Turn 0 to 10
  \draw[BlueLine, line width=2.2pt] 
    (0, 0.8) -- 
    (1*1.15, 1.23) -- 
    (2*1.15, 1.66) -- 
    (3*1.15, 2.10) -- 
    (4*1.15, 2.53) -- 
    (5*1.15, 2.96) -- 
    (6*1.15, 3.39) -- 
    (7*1.15, 3.82) -- 
    (8*1.15, 4.25) -- 
    (9*1.15, 4.69) -- 
    (10*1.15, 5.12);

  % Shaded fill under cycle 1
  \fill[fill=BlueL, opacity=0.35]
    (0, 0) -- (0, 0.8) -- 
    (1*1.15, 1.23) -- (2*1.15, 1.66) -- (3*1.15, 2.10) -- (4*1.15, 2.53) -- 
    (5*1.15, 2.96) -- (6*1.15, 3.39) -- (7*1.15, 3.82) -- (8*1.15, 4.25) -- 
    (9*1.15, 4.69) -- (10*1.15, 5.12) -- (10*1.15, 0) -- cycle;

  % Compaction 1 Drop at Turn 10
  \draw[->, line width=2.0pt, deepburgundy, dashed] (10*1.15, 5.12) -- (10*1.15, 3.20);
  
  % Badge at Turn 10
  \node[anchor=south, fill=OrangeL0, draw=OrangeLine, line width=0.8pt, rounded corners=3pt, inner sep=3.5pt] 
    at (10*1.15, 5.35) {
    \fontsize{7.5}{9.5}\selectfont \textcolor{warmochre}{\textbf{Turn 10: Compaction $\mathcal{V}(O_t) \subseteq O_t'$}}
  };

  % Cycle 2: Turn 10 to 20
  \draw[BlueLine, line width=2.2pt]
    (10*1.15, 3.20) -- 
    (11*1.15, 3.39) -- 
    (12*1.15, 3.58) -- 
    (13*1.15, 3.78) -- 
    (14*1.15, 3.97) -- 
    (15*1.15, 4.16) -- 
    (16*1.15, 4.35) -- 
    (17*1.15, 4.54) -- 
    (18*1.15, 4.74) -- 
    (19*1.15, 4.93) -- 
    (20*1.15, 5.12);

  % Shaded fill under cycle 2
  \fill[fill=BlueL, opacity=0.35]
    (10*1.15, 0) -- (10*1.15, 3.20) -- 
    (11*1.15, 3.39) -- (12*1.15, 3.58) -- (13*1.15, 3.78) -- (14*1.15, 3.97) -- 
    (15*1.15, 4.16) -- (16*1.15, 4.35) -- (17*1.15, 4.54) -- (18*1.15, 4.74) -- 
    (19*1.15, 4.93) -- (20*1.15, 5.12) -- (20*1.15, 0) -- cycle;

  % Compaction 2 Drop at Turn 20
  \draw[->, line width=2.0pt, deepburgundy, dashed] (20*1.15, 5.12) -- (20*1.15, 3.20);
  \node[anchor=south, fill=OrangeL0, draw=OrangeLine, line width=0.8pt, rounded corners=3pt, inner sep=3.5pt] 
    at (20*1.15, 5.35) {
    \fontsize{7.5}{9.5}\selectfont \textcolor{warmochre}{\textbf{Turn 20: Compaction}}
  };

  % Cadence Indicators below curve
  \draw[<->, line width=1.1pt, GreenLine] (0, 0.4) -- (10*1.15, 0.4) 
    node[midway, fill=white, inner sep=2pt] {\fontsize{8}{10}\selectfont \textbf{\textcolor{forestsage}{Cadence Cycle 1: $\Delta T = 10$ turns}}};
  \draw[<->, line width=1.1pt, GreenLine] (10*1.15, 0.4) -- (20*1.15, 0.4) 
    node[midway, fill=white, inner sep=2pt] {\fontsize{8}{10}\selectfont \textbf{\textcolor{forestsage}{Cadence Cycle 2: $\Delta T = 10$ turns}}};

\end{scope}

% -------------------------------------------------------------
% TIER 3: TURN-BY-TURN GPU PREFILL COST & RADIX HIT PROFILE
% -------------------------------------------------------------
\node[anchor=north west] (latency_title) at (0, 8.8) {
  {\fontsize{10.5}{13}\selectfont \textbf{\textcolor{DarkSlate}{GPU Serving Engine Prefill Latency ($T_{\text{prefill}}$) \& Radix Tree Cache Efficiency}}}
};

\begin{scope}[shift={(1.8, 5.0)}]
  % Axes
  \draw[->, line width=1.0pt, DarkSlate] (0,0) -- (23.4, 0);
  \draw[->, line width=1.0pt, DarkSlate] (0,0) -- (0, 3.4);

  % Rotated Y-axis title placed cleanly to the left
  \node[anchor=south, rotate=90] at (-1.1, 1.7) {
    \fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{DarkSlate}{Prefill Latency ($T_{\text{prefill}}$)}}
  };

  % Centered X-axis title
  \node[anchor=north] at (11.5, -0.60) {
    \fontsize{9.5}{12}\selectfont \textbf{\textcolor{DarkSlate}{Trajectory Execution Turn $t$}}
  };

  % X-axis ticks (Turns 0 to 20)
  \foreach \t in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} {
    \pgfmathsetmacro{\bx}{\t * 1.15}
    \draw[DarkSlate, line width=0.7pt] (\bx, 0) -- (\bx, -0.12);
    \node[anchor=north] at (\bx, -0.15) {\fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{\t}};
  }

  % Y-axis labels and gridlines
  \node[anchor=east] at (-0.15, 0.3) {\fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{$8.9\text{ ms}$}};
  \node[anchor=east] at (-0.15, 2.6) {\fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{$1.42\text{ s}$}};
  \draw[GridGray, line width=0.5pt] (0, 0.3) -- (23.0, 0.3);
  \draw[GridGray, line width=0.5pt] (0, 2.6) -- (23.0, 2.6);

  % Bars for Turns 1 to 20:
  % Scale: 8.9 ms -> height = 0.3 cm
  % 1.42 s (prefix bust) -> height = 2.6 cm
  
  % Turns 1 to 9: Warm turns (Green bars)
  \foreach \t in {1, 2, 3, 4, 5, 6, 7, 8, 9} {
    \pgfmathsetmacro{\bx}{(\t * 1.15) - 0.25}
    \draw[GreenLine, fill=GreenL, line width=0.7pt] 
      (\bx, 0) rectangle ++(0.5, 0.3);
  }

  % Arched callout over Turns 1 to 9
  \draw[decorate, decoration={brace, amplitude=5pt}, line width=0.9pt, forestsage] 
    (0.8, 0.45) -- (10.0, 0.45);
  \node[anchor=south] at (5.4, 0.65) {
    \fontsize{8}{10}\selectfont \textcolor{forestsage}{\textbf{9 Warm Turns: $\eta_{\text{cache}} \ge 96\% \to 100\%$ Radix Hit ($8.9\text{ ms}$)}}
  };

  % Turn 10: Compaction turn (CPU AST/Digest clearing < 1 ms)
  \pgfmathsetmacro{\bx}{(10 * 1.15) - 0.25}
  \draw[OrangeLine, fill=OrangeL, line width=0.7pt] (\bx, 0) rectangle ++(0.5, 0.15);
  \node[anchor=south] at (\bx+0.25, 0.20) {\fontsize{6.5}{8}\selectfont \textcolor{warmochre}{\textbf{AST}}};

  % Turn 11: PREFIX BUST COLD PREFILL (Red bar)
  \pgfmathsetmacro{\bx}{(11 * 1.15) - 0.35}
  \draw[deepburgundy, fill=RedL, line width=1.1pt] 
    (\bx, 0) rectangle ++(0.7, 2.6);
  \node[anchor=south, fill=RedL0, draw=RedLine, inner sep=2.5pt, rounded corners=2pt] at (\bx+0.35, 2.70) {
    \fontsize{7.5}{9.5}\selectfont \textcolor{deepburgundy}{\textbf{Prefix Bust: $1.42\text{ s}$}}
  };
  \node[anchor=center] at (\bx+0.35, 1.3) {
    \rotatebox{90}{\fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{deepburgundy}{Cold GEMM Prefill}}}
  };

  % Turns 12 to 19: Warm turns (Green bars)
  \foreach \t in {12, 13, 14, 15, 16, 17, 18, 19} {
    \pgfmathsetmacro{\bx}{(\t * 1.15) - 0.25}
    \draw[GreenLine, fill=GreenL, line width=0.7pt] 
      (\bx, 0) rectangle ++(0.5, 0.3);
  }

  % Arched callout over Turns 12 to 19
  \draw[decorate, decoration={brace, amplitude=5pt}, line width=0.9pt, forestsage] 
    (13.4, 0.45) -- (22.2, 0.45);
  \node[anchor=south] at (17.8, 0.65) {
    \fontsize{8}{10}\selectfont \textcolor{forestsage}{\textbf{8 Warm Turns Re-established: $100\%$ Radix Hit ($8.9\text{ ms}$)}}
  };

  % Turn 20: Compaction turn
  \pgfmathsetmacro{\bx}{(20 * 1.15) - 0.25}
  \draw[OrangeLine, fill=OrangeL, line width=0.7pt] (\bx, 0) rectangle ++(0.5, 0.15);
  \node[anchor=south] at (\bx+0.25, 0.20) {\fontsize{6.5}{8}\selectfont \textcolor{warmochre}{\textbf{AST}}};

  % Amortized Summary Box on the right
  \node[anchor=north east, fill=white, draw=DarkSlate, line width=0.8pt, rounded corners=4pt, inner sep=5pt] 
    at (22.6, 3.20) {
    \begin{minipage}{7.5cm}
      {\fontsize{8.5}{11}\selectfont \textbf{\textcolor{DarkSlate}{Amortized Cost per Turn across $\Delta T = 10$:}}}\\[3pt]
      {\fontsize{8}{10.5}\selectfont
      $\bar{C}_{\text{turn}} = \frac{1.42\text{ s} + 9 \times 0.0089\text{ s}}{10} \approx \mathbf{0.150\text{ s/turn}}$\\[3pt]
      \textcolor{forestsage}{\textbf{Latency Reduction: $89.4\%$ vs. Cold Prefilling}}
      }
    \end{minipage}
  };

\end{scope}

% -------------------------------------------------------------
% TIER 4: ARCHITECTURAL REGIMES & TRADE-OFF CARDS
% -------------------------------------------------------------
\node[anchor=north west] (regimes_title) at (0, 4.0) {
  {\fontsize{10.5}{13}\selectfont \textbf{\textcolor{DarkSlate}{Architectural Operating Regimes: The Cost of Misconfigured Hysteresis}}}
};

% Card A: Narrow Hysteresis (Thrashing)
\node[anchor=north west, fill=RedL0, draw=RedLine, line width=0.9pt, rounded corners=4pt, inner sep=6pt] 
  (cardA) at (0, 3.5) {
  \begin{minipage}{7.7cm}
    {\fontsize{9.5}{12}\selectfont \textbf{\textcolor{deepburgundy}{Regime I: Narrow Hysteresis ($\Delta \tau \to 0$)}}}\\[3pt]
    {\fontsize{8}{10.5}\selectfont \textbf{Configuration:} $\tau_{\text{high}} = 0.80 M^*, \tau_{\text{low}} = 0.75 M^*$ ($\Delta \tau = 0.05 M^*$)\\[3pt]
    \textbf{Cadence:} $\Delta T = \frac{1{,}600}{1{,}600} = 1\text{ turn}$ (Compacts every turn!)\\[3pt]
    \textbf{Failure Mode:} \textcolor{deepburgundy}{\textbf{Compaction Thrashing \& Queue Livelock}}
    \begin{itemize}\setlength{\itemsep}{1pt}\setlength{\leftmargin}{10pt}
      \item Rewrites Trunk on every invocation: \textbf{0\% cache hits}.
      \item Incurs $1.42\text{ s}$ cold prefill stall continuously.
      \item Severe serving engine tail latency and worker starvation.
    \end{itemize}
    }
  \end{minipage}
};

% Card B: Optimal Wide Hysteresis (Recommended)
\node[anchor=north west, fill=GreenL0, draw=GreenLine, line width=1.1pt, rounded corners=4pt, inner sep=6pt] 
  (cardB) at (8.3, 3.5) {
  \begin{minipage}{7.7cm}
    {\fontsize{9.5}{12}\selectfont \textbf{\textcolor{forestsage}{Regime II: Balanced Hysteresis (Optimal)}}}\\[3pt]
    {\fontsize{8}{10.5}\selectfont \textbf{Configuration:} $\tau_{\text{high}} \approx 0.80 M^*, \tau_{\text{low}} \approx 0.50 M^*$ ($\Delta \tau = 0.30 M^*$)\\[3pt]
    \textbf{Cadence:} $\Delta T \approx 10\text{ turns}$ ($\Delta L_{\text{turn}} = 0.96\text{k}$ tokens)\\[3pt]
    \textbf{Outcome:} \textcolor{forestsage}{\textbf{Amortized Efficiency \& High Attention Fidelity}}
    \begin{itemize}\setlength{\itemsep}{1pt}\setlength{\leftmargin}{10pt}
      \item 9 warm turns ($100\%$ Radix hits) amortize 1 cold prefill.
      \item Keeps sequence strictly in high-salience zone ($M \le 0.80 M^*$).
      \item Prefill latency drops from $1.42\text{ s} \to 0.150\text{ s}$ amortized.
    \end{itemize}
    }
  \end{minipage}
};

% Card C: Unbounded Expansion (Attention Saturation)
\node[anchor=north west, fill=coolparchment, draw=GrayLine, line width=0.9pt, rounded corners=4pt, inner sep=6pt] 
  (cardC) at (16.6, 3.5) {
  \begin{minipage}{7.7cm}
    {\fontsize{9.5}{12}\selectfont \textbf{\textcolor{DarkSlate}{Regime III: Unbounded Growth ($\tau_{\text{high}} \to S_{\max}$)}}}\\[3pt]
    {\fontsize{8}{10.5}\selectfont \textbf{Configuration:} No proactive compaction ($M \to 128\text{k}$)\\[3pt]
    \textbf{Cadence:} $\Delta T = \infty$ (Deferred until hard context crash)\\[3pt]
    \textbf{Failure Mode:} \textcolor{deepburgundy}{\textbf{Softmax Dilution \& Quadratic Compute}}
    \begin{itemize}\setlength{\itemsep}{1pt}\setlength{\leftmargin}{10pt}
      \item Normalizer dilutes signal: target mass drops $96.8\% \to 2.9\%$.
      \item Lost-in-the-middle amnesia causes plan hallucinations.
      \item Attention FLOPs ($4 M^2 L d$) consume $>72\%$ of prefill compute.
    \end{itemize}
    }
  \end{minipage}
};

\end{tikzpicture}
\end{document}
"""

repo_root = Path(__file__).resolve().parents[2]
tex_file = repo_root / "books/vol3/tex/hysteretic_watermark_compaction.tex"

with open(tex_file, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-halt-on-error", "-output-directory=/tmp", str(tex_file)], capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/hysteretic_watermark_compaction.pdf", "/tmp/watermark_compaction_render"])
    subprocess.run(["pdftocairo", "-svg", "/tmp/hysteretic_watermark_compaction.pdf", "/tmp/hysteretic_watermark_compaction.svg"])
    print("Successfully compiled and rendered Hysteretic Watermark Compaction Diagram!")
else:
    print(res.stdout[-1500:])
