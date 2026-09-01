#!/usr/bin/env python3
"""
Generate Chapter 2 figures for Physical AI Systems:
- fig02_latency_waterfall (The 7-Stage Sense-to-Actuation Latency Waterfall)
- fig02_stopping_distance (Dynamic Stopping Distance Physics: Reaction + Braking)
- fig02_metrology_setup (Hardware-Triggered Logic Analyzer GPIO Instrumentation)

Outputs both vector PDF and native SVG (via pdftocairo), plus PNG for visual inspection.
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CH02_FIG_DIR = os.path.join(BOOK_DIR, "chapters", "02-constraints", "figures")


# -----------------------------------------------------------------------------
# 1. FIG 02.1: SENSE-TO-ACTUATION LATENCY WATERFALL (P50 vs P99 TAILS)
# -----------------------------------------------------------------------------
WATERFALL_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Harvard Crimson & ETH Zurich Academic Semantic Palette
\definecolor{harvardcrimson}{HTML}{A51C30}
\definecolor{ethdarkblue}{HTML}{1F407A}
\definecolor{ethblue}{HTML}{215CAF}
\definecolor{ethpetrol}{HTML}{007A87}
\definecolor{ethbronze}{HTML}{B87333}
\definecolor{ethpurple}{HTML}{5B4B8A}
\definecolor{ethslate}{HTML}{475569}
\definecolor{cardbg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\definecolor{safeTeal}{HTML}{10B981}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  stagebox/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=4pt,
    line width=0.8pt,
    text width=1.12in,
    minimum height=1.65in,
    inner sep=6pt,
    align=center,
    anchor=north
  }
]

  % Top Title Banner
  \node[draw=ethdarkblue, fill=ethdarkblue!5, rounded corners=5pt, line width=1pt, text width=8.70in, inner sep=7pt, align=center] (title) at (4.35in, 0) {
    {\normalsize\bfseries\color{ethdarkblue}\faIcon{clock}\;\; THE 7-STAGE SENSE-TO-ACTUATION LATENCY WATERFALL}\\[2pt]
    {\scriptsize\color{ethslate}Where Does the Time Go? Deconstructing the Physical, Memory, Neural, and Silicon Bottlenecks}
  };

  % 7 Stages Across the Pipeline
  \node[stagebox, draw=ethblue!70] (st1) at (0, -0.65in) {
    {\scriptsize\bfseries\color{ethblue}\faIcon{satellite-dish}\; STAGE 1}\\[3pt]
    {\scriptsize\bfseries Transduce}\\[4pt]
    {\tiny\color{ethslate}Photodiode Well Exposure \& IMU Deflection}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $8\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $15\text{ ms}$}}
  };

  \node[stagebox, draw=ethpetrol!70] (st2) at (1.26in, -0.65in) {
    {\scriptsize\bfseries\color{ethpetrol}\faIcon{microchip}\; STAGE 2}\\[3pt]
    {\scriptsize\bfseries DMA Ingest}\\[4pt]
    {\tiny\color{ethslate}MIPI CSI-2 Bus Transfer into Shared DRAM}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $3\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $8\text{ ms}$}}
  };

  \node[stagebox, draw=ethblue!80] (st3) at (2.52in, -0.65in) {
    {\scriptsize\bfseries\color{ethblue}\faIcon{eye}\; STAGE 3}\\[3pt]
    {\scriptsize\bfseries Perception}\\[4pt]
    {\tiny\color{ethslate}ViT Encoders \& DINOv2\\[1pt]3D Spatial Tokens}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $12\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $25\text{ ms}$}}
  };

  \node[stagebox, draw=ethbronze!90] (st4) at (3.78in, -0.65in) {
    {\scriptsize\bfseries\color{ethbronze}\faIcon{brain}\; STAGE 4}\\[3pt]
    {\scriptsize\bfseries Policy VLA}\\[4pt]
    {\tiny\color{ethslate}Diffusion ACT / VLA\\[1pt]Action Chunk Pass}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $22\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $60\text{ ms}$}}
  };

  \node[stagebox, draw=ethpurple!80] (st5) at (5.04in, -0.65in) {
    {\scriptsize\bfseries\color{ethpurple}\faIcon{network-wired}\; STAGE 5}\\[3pt]
    {\scriptsize\bfseries Inter-IPC}\\[4pt]
    {\tiny\color{ethslate}Shared SRAM Mailbox\\[1pt]RPMSG MPU-to-MCU}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $0.8\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $2.0\text{ ms}$}}
  };

  \node[stagebox, draw=ethpetrol!90] (st6) at (6.30in, -0.65in) {
    {\scriptsize\bfseries\color{ethpetrol}\faIcon{shield-alt}\; STAGE 6}\\[3pt]
    {\scriptsize\bfseries Enforce}\\[4pt]
    {\tiny\color{ethslate}1 kHz CBF Filter \&\\[1pt]Stopping Distance}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $0.4\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $1.0\text{ ms}$}}
  };

  \node[stagebox, draw=harvardcrimson!80] (st7) at (7.56in, -0.65in) {
    {\scriptsize\bfseries\color{harvardcrimson}\faIcon{cogs}\; STAGE 7}\\[3pt]
    {\scriptsize\bfseries Actuator}\\[4pt]
    {\tiny\color{ethslate}Motor Stator $L/R$ Coil\\[1pt]Current Rise to Torque}\\[6pt]
    \rule{0.9\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{safeTeal}$P_{50}$: $6\text{ ms}$}}\\[1pt]
    {\scriptsize\textbf{\color{harvardcrimson}$P_{99}$: $15\text{ ms}$}}
  };

  % Timeline Comparisons at Bottom with Generous Vertical Clearance
  % NOMINAL P50 BAR
  \node[anchor=west, font=\sffamily\bfseries\scriptsize, text=safeTeal] at (0, -2.90in) {\faIcon{check-circle}\; Nominal Path ($P_{50} = 52.2\text{ ms}$):};
  \draw[fill=safeTeal!20, draw=safeTeal, line width=1pt, rounded corners=3pt] (2.40in, -3.00in) rectangle ++(3.13in, 0.22in);
  \node[font=\sffamily\bfseries\tiny, text=safeTeal!90] at (3.96in, -2.89in) {Safe Closed-Loop Margin ($52.2\text{ ms} \ll \tau_{\text{world}}$)};

  % TAIL P99 BAR
  \node[anchor=west, font=\sffamily\bfseries\scriptsize, text=harvardcrimson] at (0, -3.25in) {\faIcon{exclamation-triangle}\; Tail Path ($P_{99} = 126.0\text{ ms}$):};
  \draw[fill=harvardcrimson!20, draw=harvardcrimson, line width=1pt, rounded corners=3pt] (2.40in, -3.35in) rectangle ++(6.30in, 0.22in);
  \node[font=\sffamily\bfseries\tiny, text=harvardcrimson] at (5.55in, -3.24in) {CRITICAL TIMING VIOLATION ($126.0\text{ ms} > \tau_{\text{world}}$)};

  % World Deadline Vertical Marker (Safely Positioned Below Stage 7)
  \draw[dashed, line width=1.3pt, draw=harvardcrimson!90] (7.40in, -2.68in) -- (7.40in, -3.50in);
  \node[font=\sffamily\bfseries\tiny, fill=white, draw=harvardcrimson, rounded corners=2pt, inner sep=2pt, text=harvardcrimson] at (7.40in, -2.60in) {World Deadline $\tau_{\text{world}} = 100\text{ ms}$};

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 2. FIG 02.2: DYNAMIC STOPPING DISTANCE PHYSICS (REACTION + BRAKING)
# -----------------------------------------------------------------------------
STOPPING_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Harvard Crimson & ETH Zurich Academic Semantic Palette
\definecolor{harvardcrimson}{HTML}{A51C30}
\definecolor{ethdarkblue}{HTML}{1F407A}
\definecolor{ethblue}{HTML}{215CAF}
\definecolor{ethpetrol}{HTML}{007A87}
\definecolor{ethbronze}{HTML}{B87333}
\definecolor{ethslate}{HTML}{475569}
\definecolor{cardbg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\definecolor{safeTeal}{HTML}{10B981}
\definecolor{amberWarn}{HTML}{D97706}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  scale=1.0
]

  % ---------------------------------------------------------------------------
  % TOP HEADER CARD
  % ---------------------------------------------------------------------------
  \node[draw=ethdarkblue, fill=ethdarkblue!5, rounded corners=6pt, line width=1.1pt, text width=9.20in, inner sep=8pt, align=center] (title) at (4.60in, 0) {
    {\normalsize\bfseries\color{ethdarkblue}\faIcon{ruler-horizontal}\;\; DYNAMIC STOPPING DISTANCE: SENSE-TO-ACTUATION DELAY AS PHYSICAL DISPLACEMENT}\\[3pt]
    {\scriptsize\color{ethslate}Translating Milliseconds of Computational Latency into Physical Centimeters of Unguided Collision Hazard}
  };

  % Equation Box
  \node[draw=cardborder, fill=cardbg, rounded corners=5pt, line width=0.9pt, text width=9.20in, inner sep=7pt, align=center, below=0.12in of title] (eq) {
    {\small $\displaystyle d_{\text{stop}}(v_0, \Delta t_{\text{delay}}) \;=\; \underbrace{v_0 \cdot \Delta t_{\text{delay}}}_{\substack{\text{\textbf{\color{ethblue}Phase 1: Linear Reaction Travel}}\\\text{\scriptsize\color{ethslate}(Coasting during computational delay)}}} \;+\; \underbrace{\frac{v_0^2}{2 a_{\text{max}}}}_{\substack{\text{\textbf{\color{ethbronze}Phase 2: Quadratic Braking Distance}}\\\text{\scriptsize\color{ethslate}(Mechanical deceleration to rest)}}}$}
  };

  % ---------------------------------------------------------------------------
  % COORDINATE MAPPING: 1 meter = 16.0 inches (1 cm = 0.16 in)
  % Origin at 0.60in from left, spanning to 0.50m (8.00in)
  % ---------------------------------------------------------------------------
  \coordinate (X0) at (0.60in, -1.50in);
  
  % ---------------------------------------------------------------------------
  % ROW 1: NOMINAL EXECUTION (P50 = 30 ms)
  % ---------------------------------------------------------------------------
  \node[anchor=west, font=\sffamily\bfseries\scriptsize, text=ethdarkblue] at (0.20in, -1.45in) {
    \faIcon{check-circle}\;\; \textbf{Nominal Case ($P_{50}$):} \textnormal{$v_0 = 1.0\text{ m/s},\, \Delta t = 30\text{ ms},\, a_{\text{max}} = 2.0\text{ m/s}^2$}
  };

  % Phase 1: Reaction (3 cm = 0.48 in)
  \draw[fill=ethblue!25, draw=ethblue, line width=1.1pt, rounded corners=3pt] (0.60in, -1.95in) rectangle ++(0.48in, 0.36in)
    node[midway, font=\sffamily\bfseries\tiny, text=ethdarkblue] {$3\text{ cm}$};

  % Phase 2: Braking (25 cm = 4.00 in)
  \draw[fill=ethbronze!25, draw=ethbronze, line width=1.1pt, rounded corners=3pt] ($(0.60in, -1.95in) + (0.48in, 0)$) rectangle ++(4.00in, 0.36in)
    node[midway, font=\sffamily\bfseries\tiny, text=ethbronze] {Controlled Braking: $\frac{v_0^2}{2a_{\text{max}}} = 25\text{ cm}$};

  % Safe Clearance Margin (7 cm = 1.12 in)
  \draw[fill=safeTeal!15, draw=safeTeal, line width=1.0pt, dashed, rounded corners=3pt] ($(0.60in, -1.95in) + (4.48in, 0)$) rectangle ++(1.12in, 0.36in)
    node[midway, font=\sffamily\bfseries\tiny, text=safeTeal] {Margin: $+7\text{ cm}$};

  % Total Stop Tag (Nominal)
  \node[draw=safeTeal, fill=white, rounded corners=3pt, font=\sffamily\bfseries\tiny, text=safeTeal, inner sep=3.5pt, anchor=west] at ($(0.60in, -1.95in) + (5.75in, 0.18in)$) {
    \faIcon{shield-alt}\; Total $d_{\text{stop}} = 28\text{ cm}$ (\textbf{SAFE: $< 35\text{ cm}$ Clearance})
  };

  % ---------------------------------------------------------------------------
  % ROW 2: TAIL LATENCY SPIKE (P99 = 230 ms)
  % ---------------------------------------------------------------------------
  \node[anchor=west, font=\sffamily\bfseries\scriptsize, text=harvardcrimson] at (0.20in, -2.55in) {
    \faIcon{exclamation-triangle}\;\; \textbf{Tail Latency Spike ($P_{99}$):} \textnormal{$v_0 = 1.0\text{ m/s},\, \Delta t = 230\text{ ms},\, a_{\text{max}} = 2.0\text{ m/s}^2$}
  };

  % Phase 1: Expanded Reaction (23 cm = 3.68 in)
  \draw[fill=harvardcrimson!25, draw=harvardcrimson, line width=1.1pt, rounded corners=3pt] (0.60in, -3.05in) rectangle ++(3.68in, 0.36in)
    node[midway, font=\sffamily\bfseries\tiny, text=harvardcrimson] {$d_{\text{react}} = 23\text{ cm}$ ($+20\text{ cm}$ Blind Coasting Travel!)};

  % Phase 2: Braking (25 cm = 4.00 in) - Place label on the right side of the barrier line
  \draw[fill=ethbronze!25, draw=ethbronze, line width=1.1pt, rounded corners=3pt] ($(0.60in, -3.05in) + (3.68in, 0)$) rectangle ++(4.00in, 0.36in);
  \node[font=\sffamily\bfseries\tiny, text=ethbronze] at ($(0.60in, -3.05in) + (3.68in + 2.85in, 0.18in)$) {Braking: $25\text{ cm}$};

  % Crash / Impact Zone Tag
  \node[draw=harvardcrimson, fill=harvardcrimson!15, rounded corners=3pt, font=\sffamily\bfseries\tiny, text=harvardcrimson, inner sep=3.5pt, anchor=west] at ($(0.60in, -3.05in) + (7.80in, 0.18in)$) {
    \faIcon{skull-crossbones}\; $d_{\text{stop}} = 48\text{ cm}$ (\textbf{IMPACT: $+13\text{ cm}$ Breach})
  };

  % ---------------------------------------------------------------------------
  % PHYSICAL OBSTACLE BARRIER (At x = 35 cm = 0.60in + 5.60in = 6.20in)
  % ---------------------------------------------------------------------------
  \draw[line width=1.8pt, draw=harvardcrimson, dashed] (6.20in, -1.35in) -- (6.20in, -3.30in);
  \node[draw=harvardcrimson, fill=white, rounded corners=3pt, font=\sffamily\bfseries\tiny, text=harvardcrimson, inner sep=3pt] at (6.20in, -1.25in) {
    \faIcon{hand-paper}\; Physical Clearance Barrier ($d_{\text{clearance}} = 35\text{ cm}$)
  };

  % ---------------------------------------------------------------------------
  % SPATIAL DISTANCE AXIS (Bottom)
  % ---------------------------------------------------------------------------
  \draw[->, line width=1.1pt, ethslate] (0.60in, -3.45in) -- (8.90in, -3.45in)
    node[right, font=\sffamily\bfseries\tiny, text=ethslate] {Distance $x$};

  % Major Ticks: 0, 10, 20, 30, 35, 40, 50 cm
  \foreach \x/\label in {0/0\text{ cm}, 10/10\text{ cm}, 20/20\text{ cm}, 30/30\text{ cm}, 35/35\text{ cm [Barrier]}, 40/40\text{ cm}, 50/50\text{ cm}} {
    \draw[line width=0.8pt, ethslate] ($(0.60in, -3.35in) + (\x*0.16in, 0.05in)$) -- ($(0.60in, -3.35in) + (\x*0.16in, -0.05in)$);
    \node[font=\sffamily\bfseries\tiny, text=ethslate, below=2pt] at ($(0.60in, -3.35in) + (\x*0.16in, -0.05in)$) {\label};
  }

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 3. FIG 02.3: HARDWARE-TRIGGERED METROLOGY TOPOLOGY
# -----------------------------------------------------------------------------
METROLOGY_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Harvard Crimson & ETH Zurich Academic Semantic Palette
\definecolor{harvardcrimson}{HTML}{A51C30}
\definecolor{ethdarkblue}{HTML}{1F407A}
\definecolor{ethblue}{HTML}{215CAF}
\definecolor{ethpetrol}{HTML}{007A87}
\definecolor{ethbronze}{HTML}{B87333}
\definecolor{ethslate}{HTML}{475569}
\definecolor{cardbg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  box/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=5pt,
    line width=0.9pt,
    text width=3.45in,
    minimum height=2.45in,
    inner sep=9pt,
    align=left,
    anchor=north,
    text=ethdarkblue
  }
]

  % Top Title Banner
  \node[draw=ethdarkblue, fill=ethdarkblue!5, rounded corners=5pt, line width=1pt, text width=8.50in, inner sep=7pt, align=center] (title) at (4.25in, 0) {
    {\normalsize\bfseries\color{ethdarkblue}\faIcon{microchip}\;\; HARDWARE-TRIGGERED METROLOGY TOPOLOGY}\\[2pt]
    {\scriptsize\color{ethslate}Escaping Software Timestamp Delusions via Logic Analyzer GPIO Toggles and Shunt Current Probes}
  };

  % Left Card: The System Under Test
  \node[box] (dut) at (0, -0.65in) {
    {\small\bfseries\color{ethdarkblue}\faIcon{server}\; Physical AI System Under Test (SUT)}\\[3pt]
    {\scriptsize\bfseries\color{ethslate}Arduino UNO Q Dual-Brain Platform}\\[6pt]
    \textbf{\color{ethblue}1. Sensor Frame Interrupt (GPIO 1):}\\[1pt]
    {\tiny Toggled inside camera driver ISR on DMA start.}\\[4pt]
    \textbf{\color{ethbronze}2. MPU Inference Emit (GPIO 2):}\\[1pt]
    {\tiny Toggled in Linux kernel upon writing RPMSG mailbox.}\\[4pt]
    \textbf{\color{ethpetrol}3. MCU Enforcement Veto / Pass (GPIO 3):}\\[1pt]
    {\tiny Toggled inside FreeRTOS 1 kHz ISR on CBF evaluation.}\\[4pt]
    \textbf{\color{harvardcrimson}4. Inverter Gate Drive Out (PWM Pins):}\\[1pt]
    {\tiny Direct hardware timer registers driving MOSFET bridge.}
  };

  % Right Card: External Test Equipment (Positioned with 1.30in gap)
  \node[box, draw=ethpetrol, fill=ethpetrol!5] (scope) at (4.75in, -0.65in) {
    {\small\bfseries\color{ethpetrol}\faIcon{chart-line}\; External Ground-Truth Metrology}\\[3pt]
    {\scriptsize\bfseries\color{ethslate}Multi-Channel Logic Analyzer \& Oscilloscope}\\[6pt]
    \textbf{\color{ethblue}CH 1 (Digital):} Transduction Pulse $\to$ True $t_{\text{transduce}}$\\[3pt]
    \textbf{\color{ethbronze}CH 2 (Digital):} Proposal Generation $\to$ True $t_{\text{inference}}$\\[3pt]
    \textbf{\color{ethpetrol}CH 3 (Digital):} MCU Permission $\to$ True $t_{\text{enforce}}$\\[3pt]
    \textbf{\color{harvardcrimson}CH 4 (Analog):} Current Shunt $\to$ Motor Coil $L/R$ Rise Time\\[6pt]
    \rule{0.95\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{ethdarkblue}Guaranteed Result: Zero Operating System Jitter.}}
  };

  % Connecting Probes with Clear Spacing
  \draw[->, line width=1.2pt, ethblue] ($(dut.north east) + (0, -0.45in)$) -- ($(scope.north west) + (0, -0.45in)$)
    node[midway, font=\sffamily\bfseries\tiny, fill=white, draw=ethblue!40, rounded corners=2pt, inner sep=2pt, text=ethblue] {Probe 1: Ingest};

  \draw[->, line width=1.2pt, ethbronze] ($(dut.north east) + (0, -0.95in)$) -- ($(scope.north west) + (0, -0.95in)$)
    node[midway, font=\sffamily\bfseries\tiny, fill=white, draw=ethbronze!40, rounded corners=2pt, inner sep=2pt, text=ethbronze] {Probe 2: Inference};

  \draw[->, line width=1.2pt, ethpetrol] ($(dut.north east) + (0, -1.45in)$) -- ($(scope.north west) + (0, -1.45in)$)
    node[midway, font=\sffamily\bfseries\tiny, fill=white, draw=ethpetrol!40, rounded corners=2pt, inner sep=2pt, text=ethpetrol] {Probe 3: Enforce};

  \draw[->, line width=1.2pt, harvardcrimson] ($(dut.north east) + (0, -1.95in)$) -- ($(scope.north west) + (0, -1.95in)$)
    node[midway, font=\sffamily\bfseries\tiny, fill=white, draw=harvardcrimson!40, rounded corners=2pt, inner sep=2pt, text=harvardcrimson] {Probe 4: Coil Current};

\end{tikzpicture}
\end{document}
'''

def build_all():
    figures = {
        "fig02_latency_waterfall.tex": WATERFALL_TEX,
        "fig02_stopping_distance.tex": STOPPING_TEX,
        "fig02_metrology_setup.tex": METROLOGY_TEX
    }
    
    for filename, tex in figures.items():
        tex_path = os.path.join(CH02_FIG_DIR, filename)
        pdf_name = filename.replace(".tex", ".pdf")
        svg_name = filename.replace(".tex", ".svg")
        png_name = filename.replace(".tex", "_preview")
        
        with open(tex_path, "w") as f:
            f.write(tex.strip() + "\n")
        print(f"Wrote {tex_path}")
        
        subprocess.run(["lualatex", "-interaction=nonstopmode", filename], cwd=CH02_FIG_DIR, check=True)
        subprocess.run(["pdftocairo", "-svg", pdf_name, svg_name], cwd=CH02_FIG_DIR, check=True)
        subprocess.run(["pdftoppm", "-png", "-r", "200", pdf_name, png_name], cwd=CH02_FIG_DIR, check=True)
        print(f"Compiled {pdf_name} -> {svg_name} and {png_name}-1.png")

if __name__ == "__main__":
    build_all()
