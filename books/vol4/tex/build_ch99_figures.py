#!/usr/bin/env python3
"""
Generate Chapter 99 (Capstone) figures for Physical AI Systems:
- fig99_dual_brain_integration (End-to-End Dual-Brain Agent Architecture & Multi-Rate Dataflow)
- fig99_seeded_fault_timeline (Cross-Layer Seeded Fault Injection, Invariant Containment, and Dynamic Arrest Waveform)
- fig99_defense_dossier_matrix (The Cumulative Design Dossier, 4-Expert Red-Team Evaluation, and 3-Way Release Verdict Gate)

Outputs vector PDF, native SVG (via pdftocairo), and PNG preview (via pdftoppm) for visual inspection.
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CH99_FIG_DIR = os.path.join(REPO_ROOT, "book", "chapters", "99-capstone", "figures")

# -----------------------------------------------------------------------------
# 1. FIG 99.1: END-TO-END DUAL-BRAIN INTEGRATION & DATAFLOW
# -----------------------------------------------------------------------------
DUAL_BRAIN_TEX = r'''\documentclass[tikz,border=14pt]{standalone}
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
\definecolor{warnAmber}{HTML}{D97706}
\definecolor{coralRed}{HTML}{DC2626}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  hostbox/.style={
    draw=ethblue!80,
    fill=ethblue!3,
    rounded corners=6pt,
    line width=1.0pt,
    inner sep=8pt,
    text width=8.6cm,
    minimum height=7.2cm
  },
  reflexbox/.style={
    draw=ethpetrol!90,
    fill=ethpetrol!3,
    rounded corners=6pt,
    line width=1.0pt,
    inner sep=8pt,
    text width=8.6cm,
    minimum height=7.2cm
  },
  srambox/.style={
    draw=ethbronze!90,
    fill=ethbronze!6,
    rounded corners=5pt,
    line width=0.9pt,
    inner sep=6pt,
    text width=18.0cm,
    align=center
  },
  organbox/.style={
    draw=cardborder,
    fill=white,
    rounded corners=3pt,
    line width=0.6pt,
    inner sep=4.5pt,
    align=center,
    text width=8.0cm
  },
  hwbox/.style={
    draw=ethslate!80,
    fill=white,
    rounded corners=4pt,
    line width=0.8pt,
    inner sep=6pt,
    text width=8.6cm,
    align=center
  }
]

  % Title Banner
  \node[font=\sffamily\bfseries\large, text=ethdarkblue, align=center] (title) at (9.4, 11.2) {
    \faIcon{microchip}\; THE PHYSICAL AI KIT: END-TO-END DUAL-BRAIN ARCHITECTURE
  };
  \node[font=\sffamily\footnotesize, text=ethslate, align=center, below=2pt of title] (subtitle) {
    Heterogeneous Multi-Rate Pipeline $\cdot$ Untrusted Linux MPU Proposals $\longleftrightarrow$ Trusted Real-Time MCU Permission
  };

  % ----------------------------------------------------
  % LEFT: HOST BRAIN (LINUX MPU)
  % ----------------------------------------------------
  \node[hostbox, anchor=north west, minimum height=9.2cm] (host_frame) at (0, 10.5) {};
  \node[font=\sffamily\bfseries\small, text=ethblue, anchor=north west] (host_title) at (0.3, 10.3) {
    \faIcon{brain}\; HOST BRAIN: LINUX MPU (Cognitive Cortex)
  };
  \node[font=\sffamily\scriptsize\color{ethslate}, anchor=north west] at (0.3, 9.85) {
    0.5--60 Hz $\cdot$ Best-Effort Scheduling $\cdot$ Untrusted Proposals
  };

  \node[organbox, anchor=north west] (mpu_sense) at (0.3, 9.3) {
    \textbf{\color{ethdarkblue}\faIcon{satellite-dish}\; Ingestion \& Spatial Tokenizer (60 Hz)}\\
    {\scriptsize MIPI CSI-2 DMA Rings $\cdot$ DINOv2 / MobileNet $\to$ 3D Spatial Affordance Tokens}
  };

  \node[organbox, below=0.18cm of mpu_sense] (mpu_state) {
    \textbf{\color{ethdarkblue}\faIcon{database}\; Temporal World Model \& Frame Graph (50 Hz)}\\
    {\scriptsize $SE(3)$ Lie Algebra Tree $\cdot$ Proprioceptive Innovation Gate $\cdot$ Latent JEPA Belief}
  };

  \node[organbox, below=0.18cm of mpu_state] (mpu_reason) {
    \textbf{\color{ethblue}\faIcon{comments}\; Semantic Reasoning \& Deliberation (1--2 Hz)}\\
    {\scriptsize Edge VLM (e.g. PaliGemma / SmolVLM) $\to$ 3D Bounding Goal $\cdot$ Expiring Lease $t_{\text{expire}}$}
  };

  \node[organbox, below=0.18cm of mpu_reason] (mpu_plan) {
    \textbf{\color{ethbronze}\faIcon{network-wired}\; Trajectory Planning \& Action Chunking (20 Hz)}\\
    {\scriptsize Diffusion Policy / ACT $\to H=16$ Step Action Chunks $\cdot \mathcal{C}^2$ Hermite Spline Ensembling}
  };

  % ----------------------------------------------------
  % RIGHT: REFLEX BRAIN (REAL-TIME MCU)
  % ----------------------------------------------------
  \node[reflexbox, anchor=north west, minimum height=9.2cm] (reflex_frame) at (9.4, 10.5) {};
  \node[font=\sffamily\bfseries\small, text=ethpetrol, anchor=north west] (reflex_title) at (9.7, 10.3) {
    \faIcon{shield-alt}\; REFLEX BRAIN: REAL-TIME MCU (Safety Enforcer)
  };
  \node[font=\sffamily\scriptsize\color{ethslate}, anchor=north west] at (9.7, 9.85) {
    1000 Hz Hard Real-Time Loop $\cdot$ Deterministic $\cdot$ Zero Malloc
  };

  \node[organbox, anchor=north west] (mcu_cbf) at (9.7, 9.3) {
    \textbf{\color{ethpetrol}\faIcon{balance-scale}\; Control Barrier Invariants ($h(x) \ge 0$)}\\
    {\scriptsize Zero-Allocation 1 kHz QP $\cdot$ Workspace Geofence $\cdot$ Payload \& Force Bounds}
  };

  \node[organbox, below=0.18cm of mcu_cbf] (mcu_stop) {
    \textbf{\color{ethpetrol}\faIcon{stopwatch}\; Dynamic Stopping Physics ($d_{\text{stop}}$)}\\
    {\scriptsize $d_{\text{stop}}(t) = v(t) \cdot t_{\text{delay}} + \frac{v(t)^2}{2 a_{\text{max}}} \le d_{\text{clearance}} \cdot$ Veto Accelerations}
  };

  \node[organbox, below=0.18cm of mcu_stop] (mcu_wdog) {
    \textbf{\color{ethpetrol}\faIcon{heartbeat}\; Heartbeat Supervisor \& Watchdog Lease}\\
    {\scriptsize Hardware Countdown Timer $\cdot \tau_{\text{watchdog}} \le 50\text{ ms} \cdot$ Trips on Linux Crash / Hang}
  };

  \node[organbox, below=0.18cm of mcu_wdog] (mcu_fsm) {
    \textbf{\color{harvardcrimson}\faIcon{exclamation-triangle}\; ISO 13850 Fallback Escalation FSM}\\
    {\scriptsize Cat 2 (Position Hold) $\to$ Cat 1 (Dynamic Brake) $\to$ Cat 0 (Power Relay Open)}
  };

  % ----------------------------------------------------
  % MIDDLE: HETEROGENEOUS SHARED SRAM & PRIVILEGE BOUNDARY
  % ----------------------------------------------------
  % Crisp horizontal privilege boundary
  \draw[dashed, line width=1.3pt, harvardcrimson] (0, 0.9) -- (18.8, 0.9)
    node[midway, font=\sffamily\bfseries\scriptsize, fill=white, draw=harvardcrimson!50, rounded corners=3pt, inner sep=2.5pt, text=harvardcrimson] {
      \faIcon{lock}\; THE PROPOSAL--PERMISSION PRIVILEGE BOUNDARY (MPU Proposes $p_t \;\longleftrightarrow\;$ MCU Permits $u_t$)
    };

  \node[srambox, anchor=north west] (sram) at (0.4, 0.5) {
    \textbf{\color{ethbronze}\faIcon{layer-group}\; HETEROGENEOUS Tightly-Coupled Memory (TCM) Shared SRAM (Lock-Free Seqlocks \& DMA Rings)}\\[2pt]
    {\scriptsize \faIcon{arrow-right}\; \textbf{Proposals (MPU $\to$ MCU):} Expiring Intent Leases ($t_{\text{expire}}$), $H$-step Action Splines ($p_t$), Model Confidence ($\kappa$)\\[1pt]
    \faIcon{arrow-left}\; \textbf{Telemetry (MCU $\to$ MPU):} Encoders, Voltages, Veto Flags, PTP Hardware Timestamps}
  };

  % ----------------------------------------------------
  % BOTTOM: ACTUATION, SENSORS & PHYSICAL WORLD
  % ----------------------------------------------------
  \node[hwbox, anchor=north west] (sensors_hw) at (0, -1.0) {
    \textbf{\color{ethdarkblue}\faIcon{camera}\; SENSOR SUITE (Environment Ingestion)}\\
    {\scriptsize MIPI CSI-2 Vision $\cdot$ 6-DoF IMU $\cdot$ Shunt Sense $\cdot$ PTP Time Base}
  };

  \node[hwbox, anchor=north west] (actuators_hw) at (9.4, -1.0) {
    \textbf{\color{ethpetrol}\faIcon{cogs}\; ACTUATION STAGE \& SAFETY RELAYS}\\
    {\scriptsize 3-Phase Gate Drivers $\cdot$ BLDC Motor Stage $\cdot$ Emergency Cutoff Relay}
  };

  \node[organbox, draw=harvardcrimson!80, fill=harvardcrimson!5, text width=18.0cm, anchor=north west] (world) at (0.4, -2.4) {
    \textbf{\color{harvardcrimson}\faIcon{globe-americas}\; THE PHYSICAL WORLD ($W_t \longrightarrow W_{t+1}$)} $\cdot$
    Kinetic Momentum ($\mathbf{p}=m\mathbf{v}$) $\cdot$ Joule Heat ($I^2R$) $\cdot$ Contact Dynamics $\cdot$ \textbf{Zero Software Undo}
  };

  % Connecting Dataflow Paths
  \draw[->, line width=1.0pt, ethblue] (mpu_sense) -- (mpu_state);
  \draw[->, line width=1.0pt, ethblue] (mpu_state) -- (mpu_reason);
  \draw[->, line width=1.0pt, ethblue] (mpu_reason) -- (mpu_plan);
  \draw[->, line width=1.1pt, ethbronze, dashed] (mpu_plan.south) -- (mpu_plan.south |- sram.north);

  \draw[->, line width=1.1pt, ethpetrol] ($(sram.north)+(4.3cm,0)$) -- ($(reflex_frame.south)+(0,0)$);
  \draw[->, line width=1.0pt, ethpetrol] (mcu_cbf) -- (mcu_stop);
  \draw[->, line width=1.0pt, ethpetrol] (mcu_stop) -- (mcu_wdog);
  \draw[->, line width=1.0pt, ethpetrol] (mcu_wdog) -- (mcu_fsm);

  \draw[->, line width=1.1pt, ethpetrol] (mcu_fsm.south) -- (mcu_fsm.south |- sram.north);
  \draw[->, line width=1.1pt, ethpetrol] (sram.south -| actuators_hw.north) -- (actuators_hw.north);
  \draw[->, line width=1.1pt, ethpetrol] (actuators_hw.south) -- (actuators_hw.south |- world.north);

  \draw[->, line width=1.0pt, ethdarkblue] (world.north -| sensors_hw.south) -- (sensors_hw.south);
  \draw[->, line width=1.0pt, ethdarkblue] (sensors_hw.north) -- (sensors_hw.north |- sram.south);
  \draw[->, line width=1.0pt, ethblue] ($(sram.north)+(-4.3cm,0)$) -- ++(0, 0.4cm) -| (mpu_sense.south);

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 2. FIG 99.2: SEEDED FAULT INJECTION & INVARIANT CONTAINMENT TIMELINE
# -----------------------------------------------------------------------------
SEEDED_FAULT_TEX = r'''\documentclass[tikz,border=14pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Academic Semantic Palette
\definecolor{harvardcrimson}{HTML}{A51C30}
\definecolor{ethdarkblue}{HTML}{1F407A}
\definecolor{ethblue}{HTML}{215CAF}
\definecolor{ethpetrol}{HTML}{007A87}
\definecolor{ethbronze}{HTML}{B87333}
\definecolor{ethslate}{HTML}{475569}
\definecolor{cardbg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\definecolor{safeTeal}{HTML}{10B981}
\definecolor{warnAmber}{HTML}{D97706}
\definecolor{coralRed}{HTML}{DC2626}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth
]

  % Title
  \node[font=\sffamily\bfseries\large, text=ethdarkblue, align=center] (title) at (7.5, 9.8) {
    \faIcon{stopwatch}\; CROSS-LAYER SEEDED FAULT INJECTION \& DYNAMIC ARREST
  };
  \node[font=\sffamily\footnotesize, text=ethslate, align=center, below=2pt of title] (subtitle) {
    Empirical Logic Analyzer Waveform: Linux Crash $\to$ Watchdog Lease Expiry $\to$ MCU Category 1 Dynamic Arrest
  };

  % Time Axis (Bottom)
  \draw[->, line width=1.0pt, ethslate] (0, 0) -- (15.2, 0) node[right, font=\sffamily\small\bfseries] {Time $t$ (ms)};
  
  \foreach \x/\label in {0/0, 2.5/50, 5.0/100, 7.5/150, 10.0/200, 12.5/250, 14.5/290} {
    \draw[line width=0.6pt, ethslate] (\x, 0.1) -- (\x, -0.1) node[below, font=\sffamily\scriptsize] {\label};
    \draw[dotted, line width=0.4pt, ethslate!30] (\x, 0) -- (\x, 7.6);
  }

  % Critical Vertical Markers (Clean non-overlapping headers)
  % 1. Fault Injection: t = 50 ms (x = 2.5)
  \draw[line width=1.1pt, harvardcrimson, dashed] (2.5, 0) -- (2.5, 7.8);
  \node[font=\sffamily\scriptsize\bfseries, fill=white, draw=harvardcrimson, rounded corners=2pt, inner sep=2.5pt, text=harvardcrimson, anchor=south] at (2.5, 8.4) {
    \faIcon{bolt}\; Injected Fault ($50\text{ ms}$)
  };

  % 2. Watchdog Expiry: t = 85 ms (x = 4.25)
  \draw[line width=1.1pt, warnAmber, dashed] (4.25, 0) -- (4.25, 7.8);
  \node[font=\sffamily\scriptsize\bfseries, fill=white, draw=warnAmber, rounded corners=2pt, inner sep=2.5pt, text=warnAmber, anchor=south] at (4.25, 7.8) {
    \faIcon{bell}\; Watchdog Trip ($85\text{ ms}$)
  };

  % 3. Full Arrest: t = 185 ms (x = 9.25)
  \draw[line width=1.1pt, safeTeal, dashed] (9.25, 0) -- (9.25, 7.8);
  \node[font=\sffamily\scriptsize\bfseries, fill=white, draw=safeTeal, rounded corners=2pt, inner sep=2.5pt, text=safeTeal, anchor=south] at (9.25, 7.8) {
    \faIcon{check-circle}\; Full Arrest ($185\text{ ms}$)
  };

  % ----------------------------------------------------
  % TRACE 1: Linux MPU Heartbeat (GPIO Toggle)
  % ----------------------------------------------------
  \node[anchor=west, font=\sffamily\scriptsize\bfseries, text=ethblue] at (0, 7.0) {
    \faIcon{brain}\; CH 1: Linux MPU Heartbeat Toggle (Nominal 50 Hz Pulse)
  };
  \draw[line width=1.2pt, ethblue] 
    (0, 6.3) -- (0.5, 6.3) -- (0.5, 6.7) -- (1.0, 6.7) -- (1.0, 6.3) -- (1.5, 6.3) -- (1.5, 6.7) -- (2.0, 6.7) -- (2.0, 6.3) -- (2.5, 6.3)
    -- (2.5, 6.1) -- (14.8, 6.1);
  \node[font=\sffamily\tiny\bfseries, fill=coralRed!10, text=coralRed, draw=coralRed!50, rounded corners=2pt, inner sep=2pt, anchor=west] at (3.0, 6.5) {
    KERNEL PANIC / SEEDED `kill -9` $\to$ GPIO TOGGLE DIES
  };

  % ----------------------------------------------------
  % TRACE 2: Hardware Watchdog Countdown Value
  % ----------------------------------------------------
  \node[anchor=west, font=\sffamily\scriptsize\bfseries, text=ethbronze] at (0, 5.3) {
    \faIcon{stopwatch}\; CH 2: MCU Hardware Watchdog Timer ($T_{\text{count}}$ in ms)
  };
  \draw[line width=1.1pt, ethbronze]
    (0, 4.7) -- (1.0, 4.3) -- (1.0, 4.7) -- (2.0, 4.3) -- (2.0, 4.7) -- (2.5, 4.5)
    -- (4.25, 3.8)
    -- (14.8, 3.8);
  \node[font=\sffamily\tiny\bfseries, fill=warnAmber!10, text=warnAmber, draw=warnAmber!40, rounded corners=2pt, inner sep=1.5pt] at (3.1, 4.3) {Lease Decaying};
  \node[font=\sffamily\tiny\bfseries, fill=coralRed!10, text=coralRed, draw=coralRed!50, rounded corners=2pt, inner sep=2pt, anchor=west] at (4.6, 4.1) {
    WATCHDOG EXPIRED ($0\text{ ms}$) $\to$ NMI INTERRUPT
  };

  % ----------------------------------------------------
  % TRACE 3: MCU Enforcer Authority State Machine
  % ----------------------------------------------------
  \node[anchor=west, font=\sffamily\scriptsize\bfseries, text=ethpetrol] at (0, 3.2) {
    \faIcon{shield-alt}\; CH 3: MCU Enforcer State Machine ($u_t$)
  };
  \draw[line width=1.3pt, safeTeal] (0, 2.4) -- (4.25, 2.4) 
    node[pos=0.4, above, font=\sffamily\tiny\bfseries, text=safeTeal] {NORMAL PERMISSION};
  \draw[line width=1.3pt, coralRed] (4.25, 2.4) -- (4.25, 2.8) -- (9.25, 2.8);
  \node[font=\sffamily\tiny\bfseries, fill=coralRed!10, text=coralRed, draw=coralRed!30, rounded corners=2pt, inner sep=1.5pt] at (6.75, 3.1) {
    CAT 1 DYNAMIC BRAKING ($a_{\text{max}}=-4.5\text{ m/s}^2$)
  };
  \draw[line width=1.3pt, ethslate] (9.25, 2.8) -- (9.25, 2.2) -- (14.8, 2.2) 
    node[pos=0.4, above, font=\sffamily\tiny\bfseries, text=ethslate] {CAT 0 SAFE REST ($v=0$)};

  % ----------------------------------------------------
  % TRACE 4: End-Effector Physical Velocity & Stopping Margin
  % ----------------------------------------------------
  \node[anchor=west, font=\sffamily\scriptsize\bfseries, text=harvardcrimson] at (0, 1.5) {
    \faIcon{tachometer-alt}\; CH 4: Physical Velocity $v(t)$ and Dynamic Stopping Distance $d_{\text{stop}}$
  };
  \draw[line width=1.4pt, ethdarkblue] 
    (0, 0.9) -- (4.25, 0.9) node[pos=0.4, above, font=\sffamily\tiny\bfseries] {$v_0 = 0.45\text{ m/s}$}
    -- (9.25, 0.3)
    -- (14.8, 0.3);

  \node[font=\sffamily\tiny\bfseries, fill=harvardcrimson!10, text=harvardcrimson, draw=harvardcrimson!30, rounded corners=2pt, inner sep=1.5pt] at (6.6, 0.8) {
    Controlled Deceleration ($a_{\text{max}}$)
  };

  % Stopping Distance Margin Bracket
  \draw[<->, line width=0.8pt, ethdarkblue] (4.25, 0.45) -- (9.25, 0.45)
    node[midway, fill=white, draw=ethdarkblue!30, rounded corners=2pt, inner sep=2pt, font=\sffamily\tiny\bfseries, text=ethdarkblue] {
      Braking Window: $100\text{ ms} \implies d_{\text{stop}} = 5.2\text{ cm} \le d_{\text{clearance}} = 12.0\text{ cm}$
    };

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 3. FIG 99.3: CUMULATIVE DOSSIER MATRIX & DEFENSE VERDICT GATE
# -----------------------------------------------------------------------------
DOSSIER_MATRIX_TEX = r'''\documentclass[tikz,border=14pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Academic Semantic Palette
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
\definecolor{warnAmber}{HTML}{D97706}
\definecolor{coralRed}{HTML}{DC2626}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  dossierbox/.style={
    draw=ethblue!80,
    fill=white,
    rounded corners=4pt,
    line width=0.8pt,
    inner sep=5pt,
    text width=6.2cm,
    font=\sffamily\scriptsize
  },
  auditbox/.style={
    draw=ethpurple!80,
    fill=ethpurple!5,
    rounded corners=4pt,
    line width=0.8pt,
    inner sep=5pt,
    text width=6.8cm,
    font=\sffamily\scriptsize
  },
  verdictdeploy/.style={
    draw=safeTeal,
    fill=safeTeal!10,
    rounded corners=5pt,
    line width=1.2pt,
    inner sep=6pt,
    text width=5.6cm,
    align=center
  },
  verdictcondition/.style={
    draw=warnAmber,
    fill=warnAmber!10,
    rounded corners=5pt,
    line width=1.2pt,
    inner sep=6pt,
    text width=5.6cm,
    align=center
  },
  verdictrefuse/.style={
    draw=coralRed,
    fill=coralRed!10,
    rounded corners=5pt,
    line width=1.2pt,
    inner sep=6pt,
    text width=5.6cm,
    align=center
  }
]

  % Title
  \node[font=\sffamily\bfseries\large, text=ethdarkblue, align=center] (title) at (10.5, 9.2) {
    \faIcon{balance-scale}\; THE ORAL DEFENSE \& RELEASE VERDICT QUALIFICATION GATE
  };
  \node[font=\sffamily\footnotesize, text=ethslate, align=center, below=2pt of title] {
    The 11-Artifact Cumulative Design Dossier $\longleftrightarrow$ 4-Expert Red-Team Audit $\longleftrightarrow$ Accountable Release Verdict
  };

  % ----------------------------------------------------
  % COLUMN 1: THE 11-ARTIFACT CUMULATIVE DOSSIER
  % ----------------------------------------------------
  \node[font=\sffamily\bfseries\small, text=ethblue, anchor=north west] (c1_hdr) at (0, 8.4) {
    \faIcon{folder-open}\; CUMULATIVE DESIGN DOSSIER
  };

  \node[dossierbox, below=0.2cm of c1_hdr] (dossier_p1) {
    \textbf{\color{ethdarkblue}Part I: Physical Foundation}\\
    $\bullet$ \textbf{LOOP-01:} Loop Charter \& Boundaries\\
    $\bullet$ \textbf{REQ-01:} $\tau_{\text{world}}$, $P_{99}$ Latency, $d_{\text{stop}}$ Ledger\\
    $\bullet$ \textbf{RUN-01:} Multi-Rate Skeleton \& Watchdogs
  };

  \node[dossierbox, below=0.2cm of dossier_p1] (dossier_p2) {
    \textbf{\color{ethdarkblue}Part II: 5 Sensory-Motor Organs}\\
    $\bullet$ \textbf{OBS-01:} DMA Ring Buffer Observation Contract\\
    $\bullet$ \textbf{STATE-01:} $SE(3)$ Frame Tree \& Timing Model\\
    $\bullet$ \textbf{INTENT-01:} 3D Goal Grounding \& Leases $t_{\text{expire}}$\\
    $\bullet$ \textbf{PLAN-01:} Action Chunks ($H$) \& $\mathcal{C}^2$ Jerk\\
    $\bullet$ \textbf{ENF-01:} 1 kHz MCU CBF Enforcer ($h(x) \ge 0$)
  };

  \node[dossierbox, below=0.2cm of dossier_p2] (dossier_p3) {
    \textbf{\color{ethdarkblue}Part III: Integration \& Release}\\
    $\bullet$ \textbf{PLACE-01:} Silicon Ledger (FLOPs, SRAM, W)\\
    $\bullet$ \textbf{AUTH-01:} Bumpless Override \& Lineage\\
    $\bullet$ \textbf{REL-01:} ODD Specification \& CAE Safety Case
  };

  % ----------------------------------------------------
  % COLUMN 2: 4-EXPERT RED-TEAM REVIEW BOARD
  % ----------------------------------------------------
  \node[font=\sffamily\bfseries\small, text=ethpurple, anchor=north west] (c2_hdr) at (7.2, 8.4) {
    \faIcon{user-shield}\; 4-EXPERT RED-TEAM AUDIT
  };

  \node[auditbox, below=0.2cm of c2_hdr] (audit_embed) {
    \textbf{\color{ethpurple}\faIcon{microchip}\; Embedded Silicon \& Real-Time Lead}\\
    $\bullet$ Zero-allocation MCU determinism ($0\text{ bytes dynamic}$)\\
    $\bullet$ Watchdog timeout $\le 50\text{ ms}$; DMA memory contention
  };

  \node[auditbox, below=0.15cm of audit_embed] (audit_ml) {
    \textbf{\color{ethpurple}\faIcon{brain}\; Embodied ML \& Planning Lead}\\
    $\bullet$ 3D Spatial grounding; token latencies; model abstention\\
    $\bullet$ Action chunking ($H$) with $\mathcal{C}^2$ temporal continuity
  };

  \node[auditbox, below=0.15cm of audit_ml] (audit_safety) {
    \textbf{\color{ethpurple}\faIcon{shield-alt}\; Robotics \& System Safety Lead}\\
    $\bullet$ Dynamic stopping bounds ($d_{\text{stop}} \le d_{\text{clearance}}$)\\
    $\bullet$ ISO 13850 stop categories; Claim-Argument-Evidence
  };

  \node[auditbox, below=0.15cm of audit_safety] (audit_ux) {
    \textbf{\color{ethpurple}\faIcon{user-graduate}\; Student \& Practitioner UX Lead}\\
    $\bullet$ Zero-magic reproducibility; 12-min fault triage\\
    $\bullet$ Hypothesis $\to$ Bisect $\to$ Confirm diagnostic logs
  };

  % ----------------------------------------------------
  % COLUMN 3: THE 3 RELEASE VERDICTS
  % ----------------------------------------------------
  \node[font=\sffamily\bfseries\small, text=ethdarkblue, anchor=north west] (c3_hdr) at (15.0, 8.4) {
    \faIcon{gavel}\; THE RELEASE VERDICT
  };

  \node[verdictdeploy, below=0.2cm of c3_hdr] (v_deploy) {
    \textbf{\color{safeTeal}\faIcon{check-circle}\; DEPLOY}\\[2pt]
    {\scriptsize \textbf{Unconstrained In-ODD Release}}\\[2pt]
    {\tiny $100\%$ fault containment $\cdot$ Zero invariant violations $\cdot$ $P_{99.9} \le \text{budget}$}
  };

  \node[verdictcondition, below=0.35cm of v_deploy] (v_condition) {
    \textbf{\color{warnAmber}\faIcon{exclamation-circle}\; CONDITION}\\[2pt]
    {\scriptsize \textbf{Restricted Operating Envelope}}\\[2pt]
    {\tiny Limited velocity ($v \le 50\%$) $\cdot$ Reduced payload $\cdot$ Supervisor in line-of-sight}
  };

  \node[verdictrefuse, below=0.35cm of v_condition] (v_refuse) {
    \textbf{\color{coralRed}\faIcon{times-circle}\; REFUSE}\\[2pt]
    {\scriptsize \textbf{Shipment Blocked / Redesign}}\\[2pt]
    {\tiny Hardware invariant breach $\cdot$ Uncontained fault $\cdot$ Tail latency exceeds stopping margin}
  };

  % Inter-Column Arrows
  \draw[->, line width=1.0pt, ethblue] (dossier_p2.east) -- (audit_ml.west);
  \draw[->, line width=1.0pt, ethpurple] (audit_embed.east) -- (v_deploy.west);
  \draw[->, line width=1.0pt, ethpurple] (audit_safety.east) -- (v_condition.west);
  \draw[->, line width=1.0pt, ethpurple] (audit_ux.east) -- (v_refuse.west);

\end{tikzpicture}
\end{document}
'''

def build_all():
    figures = {
        "fig99_dual_brain_integration.tex": DUAL_BRAIN_TEX,
        "fig99_seeded_fault_timeline.tex": SEEDED_FAULT_TEX,
        "fig99_defense_dossier_matrix.tex": DOSSIER_MATRIX_TEX
    }
    
    os.makedirs(CH99_FIG_DIR, exist_ok=True)
    
    for filename, tex in figures.items():
        tex_path = os.path.join(CH99_FIG_DIR, filename)
        pdf_name = filename.replace(".tex", ".pdf")
        svg_name = filename.replace(".tex", ".svg")
        png_name = filename.replace(".tex", "_preview")
        
        with open(tex_path, "w") as f:
            f.write(tex.strip() + "\n")
        print(f"Wrote {tex_path}")
        
        subprocess.run(["lualatex", "-interaction=nonstopmode", filename], cwd=CH99_FIG_DIR, check=True)
        subprocess.run(["pdftocairo", "-svg", pdf_name, svg_name], cwd=CH99_FIG_DIR, check=True)
        subprocess.run(["pdftoppm", "-png", "-r", "200", pdf_name, png_name], cwd=CH99_FIG_DIR, check=True)
        print(f"Compiled {pdf_name} -> {svg_name} and {png_name}-1.png")

if __name__ == "__main__":
    build_all()
