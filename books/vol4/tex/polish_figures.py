#!/usr/bin/env python3
"""
Polish all TikZ figures for Physical AI Systems:
- TeX Gyre Heros + sfmath + fontawesome5 vector glyphs
- Zero hyphenation artifacts (guaranteed via mbox and balanced lines)
- Precise node alignment, minimum heights, and balanced whitespace
- Publication-grade badges, lines, and callouts
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CH01_FIG_DIR = os.path.join(BOOK_DIR, "chapters", "01-boundary", "figures")
os.makedirs(CH01_FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. FIG 01: AGENT ANATOMY (PROPOSAL-PERMISSION ARCHITECTURE)
# -----------------------------------------------------------------------------
ANATOMY_TEX = r'''\documentclass[tikz,border=14pt]{standalone}
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

\newcommand{\mputag}[1]{\colorbox{ethbronze!15}{\scriptsize\bfseries\color{ethbronze}\,#1\,}}
\newcommand{\mcutag}[1]{\colorbox{ethpetrol!15}{\scriptsize\bfseries\color{ethpetrol}\,#1\,}}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  organ/.style={
    draw=cardborder,
    fill=white,
    rounded corners=5pt,
    line width=0.9pt,
    inner sep=8pt,
    align=center,
    text=ethdarkblue
  }
]

  % Top Banner: Stage 7 Governance
  \node[organ, draw=ethpurple, fill=ethpurple!6, text width=7.4in, line width=1.1pt] (s7) {
    {\normalsize\bfseries\color{ethpurple}\faIcon{balance-scale}\;\; 7. GOVERNANCE, LINEAGE \& RELEASE GATE}\\[2pt]
    {\footnotesize STPA Hazard Mitigation $\cdot$ Bumpless Human Joystick Override $\cdot$ Defensible Release Case (\textbf{LOOP-01} $\to$ \textbf{REL-01})}
  };

  % --- TOP PIPELINE ROW: Ingestion to State ---
  \node[organ, text width=2.22in, below=0.35in of s7.south west, anchor=north west] (s1) {
    {\bfseries\color{ethdarkblue}\faIcon{satellite-dish}\;\; 1. SENSING}\\[2pt]
    {\footnotesize Photons / Voltages to DMA}\\[2pt]
    {\scriptsize\color{ethslate}MIPI CSI-2 $\cdot$ I2C $\cdot$ SPI Bus Priority}
  };

  \node[organ, text width=2.22in, right=0.37in of s1] (s2) {
    {\bfseries\color{ethdarkblue}\faIcon{eye}\;\; 2. PERCEPTION}\\[2pt]
    {\footnotesize Encoders to Spatial Tokens}\\[2pt]
    {\scriptsize\color{ethslate}\mbox{ViT} $\cdot$ \mbox{DINOv2} $\cdot$ 3D Affordances}
  };

  \node[organ, text width=2.22in, right=0.37in of s2] (s3) {
    {\bfseries\color{ethdarkblue}\faIcon{database}\;\; 3. WORLD MODELS}\\[2pt]
    {\footnotesize Latent Belief \& $SE(3)$ Trees}\\[2pt]
    {\scriptsize\color{ethslate}JEPA / RSSM Dynamics $\cdot$ TTL Validity}
  };

  % --- MIDDLE PIPELINE ROW: Untrusted Proposal Engine (MPU) ---
  \node[organ, draw=ethbronze, fill=ethbronze!6, text width=3.51in, below=0.45in of s3.south east, anchor=north east] (s4) {
    \mputag{\faIcon{server}\;\; SYSTEM 2 $\cdot$ LINUX MPU}\\[3pt]
    {\bfseries\color{ethbronze}\faIcon{brain}\;\; 4. SEMANTIC DELIBERATION}\\[2pt]
    {\scriptsize\color{ethslate}VLMs \& Spatial Foundation Models $\cdot$ Goals $\cdot$ Leases ($t_{\text{expire}}$)}
  };

  \node[organ, draw=ethbronze, fill=ethbronze!6, text width=3.51in, left=0.38in of s4] (s5) {
    \mputag{\faIcon{microchip}\;\; SYSTEM 1.5 $\cdot$ LINUX MPU / NPU}\\[3pt]
    {\bfseries\color{ethbronze}\faIcon{network-wired}\;\; 5. TRAJECTORY DECODERS}\\[2pt]
    {\scriptsize\color{ethslate}Diffusion Policies $\cdot$ \mbox{ACT Action Chunking} ($H$-steps) $\cdot$ Jerk Bounds}
  };

  % --- LOWER ROW: Trusted Real-Time Enforcer (MCU) ---
  \node[organ, draw=ethpetrol, fill=ethpetrol!8, line width=1.3pt, text width=7.4in, below=0.72in of s5.south west, anchor=north west] (s6) {
    \mcutag{\faIcon{shield-alt}\;\; SYSTEM 1 $\cdot$ REAL-TIME MCU (BARE-METAL / FreeRTOS)}\\[3pt]
    {\bfseries\color{ethpetrol}\faIcon{tachometer-alt}\;\; 6. REAL-TIME REFLEX \& SAFETY ENFORCEMENT}\\[2pt]
    {\scriptsize\color{ethslate}1 kHz Reflex Timing Loop $\cdot$ Control Barrier Functions ($h(x) \ge 0$) $\cdot$ Stopping Distance $d_{\text{stop}}(v_t) \le d_{\text{clearance}} \cdot$ Veto ($u_t$)}
  };

  % --- PHYSICAL WORLD ROW ---
  \node[organ, draw=harvardcrimson, fill=harvardcrimson!6, line width=1.3pt, text width=7.4in, below=0.40in of s6.south west, anchor=north west] (world) {
    {\bfseries\color{harvardcrimson}\faIcon{cogs}\;\; THE PHYSICAL WORLD ($W_t \to W_{t+1}$)}\\[3pt]
    {\scriptsize\color{ethslate}Kinetic Momentum ($p{=}mv$) $\cdot$ Joule Heat Dissipation ($I^2R$) $\cdot$ Matter Mutation $\cdot$ Friction ($\mu$) $\cdot$ Collision Dynamics}
  };

  % Proposal-Permission Privilege Boundary (Red Dashed Line)
  \coordinate (bleft) at ($(s6.north west) + (0, 0.36in)$);
  \coordinate (bright) at ($(s6.north east) + (0, 0.36in)$);
  \draw[dashed, line width=1.3pt, harvardcrimson!85] (bleft) -- (bright);

  % Position badge clearly on the right half so it never collides with the left vertical arrow
  \node[font=\sffamily\bfseries\scriptsize, fill=white, draw=harvardcrimson!50, rounded corners=3pt, inner sep=3.5pt, text=harvardcrimson] 
    at ($(bleft)!0.62!(bright)$) 
    {\faIcon{lock}\;\; THE PROPOSAL--PERMISSION PRIVILEGE BOUNDARY (NO DIRECT MOTOR ACCESS)};

  % Clean Un-occluded Feed-forward Data Flow Arrows
  \draw[->, line width=1.2pt, ethdarkblue] (s1.east) -- (s2.west);
  \draw[->, line width=1.2pt, ethdarkblue] (s2.east) -- (s3.west);
  \draw[->, line width=1.2pt, ethdarkblue] (s3.south) -- (s3.south |- s4.north);
  \draw[->, line width=1.2pt, ethbronze] (s4.west) -- (s5.east);
  
  % Vertical Dataflow with White Badge Pills (Positioned safely above dashed line)
  \draw[->, line width=1.6pt, dashed, harvardcrimson] (s5.south) -- node[pos=0.25, fill=white, draw=harvardcrimson!40, rounded corners=2pt, inner sep=2.5pt, font=\sffamily\bfseries\scriptsize\color{harvardcrimson}]{\faIcon{paper-plane}\; Expiring Proposal $p_t$} (s5.south |- s6.north);
  \draw[->, line width=1.6pt, ethpetrol] (s6.south) -- node[pos=0.5, fill=white, draw=ethpetrol!40, rounded corners=2pt, inner sep=2.5pt, font=\sffamily\bfseries\scriptsize\color{ethpetrol}]{\faIcon{check-circle}\; Permitted Action $u_t = \text{permit}(p_t)$} (world.north);

  % Closed-loop Endogenous Feedback Arrow
  \draw[->, line width=1.2pt, harvardcrimson] (world.west) -- ++(-0.45in,0) |- (s1.west)
    node[pos=0.25, above, rotate=90, font=\sffamily\bfseries\scriptsize\color{harvardcrimson}, align=center]{\faIcon{sync-alt}\;\; Endogenous Sensory Shift ($A_t \to W_{t+1} \to O_{t+1}$)};

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 2. FIG 02: THREE TRIBES SYNTHESIS
# -----------------------------------------------------------------------------
TRIBES_TEX = r'''\documentclass[tikz,border=14pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Harvard Crimson & ETH Zurich Color Palette
\definecolor{harvardcrimson}{HTML}{A51C30}
\definecolor{ethdarkblue}{HTML}{1F407A}
\definecolor{ethblue}{HTML}{215CAF}
\definecolor{ethpetrol}{HTML}{007A87}
\definecolor{ethbronze}{HTML}{B87333}
\definecolor{ethslate}{HTML}{475569}
\definecolor{cardbg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\definecolor{frontierbg}{HTML}{FEF2F2}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  tribecard/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=6pt,
    line width=0.9pt,
    text width=2.46in,
    minimum height=2.58in,
    inner sep=10pt,
    align=left,
    anchor=north
  },
  synthesiscard/.style={
    draw=harvardcrimson,
    fill=frontierbg,
    rounded corners=6pt,
    line width=1.4pt,
    text width=8.08in,
    inner sep=10pt,
    align=left
  },
  arrowlabel/.style={
    font=\sffamily\scriptsize\bfseries,
    fill=white,
    inner sep=2.5pt,
    rounded corners=2pt
  }
]

  % --- TRIBE 1: The Brain (ML / AI) ---
  \node[tribecard, draw=ethblue!70] (brain) at (0,0) {
    {\scriptsize\bfseries\color{ethblue}\colorbox{ethblue!10}{\,\faIcon{brain}\; TRIBE 1 $\cdot$ THE BRAIN\,}}\\[4pt]
    {\normalsize\bfseries\color{ethdarkblue}The ML / AI Engineer}\\[1pt]
    {\scriptsize\bfseries\color{ethslate}Computer Science \& Modern AI}\\[6pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Core Strength:}\\
    Semantic Competence \& Representation\\[2pt]
    $\bullet$ Vision-Language Models (VLMs)\\
    $\bullet$ Action Chunking (ACT / Diffusion)\\
    $\bullet$ Latent World Models (JEPAs)\\[6pt]
    \textbf{\color{harvardcrimson}The Critical Blindspot:}\\[2pt]
    \textit{The Digital Sandbox Illusion}---assuming simulation guarantees physical safety and crashes are harmless.
    }
  };

  % --- TRIBE 2: The Nervous System (Embedded / ECE) ---
  \node[tribecard, draw=ethpetrol!80] (nervous) at (2.81in,0) {
    {\scriptsize\bfseries\color{ethpetrol}\colorbox{ethpetrol!10}{\,\faIcon{microchip}\; TRIBE 2 $\cdot$ THE NERVOUS SYSTEM\,}}\\[4pt]
    {\normalsize\bfseries\color{ethdarkblue}The Embedded / ECE Engineer}\\[1pt]
    {\scriptsize\bfseries\color{ethslate}Silicon \& Real-Time Systems}\\[6pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Core Strength:}\\
    Silicon Privilege \& Multi-Rate IPC\\[2pt]
    $\bullet$ Microsecond Clocks \& $P_{99}$ Metrology\\
    $\bullet$ Zero-Copy DMA \& Shared SRAM\\
    $\bullet$ Hardware Peripheral Bus Firewalls\\[6pt]
    \textbf{\color{harvardcrimson}The Critical Blindspot:}\\[2pt]
    \textit{The Static Automation Illusion}---treating real-time loops as rigid state machines unable to adapt.
    }
  };

  % --- TRIBE 3: The Body & Control (Robotics / Mechanical) ---
  \node[tribecard, draw=ethbronze!80] (body) at (5.62in,0) {
    {\scriptsize\bfseries\color{ethbronze}\colorbox{ethbronze!10}{\,\faIcon{cogs}\; TRIBE 3 $\cdot$ THE BODY \& CONTROL\,}}\\[4pt]
    {\normalsize\bfseries\color{ethdarkblue}The Robotics Engineer}\\[1pt]
    {\scriptsize\bfseries\color{ethslate}Dynamics, Invariants \& Control}\\[6pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Core Strength:}\\
    Physical Laws \& Safety Envelopes\\[2pt]
    $\bullet$ Multi-Body Inertia $\mathbf{M}(\mathbf{q})$ \& Momentum\\
    $\bullet$ Control Barrier Functions ($h(x) \ge 0$)\\
    $\bullet$ Thermal Limits ($I^2t$) \& Jerk Bounds\\[6pt]
    \textbf{\color{harvardcrimson}The Critical Blindspot:}\\[2pt]
    \textit{The Closed-World Illusion}---distrusting learned models as black boxes; fragile when domains drift.
    }
  };

  % --- SYNTHESIS BANNER: Physical AI Systems ---
  \node[synthesiscard, anchor=north] (synthesis) at (2.81in, -3.05in) {
    \begin{minipage}{0.99\linewidth}
      \centering
      {\normalsize\bfseries\color{harvardcrimson}\faIcon{project-diagram}\;\; THE PHYSICAL AI SYSTEMS SYNTHESIS}\\[2pt]
      {\scriptsize\bfseries\color{ethdarkblue}Bridging the Brain, Nervous System, and Body across the Proposal--Permission Boundary}\\[5pt]
      \rule{0.96\linewidth}{0.4pt}\\[6pt]
      \begin{tabular*}{\linewidth}{@{}p{2.58in}@{\hfill}p{2.58in}@{\hfill}p{2.58in}@{}}
        \scriptsize\textbf{\color{ethblue}\faIcon{brain}\; 1. Unverified Proposals} & 
        \scriptsize\textbf{\color{ethpetrol}\faIcon{network-wired}\; 2. Real-Time Transport} & 
        \scriptsize\textbf{\color{ethbronze}\faIcon{shield-alt}\; 3. Physical Invariants} \\[2pt]
        \scriptsize\raggedright High-capacity VLMs \& Diffusion ACT emit candidate action chunks ($p_t$) on Linux MPU. &
        \scriptsize\raggedright Lock-free SRAM ring buffers and hardware watchdogs bound tail latency ($P_{99}$). &
        \scriptsize\raggedright 1 kHz MCU filters proposals onto safe set $\mathcal{U}_{\text{safe}}$ via CBFs before gate drive.
      \end{tabular*}\\[7pt]
      {\small\bfseries\color{ethdarkblue}Universal Definition of Success: } 
      {\small\color{harvardcrimson}\textbf{Open-World Semantic Competence} $\;\mathbf{AND}\;$ \textbf{Strict Physical Invariant Survival}}
    \end{minipage}
  };

  % Connecting Arrows from Tribes to Synthesis with Generous Clearance
  \draw[->, line width=1.2pt, draw=ethblue] (brain.south) -- ++(0,-0.20in) -| ($(synthesis.north west) + (1.25in, 0)$)
    node[pos=0.22, arrowlabel, text=ethblue] {\faIcon{paper-plane}\; Semantic Proposals ($p_t$)};

  \draw[->, line width=1.2pt, draw=ethpetrol] (nervous.south) -- (synthesis.north)
    node[pos=0.45, arrowlabel, text=ethpetrol] {\faIcon{clock}\; Multi-Rate IPC \& Watchdogs};

  \draw[->, line width=1.2pt, draw=ethbronze] (body.south) -- ++(0,-0.20in) -| ($(synthesis.north east) + (-1.25in, 0)$)
    node[pos=0.22, arrowlabel, text=ethbronze] {\faIcon{shield-alt}\; 1 kHz CBF Safe Sets ($h(x) \ge 0$)};

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 3. FIG 03: FOUR ERAS OF EMBODIED AI
# -----------------------------------------------------------------------------
ERAS_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{fontawesome5}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Harvard Crimson & ETH Zurich Color Palette
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
  card/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=6pt,
    line width=0.8pt,
    text width=2.00in,
    minimum height=2.58in,
    inner sep=9pt,
    align=left,
    anchor=north
  },
  frontiercard/.style={
    draw=harvardcrimson,
    fill=harvardcrimson!4,
    rounded corners=6pt,
    line width=1.4pt,
    text width=2.04in,
    minimum height=2.58in,
    inner sep=9pt,
    align=left,
    anchor=north
  },
  arrowpill/.style={
    font=\sffamily\bfseries\scriptsize,
    fill=white,
    inner sep=2pt,
    rounded corners=2pt
  }
]

  % --- Phase 1: Disembodied ML ---
  \node[card] (p1) at (0,0) {
    {\scriptsize\bfseries\color{ethslate}\colorbox{ethslate!12}{\,\faIcon{server}\; PHASE 1 $\cdot$ 2012--2020\,}}\\[5pt]
    {\normalsize\bfseries\color{ethdarkblue}Disembodied ML}\\[1pt]
    {\scriptsize\bfseries\color{ethslate}The Cloud Era}\\[5pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Systems Focus:}\\
    High-throughput digital predictions ($x \to y$)\\[5pt]
    $\bullet$ \textbf{Substrate:} Cloud GPU Clusters\\
    $\bullet$ \textbf{Workloads:} ResNet, BERT, RecSys\\
    $\bullet$ \textbf{Boundary:} Stateless API / Screen\\
    $\bullet$ \textbf{Failure Mode:} \texttt{try/catch} $\to$ Retry\\
    $\bullet$ \textbf{Physical Action:} None (Bits)
    }
  };

  % --- Phase 2: Edge Perception ---
  \node[card] (p2) at (2.95in,0) {
    {\scriptsize\bfseries\color{ethpetrol}\colorbox{ethpetrol!12}{\,\faIcon{microchip}\; PHASE 2 $\cdot$ 2018--2023\,}}\\[5pt]
    {\normalsize\bfseries\color{ethdarkblue}Edge Perception}\\[1pt]
    {\scriptsize\bfseries\color{ethpetrol}The TinyML Era}\\[5pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Systems Focus:}\\
    Compressing models onto MCUs\\[5pt]
    $\bullet$ \textbf{Substrate:} Bare-Metal MCUs / DSPs\\
    $\bullet$ \textbf{Workloads:} Wake-words, Anomaly\\
    $\bullet$ \textbf{Boundary:} Open-loop sensing\\
    $\bullet$ \textbf{Failure Mode:} Dropped alert\\
    $\bullet$ \textbf{Physical Action:} Passive telemetry
    }
  };

  % --- Phase 3: Generative Deliberation ---
  \node[card] (p3) at (5.90in,0) {
    {\scriptsize\bfseries\color{ethblue}\colorbox{ethblue!12}{\,\faIcon{brain}\; PHASE 3 $\cdot$ 2023--2026\,}}\\[5pt]
    {\normalsize\bfseries\color{ethdarkblue}Deliberation}\\[1pt]
    {\scriptsize\bfseries\color{ethblue}The Foundation Era}\\[5pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Systems Focus:}\\
    Spatial reasoning using foundation models\\[5pt]
    $\bullet$ \textbf{Substrate:} Edge MPUs / NPUs\\
    $\bullet$ \textbf{Workloads:} VLMs, Transformers\\
    $\bullet$ \textbf{Boundary:} Semantic planning ($1\text{ Hz}$)\\
    $\bullet$ \textbf{Failure Mode:} Hallucination, $P_{99}$ tails\\
    $\bullet$ \textbf{Physical Action:} Untrusted proposals
    }
  };

  % --- Phase 4: Physical AI Systems ---
  \node[frontiercard] (p4) at (8.95in,0) {
    {\scriptsize\bfseries\color{harvardcrimson}\colorbox{harvardcrimson!15}{\,\faIcon{robot}\; PHASE 4 $\cdot$ NOW (FRONTIER)\,}}\\[5pt]
    {\normalsize\bfseries\color{harvardcrimson}Physical AI Systems}\\[1pt]
    {\scriptsize\bfseries\color{harvardcrimson}Closed-Loop Actuation}\\[5pt]
    {\scriptsize\color{ethslate}\raggedright
    \textbf{Systems Focus:}\\
    Proposals governed by real-time enforcers\\[5pt]
    $\bullet$ \textbf{Substrate:} Linux MPU + Real-Time MCU\\
    $\bullet$ \textbf{Workloads:} Multi-Rate VLA + 1 kHz CBF\\
    $\bullet$ \textbf{Boundary:} Delegated authority\\
    $\bullet$ \textbf{Failure Mode:} Gearbox shear / Impact\\
    $\bullet$ \textbf{Physical Action:} Matter \& kinetic energy
    }
  };

  % Transition Arrows with White Badge Pills
  \draw[->, line width=1.4pt, ethslate!60] (p1.east) -- node[pos=0.5, arrowpill, text=ethslate, draw=ethslate!30]{\faIcon{compress-alt}\; Compress} (p2.west);
  \draw[->, line width=1.4pt, ethpetrol!70] (p2.east) -- node[pos=0.5, arrowpill, text=ethpetrol, draw=ethpetrol!30]{\faIcon{brain}\; Reason} (p3.west);
  \draw[->, line width=1.8pt, harvardcrimson!90] (p3.east) -- node[pos=0.5, arrowpill, text=harvardcrimson, draw=harvardcrimson!35]{\faIcon{sync-alt}\; Close Loop} (p4.west);

  % Bottom Epistemic Divider Bar
  \node[draw=cardborder, fill=white, rounded corners=4pt, line width=0.8pt, inner sep=6pt, anchor=north, font=\sffamily\scriptsize\bfseries, text=ethdarkblue] 
    at ($(p1.south)!0.5!(p3.south) - (0, 0.22in)$) {
      $\longleftarrow$ \faIcon{laptop-code}\; \textbf{Open-Loop \& Digital Sandboxes} (Software retries, idempotent computation, no motor coils)
    };
  \node[draw=harvardcrimson, fill=harvardcrimson!10, rounded corners=4pt, line width=1pt, inner sep=6pt, anchor=north, font=\sffamily\scriptsize\bfseries, text=harvardcrimson] 
    at ($(p4.south) - (0, 0.22in)$) {
      \faIcon{cogs}\; \textbf{Physical Causality} ($W_t \to W_{t+1}$, No \texttt{ctrl+z}) $\longrightarrow$
    };

\end{tikzpicture}
\end{document}
'''

def build_all():
    files = {
        "fig01_agent_anatomy.tex": ANATOMY_TEX,
        "fig01_three_tribes.tex": TRIBES_TEX,
        "fig01_eras_evolution.tex": ERAS_TEX
    }
    
    for filename, content in files.items():
        filepath = os.path.join(CH01_FIG_DIR, filename)
        with open(filepath, "w") as f:
            f.write(content.strip() + "\n")
        print(f"Wrote {filepath}")
        
        pdf_path = filepath.replace(".tex", ".pdf")
        svg_path = filepath.replace(".tex", ".svg")
        subprocess.run(["lualatex", "-interaction=nonstopmode", filename], cwd=CH01_FIG_DIR, check=True)
        subprocess.run(["pdftocairo", "-svg", filename.replace(".tex", ".pdf"), filename.replace(".tex", ".svg")], cwd=CH01_FIG_DIR, check=True)
        print(f"Compiled {pdf_path} and {svg_path}")

if __name__ == "__main__":
    build_all()
