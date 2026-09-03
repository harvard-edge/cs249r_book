#!/usr/bin/env python3
"""
Generate Chapter 3 figures for Physical AI Systems:
- fig03_great_tension (The Great Tug-of-War: Deliberation vs Physical Real-Time)
- fig03_agent_workflow (The End-to-End Physical AI Agent Lifecycle & Dataflow)
- fig03_three_cadences (The Three Cadences: 1 Hz Intent vs 20-50 Hz ACT vs 1 kHz CBF)

Outputs vector PDF, native SVG (via pdftocairo), and PNG for visual inspection.
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CH03_FIG_DIR = os.path.join(BOOK_DIR, "chapters", "03-workflow", "figures")


# -----------------------------------------------------------------------------
# 1. FIG 03.1: THE GREAT TUG-OF-WAR (AI DELIBERATION VS PHYSICAL DYNAMICS)
# -----------------------------------------------------------------------------
TENSION_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
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

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  realmcard/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=6pt,
    line width=1pt,
    text width=3.35in,
    minimum height=2.65in,
    inner sep=10pt,
    align=left,
    anchor=north west
  }
]

  % Top Title Banner
  \node[draw=ethdarkblue, fill=ethdarkblue!5, rounded corners=5pt, line width=1pt, text width=8.70in, inner sep=7pt, align=center] (title) at (4.35in, 0) {
    {\normalsize\bfseries\color{ethdarkblue}\faIcon{arrows-alt-h}\;\; THE FUNDAMENTAL TUG-OF-WAR IN PHYSICAL AI}\\[2pt]
    {\scriptsize\color{ethslate}The Structural Conflict Between High-Capacity Semantic Deliberation and Relentless Physical Dynamics}
  };

  % Left Card: The AI / Reasoning Realm (Explicit anchor=north west)
  \node[realmcard, draw=ethbronze!80, fill=ethbronze!5] (ai_realm) at (0.30in, -0.65in) {
    {\small\bfseries\color{ethbronze}\faIcon{brain}\; THE AI \& DELIBERATION REALM}\\[2pt]
    {\scriptsize\bfseries\color{ethslate}Untrusted MPU / GPU Horizon (System 2 / System 1.5)}\\[6pt]
    \textbf{\color{ethbronze}What It Demands:}\\[2pt]
    {\tiny $\bullet$ Maximum expressive capacity ($7\text{B}\text{--}70\text{B}$ VLM parameters).}\\[2pt]
    {\tiny $\bullet$ Multi-view spatial attention \& deep token contexts.}\\[2pt]
    {\tiny $\bullet$ Iterative diffusion denoising (10--100 unrolling steps).}\\[2pt]
    {\tiny $\bullet$ Bounded time to deliberate on open-vocabulary goals.}\\[6pt]
    \rule{0.95\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{ethbronze}"Give me $500\text{ ms}$ to think for higher accuracy!"}}
  };

  % Right Card: The Physical / Dynamics Realm (Explicit anchor=north west)
  \node[realmcard, draw=harvardcrimson!80, fill=harvardcrimson!5] (phys_realm) at (5.05in, -0.65in) {
    {\small\bfseries\color{harvardcrimson}\faIcon{cogs}\; THE PHYSICAL \& DYNAMICS REALM}\\[2pt]
    {\scriptsize\bfseries\color{ethslate}Deterministic Real-Time Microcontroller (System 1)}\\[6pt]
    \textbf{\color{harvardcrimson}What It Demands:}\\[2pt]
    {\tiny $\bullet$ Gravity pulls at $9.8\text{ m/s}^2$ without waiting for inference.}\\[2pt]
    {\tiny $\bullet$ Kinetic momentum ($\mathbf{p} = m\mathbf{v}$) causes blind travel.}\\[2pt]
    {\tiny $\bullet$ Information freshness decays instantly ($\sigma_{\text{pos}} = \sigma_0 + v\Delta t$).}\\[2pt]
    {\tiny $\bullet$ Motor stator coils demand microsecond PWM switching.}\\[6pt]
    \rule{0.95\linewidth}{0.3pt}\\[4pt]
    {\scriptsize\textbf{\color{harvardcrimson}"Act in $1\text{ ms}$ or the mechanism collides!"}}
  };

  % Center Tension Tug-of-War Vector (Cleanly in the 1.40in gap)
  \draw[<->, line width=2.0pt, draw=ethdarkblue] (3.85in, -1.95in) -- (4.85in, -1.95in);
  \node[font=\sffamily\bfseries\tiny, fill=white, draw=ethdarkblue, rounded corners=3pt, inner sep=2.5pt, text=ethdarkblue, align=center] at (4.35in, -1.95in) {
    \faIcon{compress-arrows-alt}\; STRUCTURAL\\TENSION
  };

  % Bottom Synthesis Resolution Banner
  \node[draw=safeTeal, fill=safeTeal!10, rounded corners=5pt, line width=1pt, text width=8.70in, inner sep=6pt, align=center] (synthesis) at (4.35in, -3.65in) {
    {\small\bfseries\color{ethpetrol}\faIcon{handshake}\; HOW PHYSICAL AI RECONCILES THE CONFLICT: THE THREE ARCHITECTURAL CONTRACTS}\\[2pt]
    {\tiny\color{ethslate}\textbf{1. Proposal--Permission Privilege Split} (MPU proposes $\to$ MCU audits) $\quad\bullet\quad$ \textbf{2. Action Chunking} (Predicts $H$ steps to amortize delay) $\quad\bullet\quad$ \textbf{3. Expiring Leases} (Auto-fallback on lag)}
  };

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 2. FIG 03.2: END-TO-END PHYSICAL AGENT WORKFLOW & DATAFLOW
# -----------------------------------------------------------------------------
WORKFLOW_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
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
  flowbox/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=4pt,
    line width=0.8pt,
    text width=1.05in,
    minimum height=1.35in,
    inner sep=5pt,
    align=center,
    anchor=north
  }
]

  % Top Title Banner
  \node[draw=ethdarkblue, fill=ethdarkblue!5, rounded corners=5pt, line width=1pt, text width=8.90in, inner sep=6pt, align=center] (title) at (4.45in, 0) {
    {\normalsize\bfseries\color{ethdarkblue}\faIcon{project-diagram}\;\; THE COMPLETE END-TO-END PHYSICAL AGENT LIFECYCLE}\\[2pt]
    {\scriptsize\color{ethslate}Tracing Information Flow from Open-World Intent to Hardware PWM and Endogenous Feedback}
  };

  % ROW 1: SENSING, PERCEPTION & STATE
  \node[flowbox, draw=ethblue] (b1) at (0.40in, -0.55in) {
    {\tiny\bfseries\color{ethblue}\faIcon{satellite-dish}\; STAGE 1}\\[2pt]
    {\scriptsize\bfseries Transduction}\\[3pt]
    {\tiny\color{ethslate}Photons \& IMU\\[1pt]DMA Ingestion}\\[5pt]
    {\tiny\textbf{\color{ethblue}Ch 4}}
  };

  \node[flowbox, draw=ethblue!80] (b2) at (1.60in, -0.55in) {
    {\tiny\bfseries\color{ethblue}\faIcon{eye}\; STAGE 2}\\[2pt]
    {\scriptsize\bfseries Perception}\\[3pt]
    {\tiny\color{ethslate}ViT / DINOv2\\[1pt]3D Affordances}\\[5pt]
    {\tiny\textbf{\color{ethblue}Ch 4}}
  };

  \node[flowbox, draw=ethpurple] (b3) at (2.80in, -0.55in) {
    {\tiny\bfseries\color{ethpurple}\faIcon{database}\; STAGE 3}\\[2pt]
    {\scriptsize\bfseries World Model}\\[3pt]
    {\tiny\color{ethslate}JEPA Latent State\\[1pt]\& $SE(3)$ Trees}\\[5pt]
    {\tiny\textbf{\color{ethpurple}Ch 5}}
  };

  \node[flowbox, draw=ethbronze] (b4) at (4.00in, -0.55in) {
    {\tiny\bfseries\color{ethbronze}\faIcon{brain}\; STAGE 4}\\[2pt]
    {\scriptsize\bfseries Deliberation}\\[3pt]
    {\tiny\color{ethslate}VLM Intent Lease\\[1pt]$\mathcal{L}_{\text{intent}}$ ($1\text{ Hz}$)}\\[5pt]
    {\tiny\textbf{\color{ethbronze}Ch 6}}
  };

  \node[flowbox, draw=ethbronze!80] (b5) at (5.20in, -0.55in) {
    {\tiny\bfseries\color{ethbronze}\faIcon{network-wired}\; STAGE 5}\\[2pt]
    {\scriptsize\bfseries Action Chunk}\\[3pt]
    {\tiny\color{ethslate}Diffusion / ACT\\[1pt]Waypoints ($20\text{ Hz}$)}\\[5pt]
    {\tiny\textbf{\color{ethbronze}Ch 7}}
  };

  \node[flowbox, draw=ethpetrol] (b6) at (6.65in, -0.55in) {
    {\tiny\bfseries\color{ethpetrol}\faIcon{shield-alt}\; STAGE 6}\\[2pt]
    {\scriptsize\bfseries Safety Reflex}\\[3pt]
    {\tiny\color{ethslate}1 kHz CBF Filter\\[1pt]\& Stopping Veto}\\[5pt]
    {\tiny\textbf{\color{ethpetrol}Ch 8}}
  };

  \node[flowbox, draw=harvardcrimson] (b7) at (7.85in, -0.55in) {
    {\tiny\bfseries\color{harvardcrimson}\faIcon{cogs}\; STAGE 7}\\[2pt]
    {\scriptsize\bfseries Actuation}\\[3pt]
    {\tiny\color{ethslate}PWM Gate Drive\\[1pt]\& $L/R$ Coil Rise}\\[5pt]
    {\tiny\textbf{\color{harvardcrimson}Ch 8/11}}
  };

  % Dataflow Arrows Across Top Row
  \draw[->, line width=1pt, draw=ethslate] (b1.east) -- (b2.west);
  \draw[->, line width=1pt, draw=ethslate] (b2.east) -- (b3.west);
  \draw[->, line width=1pt, draw=ethslate] (b3.east) -- (b4.west);
  \draw[->, line width=1pt, draw=ethslate] (b4.east) -- (b5.west);
  \draw[->, line width=1pt, dashed, draw=ethbronze] (b5.east) -- (b6.west);
  \draw[->, line width=1pt, draw=safeTeal] (b6.east) -- (b7.west);

  % Proposal-Permission Boundary Divider Line
  \draw[dashed, line width=1.2pt, draw=ethbronze!90] (5.92in, -0.45in) -- (5.92in, -2.05in);
  \node[font=\sffamily\bfseries\tiny, fill=white, draw=ethbronze, rounded corners=2pt, inner sep=2pt, text=ethbronze, rotate=90] at (5.92in, -1.25in) {PROPOSAL-PERMISSION BOUNDARY};

  % Bottom Physical World Box ($W_t \to W_{t+1}$)
  \node[draw=harvardcrimson, fill=harvardcrimson!5, rounded corners=5pt, line width=1pt, text width=8.90in, inner sep=6pt, align=center] (world) at (4.45in, -2.40in) {
    {\small\bfseries\color{harvardcrimson}\faIcon{globe}\;\; THE PHYSICAL WORLD: IRREVERSIBLE STATE MUTATION ($W_t \longrightarrow W_{t+1}$)}\\[2pt]
    {\tiny\color{ethslate}Mechanical Mass Moves $\quad\bullet\quad$ Kinetic Momentum ($\mathbf{p} = m\mathbf{v}$) Mutates $\quad\bullet\quad$ Heat ($I^2R$) Dissipates $\quad\bullet\quad$ Hardware Stresses}
  };

  % Actuation to World Arrow
  \draw[->, line width=1.3pt, draw=harvardcrimson] (b7.south) -- (b7.south |- world.north);

  % Endogenous Feedback Loop from World back to Stage 1 (Wide loop with 0.45in clearance)
  \draw[->, line width=1.3pt, draw=harvardcrimson!80] (world.west) -- ++(-0.45in, 0) |- (b1.west)
    node[pos=0.25, above, rotate=90, font=\sffamily\bfseries\scriptsize, text=harvardcrimson] {\faIcon{sync-alt}\; Endogenous Feedback ($O_{t+1}$)};

\end{tikzpicture}
\end{document}
'''

# -----------------------------------------------------------------------------
# 3. FIG 03.3: THE THREE CADENCES (MULTI-RATE TEMPORAL HIERARCHY)
# -----------------------------------------------------------------------------
CADENCES_TEX = r'''\documentclass[tikz,border=12pt]{standalone}
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
  cadencebox/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=5pt,
    line width=0.9pt,
    text width=2.40in,
    minimum height=2.40in,
    inner sep=8pt,
    align=left,
    anchor=north west
  }
]

  % Top Title Banner
  \node[draw=ethdarkblue, fill=ethdarkblue!5, rounded corners=5pt, line width=1pt, text width=8.70in, inner sep=6pt, align=center] (title) at (4.35in, 0) {
    {\normalsize\bfseries\color{ethdarkblue}\faIcon{tachometer-alt}\;\; THE THREE CADENCES OF PHYSICAL AI}\\[2pt]
    {\scriptsize\color{ethslate}Asynchronous Multi-Rate Temporal Hierarchy: Intent $\to$ Action Chunks $\to$ Microsecond Reflexes}
  };

  % Column 1: System 2 (Semantic Intent)
  \node[cadencebox, draw=ethbronze!80, fill=ethbronze!5] (c1) at (0.20in, -0.65in) {
    {\small\bfseries\color{ethbronze}\faIcon{brain}\; SYSTEM 2: INTENT}\\[2pt]
    {\scriptsize\bfseries\color{ethslate}Slow Cadence: $0.5\text{--}2\text{ Hz}$ ($500\text{--}2000\text{ ms}$)}\\[6pt]
    \textbf{\color{ethbronze}Role:} Open-vocabulary reasoning\\[3pt]
    \textbf{\color{ethbronze}Model:} Multimodal VLM ($7\text{B}\text{--}70\text{B}$)\\[3pt]
    \textbf{\color{ethbronze}Output:} Expiring Intent Lease $\mathcal{L}_{\text{intent}}$\\[3pt]
    \textbf{\color{ethbronze}Authority:} Untrusted Proposal\\[6pt]
    \rule{0.95\linewidth}{0.3pt}\\[4pt]
    {\tiny\color{ethslate}\textbf{Failure Mode:} If VLM stalls or drops frames, lease expires harmlessly.}
  };

  % Column 2: System 1.5 (Action Chunking)
  \node[cadencebox, draw=ethblue!80, fill=ethblue!5] (c2) at (3.20in, -0.65in) {
    {\small\bfseries\color{ethblue}\faIcon{network-wired}\; SYSTEM 1.5: TRAJECTORY}\\[2pt]
    {\scriptsize\bfseries\color{ethslate}Medium Cadence: $20\text{--}50\text{ Hz}$ ($20\text{--}50\text{ ms}$)}\\[6pt]
    \textbf{\color{ethblue}Role:} Action Chunk unrolling\\[3pt]
    \textbf{\color{ethblue}Model:} Diffusion ACT / Policy\\[3pt]
    \textbf{\color{ethblue}Output:} Horizon of $H$ waypoints\\[3pt]
    \textbf{\color{ethblue}Authority:} Candidate Trajectory\\[6pt]
    \rule{0.95\linewidth}{0.3pt}\\[4pt]
    {\tiny\color{ethslate}\textbf{Advantage:} Amortizes neural compute delay across continuous execution.}
  };

  % Column 3: System 1 (Real-Time Reflex)
  \node[cadencebox, draw=ethpetrol!90, fill=ethpetrol!5] (c3) at (6.20in, -0.65in) {
    {\small\bfseries\color{ethpetrol}\faIcon{shield-alt}\; SYSTEM 1: REFLEX}\\[2pt]
    {\scriptsize\bfseries\color{ethslate}Fast Cadence: $1000\text{ Hz}$ ($1.0\text{ ms} \pm 5\,\mu\text{s}$)}\\[6pt]
    \textbf{\color{ethpetrol}Role:} Control Barrier Enforcement\\[3pt]
    \textbf{\color{ethpetrol}Model:} Bare-Metal FreeRTOS CBF\\[3pt]
    \textbf{\color{ethpetrol}Output:} Permitted Gate PWM ($u_t$)\\[3pt]
    \textbf{\color{ethpetrol}Authority:} \textbf{SOLE PERMISSION}\\[6pt]
    \rule{0.95\linewidth}{0.3pt}\\[4pt]
    {\tiny\color{ethslate}\textbf{Safety Invariant:} Deterministic emergency halt on any timing fault.}
  };

  % Flow arrows between cadences with clean non-colliding badge pills in the 0.60in gaps
  \draw[->, line width=1.2pt, ethbronze] ($(c1.north east) + (0, -0.90in)$) -- ($(c2.north west) + (0, -0.90in)$)
    node[midway, font=\sffamily\bfseries\tiny, fill=white, draw=ethbronze!50, rounded corners=2pt, inner sep=2pt, text=ethbronze] {Lease};

  \draw[->, line width=1.2pt, ethblue, dashed] ($(c2.north east) + (0, -0.90in)$) -- ($(c3.north west) + (0, -0.90in)$)
    node[midway, font=\sffamily\bfseries\tiny, fill=white, draw=ethblue!50, rounded corners=2pt, inner sep=2pt, text=ethblue] {Chunks};

\end{tikzpicture}
\end{document}
'''

def build_all():
    figures = {
        "fig03_great_tension.tex": TENSION_TEX,
        "fig03_agent_workflow.tex": WORKFLOW_TEX,
        "fig03_three_cadences.tex": CADENCES_TEX
    }
    
    for filename, tex in figures.items():
        tex_path = os.path.join(CH03_FIG_DIR, filename)
        pdf_name = filename.replace(".tex", ".pdf")
        svg_name = filename.replace(".tex", ".svg")
        png_name = filename.replace(".tex", "_preview")
        
        with open(tex_path, "w") as f:
            f.write(tex.strip() + "\n")
        print(f"Wrote {tex_path}")
        
        subprocess.run(["lualatex", "-interaction=nonstopmode", filename], cwd=CH03_FIG_DIR, check=True)
        subprocess.run(["pdftocairo", "-svg", pdf_name, svg_name], cwd=CH03_FIG_DIR, check=True)
        subprocess.run(["pdftoppm", "-png", "-r", "200", pdf_name, png_name], cwd=CH03_FIG_DIR, check=True)
        print(f"Compiled {pdf_name} -> {svg_name} and {png_name}-1.png")

if __name__ == "__main__":
    build_all()
