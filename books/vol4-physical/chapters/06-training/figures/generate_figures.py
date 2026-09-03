import subprocess
import os

def build_tex(tex_path, pdf_path, svg_path):
    print(f"Compiling {tex_path}...")
    cmd_pdf = ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_path)]
    res = subprocess.run(cmd_pdf, cwd=os.path.dirname(tex_path), capture_output=True, text=True)
    if res.returncode != 0:
        print("LaTeX Error:\n", res.stdout[-2000:])
        raise RuntimeError("pdflatex failed")
    
    cmd_svg = ["pdftocairo", "-svg", os.path.basename(pdf_path), os.path.basename(svg_path)]
    res_svg = subprocess.run(cmd_svg, cwd=os.path.dirname(tex_path), capture_output=True, text=True)
    if res_svg.returncode != 0:
        print("pdftocairo Error:\n", res_svg.stderr)
        raise RuntimeError("pdftocairo failed")
    print(f"Successfully generated {pdf_path} and {svg_path}")

# Figure 1: The Multidimensional Sim-to-Real Reality Gap Vector
fig1_tex = r"""\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc,patterns}
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
\definecolor{darkslate}{HTML}{0F172A}
\definecolor{cardbg}{HTML}{FFFFFF}
\definecolor{subtlebg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\definecolor{alertred}{HTML}{DC2626}
\definecolor{amberalert}{HTML}{D97706}
\definecolor{softgreen}{HTML}{059669}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  panelbg/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=4pt,
    line width=0.8pt
  },
  headerpill/.style={
    font=\sffamily\bfseries\scriptsize,
    rounded corners=2.5pt,
    inner sep=3pt
  },
  infobox/.style={
    rounded corners=2.5pt,
    font=\tiny,
    inner sep=3.5pt,
    line width=0.6pt
  }
]

  % =========================================================================
  % TOP HEADER BANNER
  % =========================================================================
  \draw[draw=ethdarkblue, fill=ethdarkblue!6, rounded corners=4pt, line width=1.1pt] (0, 15.0) rectangle (18.2, 16.3);
  \node[font=\normalsize\bfseries\color{ethdarkblue}, anchor=center] at (9.1, 15.82) {THE MULTIDIMENSIONAL SIM-TO-REAL REALITY GAP VECTOR $\mathbf{\Delta} = [\Delta_{\text{dyn}}, \Delta_{\text{lat}}, \Delta_{\text{sens}}, \Delta_{\text{vis}}]^T$};
  \node[font=\footnotesize\color{ethslate}, anchor=center] at (9.1, 15.35) {Task-relevant mismatches across heterogeneous physical channels: each carries distinct units, failure modes, runtime detectors, and unclosable residuals.};

  % =========================================================================
  % PANEL A: DYNAMICS & CONTACT IMPEDANCE MISMATCH (\Delta_dyn)
  % =========================================================================
  \begin{scope}[shift={(0, 7.6)}]
    \draw[panelbg] (0, 0) rectangle (8.9, 7.0);

    % Header Pill
    \node[headerpill, fill=ethpetrol!15, text=ethpetrol, anchor=north west] at (0.25, 6.75) {CHANNEL 1: DYNAMICS \& CONTACT ($\Delta_{\text{dyn}}$)};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (8.65, 6.75) {Units: $\text{N}, \text{N/s}, \text{rad}/(\text{N}\cdot\text{m})$};

    % Coordinate Frame inside Panel A
    \begin{scope}[shift={(1.0, 2.3)}]
      % Axes
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (7.2, 0) node[right, font=\scriptsize] {$t\ (\text{ms})$};
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (0, 3.4) node[above, font=\scriptsize] {$F_N\ (\text{N})$};

      % Ticks & Grid
      \foreach \x/\label in {1.17/10, 2.33/20, 3.50/30, 4.67/40, 5.83/50, 7.00/60} {
        \draw[color=cardborder, line width=0.4pt] (\x, 0) -- (\x, 3.2);
        \node[font=\tiny\color{ethslate}, below] at (\x, 0) {\label};
      }
      \foreach \y/\label in {0.56/2, 1.12/4, 1.68/6, 2.24/8, 2.80/10} {
        \draw[color=cardborder, line width=0.4pt] (0, \y) -- (7.0, \y);
        \node[font=\tiny\color{ethslate}, left] at (0, \y) {\label};
      }

      % Target Force Line: 10.0 N (y = 2.80)
      \draw[dashed, line width=0.8pt, color=darkslate!70] (0, 2.80) -- (7.0, 2.80);
      \node[font=\tiny\bfseries\color{darkslate}, anchor=south east] at (7.0, 2.85) {Target $F_N = 10.0\text{ N}$};

      % Simulation Trace (Rigid, 500 N/s -> reaches 10 N at 20 ms / x=2.33, settles)
      \draw[dashed, line width=1.3pt, color=ethpetrol] (0, 0) -- (2.33, 2.80) -- (7.0, 2.80);
      \node[font=\tiny\bfseries\color{ethpetrol}, anchor=south east] at (2.4, 1.7) {Sim ($k{=}\infty,\ 500\text{ N/s}$)};

      % Hardware Trace (8 ms delay = x: 0.93, soft compliance 250 N/s, overshoots to 11.5 N / y=3.22 at 50 ms / x=5.83)
      \draw[line width=1.4pt, color=harvardcrimson] 
        (0, 0) -- (0.93, 0) 
        .. controls (1.8, 0.1) and (2.8, 1.0) .. (4.2, 2.1)
        .. controls (5.0, 2.7) and (5.4, 3.22) .. (5.83, 3.22)
        .. controls (6.3, 3.22) and (6.6, 2.85) .. (7.0, 2.80);

      % 8 ms Stator Delay Callout
      \draw[<->, line width=0.6pt, color=harvardcrimson] (0, 0.25) -- (0.93, 0.25)
        node[midway, above, font=\tiny\bfseries\color{harvardcrimson}] {$\Delta t{=}8\text{ms}$};

      % Overshoot Callout
      \draw[->, line width=0.6pt, color=harvardcrimson] (5.83, 3.3) -- (5.83, 3.55)
        node[above, font=\tiny\bfseries\color{harvardcrimson}, align=center] {Overshoot $11.5\text{ N}$\\Impulse $0.15\text{ N}\cdot\text{s}$};
    \end{scope}

    % Bottom Diagnostic Info Box
    \draw[infobox, draw=harvardcrimson!50, fill=harvardcrimson!5] (0.25, 0.25) rectangle (8.65, 1.75);
    \node[anchor=north west, font=\tiny, text width=8.1cm] at (0.35, 1.65) {
      \textbf{\color{harvardcrimson}Physical Failure Signature:} Delayed force rise interpreted as non-contact $\to$ commands extra motor torque $\to 3.0\text{ mm}$ over-travel into current trip ($I_{\text{max}}$).\\
      \textbf{\color{darkslate}Runtime Detector:} Torque/force derivative clamp ($dF/dt \le 2.5\times 10^4\text{ N/s}$) in MCU reflex loop.\\
      \textbf{\color{ethpetrol}Unclosable Residual:} Gearbox compliance ($1.5\times 10^{-3}\text{ rad/N}\cdot\text{m}$) \& contact damping.
    };
  \end{scope}

  % =========================================================================
  % PANEL B: ACTUATOR & PIPELINE LATENCY WATERFALL (\Delta_lat)
  % =========================================================================
  \begin{scope}[shift={(9.3, 7.6)}]
    \draw[panelbg] (0, 0) rectangle (8.9, 7.0);

    % Header Pill
    \node[headerpill, fill=ethbronze!15, text=ethbronze, anchor=north west] at (0.25, 6.75) {CHANNEL 2: ACTUATOR \& LATENCY ($\Delta_{\text{lat}}$)};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (8.65, 6.75) {Units: $\text{ms}, \text{mm}$};

    % Coordinate Frame inside Panel B
    \begin{scope}[shift={(0.8, 2.3)}]
      % Time Axis (0 to 55 ms, width = 7.0 cm => 1 ms = 0.127 cm)
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (7.4, 0) node[right, font=\scriptsize] {$t\ (\text{ms})$};
      \foreach \x/\label in {0/0, 1.27/10, 2.54/20, 3.81/30, 5.08/40, 6.35/50} {
        \draw[color=cardborder, line width=0.4pt] (\x, 0) -- (\x, 3.6);
        \node[font=\tiny\color{ethslate}, below] at (\x, 0) {\label};
      }

      % Sim Assumption Bar (Synchronous tick: 20 ms / 2.54 cm)
      \node[anchor=west, font=\tiny\bfseries\color{ethpetrol}] at (0, 3.4) {Sim Assumption: Synchronous Cycle ($\Delta t = 20\text{ ms}$, Latency $= 0\text{ ms}$)};
      \draw[fill=ethpetrol!25, draw=ethpetrol, line width=0.8pt, rounded corners=1.5pt] (0, 2.65) rectangle (2.54, 3.15);
      \node[font=\tiny\bfseries\color{ethpetrol}] at (1.27, 2.90) {Sim Step $k$ (Instant Actuation)};

      % Hardware Reality Waterfall
      \node[anchor=west, font=\tiny\bfseries\color{harvardcrimson}] at (0, 2.3) {Physical Hardware Reality: Segmented Pipeline Latency};
      
      % Exposure (0 to 16.7 ms -> 0 to 2.12 cm)
      \draw[fill=ethdarkblue!25, draw=ethdarkblue, line width=0.7pt] (0, 1.45) rectangle (2.12, 1.95);
      \node[font=\tiny\bfseries\color{ethdarkblue}] at (1.06, 1.70) {Exposure $16.7\text{ms}$};

      % Bus Serialization (16.7 to 20.9 ms -> 2.12 to 2.65 cm)
      \draw[fill=ethslate!30, draw=ethslate, line width=0.7pt] (2.12, 1.45) rectangle (2.65, 1.95);
      \node[font=\tiny\color{white}, rotate=90] at (2.38, 1.70) {};

      % Neural Inference P50 (20.9 to 32.9 ms -> 2.65 to 4.18 cm)
      \draw[fill=ethbronze!30, draw=ethbronze, line width=0.7pt] (2.65, 1.45) rectangle (4.18, 1.95);
      \node[font=\tiny\bfseries\color{ethbronze}] at (3.41, 1.70) {Inference $12\text{ms}$};

      % Motor Fieldbus (32.9 to 35.0 ms -> 4.18 to 4.45 cm)
      \draw[fill=ethpetrol!35, draw=ethpetrol, line width=0.7pt] (4.18, 1.45) rectangle (4.45, 1.95);

      % Latency Tail Jitter P99 (35.0 to 49.5 ms -> 4.45 to 6.29 cm)
      \draw[pattern=north east lines, pattern color=harvardcrimson!60, draw=harvardcrimson, line width=0.7pt] (4.45, 1.45) rectangle (6.29, 1.95);
      \node[font=\tiny\bfseries\color{harvardcrimson}] at (5.37, 1.70) {Tail $+14.5\text{ms}$};

      % Vertical Latency Marker Lines
      \draw[dashed, line width=0.7pt, color=ethdarkblue] (4.45, 0) -- (4.45, 1.45);
      \node[font=\tiny\bfseries\color{ethdarkblue}, above] at (4.45, 0.75) {$P_{50}{=}35\text{ms}$};

      \draw[dashed, line width=0.7pt, color=harvardcrimson] (6.29, 0) -- (6.29, 1.45);
      \node[font=\tiny\bfseries\color{harvardcrimson}, above] at (6.29, 0.75) {$P_{99}{=}49.5\text{ms}$};

      % Unobserved Travel Indicator
      \draw[fill=amberalert!18, draw=amberalert, rounded corners=2pt, line width=0.7pt] (0, 0.15) rectangle (6.29, 0.65);
      \node[font=\tiny\bfseries\color{darkslate}] at (3.14, 0.40) {Blind Travel: $\Delta x = v \cdot \Delta t = 0.20\text{ m/s} \times 29.5\text{ ms} = \mathbf{5.9\text{ mm}}$ unbraked displacement};
    \end{scope}

    % Bottom Diagnostic Info Box
    \draw[infobox, draw=ethbronze!50, fill=ethbronze!5] (0.25, 0.25) rectangle (8.65, 1.75);
    \node[anchor=north west, font=\tiny, text width=8.1cm] at (0.35, 1.65) {
      \textbf{\color{ethbronze}Physical Failure Signature:} Commands arrive $29.5\text{ ms}$ late $\to 5.9\text{ mm}$ displacement depletes phase margin $\to$ high-frequency limit-cycle oscillation and impact shocks.\\
      \textbf{\color{darkslate}Runtime Detector:} Hardware timestamp age watchdog and RTOS jitter monitor.\\
      \textbf{\color{ethpetrol}Unclosable Residual:} Inference contention tail ($P_{99} = 49.5\text{ ms}$, jitter $\pm 6.2\text{ ms}$).
    };
  \end{scope}

  % =========================================================================
  % PANEL C: SENSING & TRANSDUCTION MISMATCH (\Delta_sens)
  % =========================================================================
  \begin{scope}[shift={(0, 0.2)}]
    \draw[panelbg] (0, 0) rectangle (8.9, 7.0);

    % Header Pill
    \node[headerpill, fill=ethpurple!15, text=ethpurple, anchor=north west] at (0.25, 6.75) {CHANNEL 3: SENSING \& TRANSDUCTION ($\Delta_{\text{sens}}$)};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (8.65, 6.75) {Units: $\text{mm}, \text{N}, \text{lux}$};

    % Coordinate Frame inside Panel C
    \begin{scope}[shift={(1.0, 2.3)}]
      % Axes
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (7.2, 0) node[right, font=\scriptsize] {$t\ (\text{ms})$};
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (0, 3.4) node[above, font=\scriptsize] {$z_{\text{meas}}\ (\text{mm})$};

      % Ticks & Grid (0 to 300 ms -> 0 to 7.0 cm, 0 to 25 mm -> 0 to 3.2 cm)
      \foreach \x/\label in {1.17/50, 2.33/100, 3.50/150, 4.67/200, 5.83/250, 7.00/300} {
        \draw[color=cardborder, line width=0.4pt] (\x, 0) -- (\x, 3.2);
        \node[font=\tiny\color{ethslate}, below] at (\x, 0) {\label};
      }
      \foreach \y/\label in {0.64/5, 1.28/10, 1.92/15, 2.56/20} {
        \draw[color=cardborder, line width=0.4pt] (0, \y) -- (7.0, \y);
        \node[font=\tiny\color{ethslate}, left] at (0, \y) {\label};
      }

      % Ground Truth Trajectory (Dotted: 20 mm descent to 0 mm at 200 ms / x=4.67)
      \draw[dotted, line width=1.1pt, color=darkslate] (0, 2.56) -- (4.67, 0) -- (7.0, 0);
      \node[font=\tiny\color{darkslate}, anchor=south west] at (1.2, 2.1) {True Range $z_{\text{true}}(t)$};

      % Simulated Sensor (Clean dashed steps with 1 mm quantization)
      \draw[dashed, line width=1.1pt, color=ethpetrol] 
        (0, 2.56) -- (1.17, 1.92) -- (2.33, 1.28) -- (3.50, 0.64) -- (4.67, 0) -- (7.0, 0);
      \node[font=\tiny\bfseries\color{ethpetrol}, anchor=south west] at (2.4, 1.35) {Sim (Clean, $1\text{mm}$ quant)};

      % Hardware Sensor (+3.2 mm bias = +0.41 cm, noise sigma=1.8 mm, dropout at 120-253 ms = 2.80 to 5.90 cm)
      \draw[line width=1.2pt, color=harvardcrimson]
        (0, 2.97) -- (0.5, 2.80) -- (1.0, 2.65) -- (1.5, 2.30) -- (2.0, 2.05) -- (2.5, 1.70) -- (2.8, 1.45);

      % 4-Frame Dropout Burst Box (120 to 253 ms = 133 ms duration)
      \draw[fill=alertred!18, draw=alertred, dashed, line width=0.8pt] (2.8, 0) rectangle (5.9, 2.6);
      \node[font=\tiny\bfseries\color{alertred}, align=center] at (4.35, 1.30) {4-Frame Specular Dropout Burst\\($133\text{ ms}$ Null Telemetry $\to$ Blind Impact!)};

      % Bias Callout
      \draw[<->, line width=0.7pt, color=harvardcrimson] (0, 2.56) -- (0, 2.97)
        node[midway, left, font=\tiny\bfseries\color{harvardcrimson}] {$+3.2\text{mm}$ Bias};
    \end{scope}

    % Bottom Diagnostic Info Box
    \draw[infobox, draw=ethpurple!50, fill=ethpurple!5] (0.25, 0.25) rectangle (8.65, 1.75);
    \node[anchor=north west, font=\tiny, text width=8.1cm] at (0.35, 1.65) {
      \textbf{\color{ethpurple}Physical Failure Signature:} $+3.2\text{ mm}$ bias triggers premature regulation $3.2\text{ mm}$ above surface; $133\text{ ms}$ optical dropout blinds policy during descent $\to$ unbraked crash.\\
      \textbf{\color{darkslate}Runtime Detector:} Consecutive dropout frame counter and thermal drift estimator.\\
      \textbf{\color{ethpetrol}Unclosable Residual:} Thermal drift ($0.40\text{ N} / 10^\circ\text{C}$), noise floor ($\sigma = 1.8\text{ mm}$).
    };
  \end{scope}

  % =========================================================================
  % PANEL D: PERCEPTUAL FEATURE SPACE SHIFT (\Delta_vis)
  % =========================================================================
  \begin{scope}[shift={(9.3, 0.2)}]
    \draw[panelbg] (0, 0) rectangle (8.9, 7.0);

    % Header Pill
    \node[headerpill, fill=ethdarkblue!15, text=ethdarkblue, anchor=north west] at (0.25, 6.75) {CHANNEL 4: PERCEPTUAL / VISUAL ($\Delta_{\text{vis}}$)};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (8.65, 6.75) {Units: $\text{Cosine Dist}, \text{mm}$};

    % Visual Feature Comparison Sub-Boxes
    \begin{scope}[shift={(0.4, 2.3)}]
      % Sub-box 1: Pixel Space Metric (Misleading low error)
      \draw[draw=cardborder, fill=subtlebg, rounded corners=3pt, line width=0.6pt] (0, 0) rectangle (3.8, 3.4);
      \node[anchor=north west, font=\scriptsize\bfseries\color{darkslate}] at (0.15, 3.25) {Pixel-Space Metric (Misleading)};
      \node[anchor=north west, font=\tiny\color{ethslate}] at (0.15, 2.85) {Mean Intensity $\Delta I = 5.5\%$ (MSE $= 0.002$)};

      % Mini visual frames
      \draw[fill=ethpetrol!15, draw=ethpetrol, line width=0.6pt] (0.3, 0.8) rectangle (1.7, 2.2);
      \node[font=\tiny\bfseries\color{ethpetrol}] at (1.0, 1.5) {Sim Render};
      
      \draw[fill=harvardcrimson!15, draw=harvardcrimson, line width=0.6pt] (2.1, 0.8) rectangle (3.5, 2.2);
      \node[font=\tiny\bfseries\color{harvardcrimson}] at (2.8, 1.5) {Real Camera};

      \node[font=\tiny\bfseries\color{softgreen}, anchor=center] at (1.9, 0.4) {$\Delta_{\text{pixel}} \le 5.5\%$ (Appears Converged)};

      % Sub-box 2: Spatial Feature Space (Catastrophic Offset)
      \draw[draw=alertred!60, fill=alertred!6, rounded corners=3pt, line width=0.8pt] (4.3, 0) rectangle (8.1, 3.4);
      \node[anchor=north west, font=\scriptsize\bfseries\color{harvardcrimson}] at (4.45, 3.25) {Latent Spatial Activation (True Gap)};
      \node[anchor=north west, font=\tiny\color{darkslate}] at (4.45, 2.85) {ViT Patch Token Centroid Shift:};

      % Feature Map Canvas
      \draw[fill=white, draw=cardborder, line width=0.6pt] (4.6, 0.7) rectangle (7.8, 2.3);
      
      % Chamfer tolerance region (+/- 2 mm -> width 1.0 cm)
      \draw[fill=softgreen!20, draw=softgreen, dashed, line width=0.6pt] (5.7, 0.8) rectangle (6.7, 2.2);
      \node[font=\tiny\color{softgreen}, anchor=north] at (6.2, 2.2) {$\pm 2\text{mm}$ Chamfer};

      % Centroid markers
      \fill[ethpetrol] (5.7, 1.4) circle (2.5pt);
      \node[font=\tiny\bfseries\color{ethpetrol}, below] at (5.7, 1.35) {Sim};

      \fill[harvardcrimson] (7.2, 1.4) circle (2.5pt);
      \node[font=\tiny\bfseries\color{harvardcrimson}, below] at (7.2, 1.35) {Real};

      \draw[<->, line width=1.0pt, color=harvardcrimson] (5.7, 1.6) -- (7.2, 1.6)
        node[midway, above, font=\tiny\bfseries\color{harvardcrimson}] {$\Delta x{=}6.0\text{mm}$};

      \node[font=\tiny\bfseries\color{alertred}, anchor=center] at (6.2, 0.35) {Exceeds $2.0\text{mm}$ Chamfer $\to$ Collision};
    \end{scope}

    % Bottom Diagnostic Info Box
    \draw[infobox, draw=ethdarkblue!50, fill=ethdarkblue!5] (0.25, 0.25) rectangle (8.65, 1.75);
    \node[anchor=north west, font=\tiny, text width=8.1cm] at (0.35, 1.65) {
      \textbf{\color{ethdarkblue}Physical Failure Signature:} $5.5\%$ pixel loss masks $6.0\text{ mm}$ spatial token offset $\to$ end-effector strikes fixture shoulder instead of bore.\\
      \textbf{\color{darkslate}Runtime Detector:} Feature-space Mahalanobis distance monitor on encoder tokens.\\
      \textbf{\color{ethpetrol}Unclosable Residual:} Specular BRDF reflection and micro-texture variations.
    };
  \end{scope}

  % =========================================================================
  % BOTTOM SUMMARY FOOTER
  % =========================================================================
  \draw[draw=ethslate, fill=ethslate!8, rounded corners=4pt, line width=0.9pt] (0, -1.3) rectangle (18.2, -0.1);
  \node[font=\footnotesize\bfseries\color{ethdarkblue}, anchor=west] at (0.35, -0.45) {Systems Architectural Synthesis:};
  \node[font=\scriptsize\color{darkslate}, anchor=west, text width=17.4cm] at (0.35, -0.85) {
    The sim-to-real gap is inherently a 4D coupled vector $\mathbf{\Delta} = [\Delta_{\text{dyn}}, \Delta_{\text{lat}}, \Delta_{\text{sens}}, \Delta_{\text{vis}}]^T$. Sizing the gap requires paired-trace telemetry across each physical axis. Domain randomization broadens coverage over modeled parameters ($\Xi$) but cannot bridge structural omissions or unmodeled timing tails outside the simulator graph.
  };

\end{tikzpicture}
\end{document}
"""

with open("/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_sim2real_gap.tex", "w") as f:
    f.write(fig1_tex.strip())

build_tex(
    "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_sim2real_gap.tex",
    "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_sim2real_gap.pdf",
    "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_sim2real_gap.svg"
)

# Figure 2: Compounding Covariate Shift and State Support Drift in Behavioral Cloning
fig2_tex = r"""\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc,patterns}
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
\definecolor{darkslate}{HTML}{0F172A}
\definecolor{cardbg}{HTML}{FFFFFF}
\definecolor{subtlebg}{HTML}{F8FAFC}
\definecolor{cardborder}{HTML}{CBD5E1}
\definecolor{alertred}{HTML}{DC2626}
\definecolor{amberalert}{HTML}{D97706}
\definecolor{softgreen}{HTML}{059669}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  panelbg/.style={
    draw=cardborder,
    fill=cardbg,
    rounded corners=4pt,
    line width=0.8pt
  },
  headerpill/.style={
    font=\sffamily\bfseries\scriptsize,
    rounded corners=2.5pt,
    inner sep=3pt
  },
  infobox/.style={
    rounded corners=2.5pt,
    font=\tiny,
    inner sep=3.5pt,
    line width=0.6pt
  }
]

  % =========================================================================
  % TOP HEADER BANNER
  % =========================================================================
  \draw[draw=ethdarkblue, fill=ethdarkblue!6, rounded corners=4pt, line width=1.1pt] (0, 12.2) rectangle (18.2, 13.5);
  \node[font=\normalsize\bfseries\color{ethdarkblue}, anchor=center] at (9.1, 13.02) {COMPOUNDING COVARIATE SHIFT AND STATE SUPPORT DRIFT IN BEHAVIORAL CLONING};
  \node[font=\footnotesize\color{ethslate}, anchor=center] at (9.1, 12.55) {Closed-loop error propagation: single-step perturbation $\epsilon$ shifts physical state off demonstrated support, inducing $\mathcal{O}(T^2 \epsilon)$ quadratic cumulative cost.};

  % =========================================================================
  % PANEL A: STATE-SPACE TRAJECTORY & SUPPORT MANIFOLD
  % =========================================================================
  \begin{scope}[shift={(0, 4.8)}]
    \draw[panelbg] (0, 0) rectangle (10.4, 7.0);

    % Header Pill
    \node[headerpill, fill=ethdarkblue!15, text=ethdarkblue, anchor=north west] at (0.25, 6.75) {STATE-SPACE TRAJECTORY \& DISTRIBUTION DRIFT};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (10.15, 6.75) {Demonstrated Support $\mathcal{S}_{\text{demo}}$ vs Deployed Rollout};

    % Coordinate Frame inside Panel A
    \begin{scope}[shift={(0.9, 1.4)}]
      % Axes: Time t vs State Displacement x(t)
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (8.8, 0) node[right, font=\scriptsize] {$t\ (\text{steps})$};
      \draw[->, line width=0.7pt, color=ethslate] (0, -0.6) -- (0, 4.5) node[above, font=\scriptsize] {State $s_t\ (\text{mm})$};

      % Ticks
      \foreach \x/\label in {0/0, 2.0/100, 4.0/200, 6.0/300, 8.0/400} {
        \draw[color=cardborder, line width=0.4pt] (\x, -0.5) -- (\x, 4.2);
        \node[font=\tiny\color{ethslate}, below] at (\x, -0.5) {\label};
      }
      \foreach \y/\label in {0/0.0, 1.0/+1.0, 2.0/+2.0, 3.0/+3.0, 4.0/+4.0} {
        \draw[color=cardborder, line width=0.4pt] (0, \y) -- (8.5, \y);
        \node[font=\tiny\color{ethslate}, left] at (0, \y) {\label};
      }

      % Demonstrated State Support Tube (Shaded Green/Petrol: +/- 0.2 mm around y=0.5)
      \fill[softgreen!15] (0, 0.2) -- (8.5, 0.2) -- (8.5, 1.0) -- (0, 1.0) -- cycle;
      \draw[softgreen, dashed, line width=0.7pt] (0, 0.2) -- (8.5, 0.2);
      \draw[softgreen, dashed, line width=0.7pt] (0, 1.0) -- (8.5, 1.0);
      \node[font=\tiny\bfseries\color{softgreen}, anchor=west] at (4.5, 0.6) {Demonstrated Support $\mathcal{S}_{\text{demo}}\ (\pm 0.2\text{ mm})$};

      % Nominal Expert Demonstration Trajectory (y = 0.6)
      \draw[line width=1.2pt, color=ethpetrol] (0, 0.6) -- (8.5, 0.6);
      \node[font=\tiny\bfseries\color{ethpetrol}, anchor=south west] at (0.2, 0.65) {Expert Trajectories $\mathcal{D}$};

      % First Error Perturbation at t = tau (tau = 120 steps -> x = 2.4)
      \draw[dashed, line width=0.8pt, color=amberalert] (2.4, -0.5) -- (2.4, 4.0);
      \fill[amberalert] (2.4, 0.6) circle (2.5pt);
      \node[font=\tiny\bfseries\color{amberalert}, above right] at (2.4, 0.65) {First Error ($t{=}\tau,\ P{=}\epsilon$)};

      % Deployed Behavioral Cloning Policy (Diverging OOD Trajectory)
      \draw[line width=1.5pt, color=harvardcrimson] 
        (0, 0.6) -- (2.4, 0.6) 
        .. controls (3.2, 0.8) and (4.0, 1.4) .. (5.0, 2.3)
        .. controls (6.0, 3.2) and (7.0, 3.9) .. (8.0, 4.2);
      \node[font=\tiny\bfseries\color{harvardcrimson}, anchor=south east] at (7.8, 4.25) {Cloned Policy $\pi_{\text{BC}}$ (Covariate Shift $\to \mathcal{O}(T^2\epsilon)$)};

      % Recovery-Trained / DAgger Trajectory (Corrects back into tube)
      \draw[line width=1.2pt, color=ethblue, dashdotted]
        (2.4, 0.6) 
        .. controls (3.0, 1.1) and (3.6, 1.3) .. (4.2, 1.0)
        .. controls (4.8, 0.7) and (5.4, 0.6) .. (8.5, 0.6);
      \node[font=\tiny\bfseries\color{ethblue}, anchor=south west] at (4.5, 1.1) {On-Policy / DAgger Recovery ($\mathcal{O}(T\epsilon)$)};

      % Out-of-Distribution Shaded Zone
      \fill[alertred!8, opacity=0.8] (0, 1.0) rectangle (8.5, 4.3);
      \node[font=\tiny\bfseries\color{alertred}, anchor=north east] at (8.4, 4.1) {OUT-OF-DISTRIBUTION (OOD) STATE SPACE (Loss Unconstrained)};

      % Runtime Detector Trip Line (Mahalanobis Threshold at y=2.0 / x=4.7)
      \draw[dashed, line width=0.9pt, color=ethpurple] (0, 2.0) -- (8.5, 2.0);
      \node[font=\tiny\bfseries\color{ethpurple}, anchor=north west] at (0.2, 2.0) {Nervous System Support Monitor Trip Threshold ($D_M \ge \gamma_{\text{trip}}$)};
      \fill[ethpurple] (4.7, 2.0) circle (3pt);
      \node[font=\tiny\bfseries\color{ethpurple}, above left] at (4.7, 2.05) {Safety Interlock Trip};

      % Hard Mechanical Stop / Collision Boundary (y = 4.0)
      \draw[line width=1.1pt, color=darkslate] (0, 4.0) -- (8.5, 4.0);
      \node[font=\tiny\bfseries\color{darkslate}, anchor=south east] at (8.5, 4.05) {Physical Mechanical Stop / Collision Boundary};
    \end{scope}

    % Bottom Diagnostic Note
    \node[font=\tiny\color{darkslate}, anchor=south west] at (0.35, 0.2) {
      \textbf{\color{harvardcrimson}Silent Failure Signature:} Policy emits valid float commands while physical state drifts off $\mathcal{S}_{\text{demo}}$ support until safety trip.
    };
  \end{scope}

  % =========================================================================
  % PANEL B: CUMULATIVE COST BOUND COMPARISON (O(T^2 epsilon) vs O(T epsilon))
  % =========================================================================
  \begin{scope}[shift={(10.8, 4.8)}]
    \draw[panelbg] (0, 0) rectangle (7.4, 7.0);

    % Header Pill
    \node[headerpill, fill=ethbronze!15, text=ethbronze, anchor=north west] at (0.25, 6.75) {CUMULATIVE ERROR SCALING};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (7.15, 6.75) {Horizon $T$ vs Cost $J$};

    % Coordinate Frame inside Panel B
    \begin{scope}[shift={(0.8, 1.4)}]
      % Axes
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (6.0, 0) node[right, font=\scriptsize] {$T\ (\text{steps})$};
      \draw[->, line width=0.7pt, color=ethslate] (0, 0) -- (0, 4.5) node[above, font=\scriptsize] {Cost $J(T)$};

      % Ticks
      \foreach \x/\label in {1.0/100, 2.0/200, 3.0/300, 4.0/400, 5.0/500} {
        \draw[color=cardborder, line width=0.4pt] (\x, 0) -- (\x, 4.2);
        \node[font=\tiny\color{ethslate}, below] at (\x, 0) {\label};
      }
      \foreach \y/\label in {1.0/25, 2.0/50, 3.0/75, 4.0/100} {
        \draw[color=cardborder, line width=0.4pt] (0, \y) -- (5.5, \y);
        \node[font=\tiny\color{ethslate}, left] at (0, \y) {\label};
      }

      % Quadratic Compounding Curve: J = epsilon * T*(T+1)/2 (Crimson)
      % For epsilon=0.0008: T=500 -> J = 100 -> y=4.0
      \draw[line width=1.5pt, color=harvardcrimson]
        (0, 0) .. controls (1.6, 0.4) and (3.4, 1.8) .. (5.0, 4.0);
      \node[font=\tiny\bfseries\color{harvardcrimson}, anchor=south east] at (4.9, 4.05) {Cloning: $J = \mathcal{O}(T^2 \epsilon)$};

      % Linear On-Policy Curve: J = c * T * epsilon (Petrol)
      \draw[line width=1.3pt, color=ethpetrol] (0, 0) -- (5.0, 0.8);
      \node[font=\tiny\bfseries\color{ethpetrol}, anchor=south west] at (2.5, 0.45) {On-Policy: $J = \mathcal{O}(T \epsilon)$};

      % Statistical Error Probability Callout Box
      \draw[fill=ethpurple!8, draw=ethpurple!40, rounded corners=2pt, line width=0.6pt] (0.3, 2.2) rectangle (3.8, 3.8);
      \node[anchor=north west, font=\tiny, text width=3.3cm] at (0.4, 3.7) {
        \textbf{\color{ethpurple}Analytical Horizon Failure:}\\
        $P(\ge 1\text{ error}) = 1 - (1 - \epsilon)^T$\\[2pt]
        For $T = 500$, $\epsilon = 0.005$:\\[1pt]
        $P(\ge 1\text{ error}) \approx \mathbf{91.8\%}$ (${\approx}92\%$)
      };
    \end{scope}

    % Bottom Note
    \node[font=\tiny\color{darkslate}, anchor=south west] at (0.35, 0.2) {
      \textbf{\color{darkslate}Key Insight:} Reducing $\epsilon$ postpones divergence time ($1/\epsilon$) but cannot remove $\mathcal{O}(T^2\epsilon)$ compounding.
    };
  \end{scope}

  % =========================================================================
  % PANEL C: THE CLOSED-LOOP CAUSAL CHAIN OF COVARIATE SHIFT
  % =========================================================================
  \begin{scope}[shift={(0, 0)}]
    \draw[panelbg] (0, 0) rectangle (18.2, 4.4);

    % Header Pill
    \node[headerpill, fill=ethpurple!15, text=ethpurple, anchor=north west] at (0.25, 4.15) {CAUSAL FEEDBACK MECHANISM: FROM OFFLINE LOSS TO CLOSED-LOOP DESTRUCTION};
    \node[font=\scriptsize\color{ethslate}, anchor=north east] at (17.95, 4.15) {Privilege Boundary \& Endogenous Sensory Shift};

    % Step 1 Box: Policy Network
    \draw[draw=ethbronze, fill=ethbronze!8, rounded corners=3pt, line width=0.8pt] (0.4, 1.4) rectangle (4.2, 3.5);
    \node[font=\scriptsize\bfseries\color{ethbronze}, anchor=north west] at (0.55, 3.35) {1. LEARNED POLICY $\pi_\theta$};
    \node[font=\tiny\color{darkslate}, anchor=north west, text width=3.4cm] at (0.55, 2.95) {
      $\bullet$ Trained on $\mathcal{D} = \{(o_t, a_t)\}$\\
      $\bullet$ Zero recovery examples\\
      $\bullet$ 1-step error probability $\epsilon$\\
      $\bullet$ Emits valid float commands
    };

    % Arrow 1 -> 2
    \draw[->, line width=1.1pt, color=ethbronze] (4.2, 2.45) -- (5.0, 2.45)
      node[midway, above, font=\tiny\bfseries\color{ethbronze}] {$a_t$};

    % Step 2 Box: Privilege Boundary & Nervous System
    \draw[draw=harvardcrimson, fill=harvardcrimson!6, rounded corners=3pt, line width=0.9pt] (5.0, 1.4) rectangle (8.8, 3.5);
    \node[font=\scriptsize\bfseries\color{harvardcrimson}, anchor=north west] at (5.15, 3.35) {2. NERVOUS SYSTEM};
    \node[font=\tiny\color{darkslate}, anchor=north west, text width=3.4cm] at (5.15, 2.95) {
      $\bullet$ Privilege boundary check\\
      $\bullet$ Commands look valid\\
      $\bullet$ \textbf{Monitor:} Mahalanobis $D_M$\\
      $\bullet$ Passes command if unmonitored
    };

    % Arrow 2 -> 3
    \draw[->, line width=1.1pt, color=harvardcrimson] (8.8, 2.45) -- (9.6, 2.45)
      node[midway, above, font=\tiny\bfseries\color{harvardcrimson}] {$u_t$};

    % Step 3 Box: Physical Plant & Actuator
    \draw[draw=ethdarkblue, fill=ethdarkblue!8, rounded corners=3pt, line width=0.8pt] (9.6, 1.4) rectangle (13.4, 3.5);
    \node[font=\scriptsize\bfseries\color{ethdarkblue}, anchor=north west] at (9.75, 3.35) {3. PHYSICAL PLANT};
    \node[font=\tiny\color{darkslate}, anchor=north west, text width=3.4cm] at (9.75, 2.95) {
      $\bullet$ Actuator exerts force $F_t$\\
      $\bullet$ Moves state: $s_t \to s_{t+1}$\\
      $\bullet$ \textbf{Endogenous Shift:}\\
      $\quad s_{t+1} \notin \text{supp}(\mathcal{D}_{\text{demo}})$
    };

    % Arrow 3 -> 4
    \draw[->, line width=1.1pt, color=ethdarkblue] (13.4, 2.45) -- (14.2, 2.45)
      node[midway, above, font=\tiny\bfseries\color{ethdarkblue}] {$s_{t+1}$};

    % Step 4 Box: Sensor Transduction
    \draw[draw=ethpetrol, fill=ethpetrol!8, rounded corners=3pt, line width=0.8pt] (14.2, 1.4) rectangle (17.8, 3.5);
    \node[font=\scriptsize\bfseries\color{ethpetrol}, anchor=north west] at (14.35, 3.35) {4. SENSING};
    \node[font=\tiny\color{darkslate}, anchor=north west, text width=3.2cm] at (14.35, 2.95) {
      $\bullet$ Photons / Encoders\\
      $\bullet$ Observation $o_{t+1} \sim p_\pi(o)$\\
      $\bullet$ \textbf{Covariate Shift:}\\
      $\quad o_{t+1} \notin \text{supp}(p_{\text{demo}})$
    };

    % Closed-loop feedback return arrow: 4 -> 1
    \draw[->, line width=1.2pt, color=harvardcrimson, dashed] 
      (16.0, 1.4) -- (16.0, 0.7) -- (2.3, 0.7) -- (2.3, 1.4)
      node[pos=0.5, above, font=\tiny\bfseries\color{harvardcrimson}] {Closed-Loop Feedback: Unvisited Observation Queried Off-Distribution $\implies$ Error Compounding $\mathcal{O}(T^2\epsilon)$};

  \end{scope}

\end{tikzpicture}
\end{document}
"""

with open("/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_compounding_error.tex", "w") as f:
    f.write(fig2_tex.strip())

build_tex(
    "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_compounding_error.tex",
    "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_compounding_error.pdf",
    "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/06-training/figures/fig06_compounding_error.svg"
)
