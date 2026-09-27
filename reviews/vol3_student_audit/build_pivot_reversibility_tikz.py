import subprocess
from pathlib import Path

TIKZ_CODE = r"""\documentclass[tikz,border=22pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb}
\usetikzlibrary{calc,positioning,fit,backgrounds,shapes.geometric,shapes.symbols,arrows.meta,decorations.pathreplacing}

% =============================================================================
% CANONICAL MLSYSBOOK PALETTE
% =============================================================================
\definecolor{DarkSlate}{HTML}{0F172A}
\definecolor{slatesteel}{HTML}{334155}
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
\definecolor{VioletLine}{HTML}{7E317B}
\definecolor{VioletL}{RGB}{247,215,247}
\definecolor{amethyst}{HTML}{581C87}
\definecolor{amethystL0}{HTML}{FAF5FF}
\definecolor{GrayLine}{HTML}{64748B}
\definecolor{coolparchment}{HTML}{F8FAFC}
\definecolor{CardBorder}{HTML}{CBD5E1}
\definecolor{ChipBlack}{HTML}{181F2C}
\definecolor{GoldPin}{HTML}{EAB308}

\newcommand{\greenpass}{\tikz[baseline=-0.6ex]\draw[fill=forestsage, draw=none] (0,0) circle (2.4pt);}
\newcommand{\redfail}{\tikz[baseline=-0.6ex]\draw[fill=deepburgundy, draw=none] (0,0) circle (2.4pt);}
\newcommand{\amberwarn}{\tikz[baseline=-0.6ex]\draw[fill=warmochre, draw=none] (0,0) circle (2.4pt);}

\tikzset{
  >=Latex,
  font=\sffamily,
  flow_arrow/.style={->, >=Latex, line width=1.3pt},
  flow_arrow_thick/.style={->, >=Latex, line width=1.8pt},
  bus_arrow/.style={<->, >=Latex, line width=2.0pt},
}

\begin{document}
\begin{tikzpicture}[x=1cm, y=1cm]

% =============================================================================
% TOP TITLE BANNER (y: 17.5 to 18.9)
% =============================================================================
\draw[fill=DarkSlate, draw=none, rounded corners=4pt] (0, 17.5) rectangle (23.2, 18.9);
\node[anchor=west, font=\bfseries\large, text=white] at (0.45, 18.4) {
  THE ACTION REVERSIBILITY SPECTRUM \& PRE-PIVOT VERIFICATION GATE
};
\node[anchor=east, font=\footnotesize, text=GoldPin] at (22.75, 18.4) {
  Chapter 12: Failure Recovery $\cdot$ Section 12.3
};
\node[anchor=west, font=\scriptsize, text=white!85] at (0.45, 17.85) {
  Bifurcating Agent Execution: Compensable Preparation vs. Irreversible Commitment Across the Point of No Return
};

% =============================================================================
% 1. TOP TIER: THE TRIPARTITE ACTION REVERSIBILITY TAXONOMY
% (x: 0 to 23.2, y: 10.9 to 17.1)
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.1pt, rounded corners=5pt] (0, 10.9) rectangle (23.2, 17.1);

% Header Bar
\fill[DarkSlate, rounded corners=3pt] (0.2, 16.45) rectangle (23.0, 16.95);
\node[font=\bfseries\small, text=white] at (11.6, 16.7) {
  THE TRIPARTITE ACTION REVERSIBILITY TAXONOMY (AUTHORITY LEVELS $A_1 \to A_3$)
};

% --- PANEL 1: CLASS I - FULLY INVERTIBLE (x: 0.4 to 7.6, y: 11.15 to 16.2) ---
\draw[draw=forestsage, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.4, 11.15) rectangle (7.6, 16.2);
% Header badge
\node[draw=none, fill=forestsage, rounded corners=2pt, inner xsep=4pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b_c1) at (1.9, 15.85) {CLASS I: INVERTIBLE};
\node[anchor=east, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=forestsage] at (7.4, 15.85) {Authority $A_1$ (Sandbox)};

\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=DarkSlate] at (0.55, 15.55) {
  \begin{minipage}{6.8cm}\raggedright
    $\bullet$ \textbf{Formal Contract:} $(T_i^{-1} \circ T_i)(S) = S$ (Identity Transform)\\
    $\bullet$ \textbf{Physical Containment:} Confined to local microVM / sandbox\\
    $\bullet$ \textbf{Mechanisms:} Ephemeral CoW branches, scratch disk, tmpfs\\
    $\bullet$ \textbf{Rollback Engine:} Bitwise reversal via VFS unmount / branch deletion (\texttt{git branch -D}) in $<10\,\text{ms}$\\
    $\bullet$ \textbf{Observable Residual:} \textbf{Zero} (state physically erased)
  \end{minipage}
};
% Example Badge
\draw[draw=forestsage!60, fill=GreenL0, rounded corners=2pt] (0.55, 11.3) rectangle (7.45, 12.05);
\node[font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=forestsage] at (4.0, 11.8) {CONCRETE SYSTEM EXAMPLE};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (4.0, 11.5) {\texttt{git checkout -b fix} $\implies$ \texttt{git branch -D fix} (0 residual)};

% --- PANEL 2: CLASS II - COMPENSABLE (x: 8.0 to 15.2, y: 11.15 to 16.2) ---
\draw[draw=warmochre, fill=white, line width=0.9pt, rounded corners=3.5pt] (8.0, 11.15) rectangle (15.2, 16.2);
% Header badge
\node[draw=none, fill=warmochre, rounded corners=2pt, inner xsep=4pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b_c2) at (9.65, 15.85) {CLASS II: COMPENSABLE};
\node[anchor=east, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=warmochre] at (15.0, 15.85) {Authority $A_2$ (Reversible External)};

\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=DarkSlate] at (8.15, 15.55) {
  \begin{minipage}{6.8cm}\raggedright
    $\bullet$ \textbf{Formal Contract:} $\mathcal{V}_{\text{sem}}(C_i \circ T_i) = \mathcal{V}_{\text{sem}}(I)$ (Semantic Reset)\\
    $\bullet$ \textbf{Physical Reality:} $C_i(T_i(S)) \ne S$ (History cannot be erased)\\
    $\bullet$ \textbf{Mechanisms:} Cloud VM allocation $\to$ delete API; Stripe charge $\to$ refund; soft DB delete\\
    $\bullet$ \textbf{Rollback Engine:} LIFO Saga compensation stack $\mathcal{L}_{\text{comp}}$\\
    $\bullet$ \textbf{Observable Residual:} Immutable audit log, transient bill
  \end{minipage}
};
% Example Badge
\draw[draw=warmochre!60, fill=OrangeL0, rounded corners=2pt] (8.15, 11.3) rectangle (15.05, 12.05);
\node[font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=warmochre] at (11.6, 11.8) {CONCRETE SYSTEM EXAMPLE};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (11.6, 11.5) {\texttt{POST /v1/charges (\$50)} $\implies$ \texttt{POST /v1/refunds (\$50)}};

% --- PANEL 3: CLASS III - IRREVERSIBLE / PIVOT (x: 15.6 to 22.8, y: 11.15 to 16.2) ---
\draw[draw=deepburgundy, fill=RedL0, line width=1.0pt, rounded corners=3.5pt] (15.6, 11.15) rectangle (22.8, 16.2);
% Header badge
\node[draw=none, fill=deepburgundy, rounded corners=2pt, inner xsep=4pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b_c3) at (17.5, 15.85) {CLASS III: PIVOT ACTION};
\node[anchor=east, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (22.6, 15.85) {Authority $A_3$ (Irreversible)};

\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=DarkSlate] at (15.75, 15.55) {
  \begin{minipage}{6.8cm}\raggedright
    $\bullet$ \textbf{Formal Contract:} $C_i = \emptyset$ (\textbf{Point of No Return})\\
    $\bullet$ \textbf{Physical Reality:} Mutation committed across sovereign wires\\
    $\bullet$ \textbf{Mechanisms:} Public Anycast BGP flip, email dispatch, wire transfer settlement, production table drop\\
    $\bullet$ \textbf{Recovery Posture:} \textbf{Rollback Disabled!} Forward repair, idempotent retry, or human operator escrow\\
    $\bullet$ \textbf{Observable Residual:} Permanent change in external world
  \end{minipage}
};
% Example Badge
\draw[draw=deepburgundy!60, fill=white, rounded corners=2pt] (15.75, 11.3) rectangle (22.65, 12.05);
\node[font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=deepburgundy] at (19.2, 11.8) {CONCRETE SYSTEM EXAMPLE};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (19.2, 11.5) {\texttt{BGP Route Announce} $\implies C = \emptyset$ (traffic hits nodes)};

% =============================================================================
% 2. MIDDLE TIER: THE BIFURCATED TRAJECTORY LIFECYCLE TIMELINE
% (x: 0 to 23.2, y: 3.4 to 10.5)
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.1pt, rounded corners=5pt] (0, 3.4) rectangle (23.2, 10.5);

% Header Bar
\fill[DarkSlate, rounded corners=3pt] (0.2, 9.85) rectangle (23.0, 10.35);
\node[font=\bfseries\small, text=white] at (11.6, 10.1) {
  BIFURCATED TRAJECTORY EXECUTION DATAPATH \& PRE-PIVOT VERIFICATION GATE ($\mathcal{G}_{\mathrm{gate}}$)
};

% --- PHASE 1: PREPARATION PHASE (Compensable) (x: 0.4 to 10.2, y: 3.65 to 9.6) ---
\draw[draw=warmochre, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.4, 3.65) rectangle (10.2, 9.6);
% Title badge
\node[draw=none, fill=warmochre, rounded corners=2pt, inner xsep=4pt, inner ysep=2pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] at (2.4, 9.3) {PHASE 1: PREPARATION};
\node[anchor=east, font=\fontsize{6pt}{7pt}\selectfont, text=slatesteel] at (10.0, 9.3) {Backward Compensation Active: $\mathcal{L}_{\mathrm{comp}} \ne \emptyset$};

% Forward Sub-transactions T1, T2, T3 (y: 7.75 to 8.95)
\draw[draw=warmochre, fill=OrangeL0, rounded corners=2pt] (0.6, 7.75) rectangle (3.3, 8.95);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=warmochre] at (1.95, 8.65) {Step $T_1$: Cloud VMs};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (1.95, 8.3) {Provision 16$\times$ H100s};
\node[font=\fontsize{4.8pt}{5.8pt}\selectfont, text=warmochre] at (1.95, 7.95) {Class II $\cdot$ \$5.50/min};

\draw[draw=forestsage, fill=GreenL0, rounded corners=2pt] (3.75, 7.75) rectangle (6.45, 8.95);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=forestsage] at (5.1, 8.65) {Step $T_2$: Warm Cache};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (5.1, 8.3) {KV PagedAttention};
\node[font=\fontsize{4.8pt}{5.8pt}\selectfont, text=forestsage] at (5.1, 7.95) {Class I $\cdot$ Local Compute};

\draw[draw=warmochre, fill=OrangeL0, rounded corners=2pt] (6.9, 7.75) rectangle (10.0, 8.95);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=warmochre] at (8.45, 8.65) {Step $T_3$: Schema Escrow};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (8.45, 8.3) {Provisional DB Migration};
\node[font=\fontsize{4.8pt}{5.8pt}\selectfont, text=warmochre] at (8.45, 7.95) {Class II $\cdot$ Rollable Tx};

% Forward progression arrows
\draw[flow_arrow, draw=warmochre] (3.3, 8.35) -- (3.75, 8.35);
\draw[flow_arrow, draw=warmochre] (6.45, 8.35) -- (6.9, 8.35);

% Reverse Compensating Handlers C1, C2, C3 (y: 6.35 to 7.45)
\draw[draw=slatesteel, fill=coolparchment, rounded corners=2pt] (0.6, 6.35) rectangle (3.3, 7.45);
\node[font=\bfseries\fontsize{5.8pt}{6.8pt}\selectfont, text=slatesteel] at (1.95, 7.15) {Handler $C_1$: Teardown};
\node[font=\fontsize{5pt}{6pt}\selectfont, text=DarkSlate] at (1.95, 6.8) {\texttt{DELETE /instances}};
\node[font=\fontsize{4.6pt}{5.6pt}\selectfont, text=slatesteel] at (1.95, 6.5) {Stops billing delta};

\draw[draw=forestsage, fill=coolparchment, rounded corners=2pt] (3.75, 6.35) rectangle (6.45, 7.45);
\node[font=\bfseries\fontsize{5.8pt}{6.8pt}\selectfont, text=forestsage] at (5.1, 7.15) {Handler $C_2$: Discard};
\node[font=\fontsize{5pt}{6pt}\selectfont, text=DarkSlate] at (5.1, 6.8) {Flush GPU memory};
\node[font=\fontsize{4.6pt}{5.6pt}\selectfont, text=forestsage] at (5.1, 6.5) {Free KV cache buffers};

\draw[draw=slatesteel, fill=coolparchment, rounded corners=2pt] (6.9, 6.35) rectangle (10.0, 7.45);
\node[font=\bfseries\fontsize{5.8pt}{6.8pt}\selectfont, text=slatesteel] at (8.45, 7.15) {Handler $C_3$: Rollback};
\node[font=\fontsize{5pt}{6pt}\selectfont, text=DarkSlate] at (8.45, 6.8) {\texttt{ROLLBACK} escrow tx};
\node[font=\fontsize{4.6pt}{5.6pt}\selectfont, text=slatesteel] at (8.45, 6.5) {Restore catalog state};

% Reverse Compensation Unwind arrows (Dashed leftward)
\draw[flow_arrow, draw=deepburgundy, dashed, line width=1.1pt] (6.9, 6.9) -- (6.45, 6.9);
\draw[flow_arrow, draw=deepburgundy, dashed, line width=1.1pt] (3.75, 6.9) -- (3.3, 6.9);
\node[anchor=south, font=\bfseries\fontsize{4.8pt}{5.8pt}\selectfont, text=deepburgundy] at (6.67, 6.95) {Unwind};
\node[anchor=south, font=\bfseries\fontsize{4.8pt}{5.8pt}\selectfont, text=deepburgundy] at (3.52, 6.95) {Unwind};

% LIFO Compensation Policy Box (y: 3.8 to 6.1)
\draw[draw=slatesteel!60, fill=coolparchment, rounded corners=2.5pt] (0.6, 3.85) rectangle (10.0, 6.1);
\node[anchor=west, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=DarkSlate] at (0.8, 5.85) {
  LIFO Compensation Log $\mathcal{L}_{\mathrm{comp}} = [C_3, C_2, C_1]$ (Drained in Reverse):
};
\node[anchor=north west, font=\fontsize{5.8pt}{7.2pt}\selectfont, text=slatesteel] at (0.8, 5.55) {
  \begin{minipage}{9.0cm}\raggedright
    $\blacktriangleright$ \textbf{Failure Trigger:} If error occurs at $T_3$, drain $\mathcal{L}_{\mathrm{comp}}$ in reverse: $C_3 \to C_2 \to C_1$\\
    $\blacktriangleright$ \textbf{Baseline Invariant:} System returns to baseline $S_0$ without residual state\\
    $\blacktriangleright$ \textbf{Containment:} Zero external mutations escape prior to verification gate
  \end{minipage}
};

% --- CENTER: PRE-PIVOT VERIFICATION GATE (x: 10.5 to 15.5, y: 3.65 to 9.6) ---
\draw[draw=amethyst, fill=amethystL0, line width=1.2pt, rounded corners=4pt] (10.5, 3.65) rectangle (15.5, 9.6);

% Gate Title
\fill[amethyst, rounded corners=2pt] (10.7, 9.15) rectangle (15.3, 9.5);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=white] at (13.0, 9.32) {
  PRE-PIVOT GATE $\mathcal{G}_{\mathrm{gate}}$
};

\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=amethyst] at (13.0, 8.85) {
  THE ATOMIC THRESHOLD
};

\node[anchor=north west, font=\fontsize{5.8pt}{7.2pt}\selectfont, text=DarkSlate] at (10.7, 8.6) {
  \begin{minipage}{4.6cm}\raggedright
    \textbf{1. Deterministic Checks:}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item Verify pre-conditions satisfied
      \item Validate state delta $\Delta S$
      \item Assert leased resources healthy
    \end{itemize}
    \vspace{2pt}
    \textbf{2. Durable Outbox Staging ($e_{\mathrm{outbox}}$):}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item Synthesize HMAC Idempotency Key $\kappa_{\mathrm{pivot}}$
      \item Force non-volatile \texttt{fdatasync()} barrier
      \item Log \texttt{STATUS\_COMMITTED} in WAL
    \end{itemize}
    \vspace{2pt}
    \textbf{3. Optional Human Escrow ($A_3$):}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item Cryptographic supervisor signature
    \end{itemize}
  \end{minipage}
};

% Point of No Return Bar
\draw[draw=deepburgundy, fill=RedL!40, rounded corners=2pt] (10.7, 3.85) rectangle (15.3, 4.6);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (13.0, 4.35) {POINT OF NO RETURN};
\node[font=\fontsize{5pt}{6pt}\selectfont, text=deepburgundy] at (13.0, 4.05) {$\mathcal{L}_{\mathrm{comp}} \gets \emptyset$ (Rollback Terminated)};

% --- PHASE 2: FINALIZATION PHASE (Idempotent / Forward) (x: 15.8 to 22.8, y: 3.65 to 9.6) ---
\draw[draw=forestsage, fill=white, line width=0.9pt, rounded corners=3.5pt] (15.8, 3.65) rectangle (22.8, 9.6);
% Title badge
\node[draw=none, fill=deepburgundy, rounded corners=2pt, inner xsep=4pt, inner ysep=2pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] at (17.9, 9.3) {PHASE 2: FINALIZATION};
\node[anchor=east, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (22.6, 9.3) {Rollback Disabled};

% Pivot Execution Step (y: 7.75 to 8.95)
\draw[draw=deepburgundy, fill=RedL0, line width=1.1pt, rounded corners=2pt] (16.0, 7.75) rectangle (19.1, 8.95);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (17.55, 8.65) {Step $T_k = T_{\mathrm{pivot}}$};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (17.55, 8.3) {Activate Anycast BGP};
\node[font=\bfseries\fontsize{4.8pt}{5.8pt}\selectfont, text=deepburgundy] at (17.55, 7.95) {Class III $\cdot$ Public Traffic};

% Retriable final step
\draw[draw=forestsage, fill=GreenL0, rounded corners=2pt] (19.5, 7.75) rectangle (22.6, 8.95);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=forestsage] at (21.05, 8.65) {Step $T_{k+1}$: Telemetry};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (21.05, 8.3) {Flush Audit Metrics};
\node[font=\fontsize{4.8pt}{5.8pt}\selectfont, text=forestsage] at (21.05, 7.95) {Idempotent Retry};

% Flow between pivot and next
\draw[flow_arrow, draw=forestsage] (19.1, 8.35) -- (19.5, 8.35);

% Non-existent compensator box (y: 6.35 to 7.45)
\draw[draw=deepburgundy, fill=RedL0, dashed, rounded corners=2pt] (16.0, 6.35) rectangle (22.6, 7.45);
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (19.3, 7.15) {NO INVERSE OPERATION EXISTS ($C_{\mathrm{pivot}} = \emptyset$)};
\node[font=\fontsize{5.2pt}{6.2pt}\selectfont, text=DarkSlate] at (19.3, 6.8) {Public users already routed to nodes; bytes permanently released};
\node[font=\bfseries\fontsize{5pt}{6pt}\selectfont, text=deepburgundy] at (19.3, 6.5) {Backward Rollback Physically Impossible!};

% Finalization Rules Box (y: 3.8 to 6.1)
\draw[draw=slatesteel!60, fill=coolparchment, rounded corners=2.5pt] (16.0, 3.85) rectangle (22.6, 6.1);
\node[anchor=west, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=DarkSlate] at (16.2, 5.85) {
  Guaranteed Completion Rules ($i \ge k$):
};
\node[anchor=north west, font=\fontsize{5.8pt}{7.2pt}\selectfont, text=slatesteel] at (16.2, 5.55) {
  \begin{minipage}{6.2cm}\raggedright
    $\blacktriangleright$ \textbf{Network Timeout on $T_{\mathrm{pivot}}$:} Launch out-of-band \textbf{Reconciliation Probe} using key $\kappa_{\mathrm{pivot}}$\\
    $\blacktriangleright$ \textbf{Never Blind Retry:} Prevents duplicate billing/mutations\\
    $\blacktriangleright$ \textbf{Post-Pivot Mandate:} Monotonic forward self-healing, deterministic fallback, or freeze ACB for human operator
  \end{minipage}
};

% Connecting Arrows Across Gate
\draw[flow_arrow_thick, draw=warmochre] (10.0, 8.35) -- (10.5, 8.35);
\draw[flow_arrow_thick, draw=deepburgundy] (15.5, 8.35) -- (16.0, 8.35);

% =============================================================================
% 3. BOTTOM TIER: THE DUAL-COMMIT TRAP vs. TOPOLOGICAL SUPERVISOR COMPILATION
% (x: 0 to 23.2, y: 0.5 to 3.1)
% =============================================================================
\draw[fill=DarkSlate, draw=none, rounded corners=4pt] (0, 0.5) rectangle (23.2, 3.1);

% Left Box: The Dual-Commit Anti-Pattern
\draw[draw=RedLine!80, fill=ChipBlack, rounded corners=3pt] (0.3, 1.05) rectangle (11.2, 2.6);
\node[anchor=north west, font=\bfseries\scriptsize, text=RedLine] at (0.45, 2.45) {THE DUAL-COMMIT TRAP (Interleaved Pivots: Catastrophic State Hybrid)};
\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=white!85] at (0.45, 2.05) {
  \begin{minipage}{10.5cm}\raggedright
    $\bullet$ \textbf{Pathology:} $T_1 \to T_2 \to \mathbf{T_3\text{ (Pivot: Live BGP)}} \to \mathbf{T_4\text{ (DB Deadlock FAIL)}} \to T_5\text{ (Audit)}$.\\
    $\bullet$ \textbf{Deadlock:} Cannot roll back $T_3$ (public users hit cluster); cannot reach $T_5$ (audit never sent)!\\
    $\bullet$ \textbf{Result:} System stranded in an unrecoverable hybrid state: partially published, unmonitored.
  \end{minipage}
};

% Right Box: Topological Compilation
\draw[draw=GreenLine!80, fill=ChipBlack, rounded corners=3pt] (11.8, 1.05) rectangle (22.9, 2.6);
\node[anchor=north west, font=\bfseries\scriptsize, text=GreenLine] at (11.95, 2.45) {TOPOLOGICAL SUPERVISOR COMPILATION (Strict Pivot Boundary)};
\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=white!85] at (11.95, 2.05) {
  \begin{minipage}{10.7cm}\raggedright
    $\bullet$ \textbf{Remedy:} Reorder action DAG: stage all Class I \& II mutations in local memory escrow.\\
    $\bullet$ \textbf{Execution:} $T_1 \to T_2 \to T_4'(\text{Escrow DB}) \to \mathcal{G}_{\mathrm{gate}} \to \mathbf{T_{\mathrm{pivot}}\text{ (BGP Flip + Outbox Flush)}} \to T_5$.\\
    $\bullet$ \textbf{Guarantee:} If DB fails, rollback instances via $C_1$ ($0$ traffic dropped, $0$ invalid audit logs).
  \end{minipage}
};

% Title Strip inside bus at the BOTTOM
\fill[ChipBlack, rounded corners=2pt] (0.3, 0.58) rectangle (22.9, 0.95);
\node[anchor=west, font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=GoldPin] at (0.45, 0.77) {
  THE PIVOT BOUNDARY LAW
};
\node[anchor=east, font=\fontsize{6pt}{7.2pt}\selectfont, text=white!80] at (22.75, 0.77) {
  $|\{T \in \mathcal{T} : C(T) = \emptyset\}| \le 1$. Reversibility is a physical invariant of the target environment, not the prompt.
};

\end{tikzpicture}
\end{document}
"""

repo_root = Path("/Users/VJ/GitHub/MLSysBook-vol3-review")
tex_file = repo_root / "books/vol3/tex/tripartite_action_reversibility.tex"

with open(tex_file, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-halt-on-error", "-output-directory=/tmp", str(tex_file)], capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/tripartite_action_reversibility.pdf", "/tmp/tripartite_reversibility_render"])
    subprocess.run(["pdftocairo", "-svg", "/tmp/tripartite_action_reversibility.pdf", "/tmp/tripartite_action_reversibility.svg"])
    print("Successfully compiled and rendered Tripartite Action Reversibility!")
else:
    print(res.stdout[-1500:])
