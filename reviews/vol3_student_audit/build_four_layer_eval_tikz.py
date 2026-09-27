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
  THE FOUR-LAYER EVALUATION CONTRACT
};
\node[anchor=east, font=\footnotesize, text=GoldPin] at (22.75, 18.4) {
  Chapter 13: Evaluation \& Observability $\cdot$ Section 13.1
};
\node[anchor=west, font=\scriptsize, text=white!85] at (0.45, 17.85) {
  The Fallacy of Green Dashboards: Why Superficial Telemetry (Exit 0, HTTP 200, 5/5 Judge) Masks Physical State Corruption
};

% =============================================================================
% 1. LEFT MAIN COLUMN: THE FOUR-LAYER EVALUATION CONTRACT HIERARCHY
% (x: 0 to 14.2, y: 3.5 to 17.1) - Width 14.2 cm
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.1pt, rounded corners=5pt] (0, 3.5) rectangle (14.2, 17.1);

% Header Bar
\fill[DarkSlate, rounded corners=3pt] (0.2, 16.45) rectangle (14.0, 16.95);
\node[font=\bfseries\small, text=white] at (7.1, 16.7) {
  THE STRICT HIERARCHICAL EVALUATION CONTRACT ($L_1 \longrightarrow L_4$)
};

% Sub-banner explaining short-circuit rule
\node[font=\fontsize{6.5pt}{7.5pt}\selectfont, text=slatesteel] at (7.1, 16.15) {
  Non-Fungible Hierarchy: Failure at Any Layer Invalidates Higher Layers $\cdot$ Evaluated in Isolated Escrow
};

% --- LAYER 1: SYNTACTIC VALIDITY (y: 12.9 to 15.8) ---
\draw[draw=forestsage, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.4, 12.9) rectangle (13.8, 15.8);
% Badge (Auto-sized node)
\node[draw=none, fill=forestsage, rounded corners=2pt, inner xsep=4.5pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b1) at (1.65, 15.45) {LAYER 1: SYNTAX};
\node[anchor=west, font=\bfseries\scriptsize, text=DarkSlate] at (b1.east) [xshift=5pt] {In-Memory JSON Schema \& AST Validator};
\node[anchor=east, font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=forestsage] at (13.6, 15.45) {$c_1 = 0.4\,\text{ms} \quad \text{PASS \greenpass{}}$};

\node[anchor=north west, font=\fontsize{6.5pt}{8.2pt}\selectfont, text=DarkSlate] at (0.55, 15.1) {
  \begin{minipage}{9.6cm}\raggedright
    $\bullet$ \textbf{Verification Target:} In-memory payload intercepted before environment dispatch\\
    $\bullet$ \textbf{Audit Action:} Validates \texttt{run\_migration(shard="shard-04", batch=5000)}\\
    $\bullet$ \textbf{Mechanism:} JSON Schema validator verifies non-dangling delimiters and typing\\
    $\bullet$ \textbf{Observation:} Flawless AST conformity; arguments within allowed bounds $[100, 10000]$
  \end{minipage}
};
% Status Box
\draw[draw=forestsage, fill=GreenL0, rounded corners=2pt] (10.4, 13.1) rectangle (13.6, 14.9);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=forestsage] at (12.0, 14.55) {SYNTACTIC PASS};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=DarkSlate] at (12.0, 14.1) {Schema Validated};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=forestsage] at (12.0, 13.65) {AST Conformity: $100\%$};
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=forestsage] at (12.0, 13.3) {Advances to $L_2$};

% Downward Flow Arrow
\draw[flow_arrow, draw=forestsage] (7.1, 12.9) -- (7.1, 12.5);

% --- LAYER 2: EXECUTION INVARIANTS (y: 9.6 to 12.5) ---
\draw[draw=deepburgundy, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.4, 9.6) rectangle (13.8, 12.5);
% Badge (Auto-sized node)
\node[draw=none, fill=deepburgundy, rounded corners=2pt, inner xsep=4.5pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b2) at (1.8, 12.15) {LAYER 2: INVARIANTS};
\node[anchor=west, font=\bfseries\scriptsize, text=DarkSlate] at (b2.east) [xshift=5pt] {Process Hygiene \& Stderr Stream Monitor};
\node[anchor=east, font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=deepburgundy] at (13.6, 12.15) {\texttt{exit==0} $\quad \mathbf{BREACH\text{ }\redfail{}}$};

\node[anchor=north west, font=\fontsize{6.5pt}{8.2pt}\selectfont, text=DarkSlate] at (0.55, 11.8) {
  \begin{minipage}{9.6cm}\raggedright
    $\bullet$ \textbf{Verification Target:} Tool execution runtime, socket handlers, and signal hygiene\\
    $\bullet$ \textbf{Superficial Metric:} Shell wrapper returns operating system \texttt{exit == 0}\\
    $\bullet$ \textbf{Hidden Violation:} \texttt{stderr} warning: \textit{Lock acquisition timed out after 30s}\\
    $\bullet$ \textbf{Operational Impact:} Fallback held \texttt{AccessExclusiveLock} for $41.2\,\text{s}$, blocking 1,840 queries
  \end{minipage}
};
% Status Box
\draw[draw=deepburgundy, fill=RedL0, rounded corners=2pt] (10.4, 9.8) rectangle (13.6, 11.6);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=deepburgundy] at (12.0, 11.25) {INVARIANT BREACH};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=DarkSlate] at (12.0, 10.8) {Exclusive Lock Held};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=deepburgundy] at (12.0, 10.35) {1,840 Queries Stalled};
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (12.0, 10.0) {Swallowed Warning};

% Downward Flow Arrow
\draw[flow_arrow, draw=deepburgundy] (7.1, 9.6) -- (7.1, 9.2);

% --- LAYER 3: STATE DELTA VERIFICATION (y: 6.3 to 9.2) ---
\draw[draw=deepburgundy, fill=RedL!30, line width=1.1pt, rounded corners=3.5pt] (0.4, 6.3) rectangle (13.8, 9.2);
% Badge (Auto-sized node)
\node[draw=none, fill=deepburgundy, rounded corners=2pt, inner xsep=4.5pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b3) at (1.85, 8.85) {LAYER 3: STATE DELTA};
\node[anchor=west, font=\bfseries\scriptsize, text=deepburgundy] at (b3.east) [xshift=5pt] {Physical Mutation: $\Delta S = S_{\text{post}} - S_{\text{pre}}$};
\node[anchor=east, font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=deepburgundy] at (13.6, 8.85) {Score: $0.0 \quad \mathbf{FAIL\text{ }\redfail{}}$};

\node[anchor=north west, font=\fontsize{6.5pt}{8.2pt}\selectfont, text=DarkSlate] at (0.55, 8.5) {
  \begin{minipage}{9.6cm}\raggedright
    $\bullet$ \textbf{Verification Target:} Persistent database relations, tables, and transactional integrity\\
    $\bullet$ \textbf{Sealed Query:} \texttt{SELECT count(*) FROM audit\_events WHERE status IS NULL;}\\
    $\bullet$ \textbf{Assertion Failure:} Expected 0 $\implies$ \textbf{Returned 3,580,000 NULL rows!}\\
    $\bullet$ \textbf{Root Cause:} Statement timeout at row 8.42M; missing atomic savepoint backfill
  \end{minipage}
};
% Status Box
\draw[draw=deepburgundy, fill=white, rounded corners=2pt] (10.4, 6.5) rectangle (13.6, 8.3);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=deepburgundy] at (12.0, 7.95) {STATE CORRUPTION};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=deepburgundy] at (12.0, 7.5) {\textbf{3.58M NULL Rows}};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=DarkSlate] at (12.0, 7.05) {Partial Write Failure};
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] at (12.0, 6.7) {Ground Truth: FAIL};

% Downward Flow Arrow
\draw[flow_arrow, draw=deepburgundy] (7.1, 6.3) -- (7.1, 5.9);

% --- LAYER 4: CONSTRAINT COMPLIANCE (y: 3.8 to 5.9) ---
\draw[draw=warmochre, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.4, 3.8) rectangle (13.8, 5.9);
% Badge (Auto-sized node)
\node[draw=none, fill=warmochre, rounded corners=2pt, inner xsep=4.5pt, inner ysep=2.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=white] (b4) at (1.85, 5.55) {LAYER 4: CONSTRAINTS};
\node[anchor=west, font=\bfseries\scriptsize, text=DarkSlate] at (b4.east) [xshift=5pt] {Resource Ceilings \& Sandboxed Isolation};
\node[anchor=east, font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=warmochre] at (13.6, 5.55) {Latency: $+18.8\% \quad \mathbf{FAIL\text{ }\redfail{}}$};

\node[anchor=north west, font=\fontsize{6.5pt}{8.2pt}\selectfont, text=DarkSlate] at (0.55, 5.25) {
  \begin{minipage}{9.6cm}\raggedright
    $\bullet$ \textbf{Latency Ceiling:} $T_{\text{task}} = 142.6\,\text{s} > T_{\max} = 120.0\,\text{s}$ (\textbf{+22.6 s / +18.8\% Overrun})\\
    $\bullet$ \textbf{Token Consumption:} $C_{\text{task}} = 17{,}350\text{ tokens} \le C_{\max} = 25{,}000\text{ tokens}$ (\textbf{PASS \greenpass{}})\\
    $\bullet$ \textbf{Step Budget:} $N = 6\text{ steps} \le N_{\max} = 10\text{ steps}$ (\textbf{PASS \greenpass{}})
  \end{minipage}
};
% Status Box
\draw[draw=warmochre, fill=OrangeL0, rounded corners=2pt] (10.4, 4.0) rectangle (13.6, 5.2);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=warmochre] at (12.0, 4.85) {BUDGET OVERRUN};
\node[font=\fontsize{5.5pt}{6.5pt}\selectfont, text=DarkSlate] at (12.0, 4.45) {Latency: $+18.8\%$};
\node[font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=warmochre] at (12.0, 4.15) {SLA Breached};

% =============================================================================
% 2. RIGHT COLUMN: THE FALLACY OF GREEN DASHBOARDS
% (x: 14.8 to 23.2, y: 3.5 to 17.1) - Width 8.4 cm
% =============================================================================
\draw[draw=slatesteel, fill=coolparchment, line width=1.1pt, rounded corners=5pt] (14.8, 3.5) rectangle (23.2, 17.1);

% Header Bar
\fill[DarkSlate, rounded corners=3pt] (15.0, 16.45) rectangle (23.0, 16.95);
\node[font=\bfseries\small, text=white] at (19.0, 16.7) {
  THE FALLACY OF GREEN DASHBOARDS
};

% Sub-banner
\node[font=\fontsize{6.5pt}{7.5pt}\selectfont, text=slatesteel] at (19.0, 16.15) {
  Superficial Observability vs. Ground-Truth Production Reality
};

% --- CARD 1: SUPERFICIAL APM & LLM JUDGE DASHBOARD (y: 10.3 to 15.8) ---
\draw[draw=forestsage, fill=white, line width=1.0pt, rounded corners=3.5pt] (15.05, 10.3) rectangle (22.95, 15.8);
\fill[forestsage, rounded corners=2pt] (15.2, 15.35) rectangle (22.8, 15.7);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=white] at (19.0, 15.52) {
  WHAT THE STANDARD DASHBOARD SEES: "ALL GREEN"
};

\node[anchor=north west, font=\fontsize{6.5pt}{8.2pt}\selectfont, text=DarkSlate] at (15.2, 15.2) {
  \begin{minipage}{7.5cm}\raggedright
    \textbf{1. APM Fleet Telemetry:}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item HTTP 200 Availability: $\mathbf{99.999\%}$ (Five-Nines) \greenpass{}
      \item Process Return Status: \texttt{exit == 0} (Success) \greenpass{}
      \item RPC Heartbeat: Nominal / Active \greenpass{}
    \end{itemize}
    \vspace{2pt}
    \textbf{2. Agent Self-Reported Transcript:}
    \begin{quote}\setlength{\parskip}{0pt}\itshape\fontsize{6pt}{7.2pt}\selectfont
      "Online schema migration completed successfully. All 12,000,000 records backfilled and composite index created with zero downtime."
    \end{quote}
    \vspace{2pt}
    \textbf{3. LLM-as-a-Judge Evaluation:}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item Evaluator Rating: $\mathbf{5 / 5\text{ \textbf{Stars}}}$ (Excellent) \greenpass{}
      \item Rationale: \textit{"Clear procedural reasoning, polite tone, followed safe DB migration best practices."}
    \end{itemize}
  \end{minipage}
};

% Downward Warning Lightning
\draw[flow_arrow_thick, draw=deepburgundy] (19.0, 10.3) -- (19.0, 9.8);
\node[draw=deepburgundy, fill=RedL0, rounded corners=2pt, inner sep=1.5pt, font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=deepburgundy] 
  at (19.0, 10.05) {SHATTERED BY EMPIRICAL REALITY};

% --- CARD 2: FORENSIC PRODUCTION REALITY (y: 3.8 to 9.7) ---
\draw[draw=deepburgundy, fill=white, line width=1.0pt, rounded corners=3.5pt] (15.05, 3.8) rectangle (22.95, 9.7);
\fill[deepburgundy, rounded corners=2pt] (15.2, 9.25) rectangle (22.8, 9.6);
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=white] at (19.0, 9.42) {
  THE FORENSIC GROUND-TRUTH REALITY: "FATAL FAILURE"
};

\node[anchor=north west, font=\fontsize{6.5pt}{8.2pt}\selectfont, text=DarkSlate] at (15.2, 9.15) {
  \begin{minipage}{7.5cm}\raggedright
    \textbf{1. Catastrophic State Corruption ($\Delta S \ne \Delta S^*$):}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item $\mathbf{3{,}580{,}000\text{ NULL rows}}$ left uninitialized ($29.8\%$ of table)!
      \item Downstream billing queries throw fatal type errors.
      \item Ground-truth task correctness: $\mathbf{0.0 / 1.0}$ (\textbf{FAIL} \redfail{}).
    \end{itemize}
    \vspace{2pt}
    \textbf{2. Hidden Production Outage:}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item Exclusive table lock (\texttt{AccessExclusiveLock}) held for $\mathbf{41.2\text{ seconds}}$.
      \item $\mathbf{1{,}840\text{ incoming queries}}$ queued in pool exhaustion.
    \end{itemize}
    \vspace{2pt}
    \textbf{3. Latency SLA Overrun:}
    \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}
      \item Duration: $142.6\,\text{s} > 120.0\,\text{s}$ ($\mathbf{+18.8\%}$ over budget).
    \end{itemize}
    \vspace{2pt}
    \textbf{4. Net Audit Outcome:} $\mathbf{UNACCEPTED\text{ }TASK}$ \redfail{}
  \end{minipage}
};

% Inter-column Connecting Pointer (Layer 3 State Delta to Forensic Card)
\draw[flow_arrow, draw=deepburgundy, dashed, line width=1.4pt] 
  (13.8, 7.75) -- (15.05, 7.75);

% =============================================================================
% 3. BOTTOM TIER: CORE PRINCIPLES OF AUTONOMOUS AGENT EVALUATION
% (x: 0 to 23.2, y: 0.5 to 3.0)
% =============================================================================
\draw[fill=DarkSlate, draw=none, rounded corners=4pt] (0, 0.5) rectangle (23.2, 3.0);

% Box 1: Invariant Closure (Left)
\draw[draw=forestsage, fill=ChipBlack, rounded corners=3pt] (0.3, 0.95) rectangle (7.5, 2.5);
\node[anchor=north west, font=\bfseries\scriptsize, text=GreenLine] at (0.45, 2.35) {1. INVARIANT CLOSURE PRINCIPLE};
\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=white!85] at (0.45, 1.95) {
  \begin{minipage}{6.8cm}\raggedright
    Never verify tokens when you can verify physical environment state deltas ($\Delta S = S_{\text{post}} - S_{\text{pre}}$). An agent's deliverable is its environment mutation, not its persuasive narrative.
  \end{minipage}
};

% Box 2: Fail-Plausible Nature (Center)
\draw[draw=warmochre, fill=ChipBlack, rounded corners=3pt] (7.9, 0.95) rectangle (15.3, 2.5);
\node[anchor=north west, font=\bfseries\scriptsize, text=GoldPin] at (8.05, 2.35) {2. THE FAIL-PLAUSIBLE PARADOX};
\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=white!85] at (8.05, 1.95) {
  \begin{minipage}{7.0cm}\raggedright
    Stochastic models generate polite, syntactically fluent explanations even when underlying operations fail. LLM-as-a-judge scores rhetorical plausibility, adding zero closure evidence of its own.
  \end{minipage}
};

% Box 3: Non-Fungible Hierarchy (Right)
\draw[draw=RedLine, fill=ChipBlack, rounded corners=3pt] (15.7, 0.95) rectangle (22.9, 2.5);
\node[anchor=north west, font=\bfseries\scriptsize, text=RedLine] at (15.85, 2.35) {3. STRICT SHORT-CIRCUITING};
\node[anchor=north west, font=\fontsize{6.2pt}{7.8pt}\selectfont, text=white!85] at (15.85, 1.95) {
  \begin{minipage}{6.8cm}\raggedright
    Evaluation layers are non-fungible. Passing $L_1$ and $L_2$ confers zero credit if $L_3$ (state delta) or $L_4$ (resource ceilings) fails. An accepted task requires passing all 4 layers simultaneously.
  \end{minipage}
};

% Title Strip inside bus at the BOTTOM
\fill[ChipBlack, rounded corners=2pt] (0.3, 0.58) rectangle (22.9, 0.88);
\node[anchor=west, font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=GoldPin] at (0.45, 0.73) {
  EVALUATION CONTRACT LAW
};
\node[anchor=east, font=\fontsize{6pt}{7.2pt}\selectfont, text=white!80] at (22.75, 0.73) {
  $\mathrm{AcceptedTask} \iff \mathrm{Pass}(L_1) \land \mathrm{Pass}(L_2) \land \mathrm{Pass}(L_3) \land \mathrm{Pass}(L_4)$. Fluency without state closure is illusion.
};

\end{tikzpicture}
\end{document}
"""

repo_root = Path("/Users/VJ/GitHub/MLSysBook-vol3-review")
tex_file = repo_root / "books/vol3/tex/four_layer_evaluation_contract.tex"

with open(tex_file, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-halt-on-error", "-output-directory=/tmp", str(tex_file)], capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/four_layer_evaluation_contract.pdf", "/tmp/four_layer_eval_render"])
    subprocess.run(["pdftocairo", "-svg", "/tmp/four_layer_evaluation_contract.pdf", "/tmp/four_layer_evaluation_contract.svg"])
    print("Successfully compiled and rendered Four-Layer Evaluation Contract!")
else:
    print(res.stdout[-1500:])
