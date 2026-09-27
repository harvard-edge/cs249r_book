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
\definecolor{BlueDD}{RGB}{20,50,80}
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
\definecolor{GrayLine}{HTML}{64748B}
\definecolor{coolparchment}{HTML}{F8FAFC}
\definecolor{CardBorder}{HTML}{CBD5E1}
\definecolor{ChipBlack}{HTML}{181F2C}
\definecolor{GoldPin}{HTML}{EAB308}

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
% TOP TITLE BANNER (y: 17.2 to 18.5)
% =============================================================================
\draw[fill=DarkSlate, draw=none, rounded corners=4pt] (0, 17.1) rectangle (23.2, 18.5);
\node[anchor=west, font=\bfseries\large, text=white] at (0.45, 18.0) {
  THE DUAL-SANDBOX VERIFICATION ENCLAVE DATAPATH
};
\node[anchor=east, font=\footnotesize, text=GoldPin] at (22.75, 18.0) {
  Chapter 16: Reinforcement Learning with Verifiable Rewards (RLVR)
};
\node[anchor=west, font=\scriptsize, text=white!85] at (0.45, 17.45) {
  Hardware-Enforced Anti-Reward-Hacking Architecture: Untrusted Rollout $\to$ Invariant Airlock $\to$ Clean-Room MicroVM $\to$ GRPO Update Bus
};

% =============================================================================
% 1. LEFT CONTAINER: CHAMBER 1 (UNTRUSTED POLICY WORKSPACE)
% (x: 0 to 6.8, y: 3.5 to 16.7) - Width 6.8 cm
% =============================================================================
\draw[draw=OrangeLine, fill=OrangeL0, line width=1.2pt, rounded corners=5pt] (0, 3.5) rectangle (6.8, 16.7);

% Header Bar
\fill[OrangeLine, rounded corners=3pt] (0.2, 15.95) rectangle (6.6, 16.5);
\node[font=\bfseries\small, text=white] at (3.4, 16.22) {
  CHAMBER 1: UNTRUSTED WORKSPACE
};
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=warmochre] at (3.4, 15.65) {
  Untrusted Execution $\cdot$ Ephemeral CoW Overlay $\cdot$ Loopback
};

% Block 1.1: Policy Model Rollout (y: 13.0 to 15.25)
\draw[draw=OrangeLine, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.3, 13.0) rectangle (6.5, 15.25);
\node[anchor=north west, font=\bfseries\footnotesize, text=OrangeLine!90!black] at (0.45, 15.1) {1. Policy Model Rollout ($\pi_\theta$)};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (0.45, 14.6) {
  \begin{minipage}{5.8cm}\raggedright
    $\bullet$ Generates exploratory actions $a_t \sim \pi_\theta(\cdot \mid c_t)$\\
    $\bullet$ Dispatches tool calls: \texttt{bash}, \texttt{edit\_file}, \texttt{python}\\
    $\bullet$ \textbf{Threat:} Test-file mocking, assertion removal, evasion
  \end{minipage}
};

% Block 1.2: Ephemeral Filesystem & Public Tests (y: 10.45 to 12.7)
\draw[draw=slatesteel, fill=white, line width=0.9pt, rounded corners=3.5pt] (0.3, 10.45) rectangle (6.5, 12.7);
\node[anchor=north west, font=\bfseries\footnotesize, text=slatesteel] at (0.45, 12.55) {2. Ephemeral OverlayFS \& Public Tests};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (0.45, 12.05) {
  \begin{minipage}{5.8cm}\raggedright
    $\bullet$ Read-only base repository mount (\texttt{git HEAD})\\
    $\bullet$ Mutable upper layer discarded after rollout turn\\
    $\bullet$ Visible public unit tests ($T_{\text{public}}$) only\\
    $\bullet$ Model can modify tests to falsely force exit code 0!
  \end{minipage}
};

% Block 1.3: Network Isolation Box (y: 7.9 to 10.15)
\draw[draw=RedLine, fill=RedL0, line width=0.9pt, rounded corners=3.5pt] (0.3, 7.9) rectangle (6.5, 10.15);
\node[anchor=north west, font=\bfseries\footnotesize, text=deepburgundy] at (0.45, 10.0) {3. Severed Network Namespace};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (0.45, 9.5) {
  \begin{minipage}{5.8cm}\raggedright
    $\bullet$ Strict network namespace isolation (\texttt{ip netns})\\
    $\bullet$ Loopback only (\texttt{lo: 127.0.0.1}); external WAN severed\\
    $\bullet$ Blocked: HTTP, DNS, reverse shells, oracle probing\\
    $\bullet$ Zero credential leaks or out-of-band telemetry
  \end{minipage}
};

% Block 1.4: Text Diff Extraction (y: 3.8 to 7.55)
\draw[draw=warmochre, fill=OrangeL!35, line width=1.0pt, rounded corners=3.5pt] (0.3, 3.8) rectangle (6.5, 7.55);
\node[anchor=north west, font=\bfseries\footnotesize, text=warmochre] at (0.45, 7.4) {4. Candidate Patch Extraction};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (0.45, 6.9) {
  \begin{minipage}{5.8cm}\raggedright
    $\bullet$ Command: \texttt{git diff -{-}no-ext-diff > patch.diff}\\
    $\bullet$ Only raw text diff crosses the isolation boundary\\
    $\bullet$ Chamber 1 is killed and destroyed immediately!
  \end{minipage}
};
% Mini simulated diff badge inside 1.4
\draw[draw=CardBorder, fill=DarkSlate, rounded corners=2pt] (0.5, 4.0) rectangle (6.3, 5.3);
\node[anchor=north west, font=\ttfamily\fontsize{6pt}{7.5pt}\selectfont, text=white] at (0.65, 5.15) {diff -{-}git a/model.py b/model.py};
\node[anchor=north west, font=\ttfamily\fontsize{6pt}{7.5pt}\selectfont, text=RedLine] at (0.65, 4.75) {- def forward(x): return None};
\node[anchor=north west, font=\ttfamily\fontsize{6pt}{7.5pt}\selectfont, text=GreenLine] at (0.65, 4.35) {+ def forward(x): return self.net(x)};

% Arrows inside Chamber 1
\draw[flow_arrow, draw=OrangeLine] (3.4, 13.0) -- (3.4, 12.7);
\draw[flow_arrow, draw=OrangeLine] (3.4, 10.45) -- (3.4, 10.15);
\draw[flow_arrow, draw=OrangeLine] (3.4, 7.9) -- (3.4, 7.55);

% =============================================================================
% 2. MIDDLE CONTAINER: THE VERIFICATION GATE & AIRLOCK SANITIZER
% (x: 8.2 to 15.0, y: 3.5 to 16.7) - Width 6.8 cm
% =============================================================================
\draw[draw=BlueLine, fill=BlueL0, line width=1.2pt, rounded corners=5pt] (8.2, 3.5) rectangle (15.0, 16.7);

% Header Bar
\fill[BlueLine, rounded corners=3pt] (8.4, 15.95) rectangle (14.8, 16.5);
\node[font=\bfseries\small, text=white] at (11.6, 16.22) {
  MIDDLE AIRLOCK: SANITIZER GATE
};
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=BlueDD] at (11.6, 15.65) {
  Deterministic Invariant Checks $\cdot$ Non-Bypassable Filter
};

% Gate A: Cryptographic Test Hash Check (y: 13.0 to 15.25)
\draw[draw=RedLine, fill=white, line width=0.9pt, rounded corners=3.5pt] (8.4, 13.0) rectangle (14.8, 15.25);
\node[anchor=north west, font=\bfseries\footnotesize, text=deepburgundy] at (8.55, 15.1) {Gate A: Test Suite Hash Invariant};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (8.55, 14.6) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Computes $\mathrm{SHA256}$ of all files in \texttt{tests/}\\
    $\bullet$ Must strictly match pre-rollout manifest $\mathcal{H}_{\text{golden}}$\\
    $\bullet$ \textbf{Fails immediately if agent modified test files!}
  \end{minipage}
};

% Gate B: Scope & Path Whitelist (y: 10.45 to 12.7)
\draw[draw=OrangeLine, fill=white, line width=0.9pt, rounded corners=3.5pt] (8.4, 10.45) rectangle (14.8, 12.7);
\node[anchor=north west, font=\bfseries\footnotesize, text=warmochre] at (8.55, 12.55) {Gate B: Path Whitelist \& Size Bound};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (8.55, 12.05) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Whitelist: Allowed source directories only (\texttt{src/})\\
    $\bullet$ Rejects dotfiles, shell profiles, CI hooks (\texttt{.github/})\\
    $\bullet$ Size bound: $|\text{diff}| \le 250$ lines (anti-DoS quota)
  \end{minipage}
};

% Gate C: AST & Payload Sanitization (y: 7.9 to 10.15)
\draw[draw=amethyst, fill=white, line width=0.9pt, rounded corners=3.5pt] (8.4, 7.9) rectangle (14.8, 10.15);
\node[anchor=north west, font=\bfseries\footnotesize, text=amethyst] at (8.55, 10.0) {Gate C: AST Structure \& Token Hygiene};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (8.55, 9.5) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Validates patch parses into valid AST syntax tree\\
    $\bullet$ Rejects dangerous calls (\texttt{eval}, \texttt{ctypes}, \texttt{subprocess})\\
    $\bullet$ Strips non-ASCII unicode homoglyphs \& hidden escapes
  \end{minipage}
};

% Split Outcome in Gate: Pass vs Trap (y: 3.8 to 7.55)
% Trap Box (Left split)
\draw[draw=RedLine, fill=RedL!50, line width=1.0pt, rounded corners=3.5pt] (8.4, 3.8) rectangle (11.35, 7.55);
\node[anchor=north west, font=\bfseries\scriptsize, text=deepburgundy] at (8.5, 7.35) {REJECTION TRAP};
\node[anchor=north west, font=\fontsize{6.5pt}{8.5pt}\selectfont, text=DarkSlate] at (8.5, 6.85) {
  \begin{minipage}{2.6cm}\raggedright
    $\bullet$ \textbf{Any check fails}\\
    $\bullet$ Instant abort\\
    $\bullet$ Reward set $R_i=0$\\
    $\bullet$ Logged as gaming exploit to ledger
  \end{minipage}
};

% Pass Box (Right split)
\draw[draw=GreenLine, fill=GreenL!50, line width=1.1pt, rounded corners=3.5pt] (11.85, 3.8) rectangle (14.8, 7.55);
\node[anchor=north west, font=\bfseries\scriptsize, text=forestsage] at (11.95, 7.35) {VERIFIED PASS};
\node[anchor=north west, font=\fontsize{6.5pt}{8.5pt}\selectfont, text=DarkSlate] at (11.95, 6.85) {
  \begin{minipage}{2.6cm}\raggedright
    $\bullet$ \textbf{All gates clear}\\
    $\bullet$ Cryptographic sign\\
    $\bullet$ Forward clean diff\\
    $\bullet$ Hand off to Chamber 2 microVM
  \end{minipage}
};

% Arrows inside Middle Gate
\draw[flow_arrow, draw=RedLine] (11.6, 13.0) -- (11.6, 12.7);
\draw[flow_arrow, draw=OrangeLine] (11.6, 10.45) -- (11.6, 10.15);

% Split arrows from Gate C to Trap and Pass
\draw[flow_arrow, draw=RedLine, line width=1.2pt] (10.0, 7.9) -- (9.85, 7.55);
\draw[flow_arrow, draw=GreenLine, line width=1.4pt] (13.2, 7.9) -- (13.35, 7.55);

% AIRLOCK PIPE 1: Chamber 1 -> Middle Gate (Through 1.4 cm Gutter at x = 7.5)
\draw[flow_arrow_thick, draw=warmochre] (6.5, 5.7) -- (7.5, 5.7) -- (7.5, 14.1) -- (8.4, 14.1);
\draw[draw=warmochre, fill=OrangeL, line width=0.8pt, rounded corners=2pt] (7.1, 9.6) rectangle (7.9, 10.2);
\node[font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=warmochre] at (7.5, 9.9) {\texttt{diff}};

% =============================================================================
% 3. RIGHT CONTAINER: CHAMBER 2 (PRISTINE EVALUATION ENCLAVE)
% (x: 16.4 to 23.2, y: 3.5 to 16.7) - Width 6.8 cm
% =============================================================================
\draw[draw=GreenLine, fill=GreenL0, line width=1.2pt, rounded corners=5pt] (16.4, 3.5) rectangle (23.2, 16.7);

% Header Bar
\fill[GreenLine, rounded corners=3pt] (16.6, 15.95) rectangle (23.0, 16.5);
\node[font=\bfseries\small, text=white] at (19.8, 16.22) {
  CHAMBER 2: PRISTINE ENCLAVE
};
\node[font=\bfseries\fontsize{6.5pt}{7.5pt}\selectfont, text=forestsage] at (19.8, 15.65) {
  Clean-Room MicroVM $\cdot$ Immutable Oracles $\cdot$ Consensus
};

% Block 3.1: Pristine MicroVM Boot (y: 13.0 to 15.25)
\draw[draw=GreenLine, fill=white, line width=0.9pt, rounded corners=3.5pt] (16.6, 13.0) rectangle (23.0, 15.25);
\node[anchor=north west, font=\bfseries\footnotesize, text=forestsage] at (16.75, 15.1) {1. Clean-Room MicroVM Boot};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (16.75, 14.6) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Booted fresh from read-only golden image ($<25\,\text{ms}$)\\
    $\bullet$ Hardware virtualization isolation (Firecracker / KVM)\\
    $\bullet$ Zero persistent filesystem state inherited from Chamber 1
  \end{minipage}
};

% Block 3.2: Clean Patch Application (y: 10.45 to 12.7)
\draw[draw=BlueLine, fill=white, line width=0.9pt, rounded corners=3.5pt] (16.6, 10.45) rectangle (23.0, 12.7);
\node[anchor=north west, font=\bfseries\footnotesize, text=BlueLine] at (16.75, 12.55) {2. Pure Patch Application};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (16.75, 12.05) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Command: \texttt{git apply -{-}check patch.diff}\\
    $\bullet$ Applied strictly onto untainted golden commit HEAD\\
    $\bullet$ Aborts immediately if patch produces merge conflict
  \end{minipage}
};

% Block 3.3: Hidden Golden Test Suite (y: 7.9 to 10.15)
\draw[draw=amethyst, fill=white, line width=0.9pt, rounded corners=3.5pt] (16.6, 7.9) rectangle (23.0, 10.15);
\node[anchor=north west, font=\bfseries\footnotesize, text=amethyst] at (16.75, 10.0) {3. Hidden Golden Test Suite ($T_{\text{hidden}}$)};
\node[anchor=north west, font=\fontsize{7pt}{8.8pt}\selectfont, text=DarkSlate] at (16.75, 9.5) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Test assertions never visible to policy during rollout\\
    $\bullet$ Multi-seed consensus verifier ($K=3$ randomized seeds)\\
    $\bullet$ Eliminates flakiness, race conditions, false positives
  \end{minipage}
};

% Block 3.4: One-Way Scalar Reward Verdict (y: 3.8 to 7.55)
\draw[draw=forestsage, fill=GreenL!35, line width=1.0pt, rounded corners=3.5pt] (16.6, 3.8) rectangle (23.0, 7.55);
\node[anchor=north west, font=\bfseries\footnotesize, text=forestsage] at (16.75, 7.4) {4. One-Way 1-Bit Scalar Verdict};
\node[anchor=north west, font=\fontsize{7pt}{9.0pt}\selectfont, text=DarkSlate] at (16.75, 6.9) {
  \begin{minipage}{6.0cm}\raggedright
    $\bullet$ Strictly binary or bounded scalar reward: $R_i \in \{0, 1\}$\\
    $\bullet$ \textbf{Zero stderr, stdout, or stack traces returned to model}\\
    $\bullet$ Prevents model exploiting verifier debug side-channels!
  \end{minipage}
};
% Mini verdict indicator badge inside 3.4
\draw[draw=GreenLine, fill=white, rounded corners=2pt] (16.8, 4.0) rectangle (22.8, 5.0);
\node[font=\bfseries\fontsize{7pt}{8.5pt}\selectfont, text=forestsage] at (19.8, 4.5) {
  Consensus: 3/3 PASS $\implies$ Reward Bit $R_i = 1.0$
};

% Arrows inside Chamber 2
\draw[flow_arrow, draw=GreenLine] (19.8, 13.0) -- (19.8, 12.7);
\draw[flow_arrow, draw=BlueLine] (19.8, 10.45) -- (19.8, 10.15);
\draw[flow_arrow, draw=amethyst] (19.8, 7.9) -- (19.8, 7.55);

% AIRLOCK PIPE 2: Middle Gate (Pass) -> Chamber 2 (Patch Apply) (Through 1.4 cm Gutter at x = 15.7)
\draw[flow_arrow_thick, draw=forestsage] (14.8, 5.7) -- (15.7, 5.7) -- (15.7, 11.6) -- (16.6, 11.6);
\draw[draw=forestsage, fill=GreenL, line width=0.8pt, rounded corners=2pt] (15.2, 8.4) rectangle (16.2, 9.0);
\node[font=\bfseries\fontsize{5.5pt}{6.5pt}\selectfont, text=forestsage] at (15.7, 8.7) {\texttt{signed}};

% =============================================================================
% 4. BOTTOM TIER: RL TRAINING BUS (GRPO ADVANTAGE ESTIMATOR)
% (y: 0.5 to 2.8) - BLOCKS AT TOP, TITLE AT BOTTOM (NO ARROW CROSSINGS)
% =============================================================================
\draw[fill=DarkSlate, draw=none, rounded corners=4pt] (0, 0.5) rectangle (23.2, 2.8);

% Bus Sub-blocks (y: 1.15 to 2.6) - Ready to receive incoming arrows without crossing text!
% Block B1: Cohort Sampling
\draw[draw=BlueLine!70, fill=ChipBlack, rounded corners=3pt] (0.3, 1.15) rectangle (6.5, 2.6);
\node[anchor=north west, font=\bfseries\scriptsize, text=white] at (0.45, 2.45) {Cohort Rollout Collection};
\node[anchor=north west, font=\fontsize{6.5pt}{8pt}\selectfont, text=white!80] at (0.45, 2.05) {
  \begin{minipage}{5.8cm}\raggedright
    Sample $G$ rollouts per prompt: $\{\tau_1, \tau_2, \dots, \tau_G\}$\\
    Executed across parallel isolated Chamber 1 sandboxes
  \end{minipage}
};

% Block B2: Normalized Advantage
\draw[draw=GreenLine!70, fill=ChipBlack, rounded corners=3pt] (8.2, 1.15) rectangle (15.0, 2.6);
\node[anchor=north west, font=\bfseries\scriptsize, text=white] at (8.35, 2.45) {GRPO Normalized Advantage Computation};
\node[anchor=north west, font=\fontsize{6.5pt}{8pt}\selectfont, text=white!80] at (8.35, 2.05) {
  \begin{minipage}{6.4cm}\raggedright
    $\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_1..R_G\})}{\mathrm{std}(\{R_1..R_G\}) + \epsilon} \quad \cdot \quad \text{Strictly zero feedback leakage}$
  \end{minipage}
};

% Block B3: Policy Gradient Step
\draw[draw=amethyst!70, fill=ChipBlack, rounded corners=3pt] (16.6, 1.15) rectangle (23.0, 2.6);
\node[anchor=north west, font=\bfseries\scriptsize, text=white] at (16.75, 2.45) {Surrogate Gradient Optimization};
\node[anchor=north west, font=\fontsize{6.5pt}{8pt}\selectfont, text=white!80] at (16.75, 2.05) {
  \begin{minipage}{6.0cm}\raggedright
    $\theta_{k+1} \leftarrow \theta_k + \alpha \nabla_\theta \mathbb{E}\left[\frac{\pi_\theta(a_t \mid c_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid c_t)} \hat{A}_i\right]$
  \end{minipage}
};

% Bus arrow connections
\draw[flow_arrow, draw=GoldPin, line width=1.3pt] (6.5, 1.87) -- (8.2, 1.87);
\draw[flow_arrow, draw=GoldPin, line width=1.3pt] (15.0, 1.87) -- (16.6, 1.87);

% Title Strip inside bus at the BOTTOM (y: 0.58 to 0.98) - completely isolated!
\fill[ChipBlack, rounded corners=2pt] (0.3, 0.58) rectangle (23.0, 0.98);
\node[anchor=west, font=\bfseries\footnotesize, text=GoldPin] at (0.45, 0.78) {
  REINFORCEMENT LEARNING UPDATE BUS $\cdot$ GRPO / PPO SYNCHRONIZATION
};
\node[anchor=east, font=\fontsize{7pt}{8.5pt}\selectfont, text=white!80] at (22.85, 0.78) {
  Group Relative Policy Optimization $\cdot$ Deterministic Reward Synchronization $\cdot$ Zero-Leakage
};

% Arrow: Trap outcome down to abort status in Bus (Enters B2 cleanly from above!)
\draw[flow_arrow, draw=RedLine, dashed, line width=1.3pt] 
  (9.85, 3.8) -- (9.85, 2.6);
\node[draw=RedLine, fill=white, rounded corners=2pt, inner sep=1.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=deepburgundy] 
  at (9.85, 3.2) {Abort: $R_i=0$};

% Arrow: Verdict bit into Bus (Enters B3 cleanly from above!)
\draw[flow_arrow, draw=GreenLine, line width=1.6pt] 
  (19.8, 3.8) -- (19.8, 2.6);
\node[draw=GreenLine, fill=white, rounded corners=2pt, inner sep=1.5pt, font=\bfseries\fontsize{6pt}{7pt}\selectfont, text=forestsage] 
  at (19.8, 3.2) {Verdict Bit $R_i \in \{0, 1\}$};

% Feedback loop from B1 back to Policy in Chamber 1
% Starts at B1 left edge, routes around left margin at x = -0.7 up to Chamber 1 Block 1.1 at (0.0, 14.1)
\draw[flow_arrow, draw=amethyst, dashed, line width=1.5pt] 
  (0.3, 1.87) -- (-0.8, 1.87) -- (-0.8, 14.1) -- (0.0, 14.1);
\node[font=\bfseries\scriptsize, text=amethyst, rotate=90] 
  at (-1.1, 8.0) {$\Delta\theta$ WEIGHT UPDATE BUS (Synchronized Parameter Checkpoint)};

\end{tikzpicture}
\end{document}
"""

repo_root = Path("/Users/VJ/GitHub/MLSysBook-vol3-review")
tex_file = repo_root / "books/vol3/tex/dual_sandbox_verification_enclave.tex"

with open(tex_file, "w") as f:
    f.write(TIKZ_CODE)

res = subprocess.run(["pdflatex", "-halt-on-error", "-output-directory=/tmp", str(tex_file)], capture_output=True, text=True)
print("pdflatex return code:", res.returncode)
if res.returncode == 0:
    subprocess.run(["pdftoppm", "-png", "-r", "200", "/tmp/dual_sandbox_verification_enclave.pdf", "/tmp/dual_sandbox_render_v4"])
    subprocess.run(["pdftocairo", "-svg", "/tmp/dual_sandbox_verification_enclave.pdf", "/tmp/dual_sandbox_verification_enclave.svg"])
    print("Successfully compiled and rendered v4!")
else:
    print(res.stdout[-1500:])
