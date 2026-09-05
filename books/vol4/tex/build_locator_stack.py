#!/usr/bin/env python3
"""
Generate the four-tier "systems organ locator" stack figure that opens every
chapter of Physical AI.

One template, one chapter table. The per-chapter variation is exactly three
things: which tier is active, which card inside that tier is this chapter, and
what the header badge says. Everything else is shared, so a chapter reorder is
a one-line edit here rather than 17 near-identical TikZ files drifting apart.

Usage:  python3 build_locator_stack.py [slug ...]
Writes fig_locator_layered_stack.tex next to each chapter, compiles it with
lualatex, converts to SVG with pdftocairo, and mirrors the SVG into the
publishing tree.
"""

import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
BOOK_CHAPTERS = os.path.normpath(os.path.join(BASE, "..", "chapters"))
PUB_CHAPTERS = os.path.normpath(
    os.path.join(BASE, "..", "..", "..", "publishing", "quarto", "contents", "vol4", "chapters")
)

# Tier accent colours, keyed by layer. Second entry is the tint used to fill an
# active tier; both names are defined in the preamble below.
ACCENT = {1: ("crimson", "crimsonlight"),
          2: ("petrol", "petrollight"),
          3: ("darkblue", "bluelight"),
          4: ("purple", "purplelight")}

# slug -> (chapter number as printed, badge label, active tiers, own card id)
# The chapter number is the position in the reading order, which is why the
# nervous system directory (04-nervous) prints as chapter 3.
CHAPTERS = {
    "01-boundary":     (1,  "BOUNDARY",                (1,),   None),
    "02-body":         (2,  "THE BODY",                (1,),   None),
    "04-nervous":      (3,  "NERVOUS SYSTEM",          (2,),   None),
    "03-brain":        (4,  "THE BRAIN",               (3,),   None),
    "05-data":         (5,  "DATA \\& INGESTION",      (3,),   "b1"),
    "06-training":     (6,  "TRAINING \\& SIM-TO-REAL", (3, 1), None),
    "07-evaluation":   (7,  "EVALUATION \\& METROLOGY", (3, 1), None),
    "08-perception":   (8,  "SPATIAL PERCEPTION",      (3,),   "b2"),
    "09-memory":       (9,  "TEMPORAL MEMORY",         (3,),   "b3"),
    "10-intent":       (10, "INTENT \\& GOALS",        (3,),   "b4"),
    "11-planning":     (11, "TRAJECTORY PLANNING",     (3,),   "b5"),
    "12-enforcement":  (12, "RUNTIME ENFORCEMENT",     (2,),   "n1"),
    "13-placement":    (13, "SILICON PLACEMENT",       (2,),   "n2"),
    "14-intervention": (14, "SHARED AUTONOMY",         (4,),   "g1"),
    "15-verification": (15, "VERIFICATION",            (4,),   "g2"),
    "16-release":      (16, "SAFETY CASES",            (4,),   "g3"),
    "17-frontier":     (17, "FRONTIER LIMITS",         (4,),   "g4"),
}

PREAMBLE = r"""\documentclass[tikz,border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\usepackage{sfmath}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\usepackage{xcolor}

\renewcommand{\familydefault}{\sfdefault}

% Harvard Crimson & ETH Zurich Semantic Palette
\definecolor{crimson}{HTML}{A51C30}
\definecolor{crimsonlight}{HTML}{FEF2F2}
\definecolor{darkblue}{HTML}{1F407A}
\definecolor{bluelight}{HTML}{F0F4FA}
\definecolor{petrol}{HTML}{007A87}
\definecolor{petrollight}{HTML}{F0FDFA}
\definecolor{bronze}{HTML}{B87333}
\definecolor{purple}{HTML}{5B4B8A}
\definecolor{purplelight}{HTML}{F8FAFC}
\definecolor{slate}{HTML}{475569}
\definecolor{mutedslate}{HTML}{64748B}
\definecolor{bordercolor}{HTML}{CBD5E1}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  >=Stealth,
  node distance=0.18in,
  layercard/.style 2 args={
    draw=#1,
    fill=#2,
    rounded corners=6pt,
    line width=1.1pt,
    inner sep=8pt,
    text width=7.40in,
    align=left
  },
  subcard/.style={
    draw=bordercolor,
    fill=white,
    rounded corners=4pt,
    line width=0.8pt,
    inner sep=4pt,
    align=center
  }
]
"""


def tier_card(layer, active):
    """Style for the tier container itself."""
    if layer in active:
        acc, tint = ACCENT[layer]
        return f"layercard={{{acc}}}{{{tint}}}, line width=1.5pt", acc
    return "layercard={bordercolor}{white}, line width=0.8pt", "mutedslate"


def sub_style(layer, active, node_id, own):
    """
    Three states, so the reader can tell tier from chapter:
      inactive tier      -> grey border, white fill
      active tier        -> accent border, white fill
      this chapter's card-> accent border, accent tint, heavier rule
    """
    if layer not in active:
        return "draw=bordercolor, fill=white, line width=0.8pt"
    acc, _ = ACCENT[layer]
    if own is not None and node_id == own:
        # The chapter's own card. The active tier is already washed in its
        # accent tint, so a tint fill here would recede; go a shade deeper.
        return f"draw={acc}, fill={acc}!14, line width=1.9pt"
    return f"draw={acc}, fill=white, line width=1.1pt"


def emph(text, condition):
    return f"\\textbf{{{text}}}" if condition else text


def build_tex(slug):
    num, badge, active, own = CHAPTERS[slug]
    badge_acc, badge_tint = ACCENT[active[0]]

    l4s, l4h = tier_card(4, active)
    l3s, l3h = tier_card(3, active)
    l2s, l2h = tier_card(2, active)
    l1s, l1h = tier_card(1, active)

    g = [sub_style(4, active, f"g{i}", own) for i in range(1, 5)]
    b = [sub_style(3, active, f"b{i}", own) for i in range(1, 6)]
    n = [sub_style(2, active, f"n{i}", own) for i in range(1, 4)]
    p = [sub_style(1, active, f"p{i}", own) for i in range(1, 4)]

    # "You are here" emphasis in the tier subtitles, for the chapters that are
    # named there rather than carrying their own card.
    brain_ref = emph("Ch 04 Brain", slug == "03-brain")
    nerve_ref = emph("Ch 03 Nervous System", slug == "04-nervous")
    bound_ref = emph("Ch 01 Boundary", slug == "01-boundary")
    body_ref = emph("Ch 02 Body", slug == "02-body")
    gap_ref = emph("Ch 06/07 Reality Gap", slug in ("06-training", "07-evaluation"))

    return PREAMBLE + rf"""
  % Title & Chapter Badge
  \node[anchor=west, font=\sffamily\bfseries\normalsize, text=darkblue] at (0, 0.45in) {{PHYSICAL AI SYSTEMS ARCHITECTURE STACK}};
  \node[anchor=east, font=\sffamily\bfseries\scriptsize, fill={badge_tint}, draw={badge_acc}, rounded corners=2pt, inner sep=3.5pt, text={badge_acc}] at (7.65in, 0.45in) {{ACTIVE FOCUS: CHAPTER {num:02d} $\cdot$ {badge}}};

  % ==========================================
  % LAYER 4: SYSTEM GOVERNANCE & ASSURANCE
  % ==========================================
  \node[{l4s}, anchor=north west] (l4) at (0, 0.20in) {{
    \textbf{{\color{{{l4h}}}\normalsize LAYER 4 $\cdot$ SYSTEM GOVERNANCE \& ASSURANCE TIER [0.1--1 Hz]}}\hfill{{\scriptsize\color{{mutedslate}}Policy Supervision \& Safety Cases}}\\[5pt]
    \begin{{tikzpicture}}[node distance=0.10in]
      \node[subcard, {g[0]}, text width=1.52in] (g1) {{\textbf{{\color{{purple}}\footnotesize Ch 14 Shared Autonomy}}\\{{\tiny Policy Blending $\cdot$ Takeover}}}};
      \node[subcard, {g[1]}, text width=1.52in, right=0.14in of g1] (g2) {{\textbf{{\color{{purple}}\footnotesize Ch 15 Verification}}\\{{\tiny Sim $\to$ PIL $\to$ HIL $\to$ In-Situ}}}};
      \node[subcard, {g[2]}, text width=1.52in, right=0.14in of g2] (g3) {{\textbf{{\color{{purple}}\footnotesize Ch 16 Safety Cases}}\\{{\tiny Goal Structuring $\cdot$ UL 4600}}}};
      \node[subcard, {g[3]}, text width=1.52in, right=0.14in of g3] (g4) {{\textbf{{\color{{purple}}\footnotesize Ch 17 Frontier Limits}}\\{{\tiny Observational Limits}}}};
    \end{{tikzpicture}}
  }};

  % ==========================================
  % LAYER 3: THE COGNITIVE BRAIN (STOCHASTIC DELIBERATION)
  % ==========================================
  \node[{l3s}, below=0.18in of l4.south west, anchor=north west] (l3) {{
    \textbf{{\color{{{l3h}}}\normalsize LAYER 3 $\cdot$ THE BRAIN: COGNITIVE DELIBERATION TIER [1--50 Hz $\cdot$ Linux MPU / Edge NPU]}}\\[1pt]
    {{\scriptsize\color{{slate}}High-Capacity Foundation Models, Spatial Perception, Latent World Belief, and Action Chunking ({brain_ref})}}\\[6pt]
    \begin{{tikzpicture}}[node distance=0.08in]
      \node[subcard, {b[0]}, text width=1.20in] (b1) {{\textbf{{\color{{darkblue}}\footnotesize 1. Ingestion (Ch 05)}}\\{{\tiny MIPI CSI-2 $\cdot$ Buffers}}}};
      \node[subcard, {b[1]}, text width=1.20in, right=0.08in of b1] (b2) {{\textbf{{\color{{darkblue}}\footnotesize 2. Perception (Ch 08)}}\\{{\tiny $o_t \to z_t$ $\cdot$ ViT Encoders}}}};
      \node[subcard, {b[2]}, text width=1.20in, right=0.08in of b2] (b3) {{\textbf{{\color{{darkblue}}\footnotesize 3. Memory (Ch 09)}}\\{{\tiny $z_t \to b_t$ $\cdot$ $SE(3)$ Belief}}}};
      \node[subcard, {b[3]}, text width=1.20in, right=0.08in of b3] (b4) {{\textbf{{\color{{darkblue}}\footnotesize 4. Intent (Ch 10)}}\\{{\tiny $b_t \to L_t$ $\cdot$ TTL Lease}}}};
      \node[subcard, {b[4]}, text width=1.20in, right=0.08in of b4] (b5) {{\textbf{{\color{{darkblue}}\footnotesize 5. Planner (Ch 11)}}\\{{\tiny $L_t \to \hat{{u}}$ $\cdot$ ACT / Chunk}}}};
      \draw[->, line width=0.7pt, darkblue] (b1) -- (b2);
      \draw[->, line width=0.7pt, darkblue] (b2) -- (b3);
      \draw[->, line width=0.7pt, darkblue] (b3) -- (b4);
      \draw[->, line width=0.7pt, darkblue] (b4) -- (b5);
    \end{{tikzpicture}}
  }};

  % ==========================================
  % THE PROPOSAL-PERMISSION PRIVILEGE BOUNDARY
  % ==========================================
  \coordinate (bleft) at ($(l3.south west) + (0.10in, -0.16in)$);
  \coordinate (bright) at ($(l3.south east) + (-0.60in, -0.16in)$);
  \draw[dashed, line width=1.3pt, mutedslate] (bleft) -- (bright);
  \node[font=\sffamily\bfseries\scriptsize, fill=white, draw=mutedslate, rounded corners=3pt, inner sep=2.5pt, text=mutedslate]
    at ($(bleft)!0.46!(bright)$)
    {{PROPOSAL--PERMISSION PRIVILEGE BOUNDARY (Shared SRAM Mailbox $\cdot$ Untrusted $\hat{{u}}_{{t:t+H}} \to$ Verified $u^*$)}};

  % ==========================================
  % LAYER 2: THE NERVOUS SYSTEM (DETERMINISTIC SAFETY)
  % ==========================================
  \node[{l2s}, below=0.34in of l3.south west, anchor=north west] (l2) {{
    \textbf{{\color{{{l2h}}}\normalsize LAYER 2 $\cdot$ THE NERVOUS SYSTEM: REAL-TIME SAFETY \& TIMING TIER [1000 Hz $\cdot$ Bare-Metal MCU]}}\\[1pt]
    {{\scriptsize\color{{slate}}Zero-Allocation Deterministic Silicon, Hardware Watchdogs, and Safety Refusal ({nerve_ref})}}\\[6pt]
    \begin{{tikzpicture}}[node distance=0.10in]
      \node[subcard, {n[0]}, text width=2.08in] (n1) {{\textbf{{\color{{petrol}}\footnotesize 1 kHz Reflex Enforcer (Ch 12)}}\\{{\tiny Control Barrier Filter $h(\mathbf{{x}}) \ge 0$ $\cdot$ Veto $\hat{{u}}$}}}};
      \node[subcard, {n[1]}, text width=2.08in, right=0.16in of n1] (n2) {{\textbf{{\color{{petrol}}\footnotesize Silicon Placement \& Bus QoS (Ch 13)}}\\{{\tiny Interconnect QoS $\cdot$ SRAM Seqlock $\cdot$ Timers}}}};
      \node[subcard, {n[2]}, text width=2.08in, right=0.16in of n2] (n3) {{\textbf{{\color{{petrol}}\footnotesize Certified Actuator Latch}}\\{{\tiny Latches $u^*$ into PWM Compare Registers}}}};
      \draw[->, line width=0.7pt, petrol] (n1) -- (n2);
      \draw[->, line width=0.7pt, petrol] (n2) -- (n3);
    \end{{tikzpicture}}
  }};

  % ==========================================
  % LAYER 1: THE PHYSICAL BODY & CONTINUOUS PLANT
  % ==========================================
  \node[{l1s}, below=0.18in of l2.south west, anchor=north west] (l1) {{
    \textbf{{\color{{{l1h}}}\normalsize LAYER 1 $\cdot$ THE PHYSICAL BODY \& CONTINUOUS PLANT [Continuous Dynamical Mechanics]}}\\[1pt]
    {{\scriptsize\color{{slate}}Matter, Inertia, Transduction, Friction, and Thermal Limits ({bound_ref} $\cdot$ {body_ref} $\cdot$ {gap_ref})}}\\[6pt]
    \begin{{tikzpicture}}[node distance=0.10in]
      \node[subcard, {p[0]}, text width=2.08in] (p1) {{\textbf{{\color{{crimson}}\footnotesize Power Inverters \& Gate Drivers}}\\{{\tiny PWM Switching $\to$ Phase Current $I_{{\text{{phase}}}}$}}}};
      \node[subcard, {p[1]}, text width=2.08in, right=0.16in of p1] (p2) {{\textbf{{\color{{crimson}}\footnotesize Mechanics, Torque \& Momentum}}\\{{\tiny Lorentz Force $\to$ Torque $\tau \to$ Momentum $p = mv$}}}};
      \node[subcard, {p[2]}, text width=2.08in, right=0.16in of p2] (p3) {{\textbf{{\color{{crimson}}\footnotesize Causal Boundary \& Dissipation}}\\{{\tiny Stopping Envelope $d_{{\text{{stop}}}} \le d_{{\text{{clear}}}}$ $\cdot$ Heat}}}};
      \draw[->, line width=0.7pt, crimson] (p1) -- (p2);
      \draw[->, line width=0.7pt, crimson] (p2) -- (p3);
    \end{{tikzpicture}}
  }};

  % Flow arrows down the right side
  \draw[->, line width=1.2pt, bronze] ($(l3.south east) + (-0.25in, 0)$) -- node[right, font=\scriptsize\bfseries, text=bronze, xshift=2pt] {{Proposal $\hat{{u}}$}} ($(l2.north east) + (-0.25in, 0)$);
  \draw[->, line width=1.2pt, petrol] ($(l2.south east) + (-0.25in, 0)$) -- node[right, font=\scriptsize\bfseries, text=petrol, xshift=2pt] {{Certified $u^*$}} ($(l1.north east) + (-0.25in, 0)$);

  % Left Feedback Loop: Endogenous Sensory Shift
  \draw[->, line width=1.2pt, crimson] (l1.west) -- ++(-0.25in, 0) |- node[pos=0.25, above, rotate=90, font=\sffamily\bfseries\scriptsize, text=crimson] {{Endogenous Sensory Shift ($o_{{t+1}}$)}} (l3.west);

\end{{tikzpicture}}
\end{{document}}
"""


def build(slug):
    figdir = os.path.join(BOOK_CHAPTERS, slug, "figures")
    os.makedirs(figdir, exist_ok=True)
    stem = "fig_locator_layered_stack"
    with open(os.path.join(figdir, stem + ".tex"), "w") as fh:
        fh.write(build_tex(slug))

    r = subprocess.run(["lualatex", "-interaction=nonstopmode", stem + ".tex"],
                       cwd=figdir, capture_output=True, text=True)
    if not os.path.exists(os.path.join(figdir, stem + ".pdf")):
        print(f"  FAIL {slug}: lualatex produced no PDF")
        print("  " + "\n  ".join(r.stdout.splitlines()[-12:]))
        return False

    subprocess.run(["pdftocairo", "-svg", stem + ".pdf", "fig_locator.svg"],
                   cwd=figdir, check=True, capture_output=True)

    for junk in (stem + ".aux", stem + ".log", stem + ".pdf"):
        j = os.path.join(figdir, junk)
        if os.path.exists(j):
            os.remove(j)

    pub = os.path.join(PUB_CHAPTERS, slug, "figures", "fig_locator.svg")
    if os.path.isdir(os.path.dirname(pub)):
        shutil.copyfile(os.path.join(figdir, "fig_locator.svg"), pub)
    else:
        print(f"  WARN {slug}: no publishing mirror at {pub}")
    print(f"  ok   {slug}")
    return True


if __name__ == "__main__":
    targets = sys.argv[1:] or list(CHAPTERS)
    bad = [s for s in targets if s not in CHAPTERS]
    if bad:
        sys.exit(f"unknown chapter slug(s): {', '.join(bad)}")
    print(f"Regenerating {len(targets)} locator stack figures")
    failed = [s for s in targets if not build(s)]
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")
    print("All locator stack figures regenerated.")
