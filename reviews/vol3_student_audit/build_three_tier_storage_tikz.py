#!/usr/bin/env python3
"""
Build and render Three-Tier Storage Pyramid & Content-Addressed Causal Spine Diagram for Chapter 11.
Follows strict MLSysBook publication standards:
- Canonical color palette (DarkSlate, BlueLine, GreenLine, OrangeLine, RedLine, PurpleLine, etc.)
- Visual inspection via pdftoppm + view_file
- Zero collisions, zero overlaps, zero clipping
"""

import subprocess
from pathlib import Path

TIKZ_CODE = r"""\documentclass[tikz,border=22pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{tcolorbox}
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

\definecolor{PurpleLine}{HTML}{7C3AED}
\definecolor{PurpleL0}{HTML}{FAF5FF}
\definecolor{deeproyal}{HTML}{5B21B6}

\begin{document}
\begin{tikzpicture}[
  font=\sffamily,
  scale=1.0,
  >=Stealth,
  node distance=0.5cm
]

% -------------------------------------------------------------
% MASTER TITLE BANNER
% -------------------------------------------------------------
\node[anchor=north west, inner sep=0pt] (title) at (0, 33.2) {
  \begin{minipage}{27.6cm}
    \begin{tabular}{@{}p{20.0cm} r@{}}
      {\fontsize{13.5}{17}\selectfont \textbf{\textcolor{DarkSlate}{THE THREE-TIER STORAGE PYRAMID \& CONTENT-ADDRESSED CAUSAL SPINE}}} &
      \fontsize{9.5}{12}\selectfont \textcolor{slatesteel}{\textbf{Chapter 11: Durable Execution · Section 11.5}} \\[2pt]
      {\fontsize{9}{11.5}\selectfont \textcolor{slatesteel}{Immutable Event Sourcing ($\mathcal{S}_{\text{causal}}$) with Content-Addressed Payload Offloading ($\theta_{\text{bulk}} = 64\text{ KB}$)}} & \\
    \end{tabular}
  \end{minipage}
};

% -------------------------------------------------------------
% LEFT PANEL: END-TO-END TIERED STORAGE & CAUSAL SPINE DATAPATH
% -------------------------------------------------------------
% Left Outer Box: from (0, 31.5) to (17.8, 7.4)
\draw[draw=BlueLine!60, fill=coolparchment, line width=1.2pt, rounded corners=6pt] 
  (0, 31.5) rectangle (17.8, 7.4);

% Left Header inside Container
\node[anchor=north west, inner sep=0pt] at (0.5, 31.15) {
  \begin{minipage}{16.8cm}
    \begin{tabular}{@{}p{11.6cm} r@{}}
      {\fontsize{10}{12}\selectfont \textbf{\textcolor{BlueLine}{TIERED TRAJECTORY PERSISTENCE \& CAUSAL OFF-LOADING}}} &
      \fontsize{7.5}{9.5}\selectfont \textcolor{BlueLine}{\textbf{Lifecycle: Live $\to$ Quiescent $\to$ Archival}}
    \end{tabular}\\[1pt]
    {\fontsize{7}{8.5}\selectfont \textcolor{slatesteel}{Hardware media optimization: NVMe Direct I/O ring buffers $\to$ S3 CAS $\to$ Columnar Parquet}}
  \end{minipage}
};

% --- Stage 1: Active Trajectory Turn ---
\node[anchor=north west, fill=white, draw=slatesteel!40, line width=0.9pt, rounded corners=5pt, inner sep=6pt] 
  (stg1) at (0.5, 30.3) {
  \begin{minipage}{16.6cm}
    \begin{tabular}{@{}p{11.6cm} r@{}}
      {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{DarkSlate}{1. Active Trajectory Turn Execution ($t \in [1, 1{,}000]$)}}} &
      \fontsize{7}{9}\selectfont \textcolor{deepburgundy}{\textbf{Barrier Invariant: $e_{\text{flush}} \prec a_{\text{ext}}$}}
    \end{tabular}\\[2pt]
    {\fontsize{6.8}{8.5}\selectfont Model emits candidate action $e_{\text{prop}} \to$ Policy gate emits $e_{\text{auth}} \to$ Durable barrier flush $e_{\text{flush}} \to$ Tool dispatches $a_{\text{ext}}$}
  \end{minipage}
};

% Arrow from Stage 1 to Tier 1
\draw[->, BlueLine, line width=1.6pt] (8.9, 28.8) -- (8.9, 28.0);
\node[fill=coolparchment, inner sep=2pt, font=\fontsize{6}{7}\selectfont\bfseries, text=BlueLine] at (8.9, 28.4) {Append Intent \& Observation};

% --- Stage 2: Tier 1 - Hot Tier (Local NVMe Ring Buffer) ---
\node[anchor=north west, fill=white, draw=RedLine!60, line width=1.0pt, rounded corners=5pt, inner sep=6pt] 
  (t1_hot) at (0.5, 27.9) {
  \begin{minipage}{16.6cm}
    \begin{tabular}{@{}p{11.6cm} r@{}}
      {\fontsize{9}{11}\selectfont \textbf{\textcolor{deepburgundy}{Tier 1: Hot Tier (Local NVMe Direct I/O Ring Buffer)}}} &
      \fontsize{7.5}{9.5}\selectfont \textcolor{deepburgundy}{\textbf{Latency: $< 0.5\text{ ms}$ · $\text{WAF}_{\text{NVMe}} \approx 1.0$}}
    \end{tabular}\\[2pt]
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ Physical Media: Local PCIe Gen5 NVMe SSD / Embedded RocksDB WAL · Pre-allocated circular ring buffer\\
    \textbullet\ Direct I/O / \texttt{fdatasync()} guarantees zero page-cache buffering stalls; instantaneous local crash rollback\\
    \textbullet\ Holds active in-flight span contexts and recent checkpoints; drains asynchronously upon turn completion
    }
  \end{minipage}
};

% Arrow from Tier 1 to Bifurcation Gate
\draw[->, OrangeLine, line width=1.6pt] (8.9, 25.4) -- (8.9, 24.5);
\node[fill=coolparchment, inner sep=2pt, font=\fontsize{6}{7}\selectfont\bfseries, text=warmochre] at (8.9, 24.95) {Evaluate Payload Size $|B_{\text{tool}}|$};

% --- Stage 3: Causal Bifurcation Gate ---
\node[anchor=north west, fill=OrangeL0, draw=OrangeLine, line width=1.1pt, rounded corners=5pt, inner sep=5pt] 
  (gate_bifurc) at (0.5, 24.4) {
  \begin{minipage}{16.6cm}
    \centering
    {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{warmochre}{Content-Addressed Payload Offloading Gate: Threshold $\theta_{\text{bulk}} = 64\text{ KB}$}}}\\[1pt]
    {\fontsize{6.8}{8.5}\selectfont \textcolor{DarkSlate}{Prevents megabyte-scale tool observations (compiler dumps, DOM trees) from bloating the sequential write-ahead log}}
  \end{minipage}
};

% Forking Arrows from Bifurcation Gate
\draw[->, GreenLine, line width=1.4pt] (4.5, 23.0) -- (4.5, 21.9);
\node[fill=coolparchment, inner sep=2pt, font=\fontsize{6}{7}\selectfont\bfseries, text=forestsage] at (4.5, 22.45) {Branch A: Payload $\le 64\text{ KB}$};

\draw[->, PurpleLine, line width=1.4pt] (13.3, 23.0) -- (13.3, 21.9);
\node[fill=coolparchment, inner sep=2pt, font=\fontsize{6}{7}\selectfont\bfseries, text=deeproyal] at (13.3, 22.45) {Branch B: Payload $> 64\text{ KB}$};

% Sub-Boxes: Inline vs. Offload
% Left Sub-Box: Inline Ingest
\node[anchor=north west, fill=GreenL0, draw=GreenLine, line width=0.9pt, rounded corners=4pt, inner sep=5pt] 
  (box_inline) at (0.5, 21.8) {
  \begin{minipage}{7.8cm}
    {\fontsize{8}{10}\selectfont \textbf{\textcolor{forestsage}{Inline Ingest: Lightweight Observation}}}\\[2pt]
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ Status codes, short stdout, exit flags ($< 4\text{ KB}$)\\
    \textbullet\ Stored verbatim directly in spine event envelope\\
    \textbullet\ Zero network RPC overhead on recovery
    }
  \end{minipage}
};

% Right Sub-Box: Tier 2 Content-Addressed Storage
\node[anchor=north west, fill=PurpleL0, draw=PurpleLine, line width=0.9pt, rounded corners=4pt, inner sep=5pt] 
  (box_cas) at (9.3, 21.8) {
  \begin{minipage}{7.8cm}
    {\fontsize{8}{10}\selectfont \textbf{\textcolor{deeproyal}{Tier 2: Content-Addressed Storage (CAS)}}}\\[2pt]
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ Key: Cryptographic digest $h = \text{SHA-256}(B_{\text{tool}})$\\
    \textbullet\ Raw blob offloaded to cloud object store (S3/GCS)\\
    \textbullet\ \textbf{Fleet Deduplication:} Identical blobs stored ONCE!
    }
  \end{minipage}
};

% Joining Arrows into Causal Spine
\draw[->, GreenLine, line width=1.4pt] (4.5, 19.9) -- (4.5, 19.0);
\draw[->, PurpleLine, line width=1.4pt] (13.3, 19.9) -- (13.3, 19.0);

% --- Stage 4: The Immutable Causal Spine Envelope ---
\node[anchor=north west, fill=white, draw=BlueLine, line width=1.2pt, rounded corners=5pt, inner sep=6pt] 
  (causal_spine) at (0.5, 18.9) {
  \begin{minipage}{16.6cm}
    {\fontsize{9}{11}\selectfont \textbf{\textcolor{BlueLine}{The Immutable Causal Spine ($\mathcal{S}_{\text{causal}}$) · Event Envelope}}}\\[3pt]
    
    % Event tuple representation
    \begin{tcolorbox}[colback=coolparchment, colframe=BlueLine!50, arc=3pt, boxrule=0.8pt, left=4pt, right=4pt, top=3pt, bottom=3pt]
      \centering
      {\fontsize{7.5}{9.5}\selectfont
      $\mathcal{S}_{\text{causal}} = \left\langle \left( \mathbf{\text{LSN}_i},\, \mathbf{t_i},\, \mathbf{\text{type}_i},\, \mathbf{\text{id}_i},\, \mathbf{h_i = \text{SHA-256}(B_{\text{tool}})},\, \mathbf{h_{\text{ctx}}^{(t)}},\, \mathbf{\text{summary}} \right) \right\rangle_{i=1}^N$
      }
    \end{tcolorbox}
    
    \vspace{2pt}
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ \textbf{Strict Invariant:} Strictly immutable, append-only, and \textbf{never pruned}. Retains every exploratory turn.\\
    \textbullet\ \textbf{Compact Footprint:} A 1,000-turn trajectory spine is $\approx 4.2\text{ MB}$ (vs. $> 12\text{ GB}$ of unpruned tool payloads!)\\
    \textbullet\ \textbf{Instant Replay:} Replay engine parses the lean spine in milliseconds; fetches CAS blobs lazily for deep forensics.
    }
  \end{minipage}
};

% Arrow from Causal Spine to Tier 3 Lakehouse
\draw[->, BlueLine, line width=1.6pt] (8.9, 15.0) -- (8.9, 14.2);
\node[fill=coolparchment, inner sep=2pt, font=\fontsize{6}{7}\selectfont\bfseries, text=BlueLine] at (8.9, 14.6) {Asynchronous Batch Compaction (After 7 Days)};

% --- Stage 5: Tier 3 - Cold Lakehouse Tier ---
\node[anchor=north west, fill=white, draw=slatesteel!60, line width=1.0pt, rounded corners=5pt, inner sep=6pt] 
  (t3_lakehouse) at (0.5, 14.1) {
  \begin{minipage}{16.6cm}
    \begin{tabular}{@{}p{9.6cm} r@{}}
      {\fontsize{9}{11}\selectfont \textbf{\textcolor{DarkSlate}{Tier 3: Cold Columnar Lakehouse (Parquet + Zstd)}}} &
      \fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{\textbf{Compression: $6\times\text{--}10\times$ · Cost: $<\$0.001/\text{GB/mo}$}}
    \end{tabular}\\[2pt]
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ Completed trajectories compacted into Apache Parquet with columnar dictionary encoding and Zstandard compression\\
    \textbullet\ Vectorized OLAP scans (DuckDB/Trino) query billions of historical turns without loading raw tool bodies into memory\\
    \textbullet\ Feeds offline RLVR / SFT dataset curation pipelines (Section 11.6) and statutory compliance audits
    }
  \end{minipage}
};

% Arrow from Lakehouse to Workspace CoW Squashing
\draw[->, GreenLine, line width=1.6pt] (8.9, 11.7) -- (8.9, 10.5);
\node[fill=coolparchment, inner sep=2pt, font=\fontsize{6}{7}\selectfont\bfseries, text=forestsage] at (8.9, 11.1) {Decoupled Sandbox Storage};

% --- Stage 6: Workspace Filesystem CoW Squashing ---
\node[anchor=north west, fill=white, draw=GreenLine!70, line width=1.0pt, rounded corners=5pt, inner sep=6pt] 
  (stg6_cow) at (0.5, 10.4) {
  \begin{minipage}{16.6cm}
    \begin{tabular}{@{}p{11.6cm} r@{}}
      {\fontsize{9}{11}\selectfont \textbf{\textcolor{forestsage}{Complementary Subsystem: Workspace Filesystem CoW Squashing}}} &
      \fontsize{7.5}{9.5}\selectfont \textcolor{forestsage}{\textbf{Decoupled from Log · \texttt{ioctl(FICLONE)}}}
    \end{tabular}\\[2pt]
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ File mutations, git checkouts, and build trees are compacted at OS layer, leaving the causal log 100\% untouched\\
    \textbullet\ Consolidates volatile OverlayFS \texttt{upperdir} diffs into immutable base rootfs snapshots for rapid cold restarts\\
    \textbullet\ Reflink extent cloning (\texttt{FICLONE}) shares physical blocks across worker containers with zero storage bloat
    }
  \end{minipage}
};

% -------------------------------------------------------------
% RIGHT PANEL: THE AUTOREGRESSIVE CAUSAL CONTRADICTION
% -------------------------------------------------------------
% Right Outer Box: from (18.4, 31.5) to (27.6, 7.4)
\draw[draw=RedLine!70, fill=coolparchment, line width=1.2pt, rounded corners=6pt] 
  (18.4, 31.5) rectangle (27.6, 7.4);

% Right Header inside Container
\node[anchor=north west, inner sep=0pt] at (18.8, 31.15) {
  \begin{minipage}{8.4cm}
    {\fontsize{9.5}{11.5}\selectfont \textbf{\textcolor{deepburgundy}{AUTOREGRESSIVE CAUSALITY}}}\\[1pt]
    {\fontsize{6.8}{8.5}\selectfont \textcolor{slatesteel}{Why database LSM compaction destroys agent systems}}
  \end{minipage}
};

% Right Column Content Boxes (Upper half: The Contradiction)
\node[anchor=north west, fill=white, draw=slatesteel!40, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (rbox1) at (18.8, 30.3) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{DarkSlate}{1. Classical DB Overwrite Fallacy}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    In RocksDB/LSM stores, mutations commute:\\[1pt]
    $\langle k, v_2 \rangle$ committed at $t_2$ completely supersedes and discards $\langle k, v_1 \rangle$ committed at $t_1$.\\
    Compaction purges old versions to bound read amplification.
    }
  \end{minipage}
};

\node[anchor=north west, fill=white, draw=deepburgundy!50, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (rbox2) at (18.8, 28.2) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{deepburgundy}{2. Autoregressive Causal Conditioning}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    An agent trajectory is an autoregressive Markov chain:
    \begin{equation*}
      y_t \sim P_\theta\left(y_t \mid x,\, a_1,\, o_1,\, \dots,\, a_{t-1},\, o_{t-1}\right)
    \end{equation*}
    Every generated token is strictly conditioned on the \textbf{exact sequence} of all prior prompts, actions, and observations!
    }
  \end{minipage}
};

\node[anchor=north west, fill=RedL0, draw=RedLine, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (rbox3) at (18.8, 25.4) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{deepburgundy}{3. Fatal Violation: Replay Failure}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    \textbf{Violation 1: Replay Invariant Broken}\\
    Pruning exploratory read-only turns (e.g. empty \texttt{grep} or \texttt{ls}) causes replaying harness to crash: supervisory logic expects historical events that no longer exist!
    }
  \end{minipage}
};

\node[anchor=north west, fill=RedL0, draw=RedLine, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (rbox4) at (18.8, 23.0) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9.5}\selectfont \textbf{\textcolor{deepburgundy}{4. Fatal Violation: Context Hash Drift}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    \textbf{Violation 2: Context Hash Mismatch}\\
    During resume, harness verifies integrity:
    \begin{equation*}
      \text{SHA-256}(\text{Prompt}_t) \stackrel{?}{=} h_{\text{ctx}}^{(t)}
    \end{equation*}
    Pruned turns alter prompt bytes $\implies$ Fatal hash mismatch!
    }
  \end{minipage}
};

% Separator in Right Panel
\draw[draw=slatesteel!40, dashed, line width=0.8pt] (18.6, 20.3) -- (27.4, 20.3);

% Lower half: The Three Planes of Persistence
\node[anchor=north west, inner sep=0pt] at (18.8, 20.0) {
  \begin{minipage}{8.4cm}
    {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{DarkSlate}{THE THREE PLANES OF PERSISTENCE}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont \textcolor{slatesteel}{Orthogonal separation of concerns across runtime layers}}
  \end{minipage}
};

% Plane 1 Box
\node[anchor=north west, fill=BlueL0, draw=BlueLine, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (plane1) at (18.8, 19.1) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9}\selectfont \textbf{\textcolor{BlueLine}{I. Causal Event Plane ($\mathcal{S}_{\text{causal}}$)}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    \textbullet\ \textbf{Data:} Token history, prompt hashes, LSN, tool call IDs\\
    \textbullet\ \textbf{Storage:} NVMe ring buffer $\to$ Columnar Parquet\\
    \textbullet\ \textbf{Contract:} Strictly append-only; \textbf{zero pruning}
    }
  \end{minipage}
};

% Plane 2 Box
\node[anchor=north west, fill=PurpleL0, draw=PurpleLine, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (plane2) at (18.8, 16.7) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9}\selectfont \textbf{\textcolor{deeproyal}{II. Tool Payload Plane (CAS Blobs)}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    \textbullet\ \textbf{Data:} Bulky stdout/stderr, compiler dumps, HTML trees\\
    \textbullet\ \textbf{Storage:} S3/GCS content-addressed by SHA-256\\
    \textbullet\ \textbf{Contract:} Fleet deduplication; lazy rehydration on audit
    }
  \end{minipage}
};

% Plane 3 Box
\node[anchor=north west, fill=GreenL0, draw=GreenLine, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (plane3) at (18.8, 14.3) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9}\selectfont \textbf{\textcolor{forestsage}{III. Filesystem Plane (OS Sandbox)}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    \textbullet\ \textbf{Data:} Git trees, mutated source files, build artifacts\\
    \textbullet\ \textbf{Storage:} OverlayFS \texttt{upperdir}, Btrfs/ZFS snapshots\\
    \textbullet\ \textbf{Contract:} Compacted via \texttt{ioctl(FICLONE)} block sharing
    }
  \end{minipage}
};

% Plane 4 / Recovery Protocol Box
\node[anchor=north west, fill=OrangeL0, draw=OrangeLine, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (proto_box) at (18.8, 11.9) {
  \begin{minipage}{8.1cm}
    {\fontsize{7.5}{9}\selectfont \textbf{\textcolor{warmochre}{DETERMINISTIC CRASH RECOVERY PROTOCOL}}}\\[1pt]
    {\fontsize{6.5}{8}\selectfont
    \textbf{1. Rehydrate OS Base:} Mount rootfs snapshot via OverlayFS ($<50\text{ ms}$).\\
    \textbf{2. Stream Lean Spine $\mathcal{S}_{\text{causal}}$:} Read LSN records from WAL ($<5\text{ ms}$).\\
    \textbf{3. Verify Prompt Context:} Assert $\text{SHA-256}(p_t) == h_{\text{ctx}}^{(t)}$ (zero drift).\\
    \textbf{4. Lazy Payload Fetch:} Hydrate CAS blobs from S3 on-demand only.
    }
  \end{minipage}
};

% Takeaway Box at bottom of Right Panel
\node[anchor=north west, fill=coolparchment, draw=slatesteel!50, line width=0.8pt, rounded corners=4pt, inner sep=4pt] 
  (takeaway_box) at (18.8, 9.4) {
  \begin{minipage}{8.1cm}
    {\fontsize{7}{8.5}\selectfont \textbf{\textcolor{DarkSlate}{Core Architectural Takeaway}}}\\[1pt]
    {\fontsize{6.3}{7.8}\selectfont \textcolor{slatesteel}{Separating lightweight sequence order ($\mathcal{S}_{\text{causal}}$) from bulk CAS blobs and OS file mutations yields $\mathcal{O}(1)$ crash recovery without mutating history.}}
  \end{minipage}
};

% -------------------------------------------------------------
% BOTTOM PANEL: MATHEMATICAL DIVIDENDS & ARCHITECTURAL INVARIANTS
% -------------------------------------------------------------
\node[anchor=north west, inner sep=0pt] at (0, 6.9) {
  \begin{minipage}{27.6cm}
    {\fontsize{10.5}{12.5}\selectfont \textbf{\textcolor{DarkSlate}{MATHEMATICAL RECOVERY GUARANTEES \& DURABILITY INVARIANTS}}}\\[2pt]
    {\fontsize{7.5}{9.5}\selectfont \textcolor{slatesteel}{Formal models governing trajectory compaction, diagnostic trace sampling, and crash recovery}}
  \end{minipage}
};

% Three Bottom Cards Side by Side
% --- Bottom Card 1: Content-Addressed Storage Sizing ---
\node[anchor=north west, fill=coolparchment, draw=BlueLine!70, line width=1.0pt, rounded corners=5pt, inner sep=6pt] 
  (bcard1) at (0, 6.0) {
  \begin{minipage}{8.6cm}
    {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{BlueLine}{1. Storage Space Sizing \& Deduplication}}}\\[2pt]
    {\fontsize{7}{9}\selectfont \textcolor{DarkSlate}{Causal spine footprint is bounded by active context, not history:}}
    
    \vspace{3pt}
    \begin{equation*}
      \text{Size}(\mathcal{S}_{\text{causal}}) \approx \mathcal{O}(T \cdot \bar{b}_{\text{env}}) \ll \mathcal{O}\left(\sum_{t=1}^T |B_{\text{payload}}(t)|\right)
    \end{equation*}
    
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ \textbf{Footprint Sparing:} On a 1,000-turn software engineering task, unpruned tool payloads generate $> 12\text{ GB}$; CAS offloading restricts the active causal spine to $\approx 4.2\text{ MB}$ ($96.5\%$ space reduction).\\
    \textbullet\ \textbf{Fleet Deduplication:} If 200 agents encounter identical standard library builds or compiler warnings, content addressing stores the bytes exactly once.
    }
  \end{minipage}
};

% --- Bottom Card 2: Tail-Based Stratified Telemetry Coverage ---
\node[anchor=north west, fill=coolparchment, draw=OrangeLine!70, line width=1.0pt, rounded corners=5pt, inner sep=6pt] 
  (bcard2) at (9.4, 6.0) {
  \begin{minipage}{8.8cm}
    {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{warmochre}{2. Tail-Based Stratified Trace Sampling}}}\\[2pt]
    {\fontsize{7}{9}\selectfont \textcolor{DarkSlate}{Proving why uniform head-based sampling blinds production SREs:}}
    
    \vspace{3pt}
    \begin{equation*}
      \mathcal{C}(p, q, m) = 1 - (1 - p)^m \implies \mathcal{C}(0.01, 10) = 9.6\%
    \end{equation*}
    
    {\fontsize{6.8}{8.5}\selectfont
    \textbullet\ \textbf{Uniform Failure:} Uniform sampling at $p=0.01$ discards over $90.4\%$ of diagnostic defect traces during incident spikes.\\
    \textbullet\ \textbf{Tail-Stratified Solution:} Setting $p_{\text{fault}} = 1.0$ (for tool timeouts, sandboxes crashes, or aborts) and $p_{\text{nominal}} = 0.01$ guarantees $100\%$ incident capture ($\mathcal{C}_{\text{fault}} \equiv 1.0$) while reducing telemetry bandwidth by $> 97\%$.
    }
  \end{minipage}
};

% --- Bottom Card 3: The Three Invariants of Durable Execution ---
\node[anchor=north west, fill=coolparchment, draw=GreenLine!70, line width=1.0pt, rounded corners=5pt, inner sep=6pt] 
  (bcard3) at (18.8, 6.0) {
  \begin{minipage}{8.2cm}
    {\fontsize{8.5}{10.5}\selectfont \textbf{\textcolor{forestsage}{3. Durable Execution Invariants}}}\\[2pt]
    {\fontsize{7}{9}\selectfont \textcolor{DarkSlate}{Three core systems contracts enforced across the persistence layer:}}
    
    \vspace{3pt}
    {\fontsize{6.8}{8.5}\selectfont
    \textbf{I. Intent-Before-Effect Invariant ($e_{\text{flush}} \prec a_{\text{ext}}$):}
    Durable intent flushed to NVMe before releasing actuation to network.\\[3pt]
    \textbf{II. Causal Spine Immutability:}
    $\mathcal{S}_{\text{causal}}$ is strictly append-only; exploratory turns are never deleted.\\[3pt]
    \textbf{III. Bit-Exact Context Verification:}
    SHA-256 prompt hash verification on resume prevents silent model hallucination drift.
    }
  \end{minipage}
};

\end{tikzpicture}
\end{document}
"""

def main():
    tex_path = Path("books/vol3/tex/three_tier_storage_pyramid.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(TIKZ_CODE)
    print(f"Wrote TikZ source to {tex_path}")

    # Compile with pdflatex
    cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory=/tmp", str(tex_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"pdflatex return code: {res.returncode}")
    if res.returncode != 0:
        print("LaTeX compilation failed:")
        for line in res.stdout.splitlines()[-30:]:
            print(line)
        return

    # Render to PNG at 200 dpi
    png_prefix = "/tmp/three_tier_storage_pyramid_render"
    cmd_png = ["pdftoppm", "-png", "-r", "200", "/tmp/three_tier_storage_pyramid.pdf", png_prefix]
    subprocess.run(cmd_png, check=True)

    # Convert to SVG with inkscape
    svg_path = "/tmp/three_tier_storage_pyramid.svg"
    cmd_svg = ["/opt/homebrew/bin/inkscape", "--export-type=svg", f"--export-filename={svg_path}", "/tmp/three_tier_storage_pyramid.pdf"]
    subprocess.run(cmd_svg, check=True)

    print("Successfully compiled and rendered Three-Tier Storage Pyramid Diagram!")

if __name__ == "__main__":
    main()
