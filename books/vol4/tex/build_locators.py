#!/usr/bin/env python3
"""
Generate publication-grade pipeline locator / roadmap figures for each chapter of Physical AI Systems.
"""

import os
import subprocess

CHAPTER_ACTIVE_ORGAN = {
    "01-boundary": 0,    # Whole-system causal boundary
    "02-constraints": 0, # Whole-system physical constraints & columns
    "03-cognition": 0,   # Whole-system cognitive dimensions & rows
    "04-perception": 2,  # Stage 2: Perception (and Stage 1 Sensing)

    "05-state": 3,       # Stage 3: Memory & State
    "06-intent": 4,      # Stage 4: Reasoning (System 2)
    "07-planning": 5,    # Stage 5: Planning (System 1.5)
    "08-enforcement": 6, # Stage 6: Reflex & Safety (System 1)
    "09-placement": 0,   # Whole-system heterogeneous placement
    "10-governance": 7,  # Stage 7: Governance & Lineage
    "11-assurance": 7,   # Stage 7: Defensible Release Gate
    "99-capstone": 0,    # Whole-system capstone defense
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTERS_DIR = os.path.join(BASE_DIR, "..", "chapters")

def generate_locator(chapter_slug, organ_num):
    chap_fig_dir = os.path.join(CHAPTERS_DIR, chapter_slug, "figures")
    os.makedirs(chap_fig_dir, exist_ok=True)
    tex_path = os.path.join(chap_fig_dir, "fig_pipeline_locator.tex")
    
    # Determine style classes for each organ
    def get_style(n):
        if organ_num == 0:
            return "allactiveorgan"
        elif organ_num == n:
            return "activeorgan"
        elif organ_num == 2 and n == 1 and chapter_slug == "04-perception":
            return "activeorgan"
        else:
            return "fadedorgan"
            
    def get_arrow_style(from_n, to_n):
        if organ_num == 0:
            return "allactivearrow"
        # If either endpoint is the active organ, highlight the arrow
        if organ_num in [from_n, to_n]:
            return "activearrow"
        if organ_num == 2 and chapter_slug == "04-perception" and (from_n in [1,2] or to_n in [1,2]):
            return "activearrow"
        return "fadedarrow"

    tex_content = f"""\\documentclass[tikz,border=12pt]{{standalone}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{tgheros}}
\\usepackage{{sfmath}}
\\usepackage{{amsmath}}
\\usepackage{{fontawesome5}}
\\usepackage{{tikz}}
\\usetikzlibrary{{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}}
\\usepackage{{xcolor}}

\\renewcommand{{\\familydefault}}{{\\sfdefault}}

\\definecolor{{harvardcrimson}}{{HTML}}{{A51C30}}
\\definecolor{{ethdarkblue}}{{HTML}}{{1F407A}}
\\definecolor{{ethblue}}{{HTML}}{{215CAF}}
\\definecolor{{ethpetrol}}{{HTML}}{{007A87}}
\\definecolor{{ethbronze}}{{HTML}}{{B87333}}
\\definecolor{{ethpurple}}{{HTML}}{{5B4B8A}}
\\definecolor{{ethslate}}{{HTML}}{{475569}}
\\definecolor{{fadedslate}}{{HTML}}{{94A3B8}}
\\definecolor{{cardbg}}{{HTML}}{{F8FAFC}}
\\definecolor{{fadedbg}}{{HTML}}{{F8FAFC}}
\\definecolor{{cardborder}}{{HTML}}{{CBD5E1}}
\\definecolor{{fadedborder}}{{HTML}}{{E2E8F0}}
\\definecolor{{activebg}}{{HTML}}{{FEF2F2}}
\\definecolor{{activeborder}}{{HTML}}{{DC2626}}

\\begin{{document}}
\\begin{{tikzpicture}}[
  font=\\sffamily,
  >=Stealth,
  fadedorgan/.style={{
    draw=fadedborder,
    fill=fadedbg,
    rounded corners=4pt,
    line width=0.6pt,
    inner sep=5pt,
    align=center,
    text=fadedslate
  }},
  activeorgan/.style={{
    draw=activeborder,
    fill=activebg,
    rounded corners=4pt,
    line width=1.4pt,
    inner sep=5pt,
    align=center,
    text=harvardcrimson
  }},
  allactiveorgan/.style={{
    draw=ethdarkblue!70,
    fill=white,
    rounded corners=4pt,
    line width=0.9pt,
    inner sep=5pt,
    align=center,
    text=ethslate
  }},
  fadedarrow/.style={{
    ->,
    line width=0.5pt,
    draw=fadedslate!40
  }},
  activearrow/.style={{
    ->,
    line width=1.2pt,
    draw=harvardcrimson
  }},
  allactivearrow/.style={{
    ->,
    line width=0.8pt,
    draw=ethdarkblue!80
  }}
]

  % Top Banner: Stage 7 Governance
  \\node[{ get_style(7) }, text width=7.4in] (s7) {{
    \\textbf{{\\faIcon{{balance-scale}}\\; 7. GOVERNANCE \\& RELEASE GATE}} $\\cdot$ STPA Hazard Mitigation $\\cdot$ Bumpless Override $\\cdot$ Defensible Release Case (\\textbf{{LOOP-01}} $\\to$ \\textbf{{REL-01}})
  }};

  % --- TOP ROW: Organs 1, 2, 3 ---
  \\node[{ get_style(1) }, text width=2.22in, below=0.22in of s7.south west, anchor=north west] (s1) {{
    \\textbf{{\\faIcon{{satellite-dish}}\\; 1. SENSING}}\\\\
    {{Photons / Voltages to DMA}}\\\\
    {{\\scriptsize MIPI CSI-2 $\\cdot$ SPI Bus Priority}}
  }};

  \\node[{ get_style(2) }, text width=2.22in, right=0.37in of s1] (s2) {{
    \\textbf{{\\faIcon{{eye}}\\; 2. PERCEPTION}}\\\\
    {{\\mbox{{ViT Encoders}} $\\cdot$ \\mbox{{DINOv2}}}}\\\\
    {{\\scriptsize 3D Spatial Affordance Tokens}}
  }};

  \\node[{ get_style(3) }, text width=2.22in, right=0.37in of s2] (s3) {{
    \\textbf{{\\faIcon{{database}}\\; 3. MEMORY}}\\\\
    {{Temporal Belief $\\cdot$ $SE(3)$ Trees}}\\\\
    {{\\scriptsize JEPA / RSSM World Models}}
  }};

  % --- MIDDLE ROW: Organs 5, 4 ---
  \\node[{ get_style(4) }, text width=3.51in, below=0.25in of s3.south east, anchor=north east] (s4) {{
    \\textbf{{\\faIcon{{brain}}\\; 4. REASONING (System 2 $\\cdot$ MPU)}}\\\\
    {{Vision-Language Foundation Models $\\cdot$ 3D Goals}}\\\\
    {{\\scriptsize Leases with Expiring TTL ($t_{{\\text{{expire}}}}$)}}
  }};

  \\node[{ get_style(5) }, text width=3.51in, left=0.38in of s4] (s5) {{
    \\textbf{{\\faIcon{{network-wired}}\\; 5. PLANNING (System 1.5 $\\cdot$ MPU/NPU)}}\\\\
    {{Diffusion Policies $\\cdot$ \\mbox{{ACT Action Chunking}}}}\\\\
    {{\\scriptsize $H$-Step Action Horizons $\\cdot$ Jerk Continuity}}
  }};

  % --- LOWER ROW: Organ 6 ---
  \\node[{ get_style(6) }, text width=7.4in, below=0.42in of s5.south west, anchor=north west] (s6) {{
    \\textbf{{\\faIcon{{shield-alt}}\\; 6. REFLEX (System 1 $\\cdot$ MCU)}}\\\\
    {{1 kHz Real-Time Loop $\\cdot$ Control Barrier Functions ($h(x) \\ge 0$) $\\cdot$ Stopping Distance $d_{{\\text{{stop}}}} \\le d_{{\\text{{clearance}}}} \\cdot$ Veto ($u_t$)}}
  }};

  % --- PHYSICAL WORLD ROW ---
  \\node[{ "activeorgan" if organ_num == 0 else "allactiveorgan" }, draw=harvardcrimson!60, fill=white, text width=7.4in, below=0.22in of s6.south west, anchor=north west] (world) {{
    \\textbf{{\\color{{harvardcrimson}}\\faIcon{{cogs}}\\; THE PHYSICAL WORLD ($W_t \\to W_{{t+1}}$)}} $\\cdot$ Kinetic Momentum $\\cdot$ Joule Heat $\\cdot$ Friction $\\cdot$ Collision Dynamics
  }};

  % Proposal-Permission Boundary Line
  \\coordinate (bleft) at ($(s6.north west) + (0, 0.22in)$);
  \\coordinate (bright) at ($(s6.north east) + (0, 0.22in)$);
  \\draw[dashed, line width=0.9pt, harvardcrimson!70] (bleft) -- (bright);
  \\node[font=\\sffamily\\scriptsize\\bfseries, fill=white, draw=harvardcrimson!40, rounded corners=2pt, inner sep=2.5pt, text=harvardcrimson!90] 
    at ($(bleft)!0.5!(bright)$) 
    {{\\faIcon{{lock}}\\; THE PROPOSAL--PERMISSION PRIVILEGE BOUNDARY}};

  % Connecting Arrows
  \\draw[{ get_arrow_style(1, 2) }] (s1.east) -- (s2.west);
  \\draw[{ get_arrow_style(2, 3) }] (s2.east) -- (s3.west);
  \\draw[{ get_arrow_style(3, 4) }] (s3.south) -- (s3.south |- s4.north);
  \\draw[{ get_arrow_style(4, 5) }] (s4.west) -- (s5.east);
  \\draw[{ get_arrow_style(5, 6) }, dashed] (s5.south) -- (s5.south |- s6.north);
  \\draw[{ get_arrow_style(6, 0) }] (s6.south) -- (world.north);

  % Closed-loop feedback from physical world back to sensing
  \\draw[{ get_arrow_style(0, 1) }] (world.west) -- ++(-0.35in, 0) |- (s1.west) 
    node[pos=0.25, above, rotate=90, font=\\sffamily\\scriptsize\\bfseries, text=harvardcrimson!80] {{\\faIcon{{sync-alt}}\\; Endogenous Sensory Shift ($O_{{t+1}}$)}};

\\end{{tikzpicture}}
\\end{{document}}
"""
    with open(tex_path, "w") as f:
        f.write(tex_content)
    
    subprocess.run(["lualatex", "-interaction=nonstopmode", "fig_pipeline_locator.tex"], cwd=chap_fig_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pdftocairo", "-svg", "fig_pipeline_locator.pdf", "fig_pipeline_locator.svg"], cwd=chap_fig_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    for slug, organ in CHAPTER_ACTIVE_ORGAN.items():
        generate_locator(slug, organ)
    print("All chapter roadmap locators regenerated as PDF and SVG successfully.")
