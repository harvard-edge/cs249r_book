#!/usr/bin/env python3
"""
generate_stack_margin_figures.py

Generates standalone SVG stack margin figures for every chapter across all 4 volumes,
and injects the HTML column-margin block next to ## Purpose in each chapter's .qmd file.
"""

import os
import re
import glob
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = REPO_ROOT / "books"

VOL1_PREAMBLE = r"""\documentclass{standalone}
\usepackage[svgnames]{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,arrows.meta}
\providecolor{crimson}{HTML}{A51C30}

\tikzset{
    icon_hardware/.pic={
        \draw[thick] (-0.25,-0.25) rectangle (0.25,0.25);
        \foreach \x in {-0.15, 0, 0.15} \foreach \y in {-0.15, 0, 0.15}
            \fill (\x,\y) circle (0.03);
        \foreach \i in {-0.15, 0.15} {
            \draw (-0.25, \i) -- (-0.35, \i); \draw (0.25, \i) -- (0.35, \i);
            \draw (\i, -0.25) -- (\i, -0.35); \draw (\i, 0.25) -- (\i, 0.35);
        }
    },
    icon_frameworks/.pic={
        \draw[thick] (-0.25,-0.2) rectangle (0.25,-0.08);
        \draw[thick] (-0.2,-0.08) rectangle (0.2,0.04);
        \draw[thick] (-0.15,0.04) rectangle (0.15,0.16);
    },
    icon_models/.pic={
        \node[circle,fill,inner sep=0.8pt] (i1) at (-0.25,-0.15) {};
        \node[circle,fill,inner sep=0.8pt] (i2) at (-0.25,0.15) {};
        \node[circle,fill,inner sep=0.8pt] (h1) at (0,-0.15) {};
        \node[circle,fill,inner sep=0.8pt] (h2) at (0,0.15) {};
        \node[circle,fill,inner sep=0.8pt] (o1) at (0.25,0) {};
        \draw (i1)--(h1); \draw (i1)--(h2);
        \draw (i2)--(h1); \draw (i2)--(h2);
        \draw (h1)--(o1); \draw (h2)--(o1);
    },
    icon_training/.pic={
        \draw[thick, ->] (0,0.15) arc (90:360:0.15);
        \draw[thick, ->] (0,-0.15) arc (-90:180:0.15);
        \node[scale=0.6] at (0,0) {$\nabla$};
    },
    icon_serving/.pic={
        \draw[thick] (-0.3,-0.2) rectangle (0.3,0.2);
        \foreach \y in {-0.08, 0.08}
            \draw (-0.2,\y) -- (0.2,\y);
        \fill (0.15, 0.14) circle (0.03);
        \fill (0.15, -0.02) circle (0.03);
        \fill (0.15, -0.14) circle (0.03);
    },
    icon_operations/.pic={
        \draw[thick] (0.1,0) circle (0.13);
        \draw[thick] (-0.1,0.05) circle (0.1);
        \foreach \a in {0,60,...,300} {
            \draw[thick] (0.1,0) ++(\a:0.13) -- ++(\a:0.05);
        }
        \foreach \a in {0,72,...,288} {
            \draw[thick] (-0.1,0.05) ++(\a:0.1) -- ++(\a:0.04);
        }
    },
    icon_applications/.pic={
        \draw[thick] (-0.3,-0.2) rectangle (0.3,0.2);
        \draw (-0.3,0.1) -- (0.3,0.1);
        \node[scale=0.5, anchor=center] at (0,-0.05) {$>\_$};
    }
}

\newcommand{\stacklayer}[4]{
    \ifnum#3=0
        \def\fillcol{black!5}
        \def\textcol{black!30}
    \else
        \def\fillcol{crimson!#3}
        \ifnum#3>50
            \def\textcol{white}
        \else
            \def\textcol{crimson}
        \fi
    \fi

    \node[
        draw=white,
        line width=0.5pt,
        fill=\fillcol,
        minimum width=2.6cm,
        minimum height=1.0cm,
        anchor=south west,
        rounded corners=0pt,
        inner sep=0pt
    ] (box) at (0,#1) {};
    
    \node[anchor=west, text=\textcol, font=\usefont{T1}{phv}{b}{n}\fontsize{7.5}{8.5}\selectfont] at ($(box.west)+(0.12,0)$) {#2};
    \path (box.east) +(-0.50,0) pic[scale=0.78, draw=\textcol, fill=\textcol, text=\textcol] {#4};
}

\newcommand{\mlsysstack}[8]{
    \begin{tikzpicture}
        \stacklayer{6.0}{Applications}{#7}{icon_applications}
        \stacklayer{5.0}{Operations}{#6}{icon_operations}
        \stacklayer{4.0}{Serving}{#5}{icon_serving}
        \stacklayer{3.0}{Training}{#4}{icon_training}
        \stacklayer{2.0}{Models}{#3}{icon_models}
        \stacklayer{1.0}{Frameworks}{#2}{icon_frameworks}
        \stacklayer{0.0}{Hardware}{#1}{icon_hardware}
        
        \pgfmathtruncatemacro{\ddata}{#8}
        \ifnum\ddata=0
            \def\databarcol{black!5}
        \else
            \def\databarcol{crimson!\ddata}
        \fi
        
        \fill[\databarcol, rounded corners=1.5pt]
            (2.85, -0.1) rectangle (3.15, 7.10);
        
        \foreach \y/\val in {0.5/#1, 1.5/#2, 2.5/#3, 3.5/#4, 4.5/#5, 5.5/#6, 6.5/#7} {
            \ifnum\val=0
                \def\wirecol{black!20}
            \else
                \def\wirecol{crimson!\val}
            \fi
            \draw[\wirecol, line width=1.5pt] (2.6, \y) -- (2.85, \y);
            \fill[\wirecol] (2.85, \y) circle (0.04);
        }
        
        \node[text=white, rotate=90, font=\usefont{T1}{phv}{b}{n}\fontsize{8}{9}\selectfont]
            at (3.0, 3.5) {DATA};
    \end{tikzpicture}
}
"""

with open(BOOKS_DIR / "shared" / "tex" / "header-includes.tex") as _f:
    _lines = _f.readlines()
_FLEET_TIKZ = "".join(_lines[2465:2646])

VOL2_PREAMBLE = rf"""\documentclass{{standalone}}
\usepackage[svgnames]{{xcolor}}
\usepackage{{tikz}}
\usetikzlibrary{{calc,arrows.meta}}
\definecolor{{accentcolor}}{{HTML}}{{1F407A}}
\definecolor{{fleetinfra}}{{HTML}}{{1F407A}}
\definecolor{{fleetdist}}{{HTML}}{{1F407A}}
\definecolor{{fleetserve}}{{HTML}}{{1F407A}}
\definecolor{{fleetassure}}{{HTML}}{{1F407A}}
\definecolor{{fleetgov}}{{HTML}}{{1F407A}}

{_FLEET_TIKZ}
"""

VOL3_PREAMBLE = rf"""\documentclass{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{calc,arrows.meta}}
\input{{{BOOKS_DIR / "vol3" / "tex" / "agent-stack.tex"}}}
"""

VOL4_PREAMBLE = rf"""\documentclass{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{calc,arrows.meta}}
\usepackage{{xcolor}}
\usepackage{{fontawesome5}}
\input{{{BOOKS_DIR / "vol4" / "tex" / "physical-stack.tex"}}}
"""

PREAMBLES = {
    "vol1": (VOL1_PREAMBLE, "Machine Learning Systems Architecture Stack"),
    "vol2": (VOL2_PREAMBLE, "Fleet Machine Learning Systems Stack"),
    "vol3": (VOL3_PREAMBLE, "Agentic Machine Learning Systems Stack"),
    "vol4": (VOL4_PREAMBLE, "Physical AI Systems Stack"),
}


def compile_single_svg(args):
    vol, qmd_path, macro_call, output_svg = args
    preamble, alt_text = PREAMBLES[vol]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "stack.tex")
        pdf_path = os.path.join(tmpdir, "stack.pdf")
        with open(tex_path, "w") as f:
            f.write(f"{preamble}\n\\begin{{document}}\n{macro_call}\n\\end{{document}}\n")
        
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "stack.tex"],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if res.returncode != 0:
            return False, f"LaTeX error for {qmd_path}: {res.stdout.decode()[-300:]}"
            
        os.makedirs(os.path.dirname(output_svg), exist_ok=True)
        res2 = subprocess.run(
            ["pdftocairo", "-svg", "stack.pdf", output_svg],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if res2.returncode != 0:
            return False, f"pdftocairo error for {qmd_path}: {res2.stderr.decode()}"
            
    return True, output_svg


def main():
    tasks = []
    qmd_updates = []

    for vol in ["vol1", "vol2", "vol3", "vol4"]:
        files = sorted(glob.glob(str(BOOKS_DIR / vol / "*" / "*.qmd")))
        for f in files:
            if not re.search(r"/\d\d_", f):
                continue
            with open(f, "r", encoding="utf-8") as qmd_file:
                content = qmd_file.read()
            
            # Find the Purpose section and marginfigure
            # Pattern matches ## Purpose followed by marginfigure or content-visible
            pattern = re.compile(
                r"""(## Purpose[^\n]*\n+)"""
                r"""((?:::: \{\.content-visible when-format="pdf"\}\s*)?"""
                r"""\\begin\{marginfigure\}"""
                r"""(.*?)"""
                r"""\\end\{marginfigure\}"""
                r"""(?:\s*:::\s*)?"""
                r"""(?:\s*::: \{\.content-visible unless-format="pdf"\}\s*::: \{\.column-margin\}\s*!\[\]\([^\)]+\)\{[^\}]*\}\s*:::\s*:::\s*)?)""",
                re.DOTALL
            )
            
            match = pattern.search(content)
            if not match:
                print(f"⚠️  No Purpose marginfigure match in {f}")
                continue
                
            header = match.group(1)
            full_margin_block = match.group(2)
            inside_marginfigure = match.group(3).strip()
            
            # Extract the actual macro call to render
            # It could be bare \mlsysstack{...} or \resizebox{...}{\mlphysicalstackfull{...}}
            macro_match = re.search(r"(\\ml[a-z]+(?:full)?(?:\{[^}]*\})+)", inside_marginfigure)
            if not macro_match:
                print(f"⚠️  No macro found inside marginfigure in {f}")
                continue
                
            macro_call = macro_match.group(1)
            ch_dir = Path(f).parent
            output_svg = ch_dir / "images" / "svg" / "margin_stack.svg"
            
            tasks.append((vol, f, macro_call, str(output_svg)))
            
            # Prepare new markdown block
            alt_text = PREAMBLES[vol][1]
            captions = {
                "vol1": "*Architecture stack focus for this chapter.*",
                "vol2": "*Fleet stack focus for this chapter.*",
                "vol3": "*Agent stack focus for this chapter.*",
                "vol4": "*Physical AI stack focus for this chapter.*",
            }
            caption = captions[vol]
            new_block = (
                f"{header}"
                f"::: {{.content-visible when-format=\"pdf\"}}\n"
                f"\\begin{{marginfigure}}\n"
                f"{inside_marginfigure}\n"
                f"\\end{{marginfigure}}\n"
                f":::\n\n"
                f"::: {{.content-visible unless-format=\"pdf\"}}\n"
                f"::: {{.column-margin}}\n"
                f"![]({output_svg.name if False else 'images/svg/margin_stack.svg'}){{width=\"100%\" fig-alt=\"{alt_text}\"}}\n\n"
                f"{caption}\n"
                f":::\n"
                f":::\n\n"
            )
            
            qmd_updates.append((f, match.start(), match.end(), new_block, content))

    print(f"Compiling {len(tasks)} SVG stack figures in parallel...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compile_single_svg, tasks))

    success_count = sum(1 for ok, _ in results if ok)
    print(f"✅ Generated {success_count}/{len(tasks)} stack SVGs successfully.")
    for ok, msg in results:
        if not ok:
            print(f"❌ {msg}")

    print("Updating .qmd files...")
    for qmd_path, start, end, new_block, original_content in qmd_updates:
        # Re-read in case of parallel edits
        with open(qmd_path, "r", encoding="utf-8") as f:
            c = f.read()
        
        # Replace the matched block
        # Use exact string replacement of the matched portion
        target_str = original_content[start:end]
        if target_str in c:
            new_c = c.replace(target_str, new_block, 1)
            with open(qmd_path, "w", encoding="utf-8") as f:
                f.write(new_c)
        else:
            print(f"⚠️ Could not replace in {qmd_path} (content changed)")

    print("🎉 All chapters updated with HTML margin stack figures!")


if __name__ == "__main__":
    main()
