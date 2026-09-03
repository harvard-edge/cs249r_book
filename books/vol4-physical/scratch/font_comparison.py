import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'

tex_template = r'''\documentclass{scrbook}
\usepackage{geometry}
\geometry{paperwidth=8in, paperheight=10in, inner=0.875in, outer=1.75in, top=0.875in, bottom=0.875in}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{lettrine}
\definecolor{emerald}{HTML}{1A4D3E}
\definecolor{ink}{HTML}{1A202C}
\definecolor{softink}{HTML}{4A5568}

%FONT_CONFIG%

\begin{document}
\pagestyle{empty}

% Page 1: Cover
\noindent\null
\begin{tikzpicture}[remember picture, overlay]
  \node[anchor=north, inner sep=0pt] at ([yshift=-0.65in]current page.north) {%
    \includegraphics[width=6.0in, height=5.5in, keepaspectratio]{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png}%
  };
  \node[anchor=south west, inner sep=0pt] at ([xshift=0.875in, yshift=3.35in]current page.south west) {%
    {\fontsize{26pt}{30pt}\rmfamily\selectfont\color{softink}Physical AI}%
  };
  \node[anchor=north west, inner sep=0pt] at ([xshift=0.875in, yshift=3.15in]current page.south west) {%
    \linespread{1.05}\selectfont%
    {\fontsize{50pt}{56pt}\rmfamily\selectfont\color{ink}%
     \begin{tabular}{@{}l@{}}
       \fontsize{50pt}{56pt}\rmfamily\selectfont Machine Learning\\[0.10em]
       \fontsize{50pt}{56pt}\rmfamily\selectfont Systems
     \end{tabular}%
    }%
  };
  \node[anchor=south east, inner sep=0pt] at ([xshift=-0.875in, yshift=0.85in]current page.south east) {%
    {\fontsize{18pt}{22pt}\rmfamily\selectfont\color{ink}%
     \begin{tabular}{@{}r@{}}
       Vijay\\ Janapa Reddi
     \end{tabular}%
    }%
  };
\end{tikzpicture}
\clearpage

% Page 2: Preface Body & Lettrine
{\fontsize{28pt}{34pt}\rmfamily\bfseries\color{emerald}Preface\par}
\vspace{1.5\baselineskip}

A physical AI system is a machine with three parts.
It has a \textbf{body}, which moves mass through space, applies force through contact, or regulates a flow of energy. The body obeys mechanics and thermodynamics rather than instructions. It has momentum that must be arrested somewhere, windings that accumulate heat faster than they shed it, and transmissions that absorb whatever discontinuity a command hands them.

\vspace{1\baselineskip}
{\fontsize{16pt}{20pt}\rmfamily\bfseries\color{emerald}What This Book Is\par}
\vspace{0.8\baselineskip}

\renewcommand{\LettrineFontHook}{\rmfamily\bfseries\color{emerald}}
\renewcommand{\LettrineTextFont}{\normalfont\scshape}
\lettrine{T}{his is a systems engineering book.} It draws on machine learning, on real-time and embedded systems, and on control, but it is not a book about any of the three and it does not attempt to teach them. Its subject is composition: what each part of the machine must guarantee to the others, how authority over physical action is allocated, what happens to those guarantees when everything shares one piece of silicon, and what evidence would justify letting the result operate.

\end{document}
'''

# 1. Palatino (Canonical Volume 1 & Volume 2)
font_palatino = r'''
\setmainfont{Palatino}
\setsansfont{Helvetica Neue}[Scale=0.92]
\setmonofont{Menlo}[Scale=0.80]
'''

# 2. STIX Two Text
font_stix = r'''
\setmainfont{STIX Two Text}
\setsansfont{Avenir Next}[Scale=0.90]
\setmonofont{Menlo}[Scale=0.78]
'''

for name, cfg in [('font_palatino', font_palatino), ('font_stix', font_stix)]:
    src = tex_template.replace('%FONT_CONFIG%', cfg)
    p = os.path.join(scratch_dir, f'{name}.tex')
    with open(p, 'w') as f:
        f.write(src)
    subprocess.run(['lualatex', '-interaction=nonstopmode', f'{name}.tex'], cwd=scratch_dir, stdout=subprocess.DEVNULL)
    subprocess.run(['lualatex', '-interaction=nonstopmode', f'{name}.tex'], cwd=scratch_dir, stdout=subprocess.DEVNULL)
    subprocess.run(['pdftoppm', '-png', '-r', '150', '-f', '1', '-l', '2', f'{name}.pdf', f'render_{name}'], cwd=scratch_dir)

# Side-by-side comparison: Cover & Preface
cov_pal = Image.open(os.path.join(scratch_dir, 'render_font_palatino-1.png'))
cov_stix = Image.open(os.path.join(scratch_dir, 'render_font_stix-1.png'))
pref_pal = Image.open(os.path.join(scratch_dir, 'render_font_palatino-2.png'))
pref_stix = Image.open(os.path.join(scratch_dir, 'render_font_stix-2.png'))

w, h = cov_pal.size
comp_cov = Image.new('RGB', (w * 2 + 30, h + 60), (240, 240, 242))
draw1 = ImageDraw.Draw(comp_cov)
comp_cov.paste(cov_pal, (10, 50))
comp_cov.paste(cov_stix, (w + 20, 50))
f = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
draw1.text((10, 15), 'A. PALATINO (Canonical Volume 1 & Volume 2)', font=f, fill=(26, 77, 62))
draw1.text((w + 20, 15), 'B. STIX TWO TEXT (Alternative)', font=f, fill=(100, 100, 100))
comp_cov.save('/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e/contact_sheet_font_comparison_cover.png')

comp_pref = Image.new('RGB', (w * 2 + 30, h + 60), (240, 240, 242))
draw2 = ImageDraw.Draw(comp_pref)
comp_pref.paste(pref_pal, (10, 50))
comp_pref.paste(pref_stix, (w + 20, 50))
draw2.text((10, 15), 'A. PALATINO (Canonical Volume 1 & Volume 2)', font=f, fill=(26, 77, 62))
draw2.text((w + 20, 15), 'B. STIX TWO TEXT (Alternative)', font=f, fill=(100, 100, 100))
comp_pref.save('/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e/contact_sheet_font_comparison_preface.png')

print('Font comparison contact sheets saved!')
