import subprocess, os
from PIL import Image

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'

tex = """\\documentclass[10pt, letterpaper, twoside, openright]{scrbook}
\\usepackage{geometry}
\\geometry{paperwidth=8in, paperheight=10in, margin=0pt}
\\usepackage{fontspec}
\\setmainfont{TeX Gyre Pagella}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usepackage{graphicx}

\\definecolor{ink}{HTML}{1A202C}
\\definecolor{softink}{HTML}{4A5568}
\\definecolor{accentcolor}{HTML}{1A4D3E}

\\newlength{\\titlelen}
\\newlength{\\leftm}
\\newlength{\\rightm}

\\begin{document}
\\thispagestyle{empty}
\\null

\\settowidth{\\titlelen}{{\\fontsize{44pt}{44pt}\\selectfont Machine Learning Systems}}
\\setlength{\\leftm}{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}
\\setlength{\\rightm}{\\dimexpr\\leftm + \\titlelen\\relax}

\\begin{tikzpicture}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  % Candidate 4: Gyroscopic Multi-Axis Gimbal Artwork
  \\node[anchor=center, inner sep=0pt] at ([xshift=0.55in, yshift=1.50in]current page.center) {%
    \\includegraphics[width=0.96\\paperwidth]{""" + f"{assets_dir}/cover-physical-gyroscopic-gimbal-art.png" + """}%
  };

  % Subtitle (MIT Press Exact Match y=2.40in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {%
    {\\fontsize{23.5pt}{27.5pt}\\rmfamily\\selectfont\\color{ink}Physical AI}%
  };

  % Single-Line Main Title (MIT Press Exact Match y=1.62in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {%
    {\\fontsize{44pt}{48pt}\\rmfamily\\selectfont\\color{ink}Machine Learning Systems}%
  };

  % Single-Line Right-aligned Author (MIT Press Exact Match y=0.78in)
  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {%
    {\\fontsize{21.5pt}{25pt}\\rmfamily\\selectfont\\color{ink}Vijay Janapa Reddi}%
  };

\\end{tikzpicture}
\\newpage
\\end{document}
"""

with open(f"{scratch_dir}/candidate4_final.tex", "w") as f:
    f.write(tex)

subprocess.run(["lualatex", "-interaction=nonstopmode", "candidate4_final.tex"], cwd=scratch_dir, check=True)
subprocess.run(["lualatex", "-interaction=nonstopmode", "candidate4_final.tex"], cwd=scratch_dir, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "150", "candidate4_final.pdf", f"{scratch_dir}/candidate4_final_out"], cwd=scratch_dir, check=True)

final_img = Image.open(f"{scratch_dir}/candidate4_final_out-1.png")
final_path = os.path.join(brain_dir, "verified_physical_ai_candidate4_cover.png")
final_img.save(final_path)

print("Candidate 4 final cover rendered and saved!")
