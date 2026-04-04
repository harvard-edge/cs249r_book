import xml.etree.ElementTree as ET
import re

tree = ET.parse('fig-applicability-matrix.svg')
root = tree.getroot()

texts = []
for text in root.iter('{http://www.w3.org/2000/svg}text'):
    if text.text:
        texts.append((float(text.attrib.get('y', 0)), float(text.attrib.get('x', 0)), text.text.strip()))

texts.sort(key=lambda x: x[0])

rows = []
current_y = -1
current_row = []
for y, x, text_content in texts:
    if abs(y - current_y) > 2:
        if current_row:
            rows.append(current_row)
        current_row = []
        current_y = y
    # Escape & for latex
    clean_text = text_content.replace('&', r'\&')
    current_row.append((x, clean_text))
if current_row:
    rows.append(current_row)

parsed_items = []
for r in rows:
    if len(r) == 1 and '(' in r[0][1] and ')' in r[0][1] and r[0][0] < 50:
        parsed_items.append({'type': 'header', 'text': r[0][1]})
    elif len(r) >= 5:
        r.sort(key=lambda item: item[0])
        topic = r[0][1]
        c = r[1][1]
        e = r[2][1]
        m = r[3][1]
        t = r[4][1]
        def m2tex(mark):
            if '✓' in mark: return r'\textcolor{green!60!black}{\checkmark}'
            if '✗' in mark: return r'\textcolor{red}{\ding{55}}'
            return mark
        parsed_items.append({'type': 'topic', 'text': topic, 'marks': [m2tex(c), m2tex(e), m2tex(m), m2tex(t)]})

num_cols = 3
items_per_col = (len(parsed_items) + num_cols - 1) // num_cols

while len(parsed_items) < items_per_col * num_cols:
    parsed_items.append({'type': 'empty'})

cols = [parsed_items[i*items_per_col : (i+1)*items_per_col] for i in range(num_cols)]

latex_lines = []
latex_lines.append(r"\begin{table*}[!t]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{\textbf{Applicability matrix.} Each cell indicates whether a topic--track pair produces meaningful interview questions. Green checkmarks (\textcolor{green!60!black}{\checkmark}) indicate a physical substrate; red crosses (\textcolor{red}{\ding{55}}) indicate exclusion due to physics constraints. Of 348 possible pairs, \numapplicablepairs{} are applicable and \numexcludedpairs{} are excluded.}")
latex_lines.append(r"\label{tab:applicability-matrix}")
latex_lines.append(r"\tablebody")
latex_lines.append(r"\renewcommand{\arraystretch}{1.05}")
latex_lines.append(
    r"\begin{adjustbox}{max width=\textwidth,max height=\AppMatrixBodyMaxHt,keepaspectratio,center}"
)
latex_lines.append(r"\begin{tabular}{@{} p{2.8cm} cccc @{\hspace{1em}} p{2.8cm} cccc @{\hspace{1em}} p{2.8cm} cccc @{}}")
latex_lines.append(r"\toprule")

header_row = []
for i in range(num_cols):
    header_row.append(r"\textbf{Topic} & \rotatebox{90}{Cloud} & \rotatebox{90}{Edge} & \rotatebox{90}{Mobile} & \rotatebox{90}{TinyML}")
latex_lines.append(" & ".join(header_row) + r" \\")
latex_lines.append(r"\midrule")

for i in range(items_per_col):
    row_cells = []
    for c in range(num_cols):
        item = cols[c][i]
        if item['type'] == 'header':
            row_cells.append(f"\\textbf{{{item['text']}}} & & & &")
        elif item['type'] == 'topic':
            row_cells.append(f"{item['text']} & {item['marks'][0]} & {item['marks'][1]} & {item['marks'][2]} & {item['marks'][3]}")
        else:
            row_cells.append(" & & & &")
    latex_lines.append(" & ".join(row_cells) + r" \\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{adjustbox}")
latex_lines.append(
    r"\vspace{-3pt}% tighten float vertical list vs shipped box (avoids spurious ``Float too large'')"
)
latex_lines.append(r"\end{table*}")
latex_lines.append(r"\FloatBarrier")

with open('app_matrix.tex', 'w') as f:
    f.write("\n".join(latex_lines))

print("app_matrix.tex generated correctly.")
