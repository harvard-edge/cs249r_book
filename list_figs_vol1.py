import os
import re

def extract_figures(file_path):
    figures = []
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Catch ::: {#fig-... blocks
    div_pattern = r'::: \{#(?P<label>fig-[\w-]+).*?\}'
    for match in re.finditer(div_pattern, content, re.DOTALL):
        label = match.group('label')
        full_tag = content[match.start():content.find('}', match.start())+1]
        cap_match = re.search(r'fig-cap="(?P<caption>.*?)"', full_tag, re.DOTALL)
        caption = cap_match.group('caption') if cap_match else "[No caption in tag]"
        figures.append({'label': label, 'caption': caption})

    # 2. Catch markdown style ![]() {#fig-...}
    md_pattern = r'!\[(?P<caption>.*?)\]\(.*?\)\s*\{#(?P<label>fig-[\w-]+)\}'
    for match in re.finditer(md_pattern, content):
        if not any(f['label'] == match.group('label') for f in figures):
            figures.append({
                'label': match.group('label'),
                'caption': match.group('caption')
            })

    # 3. Catch code cell style #| label: fig-...
    cell_pattern = r'#\|\s*label:\s*(?P<label>fig-[\w-]+)'
    for match in re.finditer(cell_pattern, content):
        label = match.group('label')
        if not any(f['label'] == label for f in figures):
            start = content.rfind('```{python}', 0, match.start())
            end = content.find('```', match.start())
            if start != -1 and end != -1:
                cell_block = content[start:end]
                cap_match = re.search(r'#\|\s*fig-cap:\s*"(?P<caption>.*?)"', cell_block)
                if not cap_match:
                    cap_match = re.search(r'#\|\s*fig-cap:\s*(?P<caption>.*?)\n', cell_block)
                caption = cap_match.group('caption') if cap_match else "[No caption in cell]"
                figures.append({'label': label, 'caption': caption})

    # 4. Catch any other {#fig-...} as backup
    all_labels = re.findall(r'\{#(fig-[\w-]+)', content)
    for lbl in all_labels:
        if not any(f['label'] == lbl for f in figures):
            figures.append({'label': lbl, 'caption': '[Label only found]'})

    return figures

vol1_dir = 'book/quarto/contents/vol1/'
chapters = []
for root, dirs, files in os.walk(vol1_dir):
    for file in files:
        if file.endswith('.qmd'):
            chapters.append(os.path.join(root, file))

results = {}
for ch in sorted(chapters):
    rel_path = ch.replace(vol1_dir, '')
    results[rel_path] = extract_figures(ch)

for ch, figs in results.items():
    print(f"\n### Chapter: {ch}")
    if not figs:
        print("  (No figures found)")
    for f in figs:
        cap = ' '.join(f['caption'].split())
        print(f"  - {f['label']}: {cap}")
