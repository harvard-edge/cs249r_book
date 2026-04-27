import re
import os
import json
import urllib.request
from bs4 import BeautifulSoup
from urllib.parse import urljoin

qmd_path = "/Users/VJ/GitHub/MLSysBook-figure-audit/book/quarto/contents/vol1/model_serving/model_serving.qmd"
html_url = "https://harvard-edge.github.io/cs249r_book_dev/vol1/contents/vol1/model_serving/model_serving.html"
img_dir = "/Users/VJ/GitHub/MLSysBook-figure-audit/tmp_images"

os.makedirs(img_dir, exist_ok=True)

# 1. Fetch and parse HTML
req = urllib.request.Request(html_url, headers={'User-Agent': 'Mozilla/5.0'})
html_content = urllib.request.urlopen(req).read().decode('utf-8')
soup = BeautifulSoup(html_content, 'html.parser')

html_figures = {}
for fig in soup.find_all(id=re.compile(r'^fig-')):
    fig_id = fig.get('id')
    
    caption_el = fig.find('figcaption') or fig.find(class_='quarto-figure-caption')
    caption = caption_el.get_text(separator=' ', strip=True) if caption_el else "NO_CAPTION"
    
    img_el = fig.find('img')
    img_url = None
    local_img_path = None
    if img_el and img_el.has_attr('src'):
        img_url = urljoin(html_url, img_el['src'])
        # Download image
        ext = os.path.splitext(img_url.split('?')[0])[1]
        if not ext:
            ext = '.png'
        local_img_path = os.path.join(img_dir, f"{fig_id}{ext}")
        try:
            urllib.request.urlretrieve(img_url, local_img_path)
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")
            local_img_path = "DOWNLOAD_FAILED"
            
    html_figures[fig_id] = {
        'rendered_caption': caption,
        'img_url': img_url,
        'local_img_path': local_img_path
    }

# 2. Parse QMD
with open(qmd_path, 'r', encoding='utf-8') as f:
    qmd_lines = f.readlines()

qmd_content = "".join(qmd_lines)

# Find figures in QMD
# Match ::: {#fig-id ...}
fig_blocks = list(re.finditer(r':::\s*\{#([^ ]+)[^}]*\}', qmd_content))
figures_data = []

for i, block in enumerate(fig_blocks):
    fig_id = block.group(1)
    if not fig_id.startswith('fig-'): continue
    
    start_pos = block.start()
    # Find next ::: that closes this block
    # Simple approach: find next ::: at the start of a line
    end_match = re.search(r'^:::', qmd_content[start_pos+1:], re.MULTILINE)
    end_pos = start_pos + end_match.end() if end_match else len(qmd_content)
    
    block_text = qmd_content[start_pos:end_pos]
    start_line = qmd_content[:start_pos].count('\n') + 1
    end_line = qmd_content[:end_pos].count('\n') + 1
    
    # Extract cap and alt
    cap_match = re.search(r'fig-cap="([^"]*)"', block_text)
    alt_match = re.search(r'fig-alt="([^"]*)"', block_text)
    
    # Kind of figure
    if '```{.tikz}' in block_text or '```tikz' in block_text:
        kind = 'tikz'
    elif '```{python}' in block_text:
        kind = 'python_plot'
    elif '```mermaid' in block_text:
        kind = 'mermaid'
    else:
        # check for ![](...)
        img_match = re.search(r'!\[.*?\]\(([^)]+)\)', block_text)
        if img_match:
            ext = os.path.splitext(img_match.group(1))[1].lower()
            kind = ext[1:] if ext else 'unknown'
        else:
            kind = 'unknown'

    figures_data.append({
        'fig_id': fig_id,
        'qmd_line_start': start_line,
        'qmd_line_end': end_line,
        'caption_source_text': cap_match.group(1) if cap_match else '',
        'alt_source_text': alt_match.group(1) if alt_match else '',
        'figure_kind': kind
    })

# 3. Find prose references in QMD
# Find paragraphs containing @fig-id
paragraphs = re.split(r'\n\s*\n', qmd_content)
para_line_nums = []
curr_line = 1
for p in paragraphs:
    para_line_nums.append((p, curr_line))
    curr_line += p.count('\n') + 2

for fig in figures_data:
    fig_id = fig['fig_id']
    refs = []
    for p, l_num in para_line_nums:
        if f'@{fig_id}' in p and not p.strip().startswith(':::') and not p.strip().startswith('```'):
            # determine forward, back, concurrent
            if l_num < fig['qmd_line_start'] - 50:
                role = 'forward'
            elif l_num > fig['qmd_line_end'] + 50:
                role = 'back'
            else:
                role = 'concurrent'
            
            # expand {python} macros if possible, but actually we can just find them in HTML
            # wait, matching HTML paragraphs is safer
            refs.append({
                'paragraph_text': p.strip().replace('\n', ' '),
                'qmd_line_number': l_num,
                'reference_role': role
            })
    fig['prose_references'] = refs
    
    # Merge with HTML data
    html_data = html_figures.get(fig_id, {})
    fig['rendered_caption'] = html_data.get('rendered_caption', '')
    fig['local_img_path'] = html_data.get('local_img_path', '')

with open('/Users/VJ/GitHub/MLSysBook-figure-audit/figures_context.json', 'w') as f:
    json.dump(figures_data, f, indent=2)

print("Extracted", len(figures_data), "figures.")
