import os
import glob
import subprocess
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Default paths assuming the script is run from the repository root
book_dir = "book/quarto/contents"
out_dir = ".claude/_reviews/Figure Audit"
img_tmp_dir = ".claude/_reviews/Figure Audit/audit_images_tmp"
brief_path = ".claude/_plans/figure-audit-brief.md"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_tmp_dir, exist_ok=True)

qmd_files = []
for vol in ['vol1', 'vol2']:
    vol_dir = os.path.join(book_dir, vol)
    for root, _, files in os.walk(vol_dir):
        for f in files:
            if f.endswith('.qmd') and not f.startswith('_'):
                qmd_files.append(os.path.join(root, f))

chapters_with_figures = []
for qmd in qmd_files:
    with open(qmd, 'r', encoding='utf-8') as f:
        if 'fig-cap' in f.read():
            chapters_with_figures.append(qmd)

print(f"Found {len(chapters_with_figures)} chapters with figures.")

def download_file(url, local_path):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def process_chapter(qmd_path):
    rel_path = os.path.relpath(qmd_path, book_dir)
    parts = rel_path.split(os.sep)
    vol = parts[0]
    
    if len(parts) > 2:
        chap = parts[-2]
    else:
        chap = parts[-1].replace('.qmd', '')
        
    out_name = f"{vol}-{chap}.yml"
    out_path = os.path.join(out_dir, out_name)
    
    if os.path.exists(out_path):
        return f"Skipping {out_name}, already exists."
        
    html_url = f"https://harvard-edge.github.io/cs249r_book_dev/{vol}/contents/{rel_path.replace('.qmd', '.html')}"
    
    local_images = []
    try:
        req = urllib.request.Request(html_url, headers={'User-Agent': 'Mozilla/5.0'})
        html_content = urllib.request.urlopen(req).read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for figure in soup.find_all('figure'):
            fig_id = figure.get('id', 'unknown_fig')
            img_tag = figure.find('img')
            svg_tag = figure.find('svg')
            
            if img_tag and img_tag.get('src'):
                src = img_tag['src']
                abs_url = urllib.parse.urljoin(html_url, src)
                ext = os.path.splitext(urllib.parse.urlparse(abs_url).path)[1] or '.png'
                local_img_path = os.path.join(img_tmp_dir, f"{vol}_{chap}_{fig_id}{ext}")
                if download_file(abs_url, local_img_path):
                    local_images.append(f"{fig_id}: {os.path.abspath(local_img_path)}")
            elif svg_tag:
                local_img_path = os.path.join(img_tmp_dir, f"{vol}_{chap}_{fig_id}.svg")
                with open(local_img_path, 'w', encoding='utf-8') as f:
                    f.write(str(svg_tag))
                local_images.append(f"{fig_id}: {os.path.abspath(local_img_path)}")
                
    except Exception as e:
        print(f"Error parsing HTML for {chap}: {e}")
    
    images_instruction = "\\n".join(local_images)
    
    prompt = f"""You are the Three-Artifact Figure Audit agent. Read the brief at: {os.path.abspath(brief_path)}
Your task is to audit the chapter located at: {os.path.abspath(qmd_path)}

Instructions:
1. I have downloaded the raw rendered HTML images to your local disk so you can visually audit them. 
   Use the `read_file` tool on the following local paths to view the visual figures:
{images_instruction}

2. Compare the image you see via `read_file` to the prose, `fig-cap`, and `fig-alt` in the source QMD file at {os.path.abspath(qmd_path)}.
3. Output your findings strictly in the YAML format specified in the brief. Write the final YAML report to: {os.path.abspath(out_path)}

CRITICAL CONSTRAINTS FOR FIXES:
- **You CANNOT change the figures/images themselves.** The images are immutable.
- Your `proposed_fix` in the YAML MUST ONLY target the prose, the caption (`fig-cap`), or the alt-text (`fig-alt`) in the source `.qmd` file.
- **DO NOT rewrite the caption or prose completely.** You MUST propose the most MINIMAL, surgical tweaks to the existing `.qmd` text so that it accurately aligns with what is actually shown in the immutable image.

Do not ask for permission. Act autonomously to complete the audit and write the file."""

    print(f"Starting {out_name}...")
    cmd = ["gemini", "-m", "gemini-3.1-pro-preview", "-y", "-p", prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"Success: {out_name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        return f"Failed: {out_name}\nError: {error_msg[-500:]}"

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_chapter, chapters_with_figures))
    
print("\\n--- Audit Run Complete ---")
