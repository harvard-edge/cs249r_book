import os
import glob
import subprocess
import concurrent.futures
from pathlib import Path

BASE_DIR = Path("/Users/VJ/GitHub/MLSysBook/book/quarto")
VOL1_DIR = BASE_DIR / "contents" / "vol1"
VOL2_DIR = BASE_DIR / "contents" / "vol2"
AUDIT_DIR = BASE_DIR / "audits" / "ml_tie_ins"

# Find chapters, skip frontmatter, backmatter, parts
def get_chapters(vol_dir):
    all_qmds = glob.glob(str(vol_dir / "*/*.qmd"))
    chapters = []
    for qmd in all_qmds:
        path = Path(qmd)
        parent_name = path.parent.name
        if parent_name in ["frontmatter", "backmatter", "parts"] or path.name in ["index.qmd", "README.qmd"]:
            continue
        chapters.append(path)
    return chapters

vol1_chapters = get_chapters(VOL1_DIR)
vol2_chapters = get_chapters(VOL2_DIR)

all_chapters = [(1, p) for p in vol1_chapters] + [(2, p) for p in vol2_chapters]

def audit_chapter(task):
    vol_num, file_path = task
    chapter_name = file_path.stem
    out_file = AUDIT_DIR / f"vol{vol_num}_{chapter_name}_audit.md"
    
    if out_file.exists() and out_file.stat().st_size > 100:
        return True

    print(f"Starting audit for Vol {vol_num}: {chapter_name}...")
    
    prompt = f"""You are an expert technical editor auditing an ML Systems textbook chapter.
Your task is to review the following chapter and ensure that ALL systems concepts discussed are explicitly and clearly tied to Machine Learning systems.

1. Use the read_file tool to read the contents of: {file_path}
2. Perform an honest audit of the content. Look for general principles that are discussed in a vacuum without applying them to ML workloads. 
3. Identify where and how good ML tie-ins should be added if they are missing or weak.
4. Output your final markdown report directly. Do not use the write_file tool. Provide a structured markdown report evaluating the chapter's "ML System Context" strength and listing specific recommendations.
"""
    
    try:
        # Use YOLO mode so it has tools to read the file, avoiding ARG_MAX limits.
        cmd = ["gemini", "-m", "gemini-3.1-pro-preview", "-p", prompt, "--yolo"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        output = result.stdout
        # Extract just the markdown response if it has tool call clutter, though standard YOLO output will mostly be text.
        if "```markdown" in output:
            output = output.split("```markdown")[-1].split("```")[0]
            
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output.strip())
            
        print(f"Completed audit for {chapter_name} (Size: {len(output)})")
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout auditing {chapter_name}")
        return False
    except Exception as e:
        print(f"Error auditing {chapter_name}: {e}")
        return False

# Run in parallel
print(f"Found {len(all_chapters)} chapters to audit.")
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(audit_chapter, all_chapters))

print("All audits complete. Results saved in", AUDIT_DIR)
