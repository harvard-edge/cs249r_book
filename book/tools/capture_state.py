# book/tools/capture_state.py
import re
import json
import sys
import glob
import os
from pathlib import Path

# Add the quarto directory to sys.path so we can import mlsys
sys.path.append(os.path.abspath("book/quarto"))

def extract_python_cells(qmd_path):
    """Extracts code from ```{python} blocks in a QMD file."""
    with open(qmd_path, 'r') as f:
        content = f.read()
    
    # Regex to capture python code blocks, handling optional attributes
    # Matches ```{python} or ```{python, echo=False} etc.
    pattern = r"```\{python(?:[ ,].*?)?\}(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return "\n".join(matches)

def execute_and_capture(chapter_name, code):
    """Executes code and captures string/float variables."""
    # sandbox the execution
    local_vars = {}
    try:
        exec(code, {}, local_vars)
    except Exception as e:
        print(f"❌ Error executing {chapter_name}: {e}")
        # Print the first few lines of code to debug
        print(f"   Code snippet: {code[:200]}...")
        return {}

    # Capture only simple types (str, int, float) to avoid object serialization issues
    captured = {}
    for k, v in local_vars.items():
        if not k.startswith("_") and isinstance(v, (str, int, float)):
            captured[k] = v
    return captured

def main():
    qmd_files = sorted(glob.glob("book/quarto/contents/vol1/**/*.qmd", recursive=True))
    baseline = {}
    
    print(f"📸 Capturing baseline state for {len(qmd_files)} chapters...")
    
    for qmd_file in qmd_files:
        chapter_name = Path(qmd_file).stem
        # Skip utility files
        if chapter_name in ["404", "index", "intro", "references", "glossary"]:
            continue
            
        print(f"  - Processing {chapter_name}...", end="", flush=True)
        code = extract_python_cells(qmd_file)
        if not code:
            print(" (no code)")
            continue
            
        variables = execute_and_capture(chapter_name, code)
        baseline[chapter_name] = variables
        print(f" ✅ ({len(variables)} vars)")

    output_path = "book/tools/baseline_state.json"
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
    
    print(f"\n✨ Baseline captured to {output_path}")

if __name__ == "__main__":
    main()
