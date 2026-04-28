import os
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def repair_batch_v2(batch_files, batch_idx, total_batches):
    input_data = []
    for filepath in batch_files:
        filepath = filepath.strip()
        if not filepath: continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            input_data.append({"path": filepath, "content": content})
        except Exception as e:
            print(f"[Batch {batch_idx}] Error reading {filepath}: {e}", file=sys.stderr, flush=True)
            
    if not input_data:
        return

    input_json_str = json.dumps(input_data)
    
    prompt = """You are a YAML Syntax and Schema Specialist. These files have persistent schema/syntax errors.

### 1. SYNTAX REPAIR:
- **Colons in Unquoted Strings:** If a field like `title:`, `scenario:`, or `question:` contains a colon followed by a space, YOU MUST wrap the entire string in single quotes. 
  Example: `title: 'Result: Compute-Bound'` (NOT `title: Result: Compute-Bound`).
- **Escape Characters:** Fix any invalid escape characters in double-quoted strings (e.g., `\g` is invalid). Convert to single-quoted strings to avoid escape issues.
- **Literal Blocks:** Ensure `napkin_math` and `common_mistake` use the `|` literal block for multi-line markdown.

### 2. SCHEMA & ENUM REPAIR:
- **Valid Zones:** [analyze, design, diagnosis, evaluation, fluency, implement, mastery, optimization, realization, recall, specification]
- **Valid Bloom Levels:** [remember, understand, apply, analyze, evaluate, create]
- **Valid Human Review Status:** [not-reviewed, verified, flagged, needs-rework]
- **Zone/Bloom Consistency:** If you find a mismatch, adjust the `zone` to match the `bloom_level` using this mapping:
  - remember -> recall
  - understand -> fluency
  - apply -> implement
  - analyze -> analyze
  - evaluate -> evaluation
  - create -> design

### 3. MCQ OPTION REPAIR:
- `options` MUST be a list of simple strings. Example:
  options:
  - "Option A"
  - "Option B"
  (NOT nested dictionaries like `- Label: Desc`).

Output ONLY the corrected YAML in a raw JSON array:
[{"path": "<path>", "corrected_content": "<YAML string>"}]
Return exactly `[]` if no changes are needed."""

    print(f"[Batch {batch_idx}/{total_batches}] V2 Repairing {len(batch_files)} files...", flush=True)
    
    cmd = ['gemini', '-m', 'gemini-3.1-pro-preview', '-p', prompt]
    
    try:
        result = subprocess.run(
            cmd,
            input=input_json_str,
            capture_output=True,
            text=True,
            check=True,
            timeout=1200
        )
    except Exception as e:
        print(f"[Batch {batch_idx}] Failed: {e}", file=sys.stderr, flush=True)
        return

    output = result.stdout.strip()
    if output.startswith("```json"): output = output.replace("```json\n", "")
    if output.startswith("```"): output = output.replace("```\n", "")
    if output.endswith("```"): output = output[:-3].strip()

    if not output: return

    try:
        corrections = json.loads(output, strict=False)
        for correction in corrections:
            path = correction.get("path")
            new_content = correction.get("corrected_content")
            if path and new_content:
                print(f"[Batch {batch_idx}] V2 Repaired {path}", flush=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
    except Exception as e:
        print(f"[Batch {batch_idx}] JSON Error: {e}", file=sys.stderr, flush=True)

def main():
    with open('files_to_repair_v2.txt', 'r') as f:
        all_files = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Found {len(all_files)} YAML files for V2 Repair Pass.", flush=True)
    
    BATCH_SIZE = 15 # Even smaller batches for complex schema repair
    batches = list(chunk_list(all_files, BATCH_SIZE))
    total_batches = len(batches)
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i, batch in enumerate(batches):
            futures.append(executor.submit(repair_batch_v2, batch, i + 1, total_batches))
            time.sleep(2) # Slower staggered launch
        for future in as_completed(futures): pass

    print("\nRepair Pass V2 complete.", flush=True)

if __name__ == "__main__":
    main()
