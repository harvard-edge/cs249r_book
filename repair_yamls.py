import os
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def repair_batch(batch_files, batch_idx, total_batches):
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
    
    prompt = """You are a YAML Syntax and Schema Repair expert. 
These YAML files have failed to load due to syntax errors or schema violations introduced in a previous automated pass.

### REPAIR RULES:
1. **Fix "Mapping values not allowed here":** This usually means an unquoted string contains a colon followed by a space (e.g., `title: Result: Compute-bound`). **ALWAYS** wrap strings containing colons in single quotes (e.g., `title: 'Result: Compute-bound'`).
2. **Fix "Input should be a valid string" in `options`:** The `options` field MUST be a simple list of strings. If you see a list of dictionaries (e.g., `- Choice: Explanation`), convert it into a single string (e.g., `- "Choice: Explanation"`).
3. **Maintain Literal Blocks:** Ensure `napkin_math` and `common_mistake` continue to use literal blocks (`|`) and follow the 3-stage expert derivation and pitfall structure.
4. **No Hallucinations:** Do not change the technical meaning or logic. ONLY fix the syntax so the YAML loads correctly.

Output ONLY the corrected YAML in a raw JSON array:
[{"path": "<path>", "corrected_content": "<YAML string>"}]
Return exactly `[]` if no changes are needed. No markdown blocks."""

    print(f"[Batch {batch_idx}/{total_batches}] Repairing {len(batch_files)} files...", flush=True)
    
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
                print(f"[Batch {batch_idx}] Repaired {path}", flush=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
    except Exception as e:
        print(f"[Batch {batch_idx}] JSON Error: {e}", file=sys.stderr, flush=True)

def main():
    with open('files_to_repair.txt', 'r') as f:
        all_files = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Found {len(all_files)} YAML files for Repair Pass.", flush=True)
    
    BATCH_SIZE = 30 # Smaller batches for repair
    batches = list(chunk_list(all_files, BATCH_SIZE))
    total_batches = len(batches)
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for i, batch in enumerate(batches):
            futures.append(executor.submit(repair_batch, batch, i + 1, total_batches))
            time.sleep(1.5) # Staggered launch
        for future in as_completed(futures): pass

    print("\nRepair Pass complete.", flush=True)

if __name__ == "__main__":
    main()
