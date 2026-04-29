import os
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def repair_batch_final(batch_files, batch_idx, total_batches):
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
    
    prompt = """You are a YAML Syntax and Schema Expert. These files have parsing or schema errors.

### REPAIR RULES:
1. **Fix "Mapping values not allowed here":** ALWAYS wrap strings containing colons (e.g. `title: Result: Compute-Bound`) in single quotes: `title: 'Result: Compute-Bound'`.
2. **Fix "expected <block end>, but found '<scalar>'":** This usually happens inside a block scalar (`|`) where indentation is incorrect or unquoted colons are used in a mapping field. Ensure fields like `realistic_solution` use a literal block `|` if they have colons.
3. **Fix "invalid zone" or "invalid bloom_level":**
   - Zones: [analyze, design, diagnosis, evaluation, fluency, implement, mastery, optimization, realization, recall, specification]
   - Bloom: [remember, understand, apply, analyze, evaluate, create]
   - If `bloom_level: remember`, `zone` must be `recall`.
   - If `bloom_level: understand`, `zone` must be `fluency`.
4. **Fix "visual.path does not resolve":** If the visual SVG file is missing, remove the `visual:` block from the YAML.

Output ONLY the corrected YAML in a raw JSON array:
[{"path": "<path>", "corrected_content": "<YAML string>"}]
Return `[]` if no changes needed."""

    print(f"[Batch {batch_idx}/{total_batches}] Final Repair: {len(batch_files)} files...", flush=True)
    
    cmd = ['gemini', '-m', 'gemini-3.1-pro-preview', '-p', prompt]
    
    try:
        result = subprocess.run(cmd, input=input_json_str, capture_output=True, text=True, check=True, timeout=1200)
    except Exception as e:
        print(f"[Batch {batch_idx}] Failed: {e}", file=sys.stderr, flush=True)
        return

    output = result.stdout.strip()
    if output.startswith("```json"): output = output.replace("```json\n", "").replace("```", "")
    if output.startswith("```"): output = output.replace("```", "")

    if not output: return

    try:
        corrections = json.loads(output, strict=False)
        for correction in corrections:
            path = correction.get("path")
            new_content = correction.get("corrected_content")
            if path and new_content:
                print(f"[Batch {batch_idx}] Fixed {path}", flush=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
    except Exception as e:
        print(f"[Batch {batch_idx}] JSON Error: {e}", file=sys.stderr, flush=True)

def main():
    with open('files_to_repair_final.txt', 'r') as f:
        all_files = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Repairing {len(all_files)} files.", flush=True)
    
    BATCH_SIZE = 20 
    batches = list(chunk_list(all_files, BATCH_SIZE))
    total_batches = len(batches)
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i, batch in enumerate(batches):
            futures.append(executor.submit(repair_batch_final, batch, i + 1, total_batches))
            time.sleep(1.5)
        for future in as_completed(futures): pass

    print("\nFinal Repair complete.", flush=True)

if __name__ == "__main__":
    main()
