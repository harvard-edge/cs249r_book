import os
import json
import subprocess
import sys

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def repair_batch_final(batch_files):
    input_data = []
    for filepath in batch_files:
        filepath = filepath.strip()
        if not filepath: continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            input_data.append({"path": filepath, "content": content})
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    if not input_data:
        return

    input_json_str = json.dumps(input_data)
    
    prompt = """You are a YAML Syntax and Schema Expert. These files have parsing or schema errors.

### REPAIR RULES:
1. **Fix "Mapping values not allowed here":** ALWAYS wrap strings containing colons in single quotes.
2. **Fix "expected <block end>, but found '<scalar>'":** Fix indentation and unquoted colons.
3. **Fix "invalid zone" or "invalid bloom_level":**
   - If `bloom_level: remember`, `zone` must be `recall`.
   - If `bloom_level: understand`, `zone` must be `fluency`.
4. **Fix "visual.path does not resolve":** Remove the `visual:` block.

Output ONLY the corrected YAML in a raw JSON array:
[{"path": "<path>", "corrected_content": "<YAML string>"}]"""

    cmd = ['gemini', '-m', 'gemini-3.1-pro-preview', '-p', prompt]
    
    try:
        result = subprocess.run(cmd, input=input_json_str, capture_output=True, text=True, check=True, timeout=600)
        output = result.stdout.strip()
        if output.startswith("```json"): output = output.replace("```json\n", "").replace("```", "")
        if output.startswith("```"): output = output.replace("```", "")
        corrections = json.loads(output, strict=False)
        for correction in corrections:
            path = correction.get("path")
            new_content = correction.get("corrected_content")
            if path and new_content:
                print(f"Fixed {path}")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
    except Exception as e:
        print(f"Failed: {e}")

def main():
    with open('files_to_repair_final_v2.txt', 'r') as f:
        all_files = [line.strip() for line in f.readlines() if line.strip()]
    
    repair_batch_final(all_files)

if __name__ == "__main__":
    main()
