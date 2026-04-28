import json
import subprocess
import sys

def run_pilot():
    # Pilot files from each track
    pilot_files = [
        "interviews/vault/questions/cloud/cloud-3680.yaml",
        "interviews/vault/questions/edge/edge-2321.yaml",
        "interviews/vault/questions/mobile/mobile-1300.yaml",
        "interviews/vault/questions/tinyml/tinyml-0983.yaml",
        "interviews/vault/questions/global/global-0205.yaml"
    ]
    
    input_data = []
    for filepath in pilot_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            input_data.append({"path": filepath, "content": content})
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
            continue

    input_json_str = json.dumps(input_data)
    
    prompt = """You are a strict technical editor for the StaffML book. Your job is to standardize the `napkin_math` and `common_mistake` fields into our "Gold Standard" pedagogical format.

### NAPKIN MATH GOLD STANDARD:
Transform the `napkin_math` into a 3-part whiteboard derivation using a YAML literal block (`|`):
1. **Assumptions & Constraints:** (List the starting numbers, constants, or hardware facts using `-` bullets).
2. **Calculations:** (Show the step-by-step arithmetic using `-` bullets).
3. **Conclusion:** (Provide a sanity check and **bold the final result**).

### COMMON MISTAKE GOLD STANDARD:
Transform `common_mistake` into this pedagogical structure:
**The Pitfall:** [Short description]
**The Rationale:** [Why candidates fail]
**The Consequence:** [Impact on the system]

### STRICT STYLE RULES:
- **Always** use `1.`, `2.`, `3.` for the top-level stages of math.
- **Always** use `-` for bullet points inside those stages. **NEVER** use `*`.
- **Always** bold the stage headers (e.g., **1. Assumptions & Constraints:**).
- **Always** bold the final conclusion or result.
- **DO NOT** change any mathematical values or logic. ONLY fix the structure and style.

Return a JSON array containing ONLY the files you modified. Your output must be valid JSON matching this structure:
[
  {
    "path": "<file_path>",
    "corrected_content": "<the full corrected YAML string>"
  }
]
If a file is perfectly formatted, omit it.
IMPORTANT: Return ONLY raw JSON, with no markdown code blocks, backticks, or other text."""

    print(f"Running Markdown Linter Pilot on {len(input_data)} files...", flush=True)
    
    cmd = ['gemini', '-m', 'gemini-3.1-pro-preview', '-p', prompt]
    
    try:
        result = subprocess.run(
            cmd,
            input=input_json_str,
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        if output.startswith("```json"):
            output = output.replace("```json\n", "")
        if output.startswith("```"):
            output = output.replace("```\n", "")
        if output.endswith("```"):
            output = output[:-3].strip()
            
        if not output:
            print("No files needed formatting!")
            return
            
        corrections = json.loads(output, strict=False)
        for correction in corrections:
            path = correction.get("path")
            new_content = correction.get("corrected_content")
            if path and new_content:
                print(f"Applying markdown formatting to {path}", flush=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                    
        print("\nPilot complete. Please review the git diff.")
        
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'stderr'):
            print(e.stderr)

if __name__ == "__main__":
    run_pilot()
