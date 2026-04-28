import json
import subprocess
import sys

def simulate():
    filepath = "interviews/vault/questions/cloud/cloud-3680.yaml"
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    input_data = [{"path": filepath, "content": content}]
    input_json_str = json.dumps(input_data)
    
    prompt = """You are an expert technical reviewer for the StaffML / Machine Learning Systems book.
Your task is to refine a multiple-choice question in YAML format to meet our new strict pedagogical and formatting standards.

We have a web UI that features a "Hardware Reference" sidebar and a "Napkin Calculator". 
Therefore, scenarios should be concise and realistic.

1. **Information Completeness vs Clutter:** 
   - DO NOT dump standard hardware constants (e.g., "The H100 has 80GB of VRAM and 3.35 TB/s bandwidth", "Orin delivers 275 TOPS") in the `scenario`. Just name the hardware (e.g., "NVIDIA H100", "NVIDIA Orin"). The candidate will use the sidebar for the specs.
   - DO include situational constraints (e.g., "batch size of 16", "128K context window").
   
2. **Separation of Concerns:**
   - `scenario`: Factual context only. Ends with a period. No questions.
   - `question`: A concise, one-sentence interrogative (max 200 chars).
   
3. **Markdown Formatting for `napkin_math`:**
   - Use a YAML literal block scalar (`|`).
   - Break math down into a readable, step-by-step markdown list.
   - Emphasize the final result in bold.

4. **Pedagogical Formatting for `common_mistake`:**
   - Use a YAML literal block scalar (`|`).
   - Structure exactly as follows:
     **The Pitfall:** [Short description]
     **The Rationale:** [Why candidates make this mistake]
     **The Consequence:** [The impact of the mistake]

5. **Options:**
   - Ensure there are exactly 4 `options` and a valid `correct_index` (0-3). If missing, generate them based on the `realistic_solution` and `common_mistake`.

Output ONLY the corrected YAML in a raw JSON array matching this structure:
[
  {
    "path": "<file_path>",
    "corrected_content": "<the full corrected YAML string>"
  }
]
IMPORTANT: Return ONLY raw JSON, with no markdown code blocks, backticks, or other text."""

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
            
        corrections = json.loads(output, strict=False)
        print("--- BEFORE ---")
        print(content)
        print("\n--- AFTER ---")
        print(corrections[0]['corrected_content'])
        
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'stderr'):
            print(e.stderr)

if __name__ == "__main__":
    simulate()
