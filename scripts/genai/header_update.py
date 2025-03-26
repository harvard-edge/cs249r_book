import os
import re
import yaml
import argparse
import openai
from pathlib import Path
from collections import defaultdict

# -- Set your OpenAI model here
OPENAI_MODEL = "gpt-4"

# -- Prompt Template
PROMPT_TEMPLATE = """
You are assisting with editing section headers for a textbook on Machine Learning Systems. The headers are extracted from `.qmd` Markdown files. Your task is to revise the headers to be suitable for a professional, technically rigorous textbook.

Please follow these guidelines:
- Concise: Keep headers short (ideally under 5 words), clear, and impactful.
- Hierarchical Awareness: Analyze all headers before editing. Ensure that subheaders (e.g., ###) are meaningfully distinct from their parent headers and do not repeat information unnecessarily.
- Consistent Tone: Use an academic, systems-oriented style. Assume the reader is technically literate but learning the concepts for the first time.
- No Numbering: Do not include chapter or section numbers (e.g., “3.1”).
- No Markdown Changes: Only update the text of the headers, not the Markdown level (#, ##, etc.).

Return your output in the following YAML format:
```yaml
- original: "## Introduction to Compilation Techniques for Machine Learning"
  revised: "## Compilation Techniques"
- original: "### Explaining Why Compilers Matter in ML Pipelines"
  revised: "### Why Compilers Matter"
```

Make sure each header revision respects the hierarchy and flow of the textbook. Do not skip any headers, even if they seem fine — evaluate all.
Here is the full list of headers:
"""

def find_qmd_headers(directory):
    header_map = []
    file_line_map = defaultdict(list)

    for path in Path(directory).rglob("*.qmd"):
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if re.match(r'^\s*#{1,6} ', line):
                clean_line = line.strip()
                header_map.append(clean_line)
                file_line_map[clean_line].append((path, i, line))
    return header_map, file_line_map

def call_openai(prompt):
    print("[DEBUG] Prompt sent to OpenAI:\n", prompt[:1000], "...\n")
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response['choices'][0]['message']['content']

def apply_revisions(file_line_map, replacements):
    for item in replacements:
        original = item['original'].strip()
        revised = item['revised'].strip()
        if original == revised:
            continue
        if original in file_line_map:
            for path, idx, line in file_line_map[original]:
                print(f"[DEBUG] Updating in {path}:\n  From: {original}\n  To:   {revised}")
                with path.open(encoding="utf-8") as f:
                    content = f.readlines()
                content[idx] = revised + "\n"
                with path.open('w', encoding="utf-8") as f:
                    f.writelines(content)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set. Please export it before running.")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="Directory to scan for .qmd files")
    args = parser.parse_args()

    print(f"[INFO] Scanning headers in {args.dir}")
    headers, file_line_map = find_qmd_headers(args.dir)
    if not headers:
        print("[WARN] No headers found.")
        return

    header_list = "\n".join(headers)
    full_prompt = PROMPT_TEMPLATE.strip() + "\n" + header_list

    print("[INFO] Sending headers to OpenAI...")
    raw_output = call_openai(full_prompt)

    print("[DEBUG] Raw response:\n", raw_output[:1000], "...\n")

    print("[INFO] Parsing YAML response...")
    try:
        revisions = yaml.safe_load(raw_output)
    except yaml.YAMLError as e:
        print("[ERROR] Failed to parse YAML:", e)
        print(raw_output)
        return

    print(f"[INFO] Applying {len(revisions)} revisions...")
    apply_revisions(file_line_map, revisions)
    print("[DONE] All headers updated.")

if __name__ == "__main__":
    main()
