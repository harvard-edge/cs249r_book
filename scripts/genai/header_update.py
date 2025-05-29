import os
import re
import yaml
import argparse
from openai import OpenAI

client = OpenAI()
from pathlib import Path
from collections import defaultdict

# -- Set your OpenAI model here
OPENAI_MODEL = "gpt-4o"

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

- original: "## Introduction to Compilation Techniques for Machine Learning"
  revised: "## Compilation Techniques"
- original: "### Explaining Why Compilers Matter in ML Pipelines"
  revised: "### Why Compilers Matter"

Make sure each header revision respects the hierarchy and flow of the textbook. Do not skip any headers, even if they seem fine — evaluate all.
Here is the full list of headers:
"""

def find_qmd_headers_in_file(file_path):
    header_map = []
    file_line_map = defaultdict(list)

    with file_path.open(encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if re.match(r'^\s*#{1,6} ', line):
            clean_line = line.strip()
            header_map.append(clean_line)
            file_line_map[clean_line].append((file_path, i, line))
    return header_map, file_line_map

def find_qmd_headers_in_directory(directory):
    header_map = []
    file_line_map = defaultdict(list)

    for path in Path(directory).rglob("*.qmd"):
        file_headers, file_map = find_qmd_headers_in_file(path)
        header_map.extend(file_headers)
        for key, value in file_map.items():
            file_line_map[key].extend(value)

    return header_map, file_line_map

def call_openai(prompt):
    print("[DEBUG] Prompt sent to OpenAI:\n", prompt[:1000], "...\n")
    response = client.chat.completions.create(model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3)
    return response.choices[0].message.content

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

def strip_yaml_fence(text):
    """Remove surrounding ```yaml ... ``` fence if present."""
    lines = text.strip().splitlines()
    if lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set. Please export it before running.")
        exit(1)

    parser = argparse.ArgumentParser(description="Revise section headers in .qmd files using OpenAI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dir", help="Directory to scan for .qmd files")
    group.add_argument("-f", "--file", help="Specific .qmd file to process")
    args = parser.parse_args()

    if args.dir:
        print(f"[INFO] Scanning headers in directory {args.dir}")
        headers, file_line_map = find_qmd_headers_in_directory(args.dir)
    else:
        file_path = Path(args.file)
        if not file_path.exists() or not file_path.suffix == '.qmd':
            print(f"[ERROR] File {args.file} does not exist or is not a .qmd file.")
            exit(1)
        print(f"[INFO] Scanning headers in file {args.file}")
        headers, file_line_map = find_qmd_headers_in_file(file_path)

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
        cleaned_output = strip_yaml_fence(raw_output)
        revisions = yaml.safe_load(cleaned_output)
    except yaml.YAMLError as e:
        print("[ERROR] Failed to parse YAML:", e)
        print(raw_output)
        return

    print(f"[INFO] Applying {len(revisions)} revisions...")
    apply_revisions(file_line_map, revisions)
    print("[DONE] All headers updated.")

if __name__ == "__main__":
    main()
