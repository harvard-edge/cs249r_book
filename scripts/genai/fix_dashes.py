import argparse
import os
import re
import json
import logging
from pathlib import Path
from typing import List
from openai import OpenAI

# === Setup ===
client = OpenAI()
logging.basicConfig(level=logging.INFO)

# === Constants ===
DASH_PATTERN = re.compile(r'—([^—]+?)—')
PARAGRAPH_SPLIT_PATTERN = re.compile(r'\n\s*\n')

# === Prompt Templates ===
PROMPT_TEMPLATE = (
    "You will receive a paragraph of formal academic writing that contains one or more em-dash clauses (—like this—). "
    "Your task is to rewrite all em-dash clauses using a more formal structure, such as commas or parentheses. "
    "Do not modify any other part of the paragraph. Preserve Markdown formatting including citations (e.g., [@ref]), bold, italics, and inline code.\n\n"
    "Return a JSON object with:\n"
    "- 'original_clause': the clause inside the em dashes\n"
    "- 'revised_clause': the rewritten clause\n"
    "- 'revised_paragraph': the full paragraph with the clause replaced\n\n"
    "Examples:\n"
    "Original: These tools—including JAX, PyTorch, and TensorFlow—enable large-scale model training.\n"
    "Return:\n"
    "{{\n"
    "  \"original_clause\": \"including JAX, PyTorch, and TensorFlow\",\n"
    "  \"revised_clause\": \"including JAX, PyTorch, and TensorFlow,\",\n"
    "  \"revised_paragraph\": \"These tools, including JAX, PyTorch, and TensorFlow, enable large-scale model training.\"\n"
    "}}\n\n"
    "Now process this paragraph:\n"
    "\"{}\""
)

# === Rewrite Logic ===

def rewrite_paragraph_json(paragraph: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(paragraph)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return only valid JSON. Do not include backticks, markdown, or commentary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
        elif raw.startswith("```"):
            raw = raw[len("```"):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        return json.loads(raw)
    except Exception:
        logging.warning("⚠️ JSON parsing failed. Returning original paragraph.")
        return {
            "original_clause": None,
            "revised_clause": None,
            "revised_paragraph": paragraph
        }

def run_fix_pipeline(paragraph: str) -> dict:
    """Apply one or more rewrite functions. Currently just em-dash fix."""
    return rewrite_paragraph_json(paragraph)

# === Output Helpers ===

RED = "\033[31m"
GREEN = "\033[32m"
END = "\033[0m"

def highlight_original(para, clause):
    return para.replace(clause or "", f"{RED}{clause}{END}", 1) if clause else para

def highlight_revised(para, clause):
    return para.replace(clause or "", f"{GREEN}{clause}{END}", 1) if clause else para

# === Main Processing ===

def process_file(file_path, mode="interactive", batch_size=5):
    logging.info(f"🔍 Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = PARAGRAPH_SPLIT_PATTERN.split(content)
    updated_paragraphs = list(paragraphs)
    changed = False

    if mode == "batch":
        logging.info(f"📦 Batch mode (batch size = {batch_size})")
        for i, para in enumerate(paragraphs):
            if not DASH_PATTERN.search(para):
                continue
            result = run_fix_pipeline(para)
            updated_paragraphs[i] = result["revised_paragraph"]
            changed = True
    else:
        logging.info("📝 Interactive mode")
        for i, para in enumerate(paragraphs):
            if not DASH_PATTERN.search(para):
                continue
            result = run_fix_pipeline(para)
            revised = result["revised_paragraph"]
            print("\n--- Original ---")
            print(highlight_original(para, result["original_clause"]))
            print("\n--- Proposed ---")
            print(highlight_revised(revised, result["revised_clause"]))
            choice = input("Accept change? [y/N]: ").strip().lower()
            updated_paragraphs[i] = revised if choice == "y" else para
            if choice == "y":
                changed = True

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(updated_paragraphs))
        logging.info(f"✅ Updated file: {file_path}")
    else:
        logging.info(f"⚪ No changes needed: {file_path}")

def process_directory(directory, mode, batch_size):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".qmd"):
                process_file(Path(root) / file, mode, batch_size=batch_size)

# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="Rewrite em-dash clauses using OpenAI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a .qmd file.")
    group.add_argument("-d", "--directory", type=str, help="Path to a directory of .qmd files.")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="batch", help="Rewrite mode.")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size (currently unused).")
    args = parser.parse_args()

    if args.file:
        process_file(Path(args.file), args.mode, batch_size=args.batch_size)
    else:
        process_directory(Path(args.directory), args.mode, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
