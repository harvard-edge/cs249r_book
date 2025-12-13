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
# Updated pattern to catch both em-dashes and triple hyphens
DASH_PATTERN = re.compile(r'(?:—([^—]+?)—|(\w)---(\w))')
PARAGRAPH_SPLIT_PATTERN = re.compile(r'\n\s*\n')

# === Prompt Template ===
PROMPT_TEMPLATE = (
    "You will receive a paragraph or a Markdown bullet point that contains one or more em-dash clauses (—like this—) "
    "or triple-hyphen constructions (word---word). "
    "Your task is to rewrite all em-dash clauses and triple-hyphen constructions using a more formal structure, "
    "such as commas, parentheses, or proper punctuation. "
    "Use the full paragraph or bullet as context. Do not change any other part of the paragraph. "
    "Preserve all Markdown formatting, including:\n"
    "- List bullets (e.g., '- This point')\n"
    "- Bold or italic text\n"
    "- Inline code\n"
    "- Citations like [@ref]\n"
    "Do not flatten bullets into a single paragraph. Maintain newlines and structure.\n\n"
    "Return a JSON object with three fields:\n"
    "- 'original_clause': the clause inside the em dashes or the triple-hyphen construction\n"
    "- 'revised_clause': the rewritten clause\n"
    "- 'revised_paragraph': the full paragraph with only the clause(s) changed\n\n"
    "Examples:\n"
    "Original: These tools—including JAX, PyTorch, and TensorFlow—enable large-scale model training.\n"
    "Return:\n"
    "{{\n"
    "  \"original_clause\": \"including JAX, PyTorch, and TensorFlow\",\n"
    "  \"revised_clause\": \"including JAX, PyTorch, and TensorFlow,\",\n"
    "  \"revised_paragraph\": \"These tools, including JAX, PyTorch, and TensorFlow, enable large-scale model training.\"\n"
    "}}\n\n"
    "Original: The performance---especially in high-demand scenarios---was impressive.\n"
    "Return:\n"
    "{{\n"
    "  \"original_clause\": \"performance---especially in high-demand scenarios---was\",\n"
    "  \"revised_clause\": \"performance, especially in high-demand scenarios, was\",\n"
    "  \"revised_paragraph\": \"The performance, especially in high-demand scenarios, was impressive.\"\n"
    "}}\n\n"
    "Original: - The energy cost of training vs. inference—analyzing how different phases of AI impact sustainability.\n"
    "Return:\n"
    "{{\n"
    "  \"original_clause\": \"analyzing how different phases of AI impact sustainability\",\n"
    "  \"revised_clause\": \"analyzing how different phases of AI impact sustainability,\",\n"
    "  \"revised_paragraph\": \"- The energy cost of training vs. inference, analyzing how different phases of AI impact sustainability.\"\n"
    "}}\n\n"
    "Now process this paragraph:\n"
    "\"{}\""
)

# === Rewrite Logic ===

def has_dash_pattern(text: str) -> bool:
    """Check if text contains em-dash clauses or triple-hyphen patterns."""
    # Check for em-dash pattern
    if re.search(r'—([^—]+?)—', text):
        return True
    # Check for triple-hyphen pattern between word characters
    if re.search(r'\w---\w', text):
        return True
    return False

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
    return rewrite_paragraph_json(paragraph)

# === Highlight Helpers ===

RED = "\033[31m"
GREEN = "\033[32m"
END = "\033[0m"

def highlight_original(para, clause):
    return para.replace(clause or "", f"{RED}{clause}{END}", 1) if clause else para

def highlight_revised(para, clause):
    return para.replace(clause or "", f"{GREEN}{clause}{END}", 1) if clause else para

def split_paragraphs_respecting_bullets(text: str) -> List[str]:
    """Split text into paragraphs and bullet lines for LLM-friendly processing."""
    paragraphs = PARAGRAPH_SPLIT_PATTERN.split(text)
    result = []
    for para in paragraphs:
        lines = para.strip().splitlines()
        if all(line.strip().startswith("- ") for line in lines if line.strip()):
            result.extend([line.strip() for line in lines if line.strip()])
        else:
            result.append(para.strip())
    return result

# === Main Processing ===

def process_file(file_path, mode="interactive", batch_size=5):
    logging.info(f"🔍 Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = split_paragraphs_respecting_bullets(content)
    updated_paragraphs = list(paragraphs)
    changed = False

    for i, para in enumerate(paragraphs):
        # Use the updated function to check for both patterns
        if not has_dash_pattern(para):
            continue
        result = run_fix_pipeline(para)
        revised = result["revised_paragraph"]
        if revised == para:
            continue

        if mode == "interactive":
            print("\n--- Original ---")
            print(highlight_original(para, result["original_clause"]))
            print("\n--- Proposed ---")
            print(highlight_revised(revised, result["revised_clause"]))
            choice = input("Accept change? [y/N]: ").strip().lower()
            updated_paragraphs[i] = revised if choice == "y" else para
            if choice == "y":
                changed = True
        else:
            updated_paragraphs[i] = revised
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
    parser = argparse.ArgumentParser(description="Rewrite em-dash clauses and triple-hyphen constructions in Markdown or academic paragraphs using OpenAI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a specific .qmd file.")
    group.add_argument("-d", "--directory", type=str, help="Path to directory of .qmd files.")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="batch", help="Rewrite mode.")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size (for future use).")
    args = parser.parse_args()

    if args.file:
        process_file(Path(args.file), args.mode, batch_size=args.batch_size)
    else:
        process_directory(Path(args.directory), args.mode, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
