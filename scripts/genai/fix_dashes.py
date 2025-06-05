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
    "You will receive a paragraph of formal academic writing that contains exactly one sentence with an em-dash clause (—like this—). "
    "Your task is to rewrite only the em-dash clause using a more formal structure, such as commas or parentheses. "
    "Do not change the rest of the sentence. Use the full paragraph for context, but do not modify anything else.\n\n"
    "Return a JSON object with three fields:\n"
    "- 'original_clause': the clause inside the em dashes\n"
    "- 'revised_clause': the rewritten version using commas or parentheses\n"
    "- 'revised_paragraph': the full paragraph with only the clause replaced\n\n"
    "Examples:\n"
    "Original: These tools—including JAX, PyTorch, and TensorFlow—enable large-scale model training.\n"
    "Return:\n"
    "{{\n"
    "  \"original_clause\": \"including JAX, PyTorch, and TensorFlow\",\n"
    "  \"revised_clause\": \"including JAX, PyTorch, and TensorFlow,\",\n"
    "  \"revised_paragraph\": \"These tools, including JAX, PyTorch, and TensorFlow, enable large-scale model training.\"\n"
    "}}\n\n"
    "Original: The model’s performance—despite limited training data—was surprisingly strong.\n"
    "Return:\n"
    "{{\n"
    "  \"original_clause\": \"despite limited training data\",\n"
    "  \"revised_clause\": \"despite limited training data,\",\n"
    "  \"revised_paragraph\": \"The model’s performance, despite limited training data, was surprisingly strong.\"\n"
    "}}\n\n"
    "Now process the following paragraph:\n"
    "\"{}\""
)

BATCH_PROMPT_TEMPLATE = (
    "Below is a list of paragraphs. Each paragraph contains exactly one sentence with a clause surrounded by em dashes (—like this—). "
    "Rewrite only the em-dash clause using a more formal structure. Prefer commas where the clause is essential to the sentence, and use parentheses only for clearly optional or parenthetical information. Avoid overusing parentheses. "
    "Do not rewrite the entire sentence or paragraph. Only replace the clause. Use the paragraph for context.\n\n"
    "For each paragraph, return a JSON object with:\n"
    "- 'original_clause': the clause inside the em dashes\n"
    "- 'revised_clause': the rewritten clause using formal punctuation\n"
    "- 'revised_paragraph': the paragraph with only the clause replaced\n\n"
    "Return the results as a JSON array.\n\n"
    "Paragraphs:\n"
    "{}"
)


# === Rewrite Functions ===

def rewrite_paragraph_json(paragraph: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(paragraph)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return only valid JSON. No markdown, backticks, or extra explanation."},
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
        logging.warning("⚠️ JSON parsing failed. Using original paragraph.")
        return {
            "original_clause": None,
            "revised_clause": None,
            "revised_paragraph": paragraph
        }

def rewrite_paragraphs_batch(paragraphs: List[str]) -> List[dict]:
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(paragraphs))
    prompt = BATCH_PROMPT_TEMPLATE.format(numbered)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return only a JSON array. No markdown or explanation."},
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
        logging.warning("⚠️ Batch JSON parsing failed.")
        return [{"original_clause": None, "revised_clause": None, "revised_paragraph": p} for p in paragraphs]

# === Highlight Helpers ===

RED = "\033[31m"
GREEN = "\033[32m"
END = "\033[0m"

def highlight_original(para, clause):
    return para.replace(clause or "", f"{RED}{clause}{END}", 1) if clause else para

def highlight_revised(para, clause):
    return para.replace(clause or "", f"{GREEN}{clause}{END}", 1) if clause else para

# === Main File Processing ===

def process_file(file_path, mode="interactive", batch_size=5):
    logging.info(f"🔍 Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = PARAGRAPH_SPLIT_PATTERN.split(content)
    updated_paragraphs = list(paragraphs)
    changed = False

    if mode == "batch":
        logging.info(f"📦 Batch mode (batch size = {batch_size})")
        candidates = [(i, p) for i, p in enumerate(paragraphs) if DASH_PATTERN.search(p)]
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_paragraphs = [p for _, p in batch]
            logging.info(f"🤖 Sending batch of {len(batch_paragraphs)} paragraphs to OpenAI")
            results = rewrite_paragraphs_batch(batch_paragraphs)
            for idx, result in zip(batch_indices, results):
                updated_paragraphs[idx] = result["revised_paragraph"]
                changed = True
    else:
        logging.info("📝 Interactive mode")
        for i, para in enumerate(paragraphs):
            if not DASH_PATTERN.search(para):
                continue
            result = rewrite_paragraph_json(para)
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
    parser = argparse.ArgumentParser(description="Rewrite em-dash clauses in .qmd files using OpenAI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=str, help="Path to a specific .qmd file.")
    group.add_argument("-d", "--directory", type=str, help="Path to a directory containing .qmd files.")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="batch", help="Rewrite mode.")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for batch mode.")
    args = parser.parse_args()

    if args.file:
        process_file(Path(args.file), args.mode, batch_size=args.batch_size)
    else:
        process_directory(Path(args.directory), args.mode, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
