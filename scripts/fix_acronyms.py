import re
import os
import argparse
from termcolor import colored

"""
This script processes Markdown (.md) and Quarto Markdown (.qmd) files to detect and replace redundant acronym definitions.

When an acronym (e.g., "Tensor Processing Unit (TPU)") is first introduced, it remains unchanged.
However, subsequent redefinitions of the same acronym are replaced with just the acronym (e.g., "TPU").

Features:
- Processes entire directories of Markdown/QMD files or a single file interactively.
- Preserves correct pluralization (e.g., "ASICs" remains "ASICs").
- Maintains grammatical correctness by preserving preceding articles like "The" or "An".
- Shows context around the replacement in interactive mode for manual approval.

Usage:
- To process all files in a directory: `python script.py -d /path/to/files`
- To process a single file interactively: `python script.py -i file.md`
"""

def find_and_replace_redefinitions(text, seen_acronyms, interactive=False):
    """
    Identifies acronym redefinitions in the text and replaces redundant ones with just the acronym.
    If interactive mode is enabled, prompts the user before replacing each instance.
    Displays a few words before and after the replacement for context.
    Handles pluralization intelligently (e.g., ASIC vs. ASICs) and ensures grammatical correctness.
    """
    pattern = re.compile(r'\b(?:\b(The|An|A)\s+)?([A-Z][a-z]+(?:[-\s][A-Z]?[a-z]+)*)\s*\((\b[A-Z0-9]+s?\b)\)')

    def replacement(match):
        article, full_term, acronym = match.groups()
        if acronym in seen_acronyms:
            if interactive:
                # Find surrounding context
                start_idx = max(0, match.start() - 30)
                end_idx = min(len(text), match.end() + 30)
                context_before = text[start_idx:match.start()].strip().split()[-3:]  # Get last 3 words before
                context_after = text[match.end():end_idx].strip().split()[:3]  # Get first 3 words after
                
                original_sentence = f"... {' '.join(context_before)} {match.group(0).strip()} {' '.join(context_after)} ..."
                modified_sentence = f"... {' '.join(context_before)} {(article + ' ' if article else '') + acronym} {' '.join(context_after)} ..."
                
                print(colored("Original:", "red"), original_sentence)
                print(colored("Replacement:", "green"), modified_sentence)
                
                choice = input("Replace? (y/n): ").strip().lower()
                if choice != 'y':
                    return match.group(0)
            return (article + ' ' if article else '') + acronym  # Keep the article if it exists

        seen_acronyms.add(acronym)
        return match.group(0)  # Keep the first definition as is

    return pattern.sub(replacement, text)

def process_markdown_files(directory, interactive=False):
    """
    Processes all Markdown (.md) and Quarto Markdown (.qmd) files in the given directory,
    replacing redundant acronym definitions.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") or file.endswith(".qmd"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                seen_acronyms = set()
                updated_content = find_and_replace_redefinitions(content, seen_acronyms, interactive)
                
                if updated_content != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(updated_content)
                    print(f"Updated: {file_path}")
                else:
                    print(f"No changes: {file_path}")

def process_single_file(file_path):
    """
    Processes a single Markdown or QMD file, applying acronym replacement interactively.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    seen_acronyms = set()
    updated_content = find_and_replace_redefinitions(content, seen_acronyms, interactive=True)
    
    if updated_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated: {file_path}")
    else:
        print("No changes needed to be made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove redundant acronym definitions in Markdown and QMD files.")
    parser.add_argument("-d", "--directory", help="Process all Markdown/QMD files in a directory")
    parser.add_argument("-i", "--interactive", help="Interactively process a single Markdown/QMD file")
    args = parser.parse_args()

    if args.directory:
        process_markdown_files(args.directory, interactive=False)
    elif args.interactive:
        process_single_file(args.interactive)
