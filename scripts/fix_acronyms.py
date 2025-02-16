import os
import re
import argparse
from termcolor import colored

"""
This script processes Markdown (.md) and Quarto Markdown (.qmd) files to detect and replace redundant acronym definitions.

When an acronym (e.g., "Graphics Processing Units (GPUs)") is first introduced, it remains unchanged.
However, subsequent redefinitions of the same acronym are replaced with just the acronym (e.g., "GPU").

Features:
- Uses regex to detect acronyms inside parentheses (e.g., (GPU)).
- Walks **backward** from the acronym to extract the full definition in the same line.
- Ensures the number of words **matches the number of letters** in the acronym.
- Ignores plural forms (e.g., GPUs → GPU).
- Prompts the user for confirmation before storing a new acronym definition.
- Displays surrounding context when prompting for a definition confirmation.
- Ensures duplicate definitions are correctly recognized and replaced after the first occurrence.
- Replaces **all redundant occurrences** in a line after the first definition.
- Shows context around the replacement in interactive mode for manual approval.

Usage:
- To process all files in a directory: `python script.py -d /path/to/files`
- To process a single file interactively: `python script.py -i file.md`
"""

def find_acronym_definitions(text):
    """Finds acronym definitions by detecting acronyms inside parentheses and walking backwards to extract their definition."""
    definitions = {}
    lines = text.split('\n')
    
    for line in lines:
        matches = re.finditer(r'\((\b[A-Z]{2,}s?\b)\)', line)  # Find acronyms in parentheses
        for match in matches:
            acronym = match.group(1).rstrip('s')  # Remove trailing 's' if present
            words = re.findall(r'\b\w+\b', line)  # Extract words without punctuation
            acronym_pos = match.start()  # Find the start position of the acronym in the line
            
            # Convert words to positions
            word_positions = []
            pos = 0
            for w in words:
                word_positions.append((w, pos))
                pos += len(w) + 1  # Account for spaces
            
            # Find acronym position in the word list
            acronym_index = next((i for i, (_, p) in enumerate(word_positions) if p >= acronym_pos), None)
            if acronym_index is None:
                continue
            
            phrase = []
            j = acronym_index - 1
            acronym_letters = list(acronym)[::-1]  # Reverse acronym to match from last letter to first
            #print(acronym_letters)

            while j >= 0 and acronym_letters:
                word = word_positions[j][0]
                #print(word)
                if word[0].lower() == acronym_letters[0].lower():
                    #print(word)
                    phrase.insert(0, word)
                    acronym_letters.pop(0)  # Remove matched letter
                j -= 1
                if not acronym_letters:
                    break  # Stop once all letters are matched
            
            if not acronym_letters:  # Ensure all letters matched correctly
                definitions[' '.join(phrase)] = acronym
    return definitions

def replace_redundant_definitions(text, definitions, interactive):
    """Replaces redundant acronym definitions with just the acronym, ensuring all occurrences after the first are replaced, even across multiple lines."""
    lines = text.split('\n')
    seen_definitions = set()

    for i, line in enumerate(lines):
        for phrase, acronym in definitions.items():
            # If this is the first time we see the full definition, store it but do not replace
            if f"{phrase} ({acronym})" in line and acronym not in seen_definitions:
                seen_definitions.add(acronym)
                continue  # Skip replacing the first occurrence

            temp_line = line

            # Replace redundant occurrences of the phrase with the acronym in subsequent mentions
            if phrase in temp_line and acronym in seen_definitions:
                temp_line = line.replace(f"{phrase} ({acronym})", acronym)

            # Interactive confirmation before replacing
            if interactive and temp_line != line:
                print(colored("Original   :", "red"), line)
                print(colored("Replacement:", "green"), temp_line)
                choice = input("Replace? (y/n): ").strip().lower()
                if choice != 'y':
                    continue

            line = temp_line

        lines[i] = line

    return '\n'.join(lines)

def process_text(text, interactive):
    """Processes text to find and replace redundant acronym definitions."""
    definitions = find_acronym_definitions(text)
    print(definitions)
    return replace_redundant_definitions(text, definitions, interactive)

def process_file(file_path, interactive):
    """Reads a file, processes it, and writes back the updated content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    updated_text = process_text(text, interactive)

    if updated_text != text:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_text)
        print(f"Updated: {file_path}")
    else:
        print("No changes made.")

def process_directory(directory, interactive):
    """Processes all Markdown (.md, .qmd) files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md") or file.endswith(".qmd"):
                file_path = os.path.join(root, file)
                process_file(file_path, interactive)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and replace redundant acronym definitions in Markdown/QMD files.")
    parser.add_argument("-d", "--directory", help="Process all Markdown/QMD files in a directory")
    parser.add_argument("-i", "--interactive", help="Interactively process a single Markdown/QMD file")

    args = parser.parse_args()

    if args.directory:
        process_directory(args.directory, interactive=False)
    elif args.interactive:
        process_file(args.interactive, interactive=True)
