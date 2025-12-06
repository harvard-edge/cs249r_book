import re
import argparse
import os
from collections import defaultdict

# Regular expression to match keywords in parentheses (e.g., (CNN))
pattern = re.compile(r"\(([A-Z]{2,}s?)\)")

def process_file(file_path, keyword_lines):
    """Processes a single file and updates the keyword dictionary with line numbers."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                matches = pattern.findall(line)
                for match in matches:
                    keyword_lines[match].append(line_num)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def process_directory(directory):
    """Recursively finds all `.qmd` files in a directory and processes them."""
    keyword_lines = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".qmd"):
                file_path = os.path.join(root, file)
                process_file(file_path, keyword_lines)
    return keyword_lines

def print_results(keyword_lines):
    """Prints the extracted keywords and their corresponding line numbers, sorted by most occurrences."""
    sorted_keywords = sorted(keyword_lines.items(), key=lambda x: -len(x[1]))  # Sort by most occurrences

    for keyword, lines in sorted_keywords:
        print(f"{keyword}: {', '.join(map(str, lines))}")

def main():
    parser = argparse.ArgumentParser(description="Extract uppercase keywords in parentheses and their line numbers from .qmd files.")
    parser.add_argument("-f", "--file", help="Path to a single .qmd file")
    parser.add_argument("-d", "--directory", help="Path to a directory containing .qmd files (processed recursively)")

    args = parser.parse_args()

    if args.file and args.directory:
        print("Please provide only one of -f (file) or -d (directory), not both.")
        return

    keyword_lines = defaultdict(list)

    if args.file:
        process_file(args.file, keyword_lines)
        print_results(keyword_lines)

    elif args.directory:
        keyword_lines = process_directory(args.directory)
        print_results(keyword_lines)

    else:
        print("Please specify a file with -f or a directory with -d.")

if __name__ == "__main__":
    main()
