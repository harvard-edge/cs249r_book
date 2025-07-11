#!/usr/bin/env python3
"""
extract_headers.py

Extracts section headers (e.g., #, ##, ###) from .qmd files.
Supports either a single file (-f) or all .qmd files in a directory (-d).
Outputs a neatly formatted table of:
    - Filename
    - Header Level
    - Header Text

Usage:
    python extract_headers.py -f path/to/file.qmd
    python extract_headers.py -d path/to/dir/
"""

import os
import re
import argparse
from pathlib import Path

def extract_headers_from_file(file_path):
    """
    Reads a .qmd file and extracts all markdown-style headers.

    Args:
        file_path (str or Path): Path to the .qmd file.

    Returns:
        list of tuples: Each tuple is (header level, header text).
    """
    headers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match lines starting with 1–6 '#' followed by space and text
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                level = match.group(1)    # e.g., '##'
                text = match.group(2).strip()
                headers.append((level, text))
    return headers

def process_files(files):
    """
    Processes a list of files and extracts headers from each.

    Args:
        files (list of Path): List of .qmd files to process.

    Returns:
        list of tuples: Each tuple is (relative path, header level, header text).
    """
    results = []
    for file_path in files:
        headers = extract_headers_from_file(file_path)
        rel_path = os.path.relpath(file_path)
        for level, text in headers:
            results.append((rel_path, level, text))
    return results

def find_qmd_files(directory):
    """
    Recursively finds all .qmd files under the given directory.

    Args:
        directory (str): Directory path.

    Returns:
        list of Path: All matching .qmd files.
    """
    return list(Path(directory).rglob("*.qmd"))

def main():
    """
    Entry point. Parses arguments and runs header extraction.
    """
    parser = argparse.ArgumentParser(description="Extract section headers from .qmd files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='Path to a single .qmd file')
    group.add_argument('-d', '--directory', help='Directory containing .qmd files recursively')
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = find_qmd_files(args.directory)

    headers = process_files(files)

    # Print formatted output table
    print(f"{'Filename':<30} | {'Level':<5} | Header")
    print(f"{'-'*30}-|{'-'*6}-|{'-'*40}")
    for filename, level, header in headers:
        print(f"{filename:<30} | {level:<5} | {header}")

if __name__ == "__main__":
    main()
