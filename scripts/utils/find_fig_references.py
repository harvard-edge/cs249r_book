import os
import re
import argparse
import sys

def find_fig_references(file_or_directory):
    """
    Find references in Quarto Markdown files inside the given file or directory.
    Look for references inside {#fig-} or @fig- that contain underscores.
    """
    fig_ref_pattern = re.compile(r'(?:\{#|@)fig-([^}\s]+(?:_[^}\s]+)?)')
    found_warnings = False
    if os.path.isfile(file_or_directory):
        found_warnings = process_file(file_or_directory, fig_ref_pattern)
    elif os.path.isdir(file_or_directory):
        found_warnings = process_directory(file_or_directory, fig_ref_pattern)
    else:
        print("Error: The specified file or directory does not exist.")
    if found_warnings:
        sys.exit(1)  # Exit with non-zero error code if warnings were found

def process_file(file_path, fig_ref_pattern):
    """
    Process a single QMD file.
    """
    found_warnings = False
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_number, line in enumerate(lines, start=1):
            matches = fig_ref_pattern.findall(line)
            for match in matches:
                if '_' in match:
                    print(f"Warning: Underscore in fig reference detected "
                          f"at line {line_number} in file {file_path}")
                    print(f"    Reference: {match}")
                    found_warnings = True
    return found_warnings

def process_directory(directory, fig_ref_pattern):
    """
    Process all QMD files in the specified directory.
    """
    found_warnings = False
    print(f"Searching for invalid fig references in directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".qmd"):
                file_path = os.path.join(root, file)
                if process_file(file_path, fig_ref_pattern):
                    found_warnings = True
    return found_warnings

def main():
    parser = argparse.ArgumentParser(description="Find and warn about invalid fig references in Quarto Markdown files.",
                                     epilog="The script searches for references to figures in Quarto Markdown files. "
                                            "It looks for references inside {#fig-} or @fig- that contain underscores. "
                                            "If such references are found, a warning is printed, and the script exits "
                                            "with a non-zero error code.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_or_directory", help="QMD file or directory to search for Quarto Markdown files.")
    args = parser.parse_args()

    file_or_directory = args.file_or_directory
    find_fig_references(file_or_directory)

if __name__ == "__main__":
    main()
