"""
BibTeX Citation Key Formatter and .qmd File Updater

This script processes multiple BibTeX files to update citation keys to a specific format and subsequently updates references in .qmd files.

The citation keys are formatted as follows:
- The first word of the first author's name (lowercase, non-alphanumeric characters removed).
- The publication year.
- The first substantial word (longer than 3 characters) from the title (lowercase, non-alphanumeric characters removed).

The script searches recursively from the current directory for .bib and .qmd files, updating the citation keys and references accordingly.

Usage:
1. Place the script in the root directory where your .bib and .qmd files are located.
2. Run the script in your Python environment. Ensure you have 'bibtexparser' installed (`pip install bibtexparser`).
3. The script will automatically find, process, and update the files.

It is recommended to back up your files before running the script.
"""

import os
import re
import bibtexparser
from bibtexparser.bwriter import BibTexWriter

def is_valid_format(citation_key):
    """
    Check if the citation key already follows the desired format.
    """
    # Define a pattern that matches the desired format
    pattern = re.compile(r'[A-Za-z]+[0-9]{4}[A-Za-z]+')
    return pattern.fullmatch(citation_key) is not None

def extract_first_author(author_field):
    # Extract the first word of the first author's name
    authors = re.split(r'\sand\s|,', author_field)
    first_author = authors[0].strip()
    first_word = first_author.split()[0] if first_author else ''
    return first_word

def valid_citation_key(entry):
    # Generate a valid citation key
    author = extract_first_author(entry.get('author', '')).lower()
    author = re.sub(r'[^A-Za-z0-9]', '', author)
    year = entry.get('year', '')
    title = entry.get('title', '').split()
    keyword = next((word for word in title if len(word) > 3), '').lower()
    keyword = re.sub(r'[^A-Za-z0-9]', '', keyword)
    return f"{author}{year}{keyword}"

def process_bibtex_file(filepath):
    # Process a single BibTeX file
    print(f"Processing BibTeX file: {filepath}")
    with open(filepath) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    new_key_mapping = {}
    for entry in bib_database.entries:
        old_key = entry['ID']
        if not is_valid_format(old_key):
            new_key = valid_citation_key(entry)
            new_key_mapping[old_key] = new_key
            entry['ID'] = new_key
            print(f"Updated citation key: '{old_key}' to '{new_key}'")
        else:
            print(f"Citation key '{old_key}' already in correct format, no update needed.")

    with open(filepath, 'w') as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)

    return new_key_mapping

def update_qmd_files(qmd_file_path, key_mapping):
    """
    Update .qmd files with new citation keys. Handles multiple references to the same key in a file.
    """
    print(f"Updating .qmd file: {qmd_file_path}")
    with open(qmd_file_path, 'r') as file:
        content = file.read()

    for old_key, new_key in key_mapping.items():
        # Replace all occurrences of the old key with the new key
        content = content.replace(f"@{old_key}", f"@{new_key}")

    with open(qmd_file_path, 'w') as file:
        file.write(content)

def find_files(directory, extension):
    # Recursively find all files with the given extension
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(extension)]:
            yield os.path.join(dirpath, filename)

root_dir = os.getcwd()
all_mappings = {}
for filepath in find_files(root_dir, '.bib'):
    key_mapping = process_bibtex_file(filepath)
    all_mappings.update(key_mapping)

for filepath in find_files(root_dir, '.qmd'):
    update_qmd_files(filepath, all_mappings)

print("BibTeX and .qmd files have been successfully updated.")
