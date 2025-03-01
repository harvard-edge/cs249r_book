import re
import os
import argparse
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from pybtex.database import BibliographyData
from pybtex.database.input.bibtex import Parser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

def extract_bibliography_file(qmd_file):
    """Extract the bibliography file name from a .qmd file."""
    with open(qmd_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'bibliography:\s*([\w\-.]+\.bib)', content)
    return match.group(1) if match else None

def extract_cited_keys(qmd_file):
    """Extract all citation keys from a .qmd file, including DOIs, handling multi-line citations."""
    with open(qmd_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Capture standard @citations inside brackets and inline, including DOIs
    citation_keys = set(re.findall(r'@([\w\d:\-_]+)', content))

    # Capture DOIs inside [@...] citation brackets, handling in-text [-@...] cases
    doi_keys = set(re.findall(r'\[-?@]?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)', content))

    # Capture multi-line citations separated by ',' or ';', including page numbers
    multi_line_citations = set(re.findall(r'@([\w\d:\-_]+)(?:,|;|\s+p{0,2}\.\s*\d+|\s+and\s+passim)', content))

    return citation_keys | doi_keys | multi_line_citations

def remove_duplicate_entries(bib_file):
    """Remove duplicate bibliography entries before parsing and fix problematic characters in DOI fields."""
    seen_keys = set()
    unique_entries = []

    with open(bib_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entry_buffer = []
    current_key = None
    for line in lines:
        match = re.match(r'@\w+\{([^,]+),', line)
        if match:
            current_key = match.group(1)
            if current_key in seen_keys:
                entry_buffer = []  # Discard duplicate entry
                continue
            seen_keys.add(current_key)
            if entry_buffer:
                unique_entries.extend(entry_buffer)
            entry_buffer = []
        # Fix problematic \_ in DOIs
        line = line.replace('doi = "', 'doi = "').replace('\\_', '_')
        entry_buffer.append(line)

    if entry_buffer:
        unique_entries.extend(entry_buffer)

    with open(bib_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_entries)
    logging.info(f"Fixed problematic DOI formatting in {bib_file}.")

def clean_bib_file(bib_file, cited_keys, dry_run=False):
    """Remove unused entries from a .bib file and handle duplicate entries."""
    remove_duplicate_entries(bib_file)  # Ensure no duplicates and fix DOIs before parsing
    bib_database = Parser().parse_file(bib_file)

    original_keys = set(bib_database.entries.keys())
    used_entries = {key: entry for key, entry in bib_database.entries.items() if key in cited_keys}
    removed_keys = original_keys - cited_keys

    logging.info("="*50)
    logging.info(f"Processing {bib_file}:")

    if removed_keys:
        if dry_run:
            logging.info(f"[Dry Run] Would remove {len(removed_keys)} unused entries from {bib_file}:")
        else:
            new_bib = BibliographyData(entries=used_entries)
            with open(bib_file, 'w', encoding='utf-8') as f:
                f.write(new_bib.to_string('bibtex'))
            logging.info(f"Removed {len(removed_keys)} unused entries from {bib_file}:")
        for key in sorted(removed_keys):
            logging.info(f" - {key}")

    return removed_keys

def update_bib_file(bib_file):
    """Run betterbib update on the cleaned .bib file."""
    logging.info("="*50)
    try:
        logging.info(f"Updating {bib_file} using betterbib...")
        subprocess.run(["betterbib", "update", "-i", bib_file], check=True)
        logging.info(f"Successfully updated {bib_file}.")
    except FileNotFoundError:
        logging.error("Error: betterbib not found. Please install it or check your PATH.")
    except subprocess.CalledProcessError:
        logging.error(f"Error: betterbib failed to update {bib_file}.")

def process_qmd_files(qmd_files, dry_run=False, max_workers=4):
    """Find and clean up bib files used by qmd files."""
    bib_files = {}

    for qmd_file in qmd_files:
        bib_file = extract_bibliography_file(qmd_file)
        if bib_file:
            bib_path = str(Path(qmd_file).parent / bib_file)
            bib_files[bib_path] = bib_files.get(bib_path, set()) | extract_cited_keys(qmd_file)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for bib_file, keys in bib_files.items():
            if os.path.exists(bib_file):
                logging.error(f"Cleaning {bib_file}.")
                executor.submit(clean_bib_file, bib_file, keys, dry_run)
                if not dry_run:
                    executor.submit(update_bib_file, bib_file)  # Run betterbib update
            else:
                logging.warning(f"Warning: Bibliography file {bib_file} not found.")

def find_qmd_files(directory):
    """Recursively find all .qmd files in a directory."""
    return list(Path(directory).rglob("*.qmd"))

def main():
    parser = argparse.ArgumentParser(description="Clean unused entries from BibTeX files and update with betterbib.")
    parser.add_argument('-f', '--file', help="Specify a single .bib file to clean and update.")
    parser.add_argument('-d', '--directory', help="Specify a directory to scan for .qmd files and process associated .bib files.")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be removed without modifying files.")
    parser.add_argument('--workers', type=int, default=4, help="Number of parallel workers for processing.")
    args = parser.parse_args()

    if args.file:
        qmd_files = find_qmd_files(Path(args.file).parent)
        process_qmd_files(qmd_files, args.dry_run, args.workers)
    elif args.directory:
        qmd_files = find_qmd_files(args.directory)
        process_qmd_files(qmd_files, args.dry_run, args.workers)
    else:
        logging.error("Please specify either a .bib file (-f) or a directory (-d).")

if __name__ == "__main__":
    main()
