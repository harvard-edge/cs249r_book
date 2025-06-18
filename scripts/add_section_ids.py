# section_id_tool.py

"""
Section ID Tool for Markdown and Quarto Files

This utility adds or updates unique section IDs to level-2+ headers (## and deeper)
in Markdown (.md or .qmd) files. It ensures that each section ID is both human-readable
and globally unique by appending a deterministic 4-character hash suffix derived from
the header content and file path.

Key Features:
- Adds {#slug-hash} style section IDs to headers
- Skips headers inside fenced blocks (:::)
- Recursively processes a directory (-d) or handles a single file (-f)
- Prompts to replace existing IDs, with optional --yes override
- Supports --dry-run to show diffs without modifying files
- Logs every operation clearly for review and debugging

Typical Usage:
    python section_id_tool.py -f path/to/file.qmd
    python section_id_tool.py -d path/to/book/content/
    python section_id_tool.py -d content/ --yes
    python section_id_tool.py -d content/ --dry-run

Author: [Your Name]
"""

import argparse
import re
import hashlib
from pathlib import Path
import logging
import difflib
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

def simple_slugify(text):
    """Convert header text to a slug format, removing stopwords."""
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase and split into words
    words = text.lower().split()
    
    # Remove stopwords and non-alphanumeric characters
    filtered_words = []
    for word in words:
        # Remove non-alphanumeric characters
        word = re.sub(r'[^\w\s]', '', word)
        # Skip if word is empty or a stopword
        if word and word not in stop_words:
            filtered_words.append(word)
    
    # Join with hyphens
    return '-'.join(filtered_words)

def generate_hash_suffix(title, filepath):
    """Generate a deterministic 4-character SHA-1 suffix from the title and file path."""
    hash_input = f"{title}-{filepath}".encode("utf-8")
    return hashlib.sha1(hash_input).hexdigest()[:4]

def generate_section_id(header_text, filepath, existing_ids, chapter_title=None):
    """
    Generate a unique section ID using the format: sec-{chapter}_{section}_{hash}
    where:
    - chapter: slugified chapter title (no stopwords)
    - section: slugified section title (no stopwords)
    - hash: 4-character unique identifier
    
    Args:
        header_text (str): The section title
        filepath (Path): The file path
        existing_ids (set): Set of existing IDs to avoid collisions
        chapter_title (str, optional): The chapter title
    
    Returns:
        str: A unique section ID
    """
    # Get chapter slug
    chapter_slug = simple_slugify(chapter_title) if chapter_title else "intro"
    
    # Get section slug
    section_slug = simple_slugify(header_text)
    
    # Generate hash suffix
    hash_suffix = generate_hash_suffix(header_text, filepath)
    
    # Build the ID
    unique_id = f"sec-{chapter_slug}-{section_slug}_{hash_suffix}"
    
    # Handle collisions
    counter = 1
    while unique_id in existing_ids:
        unique_id = f"sec-{chapter_slug}-{section_slug}_{hash_suffix}_{counter}"
        counter += 1
    
    return unique_id

def ask_to_replace(old_id, new_id, auto_yes=False):
    """Ask user if they want to replace an existing ID."""
    if auto_yes:
        return True
    response = input(f"Replace existing ID '{old_id}' with '{new_id}'? [y/N] ").lower()
    return response == 'y'

def ask_to_update_format(old_id, new_id, auto_yes=False):
    """Ask user if they want to update a non-standard format ID."""
    if auto_yes:
        return True
    response = input(f"Update non-standard format ID '{old_id}' to standard format '{new_id}'? [y/N] ").lower()
    return response == 'y'

def ask_to_replace_existing(old_id, new_id, auto_yes=False):
    """Ask user if they want to replace an existing section ID."""
    if auto_yes:
        return True
    response = input(f"Replace existing section ID '{old_id}' with '{new_id}'? [y/N] ").lower()
    return response == 'y'

def show_diff(original, modified, filename):
    """
    Print a unified diff of the original and modified file contents.
    """
    diff = difflib.unified_diff(
        original, modified,
        fromfile=f"{filename} (original)",
        tofile=f"{filename} (modified)",
        lineterm=''
    )
    for line in diff:
        print(line)

def process_markdown_file(filepath, auto_yes=False, dry_run=False):
    """
    Process a single Markdown file. Update section headers with unique IDs.
    If dry_run is True, show diffs without writing changes.
    """
    logging.info(f"Processing file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as file:
        original_lines = file.readlines()

    modified_lines = []
    # Updated pattern to match headers with optional .unnumbered class and ID
    header_pattern = re.compile(r'^(#{1,6}) (.+?)(\s*\{\.unnumbered\})?(\s*\{#([^\}]+)\})?$')
    div_start_pattern = re.compile(r'^:::\s*\{\.([^\}]+)')
    div_end_pattern = re.compile(r'^:::\s*$')

    # Div types that should skip header processing
    skip_div_types = {
        'callout-tip',
        'callout-note',
        'callout-warning',
        'callout-important',
        'callout-caution',
        'callout'
    }

    inside_skip_div = False
    existing_ids = set()
    changed = False
    header_count = 0
    
    # First pass: find chapter title
    chapter_title = None
    for line in original_lines:
        match = header_pattern.match(line)
        if match and len(match.group(1)) == 1:  # Level 1 header
            chapter_title = match.group(2).strip()
            break
    
    logging.debug(f"Found chapter title: {chapter_title}")

    for line_num, line in enumerate(original_lines, 1):
        stripped = line.strip()
        
        # Check for div start/end
        div_start_match = div_start_pattern.match(stripped)
        if div_start_match:
            div_type = div_start_match.group(1)
            inside_skip_div = div_type in skip_div_types
            logging.debug(f"Line {line_num}: Entering div block of type: {div_type} (skip headers: {inside_skip_div})")
        elif div_end_pattern.match(stripped):
            inside_skip_div = False
            logging.debug(f"Line {line_num}: Exiting div block")
        
        # Process headers only if not inside a skip div
        match = header_pattern.match(line)
        if match and not inside_skip_div:
            hashes, title, unnumbered_class, full_id, id_only = match.groups()
            header_count += 1
            logging.debug(f"Line {line_num}: Found header: {title} (Level: {len(hashes)})")
            
            # Only process level 2+ headers (skip chapter title)
            if len(hashes) > 1:
                proposed_id = generate_section_id(title, filepath, existing_ids, chapter_title)
                
                # If header has .unnumbered class, append it to the ID
                if unnumbered_class:
                    proposed_id = f"{proposed_id}-unnumbered"
                
                if id_only:
                    if id_only == proposed_id:
                        # ID is already in the correct format
                        logging.debug(f"Line {line_num}: ID already correct: {id_only}")
                        existing_ids.add(id_only)
                    elif not id_only.startswith('sec-'):
                        # ID exists but doesn't have sec- prefix
                        if ask_to_update_format(id_only, proposed_id, auto_yes):
                            # Combine .unnumbered class with ID if present
                            if unnumbered_class:
                                line = f"{hashes} {title} {{.unnumbered #{proposed_id}}}\n"
                            else:
                                line = f"{hashes} {title} {{#{proposed_id}}}\n"
                            existing_ids.add(proposed_id)
                            changed = True
                            logging.info(f"Line {line_num}: Updated non-standard ID from {id_only} to {proposed_id}")
                        else:
                            existing_ids.add(id_only)
                            logging.info(f"Line {line_num}: Kept non-standard ID: {id_only}")
                    else:
                        # ID exists with sec- prefix but is different
                        if ask_to_replace_existing(id_only, proposed_id, auto_yes):
                            # Combine .unnumbered class with ID if present
                            if unnumbered_class:
                                line = f"{hashes} {title} {{.unnumbered #{proposed_id}}}\n"
                            else:
                                line = f"{hashes} {title} {{#{proposed_id}}}\n"
                            existing_ids.add(proposed_id)
                            changed = True
                            logging.info(f"Line {line_num}: Replaced existing section ID from {id_only} to {proposed_id}")
                        else:
                            existing_ids.add(id_only)
                            logging.info(f"Line {line_num}: Kept existing section ID: {id_only}")
                else:
                    # No ID exists, add new one
                    # Combine .unnumbered class with ID if present
                    if unnumbered_class:
                        line = f"{hashes} {title} {{.unnumbered #{proposed_id}}}\n"
                    else:
                        line = f"{hashes} {title} {{#{proposed_id}}}\n"
                    existing_ids.add(proposed_id)
                    logging.info(f"Line {line_num}: Added new ID: {proposed_id}")
                    changed = True
            
        modified_lines.append(line)

    logging.info(f"Processed {header_count} headers in {filepath}")
    if changed:
        if dry_run:
            show_diff(original_lines, modified_lines, filepath)
            logging.info(f"🧪 Dry run: No changes written to {filepath}\n")
        else:
            with open(filepath, "w", encoding="utf-8") as file:
                file.writelines(modified_lines)
            logging.info(f"✅ File updated: {filepath}\n")
    else:
        logging.info(f"– No changes needed: {filepath}\n")

def process_directory(directory, auto_yes=False, dry_run=False):
    """
    Recursively process all Markdown and Quarto files in a directory.
    """
    path = Path(directory)
    if not path.exists():
        logging.error(f"Directory does not exist: {directory}")
        return

    all_files = list(path.rglob("*.md")) + list(path.rglob("*.qmd"))
    if not all_files:
        logging.warning(f"No markdown files found in directory: {directory}")
        return

    for file_path in all_files:
        process_markdown_file(file_path, auto_yes=auto_yes, dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(
        description="Add or update unique section IDs to markdown headers (## and deeper)."
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to a single markdown (.md or .qmd) file"
    )
    parser.add_argument(
        "-d", "--directory",
        help="Path to a directory to recursively process markdown files"
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Automatically accept all ID replacements without prompting"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show changes without modifying any files"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Set up logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.file:
        process_markdown_file(Path(args.file), auto_yes=args.yes, dry_run=args.dry_run)
    elif args.directory:
        process_directory(Path(args.directory), auto_yes=args.yes, dry_run=args.dry_run)
    else:
        logging.error("You must provide either a file (-f) or a directory (-d).")
        parser.print_help()

if __name__ == "__main__":
    main()
