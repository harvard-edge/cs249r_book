# section_id_tool.py

"""
Section ID Management Script for Quarto/Markdown Book Projects
-------------------------------------------------------------

This script ensures that all section headers in your Markdown/Quarto book project have unique, clean, and consistent section IDs.

Workflow Philosophy:
--------------------
- **Write Freely:**
  - As you write, you can leave out section IDs or make up quick/guess IDs (e.g., {#sec-my-section}) for new sections.
  - Focus on content, not on perfecting section IDs.

- **Automated Cleanup:**
  - Before committing or publishing, run this script (manually, via pre-commit, or in CI).
  - The script will:
    - Add missing section IDs.
    - Normalize and fix any IDs that don't match the scheme.
    - Ensure all IDs are globally unique and clean.
    - Optionally update cross-references if IDs change.

- **Referencing Sections:**
  - While writing, use your best guess for section IDs in cross-references.
  - After running the script, you can look up the actual IDs (e.g., with grep or a helper script).

- **Verification:**
  - Run the script in --verify mode to check for missing or malformed IDs before a commit or build.

ID Scheme:
----------
- IDs are of the form: sec-[short-title]-[short-hash]
- The hash is generated from: file path + chapter title + section title + section counter
- This ensures global uniqueness, even for repeated section titles in different files.
- The visible part of the ID remains short and human-readable.

Best Practices:
---------------
- Use a pre-commit hook or CI job to enforce ID consistency.
- Use the script's --verify mode to check for issues without making changes.
- Optionally, add a script to list all section IDs and their locations for easy lookup.

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
import sys
import os
import glob
import random
import string
import time

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Global variable to track ID replacements
id_replacements = {}

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

def clean_text_for_id(text):
    """Clean text for use in section IDs."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    # Replace multiple hyphens with single hyphen
    text = re.sub(r'-+', '-', text)
    
    return text

def normalize_section_id(section_id):
    """Normalize a section ID to ensure consistent format."""
    if not section_id:
        return None
        
    # Ensure the ID starts with sec-
    if not section_id.startswith('sec-'):
        return None
        
    # Split into parts
    parts = section_id.split('-')
    if len(parts) < 3:  # Need at least: sec, chapter, section
        return None
        
    # Clean each part
    cleaned_parts = [clean_text_for_id(part) for part in parts]
    
    # Rejoin with hyphens
    normalized = '-'.join(cleaned_parts)
    
    return normalized

def is_properly_formatted_id(section_id, title, file_path):
    """Check if a section ID follows the correct format by comparing with a normalized version."""
    # Clean the title and file name
    clean_title = clean_text_for_id(title)
    file_name = clean_text_for_id(os.path.splitext(os.path.basename(file_path))[0])
    
    # Generate what the ID should be using the same hash generation logic
    base_id = f"sec-{file_name}-{clean_title}"
    hash_input = f"{title}{file_name}0"  # Using counter 0 for consistency
    hash_suffix = hashlib.sha1(hash_input.encode()).hexdigest()[:4]
    expected_id = f"{base_id}-{hash_suffix}"
    
    # Check if the ID has the required parts
    if not section_id.startswith('sec-'):
        return False, expected_id
    
    # Split into parts
    parts = section_id.split('-')
    if len(parts) < 3:  # Need at least: sec, chapter, section
        return False, expected_id
    
    # Check if it has a hash part (4 hex chars)
    if not re.search(r'-[a-f0-9]{4}$', section_id):
        return False, expected_id
    
    normalized = normalize_section_id(section_id)
    if not normalized:
        return False, expected_id
    
    return True, normalized

def generate_section_id(title, file_path, chapter_title, section_counter):
    """Generate a unique section ID based on the section title, with a hash that includes file path, chapter title, section title, and section counter."""
    clean_title = clean_text_for_id(title)
    # Hash includes file path, chapter title, section title, and section counter
    hash_input = f"{file_path}|{chapter_title}|{title}|{section_counter}".encode('utf-8')
    hash_suffix = hashlib.sha1(hash_input).hexdigest()[:4]
    return f"sec-{clean_title}-{hash_suffix}"

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

def process_markdown_file(file_path, auto_yes=False, dry_run=False, force=False):
    """Process a single Markdown file."""
    global id_replacements
    logging.info(f"\n📄 Processing: {file_path}")
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$')
    div_start_pattern = re.compile(r'^:::\s*\{\.([^"]+)')
    div_end_pattern = re.compile(r'^:::\s*$')

    inside_skip_div = False
    modified = False
    changes = []
    section_counter = 0
    chapter_title = None
    existing_sections = []

    for line in lines:
        match = header_pattern.match(line)
        if match and len(match.group(1)) == 1:
            chapter_title = match.group(2).strip()
            break

    if not chapter_title:
        raise ValueError(f"No chapter title found in {file_path}")

    for i, line in enumerate(lines):
        if div_start_pattern.match(line.strip()):
            inside_skip_div = True
        elif div_end_pattern.match(line.strip()):
            inside_skip_div = False

        match = header_pattern.match(line)
        if match and not inside_skip_div:
            hashes, title = match.groups()
            if len(hashes) > 1:
                # Extract existing attributes if any
                existing_attrs = ""
                if "{" in line:
                    attrs_start = line.find("{")
                    attrs_end = line.rfind("}")
                    if attrs_end > attrs_start:
                        existing_attrs = line[attrs_start:attrs_end+1]
                # Skip headers with {.unnumbered}
                if ".unnumbered" in existing_attrs:
                    continue  # Skip this header

                existing_id_matches = re.findall(r'\{#(sec-[^}]+)\}', line)
                if existing_id_matches:
                    existing_id = existing_id_matches[0]
                    existing_sections.append((title.strip(), existing_id))
                    is_proper, expected_id = is_properly_formatted_id(existing_id, title, file_path)
                    if not is_proper:
                        if auto_yes or input(f"\n🔄 Update ID for '{title}':\n  From: {existing_id}\n  To:   {expected_id}\n  Proceed? [Y/n]: ").lower() != 'n':
                            # Store the replacement
                            id_replacements[existing_id] = expected_id
                            # Replace only the sec- part while preserving other attributes
                            new_attrs = re.sub(r'#sec-[^}]+', f'#{expected_id}', existing_attrs)
                            new_line = f"{hashes} {title} {new_attrs}\n"
                            lines[i] = new_line
                            modified = True
                            logging.info(f"  ✓ Updated: {title}")
                            logging.info(f"    {line.strip()}")
                            logging.info(f"    → {new_line.strip()}")
                else:
                    new_id = generate_section_id(title, file_path, chapter_title, section_counter)
                    section_counter += 1
                    # Add ID while preserving other attributes
                    if existing_attrs:
                        # Remove any existing ID if present
                        attrs_without_id = re.sub(r'#sec-[^}]+', '', existing_attrs)
                        attrs_without_id = attrs_without_id.strip()
                        if attrs_without_id == "{}":
                            new_line = f"{hashes} {title} {{#{new_id}}}\n"
                        else:
                            new_line = f"{hashes} {title} {attrs_without_id} {{#{new_id}}}\n"
                    else:
                        new_line = f"{hashes} {title} {{#{new_id}}}\n"
                    lines[i] = new_line
                    modified = True
                    logging.info(f"  + Added: {title}")
                    logging.info(f"    {line.strip()}")
                    logging.info(f"    → {new_line.strip()}")

    # Show existing sections even if no changes were made
    if existing_sections:
        logging.info(f"  📋 Existing sections:")
        for title, section_id in existing_sections:
            logging.info(f"    • {title} → #{section_id}")

    if modified and not dry_run:
        path.write_text(''.join(lines), encoding="utf-8")
        logging.info(f"✅ Saved changes to {file_path}")
    elif not modified:
        logging.info(f"✓ No changes needed for {file_path}")

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

def verify_section_ids(filepath):
    """Verify that all headers have proper section IDs."""
    missing_ids = []
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$')
    div_start_pattern = re.compile(r'^:::\s*\{\.([^"]+)')
    div_end_pattern = re.compile(r'^:::\s*$')
    
    inside_skip_div = False
    for i, line in enumerate(lines, 1):
        if div_start_pattern.match(line.strip()):
            inside_skip_div = True
        elif div_end_pattern.match(line.strip()):
            inside_skip_div = False
            
        match = header_pattern.match(line)
        if match and not inside_skip_div:
            hashes, title = match.groups()
            if len(hashes) > 1:  # Skip chapter title
                if not re.search(r'\{#sec-[^}]+\}', line):
                    missing_ids.append({
                        'line': i,
                        'title': title.strip()
                    })
    
    return missing_ids

def update_cross_references(file_path, id_map):
    """Update cross-references in a file using the ID mapping."""
    global id_replacements
    
    logging.info(f"\n🔍 Checking references in: {file_path}")
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    
    # Track changes
    changes = []
    modified = False
    
    # Update each reference
    for old_id, new_id in id_map.items():
        # Update both @ and # references with word boundary
        pattern = rf'([@#]){re.escape(old_id)}\b'
        new_content = re.sub(pattern, rf'\1{new_id}', content)
        if new_content != content:
            content = new_content
            changes.append((old_id, new_id))
            modified = True
    
    if modified:
        path.write_text(content, encoding="utf-8")
        logging.info(f"✅ Updated {len(changes)} references:")
        for old, new in changes:
            logging.info(f"  - {old} → {new}")
        return True
    else:
        logging.info(f"  ✓ No references found to update")
    
    return False

def main():
    """Main function to process files."""
    global id_replacements
    # Reset id_replacements at the start of each run
    id_replacements = {}
    
    parser = argparse.ArgumentParser(description="Add unique section IDs to Markdown headers")
    parser.add_argument("-f", "--file", help="Process a single file")
    parser.add_argument("-d", "--directory", help="Process all .qmd files in directory")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-approve all changes")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing them")
    parser.add_argument("--verify", action="store_true", help="Verify section IDs without making changes")
    parser.add_argument("--force", action="store_true", help="Force update all section IDs, removing existing ones")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s"
    )

    if args.verify:
        if args.file:
            missing_ids = verify_section_ids(args.file)
            if missing_ids:
                logging.warning(f"❌ {args.file}")
                for header in missing_ids:
                    logging.warning(f"  Line {header['line']}: {header['title']} (missing ID)")
                sys.exit(1)
            else:
                logging.info(f"✅ {args.file}")
                sys.exit(0)
        elif args.directory:
            all_missing = []
            for filepath in glob.glob(os.path.join(args.directory, "**/*.qmd"), recursive=True):
                missing_ids = verify_section_ids(filepath)
                if missing_ids:
                    logging.warning(f"❌ {filepath}")
                    all_missing.append((filepath, missing_ids))
                else:
                    logging.info(f"✅ {filepath}")
            if all_missing:
                # After all files, print details for each file with missing IDs
                for filepath, missing_ids in all_missing:
                    for header in missing_ids:
                        logging.warning(f"  {filepath}: Line {header['line']}: {header['title']} (missing ID)")
                sys.exit(1)
            else:
                sys.exit(0)
        else:
            parser.error("--verify requires either --file or --directory")
    else:
        # First phase: Update all section IDs and build mapping
        if args.file:
            process_markdown_file(args.file, args.yes, args.dry_run, args.force)
            # Update cross-references in the same directory
            if not args.dry_run and id_replacements:
                logging.info("\n📝 Found the following ID replacements:")
                for old_id, new_id in id_replacements.items():
                    logging.info(f"  {old_id} → {new_id}")
                
                if args.yes or input("\n🔄 Would you like to update cross-references with these new IDs? [Y/n]: ").lower() != 'n':
                    logging.info("\n🔍 Searching for cross-references...")
                    file_dir = Path(args.file).parent
                    update_cross_references(args.file, id_replacements)
                    # Also check other files in the same directory
                    for other_file in file_dir.glob("*.qmd"):
                        if other_file != Path(args.file):
                            update_cross_references(str(other_file), id_replacements)
        elif args.directory:
            # Process all files first
            for filepath in glob.glob(os.path.join(args.directory, "**/*.qmd"), recursive=True):
                process_markdown_file(filepath, args.yes, args.dry_run, args.force)
            
            # Then update cross-references if we have replacements
            if not args.dry_run and id_replacements:
                logging.info("\n📝 Found the following ID replacements:")
                for old_id, new_id in id_replacements.items():
                    logging.info(f"  {old_id} → {new_id}")
                
                if args.yes or input("\n🔄 Would you like to update cross-references with these new IDs? [Y/n]: ").lower() != 'n':
                    logging.info("\n🔍 Searching for cross-references...")
                    # Update all files in the directory
                    for filepath in glob.glob(os.path.join(args.directory, "**/*.qmd"), recursive=True):
                        update_cross_references(filepath, id_replacements)
        else:
            parser.error("Either --file or --directory is required")

if __name__ == "__main__":
    main()
