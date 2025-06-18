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

def clean_text_for_id(text):
    """Clean text to be used in an ID."""
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    # Remove any non-alphanumeric characters except hyphens
    text = re.sub(r'[^a-z0-9-]', '', text)
    # Replace multiple hyphens with single hyphen
    text = re.sub(r'-+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    return text

def normalize_section_id(section_id):
    """Normalize a section ID to the standard format: sec-<chapter>-<section>-<hash>[_suffix]."""
    if not section_id.startswith('sec-'):
        return None
    
    # Split into main parts and potential suffix
    parts = section_id.split('_', 1)
    main_id = parts[0]
    suffix = f"_{parts[1]}" if len(parts) > 1 else ""
    
    # Split the main ID into components
    components = main_id.split('-')
    if len(components) < 4:  # sec, chapter, section, hash
        return None
    
    # Clean each component
    cleaned_components = []
    for comp in components:
        # Convert to lowercase and replace underscores with hyphens
        comp = comp.lower().replace('_', '-')
        # Remove any non-alphanumeric characters except hyphens
        comp = re.sub(r'[^a-z0-9-]', '', comp)
        # Replace multiple hyphens with single hyphen
        comp = re.sub(r'-+', '-', comp)
        # Remove leading/trailing hyphens
        comp = comp.strip('-')
        cleaned_components.append(comp)
    
    # Reconstruct the normalized ID
    normalized = '-'.join(cleaned_components) + suffix
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

def generate_section_id(title, file_path, existing_ids, chapter_title, section_counter):
    """Generate a unique section ID based on the chapter title and section counter."""
    if not chapter_title:
        logging.warning(f"No chapter title found for section: {title}")
        return None
        
    # Clean the title and chapter name
    clean_title = clean_text_for_id(title)
    chapter_name = clean_text_for_id(chapter_title)
    
    # Generate hash using title, chapter, and counter for determinism
    hash_input = f"{title}{chapter_title}{section_counter}"
    hash_suffix = hashlib.sha1(hash_input.encode()).hexdigest()[:4]
    
    # Create the base ID with format: sec-<chapter>-<sectiontitle>-<hashid>
    base_id = f"sec-{chapter_name}-{clean_title}-{hash_suffix}"
    
    # Normalize the generated ID
    normalized_id = normalize_section_id(base_id)
    
    return normalized_id

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
    logging.info(f"Processing file: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Pattern to match headers with optional attributes in curly braces
    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$')
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
    modified = False
    changes = []
    existing_ids = set()  # Initialize the set of existing IDs
    section_counter = 0  # Track section number

    # First pass: find chapter title and collect existing IDs
    chapter_title = None
    for line in lines:
        match = header_pattern.match(line)
        if match and len(match.group(1)) == 1:  # Level 1 header
            chapter_title = match.group(2).strip()
            break
        # Also collect any existing section IDs
        if "#sec-" in line:
            id_matches = re.findall(r'\{#(sec-[^}]+)\}', line)
            for id_match in id_matches:
                existing_ids.add(id_match)

    if not chapter_title:
        logging.error(f"No chapter title found in {file_path}")
        return

    logging.debug(f"Found chapter title: {chapter_title}")
    logging.debug(f"Found {len(existing_ids)} existing section IDs")

    # If force mode, show warning and ask for confirmation
    if force and existing_ids and not dry_run and not auto_yes:
        print(f"\n⚠️  Force mode will remove {len(existing_ids)} existing section IDs in {file_path}")
        print("This will affect all cross-references to these sections.")
        if input("Continue? (y/n): ").lower() != 'y':
            logging.info("Operation cancelled")
            return

    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check for div start/end
        div_start_match = div_start_pattern.match(stripped)
        if div_start_match:
            div_type = div_start_match.group(1)
            inside_skip_div = div_type in skip_div_types
        elif div_end_pattern.match(stripped):
            inside_skip_div = False
        
        # Process headers only if not inside a skip div
        match = header_pattern.match(line)
        if match and not inside_skip_div:
            hashes, title = match.groups()
            
            # Only process level 2+ headers (skip chapter title)
            if len(hashes) > 1:
                # Extract existing attributes if any
                existing_attrs = ""
                if "{" in line:
                    attrs_start = line.find("{")
                    attrs_end = line.rfind("}")
                    if attrs_end > attrs_start:
                        existing_attrs = line[attrs_start:attrs_end+1]

                # Check for existing section IDs
                existing_id_matches = re.findall(r'\{#(sec-[^}]+)\}', line)
                if existing_id_matches:
                    if len(existing_id_matches) > 1:
                        # Multiple section IDs found
                        if force or auto_yes or input(f"\nFound multiple section IDs in '{title}': {', '.join(existing_id_matches)}\nRegenerate with a single ID? (y/n): ").lower() == 'y':
                            # Remove all existing section IDs
                            existing_attrs = re.sub(r'\{#sec-[^}]+\}', '', existing_attrs)
                            existing_attrs = existing_attrs.strip()
                            if existing_attrs == "{}":
                                existing_attrs = ""
                        else:
                            logging.info(f"Skipping {title} - keeping existing IDs")
                            continue
                    else:
                        # Single section ID found
                        existing_id = existing_id_matches[0]
                        is_proper, expected_id = is_properly_formatted_id(existing_id, title, file_path)
                        if not is_proper:
                            if force or auto_yes or input(f"\nFound improperly formatted ID in '{title}':\n  Current: {existing_id}\n  Should be: {expected_id}\nRegenerate with proper format? (y/n): ").lower() == 'y':
                                # Remove existing ID
                                existing_attrs = re.sub(r'\{#sec-[^}]+\}', '', existing_attrs)
                                existing_attrs = existing_attrs.strip()
                                if existing_attrs == "{}":
                                    existing_attrs = ""
                            else:
                                logging.info(f"Skipping {title} - keeping existing ID: {existing_id}")
                                continue
                        else:
                            logging.info(f"Skipping {title} - has properly formatted ID: {existing_id}")
                            continue

                # Generate ID from clean title
                proposed_id = generate_section_id(title, file_path, existing_ids, chapter_title, section_counter)
                if proposed_id:
                    section_counter += 1  # Increment counter after successful generation
                
                # Construct new line with ID
                if existing_attrs:
                    # If there are existing attributes, add ID to them
                    if ".unnumbered" in existing_attrs:
                        new_line = f"{hashes} {title} {{.unnumbered #{proposed_id}}}\n"
                    else:
                        new_line = f"{hashes} {title} {existing_attrs} {{#{proposed_id}}}\n"
                else:
                    new_line = f"{hashes} {title} {{#{proposed_id}}}\n"
                
                if new_line != line:
                    changes.append((i, line, new_line))
                    lines[i] = new_line
                    modified = True

    if modified and not dry_run:
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        if force:
            logging.info(f"✅ Force updated {len(changes)} headers in {file_path}")
        else:
            logging.info(f"✅ Updated {len(changes)} headers in {file_path}")
    elif dry_run and changes:
        logging.info(f"\nWould make {len(changes)} changes to {file_path}:")
        for i, old, new in changes:
            logging.info(f"  Line {i+1}:")
            logging.info(f"    - {old.strip()}")
            logging.info(f"    + {new.strip()}")
    elif not changes:
        logging.info(f"✅ No changes needed for {file_path}")

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
    """
    Verify that all section headers have proper section IDs.
    Returns a list of headers missing IDs.
    """
    missing_ids = []
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Pattern to match headers with optional attributes in curly braces
    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$')
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

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for div start/end
        div_start_match = div_start_pattern.match(stripped)
        if div_start_match:
            div_type = div_start_match.group(1)
            inside_skip_div = div_type in skip_div_types
        elif div_end_pattern.match(stripped):
            inside_skip_div = False
        
        # Process headers only if not inside a skip div
        match = header_pattern.match(line)
        if match and not inside_skip_div:
            hashes, title = match.groups()
            
            # Only check level 2+ headers (skip chapter title)
            if len(hashes) > 1:
                # Check if line has a section ID
                has_id = "#sec-" in line
                if not has_id:
                    missing_ids.append({
                        'line': line_num,
                        'title': title,
                        'level': len(hashes),
                        'has_id': has_id
                    })

    return missing_ids

def main():
    """Main function to process files."""
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
                print(f"\nFound {len(missing_ids)} headers missing section IDs in {args.file}:")
                for header in missing_ids:
                    print(f"  Line {header['line']}: {header['title']} (missing ID)")
                sys.exit(1)
            else:
                print(f"✅ All headers in {args.file} have proper section IDs")
                sys.exit(0)
        elif args.directory:
            all_missing = []
            for filepath in glob.glob(os.path.join(args.directory, "**/*.qmd"), recursive=True):
                missing_ids = verify_section_ids(filepath)
                if missing_ids:
                    all_missing.append((filepath, missing_ids))
            
            if all_missing:
                print("\nFound headers missing section IDs:")
                for filepath, missing_ids in all_missing:
                    print(f"\n{filepath}:")
                    for header in missing_ids:
                        print(f"  Line {header['line']}: {header['title']} (missing ID)")
                sys.exit(1)
            else:
                print("✅ All headers have proper section IDs")
                sys.exit(0)
        else:
            parser.error("--verify requires either --file or --directory")
    else:
        if args.file:
            process_markdown_file(args.file, args.yes, args.dry_run, args.force)
        elif args.directory:
            for filepath in glob.glob(os.path.join(args.directory, "**/*.qmd"), recursive=True):
                process_markdown_file(filepath, args.yes, args.dry_run, args.force)
        else:
            parser.error("Either --file or --directory is required")

if __name__ == "__main__":
    main()
