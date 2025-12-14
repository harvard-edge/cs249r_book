# section_id_manager.py

"""
Comprehensive Section ID Management Script for Quarto/Markdown Book Projects
---------------------------------------------------------------------------

This script provides a complete toolkit for managing section IDs in your Markdown/Quarto book project.
It ensures that all section headers have unique, clean, and consistent section IDs while preserving
cross-references and other attributes. It also automatically updates corresponding quiz JSON files
when section IDs change to maintain synchronization between content and assessments.

Special Handling for Unnumbered Headers:
---------------------------------------
- Any header with the {.unnumbered} class is always skipped for section ID management.
- Unnumbered headers will never have a section ID added, updated, or required (including in verify mode).
- All headers (including chapter headers) are processed by default unless they have {.unnumbered}.

Smart Block Detection:
---------------------
- **Code Blocks:** Headers inside code blocks (```...```) are automatically ignored
- **Div Blocks:** Headers inside Quarto divs (::: {.class}...:::) are automatically ignored
- **Callouts:** Headers inside callout divs are automatically ignored
- **Comments:** R/Python comments like `## Section Name` inside code blocks are not treated as headers

This prevents the script from incorrectly processing code comments or headers that are part of
documentation examples rather than actual section headers.

Workflow Philosophy:
--------------------
- **Write Freely:**
  - As you write, you can leave out section IDs or make up quick/guess IDs (e.g., {#sec-my-section}) for new sections.
  - Focus on content, not on perfecting section IDs.

- **Automated Management:**
  - Before committing or publishing, run this script (manually, via pre-commit, or in CI).
  - The script can:
    - Add missing section IDs (except for unnumbered headers)
    - Repair existing IDs to match the new format
    - Remove all IDs for a fresh start
    - Verify all IDs are present and properly formatted (skipping unnumbered headers)
    - List all IDs for reference
    - Create backups before making changes
    - Update cross-references when IDs change

- **Referencing Sections:**
  - While writing, use your best guess for section IDs in cross-references.
  - After running the script, you can look up the actual IDs using the --list mode.

- **Safety First:**
  - Use --backup to create timestamped backups before making changes
  - Use --dry-run to preview changes without modifying files
  - Use --verify to check for issues before committing (unnumbered headers are always ignored)

ID Scheme:
----------
- IDs are of the form: sec-{chapter-title}-{section-title}-{hash}
- Chapter and section titles have stopwords removed for cleaner IDs
- The hash is generated from: file path + chapter title + section title + parent section hierarchy
- Section content is NOT included in the hash to ensure IDs remain stable when content changes
- This ensures GLOBAL UNIQUENESS across the entire book project
- Different files with identical section names and hierarchies will have different IDs
- Parent sections are included in the hash to handle duplicate section names naturally
- The visible part of the ID remains short and human-readable
- IDs are stable and won't change if sections are reordered (as long as hierarchy doesn't change)

Stable ID Generation:
---------------------
- Section IDs are based on structural information (file path, chapter title, section title, hierarchy)
- Section content is NOT included in the hash to ensure stability when content changes
- Running --repair multiple times will not change IDs unless structure actually changes
- This prevents ID churn and ensures cross-references remain valid when content is modified

Global Uniqueness Guarantee:
----------------------------
The hash generation includes the file path to ensure that sections with identical names
and hierarchies in different files will have different IDs. This prevents conflicts when:

- Multiple chapters have sections with the same name (e.g., "Introduction" in different files)
- Different files have identical section hierarchies (e.g., "Techniques > Advanced > Optimization")
- The same section name appears in multiple contexts across the book

Example hash inputs:
  - File A: "contents/chapter1.qmd|Getting Started|Introduction"
  - File B: "contents/chapter2.qmd|Getting Started|Introduction"
  - Result: Different 4-character hashes ensure unique IDs

Available Modes:
----------------
- **Add Mode (default):** Add missing section IDs to all headers including chapters (skips unnumbered headers and code blocks)
- **Repair Mode (--repair):** Fix existing section IDs to match the new format (stable across multiple runs)
- **Remove Mode (--remove):** Remove all section IDs (use with --backup)
- **Verify Mode (--verify):** Check that all section IDs are present and properly formatted for all headers including chapters (skips unnumbered headers and code blocks)
- **List Mode (--list):** Display all section IDs found in files (skips code blocks)

Safety Features:
----------------
- **Backup System:** --backup creates .backup.{timestamp} files before changes
- **Dry Run:** --dry-run shows what would change without modifying files
- **Interactive Prompts:** Asks for confirmation before making changes
- **Force Mode:** --force automatically accepts all confirmations without prompting
- **Attribute Preservation:** Maintains other attributes when modifying section IDs
- **Cross-reference Updates:** Automatically updates references when IDs change
- **Stable IDs:** IDs remain consistent across multiple repair runs

Best Practices:
---------------
- Use --backup when making bulk changes
- Use --verify before commits to ensure ID integrity (unnumbered headers and code blocks are always ignored)
- Use --list to audit existing section IDs
- Use --dry-run to preview changes before applying them
- Consider using this in pre-commit hooks or CI pipelines
- Run --repair as many times as needed - IDs will remain stable

Key Features:
- Comprehensive section ID management (add, repair, remove, verify, list)
- Hierarchy-based ID generation that reflects document structure
- Natural handling of duplicate section names through parent section context
- Global uniqueness guaranteed through file path inclusion in hash
- Stable IDs that don't change when sections are reordered
- Smart attribute preservation (e.g., {.class #sec-id .other-class})
- Cross-reference updating when IDs change
- Backup creation for safety
- Detailed summaries and progress reporting
- Support for both single files (-f) and directories (-d)
- Stopword removal for cleaner, more readable IDs
- **All headers including chapters are processed by default; use {.unnumbered} to skip specific headers**
- **Code blocks and divs are automatically detected and skipped**
- **Stable ID generation prevents unnecessary changes**

Code Quality:
-------------
- Shared functions eliminate code duplication
- Consistent block detection logic across all modes
- Modular design with clear separation of concerns
- Comprehensive error handling and validation

Typical Usage:
    # Add missing IDs
    python section_id_manager.py -d contents/
    python section_id_manager.py -f contents/chapter.qmd

    # Repair existing IDs (stable across multiple runs)
    python section_id_manager.py -d contents/ --repair --backup
    python section_id_manager.py -d contents/ --repair --force

    # Verify all IDs including chapters (skips unnumbered headers and code blocks)
    python section_id_manager.py -d contents/ --verify

    # List all IDs (skips code blocks)
    python section_id_manager.py -d contents/ --list

    # Remove all IDs (dangerous!)
    python section_id_manager.py -d contents/ --remove --backup

    # Preview changes
    python section_id_manager.py -d contents/ --repair --dry-run

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
import time
import json

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

# Shared regex patterns - defined once to avoid duplication
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?$')
DIV_START_PATTERN = re.compile(r'^:::\s*\{\.([^"]+)')
DIV_END_PATTERN = re.compile(r'^:::\s*$')
CODE_BLOCK_PATTERN = re.compile(r'^```[^`]*$')  # Matches code block start/end

def initialize_block_tracking():
    """Initialize block tracking state variables."""
    return {
        'inside_skip_div': False,
        'inside_code_block': False
    }

def update_block_state(line, state):
    """
    Update block tracking state based on the current line.

    Args:
        line: The current line being processed
        state: Dictionary with 'inside_skip_div' and 'inside_code_block' keys

    Returns:
        Updated state dictionary
    """
    line_stripped = line.strip()

    # Check for code block boundaries
    if CODE_BLOCK_PATTERN.match(line_stripped):
        state['inside_code_block'] = not state['inside_code_block']
        return state

    # Check for div boundaries
    if DIV_START_PATTERN.match(line_stripped):
        state['inside_skip_div'] = True
    elif DIV_END_PATTERN.match(line_stripped):
        state['inside_skip_div'] = False

    return state

def should_process_header(line, state):
    """
    Determine if a header should be processed based on current block state.

    Args:
        line: The current line
        state: Block tracking state dictionary

    Returns:
        True if the header should be processed, False otherwise
    """
    match = HEADER_PATTERN.match(line)
    if match and not state['inside_skip_div'] and not state['inside_code_block']:
        return True, match
    return False, None

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

def normalize_content_for_hash(content):
    """
    Normalize content for hashing to reduce noise from minor formatting changes.

    This function removes or normalizes formatting that doesn't change the semantic meaning
    of the content, so that minor formatting changes don't cause section IDs to change.

    Args:
        content: Raw content string from the section

    Returns:
        Normalized content string suitable for hashing
    """
    if not content:
        return ""

    # Remove extra whitespace and normalize
    normalized = re.sub(r'\s+', ' ', content.strip())

    # Remove basic markdown formatting that doesn't change meaning
    normalized = re.sub(r'[*_`]', '', normalized)  # Remove emphasis markers
    normalized = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', normalized)  # Convert links to text
    normalized = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', normalized)  # Convert images to alt text

    # Remove HTML tags (basic)
    normalized = re.sub(r'<[^>]+>', '', normalized)

    # Remove code block markers (multiline)
    normalized = re.sub(r'```[^`]*```', '', normalized, flags=re.DOTALL)
    normalized = re.sub(r'`[^`]+`', '', normalized)

    # Remove blockquote markers (multiline)
    normalized = re.sub(r'^>\s*', '', normalized, flags=re.MULTILINE)

    # Remove list markers (multiline)
    normalized = re.sub(r'^[\s]*[-*+]\s+', '', normalized, flags=re.MULTILINE)
    normalized = re.sub(r'^[\s]*\d+\.\s+', '', normalized, flags=re.MULTILINE)

    # Clean up any remaining extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized

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

def is_properly_formatted_id(section_id, title, file_path, chapter_title, section_counter):
    """Check if a section ID follows the correct format."""
    # Check if the ID has the required parts
    if not section_id.startswith('sec-'):
        return False, None

    # Split into parts
    parts = section_id.split('-')
    if len(parts) < 4:  # Need at least: sec, chapter, section, hash
        return False, None

    # Check if it has a hash part (4 hex chars)
    if not re.search(r'-[a-f0-9]{4}$', section_id):
        return False, None

    # If it passes all format checks, it's properly formatted
    return True, section_id

def generate_section_id(title, file_path, chapter_title, section_counter, parent_sections=None, section_content=None, is_chapter=False):
    """
    Generate a unique section ID based on the section title and hierarchy.

    The hash includes file path, chapter title, section title, and parent section hierarchy
    to ensure uniqueness across the entire book project. Content is NOT included in the hash
    to ensure IDs remain stable when content changes (e.g., when quizzes are added/removed).

    Args:
        title: The section title
        file_path: The file path (included in hash for location tracking)
        chapter_title: The chapter title
        section_counter: Counter for this section (not used in hash)
        parent_sections: List of parent section titles (included in hash)
        section_content: The content of the section (ignored - not used in hash)

    Returns:
        For chapters: sec-{chapter-slug}
        For sections: sec-{chapter-slug}-{section-slug}-{4-char-hash}

    Example:
        Chapter: "Introduction" → sec-introduction
        Section: "Getting Started" in "Introduction" → sec-introduction-getting-started-d212
    """
    clean_title = simple_slugify(title)

    # For chapter-level headers, use simple format
    if is_chapter:
        return f"sec-{clean_title}"

    # For subsections, use the full format with chapter prefix and hash
    clean_chapter_title = simple_slugify(chapter_title)

    # Build hierarchy string from parent sections
    hierarchy = ""
    if parent_sections:
        # Create a hierarchy string from all parent sections
        hierarchy_parts = []
        for parent in parent_sections:
            hierarchy_parts.append(simple_slugify(parent))
        hierarchy = "|".join(hierarchy_parts)

    # Hash includes file path, chapter title, section title, and parent hierarchy only
    # Content is excluded to ensure IDs remain stable when content changes
    hash_input = f"{file_path}|{chapter_title}|{title}|{hierarchy}".encode('utf-8')
    hash_suffix = hashlib.sha1(hash_input).hexdigest()[:4]  # Keep 4 chars
    return f"sec-{clean_chapter_title}-{clean_title}-{hash_suffix}"

def list_section_ids(filepath):
    """List all section IDs found in a single file."""
    logging.info(f"\n📋 Section IDs in: {filepath}")
    logging.info(f"{'='*60}")

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    state = initialize_block_tracking()
    section_count = 0

    for i, line in enumerate(lines, 1):
        # Update block state
        state = update_block_state(line, state)

        # Check if we should process this header
        should_process, match = should_process_header(line, state)
        if should_process:
            hashes, title = match.groups()
            # Process all headers (including chapter headers) unless they have {.unnumbered}
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

            section_count += 1
            existing_id_matches = re.findall(r'\{#(sec-[^}]+)\}', line)
            if existing_id_matches:
                section_id = existing_id_matches[0]
                logging.info(f"  {section_count:2d}. {title.strip()}")
                logging.info(f"      ID: #{section_id}")
            else:
                logging.info(f"  {section_count:2d}. {title.strip()} (NO ID)")

    if section_count == 0:
        logging.info("  No sections found")
    else:
        logging.info(f"\n  Total sections: {section_count}")

def list_all_section_ids(directory):
    """List all section IDs found in all files in a directory."""
    path = Path(directory)
    if not path.exists():
        logging.error(f"Directory does not exist: {directory}")
        return

    all_files = list(path.rglob("*.md")) + list(path.rglob("*.qmd"))
    if not all_files:
        logging.warning(f"No markdown files found in directory: {directory}")
        return

    total_sections = 0
    total_with_ids = 0

    for file_path in all_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        state = initialize_block_tracking()
        file_sections = 0
        file_with_ids = 0

        for line in lines:
            # Update block state
            state = update_block_state(line, state)

            # Check if we should process this header
            should_process, match = should_process_header(line, state)
            if should_process:
                hashes, title = match.groups()
                if len(hashes) > 1:  # Skip chapter title
                    file_sections += 1
                    if re.search(r'\{#sec-[^}]+\}', line):
                        file_with_ids += 1

        if file_sections > 0:
            logging.info(f"📄 {file_path}: {file_with_ids}/{file_sections} sections have IDs")
            total_sections += file_sections
            total_with_ids += file_with_ids

    logging.info(f"\n📊 SUMMARY:")
    logging.info(f"  Total files: {len(all_files)}")
    logging.info(f"  Total sections: {total_sections}")
    logging.info(f"  Sections with IDs: {total_with_ids}")
    logging.info(f"  Sections missing IDs: {total_sections - total_with_ids}")

def extract_section_content(lines, section_start_index, header_level):
    """
    Extract the content of a section from the markdown file.

    Args:
        lines: List of lines in the file
        section_start_index: Index of the section header line
        header_level: Level of the section header (2-6)

    Returns:
        String containing the section content (normalized)
    """
    content_lines = []
    i = section_start_index + 1
    state = initialize_block_tracking()  # Track code/div blocks

    while i < len(lines):
        line = lines[i]
        state = update_block_state(line, state)
        line_stripped = line.strip()

        # Only treat ## as section end if not inside a code or div block
        if not state['inside_code_block'] and not state['inside_skip_div']:
            if line_stripped.startswith('#'):
                next_header_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                if next_header_level <= header_level:
                    break
                # If this is a header, strip attributes after '{'
                if '{' in line_stripped:
                    line_stripped = line_stripped[:line_stripped.find('{')].strip()

        # Stop if we hit a div boundary (but only if not inside code/div block)
        # (This is now handled by block tracking)
        # Stop if we hit a code block boundary (also handled by block tracking)

        # For all lines, if '{' is present, strip everything after it
        if '{' in line_stripped:
            line_stripped = line_stripped[:line_stripped.find('{')].strip()

        # Add non-empty lines to content
        if line_stripped:
            content_lines.append(line_stripped)

        i += 1

    return ' '.join(content_lines)

def create_backup(file_path):
    """Create a backup of the file before making changes."""
    backup_path = f"{file_path}.backup.{int(time.time())}"
    import shutil
    shutil.copy2(file_path, backup_path)
    logging.info(f"💾 Created backup: {backup_path}")
    return backup_path

def process_markdown_file(file_path, auto_yes=False, force=False, dry_run=False, repair_mode=False, remove_mode=False, backup_mode=False):
    """Process a single Markdown file."""
    global id_replacements
    logging.info(f"\n📄 Processing: {file_path}")

    # Create backup if requested
    if backup_mode and not dry_run:
        create_backup(file_path)

    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    state = initialize_block_tracking()
    modified = False
    changes = []
    section_counter = 0
    chapter_title = None
    existing_sections = []

    # Track section hierarchy
    section_hierarchy = []  # Stack of parent sections

    file_summary = {
        'file_path': file_path,
        'added_ids': [],
        'updated_ids': [],
        'removed_ids': [],
        'existing_sections': [],
        'modified': False
    }

    # Find chapter title
    for line in lines:
        should_process, match = should_process_header(line, state)
        if should_process and len(match.group(1)) == 1:
            chapter_title = match.group(2).strip()
            break

    if not chapter_title:
        raise ValueError(f"No chapter title found in {file_path}")

    # Reset state for main processing
    state = initialize_block_tracking()

    for i, line in enumerate(lines):
        # Update block state
        state = update_block_state(line, state)

        # Check if we should process this header
        should_process, match = should_process_header(line, state)
        if should_process:
            hashes, title = match.groups()
            header_level = len(hashes)

            # Process all headers (including chapter headers) unless they have {.unnumbered}
            # Update section hierarchy based on header level
            while len(section_hierarchy) >= header_level:
                section_hierarchy.pop()

            # Add current section to hierarchy (will be used for next section)
            section_hierarchy.append(title.strip())

            # Get parent sections for current section (exclude the current section itself)
            parent_sections = section_hierarchy[:-1] if len(section_hierarchy) > 1 else []

            # Extract existing attributes if any
            existing_attrs = ""
            if "{" in line:
                attrs_start = line.find("{")
                attrs_end = line.rfind("}")
                if attrs_end > attrs_start:
                    existing_attrs = line[attrs_start:attrs_end+1]
            # Skip headers with {.unnumbered}
            if ".unnumbered" in existing_attrs:
                # Remove any existing section ID from unnumbered headers
                existing_id_matches = re.findall(r'\{#(sec-[^}]+)\}', line)
                if existing_id_matches:
                    existing_id = existing_id_matches[0]
                    # Remove the section ID while preserving other attributes
                    new_attrs = re.sub(r'#sec-[^}\s]+', '', existing_attrs)
                    # Remove duplicate .unnumbered
                    new_attrs = re.sub(r'(\.unnumbered)(?=.*\.unnumbered)', '', new_attrs)
                    # Remove extra whitespace
                    new_attrs = re.sub(r'\s+', ' ', new_attrs).strip()
                    # Remove empty braces or braces with only whitespace
                    if new_attrs in ["{}", "{ }", ""]:
                        new_line = f"{hashes} {title}\n"
                    else:
                        new_line = f"{hashes} {title} {new_attrs}\n"
                    lines[i] = new_line
                    modified = True
                    file_summary['modified'] = True
                    file_summary['removed_ids'].append((title.strip(), existing_id))
                    logging.info(f"  🗑️  Removed ID from unnumbered header: {title}")
                    logging.info(f"    {line.strip()}")
                    logging.info(f"    → {new_line.strip()}")
                continue  # Skip this header

            existing_id_matches = re.findall(r'\{#(sec-[^}]+)\}', line)
            if existing_id_matches:
                existing_id = existing_id_matches[0]
                existing_sections.append((title.strip(), existing_id))
                file_summary['existing_sections'].append((title.strip(), existing_id))

                if remove_mode:
                    # Remove the section ID
                    if auto_yes or force or input(f"\n🗑️  Remove ID for '{title}': {existing_id}? [Y/n]: ").lower() != 'n':
                        # Remove only the sec- part while preserving other attributes
                        new_attrs = re.sub(r'#sec-[^}\s]+', '', existing_attrs)
                        # Clean up any double spaces or empty braces
                        new_attrs = re.sub(r'\s+', ' ', new_attrs).strip()
                        if new_attrs == "{}":
                            new_line = f"{hashes} {title}\n"
                        else:
                            new_line = f"{hashes} {title} {new_attrs}\n"
                        lines[i] = new_line
                        modified = True
                        file_summary['modified'] = True
                        file_summary['removed_ids'].append((title.strip(), existing_id))
                        logging.info(f"  🗑️  Removed: {title}")
                        logging.info(f"    {line.strip()}")
                        logging.info(f"    → {new_line.strip()}")
                else:
                    # Extract section content for content-aware ID generation
                    section_content = extract_section_content(lines, i, header_level)

                    # Generate the new ID in the standard format with parent hierarchy and content
                    is_chapter = (header_level == 1)
                    new_id = generate_section_id(title, file_path, chapter_title, section_counter, parent_sections, section_content, is_chapter)
                    section_counter += 1

                    # Check if the existing ID needs to be repaired/updated
                    is_proper, expected_id = is_properly_formatted_id(existing_id, title, file_path, chapter_title, section_counter)

                    # In repair mode, always update to the new format
                    # In normal mode, only update if the format is improper
                    should_update = repair_mode or not is_proper

                    if should_update:
                        if existing_id == new_id:
                            continue  # No change needed, skip
                        if auto_yes or force or input(f"\n🔄 Update ID for '{title}':\n  From: {existing_id}\n  To:   {new_id}\n  Proceed? [Y/n]: ").lower() != 'n':
                            # Store the replacement
                            id_replacements[existing_id] = new_id
                            # Replace only the sec- part while preserving other attributes
                            # This handles cases like: {.class #sec-old-id .other-class}
                            new_attrs = re.sub(r'#sec-[^}\s]+', f'#{new_id}', existing_attrs)
                            new_line = f"{hashes} {title} {new_attrs}\n"
                            lines[i] = new_line
                            modified = True
                            file_summary['modified'] = True
                            file_summary['updated_ids'].append((title.strip(), existing_id, new_id))
                            logging.info(f"  ✓ Updated: {title}")
                            logging.info(f"    {line.strip()}")
                            logging.info(f"    → {new_line.strip()}")
            else:
                if not remove_mode:  # Only add IDs if not in remove mode
                    # Extract section content for content-aware ID generation
                    section_content = extract_section_content(lines, i, header_level)

                    # Generate the new ID in the standard format with parent hierarchy and content
                    is_chapter = (header_level == 1)
                    new_id = generate_section_id(title, file_path, chapter_title, section_counter, parent_sections, section_content, is_chapter)
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
                    file_summary['modified'] = True
                    file_summary['added_ids'].append((title.strip(), new_id))
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

    return file_summary

def process_directory(directory, auto_yes=False, force=False, dry_run=False, repair_mode=False, remove_mode=False, backup_mode=False):
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

    logging.info(f"\n{'='*60}")
    logging.info(f"🔍 PROCESSING DIRECTORY: {directory}")
    logging.info(f"📁 Found {len(all_files)} files to process")
    logging.info(f"{'='*60}")

    # Collect summaries from all files
    all_summaries = []
    for i, file_path in enumerate(all_files, 1):
        logging.info(f"\n📄 [{i}/{len(all_files)}] Processing: {file_path}")
        logging.info(f"{'-'*60}")

        file_summary = process_markdown_file(file_path, auto_yes=auto_yes, force=force, dry_run=dry_run, repair_mode=repair_mode, remove_mode=remove_mode, backup_mode=backup_mode)
        all_summaries.append(file_summary)

        # Add a separator between files
        if i < len(all_files):
            logging.info(f"{'-'*60}")

    # Print overall summary
    print_summary(all_summaries)

def print_summary(all_summaries):
    """Print a comprehensive summary of all changes made across files."""
    total_files = len(all_summaries)
    files_modified = sum(1 for summary in all_summaries if summary['modified'])
    total_added = sum(len(summary['added_ids']) for summary in all_summaries)
    total_updated = sum(len(summary['updated_ids']) for summary in all_summaries)
    total_removed = sum(len(summary['removed_ids']) for summary in all_summaries)
    total_existing = sum(len(summary['existing_sections']) for summary in all_summaries)

    logging.info(f"\n{'='*60}")
    logging.info(f"📊 FINAL SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"📁 Files processed: {total_files}")
    logging.info(f"✅ Files modified: {files_modified}")
    logging.info(f"➕ Section IDs added: {total_added}")
    logging.info(f"🔄 Section IDs updated: {total_updated}")
    logging.info(f"🗑️  Section IDs removed: {total_removed}")
    logging.info(f"📋 Existing sections found: {total_existing}")

    if total_added > 0 or total_updated > 0 or total_removed > 0:
        logging.info(f"\n FILES WITH CHANGES:")
        logging.info(f"{'-'*60}")

        for summary in all_summaries:
            if summary['added_ids'] or summary['updated_ids'] or summary['removed_ids']:
                logging.info(f"\n📄 {summary['file_path']}:")

                if summary['added_ids']:
                    logging.info(f"  ➕ Added {len(summary['added_ids'])} section IDs")

                if summary['updated_ids']:
                    logging.info(f"  🔄 Updated {len(summary['updated_ids'])} section IDs")
                    # Show first few updates as examples
                    for i, (title, old_id, new_id) in enumerate(summary['updated_ids'][:3]):
                        logging.info(f"    • {title}: {old_id} → {new_id}")
                    if len(summary['updated_ids']) > 3:
                        logging.info(f"    ... and {len(summary['updated_ids']) - 3} more")

                if summary['removed_ids']:
                    logging.info(f"  🗑️  Removed {len(summary['removed_ids'])} section IDs")

    logging.info(f"\n{'='*60}")

def verify_section_ids(filepath):
    """Verify that all headers have proper section IDs, skipping unnumbered headers."""
    missing_ids = []
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    state = initialize_block_tracking()
    for i, line in enumerate(lines, 1):
        # Update block state
        state = update_block_state(line, state)

        # Check if we should process this header
        should_process, match = should_process_header(line, state)
        if should_process:
            hashes, title = match.groups()
            # Process all headers (including chapter headers) unless they have {.unnumbered}
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

    # Handle JSON files differently
    if path.suffix.lower() == '.json':
        return update_quiz_json(file_path, id_map)

    # Handle text files (QMD, MD, etc.)
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

def update_quiz_json(file_path, id_map):
    """Update section IDs in a quiz JSON file."""
    global id_replacements

    logging.info(f"\n🔍 Checking quiz JSON in: {file_path}")
    path = Path(file_path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            quiz_data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return False

    # Track changes
    changes = []
    modified = False

    # First, update section_id fields in the structure
    for section in quiz_data.get('sections', []):
        old_section_id = section.get('section_id')
        if old_section_id and old_section_id in id_map:
            new_section_id = id_map[old_section_id]
            section['section_id'] = new_section_id
            changes.append((old_section_id, new_section_id))
            modified = True

    # Then, search for any other occurrences of old IDs in the entire JSON content
    # Convert to string, replace, then parse back
    json_str = json.dumps(quiz_data, indent=2)
    original_json_str = json_str

    for old_id, new_id in id_map.items():
        if old_id in json_str:
            json_str = json_str.replace(old_id, new_id)
            if json_str != original_json_str:
                modified = True
                # Only add to changes if not already added from section_id field
                if (old_id, new_id) not in changes:
                    changes.append((old_id, new_id))

    if modified:
        # Parse back to JSON to ensure it's valid
        try:
            updated_quiz_data = json.loads(json_str)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(updated_quiz_data, f, indent=2)
            logging.info(f"✅ Updated {len(changes)} section IDs in {file_path}")
            for old, new in changes:
                logging.info(f"  - {old} → {new}")
            return True
        except json.JSONDecodeError as e:
            logging.error(f"Error after updating JSON in {file_path}: {e}")
            return False
    else:
        logging.info(f"  ✓ No section IDs found to update in {file_path}")

    return False

def main():
    """Main function to process files."""
    global id_replacements
    # Reset id_replacements at the start of each run
    id_replacements = {}

    parser = argparse.ArgumentParser(
        description="Comprehensive Section ID Management for Quarto/Markdown Book Projects",
        epilog="""
Section IDs are critical for cross-referencing and navigation. This tool helps maintain them.

MODE EXAMPLES:

  Add missing IDs:
    python section_id_manager.py -d contents/
    python section_id_manager.py -f contents/chapter.qmd


  Repair existing IDs:
    python section_id_manager.py -d contents/ --repair
    python section_id_manager.py -f contents/chapter.qmd --repair


  Force repair (no prompts):
    python section_id_manager.py -d contents/ --repair --force
    python section_id_manager.py -f contents/chapter.qmd --repair --force


  Remove all IDs:
    python section_id_manager.py -d contents/ --remove
    python section_id_manager.py -f contents/chapter.qmd --remove


  Verify all IDs:
    python section_id_manager.py -d contents/ --verify
    python section_id_manager.py -f contents/chapter.qmd --verify


  List all IDs:
    python section_id_manager.py -d contents/ --list
    python section_id_manager.py -f contents/chapter.qmd --list


  Safe repair (with backup):
    python section_id_manager.py -d contents/ --repair --backup
    python section_id_manager.py -f contents/chapter.qmd --repair --backup
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-f", "--file", help="Process a single file")
    parser.add_argument("-d", "--directory", help="Process all .qmd files in directory")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-approve all changes (use with caution)")
    parser.add_argument("--force", action="store_true", help="Force all operations without confirmation prompts")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing them")
    parser.add_argument("--verify", action="store_true", help="Verify all section IDs are present (⚠️  does NOT check format)")
    parser.add_argument("--repair", action="store_true", help="Repair existing section IDs to match the new format (preserves other attributes)")
    parser.add_argument("--remove", action="store_true", help="Remove all section IDs (use with --backup for safety)")
    parser.add_argument("--list", action="store_true", help="List all section IDs found in files")
    parser.add_argument("--backup", action="store_true", help="Create backup files before making changes")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s"
    )

    # Validate mode combinations
    mode_count = sum([args.verify, args.repair, args.remove, args.list])
    if mode_count > 1:
        parser.error("Only one mode can be specified: --verify, --repair, --remove, or --list")

    if args.verify:
        logging.warning("⚠️  VERIFY MODE: This only checks if section IDs are present, not if they follow the correct format.")
        logging.warning("   Use --repair to fix IDs that don't match the expected format.")
        if not (args.yes or args.force) and input("Continue with format-agnostic verification? [Y/n]: ").lower() == 'n':
            logging.info("Verification cancelled.")
            sys.exit(0)

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
    elif args.list:
        if args.file:
            list_section_ids(args.file)
        elif args.directory:
            list_all_section_ids(args.directory)
        else:
            parser.error("--list requires either --file or --directory")
    else:
        # First phase: Update all section IDs and build mapping
        if args.file:
            file_summary = process_markdown_file(args.file, args.yes, args.force, args.dry_run, args.repair, args.remove, args.backup)
            # Print summary for single file
            print_summary([file_summary])

            # Update cross-references in the same directory
            if not args.dry_run and id_replacements:
                logging.info("\n📝 Found the following ID replacements:")
                for old_id, new_id in id_replacements.items():
                    logging.info(f"  {old_id} → {new_id}")

                if args.yes or args.force or input("\n🔄 Would you like to update cross-references with these new IDs? [Y/n]: ").lower() != 'n':
                    logging.info("\n🔍 Searching for cross-references...")
                    file_dir = Path(args.file).parent
                    update_cross_references(args.file, id_replacements)
                    # Also check other files in the same directory
                    for other_file in file_dir.glob("*.qmd"):
                        if other_file != Path(args.file):
                            update_cross_references(str(other_file), id_replacements)

                    # Update all quiz JSON files in the same directory
                    for quiz_file in file_dir.glob("*_quizzes.json"):
                        update_cross_references(str(quiz_file), id_replacements)
        elif args.directory:
            # Process all files with summary
            process_directory(args.directory, args.yes, args.force, args.dry_run, args.repair, args.remove, args.backup)

            # Then update cross-references if we have replacements
            if not args.dry_run and id_replacements:
                logging.info("\n📝 Found the following ID replacements:")
                for old_id, new_id in id_replacements.items():
                    logging.info(f"  {old_id} → {new_id}")

                if args.yes or args.force or input("\n🔄 Would you like to update cross-references with these new IDs? [Y/n]: ").lower() != 'n':
                    logging.info("\n🔍 Searching for cross-references...")
                    # Update all files in the directory
                    for filepath in glob.glob(os.path.join(args.directory, "**/*.qmd"), recursive=True):
                        update_cross_references(filepath, id_replacements)

                    # Update all quiz JSON files in the directory
                    logging.info("\n📝 Updating quiz JSON files...")
                    for quiz_file in glob.glob(os.path.join(args.directory, "**/*_quizzes.json"), recursive=True):
                        update_cross_references(quiz_file, id_replacements)
        else:
            parser.error("Either --file or --directory is required")

if __name__ == "__main__":
    main()
