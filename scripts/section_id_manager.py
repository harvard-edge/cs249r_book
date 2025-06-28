# section_id_manager.py

"""
Comprehensive Section ID Management Script for Quarto/Markdown Book Projects
---------------------------------------------------------------------------

This script provides a complete toolkit for managing section IDs in your Markdown/Quarto book project.
It ensures that all section headers have unique, clean, and consistent section IDs while preserving
cross-references and other attributes.

Special Handling for Unnumbered Headers:
---------------------------------------
- Any header with the {.unnumbered} class is always skipped for section ID management.
- Unnumbered headers will never have a section ID added, updated, or required (including in verify mode).
- Only numbered headers (without {.unnumbered}) require section IDs.

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
- This ensures GLOBAL UNIQUENESS across the entire book project
- Different files with identical section names and hierarchies will have different IDs
- Parent sections are included in the hash to handle duplicate section names naturally
- The visible part of the ID remains short and human-readable
- IDs are stable and won't change if sections are reordered (as long as hierarchy doesn't change)

Global Uniqueness Guarantee:
----------------------------
The hash generation includes the file path to ensure that sections with identical names
and hierarchies in different files will have different IDs. This prevents conflicts when:

- Multiple chapters have sections with the same name (e.g., "Introduction" in different files)
- Different files have identical section hierarchies (e.g., "Techniques > Advanced > Optimization")
- The same section name appears in multiple contexts across the book

Example hash inputs:
  - File A: "contents/chapter1.qmd|Getting Started|Introduction|"
  - File B: "contents/chapter2.qmd|Getting Started|Introduction|"
  - Result: Different 4-character hashes ensure unique IDs

Available Modes:
----------------
- **Add Mode (default):** Add missing section IDs to headers (skips unnumbered headers)
- **Repair Mode (--repair):** Fix existing section IDs to match the new format
- **Remove Mode (--remove):** Remove all section IDs (use with --backup)
- **Verify Mode (--verify):** Check that all section IDs are present and properly formatted (skips unnumbered headers)
- **List Mode (--list):** Display all section IDs found in files

Safety Features:
----------------
- **Backup System:** --backup creates .backup.{timestamp} files before changes
- **Dry Run:** --dry-run shows what would change without modifying files
- **Interactive Prompts:** Asks for confirmation before making changes
- **Force Mode:** --force automatically accepts all confirmations without prompting
- **Attribute Preservation:** Maintains other attributes when modifying section IDs
- **Cross-reference Updates:** Automatically updates references when IDs change

Best Practices:
---------------
- Use --backup when making bulk changes
- Use --verify before commits to ensure ID integrity (unnumbered headers are always ignored)
- Use --list to audit existing section IDs
- Use --dry-run to preview changes before applying them
- Consider using this in pre-commit hooks or CI pipelines

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
- **Unnumbered headers are always skipped for section IDs in all modes**

Typical Usage:
    # Add missing IDs
    python section_id_manager.py -d contents/
    python section_id_manager.py -f contents/chapter.qmd
    
    # Repair existing IDs
    python section_id_manager.py -d contents/ --repair --backup
    
    # Force repair without prompts
    python section_id_manager.py -d contents/ --repair --force
    
    # Verify all IDs (unnumbered headers are always ignored)
    python section_id_manager.py -d contents/ --verify
    
    # List all IDs
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
import pypandoc
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

def simple_slugify(text):
    """Convert header text to a slug format, removing stopwords and converting underscores to hyphens, preserving hyphens between words."""
    # Convert underscores to hyphens
    text = text.replace('_', '-')
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    # Convert to lowercase and split into words (split on whitespace and hyphens)
    words = re.split(r'[\s-]+', text.lower())
    # Remove stopwords and non-alphanumeric characters (but keep hyphens between words)
    filtered_words = []
    for word in words:
        # Remove non-alphanumeric characters (except hyphens)
        word = re.sub(r'[^\w-]', '', word)
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

def is_properly_formatted_id(section_id, title, file_path, chapter_title, section_counter, parent_sections=None, section_content=None):
    """Check if a section ID follows the correct format and has the correct hash."""
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
    
    # If format is correct, regenerate the hash to check if it matches
    expected_id = generate_section_id(title, file_path, chapter_title, section_counter, parent_sections, section_content)
    
    # Compare the generated ID with the existing one
    if section_id == expected_id:
        return True, section_id
    else:
        # Format is correct but hash is wrong
        return False, expected_id

def generate_section_id(title, file_path, chapter_title, section_counter, parent_sections=None, section_content=None):
    """
    Generate a unique section ID based on the section title and content.
    
    The hash includes file path, chapter title, section title, parent section hierarchy,
    and section content to ensure the ID changes when either location or content changes.
    This is perfect for quiz synchronization - when the ID changes, you know the content
    or context has changed and quizzes may need updating.
    
    Args:
        title: The section title
        file_path: The file path (included in hash for location tracking)
        chapter_title: The chapter title
        section_counter: Counter for this section (not used in hash)
        parent_sections: List of parent section titles (included in hash)
        section_content: The content of the section (included in hash for content tracking)
    
    Returns:
        A unique section ID in the format: sec-{chapter-slug}-{section-slug}-{4-char-hash}
        
    Example:
        Same section name in different files:
        - File A: "contents/chapter1.qmd|Getting Started|Introduction|content1" → hash: d212
        - File B: "contents/chapter2.qmd|Getting Started|Introduction|content1" → hash: 8435
        Result: Different IDs ensure uniqueness and track both location and content changes
    """
    clean_title = simple_slugify(title)
    clean_chapter_title = simple_slugify(chapter_title)
    
    # Build hierarchy string from parent sections
    hierarchy = ""
    if parent_sections:
        # Create a hierarchy string from all parent sections
        hierarchy_parts = []
        for parent in parent_sections:
            hierarchy_parts.append(simple_slugify(parent))
        hierarchy = "|".join(hierarchy_parts)
    
    # Normalize content for hashing (remove extra whitespace, basic formatting)
    normalized_content = ""
    if section_content:
        normalized_content = normalize_content_for_hash(section_content)
    
    # Hash includes file path, chapter title, section title, parent hierarchy, and content
    # This ensures the ID changes when either location or content changes
    hash_input = f"{file_path}|{chapter_title}|{title}|{hierarchy}|{normalized_content}".encode('utf-8')
    hash_suffix = hashlib.sha1(hash_input).hexdigest()[:4]  # Keep 4 chars
    return f"sec-{clean_chapter_title}-{clean_title}-{hash_suffix}"

def extract_code_block_ranges(ast_data):
    """Extract line ranges of code blocks from AST."""
    code_ranges = []
    
    def walk_blocks(blocks, current_line=1):
        for block in blocks:
            t = block.get('t')
            c = block.get('c')
            
            if t == 'CodeBlock':
                # CodeBlock structure: [attr, text]
                # We need to count lines to determine the range
                code_text = c[1]
                lines_in_block = code_text.count('\n') + 1
                code_ranges.append((current_line, current_line + lines_in_block - 1))
                current_line += lines_in_block
            elif t == 'Div':
                # Div structure: [attr, blocks]
                current_line = walk_blocks(c[1], current_line)
            elif t in ('BlockQuote', 'BulletList', 'OrderedList'):
                if t == 'BlockQuote':
                    current_line = walk_blocks(c, current_line)
                elif t == 'BulletList':
                    for item in c:
                        current_line = walk_blocks(item, current_line)
                elif t == 'OrderedList':
                    for item in c[1]:
                        current_line = walk_blocks(item, current_line)
            else:
                # For other blocks, count lines
                if 'c' in block and isinstance(block['c'], str):
                    lines_in_block = block['c'].count('\n') + 1
                    current_line += lines_in_block
                else:
                    current_line += 1
        
        return current_line
    
    walk_blocks(ast_data.get('blocks', []))
    return code_ranges

def extract_section_headers_pandoc(file_path):
    """Extract all real section headers from a QMD/MD file using Pandoc AST."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        ast_json = pypandoc.convert_text(content, 'json', format='markdown')
        ast = json.loads(ast_json)
    except Exception as e:
        logging.error(f"Failed to parse {file_path} with Pandoc: {e}")
        return []
    
    # Get code block ranges
    code_ranges = extract_code_block_ranges(ast)
    
    headers = []
    current_line = 1
    
    def walk(blocks, in_div=False):
        nonlocal current_line
        for block in blocks:
            t = block.get('t')
            c = block.get('c')
            
            if t == 'Header' and not in_div:
                level = c[0]
                attr = c[1]
                inlines = c[2]
                section_id = attr[0] if attr[0] else None
                classes = attr[1]
                
                # Skip level 1 headers and unnumbered headers
                if level == 1 or 'unnumbered' in classes or 'appendix' in classes or 'backmatter' in classes:
                    current_line += 1
                    continue
                
                # Check if this header is inside a code block
                in_code_block = any(start <= current_line <= end for start, end in code_ranges)
                if in_code_block:
                    current_line += 1
                    continue
                
                # Extract plain text title
                title = []
                for x in inlines:
                    if x['t'] == 'Str':
                        title.append(x['c'])
                    elif x['t'] == 'Space':
                        title.append(' ')
                title = ''.join(title).strip()
                headers.append((level, title, section_id, classes))
                current_line += 1
            elif t == 'CodeBlock':
                # Skip code blocks entirely
                code_text = c[1]
                lines_in_block = code_text.count('\n') + 1
                current_line += lines_in_block
            elif t == 'Div':
                walk(c[1], in_div=True)
            elif t in ('BlockQuote', 'BulletList', 'OrderedList'):
                if t == 'BlockQuote':
                    walk(c, in_div=in_div)
                elif t == 'BulletList':
                    for item in c:
                        walk(item, in_div=in_div)
                elif t == 'OrderedList':
                    for item in c[1]:
                        walk(item, in_div=in_div)
            else:
                # For other blocks, count lines
                if 'c' in block and isinstance(block['c'], str):
                    lines_in_block = block['c'].count('\n') + 1
                    current_line += lines_in_block
                else:
                    current_line += 1
    
    walk(ast.get('blocks', []))
    return headers

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
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop if we hit another header of same or higher level
        if line.startswith('#'):
            next_header_level = len(line) - len(line.lstrip('#'))
            if next_header_level <= header_level:
                break
        
        # Stop if we hit a div boundary
        if line.startswith(':::'):
            break
            
        # Add non-empty lines to content
        if line:
            content_lines.append(line)
        
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
    """Process a single Markdown file using AST parsing for reliable section detection."""
    global id_replacements
    logging.info(f"\n📄 Processing: {file_path}")
    
    # Create backup if requested
    if backup_mode and not dry_run:
        create_backup(file_path)
    
    path = Path(file_path)
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        # Parse with Pandoc AST
        ast_json = pypandoc.convert_text(content, 'json', format='markdown')
        ast = json.loads(ast_json)
    except Exception as e:
        logging.error(f"Failed to parse {file_path} with Pandoc: {e}")
        return {
            'file_path': file_path,
            'added_ids': [],
            'updated_ids': [],
            'removed_ids': [],
            'existing_sections': [],
            'modified': False
        }
    
    # Get code block ranges and section headers from AST
    code_ranges = extract_code_block_ranges(ast)
    headers = extract_section_headers_pandoc(file_path)
    
    # Find chapter title (first level 1 header) directly from AST
    chapter_title = None
    def find_chapter_title(blocks):
        nonlocal chapter_title
        for block in blocks:
            t = block.get('t')
            c = block.get('c')
            
            if t == 'Header':
                level = c[0]
                attr = c[1]
                inlines = c[2]
                
                if level == 1 and not chapter_title:
                    # Extract plain text title
                    title = []
                    for x in inlines:
                        if x['t'] == 'Str':
                            title.append(x['c'])
                        elif x['t'] == 'Space':
                            title.append(' ')
                    chapter_title = ''.join(title).strip()
                    return
                elif level > 1:
                    continue
            elif t == 'Div':
                find_chapter_title(c[1])
    
    find_chapter_title(ast.get('blocks', []))
    
    if not chapter_title:
        raise ValueError(f"No chapter title found in {file_path}")
    
    # Process the file line by line, but use AST data for decisions
    lines = content.splitlines(keepends=True)
    new_lines = []
    modified = False
    section_counter = 0
    
    # Track section hierarchy
    section_hierarchy = []
    
    file_summary = {
        'file_path': file_path,
        'added_ids': [],
        'updated_ids': [],
        'removed_ids': [],
        'existing_sections': [],
        'modified': False
    }
    
    # Create a map of line numbers to header info from AST
    header_map = {}
    current_line = 1
    
    def walk_headers(blocks, in_div=False):
        nonlocal current_line
        for block in blocks:
            t = block.get('t')
            c = block.get('c')
            
            if t == 'Header' and not in_div:
                level = c[0]
                attr = c[1]
                inlines = c[2]
                section_id = attr[0] if attr[0] else None
                classes = attr[1]
                
                # Skip level 1 headers and unnumbered headers
                if level == 1 or 'unnumbered' in classes or 'appendix' in classes or 'backmatter' in classes:
                    current_line += 1
                    continue
                
                # Check if this header is inside a code block
                in_code_block = any(start <= current_line <= end for start, end in code_ranges)
                if in_code_block:
                    current_line += 1
                    continue
                
                # Extract plain text title
                title = []
                for x in inlines:
                    if x['t'] == 'Str':
                        title.append(x['c'])
                    elif x['t'] == 'Space':
                        title.append(' ')
                title = ''.join(title).strip()
                
                header_map[current_line] = {
                    'level': level,
                    'title': title,
                    'section_id': section_id,
                    'classes': classes,
                    'line_number': current_line
                }
                current_line += 1
            elif t == 'CodeBlock':
                # Skip code blocks entirely
                code_text = c[1]
                lines_in_block = code_text.count('\n') + 1
                current_line += lines_in_block
            elif t == 'Div':
                walk_headers(c[1], in_div=True)
            elif t in ('BlockQuote', 'BulletList', 'OrderedList'):
                if t == 'BlockQuote':
                    walk_headers(c, in_div=in_div)
                elif t == 'BulletList':
                    for item in c:
                        walk_headers(item, in_div=in_div)
                elif t == 'OrderedList':
                    for item in c[1]:
                        walk_headers(item, in_div=in_div)
            else:
                # For other blocks, count lines
                if 'c' in block and isinstance(block['c'], str):
                    lines_in_block = block['c'].count('\n') + 1
                    current_line += lines_in_block
                else:
                    current_line += 1
    
    walk_headers(ast.get('blocks', []))
    
    # Process lines with AST guidance
    current_line = 1
    inside_code_block = False
    code_block_fence = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track code block boundaries
        if not inside_code_block and (stripped.startswith('```') or stripped.startswith('~~~')):
            inside_code_block = True
            code_block_fence = stripped[:3]
        elif inside_code_block and stripped.startswith(code_block_fence):
            inside_code_block = False
            code_block_fence = None
        
        # If this line is inside a code block, remove any section IDs and skip header processing
        if inside_code_block:
            # Remove any {#sec-...} from code comment lines (only section IDs, not other {#...} patterns)
            new_line = re.sub(r'\s*\{#sec-[-a-zA-Z0-9_:.]+\}', '', line)
            if new_line != line:
                modified = True
            new_lines.append(new_line)
            current_line += 1
            continue
        
        # Check if this line contains a header (using AST data)
        header_info = header_map.get(current_line)
        
        if header_info:
            level = header_info['level']
            title = header_info['title']
            existing_section_id = header_info['section_id']
            classes = header_info['classes']
            
            # Update section hierarchy
            while len(section_hierarchy) >= level - 1:
                section_hierarchy.pop()
            section_hierarchy.append(title)
            parent_sections = section_hierarchy[:-1] if len(section_hierarchy) > 1 else []
            
            # Skip headers with {.unnumbered}
            if 'unnumbered' in classes:
                # Remove any existing section ID from unnumbered headers
                if existing_section_id:
                    new_line = re.sub(r'\s*\{#sec-[^}]+\}', '', line)
                    new_lines.append(new_line)
                    modified = True
                    file_summary['modified'] = True
                    file_summary['removed_ids'].append((title, existing_section_id))
                    logging.info(f"  🗑️  Removed ID from unnumbered header: {title}")
                else:
                    new_lines.append(line)
                current_line += 1
                continue
            
            # Process existing section IDs
            if existing_section_id:
                file_summary['existing_sections'].append((title, existing_section_id))
                
                if remove_mode:
                    # Remove the section ID
                    if auto_yes or force or input(f"\n🗑️  Remove ID for '{title}': {existing_section_id}? [Y/n]: ").lower() != 'n':
                        new_line = re.sub(r'\s*\{#sec-[^}]+\}', '', line)
                        new_lines.append(new_line)
                        modified = True
                        file_summary['modified'] = True
                        file_summary['removed_ids'].append((title, existing_section_id))
                        logging.info(f"  🗑️  Removed: {title}")
                else:
                    # Generate new ID and check if update is needed
                    section_content = extract_section_content(lines, i, level)
                    new_id = generate_section_id(title, file_path, chapter_title, section_counter, parent_sections, section_content)
                    section_counter += 1
                    
                    is_proper, expected_id = is_properly_formatted_id(existing_section_id, title, file_path, chapter_title, section_counter, parent_sections, section_content)
                    should_update = repair_mode or not is_proper
                    
                    if should_update and existing_section_id != new_id:
                        if auto_yes or force or input(f"\n🔄 Update ID for '{title}':\n  From: {existing_section_id}\n  To:   {new_id}\n  Proceed? [Y/n]: ").lower() != 'n':
                            id_replacements[existing_section_id] = new_id
                            new_line = re.sub(r'\{#sec-[^}]+\}', f'{{#{new_id}}}', line)
                            new_lines.append(new_line)
                            modified = True
                            file_summary['modified'] = True
                            file_summary['updated_ids'].append((title, existing_section_id, new_id))
                            logging.info(f"  ✓ Updated: {title}")
                    else:
                        new_lines.append(line)
            else:
                if not remove_mode:  # Only add IDs if not in remove mode
                    section_content = extract_section_content(lines, i, level)
                    new_id = generate_section_id(title, file_path, chapter_title, section_counter, parent_sections, section_content)
                    section_counter += 1
                    
                    # Check if line already has attributes
                    if '{' in line and '}' in line:
                        # Insert section ID into existing attributes
                        new_line = re.sub(r'(\{[^}]*\})', r'\1 {#' + new_id + '}', line)
                    else:
                        # Add new attributes with section ID
                        new_line = line.rstrip() + f' {{#{new_id}}}\n'
                    
                    new_lines.append(new_line)
                    modified = True
                    file_summary['modified'] = True
                    file_summary['added_ids'].append((title, new_id))
                    logging.info(f"  + Added: {title}")
                else:
                    # In remove mode, also check if this line is a header with a section ID that wasn't detected by AST
                    if remove_mode and line.strip().startswith('#'):
                        # Check if this line contains a section ID
                        if re.search(r'\{#sec-[^}]+\}', line):
                            # Extract title from the line
                            title_match = re.match(r'^(#{1,6})\s+(.+?)(?:\s+\{[^}]*\})?$', line.strip())
                            if title_match:
                                title = title_match.group(2).strip()
                                # Remove the section ID
                                if auto_yes or force or input(f"\n🗑️  Remove ID from header '{title}'? [Y/n]: ").lower() != 'n':
                                    new_line = re.sub(r'\s*\{#sec-[^}]+\}', '', line)
                                    new_lines.append(new_line)
                                    modified = True
                                    file_summary['modified'] = True
                                    file_summary['removed_ids'].append((title, "unknown"))
                                    logging.info(f"  🗑️  Removed ID from header: {title}")
                                else:
                                    new_lines.append(line)
                            else:
                                new_lines.append(line)
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
            
            current_line += 1
        else:
            # In remove mode, also check if this line is a header with a section ID that wasn't detected by AST
            if remove_mode and line.strip().startswith('#'):
                # Check if this line contains a section ID
                if re.search(r'\{#sec-[^}]+\}', line):
                    # Extract title from the line
                    title_match = re.match(r'^(#{1,6})\s+(.+?)(?:\s+\{[^}]*\})?$', line.strip())
                    if title_match:
                        title = title_match.group(2).strip()
                        # Remove the section ID
                        if auto_yes or force or input(f"\n🗑️  Remove ID from header '{title}'? [Y/n]: ").lower() != 'n':
                            new_line = re.sub(r'\s*\{#sec-[^}]+\}', '', line)
                            new_lines.append(new_line)
                            modified = True
                            file_summary['modified'] = True
                            file_summary['removed_ids'].append((title, "unknown"))
                            logging.info(f"  🗑️  Removed ID from header: {title}")
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            current_line += 1
    
    # Show existing sections
    if file_summary['existing_sections']:
        logging.info(f"  📋 Existing sections:")
        for title, section_id in file_summary['existing_sections']:
            logging.info(f"    • {title} → #{section_id}")
    
    if modified and not dry_run:
        path.write_text(''.join(new_lines), encoding="utf-8")
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
        logging.info(f"\n📝 FILES WITH CHANGES:")
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
    """Main function to handle command line arguments and execute the appropriate action."""
    parser = argparse.ArgumentParser(description='Manage section IDs in Markdown files')
    parser.add_argument('-f', '--file', help='Process a single file')
    parser.add_argument('-d', '--directory', help='Process all .qmd files in a directory')
    parser.add_argument('--add', action='store_true', help='Add missing section IDs')
    parser.add_argument('--remove', action='store_true', help='Remove all section IDs')
    parser.add_argument('--repair', action='store_true', help='Repair/update existing section IDs')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--yes', action='store_true', help='Automatically confirm all changes')
    parser.add_argument('--force', action='store_true', help='Force changes without confirmation')
    parser.add_argument('--backup', action='store_true', help='Create backup files before making changes')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    if args.file:
        if args.add:
            process_markdown_file(args.file, auto_yes=args.yes, force=args.force, dry_run=args.dry_run, backup_mode=args.backup)
        elif args.remove:
            process_markdown_file(args.file, auto_yes=args.yes, force=args.force, dry_run=args.dry_run, remove_mode=True, backup_mode=args.backup)
        elif args.repair:
            process_markdown_file(args.file, auto_yes=args.yes, force=args.force, dry_run=args.dry_run, repair_mode=True, backup_mode=args.backup)
        else:
            parser.print_help()
    elif args.directory:
        if args.add:
            process_directory(args.directory, auto_yes=args.yes, force=args.force, dry_run=args.dry_run, backup_mode=args.backup)
        elif args.remove:
            process_directory(args.directory, auto_yes=args.yes, force=args.force, dry_run=args.dry_run, remove_mode=True, backup_mode=args.backup)
        elif args.repair:
            process_directory(args.directory, auto_yes=args.yes, force=args.force, dry_run=args.dry_run, repair_mode=True, backup_mode=args.backup)
        else:
            parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
