#!/usr/bin/env python3
"""
section_splitter.py

Splits .qmd chapter files into individual sections for processing.
Designed to support section-by-section editorial workflows where each
section needs to be processed independently (e.g., by stylist agent).

Key Features:
- Uses pypandoc JSON AST for robust parsing (handles code blocks, callouts correctly)
- Extracts sections based on ## headers (level 2)
- Preserves YAML frontmatter separately
- Tracks section metadata (line numbers, word counts)
- Supports both extraction (to files) and in-memory operation
- Can reassemble sections back into complete chapter

Usage:
    # List sections in a chapter
    python3 section_splitter.py -f path/to/chapter.qmd --list

    # Extract sections to individual files
    python3 section_splitter.py -f path/to/chapter.qmd --extract --output-dir ./sections/

    # Get JSON manifest of sections (for programmatic use)
    python3 section_splitter.py -f path/to/chapter.qmd --manifest

Requirements:
    - pypandoc (pip install pypandoc)
    - pandoc must be installed
"""

import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False


@dataclass
class Section:
    """Represents a single section of a chapter."""
    index: int
    title: str
    section_id: Optional[str]
    level: int  # Number of # symbols (2 for ##, 3 for ###)
    start_line: int
    end_line: int
    word_count: int
    content: str
    is_unnumbered: bool = False  # For {.unnumbered} sections like Purpose

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding content for manifest)."""
        d = asdict(self)
        d.pop('content')  # Don't include full content in manifest
        return d


@dataclass
class ChapterStructure:
    """Complete structure of a chapter."""
    file_path: str
    chapter_title: str
    chapter_id: Optional[str]
    frontmatter: str  # YAML frontmatter
    pre_content: str  # Content before first ## section (includes # title)
    sections: list[Section]
    post_content: str  # Any content after last section (rare)
    total_lines: int
    total_words: int


def count_words(text: str) -> int:
    """Count words in text, excluding code blocks and TikZ."""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove TikZ blocks
    text = re.sub(r'\{\.tikz\}[\s\S]*?(?=\n##|\n#|\Z)', '', text)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Count remaining words
    words = text.split()
    return len(words)


def parse_header(line: str) -> tuple[int, str, Optional[str], bool]:
    """
    Parse a markdown header line.

    Returns: (level, title, section_id, is_unnumbered)
    """
    match = re.match(r'^(#{1,6})\s+(.+?)(?:\s*\{([^}]+)\})?\s*$', line)
    if not match:
        return (0, '', None, False)

    level = len(match.group(1))
    title = match.group(2).strip()
    attributes = match.group(3) or ''

    # Extract section ID
    section_id = None
    id_match = re.search(r'#(sec-[^\s}]+)', attributes)
    if id_match:
        section_id = id_match.group(1)

    # Check if unnumbered
    is_unnumbered = '.unnumbered' in attributes

    return (level, title, section_id, is_unnumbered)


def extract_text_from_inlines(inlines: list) -> str:
    """Extract plain text from pandoc inline elements."""
    text_parts = []
    for inline in inlines:
        if isinstance(inline, dict):
            t = inline.get('t', '')
            if t == 'Str':
                text_parts.append(inline.get('c', ''))
            elif t == 'Space':
                text_parts.append(' ')
            elif t in ('Emph', 'Strong', 'Strikeout', 'Superscript', 'Subscript', 'SmallCaps'):
                text_parts.append(extract_text_from_inlines(inline.get('c', [])))
            elif t == 'Link':
                # Link: [attr, inlines, target]
                text_parts.append(extract_text_from_inlines(inline.get('c', [None, [], None])[1]))
            elif t == 'Quoted':
                text_parts.append(extract_text_from_inlines(inline.get('c', [None, []])[1]))
        elif isinstance(inline, str):
            text_parts.append(inline)
    return ''.join(text_parts)


def get_section_headers_from_ast(content: str) -> list[dict]:
    """
    Use pypandoc to parse the document and extract real section headers.

    This properly handles headers inside code blocks, callouts, etc.

    Args:
        content: The markdown content

    Returns:
        List of dicts with 'title', 'id', 'level', 'line_hint' (approx line)
    """
    if not PYPANDOC_AVAILABLE:
        return []

    try:
        ast_json = pypandoc.convert_text(
            content,
            'json',
            format='markdown+smart',
            extra_args=['--preserve-tabs']
        )
        ast = json.loads(ast_json)

        headers = []

        def walk_ast(element):
            if isinstance(element, dict):
                element_type = element.get('t', '')

                if element_type == 'Header':
                    # Header: [level, [id, classes, attrs], inlines]
                    c = element.get('c', [])
                    if len(c) >= 3:
                        level = c[0]
                        header_id = c[1][0] if c[1] else None
                        inlines = c[2]
                        title = extract_text_from_inlines(inlines)

                        headers.append({
                            'level': level,
                            'id': header_id,
                            'title': title
                        })

                # Recurse into content
                for key in ('c', 'content'):
                    if key in element:
                        walk_ast(element[key])

            elif isinstance(element, list):
                for item in element:
                    walk_ast(item)

        walk_ast(ast.get('blocks', []))
        return headers

    except Exception as e:
        print(f"Warning: pypandoc parsing failed: {e}", file=__import__('sys').stderr)
        return []


def is_real_section_header(line: str, in_code_block: bool, in_callout: bool) -> bool:
    """
    Determine if a line is a real section header (not inside code/callout).

    This is the fallback method when pypandoc is not available.

    Args:
        line: The line to check
        in_code_block: Whether we're currently inside a code block
        in_callout: Whether we're currently inside a callout

    Returns:
        True if this is a real ## section header
    """
    if not line.startswith('## '):
        return False

    # Skip if inside code block or callout
    if in_code_block or in_callout:
        return False

    # Must have proper header format (## followed by text)
    # and should have a section ID {#sec-...} for real sections
    # (though Purpose section may not have one)
    return True


def split_chapter(file_path: str) -> ChapterStructure:
    """
    Split a chapter file into its component sections.

    Uses pypandoc AST parsing when available for robust handling of:
    - Code blocks (``` ... ```) - headers inside are ignored
    - Callouts (::: ... :::) - headers inside are ignored
    - TikZ blocks - headers inside are ignored

    Falls back to regex-based parsing with block tracking if pypandoc unavailable.

    Args:
        file_path: Path to the .qmd file

    Returns:
        ChapterStructure with all sections parsed
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    total_lines = len(lines)

    # Extract YAML frontmatter
    frontmatter = ''
    content_start = 0
    if lines[0].strip() == '---':
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                frontmatter = '\n'.join(lines[:i+1])
                content_start = i + 1
                break

    # Try to get headers from pypandoc AST (most reliable)
    ast_headers = get_section_headers_from_ast(content)

    # Build a set of valid section header titles from AST
    # This tells us which ## lines are REAL headers (not in code/callouts)
    valid_section_titles = set()
    chapter_title_from_ast = ''
    chapter_id_from_ast = None

    for h in ast_headers:
        if h['level'] == 1 and not chapter_title_from_ast:
            chapter_title_from_ast = h['title']
            chapter_id_from_ast = h['id']
        elif h['level'] == 2:
            valid_section_titles.add(h['title'])

    # Find chapter title (# header) and track sections
    chapter_title = chapter_title_from_ast
    chapter_id = chapter_id_from_ast
    pre_content_lines = []
    first_section_line = None

    # Track block states (fallback if AST not available)
    in_code_block = False
    in_callout_depth = 0

    for i, line in enumerate(lines[content_start:], content_start):
        stripped = line.strip()

        # Track code block state (``` or ```python, ```{.tikz}, etc.)
        if stripped.startswith('```'):
            in_code_block = not in_code_block

        # Track callout state (::: {.callout-...} or just :::)
        if not in_code_block:
            if stripped.startswith(':::') and ('{' in stripped or stripped == ':::'):
                if stripped == ':::':
                    if in_callout_depth > 0:
                        in_callout_depth -= 1
                else:
                    in_callout_depth += 1

        # Check for chapter title (if AST didn't find one)
        if not chapter_title and line.startswith('# ') and not line.startswith('## '):
            if not in_code_block and in_callout_depth == 0:
                level, title, sec_id, _ = parse_header(line)
                if level == 1:
                    chapter_title = title
                    chapter_id = sec_id

        # Check for first real section header
        elif line.startswith('## '):
            _, title, _, _ = parse_header(line)
            # Use AST validation if available, otherwise use block tracking
            if valid_section_titles:
                is_real = title in valid_section_titles
            else:
                is_real = not in_code_block and in_callout_depth == 0

            if is_real:
                first_section_line = i
                break

        pre_content_lines.append(line)

    pre_content = '\n'.join(pre_content_lines)

    # Reset block tracking for section parsing
    in_code_block = False
    in_callout_depth = 0

    # Parse sections (## level)
    sections = []
    current_section_start = first_section_line
    current_section_lines = []
    current_title = ''
    current_id = None
    current_is_unnumbered = False
    section_index = 0

    if first_section_line is not None:
        for i, line in enumerate(lines[first_section_line:], first_section_line):
            stripped = line.strip()

            # Track code block state
            if stripped.startswith('```'):
                in_code_block = not in_code_block

            # Track callout state
            if not in_code_block:
                if stripped.startswith(':::') and ('{' in stripped or stripped == ':::'):
                    if stripped == ':::':
                        if in_callout_depth > 0:
                            in_callout_depth -= 1
                    else:
                        in_callout_depth += 1

            # Check if this is a real section header
            is_section_header = False
            if line.startswith('## ') and i > first_section_line:
                _, title, _, _ = parse_header(line)
                if valid_section_titles:
                    is_section_header = title in valid_section_titles
                else:
                    is_section_header = not in_code_block and in_callout_depth == 0

            if is_section_header:
                # Save previous section
                if current_section_lines:
                    section_content = '\n'.join(current_section_lines)
                    sections.append(Section(
                        index=section_index,
                        title=current_title,
                        section_id=current_id,
                        level=2,
                        start_line=current_section_start + 1,  # 1-indexed
                        end_line=i,  # Line before new section
                        word_count=count_words(section_content),
                        content=section_content,
                        is_unnumbered=current_is_unnumbered
                    ))
                    section_index += 1

                # Start new section
                current_section_start = i
                current_section_lines = [line]
                _, current_title, current_id, current_is_unnumbered = parse_header(line)
            else:
                current_section_lines.append(line)

        # Don't forget the last section
        if current_section_lines:
            section_content = '\n'.join(current_section_lines)
            sections.append(Section(
                index=section_index,
                title=current_title,
                section_id=current_id,
                level=2,
                start_line=current_section_start + 1,
                end_line=total_lines,
                word_count=count_words(section_content),
                content=section_content,
                is_unnumbered=current_is_unnumbered
            ))

    # Calculate totals
    total_words = count_words(content)

    return ChapterStructure(
        file_path=str(file_path),
        chapter_title=chapter_title,
        chapter_id=chapter_id,
        frontmatter=frontmatter,
        pre_content=pre_content,
        sections=sections,
        post_content='',  # Typically empty
        total_lines=total_lines,
        total_words=total_words
    )


def extract_sections(chapter: ChapterStructure, output_dir: str) -> list[str]:
    """
    Extract each section to its own file.

    Args:
        chapter: Parsed chapter structure
        output_dir: Directory to write section files

    Returns:
        List of created file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Write frontmatter + pre-content as section 0
    pre_path = os.path.join(output_dir, 'section_00_preamble.qmd')
    with open(pre_path, 'w', encoding='utf-8') as f:
        f.write(chapter.frontmatter + '\n' + chapter.pre_content)
    created_files.append(pre_path)

    # Write each section
    for section in chapter.sections:
        # Create safe filename from title
        safe_title = re.sub(r'[^\w\s-]', '', section.title.lower())
        safe_title = re.sub(r'\s+', '_', safe_title)[:40]
        filename = f'section_{section.index + 1:02d}_{safe_title}.qmd'

        section_path = os.path.join(output_dir, filename)
        with open(section_path, 'w', encoding='utf-8') as f:
            f.write(section.content)
        created_files.append(section_path)

    return created_files


def reassemble_chapter(chapter: ChapterStructure, modified_sections: Optional[dict[int, str]] = None) -> str:
    """
    Reassemble a chapter from its components.

    Args:
        chapter: Original chapter structure
        modified_sections: Optional dict mapping section index to new content

    Returns:
        Complete chapter content
    """
    parts = []

    # Add frontmatter
    if chapter.frontmatter:
        parts.append(chapter.frontmatter)

    # Add pre-content (includes # title)
    if chapter.pre_content:
        parts.append(chapter.pre_content)

    # Add sections (possibly modified)
    for section in chapter.sections:
        if modified_sections and section.index in modified_sections:
            parts.append(modified_sections[section.index])
        else:
            parts.append(section.content)

    return '\n'.join(parts)


def generate_manifest(chapter: ChapterStructure) -> dict:
    """
    Generate a JSON manifest of the chapter structure.

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        'file_path': chapter.file_path,
        'chapter_title': chapter.chapter_title,
        'chapter_id': chapter.chapter_id,
        'total_sections': len(chapter.sections),
        'total_lines': chapter.total_lines,
        'total_words': chapter.total_words,
        'sections': [s.to_dict() for s in chapter.sections]
    }


def list_sections(chapter: ChapterStructure) -> None:
    """Print a formatted list of sections."""
    print(f"\nChapter: {chapter.chapter_title}")
    print(f"File: {chapter.file_path}")
    print(f"Total: {len(chapter.sections)} sections, {chapter.total_words:,} words, {chapter.total_lines:,} lines")
    print("-" * 80)
    print(f"{'#':<3} {'Lines':<12} {'Words':<8} {'ID':<40} Title")
    print("-" * 80)

    for section in chapter.sections:
        line_range = f"{section.start_line}-{section.end_line}"
        sec_id = section.section_id or "(none)"
        unnumbered = " [unnumbered]" if section.is_unnumbered else ""
        print(f"{section.index + 1:<3} {line_range:<12} {section.word_count:<8} {sec_id:<40} {section.title}{unnumbered}")


def main():
    parser = argparse.ArgumentParser(
        description="Split .qmd chapter files into sections for processing"
    )
    parser.add_argument('-f', '--file', required=True,
                        help='Path to the .qmd chapter file')

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true',
                        help='List all sections in the chapter')
    action.add_argument('--extract', action='store_true',
                        help='Extract sections to individual files')
    action.add_argument('--manifest', action='store_true',
                        help='Output JSON manifest of chapter structure')
    action.add_argument('--get-section', type=int, metavar='N',
                        help='Get content of section N (1-indexed)')

    parser.add_argument('--output-dir', default='./sections',
                        help='Directory for extracted sections (default: ./sections)')

    args = parser.parse_args()

    # Parse the chapter
    chapter = split_chapter(args.file)

    if args.list:
        list_sections(chapter)

    elif args.extract:
        files = extract_sections(chapter, args.output_dir)
        print(f"Extracted {len(files)} files to {args.output_dir}/")
        for f in files:
            print(f"  {f}")

    elif args.manifest:
        manifest = generate_manifest(chapter)
        print(json.dumps(manifest, indent=2))

    elif args.get_section is not None:
        idx = args.get_section - 1  # Convert to 0-indexed
        if 0 <= idx < len(chapter.sections):
            print(chapter.sections[idx].content)
        else:
            print(f"Error: Section {args.get_section} not found. Chapter has {len(chapter.sections)} sections.")
            exit(1)


if __name__ == "__main__":
    main()
