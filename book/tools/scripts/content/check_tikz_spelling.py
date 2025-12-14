#!/usr/bin/env python3
r"""
Spell check text content within TikZ diagrams.

Extracts and validates all visible text from TikZ diagrams in .qmd files,
including node labels, inline annotations, formatted text, and comments.

Usage:
    python3 tools/scripts/content/check_tikz_spelling.py

Checks text in:
    - Node commands: \node{text}, node{text} in \draw/\path/\fill
    - Formatted text: \textbf{}, \textit{}, \emph{}, etc.
    - Drawing annotations: \draw--node{label}--
    - Custom pics: pics/name/, \pic{name}
    - Foreach loops: /{Text}/ patterns
    - Labels: label={text}, pin={text}
    - Legends: \legend{Item 1, Item 2}
    - Comments: % text

Optional: Install aspell for comprehensive dictionary checking
    macOS: brew install aspell
    Ubuntu: sudo apt-get install aspell
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Set
import subprocess


def extract_tikz_blocks(content: str, filepath: str) -> List[Tuple[str, int]]:
    """
    Extract TikZ code blocks with their starting line numbers.

    Returns:
        List of (tikz_content, start_line_number) tuples
    """
    blocks = []
    lines = content.split('\n')
    in_tikz = False
    current_block = []
    start_line = 0

    for i, line in enumerate(lines, 1):
        if r'\begin{tikzpicture}' in line:
            in_tikz = True
            start_line = i
            current_block = [line]
        elif r'\end{tikzpicture}' in line and in_tikz:
            current_block.append(line)
            blocks.append(('\n'.join(current_block), start_line))
            in_tikz = False
            current_block = []
        elif in_tikz:
            current_block.append(line)

    return blocks


def clean_latex_text(text: str) -> str:
    """
    Clean LaTeX formatting from text to get readable content.

    Args:
        text: Raw text from LaTeX/TikZ

    Returns:
        Cleaned text with LaTeX commands removed
    """
    # Replace \\ (line breaks) with spaces first
    text = text.replace('\\\\', ' ')

    # Remove size commands that appear before text (like {\huge ?})
    text = re.sub(r'\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\s+', ' ', text)

    # Remove font commands
    text = re.sub(r'\\usefont\{[^}]*\}\{[^}]*\}\{[^}]*\}\{[^}]*\}', ' ', text)
    text = re.sub(r'\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont', ' ', text)
    text = re.sub(r'\\bfseries\s*', ' ', text)

    # Handle nested formatting commands (multiple passes)
    for _ in range(3):  # Up to 3 levels of nesting
        # Remove common LaTeX formatting commands but keep the text
        text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\textit\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\emph\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\textsubscript\{([^}]+)\}', r'_\1', text)
        text = re.sub(r'\\textsuperscript\{([^}]+)\}', r'^\1', text)
        text = re.sub(r'\\textcolor\{[^}]*\}\{([^}]+)\}', r'\1', text)

    # Remove $ signs (math mode)
    text = text.replace('$', '')

    # Remove other common LaTeX commands (but preserve the text after them)
    text = re.sub(r'\\[a-zA-Z]+\s*', ' ', text)

    # Clean up whitespace
    text = ' '.join(text.split())

    return text.strip()


def extract_all_curly_brace_text(tikz_content: str) -> List[Tuple[str, str, int]]:
    """
    Extract all text content from curly braces that could be visible text.

    Returns:
        List of (text, context, char_position) tuples
    """
    texts = []

    # Find all text in curly braces that follows common TikZ commands or appears in node definitions
    # This catches: \node{text}, node{text}, \textbf{text}, etc.

    # Pattern 1: \node[options]{text} or \node(name){text}
    node_standalone = r'\\node\s*(?:\[[^\]]*\])?\s*(?:\([^)]*\))?\s*(?:at\s*\([^)]*\))?\s*\{([^}]+)\}'
    for match in re.finditer(node_standalone, tikz_content):
        text = match.group(1)
        texts.append((text, '\\node{...}', match.start()))

    # Pattern 2: node[options]{text} (inside \draw, \fill, or \path)
    node_inline = r'(?<!\\)node\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
    for match in re.finditer(node_inline, tikz_content):
        text = match.group(1)
        texts.append((text, 'node{...} in draw/path/fill', match.start()))

    # Pattern 3: Text formatting commands
    text_commands = [
        (r'\\textbf\{([^}]+)\}', '\\textbf{...}'),
        (r'\\textit\{([^}]+)\}', '\\textit{...}'),
        (r'\\emph\{([^}]+)\}', '\\emph{...}'),
        (r'\\text\{([^}]+)\}', '\\text{...}'),
    ]
    for pattern, context in text_commands:
        for match in re.finditer(pattern, tikz_content):
            text = match.group(1)
            texts.append((text, context, match.start()))

    # Pattern 4: label={text} and similar options
    label_pattern = r'(?:label|pin|xlabel|ylabel)\s*=\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
    for match in re.finditer(label_pattern, tikz_content):
        text = match.group(1)
        texts.append((text, 'label={...}', match.start()))

    # Pattern 5: legend command
    legend_pattern = r'\\legend\s*\{([^}]+)\}'
    for match in re.finditer(legend_pattern, tikz_content):
        text = match.group(1)
        texts.append((text, '\\legend{...}', match.start()))

    return texts


def extract_text_from_foreach(tikz_content: str) -> List[Tuple[str, str]]:
    r"""
    Extract text from \foreach loops which often contain labels.

    Pattern: \\foreach \\i/\\j/... in {val1/{Text 1}/val2, val2/{Text 2}/val3, ...}

    Returns:
        List of (text, context) tuples
    """
    texts = []

    # Find \foreach statements
    foreach_pattern = r'\\foreach[^{]+in\s*\{([^}]+)\}'

    for match in re.finditer(foreach_pattern, tikz_content, re.DOTALL):
        content = match.group(1)

        # Extract text from {...} within the foreach content
        # Pattern: /{text}/
        text_in_braces = re.findall(r'/\{([^}]+)\}/', content)
        for text in text_in_braces:
            cleaned = clean_latex_text(text)
            if cleaned and len(cleaned) > 2:
                # Skip if it's just a number or coordinate
                if not re.match(r'^[\d\s\.,\-\+]+$', cleaned):
                    texts.append((cleaned, f'\\foreach loop: /{{{text}}}/'))

    return texts


def extract_text_from_tikz(tikz_content: str) -> List[Tuple[str, str]]:
    """
    Extract ALL human-readable text from TikZ code.

    Returns:
        List of (text, context) tuples where context shows where the text was found
    """
    texts = []
    seen_texts = set()  # Avoid duplicates

    # Extract all text from curly braces
    for raw_text, context, pos in extract_all_curly_brace_text(tikz_content):
        # Clean the text
        cleaned = clean_latex_text(raw_text)

        # Skip if it's just numbers, coordinates, colors, or TikZ commands
        if not cleaned:
            continue
        if re.match(r'^[\d\s\.,\-\+\*\/\(\)_\^]+$', cleaned):  # Just numbers/math/subscripts
            continue
        if re.match(r'^[a-z]+!?\d*$', cleaned):  # Colors like "red", "blue!50"
            continue
        if len(cleaned) < 2:  # Too short to be meaningful text
            continue

        # Avoid duplicates
        key = (cleaned.lower(), context)
        if key not in seen_texts:
            seen_texts.add(key)
            texts.append((cleaned, f'{context}: "{raw_text}"'))

    # Extract text from \foreach loops
    for text, context in extract_text_from_foreach(tikz_content):
        key = (text.lower(), 'foreach')
        if key not in seen_texts:
            seen_texts.add(key)
            texts.append((text, context))

    # Extract text from pic names (custom TikZ pictures)
    pic_name_pattern = r'pics/([a-zA-Z_]+)/'
    for match in re.finditer(pic_name_pattern, tikz_content):
        name = match.group(1)
        if len(name) > 2:
            key = (name.lower(), 'pics')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((name, f'pics/{name}/'))

    # Extract text from pic usage
    pic_usage_pattern = r'\\pic\s*(?:\[[^\]]*\])?\s*(?:at\s*\([^)]*\))?\s*\{([^}]+)\}'
    for match in re.finditer(pic_usage_pattern, tikz_content):
        name = match.group(1)
        if len(name) > 2 and not re.match(r'^[\d\s]+$', name):
            key = (name.lower(), 'pic_usage')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((name, f'\\pic{{...}}{{{name}}}'))

    # Extract comments (often contain descriptive text)
    comment_pattern = r'%\s*(.+?)(?:\n|$)'
    for match in re.finditer(comment_pattern, tikz_content):
        comment = match.group(1).strip()
        # Skip comments that are just separators or structure
        if comment and not re.match(r'^[\-\=\*\s]+$', comment):
            key = (comment.lower(), 'comment')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((comment, f'% {comment}'))

    # Extract variable names from \def that might be words
    def_pattern = r'\\def\\([a-zA-Z]+)\{'
    for match in re.finditer(def_pattern, tikz_content):
        name = match.group(1)
        # Only check if it looks like a word (not all caps, reasonable length)
        if len(name) > 3 and not name.isupper() and not name.startswith('r'):
            key = (name.lower(), 'def')
            if key not in seen_texts:
                seen_texts.add(key)
                texts.append((name, f'\\def\\{name}'))

    return texts


def check_spelling_with_aspell(text: str) -> List[str]:
    """
    Check spelling using aspell if available, filtering out TikZ/LaTeX technical terms.

    Returns:
        List of misspelled words (excluding known technical terms)
    """
    # Terms to ignore (TikZ syntax, LaTeX commands, common technical terms, etc.)
    ignore_terms = {
        # TikZ pic parameters
        'scalefac', 'picname', 'filllcolor', 'drawcolor', 'linewidth',
        'filllcirclecolor', 'drawcircle', 'bodycolor', 'tiecolor', 'stetcolor',
        'drawchannelcolor', 'channelcolor',

        # Color names
        'brownline', 'redline', 'blueline', 'violetline', 'greenline', 'orangeline',
        'violetl', 'greenl', 'bluel', 'redl', 'orangel',
        'greend',

        # TikZ/LaTeX commands
        'tikzset', 'foreach', 'tikz', 'usefont', 'phv', 'bfseries', 'textbf',
        'pgfmathparse', 'addplot', 'sqrt',

        # Common variable names
        'cellsize', 'cellheight', 'xmax', 'ymin', 'newx', 'pos', 'sep',

        # Technical diagram terms
        'mycylinder', 'mycycle', 'myline', 'rgpoly', 'zerofill',

        # Display/UI elements
        'displaye', 'autotext',

        # Abbreviations used in diagrams
        'zgl', 'zgd', 'da', 'dcd', 'dcl', 'dsc', 'ggb', 'lca', 'sre',

        # Common acronyms and abbreviations
        'ui', 'kpis', 'oss', 'rtx', 'tpus', 'bg', 'eniac', 'fp',

        # Technical terms (keep legitimate ones but add clearly technical)
        'preprocessing', 'backprop', 'weightgradient', 'davit', 'tokenize',
        'multimodality', 'microarchitecture', 'hypercomputing', 'curation',
        'transformative',

        # Misc
        'helvetica', 'geeksforgeeks', 'lightgray', 'gaussian', 'yshift',
        'ack', 'zz', 'yy',
    }

    try:
        # Check if aspell is available
        result = subprocess.run(
            ['aspell', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return []
    except FileNotFoundError:
        return []

    # Use aspell to check spelling
    try:
        result = subprocess.run(
            ['aspell', 'list', '--lang=en'],
            input=text,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            words = [word for word in result.stdout.strip().split('\n') if word]
            # Filter out ignored terms
            filtered = [w for w in words if w.lower() not in ignore_terms]
            return filtered
    except Exception:
        pass

    return []


def simple_spell_check(text: str) -> List[str]:
    """
    Simple pattern-based spell checking for common mistakes.

    Returns:
        List of potential typos
    """
    common_typos = {
        'teh': 'the',
        'htat': 'that',
        'taht': 'that',
        'adn': 'and',
        'nad': 'and',
        'gatewey': 'gateway',
        'poihnts': 'points',
        'poitns': 'points',
        'recieve': 'receive',
        'seperate': 'separate',
        'occured': 'occurred',
        'occurance': 'occurrence',
        'begining': 'beginning',
        'lenght': 'length',
        'widht': 'width',
        'heigth': 'height',
        'coordiante': 'coordinate',
        'cooridate': 'coordinate',
        'paramter': 'parameter',
        'paramters': 'parameters',
        'intellignet': 'intelligent',
    }

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    typos = []

    for word in words:
        if word in common_typos:
            typos.append(f'{word} (suggest: {common_typos[word]})')

    return typos


def check_file(filepath: Path, use_aspell: bool = True) -> List[dict]:
    """
    Check a single file for spelling errors in TikZ diagrams.

    Returns:
        List of error dictionaries with file, line, text, and suggestions
    """
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

    tikz_blocks = extract_tikz_blocks(content, str(filepath))
    errors = []

    for tikz_content, start_line in tikz_blocks:
        texts = extract_text_from_tikz(tikz_content)

        for text, context in texts:
            # Simple pattern check (always run)
            simple_errors = simple_spell_check(text)
            if simple_errors:
                errors.append({
                    'file': str(filepath),
                    'line': start_line,
                    'text': text,
                    'context': context,
                    'suggestions': simple_errors
                })

            # Aspell check (if available and requested)
            if use_aspell:
                aspell_errors = check_spelling_with_aspell(text)
                if aspell_errors:
                    errors.append({
                        'file': str(filepath),
                        'line': start_line,
                        'text': text,
                        'context': context,
                        'suggestions': aspell_errors
                    })

    return errors


def main():
    """Main function to check all .qmd files for TikZ spelling errors."""
    # Find all .qmd files in the quarto/contents directory
    repo_root = Path(__file__).resolve().parents[3]
    contents_dir = repo_root / 'quarto' / 'contents'

    if not contents_dir.exists():
        print(f"Error: Contents directory not found at {contents_dir}", file=sys.stderr)
        return 1

    qmd_files = list(contents_dir.rglob('*.qmd'))
    print(f"Checking {len(qmd_files)} .qmd files for TikZ spelling errors...\n")

    # Check if aspell is available
    use_aspell = True
    try:
        subprocess.run(['aspell', '--version'], capture_output=True, check=True)
        print("Using aspell for comprehensive spell checking.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("aspell not found. Using pattern-based checking only.")
        print("Install aspell for more comprehensive checking: brew install aspell\n")
        use_aspell = False

    all_errors = []
    files_with_errors = 0

    for qmd_file in sorted(qmd_files):
        errors = check_file(qmd_file, use_aspell)
        if errors:
            files_with_errors += 1
            all_errors.extend(errors)

    # Print results
    if all_errors:
        print(f"\nFound {len(all_errors)} potential spelling errors in {files_with_errors} files:\n")

        current_file = None
        for error in sorted(all_errors, key=lambda e: (e['file'], e['line'])):
            if error['file'] != current_file:
                current_file = error['file']
                rel_path = Path(error['file']).relative_to(repo_root)
                print(f"\n{rel_path}")
                print("=" * len(str(rel_path)))

            print(f"  Line {error['line']}: {error['context']}")
            print(f"    → Issues: {', '.join(error['suggestions'])}")

        print(f"\n\nSummary: {len(all_errors)} potential errors in {files_with_errors} files")
        return 1
    else:
        print("\n✓ No spelling errors found in TikZ diagrams!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
