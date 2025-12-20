#!/usr/bin/env python3
"""
Add align:center and caption to mermaid diagrams in ABOUT.md files.

This script:
1. Finds all ```{mermaid} blocks
2. Adds :align: center option
3. Generates a caption from the subgraph title or flowchart context
"""

import os
import re
import sys
from pathlib import Path

# Module names for captions
MODULE_NAMES = {
    "01": "Tensor",
    "02": "Activations",
    "03": "Layers",
    "04": "Losses",
    "05": "DataLoader",
    "06": "Autograd",
    "07": "Optimizers",
    "08": "Training",
    "09": "Convolutions",
    "10": "Tokenization",
    "11": "Embeddings",
    "12": "Attention",
    "13": "Transformers",
    "14": "Profiling",
    "15": "Quantization",
    "16": "Compression",
    "17": "Acceleration",
    "18": "Memoization",
    "19": "Benchmarking",
    "20": "Capstone",
}


def extract_caption_from_mermaid(mermaid_code: str, module_num: str) -> str:
    """Extract a caption from the mermaid diagram content."""
    # Try to find subgraph title
    subgraph_match = re.search(r'subgraph\s+"([^"]+)"', mermaid_code)
    if subgraph_match:
        return subgraph_match.group(1)

    # Try to find subgraph with single quotes
    subgraph_match = re.search(r"subgraph\s+'([^']+)'", mermaid_code)
    if subgraph_match:
        return subgraph_match.group(1)

    # Try to find subgraph without quotes
    subgraph_match = re.search(r'subgraph\s+(\w+(?:\s+\w+)*)', mermaid_code)
    if subgraph_match:
        title = subgraph_match.group(1).strip()
        if title and title not in ['direction', 'end']:
            return title

    # Fall back to module name
    module_name = MODULE_NAMES.get(module_num, "Architecture")
    return f"{module_name} Architecture"


def process_file(filepath: Path) -> bool:
    """Process a single ABOUT.md file and add mermaid options."""
    content = filepath.read_text(encoding='utf-8')
    original_content = content

    # Extract module number from path
    module_dir = filepath.parent.name
    module_num = module_dir.split('_')[0] if '_' in module_dir else "00"

    # Skip if already has :align: center (already processed)
    if ':align: center' in content:
        return False

    # Pattern to match ```{mermaid} blocks followed by newline and content
    # Captures the mermaid block content until closing ```
    pattern = r'```\{mermaid\}\n((?:(?!```)[\s\S])*?)```'

    diagram_count = [0]  # Use list for mutability in nested function

    def replace_mermaid_block(match):
        diagram_count[0] += 1
        block_content = match.group(1)

        # Extract caption from block content
        caption = extract_caption_from_mermaid(block_content, module_num)

        # Return the mermaid directive with options, preserving original content
        return f'```{{mermaid}}\n:align: center\n:caption: {caption}\n{block_content}```'

    # Replace all mermaid blocks
    new_content = re.sub(pattern, replace_mermaid_block, content)

    if new_content != original_content:
        filepath.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    """Process all ABOUT.md files in src directory."""
    src_dir = Path('/Users/VJ/GitHub/MLSysBook/tinytorch/src')

    modified_files = []

    for about_file in src_dir.glob('*/ABOUT.md'):
        if process_file(about_file):
            modified_files.append(about_file)
            print(f"Modified: {about_file}")

    # Also process site markdown files
    site_dir = Path('/Users/VJ/GitHub/MLSysBook/tinytorch/site')
    for md_file in site_dir.rglob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        if '```{mermaid}' not in content:
            continue
        # Skip if already processed
        if ':align: center' in content:
            continue

        # Pattern to match ```{mermaid} blocks
        pattern = r'```\{mermaid\}\n((?:(?!```)[\s\S])*?)```'

        def replace_site_mermaid(match):
            block_content = match.group(1)
            # Try to extract caption from subgraph
            subgraph_match = re.search(r'subgraph\s+"([^"]+)"', block_content)
            if subgraph_match:
                caption = subgraph_match.group(1)
            else:
                caption = "Architecture Overview"
            return f'```{{mermaid}}\n:align: center\n:caption: {caption}\n{block_content}```'

        new_content = re.sub(pattern, replace_site_mermaid, content)
        if new_content != content:
            md_file.write_text(new_content, encoding='utf-8')
            modified_files.append(md_file)
            print(f"Modified: {md_file}")

    print(f"\nTotal files modified: {len(modified_files)}")

    if modified_files:
        sys.exit(1)  # Pre-commit convention
    sys.exit(0)


if __name__ == "__main__":
    main()
