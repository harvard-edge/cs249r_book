#!/usr/bin/env python3
"""
Fix ABOUT.md titles to include module numbers consistently.

Changes titles from "# Tensor" to "# Module 01: Tensor"
"""

import re
from pathlib import Path

# Module names mapping
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


def fix_title(filepath: Path) -> bool:
    """Fix the title in an ABOUT.md file."""
    content = filepath.read_text(encoding='utf-8')

    # Extract module number from directory name
    module_dir = filepath.parent.name
    module_num = module_dir.split('_')[0]

    if module_num not in MODULE_NAMES:
        return False

    module_name = MODULE_NAMES[module_num]

    # Check if already has "Module XX:" format
    if f"# Module {module_num}:" in content:
        return False

    # Replace the first H1 title
    # Match patterns like "# Tensor" or "# Spatial (Convolutions)"
    pattern = r'^# .+$'
    new_title = f"# Module {module_num}: {module_name}"

    new_content = re.sub(pattern, new_title, content, count=1, flags=re.MULTILINE)

    if new_content != content:
        filepath.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    src_dir = Path('/Users/VJ/GitHub/MLSysBook/tinytorch/src')
    modified = []

    for about_file in sorted(src_dir.glob('*/ABOUT.md')):
        if fix_title(about_file):
            modified.append(about_file)
            print(f"Modified: {about_file}")

    # Also fix site/modules symlinked files
    site_modules = Path('/Users/VJ/GitHub/MLSysBook/tinytorch/site/modules')
    for about_file in sorted(site_modules.glob('*_ABOUT.md')):
        content = about_file.read_text(encoding='utf-8')
        # Extract module number from filename like "01_tensor_ABOUT.md"
        module_num = about_file.name.split('_')[0]

        if module_num not in MODULE_NAMES:
            continue

        module_name = MODULE_NAMES[module_num]

        if f"# Module {module_num}:" in content:
            continue

        pattern = r'^# .+$'
        new_title = f"# Module {module_num}: {module_name}"
        new_content = re.sub(pattern, new_title, content, count=1, flags=re.MULTILINE)

        if new_content != content:
            about_file.write_text(new_content, encoding='utf-8')
            modified.append(about_file)
            print(f"Modified: {about_file}")

    print(f"\nTotal modified: {len(modified)}")


if __name__ == "__main__":
    main()
