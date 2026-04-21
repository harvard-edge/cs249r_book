#!/usr/bin/env python3
"""
LaTeX post-processor for TinyTorch PDF builds.

Removes emojis from the generated .tex file for a clean, professional PDF.
"""

import re
from pathlib import Path

# Emojis to remove entirely (for clean, professional PDF)
EMOJIS_TO_REMOVE = [
    '✅', '❌', '🧪', '🔬', '📈', '📊', '📖', '🚀', '🎓',
    '🎉', '🌍', '👨', '👩', '🏫', '👥', '🏗', '🏛', '🏆', '🍃',
    '🏅', '👁', '💡', '💻', '💼', '💾', '📍', '📦', '🔀', '🔄',
    '🔍', '🔒', '🔢', '🔧', '🛠', '🖼', '🤖', '🤝', '🧠', '🧭',
    '🎯', '⭐', '⏱', '⚠', '⚡', '✨', '🌐', '📝', '🎨', '🔗',
    '📚', '🏠', '🎮', '🔮', '💪', '🌟', '📌', '🗂', '📁', '🗃',
    '⚙', '🔩', '🔨', '⛏', '🪛', '🧰', '📐', '📏', '🧮', '💯',
    '🎲', '🎰', '🎪', '🎭', '🎬', '🎤', '🎧', '🎵', '🎶', '🎸',
    '🏃', '🚶', '🧑', '👶', '👴', '👵', '🧒', '👦', '👧', '🧓',
    '‍', '️',  # Zero-width joiner and variation selector
]

# Fire emoji replacement - use inline image for branding
FIRE_EMOJI = '🔥'
FIRE_IMAGE_LATEX = r'\raisebox{-0.1em}{\includegraphics[height=1em]{fire-emoji.png}}'

# Subscripts/superscripts - convert to LaTeX math
MATH_REPLACEMENTS = {
    'ᴺ': r'$^N$',
    'ᵐ': r'$^m$',
    'ₙ': r'$_n$',
    'ₘ': r'$_m$',
    '₀': r'$_0$',
    '₁': r'$_1$',
    '₂': r'$_2$',
    '₃': r'$_3$',
    '₄': r'$_4$',
    '₅': r'$_5$',
    '₆': r'$_6$',
    '₇': r'$_7$',
    '₈': r'$_8$',
    '₉': r'$_9$',
}

def process_latex_file(tex_file: Path) -> int:
    """Process .tex file, removing emojis and duplicate title page for clean PDF."""
    if not tex_file.exists():
        print(f"Error: {tex_file} not found")
        return 0

    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original_len = len(content)

    # FIRST: Remove the duplicate Sphinx-generated title page metadata
    # This must happen BEFORE emoji replacement to avoid breaking the regex
    # Clear \title{...}, \date{...}, \author{...} so the default title page is empty
    # Use specific line replacements to be safe
    content = content.replace(r'\title{Tiny🔥Torch}', r'\title{}')
    content = re.sub(r'\\date\{[A-Za-z]+ \d+, \d+\}', r'\\date{}', content)
    content = re.sub(r'\\author\{Prof\.\\@\{\} [^}]+\}', r'\\author{}', content)

    # Replace fire emoji with inline image (for Tiny🔥Torch branding)
    content = content.replace(FIRE_EMOJI, FIRE_IMAGE_LATEX)

    # Remove all other emojis
    for emoji in EMOJIS_TO_REMOVE:
        content = content.replace(emoji, '')

    # Replace math symbols
    for symbol, latex in MATH_REPLACEMENTS.items():
        content = content.replace(symbol, latex)

    # Clean up escaped LaTeX commands that appear literally in tables
    # These come from markdown files using LaTeX syntax that gets escaped
    # Green checkmark: \textcolor{green!70!black}{$\checkmark$} -> ✓
    content = re.sub(
        r'\\textcolor\{green!70!black\}\{\$\\checkmark\$\}',
        r'\\checkmark',
        content
    )
    # Red X: \textcolor{red!70!black}{$\times$} -> ✗
    content = re.sub(
        r'\\textcolor\{red!70!black\}\{\$\\times\$\}',
        r'$\\times$',
        content
    )

    # Fix figure placement: change [htbp] to [H] for inline placement
    content = re.sub(
        r'\\begin\{figure\}\[htbp\]',
        r'\\begin{figure}[H]',
        content
    )

    # Center all includegraphics that aren't already centered
    # Find \includegraphics not preceded by \centering and wrap them
    content = re.sub(
        r'(\\begin\{figure\}\[H\]\n)(\\includegraphics)',
        r'\1\\centering\n\2',
        content
    )

    # Scale mermaid diagrams: use adjustbox for smart max-width scaling
    # This allows small diagrams to stay natural size, but caps large ones at column width
    # First, ensure adjustbox is available by adding to preamble if not present
    if r'\usepackage{adjustbox}' not in content:
        # Add adjustbox after float package
        content = content.replace(
            r'\usepackage{float}',
            r'\usepackage{float}' + '\n' + r'\usepackage{adjustbox}'
        )

    # Replace sphinxincludegraphics for mermaid with width-constrained includegraphics
    # Using width=\linewidth ensures diagram fits within text margins
    # height=0.6\textheight allows taller diagrams while keeping them on one page
    # keepaspectratio prevents distortion - image scales to fit whichever constraint is tighter
    content = re.sub(
        r'\\sphinxincludegraphics\{(mermaid-[^}]+\.pdf)\}',
        r'\\includegraphics[width=\\linewidth,height=0.6\\textheight,keepaspectratio]{\g<1>}',
        content
    )

    # Write back
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)

    chars_removed = original_len - len(content)
    return chars_removed

def main():
    """Main entry point."""
    import sys

    # Default path
    site_dir = Path(__file__).parent.parent
    tex_file = site_dir / '_build' / 'latex' / 'tinytorch-course.tex'

    # Allow override from command line
    if len(sys.argv) > 1:
        tex_file = Path(sys.argv[1])

    print(f"Cleaning emojis from: {tex_file.name}")

    chars_removed = process_latex_file(tex_file)

    if chars_removed > 0:
        print(f"Removed {chars_removed} emoji characters for clean PDF")
    else:
        print("No emojis found")

if __name__ == '__main__':
    main()
