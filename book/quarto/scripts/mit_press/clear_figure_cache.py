#!/usr/bin/env python3
"""
Pre-Render: Clear stale figure data from previous builds.
Ensures LaTeX figure numbers align with current build.
"""

import sys
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    quarto_dir = script_dir.parent.parent
    
    latex_file = quarto_dir / 'index_figures.txt'
    
    if latex_file.exists():
        latex_file.unlink()
        print(f"[Figure Cache] Cleared: {latex_file.name}", file=sys.stderr)
    else:
        print(f"[Figure Cache] Clean (no stale data)", file=sys.stderr)

if __name__ == '__main__':
    main()
