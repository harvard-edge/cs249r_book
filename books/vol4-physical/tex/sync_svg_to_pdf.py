#!/usr/bin/env python3
"""
Convert all SVG figures in book/chapters to high-resolution vector PDF using rsvg-convert.
Ensures LaTeX/LuaLaTeX always includes the latest updated SVG artwork.
"""

import os
import glob
import subprocess

def convert_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    book_dir = os.path.abspath(os.path.join(base_dir, ".."))
    svg_files = glob.glob(os.path.join(book_dir, "chapters", "**", "figures", "*.svg"), recursive=True)
    
    print(f"Found {len(svg_files)} SVG figures to sync with vector PDF...")
    success = 0
    for svg_path in sorted(svg_files):
        pdf_path = svg_path[:-4] + ".pdf"
        cmd = ["rsvg-convert", "-f", "pdf", "-o", pdf_path, svg_path]
        try:
            subprocess.run(cmd, check=True)
            success += 1
            rel_path = os.path.relpath(svg_path, book_dir)
            print(f"  ✓ {rel_path} -> .pdf")
        except Exception as e:
            print(f"  ✗ Failed {svg_path}: {e}")
            
    print(f"\nSuccessfully converted {success}/{len(svg_files)} SVGs to PDF.")

if __name__ == "__main__":
    convert_all()
