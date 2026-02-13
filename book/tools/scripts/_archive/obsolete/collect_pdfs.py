#!/usr/bin/env python3
"""
Collect and organize chapter PDFs from build artifacts.

This script:
1. Reads the chapter order from _quarto.yml
2. Finds built PDFs in the artifacts directories
3. Copies them to a single directory with numbered prefixes matching the book order
4. Optionally keeps the detailed folder structure
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse


def extract_chapter_slug(qmd_path):
    """Extract the chapter slug from a qmd path.
    
    Examples:
    - 'contents/vol1/introduction/introduction.qmd' -> 'introduction'
    - 'contents/vol1/frontmatter/foreword.qmd' -> 'foreword'
    - 'contents/vol1/backmatter/references.qmd' -> 'references'
    """
    path = Path(qmd_path)
    
    # Skip part dividers (they don't have PDFs)
    if path.stem in ['foundations_principles', 'build_principles', 
                      'optimize_principles', 'deploy_principles']:
        return None
    
    # Skip index
    if path.stem == 'index':
        return None
    
    # For frontmatter and backmatter, use the file stem
    # For regular chapters, use the parent directory name
    parent_name = path.parent.name
    file_stem = path.stem
    
    if parent_name in ['frontmatter', 'backmatter']:
        # Use the file name (e.g., 'foreword' from 'foreword.qmd')
        return file_stem
    elif parent_name == 'glossary':
        # Glossary is special - use 'glossary'
        return 'glossary'
    elif parent_name == file_stem:
        # For regular chapters where dir matches file (e.g., introduction/introduction.qmd)
        # Return the parent directory name
        return parent_name
    else:
        # For cases where parent dir differs from file name
        # (e.g., optimizations/model_compression.qmd -> 'model_compression')
        return file_stem


def read_chapter_order(quarto_yml_path):
    """Read the chapter order from _quarto.yml."""
    with open(quarto_yml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    chapters = []
    
    # Process main chapters
    if 'book' in config and 'chapters' in config['book']:
        for chapter_path in config['book']['chapters']:
            slug = extract_chapter_slug(chapter_path)
            if slug and slug != 'index':  # Skip index and part dividers
                chapters.append(slug)
    
    # Process appendices
    if 'book' in config and 'appendices' in config['book']:
        for appendix_path in config['book']['appendices']:
            slug = extract_chapter_slug(appendix_path)
            if slug:
                chapters.append(slug)
    
    return chapters


def find_pdf(chapter_slug, logs_dir, vol):
    """Find the PDF for a given chapter."""
    artifacts_dir = logs_dir / vol / chapter_slug / 'artifacts'
    
    if not artifacts_dir.exists():
        return None
    
    # Look for PDF files
    pdf_files = list(artifacts_dir.glob('*.pdf'))
    
    if not pdf_files:
        return None
    
    # Return the first PDF found (should be only one)
    return pdf_files[0]


def collect_pdfs(vol='vol1', output_dir=None, keep_structure=False):
    """Collect and organize PDFs from build artifacts."""
    
    # Paths
    book_dir = Path(__file__).parent.parent.parent.parent
    quarto_yml = book_dir / 'quarto' / '_quarto.yml'
    logs_dir = book_dir / 'tools' / 'scripts' / 'testing' / 'logs'
    
    if output_dir is None:
        output_dir = logs_dir / f'{vol}_collected_pdfs'
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read chapter order
    print(f"Reading chapter order from {quarto_yml}...")
    chapters = read_chapter_order(quarto_yml)
    
    print(f"\nFound {len(chapters)} chapters in order:")
    for i, chapter in enumerate(chapters, 1):
        print(f"  {i:02d}. {chapter}")
    
    # Collect PDFs
    print(f"\nCollecting PDFs to {output_dir}...")
    collected = []
    missing = []
    
    for i, chapter_slug in enumerate(chapters, 1):
        pdf_path = find_pdf(chapter_slug, logs_dir, vol)
        
        if pdf_path:
            # Create new filename with number prefix
            new_filename = f"{i:02d}_{chapter_slug}.pdf"
            output_path = output_dir / new_filename
            
            # Copy the PDF
            shutil.copy2(pdf_path, output_path)
            collected.append((chapter_slug, output_path))
            print(f"  ✅ {i:02d}. {chapter_slug} -> {new_filename}")
        else:
            missing.append(chapter_slug)
            print(f"  ❌ {i:02d}. {chapter_slug} (PDF not found)")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total chapters: {len(chapters)}")
    print(f"  Collected: {len(collected)}")
    print(f"  Missing: {len(missing)}")
    print(f"\n  Output directory: {output_dir}")
    
    if missing:
        print(f"\n  Missing PDFs for:")
        for chapter in missing:
            print(f"    - {chapter}")
    
    print(f"{'='*70}\n")
    
    return collected, missing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect and organize chapter PDFs from build artifacts'
    )
    parser.add_argument(
        '--vol',
        default='vol1',
        help='Volume to collect (default: vol1)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output directory (default: logs/<vol>_collected_pdfs)'
    )
    parser.add_argument(
        '--keep-structure',
        action='store_true',
        help='Keep detailed folder structure (not implemented yet)'
    )
    
    args = parser.parse_args()
    
    collect_pdfs(vol=args.vol, output_dir=args.output, keep_structure=args.keep_structure)
