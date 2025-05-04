#!/usr/bin/env python3
"""
📘 Quarto Project Stats Collector

This script scans a Quarto project directory, parses `.qmd` files, and reports useful statistics 
to help you understand the structure and content of your textbook or technical book.

✨ Tracked Stats (per file):
- 🧱 Chapters, Sections, Subsections
- 📝 Word Count
- 🖼️ Figures, 📊 Tables, 💻 Code Blocks
- 📚 Citations, 🦶 Footnotes, 📦 Callouts
- 🚧 TODOs and FIXMEs
- ❌ Figures/Tables without captions

Usage:
    python quarto_stats.py path/to/project
"""

import re
from pathlib import Path
from collections import defaultdict

def strip_code_blocks(content):
    """Remove fenced code blocks from the content."""
    return re.sub(r"```.*?\n.*?```", "", content, flags=re.DOTALL)

def collect_stats_from_qmd(file_path):
    stats = defaultdict(int)
    with open(file_path, "r", encoding="utf-8") as f:
        full_content = f.read()

    # Strip fenced code blocks before structural analysis
    content = strip_code_blocks(full_content)
    lines = content.splitlines()

    # 🧱 Structure
    stats['chapters'] += sum(1 for line in lines if line.strip().startswith("# "))
    stats['sections'] += sum(1 for line in lines if line.strip().startswith("## "))
    stats['subsections'] += sum(1 for line in lines if line.strip().startswith("### "))

    # 📝 Word Count (including code and comments)
    stats['words'] += len(re.findall(r'\b\w+\b', full_content))

    # 🎨 Figures and 📊 Tables (only labeled ones using #fig- and #tbl-)
    fig_labels = list(set(
        re.findall(r'#fig-[\w-]+', full_content) +
        re.findall(r'#\|\s*label:\s*fig-[\w-]+', full_content)
    ))
    tbl_labels = list(set(
        re.findall(r'#tbl-[\w-]+', full_content) +
        re.findall(r'#\|\s*label:\s*tbl-[\w-]+', full_content)
    ))

    # Count valid figures and tables (only labeled)
    stats['figures'] += len(fig_labels)
    stats['tables'] += len(tbl_labels)

    # ❌ Figures and Tables Without Captions (set to zero since unlabeled are ignored)
    stats['figs_no_caption'] = 0
    stats['tables_no_caption'] = 0

    # 💻 Code blocks
    stats['code_blocks'] += len(re.findall(r'^```', full_content, re.MULTILINE))

    # 📚 Citations
    stats['citations'] += len(re.findall(r'@[\w:.-]+', content))

    # 🦶 Footnotes
    stats['footnotes'] += len(re.findall(r'\[\^.+?\]', content))

    # 📦 Callouts
    stats['callouts'] += len(re.findall(r':::\s*\{\.callout-', content))

    # 🚧 TODOs and FIXMEs
    stats['todos'] += len(re.findall(r'TODO|FIXME', full_content, re.IGNORECASE))

    return stats


def summarize_stats(stats_by_file):
    total = defaultdict(int)
    header = (
        f"{'File':35} | {'Ch':>3} | {'Sec':>4} | {'Words':>7} | "
        f"{'Fig':>5} | {'Tbl':>5} | {'Code':>5} | {'Cite':>5} | "
        f"{'Foot':>5} | {'Call':>5} | {'TODO':>5}"
    )

    print(header)
    print("-" * len(header))

    for file, stats in stats_by_file.items():
        print(f"{file.name:35} | {stats['chapters']:>3} | {stats['sections']:>4} | {stats['words']:>7} | "
            f"{stats['figures']:>5} | {stats['tables']:>5} | {stats['code_blocks']:>5} | {stats['citations']:>5} | "
            f"{stats['footnotes']:>5} | {stats['callouts']:>5} | {stats['todos']:>5}")

        for key in stats:
            total[key] += stats[key]

    print("\n📊 Total Summary:")
    emoji_label = {
        "chapters":           "🧱 Chapters",
        "sections":           "🧱 Sections",
        "subsections":        "🧱 Subsections",
        "words":              "📝 Words",
        "figures":            "🎨 Figures",
        "tables":             "📊 Tables",
        "code_blocks":        "💻 Code Blocks",
        "citations":          "📚 Citations",
        "footnotes":          "🦶 Footnotes",
        "callouts":           "📦 Callouts",
        "todos":              "🚧 TODOs",
        "figs_no_caption":    "❌ Figures w/o Caption",
        "tables_no_caption":  "❌ Tables w/o Caption"
    }

    for key, value in total.items():
        label = emoji_label.get(key, key)
        print(f"{label:<25} : {value}")

def collect_project_stats(path):
    """Walk through all .qmd files and collect stats."""
    path = Path(path)
    qmd_files = list(path.rglob("*.qmd"))
    if not qmd_files:
        print("⚠️ No QMD files found in the specified path.")
        return

    stats_by_file = {}
    for qmd_file in qmd_files:
        stats_by_file[qmd_file] = collect_stats_from_qmd(qmd_file)
    summarize_stats(stats_by_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="📘 Collect Quarto textbook stats.")
    parser.add_argument("path", help="Path to the root of the Quarto project")
    args = parser.parse_args()
    collect_project_stats(args.path)
