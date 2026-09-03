#!/usr/bin/env python3
"""
book/tools/convert_chapter_refs_to_sec.py
Replaces explicit 'Chapter X' or 'Chapters X-Y' strings with Quarto '@sec-*' cross-references.
"""

import glob
import re

SEC_MAP = {
    1: "sec-boundary",
    2: "sec-body",
    3: "sec-brain",
    4: "sec-nervous",
    5: "sec-data",
    6: "sec-training",
    7: "sec-evaluation",
    8: "sec-perception",
    9: "sec-memory",
    10: "sec-intent",
    11: "sec-planning",
    12: "sec-enforcement",
    13: "sec-placement",
    14: "sec-intervention",
    15: "sec-verification",
    16: "sec-release",
    17: "sec-frontier",
}

def replace_chapter_refs(text):
    # Avoid replacing inside HTML comments or Markdown headers
    lines = text.split("\n")
    out_lines = []
    
    for line in lines:
        if line.startswith("<!--") or line.startswith("#"):
            out_lines.append(line)
            continue
        
        # 1. Replace multi-chapter ranges: "Chapters X through Y" or "Chapters X to Y" or "Chapters X–Y"
        def range_repl(m):
            c1 = int(m.group(1))
            sep = m.group(2).strip()
            c2 = int(m.group(3))
            s1 = SEC_MAP.get(c1, f"sec-{c1}")
            s2 = SEC_MAP.get(c2, f"sec-{c2}")
            if "and" in sep:
                return f"@{s1} and @{s2}"
            return f"@{s1} through @{s2}"

        line = re.sub(r'\bChapters?\s+(\d+)\s*(through|to|and|–|-)\s*(\d+)\b', range_repl, line, flags=re.IGNORECASE)
        
        # 2. Replace single chapter: "Chapter X (@sec-foo)" -> "@sec-foo" or "Chapter X" -> "@sec-foo"
        # First handle "Chapter X (@sec-foo)" redundancy
        def redundancy_repl(m):
            c = int(m.group(1))
            sec = m.group(2)
            return f"@{sec}"
        line = re.sub(r'\bChapter\s+(\d+)\s*\(@(sec-[a-z0-9-]+)\)', redundancy_repl, line, flags=re.IGNORECASE)
        
        # Then handle "Chapter X" -> "@sec-name"
        def single_repl(m):
            c = int(m.group(1))
            s = SEC_MAP.get(c)
            if s:
                return f"@{s}"
            return m.group(0)
        
        line = re.sub(r'\bChapter\s+(\d+)\b', single_repl, line, flags=re.IGNORECASE)
        
        # Replace Section 4.10 -> @sec-nervous-4-10 or @sec-nervous
        line = re.sub(r'\bSection\s+4\.10\b', r'@sec-nervous-4-10', line, flags=re.IGNORECASE)
        
        out_lines.append(line)
        
    return "\n".join(out_lines)

def run():
    files = sorted(glob.glob("book/chapters/*/*.qmd"))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = replace_chapter_refs(content)
        if new_content != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated: {path}")
    print("Cross-reference replacement complete.")

if __name__ == "__main__":
    run()
