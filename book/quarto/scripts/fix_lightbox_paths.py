#!/usr/bin/env python3
"""
Post-render step: fix lightbox hrefs for mediabag SVGs.

Quarto's lightbox feature generates <a href="HASH.svg"> for images stored in
the mediabag directory, but the actual file lives at
chapter_files/mediabag/HASH.svg. The <img src> inside the anchor is correct;
only the lightbox href is wrong. This causes a 404 when clicking a figure
to enlarge it.

This script rewrites each lightbox <a href="HASH.svg"> to use the path from
the child <img src>, making the enlarged-view link resolve correctly.

Fixes: https://github.com/harvard-edge/cs249r_book/issues/1795
"""

import re
import sys
from pathlib import Path


LIGHTBOX_PATTERN = re.compile(
    r'<a\s+href="([0-9a-f]{20,}\.svg)"(\s+class="lightbox"[^>]*>)'
    r'(<img\s+src="([^"]+)"[^>]*>)</a>'
)


def fix_lightbox_hrefs(html_path: Path) -> int:
    if not html_path.is_file():
        return 0
    try:
        text = html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    count = 0

    def replacer(m: re.Match) -> str:
        nonlocal count
        old_href = m.group(1)
        attrs = m.group(2)
        img_tag = m.group(3)
        img_src = m.group(4)
        if old_href != img_src:
            count += 1
            return f'<a href="{img_src}"{attrs}{img_tag}</a>'
        return m.group(0)

    new_text = LIGHTBOX_PATTERN.sub(replacer, text)
    if count > 0:
        html_path.write_text(new_text, encoding="utf-8")
    return count


def main() -> None:
    if len(sys.argv) > 1:
        build_dir = Path(sys.argv[1])
    else:
        script_dir = Path(__file__).resolve().parent
        project_dir = script_dir.parent
        build_dir = project_dir / "_build"
        if not build_dir.exists():
            build_dir = project_dir

    html_files = sorted(build_dir.rglob("*.html"))
    total_fixes = 0
    files_fixed = 0

    for html_file in html_files:
        fixes = fix_lightbox_hrefs(html_file)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1

    if total_fixes > 0:
        print(f"[fix-lightbox] Fixed {total_fixes} lightbox href(s) in {files_fixed} file(s).")
    else:
        print("[fix-lightbox] No lightbox mediabag mismatches found.")


if __name__ == "__main__":
    main()
