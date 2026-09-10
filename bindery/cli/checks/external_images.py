"""
External image URL detection for MLSysBook.

Scans QMD files for external HTTP/HTTPS image references that should be local assets.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse


def extract_figure_images(content: str) -> List[Tuple[str, str, str, str, str]]:
    r"""
    Extract markdown images with or without #fig references.
    Returns list of tuples: (full_match, caption, url, fig_id, attributes)
    """
    matches = []
    lines = content.split('\n')

    for line in lines:
        if '![' not in line:
            continue

        idx = 0
        while idx < len(line):
            start = line.find('![', idx)
            if start == -1:
                break

            end_brace = line.find('}', start)
            next_img = line.find('![', start + 2)

            if end_brace != -1 and (next_img == -1 or end_brace < next_img):
                end = end_brace + 1
            elif next_img != -1:
                end = next_img
            else:
                end = len(line)

            full_match = line[start:end]
            url_patterns = list(re.finditer(r'\]\(([^)]+)\)', full_match))

            if url_patterns:
                url = url_patterns[-1].group(1).strip()
                fig_id = None
                attributes = ""
                attrs_match = re.search(r'\{([^}]+)\}', full_match)
                if attrs_match:
                    attrs_block = attrs_match.group(1)
                    fig_match = re.search(r'#(fig-[^\s}]+)', attrs_block)
                    if fig_match:
                        fig_id = fig_match.group(1)
                    attributes = attrs_block.strip()

                caption_match = re.search(r'!\[([^\]]+)\]', full_match)
                caption = caption_match.group(1) if caption_match else ""

                if url.lower().startswith(('http://', 'https://')):
                    if not fig_id:
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                        fig_id = f"fig-auto-{url_hash}"

                    matches.append((full_match, caption, url, fig_id, attributes))

            idx = end

    return matches


class ImageDownloader:
    """Validator and manager for external image assets."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def find_qmd_files(self) -> List[Path]:
        qmd_files = []
        if not self.base_dir.exists():
            return qmd_files
        for qmd_file in self.base_dir.rglob("*.qmd"):
            if qmd_file.is_file():
                qmd_files.append(qmd_file)
        return qmd_files

    def validate_external_images(self, ignore_external: bool = False) -> Tuple[int, List[Tuple[Path, str, str]]]:
        qmd_files = self.find_qmd_files()
        all_external_images = []

        for qmd_file in qmd_files:
            try:
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                figure_images = extract_figure_images(content)
                for full_match, caption, url, fig_id, attributes in figure_images:
                    all_external_images.append((qmd_file, fig_id, url))
            except Exception:
                continue

        return len(qmd_files), all_external_images
