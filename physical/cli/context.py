"""Book parsing, AST extraction, and cross-chapter context indexing."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple


class BookContext:
    """Holds parsed state and indexed definitions across the entire book."""

    def __init__(self, repo_root: Path, chapter_filter: Optional[str] = None):
        self.repo_root = repo_root.resolve()
        self.book_dir = self.repo_root / "book"
        self.chapters_dir = self.book_dir / "chapters"
        self.bib_path = self.book_dir / "references.bib"
        self.pdf_path = self.book_dir / "_build" / "Physical-AI.pdf"
        self.html_dir = (self.book_dir / "_html") if (self.book_dir / "_html").exists() else (self.book_dir / "_build")
        self.log_path = self.book_dir / "Physical-AI.log"
        self.chapter_filter = chapter_filter

        self.qmd_files: List[Path] = self._discover_qmd_files()
        self.bib_keys: Set[str] = self._load_bib_keys()

        # Definitions: name -> (filepath, line_idx)
        self.fig_defs: Dict[str, Tuple[Path, int]] = {}
        self.tbl_defs: Dict[str, Tuple[Path, int]] = {}
        self.sec_defs: Dict[str, Tuple[Path, int]] = {}
        self.eq_defs: Dict[str, Tuple[Path, int]] = {}
        self.callout_defs: Dict[str, Tuple[Path, int]] = {}

        # References in prose: list of (ref_id, filepath, line_idx)
        self.fig_refs: List[Tuple[str, Path, int]] = []
        self.tbl_refs: List[Tuple[str, Path, int]] = []
        self.sec_refs: List[Tuple[str, Path, int]] = []
        self.eq_refs: List[Tuple[str, Path, int]] = []
        self.callout_refs: List[Tuple[str, Path, int]] = []
        self.citations: List[Tuple[str, Path, int]] = []

        # Image/asset references in markdown: list of (rel_path_str, filepath, line_idx)
        self.image_includes: List[Tuple[str, Path, int]] = []

        # Physical asset files on disk in chapters
        self.disk_assets: List[Path] = self._discover_disk_assets()

        self._parse_qmd_files()

    def _discover_qmd_files(self) -> List[Path]:
        files: List[Path] = []
        if self.chapter_filter:
            for p in self.chapters_dir.glob(f"*{self.chapter_filter}*/**/*.qmd"):
                files.append(p)
            for p in self.chapters_dir.glob(f"*{self.chapter_filter}*.qmd"):
                files.append(p)
        else:
            files.extend(sorted(self.book_dir.rglob("*.qmd")))
        return [f for f in files if not f.name.startswith(".") and not f.name.startswith("_")]

    def _discover_disk_assets(self) -> List[Path]:
        assets: List[Path] = []
        search_dir = self.chapters_dir
        if self.chapter_filter:
            # Match specific chapter directory
            matches = list(self.chapters_dir.glob(f"*{self.chapter_filter}*"))
            if matches and matches[0].is_dir():
                search_dir = matches[0]

        for ext in ("*.svg", "*.png", "*.jpg", "*.jpeg", "*.pdf", "*.webp"):
            assets.extend(search_dir.rglob(ext))
        # Filter out generated PDF twins of SVGs in figures dirs
        clean_assets = []
        for a in assets:
            if a.name.startswith("."):
                continue
            # If it's a PDF and has matching SVG, it's a generated vector artifact
            if a.suffix.lower() == ".pdf" and a.with_suffix(".svg").exists():
                continue
            clean_assets.append(a)
        return clean_assets

    def _load_bib_keys(self) -> Set[str]:
        keys = set()
        if not self.bib_path.exists():
            return keys
        content = self.bib_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"@\w+\s*\{\s*([a-zA-Z0-9_:-]+)\s*,", content):
            keys.add(match.group(1).strip())
        return keys

    def _parse_qmd_files(self):
        for file_path in self.qmd_files:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            in_code_block = False

            for line_idx, line in enumerate(lines, start=1):
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue

                if in_code_block:
                    continue

                # 1. Definitions (use non-greedy matching to support LaTeX math braces in captions)
                for match in re.finditer(r"#(fig-[\w-]+)", line):
                    self.fig_defs[match.group(1)] = (file_path, line_idx)
                for match in re.finditer(r"#(tbl-[\w-]+)", line):
                    self.tbl_defs[match.group(1)] = (file_path, line_idx)
                for match in re.finditer(r"#(sec-[\w-]+)", line):
                    self.sec_defs[match.group(1)] = (file_path, line_idx)
                for match in re.finditer(r"#(eq-[\w-]+)", line):
                    self.eq_defs[match.group(1)] = (file_path, line_idx)
                for match in re.finditer(r"#(callout-[\w-]+|def-[\w-]+|law-[\w-]+|contract-[\w-]+|autopsy-[\w-]+)", line):
                    self.callout_defs[match.group(1)] = (file_path, line_idx)

                # 2. Markdown image inclusions: ![alt](path)
                for match in re.finditer(r"!\[.*?\]\((.*?)\)", line):
                    img_path_str = match.group(1).split()[0].strip("{}")
                    self.image_includes.append((img_path_str, file_path, line_idx))

                # 3. References: @fig-xxx, @tbl-xxx, @sec-xxx, @eq-xxx
                for match in re.finditer(r"@((?:fig|tbl|sec|eq)-[\w-]+)", line):
                    ref_id = match.group(1)
                    if ref_id.startswith("fig-"):
                        self.fig_refs.append((ref_id, file_path, line_idx))
                    elif ref_id.startswith("tbl-"):
                        self.tbl_refs.append((ref_id, file_path, line_idx))
                    elif ref_id.startswith("sec-"):
                        self.sec_refs.append((ref_id, file_path, line_idx))
                    elif ref_id.startswith("eq-"):
                        self.eq_refs.append((ref_id, file_path, line_idx))

                # 4. Citations: @citeKey
                for match in re.finditer(r"@([a-zA-Z][a-zA-Z0-9_-]+)", line):
                    cite_key = match.group(1)
                    if cite_key.startswith(("fig-", "tbl-", "sec-", "eq-", "callout-", "def-", "law-")):
                        continue
                    if cite_key in ("ref", "label", "cite", "include", "includegraphics", "vspace", "hspace", "linewidth", "textwidth", "caption"):
                        continue
                    self.citations.append((cite_key, file_path, line_idx))
