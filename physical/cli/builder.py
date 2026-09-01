"""Book build coordinator and Quarto execution wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

from .report import BOLD, CYAN, GREEN, RED, RESET, YELLOW


class BookBuilder:
    """Manages Quarto rendering, SVG->PDF compilation, and preview servers."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.book_dir = self.repo_root / "book"
        self.build_dir = self.book_dir / "_build"

    def sync_svg_assets(self, force: bool = False):
        """Finds all SVGs in chapters/ and converts them to PDFs if newer or missing."""
        svg_files = list(self.book_dir.rglob("*.svg"))
        converted = 0
        for svg in svg_files:
            pdf_target = svg.with_suffix(".pdf")
            if force or not pdf_target.exists() or svg.stat().st_mtime > pdf_target.stat().st_mtime:
                try:
                    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_target), str(svg)], check=True, capture_output=True)
                    converted += 1
                except Exception as e:
                    print(f"{YELLOW}Warning: Could not convert {svg.name} to PDF: {e}{RESET}")
        if converted > 0:
            print(f"{GREEN}✓ Converted {converted} updated SVG asset(s) to PDF with rsvg-convert.{RESET}")

    def clean(self):
        """Cleans build artifacts and cached LaTeX auxiliary files."""
        print(f"{CYAN}Cleaning build artifacts in {self.build_dir.name}...{RESET}")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir, ignore_errors=True)
        for pattern in ("*.aux", "*.log", "*.out", "*.toc", "*.fls", "*.fdb_latexmk"):
            for f in self.book_dir.glob(pattern):
                f.unlink(missing_ok=True)
        print(f"{GREEN}✓ Clean completed.{RESET}")

    def build_book(self, fmt: str = "pdf", clean_first: bool = False) -> bool:
        """Builds the entire book for the requested format ('pdf', 'html', 'all')."""
        if clean_first:
            self.clean()

        self.sync_svg_assets()

        cmd = ["quarto", "render"]
        if fmt == "pdf":
            cmd.extend(["--to", "pdf"])
        elif fmt == "html":
            cmd.extend(["--to", "html"])

        print(f"{BOLD}{CYAN}▶ Building Physical AI book (format: {fmt} [LuaLaTeX])...{RESET}")
        result = subprocess.run(cmd, cwd=self.book_dir)
        if result.returncode == 0:
            print(f"\n{GREEN}{BOLD}✓ Book build succeeded! Output in {self.build_dir}{RESET}\n")
            return True
        else:
            print(f"\n{RED}{BOLD}✗ Build failed with exit code {result.returncode}{RESET}\n")
            return False

    def build_chapter(self, chapter_name: str, fmt: str = "pdf") -> bool:
        """Builds a single chapter file (e.g. '01-boundary')."""
        self.sync_svg_assets()

        # Find matching chapter qmd
        matches = list(self.book_dir.glob(f"chapters/*{chapter_name}*/**/*.qmd")) + \
                  list(self.book_dir.glob(f"chapters/*{chapter_name}*.qmd"))

        if not matches:
            print(f"{RED}Error: No chapter matching '{chapter_name}' found under book/chapters/{RESET}")
            return False

        target_qmd = matches[0]
        rel_qmd = target_qmd.relative_to(self.book_dir)
        print(f"{BOLD}{CYAN}▶ Building single chapter: {rel_qmd} (format: {fmt} [LuaLaTeX])...{RESET}")

        cmd = ["quarto", "render", str(rel_qmd)]
        if fmt == "pdf":
            cmd.extend(["--to", "pdf"])
        elif fmt == "html":
            cmd.extend(["--to", "html"])

        result = subprocess.run(cmd, cwd=self.book_dir)
        if result.returncode == 0:
            print(f"\n{GREEN}{BOLD}✓ Chapter build succeeded: {rel_qmd}{RESET}\n")
            return True
        else:
            print(f"\n{RED}{BOLD}✗ Chapter build failed with exit code {result.returncode}{RESET}\n")
            return False

    def preview(self, port: int = 4200):
        """Launches Quarto live preview server."""
        print(f"{BOLD}{CYAN}▶ Launching Quarto live preview server on port {port}...{RESET}")
        subprocess.run(["quarto", "preview", "--port", str(port)], cwd=self.book_dir)
