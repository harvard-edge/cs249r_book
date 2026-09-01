"""Visual bounding box, margin overflow, and sidenote collision verification via PyMuPDF."""

from __future__ import annotations

import re
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport

try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except ImportError:
    HAVE_FITZ = False

# Layout Geometry Constants (7.5" x 10" scrbook, 540 x 720 pt)
PAGE_WIDTH = 540.0
PAGE_HEIGHT = 720.0

HEADER_BOTTOM = 46.0
USABLE_TOP = 52.0
USABLE_BOTTOM = 662.4
FOOTER_BOTTOM_LIMIT = 672.0

# Recto (Odd / Right): Inner left (57.6), Main Body (57.6 - 403.2), Outer Margin (403.2 - 528.0)
RECTO_BODY_LEFT = 57.6
RECTO_BODY_RIGHT = 403.2
RECTO_MARGIN_LEFT = 403.2
RECTO_MARGIN_RIGHT = 528.0

# Verso (Even / Left): Outer Margin (12.0 - 136.8), Main Body (136.8 - 482.4), Inner right (482.4 - 540.0)
VERSO_MARGIN_LEFT = 12.0
VERSO_MARGIN_RIGHT = 136.8
VERSO_BODY_LEFT = 136.8
VERSO_BODY_RIGHT = 482.4


@CheckRegistry.register
class VisualMarginBoundingBoxCheck(BaseCheck):
    name = "visual_margin_bounding_box"
    description = "Performs vector-level layout, margin overflow, and sidenote collision analysis on compiled PDF"
    category = "layout"

    def _index_source_files(self, repo_root: Path) -> List[Tuple[str, List[Tuple[int, str]]]]:
        """Index all .qmd files in book/ for fast snippet fuzzy search."""
        index = []
        qmd_files = glob.glob(str(repo_root / "book" / "chapters" / "**" / "*.qmd"), recursive=True)
        qmd_files += glob.glob(str(repo_root / "book" / "appendix" / "*.qmd"))
        qmd_files += glob.glob(str(repo_root / "book" / "front-matter" / "*.qmd"))
        qmd_files += glob.glob(str(repo_root / "book" / "parts" / "*.qmd"))
        
        for fpath in qmd_files:
            rel_path = os.path.relpath(fpath, repo_root)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = [(i + 1, line.strip()) for i, line in enumerate(f.readlines()) if line.strip()]
                    index.append((rel_path, lines))
            except Exception:
                pass
        return index

    def _locate_source(self, snippet: str, source_index: list) -> Tuple[Optional[str], Optional[int]]:
        """Find the source file and line number matching a text snippet."""
        if not snippet or len(snippet) < 12:
            return None, None
        
        clean = re.sub(r"[^\w\s]", "", snippet).lower()
        words = clean.split()
        if len(words) < 3:
            return None, None
        
        target_sub = " ".join(words[:5])

        for rel_path, lines in source_index:
            for line_no, line_text in lines:
                clean_line = re.sub(r"[^\w\s]", "", line_text).lower()
                if target_sub in clean_line:
                    return rel_path, line_no
        return None, None

    def run(self, ctx: BookContext, report: LintReport):
        if not HAVE_FITZ:
            report.add_issue(LintIssue(
                category="layout",
                severity="INFO",
                file="Physical-AI.pdf",
                line=None,
                page=None,
                message="PyMuPDF (fitz) is not installed; skipping vector bounding box analysis."
            ))
            return

        if not ctx.pdf_path.exists():
            report.add_issue(LintIssue(
                category="layout",
                severity="ERROR",
                file=str(ctx.pdf_path),
                line=None,
                page=None,
                message="PDF not found. Rebuild with 'pai build' first."
            ))
            return

        source_index = self._index_source_files(ctx.repo_root)
        doc = fitz.open(str(ctx.pdf_path))
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            blocks = page.get_text("blocks")
            # Extract printed page number from header
            header_text = ""
            for b in blocks:
                if b[1] < HEADER_BOTTOM:
                    header_text += " " + b[4].strip()
            page_digits = re.findall(r"\b\d+\b", header_text)
            book_page_num = int(page_digits[0]) if page_digits else page_num
            is_recto = (book_page_num % 2 == 1)

            # Collect margin blocks for collision detection
            margin_blocks = []
            max_body_y = 0.0
            body_block_count = 0
            is_chapter_opener = False

            for b in blocks:
                x0, y0, x1, y1, text, bno, btype = b
                clean_txt = " ".join(text.split())
                if not clean_txt:
                    continue

                if "Chapter" in clean_txt and y0 < 150.0:
                    is_chapter_opener = True

                if y1 <= HEADER_BOTTOM:
                    continue

                # Classify block position
                if is_recto:
                    is_margin = (x0 >= RECTO_MARGIN_LEFT - 5.0)
                    is_body = (x0 < RECTO_MARGIN_LEFT - 5.0)
                else:
                    is_margin = (x1 <= VERSO_MARGIN_RIGHT + 5.0)
                    is_body = (x1 > VERSO_MARGIN_RIGHT + 5.0)

                # --- 1. Margin Checks ---
                if is_margin:
                    margin_blocks.append((y0, y1, x0, x1, clean_txt))

                    # Bottom overflow past usable footer baseline
                    if y1 > FOOTER_BOTTOM_LIMIT:
                        over_pts = y1 - USABLE_BOTTOM
                        s_file, s_line = self._locate_source(clean_txt, source_index)
                        report.add_issue(LintIssue(
                            category="layout",
                            severity="ERROR" if over_pts > 10.0 else "WARNING",
                            file=s_file or "Physical-AI.pdf",
                            line=s_line,
                            page=book_page_num,
                            message=f"Margin content extends {over_pts:.1f}pt past footer baseline (y={y1:.1f}/{PAGE_HEIGHT:.1f}pt). Fix: shorten note or add [offset=-20pt].",
                            context=f"Book Page {book_page_num} (PDF p.{page_num}) | '{clean_txt[:60]}...'"
                        ))

                    # Outer edge protrusion
                    if is_recto and x1 > RECTO_MARGIN_RIGHT + 4.0:
                        over_x = x1 - RECTO_MARGIN_RIGHT
                        s_file, s_line = self._locate_source(clean_txt, source_index)
                        report.add_issue(LintIssue(
                            category="layout",
                            severity="ERROR" if over_x > 8.0 else "WARNING",
                            file=s_file or "Physical-AI.pdf",
                            line=s_line,
                            page=book_page_num,
                            message=f"Margin note protrudes {over_x:.1f}pt past right paper boundary. Fix: wrap code or long URLs in \\path{{...}}.",
                            context=f"Book Page {book_page_num} (PDF p.{page_num}) | '{clean_txt[:60]}...'"
                        ))
                    elif not is_recto and x0 < VERSO_MARGIN_LEFT - 4.0:
                        over_x = VERSO_MARGIN_LEFT - x0
                        s_file, s_line = self._locate_source(clean_txt, source_index)
                        report.add_issue(LintIssue(
                            category="layout",
                            severity="ERROR" if over_x > 8.0 else "WARNING",
                            file=s_file or "Physical-AI.pdf",
                            line=s_line,
                            page=book_page_num,
                            message=f"Margin note protrudes {over_x:.1f}pt past left paper boundary. Fix: wrap code or tighten note width.",
                            context=f"Book Page {book_page_num} (PDF p.{page_num}) | '{clean_txt[:60]}...'"
                        ))

                # --- 2. Body Horizontal Overrun Checks ---
                if is_body:
                    body_block_count += 1
                    max_body_y = max(max_body_y, y1)

                    if is_recto and x1 > RECTO_BODY_RIGHT + 18.0 and x0 < RECTO_BODY_RIGHT:
                        if x1 < 490.0:  # Exclude intentional fullwidth blocks
                            over_x = x1 - (RECTO_BODY_RIGHT + 12.0)
                            s_file, s_line = self._locate_source(clean_txt, source_index)
                            report.add_issue(LintIssue(
                                category="layout",
                                severity="ERROR" if over_x > 15.0 else "WARNING",
                                file=s_file or "Physical-AI.pdf",
                                line=s_line,
                                page=book_page_num,
                                message=f"Body text/equation spills {over_x:.1f}pt into margin gutter. Fix: wrap equation in \\begin{{aligned}} or split line.",
                                context=f"Book Page {book_page_num} (PDF p.{page_num}) | '{clean_txt[:60]}...'"
                            ))

            # --- 3. Sidenote-on-Sidenote Collision Detection ---
            margin_blocks.sort(key=lambda m: m[0])
            for i in range(len(margin_blocks) - 1):
                b_curr = margin_blocks[i]
                b_next = margin_blocks[i + 1]
                # If next block starts before current block ends (with 4pt tolerance)
                if b_next[0] < b_curr[1] - 4.0:
                    overlap_pts = b_curr[1] - b_next[0]
                    s_file, s_line = self._locate_source(b_next[4], source_index)
                    report.add_issue(LintIssue(
                        category="layout",
                        severity="ERROR",
                        file=s_file or "Physical-AI.pdf",
                        line=s_line,
                        page=book_page_num,
                        message=f"Margin note collision! Sidenote overlaps preceding note by {overlap_pts:.1f}pt. Fix: move footnote reference further down in prose or adjust vertical offset.",
                        context=f"Book Page {book_page_num} (PDF p.{page_num}) | Top note: '{b_curr[4][:40]}...' | Overlapping note: '{b_next[4][:40]}...'"
                    ))

            # --- 4. Premature Page Breaks (Stranded Whitespace) ---
            if (not is_chapter_opener) and (body_block_count >= 2) and (max_body_y < 490.0) and (page_num < total_pages):
                next_page = doc[page_num]
                next_blocks = next_page.get_text("blocks")
                next_first_elem = "Unknown element"
                for nb in next_blocks:
                    if nb[1] > HEADER_BOTTOM:
                        next_first_elem = " ".join(nb[4].split())[:80]
                        break

                if "Chapter " not in next_first_elem and "Part " not in next_first_elem:
                    white_gap = USABLE_BOTTOM - max_body_y
                    s_file, s_line = self._locate_source(next_first_elem, source_index)
                    report.add_issue(LintIssue(
                        category="layout",
                        severity="WARNING",
                        file=s_file or "Physical-AI.pdf",
                        line=s_line,
                        page=book_page_num,
                        message=f"Premature page break: {white_gap:.1f}pt stranded whitespace. Culprit on next page: '{next_first_elem[:50]}...'. Fix: relax table Needspace or adjust float position.",
                        context=f"Book Page {book_page_num} (PDF p.{page_num}) | Next element starts: '{next_first_elem[:60]}...'"
                    ))

        doc.close()


def render_flagged_contact_sheets(ctx: BookContext, report: LintReport, output_dir: str = "scratch/layout_audit"):
    """Render high-resolution spread contact sheets for all flagged pages."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL (Pillow) is not installed; skipping contact sheet rendering.")
        return

    out_path = ctx.repo_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    flagged_pages = sorted(list(set(issue.page for issue in report.issues if issue.page is not None)))
    if not flagged_pages:
        print("No layout issues with page numbers to render.")
        return

    import subprocess
    print(f"\n🖼️  Rendering spread contact sheets for {len(flagged_pages)} flagged page(s)...")

    spreads = set()
    for p in flagged_pages:
        if p % 2 == 1:
            spreads.add((max(1, p - 1), p))
        else:
            spreads.add((p, p + 1))

    for left_p, right_p in sorted(spreads):
        prefix = str(out_path / f"sheet_{left_p:03d}_{right_p:03d}")
        cmd = ["pdftoppm", "-png", "-r", "150", "-f", str(left_p), "-l", str(right_p), str(ctx.pdf_path), prefix]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rendered_files = sorted(glob.glob(f"{prefix}-*.png"))
            if len(rendered_files) >= 2:
                im_l = Image.open(rendered_files[0])
                im_r = Image.open(rendered_files[1])
                pw, ph = im_l.size
                sp_w = pw * 2 + 40
                sp_h = ph + 90
                sp_img = Image.new("RGB", (sp_w, sp_h), color=(240, 240, 243))
                draw = ImageDraw.Draw(sp_img)
                sp_img.paste(im_l, (20, 70))
                sp_img.paste(im_r, (pw + 20, 70))

                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
                except Exception:
                    font = ImageFont.load_default()

                draw.text((30, 20), f"Layout Audit Spread: Book Page {left_p} (Left) & Page {right_p} (Right)", fill=(20, 20, 20), font=font)
                spread_filename = out_path / f"spread_page_{left_p:03d}_{right_p:03d}.png"
                sp_img.save(spread_filename)
                for rf in rendered_files:
                    try:
                        os.remove(rf)
                    except OSError:
                        pass
                print(f"  • Saved contact sheet: {spread_filename}")
        except Exception as e:
            print(f"  Error rendering spread {left_p}-{right_p}: {e}")

