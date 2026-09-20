#!/usr/bin/env python3
"""
generate_figure_contact_sheet.py
================================
Extracts all figures (native TikZ, margin visual aids, body images, and cover art)
from any volume of the MLSysBook and generates:
1. A consolidated standalone Quarto/LaTeX document (`figures.qmd`).
2. A high-resolution PDF contact sheet (`figures.pdf`) where each figure is rendered
   with its metadata, caption, label, and source line.
3. Grid thumbnail contact sheets (`contact-sheets/sheet-*.png`) for rapid visual scanning.
4. Structured metadata export (`figures.json` and `figures.csv`).

Usage:
  python3 scripts/generate_figure_contact_sheet.py --vol2
  python3 scripts/generate_figure_contact_sheet.py --vol2 --chapter 07_fault_tolerance
  python3 scripts/generate_figure_contact_sheet.py --vol2 --type tikz --limit 10
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FigureEntry:
    index: int
    category: str       # 'tikz', 'margin', 'body_image', 'cover', 'python'
    label: str          # e.g. 'fig-young-daly'
    chapter: str        # Chapter title
    chapter_slug: str   # Directory or file stem
    source_file: str    # Relative path from repo root
    source_line: int    # 1-indexed line number in source
    caption: str        # Caption text
    alt_text: str       # Alt text if present
    content: str        # TikZ code, or image file path relative to output QMD
    raw_snippet: str = ""


class FigureContactSheetBuilder:
    def __init__(
        self,
        volume: str = "vol2",
        chapter_filter: Optional[List[str]] = None,
        figure_type: str = "all",
        limit: int = 0,
        out_dir: Optional[Path] = None,
        dpi: int = 110,
        cols: int = 3,
        rows: int = 4,
    ):
        self.volume = volume.lower()
        self.chapter_filter = chapter_filter or []
        self.figure_type = figure_type.lower()
        self.limit = limit
        self.dpi = dpi
        self.cols = cols
        self.rows = rows

        self.vol_dir = REPO_ROOT / "books" / self.volume
        if not self.vol_dir.exists():
            raise ValueError(f"Volume directory does not exist: {self.vol_dir}")

        self.out_dir = out_dir or (REPO_ROOT / "binder" / ".layout" / "figures" / self.volume)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.theme_tex = REPO_ROOT / "books" / "shared" / "tex" / f"theme-colors-{self.volume}.tex"
        if not self.theme_tex.exists():
            self.theme_tex = REPO_ROOT / "books" / "shared" / "tex" / "theme-colors-vol2.tex"

        self.header_tex = REPO_ROOT / "books" / "shared" / "tex" / "header-includes.tex"

    def collect_source_files(self) -> List[Path]:
        """Collect source QMD files in canonical book order."""
        ordered = []
        config_path = REPO_ROOT / "books" / "config" / f"_quarto-pdf-{self.volume}.yml"

        if config_path.exists():
            try:
                import yaml
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                book = raw.get("book", {})
                for section in ("chapters", "appendices"):
                    for entry in book.get(section, []) or []:
                        path_str = entry.get("file", "") if isinstance(entry, dict) else entry
                        if isinstance(path_str, str) and path_str.endswith(".qmd"):
                            p = (REPO_ROOT / "books" / path_str).resolve()
                            if p.exists() and p.name not in {"index.qmd", "references.qmd"}:
                                ordered.append(p)
            except Exception as e:
                print(f"Warning: could not parse {config_path.name}: {e}", file=sys.stderr)

        # Also discover any unlisted QMD files
        for p in sorted(self.vol_dir.rglob("*.qmd")):
            if p.name.startswith("_") or p.name in {"index.qmd", "references.qmd"}:
                continue
            if p not in ordered:
                ordered.append(p)

        # Apply chapter filter if present
        if self.chapter_filter:
            filtered = []
            for p in ordered:
                stem = p.stem.lower()
                parent_stem = p.parent.name.lower()
                if any(f.lower() in stem or f.lower() in parent_stem for f in self.chapter_filter):
                    filtered.append(p)
            return filtered

        return ordered

    def extract_figures(self) -> List[FigureEntry]:
        """Extract all visual figures from the volume QMD files."""
        qmd_files = self.collect_source_files()
        entries: List[FigureEntry] = []
        idx = 1

        for qmd in qmd_files:
            rel_qmd = str(qmd.relative_to(REPO_ROOT))
            content = qmd.read_text(encoding="utf-8")
            lines = content.splitlines()
            n = len(lines)
            i = 0

            # Extract chapter title
            ch_title = qmd.stem.replace("_", " ").title()
            for l in lines[:40]:
                tm = re.match(r"^#\s+(.+)$", l)
                if tm:
                    ch_title = tm.group(1).strip()
                    break

            while i < n:
                line = lines[i]

                # 1. Main-text Figure Div: ::: {#fig-...}
                if re.match(r"^:::\s*\{\s*#fig-[A-Za-z0-9_-]+", line):
                    start_line = i + 1
                    header_line = line

                    label_m = re.search(r"#(fig-[A-Za-z0-9_-]+)", header_line)
                    label = label_m.group(1) if label_m else f"fig-unknown-{start_line}"

                    cap_m = re.search(r'fig-cap=(["\'])(.*?)\1', header_line)
                    caption = cap_m.group(2) if cap_m else ""

                    alt_m = re.search(r'fig-alt=(["\'])(.*?)\1', header_line)
                    alt = alt_m.group(2) if alt_m else ""

                    # Collect body lines
                    depth = 1
                    body_lines = []
                    i += 1
                    while i < n:
                        cur = lines[i]
                        if cur.startswith(":::"):
                            depth -= 1
                            if depth == 0:
                                break
                        elif cur.startswith("::: {") or cur.startswith(":::{"):
                            depth += 1
                        body_lines.append(cur)
                        i += 1

                    body_text = "\n".join(body_lines).strip()

                    if r"\begin{tikzpicture}" in body_text:
                        cat = "tikz"
                        if self._type_matches(cat):
                            entries.append(FigureEntry(
                                index=idx,
                                category=cat,
                                label=label,
                                chapter=ch_title,
                                chapter_slug=qmd.parent.name,
                                source_file=rel_qmd,
                                source_line=start_line,
                                caption=caption,
                                alt_text=alt,
                                content=body_text,
                            ))
                            idx += 1
                    elif re.search(r"!\[.*?\]\((.*?)\)", body_text):
                        cat = "body_image"
                        if self._type_matches(cat):
                            img_m = re.search(r"!\[(.*?)\]\((.*?)\)", body_text)
                            raw_path = img_m.group(2).split()[0]
                            resolved_path = (qmd.parent / raw_path).resolve()
                            rel_to_out = os.path.relpath(resolved_path, self.out_dir)
                            entries.append(FigureEntry(
                                index=idx,
                                category=cat,
                                label=label,
                                chapter=ch_title,
                                chapter_slug=qmd.parent.name,
                                source_file=rel_qmd,
                                source_line=start_line,
                                caption=caption or img_m.group(1),
                                alt_text=alt,
                                content=rel_to_out,
                            ))
                            idx += 1
                    elif "```{python}" in body_text:
                        cat = "python"
                        if self._type_matches(cat):
                            entries.append(FigureEntry(
                                index=idx,
                                category=cat,
                                label=label,
                                chapter=ch_title,
                                chapter_slug=qmd.parent.name,
                                source_file=rel_qmd,
                                source_line=start_line,
                                caption=caption,
                                alt_text=alt,
                                content=body_text,
                            ))
                            idx += 1

                    i += 1
                    continue

                # 2. Margin visual notes: ::: {.column-margin}
                if "::: {.column-margin}" in line or ":::{.column-margin}" in line:
                    start_line = i + 1
                    depth = 1
                    body_lines = []
                    i += 1
                    while i < n:
                        cur = lines[i]
                        if cur.startswith(":::"):
                            depth -= 1
                            if depth == 0:
                                break
                        elif cur.startswith("::: {") or cur.startswith(":::{"):
                            depth += 1
                        body_lines.append(cur)
                        i += 1

                    body_text = "\n".join(body_lines).strip()
                    img_m = re.search(r"!\[(.*?)\]\((.*?)\)", body_text)
                    if img_m:
                        cat = "margin"
                        if self._type_matches(cat):
                            raw_path = img_m.group(2).split()[0]
                            resolved_path = (qmd.parent / raw_path).resolve()
                            rel_to_out = os.path.relpath(resolved_path, self.out_dir)
                            entries.append(FigureEntry(
                                index=idx,
                                category=cat,
                                label=f"margin-{start_line}",
                                chapter=ch_title,
                                chapter_slug=qmd.parent.name,
                                source_file=rel_qmd,
                                source_line=start_line,
                                caption=img_m.group(1),
                                alt_text="",
                                content=rel_to_out,
                            ))
                            idx += 1

                    i += 1
                    continue

                # 3. Chapter Cover Blueprint
                if "cover_" in line and "![" in line:
                    cat = "cover"
                    if self._type_matches(cat) and not line.endswith(".webp)") and ".webp" not in line:
                        img_m = re.search(r"!\[(.*?)\]\((.*?)\)", line)
                        if img_m:
                            raw_path = img_m.group(2).split()[0]
                            resolved_path = (qmd.parent / raw_path).resolve()
                            rel_to_out = os.path.relpath(resolved_path, self.out_dir)
                            entries.append(FigureEntry(
                                index=idx,
                                category=cat,
                                label=f"cover-{qmd.stem}",
                                chapter=ch_title,
                                chapter_slug=qmd.parent.name,
                                source_file=rel_qmd,
                                source_line=i + 1,
                                caption="Chapter Cover Blueprint",
                                alt_text=img_m.group(1),
                                content=rel_to_out,
                            ))
                            idx += 1

                i += 1

        if self.limit > 0:
            entries = entries[:self.limit]

        return entries

    def _type_matches(self, cat: str) -> bool:
        if self.figure_type in {"all", ""}:
            return True
        if self.figure_type == "tikz":
            return cat == "tikz"
        if self.figure_type == "margin":
            return cat == "margin"
        if self.figure_type in {"images", "raster"}:
            return cat in {"body_image", "cover"}
        return cat == self.figure_type

    def generate_qmd(self, entries: List[FigureEntry]) -> Path:
        """Generate the figures.qmd file for Quarto rendering."""
        qmd_path = self.out_dir / "figures.qmd"

        lines = [
            "---",
            "format:",
            "  pdf:",
            "    documentclass: scrbook",
            "    classoption:",
            "      - oneside",
            "    geometry:",
            "      - margin=0.75in",
            "    pdf-engine: lualatex",
            "    keep-tex: true",
            "    number-sections: false",
            "    include-in-header:",
            f'      - file: "{self.theme_tex}"',
            f'      - file: "{self.header_tex}"',
            "execute:",
            "  enabled: false",
            "---",
            "",
            "\\pagestyle{empty}",
            "",
            f"# {self.volume.upper()} Visual Figure Contact Sheet",
            f"**Total Figures Included:** {len(entries)}",
            "",
        ]

        for entry in entries:
            badge_color = {
                "tikz": "mlsysblue",
                "margin": "mlsysgreen",
                "body_image": "mlsyspurple",
                "cover": "mlsysamber",
                "python": "mlsysgray"
            }.get(entry.category, "black")

            category_name = {
                "tikz": "TikZ Vector Diagram",
                "margin": "Margin Visual Note",
                "body_image": "Body Image / Photo",
                "cover": "Cover Blueprint",
                "python": "Python Matplotlib Code"
            }.get(entry.category, entry.category.upper())

            lines.extend([
                "\\clearpage",
                "```{=latex}",
                f"\\noindent{{\\bfseries\\large {entry.index:03d} \\quad \\color{{{badge_color}}}[{category_name}] \\quad \\texttt{{{self._latex_escape(entry.label)}}}}}\\\\[3pt]",
                f"\\noindent{{\\small\\color{{gray}} Chapter: \\textbf{{{self._latex_escape(entry.chapter)}}} \\quad $\\cdot$ \\quad Source: \\texttt{{{self._latex_escape(entry.source_file)}:{entry.source_line}}}}}\\\\[6pt]",
                "```",
                ""
            ])

            if entry.caption:
                lines.extend([
                    f"**Caption:** {entry.caption}",
                    ""
                ])

            # Render content based on category
            if entry.category == "tikz":
                # Clean TikZ code of markdown fences
                clean_code = re.sub(r"^```.*?$", "", entry.content, flags=re.MULTILINE).strip()
                lines.extend([
                    "```{=latex}",
                    "\\begin{center}",
                    "\\adjustbox{max width=\\linewidth, max totalheight=0.68\\textheight, keepaspectratio}{%",
                    clean_code,
                    "}%",
                    "\\end{center}",
                    "```",
                    ""
                ])
            elif entry.category in {"margin", "body_image", "cover"}:
                lines.extend([
                    f"![]({entry.content}){{width=75% fig-align=\"center\"}}",
                    ""
                ])
            elif entry.category == "python":
                lines.extend([
                    "```python",
                    entry.content,
                    "```",
                    ""
                ])

        qmd_path.write_text("\n".join(lines), encoding="utf-8")
        return qmd_path

    @staticmethod
    def _latex_escape(text: str) -> str:
        text = text.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
        text = text.replace("#", "\\#").replace("_", "\\_")
        return text

    def render_pdf(self, qmd_path: Path) -> Optional[Path]:
        """Compile the figures.qmd to PDF with Quarto."""
        pdf_path = qmd_path.with_suffix(".pdf")
        log_path = self.out_dir / "quarto-render.log"

        print(f"Compiling figure contact sheet to {pdf_path.name} with Quarto...")
        proc = subprocess.run(
            ["quarto", "render", str(qmd_path), "--to", "pdf"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True
        )
        log_path.write_text(proc.stdout or "" + proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            print("Render failed. Last 20 lines of log:", file=sys.stderr)
            lines = (proc.stdout or "").splitlines()[-20:]
            print("\n".join(lines), file=sys.stderr)
            return None

        print(f"Render succeeded: {pdf_path}")
        return pdf_path

    def make_contact_sheets(self, pdf_path: Path) -> List[Path]:
        """Rasterize PDF pages and build thumbnail grid sheets."""
        try:
            import pypdfium2 as pdfium
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("pypdfium2 or Pillow not available; skipping thumbnail contact sheets.", file=sys.stderr)
            return []

        sheets_dir = self.out_dir / "contact-sheets"
        sheets_dir.mkdir(parents=True, exist_ok=True)
        for old in sheets_dir.glob("sheet-*.png"):
            old.unlink()

        pdf = pdfium.PdfDocument(str(pdf_path))
        num_pages = len(pdf)
        if num_pages <= 1:
            return []

        # Skip page 0 (title page)
        figure_pages = list(range(1, num_pages))
        page_w, page_h = None, None

        # Render first page to determine dimensions
        p1 = pdf[1].render(scale=self.dpi / 72.0).to_pil()
        thumb_w, thumb_h = p1.size

        # Grid configuration
        cols, rows = self.cols, self.rows
        per_sheet = cols * rows
        pad = 20
        banner_h = 40

        sheet_w = cols * thumb_w + (cols + 1) * pad
        sheet_h = rows * thumb_h + (rows + 1) * pad + banner_h

        sheet_paths = []
        num_sheets = (len(figure_pages) + per_sheet - 1) // per_sheet

        print(f"Generating {num_sheets} contact sheet(s) ({cols}x{rows} grid)...")

        for s_idx in range(num_sheets):
            sheet_img = Image.new("RGB", (sheet_w, sheet_h), color=(248, 250, 252))
            draw = ImageDraw.Draw(sheet_img)

            # Header banner
            header_text = f"{self.volume.upper()} Figure Contact Sheet — Sheet {s_idx + 1} of {num_sheets}"
            draw.text((pad, 12), header_text, fill=(30, 41, 59))

            batch = figure_pages[s_idx * per_sheet : (s_idx + 1) * per_sheet]
            for slot_idx, p_num in enumerate(batch):
                c = slot_idx % cols
                r = slot_idx // cols

                x = pad + c * (thumb_w + pad)
                y = banner_h + pad + r * (thumb_h + pad)

                # Render page
                page_img = pdf[p_num].render(scale=self.dpi / 72.0).to_pil()
                sheet_img.paste(page_img, (x, y))

                # Draw border around thumbnail
                draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(203, 213, 225), width=1)

            out_sheet = sheets_dir / f"sheet-{s_idx + 1:02d}.png"
            sheet_img.save(out_sheet)
            sheet_paths.append(out_sheet)

        return sheet_paths

    def export_metadata(self, entries: List[FigureEntry]):
        """Export JSON and CSV metadata reports."""
        json_path = self.out_dir / "figures.json"
        csv_path = self.out_dir / "figures.csv"

        data = [asdict(e) for e in entries]
        for d in data:
            d.pop("content", None)  # Don't dump full code into metadata summary

        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if data:
            fieldnames = ["index", "category", "label", "chapter", "chapter_slug", "source_file", "source_line", "caption", "alt_text"]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        print(f"Exported metadata to {json_path.name} and {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate volume figure contact sheets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vol1", dest="volume", action="store_const", const="vol1", help="Volume I")
    group.add_argument("--vol2", dest="volume", action="store_const", const="vol2", default="vol2", help="Volume II (default)")
    group.add_argument("--vol3", dest="volume", action="store_const", const="vol3", help="Volume III")
    group.add_argument("--vol4", dest="volume", action="store_const", const="vol4", help="Volume IV")

    parser.add_argument("--chapter", "-c", type=str, default="", help="Filter by chapter slug or stem (comma-separated)")
    parser.add_argument("--type", "-t", type=str, default="all", choices=["all", "tikz", "margin", "images", "cover"], help="Filter by figure type (default: all)")
    parser.add_argument("--limit", "-n", type=int, default=0, help="Limit to first N figures (0 = all)")
    parser.add_argument("--no-render", action="store_true", help="Only generate QMD, do not render PDF")
    parser.add_argument("--no-contact-sheets", action="store_true", help="Do not generate PNG contact sheets")
    parser.add_argument("--dpi", type=int, default=110, help="DPI for thumbnail contact sheets (default 110)")
    parser.add_argument("--cols", type=int, default=3, help="Columns in thumbnail contact sheet (default 3)")
    parser.add_argument("--rows", type=int, default=4, help="Rows in thumbnail contact sheet (default 4)")

    args = parser.parse_args()
    chapter_filter = [s.strip() for s in args.chapter.split(",") if s.strip()] if args.chapter else None

    builder = FigureContactSheetBuilder(
        volume=args.volume,
        chapter_filter=chapter_filter,
        figure_type=args.type,
        limit=args.limit,
        dpi=args.dpi,
        cols=args.cols,
        rows=args.rows,
    )

    print(f"=== MLSysBook Figure Contact Sheet: {args.volume.upper()} ===")
    entries = builder.extract_figures()
    print(f"Extracted {len(entries)} figure(s).")
    if not entries:
        print("No matching figures found.")
        return 0

    builder.export_metadata(entries)
    qmd_path = builder.generate_qmd(entries)
    print(f"Wrote audit document: {qmd_path}")

    if not args.no_render:
        pdf_path = builder.render_pdf(qmd_path)
        if pdf_path and not args.no_contact_sheets:
            sheets = builder.make_contact_sheets(pdf_path)
            for s in sheets:
                print(f"  Generated contact sheet: {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
