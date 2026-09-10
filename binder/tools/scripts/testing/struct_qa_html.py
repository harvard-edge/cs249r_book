#!/usr/bin/env python3
"""struct_qa_html.py

Structural QA pass over a built HTML site for the MLSysBook camera-ready
sweep. Walks every chapter HTML under --html-dir and emits one JSON line
per chapter to --out, summarising image, anchor, math, and leaked-token
checks.

Stdlib only. BeautifulSoup is used opportunistically if available; otherwise
the script falls back to regex parsing (less precise but still useful).

Usage:
    python3 struct_qa_html.py --vol vol1 \
        --html-dir /abs/path/to/_build/html-vol1 \
        --out      /abs/path/to/run-dir/qa/structural-html-vol1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from bs4 import BeautifulSoup  # type: ignore
    HAVE_BS4 = True
except Exception:  # pragma: no cover
    HAVE_BS4 = False


# ---------------------------------------------------------------------------
# Regex fallbacks
# ---------------------------------------------------------------------------
RE_IMG = re.compile(r"<img\b([^>]*)>", re.I)
RE_FIGURE = re.compile(r"<figure\b[^>]*>(.*?)</figure>", re.I | re.S)
RE_FIGCAPTION = re.compile(r"<figcaption\b", re.I)
RE_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
RE_INTERNAL_ANCHOR = re.compile(r'<a\b[^>]*href="#([^"]+)"', re.I)
RE_ID_ATTR = re.compile(r'\bid="([^"]+)"', re.I)
RE_LEAKED_LATEX = re.compile(r"\$\$|\\\(|\\\[|\\begin\{equation\}")
RE_MATH_CONTAINER = re.compile(r'class="[^"]*\bmath\b[^"]*"', re.I)
RE_LEAKED_QUARTO = re.compile(r"\[@|\?@|@fig-|@sec-|@tbl-|@eq-")
RE_CODE_BLOCK = re.compile(r"<(pre|code)[^>]*>(.*?)</\1>", re.I | re.S)
RE_MATH_BLOCK = re.compile(
    r'<(?:span|div)\b[^>]*class="[^"]*\bmath\b[^"]*"[^>]*>(.*?)</(?:span|div)>',
    re.I | re.S,
)


def strip_html_for_text_scan(html: str) -> str:
    """Remove <pre>/<code>/math containers before scanning for leaked tokens."""
    cleaned = RE_CODE_BLOCK.sub(" ", html)
    cleaned = RE_MATH_BLOCK.sub(" ", cleaned)
    return cleaned


def collect_imgs_with_bs4(soup) -> List[Dict]:
    out = []
    for img in soup.find_all("img"):
        out.append(
            {
                "src": img.get("src", ""),
                "alt": (img.get("alt") or "").strip(),
            }
        )
    return out


def collect_imgs_with_regex(html: str) -> List[Dict]:
    out = []
    for m in RE_IMG.finditer(html):
        attrs = dict(RE_ATTR.findall(m.group(1)))
        out.append({"src": attrs.get("src", ""), "alt": attrs.get("alt", "").strip()})
    return out


def collect_figures_caption_status(html: str, soup=None) -> List[bool]:
    """Return a list of booleans, one per <figure>, True iff has <figcaption>."""
    if soup is not None:
        return [fig.find("figcaption") is not None for fig in soup.find_all("figure")]
    return [bool(RE_FIGCAPTION.search(body)) for body in RE_FIGURE.findall(html)]


def collect_internal_anchors(html: str, soup=None) -> List[str]:
    if soup is not None:
        return [
            a.get("href", "")[1:]
            for a in soup.find_all("a", href=True)
            if a.get("href", "").startswith("#")
        ]
    return RE_INTERNAL_ANCHOR.findall(html)


def collect_ids(html: str, soup=None) -> set:
    if soup is not None:
        return {tag.get("id") for tag in soup.find_all(id=True) if tag.get("id")}
    return set(RE_ID_ATTR.findall(html))


def has_math_container(html: str) -> bool:
    return bool(RE_MATH_CONTAINER.search(html))


def chapter_name_from_path(html_path: Path, html_root: Path) -> str:
    rel = html_path.relative_to(html_root)
    parts = list(rel.parts)
    # Typical path: contents/<vol>/<group>/<chapter>/<chapter>.html
    # Use the chapter directory name when available; else fallback to stem.
    if len(parts) >= 2:
        return parts[-2] if parts[-1].endswith(".html") else parts[-1]
    return html_path.stem


def gather_global_ids(html_files: List[Path]) -> set:
    ids = set()
    for f in html_files:
        try:
            html = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if HAVE_BS4:
            soup = BeautifulSoup(html, "html.parser")
            ids.update(collect_ids(html, soup))
        else:
            ids.update(collect_ids(html))
    return ids


def check_chapter(html_path: Path, html_root: Path, global_ids: set, vol: str) -> Dict:
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "chapter": chapter_name_from_path(html_path, html_root),
            "vol": vol,
            "format": "html",
            "checks": {"read_error": str(e)},
            "ok": False,
        }

    soup = BeautifulSoup(html, "html.parser") if HAVE_BS4 else None

    # Images
    imgs = collect_imgs_with_bs4(soup) if soup else collect_imgs_with_regex(html)
    broken_imgs: List[str] = []
    missing_alt: List[str] = []
    base_dir = html_path.parent
    for img in imgs:
        src = img["src"]
        if not src or src.startswith(("http://", "https://", "data:", "//")):
            if not img["alt"]:
                missing_alt.append(src or "<no-src>")
            continue
        # strip url fragment / query
        clean = src.split("#", 1)[0].split("?", 1)[0]
        if clean.startswith("/"):
            # site-absolute; resolve relative to html_root
            target = (html_root / clean.lstrip("/")).resolve()
        else:
            target = (base_dir / clean).resolve()
        if not target.exists() or target.stat().st_size == 0:
            broken_imgs.append(src)
        if not img["alt"]:
            missing_alt.append(src)

    # Figures
    fig_caption_present = collect_figures_caption_status(html, soup)
    missing_captions = [
        f"figure-{i}" for i, has_cap in enumerate(fig_caption_present) if not has_cap
    ]

    # Leaked LaTeX & Quarto tokens (scan body text only, exclude code/math)
    body_text = strip_html_for_text_scan(html)
    leaked_latex = sorted({m.group(0) for m in RE_LEAKED_LATEX.finditer(body_text)})
    leaked_quarto = sorted({m.group(0) for m in RE_LEAKED_QUARTO.finditer(body_text)})

    # Internal anchor resolution
    anchor_targets = collect_internal_anchors(html, soup)
    broken_anchors = [a for a in anchor_targets if a and a not in global_ids]

    # Math container presence (only meaningful if equations exist in source --
    # we approximate by checking if body has any math markup tokens at all).
    expected_math = bool(re.search(r"\\\(|\\\[|\$\$|\\begin\{equation\}", html))
    math_container_ok = (not expected_math) or has_math_container(html)

    checks = {
        "images_ok": len(broken_imgs) == 0,
        "broken_imgs": broken_imgs,
        "missing_alt": missing_alt,
        "missing_captions": missing_captions,
        "leaked_latex": leaked_latex,
        "leaked_quarto": leaked_quarto,
        "broken_anchors": broken_anchors,
        "math_container_ok": math_container_ok,
    }
    ok = (
        checks["images_ok"]
        and not missing_alt
        and not missing_captions
        and not leaked_latex
        and not leaked_quarto
        and not broken_anchors
        and math_container_ok
    )

    return {
        "chapter": chapter_name_from_path(html_path, html_root),
        "vol": vol,
        "format": "html",
        "checks": checks,
        "ok": ok,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vol", required=True, choices=["vol1", "vol2"])
    p.add_argument("--html-dir", required=True, help="abs path to built HTML root")
    p.add_argument("--out", required=True, help="abs path to JSONL output file")
    args = p.parse_args()

    html_root = Path(args.html_dir).resolve()
    if not html_root.is_dir():
        print(f"struct_qa_html: html-dir not found: {html_root}", file=sys.stderr)
        return 2

    html_files = sorted(html_root.glob("contents/**/*.html"))
    if not html_files:
        # also accept flat layout
        html_files = sorted(html_root.glob("**/*.html"))

    global_ids = gather_global_ids(html_files)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    with out_path.open("w", encoding="utf-8") as f:
        for html_path in html_files:
            rec = check_chapter(html_path, html_root, global_ids, args.vol)
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if rec.get("ok"):
                n_ok += 1
            else:
                n_bad += 1
    print(f"struct_qa_html: {args.vol} ok={n_ok} bad={n_bad} total={n_ok+n_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
