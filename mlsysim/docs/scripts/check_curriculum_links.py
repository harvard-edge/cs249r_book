#!/usr/bin/env python3
"""Validate all textbook chapter, interactive lab, slide, and solver links in mlsysim/docs/.

Checks:
  1. Every volume/chapter entry in ``config/curriculum_links.yml`` matches the
     canonical chapter list in ``books/config/_quarto-html-vol{1,2,3,4}.yml``.
  2. Every lab slug in ``config/curriculum_links.yml`` exists in
     ``labs/config/lab-inventory.qmd``.
  3. Every slide anchor ``#ch-NN`` exists in ``slides/tools/lectures.py``.
  4. Every tutorial, solver, and ref in ``config/curriculum_links.yml`` resolves
     to an existing file / exported solver / canonical chapter.
  5. Every ``https://mlsysbook.ai/vol{1,2,3,4}/...`` and
     ``https://mlsysbook.ai/labs/vol{1,2,3,4}/...`` link in ``mlsysim/docs/**/*.{qmd,md}``
     uses the canonical ``{NN_slug}/{NN_slug}.html`` / ``lab_.../`` path (never
     legacy unprefixed redirect stubs).
  6. If ``mlsysim/docs/_build/`` exists, every ``vol3`` or ``vol4`` book, lab,
     or slide link in rendered HTML sits inside a ``.mls-unreleased`` ancestor.
"""

from __future__ import annotations

import html.parser
import re
import sys
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent.parent
REPO = DOCS.parent.parent
sys.path.insert(0, str(DOCS / "scripts"))
sys.path.insert(0, str(DOCS.parent))

import curriculum_links  # noqa: E402
import mlsysim.solvers  # noqa: E402


class _UnreleasedLinkChecker(html.parser.HTMLParser):
    """Detect any vol3/vol4 link outside an element carrying class='mls-unreleased'."""

    UNRELEASED_RE = re.compile(
        r"(?:https://mlsysbook\.ai)?/(?:vol[34]/|labs/vol[34]/|slides/vol[34]\.html)"
    )

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, bool]] = []
        self.violations: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        classes = (attr_map.get("class") or "").split()
        in_unreleased = ("mls-unreleased" in classes) or any(flag for _, flag in self.stack)
        # Navigation chrome is already handled by shared/config/site-head.html :has() rules
        # (.navbar, .sidebar, .nav-page — NOT arbitrary <nav> elements like 404.qmd's .j404-nav)
        in_nav = any(
            c in classes for c in ("navbar", "sidebar", "nav-page")
        ) or any(flag for tag_name, flag in self.stack if tag_name in ("__nav_chrome__",))
        is_nav_chrome = any(c in classes for c in ("navbar", "sidebar", "nav-page"))
        effective_safe = in_unreleased or in_nav or is_nav_chrome
        if tag not in ("br", "hr", "img", "meta", "link", "input"):
            self.stack.append(("__nav_chrome__" if is_nav_chrome else tag, effective_safe))
        if tag == "a":
            href = attr_map.get("href") or ""
            if self.UNRELEASED_RE.search(href) and not effective_safe:
                self.violations.append(href)

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def check_all() -> list[str]:
    errors: list[str] = []
    manifest = curriculum_links.load_manifest()

    # 1. Validate canonical book chapters against books/config/_quarto-html-volN.yml
    canonical_book_urls: set[str] = set()
    for vol_key, vol_data in manifest["volumes"].items():
        cfg_path = REPO / "books" / "config" / f"_quarto-html-{vol_key}.yml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            pat = re.compile(rf"^{vol_key}/((\d+_[^/]+)/\2)\.qmd$")
            actual_slugs = set()
            for qmd in (cfg.get("project") or {}).get("render") or []:
                m = pat.match(str(qmd))
                if m:
                    actual_slugs.add(m.group(2))
            manifest_slugs = set(vol_data["chapters"].keys())
            if manifest_slugs != actual_slugs:
                missing = sorted(actual_slugs - manifest_slugs)
                extra = sorted(manifest_slugs - actual_slugs)
                errors.append(
                    f"{vol_key} chapter mismatch vs {cfg_path.name}: missing={missing}, extra={extra}"
                )
        for slug in vol_data["chapters"]:
            canonical_book_urls.add(f"{vol_data['book_base']}/{slug}/{slug}.html")
        # Each volume's Study Sheets page (added 2026-11-24, when study sheets
        # replaced Labs on the live site).
        canonical_book_urls.add(f"{vol_data['book_base']}/backmatter/studysheets.html")

    # 2. Validate canonical lab slugs against labs/config/lab-inventory.qmd
    canonical_lab_urls: set[str] = set()
    lab_inv_path = REPO / "labs" / "config" / "lab-inventory.qmd"
    actual_labs: set[str] = set()
    if lab_inv_path.exists():
        actual_labs = set(
            re.findall(r'href="(vol[1-4]/lab_[^"/]+)/"', lab_inv_path.read_text(encoding="utf-8"))
        )
    for vol_key, vol_data in manifest["volumes"].items():
        for slug, ch in vol_data["chapters"].items():
            rel_lab = f"{vol_key}/{ch['lab']}"
            if actual_labs and rel_lab not in actual_labs:
                errors.append(f"Lab {rel_lab!r} (for {vol_key}.{slug}) not in lab-inventory.qmd")
            canonical_lab_urls.add(f"{vol_data['labs_base']}/{ch['lab']}/")
        # Also allow onboarding lab_00_*
        for lab_rel in actual_labs:
            if lab_rel.startswith(f"{vol_key}/"):
                canonical_lab_urls.add(f"https://mlsysbook.ai/labs/{lab_rel}/")

    # 3. Validate slide anchors via slides/tools/lectures.py
    lectures_dir = REPO / "slides" / "tools"
    if lectures_dir.exists():
        sys.path.insert(0, str(lectures_dir))
        import lectures  # type: ignore

        for vol_key, vol_data in manifest["volumes"].items():
            valid_anchors = {
                c["anchor"]
                for c in lectures.volume(vol_key, vol_data["slides_page"])
                if c["chapter"]
            }
            for slug, ch in vol_data["chapters"].items():
                anchor = f"ch-{int(ch['num']):02d}"
                if anchor not in valid_anchors:
                    errors.append(f"Slide anchor #{anchor} for {vol_key}.{slug} missing in slides/{vol_key}")

    # 4. Validate tutorial & thread references in manifest
    exported_solvers = set(mlsysim.solvers.__all__)
    for tut_key, tut_entry in manifest["tutorials"].items():
        tut_path = DOCS / "tutorials" / f"{tut_key}.qmd"
        if not tut_path.exists():
            errors.append(f"Tutorial file missing for manifest key {tut_key!r}: {tut_path}")
        for link in tut_entry.get("links", []):
            try:
                curriculum_links.resolve_chapter(link["ref"])
            except Exception as exc:
                errors.append(f"Invalid chapter ref {link.get('ref')!r} in tutorial {tut_key}: {exc}")

    for thread in manifest["threads"]:
        for s in thread.get("solvers", []):
            if s not in exported_solvers:
                errors.append(f"Thread {thread['id']} references unknown solver {s!r}")
        for t in thread.get("tutorials", []):
            if not (DOCS / t["href"]).exists():
                errors.append(f"Thread {thread['id']} references missing tutorial {t['href']!r}")
        for stop in thread.get("stops", []):
            try:
                curriculum_links.resolve_chapter(stop["ref"])
            except Exception as exc:
                errors.append(f"Invalid stop ref {stop.get('ref')!r} in thread {thread['id']}: {exc}")

    for ch_ref, (solvers_csv, guides) in curriculum_links._CHAPTER_TOOLS.items():
        try:
            curriculum_links.resolve_chapter(ch_ref)
        except Exception as exc:
            errors.append(f"_CHAPTER_TOOLS invalid chapter ref {ch_ref!r}: {exc}")
        for s in (x.strip() for x in solvers_csv.split(",") if x.strip()):
            if s not in exported_solvers:
                errors.append(f"_CHAPTER_TOOLS[{ch_ref}] unknown solver {s!r}")
        for _, href in guides:
            if not href.startswith("api/") and not (DOCS / href).exists():
                errors.append(f"_CHAPTER_TOOLS[{ch_ref}] missing guide {href!r}")

    # 5. Scan all source .qmd/.md files in mlsysim/docs/ for non-canonical book/lab URLs
    book_url_re = re.compile(r"https://mlsysbook\.ai/vol[1-4]/[A-Za-z0-9_./-]+")
    lab_url_re = re.compile(r"https://mlsysbook\.ai/labs/vol[1-4]/[A-Za-z0-9_./-]+")
    for src in sorted([*DOCS.rglob("*.qmd"), *DOCS.rglob("*.md")]):
        if "_build" in src.parts or ".quarto" in src.parts:
            continue
        text = src.read_text(encoding="utf-8")
        for m in book_url_re.finditer(text):
            url = m.group(0).rstrip(".,;)")
            if url.rstrip("/") in {f"https://mlsysbook.ai/vol{i}" for i in (1, 2, 3, 4)}:
                continue
            if url not in canonical_book_urls:
                errors.append(f"{src.relative_to(DOCS)}: non-canonical book chapter URL {url!r}")
        for m in lab_url_re.finditer(text):
            url = m.group(0).rstrip(".,;)")
            if not url.endswith("/"):
                url_check = url + "/"
            else:
                url_check = url
            if url_check not in canonical_lab_urls:
                errors.append(f"{src.relative_to(DOCS)}: unknown lab URL {url!r}")

    # 6. If _build/ exists, verify every vol3/vol4 link is gated by .mls-unreleased
    build_dir = DOCS / "_build"
    if build_dir.is_dir():
        for html_file in sorted(build_dir.rglob("*.html")):
            checker = _UnreleasedLinkChecker()
            checker.feed(html_file.read_text(encoding="utf-8", errors="replace"))
            for href in checker.violations:
                errors.append(
                    f"_build/{html_file.relative_to(build_dir)}: ungated vol3/vol4 link {href!r} (wrap in .mls-unreleased)"
                )

    return errors


def main() -> int:
    errors = check_all()
    if errors:
        print("Curriculum link validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("Curriculum link validation OK: all chapter, lab, slide, and solver links verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
