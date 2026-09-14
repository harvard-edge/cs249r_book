#!/usr/bin/env python3
"""Browser-level render verification for chapter HTML (MathJax + visible text)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # repo root
BOOK = Path(__file__).resolve().parents[2]  # book/
sys.path.insert(0, str(REPO_ROOT / "tools" / "audit"))
from audit_math_rendering import LEAK_PATTERNS  # noqa: E402

_FMT = REPO_ROOT / "binder" / "tools" / "audit" / "fmt"
sys.path.insert(0, str(_FMT))
from audit_prose import INLINE_PY, _exec_python_cells, _resolve_ref
from cell_exec import setup_headless_matplotlib

setup_headless_matplotlib()  # noqa: E402

ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)", re.I),
    re.compile(r"\bNameError:\b", re.I),
    re.compile(r"\bKeyError:\b", re.I),
    re.compile(r"\bAttributeError:\b", re.I),
    re.compile(r"Error rendering", re.I),
    re.compile(r"\{python\}"),
]
NUM_TOKEN = re.compile(r"\d[\d,]*\.?\d*")


def _normalize(s: str) -> str:
    """Normalize text for loose matching.

    Lowercases, drops ``$``, backslashes, braces, and commas, maps ``×`` to
    ``x``, and collapses whitespace.
    """
    s = (
        s.replace("\\$", "$").replace("$", "").replace("\\", "")
        .replace("{", "").replace("}", "").replace(",", "")
        .replace("\u00d7", "x").replace("×", "x").lower()
    )
    return re.sub(r"\s+", " ", s).strip()


def _value_in_text(value: str, text: str) -> bool:
    """Return True when a resolved inline value plausibly appears in the rendered text.

    Tries an exact substring match, then normalized matches. Failing those, it
    requires every significant numeric token (anything but a lone 0, 1, or 2)
    to appear, or, for values without one, the first 12 characters of the
    normalized alphanumeric form. Empty and ``<MISSING:...>`` values never match.
    """
    if not value or value.startswith("<MISSING:"):
        return False
    plain = value.strip()
    if plain in text:
        return True
    norm_t = _normalize(text)
    for cand in (plain, plain.replace(",", ""), _normalize(plain)):
        if cand and cand in norm_t:
            return True
    nums = NUM_TOKEN.findall(plain.replace(",", ""))
    sig = [n for n in nums if len(n.lstrip("-")) > 1 or n not in {"0", "1", "2"}]
    if sig:
        return all(n.replace(",", "") in norm_t or n in text for n in sig)
    chunk = re.sub(r"[^a-zA-Z0-9.%+-]", "", _normalize(plain))
    return bool(chunk and chunk[:12] in norm_t)


@dataclass
class ChapterBrowserReport:
    """Browser verification result for one chapter page, serialized into the JSON report.

    ``leaks`` stops at 50 entries, ``errors`` lists matched error patterns or a
    cell-execution failure, and ``ref_failures`` lists inline references whose
    values were not found in the page text.
    """

    chapter: str
    vol: str
    html: str
    qmd: str | None = None
    ok: bool = True
    leak_count: int = 0
    leaks: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    ref_failures: list = field(default_factory=list)
    refs_checked: int = 0


def _stem_from_artifact(name: str) -> str:
    """Strip the extension and any leading ``NN_`` order prefix from an artifact file name."""
    return re.sub(r"^\d+_", "", Path(name).stem)


def _find_qmd(vol: str, stem: str) -> Path | None:
    """Return the source QMD for a chapter stem, or None if absent.

    Appendix stems are looked up in ``backmatter/``.
    """
    base = REPO_ROOT  / "books" / vol
    if stem.startswith("appendix_"):
        p = base / "backmatter" / f"{stem}.qmd"
        return p if p.is_file() else None
    p = base / stem / f"{stem}.qmd"
    return p if p.is_file() else None


def _collect_inline_refs(qmd: Path) -> list[str]:
    """Return the unique inline ``{python}`` references in the prose of ``qmd``.

    References keep first-seen order; lines inside code cells are skipped.
    """
    refs: list[str] = []
    in_cell = False
    for line in qmd.read_text(encoding="utf-8").splitlines():
        if line.startswith("```{python}"):
            in_cell = True
            continue
        if in_cell and line.strip() == "```":
            in_cell = False
            continue
        if in_cell:
            continue
        refs.extend(INLINE_PY.findall(line))
    return list(dict.fromkeys(refs))


def verify_chapter_browser(page, html_path: Path, *, vol: str, check_refs: bool, screenshot_dir: Path | None) -> ChapterBrowserReport:
    """Load one HTML page in a Playwright ``page`` and check its rendered text.

    Waits for MathJax startup when present (a timeout is ignored), extracts the
    visible text of ``main`` while skipping code and math elements, and records
    LaTeX leaks and error patterns. With ``check_refs`` and a chapter QMD that
    has inline Python, it executes the QMD's cells and checks that each inline
    reference's value appears in the page; a cell failure marks the report
    failed and returns at once. Failing pages are screenshotted into
    ``screenshot_dir`` when given. ``html_path`` must lie inside the repository.
    """
    stem = _stem_from_artifact(html_path.name)
    rep = ChapterBrowserReport(chapter=stem, vol=vol, html=str(html_path.resolve().relative_to(REPO_ROOT.resolve())))
    qmd = _find_qmd(vol, stem)
    if qmd:
        rep.qmd = str(qmd.resolve().relative_to(REPO_ROOT.resolve()))

    page.goto(html_path.resolve().as_uri(), wait_until="domcontentloaded", timeout=120_000)
    try:
        page.wait_for_function(
            "() => typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.promise",
            timeout=30_000,
        )
        page.evaluate("() => MathJax.startup.promise")
    except Exception:
        pass
    page.wait_for_timeout(500)

    visible = page.evaluate(
        """() => {
            const main = document.querySelector('main') || document.body;
            if (!main) return '';
            const skip = new Set(['SCRIPT','STYLE','PRE','CODE','MJX-CONTAINER']);
            const walk = (node) => {
                if (node.nodeType === Node.TEXT_NODE) return node.textContent || '';
                if (node.nodeType !== Node.ELEMENT_NODE) return '';
                const el = node;
                if (skip.has(el.tagName)) return '';
                if (el.tagName === 'SPAN' && (el.className||'').includes('math')) return '';
                return Array.from(el.childNodes).map(walk).join(' ');
            };
            return walk(main).replace(/\\s+/g, ' ').trim();
        }"""
    )

    for label, pat in LEAK_PATTERNS:
        for m in pat.finditer(visible):
            rep.leak_count += 1
            rep.leaks.append({"pattern": label, "match": m.group(0)})
            if rep.leak_count >= 50:
                break
        if rep.leak_count >= 50:
            break

    for pat in ERROR_PATTERNS:
        if pat.search(visible):
            rep.errors.append(pat.pattern)

    if check_refs and qmd and INLINE_PY.search(qmd.read_text(encoding="utf-8")):
        try:
            ns = _exec_python_cells(qmd.read_text(encoding="utf-8").splitlines())
        except RuntimeError as exc:
            rep.errors.append(f"cell_exec: {exc}")
            rep.ok = False
            return rep
        for ref in _collect_inline_refs(qmd):
            rep.refs_checked += 1
            val = _resolve_ref(ref, ns)
            if not _value_in_text(val, visible):
                rep.ref_failures.append({"ref": ref, "expected": val[:120]})

    rep.ok = rep.leak_count == 0 and not rep.errors and not rep.ref_failures
    if not rep.ok and screenshot_dir:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            page.screenshot(path=str(screenshot_dir / f"{vol}_{stem}.png"))
        except Exception:
            pass
    return rep


def _artifact_dirs(vol: str, explicit: Path | None) -> Path:
    """Return ``explicit`` or the newest debug run's ``phase1/artifacts`` directory for ``vol``.

    Exits when the debug root has no runs; a missing debug root raises
    ``FileNotFoundError``.
    """
    if explicit:
        return explicit
    debug_root = REPO_ROOT / "books/_build/debug" / vol / "html"
    runs = sorted(p for p in debug_root.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"No debug runs in {debug_root}")
    return runs[-1] / "phase1" / "artifacts"


def main() -> int:
    """Verify one volume's chapter pages in headless Chromium and write a JSON report.

    Pages come from ``books/_build/html-audit/<vol>`` with ``--html-audit``,
    otherwise from ``--artifact-dir`` or the newest debug run; ``--chapter``
    narrows to one stem. The report defaults to
    ``binder/tools/audit/artifacts/playwright_<vol>.json``. Returns 0 only when
    every page passed.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vol", choices=("vol1", "vol2"), required=True)
    ap.add_argument("--artifact-dir", type=Path)
    ap.add_argument("--html-audit", action="store_true")
    ap.add_argument("--chapter")
    ap.add_argument("--no-ref-parity", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--screenshots", type=Path)
    args = ap.parse_args()

    html_dir = REPO_ROOT / "books/_build/html-audit" / args.vol if args.html_audit else _artifact_dirs(args.vol, args.artifact_dir)
    html_files = sorted(html_dir.glob("*.html"))
    if args.chapter:
        html_files = [p for p in html_files if _stem_from_artifact(p.name) == args.chapter]

    from playwright.sync_api import sync_playwright

    reports = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        for i, hp in enumerate(html_files, 1):
            print(f"[{i}/{len(html_files)}] {hp.name} ... ", end="", flush=True)
            rep = verify_chapter_browser(page, hp, vol=args.vol, check_refs=not args.no_ref_parity, screenshot_dir=args.screenshots)
            reports.append(rep)
            print("ok" if rep.ok else f"FAIL leaks={rep.leak_count} refs={len(rep.ref_failures)}")
        browser.close()

    out = args.report or REPO_ROOT / "binder/tools/audit/artifacts/playwright_{}.json".format(args.vol)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in reports], indent=2))
    passed = sum(1 for r in reports if r.ok)
    print(f"Playwright {args.vol}: {passed}/{len(reports)} pass")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
