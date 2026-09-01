"""Automated headless browser rendering, visual layout, and asset integrity verification via Playwright."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import BaseCheck, CheckRegistry
from ..context import BookContext
from ..report import LintIssue, LintReport

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False


@CheckRegistry.register
class PlaywrightVisualCheck(BaseCheck):
    name = "playwright_visual"
    description = "Renders HTML chapters in headless Chromium to check for console errors, broken images, and layout overflows"
    category = "playwright"

    def run(self, ctx: BookContext, report: LintReport):
        if not HAVE_PLAYWRIGHT:
            report.add_issue(LintIssue(
                category="playwright",
                severity="INFO",
                file="HTML",
                line=None,
                page=None,
                message="Playwright is not installed; skipping browser layout and rendering verification."
            ))
            return

        html_files: List[Path] = []
        if ctx.chapter_filter:
            matches = list(ctx.html_dir.glob(f"chapters/*{ctx.chapter_filter}*/**/*.html")) + \
                      list(ctx.html_dir.glob(f"chapters/*{ctx.chapter_filter}*.html"))
            html_files.extend(matches)
        else:
            html_files.extend(sorted(ctx.html_dir.rglob("*.html")))

        # Filter out temp/non-chapter html
        html_files = [f for f in html_files if "chapters" in f.parts or f.name == "index.html"]

        if not html_files:
            report.add_issue(LintIssue(
                category="playwright",
                severity="WARNING",
                file="HTML",
                line=None,
                page=None,
                message="No compiled HTML files found in _build/. Run 'pai build --to html' before running browser checks."
            ))
            return

        screenshot_dir = ctx.book_dir / "_build" / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--allow-file-access-from-files", "--disable-web-security"]
            )

            for html_path in html_files:
                rel_path = str(html_path.relative_to(ctx.repo_root))
                page = browser.new_page(viewport={"width": 1280, "height": 800})

                console_errors: List[str] = []
                page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
                page.on("pageerror", lambda err: console_errors.append(str(err)))

                file_uri = html_path.resolve().as_uri()
                page.goto(file_uri, wait_until="networkidle")

                # 1. Check Console Errors
                for err in console_errors:
                    report.add_issue(LintIssue(
                        category="playwright",
                        severity="ERROR",
                        file=rel_path,
                        line=None,
                        page=None,
                        message=f"Browser console error: {err[:100]}",
                        context=err
                    ))

                # 2. Check Broken Images (naturalWidth === 0)
                broken_images = page.evaluate("""() => {
                    const imgs = Array.from(document.querySelectorAll('img'));
                    return imgs.filter(img => img.naturalWidth === 0).map(img => img.src);
                }""")

                for broken_src in broken_images:
                    report.add_issue(LintIssue(
                        category="playwright",
                        severity="ERROR",
                        file=rel_path,
                        line=None,
                        page=None,
                        message=f"Broken image load: '{broken_src}'",
                        context="Image failed to render in browser (naturalWidth is 0)"
                    ))

                # 3. Check Horizontal Overflow
                has_overflow = page.evaluate("""() => {
                    return document.documentElement.scrollWidth > window.innerWidth + 10;
                }""")
                if has_overflow:
                    report.add_issue(LintIssue(
                        category="playwright",
                        severity="WARNING",
                        file=rel_path,
                        line=None,
                        page=None,
                        message="Horizontal scrollbar detected: content exceeds viewport width",
                        context="Page has horizontal layout overflow on standard 1280px viewport"
                    ))

                # 4. Capture Key Section Visuals for Inspection
                # Chapter Opener Viewport
                shot_name = html_path.stem
                page.screenshot(path=str(screenshot_dir / f"{shot_name}_opener.png"))

                # Capture callouts / autopsies if present
                autopsy_el = page.query_selector(".callout-autopsy, .callout-warning, .callout-objective")
                if autopsy_el:
                    autopsy_el.screenshot(path=str(screenshot_dir / f"{shot_name}_callout.png"))

                page.close()

            browser.close()
