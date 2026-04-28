#!/usr/bin/env python3
# =============================================================================
# Visual Smoke Test — Quarto site rendering sanity check
# =============================================================================
# Runs four cheap assertions against a built _build/ directory, at four
# viewport widths × light + dark color schemes, on each --page passed in:
#
#   1. STYLESHEETS_RESOLVE — every <link rel="stylesheet"> returns 200.
#      Catches the site_libs/ deploy regression that left mlsysbook.ai/about/
#      shipping a raw <ul> navbar with no Bootstrap (commit 6fdf81dd4).
#
#   2. NO_CONSOLE_ERRORS — Playwright's console listener captures zero
#      severity=error messages during load. Catches JS failures from
#      missing CDNs, broken theme-bridge scripts, and similar.
#
#   3. PAGE_HAS_HEIGHT — body.scrollHeight > 1.5 × viewport_h on the
#      homepage. The about-page failure mode rendered at body height
#      ~equal to the viewport because all the styled content was stacked
#      below 3000 px of unstyled whitespace; this assertion bottoms out
#      that class of regression.
#
#   4. NAVBAR_COLLAPSE_AT_XL — at widths ≥ 1200 px the .navbar-collapse
#      element should be visible (no hamburger); at widths ≤ 1199 px it
#      should be hidden (hamburger only). This is the shared-chrome
#      breakpoint that issues #1 and #2 broke; one assertion proves the
#      shared navbar is consistent across every site that consumes
#      shared/config/navbar-common.yml.
#
# Usage:
#   visual_smoke.py --build-dir tinytorch/quarto/_build --site tinytorch
#                   --pages /index.html /preface.html
#                   [--report-dir _smoke-report] [--port 0]
#
# Exit codes:
#   0  all assertions passed
#   1  one or more assertions failed (see report-dir/results.json)
#   2  setup error (Playwright not installed, build dir missing, etc.)
# =============================================================================

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
    sys.exit(2)


# Viewport widths chosen to bracket the navbar's xl collapse breakpoint:
# 1199 → just below collapse, must be hamburger; 1200 → just at, must be expanded.
VIEWPORTS: list[tuple[int, int]] = [(1400, 900), (1200, 900), (1199, 900), (992, 900)]
COLOR_SCHEMES: list[str] = ["light", "dark"]
NAVBAR_EXPANDED_MIN_WIDTH = 1200  # matches collapse-below: xl in navbar-common.yml


@dataclass
class Failure:
    """One assertion that failed. Aggregated and reported at the end."""
    page: str
    viewport: str
    scheme: str
    assertion: str
    detail: str


@dataclass
class Report:
    site: str
    failures: list[Failure] = field(default_factory=list)
    pages_checked: int = 0
    matrix_runs: int = 0  # pages × viewports × schemes

    def add(self, page: str, vp: tuple[int, int], scheme: str, assertion: str, detail: str) -> None:
        self.failures.append(Failure(page, f"{vp[0]}x{vp[1]}", scheme, assertion, detail))

    def to_dict(self) -> dict:
        return {
            "site": self.site,
            "pages_checked": self.pages_checked,
            "matrix_runs": self.matrix_runs,
            "failure_count": len(self.failures),
            "failures": [f.__dict__ for f in self.failures],
        }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _serve(directory: Path, port: int):
    """Boot a SimpleHTTPServer rooted at directory; teardown on exit."""
    handler = lambda *a, **kw: SimpleHTTPRequestHandler(*a, directory=str(directory), **kw)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Tiny wait so the first request doesn't race the bind().
    time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _check_page(page, base_url: str, page_path: str, vp: tuple[int, int], scheme: str, report: Report) -> None:
    """Run all four assertions on one (page, viewport, scheme) tuple."""
    url = base_url.rstrip("/") + page_path
    console_errors: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

    page.goto(url, wait_until="networkidle", timeout=30000)

    # 1. STYLESHEETS_RESOLVE — every <link rel=stylesheet> returns 200.
    css_results = page.evaluate(
        """async () => {
          const links = Array.from(document.querySelectorAll('link[rel~="stylesheet"]'));
          const out = [];
          for (const l of links) {
            try {
              const r = await fetch(l.href, { method: 'HEAD' });
              out.push({ href: l.href, status: r.status });
            } catch (e) {
              out.push({ href: l.href, status: 0, error: String(e) });
            }
          }
          return out;
        }"""
    )
    bad_css = [c for c in css_results if c["status"] != 200]
    if bad_css:
        report.add(page_path, vp, scheme, "STYLESHEETS_RESOLVE",
                   f"{len(bad_css)} stylesheet(s) failed: " +
                   ", ".join(f"{c['href']}→{c['status']}" for c in bad_css[:3]))

    # 2. NO_CONSOLE_ERRORS — capture window for the page load. (Listener was
    # attached above; any errors during the goto/networkidle window are now
    # in console_errors.)
    if console_errors:
        report.add(page_path, vp, scheme, "NO_CONSOLE_ERRORS",
                   f"{len(console_errors)} console error(s): " +
                   "; ".join(console_errors[:2]))

    # 3. PAGE_HAS_HEIGHT — body taller than 1.5× viewport. We only enforce
    # this on the homepage of each site to avoid false positives on
    # intentionally short pages (404s, redirect stubs).
    if page_path in ("/", "/index.html"):
        body_h = page.evaluate("document.body.scrollHeight")
        if body_h < vp[1] * 1.5:
            report.add(page_path, vp, scheme, "PAGE_HAS_HEIGHT",
                       f"body.scrollHeight={body_h}px < {int(vp[1] * 1.5)}px (viewport_h × 1.5)")

    # 4. NAVBAR_COLLAPSE_AT_XL — collapse boundary is consistent.
    nav_state = page.evaluate(
        """() => {
          const collapse = document.querySelector('.navbar-collapse');
          const toggler = document.querySelector('.navbar-toggler');
          if (!collapse || !toggler) return null;
          // Bootstrap toggles `display: none` on the toggler at expanded
          // breakpoints. That's a more reliable signal than .navbar-collapse
          // visibility, which Bootstrap always renders (it's the *expanded*
          // collapse-target, just with `.collapse` toggled).
          const togglerVisible = getComputedStyle(toggler).display !== 'none';
          return { togglerVisible };
        }"""
    )
    if nav_state is not None:
        expected_collapsed = vp[0] < NAVBAR_EXPANDED_MIN_WIDTH
        actually_collapsed = nav_state["togglerVisible"]
        if expected_collapsed != actually_collapsed:
            report.add(page_path, vp, scheme, "NAVBAR_COLLAPSE_AT_XL",
                       f"width={vp[0]}: expected collapsed={expected_collapsed}, "
                       f"actual collapsed={actually_collapsed} (toggler visible={nav_state['togglerVisible']})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run visual smoke checks against a built Quarto site.")
    parser.add_argument("--build-dir", type=Path, required=True, help="Path to _build/ directory")
    parser.add_argument("--site", required=True, help="Site name (for the report)")
    parser.add_argument("--pages", nargs="+", default=["/index.html"], help="Page paths to check (default: /index.html)")
    parser.add_argument("--report-dir", type=Path, default=Path("_smoke-report"), help="Where to write results.json + failure screenshots")
    parser.add_argument("--port", type=int, default=0, help="Port to serve on (0 = pick free port)")
    args = parser.parse_args()

    if not args.build_dir.is_dir():
        print(f"ERROR: --build-dir {args.build_dir} does not exist or is not a directory", file=sys.stderr)
        return 2

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = Report(site=args.site)
    port = args.port or _free_port()

    print(f"🔎 visual smoke: site={args.site} build={args.build_dir} pages={args.pages}")

    with _serve(args.build_dir, port) as base_url:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                for page_path in args.pages:
                    report.pages_checked += 1
                    for vp in VIEWPORTS:
                        for scheme in COLOR_SCHEMES:
                            ctx = browser.new_context(
                                viewport={"width": vp[0], "height": vp[1]},
                                color_scheme=scheme,
                            )
                            page = ctx.new_page()
                            try:
                                _check_page(page, base_url, page_path, vp, scheme, report)
                            except Exception as e:
                                report.add(page_path, vp, scheme, "PAGE_LOAD",
                                           f"{type(e).__name__}: {e}")
                                # Snapshot the broken state so the failure is debuggable.
                                with contextlib.suppress(Exception):
                                    snap = args.report_dir / f"fail_{page_path.strip('/').replace('/', '_') or 'index'}_{vp[0]}_{scheme}.png"
                                    page.screenshot(path=str(snap), full_page=False)
                            finally:
                                ctx.close()
                            report.matrix_runs += 1
            finally:
                browser.close()

    results_path = args.report_dir / "results.json"
    results_path.write_text(json.dumps(report.to_dict(), indent=2))

    if report.failures:
        print(f"\n❌ {len(report.failures)} failure(s) across {report.matrix_runs} runs:")
        for f in report.failures:
            print(f"   [{f.assertion}] {f.page} @ {f.viewport} {f.scheme}: {f.detail}")
        print(f"\nResults: {results_path}")
        return 1

    print(f"\n✅ {report.matrix_runs} runs across {report.pages_checked} page(s), all green.")
    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
