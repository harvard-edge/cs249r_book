#!/usr/bin/env python3
"""Verify the rendered MLPerf EDU site at desktop and narrow viewports."""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VIEWPORTS = ((1440, 1000, "desktop"), (390, 844, "narrow"))
SCREENSHOT_PAGES = (
    "index.html",
    "getting-started.html",
    "readiness.html",
    "benchmarks/index.html",
    "benchmarks/language/causal-language-modeling.html",
    "labs/index.html",
    "guide/instructors.html",
    "guide/research.html",
    "guide/results.html",
    "guide/troubleshooting.html",
    "reference/cli.html",
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _serve(directory: Path):
    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args, directory=str(directory), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _rendered_pages(build_dir: Path) -> list[str]:
    return [
        path.relative_to(build_dir).as_posix()
        for path in sorted(build_dir.rglob("*.html"))
    ]


def _failure(page: str, viewport: str, check: str, detail: str) -> dict[str, str]:
    return {"page": page, "viewport": viewport, "check": check, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check every rendered page for load, console, and viewport failures."
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    report_dir = args.report_dir.resolve()
    if not build_dir.is_dir():
        parser.error(f"rendered site directory does not exist: {build_dir}")

    pages = _rendered_pages(build_dir)
    if not pages:
        parser.error(f"no rendered HTML pages found under {build_dir}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is required; install it and its Chromium browser before running this gate",
            file=sys.stderr,
        )
        return 2

    report_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    runs: list[dict[str, object]] = []

    with _serve(build_dir) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for width, height, label in VIEWPORTS:
                context = browser.new_context(
                    viewport={"width": width, "height": height}
                )
                try:
                    for relative_path in pages:
                        page = context.new_page()
                        console_errors: list[str] = []
                        page_errors: list[str] = []
                        page.on(
                            "console",
                            lambda message, errors=console_errors: (
                                errors.append(message.text)
                                if message.type == "error"
                                else None
                            ),
                        )
                        page.on(
                            "pageerror",
                            lambda error, errors=page_errors: errors.append(str(error)),
                        )
                        try:
                            response = page.goto(
                                f"{base_url}/{relative_path}",
                                wait_until="networkidle",
                                timeout=30_000,
                            )
                            status = response.status if response else 0
                            layout = page.evaluate(
                                """() => ({
                                  viewportWidth: document.documentElement.clientWidth,
                                  documentWidth: document.documentElement.scrollWidth,
                                  bodyWidth: document.body.scrollWidth,
                                  heading: document.querySelector('h1')?.innerText?.trim() || '',
                                  mainPresent: Boolean(document.querySelector('main')),
                                  wideTableCount: Array.from(document.querySelectorAll('main table'))
                                    .filter((table) => table.scrollWidth > table.clientWidth + 1).length,
                                  uncuedWideTableCount: Array.from(document.querySelectorAll('main table'))
                                    .filter((table) => table.scrollWidth > table.clientWidth + 1)
                                    .filter((table) => !getComputedStyle(table, '::before').content.includes('Swipe horizontally')).length
                                })"""
                            )
                            if status != 200:
                                failures.append(
                                    _failure(
                                        relative_path, label, "HTTP_STATUS", str(status)
                                    )
                                )
                            if console_errors:
                                failures.append(
                                    _failure(
                                        relative_path,
                                        label,
                                        "CONSOLE_ERRORS",
                                        "; ".join(console_errors[:3]),
                                    )
                                )
                            if page_errors:
                                failures.append(
                                    _failure(
                                        relative_path,
                                        label,
                                        "PAGE_ERRORS",
                                        "; ".join(page_errors[:3]),
                                    )
                                )
                            if not layout["mainPresent"] or not layout["heading"]:
                                failures.append(
                                    _failure(
                                        relative_path,
                                        label,
                                        "CONTENT_SHELL",
                                        "rendered page is missing main content or an h1 heading",
                                    )
                                )
                            viewport_width = int(layout["viewportWidth"])
                            overflow = (
                                max(
                                    int(layout["documentWidth"]),
                                    int(layout["bodyWidth"]),
                                )
                                - viewport_width
                            )
                            if overflow > 1:
                                failures.append(
                                    _failure(
                                        relative_path,
                                        label,
                                        "HORIZONTAL_PAGE_OVERFLOW",
                                        f"page is {overflow}px wider than its {viewport_width}px viewport",
                                    )
                                )
                            if label == "narrow" and layout["uncuedWideTableCount"]:
                                failures.append(
                                    _failure(
                                        relative_path,
                                        label,
                                        "TABLE_SCROLL_CUE",
                                        f"{layout['uncuedWideTableCount']} of {layout['wideTableCount']} horizontally scrollable tables lack a visible cue",
                                    )
                                )

                            runs.append(
                                {
                                    "page": relative_path,
                                    "viewport": label,
                                    "status": status,
                                    "heading": layout["heading"],
                                    "document_width": layout["documentWidth"],
                                    "viewport_width": viewport_width,
                                }
                            )
                            if relative_path in SCREENSHOT_PAGES:
                                screenshot = report_dir / (
                                    f"{relative_path.removesuffix('.html').replace('/', '-')}-{label}.png"
                                )
                                page.screenshot(path=screenshot, full_page=True)
                        except Exception as error:  # noqa: BLE001
                            failures.append(
                                _failure(relative_path, label, "PAGE_LOAD", repr(error))
                            )
                        finally:
                            page.close()
                finally:
                    context.close()
        finally:
            browser.close()

    report = {
        "schema_version": "0.1",
        "build_dir": str(build_dir),
        "page_count": len(pages),
        "viewport_count": len(VIEWPORTS),
        "run_count": len(runs),
        "failure_count": len(failures),
        "failures": failures,
        "runs": runs,
    }
    report_path = report_dir / "results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if failures:
        print(f"site layout failed with {len(failures)} issue(s); see {report_path}")
        for failure in failures[:20]:
            print(
                f"- {failure['page']} [{failure['viewport']}] "
                f"{failure['check']}: {failure['detail']}"
            )
        return 1

    print(
        f"site layout passed for {len(pages)} pages across "
        f"{len(VIEWPORTS)} viewports ({len(runs)} runs); report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
