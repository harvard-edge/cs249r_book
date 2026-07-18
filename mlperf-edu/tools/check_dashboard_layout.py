#!/usr/bin/env python3
"""Capture and verify rendered MLPerf EDU result dashboards."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VIEWPORTS = ((1440, 1000, "desktop"), (390, 844, "narrow"))


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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _parse_page(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("page must use LABEL=/path/to/report.html")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip():
        raise argparse.ArgumentTypeError("page label cannot be empty")
    if not path.is_file() or path.suffix.lower() != ".html":
        raise argparse.ArgumentTypeError(f"HTML report does not exist: {path}")
    return label.strip(), path


def _failure(label: str, viewport: str, check: str, detail: str) -> dict[str, str]:
    return {
        "page": label,
        "viewport": viewport,
        "check": check,
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check result dashboards at desktop and narrow viewports."
    )
    parser.add_argument(
        "--page",
        action="append",
        required=True,
        type=_parse_page,
        metavar="LABEL=PATH",
        help="named HTML report to inspect; may be repeated",
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.page]
    if len(labels) != len(set(labels)):
        parser.error("page labels must be unique")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is required; install it and its Chromium browser before running this gate",
            file=sys.stderr,
        )
        return 2

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    runs: list[dict[str, object]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for label, report_path in args.page:
                with _serve(report_path.parent) as base_url:
                    for width, height, viewport_label in VIEWPORTS:
                        context = browser.new_context(
                            viewport={"width": width, "height": height}
                        )
                        page = context.new_page()
                        console_errors: list[str] = []
                        page_errors: list[str] = []
                        failed_requests: list[str] = []
                        bad_responses: list[str] = []
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
                        page.on(
                            "requestfailed",
                            lambda request, errors=failed_requests: errors.append(
                                request.url
                            ),
                        )
                        page.on(
                            "response",
                            lambda response, errors=bad_responses: (
                                errors.append(f"{response.status} {response.url}")
                                if response.status >= 400
                                else None
                            ),
                        )
                        try:
                            response = page.goto(
                                f"{base_url}/{report_path.name}",
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
                                  textLength: document.body.innerText.trim().length
                                })"""
                            )
                            if status != 200:
                                failures.append(
                                    _failure(
                                        label,
                                        viewport_label,
                                        "HTTP_STATUS",
                                        str(status),
                                    )
                                )
                            if bad_responses:
                                failures.append(
                                    _failure(
                                        label,
                                        viewport_label,
                                        "RESOURCE_STATUS",
                                        "; ".join(bad_responses[:3]),
                                    )
                                )
                            if failed_requests:
                                failures.append(
                                    _failure(
                                        label,
                                        viewport_label,
                                        "REQUEST_FAILURES",
                                        "; ".join(failed_requests[:3]),
                                    )
                                )
                            if console_errors:
                                failures.append(
                                    _failure(
                                        label,
                                        viewport_label,
                                        "CONSOLE_ERRORS",
                                        "; ".join(console_errors[:3]),
                                    )
                                )
                            if page_errors:
                                failures.append(
                                    _failure(
                                        label,
                                        viewport_label,
                                        "PAGE_ERRORS",
                                        "; ".join(page_errors[:3]),
                                    )
                                )
                            if (
                                not layout["mainPresent"]
                                or not layout["heading"]
                                or int(layout["textLength"]) < 40
                            ):
                                failures.append(
                                    _failure(
                                        label,
                                        viewport_label,
                                        "CONTENT_SHELL",
                                        "report is missing its main content, heading, or result text",
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
                                        label,
                                        viewport_label,
                                        "HORIZONTAL_PAGE_OVERFLOW",
                                        f"page is {overflow}px wider than its {viewport_width}px viewport",
                                    )
                                )

                            screenshot_path = report_dir / (
                                f"{_slug(label)}-{viewport_label}.png"
                            )
                            page.screenshot(path=screenshot_path, full_page=True)
                            preview_path = report_dir / (
                                f"{_slug(label)}-{viewport_label}-preview.png"
                            )
                            page.screenshot(path=preview_path)
                            runs.append(
                                {
                                    "page": label,
                                    "source": str(report_path),
                                    "viewport": viewport_label,
                                    "status": status,
                                    "heading": layout["heading"],
                                    "document_width": layout["documentWidth"],
                                    "viewport_width": viewport_width,
                                    "screenshot": str(screenshot_path),
                                    "preview": str(preview_path),
                                }
                            )
                        except Exception as error:  # noqa: BLE001
                            failures.append(
                                _failure(
                                    label, viewport_label, "PAGE_LOAD", repr(error)
                                )
                            )
                        finally:
                            page.close()
                            context.close()
        finally:
            browser.close()

    report = {
        "schema_version": "0.1",
        "page_count": len(args.page),
        "viewport_count": len(VIEWPORTS),
        "run_count": len(runs),
        "failure_count": len(failures),
        "failures": failures,
        "runs": runs,
    }
    report_path = report_dir / "results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if failures:
        print(
            f"dashboard layout failed with {len(failures)} issue(s); see {report_path}"
        )
        for failure in failures[:20]:
            print(
                f"- {failure['page']} [{failure['viewport']}] "
                f"{failure['check']}: {failure['detail']}"
            )
        return 1

    print(
        f"dashboard layout passed for {len(args.page)} pages across "
        f"{len(VIEWPORTS)} viewports ({len(runs)} runs); report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
