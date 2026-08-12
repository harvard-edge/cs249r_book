#!/usr/bin/env python3
"""Browser render smoke checks for Marimo lab apps.

This tool is intentionally separate from the default pytest suite. It starts
read-only Marimo servers, opens each lab in Chromium, clicks through the
canonical tracks, checks for common browser/runtime failures, and writes
screenshots to a local output directory. Non-orientation labs must render four
distinct page states after track selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:  # pragma: no cover - CLI dependency guard
    raise SystemExit(f"Playwright is required: {exc}") from exc


TRACKS = ("iPhone", "Oura Ring", "RoboTaxi", "Cloud Fleet")
ERROR_MARKERS = (
    "Traceback",
    "Exception",
    "NameError",
    "ModuleNotFoundError",
    "Accessing the value of a UIElement",
)


@dataclass(frozen=True)
class TrackCheck:
    track: str
    track_seen: bool
    selector_count: int
    overflowing_fields: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class LabCheck:
    lab: str
    url: str
    screenshot: str
    page_errors: tuple[str, ...]
    page_overflowing_fields: int
    unique_track_texts: int
    checks: tuple[TrackCheck, ...]

    @property
    def passed(self) -> bool:
        return (
            not self.page_errors
            and self.page_overflowing_fields == 0
            and all(
                check.track_seen
                and check.selector_count <= 1
                and check.overflowing_fields == 0
                and not check.errors
                for check in self.checks
            )
            and (not self.checks or self.unique_track_texts == len(self.checks))
        )


def wait_for_server(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - timing dependent
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Server did not respond at {url}: {last_error}")


def start_server(lab: Path, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "marimo",
            "run",
            str(lab),
            "--headless",
            "--no-token",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_server(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
        try:
            return process.communicate(timeout=5)[0]
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
            return process.communicate(timeout=5)[0]
    return process.communicate(timeout=5)[0]


def check_lab(lab: Path, url: str, screenshot: Path) -> LabCheck:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(5_000)
        initial_text = page.locator("body").inner_text(timeout=10_000)
        page_errors = tuple(marker for marker in ERROR_MARKERS if marker in initial_text)
        page_overflowing_fields = int(
            page.locator(".mlsysbook-field").evaluate_all(
                "els => els.filter(e => e.scrollWidth > e.clientWidth + 1).length"
            )
        )
        if lab.stem.startswith("lab_00"):
            page.screenshot(path=str(screenshot), full_page=True)
            browser.close()
            return LabCheck(
                lab=str(lab),
                url=url,
                screenshot=str(screenshot),
                page_errors=page_errors,
                page_overflowing_fields=page_overflowing_fields,
                unique_track_texts=1,
                checks=(),
            )
        checks: list[TrackCheck] = []
        track_texts: list[str] = []
        for track in TRACKS:
            text = page.locator("body").inner_text(timeout=10_000)
            errors: tuple[str, ...]
            try:
                page.get_by_text(track, exact=True).first.click(timeout=5_000)
                page.wait_for_timeout(1_200)
                text = page.locator("body").inner_text(timeout=10_000)
                errors = tuple(marker for marker in ERROR_MARKERS if marker in text)
            except Exception as exc:
                errors = (f"click_failed: {str(exc).splitlines()[0]}",)
            overflowing_fields = int(
                page.locator(".mlsysbook-field").evaluate_all(
                    "els => els.filter(e => e.scrollWidth > e.clientWidth + 1).length"
                )
            )
            checks.append(
                TrackCheck(
                    track=track,
                    track_seen=track in text,
                    selector_count=text.count("Your Track"),
                    overflowing_fields=overflowing_fields,
                    errors=errors,
                )
            )
            track_texts.append(text)
        page.screenshot(path=str(screenshot), full_page=True)
        browser.close()
    return LabCheck(
        lab=str(lab),
        url=url,
        screenshot=str(screenshot),
        page_errors=page_errors,
        page_overflowing_fields=page_overflowing_fields,
        unique_track_texts=len(set(track_texts)),
        checks=tuple(checks),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", nargs="+", required=True, help="Lab .py files to render.")
    parser.add_argument("--port-start", type=int, default=29100)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/mlsysbook-render-smoke"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[LabCheck] = []
    for index, raw_lab in enumerate(args.labs):
        lab = Path(raw_lab)
        port = args.port_start + index
        url = f"http://127.0.0.1:{port}"
        screenshot = args.output_dir / f"{lab.stem}.png"
        process = start_server(lab, port)
        try:
            wait_for_server(url)
            result = check_lab(lab, url, screenshot)
            results.append(result)
        finally:
            output = stop_server(process)
            if process.returncode not in (0, -15, None):
                print(output, file=sys.stderr)

    payload = [asdict(result) | {"passed": result.passed} for result in results]
    print(json.dumps(payload, indent=2))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
