#!/usr/bin/env python3
"""Deep browser interaction smoke checks for MLSysBook lab pages.

This tool starts each Marimo lab, loads it in Chromium, scrolls through the
rendered page, switches canonical tracks, opens visible tabs, clicks safe radio
controls, checks for runtime/layout failures, and writes screenshots plus a JSON
report. It also checks the static lab dashboard/catalog HTML pages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception as exc:  # pragma: no cover - CLI dependency guard
    raise SystemExit(f"Playwright is required: {exc}") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS = ("iPhone", "Oura Ring", "RoboTaxi", "Cloud Fleet")
ERROR_MARKERS = (
    "Traceback",
    "Exception",
    "NameError",
    "ModuleNotFoundError",
    "Accessing the value of a UIElement",
)
TRANSIENT_MARKER = "This output is stale"
TAB_TEXT_PATTERNS = ("Part ", "Synthesis")
ANSWER_OPTION_RE = re.compile(r"^[A-D]\)\s+.+")
DEFAULT_HTML_PAGES = (
    REPO_ROOT / "labs" / "lab-plan-dashboard.html",
    REPO_ROOT / "labs" / "lab-modality-catalog.html",
)
FALLBACK_TAB_LABELS = ("Part A", "Part B", "Part C", "Part D", "Part E", "Synthesis")
LAB00_UNLOCK_ACTIONS = (
    ("orientation check 1", ("C)  Practice a repeated workflow", "C) Practice a repeated workflow")),
    ("track changes: story", ("The story and stakeholder voice",)),
    ("track changes: device", ("The device and hardware assumptions",)),
    ("track changes: metrics", ("The primary metric and guardrails",)),
    ("track changes: report", ("The report framing",)),
    (
        "orientation check 3",
        (
            "A)  Read the case, predict first, manipulate controls, inspect evidence, decide, then report",
            "A) Read the case, predict first, manipulate controls, inspect evidence, decide, then report",
            "A)  Read the case, guess first",
            "A) Read the case, guess first",
        ),
    ),
)


@dataclass(frozen=True)
class ClickCheck:
    label: str
    clicked: bool
    errors: tuple[str, ...]
    stale: bool
    note: str = ""


@dataclass(frozen=True)
class PartCheck:
    label: str
    clicked: bool
    text_words: int
    piece_categories: tuple[str, ...]
    visible_controls: int
    visible_plots: int
    visible_fields: int
    has_multiple_pieces: bool
    errors: tuple[str, ...]
    stale: bool
    note: str = ""


@dataclass(frozen=True)
class LabInteractionCheck:
    lab: str
    url: str
    loaded: bool
    body_length: int
    screenshot: str
    page_errors: tuple[str, ...]
    console_errors: tuple[str, ...]
    horizontal_overflow_px: int
    offscreen_left_elements: int
    overflowing_fields: int
    scroll_positions: tuple[int, ...]
    track_checks: tuple[ClickCheck, ...]
    tab_checks: tuple[ClickCheck, ...]
    part_checks: tuple[PartCheck, ...]
    radio_checks: tuple[ClickCheck, ...]
    server_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.loaded
            and self.body_length > 200
            and not self.page_errors
            and not self.console_errors
            and self.horizontal_overflow_px <= 2
            and self.offscreen_left_elements == 0
            and self.overflowing_fields == 0
            and all(check.clicked and not check.errors and not check.stale for check in self.track_checks)
            and all(not check.errors and not check.stale for check in self.tab_checks)
            and (
                not self.part_checks
                or all(
                    check.clicked
                    and check.has_multiple_pieces
                    and not check.errors
                    and not check.stale
                    for check in self.part_checks
                )
            )
            and all(not check.errors and not check.stale for check in self.radio_checks)
            and not self.server_errors
        )


@dataclass(frozen=True)
class HtmlPageCheck:
    page: str
    url: str
    loaded: bool
    screenshot: str
    page_errors: tuple[str, ...]
    console_errors: tuple[str, ...]
    horizontal_overflow_px: int
    scroll_positions: tuple[int, ...]
    checks: dict[str, int | bool]

    @property
    def passed(self) -> bool:
        return (
            self.loaded
            and not self.page_errors
            and not self.console_errors
            and self.horizontal_overflow_px <= 2
            and all(bool(value) for value in self.checks.values())
        )


def default_labs() -> list[Path]:
    return sorted((REPO_ROOT / "labs" / "vol1").glob("lab_*.py")) + sorted(
        (REPO_ROOT / "labs" / "vol2").glob("lab_*.py")
    )


def _safe_marimo_code(lab: Path, port: int) -> str:
    argv = [
        "marimo",
        "run",
        str(lab),
        "--headless",
        "--no-token",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    # Some local editable installs register import finders that raise while
    # Marimo's package detector probes modules. Filter editable/broken finders
    # and wrap find_spec before Marimo captures it in its formatter hooks.
    return f"""
import importlib.util
import runpy
import sys


def _mlsysbook_finder_name(finder):
    return " ".join(
        str(part)
        for part in (
            repr(finder),
            getattr(finder, "__module__", ""),
            getattr(type(finder), "__module__", ""),
        )
    )


def _mlsysbook_keep_finder(finder):
    name = _mlsysbook_finder_name(finder)
    if "__editable__" in name:
        return False
    probe = getattr(finder, "find_spec", None)
    if probe is None:
        return True
    try:
        probe("__mlsysbook_missing_probe__", None, None)
    except TypeError:
        try:
            probe("__mlsysbook_missing_probe__", None)
        except Exception:
            return False
    except Exception:
        return False
    return True


sys.meta_path = [finder for finder in sys.meta_path if _mlsysbook_keep_finder(finder)]
_mlsysbook_original_find_spec = importlib.util.find_spec


def _mlsysbook_safe_find_spec(name, package=None):
    try:
        return _mlsysbook_original_find_spec(name, package)
    except NameError as exc:
        if "spec_from_loader" in str(exc) or "spec_from_file_location" in str(exc):
            return None
        raise


importlib.util.find_spec = _mlsysbook_safe_find_spec
sys.argv = {argv!r}
runpy.run_module("marimo.__main__", run_name="__main__")
"""


def safe_marimo_env() -> dict[str, str]:
    env = os.environ.copy()
    sitecustomize_dir = Path("/tmp/mlsysbook-marimo-sitecustomize")
    sitecustomize_dir.mkdir(parents=True, exist_ok=True)
    (sitecustomize_dir / "sitecustomize.py").write_text(
        """
import importlib.util
import sys


def _mlsysbook_finder_name(finder):
    return " ".join(
        str(part)
        for part in (
            repr(finder),
            getattr(finder, "__module__", ""),
            getattr(type(finder), "__module__", ""),
        )
    )


def _mlsysbook_keep_finder(finder):
    name = _mlsysbook_finder_name(finder)
    if "__editable__" in name:
        return False
    probe = getattr(finder, "find_spec", None)
    if probe is None:
        return True
    try:
        probe("__mlsysbook_missing_probe__", None, None)
    except TypeError:
        try:
            probe("__mlsysbook_missing_probe__", None)
        except Exception:
            return False
    except Exception:
        return False
    return True


sys.meta_path = [finder for finder in sys.meta_path if _mlsysbook_keep_finder(finder)]
_mlsysbook_original_find_spec = importlib.util.find_spec


def _mlsysbook_safe_find_spec(name, package=None):
    try:
        return _mlsysbook_original_find_spec(name, package)
    except NameError as exc:
        if "spec_from_loader" in str(exc) or "spec_from_file_location" in str(exc):
            return None
        raise


importlib.util.find_spec = _mlsysbook_safe_find_spec
""".lstrip(),
        encoding="utf-8",
    )
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(sitecustomize_dir)
        if not current_pythonpath
        else f"{sitecustomize_dir}{os.pathsep}{current_pythonpath}"
    )
    return env


def start_server(lab: Path, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _safe_marimo_code(lab, port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=safe_marimo_env(),
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


def wait_for_server(url: str, timeout_s: float = 25.0) -> None:
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


def wait_for_lab_ready(page, timeout_ms: int = 30_000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        try:
            text = page.locator("body").inner_text(timeout=2_000)
            chips = page.locator(".mlsysbook-chip").count()
            if chips or len(text.strip()) > 500 or any(marker in text for marker in ERROR_MARKERS):
                return bool(text.strip())
        except Exception:
            pass
        page.wait_for_timeout(500)
    return False


def page_text(page) -> str:
    try:
        return str(page.evaluate(
            """
            () => {
              const roots = [];
              const visit = (root) => {
                roots.push(root);
                for (const el of root.querySelectorAll?.("*") || []) {
                  if (el.shadowRoot) {
                    visit(el.shadowRoot);
                  }
                }
              };
              visit(document);
              const texts = roots.map((root) => root.body?.innerText || root.textContent || "");
              return texts.join("\\n").replace(/\\s+/g, " ").trim();
            }
            """
        ))
    except Exception:
        try:
            return page.locator("body").inner_text(timeout=10_000)
        except Exception:
            return ""


def visible_error_markers(page) -> tuple[str, ...]:
    text = page_text(page)
    return tuple(marker for marker in ERROR_MARKERS if marker in text)


def stale_visible(page) -> bool:
    return TRANSIENT_MARKER in page_text(page)


def wait_for_settled(page, timeout_ms: int = 5_000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    stale = stale_visible(page)
    while stale and time.time() < deadline:
        page.wait_for_timeout(500)
        stale = stale_visible(page)
    return not stale


def layout_metrics(page) -> tuple[int, int, int]:
    metrics = page.evaluate(
        """
        () => {
          const root = document.documentElement;
          const body = document.body;
          const overflow = Math.max(0, root.scrollWidth - root.clientWidth, body.scrollWidth - root.clientWidth);
          const fields = [...document.querySelectorAll('.mlsysbook-field, .mlsysbook-compact-field')];
          const overflowingFields = fields.filter((el) => el.scrollWidth > el.clientWidth + 1).length;
          const offscreenLeft = [...document.querySelectorAll('marimo-tabs, .mlsysbook-panel, .mlsysbook-recap, .mlsysbook-lab-header')]
            .filter((el) => {
              const rect = el.getBoundingClientRect();
              const style = window.getComputedStyle(el);
              return rect.width > 0
                && rect.height > 0
                && rect.bottom >= 0
                && rect.top <= window.innerHeight
                && style.display !== 'none'
                && style.visibility !== 'hidden'
                && rect.left < -2;
            }).length;
          return { overflow, overflowingFields, offscreenLeft };
        }
        """
    )
    return int(metrics["overflow"]), int(metrics["overflowingFields"]), int(metrics["offscreenLeft"])


def scroll_through(page, max_steps: int = 12) -> tuple[int, ...]:
    targets = page.evaluate(
        """
        () => {
          const candidates = [document.scrollingElement, document.documentElement, document.body, ...document.querySelectorAll('*')];
          const seen = new Set();
          const targets = [];
          for (const el of candidates) {
            if (!el || seen.has(el)) continue;
            seen.add(el);
            const maxScroll = el.scrollHeight - el.clientHeight;
            if (maxScroll <= 80) continue;
            const rect = el.getBoundingClientRect ? el.getBoundingClientRect() : {width: window.innerWidth, height: window.innerHeight};
            if (rect.width <= 0 || rect.height <= 0) continue;
            const id = `smoke-scroll-${targets.length}`;
            el.setAttribute('data-smoke-scroll-target', id);
            targets.push({id, maxScroll});
          }
          return targets
            .sort((a, b) => b.maxScroll - a.maxScroll)
            .slice(0, 4);
        }
        """
    )
    if not targets:
        return (0,)
    seen: list[int] = []
    for target in targets:
        max_scroll = int(target["maxScroll"])
        step = max(320, max_scroll // max_steps)
        positions = list(range(0, max_scroll + step, step))
        positions[-1] = max_scroll
        selector = f'[data-smoke-scroll-target="{target["id"]}"]'
        for position in positions[: max_steps + 1]:
            page.locator(selector).evaluate("(el, y) => { el.scrollTop = y; }", position)
            if target["id"] == "smoke-scroll-0":
                page.evaluate("(y) => window.scrollTo(0, y)", position)
            page.wait_for_timeout(120)
            seen.append(position)
        page.locator(selector).evaluate("(el) => { el.scrollTop = 0; }")
    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_timeout(120)
    return tuple(dict.fromkeys(seen))


def collect_visible_text_across_scroll(page, max_steps: int = 8) -> str:
    """Collect viewport text while scrolling Marimo/window scroll containers."""
    script = """
        (maxSteps) => {
          const visible = (el) => {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            return rect.width > 0
              && rect.height > 0
              && rect.bottom >= 0
              && rect.top <= window.innerHeight
              && style.display !== 'none'
              && style.visibility !== 'hidden';
          };
          const capture = () => {
            const textNodes = [];
            const seenText = new Set();
            for (const el of document.querySelectorAll('body *')) {
              if (!visible(el)) continue;
              const text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
              if (text && text.length < 3500 && !seenText.has(text)) {
                seenText.add(text);
                textNodes.push(text);
              }
            }
            return textNodes.join('\\n');
          };
          const candidates = [document.scrollingElement, document.documentElement, document.body, ...document.querySelectorAll('*')];
          const seen = new Set();
          const targets = [];
          for (const el of candidates) {
            if (!el || seen.has(el)) continue;
            seen.add(el);
            const maxScroll = el.scrollHeight - el.clientHeight;
            if (maxScroll <= 80) continue;
            const rect = el.getBoundingClientRect ? el.getBoundingClientRect() : {width: window.innerWidth, height: window.innerHeight};
            if (rect.width <= 0 || rect.height <= 0) continue;
            targets.push({el, maxScroll});
          }
          targets.sort((a, b) => b.maxScroll - a.maxScroll);
          const chunks = [capture()];
          for (const target of targets.slice(0, 4)) {
            const step = Math.max(320, Math.floor(target.maxScroll / maxSteps));
            const positions = [];
            for (let y = 0; y <= target.maxScroll; y += step) {
              positions.push(Math.min(y, target.maxScroll));
            }
            positions.push(target.maxScroll);
            for (const y of [...new Set(positions)].slice(0, maxSteps + 1)) {
              target.el.scrollTop = y;
              if (target.el === document.scrollingElement || target.el === document.documentElement || target.el === document.body) {
                window.scrollTo(0, y);
              }
              chunks.push(capture());
            }
            target.el.scrollTop = 0;
          }
          window.scrollTo(0, 0);
          return chunks.join('\\n');
        }
    """
    try:
        return str(page.evaluate(script, max_steps))
    except Exception:
        return ""


def click_first_matching_text(page, labels: tuple[str, ...], timeout: int = 5_000) -> str:
    last_error = ""
    for label in labels:
        try:
            page.get_by_text(label, exact=True).first.click(timeout=timeout)
            return label
        except Exception as exc:
            last_error = str(exc).splitlines()[0]
        try:
            page.get_by_text(label, exact=False).last.click(timeout=timeout)
            return label
        except Exception as exc:
            last_error = str(exc).splitlines()[0]
    shadow_match = click_marimo_shadow_control(page, labels)
    if shadow_match:
        return shadow_match
    raise PlaywrightTimeoutError(last_error or "no matching text found")


def click_marimo_shadow_control(page, labels: tuple[str, ...]) -> str:
    """Click a Marimo radio/checkbox whose option text is hidden in shadow DOM."""
    match = page.evaluate(
        """
        (labels) => {
          const allElements = () => {
            const elements = [];
            const visit = (root) => {
              for (const el of root.querySelectorAll?.("*") || []) {
                elements.push(el);
                if (el.shadowRoot) {
                  visit(el.shadowRoot);
                }
              }
            };
            visit(document);
            return elements;
          };
          const stripHtml = (value) => String(value || "")
            .replace(/<[^>]+>/g, " ")
            .replace(/&quot;/g, '"')
            .replace(/&amp;/g, "&")
            .replace(/\\s+/g, " ")
            .trim();
          const normalize = (value) => stripHtml(value).toLowerCase();
          const wanted = labels.map(normalize).filter(Boolean);
          const matches = (candidate) => {
            const text = normalize(candidate);
            return wanted.some((label) => text.includes(label) || label.includes(text));
          };
          const clickControl = (control) => {
            control.scrollIntoView({block: "center", inline: "nearest"});
            control.click();
          };

          for (const radio of allElements().filter((el) => el.tagName?.toLowerCase() === "marimo-radio")) {
            const controls = [...(radio.shadowRoot?.querySelectorAll("[role=radio]") || [])];
            for (const control of controls) {
              const candidate = control.value || control.getAttribute("value") || control.textContent || "";
              if (matches(candidate)) {
                clickControl(control);
                return stripHtml(candidate);
              }
            }
          }

          for (const checkbox of allElements().filter((el) => el.tagName?.toLowerCase() === "marimo-checkbox")) {
            const candidate = checkbox.getAttribute("data-label") || checkbox.textContent || "";
            if (!matches(candidate)) {
              continue;
            }
            const control = checkbox.shadowRoot?.querySelector("[role=checkbox], button, input");
            if (control) {
              clickControl(control);
              return stripHtml(candidate);
            }
          }
          return "";
        }
        """,
        list(labels),
    )
    return str(match or "")


def click_first_generic_marimo_radio(page) -> tuple[ClickCheck, ...]:
    """Click the first visible non-track Marimo radio option.

    Some labs use sentence-style radio labels instead of A/B/C prefixes. This
    keeps the render audit from judging only the pre-unlock prediction prompt.
    """
    match = page.evaluate(
        """
        (trackLabels) => {
          const normalize = (value) => String(value || "")
            .replace(/<[^>]+>/g, " ")
            .replace(/&quot;/g, '"')
            .replace(/&amp;/g, "&")
            .replace(/\\s+/g, " ")
            .trim();
          const tracks = trackLabels.map((label) => normalize(label).toLowerCase());
          const roots = [];
          const visit = (root) => {
            roots.push(root);
            for (const el of root.querySelectorAll?.("*") || []) {
              if (el.shadowRoot) {
                visit(el.shadowRoot);
              }
            }
          };
          visit(document);
          for (const root of roots) {
            for (const radio of root.querySelectorAll?.("marimo-radio") || []) {
              const hostRect = radio.getBoundingClientRect();
              const hostStyle = window.getComputedStyle(radio);
              if (
                hostRect.width <= 0
                || hostRect.height <= 0
                || hostStyle.display === "none"
                || hostStyle.visibility === "hidden"
              ) {
                continue;
              }
              const controls = [...(radio.shadowRoot?.querySelectorAll("[role=radio]") || [])];
              for (const control of controls) {
                const label = normalize(
                  control.value
                    || control.getAttribute("value")
                    || control.getAttribute("aria-label")
                    || control.textContent
                    || ""
                );
                if (!label) {
                  continue;
                }
                const lower = label.toLowerCase();
                if (tracks.some((track) => lower.includes(track))) {
                  continue;
                }
                control.scrollIntoView({block: "center", inline: "nearest"});
                control.click();
                return label;
              }
            }
          }
          return "";
        }
        """,
        list(TRACKS),
    )
    label = str(match or "")
    if not label:
        return ()
    page.wait_for_timeout(1_200)
    wait_for_settled(page)
    return (
        ClickCheck(
            label=label,
            clicked=True,
            errors=visible_error_markers(page),
            stale=stale_visible(page),
        ),
    )


def unlock_lab00_orientation(page) -> tuple[ClickCheck, ...]:
    checks: list[ClickCheck] = []
    for label, variants in LAB00_UNLOCK_ACTIONS:
        try:
            matched = click_first_matching_text(page, variants, timeout=5_000)
            page.wait_for_timeout(1_200)
            wait_for_settled(page)
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=True,
                    errors=visible_error_markers(page),
                    stale=stale_visible(page),
                    note=f"clicked {matched}",
                )
            )
        except Exception as exc:
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=False,
                    errors=(f"click_failed: {str(exc).splitlines()[0]}",),
                    stale=stale_visible(page),
                )
            )
    return tuple(checks)


def click_track_options(page) -> tuple[ClickCheck, ...]:
    checks: list[ClickCheck] = []
    for track in TRACKS:
        try:
            matched = click_first_matching_text(page, (track,), timeout=5_000)
            page.wait_for_timeout(1_000)
            wait_for_settled(page)
            text = page_text(page)
            checks.append(
                ClickCheck(
                    label=track,
                    clicked=track in text,
                    errors=visible_error_markers(page),
                    stale=stale_visible(page),
                    note=(
                        f"clicked {matched}; track text visible"
                        if track in text
                        else f"clicked {matched}; track text missing after click"
                    ),
                )
            )
        except Exception as exc:
            checks.append(
                ClickCheck(
                    label=track,
                    clicked=False,
                    errors=(f"click_failed: {str(exc).splitlines()[0]}",),
                    stale=stale_visible(page),
                )
            )
    return tuple(checks)


def visible_tab_labels(page, max_tabs: int) -> tuple[str, ...]:
    labels = page.evaluate(
        """
        (patterns) => {
          const roots = [];
          const visit = (root) => {
            roots.push(root);
            for (const el of root.querySelectorAll?.("*") || []) {
              if (el.shadowRoot) {
                visit(el.shadowRoot);
              }
            }
          };
          visit(document);
          const seen = new Set();
          const labels = [];
          for (const root of roots) {
            for (const el of root.querySelectorAll?.('button, [role="tab"]') || []) {
              const text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
              const rect = el.getBoundingClientRect();
              const style = window.getComputedStyle(el);
              if (!text || rect.width <= 0 || rect.height <= 0 || style.display === 'none' || style.visibility === 'hidden') {
                continue;
              }
              if (!patterns.some((pattern) => text.includes(pattern))) {
                continue;
              }
              if (!seen.has(text)) {
                seen.add(text);
                labels.push(text);
              }
            }
          }
          return labels;
        }
        """,
        list(TAB_TEXT_PATTERNS),
    )
    return tuple(str(label) for label in labels[:max_tabs])


def candidate_tab_labels(page, max_tabs: int) -> tuple[str, ...]:
    labels = list(visible_tab_labels(page, max_tabs=max_tabs))
    for label in FALLBACK_TAB_LABELS:
        if label not in labels:
            labels.append(label)
    return tuple(labels[:max_tabs])


def click_tabs(page, max_tabs: int) -> tuple[ClickCheck, ...]:
    checks: list[ClickCheck] = []
    visible_labels = set(visible_tab_labels(page, max_tabs=max_tabs))
    for label in candidate_tab_labels(page, max_tabs):
        try:
            clicked = page.evaluate(
                """
                (label) => {
                  const normalize = (value) => String(value || "").replace(/\\s+/g, " ").trim();
                  const roots = [];
                  const visit = (root) => {
                    roots.push(root);
                    for (const el of root.querySelectorAll?.("*") || []) {
                      if (el.shadowRoot) {
                        visit(el.shadowRoot);
                      }
                    }
                  };
                  visit(document);
                  for (const root of roots) {
                    for (const el of root.querySelectorAll?.('button, [role="tab"]') || []) {
                      const text = normalize(el.innerText || el.textContent);
                      if (!text.includes(label)) {
                        continue;
                      }
                      el.scrollIntoView({block: "center", inline: "nearest"});
                      el.click();
                      return true;
                    }
                  }
                  return false;
                }
                """,
                label,
            )
            if not clicked:
                page.get_by_text(label, exact=False).last.click(timeout=5_000)
            page.wait_for_timeout(1_000)
            wait_for_settled(page)
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=True,
                    errors=visible_error_markers(page),
                    stale=stale_visible(page),
                )
            )
        except Exception as exc:
            if label not in visible_labels:
                continue
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=False,
                    errors=(f"click_failed: {str(exc).splitlines()[0]}",),
                    stale=stale_visible(page),
                )
            )
    return tuple(checks)


def inspect_visible_part(page, label: str, clicked: bool = True, note: str = "") -> PartCheck:
    data = page.evaluate(
        """
        () => {
          const visible = (el) => {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            return rect.width > 0
              && rect.height > 0
              && rect.bottom >= 0
              && rect.top <= window.innerHeight
              && style.display !== 'none'
              && style.visibility !== 'hidden';
          };
          const textNodes = [];
          const seen = new Set();
          for (const el of document.querySelectorAll('body *')) {
            if (!visible(el)) continue;
            const text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
            if (text && text.length < 2500 && !seen.has(text)) {
              seen.add(text);
              textNodes.push(text);
            }
          }
          const visibleControls = [...document.querySelectorAll('input, select, textarea, button')]
            .filter((el) => visible(el)).length;
          const visiblePlots = [...document.querySelectorAll('svg, canvas, .js-plotly-plot, .plotly')]
            .filter((el) => visible(el)).length;
          const visibleFields = [...document.querySelectorAll('.mlsysbook-field, .mlsysbook-compact-field')]
            .filter((el) => visible(el)).length;
          return {
            text: textNodes.join('\\n'),
            visibleControls,
            visiblePlots,
            visibleFields,
          };
        }
        """
    )
    visible_text = str(data["text"])
    body = page_text(page)
    part_key = label.split(":")[0].split("--")[0].strip()
    start = body.lower().rfind(part_key.lower()) if part_key else -1
    text = body[start:] if start >= 0 else visible_text
    if len(text.strip()) < len(visible_text.strip()):
        text = visible_text
    category_patterns = {
        "narrative": ("scenario", "stakeholder", "incoming message", "objective", "systems question"),
        "prediction": ("prediction", "predict", "select your prediction", "what do you expect", "how much", "which"),
        "controls": ("control", "controls", "slider", "select", "adjust", "switch", "tune", "move", "precision", "model", "budget", "threshold", "knob", "try it", "explore"),
        "evidence": ("evidence", "inspect", "compare", "computed", "result", "frontier", "chart", "plot", "summary", "feasible", "candidate"),
        "source_math": ("source", "math", "formula", "calculation", "solver", "mlsysim"),
        "reflection": ("reflection", "checkpoint", "decision", "decide", "defend", "write", "takeaway", "report", "residual risk", "synthesis"),
    }

    def detected_categories(value: str) -> list[str]:
        lower = value.lower()
        return [
            category
            for category, patterns in category_patterns.items()
            if any(pattern in lower for pattern in patterns)
        ]

    categories = detected_categories(text)
    words = len(re.findall(r"\b\w+\b", text))
    if (words < 45 or len(categories) < 3) and part_key:
        first_start = body.lower().find(part_key.lower())
        if first_start >= 0 and first_start != start:
            first_text = body[first_start:]
            first_categories = detected_categories(first_text)
            first_words = len(re.findall(r"\b\w+\b", first_text))
            if len(first_categories) > len(categories) or first_words > words:
                text = first_text
                categories = first_categories
                words = first_words
    if words < 45 or len(categories) < 3:
        combined_text = f"{text}\n{visible_text}"
        combined_categories = detected_categories(combined_text)
        combined_words = len(re.findall(r"\b\w+\b", combined_text))
        if len(combined_categories) > len(categories) or combined_words > words:
            text = combined_text
            categories = combined_categories
            words = combined_words
    if words < 45 or len(categories) < 3:
        scrolled_text = collect_visible_text_across_scroll(page)
        combined_text = f"{text}\n{scrolled_text}"
        combined_categories = detected_categories(combined_text)
        combined_words = len(re.findall(r"\b\w+\b", combined_text))
        if len(combined_categories) > len(categories) or combined_words > words:
            text = combined_text
            categories = combined_categories
            words = combined_words

    if int(data["visibleControls"]) > 0 and "controls" not in categories:
        categories.append("controls")
    if int(data["visiblePlots"]) > 0 and "evidence" not in categories:
        categories.append("evidence")
    if int(data["visibleFields"]) > 0 and "evidence" not in categories:
        categories.append("evidence")

    lower = text.lower()
    has_synthesis_checkpoint = words >= 100 and "reflection" in categories
    has_multiple_pieces = (words >= 45 and len(categories) >= 3) or has_synthesis_checkpoint
    return PartCheck(
        label=label,
        clicked=clicked,
        text_words=words,
        piece_categories=tuple(categories),
        visible_controls=int(data["visibleControls"]),
        visible_plots=int(data["visiblePlots"]),
        visible_fields=int(data["visibleFields"]),
        has_multiple_pieces=has_multiple_pieces,
        errors=visible_error_markers(page),
        stale=stale_visible(page),
        note=note,
    )


def click_parts(page, max_tabs: int) -> tuple[tuple[ClickCheck, ...], tuple[PartCheck, ...]]:
    tab_checks: list[ClickCheck] = []
    part_checks: list[PartCheck] = []
    visible_labels = set(visible_tab_labels(page, max_tabs=max_tabs))
    for label in candidate_tab_labels(page, max_tabs):
        try:
            page.get_by_text(label, exact=False).last.click(timeout=3_500)
            page.wait_for_timeout(1_000)
            wait_for_settled(page)
            control_clicks = click_common_answer_prefixes(page, max_radios=1)
            if not control_clicks:
                control_clicks = click_first_generic_marimo_radio(page)
            if control_clicks:
                page.wait_for_timeout(1_000)
                wait_for_settled(page)
            errors = visible_error_markers(page)
            stale = stale_visible(page)
            tab_checks.append(ClickCheck(label=label, clicked=True, errors=errors, stale=stale))
            note = ""
            if control_clicks:
                note = f"clicked {control_clicks[0].label}"
            part_checks.append(inspect_visible_part(page, label, clicked=True, note=note))
        except Exception as exc:
            if label not in visible_labels:
                continue
            error = f"click_failed: {str(exc).splitlines()[0]}"
            tab_checks.append(ClickCheck(label=label, clicked=False, errors=(error,), stale=stale_visible(page)))
            part_checks.append(
                PartCheck(
                    label=label,
                    clicked=False,
                    text_words=0,
                    piece_categories=(),
                    visible_controls=0,
                    visible_plots=0,
                    visible_fields=0,
                    has_multiple_pieces=False,
                    errors=(error,),
                    stale=stale_visible(page),
                )
            )
    if not part_checks:
        part_checks.append(inspect_visible_part(page, "Full page", clicked=True, note="no visible part tabs"))
    return tuple(tab_checks), tuple(part_checks)


def answer_option_labels(page) -> tuple[str, ...]:
    labels: list[str] = []
    for raw_line in page_text(page).splitlines():
        line = " ".join(raw_line.strip().split())
        if ANSWER_OPTION_RE.match(line) and line not in labels:
            labels.append(line)
    return tuple(labels)


def click_text_answer_options(page, max_radios: int) -> tuple[ClickCheck, ...]:
    checks: list[ClickCheck] = []
    for label in answer_option_labels(page):
        if len(checks) >= max_radios:
            break
        try:
            page.get_by_text(label, exact=False).first.click(timeout=5_000)
            page.wait_for_timeout(2_000)
            wait_for_settled(page)
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=True,
                    errors=visible_error_markers(page),
                    stale=stale_visible(page),
                )
            )
        except Exception as exc:
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=False,
                    errors=(f"click_failed: {str(exc).splitlines()[0]}",),
                    stale=stale_visible(page),
                )
            )
    return tuple(checks)


def click_common_answer_prefixes(page, max_radios: int) -> tuple[ClickCheck, ...]:
    checks: list[ClickCheck] = []
    text = page_text(page)
    if not any(prefix in text for prefix in ("A)", "B)", "C)", "D)")):
        return ()
    for prefix in ("B)", "C)", "A)", "D)"):
        if len(checks) >= max_radios:
            break
        try:
            page.get_by_text(prefix, exact=False).first.click(timeout=700)
            page.wait_for_timeout(1_200)
            wait_for_settled(page)
            checks.append(
                ClickCheck(
                    label=f"{prefix} visible answer option",
                    clicked=True,
                    errors=visible_error_markers(page),
                    stale=stale_visible(page),
                )
            )
        except PlaywrightTimeoutError:
            continue
        except Exception as exc:
            checks.append(
                ClickCheck(
                    label=f"{prefix} visible answer option",
                    clicked=False,
                    errors=(f"click_failed: {str(exc).splitlines()[0]}",),
                    stale=stale_visible(page),
                )
            )
    return tuple(checks)


def radio_candidates(page) -> tuple[tuple[int, str], ...]:
    candidates = page.evaluate(
        """
        () => {
          const candidates = [];
          const inputs = [...document.querySelectorAll('input[type="radio"]')];
          inputs.forEach((input, index) => {
            const label = input.closest('label');
            const text = (label?.innerText || label?.textContent || input.value || '').trim().replace(/\\s+/g, ' ');
            const rect = input.getBoundingClientRect();
            if (text && rect.width > 0 && rect.height > 0) {
              candidates.push([index, text]);
            }
          });
          return candidates;
        }
        """
    )
    return tuple((int(index), str(label)) for index, label in candidates)


def click_non_track_radios(page, max_radios: int) -> tuple[ClickCheck, ...]:
    if max_radios <= 0:
        return ()
    checks: list[ClickCheck] = list(click_common_answer_prefixes(page, max_radios=max_radios))
    if len(checks) >= max_radios:
        return tuple(checks[:max_radios])
    checks.extend(click_text_answer_options(page, max_radios=max_radios - len(checks)))
    if len(checks) >= max_radios:
        return tuple(checks[:max_radios])
    clicked_labels: set[str] = set()
    while len(checks) < max_radios:
        next_candidate = None
        for index, label in radio_candidates(page):
            if label in clicked_labels:
                continue
            if label in TRACKS or any(label.startswith(track) for track in TRACKS):
                continue
            next_candidate = (index, label)
            break
        if next_candidate is None:
            break
        index, label = next_candidate
        clicked_labels.add(label)
        if len(checks) >= max_radios:
            break
        try:
            page.locator('input[type="radio"]').nth(index).click(force=True, timeout=5_000)
            page.wait_for_timeout(2_000)
            wait_for_settled(page)
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=True,
                    errors=visible_error_markers(page),
                    stale=stale_visible(page),
                )
            )
        except Exception as exc:
            checks.append(
                ClickCheck(
                    label=label,
                    clicked=False,
                    errors=(f"click_failed: {str(exc).splitlines()[0]}",),
                    stale=stale_visible(page),
                )
            )
    return tuple(checks)


def exercise_tabs_and_radios(page, max_tabs: int, max_radios: int) -> tuple[tuple[ClickCheck, ...], tuple[ClickCheck, ...]]:
    tab_checks: list[ClickCheck] = []
    radio_checks: list[ClickCheck] = []
    if max_radios <= 0:
        return click_tabs(page, max_tabs=max_tabs), ()
    radios_per_tab = max(1, min(2, max_radios))
    for tab_check in click_tabs(page, max_tabs=max_tabs):
        tab_checks.append(tab_check)
        if not tab_check.clicked or tab_check.errors:
            continue
        radio_checks.extend(click_non_track_radios(page, max_radios=radios_per_tab))
        if len(radio_checks) >= max_radios:
            break
    if len(radio_checks) < max_radios:
        radio_checks.extend(click_non_track_radios(page, max_radios=max_radios - len(radio_checks)))
    return tuple(tab_checks), tuple(radio_checks[:max_radios])


def server_error_lines(output: str) -> tuple[str, ...]:
    lines = []
    for line in output.splitlines():
        if any(marker in line for marker in ERROR_MARKERS):
            lines.append(line.strip())
    return tuple(lines[:12])


def check_lab(playwright, lab: Path, url: str, screenshot: Path, max_tabs: int, max_radios: int) -> LabInteractionCheck:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1400}, accept_downloads=True)
    console_errors: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda err: console_errors.append(str(err)))

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        loaded = wait_for_lab_ready(page)
        page.wait_for_timeout(1_000)
        page_errors = visible_error_markers(page)
        scroll_positions = scroll_through(page)
        unlock_checks = (
            unlock_lab00_orientation(page)
            if lab.name == "lab_00_introduction.py"
            else ()
        )
        track_checks = click_track_options(page)
        tab_checks, part_checks = click_parts(page, max_tabs=max_tabs)
        radio_checks = click_non_track_radios(page, max_radios=max_radios)
        page.wait_for_timeout(1_000)
        wait_for_settled(page)
        page_errors = tuple(dict.fromkeys(page_errors + visible_error_markers(page)))
        horizontal_overflow_px, overflowing_fields, offscreen_left_elements = layout_metrics(page)
        body_length = len(page_text(page))
        page.screenshot(path=str(screenshot), full_page=True)
        return LabInteractionCheck(
            lab=str(lab),
            url=url,
            loaded=loaded or body_length > 200,
            body_length=body_length,
            screenshot=str(screenshot),
            page_errors=page_errors,
            console_errors=tuple(dict.fromkeys(console_errors)),
            horizontal_overflow_px=horizontal_overflow_px,
            offscreen_left_elements=offscreen_left_elements,
            overflowing_fields=overflowing_fields,
            scroll_positions=scroll_positions,
            track_checks=unlock_checks + track_checks,
            tab_checks=tab_checks,
            part_checks=part_checks,
            radio_checks=radio_checks,
            server_errors=(),
        )
    finally:
        browser.close()


def check_html_page(playwright, path: Path, screenshot: Path) -> HtmlPageCheck:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1400})
    console_errors: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda err: console_errors.append(str(err)))
    url = path.resolve().as_uri()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(500)
        scroll_positions = scroll_through(page)
        checks: dict[str, int | bool] = {}
        if path.name == "lab-plan-dashboard.html":
            checks = page.evaluate(
                """
                () => {
                  const search = document.getElementById('searchBox');
                  const before = document.querySelectorAll('#activityBody tr').length;
                  search.value = 'compression';
                  search.dispatchEvent(new Event('input', { bubbles: true }));
                  const filtered = document.querySelectorAll('#activityBody tr').length;
                  const v2 = document.querySelector('[data-volume="V2"]');
                  v2.click();
                  const v2Rows = document.querySelectorAll('#activityBody tr').length;
                  return {
                    trackCards: document.querySelectorAll('#trackGrid .track-card').length === 4,
                    arcCards: document.querySelectorAll('#arcGrid .track-card').length === 4,
                    coverageRows: document.querySelectorAll('#coverageBody tr').length === 34,
                    conceptRows: document.querySelectorAll('#conceptBody tr').length === 14,
                    activityRowsBeforeFilter: before === 103,
                    searchFiltersRows: filtered > 0 && filtered < before,
                    volumeFilterWorks: v2Rows > 0 && v2Rows < before,
                  };
                }
                """
            )
        else:
            checks = page.evaluate(
                """
                () => ({
                  hasMainContent: document.body.innerText.trim().length > 1000,
                  hasCatalogCards: document.querySelectorAll('section, table, article, .card').length > 0,
                })
                """
            )
        horizontal_overflow_px = int(
            page.evaluate(
                "() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth)"
            )
        )
        text = page_text(page)
        page_errors = tuple(marker for marker in ERROR_MARKERS if marker in text)
        page.screenshot(path=str(screenshot), full_page=True)
        return HtmlPageCheck(
            page=str(path),
            url=url,
            loaded=len(text.strip()) > 100,
            screenshot=str(screenshot),
            page_errors=page_errors,
            console_errors=tuple(dict.fromkeys(console_errors)),
            horizontal_overflow_px=horizontal_overflow_px,
            scroll_positions=scroll_positions,
            checks={key: bool(value) for key, value in checks.items()},
        )
    finally:
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs", nargs="*", type=Path, default=None, help="Lab .py files to audit.")
    parser.add_argument("--html-pages", nargs="*", type=Path, default=list(DEFAULT_HTML_PAGES))
    parser.add_argument("--port-start", type=int, default=30400)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/mlsysbook-interaction-smoke"))
    parser.add_argument("--max-tabs", type=int, default=8)
    parser.add_argument("--max-radios", type=int, default=4)
    args = parser.parse_args()

    labs = args.labs if args.labs is not None and len(args.labs) > 0 else default_labs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lab_results: list[LabInteractionCheck] = []
    html_results: list[HtmlPageCheck] = []

    with sync_playwright() as playwright:
        for index, lab in enumerate(labs):
            port = args.port_start + index
            url = f"http://127.0.0.1:{port}"
            screenshot = args.output_dir / f"{lab.stem}.png"
            process = start_server(lab, port)
            output = ""
            try:
                wait_for_server(url)
                result = check_lab(
                    playwright,
                    lab,
                    url,
                    screenshot,
                    max_tabs=args.max_tabs,
                    max_radios=args.max_radios,
                )
            except Exception as exc:
                result = LabInteractionCheck(
                    lab=str(lab),
                    url=url,
                    loaded=False,
                    body_length=0,
                    screenshot=str(screenshot),
                    page_errors=(f"audit_failed: {str(exc).splitlines()[0]}",),
                    console_errors=(),
                    horizontal_overflow_px=0,
                    offscreen_left_elements=0,
                    overflowing_fields=0,
                    scroll_positions=(),
                    track_checks=(),
                    tab_checks=(),
                    part_checks=(),
                    radio_checks=(),
                    server_errors=(),
                )
            finally:
                output = stop_server(process)
            result = replace(result, server_errors=server_error_lines(output))
            lab_results.append(result)

        for path in args.html_pages:
            screenshot = args.output_dir / f"{path.stem}.png"
            try:
                html_results.append(check_html_page(playwright, path, screenshot))
            except Exception as exc:
                html_results.append(
                    HtmlPageCheck(
                        page=str(path),
                        url=path.resolve().as_uri(),
                        loaded=False,
                        screenshot=str(screenshot),
                        page_errors=(f"audit_failed: {str(exc).splitlines()[0]}",),
                        console_errors=(),
                        horizontal_overflow_px=0,
                        scroll_positions=(),
                        checks={},
                    )
                )

    payload = {
        "labs": [asdict(result) | {"passed": result.passed} for result in lab_results],
        "html_pages": [asdict(result) | {"passed": result.passed} for result in html_results],
        "summary": {
            "lab_count": len(lab_results),
            "lab_passed": sum(result.passed for result in lab_results),
            "html_count": len(html_results),
            "html_passed": sum(result.passed for result in html_results),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0 if all(result.passed for result in lab_results) and all(result.passed for result in html_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
