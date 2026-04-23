"""Preflight validation for draft markdown before pushing to Buttondown.

A single entry point (`validate_draft`) returns a structured report that
the CLI renders. Both `news check` (user-invoked) and `news push`
(automatic) call this so we never push something broken.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Severity = Literal["ok", "warn", "error"]

# Same regex as PushCommand. Local image refs only.
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(((?!https?://)[^)]+)\)")

# Frontmatter fields we care about.
REQUIRED_FIELDS = ("title", "date", "author", "categories", "description")
VALID_CATEGORIES = {"essay", "community", "hands-on"}
RUNTIME_DEPS = ("rich", "requests", "frontmatter")


@dataclass
class Check:
    """One line item in the preflight report."""

    name: str
    severity: Severity
    message: str


@dataclass
class Report:
    """Preflight result. `ok` iff no ERROR-severity checks."""

    draft: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.severity == "error" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.severity == "warn" for c in self.checks)


def _check_dependencies() -> list[Check]:
    """Make sure all runtime deps are importable."""
    results: list[Check] = []
    for dep in RUNTIME_DEPS:
        try:
            __import__(dep)
            results.append(Check(f"dep: {dep}", "ok", "installed"))
        except ImportError:
            results.append(
                Check(f"dep: {dep}", "error", "not installed (pip install -r site/newsletter/requirements.txt)")
            )
    return results


def _check_frontmatter(post) -> list[Check]:
    """Every required frontmatter field is present and well-formed."""
    results: list[Check] = []
    meta = post.metadata

    for field_name in REQUIRED_FIELDS:
        value = meta.get(field_name)
        if not value:
            results.append(
                Check(f"frontmatter: {field_name}", "error", "missing or empty")
            )
        else:
            # Short descriptions often fit on one line; no need to echo value.
            results.append(Check(f"frontmatter: {field_name}", "ok", "present"))

    # Category must be one of the allowed values.
    categories = meta.get("categories") or []
    if categories:
        first = categories[0] if isinstance(categories, list) else categories
        if first not in VALID_CATEGORIES:
            results.append(
                Check(
                    "frontmatter: categories",
                    "error",
                    f"{first!r} is not one of {sorted(VALID_CATEGORIES)}",
                )
            )

    # Date format sanity check.
    date_val = str(meta.get("date", ""))
    if date_val and not re.match(r"^\d{4}-\d{2}-\d{2}", date_val):
        results.append(
            Check(
                "frontmatter: date",
                "warn",
                f"{date_val!r} is not YYYY-MM-DD format; update before publishing",
            )
        )

    # Still marked draft? That's a warning at check-time, intentional before send.
    if meta.get("draft") is True:
        results.append(
            Check(
                "frontmatter: draft",
                "warn",
                "still `draft: true` (fine for pushing to Buttondown; must remove before archive)",
            )
        )

    return results


def _check_images(content: str, draft_path: Path) -> list[Check]:
    """Every local image referenced in the markdown exists on disk."""
    results: list[Check] = []
    draft_dir = draft_path.parent
    matches = list(LOCAL_IMAGE_RE.finditer(content))

    if not matches:
        results.append(Check("images", "ok", "no local images referenced"))
        return results

    for match in matches:
        local_path = match.group(2).strip()
        resolved = (draft_dir / local_path).resolve()
        if resolved.exists():
            size_kb = resolved.stat().st_size // 1024
            results.append(
                Check(f"image: {local_path}", "ok", f"{size_kb} KB")
            )
        else:
            results.append(
                Check(f"image: {local_path}", "error", f"not found at {resolved}")
            )
    return results


def _check_dashes(content: str) -> list[Check]:
    """Em-dashes and en-dashes are a known style issue for this newsletter."""
    em_count = content.count("\u2014")
    en_count = content.count("\u2013")
    if em_count + en_count == 0:
        return [Check("style: dashes", "ok", "no em/en dashes")]
    return [
        Check(
            "style: dashes",
            "warn",
            f"found {em_count} em-dash(es) and {en_count} en-dash(es)",
        )
    ]


def _check_api_key(config_env_file: Path) -> list[Check]:
    """The Buttondown API key is set in env or the .env file."""
    import os

    if os.environ.get("BUTTONDOWN_API_KEY"):
        return [Check("auth", "ok", "BUTTONDOWN_API_KEY in environment")]
    if config_env_file.exists():
        text = config_env_file.read_text()
        for line in text.splitlines():
            if line.startswith("BUTTONDOWN_API_KEY=") and line.split("=", 1)[1].strip().strip('"\''):
                return [Check("auth", "ok", f"BUTTONDOWN_API_KEY in {config_env_file.name}")]
    return [
        Check(
            "auth",
            "error",
            f"BUTTONDOWN_API_KEY is not set (check {config_env_file} or env)",
        )
    ]


def _check_git_installed() -> list[Check]:
    """Ensure git is reachable so the archive/commit workflow works."""
    if shutil.which("git"):
        return [Check("git", "ok", "available on PATH")]
    return [Check("git", "warn", "git not on PATH; archive step will need manual move")]


def validate_draft(draft_path: Path, env_file: Path) -> Report:
    """Run all preflight checks. Never raises; returns a Report."""
    report = Report(draft=draft_path)

    # Always run: dependencies and git.
    report.checks.extend(_check_dependencies())
    report.checks.extend(_check_git_installed())

    # Requires frontmatter lib. If missing, dep check above already errored.
    try:
        import frontmatter
    except ImportError:
        report.checks.append(
            Check(
                "draft",
                "error",
                "cannot parse draft without python-frontmatter",
            )
        )
        return report

    if not draft_path.exists():
        report.checks.append(Check("draft", "error", f"file not found: {draft_path}"))
        return report

    post = frontmatter.load(draft_path)
    report.checks.extend(_check_frontmatter(post))
    report.checks.extend(_check_images(post.content, draft_path))
    report.checks.extend(_check_dashes(post.content))
    report.checks.extend(_check_api_key(env_file))

    return report
