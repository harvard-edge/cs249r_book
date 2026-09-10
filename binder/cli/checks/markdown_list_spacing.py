"""Flag bold lead-in paragraphs that are immediately followed by lists.

Pandoc treats a paragraph followed directly by ``- item`` as one paragraph,
not as a separate list block. In rendered output this can collapse into text
like ``**Step 2**: ... - item``. This check catches the high-signal house
style case: a non-list paragraph containing Markdown bold immediately followed
by a bullet or numbered list item with no blank line between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


BOLD_RE = re.compile(r"(?<!\\)\*\*[^*\n]+\*\*")
FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+\S|\d+[.)]\s+\S)")


@dataclass(frozen=True)
class MarkdownListSpacingIssue:
    """A missing blank line before a Markdown list."""

    line_number: int
    previous_line: str
    list_line: str

    @property
    def context(self) -> str:
        return f"{self.previous_line} / {self.list_line}"

    @property
    def suggestion(self) -> str:
        return f"{self.previous_line}\n{self.list_line} -> {self.previous_line}\n\n{self.list_line}"


def _is_list_item(line: str) -> bool:
    return bool(LIST_ITEM_RE.match(line))


def _is_skippable_previous_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _is_list_item(line):
        return True
    if stripped.startswith(("#", "#|", "%%|", "|", ":::", "```", "~~~", "<!--")):
        return True
    if stripped in {"---", "..."}:
        return True
    return False


def _is_bold_leadin(line: str) -> bool:
    if _is_skippable_previous_line(line):
        return False
    return bool(BOLD_RE.search(line))


def check_text(text: str) -> list[MarkdownListSpacingIssue]:
    """Return missing-blank issues in one QMD source string."""
    lines = text.splitlines()
    issues: list[MarkdownListSpacingIssue] = []
    in_yaml = False
    in_fence = False

    for i, line in enumerate(lines[:-1]):
        stripped = line.strip()

        if i == 0 and stripped == "---":
            in_yaml = True
            continue
        if in_yaml:
            if stripped == "---":
                in_yaml = False
            continue

        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        next_line = lines[i + 1]
        if not _is_list_item(next_line):
            continue
        if not _is_bold_leadin(line):
            continue

        issues.append(
            MarkdownListSpacingIssue(
                line_number=i + 2,
                previous_line=line.strip(),
                list_line=next_line.strip(),
            )
        )

    return issues


def check_file(path: Path) -> list[MarkdownListSpacingIssue]:
    """Return missing-blank issues for one QMD file."""
    return check_text(path.read_text(encoding="utf-8"))
