"""Topic → textbook chapter resolution for the StaffML "Learn more" funnel.

Every question carries a ``topic`` (one of the 87 curated taxonomy ids). The
``schema/topic_chapter_map.yaml`` file maps each topic to the book chapter(s)
that develop it. This module joins the two so the build can emit a ``book_refs``
list onto each question — a "go deeper" pointer back into the textbook, derived
once per topic rather than hand-authored across all ~10.7k questions.

This is a *go-deeper pointer, not an answer key*: the reference is tied to the
question's topic, not to where its solution lives. See
``interviews/staffml/docs/proposals/book-refs-analysis.md`` and
harvard-edge/cs249r_book#1822.

URL pattern (verified live, HTTP 200):

    https://mlsysbook.ai/vol{N}/contents/vol{N}/{chapter}/{chapter}.html

Chapter titles are read from the chapter's ``.qmd`` H1 when the book tree is
available; that same pass is the build-time **link-checker** — a mapped chapter
whose ``.qmd`` source is missing fails the build, which is what turns the old
"defer until URLs stabilize" blocker into "URLs are enforced valid". When the
book tree is absent (vault-cli used standalone, outside the monorepo) titles
fall back to slug title-casing and the link-check is skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from vault_cli import yaml_io

# Default location of the topic→chapter map, relative to the vault dir.
_MAP_RELPATH = ("schema", "topic_chapter_map.yaml")

# Book source tree, relative to the repo root (vault_dir.parent.parent).
_CONTENTS_RELPATH = ("book", "quarto", "contents")

# `# Chapter Title {#sec-...}` — capture the title, drop the optional anchor.
_H1_RE = re.compile(r"^#\s+(.+?)\s*(?:\{#.*\})?\s*$")


class BookRefError(ValueError):
    """A mapped chapter has no corresponding ``.qmd`` source (link-check failure)."""


def _chapter_url(vol: int, chapter: str) -> str:
    return f"https://mlsysbook.ai/vol{vol}/contents/vol{vol}/{chapter}/{chapter}.html"


def _slug_title(chapter: str) -> str:
    """Fallback display title when the book tree is unavailable."""
    return chapter.replace("_", " ").title()


class BookRefResolver:
    """Resolves a question ``topic`` to its textbook ``book_refs``.

    Construct once per build and reuse — the map is read once, chapter titles
    are cached, and the link-check runs eagerly at construction so a bad map
    fails the build before any artifact is written.
    """

    def __init__(self, vault_dir: Path, contents_root: Path | None = None) -> None:
        self._map = _load_map(vault_dir)
        if contents_root is None:
            # vault_dir is interviews/vault → repo root is two levels up.
            contents_root = vault_dir.parent.parent.joinpath(*_CONTENTS_RELPATH)
        self._contents_root = contents_root
        self._title_cache: dict[tuple[int, str], str] = {}
        # Only enforce the link-check when the book tree is actually present.
        self._link_check = contents_root.exists()
        if self._link_check:
            self._validate()

    # -- public API --------------------------------------------------------

    def refs_for_topic(self, topic: str) -> list[dict[str, Any]]:
        """Return the ordered book_refs for a topic (primary first, then also_see).

        Empty list when the topic is unmapped — the frontend simply renders no
        "Learn more" card in that case.
        """
        entry = self._map.get(topic)
        if not entry:
            return []
        refs: list[dict[str, Any]] = []
        primary = entry.get("primary")
        if primary:
            refs.append(self._ref(primary, role="primary", why=entry.get("why")))
        for also in entry.get("also_see") or []:
            refs.append(self._ref(also, role="also_see"))
        return refs

    # -- internals ---------------------------------------------------------

    def _ref(
        self, raw: dict[str, Any], *, role: str, why: str | None = None
    ) -> dict[str, Any]:
        vol = int(raw["vol"])
        chapter = str(raw["chapter"])
        ref: dict[str, Any] = {
            "vol": vol,
            "chapter": chapter,
            "title": self._title_for(vol, chapter),
            "url": _chapter_url(vol, chapter),
            "role": role,
        }
        # `why` is authored per-topic and describes the primary mapping, so it
        # only rides along on the primary ref.
        if role == "primary" and why:
            ref["why"] = why
        return ref

    def _qmd_path(self, vol: int, chapter: str) -> Path:
        return self._contents_root / f"vol{vol}" / chapter / f"{chapter}.qmd"

    def _title_for(self, vol: int, chapter: str) -> str:
        key = (vol, chapter)
        if key in self._title_cache:
            return self._title_cache[key]
        title = self._read_h1(self._qmd_path(vol, chapter)) or _slug_title(chapter)
        self._title_cache[key] = title
        return title

    @staticmethod
    def _read_h1(qmd: Path) -> str | None:
        if not qmd.exists():
            return None
        for line in qmd.read_text(encoding="utf-8").splitlines():
            m = _H1_RE.match(line)
            if m:
                return m.group(1).strip()
        return None

    def _validate(self) -> None:
        """Fail the build if any mapped chapter lacks a ``.qmd`` source."""
        missing: list[str] = []
        for topic, entry in self._map.items():
            if not isinstance(entry, dict):
                continue
            for raw in [entry.get("primary"), *(entry.get("also_see") or [])]:
                if not raw:
                    continue
                vol, chapter = int(raw["vol"]), str(raw["chapter"])
                if not self._qmd_path(vol, chapter).exists():
                    missing.append(f"{topic} → vol{vol}/{chapter} ({self._qmd_path(vol, chapter)})")
        if missing:
            raise BookRefError(
                "topic_chapter_map.yaml references chapters with no .qmd source:\n  "
                + "\n  ".join(sorted(set(missing)))
            )


def _load_map(vault_dir: Path) -> dict[str, Any]:
    path = vault_dir.joinpath(*_MAP_RELPATH)
    if not path.exists():
        return {}
    data = yaml_io.load_file(path)
    return data if isinstance(data, dict) else {}


__all__ = ["BookRefError", "BookRefResolver"]
