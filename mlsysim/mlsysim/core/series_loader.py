"""Lightweight CSV table loader for chapter-local and deck-local figure series."""
from __future__ import annotations

import csv as _csv
import functools as _functools
import math as _math
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[3]
_BOOKS_DIR = _REPO_ROOT / "books"


def _coerce_csv_value(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        v = float(value)
        return v if _math.isfinite(v) else value
    except ValueError:
        return value


class LocalSeries:
    """A figure-local table loaded lazily from a chapter or deck ``data/*.csv`` file.

    Leading ``#`` lines before the CSV header row are stored on
    ``provenance_comments`` so each plot dataset carries its source provenance
    directly alongside the figure.
    """

    def __init__(self, path: str | _Path, numeric: tuple[str, ...] | None = None):
        p = _Path(path)
        if not p.exists():
            if (_BOOKS_DIR / p).exists():
                p = _BOOKS_DIR / p
            elif (_REPO_ROOT / p).exists():
                p = _REPO_ROOT / p
            else:
                raise FileNotFoundError(f"Figure data series not found: {path}")
        self.path = p.resolve()
        self._numeric = tuple(numeric) if numeric is not None else None
        self._provenance_comments: tuple[str, ...] | None = None
        self._columns: tuple[str, ...] | None = None
        self._entries: tuple[dict, ...] | None = None

    def _ensure_loaded(self) -> None:
        if self._entries is not None:
            return
        raw_lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        comments: list[str] = []
        data_start = 0
        for idx, line in enumerate(raw_lines):
            if line.lstrip().startswith("#"):
                comments.append(line.strip())
                data_start = idx + 1
            else:
                break
        self._provenance_comments = tuple(comments)
        reader = _csv.DictReader(raw_lines[data_start:])
        self._columns = tuple(reader.fieldnames or ())
        if self._numeric is not None:
            num_set = set(self._numeric)
            self._entries = tuple(
                {
                    k: (float(v) if v not in ("", None) else None) if k in num_set else v
                    for k, v in row.items()
                }
                for row in reader
            )
        else:
            self._entries = tuple({k: _coerce_csv_value(v) for k, v in row.items()} for row in reader)

    @property
    def provenance_comments(self) -> tuple[str, ...]:
        self._ensure_loaded()
        return self._provenance_comments  # type: ignore[return-value]

    @property
    def columns(self) -> tuple[str, ...]:
        self._ensure_loaded()
        return self._columns  # type: ignore[return-value]

    @property
    def entries(self) -> tuple[dict, ...]:
        self._ensure_loaded()
        return self._entries  # type: ignore[return-value]

    @property
    def list(self) -> tuple[dict, ...]:
        return self.entries

    def where(self, **match):
        return tuple(e for e in self.entries if all(e.get(k) == v for k, v in match.items()))

    def column(self, key: str) -> list:
        return [r[key] for r in self.entries]

    def largest(self, key: str, date_key: str | None = None, on_or_before: str | None = None) -> dict:
        rows = [
            r for r in self.entries
            if r[key] is not None and (on_or_before is None or str(r[date_key]) <= str(on_or_before))
        ]
        return max(rows, key=lambda r: r[key])

    def find(self, key: str, fragment: str) -> dict:
        return next(r for r in self.entries if fragment in str(r[key]))

    def __len__(self) -> int:
        return len(self.entries)


@_functools.lru_cache(maxsize=None)
def _cached_load_series(path_str: str, numeric: tuple[str, ...] | None) -> LocalSeries:
    return LocalSeries(path_str, numeric=numeric)


def load_series(path: str | _Path, numeric: tuple[str, ...] | None = None) -> LocalSeries:
    """Load a chapter-local or figure-local CSV series."""
    num_tuple = tuple(numeric) if numeric is not None else None
    return _cached_load_series(str(path), num_tuple)
