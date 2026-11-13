"""Lightweight CSV table loader for chapter-local and deck-local figure series.

Tier-2 figure and table CSVs under ``books/volN/<chapter>/data/*.csv`` and
``slides/volN/<deck>/images/data/*.csv`` carry structured leading ``#`` metadata
lines before the header row so every empirical plot dataset keeps its source,
column schema, and computation lineage co-located with the rows:

    # Figures: vol1:fig-ai-timeline
    # Source: Primary citation or dataset title (@bibkey when in references.bib)
    # Locator: Exact table, figure, section, API endpoint, or archive file
    # URL: https://...
    # License: CC-BY-4.0 | Factual-Extract (...) | ...
    # Verified: YYYY-MM-DD
    # Epistemic: measured | modeled | assumed | constructed (plus boundary scope)
    # Method: transcribed | digitized | scripted | derived (plus optional detail)
    # Generator: python3 binder/tools/scripts/<script>.py ...
    # Upstream-SHA256: <64-hex digest of raw upstream snapshot, when applicable>
    # Computation: Filtering criteria, exclusions, units, and derivation formulas
    # Columns:
    #   year: integer [year] — Milestone publication year
    #   value: number [FLOP] — Reported training compute
    # Notes: Caveats and accounting boundaries

``LocalSeries.provenance`` parses these ``# Key: Value`` lines into a normalized
read-only mapping (for example ``series.provenance["source"]``,
``series.provenance["locator"]``, ``series.provenance["epistemic"]``), while
``LocalSeries.column_schema`` exposes structured :class:`ColumnSpec` descriptors.
"""
from __future__ import annotations

import csv as _csv
from dataclasses import dataclass as _dataclass
import functools as _functools
import hashlib as _hashlib
import math as _math
import re as _re
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType
from typing import Mapping as _Mapping

_REPO_ROOT = _Path(__file__).resolve().parents[3]
_BOOKS_DIR = _REPO_ROOT / "books"
_HEADER_KEY_RE = _re.compile(r"^#\s?([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$")
_CONTINUATION_RE = _re.compile(r"^#\s{2,}(\S.*)$")
_INLINE_VERIFIED_RE = _re.compile(r"\|\s*Verified\s*:\s*(\d{4}-\d{2}-\d{2})\b", _re.I)
_COLUMN_SPEC_RE = _re.compile(
    r"^([A-Za-z0-9_.-]+)\s*:\s*([A-Za-z0-9_-]+)\s+\[([^\[\]\n]+)\]\s*(?:—|--|-)\s*(.+)$"
)


@_dataclass(frozen=True)
class ColumnSpec:
    """Typed column descriptor parsed from a CSV ``# Columns:`` header block."""

    name: str
    type: str
    unit: str
    description: str


def parse_column_schema(raw_columns: str) -> tuple[tuple[ColumnSpec, ...], tuple[str, ...]]:
    """Parse newline-delimited ``# Columns:`` entries into :class:`ColumnSpec` items.

    Returns ``(specs, malformed_lines)`` so validators can report syntax errors on
    individual column declarations without losing parsed entries.
    """
    if not raw_columns.strip():
        return (), ()
    specs: list[ColumnSpec] = []
    malformed: list[str] = []
    for raw_line in raw_columns.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _COLUMN_SPEC_RE.match(line)
        if not m:
            malformed.append(line)
            continue
        specs.append(
            ColumnSpec(
                name=m.group(1).strip(),
                type=m.group(2).strip().lower(),
                unit=m.group(3).strip(),
                description=m.group(4).strip(),
            )
        )
    return tuple(specs), tuple(malformed)


def parse_series_provenance(comments: tuple[str, ...] | list[str]) -> _Mapping[str, str]:
    """Parse leading ``# Key: Value`` provenance lines into an immutable mapping.

    Keys are normalized to lowercase ``snake_case`` (for example,
    ``Upstream-SHA256`` becomes ``upstream_sha256``). Indented ``#   ...`` lines
    continue the preceding field (newline-joined for ``columns``, space-joined
    for prose fields), and inline ``| Verified: YYYY-MM-DD`` tokens inside
    ``# License:`` lines populate ``verified`` when no standalone
    ``# Verified:`` line is present.
    """
    fields: dict[str, str] = {}
    last_key: str | None = None
    for raw in comments:
        m = _HEADER_KEY_RE.match(raw)
        if m:
            key = m.group(1).strip().lower().replace("-", "_")
            val = m.group(2).strip()
            if key == "license" and "verified" not in fields:
                vm = _INLINE_VERIFIED_RE.search(val)
                if vm:
                    fields["verified"] = vm.group(1)
            if key in fields and fields[key] and val:
                sep = "\n" if key == "columns" else " "
                fields[key] = f"{fields[key]}{sep}{val}"
            elif key not in fields:
                fields[key] = val
            last_key = key
            continue
        cm = _CONTINUATION_RE.match(raw)
        if cm and last_key is not None:
            item = cm.group(1).strip()
            if last_key == "columns":
                fields[last_key] = f"{fields[last_key]}\n{item}".strip() if fields[last_key] else item
            else:
                fields[last_key] = f"{fields[last_key]} {item}".strip() if fields[last_key] else item
        else:
            last_key = None
    return _MappingProxyType(fields)


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
        self.path = _resolve_series_path(path)
        self._numeric = tuple(numeric) if numeric is not None else None
        self._provenance_comments: tuple[str, ...] | None = None
        self._provenance: _Mapping[str, str] | None = None
        self._column_schema: tuple[ColumnSpec, ...] | None = None
        self._malformed_column_lines: tuple[str, ...] | None = None
        self._columns: tuple[str, ...] | None = None
        self._raw_rows: tuple[dict[str, str], ...] | None = None
        self._entries: tuple[dict, ...] | None = None
        self._file_bytes: int | None = None
        self._file_sha256: str | None = None
        self._data_sha256: str | None = None

    def _ensure_loaded(self) -> None:
        if self._entries is not None:
            return
        raw_bytes = self.path.read_bytes()
        self._file_bytes = len(raw_bytes)
        self._file_sha256 = _hashlib.sha256(raw_bytes).hexdigest()
        raw_lines = raw_bytes.decode("utf-8").splitlines(keepends=True)
        comments: list[str] = []
        data_start = 0
        for idx, line in enumerate(raw_lines):
            if line.lstrip().startswith("#"):
                comments.append(line.strip())
                data_start = idx + 1
            else:
                break
        self._provenance_comments = tuple(comments)
        self._provenance = parse_series_provenance(self._provenance_comments)
        self._column_schema, self._malformed_column_lines = parse_column_schema(
            self._provenance.get("columns", "")
        )
        canonical_data = "".join(line.rstrip("\r\n") + "\n" for line in raw_lines[data_start:])
        self._data_sha256 = _hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()
        reader = _csv.DictReader(raw_lines[data_start:])
        self._columns = tuple(reader.fieldnames or ())
        raw_rows = tuple(
            {k: (v if v is not None else "") for k, v in row.items() if k is not None}
            for row in reader
        )
        self._raw_rows = raw_rows
        if self._numeric is not None:
            num_set = set(self._numeric)
            self._entries = tuple(
                {
                    k: (float(v) if v not in ("", None) else None) if k in num_set else v
                    for k, v in row.items()
                }
                for row in raw_rows
            )
        else:
            self._entries = tuple({k: _coerce_csv_value(v) for k, v in row.items()} for row in raw_rows)

    @property
    def provenance_comments(self) -> tuple[str, ...]:
        self._ensure_loaded()
        return self._provenance_comments  # type: ignore[return-value]

    @property
    def provenance(self) -> _Mapping[str, str]:
        """Structured ``# Key: Value`` provenance fields parsed from the CSV header."""
        self._ensure_loaded()
        return self._provenance  # type: ignore[return-value]

    @property
    def column_schema(self) -> tuple[ColumnSpec, ...]:
        """Typed :class:`ColumnSpec` entries parsed from ``# Columns:``."""
        self._ensure_loaded()
        return self._column_schema  # type: ignore[return-value]

    @property
    def malformed_column_lines(self) -> tuple[str, ...]:
        """Any unparsable lines encountered inside ``# Columns:``."""
        self._ensure_loaded()
        return self._malformed_column_lines  # type: ignore[return-value]

    @property
    def file_bytes(self) -> int:
        self._ensure_loaded()
        return self._file_bytes  # type: ignore[return-value]

    @property
    def file_sha256(self) -> str:
        """SHA-256 hex digest of the full UTF-8 CSV file bytes (comments + data)."""
        self._ensure_loaded()
        return self._file_sha256  # type: ignore[return-value]

    @property
    def data_sha256(self) -> str:
        """SHA-256 hex digest of LF-normalized CSV header and data rows (excluding ``#`` headers)."""
        self._ensure_loaded()
        return self._data_sha256  # type: ignore[return-value]

    @property
    def raw_rows(self) -> tuple[dict[str, str], ...]:
        """Uncoerced string rows as read by :class:`csv.DictReader`."""
        self._ensure_loaded()
        return self._raw_rows  # type: ignore[return-value]

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

    @property
    def rows(self) -> tuple[dict, ...]:
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


def _resolve_series_path(path: str | _Path) -> _Path:
    p = _Path(path)
    if not p.exists():
        if (_BOOKS_DIR / p).exists():
            p = _BOOKS_DIR / p
        elif (_REPO_ROOT / p).exists():
            p = _REPO_ROOT / p
        else:
            raise FileNotFoundError(f"Figure data series not found: {path}")
    return p.resolve()


@_functools.lru_cache(maxsize=None)
def _cached_load_series(
    resolved_path: str,
    mtime_ns: int,
    size_bytes: int,
    numeric: tuple[str, ...] | None,
) -> LocalSeries:
    del mtime_ns, size_bytes
    return LocalSeries(resolved_path, numeric=numeric)


def load_series(path: str | _Path, numeric: tuple[str, ...] | None = None) -> LocalSeries:
    """Load a chapter-local or figure-local CSV series."""
    resolved = _resolve_series_path(path)
    st = resolved.stat()
    num_tuple = tuple(numeric) if numeric is not None else None
    return _cached_load_series(str(resolved), st.st_mtime_ns, st.st_size, num_tuple)


load_series.cache_clear = _cached_load_series.cache_clear  # type: ignore[attr-defined]
