#!/usr/bin/env python3
"""Provenance, schema, anti-tamper, and Frictionless Data Package gates for figure datasets.

This powers::

    ./binder/binder check figures --scope data-provenance
    ./binder/binder check figures --scope cell-provenance
    python3 binder/cli/checks/figure_data_provenance.py --seal
    python3 binder/cli/checks/figure_data_provenance.py --seal-data --reason "<provenance note>"

Why this exists:
  Tier-1 constants in ``mlsysim/`` are gated by ``audit_provenance.py`` and
  case-study callouts are gated by ``case_study_provenance.py``. Tier-2
  empirical plot and table series (``books/volN/<chapter>/data/*.csv`` loaded via
  ``load_series(...)``) follow the Tier-2 Data Provenance Standard v1: an inline
  ``# Key: Value`` header with typed per-column units, mandatory cross-sealing to
  ``bibs/manifest.json``, side-door loader guards on ``.qmd`` chapters, and a
  cryptographically sealed Frictionless Data Package manifest at
  ``books/datapackage.json``.

Anti-reward-hacking controls:
  1. ``binder check --fix`` never rewrites ``books/datapackage.json``.
  2. ``--seal`` is metadata-only: it validates all checks first and refuses to
     overwrite ``books/datapackage.json`` if any resource's ``data_sha256`` row
     digest changed or if CSV files were added or removed.
  3. ``--seal-data --reason "<>=5 words>"`` is required to admit new CSVs or
     row-level changes, and authenticates ``# URL:`` live over HTTP(S).
  4. Every book CSV must cite at least one ``@bibkey`` sealed as ``verified`` in
     ``bibs/manifest.json`` with matching normalized BibTeX SHA-256 digest.
  5. Direct file readers (``pd.read_csv``, ``csv.reader``, ``np.loadtxt``, ...),
     non-literal ``load_series(var)`` calls, stray tabular/binary files outside
     ``data/``, ``auxiliary (...)`` paths that do not reference the CSV, and
     inline empirical float literal arrays in ``measured`` figure cells all fail.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import re
import shlex
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from cli.checks.generated_paths import is_generated
except ImportError:  # pragma: no cover - standalone or package-qualified import
    from binder.cli.checks.generated_paths import is_generated

REPO_ROOT = Path(__file__).resolve().parents[3]
MLSYSIM_DIR = REPO_ROOT / "mlsysim"
if str(MLSYSIM_DIR) not in sys.path:
    sys.path.insert(0, str(MLSYSIM_DIR))

from mlsysim.core.series_loader import ColumnSpec, LocalSeries, load_series

VALID_EPISTEMIC = frozenset({"measured", "modeled", "assumed", "constructed"})
VALID_METHODS = frozenset({"transcribed", "digitized", "scripted", "derived"})
VALID_COLUMN_TYPES = frozenset({"integer", "number", "string", "boolean", "date", "datetime"})
CITATION_COLUMNS = frozenset({"source_key", "bib_key", "bibkey", "citekey", "citation_key"})
PLACEHOLDER_UNITS = frozenset({"unit", "units", "value", "values", "tbd", "todo", "unknown", "?", "n/a", "none"})
STANDALONE_PLACEHOLDERS = frozenset({"tbd", "todo", "fixme", "xxx", "n/a", "na", "none", "unknown", "null", "placeholder"})
AUXILIARY_ALLOWED_PREFIXES = ("mlsysim/", "slides/", "binder/", "labs/", "shared/")
AUXILIARY_ALLOWED_SUFFIXES = (".py", ".qmd", ".sh")
STRAY_DATASET_SUFFIXES = frozenset({".csv", ".tsv", ".parquet", ".feather", ".npy", ".npz", ".pkl", ".xlsx"})

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
DUMMY_HOST_RE = re.compile(r"^https?://(?:[^/]+\.)?(?:example\.(?:com|org|net)|localhost|127\.0\.0\.1|0\.0\.0\.0)\b", re.IGNORECASE)
PLACEHOLDER_TOKEN_RE = re.compile(r"\b(?:TODO|TBD|FIXME|XXX|PLACEHOLDER|LOREM\s+IPSUM)\b", re.IGNORECASE)
# Hex-escaped banned reassurance tokens per evidence-and-provenance rule 8.
BANNED_REASSURANCE_RE = re.compile(
    r"\b(?:\x67enuine|\x73trictly|label\x2dblind|zero\s+blending|clean\x2droom)\b"
    r"|100\x25"
    r"|\b\x72eal(?!-(?:time|world|valued|life|estate|number))\b",
    re.IGNORECASE,
)
INT_RE = re.compile(r"^-?\d+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MIN_VERIFIED_DATE = dt.date(2020, 1, 1)
FIGURE_TARGET_RE = re.compile(r"^(vol[1-4]):((?:fig|tbl|sec)-[A-Za-z0-9_-]+)$")
AUXILIARY_TARGET_RE = re.compile(r"^auxiliary\s*\(\s*([^()\s]+)\s*\)$")
ANCHOR_ID_RE = re.compile(r"\{#((?:fig|tbl|sec)-[A-Za-z0-9_-]+)")
LOAD_SERIES_RE = re.compile(r"""\bload_series\(\s*["']([^"']+)["']""")
DYNAMIC_LOAD_SERIES_RE = re.compile(r"""\bload_series\(\s*(?!["'\s])([^)\n]+)""")
BYPASS_READER_RE = re.compile(
    r"\b(?:"
    r"(?:pd|pandas|pl|polars)\s*\.\s*read_(?:csv|tsv|table|parquet|feather|excel|fwf|orc)"
    r"|csv\s*\.\s*(?:reader|DictReader)"
    r"|(?:np|numpy)\s*\.\s*(?:loadtxt|genfromtxt|fromfile|load)"
    r"|pickle\s*\.\s*load"
    r"|pyarrow\s*\.\s*csv"
    r")\s*\("
)
RAW_CSV_OPEN_RE = re.compile(
    r"""\b(?:open|Path)\s*\([^)\n]*\.(?:csv|tsv|parquet|npz|npy)["']""",
)
TIKZ_CSV_SOURCE_RE = re.compile(r"(?m)^\s*%\s*Source:\s*(books/vol[1-4]/[^\s]+\.csv)\b")
DIV_OPEN_RE = re.compile(r"^(:{3,})\s*\{#((?:fig|tbl)-[^\s}]+)")
DIV_FENCE_RE = re.compile(r"^(:{3,})[ \t]*$")
FIG_DIV_RE = re.compile(r"(?m)^(:{3,})\s*\{#(fig-[^\s}]+)[^\n]*\n(.*?)^\1[ \t]*$", re.DOTALL)
PY_CELL_RE = re.compile(r"```\{python\}\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
PROVENANCE_LINE_RE = re.compile(r"^\s*#.*?\bProvenance\s*:", re.MULTILINE)
CELL_EPISTEMIC_RE = re.compile(
    r"\b(?:measured|modeled|assumed|constructed|analytical|schematic|illustrative|derived|simulated)\b",
    re.IGNORECASE,
)
POS_MEASURED_RE = re.compile(r"(?<!\bnot\s)(?<!\bnon-)\bmeasured\b", re.IGNORECASE)
MLSYSIM_RECEIPT_RE = re.compile(r"\b(?:mlsysim|Scenarios|Specs|Literature|Constants)\b")
LAYOUT_ARRAY_NAME_RE = re.compile(
    r"(?:tick|lim|figsize|ratio|width|height|alpha|zorder|dash|pad|margin|box|level)",
    re.IGNORECASE,
)
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CITEKEY_RE = re.compile(r"(?<![A-Za-z0-9._+-])@([A-Za-z][A-Za-z0-9_:+.-]*[A-Za-z0-9])")
NOT_A_CITEKEY_RE = re.compile(
    r"^(fig|sec|tbl|eq|lst|thm|def|dfn|pri|nbk|exm|exmp|cs|ws|chk|psp|lhs)-", re.IGNORECASE
)
REPO_SCRIPT_PREFIXES = ("binder/", "shared/", "mlsysim/", "books/", "slides/", "scripts/")
DATAPACKAGE_REL_PATH = Path("books/datapackage.json")
EPOCH_RECEIPT_REL_PATH = Path("binder/tools/audits/epoch_sustainability_manifest.json")


@dataclass(frozen=True)
class ProvenanceViolation:
    """One figure-data or figure-cell provenance violation."""

    file: str
    line: int
    code: str
    message: str
    severity: str = "error"
    context: str = ""
    suggestion: str = ""

    def to_issue_kwargs(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
            "suggestion": self.suggestion,
        }


@dataclass
class BibVerificationContext:
    """Cached bibliography keys and cryptographic ``bibs/manifest.json`` state."""

    has_manifest: bool
    manifest: dict[str, dict[str, Any]]
    bib_text_by_vol: dict[str | None, str]
    keys_by_vol: dict[str | None, set[str]]
    entry_hash_cache: dict[tuple[str | None, str], str | None]


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _is_book_csv(csv_path: Path, repo_root: Path) -> bool:
    try:
        parts = csv_path.resolve().relative_to((repo_root / "books").resolve()).parts
    except ValueError:
        return False
    return bool(parts and parts[0].startswith("vol") and "data" in parts and csv_path.suffix == ".csv")


def _load_bib_context(repo_root: Path) -> BibVerificationContext:
    manifest_path = repo_root / "bibs" / "manifest.json"
    manifest: dict[str, dict[str, Any]] = {}
    has_manifest = manifest_path.is_file()
    if has_manifest:
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                manifest = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, ValueError, TypeError):
            manifest = {}
    return BibVerificationContext(
        has_manifest=has_manifest,
        manifest=manifest,
        bib_text_by_vol={},
        keys_by_vol={},
        entry_hash_cache={},
    )


def _bib_text_and_keys(vol_name: str | None, repo_root: Path, ctx: BibVerificationContext) -> tuple[str, set[str]]:
    if vol_name in ctx.keys_by_vol:
        return ctx.bib_text_by_vol[vol_name], ctx.keys_by_vol[vol_name]
    candidates: list[Path] = []
    if vol_name:
        candidates.append(repo_root / "books" / f"references-{vol_name}.bib")
    candidates.extend([repo_root / "books" / "references.bib", repo_root / "bibs" / "references.bib"])
    for cand in candidates:
        if cand.exists():
            text = cand.read_text(encoding="utf-8", errors="replace")
            keys = set(re.findall(r"^@\w+\{([^,\s]+)\s*,", text, re.MULTILINE))
            ctx.bib_text_by_vol[vol_name] = text
            ctx.keys_by_vol[vol_name] = keys
            return text, keys
    ctx.bib_text_by_vol[vol_name] = ""
    ctx.keys_by_vol[vol_name] = set()
    return "", set()


def _compute_bib_entry_hash(vol_name: str | None, key: str, bib_text: str, ctx: BibVerificationContext) -> str | None:
    cache_key = (vol_name, key)
    if cache_key in ctx.entry_hash_cache:
        return ctx.entry_hash_cache[cache_key]
    m = re.search(rf"^@\w+\{{\s*{re.escape(key)}\s*,", bib_text, re.MULTILINE)
    if not m:
        ctx.entry_hash_cache[cache_key] = None
        return None
    start = m.start()
    brace_open = bib_text.find("{", start)
    if brace_open == -1:
        ctx.entry_hash_cache[cache_key] = None
        return None
    depth = 0
    idx = brace_open
    n = len(bib_text)
    while idx < n:
        ch = bib_text[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw_entry = bib_text[start : idx + 1]
                normalized = re.sub(r"\s+", " ", raw_entry.strip())
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                ctx.entry_hash_cache[cache_key] = digest
                return digest
        idx += 1
    ctx.entry_hash_cache[cache_key] = None
    return None


def _anchor_ids_for_volume(vol_name: str, repo_root: Path) -> set[str]:
    vol_dir = repo_root / "books" / vol_name
    if not vol_dir.is_dir():
        return set()
    anchors: set[str] = set()
    for qmd in sorted(vol_dir.rglob("*.qmd")):
        if is_generated(qmd, repo_root):
            continue
        text = qmd.read_text(encoding="utf-8", errors="replace")
        anchors.update(ANCHOR_ID_RE.findall(text))
    return anchors


def _volume_of(path: Path, repo_root: Path) -> str | None:
    try:
        parts = path.resolve().relative_to((repo_root / "books").resolve()).parts
    except ValueError:
        return None
    if parts and parts[0].startswith("vol"):
        return parts[0]
    return None


def _extract_citekeys(text: str) -> list[str]:
    keys: list[str] = []
    for match in CITEKEY_RE.finditer(text):
        key = match.group(1)
        if not NOT_A_CITEKEY_RE.match(key):
            keys.append(key)
    return keys


def _word_count(text: str) -> int:
    return len(WORD_TOKEN_RE.findall(text))


def _has_placeholder(text: str) -> str | None:
    stripped = text.strip().lower()
    if stripped in STANDALONE_PLACEHOLDERS:
        return text.strip()
    m = PLACEHOLDER_TOKEN_RE.search(text)
    if m:
        return m.group(0)
    return None


def parse_figure_targets(raw_figures: str) -> list[str]:
    """Split a comma-separated ``# Figures:`` value into trimmed target tokens."""
    if not raw_figures.strip():
        return []
    return [tok.strip() for tok in raw_figures.split(",") if tok.strip()]


def _cell_matches_type(raw_val: str, col_type: str) -> bool:
    if raw_val == "":
        return True
    if col_type == "string":
        return True
    if col_type == "integer":
        return bool(INT_RE.match(raw_val))
    if col_type == "number":
        if raw_val.lower() in {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"}:
            return False
        try:
            return math.isfinite(float(raw_val))
        except ValueError:
            return False
    if col_type == "boolean":
        return raw_val in {"True", "False", "true", "false"}
    if col_type == "date":
        if not ISO_DATE_RE.match(raw_val):
            return False
        try:
            dt.date.fromisoformat(raw_val)
            return True
        except ValueError:
            return False
    if col_type == "datetime":
        if "T" not in raw_val and " " not in raw_val:
            return False
        try:
            dt.datetime.fromisoformat(raw_val.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    return False


def _validate_rectangular_csv_structure(csv_path: Path, rel: str) -> list[ProvenanceViolation]:
    """Verify non-empty, unique-header, rectangular CSV rows without silent DictReader padding."""
    violations: list[ProvenanceViolation] = []
    raw_lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    comment_count = 0
    for ln in raw_lines:
        if ln.lstrip().startswith("#"):
            comment_count += 1
        else:
            break
    data_lines = raw_lines[comment_count:]
    parsed_rows = list(csv.reader(data_lines))
    header_lineno = comment_count + 1

    if len(parsed_rows) < 2 or not parsed_rows[0]:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=header_lineno,
                code="series_empty_rows",
                message=f"{csv_path.name} must contain a header row and at least one data row.",
                context=data_lines[0] if data_lines else "(empty data section)",
                suggestion="Populate the empirical data rows or remove unused empty CSV stubs.",
            )
        )
        return violations

    header = parsed_rows[0]
    if any(not h.strip() for h in header) or len(header) != len(set(header)):
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=header_lineno,
                code="series_empty_rows",
                message=f"{csv_path.name} CSV header row contains blank or duplicate column names: {header}.",
                context=",".join(header),
                suggestion="Give every CSV column a unique, non-empty snake_case identifier.",
            )
        )
        return violations

    ncols = len(header)
    for offset, row in enumerate(parsed_rows[1:], start=1):
        if len(row) != ncols:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=header_lineno + offset,
                    code="series_ragged_row",
                    message=(
                        f"{csv_path.name} data row {offset} has {len(row)} field(s), "
                        f"but header declares {ncols} column(s)."
                    ),
                    context=",".join(row)[:140],
                    suggestion="Quote fields containing commas and ensure every row matches the header column count.",
                )
            )
            return violations

    for col_idx, col_name in enumerate(header):
        if all(row[col_idx].strip() == "" for row in parsed_rows[1:]):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=header_lineno,
                    code="series_empty_column",
                    message=f"{csv_path.name} column '{col_name}' is blank across every row.",
                    context=col_name,
                    suggestion="Populate values for the column or drop unused blank placeholder columns.",
                )
            )

    return violations


def _validate_columns_block(
    series: LocalSeries,
    rel: str,
    csv_name: str,
    first_comment: str,
) -> list[ProvenanceViolation]:
    violations: list[ProvenanceViolation] = []
    raw_columns = series.provenance.get("columns", "").strip()
    if not raw_columns and not series.column_schema and not series.malformed_column_lines:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_missing_columns",
                message=f"{csv_name} is missing a '# Columns:' typed column-schema block.",
                context=first_comment,
                suggestion=(
                    "Add '# Columns:' followed by '#   <col>: <type> [<unit>] — <description>' "
                    "for every CSV column in header order."
                ),
            )
        )
        return violations

    for bad_line in series.malformed_column_lines:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_invalid_column_spec",
                message=(
                    f"{csv_name} has malformed '# Columns:' entry {bad_line!r}; "
                    "expected '<name>: <type> [<unit>] — <description>'."
                ),
                context=f"#   {bad_line}",
                suggestion=(
                    "Format each indented column entry as "
                    "'#   <name>: <integer|number|string|boolean|date|datetime> [<unit>] — <description>'."
                ),
            )
        )

    for spec in series.column_schema:
        if spec.type not in VALID_COLUMN_TYPES:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_invalid_column_spec",
                    message=(
                        f"{csv_name} column '{spec.name}' declares unsupported type '{spec.type}'; "
                        f"expected one of {sorted(VALID_COLUMN_TYPES)}."
                    ),
                    context=f"#   {spec.name}: {spec.type} [{spec.unit}] — {spec.description}",
                    suggestion=f"Use a Frictionless Table Schema v1 type from {sorted(VALID_COLUMN_TYPES)}.",
                )
            )
        if spec.unit.strip().lower() in PLACEHOLDER_UNITS or _has_placeholder(spec.unit):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_placeholder_metadata",
                    message=(
                        f"{csv_name} column '{spec.name}' uses placeholder unit [{spec.unit}]; "
                        "specify concrete physical, currency, or semantic units."
                    ),
                    context=f"#   {spec.name}: {spec.type} [{spec.unit}] — {spec.description}",
                    suggestion="Use concrete units such as [ms], [GB/s], [FLOP], [USD], [ratio], [year], or [identifier].",
                )
            )
        ph_desc = _has_placeholder(spec.description)
        if ph_desc:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_placeholder_metadata",
                    message=f"{csv_name} column '{spec.name}' description contains placeholder token {ph_desc!r}.",
                    context=f"#   {spec.name}: {spec.type} [{spec.unit}] — {spec.description}",
                    suggestion="Write a concrete domain description of what the column measures or identifies.",
                )
            )
        elif (
            _word_count(spec.description) < 3
            or spec.description.strip().lower().replace(" ", "_") == spec.name.strip().lower()
        ):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_vacuous_metadata",
                    message=(
                        f"{csv_name} column '{spec.name}' has a vacuous description ({spec.description!r}); "
                        "at least 3 words of domain meaning are required."
                    ),
                    context=f"#   {spec.name}: {spec.type} [{spec.unit}] — {spec.description}",
                    suggestion="Describe the column's physical quantity, denominator, or semantic role in >= 3 words.",
                )
            )

    if series.malformed_column_lines:
        return violations

    declared_names = tuple(spec.name for spec in series.column_schema)
    actual_names = series.columns
    if declared_names != actual_names:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_column_mismatch",
                message=(
                    f"{csv_name} '# Columns:' schema {list(declared_names)} does not match "
                    f"CSV header columns {list(actual_names)} in exact order."
                ),
                context=",".join(actual_names),
                suggestion="List every CSV header column in exact 1-to-1 order inside '# Columns:'.",
            )
        )
        return violations

    spec_by_name: dict[str, ColumnSpec] = {spec.name: spec for spec in series.column_schema}
    header_line_no = len(series.provenance_comments) + 1
    for col_name, spec in spec_by_name.items():
        if spec.type not in VALID_COLUMN_TYPES:
            continue
        for row_idx, row in enumerate(series.raw_rows):
            cell = row.get(col_name, "")
            if not _cell_matches_type(cell, spec.type):
                violations.append(
                    ProvenanceViolation(
                        file=rel,
                        line=header_line_no + 1 + row_idx,
                        code="series_column_type_mismatch",
                        message=(
                            f"{csv_name} column '{col_name}' (declared '{spec.type}') has incompatible "
                            f"value {cell!r} at data row {row_idx + 1}."
                        ),
                        context=f"{col_name}={cell!r}",
                        suggestion=(
                            f"Update the value at row {row_idx + 1} or change '{col_name}' type in '# Columns:' "
                            "(for example, use 'number' when values contain decimal points)."
                        ),
                    )
                )
                break

    return violations


def _verify_citekey_against_ledger(
    key: str,
    vol: str | None,
    rel: str,
    lineno: int,
    csv_name: str,
    context_str: str,
    repo_root: Path,
    bib_ctx: BibVerificationContext,
) -> list[ProvenanceViolation]:
    bib_text, known_keys = _bib_text_and_keys(vol, repo_root, bib_ctx)
    if known_keys and key not in known_keys:
        return [
            ProvenanceViolation(
                file=rel,
                line=lineno,
                code="series_unresolved_citekey",
                message=f"{csv_name} cites @{key}, which is absent from the volume bibliography.",
                context=context_str,
                suggestion=(
                    "Authenticate and add the primary source via './binder/binder bib add ...' "
                    "or use an existing verified key from bibs/references.bib."
                ),
            )
        ]

    if bib_ctx.has_manifest:
        rec = bib_ctx.manifest.get(key)
        if rec is None or rec.get("status") != "verified":
            return [
                ProvenanceViolation(
                    file=rel,
                    line=lineno,
                    code="series_unverified_citekey",
                    message=(
                        f"{csv_name} cites @{key}, which is not cryptographically sealed as 'verified' "
                        "in bibs/manifest.json."
                    ),
                    context=context_str,
                    suggestion="Authenticate the source via './binder/binder bib add' so bibs/manifest.json seals it.",
                )
            ]
        expected_hash = rec.get("entry_hash")
        if expected_hash and bib_text:
            actual_hash = _compute_bib_entry_hash(vol, key, bib_text, bib_ctx)
            if actual_hash is not None and actual_hash != expected_hash:
                return [
                    ProvenanceViolation(
                        file=rel,
                        line=lineno,
                        code="series_unverified_citekey",
                        message=(
                            f"{csv_name} cites @{key}, whose BibTeX entry in references.bib drifted from "
                            "its cryptographic SHA-256 seal in bibs/manifest.json."
                        ),
                        context=context_str,
                        suggestion="Restore the sealed BibTeX entry or re-verify it via './binder/binder bib reseal'.",
                    )
                ]
    return []


def check_csv_file(
    csv_path: Path,
    repo_root: Path = REPO_ROOT,
    bib_cache: BibVerificationContext | dict[str | None, set[str]] | None = None,
    anchor_cache: dict[str, set[str]] | None = None,
    strict_v1: bool | None = None,
) -> list[ProvenanceViolation]:
    """Validate structured provenance metadata, anti-hack invariants, and column schema on one CSV."""
    rel = _rel(csv_path, repo_root)
    enforce_v1 = _is_book_csv(csv_path, repo_root) if strict_v1 is None else strict_v1
    series = load_series(csv_path)
    prov = series.provenance
    comments = series.provenance_comments
    first_comment = comments[0] if comments else ""
    violations: list[ProvenanceViolation] = []

    if isinstance(bib_cache, BibVerificationContext):
        bib_ctx = bib_cache
    else:
        bib_ctx = _load_bib_context(repo_root)

    # Check banned reassurance tokens across all comment lines
    for idx, cmt in enumerate(comments, start=1):
        bm = BANNED_REASSURANCE_RE.search(cmt)
        if bm:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=idx,
                    code="series_banned_reassurance_word",
                    message=(
                        f"{csv_path.name} provenance comment contains banned reassurance token "
                        f"{bm.group(0)!r}; state the measurement or derivation mechanism instead."
                    ),
                    context=cmt[:140],
                    suggestion="Remove reassurance wording and state the concrete measurement boundary or formula.",
                )
            )
            break

    source = prov.get("source", "").strip()
    if not source:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_missing_source",
                message=f"{csv_path.name} is missing a '# Source:' provenance header line.",
                context=first_comment or "(no leading '#' header lines)",
                suggestion="Add '# Source: Author (Year), Title (@bibkey)' in the CSV header.",
            )
        )
    elif _has_placeholder(source):
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_placeholder_metadata",
                message=f"{csv_path.name} '# Source:' contains placeholder text ({_has_placeholder(source)!r}).",
                context=f"# Source: {source}",
                suggestion="Cite the exact primary publication, benchmark archive, or specification with (@bibkey).",
            )
        )

    verified = prov.get("verified", "").strip()
    if not verified:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_missing_verified",
                message=f"{csv_path.name} is missing a '# Verified: YYYY-MM-DD' provenance line.",
                context=first_comment,
                suggestion="Add '# Verified: YYYY-MM-DD' recording when the CSV was checked against its primary source.",
            )
        )
    else:
        try:
            vdate = dt.date.fromisoformat(verified)
            today = dt.date.today()
            if vdate < MIN_VERIFIED_DATE or vdate > today:
                violations.append(
                    ProvenanceViolation(
                        file=rel,
                        line=1,
                        code="series_invalid_verified",
                        message=(
                            f"{csv_path.name} has out-of-range '# Verified: {verified}'; "
                            f"date must lie between {MIN_VERIFIED_DATE.isoformat()} and {today.isoformat()}."
                        ),
                        context=f"# Verified: {verified}",
                        suggestion="Record the actual date (YYYY-MM-DD, not in the future) when the source was verified.",
                    )
                )
        except ValueError:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_invalid_verified",
                    message=f"{csv_path.name} has non-ISO '# Verified: {verified}'; expected YYYY-MM-DD.",
                    context=f"# Verified: {verified}",
                    suggestion="Format the verification date as ISO YYYY-MM-DD (for example, 2026-10-12).",
                )
            )

    computation = prov.get("computation", "").strip()
    notes = prov.get("notes", "").strip()
    if not computation and not notes:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_missing_computation_or_notes",
                message=f"{csv_path.name} lacks both '# Computation:' and '# Notes:' provenance lines.",
                context=first_comment,
                suggestion=(
                    "Add '# Computation: <filters, units, and derivation formulas>' and/or "
                    "'# Notes: <transcription scope and accounting boundaries>'."
                ),
            )
        )
    elif enforce_v1:
        for field_label, val in (("# Computation:", computation), ("# Notes:", notes)):
            if val and _has_placeholder(val):
                violations.append(
                    ProvenanceViolation(
                        file=rel,
                        line=1,
                        code="series_placeholder_metadata",
                        message=f"{csv_path.name} {field_label} contains placeholder token {_has_placeholder(val)!r}.",
                        context=f"{field_label} {val[:120]}",
                        suggestion=f"Replace placeholder text in {field_label} with concrete filtering or derivation details.",
                    )
                )
        if _word_count(f"{computation} {notes}") < 6:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_vacuous_metadata",
                    message=(
                        f"{csv_path.name} '# Computation:' / '# Notes:' is too brief (< 6 words) "
                        "to document row selection, units, or accounting boundaries."
                    ),
                    context=computation or notes,
                    suggestion="Document exact inclusion/exclusion rules, unit conversions, or transcription scope (>= 6 words).",
                )
            )

    epistemic = prov.get("epistemic", "").strip()
    if not epistemic:
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_missing_epistemic",
                message=(
                    f"{csv_path.name} is missing a '# Epistemic: <measured|modeled|assumed|constructed>' "
                    "provenance line."
                ),
                context=first_comment,
                suggestion=(
                    "Add '# Epistemic: <measured|modeled|assumed|constructed> (<accounting boundary>)' "
                    "to the CSV header."
                ),
            )
        )
    else:
        head_token = re.split(r"[\s(;,:]+", epistemic, maxsplit=1)[0].lower()
        if head_token not in VALID_EPISTEMIC:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_invalid_epistemic",
                    message=(
                        f"{csv_path.name} has '# Epistemic: {epistemic}'; first token must be one of "
                        f"{sorted(VALID_EPISTEMIC)}."
                    ),
                    context=f"# Epistemic: {epistemic}",
                    suggestion=(
                        "Start '# Epistemic:' with measured, modeled, assumed, or constructed, "
                        "followed by boundary details."
                    ),
                )
            )
        elif enforce_v1 and (_word_count(epistemic) < 3 or _has_placeholder(epistemic)):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_vacuous_metadata",
                    message=(
                        f"{csv_path.name} '# Epistemic: {epistemic}' omits the accounting boundary; "
                        "state what was measured, modeled, assumed, or constructed (>= 3 words)."
                    ),
                    context=f"# Epistemic: {epistemic}",
                    suggestion="Format as '# Epistemic: measured (<instrument, benchmark split, or system boundary>)'.",
                )
            )

    vol = _volume_of(csv_path, repo_root)
    cited_header_keys = _extract_citekeys(
        " ".join(v for k, v in prov.items() if k in {"source", "locator", "computation", "notes"})
    )

    if enforce_v1:
        violations.extend(_validate_rectangular_csv_structure(csv_path, rel))

        if not cited_header_keys:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_missing_citekey",
                    message=(
                        f"{csv_path.name} '# Source:' cites no @bibkey; every Tier-2 book dataset must "
                        "anchor to at least one verified primary entry in bibs/manifest.json."
                    ),
                    context=f"# Source: {source}",
                    suggestion=(
                        "Authenticate the primary source via './binder/binder bib add ...' and cite "
                        "its @bibkey in '# Source:'."
                    ),
                )
            )

        figures_raw = prov.get("figures", "").strip()
        targets = parse_figure_targets(figures_raw)
        if not targets:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_missing_figures",
                    message=f"{csv_path.name} is missing a '# Figures:' target back-link header line.",
                    context=first_comment,
                    suggestion=(
                        "Add '# Figures: volN:fig-<slug>' (or comma-separated volN:fig-*/tbl-*/sec-* targets, "
                        "or 'auxiliary (<repo_path>)') at line 1."
                    ),
                )
            )
        else:
            acache = anchor_cache if anchor_cache is not None else {}
            for tok in targets:
                aux_match = AUXILIARY_TARGET_RE.match(tok)
                if aux_match:
                    aux_rel = aux_match.group(1)
                    aux_path = repo_root / aux_rel
                    if (
                        not aux_rel.startswith(AUXILIARY_ALLOWED_PREFIXES)
                        or not aux_rel.endswith(AUXILIARY_ALLOWED_SUFFIXES)
                        or not aux_path.is_file()
                    ):
                        violations.append(
                            ProvenanceViolation(
                                file=rel,
                                line=1,
                                code="series_unresolved_figure_target",
                                message=(
                                    f"{csv_path.name} '# Figures:' auxiliary consumer '{aux_rel}' must be an "
                                    f"existing {AUXILIARY_ALLOWED_SUFFIXES} file under {AUXILIARY_ALLOWED_PREFIXES}."
                                ),
                                context=f"# Figures: {figures_raw}",
                                suggestion="Point 'auxiliary (<repo_path>)' to the test, deck, or script that loads this CSV.",
                            )
                        )
                    else:
                        aux_text = aux_path.read_text(encoding="utf-8", errors="replace")
                        if csv_path.name not in aux_text:
                            violations.append(
                                ProvenanceViolation(
                                    file=rel,
                                    line=1,
                                    code="series_unresolved_figure_target",
                                    message=(
                                        f"{csv_path.name} declares 'auxiliary ({aux_rel})', "
                                        f"but '{aux_rel}' never references '{csv_path.name}'."
                                    ),
                                    context=f"# Figures: {figures_raw}",
                                    suggestion=(
                                        "Do not park unreferenced CSVs on unrelated files; bind '# Figures:' "
                                        "to a chapter figure/table or a script/test that loads this CSV."
                                    ),
                                )
                            )
                    continue
                fm = FIGURE_TARGET_RE.match(tok)
                if not fm:
                    violations.append(
                        ProvenanceViolation(
                            file=rel,
                            line=1,
                            code="series_unresolved_figure_target",
                            message=(
                                f"{csv_path.name} has malformed '# Figures:' target {tok!r}; "
                                "expected 'vol<1-4>:(fig|tbl|sec)-<slug>' or 'auxiliary (<repo_path>)'."
                            ),
                            context=f"# Figures: {figures_raw}",
                            suggestion="Format targets as 'vol1:fig-example', 'vol2:tbl-example', or 'vol3:sec-example'.",
                        )
                    )
                    continue
                target_vol, anchor_id = fm.group(1), fm.group(2)
                if target_vol not in acache:
                    acache[target_vol] = _anchor_ids_for_volume(target_vol, repo_root)
                if acache[target_vol] and anchor_id not in acache[target_vol]:
                    violations.append(
                        ProvenanceViolation(
                            file=rel,
                            line=1,
                            code="series_unresolved_figure_target",
                            message=(
                                f"{csv_path.name} '# Figures:' target '{tok}' was not found in "
                                f"books/{target_vol}/."
                            ),
                            context=f"# Figures: {figures_raw}",
                            suggestion=f"Use an existing {{#{anchor_id}}} figure, table, or section ID in books/{target_vol}/.",
                        )
                    )

        locator = prov.get("locator", "").strip()
        if not locator:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_missing_locator",
                    message=f"{csv_path.name} is missing a '# Locator:' line pinpointing where values appear in Source.",
                    context=first_comment,
                    suggestion=(
                        "Add '# Locator: <Table X, p. Y | Figure Z | API endpoint | snapshot filename>' "
                        "to the CSV header."
                    ),
                )
            )
        elif _has_placeholder(locator):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_placeholder_metadata",
                    message=f"{csv_path.name} '# Locator:' contains placeholder text ({_has_placeholder(locator)!r}).",
                    context=f"# Locator: {locator}",
                    suggestion="Pinpoint the exact table, figure, page, API endpoint, or archive file inside Source.",
                )
            )
        elif _word_count(locator) < 2 or len(locator) < 8:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_vacuous_metadata",
                    message=(
                        f"{csv_path.name} '# Locator: {locator}' is too vague; name the specific "
                        "table, figure, section, endpoint, or archive filename."
                    ),
                    context=f"# Locator: {locator}",
                    suggestion="Specify e.g. 'Table 3, p. 7' or 'benchmark_data.zip: results.csv' (>= 2 words, >= 8 chars).",
                )
            )

        url = prov.get("url", "").strip()
        if not url:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_missing_url",
                    message=f"{csv_path.name} is missing a '# URL:' provenance header line.",
                    context=first_comment,
                    suggestion="Add '# URL: https://...' pointing to the primary paper, dataset landing page, or archive.",
                )
            )
        elif not URL_RE.match(url):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_invalid_url",
                    message=f"{csv_path.name} has invalid '# URL: {url}'; expected a single http(s):// URL.",
                    context=f"# URL: {url}",
                    suggestion="Provide a valid https:// or http:// URL without spaces.",
                )
            )
        elif DUMMY_HOST_RE.match(url) or _has_placeholder(url):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_placeholder_metadata",
                    message=f"{csv_path.name} '# URL: {url}' points to a dummy or placeholder host.",
                    context=f"# URL: {url}",
                    suggestion="Point '# URL:' to the authoritative DOI, arXiv, archive, or vendor specification URL.",
                )
            )

        license_val = prov.get("license", "").strip()
        if not license_val:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_missing_license",
                    message=f"{csv_path.name} is missing a '# License:' provenance header line.",
                    context=first_comment,
                    suggestion=(
                        "Add '# License: <CC-BY-4.0 | Apache-2.0 | Public-Domain | "
                        "Factual-Extract (...)>' to the CSV header."
                    ),
                )
            )
        elif _has_placeholder(license_val):
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_placeholder_metadata",
                    message=f"{csv_path.name} '# License: {license_val}' uses placeholder text.",
                    context=f"# License: {license_val}",
                    suggestion="Declare SPDX license (e.g. CC-BY-4.0, Apache-2.0, MIT) or 'Factual-Extract (<basis>)'.",
                )
            )

        method_val = prov.get("method", "").strip()
        if not method_val:
            violations.append(
                ProvenanceViolation(
                    file=rel,
                    line=1,
                    code="series_missing_method",
                    message=(
                        f"{csv_path.name} is missing a '# Method: <transcribed|digitized|scripted|derived>' "
                        "provenance line."
                    ),
                    context=first_comment,
                    suggestion=(
                        "Add '# Method: <transcribed|digitized|scripted|derived> [(<detail>)]' "
                        "describing how rows were obtained."
                    ),
                )
            )
        else:
            method_head = re.split(r"[\s(;,:]+", method_val, maxsplit=1)[0].lower()
            if method_head not in VALID_METHODS:
                violations.append(
                    ProvenanceViolation(
                        file=rel,
                        line=1,
                        code="series_invalid_method",
                        message=(
                            f"{csv_path.name} has '# Method: {method_val}'; first token must be one of "
                            f"{sorted(VALID_METHODS)}."
                        ),
                        context=f"# Method: {method_val}",
                        suggestion="Start '# Method:' with transcribed, digitized, scripted, or derived.",
                    )
                )
            elif _has_placeholder(method_val):
                violations.append(
                    ProvenanceViolation(
                        file=rel,
                        line=1,
                        code="series_placeholder_metadata",
                        message=f"{csv_path.name} '# Method:' contains placeholder text.",
                        context=f"# Method: {method_val}",
                        suggestion="Describe the actual extraction or transcription mechanism.",
                    )
                )

        violations.extend(_validate_columns_block(series, rel, csv_path.name, first_comment))

        # Validate per-row citation columns (e.g. source_key) against bibs/manifest.json
        header_line_no = len(comments) + 1
        for cite_col in [c for c in series.columns if c in CITATION_COLUMNS]:
            for row_idx, row in enumerate(series.raw_rows):
                raw_cell = row.get(cite_col, "").strip()
                if not raw_cell:
                    continue
                for raw_tok in raw_cell.replace(";", ",").split(","):
                    ck = raw_tok.strip().lstrip("@")
                    if not ck:
                        continue
                    row_v = _verify_citekey_against_ledger(
                        ck,
                        vol=vol,
                        rel=rel,
                        lineno=header_line_no + 1 + row_idx,
                        csv_name=csv_path.name,
                        context_str=f"{cite_col}={raw_cell!r}",
                        repo_root=repo_root,
                        bib_ctx=bib_ctx,
                    )
                    if row_v:
                        violations.extend(row_v)
                        break

    upstream_sha = prov.get("upstream_sha256", "").strip()
    if upstream_sha and not SHA256_RE.match(upstream_sha):
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_invalid_sha256",
                message=(
                    f"{csv_path.name} has malformed '# Upstream-SHA256: {upstream_sha}' "
                    "(expected 64 lowercase hex chars)."
                ),
                context=f"# Upstream-SHA256: {upstream_sha}",
                suggestion="Record the full 64-character lowercase SHA-256 hex digest of the raw upstream snapshot.",
            )
        )

    generator = prov.get("generator", "").strip()
    if generator and not generator.lower().startswith("manual"):
        try:
            tokens = shlex.split(generator)
        except ValueError:
            tokens = generator.split()
        for tok in tokens:
            if tok.startswith(REPO_SCRIPT_PREFIXES) and tok.endswith((".py", ".sh", ".R")):
                script_path = repo_root / tok
                if not script_path.exists():
                    violations.append(
                        ProvenanceViolation(
                            file=rel,
                            line=1,
                            code="series_missing_generator_script",
                            message=(
                                f"{csv_path.name} names generator script '{tok}', "
                                "which does not exist in the repository."
                            ),
                            context=f"# Generator: {generator}",
                            suggestion="Point '# Generator:' to a committed repository script path or mark manual transcription.",
                        )
                    )
                elif enforce_v1:
                    script_text = script_path.read_text(encoding="utf-8", errors="replace")
                    if csv_path.name not in script_text:
                        violations.append(
                            ProvenanceViolation(
                                file=rel,
                                line=1,
                                code="series_missing_generator_script",
                                message=(
                                    f"{csv_path.name} names generator script '{tok}', "
                                    f"but '{tok}' does not reference '{csv_path.name}'."
                                ),
                                context=f"# Generator: {generator}",
                                suggestion="Reference the actual script that writes this CSV, not an unrelated script.",
                            )
                        )

    for key in sorted(set(cited_header_keys)):
        violations.extend(
            _verify_citekey_against_ledger(
                key,
                vol=vol,
                rel=rel,
                lineno=1,
                csv_name=csv_path.name,
                context_str=source,
                repo_root=repo_root,
                bib_ctx=bib_ctx,
            )
        )

    return violations


def build_datapackage_resource(csv_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Construct the canonical Frictionless Data Resource entry for one book CSV."""
    books_dir = (repo_root / "books").resolve()
    rel_books = csv_path.resolve().relative_to(books_dir).as_posix()
    slug = rel_books[:-4].replace("/data/", "-").replace("/", "-").lower()
    series = load_series(csv_path)
    prov = series.provenance

    custom: dict[str, Any] = {
        "data_sha256": series.data_sha256,
        "row_count": len(series.raw_rows),
        "figures": parse_figure_targets(prov.get("figures", "")),
        "locator": prov.get("locator", "").strip(),
        "verified": prov.get("verified", "").strip(),
        "epistemic": prov.get("epistemic", "").strip(),
        "method": prov.get("method", "").strip(),
    }
    for opt_key in ("generator", "upstream_sha256", "computation", "notes"):
        opt_val = prov.get(opt_key, "").strip()
        if opt_val:
            custom[opt_key] = opt_val

    return {
        "profile": "tabular-data-resource",
        "name": slug,
        "path": rel_books,
        "format": "csv",
        "mediatype": "text/csv",
        "encoding": "utf-8",
        "hash": f"sha256:{series.file_sha256}",
        "bytes": series.file_bytes,
        "dialect": {
            "commentChar": "#",
            "header": True,
            "lineTerminator": "\n",
        },
        "licenses": [{"name": prov.get("license", "").strip()}],
        "sources": [
            {
                "title": prov.get("source", "").strip(),
                "path": prov.get("url", "").strip(),
            }
        ],
        "schema": {
            "fields": [
                {
                    "name": col.name,
                    "type": col.type,
                    "description": col.description,
                    "custom": {"unit": col.unit},
                }
                for col in series.column_schema
            ]
        },
        "custom": custom,
    }


def discover_book_csvs(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return all non-generated Tier-2 ``books/vol*/**/data/*.csv`` files in sorted order."""
    books_dir = repo_root / "books"
    if not books_dir.is_dir():
        return []
    return [
        p.resolve()
        for p in sorted(books_dir.rglob("data/*.csv"))
        if _is_book_csv(p, repo_root) and not is_generated(p, repo_root)
    ]


def discover_stray_book_datasets(scan_root: Path, repo_root: Path = REPO_ROOT) -> list[Path]:
    """Find tabular or binary dataset files placed under ``books/vol*`` outside ``data/``."""
    books_dir = (repo_root / "books").resolve()
    if not books_dir.is_dir():
        return []
    if scan_root.is_file():
        candidates = [scan_root.resolve()] if scan_root.suffix.lower() in STRAY_DATASET_SUFFIXES else []
    elif scan_root.is_dir():
        candidates = [
            p.resolve()
            for ext in sorted(STRAY_DATASET_SUFFIXES)
            for p in sorted(scan_root.rglob(f"*{ext}"))
        ]
    else:
        return []

    strays: list[Path] = []
    for p in candidates:
        if is_generated(p, repo_root):
            continue
        try:
            rel_parts = p.relative_to(books_dir).parts
        except ValueError:
            continue
        if not rel_parts or not rel_parts[0].startswith("vol"):
            continue
        # Only CSVs inside a 'data' directory are sanctioned Tier-2 datasets
        if p.suffix.lower() == ".csv" and "data" in rel_parts[:-1]:
            continue
        strays.append(p)
    return strays


def build_datapackage_dict(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Construct the full Frictionless ``books/datapackage.json`` payload."""
    resources = [build_datapackage_resource(p, repo_root=repo_root) for p in discover_book_csvs(repo_root)]
    return {
        "profile": "tabular-data-package",
        "name": "mlsysbook-tier2-series",
        "title": "Machine Learning Systems — Tier-2 Empirical Figure and Table Datasets",
        "description": (
            "Provenance-sealed empirical plot and table CSV datasets across Volumes I-IV "
            "of Machine Learning Systems."
        ),
        "homepage": "https://mlsysbook.ai",
        "version": "1.0.0",
        "resources": resources,
    }


def verify_live_url(url: str, timeout: float = 12.0) -> tuple[bool, str]:
    """Authenticate that *url* resolves over HTTP(S) without 404/410/5xx or DNS failure."""
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except (ImportError, OSError, ssl.SSLError):
        ctx = ssl.create_default_context()

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MLSysBook-DataProvenance/1.0; +https://mlsysbook.ai)",
        "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*;q=0.8",
    }
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                status = getattr(resp, "status", 200)
                if 200 <= status < 400:
                    return True, f"HTTP {status}"
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 429}:
                # Bot-wall / rate-limit on an existing TLS endpoint proves DNS + host route exist;
                # 404 / 410 / 5xx indicate dead or hallucinated paths.
                return True, f"HTTP {exc.code} (host reachable)"
            if method == "HEAD" and exc.code in {400, 405, 501}:
                continue
            return False, f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if method == "HEAD":
                continue
            return False, str(exc)
    return False, "unreachable"


def seal_datapackage(
    repo_root: Path = REPO_ROOT,
    allow_data_change: bool = False,
    reason: str | None = None,
    url_verifier: Callable[[str], tuple[bool, str]] | None = None,
) -> Path:
    """Validate all provenance checks first, enforce the ``data_sha256`` ratchet, and write ``books/datapackage.json``.

    * Default (``allow_data_change=False``, CLI ``--seal``): permits metadata-only updates
      on existing CSVs whose ``data_sha256`` row digest is unchanged. Refuses if any CSV
      was added, removed, or had data rows modified.
    * Row/resource admission (``allow_data_change=True``, CLI ``--seal-data --reason ...``):
      requires a substantive ``reason`` (>= 5 words, no placeholders) and authenticates
      ``# URL:`` live over HTTP(S) for every added or row-modified CSV.
    """
    load_series.cache_clear()
    books_dir = repo_root / "books"
    qmds = [p for p in sorted(books_dir.rglob("*.qmd")) if not is_generated(p, repo_root)] if books_dir.is_dir() else []
    _, data_violations = collect_data_provenance(
        qmds,
        scan_root=books_dir,
        repo_root=repo_root,
        verify_manifest=False,
    )
    _, cell_violations = collect_cell_provenance(qmds, repo_root=repo_root)
    all_violations = data_violations + cell_violations
    if all_violations:
        sample = "; ".join(f"{v.file}:{v.line} [{v.code}] {v.message}" for v in all_violations[:5])
        raise RuntimeError(
            f"Refusing to seal books/datapackage.json with {len(all_violations)} provenance violation(s): {sample}"
        )

    pkg = build_datapackage_dict(repo_root=repo_root)
    out_path = repo_root / DATAPACKAGE_REL_PATH
    new_by_path: dict[str, dict[str, Any]] = {r["path"]: r for r in pkg["resources"]}

    added_paths: list[str] = []
    removed_paths: list[str] = []
    changed_data_paths: list[str] = []

    if out_path.is_file():
        try:
            prev_pkg = json.loads(out_path.read_text(encoding="utf-8"))
            prev_by_path = {
                r["path"]: r
                for r in prev_pkg.get("resources", [])
                if isinstance(r, dict) and isinstance(r.get("path"), str)
            }
        except (OSError, ValueError, TypeError, AttributeError):
            prev_by_path = {}
        added_paths = sorted(set(new_by_path) - set(prev_by_path))
        removed_paths = sorted(set(prev_by_path) - set(new_by_path))
        for rpath, new_res in new_by_path.items():
            old_res = prev_by_path.get(rpath)
            if old_res is not None:
                old_dsha = (old_res.get("custom") or {}).get("data_sha256")
                new_dsha = (new_res.get("custom") or {}).get("data_sha256")
                if old_dsha != new_dsha:
                    changed_data_paths.append(rpath)
    else:
        added_paths = sorted(new_by_path)

    if added_paths or removed_paths or changed_data_paths:
        if not allow_data_change:
            parts: list[str] = []
            if changed_data_paths:
                parts.append(f"modified data_sha256 rows in {changed_data_paths}")
            if added_paths:
                parts.append(f"new unsealed CSVs {added_paths}")
            if removed_paths:
                parts.append(f"removed sealed CSVs {removed_paths}")
            raise RuntimeError(
                "Refusing metadata-only --seal because dataset rows or inventory changed ("
                + "; ".join(parts)
                + "). Pass --seal-data --reason '<>=5-word provenance explanation>' after verifying upstream data."
            )

        clean_reason = (reason or "").strip()
        if (
            _word_count(clean_reason) < 5
            or _has_placeholder(clean_reason) is not None
            or BANNED_REASSURANCE_RE.search(clean_reason) is not None
        ):
            raise RuntimeError(
                "--seal-data requires --reason with at least 5 concrete words explaining the "
                "upstream data source or row change (no placeholders or reassurance words)."
            )

        verifier = url_verifier or verify_live_url
        for rpath in sorted(set(added_paths) | set(changed_data_paths)):
            url = new_by_path[rpath]["sources"][0]["path"]
            ok, detail = verifier(url)
            if not ok:
                raise RuntimeError(
                    f"Refusing --seal-data for '{rpath}': live URL check failed for {url!r} ({detail})."
                )

    out_path.write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def _describe_resource_drift(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    drifted: list[str] = []
    for top_key in ("hash", "bytes", "licenses", "sources", "schema"):
        if actual.get(top_key) != expected.get(top_key):
            drifted.append(top_key)
    act_custom = actual.get("custom") or {}
    exp_custom = expected.get("custom") or {}
    for ckey in sorted(set(act_custom) | set(exp_custom)):
        if act_custom.get(ckey) != exp_custom.get(ckey):
            drifted.append(f"custom.{ckey}")
    return drifted or ["resource"]


def check_datapackage(
    csv_paths: Iterable[Path],
    repo_root: Path = REPO_ROOT,
    check_orphans: bool = True,
) -> list[ProvenanceViolation]:
    """Verify ``books/datapackage.json`` seals every checked book CSV without drift or orphans."""
    book_csvs = [p.resolve() for p in csv_paths if _is_book_csv(p, repo_root)]
    manifest_path = repo_root / DATAPACKAGE_REL_PATH
    manifest_rel = str(DATAPACKAGE_REL_PATH)
    if not book_csvs and not check_orphans:
        return []
    if not (repo_root / "books").is_dir():
        return []

    if not manifest_path.is_file():
        return [
            ProvenanceViolation(
                file=manifest_rel,
                line=1,
                code="series_manifest_missing",
                message="Frictionless manifest books/datapackage.json is missing.",
                suggestion=(
                    "Run 'python3 binder/cli/checks/figure_data_provenance.py --seal-data "
                    "--reason \"<provenance note>\"' after authenticating primary sources."
                ),
            )
        ]

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        resources = payload.get("resources") if isinstance(payload, dict) else None
        if not isinstance(resources, list):
            raise TypeError("top-level 'resources' must be a list")
    except (OSError, ValueError, TypeError) as exc:
        return [
            ProvenanceViolation(
                file=manifest_rel,
                line=1,
                code="series_manifest_missing",
                message=f"books/datapackage.json is not a valid Frictionless Data Package JSON ({exc}).",
                suggestion="Restore valid JSON in books/datapackage.json and re-verify hashes.",
            )
        ]

    by_path: dict[str, dict[str, Any]] = {}
    violations: list[ProvenanceViolation] = []
    books_dir = (repo_root / "books").resolve()

    for entry in resources:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        rpath = entry["path"]
        by_path[rpath] = entry
        if check_orphans:
            disk_target = books_dir / rpath
            if not disk_target.is_file():
                violations.append(
                    ProvenanceViolation(
                        file=manifest_rel,
                        line=1,
                        code="series_manifest_orphan",
                        message=(
                            f"books/datapackage.json lists '{rpath}', which no longer exists under books/."
                        ),
                        context=rpath,
                        suggestion=(
                            "Restore the missing CSV or run "
                            "'python3 binder/cli/checks/figure_data_provenance.py --seal-data --reason \"...\"'."
                        ),
                    )
                )

    for csv_path in book_csvs:
        rel_books = csv_path.relative_to(books_dir).as_posix()
        rel_repo = _rel(csv_path, repo_root)
        actual = by_path.get(rel_books)
        if actual is None:
            violations.append(
                ProvenanceViolation(
                    file=rel_repo,
                    line=1,
                    code="series_manifest_unsealed",
                    message=f"{csv_path.name} is not sealed in books/datapackage.json.",
                    context=rel_books,
                    suggestion=(
                        "Verify primary citations in bibs/manifest.json, then run "
                        "'python3 binder/cli/checks/figure_data_provenance.py --seal-data --reason \"...\"'."
                    ),
                )
            )
            continue
        expected = build_datapackage_resource(csv_path, repo_root=repo_root)
        if actual != expected:
            drifted_fields = ", ".join(_describe_resource_drift(actual, expected))
            data_changed = (actual.get("custom") or {}).get("data_sha256") != expected["custom"]["data_sha256"]
            seal_cmd = (
                "python3 binder/cli/checks/figure_data_provenance.py --seal-data --reason \"<why rows changed>\""
                if data_changed
                else "python3 binder/cli/checks/figure_data_provenance.py --seal"
            )
            violations.append(
                ProvenanceViolation(
                    file=rel_repo,
                    line=1,
                    code="series_manifest_drift",
                    message=(
                        f"{csv_path.name} has drifted from its sealed entry in books/datapackage.json "
                        f"(changed: {drifted_fields})."
                    ),
                    context=f"expected {expected['hash']}, manifest has {actual.get('hash')}",
                    suggestion=f"Confirm the edit against primary sources, then run '{seal_cmd}'.",
                )
            )

    # Cross-verify upstream extraction receipt manifests when present
    epoch_receipt_path = repo_root / EPOCH_RECEIPT_REL_PATH
    if check_orphans and epoch_receipt_path.is_file():
        try:
            receipt = json.loads(epoch_receipt_path.read_text(encoding="utf-8"))
            outputs = receipt.get("outputs") or {}
            checked_rel_set = {_rel(p, repo_root) for p in book_csvs}
            for out_entry in outputs.values():
                if not isinstance(out_entry, dict):
                    continue
                out_rel = out_entry.get("path")
                out_sha = out_entry.get("sha256")
                if not isinstance(out_rel, str) or not isinstance(out_sha, str):
                    continue
                if out_rel not in checked_rel_set:
                    continue
                disk_file = repo_root / out_rel
                if disk_file.is_file():
                    actual_sha = load_series(disk_file).file_sha256
                    if actual_sha != out_sha:
                        violations.append(
                            ProvenanceViolation(
                                file=out_rel,
                                line=1,
                                code="series_receipt_mismatch",
                                message=(
                                    f"{disk_file.name} SHA-256 ({actual_sha[:12]}...) disagrees with "
                                    f"{EPOCH_RECEIPT_REL_PATH} ({out_sha[:12]}...)."
                                ),
                                context=str(EPOCH_RECEIPT_REL_PATH),
                                suggestion=(
                                    "Re-run binder/tools/scripts/extract_epoch_sustainability_data.py "
                                    "instead of hand-editing scripted CSV outputs."
                                ),
                            )
                        )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            violations.append(
                ProvenanceViolation(
                    file=str(EPOCH_RECEIPT_REL_PATH),
                    line=1,
                    code="series_receipt_mismatch",
                    message=f"Receipt manifest {EPOCH_RECEIPT_REL_PATH} could not be parsed ({exc}).",
                    context=str(EPOCH_RECEIPT_REL_PATH),
                    suggestion="Restore valid JSON in the extraction receipt manifest.",
                )
            )

    return violations


def _resolve_series_arg(arg: str, qmd_path: Path, repo_root: Path) -> Path | None:
    p = Path(arg)
    candidates = [
        qmd_path.parent / p,
        repo_root / "books" / p,
        repo_root / p,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return None


def _find_fig_tbl_div_spans(text: str) -> list[tuple[int, int, str]]:
    """Linear-time scan for ``::: {#fig-...}`` / ``::: {#tbl-...}`` character spans."""
    spans: list[tuple[int, int, str]] = []
    open_fence: str | None = None
    open_id: str = ""
    open_start: int = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if open_fence is None:
            m = DIV_OPEN_RE.match(stripped)
            if m:
                open_fence = m.group(1)
                open_id = m.group(2)
                open_start = offset
        else:
            fm = DIV_FENCE_RE.match(stripped)
            if fm and fm.group(1) == open_fence:
                spans.append((open_start, offset + len(line), open_id))
                open_fence = None
        offset += len(line)
    return spans


def _audit_qmd_side_doors(qmd: Path, text: str, repo_root: Path) -> list[ProvenanceViolation]:
    """Block direct CSV/binary file readers and non-literal ``load_series(var)`` bypasses in ``.qmd`` code."""
    if _volume_of(qmd, repo_root) is None:
        return []
    violations: list[ProvenanceViolation] = []
    rel_qmd = _rel(qmd, repo_root)
    for py_match in PY_CELL_RE.finditer(text):
        code = py_match.group(1)
        cell_start = py_match.start(1)
        for rx in (BYPASS_READER_RE, RAW_CSV_OPEN_RE):
            for hit in rx.finditer(code):
                lineno = text.count("\n", 0, cell_start + hit.start()) + 1
                line_str = text.splitlines()[lineno - 1].strip()
                if line_str.startswith("#"):
                    continue
                violations.append(
                    ProvenanceViolation(
                        file=rel_qmd,
                        line=lineno,
                        code="qmd_bypass_series_loader",
                        message=(
                            f"Direct dataset reader {hit.group(0).strip()!r} bypasses Tier-2 provenance; "
                            "load chapter datasets via load_series('volN/<chapter>/data/<file>.csv')."
                        ),
                        context=line_str[:140],
                        suggestion="Place the CSV under books/volN/<chapter>/data/ and read it with load_series(...).",
                    )
                )
        for dhit in DYNAMIC_LOAD_SERIES_RE.finditer(code):
            lineno = text.count("\n", 0, cell_start + dhit.start()) + 1
            line_str = text.splitlines()[lineno - 1].strip()
            if line_str.startswith("#") or "def load_series" in line_str:
                continue
            violations.append(
                ProvenanceViolation(
                    file=rel_qmd,
                    line=lineno,
                    code="qmd_dynamic_load_series",
                    message=(
                        "load_series(...) in book chapters must pass a static string literal CSV path "
                        "so consumer provenance can be verified statically."
                    ),
                    context=line_str[:140],
                    suggestion="Pass a literal path string such as load_series('vol1/01_introduction/data/example.csv').",
                )
            )
    return violations


def collect_data_provenance(
    qmd_files: Iterable[Path],
    scan_root: Path,
    repo_root: Path = REPO_ROOT,
    verify_manifest: bool = True,
) -> tuple[int, list[ProvenanceViolation]]:
    """Validate Tier-2 CSVs under *scan_root*, ``.qmd`` consumers, side-door guards, and ``datapackage.json``."""
    load_series.cache_clear()
    bib_ctx = _load_bib_context(repo_root)
    anchor_cache: dict[str, set[str]] = {}
    checked_csvs: set[Path] = set()
    violations: list[ProvenanceViolation] = []

    qmd_list = [q for q in qmd_files if q.exists() and not is_generated(q, repo_root)]
    qmd_texts: list[tuple[Path, str, str | None, set[str]]] = []

    books_dir = (repo_root / "books").resolve()
    full_books_scan = scan_root.exists() and scan_root.resolve() == books_dir
    for qmd in qmd_list:
        text = qmd.read_text(encoding="utf-8", errors="replace")
        qvol = _volume_of(qmd, repo_root)
        qanchors = set(ANCHOR_ID_RE.findall(text))
        qmd_texts.append((qmd, text, qvol, qanchors))
        if qvol and full_books_scan:
            anchor_cache.setdefault(qvol, set()).update(qanchors)

    # Reject tabular or binary dataset files placed outside chapter data/ directories
    for stray in discover_stray_book_datasets(scan_root, repo_root=repo_root):
        violations.append(
            ProvenanceViolation(
                file=_rel(stray, repo_root),
                line=1,
                code="stray_chapter_dataset",
                message=(
                    f"Dataset file '{stray.name}' sits outside a chapter 'data/' directory, "
                    "which would bypass Tier-2 provenance and manifest sealing."
                ),
                context=_rel(stray, repo_root),
                suggestion="Move empirical datasets to books/volN/<chapter>/data/<name>.csv and seal books/datapackage.json.",
            )
        )

    if scan_root.is_dir():
        csv_candidates = [
            p.resolve()
            for p in sorted(scan_root.rglob("data/*.csv"))
            if not is_generated(p, repo_root)
        ]
    elif scan_root.is_file() and scan_root.suffix == ".csv":
        csv_candidates = [scan_root.resolve()]
    else:
        csv_candidates = []

    for csv_path in csv_candidates:
        checked_csvs.add(csv_path)
        violations.extend(
            check_csv_file(
                csv_path,
                repo_root=repo_root,
                bib_cache=bib_ctx,
                anchor_cache=anchor_cache,
            )
        )

    qmd_count = len(qmd_texts)
    for qmd, text, qmd_vol, qmd_anchors in qmd_texts:
        violations.extend(_audit_qmd_side_doors(qmd, text, repo_root))
        if "load_series(" not in text and "% Source:" not in text:
            continue
        rel_qmd = _rel(qmd, repo_root)
        ref_hits: list[tuple[str, int]] = [
            (m.group(1), m.start()) for m in LOAD_SERIES_RE.finditer(text)
        ]
        ref_hits.extend((m.group(1), m.start()) for m in TIKZ_CSV_SOURCE_RE.finditer(text))
        if not ref_hits:
            continue
        div_spans = _find_fig_tbl_div_spans(text)

        for arg, match_pos in ref_hits:
            lineno = text.count("\n", 0, match_pos) + 1
            src_line = text.splitlines()[lineno - 1].strip()
            resolved = _resolve_series_arg(arg, qmd, repo_root)
            if resolved is None:
                violations.append(
                    ProvenanceViolation(
                        file=rel_qmd,
                        line=lineno,
                        code="missing_figure_series",
                        message=f"load_series({arg!r}) target CSV was not found on disk.",
                        context=src_line,
                        suggestion=(
                            "Place the CSV under books/volN/<chapter>/data/ with a canonical "
                            "Tier-2 provenance header and seal books/datapackage.json."
                        ),
                    )
                )
                continue
            if qmd_vol and not _is_book_csv(resolved, repo_root):
                violations.append(
                    ProvenanceViolation(
                        file=rel_qmd,
                        line=lineno,
                        code="stray_chapter_dataset",
                        message=(
                            f"Book chapter loads '{_rel(resolved, repo_root)}' outside books/vol*/**/data/*.csv."
                        ),
                        context=src_line,
                        suggestion="Store book figure CSVs under books/volN/<chapter>/data/*.csv so books/datapackage.json seals them.",
                    )
                )
            if resolved not in checked_csvs:
                checked_csvs.add(resolved)
                violations.extend(
                    check_csv_file(
                        resolved,
                        repo_root=repo_root,
                        bib_cache=bib_ctx,
                        anchor_cache=anchor_cache,
                    )
                )

            if qmd_vol and _is_book_csv(resolved, repo_root):
                declared_targets = set(parse_figure_targets(load_series(resolved).provenance.get("figures", "")))
                if declared_targets:
                    enclosing_div = next(
                        (div_id for start, end, div_id in div_spans if start <= match_pos <= end),
                        None,
                    )
                    if enclosing_div is not None:
                        expected_target = f"{qmd_vol}:{enclosing_div}"
                        if expected_target not in declared_targets:
                            violations.append(
                                ProvenanceViolation(
                                    file=rel_qmd,
                                    line=lineno,
                                    code="series_unlisted_figure_consumer",
                                    message=(
                                        f"#{enclosing_div} consumes '{resolved.name}', but '{expected_target}' "
                                        f"is not listed in {resolved.name}'s '# Figures:' header."
                                    ),
                                    context=src_line,
                                    suggestion=(
                                        f"Add '{expected_target}' to '# Figures:' in {_rel(resolved, repo_root)} "
                                        "and run 'python3 binder/cli/checks/figure_data_provenance.py --seal'."
                                    ),
                                )
                            )
                    else:
                        qmd_qualified = {f"{qmd_vol}:{aid}" for aid in qmd_anchors}
                        if not (declared_targets & qmd_qualified):
                            violations.append(
                                ProvenanceViolation(
                                    file=rel_qmd,
                                    line=lineno,
                                    code="series_unlisted_figure_consumer",
                                    message=(
                                        f"{rel_qmd} consumes '{resolved.name}', but no figure, table, or "
                                        f"section anchor from {qmd.name} appears in {resolved.name}'s '# Figures:' header."
                                    ),
                                    context=src_line,
                                    suggestion=(
                                        f"Add the consuming '{qmd_vol}:fig-...' (or tbl-... / sec-...) anchor to "
                                        f"'# Figures:' in {_rel(resolved, repo_root)} and run '--seal'."
                                    ),
                                )
                            )

    if verify_manifest:
        manifest_csvs = discover_book_csvs(repo_root) if full_books_scan else checked_csvs
        if manifest_csvs or full_books_scan:
            violations.extend(
                check_datapackage(
                    manifest_csvs,
                    repo_root=repo_root,
                    check_orphans=full_books_scan,
                )
            )

    return len(checked_csvs) + qmd_count, violations


def _is_power_of_ten_sequence(vals: list[int | float]) -> bool:
    for v in vals:
        if v <= 0:
            return False
        lg = math.log10(v)
        if abs(lg - round(lg)) > 1e-9:
            return False
    return True


def _find_inline_empirical_float_array(code: str) -> tuple[str, list[int | float]] | None:
    """Detect >=5-element hardcoded float literal sequences in a ``measured`` figure cell."""
    clean_code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("#|"))
    try:
        tree = ast.parse(clean_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        tname = ast.unparse(node.targets[0])
        if LAYOUT_ARRAY_NAME_RE.search(tname):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, (ast.List, ast.Tuple)) and len(sub.elts) >= 5:
                vals = [
                    e.value
                    for e in sub.elts
                    if isinstance(e, ast.Constant)
                    and isinstance(e.value, (int, float))
                    and not isinstance(e.value, bool)
                ]
                if (
                    len(vals) == len(sub.elts)
                    and not _is_power_of_ten_sequence(vals)
                    and any(isinstance(v, float) and (v * 2) != int(v * 2) for v in vals)
                ):
                    return tname, vals[:6]
    return None


def _csv_backed_figure_targets(repo_root: Path) -> set[str]:
    manifest_path = repo_root / DATAPACKAGE_REL_PATH
    targets: set[str] = set()
    if manifest_path.is_file():
        try:
            pkg = json.loads(manifest_path.read_text(encoding="utf-8"))
            for res in pkg.get("resources", []):
                if isinstance(res, dict):
                    figs = (res.get("custom") or {}).get("figures") or []
                    targets.update(f for f in figs if isinstance(f, str))
            return targets
        except (OSError, ValueError, TypeError, AttributeError):
            targets.clear()
    for csv_path in discover_book_csvs(repo_root):
        targets.update(parse_figure_targets(load_series(csv_path).provenance.get("figures", "")))
    return targets


def collect_cell_provenance(
    qmd_files: Iterable[Path],
    repo_root: Path = REPO_ROOT,
) -> tuple[int, list[ProvenanceViolation]]:
    """Audit matplotlib ``#fig-*`` cells for substantive ``Provenance:`` blocks and receipt-backed measurements."""
    checked_figures = 0
    violations: list[ProvenanceViolation] = []
    csv_backed_figs: set[str] | None = None

    for qmd in qmd_files:
        if not qmd.exists() or is_generated(qmd, repo_root):
            continue
        text = qmd.read_text(encoding="utf-8", errors="replace")
        rel_qmd = _rel(qmd, repo_root)
        vol = _volume_of(qmd, repo_root)
        chap_has_mlsysim = bool(MLSYSIM_RECEIPT_RE.search(text))

        for div_match in FIG_DIV_RE.finditer(text):
            fig_id = div_match.group(2)
            div_body = div_match.group(3)
            py_match = PY_CELL_RE.search(div_body)
            if py_match is None:
                continue
            checked_figures += 1
            code = py_match.group(1)
            abs_offset = div_match.start(3) + py_match.start(1)
            code_lines = code.splitlines()
            prov_idx = next((i for i, ln in enumerate(code_lines) if PROVENANCE_LINE_RE.match(ln)), None)
            if prov_idx is None:
                lineno = text.count("\n", 0, abs_offset) + 1
                violations.append(
                    ProvenanceViolation(
                        file=rel_qmd,
                        line=lineno,
                        code="figure_missing_cell_provenance",
                        message=(
                            f"#{fig_id} Python figure cell has no 'Provenance:' block in its header "
                            "documenting data/registry inputs, computation recipe, and epistemic boundary."
                        ),
                        context=f"::: {{#{fig_id} ...}}",
                        suggestion=(
                            "Add '# │ Provenance:' lines to the figure cell box header naming the CSV or "
                            "mlsysim inputs, derivation equation/fit, and epistemic status (measured/modeled/constructed/assumed)."
                        ),
                    )
                )
                continue

            prov_lineno = text.count("\n", 0, abs_offset) + 1 + prov_idx
            prov_lines: list[str] = []
            for ln in code_lines[prov_idx:]:
                if ln.strip().startswith("#"):
                    prov_lines.append(ln.strip())
                else:
                    break
            prov_blob = " ".join(prov_lines)

            bm = BANNED_REASSURANCE_RE.search(prov_blob)
            ph = _has_placeholder(prov_blob)
            if bm or ph:
                bad_tok = bm.group(0) if bm else ph
                violations.append(
                    ProvenanceViolation(
                        file=rel_qmd,
                        line=prov_lineno,
                        code="figure_stub_cell_provenance",
                        message=(
                            f"#{fig_id} 'Provenance:' block contains disallowed placeholder or "
                            f"reassurance token {bad_tok!r}."
                        ),
                        context=prov_lines[0][:140],
                        suggestion="Replace placeholder or reassurance words with the concrete input series, equation, and boundary.",
                    )
                )
                continue

            # Require >= 6 words (excluding 'Provenance' label) and an explicit epistemic token or CSV path
            has_csv_ref = ".csv" in prov_blob
            if _word_count(prov_blob) < 6 or (not CELL_EPISTEMIC_RE.search(prov_blob) and not has_csv_ref):
                violations.append(
                    ProvenanceViolation(
                        file=rel_qmd,
                        line=prov_lineno,
                        code="figure_stub_cell_provenance",
                        message=(
                            f"#{fig_id} 'Provenance:' block is a stub; it must contain >= 6 words and "
                            "explicitly state its epistemic status (measured/modeled/assumed/constructed) or CSV source."
                        ),
                        context=prov_lines[0][:140],
                        suggestion="Document concrete data/mlsysim inputs, equation/fit, and epistemic status in >= 6 words.",
                    )
                )
                continue

            # Anti-hardcoding guard on figure cells claiming empirical measurement
            claims_measured = bool(POS_MEASURED_RE.search(prov_blob)) and "inputs (constructed)" not in prov_blob.lower()
            if claims_measured and vol is not None:
                if csv_backed_figs is None:
                    csv_backed_figs = _csv_backed_figure_targets(repo_root)
                qual_fig = f"{vol}:{fig_id}"
                backed_by_csv = qual_fig in csv_backed_figs or "load_series(" in code
                backed_by_mlsysim = bool(MLSYSIM_RECEIPT_RE.search(code)) or chap_has_mlsysim
                if not backed_by_csv and not backed_by_mlsysim:
                    violations.append(
                        ProvenanceViolation(
                            file=rel_qmd,
                            line=prov_lineno,
                            code="figure_unbacked_measured_cell",
                            message=(
                                f"#{fig_id} claims 'measured' provenance, but neither a sealed Tier-2 CSV "
                                "nor an mlsysim registry/scenario source backs this figure."
                            ),
                            context=prov_lines[0][:140],
                            suggestion=(
                                "Move empirical measurements into books/volN/<chapter>/data/*.csv via load_series(...) "
                                "or mlsysim, or label illustrative/analytical curves as 'modeled' or 'constructed'."
                            ),
                        )
                    )
                    continue

                inline_arr = _find_inline_empirical_float_array(code)
                if inline_arr is not None:
                    arr_name, sample_vals = inline_arr
                    violations.append(
                        ProvenanceViolation(
                            file=rel_qmd,
                            line=prov_lineno,
                            code="figure_unbacked_measured_cell",
                            message=(
                                f"#{fig_id} claims 'measured' provenance while embedding hardcoded float "
                                f"literal array '{arr_name} = {sample_vals}...' directly in Python code."
                            ),
                            context=f"{arr_name} = {sample_vals}",
                            suggestion=(
                                "Do not hardcode empirical data arrays inside figure Python cells; "
                                "store measured rows in a provenance-sealed chapter data/*.csv file."
                            ),
                        )
                    )

    return checked_figures, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Optional .qmd, .csv, or directory paths to scan")
    parser.add_argument(
        "--scope",
        choices=("data-provenance", "cell-provenance"),
        default="data-provenance",
        help="Which figure provenance layer to check",
    )
    parser.add_argument(
        "--seal",
        action="store_true",
        help="Re-seal metadata changes in books/datapackage.json (refuses if any CSV data_sha256 or file set changed)",
    )
    parser.add_argument(
        "--seal-data",
        action="store_true",
        help="Admit new CSVs or modified data_sha256 rows into books/datapackage.json after live URL + citation checks",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Required >=5-word provenance explanation when --seal-data admits new or modified CSV data rows",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args(argv)

    if args.seal and args.seal_data:
        print("Pass either --seal (metadata-only) or --seal-data --reason '...', not both.", file=sys.stderr)
        return 2

    if args.seal or args.seal_data:
        try:
            out_manifest = seal_datapackage(
                repo_root=REPO_ROOT,
                allow_data_change=args.seal_data,
                reason=args.reason,
            )
        except RuntimeError as exc:
            if args.json:
                print(json.dumps([{"code": "seal_rejected", "message": str(exc)}], indent=2))
            else:
                print(f"ERROR [seal_rejected] {exc}", file=sys.stderr)
            return 1
        if not args.json:
            mode = "data+metadata" if args.seal_data else "metadata-only"
            print(f"Sealed Frictionless Data Package ({mode}): {_rel(out_manifest, REPO_ROOT)}")

    if args.paths:
        targets = [
            (REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
            for p in args.paths
        ]
        qmds: list[Path] = []
        explicit_csvs: list[Path] = []
        for t in targets:
            if t.is_dir():
                qmds.extend(p for p in sorted(t.rglob("*.qmd")) if not is_generated(p, REPO_ROOT))
            elif t.suffix == ".qmd":
                qmds.append(t)
            elif t.suffix == ".csv":
                explicit_csvs.append(t)
        scan_root = targets[0] if len(targets) == 1 else REPO_ROOT / "books"
    else:
        scan_root = REPO_ROOT / "books"
        qmds = [p for p in sorted(scan_root.rglob("*.qmd")) if not is_generated(p, REPO_ROOT)]
        explicit_csvs = []

    if args.scope == "data-provenance":
        checked, violations = collect_data_provenance(qmds, scan_root=scan_root, repo_root=REPO_ROOT)
        if len(explicit_csvs) > 1:
            bib_ctx = _load_bib_context(REPO_ROOT)
            anchor_cache: dict[str, set[str]] = {}
            for extra_csv in explicit_csvs:
                checked += 1
                violations.extend(
                    check_csv_file(
                        extra_csv,
                        repo_root=REPO_ROOT,
                        bib_cache=bib_ctx,
                        anchor_cache=anchor_cache,
                    )
                )
            violations.extend(check_datapackage(explicit_csvs, repo_root=REPO_ROOT, check_orphans=False))
    else:
        checked, violations = collect_cell_provenance(qmds, repo_root=REPO_ROOT)

    if args.json:
        print(json.dumps([v.to_issue_kwargs() for v in violations], indent=2))
        return 1 if violations else 0

    for v in violations:
        print(f"{v.file}:{v.line} [{v.code}] {v.message}")
        if v.context:
            print(f"  source: {v.context}")
        if v.suggestion:
            print(f"  fix: {v.suggestion}")
    print(f"\nfigure {args.scope}: {checked} item(s) checked, {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
