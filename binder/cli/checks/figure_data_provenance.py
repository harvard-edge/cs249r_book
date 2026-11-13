#!/usr/bin/env python3
"""Provenance, schema, and Frictionless Data Package checks for figure datasets.

This powers::

    ./binder/binder check figures --scope data-provenance
    ./binder/binder check figures --scope data-provenance --fix
    ./binder/binder check figures --scope cell-provenance
    python3 binder/cli/checks/figure_data_provenance.py --seal

Why this exists:
  Tier-1 constants in ``mlsysim/`` are gated by ``audit_provenance.py`` and
  case-study callouts are gated by ``case_study_provenance.py``. Tier-2
  empirical plot and table series (``books/volN/<chapter>/data/*.csv`` loaded via
  ``load_series(...)``) follow the Tier-2 Data Provenance Standard v1: an inline
  ``# Key: Value`` header with typed per-column units plus a cryptographically
  sealed Frictionless Data Package manifest at ``books/datapackage.json``.

Canonical CSV header template (``books/volN/<chapter>/data/*.csv``)::

    # Figures: vol1:fig-ai-timeline
    # Source: Primary citation or dataset title (@bibkey when in references.bib)
    # Locator: Table 4, p. 8 (or API endpoint / archive snapshot identifier)
    # URL: https://...
    # License: CC-BY-4.0
    # Verified: YYYY-MM-DD
    # Epistemic: measured | modeled | assumed | constructed (<accounting boundary>)
    # Method: transcribed | digitized | scripted | derived [(<mechanism detail>)]
    # Generator: python3 binder/tools/scripts/<script>.py ...     [optional]
    # Upstream-SHA256: <64-hex raw upstream snapshot digest>      [optional]
    # Computation: Filtering criteria, exclusions, and formulas   [optional if Notes present]
    # Columns:
    #   <col_name>: <integer|number|string|boolean|date|datetime> [<unit>] — <description>
    # Notes: Caveats and accounting boundaries                    [optional if Computation present]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from cli.checks.generated_paths import is_generated
except ImportError:  # pragma: no cover - standalone or package-qualified import
    from binder.cli.checks.generated_paths import is_generated

REPO_ROOT = Path(__file__).resolve().parents[3]
MLSYSIM_DIR = REPO_ROOT / "mlsysim"
if str(MLSYSIM_DIR) not in sys.path:
    sys.path.insert(0, str(MLSYSIM_DIR))

from mlsysim.core.series_loader import ColumnSpec, LocalSeries, load_series  # noqa: E402

VALID_EPISTEMIC = frozenset({"measured", "modeled", "assumed", "constructed"})
VALID_METHODS = frozenset({"transcribed", "digitized", "scripted", "derived"})
VALID_COLUMN_TYPES = frozenset({"integer", "number", "string", "boolean", "date", "datetime"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.I)
INT_RE = re.compile(r"^-?\d+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FIGURE_TARGET_RE = re.compile(r"^(vol[1-4]):((?:fig|tbl|sec)-[A-Za-z0-9_-]+)$")
AUXILIARY_TARGET_RE = re.compile(r"^auxiliary\s*\(\s*([^()\s]+)\s*\)$")
ANCHOR_ID_RE = re.compile(r"\{#((?:fig|tbl|sec)-[A-Za-z0-9_-]+)")
LOAD_SERIES_RE = re.compile(r"""\bload_series\(\s*["']([^"']+)["']""")
TIKZ_CSV_SOURCE_RE = re.compile(r"(?m)^\s*%\s*Source:\s*(books/vol[1-4]/[^\s]+\.csv)\b")
DIV_OPEN_RE = re.compile(r"^(:{3,})\s*\{#((?:fig|tbl)-[^\s}]+)")
DIV_FENCE_RE = re.compile(r"^(:{3,})[ \t]*$")
FIG_DIV_RE = re.compile(r"(?m)^(:{3,})\s*\{#(fig-[^\s}]+)[^\n]*\n(.*?)^\1[ \t]*$", re.S)
PY_CELL_RE = re.compile(r"```\{python\}\n(.*?)^```[ \t]*$", re.M | re.S)
PROVENANCE_LINE_RE = re.compile(r"^\s*#.*?\bProvenance\s*:", re.M)
CITEKEY_RE = re.compile(r"(?<![A-Za-z0-9._+-])@([A-Za-z][A-Za-z0-9_:+.-]*[A-Za-z0-9])")
NOT_A_CITEKEY_RE = re.compile(
    r"^(fig|sec|tbl|eq|lst|thm|def|dfn|pri|nbk|exm|exmp|cs|ws|chk|psp|lhs)-", re.I
)
REPO_SCRIPT_PREFIXES = ("binder/", "shared/", "mlsysim/", "books/", "slides/", "scripts/")
DATAPACKAGE_REL_PATH = Path("books/datapackage.json")


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


def _bib_keys_for_volume(vol_name: str | None, repo_root: Path) -> set[str]:
    candidates: list[Path] = []
    if vol_name:
        candidates.append(repo_root / "books" / f"references-{vol_name}.bib")
    candidates.extend([repo_root / "books" / "references.bib", repo_root / "bibs" / "references.bib"])
    for cand in candidates:
        if cand.exists():
            return set(
                re.findall(
                    r"^@\w+\{([^,]+),",
                    cand.read_text(encoding="utf-8", errors="replace"),
                    re.M,
                )
            )
    return set()


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


def check_csv_file(
    csv_path: Path,
    repo_root: Path = REPO_ROOT,
    bib_cache: dict[str | None, set[str]] | None = None,
    anchor_cache: dict[str, set[str]] | None = None,
    strict_v1: bool | None = None,
) -> list[ProvenanceViolation]:
    """Validate structured provenance metadata and column schema on one ``data/*.csv`` file."""
    rel = _rel(csv_path, repo_root)
    enforce_v1 = _is_book_csv(csv_path, repo_root) if strict_v1 is None else strict_v1
    series = load_series(csv_path)
    prov = series.provenance
    comments = series.provenance_comments
    first_comment = comments[0] if comments else ""
    violations: list[ProvenanceViolation] = []

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
            dt.date.fromisoformat(verified)
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

    if enforce_v1:
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
                    if not (repo_root / aux_rel).exists():
                        violations.append(
                            ProvenanceViolation(
                                file=rel,
                                line=1,
                                code="series_unresolved_figure_target",
                                message=(
                                    f"{csv_path.name} '# Figures:' auxiliary consumer '{aux_rel}' "
                                    "does not exist in the repository."
                                ),
                                context=f"# Figures: {figures_raw}",
                                suggestion="Point 'auxiliary (<repo_path>)' to an existing committed file path.",
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
                # Only fail when volume qmd files exist on disk (supports isolated unit tests)
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

        violations.extend(_validate_columns_block(series, rel, csv_path.name, first_comment))

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

    vol = _volume_of(csv_path, repo_root)
    cache = bib_cache if bib_cache is not None else {}
    if vol not in cache:
        cache[vol] = _bib_keys_for_volume(vol, repo_root)
    known_keys = cache[vol]
    if known_keys:
        cited = _extract_citekeys(
            " ".join(v for k, v in prov.items() if k in {"source", "locator", "computation", "notes"})
        )
        for key in sorted(set(cited)):
            if key not in known_keys:
                violations.append(
                    ProvenanceViolation(
                        file=rel,
                        line=1,
                        code="series_unresolved_citekey",
                        message=f"{csv_path.name} header cites @{key}, which is absent from the volume bibliography.",
                        context=source,
                        suggestion="Use the exact citekey from bibs/references.bib or remove the '@' prefix if not a BibTeX key.",
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


def seal_datapackage(repo_root: Path = REPO_ROOT) -> Path:
    """Write ``books/datapackage.json`` deterministically from all Tier-2 book CSVs."""
    load_series.cache_clear()
    pkg = build_datapackage_dict(repo_root=repo_root)
    out_path = repo_root / DATAPACKAGE_REL_PATH
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
                    "Run 'python3 binder/cli/checks/figure_data_provenance.py --seal' "
                    "(or './binder/binder check figures --scope data-provenance --fix') to generate it."
                ),
            )
        ]

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        resources = payload.get("resources")
        if not isinstance(resources, list):
            raise ValueError("top-level 'resources' must be a list")
    except Exception as exc:
        return [
            ProvenanceViolation(
                file=manifest_rel,
                line=1,
                code="series_manifest_missing",
                message=f"books/datapackage.json is not a valid Frictionless Data Package JSON ({exc}).",
                suggestion="Regenerate books/datapackage.json with 'python3 binder/cli/checks/figure_data_provenance.py --seal'.",
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
                            "Run 'python3 binder/cli/checks/figure_data_provenance.py --seal' "
                            "to remove stale entries from books/datapackage.json."
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
                        "Run 'python3 binder/cli/checks/figure_data_provenance.py --seal' "
                        "(or './binder/binder check figures --scope data-provenance --fix') to seal this dataset."
                    ),
                )
            )
            continue
        expected = build_datapackage_resource(csv_path, repo_root=repo_root)
        if actual != expected:
            drifted_fields = ", ".join(_describe_resource_drift(actual, expected))
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
                    suggestion=(
                        "Verify the CSV change is intentional, then run "
                        "'python3 binder/cli/checks/figure_data_provenance.py --seal' "
                        "(or './binder/binder check figures --scope data-provenance --fix')."
                    ),
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


def collect_data_provenance(
    qmd_files: Iterable[Path],
    scan_root: Path,
    repo_root: Path = REPO_ROOT,
    verify_manifest: bool = True,
) -> tuple[int, list[ProvenanceViolation]]:
    """Validate Tier-2 CSVs under *scan_root*, ``load_series(...)`` consumers, and ``datapackage.json``."""
    load_series.cache_clear()
    bib_cache: dict[str | None, set[str]] = {}
    anchor_cache: dict[str, set[str]] = {}
    checked_csvs: set[Path] = set()
    violations: list[ProvenanceViolation] = []

    qmd_list = [q for q in qmd_files if q.exists() and not is_generated(q, repo_root)]
    qmd_texts: list[tuple[Path, str, str | None, set[str]]] = []
    covered_vols: set[str] = set()

    # Pre-populate volume anchor sets when scanning full volume/books trees so .qmd files are read once.
    books_dir = (repo_root / "books").resolve()
    full_books_scan = scan_root.exists() and scan_root.resolve() == books_dir
    for qmd in qmd_list:
        text = qmd.read_text(encoding="utf-8", errors="replace")
        qvol = _volume_of(qmd, repo_root)
        qanchors = set(ANCHOR_ID_RE.findall(text))
        qmd_texts.append((qmd, text, qvol, qanchors))
        if qvol and full_books_scan:
            covered_vols.add(qvol)
            anchor_cache.setdefault(qvol, set()).update(qanchors)

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
                bib_cache=bib_cache,
                anchor_cache=anchor_cache,
            )
        )

    qmd_count = len(qmd_texts)
    for qmd, text, qmd_vol, qmd_anchors in qmd_texts:
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
            if resolved not in checked_csvs:
                checked_csvs.add(resolved)
                violations.extend(
                    check_csv_file(
                        resolved,
                        repo_root=repo_root,
                        bib_cache=bib_cache,
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
                                        "and reseal books/datapackage.json."
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
                                        f"'# Figures:' in {_rel(resolved, repo_root)} and reseal books/datapackage.json."
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


def collect_cell_provenance(
    qmd_files: Iterable[Path],
    repo_root: Path = REPO_ROOT,
) -> tuple[int, list[ProvenanceViolation]]:
    """Audit matplotlib ``#fig-*`` cells for an explicit ``Provenance:`` header block."""
    checked_figures = 0
    violations: list[ProvenanceViolation] = []

    for qmd in qmd_files:
        if not qmd.exists() or is_generated(qmd, repo_root):
            continue
        text = qmd.read_text(encoding="utf-8", errors="replace")
        rel_qmd = _rel(qmd, repo_root)
        for div_match in FIG_DIV_RE.finditer(text):
            fig_id = div_match.group(2)
            div_body = div_match.group(3)
            py_match = PY_CELL_RE.search(div_body)
            if py_match is None:
                continue
            checked_figures += 1
            code = py_match.group(1)
            if not PROVENANCE_LINE_RE.search(code):
                abs_offset = div_match.start(3) + py_match.start()
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
        help="Regenerate and seal books/datapackage.json before running validation",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args(argv)

    if args.seal:
        out_manifest = seal_datapackage(repo_root=REPO_ROOT)
        if not args.json:
            print(f"Sealed Frictionless Data Package: {_rel(out_manifest, REPO_ROOT)}")

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
            bib_cache: dict[str | None, set[str]] = {}
            anchor_cache: dict[str, set[str]] = {}
            for extra_csv in explicit_csvs:
                checked += 1
                violations.extend(
                    check_csv_file(
                        extra_csv,
                        repo_root=REPO_ROOT,
                        bib_cache=bib_cache,
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
