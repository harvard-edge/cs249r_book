#!/usr/bin/env python3
"""Provenance and computation-lineage checks for figure datasets and figure cells.

This powers::

    ./binder/binder check figures --scope data-provenance
    ./binder/binder check figures --scope cell-provenance

Why this exists:
  Tier-1 constants in ``mlsysim/`` are gated by ``audit_provenance.py`` and
  case-study callouts are gated by ``case_study_provenance.py``, while Tier-2
  empirical plot series (``books/volN/<chapter>/data/*.csv`` loaded via
  ``load_series(...)``) previously relied on convention alone. This module makes
  figure dataset and derivation provenance enforceable by the build.

Error codes (``data-provenance``, ``default=True``):

  ``missing_figure_series``
      Bad::
          rows = viz.load_series("vol2/15_sustainable_ai/data/missing.csv").entries
      Good::
          rows = viz.load_series("vol2/15_sustainable_ai/data/notable_model_training_energy.csv").entries

  ``series_missing_source``
      Bad (CSV starts directly with column names)::
          year,energy_mwh
          2020,1287
      Good::
          # Source: Patterson et al. (2021), Table 4 (@patterson2021carbon)
          # Verified: 2026-10-20
          # Epistemic: measured (metered system power * PUE)
          # Computation: transcribed from Table 4; energy_mwh = chips * (measured_w / 1e6) * (days * 24) * pue
          # Notes: Final-run figures across five NLP models.
          year,energy_mwh

  ``series_missing_verified`` / ``series_invalid_verified``
      Bad::
          # Verified: Oct 2026
      Good::
          # Verified: 2026-10-12

  ``series_missing_computation_or_notes``
      Bad::
          # Source: Epoch AI (2026)
          # Verified: 2026-10-12
          model,date,training_power_draw_w
      Good::
          # Source: Epoch AI (2026) (@epochai2026notable)
          # Verified: 2026-10-12
          # Computation: filter Publication date >= 2016-01-01 and empty Base model; E_MWh = power_w * hours / 1e6
          # Notes: standalone pretraining runs (N=122, frontier n=23).
          model,date,training_power_draw_w

  ``series_invalid_epistemic``
      Bad::
          # Epistemic: empirical
      Good::
          # Epistemic: modeled (peak cluster power capacity * wall-clock hours)

  ``series_invalid_sha256``
      Bad::
          # Upstream-SHA256: bacde4f9
      Good::
          # Upstream-SHA256: bacde4f9af388703ceebd01cf71a14854f93b8995e425141b016e48f21d4e7df

  ``series_missing_generator_script``
      Bad::
          # Generator: python3 binder/tools/scripts/nonexistent.py
      Good::
          # Generator: python3 binder/tools/scripts/extract_epoch_sustainability_data.py --download

  ``series_unresolved_citekey``
      Bad::
          # Source: Epoch AI (@missing2099key)
      Good::
          # Source: Epoch AI (@epochai2026notable)

Opt-in cell audit (``cell-provenance``, ``default=False``):

  ``figure_missing_cell_provenance``
      Flags matplotlib ``#fig-*`` cells whose box header lacks a ``Provenance:``
      entry summarizing data/registry inputs, computation recipe, and epistemic
      boundary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from cli.checks.generated_paths import is_generated
except ImportError:  # pragma: no cover - standalone or package-qualified import
    from binder.cli.checks.generated_paths import is_generated

REPO_ROOT = Path(__file__).resolve().parents[3]
MLSYSIM_DIR = REPO_ROOT / "mlsysim"
if str(MLSYSIM_DIR) not in sys.path:
    sys.path.insert(0, str(MLSYSIM_DIR))

from mlsysim.core.series_loader import LocalSeries  # noqa: E402

VALID_EPISTEMIC = frozenset({"measured", "modeled", "assumed", "constructed"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOAD_SERIES_RE = re.compile(r"""\bload_series\(\s*["']([^"']+)["']""")
FIG_DIV_RE = re.compile(r"(?m)^(:{3,})\s*\{#(fig-[^\s}]+)[^\n]*\n(.*?)^\1[ \t]*$", re.S)
PY_CELL_RE = re.compile(r"```\{python\}\n(.*?)^```[ \t]*$", re.M | re.S)
PROVENANCE_LINE_RE = re.compile(r"^\s*#.*?\bProvenance\s*:", re.M)
CITEKEY_RE = re.compile(r"(?<![A-Za-z0-9._+-])@([A-Za-z][A-Za-z0-9_:+.-]*[A-Za-z0-9])")
NOT_A_CITEKEY_RE = re.compile(r"^(fig|sec|tbl|eq|lst|thm|def|dfn|pri|nbk|exm|exmp|cs|ws|chk|psp|lhs)-", re.I)
REPO_SCRIPT_PREFIXES = ("binder/", "shared/", "mlsysim/", "books/", "scripts/")


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


def _bib_keys_for_volume(vol_name: str | None, repo_root: Path) -> set[str]:
    candidates: list[Path] = []
    if vol_name:
        candidates.append(repo_root / "books" / f"references-{vol_name}.bib")
    candidates.extend([repo_root / "books" / "references.bib", repo_root / "bibs" / "references.bib"])
    for cand in candidates:
        if cand.exists():
            return set(re.findall(r"^@\w+\{([^,]+),", cand.read_text(encoding="utf-8", errors="replace"), re.M))
    return set()


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


def check_csv_file(
    csv_path: Path,
    repo_root: Path = REPO_ROOT,
    bib_cache: dict[str | None, set[str]] | None = None,
) -> list[ProvenanceViolation]:
    """Validate structured provenance metadata on one Tier-2 ``data/*.csv`` file."""
    rel = _rel(csv_path, repo_root)
    series = LocalSeries(csv_path)
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
                suggestion="Add '# Source: Author (Year), Title (@bibkey)' at line 1 of the CSV.",
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
                suggestion="Add '# Verified: YYYY-MM-DD' recording when the CSV was extracted or checked against its primary source.",
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
    if epistemic:
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
                    suggestion="Start '# Epistemic:' with measured, modeled, assumed, or constructed, followed by boundary details.",
                )
            )

    upstream_sha = prov.get("upstream_sha256", "").strip()
    if upstream_sha and not SHA256_RE.match(upstream_sha):
        violations.append(
            ProvenanceViolation(
                file=rel,
                line=1,
                code="series_invalid_sha256",
                message=f"{csv_path.name} has malformed '# Upstream-SHA256: {upstream_sha}' (expected 64 lowercase hex chars).",
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
                            message=f"{csv_path.name} names generator script '{tok}', which does not exist in the repository.",
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
        cited = _extract_citekeys(" ".join(v for k, v in prov.items() if k in {"source", "computation", "notes"}))
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


def collect_data_provenance(
    qmd_files: Iterable[Path],
    scan_root: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[int, list[ProvenanceViolation]]:
    """Validate all Tier-2 CSVs under *scan_root* and all ``load_series(...)`` calls in *qmd_files*."""
    bib_cache: dict[str | None, set[str]] = {}
    checked_csvs: set[Path] = set()
    violations: list[ProvenanceViolation] = []

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
        violations.extend(check_csv_file(csv_path, repo_root=repo_root, bib_cache=bib_cache))

    qmd_count = 0
    for qmd in qmd_files:
        if not qmd.exists() or is_generated(qmd, repo_root):
            continue
        qmd_count += 1
        text = qmd.read_text(encoding="utf-8", errors="replace")
        rel_qmd = _rel(qmd, repo_root)
        for match in LOAD_SERIES_RE.finditer(text):
            arg = match.group(1)
            lineno = text.count("\n", 0, match.start()) + 1
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
                        suggestion="Place the CSV under books/volN/<chapter>/data/ with a '# Source:' / '# Verified:' / '# Computation:' header.",
                    )
                )
                continue
            if resolved not in checked_csvs:
                checked_csvs.add(resolved)
                violations.extend(check_csv_file(resolved, repo_root=repo_root, bib_cache=bib_cache))

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
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args(argv)

    if args.paths:
        targets = [(REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in args.paths]
        qmds: list[Path] = []
        for t in targets:
            if t.is_dir():
                qmds.extend(p for p in sorted(t.rglob("*.qmd")) if not is_generated(p, REPO_ROOT))
            elif t.suffix == ".qmd":
                qmds.append(t)
        scan_root = targets[0] if len(targets) == 1 else REPO_ROOT / "books"
    else:
        scan_root = REPO_ROOT / "books"
        qmds = [p for p in sorted(scan_root.rglob("*.qmd")) if not is_generated(p, REPO_ROOT)]

    if args.scope == "data-provenance":
        checked, violations = collect_data_provenance(qmds, scan_root=scan_root, repo_root=REPO_ROOT)
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
