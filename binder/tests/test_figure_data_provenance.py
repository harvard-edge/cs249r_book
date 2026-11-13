from __future__ import annotations

import json
from pathlib import Path

from mlsysim.core.series_loader import LocalSeries, parse_column_schema, parse_series_provenance

from binder.cli.checks import figure_data_provenance as fdp


VALID_V1_CSV = """\
# Figures: vol1:fig-demo-curve
# Source: Demo et al. (2025), Example Benchmark (@demo2025paper)
# Locator: Table 2, p. 4
# URL: https://example.org/demo-benchmark
# License: CC-BY-4.0
# Verified: 2026-10-15
# Epistemic: measured (board-level power meter integrated over run duration)
# Method: transcribed (manual extraction of Table 2 rows)
# Computation: Direct transcription of published latency and energy columns.
# Columns:
#   step: integer [step] — Benchmark iteration index
#   latency_ms: number [ms] — Median request latency
#   model_tag: string [identifier] — Model checkpoint identifier
#   converged: boolean [flag] — Whether run reached target accuracy
#   run_date: date [ISO-8601 date] — Benchmark execution date
# Notes: Single-node baseline measurements.
step,latency_ms,model_tag,converged,run_date
1,12.5,demo-small,True,2026-01-15
2,24.0,demo-large,False,2026-02-20
"""


def _make_mini_repo(tmp_path: Path, csv_content: str = VALID_V1_CSV) -> tuple[Path, Path, Path]:
    (tmp_path / "books" / "vol1" / "01_intro" / "data").mkdir(parents=True)
    (tmp_path / "bibs").mkdir(parents=True)
    (tmp_path / "bibs" / "references.bib").write_text(
        "@inproceedings{demo2025paper,\n  title={Example},\n  author={Demo, A.},\n  year={2025}\n}\n",
        encoding="utf-8",
    )
    qmd = tmp_path / "books" / "vol1" / "01_intro" / "01_intro.qmd"
    qmd.write_text(
        '# Intro {#sec-intro}\n\n'
        '::: {#fig-demo-curve fig-env="figure" fig-cap="**Demo**: Text." fig-alt="Line chart."}\n'
        '```{python}\n'
        '# │ Provenance: Measured rows from vol1/01_intro/data/demo.csv.\n'
        'rows = load_series("vol1/01_intro/data/demo.csv").entries\n'
        '```\n'
        ':::\n',
        encoding="utf-8",
    )
    csv_path = tmp_path / "books" / "vol1" / "01_intro" / "data" / "demo.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return tmp_path, qmd, csv_path


def test_series_loader_parses_v1_schema_and_hashes(tmp_path: Path):
    _, _, csv_path = _make_mini_repo(tmp_path)
    series = LocalSeries(csv_path)

    assert series.provenance["figures"] == "vol1:fig-demo-curve"
    assert series.provenance["locator"] == "Table 2, p. 4"
    assert series.provenance["license"] == "CC-BY-4.0"
    assert series.provenance["method"].startswith("transcribed")
    assert len(series.column_schema) == 5
    assert series.malformed_column_lines == ()
    assert [c.name for c in series.column_schema] == [
        "step",
        "latency_ms",
        "model_tag",
        "converged",
        "run_date",
    ]
    assert series.column_schema[1].type == "number"
    assert series.column_schema[1].unit == "ms"
    assert len(series.file_sha256) == 64
    assert len(series.data_sha256) == 64

    # Comment-only edits change file_sha256 while preserving data_sha256
    comment_tweaked = tmp_path / "tweaked.csv"
    comment_tweaked.write_text(
        VALID_V1_CSV.replace("Table 2, p. 4", "Table 2, page 4"),
        encoding="utf-8",
    )
    series2 = LocalSeries(comment_tweaked)
    assert series2.file_sha256 != series.file_sha256
    assert series2.data_sha256 == series.data_sha256


def test_series_loader_backwards_compatible_with_legacy_slide_csv(tmp_path: Path):
    legacy = tmp_path / "legacy_slide.csv"
    legacy.write_text(
        "# Source: Slide Author (2024)\n"
        "#   continued citation line\n"
        "# License: CC-BY-4.0 | Verified: 2026-09-01\n"
        "# Notes: Slide-local series without Columns block.\n"
        "x,y\n"
        "1,2.5\n",
        encoding="utf-8",
    )
    s = LocalSeries(legacy)
    assert s.provenance["source"] == "Slide Author (2024) continued citation line"
    assert s.provenance["verified"] == "2026-09-01"
    assert s.column_schema == ()
    assert s.columns == ("x", "y")
    assert s.entries == ({"x": 1.0, "y": 2.5},)


def test_parse_column_schema_flags_malformed_lines():
    specs, bad = parse_column_schema(
        "year: integer [year] — Publication year\n"
        "broken_line_without_brackets\n"
        "score: number [percent] -- Benchmark score\n"
    )
    assert [s.name for s in specs] == ["year", "score"]
    assert bad == ("broken_line_without_brackets",)


def test_parse_series_provenance_keeps_columns_separate_from_top_keys():
    prov = parse_series_provenance(
        [
            "# Source: Paper (@key)",
            "# Columns:",
            "#   alpha: number [ratio] — Learning rate",
            "#   beta: integer [count] — Step count",
            "# Notes: Trailing note.",
        ]
    )
    assert "alpha" not in prov
    assert "beta" not in prov
    assert prov["notes"] == "Trailing note."
    specs, bad = parse_column_schema(prov["columns"])
    assert bad == ()
    assert [s.name for s in specs] == ["alpha", "beta"]


def test_valid_mini_repo_seals_and_passes_zero_violations(tmp_path: Path):
    repo_root, qmd, _ = _make_mini_repo(tmp_path)
    manifest = fdp.seal_datapackage(repo_root=repo_root)
    assert manifest.is_file()

    pkg = json.loads(manifest.read_text(encoding="utf-8"))
    assert pkg["profile"] == "tabular-data-package"
    assert len(pkg["resources"]) == 1
    res = pkg["resources"][0]
    assert res["path"] == "vol1/01_intro/data/demo.csv"
    assert res["hash"].startswith("sha256:")
    assert res["custom"]["row_count"] == 2
    assert len(res["schema"]["fields"]) == 5

    checked, violations = fdp.collect_data_provenance(
        [qmd], scan_root=repo_root / "books", repo_root=repo_root
    )
    assert checked == 2
    assert violations == []


def test_detects_missing_v1_fields_and_invalid_tokens(tmp_path: Path):
    incomplete = (
        "# Source: Demo (@missing2099cite)\n"
        "# URL: ftp://bad-scheme.example.org\n"
        "# Verified: 2026-10-15\n"
        "# Epistemic: empirical\n"
        "# Method: scraped\n"
        "# Notes: Incomplete header.\n"
        "a,b\n"
        "1,2\n"
    )
    repo_root, _, csv_path = _make_mini_repo(tmp_path, csv_content=incomplete)
    violations = fdp.check_csv_file(csv_path, repo_root=repo_root)
    codes = {v.code for v in violations}
    assert "series_missing_figures" in codes
    assert "series_missing_locator" in codes
    assert "series_invalid_url" in codes
    assert "series_missing_license" in codes
    assert "series_invalid_epistemic" in codes
    assert "series_invalid_method" in codes
    assert "series_missing_columns" in codes
    assert "series_unresolved_citekey" in codes


def test_detects_column_order_and_cell_type_mismatches(tmp_path: Path):
    bad_order = VALID_V1_CSV.replace(
        "step,latency_ms,model_tag,converged,run_date",
        "latency_ms,step,model_tag,converged,run_date",
    )
    repo_root, _, csv_path = _make_mini_repo(tmp_path, csv_content=bad_order)
    codes = {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}
    assert "series_column_mismatch" in codes

    bad_int_cell = VALID_V1_CSV.replace("1,12.5,demo-small", "1.5,12.5,demo-small")
    csv_path.write_text(bad_int_cell, encoding="utf-8")
    codes2 = {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}
    assert "series_column_type_mismatch" in codes2


def test_detects_unresolved_and_unlisted_figure_targets(tmp_path: Path):
    wrong_target = VALID_V1_CSV.replace("vol1:fig-demo-curve", "vol1:fig-nonexistent")
    repo_root, qmd, _ = _make_mini_repo(tmp_path, csv_content=wrong_target)
    fdp.seal_datapackage(repo_root=repo_root)

    _, violations = fdp.collect_data_provenance(
        [qmd], scan_root=repo_root / "books", repo_root=repo_root
    )
    codes = {v.code for v in violations}
    assert "series_unresolved_figure_target" in codes
    assert "series_unlisted_figure_consumer" in codes


def test_detects_manifest_drift_and_orphan_entries(tmp_path: Path):
    repo_root, qmd, csv_path = _make_mini_repo(tmp_path)
    fdp.seal_datapackage(repo_root=repo_root)

    # Modify a cell without resealing -> series_manifest_drift
    csv_path.write_text(VALID_V1_CSV.replace("12.5", "13.0"), encoding="utf-8")
    _, drift_violations = fdp.collect_data_provenance(
        [qmd], scan_root=repo_root / "books", repo_root=repo_root
    )
    assert any(v.code == "series_manifest_drift" for v in drift_violations)

    # Reseal, then delete the CSV while leaving datapackage.json untouched -> series_manifest_orphan
    fdp.seal_datapackage(repo_root=repo_root)
    csv_path.unlink()
    orphan_violations = fdp.check_datapackage([], repo_root=repo_root, check_orphans=True)
    assert any(v.code == "series_manifest_orphan" for v in orphan_violations)
