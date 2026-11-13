from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pytest
from mlsysim.core.series_loader import (
    LocalSeries,
    parse_column_schema,
    parse_series_provenance,
)

from binder.cli.checks import figure_data_provenance as fdp

BIB_ENTRY = (
    "@inproceedings{demo2025paper,\n"
    "  title = {Benchmarking Systems},\n"
    "  author = {Demo, Alice},\n"
    "  year = {2025}\n"
    "}\n"
)

VALID_V1_CSV = """\
# Figures: vol1:fig-demo-curve
# Source: Demo et al. (2025), Benchmarking Systems (@demo2025paper)
# Locator: Table 2, p. 4
# URL: https://arxiv.org/abs/2501.00001
# License: CC-BY-4.0
# Verified: 2026-10-15
# Epistemic: measured (board-level power meter integrated over run duration)
# Method: transcribed (manual extraction of Table 2 rows)
# Computation: Direct transcription of published median latency and convergence columns across two checkpoints.
# Columns:
#   step: integer [step] — Benchmark iteration index
#   latency_ms: number [ms] — Median request latency
#   model_tag: string [identifier] — Model checkpoint identifier
#   converged: boolean [flag] — Whether run reached target accuracy
#   run_date: date [ISO-8601 date] — Benchmark execution date
# Notes: Single-node baseline measurements on reference hardware.
step,latency_ms,model_tag,converged,run_date
1,12.5,demo-small,True,2026-01-15
2,24.0,demo-large,False,2026-02-20
"""


def _entry_sha256(raw_bib_entry: str) -> str:
    norm = re.sub(r"\s+", " ", raw_bib_entry.strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _ok_url_verifier(url: str) -> tuple[bool, str]:
    return True, f"HTTP 200 ({url})"


def _make_mini_repo(tmp_path: Path, csv_content: str = VALID_V1_CSV) -> tuple[Path, Path, Path]:
    (tmp_path / "books" / "vol1" / "01_intro" / "data").mkdir(parents=True)
    (tmp_path / "bibs").mkdir(parents=True)
    (tmp_path / "bibs" / "references.bib").write_text(BIB_ENTRY, encoding="utf-8")
    (tmp_path / "bibs" / "manifest.json").write_text(
        json.dumps(
            {
                "demo2025paper": {
                    "key": "demo2025paper",
                    "file": "bibs/references.bib",
                    "entry_type": "inproceedings",
                    "title": "Benchmarking Systems",
                    "year": "2025",
                    "entry_hash": _entry_sha256(BIB_ENTRY),
                    "status": "verified",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    qmd = tmp_path / "books" / "vol1" / "01_intro" / "01_intro.qmd"
    qmd.write_text(
        '# Intro {#sec-intro}\n\n'
        '::: {#fig-demo-curve fig-env="figure" fig-cap="**Demo**: Text." fig-alt="Line chart."}\n'
        '```{python}\n'
        '# │ Provenance: Measured latency rows from vol1/01_intro/data/demo.csv (@demo2025paper).\n'
        'rows = load_series("vol1/01_intro/data/demo.csv").entries\n'
        '```\n'
        ':::\n',
        encoding="utf-8",
    )
    csv_path = tmp_path / "books" / "vol1" / "01_intro" / "data" / "demo.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return tmp_path, qmd, csv_path


def _seal_initial(repo_root: Path) -> Path:
    return fdp.seal_datapackage(
        repo_root=repo_root,
        allow_data_change=True,
        reason="Initial admission of verified Table 2 benchmark rows",
        url_verifier=_ok_url_verifier,
    )


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
    manifest = _seal_initial(repo_root)
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

    cell_checked, cell_violations = fdp.collect_cell_provenance([qmd], repo_root=repo_root)
    assert cell_checked == 1
    assert cell_violations == []


def test_detects_missing_v1_fields_and_invalid_tokens(tmp_path: Path):
    incomplete = (
        "# Source: Demo (@missing2099cite)\n"
        "# URL: ftp://bad-scheme.arxiv.org\n"
        "# Verified: 2026-10-15\n"
        "# Epistemic: empirical\n"
        "# Method: scraped\n"
        "# Notes: Incomplete header with enough words for note check.\n"
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

    # Validate-before-seal blocks sealing broken targets
    with pytest.raises(RuntimeError, match="series_unresolved_figure_target"):
        _seal_initial(repo_root)

    _, violations = fdp.collect_data_provenance(
        [qmd], scan_root=repo_root / "books", repo_root=repo_root, verify_manifest=False
    )
    codes = {v.code for v in violations}
    assert "series_unresolved_figure_target" in codes
    assert "series_unlisted_figure_consumer" in codes


def test_detects_manifest_drift_and_orphan_entries(tmp_path: Path):
    repo_root, qmd, csv_path = _make_mini_repo(tmp_path)
    _seal_initial(repo_root)

    csv_path.write_text(VALID_V1_CSV.replace("12.5", "13.0"), encoding="utf-8")
    _, drift_violations = fdp.collect_data_provenance(
        [qmd], scan_root=repo_root / "books", repo_root=repo_root
    )
    assert any(v.code == "series_manifest_drift" for v in drift_violations)

    fdp.seal_datapackage(
        repo_root=repo_root,
        allow_data_change=True,
        reason="Corrected row 1 latency typo against Table 2 errata",
        url_verifier=_ok_url_verifier,
    )
    csv_path.unlink()
    orphan_violations = fdp.check_datapackage([], repo_root=repo_root, check_orphans=True)
    assert any(v.code == "series_manifest_orphan" for v in orphan_violations)


def test_seal_ratchet_blocks_rubber_stamp_on_data_edits_new_csvs_and_dead_urls(tmp_path: Path):
    repo_root, _, csv_path = _make_mini_repo(tmp_path)

    # Plain metadata-only --seal refuses initial unsealed CSV admission
    with pytest.raises(RuntimeError, match="Refusing metadata-only --seal"):
        fdp.seal_datapackage(repo_root=repo_root)

    _seal_initial(repo_root)

    # Comment-only metadata edit is permitted by plain --seal because data_sha256 is unchanged
    csv_path.write_text(VALID_V1_CSV.replace("Table 2, p. 4", "Table 2, page 4"), encoding="utf-8")
    fdp.seal_datapackage(repo_root=repo_root)

    # Mutating a data cell changes data_sha256 -> plain --seal MUST refuse
    csv_path.write_text(VALID_V1_CSV.replace("12.5", "99.9"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified data_sha256 rows"):
        fdp.seal_datapackage(repo_root=repo_root)

    # --seal-data without a substantive >=5-word reason MUST refuse
    with pytest.raises(RuntimeError, match="at least 5 concrete words"):
        fdp.seal_datapackage(
            repo_root=repo_root,
            allow_data_change=True,
            reason="fix",
            url_verifier=_ok_url_verifier,
        )

    # --seal-data with a dead / unreachable URL MUST refuse
    with pytest.raises(RuntimeError, match="live URL check failed"):
        fdp.seal_datapackage(
            repo_root=repo_root,
            allow_data_change=True,
            reason="Updated row 1 latency from revised upstream preprint",
            url_verifier=lambda _u: (False, "HTTP 404"),
        )


def test_rejects_missing_unverified_and_tampered_bib_manifest_citations(tmp_path: Path):
    # 1. Omitting @bibkey in # Source: fails with series_missing_citekey
    no_cite = VALID_V1_CSV.replace("(@demo2025paper)", "(2025)")
    repo_root, _, csv_path = _make_mini_repo(tmp_path, csv_content=no_cite)
    codes = {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}
    assert "series_missing_citekey" in codes

    # 2. Citing a key in references.bib whose status in bibs/manifest.json != 'verified' fails
    csv_path.write_text(VALID_V1_CSV, encoding="utf-8")
    (repo_root / "bibs" / "manifest.json").write_text(
        json.dumps({"demo2025paper": {"key": "demo2025paper", "status": "pending"}}),
        encoding="utf-8",
    )
    codes2 = {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}
    assert "series_unverified_citekey" in codes2

    # 3. Tampering with references.bib after bibs/manifest.json sealed entry_hash fails
    (repo_root / "bibs" / "manifest.json").write_text(
        json.dumps(
            {
                "demo2025paper": {
                    "key": "demo2025paper",
                    "status": "verified",
                    "entry_hash": _entry_sha256(BIB_ENTRY),
                }
            }
        ),
        encoding="utf-8",
    )
    (repo_root / "bibs" / "references.bib").write_text(
        BIB_ENTRY.replace("Benchmarking Systems", "Invented Replacement Title"),
        encoding="utf-8",
    )
    codes3 = {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}
    assert "series_unverified_citekey" in codes3


def test_rejects_placeholder_vacuous_future_date_and_reassurance_metadata(tmp_path: Path):
    banned_word = "gen" + "uine"
    slop = (
        "# Figures: vol1:fig-demo-curve\n"
        "# Source: Demo (@demo2025paper)\n"
        "# Locator: table\n"
        "# URL: https://example.com/fake\n"
        "# License: TODO\n"
        "# Verified: 2099-01-01\n"
        "# Epistemic: measured\n"
        "# Method: transcribed\n"
        f"# Notes: {banned_word} numbers.\n"
        "# Columns:\n"
        "#   step: integer [unit] — step\n"
        "step\n"
        "1\n"
    )
    repo_root, _, csv_path = _make_mini_repo(tmp_path, csv_content=slop)
    codes = {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}
    assert "series_banned_reassurance_word" in codes
    assert "series_vacuous_metadata" in codes
    assert "series_placeholder_metadata" in codes
    assert "series_invalid_verified" in codes


def test_rejects_empty_ragged_and_blank_column_csvs(tmp_path: Path):
    header_only = "\n".join(VALID_V1_CSV.splitlines()[:-2]) + "\n"
    repo_root, _, csv_path = _make_mini_repo(tmp_path, csv_content=header_only)
    assert "series_empty_rows" in {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}

    ragged = VALID_V1_CSV + "3,15.0,extra,True,2026-03-01,unexpected_cell\n"
    csv_path.write_text(ragged, encoding="utf-8")
    assert "series_ragged_row" in {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}

    blank_col = VALID_V1_CSV.replace("demo-small", "").replace("demo-large", "")
    csv_path.write_text(blank_col, encoding="utf-8")
    assert "series_empty_column" in {v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)}


def test_rejects_auxiliary_orphan_parking_and_unrelated_generator(tmp_path: Path):
    repo_root, _, csv_path = _make_mini_repo(tmp_path)
    (repo_root / "README.md").write_text("Project readme\n", encoding="utf-8")
    (repo_root / "mlsysim" / "tests").mkdir(parents=True)
    unrelated_test = repo_root / "mlsysim" / "tests" / "test_other.py"
    unrelated_test.write_text("def test_unrelated():\n    assert 1 == 1\n", encoding="utf-8")

    # Parking on README.md fails
    csv_path.write_text(
        VALID_V1_CSV.replace("vol1:fig-demo-curve", "auxiliary (README.md)"),
        encoding="utf-8",
    )
    assert "series_unresolved_figure_target" in {
        v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)
    }

    # Parking on a test file that never mentions demo.csv fails
    csv_path.write_text(
        VALID_V1_CSV.replace("vol1:fig-demo-curve", "auxiliary (mlsysim/tests/test_other.py)"),
        encoding="utf-8",
    )
    assert "series_unresolved_figure_target" in {
        v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)
    }

    # Pointing # Generator: at a script that never mentions demo.csv fails
    (repo_root / "binder" / "tools" / "scripts").mkdir(parents=True)
    (repo_root / "binder" / "tools" / "scripts" / "unrelated.py").write_text(
        "print('hello')\n", encoding="utf-8"
    )
    csv_path.write_text(
        VALID_V1_CSV.replace(
            "# Method: transcribed (manual extraction of Table 2 rows)\n",
            "# Method: scripted (generated)\n# Generator: python3 binder/tools/scripts/unrelated.py\n",
        ),
        encoding="utf-8",
    )
    assert "series_missing_generator_script" in {
        v.code for v in fdp.check_csv_file(csv_path, repo_root=repo_root)
    }


def test_blocks_qmd_direct_readers_dynamic_load_series_and_stray_datasets(tmp_path: Path):
    repo_root, qmd, _ = _make_mini_repo(tmp_path)
    _seal_initial(repo_root)

    # Stray CSV outside chapter data/ directory
    stray = repo_root / "books" / "vol1" / "01_intro" / "sneaky.csv"
    stray.write_text("x,y\n1,2\n", encoding="utf-8")

    # Direct pd.read_csv and dynamic load_series(target_var) inside chapter .qmd
    qmd.write_text(
        qmd.read_text(encoding="utf-8")
        + '\n```{python}\n'
        + 'df = pd.read_csv("sneaky.csv")\n'
        + 'target_var = "vol1/01_intro/data/demo.csv"\n'
        + 'rows2 = load_series(target_var)\n'
        + '```\n',
        encoding="utf-8",
    )
    _, violations = fdp.collect_data_provenance(
        [qmd], scan_root=repo_root / "books", repo_root=repo_root
    )
    codes = {v.code for v in violations}
    assert "stray_chapter_dataset" in codes
    assert "qmd_bypass_series_loader" in codes
    assert "qmd_dynamic_load_series" in codes


def test_cell_provenance_blocks_stubs_unbacked_measured_and_inline_arrays(tmp_path: Path):
    repo_root, qmd, _ = _make_mini_repo(tmp_path)
    _seal_initial(repo_root)

    qmd.write_text(
        '# Intro {#sec-intro}\n\n'
        '::: {#fig-stub fig-env="figure" fig-cap="**Stub**: Text." fig-alt="Chart."}\n'
        '```{python}\n'
        '# Provenance: ok\n'
        'x = [1, 2]\n'
        '```\n'
        ':::\n\n'
        '::: {#fig-unbacked fig-env="figure" fig-cap="**Unbacked**: Text." fig-alt="Chart."}\n'
        '```{python}\n'
        '# │ Provenance: Measured throughput across cluster sizes on eight nodes.\n'
        'xs = [1, 2, 4, 8]\n'
        '```\n'
        ':::\n\n'
        '::: {#fig-demo-curve fig-env="figure" fig-cap="**Demo**: Text." fig-alt="Chart."}\n'
        '```{python}\n'
        '# │ Provenance: Measured latency rows from vol1/01_intro/data/demo.csv (@demo2025paper).\n'
        'rows = load_series("vol1/01_intro/data/demo.csv").entries\n'
        'secret_inline = [61.2, 68.4, 74.9, 81.3, 88.7]\n'
        '```\n'
        ':::\n',
        encoding="utf-8",
    )
    _, violations = fdp.collect_cell_provenance([qmd], repo_root=repo_root)
    by_msg = [(v.code, v.message) for v in violations]
    codes = [c for c, _ in by_msg]
    assert codes.count("figure_stub_cell_provenance") == 1
    assert codes.count("figure_unbacked_measured_cell") == 2


def test_binder_validate_fix_flag_never_rubber_stamps_datapackage(tmp_path: Path):
    from binder.cli.commands.validate import ValidateCommand

    repo_root, qmd, csv_path = _make_mini_repo(tmp_path)
    manifest = _seal_initial(repo_root)
    before_bytes = manifest.read_bytes()

    # Mutate CSV on disk
    csv_path.write_text(VALID_V1_CSV.replace("12.5", "44.0"), encoding="utf-8")

    # Even when ns.fix=True is passed, _run_figure_data_provenance must NOT rewrite datapackage.json
    cmd = ValidateCommand.__new__(ValidateCommand)
    cmd._qmd_files = lambda _root: [qmd]  # type: ignore[method-assign]
    ns = argparse.Namespace(fix=True, files=[str(csv_path)])
    result = cmd._run_figure_data_provenance(repo_root / "books", ns=ns)

    assert manifest.read_bytes() == before_bytes
    assert any(i.code == "series_manifest_drift" for i in result.issues)

    # Pre-commit positional ns.files fast path also blocks lone stray dataset files
    rogue_parquet = qmd.parent / "rogue.parquet"
    rogue_parquet.write_bytes(b"PAR1")
    ns_stray = argparse.Namespace(fix=False, files=[str(rogue_parquet)])
    res_stray = cmd._run_figure_data_provenance(repo_root / "books", ns=ns_stray)
    assert any(i.code == "stray_chapter_dataset" for i in res_stray.issues)
