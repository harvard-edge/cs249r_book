"""Regression checks for numerical impact reporting, including failure paths."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/downstream_impact.py"
spec = importlib.util.spec_from_file_location("downstream_impact", SCRIPT)
impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impact)


def snapshot_data():
    """Construct a declared numerical comparison, not a hardware measurement."""
    consumer = "books/vol1/example.qmd"
    return {
        "schema": impact.SCHEMA, "selection": {"consumers": [consumer]},
        "inputs_sha256": {consumer: "unchanged-source"}, "tool_sha256": "same-tool",
        "python": "same-interpreter", "environment": {"pint": "same-version"}, "package_sha256": {"model.py": "baseline-code"},
        "errors": [], "consumers": {consumer: {
            "source_sha256": "unchanged-source", "imports_sha256": {}, "cells": [{"line": 1, "sha256": "same-cell"}],
            "errors": [], "skipped": [], "raw": {"Example": {"latency": 1.001}},
            "inline": {"line:Example.latency_str": {
                "expression": "Example.latency_str", "displayed": "1.0", "raw": "1.0",
                "line": 5, "prose": "The latency is `{python} Example.latency_str` ms.",
            }},
        }},
    }


def test_raw_change_survives_identical_rounded_prose():
    before = snapshot_data()
    after = deepcopy(before)
    after["consumers"]["books/vol1/example.qmd"]["raw"]["Example"]["latency"] = 1.002
    report = impact.compare(before, after)
    assert report["status"] == "review_required"
    assert report["changes"][0]["output"] == "Example.latency"
    assert report["changes"][0]["nearby_prose"][0]["line"] == 5


def test_fixed_consumers_with_unchanged_values_pass():
    before = snapshot_data()
    after = deepcopy(before)
    after["package_sha256"] = {"model.py": "candidate-code"}
    assert impact.compare(before, after)["status"] == "unchanged"


@pytest.mark.parametrize("field", ["schema", "selection", "inputs_sha256", "tool_sha256", "python", "environment"])
def test_different_or_missing_comparison_context_cannot_pass(field):
    before = snapshot_data()
    after = deepcopy(before)
    del after[field]
    assert impact.compare(before, after)["status"] == "incomplete"


@pytest.mark.parametrize("field", ["source_sha256", "cells", "inline", "raw"])
def test_missing_consumer_coverage_cannot_pass(field):
    before = snapshot_data()
    after = deepcopy(before)
    del after["consumers"]["books/vol1/example.qmd"][field]
    assert impact.compare(before, after)["status"] == "incomplete"


def test_missing_consumer_cannot_pass():
    before = snapshot_data()
    after = deepcopy(before)
    after["consumers"] = {}
    assert impact.compare(before, after)["status"] == "incomplete"


def test_execution_error_cannot_be_called_unchanged():
    before = snapshot_data()
    after = deepcopy(before)
    after["consumers"]["books/vol1/example.qmd"]["errors"] = [{"error": "ValueError: invalid model"}]
    assert impact.compare(before, after)["status"] == "incomplete"


def test_identically_skipped_cells_require_review():
    before = snapshot_data()
    before["consumers"]["books/vol1/example.qmd"]["skipped"] = [{"line": 2, "reason": "eval:false"}]
    report = impact.compare(before, deepcopy(before))
    assert report["status"] == "review_required"
    assert report["review_requirements"]


def test_changed_skip_coverage_is_incomplete():
    before = snapshot_data()
    after = deepcopy(before)
    after["consumers"]["books/vol1/example.qmd"]["skipped"] = [{"line": 2, "reason": "eval:false"}]
    assert impact.compare(before, after)["status"] == "incomplete"


def test_qmd_capture_keeps_unrounded_classes_and_prose_but_excludes_examples(tmp_path, monkeypatch):
    monkeypatch.setattr(impact, "suppress_figures", lambda events: None)
    path = tmp_path / "example.qmd"
    path.write_text('''```{python}
class Example:
    latency = 1.001
    latency_str = f"{latency:.1f}"
```

Latency is `{python} Example.latency_str` ms.

```text
A non-executable example: `{python} nonexistent`
```

```{python}
#| eval: false
raise RuntimeError("must not execute")
```
''')
    row = impact.snapshot_qmd(path, tmp_path)
    assert row["errors"] == []
    assert row["raw"]["Example"]["latency"] == 1.001
    assert len(row["inline"]) == 1
    assert next(iter(row["inline"].values()))["displayed"] == "1.0"
    assert row["skipped"] == [{"line": 13, "reason": "eval:false"}]


def test_snapshot_uses_selected_package_in_isolated_process(tmp_path):
    root = tmp_path / "consumer"
    (root / "books/vol1").mkdir(parents=True)
    (root / "books/vol1/example.qmd").write_text('''```{python}
from mlsysim import VALUE
class Example:
    raw = VALUE
    displayed = f"{VALUE:.1f}"
```
The estimate is `{python} Example.displayed`.
''')
    snapshots = []
    for label, value in (("before", "1.001"), ("after", "1.002")):
        package = tmp_path / label
        (package / "mlsysim").mkdir(parents=True)
        (package / "mlsysim/__init__.py").write_text(f"VALUE = {value}\n")
        data = impact.snapshot(root, package, ["books/vol1/*.qmd"], False, 1, 30)
        assert data["errors"] == []
        row = data["consumers"]["books/vol1/example.qmd"]
        assert row["errors"] == []
        assert row["package_origin"] == str(package / "mlsysim/__init__.py")
        snapshots.append(data)
    report = impact.compare(*snapshots)
    assert report["status"] == "review_required"
    assert any(item["output"] == "Example.raw" for item in report["changes"])
    assert not any(item["kind"] == "inline" for item in report["changes"])


def test_document_disabled_execution_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(impact, "suppress_figures", lambda events: None)
    path = tmp_path / "example.qmd"
    path.write_text("---\nexecute:\n  eval: false\n---\n\n```{python}\nraise RuntimeError('disabled')\n```\n")
    row = impact.snapshot_qmd(path, tmp_path)
    assert row["errors"] == []
    assert row["skipped"][0]["reason"] == "document eval:false"


def test_empty_observation_is_not_a_success():
    before = snapshot_data()
    record = before["consumers"]["books/vol1/example.qmd"]
    record["inline"] = {}
    record["raw"] = {}
    assert impact.compare(before, deepcopy(before))["status"] == "review_required"


def test_unmatched_declared_glob_is_incomplete_even_when_other_consumers_exist(tmp_path):
    root = tmp_path / "consumer"
    (root / "books/vol1").mkdir(parents=True)
    (root / "books/vol1/example.qmd").write_text("```{python}\nvalue = 1\n```\n")
    package = tmp_path / "package"
    (package / "mlsysim").mkdir(parents=True)
    (package / "mlsysim/__init__.py").write_text("")
    row = impact.snapshot(root, package, ["books/vol1/*.qmd", "books/missing/*.qmd"], False, 1, 30)
    assert row["consumers"]
    assert any("No QMD sources matched" in error for error in row["errors"])
    assert impact.compare(row, deepcopy(row))["status"] == "incomplete"


def test_quantity_instances_are_recorded_but_imported_classes_are_not():
    from mlsysim import Q_
    assert impact.encode(Q_) is None
    assert impact.encode(Q_("1.001 ms")) == {"quantity": 1.001, "unit": "millisecond",
                                            "unit_powers": {"millisecond": "1"}}


def test_indented_markdown_python_fence_is_executed(tmp_path, monkeypatch):
    monkeypatch.setattr(impact, "suppress_figures", lambda events: None)
    path = tmp_path / "example.qmd"
    path.write_text("   ```{python}\n   value = 7\n   ```\n\nThe value is `{python} value`.\n")
    row = impact.snapshot_qmd(path, tmp_path)
    assert row["errors"] == []
    assert row["raw"]["value"] == 7
    assert next(iter(row["inline"].values()))["displayed"] == "7"


def test_unsupported_fence_cannot_be_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(impact, "suppress_figures", lambda events: None)
    path = tmp_path / "example.qmd"
    path.write_text("```{python}\nvalue = 7\n```\n\n```{python}\nvalue = 9\n")
    row = impact.snapshot_qmd(path, tmp_path)
    assert any("coverage is incomplete" in error["error"] for error in row["errors"])


def test_empty_container_shape_changes_require_review():
    before = snapshot_data()
    after = deepcopy(before)
    before["consumers"]["books/vol1/example.qmd"]["raw"]["values"] = []
    after["consumers"]["books/vol1/example.qmd"]["raw"]["values"] = [[], {}]
    assert impact.compare(before, after)["status"] == "review_required"


def test_stale_bytecode_cannot_hide_changed_source(tmp_path):
    import os
    import py_compile

    root = tmp_path / "consumer"
    (root / "books/vol1").mkdir(parents=True)
    (root / "books/vol1/example.qmd").write_text("```{python}\nfrom mlsysim import VALUE\nvalue = VALUE\n```\n")
    package = tmp_path / "package"
    (package / "mlsysim").mkdir(parents=True)
    source = package / "mlsysim/__init__.py"
    source.write_text("VALUE = 1.001\n")
    py_compile.compile(str(source), doraise=True)
    initial = source.stat()
    before = impact.snapshot(root, package, ["books/vol1/*.qmd"], False, 1, 30)
    source.write_text("VALUE = 9.009\n")
    os.utime(source, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    after = impact.snapshot(root, package, ["books/vol1/*.qmd"], False, 1, 30)
    assert before["consumers"]["books/vol1/example.qmd"]["raw"]["value"] == 1.001
    assert after["consumers"]["books/vol1/example.qmd"]["raw"]["value"] == 9.009
    assert impact.compare(before, after)["status"] == "review_required"


def test_fractional_unit_power_change_survives_identical_display_units():
    import math
    from mlsysim import Q_

    power = 1 / 3
    adjacent = math.nextafter(power, math.inf)
    first = impact.encode(Q_(1, f"hertz ** {power!r}"))
    second = impact.encode(Q_(1, f"hertz ** {adjacent!r}"))
    assert first["unit"] == second["unit"]
    assert first["unit_powers"] != second["unit_powers"]
    before = snapshot_data()
    after = deepcopy(before)
    before["consumers"]["books/vol1/example.qmd"]["raw"]["fractional"] = first
    after["consumers"]["books/vol1/example.qmd"]["raw"]["fractional"] = second
    report = impact.compare(before, after)
    assert report["status"] == "review_required"
    assert any("unit_powers" in row["output"] for row in report["changes"])
