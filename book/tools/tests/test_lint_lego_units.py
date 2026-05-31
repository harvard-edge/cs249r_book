"""Tests for LEGO unit discipline linter."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "book" / "tools" / "scripts"))

from lint_lego_units import lint_file, main  # noqa: E402


def _write_qmd(tmp_path: Path, rel: str, body: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _rules(issues):
    return {issue.rule for issue in issues}


def test_l019_blocks_m_as():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qmd = _write_qmd(
            root,
            "book/quarto/contents/vol1/foo/foo.qmd",
            """```{python}
x = latency.to(millisecond).m_as('ms')
```""",
        )
        issues = lint_file(qmd, root)
        assert "L019" in _rules(issues)
        assert any(i.severity == "error" for i in issues if i.rule == "L019")


def test_l004_carbon_thousand_division():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qmd = _write_qmd(
            root,
            "book/quarto/contents/vol1/foo/foo.qmd",
            """```{python}
carbon_tonnes = energy_kwh * grid_kg_co2_per_kwh / THOUSAND
```""",
        )
        issues = lint_file(qmd, root)
        assert "L004" in _rules(issues)


def test_l006_allowed_unit_label():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qmd = _write_qmd(
            root,
            "book/quarto/contents/vol1/foo/foo.qmd",
            """```{python}
mem_str = fmt_qty(mem, GiB, precision=0, commas=False, unit_label="GB")
```""",
        )
        issues = lint_file(qmd, root)
        assert "L006" not in _rules(issues)


def test_l006_disallowed_unit_label():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qmd = _write_qmd(
            root,
            "book/quarto/contents/vol1/foo/foo.qmd",
            """```{python}
mem_str = fmt_qty(mem, GiB, precision=0, commas=False, unit_label="gigabytes")
```""",
        )
        issues = lint_file(qmd, root)
        assert "L006" in _rules(issues)


def test_baseline_suppresses_known_warning():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rel = "book/quarto/contents/vol1/foo/foo.qmd"
        qmd = _write_qmd(
            root,
            rel,
            """```{python}
mem_str = fmt_qty(mem, GiB, precision=0, commas=False, unit_label="gigabytes")
```""",
        )
        baseline = root / "baseline.json"
        baseline.write_text(
            json.dumps(
                [
                    {
                        "rule": "L006",
                        "file": rel,
                        "line": 1,
                        "message": "Prefer domain formatter over unit_label=.",
                        "severity": "warning",
                    }
                ]
            ),
            encoding="utf-8",
        )
        rc = main(
            [
                str(qmd.relative_to(root)),
                "--baseline",
                str(baseline),
                "--fail-on",
                "warning",
            ]
        )
        assert rc == 0


def test_l014_closed_name_uses_fmt():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qmd = _write_qmd(
            root,
            "book/quarto/contents/vol1/foo/foo.qmd",
            """```{python}
class X:
    energy_kwh_str = fmt(1.0, precision=0)
    label_str = fmt(1.0, precision=0)
```""",
        )
        issues = lint_file(qmd, root)
        l014 = [i for i in issues if i.rule == "L014"]
        assert len(l014) == 1
        assert "energy_kwh_str" in l014[0].message
        assert not any("label_str" in i.message for i in l014)


def test_full_book_lint_with_baseline():
    """Production corpus: warnings allowed only via baseline (Phase 8½-A)."""
    baseline = ROOT / "book/tools/audit/lego_units_baseline.json"
    assert baseline.exists(), "lego_units_baseline.json required after 8½-A3"
    rc = main(["--fail-on", "warning", "--baseline", str(baseline)])
    assert rc == 0
