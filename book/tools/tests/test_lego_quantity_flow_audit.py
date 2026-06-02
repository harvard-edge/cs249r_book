"""Tests for the advisory LEGO quantity-flow audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "book" / "tools" / "audit"))

from book_check_lego_quantity_flow import check_file  # noqa: E402


def _write_qmd(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "chapter.qmd"
    path.write_text(body, encoding="utf-8")
    return path


def _rules(path: Path) -> set[str]:
    return {issue.rule for issue in check_file(path)}


def test_flags_scalar_reattachment_and_alias_fallback(tmp_path: Path):
    qmd = _write_qmd(
        tmp_path,
        """```{python}
# ┌── LEGO ───────────────────────────────────────────────
from mlsysim.core.units import ureg

class X:
    # ┌── 1. LOAD ─────────────────────────────────────────
    bandwidth = 1.9 * TB / second
    # ┌── 2. EXECUTE ──────────────────────────────────────
    bw_tbs = bandwidth.to(TB / second).magnitude
    # ┌── 4. OUTPUT ───────────────────────────────────────
    bw_str = fmt_qty(bw_tbs * TB / second, TB / second, precision=1)
    e_str = fmt_qty(energy, ureg.joule, precision=0)
```""",
    )
    assert {"QF001", "QF002", "QF004", "ST001"} <= _rules(qmd)


def test_ignores_non_lego_cells_by_default(tmp_path: Path):
    qmd = _write_qmd(
        tmp_path,
        """```{python}
bw_tbs = bandwidth.to(TB / second).magnitude
```""",
    )
    assert _rules(qmd) == set()


def test_flags_fmt_count_precision_zero_boilerplate(tmp_path: Path):
    qmd = _write_qmd(
        tmp_path,
        """```{python}
# │ Exports: X.params_str
class X:
    params_str = fmt_count(params, scale="B", precision=0, commas=False)
```""",
    )
    assert "QF005" in _rules(qmd)


def test_flags_formatted_string_reused_as_float(tmp_path: Path):
    qmd = _write_qmd(
        tmp_path,
        """```{python}
# │ Exports: X.t_str
class X:
    bandwidth_str = fmt(bandwidth_gbs, precision=0)
    seconds = payload_gb / float(bandwidth_str)
    t_str = fmt_time(seconds, second)
```""",
    )
    assert "QF007" in _rules(qmd)
