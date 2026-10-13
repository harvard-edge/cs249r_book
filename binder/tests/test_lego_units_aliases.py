"""L009 follows units.py's exported aliases; L002 catches units glued on via suffix= (2026-10-13)."""
import sys
from pathlib import Path

BINDER_ROOT = Path(__file__).resolve().parents[1]
if str(BINDER_ROOT) not in sys.path:
    sys.path.insert(0, str(BINDER_ROOT))

from cli.checks.lego_units import lint_file


def _lint(tmp_path: Path, code: str) -> list[str]:
    qmd = tmp_path / "c.qmd"
    qmd.write_text("```{python}\n" + code + "\n```\n", encoding="utf-8")
    return [i.rule for i in lint_file(qmd, tmp_path)]


def test_l009_flags_any_exported_alias(tmp_path: Path):
    assert "L009" in _lint(tmp_path, "x = 3 * ureg.millimeter")
    assert "L009" in _lint(tmp_path, "y = 2 * ureg.terawatt_hour")
    assert "L009" in _lint(tmp_path, "z = 5 * ureg.degC")


def test_l009_leaves_constructors_and_unaliased_constants(tmp_path: Path):
    assert "L009" not in _lint(tmp_path, "q = ureg.Quantity(20.0, degC)")
    assert "L009" not in _lint(tmp_path, "c = 1 * ureg.speed_of_light")


def test_l002_flags_units_glued_on_with_suffix(tmp_path: Path):
    for suffix in (" V", " hours", " Mb/s", " N/mm", " rad/s²", " TWh", " Mt"):
        assert "L002" in _lint(tmp_path, f'v_str = fmt(v, precision=0, suffix="{suffix}")'), suffix


def test_l002_allows_thousands_marker_and_typed_formatters(tmp_path: Path):
    assert "L002" not in _lint(tmp_path, 'n_str = fmt_count(n, commas=False, suffix="K")')
    assert "L002" not in _lint(tmp_path, 'v_str = fmt_voltage(v, unit=volt, precision=2)')
