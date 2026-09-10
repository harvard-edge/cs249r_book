"""Generated macros the paper cites must exist and keep their rendered form.

An undefined macro fails the LaTeX build loudly, which is safe. The dangerous
case is quieter: a workload's metric key changes, the percent-versus-decimal
branch in the generator flips, and a sentence that read "78.52%" silently
renders "0.79" while still compiling and still passing the placeholder check.
These tests pin both the existence and the shape.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER_TEX = ROOT / "paper" / "paper.tex"
GENERATED = ROOT / "paper" / "generated_registry.tex"

DEFINED = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
PERCENT_FORM = re.compile(r"^\d+\.\d{2}\\%$")
DECIMAL_FORM = re.compile(r"^\d+\.\d+$")


def defined_macros() -> dict[str, str]:
    return dict(DEFINED.findall(GENERATED.read_text(encoding="utf-8")))


def cited_macros() -> set[str]:
    body = PAPER_TEX.read_text(encoding="utf-8")
    return set(re.findall(r"\\([A-Z][A-Za-z]*)\{\}", body))


def test_every_macro_the_paper_cites_is_generated():
    missing = sorted(cited_macros() - set(defined_macros()))
    assert not missing, f"paper.tex cites undefined generated macros: {missing}"


def test_measured_and_published_macros_come_in_pairs():
    """A comparison sentence needs both halves, or it silently loses one side."""
    macros = defined_macros()
    for name in macros:
        if name.endswith("Measured"):
            assert f"{name[:-8]}Published" in macros, f"{name} has no Published pair"
        if name.endswith("Published"):
            assert f"{name[:-9]}Measured" in macros, f"{name} has no Measured pair"


def test_measured_macros_render_in_the_form_their_metric_implies():
    """A flipped percent/decimal branch compiles fine and reads wrong."""
    macros = defined_macros()
    suites = ROOT / "registry" / "suites"
    for path in sorted(suites.glob("*/*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        contract = spec.get("canonical_max_contract") or {}
        evidence = contract.get("measured_evidence") or {}
        if evidence.get("score", evidence.get("best_score")) is None:
            continue
        name = "".join(part.capitalize() for part in path.stem.split("-"))
        gate = contract.get("quality") or {}
        key = str(gate.get("metric_key") or gate.get("metric") or "")
        expected_percent = key.endswith(("accuracy", "pass_at_1", "ndcg_at_10"))
        for suffix in ("Measured", "Published"):
            value = macros.get(f"{name}{suffix}")
            assert value is not None, f"{name}{suffix} was not generated"
            pattern = PERCENT_FORM if expected_percent else DECIMAL_FORM
            assert pattern.match(value), (
                f"{name}{suffix} rendered {value!r}, which does not match the "
                f"form implied by metric {key!r}"
            )


def test_generated_file_is_marked_as_generated():
    assert "Do not edit by hand" in GENERATED.read_text(encoding="utf-8")
