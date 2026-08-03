"""The paper's figures must be reproducible by anyone who has the repository.

A figure generated from a scratch directory looks identical in the built PDF to
one generated from committed evidence, and the LaTeX build cannot tell the
difference. That is how a stale figure survives: `make` verifies that the PDF
compiles, not that the picture inside it still reflects the registry. These
tests close that gap by regenerating every figure the paper includes, using
only what is committed, and failing if any of them cannot be produced.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_TEX = ROOT / "paper" / "paper.tex"
FIGURE_DIR = ROOT / "paper" / "figures"
COMMITTED_RUNS = ROOT / "paper" / "evidence" / "runs"

INCLUDED = re.compile(r"\\includegraphics\[[^\]]*\]\{figures/([A-Za-z0-9_\-]+)\.pdf\}")


def included_figures() -> list[str]:
    return sorted(set(INCLUDED.findall(PAPER_TEX.read_text(encoding="utf-8"))))


def load_generator():
    """Import the generator by path; tools/ is not an installed package."""
    path = ROOT / "tools" / "generate_paper_figures.py"
    spec = importlib.util.spec_from_file_location("generate_paper_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paper_includes_figures():
    """A regression guard: the paper used to ship with no figures at all."""
    assert included_figures(), "paper.tex includes no figures"


def test_every_included_figure_is_committed():
    for name in included_figures():
        assert (FIGURE_DIR / f"{name}.pdf").is_file(), (
            f"paper.tex includes figures/{name}.pdf but it is not committed; "
            "the paper would not build from a clean checkout"
        )


def test_run_reports_backing_the_figures_are_committed():
    reports = sorted(COMMITTED_RUNS.glob("*_max_report.json"))
    assert reports, (
        f"no run reports under {COMMITTED_RUNS.relative_to(ROOT)}; the figures "
        "would fall back to whatever a scratch directory happened to hold"
    )


def test_every_included_figure_regenerates_from_committed_inputs(tmp_path, monkeypatch):
    """Render into a temp directory and require each included figure to appear.

    The generator skips a figure when its data is missing rather than failing,
    so absence from the output is the signal, not an exception.
    """
    module = load_generator()
    monkeypatch.setattr(module, "OUT", tmp_path)

    module.style()
    workloads = module.load_registry(ROOT / "registry")
    reports = module.load_reports([COMMITTED_RUNS])
    evidence = module.load_evidence()

    module.fig_quality_vs_target(workloads, reports, evidence)
    module.fig_runtime(workloads, reports, evidence)
    module.fig_training_curves(workloads, reports)
    module.fig_repeatability(evidence)

    produced = {path.stem for path in tmp_path.glob("*.pdf")}
    missing = [name for name in included_figures() if name not in produced]
    assert not missing, (
        f"paper.tex includes {missing} but the generator could not produce them "
        "from committed inputs; the shipped PDFs are stale"
    )
