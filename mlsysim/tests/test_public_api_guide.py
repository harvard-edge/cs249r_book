"""Execute the documented public entry points without importing private helpers."""

from pathlib import Path
import re

from mlsysim.engine.evaluation import SystemEvaluation
from mlsysim.engine.results import ServingResult
from mlsysim.engine.engine import PerformanceProfile


def test_python_api_guide_examples_execute():
    """Keep the onboarding examples and their documented result types usable."""
    guide = Path(__file__).parents[1] / "docs" / "choosing-an-api.qmd"
    examples = re.findall(r"```\{python\}\n(.*?)\n```", guide.read_text(), re.DOTALL)
    assert len(examples) == 3, "Review result contracts when adding an API example."
    results = []
    for source in examples:
        namespace = {}
        exec(compile(source, str(guide), "exec"), namespace)
        results.append(namespace)

    assert isinstance(results[0]["profile"], PerformanceProfile)
    assert isinstance(results[1]["serving"], ServingResult)
    assert isinstance(results[2]["scorecard"], SystemEvaluation)
    assert results[0]["profile"].latency.check("[time]")
    assert results[1]["serving"].itl.check("[time]")
    assert results[1]["serving"].kv_cache_size.check("byte")


def test_code_walkthrough_reaches_the_documented_result_contract():
    guide = Path(__file__).parents[1] / "docs" / "code-walkthrough.qmd"
    examples = re.findall(r"```\{python\}\n(.*?)\n```", guide.read_text(), re.DOTALL)
    assert len(examples) == 1
    namespace = {}
    exec(compile(examples[0], str(guide), "exec"), namespace)
    assert type(namespace["case"]).__name__ == "ServingScenario"
    assert isinstance(namespace["result"], ServingResult)
    assert namespace["result"].itl.check("[time]")
    # The walkthrough names physical owners, not compatibility __module__ values.
    paths = re.findall(r"\|[^\n]+\| `([^`]+\.py)` \|", guide.read_text())
    engine = Path(__file__).parents[1] / "mlsysim" / "engine"
    assert len(paths) == 4
    for path in paths:
        assert (engine / path).is_file(), f"Walkthrough source moved: {path}"
    catalog = engine / paths[0]
    assert "Llama3_70B_ServingH100 =" in catalog.read_text()
