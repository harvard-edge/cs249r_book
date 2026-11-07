"""The Volume IV generator template must track the installed MLSysIM wheel."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = ROOT / "labs/vol4/build_companions.py"
SPEC = importlib.util.spec_from_file_location("build_companions", GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_rendered_bootstrap_uses_current_mlsysim_version_without_writing_notebooks():
    target = ROOT / "labs/vol4/lab_18_frontier.py"
    before = target.read_bytes()
    contract = {
        "chapter": 18,
        "source_notebook": "labs/vol4/lab_18_frontier.py",
        "book_anchor": "sec-frontier",
    }

    rendered = GENERATOR.render_entrypoint(contract)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    version = tomllib.loads((ROOT / "mlsysim/pyproject.toml").read_text())["project"]["version"]

    assert f"../../wheels/mlsysim-{version}-py3-none-any.whl" in rendered
    assert "../../wheels/mlsysim-0.3.0-py3-none-any.whl" not in rendered
    assert target.read_bytes() == before
