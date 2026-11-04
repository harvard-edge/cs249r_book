"""Guard scenario API discovery and exact member coverage without a Quarto build."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "docs" / "scripts" / "build_api.py"
SPEC = importlib.util.spec_from_file_location("scenario_api_build", SCRIPT)
API_BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API_BUILD)


def test_scenario_reference_discovers_classes_and_public_properties(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = tmp_path / "mlsysim" / "engine"
    source.mkdir(parents=True)
    (source / "scenario_types.py").write_text(
        "class NewScenario:\n"
        "    @property\n"
        "    def latency(self):\n"
        "        raise RuntimeError('discovery must not evaluate properties')\n"
        "    def _private(self):\n"
        "        pass\n"
        "class _PrivateScenario:\n"
        "    pass\n"
    )
    monkeypatch.setattr(API_BUILD, "DOCS", docs)
    assert API_BUILD.scenario_contracts() == {"NewScenario": ["latency"]}


def test_scenario_reference_rejects_missing_exact_member_anchor(monkeypatch):
    monkeypatch.setattr(API_BUILD, "scenario_contracts", lambda: {"NewScenario": ["latency"]})
    # Isolate the scenario check from unrelated registry/solver coverage.
    monkeypatch.setattr(API_BUILD.importlib, "import_module", lambda name: SimpleNamespace(__all__=[]))
    name = "mlsysim.engine.scenario_types.NewScenario"
    anchors = {name, "mlsysim.engine.scenario_types.OtherScenario.latency"}
    monkeypatch.setattr(API_BUILD, "documented_anchors", lambda: anchors)
    assert API_BUILD.check() == [f"{name}.latency"]
    anchors.add(f"{name}.latency")
    assert API_BUILD.check() == []
    anchors.remove(name)
    assert API_BUILD.check() == [name]


def test_scenario_module_marker_expands_to_class_pages(monkeypatch):
    monkeypatch.setattr(API_BUILD, "scenario_contracts", lambda: {"ZScenario": [], "AScenario": []})
    config = API_BUILD.expanded_config()
    sections = config["quartodoc"]["sections"]
    scenarios = [section for section in sections if section["title"] == API_BUILD.SCENARIO_TITLE]
    assert len(scenarios) == 1
    assert scenarios[0]["contents"] == [
        {"kind": "page", "flatten": True, "path": "engine.scenario_types.AScenario", "contents": [
            {"name": "engine.scenario_types.AScenario", "include_empty": True}]},
        {"kind": "page", "flatten": True, "path": "engine.scenario_types.ZScenario", "contents": [
            {"name": "engine.scenario_types.ZScenario", "include_empty": True}]},
    ]
    assert sections[-1] is scenarios[0], "Keep scenario tables off the API landing page."


def test_scenario_page_format_keeps_anchors_and_exposes_input_contract(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    api = docs / "api"
    api.mkdir(parents=True)
    source = tmp_path / "mlsysim" / "engine"
    source.mkdir(parents=True)
    (source / "scenario_types.py").write_text(
        "class NewScenario:\n    batch: int = 8\n    @property\n"
        "    def latency(self):\n        raise RuntimeError('must not evaluate')\n"
    )
    prefix = "engine.scenario_types.NewScenario"
    anchor = f"mlsysim.{prefix}"
    page = api / f"{prefix}.qmd"
    page.write_text(
        f"# {prefix} {{ #{anchor} }}\n\n"
        f"# {prefix}.batch {{ #{anchor}.batch }}\n\n`{prefix}.batch`\n\n"
        f"# {prefix}.latency {{ #{anchor}.latency }}\n\nEquation: bytes / bandwidth.\n"
    )
    monkeypatch.setattr(API_BUILD, "DOCS", docs)
    monkeypatch.setattr(API_BUILD, "API", api)
    API_BUILD.format_scenario_pages()
    text = page.read_text()
    assert f"# NewScenario {{ #{anchor} }}" in text
    assert f"## latency {{ #{anchor}.latency }}" in text
    assert "**Type:** `int`." in text
    assert "**Declared default:** `8`." in text
    assert "Equation: bytes / bandwidth." in text


def test_scenario_pages_render_properties_and_inputs_without_inherited_framework_members():
    config = API_BUILD.expanded_config()
    sections = config["quartodoc"]["sections"]
    scenarios = next(section for section in sections if section["title"] == API_BUILD.SCENARIO_TITLE)
    name = "engine.scenario_types.ReviewStaffingScenario"
    page = next(page for page in scenarios["contents"] if page["path"] == name)
    assert f"{name}.appeals_per_day" in page["contents"]
    assert f"{name}.decisions_per_day" in page["contents"]
    assert f"{name}.model_dump" not in page["contents"]
    assert config["quartodoc"]["options"]["include_inherited"] is False


def test_scenario_domains_follow_defining_files_and_keep_legacy_links(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    engine = tmp_path / "mlsysim" / "engine"
    definitions = engine / "scenario_definitions"
    definitions.mkdir(parents=True)
    (definitions / "serving.py").write_text(
        '"""Serving latency and memory accounting."""\n'
        'class DecodeScenario:\n'
        '    """Decode one autoregressive step."""\n'
        '    __module__ = "mlsysim.engine.scenario_types"\n'
        '    @property\n'
        '    def latency(self):\n'
        '        raise RuntimeError("discovery must not execute")\n'
    )
    (engine / "scenario_types.py").write_text(
        'from .scenario_definitions.serving import DecodeScenario\n'
    )
    monkeypatch.setattr(API_BUILD, "DOCS", docs)
    assert API_BUILD.scenario_contracts() == {"DecodeScenario": ["latency"]}
    pages = API_BUILD.scenario_domain_pages()
    assert 'scenario-domain-serving.qmd' in pages['scenario-domains.qmd']
    serving = pages['scenario-domain-serving.qmd']
    assert 'Serving latency and memory accounting.' in serving
    assert 'scenario_definitions/serving.py' in serving
    assert '(engine.scenario_types.DecodeScenario.qmd)' in serving
    assert 'DecodeScenario' in API_BUILD.scenario_index()


def test_scenario_discovery_rejects_duplicate_class_ownership(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    definitions = tmp_path / "mlsysim" / "engine" / "scenario_definitions"
    definitions.mkdir(parents=True)
    for domain in ['serving', 'training']:
        (definitions / f'{domain}.py').write_text('class SharedScenario:\n    pass\n')
    monkeypatch.setattr(API_BUILD, "DOCS", docs)
    with pytest.raises(ValueError, match='SharedScenario.*more than one module'):
        API_BUILD.scenario_definitions()


def test_nested_scenario_definitions_keep_the_top_level_domain(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    definitions = tmp_path / "mlsysim" / "engine" / "scenario_definitions"
    robotics = definitions / "robotics"
    robotics.mkdir(parents=True)
    (robotics / "__init__.py").write_text('"""Robotic sensing and control."""\n')
    (robotics / "control.py").write_text(
        'class ControlScenario:\n    """A control calculation."""\n'
        '    @property\n    def latency(self):\n        raise RuntimeError("do not run")\n'
    )
    (robotics / "perception.py").write_text('class PerceptionScenario:\n    pass\n')
    (definitions / "base.py").write_text(
        'class Scenario:\n    """Generic scorecard has an existing separate reference."""\n'
    )
    monkeypatch.setattr(API_BUILD, "DOCS", docs)
    assert API_BUILD.scenario_contracts() == {
        "ControlScenario": ["latency"], "PerceptionScenario": []
    }
    pages = API_BUILD.scenario_domain_pages()
    assert 'scenario-domain-robotics.qmd' in pages
    assert 'scenario-domain-control.qmd' not in pages
    assert 'Robotic sensing and control.' in pages['scenario-domain-robotics.qmd']
    assert '(engine.scenario_types.ControlScenario.qmd)' in pages['scenario-domain-robotics.qmd']
    assert '### Control' in pages['scenario-domain-robotics.qmd']
    assert '### Perception' in pages['scenario-domain-robotics.qmd']
