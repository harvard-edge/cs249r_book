#!/usr/bin/env python3
"""Regenerate the MLSys·im API reference and fail if the public surface drifts.

Runs as a Quarto ``pre-render`` hook (see ``_quarto.yml``) and can be invoked
directly::

    python scripts/build_api.py          # regenerate api/ and check coverage
    python scripts/build_api.py --check  # coverage check only, no rebuild

Why it exists
-------------
The API pages under ``api/`` used to be committed snapshots that nothing
regenerated, so they drifted from the code (the physics page documented no
functions at all). Now ``api/`` is build output: every full render rebuilds it
from docstrings, and the coverage check below fails the build when an object
in the declared coverage scope has no page. To document a new symbol, add it to the ``quartodoc``
block in ``_quarto.yml`` (physics modules pick up new functions automatically).

The check covers:
  * every function exported by ``mlsysim.physics``
  * every concrete solver exported by ``mlsysim.solvers``
  * every class exported by the registry packages that define ``__all__``
  * every authored public calculation scenario class and its public members
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parent.parent
API = DOCS / "api"

# Pages that existed before the API was generated from code. Each now
# redirects to the API index so old links keep resolving.
LEGACY_PAGES = [
    "agents", "core", "datasets", "embodied", "engine", "hardware",
    "hardware.registry", "hardware.registry.Hardware", "hardware.types",
    "hardware.types.HardwareNode", "infrastructure", "literature", "models",
    "models.CNNWorkload", "models.registry", "models.registry.Models",
    "models.TransformerWorkload", "models.types", "models.Workload", "ops",
    "physics", "platforms", "solvers", "systems", "systems.Fleet",
    "systems.Node", "systems.registry", "systems.registry.Systems",
    "systems.types",
]

REGISTRY_PACKAGES = [
    "platforms", "datasets", "literature", "ops", "embodied", "sensors",
    "actuators", "agents",
]

# Exported names that are intentionally not given their own page.
SKIP = {
    "embodied.data_sheet_lookup", "embodied.machine_card",
    "embodied.machine_data_sheet",  # documented in the Embodied Zoo
    "platforms.PlatformEnvelope", "datasets.DatasetProfile",
    "sensors.CameraSensor", "sensors.LiDARSensor", "sensors.IMUSensor",
    "agents.ResearchWorkload", "agents.RemediationWorkload",
    "agents.SupportWorkload", "agents.AgentPlatforms",
    "agents.ReferencePlatforms",
}

INDEX_HEADER = """---
title: "API Reference"
description: "MLSys·im entry points, solvers, calculation scenarios, registries, and physics formulas, generated from package docstrings."
toc: false
aliases:
{aliases}
---

::: {{.callout-note appearance="simple"}}
For a first calculation, use [Choosing a Python API](../choosing-an-api.qmd).
For implementation navigation, follow the [code walkthrough](../code-walkthrough.qmd).

This reference is generated from package docstrings during a full documentation
build. It covers the entry points declared in the documentation configuration
and public calculation classes in `engine/scenario_definitions/`. To browse
registry entries (GPUs, models, or fleets), use [the Zoo](../zoo/index.qmd).
:::

"""


SCENARIO_MODULE = "engine.scenario_types"
SCENARIO_TITLE = "Calculation scenario types"
DOMAIN_TITLES = {
    "base": "Shared contracts",
    "training": "Training",
    "serving": "Serving",
    "communication": "Communication",
    "data": "Data",
    "reliability": "Reliability",
    "sustainability": "Sustainability",
    "economics": "Economics",
    "optimization": "Optimization",
    "agents": "Agents",
    "robotics": "Robotics",
    "statistics": "Statistics",
}


def scenario_definitions() -> dict[str, tuple[ast.ClassDef, Path]]:
    """Find class ownership in source, independently of compatibility metadata."""
    engine = DOCS.parent / "mlsysim" / "engine"
    directory = engine / "scenario_definitions"
    sources = sorted(directory.rglob("*.py")) if directory.is_dir() else [engine / "scenario_types.py"]
    definitions = {}
    for source in sources:
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            # The generic scorecard retains its separate engine.scenarios page.
            if node.name == "Scenario":
                continue
            if node.name in definitions:
                raise ValueError(f"Scenario class {node.name} is defined in more than one module")
            definitions[node.name] = (node, source)
    return definitions


def scenario_domain(source: Path) -> str:
    """Use the first defining-package component, including nested subject files."""
    directory = DOCS.parent / "mlsysim" / "engine" / "scenario_definitions"
    if source.is_relative_to(directory):
        return source.relative_to(directory).parts[0].removesuffix('.py')
    return "base"


def scenario_contracts() -> dict[str, list[str]]:
    """Discover authored scenario types and members without evaluating properties."""
    return {
        node.name: [
            member.name for member in node.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not member.name.startswith("_")
        ]
        for node, _ in scenario_definitions().values()
    }


def expanded_config() -> dict:
    """Expand the scenario module marker into one reference page per class."""
    config = yaml.safe_load((DOCS / "_quarto.yml").read_text(encoding="utf-8"))
    classes = sorted(scenario_contracts())
    # Quartodoc summarizes class attributes but does not render their bodies.
    # Add authored fields/properties explicitly to keep their anchors and notes.
    attributes = {}
    for node, _ in scenario_definitions().values():
        names = []
        for member in node.body:
            if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                names.append(member.target.id)
            elif isinstance(member, ast.FunctionDef) and any(
                isinstance(decorator, ast.Name) and decorator.id == "property"
                for decorator in member.decorator_list
            ):
                names.append(member.name)
        attributes[node.name] = [name for name in names if not name.startswith("_")]
    for section in config["quartodoc"]["sections"]:
        if section.get("contents") == [SCENARIO_MODULE]:
            section["contents"] = [
                {
                    "kind": "page",
                    "flatten": True,
                    "path": f"{SCENARIO_MODULE}.{name}",
                    "contents": [
                        {"name": f"{SCENARIO_MODULE}.{name}", "include_empty": True},
                        *[f"{SCENARIO_MODULE}.{name}.{attr}"
                          for attr in attributes.get(name, [])],
                    ],
                }
                for name in classes
            ]
    return config


def format_scenario_pages() -> None:
    """Keep class pages readable while preserving quartodoc's public anchors.

    Quartodoc renders attribute names and docstrings but omits their annotations
    and defaults. Show those source expressions without evaluating any property.
    """
    for node, source in scenario_definitions().values():
        page = API / f"{SCENARIO_MODULE}.{node.name}.qmd"
        body = page.read_text(encoding="utf-8")
        prefix = f"{SCENARIO_MODULE}.{node.name}"
        body = body.replace(f"# {prefix} {{", f"# {node.name} {{", 1)
        if "scenario_definitions" in source.parts:
            source_path = source.relative_to(DOCS.parent).as_posix()
            domain = scenario_domain(source)
            heading, rest = body.split("\n", 1)
            body = (f"{heading}\n\n"
                    f"[{DOMAIN_TITLES.get(domain, domain.title())} scenarios]"
                    f"(scenario-domain-{domain}.qmd) · "
                    f"Implementation: `{source_path}`.\n{rest}")
        # Static inspection cannot infer Pydantic keyword constructors.
        body = body.replace(
            f"```python\n{prefix}()\n```",
            "Construct this scenario with keyword inputs. The fields below show "
            "the types and defaults declared by this class.",
            1,
        )
        body = re.sub(
            rf"^(#+) {re.escape(prefix)}\.(\w+) ",
            lambda match: f"{'#' * max(2, len(match[1]))} {match[2]} ",
            body, flags=re.M,
        )
        for field in node.body:
            if not isinstance(field, ast.AnnAssign) or not isinstance(field.target, ast.Name):
                continue
            name = field.target.id
            if name.startswith("_"):
                continue
            annotation = ast.unparse(field.annotation)
            default = ast.unparse(field.value) if field.value is not None else "required"
            marker = f"`{prefix}.{name}`"
            body = body.replace(marker, f"**Type:** `{annotation}`.\n\n**Declared default:** `{default}`.", 1)
        page.write_text(body, encoding="utf-8")


def scenario_index() -> str:
    """Build a compact alphabetical directory; contracts remain on class pages."""
    definitions = scenario_definitions()
    lines = [
        '---', 'title: "Scenario Classes A–Z"', 'toc: true', '---', '',
        'Calculation scenarios bind named inputs to a reusable calculation. These',
        'pages document the classes in `mlsysim/engine/scenario_definitions/`,',
        'including their equations, exclusions, examples, and calculated properties.',
        'Select instances through `mlsysim.Scenarios`; the class name describes their',
        'shared contract, not a separate registry entry.', '',
        'Browse [scenarios by domain](scenario-domains.qmd) when you know the',
        'modeling question; use this alphabetical list when you know a class name.', '',
        'The generic [Scenario](engine.scenarios.Scenario.qmd) scorecard is a different',
        'entry point. See [Choosing a Python API](../choosing-an-api.qmd) before',
        'selecting one, and the [Scenarios Zoo](../zoo/scenarios.qmd) for registry use.', '',
        'This directory is generated from authored public classes. Inherited methods',
        'are documented on [BaseScenario](engine.scenario_types.BaseScenario.qmd);',
        'a class page shows the properties and methods that it defines itself.', '',
    ]
    letter = None
    for name in sorted(definitions):
        if name[0] != letter:
            letter = name[0]
            lines.extend(['', f'## {letter}', ''])
        summary = (ast.get_docstring(definitions[name][0]) or '').split('\n\n', 1)[0]
        summary = ' '.join(summary.split())
        lines.append(f'- [{name}]({SCENARIO_MODULE}.{name}.qmd): {summary}')
    return '\n'.join(lines) + '\n'


def scenario_domain_pages() -> dict[str, str]:
    """Generate subject routes from defining files, without a second class catalog."""
    groups: dict[str, list[tuple[ast.ClassDef, Path]]] = {}
    for node, source in scenario_definitions().values():
        domain = scenario_domain(source)
        groups.setdefault(domain, []).append((node, source))
    pages = {}
    overview = [
        '---', 'title: "Scenarios by Domain"', 'toc: false', '---', '',
        'Choose the subject of the calculation. Each domain page links the typed',
        'scenario contracts and identifies their implementation files.', '',
        'For a first calculation, start with [Choosing a Python API](../choosing-an-api.qmd).',
        'For a known class name, use [Scenario Classes A–Z](scenario-index.qmd).', '',
        'Scenario definitions describe inputs and calculations. Named configurations',
        'in `scenario_catalog/` select those inputs; result types in `result_types/`',
        'describe returned values. These packages use the same domain names.',
        'The domains follow modeling subjects across all textbook volumes.', '',
    ]
    order = [domain for domain in DOMAIN_TITLES if domain in groups]
    order.extend(sorted(set(groups) - set(order)))
    for domain in order:
        title = DOMAIN_TITLES.get(domain, domain.replace('_', ' ').title())
        members = sorted(groups[domain], key=lambda item: item[0].name)
        source = members[0][1]
        domain_root = DOCS.parent / "mlsysim" / "engine" / "scenario_definitions" / domain
        description_source = domain_root / "__init__.py" if domain_root.is_dir() else source
        module_doc = ast.get_docstring(ast.parse(description_source.read_text(encoding="utf-8"))) or title
        summary = ' '.join(module_doc.split('\n\n', 1)[0].split())
        filename = f"scenario-domain-{domain}.qmd"
        overview.append(f'- [{title}]({filename}): {summary}')
        page_title = "Shared Scenario Contracts" if domain == "base" else f"{title} Scenarios"
        nested = len({source for _, source in members}) > 1
        lines = ['---', f'title: "{page_title}"', f'toc: {str(nested).lower()}',
                 'toc-depth: 3', '---', '',
                 summary, '', '[All domains](scenario-domains.qmd) · '
                 '[Classes A–Z](scenario-index.qmd)', '', '## Source locations', '']
        engine = DOCS.parent / "mlsysim" / "engine"
        for package, role in [('scenario_definitions', 'Inputs and calculations'),
                              ('scenario_catalog', 'Named configurations'),
                              ('result_types', 'Result schemas')]:
            path = engine / package / domain
            if not path.is_dir():
                path = path.with_suffix('.py')
            if path.exists():
                suffix = '/' if path.is_dir() else ''
                lines.append(f'- {role}: `{path.relative_to(DOCS.parent).as_posix()}{suffix}`.')
        lines.extend(['', 'Public imports and existing class reference URLs remain stable.', '',
                      '## Scenario contracts', ''])
        subjects: dict[str, list[ast.ClassDef]] = {}
        for node, source in members:
            subjects.setdefault(source.stem if nested else '', []).append(node)
        for subject, nodes in sorted(subjects.items()):
            if subject:
                lines.extend(['', f"### {subject.replace('_', ' ').title()}", ''])
            for node in nodes:
                description = ' '.join((ast.get_docstring(node) or '').split('\n\n', 1)[0].split())
                lines.append(f'- [{node.name}]({SCENARIO_MODULE}.{node.name}.qmd): {description}')
        pages[filename] = '\n'.join(lines) + '\n'
    pages['scenario-domains.qmd'] = '\n'.join(overview) + '\n'
    return pages


def build() -> None:
    # Keep the temporary config beside _quarto.yml so relative paths resolve
    # identically. The checked-in marker is the single discovery opt-in.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", prefix=".api-", dir=DOCS) as config:
        yaml.safe_dump(expanded_config(), config, sort_keys=False)
        config.flush()
        subprocess.run(
            [sys.executable, "-m", "quartodoc", "build", "--config", config.name],
            cwd=DOCS, check=True,
        )
    index = API / "index.qmd"
    body = index.read_text(encoding="utf-8")
    body = re.sub(r"^# API Reference \{[^}]*\}\n+", "", body)
    # The calculation directory has its own route rather than adding hundreds
    # of rows to the API landing page. This section is last in the config.
    body = body.split(f"\n## {SCENARIO_TITLE}", 1)[0]
    body += ("\n## Calculation scenarios\n\n"
             "Browse [scenarios by domain](scenario-domains.qmd), or find a known "
             "class in [the alphabetical index](scenario-index.qmd). Each type "
             "has its own contract page.\n")
    aliases = "\n".join(f"  - {name}.html" for name in LEGACY_PAGES)
    index.write_text(INDEX_HEADER.format(aliases=aliases) + body, encoding="utf-8")
    (API / "scenario-index.qmd").write_text(scenario_index(), encoding="utf-8")
    for filename, text in scenario_domain_pages().items():
        (API / filename).write_text(text, encoding="utf-8")
    format_scenario_pages()


def documented_anchors() -> set[str]:
    anchors: set[str] = set()
    for page in API.glob("*.qmd"):
        anchors.update(re.findall(r"\{ #(mlsysim\.[\w.]+) \}", page.read_text(encoding="utf-8")))
    return anchors


def short(anchors: set[str]) -> set[str]:
    """Final dotted component, e.g. mlsysim.physics.memory.calc_x -> calc_x."""
    return {a.rsplit(".", 1)[-1] for a in anchors}


def check() -> list[str]:
    sys.path.insert(0, str(DOCS.parent))
    anchors = documented_anchors()
    names = short(anchors)
    missing: list[str] = []

    physics = importlib.import_module("mlsysim.physics")
    for name in physics.__all__:
        obj = getattr(physics, name)
        # Aliases such as calc_training_time = dTime are documented under the
        # name of the function object they point to.
        if inspect.isfunction(obj) and obj.__name__ not in names:
            missing.append(f"mlsysim.physics.{name}  (defined in {obj.__module__})")

    solvers = importlib.import_module("mlsysim.solvers")
    for name in solvers.__all__:
        if name.startswith("Base"):
            continue
        if name not in names:
            missing.append(f"mlsysim.solvers.{name}")

    # Use fully qualified anchors: a result property named "latency" on an
    # unrelated class must not satisfy a missing scenario contract.
    for name, members in scenario_contracts().items():
        qualified = f"mlsysim.{SCENARIO_MODULE}.{name}"
        for anchor in [qualified, *(f"{qualified}.{member}" for member in members)]:
            if anchor not in anchors:
                missing.append(anchor)

    for pkg in REGISTRY_PACKAGES:
        mod = importlib.import_module(f"mlsysim.{pkg}")
        for name in getattr(mod, "__all__", []):
            if f"{pkg}.{name}" in SKIP:
                continue
            if name not in names:
                missing.append(f"mlsysim.{pkg}.{name}")
    return missing


def stamp_version() -> str:
    """Write config/sidebar-brand.html with the installed package version."""
    sys.path.insert(0, str(DOCS.parent))
    version = importlib.import_module("mlsysim").__version__
    template = (DOCS / "config" / "sidebar-brand.html.in").read_text(encoding="utf-8")
    (DOCS / "config" / "sidebar-brand.html").write_text(
        template.replace("__MLSYSIM_VERSION__", version), encoding="utf-8"
    )
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="check coverage only")
    args = parser.parse_args()

    # The sidebar include must exist for every render, including single pages.
    stamp_version()

    # As a pre-render hook, only rebuild the API on full-project renders so
    # that single-page previews stay fast.
    in_quarto = "QUARTO_PROJECT_RENDER_ALL" in os.environ
    if in_quarto and os.environ.get("QUARTO_PROJECT_RENDER_ALL") != "1":
        return 0

    if not args.check:
        build()

    missing = check()
    if missing:
        print("API coverage check failed. These public symbols have no API page:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("Add them to the quartodoc block in mlsysim/docs/_quarto.yml.", file=sys.stderr)
        return 1
    print(f"API coverage OK: {len(documented_anchors())} documented objects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
