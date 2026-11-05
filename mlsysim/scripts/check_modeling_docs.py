"""Validate discovered calculation contracts and execute their examples.

Discovery follows implementation definitions, not a hand-maintained allowlist.
These checks establish documentation structure and executable examples; a source
URL, derivation paragraph, or passing example does not prove scientific validity.
"""

from __future__ import annotations

import ast
import doctest
import importlib
import inspect
import io
from pathlib import Path
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from numpydoc.docscrape import NumpyDocString
from numpydoc.validate import validate

ROOT = Path(__file__).resolve().parents[1]


def _public_function(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")


def definition_sources(root, legacy_file, domain_package):
    """Yield source files from a compatibility module and its domain package.

    Read physical definitions instead of runtime ``__module__``: relocated classes
    retain their historical module identity for pickle and Pydantic compatibility.
    Import-only facades do not create duplicate definitions. Include package
    initializers and base modules so moving a contract never removes its coverage.
    """
    legacy = root / legacy_file
    if legacy.is_file():
        yield legacy
    package = root / domain_package
    if package.is_dir():
        yield from sorted(package.rglob("*.py"))


def discover_contracts(root=ROOT):
    """Find authored physics functions and concrete numerical solver methods.

    Private implementation modules and abstract interface methods are not public
    calculations. Imported aliases are checked at their defining implementation.
    New public definitions in these trees are included without a configuration edit.
    """
    contracts = []
    for relative in ("mlsysim/physics", "mlsysim/engine/solvers"):
        for path in sorted((root / relative).rglob("*.py")):
            if path.name.startswith("_") or path.name == "base.py":
                continue
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            for node in ast.parse(path.read_text()).body:
                if _public_function(node):
                    contracts.append(f"{module}.{node.name}")
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    for method in node.body:
                        if _public_function(method):
                            contracts.append(f"{module}.{node.name}.{method.name}")
    for relative, classes in (
        ("mlsysim/engine/engine.py", {"Engine", "PerformanceProfile"}),
        ("mlsysim/models/types.py", None),
        ("mlsysim/hardware/types.py", None),
        ("mlsysim/systems/types.py", None),
        ("mlsysim/infrastructure/types.py", None),
        ("mlsysim/datasets/types.py", None),
        ("mlsysim/agents/types.py", None),
        ("mlsysim/embodied/types.py", None),
        ("mlsysim/actuators/types.py", None),
        ("mlsysim/hardware/tech.py", {"ReticleField", "BatteryCell"}),
        ("mlsysim/systems/reliability.py", {"RecoveryProfile"}),
        ("mlsysim/engine/serving_profile.py", None),
        ("mlsysim/engine/constraint_explorer.py", None),
        ("mlsysim/engine/empirical.py", {"EmpiricalAnchor"}),
    ):
        path = root / relative
        if not path.exists():
            continue
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            if _public_function(node) and relative in (
                "mlsysim/engine/serving_profile.py",
                "mlsysim/engine/constraint_explorer.py",
            ):
                contracts.append(f"{module}.{node.name}")
            if not isinstance(node, ast.ClassDef) or (classes is not None and node.name not in classes):
                continue
            for method in node.body:
                if _public_function(method) and method.name not in {"summary", "provenance_ids"}:
                    contracts.append(f"{module}.{node.name}.{method.name}")
    for path in definition_sources(root, "mlsysim/engine/results.py", "mlsysim/engine/result_types"):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                for method in node.body:
                    if _public_function(method) and method.name not in {"summary", "provenance_ids"}:
                        contracts.append(f"{module}.{node.name}.{method.name}")
    return sorted(set(contracts))


def discover_scenarios(root=ROOT):
    """Find authored scenario contracts, following nested physical source modules."""
    scenarios = {}
    for path in definition_sources(root, "mlsysim/engine/scenario_types.py", "mlsysim/engine/scenario_definitions"):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                # Scenario composes Engine evaluation; its ordinary API contract
                # remains checked separately rather than inventing a class equation.
                if module == "mlsysim.engine.scenario_definitions.base" and node.name == "Scenario":
                    continue
                scenarios[f"{module}.{node.name}"] = [method.name for method in node.body if _public_function(method)]
    return scenarios


def discover_api_contracts(root=ROOT):
    """Identify composition and conversion APIs that do not introduce a model.

    These entry points need precise parameters, results, and examples, but not
    artificial equations or citations for dispatch, validation, or unit conversion.
    """
    selections = {
        "mlsysim/engine/scenarios.py": {"Scenario.is_distributed", "Scenario.evaluate", "Scenario.validate"},
        "mlsysim/engine/evaluation.py": {"SystemEvaluator.evaluate", "SystemEvaluation.passed_all"},
        "mlsysim/engine/explainers.py": {"DifferentialExplainer.compare_performance"},
        "mlsysim/engine/dse.py": {"DSE.search"},
        "mlsysim/engine/pipeline.py": {"Pipeline.run"},
        "mlsysim/engine/config.py": {"SimulationConfig.validate_physical_feasibility"},
        "mlsysim/core/units.py": {"normalize_precision", "resolve_precision"},
        "mlsysim/core/provenance.py": {"Sourced.to", "scalar_value", "fleet_mttf_hours"},
        "mlsysim/hardware/tech.py": {"resolve_latency"},
    }
    scenario_base = root / "mlsysim/engine/scenario_definitions/base.py"
    if scenario_base.exists() and any(
        isinstance(node, ast.ClassDef) and node.name == "Scenario" for node in ast.parse(scenario_base.read_text()).body
    ):
        selections["mlsysim/engine/scenario_definitions/base.py"] = selections.pop("mlsysim/engine/scenarios.py")
    result = []
    for relative, selected in selections.items():
        path = root / relative
        if path.exists():
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            result.extend(f"{module}.{name}" for name in selected)
    return sorted(result)


def configuration():
    """Read checker options and discover the complete calculation scope."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)["tool"]["mlsysim"]["documentation"]
    config["objects"] = discover_contracts()
    config["scenarios"] = discover_scenarios()
    config["api_objects"] = discover_api_contracts()
    return config


def resolve(name):
    """Resolve a qualified public function, method, property, or scenario class."""
    parts = name.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError as exc:
            if exc.name != ".".join(parts[:split]):
                raise
            continue
        for part in parts[split:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(name)


def check_references(references, keys=None):
    """Require a source or a stated derivation without inventing an attribution.

    Elementary accounting may explain its own derivation. Literature-derived
    methods link a source; supplied bibliography keys must resolve when the
    repository bibliography is available. Semantic source support remains review work.
    """
    errors = []
    cited_keys = re.findall(r"bibliography key: ([\w-]+)", references)
    derivation = re.search(r"Derivation:\s*(.+)", references, re.DOTALL)
    implementation = []
    if re.search(r"Implementation references?:", references):
        implementation = re.findall(r"``(mlsysim[\w.]+)``", references)
    if not re.search(r"https?://\S+", references) and not derivation and not implementation:
        errors.append(
            "MODEL02: References must include a source URL, concrete Derivation:, or qualified Implementation reference:"
        )
    for target in implementation:
        try:
            resolve(target)
        except (ImportError, AttributeError):
            errors.append(f"MODEL07: Unknown implementation reference: {target}")
    if derivation and len(derivation.group(1).split()) < 6:
        errors.append("MODEL03: Derivation must explain the accounting or algebra, not just label it")
    if keys is not None:
        for key in cited_keys:
            if key not in keys:
                errors.append(f"MODEL04: Unknown bibliography key: {key}")
    return errors


def check_math_markup(doc):
    """Reject multiplication tokens that Markdown could render as emphasis.

    Inline code and math are already protected; doctest code is never rewritten.
    This check is about displaying the written equation, not its mathematical truth.
    """
    protected = re.compile(r"(``[^`]*``|`[^`]*`|\$[^$]*\$)")
    product = re.compile(r"[A-Za-z0-9_()./+^\-]+(?:\*{1,2}[A-Za-z0-9_()./+^\-]+)+")
    # Mask complete inline spans before splitting lines: NumPy docstrings often
    # wrap a single inline literal across source lines.
    masked = protected.sub(lambda match: "".join("\n" if c == "\n" else " " for c in match.group()), doc)
    lines = masked.splitlines()
    active = False
    for i, line in enumerate(lines):
        content = line.strip()
        if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            active = content in {"Notes", "References"}
        if "Equation:" in line or "Derivation:" in line:
            active = True
        if active and not content.startswith((">>>", "...", "http", "..")):
            if product.search(line):
                return ["MODEL08: Put multiplication expressions in inline code or math so Markdown preserves them"]
    return []


def check_examples(obj):
    """Run only this object's examples, without duplicating nested methods."""
    if isinstance(obj, property):
        obj = obj.fget
    runner = doctest.DocTestRunner()
    output = io.StringIO()
    namespace = vars(importlib.import_module(obj.__module__))
    for example in doctest.DocTestFinder(recurse=False).find(obj, globs=namespace):
        runner.run(example, out=output.write)
    errors = []
    if not runner.tries:
        errors.append("MODEL05: Examples must contain an executable doctest")
    if runner.failures:
        errors.append("MODEL06: Example failed\n" + output.getvalue())
    return errors


def check_object(name, ignored=(), bibliography_keys=None):
    """Return structural, source-label, and example errors for one calculation."""
    obj = resolve(name)
    errors = [f"{code}: {message}" for code, message in validate(name)["errors"] if code not in ignored]
    doc = inspect.getdoc(obj) or ""
    errors += check_math_markup(doc)
    parsed = NumpyDocString(doc)
    notes = "\n".join(parsed["Notes"])
    for label in ("Equation:", "Assumptions:", "Excludes:"):
        if label not in notes:
            errors.append(f"MODEL01: Notes must include {label}")
    errors += check_references("\n".join(parsed["References"]), bibliography_keys)
    errors += check_examples(obj)
    return errors


def check_api_object(name, ignored=()):
    """Check ordinary API documentation without implying a scientific equation."""
    errors = [f"{code}: {message}" for code, message in validate(name)["errors"] if code not in ignored]
    return errors + check_examples(resolve(name))


def check_scenario(name, methods, bibliography_keys=None):
    """Check shared scenario context and concise property-level calculation traces.

    A scenario class owns its assumptions, source/derivation, and runnable example.
    Properties state their meaning and equation or delegate. Repeating a full
    scientific contract on each simple alias would obscure that shared context.
    """
    obj = resolve(name)
    doc = inspect.getdoc(obj) or ""
    errors = check_math_markup(doc)
    for label in ("Equation:", "Assumptions:", "Excludes:"):
        if label not in doc:
            errors.append(f"SCENARIO01: Class contract must include {label}")
    parsed = NumpyDocString(doc)
    errors += check_references("\n".join(parsed["References"]), bibliography_keys)
    errors += check_examples(obj)
    for method in methods:
        member = getattr(obj, method)
        if isinstance(member, property):
            member = member.fget
        member_doc = inspect.getdoc(member) or ""
        errors += [f"{method}: {error}" for error in check_math_markup(member_doc)]
        for target in re.findall(r"``(mlsysim[\w.]+)``", member_doc):
            try:
                resolve(target)
            except (ImportError, AttributeError):
                errors.append(f"SCENARIO04: {method} links unknown implementation: {target}")
        if not member_doc.strip():
            errors.append(f"SCENARIO02: {method} must explain its meaning and units")
        if not any(
            label in member_doc for label in ("Equation:", "See Also", "Delegates to", "Delegates:", "Calculation:")
        ):
            errors.append(f"SCENARIO03: {method} must state its equation or underlying contract")
    return errors


def bibliography_keys():
    """Read paper citation keys when running in the repository checkout."""
    path = ROOT / "paper" / "references.bib"
    if not path.exists():
        if (ROOT.parent / ".git").exists():
            raise FileNotFoundError(f"Repository bibliography is missing: {path}")
        return None
    return set(re.findall(r"@\w+\{([^,]+),", path.read_text()))


def main():
    """Run discovered contract checks, returning nonzero for any failure."""
    config = configuration()
    if not config["files"] or not config["objects"] or not config["scenarios"]:
        raise ValueError("Documentation scope must not be empty")
    failed = False
    for arguments in (["check", "--select", "D"], ["format", "--check"]):
        result = subprocess.run([sys.executable, "-m", "ruff", *arguments, *config["files"]], cwd=ROOT)
        failed |= result.returncode != 0
    keys = bibliography_keys()
    if keys is None:
        print("Bibliography cross-check unavailable outside the repository; source labels still checked.")
    targets = [
        (name, lambda name=name: check_object(name, config["numpydoc_ignore"], keys)) for name in config["objects"]
    ]
    targets += [
        (name, lambda name=name: check_api_object(name, config["numpydoc_ignore"])) for name in config["api_objects"]
    ]
    targets += [
        (name, lambda name=name, methods=methods: check_scenario(name, methods, keys))
        for name, methods in config["scenarios"].items()
    ]
    for name, check in targets:
        try:
            errors = check()
        except Exception as exc:
            errors = [f"Cannot validate target: {exc}"]
        for error in errors:
            print(f"{name}: {error}")
        failed |= bool(errors)
    if not failed:
        print(
            f"Modeling documentation checks passed for {len(config['objects'])} calculation contracts, "
            f"{len(config['scenarios'])} scenario classes, and {len(config['api_objects'])} composition/conversion APIs."
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
