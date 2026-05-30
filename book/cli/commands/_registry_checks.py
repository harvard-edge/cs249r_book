"""Registry migration gates for `binder check registry`."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegistryIssue:
    code: str
    message: str
    file: str = "(registry)"
    line: int = 0
    severity: str = "error"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def yaml_dir(root: Path) -> Path:
    return root / "book" / "tools" / "audits" / "mlsysim_constants"


def count_should_change(root: Path) -> int:
    import yaml

    total = 0
    for path in yaml_dir(root).glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for cell in data.get("python_cells") or []:
            for entry in cell.get("constants") or []:
                if entry.get("should_change") is True:
                    total += 1
    return total


def _load_script_module(name: str, script: Path):
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_registry_sources(root: Path, paths: list[Path] | None = None) -> list[RegistryIssue]:
    """Scan QMD LEGO cells for banned legacy constant/registry patterns."""
    mod = _load_script_module("book_check_registry_sources", root / "book" / "tools" / "audit" / "book_check_registry_sources.py")

    if paths is None:
        paths = sorted((root / "book" / "quarto" / "contents").rglob("*.qmd"))

    issues: list[RegistryIssue] = []
    for path in paths:
        p = path if path.is_absolute() else root / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        for msg in mod.check_file(p):
            issues.append(RegistryIssue(
                code="registry_source",
                message=msg,
                file=str(p.relative_to(root)),
            ))
    return issues


def check_lego_prose_literals(root: Path, paths: list[Path] | None = None) -> list[RegistryIssue]:
    """Flag hardcoded numeric literals in LEGO walkthrough prose."""
    mod = _load_script_module(
        "book_check_lego_prose_literals",
        root / "book" / "tools" / "audit" / "book_check_lego_prose_literals.py",
    )

    if paths is None:
        paths = sorted((root / "book" / "quarto" / "contents").rglob("*.qmd"))

    issues: list[RegistryIssue] = []
    for path in paths:
        p = path if path.is_absolute() else root / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        for lineno, snippet, labels in mod.check_file(p):
            uniq = ", ".join(dict.fromkeys(labels))
            issues.append(RegistryIssue(
                code="lego_prose_literal",
                message=f"L{lineno}: {uniq} — {snippet}",
                file=str(p.relative_to(root)),
                line=lineno,
            ))
    return issues


def _load_check_module(name: str, root: Path):
    return _load_script_module(name, root / "book" / "tools" / "audit" / f"{name}.py")


def check_lego_prose_units(root: Path, paths: list[Path] | None = None) -> list[RegistryIssue]:
    """Flag unit/currency tokens immediately after {python} *_str refs."""
    mod = _load_check_module("book_check_lego_prose_units", root)
    if paths is None:
        paths = sorted((root / "book" / "quarto" / "contents").rglob("*.qmd"))
    issues: list[RegistryIssue] = []
    for path in paths:
        p = path if path.is_absolute() else root / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        for lineno, snippet, labels in mod.check_file(p):
            uniq = ", ".join(dict.fromkeys(labels))
            issues.append(RegistryIssue(
                code="lego_prose_unit",
                message=f"L{lineno}: {uniq} — {snippet}",
                file=str(p.relative_to(root)),
                line=lineno,
            ))
    return issues


def check_lego_load_pint(root: Path, paths: list[Path] | None = None) -> list[RegistryIssue]:
    """Static lint: physical *_value must use ureg/registry."""
    mod = _load_check_module("book_check_lego_load_pint", root)
    if paths is None:
        paths = sorted((root / "book" / "quarto" / "contents").rglob("*.qmd"))
    issues: list[RegistryIssue] = []
    for path in paths:
        p = path if path.is_absolute() else root / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        for lineno, snippet, labels in mod.check_file(p):
            uniq = ", ".join(dict.fromkeys(labels))
            issues.append(RegistryIssue(
                code="lego_load_pint",
                message=f"L{lineno}: {uniq} — {snippet}",
                file=str(p.relative_to(root)),
                line=lineno,
            ))
    return issues


def check_lego_equations(root: Path, paths: list[Path] | None = None) -> list[RegistryIssue]:
    """Verify A/B=C prose lines numerically."""
    mod = _load_check_module("book_check_lego_equations", root)
    if paths is None:
        paths = sorted((root / "book" / "quarto" / "contents").rglob("*.qmd"))
    issues: list[RegistryIssue] = []
    for path in paths:
        p = path if path.is_absolute() else root / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        for lineno, snippet, labels in mod.check_file(p):
            uniq = ", ".join(dict.fromkeys(labels))
            issues.append(RegistryIssue(
                code="lego_equation",
                message=f"L{lineno}: {uniq} — {snippet}",
                file=str(p.relative_to(root)),
                line=lineno,
            ))
    return issues


def run_registry_pytest(root: Path) -> list[RegistryIssue]:
    """Run mlsysim registry gate tests."""
    tests = [
        "tests/test_constants_allowlist.py",
        "tests/test_no_legacy_constant_refs.py",
        "tests/test_appendix_constants.py",
        "tests/test_registry_no_duplicate_specs.py",
        "tests/test_provenance_audit.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q"],
        cwd=root / "mlsysim",
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return []
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    summary = tail[-1] if tail else f"pytest exited {proc.returncode}"
    return [RegistryIssue(code="registry_pytest", message=summary)]


def verify_appendix_lego(root: Path) -> list[RegistryIssue]:
    """Verify appendix LEGO cells match registry specs."""
    mod = _load_script_module(
        "generate_appendix_constants",
        root / "book" / "tools" / "audit" / "generate_appendix_constants.py",
    )

    issues: list[RegistryIssue] = []
    spec_errors = mod.verify_interconnect_spec()
    for err in spec_errors:
        issues.append(RegistryIssue(code="appendix_interconnect", message=err))

    results = mod.verify_all()
    failed = [r for r in results if not r.ok]
    for r in failed:
        issues.append(RegistryIssue(
            code="appendix_lego",
            message=f"{r.class_name or r.label or r.path.name}: {r.error}",
            file=str(r.path.relative_to(root)) if r.path.is_relative_to(root) else str(r.path),
        ))
    return issues


def verify_paper_anchors(root: Path) -> list[RegistryIssue]:
    """Validate paper anchor consistency."""
    script = root / "mlsysim" / "paper" / "scripts" / "validate_anchors.py"
    if not script.exists():
        return [RegistryIssue(code="anchors_missing", message=f"Script not found: {script}")]

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root / "mlsysim",
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return []
    output = (proc.stdout or proc.stderr or "").strip()
    return [RegistryIssue(code="paper_anchors", message=output[:500] or "anchor validation failed")]


def check_yaml_pending(root: Path) -> list[RegistryIssue]:
    """Fail when audit YAML still has should_change=true entries."""
    pending = count_should_change(root)
    if pending == 0:
        return []
    return [RegistryIssue(
        code="yaml_should_change",
        message=(
            f"{pending} audit YAML constant(s) still marked should_change=true — "
            "run: python3 book/tools/audit/refresh_mlsysim_constants_yamls.py --finalize"
        ),
    )]
