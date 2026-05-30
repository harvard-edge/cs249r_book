#!/usr/bin/env python3
"""Verify and regenerate appendix assumption-table LEGO cells from registries.

Appendix ``*Constants`` classes in vol1/vol2 ``appendix_assumptions.qmd`` should
source hardware/model/fabric values from typed registries (``Hardware.*``,
``Models.*``, ``Systems.*``, ``Literature.*``, ``Infrastructure.*``) or physics-only ``constants``.

Usage:
    python3 generate_appendix_constants.py --verify
    python3 generate_appendix_constants.py --write interconnect
    python3 generate_appendix_constants.py --refresh-yaml
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MLSYSIM_ROOT = REPO_ROOT / "mlsysim"
APPENDIX_PATHS = [
    REPO_ROOT / "book/quarto/contents/vol1/backmatter/appendix_assumptions.qmd",
    REPO_ROOT / "book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd",
]

CELL_START = re.compile(r"^```\{python\}\s*$")
CELL_END = re.compile(r"^```\s*$")
LABEL_RE = re.compile(r"^#\|\s*label:\s*(\S+)")
CLASS_RE = re.compile(r"^class\s+(\w+)", re.M)
GENERATED_MARKER = "# │ AUTO-GENERATED body — edit appendix_registry_specs.py, then --write"

# Simple interconnect table: registry/constants → fmt_val/fmt_unit pairs.
INTERCONNECT_FIELDS: list[tuple[str, str]] = [
    ("NVLINK_V100_BW", "Hardware.Cloud.V100.nvlink.bandwidth"),
    ("NVLINK_A100_BW", "Hardware.Cloud.A100.nvlink.bandwidth"),
    ("NVLINK_H100_BW", "Hardware.Cloud.H100.nvlink.bandwidth"),
    ("INFINIBAND_HDR_BW", "Systems.Fabrics.InfiniBand_HDR.bandwidth"),
    ("INFINIBAND_NDR_BW", "Systems.Fabrics.InfiniBand_NDR.bandwidth"),
    ("INFINIBAND_XDR_BW", "Systems.Fabrics.InfiniBand_XDR.bandwidth"),
    ("INFINIBAND_GXDR_BW", "Systems.Fabrics.InfiniBand_GXDR.bandwidth"),
    ("PCIE_GEN4_BW", "Hardware.Cloud.A100.interconnect.bandwidth"),
    ("PCIE_GEN5_BW", "Hardware.Cloud.H100.interconnect.bandwidth"),
    ("NVME_SEQUENTIAL_BW", "Hardware.Tech.Storage.NvmeGen4.bandwidth"),
    ("NETWORK_10G_BW", "Systems.Fabrics.Ethernet_10G.bandwidth"),
    ("NETWORK_100G_BW", "Systems.Fabrics.Ethernet_100G.bandwidth"),
    ("SPEED_OF_LIGHT_FIBER_KM_S", "SPEED_OF_LIGHT_FIBER_KM_S"),
]

@dataclass
class PythonCell:
    path: Path
    start_line: int
    end_line: int
    label: str | None
    class_name: str | None
    source: str
    body: str  # code inside fences, excluding fence lines

@dataclass
class VerifyResult:
    path: Path
    label: str | None
    class_name: str | None
    ok: bool
    error: str | None = None
    registry_sources: list[str] = field(default_factory=list)

def _extract_cells(path: Path) -> list[PythonCell]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[PythonCell] = []
    i = 0
    while i < len(lines):
        if not CELL_START.match(lines[i]):
            i += 1
            continue
        start = i
        i += 1
        label: str | None = None
        while i < len(lines) and not CELL_END.match(lines[i]):
            m = LABEL_RE.match(lines[i].strip())
            if m:
                label = m.group(1)
            i += 1
        if i >= len(lines):
            break
        block = "\n".join(lines[start : i + 1])
        body = "\n".join(lines[start + 1 : i])
        cls = CLASS_RE.search(body)
        cells.append(
            PythonCell(
                path=path,
                start_line=start + 1,
                end_line=i + 1,
                label=label,
                class_name=cls.group(1) if cls else None,
                source=block,
                body=body,
            )
        )
        i += 1
    return cells

def _exec_preamble() -> str:
    return textwrap.dedent(
        """
        from mlsysim import *
        from mlsysim.core.constants import *
        from mlsysim.core import constants
        from mlsysim.physics import SPEED_OF_LIGHT_FIBER_KM_S
        from mlsysim.fmt import fmt, fmt_val, fmt_unit, fmt_int, MarkdownStr, check, sci_latex, fmt_math
        """
    ).strip()

def _registry_sources_in_body(body: str) -> list[str]:
    patterns = [
        r"Hardware\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"Models\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"Systems\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"Datasets\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"Infrastructure\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"Literature\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"Ops\.[\w.]+(?:\[[^\]]+\])?(?:\.[\w]+)*",
        r"constants\.[A-Z][A-Z0-9_]+",
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, body))
    return sorted(dict.fromkeys(found))

def verify_cell(cell: PythonCell) -> VerifyResult:
    code = _exec_preamble() + "\n\n" + cell.body
    namespace: dict = {}
    try:
        exec(compile(code, f"{cell.path.name}:{cell.start_line}", "exec"), namespace)
        ok = True
        err = None
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"
    return VerifyResult(
        path=cell.path,
        label=cell.label,
        class_name=cell.class_name,
        ok=ok,
        error=err,
        registry_sources=_registry_sources_in_body(cell.body),
    )

def verify_all() -> list[VerifyResult]:
    if str(MLSYSIM_ROOT) not in sys.path:
        sys.path.insert(0, str(MLSYSIM_ROOT))
    results: list[VerifyResult] = []
    for path in APPENDIX_PATHS:
        for cell in _extract_cells(path):
            if cell.class_name is None:
                continue
            results.append(verify_cell(cell))
    return results

def _render_interconnect_class() -> str:
    lines = [
        "class InterconnectConstants:",
        '    """Formatted constants for Interconnect and Network Bandwidth."""',
        "",
        "    # ┌── 4. OUTPUT (Formatting) ───────────────────────────",
        f"    {GENERATED_MARKER}",
    ]
    for name, source in INTERCONNECT_FIELDS:
        val_pad = max(1, 36 - len(name))
        unit_pad = max(1, 35 - len(name))
        lines.append(f"    {name}_val_str{' ' * val_pad}= fmt_val({source})")
        lines.append(f"    {name}_unit_str{' ' * unit_pad}= fmt_unit({source})")
    return "\n".join(lines)

def _render_interconnect_cell() -> str:
    body = _render_interconnect_class()
    return (
        "```{python}\n"
        "#| echo: false\n"
        "from mlsysim import *\n"
        "from mlsysim.core.constants import *\n"
        "#| label: appendix-interconnectconstants\n"
        "# ┌── LEGO ───────────────────────────────────────────────\n"
        "# │ Context: ## Interconnect and Network Bandwidth {.unnumbered}\n"
        "# │ Goal: Formatted value/unit pairs for reference tables in this section\n"
        "# │ Exports: InterconnectConstants.*_val_str, InterconnectConstants.*_unit_str\n"
        "# │ Source:  book/tools/audit/generate_appendix_constants.py (--write interconnect)\n"
        "from mlsysim.core import constants\n"
        "from mlsysim.fmt import fmt, fmt_val, fmt_unit\n"
        "\n"
        f"{body}\n"
        "```"
    )

def _replace_cell_by_label(text: str, label: str, new_cell: str) -> str | None:
    """Replace a single ```{python} cell identified by ``#| label: <label>``."""
    label_pat = re.compile(rf"^#\|\s*label:\s*{re.escape(label)}\s*$", re.MULTILINE)
    match = label_pat.search(text)
    if not match:
        return None

    start = text.rfind("```{python}", 0, match.start())
    if start == -1:
        return None
    end = text.find("\n```", match.start())
    if end == -1:
        return None
    end += len("\n```")
    return text[:start] + new_cell + text[end:]

def write_interconnect() -> bool:
    path = APPENDIX_PATHS[0]
    text = path.read_text(encoding="utf-8")
    new_cell = _render_interconnect_cell()
    updated = _replace_cell_by_label(text, "appendix-interconnectconstants", new_cell)
    if updated is None:
        print(
            f"ERROR: appendix-interconnectconstants cell not found in {path}",
            file=sys.stderr,
        )
        return False
    if updated == text:
        print("InterconnectConstants cell unchanged")
        return False
    path.write_text(updated, encoding="utf-8")
    print(f"Wrote {path.name}: appendix-interconnectconstants")
    return True

def _resolve_source(expr: str):
    if str(MLSYSIM_ROOT) not in sys.path:
        sys.path.insert(0, str(MLSYSIM_ROOT))
    namespace: dict = {}
    exec(_exec_preamble(), namespace)
    return eval(expr, namespace)

def verify_interconnect_spec() -> list[str]:
    """Ensure generated interconnect fields resolve to quantities."""
    errors: list[str] = []
    for name, source in INTERCONNECT_FIELDS:
        try:
            _resolve_source(source)
        except Exception as exc:
            errors.append(f"{name}: {source} -> {exc}")
    return errors

def refresh_yaml() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "refresh_mlsysim_constants_yamls",
        Path(__file__).parent / "refresh_mlsysim_constants_yamls.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()

def _check_ast_no_legacy_imports(cell: PythonCell) -> list[str]:
    """Flag ``from mlsysim.core.constants import H100_*`` style imports in table cells."""
    issues: list[str] = []
    try:
        tree = ast.parse(cell.body)
    except SyntaxError:
        return issues
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "mlsysim.core.constants":
            for alias in node.names:
                if alias.name not in ("Q_", "*") and alias.name.isupper():
                    issues.append(f"legacy import: {alias.name}")
    return issues

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Execute all appendix LEGO cells against live mlsysim",
    )
    parser.add_argument(
        "--write",
        choices=["interconnect"],
        help="Regenerate a spec-driven appendix cell",
    )
    parser.add_argument(
        "--refresh-yaml",
        action="store_true",
        help="Refresh book/tools/audits/mlsysim_constants/*.yaml target_source fields",
    )
    args = parser.parse_args(argv)

    if not any([args.verify, args.write, args.refresh_yaml]):
        parser.print_help()
        return 2

    exit_code = 0

    if args.write == "interconnect":
        spec_errors = verify_interconnect_spec()
        if spec_errors:
            for err in spec_errors:
                print(f"SPEC ERROR: {err}", file=sys.stderr)
            return 1
        if not write_interconnect():
            return 1

    if args.verify:
        spec_errors = verify_interconnect_spec()
        for err in spec_errors:
            print(f"SPEC ERROR: {err}", file=sys.stderr)
            exit_code = 1

        results = verify_all()
        failed = [r for r in results if not r.ok]
        print(f"Appendix LEGO verify: {len(results) - len(failed)}/{len(results)} cells OK")
        for r in results:
            if r.ok:
                src = ", ".join(r.registry_sources[:3])
                if len(r.registry_sources) > 3:
                    src += f", +{len(r.registry_sources) - 3} more"
                print(f"  OK  {r.path.name}:{r.label or r.class_name} [{src}]")
            else:
                print(
                    f"  FAIL {r.path.name}:{r.label or r.class_name}: {r.error}",
                    file=sys.stderr,
                )

        for path in APPENDIX_PATHS:
            for cell in _extract_cells(path):
                if not cell.class_name or not cell.class_name.endswith("Constants"):
                    continue
                legacy = _check_ast_no_legacy_imports(cell)
                for issue in legacy:
                    print(
                        f"  WARN {path.name}:{cell.label}: {issue}",
                        file=sys.stderr,
                    )

        if failed:
            exit_code = 1

    if args.refresh_yaml:
        refresh_yaml()

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
