"""CI gate: website docs must use canonical registry paths and valid top-level imports."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MLSYSIM_ROOT = REPO_ROOT / "mlsysim"

SCAN_ROOTS = (
    MLSYSIM_ROOT / "docs",
    MLSYSIM_ROOT / "tutorial",
    MLSYSIM_ROOT / "tutorial" / "slides",
)

SKIP_FILES = {
    "DATA_MODEL.md",  # intentional anti-pattern examples
    "provenance.qmd",  # documents removed alias patterns
}

SCAN_SUFFIXES = {".qmd", ".md", ".tex"}

# Not exported from mlsysim.__init__ — docs must use submodule imports.
NON_TOP_LEVEL_SYMBOLS = frozenset({
    "TransformerWorkload",
    "CNNWorkload",
    "SparseTransformerWorkload",
    "SSMWorkload",
    "DiffusionWorkload",
    "MoERoutingModel",
    "CompressionModel",
    "ParallelismOptimizer",
    "BatchingOptimizer",
    "NetworkRooflineModel",
    "ForwardModel",
    "SensitivitySolver",
    "SynthesisSolver",
    "CheckpointModel",
    "OrchestrationModel",
    "ContinuousBatchingModel",
    "WeightStreamingModel",
    "TailLatencyModel",
    "TransformationModel",
    "TopologyModel",
    "ScalingModel",
    "InferenceScalingModel",
    "EfficiencyModel",
    "ResponsibleEngineeringModel",
})

# Flat paths removed when registry shims were deleted (CLI short names still OK in bash).
FLAT_MODEL_RE = re.compile(
    r"\b(?:mlsysim\.)?Models\.(ResNet50|Llama3_8B|Llama3_70B|Llama3_405B|"
    r"GPT2|GPT3|GPT4|MobileNetV2|AlexNet|MobileNet)\b"
)
FLAT_HW_RE = re.compile(
    r"\b(?:mlsysim\.)?Hardware\.(A100|H100|H200|T4|V100|B200|MI300X)\b"
)
FLAT_ATTR_RE = re.compile(
    r"\bmlsysim\.(" + "|".join(NON_TOP_LEVEL_SYMBOLS) + r")\b"
)

FROM_MLSYSIM_RE = re.compile(r"^\s*from\s+mlsysim\s+import\s+(.+)$")


def _parse_imported_names(clause: str) -> set[str]:
    names: set[str] = set()
    for part in clause.split(","):
        part = part.strip()
        if not part or part == "*":
            continue
        if " as " in part:
            part = part.split(" as ", 1)[0].strip()
        if "(" in part:
            continue  # multiline import — handled by parent scan
        names.add(part)
    return names


def _scan_file(path: Path) -> list[str]:
    hits: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.relative_to(REPO_ROOT)

    for lineno, line in enumerate(text.splitlines(), start=1):
        if (
            FLAT_MODEL_RE.search(line)
            or FLAT_HW_RE.search(line)
            or FLAT_ATTR_RE.search(line)
        ):
            hits.append(f"{rel}:{lineno}: invalid mlsysim API reference\n  {line.strip()}")

        m = FROM_MLSYSIM_RE.match(line)
        if m:
            bad = sorted(_parse_imported_names(m.group(1)) & NON_TOP_LEVEL_SYMBOLS)
            if bad:
                hits.append(
                    f"{rel}:{lineno}: import {', '.join(bad)} from mlsysim "
                    f"(use mlsysim.engine.solver or mlsysim.models.types)\n  {line.strip()}"
                )

    # Multiline `from mlsysim import (` blocks in markdown/python fences
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return hits

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in ("mlsysim", None):
            continue
        if node.level:
            continue
        imported = {alias.name for alias in node.names if alias.name != "*"}
        bad = sorted(imported & NON_TOP_LEVEL_SYMBOLS)
        if bad:
            hits.append(
                f"{rel}:{node.lineno}: import {', '.join(bad)} from mlsysim "
                f"(use mlsysim.engine.solver or mlsysim.models.types)"
            )
    return hits


def test_doc_registry_paths_and_imports() -> None:
    violations: list[str] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in SCAN_SUFFIXES:
                continue
            if path.name in SKIP_FILES:
                continue
            violations.extend(_scan_file(path))

    assert not violations, (
        "Website docs must use canonical nested registry paths and valid imports:\n"
        + "\n".join(violations[:50])
        + (f"\n... and {len(violations) - 50} more" if len(violations) > 50 else "")
    )
