#!/usr/bin/env python3
"""Advisory audit for LEGO inputs that should move into MLSysIM.

This checker is intentionally advisory. It answers a different question from
the unit lints:

    Does this LEGO LOAD-stage value define a book scenario, system, hardware
    object, infrastructure fact, model, dataset, pricing point, or workload
    assumption that should live in MLSysIM instead of a QMD cell?

It emits a work queue for the registry/source-of-truth migration. It should not
be promoted to a blocking gate until the findings have been migrated or
honestly allowlisted.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS = REPO_ROOT / "book" / "quarto" / "contents"

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
LEGO_MARK = re.compile(r"#\s*[│┌].*LEGO|#\s*\│ Exports:|#\s*Exports:")

STAGE_MARK = re.compile(r"#.*\b(?P<stage>LOAD|EXECUTE|GUARD|OUTPUT)\b", re.I)
CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)

REGISTRY_ROOT = re.compile(
    r"\b(?:Hardware|Systems|Infrastructure|Models|Datasets|Literature|Ops|"
    r"Scenarios|Platforms|Monitoring|calibration)\."
)

UNIT_TOKEN = re.compile(
    r"\b(?:"
    r"byte|bytes|bit|KB|MB|GB|TB|PB|KiB|MiB|GiB|TiB|"
    r"J|kJ|mJ|uJ|pJ|joule|kilojoule|microjoule|"
    r"Wh|kWh|MWh|GWh|watt|kilowatt|megawatt|gigawatt|"
    r"W|kW|MW|GW|mW|uW|"
    r"second|seconds|minute|minutes|hour|hours|day|days|month|months|year|years|"
    r"ms|us|ns|millisecond|microsecond|nanosecond|"
    r"flop|FLOP|TFLOP|TFLOPs|GFLOP|GFLOPs|PFLOP|PFLOPs|"
    r"param|Kparam|Mparam|Bparam|Tparam|"
    r"USD|dollar|gram|kg|kilogram|metric_ton|tonne|tonnes"
    r")\b"
)

CONSTRUCTOR_TARGETS = {
    "Fleet": "Systems.Clusters",
    "Node": "Systems.Nodes",
    "NetworkFabric": "Systems.Fabrics",
    "PodEnvelope": "Systems.Pods",
    "StorageHierarchy": "Hardware.*.storage or Systems.Storage",
    "StorageTier": "Hardware.Tech.Storage or Systems.Storage",
    "StorageSubsystem": "Systems.Storage",
    "NodeStorageConfig": "Systems.Storage",
    "CheckpointStoragePath": "Systems.Storage",
    "HardwareNode": "Hardware.*",
    "ComputeCore": "Hardware.*",
    "MemoryHierarchy": "Hardware.*.memory",
    "IOInterconnect": "Hardware.*.interconnect",
    "Datacenter": "Infrastructure.Datacenters",
    "GridProfile": "Infrastructure.Grids",
    "PricePoint": "Infrastructure.Pricing",
}

KEEP_LOCAL_NAME = re.compile(
    r"(?:^|_)(?:"
    r"i|j|k|n|m|x|y|z|row|col|idx|index|offset|seed|"
    r"plot|axis|xmin|xmax|ymin|ymax|label|color|alpha|"
    r"precision|digits|round|decimal|commas|width|height"
    r")(?:_|$)",
    re.I,
)

STORAGE_WORD = re.compile(
    r"(?:storage|checkpoint|ckpt|nvme|ssd|hdd|pfs|filesystem|file_system|"
    r"object_store|bucket|s3|drive|disk|staging|ingest|corpus)",
    re.I,
)
STORAGE_SYSTEM_WORD = re.compile(
    r"(?:drive|disk|nvme|ssd|hdd|pfs|filesystem|file_system|object_store|bucket|"
    r"staging|stage|capacity|bandwidth|bw|throughput|latency)",
    re.I,
)
STORAGE_MEASUREMENT_WORD = re.compile(
    r"(?:^|_)(?:bw|bandwidth|throughput|capacity|iops|latency|drive|drives|"
    r"disk|disks)(?:_|$)",
    re.I,
)
STORAGE_OBSERVATION_WORD = re.compile(
    r"(?:sat|saturation|util|utilization|pct|percent|ratio)", re.I
)
MODEL_WORD = re.compile(
    r"(?:model|params?|parameters?|tokens?|layers?|heads?|hidden|vocab|context|"
    r"seq|sequence|embedding|kv|cache|experts?|moe|gpt|llama|bert|resnet|"
    r"mobilenet|yolo|transformer)",
    re.I,
)
DATASET_WORD = re.compile(
    r"(?:dataset|corpus|imagenet|mnist|cifar|samples?|examples?|images?|records?)",
    re.I,
)
SYSTEM_WORD = re.compile(
    r"(?:^|_)(?:fleet|cluster|node|nodes|gpu|gpus|accelerator|server|rack|"
    r"pod|nic|nics|psu|psus|fabric|network|topology|switch|infiniband|"
    r"ethernet|roce|nvlink|pcie|mtbf|failure)(?:_|$)",
    re.I,
)
HARDWARE_WORD = re.compile(
    r"(?:h100|h200|a100|b200|v100|t4|tpu|gpu|cpu|npu|chip|die|hbm|dram|vram|"
    r"memory|tdp|flops?|tops?|pflo|tflo|int8|fp16|fp32|fp8|fp4|nvlink|pcie)",
    re.I,
)
INFRA_WORD = re.compile(
    r"(?:grid|carbon|co2|emissions?|pue|datacenter|data_center|facility|cooling|"
    r"water|energy|power|renewable|region|quebec|poland)",
    re.I,
)
PRICE_WORD = re.compile(
    r"(?:usd|dollar|price|cost|rate|fee|capex|opex|hourly|monthly)",
    re.I,
)
GPU_HOUR_PRICE_WORD = re.compile(
    r"(?:gpu.*(?:hr|hour)|(?:hr|hour).*gpu|gpu_price|gpu_cost|instance|fleet)",
    re.I,
)
WORKLOAD_WORD = re.compile(
    r"(?:qps|queries?|requests?|users?|traffic|arrival|sla|deadline|duration|"
    r"window|days?|hours?|minutes?|batch|epochs?|steps?|sampling|monitoring|"
    r"telemetry|retraining|error|accuracy|threshold|drift|psi|ks)",
    re.I,
)
WORKLOAD_PHASE_WORD = re.compile(
    r"(?:preprocess|postprocess|forward|backward|allreduce|warmup|compile|"
    r"cuda|runtime|latency|(?:^|_)lat(?:_|$)|deadline|budget|overhead|"
    r"step|batch|bucket|"
    r"profile|pipeline|dispatch|sample|query|request|per_layer)",
    re.I,
)
SCENARIO_COMMENT_WORD = re.compile(
    r"#.*\b(?:scenario|hypothetical|illustrative|reference|baseline|budget)\b",
    re.I,
)
FABRIC_SPEC_WORD = re.compile(
    r"(?:^|_)(?:fabric|switch|infiniband|ethernet|roce|nvlink|pcie|ports?|"
    r"spine|leaf|endpoint_links|links_per_gpu|gbps|tbps|nvswitch)(?:_|$)",
    re.I,
)
TOPOLOGY_COUNT_WORD = re.compile(
    r"(?:gpu_count|n_gpus|num_gpus|node_count|n_nodes|num_nodes|nodes_per|"
    r"gpus_per_node|nics_per_node|racks?|pods?)",
    re.I,
)
HARDWARE_WORKLOAD_FLOPS_WORD = re.compile(
    r"(?:gemm|decode|forward|backward|achieved|per_sample|workload|sample)",
    re.I,
)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    chapter: str
    cell: str
    stage: str
    symbol: str
    rhs: str
    target: str
    confidence: str
    reason: str
    action: str = "review_for_mlsysim"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _chapter_key(path: Path) -> str:
    rel = _repo_rel(path)
    match = re.search(r"book/quarto/contents/(vol[12]/(?:backmatter/)?[^/]+)", rel)
    return match.group(1) if match else rel


def _resolve_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return sorted(CONTENTS.rglob("*.qmd"))
    out: list[Path] = []
    for path in paths:
        p = path if path.is_absolute() else REPO_ROOT / path
        if p.is_dir():
            out.extend(sorted(p.rglob("*.qmd")))
        elif p.suffix == ".qmd":
            out.append(p)
    return sorted(dict.fromkeys(out))


def _python_cells(path: Path) -> list[tuple[int, str, bool]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[tuple[int, str, bool]] = []
    in_cell = False
    start = 0
    buf: list[str] = []
    for line_no, line in enumerate(lines, 1):
        if CELL_START.match(line):
            in_cell = True
            start = line_no
            buf = []
            continue
        if in_cell and CELL_END.match(line):
            code = "\n".join(buf)
            is_lego = bool(LEGO_MARK.search(code) or "Exports:" in code)
            cells.append((start + 1, code, is_lego))
            in_cell = False
            continue
        if in_cell:
            buf.append(line)
    return cells


def _stage_by_line(code: str) -> dict[int, str]:
    current = ""
    stages: dict[int, str] = {}
    for idx, line in enumerate(code.splitlines(), start=1):
        if match := STAGE_MARK.search(line):
            current = match.group("stage").upper()
        stages[idx] = current
    return stages


def _class_name(code: str) -> str:
    match = CLASS_RE.search(code)
    return match.group(1) if match else ""


def _source_for(code: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(code, node)
    if segment:
        return " ".join(segment.strip().split())
    return ""


def _node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _node_name(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    return None


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _node_name(child.func)
            if name:
                names.add(name)
    return names


def _has_numeric(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
            if isinstance(child.value, bool):
                continue
            return True
    return False


def _is_registry_sourced(rhs: str) -> bool:
    return bool(REGISTRY_ROOT.search(rhs))


def _is_safe_local(name: str, rhs: str) -> bool:
    text = f"{name} {rhs}"
    if KEEP_LOCAL_NAME.search(text) and not UNIT_TOKEN.search(text):
        return True
    if re.fullmatch(r"[\[\]\(\),\s\d_.'\"+-]+", rhs):
        if re.search(r"label|tick|color|plot|axis|bar|range", name, re.I):
            return True
    return False


def _classify(name: str, rhs: str, calls: set[str]) -> tuple[str, str, str]:
    text = f"{name} {rhs}"
    lower = text.lower()

    if "type" in calls and re.search(r"dummy|fleet|node|grid|storage", text, re.I):
        if "fleet" in lower or "node" in lower:
            return "Systems.Clusters or Systems.Nodes", "high", "local Dummy system object"
        if "grid" in lower or "pue" in lower or "carbon" in lower:
            return "Infrastructure.* or Scenarios.Sustainability", "high", "local Dummy infrastructure/scenario object"
        return "Scenarios.*", "medium", "local Dummy object"

    for call in sorted(calls):
        short = call.rsplit(".", 1)[-1]
        if short in CONSTRUCTOR_TARGETS:
            return CONSTRUCTOR_TARGETS[short], "high", f"direct {short} constructor in QMD"

    if PRICE_WORD.search(text):
        if STORAGE_WORD.search(text):
            return "Infrastructure.Pricing.Storage", "high", "storage price/rate belongs in pricing registry"
        if any(word in lower for word in ("label", "clinical", "radiologist", "specialist")):
            return "Infrastructure.Pricing.Labeling or Scenarios.*", "medium", "human-labeling price input"
        if GPU_HOUR_PRICE_WORD.search(text):
            return "Infrastructure.Pricing.Cloud or Infrastructure.Pricing.Fleet", "high", "cloud/fleet price point"
        return "Infrastructure.Pricing.* or Scenarios.*", "medium", "economic input or scenario price"

    if STORAGE_WORD.search(text):
        if any(word in lower for word in ("interval", "period", "cadence", "every")):
            return "Scenarios.TrainingRuns or Systems.Reliability", "medium", "checkpoint policy/workload cadence"
        if STORAGE_OBSERVATION_WORD.search(name):
            return "Scenarios.* or Ops.*", "medium", "storage observation or utilization scenario"
        if STORAGE_MEASUREMENT_WORD.search(name):
            return "Systems.Storage", "high", "storage subsystem fact"
        if STORAGE_SYSTEM_WORD.search(text):
            return "Systems.Storage or Scenarios.*", "medium", "storage-related scenario input"
        return "Systems.Storage or Datasets.*", "medium", "storage/data fact"

    if HARDWARE_WORD.search(text) and UNIT_TOKEN.search(text):
        if HARDWARE_WORKLOAD_FLOPS_WORD.search(name):
            return "Models.* or Scenarios.TrainingRuns", "medium", "workload compute requirement"
        if any(word in lower for word in ("gpu_count", "n_gpus", "nodes", "node_count", "fleet", "cluster")):
            return "Systems.Clusters or Systems.Nodes", "high", "fleet/topology fact"
        if any(word in lower for word in ("nvlink", "pcie", "hbm", "memory", "dram", "vram", "tdp", "flops", "tflop", "pflo")):
            return "Hardware.* or Hardware.Tech.*", "high", "hardware specification"
        return "Hardware.*", "medium", "hardware-related quantitative input"

    if SYSTEM_WORD.search(text):
        if WORKLOAD_PHASE_WORD.search(text):
            return "Scenarios.* or Ops.*", "medium", "scenario/workload policy"
        if FABRIC_SPEC_WORD.search(text):
            return "Systems.Fabrics or Systems.SwitchFabric", "high", "network/fabric system fact"
        if TOPOLOGY_COUNT_WORD.search(text):
            return "Systems.Clusters or Systems.Nodes", "high", "fleet/topology fact"
        return "Systems.*", "medium", "system-level fact"

    if INFRA_WORD.search(text):
        if any(word in lower for word in ("grid", "carbon", "co2", "pue", "datacenter", "facility", "cooling", "water")):
            return "Infrastructure.* or Scenarios.Sustainability", "high", "infrastructure/sustainability fact"
        return "Infrastructure.*", "medium", "infrastructure input"

    if MODEL_WORD.search(text):
        if DATASET_WORD.search(text) and "tokens" not in lower:
            return "Datasets.* or Scenarios.DataWorkloads", "medium", "dataset/workload size"
        return "Models.* or Scenarios.TrainingRuns", "medium", "model/workload specification"

    if DATASET_WORD.search(text):
        return "Datasets.* or Scenarios.DataWorkloads", "medium", "dataset/workload specification"

    if WORKLOAD_WORD.search(text):
        return "Scenarios.* or Ops.*", "medium", "scenario/workload policy"

    if UNIT_TOKEN.search(text):
        return "Scenarios.*", "medium", "unit-bearing scenario input"

    return "Scenarios.* or keep local", "low", "bare numeric scenario input"


def _downgrade_scenario_comment(
    line: str,
    target: str,
    confidence: str,
    reason: str,
) -> tuple[str, str, str]:
    if confidence != "high" or not SCENARIO_COMMENT_WORD.search(line):
        return target, confidence, reason
    if target.startswith("Infrastructure.Pricing"):
        return "Infrastructure.Pricing.* or Scenarios.*", "medium", "scenario/profile input"
    if target.startswith("Infrastructure"):
        return "Infrastructure.* or Scenarios.*", "medium", "scenario/profile input"
    if target.startswith("Systems.Storage"):
        return "Systems.Storage or Scenarios.*", "medium", "scenario/profile input"
    if target.startswith("Systems"):
        return "Systems.* or Scenarios.*", "medium", "scenario/profile input"
    if target.startswith("Hardware"):
        return "Hardware.* or Scenarios.*", "medium", "scenario/profile input"
    return "Scenarios.*", "medium", "scenario/profile input"


def _target_names(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            yield from _target_names(elt)


def _findings_for_cell(path: Path, cell_start: int, code: str) -> list[Finding]:
    findings: list[Finding] = []
    stages = _stage_by_line(code)
    cell = _class_name(code)
    chapter = _chapter_key(path)
    rel = _repo_rel(path)
    code_lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.lower().startswith("dummy"):
            line = cell_start + node.lineno - 1
            target, confidence, reason = _classify(node.name, node.name, set())
            findings.append(
                Finding(
                    file=rel,
                    line=line,
                    chapter=chapter,
                    cell=cell or node.name,
                    stage=stages.get(node.lineno, ""),
                    symbol=node.name,
                    rhs="class definition",
                    target=target,
                    confidence=confidence,
                    reason=reason,
                )
            )

        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        lineno = getattr(node, "lineno", 0)
        stage = stages.get(lineno, "")
        if stage != "LOAD":
            continue

        targets: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                targets.extend(_target_names(target))
        else:
            targets.extend(_target_names(node.target))
        if not targets:
            continue

        rhs = _source_for(code, value)
        calls = _call_names(value)
        has_constructor = any(call.rsplit(".", 1)[-1] in CONSTRUCTOR_TARGETS or call == "type" for call in calls)
        has_domain_value = _has_numeric(value) or has_constructor
        if not has_domain_value:
            continue
        if _is_registry_sourced(rhs):
            continue

        for name in targets:
            if name.endswith(("_str", "_math", "_eq", "_frac")):
                continue
            if _is_safe_local(name, rhs):
                continue
            target, confidence, reason = _classify(name, rhs, calls)
            line_text = code_lines[lineno - 1] if 0 < lineno <= len(code_lines) else ""
            target, confidence, reason = _downgrade_scenario_comment(
                line_text, target, confidence, reason
            )
            if confidence == "low" and not UNIT_TOKEN.search(rhs) and not has_constructor:
                continue
            findings.append(
                Finding(
                    file=rel,
                    line=cell_start + lineno - 1,
                    chapter=chapter,
                    cell=cell,
                    stage=stage,
                    symbol=name,
                    rhs=rhs,
                    target=target,
                    confidence=confidence,
                    reason=reason,
                )
            )

    return findings


def check_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for cell_start, code, is_lego in _python_cells(path):
        if not is_lego:
            continue
        findings.extend(_findings_for_cell(path, cell_start, code))
    return sorted(findings, key=lambda item: (item.file, item.line, item.symbol))


def _print_summary(findings: list[Finding]) -> None:
    by_target = Counter(f.target for f in findings)
    by_chapter = Counter(f.chapter for f in findings)
    by_reason = Counter(f.reason for f in findings)
    print(f"Findings: {len(findings)}")
    print("By target:")
    for target, count in by_target.most_common():
        print(f"  {count:4d}  {target}")
    print("Top chapters:")
    for chapter, count in by_chapter.most_common(20):
        print(f"  {count:4d}  {chapter}")
    print("Top reasons:")
    for reason, count in by_reason.most_common(20):
        print(f"  {count:4d}  {reason}")


def _print_markdown(findings: list[Finding]) -> None:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.chapter].append(finding)

    print("# LEGO MLSysIM Source-of-Truth Audit")
    print()
    print("This is an advisory work queue for values that likely define")
    print("hardware, systems, infrastructure, pricing, models, datasets,")
    print("storage subsystems, or scenarios in LEGO LOAD stages.")
    print()
    _print_summary(findings)
    print()
    for chapter in sorted(grouped):
        print(f"## {chapter}")
        print()
        print("| File:Line | Cell | Symbol | Target | Reason | RHS |")
        print("|---|---|---|---|---|---|")
        for f in grouped[chapter]:
            rhs = f.rhs.replace("|", "\\|")
            if len(rhs) > 120:
                rhs = rhs[:117] + "..."
            print(
                f"| `{f.file}:{f.line}` | `{f.cell}` | `{f.symbol}` | "
                f"`{f.target}` | {f.reason} | `{rhs}` |"
            )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="QMD files or directories")
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--summary", action="store_true", help="Print grouped counts only")
    parser.add_argument("--output", type=Path, help="Write output to a file")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit 1 when findings exist")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in _resolve_paths(args.paths):
        if path.exists() and path.suffix == ".qmd":
            findings.extend(check_file(path))
    findings.sort(key=lambda item: (item.file, item.line, item.symbol))

    if args.format == "json":
        content = json.dumps([asdict(f) for f in findings], indent=2)
    elif args.format == "markdown":
        from io import StringIO
        import contextlib

        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _print_markdown(findings)
        content = buf.getvalue()
    elif args.summary:
        from io import StringIO
        import contextlib

        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            _print_summary(findings)
        content = buf.getvalue()
    else:
        lines = []
        for f in findings:
            lines.append(
                f"{f.file}:{f.line}: [{f.confidence}] {f.symbol} -> "
                f"{f.target}: {f.reason}"
            )
            lines.append(f"    {f.rhs}")
        content = "\n".join(lines)

    if args.output:
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    else:
        print(content, end="" if content.endswith("\n") else "\n")

    return 1 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
