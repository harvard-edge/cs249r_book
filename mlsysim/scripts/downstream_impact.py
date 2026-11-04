"""Compare modeled textbook/paper outputs under two package versions.

Only execute trusted repository sources. This is a headless numerical review,
not a sandbox or a rendered-publication check. Snapshot both versions against
the SAME consumer checkout, then compare. Matplotlib save/show calls are suppressed and recorded;
unsupported execution and skipped cells remain visible review requirements.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import distributions
import io
import json
import math
import numbers
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SCHEMA = 1
FENCE = re.compile(
    r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n"
    r"(?P<body>.*?)^ {0,3}(?P=fence)[ \t]*$", re.M | re.S
)
INLINE = re.compile(r"(?<!`)`\{python\}[ \t]+([^`\n]+)`(?!`)")
LIMITS = [
    "Headless Python execution; no rendered prose, browser, plot, or layout verification.",
    "Inline expressions use the final document namespace, as in the previous numerical audit.",
    "Raw exports cover scalars, quantities, containers, local classes, and model results; not arbitrary object internals.",
    "Only discovered textbook Python cells and paper generator scenarios are executed; other consumers need separate review.",
]


def digest(path: Path) -> str:
    """Hash source bytes, without normalizing whitespace or line endings."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(root: Path, paths) -> dict[str, str]:
    """Record all specified files relative to their declared root."""
    return {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))}


def package_inventory(package: Path) -> dict[str, str]:
    """Identify source and registry inputs of the selected implementation."""
    return source_inventory(package, [p for p in (package / "mlsysim").rglob("*")
                                      if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".json"}])


def encode(value, depth=0):
    """Return deterministic data for accessible numerical outputs, or None."""
    if depth > 8:
        return None
    if isinstance(value, type) and value.__module__ != "__downstream__":
        # Quantity classes expose descriptor objects named magnitude/units;
        # those are not values and their repr contains process-local addresses.
        return None
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return {"quantity": encode(value.magnitude, depth + 1), "unit": format(value.units, "D"),
                # Display formatting can round fractional exponents. Retain
                # their full stored representations for numerical comparison.
                "unit_powers": {name: repr(power) for name, power in sorted(value.units._units.items())}}
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else {"nonfinite": str(numeric)}
    if isinstance(value, (list, tuple)):
        return [encode(x, depth + 1) for x in value]
    if isinstance(value, dict):
        return {str(k): encode(v, depth + 1) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return encode(value.tolist(), depth + 1)
    if (isinstance(value, type) and value.__module__ == "__downstream__") or (
        type(value).__module__ == "__downstream__"
    ) or type(value).__module__.startswith("mlsysim.engine.results"):
        result = {}
        for name, item in sorted(vars(value).items()):
            if name.startswith("_") or callable(item):
                continue
            encoded = encode(item, depth + 1)
            if encoded is not None:
                result[name] = encoded
        return result
    return None


def assert_package(package: Path) -> str:
    """Reject a snapshot that accidentally imports a different installation."""
    import mlsysim
    expected = (package / "mlsysim").resolve()
    for name, module in list(sys.modules.items()):
        if name == "mlsysim" or name.startswith("mlsysim."):
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(expected):
                raise RuntimeError(f"Wrong package imported: {name}: {path}; expected {expected}")
    return str(Path(mlsysim.__file__).resolve())


def suppress_figures(events: list[str]) -> None:
    """Avoid rewriting source figure artifacts during numerical comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    plt.show = lambda *a, **k: events.append("matplotlib.show")
    Figure.savefig = lambda *a, **k: events.append("matplotlib.savefig")


def context_at(text: str, start: int, end: int) -> dict:
    """Locate the containing prose paragraph for a human editorial review."""
    left = text.rfind("\n\n", 0, start) + 2
    right = text.find("\n\n", end)
    if right < 0:
        right = len(text)
    return {"line": text.count("\n", 0, start) + 1, "prose": text[max(0, left):right].strip()}


def snapshot_qmd(path: Path, root: Path) -> dict:
    """Execute enabled cells and capture inline and raw final-namespace values."""
    text = path.read_text()
    document_disabled = False
    if text.startswith("---\n"):
        import yaml
        boundary = text.find("\n---", 4)
        metadata = yaml.safe_load(text[4:boundary]) if boundary >= 0 else {}
        execution = (metadata or {}).get("execute", {})
        document_disabled = execution is False or (isinstance(execution, dict) and execution.get("eval") is False)
    scope = {"__name__": "__downstream__"}
    errors, skipped, cells, plots = [], [], [], []
    suppress_figures(plots)
    narrative = list(text)
    output = io.StringIO()
    for match in FENCE.finditer(text):
        # Preserve offsets for inline prose locations, excluding all code fences.
        narrative[match.start():match.end()] = " " * (match.end() - match.start())
        if not re.match(r"\{python(?:[ ,}]|$)", match["info"].strip()):
            continue
        line = text.count("\n", 0, match.start()) + 1
        indentation = len(match["indent"])
        code = "\n".join(line[min(indentation, len(line) - len(line.lstrip(" "))):]
                         for line in match["body"].split("\n"))
        cells.append({"line": line, "sha256": hashlib.sha256(code.encode()).hexdigest()})
        if document_disabled or re.search(r"^#\|\s*eval:\s*false\b", code, re.M | re.I):
            skipped.append({"line": line, "reason": "document eval:false" if document_disabled else "eval:false"})
            continue
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(code, str(path), "exec"), scope)
        except Exception as exc:
            errors.append({"line": line, "error": f"{type(exc).__name__}: {exc}"})
            break
    remaining = "".join(narrative)
    for match in re.finditer(r"^[ \t]*(?:`{3,}|~{3,})\{python\b", remaining, re.M):
        errors.append({"line": text.count("\n", 0, match.start()) + 1,
                       "error": "Unsupported or unclosed Python fence; execution coverage is incomplete"})
    inline = {}
    for match in INLINE.finditer(remaining):
        expr = match[1]
        key = f"{text.count(chr(10), 0, match.start()) + 1}:{match.start()}:{expr}"
        record = {"expression": expr, **context_at(text, match.start(), match.end())}
        try:
            value = eval(expr, scope)
            record.update(displayed=str(value), raw=encode(value))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors.append({"line": record["line"], "error": record["error"]})
        inline[key] = record
    raw = {}
    for name, value in sorted(scope.items()):
        if name.startswith("_"):
            continue
        encoded = encode(value)
        if encoded is not None:
            raw[name] = encoded
    return {"kind": "textbook", "source_sha256": digest(path), "cells": cells,
            "skipped": skipped, "errors": errors, "inline": inline, "raw": raw,
            "suppressed_figures": plots, "stdout": output.getvalue()}


def snapshot_paper(root: Path) -> dict:
    """Run deterministic paper scenarios without rewriting generated artifacts."""
    sys.path.insert(0, str(root / "mlsysim/paper/scripts"))
    generator = importlib.import_module("generate_paper_values")
    values = generator.Values()
    # Exclude machine-dependent counts and runtime measurements. These are not
    # modeled paper outputs and have a separate provenance/freshness gate.
    sections = ["section_registry", "section_anchors", "section_walls", "section_instructor",
                "section_carbon", "section_students", "section_researchers", "section_overview",
                "section_chain_example", "section_listings"]
    errors = []
    for section in sections:
        try:
            fn = getattr(generator, section)
            if section in {"section_instructor", "section_overview"}:
                fn(values, True)
            else:
                fn(values)
        except Exception as exc:
            errors.append({"section": section, "error": f"{type(exc).__name__}: {exc}"})
    text = (root / "mlsysim/paper/paper.tex").read_text()
    inline = {}
    for name, record in values.records.items():
        if record["volatile"]:
            continue
        consumers = []
        for match in re.finditer(r"\\" + re.escape(name) + r"\b", text):
            consumers.append(context_at(text, match.start(), match.end()))
        inline[name] = {"expression": name, "displayed": record["printed"], "raw": record["raw"],
                        "source": record["source"], "consumers": consumers}
    return {"kind": "paper", "source_sha256": digest(root / "mlsysim/paper/paper.tex"),
            "cells": sections, "skipped": [], "errors": errors, "inline": inline,
            "raw": {name: row["raw"] for name, row in inline.items()},
            "excluded": ["runtime measurements", "environment/implementation counts", "diagram label aliases"]}


def run_worker(root: Path, package: Path, consumer: str) -> dict:
    """Import a selected implementation before any consumer modifies sys.path."""
    sys.path.insert(0, str(package))
    sys.path.insert(1, str(root))
    import mlsysim  # noqa: F401
    origin = assert_package(package)
    os.chdir(root / "books" if consumer != "paper" else root)
    if consumer == "paper":
        result = snapshot_paper(root)
    else:
        result = snapshot_qmd(root / consumer, root)
    result["package_origin"] = origin
    dependencies = [Path(module.__file__).resolve() for name, module in list(sys.modules.items())
                    if getattr(module, "__file__", None) and not (name == "mlsysim" or name.startswith("mlsysim."))
                    and Path(module.__file__).resolve().is_relative_to(root)
                    and Path(module.__file__).suffix == ".py"
                    and not any(part in {".venv", "site-packages"} for part in Path(module.__file__).parts)]
    result["imports_sha256"] = source_inventory(root, dependencies)
    assert_package(package)
    return result


def snapshot(root: Path, package: Path, selections: list[str], paper: bool, workers: int, timeout: float) -> dict:
    """Execute each consumer in its own process and retain failures as data."""
    matches = {pattern: [p for p in root.glob(pattern) if p.is_file() and p.suffix == ".qmd"]
               for pattern in selections}
    paths = sorted({p for found in matches.values() for p in found})
    consumers = [str(p.relative_to(root)) for p in paths if re.search(r"\{python[}, ]", p.read_text())]
    if paper:
        consumers.append("paper")
    inputs = list(paths)
    for volume in range(1, 5):
        inputs.extend(p for p in (root / f"books/vol{volume}").rglob("*")
                      if p.is_file() and p.suffix in {".csv", ".yaml", ".yml", ".json", ".py"})
    if paper:
        inputs.extend(p for directory in ("mlsysim/paper/scripts", "mlsysim/paper/data")
                      for p in (root / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".csv", ".json"})
        inputs.append(root / "mlsysim/paper/paper.tex")
    initial_inputs = source_inventory(root, inputs)
    initial_package = package_inventory(package)

    def execute(consumer):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(package), str(root)]),
                   MPLBACKEND="Agg", PYTHONPYCACHEPREFIX=cache_root)
        with tempfile.TemporaryDirectory(prefix="mlsysim-impact-") as tmp:
            result_path = Path(tmp) / "result.json"
            command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--root", str(root),
                       "--package", str(package), "--consumer", consumer, "--output", str(result_path)]
            try:
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
                if run.returncode == 0:
                    return consumer, json.loads(result_path.read_text())
                return consumer, {"errors": [{"error": run.stderr[-4000:] or run.stdout[-4000:]}]}
            except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                return consumer, {"errors": [{"error": f"{type(exc).__name__}: {exc}"}]}

    # Existing timestamp-valid .pyc files can hide changed source bytes. A new
    # cache for each snapshot makes imports compile the selected source version.
    # Workers may share this fresh cache; mid-run source edits invalidate the run.
    with tempfile.TemporaryDirectory(prefix="mlsysim-impact-bytecode-") as cache_root:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(pool.map(execute, consumers))
    errors = [f"No QMD sources matched selection: {pattern}" for pattern, found in matches.items() if not found]
    if initial_inputs != source_inventory(root, inputs):
        errors.append("Consumer/input files changed during snapshot")
    if initial_package != package_inventory(package):
        errors.append("Package files changed during snapshot")
    if not consumers:
        errors.append("No executable consumers selected")
    return {"schema": SCHEMA, "kind": "modeled downstream outputs", "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "python": sys.version,
            "root": str(root), "package": str(package), "package_sha256": initial_package,
            "bytecode_policy": "fresh shared cache per snapshot; no inherited package bytecode",
            "environment": {dist.metadata["Name"]: dist.version for dist in distributions()
                            if dist.metadata["Name"] and dist.metadata["Name"].lower() != "mlsysim"},
            "inputs_sha256": initial_inputs, "tool_sha256": digest(Path(__file__)),
            "selection": {"globs": selections, "paper": paper, "consumers": consumers},
            "errors": errors, "limitations": LIMITS, "consumers": results}


def leaves(value, prefix=""):
    """Flatten raw outputs to give each changed scalar an actionable path."""
    if isinstance(value, dict):
        yield prefix, {"container": "mapping", "keys": sorted(value)}
        for key, item in value.items():
            escaped = key.replace("\\", "\\\\").replace(".", "\\.")
            yield from leaves(item, f"{prefix}.{escaped}" if prefix else escaped)
    elif isinstance(value, list):
        yield prefix, {"container": "sequence", "length": len(value)}
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def compare(before: dict, after: dict) -> dict:
    """Fail closed on incompatible inputs, errors, missing coverage, or skips."""
    problems, changes, reviews = [], [], []
    for key in ("schema", "selection", "inputs_sha256", "tool_sha256", "python", "environment"):
        if key not in before or before.get(key) != after.get(key):
            problems.append(f"Incompatible or missing {key}; capture both versions against the same sources and environment")
    for label, data in (("before", before), ("after", after)):
        if data.get("schema") != SCHEMA or not data.get("package_sha256"):
            problems.append(f"{label}: unsupported schema or missing package provenance")
        problems.extend(f"{label}: {item}" for item in data.get("errors", []))
        if not data.get("consumers"):
            problems.append(f"{label}: missing consumer coverage")
        if set(data.get("consumers", {})) != set(data.get("selection", {}).get("consumers", [])):
            problems.append(f"{label}: consumer results do not cover selection")
        for path, record in data.get("consumers", {}).items():
            problems.extend(f"{label} {path}: {error}" for error in record.get("errors", []))
            if not record.get("inline") and not record.get("raw"):
                reviews.append({"consumer": path, "version": label, "reason": "no captured numerical or inline outputs"})
            if record.get("skipped"):
                reviews.append({"consumer": path, "version": label, "reason": "skipped cells", "details": record["skipped"]})
    common = set(before.get("consumers", {})) & set(after.get("consumers", {}))
    for path in sorted(common):
        old, new = before["consumers"][path], after["consumers"][path]
        for field in ("source_sha256", "cells", "skipped", "imports_sha256"):
            if field not in old or old.get(field) != new.get(field):
                problems.append(f"{path}: missing or changed {field} coverage")
        if old.get("stdout", "") != new.get("stdout", ""):
            changes.append({"consumer": path, "kind": "stdout", "output": "cell output",
                            "before": old.get("stdout", ""), "after": new.get("stdout", ""),
                            "editorial_review": "Review changed printed output and nearby explanations."})
        for field in ("inline", "raw"):
            if field not in old or field not in new:
                problems.append(f"{path}: missing {field} output")
                continue
            old_values = dict(leaves(old[field])) if field == "raw" else old[field]
            new_values = dict(leaves(new[field])) if field == "raw" else new[field]
            for name in sorted(set(old_values) | set(new_values)):
                if old_values.get(name) == new_values.get(name) and (name in old_values) == (name in new_values):
                    continue
                entry = {"consumer": path, "kind": field, "output": name,
                         "before": old_values.get(name), "after": new_values.get(name),
                         "editorial_review": "Review associated explanations and constraints; do not automatically rewrite prose."}
                if field == "raw":
                    owner = name.split(".")[0]
                    entry["nearby_prose"] = [row for row in new.get("inline", {}).values()
                                             if row.get("expression", "").split(".")[0] == owner]
                changes.append(entry)
    return {"schema": SCHEMA, "status": "incomplete" if problems else "review_required" if changes or reviews else "unchanged",
            "problems": problems, "changes": changes, "review_requirements": reviews, "limitations": LIMITS,
            "baseline_package_sha256": before.get("package_sha256"), "candidate_package_sha256": after.get("package_sha256")}


def main() -> int:
    """Capture trusted consumers or compare immutable snapshot files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    capture.add_argument("--package", type=Path, required=True, help="Directory containing the mlsysim package")
    capture.add_argument("--glob", action="append", dest="globs", help="Repository-relative trusted QMD glob; repeatable")
    capture.add_argument("--no-paper", action="store_true")
    capture.add_argument("--workers", type=int, default=4)
    capture.add_argument("--timeout", type=float, default=90)
    capture.add_argument("--output", type=Path, required=True)
    diff = commands.add_parser("compare")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    for name in ("root", "package", "output"):
        worker.add_argument(f"--{name}", type=Path, required=True)
    worker.add_argument("--consumer", required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "compare":
        result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
        result["snapshots_sha256"] = {"before": digest(args.before), "after": digest(args.after)}
        status = {"unchanged": 0, "review_required": 1, "incomplete": 2}[result["status"]]
    elif args.command == "snapshot":
        if args.workers < 1 or args.timeout <= 0:
            parser.error("workers and timeout must be positive")
        result = snapshot(args.root.resolve(), args.package.resolve(), args.globs or [f"books/vol{v}/**/*.qmd" for v in range(1, 5)],
                          not args.no_paper, args.workers, args.timeout)
        status = int(bool(result["errors"] or any(row.get("errors") for row in result["consumers"].values())))
    else:
        result = run_worker(args.root.resolve(), args.package.resolve(), args.consumer)
        status = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"{args.command}: {args.output}" + (f" ({result['status']})" if "status" in result else ""))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
