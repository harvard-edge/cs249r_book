from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from mlperf.registry import PRODUCT_SUITES, Workload, load_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the native registry and packaged data catalogs."
    )
    parser.add_argument(
        "--source", default="registry", help="Native registry directory to export."
    )
    parser.add_argument(
        "--output",
        default="workloads.yaml",
        help="Compatibility flat registry YAML path.",
    )
    parser.add_argument(
        "--package-output",
        default="src/mlperf_edu/workloads.yaml",
        help="Packaged flat registry YAML copy included in wheels.",
    )
    parser.add_argument(
        "--dataset-source",
        default="datasets.yaml",
        help="Authored dataset catalog included with the package.",
    )
    parser.add_argument(
        "--dataset-package-output",
        default="src/mlperf_edu/datasets.yaml",
        help="Packaged dataset catalog copy included in wheels.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the flat registry and packaged catalogs without writing.",
    )
    args = parser.parse_args()

    workloads = load_registry(args.source)
    content = dump_yaml(native_to_flat(workloads))
    output = Path(args.output)
    package_output = Path(args.package_output) if args.package_output else None
    dataset_source = Path(args.dataset_source) if args.dataset_source else None
    dataset_package_output = (
        Path(args.dataset_package_output) if args.dataset_package_output else None
    )
    dataset_content = (
        dataset_source.read_text(encoding="utf-8")
        if dataset_source is not None
        else None
    )

    if args.check:
        problems = check_output(output, content)
        if package_output is not None:
            problems.extend(check_output(package_output, content))
        if dataset_package_output is not None and dataset_content is not None:
            problems.extend(check_output(dataset_package_output, dataset_content))
        if problems:
            for problem in problems:
                print(problem)
            print("run: python3 tools/export_flat_registry.py")
            return 1
        checked = f"{output}"
        if package_output is not None:
            checked += f" and {package_output}"
        if dataset_package_output is not None:
            checked += f" and {dataset_package_output}"
        print(f"{checked} are current ({len(workloads)} workload(s))")
        return 0

    output.write_text(content, encoding="utf-8")
    written = [str(output)]
    if package_output is not None:
        package_output.parent.mkdir(parents=True, exist_ok=True)
        package_output.write_text(content, encoding="utf-8")
        written.append(str(package_output))
    if dataset_package_output is not None and dataset_content is not None:
        dataset_package_output.parent.mkdir(parents=True, exist_ok=True)
        dataset_package_output.write_text(dataset_content, encoding="utf-8")
        written.append(str(dataset_package_output))
    print(f"wrote {len(workloads)} workload(s) to {', '.join(written)}")
    return 0


def check_output(path: Path, expected: str) -> list[str]:
    if not path.exists():
        return [f"missing {path}"]
    if path.read_text(encoding="utf-8") != expected:
        return [f"stale {path}"]
    return []


def native_to_flat(workloads: dict[str, Workload]) -> dict[str, Any]:
    suites: dict[str, dict[str, Any]] = {suite: {} for suite in PRODUCT_SUITES}
    for workload in workloads.values():
        suites.setdefault(workload.suite, {})[workload.id] = workload.raw
    return {"suites": {suite: entries for suite, entries in suites.items() if entries}}


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    raise SystemExit(main())
