from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from mlperf.registry import PRODUCT_SUITES, Workload, load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the native registry layout into legacy workloads.yaml.")
    parser.add_argument("--source", default="registry", help="Native registry directory to export.")
    parser.add_argument("--output", default="workloads.yaml", help="Compatibility flat registry YAML path.")
    parser.add_argument("--check", action="store_true", help="Verify workloads.yaml is current without writing.")
    args = parser.parse_args()

    workloads = load_registry(args.source)
    content = dump_yaml(native_to_flat(workloads))
    output = Path(args.output)

    if args.check:
        if not output.exists():
            print(f"missing {output}")
            print("run: python3 tools/export_flat_registry.py")
            return 1
        if output.read_text(encoding="utf-8") != content:
            print(f"stale {output}")
            print("run: python3 tools/export_flat_registry.py")
            return 1
        print(f"{output} is current ({len(workloads)} workload(s))")
        return 0

    output.write_text(content, encoding="utf-8")
    print(f"wrote {len(workloads)} workload(s) to {output}")
    return 0


def native_to_flat(workloads: dict[str, Workload]) -> dict[str, Any]:
    suites: dict[str, dict[str, Any]] = {suite: {} for suite in PRODUCT_SUITES}
    for workload in workloads.values():
        suites.setdefault(workload.suite, {})[workload.id] = workload.raw
    return {"suites": {suite: entries for suite, entries in suites.items() if entries}}


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    raise SystemExit(main())
