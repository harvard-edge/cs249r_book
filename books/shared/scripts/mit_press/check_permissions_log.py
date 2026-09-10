#!/usr/bin/env python3
"""Validate the Volume I figure-permissions log against QMD sources."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
QUARTO_DIR = SCRIPT_DIR.parent.parent
DEFAULT_LOG = SCRIPT_DIR / "PERMISSIONS_FIGURES_VOL1.csv"

SOURCE_TYPES = {
    "original",
    "adapted",
    "used_with_permission",
    "public_domain",
    "cc_licensed",
    "fair_use",
}
PERMISSION_STATES = {"yes", "no", "pending", "na"}
PERMISSION_FIELDS = (
    "permission_print",
    "permission_electronic",
    "permission_world_lang",
)


def load_figure_extractor():
    path = SCRIPT_DIR / "generate_figure_list.py"
    spec = importlib.util.spec_from_file_location("generate_figure_list", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--require-resolved",
        action="store_true",
        help="fail if any permission scope is pending or denied",
    )
    args = parser.parse_args()

    extractor = load_figure_extractor()
    source_figures = extractor.extract_qmd_figures(QUARTO_DIR)
    source_labels = [figure["label"] for figure in source_figures]

    with args.log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    log_labels = [row.get("fig_label", "") for row in rows]

    errors: list[str] = []
    duplicates = sorted({label for label in log_labels if log_labels.count(label) > 1})
    missing = sorted(set(source_labels) - set(log_labels))
    extra = sorted(set(log_labels) - set(source_labels))
    if duplicates:
        errors.append(f"duplicate labels: {', '.join(duplicates)}")
    if missing:
        errors.append(f"missing labels: {', '.join(missing)}")
    if extra:
        errors.append(f"stale labels: {', '.join(extra)}")

    unresolved = 0
    for row_number, row in enumerate(rows, start=2):
        label = row.get("fig_label", "") or f"row {row_number}"
        source_type = row.get("source_type", "")
        if source_type not in SOURCE_TYPES:
            errors.append(f"{label}: invalid source_type {source_type!r}")
        for field in PERMISSION_FIELDS:
            state = row.get(field, "")
            if state not in PERMISSION_STATES:
                errors.append(f"{label}: invalid {field} {state!r}")
            if state in {"pending", "no"}:
                unresolved += 1

    print(
        f"permissions log: {len(rows)} rows; "
        f"source figures: {len(source_labels)}; unresolved scopes: {unresolved}"
    )
    for error in errors:
        print(f"ERROR: {error}")

    if args.require_resolved and unresolved:
        print("ERROR: permission scopes remain unresolved")
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
