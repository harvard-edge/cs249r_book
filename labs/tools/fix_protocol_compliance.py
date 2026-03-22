#!/usr/bin/env python3
"""
Fix protocol compliance issues across all labs.

Handles:
1. Add ZONE comments where missing
2. Add ledger.save() where missing
3. Add mo.stop() gates where missing

Usage:
  python3 labs/tools/fix_protocol_compliance.py --dry-run   # Preview changes
  python3 labs/tools/fix_protocol_compliance.py              # Apply changes
"""

import re
import sys
import glob
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

REPO_ROOT = Path(__file__).resolve().parents[2]
ALL_LABS = sorted(
    glob.glob(str(REPO_ROOT / "labs" / "vol1" / "lab_*.py"))
    + glob.glob(str(REPO_ROOT / "labs" / "vol2" / "lab_*.py"))
)

SKIP_LABS = {"lab_00_introduction"}


def lab_number(path: str) -> int:
    m = re.search(r"lab_(\d+)_", Path(path).name)
    return int(m.group(1)) if m else -1


def lab_stem(path: str) -> str:
    return Path(path).stem


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def write(path: str, content: str):
    if DRY_RUN:
        print(f"  [dry-run] Would write {path}")
    else:
        with open(path, "w") as f:
            f.write(content)


def fix_zone_comments(path: str, source: str) -> str:
    """Add ZONE A/B/C/D comments if missing."""
    if "ZONE A" in source and "ZONE D" in source:
        return source

    lines = source.split("\n")
    result = []
    added_zones = set()

    # Find key structural points
    first_cell_line = None
    widget_cell_line = None
    tabs_cell_line = None
    last_cell_line = None
    cell_indices = []

    for i, line in enumerate(lines):
        if "@app.cell" in line:
            cell_indices.append(i)

    if len(cell_indices) < 3:
        return source  # Too few cells to structure

    # Strategy:
    # ZONE A = before first cell (or at first cell)
    # ZONE B = at widget cells (prediction radio/slider cells)
    # ZONE C = at the tabs cell (contains mo.ui.tabs)
    # ZONE D = at the last cell (ledger HUD)

    # Find the tabs cell
    tabs_cell_idx = None
    for ci in cell_indices:
        # Look ahead ~100 lines for mo.ui.tabs
        for j in range(ci, min(ci + 200, len(lines))):
            if "mo.ui.tabs" in lines[j]:
                tabs_cell_idx = ci
                break
        if tabs_cell_idx is not None:
            break

    # Find widget cells (cells that define prediction widgets)
    widget_cell_idx = None
    for ci in cell_indices:
        for j in range(ci, min(ci + 30, len(lines))):
            if any(w in lines[j] for w in ["mo.ui.radio", "mo.ui.number", "partA_prediction", "partB_prediction"]):
                widget_cell_idx = ci
                break
        if widget_cell_idx is not None:
            break

    for i, line in enumerate(lines):
        # Add ZONE A before the first @app.cell
        if i == cell_indices[0] and "ZONE A" not in source:
            if i > 0 and "ZONE" not in lines[i - 1]:
                zone_a = [
                    "",
                    "# " + "=" * 75,
                    "# ZONE A: OPENING",
                    "# " + "=" * 75,
                    "",
                ]
                # Don't add if there's already a zone-like comment nearby
                if not any("ZONE" in lines[max(0, i-3):i+1][k] for k in range(min(4, i+1))):
                    result.extend(zone_a)
                    added_zones.add("A")

        # Add ZONE B before widget cells
        if widget_cell_idx and i == widget_cell_idx and "ZONE B" not in source:
            if not any("ZONE" in lines[max(0, i-3):i+1][k] for k in range(min(4, i+1))):
                zone_b = [
                    "",
                    "# " + "=" * 75,
                    "# ZONE B: WIDGET DEFINITIONS",
                    "# " + "=" * 75,
                    "",
                ]
                result.extend(zone_b)
                added_zones.add("B")

        # Add ZONE C before tabs cell
        if tabs_cell_idx and i == tabs_cell_idx and "ZONE C" not in source:
            if not any("ZONE" in lines[max(0, i-3):i+1][k] for k in range(min(4, i+1))):
                zone_c = [
                    "",
                    "# " + "=" * 75,
                    "# ZONE C: SINGLE TABS CELL",
                    "# " + "=" * 75,
                    "",
                ]
                result.extend(zone_c)
                added_zones.add("C")

        # Add ZONE D before last cell (if it's the HUD)
        if i == cell_indices[-1] and "ZONE D" not in source:
            if not any("ZONE" in lines[max(0, i-3):i+1][k] for k in range(min(4, i+1))):
                zone_d = [
                    "",
                    "# " + "=" * 75,
                    "# ZONE D: LEDGER HUD",
                    "# " + "=" * 75,
                    "",
                ]
                result.extend(zone_d)
                added_zones.add("D")

        result.append(line)

    if added_zones:
        print(f"  + Added zones: {', '.join(sorted(added_zones))}")
        return "\n".join(result)
    return source


def fix_ledger_save(path: str, source: str) -> str:
    """Add ledger.save() to the last cell if missing."""
    if "ledger.save" in source:
        return source

    lab_num = lab_number(path)
    vol = "v1" if "vol1" in path else "v2"

    # Find the last @app.cell and its function body
    # We need to insert ledger.save into the HUD cell
    # Look for the lab-hud div pattern
    hud_match = re.search(r'(<div class="lab-hud")', source)
    if not hud_match:
        print(f"  ! No lab-hud div found, cannot add ledger.save")
        return source

    # Insert ledger.save before the HUD HTML
    # Find the function def that contains lab-hud
    lines = source.split("\n")
    hud_line = None
    for i, line in enumerate(lines):
        if 'lab-hud' in line:
            hud_line = i
            break

    if hud_line is None:
        return source

    # Walk backwards to find the def line and determine indentation
    indent = "    "
    for i in range(hud_line, -1, -1):
        if lines[i].strip().startswith("def _"):
            # Get the indentation of the first content line
            for j in range(i + 1, hud_line):
                if lines[j].strip():
                    indent = re.match(r"(\s*)", lines[j]).group(1)
                    break
            break

    save_block = [
        f"{indent}ledger.save(chapter={lab_num}, design={{",
        f'{indent}    "chapter": "{vol}_{lab_num:02d}",',
        f'{indent}    "completed": True,',
        f"{indent}}})",
        f"",
    ]

    # Insert before the mo.Html line that contains lab-hud
    # Walk back from hud_line to find where to insert
    insert_at = hud_line
    for i in range(hud_line, -1, -1):
        if "mo.Html" in lines[i] or 'f"""' in lines[i] or "return" in lines[i]:
            insert_at = i
            break

    result = lines[:insert_at] + save_block + lines[insert_at:]
    print(f"  + Added ledger.save(chapter={lab_num})")
    return "\n".join(result)


def fix_mo_stop(path: str, source: str) -> str:
    """Add mo.stop() gates in prediction widget cells if missing."""
    if "mo.stop(" in source:
        return source

    # Find prediction widget cells (partX_prediction pattern)
    # These cells define radio/number widgets and should gate with mo.stop
    lines = source.split("\n")
    result = []
    added = 0

    i = 0
    while i < len(lines):
        result.append(lines[i])

        # Look for cells that read prediction widget values
        # Patterns: def _(mo, partX_prediction), def _(mo, partX_pred), def _(mo, pX_pred)
        pred_match = re.search(
            r"def _\(.*?((?:part[A-E]_pred(?:iction)?|p[A-E]_pred))\b", lines[i]
        )
        if pred_match:
            pred_var = pred_match.group(1)
            # This is a prediction-reading cell. Find the return statement
            # and add mo.stop before the main content
            # Look ahead for the first content after the def
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("#")):
                result.append(lines[j])
                j += 1

            # Now we're at the first content line — insert mo.stop
            if j < len(lines):
                content_indent = re.match(r"(\s*)", lines[j]).group(1)
                stop_line = (
                    f'{content_indent}mo.stop({pred_var}.value is None, '
                    f'mo.md("**Make your prediction above to unlock this part.**"))'
                )
                result.append(stop_line)
                result.append("")
                added += 1

            i = j
            continue

        i += 1

    if added:
        print(f"  + Added {added} mo.stop() gate(s)")
        return "\n".join(result)
    return source


def main():
    print(f"{'[DRY RUN] ' if DRY_RUN else ''}Fixing protocol compliance across {len(ALL_LABS)} labs\n")

    changes = 0
    for path in ALL_LABS:
        stem = lab_stem(path)
        if stem in SKIP_LABS:
            continue

        source = read(path)
        original = source

        print(f"{stem}:")

        # Apply fixes in order
        source = fix_zone_comments(path, source)
        source = fix_ledger_save(path, source)
        source = fix_mo_stop(path, source)

        if source != original:
            write(path, source)
            changes += 1
        else:
            print("  (no changes needed)")

        print()

    print(f"\n{'Would modify' if DRY_RUN else 'Modified'} {changes} files")


if __name__ == "__main__":
    main()
