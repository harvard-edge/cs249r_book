#!/usr/bin/env python3
"""
Generate navbar YAML for subsites by merging:
  1. The shared navbar (ecosystem-wide dropdowns + right side)
  2. An optional site-local dropdown (e.g., "Lecture Slides" for slides)

Usage:
    python3 .github/scripts/generate-navbar.py --all          # Regenerate ALL subsites
    python3 .github/scripts/generate-navbar.py slides          # Single subsite
    python3 .github/scripts/generate-navbar.py mlsysim/docs    # Nested path

Each subsite needs:
    _quarto.yml          — must include `metadata-files: [_navbar-generated.yml]`
    _navbar-local.yml    — (optional) site-specific dropdown to insert after "Read"
    _navbar-generated.yml — OUTPUT: complete merged navbar (committed to repo)

To update the navbar everywhere:
    1. Edit book/quarto/config/shared/html/navbar-common.yml
    2. Run: python3 .github/scripts/generate-navbar.py --all
    3. Commit all _navbar-generated.yml files
"""

import sys
import yaml
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHARED_NAVBAR = REPO_ROOT / "book" / "quarto" / "config" / "shared" / "html" / "navbar-common.yml"

# All subsites that get a generated navbar.
# Key = directory relative to repo root. Value = has local dropdown.
SUBSITES = [
    "slides",
    "instructors",
    "mlsysim/docs",
    "kits",
    "labs",
    "newsletter",
]


def generate_one(subsite_rel: str) -> None:
    subsite_dir = REPO_ROOT / subsite_rel

    if not subsite_dir.is_dir():
        print(f"⚠️  Skipping {subsite_rel} (directory not found)")
        return

    # Load shared navbar
    with open(SHARED_NAVBAR) as f:
        shared = yaml.safe_load(f)
    navbar = deepcopy(shared["website"]["navbar"])

    # Check for site-local dropdown
    local_path = subsite_dir / "_navbar-local.yml"
    if local_path.exists():
        with open(local_path) as f:
            local_entry = yaml.safe_load(f)
        if local_entry and "left" in navbar:
            navbar["left"].insert(1, local_entry)

    # Write generated navbar
    output = {"website": {"navbar": navbar}}
    output_path = subsite_dir / "_navbar-generated.yml"
    with open(output_path, "w") as f:
        f.write("# AUTO-GENERATED — do not edit manually.\n")
        f.write("# Source: book/quarto/config/shared/html/navbar-common.yml\n")
        if local_path.exists():
            f.write(f"# Local:  {subsite_rel}/_navbar-local.yml\n")
        f.write("# Regen:  python3 .github/scripts/generate-navbar.py --all\n\n")
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    local_info = f" + {subsite_rel}/_navbar-local.yml" if local_path.exists() else ""
    print(f"  ✅ {subsite_rel}/_navbar-generated.yml{local_info}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--all":
        print(f"🧭 Generating navbars for {len(SUBSITES)} subsites...")
        for subsite in SUBSITES:
            generate_one(subsite)
        print("Done.")
    else:
        generate_one(sys.argv[1])


if __name__ == "__main__":
    main()
