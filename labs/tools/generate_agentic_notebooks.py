"""Generate thin Volume III entrypoints from one reactive template and concept maps."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).parent / "templates" / "agentic_lab.py.tmpl"


def generate(*, check: bool = False) -> list[Path]:
    template = TEMPLATE.read_text(encoding="utf-8")
    mismatches = []
    for source in sorted((ROOT / "labs/vol3/concepts").glob("*_concepts.yml")):
        contract = yaml.safe_load(source.read_text(encoding="utf-8"))["lab_concept_map"]
        target = ROOT / contract["source_notebook"]
        rendered = template.replace("LAB_TITLE", contract["title"]).replace("CHAPTER_NUMBER", str(contract["chapter"])).replace("NOTEBOOK_KEY", contract["source_notebook"].removeprefix("labs/"))
        if not target.exists() or target.read_text(encoding="utf-8") != rendered:
            mismatches.append(target)
            if not check:
                target.write_text(rendered, encoding="utf-8")
    return mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    differences = generate(check=args.check)
    if args.check and differences:
        for path in differences:
            print(f"Generated notebook differs: {path.relative_to(ROOT)}")
        sys.exit(1)
    print("Agentic notebook template check passed." if args.check else f"Updated {len(differences)} agentic entrypoints.")
