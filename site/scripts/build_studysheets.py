#!/usr/bin/env python3
"""Write the chapter list for the Study Sheets page (site/studysheets/index.qmd).

The list comes from the study-sheet manifest (books/shared/studysheet/build.py --manifest),
so the page lists exactly the volumes and chapters published to the studysheets-latest
release. Only LIVE_VOLUMES appear. Output: site/studysheets/_sheets.md (generated, gitignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
BUILD = REPO / "books/shared/studysheet/build.py"
OUT = SITE / "studysheets/_sheets.md"
ROMAN = {"vol1": "I", "vol2": "II", "vol3": "III", "vol4": "IV"}


def volume_title(vol: str) -> str:
    config = yaml.safe_load((REPO / f"books/config/_quarto-html-{vol}.yml").read_text())
    return config["website"]["title"]


def main() -> None:
    live = subprocess.run([sys.executable, str(BUILD), "--live-volumes"], check=True,
                          capture_output=True, text=True).stdout.split()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        subprocess.run([sys.executable, str(BUILD), "--manifest", str(path)], check=True,
                       capture_output=True)
        manifest = json.loads(path.read_text())

    lines: list[str] = []
    for vol in live:
        info = manifest[vol]["_volume"]
        lines += [f"## Volume {ROMAN[vol]}: {volume_title(vol)} {{#{vol}}}", ""]
        lines += [f"[All {info['count']} sheets in one PDF]({info['url']}) ({info['pages']} pages)"
                  f" · [Read Volume {ROMAN[vol]} online](https://mlsysbook.ai/{vol}/)", ""]
        # Nested like the book's table of contents: unparted chapters (introduction,
        # conclusion) at the top level, every other chapter under its part.
        last = None
        for n, sheet in enumerate(info["sheets"], 1):
            link = f"[{n}. {sheet['title']}]({sheet['url']})"
            if not sheet["part"]:
                lines.append(f"- {link}")
            else:
                if sheet["part"] != last:
                    lines.append(f"- {sheet['part']}")
                lines.append(f"    - {link}")
            last = sheet["part"]
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"build_studysheets.py: {', '.join(live)} -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
