#!/usr/bin/env python3
"""Reuse the community adopter data and styles in the canonical course directory."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

COURSES = Path(__file__).resolve().parents[1]
ROOT = COURSES.parent
SITE = ROOT / "site"


def prepare():
    generated = COURSES / "_generated"
    generated.mkdir(exist_ok=True)
    shutil.copy2(SITE / "community/community.css", generated / "community.css")


def publish():
    spec = importlib.util.spec_from_file_location("course_adopters", SITE / "scripts/build_adopters.py")
    adopters = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adopters)
    data = adopters.payload()
    build = COURSES / "_build"
    page = build / "index.html"
    html = page.read_text(encoding="utf-8")
    marker = "<!-- adopters:courses -->"
    end_marker = "<!-- /adopters:courses -->"
    if marker not in html:
        raise RuntimeError("The course directory is missing its adopter-data marker")
    embedded = ('<script type="application/json" id="adp-data">'
                + json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
                + "</script>\n" + adopters.land_svg())
    start = html.index(marker) + len(marker)
    end = html.find(end_marker, start)
    if end == -1:
        # A fresh Quarto render contains only the opening marker.
        html = html[:start] + "\n" + embedded + "\n" + end_marker + html[start:]
    else:
        # Selective renders can leave the already-populated directory intact.
        html = html[:start] + "\n" + embedded + "\n" + html[end:]
    page.write_text(html, encoding="utf-8")
    (build / "adopters.json").write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copytree(SITE / "community/assets/images/adopters",
                    build / "assets/images/adopters", dirs_exist_ok=True)
    print("Course directory: embedded canonical adopter data and map")


if __name__ == "__main__":
    prepare() if sys.argv[1:] == ["prepare"] else publish()
