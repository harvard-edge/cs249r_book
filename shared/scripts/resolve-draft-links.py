#!/usr/bin/env python3
"""Resolve unpublished reading links in the course and instructor projects."""

from pathlib import Path
import os
import re
import sys

DRAFT_SITE = "https://harvard-edge.github.io/cs249r_book_dev/"


def resolve_links(build, project):
    for page in build.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        for prefix in ("vol3/", "vol4/", "labs/vol4/"):
            html = html.replace("https://mlsysbook.ai/" + prefix, DRAFT_SITE + prefix)

        def local_resource(match):
            section, path = match[1], match[2]
            target, separator, fragment = path.partition("#")
            destination = Path(section) / target
            if not destination.suffix:
                destination /= "index.html"
            parent = Path(project) / page.relative_to(build).parent
            relative = Path(os.path.relpath(destination, parent)).as_posix()
            return f'href="{relative}{separator}{fragment}"'

        html = re.sub(r'href="https://mlsysbook.ai/(courses|instructors)/([^"?]*)"',
                      local_resource, html)
        page.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    resolve_links(Path(sys.argv[1]), sys.argv[2])
