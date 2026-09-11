#!/usr/bin/env python3
"""Substitute {{stats.*}} placeholders in a rendered Quarto site.

Runs as a Quarto post-render step for any site in this repo. Working on the built HTML rather
than on the .qmd source means one mechanism covers every context a number
appears in: body prose, attribute values such as iframe title=, and the <text>
nodes inside inline SVG mockups. A span-and-hydrate approach would reach the
first two but not the third, and would flash empty before scripts run.

Placeholder syntax:  {{stats.key}}       e.g. {{stats.stars}}
                     {{stats.vol2_ch.fault_tolerance}}

An unresolved placeholder is a build failure. Shipping a literal "{{stats.foo}}"
to a reader is worse than any stale number, so a typo in a key stops the build
instead of leaking markup.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Shared across every Quarto site in the repo. The values themselves stay in
# one place: site/config/stats-cache.json, written by site/scripts/build_stats.py
# and refreshed on a schedule by .github/workflows/site-refresh-stats.yml. A
# second cache would mean two numbers for one fact.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = REPO_ROOT / "site" / "config" / "stats-cache.json"

# Quarto runs post-render from the project directory and names the output
# directory in the environment, so the same script serves any site without
# knowing which one invoked it.
_out = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
BUILD_DIR = (Path.cwd() / _out).resolve() if _out else (Path.cwd() / "_build").resolve()

PLACEHOLDER = re.compile(r"\{\{stats\.([A-Za-z0-9_.]+)\}\}")


def main() -> int:
    if not CACHE_PATH.is_file():
        print(f"error: {CACHE_PATH} missing; run build_stats.py first", file=sys.stderr)
        return 1
    if not BUILD_DIR.is_dir():
        print(f"error: {BUILD_DIR} missing; nothing rendered", file=sys.stderr)
        return 1

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    display: dict[str, str] = cache.get("display", {})
    generated: str = cache.get("generated", "")

    unresolved: dict[str, set[str]] = {}
    substitutions = 0
    touched = 0

    # search.json carries the same prose as the pages, extracted before this
    # step runs. Skipping it leaves raw {{stats.*}} tokens in the site search
    # index, so a reader searching "stars" gets the placeholder as a result.
    targets = sorted(
        set(BUILD_DIR.rglob("*.html")) | set(BUILD_DIR.rglob("*.json"))
    )

    for path in targets:
        original = path.read_text(encoding="utf-8")
        if "{{stats." not in original:
            continue

        # Values are substituted into JSON string literals, so they must be
        # escaped the way JSON expects rather than dropped in raw.
        is_json = path.suffix == ".json"
        missing: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in display:
                missing.add(key)
                return match.group(0)
            value = display[key]
            if is_json:
                return json.dumps(value)[1:-1]
            # Wrap body-text values so the page can refresh them at load time
            # from stats.json. Values inside an attribute or an SVG <text> node
            # cannot carry a wrapper element, so those stay static; they are
            # decorative or metadata rather than the figures a reader reads.
            start = match.start()
            in_attr = original.rfind("<", 0, start) > original.rfind(">", 0, start)
            in_svg_text = original.rfind("<text", 0, start) > original.rfind("</text", 0, start)
            if in_attr or in_svg_text:
                return value
            return f'<span data-stat="{key}">{value}</span>' 

        updated, count = PLACEHOLDER.subn(replace, original)
        if missing:
            unresolved[str(path.relative_to(BUILD_DIR))] = missing
        if count:
            path.write_text(updated, encoding="utf-8")
            substitutions += count
            touched += 1

    # Publish the values at a stable URL. A scheduled job can refresh this one
    # file without rebuilding the site, and every page load picks it up.
    (BUILD_DIR / "stats.json").write_text(
        json.dumps({"generated": generated, "display": display},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print(f"Injected {substitutions} stat value(s) across {touched} page(s); "
          f"wrote stats.json", file=sys.stderr)

    if unresolved:
        print("\nerror: unresolved stat placeholders", file=sys.stderr)
        for page, keys in sorted(unresolved.items()):
            for key in sorted(keys):
                print(f"  {page}: {{{{stats.{key}}}}}", file=sys.stderr)
        print("\nAdd the key in build_stats.py or fix the typo in the .qmd.",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
