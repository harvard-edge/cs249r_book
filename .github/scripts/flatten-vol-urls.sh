#!/usr/bin/env bash
# flatten-vol-urls.sh — Flatten Quarto's nested vol URL structure at deploy time.
#
# Quarto renders chapters at:
#   html-vol{N}/contents/vol{N}/{chapter}/{chapter}.html   (depth 3)
# which deploys to:
#   mlsysbook.ai/vol{N}/contents/vol{N}/{chapter}/{chapter}.html
#
# This script moves all content under contents/vol{N}/ up to the site root so
# deployed URLs become the clean form:
#   mlsysbook.ai/vol{N}/{chapter}/{chapter}.html
#
# Why relative paths need fixing:
#   Quarto generates ../../../site_libs/ (3 levels up from contents/vol{N}/chapter/).
#   After the move to {chapter}/ (depth 1), those same assets are at ../site_libs/.
#   The generated sidebar, breadcrumbs, and next/prev links also point at the
#   pre-flatten contents/vol{N}/ paths, so after moving files we rewrite links
#   across the full prepared volume site.
#
# Usage:
#   flatten-vol-urls.sh <site-dir> <vol>
#   flatten-vol-urls.sh ./vol1-site vol1
#   flatten-vol-urls.sh ./vol2-site vol2

set -euo pipefail

SITE_DIR="${1:?Usage: flatten-vol-urls.sh <site-dir> <vol>}"
VOL="${2:?Usage: flatten-vol-urls.sh <site-dir> <vol>}"

NESTED_DIR="$SITE_DIR/contents/$VOL"

if [ ! -d "$NESTED_DIR" ]; then
  echo "ℹ️  $NESTED_DIR not found — nothing to flatten for $VOL"
  exit 0
fi

echo "🔧 Flattening $VOL: contents/$VOL/* → site root..."

# Step 1: Fix 3-level relative asset paths → 1-level in files being moved.
# Chapters are at depth 3 (contents/vol{N}/chapter/); after the move they're
# at depth 1 ({chapter}/). All Quarto-generated relative refs use exactly 3
# levels of ../../../  since every rendered file is at the same depth.
echo "  Fixing relative paths (../../../ → ../) in moved files..."
python3 - "$NESTED_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("*.html"):
    text = path.read_text(encoding="utf-8")
    updated = text.replace("../../../", "../")
    if updated != text:
        path.write_text(updated, encoding="utf-8")
PY

# Step 2: Fix absolute https://mlsysbook.ai/{vol}/contents/{vol}/ refs.
# These appear in og:image, twitter:image, and any full-URL inline refs.
echo "  Fixing absolute URL refs in moved files..."
python3 - "$NESTED_DIR" "$VOL" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
vol = sys.argv[2]
old = f"https://mlsysbook.ai/{vol}/contents/{vol}/"
new = f"https://mlsysbook.ai/{vol}/"
for path in root.rglob("*.html"):
    text = path.read_text(encoding="utf-8")
    updated = text.replace(old, new)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
PY

# Step 3: Move all content from contents/vol{N}/ up to the site root.
# --ignore-existing: the real homepage is at root index.html; any
# contents/vol{N}/index.html (the sidebar "Homepage" entry) is skipped.
echo "  Moving contents/$VOL/* → site root..."
rsync -a --ignore-existing "$NESTED_DIR/" "$SITE_DIR/"

# Step 4: Fix search.json hrefs (root-relative, no leading slash).
# Before: "contents/vol{N}/chapter/chapter.html"
# After:  "chapter/chapter.html"
if [ -f "$SITE_DIR/search.json" ]; then
  echo "  Fixing search.json hrefs..."
  python3 - "$SITE_DIR/search.json" "$VOL" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
vol = sys.argv[2]
text = path.read_text(encoding="utf-8")
path.write_text(text.replace(f'"contents/{vol}/', '"'), encoding="utf-8")
PY
fi

# Step 5: Fix sitemap.xml (full absolute URLs).
# Before: https://mlsysbook.ai/vol{N}/contents/vol{N}/...
# After:  https://mlsysbook.ai/vol{N}/...
if [ -f "$SITE_DIR/sitemap.xml" ]; then
  echo "  Fixing sitemap.xml..."
  python3 - "$SITE_DIR/sitemap.xml" "$VOL" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
vol = sys.argv[2]
text = path.read_text(encoding="utf-8")
path.write_text(text.replace(f"/{vol}/contents/{vol}/", f"/{vol}/"), encoding="utf-8")
PY
fi

# Step 6: Replace the old nested tree with compatibility redirects.
# Cached pages, old bookmarks, and search results may still point at
# /vol{N}/contents/vol{N}/... after a release. Keep those paths alive as
# lightweight static redirects to the clean canonical chapter URLs.
echo "  Creating compatibility redirects for old contents/$VOL paths..."
python3 - "$SITE_DIR" "$NESTED_DIR" "$VOL" <<'PY'
from __future__ import annotations

import html
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

site_dir = Path(sys.argv[1])
nested_dir = Path(sys.argv[2])
vol = sys.argv[3]

legacy_pages = sorted(path.relative_to(nested_dir) for path in nested_dir.rglob("*.html"))

shutil.rmtree(nested_dir)
nested_dir.mkdir(parents=True, exist_ok=True)


def relative_target(alias_path: Path, rel: Path) -> str:
    if rel.name == "index.html":
        target = site_dir
    else:
        target = site_dir / rel
    return quote(os.path.relpath(target, start=alias_path.parent), safe="/#?=&")


def redirect_page(target: str) -> str:
    escaped = html.escape(target, quote=True)
    script_target = target.replace("\\", "\\\\").replace('"', '\\"')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={escaped}">
  <link rel="canonical" href="{escaped}">
  <meta name="robots" content="noindex">
  <title>Redirecting...</title>
  <script>
    const target = new URL("{script_target}", window.location.href);
    target.search = window.location.search;
    target.hash = window.location.hash;
    window.location.replace(target.href);
  </script>
</head>
<body>
  <p>Redirecting to <a href="{escaped}">the current {html.escape(vol)} page</a>.</p>
</body>
</html>
"""


for rel in legacy_pages:
    alias_path = nested_dir / rel
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_text(redirect_page(relative_target(alias_path, rel)), encoding="utf-8")

print(f"  Created {len(legacy_pages)} compatibility redirect pages.")
PY

# Step 7: Rewrite generated links across the whole prepared volume site.
# Root pages need ./chapter/..., pages one directory deep need ../chapter/...,
# and deeper pages need the corresponding parent prefix. The shared SocratiQ
# page intentionally remains under contents/frontmatter/ unless the build
# manifest changes.
echo "  Rewriting generated sidebar/breadcrumb links..."
python3 - "$SITE_DIR" "$VOL" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

site_dir = Path(sys.argv[1])
vol = sys.argv[2]

volume_path = re.compile(r'(?:(?:\./|\.\./)+)?contents/' + re.escape(vol) + r'/')
volume_index = re.compile(r'(?:(?:\./|\.\./)+)?contents/' + re.escape(vol) + r'/index\.qmd')
shared_frontmatter = re.compile(r'(?:(?:\./|\.\./)+)?contents/frontmatter/')


def prefix_for(path: Path) -> str:
    rel_parent = path.relative_to(site_dir).parent
    if str(rel_parent) == ".":
        return "./"
    return "../" * len(rel_parent.parts)


for path in site_dir.rglob("*.html"):
    text = path.read_text(encoding="utf-8")
    prefix = prefix_for(path)
    updated = text
    updated = updated.replace(
        f"https://mlsysbook.ai/{vol}/contents/{vol}/",
        f"https://mlsysbook.ai/{vol}/",
    )
    updated = volume_index.sub(prefix, updated)
    updated = volume_path.sub(prefix, updated)
    updated = shared_frontmatter.sub(prefix + "contents/frontmatter/", updated)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
PY

# Step 8: Fail before deployment if any generated link still targets the
# removed contents/vol{N}/ tree.
echo "  Checking for stale contents/$VOL links..."
python3 - "$SITE_DIR" "$VOL" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

site_dir = Path(sys.argv[1])
vol = sys.argv[2]
stale_attr = re.compile(r'(?:href|src|action)=["\'][^"\']*contents/' + re.escape(vol) + r'/')
failures: list[str] = []

for path in site_dir.rglob("*.html"):
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if stale_attr.search(line):
            failures.append(f"{path.relative_to(site_dir)}:{line_no}: {line.strip()[:220]}")

if failures:
    print(f"stale contents/{vol}/ links remain after flatten:", file=sys.stderr)
    for failure in failures[:50]:
        print(f"  {failure}", file=sys.stderr)
    if len(failures) > 50:
        print(f"  ... {len(failures) - 50} more", file=sys.stderr)
    sys.exit(1)
PY

echo "✅ $VOL flattened — chapter URLs now at mlsysbook.ai/$VOL/{chapter}/{chapter}.html"
