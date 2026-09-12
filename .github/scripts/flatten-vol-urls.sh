#!/usr/bin/env bash
# flatten-vol-urls.sh — Publish a volume at clean chapter URLs.
#
# Quarto renders a volume's pages under the volume's own source folder:
#   html-vol{N}/vol{N}/{chapter}/{chapter}.html
# which would deploy as:
#   mlsysbook.ai/vol{N}/vol{N}/{chapter}/{chapter}.html
#
# This script moves everything under vol{N}/ up to the site root, so pages
# deploy at:
#   mlsysbook.ai/vol{N}/{chapter}/{chapter}.html
#
# Every relative href/src and inline CSS url() is resolved against the page's
# original location and re-expressed from its new location, so pages at any
# depth stay correct.
# Links to vol{N}/index.qmd (Quarto's sidebar "Homepage") point at the site
# root. search.json and sitemap.xml are rewritten to the clean paths.
#
# Both legacy URL trees keep working as static redirects to the clean pages:
#   /vol{N}/contents/vol{N}/...  the published form before URLs were flattened
#   /vol{N}/vol{N}/...           the unflattened render layout
# Chapter folders carry a two-digit order prefix (08_training). The unprefixed
# chapter URL (/vol{N}/training/training.html, in all three trees) redirects to
# the numbered page, so links published before that rename keep working.
#
# The script fails if the site has no vol{N}/ tree to flatten, and it fails
# before deployment if any generated link still targets a redirect tree.
# Re-running it on an already flattened site does nothing.
#
# Usage:
#   flatten-vol-urls.sh <site-dir> <vol>
#   flatten-vol-urls.sh ./vol1-site vol1

set -euo pipefail

SITE_DIR="${1:?Usage: flatten-vol-urls.sh <site-dir> <vol>}"
VOL="${2:?Usage: flatten-vol-urls.sh <site-dir> <vol>}"

echo "🔧 Flattening $VOL: $VOL/* → site root..."
python3 - "$SITE_DIR" "$VOL" <<'PY'
from __future__ import annotations

import html
import posixpath
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

site = Path(sys.argv[1]).resolve()
vol = sys.argv[2]
nested = site / vol
BASE = "https://mlsysbook.ai"
REDIRECT_TITLE = "<title>Redirecting...</title>"
STUB_ROOTS = (f"{vol}/", f"contents/{vol}/")

ATTR = re.compile(r"""(\s(?:href|src|action|poster)=)(["'])(.*?)\2""", re.S)
CSS_URL = re.compile(r"""url\((["']?)([^"')\s]+)\1\)""")
NOT_RELATIVE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|/|#)")

if not nested.is_dir():
    sys.exit(f"error: {nested} does not exist; the render layout has changed, "
             f"so there is no {vol}/ tree to flatten.")

pages_under_nested = sorted(nested.rglob("*.html"))
if pages_under_nested and all(
    REDIRECT_TITLE in p.read_text(encoding="utf-8", errors="replace") for p in pages_under_nested
):
    print(f"  {vol} is already flattened; nothing to do.")
    sys.exit(0)


def flat(rel: str) -> str:
    """Site-relative path of a file once vol/ has been moved to the root."""
    if rel == vol:
        return ""
    if rel.startswith(vol + "/"):
        return rel[len(vol) + 1:]
    return rel


def split_url(url: str) -> tuple[str, str]:
    m = re.match(r"([^?#]*)(.*)", url, re.S)
    return m.group(1), m.group(2)


def site_target(page_dir: str, path: str) -> str | None:
    """Site-relative target of a relative path, or None if it leaves the site."""
    target = posixpath.normpath(posixpath.join(page_dir, path))
    if target == "..":
        return None
    if target.startswith("../"):
        return None
    return "" if target == "." else target


def rewrite(text: str, old_rel: str) -> str:
    old_dir = posixpath.dirname(old_rel)
    new_dir = posixpath.dirname(flat(old_rel))
    moved = new_dir != old_dir

    def relink(url: str) -> str:
        if not url or NOT_RELATIVE.match(url):
            return url
        path, suffix = split_url(url)
        if not path:
            return url
        target = site_target(old_dir, path)
        if target is None:
            return url
        dir_link = path.endswith("/")
        if target == "index.qmd" or target.endswith("/index.qmd"):
            target, dir_link = posixpath.dirname(target), True
        mapped = flat(target)
        if not moved and mapped == target and not dir_link:
            return url
        rel = posixpath.relpath(mapped or ".", new_dir or ".")
        if rel == ".":
            rel = "./"
        elif dir_link and not rel.endswith("/"):
            rel += "/"
        if path.startswith("./") and not rel.startswith("."):
            rel = "./" + rel
        return rel + suffix

    def fix(m: re.Match) -> str:
        lead, q, url = m.group(1), m.group(2), m.group(3)
        new = relink(url)
        return m.group(0) if new == url else f"{lead}{q}{new}{q}"

    # 2026-09-12: inline <style> blocks carry url() references (callout icons)
    # that break when pages move up a level, just like href/src.
    def fix_css(m: re.Match) -> str:
        q, url = m.group(1), m.group(2)
        new = relink(url)
        return m.group(0) if new == url else f"url({q}{new}{q})"

    text = ATTR.sub(fix, text)
    text = CSS_URL.sub(fix_css, text)
    for legacy in (f"{BASE}/{vol}/contents/{vol}/", f"{BASE}/{vol}/{vol}/"):
        text = text.replace(legacy, f"{BASE}/{vol}/")
    return text


# 1. Rewrite links in every page while it still sits at its original location.
for page in sorted(site.rglob("*.html")):
    rel = page.relative_to(site).as_posix()
    text = page.read_text(encoding="utf-8")
    updated = rewrite(text, rel)
    if updated != text:
        page.write_text(updated, encoding="utf-8")

# 2. Move vol/ up to the site root. The root wins on a name collision
#    (vol/index.html is the sidebar "Homepage" entry; the real home is index.html).
legacy_pages = [p.relative_to(nested).as_posix() for p in pages_under_nested]
moved = 0
for path in sorted(nested.rglob("*")):
    if path.is_dir():
        continue
    dest = site / path.relative_to(nested)
    if dest.exists():
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), dest)
    moved += 1
shutil.rmtree(nested)
print(f"  Moved {moved} files from {vol}/ to the site root.")

# 3. search.json and sitemap.xml
search = site / "search.json"
if search.exists():
    t = search.read_text(encoding="utf-8")
    t = re.sub(r'("href"\s*:\s*")(?:contents/)?' + re.escape(vol) + r"/", r"\1", t)
    search.write_text(t, encoding="utf-8")
sitemap = site / "sitemap.xml"
if sitemap.exists():
    t = sitemap.read_text(encoding="utf-8")
    for legacy in (f"/{vol}/contents/{vol}/", f"/{vol}/{vol}/"):
        t = t.replace(legacy, f"/{vol}/")
    sitemap.write_text(t, encoding="utf-8")


# 4. Redirect pages at both legacy trees.
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
  {REDIRECT_TITLE}
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


stubs = 0
for tree in (vol, f"contents/{vol}"):
    for rel in legacy_pages:
        alias = site / tree / rel
        if alias.exists():
            continue
        target_rel = "" if rel == "index.html" else rel
        target_url = posixpath.relpath(target_rel or ".", posixpath.dirname(f"{tree}/{rel}"))
        if target_rel == "":
            target_url += "/"
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(redirect_page(quote(target_url, safe="/#?=&.")), encoding="utf-8")
        stubs += 1
print(f"  Wrote {stubs} redirect pages under {vol}/ and contents/{vol}/.")

# 4b. Chapter folders carry a two-digit reading-order prefix
#     (08_training/08_training.html). Links published before that rename use the
#     unprefixed name, so the unprefixed form in all three trees redirects to
#     the numbered clean page.
CHAPTER_PREFIX = re.compile(r"^\d{2}_(.+)$")
alias_roots: set[str] = set()
unprefixed = 0
for rel in legacy_pages:
    parts = rel.split("/")
    m = CHAPTER_PREFIX.match(parts[0])
    if not m:
        continue
    bare = m.group(1)
    tail = [f"{bare}.html" if p == f"{parts[0]}.html" else p for p in parts[1:]]
    alias_rel = "/".join([bare, *tail])
    alias_roots.add(f"{bare}/")
    for tree in ("", vol, f"contents/{vol}"):
        alias_path = f"{tree}/{alias_rel}" if tree else alias_rel
        alias = site / alias_path
        if alias.exists():
            continue
        target_url = posixpath.relpath(rel, posixpath.dirname(alias_path))
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(redirect_page(quote(target_url, safe="/#?=&.")), encoding="utf-8")
        unprefixed += 1
print(f"  Wrote {unprefixed} redirect pages for unprefixed chapter URLs.")

# 5. Fail before deployment if any page still links into a redirect tree.
failures: list[str] = []
for page in sorted(site.rglob("*.html")):
    rel = page.relative_to(site).as_posix()
    if rel.startswith(STUB_ROOTS):
        continue
    page_dir = posixpath.dirname(rel)
    for line_no, line in enumerate(page.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for m in ATTR.finditer(line):
            url = m.group(3)
            if f"{BASE}/{vol}/{vol}/" in url or f"{BASE}/{vol}/contents/{vol}/" in url:
                failures.append(f"{rel}:{line_no}: {url}")
                continue
            if not url or NOT_RELATIVE.match(url):
                continue
            path, _ = split_url(url)
            target = site_target(page_dir, path) if path else None
            if target is not None and (
                target == vol
                or target.startswith(STUB_ROOTS)
                or any(target.startswith(root) for root in alias_roots)
            ):
                failures.append(f"{rel}:{line_no}: {url}")
if failures:
    print(f"error: {len(failures)} link(s) still target the {vol}/ redirect trees:", file=sys.stderr)
    for f in failures[:50]:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
PY

echo "✅ $VOL flattened — chapter URLs now at mlsysbook.ai/$VOL/{chapter}/{chapter}.html"
