#!/usr/bin/env python3
"""Aggregate per-subsite sitemap.xml files into a single root-level
sitemap-index.xml at mlsysbook.ai/sitemap.xml.

Why aggregate instead of one-sitemap-per-subsite?
-------------------------------------------------
Each subsite (Vol I, Vol II, TinyTorch, labs, …) emits its own
`sitemap.xml` at `<subsite>/sitemap.xml` because Quarto/Next produce
those automatically. That works, but search engines need a single
authoritative entry point per *domain*. Sitemap *indexes* are the
correct primitive for this: one root file at
`https://mlsysbook.ai/sitemap.xml` that points to each subsite's
sitemap.xml as `<sitemap><loc>...</loc></sitemap>` entries.

Behavior
--------
This script takes a deployed gh-pages tree as input, finds every
sitemap.xml under it (one per subsite), and writes a single
`sitemap.xml` at the tree root containing a sitemap-index pointing to
every per-subsite sitemap. It also emits a `robots.txt` (or appends to
an existing one) that surfaces the index.

Excludes:
  - the root sitemap-index itself (no recursion)
  - `legacy-backup/**` (rollback snapshots are not crawl targets)
  - `_archive/**`, `_drafts/**`, `_site/**` (build artifacts that
    occasionally leak into deploys)

Usage
-----
  build-sitemap.py --root path/to/gh-pages/tree \\
                   --base-url https://mlsysbook.ai \\
                   [--include-subsite vol1 --include-subsite vol2 ...]
                   [--check]

  --include-subsite  Optional allowlist. If passed, only sub-sitemaps under
                     these top-level paths will be aggregated. Default is
                     "every sitemap.xml found under root, minus exclusions".
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

SKIP_PATH_PARTS = {"legacy-backup", "_archive", "_drafts", "_site", ".git"}

INDEX_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
)
INDEX_FOOTER = "</sitemapindex>\n"


def discover_sitemaps(
    root: Path,
    include: list[str] | None,
) -> list[Path]:
    """Walk `root` and return every sitemap.xml that should be indexed,
    skipping the root one and known-excluded subtrees."""
    found: list[Path] = []
    for path in root.rglob("sitemap.xml"):
        # Never include the root-level sitemap (we're about to overwrite it)
        if path == root / "sitemap.xml":
            continue
        rel_parts = path.relative_to(root).parts
        # Skip excluded subtrees
        if any(part in SKIP_PATH_PARTS for part in rel_parts):
            continue
        # Apply allowlist if given (top-level path part must match)
        if include and rel_parts[0] not in set(include):
            continue
        found.append(path)
    found.sort()
    return found


def write_root_index(
    root: Path,
    base_url: str,
    sitemaps: list[Path],
) -> Path:
    """Write `<root>/sitemap.xml` as a sitemap-index pointing to each
    discovered per-subsite sitemap."""
    base = base_url.rstrip("/")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [INDEX_HEADER]
    for sm in sitemaps:
        rel = sm.relative_to(root).as_posix()
        loc = f"{base}/{rel}"
        lines.append("  <sitemap>\n")
        lines.append(f"    <loc>{loc}</loc>\n")
        lines.append(f"    <lastmod>{now}</lastmod>\n")
        lines.append("  </sitemap>\n")
    lines.append(INDEX_FOOTER)

    target = root / "sitemap.xml"
    target.write_text("".join(lines), encoding="utf-8")
    return target


def update_robots_txt(root: Path, base_url: str) -> Path:
    """Ensure `<root>/robots.txt` exists and surfaces the sitemap-index."""
    robots = root / "robots.txt"
    sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"
    line = f"Sitemap: {sitemap_url}\n"
    if robots.exists():
        existing = robots.read_text(encoding="utf-8")
        if sitemap_url in existing:
            return robots
        # Append (preserving any User-agent directives already present)
        if not existing.endswith("\n"):
            existing += "\n"
        robots.write_text(existing + line, encoding="utf-8")
    else:
        robots.write_text("User-agent: *\nAllow: /\n\n" + line, encoding="utf-8")
    return robots


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="Deployed gh-pages tree root")
    ap.add_argument(
        "--base-url",
        default="https://mlsysbook.ai",
        help="Public base URL (default: https://mlsysbook.ai)",
    )
    ap.add_argument(
        "--include-subsite",
        action="append",
        default=None,
        help="Allowlist a subsite by top-level dir name. May be repeated. "
        "If omitted, every discovered sitemap.xml is indexed.",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Discover and report sitemaps; do not write anything.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"❌ --root '{root}' is not a directory", file=sys.stderr)
        return 2

    sitemaps = discover_sitemaps(root, args.include_subsite)
    if not sitemaps:
        print("❌ No subsite sitemap.xml files found under root.", file=sys.stderr)
        print("   Subsites are expected to publish their own sitemap.xml at", file=sys.stderr)
        print("   <subsite>/sitemap.xml during build.", file=sys.stderr)
        return 1

    print(f"📚 Discovered {len(sitemaps)} subsite sitemap(s):")
    for sm in sitemaps:
        print(f"   - {sm.relative_to(root)}")

    if args.check:
        return 0

    index_path = write_root_index(root, args.base_url, sitemaps)
    robots_path = update_robots_txt(root, args.base_url)
    print(f"✅ Wrote {index_path.relative_to(root)} (sitemap index)")
    print(f"✅ Updated {robots_path.relative_to(root)} (Sitemap: directive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
