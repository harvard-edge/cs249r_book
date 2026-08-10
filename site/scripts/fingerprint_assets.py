#!/usr/bin/env python3
"""Version local CSS and JS references by content hash.

The site serves HTML with max-age=600 but stylesheets with max-age=14400, and
Quarto emits plain references such as `href="landing-v3.css"`. A returning
visitor inside that four-hour window therefore fetches new HTML against a
stale stylesheet. When a release changes markup and CSS together, as the
impact-card rebuild did, the new markup lands with none of its rules and the
page renders as unstyled defaults.

Appending a content hash to each reference makes a changed asset a different
URL, so caches cannot pair new markup with old styles. Unchanged assets keep
their hash and stay cached.

Runs as a Quarto post-render step, after inject_stats.py.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

SITE_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = SITE_DIR / "_build"

# Matches href/src values ending in .css or .js, capturing the quote style so
# the replacement can preserve it.
ASSET_REF = re.compile(
    r'(?P<attr>\b(?:href|src)=)(?P<q>["\'])(?P<url>[^"\']+?\.(?:css|js))(?P=q)'
)

_hashes: dict[Path, str] = {}


def content_hash(path: Path) -> str | None:
    if path in _hashes:
        return _hashes[path]
    try:
        digest = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return None
    _hashes[path] = digest
    return digest


def resolve(url: str, html_path: Path) -> Path | None:
    """Map a reference to a file inside the build, or None if external."""
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc:
        return None  # CDN or other origin; not ours to version
    clean = parsed.path
    if not clean:
        return None
    candidate = (BUILD_DIR / clean.lstrip("/")) if clean.startswith("/") \
        else (html_path.parent / clean)
    try:
        candidate = candidate.resolve()
        candidate.relative_to(BUILD_DIR.resolve())
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def main() -> int:
    if not BUILD_DIR.is_dir():
        print(f"error: {BUILD_DIR} missing", file=sys.stderr)
        return 1

    versioned = 0
    pages = 0

    for html_path in BUILD_DIR.rglob("*.html"):
        original = html_path.read_text(encoding="utf-8")
        count = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal count
            url = match.group("url")
            if "?" in url:  # already carries a query; leave it alone
                return match.group(0)
            target = resolve(url, html_path)
            if target is None:
                return match.group(0)
            digest = content_hash(target)
            if digest is None:
                return match.group(0)
            count += 1
            q = match.group("q")
            return f'{match.group("attr")}{q}{url}?v={digest}{q}'

        updated = ASSET_REF.sub(replace, original)
        if count:
            html_path.write_text(updated, encoding="utf-8")
            versioned += count
            pages += 1

    print(f"Fingerprinted {versioned} asset reference(s) across {pages} page(s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
