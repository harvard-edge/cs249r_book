#!/usr/bin/env python3
"""Render question visuals to ship-ready SVG.

The schema the website cares about is minimal::

    visual:
      kind: svg                # always svg — that's what the website ships
      path: <id>.svg           # the static asset
      alt: <text>
      caption: <text>
      # Build metadata (optional, ignored by website):
      source_format: dot | matplotlib | hand   # default: hand

The runtime story for the practice page is "load `<id>.svg` as an
`<img>`". The renderer's job is to produce that SVG when it's a build
artifact (DOT or matplotlib source). For hand-authored SVGs, the source
IS the asset, no build needed.

Source files live next to the asset by naming convention:

    interviews/vault/visuals/<track>/<id>.svg     # the asset (always)
    interviews/vault/visuals/<track>/<id>.dot     # iff source_format=dot
    interviews/vault/visuals/<track>/<id>.py      # iff source_format=matplotlib

Usage:
    python3 render_visuals.py                       # render all stale
    python3 render_visuals.py --force               # force re-render
    python3 render_visuals.py --id cloud-2846       # single question
    python3 render_visuals.py --dry-run             # plan only

Architecture: see interviews/vault/visuals/ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

VAULT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = VAULT_DIR / "questions"
VISUALS_DIR = VAULT_DIR / "visuals"

VALID_KINDS = {"svg"}                          # what the website renders
VALID_SOURCE_FORMATS = {"dot", "matplotlib", "hand"}
SOURCE_EXT = {"dot": "dot", "matplotlib": "py"}

# Book SVG style: enforce these properties on every rendered SVG so DOT,
# matplotlib, and hand-SVG outputs render identically in the practice page.
SVG_FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_visuals() -> list[dict[str, Any]]:
    """Return one record per question with a `visual:` block.

    Reads only the production-schema fields: kind, path, alt, caption.
    Build metadata: source_format (optional).
    """
    records = []
    for yaml_path in QUESTIONS_DIR.glob("**/*.yaml"):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ! {yaml_path}: parse error {exc}", file=sys.stderr)
            continue
        if not data or "visual" not in data:
            continue
        v = data["visual"]
        if not isinstance(v, dict):
            continue

        kind = v.get("kind", "svg")
        if kind not in VALID_KINDS:
            print(f"  ! {data.get('id')}: unsupported kind={kind!r} "
                  f"(only {VALID_KINDS} ship)", file=sys.stderr)
            continue

        path = v.get("path")
        if not path:
            print(f"  ! {data.get('id')}: missing visual.path", file=sys.stderr)
            continue

        track = data.get("track", "global")
        track_dir = VISUALS_DIR / track
        asset_path = track_dir / path

        # Source format defaults to "hand" (no build step needed)
        source_format = v.get("source_format", "hand")
        if source_format not in VALID_SOURCE_FORMATS:
            print(f"  ! {data.get('id')}: unknown source_format={source_format!r}",
                  file=sys.stderr)
            continue

        # Source file: same basename, extension by source_format
        source_path = None
        if source_format != "hand":
            ext = SOURCE_EXT[source_format]
            source_path = track_dir / f"{Path(path).stem}.{ext}"

        records.append({
            "id": data["id"],
            "track": track,
            "asset_path": asset_path,
            "source_path": source_path,
            "source_format": source_format,
            "yaml_path": yaml_path,
        })
    return records


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_one(rec: dict[str, Any], force: bool = False, dry_run: bool = False) -> str:
    """Render or pass through a single visual. Returns 'rendered'|'skipped'|error."""
    qid = rec["id"]
    fmt = rec["source_format"]
    src = rec["source_path"]
    out = rec["asset_path"]

    if fmt == "hand":
        if not out.exists():
            return f"error:{qid}:hand-authored asset missing at {out}"
        return "skipped"

    # Build artifact path: needs source
    if not src or not src.exists():
        return f"error:{qid}:{fmt} source missing at {src}"
    if not force and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return "skipped"
    if dry_run:
        print(f"  + would render {qid}: {src.name} -> {out.name}")
        return "rendered"
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "dot":
        _render_dot(src, out)
    else:
        _render_matplotlib(src, out)
    _normalize_svg(out)
    return "rendered"


def _render_dot(src: Path, out: Path) -> None:
    result = subprocess.run(
        ["dot", "-Tsvg", str(src), "-o", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dot failed: {result.stderr.strip()}")


def _render_matplotlib(src: Path, out: Path) -> None:
    """Execute the source script with VISUAL_OUT_PATH env var."""
    import os
    env = dict(os.environ)
    env["VISUAL_OUT_PATH"] = str(out)
    result = subprocess.run(
        ["python3", str(src)],
        capture_output=True, text=True, timeout=60, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"matplotlib script failed: {result.stderr.strip()}")
    if not out.exists():
        raise RuntimeError(
            f"matplotlib script ran but did not write to {out}; "
            "did the script use os.environ['VISUAL_OUT_PATH']?"
        )


def _normalize_svg(path: Path) -> None:
    """Apply book-style normalization to a rendered SVG."""
    text = path.read_text(encoding="utf-8")
    if "data:image/" in text or "<image " in text:
        raise RuntimeError(f"{path} contains embedded raster — not allowed")
    text = re.sub(r"<!--\s*Generated by [^>]*?-->\s*", "", text)
    if "font-family=" not in text.split(">", 1)[0]:
        text = re.sub(
            r"<svg(\s[^>]*?)>",
            lambda m: f'<svg{m.group(1)} font-family="{SVG_FONT_FAMILY}">',
            text, count=1,
        )
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if output is fresh.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--id", help="Render only this question id.")
    args = parser.parse_args()

    recs = discover_visuals()
    if args.id:
        recs = [r for r in recs if r["id"] == args.id]
        if not recs:
            print(f"No visual found for id={args.id}", file=sys.stderr)
            return 1

    print(f"Discovered {len(recs)} visual(s).")
    counts = {"rendered": 0, "skipped": 0, "error": 0}
    for rec in recs:
        try:
            status = render_one(rec, force=args.force, dry_run=args.dry_run)
        except Exception as exc:
            status = f"error:{rec['id']}:{exc}"
        if status.startswith("error"):
            print(f"  ✗ {status}")
            counts["error"] += 1
        else:
            print(f"  {'✓' if status == 'rendered' else '·'} {rec['id']:30s} "
                  f"[{rec['source_format']:11s}] {status}")
            counts[status] += 1

    print(f"\nSummary: rendered={counts['rendered']} "
          f"skipped={counts['skipped']} errors={counts['error']}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
