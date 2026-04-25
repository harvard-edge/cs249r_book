#!/usr/bin/env python3
"""Render question visuals from source artifacts to ship-ready SVG.

Single entry point that dispatches by `visual.kind`:
  - `svg`       -> no-op (source is already the rendered output)
  - `dot`       -> graphviz `dot -Tsvg` auto-layout
  - `matplotlib`-> execute the source script, which must `savefig` to the
                  rendered path

Reads each YAML under `interviews/vault/questions/`, picks up its
`visual:` block (if any), and renders. Idempotent: skips items whose
rendered SVG is newer than the source.

Usage:
    python3 render_visuals.py                       # render all stale
    python3 render_visuals.py --force               # force re-render
    python3 render_visuals.py --id cloud-visual-001 # single question
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

SUPPORTED_KINDS = {"svg", "dot", "matplotlib"}

# Book SVG style: enforce these properties on every rendered SVG so DOT,
# matplotlib, and hand-SVG outputs render identically in the practice page.
SVG_FONT_FAMILY = "Helvetica Neue, Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_visuals() -> list[dict[str, Any]]:
    """Return one record per question with a non-empty `visual:` block."""
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
        if kind not in SUPPORTED_KINDS:
            print(f"  ! {data.get('id')}: unknown visual.kind={kind!r}", file=sys.stderr)
            continue
        track = data.get("track", "global")
        track_dir = VISUALS_DIR / track
        # `path` is the legacy field (cloud-visual-001 uses it). Treat it
        # as the rendered output for kind=svg, and as the source for
        # kind=dot/matplotlib if `source` is unset.
        legacy_path = v.get("path")
        source_rel = v.get("source") or legacy_path
        rendered_rel = v.get("rendered") or legacy_path or _default_rendered_name(data["id"])

        records.append({
            "id": data["id"],
            "track": track,
            "kind": kind,
            "source_path": (track_dir / source_rel) if source_rel else None,
            "rendered_path": track_dir / rendered_rel,
            "yaml_path": yaml_path,
            "alt": v.get("alt", ""),
            "caption": v.get("caption", ""),
        })
    return records


def _default_rendered_name(qid: str) -> str:
    return f"{qid}.svg"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_one(rec: dict[str, Any], force: bool = False, dry_run: bool = False) -> str:
    """Render a single visual record. Returns 'rendered'|'skipped'|'error'."""
    qid = rec["id"]
    kind = rec["kind"]
    src = rec["source_path"]
    out = rec["rendered_path"]

    if kind == "svg":
        if not (src and src.exists()):
            return f"error:{qid}:svg source missing at {src}"
        # No-op: source IS the output. Just confirm presence.
        return "skipped"

    if kind in ("dot", "matplotlib"):
        if not src or not src.exists():
            return f"error:{qid}:source missing at {src}"
        if not force and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            return "skipped"
        if dry_run:
            print(f"  + would render {qid}: {src.name} -> {out.name}")
            return "rendered"
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "dot":
            _render_dot(src, out)
        else:
            _render_matplotlib(src, out)
        _normalize_svg(out)
        return "rendered"

    return f"error:{qid}:unknown kind {kind}"


def _render_dot(src: Path, out: Path) -> None:
    """Run `dot -Tsvg src -o out`. Raises on failure."""
    result = subprocess.run(
        ["dot", "-Tsvg", str(src), "-o", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dot failed: {result.stderr.strip()}")


def _render_matplotlib(src: Path, out: Path) -> None:
    """Execute the source script with OUT_PATH passed as env var.

    The script is expected to read os.environ['VISUAL_OUT_PATH'] and
    `savefig(out_path, format="svg", bbox_inches="tight")` itself.
    """
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
    """Apply book-style normalization to a rendered SVG.

    - Set font-family on the root <svg>
    - Strip <!--Generated by ...--> comments that vary by tool version
    - Confirm no embedded raster
    """
    text = path.read_text(encoding="utf-8")
    if "data:image/" in text or "<image " in text:
        raise RuntimeError(f"{path} contains embedded raster — not allowed")

    # Strip generator comment (graphviz / matplotlib both emit one)
    text = re.sub(r"<!--\s*Generated by [^>]*?-->\s*", "", text)

    # Inject our font-family on the root <svg> if not already present
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
                        help="Re-render even if the output is fresh.")
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
            kind_tag = rec["kind"]
            print(f"  {'✓' if status == 'rendered' else '·'} {rec['id']:30s} "
                  f"[{kind_tag:11s}] {status}")
            counts[status] += 1

    print(f"\nSummary: rendered={counts['rendered']} "
          f"skipped={counts['skipped']} errors={counts['error']}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
