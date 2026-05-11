#!/usr/bin/env python3
"""Generate LaTeX macros — Phase-2 thin wrapper over ``vault export-paper``.

The figures pipeline (unchanged) still uses a **generated** monolithic JSON from
``vault build --local-json`` (see ``analyze_corpus.py``) for
``corpus_stats.json``. This entry point is **not** the stats sidecar.

The Phase-2 pipeline is a single command::

    vault.db → vault export-paper → macros.tex + corpus_stats_export.json

This script is now a thin wrapper that delegates to ``vault export-paper``. It
keeps the historical entry point (``python3 generate_macros.py``) working so
the paper build — and any CI job that calls this path — does not break during
the transition. Both the legacy \\num* macro namespace and the new \\staffml*
namespace are emitted so paper.tex needs no edits.

Reads: ``interviews/vault/releases/<version>/vault.db`` (preferred), or
``interviews/vault/vault.db`` (HEAD build) as fallback.

Writes: ``macros.tex`` + ``corpus_stats_export.json`` (DB snapshot sidecar) alongside
this script's parent. The canonical ``corpus_stats.json`` for matplotlib figures
comes from ``analyze_corpus.py`` (reads generated staffml/vault ``corpus.json``); the exporter does not overwrite it.

ARCHITECTURE.md §4.2 (vault export-paper), REVIEWS.md §R2-3 B.1.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPTS_DIR.parent
VAULT_DIR = PAPER_DIR.parent / "vault"
RELEASES_DIR = VAULT_DIR / "releases"


def resolve_release_version(explicit: str | None) -> str:
    """Pick the release to export. Prefer explicit; else ``latest`` symlink;
    else most-recent numerically sorted release; else fall back to the HEAD build."""
    if explicit:
        return explicit
    link = RELEASES_DIR / "latest"
    if link.exists():
        target = link.readlink() if link.is_symlink() else link
        return Path(target).name
    if RELEASES_DIR.exists():
        candidates = sorted(
            [p.name for p in RELEASES_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
        )
        if candidates:
            return candidates[-1]
    return "dev"  # HEAD build; vault export-paper will read vault/vault.db


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="Release version to export (default: latest).")
    args = parser.parse_args(argv)

    version = resolve_release_version(args.version)

    # Use `sys.executable -m vault_cli.main` so the same interpreter that runs this
    # script is used — no dependency on PATH containing the venv's bin/ dir.
    cmd = [sys.executable, "-m", "vault_cli.main", "export-paper", version,
           "--releases-dir", str(RELEASES_DIR),
           "--paper-dir", str(PAPER_DIR)]

    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        sys.stderr.write(
            "error: vault_cli module not importable.\n"
            "Install with: pip install -e interviews/vault-cli/\n"
        )
        return 3

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
