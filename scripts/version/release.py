#!/usr/bin/env python3
"""Shared release-versioning helpers for MLSysBook artifacts.

Single source of truth for "what release is this artifact?" across every
publishable project in the repo (StaffML, TinyTorch, Book Vol I/II,
MLSYSIM, Kits, Labs, Instructors). Designed to be additive: every helper
either emits a NEW file (release.json, manifest.json) or computes a
value to be passed to a publish workflow. Nothing here mutates an
existing build artifact or rewrites pyproject.toml-style canonical
sources.

Two adoption tiers (see docs/VERSIONING.md):

  Tier A (citable: StaffML, TinyTorch, Book, MLSYSIM)
    - Full release.json with input_paths and per-file hashes.
    - Merkle-style root hash bound into release_hash.
    - CHANGELOG.md per project, GitHub Release with notes.

  Tier B (rapidly-iterating: Kits, Labs, Instructors)
    - Flat SHA-256 over content directory (no Merkle).
    - Single release.json at project root, history in CHANGELOG.md.

The CLI is entry-pointed via ``__main__`` for use from any GitHub
Actions step regardless of project language.

Usage from a workflow step:

    python3 scripts/version/release.py compute-id \\
        --previous staffml-v0.1.0 --bump patch --prefix staffml-v
    # → 0.1.1 on stdout

    python3 scripts/version/release.py compute-hash \\
        --paths interviews/vault/questions interviews/vault/release-policy.yaml
    # → 64-char hex hash on stdout

    python3 scripts/version/release.py emit-release \\
        --project staffml --release-id 0.1.1 --tier A \\
        --release-hash <hash> --git-sha <sha> \\
        --output releases/staffml-0.1.1/release.json

    python3 scripts/version/release.py emit-manifest \\
        --project staffml --release-id 0.1.1 --release-hash <hash> \\
        --output interviews/staffml/src/data/vault-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Schema constants ──────────────────────────────────────────────────────────

RELEASE_SCHEMA_VERSION = "1"
"""Bumped when the shape of release.json itself changes."""

VALID_TIERS = frozenset({"A", "B"})

VALID_BUMPS = frozenset({"patch", "minor", "major", "none"})

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


# ── Hashing ──────────────────────────────────────────────────────────────────


def _iter_files(paths: list[Path], exclude: list[str] | None = None) -> list[Path]:
    """Walk paths, returning sorted list of files. Stable for hashing.

    Excludes match against POSIX-style relative paths via simple glob
    (``fnmatch``). We intentionally do NOT use ``.gitignore`` here —
    inputs to the release MUST be deterministic and not depend on
    transient ignore rules. Callers pass explicit excludes.
    """
    import fnmatch

    exclude = exclude or []
    out: list[Path] = []
    for p in paths:
        if p.is_file():
            out.append(p)
            continue
        if not p.is_dir():
            raise FileNotFoundError(f"input path does not exist: {p}")
        for root, dirs, files in os.walk(p):
            # Filter dirs in-place so os.walk skips them entirely.
            dirs[:] = sorted(
                d for d in dirs
                if not any(fnmatch.fnmatch(d, pat) for pat in exclude)
            )
            for f in sorted(files):
                rel = (Path(root) / f)
                if any(fnmatch.fnmatch(str(rel), pat) for pat in exclude):
                    continue
                if any(fnmatch.fnmatch(f, pat) for pat in exclude):
                    continue
                out.append(rel)
    return sorted(out)


def compute_dir_hash(
    paths: list[Path],
    exclude: list[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """SHA-256 over sorted (path, content) pairs across ``paths``.

    Returns ``(hex_hash, file_index)`` where file_index is a list of
    ``{"path": "...", "hash": "..."}`` entries — useful for Tier A
    release.json's ``files`` array (Merkle-ish: the root hash binds
    every per-file hash). Tier B can drop the index.

    The hash is intentionally newline-sensitive but case-sensitive on
    paths; it's stable across machines as long as filesystems agree on
    case (mac+linux do via case-folding awareness, since we hash the
    on-disk bytes). Symlinks are followed (hash the target's bytes).
    """
    sha = hashlib.sha256()
    index: list[dict[str, str]] = []
    files = _iter_files([Path(p) for p in paths], exclude)
    for f in files:
        per_file = hashlib.sha256()
        # Hash the relative path first so reordering or renaming changes
        # the root hash even if bytes are identical.
        rel = f.as_posix()
        per_file.update(rel.encode("utf-8"))
        per_file.update(b"\x00")
        with open(f, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                per_file.update(chunk)
        digest = per_file.hexdigest()
        index.append({"path": rel, "hash": digest})
        sha.update(digest.encode("ascii"))
        sha.update(b"\n")
    return sha.hexdigest(), index


# ── Semver ───────────────────────────────────────────────────────────────────


def parse_semver(s: str) -> tuple[int, int, int, str | None]:
    """Parse a semver string. Strips a leading ``v`` and any prefix-tag."""
    s = s.strip()
    # Strip prefix like "staffml-v" or "vol1-v" or just "v".
    s = re.sub(r"^[A-Za-z][A-Za-z0-9-]*-?v", "", s)
    s = s.lstrip("v")
    m = SEMVER_RE.match(s)
    if not m:
        raise ValueError(f"not a semver: {s!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def compute_release_id(
    previous: str | None,
    bump: str,
    *,
    explicit: str | None = None,
) -> str:
    """Return the new release_id given a previous tag and a bump type.

    - ``explicit`` overrides everything: returned as-is (after stripping
      a leading ``v`` if present, so the caller can pass either form).
    - ``bump=none`` returns the previous version unchanged. Caller is
      responsible for treating this as "site-only redeploy, no tag".
    - ``previous`` may be a bare ``X.Y.Z``, a ``vX.Y.Z``, or any
      project-prefixed tag like ``staffml-v0.1.0`` or ``vol1-v0.6.0``.
    - On no previous (first release ever), returns ``0.1.0`` for any
      bump that's not ``major`` (which returns ``1.0.0``).
    """
    if explicit:
        return explicit.lstrip("v").strip()
    if bump not in VALID_BUMPS:
        raise ValueError(f"bump must be one of {sorted(VALID_BUMPS)}: got {bump!r}")
    if bump == "none":
        if not previous:
            raise ValueError("bump=none requires a previous version")
        major, minor, patch, _ = parse_semver(previous)
        return f"{major}.{minor}.{patch}"
    if not previous:
        return "1.0.0" if bump == "major" else "0.1.0"
    major, minor, patch, _ = parse_semver(previous)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


# ── Release artifact + manifest emitters ─────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def emit_release_json(
    *,
    output: Path,
    project: str,
    tier: str,
    release_id: str,
    release_hash: str,
    schema_version: str,
    previous_release_id: str | None,
    git_sha: str,
    input_paths: list[str],
    file_index: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Write the canonical release.json artifact. Tier A includes file_index."""
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(VALID_TIERS)}: got {tier!r}")
    if not release_hash or len(release_hash) < 16:
        raise ValueError("release_hash must be a full hex digest (>= 16 chars)")

    payload: dict[str, Any] = {
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "project": project,
        "tier": tier,
        "release_id": release_id,
        "release_hash": release_hash,
        "schema_version": schema_version,
        "previous_release_id": previous_release_id,
        "git_sha": git_sha,
        "created_at": _utc_now_iso(),
        "input_paths": list(input_paths),
        "metadata": metadata or {},
    }
    if description:
        payload["description"] = description
    if tier == "A" and file_index is not None:
        payload["files"] = file_index

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def emit_manifest(
    *,
    output: Path,
    project: str,
    tier: str,
    release_id: str,
    release_hash: str,
    schema_version: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the build-time manifest the deployable bundles.

    This is the "single source of truth" the site reads at build time.
    A Tier A project may extend this with project-specific keys (e.g.
    StaffML's vault-manifest.json adds questionCount/trackDistribution);
    callers that need that should write a wrapper rather than tacking
    extra keys on through metadata. The shape here is the MINIMUM every
    project agrees on.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(VALID_TIERS)}: got {tier!r}")
    if not release_hash or len(release_hash) < 16:
        raise ValueError("release_hash must be a full hex digest (>= 16 chars)")

    payload: dict[str, Any] = {
        "releaseId": release_id,
        "releaseHash": release_hash,
        "schemaVersion": schema_version,
        "tier": tier,
        "project": project,
        "buildDate": _utc_now_iso(),
    }
    if metadata:
        payload["metadata"] = metadata

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cmd_compute_id(args: argparse.Namespace) -> int:
    out = compute_release_id(
        previous=args.previous or None,
        bump=args.bump,
        explicit=args.explicit or None,
    )
    print(out)
    return 0


def _cmd_compute_hash(args: argparse.Namespace) -> int:
    digest, _ = compute_dir_hash(
        [Path(p) for p in args.paths],
        exclude=args.exclude or [],
    )
    print(digest)
    return 0


def _cmd_emit_release(args: argparse.Namespace) -> int:
    file_index = None
    if args.tier == "A" and args.input_paths:
        _, file_index = compute_dir_hash(
            [Path(p) for p in args.input_paths],
            exclude=args.exclude or [],
        )
    metadata = json.loads(args.metadata) if args.metadata else {}
    payload = emit_release_json(
        output=Path(args.output),
        project=args.project,
        tier=args.tier,
        release_id=args.release_id,
        release_hash=args.release_hash,
        schema_version=args.schema_version,
        previous_release_id=args.previous or None,
        git_sha=args.git_sha,
        input_paths=args.input_paths or [],
        file_index=file_index,
        metadata=metadata,
        description=args.description or None,
    )
    print(json.dumps({"output": str(args.output), "release_id": payload["release_id"]}))
    return 0


def _cmd_emit_manifest(args: argparse.Namespace) -> int:
    metadata = json.loads(args.metadata) if args.metadata else {}
    payload = emit_manifest(
        output=Path(args.output),
        project=args.project,
        tier=args.tier,
        release_id=args.release_id,
        release_hash=args.release_hash,
        schema_version=args.schema_version,
        metadata=metadata,
    )
    print(json.dumps({"output": str(args.output), "releaseId": payload["releaseId"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release",
        description="Shared release-versioning helpers for MLSysBook.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_id = sub.add_parser("compute-id", help="Compute next release_id from previous + bump.")
    p_id.add_argument("--previous", default="", help="Previous tag/version. Empty = first release.")
    p_id.add_argument("--bump", default="patch", choices=sorted(VALID_BUMPS))
    p_id.add_argument("--explicit", default="", help="Override: explicit X.Y.Z to use, bypasses bump math.")
    p_id.set_defaults(func=_cmd_compute_id)

    p_hash = sub.add_parser("compute-hash", help="Compute SHA-256 dir hash over input paths.")
    p_hash.add_argument("--paths", nargs="+", required=True)
    p_hash.add_argument("--exclude", nargs="*", default=[])
    p_hash.set_defaults(func=_cmd_compute_hash)

    p_rel = sub.add_parser("emit-release", help="Write releases/<id>/release.json.")
    p_rel.add_argument("--output", required=True)
    p_rel.add_argument("--project", required=True)
    p_rel.add_argument("--tier", required=True, choices=sorted(VALID_TIERS))
    p_rel.add_argument("--release-id", required=True)
    p_rel.add_argument("--release-hash", required=True)
    p_rel.add_argument("--schema-version", default="1")
    p_rel.add_argument("--previous", default="")
    p_rel.add_argument("--git-sha", required=True)
    p_rel.add_argument("--input-paths", nargs="*", default=[])
    p_rel.add_argument("--exclude", nargs="*", default=[])
    p_rel.add_argument("--metadata", default="", help="Extra JSON object to merge into metadata.")
    p_rel.add_argument("--description", default="")
    p_rel.set_defaults(func=_cmd_emit_release)

    p_man = sub.add_parser("emit-manifest", help="Write the build-time manifest the deployable reads.")
    p_man.add_argument("--output", required=True)
    p_man.add_argument("--project", required=True)
    p_man.add_argument("--tier", required=True, choices=sorted(VALID_TIERS))
    p_man.add_argument("--release-id", required=True)
    p_man.add_argument("--release-hash", required=True)
    p_man.add_argument("--schema-version", default="1")
    p_man.add_argument("--metadata", default="")
    p_man.set_defaults(func=_cmd_emit_manifest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
