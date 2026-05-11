#!/usr/bin/env python3
"""Detect chain decay — questions that have drifted semantically away from
their chain mates after an edit.

Phase 4.7 of CHAIN_ROADMAP.md. **Advisory, not blocking** on first ship.
Run as a manual gut-check after editing chain-member questions, or wire
into pre-commit if you want it on every change to vault YAMLs.

What it does:
  1. Find chain members whose YAML differs from origin/dev (or against
     a configurable base ref) — i.e., the questions you're about to
     commit / push that are in chains.
  2. Re-embed each one with the same model used for the corpus
     (BAAI/bge-small-en-v1.5).
  3. For each chain it belongs to, compute the cosine between the
     new embedding and every chain-mate's cached embedding from
     embeddings.npz.
  4. Report:
       - chain_id, qid, mate_id, cosine
       - whether the min mate-cosine fell below a threshold
         (default 0.40 — calibrated against the post-Phase-1 corpus)
  5. Exit 0 (advisory) by default; pass --strict to exit non-zero on
     any below-threshold result.

Usage:
  python3 check_chain_decay.py                          # advisory, vs origin/dev
  python3 check_chain_decay.py --base HEAD~5            # check vs an older base
  python3 check_chain_decay.py --strict                 # blocking (CI gate)
  python3 check_chain_decay.py --files <a.yaml> <b.yaml>  # explicit files
  python3 check_chain_decay.py --threshold 0.50         # tighter cutoff

Cost note: The first invocation downloads BAAI/bge-small-en-v1.5
(~135 MB). Subsequent runs use the local HuggingFace cache and finish
in seconds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VAULT_DIR = REPO_ROOT / "interviews" / "vault"
QUESTIONS_DIR = VAULT_DIR / "questions"
CHAINS_PATH = VAULT_DIR / "chains.json"
EMBEDDINGS_PATH = VAULT_DIR / "embeddings.npz"

DEFAULT_BASE_REF = "origin/dev"
DEFAULT_THRESHOLD = 0.40


# ─── changed-files discovery ──────────────────────────────────────────────


def changed_chain_member_yamls(base_ref: str | None,
                                explicit_files: list[Path] | None) -> list[Path]:
    if explicit_files:
        return [p for p in explicit_files if p.exists()]
    if base_ref is None:
        return []
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "--",
             str(QUESTIONS_DIR.relative_to(REPO_ROOT))],
            capture_output=True, text=True, cwd=REPO_ROOT, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  git diff failed: {e}", file=sys.stderr)
        return []
    files = [REPO_ROOT / line for line in out.stdout.splitlines()
             if line.endswith(".yaml")]
    return [p for p in files if p.exists()]


# ─── chain registry / corpus indexing ─────────────────────────────────────


def load_chain_membership() -> dict[str, list[dict]]:
    """qid -> list of {chain_id, mates: [qid, ...]} entries."""
    if not CHAINS_PATH.exists():
        return {}
    chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for ch in chains:
        cid = ch.get("chain_id")
        if not cid:
            continue
        mates = [m.get("id") for m in ch.get("questions", []) if m.get("id")]
        for qid in mates:
            out.setdefault(qid, []).append({
                "chain_id": cid,
                "tier": ch.get("tier", "primary"),
                "mates": [m for m in mates if m != qid],
            })
    return out


# ─── embedding ────────────────────────────────────────────────────────────


_embed_state: dict[str, Any] = {}


def _load_embeddings():
    if "model" in _embed_state:
        return _embed_state
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"missing {EMBEDDINGS_PATH} — run the corpus embedding script first"
        )
    npz = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    model_name = str(npz["model_name"])
    model = SentenceTransformer(model_name)
    _embed_state.update({
        "model": model,
        "model_name": model_name,
        "vectors": npz["vectors"],
        "qids": [str(x) for x in npz["qids"]],
        "qid_to_row": {str(q): i for i, q in enumerate(npz["qids"])},
    })
    return _embed_state


def embed_text_for_question(yaml_body: dict) -> Any:
    """Embed the same way the corpus's embeddings.npz did:
    title + scenario + question concatenated, normalised."""
    state = _load_embeddings()
    text = "\n".join([
        yaml_body.get("title", "") or "",
        yaml_body.get("scenario", "") or "",
        yaml_body.get("question", "") or "",
    ])
    vec = state["model"].encode([text], normalize_embeddings=True)[0]
    return vec


def cached_embedding(qid: str):
    state = _load_embeddings()
    row = state["qid_to_row"].get(qid)
    if row is None:
        return None
    return state["vectors"][row]


# ─── decay check ──────────────────────────────────────────────────────────


def check_decay(
    yaml_path: Path,
    membership: dict[str, list[dict]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Returns list of report rows (one per chain the question is in)."""
    import numpy as np
    body = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    qid = body.get("id")
    if not qid:
        return []
    chains = membership.get(qid, [])
    if not chains:
        return []

    new_vec = embed_text_for_question(body)
    rows: list[dict[str, Any]] = []
    for ch in chains:
        mate_cosines: list[tuple[str, float]] = []
        for mate_qid in ch["mates"]:
            mate_vec = cached_embedding(mate_qid)
            if mate_vec is None:
                continue
            cos = float(np.dot(new_vec, mate_vec))
            mate_cosines.append((mate_qid, cos))
        if not mate_cosines:
            continue
        min_mate, min_cos = min(mate_cosines, key=lambda x: x[1])
        rows.append({
            "qid": qid,
            "chain_id": ch["chain_id"],
            "tier": ch["tier"],
            "min_mate": min_mate,
            "min_cosine": round(min_cos, 4),
            "below_threshold": min_cos < threshold,
            "all_cosines": [(m, round(c, 4)) for m, c in mate_cosines],
        })
    return rows


# ─── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE_REF,
                    help=f"git base ref to diff against (default {DEFAULT_BASE_REF})")
    ap.add_argument("--files", nargs="+", type=Path, default=None,
                    help="explicit YAML paths to check (overrides --base)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"min mate-cosine before flagging (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any chain falls below threshold "
                         "(use as CI gate; default is advisory)")
    args = ap.parse_args()

    files = changed_chain_member_yamls(
        base_ref=None if args.files else args.base,
        explicit_files=args.files,
    )
    if not files:
        ref = "explicit files" if args.files else args.base
        print(f"no changed YAMLs vs {ref}")
        return 0

    print(f"checking {len(files)} changed YAML(s) for chain decay "
          f"(threshold={args.threshold})")
    membership = load_chain_membership()

    all_rows: list[dict[str, Any]] = []
    for p in files:
        try:
            rows = check_decay(p, membership, args.threshold)
        except Exception as e:
            print(f"  ✗ {p.relative_to(REPO_ROOT)}: {e}", file=sys.stderr)
            continue
        all_rows.extend(rows)

    if not all_rows:
        print("no chain members in changed YAMLs (or none had cached mate embeddings)")
        return 0

    flagged = [r for r in all_rows if r["below_threshold"]]
    print()
    for r in all_rows:
        marker = "⚠" if r["below_threshold"] else "✓"
        print(f"  {marker} {r['qid']:14s} chain={r['chain_id']:40s}"
              f" tier={r['tier']:9s} min-mate={r['min_mate']:14s}"
              f" cos={r['min_cosine']}")

    print(f"\nsummary: {len(flagged)}/{len(all_rows)} chain memberships "
          f"flagged below {args.threshold}")
    if flagged:
        print("(advisory — investigate whether the edit drifted the question "
              "semantically away from its chain mates)")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
