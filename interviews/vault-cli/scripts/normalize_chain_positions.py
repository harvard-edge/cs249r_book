#!/usr/bin/env python3
"""One-shot: normalize chain positions to contiguous [1..N] per chain.

The Phase-1 split preserved legacy ``chain_ids[0]`` as each question's chain
and took ``position + 1`` from the legacy 0-indexed value. Legacy corpus had
multi-chain membership: a single question could appear in up to 4 chains.
Our new single-chain schema kept only the first; the other chains are left
with gaps wherever their other members chose a different chain_ids[0].

This script re-numbers each chain's members to [1..N] sorted by their
current position, closing the gaps. Content-hash of affected questions
changes; re-run `vault build` afterward and update
``corpus-equivalence-hash.txt``.

Idempotent: running again with all chains already contiguous is a no-op.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "interviews" / "vault-cli" / "src"))

from vault_cli.loader import load_all  # noqa: E402
from vault_cli.yaml_io import dump_str, load_file  # noqa: E402


def main() -> int:
    vault_dir = _REPO / "interviews" / "vault"
    loaded, errors = load_all(vault_dir)
    if errors:
        sys.stderr.write(f"warning: {len(errors)} load errors — skipping those files\n")

    # Group by chain_id; each entry is (original_position, LoadedQuestion).
    chains: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for lq in loaded:
        if lq.question.chain is not None:
            chains[lq.question.chain.id].append((lq.question.chain.position, lq))

    rewritten = 0
    for _chain_id, members in chains.items():
        members.sort(key=lambda t: t[0])
        positions = [m[0] for m in members]
        expected = list(range(1, len(members) + 1))
        if positions == expected:
            continue

        # Rewrite each member's chain.position to its new index.
        for new_pos, (_old_pos, lq) in enumerate(members, start=1):
            # Load raw YAML (preserves untouched fields) then update.
            data = load_file(lq.path)
            if "chain" in data and isinstance(data["chain"], dict):
                if data["chain"].get("position") == new_pos:
                    continue
                data["chain"]["position"] = new_pos
                lq.path.write_text(dump_str(data), encoding="utf-8")
                rewritten += 1

    print(f"normalized: {rewritten} questions rewritten across {len(chains)} chains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
