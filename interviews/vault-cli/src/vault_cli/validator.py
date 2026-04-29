"""Invariant checker.

Implements the tiered checks in ARCHITECTURE.md §5. Fast-tier checks run in
the pre-commit hook; structural-tier checks run in CI; slow-tier checks run
nightly.

This module is the engine; tier selection and reporting are in
``commands/check.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vault_cli.loader import LoadedQuestion
from vault_cli.paths import is_lowercase, vault_questions_root
from vault_cli.yaml_io import load_file


@dataclass(frozen=True)
class InvariantFailure:
    tier: str
    check: str
    question_id: str | None
    path: Path | None
    message: str


def _fail(tier: str, check: str, *, message: str, qid: str | None = None, path: Path | None = None) -> InvariantFailure:
    return InvariantFailure(tier=tier, check=check, question_id=qid, path=path, message=message)


def fast_tier(loaded: list[LoadedQuestion], vault_dir: Path) -> list[InvariantFailure]:
    """Fast-tier invariants — pre-commit-hook speed."""
    failures: list[InvariantFailure] = []

    # Check #2: unique IDs across published+draft
    ids = Counter(q.id for q in loaded)
    for qid, n in ids.items():
        if n > 1:
            failures.append(_fail("fast", "unique-id", qid=qid, message=f"ID appears {n} times"))

    # Check #4: path components lowercase + path mirrors body metadata.
    # Hierarchical layout: <track>/<competency_area>/<id>.yaml.
    # The path is a *derived* index over body fields — the body is the source
    # of truth, the path must mirror it. This invariant prevents drift like
    # "track field rewritten during audit but file not moved" silently breaking
    # the directory structure.
    root = vault_questions_root(vault_dir)
    for lq in loaded:
        rel = lq.path.relative_to(root)
        parts = rel.parts

        # Lowercase check on path components (excluding filename)
        for component in parts[:-1]:
            if not is_lowercase(component):
                failures.append(
                    _fail(
                        "fast",
                        "path-lowercase",
                        qid=lq.id,
                        path=lq.path,
                        message=f"path component {component!r} is not lowercase",
                    )
                )

        # Hierarchical layout check: <track>/<area>/<file>
        if len(parts) != 3:
            failures.append(
                _fail(
                    "fast",
                    "path-hierarchy-shape",
                    qid=lq.id,
                    path=lq.path,
                    message=(
                        f"path has {len(parts)} segments under questions/; "
                        "expected 3 (<track>/<competency_area>/<filename>)"
                    ),
                )
            )
            continue

        path_track, path_area, path_filename = parts
        if path_track != lq.question.track:
            failures.append(
                _fail(
                    "fast",
                    "path-track-match",
                    qid=lq.id,
                    path=lq.path,
                    message=(
                        f"path track segment {path_track!r} != body track "
                        f"{lq.question.track!r}"
                    ),
                )
            )
        if path_area != lq.question.competency_area:
            failures.append(
                _fail(
                    "fast",
                    "path-area-match",
                    qid=lq.id,
                    path=lq.path,
                    message=(
                        f"path area segment {path_area!r} != body competency_area "
                        f"{lq.question.competency_area!r}"
                    ),
                )
            )
        expected_filename = f"{lq.question.id}.yaml"
        if path_filename != expected_filename:
            failures.append(
                _fail(
                    "fast",
                    "path-id-match",
                    qid=lq.id,
                    path=lq.path,
                    message=f"filename {path_filename!r} != <id>.yaml ({expected_filename!r})",
                )
            )

    return failures


def _load_yaml_set(vault_dir: Path, filename: str, key: str) -> set[str]:
    """Load ``{vault_dir}/{filename}`` and extract IDs from its top-level ``key`` list."""
    path = vault_dir / filename
    if not path.exists():
        return set()
    data = load_file(path)
    if not isinstance(data, dict):
        return set()
    items = data.get(key, []) or []
    return {item["id"] if isinstance(item, dict) and "id" in item else item for item in items if item}


def structural_tier(
    loaded: list[LoadedQuestion],
    vault_dir: Path,
) -> list[InvariantFailure]:
    """Structural invariants — CI-tier checks."""
    failures: list[InvariantFailure] = []

    # #11: every `topic` exists in taxonomy.yaml
    known_topics = _load_yaml_set(vault_dir, "taxonomy.yaml", "topics")
    if known_topics:
        for lq in loaded:
            if lq.question.topic not in known_topics:
                failures.append(
                    _fail(
                        "structural",
                        "topic-in-taxonomy",
                        qid=lq.id,
                        path=lq.path,
                        message=f"topic {lq.question.topic!r} not found in taxonomy.yaml",
                    )
                )

    # #12-13: chain membership + position integrity.
    # v1.0 uses plural `chains: [{id, position}]` — a question can belong to
    # multiple chains. chains.json is the canonical registry.
    known_chains: set[str] = set()
    chain_registry_topic: dict[str, str] = {}
    import json as _json
    chains_json = vault_dir / "chains.json"
    if chains_json.exists():
        for entry in _json.loads(chains_json.read_text(encoding="utf-8")):
            cid = entry.get("chain_id") or entry.get("id")
            if cid:
                known_chains.add(cid)
                if entry.get("topic"):
                    chain_registry_topic[cid] = entry["topic"]

    # Indexed view: chain_id -> list of (question_id, topic, level, position)
    LEVEL_RANK = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6+": 6}
    chain_members: dict[str, list[tuple[str, str, str, int]]] = {}
    for lq in loaded:
        for c in lq.question.chains or []:
            if known_chains and c.id not in known_chains:
                failures.append(
                    _fail(
                        "structural",
                        "chain-ref-exists",
                        qid=lq.id,
                        path=lq.path,
                        message=f"chain {c.id!r} not found in chains.json",
                    )
                )
            chain_members.setdefault(c.id, []).append(
                (lq.id, lq.question.topic, lq.question.level, c.position)
            )

    # #14: no duplicate positions within a chain.
    for chain_id, members in chain_members.items():
        positions = [m[3] for m in members]
        if len(positions) != len(set(positions)):
            failures.append(
                _fail(
                    "structural",
                    "chain-position-duplicate",
                    qid=None,
                    message=f"chain {chain_id!r} has duplicate positions among {positions}",
                )
            )

    # #15: every chain is single-topic (chain identity IS its topic).
    for chain_id, members in chain_members.items():
        topics = {m[1] for m in members}
        if len(topics) > 1:
            failures.append(
                _fail(
                    "structural",
                    "chain-single-topic",
                    qid=None,
                    message=(
                        f"chain {chain_id!r} spans {len(topics)} topics {sorted(topics)} — "
                        "chains must be topic-bound per paper §3.3"
                    ),
                )
            )

    # #16: every question's topic matches its chain's registered topic.
    for chain_id, members in chain_members.items():
        reg_topic = chain_registry_topic.get(chain_id)
        if not reg_topic:
            continue
        for qid, q_topic, _level, _pos in members:
            if q_topic != reg_topic:
                failures.append(
                    _fail(
                        "structural",
                        "chain-registry-topic-match",
                        qid=qid,
                        message=(
                            f"question topic {q_topic!r} doesn't match chain {chain_id!r} "
                            f"registry topic {reg_topic!r}"
                        ),
                    )
                )

    # #17: chain membership is at least 2. Single-member chains aren't chains.
    for chain_id, members in chain_members.items():
        if len(members) < 2:
            failures.append(
                _fail(
                    "structural",
                    "chain-min-members",
                    qid=None,
                    message=f"chain {chain_id!r} has only {len(members)} member(s); min 2",
                )
            )

    # #18: chain levels are Bloom-monotonic (non-decreasing over position).
    # This is a soft pedagogical promise — surfaced as an error because the
    # pipeline re-orders positions automatically when a chain drifts.
    for chain_id, members in chain_members.items():
        members_by_pos = sorted(members, key=lambda m: m[3])
        ranks = [LEVEL_RANK.get(m[2], 0) for m in members_by_pos]
        if ranks != sorted(ranks):
            failures.append(
                _fail(
                    "structural",
                    "chain-bloom-monotonic",
                    qid=None,
                    message=(
                        f"chain {chain_id!r} levels by position are {[m[2] for m in members_by_pos]} — "
                        "expected non-decreasing (L1 → L6+)"
                    ),
                )
            )

    # #18: provenance metadata consistency. Author-facing metadata lives in
    # `authors` (free-form list) rather than the old generation_meta block,
    # which was never populated on the imported corpus. Post-v1.0 we let
    # human review flow through human_reviewed instead.

    # #14: taxonomy prerequisite graph is a DAG (B.8).
    failures.extend(_check_taxonomy_dag(vault_dir))

    # #15: applicability matrix respected — no questions in excluded (track, topic) cells (B.8).
    failures.extend(_check_applicability(loaded, vault_dir))

    return failures


def _check_taxonomy_dag(vault_dir: Path) -> list[InvariantFailure]:
    """Invariant #14: taxonomy prerequisite edges form a DAG (B.8 / §5)."""
    tax_path = vault_dir / "taxonomy.yaml"
    if not tax_path.exists():
        return []
    data = load_file(tax_path)
    if not isinstance(data, dict):
        return []
    edges = data.get("edges", []) or []
    graph: dict[str, set[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source") or edge.get("from")
        dst = edge.get("target") or edge.get("to")
        if src and dst:
            graph.setdefault(src, set()).add(dst)

    # Detect cycle via DFS with 3-color marking.
    WHITE, GRAY, BLACK = 0, 1, 2
    # Chip R4-H-4: include pure-target nodes (never appear as edge.source)
    # so DFS traversal through them doesn't implicitly create WHITE
    # entries with wrong semantics mid-walk.
    all_nodes = set(graph)
    for nbrs in graph.values():
        all_nodes.update(nbrs)
    color: dict[str, int] = {n: WHITE for n in all_nodes}

    def visit(node: str, stack: list[str]) -> list[str] | None:
        color[node] = GRAY
        stack.append(node)
        for nbr in graph.get(node, ()):
            c = color.get(nbr, WHITE)
            if c == GRAY:
                idx = stack.index(nbr) if nbr in stack else 0
                return stack[idx:] + [nbr]
            if c == WHITE:
                cycle = visit(nbr, stack)
                if cycle:
                    return cycle
        stack.pop()
        color[node] = BLACK
        return None

    for node in list(graph):
        if color[node] == WHITE:
            cycle = visit(node, [])
            if cycle:
                return [
                    _fail(
                        "structural",
                        "taxonomy-dag",
                        message=f"taxonomy edges contain a cycle: {' → '.join(cycle)}",
                    )
                ]
    return []


def _check_applicability(
    loaded: list[LoadedQuestion], vault_dir: Path
) -> list[InvariantFailure]:
    """Invariant #15: no questions in (track, topic) cells marked excluded (B.8 / §5)."""
    matrix_path = vault_dir / "data" / "applicable_cells.json"
    if not matrix_path.exists():
        return []
    import json
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    # Gemini R5-L-1: normalize both sides to lowercase so a hand-maintained
    # applicable_cells.json that uses "Cloud" / "Edge" still matches the
    # lowercase-normalized path components in the loaded questions.
    excluded = {
        (str(c.get("track", "")).lower(), str(c.get("topic", "")).lower())
        for c in (matrix.get("excluded_cells") or [])
        if isinstance(c, dict)
    }
    if not excluded:
        return []
    out: list[InvariantFailure] = []
    for lq in loaded:
        track = lq.question.track.lower()
        topic = lq.question.topic.lower()
        if (track, topic) in excluded:
            out.append(
                _fail(
                    "structural",
                    "applicability-excluded",
                    qid=lq.id,
                    path=lq.path,
                    message=(
                        f"({lq.question.track!r}, {lq.question.topic!r}) "
                        "is in the excluded-cells set per applicable_cells.json"
                    ),
                )
            )
    return out


def slow_tier(loaded: list[LoadedQuestion], vault_dir: Path) -> list[InvariantFailure]:
    """Nightly tier — #21 scenario near-duplicate detection via MinHash/LSH blocking,
    then Jaro-Winkler within candidate pairs (B.9 / §5 invariant 21-22).
    """
    return _scenario_dedup_lsh(loaded, vault_dir)


def _scenario_dedup_lsh(loaded: list[LoadedQuestion], vault_dir: Path) -> list[InvariantFailure]:
    """LSH-blocked scenario-duplicate detection.

    Uses character k-shingles + MinHash signatures + LSH bucketing, then
    Jaro-Winkler similarity on the within-bucket candidate pairs only.
    Pure-Python so no heavy deps; runs in nightly CI.
    """
    import hashlib
    from collections import defaultdict

    K = 5
    NUM_HASHES = 64
    JW_THRESHOLD = 0.95
    # Chip R4-H-4: cap scenario length before shingling to prevent a malicious
    # 256KB-scenario YAML from blowing the nightly-CI budget. 8000 chars is
    # well past the signal floor for near-duplicate detection.
    MAX_SHINGLE_LEN = 8000

    def shingles(text: str, k: int = K) -> set[str]:
        t = " ".join(text.split())
        if len(t) > MAX_SHINGLE_LEN:
            t = t[:MAX_SHINGLE_LEN]
        if len(t) <= k:
            return {t}
        return {t[i : i + k] for i in range(len(t) - k + 1)}

    def minhash(shs: set[str], seeds: list[int]) -> list[int]:
        if not shs:
            return [0] * len(seeds)
        return [
            min(
                int.from_bytes(
                    hashlib.blake2b(sh.encode("utf-8"), digest_size=8, person=str(s).encode()).digest(),
                    "big",
                )
                for sh in shs
            )
            for s in seeds
        ]

    def jaro_winkler(a: str, b: str) -> float:
        if a == b:
            return 1.0
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        match_dist = max(la, lb) // 2 - 1
        ma = [False] * la
        mb = [False] * lb
        matches = 0
        for i, ca in enumerate(a):
            lo = max(0, i - match_dist)
            hi = min(i + match_dist + 1, lb)
            for j in range(lo, hi):
                if mb[j] or b[j] != ca:
                    continue
                ma[i] = True
                mb[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        trans = 0
        k = 0
        for i in range(la):
            if not ma[i]:
                continue
            while not mb[k]:
                k += 1
            if a[i] != b[k]:
                trans += 1
            k += 1
        t = trans / 2
        jaro = (matches / la + matches / lb + (matches - t) / matches) / 3
        # Winkler prefix bonus.
        p = 0
        for i in range(min(4, la, lb)):
            if a[i] != b[i]:
                break
            p += 1
        return jaro + p * 0.1 * (1 - jaro)

    seeds = list(range(NUM_HASHES))
    sigs = [minhash(shingles(lq.question.scenario), seeds) for lq in loaded]

    # LSH: bucket by consecutive band-of-hashes; within-bucket candidate pairs.
    bands = 16
    rows = NUM_HASHES // bands
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            band_key = (b,) + tuple(sig[b * rows : (b + 1) * rows])
            buckets[band_key].append(idx)

    # Gemini R5-H-4 + Chip R7-M-4: pass vault_dir so we find the right
    # dedup-acks.yaml rather than a CWD-relative guess.
    try:
        from vault_cli.commands.dup import ack_pairs
        acked = ack_pairs(vault_dir)
    except Exception:  # noqa: BLE001 — validator must never crash on ack-read
        acked = set()

    seen: set[tuple[int, int]] = set()
    failures: list[InvariantFailure] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                pair_key = tuple(sorted((loaded[a].id, loaded[b].id)))
                if pair_key in acked:
                    continue   # operator acknowledged as intentional
                score = jaro_winkler(loaded[a].question.scenario, loaded[b].question.scenario)
                if score > JW_THRESHOLD:
                    failures.append(
                        _fail(
                            "slow",
                            "scenario-near-duplicate",
                            qid=loaded[a].id,
                            message=(
                                f"scenario Jaro-Winkler={score:.3f} vs {loaded[b].id!r}; "
                                f"acknowledge with `vault dup --ack {loaded[a].id} {loaded[b].id}` "
                                "if intentional"
                            ),
                        )
                    )
    return failures


def run_all(loaded: list[LoadedQuestion], vault_dir: Path) -> list[InvariantFailure]:
    return fast_tier(loaded, vault_dir) + structural_tier(loaded, vault_dir)


__all__ = ["InvariantFailure", "fast_tier", "structural_tier", "run_all"]
