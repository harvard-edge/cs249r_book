# Vault ID schemes

**Status:** active (proposed 2026-04-21, effective for new content)
**Supersedes:** ad-hoc ID conventions in pre-v1.0 corpus (`cell-*`, `fill-*`, `fill2-*`, `sus-*`, `r2-*`, `crit-*`, `new-*`, `top-*` prefix markers)

---

## The durability principle

Identifiers are *load-bearing*: they appear in chain references, URLs, the
paper's appendix, contributor bookmarks, and the audit trail. Once assigned,
an ID should **never change**. This single constraint rules out encoding any
mutable attribute in the ID.

The corpus has four mutable axes (`level`, `zone`, `topic`, `status`) and
one near-immutable axis (`track`). Only the immutable axis is appropriate
for ID encoding.

Everything else — "what level is this?", "what zone is this?", "which chain
is this in?" — is answered by **tooling**, not by the ID.

---

## Question ID scheme

```
<track>-<yyyymm>-<4hex>
```

| Segment | Values | Source | Mutable? |
|---|---|---|---|
| `track` | `cloud \| edge \| mobile \| tinyml \| global` | author's choice | no |
| `yyyymm` | 6-digit year-month | creation time | no |
| `4hex` | 4 hex chars | `sha256(title + "\n" + topic)[:4]` | no (content-addressed) |

**Examples:**
- `cloud-202604-a3f2`
- `tinyml-202605-9b01`
- `edge-202604-c5de`

**Properties:**
- **18-char maximum length.** Easy to copy/paste.
- **Chronologically sortable** — sort the filename list and you're ordered by creation month.
- **Collision-resistant**: 65 536 slots per `(track, yyyymm)` bucket. On collision `vault new` auto-increments `4hex` to the next free hex value.
- **No mutable state.** If a question's level/zone/topic later changes, the ID stays.
- **Visually greppable.** `cloud-202604-*` = "cloud questions authored April 2026" without opening anything.

---

## Chain ID scheme

Chains are **topic-bound by definition** — a chain's identity is its topic,
not its arbitrary sequence number. Encode that directly:

```
chain-<track>-<topic-slug>-<yyyymm>[-<suffix>]
```

The optional `-<suffix>` (single letter `a`, `b`, `c`, …) only appears when
two chains in the same `(track, topic, month)` bucket need to disambiguate.

**Examples:**
- `chain-cloud-kv-cache-management-202604`
- `chain-tinyml-ota-firmware-updates-202604`
- `chain-edge-quantization-fundamentals-202605`
- `chain-cloud-latency-decomposition-202604-b` (second chain that month in that bucket)

**Properties:**
- **Self-describing.** Read the chain ID and you know what it's about.
- **Stable under corpus growth.** Adding questions to a chain doesn't rename the chain.
- **Per-chain topic invariant.** `vault lint` warns if any question's `topic` differs from the chain's topic — guards against a chain silently drifting off-topic.

---

## Migration policy

**Legacy IDs are preserved.** Every existing question ID
(`cloud-0185`, `tinyml-cell-13212`, `mobile-fill-00191`, …) stays as-is.
Same for existing chain IDs (`cloud-chain-432`, …). Renaming would break
3 100+ chain references, external bookmarks, the Phase 2 audit JSONLs, and
`git blame` chains.

**New content uses the new scheme.** `vault new` emits IDs in the new
format as of this policy. Old and new coexist freely in `id-registry.yaml`
and `chains.yaml`.

**Status today (2026-04-21):**
- Question IDs: 9 657 legacy (cell/fill/sus/r2/crit/new/top/plain prefixes)
- Chain IDs: 970 legacy (`<track>-chain-<NNN>`)
- New scheme IDs: 0 (this doc is the spec; first use lands with the next `vault new`)

---

## Companion tooling

The "I want to know X at a glance" pain point is solved by the CLI, not by
the ID:

| Need | Command |
|---|---|
| Browse all questions with filters | `vault ls [--track --level --zone --topic --status --in-chains]` |
| Inspect a single question + its chain context | `vault show <question-id>` |
| Walk a chain end-to-end | `vault chain show <chain-id>` |
| All chains for a track/topic | `vault chain ls [--track --topic]` |
| Confirm no orphan chain refs | `vault check --strict` |

All three read `vault.db` (compiled from YAMLs via `vault build`) or the
live YAMLs directly, so the outputs always reflect current classifications.

---

## Position semantics inside a chain

Chain positions are **non-negative integers**. Where a chain spans Bloom's
levels (the most common pattern), convention is:

- `position: 0` = L1
- `position: 1` = L2
- …
- `position: 5` = L6+

`vault lint` warns when a chain's level sequence isn't monotonically
non-decreasing over its positions. This is a soft warning, not a
hard rule — some chains legitimately iterate within a level (two L3
fluency variants of the same concept, for instance).

A question may belong to multiple chains (≈101 of them do in the
current corpus); its position is independent in each chain.

---

## Hash function (reproducibility)

For the `4hex` question-ID suffix, the deterministic recipe is:

```python
import hashlib
def question_id_hash(title: str, topic: str) -> str:
    payload = f"{title}\n{topic}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:4]
```

The topic is included so two unrelated questions with identical titles
(rare but possible) hash differently. Canonical ordering is `title + "\n" + topic`;
no normalization beyond UTF-8 encoding.

---

## Why not include level/zone/topic in the ID?

We considered — and rejected — an alternative scheme that encoded
`level` or `zone` in the ID (e.g. `cloud-L4-kv-cache-management-0042` or
"L1 questions are 1000, L2 are 2000…"). Concrete reasons:

1. **Levels change.** The Phase 2 content audit (2026-04-21) moved 25
   questions across level boundaries. Every move would force an ID
   rename under a level-encoded scheme.
2. **Chain references.** ≈3 100 question references live inside chain
   YAMLs. Every rename cascades.
3. **URL rot.** The Next.js site uses IDs as anchors (`/practice?id=…`).
   ID renames break bookmarks and external links.
4. **Git history.** Renames are traceable via `--follow`, but audit
   trails and commit messages referring to the old ID become stale.
5. **We don't need it.** The 4-axis classification is a `vault.db`
   query or a `grep ^level:` away. Tooling is cheaper than renaming.

The rule we settled on: **encode only the immutable (track + time +
content-hash). Push everything else to the query layer.**
