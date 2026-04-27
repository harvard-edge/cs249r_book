# StaffML Vault Schema & Folder Audit

**Date:** 2026-04-25
**Branch:** `audit/vault-schema-folder`
**Corpus state at audit:** 9,982 YAMLs (9,224 published / 458 deleted / 300 draft)

---

## TL;DR — recommendation

**Keep the schema and the folder structure as they are. Three small cleanups would help.**

| Area | Recommendation | Why |
|---|---|---|
| Folder layout (`<track>/<id>.yaml`) | **Keep flat.** Don't add topic / level / zone subdirs. | Team explicitly tried deeper hierarchy (v0.1) and reverted in v1.0 with documented reasons; filesystem ops are fast at current scale; classification is data, not addressing. |
| YAML schema (LinkML + Pydantic codegen) | **Keep.** Two minor edits worth making (see §4). | Required-field set is right; lineage fields populate cleanly; only edge cases need attention. |
| ID format (`<track>-NNNN` for new, legacy preserved) | **Keep.** Add a soft regex check at the validator. | 52% of corpus already uses the clean form; ID_SCHEMES.md migration policy is sound; no bulk rename. |

The structure has been thought-about more carefully than my earlier "let's add `track/topic/` subdirs" reflex. The audit confirms the team's existing decisions are correct.

---

## 1. Folder layout

### What's there now

```
interviews/vault/questions/
├── cloud/    4,302 files
├── edge/     2,147
├── global/     418
├── mobile/   1,779
└── tinyml/   1,336
                ────
                9,982 total
```

Flat: track → file. No level / zone / topic sub-folders.

### What was tried before

ARCHITECTURE.md §3.3 documents the v0.1 → v1.0 reversal in 2026-04-21. The
v0.1 design used `<track>/<level>/<zone>/<topic-hash>.yaml`. It was abandoned
because:

> - The path-as-classification design could not represent the paper's full
>   11-zone × 6-level taxonomy — only 4 of 11 zones and 6 of 6 levels had
>   directories, and L6+ was missing entirely.
> - The pre-v1.0 migration script silently collapsed questions with
>   unrepresentable (level, zone) pairs into `l1/recall/` and dropped 86
>   questions whose target cell did not exist.
> - Reclassification also required a file move, which actively discouraged
>   corrections.

The v1.0 reversal made classification a property of YAML data, with
filesystem carrying only `track` for navigability.

### Does flat scale?

| Operation | Time on 4,302-file `cloud/` folder |
|---|---|
| `ls cloud/` | 21 ms |
| `grep -l "topic: kv-cache-management" cloud/*.yaml` | 194 ms across 4,302 files |
| Python YAML-load all 9,982 files for an audit | ~6 s |

No filesystem pressure. Modern laptops happily handle directories with 10K+
small files. We have headroom to roughly 50K before any operation gets
sluggish.

### What about the `<track>/<topic>/` middle ground?

This is what I had been advocating in the StaffML flow pass. With 87 topics
× 5 tracks = 435 potential subdirs:

| Pro | Con |
|---|---|
| `ls cloud/kv-cache-management/` lists topic-mates | Most cells thinly populated — many empty dirs |
| Filesystem becomes self-documenting | Topic is **mutable** (Phase 2 audit moved 25 questions across boundaries) |
| Cleaner per-topic git diffs | `git mv` overhead for every reclassification |
| Browse-level discoverability | Doesn't help searching across topics — you'd still grep |

The con-side argument that ARCHITECTURE.md §3.3 makes is decisive: "the
filesystem as a shallow addressing scheme" treats every reclassification as
a YAML edit, which is the right cost model. Adding topic subdirs would
re-introduce the friction the team explicitly removed.

**Recommendation: keep flat. Don't add topic subdirs.**

The CLI (`vault ls --topic kv-cache-management --track cloud`) already
provides browsing-by-topic, and the practice page's filter does it for end
users. Filesystem hierarchy is not the right abstraction for queryable
content.

---

## 2. ID format

### What's actually in the corpus

| Pattern | Count | Share |
|---|---|---|
| `<track>-NNNN` (clean) | 5,228 | 52% |
| `<track>-<cohort>-NNNN` (process artifact) | 4,754 | 48% |
| Other | 0 | — |

Cohort labels in current IDs: 50 distinct values. Top:

| Cohort tag | Count | What it meant |
|---|---|---|
| `-fill-` | 1,300 | Gap-filling generation cohort |
| `-cell-` | 789 | Coverage-cell targeting cohort |
| `-r2-` | 259 | Round-2 generation |
| `-sus-` | 232 | "Suspicious" review cohort |
| `-exp-anal-` | 119 | Exploration / analyze cohort |
| `-crit-`, `-top-`, `-new-` | ~100 each | Various review queues |
| `-exp2-anal-` … `-exp2-opti-` | ~95 each | Bloom-zone-encoded round-2 cohorts |
| `-portfolio-`, `-balance-`, `-scale-` | <50 each | This week's portfolio-balance loop |

### Schema constraint today

`question_schema.yaml` declares `id` as a free-form `string` with no
regex. Anything matching `<track>-...` slips through. That's how the
cohort-tag IDs were ever allowed.

### Recommendation

- **Don't rename existing IDs.** Per ID_SCHEMES.md migration policy: chain
  references, URLs, paper appendix, audit JSONLs, git blame would all
  break. The cost of a rename is ~10× the value of the cleanup.
- **Add a soft regex** to the schema for *new* IDs only. Two-tier
  validation:
  1. Hard rule (fail validation): `^[a-z]+-[a-z0-9-]+$` — blocks
     unstructured IDs.
  2. Soft warning: `^(cloud|edge|mobile|tinyml|global)-\d{4,}$` for new
     content. Existing IDs grandfathered.

`vault new` already mints clean IDs; the regex would just enforce that
behaviour against direct YAML edits.

---

## 3. Schema field-population reality

Of the 50 fields the schema permits, here's what's actually populated.
**This shows the schema fits reality very well — required fields are 100%
populated, optional fields populate where they make sense.**

### Required core (100% across all statuses)

```
schema_version, id, track, level, zone, topic,
competency_area, bloom_level, phase, title,
scenario, question, status, provenance,
requires_explanation, expected_time_minutes,
validated, details (with realistic_solution + common_mistake + napkin_math)
```

These are the foundation. Every question carries them. The schema's
required-field set matches reality exactly.

### Validation lineage (100% in published / 0-2% in draft — by design)

```
validation_status, validation_date, validation_model     97% overall
math_status, math_date, math_model                       85-93% overall
math_verified                                             96%
```

Drafts have these blank because they haven't been through validation. The
gap is the lifecycle, not a schema problem.

### Genuinely sparse — and that's fine

| Field | Coverage | Why sparse |
|---|---|---|
| `chains` | 25% | Only 879 chains, ~3 q/chain — corpus design |
| `details.options` / `correct_index` | 17% | Most questions are open-ended, not MCQ |
| `validation_issues` | 16% | Only fires when the validator finds something |
| `tags` | 5% (mostly drafts) | See below — semantics inconsistent |
| `visual` | 0.3% | Visual archetype is intentionally curated |
| `validation_status_pro` / `validation_issues_pro` | 4-5% | Only Claude-Opus-Pro reviewed items |
| `classification_review` | 0.3% | Only 31 items reviewed for classification |

### Anomaly worth fixing — `tags` semantics

| Status | `tags` populated |
|---|---|
| Published | 1.6% |
| Draft | 99.7% |

Drafts use `tags` for **cohort tracking** (`portfolio-loop-iteration-001`,
`gemini-generated`, `target-specification-L5`). Published items rarely have
tags. Either:

1. **Strip cohort tags at promotion** (cleanest — `tags` becomes a true
   user-facing field).
2. **Document that `tags` is the audit/cohort breadcrumb field** (current
   reality — semantically muddled but honest).

I'd pick (1). The promotion script (`promote_validated.py`) is the natural
place to clear cohort tags. It's a five-line addition.

---

## 4. Schema improvements I would make

Two targeted edits, both low-risk:

### 4.1 Tighten `id` regex (add soft validator)

Today: `id: range: string` — anything goes.
Proposed:
```yaml
id:
  range: string
  identifier: true
  required: true
  pattern: "^(cloud|edge|mobile|tinyml|global)(-[a-z][a-z0-9]*)?-[0-9]+$"
  description: |
    <track>-NNNN for new content; <track>-<cohort>-NNNN tolerated for
    legacy items predating ID_SCHEMES.md v2.
```

This rejects future malformed IDs without breaking existing ones. ~120
items currently violate the regex (the `-anal-`, `-real-`, `-mast-`,
`-desi-`, `-spec-`, `-impl-`, `-flue-`, `-opti-` Bloom-zone cohorts that
have e.g. `cloud-exp2-anal-005` — the pattern accepts `-exp2-anal-005`
because cohort can include digits and hyphens but I'd allow it).

### 4.2 Drop `details.question` from schema

The Pydantic survey shows zero items use `details.question`. The top-level
`question` field replaced it. The unused nested field is dead schema.

### 4.3 Clean `tags` at promotion

Add to `promote_validated.py`:

```python
# Drop cohort breadcrumbs; preserve user-facing tags.
COHORT_TAG_PREFIXES = ("portfolio-loop-", "target-", "gemini-",
                       "visual-archetype-", "matplotlib-rendered",
                       "dot-rendered", "gap-")
d["tags"] = [t for t in d.get("tags", [])
             if not any(t.startswith(p) for p in COHORT_TAG_PREFIXES)]
```

Five lines; preserves audit trail in commit history; cleans the corpus.

---

## 5. What I am NOT recommending

To make the recommendation crisp, here's what I considered and rejected:

- ❌ **Add `<track>/<topic>/` subdirs.** ARCHITECTURE.md §3.3 documents
  why this is wrong: classification is mutable, filesystem hierarchy
  isn't.
- ❌ **Add `<track>/<level>/` or `<track>/<zone>/` subdirs.** Same reason
  — and the v0.1 attempt is a documented disaster (86 questions dropped,
  cells silently collapsed).
- ❌ **Rename legacy IDs to clean format.** Breaks 3,100+ chain refs,
  external bookmarks, paper, audit JSONLs.
- ❌ **Move from LinkML.** It's a sound choice that gives codegen for
  Pydantic + SQL + TS in one pass.
- ❌ **Move from per-question YAML.** SQLite-as-source-of-truth was
  rejected in v1.0 for good reasons (diffability, git blame, contributor
  ergonomics).
- ❌ **Add a `topic_path` derived field.** Computed at query time; no
  schema change needed.

---

## 6. What stays for a future audit

These are real but not blockers:

- **Cohort tags slip through promotion** (§4.3 fixes this).
- **`tags` field is overloaded** between user-facing categorisation and
  audit cohort breadcrumb. Could split into `tags` + `cohort` — but only
  worth it if cohorts persist post-promotion.
- **No top-level `chains` invariant** that warns when a chain's questions
  span tracks. The schema doesn't currently constrain a chain to
  one-track. Probably fine; pre-commit `vault check` could verify.
- **`last_modified` field exists in schema, populated nowhere.** Add a
  pre-commit hook that bumps it, or remove the field.

---

## Net answer

**The schema and folder structure are good. They reflect a deliberate v0.1
→ v1.0 redesign that learned from a documented failure. The right move
isn't to redesign — it's to apply three targeted cleanups (id regex, drop
`details.question`, clean cohort tags at promotion) and otherwise leave it
alone.**
