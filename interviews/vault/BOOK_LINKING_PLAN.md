# Book-Linking Resume Plan

> **Status (2026-04-16 follow-up — Option B RETIRED in favor of author-curated resources)**:
> The registry recommendation below is superseded. Implemented model:
> `details.resources: [{name, url}]` — a multi-valued, author-curated
> list on each question. Per-question book links come back when
> `mlsysbook.ai` URLs stabilize; until then authors simply don't list
> book entries. Site-level cross-linking to the book is covered by the
> existing `EcosystemBar` and the new page-end `Footer` component.
>
> See commit d9fcf8af2 ("refactor(vault): replace singular deep_dive
> with author-curated resources list") and the commits through
> 435d5a310 for the full decoupling sequence.
>
> **Why retired**: the registry would have pushed the book-URL-churn
> problem into a single file that still had to be regenerated every
> book release. The resources-list model sidesteps this entirely —
> there is no shared dependency surface to regenerate. StaffML is now
> self-contained for book-linking purposes, and book resources are
> added per-question by editorial act when the author has a stable
> URL worth citing.
>
> **What this file still documents** (below, kept for historical record):
> the four options evaluated, the heading-churn evidence, and the
> registry schema sketch. The "Options evaluated" section, "Resume-session
> task list", and "Files to touch" are kept verbatim so future readers
> can audit the decision trail, but the *recommendation* from the
> original draft no longer reflects shipped reality.

---

# ORIGINAL DRAFT (pre-2026-04-16 follow-up)

> **Status**: Deferred design. Resume in a fresh session after basic vault
> functionality is validated. Not a blocker for the Phase 4 cutover.
>
> **Context**: This plan was extracted from the vault migration session
> (2026-04-16) so the vault work could ship without entangling it in a
> separate product decision.

---

## The problem in one paragraph

StaffML questions are tagged with ML-Systems topics (e.g., `kv-cache`,
`distributed-training`, `quantization`). For each tag we want a stable,
deterministic link from the question card back to the corresponding section
of the textbook at `https://mlsysbook.ai`. The book is changing — chapters
get reorganized, headings rename, and eventually the book splits into
`mlsysbook.ai/vol1/` and `mlsysbook.ai/vol2/`. Naive links (`/chapter-7/#kv-cache`)
break on every rewrite. We need a mechanism that survives book evolution
without requiring StaffML redeploys.

---

## Options evaluated (2026-04-16 session)

### A. Unified Pagefind index at site-deploy time
Merge both volumes' Pagefind indexes into one when publishing
`mlsysbook.ai`. StaffML links to a search URL with the topic as the query.

- **Effort**: Medium — needs a site-build hook that merges Pagefind
  `_pagefind/` from vol1 and vol2.
- **Stability**: Weak — ranking depends on current heading text; a rewrite
  that changes the primary heading can demote the right page.
- **Cost**: Free.

### B. Topic→volume registry (RECOMMENDED in the session)
Book publishes a JSON/YAML file (e.g., `https://mlsysbook.ai/topics.json`)
mapping each of the ~87 topics to `{volume, url, anchor, last_verified}`.
StaffML hits the registry at click time (or bakes it into the build).

- **Effort**: Low — one file, updated once per book release.
- **Stability**: Highest — the book controls the anchor; rewrites update
  the registry, never break consumers.
- **Cost**: Free.
- **Volume split**: Each entry carries `volume: 1|2` — solves the split
  natively.
- **Bonus**: The registry is itself academically citable — it's the
  canonical topic→content mapping the paper can reference.

### C. Click-time fallback (punt)
StaffML links to `https://mlsysbook.ai/?q=<topic>` and lets the book's own
search handle it. No registry, no integration.

- **Effort**: Zero.
- **Stability**: Worst — user lands on a search result page, not the
  actual section.
- **Cost**: Free.

### D. Domain-scoped custom search
Use Google Programmable Search Engine or self-hosted Typesense scoped to
`site:mlsysbook.ai`. Transparently covers both volumes.

- **Effort**: Zero (PSE) to High (self-hosted).
- **Stability**: Crawler-dependent — Google indexing lag, opaque ranking.
- **Cost**: Free with ads (PSE free tier) or $5/1000 queries (PSE paid)
  or $0–20/mo (self-hosted).
- **Weakness**: Ads on the free tier undercut the academic feel; result
  page is not ours.

---

## Recommendation going in

**Option B (topic registry)** — rationale:

1. The problem is deterministic linking, not search. A registry gives
   deterministic; search gives probabilistic.
2. 87 topics × 2 volumes is a single YAML file, not infrastructure.
3. The registry survives book rewrites (heading renames → registry
   edits, never broken StaffML links).
4. It's additive with search — Option D can still layer on as a
   fallback for topics not in the registry.

---

## Resume-session task list

When picking this back up, do these in order:

### Step 1 — Decide the mechanism (15 min)
Re-read Options A–D above with fresh eyes. If still going with B,
proceed. If changing direction, update this plan first.

### Step 2 — Define the registry schema (30 min)
Propose a YAML schema at `book/quarto/topics-registry.yaml`:

```yaml
schema_version: 1
generated_at: 2026-mm-ddT..Z
book_revision: <git-sha>
topics:
  kv-cache:
    volume: 1
    chapter: "@sec-inference-kv-cache"
    url_path: "/vol1/chapters/inference/#kv-cache"
    title: "Key-Value Cache"
    last_verified: 2026-mm-dd
  distributed-training:
    volume: 2
    chapter: "@sec-distributed-dp"
    url_path: "/vol2/chapters/distributed/#data-parallelism"
    title: "Data Parallelism"
    last_verified: 2026-mm-dd
  # ... ~85 more
```

Publish as `https://mlsysbook.ai/topics.json` (JSON for consumer ease).

### Step 3 — Book-side generator (1–2h)
Add a Quarto post-render hook or a standalone script:

- Walks `@sec-` anchors across both volumes.
- Cross-references against the StaffML topic list
  (`interviews/vault/taxonomy.json` → 79 topics).
- Emits `topics.json` with one entry per topic.
- Flags untagged topics (book heading exists but no StaffML questions)
  and orphan topics (StaffML has questions but no book section) — this
  is actually a second, useful bug detector.

### Step 4 — StaffML-side consumer (1–2h)
Two options for consumption:

- **Build-time**: Next.js build fetches `topics.json`, inlines a map.
  Links resolve instantly. Requires StaffML redeploy on book release.
- **Runtime**: Question card fetches `topics.json` once per session
  (cached), resolves lazily. No redeploy needed on book release.

Recommend **runtime** — it decouples StaffML from book release cadence.
Cache the registry in localStorage with a 24h TTL; fall back to a bundled
copy if fetch fails.

### Step 5 — Publish + verify (1h)
- Book-side: add `topics.json` to the Quarto `_quarto.yml` resources.
- StaffML: add UI affordance ("read the book section") on question cards.
- Verify: click through 5 random topics from both volumes, confirm
  anchor resolution.

### Step 6 — CI check (30 min)
Add a CI job that flags any StaffML topic missing from the registry,
and any registry entry pointing to a non-existent `@sec-` anchor. This
is how we keep the registry honest over time.

---

## Open questions for the resume session

1. **Volume split timing**: When does `mlsysbook.ai/vol1/` vs `mlsysbook.ai/vol2/`
   actually happen? If it's >6 months out, we can ship a single-volume
   registry now and add `volume:` field later. If imminent, do both
   at once.

2. **Registry host**: Publish under `mlsysbook.ai/topics.json` (CDN-cached,
   simple) vs `mlsysbook.ai/api/topics` (Worker-backed, gives us
   headers/caching control). Default: static JSON unless a reason emerges.

3. **Heading churn**: How often do book section anchors actually change?
   If heading-churn is already low, a registry is overkill and Option A
   (Pagefind search) may suffice. Measure first via
   `git log --follow` on a few chapter files.

4. **Multi-anchor topics**: Some topics (e.g., `quantization`) are
   discussed in multiple book sections. Registry schema should support
   `primary_anchor` + optional `related_anchors[]`.

---

## Files to touch when resuming

| File | Role |
|------|------|
| `book/quarto/topics-registry.yaml` (new) | Source of truth — hand-edited or generated |
| `book/quarto/scripts/emit-topics-registry.py` (new) | Walks `@sec-` anchors, emits JSON |
| `book/quarto/_quarto.yml` | Add `topics.json` to `resources:` |
| `interviews/staffml/src/lib/book-links.ts` (new) | Runtime fetch + cache of registry |
| `interviews/staffml/src/components/QuestionCard.tsx` | Add "read in book" button |
| `interviews/vault/taxonomy.json` | Cross-check against registry coverage |
| `.github/workflows/topics-registry-sli.yml` (new) | CI: registry ↔ vault topic parity |

---

## Non-goals (do not scope-creep)

- Building a full cross-textbook search UX — that's a separate project.
- Mapping sub-topics or paragraph-level anchors — topic→section is the
  bar, nothing finer.
- Backward compatibility with pre-volume-split URLs — after the split,
  the registry is authoritative; old URLs can 301-redirect.

---

**End of resume plan.** Load this file at the start of the next session
along with the current state of `interviews/vault/taxonomy.json` and the
book's `_quarto.yml`.

---

## 2026-04-16 follow-up decision (supersedes the registry recommendation)

After re-reading the four options with fresh eyes, we chose **none of
them** — the registry was solving a coupling problem that a simpler
schema change dissolves entirely.

### The actual decision

Replace `details.deep_dive: {title, url}` (singular) with
`details.resources: [{name, url}]` (multi-valued, author-curated).

**Why this wins over every option above:**

| Option | Left a coupling? | Auto-failure mode |
|---|---|---|
| A: Unified Pagefind | Yes — StaffML ranks against book headings | Book rewrite → wrong ranking |
| B: Registry | Yes — book publishes file, StaffML consumes | Book release → registry regen required |
| C: Click-time search | Yes — assumes book has URL-queryable search | Book loses Pagefind → silent fail |
| D: PSE / Typesense | Yes — crawler freshness dependency | Google indexing lag |
| **Resources list** | **No** | — |

The resources list carries zero coupling to book state. When book URLs
stabilize, authors add `{name: "MLSysBook.ai — KV Cache", url: "..."}`
entries to individual questions at their editorial discretion. When
the book reorganizes, those entries either keep working (URL still
resolves) or the author updates them. No registry, no CI probe, no
manifest, no hostname classifier, no regeneration step.

### What shipped on 2026-04-16 (branch feat/vault-architecture)

1. `schema/question_schema.yaml`: `DeepDive` → `Resource`; `Details.deep_dive`
   → `Details.resources` (multivalued). `schema.py` Pydantic mirror
   updated with https-only + name-length validators.
2. `interviews/vault/corpus.json`: scrubbed 9,495 stale `deep_dive_*`
   fields that predated the YAML cleanup; added empty `resources: []`
   to all 9,657 questions.
3. `interviews/staffml/src/app/practice/page.tsx`: deleted two dead
   "Learn more on MLSysBook.ai" UI blocks (rendered for 0 questions).
4. `interviews/staffml/src/lib/refs.ts`: 178-line hostname classifier
   deleted outright — authors write their own labels now.
5. `interviews/staffml/src/data/chapter-urls.json`, the
   `check-deep-dive-links.py` probe, and
   `.github/workflows/staffml-link-check.yml`: all deleted. The 27-entry
   chapter manifest was 100% 404 against production (`/contents/vol1|vol2/`
   paths vs. live `/contents/core/...`).
6. `TopicDetail.tsx` "Learn first — Read on MLSysBook.ai" block
   removed (was powered by the now-deleted chapter-urls.json).
7. New `Footer.tsx` at page bottom provides stable site-level
   cross-linking back to `mlsysbook.ai`, `/vol1/`, `/vol2/`, and
   the GitHub repo — homepages and volume roots, never anchors.

### Non-goals (now and forever under this model)

- No registry, ever. Adding one would re-introduce the coupling.
- No per-URL health CI. External link rot is an editorial problem,
  not an infrastructure one.
- No hostname-based classification. Authors write the label.

### Re-entry for per-question book resources

When `mlsysbook.ai/vol1/` ships with canonical, stable URLs, a
maintainer makes one editorial pass: for the ~20 chapters that map
to vault topics, add a book `Resource` entry to the 5–15
most-relevant questions per chapter. Estimated scope: 100–300
curated book resources across the corpus. Zero code changes
required — pure YAML authoring.
