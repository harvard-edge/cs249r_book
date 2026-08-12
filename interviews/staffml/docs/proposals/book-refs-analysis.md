# Recommended Reading: connecting StaffML questions back to the textbook

**Status:** analysis / proposal — no code changes yet.
**Worktree:** `MLSysBook-staffml-book-refs` (branch `feat/staffml-book-refs`, off local `dev`).

---

## Framing (settled with VJ)

This is a **"go deeper" pointer, not an answer key.** The reference does *not* claim the
question's answer lives in a given section — it says *"if you want to read about or learn
this topic, here's where the book develops it."* It is tied to the question's **topic**,
not to its solution. Consequences: the label is curiosity/consolidation framing
("Learn more about this" / "Go deeper"), it renders **after** the attempt, and the
topic→chapter mapping is a subject-matter judgment rather than an answer-location hunt.

**Scope (settled):** book-only (no papers/docs/video for v1), covering **both Volume 1
and Volume 2**, at **chapter** granularity.

---

## TL;DR

The plumbing for this already exists and is **dormant** — a `Resource` type and a
`details.resources` field are defined in the schema, the Pydantic-derived TS types,
and the corpus bundle, but **zero of the 10,711 questions populate it** and nothing
renders it. The right move is *not* to hand-author a reference for every question.
Map the **87 topics → book chapters once** and derive each question's reference
automatically from the `topic` it already carries. The book-URL stability concern
that caused the team to defer this (see `Footer.tsx`) is now resolvable: the live
chapter URLs are stable and verified.

---

## 1. What already exists (don't rebuild it)

| Layer | Artifact | State |
|---|---|---|
| Schema | `Resource` class + `details.resources` on `Question` (`interviews/vault/schema/question_schema.yaml`) | ✅ defined |
| TS types | `interface Resource { name; url }`, `details.resources?: Resource[]` (`corpus.ts`) | ✅ defined |
| Bundle | summary corpus carries `details` | ✅ wired |
| Authored data | questions with a populated `resources` | 🔴 **0 of 10,711** |
| UI render | practice page rendering of `resources` | 🔴 none |
| Funnel today | site-level footer link to book homepage only (`Footer.tsx`) | 🟡 coarse |

`Footer.tsx` states the intent and the blocker verbatim:

> *"Per-question book links are deferred until mlsysbook.ai URLs stabilize. In the
> meantime the site-level cross-link to the book homepage — which cannot 404 — gives
> every StaffML page a closing funnel back to the textbook."*

So this proposal is really **"un-defer the per-question link"** — the schema slot was
left open on purpose.

## 2. The mapping problem — and why it's small

There are 10,711 questions but the corpus is already classified along axes that are
*far* smaller and already curated:

```
10,711 questions
     └── 87 topics            ← map THIS to chapters (one-time, ~87 rows)
            └── 13 competency areas
                   └── 5 tracks
```

Every question carries a `topic` (one of 87 curated IDs in `taxonomy_data.yaml`) and a
`competency_area` (one of 13). **Map topic → chapter once** and every question inherits
a textbook reference for free. ~87 curated rows vs. 10,711 hand edits. The taxonomy is
already a knowledge graph with prerequisite/related edges, which we can exploit later
(§6).

The book has 24 content chapters across two volumes (from the Quarto configs):

- **Vol 1 (Foundations → Build → Optimize → Deploy):** introduction, ml_systems,
  ml_workflow, data_engineering, nn_computation, nn_architectures, frameworks, training,
  data_selection, model_compression, hw_acceleration, benchmarking, model_serving,
  ml_ops, responsible_engr, conclusion.
- **Vol 2 (Fleet → Distributed → Deployment → Responsible Fleet):** compute_infrastructure,
  network_fabrics, data_storage, distributed_training, collective_communication,
  fault_tolerance, fleet_orchestration, performance_engineering, inference,
  edge_intelligence, ops_scale, security_privacy, robust_ai, sustainable_ai,
  responsible_ai, conclusion.

The competency areas line up suggestively with the volume split (e.g. `power`,
`reliability`, `networking`, `parallelism` are Vol-2-heavy), which is almost certainly
the "between Vol 1 and Vol 2 / recommended reading" idea that was suggested.

## 3. URL stability — the blocker, now resolved

The deferral reason was "until mlsysbook.ai URLs stabilize." Verified live (HTTP 200):

```
https://mlsysbook.ai/vol1/contents/vol1/training/training.html      → 200
https://mlsysbook.ai/vol2/contents/vol2/inference/inference.html    → 200
```

Pattern: `https://mlsysbook.ai/vol{N}/contents/vol{N}/{chapter}/{chapter}.html`

This maps **1:1** from the `.qmd` source path (`contents/vol1/training/training.qmd`),
so the map can be generated from the Quarto config rather than typed by hand.

**Granularity recommendation — chapter, not section.** Chapter anchors are
human-authored and stable (`# Model Training {#sec-model-training}`). Section anchors
carry auto-generated hash suffixes that regenerate on rebuild and *will* rot:

```
## Iron Law of Training Performance {#sec-model-training-iron-law-training-performance-a53f}
                                                                                      ^^^^ regenerates
```

Link at chapter granularity for v1 (optionally `…training.html#sec-model-training` for
the chapter top). Defer section deep-links until anchors are stabilized or a checker
guarantees them.

**Guardrail instead of indefinite deferral:** add a build-time link-checker that fails
the vault build if a mapped chapter file doesn't exist (source-side check — no network).
That converts "wait until URLs stabilize" into "URLs are enforced to be valid," which is
why this can ship now.

## 4. Field / schema design

Three options considered:

| Option | Mechanism | Pro | Con |
|---|---|---|---|
| A. Reuse `details.resources` per question | hand-author name+url on each YAML | uses existing field | 10,711 edits; semantics lost (book vs. arbitrary link); diff noise |
| **B. Derive `book_refs` from a topic→chapter map** ✅ | one `topic_chapter_map.yaml`; vault build joins → emits `book_refs` into corpus | ~87 rows; auto-coverage; clean YAML diffs; matches how SVG visuals are kept out of YAML | needs a small build step + a new summary field |
| C. Per-question column in every YAML | store resolved chapter on each YAML | explicit | reintroduces the 10k-edit + diff-noise problem |

**Recommend B, with A as an optional override.** Source of truth is a single
`interviews/vault/schema/topic_chapter_map.yaml`:

```yaml
# topic_chapter_map.yaml  (one row per topic; 87 total)
roofline-analysis:
  primary:   { volume: 1, chapter: hw_acceleration }
  also_see:  [ { volume: 1, chapter: benchmarking } ]
gpu-compute-architecture:
  primary:   { volume: 1, chapter: hw_acceleration }
```

A `BookRef` shape (volume + chapter + derived title/url) gets emitted into the summary
bundle so the card renders synchronously (no extra worker fetch). The dormant
`details.resources` field stays — now meaningfully used for **per-question overrides**:
a specific paper, a particular section, or a hand-picked extra beyond the topic default.
Give `Resource` an optional `kind: book | paper | docs | video` so the structure is
ready for the "broader set of sources" the user mentioned, without a future migration.

## 5. Where & when to render (the pedagogy)

The single most important design decision is **when** the link appears.

> **Recommendation: reveal recommended reading *after* the attempt, not before.**
> A student who can jump to the chapter before thinking will read instead of reason —
> which defeats the napkin-math, "reason under uncertainty" purpose of StaffML. The
> textbook is the *consolidation / go-deeper* step after the productive struggle, not a
> shortcut around it. The card answers "want to learn more about this?" — not "here's
> where the answer was."

Concretely:

- A **"Learn more in the textbook"** card rendered below the revealed solution on the
  practice page (the details render around `practice/page.tsx:1146`), showing the
  primary chapter with a Volume badge, plus 0–2 "also see" chapters.
- One **"why this chapter" line** per ref — connect the question's concept to the
  chapter, not a bare link. ("This question is about PUE as a multiplier — *Sustainable
  AI* develops the full datacenter energy model.")
- **Struggle-gated depth:** show only the primary chapter by default; expand to
  prerequisites + related chapters when the student got it wrong or asked for a hint
  (the wrong-answer path already exists in the scoring flow).

This upgrades the footer funnel (coarse, site-level, intent-stage) to a per-question,
concept-level funnel — which is exactly the gap the footer comment describes.

## 6. Other ideas worth folding in

1. **Prerequisite-aware remediation.** The taxonomy already encodes `prerequisite`
   edges between topics. On a wrong answer, recommend the *prerequisite* topic's chapter
   ("Shaky on this? Review X first"), not just the current one. This is the highest-value
   pedagogical add and it's almost free given the existing graph.
2. **Bidirectional links.** Chapters could link *out* to a filtered StaffML practice set
   (`/practice?topic=…`). Reader finishes a chapter → practices; student finishes a
   question → reads. Closes the loop both ways and reinforces the "one ecosystem" framing.
3. **Source tiers / multi-source (the user's "future" ask).** The `kind` enum (§4) lets a
   question point at book (primary) → canonical paper → vendor docs → talk, rendered as a
   tiered list with the textbook always first. No schema migration needed later.
4. **Track-sensitive mapping.** The same topic can be taught differently across volumes
   (e.g. inference at the edge vs. in the fleet). Allow the map to vary the chapter by
   `track` where it matters; default to a single primary otherwise.
5. **Coverage report.** A build artifact listing topics with no chapter mapped — keeps the
   87-row map honest as the taxonomy grows, and surfaces genuinely book-orphaned topics.

## 7. Phasing & effort

| Phase | Scope | Effort | Value |
|---|---|---|---|
| **1 — MVP** | `topic_chapter_map.yaml` (87 rows) + build join → `book_refs` in bundle + chapter-level card after reveal + link-checker | M | High — un-defers the feature for *all* questions at once |
| **2 — Pedagogy** | prerequisite-on-wrong-answer; "why this chapter" lines; per-question `resources` overrides + `kind` enum | M | High |
| **3 — Ecosystem** | bidirectional chapter→practice links; multi-source tiers; section deep-links once anchors checked | L | Medium |

## 8. Open questions for VJ

1. **Granularity:** chapter-level for v1 (recommended), or do you want section-level deep
   links from the start (needs anchor-stability work first)?
2. **Authoring the 87-row map:** want me to draft a first-pass `topic_chapter_map.yaml`
   (topic → chapter) for you to correct, or do you want to drive the mapping since you
   know where each concept is *sourced* in the book?
3. **Reveal timing:** confirm "after the attempt" is the right pedagogy (vs. always
   visible).
4. **Scope of "broader sources":** book-only for v1 with the `kind` enum reserved for
   later, or seed a few paper/doc references now?
