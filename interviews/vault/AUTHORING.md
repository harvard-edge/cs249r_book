# Authoring StaffML Vault Questions

This is the single-source authoring reference. If a convention isn't
documented here, it isn't a convention — open an issue.

## Quickstart

```bash
vault new --title "KV Cache Bandwidth Bottleneck on H100" \
          --topic kv-cache-management \
          --track cloud --level L4 --zone diagnosis
```

`vault new` allocates a content-addressed ID, scaffolds a YAML at the
canonical path, and opens it in `$EDITOR`. Fill in every `<TODO>`
marker (and the `competency_area` field, which the scaffold leaves
blank). Save. The validator either accepts the file or injects an
error-comment block at the top and re-opens for iteration.

```bash
vault check --strict      # all 26 invariants on the live corpus
git add interviews/vault/questions/<track>/<area>/<id>.yaml
git commit
```

For LLM-driven authoring, see `vault generate --help` instead — that
path skips this manual flow but still validates against the same
schema.

## Required fields

Every YAML must include these (Pydantic `Question` model). The schema is
the LinkML file at `interviews/vault/schema/question_schema.yaml`; this
table is its human-readable companion.

| field | type | constraint | example |
|---|---|---|---|
| `schema_version` | string | always `"1.0"` | `'1.0'` |
| `id` | string | content-addressed `<track>-<NNNN>`; assigned by `vault new` | `cloud-4539` |
| `track` | enum | `cloud / edge / mobile / tinyml / global` | `cloud` |
| `level` | enum | `L1 / L2 / L3 / L4 / L5 / L6+` | `L3` |
| `zone` | enum | one of 11 zones (see § Zones) | `implement` |
| `topic` | string | one of 87 topics in `taxonomy.yaml` | `quantization-fundamentals` |
| `competency_area` | enum | one of 13 areas (see § Competency areas) | `precision` |
| `title` | string | ≤ 120 chars, plaintext, no trailing period, no LaTeX | `W8A16 KV Cache Expansion` |
| `scenario` | string | ≥ 30 chars, plaintext (no HTML), 1-3 sentences | see worked example |
| `details.realistic_solution` | string | 1-3 sentence canonical answer | see worked example |
| `status` | enum | `draft / published / flagged / archived / deleted` | `draft` for new authoring |
| `provenance` | enum | `human / llm-draft / llm-then-human-edited / imported` | `human` for `vault new` flow |

Optional but **strongly recommended**:

| field | type | when to use |
|---|---|---|
| `bloom_level` | enum (`remember / understand / apply / analyze / evaluate / create`) | always — informs the schema's zone × bloom check |
| `phase` | enum (`training / inference / both`) | always |
| `question` | string ≤ 200 chars | the explicit interrogative; the practice page renders it as "Your task" |
| `details.common_mistake` | string | always — see § Markup conventions |
| `details.napkin_math` | string | for L3+ when there's quantitative reasoning to show |
| `expected_time_minutes` | integer ≥ 0 | typical 5-15 |

## Markup conventions

These two fields use a 3-part bold-marker structure. The structure is
**enforced** by `vault check --strict` (CORPUS_HARDENING_PLAN.md Phase
6 will lift this from regex to LinkML pattern). Authoring without the
markers will fail CI.

### `common_mistake` — Pitfall / Rationale / Consequence

```yaml
details:
  common_mistake: |
    **The Pitfall:** <the wrong intuition or shortcut a candidate takes>
    **The Rationale:** <why that intuition is wrong, in one sentence>
    **The Consequence:** <the operational symptom — latency, cost, failure mode>
```

All three markers are required. Order matters. Use `|` (literal block
scalar) so newlines are preserved; the renderer respects them.

### `napkin_math` — Assumptions / Calculations / Conclusion

```yaml
details:
  napkin_math: |
    **Assumptions & Constraints:**
    - <assumption 1 — hardware, model size, batch, etc.>
    - <assumption 2>

    **Calculations:**
    - <step 1 with units>
    - <step 2>

    **Conclusion:** <one-sentence interpretation of the result>
```

The `Assumptions` marker may be either `**Assumptions & Constraints:**`
or `**Assumptions:**`. The `Conclusion` marker may be either
`**Conclusion:**` or `**Conclusion & Interpretation:**`. The `Calculations`
marker is exact: `**Calculations:**`.

## Worked example

`cloud-4539` (L3, zone=implement, area=precision) — verified by
expert review on 2026-04-28. Use as a template:

```yaml
schema_version: '1.0'
id: cloud-4539
track: cloud
level: L3
zone: implement
topic: quantization-fundamentals
competency_area: precision
bloom_level: apply
phase: both
title: W8A16 KV Cache Expansion
scenario: A 14B parameter LLM is being prepared for serving on a single GPU. The weights in FP16 take 28 GB. The serving target is a concurrent batch of 32 users, each at 4096 max-context tokens. Llama-2-13B-class architecture (40 layers, 40 KV heads, head_dim=128) keeps KV at FP16.
question: 'Calculate the W8A16 weight footprint, then compute the maximum sustainable batch size given the 32-user, 4096-token KV cache requirement, and determine if W8A16 is sufficient.'
details:
  realistic_solution: 'W8A16 quantization compresses weights but leaves the massive KV cache footprint untouched. To hit the 32-user target, W8A16 is insufficient on its own. The team must additionally quantize the KV cache to INT8, cap the maximum context length, or implement PagedAttention to reduce fragmentation.'
  common_mistake: |
    **The Pitfall:** Reporting the weight savings without calculating the corresponding KV cache requirements.
    **The Rationale:** Candidates often assume that if the model weights fit, the system is ready for production serving.
    **The Consequence:** The deployed model experiences immediate Out-Of-Memory (OOM) errors as concurrent users fill up the KV cache.
  napkin_math: |
    **Assumptions & Constraints:**
    - 14B params at W8A16 (1 byte/param), 80GB HBM.
    - 40 layers, 40 KV heads, 128 head_dim, FP16 (2 bytes).

    **Calculations:**
    - W8 Weights: 14B * 1 byte = 14 GB.
    - Available HBM: 80 GB - 14 GB = 66 GB.
    - KV Cache per Token: 2 (K,V) * 40 * 40 * 128 * 2 bytes = 819,200 bytes (~800 KB).
    - KV Cache per User: 800 KB * 4096 = ~3.125 GB.
    - Max Users Supported: floor(66 GB / 3.125 GB) = 21 users.

    **Conclusion & Interpretation:**
    - **Result: Memory-Bound (OOM)**. W8A16 is insufficient to hit the 32-user target by 11 users.
status: published
provenance: llm-draft
expected_time_minutes: 6
```

Reference questions per `(track, level)` cell are populated as
CORPUS_HARDENING_PLAN.md Phase 4's audit findings identify gold-standard
candidates.

## Title conventions

- **Length:** ≤ 120 characters. Pydantic enforces.
- **No trailing period.** "KV Cache Expansion" not "KV Cache Expansion."
- **No LaTeX.** No `$math$`, no `\command`. Indexes break (see
  `book-prose.md` LaTeX-in-attribute-strings § for the rendering bug).
- **No underscores.** `KV-Cache` or `KV Cache`, never `KV_Cache`.
  Underscores break LaTeX `\index{}` macros and the index sort key.
- **No markdown.** No `**bold**`, no `_italic_`. Plaintext only.
- **Descriptive, not generic.** "KV Cache Bandwidth Bottleneck on H100"
  ✓; "KV Cache Q1" ✗.
- **Concrete vendor names** when actually invoked. "Apple Neural Engine"
  ✓; "the on-device accelerator" — only if the question deliberately
  abstracts the hardware. Do not invent vendor names — the audit's
  vendor-fabrication failure mode catches `"Coral Edge TPU XL"` and
  similar.

## Levels and Bloom mapping

| Level | Bloom verb | Cognitive demand |
|---|---|---|
| L1 | remember | recall a fact, definition, ratio |
| L2 | understand | explain a concept; identify a category |
| L3 | apply | execute a calculation given the inputs; pick the matching technique |
| L4 | analyze | decompose a problem; root-cause; pick from competing trade-offs |
| L5 | evaluate | judge a design; weigh alternatives quantitatively |
| L6+ | create | synthesize a new design under unusual constraints (Staff+ scope) |

A common failure (caught by the audit's `level_fit` gate) is **level
inflation**: stamping L4 on a question that's actually a fill-in-the-blank
multiplication (L1 or L2). If you can't articulate the *decomposition*
or *trade-off* the candidate must perform, the question is not L4.

## Zones

Eleven zones. Four pure, six compound, one mastery.

| Zone | Skills it requires |
|---|---|
| **recall** | remember |
| **analyze** | analyze |
| **design** | create |
| **implement** | apply |
| **fluency** | recall + quantify |
| **diagnosis** | recall + analyze |
| **specification** | recall + design |
| **optimization** | analyze + quantify |
| **evaluation** | analyze + design |
| **realization** | design + quantify |
| **mastery** | all four — Staff+ synthesis |

## Zone × Bloom affinity (HARD constraint)

`vault check --strict` rejects YAMLs where `zone` and `bloom_level`
disagree. The matrix:

| zone | admits bloom levels |
|---|---|
| recall | remember, understand |
| fluency | remember, understand, apply |
| analyze | apply, analyze |
| diagnosis | apply, analyze, evaluate |
| evaluation | analyze, evaluate |
| design | apply, analyze, evaluate, create |
| specification | apply, analyze, evaluate, create |
| optimization | apply, analyze, evaluate, create |
| realization | apply, analyze, evaluate, create |
| mastery | analyze, evaluate, create |
| implement | understand, apply, analyze, evaluate, create |

If the validator complains about `zone × bloom_level` mismatch, one of
the two fields is wrong — pick whichever better captures the cognitive
demand of the actual question. When in doubt, trust `bloom_level` and
adjust `zone`.

## Competency areas

13 closed-enum areas (paper §4 Table 3):

`compute`, `memory`, `latency`, `precision`, `power`, `architecture`,
`optimization`, `parallelism`, `networking`, `deployment`,
`reliability`, `data`, `cross-cutting`.

Pick the one that's the question's primary axis. A question about
"quantizing KV cache to fit a memory budget" might touch precision and
memory; pick the one whose lens the answer leans on (here, `memory`
because the constraint is HBM size).

## Topics

87 closed-enum topics in
`interviews/vault/schema/enums.py:VALID_TOPICS`. The CLI tab-completes
when you `vault new --topic <Tab>`. Adding a new topic is a schema
change — open a PR against `enums.py` *and* `taxonomy.yaml`, then run
`vault codegen`.

## Phase

`training`, `inference`, or `both`. A pre-deployment `quantization`
question is `inference`; a `mixed-precision-training` question is
`training`; a question about throughput-vs-latency trade-offs that
applies to either is `both`.

## Gotchas

- **`I/O`, not `IO`.** Both spell-checkers and the index-tag generator
  prefer the slashed form. (See `book-prose.md` index-tag § for why.)
- **Straight apostrophes (`'`), not curly (`’`).** Curly apostrophes
  duplicate-up the index entries (`Moore's Law` ≠ `Moore's Law`).
- **No markdown in titles, scenarios, or questions.** They render as
  literal asterisks, not bold.
- **No `<script>`, `javascript:`, or `data:text/html`** in scenarios.
  The validator rejects these.
- **`visual.path` must resolve.** If you reference an SVG, render it
  first (`render_visuals.py --id <qid>`) — `vault check --strict` fails
  on a dangling reference.
- **`provenance: human`** for `vault new` flow; **`llm-draft`** for
  `vault generate` output; **`imported`** for the historical corpus;
  **`llm-then-human-edited`** when an `llm-draft` was substantively
  rewritten by a human.

## How to test your draft

```bash
# 1. Schema + invariant check (fastest — runs on the whole corpus, <60s)
vault check --strict

# 2. Format-marker compliance (no LLM call)
python3 interviews/vault-cli/scripts/validate_drafts.py --no-llm-judge

# 3. Full LLM-judge gate (level_fit, coherence, bridge, teaching_power)
python3 interviews/vault-cli/scripts/validate_drafts.py
```

## The teaching-power bar (Bloom-graded)

The `teaching_power` gate enforces that a question delivers the cognitive depth
its **level** promises — recall at the bottom of Bloom's ladder, reasoning at the
top. It is NOT a blanket "every question must compute": recall is a first-class
skill, and L1/Remember & L2/Understand questions are warm-up and screening, so a
clean recall item there passes. The gate fails only when a question scores BELOW
the floor for its level:

| Level (Bloom) | Floor | Meaning |
|---|---|---|
| L1 remember / L2 understand | `tp >= 1` | recall is the job — passes |
| L3 apply | `tp >= 2` | must apply a formula, not just recall |
| L4 analyze / L5 evaluate / L6+ create | `tp >= 3` | must reason; recall masquerading as hard fails |

The failure mode is a **mismatch**: a question tagged L5/evaluate that is really an
L1 lookup. The fix is to lift its reasoning to match the level, or relabel it to
the level it actually tests. The gate calibrates against gold-standard exemplars,
not mediocre same-cell peers.

## End-to-end flow

```
vault new
   ↓ (allocates id, scaffolds YAML, opens $EDITOR)
edit YAML — fill <TODO>s, supply competency_area, write all 3 Pitfall/Rationale/Consequence segments
   ↓ (vault edit re-validates on save; injects error block + re-opens on failure)
vault check --strict
   ↓
git add interviews/vault/questions/<track>/<area>/<id>.yaml
git commit
   ↓
push, CI runs staffml-validate-vault.yml (full validator + tests + lint)
```

## When you genuinely need to deviate

The schema's `extra="allow"` on `Question` permits unknown top-level
fields (the `validation_*` and `math_*` audit-stamp fields, for
instance). On `Details`, CORPUS_HARDENING_PLAN.md Phase 6 flips this to
`extra="forbid"` — every legitimate extra detail field must be added
explicitly to the model. If you find yourself reaching for an
unrecognized field, it's a schema-evolution conversation. Open an
issue; don't sneak it in.

## Where to look next

- `interviews/vault/schema/question_schema.yaml` — the LinkML schema (canonical)
- `interviews/vault/schema/enums.py` — the Python frozensets (mirror)
- `interviews/vault-cli/src/vault_cli/models.py` — the Pydantic model (derived)
- `interviews/vault/ARCHITECTURE.md` § 3.6.1 — the markup-convention rationale
- `interviews/vault-cli/docs/CORPUS_HARDENING_PLAN.md` — the active workplan
