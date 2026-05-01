# Vault Chain Coverage Roadmap

**Status:** active workstream
**Branch:** `yaml-audit` (off `dev`)
**Worktree:** `/Users/VJ/GitHub/MLSysBook-yaml-audit`
**Last updated:** 2026-05-01 (Phase 3.c pilot run + 3.d promotion shipped; 4 new draft questions in corpus, awaiting human review)

This document is the canonical resumable plan for the vault chain rebuild
+ corpus growth work. **Future Claude sessions: read the "Resume Here"
section first, then the "Progress Log" to see what was done, then the
relevant Phase section for the step you're picking up.**

---

## Resume Here (read first)

### How to resume in a new session

1. **Confirm worktree + branch:**
   ```bash
   pwd          # /Users/VJ/GitHub/MLSysBook-yaml-audit
   git branch   # * yaml-audit
   git log --oneline -5
   ```
2. **Run baseline validators** to confirm tree is in known-good state:
   ```bash
   vault check --strict           # expect: 10,701 loaded, 0 invariant failures
   vault build --local-json      # expect: clean build, releaseId=dev, 9438 published
   ```
3. **Read the most recent entry in the Progress Log section below** to see
   what step was last completed, what was decided, and what's next.
4. **Check for in-flight artifacts:**
   ```bash
   ls -la interviews/vault/chains.proposed*.json 2>/dev/null
   ls -la interviews/vault/gaps.proposed.json 2>/dev/null
   ```
5. **Pick up at the next step** in the relevant Phase section, follow its
   detailed substeps, then **append a new Progress Log entry** when done.

### Current state snapshot

- **Sidecar architecture:** active. `chains.json` is authoritative; YAML
  `chains:` field stripped from all 10,701 question YAMLs.
- **Hierarchy:** all questions at `interviews/vault/questions/<track>/<area>/<id>.yaml`.
- **Live chain count:** 879 (post Phase 1: 373 primary + 506 secondary, tier field tagged).
- **Gap backlog:** 138 (strict) + 269 (lenient) = 407 entries across
  `gaps.proposed.json` and `gaps.proposed.lenient.json`.
- **Pre-Gemini chains backed up:** `interviews/vault/chains.json.bak` (726 chains).
- **Validators:** all green as of last commit `1ac7d4c56`.
- **UI tests:** `interviews/staffml/tests/chain-and-vault-smoke.mjs` — 13/13 pass.
- **Remote:** `origin/yaml-audit` pushed through `1ac7d4c56`.
- **Dev branch state:** `origin/dev` has the hierarchy migration merged
  (commit `99b37f021`) but NOT the Gemini chain rebuild or sidecar work.
  Re-merge to dev pending after Phase 1.

### What we're building (motivation)

StaffML is an ML systems interview prep platform. Chains are pedagogical
progressions through Bloom levels (L1→L2→L3→L4→L5→L6+) within a single
topic, where each member builds on the previous. They power:

- Daily challenge sequencing (must show progression, not random)
- "Drill this topic" structured paths
- Mock interview question ordering

**373 chains is too sparse:** only 54 of 87 topics have any chain at all,
and edge/mobile/tinyml tracks have <0.6 chains/topic vs cloud's 3.0.
Target: 700-900 chains (~30-40% of corpus chained).

---

## Phase 1 — Second-pass coverage build

**Status:** `complete` (2026-04-30)
**Goal:** double chain count from 373 → 700-800 by relaxing the prompt for
the ~150 buckets that produced 0 chains in the strict pass. Tag these as
`tier: secondary` so the UI can deprioritize them.

### Phase 1 substeps

#### 1.1 — Diagnose uncovered buckets

**Deliverable:** `interviews/vault-cli/scripts/diagnose_chain_coverage.py`

**What it does:**
- Loads YAML corpus (published only) and current `chains.json`
- For each (track, topic) bucket, reports:
  - Number of published questions
  - Current chain count
  - List of qids
- Output: `interviews/vault/chain-coverage.json` with two arrays:
  - `uncovered_buckets`: ≥3 questions, 0 chains
  - `under_covered_buckets`: ≥6 questions, only 1 chain
- Print summary to stdout: per-track totals, biggest gaps

**Validation:** Expect ~150 uncovered buckets (sanity check on numbers).
Cloud bucket count should be much lower than edge/mobile/tinyml.

**Estimated time:** 30 min

**Files:**
- create: `interviews/vault-cli/scripts/diagnose_chain_coverage.py`
- create: `interviews/vault/chain-coverage.json` (gitignored — regeneratable)

**Commit message:** `feat(vault-cli): diagnose_chain_coverage.py — surface buckets needing chains`

---

#### 1.2 — Build relaxed Gemini prompt

**Deliverable:** `--mode lenient` flag added to `build_chains_with_gemini.py`

**Prompt changes vs current strict (in `PROMPT_TEMPLATE`):**

```
LEVEL PROGRESSION RULES (LENIENT MODE):
  - Each consecutive pair of members satisfies: cand_level - prev_level ∈ {0, 1, 2, 3}
  - Strongly prefer +1; +2 acceptable; +3 only when no intermediate exists
  - Δ=0 (same-level pair) IS allowed when both questions clearly share
    the same scenario thread but explore different angles of it (e.g.,
    diagnosis vs design at the same Bloom level on the same setup).
    Do NOT use Δ=0 for unrelated same-level questions.
  - Never backward (Δ < 0)

DEFAULT MODE INSTRUCTION:
  - Find at least one chain per bucket if any pedagogical clustering
    exists. Only return zero chains when the questions are genuinely
    unrelated even on the loosest reading.
```

Other constraints unchanged (single-topic, single-track, 2-6 members,
multi-membership cap=2 for L1/L2 anchors).

**Code change:** Add `--mode {strict,lenient}` to argparse; lenient
swaps in the LENIENT prompt. Keep strict prompt as default.

**Validation:** Run on one previously-uncovered bucket as smoke test
before the full sweep.

**Estimated time:** 30 min

**Files:**
- modify: `interviews/vault-cli/scripts/build_chains_with_gemini.py`

**Commit message:** `feat(chains): --mode lenient for second-pass coverage on uncovered buckets`

---

#### 1.3 — Update validators to accept lenient chains + tier metadata

**Deliverable:** `validate_chain` in build script + `apply_proposed_chains.py`
both accept Δ ∈ {0, 1, 2, 3} (Δ=0 only when caller is in lenient mode).
Every chain produced by the lenient sweep is tagged `tier: "secondary"`.

**Schema additions:**
- Each chain object in chains.proposed.lenient.json gets `"tier": "secondary"`
- `apply_proposed_chains.py` validation tolerates the new tier field

**Validation:** New unit test in `interviews/vault-cli/tests/test_chain_validation.py`
covering Δ=0 acceptance under lenient mode, rejection under strict.

**Estimated time:** 30 min

**Files:**
- modify: `interviews/vault-cli/scripts/build_chains_with_gemini.py`
- modify: `interviews/vault-cli/scripts/apply_proposed_chains.py`
- create: `interviews/vault-cli/tests/test_chain_validation.py` (or add to existing)

**Commit message:** `feat(chains): tier field + lenient-mode Δ=0 acceptance for same-scenario pairs`

---

#### 1.4 — Run second sweep on uncovered buckets only

**Deliverable:** `interviews/vault/chains.proposed.lenient.json` with chains
for the ~150 uncovered buckets.

**Command:**
```bash
python3 interviews/vault-cli/scripts/build_chains_with_gemini.py \
  --mode lenient \
  --buckets-from interviews/vault/chain-coverage.json \
  --output interviews/vault/chains.proposed.lenient.json
```

(May need to add `--buckets-from` flag to take an explicit bucket list.)

**Expected output:**
- ~30-40 calls (within 250/day budget)
- ~200-400 chains added
- ~30 min wall time

**Process management:**
- Launch with `nohup` so shell exit doesn't kill it
- Monitor via `chains.proposed.lenient.json` size growth (incremental persistence)
- Check progress: `python3 -c "import json; print(len(json.load(open('interviews/vault/chains.proposed.lenient.json'))))"`

**Validation:**
- Per-track distribution should now be more balanced
- Δ=0 chains should be a meaningful fraction (~10-20%)
- No structural violations on apply

**Estimated time:** 30 min wall + 10 min review

**Files:**
- create: `interviews/vault/chains.proposed.lenient.json`
- update: `interviews/vault/gaps.proposed.json` (lenient pass also emits gaps)

**Commit message:** *(no commit yet — staging file, see step 1.5)*

---

#### 1.5 — Merge primary + secondary into chains.json

**Deliverable:** `interviews/vault-cli/scripts/merge_chain_passes.py`

**Logic:**
1. Load live `chains.json` → all entries get `tier: "primary"` if not set
2. Load `chains.proposed.lenient.json` → entries already have `tier: "secondary"`
3. **Reject any secondary chain whose qids are already in 2 primary chains**
   (multi-chain cap is 2; can't exceed)
4. **Reject any secondary chain whose qids are already in 1 primary chain
   AND the qid is not L1/L2** (cap rule)
5. Concatenate accepted chains, sort by chain_id, write `chains.json`
6. Report stats: primary kept, secondary added, secondary rejected (and why)

**Validation:** Run `apply_proposed_chains.py --proposed chains.json --dry-run`
against the merged file as final structural gate.

**Estimated time:** 1 hour

**Files:**
- create: `interviews/vault-cli/scripts/merge_chain_passes.py`
- modify: `interviews/vault/chains.json`

**Commit message:** *(no commit yet — see 1.6)*

---

#### 1.6 — Rebuild + verify + commit

**Steps:**
1. `vault check --strict` → expect 0 invariant failures
2. `vault build --local-json` → clean, chainCount jumps from 373 to ~700
3. Restart staffml dev server, run playwright suite → expect 13/13 (or
   add tier-related test if scope justifies it)
4. Commit Phase 1 work as a single conceptual commit:

**Commit message:**
```
feat(vault): Phase 1 — second-pass chain coverage build (~373 → ~700)

Diagnoses uncovered (track, topic) buckets and runs a relaxed Gemini
sweep targeting them. New chains are tier="secondary"; pre-existing
chains stay tier="primary".

Tools:
  - diagnose_chain_coverage.py: surface buckets without chains
  - build_chains_with_gemini.py: --mode lenient adds Δ=0 + Δ=3 acceptance
  - merge_chain_passes.py: merges primary + secondary with cap enforcement

Coverage gains:
  - Total chains: 373 -> N (TBD after run)
  - Per-track: edge/mobile/tinyml lifted from <0.6 to >1.0 chains/topic
  - Topics with ≥1 chain: 54/87 -> M/87 (TBD)

Validation: vault check --strict 0 failures, vault build clean,
playwright UI suite 13/13 pass.
```

5. `git push origin yaml-audit`

**Estimated time:** 30 min

---

### Phase 1 risks & mitigations

| Risk | Mitigation |
|---|---|
| Gemini at lenient mode hallucinates poor chains | tier=secondary so UI deprioritizes; merge step rejects cap violations |
| Cross-pass duplicate qids | merge_chain_passes explicitly rejects |
| Δ=0 rule too permissive (random same-level pairs accepted) | Prompt restricts to "shared scenario thread"; spot-check 10 random Δ=0 chains post-sweep |
| Daily Gemini budget tight | Phase 1 needs ~30-40 calls; well under 250 |

---

## Phase 2 — Tier surfacing (schema + UI)

**Status:** `complete` (2026-05-01)
**Goal:** chains carry their tier as authoritative metadata; UI prefers
primary chains in default surfaces, exposes secondary in "more paths."

### Phase 2 substeps

#### 2.1 — Schema migration

**Deliverable:** `tier` field on chain entries is required (default "primary").

**Changes:**
- `chains.json`: every chain has `tier: "primary" | "secondary"`
- `interviews/vault-cli/src/vault_cli/validator.py`: tier required, default to "primary" if missing
- `interviews/vault-cli/src/vault_cli/legacy_export.py`: include tier in
  corpus.json output as `chain_tiers: {chain_id: "primary"|"secondary"}`
  per question (mirrors `chain_positions` shape)

**Files:**
- modify: `interviews/vault/chains.json` (backfill primary tag)
- modify: `interviews/vault-cli/src/vault_cli/validator.py`
- modify: `interviews/vault-cli/src/vault_cli/legacy_export.py`

**Estimated time:** 30 min

**Commit message:** `feat(vault): tier field on chains, derived chain_tiers in corpus.json`

---

#### 2.2 — TypeScript types

**Files:**
- modify: `interviews/staffml/src/lib/corpus.ts`
  - `Question` interface: add `chain_tiers?: Record<string, "primary" | "secondary">`
  - `ChainInfo` interface: add `tier: "primary" | "secondary"`
  - `getChainForQuestion` populates tier
  - `getAllChainsForQuestion` populates tier per chain
  - **New:** `getPrimaryChainForQuestion(qid)` — returns first primary, falls back to first secondary

**Estimated time:** 15 min

**Commit message:** *(included in 2.3 commit)*

---

#### 2.3 — UI: prefer primary in default surfaces

**Files to modify:**
- `interviews/staffml/src/components/ChainStrip.tsx` — default to primary; subtle "Alternative path" hint when only secondary
- `interviews/staffml/src/app/practice/page.tsx` — "next in chain" prefers primary
- `interviews/staffml/src/app/explore/page.tsx` — filter dropdown "Primary only / All"
- Any daily-challenge / mock-interview routing — sequence primary chains only
- URL param: `?chain=<id>` already supported via Phase A; verify both tiers reachable

**Estimated time:** 3-4 hours

**Commit message:** `feat(staffml): UI tier-awareness — primary chains default, secondary opt-in`

---

#### 2.4 — Tests

**Files:**
- modify: `interviews/staffml/tests/chain-and-vault-smoke.mjs`
  - Add: "primary chain rendered by default when question has both"
  - Add: "secondary chain reachable via ?chain= URL param"
- Run full suite → expect 15/15 pass

**Estimated time:** 1 hour

**Commit message:** `test(staffml): tier-aware playwright cases (primary default, secondary via URL)`

---

#### 2.5 — Push

`git push origin yaml-audit`

---

## Phase 3 — Gap-driven question authoring

**Status:** `pilot run shipped (3.c + 3.d); 3.e gated on human review of drafts`
**Goal:** Use the 138+ entries in `gaps.proposed.json` to author new
questions filling missing rungs, validated independently before commit.
This is the durable corpus growth strategy.

### Phase 3 substeps

#### 3a — Authoring tool design (1 day)

**Deliverable:** `interviews/vault-cli/scripts/generate_question_for_gap.py`

**Inputs:**
- A single gap entry from `gaps.proposed.json`:
  ```json
  {
    "track": "edge",
    "topic": "memory-mapped-inference",
    "missing_level": "L3",
    "between": ["edge-0220", "edge-0224"],
    "rationale": "Bridge demand-paging concept to fault-tolerance application"
  }
  ```

**What it does:**
1. Loads the `between` questions in full (scenario + question + solution)
2. Loads 2-3 exemplar questions from same `(track, topic)` at the target level
3. Loads the question Pydantic schema as a textual summary
4. Prompts Gemini 3.1 Pro Preview with all the above + instruction to author
   a question matching the schema, fitting the bridge requirement
5. Validates output against Pydantic schema
6. Writes to `interviews/vault/questions/<track>/<area>/<auto-id>.yaml.draft`
   (`.draft` suffix prevents `vault check` from loading it as published)
7. Records authoring metadata: gap source, model, timestamp

**Validation:** Schema check passes before persisting. Filename matches
`<track>-<NNNN>.yaml.draft` convention.

**Estimated time:** 1 day

**Files:**
- create: `interviews/vault-cli/scripts/generate_question_for_gap.py`

**Commit message:** `feat(vault-cli): generate_question_for_gap.py — Gemini-author candidate questions from gaps file`

---

#### 3b — Validation framework (1 day)

**Deliverable:** `interviews/vault-cli/scripts/validate_drafts.py`

**Checks per draft:**
1. **Schema validation** (Pydantic) — same gates as published questions
2. **Originality:** embed the draft + nearest-neighbor cosine in same bucket;
   reject if cosine > 0.92 (too duplicative of existing)
3. **Level fit:** LLM-judge call — "is this question's cognitive load
   consistent with `level=L<N>`?" Sample 5 existing L<N> questions in
   the same topic for calibration.
4. **Scenario coherence:** Gemini check — scenario, question, and
   realistic_solution should be internally consistent
5. **Bridge check:** the new question genuinely chains between the gap's
   `between` questions (LLM-judge with both `between` questions in context)

**Output:** Scorecard JSON per draft:
```json
{
  "draft_id": "edge-2545",
  "schema_ok": true,
  "originality_cosine": 0.81,
  "level_fit": "yes",
  "scenario_coherence": "yes",
  "bridge_check": "yes",
  "verdict": "pass",
  "rationale": "..."
}
```

**Estimated time:** 1 day

**Files:**
- create: `interviews/vault-cli/scripts/validate_drafts.py`

**Commit message:** `feat(vault-cli): validate_drafts.py — schema + originality + level + coherence + bridge checks`

---

#### 3c — Pilot run on highest-value gaps (1-2 hours)

**Steps:**
1. From `gaps.proposed.json`, prioritize:
   - Gaps where the bucket has 4+ questions already (just missing the bridge)
   - Gaps in tracks with low chain density (tinyml, mobile)
2. Pick top 30 gaps
3. Run `generate_question_for_gap.py` on each → 30 draft files
4. Run `validate_drafts.py` → expect ~60-75% pass rate
5. Manual review of passing drafts (~30 min)

**Files:**
- create: `interviews/vault/questions/<track>/<area>/<auto-id>.yaml.draft` × 30
- create: `interviews/vault/draft-validation-scorecard.json`

**Commit message:** *(no commit until 3d)*

---

#### 3d — Promote drafts (1 hour)

**For each accepted draft:**
1. Rename `.yaml.draft` → `.yaml`
2. Add `authoring: { origin: "gemini-3.1-pro-preview", reviewed_by: "<user>", date: "<>" }`
3. Set `status: published` (or `draft` for further iteration)
4. Commit each individually for granular review:
   `feat(vault): add <id> filling chain gap (track=<>, topic=<>, level=<>)`

---

#### 3e — Re-run chain build to absorb (~30 min)

```bash
python3 interviews/vault-cli/scripts/build_chains_with_gemini.py --all \
  --output interviews/vault/chains.proposed.json
python3 interviews/vault-cli/scripts/apply_proposed_chains.py
vault check --strict
vault build --local-json
```

**Expected:** chain count grows by ~50% of newly authored questions
(since they were authored TO fit chains).

---

#### 3f — Iterate weekly until gap count < threshold

Repeat 3c-3e weekly, tracking metrics:
- Gaps closed per week
- Chains added per week
- Total corpus size

---

## Phase 4 — Other pending items (parallel/ongoing)

These can slot between major phases. Order roughly:

### 4.1 — Chain audit CI gate
**When:** before Phase 3 (gates corpus growth)
**Files:** `.github/workflows/staffml-validate-vault.yml`
**Change:** add `vault chains audit --strict --max-orphan-rate 0.02 --max-drift-regression 0.05`

### 4.2 — Multi-chain UI verification
**Status:** audited 2026-05-01 — `0` qids in >1 chain (lenient sweep
was scoped to uncovered buckets, so no overlap with primary). Becomes
live once Phase 3 authoring lands; deferred until then.

### 4.3 — Authoring UX integration
**When:** after Phase 3
**Deliverable:** `vault new` post-write hook calls `vault chains suggest` to
propose chain memberships for the new question.

### 4.4 — Deploy pipeline lockstep
**When:** anytime (independent)
**Files:** `.github/workflows/staffml-publish-live.yml`
**Change:** wait for cloudflare worker `release_id` match before site deploy.

### 4.5 — Cross-encoder reranking experiment
**When:** low priority
**Action:** re-run `interviews/vault-cli/scripts/cross_encoder_rerank_experiment.py`
on a beefier machine; OOM'd on 16GB.

### 4.6 — Periodic chain rebuild automation
**When:** after Phase 1 + 2 stabilize
**Deliverable:** weekly cron action that runs `build_chains_with_gemini.py`
on incremental corpus changes; opens auto-PR with proposed delta.

### 4.7 — Chain decay detection
**When:** after Phase 2
**Deliverable:** pre-commit hook recomputes embedding for changed YAMLs;
flags chain mate cosine drops below threshold.

### 4.8 — Update docs
**Status:** complete 2026-05-01 — `f086b6f42`. ARCHITECTURE.md §3.6
captures sidecar + hierarchy + tier; README.md gains a "Chain build
pipeline" section + updated layout/status.

### 4.9 — gitignore CI guard
**When:** anytime, low effort
**Deliverable:** CI check that every YAML under `interviews/vault/questions/`
is git-tracked (catches the `data/` regression class).

### 4.10 — Merge yaml-audit → dev (re-merge with chain rebuild)
**When:** after Phase 1
**Action:** `git merge --no-ff yaml-audit` from dev worktree, push.
Triggers `staffml-validate-vault.yml` on the full corpus including new
sidecar architecture.

---

## Recommended execution order

```
Week 1
  Day 1-2:  Phase 1 (1.1 → 1.6)
  Day 3:    4.1 (CI gate) + 4.10 (merge to dev)
  Day 4-5:  Phase 2 (2.1 → 2.5)

Week 2
  Day 1:    4.2 (multi-chain verify) + 4.7 (chain decay)
  Day 2-5:  Phase 3a + 3b (authoring tool + validator)

Week 3
  Day 1-2:  Phase 3c (pilot batch)
  Day 3:    Phase 3d-3e (promote + re-chain)
  Day 4:    4.6 (periodic rebuild) + 4.4 (deploy lockstep)
  Day 5:    4.8 (docs) + 4.9 (gitignore CI)

Week 4+
  Phase 3f iterations on remaining gaps
  4.3 (authoring UX), 4.5 (cross-encoder)
```

---

## Progress Log

> **Append-only.** New entries at the bottom. Each step that ships should
> add a dated entry with: what was done, validation results, commits,
> notes for the next session.

---

### 2026-04-30 — Roadmap document created

**What was done:**
- Created this roadmap document (`interviews/vault-cli/docs/CHAIN_ROADMAP.md`)
- Captures full plan: Phase 1-4, recommended execution order, resume instructions
- Initialized Progress Log for append-only step notes

**State at this point:**
- Branch `yaml-audit` at `1ac7d4c56` (Gemini chain rebuild applied)
- 373 chains in `chains.json` (sidecar-authoritative, no `chains:` field in YAMLs)
- 138 gaps in `gaps.proposed.json` (authoring backlog)
- 10,701 YAMLs at hierarchical paths (`<track>/<area>/<id>.yaml`)
- vault check --strict: 0 failures
- vault build: clean, chainCount=373
- playwright UI suite: 13/13
- `chains.json.bak` is the pre-Gemini 726-chain backup

**Files committed in this session (chronological):**
- `aa9373f88` paths.py + scripts hierarchy-tolerant
- `2a48177ac` migrate 10,701 YAMLs to hierarchical layout
- `f7d7a328a` path-vs-body invariants in vault check
- `d476b63df` prune stale chains.json entries
- `367cda468` rescue 924 yamls from gitignore via `/data/` anchor
- `efeedb8cc` sidecar architecture (chains.json authoritative, strip YAML chains:)
- `8423dcb08` build_chains_with_gemini.py + apply_proposed_chains.py
- `0b14e08b5` summarize_proposed_chains.py
- `d8a55f333` strict progression rules + multi-chain cap
- `681e40463` gap detection + multi-chain UI helpers
- `d82a4f00a` Gemini CLI exit-1 tolerance + inter-call backoff
- `1ac7d4c56` apply 373 chains from Gemini rebuild

**Next step:** Phase 1.1 — write `diagnose_chain_coverage.py` to identify
the ~150 buckets that need a second pass.

---

### 2026-04-30 — Phase 1.1: diagnose_chain_coverage.py

**What was done:**
- Added `interviews/vault-cli/scripts/diagnose_chain_coverage.py`
- Loads published corpus via `vault_cli.policy.is_published` (single source
  of truth — same predicate as `vault build`) and current `chains.json`.
- Buckets by `(track, topic)`; emits per-bucket question_count, chain_count,
  qids, chain_ids; classifies into `uncovered_buckets` (≥3 q, 0 chains) and
  `under_covered_buckets` (≥6 q, ≤1 chain). Prints per-track summary +
  top-10 uncovered for quick read.
- Output: `interviews/vault/chain-coverage.json` (gitignored —
  regeneratable). Added `/chain-coverage.json` to `interviews/vault/.gitignore`.

**Validation results (run on tree at `1ac7d4c56` + this commit):**
- 313 buckets, 9438 published questions (matches `vault build` ✓), 373 chains.
- **Uncovered buckets: 211** (roadmap estimate was ~150). Higher than expected
  but same order of magnitude — the gap is mostly in `global` (32 uncovered
  with avg ~6.5 q/bucket) and a long tail of small `mobile`/`tinyml` topics.
- **Sanity check passes:** cloud chain density 2.95/topic vs edge 0.64,
  mobile 0.74, tinyml 0.80, global 0.00 — matches the "<0.6 vs 3.0" claim.
- Every chain in `chains.json` lands in a published bucket (0 orphan
  chain-buckets) — confirms chains.json is consistent with the released
  corpus.
- Notable: **`cloud:roofline-analysis` (144 questions, 0 chains)** is the
  single largest uncovered bucket — first-pass Gemini sweep missed it
  entirely despite cloud's high overall coverage. Worth a targeted retry
  in Phase 1.4.

**Per-track table:**
```
track       buckets  questions  chains  chains/topic  uncov  undercov
cloud            82       4028     242          2.95     41         0
edge             76       2077      49          0.64     58         0
global           48        313       0          0.00     32         0
mobile           62       1818      46          0.74     46         1
tinyml           45       1202      36          0.80     34         0
```

**Validators (re-run as a sanity gate on the unmodified corpus):**
- `vault check --strict` → 10,701 loaded, 0 invariant failures ✓
- `vault build --local-json` → releaseId=dev, 9438 published, chainCount=373 ✓

**Files committed:**
- `interviews/vault-cli/scripts/diagnose_chain_coverage.py` (new)
- `interviews/vault/.gitignore` (add `/chain-coverage.json`)
- this Progress Log entry

**Notes for next session:**
- Higher-than-expected uncovered count (211 vs ~150) means Phase 1.4
  Gemini call budget should be re-checked: at 1 call per ~5-7 buckets we
  may need ~30-45 calls (still under 250/day, but worth monitoring).
- The single `under_covered` bucket (`mobile:transformer-systems-cost`,
  54 questions, 1 chain) is a candidate for a focused retry alongside
  the lenient sweep.
- `chain-coverage.json` is regenerated each run; do not check it in. The
  roadmap step 1.4 will read this file via `--buckets-from`.

**Next step:** Phase 1.2 — add `--mode lenient` to
`build_chains_with_gemini.py` (relaxed Δ rules + Δ=0 for shared-scenario
pairs).

---

### 2026-04-30 — Phase 1.2 + 1.3: --mode lenient + tier field

**What was done:**
- `build_chains_with_gemini.py`: split `PROMPT_TEMPLATE` into
  `STRICT_PROMPT_TEMPLATE` and `LENIENT_PROMPT_TEMPLATE`. The lenient
  template tells Gemini to accept Δ ∈ {0, 1, 2, 3} where Δ=0 is
  shared-scenario only and Δ=3 is last-resort. Single `MODE_CONFIG`
  dict maps mode → (template, allowed Δ set) so `build_prompt` and
  `validate_chain` stay in lockstep.
- `validate_chain` now takes `mode=`; rejects Δ=0 / Δ=3 in strict,
  accepts them in lenient. Both modes still reject backward Δ,
  multi-topic, and out-of-range size.
- `process_batch` tags lenient-mode chains with `tier: "secondary"` and
  uses chain_id suffix `-secondary` so primary/secondary IDs never collide.
- New CLI flags: `--mode {strict,lenient}` (default strict);
  `--buckets-from <chain-coverage.json>` to restrict the run to the
  uncovered_buckets list from `diagnose_chain_coverage.py`.
- `apply_proposed_chains.py`: docstring note acknowledging tier is
  intentionally unvalidated (UI hint, not a structural invariant).
  No logic change — its existing non-strict monotonicity check already
  accepts Δ=0.
- `tests/test_chain_validation.py`: 19 cases covering both modes.
  Loads the script via `importlib` (it's not in the importable
  `vault_cli` package). All 19 pass.

**Smoke check:**
- `--dry-run --buckets-from chain-coverage.json --mode lenient` →
  17 calls planned, 211 buckets selected, well under the 200/day cap.
- Direct call to `validate_chain` on 12 hand-built cases (Δ=0/Δ=3
  accept-in-lenient/reject-in-strict, backward, multi-topic, sizes 1
  and 7) → 12/12 pass.

**Commit:** `6cef27ea2 feat(chains): --mode lenient + tier field for second-pass coverage`

---

### 2026-04-30 — Phase 1.4 + 1.5 + 1.6: lenient sweep, merge, validate

**What was done:**
- **Phase 1.4 — lenient sweep:** Ran
  `build_chains_with_gemini.py --mode lenient --buckets-from chain-coverage.json`
  against the 211 uncovered buckets. 17 Gemini-3.1-pro-preview calls,
  ~22 min wall time. Output: `chains.proposed.lenient.json` with
  **506 chains** (above the 200-400 estimate) and
  `gaps.proposed.lenient.json` with **269 new gaps**. The validator
  caught a few cross-bucket and Δ=4 hallucinations in calls 15 + 16
  and rejected them inline.
- **Phase 1.5 — merge:** Wrote
  `interviews/vault-cli/scripts/merge_chain_passes.py`. Backfills
  `tier=primary` on existing chains, layers in `tier=secondary` chains,
  and rejects any secondary that would violate the multi-membership
  cap (max 2 chains per qid; non-L1/L2 capped at 1). Smoke-tested on
  5 synthetic cases — caps enforce as designed.
- **Phase 1.5 — merge run:** 0 cap rejections (expected — the lenient
  sweep was scoped to *uncovered* buckets, so secondary qids are by
  definition fresh). Final merged count: **373 + 506 = 879 chains**.
- **Phase 1.6 — validate:**
  - `apply_proposed_chains.py --proposed chains.json --dry-run` →
    validation clean (879 chains).
  - `vault check --strict` → 10,701 loaded, 0 invariant failures.
  - `vault build --local-json` → releaseId=dev, published_count=9438,
    **chainCount=879** (was 373); release_hash changes to `04ee8a23…`.
  - Started `next dev` server; ran
    `node interviews/staffml/tests/chain-and-vault-smoke.mjs` →
    **13/13 pass**. Server stopped post-run.

**Distribution & quality checks (lenient pass):**
| Δ | count | %    | (note) |
|---|------:|-----:|---|
| 0 |    55 |  5.2 | shared-scenario pairs |
| 1 |   736 | 69.1 | preferred shape |
| 2 |   225 | 21.1 | one-rung skip |
| 3 |    49 |  4.6 | last-resort |

- Chains with at least one Δ=0: **55 / 506 (10.9%)** — within the
  roadmap's expected 10-20% band.
- Random spot-check of 5 Δ=0 chains: all show genuine shared-scenario
  threads (DMA optimization on nRF5340; SRAM OOM on Cortex-M4
  residual blocks; CMSIS-NN performance variations; on-device vs
  cloud routing; PB-scale data pipelines). No "two unrelated L3s
  glued together" hits in the sample.

**Coverage gains:**

| track  | primary | secondary | total | chains/topic before → after |
|--------|--------:|----------:|------:|---|
| cloud  |     242 |       116 |   358 | 2.95 → 4.37 |
| edge   |      49 |       148 |   197 | 0.64 → 2.59 |
| mobile |      46 |       113 |   159 | 0.74 → 2.56 |
| tinyml |      36 |        83 |   119 | 0.80 → 2.64 |
| global |       0 |        46 |    46 | 0.00 → 0.96 |
| **all**|     373 |       506 |   879 | **1.19 → 2.81** |

Buckets with ≥1 chain: **285 / 313 (91%)** — was 102 / 313 (33%)
before the lenient pass. The 28 remaining un-chained buckets are
either tiny (≤2 questions) or the lenient sweep judged them genuinely
unrelated.

**Files committed in the Phase 1 commit:**
- `interviews/vault-cli/scripts/merge_chain_passes.py` (new)
- `interviews/vault/chains.json` (373 → 879)
- `interviews/vault/chains.proposed.lenient.json` (durable record of the
  lenient pass)
- `interviews/vault/gaps.proposed.lenient.json` (269 new gaps for Phase 3)
- `interviews/staffml/src/data/vault-manifest.json` (chainCount + releaseHash)
- `interviews/staffml/src/data/corpus-summary.json` (chain memberships
  per question — derived by `vault build`)
- `interviews/vault-cli/docs/CHAIN_ROADMAP.md` (this Progress Log entry,
  + status flips: Phase 1 complete, snapshot updated)

**Notes for next session:**
- Phase 1 done. Phase 2 (tier surfacing in UI) is now unblocked. Start
  at 2.1 — schema migration: every chain entry needs `tier`, validator
  should default missing-tier to `"primary"`, `legacy_export.py` needs
  to emit `chain_tiers` per question (mirrors `chain_positions`).
- Phase 3 (gap-driven authoring) inherits a much bigger backlog now:
  407 gaps total (138 strict + 269 lenient). Prioritize buckets where
  the bucket already has 4+ questions and just needs the bridge.
- Consider running 4.1 (CI gate) before Phase 2 so any tier-related
  regressions during 2.x get caught in CI. Roadmap says "before
  Phase 3 (gates corpus growth)" — could pull it forward.
- Pre-merge backup `chains.json.pre-merge.bak` was deleted; canonical
  pre-Gemini backup remains at `chains.json.bak` (the original 726-chain
  pre-rebuild snapshot).

**Next step:** Phase 2.1 — schema migration (tier required on chain
entries, `chain_tiers` derived in `legacy_export.py`).

---

### 2026-05-01 — Phase 2: tier surfacing schema → TS → UI

**What was done:**

**Phase 2.1 — backend / schema:**
- `legacy_export.py`: added `_build_chain_tier_index` (qid → {chain_id: tier})
  parallel to the existing `_build_chain_index`. `_adapt` emits a new
  `chain_tiers` field on every legacy item that has `chain_ids`,
  defaulting any missing chain-tier to `"primary"`.
- `vault build` re-run: 2953 chained questions, 2953 carry `chain_tiers`
  (100% coverage). releaseHash unchanged from Phase 1 (`04ee8a23…`) since
  the new field doesn't perturb the manifest hash inputs.
- No validator changes — tier is a UI-routing hint, not a structural
  invariant. Missing tier defaults to "primary" everywhere.
- Test fixes: existing `test_chain_positions_plural_preserved` and
  `test_multi_chain_membership` were stale (still asserted on the v1.0
  YAML `chains:` field path; v1.1 made chains.json the sidecar source
  so the tests were silently broken). Rewrote to write a chains.json
  fixture into `tmp_path` and added `chain_tiers` assertions, plus a
  new `test_chain_tiers_emitted_per_membership` covering primary +
  secondary + missing-tier cases.

**Phase 2.2 — TypeScript types:**
- `staffml/src/lib/corpus.ts`: `Question.chain_tiers?` added (optional
  `Record<string, "primary" | "secondary">`). New `ChainTier` exported
  type. `ChainInfo` gains a required `tier` field.
- Internal `_chainTier: Map<chainId, ChainTier>` built alongside
  `_chainIndex` so the runtime can answer "what tier is this chain?"
  in O(1) without re-scanning questions.
- `getChainForQuestion` and `getAllChainsForQuestion` populate `tier`
  on returned ChainInfo objects. `getAllChainsForQuestion` now sorts
  primary chains first.
- New `getPrimaryChainForQuestion(qid)`: returns the first primary
  chain, falling back to the first secondary, falling back to null.
  This is the default-surface helper for UI components.
- `npx tsc --noEmit`: 0 errors after the change.

**Phase 2.3 — UI:**
- `practice/page.tsx`: reads `?chain=<id>` URL param. Uses
  `getChainForQuestion(qid, chainParam)` when set, otherwise
  `getPrimaryChainForQuestion(qid)`. Existing pre-reveal ChainBadge
  + collapsible ChainStrip rendering paths preserved.
- `ChainBadge.tsx`: added optional `tier` prop. When `tier === "secondary"`,
  the badge renders an "alt path" pill inline (always-visible — no
  click required to discover the tier). Default is `"primary"` so
  existing call sites don't need updating.
- `ChainStrip.tsx`: same "alt path" pill in the progress-dot row when
  the rendered chain is secondary, for users who do click in.
- `explore/page.tsx`: when a question is in multiple chains, the
  explorer prefers the first non-secondary chain when picking
  `activeChainId` for the related-questions panel.
- **Deferred from the roadmap's Phase 2.3 scope (tracked for a follow-up):**
  - "Primary only / All" filter dropdown on the explore page
  - Daily-challenge / mock-interview routing changes (those flows
    don't currently key on chain tier; punted to a focused later commit)

**Phase 2.4 — playwright tests:**
- Added `test7_tier_aware_chain_routing` to
  `chain-and-vault-smoke.mjs`. Covers four assertions:
  1. Secondary chain reachable via `?chain=<id>` URL param
  2. "alt path" badge visible on the secondary chain
  3. Primary-chain question still loads (regression check)
  4. "alt path" badge ABSENT on primary chain (negative check)
- Full suite: **17/17 pass** (was 13/13). Roadmap target was 15/15;
  added one more sub-assertion than planned for the negative check.
- Test fixtures pinned to `cloud-0231` (secondary-only) +
  `cloud-chain-auto-secondary-013-04` and `cloud-0001` (primary).

**Validators (re-confirmed end of Phase 2):**
- `vault check --strict`: 10,701 loaded, 0 invariant failures
- `vault build --local-json`: 9438 published, chainCount=879
- `pytest interviews/vault-cli/tests/`: 74/74 pass
- `npx tsc --noEmit`: 0 errors
- `node interviews/staffml/tests/chain-and-vault-smoke.mjs`: 17/17

**Notes for next session:**
- Phase 2 done. Phase 3 (gap-driven authoring) is unblocked. Backlog
  for authoring is now **407 gaps** (138 strict + 269 lenient).
- The deferred explore-page filter is not load-bearing — secondary
  chains are reachable via `?chain=` and don't pollute the default
  surfaces. Worth picking up before Phase 4.x scaffolding.
- 0 questions currently belong to BOTH a primary and secondary chain
  (because the lenient sweep was scoped to uncovered buckets). When
  Phase 3 authors new questions into already-chained buckets, the
  cap rules in `merge_chain_passes.py` will start mattering for real.
- Consider scheduling a one-time agent to merge `yaml-audit` → `dev`
  again now that Phase 2 is shipped (the local `dev` worktree has
  Phase 1 only — Phase 2 + the CHAIN_ROADMAP updates are not in dev).

**Next step:** Phase 3.a — `generate_question_for_gap.py` (Gemini
authoring tool that takes a gap entry and drafts a candidate question
fitting the bridge requirement).

---

### 2026-05-01 — Phase 4.8 docs + Phase 4.2 audit

**Phase 4.8 — docs (shipped):**
- `interviews/vault/ARCHITECTURE.md` gains a new §3.6 capturing the
  three v1.1 deltas: hierarchy, sidecar chain registry, tier model.
  Additive to v1, not replacements; cross-refs CHAIN_ROADMAP.md.
- `interviews/vault-cli/README.md`: status line bumped from "Phase 0
  scaffolding" to v1.1; new "Chain build pipeline" section with
  invocation examples for diagnose / build / apply / merge; layout
  block reflects scripts/ + actual src/ contents.
- Commit: `f086b6f42 docs(vault): document v1.1 sidecar + hierarchy + tier model`

**Phase 4.2 — multi-chain UI audit (no-op for now):**
- Audited `chains.json`: **0 qids in >1 chain.** Reason: the strict
  pass already enforces the multi-membership cap within-tier, and the
  lenient pass was scoped to *uncovered* buckets, so no qid in any
  primary chain was reachable for a secondary chain to bind to. The
  merge step's cap rules consequently never fired (0 rejections).
- Action: **defer the focused playwright test**. The case becomes
  exercisable when Phase 3 authoring fills bucket gaps and a re-run
  of `build_chains_with_gemini.py --all` (which will see those new
  questions in already-chained buckets) produces a multi-chain qid.
- No commit needed — zero state change.

**Notes for next session:**
- Phase 1, Phase 2, and Phase 4.8 are all shipped on
  `origin/yaml-audit`. Local `dev` worktree has Phase 1 only (Phase 2
  + docs not re-merged) — the user has been doing parallel workflow
  refactoring on dev, so I held off on a second yaml-audit → dev
  merge to avoid colliding with their `.github/workflows/` edits.
  When the user is ready, the merge can be done from a clean dev
  worktree state with `git merge --no-ff yaml-audit`.
- Phase 4.1 (CI gate), 4.4 (deploy lockstep), 4.6 (periodic rebuild
  automation), and 4.9 (gitignore CI guard) all touch
  `.github/workflows/` — the user has uncommitted changes there, so
  these were intentionally skipped this session.

**Next step:** Phase 3.a — `generate_question_for_gap.py`. This is the
first of the gap-driven authoring tools. The roadmap budgets it at "1
day" because it's the substantive new capability of Phase 3 (Gemini
authoring vs. just chain construction). Best done with the user
available to review the first few generated drafts.

---

### 2026-05-01 — Phase 3.a + 3.b: authoring + validation tooling

**What was done:**

**Phase 3.a — `generate_question_for_gap.py`:**
- Reads a gap entry (`{track, topic, missing_level, between, rationale}`)
  from gaps.proposed.json (or .lenient.json), loads the between-questions
  in full + up to 3 same-bucket exemplars at the target level, prompts
  Gemini-3.1-pro-preview with the schema summary + bridge context, and
  writes a candidate question to
  `interviews/vault/questions/<track>/<area>/<id>.yaml.draft`.
- ID allocator scans the existing corpus + already-written drafts so a
  batch run gets distinct fresh IDs without touching `id-registry.yaml`
  (registry append happens at promotion time, not generation).
- Authoring metadata stamped under a private `_authoring` block:
  origin model, tool name, timestamp, and the source gap entry. The
  Pydantic Question model has `extra="allow"`, so this passes schema.
- Modes: `--gap-index <N>` (single gap), `--gaps-from <path> --limit N`
  (batch), `--dry-run` (build prompts without calling Gemini).
- Smoke checks:
  - `--dry-run --gap-index 0` resolves the first gap, finds 3 exemplars,
    builds the prompt, allocates `cloud-4579`. ✓
  - Synthetic Gemini response → `assemble_draft` → `Question.model_validate`
    passes; YAML preview looks right (12-field body, sensible details). ✓

**Phase 3.b — `validate_drafts.py`:**
- Five-gate scorecard per draft:
  1. **schema** — Pydantic Question (mandatory; downstream gates skip
     on schema fail to avoid spurious LLM calls)
  2. **originality** — embeds `title + scenario + question` with
     `BAAI/bge-small-en-v1.5` (matches the corpus embeddings.npz model
     so cosines are directly comparable), compares against in-bucket
     neighbors, flags any `cosine ≥ 0.92`
  3. **level_fit** — Gemini-judge against ≤5 published exemplars at the
     target level in the same (track, topic)
  4. **coherence** — Gemini-judge: scenario / question /
     realistic_solution mutually consistent
  5. **bridge** — Gemini-judge: candidate genuinely chains between the
     two `between` questions named in `_authoring.gap`
- Skips: `--no-originality` (skip embed model load),
  `--no-llm-judge` (skip Gemini gates). Schema gate is unconditional.
- Output: `interviews/vault/draft-validation-scorecard.json` with per-row
  detail + final verdict (`pass | fail | error`).
- Smoke check: synthetic draft in /tmp passed schema + originality
  (top-neighbor cosine 0.73 vs 0.92 threshold). End-to-end runner
  produced a well-formed scorecard. ✓

**What was deliberately not done tonight:**
- **Phase 3.c (pilot run on 30 highest-value gaps):** This generates
  new YAML question content that needs human review *before* promotion.
  Running 30 unsupervised generations and 30×4 LLM-judge calls without
  the user available to spot-check the first few outputs is the wrong
  shape of work for an overnight slot. The tooling is ready when the
  user is.
- **Phase 3.d–3.f:** Promotion + re-chain are downstream of 3.c
  acceptance.

**Recommended pilot when the user is back:**
1. Pick 30 gaps from `gaps.proposed.lenient.json` where the bucket has
   ≥4 questions already (just missing the bridge):
   ```bash
   python3 interviews/vault-cli/scripts/generate_question_for_gap.py \
     --gaps-from interviews/vault/gaps.proposed.lenient.json \
     --limit 30
   ```
2. Validate:
   ```bash
   python3 interviews/vault-cli/scripts/validate_drafts.py
   ```
3. Manually review the passing drafts (~20-25 expected).
4. Promote: rename `.yaml.draft` → `.yaml`, append to id-registry.
5. Re-run `build_chains_with_gemini.py --all` so the new questions get
   absorbed into chains.

**Files committed:**
- `interviews/vault-cli/scripts/generate_question_for_gap.py` (new)
- `interviews/vault-cli/scripts/validate_drafts.py` (new)
- `interviews/vault-cli/docs/CHAIN_ROADMAP.md` (this Progress Log entry +
  status flips)

**Notes for next session:**
- Both scripts assume `gemini` CLI on PATH (gemini-3.1-pro-preview) and,
  for originality, the corpus's `embeddings.npz` (gitignored, regenerable
  by the existing embedding scripts). `validate_drafts --no-llm-judge`
  is a fast first cut that only exercises schema + originality if you
  want to triage drafts before paying for the LLM-judge calls.
- Heads up: each draft in 3.b consumes ~3 Gemini calls (level_fit +
  coherence + bridge). 30 drafts → ~90 calls. Daily cap is 250.
- `id-registry.yaml` is append-only and CI-enforced. Promotion (3.d)
  needs to add new IDs to it; that's not yet wired into a script —
  manual append for the pilot, then we can extract a `vault promote`
  helper from the pattern.

**Next step:** Phase 3.c — pilot run on 30 high-value gaps (best done
with the user available to spot-check the first few outputs).

---

### 2026-05-01 — Phase 3.c + 3.d: pilot run + promotion (5 gaps)

**Pilot scope (sized down from the roadmap's 30):** 5 high-value gaps,
selected from `gaps.proposed.lenient.json` favoring (track, topic)
buckets with ≥4 published questions and biased toward low-density
tracks. All 5 picks landed in edge/mobile (the densities the lenient
sweep most needed help on).

**Phase 3.c — generate (`generate_question_for_gap.py`):**
| target | gap | result |
|---|---|---|
| edge-2535 | edge/latency-decomposition L?→L3 between=[edge-1883, edge-1701] | written |
| edge-2536 | edge/pruning-sparsity L?→L4 between=[edge-1960, edge-1957] | written |
| edge-2537 | edge/tco-cost-modeling L?→L3 between=[edge-0731, edge-1154] | written |
| mobile-2146 | mobile/duty-cycling L?→L3 between=[mobile-0367, mobile-2034] | written |
| mobile-2147 | mobile/model-format-conversion L?→L2 between=[mobile-0984, mobile-1022] | written |

5/5 generated cleanly. Each draft passed Pydantic schema validation
inline (the `assemble_draft` → `Question.model_validate` gate); none
were rejected at the file-write step.

Spot-checking `edge-2535`: realistic ML-systems scenario (Coral USB
TPU + MobileNetV2-SSD + INT8), concrete numbers, calculation-driven
question consistent with L3/apply, solution gets at the actual
insight (host-side bottleneck). Other 4 are similarly competent.

**Phase 3.b run — `validate_drafts.py`:**

| draft | originality | level_fit | coherence | bridge | verdict |
|---|---|---|---|---|---|
| edge-2535 | **fail** (cos=0.933 vs edge-1883) | pass | pass | pass | **fail** |
| edge-2536 | pass | pass | pass | pass | **pass** |
| edge-2537 | pass | pass | pass | pass | **pass** |
| mobile-2146 | pass | pass | pass | pass | **pass** |
| mobile-2147 | pass | pass | pass | pass | **pass** |

**4/5 pass = 80% pass rate** (above the roadmap's 60-75% estimate).
The one fail was correctly caught — `edge-2535`'s draft scenario
turned out too similar to one of its between-questions
(`edge-1883`), cosine 0.933 over the 0.92 threshold. This is the
gate working as designed: Gemini occasionally drafts a "bridge" that's
just a paraphrase of one of its anchors instead of a true L3
intermediate. The gate filtered it.

**Phase 3.d — promotion (4 passing drafts):**
- `.yaml.draft` → `.yaml` rename for the 4 passes.
- `_authoring` private metadata stripped at promotion; replaced with:
  - `provenance: llm-draft`
  - `status: draft` (not `published` — gating on human review)
  - `authors: ["gemini-3.1-pro-preview"]`
  - `human_reviewed: { status: not-reviewed, ... }` so the
    not-yet-reviewed state is honest and machine-checkable.
  - `tags`: original tags preserved + a new `gap-bridge:<from>-<to>`
    tag so these can be queried later.
- IDs appended to `id-registry.yaml`: `edge-2536`, `edge-2537`,
  `mobile-2146`, `mobile-2147` — created_by `generate_question_for_gap.py`.
- `edge-2535.yaml.draft` was **kept in place** (still .yaml.draft).
  Decision for the human reviewer when they triage: rewrite + retry,
  or delete.

**Validation post-promotion:**
- `vault check --strict` → 10,705 loaded (was 10,701; +4 ✓), 0 invariant
  failures.
- `vault build --local-json` → released set unchanged: 9438 published,
  chainCount=879, releaseHash=04ee8a23… (drafts have status=draft, so
  the publishing filter excludes them — by design).

**Phase 3.e — chain rebuild (deferred):**
Skipped tonight. The new questions are `status: draft` and the
chain-builder filters on published, so a rebuild wouldn't pick them
up. The right sequence is: human reviews the 4 drafts → flips status
to `published` (and `human_reviewed.status` to `verified`) → then
re-runs `build_chains_with_gemini.py --all`. At that point chainCount
is expected to grow modestly (the 4 new questions were authored TO
fit chains, so they should land in their bridge slots).

**Files changed in the Phase 3 pilot commit:**
- `interviews/vault/questions/edge/cross-cutting/edge-2537.yaml` (new)
- `interviews/vault/questions/edge/optimization/edge-2536.yaml` (new)
- `interviews/vault/questions/mobile/deployment/mobile-2147.yaml` (new)
- `interviews/vault/questions/mobile/power/mobile-2146.yaml` (new)
- `interviews/vault/questions/edge/latency/edge-2535.yaml.draft` (new — failed validation, awaiting reviewer disposition)
- `interviews/vault/draft-validation-scorecard.json` (new — per-row record)
- `interviews/vault/id-registry.yaml` (4 appended entries)
- `interviews/vault-cli/docs/CHAIN_ROADMAP.md` (this entry)

**Notes for next session — review checklist:**
1. Read each of the 4 promoted drafts. Spot-checks suggest they're
   competent but cognitive-load calibration is the place where Gemini
   drift is most likely. Each scorecard row has the `level_fit` rationale
   from the LLM judge — those are first-cut signals, not authoritative.
2. For the failed `edge-2535`: read it next to its high-cosine
   neighbour (`edge-1883`). If it's too duplicative as the originality
   gate suggests, delete; if it's actually distinct enough, edit and
   re-validate (you can re-run `validate_drafts.py` after editing).
3. Once you're happy with N drafts, flip their `status: draft → published`
   and `human_reviewed.status → verified`, set `human_reviewed.by`, then:
   ```bash
   vault check --strict
   vault build --local-json    # released question count goes up by N
   python3 interviews/vault-cli/scripts/build_chains_with_gemini.py --all \
     --output interviews/vault/chains.proposed.json
   python3 interviews/vault-cli/scripts/apply_proposed_chains.py
   ```
4. If the pilot's 80% rate holds at scale, a 30-gap batch would land
   ~24 promotable drafts and absorb ~12-15 of them into chains
   (chain rebuild typically picks up ~50% of new questions per the
   roadmap).

**Cost note:** This pilot used 5 generation calls + 5 × 3 judge calls = 20 Gemini
calls. A 30-gap batch would be ~120 calls (still under the 250/day cap but
worth budgeting around).

**Next step:** Phase 3.e — chain rebuild. Gated on human review of the
4 drafts now in the tree.

---

<!-- Append new entries above this comment, in reverse chronological is fine,
     but keep entries dated and self-contained for resume context. -->
