# Vault Chain Coverage Roadmap

**Status:** active workstream
**Branch:** `yaml-audit` (off `dev`)
**Worktree:** `/Users/VJ/GitHub/MLSysBook-yaml-audit`
**Last updated:** 2026-04-30 (Phase 1 complete — chains 373 → 879)

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
   vault build --legacy-json      # expect: clean build, releaseId=dev, 9438 published
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
2. `vault build --legacy-json` → clean, chainCount jumps from 373 to ~700
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

**Status:** `not started`
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

**Status:** `not started`
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
vault build --legacy-json
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
**When:** end of Phase 1
**Action:** Audit current chains.json for qids with 2-chain memberships.
If non-zero (likely after lenient pass), add focused playwright test.

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
**When:** end of Phase 2
**Files:**
- `interviews/vault/ARCHITECTURE.md` (sidecar architecture, hierarchy, tier model)
- `interviews/vault-cli/README.md` (command list)

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
- `vault build --legacy-json` → releaseId=dev, 9438 published, chainCount=373 ✓

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
  - `vault build --legacy-json` → releaseId=dev, published_count=9438,
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

<!-- Append new entries above this comment, in reverse chronological is fine,
     but keep entries dated and self-contained for resume context. -->
