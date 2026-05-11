# Codex deep-review follow-up — next-session plan

**Author**: Claude (Opus 4.7), session 2026-05-11
**Worktree**: `/Users/VJ/GitHub/MLSysBook-codex-deep-review`
**Branch**: `chore/codex-deep-review`
**HEAD**: `c3f378ffc`
**Status**: Codex deep-review prose pass + discussion walkthrough COMPLETE.

**Resume triggers**: open this file or say one of the queue-specific phrases in §3.

---

## 0. Context (read this first)

The codex deep-review prose pass is fully landed across five commits. The work resolved every YAML-flagged issue across vol1 (26 chapters) and vol2 (24 chapters + 4 part openers + frontmatter), then walked the `review/PENDING_DISCUSSIONS.md` queue chapter by chapter with the user to either fix items inline or stage them for follow-up.

What remains is **not more codex-deep-review work**. It is six clearly-scoped focused-follow-up passes, each marked in the source with TODO comments that the next session can grep for and walk in order.

### Commit history

```
c3f378ffc chore(vol2): resolve pending-discussion items for remaining 13 chapters
6bd859590 chore(vol2): resolve pending-discussion items for frontmatter + first 4 chapters
7b811b5a1 chore(vol2): MIT Press prose pass from codex deep-review
f3f4af10e fix(vol1): resolve precommit findings from vol1 prose pass
187c5e9da chore(vol1): MIT Press prose pass from codex deep-review
```

All pre-commit hooks green on `c3f378ffc`. PENDING_DISCUSSIONS.md reflects the final state.

---

## 1. The six follow-up queues, in priority order

Each queue corresponds to a TODO marker in the source. Find them with the grep commands below, then walk in canonical chapter order.

### Queue A — Numerical-audit pass

**Why first**: Math errors mislead readers and undermine the chapter's quantitative claims. Fix before render or external review.

**Find the sites**:
```bash
cd /Users/VJ/GitHub/MLSysBook-codex-deep-review
grep -rln 'TODO(numerical-audit' book/quarto/contents/vol2/ --include='*.qmd'
```

**Known sites** (consolidated TODO blocks at top of chapter unless noted):

| Chapter | Issues |
|---|---|
| `data_storage` | 7 inconsistencies: HBM 300,000-fold bandwidth gap vs ~33.5-fold; storage pyramid caption vs `@tbl-storage-hierarchy-merged`; archive 5x vs 20x; 2-year archive cost $960 vs $9,600; idle cost $9,600 vs $13,000/day; ZeRO-3 70B shard math; 175B 256-node 4 GB vs 6.8 GB |
| `network_fabrics` | Fat-tree scaling claims (lines 859, 861, 907, 978, 980, 1078): host and switch counts can't all be true under one topology. TODO comment at top of `## Fat-tree` callout |
| `distributed_training` | Slow-AllReduce debugging arithmetic at line ~648: 50/100/120/240 ms numbers don't reconcile. TODO comment in chapter source |
| `sustainable_ai` | Operational-carbon PUE double-count (line ~1256); two near-duplicate energy-wall figures (lines ~373 vs ~638); fallacy 367/34.5 = 10.6 not 21x (line ~2848); embodied-carbon 108 kg vs 10.5 t inconsistency (lines ~2852 vs ~1501); GPT-3 household-energy drift |
| `ops_scale` | Inference-cost equation `24 x 365` factor double-counts time (line ~3265); 3-sigma alerts weekly vs daily mismatch (line ~2193) |
| `responsible_ai` | SHAP overhead drift: 50--200% vs 50--1000x vs 200ms--5s+ (line ~1044) |
| `conclusion` | `1,000 TFLOPS` hardcoded at line ~364 vs `H100_FLOPS_FP16_TENSOR = 989` |

**Approach**: Walk one chapter at a time, recompute each claim against the chapter's running examples, update both the prose claim and any inline-Python sources to track. Verify the numbers cross-reference cleanly.

**Resume trigger**: *"start the numerical-audit pass"*

---

### Queue B — Bib-audit pass

**Why next**: Citation gaps undermine credibility on quantitative claims. The standard policy (per `bib-check.md`) is that adding bib entries is a bib-pass concern, separate from prose work.

**Find the sites**:
```bash
grep -rln 'TODO(bib-pass' book/quarto/contents/vol1/ book/quarto/contents/vol2/ --include='*.qmd'
```

**Known sites**:

| Chapter | Items |
|---|---|
| `vol1/model_compression` | Figure attributions at lines 2984, 4508, 5721, 7191; fallacies-section empirical claims at line 7893 |
| `vol1/ml_ops` | Definition-callout claims at line 147, retail-failure example at line 153, others |
| `vol1/benchmarking` | Pricing-data citations at lines 197 and 4228 |
| `vol1/model_serving` | Pricing-data citations at lines 197 and 4228 |
| `vol2/compute_infrastructure` | Environmental-impact paragraph at line ~2165 (GPT-3 energy, carbon-intensity, Microsoft nuclear, Google geothermal) |
| `vol2/fault_tolerance` | Case-study claims at line ~3293 (OPT-175B failure counts, Google TPU pod, Netflix MTTR, DeepSpeed overhead). TODO comment at `## Case Studies` |
| `vol2/inference` | Case-study proprietary claims at line ~5257 (Meta, OpenAI, Google, TikTok). TODO at chapter top |
| `vol2/robust_ai` | Lines 728, 744, 746: nighttime degradation, weather mAP, fraud drift. TODO at chapter top |
| `vol2/responsible_ai` | FCC rural-broadband statistic at line ~865. TODO at chapter top |
| `vol2/about.qmd` | AlexNet "five to six days on two GPUs", GPT-4 "25,000 GPUs ... three months", 2 percent failure rate at lines 11, 13 |

**Approach**: For each site, either (a) write a bib entry per `bib-check.md` §8 staging workflow and add the cite, or (b) recast the claim as explicitly illustrative ("estimated", "reported", or scoped to a stated assumption).

**Resume trigger**: *"start the bib-audit pass"*

---

### Queue C — Figure-asset pass

**Why this matters**: Figures rendered with wrong captions or duplicate assets confuse readers and waste figure budget.

**Known sites** (no single grep target; consolidated here):

| Chapter | Issue |
|---|---|
| `vol2/edge_intelligence` | `@fig-...` A/B/C region labels inconsistent between figure caption and prose at line ~308 (Local Only vs Centralized Cloud flip) |
| `vol2/robust_ai` | `@fig-poisoning` caption-vs-prose contradiction at line ~1866: caption describes online-learning incremental poisoning; prose + alt text describe Nightshade concept poisoning. Caption needs to match the rendered figure |
| `vol2/responsible_ai` | `images/svg/bias-loop.svg` reused for both `@fig-bias-amplification` (line ~907) and `@fig-bias-loop` (line ~2117). Generate distinct asset or consolidate |
| `vol2/network_fabrics` | `@fig-hierarchical-staircase` rendered asset at line ~1350 doesn't match prose intent. Replace asset OR revise prose |
| `vol2/inference` | Recurring case-study labels at line ~5259 rendered as plain paragraphs: "Scale and requirements", "Architecture overview", "Key design decisions", "Lessons learned". Convert to subheadings |
| `vol2/inference` | Quantization-section labels at line ~4991 rendered as plain text: "Key insight", "Algorithm", "Core technique", "Critical distinction" |
| `vol2/edge_intelligence` | Adaptation-strategy table caption vs row contents mismatch at line ~1605: caption names "full finetuning" which is not a row |
| `vol2/edge_intelligence` | Constraint-solution table missing Federated Coordination pillar at line ~972 (table covers two of three) |
| `vol2/ops_scale` | Organizational-pattern figure caption left/center/right not visible at line ~3768 |
| `vol2/robust_ai` | Attack-taxonomy table single-cell concatenation at line ~1283 (`FGSM PGD JSMA C&W` without separators) |

**Render-verify queue** (vol1 structural changes from earlier passes):
- `vol1/hw_acceleration` `@fig-memory-wall` subsection structure
- `vol1/data_selection` `@fig-ppd-curve` label
- `vol1/responsible_engr` fairness-frontier axis vs caption

**Approach**: Render the chapter, inspect each figure, fix asset or caption.

**Resume trigger**: *"start the figure-asset pass"* or *"render and verify the deferred figure items"*

---

### Queue D — Style-cleanup pass

**Find the sites**:
```bash
grep -rln 'TODO(style-cleanup\|TODO(focused-followup' book/quarto/contents/vol2/ --include='*.qmd'
```

**Known sites** (consolidated TODO blocks at the top of each chapter):

| Chapter | Items |
|---|---|
| `vol2/inference` | Quantization-section plaintext labels (~4991); case-study labels (~5259); cross-cutting bold starters (~5545); dangling fallacy (~5594); range form (~5572); Fallacies close hype (~5641) |
| `vol2/edge_intelligence` | Client-scheduling repetition (~2383); performance-metrics comma splices (~2976); notebook second-person at lines 160/921/1024/1819; worked-example plaintext label (~2279); `$E = 2$-$5$` math hyphen (~2309) |
| `vol2/ops_scale` | Notebook second-person at lines 326/600/725/1396/1493/2401; promotional register at line ~161; bold callout starter at ~3456; organizational-pattern bold list at ~3841; redundant worked example at ~3933; Netflix "dramatic" intensifier at ~3984 |
| `vol2/security_privacy` | Health-monitoring notebook second-person at ~2446; Tesla/Zoox alleged-vs-established at ~1386; knowledge-check second-person at ~1875; hardware-security table split rows at ~2382; salary-example second-person at ~2813 |
| `vol2/robust_ai` | Notebook callouts at 627/1936/1945; informal figure-caption sources at 1225/2056/2209 (`ivezic`, `li`, `dertat`, uppercase HTTPS); spaced `---` at 83 and 97 |
| `vol2/sustainable_ai` | "Noise Event", "Voltage Spike" invented-label capitalization at ~653; OpenAI supercomputer uncited/undated at ~1908 |
| `vol2/responsible_ai` | Non-protected callout direct address at 125/281/1098/1522; navigation callout subject-verb at 117; missing apostrophes at 190/929/1717/2131; compound-form drift at ~180; hyphen ranges in decision-framework table at ~2427 |
| `vol2/conclusion` | Compound capability law callout (~177); speculative formula (~184); notebook rhetorical Qs at 226/277; author signature placement (~388); time-sensitive phrases (~173); manifesto closing (~386) |

**Approach**: Per chapter, walk the TODO comment block in order. Most are 2–5-minute fixes; some need a small judgment call (e.g., should the case-study labels become H4 subheadings or bold lead-ins).

**Resume trigger**: *"start the style-cleanup pass"*

---

### Queue E — Lockstep slug renames

Heading-case hook treats section-ID slugs as authoritative. Four proper-noun headings need their visible text and slug renamed in lockstep so the slug doesn't lose the proper noun:

| Chapter | Heading | Current slug |
|---|---|---|
| `vol2/security_privacy` line 837 | "Jeep cherokee hack" | `#sec-security-privacy-insufficient-isolation-jeep-cherokee-hack-6a7c` |
| `vol2/security_privacy` line 3063 | "Rnyi differential privacy and composition" | `#sec-security-privacy-rnyi-differential-privacy-composition-f69a` |
| `vol2/network_fabrics` line 1360 | "Case study: Meta grand teton" | `#sec-network-fabrics-grand-teton` |
| `vol2/fleet_orchestration` line 589 | "Advanced slurm configuration for ML" | `#sec-fleet-orchestration-slurm-advanced` |

**Approach**: For each, run a corpus-wide grep for the slug; rename heading + slug + every cross-reference site in one commit per slug. Verify the heading-case hook passes after the rename.

```bash
# Example for Cherokee:
grep -rln 'jeep-cherokee-hack-6a7c' book/ review/ --include='*.qmd' --include='*.json'
# Then sed the slug + visible text in lockstep
```

**Resume trigger**: *"start the slug-rename pass"*

---

### Queue F — `model_serving` T_window / T_optimal

**One careful follow-up flagged by user**: the batching-window formula at `vol1/model_serving.qmd` line 3419 was relabeled from `T_optimal` to `T_window` because no derivation was provided. User wants this revisited carefully.

**Decision**: either (a) keep `T_window` heuristic framing as-is, or (b) restore `T_optimal` and add a derivation/citation (e.g., M/M/1 cost-balance result or a specific batching paper).

**Resume trigger**: *"revisit the model_serving T_window decision"*

---

## 2. Suggested order

A through F is the recommended order, but each queue is self-contained — they can be done in parallel by separate sessions or interleaved.

- **A (numerical-audit)** should come first because the figure-asset and style-cleanup passes may regenerate captions that reference the corrected numbers.
- **B (bib-audit)** can run in parallel with A; it touches `references.bib` and citation sites, no math overlap.
- **C (figure-asset)** depends on a render; do after A so the corrected captions don't get baked into wrong figures.
- **D (style-cleanup)** depends on nothing; can run any time. Likely the largest by edit count but smallest per-edit.
- **E (slug-rename)** is independent; one focused commit per slug.
- **F (T_window)** is independent; one focused commit.

---

## 3. How a fresh session resumes

```bash
cd /Users/VJ/GitHub/MLSysBook-codex-deep-review
git log --oneline -5   # confirms HEAD at c3f378ffc
cat review/NEXT_SESSION_PLAN.md   # this file
cat review/PENDING_DISCUSSIONS.md   # detailed item-by-item record
```

Or trigger the specific queue:
- *"start the numerical-audit pass"*
- *"start the bib-audit pass"*
- *"start the figure-asset pass"*
- *"start the style-cleanup pass"*
- *"start the slug-rename pass"*
- *"revisit the model_serving T_window decision"*

---

## 4. What is COMPLETE and does NOT need revisiting

- **Vol1 prose pass** (commit `187c5e9da` + entry-task fixes `f3f4af10e`)
- **Vol2 prose pass** (commit `7b811b5a1`) — 24 chapters + 4 part openers + frontmatter
- **Vol2 discussion walkthrough**: commits `6bd859590` and `c3f378ffc` resolved every item in `review/PENDING_DISCUSSIONS.md` either inline or via TODO comments in the source

The remaining work is the six follow-up queues above. Each is **scoped and ready** — the next session does not need to reread the YAML feedback files or rewalk the discussion list.

---

## 5. Things to NOT do

- Do not commit the `book/quarto/contents/vol1/benchmarking/.gitignore` or `book/quarto/contents/vol2/performance_engineering/.gitignore` untracked files — these are render artifacts.
- Do not convert `\ref{pri-...}` cross-references to Quarto `@pri-...` form. The `\ref{}` form is canonical for principle references in both vol1 and vol2 (correction made during the walkthrough).
- Do not change the canonical lighthouse-archetype callout placement in `vol2/introduction.qmd` — the user explicitly retained the early callout (vol2 is independent of vol1, readers need the introduction at first mention).
- Do not "fix" centered-numeric table columns to right-aligned in tables Zeljko's PRs have touched (per `book-prose.md` Table Formatting rules).

---

## Notes from the user during the walkthrough

- Vol1 and vol2 are completely independent textbooks — first-mention conventions apply per volume.
- `\ref{Principle}` is the correct cross-reference form for canonical principle callouts; do not convert to Quarto `@pri-` form.
- Retire commented-out source rather than preserving it in the prose; git history is the source of truth.
- Keep web-register HTML-only sections (Support Mission, Podcast, Want to Help Out?) as-is on the volume splash pages; the PDF Author's Note is the textbook-register version.
- Acknowledgements use a deliberately warmer register; "someone out there cares" stays.
