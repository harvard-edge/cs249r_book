# Render + prose audit — findings ledger

**Date:** 2026-08-16
**Branch:** `audit/render-prose-signoff` (worktree `MLSysBook-render-audit`)
**Method:** both volumes built to HTML, then one reviewing agent per rendered
chapter reading section by section as the reader sees it, cross-checking every
printed calculation against the chapter's own text.

**Coverage:** 25 rendered chapters + 6 appendix/conclusion pages, ~1,300 sections.

Everything below was verified against what the chapter itself states elsewhere.
Items marked **FIXED** are corrected on this branch. Items marked **OPEN** need
an authorial decision, because two plausible values exist and choosing between
them changes a published worked example.

Layout is out of scope for this branch. Two rendering issues belong to the
copyeditor's domain and are recorded at the end rather than patched.

---

## 1. Build-blocking defects — FIXED

Neither volume built when this started.

| Chapter | Defect |
|---|---|
| 12 chapters, both vols | branded capacity formatted with the decimal unit: `80` became `85.899`, tripping the precision guard |
| 5 worked examples | nameplate budget mixed with decimal operands, so a printed `80 − 35 = 45` stopped holding |
| `vol2/responsible_ai` | `alpha`, `alpha_high` deleted as dead code by `23ad1a4786` while live at four call sites |
| `vol2/responsible_ai` | prose referenced `FairnessMetricsTable` 20 lines above the cell defining it; Quarto executes top to bottom |
| `vol2/inference` | an H100 guard asserted a nameplate in the wrong unit space |

## 2. Reader-visible rendering defects — FIXED

| Defect | Count |
|---|---|
| inline refs missing a backtick, printing the raw expression | 4 |
| doubled units where a self-labeling formatter already carried the noun | 14 |
| bare scaled counts missing their noun (`73.2T`, `300T`) | 2 |
| a cross-reference rendering as its own section title, duplicating the phrase | 2 |

## 3. Numbers contradicting the chapter's own text — FIXED

| Chapter | Was | Now | Basis |
|---|---|---|---|
| `training` | GPT-2 at `1.4e20` FLOPs | registry value | its own table and the registry say `1.5e21` |
| `training` | `T_comm` 1.96 s then 2.24 s | derived once, with a guard | printed twice, two lines apart |
| `training` | `T_prep` 180 ms | derived from the trace | the stage trace sums to ~66 ms |
| `training` | 8-V100 cluster | derived | the cell sets 32 |
| `training` | prefetch 105 s → 55 s | minutes | both figure captions say minutes |
| `frameworks` | 5 µs per-op tax | registry (15 µs) | the chapter teaches 15 µs |
| `introduction` | "2.5M GPU-days (25,000 × 90)" | states the shape | that product is 2.25M |
| `appendix_algorithm` | "(80 GB in decimal units)" | 85.9 GB | its own subtraction needs 85.9 |
| `hw_acceleration` | "SRAM access" | "L1 SRAM access" | 128× ratio uses an 8 KB read |
| `nn_computation` | Adam "doubles" gradient memory | triples | Table 8 shows 2× the gradient block |
| `ml_systems` | MCU gap 1,000× | 10,000× | stated twice elsewhere |
| `nn_architectures` | "six major families" | five | the table has five rows |
| `data_engineering` | "sub-kilobyte SRAM" | tens of kilobytes | next box states a 64 KB budget |
| `fleet_orchestration` | 900 GPU-hours invested | 90 | its own setup implies 100 total |
| `model_compression` | GTX 580 with HBM | GDDR5 | HBM shipped in 2015 |

## 4. Quiz/Self-Check values contradicting their chapters — FIXED

The injected Self-Check sections were systematically out of sync. Most carried
the binary reading of a decimal quantity; several asserted specifics the chapter
never states.

`nn_computation` 427→438 KB, Adam 13/52/26/104→14/56/28/112 GB · `training`
130.4→140 GB, 52.2→56 GB, 22.8→24.5 GB · `nn_architectures` 32→33.6 MB,
18.6→20 GB, 238→256 GB · `ml_systems` 23.7→24.9 MB, wake-word 100 µs→1–10 ms ·
`ml_workflow` 8.7→8.3 h, 5,120→5,000× · `data_engineering` 685→748.8 GB ·
`conclusion` 13→14 GB · `benchmarking` ImageNet 7.3→3.57 percent, dropped a
"5–10 runs" rule the chapter never states · `ml_ops` CACHE→CACE ·
`model_compression` 13→14 GB, 3.3→3.5 GB, 3.3 MB/7×→3.5 MB/6.7× ·
`fleet_orchestration` 6,000 GPUs→3,000 GPU-equivalents, 60–70→60–90 percent

Also removed three fabricated specifics: an "eight-percentage-point" drop, a
"3 to 5 days" failure window, and a 100 µs wake-word latency — none appear in
the chapters the answers cite.

---

## 5. OPEN — needs an authorial decision

Each is a genuine contradiction. Fixing it means choosing which value is
canonical, and that choice changes a published worked example.

### High

| # | Chapter | Conflict |
|---|---|---|
| O1 | `performance_engineering` | overlap example: printed sums do not equal their totals (`72.6+145.2+27.2 = 245`, stated 311.9); backward pass given as both 166.3 and 145.2 ms in adjacent sentences |
| O2 | `performance_engineering` | KV headroom per GPU is 62.5 GB in §10 and 68.4 GB in Napkin 1.2, which §10 forward-references as the same quantity |
| O3 | `performance_engineering` | `80 GB − 4.4 GB = 81.5 GB` and `80 GB − 17.5 GB = 68.4 GB` — both printed equations use a decimal minuend with a binary result |
| O4 | `compute_infrastructure` | Example 1.4: "optimizer state alone requires 175 GB per GPU, causing per-GPU memory (113 GB) to exceed 80 GB" — the 113 GB total excludes the 175 GB it is attributed to |
| O5 | `compute_infrastructure` | Definition 1.3 prints the same **Significance** bullet twice (em-dash vs comma) and has no **Distinction** bullet |
| O6 | `compute_infrastructure` | quiz says a 10,000-GPU fleet fails hourly; the chapter says every five hours in three places |
| O7 | `compute_infrastructure` | three quiz items assert a ~30 kW air-cooling ceiling that the body explicitly denies |
| O8 | `model_serving` | Napkin 1.3 uses 5 ms as the TensorRT FP16 service time; Tables 2, 19, 20 give 1.4 ms. The 47→62 GPU result rests on it |
| O9 | `model_serving` | "the 72 GB reserve" is never stated; the sentence before establishes 81.9 GB |
| O10 | `model_compression` | "28 percent end-to-end improvement" appears nowhere; the chapter says 1.5× of an 8× target = 18.8 percent |
| O11 | `data_selection` | "6.2 percent overhead" needs an 8-hour baseline that never appears; against the stated 2-hour run it is 25 percent |
| O12 | `distributed_training` | quiz says 500–700× on 1,000 GPUs is routine; the body says 500× would be exceptional and 100–200× typical |
| O13 | `fault_tolerance` | quiz cites a 15–25 / 65–85 percent incident split the chapter never states (and which does not sum) |
| O14 | `edge_intelligence` | reference smartphone RAM is 9 GB twice and 8 GB in five places — the same registry value formatted two ways |
| O15 | `hw_acceleration` | quiz teaches runtime precision switching that §1.9.3 explicitly denies |
| O16 | `responsible_ai` | quiz attributes ε=8 to Apple's keyboard; footnote 28 disclaims exactly that attribution |

### Medium

`training` ResNet-50 backward working set 4 GB vs 3.2 GB · `data_engineering`
1,843 vs 12,682 img/s (different quantities — a modeled rate vs a FLOPs ceiling
— but the text does not distinguish them) · `fleet_orchestration` static
partition 400+618 leaves 6 of 1,024 GPUs unexplained (changing it moves four
guards and several published percentages) · `compute_infrastructure` HBM energy
4 pJ/bit here vs 2 pJ/bit in four other places; 2.9–3.2 TB budget "established
earlier" was 2.1 TB; A100 23 percent vs B200 55.2 percent use different byte
bases; TDP definition says cooling lets a chip *exceed* its rated speed; power
components 500+250 W exceed the 700 W budget; single-node training "several
months" vs its own 17 months; `Systems.Reliability.Gpu.mttf_hours` printed as
reader-facing prose · `distributed_training` MFU rendered without a percent
sign; `\ref{pri-bandwidth-latency-tradeoff}` renders as raw LaTeX (the label
does not exist); "80 GB–80 GB" range with identical endpoints ·
`responsible_ai` three stray backticks after math delimiters; Table 8 recaps a
loan example whose every metric conflicts with the chapter's running example;
Kleinberg stated as "at most one of three" · `edge_intelligence` 3–5× vs 5–10×
bandwidth amplification; TinyTL 10× vs a footnote implying 33×; adapter memory
2–5 MB vs 200 KB for a higher rank; device tiers 2–4 GB vs 4–9 GB; 10 W
sustained vs 2–3 W sustained; MobileNetV2 1.883 ms vs 75 ms; flagship 8 GB vs
16 GB · `benchmarking` two refs point at Table 18 for data it does not contain;
"energy implications in §1.7.2.4" is the wrong section; "(~2.2 mJ vs. 2.2 mJ)";
"the 1,825-sample estimate" never derived · `responsible_engr` a 10 ms
reduction on a 10 ms budget is 100 percent (elsewhere 20 percent); "The table"
with no antecedent; a dangling penalty sentence · `ml_workflow` quiz answers
assert saturation and super-linear scaling the body hedges to the opposite ·
`introduction` a 36,000-frame setup whose arithmetic was deleted · `conclusion`
"That lesson" with no antecedent · `appendix_machine` 581× vs 582× ·
`appendix_assumptions` claims every estimate traces to its tables, but TPU v5p,
mobile memory capacity, and component MTTF have no rows

---

## 5b. Wave-2 deep review — 8 Vol 1 chapters, one agent per rendered file

Every section of `nn_architectures`, `data_selection`, `model_compression`,
`hw_acceleration`, `benchmarking`, `model_serving`, `ml_ops`, and
`responsible_engr` was read as the reader sees it, against the LEGO cell that
produced each number. **127 findings.** The mechanical and clearly-correct ones
are fixed on this branch. The three below are blockers that need an authorial
decision, because each one is a published figure or worked example whose
correct value is a judgment call.

### B1 — `hw_acceleration` figure 10 plots a fabricated series (BLOCKING)

The caption states that between 2000 and 2025 compute reached `10^12` times its
baseline against bandwidth's `10^3`, "leaving a nine-order-of-magnitude
separation." The plotted arrays are hardcoded:

```python
compute_performance = [1e3, 1e5, 1e7, 1e9, 1e12, 1e15]
memory_bandwidth    = [1, 10, 50, 100, 500, 1000]
```

The chapter's own measured data says otherwise. The compute-to-bandwidth ratio
**is** the ridge point, and figure 11 and table 12, both in the same section,
give V100 138.9, A100 153, H100 295.2, B200 281 FLOP/byte: a 2.1x change over
seven years, not 10^9 over twenty-five. Real anchors put compute growth near
10^6 and bandwidth near 10^3.5, so the gap is two to three orders.

Fixing this means choosing sourced endpoints for both series, which changes a
published figure. Not touched: an unsourceable figure does not ship, and neither
does a replacement invented here.

### B2 — `model_serving` Napkin Math 1.3 rests on a service time the chapter contradicts

The napkin uses 5 ms as the TensorRT FP16 service time. Tables 2, 19, and 20 all
give 1.4 ms for the same model on the same GPU. Every downstream number depends
on it: rho <= 1 - (4.6 x 5)/50 = 0.54, 47 GPUs, 62 with headroom, and the "62
V100s" restated in the economics section. Redone at 1.4 ms the answer is 9 GPUs,
12 with headroom. Either the napkin's 5 ms is a deliberately conservative
batch-inclusive figure that the box should say out loud, or the whole worked
example needs rebuilding at 1.4 ms.

### B3 — `model_serving` prefill ceiling rests on an unstated 10,000 tokens/s

The unit-economics ceiling of 45.2 million tokens/hour and the $0.066/million
lower bound come from `prefill_tokens_per_s_value = 10_000`, which never reaches
the page and contradicts the same section's 41.9 ms TTFT for a 1,000-token
prompt (implying 23,866 tokens/s). A sustained rate cannot be below the
single-request rate the chapter just derived.

### Also open, lower stakes

`data_selection` FixMatch 8.1x is computed against a 4,000-label baseline while
the callout tells the reader the baseline is 50K, and none of its inputs are
shown; the ImageNet size prints as 1.28M, 1.2M, and 1.3M in three worked
examples; a quiz gives $46,000 as the recurring annual net when it is the first
year's. `model_compression` prints the INT8-vs-FP32 arithmetic-energy gain as
18.5x, 30x, and 20x in three adjacent sections; a self-check asserts a ~90
percent sparsity threshold the body twice says does not exist; a BERT footprint
ratio is stated as 16x where 440/28 = 15.7. `benchmarking` puts the
microcontroller-to-cluster power span at "nearly eight orders" in a keyed answer
where its own table caption says nearly eleven; an "application benchmarks"
scope is named once and never defined. `ml_ops` states that Knight Capital was
bankrupted (it was rescued, then acquired) and gives 440 million shares where
the SEC order it cites says 397 million; four corporate authors render as
inverted personal names ("Cloud, Google"). `responsible_engr` calls 85 percent
the aggregate accuracy where the body assigns it to the majority group and 82.5
percent to the aggregate. `hw_acceleration` has three self-check answers that
teach runtime precision switching and runtime re-tiling the body explicitly
denies, and one that assigns BatchNorm's per-channel statistics to LayerNorm.

---

## 5c. Verification pass — 2026-08-17

The §5 and §5b lists above were written against an earlier state of the branch
and are now **partly stale**: the intervening fix waves closed many of them
without striking them from the ledger. Every mechanical candidate was
re-checked against current source before any edit. Do not re-chase a row below
without re-verifying it first.

### Applied

| Finding | Fix |
|---|---|
| `ml_ops` Knight Capital: 440M shares, "bankrupted the firm within 48 hours" | The cited SEC order (Release No. 70694) gives over 4 million executions covering more than 397 million shares and a \$460M loss. The firm was never bankrupted: emergency rescue financing days later, acquisition months after. Both corrected; citation stays on Context per `citation.md` (same source). |
| `benchmarking` quiz: microcontroller-to-training-cluster span "nearly eight orders" | Eight orders is the **table's** span (150 µW → 10 kW, 10^7.8). The caption gives the training-cluster span as 5.6 µW → ~498 kW, nearly eleven orders (10^10.9). Magnitude corrected, endpoint kept — the keyed answer's own explanation depends on it. |
| `hw_acceleration` quiz: LayerNorm "per-channel statistics" | Those are BatchNorm's. LayerNorm normalizes each sample across the feature dimension. Mechanism phrase only; the arithmetic-intensity claim and keyed answer were already right. |
| `ml_ops` four corporate authors render as inverted personal names | Five `author` fields were single-braced, so BibTeX parsed them as Surname/Firstname. Double-braced Google Cloud (×2), NVIDIA Corporation, European Commission; split the mangled Greenhouse Gas Protocol field into its two real corporate authors. |

### Verified already resolved — no action needed

`distributed_training` `\ref{pri-bandwidth-latency-tradeoff}` rendering as raw
LaTeX (the label never existed; the real one is `pri-alpha-beta-model`) ·
`distributed_training` "80 GB–80 GB" identical endpoints · `responsible_ai`
stray backticks after math delimiters · `appendix_machine` 581× vs 582× ·
`benchmarking` "(~2.2 mJ vs. 2.2 mJ)" · `benchmarking` table caption
eight-vs-eleven orders, which now states both spans explicitly and correctly ·
**O5** `compute_infrastructure` Definition 1.3 duplicate **Significance** and
missing **Distinction** — all seven definition callouts in that chapter now
carry the complete `{Significance, Distinction, Common pitfall}` set ·
`responsible_engr` 85 percent vs 82.5 percent aggregate accuracy — no 82.5
appears anywhere in the chapter and 85 percent is both declared as
`aggregate_accuracy` and labelled "Aggregate Accuracy" in prose · `conclusion`
"That lesson" — the preceding sentence is a clear antecedent ·
`data_selection` ImageNet printed as 1.28M/1.2M/1.3M — one 1.2M reference
remains, inside a guard message, not reader-facing.

### Verified NOT a defect — do not "fix"

`model_compression` BERT footprint ratio stated as 16× where 440/28 = 15.71.
The cell computes `round(full_mb / compressed_mb)` with a guard pinning 16, the
prose calls the pipeline "illustrative", and both endpoints are printed beside
the ratio so a reader can check it. That is honest rounding of a documented
scenario, not drift.

### Still open and deliberately untouched

Everything in §5b B1–B3 and the §5 High table that turns on **which of two
values is canonical** stays for the author (item E). The quiz-policy question
is unresolved and gates a further batch: many self-check answers assert
statistics their chapter never states, and one ruling — rewrite against the
body, or retire the item — would settle dozens of findings at once.

## 6. Build domain — reported, not touched

The first two are Zeljko's; the markers stay in the source exactly as he placed
them. The third is a property of the website render, not of the prose.

| Issue | Effect | Scope |
|---|---|---|
| non-LaTeX writers drop a raw TeX inline **with its contents** | five Vol 1 sentences lose a word on the web build | `\mbox{}` × 5 |
| `[offset=NNmm]` is stripped only in the LaTeX branch of `sidenote.lua` | prints as literal text at the head of footnotes on the web | 32 across both volumes |
| a cross-chapter `@sec-` renders as the target's **title** on the web and as its **number** in print | the web build reads as a tautology where the prose also names the concept | 123 sites, 35 pages |

### The title-echo class — why the prose is right and the web build is wrong

The PDF is a Quarto **book** and numbers its sections; the website is a Quarto
**site** and has no numbering, so a cross-chapter reference falls back to the
target's title. The same sentence therefore reads:

| Source | PDF (the deliverable) | Web |
|---|---|---|
| `The Roofline Model in @sec-...` | The Roofline Model in section 11.3 | The Roofline Model in Roofline Model |
| `The Lottery Ticket Hypothesis (@sec-...)` | The Lottery Ticket Hypothesis (section 10.3.1.7) | The Lottery Ticket Hypothesis (Lottery ticket hypothesis) |
| `Model compression (@sec-...)` | Model compression (Chapter 10) | Model compression (Model Compression) |
| `...established in @sec-...` | …established in section 1.7 | …established in Iron Law of ML Systems |

Every one of these was verified against the built Vol 1 PDF. **Rewriting the
prose to read well on the web would damage the print edition**, which is the
MIT Press deliverable, so nothing here was touched. The fix, if wanted, is a
site-side crossref decision (number the website's sections, or set a crossref
format that emits "Chapter N"), not 123 prose edits.

Distinguish this from the eight **bare-`equation`** refs, which were fixed: those
rendered as the literal word "equation" with no number and left the sentence
ungrammatical on the web (`"in equation actionable"`), and naming the equation in
prose improved both formats.

The first two rows need a layout-side decision; the third needs a build-config
decision. None is a prose edit.

---

## 7. Checks added

Run before any push; each catches a class found here.

| Tool | Catches | Cost |
|---|---|---|
| `book/tools/audit/check_lego_health.py` | every precision-guard and narrative-guard failure in one pass, guards softened so nothing hides behind an earlier failure | ~2 min |
| `book/tools/audit/check_inline_ref_order.py` | an inline ref used before its cell defines it — the defect that killed the Vol 2 build at chapter 28 of 38 | ~1 s |
| `book/tools/audit/check_rendered_html.py` | reader-visible leaks in a built volume: literal `{python}`, `?@ref`, tracebacks, `[offset=]`, dropped LaTeX macros, doubled units | ~5 s |
| `book/tools/audit/render_reader_view.py` | not a check — flattens a built chapter into the reader's view so a review reads what ships | — |

Verified against the pre-fix revision: `check_lego_health` reports all 19
original defects; `check_inline_ref_order` pinpoints the exact line.

### Sign-off state

Both volumes build clean (36 + 38 pages, exit 0). Gates 1 and 2 pass with zero
findings. Gate 3 reports 34, all accounted for:

| Class | Count | Status |
|---|---|---|
| `offset-directive` | 31 | layout-owned; markers left exactly as placed |
| `doubled-unit` | 3 | false positives: math-stripped `6 x params x tokens`, and a step boundary |

The doubled-unit check now requires a plural first unit, since only a
self-labeling formatter emits one ("128 GPUs accelerators"). A singular first
unit is ordinary English ("64 GPU workers") and no longer fires.
