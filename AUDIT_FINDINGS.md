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

## 6. Layout domain — reported, not touched

These are Zeljko's. The markers stay in the source exactly as he placed them.

| Issue | Effect | Scope |
|---|---|---|
| non-LaTeX writers drop a raw TeX inline **with its contents** | five Vol 1 sentences lose a word on the web build | `\mbox{}` × 5 |
| `[offset=NNmm]` is stripped only in the LaTeX branch of `sidenote.lua` | prints as literal text at the head of footnotes on the web | 32 across both volumes |

Both need a layout-side decision, not a prose edit.

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
