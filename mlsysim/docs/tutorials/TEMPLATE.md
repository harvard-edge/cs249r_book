# Tutorial Template — MLSys·im

This document defines the canonical structure for all MLSys·im tutorials.
Every tutorial follows this template exactly. Consistency is pedagogical.

---

## Design Principles

1. **One tutorial = one core question.** The title IS the question. Every cell exists to answer it.
2. **Predict → Compute → Reflect.** Before showing code output, tell the reader what to expect.
   After showing it, explain what it means. This is the learning cycle.
3. **The "aha moment" is sacred.** Every tutorial has exactly ONE insight that changes how the
   reader thinks. Frame it, build to it, land it in the Key Insight callout.
4. **Real hardware, real models.** No toy examples. Every tutorial uses published specs from the Zoo.
5. **Code-first, prose-second.** Keep prose tight. The insight comes from the numbers, not from
   reading paragraphs. Explanations serve the code, not the other way around.
6. **Sub-30-second runtime.** Every tutorial runs on a laptop in under 30 seconds. No GPU needed.

---

## Canonical Structure

Every tutorial has exactly these sections, in this order:

```
---
title: "<Core Question as Statement>"
subtitle: "<One-sentence hook — what makes this surprising or important>"
description: "<2-sentence summary for search/SEO>"
---

## The Question                          ← 2-3 sentences framing WHY this matters
                                          No code. Pure motivation.

::: {.callout-note}
## Prerequisites
<What tutorials must be completed first. Link them.>
:::

::: {.callout-note}
## What You Will Learn
<Exactly 3–4 bullet points. Each starts with a verb. Measurable outcomes.>
:::

::: {.callout-tip}
## Background: <Concept Name>            ← Jargon explained for newcomers
<Define key terms needed for THIS tutorial. Keep it to one concept.>
:::

---

## 1. Setup                              ← Hidden import cell + visible 2-line import

## 2. <First Analysis Step>              ← Name describes what we DO, not what we learn
   - Code cell (3–8 lines, heavily commented)
   - Brief prose interpreting the output

## 3. <Second Analysis Step>             ← The sweep / comparison / composition
   - Code cell
   - Prose or callout interpreting the pattern

## 4. <The Reveal>                       ← Where the aha moment lands
   - Code cell showing the surprising result
   - KEY INSIGHT callout (see below)

::: {.callout-important}
## Key Insight
<1–3 sentences. The ONE thing the reader should remember from this tutorial.
This is the "tweet-length" summary. Bold the core claim.>
:::

## 5. <Extension / Composition>          ← OPTIONAL: chain another solver
   - Shows how this connects to the broader system

---

## Your Turn                             ← Always exactly 3 exercises

::: {.callout-caution}
## Exercises

**Exercise 1: Predict before you compute.**
<Always the first exercise. Forces the reader to form a hypothesis before running code.
Structure: predict → run → compare → explain the gap.>

**Exercise 2: Change one variable.**
<Modify a single parameter and predict the effect. Builds intuition for sensitivity.>

**Exercise 3: Connect to another domain.**
<Use a different solver or compose solvers. Shows that walls interact.>

**Self-check:** <One quick mental calculation to verify understanding.>
:::

---

## Key Takeaways

::: {.callout-tip}
## Summary
<Exactly 3–5 bullet points. Each maps to a "What You Will Learn" objective.
Use the same verb structure. The reader should be able to check them off.>
:::

---

## Next Steps
<3–4 links to related tutorials, organized by domain cluster.
Format: **[Tutorial Name](link)** — one-sentence description of what it adds.>
```

---

## Callout Usage (Strict Rules)

| Callout Type | Purpose | When to Use |
|-------------|---------|-------------|
| `{.callout-note}` | Prerequisites, Background, What You Will Learn | Factual, informational |
| `{.callout-tip}` | Background concepts, Summary/Takeaways | Helpful context |
| `{.callout-important}` | **Key Insight** (the aha moment) | Exactly ONCE per tutorial |
| `{.callout-caution}` | Exercises | Always in "Your Turn" section |
| `{.callout-warning}` | Common mistakes / pitfalls | Only when there's a real trap |

---

## Code Cell Guidelines

- **Hidden setup cell**: Always first. `#| echo: false` + `#| output: false`.
  Uses `importlib` path hack for dev, shows clean `pip install` import after.
- **Visible cells**: 3–8 lines each. Every line has a comment or is self-explanatory.
- **Output formatting**: Use f-strings with aligned columns for tables. Always include units.
- **Sweeps**: Use a simple `for` loop with a print header. No pandas/matplotlib unless essential.
- **Variable names**: Use domain terms (`model`, `hardware`, `fleet`, `solver`, `result`).

---

## Naming Convention

Tutorials are numbered by cluster:

```
tutorials/
├── index.qmd                    ← Cluster-organized landing page
├── 00_hello_roofline.qmd        ← Cluster 0: Start Here
├── 01_memory_wall.qmd           ← Cluster 1: Node
├── 02_two_phases.qmd            ← Cluster 1: Node
├── 03_kv_cache.qmd              ← Cluster 1: Node
├── 04_starving_the_gpu.qmd      ← Cluster 2: Data
├── 05_quantization.qmd          ← Cluster 3: Algorithm
├── 06_scaling_1000_gpus.qmd     ← Cluster 4: Fleet
├── 07_geography.qmd             ← Cluster 5: Ops
├── 08_nine_million_dollar.qmd   ← Cluster 5: Ops
├── 09_sensitivity.qmd           ← Cluster 6: Analysis
├── 10_gpu_vs_wafer.qmd          ← Cluster 6: Analysis
├── 12_full_stack_audit.qmd      ← Cluster 7: Capstone
└── extending.qmd                ← Developer appendix (unnumbered)
```

---

## Domain Cluster Tags

Every tutorial's YAML front matter includes a `categories` field for cluster membership:

```yaml
categories: ["node", "beginner"]       # Cluster 1, difficulty level
categories: ["fleet", "advanced"]      # Cluster 4, difficulty level
categories: ["capstone", "advanced"]   # Cluster 7, difficulty level
```

Valid clusters: `start`, `node`, `data`, `algorithm`, `fleet`, `ops`, `analysis`, `capstone`
Valid levels: `beginner`, `intermediate`, `advanced`
