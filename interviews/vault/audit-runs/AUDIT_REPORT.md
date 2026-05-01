# Vault chain pipeline — independent audit report

**Generated:** 2026-05-01T22:01:57+00:00
**Auditor:** gemini-3.1-pro-preview (independent of the pipeline's own judges)
**Audit run dir:** `interviews/vault/audit-runs/20260501T213817Z`

---

## Summary

The ML systems interview question pipeline audit reveals mixed performance across generation stages. While primary and secondary chains achieved pass rates above 60%, the delta_zero pairs failed almost universally due to disjoint scenarios. Additionally, gap detection is highly unreliable, frequently hallucinating missing levels between disconnected questions.

## Headline findings

- Delta_zero pairs have a near-zero pass rate (2%) because the pipeline consistently fails to maintain a shared scenario thread across consecutive questions.
- Gap detection hallucinates over 50% of the time, inventing bridges between anchor questions that lack a coherent shared context or hardware platform.
- Secondary and Primary chains suffer heavily from duplicate pairs, where adjacent levels test the exact same concept or scenario with only minor rephrasing.
- Draft generation struggles with basic realism, occasionally producing physically absurd hardware scenarios or questions with inappropriately low cognitive load.

## Per-category

### drafts

- pass rate: **50%**
- key issue: Unrealistic hardware scenarios and mismatched cognitive load for the target level.

### secondary

- pass rate: **61%**
- key issue: High prevalence of duplicate question pairs testing identical concepts, particularly at L4/L5.

### delta_zero

- pass rate: **2%**
- key issue: Near-total failure to maintain a continuous scenario thread across consecutive questions.

### primary

- pass rate: **64%**
- key issue: Frequent duplication of concepts across levels and repetitive diagnosis scenarios.

### gaps

- pass rate: **48%**
- key issue: Over 50% of detected gaps are hallucinated, attempting to bridge anchor questions with completely disjoint scenarios.

### Tier quality delta (primary vs secondary)

Secondary chains perform only marginally worse than primary chains (61% vs 64%), with both suffering from the same systemic issue of duplicate pairs.

## Recommendations

- Overhaul the delta_zero generation prompt to enforce strict inheritance of the previous question's scenario context.
- Implement a pre-filter in the gap detection pipeline to ensure anchor questions share a semantic topic and hardware platform before attempting to generate a bridge.
- Introduce a cross-level similarity penalty during primary and secondary chain generation to prevent L4 and L5 questions from collapsing into identical scenarios.
- Add physics and hardware boundary checks to the draft generation phase to prevent unrealistic constraints like physically impossible wake-up times.

## Appendix — verified raw counts

The synthesis above is Gemini's compression of the per-category judgements.
This appendix is the human-extracted exact counts straight out of the
per-call JSON traces, so the headline claims can be cross-checked
without re-reading every response.

### Drafts (n=4)

| qid | verdict | rationale |
|---|---|---|
| `mobile-2147` | **accept** | passed all four sub-checks |
| `edge-2536` | **edit** | technical content excellent for L4, but scenario text truncated |
| `edge-2537` | **reject** | "cognitive load too low for L3 — basic arithmetic word problem" |
| `mobile-2146` | **reject** | "0.5 second wake-up at 4W is physically absurd for a mobile NPU" |

### Δ=0 chains (n=55, all of them)

- 54 verdict=`bad`, 1 verdict=`good`
- shared_scenario_for_d0_pair: 54 = `no`, 1 = `yes`
- **the 1 chain that passed:** check the per-call JSON to identify;
  this is the only Δ=0 chain the audit considered legitimate.

### Gaps (n=40 sampled from 386 across both gap files)

- 21 hallucinated (52.5%), 19 real (47.5%)

### Primary chains (n=100 sampled from 373)

- 64 good (64%), 22 weak (22%), 14 bad (14%)

### Secondary chains (n=100 sampled from 506)

- 61 good (61%), 33 weak (33%), 6 bad (6%)
- **note:** secondary "bad" rate (6%) is *lower* than primary (14%) —
  one possible explanation is that the primary sample includes more
  L3-L5 chains where the audit applied stricter judgement, while the
  secondary sample skewed lower-level. Worth a follow-up split-by-level.

---

Per-call traces are in `interviews/vault/audit-runs/20260501T213817Z/`. Each `0N_*.json` file contains the prompt-char count, the IDs in scope, and the raw Gemini response. Use these for ground-truth follow-up — the synthesis above is one model's compression of the underlying judgements.
