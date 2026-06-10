# Float-explanation worklist — sustainable_ai.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 25 | 24 | 1 | 0 |
| table | 12 | 12 | 0 | 0 |
| listing | 1 | 1 | 0 | 0 |
| algorithm | 0 | 0 | 0 | 0 |
| equation | 17 | 17 | 0 | 0 |
| **dangling ref** | 1 | — | — | 1 |
| **total** | 56 | 54 | 1 | 1 |

---

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

---

### ⚠️ `fig-mckinsey-analysis` — def L2832  (Thin)

- **Caption:** **AI Hardware Market Growth**: McKinsey analysis comparing 2017 market estimates with then-projected 2025 data center and edge markets. Inference workloads dominated the projected growth, with edge inference treated as a significant new segment while training markets grew more gradually in the projection.
- **Ref:** L2830 `@fig-mckinsey-analysis`: "Early market forecasts anticipated this shift: a 2017 McKinsey projection (@fig-mckinsey-analysis) expected data center and edge inference markets to grow faster than training through 2025. The stronger evidence is physical rather than economic. The Meta lifecycle measurements in @fig-meta-analysis show inference serving at scale rivaling or exceeding training emissions for deployed recommendation models, and the chapter's own accounting shows continuous serving overtaking a one-time training run once request volume is large enough."
- **Context checked:** ref ✓ (names the projection) · prev ¶ ✓ (context of inference serving energy) · next ¶ ✗ (figure definition) · caption ✓ (restates) · payoff ¶ ✗ (L2932 discusses Alexa/Siri/Google scaling, not the McKinsey figure's content)
- **Issue:** The reference sentence immediately dismisses the figure ("the stronger evidence is physical rather than economic") without telling the reader what specific numbers or market shares the figure actually shows. The caption restates that inference dominated projected growth, but no prose draws out what the concrete values are (for example, inference doubling from 4-5 to 9-10 billion dollars, edge growing from near zero to 4-4.5 billion), nor does any payoff paragraph revisit it. The figure is cited only to be superseded, leaving the reader with no reason to consult it.
- **Suggested rewrite (flag-only):**
  ```diff
  - Early market forecasts anticipated this shift: a 2017 McKinsey projection (@fig-mckinsey-analysis) expected data center and edge inference markets to grow faster than training through 2025. The stronger evidence is physical rather than economic.
  + Early market forecasts anticipated this shift. A 2017 McKinsey projection (@fig-mckinsey-analysis) showed data center inference doubling and edge inference growing from near-zero to 4 billion dollars by 2025, while training grew only gradually. That economic signal reinforces the physical evidence: the Meta lifecycle measurements in @fig-meta-analysis show inference serving at scale rivaling or exceeding training emissions for deployed recommendation models, and the chapter's own accounting shows continuous serving overtaking a one-time training run once request volume is large enough.
  ```

---

### 🛑 `@tbl-prefill-decode` — dangling reference at L2942  (Missing definition)

- **Ref:** L2942: "The prefill/decode distinction summarized in @tbl-prefill-decode extends beyond latency into energy efficiency."
- **Issue:** No table with label `tbl-prefill-decode` is defined anywhere in `sustainable_ai.qmd`. The scanner confirms this is a dangling reference. The explanation of the prefill/decode energy distinction is present in prose at L2944-2947 (bullet list and the static-power-waste paragraph), so the underlying content exists, but the table the reference points to does not. Either the table was removed during editing without updating the reference, or it lives in a different chapter (the inference chapter, where prefill/decode is introduced) and the cross-chapter reference label was intended. This reference will produce a broken link in the rendered output.
- **Context checked:** ref ✗ (points to nonexistent label) · prev ¶ ✓ (fig-prefill-decode-energy just defined) · next ¶ ✓ (prose explains the distinction in bullets) · caption N/A (no table) · payoff ✓ (prose covers the implication)
- **Suggested fix (flag-only):**
  Option A — if the table exists in the inference chapter: replace `@tbl-prefill-decode` with the correct cross-chapter label (for example `@tbl-prefill-decode-energy` or whatever the inference chapter uses).
  Option B — if the table was removed: delete the dangling reference phrase and let the existing prose carry the explanation:
  ```diff
  - The prefill/decode distinction summarized in @tbl-prefill-decode extends beyond latency into energy efficiency.
  + The prefill/decode distinction extends beyond latency into energy efficiency.
  ```
