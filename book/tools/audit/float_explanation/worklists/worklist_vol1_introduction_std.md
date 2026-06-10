# Float exposition eval — introduction.qmd (vol1)
Standard: FLOAT_EXPOSITION_STANDARD.md (caption excluded from prose budget)

## Summary

| type | level | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|---|
| equation | 🔴 | 6 | 6 | 0 | 0 |
| algorithm | 🔴 | 0 | — | — | — |
| table | 🟠 | 9 | 4 | 5 | 0 |
| figure | 🟠 | 8 | 8 | 0 | 0 |
| listing | 🟡 | 0 | — | — | — |
| **total** | | **23** | **18** | **5** | **0** |

---

## Findings (⚠️ only)

### ⚠️ `tbl-software-1-vs-2` (table 🟠) — def L128

- **Ref (body prose):** "@Tbl-software-1-vs-2 summarizes this paradigm shift." (L116)
- **Missing move:** lead-out. The citation is a bare pointer. The payoff sentence (L130) pivots to technical debt without extracting any conclusion from the table. The key insight — that failure mode shifts from loud (crash) to silent (metric degradation), and debugging moves upstream from code to data — lives only in the cells and the caption.
- **Suggested rewrite (no em-dash/hyphen, one colon max, content leads):**
  ```diff
  - @Tbl-software-1-vs-2 summarizes this paradigm shift.
  + @Tbl-software-1-vs-2 makes the engineering consequence concrete: in Software 2.0, failures no longer announce themselves as crashes but degrade silently as metric drift, and debugging shifts upstream from tracing execution paths to inspecting data distributions.
  ```

---

### ⚠️ `tbl-dam-taxonomy` (table 🟠) — def L1550

- **Ref (body prose):** "ML systems engineering is the discipline of keeping all three axes in balance. @Tbl-dam-taxonomy formalizes each axis's role." (L1542)
- **Missing move:** lead-out. The citation is a bare pointer. The payoff (L1552) pivots to the four-layer stack. The table's specific load-bearing content — that Data is the fuel (information), Algorithm is the blueprint (logic), and Machine is the engine (physics) — lives only in the cells.
- **Suggested rewrite:**
  ```diff
  - ML systems engineering is the discipline of keeping all three axes in balance. @Tbl-dam-taxonomy formalizes each axis's role.
  + ML systems engineering is the discipline of keeping all three axes in balance. @Tbl-dam-taxonomy formalizes each axis's role: Data is the fuel that defines what the system learns, Algorithm is the blueprint that defines how patterns are captured, and Machine is the engine that defines where and how fast computation occurs.
  ```

---

### ⚠️ `tbl-lighthouse-examples` (table 🟠) — def L1938

- **Ref (body prose):** "@Tbl-lighthouse-examples summarizes why each lighthouse model serves as a diagnostic tool for specific system bottlenecks." (L1926)
- **Missing move:** lead-out. The citation is a bare pointer. The payoff (L1946) restates the division-of-labor framing without naming the table's key result. The concrete diagnostic mappings (ResNet-50 is compute-bound, GPT-2 is bandwidth-bound, DLRM is capacity-bound, MobileNetV2 is latency-bound, Keyword Spotting is power-bound) live only in the cells. The excellent analytic prose at L1924-1928 precedes the citation and functions as lead-in, not lead-out.
- **Suggested rewrite:**
  ```diff
  - @Tbl-lighthouse-examples summarizes why each lighthouse model serves as a diagnostic tool for specific system bottlenecks.
  + @Tbl-lighthouse-examples summarizes the diagnostic mapping: ResNet-50 probes compute throughput, GPT-2/Llama probes memory bandwidth, DLRM probes memory capacity, MobileNetV2 probes latency and power, and Keyword Spotting probes the extreme power envelope where milliwatts are the binding constraint.
  ```

---

### ⚠️ `tbl-efficiency-priorities` (table 🟠) — def L2982

- **Ref (body prose):** "Each position on this deployment spectrum creates distinct bottlenecks that determine which efficiency dimensions matter most, as summarized in @tbl-efficiency-priorities:" (L2972)
- **Missing move:** lead-out. The colon at the end of the citation sentence suggests a payoff is coming, but L2972 ends there with no following prose before the table. The payoff paragraph (L2986) opens a new topic ("The deployment spectrum represents more than different hardware configurations") without extracting the table's conclusion. The key contrast — that cloud systems optimize throughput and cost while TinyML requires extreme compression across all dimensions simultaneously — lives only in the cells.
- **Suggested rewrite:**
  ```diff
  - Each position on this deployment spectrum creates distinct bottlenecks that determine which efficiency dimensions matter most, as summarized in @tbl-efficiency-priorities:
  + Each position on this deployment spectrum creates distinct bottlenecks that determine which efficiency dimensions matter most, as @tbl-efficiency-priorities details. Cloud training optimizes for distributed throughput; TinyML requires extreme compression across every dimension simultaneously because memory and power budgets allow no slack in any axis.
  ```

---

### ⚠️ `tbl-book-structure` (table 🟠) — def L3116

- **Ref (body prose):** "@Tbl-book-structure outlines this organization." (L3107)
- **Missing move:** lead-out. The citation is a bare pointer. The payoff paragraphs (L3118-3124) narrate each part's chapters in detail but never state the table's load-bearing conclusion: that each later part assumes cumulative mastery of all earlier parts, making the sequence non-negotiable rather than optional. That dependency logic appears only in the caption.
- **Suggested rewrite:**
  ```diff
  - @Tbl-book-structure outlines this organization.
  + @Tbl-book-structure outlines this organization. The sequence is non-optional: Part III's optimization techniques assume the model architectures of Part II, and Part IV's deployment practices assume both, so a reader who skips a part loses the constraint vocabulary that the next part takes for granted.
  ```

---

## Notes on ✅ floats

All six **equations** pass: every symbol is named in prose before or immediately after the display, the consequence is stated (what the equation implies for engineering practice), and the payoff paragraphs are substantive. `eq-intro-iron-law` is the strongest instance — the surrounding prose teaches the equation's diagnostic purpose, its distinction from the P&H multiplicative law, and its three-term physical decomposition.

All eight **figures** pass: each citation sentence delivers the figure's takeaway in body prose (not only in the caption). `fig-alexnet` is particularly strong: the citation sentence names the co-design mechanism (two parallel streams reflecting GTX 580 memory limits) rather than only pointing at the figure. `fig-algo-efficiency` is backed by a payoff paragraph that quantifies the ~44× improvement and places it against the Moore's Law cadence.

Four of the nine **tables** pass: `tbl-ai-evolution-strengths`, `tbl-ai-evolution-performance`, `tbl-introduction-deployment-paradigms`, and `tbl-introduction-engineering-missions` all have citation sentences or payoff paragraphs that state the table's conclusion in prose rather than delegating it to the cells.
