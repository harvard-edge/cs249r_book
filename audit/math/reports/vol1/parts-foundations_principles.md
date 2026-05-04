# Math Audit Report: `book/quarto/contents/vol1/parts/foundations_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/parts/foundations_principles.qmd` for mathematical statements, displayed equations, numeric claims, unit consistency, complexity claims, and prose-equation consistency using direct reasoning only. The file contains two displayed equations and several qualitative complexity/constraint claims. No unit conversions or explicit numeric calculations are present.

## Findings

### Medium: Data-only behavior equation omits the model/program and training process

- **Lines:** 8-9
- **Current text/equation:** Line 8 states that a training dataset change `$\Delta D$` is functionally equivalent to executable logic change `$\Delta P$`, followed by
  `$$ \text{System Behavior} \approx f(\text{Data}) $$`
- **Issue:** The displayed equation makes system behavior a function of data alone. That is stronger than the surrounding claim and is not generally true for ML systems: behavior also depends on the model family, training algorithm, hyperparameters, random seed, objective, feature pipeline, inference code, and serving environment. Even if the intended point is that data acts like source code, a change in training data only affects deployed behavior after the training/retraining pipeline incorporates it. As written, the equation conflicts with line 8's own comparison between data changes and executable-logic changes because `$\Delta P$` has no role in the displayed relationship.
- **Proposed correction:** Include the program/training process in the functional relationship, or explicitly scope the approximation to fixed code and retraining. For example:
  `$$ \text{System Behavior} \approx f(P, D) $$`
  or
  `$$ \text{Trained Model} = \mathcal{A}(P, D), \qquad \text{System Behavior} \approx g(\mathcal{A}(P,D)) $$`
  If the prose should stay lightweight, revise line 8 to: "For fixed training code and configuration, a change in the training dataset (`$\Delta D$`) can change the learned executable behavior much like a change in program logic (`$\Delta P$`)."

### Medium: Data-gravity comparison is dimensionally and operationally ambiguous

- **Lines:** 17-18
- **Current text/equation:** Line 17 says movement cost grows with dataset scale `$(D)$` and can exceed moving compute, followed by
  `$$ C_{\text{move}}(D) \gg C_{\text{move}}(Compute) $$`
- **Issue:** `D` denotes a dataset scale, while `Compute` is not a comparable quantity with a defined size, unit, or movement operation. Moving a dataset has costs proportional to bytes transferred, bandwidth, latency, and energy; "moving compute" could mean shipping code, dispatching a query, moving a model binary, scheduling a job near storage, or physically provisioning hardware. Without defining the object being moved, both sides of the inequality are not well typed. The `\gg` relation is also too absolute: it is true only in regimes where the dataset is much larger than the shipped code/query/model state and the dominant cost is data transfer.
- **Proposed correction:** Define the comparison in terms of data movement versus shipping a smaller compute payload to where data resides. For example:
  `$$ C_{\text{move}}(D) \approx \alpha |D|,\qquad C_{\text{ship}}(K) \approx \alpha |K| + O_{\text{dispatch}},\qquad |D| \gg |K| \Rightarrow C_{\text{move}}(D) \gg C_{\text{ship}}(K) $$`
  where `K` is the code/query/model state sent to the storage location. A prose-only correction would be: "As dataset size `|D|` grows far beyond the size of the code, query, or model state `|K|`, the cost of moving the data can dominate the cost of shipping computation to the data."

## No-Issue Checks

- **Line 3:** "Conservation of Complexity" is presented as a qualitative meta-principle, not a formal conservation law with units or an invariant equation. No mathematical correction is required, though the wording is metaphorical rather than a physically conserved quantity.
- **Line 3:** "Parts I through IV" is internally consistent as a four-part range.
- **Line 5:** The analogy to physical laws is qualitative and does not introduce a checkable equation or unit claim.
- **Lines 14-20:** The "data resists movement" and "Data-to-Compute" / "Compute-to-Data" claims are directionally consistent with the corrected data-gravity formulation above.

