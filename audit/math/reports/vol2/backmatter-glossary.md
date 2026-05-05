# Math Audit: Volume 2 Backmatter Glossary

Source audited: `book/quarto/contents/vol2/backmatter/glossary/glossary.qmd`

This glossary is mostly prose definitions. I audited the entries containing explicit numbers, units, metric definitions, ratios, probability statements, and complexity/scaling claims. The following issues are substantive math/numeric/prose-equation consistency concerns.

## Findings

1. **Line 468-470: F1 score is described as "model accuracy."** The F1 score is the harmonic mean of precision and recall, usually \(2PR/(P+R)\), but it is not the same as accuracy \((TP+TN)/(TP+TN+FP+FN)\). Calling it "a measure of model accuracy" can mislead readers because F1 intentionally omits true negatives and behaves differently from accuracy under class imbalance. Suggested revision: "A classification metric that combines precision and recall into a single score, calculated as their harmonic mean."

2. **Lines 508-510: `flops` expands to the wrong quantity for the casing used.** The entry title is lowercase "flops", which conventionally means floating-point operations, while FLOPS means floating-point operations per second. The definition gives "Floating-point operations per second," so the term and unit are inconsistent. Suggested revision: either rename the term to "FLOPS" for a rate, or define "FLOPs" as floating-point operations and mention FLOPS separately as operations per second.

3. **Lines 877-879: non-IID is incorrectly tied to "uniformly distributed."** IID means independent and identically distributed; the identical distribution need not be uniform. Data can be IID while non-uniform, and non-IID data can arise from dependence, different client distributions, temporal drift, or unequal label/feature distributions. Suggested revision: "Data whose samples are not independent, not identically distributed, or both across devices or time..."

4. **Lines 830-832: Moore's law conflates transistor doubling with a halving claim about computer cost.** Moore's law is the observation that transistor counts on integrated circuits roughly double over a period such as about two years. The added statement that "the cost of computers is halved" is not the standard mathematical form and is too broad: the historical cost claim is closer to falling cost per transistor or per unit compute, not the price of computers generally. Suggested revision: "The observation that transistor counts on integrated circuits have historically doubled approximately every two years, reducing cost per transistor or increasing compute density over time."

## No Issue

- Line 296: The 8-9$\times$ reduction for depthwise separable convolutions is a reasonable rule-of-thumb for common 3x3 convolutions with many channels.
- Lines 433, 685, 757, 812, 1109, 1169, and 1201: The numeric units and thresholds checked here are internally consistent as glossary-level approximations.
- Lines 928 and 1285: PUE and WUE are stated with the correct numerator/denominator relationship at glossary level, though WUE has compound units rather than being dimensionless.
