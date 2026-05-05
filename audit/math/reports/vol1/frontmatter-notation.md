# Math Audit: vol1/frontmatter/notation.qmd

Source audited: `book/quarto/contents/vol1/frontmatter/notation.qmd`

Effective content audited: `book/quarto/contents/vol1/frontmatter/_notation_body.qmd`, included by `notation.qmd` line 7.

## Summary

The notation page defines the book's core symbol and unit conventions. The main algebraic forms are mostly dimensionally coherent, but several definitions introduce the same collisions the page is trying to prevent. The largest issues are in the degradation equation, where `$D$` is reused for divergence despite `$D$` being reserved for dataset size, and `$P_t$`/`$P_0$` are used for distributions despite `$P$` being reserved for parameter count.

No source `.qmd` files were modified.

## Findings

### 1. Degradation equation reuses `$D$` after reserving `$D$` for dataset size

- `_notation_body.qmd` lines 53 and 61-62: `D(P_t \| P_0)` denotes statistical divergence.
- `_notation_body.qmd` lines 25, 83, and 111: `$D$` is explicitly reserved for dataset size/training data, while data volume should use `$D_{\text{vol}}$`.
- Issue: This creates a direct notation collision. The page's stated convention is that ML single-letter meanings take precedence and systems/statistical concepts should receive disambiguating notation. A reader can reasonably parse `$D(P_t \| P_0)$` as a function named `D`, but the surrounding table also says `$D$` means dataset size, so the convention is not self-consistent.
- Proposed correction: Replace the divergence notation with a calligraphic or named divergence operator, for example:
  - Equation: `\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot \mathcal{D}(P_t \lVert P_0)`
  - Table symbol: `$\mathcal{D}(P_t \lVert P_0)$`
  - Threshold condition: `$\mathcal{D}(P_t \lVert P_0) > \tau$`

### 2. Degradation equation reuses `$P$` for distributions while reserving `$P$` for parameter count

- `_notation_body.qmd` lines 53 and 59-60: `$P_t$` and `$P_0$` are current/training distributions.
- `_notation_body.qmd` lines 28, 82, and 110: `$P$` is reserved for parameter count.
- Issue: The table note on line 59 says "`Not parameters--use $P$ for parameter count`" while the same row uses `$P_t$` for a distribution. That is internally contradictory for a notation guide.
- Proposed correction: Use a distribution symbol that does not collide with parameter count, for example:
  - `$\mathcal{P}_t$`: current data distribution at time `$t$`
  - `$\mathcal{P}_0$`: training data distribution
  - Equation: `\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot \mathcal{D}(\mathcal{P}_t \lVert \mathcal{P}_0)`

### 3. Sensitivity `$\lambda$` is not a unitless scalar in the stated equation

- `_notation_body.qmd` lines 53 and 58: `$\lambda$` is listed as `Scalar`.
- Issue: In `Accuracy(t) = Accuracy_0 - \lambda \cdot divergence`, `$\lambda$` must carry the units of the accuracy measure per unit divergence. If accuracy is reported as a fraction, `$\lambda$` has units of fraction per divergence unit; if accuracy is reported as percent, it has units of percentage points per divergence unit. Calling it only a scalar hides the dimensional dependency.
- Proposed correction: Change the unit/type entry to "accuracy units per divergence unit" or "fraction (or percentage points) per unit divergence." If the book intends divergence to be dimensionless and accuracy to be a unitless fraction, say that explicitly.

### 4. Divergence choices do not all have the same units

- `_notation_body.qmd` line 61: common choices include KL divergence, total variation, and Wasserstein.
- Issue: KL divergence and total variation are dimensionless for probability distributions, while Wasserstein distance inherits units from the underlying metric on the sample space. This matters because lines 58 and 62 define `$\lambda$` and `$\tau$`; their units depend on which divergence/distance is used.
- Proposed correction: Either restrict the examples to dimensionless divergences in this notation page, or revise the notes:
  - `$\mathcal{D}$ may be dimensionless, as with KL or total variation, or metric-valued, as with Wasserstein; $\lambda$ and $\tau$ must be calibrated in matching units.`

### 5. Learning-rate row contradicts the hardware-efficiency convention

- `_notation_body.qmd` line 87: `$\eta$` is learning rate and "Also efficiency--context distinguishes."
- `_notation_body.qmd` lines 29 and 113: hardware efficiency is explicitly `$\eta_{\text{hw}}$`, and bare `$\eta$` is learning rate.
- Issue: The line 87 note weakens the page's own convention. If bare `$\eta$` can also mean efficiency by context, the collision is not actually resolved.
- Proposed correction: Replace the note with: `*(Never bare hardware efficiency; use $\eta_{\text{hw}}$.)*`

### 6. Energy comparison uses an imprecise inequality across different per-event units

- `_notation_body.qmd` line 72: `E_{\text{move}} \gg E_{\text{compute}}`.
- `_notation_body.qmd` lines 72-73: `E_{\text{move}}` is joules/byte and `E_{\text{compute}}` is joules/FLOP.
- Issue: Both are energies per counted event, but the events differ. The comparison is common in systems prose, but as written in a notation table it can look like a direct unit comparison between joules/byte and joules/FLOP. The equation on line 68 is dimensionally valid; the note should be phrased more carefully.
- Proposed correction: `For typical hardware, moving one byte often costs more energy than performing one floating-point operation, so data movement can dominate total energy.`

### 7. FLOP count and FLOP-rate conventions need sharper separation

- `_notation_body.qmd` lines 27-28: `$O$` has unit `FLOPs`; `$R_{\text{peak}}$` has unit `FLOP/s`.
- `_notation_body.qmd` lines 94-95: "GFLOPs, TFLOPs" and `1 TFLOP = 10^{12} FLOPs`.
- Issue: The page uses pluralized FLOPs for operation counts and FLOP/s for rates, which is mostly workable. However, elsewhere in the book peak compute is often written as TFLOPS, so this notation page should explicitly distinguish count abbreviations from rate abbreviations to prevent dimensional slips.
- Proposed correction: Add a sentence under Compute: `Use FLOP, GFLOP, and TFLOP for operation counts; use FLOP/s, GFLOP/s, and TFLOP/s for rates. Avoid using TFLOPS when a count is meant.`

### 8. Decimal byte-prefix notation conflicts with strict SI symbol casing

- `_notation_body.qmd` line 93: `KB = 10^3 bytes`.
- Issue: The page says it uses "decimal SI prefixes only." In SI-style notation, kilo is lowercase `k`, so the strict decimal byte symbol is `kB`, not `KB`. The book may intentionally prefer the common computing style `KB`, but the current wording claims strict SI while using non-SI casing.
- Proposed correction: Either use `kB = 10^3 bytes`, or revise the prose to say the book uses decimal byte prefixes with conventional computing spellings: `KB = 10^3 bytes, MB = 10^6 bytes, ...`.

### 9. Duplicate prose line at the end

- `_notation_body.qmd` lines 115-116: line 115 ends with "the vast ML literature"; line 116 repeats `ML literature.`
- Issue: This is not mathematical, but it is a prose consistency error in the notation page and may confuse readers at the end of the convention statement.
- Proposed correction: Delete line 116.

## Checked But No Issue

- `_notation_body.qmd` line 18: The iron-law equation is dimensionally consistent: bytes divided by bytes/s gives seconds, FLOPs divided by FLOP/s gives seconds, and latency is seconds.
- `_notation_body.qmd` line 68: The total-energy equation is dimensionally consistent: bytes times joules/byte plus FLOPs times joules/FLOP gives joules.
- `_notation_body.qmd` lines 97-101: Precision byte counts are correct for FP32, FP16, BF16, FP8, and INT8 as stated.
- `_notation_body.qmd` lines 24-30 and 79-88: The core ML/systems collision table is coherent except for the degradation-equation collisions noted above.
