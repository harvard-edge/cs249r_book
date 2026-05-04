# Math Audit Report: `book/quarto/contents/vol2/frontmatter/notation.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/frontmatter/notation.qmd` for notation definitions, equations, units, symbols, and prose consistency. The file itself is a wrapper:

- `notation.qmd` line 7 includes `book/quarto/contents/vol1/frontmatter/_notation_body.qmd`.
- `notation.qmd` line 9 includes `book/quarto/contents/vol2/frontmatter/_notation_distributed.qmd`.

The findings below refer to those effective source lines. No source `.qmd` files were modified.

## Findings

### 1. High: Degradation equation reuses `$D$` after reserving `$D$` for dataset size

- `_notation_body.qmd` lines 53 and 61-62 use `D(P_t \| P_0)` for statistical divergence.
- `_notation_body.qmd` lines 25, 83, and 111 reserve `$D$` for dataset size/training data and use `$D_{\text{vol}}$` for data volume.

Explanation: This is a direct collision in a notation chapter whose purpose is to prevent collisions. A reader can parse `D(...)` as a divergence operator, but the same page has already assigned `$D$` to dataset size.

Proposed correction: Use a disambiguated divergence operator, for example:

```tex
\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot \mathcal{D}(P_t \lVert P_0)
```

Then update the table and threshold condition to `$\mathcal{D}(P_t \lVert P_0)$` and `$\mathcal{D}(P_t \lVert P_0) > \tau$`.

### 2. High: Degradation equation reuses `$P$` for distributions while reserving `$P$` for parameter count

- `_notation_body.qmd` lines 53 and 59-60 define `$P_t$` and `$P_0$` as data distributions.
- `_notation_body.qmd` lines 28, 82, and 110 reserve `$P$` for parameter count.
- `_notation_body.qmd` line 59 says `$P_t$` is "Not parameters--use `$P$` for parameter count" while still using a `$P$`-based symbol.

Explanation: The note is internally contradictory. Subscripts do not fully remove the collision because the base symbol remains `$P$`.

Proposed correction: Use distribution notation that does not collide with parameter count, such as `$\mathcal{P}_t$` and `$\mathcal{P}_0$`:

```tex
\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot \mathcal{D}(\mathcal{P}_t \lVert \mathcal{P}_0)
```

### 3. Medium: Sensitivity `$\lambda$` is underspecified dimensionally

- `_notation_body.qmd` lines 53 and 58 define `$\lambda$` as a scalar sensitivity.

Explanation: In `Accuracy(t) \approx Accuracy_0 - \lambda \cdot divergence`, `$\lambda$` must have units matching the reported accuracy scale per unit divergence. If accuracy is a fraction and the divergence is dimensionless, `$\lambda$` is dimensionless; if accuracy is percentage points, or if the distance is Wasserstein, its units change.

Proposed correction: Change the unit/type entry to "accuracy units per divergence unit" or explicitly state the assumed convention: "dimensionless when accuracy is a fraction and `$\mathcal{D}$` is dimensionless."

### 4. Medium: Divergence examples do not share one unit convention

- `_notation_body.qmd` line 61 lists KL divergence, total variation, and Wasserstein as common choices.
- `_notation_body.qmd` lines 58 and 62 define `$\lambda$` and `$\tau$` without explaining that their units depend on the chosen divergence/distance.

Explanation: KL divergence and total variation are dimensionless for probability distributions, while Wasserstein distance inherits units from the underlying metric. A single threshold `$\tau$` and sensitivity `$\lambda$` are only meaningful after fixing the divergence/distance and its units.

Proposed correction: Add a note such as: "`$\mathcal{D}$` may be dimensionless, as with KL or total variation, or metric-valued, as with Wasserstein; `$\lambda$` and `$\tau$` must be calibrated in matching units."

### 5. Medium: Learning-rate row weakens the hardware-efficiency convention

- `_notation_body.qmd` line 87 says `$\eta$` is learning rate and "Also efficiency--context distinguishes."
- `_notation_body.qmd` lines 29 and 113 reserve `$\eta_{\text{hw}}$` for hardware efficiency and bare `$\eta$` for learning rate.

Explanation: The note on line 87 undercuts the page's own convention. If bare `$\eta$` can also mean efficiency, the collision has not been resolved.

Proposed correction: Replace the note with: "Never bare hardware efficiency; use `$\eta_{\text{hw}}$`."

### 6. Medium: Reliability equation uses `$\lambda$` as a rate, but the table gives FIT without the required conversion

- `_notation_distributed.qmd` line 50 defines `R_{\text{system}}(t) = e^{-N\lambda t}`.
- `_notation_distributed.qmd` line 59 gives `$\lambda$` unit `FIT`, failures per billion device-hours.

Explanation: The exponent of an exponential must be dimensionless. If `$\lambda$` is a per-device failure rate in `1/hour` and `$t$` is in hours, `N\lambda t` is dimensionless. If `$\lambda$` is reported in FIT, the exponent should use `N \lambda_{\text{FIT}} t / 10^9` for `$t$` in hours. As written, the formula and unit table are inconsistent.

Proposed correction: Either define `$\lambda$` as a failure rate in `1/time`, with FIT only as a reporting convention, or rewrite the equation for FIT explicitly:

```tex
R_{\text{system}}(t) = \exp\!\left(-\frac{N\lambda_{\text{FIT}}t_{\text{hours}}}{10^9}\right)
```

### 7. Medium: Checkpoint interval formula mixes seconds and hours

- `_notation_distributed.qmd` line 54 defines `\tau_{\text{opt}} = \sqrt{2 \cdot T_{\text{write}} \cdot \text{MTBF}}`.
- `_notation_distributed.qmd` lines 60, 62, and 63 list `MTBF` in hours, `\tau_{\text{opt}}` in seconds, and `T_{\text{write}}` in seconds.

Explanation: The square root of `seconds * hours` is not seconds. Young-Daly-style checkpoint formulas require `T_{\text{write}}`, `MTBF`, and `\tau_{\text{opt}}` to use a consistent time unit.

Proposed correction: State that `MTBF` must be converted to seconds before applying the formula, or list all three quantities in the same time unit:

```tex
\tau_{\text{opt}} = \sqrt{2 T_{\text{write}} \text{MTBF}_{\text{seconds}}}
```

### 8. Medium: Reliability function symbol in equation and table do not match

- `_notation_distributed.qmd` line 50 uses `$R_{\text{system}}(t)$`.
- `_notation_distributed.qmd` line 58 defines `$R(t)$` as the reliability function.

Explanation: This notation table should define the exact symbol used in the equation. As written, `$R_{\text{system}}(t)$` appears in the equation but `$R(t)$` appears in the table.

Proposed correction: Change the table symbol to `$R_{\text{system}}(t)$`, or change the equation to use `$R(t)$` and reserve the subscript for prose.

### 9. Medium: Prose says "single-component reliability" but the equation is system reliability

- `_notation_distributed.qmd` lines 48-50 state: "Single-component reliability follows an exponential distribution" followed by `R_{\text{system}}(t) = e^{-N\lambda t}`.

Explanation: The equation is for a system of `$N$` independent components, each with constant failure rate `$\lambda$`, assuming the system fails when any component fails. The prose describes a single component, but the formula includes the system-size factor `$N$`.

Proposed correction: Split the definitions:

```tex
R_{\text{component}}(t) = e^{-\lambda t}, \qquad
R_{\text{system}}(t) = e^{-N\lambda t}
```

Then state the independence and fail-on-any-component assumptions.

### 10. Medium: Distributed step-time law defines `T_sync` but does not connect it to the equation

- `_notation_distributed.qmd` line 12 defines `T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}`.
- `_notation_distributed.qmd` line 21 defines `$T_{\text{sync}}$` as total non-overlapped synchronization cost per step.

Explanation: `$T_{\text{sync}}$` is in the table but absent from the equation. It appears to be the effective communication overhead after overlap, but that relationship is not stated. Without a bound, the equation also permits `T_{\text{overlap}} > T_{\text{comm}}(N)`, which would make synchronization contribute negative time.

Proposed correction: Define the relationship explicitly, for example:

```tex
T_{\text{sync}}(N) = \max(0, T_{\text{comm}}(N) - T_{\text{overlap}}(N))
T_{\text{step}}(N) = \frac{T_{\text{compute}}}{N} + T_{\text{sync}}(N)
```

### 11. Low: `$\alpha$` collision table puts learning rate in the wrong column

- `_notation_distributed.qmd` line 31 says `$\alpha$` is network latency, not learning rate.
- `_notation_distributed.qmd` line 85 lists ML Meaning as blank/dash and Distributed Meaning as `Network Latency/Learning rate`.

Explanation: Learning rate is not a distributed-systems meaning for `$\alpha$`; it is an ML/optimization convention in some literature. The table column placement conflicts with the note on line 31.

Proposed correction: Change the row to:

| Symbol | ML Meaning | Distributed Meaning | Our Convention |
|:--|:--|:--|:--|
| `$\alpha$` | Learning rate in some optimization literature | Network latency | Network latency in the `$\alpha$`-`$\beta$` model; context disambiguates. |

### 12. Low: Energy comparison directly compares different per-event units

- `_notation_body.qmd` line 72 says `E_{\text{move}} \gg E_{\text{compute}}`.
- `_notation_body.qmd` lines 72-73 define `E_{\text{move}}` as joules/byte and `E_{\text{compute}}` as joules/FLOP.

Explanation: The total-energy equation on line 68 is dimensionally valid, but the note compares joules/byte to joules/FLOP as if the denominator were the same kind of event. The intended point is valid, but the notation table should phrase it as a per-byte versus per-operation cost comparison.

Proposed correction: Replace the note with: "For typical hardware, moving one byte often costs more energy than performing one floating-point operation, so data movement can dominate total energy."

### 13. Low: Decimal byte-prefix wording conflicts with strict SI casing

- `_notation_body.qmd` line 93 states `KB = 10^3 bytes` while saying the book uses "decimal SI prefixes only."

Explanation: Strict SI-style casing would use `kB`, not `KB`, for kilobytes. The book may intentionally prefer common computing notation, but the wording should not imply strict SI symbol casing if it uses `KB`.

Proposed correction: Either change `KB` to `kB`, or revise the prose to say the book uses decimal byte prefixes with conventional computing spellings: `KB = 10^3 bytes, MB = 10^6 bytes, ...`.

### 14. Low: Duplicate prose at end of shared notation body

- `_notation_body.qmd` lines 115-116 end with "the vast ML literature." followed by a duplicate fragment: "ML literature."

Explanation: This is a prose consistency issue in the notation page.

Proposed correction: Delete `_notation_body.qmd` line 116.

## Checked but no issue

- `_notation_body.qmd` line 18: The iron-law equation is dimensionally consistent: bytes divided by bytes/s gives seconds, FLOPs divided by FLOP/s gives seconds, and latency is seconds.
- `_notation_body.qmd` line 68: The total-energy equation is dimensionally consistent: bytes times joules/byte plus FLOPs times joules/FLOP gives joules.
- `_notation_body.qmd` lines 97-101: Precision byte counts are correct for FP32, FP16, BF16, FP8, and INT8 as stated.
- `_notation_distributed.qmd` lines 27 and 34: In the `\alpha`-`\beta` model, `T(n)=\alpha+n/\beta` and `n^*=\alpha\beta` are dimensionally consistent when `\alpha` is seconds and `\beta` is bytes/s.
- `_notation_distributed.qmd` line 38: Scaling efficiency `\eta_{\text{scaling}}=T_1/(N T_N)` is dimensionless and has ideal value 1 under linear speedup.
- `_notation_distributed.qmd` line 69: `N_{\text{total}} = d \times p \times t` is dimensionally consistent when all three parallelism degrees are integer counts.
