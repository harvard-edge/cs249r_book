# Float Exposition Audit — `robust_ai.qmd` (vol2)

Audited: 2026-06-09
Standard: FLOAT_EXPOSITION_STANDARD.md
Method: scan_floats.py bundle + ±40-line body-prose reads

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Figure | 🟠 high | 22 | 15 | 6 | 1 |
| Table | 🟠 high | 5 | 3 | 2 | 0 |
| Listing | 🟡 medium | 1 | 1 | 0 | 0 |
| **Total** | | **28** | **19** | **8** | **1** |

---

## Findings (⚠️ and 🛑 only)

---

### `fig-sdc-robust` (figure 🟠) — def L124

**Ref sentence (L120):**
> In another case [@dixit2021silent], Facebook encountered a silent data corruption (SDC) issue in its distributed querying infrastructure (@fig-sdc-robust).

**Missing move + where the takeaway currently lives:**
The lead-in paragraph walks through the Facebook failure mechanism in detail (the file-size fault, decompression failures, sporadicity). The figure is introduced inline as the cited infrastructure diagram. However, no sentence after the float delivers the takeaway from the figure: what the reader should conclude from seeing this diagram. The payoff paragraph (L243) is separated by 120+ lines of intervening content (a footnote definition, another figure, and additional prose) and discusses ML-specific SDC consequences — it is not a lead-out for this figure. The immediate post-figure prose is a footnote (L122). The takeaway (that SDC in this infrastructure illustrates how silent faults at the data layer propagate undetected through entire query pipelines, motivating ML-specific detection layers) lives in the caption and footnote, not in body prose adjacent to the figure.

**Grade: ⚠️ Partial** — cited with good setup, but no body-prose interpret move immediately following the float; caption carries the conclusion the prose should state.

**Rule-compliant diff rewrite** — add one sentence of lead-out immediately before or after the figure block (after L120 paragraph, before L122 footnote):

```diff
- In another case [@dixit2021silent], Facebook encountered a silent data
- corruption (SDC) issue in its distributed querying infrastructure
- (@fig-sdc-robust). SDC refers to undetected errors ... making diagnosis
- particularly difficult.
+ In another case [@dixit2021silent], Facebook encountered a silent data
+ corruption (SDC) issue in its distributed querying infrastructure
+ (@fig-sdc-robust). SDC refers to undetected errors ... making diagnosis
+ particularly difficult. The diagram shows how a single hardware fault in
+ one processing stage cascades into missing output rows that no health
+ check detects, the same propagation pattern that makes SDC dangerous in
+ distributed ML training.
```

---

### `fig-shift-types` (figure 🟠) — def L802

**Ref sentence (L806):**
> @Fig-shift-types categorizes the three fundamental types of distribution shift.

**Missing move + where the takeaway currently lives:**
The figure appears *before* the citation (L806): the float is at L802–804 and the prose citing it is at L806. This is a forward-reference violation (float appears before the text that introduces it). Beyond the ordering issue, the cite sentence is a bare pointer ("categorizes the three fundamental types"), with the interpret move absent. The practical consequence — that diagnosing *which* type of shift is present determines which response is appropriate (re-weighting vs. retraining) — lives only in the caption. The post-ref sentence pivots immediately to examples of organic shifts without drawing the conceptual lesson.

**Grade: ⚠️ Partial** — orphan-ordering problem (float precedes its prose) and the interpret move is caption-only; prose points but does not conclude.

**Rule-compliant diff rewrite** — move the figure to after L806, and add a lead-out:

```diff
- @Fig-shift-types categorizes the three fundamental types of distribution
- shift. These shifts occur naturally as environments evolve. User
- preferences change seasonally...
+ Three fundamental types of distribution shift arise in production. The
+ distinction matters for defense selection: covariate shift ($p(x)$
+ changes) can often be corrected by importance re-weighting without
+ retraining, while concept drift ($p(y \mid x)$ changes) requires a new
+ model because the correct answer for a given input has changed
+ (@Fig-shift-types). These shifts occur naturally as environments evolve.
+ User preferences change seasonally...
```

---

### `fig-boundary-shift` (figure 🟠) — def L812

**Ref sentence (L810 and L816):**
> These environmental changes effectively shift data points relative to the learned decision boundary (@fig-boundary-shift), causing misclassification without any change to the model itself.

> @Fig-boundary-shift illustrates the case where input distributions move while the true mapping $p(y \mid x)$ stays fixed.

**Missing move + where the takeaway currently lives:**
The float has two citation sites. The L810 cite is a mid-sentence parenthetical with a good lead-in about covariate shift magnitudes. The L816 cite is a bare pointer ("illustrates the case where…") used as a pivot to introduce concept drift, which then immediately enters a callout definition. No sentence delivers the figure's lesson in full: that the geometric consequence of covariate shift is that a well-trained decision boundary becomes wrong for the shifted population without any change to the model weights or the classification task. The insight lives implicitly in the two cite sentences together, but neither sentence stands as an interpret move on its own.

**Grade: ⚠️ Partial** — cited twice with adequate lead-in at L810; the L816 cite is a bare pivot-away pointer and neither cite delivers a complete interpret sentence.

**Rule-compliant diff rewrite** — strengthen the L816 cite into the interpret move:

```diff
- @Fig-boundary-shift illustrates the case where input distributions move
- while the true mapping $p(y \mid x)$ stays fixed. A more insidious
- variant occurs when the mapping itself changes...
+ @Fig-boundary-shift makes the consequence geometric: the decision
+ boundary the model learned from training data remains fixed in feature
+ space, but the live inputs now cluster on the wrong side of it, so
+ accuracy degrades even though neither the model nor the task has
+ changed. A more insidious variant occurs when the mapping itself
+ changes...
```

---

### `fig-distribution-shift-detector` (figure 🟠) — def L934

**Ref sentence (L932):**
> As @fig-distribution-shift-detector shows, monitoring PSI over time reveals when a model drifts from stable (Green Zone) into warning (Orange) and critical (Red Zone) regions, triggering automated retraining.

**Missing move + where the takeaway currently lives:**
The cite sentence is a float-announcer: it narrates what the figure shows ("monitoring PSI over time reveals…") rather than delivering the insight the figure makes concrete. The key lesson is that PSI monitoring provides a leading indicator that separates routine noise from drift requiring action, and that automated retraining can restore model stability — and the figure gives specific week numbers that quantify the detection-to-recovery cycle. That quantified lesson (drift detectable at week 25, critical at week 32, retraining restored stability at week 41) lives in the caption, not in the prose. The sentence as written is a re-statement of the caption in abbreviated form.

**Grade: ⚠️ Partial** — cited, but the prose sentence is a float-announcer that delegates the takeaway to the figure rather than delivering it.

**Rule-compliant diff rewrite:**

```diff
- As @fig-distribution-shift-detector shows, monitoring PSI over time
- reveals when a model drifts from stable (Green Zone) into warning
- (Orange) and critical (Red Zone) regions, triggering automated
- retraining.
+ PSI monitoring provides a leading indicator: the threshold bands turn
+ drift from a continuous statistical quantity into a discrete operational
+ signal, separating noise (Green Zone) from investigation-worthy shifts
+ (Orange) and retraining triggers (Red Zone). In the example in
+ @fig-distribution-shift-detector, automated retraining restores
+ stability sixteen weeks after the first drift signal, illustrating both
+ the detection lag and the recovery time that operators must budget for.
```

---

### `fig-adversarial-googlenet` (figure 🟠) — def L1357

**Ref sentence (L1355):**
> The reason this defense budget matters is that adversarial attacks extend far beyond simple misclassification (@fig-adversarial-googlenet).

**Missing move + where the takeaway currently lives:**
The cite is embedded in a sentence about defense budgets; it is a parenthetical pointer and the float appears immediately after. No lead-out follows the figure before the next paragraph (L1363) pivots to physical stop-sign attacks. The lesson the figure demonstrates — that perturbations invisible to human eyes produce high-confidence misclassification in a well-known network, confirming the mismatch between human and machine feature sensitivity — lives in the caption. The payoff paragraph discusses systemic risks across domains but does not articulate what the specific panda-vs-gibbon example shows.

**Grade: ⚠️ Partial** — parenthetical pointer only; the figure's specific demonstration (human-imperceptible noise, high-confidence misclassification, GoogLeNet) is never interpreted in body prose.

**Rule-compliant diff rewrite** — add a lead-out sentence between the figure and the next paragraph:

```diff
  ::: {#fig-adversarial-googlenet ...}
  ...
  :::

+ The GoogLeNet example makes the gap concrete: the perturbation that
+ flips a panda to a gibbon at 99.3 percent confidence is invisible to a
+ human but exploits the geometry of the model's decision boundary in
+ pixel space, confirming that high accuracy on a clean benchmark is not
+ evidence of robustness.

  The physical sticker attack on stop signs...
```

---

### `fig-graffiti` (figure 🟠) — def L1365

**Ref sentence (L1363):**
> The implication for autonomous vehicles is direct: stickers deployed on actual roads could cause a self-driving car to misread a stop sign as a speed limit, leading to rolling stops or unintended acceleration into intersections (@fig-graffiti).

**Missing move + where the takeaway currently lives:**
The cite is parenthetical; the sentence before it delivers the safety implication well. However, there is no lead-out after the figure: the post-figure paragraph (L1371) pivots to cascading systemic risks in healthcare and finance without interpreting what the graffiti photograph adds beyond the already-stated stop-sign misclassification claim. The figure is a photographic illustration of the attack category, and the prose around it frames the consequence but not why *this visual* is the right exhibit — i.e., that physical-world attacks are legible to humans (overt), yet the classification failure is the same as with inconspicuous stickers, which proves the attack class does not depend on imperceptibility.

**Grade: ⚠️ Partial** — the implication is stated in the lead-in, but no body prose interprets what the figure specifically demonstrates; the conceptual point (overt-vs.-covert attack equivalence) lives only in the caption.

**Rule-compliant diff rewrite** — add one sentence of lead-out after the figure:

```diff
  ::: {#fig-graffiti ...}
  ...
  :::

+ The high-visibility variant shown here is pedagogically useful because
+ it confirms that classification failures do not require imperceptible
+ perturbations: the same misclassification rate occurs whether the
+ modification is conspicuous graffiti or a small printed sticker, which
+ means defenses cannot rely on imperceptibility as a detection signal.

  Beyond performance degradation...
```

---

### `fig-distribution-shift-example` (figure 🟠) — def L1431

**Ref sentence (L1429):**
> The most direct mechanism is label modification: an attacker selects a subset of training samples and alters their labels, flipping $y = 1$ to $y = 0$ or reassigning categories in multi-class settings (@fig-distribution-shift-example).

**Missing move + where the takeaway currently lives:**
The cite is a parenthetical in the middle of a sentence about label modification. The payoff paragraph (L1675) is separated from the float by over 240 lines of intervening content (the entire TikZ block definition, multiple sections, and additional prose). There is no lead-out after the figure; the next body prose (L1675) discusses feature corruption rather than interpreting the diagram. The figure's lesson — that even small-scale label flipping shifts the learned decision boundary in a measurable, visually demonstrable way — lives only in the caption.

**Grade: 🛑 Fails** — parenthetical cite only; no interpret move anywhere near the float; the takeaway lives in the caption and the payoff paragraph is separated by 240+ lines of unrelated content.

**Rule-compliant diff rewrite** — add a brief interpret sentence immediately before or after the float:

```diff
  The most direct mechanism is label modification: an attacker selects a
  subset of training samples and alters their labels, flipping $y = 1$ to
  $y = 0$ or reassigning categories in multi-class settings
  (@fig-distribution-shift-example). Even small-scale label corruption can
  shift decision boundaries significantly.
+ The diagram illustrates why: each flipped label relocates training
+ pressure to the wrong side of the boundary, and a handful of such
+ examples in a dense region can move the boundary enough to flip
+ production predictions on legitimate inputs.
```

---

### `tbl-kl-divergence-thresholds` (table 🟠) — def L1074

**Ref sentence (L1066):**
> For drift monitoring in production, @tbl-kl-divergence-thresholds gives practical thresholds for interpreting KL divergence values.

**Missing move + where the takeaway currently lives:**
The cite is a bare pointer: "gives practical thresholds" is a label-solution pair that summarizes what the table contains rather than what it means. The interpret move — which band the engineer should actually act on, and why the cutoffs are set where they are — lives in the caption (which explains the three-tier threshold) and is never stated in body prose. The post-table sentence (L1076) pivots immediately to KDE computation cost, not to the decision implications of the threshold bands.

**Grade: ⚠️ Partial** — cited, but the cite sentence is a pure label-pointer; the decision implication of the threshold bands lives only in the caption.

**Rule-compliant diff rewrite:**

```diff
- For drift monitoring in production, @tbl-kl-divergence-thresholds gives
- practical thresholds for interpreting KL divergence values.
+ In production, the key operational cut is $\mathcal{D}_{\text{KL}} =
+ 0.1$: values below 0.05 represent negligible distributional change, the
+ 0.05–0.1 band warrants investigation before acting, and 0.1 or above
+ typically triggers a retraining workflow (@tbl-kl-divergence-thresholds).
+ The bounded Jensen-Shannon divergence is often preferable for automation
+ because it makes threshold calibration consistent across features.
```

---

### `tbl-psi-country-example` (table 🟠) — def L1122

**Ref sentence (L1112):**
> @Tbl-psi-country-example compares the baseline (training) distribution and current (production) distribution for the top 5 countries.

**Missing move + where the takeaway currently lives:**
The cite sentence is a pure table-announcer: "compares the baseline… and current… distribution" is a description of table content, not a takeaway. The post-table payoff (L1128) gives the verdict ("PSI of 0.065 indicates negligible drift… No action required"), which is the interpret move, but it is separated from the cite sentence by the entire table body and a computed PSI equation. The structure works if the reader is expected to trace through the table themselves, but the standard requires that the body prose state the conclusion: the "Other" bucket drives most of the divergence, and the overall PSI still falls below the action threshold.

**Grade: ⚠️ Partial** — the interpret move exists at L1128 but the cite sentence is a float-announcer; the lead-in needs to frame the key tension (is the "Other" bucket growth enough to trigger action?) rather than just describing the table columns.

**Rule-compliant diff rewrite:**

```diff
- @Tbl-psi-country-example compares the baseline (training) distribution
- and current (production) distribution for the top 5 countries.
+ The country-feature PSI decomposition in @Tbl-psi-country-example
+ surfaces a pattern common in global fraud models: most of the
+ divergence (0.0470 of 0.065 total PSI) comes from growth in the
+ residual "Other" bucket, not from changes in the major markets, so
+ the model's training coverage of emerging-market traffic has become the
+ weak point.
```

---
