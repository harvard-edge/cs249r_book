# Verified findings — robust_ai.qmd (vol2)
Prior findings: 1 | Survived: 0 | Refuted: 1

## SURVIVING findings

(none)

## REFUTED findings

- `fig-adversarial-googlenet` — REFUTED: explanation in caption (L1357) and payoff paragraph (L1363).

  Caption: "Subtle, intentionally crafted noise added to an image can cause a trained deep neural network (GoogLeNet) to misclassify it, even though the perturbed image remains visually indistinguishable to humans. This vulnerability underscores the lack of robustness in many machine learning models and motivates research into adversarial training and defense mechanisms."

  The caption explains precisely what the figure shows (imperceptible noise flips GoogLeNet's classification) and why it matters (robustness gap, motivates defenses). The prior worklist flagged a "mismatched claim" in the ref sentence, which is a rhetorical issue rather than an absence of explanation: the ref sentence appends the figure as a parenthetical supporting a broader systemic-risk claim, but the figure's own content and implication are fully explained in the caption. The payoff paragraph at L1363 extends the lesson to physical attacks on stop signs, reinforcing the real-world stakes. Per the refutation bar, the caption's clear delivery of what the figure shows and why it matters is sufficient. No true dead-end exists.
