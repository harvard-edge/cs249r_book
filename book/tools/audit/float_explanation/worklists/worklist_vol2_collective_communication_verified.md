# Verified findings — collective_communication.qmd (vol2)
Prior findings: 1 | Survived: 0 | Refuted: 1

## SURVIVING findings

*(none)*

## REFUTED findings

- `fig-fleet-stack` — REFUTED: legitimate cross-chapter reference to the book's organizing framework; no local re-explanation required.

  The figure is defined in vol2/introduction and functions as the organizing framework referenced across eight chapters. Per the audit brief, a cross-chapter reference need not re-explain a float defined elsewhere. The sentence at L50 — "In the fleet stack shown in @fig-fleet-stack, communication algorithms sit squarely in the Distribution Layer" — is not a dead-end: it is a positioning statement that locates this chapter's subject within a framework the reader already knows from the introduction. The sentence immediately following (L52) extends the argument by referencing parallelism strategies from @sec-distributed-training-systems and their shared assumption, grounding the "Distribution Layer" claim in concrete distributed-training mechanics. The local prose does not leave the reader stranded; it moves directly from the positional claim into a full argument about the asymmetry between computation and communication scaling (L54). The first pass's concern that "the three-layer model is never explained in surrounding prose" is correct as a factual observation but does not meet the survival threshold: the framework is introduced and defined once, in the introduction, and all subsequent chapters are entitled to use it by reference. The default is REFUTED when uncertain, and the cross-chapter-ref exception explicitly applies here.
