Figures-first chapter review — Volumes I–IV

This is a local editorial planning page requested for a fresh figure-needs pass.
It does not create manuscript figures, enroll animations, change captions, or
publish anything. The sequence is static coverage, selective animation design,
then implementation and learner evaluation. An existing adequate figure can
enter animation design without waiting for unrelated figure proposals.

Inputs:
- vol1.json through vol4.json: independent source/context chapter reviews.
- ../animation-catalog/inventory.json: existing named body-figure inventory.
- Canonical PDF chapter lists and current source files: coverage and anchors.
- Existing figure registry: registration status only, not a quality endorsement.

Build:
  python3 books/_prototypes/vol1-web-animations/figure-plan/build-review.py
Serve books locally, then open:
  /_prototypes/vol1-web-animations/figure-plan/index.html

Scope:
All 71 numbered chapters (16/17/20/18). Each review considers chapter outline,
all body-figure captions/anchors, and targeted prose around proposed gaps.
Relevant existing margin and cross-volume coverage is considered to avoid
false missing-figure claims. This is not an exhaustive sentence-level audit
or visual approval of the hundreds of unchanged illustrations. Supplementary
frontmatter/part/appendix figures are inventoried separately; their visual needs
are not audited here. No actual students participated in this planning pass.

Acceptance:
A new figure must explain a relationship or mechanism the local argument needs,
with an explicit placement and a complete static form. A later animation must
reveal change, dependencies, resource retention, or a useful parameter effect.
Taxonomies, landscapes, and static maps do not earn players by being present.
Expanded labs are optional; the inline explanation remains useful on its own.
The audit's proposals require author review before figure production. No count
is a quota. Source revision and chapter hashes record the reviewed snapshot;
a changed chapter requires revisiting its recommendations.
