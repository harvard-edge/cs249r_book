# MLSysBook Release Draft: vol1-v0.7.0 + vol2-v0.2.0

Status: draft for the next live book publish.

This draft is written for the next combined book release after `vol1-v0.6.2+vol2-v0.1.2`. It assumes the publish workflow is run with `deploy_target=all` and `release_type=minor`, which computes `vol1-v0.7.0+vol2-v0.2.0` from the latest per-volume tags.

## Version Rationale

### Volume I: `vol1-v0.7.0`

Volume I should move from the `0.6.x` track to `0.7.0` because this release is more than a patch. It represents a full release-quality audit of the textbook: prose flow, progressive disclosure, object integration, citation hygiene, acronym first use, layout quality, margin figures, and source-to-artifact validation all moved together. The content is still pre-1.0, but the release is a meaningful reader-facing maturation checkpoint.

Use `0.7.0` rather than `0.6.4` because the most important change is not an isolated bug fix. It is a coordinated polish and validation pass that makes the volume feel more like a publishable textbook.

### Volume II: `vol2-v0.2.0`

Volume II should move from the `0.1.x` track to `0.2.0` for the same reason, scaled to its own maturity. The volume remains earlier than Volume I, so it should not mirror the absolute `0.7.0` number. But it did receive a full pass over prose flow, figures, tables, callouts, margin figures, LEGO-derived quantitative material, citations, and release validation. That is a minor release in a pre-1.0 track.

Use `0.2.0` rather than `0.1.4` because this is the first release after Volume II's initial public release line that materially improves whole-volume readability and release readiness.

### Why Not `1.0.0`

Both volumes should remain pre-1.0. The release is a strong publish-ready checkpoint, but `1.0.0` should be reserved for a deliberate public/final milestone where the citation story, release artifacts, and publication posture are all intentionally frozen.

## Proposed Release Title

`vol1-v0.7.0+vol2-v0.2.0: Release-quality audit for the AI engineering textbook`

## Release Notes

This release is a coordinated quality pass across both MLSysBook volumes. It focuses on making the textbook easier to read, easier to teach from, and safer to publish: cleaner prose flow, stronger progressive disclosure, better-integrated figures and tables, more consistent callouts, clearer quantitative examples, and a more reliable release gate.

### Reader-Facing Improvements

- Strengthened prose flow across both volumes by folding stranded one-sentence paragraphs into their surrounding context where they did not deserve to stand alone.
- Reduced over-listing by converting mechanical bullet sequences into prose or tables when that better served student understanding.
- Tightened progressive disclosure so terms, acronyms, principles, and cross-references are introduced in the right local context.
- Improved citation integration so sources are woven into the prose rather than appearing as detached reference lines.
- Cleaned emphasis and heading style so typographic emphasis carries meaning rather than acting as decoration.

### Figures, Tables, Callouts, and Margin Figures

- Improved object integration across figures, tables, callouts, lists, equations, algorithms, and margin figures.
- Checked that figures and tables are introduced by the surrounding prose and that captions describe what readers should learn from them.
- Fixed table-caption placement and cross-reference behavior so table references resolve and render consistently.
- Tuned margin figure placement and geometry where figures were detached from the prose they support or visually cramped.
- Polished several SVG assets after visual inspection so labels, keys, line endpoints, and colors read cleanly in the PDF.

### Quantitative Examples and LEGO-Derived Material

- Added and exercised Binder checks for LEGO prose/unit contracts, equation consistency, rendered-output hygiene, and typed suffix conventions.
- Improved alignment between computed values and prose claims, especially where quantitative examples support a systems trade-off.
- Clarified unit and abbreviation conventions so first use and later reuse are easier for readers to follow.

### Layout and Artifact Quality

- Ran a release-oriented PDF layout pass over both volumes, with special attention to margin notes, overflowing boxes, figure sizing, and object placement.
- Reduced avoidable whitespace caused by nonbreaking objects and stale manual layout interventions.
- Verified the release gate locally after final layout edits and confirmed the pushed `dev` branch passed GitHub validation.

### Release Infrastructure

- Added `binder release` as the high-level pre-release gate for book validation.
- Standardized release-stage output so checks report structured, actionable results.
- Included source checks, object-flow checks, LEGO checks, math checks, PDF artifact checks, and final validation in the release gate.
- Confirmed the pushed `dev` branch passed Codespell, CodeQL, `Book Validate (Dev)`, and the follow-up preview workflow.

## Volume-Specific Notes

### Volume I

Volume I now reads more consistently as a foundation for AI engineering. The pass strengthens the progression from systems thinking and the Data-Algorithm-Machine framing into workflows, neural computation, training, optimization, deployment, and responsible engineering. Part principles are easier to connect back to chapter summaries and quantitative examples.

### Volume II

Volume II now reads more consistently as the scale-out companion. The pass strengthens the arc from fleet infrastructure through distributed training, inference, operations, edge intelligence, security, robustness, sustainability, and responsible fleet governance. Several margin figures and object placements were adjusted so fleet-scale concepts are anchored closer to the prose that explains them.

## Validation

- Local release gate passed after the final layout adjustments.
- `dev` was pushed and GitHub Actions completed successfully.
- The workflow-added contributors update committed only `site/about/contributors.json` with `[skip ci]`; its follow-up preview and CodeQL checks also passed.

## Suggested Publish Inputs

- `deploy_target`: `all`
- `release_type`: `minor`
- `description`: `Release-quality audit for the AI engineering textbook`
- Expected tag summary: `vol1-v0.7.0+vol2-v0.2.0`
