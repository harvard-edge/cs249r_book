# Section Flow Audit

Worktree: `/Users/VJ/GitHub/MLSysBook-section-flow-audit`

Branch: `codex/section-flow-audit`

Date: 2026-06-04

## Purpose

This pass checks whether short subsections, one-paragraph headings, and nearby paragraph clusters read as a textbook narrative rather than as local notes. The criterion is paragraph-to-paragraph motion: the previous paragraph should set up the current paragraph, the current paragraph should advance the argument, and the next paragraph should follow from it.

## Scan Inputs

- Ran the layout-proxy scan across both volumes. The scan reported 177 layout proxies, mostly one-paragraph headings, with the largest clusters in `vol2/ops_scale.qmd`, `vol1/benchmarking.qmd`, `vol1/model_serving.qmd`, `vol1/ml_ops.qmd`, and `vol1/frameworks.qmd`.
- Launched four read-only chapter-slice agents covering early Volume 1, later Volume 1, early/mid Volume 2, and later Volume 2.
- Treated bullets, tables, worked examples, fallacies/pitfalls, and callouts as valid structures when they teach sequence, contrast, lookup, or diagnosis more clearly than prose.

## Accepted Edits

- `vol1/ml_systems.qmd`: Split an overstuffed Edge ML paragraph into a deployment-energy paragraph followed by a model-size/locality paragraph, preserving the quantitative edge/cloud energy gap and compression connection.
- `vol1/ml_workflow.qmd`: Split the distributed-clinic data paragraph into a heterogeneity setup and an infrastructure/point-of-capture validation response.
- `vol1/data_engineering.qmd`: Fused a small assertion paragraph into the CI/CD integration explanation so validation-as-code, deployment blocking, and expectation-suite versioning read as one operational pattern.
- `vol1/nn_architectures.qmd`: Added the missing representation-to-systems bridge for recommendation embeddings, connecting raw ID geometry to random embedding-table reads.
- `vol2/network_fabrics.qmd`: Added a causal bridge after the five-level fabric model so the levels read as an ML traffic chain rather than an inventory.
- `vol2/fleet_orchestration.qmd`: Removed duplicate tragedy-of-the-commons setup and made quotas the scheduler-enforced response.
- `vol2/ops_scale.qmd`: Collapsed an unreferenced freshness-monitoring subheading into the feature-freshness narrative and tied staleness measurement directly to the table SLOs.

## Deferred

- Cross-reference rendering and casing, including `[Algorithm @alg-id]` and `[algorithm @alg-id]`, will be handled in the next dedicated pass.
- Larger chapter-level audits for engineering-book feel, index placement, notation, LEGO output naming, emphasis, references, and algorithm-conversion remain separate task branches so this commit stays limited to section flow.
