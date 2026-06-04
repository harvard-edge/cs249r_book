# Progressive Disclosure Concept-Map Audit

Worktree: `/Users/VJ/GitHub/MLSysBook-progressive-concept-map-audit`

Branch: `codex/progressive-concept-map-audit`

Base: local `dev` at merge commit `07acdf9603`

## Purpose

This pass checks Volume 1 and Volume 2 as separate learning sequences. Volume 1 starts from the assumed student background in `.claude/rules/book-prose.md`. Volume 2 may assume Volume 1 knowledge and appendices, but it should not assume concepts introduced later in Volume 2.

The concept maps are the starting ledgers for prior knowledge. They are not treated as authoritative: each finding must be verified against chapter prose.

Roadmap or preview mentions in an introduction concept map do not automatically count as taught concepts. A later chapter may rely on a prior concept only when the earlier prose defined it, used it in an explanatory way, or gave students enough local context to carry the term forward.

## Technical Register

Progressive disclosure means sequencing and local bridging, not simplification. The audit should preserve concrete systems anchors when they teach scale or constraints, including named processors, accelerators, interconnects, models, and metrics such as NVIDIA A100, H100, DGX H100, V100, NVLink, InfiniBand, MFU, FLOP/s, bandwidth, latency, and power.

## Checklist

- [x] Create a new worktree from updated local `dev`.
- [x] Validate concept-map structure with `book/tools/scripts/audit/validate_concept_maps.py`.
- [x] Derive chapter order from the Volume 1 and Volume 2 Quarto manifests.
- [x] Launch read-only parallel agents over ordered chapter slices.
- [x] Collect YAML findings from all agents.
- [x] Triage findings against actual prose and concept maps.
- [x] Apply only minimal bridge/map fixes that improve progressive disclosure without lowering technical specificity.
- [x] Run pre-commit on changed files.
- [ ] Commit as one task-level progressive-disclosure concept-map pass.

## Agent Assignments

- Volume 1 chapters 1-8: `019e9373-53c0-7a73-9399-d6995bf1d6c4`
- Volume 1 chapters 9-16: `019e9373-540d-7f30-b274-26f144690bd9`
- Volume 2 chapters 1-8: `019e9373-5470-72e2-af1a-8c7727797e7f`
- Volume 2 chapters 9-17: `019e9373-548d-7021-8fb0-ad65da17dcca`

## Findings Resolved

- `vol1/data_engineering.qmd`: Retargeted the energy-movement invariant from future hardware/compression chapters to the already introduced iron law and memory-wall discussion, while keeping later optimization chapters as an extension rather than a prerequisite.
- `vol1/nn_architectures.qmd`: Reframed architecture selection as building on local architectural analysis plus prior deployment/lifecycle constraints; future data-selection and operations chapters are now extensions, not required prior knowledge.
- `vol1/introduction_concepts.yml`: Removed `Federated learning` from the chapter-1 technical-term ledger because the introduction does not teach it.
- `vol1/ml_workflow_concepts.yml`: Replaced the premature `MLOps` technical-term entry with `Operational practices` and `Workflow automation`.
- `vol1/ml_ops.qmd`: Retargeted verification/degradation references to the introduction equations and removed forward dependency on the conclusion's invariant synthesis.
- `vol1/ml_ops.qmd`: Defined data drift locally as the input-distribution subtype and retargeted the broader taxonomy to the earlier data-engineering chapter instead of the following responsible-engineering chapter.
- `vol1/responsible_engr_concepts.yml`: Added `Bias Feedback Invariant and subgroup monitoring` so the map records the named invariant activated in that chapter.
- `vol2/ops_scale.qmd`: Replaced a forward reference to Sustainable AI's rack-power table with a local rack-density sanity check.
- `vol2/robust_ai.qmd`: Reframed robustness overhead as local energy, heat, and capacity trade-offs, with sustainability accounting marked as a next-chapter formalization rather than assumed prior knowledge.
