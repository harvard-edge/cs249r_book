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

## Final Ordered Read-Through Pass

Worktree: `/Users/VJ/GitHub/MLSysBook-progressive-disclosure-final-audit`

Branch: `codex/progressive-disclosure-final-audit`

Base: local `dev` at merge commit `a83785d51c`

Read-only agents audited Volume 1 and Volume 2 in chapter order after the algorithm pass. The editor pass accepted findings that repaired an actual prerequisite gap, an undefined acronym, or a forward reference phrased as prior knowledge. It declined findings that would have removed concrete A100, H100, NVLink, HBM, or Tensor Core anchors solely because the hardware chapter appears later; those anchors are retained when the surrounding prose uses them as engineering scale examples rather than requiring full hardware-architecture knowledge.

### Accepted Edits

- `vol1/ml_systems.qmd`: Added local role language before using quantization and distributed-training strategy vocabulary.
- `vol1/nn_computation.qmd`: Replaced early algorithm text that named GPU occupancy and HBM capacity with accelerator-level memory/occupancy language.
- `vol1/data_selection.qmd`: Defined INT4 locally as a four-bit integer representation in the data-selection/compression bridge.
- `vol1/model_compression.qmd`: Added local definitions for observers, granularity, and zero-point before the PTQ calibration algorithm.
- `vol1/benchmarking.qmd`: Rephrased the responsible-engineering pointer as a future same-volume formalization rather than a prerequisite.
- `vol2/introduction.qmd`: Mapped Horovod, NCCL, Tensor Fusion, Megatron-LM, and ZeRO to their systems roles on first use.
- `vol2/compute_infrastructure.qmd`: Expanded ZeRO as Zero Redundancy Optimizer at the HBM/DDR boundary discussion.
- `vol2/network_fabrics.qmd`: Made the alpha-beta model local to the network chapter and treated the collective-communication section as later reuse; defined rank locally in the rail-topology discussion.
- `vol2/distributed_training.qmd`: Defined process rank at first use in dataset splitting.
- `vol2/inference.qmd`: Replaced undefined OBS acronym with a local description of the naive full inverse-Hessian baseline.
- `vol2/edge_intelligence.qmd`: Recast MLOps observability as a centralized-monitoring baseline rather than a future prerequisite.
- `vol2/ops_scale.qmd`: Marked fairness metrics as executable release thresholds here, with the responsible-AI chapter as the later social/statistical treatment.
- `vol2/sustainable_ai.qmd`: Rephrased the power-delivery reference as a forward bridge instead of "as discussed."

### Deliberately Kept

- Concrete A100/H100/V100/NVLink/HBM examples in early Volume 1 were kept when they function as scale anchors or engineering numbers. The progressive-disclosure rule here is sequencing and local bridging, not removing hardware specificity.
- Algorithm references from the prior pass were left in bracketed Quarto theorem-reference form (`[algorithm @alg-id]` mid-sentence, `[Algorithm @alg-id]` at sentence start). A later cross-reference pass will verify rendered casing for sections, figures, tables, listings, and algorithms together.
