# Content Structure

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate: python3 bindery/tools/scripts/structure/gen_structure.py -->

Chapter order is derived from each volume's Quarto PDF config
(`books/config/_quarto-pdf-<vol>.yml`), which is the canonical
reading order. This file is the human- and agent-readable view of that order.

## Shared skeleton

Every volume uses the same layout:

```
contents/<vol>/
├── index.qmd            volume home
├── README.md            volume readme
├── frontmatter/         *.qmd
├── parts/               <name>_principles.qmd + summaries.yml
├── <chapter>/           <chapter>.qmd + images/
└── backmatter/          references.qmd, appendix_*.qmd, glossary/
```

The load-bearing invariant is `<chapter>/<chapter>.qmd`: a chapter directory
and its main file share a name. `bindery/cli/commands/build.py` relies on it,
and `bindery/tests/test_content_structure.py` enforces it.

## Volume I: Introduction to Machine Learning Systems (`vol1`)

16 chapters. Reading order:

| # | Kind | Path |
|---:|---|---|
|  | frontmatter | `frontmatter/foreword.qmd` |
|  | frontmatter | `frontmatter/author_note.qmd` |
|  | frontmatter | `frontmatter/about.qmd` |
|  | frontmatter | `frontmatter/acknowledgements.qmd` |
|  | frontmatter | `frontmatter/ai_use.qmd` |
|  | frontmatter | `frontmatter/notation.qmd` |
| 1 | chapter | `introduction/introduction.qmd` |
|  | part | `parts/foundations_principles.qmd` |
| 2 | chapter | `ml_systems/ml_systems.qmd` |
| 3 | chapter | `ml_workflow/ml_workflow.qmd` |
| 4 | chapter | `data_engineering/data_engineering.qmd` |
|  | part | `parts/build_principles.qmd` |
| 5 | chapter | `nn_computation/nn_computation.qmd` |
| 6 | chapter | `nn_architectures/nn_architectures.qmd` |
| 7 | chapter | `frameworks/frameworks.qmd` |
| 8 | chapter | `training/training.qmd` |
|  | part | `parts/optimize_principles.qmd` |
| 9 | chapter | `data_selection/data_selection.qmd` |
| 10 | chapter | `model_compression/model_compression.qmd` |
| 11 | chapter | `hw_acceleration/hw_acceleration.qmd` |
| 12 | chapter | `benchmarking/benchmarking.qmd` |
|  | part | `parts/deploy_principles.qmd` |
| 13 | chapter | `model_serving/model_serving.qmd` |
| 14 | chapter | `ml_ops/ml_ops.qmd` |
| 15 | chapter | `responsible_engr/responsible_engr.qmd` |
| 16 | chapter | `conclusion/conclusion.qmd` |
|  | backmatter | `backmatter/appendix_dam.qmd` |
|  | backmatter | `backmatter/appendix_data.qmd` |
|  | backmatter | `backmatter/appendix_algorithm.qmd` |
|  | backmatter | `backmatter/appendix_machine.qmd` |
|  | backmatter | `backmatter/appendix_assumptions.qmd` |
|  | backmatter | `backmatter/glossary/glossary.qmd` |
|  | backmatter | `backmatter/references.qmd` |

## Volume II: Machine Learning Systems at Scale (`vol2`)

17 chapters. Reading order:

| # | Kind | Path |
|---:|---|---|
|  | frontmatter | `frontmatter/dedication.qmd` |
|  | frontmatter | `frontmatter/author_note.qmd` |
|  | frontmatter | `frontmatter/about.qmd` |
|  | frontmatter | `frontmatter/acknowledgements.qmd` |
|  | frontmatter | `frontmatter/ai_use.qmd` |
|  | frontmatter | `frontmatter/notation.qmd` |
| 1 | chapter | `introduction/introduction.qmd` |
|  | part | `parts/fleet_principles.qmd` |
| 2 | chapter | `compute_infrastructure/compute_infrastructure.qmd` |
| 3 | chapter | `network_fabrics/network_fabrics.qmd` |
| 4 | chapter | `data_storage/data_storage.qmd` |
|  | part | `parts/distributed_ml_principles.qmd` |
| 5 | chapter | `distributed_training/distributed_training.qmd` |
| 6 | chapter | `collective_communication/collective_communication.qmd` |
| 7 | chapter | `fault_tolerance/fault_tolerance.qmd` |
| 8 | chapter | `fleet_orchestration/fleet_orchestration.qmd` |
|  | part | `parts/deployment_principles.qmd` |
| 9 | chapter | `performance_engineering/performance_engineering.qmd` |
| 10 | chapter | `inference/inference.qmd` |
| 11 | chapter | `edge_intelligence/edge_intelligence.qmd` |
| 12 | chapter | `ops_scale/ops_scale.qmd` |
|  | part | `parts/responsible_fleet_principles.qmd` |
| 13 | chapter | `security_privacy/security_privacy.qmd` |
| 14 | chapter | `robust_ai/robust_ai.qmd` |
| 15 | chapter | `sustainable_ai/sustainable_ai.qmd` |
| 16 | chapter | `responsible_ai/responsible_ai.qmd` |
| 17 | chapter | `conclusion/conclusion.qmd` |
|  | backmatter | `backmatter/references.qmd` |
|  | backmatter | `backmatter/appendix_dam.qmd` |
|  | backmatter | `backmatter/appendix_c3.qmd` |
|  | backmatter | `backmatter/appendix_fleet.qmd` |
|  | backmatter | `backmatter/appendix_communication.qmd` |
|  | backmatter | `backmatter/appendix_reliability.qmd` |
|  | backmatter | `backmatter/appendix_inference.qmd` |
|  | backmatter | `backmatter/appendix_assumptions.qmd` |
|  | backmatter | `backmatter/glossary/glossary.qmd` |

## Volume III: Agentic Machine Learning Systems (`vol3`)

15 chapters. Reading order:

| # | Kind | Path |
|---:|---|---|
|  | frontmatter | `frontmatter/author_note.qmd` |
|  | frontmatter | `frontmatter/about.qmd` |
| 1 | chapter | `introduction/introduction.qmd` |
|  | part | `parts/foundations_principles.qmd` |
| 2 | chapter | `execution_descriptors/execution_descriptors.qmd` |
| 3 | chapter | `control_topologies/control_topologies.qmd` |
|  | part | `parts/training_adaptation_principles.qmd` |
| 4 | chapter | `trajectory_fine_tuning/trajectory_fine_tuning.qmd` |
| 5 | chapter | `reinforcement_learning/reinforcement_learning.qmd` |
| 6 | chapter | `test_time_search/test_time_search.qmd` |
|  | part | `parts/serving_memory_principles.qmd` |
| 7 | chapter | `context_working_sets/context_working_sets.qmd` |
| 8 | chapter | `prefix_caching_paging/prefix_caching_paging.qmd` |
| 9 | chapter | `trajectory_scheduling/trajectory_scheduling.qmd` |
|  | part | `parts/security_isolation_principles.qmd` |
| 10 | chapter | `tool_interfaces/tool_interfaces.qmd` |
| 11 | chapter | `sandboxing_isolation/sandboxing_isolation.qmd` |
| 12 | chapter | `verification_recovery/verification_recovery.qmd` |
|  | part | `parts/scale_operations_principles.qmd` |
| 13 | chapter | `multi_agent_coordination/multi_agent_coordination.qmd` |
| 14 | chapter | `telemetry_evaluation/telemetry_evaluation.qmd` |
| 15 | chapter | `conclusion/conclusion.qmd` |
|  | backmatter | `backmatter/appendix_math.qmd` |

## Volume IV: Physical AI Systems (`vol4`)

17 chapters. Reading order:

| # | Kind | Path |
|---:|---|---|
|  | frontmatter | `frontmatter/about_author.qmd` |
|  | frontmatter | `frontmatter/prerequisites.qmd` |
|  | frontmatter | `frontmatter/syllabus.qmd` |
|  | frontmatter | `frontmatter/notation.qmd` |
|  | frontmatter | `frontmatter/author_note.qmd` |
| 1 | chapter | `boundary/boundary.qmd` |
|  | part | `parts/anatomy_principles.qmd` |
| 2 | chapter | `body/body.qmd` |
| 3 | chapter | `nervous/nervous.qmd` |
| 4 | chapter | `brain/brain.qmd` |
|  | part | `parts/teaching_principles.qmd` |
| 5 | chapter | `data/data.qmd` |
| 6 | chapter | `training/training.qmd` |
| 7 | chapter | `evaluation/evaluation.qmd` |
|  | part | `parts/running_principles.qmd` |
| 8 | chapter | `perception/perception.qmd` |
| 9 | chapter | `memory/memory.qmd` |
| 10 | chapter | `intent/intent.qmd` |
| 11 | chapter | `planning/planning.qmd` |
| 12 | chapter | `enforcement/enforcement.qmd` |
| 13 | chapter | `placement/placement.qmd` |
|  | part | `parts/governing_principles.qmd` |
| 14 | chapter | `intervention/intervention.qmd` |
| 15 | chapter | `verification/verification.qmd` |
| 16 | chapter | `release/release.qmd` |
| 17 | chapter | `frontier/frontier.qmd` |
|  | backmatter | `backmatter/references.qmd` |
|  | backmatter | `backmatter/appendix_math.qmd` |
|  | backmatter | `backmatter/appendix_hardware.qmd` |
|  | backmatter | `backmatter/appendix_heterogeneous_soc.qmd` |
|  | backmatter | `backmatter/appendix_standards.qmd` |
|  | backmatter | `backmatter/appendix_lexicon.qmd` |
|  | backmatter | `backmatter/glossary/glossary.qmd` |
