# Content Structure

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate: python3 binder/tools/scripts/structure/gen_structure.py -->

Chapter order is derived from each volume's Quarto PDF config
(`books/config/_quarto-pdf-<vol>.yml`), which is the canonical
reading order. This file is the human- and agent-readable view of that order.

## Shared skeleton

Every volume uses the same layout:

```
books/<vol>/
├── index.qmd            volume home
├── README.md            volume readme
├── frontmatter/         *.qmd
├── parts/               <name>_principles.qmd + summaries.yml
├── <chapter>/           <chapter>.qmd + images/
└── backmatter/          references.qmd, appendix_*.qmd, glossary/
```

The load-bearing invariant is `<chapter>/<chapter>.qmd`: a chapter directory
and its main file share a name. `binder/cli/commands/build.py` relies on it,
and `binder/tests/test_content_structure.py` enforces it.

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

21 chapters. Reading order:

| # | Kind | Path |
|---:|---|---|
|  | frontmatter | `frontmatter/dedication.qmd` |
|  | frontmatter | `frontmatter/author_note.qmd` |
|  | frontmatter | `frontmatter/about.qmd` |
|  | frontmatter | `frontmatter/acknowledgements.qmd` |
|  | frontmatter | `frontmatter/ai_use.qmd` |
|  | frontmatter | `frontmatter/notation.qmd` |
| 1 | chapter | `introduction/introduction.qmd` |
|  | part | `parts/foundations_principles.qmd` |
| 2 | chapter | `01_processor/01_processor.qmd` |
| 3 | chapter | `02_deliberation/02_deliberation.qmd` |
|  | part | `parts/serving_memory_principles.qmd` |
| 4 | chapter | `03_working_sets/03_working_sets.qmd` |
| 5 | chapter | `04_virtual_memory/04_virtual_memory.qmd` |
| 6 | chapter | `05_episodic_memory/05_episodic_memory.qmd` |
| 7 | chapter | `06_checkpointing/06_checkpointing.qmd` |
| 8 | chapter | `07_scheduling/07_scheduling.qmd` |
|  | part | `parts/security_isolation_principles.qmd` |
| 9 | chapter | `08_actuation/08_actuation.qmd` |
| 10 | chapter | `09_interrupts/09_interrupts.qmd` |
| 11 | chapter | `10_virtualization/10_virtualization.qmd` |
|  | part | `parts/training_adaptation_principles.qmd` |
| 12 | chapter | `11_data_flywheel/11_data_flywheel.qmd` |
| 13 | chapter | `12_sft/12_sft.qmd` |
| 14 | chapter | `13_rlvr/13_rlvr.qmd` |
|  | part | `parts/scale_operations_principles.qmd` |
| 15 | chapter | `14_multi_agent/14_multi_agent.qmd` |
| 16 | chapter | `15_observability/15_observability.qmd` |
| 17 | chapter | `16_tokenomics/16_tokenomics.qmd` |
| 18 | chapter | `conclusion/conclusion.qmd` |
| 19 | chapter | `appendices/app_a_reference_architecture.qmd` |
| 20 | chapter | `appendices/app_b_tool_design.qmd` |
| 21 | chapter | `appendices/app_c_failure_taxonomy.qmd` |
|  | backmatter | `backmatter/appendix_math.qmd` |
|  | backmatter | `backmatter/glossary/glossary.qmd` |

## Volume IV: Physical AI Systems (`vol4`)

17 chapters. Reading order:

| # | Kind | Path |
|---:|---|---|
|  | frontmatter | `frontmatter/dedication.qmd` |
|  | frontmatter | `frontmatter/author_note.qmd` |
|  | frontmatter | `frontmatter/about_author.qmd` |
|  | frontmatter | `frontmatter/prerequisites.qmd` |
|  | frontmatter | `frontmatter/syllabus.qmd` |
|  | frontmatter | `frontmatter/acknowledgements.qmd` |
|  | frontmatter | `frontmatter/ai_use.qmd` |
|  | frontmatter | `frontmatter/notation.qmd` |
| 1 | chapter | `01_boundary/01_boundary.qmd` |
|  | part | `parts/anatomy_principles.qmd` |
| 2 | chapter | `02_body/02_body.qmd` |
| 3 | chapter | `03_nervous/03_nervous.qmd` |
| 4 | chapter | `04_brain/04_brain.qmd` |
|  | part | `parts/teaching_principles.qmd` |
| 5 | chapter | `05_data/05_data.qmd` |
| 6 | chapter | `06_training/06_training.qmd` |
| 7 | chapter | `07_evaluation/07_evaluation.qmd` |
|  | part | `parts/running_principles.qmd` |
| 8 | chapter | `08_perception/08_perception.qmd` |
| 9 | chapter | `09_memory/09_memory.qmd` |
| 10 | chapter | `10_intent/10_intent.qmd` |
| 11 | chapter | `11_planning/11_planning.qmd` |
| 12 | chapter | `12_enforcement/12_enforcement.qmd` |
| 13 | chapter | `13_placement/13_placement.qmd` |
|  | part | `parts/governing_principles.qmd` |
| 14 | chapter | `14_intervention/14_intervention.qmd` |
| 15 | chapter | `15_verification/15_verification.qmd` |
| 16 | chapter | `16_release/16_release.qmd` |
| 17 | chapter | `17_frontier/17_frontier.qmd` |
|  | backmatter | `backmatter/references.qmd` |
|  | backmatter | `backmatter/appendix_ml.qmd` |
|  | backmatter | `backmatter/appendix_control.qmd` |
|  | backmatter | `backmatter/appendix_systems.qmd` |
|  | backmatter | `backmatter/glossary/glossary.qmd` |
