# Chapter 13: Model Serving - Developmental Edit Report

## 1. Structural Flow & Progressive Disclosure (The Primary Bottleneck)
The chapter front-loads advanced queuing theory and OS-level trivia before the student has a mental model of what a serving pipeline actually is.
- **The Bottleneck**: In `## Serving Load, Latency, and Architecture`, the text immediately jumps into $M/M/1$ queues, Little's Law, and Black Friday traffic spikes. It then veers into static vs. dynamic inference, the spectrum of architectures (TinyML vs Cloud), briefly mentions load balancers, and then plunges into OS-level CPU pinning and NUMA boundaries. *Only after all this* does it explain the Inference Server anatomy.
- **Actionable Fix**: Introduce `## Serving System Architecture` (Triton/TF Serving) and the `## Request Lifecycle` *first*. Students need to understand the physical flow of a request before they can understand how that flow degrades under load. Move the "Black Friday" traffic spike and Tail Latency explosion discussions to `## Queuing Theory`.

## 2. Redundancy & Encyclopedic Bloat
Several concepts are duplicated, causing bloat and breaking logical flow:
- **NUMA and Thread Pinning**: The early section `### Deterministic latency and resource isolation` goes deep into CPU pinning, NUMA, `mlock`, and IRQ shielding. These exact concepts are repeated later in `## Node-Level Optimization` > `### CPU inference optimization`.
  - *Fix*: Delete the early section entirely. Move unique OS-level details (`mlock`, IRQ shielding) into `Node-Level Optimization` where they contextually belong.
- **Serialization Overhead**: The chapter thoroughly explains JSON vs. Protobuf/Flatbuffers in `### Interface protocols and serialization`, but then re-litigates the microsecond costs of deserialization a few pages later in `#### The serving tax bill`.
  - *Fix*: Consolidate the "serving tax" microsecond calculations directly into the serialization section.
- **The Load Balancer Layer**: This H3 section is essentially a placeholder that points the student to another chapter (`@sec-ml-operations`).
  - *Fix*: Remove the dedicated header. Briefly mention load balancing in the networked serving introduction, leaving the full treatment for the MLOps chapter.

## 3. Orphaned Visuals & Misplaced Figures
- **Orphaned SVG**: `images/svg/model_serving_blast_radius.svg` is placed in a margin callout under `### Deterministic latency and resource isolation` with a caption but lacks a formal `@fig-` label and is never referenced in the main text.
  - *Fix*: Give it a proper figure reference and tie it directly to the text (or move it to `Node-Level Optimization` along with the text).
- **Misplaced Economics**: `@fig-intelligence-deflation` (API token cost collapse) and the "Facebook Inference Tax" callout are placed in the introductory `## Serving Paradigm` section. This is a fascinating tangent, but it derails the technical introduction.
  - *Fix*: Relocate these economic elements to `## Economics and Planning` at the end of the chapter to create a cohesive unit on serving costs.
