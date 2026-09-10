# StaffML Question Quality Rubric

**Status:** active guidance for pilot review and new question generation.

StaffML questions should test ML systems judgment, not vocabulary recall dressed
up with hardware names. A question is A-grade only if the scenario, explicit
ask, solution, common mistake, and napkin math all point at the same systems
competency.

## Core Standard

Every accepted question should satisfy five checks:

1. **Competency claim is clear.** The item has an intentional
   `topic × zone × track × level × phase` target.
2. **Scenario is realistic.** The setup resembles a decision, incident, design
   review, capacity plan, or debugging session an ML systems engineer could
   actually encounter.
3. **Answer teaches the principle.** The solution explains the binding
   constraint or tradeoff, not just the final number.
4. **Common mistake is plausible.** The mistake should reflect a real engineer's
   likely failure mode: peak-vs-effective bandwidth, forgotten KV cache, ignored
   queueing, missing factor of two, storage-vs-memory confusion, and so on.
5. **Question is reviewable.** The explicit `question` is one sentence,
   concrete, and exactly matches what the solution answers.

## Zone Rubrics

| Zone | A-grade evidence | Common weak version |
|---|---|---|
| `recall` | Names a concept because the scenario points to it. | Bare definition trivia. |
| `fluency` | Uses a remembered formula or constant to produce a concrete estimate. | Plug-and-chug with all formula pieces already spoon-fed. |
| `analyze` | Explains why a system behaves as observed. | Vague pros/cons list. |
| `diagnosis` | Identifies one binding cause from symptoms and rules out plausible distractors. | “It is slow because bottleneck” without mechanism. |
| `design` | Proposes an architecture that satisfies stated constraints. | Generic architecture with no constraint tie-in. |
| `specification` | Converts requirements into concrete system constraints or hardware needs. | Repeats the requirements in different words. |
| `evaluation` | Compares alternatives under tradeoffs and chooses with justification. | “Option A is better” without quantifying the trade. |
| `realization` | Turns a design into numbers: resources, budgets, windows, rates, limits. | Design prose without sizing. |
| `optimization` | Identifies the binding bottleneck and quantifies the improvement path. | Lists optimizations without explaining why they help. |
| `mastery` | Integrates multiple constraints, ambiguity, and missing information. | Long scenario whose answer is still a single direct formula. |

## Item Archetypes

Use archetypes to make the bank feel varied and product-grade:

- **Bottleneck diagnosis:** symptoms and counters imply a binding wall.
- **SLA sizing:** derive resources from latency, throughput, or reliability SLOs.
- **Tradeoff evaluation:** compare two or three plausible designs.
- **Incomplete information:** state what is missing, bound what can be known.
- **Incident postmortem:** diagnose a regression from traces or operational facts.
- **Design review:** critique a proposed architecture before launch.
- **Capacity planning:** convert demand into pods, GPUs, buffers, or bandwidth.
- **Visual reasoning:** use a diagram or timeline as evidence.
- **Rollout/rollback decision:** reason about canaries, failure domains, and risk.
- **Cost/carbon/power budget:** estimate economics or environmental impact.

## Expected Answer Shapes

Open-ended items should make the expected answer shape clear. Good answer
shapes include:

- final numeric estimate,
- binding constraint,
- assumptions,
- tradeoff comparison,
- recommendation,
- failure mode avoided,
- what additional information is needed.

For future metadata, use these as internal candidates:

- `item_archetype`
- `expected_answer_shape`
- `requires_math`
- `requires_architecture`
- `requires_debugging`
- `requires_missing_info`
- `has_visual`
- `interview_loop_slot`

Keep this metadata optional until review workflows stabilize.

## MCQ Distractor Standard

Distractors should encode real mistakes:

- forgetting key/value factor of two,
- bits vs bytes,
- peak vs effective bandwidth,
- compute-bound vs memory-bound confusion,
- ignoring queueing or tail latency,
- ignoring KV-cache growth,
- applying training assumptions to inference,
- using cloud networking assumptions on edge/TinyML.

Avoid distractors that are obviously silly or grammatically incompatible with
the stem.

## Visual Question Standard

A visual question must earn the visual:

- the `question` should refer to the diagram or figure when interpretation is
  required;
- `visual.alt` must be useful without the image;
- the solution should explain how to read the visual;
- the visual should test structure, ordering, timing, topology, memory layout,
  or flow, not merely decorate the text.

## Promotion Rule

Generated questions start as `status: draft`. Promote only after:

1. schema/gate validation passes,
2. math is checked where applicable,
3. topic/zone/level fit is reviewed,
4. the item meets the zone rubric above,
5. any visual asset renders in the StaffML UI.
