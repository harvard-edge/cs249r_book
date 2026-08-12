# Lab Experience And MLSysIM Readiness Plan

This document organizes the remaining work needed to make the MLSysBook labs effective, professional, and technically grounded. The raw material exists: book chapters, current Marimo labs, MLSysIM registries, solvers, constants, and physics models. The next step is to make the engine and lab experience fit together cleanly.

The consolidated implementation contract is in `LAB_SYSTEM_SPECIFICATION.md`. This document focuses on the student experience, visual direction, pedagogy, and MLSysIM readiness details behind that contract.

## The Core Requirement

The labs should teach ML systems engineering through trade-off exploration. A student should be encouraged to read the relevant chapter, but the lab cannot assume they already have. Each lab needs a compact chapter recap that names the idea the book emphasizes, translates it into a systems question, and tells the student what to focus on while exploring.

A student should open the lab, understand the relevant chapter idea, choose a deployment context, manipulate system knobs, observe what breaks, explain why, and record a design decision.

At the end of each lab, the student should be able to download a simple report and submit it as evidence of completion. This should stay lightweight: no LMS, no account system, and no complicated grading workflow. The report should show the track, scenario, predictions, binding constraints, final decision, rationale, residual risk, and MLSysIM result snapshot.

The lab should feel like part of the Machine Learning Systems book:

- Professional and academic.
- Clean, readable, and consistent.
- Grounded in the actual chapter content.
- Driven by MLSysIM rather than notebook-local formulas.
- Interactive enough to reveal trade-offs, but structured enough that students know what they are learning.

## What MLSysIM Already Provides

Current MLSysIM already has many of the necessary foundations:

- `Engine.solve()` for single-node performance and bottleneck analysis.
- `Engine.sweep()` for basic hardware/batch/precision exploration.
- `Scenarios` with executable scenarios.
- `Hardware`, `Models`, `Systems`, `Infrastructure`, `Datasets`, and `ReferenceStats` registries.
- Physics modules for memory, performance, networking, communication, reliability, serving, economics, statistics, and transformers.
- Solver modules for performance, training, serving, distributed systems, data, compression, orchestration, reliability, economics, sustainability, and responsible engineering.
- Typed result objects such as `PerformanceProfile`, `TrainingMemoryResult`, `ServingResult`, `TailLatencyResult`, `ReliabilityResult`, `EconomicsResult`, `DataResult`, `CompressionResult`, and `SensitivityResult`.

This means the work is not starting from scratch. The missing pieces are mostly consistency, composition, scenario coverage, and lab-ready result APIs.

## MLSysIM Capabilities To Add Or Tighten

| Capability | Why it is needed | Where it shows up |
|---|---|---|
| `ScenarioPack` or equivalent | Labs need track-specific defaults without putting curriculum logic in MLSysIM | All labs |
| `WorkloadProfile` distributions | Students need to change burstiness, stream count, sequence length, sensor rate, data rate, and job size | Serving, benchmarking, edge, orchestration, storage |
| `BottleneckReport` | Every Part A should clearly say what breaks first and why | All Part A activities |
| `TradeoffSweepResult` | Every Part B needs consistent sweep tables for plotting and filtering | All Part B activities |
| `ParetoFrontier` | Students should see dominated and non-dominated designs | Compression, serving, hardware, training, infrastructure |
| `ConstraintBudget` | Labs need a unified budget view across memory, latency, energy, cost, reliability, carbon, and privacy | All labs |
| `SensitivityReport` | Synthesis labs need to perturb prior decisions and show fragility | V1-16, V2 synthesis |
| `OperatorCostModel` | Neural computation labs need params, activations, MACs, bytes moved, and memory cliffs | V1-05, V1-11, V2-09 |
| `ArchitectureCostModel` | Architecture labs need comparable signatures for CNNs, transformers, SSMs, MoE, diffusion, etc. | V1-06, V2 distributed/inference |
| `RuntimeExecutionModel` | Framework labs need dispatch tax, kernel launches, fusion, compilation amortization, and delegate support | V1-07, V2-09 |
| `BenchmarkProtocolModel` | Benchmarking needs warm/cold, component/end-to-end, steady-state, tail, thermal, and energy behavior | V1-12 |
| `DataSelectionModel` | Data selection needs data quality, coverage, class balance, rare events, noise, and labeling cost | V1-09, responsible labs |
| `CommunicationModel` | Volume 2 needs first-class fabrics and collectives, not scattered formulas | V2 network, collectives, distributed training |
| `StoragePipelineModel` | Data/storage labs need sharding, cache, prefetch, checkpoint IO, and accelerator stalls | V1-04, V2-04 |
| `OpsModel` | MLOps labs need drift, monitors, retraining, canaries, alert fatigue, and platform ROI | V1-14, V2 ops |
| `ResponsibleSystemsModel` | Responsible labs need fairness, privacy, security, robustness, carbon, and governance as measurable constraints | V1-15, V2 responsible/safety/sustainability |
| Provenance on result fields | The book and labs both need traceability to constants, equations, and cited assumptions | Textbook callouts, lab math peek, instructor support |

## Lab Helper Components To Add

These should live in `mlsysbook_labs`, not MLSysIM.

The interaction-device details for these components are defined in `LAB_INTERACTION_DEVICE_CATALOG.md`.

| Component | Purpose | Notes |
|---|---|---|
| `lab_header()` | Clean academic header with volume, chapter, lab title, track, and book reading link | Replace dark gradient banners |
| `reading_anchor()` | Points to the relevant chapter and section-level ideas | Keeps students connected to the book |
| `chapter_recap()` | Compact recap of the chapter idea, key terms, and systems translation | Makes labs usable without assuming prior reading |
| `track_selector()` | Lets students choose Mobile ML, TinyML, Edge AI, or Cloud/Scale | Maps to MLSysIM scenarios |
| `scenario_brief()` | Introduces the engineering situation and stakeholder constraint | Should be concise and professional |
| `learning_question()` | Names the core systems question for the part | Avoids loose slider exploration |
| `prediction_card()` | Captures prior belief before revealing computed results | Structured choice or numeric prediction |
| `knob_panel()` | Groups the actual system controls | Keeps sliders from feeling random |
| `constraint_budget()` | Shows current memory, latency, energy, cost, reliability, carbon, or privacy budget | Standard visual across labs |
| `tradeoff_poster()` | Main visualization: frontier, heatmap, roofline, queue, timeline, or budget stack | The visual artifact students remember |
| `source_trace()` | Shows equation, constants, and MLSysIM provenance | Replaces notebook-local formula explanations |
| `reflection_prompt()` | Converts exploration into explanation | Mix structured fields with short rationale |
| `decision_card()` | Captures final engineering choice and residual risk | Saves to design ledger |
| `report_export()` | Builds and downloads a lab report from the ledger and MLSysIM result snapshot | Markdown first, JSON snapshot optional |
| `part_summary()` | States what changed and what principle was learned | Helps students consolidate |
| `design_ledger()` | Persists decisions across labs | Lab-layer state only |
| `advanced_knob_drawer()` | Holds optional controls for deeper exploration | Hidden by default and never required for completion |
| `instructor_metadata()` | Stores teaching goal, misconception, discussion hook, and rubric hints | Not shown in the student flow by default |
| `adoption_pack()` | Provides assignment text, report expectations, rubric, and instructor notes | Helps instructors adopt a lab with minimal prep |

## Visual Design Direction

The current dark gradient lab banner should be retired. It reads more like a dashboard/game than a professional academic lab, and it is inconsistent with the book identity.

Recommended visual system:

- Light background, white panels, restrained borders, high readability.
- Volume I accent: Harvard Crimson `#A51C30`.
- Volume II accent: ETH Blue `#1F407A`.
- Neutral text: slate/ink tones, not pure black.
- Status colors: muted green for feasible, amber for caution, red for violated constraints.
- No heavy dark hero banners.
- No decorative gradients as the main identity.
- Compact academic header with a colored volume stripe, chapter link, and track selector.
- Cards only for individual repeated items, prompts, decisions, or framed tools.
- Dense but readable layouts: left controls, center visual, right explanation/decision works well for many labs.

Recommended lab header structure:

1. Small metadata row: `Volume I | Chapter 5 | Neural Computation | Suggested reading`.
2. Lab title and one-sentence systems question.
3. Mini chapter recap: what the chapter emphasizes and what this lab will focus on.
4. Track selector and current scenario chip.
5. Three compact learning goals.
6. Current binding constraint preview after the active nugget begins.

Example tone:

> In this lab, you will test when neural network cost is dominated by arithmetic, weights, activations, or memory movement. The goal is not to make the model smaller in the abstract; the goal is to decide which resource is binding for your deployment track.

## Pedagogical Structure

Each lab should follow a stable pattern, but the unit of repetition should be a learning nugget, not necessarily the entire lab.

A learning nugget is one focused chapter idea that students explore through MLSysIM. A lab can have one, two, or three nuggets depending on the chapter. The student should quickly learn the rhythm: identify the idea, predict what will happen, move system knobs, inspect the constraint and trade-off, make a decision, and reflect.

This preserves the spirit of Part A, Part B, and Part C without forcing every chapter into exactly three top-level parts. A chapter with several important systems ideas can have several nuggets, and each nugget follows the same methodical flow.

### Lab-Level Wrapper

Every lab should open the same way:

1. Volume, chapter, and lab title.
2. Suggested chapter reading, stated as the chapter and key ideas rather than a fragile deep link.
3. Mini recap of the chapter emphasis.
4. ML-to-systems translation for the lab.
5. Track selector.
6. Scenario brief.
7. List of the nuggets students will explore.
8. Final synthesis question for the lab.

The chapter anchor should look more like:

> Suggested reading: Volume I, Chapter 5, Neural Computation. Key ideas: operator cost, activation memory, arithmetic intensity, and memory movement.

This keeps students connected to the book without overfitting the lab to specific section URLs.

### Chapter Recap Contract

The recap is not a replacement for the book chapter. It is an orientation layer that lets an independent student or a busy instructor use the lab without requiring a separate lecture first.

Every recap should include:

- Chapter emphasis: the core idea the book chapter wants students to understand.
- Key terms: three to five terms the lab will use.
- Systems translation: how the ML idea becomes a deployment, scale, cost, reliability, or operations question.
- What to watch: the binding resource or trade-off surface the student should pay attention to.
- Common trap: one misconception the lab is designed to correct.
- Suggested reading: the volume and chapter, with key ideas rather than fragile section links.

### Nugget-Level Cycle

Each nugget should follow this cycle:

1. Chapter idea or reading anchor.
2. Scenario setup.
3. Prediction.
4. Binding constraint.
5. Trade-off surface.
6. Engineering decision.
7. Reflection.
8. Ledger update.

For implementation, this can render as:

| Nugget stage | Student-facing purpose |
|---|---|
| Chapter idea | Names the specific book concept being exercised |
| System setup | Shows the track-specific scenario, workload, hardware, and constraints |
| Prediction | Captures the student's prior belief before computed results |
| Binding constraint | Shows what breaks first and why |
| Trade-off surface | Lets students move knobs and inspect the frontier or curve |
| Engineering decision | Requires a defensible configuration or policy choice |
| Reflection | Asks what changed, what got worse, and what assumption matters |
| Ledger update | Saves the decision, result snapshot, and residual risk |

This supports the desired student experience:

- Read the book chapter or section.
- Bring a prior belief into the lab.
- Test that belief against MLSysIM.
- See a trade-off, not a single answer.
- Explain what changed.
- Save a decision and residual risk.

### Recommended Lab Shape

| Lab element | Pedagogical role | Student output |
|---|---|---|
| Lab wrapper | Connects lab to book chapter and deployment track | Chapter key ideas, selected scenario, nugget list |
| Nugget | Explores one core chapter idea | Prediction, result, trade-off, decision, reflection |
| Synthesis | Connects nuggets into a chapter-level systems principle | Final design decision and reusable ledger record |

The labs can be sized for a 30 to 45 minute learning experience without displaying rigid minute estimates in every header. The better constraint is scope:

- One chapter.
- One selected track.
- One scenario family.
- One to three learning nuggets.
- One main trade-off visual per nugget.
- One final synthesis decision.

## Reflection Design

The current labs have predictions and sliders, but the next version should more deliberately ask students what the result means.

Each nugget should include a short reflection pattern:

| Prompt type | Example |
|---|---|
| Bottleneck explanation | What constraint became binding first, and what evidence shows that? |
| Trade-off interpretation | Which metric improved, and which metric got worse? |
| Counterfactual | What would change if this ran on your other track? |
| Design rationale | Why is your chosen configuration defensible? |
| Residual risk | What assumption would invalidate your decision? |
| Book connection | Which chapter principle explains the behavior you observed? |

Reflection should not become unstructured busywork. Use a mix of:

- Multiple choice for diagnosis.
- Numeric estimate for magnitude.
- One short rationale field for design reasoning.
- A final decision card with residual risk.

## Narrative Requirements For Each Lab

Each lab should explicitly state:

- The chapter reading anchor and key ideas.
- The system context.
- The nuggets students will explore.
- The core trade-off in each nugget.
- The knobs students will move in each nugget.
- The metrics they should watch in each nugget.
- The failure boundary they are expected to find in each nugget.
- The final decision they must make.

Each nugget should state:

- What we are trying to test.
- Which knobs matter.
- Which output is the key evidence.
- What misconception this part is designed to correct.
- How it connects back to the book.

## Additional Things Labs Should Capture

Beyond raw simulator outputs, labs should capture:

- Selected track.
- Scenario ID.
- Student prediction.
- Binding constraint.
- Chosen configuration.
- Pareto/frontier snapshot.
- Reflection answer.
- Residual risk.
- Book principle or chapter section referenced.
- Whether the student changed their mind after seeing the trade-off.

This should live in the design ledger managed by `mlsysbook_labs`. MLSysIM should provide the computed result snapshot; the lab layer should decide how to store student state.

## Implementation Sequence

1. Split lab UI/state from `mlsysim.labs` into `mlsysbook_labs`.
2. Add the lab-ready MLSysIM result schemas: `BottleneckReport`, `ConstraintBudget`, `TradeoffSweepResult`, `ParetoFrontier`, `SensitivityReport`.
3. Add reusable scenario packs for Mobile ML, TinyML, Edge AI, and Cloud/Scale.
4. Build the new light academic lab design system in `mlsysbook_labs`.
5. Add `lab_nugget()` and `nugget_reflection()` helpers so every chapter uses the same learning rhythm.
6. Replace dark header banners with `lab_header()`.
7. Pilot the new experience in V1-01, V1-05, and V2 Inference or Communication.
8. Add browser screenshots to verify the new visual style.
9. Convert the remaining labs chapter by chapter.
10. Add tests that prevent old dark banner patterns and notebook-local duplicate formulas from returning.

## Immediate Style Changes To Make In The Existing Labs

The current notebooks repeatedly use inline dark gradients such as:

```html
background: linear-gradient(135deg, #0f172a 0%, #1e293b ...)
```

Replace those with a shared helper:

```python
lab_header(
    volume="I",
    chapter="Neural Computation",
    lab="V1-05",
    title="The Activation Tax",
    question="When do activations, memory movement, and operator shape dominate neural network cost?",
    reading_href="../book/...",
)
```

That helper should render a light, professional header using the correct volume accent. This one change will remove most of the visual inconsistency across the labs.
