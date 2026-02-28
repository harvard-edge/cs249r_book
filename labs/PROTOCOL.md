# Lab Developer Protocol: Gold Standard Specification

This is the authoritative instruction set for any agent or developer building an interactive lab for the *Machine Learning Systems* textbook. Labs are not demos. They are **pedagogical instruments**: every element — slider range, chart axis, prediction question, reflection prompt — must be traceable to a specific quantitative claim in the chapter.

A lab that cannot cite its chapter source for every number is not finished.

---

## Part I: What These Labs Are For

The textbook teaches students to reason quantitatively about ML infrastructure. The labs force students to *experience* the consequences of the invariants they just read about. The sequence is:

> **Read** the chapter → **Predict** what will happen → **Discover** that reality differs → **Explain** why using the chapter's math.

The prediction step is the most important. A student who predicts wrong and then discovers why has learned more than a student who reads a correct answer. Every lab must manufacture that productive failure.

Labs are **not**:
- Demos that illustrate concepts students already accept
- Tutorials that walk students through known steps
- Exploratory sandboxes with no expected destination

Labs **are**:
- Structured confrontations with a quantitative reality that surprises
- Diagnosis instruments that surface root causes students couldn't see in the text
- Design challenges where constraints collide and every choice has a cost

---

## Part II: The Invariants (Non-Negotiable Quality Gates)

Every lab plan and every lab implementation must satisfy **all** of the following. These are not suggestions.

### Invariant 1: Every Number Has a Source

Before writing a single line of the lab plan, the developer must extract the **actual quantitative claims** from the chapter text. Then every number in the lab must trace to a specific claim.

The plan must contain a **Traceability Table** (Section 8 of the plan template) that maps:

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| `[slider range / chart value / threshold]` | `[@sec-... or line number]` | `"[exact quote or formula from chapter]"` |

A plan without a complete traceability table is **rejected**.

### Invariant 2: Structured Predictions, Never Free Text

The `mo.stop(prediction == "", ...)` gate currently accepts "idk" and unlocks all instruments. This defeats the entire pedagogical purpose.

**Required prediction formats:**

| Format | When to Use | Implementation |
|---|---|---|
| **Multiple choice (4 options)** | When the answer is a specific ratio, threshold, or category | Radio buttons; exactly one correct answer; distractor options at 5×, 10×, 50× etc. |
| **Numeric range entry** | When the answer is a quantity the student can estimate (memory, latency, FLOPS) | Number input field; system records estimate for later comparison overlay |
| **Sentence completion with dropdown** | When the reflection requires using chapter terminology | Partial sentence with 3–4 dropdown options; only one is semantically correct |

**Never use:** open text fields for predictions, "type your hypothesis," or gates that accept any non-empty string.

After the act completes, the lab **must overlay the student's prediction** on the actual result with an annotation:
> "You predicted [X]. The actual value is [Y]. You were off by [Z]×."

This overlay is the learning moment. Do not skip it.

### Invariant 3: Failure States Are Mandatory

Every lab must have at least one instrument that can reach a visually distinct **failure state** when the student's design violates a physical constraint. Failure states teach constraints more effectively than captions.

Required failure states by constraint type:

| Constraint | Failure State Visual | Trigger Condition |
|---|---|---|
| Memory wall | Bar chart turns red; banner: **"OOM — Training infeasible on this device"** | `memory_footprint > device.ram` |
| Latency budget | Timeline bar turns red; banner: **"SLA violated — P99 exceeds budget"** | `latency_p99 > sla_budget` |
| Power/thermal | Gauge turns orange → red; banner: **"Thermal throttle — sustained throughput drops to [X]%"** | `power > tdp` |
| Compute budget | ROI gauge drops below zero; banner: **"Negative ROI — cost exceeds benefit"** | `cost > revenue_gain` |

The failure state must be **reversible**: students should be able to pull sliders back and watch the system recover. The point is to find the boundary, not to punish the student.

### Invariant 4: 2-Act Structure (Not 3 KATs)

Labs use a **2-Act structure**. The 3-KAT (three 15-minute Key Analysis Tasks) format produces 45–90 minute sessions, which students abandon mid-lab.

```
Act 1: Calibration (10–15 minutes)
  - One focused prediction question
  - One primary instrument with 1–2 controls
  - One structured reflection
  - Outcome: Student has a wrong prior corrected by data

Act 2: Design Challenge (18–25 minutes)
  - One numeric prediction
  - The full instrument set (2–3 charts, multiple controls)
  - One scaling challenge: push the system to its physical limit
  - Structured multi-choice reflection
  - Outcome: Student has made a design decision with quantified trade-offs
```

**Total target: 35–40 minutes.** If a lab plan requires more than 40 minutes to complete both acts, it must be trimmed.

### Invariant 5: 2 Deployment Contexts (Not 4 Narrative Tracks)

The original 4-track system (Cloud Titan, Edge Guardian, Mobile Nomad, Tiny Pioneer) requires 128 track-specific scenarios across 16 labs. This is unsustainable to maintain and does not provide proportional pedagogical value.

**Replacement:** Every lab uses **2 deployment contexts** as a comparison toggle, not as a persistent narrative identity.

The two contexts for Volume 1 are always drawn from:

| Context Pool | Device | Key Constraint |
|---|---|---|
| Training Node | H100 (80 GB) | Maximize throughput; memory is abundant |
| Edge Inference | Mobile GPU (2 GB) | Minimize latency; memory is the wall |
| MCU | ARM Cortex-M (256 KB SRAM) | Sub-1mW; only quantized INT8 models fit |

Each lab chooses the **two contexts most relevant to its chapter's invariant** and presents them as a comparison toggle. Students see the same system behave differently under different constraints. This is more instructive than narrative identity.

The Design Ledger carries the student's chapter-5 deployment context into chapter-8, so cross-lab continuity is preserved without maintaining 4 parallel track scripts.

### Invariant 6: No Instruments Before Chapter Introduction

Progressive disclosure is enforced strictly:

| Lab | Instruments Available |
|---|---|
| lab_01 | Magnitude Gap slider, D·A·M triangle |
| lab_02 | + Latency Waterfall |
| lab_05 | + Activation Comparator, Memory Ledger |
| lab_09 | + Pareto Curve |
| lab_10 | + Compression Trade-off Frontier |
| lab_11 | + Roofline Model |
| lab_13 | + P99 Latency Histogram, Little's Law Calculator |

Agents **must not** use an instrument in lab N if it is introduced in the chapter for lab N+k. Verify against the chapter text.

---

## Part III: The Plan Template (Required Structure)

Every lab plan must contain exactly these 8 sections in this order. A plan missing any section is not ready for implementation.

---

### Section 1: Chapter Alignment
```
- Chapter: [Title] (`@sec-[slug]`)
- Core Invariant: [One sentence. This is the chapter's central quantitative claim.]
- Central Tension: [Two sentences. What wrong prior does the student bring? What does the data reveal?]
- Target Duration: [X–Y minutes (2 acts)]
```

### Section 2: The Two-Act Structure Overview
One paragraph per act stating the pedagogical goal of that act in plain English. No bullet lists. Write it as a statement of what the student will experience and what they will learn.

### Section 3: Act 1 — Calibration
Required subsections:
1. **Pedagogical Goal** — one paragraph stating the wrong prior and what will correct it
2. **The Lock (Structured Prediction)** — the exact prediction question, with all answer choices listed, the correct answer marked, and the reason each distractor is plausible
3. **The Instrument** — describe every control (slider range, toggle options, selectors) and every output (chart type, axis labels, what updates on each control change)
4. **The Reveal** — the exact overlay text shown after interaction, including the prediction-vs-reality gap annotation
5. **Reflection (Structured)** — sentence completion with dropdown, or 4-option multiple choice. The exact text of the prompt and all options, with correct answer marked.
6. **Math Peek** — the LaTeX formula that governs this act (collapsible panel)

### Section 4: Act 2 — Design Challenge
Required subsections:
1. **Pedagogical Goal** — one paragraph
2. **The Lock (Numeric Prediction)** — exact question text, expected answer range, what the system will show afterward
3. **The Instrument** — complete control inventory with ranges, and complete output inventory with formulas
4. **The Scaling Challenge** — a specific target the student must hit by exploring the design space (e.g., "find maximum W where training fits on a Laptop GPU")
5. **The Failure State** — exact trigger condition, exact visual change, exact banner text
6. **Structured Reflection** — exact prompt and all options
7. **Math Peek** — the governing equation

### Section 5: Visual Layout Specification
List every chart in priority order. For each chart:
- Chart type (histogram, stacked bar, scatter, waterfall, etc.)
- X axis: label, range, units
- Y axis: label, range, units
- What data series are shown
- When/how it enters failure state (if applicable)

### Section 6: Deployment Context Definitions
Table with exactly 2 rows (the two chosen contexts):
| Context | Device | RAM | Power Budget | Key Constraint |
Explain in one sentence what distinguishes the two contexts for this chapter's invariant.

### Section 7: Design Ledger Output
The exact JSON fields the lab records at completion, and which future labs read those fields. If no future lab reads a field, it should not be recorded.

```json
{
  "chapter": N,
  "field_name": "<value>",
  ...
}
```

### Section 8: Traceability Table (Mandatory)
Every quantitative value in the lab must appear in this table.

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| [every slider range, threshold, formula, ratio] | [@sec-... or line number] | ["exact quote"] |

Rows without a chapter source are placeholder content and must be replaced with chapter-grounded values before the plan is complete.

---

## Part IV: Technical Implementation Specification

### 4.1 Format
- Single-file Marimo Notebook (`.py`)
- WASM-first: zero local file I/O; all data in native Python dicts
- Named `lab_NN_slug.py` with underscore-separated slug matching chapter slug

### 4.2 Physics Engine
All computations must call `mlsys.Engine.solve()`. No hardcoded physics constants outside `mlsys/constants.py`.

```python
from mlsys import Engine, Models, Systems

profile = Engine.solve(
    model=Models.ResNet50,
    system=Systems.Mobile,
    batch_size=32,
    precision="int8",     # "fp32" | "fp16" | "int8"
    efficiency=0.5        # float 0.0–1.0
)

# Guaranteed fields on the returned PerformanceProfile:
# .latency            (Pint ms)
# .latency_compute    (Pint ms)
# .latency_memory     (Pint ms)
# .latency_overhead   (Pint ms) — dispatch tax; always shown
# .throughput         (Pint samples/sec)
# .bottleneck         (str: "Compute" | "Memory")
# .energy             (Pint joule)
# .memory_footprint   (Pint byte)
# .feasible           (bool: memory_footprint <= system.ram)
```

### 4.3 Prediction Lock Implementation

The current `mo.stop(prediction == "", ...)` implementation is **not compliant**. The compliant prediction lock must:

1. For **multiple choice**: use a radio group; `mo.stop` fires until a radio option is selected (not just any text)
2. For **numeric entry**: use a number input; `mo.stop` fires until the value is non-null and within a plausible order-of-magnitude range
3. After act completion: **always** show the prediction-vs-reality overlay in a dedicated card above the reflection prompt

```python
# Compliant multiple-choice lock
prediction_choice = mo.ui.radio(
    options={"A) ~1-2×": "1x", "B) ~5×": "5x", "C) ~20×": "20x", "D) ~50×": "50x"},
    label="How much more expensive is Sigmoid than ReLU in transistors?"
)
mo.stop(prediction_choice.value is None, mo.md("⚠️ Select your prediction to unlock instruments."))

# Reveal overlay (shown after Act 1 completes)
actual = 50
predicted = {"1x": 1, "5x": 5, "20x": 20, "50x": 50}[prediction_choice.value]
gap = actual / predicted
mo.md(f"**You predicted {predicted}×. Actual: {actual}×. You were off by {gap:.1f}×.**")
```

### 4.4 Failure State Implementation

```python
# OOM failure state pattern
memory_total = weights + gradients + optimizer_state + activations
oom = memory_total > system.ram

if oom:
    # Chart bars turn red (update Plotly trace colors)
    bar_colors = ["#CB202D"] * 4  # RedLine
    banner = mo.callout(
        mo.md("🔴 **OOM — Training infeasible on this device.**  "
              f"Required: {memory_total:.1f} GB | Available: {system.ram:.1f} GB"),
        kind="danger"
    )
else:
    bar_colors = ["#006395", "#008F45", "#CC5500", "#4B0082"]  # BlueLine, GreenLine, OrangeLine, Purple
    banner = mo.md("")
```

### 4.5 Variable Naming
Marimo treats the notebook as a single dataflow graph. Variable names must be unique across all cells. Use cell-specific prefixes:

```python
# Act 1 variables
act1_prediction = mo.ui.radio(...)
act1_depth_slider = mo.ui.slider(...)
act1_fig = go.Figure(...)

# Act 2 variables
act2_prediction = mo.ui.number(...)
act2_batch_slider = mo.ui.slider(...)
act2_fig_memory = go.Figure(...)
```

### 4.6 Visual Identity
Import all components from `labs.core`. Use the canonical color palette:

| Token | Hex | Use |
|---|---|---|
| BlueLine | #006395 | Primary data, healthy state |
| GreenLine | #008F45 | Target/goal, success state |
| RedLine | #CB202D | Failure state, violation |
| OrangeLine | #CC5500 | Warning, secondary constraint |

Every chart must include a `MathPeek` toggle revealing the governing equation. Every latency waterfall must show the overhead/dispatch tax term.

---

## Part V: Developer Workflow

Every lab goes through exactly these steps in order. No step may be skipped.

### Step 1: Read the Chapter Completely
Read the full chapter text, not the learning objectives summary. Extract:
- Every quantitative claim with its value and units
- Every formula with variable definitions
- Every named invariant or law
- Every footnote with a numerical assertion

Do not write the plan until this extraction is complete.

### Step 2: Identify the Central Tension
Answer these two questions:
1. What does a typical student *believe* before reading this chapter?
2. What does the chapter's data reveal that contradicts that belief?

The answer to (2) minus the answer to (1) is the lab's pedagogical purpose. Every element of the lab must serve this gap.

### Step 3: Build the Traceability Table First
Before writing any other section, fill in the traceability table with the chapter's quantitative claims. If you can fill fewer than 4 rows, the chapter may not have enough quantitative content for a lab, or you have not read the chapter thoroughly enough. Re-read.

### Step 4: Design the Prediction Questions
For each act, write the prediction question and all answer options. The correct answer should be surprising — students who are overconfident about intuition should be wrong. The distractor options should map to common misconceptions (e.g., "about the same," "2× more expensive," "scales quadratically").

Test the question against the chapter: can a student who read the chapter carefully get the answer right? If yes, proceed. If the answer requires outside knowledge, revise.

### Step 5: Design the Instruments
For each instrument, specify:
- Every slider: min, max, step, default value, and the formula that maps slider position to output
- Every chart: x-axis, y-axis, all data series, the formula for each series
- Every threshold line: value, units, source in chapter

### Step 6: Design the Failure State
Identify the physical boundary the student will cross. Write the exact trigger condition as a Python boolean expression. Write the exact banner text. Test that the failure state is reachable within the instrument's slider ranges.

### Step 7: Write the Full Plan
Write all 8 sections of the plan template. Every number must be in the traceability table. Every prediction question must have all options listed. Every reflection must have all options listed.

### Step 8: Depth Check
Verify the plan meets minimum depth:
- ≥ 150 lines
- ≥ 4 rows in the traceability table
- All 8 sections present
- No placeholder text ("TBD", "see chapter", "varies")
- Every slider range is a specific number, not "appropriate range"
- Every act has a prediction question, instrument description, failure state (Act 2), and structured reflection

A plan that fails this check is not submitted for implementation.

---

## Part VI: The Design Ledger (Persistence Schema)

The Design Ledger carries student decisions across labs. It is a Python dict (not file I/O) persisted via Marimo's reactive state.

```python
# Schema — all fields are optional except chapter and timestamp
ledger = {
    "chapter": int,           # current chapter number
    "context": str,           # deployment context chosen: "training_node" | "edge_inference" | "mcu"
    "timestamp": str,         # ISO 8601

    # Chapter-specific fields (added by each lab, never overwritten)
    "ch05": {
        "activation_choice": str,           # "relu" | "sigmoid" | "tanh" | "gelu"
        "max_trainable_width_laptop_gpu": int,
        "training_memory_estimate_error_kb": float,
        "batch_size_chosen": int
    },
    # ch06, ch07, ch08, ... added by subsequent labs
}
```

**Rules:**
- Each lab adds exactly one `chNN` key. It never modifies prior chapter keys.
- Downstream labs READ prior chapter values to initialize their default slider positions.
- Example: lab_10 reads `ledger["ch05"]["activation_choice"]` to set the default activation in the compression comparison.
- If the ledger is empty (student starting mid-book), labs initialize from `mlsys.Systems.DefaultPreset`.

---

## Part VII: Validation Checklist

Before submitting a plan for implementation, verify every item:

**Content**
- [ ] Core Invariant is stated in one sentence with a specific quantitative claim
- [ ] Both prediction questions are structured (not free text)
- [ ] Each prediction question has exactly 4 options with one correct answer marked
- [ ] Correct answer is surprising (students who haven't read chapter likely get it wrong)
- [ ] Traceability table has ≥ 4 rows, all with chapter citations
- [ ] No placeholder values in slider ranges or chart axes

**Structure**
- [ ] All 8 plan sections present
- [ ] 2-Act structure (not 3 KATs)
- [ ] Target duration is 35–40 minutes
- [ ] 2 deployment contexts defined (not 4 narrative tracks)

**Instruments**
- [ ] Every slider has specific min, max, step, default values
- [ ] Every chart has labeled axes with units
- [ ] Act 2 has at least one failure state with trigger condition and banner text
- [ ] Prediction-vs-reality overlay is specified for both acts

**Design Ledger**
- [ ] Output fields are listed in Section 7
- [ ] At least one future lab reads a field from this lab's ledger output

**Implementation Readiness**
- [ ] Plan is ≥ 150 lines
- [ ] All numbers are chapter-grounded (no invented values)
- [ ] No instruments used that are introduced in a later chapter

---

## Appendix A: Lab Slug List

**Volume 1 (Foundations)**

| Lab | Slug | Chapter |
|---|---|---|
| 00 | lab_00_the_map | Architect's Portal (special case: no prediction lock, track orientation only) |
| 01 | lab_01_ml_intro | ML Introduction |
| 02 | lab_02_ml_systems | ML Systems |
| 03 | lab_03_ml_workflow | ML Workflow |
| 04 | lab_04_data_engr | Data Engineering |
| 05 | lab_05_nn_compute | Neural Computation |
| 06 | lab_06_nn_arch | NN Architectures |
| 07 | lab_07_ml_frameworks | ML Frameworks |
| 08 | lab_08_model_train | Training |
| 09 | lab_09_data_selection | Data Selection |
| 10 | lab_10_model_compress | Model Compression |
| 11 | lab_11_hw_accel | HW Acceleration |
| 12 | lab_12_perf_bench | Performance Benchmarking |
| 13 | lab_13_model_serving | Model Serving |
| 14 | lab_14_ml_ops | ML Operations |
| 15 | lab_15_responsible_engr | Responsible Engineering |
| 16 | lab_16_ml_conclusion | Conclusion (special case: synthesis across all invariants) |

**Volume 2 (Scale)** — same conventions apply; labs prefix `v2_`.

---

## Appendix B: Instrument Library

Available instruments from `labs.core`. Use only what is listed here. Do not invent new component names.

| Component | Import | Chapter First Available |
|---|---|---|
| `LatencyWaterfall` | `labs.core.components` | lab_02 |
| `MathPeek` | `labs.core.components` | lab_01 |
| `ComparisonRow` | `labs.core.components` | lab_01 |
| `MetricRow` | `labs.core.components` | lab_01 |
| `StakeholderMessage` | `labs.core.components` | lab_03 |
| `RooflineVisualizer` | `labs.core.components` | lab_11 |
| `PredictionLock` | `labs.core.components` | lab_01 (use compliant version from §4.3) |

---

*This document governs all lab development. When in doubt, ask: "Could Hennessy and Patterson point to the chapter line that justifies this slider range?" If not, it doesn't belong in the lab.*
